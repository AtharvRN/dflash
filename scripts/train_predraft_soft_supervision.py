"""Pre-draft soft-supervision ablation on frozen B16 replay candidates.

Only `fused` enters the model. Probability labels are training-only privileged
information. Evaluation clips unchanged B16 candidates, NOT actual shorter
drafts. Soft profile products are ranking surrogates, not greedy probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import calibrate, apply_setting, metrics
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.collect_midverify_probe import backup_file
from scripts.gpu_runtime import configure_gpu_runtime
from scripts.train_midverify_scaling import batch_stream, parameter_sha
from scripts.train_soft_supervision import load_cache, bootstrap, descriptive_curve, TARGETS

ARMS = ("hard", "soft_tv", "soft_target", "hard_tv", "hard_target")
BCE_ARMS = ("hard", "soft_tv_bce", "mixed_tv_bce", "mixed_margin_bce")
SOFT_WEIGHT = .25
MARGIN_TEMPERATURE = 1.0
LIMITATIONS = [
    "Pre-draft inputs only; B16 candidate-preserving clipping, NOT actual smaller-block drafting",
    "No online throughput or draft-work saving established by this experiment",
    "TV overlap is stochastic distribution agreement; target mass is not greedy correctness probability",
    "Products of predicted soft scores are policy ranking surrogates, not calibrated greedy survival",
    "All training losses use at-risk positions including first rejection; off-path suffix excluded",
    "Calibration retention is not an assessment guarantee; compare curves, not unmatched-retention rankings",
    "Small pilot and previously inspected development assessment; not an untouched final test",
]


def predraft_inputs(arrays):
    """Explicit inference feature allowlist. Never concatenate candidate/label arrays."""
    x = np.asarray(arrays["fused"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2560 or not np.isfinite(x).all():
        raise ValueError("Expected finite pre-draft fused vectors of width 2560")
    return x


def build_model(width=2560):
    from torch import nn
    # Same inexpensive backbone as the previous fused MLP. All arms instantiate
    # all three heads to match parameter counts, RNG draws and initialization.
    return nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(.05),
                         nn.Linear(512, 256), nn.GELU(), nn.Dropout(.05),
                         nn.Linear(256, 128), nn.GELU(), nn.Dropout(.05),
                         nn.Linear(128, 45), nn.Unflatten(-1, (3, 15)))


def objective(output, hard, tv, target, mask, arm):
    import torch.nn.functional as F
    if arm not in ARMS:
        raise ValueError("Unknown supervision arm")
    # Equal loss reduction across arms. SmoothL1 beta=.1 fixed before training.
    # No invented failures beyond the observed cap, nor off-path supervision.
    mean = lambda value: (value * mask).sum() / mask.sum().clamp_min(1)
    if arm == "hard" or arm.startswith("hard_"):
        loss = mean(F.binary_cross_entropy_with_logits(output[:, 0], hard, reduction="none"))
    else:
        loss = output.new_zeros(())
    if arm.endswith("tv"):
        loss = loss + mean(F.smooth_l1_loss(output[:, 1].sigmoid(), tv, beta=.1, reduction="none"))
    elif arm.endswith("target"):
        loss = loss + mean(F.smooth_l1_loss(output[:, 2].sigmoid(), target, beta=.1, reduction="none"))
    return loss


def same_head_targets(hard, tv, margin, arm):
    """Fixed labels for the SAME primary output; no assessment-fitted parameters."""
    if arm not in BCE_ARMS:
        raise ValueError("Unknown same-head arm")
    if arm == "hard":
        return hard
    if arm == "soft_tv_bce":
        return tv.clamp(0, 1)  # FP32 summation can exceed one by tiny roundoff.
    soft = tv.clamp(0, 1) if arm == "mixed_tv_bce" else (margin / MARGIN_TEMPERATURE).sigmoid()
    # At exact argmax ties, retain the actual hard tie-breaking outcome in the
    # hard component; the margin soft component is neutral (0.5).
    return (1-SOFT_WEIGHT)*hard + SOFT_WEIGHT*soft


def same_head_objective(output, hard, tv, margin, mask, arm):
    import torch.nn.functional as F
    labels = same_head_targets(hard, tv, margin, arm)
    values = F.binary_cross_entropy_with_logits(output[:, 0], labels, reduction="none")
    return (values*mask).sum()/mask.sum().clamp_min(1)


def policy_scores(output, arm):
    import torch.nn.functional as F
    if arm not in ARMS + BCE_ARMS:
        raise ValueError("Unknown supervision arm")
    head = 1 if arm == "soft_tv" else 2 if arm == "soft_target" else 0
    # Log-space prefix products: stable, monotonically decreasing by position.
    return F.logsigmoid(output[:, head]).cumsum(-1)


def prefix_calibration(scores, accepted):
    """Descriptive reliability against real greedy prefix outcomes, all 15 slots.

    Once a rejection occurs, prefix-survival labels are known zero even though
    subsequent token-match labels are off-path. No fitting or policy selection.
    """
    values = np.asarray(scores, dtype=np.float64)
    a = np.asarray(accepted)
    if values.shape != (len(a), 15) or not np.isfinite(values).all() or np.any(values > 1e-6):
        raise ValueError("Invalid log-prefix scores")
    if np.any((a < 0) | (a > 15) | (a != a.astype(int))):
        raise ValueError("Invalid accepted lengths")
    probability = np.exp(values).clip(0, 1)
    truth = a[:, None] >= np.arange(1, 16)[None]
    per_position = []
    for j in range(15):
        bins = np.minimum((probability[:, j]*10).astype(int), 9)
        reliability = []
        ece = 0.
        for b in range(10):
            take = bins == b
            if not take.any():
                continue
            confidence, frequency = float(probability[take, j].mean()), float(truth[take, j].mean())
            ece += float(take.mean())*abs(confidence-frequency)
            reliability.append({"bin": b, "count": int(take.sum()), "predicted": confidence, "observed": frequency})
        per_position.append({"position": j+1, "brier": float(np.square(probability[:, j]-truth[:, j]).mean()),
                             "ece_10_equal_width_bins": ece, "reliability": reliability})
    return {"mean_brier": float(np.square(probability-truth).mean()),
            "mean_position_ece": float(np.mean([v["ece_10_equal_width_bins"] for v in per_position])),
            "surrogate_expected_A_mae": float(np.abs(probability.sum(1)-a).mean()),
            "positions": per_position,
            "scope": "diagnostic only; soft/mixed prefix products are not assumed to be greedy probabilities"}


def score(model, x, arm):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([policy_scores(model(x[i:i+256]), arm).cpu().numpy()
                               for i in range(0, len(x), 256)])


def label_diagnostics(model, x, arrays, indices):
    """Descriptive only; never participates in checkpoint or threshold selection."""
    import torch
    model.eval()
    with torch.inference_mode():
        pred = np.concatenate([model(x[i:i+256]).sigmoid().cpu().numpy()
                               for i in range(0, len(x), 256)])
    mask = arrays["at_risk"][indices].astype(bool)
    return {name: {"at_risk_mse": float(np.square(pred[:, j]-arrays[field][indices])[mask].mean()),
                   "at_risk_positions": int(mask.sum())}
            for j, (name, field) in enumerate((("hard", "matches"), ("tv", "tv_overlap"),
                                              ("target", "target_candidate_prob")))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "output", "backup"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--suite", choices=("auxiliary", "same_head_bce"), default="auxiliary")
    args = p.parse_args()
    same_head = args.suite == "same_head_bce"
    arms = BCE_ARMS if same_head else ARMS
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Fresh independent destinations required")
    collection, rows, arrays = load_cache(args.cache, view="predraft")
    if bool(collection["smoke"]) != args.smoke:
        raise ValueError("Smoke/full cache mismatch")
    group = np.array([r["group"] for r in rows])
    select = {g: np.flatnonzero((group == g) & arrays["eligible"])
              for g in ("train", "calibration", "assessment")}
    tr, ca, ass = (select[g] for g in ("train", "calibration", "assessment"))
    if min(map(len, select.values())) == 0:
        raise ValueError("Empty eligible partition")
    features = predraft_inputs(arrays)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    runtime = configure_gpu_runtime(use_container_gpu=True, require_gpu=True)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    x = torch.as_tensor(features, device=device)
    tx, cx, ax = x[tr], x[ca], x[ass]
    y, tv, target, risk = (torch.as_tensor(arrays[k][tr], dtype=torch.float32, device=device)
                           for k in ("matches", "tv_overlap", "target_candidate_prob", "at_risk"))
    margin = torch.as_tensor(arrays["target_margin"][tr], dtype=torch.float32, device=device)
    accepted = arrays["accepted_len"]
    prompts = np.array([r["prompt_id"] for r in rows])[ass]
    seeds, updates, every = ([913], 8, 4) if args.smoke else ([913, 914, 915], 1024, 64)
    evaluation_steps = sorted(set(range(every, updates+1, every)) |
                              ({8, 16, 32, 48} if same_head and not args.smoke else set()))
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)

    def save_json(name, data):
        atomic_json(args.output/name, data)
        backup_file(args.output/name, args.backup/name)

    config = {"schema": "predraft_soft_supervision_v1", "cache_complete_sha256": sha256(args.cache/"COMPLETE.json"),
              "runtime": runtime, "seeds": seeds, "updates": updates, "calibrate_every": every,
              "batch_size": 128, "arms": arms, "suite": args.suite, "evaluation_steps": evaluation_steps,
              "input_allowlist": ["fused"],
              "inputs": "2560-d latest committed-token fused vector captured BEFORE current draft; no extra normalization",
              "architecture": "2560-512-256-128-45; GELU/LayerNorm/dropout.05; three 15-position heads in EVERY arm",
              "loss": "masked hard BCE OR soft SmoothL1(sigmoid, label; beta=.1) OR sum of hard and soft losses",
              "soft_label_weight": 1.0, "mask": "at-risk including first rejection; excludes off-path suffix",
              "optimizer": "AdamW3e-4, wd.01, clip1; FP32; TF32 off",
              "checkpoint_selection": "minimum calibration mean budget at >=96% retention; tie higher retention, then earlier update",
              "decision": "threshold log-prefix-product; hard/joint uses hard head, soft-only uses corresponding soft head; 0..15 proposals",
              "counts": {g: {"planned": int((group == g).sum()), "eligible": len(v),
                              "prompts": len({rows[i]["prompt_id"] for i in v})} for g, v in select.items()},
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": sha256(Path(__file__)), "smoke": args.smoke,
              "limitations": LIMITATIONS}
    if same_head:
        config.update(schema="predraft_same_head_bce_v1",
                      loss="at-risk BCE on primary output only; labels hard, TV, .75*hard+.25*TV, or .75*hard+.25*sigmoid(margin/1.0)",
                      soft_label_weight=SOFT_WEIGHT, margin_temperature=MARGIN_TEMPERATURE,
                      decision="threshold log-prefix-product of the SAME primary head in every arm; 0..15 proposals",
                      unused_outputs="two auxiliary output slots retained in every model to preserve initialization against prior baseline; receive no loss",
                      hyperparameters_selected_on_assessment=False,
                      prefix_calibration="diagnostic Brier/ECE against actual greedy prefix outcomes; no extra fitted calibration")
    save_json("config.json", config)
    save_json("row_selection.json", {g: v.tolist() for g, v in select.items()})
    summaries, choices, predictions, curves = {}, {}, {}, {}
    calibration_predictions = {}
    started = time.monotonic()
    for seed in seeds:
        initial_reference, order_reference = None, None
        for arm in arms:
            torch.manual_seed(seed)
            model = build_model().to(device)
            initial = parameter_sha(model.state_dict())
            if initial_reference is None:
                initial_reference = initial
            if initial != initial_reference:
                raise ValueError("Unmatched initialization")
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            stream = batch_stream(len(tr), 128, seed)
            order_hash = hashlib.sha256()
            history, best, best_key = [], None, None
            for step in range(1, updates+1):
                b = next(stream)
                order_hash.update(b.numpy().tobytes())
                b = b.to(device)
                model.train()
                opt.zero_grad(set_to_none=True)
                loss = (same_head_objective(model(tx[b]), y[b], tv[b], margin[b], risk[b], arm) if same_head else
                        objective(model(tx[b]), y[b], tv[b], target[b], risk[b], arm))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                if step not in evaluation_steps:
                    continue
                qc = score(model, cx, arm)
                point = calibrate(qc, accepted[ca], (.96,))["0.96"]
                key = (point["mean_kept_rows"], -point["retention"], step)
                if best_key is None or key < best_key:
                    best_key = key
                    best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, step, qc.copy())
                record = {"update": step, "loss": float(loss.detach()), "calibration": point}
                history.append(record)
                print("UPDATE", arm, seed, json.dumps(record), flush=True)
            if order_reference is None:
                order_reference = order_hash.hexdigest()
            if order_reference != order_hash.hexdigest():
                raise ValueError("Unmatched minibatch order")
            weights, step, qc = best
            name = f"{arm}_seed{seed}"
            path = args.output/(name+".pt")
            torch.save({"model": weights, "seed": seed, "arm": arm, "selected_update": step, "config": config}, path)
            backup_file(path, args.backup/path.name)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
            if not np.array_equal(score(model, cx, arm), qc):
                raise ValueError("Checkpoint reload changed calibration scores")
            settings = calibrate(qc, accepted[ca], TARGETS)
            qa = score(model, ax, arm)
            points = {t: {"calibration": s, "assessment": metrics(accepted[ass], apply_setting(qa, s))}
                      for t, s in settings.items()}
            k = apply_setting(qa, settings["0.96"])
            choices[name] = k
            summaries[name] = {"selected_update": step, "initial_parameter_sha256": initial,
                "minibatch_order_sha256": order_hash.hexdigest(), "parameter_count": sum(v.numel() for v in model.parameters()),
                "primary": points["0.96"], "operating_points": points,
                "bootstrap": bootstrap(accepted[ass], k, choices[f"hard_seed{seed}"], prompts, 100 if args.smoke else 2000),
                "label_diagnostics": label_diagnostics(model, ax, arrays, ass) if not same_head else None,
                "trained_heads": (["primary"] if same_head else ["hard"] if arm == "hard" else [arm.removeprefix("soft_")] if arm.startswith("soft_")
                                  else ["hard", arm.removeprefix("hard_")])}
            if same_head:
                summaries[name]["prefix_calibration"] = {"calibration": prefix_calibration(qc, accepted[ca]),
                                                         "assessment": prefix_calibration(qa, accepted[ass])}
                summaries[name]["assessment_matched_retention_DESCRIPTIVE"] = calibrate(qa, accepted[ass], (.96,))["0.96"]
            predictions[name], calibration_predictions[name] = qa, qc
            curves[name] = descriptive_curve(qa, accepted[ass])
            save_json(name+"_history.json", history)
            save_json("progress.json", {"completed_models": list(summaries), "last": summaries[name],
                                        "elapsed_seconds": time.monotonic()-started})
    # Constant, training-estimated conditional acceptance profile: no state signal.
    train_q = (arrays["matches"][tr]*arrays["at_risk"][tr]).sum(0) / np.maximum(arrays["at_risk"][tr].sum(0), 1)
    constant = np.broadcast_to(np.log(np.clip(train_q, 1e-7, 1)).cumsum(), (len(rows), 15))
    controls = {}
    for name, scores in (("constant_predraft", constant), ("raw_confidence_POSTDRAFT_reference_only", arrays["draft_stats"][:, :, 0])):
        settings = calibrate(scores[ca], accepted[ca], TARGETS)
        controls[name] = {t: {"calibration": s, "assessment": metrics(accepted[ass], apply_setting(scores[ass], s))}
                          for t, s in settings.items()}
    summary = {"models": summaries, "controls": controls, "counts": config["counts"], "across_seeds": {},
               "full_b16": metrics(accepted[ass], np.full(len(ass), 16)),
               "elapsed_seconds": time.monotonic()-started, "limitations": LIMITATIONS}
    for arm in arms:
        summary["across_seeds"][arm] = {}
        for field in ("aggregate_accept_ratio", "retention", "mean_kept_rows", "mean_accepted"):
            values = [summaries[f"{arm}_seed{s}"]["primary"]["assessment"][field] for s in seeds]
            summary["across_seeds"][arm][field] = {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    save_json("summary.json", summary)
    save_json("assessment_descriptive_curves.json", curves)
    for name, preds, indices in (("assessment", predictions, ass), ("calibration", calibration_predictions, ca)):
        np.savez(args.output/(name+"_predictions.npz"), **preds, accepted=accepted[indices],
                 row_indices=indices, prompt_ids=np.array([rows[i]["prompt_id"] for i in indices]))
        backup_file(args.output/(name+"_predictions.npz"), args.backup/(name+"_predictions.npz"))
    files = sorted(p.name for p in args.output.iterdir() if p.is_file())
    save_json("COMPLETE.json", {"complete": True, "binding": {name: sha256(args.output/name) for name in files}})
    print("COMPLETE", json.dumps({"across_seeds": summary["across_seeds"], "counts": config["counts"]}), flush=True)


if __name__ == "__main__":
    main()
