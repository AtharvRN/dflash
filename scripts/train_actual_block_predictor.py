"""Matched actual-block versus clipped-B16 supervision; no assessment tuning."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256, validate_states
from scripts.collect_policy_granularity import CHECKPOINT_SHA, atomic_json
from scripts.analyze_policy_granularity import load_pairs, select_operating_points, budgets_from_survival
from dflash.block_response import (BlockResponseMLP, TARGETS, training_targets,
    choose_response_budget, response_metrics, calibration_curve, select_response_points)


def load_training_pairs(root, split_dir, evaluation_config):
    """Verify every receipt, then select the first N eligible rows in saved order."""
    config = json.loads((root/"config.json").read_text())
    complete = json.loads((root/"COMPLETE.json").read_text())
    summary = json.loads((root/"collection_summary.json").read_text())
    target_rows = config.get("training_rows", 0)
    if config.get("collection_kind") != "training" or target_rows < 1 or not complete["sample_complete"]:
        raise ValueError("Require complete training-only collection")
    if set(complete["binding"]) != {"config.json", "receipts.json", "collection_summary.json"}:
        raise ValueError("Incomplete completion binding")
    for name, digest in complete["binding"].items():
        if sha256(root/name) != digest:
            raise ValueError("Training completion hash mismatch")
    keys = ("models", "blocks", "dtype", "attention", "tf32", "temperature", "thinking",
            "states_per_prompt", "max_new_tokens", "max_prompt_tokens", "feature")
    if any(config[k] != evaluation_config[k] for k in keys):
        raise ValueError("Training/evaluation collection configuration mismatch")
    train = set(map(int, json.loads((split_dir/"train_prompt_ids.json").read_text())["train_prompt_ids"]))
    val = set(map(int, json.loads((split_dir/"val_prompt_ids.json").read_text())["val_prompt_ids"]))
    if train & val or not set(map(int, evaluation_config["prompt_ids"])) <= val:
        raise ValueError("Canonical split/evaluation mismatch")
    for name in ("train_prompt_ids.json", "val_prompt_ids.json"):
        if config["input_hashes"][str(split_dir/name)] != sha256(split_dir/name):
            raise ValueError("Canonical split bytes changed")
    eval_hashes = set(evaluation_config["prompt_content_hashes"].values())
    if eval_hashes & set(config["prompt_content_hashes"].values()):
        raise ValueError("Cross-split duplicate prompt content")
    receipts = json.loads((root/"receipts.json").read_text())
    received = [r["prompt_id"] for r in receipts]
    if received != config["prompt_ids"][:len(receipts)] or len(set(received)) != len(received):
        raise ValueError("Training receipt ordering/identity mismatch")
    rows, chunks = [], []
    for receipt in receipts:
        pid = receipt["prompt_id"]
        if pid not in train or pid in val or receipt["group"] != "train" or config["prompt_groups"][str(pid)] != "train":
            raise ValueError("Nontraining prompt in training cache")
        if json.loads((root/f"receipt_{pid}.json").read_text()) != receipt:
            raise ValueError("Training receipt mismatch")
        expected = {f"prompt_{pid}.json"} | ({f"prompt_{pid}_fused.npy"} if receipt["states"] else set())
        if set(receipt["files"]) != expected:
            raise ValueError("Missing/unexpected receipt files")
        for name, digest in receipt["files"].items():
            if sha256(root/name) != digest:
                raise ValueError("Training shard hash mismatch")
        states = json.loads((root/f"prompt_{pid}.json").read_text())["states"]
        validate_states(states, config["blocks"])
        if len(states) != receipt["states"] or any(int(s["prompt_id"]) != pid or s["group"] != "train" or s["source"] != receipt["source"] for s in states):
            raise ValueError("Training row/feature identity mismatch")
        if not states:
            continue
        features = np.load(root/f"prompt_{pid}_fused.npy", allow_pickle=False)
        if features.shape != (len(states), 2560) or features.dtype != np.float16 or not np.isfinite(features).all():
            raise ValueError("Invalid training features")
        rows.extend(states)
        chunks.append(features)
    eligible = np.array([r["eligible"] for r in rows], dtype=bool)
    if (len(rows) != complete["states"] or len(rows) != summary["states"] or
            int(eligible.sum()) != summary["eligible_states"] or int(eligible.sum()) != complete["eligible_states"] or
            len(receipts) != summary["prompts"] or not summary["sample_complete"] or eligible.sum() < target_rows):
        raise ValueError("Incomplete/misaligned training counts")
    if summary["canonical_disagreements"] or sum(r["canonical_disagreements"] for r in rows):
        raise ValueError("Investigate canonical numerical disagreements before training")
    take = np.flatnonzero(eligible)[:target_rows]
    selected = [rows[i] for i in take]
    features = np.concatenate(chunks)[take]
    actual = np.array([[r["outcomes"][str(b)]["accepted"] for b in range(2, 17)] for r in selected], dtype=np.int64)
    training_targets(actual, "actual")
    audit = {"passed": True, "training_rows": len(selected), "training_prompts": len({r["prompt_id"] for r in selected}),
        "collected_rows": len(rows), "eligible_collected": int(eligible.sum()), "unused_eligible_tail": int(eligible.sum())-len(selected),
        "selection": "first requested number of eligible states in immutable receipt/within-prompt order",
        "source_rows": dict(Counter(r["source"] for r in selected)),
        "training_completion_sha256": sha256(root/"COMPLETE.json"),
        "evaluation_configuration_sha256": hashlib.sha256(json.dumps(evaluation_config, sort_keys=True).encode()).hexdigest(),
        "canonical_split_disjoint": True, "receipt_and_prefix_hashes_verified": True,
        "canonical_disagreements": 0, "feature_label_alignment": "same collection/state, no old/new joins",
        "label_disagreement_by_block": (actual != training_targets(actual, "clipped")).mean(0).tolist()}
    index = [{k: r[k] for k in ("prompt_id", "cycle", "prefix_sha256", "source")} for r in selected]
    return features, actual, audit, index


def weight_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def predict_cpu(model, features):
    import torch
    model = model.cpu().eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            values.append(model(torch.from_numpy(features[start:start+256].astype(np.float32))).numpy())
    return np.concatenate(values)


def bootstrap(actual, prompt, choices, count=2000, seed=929):
    groups, inverse = np.unique(prompt, return_inverse=True)
    draws = np.random.default_rng(seed).integers(len(groups), size=(count, len(groups)))
    sampled, intervals = {}, {}
    quantile = lambda x: np.quantile(x, [.025, .975]).tolist()
    for name, budget in choices.items():
        values = (actual[np.arange(len(actual)), budget-1], budget, actual[:, -1], np.ones(len(actual)))
        totals = np.column_stack([np.bincount(inverse, weights=v, minlength=len(groups)) for v in values])
        q = totals[draws].sum(1)
        sampled[name] = q
        intervals[name] = {"ratio_ci95": quantile(q[:, 0]/q[:, 1]), "retention_ci95": quantile(q[:, 0]/q[:, 2]),
                           "mean_budget_ci95": quantile(q[:, 1]/q[:, 3])}
    paired = {}
    for name, a in sampled.items():
        if not name.startswith("actual_seed_"):
            continue
        control = name.replace("actual_", "clipped_", 1)
        c = sampled[control]
        paired[name+"_minus_"+control] = {"ratio_ci95": quantile(a[:, 0]/a[:, 1]-c[:, 0]/c[:, 1]),
            "retention_ci95": quantile((a[:, 0]-c[:, 0])/a[:, 2]),
            "relative_budget_saving_ci95": quantile(1-a[:, 1]/c[:, 1])}
    return {"resamples": count, "prompts": len(groups), "policies": intervals, "paired_supervision_differences": paired,
            "scope": "paired whole-prompt resampling; checkpoint and calibration selection held fixed"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("train-cache", "eval-cache", "split-dir", "checkpoint", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--gpu", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=[913, 914, 915])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Invalid training limits or existing output")
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Wrong frozen baseline checkpoint")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu is None else str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from dflash.policy import HorizonPredictorRuntime
    from dflash.context_attention import acceptance_survival
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.gpu is not None:
        used = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(used.strip()) > 1024:
            raise RuntimeError("GPU is occupied; no training launched")
    eval_config, eval_rows, eval_features, _ = load_pairs(args.eval_cache)
    if eval_config.get("collection_kind", "evaluation") != "evaluation" or any(r["canonical_disagreements"] for r in eval_rows):
        raise ValueError("Invalid evaluation collection")
    eval_binding = sha256(args.eval_cache/"COMPLETE.json")
    x, actual_train, audit, index = load_training_pairs(args.train_cache, args.split_dir, eval_config)
    if not args.smoke and (len(x) != 10000 or args.epochs != 6 or args.seeds != [913, 914, 915] or args.batch_size != 128):
        raise ValueError("Non-smoke run must match the fixed 10k/three-seed/six-epoch pilot protocol")
    group = np.array([r["group"] for r in eval_rows])
    eligible = np.array([r["eligible"] for r in eval_rows], dtype=bool)
    cal, assess = eligible & (group == "calibration"), eligible & (group == "assessment")
    if not cal.any() or not assess.any():
        raise ValueError("Missing calibration/assessment group")
    actual_eval = np.array([[r["outcomes"][str(b)]["accepted"] for b in range(2, 17)] for r in eval_rows], dtype=np.int64)
    prompt_eval = np.array([int(r["prompt_id"]) for r in eval_rows])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_cal = set(map(int, checkpoint["model_config"]["cache_manifest"]["calibration_prompt_ids"]))
    if checkpoint["architecture"] != "last_mlp" or checkpoint["epoch"] != 4:
        raise ValueError("Frozen checkpoint architecture/epoch mismatch")
    if any((int(pid) in original_cal) != (g == "calibration") for pid, g in eval_config["prompt_groups"].items()):
        raise ValueError("Original baseline calibration membership changed")
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"architecture": "2560-512-256-128-15, GELU, first-hidden LayerNorm, dropout 0.05",
        "output": "mu_B = (B-1)*sigmoid(logit_B), independent B2--B16 means",
        "loss": "unweighted mean squared error in accepted-token counts across all state/action pairs",
        "optimizer": "AdamW lr=3e-4 weight_decay=0.01, grad_norm_clip=1.0, no scheduler",
        "checkpoint_selection": "minimum calibration budget at >=96% actual B16 retention; ties higher retention then earlier epoch",
        "policy": "argmax_B mu_B - lambda*(B-1), calibration upper-envelope breakpoints; fixed B16 fallback",
        "initialization": "from scratch; identical initial parameters, epoch permutations and dropout RNG stream within seed pair",
        "prediction": "CPU FP32 for all calibration/assessment scores, dropout disabled; GPU FP32 training with TF32 off",
        "baseline": "frozen 100k-trained survival MLP, original .001 alpha rule; not training-size matched to the new 10k heads",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "checkpoint_sha256": CHECKPOINT_SHA,
        "evaluation_completion_sha256": eval_binding,
        "source_hashes": {str(f): sha256(f) for f in (Path(__file__), Path("dflash/block_response.py"),
            Path("scripts/collect_policy_granularity.py"), Path("docs/actual_block_predictor_20260928.md"))}})
    atomic_json(args.output/"config.json", config)
    atomic_json(args.output/"audit.json", audit)
    atomic_json(args.output/"training_index.json", index)
    print("AUDIT_PASSED", json.dumps(audit), flush=True)
    started = time.monotonic()
    device = "cuda" if args.gpu is not None else "cpu"
    tx = torch.from_numpy(x.astype(np.float32)).to(device)
    selected_models, histories, initialization_hashes = {}, {}, {}
    for seed in args.seeds:
        pair_hashes, pair_orders = [], []
        for mode in ("clipped", "actual"):
            name = f"{mode}_seed_{seed}"
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = BlockResponseMLP().to(device)
            initial_hash = weight_digest(model.state_dict())
            pair_hashes.append(initial_hash)
            ty = torch.from_numpy(training_targets(actual_train, mode)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            generator = torch.Generator().manual_seed(seed)
            order_digest = hashlib.sha256()
            history, best, best_score = [], None, None
            for epoch in range(1, args.epochs+1):
                model.train()
                order = torch.randperm(len(x), generator=generator)
                order_digest.update(order.numpy().tobytes())
                total_loss = 0.0
                for start in range(0, len(x), args.batch_size):
                    batch = order[start:start+args.batch_size].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(tx[batch])-ty[batch]).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += float(loss.detach())*len(batch)
                # deepcopy consumes no RNG; matched dropout streams are not changed by scoring.
                cal_model = copy.deepcopy(model).cpu().eval()
                mu = predict_cpu(cal_model, eval_features[cal])
                curves = calibration_curve(mu, actual_eval[cal])
                points = select_response_points(curves)
                primary = points["0.96"]
                record = {"epoch": epoch, "train_mse": total_loss/len(x),
                    "actual_calibration_mse": float(np.mean((mu-actual_eval[cal])**2)), "primary_calibration": primary}
                history.append(record)
                print("EPOCH", name, json.dumps(record), flush=True)
                score = (primary["mean_budget"], -primary["retention"], epoch)
                if best_score is None or score < best_score:
                    best_score = score
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best = {"name": name, "mode": mode, "seed": seed, "epoch": epoch, "model": state,
                        "calibration_points": points, "initial_state_sha256": initial_hash,
                        "state_sha256": weight_digest(state), "parameter_count": sum(p.numel() for p in model.parameters()),
                        "training_config_sha256": sha256(args.output/"config.json")}
                    path = args.output/f"{name}.pt"
                    temporary = path.with_suffix(".pt.tmp")
                    torch.save(best, temporary)
                    temporary.replace(path)
                    atomic_json(args.output/f"{name}_calibration.json", {"epoch": epoch, "selected": points, "curves": curves})
                atomic_json(args.output/f"{name}_history.json", history)
                del cal_model
            digest = order_digest.hexdigest()
            pair_orders.append(digest)
            histories[name] = history
            initialization_hashes[name] = {"initial_state": initial_hash, "epoch_orders": digest}
            selected_models[name] = {k: v for k, v in best.items() if k != "model"}
            del optimizer, model, ty
        if len(set(pair_hashes)) != 1 or len(set(pair_orders)) != 1:
            raise AssertionError("Supervision pair did not share initialization/order")
    del tx
    if args.gpu is not None:
        torch.cuda.empty_cache()
    # All six epochs/checkpoints are frozen before evaluating assessment outcomes.
    atomic_json(args.output/"selection_frozen.json", {"models": selected_models, "matched_randomness": initialization_hashes,
        "evaluation_completion_sha256": eval_binding})
    assessments, choices, prediction_arrays, assessment_curves = {}, {}, {}, {}
    for name, selection in selected_models.items():
        saved = torch.load(args.output/f"{name}.pt", map_location="cpu", weights_only=False)
        if weight_digest(saved["model"]) != selection["state_sha256"]:
            raise ValueError("Selected checkpoint reload mismatch")
        model = BlockResponseMLP().eval()
        model.load_state_dict(saved["model"])
        mu = predict_cpu(model, eval_features[assess])
        prediction_arrays[name] = mu
        points = selection["calibration_points"]
        assessments[name] = {"seed": selection["seed"], "epoch": selection["epoch"], "mode": selection["mode"],
            "actual_mse_by_block": ((mu-actual_eval[assess])**2).mean(0).tolist(),
            "actual_mae_by_block": np.abs(mu-actual_eval[assess]).mean(0).tolist(), "operating_points": {}}
        for target, point in points.items():
            d = choose_response_budget(mu, point["setting"])
            assessments[name]["operating_points"][target] = {"setting": point["setting"],
                "calibration_retention": point["retention"], **response_metrics(actual_eval[assess], d)}
            if target == "0.96":
                choices[name] = d
        curves = json.loads((args.output/f"{name}_calibration.json").read_text())["curves"]
        assessment_curves[name] = [{"setting": p["setting"], **response_metrics(actual_eval[assess], choose_response_budget(mu, p["setting"]))} for p in curves]
    frozen = HorizonPredictorRuntime(input_dim=2560, proj_dim=512, hidden_size=256, num_slots=15,
        architecture="last_mlp", num_layers=1, dropout=.05, context_window=16).eval().requires_grad_(False)
    frozen.load_state_dict(checkpoint["model"])
    ss = []
    with torch.inference_mode():
        for start in range(0, len(eval_features), 256):
            xx = torch.from_numpy(eval_features[start:start+256].astype(np.float32))[:, None]
            ss.append(acceptance_survival(frozen(xx, torch.ones(xx.shape[:2]))).numpy())
    survival = np.concatenate(ss)
    base_points, _ = select_operating_points(actual_eval[cal], survival[cal], survival[cal])
    controls = {}
    for name, policy in (("frozen_100k_mlp", "cycle"), ("fixed", "fixed")):
        controls[name] = {}
        for target, points in base_points.items():
            point = points[policy]
            d = (budgets_from_survival(survival[assess], point["setting"]) if policy == "cycle"
                 else np.full(int(assess.sum()), point["setting"]))
            controls[name][target] = {"setting": point["setting"], "calibration_retention": point["retention"],
                                      **response_metrics(actual_eval[assess], d)}
            if target == "0.96":
                choices[name] = d
    uncertainty = bootstrap(actual_eval[assess], prompt_eval[assess], choices)
    for name, value in assessments.items():
        primary = value["operating_points"]["0.96"]
        value["meets_70_ratio_and_96_retention_point_estimate"] = primary["aggregate_accept_ratio"] >= .70 and primary["retention"] >= .96
    across_seeds = {}
    for mode in ("clipped", "actual"):
        points = [value["operating_points"]["0.96"] for value in assessments.values() if value["mode"] == mode]
        across_seeds[mode] = {metric: {"mean": float(np.mean([p[metric] for p in points])),
            "std": float(np.std([p[metric] for p in points], ddof=1)) if len(points)>1 else None,
            "min": float(min(p[metric] for p in points)), "max": float(max(p[metric] for p in points))}
            for metric in ("aggregate_accept_ratio", "retention", "mean_budget")}
    if sha256(args.eval_cache/"COMPLETE.json") != eval_binding:
        raise ValueError("Evaluation completion binding changed during training")
    summary = {"scope": "SMOKE ONLY" if args.smoke else "10k training-cycle matched supervision pilot; previously inspected development evaluation",
        "counts": {"training": {"rows": len(x), "prompts": audit["training_prompts"]},
            **{g: {"rows": int(mask.sum()), "prompts": len(np.unique(prompt_eval[mask]))} for g, mask in (("calibration", cal), ("assessment", assess))}},
        "models": assessments, "controls": controls, "across_seeds": across_seeds, "uncertainty": uncertainty,
        "elapsed_s": time.monotonic()-started, "reload_and_initialization_checks_passed": True,
        "limitations": ["actual-block outcomes on shared B16 trajectories, not closed-loop", "development assessment already inspected",
            "no throughput claim", "three seeds do not cover all uncertainty", "calibration retention is not an assessment guarantee",
            "frozen baseline had 100k historical training rows, new heads have 10k fresh rows; only the label pair is fully matched",
            "at most eight progress-spaced states per prompt; not the full cycle population"]}
    atomic_json(args.output/"summary.json", summary)
    atomic_json(args.output/"assessment_curves.json", assessment_curves)
    np.savez_compressed(args.output/"assessment_predictions.npz", **prediction_arrays,
        actual=actual_eval[assess], prompt_id=prompt_eval[assess], frozen_survival=survival[assess])
    lines = ["# Actual-block supervision pilot", "", summary["scope"], "",
        "Primary: settings and checkpoint selected on calibration for >=96% actual B16 retention.", "",
        "| Model | Epoch | Cal retention | Assessment ratio | Assessment retention | Mean budget |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, value in controls.items():
        q = value["0.96"]
        lines.append(f"| {name} | frozen | {q['calibration_retention']:.5f} | {q['aggregate_accept_ratio']:.5f} | {q['retention']:.5f} | {q['mean_budget']:.4f} |")
    for name, value in assessments.items():
        q = value["operating_points"]["0.96"]
        lines.append(f"| {name} | {value['epoch']} | {q['calibration_retention']:.5f} | {q['aggregate_accept_ratio']:.5f} | {q['retention']:.5f} | {q['mean_budget']:.4f} |")
    lines += ["", "Every seed is reported; no assessment-based seed or checkpoint selection.", "",
        "Full operating curves, paired whole-prompt intervals, and seed summaries are in the JSON artifacts.", "",
        "## Limitations", ""]+["- "+s for s in summary["limitations"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    files = [p for p in args.output.iterdir() if p.is_file() and p.name != "COMPLETE.json"]
    atomic_json(args.output/"COMPLETE.json", {"success": True, "smoke": args.smoke,
        "binding": {p.name: sha256(p) for p in files}})
    print("COMPLETE", json.dumps({"counts": summary["counts"], "across_seeds": across_seeds, "elapsed_s": summary["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
