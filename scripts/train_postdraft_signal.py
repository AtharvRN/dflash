"""Frozen-DFlash post-draft signal audit; no redrafting or target inference.

All arms see identical cycles and use the same calibration-only policy family.
DSpark-form arms adapt its linear hidden/previous-token head, NOT its jointly
trained Markov module or full serving scheduler. Greedy-only evaluation.
"""
from __future__ import annotations

import argparse
import copy
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
from scripts.gpu_runtime import wait_gpu_runtime
from scripts.train_soft_supervision import load_cache, build_model as old_model, TARGETS
from scripts.train_midverify_scaling import batch_stream, parameter_sha, paired_intervals

ARMS = ("confidence", "dspark_form_hard", "dspark_form_tv", "residual_local", "residual_prefix")


def load_features(root):
    config, rows, arrays = load_cache(root)
    extra = {k: [] for k in ("draft_hidden", "candidate_ids")}
    for r in json.loads((root / "receipts.json").read_text()):
        if sha256(root / r["file"]) != r["sha256"]:
            raise ValueError("Changed cache during read")
        with np.load(root / r["file"], allow_pickle=False) as f:
            for k in extra:
                extra[k].append(f[k])
    arrays.update({k: np.concatenate(v) for k, v in extra.items()})
    n = len(rows)
    if arrays["draft_hidden"].shape != (n, 15, 2560) or not np.isfinite(arrays["draft_hidden"]).all():
        raise ValueError("Invalid draft hidden states")
    ids = arrays["candidate_ids"]
    if ids.shape != (n, 15) or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 151936)):
        raise ValueError("Invalid candidate token IDs")
    anchors = np.asarray([r["prefix_token_ids"][-1] for r in rows], dtype=np.int64)
    arrays["previous_ids"] = np.concatenate((anchors[:, None], ids[:, :-1]), 1)
    return config, rows, arrays


def policy_scores(logits, rule):
    if rule == "local":
        return np.asarray(logits, dtype=np.float64)
    if rule == "survival":
        return np.cumsum(-np.logaddexp(0, -np.asarray(logits, dtype=np.float64)), axis=1)
    raise ValueError("Unknown policy rule")


def select_policy(logits, accepted, targets=TARGETS):
    # Same two predeclared prefix rules for EVERY learned and raw baseline.
    rules = {r: calibrate(policy_scores(logits, r), accepted, targets) for r in ("local", "survival")}
    return {t: min((dict(p[t], rule=r) for r, p in rules.items()),
                   key=lambda s: (s["mean_kept_rows"], -s["retention"], s["rule"]))
            for t in rules["local"]}


def decisions(logits, setting):
    return apply_setting(policy_scores(logits, setting["rule"]), setting)


def raw_logits(logprob):
    p = np.clip(np.exp(np.asarray(logprob, dtype=np.float64)), 1e-9, 1-1e-9)
    return np.log(p)-np.log1p(-p)


def make_model(arm, width=2560, vocab=151936, base=None):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            if arm == "confidence":
                self.net = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 1))
            elif arm.startswith("dspark_form_"):
                self.previous = nn.Embedding(vocab, 32)
                self.readout = nn.Linear(width+32, 1, bias=False)
            elif arm in ("residual_local", "residual_prefix"):
                if base is None:
                    raise ValueError("Residual requires fitted confidence base")
                self.base = copy.deepcopy(base).requires_grad_(False)
                self.hidden_proj = nn.Linear(width, 32)
                self.token_proj = nn.Linear(width, 32)
                self.net = nn.Sequential(nn.Linear(195, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 1))
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)
            else:
                raise ValueError("Unknown arm")

        def forward(self, conf, hidden, candidate, previous):
            if self.arm == "confidence":
                return self.net(conf).squeeze(-1)
            if self.arm.startswith("dspark_form_"):
                return self.readout(torch.cat((hidden, self.previous(previous)), -1)).squeeze(-1)
            self.base.eval()
            h = torch.nn.functional.gelu(self.hidden_proj(hidden))
            e = torch.nn.functional.gelu(self.token_proj(candidate))
            z = torch.cat((h, e, h*e), -1)
            context = z if self.arm == "residual_local" else z.cumsum(1)/torch.arange(
                1, z.shape[1]+1, device=z.device, dtype=z.dtype)[None, :, None]
            return self.base(conf, hidden, candidate, previous) + self.net(torch.cat((z, context, conf), -1)).squeeze(-1)

    return Head()


def loss_fn(logits, hard, tv, risk, arm):
    import torch.nn.functional as F
    target = tv if arm == "dspark_form_tv" else hard
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per*risk).sum()/risk.sum().clamp_min(1)


def score(model, features, indices, batch=256):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(*(v[indices[i:i+batch]] for v in features)).cpu().numpy()
                               for i in range(0, len(indices), batch)])


def auc_parts(scores, labels):
    """Pair count weighted AUC numerator; exact average ranks for ties."""
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return 0., 0
    _, inv, count = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = np.cumsum(count)-(count-1)/2
    return float(ranks[inv][labels].sum()-pos*(pos+1)/2), pos*neg


def conditioned_auc(logits, arrays, train, assess):
    # Bin edges are fitted ONLY to training at-risk draft confidence, separately
    # per position. No assessment-based feature/threshold/checkpoint selection.
    numerator, denominator, groups = 0., 0, 0
    for j in range(15):
        q = arrays["draft_stats"][:, j, 0]
        trisk = arrays["at_risk"][train, j]
        if not trisk.any():
            continue
        edges = np.unique(np.quantile(q[train][trisk], [.2, .4, .6, .8]))
        bins = np.searchsorted(edges, q[assess], side="right")
        for b in range(len(edges)+1):
            use = (bins == b) & arrays["at_risk"][assess, j]
            num, den = auc_parts(logits[use, j], arrays["matches"][assess, j][use])
            numerator += num
            denominator += den
            groups += den > 0
    return {"auc": numerator/denominator if denominator else None, "positive_negative_pairs": denominator,
            "groups": int(groups), "conditioning": "position and training-fitted confidence quintile; at-risk only",
            "scope": "descriptive residual ranking signal; coarse bins do not remove all confidence variation"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "old_training", "output", "backup"):
        p.add_argument("--"+k.replace("_", "-"), type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.backup.exists() or args.output.resolve() == args.backup.resolve():
        raise ValueError("Fresh independent destinations required")
    started = time.monotonic()
    runtime = wait_gpu_runtime(use_container_gpu=True, require_gpu=True)
    collection, rows, arrays = load_features(args.cache)
    # Smoke uses exactly the same audited cache but a bounded eight-update fit.
    groups = np.array([r["group"] for r in rows])
    select = {g: np.flatnonzero((groups == g) & arrays["eligible"]) for g in ("train", "calibration", "assessment")}
    tr, ca, ass = (select[g] for g in select)
    if min(map(len, select.values())) == 0:
        raise ValueError("Empty eligible group")
    train_stats = arrays["draft_stats"][tr, :, :2][arrays["at_risk"][tr]]
    mean, std = train_stats.mean(0, dtype=np.float64), np.maximum(train_stats.std(0, dtype=np.float64), 1e-6)
    conf = ((arrays["draft_stats"][:, :, :2]-mean)/std).astype(np.float32)
    position = np.broadcast_to(np.arange(15, dtype=np.float32)[None, :, None]/14, (*conf.shape[:2], 1))
    conf = np.concatenate((conf, position), -1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    floats = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda")
    norm = lambda x: x/torch.sqrt(x.square().mean(-1, keepdim=True)+1e-6)
    # Preserve raw hidden scale for the linear DSpark-form head; residual heads
    # get RMS-normalized hidden features via a separate view below.
    hidden = floats(arrays["draft_hidden"])
    features = (floats(conf), hidden, norm(floats(arrays["candidate_vectors"])),
                torch.as_tensor(arrays["previous_ids"], dtype=torch.long, device="cuda"))
    residual_features = (features[0], norm(hidden), *features[2:])
    y, tv, risk = (floats(arrays[k]) for k in ("matches", "tv_overlap", "at_risk"))
    accepted = arrays["accepted_len"]
    seeds, updates, every = ([913], 8, 4) if args.smoke else ([913, 914, 915], 1024, 64)
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    def save(name, obj):
        atomic_json(args.output/name, obj)
        backup_file(args.output/name, args.backup/name)
    config = {"schema": "postdraft_signal_v1", "runtime": runtime, "cache_sha256": sha256(args.cache/"COMPLETE.json"),
              "old_training_sha256": sha256(args.old_training/"COMPLETE.json"), "seeds": seeds, "updates": updates,
              "calibrate_every": every, "batch_size": 128, "arms": ARMS, "smoke": args.smoke,
              "normalization": {"mean": mean.tolist(), "std": std.tolist(), "scope": "training at-risk only"},
              "counts": {g: len(idx) for g, idx in select.items()},
              "optimizer": "AdamW3e-4 weight_decay.01 clip1; same seed/minibatch stream; no architecture sweep",
              "policy": "same local-logit gate and log-survival gate candidates for all arms; calibration-only selection",
              "checkpoint_selection": "minimum calibration mean K at >=96% retention; higher retention then earlier update",
              "dspark_boundary": "linear [raw draft hidden; learned previous-token embedding32], TV-only or hard-only at-risk BCE; standalone frozen DFlash adaptation, not shared Markov embedding or full DSpark reproduction",
              "residual": "frozen confidence MLP + zero-initialized correction from projected draft hidden/candidate/product; matched local-vs-prefix-mean architecture",
              "confidence": "log candidate probability, entropy, position; cached candidate-minus-max logit is zero for greedy and excluded",
              "scope": "B16 greedy unchanged candidates; offline work, not speedup; inspected development assessment",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": sha256(Path(__file__))}
    save("config.json", config)
    save("row_selection.json", {g: idx.tolist() for g, idx in select.items()})
    results, predictions, ref_k = {}, {}, {}
    def report(name, qc, qa, metadata):
        settings = select_policy(qc, accepted[ca])
        operating = {}
        for t, setting in settings.items():
            k = decisions(qa, setting)
            operating[t] = {"calibration": setting, "assessment": metrics(accepted[ass], k),
                            "delta_vs_raw": paired_intervals(accepted[ass], ref_k.get(t, k), k,
                                np.array([rows[i]["prompt_id"] for i in ass]), 0,
                                draws=100 if args.smoke else 1000)}
            if name == "raw":
                ref_k[t] = k
        prob = np.exp(-np.logaddexp(0, -qa))
        mask = arrays["at_risk"][ass]
        results[name] = {**metadata, "operating_points": operating,
                         "conditional_brier": float(((prob-arrays["matches"][ass])**2)[mask].mean()),
                         "within_confidence_auc": conditioned_auc(qa, arrays, tr, ass)}
        predictions[name] = qa
        np.savez(args.output/(name+"_predictions.npz"), calibration=qc, assessment=qa)
        backup_file(args.output/(name+"_predictions.npz"), args.backup/(name+"_predictions.npz"))
        save("progress.json", {"complete": False, "completed": list(results), "latest": results[name],
                               "elapsed_seconds": time.monotonic()-started})
        print("MODEL_COMPLETE", name, json.dumps(operating["0.96"]), flush=True)

    raw = raw_logits(arrays["draft_stats"][:, :, 0])
    report("raw", raw[ca], raw[ass], {"parameters": 0})
    # Reuse all old arms/seeds; never choose the 'best' using assessment results.
    complete = json.loads((args.old_training/"COMPLETE.json").read_text())
    if not complete.get("complete"):
        raise ValueError("Old training incomplete")
    old_config = json.loads((args.old_training/"config.json").read_text())
    if old_config["cache_complete_sha256"] != config["cache_sha256"]:
        raise ValueError("Old head trained on another cache")
    for filename in ("config.json", "row_selection.json"):
        if sha256(args.old_training/filename) != complete["binding"][filename]:
            raise ValueError("Old provenance checksum mismatch")
    if json.loads((args.old_training/"row_selection.json").read_text()) != {g: idx.tolist() for g, idx in select.items()}:
        raise ValueError("Old evaluation membership changed")
    old_stats = old_config["normalization"]
    old_x = torch.cat((features[2], floats((arrays["draft_stats"]-np.array(old_stats["draft_mean"]))/np.array(old_stats["draft_std"]))), -1)
    for seed in seeds:
        for arm in ("hard", "hard_tv", "hard_margin"):
            filename = f"{arm}_seed{seed}.pt"
            if sha256(args.old_training/filename) != complete["binding"][filename]:
                raise ValueError("Old checkpoint checksum mismatch")
            model = old_model().cuda().eval()
            checkpoint = torch.load(args.old_training/filename, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            with torch.inference_mode():
                out = np.concatenate([model(old_x[i:i+256])[..., 0].cpu().numpy() for i in range(0, len(rows), 256)])
            report("existing_"+filename[:-3], out[ca], out[ass], {"reused_checkpoint_sha256": sha256(args.old_training/filename)})
    del old_x, model
    for seed in seeds:
        base, paired_initial, order_reference = None, {}, None
        for arm in ARMS:
            torch.manual_seed(seed)
            model = make_model(arm, base=base).cuda()
            initial = parameter_sha(model.state_dict())
            pair = "dspark" if arm.startswith("dspark") else "residual" if arm.startswith("residual") else arm
            if pair in paired_initial and paired_initial[pair] != initial:
                raise ValueError("Paired architecture initialization differs")
            paired_initial[pair] = initial
            f = residual_features if arm.startswith("residual") else features
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
            stream, order = batch_stream(len(tr), 128, seed), hashlib.sha256()
            best, history = None, []
            # Zero-correction checkpoint is admissible: residual cannot silently
            # exclude its own frozen confidence baseline on calibration.
            for step in range(updates+1):
                if step:
                    idx = next(stream)
                    order.update(idx.numpy().tobytes())
                    b = torch.as_tensor(tr[idx.numpy()], device="cuda")
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(*(v[b] for v in f)), y[b], tv[b], risk[b], arm)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                if step % every:
                    continue
                qc = score(model, f, ca)
                setting = select_policy(qc, accepted[ca], (.96,))["0.96"]
                key = (setting["mean_kept_rows"], -setting["retention"], step)
                if best is None or key < best[0]:
                    best = (key, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, qc.copy())
                history.append({"step": step, "loss": float(loss.detach()) if step else None, "calibration": setting})
                print("UPDATE", arm, seed, step, setting["mean_kept_rows"], flush=True)
            if order_reference is not None and order_reference != order.hexdigest():
                raise ValueError("Unmatched training minibatches")
            order_reference = order.hexdigest()
            name = f"{arm}_seed{seed}"
            path = args.output/(name+".pt")
            torch.save({"model": best[1], "config": config, "arm": arm, "seed": seed, "selected_step": best[0][2]}, path)
            backup_file(path, args.backup/path.name)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            qc = score(model, f, ca)
            if not np.array_equal(qc, best[2]):
                raise ValueError("Reload changed calibration predictions")
            if arm == "confidence":
                base = copy.deepcopy(model).eval()
            save(name+"_history.json", history)
            report(name, qc, score(model, f, ass), {"selected_step": best[0][2], "initial_sha256": initial,
                   "order_sha256": order.hexdigest(), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    save("summary.json", {"models": results, "config": config, "elapsed_seconds": time.monotonic()-started,
                           "limitations": ["Calibration retention is not assessment guarantee; compare full frontiers",
                               "DSpark-form TV label is temperature-one stochastic overlap, NOT greedy match probability",
                               "No extra target forward, no drafter changes, no editor, no throughput claim",
                               "History and cross-drafter transfer not evaluated in this first signal audit"]})
    save("COMPLETE.json", {"complete": True, "binding": {f.name: sha256(f) for f in args.output.iterdir() if f.is_file()}})
    print("COMPLETE", args.backup, flush=True)


if __name__ == "__main__":
    main()
