"""Exploratory proper-score test of incremental causal verifier information."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_acceptance_geometry import digest, grouped_indices


def score_rows(q, accepted):
    """Conditional-success hazards, with A=15 right-censored at the block cap."""
    q = np.clip(q, 1e-7, 1 - 1e-7)
    slots = np.arange(1, 16)[None, :]
    success = slots <= accepted[:, None]
    failure = slots == accepted[:, None] + 1
    nll = -(np.log(q) * success + np.log1p(-q) * failure).sum(1)
    survival = np.cumprod(q, 1)
    return {"nll": nll, "brier": np.square(survival - success).mean(1),
            "mae": np.abs(survival.sum(1) - accepted)}, survival


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing an existing output")
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize, StandardScaler
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    audit = json.loads((args.replay / "COMPLETE.json").read_text())
    for name in ("rows.jsonl", "fused.npy", "anchor_embedding.npy", "predraft_history.npy", "predraft_history_mask.npy"):
        if digest(args.replay / name) != audit["binding"][name]:
            raise ValueError("Audit mismatch")
    original = [json.loads(line) for line in (args.replay / "rows.jsonl").read_text().splitlines()]
    keep = np.array([r["eligible"] for r in original])
    rows = [r for r in original if r["eligible"]]
    y = np.array([r["accepted_len"] for r in rows])
    group = np.array([r["group"] for r in rows])
    prompts = np.array([r["prompt_id"] for r in rows])
    tr, cal, assess = group == "train", group == "calibration", group == "assessment"
    def load(name):
        return np.load(args.replay / f"{name}.npy", mmap_mode="r")[keep].astype(np.float32)
    def pc(name, n):
        x = normalize(load(name))
        return PCA(n_components=n, svd_solver="randomized", iterated_power=4, random_state=927).fit(x[tr]).transform(x)
    sources = np.array([r["source"] for r in rows])
    meta = np.column_stack([*(sources == s for s in sorted(set(sources))),
        np.log1p([r["prompt_tokens"] for r in rows]), np.log1p([r["prefix_length"] for r in rows]),
        [r["cycle_id"] for r in rows], [r["prefix_length"] - r["prompt_tokens"] for r in rows]])
    base = np.column_stack([meta, pc("fused", 64)])
    history, mask = load("predraft_history"), load("predraft_history_mask")
    inputs = {"Fused": base,
              "Fused + latest verifier confidence": np.column_stack([base, history[:, -1]]),
              "Fused + verifier history": np.column_stack([base, history.reshape(len(rows), -1), mask]),
              "Fused + anchor + confidence": np.column_stack([base, pc("anchor_embedding", 32), history[:, -1]])}
    result = {"scope": "Post-hoc exploratory distributional probe after linear/nonlinear mean probes. Previously inspected development assessment, not confirmatory evidence.",
              "model": "15 conditional-success logistic heads, correctly censored at A=15; L2 penalty; C selected by calibration first-rejection NLL.",
              "script_sha256": digest(__file__), "probes": {}}
    predictions, all_scores = {}, {}
    for name, x in inputs.items():
        x = StandardScaler().fit(x[tr]).transform(x)
        candidates = []
        for c in (.01, .1, 1., 10.):
            q = np.empty((len(y), 15))
            for k in range(1, 16):
                risk = tr & (y >= k - 1)
                target = (y[risk] >= k).astype(int)
                if len(np.unique(target)) < 2:
                    q[:, k - 1] = (target.sum() + .5) / (len(target) + 1)
                else:
                    model = LogisticRegression(C=c, max_iter=500, solver="lbfgs", tol=1e-6, random_state=927).fit(x[risk], target)
                    q[:, k - 1] = model.predict_proba(x)[:, 1]
            scores, survival = score_rows(q, y)
            candidates.append((float(scores["nll"][cal].mean()), c, scores, survival))
        value, c, scores, survival = min(candidates, key=lambda a:a[0])
        predictions[name], all_scores[name] = survival, scores
        result["probes"][name] = {"C": c, "calibration_nll": value,
            **{k:float(v[assess].mean()) for k,v in scores.items()}}
        print(name, json.dumps(result["probes"][name]), flush=True)
    indices = grouped_indices(prompts[assess])
    counts = np.array([len(i) for i in indices])
    draws = np.random.default_rng(927).integers(len(indices), size=(1000, len(indices)))
    for name, scores in all_scores.items():
        for key, values in scores.items():
            delta = values[assess] - all_scores["Fused"][key][assess]
            sums = np.array([delta[i].sum() for i in indices])
            boot = sums[draws].sum(1) / counts[draws].sum(1)
            result["probes"][name]["delta_" + key] = float(delta.mean())
            result["probes"][name]["delta_" + key + "_95"] = np.quantile(boot, [.025, .975]).tolist()
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", y=y, group=group, prompt_id=prompts,
                        **{f"survival_{i}":v for i,v in enumerate(predictions.values())})
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
