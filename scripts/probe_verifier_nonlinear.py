"""Exploratory nonlinear follow-up after inspecting the linear signal probes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_acceptance_geometry import digest, assessment_metrics, grouped_indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refusing an existing destination")
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import normalize, StandardScaler
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    audit = json.loads((args.replay / "COMPLETE.json").read_text())
    for name in ("rows.jsonl", "fused.npy", "anchor_embedding.npy", "predraft_history.npy", "predraft_history_mask.npy"):
        if digest(args.replay / name) != audit["binding"][name]:
            raise ValueError("Audit mismatch: " + name)
    original = [json.loads(line) for line in (args.replay / "rows.jsonl").read_text().splitlines()]
    keep = np.array([r["eligible"] for r in original])
    rows = [r for r in original if r["eligible"]]
    y = np.array([r["accepted_len"] for r in rows], dtype=float)
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
    result = {"scope": "Exploratory nonlinear diagnostic added after seeing linear-probe results; development assessment, not a fresh confirmatory test.",
              "model": "HistGradientBoostingRegressor; squared-error loss; 200 iterations; learning rate .05; min leaf 20; no early stopping",
              "selection": "Calibration MSE over max_leaf_nodes=(7,15), l2_regularization=(1,10). Train only fits.",
              "script_sha256": digest(__file__), "seed": 927, "probes": {}}
    predictions = {}
    for name, x in inputs.items():
        candidates = []
        for leaves in (7, 15):
            for l2 in (1., 10.):
                model = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=20, l2_regularization=l2, early_stopping=False, random_state=927).fit(x[tr], y[tr])
                pred = np.clip(model.predict(x), 0, 15)
                candidates.append((float(np.mean((pred[cal] - y[cal])**2)), leaves, l2, pred))
        score, leaves, l2, prediction = min(candidates, key=lambda c:c[0])
        predictions[name] = prediction
        stats = assessment_metrics(y[assess], prediction[assess], prompts[assess], 927)
        stats.update({"selected_leaves": leaves, "selected_l2": l2, "calibration_mse": score})
        result["probes"][name] = stats
        print(name, json.dumps(stats), flush=True)
    indices = grouped_indices(prompts[assess])
    counts = np.array([len(i) for i in indices])
    draws = np.random.default_rng(927).integers(len(indices), size=(1000, len(indices)))
    baseline = np.abs(y[assess] - predictions["Fused"][assess])
    for name, prediction in predictions.items():
        delta = np.abs(y[assess] - prediction[assess]) - baseline
        sums = np.array([delta[i].sum() for i in indices])
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        result["probes"][name].update({"delta_mae_vs_fused": float(delta.mean()),
                                       "paired_delta_95": np.quantile(boot, [.025, .975]).tolist()})
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", y=y, group=group, prompt_id=prompts,
                        **{f"pred_{i}": v for i,v in enumerate(predictions.values())})
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
