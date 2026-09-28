"""CPU-only exploratory geometry, recent-text, and temporal acceptance analysis.

Reads an audited paired B16 cache without modifying it. No target/draft model
is loaded. All predictive transforms fit train only; calibration selects Ridge
regularization, and assessment prompts are used once for reporting. The 2-D
embeddings are descriptive, not the space used to evaluate prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def recent_tokens(trajectory, start, window):
    """Strictly before the current anchor and all current draft/verify tokens."""
    if not 0 <= start < len(trajectory) or window < 1:
        raise ValueError("Invalid state/window")
    return trajectory[max(0, start - window):start]


def history_features(labels, cycles, fallback=0.0):
    """Only earlier observed labels; gaps invalidate the immediately-previous A."""
    result = np.zeros((len(labels), 5), dtype=np.float32)
    total, count, ema = 0.0, 0, fallback
    for i, (label, cycle) in enumerate(zip(labels, cycles)):
        consecutive = i > 0 and cycles[i - 1] == cycle - 1
        result[i] = [labels[i - 1] if consecutive else fallback,
                     total / count if count else fallback, ema,
                     float(consecutive), float(count > 0)]
        total += float(label)
        count += 1
        ema = float(label) if count == 1 else .75 * ema + .25 * float(label)
    return result


def grouped_indices(groups):
    return [np.flatnonzero(groups == g) for g in np.unique(groups)]


def prompt_balanced_indices(prompts, cap, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(idx, min(cap, len(idx)), replace=False)
                                   for idx in grouped_indices(prompts)]))


def demean_by_prompt(values, prompts):
    result = np.asarray(values, dtype=float).copy()
    for idx in grouped_indices(prompts):
        result[idx] -= result[idx].mean()
    return result


def variance_partition(y, prompts):
    residual = demean_by_prompt(y, prompts)
    total = float(np.square(y - y.mean()).sum())
    within = float(np.square(residual).sum())
    return {"within_prompt_fraction": within / total,
            "between_prompt_fraction": 1 - within / total,
            "note": "Descriptive sums-of-squares partition; not noise-corrected ICC."}


def load_cache(cache, prompt_manifest):
    info = json.loads((cache / "manifest.json").read_text())
    audit = json.loads((cache / "audit.json").read_text())
    required = {"manifest.json"} | {f"{s}/{f}.npy" for s in ("train", "val")
                  for f in ("features", "raw_features", "row_index", "accepted_len")}
    required.add("val/calibration.npy")
    if not required <= set(audit["binding"]):
        raise ValueError("Audit does not bind the analysis inputs")
    for name, expected in audit["binding"].items():
        if digest(cache / name) != expected:
            raise ValueError(f"Stale/tampered audit binding: {name}")
    if info["format"] != "dflash_prefusion_cache_v1":
        raise ValueError("Wrong cache format")
    ids = np.concatenate([np.load(cache / s / "row_index.npy") for s in ("train", "val")])
    y = ids[:, 4].astype(float)
    n_train = info["train_rows"]
    calibration = np.load(cache / "val/calibration.npy")
    group = np.array(["train"] * n_train + ["calibration" if x else "assessment" for x in calibration])
    prompts, cycles = ids[:, 2], ids[:, 3]
    group_prompts = {g: set(prompts[group == g]) for g in np.unique(group)}
    for a, b in (("train", "calibration"), ("train", "assessment"), ("calibration", "assessment")):
        if group_prompts[a] & group_prompts[b]:
            raise ValueError("Prompt split leakage")
    source_by_prompt = {}
    wanted = set(prompts.tolist())
    with prompt_manifest.open() as f:
        for line in f:
            row = json.loads(line)
            if row["manifest_index"] in wanted:
                source_by_prompt[row["manifest_index"]] = row["source"]
    if set(source_by_prompt) != wanted:
        raise ValueError("Missing source provenance")
    n = len(y)
    starts, prompt_lens = np.zeros(n, int), np.zeros(n, int)
    histories = np.zeros((n, 5), np.float32)
    documents = {key: [""] * n for key in ("prompt", "recent16", "recent64", "recent256")}
    snippets, assigned = {}, np.zeros(n, bool)
    for shard in info["shards"]:
        if not shard.get("rows"):
            continue
        path = cache / "shards" / Path(shard["path"]).name
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path}")
        idx = np.flatnonzero(prompts == shard["prompt_id"])
        with np.load(path, allow_pickle=False) as data:
            if len(idx) != shard["rows"] or not np.array_equal(data["accepted_len"], y[idx]):
                raise ValueError("Shard label alignment")
            if not np.array_equal(data["cycle_id"], cycles[idx]):
                raise ValueError("Shard cycle alignment")
            trajectory = data["trajectory_token_ids"]
            starts[idx] = data["prefix_length"]
            prompt_lens[idx] = shard["prompt_tokens"]
            histories[idx] = history_features(y[idx], cycles[idx])
            prompt_text = " ".join(map(str, trajectory[:shard["prompt_tokens"]]))
            for j, row_idx in enumerate(idx):
                start = int(starts[row_idx])
                prefix = trajectory[:start + 1].astype(np.int64)
                if hashlib.sha256(prefix.tobytes()).hexdigest() != str(data["prefix_sha256"][j]):
                    raise ValueError("Prefix hash mismatch")
                if int(trajectory[start]) != int(data["anchor_id"][j]):
                    raise ValueError("Anchor mismatch")
                documents["prompt"][row_idx] = prompt_text
                for w in (16, 64, 256):
                    documents[f"recent{w}"][row_idx] = " ".join(map(str, recent_tokens(trajectory, start, w)))
                snippets[int(row_idx)] = recent_tokens(trajectory, start, 64).tolist()
        assigned[idx] = True
    if not assigned.all() or (starts < prompt_lens).any():
        raise ValueError("Incomplete or invalid state metadata")
    sources = np.array([source_by_prompt[int(p)] for p in prompts])
    return {"info": info, "y": y, "prompts": prompts, "cycles": cycles, "group": group,
            "sources": sources, "starts": starts, "prompt_lens": prompt_lens,
            "history": histories, "documents": documents, "snippets": snippets,
            "audit": {"all_original_audit_bindings_verified": True,
                      "shard_hashes_and_prefixes_verified": True,
                      "cache_manifest_sha256": digest(cache / "manifest.json"),
                      "prompt_manifest_sha256": digest(prompt_manifest)}}


def assessment_metrics(y, prediction, prompts, seed):
    from scipy.stats import spearmanr
    prediction = np.clip(prediction, 0, 15)
    error = np.abs(y - prediction)
    idxs = grouped_indices(prompts)
    totals = np.array([error[i].sum() for i in idxs])
    counts = np.array([len(i) for i in idxs])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(idxs), size=(1000, len(idxs)))
    boot = totals[draws].sum(1) / counts[draws].sum(1)
    residual = demean_by_prompt(y, prompts)
    pred_residual = demean_by_prompt(prediction, prompts)
    rho = float(spearmanr(y, prediction).statistic) if np.std(prediction) > 1e-10 else None
    return {"mae": float(error.mean()), "mae_prompt_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "prompt_equal_weight_mae": float(np.mean(totals / counts)),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "r2": float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)),
            "spearman": rho,
            "within_prompt_r2_descriptive": float(1 - np.sum((residual - pred_residual) ** 2) / np.sum(residual ** 2))}


def ridge_probe(x, data):
    from sklearn.linear_model import Ridge
    tr, cal = data["group"] == "train", data["group"] == "calibration"
    scores, models = {}, {}
    for alpha in (.1, 1., 10., 100.):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=1000).fit(x[tr], data["y"][tr])
        scores[str(alpha)] = float(np.mean((np.clip(model.predict(x[cal]), 0, 15) - data["y"][cal]) ** 2))
        models[str(alpha)] = model
    best = min(scores, key=scores.get)
    return np.clip(models[best].predict(x), 0, 15), {"alpha": float(best), "calibration_mse": scores}


def temporal_analysis(data, seed):
    y, p, cycles = data["y"], data["prompts"], data["cycles"]
    groups = grouped_indices(p)
    residual = demean_by_prompt(y, p)
    output = {"variance": variance_partition(y, p), "lags": {}}
    rng = np.random.default_rng(seed)
    for lag in (1, 2, 4, 8, 16):
        left, right = [], []
        for idx in groups:
            lookup = {int(cycles[j]): j for j in idx}
            for j in idx:
                if int(cycles[j]) + lag in lookup:
                    left.append(j)
                    right.append(lookup[int(cycles[j]) + lag])
        left, right = np.array(left), np.array(right)
        observed = float(np.corrcoef(y[left], y[right])[0, 1])
        centered = float(np.corrcoef(residual[left], residual[right])[0, 1])
        null = []
        for _ in range(200):
            shuffled = residual.copy()
            for idx in groups:
                shuffled[idx] = rng.permutation(residual[idx])
            null.append(float(np.corrcoef(shuffled[left], shuffled[right])[0, 1]))
        output["lags"][str(lag)] = {"pairs": len(left), "raw_correlation": observed,
            "within_prompt_centered_correlation": centered,
            "within_prompt_shuffle_null_95": np.quantile(null, [.025, .975]).tolist(),
            "within_prompt_shuffle_null_mean": float(np.mean(null))}
    return output


def cross_prompt_neighbors(z, data, sample, seed):
    from sklearn.neighbors import NearestNeighbors
    p, y, source = data["prompts"][sample], data["y"][sample], data["sources"][sample]
    indices = NearestNeighbors(n_neighbors=min(64, len(sample)), n_jobs=4).fit(z[sample]).kneighbors(return_distance=False)
    other = []
    for i, row in enumerate(indices):
        candidates = row[p[row] != p[i]][:15]
        if len(candidates) != 15:
            raise ValueError("Insufficient cross-prompt neighbors")
        other.append(candidates)
    other = np.array(other)
    observed = float(np.abs(y[:, None] - y[other]).mean())
    source_match = float((source[:, None] == source[other]).mean())
    # Descriptive permutation control, not a prompt-independent hypothesis test.
    strata = np.array([f"{s}:{int(c)//8}" for s, c in zip(source, data["cycles"][sample])])
    strata_groups = grouped_indices(strata)
    rng, null = np.random.default_rng(seed), []
    for _ in range(200):
        shuffled = y.copy()
        for idx in strata_groups:
            shuffled[idx] = rng.permutation(y[idx])
        null.append(float(np.abs(shuffled[:, None] - shuffled[other]).mean()))
    return {"neighbor_label_absolute_difference": observed,
            "source_and_cycle_stratified_shuffle_mean": float(np.mean(null)),
            "shuffle_range_95": np.quantile(null, [.025, .975]).tolist(),
            "relative_reduction_vs_shuffle": float(1 - observed / np.mean(null)),
            "same_source_neighbor_fraction": source_match,
            "same_prompt_neighbors_excluded": True,
            "note": "PCA64 neighborhoods, descriptive permutation reference; not a significance test."}


def save_figure(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def geometry_figure(embeddings, data, sample, explained, out):
    import matplotlib.pyplot as plt
    y, source = data["y"][sample], data["sources"][sample]
    resid = demean_by_prompt(data["y"], data["prompts"])[sample]
    categories = sorted(set(source))
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, key in enumerate(("fused_pca", "fused_tsne_p50_s0")):
        xy = embeddings[key]
        a = axes[row, 0].scatter(*xy.T, c=y, cmap="viridis", vmin=0, vmax=15, s=7, alpha=.7, rasterized=True)
        fig.colorbar(a, ax=axes[row, 0], label="Accepted proposed tokens A (0–15)")
        for color, name in zip(palette, categories):
            mask = source == name
            axes[row, 1].scatter(*xy[mask].T, s=7, c=color, alpha=.65, label=name, rasterized=True)
        axes[row, 1].legend(markerscale=2, fontsize=8, loc="best")
        a = axes[row, 2].scatter(*xy.T, c=resid, cmap="coolwarm", vmin=-10, vmax=10, s=7, alpha=.7, rasterized=True)
        fig.colorbar(a, ax=axes[row, 2], label="A − same-prompt mean (descriptive)")
        for col in range(3):
            title = ["Acceptance length", "Dataset source", "Within-prompt acceptance variation"][col]
            axes[row, col].set_title(title)
            if row == 0:
                axes[row, col].set_xlabel(f"PC1 ({100*explained['fused'][0]:.1f}% variance)")
                axes[row, col].set_ylabel(f"PC2 ({100*explained['fused'][1]:.1f}% variance)")
            else:
                axes[row, col].set_xlabel("t-SNE 1 (arbitrary units)")
                axes[row, col].set_ylabel("t-SNE 2 (arbitrary units)")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"Pre-draft fused features: {len(sample):,} cycles, at most 4 per prompt\n"
                 "Same coordinates recolored; labels never used to construct embeddings", fontsize=13)
    save_figure(fig, out / "feature_geometry")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, key in zip(axes, ("fused_tsne_p15_s0", "fused_tsne_p50_s0", "fused_tsne_p50_s1")):
        ax.scatter(*embeddings[key].T, c=y, cmap="viridis", vmin=0, vmax=15, s=7, alpha=.7, rasterized=True)
        ax.set_title(key.replace("fused_tsne_", "").replace("_", ", "))
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("t-SNE sensitivity: identical rows, different perplexities/seeds")
    save_figure(fig, out / "tsne_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key in zip(axes, ("raw", "fused")):
        a = ax.scatter(*embeddings[key + "_pca"].T, c=y, cmap="viridis", vmin=0, vmax=15, s=7, alpha=.7, rasterized=True)
        ax.set_title(f"{key.title()} pre-draft features")
        ax.set_xlabel(f"PC1 ({100*explained[key][0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*explained[key][1]:.1f}%)")
    fig.colorbar(a, ax=axes, label="Accepted proposed tokens A")
    save_figure(fig, out / "raw_fused_pca")


def results_figures(data, summary, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    y = data["y"]
    for group in ("train", "calibration", "assessment"):
        yy = y[data["group"] == group]
        counts = np.bincount(yy.astype(int), minlength=16)
        axes[0].plot(np.arange(16), counts / len(yy), marker=".", label=f"{group} ({len(yy):,})")
    axes[0].set(xlabel="Accepted proposed tokens A", ylabel="Fraction of cycles", title="B16 acceptance distribution")
    axes[0].legend(fontsize=8)
    variance = summary["temporal"]["variance"]
    vals = [variance["between_prompt_fraction"], variance["within_prompt_fraction"]]
    axes[1].bar(["Between prompts", "Within prompts"], vals, color=["#0072B2", "#E69F00"])
    axes[1].set(ylabel="Fraction of total label variation", ylim=(0, 1), title="Descriptive variance decomposition")
    for i, val in enumerate(vals):
        axes[1].text(i, val + .02, f"{100*val:.1f}%", ha="center")
    lag = summary["temporal"]["lags"]
    xx = np.array(list(map(int, lag)))
    axes[2].plot(xx, [v["raw_correlation"] for v in lag.values()], marker="o", label="Raw acceptance")
    axes[2].plot(xx, [v["within_prompt_centered_correlation"] for v in lag.values()], marker="o", label="After prompt demeaning")
    axes[2].fill_between(xx, [v["within_prompt_shuffle_null_95"][0] for v in lag.values()],
                         [v["within_prompt_shuffle_null_95"][1] for v in lag.values()],
                         alpha=.25, color="gray", label="Within-prompt shuffled reference")
    axes[2].axhline(0, color="gray", linewidth=.6)
    axes[2].set(xlabel="Cycle lag", ylabel="Pearson correlation", title="Does recent acceptance persist?")
    axes[2].legend(fontsize=8)
    save_figure(fig, out / "temporal_structure")
    probes = summary["probes"]
    names = list(probes)
    mae = np.array([probes[n]["mae"] for n in names])
    intervals = np.array([probes[n]["mae_prompt_bootstrap_95"] for n in names])
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    yy = np.arange(len(names))
    ax.barh(yy, mae, color="#0072B2", alpha=.75)
    ax.errorbar(mae, yy, xerr=np.maximum(0, np.stack([mae - intervals[:, 0], intervals[:, 1] - mae])),
                fmt="none", ecolor="#222222", capsize=3)
    ax.set_yticks(yy, names, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Expected-length MAE (tokens; lower is better)",
           title="Diagnostic probes on fixed assessment prompts\nError bars: 95% prompt-bootstrap intervals; no policy/throughput claim")
    ax.set_xlim(0, intervals[:, 1].max() + .45)
    for i, val in enumerate(mae):
        ax.text(intervals[i, 1] + .06, i, f"{val:.3f}", va="center", fontsize=9)
    save_figure(fig, out / "heldout_probes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int, default=927)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output destination; existing evidence is immutable")
    args.output.mkdir(parents=True)
    start = time.time()
    import matplotlib
    matplotlib.use("Agg")
    import scipy
    from scipy import sparse
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler, normalize
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    print("Auditing and reconstructing causally available token windows", flush=True)
    data = load_cache(args.cache, args.prompt_manifest)
    group, y = data["group"], data["y"]
    tr, assessment = group == "train", group == "assessment"
    sample = prompt_balanced_indices(data["prompts"], 4, args.seed)
    summary = {"scope": "Exploratory analysis of completed paired B16 pilot; not historical full validation, not new policy training.",
        "cache": str(args.cache), "seed": args.seed, "audit": data["audit"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": digest(__file__),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "counts": {g: {"rows": int((group == g).sum()), "prompts": len(set(data["prompts"][group == g]))}
                   for g in ("train", "calibration", "assessment")},
        "input_timing": "Text stops at prefix_length-1: excludes current known anchor and all draft/verification future tokens.",
        "plot_sample": {"rows": len(sample), "max_rows_per_prompt": 4, "labels_used_for_selection": False},
        "probes": {}, "probe_selection": {}, "geometry": {},
        "limitations": ["Previously inspected development validation, not a fresh final test.",
            "B16 outcomes do not identify acceptance at different actual drafting blocks.",
            "Token unigram/bigram TF-IDF is a lexical baseline, not a semantic text encoder.",
            "PCA/t-SNE overlap does not prove that acceptance is unpredictable.",
            "t-SNE fitted on a pooled descriptive sample; held-out probes do not use t-SNE.",
            "Prompt demeaning uses full trajectories only for descriptive analysis, never prediction.",
            "Bootstrap intervals are conditional on this fit; not training-seed uncertainty.",
            "PCA uses row-L2-normalized inputs, then train-only centering, no whitening."]}
    predictions = {}

    def record(name, prediction, selection=None):
        predictions[name] = prediction
        summary["probes"][name] = assessment_metrics(y[assessment], prediction[assessment], data["prompts"][assessment], args.seed)
        if selection is not None:
            summary["probe_selection"][name] = selection
        print(name, json.dumps(summary["probes"][name]), flush=True)

    record("Training mean", np.full(len(y), y[tr].mean()))
    categories = sorted(set(data["sources"]))
    meta = np.column_stack([*(data["sources"] == s for s in categories),
                           np.log1p(data["prompt_lens"]), np.log1p(data["starts"]),
                           data["cycles"], data["starts"] - data["prompt_lens"]]).astype(np.float32)
    meta = StandardScaler().fit(meta[tr]).transform(meta)
    prediction, selected = ridge_probe(meta, data)
    record("Source + progress", prediction, selected)
    history = StandardScaler().fit(data["history"][tr]).transform(data["history"])
    prediction, selected = ridge_probe(np.column_stack([meta, history]), data)
    record("Source + progress + prior acceptance", prediction, selected)
    print("Temporal diagnostics", flush=True)
    summary["temporal"] = temporal_analysis(data, args.seed)
    embeddings, pcs, explained = {}, {}, {}
    for kind, file in (("fused", "features"), ("raw", "raw_features")):
        print(f"Train-only PCA64: {kind}", flush=True)
        x = np.concatenate([np.load(args.cache / split / f"{file}.npy")[:, 0] for split in ("train", "val")]).astype(np.float32)
        x = normalize(x, copy=False)
        pca = PCA(n_components=64, svd_solver="randomized", iterated_power=4, random_state=args.seed).fit(x[tr])
        z = pca.transform(x)
        del x
        pcs[kind] = z
        explained[kind] = pca.explained_variance_ratio_.tolist()
        embeddings[kind + "_pca"] = z[sample, :2]
        summary["geometry"][kind] = {"explained_variance_ratio": explained[kind],
                                    "neighbors": cross_prompt_neighbors(z, data, sample, args.seed)}
        scaled = StandardScaler().fit(z[tr]).transform(z)
        prediction, selected = ridge_probe(np.column_stack([meta, scaled]), data)
        record(f"Source + progress + {kind} PCs", prediction, selected)
        if kind == "fused":
            prediction, selected = ridge_probe(np.column_stack([meta, scaled, history]), data)
            record("Source + progress + fused PCs + prior acceptance", prediction, selected)
        knn = KNeighborsRegressor(n_neighbors=50, weights="uniform", n_jobs=4).fit(z[tr], y[tr])
        prediction = np.full(len(y), np.nan)
        prediction[~tr] = knn.predict(z[~tr])
        record(f"{kind.title()} PCA64 cross-prompt kNN (k=50)", prediction)
    for kind in ("prompt", "recent16", "recent64", "recent256"):
        print("Lexical probe:", kind, flush=True)
        docs = data["documents"][kind]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\d+\b", ngram_range=(1, 2),
                                    min_df=3, max_features=20000, sublinear_tf=True, dtype=np.float32)
        # Avoid repeating identical original prompts while estimating document frequencies.
        fit_docs = [docs[i] for i in np.flatnonzero(tr)]
        if kind == "prompt":
            fit_docs = list(dict.fromkeys(fit_docs))
        vectorizer.fit(fit_docs)
        x = vectorizer.transform(docs)
        features = sparse.hstack([sparse.csr_matrix(meta), x], format="csr")
        prediction, selected = ridge_probe(features, data)
        selected["vocabulary_size"] = len(vectorizer.vocabulary_)
        label = "Original prompt text" if kind == "prompt" else kind.replace("recent", "Recent ") + " tokens"
        record("Source + progress + " + label, prediction, selected)
        if kind == "recent64":
            features = sparse.hstack([features, sparse.csr_matrix(history)], format="csr")
            prediction, selected = ridge_probe(features, data)
            record("Source + progress + recent64 + prior acceptance", prediction, selected)
    # Paired prompt bootstrap of gains over source/progress; negative means lower MAE.
    baseline_errors = np.abs(y[assessment] - predictions["Source + progress"][assessment])
    assess_groups = grouped_indices(data["prompts"][assessment])
    counts = np.array([len(i) for i in assess_groups])
    draws = np.random.default_rng(args.seed).integers(len(counts), size=(1000, len(counts)))
    for name, prediction in predictions.items():
        delta = np.abs(y[assessment] - prediction[assessment]) - baseline_errors
        sums = np.array([delta[i].sum() for i in assess_groups])
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        summary["probes"][name]["mae_delta_vs_source_progress"] = float(delta.mean())
        summary["probes"][name]["paired_delta_prompt_bootstrap_95"] = np.quantile(boot, [.025, .975]).tolist()
    write_json(args.output / "summary.json", summary)
    np.savez_compressed(args.output / "probe_predictions.npz", y=y, prompt_id=data["prompts"],
                        group=group, **{f"pred_{i}": p for i, p in enumerate(predictions.values())})
    for perplexity, seed in ((15, 0), (50, 0), (50, 1)):
        print(f"Descriptive t-SNE: perplexity={perplexity}, seed={seed}", flush=True)
        key = f"fused_tsne_p{perplexity}_s{seed}"
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                    max_iter=1000, random_state=seed, n_jobs=4)
        embeddings[key] = tsne.fit_transform(pcs["fused"][sample, :50])
        summary["geometry"][key] = {"kl_divergence": float(tsne.kl_divergence_)}
    np.savez_compressed(args.output / "plot_coordinates.npz", rows=sample, accepted_len=y[sample],
                        prompt_id=data["prompts"][sample], source=data["sources"][sample],
                        cycle_id=data["cycles"][sample], group=group[sample], **embeddings)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        with (args.output / "plot_rows.jsonl").open("w") as f:
            for i in sample:
                f.write(json.dumps({"row": int(i), "prompt_id": int(data["prompts"][i]),
                    "cycle": int(data["cycles"][i]), "source": str(data["sources"][i]),
                    "group": str(group[i]), "accepted_len": int(y[i]),
                    "recent_committed_text_64": tokenizer.decode(data["snippets"][int(i)])}) + "\n")
    geometry_figure(embeddings, data, sample, explained, args.output)
    results_figures(data, summary, args.output)
    summary["elapsed_seconds"] = time.time() - start
    write_json(args.output / "summary.json", summary)
    lines = ["# Pre-draft acceptance geometry and recent-text analysis", "", summary["scope"], "",
             "## Fixed prompt-level partitions", "", json.dumps(summary["counts"], indent=2), "",
             "## Assessment probes", "", "MAE is prediction error, not a policy budget or throughput metric.", "",
             "| Diagnostic | MAE | R² | Within-prompt R² (descriptive) |", "|---|---:|---:|---:|"]
    for name, stats in summary["probes"].items():
        lines.append(f"| {name} | {stats['mae']:.4f} | {stats['r2']:.4f} | {stats['within_prompt_r2_descriptive']:.4f} |")
    lines += ["", "## Figures", "", "![Feature geometry](feature_geometry.png)", "",
              "![Temporal structure](temporal_structure.png)", "", "![Assessment probes](heldout_probes.png)", "",
              "![t-SNE sensitivity](tsne_sensitivity.png)", "", "![Raw versus fused PCA](raw_fused_pca.png)", "",
              "## Interpretation limits", ""] + ["- " + x for x in summary["limitations"]]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(args.output / "COMPLETE.json", {"summary_sha256": digest(args.output / "summary.json"),
               "elapsed_seconds": summary["elapsed_seconds"]})
    print("COMPLETE", args.output, summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
