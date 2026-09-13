"""Stage original W-token traces with strict canonical prompt splits and row provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def prompt_ids(path, key):
    values = json.loads(path.read_text())[key]
    result = {int(x) for x in values}
    if len(result) != len(values):
        raise ValueError(f"Duplicate prompt IDs: {path}")
    return result


def scan_shard(task):
    index, path = task
    ids, cycles, labels = [], [], []
    with (path / "metadata.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("manifest_index") is None:
                raise ValueError(f"Missing global manifest_index: {path}")
            ids.append(int(row["manifest_index"]))
            cycles.append(int(row["cycle_id"]))
            labels.append(int(row["accepted_draft_len"]))
    n = len(ids)
    data = np.empty((n, 5), dtype=np.int64)
    data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4] = index, np.arange(n), ids, cycles, labels
    return data


def split_rows(rows, train_ids, val_ids, max_train_rows, seed):
    if train_ids & val_ids:
        raise ValueError("Canonical training and validation prompts overlap")
    train_mask = np.isin(rows[:, 2], list(train_ids))
    val_mask = np.isin(rows[:, 2], list(val_ids))
    if not (train_mask | val_mask).all():
        raise ValueError("Trace contains prompt IDs outside the canonical split")
    train = rows[train_mask]
    if max_train_rows is not None and len(train) > max_train_rows:
        selected = np.sort(np.random.default_rng(seed).choice(len(train), max_train_rows, replace=False))
        train = train[selected]
    return train, rows[val_mask]


def materialize(paths, rows, output, *, window, width, workers):
    output.mkdir()
    n = len(rows)
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float16,
                                        shape=(n, window, width))
    masks = np.lib.format.open_memmap(output / "mask.npy", mode="w+", dtype=np.uint8, shape=(n, window))
    labels = np.lib.format.open_memmap(output / "accepted_len.npy", mode="w+", dtype=np.int64, shape=(n,))
    np.save(output / "row_index.npy", rows)
    tasks = [(int(s), np.flatnonzero(rows[:, 0] == s)) for s in np.unique(rows[:, 0])]

    def copy_selected(task):
        shard, dest = task
        source = rows[dest, 1]
        path = paths[shard]
        x = np.load(path / "features.npy", mmap_mode="r")
        m = np.load(path / "mask.npy", mmap_mode="r")
        y = np.load(path / "accepted_len.npy", mmap_mode="r")
        if x.ndim != 3 or x.shape[1:] != (window, width) or m.shape != x.shape[:2]:
            raise ValueError(f"Incompatible feature/mask shape: {path}")
        if source.max() >= min(len(x), len(y)):
            raise ValueError(f"Metadata exceeds allocated rows: {path}")
        selected_x, selected_m, selected_y = np.asarray(x[source]), np.asarray(m[source]), np.asarray(y[source])
        if not np.isfinite(selected_x).all() or not np.isin(selected_m, [0, 1]).all() or not selected_m.any(1).all():
            raise ValueError(f"Invalid features or masks: {path}")
        if not np.array_equal(selected_y, rows[dest, 4]) or not np.isin(selected_y, np.arange(16)).all():
            raise ValueError(f"Label/metadata disagreement: {path}")
        # Compare stored survival labels as a separate off-by-one/alignment check.
        survival = np.load(path / "survival.npy", mmap_mode="r")[source]
        if not np.array_equal(survival, selected_y[:, None] >= np.arange(1, 16)):
            raise ValueError(f"Survival labels do not encode accepted draft length: {path}")
        features[dest], masks[dest], labels[dest] = selected_x, selected_m, selected_y
        return len(dest)

    copied = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, count in enumerate(pool.map(copy_selected, tasks)):
            copied += count
            if (i + 1) % 32 == 0 or i + 1 == len(tasks):
                print(json.dumps({"split": output.name, "shards": i+1, "rows": copied, "total_rows": n}), flush=True)
    for array in (features, masks, labels):
        array.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-manifest", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-train-rows", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=913)
    args = p.parse_args()
    if args.max_train_rows < 1 or args.workers < 1:
        raise ValueError("Positive sample/worker counts required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads(args.trace_manifest.read_text())
    paths = [Path(s["path"]) if Path(s["path"]).is_absolute() else args.trace_manifest.parent / s["path"]
             for s in manifest["shards"]]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate shard paths")
    train_ids = prompt_ids(args.split_dir / "train_prompt_ids.json", "train_prompt_ids")
    val_ids = prompt_ids(args.split_dir / "val_prompt_ids.json", "val_prompt_ids")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        chunks = []
        for i, chunk in enumerate(pool.map(scan_shard, enumerate(paths))):
            chunks.append(chunk)
            if (i+1) % 64 == 0:
                print(json.dumps({"metadata_shards": i+1, "elapsed_s": time.monotonic()-started}), flush=True)
    rows = np.concatenate(chunks)
    if len(np.unique(rows[:, 2:4], axis=0)) != len(rows):
        raise ValueError("Duplicate (manifest_index, cycle_id) states")
    train, val = split_rows(rows, train_ids, val_ids, args.max_train_rows, args.seed)
    canonical = json.loads((args.split_dir / "manifest.json").read_text())
    if len(rows) != canonical["num_rows_total"] or len(val) != canonical["val"]["rows"]:
        raise ValueError("Metadata counts differ from canonical split")
    first = np.load(paths[0] / "features.npy", mmap_mode="r")
    window, width = first.shape[1:]
    if width != 2560 or window != 16:
        raise ValueError("This pilot expects original Qwen3-4B fused W16 traces")
    for name, selected in (("train", train), ("val", val)):
        materialize(paths, selected, args.output_dir / name, window=window, width=width, workers=args.workers)
    # Preserve the canonical validation set; partition its prompts only for calibration vs assessment.
    observed_val = np.unique(val[:, 2])
    shuffled = np.random.default_rng(args.seed+1).permutation(observed_val)
    calibration_ids = shuffled[:max(1, len(shuffled)//5)]
    np.save(args.output_dir / "val" / "calibration.npy", np.isin(val[:, 2], calibration_ids))
    np.save(args.output_dir / "source_rows.npy", rows)
    info = {"format": "dflash_context_attention_cache_v1", "input_kind": "predraft_fused",
            "anchor_available": False, "context_window": window, "input_dim": width, "num_slots": 15,
            "train_rows": len(train), "val_rows": len(val), "train_prompts": len(np.unique(train[:, 2])),
            "val_prompts": len(observed_val), "calibration_prompt_ids": calibration_ids.tolist(),
            "seed": args.seed, "source_shards": [str(x) for x in paths],
            "trace_manifest": str(args.trace_manifest), "split_dir": str(args.split_dir),
            "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       (args.trace_manifest, args.split_dir/"train_prompt_ids.json", args.split_dir/"val_prompt_ids.json")},
            "elapsed_s": time.monotonic()-started}
    atomic_json(args.output_dir / "manifest.json", info)
    print(json.dumps({k:v for k,v in info.items() if k not in ("source_shards", "calibration_prompt_ids")}), flush=True)


if __name__ == "__main__":
    main()
