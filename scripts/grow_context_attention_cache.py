"""Stage a larger training subset while reusing the exact existing validation cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_context_attention_cache import atomic_json, materialize, prompt_ids, split_rows, validate_context_masks, validate_feature_kind
from scripts.train_context_attention import buffered_backup


def grow_cache(parent, output, train_rows, workers, seed):
    info = json.loads((parent / "manifest.json").read_text())
    if info["format"] != "dflash_context_attention_cache_v1" or info["input_kind"] != "predraft_fused":
        raise ValueError("Expected an existing verified pre-draft context cache")
    for path, expected in info["hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source provenance changed: {path}")
    source = json.loads(Path(info["trace_manifest"]).read_text())
    validate_feature_kind(source)
    split = Path(info["split_dir"])
    canonical = json.loads((split / "manifest.json").read_text())
    if source["source_trace_dir"] != canonical["trace_dir"]:
        raise ValueError("Canonical trace source changed")
    train_ids = prompt_ids(split / "train_prompt_ids.json", "train_prompt_ids")
    val_ids = prompt_ids(split / "val_prompt_ids.json", "val_prompt_ids")
    rows = np.load(parent / "source_rows.npy")
    if len(rows) != canonical["num_rows_total"]:
        raise ValueError("Cached source row count differs from canonical split")
    train, val = split_rows(rows, train_ids, val_ids, train_rows, seed)
    if not np.array_equal(val, np.load(parent / "val" / "row_index.npy")):
        raise ValueError("Validation row membership or ordering changed")
    if len(val) != canonical["val"]["rows"] or len(train) != train_rows:
        raise ValueError("Requested split sizes unavailable")
    calibration = np.load(parent / "val" / "calibration.npy")
    if not np.array_equal(calibration, np.isin(val[:, 2], info["calibration_prompt_ids"])):
        raise ValueError("Calibration prompt membership changed")
    validate_context_masks(np.load(parent / "val" / "mask.npy"))
    output.mkdir(parents=True, exist_ok=False)
    materialize([Path(p) for p in info["source_shards"]], train, output / "train",
                window=info["context_window"], width=info["input_dim"], workers=workers)
    for relative in [Path("source_rows.npy"), *[Path("val") / p.name for p in (parent/"val").glob("*.npy")]]:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (parent/relative).stat().st_dev == destination.parent.stat().st_dev:
            os.link(parent/relative, destination)
        else:
            buffered_backup(parent/relative, destination)
    info.update({"train_rows": len(train), "train_prompts": len(np.unique(train[:, 2])),
                 "seed": seed, "reused_validation_cache": str(parent),
                 "validation_calibration_seed": info.get("validation_calibration_seed", info["seed"]+1)})
    atomic_json(output / "manifest.json", info)
    print(json.dumps({k:info[k] for k in ("train_rows", "train_prompts", "val_rows", "val_prompts", "reused_validation_cache")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-rows", type=int, default=100000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=913)
    args = p.parse_args()
    if args.train_rows < 1 or args.workers < 1:
        raise ValueError("Positive row and worker counts required")
    grow_cache(args.parent_cache, args.output_dir, args.train_rows, args.workers, args.seed)


if __name__ == "__main__":
    main()
