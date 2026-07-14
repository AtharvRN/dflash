from __future__ import annotations

import argparse
import bisect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from train_dflashv2_horizon_predictor import _make_splits


@dataclass
class IdShard:
    path: Path
    trace_idx: int
    rows: int
    prompt_index: np.ndarray
    cycle_id: np.ndarray


def _load_id_shards(trace_dirs: list[Path]) -> list[IdShard]:
    shards: list[IdShard] = []
    for trace_idx, trace_dir in enumerate(trace_dirs):
        manifest_path = trace_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing trace manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["shards"]:
            rows = int(item["rows"])
            if rows <= 0:
                continue
            shard_dir = trace_dir / item["path"]
            shards.append(
                IdShard(
                    path=shard_dir,
                    trace_idx=trace_idx,
                    rows=rows,
                    prompt_index=np.load(shard_dir / "prompt_index.npy", mmap_mode="r"),
                    cycle_id=np.load(shard_dir / "cycle_id.npy", mmap_mode="r"),
                )
            )
    if not shards:
        raise ValueError("no non-empty shards found")
    return shards


def _build_offsets(shards: list[IdShard]) -> tuple[list[int], int]:
    offsets: list[int] = []
    total = 0
    for shard in shards:
        offsets.append(total)
        total += int(shard.rows)
    return offsets, total


def _write_split_ids(
    *,
    split_name: str,
    cache_dir: Path,
    shards: list[IdShard],
    offsets: list[int],
    indices: np.ndarray,
    sequence_multiplier: int,
) -> dict[str, Any]:
    split_dir = cache_dir / split_name
    if not (split_dir / "features.npy").exists():
        raise FileNotFoundError(f"missing compact split: {split_dir}")
    rows = int(np.load(split_dir / "accepted_len.npy", mmap_mode="r").shape[0])
    order = np.sort(np.asarray(indices, dtype=np.int64))
    if rows != int(order.shape[0]):
        raise ValueError(f"{split_dir} has {rows} rows but split indices have {order.shape[0]}")

    sequence_id = np.lib.format.open_memmap(
        split_dir / "sequence_id.npy",
        mode="w+",
        dtype=np.int64,
        shape=(rows,),
    )
    cycle_id = np.lib.format.open_memmap(
        split_dir / "cycle_id.npy",
        mode="w+",
        dtype=np.int32,
        shape=(rows,),
    )
    global_index = np.lib.format.open_memmap(
        split_dir / "global_index.npy",
        mode="w+",
        dtype=np.int64,
        shape=(rows,),
    )

    started = time.time()
    cursor = 0
    for shard_idx, shard in enumerate(shards):
        shard_start = offsets[shard_idx]
        shard_end = shard_start + int(shard.rows)
        left = np.searchsorted(order, shard_start, side="left")
        right = np.searchsorted(order, shard_end, side="left")
        if right <= left:
            continue
        local = (order[left:right] - shard_start).astype(np.int64, copy=False)
        prompts = np.asarray(shard.prompt_index[local], dtype=np.int64)
        sequence_id[left:right] = int(shard.trace_idx) * int(sequence_multiplier) + prompts
        cycle_id[left:right] = np.asarray(shard.cycle_id[local], dtype=np.int32)
        global_index[left:right] = order[left:right]
        cursor = right
        print(
            json.dumps(
                {
                    "event": "ids_shard_done",
                    "split": split_name,
                    "shard": str(shard.path),
                    "rows_done": int(cursor),
                    "rows_total": rows,
                    "elapsed_s": round(time.time() - started, 3),
                }
            ),
            flush=True,
        )

    sequence_id.flush()
    cycle_id.flush()
    global_index.flush()
    meta = {
        "split": split_name,
        "rows": rows,
        "sequence_multiplier": int(sequence_multiplier),
        "elapsed_s": round(time.time() - started, 3),
    }
    (split_dir / "sequence_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "ids_split_done", **meta}), flush=True)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach prompt/cycle IDs to an existing compact DFlashv2 cache.")
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--compact-cache-dir", type=Path, required=True)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-multiplier", type=int, default=1_000_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps({"event": "load_id_shards_start"}), flush=True)
    shards = _load_id_shards(args.trace_dir)
    offsets, total_rows = _build_offsets(shards)
    print(json.dumps({"event": "load_id_shards_done", "total_rows": total_rows, "num_shards": len(shards)}), flush=True)
    train_indices, val_indices = _make_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    train_meta = _write_split_ids(
        split_name="train",
        cache_dir=args.compact_cache_dir,
        shards=shards,
        offsets=offsets,
        indices=train_indices,
        sequence_multiplier=args.sequence_multiplier,
    )
    val_meta = _write_split_ids(
        split_name="val",
        cache_dir=args.compact_cache_dir,
        shards=shards,
        offsets=offsets,
        indices=val_indices,
        sequence_multiplier=args.sequence_multiplier,
    )
    meta = {
        "trace_dirs": [str(path) for path in args.trace_dir],
        "compact_cache_dir": str(args.compact_cache_dir),
        "total_rows": total_rows,
        "train": train_meta,
        "val": val_meta,
        "seed": args.seed,
        "calibration_rows": args.calibration_rows,
        "max_total_rows": args.max_total_rows,
        "sequence_multiplier": args.sequence_multiplier,
    }
    (args.compact_cache_dir / "sequence_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"event": "ids_done", **meta}), flush=True)


if __name__ == "__main__":
    main()
