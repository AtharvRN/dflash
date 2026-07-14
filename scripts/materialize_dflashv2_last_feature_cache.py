from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from train_dflashv2_horizon_predictor import (
    _compact_cache_ready,
    _load_shards_multi,
    _make_splits,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _build_offsets(shards: list[Any]) -> tuple[list[int], int]:
    offsets: list[int] = []
    total = 0
    for shard in shards:
        offsets.append(total)
        total += int(shard.rows)
    return offsets, total


def _materialize_split(
    *,
    name: str,
    shards: list[Any],
    offsets: list[int],
    indices: np.ndarray,
    output_dir: Path,
    progress_every: int,
) -> dict[str, Any]:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)

    order = np.sort(np.asarray(indices, dtype=np.int64))
    input_dim = int(shards[0].features.shape[2])
    num_slots = int(shards[0].survival.shape[1])

    features = np.lib.format.open_memmap(
        split_dir / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(order.shape[0], 1, input_dim),
    )
    mask = np.lib.format.open_memmap(
        split_dir / "mask.npy",
        mode="w+",
        dtype=np.float32,
        shape=(order.shape[0], 1),
    )
    survival = np.lib.format.open_memmap(
        split_dir / "survival.npy",
        mode="w+",
        dtype=np.float32,
        shape=(order.shape[0], num_slots),
    )
    accepted = np.lib.format.open_memmap(
        split_dir / "accepted_len.npy",
        mode="w+",
        dtype=np.float32,
        shape=(order.shape[0],),
    )

    started = time.time()
    cursor = 0
    shard_rows: list[dict[str, Any]] = []
    for shard_idx, shard in enumerate(shards):
        shard_start = offsets[shard_idx]
        shard_end = shard_start + int(shard.rows)
        left = np.searchsorted(order, shard_start, side="left")
        right = np.searchsorted(order, shard_end, side="left")
        if right <= left:
            continue

        shard_started = time.time()
        local_indices = order[left:right] - shard_start
        if cursor != left:
            raise RuntimeError(f"internal cursor mismatch for split={name}: cursor={cursor}, left={left}")
        for rel_idx, local_idx64 in enumerate(local_indices):
            out_idx = left + rel_idx
            local_idx = int(local_idx64)
            raw_mask = np.asarray(shard.mask[local_idx], dtype=np.float32)
            valid = np.flatnonzero(raw_mask > 0.5)
            feature_idx = int(valid[-1]) if valid.size else int(raw_mask.shape[0] - 1)
            features[out_idx, 0] = np.asarray(shard.features[local_idx, feature_idx], dtype=np.float16)
            mask[out_idx, 0] = 1.0 if valid.size else 0.0
            survival[out_idx] = np.asarray(shard.survival[local_idx], dtype=np.float32)
            accepted[out_idx] = float(shard.accepted_len[local_idx])

            processed = rel_idx + 1
            if progress_every > 0 and processed % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "materialize_progress",
                            "split": name,
                            "shard": str(shard.path),
                            "shard_index": shard_idx,
                            "split_rows_done": int(out_idx + 1),
                            "split_rows_total": int(order.shape[0]),
                            "shard_rows_done": int(processed),
                            "shard_rows_total": int(local_indices.shape[0]),
                            "elapsed_s": round(time.time() - started, 3),
                        }
                    ),
                    flush=True,
                )
        cursor = right
        item = {
            "shard": str(shard.path),
            "shard_index": shard_idx,
            "rows": int(local_indices.shape[0]),
            "elapsed_s": round(time.time() - shard_started, 3),
        }
        shard_rows.append(item)
        print(
            json.dumps(
                {
                    "event": "materialize_shard_done",
                    "split": name,
                    "split_rows_done": int(cursor),
                    "split_rows_total": int(order.shape[0]),
                    **item,
                }
            ),
            flush=True,
        )

    features.flush()
    mask.flush()
    survival.flush()
    accepted.flush()
    meta = {
        "split": name,
        "rows": int(order.shape[0]),
        "input_dim": input_dim,
        "num_slots": num_slots,
        "elapsed_s": round(time.time() - started, 3),
        "shards": shard_rows,
    }
    _write_json(split_dir / "meta.json", meta)
    print(json.dumps({"event": "materialize_split_done", **meta}), flush=True)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize compact last-fused-vector DFlashv2 horizon features. "
            "This turns large (rows, window, hidden) traces into (rows, 1, hidden) "
            "train/val caches for fast policy sweeps."
        )
    )
    parser.add_argument("--trace-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not args.overwrite
        and _compact_cache_ready(args.output_dir / "train")
        and _compact_cache_ready(args.output_dir / "val")
    ):
        print(json.dumps({"event": "compact_cache_ready", "output_dir": str(args.output_dir)}), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "load_shards_start",
                "trace_dirs": [str(path) for path in args.trace_dir],
            }
        ),
        flush=True,
    )
    shards = _load_shards_multi(args.trace_dir)
    offsets, total_rows = _build_offsets(shards)
    print(
        json.dumps(
            {
                "event": "load_shards_done",
                "total_rows": total_rows,
                "num_shards": len(shards),
            }
        ),
        flush=True,
    )

    train_indices, val_indices = _make_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
    )
    run_meta: dict[str, Any] = {
        "trace_dirs": [str(path) for path in args.trace_dir],
        "output_dir": str(args.output_dir),
        "total_rows": int(total_rows),
        "selected_rows": int(train_indices.shape[0] + val_indices.shape[0]),
        "train_rows": int(train_indices.shape[0]),
        "val_rows": int(val_indices.shape[0]),
        "max_total_rows": args.max_total_rows,
        "calibration_rows": args.calibration_rows,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(args.output_dir / "meta.json", run_meta)
    print(json.dumps({"event": "materialize_start", **run_meta}), flush=True)

    train_meta = _materialize_split(
        name="train",
        shards=shards,
        offsets=offsets,
        indices=train_indices,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    val_meta = _materialize_split(
        name="val",
        shards=shards,
        offsets=offsets,
        indices=val_indices,
        output_dir=args.output_dir,
        progress_every=max(args.progress_every, 0),
    )
    run_meta.update(
        {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "train": train_meta,
            "val": val_meta,
        }
    )
    _write_json(args.output_dir / "meta.json", run_meta)
    print(json.dumps({"event": "materialize_done", **run_meta}), flush=True)


if __name__ == "__main__":
    main()
