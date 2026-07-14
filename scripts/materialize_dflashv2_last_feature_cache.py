from __future__ import annotations

import argparse
import json
import shutil
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
    max_read_rows: int,
    max_output_rows: int,
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
        rel_idx = 0
        while rel_idx < int(local_indices.shape[0]):
            chunk_start = rel_idx
            first_local = int(local_indices[chunk_start])
            rel_idx += 1
            while rel_idx < int(local_indices.shape[0]):
                next_local = int(local_indices[rel_idx])
                if rel_idx - chunk_start >= max_output_rows:
                    break
                if next_local - first_local >= max_read_rows:
                    break
                rel_idx += 1

            chunk = local_indices[chunk_start:rel_idx].astype(np.int64, copy=False)
            read_start = int(chunk[0])
            read_end = int(chunk[-1]) + 1
            positions = chunk - read_start
            out_start = left + chunk_start
            out_end = left + rel_idx

            raw_mask = np.asarray(shard.mask[read_start:read_end], dtype=np.float32)
            valid_mask = raw_mask > 0.5
            has_valid = valid_mask.any(axis=1)
            last_valid = valid_mask.shape[1] - 1 - np.argmax(valid_mask[:, ::-1], axis=1)
            last_valid = np.where(has_valid, last_valid, valid_mask.shape[1] - 1)
            selected_last = last_valid[positions]

            feature_block = np.asarray(shard.features[read_start:read_end], dtype=np.float16)
            features[out_start:out_end, 0] = feature_block[positions, selected_last]
            mask[out_start:out_end, 0] = has_valid[positions].astype(np.float32)
            survival[out_start:out_end] = np.asarray(shard.survival[chunk], dtype=np.float32)
            accepted[out_start:out_end] = np.asarray(shard.accepted_len[chunk], dtype=np.float32)

            processed = rel_idx
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


def _copy_compact_subset(
    *,
    name: str,
    source_dir: Path,
    output_dir: Path,
    positions: np.ndarray,
    copy_rows: int,
) -> dict[str, Any]:
    split_dir = output_dir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=np.int64)
    src_features = np.load(source_dir / "features.npy", mmap_mode="r")
    src_mask = np.load(source_dir / "mask.npy", mmap_mode="r")
    src_survival = np.load(source_dir / "survival.npy", mmap_mode="r")
    src_accepted = np.load(source_dir / "accepted_len.npy", mmap_mode="r")

    features = np.lib.format.open_memmap(
        split_dir / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(positions.shape[0], src_features.shape[1], src_features.shape[2]),
    )
    mask = np.lib.format.open_memmap(
        split_dir / "mask.npy",
        mode="w+",
        dtype=np.float32,
        shape=(positions.shape[0], src_mask.shape[1]),
    )
    survival = np.lib.format.open_memmap(
        split_dir / "survival.npy",
        mode="w+",
        dtype=np.float32,
        shape=(positions.shape[0], src_survival.shape[1]),
    )
    accepted = np.lib.format.open_memmap(
        split_dir / "accepted_len.npy",
        mode="w+",
        dtype=np.float32,
        shape=(positions.shape[0],),
    )

    started = time.time()
    for start in range(0, int(positions.shape[0]), copy_rows):
        end = min(start + copy_rows, int(positions.shape[0]))
        chunk = positions[start:end]
        features[start:end] = src_features[chunk]
        mask[start:end] = src_mask[chunk]
        survival[start:end] = src_survival[chunk]
        accepted[start:end] = src_accepted[chunk]
        print(
            json.dumps(
                {
                    "event": "copy_split_progress",
                    "split": name,
                    "rows_done": end,
                    "rows_total": int(positions.shape[0]),
                    "elapsed_s": round(time.time() - started, 3),
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
        "rows": int(positions.shape[0]),
        "elapsed_s": round(time.time() - started, 3),
        "source_dir": str(source_dir),
    }
    _write_json(split_dir / "meta.json", meta)
    print(json.dumps({"event": "copy_split_done", **meta}), flush=True)
    return meta


def _make_materializer_splits(
    *,
    total_rows: int,
    max_total_rows: int | None,
    calibration_rows: int,
    seed: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if selection_mode == "random":
        return _make_splits(
            total_rows=total_rows,
            max_total_rows=max_total_rows,
            calibration_rows=calibration_rows,
            seed=seed,
        )
    if selection_mode != "prefix":
        raise ValueError(f"unsupported selection_mode={selection_mode!r}")
    use_rows = total_rows if max_total_rows is None else min(total_rows, max_total_rows)
    if calibration_rows >= use_rows:
        raise ValueError(f"calibration_rows={calibration_rows} must be smaller than selected rows={use_rows}")
    selected = np.arange(use_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    return selected[calibration_rows:], selected[:calibration_rows]


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
    parser.add_argument("--max-read-rows", type=int, default=1024)
    parser.add_argument("--max-output-rows", type=int, default=1024)
    parser.add_argument("--copy-rows", type=int, default=65536)
    parser.add_argument("--selection-mode", choices=("random", "prefix"), default="random")
    parser.add_argument("--keep-all-cache", action="store_true")
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

    train_indices, val_indices = _make_materializer_splits(
        total_rows=total_rows,
        max_total_rows=args.max_total_rows,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
        selection_mode=args.selection_mode,
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
        "selection_mode": args.selection_mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(args.output_dir / "meta.json", run_meta)
    print(json.dumps({"event": "materialize_start", **run_meta}), flush=True)

    combined_indices = np.sort(np.concatenate([train_indices, val_indices]).astype(np.int64, copy=False))
    all_dir = args.output_dir / "_all"
    all_meta = _materialize_split(
        name="_all",
        shards=shards,
        offsets=offsets,
        indices=combined_indices,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
        max_read_rows=args.max_read_rows,
        max_output_rows=args.max_output_rows,
    )

    rank = np.empty((total_rows,), dtype=np.int64)
    rank[combined_indices] = np.arange(combined_indices.shape[0], dtype=np.int64)
    train_positions = rank[np.sort(train_indices)]
    val_positions = rank[np.sort(val_indices)]
    train_meta = _copy_compact_subset(
        name="train",
        source_dir=all_dir,
        output_dir=args.output_dir,
        positions=train_positions,
        copy_rows=args.copy_rows,
    )
    val_meta = _copy_compact_subset(
        name="val",
        source_dir=all_dir,
        output_dir=args.output_dir,
        positions=val_positions,
        copy_rows=args.copy_rows,
    )
    if not args.keep_all_cache:
        shutil.rmtree(all_dir)
        print(json.dumps({"event": "removed_temporary_all_cache", "path": str(all_dir)}), flush=True)
    run_meta.update(
        {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "all": all_meta,
            "train": train_meta,
            "val": val_meta,
        }
    )
    _write_json(args.output_dir / "meta.json", run_meta)
    print(json.dumps({"event": "materialize_done", **run_meta}), flush=True)


if __name__ == "__main__":
    main()
