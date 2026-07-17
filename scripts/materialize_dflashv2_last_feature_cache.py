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
                            "split_rows_done": int(left + processed),
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


def _metadata_prompt_key(row: dict[str, Any], prompt_key: str) -> str:
    value = row.get(prompt_key)
    if value is None:
        value = row.get("manifest_index")
    if value is None:
        value = row.get("source_id")
    if value is None:
        value = row.get("prompt_index")
    if value is None:
        raise KeyError(f"metadata row has no prompt key {prompt_key!r}, manifest_index, source_id, or prompt_index")
    return str(value)


def _load_or_create_val_prompts(
    *,
    prompt_ids: list[str],
    path: Path,
    val_prompt_count: int,
    seed: int,
) -> set[str]:
    if path.exists():
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            values = payload.get("val_prompt_ids")
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError(f"{path} must contain a list or a dict with val_prompt_ids")
        return {str(item) for item in values}

    unique = sorted(set(prompt_ids))
    if not unique:
        raise ValueError("cannot create validation prompt split from empty prompt set")
    count = min(int(val_prompt_count), len(unique))
    rng = np.random.default_rng(seed)
    chosen = sorted(str(unique[int(idx)]) for idx in rng.choice(len(unique), size=count, replace=False))
    payload = {
        "format": "dflashv2_fixed_val_prompt_ids_v1",
        "seed": int(seed),
        "val_prompt_count_requested": int(val_prompt_count),
        "num_observed_prompts": int(len(unique)),
        "num_val_prompts": int(len(chosen)),
        "val_prompt_ids": chosen,
    }
    _write_json(path, payload)
    return set(chosen)


def _make_prompt_splits_from_metadata(
    *,
    shards: list[Any],
    offsets: list[int],
    prompt_key: str,
    val_prompt_ids_path: Path,
    val_prompt_count: int,
    seed: int,
    max_total_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    row_indices: list[int] = []
    prompt_ids: list[str] = []
    missing_metadata: list[str] = []
    for shard_idx, shard in enumerate(shards):
        metadata_path = shard.path / "metadata.jsonl"
        if not metadata_path.exists():
            missing_metadata.append(str(metadata_path))
            continue
        shard_start = offsets[shard_idx]
        with metadata_path.open() as f:
            for local_idx, line in enumerate(f):
                if local_idx >= int(shard.rows):
                    break
                row = json.loads(line)
                prompt_id = _metadata_prompt_key(row, prompt_key)
                row_indices.append(shard_start + local_idx)
                prompt_ids.append(prompt_id)

    if missing_metadata:
        raise FileNotFoundError(
            "prompt split requires metadata.jsonl for every shard; missing: "
            + ", ".join(missing_metadata[:5])
        )
    if not row_indices:
        raise ValueError("no rows found while reading metadata for prompt split")

    val_prompt_ids = _load_or_create_val_prompts(
        prompt_ids=prompt_ids,
        path=val_prompt_ids_path,
        val_prompt_count=val_prompt_count,
        seed=seed,
    )
    train: list[int] = []
    val: list[int] = []
    for idx, prompt_id in zip(row_indices, prompt_ids, strict=True):
        if prompt_id in val_prompt_ids:
            val.append(idx)
        else:
            train.append(idx)

    rng = np.random.default_rng(seed)
    train_arr = np.asarray(train, dtype=np.int64)
    if max_total_rows is not None and train_arr.shape[0] + len(val) > max_total_rows:
        train_budget = max(0, int(max_total_rows) - len(val))
        if train_budget < train_arr.shape[0]:
            train_arr = rng.choice(train_arr, size=train_budget, replace=False)
    val_arr = np.asarray(val, dtype=np.int64)
    meta = {
        "split_by_prompt_metadata": True,
        "prompt_key": prompt_key,
        "val_prompt_ids_path": str(val_prompt_ids_path),
        "num_observed_rows": int(len(row_indices)),
        "num_observed_prompts": int(len(set(prompt_ids))),
        "num_val_prompts_configured": int(len(val_prompt_ids)),
        "num_val_prompts_observed": int(len(set(prompt_ids).intersection(val_prompt_ids))),
        "train_rows": int(train_arr.shape[0]),
        "val_rows": int(val_arr.shape[0]),
    }
    return train_arr, val_arr, meta


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
    parser.add_argument(
        "--split-by-prompt-metadata",
        action="store_true",
        help="Split train/val by prompt id from shard metadata.jsonl instead of random rows.",
    )
    parser.add_argument(
        "--prompt-key",
        default="manifest_index",
        help="Metadata key used for fixed prompt split; falls back to manifest_index/source_id/prompt_index.",
    )
    parser.add_argument(
        "--val-prompt-count",
        type=int,
        default=5000,
        help="Number of observed prompts to assign to validation when creating a new prompt-id file.",
    )
    parser.add_argument(
        "--val-prompt-ids-path",
        type=Path,
        default=None,
        help="JSON file containing fixed validation prompt ids. Created if missing.",
    )
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

    prompt_split_meta: dict[str, Any] = {}
    if args.split_by_prompt_metadata:
        val_prompt_ids_path = args.val_prompt_ids_path or (args.output_dir / "val_prompt_ids.json")
        train_indices, val_indices, prompt_split_meta = _make_prompt_splits_from_metadata(
            shards=shards,
            offsets=offsets,
            prompt_key=args.prompt_key,
            val_prompt_ids_path=val_prompt_ids_path,
            val_prompt_count=args.val_prompt_count,
            seed=args.seed,
            max_total_rows=args.max_total_rows,
        )
    else:
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
        **prompt_split_meta,
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
