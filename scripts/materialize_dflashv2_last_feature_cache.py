from __future__ import annotations

import argparse
import concurrent.futures
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
    include_predraft_token_ids: bool,
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
    token_ids = None
    token_mask = None
    if include_predraft_token_ids:
        token_window = int(np.load(shards[0].path / "predraft_token_ids.npy", mmap_mode="r").shape[1])
        token_ids = np.lib.format.open_memmap(
            split_dir / "predraft_token_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(order.shape[0], token_window),
        )
        token_mask = np.lib.format.open_memmap(
            split_dir / "predraft_token_mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(order.shape[0], token_window),
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
        shard_token_ids = (
            np.load(shard.path / "predraft_token_ids.npy", mmap_mode="r")
            if include_predraft_token_ids
            else None
        )
        shard_token_mask = (
            np.load(shard.path / "predraft_token_mask.npy", mmap_mode="r")
            if include_predraft_token_ids
            else None
        )
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
            if include_predraft_token_ids:
                assert token_ids is not None
                assert token_mask is not None
                assert shard_token_ids is not None
                assert shard_token_mask is not None
                token_ids[out_start:out_end] = np.asarray(shard_token_ids[chunk], dtype=np.int64)
                token_mask[out_start:out_end] = np.asarray(shard_token_mask[chunk], dtype=np.float32)

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
    if token_ids is not None:
        token_ids.flush()
    if token_mask is not None:
        token_mask.flush()
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


def _materialize_shard_task(task: dict[str, Any]) -> dict[str, Any]:
    shard_path = Path(task["shard_path"])
    split_dir = Path(task["split_dir"])
    shard_idx = int(task["shard_index"])
    local_indices = np.asarray(task["local_indices"], dtype=np.int64)
    out_base = int(task["out_base"])
    max_read_rows = int(task["max_read_rows"])
    max_output_rows = int(task["max_output_rows"])
    include_predraft_token_ids = bool(task.get("include_predraft_token_ids", False))

    shard_started = time.time()
    shard_features = np.load(shard_path / "features.npy", mmap_mode="r")
    shard_mask = np.load(shard_path / "mask.npy", mmap_mode="r")
    shard_survival = np.load(shard_path / "survival.npy", mmap_mode="r")
    shard_accepted = np.load(shard_path / "accepted_len.npy", mmap_mode="r")
    shard_token_ids = np.load(shard_path / "predraft_token_ids.npy", mmap_mode="r") if include_predraft_token_ids else None
    shard_token_mask = (
        np.load(shard_path / "predraft_token_mask.npy", mmap_mode="r") if include_predraft_token_ids else None
    )

    features = np.load(split_dir / "features.npy", mmap_mode="r+")
    mask = np.load(split_dir / "mask.npy", mmap_mode="r+")
    survival = np.load(split_dir / "survival.npy", mmap_mode="r+")
    accepted = np.load(split_dir / "accepted_len.npy", mmap_mode="r+")
    token_ids = np.load(split_dir / "predraft_token_ids.npy", mmap_mode="r+") if include_predraft_token_ids else None
    token_mask = np.load(split_dir / "predraft_token_mask.npy", mmap_mode="r+") if include_predraft_token_ids else None

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
        out_start = out_base + chunk_start
        out_end = out_base + rel_idx

        raw_mask = np.asarray(shard_mask[read_start:read_end], dtype=np.float32)
        valid_mask = raw_mask > 0.5
        has_valid = valid_mask.any(axis=1)
        last_valid = valid_mask.shape[1] - 1 - np.argmax(valid_mask[:, ::-1], axis=1)
        last_valid = np.where(has_valid, last_valid, valid_mask.shape[1] - 1)
        selected_last = last_valid[positions]

        feature_block = np.asarray(shard_features[read_start:read_end], dtype=np.float16)
        features[out_start:out_end, 0] = feature_block[positions, selected_last]
        mask[out_start:out_end, 0] = has_valid[positions].astype(np.float32)
        survival[out_start:out_end] = np.asarray(shard_survival[chunk], dtype=np.float32)
        accepted[out_start:out_end] = np.asarray(shard_accepted[chunk], dtype=np.float32)
        if include_predraft_token_ids:
            assert token_ids is not None
            assert token_mask is not None
            assert shard_token_ids is not None
            assert shard_token_mask is not None
            token_ids[out_start:out_end] = np.asarray(shard_token_ids[chunk], dtype=np.int64)
            token_mask[out_start:out_end] = np.asarray(shard_token_mask[chunk], dtype=np.float32)

    features.flush()
    mask.flush()
    survival.flush()
    accepted.flush()
    if token_ids is not None:
        token_ids.flush()
    if token_mask is not None:
        token_mask.flush()
    return {
        "shard": str(shard_path),
        "shard_index": shard_idx,
        "rows": int(local_indices.shape[0]),
        "elapsed_s": round(time.time() - shard_started, 3),
    }


def _materialize_split_parallel(
    *,
    name: str,
    shards: list[Any],
    offsets: list[int],
    indices: np.ndarray,
    output_dir: Path,
    max_read_rows: int,
    max_output_rows: int,
    parallel_workers: int,
    include_predraft_token_ids: bool,
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
    token_ids = None
    token_mask = None
    if include_predraft_token_ids:
        token_window = int(np.load(shards[0].path / "predraft_token_ids.npy", mmap_mode="r").shape[1])
        token_ids = np.lib.format.open_memmap(
            split_dir / "predraft_token_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(order.shape[0], token_window),
        )
        token_mask = np.lib.format.open_memmap(
            split_dir / "predraft_token_mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(order.shape[0], token_window),
        )
    features.flush()
    mask.flush()
    survival.flush()
    accepted.flush()
    if token_ids is not None:
        token_ids.flush()
    if token_mask is not None:
        token_mask.flush()
    del features, mask, survival, accepted, token_ids, token_mask

    tasks: list[dict[str, Any]] = []
    for shard_idx, shard in enumerate(shards):
        shard_start = offsets[shard_idx]
        shard_end = shard_start + int(shard.rows)
        left = np.searchsorted(order, shard_start, side="left")
        right = np.searchsorted(order, shard_end, side="left")
        if right <= left:
            continue
        local_indices = order[left:right] - shard_start
        tasks.append(
            {
                "shard_path": str(shard.path),
                "split_dir": str(split_dir),
                "shard_index": shard_idx,
                "local_indices": local_indices,
                "out_base": int(left),
                "max_read_rows": int(max_read_rows),
                "max_output_rows": int(max_output_rows),
                "include_predraft_token_ids": bool(include_predraft_token_ids),
            }
        )

    started = time.time()
    shard_rows: list[dict[str, Any]] = []
    rows_done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(_materialize_shard_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            shard_rows.append(item)
            rows_done += int(item["rows"])
            print(
                json.dumps(
                    {
                        "event": "materialize_shard_done",
                        "split": name,
                        "parallel_workers": int(parallel_workers),
                        "split_rows_done": int(rows_done),
                        "split_rows_total": int(order.shape[0]),
                        **item,
                    }
                ),
                flush=True,
            )

    shard_rows.sort(key=lambda item: int(item["shard_index"]))
    meta = {
        "split": name,
        "rows": int(order.shape[0]),
        "input_dim": input_dim,
        "num_slots": num_slots,
        "elapsed_s": round(time.time() - started, 3),
        "parallel_workers": int(parallel_workers),
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
    src_token_ids = (
        np.load(source_dir / "predraft_token_ids.npy", mmap_mode="r")
        if (source_dir / "predraft_token_ids.npy").exists()
        else None
    )
    src_token_mask = (
        np.load(source_dir / "predraft_token_mask.npy", mmap_mode="r")
        if (source_dir / "predraft_token_mask.npy").exists()
        else None
    )

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
    token_ids = None
    token_mask = None
    if src_token_ids is not None:
        token_ids = np.lib.format.open_memmap(
            split_dir / "predraft_token_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(positions.shape[0], src_token_ids.shape[1]),
        )
    if src_token_mask is not None:
        token_mask = np.lib.format.open_memmap(
            split_dir / "predraft_token_mask.npy",
            mode="w+",
            dtype=np.float32,
            shape=(positions.shape[0], src_token_mask.shape[1]),
        )

    started = time.time()
    for start in range(0, int(positions.shape[0]), copy_rows):
        end = min(start + copy_rows, int(positions.shape[0]))
        chunk = positions[start:end]
        features[start:end] = src_features[chunk]
        mask[start:end] = src_mask[chunk]
        survival[start:end] = src_survival[chunk]
        accepted[start:end] = src_accepted[chunk]
        if token_ids is not None:
            assert src_token_ids is not None
            token_ids[start:end] = src_token_ids[chunk]
        if token_mask is not None:
            assert src_token_mask is not None
            token_mask[start:end] = src_token_mask[chunk]
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
    if token_ids is not None:
        token_ids.flush()
    if token_mask is not None:
        token_mask.flush()
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


def _read_prompt_split_shard_task(task: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(task["metadata_path"])
    shard_start = int(task["shard_start"])
    shard_rows = int(task["shard_rows"])
    prompt_key = str(task["prompt_key"])
    val_prompt_ids = {str(item) for item in task["val_prompt_ids"]}
    shard_index = int(task["shard_index"])

    if not metadata_path.exists():
        return {
            "shard_index": shard_index,
            "metadata_path": str(metadata_path),
            "missing": True,
            "train": [],
            "val": [],
            "observed_prompts": [],
            "observed_val_prompts": [],
        }

    train: list[int] = []
    val: list[int] = []
    observed_prompts: set[str] = set()
    observed_val_prompts: set[str] = set()
    with metadata_path.open() as f:
        for local_idx, line in enumerate(f):
            if local_idx >= shard_rows:
                break
            row = json.loads(line)
            prompt_id = _metadata_prompt_key(row, prompt_key)
            observed_prompts.add(prompt_id)
            idx = shard_start + local_idx
            if prompt_id in val_prompt_ids:
                val.append(idx)
                observed_val_prompts.add(prompt_id)
            else:
                train.append(idx)

    return {
        "shard_index": shard_index,
        "metadata_path": str(metadata_path),
        "missing": False,
        "train": train,
        "val": val,
        "observed_prompts": sorted(observed_prompts),
        "observed_val_prompts": sorted(observed_val_prompts),
    }


def _make_prompt_splits_from_metadata(
    *,
    shards: list[Any],
    offsets: list[int],
    prompt_key: str,
    val_prompt_ids_path: Path,
    val_prompt_count: int,
    seed: int,
    max_total_rows: int | None,
    parallel_workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if val_prompt_ids_path.exists():
        val_prompt_ids = _load_or_create_val_prompts(
            prompt_ids=[],
            path=val_prompt_ids_path,
            val_prompt_count=val_prompt_count,
            seed=seed,
        )
        tasks = [
            {
                "metadata_path": str(shard.path / "metadata.jsonl"),
                "shard_start": int(offsets[shard_idx]),
                "shard_rows": int(shard.rows),
                "prompt_key": prompt_key,
                "val_prompt_ids": sorted(val_prompt_ids),
                "shard_index": int(shard_idx),
            }
            for shard_idx, shard in enumerate(shards)
        ]
        results: list[dict[str, Any]] = []
        started = time.time()
        if parallel_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(_read_prompt_split_shard_task, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(
                        json.dumps(
                            {
                                "event": "metadata_split_shard_done",
                                "parallel_workers": int(parallel_workers),
                                "shards_done": len(results),
                                "shards_total": len(tasks),
                                "shard_index": int(result["shard_index"]),
                                "train_rows": len(result["train"]),
                                "val_rows": len(result["val"]),
                                "elapsed_s": round(time.time() - started, 3),
                            }
                        ),
                        flush=True,
                    )
        else:
            for task in tasks:
                result = _read_prompt_split_shard_task(task)
                results.append(result)

        results.sort(key=lambda item: int(item["shard_index"]))
        missing_metadata = [str(item["metadata_path"]) for item in results if item["missing"]]
        if missing_metadata:
            raise FileNotFoundError(
                "prompt split requires metadata.jsonl for every shard; missing: "
                + ", ".join(missing_metadata[:5])
            )

        train = [idx for item in results for idx in item["train"]]
        val = [idx for item in results for idx in item["val"]]
        observed_prompts = {prompt for item in results for prompt in item["observed_prompts"]}
        observed_val_prompts = {prompt for item in results for prompt in item["observed_val_prompts"]}
        if not train and not val:
            raise ValueError("no rows found while reading metadata for prompt split")

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
            "num_observed_rows": int(len(train) + len(val)),
            "num_observed_prompts": int(len(observed_prompts)),
            "num_val_prompts_configured": int(len(val_prompt_ids)),
            "num_val_prompts_observed": int(len(observed_val_prompts)),
            "train_rows": int(train_arr.shape[0]),
            "val_rows": int(val_arr.shape[0]),
            "metadata_parallel_workers": int(parallel_workers),
            "metadata_elapsed_s": round(time.time() - started, 3),
        }
        return train_arr, val_arr, meta

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
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Shard-level worker processes for materializing the temporary _all cache.",
    )
    parser.add_argument(
        "--include-predraft-token-ids",
        action="store_true",
        help="Also materialize predraft_token_ids.npy and predraft_token_mask.npy when present in every shard.",
    )
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
    include_predraft_token_ids = bool(args.include_predraft_token_ids)
    if include_predraft_token_ids:
        missing = [
            str(shard.path)
            for shard in shards
            if not (shard.path / "predraft_token_ids.npy").exists()
            or not (shard.path / "predraft_token_mask.npy").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "--include-predraft-token-ids requires predraft token arrays in every shard; "
                f"missing in {missing[:5]}"
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
            parallel_workers=args.parallel_workers,
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
        "include_predraft_token_ids": include_predraft_token_ids,
        **prompt_split_meta,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(args.output_dir / "meta.json", run_meta)
    print(json.dumps({"event": "materialize_start", **run_meta}), flush=True)

    combined_indices = np.sort(np.concatenate([train_indices, val_indices]).astype(np.int64, copy=False))
    all_dir = args.output_dir / "_all"
    if args.parallel_workers > 1:
        all_meta = _materialize_split_parallel(
            name="_all",
            shards=shards,
            offsets=offsets,
            indices=combined_indices,
            output_dir=args.output_dir,
            max_read_rows=args.max_read_rows,
            max_output_rows=args.max_output_rows,
            parallel_workers=args.parallel_workers,
            include_predraft_token_ids=include_predraft_token_ids,
        )
    else:
        all_meta = _materialize_split(
            name="_all",
            shards=shards,
            offsets=offsets,
            indices=combined_indices,
            output_dir=args.output_dir,
            progress_every=args.progress_every,
            max_read_rows=args.max_read_rows,
            max_output_rows=args.max_output_rows,
            include_predraft_token_ids=include_predraft_token_ids,
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
