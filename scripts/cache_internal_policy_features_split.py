from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _scan_inputs(paths: list[Path], val_fraction: float, seed: int) -> tuple[set[str], dict[str, int]]:
    prompt_ids: set[str] = set()
    rows_by_prompt: dict[str, int] = {}
    for path in paths:
        with path.open("rb") as f:
            for line in f:
                row = _loads(line)
                prompt_id = str(row["prompt_id"])
                prompt_ids.add(prompt_id)
                rows_by_prompt[prompt_id] = rows_by_prompt.get(prompt_id, 0) + 1

    prompts = sorted(prompt_ids)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    num_val = max(1, int(round(len(prompts) * val_fraction)))
    return set(prompts[:num_val]), rows_by_prompt


def _open_cache(output_dir: Path, num_rows: int, feature_dim: int, window: int, num_draft_slots: int) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "context": np.lib.format.open_memmap(
            output_dir / "dflash_context.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_rows, feature_dim),
        ),
        "window": np.lib.format.open_memmap(
            output_dir / "dflash_context_window_seq.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_rows, window, feature_dim + 1),
        ),
        "survival": np.lib.format.open_memmap(
            output_dir / "survival.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_rows, num_draft_slots),
        ),
        "accepted_len": np.lib.format.open_memmap(
            output_dir / "accepted_len.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_rows,),
        ),
    }


def _write_row(
    cache: dict[str, np.ndarray],
    idx: int,
    row: dict[str, Any],
    *,
    feature_dim: int,
) -> None:
    inputs = row["inputs"]
    cache["context"][idx] = np.asarray(inputs["dflash_context"], dtype=np.float32)
    context_window = np.asarray(inputs["dflash_context_window"], dtype=np.float32)
    mask = np.asarray(inputs["dflash_context_window_mask"], dtype=np.float32)
    cache["window"][idx, :, :feature_dim] = context_window
    cache["window"][idx, :, feature_dim] = mask
    labels = row["labels"]
    cache["survival"][idx] = np.asarray(labels["draft_survival"], dtype=np.float32)
    cache["accepted_len"][idx] = float(labels["accepted_draft_len"])


def _write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DFlash internal traces by prompt and cache features directly.")
    parser.add_argument("--jsonl", type=Path, action="append", required=True)
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--val-output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--num-draft-slots", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_prompts, rows_by_prompt = _scan_inputs(args.jsonl, args.val_fraction, args.seed)
    train_rows = sum(count for prompt, count in rows_by_prompt.items() if prompt not in val_prompts)
    val_rows = sum(count for prompt, count in rows_by_prompt.items() if prompt in val_prompts)
    print(
        f"prompts={len(rows_by_prompt)} train_rows={train_rows} val_rows={val_rows}",
        flush=True,
    )

    train_cache = _open_cache(
        args.train_output_dir,
        train_rows,
        args.feature_dim,
        args.window,
        args.num_draft_slots,
    )
    val_cache = _open_cache(
        args.val_output_dir,
        val_rows,
        args.feature_dim,
        args.window,
        args.num_draft_slots,
    )
    train_idx = 0
    val_idx = 0
    for path in args.jsonl:
        print(f"caching {path}", flush=True)
        with path.open("rb") as f:
            for line in f:
                row = _loads(line)
                prompt_id = str(row["prompt_id"])
                if prompt_id in val_prompts:
                    _write_row(val_cache, val_idx, row, feature_dim=args.feature_dim)
                    val_idx += 1
                else:
                    _write_row(train_cache, train_idx, row, feature_dim=args.feature_dim)
                    train_idx += 1
                if (train_idx + val_idx) % 25000 == 0:
                    print(f"cached {train_idx + val_idx}/{train_rows + val_rows}", flush=True)

    metadata_common = {
        "source_jsonl": [str(path) for path in args.jsonl],
        "feature_dim": args.feature_dim,
        "window": args.window,
        "num_draft_slots": args.num_draft_slots,
    }
    _write_metadata(args.train_output_dir, {**metadata_common, "num_rows": train_idx})
    _write_metadata(args.val_output_dir, {**metadata_common, "num_rows": val_idx})

    manifest = {
        "source_jsonl": [str(path) for path in args.jsonl],
        "train_output_dir": str(args.train_output_dir),
        "val_output_dir": str(args.val_output_dir),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_prompts": len(rows_by_prompt),
        "num_train_prompts": len(rows_by_prompt) - len(val_prompts),
        "num_val_prompts": len(val_prompts),
        "num_train_rows": train_idx,
        "num_val_rows": val_idx,
        "val_prompt_ids": sorted(val_prompts),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
