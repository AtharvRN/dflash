from __future__ import annotations

import argparse
import json
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


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache DFlash internal trace features as binary numpy arrays.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--num-draft-slots", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_rows = _count_lines(args.jsonl)
    print(f"caching {num_rows} rows from {args.jsonl}", flush=True)

    context = np.lib.format.open_memmap(
        args.output_dir / "dflash_context.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows, args.feature_dim),
    )
    window = np.lib.format.open_memmap(
        args.output_dir / "dflash_context_window_seq.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows, args.window, args.feature_dim + 1),
    )
    survival = np.lib.format.open_memmap(
        args.output_dir / "survival.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows, args.num_draft_slots),
    )
    accepted_len = np.lib.format.open_memmap(
        args.output_dir / "accepted_len.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows,),
    )

    with args.jsonl.open("rb") as f:
        for idx, line in enumerate(f):
            row = _loads(line)
            inputs = row["inputs"]
            context[idx] = np.asarray(inputs["dflash_context"], dtype=np.float32)
            context_window = np.asarray(inputs["dflash_context_window"], dtype=np.float32)
            mask = np.asarray(inputs["dflash_context_window_mask"], dtype=np.float32)
            window[idx, :, : args.feature_dim] = context_window
            window[idx, :, args.feature_dim] = mask
            labels = row["labels"]
            survival[idx] = np.asarray(labels["draft_survival"], dtype=np.float32)
            accepted_len[idx] = float(labels["accepted_draft_len"])
            if (idx + 1) % 25000 == 0:
                print(f"cached {idx + 1}/{num_rows}", flush=True)

    metadata = {
        "source_jsonl": str(args.jsonl),
        "num_rows": num_rows,
        "feature_dim": args.feature_dim,
        "window": args.window,
        "num_draft_slots": args.num_draft_slots,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote cache to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
