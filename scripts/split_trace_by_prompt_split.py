from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a trace JSONL using prompt IDs from a validation trace.")
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--val-reference-jsonl", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_prompt_ids: set[str] = set()
    with args.val_reference_jsonl.open("rb") as f:
        for line in f:
            val_prompt_ids.add(str(_loads(line)["prompt_id"]))

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)

    train_rows = 0
    val_rows = 0
    train_prompt_ids: set[str] = set()
    observed_val_prompt_ids: set[str] = set()
    with args.source_jsonl.open("rb") as src, args.train_output.open("wb") as train_f, args.val_output.open("wb") as val_f:
        for idx, line in enumerate(src, 1):
            row = _loads(line)
            prompt_id = str(row["prompt_id"])
            if prompt_id in val_prompt_ids:
                val_f.write(line)
                val_rows += 1
                observed_val_prompt_ids.add(prompt_id)
            else:
                train_f.write(line)
                train_rows += 1
                train_prompt_ids.add(prompt_id)
            if idx % 25000 == 0:
                print(f"processed {idx} train_rows={train_rows} val_rows={val_rows}", flush=True)

    print(
        json.dumps(
            {
                "train_rows": train_rows,
                "val_rows": val_rows,
                "train_prompts": len(train_prompt_ids),
                "val_prompts": len(observed_val_prompt_ids),
                "missing_val_prompts": len(val_prompt_ids - observed_val_prompt_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
