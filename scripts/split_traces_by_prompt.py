from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DFlash trace rows by prompt_id.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_ids: set[str] = set()
    with args.input_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            prompt_ids.add(str(row["prompt_id"]))

    prompts = sorted(prompt_ids)
    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    num_val = max(1, int(round(len(prompts) * args.val_fraction)))
    val_prompts = set(prompts[:num_val])
    train_prompts = set(prompts[num_val:])

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    train_rows = 0
    val_rows = 0
    with args.input_jsonl.open() as src, args.train_output.open("w") as train_f, args.val_output.open("w") as val_f:
        for line in src:
            row = json.loads(line)
            prompt_id = str(row["prompt_id"])
            if prompt_id in val_prompts:
                val_f.write(line)
                val_rows += 1
            else:
                train_f.write(line)
                train_rows += 1

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "train_output": str(args.train_output),
        "val_output": str(args.val_output),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "num_train_prompts": len(train_prompts),
        "num_val_prompts": len(val_prompts),
        "num_train_rows": train_rows,
        "num_val_rows": val_rows,
        "val_prompt_ids": sorted(val_prompts),
    }
    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
