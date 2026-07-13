from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dflash.benchmark import load_and_process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt-only manifests from DFlash benchmark datasets."
    )
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_and_process_dataset(args.dataset)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w") as f:
        for idx, row in enumerate(rows):
            for turn_idx, prompt in enumerate(row["turns"]):
                payload = {
                    "source": args.dataset,
                    "source_id": str(idx),
                    "manifest_index": idx,
                    "turn_index": turn_idx,
                    "prompt": prompt,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1

    print(json.dumps({"dataset": args.dataset, "output": str(args.output), "rows": count}, indent=2))


if __name__ == "__main__":
    main()
