#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


TRAIN_MIX_SPEC = [
    {
        "name": "tatsu-lab/alpaca",
        "config": None,
        "split": "train",
        "count": 4940,
        "task_family": "instruction",
        "source_name": "alpaca_train",
    },
    {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "count": 4940,
        "task_family": "math",
        "source_name": "gsm8k_train",
    },
    {
        "name": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "train",
        "count": 120,
        "task_family": "code",
        "source_name": "mbpp_train",
    },
]


DEV_MIX_SPEC = [
    {
        "name": "tatsu-lab/alpaca",
        "config": None,
        "split": "train",
        "count": 500,
        "task_family": "instruction",
        "source_name": "alpaca_train_dev_holdout",
    },
    {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "count": 500,
        "task_family": "math",
        "source_name": "gsm8k_train_dev_holdout",
    },
    {
        "name": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "validation",
        "count": 43,
        "task_family": "code",
        "source_name": "mbpp_validation_dev_holdout",
    },
]


def _format_turns(spec: dict, item: dict) -> list[str]:
    source_name = spec["source_name"]
    if source_name.startswith("alpaca"):
        instruction = item["instruction"]
        user_input = item.get("input", "")
        formatted = f"{instruction}\n\nInput:\n{user_input}" if user_input else instruction
        return [formatted]
    if source_name.startswith("gsm8k"):
        prompt_fmt = (
            "{question}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        return [prompt_fmt.format(**item)]
    if source_name.startswith("mbpp"):
        return [item["prompt"]]
    raise ValueError(f"Unsupported source for formatting: {source_name}")


def _sample_rows(spec: dict, *, seed: int, offset: int = 0) -> list[dict]:
    dataset = load_dataset(spec["name"], spec["config"], split=spec["split"])
    total = len(dataset)
    needed = int(spec["count"])
    if total < offset + needed:
        raise ValueError(
            f"Not enough rows in {spec['source_name']}: total={total}, "
            f"offset={offset}, needed={needed}"
        )

    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = indices[offset : offset + needed]

    rows = []
    for rank, dataset_idx in enumerate(chosen):
        item = dataset[int(dataset_idx)]
        rows.append(
            {
                "id": f"{spec['source_name']}:{int(dataset_idx)}",
                "source_dataset": spec["name"],
                "source_config": spec["config"],
                "source_split": spec["split"],
                "source_name": spec["source_name"],
                "task_family": spec["task_family"],
                "source_index": int(dataset_idx),
                "sample_rank_within_source": int(rank),
                "turns": _format_turns(spec, item),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a predictor-training prompt mix from train-like splits only. "
            "Writes a 10k-train JSONL plus a small dev holdout JSONL."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    train_offsets: dict[str, int] = {}
    source_seeds = {
        "alpaca_train": int(args.seed) + 1,
        "gsm8k_train": int(args.seed) + 2,
        "mbpp_train": int(args.seed) + 3,
    }

    for spec in TRAIN_MIX_SPEC:
        seed = source_seeds[spec["source_name"]]
        rows = _sample_rows(spec, seed=seed, offset=0)
        train_rows.extend(rows)
        train_offsets[spec["source_name"]] = int(spec["count"])

    for spec in DEV_MIX_SPEC:
        offset = 0
        seed = int(args.seed) + 1000
        if spec["source_name"].startswith("alpaca_train_dev_holdout"):
            offset = train_offsets["alpaca_train"]
            seed = source_seeds["alpaca_train"]
        elif spec["source_name"].startswith("gsm8k_train_dev_holdout"):
            offset = train_offsets["gsm8k_train"]
            seed = source_seeds["gsm8k_train"]
        elif spec["source_name"].startswith("mbpp_validation"):
            seed = int(args.seed) + 4
        rows = _sample_rows(spec, seed=seed, offset=offset)
        dev_rows.extend(rows)

    train_path = output_dir / "predictor_train_mix_10000.jsonl"
    dev_path = output_dir / "predictor_dev_mix_1043.jsonl"
    manifest_path = output_dir / "predictor_mix_manifest.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)

    manifest = {
        "seed": int(args.seed),
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "train_mix_spec": TRAIN_MIX_SPEC,
        "dev_mix_spec": DEV_MIX_SPEC,
        "final_eval_policy": {
            "do_not_train_on": [
                "aime25",
                "gsm8k fixed128",
                "humaneval test",
                "mt-bench prompts",
            ],
            "benchmark_only": [
                "aime25",
                "gsm8k fixed subset",
                "humaneval",
                "mt-bench",
            ],
        },
        "files": {
            "train_jsonl": str(train_path),
            "dev_jsonl": str(dev_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote train rows: {len(train_rows)} -> {train_path}")
    print(f"Wrote dev rows: {len(dev_rows)} -> {dev_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
