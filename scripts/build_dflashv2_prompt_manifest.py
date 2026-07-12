from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


def _iter_nemotron_rows(root: Path, categories: set[str]) -> Iterable[dict[str, Any]]:
    data_dir = root / "data"
    for category in sorted(categories):
        for path in sorted(data_dir.glob(f"{category}-*.parquet")):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=2048):
                for row in batch.to_pylist():
                    yield row


def _normalize_nemotron(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    prompt_messages = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "assistant":
            break
        if role in {"system", "user"} and content:
            prompt_messages.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in prompt_messages):
        return None
    return {
        "source": "nemotron_v2",
        "source_id": row.get("uuid"),
        "category": row.get("category"),
        "reasoning": row.get("reasoning"),
        "messages": prompt_messages,
    }


def _iter_codealpaca_rows(path: Path) -> Iterable[dict[str, Any]]:
    rows = json.loads(path.read_text())
    for idx, row in enumerate(rows):
        instruction = (row.get("instruction") or "").strip()
        input_text = (row.get("input") or "").strip()
        if not instruction:
            continue
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{input_text}"
        else:
            prompt = instruction
        yield {
            "source": "codealpaca",
            "source_id": str(idx),
            "category": "code",
            "reasoning": None,
            "messages": [{"role": "user", "content": prompt}],
        }


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.nemotron_root is not None:
        categories = {x.strip() for x in args.nemotron_categories.split(",") if x.strip()}
        for row in _iter_nemotron_rows(args.nemotron_root, categories):
            normalized = _normalize_nemotron(row)
            if normalized is not None:
                rows.append(normalized)
    if args.codealpaca_json is not None:
        rows.extend(_iter_codealpaca_rows(args.codealpaca_json))
    if not rows:
        raise ValueError("no prompt rows were loaded")
    return rows


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_prompts: int | None,
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_prompts is not None:
        rows = rows[:max_prompts]
    val_count = int(round(len(rows) * val_fraction))
    val = rows[:val_count]
    train = rows[val_count:]
    return train, val


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for idx, row in enumerate(rows):
            out = dict(row)
            out["manifest_index"] = idx
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prompt-only manifest for DFlashv2 horizon trace collection."
    )
    parser.add_argument("--nemotron-root", type=Path, default=None)
    parser.add_argument(
        "--nemotron-categories",
        default="chat,code,math",
        help="Comma-separated Nemotron parquet prefixes to include.",
    )
    parser.add_argument("--codealpaca-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="dflashv2_sources")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_rows(args)
    train, val = _split_rows(
        rows,
        seed=args.seed,
        max_prompts=args.max_prompts,
        val_fraction=args.val_fraction,
    )
    _write_jsonl(args.output_dir / f"{args.prefix}_train.jsonl", train)
    _write_jsonl(args.output_dir / f"{args.prefix}_val.jsonl", val)
    summary = {
        "num_loaded": len(rows),
        "num_train": len(train),
        "num_val": len(val),
        "seed": args.seed,
        "max_prompts": args.max_prompts,
        "val_fraction": args.val_fraction,
        "nemotron_root": str(args.nemotron_root) if args.nemotron_root else None,
        "nemotron_categories": args.nemotron_categories,
        "codealpaca_json": str(args.codealpaca_json) if args.codealpaca_json else None,
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
