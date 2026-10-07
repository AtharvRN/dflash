"""Freeze a zero-shot GSM8K workload for the existing SGLang serving runner."""
import argparse
import hashlib
import json
from pathlib import Path
import random

REVISION = "740312add88f781978c0658806c59bc2815b9866"
PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def content_hash(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()


def build_workload(train, test, tokenizer, warmup=128, seed=934):
    # No test-set filtering or subsampling: fail if any prompt cannot be served.
    test_hashes = {content_hash(row["question"]) for row in test}
    test_ids = list(range(len(test)))
    random.Random(seed).shuffle(test_ids)
    train_ids = list(range(len(train)))
    random.Random(seed).shuffle(train_ids)
    seen = set(test_hashes)
    warm_ids = []
    for i in train_ids:
        h = content_hash(train[i]["question"])
        if h not in seen:
            warm_ids.append(i)
            seen.add(h)
        if len(warm_ids) == warmup:
            break
    if len(warm_ids) != warmup:
        raise ValueError("Insufficient disjoint training-split warmup questions")

    def encode(row, split, index):
        question = row["question"]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=question)}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=False)
        if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
            raise ValueError("Expected a nonempty flat token list")
        if len(ids) > 2048:
            raise ValueError("Full test coverage requires every prompt to fit the input limit")
        if "####" not in row["answer"]:
            raise ValueError("Missing GSM8K numeric reference delimiter")
        return {"prompt_id": f"gsm8k_{split}_{index}", "source": "openai/gsm8k",
                "dataset_index": index, "split": split,
                "question_sha256": content_hash(question), "input_ids": ids,
                # Request code sends only input_ids, never this evaluation-only field.
                "gold_answer": row["answer"].rsplit("####", 1)[1].strip()}

    return {"dataset": "openai/gsm8k", "subset": "main", "revision": REVISION,
            "protocol": "zero-shot chat; thinking off; boxed final answer; not lm-eval few-shot",
            "prompt_template": PROMPT, "seed": seed,
            "measurement_split": "test", "warmup_split": "train",
            "test_rows": len(test), "test_unique_questions": len(test_hashes),
            "source_rows_sha256": hashlib.sha256(json.dumps(
                {"train": train, "test": test}, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "warmup": [encode(train[i], "train", i) for i in warm_ids],
            "measurement": [encode(test[i], "test", i) for i in test_ids]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--cache", default="/tmp/postdraft-gsm8k-datasets")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("openai/gsm8k", "main", revision=REVISION, cache_dir=a.cache)
    train, test = list(ds["train"]), list(ds["test"])
    if (len(train), len(test)) != (7473, 1319):
        raise ValueError("Unexpected pinned dataset split sizes")
    model = json.loads(a.models.read_text())["target"]
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
    result = build_workload(train, test, tokenizer)
    result["tokenizer"] = model
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"path": str(a.output), "test_rows": len(test), "warmup_rows": 128,
                      "sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                      "max_prompt_tokens": max(len(x["input_ids"]) for x in result["measurement"])}))


if __name__ == "__main__":
    main()
