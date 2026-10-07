"""Numeric answer checks for our zero-shot chat protocol, not leaderboard GSM8K."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import statistics

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def numeric(value):
    if value is None:
        return None
    text = value.strip().replace(",", "").replace("−", "-")
    text = text.replace(r"\$", "").strip("$ ")
    frac = re.fullmatch(r"\\(?:d?frac)\{(" + NUMBER + r")\}\{(" + NUMBER + r")\}", text)
    try:
        if frac:
            return Fraction(frac[1]) / Fraction(frac[2])
        if re.fullmatch(NUMBER, text):
            return Fraction(text)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def last_box(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def predictions(text):
    strict = numeric(last_box(text))
    # Secondary diagnostic only. May select an intermediate number if no final
    # box exists; never silently substitute it for the primary boxed metric.
    numbers = re.findall(NUMBER, text.replace(",", "").replace("−", "-"))
    relaxed = strict if strict is not None else numeric(numbers[-1]) if numbers else None
    return strict, relaxed


def summarize(root):
    workload_path = root / "workload.json"
    workload = json.loads(workload_path.read_text())
    expected_hash = hashlib.sha256(workload_path.read_bytes()).hexdigest()
    gold = {r["prompt_id"]: numeric(r["gold_answer"]) for r in workload["measurement"]}
    assert len(gold) == len(workload["measurement"]) and all(v is not None for v in gold.values())
    phases, groups = {}, defaultdict(list)
    for path in sorted(root.glob("*/c*_r*.json")):
        run = json.loads(path.read_text())
        assert run["workload_sha256"] == expected_hash
        assert len(run["results"]) == len(gold) == run["requests"]
        assert {r["prompt_id"] for r in run["results"]} == set(gold)
        counts, finishes = Counter(), Counter()
        for item in run["results"]:
            response = item["response"]
            meta = response["meta_info"]
            strict, relaxed = predictions(response["text"])
            reference = gold[item["prompt_id"]]
            counts.update(strict_correct=int(strict == reference),
                          relaxed_correct=int(relaxed == reference),
                          boxed_parse_failures=int(strict is None),
                          cycles=meta.get("spec_verify_ct", 0),
                          accepted=meta.get("spec_num_correct_drafts", 0),
                          output_tokens=meta["completion_tokens"])
            finishes[meta["finish_reason"]["type"]] += 1
        assert sum(finishes.values()) == len(gold)
        assert set(finishes) <= {"stop", "length"}, finishes
        assert counts["output_tokens"] == run["output_tokens"]
        n = len(gold)
        row = dict(counts) | {"requests": n, "strict_boxed_accuracy": counts["strict_correct"]/n,
            "relaxed_numeric_accuracy": counts["relaxed_correct"]/n,
            "length_cap_fraction": finishes["length"]/n, "finish_reasons": dict(finishes),
            "mean_accepted_drafts": counts["accepted"]/counts["cycles"] if counts["cycles"] else None,
            "throughput_tok_s": run["throughput_tok_s"], "repeat": run["repeat"]}
        phases[f"{run['case']}_c{run['concurrency']}_r{run['repeat']}"] = row
        groups[f"{run['case']}_c{run['concurrency']}"].append(row)
    metrics = {}
    for key, rows in groups.items():
        n = sum(r["requests"] for r in rows)
        cycles = sum(r["cycles"] for r in rows)
        metrics[key] = {"repeats": len(rows), "scored_responses": n,
            "strict_boxed_accuracy": sum(r["strict_correct"] for r in rows)/n,
            "relaxed_numeric_accuracy": sum(r["relaxed_correct"] for r in rows)/n,
            "boxed_parse_failures": sum(r["boxed_parse_failures"] for r in rows),
            "length_cap_fraction": sum(r["finish_reasons"].get("length", 0) for r in rows)/n,
            "mean_accepted_drafts": sum(r["accepted"] for r in rows)/cycles if cycles else None,
            "throughput_tok_s_mean": statistics.mean(r["throughput_tok_s"] for r in rows)}
    return {"complete": (root/"COMPLETE.json").exists(), "dataset_revision": workload["revision"],
            "workload_sha256": expected_hash, "unique_test_questions": len(gold),
            "protocol": workload["protocol"], "metrics": metrics, "phases": phases,
            "caveats": ["Custom zero-shot chat protocol, not standard few-shot lm-eval score.",
                        "Repeated questions are not independent accuracy samples.",
                        "Strict metric requires a parseable final numeric box; failures count as incorrect.",
                        "Relaxed last-number fallback is diagnostic, not the primary metric."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    result = summarize(a.run)
    (a.run/"gsm8k_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
