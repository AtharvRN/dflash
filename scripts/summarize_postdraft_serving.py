"""Summarize measured whole-request throughput; never infer from acceptance."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    groups = {}
    prompts = {}
    workloads = set()
    for path in sorted(root.glob("*/c*_r*.json")):
        row = json.loads(path.read_text())
        results = row["results"]
        assert len(results) == row["requests"]
        assert len({r["prompt_id"] for r in results}) == len(results)
        tokens = sum(r["response"]["meta_info"]["completion_tokens"] for r in results)
        for r in results:
            if "output_ids" in r["response"]:
                assert len(r["response"]["output_ids"]) == r["response"]["meta_info"]["completion_tokens"]
        assert tokens == row["output_tokens"]
        assert abs(tokens/row["wall_time_s"] - row["throughput_tok_s"]) < 1e-8
        workloads.add(row["workload_sha256"])
        key = (row["case"], row["concurrency"])
        groups.setdefault(key, []).append(row)
        prompts[(row["case"], row["concurrency"], row["repeat"])] = {
            r["prompt_id"]: r["response"] for r in results}
    assert len(workloads) <= 1, "Comparisons require identical prompts and warmup"
    metrics = {}
    for (case, c), rows in groups.items():
        tps = [r["throughput_tok_s"] for r in rows]
        metric = {"runs": len(rows), "throughput_tok_s_mean": statistics.mean(tps),
                  "throughput_tok_s_min": min(tps), "throughput_tok_s_max": max(tps),
                  "output_tokens": [r["output_tokens"] for r in rows],
                  "wall_time_s": [r["wall_time_s"] for r in rows]}
        if ("fixed16", c) in groups:
            metric["speedup_vs_fixed16"] = statistics.mean(tps) / statistics.mean(
                r["throughput_tok_s"] for r in groups[("fixed16", c)])
        comparisons = []
        for row in rows:
            actual = prompts[(case, c, row["repeat"])]
            for reference in ["target_ar", "fixed16"]:
                ref = prompts.get((reference, c, row["repeat"]))
                if ref is None:
                    continue
                assert actual.keys() == ref.keys()
                comparisons.append({"reference": reference, "repeat": row["repeat"],
                    "exact_text_matches": sum(actual[p].get("text") == ref[p].get("text") for p in actual),
                    "exact_token_matches": sum(actual[p].get("output_ids") == ref[p].get("output_ids")
                                               for p in actual if "output_ids" in actual[p] and "output_ids" in ref[p]),
                    "requests": len(actual),
                    "note": "Output agreement is not task-accuracy evaluation; shapes/batching can change BF16 argmax."})
        metric["output_agreement"] = comparisons
        metrics[f"{case}_c{c}"] = metric
    return {"measurement": "Actual HTTP output tokens / full workload elapsed wall time, including prefill and all successive cycles",
            "complete": (root / "COMPLETE.json").exists(), "metrics": metrics}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    result = summarize(a.run)
    (a.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
