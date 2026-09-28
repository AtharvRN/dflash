"""Attribute CUDA activities through launch correlations to DFlash CPU ranges.

Unattributed activities are counted explicitly, never guessed into a stage.
Kernel duration sums and GPU envelopes are not end-to-end cycle latency.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


def kernel_category(name):
    lower = name.lower()
    if any(x in lower for x in ("attention", "flash_fwd", "flash_bwd")):
        return "attention"
    if any(x in lower for x in ("gemm", "xmma", "cutlass", "cublas", "matmul")):
        return "gemm"
    if any(x in lower for x in ("rmsnorm", "layernorm", "rms_norm", "layer_norm")):
        return "normalization"
    if any(x in lower for x in ("rope", "rotary")):
        return "rotary"
    if any(x in lower for x in ("argmax", "softmax", "reduce", "reduction")):
        return "reductions_sampling"
    if any(x in lower for x in ("index", "gather", "scatter", "copy", "cat", "assign", "materialize", "cache")):
        return "indexing_copies_kv"
    return "other"


def interval_union(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop-max(start,end))
        end = max(end,stop)
    return total


def analyze(events, batch_size):
    ranges = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("DFLASH/")]
    roots = [e for e in ranges if e["name"].startswith("DFLASH/decode_cycle|") and f"|C={batch_size}|" in e["name"]]
    if not roots:
        raise ValueError("No annotated full-batch decode cycles in trace")
    runtime = [e for e in events if e.get("cat") in {"cuda_runtime", "cuda_driver"} and e.get("ph") == "X"]
    # Each correlation identifies the host launch, including cudaGraphLaunch.
    owners = {}
    cpu_api = defaultdict(float)
    for event in runtime:
        t = event["ts"]
        candidates = [r for r in roots if r.get("pid") == event.get("pid") and r.get("tid") == event.get("tid")
                      and r["ts"] <= t < r["ts"]+r["dur"]]
        if not candidates:
            continue
        root = candidates[0]
        inner = [r for r in ranges if r.get("pid") == event.get("pid") and r.get("tid") == event.get("tid")
                 and root["ts"] <= r["ts"] and r["ts"] <= t < r["ts"]+r["dur"] <= root["ts"]+root["dur"]+1]
        stage = min(inner, key=lambda r:r["dur"])["name"].split("|")[0] if inner else "DFLASH/decode_cycle"
        corr = event.get("args", {}).get("correlation")
        if corr not in (None, 0):
            owners[corr] = (root["name"], stage)
        cpu_api[event["cat"] + "/" + event["name"]] += event["dur"]
    kernels, stages, categories, activity = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(list)
    stage_kernels = defaultdict(lambda: defaultdict(float))
    all_gpu_us, attributed_us, unmatched_count = 0.0, 0.0, 0
    cuda_categories = Counter()
    for event in events:
        cat = event.get("cat", "")
        if cat not in {"kernel", "gpu_memcpy", "gpu_memset"} or event.get("ph") != "X":
            continue
        cuda_categories[cat] += 1
        all_gpu_us += event["dur"]
        owner = owners.get(event.get("args", {}).get("correlation"))
        if owner is None:
            unmatched_count += 1
            continue
        root, stage = owner
        duration = event["dur"]
        attributed_us += duration
        kernels[event["name"]] += duration
        stages[stage] += duration
        stage_kernels[stage][event["name"]] += duration
        category = kernel_category(event["name"]) if cat == "kernel" else cat
        categories[category] += duration
        activity[root].append((event["ts"], event["ts"]+duration))
    if not attributed_us:
        raise ValueError("No GPU launch correlations mapped to decode; inspect trace format")
    per_cycle = []
    for root in roots:
        intervals = activity.get(root["name"], [])
        if not intervals:
            continue
        envelope = max(b for a,b in intervals)-min(a for a,b in intervals)
        active = interval_union(intervals)
        per_cycle.append({"name": root["name"], "gpu_envelope_ms": envelope/1000,
                          "gpu_active_union_ms": active/1000, "gaps_inside_envelope_ms": (envelope-active)/1000})
    n = len(roots)
    return {"annotated_full_batch_decode_cycles": n, "cycles_with_attributed_gpu": len(per_cycle),
            "gpu_categories_in_entire_trace": dict(cuda_categories),
            "entire_trace_gpu_activity_ms": all_gpu_us/1000,
            "attributed_full_batch_decode_gpu_activity_ms": attributed_us/1000,
            "gpu_activities_outside_selected_roots_or_unattributed": unmatched_count,
            "stage_gpu_sum_ms_per_cycle": {k:v/1000/n for k,v in sorted(stages.items())},
            "stage_top_kernels": {stage: [{"name": k, "ms_per_cycle": v/1000/n}
                                          for k,v in sorted(values.items(), key=lambda kv:-kv[1])[:10]]
                                  for stage, values in sorted(stage_kernels.items())},
            "kernel_category_gpu_sum_ms_per_cycle": {k:v/1000/n for k,v in sorted(categories.items())},
            "cpu_cuda_api_ms_per_cycle": {k:v/1000/n for k,v in sorted(cpu_api.items(),key=lambda kv:-kv[1])},
            "top_kernels": [{"name": k, "total_ms": v/1000, "ms_per_cycle": v/1000/n}
                            for k,v in sorted(kernels.items(),key=lambda kv:-kv[1])[:30]],
            "cycle_activity": per_cycle,
            "limitations": ["Only launch-correlated GPU activities inside annotated C-matched decode roots are attributed.",
                            "The entire trace may also contain prefill and lower-batch decode; its total is not the selected-cycle denominator.",
                            "CPU API time overlaps GPU execution; runtime/driver APIs may also nest. Do not add these durations.",
                            "GPU envelope excludes leading/trailing gaps; kernel category labels are heuristics, exact names retained.",
                            "Instrumented trace timing is not clean serving throughput."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as stream:
        data = json.load(stream)
    result = analyze(data["traceEvents"], args.batch_size)
    result["source_trace"] = str(args.trace)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k:v for k,v in result.items() if k not in {"top_kernels", "stage_top_kernels", "cycle_activity"}}, indent=2))


if __name__ == "__main__":
    main()
