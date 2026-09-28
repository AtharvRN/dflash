"""Report clean throughput separately from instrumented component spans."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def compare_outputs(reference, row):
    a, b = reference["requests"], row["requests"]
    if [x["prompt_id"] for x in a] != [x["prompt_id"] for x in b]:
        raise ValueError("Unmatched workload order")
    differences = [{"request_index": i, "prompt_id": x["prompt_id"]}
                   for i, (x, y) in enumerate(zip(a, b))
                   if x["response"]["output_ids"] != y["response"]["output_ids"]]
    return {"requests": len(a), "different_token_sequences": len(differences),
            "difference_rows": differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    grouped = defaultdict(list)
    raw = {}
    for path in args.run.glob("*/measured_*.json"):
        label = path.stem
        parts = label.split("_")
        block, concurrency, mode, rep = int(parts[1][1:]), int(parts[2][1:]), parts[3], int(parts[4][1:])
        row = json.loads(path.read_text())
        raw[(block, concurrency, mode, rep)] = row
        if mode == "clean":
            grouped[(concurrency, block)].append(row["output_tokens_per_s"])
    throughput = []
    for (concurrency, block), rates in sorted(grouped.items()):
        ref = grouped.get((concurrency, 16))
        throughput.append({"concurrency": concurrency, "block": block, "repeats": len(rates),
                           "mean_tokens_per_s": statistics.mean(rates),
                           "stdev_tokens_per_s": statistics.stdev(rates) if len(rates)>1 else 0,
                           "min_tokens_per_s": min(rates), "max_tokens_per_s": max(rates),
                           "ratio_to_B16_mean": statistics.mean(rates)/statistics.mean(ref) if ref else None})
    parity = []
    for (block, concurrency, mode, rep), row in sorted(raw.items()):
        reference = raw.get((16 if mode == "clean" else block, concurrency, "clean", rep))
        if reference is None or reference is row:
            continue
        parity.append({"block": block, "concurrency": concurrency, "mode": mode, "repeat": rep,
                       "comparison": "against_B16" if mode == "clean" else "against_clean_same_B",
                       **compare_outputs(reference, row)})
    for (block, concurrency, mode, rep), row in sorted(raw.items()):
        if mode == "clean" and rep > 0:
            reference = raw[(block, concurrency, mode, 0)]
            parity.append({"block": block, "concurrency": concurrency, "mode": mode, "repeat": rep,
                           "comparison": "against_repeat_zero_same_B",
                           **compare_outputs(reference, row)})
    components = []
    for label, strata in summary["components"].items():
        row = strata.get("full_batch_decode")
        if row is None:
            continue
        parts = label.split("_")
        components.append({"concurrency": int(parts[2][1:]), "block": int(parts[1][1:]),
            "cycles": row["cycles"], "prefix_mean": row["prefix_length"]["mean"],
            "cycle_ms_mean": row["stream_cycle_ms"]["mean"], "cycle_ms_stdev": row["stream_cycle_ms"]["stdev"],
            "cycle_ms_median": row["stream_cycle_ms"]["median"], "cycle_ms_p95": row["stream_cycle_ms"]["p95"],
            "accepted_drafts_per_request_cycle": row["accepted_drafts"]/row["request_cycles"],
            "request_cycles": row["request_cycles"],
            "components_ms": {k:v["mean"] for k,v in row["stream_components_ms"].items()},
            "target_graph_fraction": row["target_graph_fraction"],
            "draft_graph_fraction": row["draft_graph_observed_fraction"]})
    report = {"throughput": throughput, "full_batch_components": components, "output_parity": parity}
    (args.run / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    target_only = config["args"].get("target_only", False)
    lines = ["# Serving-cost study", "", f"GPU: {config['gpu']['name']}, index {config['gpu']['index']}.",
             "", "Original Qwen3-4B + original DFlash B16 drafter; BF16, greedy, TP1, Triton attention, CUDA graphs enabled.",
             "Pinned cached SGLang 0.5.13 spec-v1 runtime; no overlap scheduler. This is NOT the historical ragged runtime.",
             "", "## Uninstrumented HTTP output throughput", "",
             "Includes prefill and draining tails. Same prompt order/output cap across block sizes, radix reuse disabled.",
             "Three repetitions where available; the +/- column is sample standard deviation, not a confidence interval.",
             "", "| C | B | Repeats | Tokens/s mean +/- SD | Relative to B16 |", "|---:|---:|---:|---:|---:|"]
    for r in throughput:
        relative = f"{r['ratio_to_B16_mean']:.4f}x" if r["ratio_to_B16_mean"] is not None else "N/A"
        lines.append(f"| {r['concurrency']} | {r['block']} | {r['repeats']} | {r['mean_tokens_per_s']:.2f} +/- {r['stdev_tokens_per_s']:.2f} | {relative} |")
    if target_only:
        lines[4] = "Qwen3-4B target-only AR control; BF16, greedy, TP1, Triton attention, CUDA graphs enabled."
        lines[5] = "Pinned cached SGLang 0.5.13; native overlap scheduling remains enabled, unlike spec-v1 DFlash. B=1 denotes AR here."
    lines += ["", "## Full-batch decode component spans", "",
              "CUDA-stream elapsed milliseconds, including idle gaps while the CPU submits work. Not pure kernel duration.",
              "Instrumented runs are separate from throughput runs. Nested spans are made disjoint before summing.",
              "", "| C | B | Cycles | Mean prefix | Cycle mean +/- SD | Draft | Draft projection | Verify | Other/upkeep |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(components, key=lambda x:(x["concurrency"],x["block"])):
        c = r["components_ms"]
        overhead = sum(v for k,v in c.items() if k not in {"draft_forward","draft_projection_argmax","target_forward_including_logits"})
        lines.append(f"| {r['concurrency']} | {r['block']} | {r['cycles']} | {r['prefix_mean']:.1f} | {r['cycle_ms_mean']:.3f} +/- {r['cycle_ms_stdev']:.3f} | {c['draft_forward']:.3f} | {c['draft_projection_argmax']:.3f} | {c['target_forward_including_logits']:.3f} | {overhead:.3f} |")
    lines += ["", "Detailed disjoint components and host-call timings are in analysis.json and summary.json.",
              "", "## Output checks", ""]
    differences = [r for r in parity if r["different_token_sequences"]]
    if differences:
        lines.append("Some matched requests have different token sequences. These may reflect numerical/execution differences; causes have not yet been isolated. Do not claim bitwise losslessness from this run.")
        lines += [f"- B{r['block']} C{r['concurrency']} {r['mode']} rep{r['repeat']} ({r['comparison']}): {r['different_token_sequences']}/{r['requests']} differ." for r in differences]
    else:
        lines.append("Every matched token sequence agrees for the comparisons recorded in analysis.json.")
    lines += ["", "## Interpretation limits", "",
              "These are fixed-width measurements. A mixed per-request batch with mean B=12 is not necessarily as cheap as fixed B12.",
              "The current predictor, packed drafting, graph-bucket padding and ragged KV remapping are not integrated here; their end-to-end overhead remains unmeasured.",
              "The fixed runs contain no predictor. A separate predictor microbenchmark excludes feature gather and packing; synchronous CPU length readback is separately labeled.",
              "Full-batch cycle rows are correlated and come from policy-dependent trajectories. Mean prefix lengths are reported to expose differences.",
              "This development workload is not an untouched final benchmark. Kernel-trace totals and instrumented throughput must not be substituted for clean serving throughput.",
              "", "## Provenance", "", f"Run: `{args.run}`", f"Image: `{config['image']}`", f"Harness commit: `{config['code_commit']}`", "",
              "Each variant preserves launch arguments, server configuration/logs, per-request results, GPU telemetry and raw event records."]
    (args.run / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k,v in report.items() if k != "output_parity"}, indent=2))


if __name__ == "__main__":
    main()
