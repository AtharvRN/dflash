"""Conditional planning scenarios; never a substitute for adaptive serving."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def interpolate_cost(block, costs):
    keys = sorted(costs)
    for a, b in zip(keys, keys[1:]):
        if a <= block <= b:
            return costs[a] + (block-a)/(b-a)*(costs[b]-costs[a])
    raise ValueError("No extrapolation outside measured uniform block sizes")


def phase_accounting(run, block):
    label = f"measured_b{block}_c64_events_r0"
    stage = run/f"events_b{block}"
    wall = json.loads((stage/f"{label}.json").read_text())["wall_s"]
    seconds, counts = defaultdict(float), defaultdict(int)
    for path in stage.glob("cycles_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["label"] != label:
                continue
            assert row["spans"][0]["parent"] is None
            seconds[row["phase"]] += row["spans"][0]["stream_elapsed_ms"]/1000
            counts[row["phase"]] += 1
    residual = wall-sum(seconds.values())
    if residual < 0:
        raise ValueError("Worker spans exceed wall time; inspect timing boundaries")
    return {"wall_s": wall, "worker_stream_s": dict(seconds), "cycles": dict(counts),
            "outside_worker_residual_s": residual,
            "decode_wall_fraction": seconds["decode"]/wall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-root", type=Path, required=True)
    parser.add_argument("--predictor-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis_path = args.latency_root/"analysis.json"
    analysis = json.loads(analysis_path.read_text())
    predictor = json.loads(args.predictor_summary.read_text())
    costs = {r["block"]:r["cycle_ms_mean"] for r in analysis["full_batch_components"] if r["concurrency"] == 64}
    base = predictor["controls"]["frozen_100k_mlp"]["1.0"]["mean_accepted"]
    actual = [row["operating_points"]["0.96"] for row in predictor["models"].values() if row["mode"] == "actual"]
    policies = {
        "frozen_100k": predictor["controls"]["frozen_100k_mlp"]["0.96"],
        "actual_three_seed_mean": {"mean_accepted": statistics.mean(r["mean_accepted"] for r in actual),
                                   "mean_budget": statistics.mean(r["mean_budget"] for r in actual)},
        "hypothetical_70pct_ratio_at_96pct_retention": {"mean_accepted": .96*base, "mean_budget": .96*base/.70},
    }
    phases = {str(b):phase_accounting(args.latency_root, b) for b in sorted(costs)}
    fraction = phases["16"]["decode_wall_fraction"]
    scenarios = {}
    for name, policy in policies.items():
        block = policy["mean_budget"]+1
        cost = interpolate_cost(block, costs)
        progress = (policy["mean_accepted"]+1)/(base+1)
        samples = []
        for added in (0, .1, .5, 1, 2, 4, 5):
            speedup = progress*costs[16]/(cost+added)
            samples.append({"added_ms_per_cycle": added, "conditional_decode_speedup": speedup,
                            "conditional_workload_speedup": 1/(1-fraction+fraction/speedup)})
        scenarios[name] = {"mean_B": block, "mean_accepted": policy["mean_accepted"],
            "accepted_retention": policy["mean_accepted"]/base, "advanced_token_fraction": progress,
            "uniform_cost_interpolation_ms": cost, "break_even_added_ms": progress*costs[16]-cost,
            "max_added_ms_for_5pct_decode_gain": progress*costs[16]/1.05-cost,
            "sensitivity": samples}
    out = {"kind": "conditional planning scenarios, not measured adaptive speedups or rigorous bounds",
        "measured_uniform_cycle_cost_ms": costs, "instrumented_wall_accounting": phases,
        "offline_reference_mean_accepted": base, "scenarios": scenarios,
        "assumptions": [
            "Mixed integer-B batches cost the linear interpolation of uniform full-batch C64 measurements at mean B; this is unverified.",
            "Offline B16-state progress retention transfers to policy-induced serving trajectories; this is unverified.",
            "The eight progress-spaced states per prompt are representative of serving cycles; they are not an all-cycle sample.",
            "Decode-only gains transfer from full C64 batches to the whole draining workload; this is unverified.",
            "Prefill and the outside-worker residual stay unchanged in the workload-level calculation.",
            "Added milliseconds include predictor and all incremental feature gathering, packing, padding and KV/remapping costs.",
            "Current models have not achieved 70% aggregate ratio at 96% retention; that row is hypothetical.",
            "These timings describe pinned Triton/spec-v1 on RTX PRO 6000, not an optimized ragged/overlap engine."],
        "sources": {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (analysis_path,args.predictor_summary)}}
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n")
    print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    main()
