"""Audit selected verification length against the serving response cycle counts."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path


def summarize(root):
    audited = defaultdict(lambda: Counter(cycles=0, selected=0, accepted=0))
    hist = defaultdict(Counter)
    physical = defaultdict(Counter)
    mixed = []
    for path in sorted(root.glob("*/verify_lengths.jsonl")):
        for line in path.open():
            row = json.loads(line)
            phases = {x["rid"].rsplit(":",1)[0] for x in row["requests"]}
            if len(phases) == 1:
                phase = next(iter(phases))
                physical[phase].update(batches=1, packed=row["packed_tokens"], executed=row["executed_tokens"],
                                       all_cycles=len(row["requests"]), graph_batches=int(row["cuda_graph"]))
            else:
                mixed.append(sorted(phases))
            for x in row["requests"]:
                if x["counted"]:
                    audited[x["rid"]].update(cycles=1, selected=x["verify_length"], accepted=x["accepted"])
                    hist[x["rid"]][x["verify_length"]] += 1
    result = {}
    for path in sorted(root.glob("*/c*_r*.json")):
        response = json.loads(path.read_text())
        phase = f"pdv:{response['case']}_c{response['concurrency']}_r{response['repeat']}"
        totals = Counter()
        histogram = Counter()
        for item in response["results"]:
            meta = item["response"]["meta_info"]
            rid = f"{phase}:{item['prompt_id']}"
            assert meta["id"] == rid
            counts = audited[rid]
            assert counts["cycles"] == meta["spec_verify_ct"], (rid, "cycles", counts, meta["spec_verify_ct"])
            assert counts["accepted"] == meta["spec_num_correct_drafts"], (rid, "accepted")
            totals.update(counts)
            histogram.update(hist[rid])
        assert totals["cycles"] > 0
        n = totals["cycles"]
        phys = physical[phase]
        mixed_for_phase = [x for x in mixed if phase in x]
        result[phase] = dict(totals) | {
            "average_verification_length_including_anchor": totals["selected"]/n,
            "average_verified_proposals": (totals["selected"]-n)/n,
            "mean_accepted_drafts": totals["accepted"]/n,
            "accepted_over_selected_proposals": totals["accepted"]/(totals["selected"]-n),
            "verify_length_histogram": {str(k):histogram[k] for k in range(1,17)},
            "physical": dict(phys), "mixed_phase_batches": len(mixed_for_phase),
            "executed_rows_per_counted_cycle": phys["executed"]/n if not mixed_for_phase else None,
            "packed_rows_per_counted_cycle": phys["packed"]/n if not mixed_for_phase else None,
            "requests_audited": len(response["results"])}
    return {"schema": "postdraft_verify_length_audit_v1", "phases": result,
            "definition": "Selected length includes anchor; mean weighted by counted request-cycles, excluding warmup and finished/retracted overlap results. Physical work includes discarded overlap cycles. Not a throughput benchmark."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    summary = summarize(a.run)
    (a.run / "verify_length_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
