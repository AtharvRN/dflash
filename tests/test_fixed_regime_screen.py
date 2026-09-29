import json

import pytest

from scripts.screen_fixed_regimes import PHASES, components, summarize_events


def cycle(bs=64):
    return {"label": "measured_c64_l512_r1", "batch_size": bs, "block_size": 16,
            "prefix_lens": [600]*bs, "native": {"total_accepted_drafts": 5*bs}, "target_graph": True,
            "spans": [{"name": "decode_cycle", "stream_elapsed_ms": 20, "host_call_ms": 12}]+
                     [{"name": p, "stream_elapsed_ms": 2, "host_call_ms": 1} for p in PHASES]}


def test_disjoint_fixed_v2_accounting():
    values, total = components(cycle())
    assert total == sum(values.values()) == 20
    assert values["other_worker"] == 4
    assert components(cycle(), "host_call_ms")[0]["other_worker"] == 4


def test_reject_nested_or_missing_phases():
    row = cycle()
    row["spans"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        components(row)
    row = cycle()
    row["spans"].append({"name": "postdraft_policy", "stream_elapsed_ms": 2, "host_call_ms": 1})
    with pytest.raises(ValueError, match="Unexpected"):
        components(row)


def test_actual_occupancy_and_aggregate_ratio(tmp_path):
    rows = [cycle() for _ in range(12)]+[cycle(32)]
    rows.append({**cycle(), "label": "warmup_c64_l512_r0"})
    (tmp_path/"cycles_1.jsonl").write_text("\n".join(map(json.dumps, rows))+"\n")
    result = summarize_events(tmp_path)["measured_c64_l512_r1"]
    full = result["full_batch_decode"]
    assert full["cycles"] == 12 and full["usable"]
    assert full["aggregate_ratio"] == pytest.approx(1/3)
    assert result["all_decode"]["cycles"] == 13
