import json
import sys
from types import SimpleNamespace

import pytest

from scripts.screen_fixed_regimes import PHASES, components, summarize_events
from scripts import screen_fixed_regimes as screen


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


def test_workload_never_uses_saved_assistant_answer(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "USER:"+messages[0]["content"]+"ASSISTANT:"

        def encode(self, text, **kwargs):
            return list(text.encode())

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    data = tmp_path/"data/dflashv2_data"
    (data/"manifests").mkdir(parents=True)
    (data/"manifests/qwen3_4b_instruct_100k_messages.jsonl").write_text("{}")
    candidates = [{"manifest_index": i, "source": "test", "content_sha256": str(i),
                   "messages": [{"role": "user", "content": "x"*1100},
                                {"role": "assistant", "content": "DO_NOT_USE_SAVED_ANSWER"}]}
                  for i in range(128)]
    monkeypatch.setattr(screen, "select_training_prompts", lambda *a: candidates)
    result = screen.make_workload({k: {"target": {"path": k, "revision": k}} for k in ("dense", "moe")})
    assert len(result["rows"]) == 128
    for model in ("dense", "moe"):
        for length, ids in result["rows"][0]["input_ids"][model].items():
            assert len(ids) == int(length)
            assert bytes(ids).decode() == "USER:"+"x"*(int(length)-15)+"ASSISTANT:"


def test_wave_uses_explicit_neutral_greedy_settings(monkeypatch):
    def post(url, json, timeout):
        p = json["sampling_params"]
        assert p["repetition_penalty"] == p["top_p"] == p["top_k"] == 1
        assert p["temperature"] == p["frequency_penalty"] == p["presence_penalty"] == p["min_p"] == 0
        assert p["ignore_eos"]
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: [{"meta_info": {"completion_tokens": 32}}])
    monkeypatch.setattr(screen.requests, "post", post)
    result = screen.wave("http://localhost", [{"prompt_id": 1, "input_ids": {"moe": {"512": [1]*512}}}], "moe", 512, 32)
    assert result["output_tokens"] == 32
