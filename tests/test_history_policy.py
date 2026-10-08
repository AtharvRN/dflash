import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from dflash.history_policy import History, HistoryValueTable, evaluate_rows, select_block


def test_pilot_configurable_blocks():
    from scripts.run_history_pilot import pilot_blocks
    assert pilot_blocks('4,8,12,16') == (4, 8, 12, 16)
    assert pilot_blocks(','.join(map(str, range(2, 17)))) == tuple(range(2, 17))
    for invalid in ('4,8,12', '4,8,16,24', '8,4,16', '4,4,16', '1,16'):
        with pytest.raises(ValueError):
            pilot_blocks(invalid)


def row(pid="1", group="train", accepted=(2, 3), history=None, cycle=0):
    return {"schema_version": 1, "outcome_kind": "actual_redraft", "prompt_id": pid,
            "cycle": cycle, "group": group, "history": history or History().snapshot(),
            "eligible": True, "terminal_or_capped": False,
            "outcomes": {str(b): {"accepted": a} for b, a in zip((4, 8), accepted)}}


def test_history_only_two_previous_cycles_and_full_includes_anchor():
    h = History()
    assert h.snapshot()["certainty"] is None
    h.observe(4, 3, 1.)
    snapshot = h.snapshot()
    h.observe(8, 0, 3.)
    assert h.snapshot()["certainty"] == -2
    assert not h.snapshot()["previous_full"]
    h.observe(4, 2, 5.)
    assert h.snapshot()["certainty"] == -4
    assert snapshot["previous_full"]  # Snapshot is not mutated by future outcomes.


def test_table_learns_actual_nonmonotonic_outcomes():
    # A shorter actual draft can outperform a longer one; do not cummin/clip labels.
    table = HistoryValueTable.fit([row(accepted=(3, 1))], (4, 8), prior_count=0)
    assert table.predict(History().snapshot()) == [4., 2.]
    assert select_block(table, History().snapshot(), mode="retention", alpha=1)[0] == 4
    assert HistoryValueTable(json.loads(json.dumps(table.payload))).predict(History().snapshot()) == [4., 2.]


def test_distinct_history_cells_and_unseen_backoff():
    h = History()
    h.observe(4, 3, .2)
    other = History()
    other.observe(8, 0, 3.)
    table = HistoryValueTable.fit([row(history=h.snapshot(), accepted=(1, 7)),
                                  row("2", history=other.snapshot(), accepted=(3, 3))],
                                 (4, 8), bins=2, prior_count=0)
    assert table.predict(h.snapshot()) == [2., 8.]
    assert table.predict(other.snapshot()) == [4., 4.]
    assert table.predict(History().snapshot()) == [3., 6.]
    _, values = select_block(table, h.snapshot(), mode="retention", alpha=.96, history_free=True)
    assert values == [3., 6.]


def test_pricing_requires_real_costs_and_has_deterministic_ties():
    table = HistoryValueTable.fit([row()], (4, 8))
    assert select_block(table, History().snapshot(), mode="priced", costs_ms={4:1., 8:2.}, rho=1)[0] == 4
    for kwargs in ({}, {"costs_ms": {4: 1}, "rho": 1},
                   {"costs_ms": {4: 1, 8: float("nan")}, "rho": 1}):
        with pytest.raises(ValueError):
            select_block(table, History().snapshot(), mode="priced", **kwargs)


@pytest.mark.parametrize("change", [{"group": "assessment"}, {"eligible": False},
    {"terminal_or_capped": True}, {"outcome_kind": "clipped"}, {"outcomes": {"4": {"accepted": 2}}}])
def test_fit_rejects_leakage_censoring_and_missing_actual_arms(change):
    with pytest.raises(ValueError):
        HistoryValueTable.fit([{**row(), **change}], (4, 8))


def test_evaluation_uses_aggregate_ratio_and_disjoint_prompts():
    table = HistoryValueTable.fit([row()], (4, 8))
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_rows(table, [row(group="assessment")], mode="retention", alpha=.96)
    result = evaluate_rows(table, [row("10", "assessment", (0, 7)), row("11", "assessment", (3, 2))],
                           mode="retention", alpha=.5)
    assert result["adaptive"]["aggregate_accept_ratio"] == .5  # (0+3)/(3+3)
    assert result["adaptive"]["retention_vs_largest"] == pytest.approx(1/3)


def test_duplicate_rows_and_invalid_values_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        HistoryValueTable.fit([row(), row()], (4, 8))
    for entropy in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            History().observe(4, 2, entropy)
    with pytest.raises(ValueError):
        History().observe(4, 4, .1)


def test_entropy_does_not_read_beyond_first_rejection():
    torch = pytest.importorskip("torch")
    from dflash.history_generate import valid_verifier_entropy
    logits = torch.tensor([[[0., 0.], [20., -20.], [-20., 20.], [0., 0.]]])
    assert valid_verifier_entropy(logits, 0) == pytest.approx(math.log(2))
    assert valid_verifier_entropy(logits, 1) == pytest.approx(math.log(2)/2)
    changed = logits.clone()
    changed[:, 2:] = float("nan")
    assert valid_verifier_entropy(changed, 1) == valid_verifier_entropy(logits, 1)


def tiny_models():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from dflash.model import DFlashDraftModel
    torch.manual_seed(5)
    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=256, attention_dropout=0.)
    target_config = Qwen3Config(**common, num_hidden_layers=3)
    target_config._attn_implementation = "eager"
    target = Qwen3ForCausalLM(target_config).eval()
    cfg = Qwen3Config(**common, num_hidden_layers=1)
    cfg._attn_implementation = "eager"
    cfg.num_target_layers, cfg.block_size = 3, 8
    cfg.dflash_config = {"target_layer_ids": [1], "mask_token_id": 31}
    return target, DFlashDraftModel(cfg).eval()


def test_real_tiny_models_probes_do_not_change_closed_loop_or_history():
    import torch
    from dflash.history_generate import generate_history
    target, draft = tiny_models()
    ids = torch.tensor([[2, 7, 9]])
    choose = lambda history, cycle: (4, 8)[cycle % 2]
    kwargs = dict(blocks=(4, 8), choose=choose, max_new_tokens=30)
    plain = generate_history(draft, target, ids, **kwargs)
    paired = generate_history(draft, target, ids, collect=True, **kwargs)
    assert torch.equal(plain["output_ids"], paired["output_ids"])
    assert [r["history"] for r in plain["cycles"]] == [r["history"] for r in paired["cycles"]]
    assert all(set(r["outcomes"]) == {"4", "8"} for r in paired["cycles"])
    assert paired["cycles"][-1]["terminal_or_capped"]
    assert paired["cycles"][0]["history"] == History().snapshot()
    with torch.inference_mode():
        ar = ids.clone()
        for _ in range(30):
            ar = torch.cat((ar, target(ar).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(ar, plain["output_ids"])


def test_prefill_eos_and_one_token_limit_do_not_draft():
    import torch
    from dflash.history_generate import generate_history
    target, draft = tiny_models()
    ids = torch.tensor([[2, 7, 9]])
    eos = int(target(ids).logits[:, -1].argmax(-1).item())
    def forbidden(*args):
        raise AssertionError("no drafting after terminal anchor")
    for kwargs in ({"max_new_tokens": 1}, {"max_new_tokens": 20, "stop_token_ids": [eos]}):
        result = generate_history(draft, target, ids, blocks=(4, 8), choose=forbidden, **kwargs)
        assert result["output_ids"].shape[1] == 4
        assert result["cycles"] == []


def test_fit_cli_roundtrip(tmp_path):
    from scripts.dflash_history import digest
    run = tmp_path/"run"
    run.mkdir()
    (run/"config.json").write_text(json.dumps({"command": "collect", "model_identity": {"test": "tiny"}}))
    (run/"cycles.jsonl").write_text(json.dumps(row())+"\n")
    (run/"summary.json").write_text(json.dumps({"cycles_sha256": digest(run/"cycles.jsonl")}))
    table = tmp_path/"table.json"
    script = Path(__file__).resolve().parents[1]/"scripts/dflash_history.py"
    subprocess.run([sys.executable, str(script), "fit", "--runs", str(run), "--blocks", "4,8", "--output", str(table)], check=True)
    assert json.loads(table.read_text())["fit_rows"] == 1
    result = subprocess.run([sys.executable, str(script), "fit", "--runs", str(run), "--blocks", "4,8", "--output", str(table)], capture_output=True)
    assert result.returncode != 0


def test_prompt_selection_preserves_groups_and_excludes_cross_group_duplicates(tmp_path):
    from scripts.dflash_history import select_prompts
    (tmp_path/"train_prompt_ids.json").write_text(json.dumps({"train_prompt_ids": [0, 1]}))
    (tmp_path/"val_prompt_ids.json").write_text(json.dumps({"val_prompt_ids": [2, 3, 4]}))
    pilot = {"shards": [{"prompt_id": p, "group": g, "rows": 1} for p,g in
                         [(2, "calibration"), (3, "assessment"), (4, "assessment")]]}
    (tmp_path/"pilot.json").write_text(json.dumps(pilot))
    rows = [{"manifest_index": i, "messages": [{"role": "user", "content": x}]} for i,x in
            enumerate(["train-only", "train-val-copy", "cal-assess-copy", "cal-assess-copy", "train-val-copy"])]
    (tmp_path/"prompts.jsonl").write_text("\n".join(map(json.dumps, rows)))
    args = (tmp_path/"prompts.jsonl", tmp_path)
    assert [r["manifest_index"] for r in select_prompts(*args, "train", tmp_path/"pilot.json", 10, 0)] == [0]
    assert select_prompts(*args, "calibration", tmp_path/"pilot.json", 10, 0) == []
    assert select_prompts(*args, "assessment", tmp_path/"pilot.json", 10, 0) == []


def test_arbitrary_integer_sizes_and_cold_start():
    r = row()
    r["outcomes"] = {"3": {"accepted": 1}, "7": {"accepted": 5}, "17": {"accepted": 6}}
    table = HistoryValueTable.fit([r], (3, 7, 17))
    assert select_block(table, History().snapshot(), mode="retention", alpha=.8)[0] == 7


def test_closed_loop_adaptive_table_with_real_tiny_models():
    import torch
    from dflash.history_generate import generate_history
    target, draft = tiny_models()
    ids = torch.tensor([[2, 7, 9]])
    collected = generate_history(draft, target, ids, blocks=(4, 8), choose=lambda h,c: (4,8)[c%2],
                                 max_new_tokens=40, collect=True, prompt_id="train", group="train")
    table = HistoryValueTable.fit([r for r in collected["cycles"] if r["eligible"]], (4,8))
    result = generate_history(draft, target, ids, blocks=(4, 8),
                              choose=lambda h,c: select_block(table, h, mode="retention", alpha=.96)[0],
                              max_new_tokens=30)
    with torch.inference_mode():
        ar = ids.clone()
        for _ in range(30):
            ar = torch.cat((ar, target(ar).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(result["output_ids"], ar)
