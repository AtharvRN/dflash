import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.check_ragged_primitives import acceptance_case, load_worker
from scripts.recover_legacy_ragged import digest
from scripts.summarize_ragged_audit import aggregate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vendor/sglang_ragged_20260723"


def test_recovery_bytes_match_provenance():
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    assert digest(PACKAGE / "base-python.tar.gz") == manifest["base_archive_sha256"]
    assert len(manifest["overlay"]) == 16
    for rel, row in manifest["overlay"].items():
        assert digest(PACKAGE / "overlay" / rel) == row["sha256"]


def test_original_eager_acceptance_all_lengths_and_rejections():
    worker = load_worker(PACKAGE / "overlay", "cpu")
    lengths = [b for b in range(1, 17) for _ in range(b)]
    accepts = [a for b in range(1, 17) for a in range(b)]
    assert len(lengths) == 136
    acceptance_case(worker, lengths, accepts, seed=991)


def test_summary_does_not_hide_top1_disagreement():
    rows = [{"bitwise_equal": False, "max_abs": .01, "relative_l2": .001,
             "top1_mismatches": 1, "tokens": 16}]
    result = aggregate(rows)
    assert result["top1_mismatches"] == 1
    assert not result["all_bitwise_equal"]
    assert result["logit_token_positions"] == 16


def test_shadow_forward_preserves_inplace_embedding_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("DFLASH_RAGGED_AUDIT_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location("test_audit_runtime", ROOT / "scripts/ragged_audit_hook/audit_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def original(fb, *args, **kwargs):
        fb.input_embeds.add_(10)
        return SimpleNamespace(logits_output=SimpleNamespace(hidden_states=fb.input_embeds.clone(), next_token_logits=None),
                               can_run_graph=False)

    # The perturbation control also changes token IDs outside the protected row.
    def view(fb, rows, lengths, physical):
        offsets = [0]
        for n in physical:
            offsets.append(offsets[-1]+n)
        ids = [offsets[i]+j for i in rows for j in range(lengths[i])]
        return SimpleNamespace(input_embeds=fb.input_embeds[ids].clone(), input_ids=fb.input_ids[ids].clone())

    monkeypatch.setattr(module, "batch_view", view)
    fb = SimpleNamespace(input_embeds=torch.ones(5, 4), input_ids=torch.arange(5), batch_size=2,
                         forward_mode=SimpleNamespace(is_target_verify=lambda: True),
                         spec_info=SimpleNamespace(draft_token_num=3, draft_token_lens=torch.tensor([2, 3]), graph_draft_token_lens=None),
                         seq_lens=torch.tensor([12, 34]))
    result = module.audited_forward(SimpleNamespace(forward=original), "draft")(fb)
    assert torch.equal(result.logits_output.hidden_states, torch.full((5, 4), 11.0))
    records = [json.loads(x) for x in next(tmp_path.glob("audit_*.jsonl")).read_text().splitlines()]
    assert records[0]["restored_hidden"]["bitwise_equal"]
    assert all(row["hidden"]["bitwise_equal"] for row in records[0]["requests"])
    assert records[0]["isolation_hidden"]["bitwise_equal"]
    tied = torch.ones(16, 64)
    assert module.compare(tied, tied, logits=True)["top1_mismatches"] == 0
    assert module.stop_for_hidden_difference(.062)
    assert not module.stop_for_hidden_difference(.062, outlier_capture=True)
    assert module.stop_for_hidden_difference(.908, outlier_capture=True)
    actual, reference = torch.ones(3, 8), torch.ones(3, 8)
    actual[2, 5] = 9
    detail = module.hidden_detail(actual, reference, width=4)
    assert (detail["max_error_position"], detail["max_error_channel"]) == (2, 5)
    assert detail["feature_groups"][0]["bitwise_equal"]
    assert detail["actual_at_max_error"] == 9
    slots_fb = SimpleNamespace(req_pool_indices=torch.tensor([0, 1]), seq_lens=torch.tensor([2, 2]),
                               out_cache_loc=torch.tensor([3, 6]))
    slots_runner = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[1, 2, 3], [4, 5, 6]])))
    assert module.slot_audit(slots_runner, slots_fb)["output_overlapping_any_prefix"] == 0
    slots_fb.out_cache_loc[0] = 4
    try:
        module.slot_audit(slots_runner, slots_fb)
        assert False, "Overlapping writes must fail the audit"
    except AssertionError as error:
        assert "Unsafe shadow/cache slot mapping" in str(error)

    # Shadow projection must not create inference tensors in production caches.
    class Worker:
        target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=None)))

        def _greedy_sample_from_vocab_parallel_head(self, hidden_states, lm_head):
            self.buffer = torch.empty(hidden_states.shape[0], dtype=torch.long)
            return self.buffer.zero_()

    worker = Worker()
    with torch.inference_mode():
        module.project_without_inference_buffers(worker, torch.ones(257, 4))
    assert not worker.buffer.is_inference()
    worker.buffer.fill_(42)


def test_saved_state_comparison_counts_bonus_and_hidden_error(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("saved_state_diagnostic", ROOT / "scripts/diagnose_saved_ragged_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokens = torch.tensor([4, 1, 2])
    logits = torch.zeros(3, 5)
    logits[0, 1], logits[1, 2], logits[2, 3] = 1, 1, 1
    hidden = torch.ones(3, 2560)
    report = module.summarize(hidden, logits, tokens, hidden.clone(), logits.clone())
    assert report["A"] == 2 and report["bonus"] == 3
    assert report["hidden"]["bitwise_equal"]
    assert report["top1_mismatches"] == 0
    changed = hidden.clone()
    changed[2, 123] = 10
    assert module.compare(changed, hidden)["max_abs"] == 9
    state = {"input_ids": tokens, "positions": torch.arange(42, 45), "prefix_length": 42}
    ids, positions = module.replay_queries(state, batch=48, block=12, device="cpu")
    assert ids.shape == positions.shape == (48, 12)
    assert torch.equal(ids[:, :3], tokens[None].expand(48, -1))
    assert not ids[:, 3:].any()
    assert torch.equal(positions[0], torch.arange(42, 54))
    state["positions"] += 1
    try:
        module.replay_queries(state, batch=1, block=3, device="cpu")
        assert False, "Replay must not silently change real positions"
    except ValueError as error:
        assert "Saved positions" in str(error)


def test_outlier_fixture_matches_executed_not_only_real_token_rows():
    spec = importlib.util.spec_from_file_location("outlier_fixture", ROOT / "scripts/ragged_audit_hook/saved_outlier_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = {"input_ids": [1]*15, "packed_input_tokens": 630, "packed_batch_size": 63,
             "captured_batch_size": 64, "graph_used": True, "num_tokens_per_batch": 10}
    assert module.uniform_shape(state, 64) == ((40, 16), 640)
    state.update(input_ids=[1]*7, graph_used=False)
    assert module.uniform_shape(state, 64) == ((63, 10), 630)
