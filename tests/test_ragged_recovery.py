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
