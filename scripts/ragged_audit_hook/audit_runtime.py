"""Expensive same-state shadow forwards, for correctness ONLY, never timing.

Each packed forward is compared to independent actual-B forwards on the SAME
committed KV prefix. Target graph output is also compared with eager execution
of the same physical packed batch. Finally the original packed forward is rerun
to restore all suffix KV writes before normal decoding continues.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ["DFLASH_RAGGED_AUDIT_DIR"])
LIMIT = int(os.environ.get("DFLASH_RAGGED_AUDIT_FORWARDS", "12"))


def emit(row):
    with (ROOT / f"audit_{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def compare(actual, reference, logits=False):
    if actual.shape != reference.shape:
        raise AssertionError(f"Shadow shape mismatch {actual.shape} vs {reference.shape}")
    a, b = actual.float(), reference.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite shadow-forward values")
    delta = a-b
    result = {"shape": list(actual.shape), "max_abs": float(delta.abs().max()),
              "rms": float(delta.square().mean().sqrt()),
              "relative_l2": float(delta.norm()/b.norm().clamp_min(1e-12)),
              "bitwise_equal": bool(torch.equal(actual, reference))}
    if logits:
        top = a.topk(2, dim=-1)
        ref = b.argmax(-1)
        mismatch = top.indices[:, 0] != ref
        result.update({"top1_mismatches": int(mismatch.sum()), "tokens": len(ref),
                       "mismatch_positions": mismatch.nonzero().flatten().tolist(),
                       "mismatch_top1_gaps": (top.values[:, 0]-top.values[:, 1])[mismatch].tolist(),
                       "actual_top1_on_mismatch": top.indices[:, 0][mismatch].tolist(),
                       "reference_top1_on_mismatch": ref[mismatch].tolist()})
    return result


def batch_view(fb, rows, lengths, physical):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.dflash_info import DFlashRaggedVerifyInput
    offsets = [0]
    for size in physical:
        offsets.append(offsets[-1] + size)
    indices = torch.tensor([offsets[i]+j for i in rows for j in range(lengths[i])],
                           dtype=torch.int64, device=fb.input_ids.device)
    row_indices = torch.tensor(rows, dtype=torch.int64, device=fb.input_ids.device)
    ids = fb.input_ids.index_select(0, indices)
    positions = fb.positions.index_select(0, indices)
    seq = fb.seq_lens.index_select(0, row_indices)
    spec = DFlashRaggedVerifyInput(
        draft_token=ids, positions=positions,
        draft_token_lens=torch.tensor([lengths[i] for i in rows], device=seq.device, dtype=torch.int32),
        draft_token_num=max(lengths[i] for i in rows), num_tokens_per_batch=max(lengths[i] for i in rows),
        total_draft_tokens=len(indices), capture_hidden_mode=fb.capture_hidden_mode, disable_cuda_graph=True)
    return ForwardBatch(
        forward_mode=fb.forward_mode, batch_size=len(rows), input_ids=ids,
        req_pool_indices=fb.req_pool_indices.index_select(0, row_indices), seq_lens=seq,
        out_cache_loc=fb.out_cache_loc.index_select(0, indices), seq_lens_sum=int(seq.sum()),
        seq_lens_cpu=seq.cpu(), positions=positions, spec_info=spec,
        spec_algorithm=fb.spec_algorithm, capture_hidden_mode=fb.capture_hidden_mode,
        input_embeds=fb.input_embeds.index_select(0, indices) if fb.input_embeds is not None else None)


def audited_forward(runner, role):
    original = runner.forward
    checked = 0

    @wraps(original)
    def forward(fb, *args, **kwargs):
        nonlocal checked
        spec = fb.spec_info
        should_check = (fb.forward_mode.is_target_verify() and
                        getattr(spec, "draft_token_lens", None) is not None and
                        fb.batch_size > 1 and checked < LIMIT)
        out = original(fb, *args, **kwargs)
        if not should_check:
            return out
        checked += 1
        real = spec.draft_token_lens.tolist()
        physical = (spec.graph_draft_token_lens.tolist()
                    if spec.graph_draft_token_lens is not None else real)
        actual_hidden = out.logits_output.hidden_states.detach().clone()
        actual_logits = (out.logits_output.next_token_logits.detach().clone()
                         if out.logits_output.next_token_logits is not None else None)
        record = {"kind": "same_state_forward", "role": role, "number": checked,
                  "batch_size": fb.batch_size, "real_lengths": real, "physical_lengths": physical,
                  "prefix_lengths": fb.seq_lens.tolist(), "graph_used": bool(out.can_run_graph),
                  "reference": "independent actual-B, same committed KV prefix", "requests": []}
        # Test all requests, in reverse order to expose hidden batch-order state.
        offsets = [0]
        for size in physical:
            offsets.append(offsets[-1]+size)
        try:
            for i in reversed(range(fb.batch_size)):
                view = batch_view(fb, [i], real, physical)
                ref = original(view)
                begin, end = offsets[i], offsets[i]+real[i]
                row = {"row": i, "block_size": real[i],
                       "hidden": compare(actual_hidden[begin:end], ref.logits_output.hidden_states)}
                if actual_logits is not None:
                    row["logits"] = compare(actual_logits[begin:end], ref.logits_output.next_token_logits, logits=True)
                record["requests"].append(row)
            if out.can_run_graph:
                # Identical physical layout: isolate graph replay from ragged
                # vs single-request floating-point shape effects.
                view = batch_view(fb, list(range(fb.batch_size)), physical, physical)
                ref = original(view)
                record["same_batch_eager_hidden"] = compare(actual_hidden, ref.logits_output.hidden_states)
                if actual_logits is not None:
                    record["same_batch_eager_logits"] = compare(actual_logits, ref.logits_output.next_token_logits, logits=True)
        finally:
            # Single-request shadows reuse the same suffix slots. Re-execute the
            # original batch to restore the actual batched KV and graph buffers.
            fb.forward_metadata_ready = False
            restore_kwargs = dict(kwargs)
            restore_kwargs.pop("skip_attn_backend_init", None)
            restored = original(fb, *args, **restore_kwargs)
        record["restored_hidden"] = compare(actual_hidden, restored.logits_output.hidden_states)
        if actual_logits is not None:
            record["restored_logits"] = compare(actual_logits, restored.logits_output.next_token_logits, logits=True)
        emit(record)
        # A substantial hidden discrepancy is a correctness failure, not a
        # throughput observation. Small BF16/top-1 differences remain visible
        # in the report and require diagnosis, never silently called parity.
        if any(r["hidden"]["relative_l2"] > 0.05 for r in record["requests"]):
            raise AssertionError("Large same-state packed-vs-independent discrepancy; inspect audit")
        return restored

    return forward


def install(module):
    ROOT.mkdir(parents=True, exist_ok=True)
    cls = module.DFlashWorkerV2
    original_init = cls.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.draft_model_runner.forward = audited_forward(self.draft_model_runner, "draft")
        self.target_worker.model_runner.forward = audited_forward(self.target_worker.model_runner, "target")
        emit({"kind": "installation", "worker": module.__file__,
              "worker_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "limit_per_role": LIMIT})

    cls.__init__ = initialize
    original_forced = cls._forced_ragged_block_sizes

    def forced(self, bs):
        values = original_forced(self, bs)
        if values is not None and os.environ.get("DFLASH_RAGGED_AUDIT_ROTATE") == "1":
            cycle = getattr(self, "_audit_pattern_cycle", 0)
            self._audit_pattern_cycle = cycle + 1
            values = (values-2+cycle) % 15 + 2
        return values

    cls._forced_ragged_block_sizes = forced
