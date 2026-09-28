"""Expensive same-state shadow forwards, for correctness ONLY, never timing.

Each packed forward is compared to independent actual-B forwards on the SAME
committed KV prefix. Target graph output is also compared with eager execution
of the same physical packed batch. Finally the original packed forward is rerun
to restore all suffix KV writes before normal decoding continues.
"""
from __future__ import annotations

from functools import wraps
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ["DFLASH_RAGGED_AUDIT_DIR"])
LIMIT = int(os.environ.get("DFLASH_RAGGED_AUDIT_FORWARDS", "12"))
MIN_BATCH = int(os.environ.get("DFLASH_RAGGED_AUDIT_MIN_BATCH", "2"))


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
        actual_top1 = a.argmax(-1)
        ref = b.argmax(-1)
        # topk is not the verifier's argmax tie-breaking rule. Comparing topk's
        # first index with argmax falsely reports differences for equal tensors.
        mismatch = actual_top1 != ref
        result.update({"top1_mismatches": int(mismatch.sum()), "tokens": len(ref),
                       "mismatch_positions": mismatch.nonzero().flatten().tolist(),
                       "mismatch_top1_gaps": (top.values[:, 0]-top.values[:, 1])[mismatch].tolist(),
                       "actual_top1_on_mismatch": actual_top1[mismatch].tolist(),
                       "reference_top1_on_mismatch": ref[mismatch].tolist()})
    return result


def acceptance_summary(tokens, actual_logits, reference_logits):
    def decision(logits):
        pred = logits.argmax(-1)
        accepted = int((tokens[1:] == pred[:-1]).int().cumprod(0).sum())
        return accepted, int(pred[accepted])
    actual, reference = decision(actual_logits), decision(reference_logits)
    return {"actual_A": actual[0], "reference_A": reference[0],
            "actual_bonus": actual[1], "reference_bonus": reference[1],
            "same_emitted_tokens": actual == reference}


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


def audited_forward(runner, role, project=None):
    original = runner.forward
    checked = 0

    @wraps(original)
    def forward(fb, *args, **kwargs):
        nonlocal checked
        spec = fb.spec_info
        should_check = (fb.forward_mode.is_target_verify() and
                        getattr(spec, "draft_token_num", 0) > 0 and
                        fb.batch_size >= MIN_BATCH and checked < LIMIT)
        # DFlash's fused residual RMSNorm mutates the input-embedding buffer.
        # Preserve PRE-forward inputs; references made from fb afterwards are
        # not same-state controls, even though token IDs/KV prefix are identical.
        saved_embeds = (fb.input_embeds.detach().clone()
                        if should_check and fb.input_embeds is not None else None)
        out = original(fb, *args, **kwargs)
        if not should_check:
            return out
        checked += 1
        lens_tensor = getattr(spec, "draft_token_lens", None)
        real = lens_tensor.tolist() if lens_tensor is not None else [int(spec.draft_token_num)]*fb.batch_size
        graph_lens_tensor = getattr(spec, "graph_draft_token_lens", None)
        physical = graph_lens_tensor.tolist() if graph_lens_tensor is not None else real
        actual_hidden = out.logits_output.hidden_states.detach().clone()
        actual_logits = (out.logits_output.next_token_logits.detach().clone()
                         if out.logits_output.next_token_logits is not None else None)
        record = {"kind": "same_state_forward", "role": role, "number": checked,
                  "batch_size": fb.batch_size, "real_lengths": real, "physical_lengths": physical,
                  "prefix_lengths": fb.seq_lens.tolist(), "graph_used": bool(out.can_run_graph),
                  "reference": "independent actual-B, same committed KV prefix", "requests": []}
        graph_runner = getattr(runner, "decode_cuda_graph_runner", None)
        record["captured_batch_size"] = (getattr(graph_runner, "bs", None) if out.can_run_graph else None)
        record["input_tokens"] = int(fb.input_ids.numel())
        reference_inputs = copy.copy(fb)
        if saved_embeds is not None:
            record["input_embedding_mutation"] = compare(saved_embeds, fb.input_embeds)
            reference_inputs.input_embeds = saved_embeds
        # Test all requests, in reverse order to expose hidden batch-order state.
        offsets = [0]
        for size in physical:
            offsets.append(offsets[-1]+size)
        if project is not None:
            projection = torch.tensor([offsets[i]+j for i in range(fb.batch_size) for j in range(1, real[i])],
                                      dtype=torch.int64, device=actual_hidden.device)
            actual_draft_ids = project(actual_hidden.index_select(0, projection)).clone()
            draft_offsets = [0]
            for size in real:
                draft_offsets.append(draft_offsets[-1]+size-1)
        try:
            for i in reversed(range(fb.batch_size)):
                view = batch_view(reference_inputs, [i], real, physical)
                ref = original(view)
                begin, end = offsets[i], offsets[i]+real[i]
                row = {"row": i, "block_size": real[i],
                       "hidden": compare(actual_hidden[begin:end], ref.logits_output.hidden_states)}
                if actual_logits is not None:
                    row["logits"] = compare(actual_logits[begin:end], ref.logits_output.next_token_logits, logits=True)
                    row["acceptance"] = acceptance_summary(
                        fb.input_ids[begin:end], actual_logits[begin:end], ref.logits_output.next_token_logits)
                if project is not None:
                    ref_ids = project(ref.logits_output.hidden_states[1:]).clone()
                    actual_ids = actual_draft_ids[draft_offsets[i]:draft_offsets[i+1]]
                    row["draft_tokens"] = {"tokens": len(ref_ids),
                                           "mismatches": int((actual_ids != ref_ids).sum())}
                record["requests"].append(row)
            # Identical physical layout isolates graphs from shape effects and
            # supplies a matched reference for cross-request attention isolation.
            view = batch_view(reference_inputs, list(range(fb.batch_size)), physical, physical)
            ref = original(view)
            record["same_batch_eager_hidden"] = compare(actual_hidden, ref.logits_output.hidden_states)
            if actual_logits is not None:
                record["same_batch_eager_logits"] = compare(actual_logits, ref.logits_output.next_token_logits, logits=True)
                record["same_batch_eager_acceptance"] = [acceptance_summary(
                    fb.input_ids[offsets[i]:offsets[i]+real[i]], actual_logits[offsets[i]:offsets[i]+real[i]],
                    ref.logits_output.next_token_logits[offsets[i]:offsets[i]+real[i]]) for i in range(fb.batch_size)]
            protected = checked % fb.batch_size
            first, last = offsets[protected], offsets[protected]+real[protected]
            protected_hidden = ref.logits_output.hidden_states[first:last].clone()
            protected_logits = (ref.logits_output.next_token_logits[first:last].clone()
                                if actual_logits is not None else None)
            perturb = batch_view(reference_inputs, list(range(fb.batch_size)), physical, physical)
            other = torch.ones_like(perturb.input_ids, dtype=torch.bool)
            other[offsets[protected]:offsets[protected+1]] = False
            perturb.input_ids[other] = (perturb.input_ids[other]+101) % 10000
            if perturb.input_embeds is not None:
                perturb.input_embeds[other] *= -1
            changed = original(perturb)
            record["isolation_protected_row"] = protected
            record["isolation_hidden"] = compare(protected_hidden, changed.logits_output.hidden_states[first:last])
            if protected_logits is not None:
                record["isolation_logits"] = compare(protected_logits, changed.logits_output.next_token_logits[first:last], logits=True)
        finally:
            # Single-request shadows reuse the same suffix slots. Re-execute the
            # original batch to restore the actual batched KV and graph buffers.
            fb.forward_metadata_ready = False
            if saved_embeds is not None:
                fb.input_embeds = saved_embeds.clone()
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
        if not record["isolation_hidden"]["bitwise_equal"]:
            raise AssertionError("Changing other requests changed the protected request; inspect isolation audit")
        return restored

    return forward


def install(module):
    ROOT.mkdir(parents=True, exist_ok=True)
    cls = module.DFlashWorkerV2
    original_init = cls.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        def project(hidden):
            return self._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden, lm_head=self.target_worker.model_runner.model.lm_head)

        self.draft_model_runner.forward = audited_forward(self.draft_model_runner, "draft", project)
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
