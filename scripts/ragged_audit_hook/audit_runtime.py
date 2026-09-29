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
DIAGNOSTICS = os.environ.get("DFLASH_RAGGED_AUDIT_DIAGNOSTICS") == "1"
OUTLIER_CAPTURE = os.environ.get("DFLASH_RAGGED_OUTLIER_CAPTURE") == "1"
CAPTURE_EMISSION = os.environ.get("DFLASH_RAGGED_CAPTURE_EMISSION") == "1"
AUDIT_ROLES = set(os.environ.get("DFLASH_RAGGED_AUDIT_ROLES", "draft,target").split(","))


def stop_for_hidden_difference(relative_l2, outlier_capture=False):
    # The normal 5% diagnostic gate is unchanged. Explicit outlier investigation
    # continues through smaller, fully reported discrepancies to preserve a
    # severe recurrence; its completion is NOT a correctness pass.
    return relative_l2 > (0.5 if outlier_capture else 0.05)


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


def hidden_detail(actual, reference, width=2560):
    """Localize discrepancies without letting large feature norms hide them."""
    a, b = actual.float(), reference.float()
    flat = int((a-b).abs().argmax())
    position, channel = divmod(flat, a.shape[-1])
    return {"actual_absmax": float(a.abs().max()),
            "reference_absmax": float(b.abs().max()),
            "max_error_position": position, "max_error_channel": channel,
            "actual_at_max_error": float(a[position, channel]),
            "reference_at_max_error": float(b[position, channel]),
            "per_token_relative_l2": ((a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)).tolist(),
            "feature_groups": [compare(a[:, i:i+width], b[:, i:i+width])
                               for i in range(0, a.shape[-1], width)]}


def slot_audit(runner, fb):
    """Check the shadow writes cannot overlap any live committed prefix."""
    table = runner.req_to_token_pool.req_to_token
    prefixes = torch.cat([table[int(req), :int(n)] for req, n in
                          zip(fb.req_pool_indices.tolist(), fb.seq_lens.tolist())]).long()
    output = fb.out_cache_loc.long()
    report = {"prefix_slots": len(prefixes), "output_slots": len(output),
              "duplicate_output_slots": len(output)-len(output.unique()),
              "output_overlapping_any_prefix": int(torch.isin(output, prefixes).sum())}
    if report["duplicate_output_slots"] or report["output_overlapping_any_prefix"]:
        raise AssertionError(f"Unsafe shadow/cache slot mapping: {report}")
    return report


def project_without_inference_buffers(worker, hidden):
    # The audit runs *inside* the draft-forward inference_mode context, whereas
    # the normal worker projects afterwards, outside it. Growing its persistent
    # projection buffers inside inference_mode would poison their next normal
    # out= update. Keep these test-created buffers ordinary no-grad tensors.
    with torch.inference_mode(False), torch.no_grad():
        return worker._greedy_sample_from_vocab_parallel_head(
            hidden_states=hidden, lm_head=worker.target_worker.model_runner.model.lm_head)


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
    largest_saved_error = 0.0
    near_miss_snapshots = 0

    @wraps(original)
    def forward(fb, *args, **kwargs):
        nonlocal checked, largest_saved_error, near_miss_snapshots
        spec = fb.spec_info
        should_check = (role in AUDIT_ROLES and fb.forward_mode.is_target_verify() and
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
                  "outlier_capture_only": OUTLIER_CAPTURE,
                  "emission_capture_only": CAPTURE_EMISSION,
                  "reference": "independent actual-B, same committed KV prefix", "requests": []}
        graph_runner = getattr(runner, "decode_cuda_graph_runner", None)
        record["captured_batch_size"] = (getattr(graph_runner, "bs", None) if out.can_run_graph else None)
        record["input_tokens"] = int(fb.input_ids.numel())
        reference_inputs = copy.copy(fb)
        if saved_embeds is not None:
            record["input_embedding_mutation"] = compare(saved_embeds, fb.input_embeds)
            reference_inputs.input_embeds = saved_embeds
        if DIAGNOSTICS:
            record["slot_audit"] = slot_audit(runner, fb)
            record["prefix_lengths_cpu"] = fb.seq_lens_cpu.tolist() if fb.seq_lens_cpu is not None else None
            record["prefix_lengths_sum_field"] = fb.seq_lens_sum
            record["packed_input_ids"] = fb.input_ids.tolist()
            record["packed_positions"] = fb.positions.tolist()
            record["req_pool_indices"] = fb.req_pool_indices.tolist()
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
        worst_reference = None
        emission_reference = None
        worst_error = -1
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
                    if CAPTURE_EMISSION and emission_reference is None and not row["acceptance"]["same_emitted_tokens"]:
                        emission_reference = (i, ref.logits_output.hidden_states.clone(),
                                              ref.logits_output.next_token_logits.clone())
                if project is not None:
                    ref_ids = project(ref.logits_output.hidden_states[1:]).clone()
                    actual_ids = actual_draft_ids[draft_offsets[i]:draft_offsets[i+1]]
                    row["draft_tokens"] = {"tokens": len(ref_ids),
                                           "mismatches": int((actual_ids != ref_ids).sum())}
                if DIAGNOSTICS and row["hidden"]["relative_l2"] > worst_error:
                    worst_error = row["hidden"]["relative_l2"]
                    worst_reference = (i, ref.logits_output.hidden_states.clone(),
                                       ref.logits_output.next_token_logits.clone() if actual_logits is not None else None)
                if DIAGNOSTICS and row["hidden"]["relative_l2"] > 0.05:
                    row["hidden_detail"] = hidden_detail(actual_hidden[begin:end], ref.logits_output.hidden_states)
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
            diagnostic_reference = emission_reference or worst_reference
            protected = diagnostic_reference[0] if DIAGNOSTICS else checked % fb.batch_size
            record["diagnostic_focus_reason"] = "emitted_tokens" if emission_reference is not None else "worst_hidden"
            first, last = offsets[protected], offsets[protected]+real[protected]
            protected_hidden = ref.logits_output.hidden_states[first:last].clone()
            protected_logits = (ref.logits_output.next_token_logits[first:last].clone()
                                if actual_logits is not None else None)
            if DIAGNOSTICS:
                focus_label = "emission_row" if emission_reference is not None else "worst_row"
                record[f"{focus_label}_same_batch_eager_hidden"] = compare(actual_hidden[first:last], protected_hidden)
                record[f"{focus_label}_same_batch_eager_detail"] = hidden_detail(actual_hidden[first:last], protected_hidden)
                # Preserve the precise worst case for analysis after a deliberate
                # failure. This is a bounded local artifact, never training data.
                retain_near_miss = (OUTLIER_CAPTURE and worst_error > max(0.1, largest_saved_error)
                                    and near_miss_snapshots < 4)
                if stop_for_hidden_difference(worst_error, OUTLIER_CAPTURE) or retain_near_miss or emission_reference is not None:
                    req = int(fb.req_pool_indices[protected])
                    prefix_n = int(fb.seq_lens[protected])
                    prefix_slots = runner.req_to_token_pool.req_to_token[req, :prefix_n].long()
                    path = ROOT / f"discrepancy_{role}_{checked}_{os.getpid()}.pt"
                    payload = {"format_version": 2, "role": role, "number": checked, "row": protected,
                               "prefix_length": prefix_n,
                               "packed_input_tokens": int(fb.input_ids.numel()),
                               "packed_batch_size": fb.batch_size,
                               "captured_batch_size": record["captured_batch_size"],
                               "graph_used": record["graph_used"],
                               "num_tokens_per_batch": int(spec.num_tokens_per_batch),
                               "real_lengths": real, "physical_lengths": physical,
                               "physical_input_ids": fb.input_ids[first:offsets[protected+1]].cpu(),
                               "input_ids": fb.input_ids[first:last].cpu(),
                               "positions": fb.positions[first:last].cpu(),
                               "actual_hidden": actual_hidden[first:last].cpu(),
                               "independent_hidden": diagnostic_reference[1].cpu(),
                               "same_batch_eager_hidden": protected_hidden.cpu(),
                               "actual_logits": actual_logits[first:last].cpu() if actual_logits is not None else None,
                               "independent_logits": diagnostic_reference[2].cpu() if diagnostic_reference[2] is not None else None,
                               "same_batch_eager_logits": protected_logits.cpu() if protected_logits is not None else None,
                               "capture_reason": record["diagnostic_focus_reason"],
                               "prefix_kv": runner.token_to_kv_pool.get_cpu_copy(prefix_slots)}
                    torch.save(payload, path)
                    largest_saved_error = max(largest_saved_error, worst_error)
                    near_miss_snapshots += int(retain_near_miss)
                    record["discrepancy_artifact"] = str(path)
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
            if DIAGNOSTICS:
                repeated = original(batch_view(reference_inputs, [protected], real, physical))
                record["independent_repeat_hidden"] = compare(diagnostic_reference[1], repeated.logits_output.hidden_states)
                if diagnostic_reference[2] is not None:
                    record["independent_repeat_logits"] = compare(diagnostic_reference[2], repeated.logits_output.next_token_logits, logits=True)
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
        maximum_error = max(r["hidden"]["relative_l2"] for r in record["requests"])
        record["normal_5pct_gate_exceeded"] = stop_for_hidden_difference(maximum_error)
        emit(record)
        # A substantial hidden discrepancy is a correctness failure, not a
        # throughput observation. Small BF16/top-1 differences remain visible
        # in the report and require diagnosis, never silently called parity.
        if stop_for_hidden_difference(maximum_error, OUTLIER_CAPTURE):
            raise AssertionError("Large same-state packed-vs-independent discrepancy; inspect audit")
        if not record["isolation_hidden"]["bitwise_equal"]:
            raise AssertionError("Changing other requests changed the protected request; inspect isolation audit")
        if emission_reference is not None:
            raise AssertionError("Preserved an emitted-token disagreement for exact-state replay; not a parity pass")
        return restored

    return forward


def install(module):
    ROOT.mkdir(parents=True, exist_ok=True)
    cls = module.DFlashWorkerV2
    original_init = cls.__init__

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("DFLASH_SAVED_TARGET_FIXTURE"):
            if os.environ.get("DFLASH_SAVED_FIXTURE_EXTENDED") == "1":
                from saved_outlier_fixture import run
            else:
                from saved_state_fixture import run
            run(self.target_worker.model_runner, Path(os.environ["DFLASH_SAVED_TARGET_FIXTURE"]), ROOT)
        def project(hidden):
            return project_without_inference_buffers(self, hidden)

        self.draft_model_runner.forward = audited_forward(self.draft_model_runner, "draft", project)
        self.target_worker.model_runner.forward = audited_forward(self.target_worker.model_runner, "target")
        emit({"kind": "installation", "worker": module.__file__,
              "worker_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "limit_per_role": LIMIT,
              "outlier_capture_only": OUTLIER_CAPTURE, "emission_capture_only": CAPTURE_EMISSION,
              "audited_roles": sorted(AUDIT_ROLES)})

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
