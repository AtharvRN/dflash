"""Bounded same-runtime dense/ragged-metadata replay, before serving requests."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch


@torch.inference_mode(False)
@torch.no_grad()
def run(runner, artifact, output):
    from audit_runtime import compare, hidden_detail, acceptance_summary
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput, DFlashRaggedVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if state["role"] != "target" or list(runner.dflash_target_layer_ids) != [1, 9, 17, 25, 33]:
        raise ValueError("Fixture/model capture mismatch")
    n, real = state["prefix_length"], state["input_ids"].numel()
    if real != 7:
        raise ValueError("This fixture explicitly controls the saved B7 case")
    requests = [SimpleNamespace(req_pool_idx=None) for _ in range(63)]
    pool, allocator = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    if allocator.page_size != 1:
        raise ValueError("Fixture requires page size 1")
    before = (pool.available_size(), allocator.available_size())
    reqs = pool.alloc(requests)
    if reqs is None:
        raise RuntimeError("Insufficient free request slots")
    slots = allocator.alloc(n+63*16)
    if slots is None:
        for req in requests:
            pool.free(req)
        raise RuntimeError("Insufficient free KV slots")
    device = slots.device
    req_indices = torch.tensor(reqs, device=device, dtype=torch.int64)
    saved_table = pool.req_to_token[req_indices, :n+16].clone()
    reports, tensors = {}, {}
    try:
        allocator.load_cpu_copy(state["prefix_kv"], slots[:n])
        pool.req_to_token[req_indices, :n] = slots[:n].to(torch.int32)[None]
        for label, batch, block, ragged in (
                ("fixed_c1_b7", 1, 7, False),
                ("fixed_c1_b10", 1, 10, False),
                ("fixed_c63_b10", 63, 10, False),
                ("ragged_metadata_c63_b10", 63, 10, True),
                ("fixed_c1_b16", 1, 16, False),
                ("fixed_c63_b16", 63, 16, False)):
            ids = torch.zeros((batch, block), dtype=torch.long, device=device)
            ids[:, :real] = state["input_ids"].to(device)[None]
            positions = torch.arange(n, n+block, device=device)[None].expand(batch, -1).contiguous()
            out_slots = slots[n:n+batch*block]
            pool.req_to_token[req_indices[:batch], n:n+block] = out_slots.reshape(batch, block).to(torch.int32)
            spec_kwargs = dict(draft_token=ids.flatten(), positions=positions.flatten(),
                               draft_token_num=block, num_tokens_per_batch=block,
                               capture_hidden_mode=CaptureHiddenMode.FULL)
            if ragged:
                spec = DFlashRaggedVerifyInput(**spec_kwargs,
                    draft_token_lens=torch.full((batch,), block, dtype=torch.int32, device=device),
                    total_draft_tokens=batch*block, disable_cuda_graph=True)
            else:
                spec = DFlashVerifyInput(**spec_kwargs)
                spec.disable_cuda_graph = True
            fb = ForwardBatch(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=batch,
                input_ids=ids.flatten(), positions=positions.flatten(), req_pool_indices=req_indices[:batch],
                seq_lens=torch.full((batch,), n, dtype=torch.int32, device=device),
                seq_lens_cpu=torch.full((batch,), n, dtype=torch.int32), seq_lens_sum=batch*n,
                out_cache_loc=out_slots, spec_info=spec, spec_algorithm=SpeculativeAlgorithm.DFLASH,
                capture_hidden_mode=CaptureHiddenMode.FULL)
            result = runner.forward(fb)
            if result.can_run_graph:
                raise AssertionError("Dense fixture was intended to be eager")
            hidden = result.logits_output.hidden_states.reshape(batch, block, -1)[:, :real].cpu()
            logits = result.logits_output.next_token_logits.reshape(batch, block, -1)[:, :real].cpu()
            tensors[label] = {"hidden": hidden[0].clone(), "logits": logits[0].clone()}
            row = {"batch_size": batch, "block_size": block, "query_tokens": batch*block,
                   "metadata": type(spec).__name__,
                   "vs_saved_packed_hidden": compare(hidden[0], state["actual_hidden"]),
                   "vs_saved_independent_hidden": compare(hidden[0], state["independent_hidden"]),
                   "vs_saved_packed_logits": compare(logits[0], state["actual_logits"], logits=True),
                   "acceptance": acceptance_summary(state["input_ids"], logits[0], state["actual_logits"]),
                   "replica_logits_bitwise_equal": all(torch.equal(logits[0], value) for value in logits)}
            base = tensors["fixed_c1_b7"]
            row["vs_fixed_c1_b7_hidden"] = compare(hidden[0], base["hidden"])
            row["vs_fixed_c1_b7_hidden_detail"] = hidden_detail(hidden[0], base["hidden"])
            row["vs_fixed_c1_b7_logits"] = compare(logits[0], base["logits"], logits=True)
            if ragged:
                row["vs_fixed_same_shape_hidden"] = compare(hidden[0], tensors["fixed_c63_b10"]["hidden"])
                row["vs_fixed_same_shape_logits"] = compare(logits[0], tensors["fixed_c63_b10"]["logits"], logits=True)
            reports[label] = row
            (output / "FIXTURE_PARTIAL.json").write_text(json.dumps(reports, indent=2)+"\n")
            del result, hidden, logits
    finally:
        pool.req_to_token[req_indices, :n+16] = saved_table
        allocator.free(slots)
        for req in requests:
            pool.free(req)
        after = (pool.available_size(), allocator.available_size())
        if after != before:
            raise AssertionError(f"Fixture leaked allocation: {before} -> {after}")
    torch.save(tensors, output / "fixture_outputs.pt")
    (output / "FIXTURE.json").write_text(json.dumps({"artifact": str(artifact), "reports": reports,
        "pool_sizes_before": before, "pool_sizes_after": after,
        "limits": "One same-prefix numerical fixture, not end-to-end correctness or throughput."}, indent=2)+"\n")
