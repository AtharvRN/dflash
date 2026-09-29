"""Replay a preserved outlier with matched GEMM rows and original graph layout."""
import json
from types import SimpleNamespace

import torch


def uniform_shape(state, capacity):
    real = len(state["input_ids"])
    total = int(state["packed_input_tokens"])
    if state["graph_used"]:
        ntpb = state.get("num_tokens_per_batch")
        if ntpb is None:
            if total % state["packed_batch_size"]:
                raise ValueError("Cannot infer graph token bucket")
            ntpb = total // state["packed_batch_size"]
        total = int(state["captured_batch_size"]) * int(ntpb)
    candidates = [(total//block, block) for block in range(real, 33)
                  if total % block == 0 and 1 <= total//block <= capacity]
    if not candidates:
        raise ValueError(f"No bounded dense shape for real B={real}, execution rows={total}")
    return candidates[0], total


@torch.inference_mode(False)
@torch.no_grad()
def run(runner, artifact, output):
    from audit_runtime import compare, hidden_detail, acceptance_summary
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput, DFlashRaggedVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if state.get("format_version") != 2 or state["role"] != "target":
        raise ValueError("Expected a version-2 target snapshot")
    if list(runner.dflash_target_layer_ids) != [1, 9, 17, 25, 33]:
        raise ValueError("Target capture layers differ from the saved state")
    pool, allocator = runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    if allocator.page_size != 1:
        raise ValueError("Only page-size-1 fixtures are implemented")
    capacity = min(64, pool.available_size())
    (dense_bs, dense_b), execution_rows = uniform_shape(state, capacity)
    n, real = state["prefix_length"], len(state["input_ids"])
    # Dummies are suffix-only target queries, never newly proposed draft tokens.
    cases = [("single_actual", [real], "fixed", 0, False),
             ("single_padded", [dense_b], "fixed", 0, False),
             ("dense_matched_rows", [dense_b]*dense_bs, "fixed", 0, False),
             ("ragged_uniform_matched_rows", [dense_b]*dense_bs, "ragged", 0, False),
             ("dense_b16", [16]*capacity, "fixed", 0, False),
             ("original_layout_eager", state["physical_lengths"], "original", state["row"], False)]
    if runner.decode_cuda_graph_runner is not None:
        cases.append(("original_layout_graph", state["physical_lengths"], "original", state["row"], True))
    max_bs = max(len(lens) for _, lens, _, _, _ in cases)
    max_block = max(max(lens) for _, lens, _, _, _ in cases)
    max_queries = max(sum(lens) for _, lens, _, _, _ in cases)
    if max_bs > capacity or n+max_block > pool.req_to_token.shape[1] or real > 16:
        raise ValueError("Snapshot exceeds bounded fixture capacity")
    before = (pool.available_size(), allocator.available_size())
    requests = [SimpleNamespace(req_pool_idx=None) for _ in range(max_bs)]
    reqs = pool.alloc(requests)
    if reqs is None:
        raise RuntimeError("No request slots available")
    slots = allocator.alloc(n+max_queries)
    if slots is None:
        for req in requests:
            pool.free(req)
        raise RuntimeError("No KV slots available")
    device = slots.device
    req_indices = torch.tensor(reqs, dtype=torch.int64, device=device)
    saved_table = pool.req_to_token[req_indices, :n+max_block].clone()
    reports, tensors = {}, {}
    try:
        allocator.load_cpu_copy(state["prefix_kv"], slots[:n])
        pool.req_to_token[req_indices, :n] = slots[:n].int()[None]
        for label, lengths, metadata, focus, allow_graph in cases:
            batch, total = len(lengths), sum(lengths)
            blocks, positions, offsets = [], [], [0]
            for i, length in enumerate(lengths):
                ids = torch.zeros(length, device=device, dtype=torch.long)
                ids[:min(real, length)] = state["input_ids"][:min(real, length)].to(device)
                if metadata == "original" and i == focus:
                    ids.copy_(state["physical_input_ids"].to(device))
                blocks.append(ids)
                positions.append(torch.arange(n, n+length, device=device))
                start = offsets[-1]
                pool.req_to_token[req_indices[i], n:n+length] = slots[n+start:n+start+length].int()
                offsets.append(start+length)
            ids, positions = torch.cat(blocks), torch.cat(positions)
            ntpb = total//batch if metadata == "original" else lengths[0]
            spec_kwargs = dict(draft_token=ids, positions=positions, draft_token_num=max(lengths),
                               num_tokens_per_batch=ntpb, capture_hidden_mode=CaptureHiddenMode.FULL)
            if metadata == "fixed":
                spec = DFlashVerifyInput(**spec_kwargs)
                spec.disable_cuda_graph = True
            else:
                semantic = state["real_lengths"] if metadata == "original" else lengths
                spec = DFlashRaggedVerifyInput(**spec_kwargs,
                    draft_token_lens=torch.tensor(semantic, device=device, dtype=torch.int32),
                    graph_draft_token_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
                    total_draft_tokens=total, disable_cuda_graph=not allow_graph)
            fb = ForwardBatch(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=batch,
                input_ids=ids, positions=positions, req_pool_indices=req_indices[:batch],
                seq_lens=torch.full((batch,), n, dtype=torch.int32, device=device),
                seq_lens_cpu=torch.full((batch,), n, dtype=torch.int32), seq_lens_sum=batch*n,
                out_cache_loc=slots[n:n+total], spec_info=spec, spec_algorithm=SpeculativeAlgorithm.DFLASH,
                capture_hidden_mode=CaptureHiddenMode.FULL)
            result = runner.forward(fb)
            if bool(result.can_run_graph) != allow_graph:
                raise AssertionError(f"Unexpected graph execution in {label}")
            first, last = offsets[focus], offsets[focus]+real
            hidden = result.logits_output.hidden_states[first:last].cpu()
            logits = result.logits_output.next_token_logits[first:last].cpu()
            tensors[label] = {"hidden": hidden, "logits": logits}
            row = {"batch_size": batch, "physical_lengths": lengths, "focus": focus,
                   "query_tokens": total, "graph_used": bool(result.can_run_graph),
                   "captured_batch_size": runner.decode_cuda_graph_runner.bs if allow_graph else None,
                   "acceptance": acceptance_summary(state["input_ids"], logits, state["actual_logits"])}
            for name in ("actual", "independent", "same_batch_eager"):
                row[f"vs_saved_{name}_hidden"] = compare(hidden, state[f"{name}_hidden"])
                row[f"vs_saved_{name}_logits"] = compare(logits, state[f"{name}_logits"], logits=True)
            base = tensors["single_actual"]
            row["vs_single_hidden"] = compare(hidden, base["hidden"])
            row["vs_single_detail"] = hidden_detail(hidden, base["hidden"])
            row["vs_single_logits"] = compare(logits, base["logits"], logits=True)
            if label == "ragged_uniform_matched_rows":
                row["vs_dense_same_shape_hidden"] = compare(hidden, tensors["dense_matched_rows"]["hidden"])
                row["vs_dense_same_shape_logits"] = compare(logits, tensors["dense_matched_rows"]["logits"], logits=True)
            reports[label] = row
            (output / "FIXTURE_PARTIAL.json").write_text(json.dumps(reports, indent=2)+"\n")
            del result
        prefix_after = allocator.get_cpu_copy(slots[:n])
        prefix_unchanged = (len(prefix_after) == len(state["prefix_kv"]) and
            all(len(a) == len(b) and all(torch.equal(x, y) for pair_a, pair_b in zip(a, b)
                for x, y in zip(pair_a, pair_b)) for a, b in zip(prefix_after, state["prefix_kv"])))
        if not prefix_unchanged:
            raise AssertionError("Fixture mutated committed prefix KV")
    finally:
        pool.req_to_token[req_indices, :n+max_block] = saved_table
        allocator.free(slots)
        for req in requests:
            pool.free(req)
        after = (pool.available_size(), allocator.available_size())
        if before != after:
            raise AssertionError(f"Fixture leaked allocations: {before} -> {after}")
    torch.save(tensors, output / "fixture_outputs.pt")
    (output / "FIXTURE.json").write_text(json.dumps({"artifact": str(artifact), "reports": reports,
        "original_execution_rows": execution_rows, "uniform_shape": [dense_bs, dense_b],
        "prefix_unchanged": prefix_unchanged, "pool_sizes_before": before, "pool_sizes_after": after,
        "limits": "One preserved prefix repeated across requests; not the full original batch or a throughput test."}, indent=2)+"\n")
