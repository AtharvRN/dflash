"""Diagnostic-only FA2 plan ablations. Never changes the benchmark's backend.

Independent wrappers vary query tiling and split-KV, using identical logical
inputs. The uniform-query control uses the installed FlashInfer 0.6.18 internal
planner API, whose signature is checked explicitly. It is NOT a serving fix.
"""
from __future__ import annotations

import inspect
import numpy as np
import torch


PLAN_FIELDS = (
    'padded_batch_size', 'total_num_rows', 'total_num_rows_offset', 'cta_tile_q',
    'request_indices_offset', 'qo_tile_indices_offset', 'kv_tile_indices_offset',
    'merge_indptr_offset', 'o_indptr_offset', 'kv_chunk_size_ptr_offset',
    'v_offset', 's_offset', 'block_valid_mask_offset', 'enable_cuda_graph', 'split_kv',
)


def decode_plan(plan):
    if len(plan) != len(PLAN_FIELDS):
        raise ValueError('Unsupported FlashInfer FA2 plan layout')
    return dict(zip(PLAN_FIELDS, map(int, plan)))


def dense_causal_reference(q, k, v, scale, k_scale=1., v_scale=1.):
    """FP64 reference for already-RoPE'd, stored BF16 Q/K/V (NHD, one request)."""
    nq, nh, dim = q.shape
    nk, nkh, kd = k.shape
    if dim != kd or nh % nkh or nq > nk or v.shape != k.shape:
        raise ValueError('Unsupported Q/K/V reference shape')
    q = q.double().transpose(0, 1)
    k = k.double().repeat_interleave(nh // nkh, dim=1).transpose(0, 1)
    v = v.double().repeat_interleave(nh // nkh, dim=1).transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)) * float(scale) * float(k_scale)
    causal = torch.arange(nk, device=q.device)[None, :] <= (
        nk - nq + torch.arange(nq, device=q.device)[:, None])
    scores.masked_fill_(~causal[None], float('-inf'))
    return (scores.softmax(-1) @ (v * float(v_scale))).transpose(0, 1)


def make_wrapper(template, updater, graph, no_split, uniform):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    qo = template._qo_indptr_buf.clone()
    ptr = template._paged_kv_indptr_buf.clone()
    indices = template._paged_kv_indices_buf[:int(ptr[-1])].clone()
    last = template._paged_kv_last_page_len_buf.clone()
    assert template._custom_mask_buf is None, 'Only ordinary causal verification'
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=qo.device)
    buffers = dict(qo_indptr_buf=qo.clone(), paged_kv_indptr_buf=ptr.clone(),
                   paged_kv_indices_buf=indices.clone(), paged_kv_last_page_len_buf=last.clone()) if graph else {}
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace, 'NHD', use_cuda_graph=graph, backend='fa2', **buffers)
    wrapper.plan(qo, ptr, indices, last, updater.num_qo_heads, updater.num_kv_heads,
                 updater.head_dim, 1, q_data_type=updater.q_data_type,
                 kv_data_type=updater.data_type, disable_split_kv=no_split)
    if uniform:
        lengths = qo[1:] - qo[:-1]
        if not torch.all(lengths == uniform):
            raise ValueError('Uniform-query diagnostic requires genuinely uniform lengths')
        # Exact call signature from this installed version's public plan method.
        source = inspect.getsource(BatchPrefillWithPagedKVCacheWrapper.plan)
        if 'args.append(0)  # uniform_q_len' not in source:
            raise ValueError('Unrecognized FlashInfer uniform-query planner API')
        wrapper._plan_info = wrapper._cached_module.plan(
            wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
            wrapper._pin_memory_int_workspace_buffer, qo.cpu(), ptr.cpu(),
            (ptr[1:] - ptr[:-1]).cpu(), int(qo[-1]), len(qo)-1,
            updater.num_qo_heads, updater.num_kv_heads, 1, graph,
            updater.head_dim, updater.head_dim, False, -1, -1, no_split, 0, uniform)
    torch.cuda.synchronize()
    decode_plan(wrapper._plan_info)
    return wrapper


@torch.no_grad()
def investigate(case, fb, idx, eager_h, graph_h, target_range, difference, save):
    """Ablate plans, then compare kernels on identical layer-local Q/K/V."""
    from sglang.srt.layers.attention.flashinfer_backend import PrefillMetadata
    from dflash.midverify_latency import accepted_from_top1
    runner, backend = case.s.runner, case.s.runner.attn_backend
    saved_metadata = backend.forward_metadata
    backend.init_forward_metadata(fb)
    template = backend.forward_metadata.prefill_wrappers[0]
    lengths = template._qo_indptr_buf[1:] - template._qo_indptr_buf[:-1]
    if not torch.all(lengths == lengths[0]):
        return {'skipped': 'Plan-factorial control requires a uniform-B case'}
    uniform = int(lengths[0])
    row_errors = (graph_h.float()-eager_h.float()).norm(dim=-1) / eager_h.float().norm(dim=-1).clamp_min(1e-12)
    packed = int(row_errors.argmax())
    request, position = divmod(int(idx[packed]), 16)
    qo = template._qo_indptr_buf.clone()
    ptr = template._paged_kv_indptr_buf.clone()
    slots = template._paged_kv_indices_buf[int(ptr[request]):int(ptr[request+1])].long().clone()
    lo, hi = int(qo[request]), int(qo[request+1])
    report = {'scope': 'Diagnosis only: independent uncaptured wrappers; no timing claims',
              'selected_request': request, 'selected_query_position': position,
              'selected_source_row': case.s.rows[request]['row'],
              'selected_prompt_id': case.s.rows[request]['prompt_id'],
              'selected_cycle': case.s.rows[request]['cycle'],
              'selected_prefix_length': case.s.lengths[request],
              'reference': 'FP64 attention on identical stored BF16 Q/K/V, not FP64 whole-model inference',
              'plans': {}, 'full_forward': {}, 'same_input_attention': []}
    specs = [('eager', False, False, 0), ('graph_default', True, False, 0),
             ('graph_no_split', True, True, 0),
             ('graph_uniform', True, False, uniform),
             ('graph_uniform_no_split', True, True, uniform)]
    wrappers, base_layers, handles = {}, [], []
    try:
        for name, graph, no_split, width in specs:
            wrappers[name] = make_wrapper(template, backend.indices_updater_prefill, graph, no_split, width)
            report['plans'][name] = decode_plan(wrappers[name]._plan_info)
        save(report)
        base_top = None
        for name, wrapper in wrappers.items():
            layer_deltas = []
            def hook(_module, _inputs, output):
                h, residual = output
                combined = h.float() + residual.float() if residual is not None else h.float()
                layer = len(base_layers) if name == 'eager' else len(layer_deltas)
                if name == 'eager':
                    base_layers.append(combined.clone())
                else:
                    layer_deltas.append({'layer': layer,
                        'all_rows': difference(combined, base_layers[layer]),
                        'selected_row': difference(combined[packed], base_layers[layer][packed])})
            handles = [layer.register_forward_hook(hook) for layer in runner.model.model.layers]
            backend.forward_metadata = PrefillMetadata([wrapper], False, False)
            out = target_range(runner, fb, 0, 36)
            dense = np.full((case.s.n, 16), -1, dtype=np.int64)
            dense.reshape(-1)[idx.cpu().numpy()] = out[2].cpu().numpy()
            accepted, bonus = accepted_from_top1(case.last['candidate_blocks'], dense, case.last['end'])
            if base_top is None:
                base_top, base_a, base_b = dense, accepted, bonus
            report['full_forward'][name] = {
                'vs_original_eager_hidden': difference(out[0], eager_h),
                'vs_original_graph_hidden': difference(out[0], graph_h),
                'top1_differences_vs_eager': int((dense != base_top).sum()),
                'acceptance_differences_vs_eager': int((accepted != base_a).sum()),
                'bonus_differences_vs_eager': int((bonus != base_b).sum()),
                'layers_vs_eager': layer_deltas}
            for handle in handles:
                handle.remove()
            handles = []
            save(report)
        # Execute the eager model path once more, but inspect every attention
        # invocation with the other plans on exactly the SAME q and KV tensors.
        wrapper = wrappers['eager']
        original_forward = wrapper.forward
        def checked_attention(q, kv, **kwargs):
            output = original_forward(q, kv, **kwargs)
            assert kwargs.get('causal') is True
            assert kwargs.get('window_left', -1) == -1
            assert not kwargs.get('logits_soft_cap')
            assert isinstance(kv, tuple) and len(kv) == 2
            k, v = [x[slots].reshape(len(slots), backend.indices_updater_prefill.num_kv_heads, q.shape[-1]) for x in kv]
            reference = dense_causal_reference(q[lo:hi], k, v, kwargs['sm_scale'],
                kwargs.get('k_scale') or 1., kwargs.get('v_scale') or 1.)
            row = {'layer': len(report['same_input_attention']), 'kernels': {},
                   'nearest_bf16_reference_error': difference(reference.to(q.dtype), reference),
                   'selected_query_norm': float(q[packed].float().norm())}
            for key, other in wrappers.items():
                actual = output if key == 'eager' else other.forward(q, kv, **kwargs)
                row['kernels'][key] = {'vs_eager_all': difference(actual, output),
                    'vs_fp64_request': difference(actual[lo:hi], reference),
                    'vs_fp64_selected_row': difference(actual[packed], reference[position])}
            report['same_input_attention'].append(row)
            return output
        wrapper.forward = checked_attention
        try:
            backend.forward_metadata = PrefillMetadata([wrapper], False, False)
            result = target_range(runner, fb, 0, 36)
            report['instrumented_eager_vs_original_hidden'] = difference(result[0], eager_h)
        finally:
            wrapper.forward = original_forward
        report['complete'] = True
        save(report)
        return report
    finally:
        for handle in handles:
            handle.remove()
        backend.forward_metadata = saved_metadata
