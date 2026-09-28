import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    verified_id_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    verified_id_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    verified_id = tl.load(verified_id_ptr + row * verified_id_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, verified_id.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _prepare_dflash_draft_block_unchecked(
    verified_id: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(verified_id.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        verified_id,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        verified_id.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _prepare_dflash_ragged_draft_block_kernel(
    verified_id_ptr,
    prefix_lens_ptr,
    block_lens_ptr,
    offsets_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    projection_indices_out_ptr,
    verified_id_stride,
    prefix_lens_stride,
    block_lens_stride,
    offsets_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    req_to_token_width,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    block_len = tl.load(block_lens_ptr + row * block_lens_stride).to(tl.int64)
    offset = tl.load(offsets_ptr + row * offsets_stride).to(tl.int64)
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(tl.int64)
    verified_id = tl.load(verified_id_ptr + row * verified_id_stride).to(tl.int64)

    valid = cols < block_len
    logical_pos = prefix_len + cols
    flat_idx = offset + cols

    token_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    token_ids = tl.where(cols == 0, verified_id, token_ids)
    tl.store(block_ids_out_ptr + flat_idx, token_ids, mask=valid)
    tl.store(positions_out_ptr + flat_idx, logical_pos, mask=valid)

    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    cache_loc = tl.load(
        req_row_ptr + logical_pos,
        mask=valid & (logical_pos < req_to_token_width),
        other=0,
    )
    tl.store(cache_loc_out_ptr + flat_idx, cache_loc.to(tl.int64), mask=valid)

    proj_valid = (cols > 0) & valid
    # Compact all non-first positions by removing one token per preceding row.
    proj_idx = offset - row + cols - 1
    tl.store(projection_indices_out_ptr + proj_idx, flat_idx, mask=proj_valid)


def _prepare_dflash_ragged_draft_block_unchecked(
    verified_id: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_lens: torch.Tensor,
    offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    projection_indices_out: torch.Tensor,
    mask_token_id: int,
    max_block_size: int,
) -> None:
    batch_size = int(block_lens.numel())
    if batch_size == 0:
        return
    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError(
            "DFLASH Triton ragged prepare_block requires row-major req_to_token."
        )
    for name, tensor in (
        ("block_ids_out", block_ids_out),
        ("positions_out", positions_out),
        ("cache_loc_out", cache_loc_out),
        ("projection_indices_out", projection_indices_out),
    ):
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError(
                f"DFLASH Triton ragged prepare_block requires contiguous {name}."
            )

    block = triton.next_power_of_2(int(max_block_size))
    num_warps = _pick_num_warps(block)
    _prepare_dflash_ragged_draft_block_kernel[(batch_size,)](
        verified_id,
        prefix_lens,
        block_lens,
        offsets,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        projection_indices_out,
        verified_id.stride(0),
        prefix_lens.stride(0),
        block_lens.stride(0),
        offsets.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        int(req_to_token.shape[1]),
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _prepare_dflash_padded_ragged_verify_kernel(
    draft_tokens_ptr,
    prefix_lens_ptr,
    block_lens_ptr,
    graph_lens_ptr,
    offsets_ptr,
    graph_offsets_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    graph_draft_tokens_out_ptr,
    graph_positions_out_ptr,
    graph_cache_loc_out_ptr,
    graph_real_indices_out_ptr,
    prefix_lens_stride,
    block_lens_stride,
    graph_lens_stride,
    offsets_stride,
    graph_offsets_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    req_to_token_width,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    block_len = tl.load(block_lens_ptr + row * block_lens_stride).to(tl.int64)
    graph_len = tl.load(graph_lens_ptr + row * graph_lens_stride).to(tl.int64)
    offset = tl.load(offsets_ptr + row * offsets_stride).to(tl.int64)
    graph_offset = tl.load(graph_offsets_ptr + row * graph_offsets_stride).to(tl.int64)
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(tl.int64)

    graph_valid = cols < graph_len
    real_valid = cols < block_len
    real_idx = offset + cols
    graph_idx = graph_offset + cols
    logical_pos = prefix_len + cols

    token_ids = tl.load(draft_tokens_ptr + real_idx, mask=real_valid, other=mask_token_id)
    token_ids = tl.where(real_valid, token_ids, mask_token_id)
    tl.store(graph_draft_tokens_out_ptr + graph_idx, token_ids, mask=graph_valid)
    tl.store(graph_positions_out_ptr + graph_idx, logical_pos, mask=graph_valid)

    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    cache_loc = tl.load(
        req_row_ptr + logical_pos,
        mask=graph_valid & (logical_pos < req_to_token_width),
        other=0,
    )
    tl.store(graph_cache_loc_out_ptr + graph_idx, cache_loc.to(tl.int64), mask=graph_valid)
    tl.store(graph_real_indices_out_ptr + real_idx, graph_idx, mask=real_valid)


def _prepare_dflash_padded_ragged_verify_unchecked(
    draft_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_lens: torch.Tensor,
    graph_lens: torch.Tensor,
    offsets: torch.Tensor,
    graph_offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    graph_draft_tokens_out: torch.Tensor,
    graph_positions_out: torch.Tensor,
    graph_cache_loc_out: torch.Tensor,
    graph_real_indices_out: torch.Tensor,
    mask_token_id: int,
    max_block_size: int,
) -> None:
    batch_size = int(block_lens.numel())
    if batch_size == 0:
        return
    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError(
            "DFLASH Triton padded ragged verify requires row-major req_to_token."
        )
    for name, tensor in (
        ("draft_tokens", draft_tokens),
        ("block_lens", block_lens),
        ("graph_lens", graph_lens),
        ("offsets", offsets),
        ("graph_offsets", graph_offsets),
        ("graph_draft_tokens_out", graph_draft_tokens_out),
        ("graph_positions_out", graph_positions_out),
        ("graph_cache_loc_out", graph_cache_loc_out),
        ("graph_real_indices_out", graph_real_indices_out),
    ):
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError(
                f"DFLASH Triton padded ragged verify requires contiguous 1D {name}."
            )

    block = triton.next_power_of_2(int(max_block_size))
    num_warps = _pick_num_warps(block)
    _prepare_dflash_padded_ragged_verify_kernel[(batch_size,)](
        draft_tokens,
        prefix_lens,
        block_lens,
        graph_lens,
        offsets,
        graph_offsets,
        req_pool_indices,
        req_to_token,
        graph_draft_tokens_out,
        graph_positions_out,
        graph_cache_loc_out,
        graph_real_indices_out,
        prefix_lens.stride(0),
        block_lens.stride(0),
        graph_lens.stride(0),
        offsets.stride(0),
        graph_offsets.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        int(req_to_token.shape[1]),
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
