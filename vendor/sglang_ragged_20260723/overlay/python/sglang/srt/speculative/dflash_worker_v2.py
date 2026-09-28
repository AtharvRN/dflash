import logging
import os
from typing import List, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashRaggedVerifyInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_postdraft_policy import (
    DFlashPostdraftPolicy,
    dflash_confidence_from_logits,
)
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
    _compute_dflash_accept_bonus_triton_unchecked,
)
try:
    from sglang.srt.speculative.triton_ops.dflash_accept_bonus import (
        _compute_dflash_ragged_accept_bonus_triton_unchecked,
    )
except ImportError:
    _compute_dflash_ragged_accept_bonus_triton_unchecked = None
from sglang.srt.speculative.triton_ops.dflash_prepare_block import (
    _prepare_dflash_draft_block_unchecked,
    _prepare_dflash_padded_ragged_verify_unchecked,
    _prepare_dflash_ragged_draft_block_unchecked,
)
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)


class DFlashWorkerV2(DFlashWorker):
    """DFLASH speculative decoding worker (spec-v2 overlap scheduling).

    This is intentionally implemented as a *separate* worker from the existing
    spec-v1 `DFlashWorker` (non-overlap), to keep the v1 path stable and to
    minimize risk while bringing up overlap scheduling.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._accept_len_buf: Optional[torch.Tensor] = None
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []
        self._runtime_block_buffer_cap: dict[int, int] = {}
        self._runtime_block_buffer_slot: dict[int, int] = {}
        self._runtime_block_buffers: dict[int, dict[str, List[torch.Tensor]]] = {}
        self._rel_2d_cache: dict[int, torch.Tensor] = {}
        self._ragged_verify_buffer_cap: int = 0
        self._ragged_verify_req_cap: int = 0
        self._ragged_verify_buffer_slot: int = 0
        self._ragged_verify_buffers: dict[str, List[torch.Tensor]] = {}
        self.postdraft_policy: Optional[DFlashPostdraftPolicy] = None
        if server_args.speculative_dflash_postdraft_policy_path:
            arms = tuple(
                int(x.strip())
                for x in str(server_args.speculative_dflash_postdraft_arms).split(",")
                if x.strip()
            )
            self.postdraft_policy = DFlashPostdraftPolicy(
                checkpoint_path=server_args.speculative_dflash_postdraft_policy_path,
                alpha=float(server_args.speculative_dflash_postdraft_alpha),
                arms=arms,
                device=self.device,
            )

    def _get_runtime_block_sizes(
        self, bs: int, req_pool_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (
            not self.predraft_entropy_ragged
            or self.predraft_entropy_policy is None
        ):
            return torch.full(
                (int(bs),),
                int(self._get_runtime_block_size(bs)),
                dtype=torch.int64,
                device=self.device,
            )

        policy_context = None
        if (
            req_pool_indices is not None
            and self.predraft_entropy_context_by_req is not None
            and self.predraft_entropy_context_valid is not None
            and req_pool_indices.numel() == int(bs)
        ):
            req_idx = req_pool_indices.to(
                device=self.predraft_entropy_context_valid.device, dtype=torch.int64
            )
            if int(req_idx.max().item()) < int(self.predraft_entropy_context_valid.shape[0]):
                valid = self.predraft_entropy_context_valid[req_idx]
                if bool(valid.all().item()):
                    policy_context = self.predraft_entropy_context_by_req[req_idx].to(
                        device=self.device
                    )

        if policy_context is None:
            if (
                self.predraft_entropy_context is None
                or self.predraft_entropy_context.numel() == 0
            ):
                return torch.full(
                    (int(bs),),
                    int(self._get_runtime_block_size(bs)),
                    dtype=torch.int64,
                    device=self.device,
                )
            policy_context = self.predraft_entropy_context

        block_sizes = self.predraft_entropy_policy.request_block_sizes(
            policy_context,
            max_block_size=int(self.block_size),
        )
        if int(block_sizes.numel()) != int(bs):
            fallback = int(
                self.predraft_entropy_policy.select_block_size(
                    policy_context,
                    max_block_size=int(self.block_size),
                )
            )
            logger.warning(
                "DFLASH ragged pre-draft policy batch mismatch; using rectangular "
                "fallback for this cycle. policy_bs=%s batch_bs=%s fallback_block=%s",
                int(block_sizes.numel()),
                int(bs),
                fallback,
            )
            return torch.full(
                (int(bs),),
                fallback,
                dtype=torch.int64,
                device=self.device,
            )
        if bool((block_sizes < 2).any().item()) or bool(
            (block_sizes > int(self.block_size)).any().item()
        ):
            raise RuntimeError(
                "Invalid DFLASH ragged runtime block sizes: "
                f"min={int(block_sizes.min().item())}, max={int(block_sizes.max().item())}, "
                f"allowed=[2,{int(self.block_size)}]."
            )
        self.current_block_size = int(block_sizes.max().item())
        return block_sizes.to(device=self.device, dtype=torch.int64)

    @staticmethod
    def _is_uniform_block_size(block_lens: torch.Tensor) -> bool:
        return bool(
            block_lens.numel() > 0
            and torch.all(block_lens == block_lens.reshape(-1)[0]).item()
        )

    def _forced_ragged_block_sizes(self, bs: int) -> Optional[torch.Tensor]:
        pattern_raw = os.environ.get("SGLANG_DFLASH_FORCE_RAGGED_BLOCK_PATTERN", "")
        if not pattern_raw.strip():
            return None
        pattern = [
            int(x.strip())
            for x in pattern_raw.split(",")
            if x.strip()
        ]
        if not pattern:
            return None
        if any(x < 1 or x > int(self.block_size) for x in pattern):
            raise RuntimeError(
                "Invalid SGLANG_DFLASH_FORCE_RAGGED_BLOCK_PATTERN="
                f"{pattern_raw!r}; expected values in [1,{int(self.block_size)}]."
            )
        values = [pattern[i % len(pattern)] for i in range(bs)]
        return torch.tensor(values, dtype=torch.int64, device=self.device)

    def _postdraft_confidence_from_hidden(
        self,
        *,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        lm_head,
    ) -> torch.Tensor:
        tp_group = get_tp_group()
        if int(tp_group.world_size) != 1:
            raise RuntimeError(
                "DFLASH post-draft confidence features currently require tp-size=1."
            )
        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH post-draft confidence requires vocab-parallel lm_head weight."
            )
        shard = lm_head.shard_indices
        weight = lm_head.weight
        num_org = int(shard.num_org_elements)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        if num_added != 0:
            raise RuntimeError(
                "DFLASH post-draft confidence does not currently support added vocab shards."
            )
        local_ids = (token_ids.to(torch.int64) - org_vocab_start).clamp(
            min=0, max=max(0, num_org - 1)
        )
        logits = torch.matmul(hidden.to(dtype=weight.dtype), weight[:num_org].T)
        return dflash_confidence_from_logits(logits, local_ids)

    def _select_ragged_graph_num_tokens_per_batch(
        self, *, bs: int, total_tokens: int, max_block_size: int
    ) -> int:
        """Choose the smallest captured DFlash graph bucket covering total tokens.

        For packed-ragged draft we want to pad only the flattened token dimension
        to a graph bucket, not every request to `max_block_size`.
        """

        if bs <= 0:
            return max(1, int(max_block_size))
        target_tokens = int(total_tokens)
        fixed_total_raw = os.environ.get("SGLANG_DFLASH_RAGGED_GRAPH_TOTAL_TOKENS", "")
        if fixed_total_raw:
            try:
                fixed_total = int(fixed_total_raw)
            except ValueError:
                fixed_total = 0
            if fixed_total > 0:
                target_tokens = max(target_tokens, fixed_total)
        max_tokens_per_batch = max(int(max_block_size), 1)
        required = max(1, (int(target_tokens) + int(bs) - 1) // int(bs))
        required = min(required, max_tokens_per_batch)
        buckets: List[int] = []
        controller = getattr(self, "dynamic_block_controller", None)
        if controller is not None:
            buckets.extend(int(x) for x in getattr(controller, "arms", []) or [])
        if self.predraft_entropy_policy is not None:
            min_budget = max(
                1,
                int(
                    getattr(
                        self.server_args,
                        "speculative_dflash_predraft_entropy_min_budget",
                        1,
                    )
                ),
            )
            max_budget = min(
                max_tokens_per_batch,
                int(
                    getattr(
                        self.server_args,
                        "speculative_dflash_predraft_entropy_max_budget",
                        max_tokens_per_batch,
                    )
                ),
            )
            if max_budget >= min_budget:
                # The CUDA graph runner captures these integer DFlash buckets for
                # pre-draft entropy policies; selecting from the same range avoids
                # rounding packed ragged batches all the way to the coarse serving
                # arms (for example 13 -> 16).
                buckets.extend(range(min_budget, max_budget + 1))
        if self.postdraft_policy is not None:
            buckets.extend(int(x) for x in getattr(self.postdraft_policy, "arms", []) or [])
        raw_arms = getattr(self.server_args, "speculative_dflash_dynamic_block_arms", "")
        if raw_arms:
            for part in str(raw_arms).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    buckets.append(int(part))
                except ValueError:
                    continue
        buckets.append(int(max_block_size))
        valid = sorted({x for x in buckets if required <= x <= int(max_block_size)})
        return int(valid[0]) if valid else int(max_block_size)

    def _make_ragged_block(
        self,
        *,
        verified_id: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        bs = int(block_lens.numel())
        device = self.device
        max_block = int(block_lens.max().item()) if bs > 0 else 0
        offsets = torch.zeros((bs + 1,), dtype=torch.int64, device=device)
        offsets[1:] = torch.cumsum(block_lens.to(torch.int64), dim=0)
        total = int(offsets[-1].item())

        block_ids = torch.empty((total,), dtype=torch.long, device=device)
        positions = torch.empty((total,), dtype=torch.int64, device=device)
        verify_out_cache_loc = torch.empty((total,), dtype=torch.int64, device=device)
        projection_indices = torch.empty(
            (max(total - bs, 0),), dtype=torch.int64, device=device
        )
        _prepare_dflash_ragged_draft_block_unchecked(
            verified_id=verified_id.to(torch.long),
            prefix_lens=prefix_lens.to(torch.int64),
            block_lens=block_lens.to(torch.int64),
            offsets=offsets,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            block_ids_out=block_ids,
            positions_out=positions,
            cache_loc_out=verify_out_cache_loc,
            projection_indices_out=projection_indices,
            mask_token_id=int(self._mask_token_id),
            max_block_size=max_block,
        )
        return block_ids, positions, verify_out_cache_loc, offsets, projection_indices

    def _make_graph_padded_ragged_verify(
        self,
        *,
        draft_tokens: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_lens: torch.Tensor,
        offsets: torch.Tensor,
        real_total: Optional[int] = None,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        max_block_size: int,
        graph_num_tokens_per_batch: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad packed-ragged verify tensors to a CUDA-graph token bucket.

        Real tokens keep their per-request order. Dummy padding is inserted only
        after a request's real verify segment and is later removed with
        `graph_real_indices` before acceptance/materialization.
        """

        bs = int(block_lens.numel())
        if bs <= 0:
            return (
                draft_tokens,
                torch.empty((0,), dtype=torch.int64, device=self.device),
                torch.empty((0,), dtype=torch.int64, device=self.device),
                block_lens.to(torch.int32),
                torch.empty((0,), dtype=torch.int64, device=self.device),
                1,
            )

        device = self.device
        real_total = int(block_lens.sum().item() if real_total is None else real_total)
        required_ntpb = max(1, (real_total + bs - 1) // bs)
        ntpb = (
            int(graph_num_tokens_per_batch)
            if graph_num_tokens_per_batch is not None
            else required_ntpb
        )
        ntpb = max(required_ntpb, ntpb)
        ntpb = min(int(max_block_size), ntpb)
        graph_total = bs * ntpb
        pad_needed = graph_total - real_total
        if pad_needed <= 0:
            real_lens = block_lens.to(torch.int64)
            max_real_len = int(real_lens.max().item())
            rel_2d = self._get_rel_2d(max_real_len)
            real_valid = rel_2d < real_lens.unsqueeze(1)
            positions = (
                prefix_lens.to(torch.int64).unsqueeze(1) + rel_2d
            ).expand(bs, max_real_len)[real_valid]
            end_offset = prefix_lens + real_lens.to(prefix_lens.dtype)
            out_cache_loc_over = assign_extend_cache_locs_func(
                req_pool_indices=req_pool_indices,
                req_to_token=req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=max_real_len,
                device=device,
            )
            return (
                draft_tokens,
                positions,
                out_cache_loc_over[:real_total].contiguous(),
                block_lens.to(torch.int32),
                self._get_rel_2d(real_total).view(-1),
                ntpb,
            )

        real_lens = block_lens.to(torch.int64)
        slack = (int(max_block_size) - real_lens).clamp_min(0)
        if int(slack.sum().item()) < pad_needed:
            raise RuntimeError(
                "Unable to pad DFLASH ragged verify to graph bucket: "
                f"remaining_pad={pad_needed}, real_total={real_total}, "
                f"bs={bs}, ntpb={ntpb}, max_block_size={max_block_size}."
            )

        # Add dummy graph padding to suffix requests. This matches the previous
        # tail-padding layout but avoids a per-cycle CPU copy/Python loop.
        slack_rev = torch.flip(slack, dims=(0,))
        slack_before_rev = torch.cumsum(slack_rev, dim=0) - slack_rev
        pad_remaining_rev = (int(pad_needed) - slack_before_rev).clamp_min(0)
        pad_add_rev = torch.minimum(slack_rev, pad_remaining_rev)
        graph_lens = real_lens + torch.flip(pad_add_rev, dims=(0,))

        max_graph_len = int(max_block_size)

        (
            graph_draft_tokens,
            graph_positions,
            graph_out_cache_loc,
            graph_real_indices_buf,
            graph_offsets,
        ) = self._next_ragged_verify_buffers(graph_total, real_total, bs)

        graph_offsets[0].zero_()
        torch.cumsum(graph_lens, dim=0, out=graph_offsets[1 : bs + 1])
        graph_real_indices = graph_real_indices_buf[:real_total]
        _prepare_dflash_padded_ragged_verify_unchecked(
            draft_tokens=draft_tokens,
            prefix_lens=prefix_lens.to(torch.int64),
            block_lens=real_lens.contiguous(),
            graph_lens=graph_lens.contiguous(),
            offsets=offsets.contiguous(),
            graph_offsets=graph_offsets[: bs + 1].contiguous(),
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            graph_draft_tokens_out=graph_draft_tokens,
            graph_positions_out=graph_positions,
            graph_cache_loc_out=graph_out_cache_loc,
            graph_real_indices_out=graph_real_indices,
            mask_token_id=int(self._mask_token_id),
            max_block_size=max_graph_len,
        )
        return (
            graph_draft_tokens,
            graph_positions,
            graph_out_cache_loc,
            graph_lens.to(torch.int32),
            graph_real_indices,
            ntpb,
        )

    def _compute_ragged_accept_bonus(
        self,
        *,
        draft_tokens: torch.Tensor,
        target_predict: torch.Tensor,
        block_lens: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = int(block_lens.numel())
        device = self.device
        max_block = int(block_lens.max().item()) if bs > 0 else 0
        if bs == 0:
            empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
            empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
            return empty_i32, empty_i32, empty_i64, empty_i64, empty_i64

        if (
            self._use_triton_accept_bonus
            and _compute_dflash_ragged_accept_bonus_triton_unchecked is not None
        ):
            try:
                accept_len, commit_lens, bonus, out_2d, _ = (
                    self._next_accept_bonus_buffers(bs)
                )
                _compute_dflash_ragged_accept_bonus_triton_unchecked(
                    draft_tokens=draft_tokens.contiguous(),
                    target_top1=target_predict.contiguous(),
                    block_lens=block_lens.to(torch.int64).contiguous(),
                    offsets=offsets.to(torch.int64).contiguous(),
                    accept_lens_out=accept_len,
                    commit_lens_out=commit_lens,
                    bonus_ids_out=bonus,
                    out_tokens_out=out_2d,
                    max_block_size=max_block,
                )
                rel_2d = torch.arange(
                    max_block, dtype=torch.int64, device=device
                ).unsqueeze(0)
                idx_2d = offsets[:-1].to(torch.int64).unsqueeze(1) + rel_2d
                commit_mask = rel_2d < commit_lens.to(torch.int64).unsqueeze(1)
                commit_indices = idx_2d[commit_mask].contiguous()
                out_tokens = out_2d[:, :max_block][commit_mask].contiguous()
                return accept_len, commit_lens, bonus, out_tokens, commit_indices
            except Exception as e:
                self._use_triton_accept_bonus = False
                logger.warning(
                    "DFLASH Triton ragged accept/bonus failed; falling back to eager path: %s",
                    e,
                )

        rel_2d = torch.arange(max_block, dtype=torch.int64, device=device).unsqueeze(0)
        idx_2d = offsets[:-1].unsqueeze(1) + rel_2d
        valid_2d = rel_2d < block_lens.to(torch.int64).unsqueeze(1)
        idx_2d = idx_2d.masked_fill(~valid_2d, 0)
        draft_2d = draft_tokens[idx_2d].to(torch.int64)
        pred_2d = target_predict[idx_2d].to(torch.int64)

        if max_block > 1:
            rel_cmp = torch.arange(max_block - 1, dtype=torch.int64, device=device)
            cmp_valid = rel_cmp.unsqueeze(0) < (
                block_lens.to(torch.int64).unsqueeze(1) - 1
            )
            cmp_match = (draft_2d[:, 1:] == pred_2d[:, :-1]) & cmp_valid
            accept_len_i64 = cmp_match.to(torch.int64).cumprod(dim=1).sum(dim=1)
        else:
            accept_len_i64 = torch.zeros((bs,), dtype=torch.int64, device=device)

        commit_lens_i64 = accept_len_i64 + 1
        bonus = pred_2d[
            torch.arange(bs, dtype=torch.int64, device=device), accept_len_i64
        ]

        if max_block > 1:
            accept_src_idx = torch.clamp(rel_2d + 1, max=max_block - 1).expand(
                bs, max_block
            )
            accept_tokens = torch.gather(draft_2d, 1, accept_src_idx)
        else:
            accept_tokens = draft_2d
        out_2d = torch.where(
            rel_2d < accept_len_i64.unsqueeze(1), accept_tokens, bonus.unsqueeze(1)
        )
        commit_mask = rel_2d < commit_lens_i64.unsqueeze(1)
        out_tokens = out_2d[commit_mask].contiguous()
        commit_indices = idx_2d[commit_mask].contiguous()

        accept_len = accept_len_i64.to(torch.int32)
        commit_lens = commit_lens_i64.to(torch.int32)
        return accept_len, commit_lens, bonus, out_tokens, commit_indices

    def _pack_ragged_committed(
        self,
        *,
        flat: torch.Tensor,
        block_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        bs = int(block_lens.numel())
        if bs == 0:
            return flat[:0]
        max_block = int(block_lens.max().item())
        device = flat.device
        rel_2d = torch.arange(max_block, dtype=torch.int64, device=device).unsqueeze(0)
        idx_2d = offsets[:-1].to(device=device).unsqueeze(1) + rel_2d
        valid_2d = rel_2d < commit_lens.to(device=device, dtype=torch.int64).unsqueeze(
            1
        )
        return flat[idx_2d[valid_2d]].contiguous()

    def _get_rel_2d(self, width: int) -> torch.Tensor:
        width = int(width)
        cached = self._rel_2d_cache.get(width)
        if cached is None or cached.device != self.device:
            cached = torch.arange(width, dtype=torch.int64, device=self.device).unsqueeze(0)
            self._rel_2d_cache[width] = cached
        return cached

    def _ensure_ragged_verify_buffers(
        self, graph_total: int, real_total: int, bs: int
    ) -> None:
        graph_total = max(int(graph_total), 1)
        real_total = max(int(real_total), 1)
        bs = max(int(bs), 1)
        if (
            self._ragged_verify_buffer_cap >= graph_total
            and self._ragged_verify_buffer_cap >= real_total
            and self._ragged_verify_req_cap >= bs + 1
        ):
            return

        new_token_cap = max(
            graph_total,
            real_total,
            (
                self._ragged_verify_buffer_cap * 2
                if self._ragged_verify_buffer_cap > 0
                else 1
            ),
        )
        new_req_cap = max(
            bs + 1,
            (
                self._ragged_verify_req_cap * 2
                if self._ragged_verify_req_cap > 0
                else 1
            ),
        )
        device = self.device
        self._ragged_verify_buffers = {
            "draft_tokens": [
                torch.empty((new_token_cap,), dtype=torch.long, device=device)
                for _ in range(2)
            ],
            "positions": [
                torch.empty((new_token_cap,), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
            "cache_loc": [
                torch.empty((new_token_cap,), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
            "real_indices": [
                torch.empty((new_token_cap,), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
            "offsets": [
                torch.empty((new_req_cap,), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
        }
        self._ragged_verify_buffer_cap = new_token_cap
        self._ragged_verify_req_cap = new_req_cap

    def _next_ragged_verify_buffers(
        self, graph_total: int, real_total: int, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_ragged_verify_buffers(graph_total, real_total, bs)
        slot = self._ragged_verify_buffer_slot
        self._ragged_verify_buffer_slot = (slot + 1) % 2
        bufs = self._ragged_verify_buffers
        return (
            bufs["draft_tokens"][slot][:graph_total],
            bufs["positions"][slot][:graph_total],
            bufs["cache_loc"][slot][:graph_total],
            bufs["real_indices"][slot][:real_total],
            bufs["offsets"][slot][: bs + 1],
        )

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._accept_len_buf = torch.empty((new_cap,), dtype=torch.int32, device=device)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        assert self._accept_len_buf is not None
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._accept_len_buf[:bs],
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _ensure_runtime_block_buffers(self, bs: int, block_size: int) -> None:
        block_size = int(block_size)
        current_cap = self._runtime_block_buffer_cap.get(block_size, 0)
        if current_cap >= int(bs):
            return

        new_cap = max(int(bs), current_cap * 2 if current_cap > 0 else int(bs))
        device = self.device
        self._runtime_block_buffers[block_size] = {
            "block_ids": [
                torch.empty((new_cap, block_size), dtype=torch.long, device=device)
                for _ in range(2)
            ],
            "positions": [
                torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
            "draft_tokens": [
                torch.empty((new_cap, block_size), dtype=torch.long, device=device)
                for _ in range(2)
            ],
            "verify_cache_loc": [
                torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
            "out_tokens": [
                torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
                for _ in range(2)
            ],
        }
        self._runtime_block_buffer_cap[block_size] = new_cap
        self._runtime_block_buffer_slot.setdefault(block_size, 0)

    def _next_runtime_block_buffers(
        self, bs: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact-width contiguous buffers for runtime block sizes below max."""

        if int(block_size) == int(self.block_size):
            assert self._draft_block_ids_buf is not None
            assert self._draft_block_positions_buf is not None
            assert self._draft_block_tokens_buf is not None
            assert self._draft_verify_out_cache_loc_buf is not None
            return (
                self._draft_block_ids_buf[:bs],
                self._draft_block_positions_buf[:bs],
                self._draft_block_tokens_buf[:bs],
                self._draft_verify_out_cache_loc_buf[:bs],
            )

        self._ensure_runtime_block_buffers(bs, int(block_size))
        slot = self._runtime_block_buffer_slot[int(block_size)]
        self._runtime_block_buffer_slot[int(block_size)] = (slot + 1) % 2
        bufs = self._runtime_block_buffers[int(block_size)]
        return (
            bufs["block_ids"][slot][:bs],
            bufs["positions"][slot][:bs],
            bufs["draft_tokens"][slot][:bs],
            bufs["verify_cache_loc"][slot][:bs],
        )

    def _runtime_out_tokens_buffer(
        self, bs: int, block_size: int
    ) -> Optional[torch.Tensor]:
        if int(block_size) == int(self.block_size):
            return None
        self._ensure_runtime_block_buffers(bs, int(block_size))
        slot = (self._runtime_block_buffer_slot[int(block_size)] - 1) % 2
        return self._runtime_block_buffers[int(block_size)]["out_tokens"][slot][:bs]

    def _validate_phase1_sampling_support(
        self, model_worker_batch: ScheduleBatch
    ) -> None:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        verified_id: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = verified_id.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            verified_id=verified_id.to(dtype=torch.int32),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _forward_batch_generation_ragged_decode(
        self,
        *,
        model_worker_batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        block_lens: Optional[torch.Tensor] = None,
        on_publish=None,
    ) -> GenerationBatchResult:
        bs = len(model_worker_batch.seq_lens)
        device = self.device
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise RuntimeError(
                "DFLASH packed-ragged pre-draft entropy path currently supports greedy decoding only."
            )

        self._timing_start_cycle(batch=model_worker_batch)
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with `weight`."
        )
        _t = self._timing_begin("runtime_block_policy")
        if block_lens is None:
            block_lens = self._get_runtime_block_sizes(
                bs, model_worker_batch.req_pool_indices
            )
        block_lens = block_lens.to(device=device, dtype=torch.int64)
        max_block_size = int(block_lens.max().item())
        max_graph_block_size = int(self.block_size)
        total_block_tokens = int(block_lens.sum().item())
        graph_num_tokens_per_batch = self._select_ragged_graph_num_tokens_per_batch(
            bs=bs,
            total_tokens=total_block_tokens,
            max_block_size=max_graph_block_size,
        )
        self._timing_end("runtime_block_policy", _t)
        self._timing_set("runtime_block_size", int(max_block_size))
        self._timing_set("runtime_block_size_mean", float(block_lens.float().mean().item()))
        self._timing_set("runtime_block_tokens", int(total_block_tokens))
        self._timing_set("ragged_graph_max_block_size", int(max_graph_block_size))
        self._timing_set("draft_graph_num_tokens_per_batch", int(graph_num_tokens_per_batch))
        self._timing_set(
            "draft_graph_total_tokens", int(bs) * int(graph_num_tokens_per_batch)
        )

        _t = self._timing_begin("draft_block_setup")
        self._ensure_draft_block_buffers(bs)
        assert self._draft_seq_lens_cpu_buf is not None
        prefix_lens = model_worker_batch.seq_lens
        (
            block_ids,
            positions,
            verify_out_cache_loc,
            offsets,
            projection_indices,
        ) = self._make_ragged_block(
            verified_id=draft_input.verified_id.view(-1),
            prefix_lens=prefix_lens.view(-1),
            block_lens=block_lens,
            req_pool_indices=model_worker_batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
        )

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))
            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(torch.int64)
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )
            block_end = draft_prefix_lens + block_lens.to(draft_prefix_lens.dtype)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(model_worker_batch.seq_lens_sum)
                    if model_worker_batch.seq_lens_sum is not None
                    else int(model_worker_batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        draft_spec_info = DFlashRaggedVerifyInput(
            draft_token=block_ids,
            positions=positions,
            draft_token_lens=block_lens.to(torch.int32),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            draft_token_num=max_block_size,
            num_tokens_per_batch=graph_num_tokens_per_batch,
            total_draft_tokens=total_block_tokens,
            disable_cuda_graph=True,
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids,
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=draft_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._timing_end("draft_block_setup", _t)

        _t = self._timing_begin("draft_model_forward")
        with torch.inference_mode():
            draft_cuda_graph_runner = self.draft_model_runner.decode_cuda_graph_runner
            self._timing_set(
                "draft_cuda_graph",
                int(
                    bool(
                        draft_cuda_graph_runner
                        and draft_cuda_graph_runner.can_run(forward_batch)
                    )
                ),
            )
            draft_logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        self._timing_end("draft_model_forward", _t)

        _t = self._timing_begin("draft_token_projection")
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_tokens = block_ids.clone()
        if int(projection_indices.numel()) > 0:
            draft_next = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden.index_select(0, projection_indices),
                lm_head=lm_head,
            )
            draft_tokens[projection_indices] = draft_next
        self._timing_end("draft_token_projection", _t)

        _t = self._timing_begin("verify_preparation")
        (
            verify_draft_tokens,
            verify_positions,
            verify_graph_out_cache_loc,
            graph_block_lens,
            graph_real_indices,
            graph_num_tokens_per_batch,
        ) = self._make_graph_padded_ragged_verify(
            draft_tokens=draft_tokens,
            prefix_lens=prefix_lens,
            block_lens=block_lens,
            offsets=offsets,
            req_pool_indices=model_worker_batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            max_block_size=max_graph_block_size,
            graph_num_tokens_per_batch=graph_num_tokens_per_batch,
            real_total=int(total_block_tokens),
        )
        graph_draft_token_num = int(graph_block_lens.max().item())
        self._timing_set("verify_graph_num_tokens_per_batch", int(graph_num_tokens_per_batch))
        self._timing_set("verify_graph_total_tokens", int(verify_draft_tokens.numel()))
        self._timing_set("verify_graph_max_block_size", int(graph_draft_token_num))
        verify_input = DFlashRaggedVerifyInput(
            draft_token=verify_draft_tokens,
            positions=verify_positions,
            draft_token_lens=block_lens.to(torch.int32),
            graph_draft_token_lens=graph_block_lens,
            graph_real_indices=graph_real_indices,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            draft_token_num=graph_draft_token_num,
            num_tokens_per_batch=graph_num_tokens_per_batch,
            total_draft_tokens=int(verify_draft_tokens.numel()),
            disable_cuda_graph=False,
        )
        model_worker_batch.out_cache_loc = verify_graph_out_cache_loc
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        if draft_input.planning_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
        elif draft_input.reserved_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_v2_verify(
            model_worker_batch, self.target_worker
        )
        model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
        model_worker_batch.seq_lens_sum = seq_lens_sum_backup
        self._timing_end("verify_preparation", _t)

        _t = self._timing_begin("target_verify_forward")
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        self._timing_end("target_verify_forward", _t)
        self._timing_set("target_verify_cuda_graph", int(bool(can_run_cuda_graph)))
        logits_output = target_out.logits_output
        next_token_logits = logits_output.next_token_logits
        if next_token_logits is None:
            raise RuntimeError("DFLASH verify requires target logits, but got None.")
        target_predict = torch.argmax(next_token_logits, dim=-1)
        if (
            graph_real_indices is not None
            and int(graph_real_indices.numel()) != int(verify_draft_tokens.numel())
        ):
            # Avoid compacting the full [tokens, vocab] logits matrix. We only
            # need the verifier top-1 token for acceptance; compacting after
            # argmax turns this into a cheap 1D gather.
            target_predict = target_predict.index_select(0, graph_real_indices)
            logits_output.next_token_logits = None
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.index_select(
                    0, graph_real_indices
                )

        _t = self._timing_begin("acceptance_bonus")
        accept_len, commit_lens, bonus, out_tokens, commit_indices = (
            self._compute_ragged_accept_bonus(
                draft_tokens=draft_tokens,
                target_predict=target_predict,
                block_lens=block_lens,
                offsets=offsets,
            )
        )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)
        self._timing_end("acceptance_bonus", _t)
        self._timing_set("mean_accepted_drafts", float(accept_len.float().mean().item()))
        self._timing_set("total_accepted_drafts", int(accept_len.sum().item()))
        self._timing_set("max_accepted_drafts", int(accept_len.max().item()))
        self._timing_set("min_accepted_drafts", int(accept_len.min().item()))

        _t = self._timing_begin("post_verify_kv_materialize")
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH verify requires target hidden states, but got None.")
        committed_hidden = hidden.index_select(0, commit_indices)
        committed_cache_loc = verify_out_cache_loc.index_select(0, commit_indices)
        committed_positions = positions.index_select(0, commit_indices)
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=committed_hidden,
            cache_loc=committed_cache_loc,
            positions=committed_positions,
            ctx_lens=commit_lens,
            req_pool_indices=model_worker_batch.req_pool_indices,
        )
        self._timing_end("post_verify_kv_materialize", _t)
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done
        self._timing_finish_cycle()

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens,
            accept_lens=commit_lens,
            can_run_cuda_graph=bool(can_run_cuda_graph),
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=max_block_size,
            speculative_num_draft_tokens_per_req=commit_lens,
            new_seq_lens=new_seq_lens,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )
            batch_output.new_seq_lens = model_worker_batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_lens is None
                or model_worker_batch.prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, "
                    "but got None."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            ctx_lens = torch.tensor(
                model_worker_batch.extend_lens, dtype=torch.int32, device=device
            )
            draft_seq_lens = torch.tensor(
                model_worker_batch.prefix_lens, dtype=torch.int32, device=device
            )

            if model_worker_batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                draft_seq_lens,
                ctx_lens,
                int(sum(model_worker_batch.extend_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=model_worker_batch.out_cache_loc,
                positions=positions,
                ctx_lens=ctx_lens,
                req_pool_indices=model_worker_batch.req_pool_indices,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                verified_id=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
                cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            batch_output.next_draft_input.verify_done = verify_done
            return batch_output

        # Decode / target-verify stage.
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=torch.empty((0,), device=self.device, dtype=torch.int32),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(self.block_size),
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(model_worker_batch.seq_lens)
        device = self.device

        # --- 1) Draft a fixed block with the draft model.
        self._timing_start_cycle(batch=model_worker_batch)
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with `weight`."
            )
        if (
            forced_block_lens := self._forced_ragged_block_sizes(bs)
        ) is not None:
            _t = self._timing_begin("runtime_block_policy")
            ragged_block_lens = forced_block_lens
            self._timing_end("runtime_block_policy", _t)
            self._timing_set("runtime_block_policy_forced", 1)
            if not self._is_uniform_block_size(ragged_block_lens):
                return self._forward_batch_generation_ragged_decode(
                    model_worker_batch=model_worker_batch,
                    draft_input=draft_input,
                    block_lens=ragged_block_lens,
                    on_publish=on_publish,
                )

            self._predraft_entropy_rectangular_override = int(
                ragged_block_lens.reshape(-1)[0].item()
            )
            self._timing_set("runtime_block_rectangular_graph_candidate", 1)

        if (
            self.predraft_entropy_ragged
            and self.predraft_entropy_policy is not None
            and self.predraft_entropy_context is not None
            and self.predraft_entropy_context.numel() > 0
        ):
            _t = self._timing_begin("runtime_block_policy")
            ragged_block_lens = self._get_runtime_block_sizes(
                bs, model_worker_batch.req_pool_indices
            )
            self._timing_end("runtime_block_policy", _t)
            self._timing_set("runtime_block_policy_precomputed", 1)
            if not self._is_uniform_block_size(ragged_block_lens):
                return self._forward_batch_generation_ragged_decode(
                    model_worker_batch=model_worker_batch,
                    draft_input=draft_input,
                    block_lens=ragged_block_lens,
                    on_publish=on_publish,
                )

            self._predraft_entropy_rectangular_override = int(
                ragged_block_lens.reshape(-1)[0].item()
            )
            self._timing_set("runtime_block_rectangular_graph_candidate", 1)

        try:
            _t = self._timing_begin("runtime_block_policy")
            block_size = int(self._get_runtime_block_size(bs))
            self._timing_end("runtime_block_policy", _t)
        finally:
            self._predraft_entropy_rectangular_override = None
        self._timing_set("runtime_block_size", int(block_size))
        self._draft_block_spec_info.draft_token_num = int(block_size)
        self._draft_block_spec_info.num_tokens_per_batch = int(block_size)
        self._draft_block_spec_info.disable_cuda_graph = True
        _t = self._timing_begin("draft_block_setup")
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        (
            block_ids,
            positions_2d,
            draft_tokens,
            verify_out_cache_loc_2d,
        ) = self._next_runtime_block_buffers(bs, block_size)
        prefix_lens = model_worker_batch.seq_lens
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    verified_id=draft_input.verified_id.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.verified_id)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets[:block_size],
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=model_worker_batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.verified_id)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets[:block_size],
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(model_worker_batch.seq_lens_sum)
                    if model_worker_batch.seq_lens_sum is not None
                    else int(model_worker_batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._timing_end("draft_block_setup", _t)

        _t = self._timing_begin("draft_model_forward")
        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output
        self._timing_end("draft_model_forward", _t)

        _t = self._timing_begin("draft_token_projection")
        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(block_size), -1)
        need_postdraft_scalar = (
            self.postdraft_policy is not None
            and int(self.postdraft_policy.scalar_dim) > 0
        )
        draft_projection_out = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
            return_confidence=need_postdraft_scalar,
        )
        if need_postdraft_scalar:
            draft_next_flat, draft_scalar_flat = draft_projection_out
            draft_next = draft_next_flat.view(bs, int(block_size) - 1)
            draft_scalar = draft_scalar_flat.view(bs, int(block_size) - 1, -1)
        else:
            draft_next = draft_projection_out.view(bs, int(block_size) - 1)
            draft_scalar = None

        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        self._timing_end("draft_token_projection", _t)

        if self.postdraft_policy is not None:
            sampling_info = model_worker_batch.sampling_info
            if sampling_info is not None and not sampling_info.is_all_greedy:
                raise RuntimeError(
                    "DFLASH post-draft ragged verification currently supports greedy decoding only."
                )

            _t = self._timing_begin("postdraft_policy")
            postdraft_hidden = draft_hidden[:, 1:, :]
            if int(self.postdraft_policy.scalar_dim) > 0:
                _t_conf = self._timing_begin("postdraft_confidence")
                if draft_scalar is None:
                    raise RuntimeError(
                        "DFLASH post-draft scalar confidence was not captured during draft sampling."
                    )
                scalar = draft_scalar
                self._timing_end("postdraft_confidence", _t_conf)
            else:
                scalar = torch.empty(
                    (bs, int(block_size) - 1, 0),
                    dtype=torch.float32,
                    device=device,
                )
            _t_head = self._timing_begin("postdraft_policy_head")
            verify_block_lens = self.postdraft_policy.select_verify_lens(
                hidden=postdraft_hidden,
                prev_token_ids=draft_tokens[:, :-1],
                scalar=scalar,
                max_block_size=int(block_size),
            )
            verify_block_lens = verify_block_lens.to(device=device, dtype=torch.int64)
            self._timing_end("postdraft_policy_head", _t_head)
            self._timing_end("postdraft_policy", _t)
            self._timing_set(
                "postdraft_verify_block_size_mean",
                float(verify_block_lens.float().mean().item()),
            )
            self._timing_set(
                "postdraft_verify_block_size_max",
                int(verify_block_lens.max().item()),
            )

            _t = self._timing_begin("verify_preparation")
            rel_2d = self._get_rel_2d(int(block_size))
            valid = rel_2d < verify_block_lens.unsqueeze(1)
            packed_draft_tokens = draft_tokens[valid].contiguous()
            packed_positions = positions_2d[valid].contiguous()
            packed_cache_loc = verify_out_cache_loc_2d[valid].contiguous()
            offsets = torch.empty((bs + 1,), dtype=torch.int64, device=device)
            offsets[0].zero_()
            torch.cumsum(verify_block_lens, dim=0, out=offsets[1:])
            real_total = int(packed_draft_tokens.numel())
            graph_num_tokens_per_batch = self._select_ragged_graph_num_tokens_per_batch(
                bs=bs,
                total_tokens=real_total,
                max_block_size=int(block_size),
            )
            (
                verify_draft_tokens,
                verify_positions,
                verify_graph_out_cache_loc,
                graph_block_lens,
                graph_real_indices,
                graph_num_tokens_per_batch,
            ) = self._make_graph_padded_ragged_verify(
                draft_tokens=packed_draft_tokens,
                prefix_lens=prefix_lens,
                block_lens=verify_block_lens,
                offsets=offsets,
                real_total=real_total,
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                max_block_size=int(block_size),
                graph_num_tokens_per_batch=graph_num_tokens_per_batch,
            )
            graph_draft_token_num = int(graph_block_lens.max().item())
            self._timing_set("verify_graph_num_tokens_per_batch", int(graph_num_tokens_per_batch))
            self._timing_set("verify_graph_total_tokens", int(verify_draft_tokens.numel()))
            self._timing_set("verify_graph_max_block_size", int(graph_draft_token_num))
            verify_input = DFlashRaggedVerifyInput(
                draft_token=verify_draft_tokens,
                positions=verify_positions,
                draft_token_lens=verify_block_lens.to(torch.int32),
                graph_draft_token_lens=graph_block_lens,
                graph_real_indices=graph_real_indices,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                draft_token_num=graph_draft_token_num,
                num_tokens_per_batch=graph_num_tokens_per_batch,
                total_draft_tokens=int(verify_draft_tokens.numel()),
                disable_cuda_graph=False,
            )
            model_worker_batch.out_cache_loc = verify_graph_out_cache_loc
            need_mamba_verify_commit = hasattr(
                self.target_worker.model_runner.attn_backend,
                "update_mamba_state_after_mtp_verify",
            )
            seq_lens_pre_verify = (
                model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
            )
            seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
            seq_lens_sum_backup = model_worker_batch.seq_lens_sum
            if draft_input.planning_seq_lens_cpu is not None:
                model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
                model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                model_worker_batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

            verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_v2_verify(
                model_worker_batch, self.target_worker
            )
            model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
            model_worker_batch.seq_lens_sum = seq_lens_sum_backup
            self._timing_end("verify_preparation", _t)

            _t = self._timing_begin("target_verify_forward")
            target_out = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
            self._timing_end("target_verify_forward", _t)
            self._timing_set("target_verify_cuda_graph", int(bool(can_run_cuda_graph)))
            logits_output = target_out.logits_output
            next_token_logits = logits_output.next_token_logits
            if next_token_logits is None:
                raise RuntimeError("DFLASH verify requires target logits, but got None.")
            target_predict = torch.argmax(next_token_logits, dim=-1)
            if (
                graph_real_indices is not None
                and int(graph_real_indices.numel()) != int(verify_draft_tokens.numel())
            ):
                target_predict = target_predict.index_select(0, graph_real_indices)
                logits_output.next_token_logits = None
                if logits_output.hidden_states is not None:
                    logits_output.hidden_states = logits_output.hidden_states.index_select(
                        0, graph_real_indices
                    )

            _t = self._timing_begin("acceptance_bonus")
            accept_len, commit_lens, bonus, out_tokens, commit_indices = (
                self._compute_ragged_accept_bonus(
                    draft_tokens=packed_draft_tokens,
                    target_predict=target_predict,
                    block_lens=verify_block_lens,
                    offsets=offsets,
                )
            )
            if need_mamba_verify_commit:
                assert seq_lens_pre_verify is not None
                self._update_target_mamba_state_after_verify(
                    batch=model_worker_batch,
                    seq_lens_pre_verify=seq_lens_pre_verify,
                    commit_lens=commit_lens,
                )
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            if on_publish is not None:
                on_publish(new_seq_lens)
            self._timing_end("acceptance_bonus", _t)
            self._timing_set("mean_accepted_drafts", float(accept_len.float().mean().item()))
            self._timing_set("total_accepted_drafts", int(accept_len.sum().item()))
            self._timing_set("max_accepted_drafts", int(accept_len.max().item()))
            self._timing_set("min_accepted_drafts", int(accept_len.min().item()))

            _t = self._timing_begin("post_verify_kv_materialize")
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError("DFLASH verify requires target hidden states, but got None.")
            committed_hidden = hidden.index_select(0, commit_indices)
            committed_cache_loc = packed_cache_loc.index_select(0, commit_indices)
            committed_positions = packed_positions.index_select(0, commit_indices)
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=committed_hidden,
                cache_loc=committed_cache_loc,
                positions=committed_positions,
                ctx_lens=commit_lens,
                req_pool_indices=model_worker_batch.req_pool_indices,
            )
            self._timing_end("post_verify_kv_materialize", _t)
            logits_output.hidden_states = None

            next_draft_input = self._make_next_draft_input_decode(
                verified_id=bonus,
                new_seq_lens=new_seq_lens,
                cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            self._timing_finish_cycle()

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=out_tokens,
                accept_lens=commit_lens,
                can_run_cuda_graph=bool(can_run_cuda_graph),
                next_draft_input=next_draft_input,
                speculative_num_draft_tokens=int(block_size),
                speculative_num_draft_tokens_per_req=commit_lens,
                new_seq_lens=new_seq_lens,
            )

        # --- 2) Target verify.
        _t = self._timing_begin("verify_preparation")
        # TARGET_VERIFY uses standard causal masking; custom masks are unnecessary here.
        custom_mask = None

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        model_worker_batch.out_cache_loc = verify_out_cache_loc
        sampling_info = model_worker_batch.sampling_info

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        if draft_input.planning_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
        elif draft_input.reserved_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_v2_verify(
            model_worker_batch, self.target_worker
        )
        model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
        model_worker_batch.seq_lens_sum = seq_lens_sum_backup
        self._timing_end("verify_preparation", _t)

        _t = self._timing_begin("target_verify_forward")
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        self._timing_end("target_verify_forward", _t)
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        self._timing_set("target_verify_cuda_graph", int(bool(can_run_cuda_graph)))

        _t = self._timing_begin("acceptance_bonus")
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(block_size),
            )

        candidates = draft_tokens
        new_seq_lens = None
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            commit_lens = accept_len.to(torch.int32) + 1  # [bs]
            out_tokens = torch.empty(
                (bs, int(block_size)), dtype=torch.int64, device=device
            )
            if int(block_size) > 1:
                out_tokens[:, : int(block_size) - 1].copy_(candidates[:, 1:])
            out_tokens[:, int(block_size) - 1].fill_(0)
            out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(block_size)
            )
            if self._use_triton_accept_bonus:
                try:
                    (
                        accept_len,
                        commit_lens,
                        bonus,
                        out_tokens_buf,
                        new_seq_lens,
                    ) = self._next_accept_bonus_buffers(bs)
                    runtime_out_tokens = self._runtime_out_tokens_buffer(
                        bs, block_size
                    )
                    out_tokens = (
                        out_tokens_buf
                        if runtime_out_tokens is None
                        else runtime_out_tokens
                    )
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                        prefix_lens=prefix_lens,
                        new_seq_lens_out=new_seq_lens,
                    )
                except Exception as e:
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                        e,
                    )
                    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                    out_tokens = torch.empty(
                        (bs, int(block_size)),
                        dtype=torch.int64,
                        device=device,
                    )
                    if int(block_size) > 1:
                        out_tokens[:, : int(block_size) - 1].copy_(
                            candidates[:, 1:]
                        )
                    out_tokens[:, int(block_size) - 1].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )
            else:
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                out_tokens = torch.empty(
                    (bs, int(block_size)), dtype=torch.int64, device=device
                )
                if int(block_size) > 1:
                    out_tokens[:, : int(block_size) - 1].copy_(candidates[:, 1:])
                out_tokens[:, int(block_size) - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)
        self._timing_end("acceptance_bonus", _t)
        self._timing_set("mean_accepted_drafts", float(accept_len.float().mean().item()))
        self._timing_set("total_accepted_drafts", int(accept_len.sum().item()))
        self._timing_set("max_accepted_drafts", int(accept_len.max().item()))
        self._timing_set("min_accepted_drafts", int(accept_len.min().item()))

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        _t = self._timing_begin("post_verify_kv_materialize")
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(block_size), -1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
            req_pool_indices=model_worker_batch.req_pool_indices,
        )
        self._timing_end("post_verify_kv_materialize", _t)

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            verified_id=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done
        self._timing_finish_cycle()

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(block_size),
            # The non-overlap (sync) scheduler path advances batch.seq_lens
            # from the result; overlap carries it via next_draft_input instead.
            new_seq_lens=new_seq_lens,
        )
