from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def parse_dflash_dynamic_block_arms(value: str | Sequence[int]) -> list[int]:
    """Parse and validate DFLASH dynamic block-size arms."""

    if isinstance(value, str):
        raw_arms = [part.strip() for part in value.split(",")]
        arms = [int(part) for part in raw_arms if part]
    else:
        arms = [int(arm) for arm in value]

    arms = sorted(set(arms))
    if not arms:
        raise ValueError("DFLASH dynamic block-size arms cannot be empty.")
    if arms[0] < 2:
        raise ValueError(
            "DFLASH dynamic block-size arms must be >= 2 because the first "
            "position is the bonus token."
        )
    return arms


@dataclass
class DFlashDynamicBlockController:
    """Batch-level controller for choosing a DFLASH verify block size.

    The controller uses only post-verify accepted draft lengths, so it is a
    serving-safe baseline and can later be replaced by a learned policy head.
    """

    arms: Sequence[int]
    initial_block_size: int
    target_accept_ratio: float = 0.80
    ema_alpha: float = 0.20
    warmup_batches: int = 10
    update_interval: int = 5

    def __post_init__(self) -> None:
        self.arms = parse_dflash_dynamic_block_arms(self.arms)
        if self.initial_block_size not in self.arms:
            self.current_block_size_value = max(
                [arm for arm in self.arms if arm <= self.initial_block_size],
                default=self.arms[-1],
            )
        else:
            self.current_block_size_value = int(self.initial_block_size)

        if not (0.0 < float(self.target_accept_ratio) <= 1.0):
            raise ValueError(
                "target_accept_ratio must be in (0, 1], "
                f"got {self.target_accept_ratio}."
            )
        if not (0.0 < float(self.ema_alpha) <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}.")
        self.warmup_batches = max(0, int(self.warmup_batches))
        self.update_interval = max(1, int(self.update_interval))

        self.num_batches = 0
        self.accepted_drafts_ema: float | None = None
        self.accept_ratio_ema: float | None = None
        self.full_accept_rate_ema: float | None = None
        self.last_change_accepted_drafts_ema: float | None = None
        self.last_change_accept_ratio_ema: float | None = None
        self.last_change_full_accept_rate_ema: float | None = None

    def current_block_size(self) -> int:
        return int(self.current_block_size_value)

    def on_verify_complete(
        self, num_correct_drafts_per_req: Sequence[int], batch_size: int = 0
    ) -> int | None:
        if not num_correct_drafts_per_req:
            return None

        self.num_batches += 1
        bs = int(batch_size) if batch_size else len(num_correct_drafts_per_req)
        bs = max(bs, 1)
        current_block = int(self.current_block_size_value)
        max_drafts = max(current_block - 1, 1)

        accepted_sum = float(sum(max(0, int(x)) for x in num_correct_drafts_per_req))
        accepted_mean = accepted_sum / bs
        accept_ratio = accepted_mean / max_drafts
        full_accept_count = sum(
            1 for x in num_correct_drafts_per_req if int(x) >= max_drafts
        )
        full_accept_rate = float(full_accept_count) / bs

        self.accepted_drafts_ema = self._ema(self.accepted_drafts_ema, accepted_mean)
        self.accept_ratio_ema = self._ema(self.accept_ratio_ema, accept_ratio)
        self.full_accept_rate_ema = self._ema(
            self.full_accept_rate_ema, full_accept_rate
        )

        if self.num_batches <= self.warmup_batches:
            return None
        if (self.num_batches - self.warmup_batches) % self.update_interval != 0:
            return None

        next_block = self._choose_next_block()
        if next_block == current_block:
            return None

        self.last_change_accepted_drafts_ema = self.accepted_drafts_ema
        self.last_change_accept_ratio_ema = self.accept_ratio_ema
        self.last_change_full_accept_rate_ema = self.full_accept_rate_ema
        self.current_block_size_value = next_block
        # Acceptance ratio and full-accept rate are arm-dependent. Reset them
        # after a block-size change to avoid stale pressure from the previous arm.
        self.accept_ratio_ema = None
        self.full_accept_rate_ema = None
        return next_block

    def _ema(self, old_value: float | None, new_value: float) -> float:
        if old_value is None:
            return float(new_value)
        alpha = float(self.ema_alpha)
        return alpha * float(new_value) + (1.0 - alpha) * float(old_value)

    def _choose_next_block(self) -> int:
        assert self.accepted_drafts_ema is not None
        assert self.accept_ratio_ema is not None
        assert self.full_accept_rate_ema is not None

        current = int(self.current_block_size_value)
        current_idx = self.arms.index(current)

        # If many requests consume the full available draft budget, the observed
        # mean is censored by the current arm. Grow one arm to recover length.
        if self.full_accept_rate_ema >= 0.25 and current_idx + 1 < len(self.arms):
            return int(self.arms[current_idx + 1])

        if self.accept_ratio_ema >= self.target_accept_ratio:
            return current

        accepted = max(0.0, float(self.accepted_drafts_ema))
        candidates: list[int] = []
        for arm in self.arms:
            draft_slots = max(int(arm) - 1, 1)
            # Avoid knowingly clipping the current accepted-length estimate.
            if draft_slots + 0.5 >= accepted:
                candidates.append(int(arm))

        if not candidates:
            return int(self.arms[-1])

        # Pick the smallest arm that preserves the accepted-length estimate. This
        # raises accepted/block ratio without needing handcrafted token features.
        return min(candidates)
