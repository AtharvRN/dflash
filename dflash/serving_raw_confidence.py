"""Post-draft greedy verification trimming; copied into the recovered SGLang tree.

The known anchor is always verified. A failed confidence test discards that
proposal and all successors, not just individual low-confidence positions.
This changes verification work, never the full-block draft candidates.
"""
import math
import torch


class DFlashRawConfidencePolicy:
    scalar_dim = 4
    confidence_mode = "logprob_only"

    def __init__(self, threshold, max_block_size=16):
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or self.threshold > 0:
            raise ValueError("Raw-confidence threshold must be finite and <= 0")
        if max_block_size < 2:
            raise ValueError("Draft block must include anchor and proposals")
        # Graph buckets, NOT a restriction of the per-request decision space.
        self.arms = tuple(range(1, int(max_block_size) + 1))

    @torch.no_grad()
    def select_verify_lens(self, *, hidden, prev_token_ids, scalar, max_block_size):
        if scalar.ndim != 3 or scalar.shape[1:] != (max_block_size - 1, 4):
            raise ValueError("Expected one draft logprob per proposed token, no anchor")
        logprob = scalar[..., 3]
        passing = torch.isfinite(logprob) & (logprob.double() >= self.threshold)
        return 1 + passing.long().cumprod(dim=1).sum(dim=1)
