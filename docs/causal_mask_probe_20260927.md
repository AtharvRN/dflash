# Post-hoc causal draft attention diagnostic

Frozen original Qwen3-4B and DFlash B16 snapshots, workstation GPU 4;
BF16 SDPA, greedy, TF32 disabled. No training or throughput measurement.

Select four prompts per source from the saved headroom training-development
states using seed 927; up to four eligible states per prompt, spaced over the
saved states. Compare B4/B8/B12/B16/B20 on exactly the same prefixes.

Both variants reconstruct target features with a full-prefix target forward.
The prefix includes the known anchor, which is excluded from that forward.
Each draft gets a fresh cache and identical full-prefix target features.
Bidirectional uses the original absent mask. Causal uses an explicit additive
mask of shape [1, 1, B, prefix_length+B]: all context columns allowed,
lower-triangular draft columns allowed. Both use is_causal=False so the explicit
rectangular mask alone specifies visibility. No target attention is changed.

Every outcome is verified with the unchanged target model and independently
checked against a greedy target continuation. Exclude a paired state if any
variant accepts EOS. Report prefix changes and acceptance regressions between
adjacent tested block sizes, plus the subset where the smaller block was fully
accepted. Larger B need not increase acceptance even when prefixes are stable.

Full-prefix reconstruction and workstation hardware differ from historical
incremental A100 collection. Saved B16 token-and-label replay agreement is
recorded, but the comparison uses newly recomputed paired outcomes only.
This is a local intervention at common B16-trajectory states, not an end-to-end
causal-policy rollout or held-out generalization evaluation.

Script: `scripts/causal_mask_probe.py`. Four-state smoke output:
`/data/scratch/zekaili/atharv/dflash/runs/causal_mask_smoke_20260927`.
Main output: `/data/scratch/zekaili/atharv/dflash/runs/causal_mask_16p_20260927`.
State results are saved individually; `summary.json` is written only at completion.
