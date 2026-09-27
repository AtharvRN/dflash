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

## Completed results

64/64 states eligible, 16 prompts, zero disagreements with the independent
greedy target acceptance check across 640 outcomes.

| B (including anchor) | Bidirectional mean A | Causal mean A |
| --- | ---: | ---: |
| 4 | 2.09375 | 1.96875 |
| 8 | 4.203125 | 3.609375 |
| 12 | 5.75 | 4.90625 |
| 16 | 7.078125 | 5.65625 |
| 20 | 4.421875 | 6.109375 |

Paired causal-minus-bidirectional mean A at B16: -1.421875, prompt bootstrap
95% interval [-2.140625, -0.78125]. At B20: +1.6875, interval
[0.546875, 2.8125]. Descriptive 10,000 prompt resamples, seed 927; not a
generalization claim or training-seed interval.

Causal attention had zero acceptance regressions across 256 adjacent-block
comparisons; bidirectional had 38. Causal token prefixes were identical for
B4->B8, B8->B12, B12->B16. For B16->B20, 15/64 causal token prefixes changed,
all strictly after the first rejection, with unchanged acceptance for those
15 states. Shape-dependent numerical effects are a hypothesis, not established
by this test; do not claim exact token-prefix invariance. Bidirectional B16->B20
had 60 prefix changes and 28 acceptance regressions; 12/15 fully accepted B16
states regressed, versus 0/11 for causal drafting's own fully accepted subset.

44/64 original saved B16 token-and-label outcomes were reproduced exactly.
Use only the newly recomputed paired comparison; historical replay differences
may involve hardware and full-prefix versus incremental numerical effects.

Conclusion: the post-hoc mask trades B16 acceptance for more stable acceptance
across lengths. Causal B20 still accepts fewer tokens than original B16 while
proposing more. It is not an efficiency win demonstrated by this experiment.

## B24/B32 extension

Run `causal_mask_b32_16p_20260927`, code commit `cc03e2b`, blocks
16/20/24/32, independent greedy continuation extended to 31 proposals.
Same 64 states, all eligible; exact prefix identities checked against the first
run. Both modes' B16/B20 outcomes reproduced exactly, including draft token IDs.

| B | Bidirectional block-verified mean A | Causal block-verified mean A |
| --- | ---: | ---: |
| 16 | 7.078125 | 5.65625 |
| 20 | 4.421875 | 6.109375 |
| 24 | 3.296875 | 6.28125 |
| 32 | 3.140625 | 6.265625 |

One of 512 outcomes disagreed with the independent token-by-token target
reference: prompt 83143, cycle 0, bidirectional B32 accepted 4 in block
verification versus 1 against that reference. Reference-based bidirectional B32
mean is therefore 3.09375; all other means agree. Cause not isolated; do not
describe the entire run as exact greedy-equivalence validated.

Causal B20->B24 had zero token-prefix changes and zero acceptance regressions.
Causal B24->B32 had 19/64 token-prefix changes and one acceptance regression:
prompt 69571, cycle 13, A=2->0, first proposed token changed. Both acceptance
values agree with the independent target reference. Numerical shape effects are
plausible but unproven; exact monotonicity does not hold in this BF16 execution.

No throughput measured. Causal acceptance plateaus around 6.3, below original
bidirectional B16's 7.08 despite substantially larger proposal budgets. These
are paired training-development states, not an end-to-end or held-out result.
