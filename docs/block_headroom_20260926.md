# Actual block-size headroom: fresh diagnostic

Original frozen Qwen3-4B and DFlash B16 snapshots; BF16 SDPA, greedy,
thinking off, TF32 disabled. Separate checkout based on committed `9c2e859`;
no uncommitted model changes from the earlier systems work.

Sample 32 canonical training prompts per source (128 total), excluding exact
message duplicates crossing the canonical train/validation boundary and
deduplicating training content. Interleave sources in execution order.
Skip prompts exceeding 2048 tokens; report skips rather than replacing them.
Reference trajectories use B16, up to 256 output tokens and eight sampled states
per prompt, spaced across generated-token offsets. Short responses yield fewer
states. The sample is deliberately source-balanced, not population weighted.

At each state independently restore target/draft caches and execute every integer
B2–B20. B includes the known anchor; A counts accepted proposed tokens. Exclude
the entire paired state if any accepted candidate contains EOS. Only sample
states with at least 20 remaining output slots. Store prefixes, hashes, candidate
tokens, actual acceptance, and last fused vectors. These are training development
data, not a fresh assessment set or training run.

Checks: random alternative order; reverse-order exact draft/label replay;
canonical greedy verification on initial states through 19 draft positions;
same-candidate target-width controls only for B<16. B>16 cannot be inferred by
clipping the B16 candidates. Numerical canonical disagreements are reported,
not silently relabeled.

Analysis uses exact multiple-choice integer dynamic programming to minimize total
draft budget subject to 90/95/96/98/100% retention relative to B16, for three
decision granularities: one global block, one block per prompt, one per state.
Also report unconstrained maximum acceptance, within/between-prompt B16 variance,
and transitions between disjoint sets of best sizes (ties preserved). These are
hindsight oracles at shared states; even the prompt oracle is not a full-response
BlockPilot oracle. No learned reward, timing objective, deployed-policy claim,
or throughput inference. Early results are descriptive, without confidence
intervals or held-out generalization claims.

Tiny smoke runs use separate output destinations and do not count toward the main
sample. Collection writes to /tmp with asynchronous atomic, checksum-verified
per-prompt PVC backups. Refuse nonempty destinations. Stop starting prompts at a
time budget, then finish analysis/backups. No recurring monitor is installed.
