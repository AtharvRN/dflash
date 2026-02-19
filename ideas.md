# Ideas Backlog

This file tracks speculative decoding ideas to test in this repo.

Status legend:
- `proposed`: not implemented
- `implemented`: code exists
- `validated`: tested with clear evidence
- `rejected`: tested and not useful

## Latest Results Snapshot (2026-02-18)
- Suffix-seed experiment (`bs=16`, AIME25, `max_samples=10`, `max_new_tokens=2048`, A100):
  - `seed_mode=none`: TPOT `0.006258`, tau `7.92`, speedup `6.44x`
  - `seed_mode=sparse` (`seed_max_tokens=2`): TPOT `0.011713`, tau `4.27`, derived speedup `3.44x`
  - `seed_mode=dense` (`seed_max_tokens=-1`): TPOT `0.019733`, tau `2.52`, derived speedup `2.04x`
- Multi-step draft refinement (`--draft-steps=2`) in `benchmark.py`:
  - TPOT `0.044606`, tau `1.20`, tokens/s `22.38` (near-baseline behavior; severe acceptance collapse)
- Interpretation:
  - naive suffix reuse and naive multi-step draft refinement both hurt acceptance and throughput on current setup.

## 1) EWMA Throughput Scheduler
- Status: `implemented`
- Goal: choose block size dynamically based on measured performance, not fixed thresholds.
- Core signal: `score(bs) = tau_hat(bs) / cycle_hat(bs)` where hats are EWMA.
- Current implementation:
  - warmup probing
  - periodic probe interval
  - switch margin + streak + cooldown
  - low-accept fallback (step down)
- Files:
  - `benchmark_dynamic_schedule.py`
  - `docs/QUICK_EXPERIMENTS.md`
- Next:
  - validate on AIME25 full split with `max_new_tokens=2048`
  - compare against fixed `bs=16`

## 2) Per-Cycle Profiling in `benchmark.py`
- Status: `implemented`
- Goal: understand where time is spent and how tau evolves over generation.
- Tracked:
  - per cycle: `tau`, `effective_block_size`, `acceptance_ratio`, `draft_s`, `target_s`, `cycle_s`, token index
  - per sample: profile summary and shares
- Flags:
  - `--collect-profile`
  - `--save-cycle-trace-path`
- Files:
  - `benchmark.py`
- Next:
  - run profiling on larger sample counts and compare across GPUs

## 3) Exact Multi-Round Speculative Sampling (Algorithm 1 style)
- Status: `implemented` (research script)
- Goal: test exact acceptance-residual multi-round sampling in DFlash context.
- Notes:
  - mathematically aligned with accept/reject residual procedure
  - currently slower than normal DFlash on tested setup
- File:
  - `benchmark_multiround_spec.py`
- Current verdict:
  - promising for algorithm study, not throughput win yet

## 4) Recycle Rejected Draft Suffix After First Mismatch
- Status: `rejected` (current naive variants)
- Problem:
  - when first reject happens at position `k`, draft tokens `k+1...` are discarded.
  - target logits for those positions were computed but under wrong prefix, so cannot be committed directly.
- Idea:
  - treat discarded suffix as proposal hints for next cycle after correction token.
  - verify as usual; do not commit without target verification.
- Safer variant:
  - confidence-gated reuse (only high-margin positions).
- Risks:
  - little gain if recycled hints are often invalid after corrected token.
  - added logic/caching complexity.
- Next experiment:
  - if revisited, add strict confidence gating / limited seed budget first
  - avoid dense seeding by default
- Current implementation details:
  - Script: `benchmark_suffix_seed.py`
  - Seed modes:
    - `none`: normal DFlash behavior
    - `dense`: seed most suffix positions
    - `sparse`: alternate seed/mask pattern (`[corr, M, old, M, old, ...]`)
  - Logged metrics:
    - seed attempt cycles
    - seeded cycles / seeded token count
    - per-cycle seeded count + tau in cycle trace

## 5) Multi-Step Draft Refinement per Cycle
- Status: `rejected` (current inference-only implementation)
- What was tried:
  - `benchmark.py --draft-steps=2` (iterative block rewrites before verification)
- Result:
  - acceptance collapsed (tau ~1.2) and throughput regressed strongly.
- Why likely:
  - DFlash paper training/inference is single-pass parallel block drafting.
  - repeated self-conditioning at inference is off-distribution for current draft model.
- If revisited:
  - try selective refinement (only low-confidence positions), not full-block rewrite.

## 6) Dynamic Scheduler Objective: Break-Even by Real Cycle Time
- Status: `proposed`
- Motivation:
  - threshold like `tau < 8` is insufficient.
  - use measured cycle time to compute true break-even (`tau/cycle_time`).
- Plan:
  - derive online break-even between adjacent block sizes
  - compare against current EWMA score policy

## 7) Candidate Diversity for Multi-Round
- Status: `proposed`
- Problem:
  - DFlash draft is effectively single-trajectory; multi-round proposals can become low-diversity overhead.
- Idea:
  - diversify proposal sources (e.g., branch proposals / varied proposal heads / alternative proposal distributions).
- Note:
  - higher complexity; defer until simpler suffix-recycle experiment is measured.

## 8) DFlash + Tree Verification (EAGLE-style candidate packing)
- Status: `proposed`
- Goal:
  - raise acceptance by verifying multiple draft candidates per cycle, not only one trajectory.
- Idea:
  - generate `K` diffusion draft candidates for the same block (e.g., `K=2/4`),
  - pack into a trie/tree,
  - run one target verification pass over packed candidates,
  - choose path with longest accepted prefix (start with `temperature=0` setting).
- Main risk:
  - verification cost growth may erase gains if tree is too wide.
- Initial measurement plan:
  - microbenchmark target verification latency for packed node count `q=16` vs `q=32` vs `q=48`
  - estimate required tau gain at each cost multiplier before full implementation.
- Success criterion:
  - improved realized tokens/sec (not just higher tau).

## 9) Tree-Lite for Non-Autoregressive DFlash (bs=8, top-k=2 prototype)
- Status: `proposed`
- Motivation:
  - DFlash draft is block-parallel (non-autoregressive within the block), so one draft pass can miss alternative early tokens where most rejections occur.
- Core design:
  - keep block size fixed (e.g., `bs=16` in main runs; use `bs=8` toy case for debugging),
  - branch only early positions with small beam (`top-k=2`, depth `m≈5..6`),
  - verify packed tree nodes in one target pass with ancestor-only attention mask.
- Attention-mask rule (verification stage):
  - token node attends to:
    - prompt/prefix context (outside packed matrix),
    - itself,
    - its ancestor nodes on the same path.
  - no sibling/cross-branch attention.
- Non-AR caveat:
  - branch candidates built from one diffusion pass are approximate (later positions are not strictly conditioned on chosen early branch tokens).
  - if needed, add one cheap candidate-refinement pass for selected branches only.
- Measurements:
  - compare `target_s/cycle`, `tau`, `tau/target_s`, `tokens/s` vs single-candidate baseline.

## 10) Candidate-Solution Generator Benchmark
- Status: `implemented`
- Goal:
  - generate a bounded set of draft block candidates each cycle and test if candidate diversity can improve accepted length.
- Script:
  - `benchmark_candidate_solutions.py`
- Current design:
  - keep normal DFlash draft pass for a block,
  - select early branch positions (`--branch-depth`) optionally filtered by uncertainty margin (`--branch-margin-threshold`),
  - create up to `--max-candidates` blocks using per-position top-k (`--branch-top-k`) combinations,
  - verify each candidate with target, choose best by `tau` (tie-break by draft score), then commit.
- Important caveat:
  - this is a research testbed and adds extra target verifications per cycle; it is not expected to be throughput-optimal yet.
