# History based block selection for DFlash

This is a working greedy Transformers reference for a RecGuide-inspired pre-draft controller on frozen DFlash. It includes matched-state collection, CPU-only table fitting, offline evaluation, and closed-loop adaptive generation. It is not yet integrated into SGLang and has no real-model performance result.

## Scope and conventions

Only the history-based controller is adapted from [RecGuide section 5.2](https://arxiv.org/html/2609.34388v1#S5.S2). Its shared-backbone prospective drafting mechanism is not implemented. RecGuide motivates recent verifier entropy and previous block outcomes; the histogram bins, smoothing, fallback, and retention mode below are explicit choices in this implementation, not a claimed exact reproduction of an unpublished calibration routine.

- B includes one anchor; there are B−1 proposed tokens.
- A is the consecutive accepted proposal count, 0 through B−1.
- V_B predicts cycle progress A+1. It does not predict an acceptance probability.
- Each request starts with empty history. After verification, keep only the previous two entropy values, the last executed B, and whether A=B−1.
- Entropy is computed in FP32 and measured in nats. Target logit rows 0 through A inclusive are used: these predict the accepted proposals and the correction or bonus on valid prefixes. Row A+1 would consume a rejected proposal and is excluded.
- The controller chooses B before the current draft runs. No current candidate or counterfactual outcome can update the decision's history.

Candidate sizes are configurable sorted integers >=2. They are not restricted to powers of two or four arms. B24/B32 can be explicitly tested, but support in this reference is not evidence that a B16-trained drafter performs well at those widths.

## Implementation

- `dflash/history_policy.py`: request history, value-table fitting and serialization, selection, and matched-state metrics. Standard-library only.
- `dflash/history_generate.py`: actual greedy DFlash generation, entropy extraction, and optional paired counterfactual collection.
- `scripts/dflash_history.py`: collect, fit, replay, and rollout commands.
- `tests/test_history_policy.py`: estimator, leakage guards, CLI, entropy alignment, and tiny real-model correctness tests.

The table partitions certainty using training-only quantile edges, with separate keys for one/two previous observations, previous B, and full acceptance. Cold-start states have a separate key. Each cell's actual A_B+1 observations are shrunk toward that block's global training mean using a configurable pseudo-count (default 20). Unseen cells fall back to global means. Values are not forced to be monotone across B: changing the bidirectional draft width can change candidates and acceptance.

Priced mode chooses the largest score V_B − rho × T_C(B). Costs must be explicit whole-cycle milliseconds and rho is in tokens/ms. No fabricated timing curve is supplied. Smallest B wins ties.

Retention mode chooses the smallest B whose predicted accepted proposal count V_B−1 reaches alpha times the prediction at the largest candidate B. This mode separates predictor analysis from hardware. Alpha is a policy parameter, not a guarantee of measured assessment retention.

## Collection and split discipline

For each state, collection independently drafts and verifies every candidate B from cloned identical target/draft caches. Only the selected behavior arm commits its KV/features and updates history. The default behavior samples B uniformly each cycle, independent of current counterfactual outcomes; `--fixed-block` provides a fixed-history control. This produces block-conditioned supervision rather than clipping a full-block label.

Rows are excluded from fitting/replay if any candidate would produce EOS or if the largest arm lacks sufficient output space including its bonus. Such rows remain in raw evidence. Warm-up history is explicit rather than borrowing future cycles. Every row records prompt, cycle, prefix hash, pre-decision history, actual arm outcomes, selected action, and selected verifier entropy.

Training membership comes from the canonical train IDs. Calibration/assessment membership comes from the existing pilot manifest and must belong to canonical validation. Cross-partition exact message duplicates are excluded. Fitting accepts training rows only, and evaluation rejects fitting-prompt overlap. Do not pool calibration and assessment for threshold tuning.

Outputs use fresh destinations. A completed collection has config.json, cycles.jsonl, progress.json, and summary.json; the final receipt includes the cycle-file hash. Fitting refuses incomplete or tampered completed runs. Model identity records declared immutable repository revisions and config hashes; this does not independently hash the complete weight files. Preserve the pinned snapshots. Source hashes and split/manifest hashes are recorded.

## Commands

Run from the repository with Torch and Transformers 4.57.1 for collection/rollout. Fit/replay need only Python. Paths below are placeholders to be replaced by verified paths on the selected host; these commands have not been launched on a GPU.

```bash
python scripts/dflash_history.py collect \
  --models /absolute/path/models.json \
  --manifest /absolute/path/messages.jsonl \
  --split-dir /absolute/path/canonical_split \
  --pilot-manifest /absolute/path/pilot/cache/manifest.json \
  --group train --blocks 4,8,12,16 --limit-prompts 32 \
  --max-new-tokens 256 --max-cycles 32 --device cuda:0 \
  --output /absolute/path/new_history_train

python scripts/dflash_history.py fit \
  --runs /absolute/path/new_history_train --blocks 4,8,12,16 \
  --bins 8 --prior-count 20 --output /absolute/path/history_table.json
```

Collect calibration and assessment into separate fresh directories using the same configuration and their respective `--group` values. Membership stays fixed; do not regenerate a random validation partition.

```bash
python scripts/dflash_history.py replay \
  --table /absolute/path/history_table.json \
  --runs /absolute/path/new_history_calibration /absolute/path/new_history_assessment \
  --mode retention --alpha 0.96 --output /absolute/path/history_replay.json

python scripts/dflash_history.py rollout \
  --table /absolute/path/history_table.json --mode retention --alpha 0.96 \
  --models /absolute/path/models.json \
  --manifest /absolute/path/messages.jsonl \
  --split-dir /absolute/path/canonical_split \
  --pilot-manifest /absolute/path/pilot/cache/manifest.json \
  --group assessment --limit-prompts 16 --max-new-tokens 256 --max-cycles 0 \
  --device cuda:0 --output /absolute/path/new_history_rollout
```

Repeat rollout with each `--fixed-block` using the same prompt selection and token limits. The fixed baseline controls actual drafting length, not post-draft truncation. Inspect output IDs and actual returned completion lengths. A cycle limit can truncate outputs, so use `--max-cycles 0` for complete bounded-token comparisons.

Add `--history-free` to replay or rollout for the same selector using global training means instead of history-conditioned values. This isolates the value of history from the choice of scheduling rule.

For the paper-style cost objective, use `--mode priced --rho VALUE --cost-profile FILE --concurrency C`. The profile schema is:

```text
{
  "units": "ms",
  "scope": "whole_cycle",
  "costs_ms": {"64": {"4": measured_ms, "8": measured_ms, ...}}
}
```

The example above is schematic, not a valid measurement file. Supply every candidate B. The runner does not automatically verify that an external cost profile matches GPU/model/backend; that provenance must be checked before interpreting priced-policy results.

## Verification and remaining work

CPU tests use tiny randomly initialized Qwen3 target and DFlash models. They check greedy output against target-only generation across changing widths, counterfactual-cache isolation, policy-driven rollouts, entropy masking, EOS at prefill, and output limits. They do not establish BF16 numerical parity, useful prediction, or speedup on Qwen3-4B.

The targeted local suite passes 32 tests: the new history tests plus existing policy-granularity and post-draft serving regression tests. Local test environment: Torch 2.14.1 CPU, Transformers 4.57.1. Tests use no downloaded model weights.

The reference intentionally uses cache copies and host-side scalar decisions. Collection timing includes counterfactual probes; rollout timing is a single-request diagnostic including prefill and history bookkeeping. Neither is an SGLang serving benchmark.

Next gate: collect a bounded real-model paired dataset, fit on train, choose policy settings on calibration, and report assessment against every fixed arm and a history-free global-table control. Then run closed-loop assessment because an adaptive policy changes future histories and visited states. Only after a signal improvement should the controller be integrated into SGLang with per-request state lifecycle, device-side entropy reduction, graph-compatible packing or block buckets, and measured overhead. No SGLang or live cluster files were changed for this reference implementation.
