# Predictor Head Results (2026-03-11)

## Scope

- Worktree: `/workspace/DSC291/dflash-predictor-head`
- Branch: `predictor-head-track-persistent`
- Task scope: predictor-head / adaptive verify-length prediction only
- Dataset focus: GSM8K fixed128
- Drafting system: DFLASH with diffusion draft model

This document consolidates the predictor-head experiments run on March 11, 2026, including training, offline evaluation, and online A/B results.

## Key conclusion

- The hidden-only predictor checkpoint is usable as a direct prefix-survival predictor by draft index.
- Using cumulative products over the current head outputs is the wrong control rule for this diffusion setting.
- Switching to direct prefix-survival thresholding fixes the offline metric mismatch.
- A boundary-aware retrain improves strict offline boundary metrics further and was then validated in a fresh `c=1` online run.
- Online, `aggregate=max` preserves baseline tau, but still loses throughput versus plain DFLASH on GSM8K fixed128.
- Best online point tested so far:
  - `aggregate=max`
  - `h=0.15`
  - tau matches baseline
  - throughput is still `-6.74%` versus plain DFLASH
- Current recommendation: do not deploy predictor gating yet.

## Environment notes

- Pod: `dflash-pod`
- Conda env: `dflash`
- Required setup after pod restart:
  - `pip install -r requirements.txt`
  - `pip install -e ./third_party/sglang/python --no-deps`
- Verified runtime dependencies during this session:
  - `torch==2.9.1+cu128`
  - `flash-attn==2.8.3`
  - editable local `sglang`

## Predictor checkpoints used

Initial hidden-only checkpoint retrained in the pod:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_train_20260311_090051/model.pt`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_train_20260311_090051/metrics.json`

Training summary:

- input dim: `2560`
- train loss: `0.2037`
- val loss: `0.2112`
- val accuracy: `0.9066`

Boundary-aware retrain from the same hidden-only features:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_20260311_214147/model.pt`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_20260311_214147/metrics.json`

Training summary:

- input dim: `2560`
- hidden dim: `1024`
- epochs: `6`
- loss weighting: `boundary`
- boundary weight alpha: `4.0`
- boundary weight tau: `1.5`
- best val loss: `0.4232` at epoch `5`
- final val accuracy: `0.8450`

## Phase 1: Hidden-only + cumulative stopping

Initial attempt was a SpecDec++-style cumulative stopping rule:

- head input: hidden vector only
- predicted quantity used online/offline: per-index acceptance probability
- stopping rule:
  - stop at first `i` where `1 - prod_{j=1..i} p_j > h`

### Offline eval on GSM8K fixed128

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_eval_gsm8k_fixed128_20260311_092118/predictor_eval.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_eval_gsm8k_fixed128_20260311_092118/predictor_eval.md`

Token metrics:

- accuracy: `0.8663`
- F1: `0.7987`
- BCE: `0.3005`
- Brier: `0.0943`

Threshold sweep results:

| h | mean verify k | verify reduction | tau retention |
|---|---:|---:|---:|
| `0.50` | `4.527` | `71.7%` | `0.721` |
| `0.70` | `5.278` | `67.0%` | `0.796` |
| `0.80` | `5.710` | `64.3%` | `0.832` |

Conclusion:

- No threshold in the initial sweep reached `tau_retention >= 0.95`.

### High-threshold cumulative sweep

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_eval_gsm8k_fixed128_hi_20260311_183707/predictor_eval.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_specdecpp_hidden_eval_gsm8k_fixed128_hi_20260311_183707/predictor_eval.md`

Selected results:

| h | mean verify k | verify reduction | tau retention |
|---|---:|---:|---:|
| `0.95` | `6.804` | `57.5%` | `0.9013` |
| `0.97` | `7.099` | `55.6%` | `0.9163` |
| `0.98` | `7.320` | `54.3%` | `0.9261` |
| `0.99` | `7.666` | `52.1%` | `0.9395` |

Conclusion:

- Even at `h=0.99`, cumulative stopping did not reach `tau_retention >= 0.95`.

## Phase 2: Diagnose what the head is actually predicting

The next step was to test whether the current head behaves like:

- conditional accept probabilities, which justify cumulative products, or
- direct prefix-survival probabilities by index, which do not

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_index_profile_gsm8k_fixed128_20260311_190000/predictor_index_profile.png`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_index_profile_gsm8k_fixed128_20260311_190000/predictor_index_profile.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_index_profile_gsm8k_fixed128_20260311_190000/predictor_index_profile.json`

Result:

- weighted direct abs gap vs empirical rate: `0.0051`
- weighted cumulative abs gap vs empirical rate: `0.1036`

Selected indices:

| index i | empirical `P(tau >= i)` | direct head output | cumulative product |
|---|---:|---:|---:|
| `1` | `0.8785` | `0.8776` | `0.8776` |
| `3` | `0.6140` | `0.5919` | `0.4860` |
| `4` | `0.5113` | `0.4941` | `0.3610` |
| `8` | `0.2580` | `0.2582` | `0.1262` |
| `15` | `0.0815` | `0.0781` | `0.0182` |

Conclusion:

- The existing hidden-only checkpoint already behaves like direct `P(tau >= i)`.
- Cumulative products were double-counting the decay.

## Phase 3: Direct prefix-survival policy

After the diagnosis above, offline eval and runtime gating were changed to:

- interpret head output at index `i` as direct `P(tau >= i)`
- choose verify length using:
  - first `i` where `p_i < h`
  - otherwise verify the full runtime block size

### Offline direct prefix-survival eval

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_20260311_190405/predictor_eval.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_20260311_190405/predictor_eval.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_low_20260311_190504/predictor_eval.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_low_20260311_190504/predictor_eval.md`

Best operating points:

| target | h | tau retention | verify reduction | mean verify k |
|---|---:|---:|---:|---:|
| `tau_retention >= 0.99` | `0.03` | `0.992` | `0.362` | `10.206` |
| `tau_retention >= 0.95` | `0.15` | `0.957` | `0.499` | `8.013` |

Conclusion:

- Direct prefix-survival thresholding is much better aligned than cumulative stopping.

### Strict boundary metrics on the same fixed128 slice

The offline evaluator was then extended to score verify-length prediction directly against the true cycle boundary:

- exact-match accuracy: `pred_verify_len == true_boundary`
- within-1 accuracy: `abs(pred_verify_len - true_boundary) <= 1`
- mean absolute error
- signed error
- under / over rates
- under / over amount

Artifact:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_strict_20260311_200939/predictor_eval.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_strict_20260311_200939/predictor_eval.md`

Fixed128 summary:

- mean true boundary tokens: `6.083`
- mean true accept tokens: `5.083`

Selected thresholds:

| h | mean verify k | mean true boundary | tau retention | exact | within-1 | MAE | signed err | under rate | over rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.03` | `10.206` | `6.083` | `0.992` | `0.135` | `0.243` | `4.272` | `4.122` | `0.035` | `0.830` |
| `0.10` | `8.632` | `6.083` | `0.973` | `0.168` | `0.377` | `3.035` | `2.549` | `0.103` | `0.728` |
| `0.15` | `8.013` | `6.083` | `0.957` | `0.192` | `0.440` | `2.639` | `1.930` | `0.137` | `0.670` |
| `0.20` | `7.539` | `6.083` | `0.944` | `0.214` | `0.485` | `2.381` | `1.456` | `0.177` | `0.609` |
| `0.25` | `7.137` | `6.083` | `0.929` | `0.232` | `0.526` | `2.199` | `1.053` | `0.211` | `0.557` |

Best strict metrics in the tested sweep:

- best exact-match accuracy: `0.279` at `h=0.60`
- best within-1 accuracy: `0.594` at `h=0.50`
- best MAE: `1.993` at `h=0.50`

Conclusion:

- The current checkpoint still over-verifies at tau-safe thresholds.
- The best tau-safe threshold we tested remains `h=0.15`, but it has only `19.2%` exact boundary accuracy and a strong positive bias (`+1.93` tokens).
- Boundary accuracy improves when the threshold is increased, but that gains efficiency by under-verifying and losing tau.

### Boundary-aware retrain on the same fixed128 slice

The training objective was then changed to upweight positions near the true verify boundary while keeping the same hidden-only input. This does not change the deployment rule; it changes which examples dominate training.

Artifacts:

- training:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_20260311_214147/metrics.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_20260311_214147/model.pt`
- eval:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_20260311_214309/predictor_eval.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_20260311_214309/predictor_eval.md`

Training notes:

- training loss continued falling through epoch `6`
- validation loss bottomed out at epoch `5`
- the retrain is stronger on the boundary metrics that matter for verify-length control, but less calibrated if the goal is to keep very low thresholds

Token metrics for the new checkpoint:

- accuracy: `0.8644`
- F1: `0.8017`
- BCE: `0.3121`
- Brier: `0.0964`

Selected thresholds for the new checkpoint:

| h | mean verify k | tau retention | exact | within-1 | MAE | signed err | under rate | over rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.03` | `13.831` | `0.9998` | `0.089` | `0.125` | `7.754` | `7.748` | `0.002` | `0.909` |
| `0.10` | `10.503` | `0.9940` | `0.141` | `0.271` | `4.547` | `4.420` | `0.033` | `0.826` |
| `0.15` | `9.308` | `0.9823` | `0.179` | `0.367` | `3.540` | `3.225` | `0.068` | `0.753` |
| `0.20` | `8.484` | `0.9697` | `0.215` | `0.444` | `2.912` | `2.401` | `0.102` | `0.683` |
| `0.25` | `7.852` | `0.9536` | `0.242` | `0.499` | `2.521` | `1.768` | `0.140` | `0.618` |
| `0.40` | `6.349` | `0.8911` | `0.307` | `0.602` | `1.913` | `0.266` | `0.270` | `0.423` |
| `0.60` | `4.785` | `0.7628` | `0.321` | `0.604` | `2.069` | `-1.298` | `0.478` | `0.201` |

Direct comparison against the previous hidden-only checkpoint:

- best threshold with `tau_retention >= 0.95`
  - old: `h=0.15`, mean verify `8.013`, exact `0.192`, within-1 `0.440`, MAE `2.639`
  - new: `h=0.25`, mean verify `7.852`, exact `0.242`, within-1 `0.499`, MAE `2.521`
- best exact-match accuracy
  - old: `0.279` at `h=0.60`
  - new: `0.321` at `h=0.60`
- best MAE
  - old: `1.993` at `h=0.50`
  - new: `1.913` at `h=0.40`

Conclusion:

- The boundary-aware loss improved strict boundary quality materially.
- It also made the checkpoint more conservative at low thresholds, so the tau-safe operating point shifted upward from `h=0.15` to roughly `h=0.25`.

### Boundary-aware retrain: fresh `c=1` online A/B

The new checkpoint was then tested online in a fresh single-request run to isolate predictor quality from aggregation. These numbers should be compared only within this fresh pod session because the absolute throughput differs from the earlier `c=1` run.

Artifacts:

- fresh baseline:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_bw_base_20260311_215051/predictor_online_ab_c1_fixed128_bw_base_20260311_215051_gsm8k_bs16__c1.md`
- boundary-aware checkpoint `h=0.20`, `aggregate=max`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_bw_h020_20260311_215755/predictor_online_ab_c1_fixed128_bw_h020_20260311_215755_gsm8k_bs16__c1.md`
- boundary-aware checkpoint `h=0.25`, `aggregate=max`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_bw_h025_20260311_220046/predictor_online_ab_c1_fixed128_bw_h025_20260311_220046_gsm8k_bs16__c1.md`

Results:

| policy | tok/s | delta vs fresh baseline | tau | accept rate | mean verify len |
|---|---:|---:|---:|---:|---:|
| fresh baseline | `499.98` | `0.00%` | `6.350` | `0.355` | `16.000` |
| boundary-aware `h=0.20`, `max` | `452.89` | `-9.42%` | `6.085` | `0.650` | `8.438` |
| boundary-aware `h=0.25`, `max` | `441.02` | `-11.79%` | `5.911` | `0.687` | `7.781` |

Cycle-level means from `spec_cycle_trace`:

| policy | mean verify len | mean tau per cycle | mean cycle e2e |
|---|---:|---:|---:|
| fresh baseline | `16.000` | `6.057` | `10.334 ms` |
| boundary-aware `h=0.20`, `max` | `8.438` | `5.776` | `10.960 ms` |
| boundary-aware `h=0.25`, `max` | `7.781` | `5.600` | `10.926 ms` |

Conclusion:

- The boundary-aware checkpoint is stronger on the control metrics:
  - mean verify length is reduced by about half at `h=0.20`
  - accept rate rises sharply
- But even in `c=1`, where aggregation is irrelevant, the new checkpoint still loses throughput versus the fresh baseline.
- Increasing `h` from `0.20` to `0.25` buys lower verify length and higher accept rate, but at the cost of more tau erosion and lower throughput.
- For this new checkpoint, `h=0.20` is the safer `c=1` point of the two tested values.

### Full-data boundary-aware retrains

The next step was to keep the same boundary-aware objective and hidden-only input, but retrain on the larger predictor corpus:

- feature dir:
  - `/workspace/DSC291/dflash/outputs/predictor_train_trace_qwen4b_20260309/feature_shards`
- shards: `100`
- requests: `9953`

#### Full-data fast run (`lr=1e-3`)

Artifacts:

- training:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_full_fast_20260311_230749/metrics.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_full_fast_20260311_230749/model.pt`
- eval:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_fullfast_20260311_232726/predictor_eval.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_fullfast_20260311_232726/predictor_eval.md`

Training summary:

- train rows: `9,890,340`
- val rows: `117,240`
- train loss: `0.3140`
- train accuracy: `0.8945`
- val loss: `0.2934`
- val accuracy: `0.9085`

Fixed128 summary:

| h | mean verify k | tau retention | exact | within-1 | MAE |
|---|---:|---:|---:|---:|---:|
| `0.10` | `9.422` | `0.984` | `0.159` | `0.325` | `3.634` |
| `0.15` | `8.439` | `0.968` | `0.191` | `0.426` | `2.910` |
| `0.20` | `7.648` | `0.946` | `0.228` | `0.510` | `2.441` |
| `0.25` | `6.951` | `0.914` | `0.255` | `0.559` | `2.201` |

Conclusion:

- Relative to the smaller-data boundary-aware checkpoint, this full-data `1e-3` run is more aggressive:
  - lower verify lengths
  - better boundary metrics at moderate thresholds
  - worse tau retention at the same thresholds
- It did not produce a strictly better checkpoint.

#### Full-data lower-LR run (`lr=3e-4`, no validation split)

To check whether the full-data `1e-3` run was simply too aggressive, the same full dataset was retrained at `lr=3e-4` with `train_frac=1.0` and `val_frac=0.0`.

Artifacts:

- training:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_full_lr3e4_noval_20260311_235123/metrics.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_boundary_weighted_train_full_lr3e4_noval_20260311_235123/model.pt`
- eval:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_full_lr3e4_20260312_000244/predictor_eval.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_prefixsurv_eval_gsm8k_fixed128_boundary_weighted_full_lr3e4_20260312_000244/predictor_eval.md`

Training summary:

- train rows: `10,007,580`
- train loss: `0.3118`
- train accuracy: `0.8953`
- positive rate: `0.1992`
- predicted positive rate: `0.1390`

Token metrics:

- accuracy: `0.8516`
- F1: `0.7528`
- BCE: `0.3155`
- Brier: `0.1014`

Fixed128 summary:

| h | mean verify k | tau retention | exact | within-1 | MAE |
|---|---:|---:|---:|---:|---:|
| `0.10` | `8.661` | `0.972` | `0.187` | `0.399` | `3.061` |
| `0.15` | `7.777` | `0.948` | `0.221` | `0.493` | `2.536` |
| `0.20` | `7.093` | `0.921` | `0.250` | `0.549` | `2.242` |
| `0.25` | `6.511` | `0.891` | `0.271` | `0.586` | `2.077` |

Direct comparison across boundary-aware checkpoints:

| model | best `tau>=0.95` threshold | mean verify k | tau retention | exact | within-1 | MAE |
|---|---:|---:|---:|---:|---:|---:|
| smaller-data | `0.25` | `7.852` | `0.954` | `0.242` | `0.499` | `2.521` |
| full-data `1e-3` | `0.15` | `8.439` | `0.968` | `0.191` | `0.426` | `2.910` |
| full-data `3e-4` | `0.10` | `8.661` | `0.972` | `0.187` | `0.399` | `3.061` |

Conclusion:

- Lowering the learning rate to `3e-4` did not fix the calibration problem.
- Compared with the full-data `1e-3` run, the `3e-4` run is actually more conservative at the same threshold:
  - higher tau retention
  - higher verify length
  - weaker boundary accuracy
- Compared with the smaller-data checkpoint, the `3e-4` run is strictly worse on the useful tau-safe frontier.
- So the best boundary-aware checkpoint remains the smaller-data run, not either of the full-data retrains.

#### Full-data hazard retrain (`lr=3e-4`)

To use a more research-grounded objective, the trainer was extended with a discrete-time hazard objective:

- hidden-only input stays the same
- one hazard logit is predicted per draft position
- the first rejection position is modeled as a discrete-time event
- prefix-survival `P(tau >= i)` is derived as the cumulative product of `1 - hazard`

Artifacts:

- training:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_hazard_train_full_lr3e4_20260312_004357/metrics.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_hazard_train_full_lr3e4_20260312_004357/model.pt`
- eval:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_hazard_eval_gsm8k_fixed128_full_lr3e4_20260312_005125/predictor_eval.json`
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_hazard_eval_gsm8k_fixed128_full_lr3e4_20260312_005125/predictor_eval.md`

Training summary:

- objective: `hazard`
- train rows: `10,007,580`
- train cycles: `667,172`
- train loss: `1.5962`
- train token accuracy (derived from survival): `0.9049`

Token metrics on GSM8K fixed128:

- accuracy: `0.8679`
- F1: `0.8003`
- BCE: `0.2871`
- Brier: `0.0916`

Selected thresholds:

| h | mean verify k | tau retention | exact | within-1 | MAE |
|---|---:|---:|---:|---:|---:|
| `0.05` | `10.597` | `0.996` | `0.126` | `0.206` | `4.596` |
| `0.10` | `9.539` | `0.989` | `0.149` | `0.288` | `3.666` |
| `0.15` | `8.812` | `0.980` | `0.173` | `0.361` | `3.104` |
| `0.20` | `8.235` | `0.968` | `0.191` | `0.421` | `2.720` |
| `0.25` | `7.745` | `0.954` | `0.215` | `0.471` | `2.444` |
| `0.30` | `7.305` | `0.937` | `0.235` | `0.514` | `2.257` |

Direct comparison on the `tau_retention >= 0.95` frontier:

| model | best `tau>=0.95` threshold | mean verify k | tau retention | exact | within-1 | MAE |
|---|---:|---:|---:|---:|---:|---:|
| smaller-data boundary-aware | `0.25` | `7.852` | `0.954` | `0.242` | `0.499` | `2.521` |
| full-data BCE `3e-4` | `0.10` | `8.661` | `0.972` | `0.187` | `0.399` | `3.061` |
| full-data hazard `3e-4` | `0.25` | `7.745` | `0.954` | `0.215` | `0.471` | `2.444` |

Conclusion:

- The hazard objective is clearly better than the full-data BCE objective.
- It restores a much healthier tau-safe frontier while keeping full-data training.
- It still does not beat the smaller-data boundary-aware checkpoint on exact or within-1 accuracy.
- But it is now close enough to justify an online A/B before deciding whether the smaller-data checkpoint still dominates in practice.

## Phase 4: Online A/B on GSM8K fixed128

### Benchmark setup

- dataset: GSM8K
- fixed subset: first `128` questions
- concurrency: `16`
- `DFLASH_BLOCK_SIZE=16`
- adaptive block-size disabled during these online tests
- batching enabled
- same benchmark harness used across all runs

### Single-request isolation run (`c=1`)

To separate batch aggregation effects from predictor quality, a separate online A/B was run at concurrency `1` on the same fixed128 slice.

Artifacts:

- baseline:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_dflash_base_20260311_211421/predictor_online_ab_c1_fixed128_dflash_base_20260311_211421.md`
- predictor `h=0.03`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_h003_20260311_211624/predictor_online_ab_c1_fixed128_h003_20260311_211624.md`
- predictor `h=0.15`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_c1_fixed128_h015_20260311_211859/predictor_online_ab_c1_fixed128_h015_20260311_211859.md`

Results:

| policy | tok/s | delta vs baseline | tau | accept rate | mean verify len | verify time/cycle |
|---|---:|---:|---:|---:|---:|---:|
| baseline | `480.08` | `0.00%` | `6.350` | `0.355` | `16.000` | `7.622 ms` |
| predictor `h=0.03` | `451.26` | `-6.00%` | `6.298` | `0.547` | `10.287` | `7.691 ms` |
| predictor `h=0.15` | `432.27` | `-9.96%` | `6.004` | `0.673` | `8.074` | `7.639 ms` |

Conclusion:

- At `c=1`, aggregation is not the issue because there is only one request.
- The predictor does reduce mean verify length substantially.
- But that does not materially reduce measured verify time per cycle on this workload.
- Throughput still drops relative to plain DFLASH.
- This strongly suggests the current predictor/control path is not a throughput win even without batch aggregation effects.

### `c=1` cycle-level oracle analysis

To test whether a perfect verify-length controller would help at all under the measured runtime, a cycle-level oracle analysis was run on the `c=1` traces.

Artifact:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_c1_oracle_analysis_20260311_213048/c1_oracle_analysis.json`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_c1_oracle_analysis_20260311_213048/c1_oracle_analysis.md`

Method:

- use the `c=1` baseline and predictor traces
- build an empirical verify-time curve as a function of actual `verify_len`
- define oracle `verify_len = accept_length` for every baseline cycle
- keep measured draft time and measured non-draft/non-verify overhead fixed
- estimate oracle total verify time and an optimistic throughput upper bound

Observed verify-time curve:

- verify time is nearly flat across `verify_len = 1..16`
- empirical range is approximately `7.41 ms` to `7.80 ms`

Examples:

| verify len | mean verify time |
|---:|---:|
| `1` | `7.448 ms` |
| `4` | `7.581 ms` |
| `8` | `7.607 ms` |
| `12` | `7.713 ms` |
| `16` | `7.672 ms` |

Oracle result on the baseline trace:

- baseline mean verify len: `16.000`
- oracle mean verify len: `6.057`
- baseline total verify time: `49.039 s`
- oracle total verify time: `48.670 s`
- verify-time savings: `0.369 s` (`0.75%`)

Optimistic throughput upper bound:

- baseline throughput: `480.08 tok/s`
- oracle optimistic throughput: `482.26 tok/s`
- optimistic gain: `+0.45%`

Conclusion:

- Even a perfect cycle-level oracle would barely help at `c=1` under the measured timing behavior.
- The core issue is not just predictor error.
- The verify kernel cost is almost flat in `verify_len`, so reducing verify length alone does not buy meaningful throughput on this runtime.

### Plain DFLASH baseline

Artifact:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_dflash_base_20260311_191228/predictor_online_ab_fixed128_dflash_base_20260311_191228.md`

Metrics:

- throughput: `2845.50 tok/s`
- tau: `6.326`
- accept rate: `0.354`
- verify calls per second: `468.34`

### Predictor online A/B: `aggregate=q10`

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h003_20260311_191724/predictor_online_ab_fixed128_h003_20260311_191724.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h015_20260311_191924/predictor_online_ab_fixed128_h015_20260311_191924.md`

Results:

| h | tok/s | delta vs baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| `0.03` | `2397.96` | `-15.73%` | `4.415` | `0.704` | `543.98` |
| `0.15` | `2187.63` | `-23.12%` | `3.539` | `0.802` | `615.76` |

Conclusion:

- `q10` is too aggressive.
- It under-verifies and damages tau badly.

### Predictor online A/B: `aggregate=median`

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h003_median_20260311_192519/predictor_online_ab_fixed128_h003_median_20260311_192519.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h015_median_20260311_192858/predictor_online_ab_fixed128_h015_median_20260311_192858.md`

Results:

| h | tok/s | delta vs baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| `0.03` | `2638.00` | `-7.29%` | `5.797` | `0.510` | `467.52` |
| `0.15` | `2532.08` | `-11.01%` | `5.236` | `0.620` | `491.52` |

Conclusion:

- `median` is substantially better than `q10`.
- It still does not preserve baseline tau.

### Predictor online A/B: `aggregate=max`

Artifacts:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h003_max_20260311_192708/predictor_online_ab_fixed128_h003_max_20260311_192708.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h005_max_20260311_194226/predictor_online_ab_fixed128_h005_max_20260311_194226.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h010_max_20260311_194415/predictor_online_ab_fixed128_h010_max_20260311_194415.md`
- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h015_max_20260311_193049/predictor_online_ab_fixed128_h015_max_20260311_193049.md`

Results:

| h | tok/s | delta vs baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| `0.03` | `2619.60` | `-7.94%` | `6.351` | `0.364` | `429.77` |
| `0.05` | `2634.16` | `-7.43%` | `6.354` | `0.367` | `432.16` |
| `0.10` | `2643.08` | `-7.11%` | `6.380` | `0.376` | `431.99` |
| `0.15` | `2653.84` | `-6.74%` | `6.326` | `0.382` | `436.00` |

Conclusion:

- `aggregate=max` maintains baseline tau.
- Throughput improves monotonically across the tested `h` values from `0.03` to `0.15`.
- Best online point tested so far is `h=0.15`, `aggregate=max`.
- Even that point is still slower than baseline plain DFLASH.

### Boundary-aware checkpoint online A/B: `aggregate=max`, `h=0.20`

The newer boundary-aware checkpoint was also tested online at concurrency `16` using the same fixed128 setup and the same single batch-wide verify policy (`aggregate=max`).

Artifact:

- `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_bw_h020_max_20260311_221125/predictor_online_ab_fixed128_bw_h020_max_20260311_221125.md`

Top-line result:

| policy | tok/s | delta vs baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| baseline | `2845.50` | `0.00%` | `6.326` | `0.354` | `468.34` |
| old hidden-only `h=0.15`, `max` | `2653.84` | `-6.74%` | `6.326` | `0.382` | `436.00` |
| boundary-aware `h=0.20`, `max` | `2635.93` | `-7.37%` | `6.360` | `0.368` | `431.53` |

Cycle-level means from `spec_cycle_trace`:

| policy | mean verify len | mean tau per cycle | verify time/cycle |
|---|---:|---:|---:|
| baseline | `16.000` | `6.055` | `1.324 ms` |
| old hidden-only `h=0.15`, `max` | `14.733` | `6.066` | `1.347 ms` |
| boundary-aware `h=0.20`, `max` | `15.322` | `6.088` | `1.357 ms` |

Conclusion:

- The boundary-aware checkpoint at `c=16` is slightly more conservative than the earlier hidden-only `h=0.15`, `max` point.
- It gives marginally higher tau than both the old checkpoint and baseline.
- But it also increases mean verify length relative to the old checkpoint and loses a bit more throughput.
- So this new `c=16` point is not an improvement over the previous best online `aggregate=max` result.

### Boundary-aware checkpoint aggregation sweep at `c=16`, `h=0.20`

To test whether the online gap is mostly an aggregation issue, the same boundary-aware checkpoint and threshold were rerun with different single-call batch aggregation rules.

Artifacts:

- `mean`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_bw_h020_mean_20260311_221659/predictor_online_ab_fixed128_bw_h020_mean_20260311_221659.md`
- `median`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_bw_h020_median_20260311_221827/predictor_online_ab_fixed128_bw_h020_median_20260311_221827.md`
- `q75`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_bw_h020_q75_20260311_221958/predictor_online_ab_fixed128_bw_h020_q75_20260311_221958.md`
- `q90`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_bw_h020_q90_20260311_222126/predictor_online_ab_fixed128_bw_h020_q90_20260311_222126.md`

Top-line results:

| aggregate | tok/s | delta vs baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| baseline | `2845.50` | `0.00%` | `6.326` | `0.354` | `468.34` |
| `mean` | `2556.64` | `-10.15%` | `5.465` | `0.569` | `478.32` |
| `median` | `2498.91` | `-12.18%` | `5.359` | `0.590` | `475.30` |
| `q75` | `2557.01` | `-10.14%` | `6.035` | `0.458` | `437.63` |
| `q90` | `2634.81` | `-7.41%` | `6.279` | `0.395` | `436.36` |
| `max` | `2635.93` | `-7.37%` | `6.360` | `0.368` | `431.53` |

Cycle-level means:

| aggregate | mean verify len | mean tau per cycle | verify time/cycle |
|---|---:|---:|---:|
| baseline | `16.000` | `6.055` | `1.324 ms` |
| `mean` | `8.824` | `5.327` | `1.149 ms` |
| `median` | `8.416` | `5.240` | `1.144 ms` |
| `q75` | `11.913` | `5.823` | `1.287 ms` |
| `q90` | `14.171` | `6.018` | `1.335 ms` |
| `max` | `15.322` | `6.088` | `1.357 ms` |

Conclusion:

- `mean` and `median` reduce verify length the most, but they under-verify and lose too much tau.
- `q75` is a middle point, but it still loses a material amount of tau without improving throughput enough.
- `q90` is the closest alternative to `max`:
  - nearly identical throughput
  - slightly lower verify length
  - slightly lower tau
- For this checkpoint, the useful range is clearly at the conservative end (`q90` or `max`), not `mean` or `median`.
- Even with that sweep, none of the new aggregation choices beat the earlier hidden-only `h=0.15`, `max` result.

### Clean rerun without trace / stage timing / server metrics

To separate real predictor overhead from instrumentation overhead, the `c=16` runs were repeated with:

- `enable_dflash_cycle_trace = False`
- `enable_dflash_stage_timing = False`
- `enable_server_metrics = False`

Artifacts:

- clean baseline:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_base_20260311_223354/predictor_online_ab_fixed128_clean_base_20260311_223354.md`
- clean boundary-aware `q90`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_bw_h020_q90_20260311_223533/predictor_online_ab_fixed128_clean_bw_h020_q90_20260311_223533.md`
- clean boundary-aware `max`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_bw_h020_max_20260311_223709/predictor_online_ab_fixed128_clean_bw_h020_max_20260311_223709.md`

Results:

| policy | tok/s | delta vs clean baseline | tau | accept rate | verify/s | wall/verify |
|---|---:|---:|---:|---:|---:|---:|
| clean baseline | `2903.73` | `0.00%` | `6.326` | `0.354` | `477.92` | `0.002092 s` |
| clean `q90` | `2764.08` | `-4.81%` | `6.279` | `0.395` | `457.77` | `0.002185 s` |
| clean `max` | `2759.85` | `-4.95%` | `6.360` | `0.368` | `451.82` | `0.002213 s` |

Comparison to the traced versions:

| policy | traced tok/s | clean tok/s | improvement from removing instrumentation |
|---|---:|---:|---:|
| baseline | `2845.50` | `2903.73` | `+2.05%` |
| `q90` | `2634.81` | `2764.08` | `+4.91%` |
| `max` | `2635.93` | `2759.85` | `+4.70%` |

Conclusion:

- Instrumentation was costing real throughput:
  - about `2%` on baseline
  - about `4.7%` to `4.9%` on predictor runs
- After removing that instrumentation, predictor `q90` and `max` are still slower than baseline by about `5%`.
- So the slowdown is not just tracing overhead.
- `max` still keeps verify too close to full block, so it does not remove enough verify work to offset predictor-path overhead.
- `q90` is effectively tied with `max` in clean mode, which suggests the main issue is not the `max` reduction itself but the overall predictor control path plus insufficient verify savings.

### Hazard checkpoint online A/B: clean `aggregate=max`

The full-data hazard checkpoint was then tested online on the same fixed128 slice with the same clean settings:

- `enable_dflash_cycle_trace = False`
- `enable_dflash_stage_timing = False`
- `enable_gpu_monitor = False`
- `aggregate=max`

Artifacts:

- clean baseline:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_hazard_base_20260312_005514/predictor_online_ab_fixed128_clean_hazard_base_20260312_005514.md`
- hazard checkpoint `h=0.20`, `max`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_hazard_h020_max_20260312_005937/predictor_online_ab_fixed128_clean_hazard_h020_max_20260312_005937.md`
- hazard checkpoint `h=0.25`, `max`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_clean_hazard_h025_max_20260312_010220/predictor_online_ab_fixed128_clean_hazard_h025_max_20260312_010220.md`

Results:

| policy | tok/s | delta vs clean baseline | tau | accept rate | verify/s |
|---|---:|---:|---:|---:|---:|
| clean baseline | `2898.19` | `0.00%` | `6.326` | `0.354` | `477.01` |
| hazard `h=0.20`, `max` | `2679.94` | `-7.53%` | `6.353` | `0.381` | `440.90` |
| hazard `h=0.25`, `max` | `2552.29` | `-11.93%` | `6.347` | `0.386` | `419.79` |

Conclusion:

- Both hazard runs preserve baseline tau online.
- `h=0.20` is clearly better than `h=0.25` on throughput.
- The hazard objective did not turn predictor gating into an online throughput win.
- Relative to the earlier best clean predictor point, the hazard checkpoint is still worse:
  - earlier boundary-aware `h=0.20`, `max`: `2759.85 tok/s`
  - hazard `h=0.20`, `max`: `2679.94 tok/s`

### Predictor online A/B: grouped verify

Grouped verify was tested with the same predictor policy as the best `aggregate=max` run:

- threshold `h=0.15`
- predictor mode
- confidence gate min verify tokens `1`

Artifacts:

- exact per-request groups:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h015_grouped_exact_20260311_203441/predictor_online_ab_fixed128_h015_grouped_exact_20260311_203441.md`
- bucketed groups `4 8 12 16`:
  - `/workspace/DSC291/dflash-predictor-head/logs/workstreams/predictor_online_ab_fixed128_h015_grouped_b481216_20260311_203948/predictor_online_ab_fixed128_h015_grouped_b481216_20260311_203948.md`

Results:

| policy | tok/s | delta vs baseline | tau | accept rate | verify/s | mean verify len | verify time/cycle |
|---|---:|---:|---:|---:|---:|---:|---:|
| `h=0.15, aggregate=max` | `2653.84` | `-6.74%` | `6.326` | `0.382` | `436.00` | `14.733` | `1.347 ms` |
| `h=0.15, grouped exact` | `777.87` | `-72.66%` | `6.029` | `0.679` | `135.08` | `14.721` | `5.007 ms` |
| `h=0.15, grouped 4/8/12/16` | `1281.12` | `-54.98%` | `6.122` | `0.592` | `217.72` | `14.686` | `2.967 ms` |

Additional observations:

- grouped exact applied on `6433 / 6576` cycles
- grouped `4/8/12/16` applied on `6220 / 6409` cycles
- grouped exact kept `verify_can_run_cuda_graph = True` for all cycles
- grouped `4/8/12/16` kept `verify_can_run_cuda_graph = True` for `98.4%` of cycles

Conclusion:

- Grouped verify is much worse than the single-batch `aggregate=max` policy on this workload.
- The current per-request verify predictions are too fragmented to make grouped verify efficient.
- Exact grouped verify is especially bad because it turns one verify pass into many tiny verify sub-batches.
- Coarse bucketing helps relative to exact groups, but is still far worse than `aggregate=max`.

## Overall ranking of batch aggregates

For the current predictor checkpoint and direct prefix-survival policy:

- best for tau preservation: `max`
- second-best: `median`
- worst: `q10`

Practical interpretation:

- `max` is the only tested aggregate that maintained baseline tau online.
- `median` improves over `q10` but is still not safe if the goal is to match baseline tau.
- `q10` is not suitable for this predictor setup.

## Current state of the code

This session updated the predictor control rule to direct prefix-survival thresholding.

Relevant files:

- `scripts/train_dflash_block_predictor.py`
- `scripts/eval_dflash_predictor.py`
- `scripts/profile_dflash_predictor_by_index.py`
- `third_party/sglang/python/sglang/srt/speculative/dflash_worker.py`

Committed state:

- superproject commit: `00603aa`
- submodule commit: `a63d7c870`

## Recommended next steps

1. Run online fixed128 A/B for the new boundary-aware checkpoint, starting with `aggregate=max` at `h=0.20` and `h=0.25`.
2. Compare the new checkpoint against the old one at the same online settings before changing aggregation again.
3. Keep one verify call per cycle as a hard constraint; do not revisit grouped verify.
4. Do not deploy predictor gating until the new checkpoint shows an online throughput win under tau preservation.
