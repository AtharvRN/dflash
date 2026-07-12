# Axis 2: Pre-Draft Block-Size Policy

## Goal

Choose a DFlash block size before each drafting cycle:

```text
B_t in {4, 8, 12, 16}
```

The baseline fixed setting is `B=16`, which means `15` drafted tokens. On the GSM8K test trace:

```text
mean accepted draft length = 5.060
mean acceptance ratio      = 5.060 / 15 = 0.337
```

The policy goal is not raw throughput at batch size 1. The useful proxy is:

```text
increase mean acceptance ratio
while maintaining mean accepted draft length
```

For a chosen block size `B`:

```text
draft_budget(B) = B - 1
accepted_B      = min(accepted_len_from_b16_trace, draft_budget(B))
acceptance_ratio_B = accepted_B / draft_budget(B)
```

## Oracle Frontier

Using the existing fixed `B=16` GSM8K test trace and choosing the smallest sufficient block per cycle:

```text
fixed B=16:
  mean accepted draft length = 5.060
  mean acceptance ratio      = 0.337
  mean draft budget          = 15.000
  mean waste                 = 9.940

oracle smallest sufficient block:
  mean accepted draft length = 5.060
  mean acceptance ratio      = 0.649
  mean draft budget          = 6.609
  mean waste                 = 1.549
```

This confirms that the proxy target has real headroom.

## Reward

For each trace row and candidate block size:

```text
A = accepted_draft_len from the max-block trace
accepted_B = min(A, B - 1)
lost_acceptance_B = A - accepted_B
acceptance_ratio_B = accepted_B / (B - 1)
```

The first ratio-preserving reward is:

```text
reward(B) = acceptance_ratio_B - lambda * lost_acceptance_B
```

`lambda` controls the retention/ratio tradeoff:

```text
lambda too low  -> high ratio, poor accepted-length retention
lambda too high -> collapses back toward B=16
```

Initial sweep:

```text
lambda in {0.25, 0.5, 1.0, 2.0}
```

## Metrics

Primary metrics:

```text
mean_accepted_draft_len
mean_acceptance_ratio
accepted_len_retention_vs_fixed_B16
```

Secondary metrics:

```text
mean_draft_budget
mean_wasted_drafts
under_arm_rate
over_arm_rate
arm_hist
```

Success criterion:

```text
mean_acceptance_ratio > 0.337
accepted_len_retention >= 0.90 initially, then >= 0.95
```

## Internal Features

The preferred input is DFlash's own pre-draft fused context representation, not handcrafted entropy features.

The fused context is:

```python
target_hidden = extract_context_feature(
    target.hidden_states,
    draft_model.target_layer_ids,
)

fused_context = draft_model.hidden_norm(
    draft_model.fc(target_hidden)
)
```

This is the context representation consumed by the DFlash drafter. It is causal before drafting.

## DFlashv2 Horizon Predictor

The model-level objective should be phrased as horizon prediction, not direct
throughput optimization:

```text
Given the current decoded prefix and model state, predict the local
predictability horizon H_t: how many future draft tokens are likely to survive
target verification.
```

For a max DFlash block `B_max = 16`, the label is:

```text
H_t = accepted_draft_len_t,  H_t in {0, 1, ..., 15}
```

This label is measured from normal fixed-`B=16` DFlash traces. It is a property
of the current prefix, target model, and DFlash drafter, not a property of the
serving hardware.

### Output Parameterization

Use a survival/hazard parameterization rather than direct block classification:

```text
survival output:
  s_k = P(H_t >= k | state_t),  k = 1..15

or hazard output:
  q_k = P(token k accepted | tokens 1..k-1 accepted, state_t)
  s_k = product_{j=1..k} q_j
```

The hazard form is attractive because it enforces monotonic survival by
construction. The direct survival form is simpler and can be monotonicized with:

```python
s = torch.cummin(torch.sigmoid(logits), dim=-1).values
```

The block-size decision is kept outside the learned model:

```text
expected_accept(B) = sum_{k=1}^{B-1} s_k
B* = decision_rule(expected_accept, runtime_cost_or_budget)
```

This keeps the predictor reusable across hardware, concurrency, and serving
frameworks.

### Inputs

The primary input should be the DFlash fused context window:

```text
X_t = [c_{t-W+1}, ..., c_t],  c_i = hidden_norm(fc(target_hidden_i))
```

This is the same representation consumed by the DFlash drafter before drafting,
so it is causal and architecture-aligned. Recommended starting dimensions:

```text
window W: 16 or 32 committed tokens
input dim: full DFlash fused-context hidden size, e.g. 4096
learned bottleneck: Linear(input_dim -> 256/512/1024)
```

Optional uncertainty channels can be appended per token:

```text
verifier entropy
verifier pmax
verifier top1-top2 margin
token logprob
```

These should be treated as an ablation rather than the default, because the
cleanest DFlashv2 claim is that the drafter's fused target context already
contains the horizon signal.

Previous accepted length should not be a primary feature. It is an outcome proxy,
not a causal model-state signal. It can remain as a diagnostic baseline.

### Architecture

Start with a lightweight temporal encoder:

```text
input:  [batch, W, d]
encoder: 2-layer GRU or small causal Transformer encoder
pool:   final valid token or attention pooling
head:   MLP -> 15 logits
output: survival or hazard probabilities
```

Recommended first model:

```text
TemporalSurvivalHead
  GRU hidden size: 192 or 256
  MLP: hidden -> hidden/2 -> 15
  dropout: 0.05-0.10
```

Then run these ablations:

```text
A. entropy_latest
   current verifier entropy/pmax/margin only

B. entropy_window
   last W verifier uncertainty values

C. fused_last_mlp
   latest fused context vector only

D. fused_window_gru
   last W fused context vectors

E. fused_window_transformer
   small temporal Transformer over fused context

F. fused_window_plus_entropy
   fused context plus verifier uncertainty channels
```

### Training Loss

For each trace row with label `H_t`, construct binary survival labels:

```text
y_k = 1[H_t >= k],  k = 1..15
```

Use binary cross entropy over all horizons:

```text
L_survival = mean_k BCE(logit_k, y_k)
```

Recommended additions:

```text
monotonic regularizer:
  mean ReLU(s_{k+1} - s_k)

calibration loss or temperature scaling:
  fit on validation split after training

class/horizon weighting:
  slightly upweight larger k because long accepted horizons are rarer
```

Do not train directly on measured runtime cost. Cost belongs to the inference
decision layer, not the horizon predictor.

### Inference

At each DFlash cycle:

```text
1. target verifies/produces the latest committed token(s)
2. extract fused context for committed token(s)
3. update the horizon predictor's rolling context window
4. predict s_k = P(H_t >= k)
5. convert survival curve into a draft block size
6. run DFlash drafting with the selected block size
```

Simple cost-free decision rules for model research:

```text
threshold rule:
  choose largest B such that s_{B-1} >= tau

retention rule:
  choose smallest B such that E[min(H, B-1)] >= alpha * E[min(H, 15)]

budgeted rule:
  choose B that maximizes E[min(H, B-1)] - lambda * (B-1)
```

Deployment can replace these with a measured cost adapter:

```text
B* = argmax_B E[min(H, B-1)] / C_runtime(B)
```

but the learned horizon model stays unchanged.

### Evaluation

Evaluate the predictor before evaluating throughput:

```text
survival AUROC for H >= 4, 8, 12
survival calibration error
horizon MAE / ordinal error
expected accepted-token retention
chosen block histogram
oracle gap against true H_t
```

The key research question is:

```text
Is the DFlash predictable horizon recoverable from pre-draft model state?
```

Throughput and serving integration should come after this question is answered.

### Training Data Choice

For DFlashv2 horizon-predictor training, use the same source mixture reported by
the DFlash paper for its main draft models:

```text
source prompts:
  NVIDIA Nemotron Post-Training Dataset V2
  CodeAlpaca

scale:
  DFlash reports around 800K source samples

alignment:
  DFlash does not train directly on original dataset responses; it constructs
  target-model-generated responses for better target alignment.
```

For the horizon predictor, the equivalent procedure is:

```text
1. sample prompts/instructions from the same source mixture
2. decode with the frozen target + DFlash drafter using fixed B=16
3. store pre-draft state features for each cycle
4. label each cycle with the observed accepted draft length H_t
```

Do not train on final evaluation benchmark prompts. Math500, HumanEval, MBPP,
LCB, GSM8K, and MT-Bench-style reported test prompts should remain evaluation
only unless explicitly used as a separate ablation.

Start with a smaller DFlashv2 trace subset before scaling:

```text
smoke:      5K-10K cycles
pilot:      50K cycles
main:       200K-300K cycles
full scale: 500K+ cycles if the pilot shows signal
```

The split should be by source prompt, not by cycle, so all cycles from a prompt
belong to exactly one of train/validation/test.

### DFlashv2 Pipeline Commands

Build prompt manifests from the downloaded DFlash source datasets:

```bash
cd /workspace/dflash-fresh-zlab-main

python scripts/build_dflashv2_prompt_manifest.py \
  --nemotron-root /workspace/dflashv2_data/datasets/nemotron_v2 \
  --codealpaca-json /workspace/dflashv2_data/datasets/codealpaca/code_alpaca_20k.json \
  --output-dir /workspace/dflashv2_data/manifests \
  --prefix nemotron_codealpaca \
  --val-fraction 0.02 \
  --seed 0
```

Collect a small full-fused-context trace smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_dflashv2_horizon_traces.py \
  --manifest /workspace/dflashv2_data/manifests/nemotron_codealpaca_train.jsonl \
  --output-dir /workspace/dflashv2_data/traces/qwen3_8b_b16_fullctx_smoke_train \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --max-cycles 10000 \
  --max-new-tokens 256 \
  --block-size 16 \
  --context-window 16 \
  --rows-per-shard 2048 \
  --temperature 0.0
```

Collect validation traces from the held-out prompt manifest:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_dflashv2_horizon_traces.py \
  --manifest /workspace/dflashv2_data/manifests/nemotron_codealpaca_val.jsonl \
  --output-dir /workspace/dflashv2_data/traces/qwen3_8b_b16_fullctx_smoke_val \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --max-cycles 2000 \
  --max-new-tokens 256 \
  --block-size 16 \
  --context-window 16 \
  --rows-per-shard 2048 \
  --temperature 0.0
```

Train the first full-fused-context horizon predictor:

```bash
python scripts/train_dflashv2_horizon_predictor.py \
  --train-dir /workspace/dflashv2_data/traces/qwen3_8b_b16_fullctx_smoke_train \
  --val-dir /workspace/dflashv2_data/traces/qwen3_8b_b16_fullctx_smoke_val \
  --output-dir /workspace/dflashv2_data/runs/horizon_gru_fullctx_smoke \
  --architecture gru \
  --proj-dim 512 \
  --hidden-size 256 \
  --epochs 8 \
  --batch-size 256 \
  --monotonicize-eval
```

Evaluate a saved checkpoint:

```bash
python scripts/train_dflashv2_horizon_predictor.py \
  --eval-only \
  --checkpoint /workspace/dflashv2_data/runs/horizon_gru_fullctx_smoke/best.pt \
  --val-dir /workspace/dflashv2_data/traces/qwen3_8b_b16_fullctx_smoke_val \
  --output-dir /workspace/dflashv2_data/runs/horizon_gru_fullctx_smoke_eval \
  --architecture gru \
  --proj-dim 512 \
  --hidden-size 256 \
  --batch-size 256 \
  --monotonicize-eval
```

## Trace Fields

When collecting with:

```bash
--log-internal-features --internal-feature-window 16 --internal-feature-dim 128
```

each row stores:

```text
inputs.dflash_context
  Latest projected fused context vector.
  Shape: [128]

inputs.dflash_context_window
  Padded sequence of recent projected fused context vectors.
  Shape: [16, 128]

inputs.dflash_context_window_mask
  Padding mask for the context window.
  Shape: [16]
```

The projection is a fixed random Gaussian projection, seeded by `--internal-feature-seed`.

## Data Collection Plan

Use `a100-gpu-test` with two shards, one per A100. Active trace IO should go under the pod home directory because `/workspace` is network-backed and has shown high IO wait on large JSONL files.

```text
active output:
  ~/dflash_axis2_runs/traces/

active logs:
  ~/dflash_axis2_runs/logs/

archive/copy destination after completion:
  /workspace/dflash-fresh-zlab-main/runs/traces/internal/
```

Launch train split as two shards:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/dflash-fresh-zlab-main \
/opt/conda/envs/cbm/bin/python /workspace/dflash-fresh-zlab-main/scripts/collect_dflash_traces.py \
  --split train \
  --output ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128_shard0of2.jsonl \
  --block-size 16 \
  --max-new-tokens 256 \
  --log-internal-features \
  --internal-feature-window 16 \
  --internal-feature-dim 128 \
  --num-shards 2 \
  --shard-index 0

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/dflash-fresh-zlab-main \
/opt/conda/envs/cbm/bin/python /workspace/dflash-fresh-zlab-main/scripts/collect_dflash_traces.py \
  --split train \
  --output ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128_shard1of2.jsonl \
  --block-size 16 \
  --max-new-tokens 256 \
  --log-internal-features \
  --internal-feature-window 16 \
  --internal-feature-dim 128 \
  --num-shards 2 \
  --shard-index 1
```

After both shards finish:

```bash
cat ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128_shard0of2.jsonl \
    ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128_shard1of2.jsonl \
  > ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128.jsonl

mkdir -p /workspace/dflash-fresh-zlab-main/runs/traces/internal
cp ~/dflash_axis2_runs/traces/gsm8k_train_internal_w16d128*.jsonl \
   /workspace/dflash-fresh-zlab-main/runs/traces/internal/
```

## Ablations

Run these in order:

```text
A. internal_last_mlp
   input: dflash_context
   architecture: MLP

B. internal_window_gru_8
   input: last 8 rows of dflash_context_window
   architecture: GRU

C. internal_window_gru_16
   input: full dflash_context_window
   architecture: GRU

D. internal_window_pool_mlp
   input: concat(last, mean, std) over dflash_context_window
   architecture: MLP

E. internal_plus_uncertainty
   input: dflash_context plus latest verifier entropy/pmax/margin/logprob
   architecture: MLP
```

Start with A and C:

```text
internal_last_mlp gives the cleanest low-overhead baseline.
internal_window_gru_16 tests whether local context trajectory improves block-size prediction.
```

## Qwen3-8B Math500 Fixed-Block Baseline

Corrected high-concurrency run:

```text
run_dir=/workspace/dflash-axis2-runs/math500_qwen3_8b_fixed_highc/run_20260711_190558
model=Qwen/Qwen3-8B
draft=z-lab/Qwen3-8B-DFlash-b16
dataset=Math500
num_prompts=128
max_new_tokens=512
enable_thinking=false
concurrency={32,48,64}
block={8,12,16}
```

Important measurement detail: `num_prompts` must be at least the requested
concurrency. An earlier partial run used `num_prompts=32`, so its `C=64` point
only had 32 active requests and should not be used.

Throughput summary:

```text
C=32:
  B8   1569.24 tok/s, accept_len=5.62
  B12  1979.35 tok/s, accept_len=6.87
  B16  1436.70 tok/s, accept_len=7.84

C=48:
  B8   1797.25 tok/s, accept_len=5.59
  B12  1999.88 tok/s, accept_len=6.82
  B16  1647.32 tok/s, accept_len=7.81

C=64:
  B8   1751.33 tok/s, accept_len=5.64
  B12  2252.11 tok/s, accept_len=6.91
  B16  1914.90 tok/s, accept_len=7.87
```

For Qwen3-8B on Math500 in this setup, fixed `B=12` is the winner at all three
high-concurrency points. `B=16` accepts more tokens per cycle, but the larger
draft/verify block is expensive enough that throughput drops.

Timing-profile run:

```text
run_dir=/workspace/dflash-axis2-runs/math500_qwen3_8b_fixed_highc_profile/run_20260711_192808
concurrency=64
num_prompts=128
max_new_tokens=512
```

The timing run enables CUDA-synchronizing per-cycle profiling, so its throughput
is not comparable to the baseline run. Use it only for component timing:

```text
C=64 timing, mean profiled cycles only:
  B8:
    cycle_total_ms=88.12
    draft_forward_ms=9.13
    target_verify_forward_ms=68.14
    accept_verify_ms=4.77
    draft_sample_ms=2.86
    cycle_mean_commit_len=5.45

  B12:
    cycle_total_ms=113.38
    draft_forward_ms=11.31
    target_verify_forward_ms=89.76
    accept_verify_ms=4.91
    draft_sample_ms=4.11
    cycle_mean_commit_len=6.54

  B16:
    cycle_total_ms=140.41
    draft_forward_ms=13.79
    target_verify_forward_ms=112.25
    accept_verify_ms=5.50
    draft_sample_ms=5.43
    cycle_mean_commit_len=7.37
```

This supports the dynamic-block motivation: reducing block size materially
reduces runtime at high concurrency, mostly through target verify time. The
hard part is preserving enough accepted length while choosing the smaller arm.

### Verification-Time Reduction Target

For the profiled `Qwen3-8B` / Math500 / `C=64` run, target verification is the
dominant part of each DFlash cycle:

```text
B8:   target_verify_forward_ms =  68.14 /  88.12 total = 77.3%
B12:  target_verify_forward_ms =  89.76 / 113.38 total = 79.2%
B16:  target_verify_forward_ms = 112.25 / 140.41 total = 79.9%
```

Growth from `B=8` to `B=16`:

```text
cycle_total_ms:            +52.29 ms, +59.3%
draft_forward_ms:           +4.66 ms, +51.0%
target_verify_forward_ms:  +44.11 ms, +64.7%
accept_verify_ms:           +0.73 ms, +15.3%
```

Growth from `B=12` to `B=16`:

```text
cycle_total_ms:            +27.03 ms, +23.8%
draft_forward_ms:           +2.48 ms, +21.9%
target_verify_forward_ms:  +22.49 ms, +25.1%
accept_verify_ms:           +0.59 ms, +12.0%
```

The practical implication is that block-size optimization is mostly a
verification-length optimization. If we can draft a full-length block but verify
only the useful prefix for each request, we should capture most of the runtime
benefit of smaller blocks while keeping the ability to accept longer prefixes
when the policy predicts they are worthwhile.

This creates a different implementation problem from choosing one uniform
runtime block size for the whole batch:

```text
draft block length:      fixed/full, e.g. 16
verification length:     per request, e.g. 8/12/16
target forward shape:    ragged or bucketed by verification length
batch decision target:   minimize target verification work without truncating
                         requests that would have accepted longer prefixes
```

The easiest implementation is bucketed verification: group requests by predicted
verification length and run one target verify forward per bucket. The risk is
that sequential bucket forwards can erase the savings. The more ambitious
implementation is ragged verification in one batched target forward, where each
request contributes only its selected verification prefix length. That is the
right direction if SGLang's batching and attention metadata can represent the
ragged verify sequences cleanly.

## Survival-Curve Policy

The next policy family predicts the accepted-length survival curve before drafting:

```text
input:  pre-draft DFlash context or DFlash context window
output: P(A >= 1), P(A >= 2), ..., P(A >= 15)
```

This keeps the decision pre-draft, unlike post-draft confidence heads, but gives the model a structured target instead of a direct block label.

For a candidate block size `B`:

```text
predicted_accepted(B) = sum_{k=1}^{B-1} P(A >= k)
```

Then choose the smallest block satisfying:

```text
predicted_accepted(B) >= alpha * predicted_accepted(16)
```

`alpha` controls the retention/ratio tradeoff. Larger `alpha` picks larger blocks and preserves more accepted length.

Initial validation results on the prompt-held-out GSM8K train split:

```text
fixed B=16:
  mean accepted draft length = 5.052
  mean acceptance ratio      = 0.337

oracle smallest sufficient:
  mean accepted draft length = 5.052
  mean acceptance ratio      = 0.648
  mean draft budget          = 6.609

internal_window_gru16_survival:
  alpha=0.85:
    mean accepted draft length = 4.679
    retention                  = 0.926
    mean acceptance ratio      = 0.502
    mean draft budget          = 8.803

  alpha=0.90:
    mean accepted draft length = 4.827
    retention                  = 0.956
    mean acceptance ratio      = 0.468
    mean draft budget          = 9.710

internal_last_mlp_survival:
  alpha=0.85:
    mean accepted draft length = 4.680
    retention                  = 0.926
    mean acceptance ratio      = 0.479
    mean draft budget          = 9.330

  alpha=0.90:
    mean accepted draft length = 4.858
    retention                  = 0.962
    mean acceptance ratio      = 0.442
    mean draft budget          = 10.472
```

The GRU window survival model is the best current operating point. It reaches the target retention regime while still improving acceptance ratio substantially over fixed `B=16`.

Held-out GSM8K test results:

```text
fixed B=16:
  mean accepted draft length = 5.060
  mean acceptance ratio      = 0.337

oracle smallest sufficient:
  mean accepted draft length = 5.060
  mean acceptance ratio      = 0.649
  mean draft budget          = 6.609

internal_window_gru16_survival:
  alpha=0.85:
    mean accepted draft length = 4.686
    retention                  = 0.926
    mean acceptance ratio      = 0.502
    mean draft budget          = 8.802

  alpha=0.90:
    mean accepted draft length = 4.837
    retention                  = 0.956
    mean acceptance ratio      = 0.468
    mean draft budget          = 9.718

internal_last_mlp_survival:
  alpha=0.85:
    mean accepted draft length = 4.690
    retention                  = 0.927
    mean acceptance ratio      = 0.479
    mean draft budget          = 9.350

  alpha=0.90:
    mean accepted draft length = 4.871
    retention                  = 0.963
    mean acceptance ratio      = 0.442
    mean draft budget          = 10.501
```

The held-out test curve closely matches validation. The recommended operating points are:

```text
alpha=0.85 for better ratio:
  retention ~0.926, acceptance ratio ~0.502

alpha=0.90 for safer retention:
  retention ~0.956, acceptance ratio ~0.468
```

## Literature Anchors

DISCO frames static speculation length as suboptimal and trains a lightweight classifier for dynamic speculation length.

SmartSpec uses estimated accepted length/rate and execution cost to choose speculation length under serving load.

PEARL also adapts draft length, motivated by fixed-length speculation waste and draft/verify waiting.

Our DFlash-specific version is:

```text
pre-draft fused DFlash context -> policy head -> block size
```

## Runtime Decoder Integration

The offline survival policy can now run inside DFlash decoding before each draft
cycle:

- `dflash.policy.DFlashSurvivalBlockPolicy` loads a trained `best.pt` survival
  checkpoint, maintains the last 16 projected DFlash fused-context vectors, and
  selects the smallest block in `{4, 8, 12, 16}` whose expected accepted draft
  length reaches `alpha * E[accepted length at B16]`.
- `dflash.dynamic.dflash_generate_dynamic` mirrors the original Transformers
  decode loop but calls the policy at the start of each cycle and runs the draft
  model with that cycle's selected block size.
- `scripts/benchmark_dynamic_survival_policy.py` compares fixed B16 against the
  dynamic policy during actual generation and reports mean accepted draft length,
  mean acceptance ratio, mean block size, block histogram, and token latency.

Initial runtime smoke command:

```bash
/workspace/dflash-axis2-train-uv/bin/python scripts/benchmark_dynamic_survival_policy.py \
  --model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --policy-checkpoint /workspace/dflash-fresh-zlab-main/runs/survival_policy/internal_w16d128/internal_window_gru16_survival_full/best.pt \
  --dataset gsm8k \
  --max-samples 8 \
  --max-new-tokens 128 \
  --alpha 0.90 \
  --compare-output-ids \
  --output-json runs/survival_policy/internal_w16d128/runtime_dynamic_smoke_alpha090.json
```

Runtime smoke on `a100-gpu-test-v2` with `alpha=0.90`, 8 GSM8K prompts, and
128 generated tokens per prompt:

```text
fixed B16:
  mean accepted draft length = 4.768
  mean acceptance ratio      = 0.318
  mean draft budget          = 15.000

dynamic survival policy:
  mean accepted draft length = 4.521
  mean acceptance ratio      = 0.446
  mean draft budget          = 9.625
  block histogram            = B8: 90, B12: 78, B16: 24
```

The dynamic decoder also matches the original fixed-B16 decoder exactly when
the policy arms are forced to `{16}`, which verifies the cache mechanics of the
new runtime path. Exact token equality between fixed B16 and variable block
sizes is not stable under bf16/SDPA for longer generations because target
verification chunk sizes differ.

Runtime alpha sweep on 100 GSM8K prompts, 128 generated tokens per prompt:

```text
fixed B16:
  mean accepted draft length = 4.430
  mean acceptance ratio      = 0.295
  mean draft budget          = 15.000

dynamic survival policy:
  alpha=0.80:
    mean accepted draft length = 3.681
    mean acceptance ratio      = 0.467
    mean draft budget          = 7.455
    block histogram            = B4: 379, B8: 1792, B12: 564, B16: 67

  alpha=0.85:
    mean accepted draft length = 3.829
    mean acceptance ratio      = 0.438
    mean draft budget          = 8.239
    block histogram            = B4: 163, B8: 1695, B12: 717, B16: 144

  alpha=0.90:
    mean accepted draft length = 3.966
    mean acceptance ratio      = 0.409
    mean draft budget          = 9.112
    block histogram            = B4: 31, B8: 1456, B12: 899, B16: 266

  alpha=0.95:
    mean accepted draft length = 4.089
    mean acceptance ratio      = 0.368
    mean draft budget          = 10.484
    block histogram            = B4: 1, B8: 884, B12: 1144, B16: 553
```

The best current tradeoff depends on how much accepted-length retention we want.
`alpha=0.90` is the balanced point from this runtime sweep: it improves
acceptance ratio from 0.295 to 0.409 while retaining about 89.5% of fixed-B16
accepted draft length. `alpha=0.95` is safer if the next experiments prioritize
accepted length over draft efficiency.
