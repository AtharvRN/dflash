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

### Recent Token-ID Ablation

The fused context may not expose enough discrete lexical information for exact
horizon prediction. Add a causal token-identity ablation:

```text
input:
  latest fused context vector c_t
  recent committed token ids [x_{t-W+1}, ..., x_t]

token tower:
  token embedding + learned position embedding
  small GRU over the left-padded recent-token window
  final valid token state

head:
  concat(project(c_t), token_gru_state) -> MLP -> P(H in {0..15})
```

This keeps inference overhead small: one embedding lookup table, one tiny GRU
over a short window such as `W=32`, and the same MLP horizon head.

Important data constraint:

```text
The existing 3.51M Qwen3-4B trace collected on 2026-07-16 does not contain
predraft_token_ids.npy. It cannot train this ablation.
```

Token-ID experiments require recollecting both train and held-out eval traces
with:

```bash
--log-predraft-token-ids --token-window 32
```

Then materialize and run:

```bash
TRACE=/workspace/dflashv2_data/traces/dflashv2_qwen3_4b_b16_instruct100k_tokenids \
MATH=/workspace/dflashv2_data/traces/math500_qwen3_4b_b16_eval_tokenids \
bash scripts/run_dflashv2_token_horizon_experiment.sh
```

The token sweep compares:

```text
fused_ce_dist0p2
token_gru_ce_dist0p2
token_gru_softce_tau1_dist0p2
token_gru_ce_emd0p5_dist0p1
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

## 2026-07-14 Qwen3-8B Math500 Horizon-Policy Results

This section records the latest results after switching the main offline
evaluation target to Qwen3-8B on Math500. These results use fixed `B=16` traces
as the source of accepted-length labels and evaluate block choices offline.

### Evaluation Setup

```text
target model:       Qwen/Qwen3-8B
draft model:        z-lab/Qwen3-8B-DFlash-b16
held-out eval set:  Math500
max block trace:    B=16
policy arms:        {4, 8, 12, 16}
survival outputs:   P(A >= 1), ..., P(A >= 15)
selection rule:     choose smallest B whose predicted accepted length reaches
                    alpha * predicted accepted length at B16
main constraint:    accepted-length retention >= 0.95
```

### Fixed Baseline And Oracle

```text
fixed B16:
  mean accepted draft length = 6.3657
  mean acceptance ratio      = 0.4244

oracle smallest sufficient block:
  mean block                 = 8.7366
  mean acceptance ratio      = 0.7013
```

The oracle result shows that the offline decision problem still has substantial
headroom. If the policy could reliably predict the per-cycle accepted horizon,
it could keep nearly the same accepted length while using much smaller blocks.

### 1M-Row Last-Fused-Vector MLP

Training data:

```text
source trace rows: 1,000,002
train rows:        950,002
validation rows:    50,000
input feature:     latest DFlash fused context vector
feature size:      4096
model:             MLP -> 15 survival logits
```

Best Math500 result from the 1M sweep:

```text
run:                       bw3_aux0p1
selected alpha:            0.84
mean block:                11.7478
mean accepted draft length: 6.0479
accepted-length retention: 0.9501
mean acceptance ratio:     0.5281
```

This remains the strongest current offline policy result. Compared with fixed
`B=16`, it increases acceptance ratio from `0.4244` to `0.5281` while retaining
about 95% of the accepted draft length.

The 1M-row result is very close to the earlier 500K-row result, so simply
scaling the same last-fused-vector MLP dataset did not materially close the
oracle gap.

### 1M-Row Stateful GRU Experiment

Motivation: test whether maintaining a learned temporal state over previous
cycles improves the policy beyond the latest fused vector alone.

Architecture:

```text
input per cycle:    latest DFlash fused context vector, 4096 dims
projection:         Linear(4096 -> proj_dim), GELU, LayerNorm, Dropout
temporal encoder:   GRU over cycles from the same request
head:               MLP -> 15 survival logits
auxiliary head:      optional arm classifier over {4, 8, 12, 16}
loss:               survival BCE + length SmoothL1 + monotonic penalty
                    + auxiliary arm CE
```

Math500 result from the best checkpoint:

```text
selected alpha:            0.86
mean block:                12.0046
mean accepted draft length: 6.1037
accepted-length retention: 0.9588
mean acceptance ratio:     0.5195
```

Alpha table on Math500:

```text
alpha=0.80: block=10.9145, accepted=5.8470, retention=0.9185, ratio=0.5538
alpha=0.82: block=11.2686, accepted=5.9479, retention=0.9344, ratio=0.5426
alpha=0.84: block=11.6423, accepted=6.0336, retention=0.9478, ratio=0.5308
alpha=0.86: block=12.0046, accepted=6.1037, retention=0.9588, ratio=0.5195
alpha=0.88: block=12.3722, accepted=6.1648, retention=0.9684, ratio=0.5082
alpha=0.90: block=12.7774, accepted=6.2201, retention=0.9771, ratio=0.4965
alpha=0.92: block=13.2633, accepted=6.2708, retention=0.9851, ratio=0.4830
alpha=0.94: block=13.7980, accepted=6.3079, retention=0.9909, ratio=0.4687
alpha=0.95: block=14.0974, accepted=6.3244, retention=0.9935, ratio=0.4612
alpha=0.96: block=14.4172, accepted=6.3393, retention=0.9959, ratio=0.4536
alpha=0.98: block=15.1667, accepted=6.3607, retention=0.9992, ratio=0.4375
```

Interpretation:

```text
The GRU/stateful policy did not clearly beat the stateless MLP.
At the selected >=0.95 retention point, it has higher accepted length
but lower acceptance ratio than the MLP because it chooses larger blocks.
At alpha=0.84 it reaches ratio 0.5308, but retention is 0.9478, just below
the 0.95 constraint.
```

Current conclusion: temporal state may still help, but this simple GRU
formulation is not yet the missing signal. The best current baseline remains
the latest-fused-vector MLP with survival outputs.

### Current Best Offline Comparison

```text
fixed B16:
  mean accepted = 6.3657
  ratio         = 0.4244

last-fused MLP, 1M rows:
  mean block    = 11.7478
  mean accepted = 6.0479
  retention     = 0.9501
  ratio         = 0.5281

stateful GRU, 1M rows:
  mean block    = 12.0046
  mean accepted = 6.1037
  retention     = 0.9588
  ratio         = 0.5195

oracle:
  mean block    = 8.7366
  ratio         = 0.7013
```

### Next Candidate: Fused Context Plus Confidence Stats

The next model should preserve the deployable stateless structure while adding
small confidence features:

```text
input:  latest DFlash fused context vector
      + verifier latest-token confidence stats
      + optional previous-cycle summary stats

model:  MLP -> 15 survival logits
```

Start with a small confidence feature set:

```text
verifier latest-token entropy
verifier latest-token top-1 probability
verifier latest-token top-1/top-2 margin
previous accepted length
previous acceptance ratio
previous chosen block size
```

Potential ablations:

```text
A. fused_last_mlp
   latest fused context only

B. fused_last_plus_verifier_confidence
   latest fused context plus verifier entropy/top1/margin

C. fused_last_plus_prev_outcome
   latest fused context plus previous accepted length/ratio/block

D. fused_last_plus_all_small_stats
   latest fused context plus verifier confidence and previous outcome stats
```

The key question is whether explicit confidence statistics help close the gap
between the current MLP ratio `0.5281` and oracle ratio `0.7013` without making
the policy too handcrafted or hardware-specific.

## Post-Draft Verification-Length Predictor

The next direction is to move the decision point from before drafting to after
drafting but before target verification.

Instead of asking:

```text
Before drafting, how many tokens should DFlash draft?
```

ask:

```text
After drafting a full B16 block, how many drafted positions are worth verifying
with the target model?
```

This is less pure as a pre-draft block-size policy, but it is more directly
aligned with the profiling result: at high concurrency, target verification is
around 77-80% of cycle time and grows strongly with block size.

### Allowed Inputs

A verification-length policy may use features available after the drafter has
produced the candidate block, but before the target model verifies it:

```text
allowed:
  latest DFlash fused context vector
  drafted token ids
  drafter logits/confidence for each drafted position
  drafter hidden states for each drafted position
  drafter entropy / token probability / top1-top2 margin

not allowed:
  target logits over drafted tokens
  target entropy over drafted tokens
  target probability assigned to drafted tokens
  draft/target agreement
```

The disallowed features require the expensive full target verification pass and
therefore cannot be used to decide the verification length.

### Initial Feature Set

The first implemented trace extension stores:

```text
postdraft_confidence.npy shape: [rows, 15, 4]

columns:
  draft_entropy
  draft_token_prob
  draft_top1_top2_margin
  draft_token_logprob
```

The first verification-length model uses:

```text
input = concat(
  latest DFlash fused context vector,       # 4096 dims for Qwen3-8B
  flatten(postdraft_confidence[15, 4])      # 60 dims
)

model = MLP -> 15 survival logits
```

The label is unchanged:

```text
A = accepted_draft_len measured from a full B16 target verification
y_k = 1[A >= k], k=1..15
```

At inference, the policy predicts a survival curve after drafting and selects a
verification length from `{4, 8, 12, 16}` using the same retention-constrained
decision rule as the pre-draft horizon policy.

### Why This Might Be More Promising

The pre-draft predictor only sees the current prefix state. The post-draft
verification predictor also sees the actual proposed tokens and how confident
the drafter was at each proposed position. This should be closer to the true
acceptance event:

```text
accepted token k depends on whether target agrees with the already drafted
token at position k, not only on the prefix before drafting.
```

This makes the problem less theoretically clean but likely easier
statistically.

### First Experiment

Collect a small post-draft confidence trace:

```bash
python scripts/collect_dflashv2_horizon_traces.py \
  --manifest /workspace/dflashv2_data/manifests/nemotron_codealpaca_train.jsonl \
  --output-dir /tmp/dflashv2_verify_len_smoke_train \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --max-cycles 50000 \
  --max-new-tokens 256 \
  --block-size 16 \
  --context-window 1 \
  --rows-per-shard 4096 \
  --temperature 0.0 \
  --log-postdraft-confidence
```

Train the first verification-length predictor:

```bash
python scripts/train_dflashv2_verify_length_predictor.py \
  --trace-dir /tmp/dflashv2_verify_len_smoke_train \
  --eval-trace-dir /tmp/dflashv2_verify_len_smoke_math500 \
  --output-dir /workspace/dflashv2_data/runs/verify_len_mlp_smoke \
  --epochs 8 \
  --batch-size 4096 \
  --boundary-weight 2.0 \
  --aux-arm-weight 0.1 \
  --monotonicize-eval \
  --selection-min-retention 0.95
```

Compare against the current pre-draft best:

```text
pre-draft last-fused MLP:
  mean accepted = 6.0479
  retention     = 0.9501
  ratio         = 0.5281
```

The post-draft verification-length idea is promising only if it improves this
ratio at similar retention. A useful first target is:

```text
retention >= 0.95
acceptance ratio > 0.55
```

If it cannot beat the pre-draft MLP with access to drafted-token confidence,
then the bottleneck is probably not the model class but the intrinsic
predictability of target agreement from drafter-side features.

### DSPARK-Style DFlash Confidence Head

A closer analogue to DSPARK is now implemented as a separate path. Instead of
feeding only scalar entropy/probability summaries into an MLP, this version uses
the actual DFlash per-position draft hidden states.

Trace collection:

```bash
python scripts/collect_dflashv2_horizon_traces.py \
  --manifest /workspace/dflashv2_data/manifests/nemotron_codealpaca_train.jsonl \
  --output-dir /tmp/dflashv2_dspark_conf_train \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --max-cycles 50000 \
  --max-new-tokens 256 \
  --block-size 16 \
  --context-window 1 \
  --rows-per-shard 1024 \
  --temperature 0.0 \
  --log-postdraft-hidden \
  --log-postdraft-confidence
```

Additional fields:

```text
postdraft_hidden.npy
  shape: [rows, 15, hidden_size]
  DFlash draft hidden state h_k for each drafted position.

postdraft_token_ids.npy
  shape: [rows, 16]
  anchor token followed by 15 drafted tokens.

postdraft_confidence.npy
  shape: [rows, 15, 4]
  optional scalar drafter confidence stats.
```

Storage note for Qwen3-8B:

```text
postdraft_hidden size per row = 15 * 4096 * 2 bytes ~= 120 KB
50K rows ~= 6 GB
100K rows ~= 12 GB
1M rows ~= 120 GB
```

So start with 50K-100K rows, not 1M.

Model:

```text
for each drafted position k:
  h_k       = DFlash draft hidden state at position k
  x_{k-1}   = previous token, anchor for k=1 and drafted token k-1 otherwise
  s_k       = optional scalar draft confidence stats

  c_k = sigmoid(MLP([Linear(h_k), Embedding(x_{k-1}), Linear(s_k)]))
```

Here `c_k` is the conditional confidence:

```text
c_k = P(position k survives target verification | positions < k survived)
```

Prefix survival is:

```text
a_j = product_{k <= j} c_k
```

The scheduler uses `a_j` exactly like the earlier survival predictors:

```text
predicted_accepted(B) = sum_{j=1}^{B-1} a_j
choose smallest B satisfying retention rule
```

Training script:

```bash
python scripts/train_dflashv2_dspark_confidence_head.py \
  --trace-dir /tmp/dflashv2_dspark_conf_train \
  --eval-trace-dir /tmp/dflashv2_dspark_conf_math500 \
  --output-dir /workspace/dflashv2_data/runs/dspark_conf_head_smoke \
  --epochs 8 \
  --batch-size 512 \
  --proj-dim 512 \
  --markov-dim 64 \
  --head-hidden-size 512 \
  --use-scalar-confidence \
  --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
  --selection-min-retention 0.95
```

Training labels:

```text
A = accepted_draft_len from full B16 target verification
survival target y_j = 1[A >= j]

conditional loss:
  supervise c_k only up to the first rejected position

survival loss:
  BCE(product_{i <= j} c_i, y_j)
```

Evaluation reports two decision rules:

```text
alpha rule:
  same retention rule used by the earlier pre-draft policies.
  Useful for apples-to-apples comparison with last-fused MLP.

threshold rule:
  more DSPARK-like. Keep a prefix based on cumulative survival threshold,
  then round to the nearest supported arm in {4, 8, 12, 16}.
```

This is the best current DFlashv2 direction because it uses the same post-draft
information that DSPARK uses for confidence-scheduled verification, while
remaining specific to DFlash's context-fused parallel drafter.

## DSPARK Teacher Distillation To Pre-Draft Policy, 2026-07-15

Motivation: use the stronger post-draft DSPARK-style hidden-state head as a
teacher for a cheap pre-draft policy.

Teacher export:

```text
teacher:
  /workspace/dflashv2_data/runs/dspark_conf_head_qwen3_8b_100k_20260715_032434/best.pt

trace:
  /workspace/dflashv2_data/traces/dflashv2_dspark_qwen3_8b_b16_100k_20260715_002253

export:
  /workspace/dflashv2_data/teacher/dspark_qwen3_8b_b16_100k_20260715_002253/teacher_survival.npy
```

The teacher survival curves are exported once as memmapped `.npy` files, so the
pre-draft student training does not repeatedly run the post-draft head.

Student:

```text
input:     latest pre-draft DFlash fused context vector, 4096 dims
model:     last_mlp, Linear(4096 -> 512) + MLP -> 15 survival logits
target:    DSPARK teacher soft survival curve plus hard survival labels
split:     same 50K Nemotron validation rows used for post-draft comparison
loss:      0.7 * teacher BCE
         + 0.3 * hard-label BCE
         + 0.05 * teacher length SmoothL1
         + 0.05 * hard length SmoothL1
         + 0.02 * monotonic penalty
```

Run:

```text
/workspace/dflashv2_data/runs/predraft_distill_dspark_teacher_100k_20260715_230105
```

Best validation result:

```text
epoch:                       10
selected alpha:              0.90
expected_len_mae:            1.812
threshold_len_mae:           1.797
AUROC H>=8:                  0.918
AUROC H>=12:                 0.945
mean block:                  8.087
mean accepted draft length:  2.723
accepted-length retention:   0.952
mean acceptance ratio:       0.349
```

Comparison on the same 50K Nemotron validation split:

```text
pre-draft hard survival, 1M:
  accept ratio = 0.390 at retention 0.953

pre-draft hazard, 1M:
  accept ratio = 0.383 at retention 0.957

pre-draft DSPARK-teacher distillation, 100K:
  accept ratio = 0.349 at retention 0.952

post-draft DSPARK-style head:
  accept ratio = 0.450 at retention 0.954
```

Conclusion: strong teacher distillation did not help the pre-draft policy. The
post-draft teacher sees drafted hidden states and token-confidence information
that the pre-draft student cannot access. Forcing the pre-draft student to match
that teacher appears to blur the decision boundary rather than close the oracle
gap. A follow-up ablation should treat teacher information as a weak auxiliary
signal or distill only arm/boundary decisions, while keeping the hard accepted
horizon as the main target.

Follow-up controls on the same 100K trace:

```text
strong teacher:
  run:       predraft_distill_dspark_teacher_100k_20260715_230105
  weights:   teacher=0.7, hard=0.3
  best:      epoch 10
  MAE:       1.812
  ratio:     0.349
  retention: 0.952

weak teacher:
  run:       predraft_distill_dspark_teacher_100k_tw0p2_hw0p8_20260715_230651
  weights:   teacher=0.2, hard=0.8
  best:      epoch 6
  MAE:       1.849
  ratio:     0.349
  retention: 0.951

hard-only control:
  run:       predraft_distill_control_hardonly_100k_20260715_230913
  weights:   teacher=0.0, hard=1.0
  best:      epoch 6
  MAE:       1.851
  ratio:     0.350
  retention: 0.951
```

This means the distillation result is not merely a bad loss weight. With the
same 100K training trace, teacher supervision is essentially neutral to slightly
negative compared with hard-label training. The larger 1M hard-label pre-draft
run is still better on this validation split (`ratio=0.390` at similar
retention), while the post-draft DSPARK teacher remains much stronger
(`ratio=0.450`). The current evidence supports using the DSPARK head as an
analysis/upper-bound teacher, not as a direct soft-label source for the
pre-draft last-fused-vector MLP.

## Online Pre-Draft MLP Integration, 2026-07-15

Integrated the 4096-d last-fused-context survival checkpoint into the SGLang
DFlash worker:

```text
checkpoint:
  /workspace/dflashv2_data/runs/horizon_sweep_1m_last_cached_qwen3_8b_20260714_183551/bw2_aux0p1/best.pt

architecture:
  latest fused context c_t in R^4096
  Linear(4096 -> 512) + GELU + LayerNorm
  MLP 512 -> 256 -> 128 -> 15 survival logits
  s = cummin(sigmoid(logits))
```

Online implementation notes:

```text
per request:
  store only the latest fused vector on GPU

per cycle:
  stack active request vectors into [batch, 4096]
  run one batched MLP forward
  compute expected accepted length for arms {4, 8, 12, 16}
  choose one batch-level block size using alpha retention
```

This avoids the earlier GRU/history replay path. The deployment used a full
Python package overlay in the pod because the local ragged-verify branch and the
base SGLang image had different package layouts.

Math500 online benchmark:

```text
model: Qwen/Qwen3-8B
draft: z-lab/Qwen3-8B-DFlash-b16
dataset: Math500
concurrency: 64
max_new_tokens: 512
enable_thinking: false
policy: last_mlp, alpha=0.84, arms={4,8,12,16}
cuda graph: disabled

throughput: 3909 tok/s
mean accepted length: 7.296
mean completion tokens: 433.1
```

High-batch cycle profile, filtering `batch_size >= 48`:

```text
cycles: 160
selected blocks: B12=130 cycles, B16=30 cycles
mean selected verify length: 12.75
mean accepted drafts: 5.94

predraft policy:        1.34 ms/cycle
draft forward:         11.03 ms/cycle
target verify forward: 60.20 ms/cycle
verify prepare compact: 3.12 ms/cycle
total profiled:        93.15 ms/cycle
```

Main takeaway:

```text
The 4096-d MLP policy is cheap enough online (~1 ms/cycle at C=64).
The remaining gap is policy quality / batch-level aggregation, not model
inference overhead.
```

## Exact-Horizon Hazard Objective, 2026-07-15

The next pre-draft policy objective should predict the exact accepted horizon
`H` before drafting, with `H in {0, ..., 15}` for a max DFlash block of 16.
The model can still expose survival probabilities for the block-size policy,
but training should use an ordinal/hazard formulation rather than independent
survival BCE.

For threshold `k = 1..15`, predict:

```text
q_k = P(H >= k | H >= k - 1, x)
s_k = P(H >= k | x) = product_{j=1..k} q_j
E[H | x] = sum_k s_k
```

For a true horizon `H`, only reachable thresholds are trained:

```text
loss_hazard =
  - sum_{k=1..H} log q_k
  - 1[H < 15] log(1 - q_{H+1})
```

Add a distance-aware term on the expected horizon:

```text
loss = loss_hazard + lambda * SmoothL1(E[H | x], H)
```

Primary offline model metric:

```text
expected_len_mae = mean(abs(E[H | x] - H))
```

Secondary metrics:

```text
rounded_expected_len_mae
rounded_expected_len_exact
AUROC for H >= {1, 4, 8, 12, 15}
selected_accept_ratio at a fixed selected_accept_retention target
```

The final policy metric is still task-level utility: maximize acceptance ratio
while preserving acceptance retention relative to fixed B16. MAE is the cleanest
model-quality metric, but it is not sufficient by itself because small horizon
errors near arm boundaries can matter more than equal-sized errors away from
boundaries.

Initial run:

```text
run:
  /workspace/dflashv2_data/runs/horizon_hazard_1m_last_cached_qwen3_8b_20260715_213716

model:
  Qwen3-8B DFlash B16, last fused vector in R^4096
  Linear(4096 -> 512) + GELU + LayerNorm
  MLP 512 -> 256 -> 128 -> 15 hazard logits

training:
  objective = hazard
  distance term = 0.10 * SmoothL1(E[H], H)
  aux arm CE weight = 0.10
  rows = 1M cache with 50k validation
  best checkpoint selection = validation expected_len_mae

best epoch:
  epoch 3
  val expected_len_mae = 1.703
  val rounded_expected_len_mae = 1.686
  val selected alpha = 0.90
  val selected retention = 0.955
  val selected accept ratio = 0.367
```

Held-out Math500 offline trace evaluation:

```text
mean target accepted length: 6.366

fixed B16:
  mean accepted = 6.366
  accept ratio = 0.424

hazard policy:
  selected alpha = 0.86
  mean block = 11.808
  mean accepted = 6.078
  retention = 0.955
  accept ratio = 0.527

oracle:
  mean block = 8.737
  mean accepted = 6.366
  retention = 1.000
  accept ratio = 0.701
```

Conclusion: the hazard+distance objective improves the formulation but does not
materially move the Math500 policy result versus the earlier independent
survival BCE model (`accept ratio ~0.528` at similar retention). The bottleneck
is likely not only the loss; we need more predictive signal or a policy that
optimizes boundary/arm decisions more directly while preserving exact-horizon
calibration.

## Stateful Augmented Pre-Draft Policy, 2026-07-15

The next pre-draft model uses the latest fused context plus a small causal
confidence/state vector and trains on full request sequences rather than
shuffled independent cycles.

Input per cycle:

```text
fused_context_t in R^4096

predraft_stats_t in R^8:
  verifier_entropy_latest
  verifier_token_prob_latest
  verifier_top1_top2_margin_latest
  verifier_token_logprob_latest
  prev_accepted_len_norm
  prev_accept_ratio
  prev_budget_norm
  has_prev_cycle
```

The verifier confidence entries are measured from the latest target-produced
token before drafting. This is causal: it is available after the previous
target verification step and before the next DFlash draft.

Architecture:

```text
fused_context_t -> Linear(4096 -> 512) + GELU + LayerNorm
predraft_stats_t -> Linear(8 -> 32) + GELU + LayerNorm
concat -> GRU over request cycles -> MLP -> 15 hazard/survival logits
```

At training time, rows are grouped by `(trace_id, prompt_index)` and sorted by
`cycle_id`, so the GRU hidden state matches the real online update order. At
inference time, this becomes a cheap per-request recurrent state:

```text
h_t = GRUCell(project([fused_context_t, predraft_stats_t]), h_{t-1})
survival_logits_t = MLP(h_t)
```

Loss:

```text
hazard loss
+ SmoothL1(E[H], H)
+ arm utility loss over {4, 8, 12, 16}
+ monotonic penalty when using direct survival outputs
```

The new collection path writes `predraft_stats.npy` into each trace shard. Older
traces can still train this model, but verifier confidence is zero-filled and
only the previous-cycle outcome summary is reconstructed from `accepted_len`.

Training command:

```bash
python scripts/train_dflashv2_stateful_augmented_horizon.py \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu0 \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu1 \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu2 \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu3 \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu4 \
  --trace-dir /workspace/dflashv2_data/traces/dflashv2_1m_qwen3_8b_b16_20260713_183723/gpu5 \
  --output-dir /workspace/dflashv2_data/runs/stateful_augmented_qwen3_8b_1m \
  --objective hazard \
  --proj-dim 512 \
  --stats-proj-dim 32 \
  --hidden-size 256 \
  --epochs 8 \
  --batch-size 64 \
  --length-loss-weight 0.10 \
  --arm-utility-weight 0.10 \
  --arm-utility-lambda 0.02 \
  --selection-min-retention 0.95 \
  --checkpoint-selection selected_accept_ratio
```

Initial result on the existing 100K DSPARK trace:

```text
trace:
  /workspace/dflashv2_data/traces/dflashv2_dspark_qwen3_8b_b16_100k_20260715_002253

important limitation:
  this trace was collected before predraft_stats.npy existed, so verifier
  confidence features are zero-filled. This run tests the recurrent state plus
  reconstructed previous-cycle accepted-length summary only.

run:
  /workspace/dflashv2_data/runs/stateful_augmented_prevonly_qwen3_8b_100k_20260716_002658

best epoch:
  epoch 8

metrics:
  expected_len_mae:            1.903
  threshold_len_mae:           1.872
  AUROC H>=8:                  0.906
  AUROC H>=12:                 0.936
  selected alpha:              0.95
  mean block:                  7.902
  mean accepted draft length:  2.676
  accepted-length retention:   0.952
  mean acceptance ratio:       0.355
```

Interpretation:

```text
The recurrent previous-cycle state gives only a small gain over the 100K
hard-only control (~0.355 vs ~0.350 accept ratio at similar retention).
This is not enough to change the conclusion. The next meaningful test requires
a newly collected trace with real verifier-confidence predraft_stats.
```
