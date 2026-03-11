# Subagent: Block Predictor

Mission:
- Train an offline predictor for DFLASH verify length / block length from the
  feature shards that were already generated.
- Only integrate into runtime after offline evaluation shows a clear win.

Available artifacts:
- Pod prompt mix:
  - `/workspace/DSC291/dflash/outputs/predictor_training_mix_20260310_013726/predictor_train_mix_10000.jsonl`
  - `/workspace/DSC291/dflash/outputs/predictor_training_mix_20260310_013726/predictor_dev_mix_1043.jsonl`
  - `/workspace/DSC291/dflash/outputs/predictor_training_mix_20260310_013726/predictor_mix_manifest.json`
- Pod feature shards:
  - `/workspace/DSC291/dflash/outputs/predictor_train_trace_qwen4b_20260309/feature_shards/predictor_features_gpu0_tp0_dp0_index.json`
  - `/workspace/DSC291/dflash/outputs/predictor_train_trace_qwen4b_20260309/feature_shards/predictor_features_gpu0_tp0_dp0_shard*.pt`

Dataset schema:
- Written by
  `third_party/sglang/python/sglang/srt/speculative/predictor_dataset.py`
- Each row stores:
  - `request_id`
  - `cycle_idx`
  - `draft_pos`
  - `runtime_block_size`
  - `verify_token_num`
  - `proposed_draft_tokens`
  - `accepted_draft_tokens`
  - `token_accepted`
  - `first_reject_here`
  - `draft_token_id`
  - `draft_hidden`

Relevant scripts:
- `scripts/build_predictor_training_mix.py`
- `scripts/collect_predictor_dataset_sglang.py`
- `scripts/extract_dflash_predictor_dataset.py`

Recommended modeling order:
1. Build an offline supervised baseline first.
2. Predict per-position rejection probability from `draft_hidden`, `draft_pos`,
   and optionally `runtime_block_size`.
3. Derive `verify_token_num` from the first position whose reject probability
   crosses a threshold.
4. Compare against a direct-length classifier/regressor only after the
   per-position baseline is working.

Evaluation targets:
- Exact or near-exact `verify_token_num`
- Retained `tau`
- Retained acceptance ratio
- Proxy runtime objective:
  - minimize verified tokens while preserving accepted prefix quality

Do not train on:
- `aime25`
- `gsm8k fixed128`
- `humaneval`
- `mt-bench`

These are already encoded as benchmark-only datasets in
`scripts/build_predictor_training_mix.py`.

Immediate tasks:
1. Load the shard index and inspect shard sizes and hidden dimension.
2. Split by `request_id`, not by flat rows, to avoid leakage across cycles from
   the same request.
3. Train a small MLP baseline on the shard data.
4. Export offline metrics and failure slices before any server integration work.

Runtime integration should come later:
- First prove the predictor beats fixed-threshold confidence gating offline.
- Then wire it into SGLang as a read-only inference module behind a flag.
