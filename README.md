# DFlash Experiment Reproduction

This repository snapshot contains the core code paths needed to reproduce the experiments used in the report.

## Branch Contents

- Core Transformers benchmarks:
  - `benchmark.py`
  - `benchmark_candidate_solutions.py`
  - `benchmark_sample_multi_tree.py`
  - `benchmark_dynamic_schedule.py`
- Core SGLang benchmark:
  - `benchmark_sglang.py`
- Experiment launchers:
  - `run_sglang_tp1_sweep.sh`
  - `run_sglang_dynamic_c16.sh`
  - `run_sglang_c16_policy_matrix.sh`
  - `run_sglang_static_ds_conc_sweep.sh`
  - `run_benchmark.sh`
- Predictor-head pipeline:
  - `scripts/extract_dflash_predictor_dataset.py`
  - `scripts/train_dflash_block_predictor.py`
  - `scripts/eval_dflash_predictor.py`
  - `scripts/profile_dflash_predictor_by_index.py`
  - `scripts/collect_predictor_dataset_sglang.py`
  - `scripts/build_predictor_training_mix.py`
- SGLang integration code via submodule:
  - `third_party/sglang` (DFLASH PR code included)

## Environment Setup (atharv-rwx-pod)

```bash
kubectl -n wenglab-interpretable-ai exec -it atharv-rwx-pod -- bash
source ~/.bashrc
cd /workspace/DSC291/dflash

conda create -n dflash python=3.12 -y
conda activate dflash

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# Ensure local SGLang (submodule) is used.
python -m pip install -e ./third_party/sglang/python --no-deps
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__)"
sglang serve --help | grep speculative-dflash || true
```

## Quick Repro Commands

### 1) Transformers baseline DFlash

```bash
python benchmark.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name aime25 \
  --max-samples 30 \
  --block-size 16 \
  --max-new-tokens 512
```

### 2) Transformers multi-candidate (tree / batch)

```bash
python benchmark_candidate_solutions.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name aime25 \
  --max-samples 30 \
  --block-size 16 \
  --max-new-tokens 512 \
  --candidate-mode sample_multi \
  --verify-mode tree \
  --max-candidates 16
```

```bash
python benchmark_sample_multi_tree.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name aime25 \
  --max-samples 30 \
  --block-size 16 \
  --max-new-tokens 512 \
  --verify-mode batch \
  --max-candidates 12
```

### 3) SGLang baseline vs multi-candidate

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python benchmark_sglang.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name gsm8k \
  --concurrencies 1 \
  --max-samples 30 \
  --speculative-algorithm dflash
```

### 4) Dynamic scheduling / confidence gating

```bash
python benchmark_dynamic_schedule.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name aime25 \
  --max-samples 30 \
  --block-size 16 \
  --accept-confidence-threshold 0.2
```

### 5) Predictor-head pipeline

```bash
python scripts/extract_dflash_predictor_dataset.py --help
python scripts/train_dflash_block_predictor.py --help
python scripts/eval_dflash_predictor.py --help
```

## Marquee Results (from report run archives)

- Transformers backend (AIME25, c=1):
  - Vanilla DFlash (bs=16): **129.25 tok/s**
  - Multi-candidate tree best (mc=16): **146.60 tok/s**
  - Multi-candidate batch best: throughput improvement vs vanilla, but below tree best.
- SGLang backend (GSM8K, c=1, fixed mc=4):
  - Vanilla DFlash: **453.08 tok/s**
  - MC-tree (mc=4): **376.57 tok/s**
  - MC-batch (mc=4): **229.90 tok/s**

Interpretation: multi-candidate verification is beneficial in the Transformers backend low-concurrency regime, while in SGLang it requires additional systems-level optimization to preserve the verify-path cost model.

## Notes

- Most experiment families in this project were executed as single runs per setting; compare trends primarily within the same run family.
- Keep fixed question windows (`--fixed-question-count`, `--fixed-question-offset`) for apples-to-apples policy comparisons.
