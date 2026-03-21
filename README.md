# DFlash Extensions

This repository contains DFlash evaluation code plus extension tracks for:

- Multi-candidate speculative verification in Transformers (`sample_multi` with `tree`/`batch` verify).
- Dynamic verify-length scheduling (fixed/adaptive confidence gating).
- Predictor-head training/evaluation pipeline for verify-boundary estimation.
- SGLang integration path through `third_party/sglang`.

## Repository Layout

- Transformers benchmarks:
  - `benchmark.py`
  - `benchmark_candidate_solutions.py`
  - `benchmark_sample_multi_tree.py`
  - `benchmark_dynamic_schedule.py`
- SGLang benchmark:
  - `benchmark_sglang.py`
- Predictor scripts:
  - `scripts/extract_dflash_predictor_dataset.py`
  - `scripts/train_dflash_block_predictor.py`
  - `scripts/eval_dflash_predictor.py`
- SGLang submodule:
  - `third_party/sglang`

## Quick Setup

```bash
conda create -n dflash python=3.12 -y
conda activate dflash

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./third_party/sglang/python --no-deps
```

## Core Command (Best Extension Track)

Transformers multi-candidate tree verification (`sample_multi`):

```bash
python benchmark_candidate_solutions.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset aime25 \
  --block-size 16 \
  --max-samples 30 \
  --max-new-tokens 512 \
  --candidate-mode sample_multi \
  --max-candidates 16 \
  --candidate-verify-mode tree \
  --candidate-verify-static-shape \
  --verify-cache-clone-mode inplace \
  --detailed-cycle-metadata
```

## Additional Reproduction Instructions

Detailed reproducibility instructions, including pod setup and sweep commands, are in `REPRODUCE.md`.
