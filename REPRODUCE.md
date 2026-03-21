# Reproduce Experiments

This file provides runnable commands to reproduce major experiment tracks.

## 1) Pod Setup (atharv-rwx-pod)

```bash
kubectl -n wenglab-interpretable-ai exec -it atharv-rwx-pod -- bash
source ~/.bashrc
source /opt/conda/etc/profile.d/conda.sh

cd /workspace/DSC291/dflash
git fetch origin
git checkout submission-ready
git pull --ff-only origin submission-ready
git submodule sync --recursive
git submodule update --init --recursive

conda create -n dflash python=3.12 -y
conda activate dflash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./third_party/sglang/python --no-deps
```

Sanity checks:

```bash
python -c "import torch; print('torch', torch.__version__)"
sglang serve --help | grep speculative-dflash | head -n 5
```

## 2) Transformers Baselines

```bash
python benchmark.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset aime25 \
  --block-size 16 \
  --max-samples 30 \
  --max-new-tokens 512 \
  --save-outputs-path outputs/aime25_dflash_baseline.jsonl \
  --save-cycle-trace-path outputs/aime25_dflash_baseline_cycles.jsonl \
  --collect-profile
```

## 3) Transformers Multi-Candidate Sweep (sample_multi)

Tree mode example:

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
  --detailed-cycle-metadata \
  --save-outputs-path outputs/aime25_mc16_tree.jsonl \
  --save-cycle-trace-path outputs/aime25_mc16_tree_cycles.jsonl \
  --collect-profile
```

Batch mode example:

```bash
python benchmark_candidate_solutions.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset aime25 \
  --block-size 16 \
  --max-samples 30 \
  --max-new-tokens 512 \
  --candidate-mode sample_multi \
  --max-candidates 12 \
  --candidate-verify-mode batch \
  --candidate-verify-static-shape \
  --verify-cache-clone-mode inplace \
  --detailed-cycle-metadata \
  --save-outputs-path outputs/aime25_mc12_batch.jsonl \
  --save-cycle-trace-path outputs/aime25_mc12_batch_cycles.jsonl \
  --collect-profile
```

## 4) Dynamic Verify-Length (Confidence Gating)

```bash
python benchmark_dynamic_schedule.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset aime25 \
  --block-size 16 \
  --max-samples 30 \
  --max-new-tokens 512 \
  --accept-confidence-threshold 0.2
```

## 5) Predictor-Head Pipeline

```bash
python scripts/extract_dflash_predictor_dataset.py --help
python scripts/train_dflash_block_predictor.py --help
python scripts/eval_dflash_predictor.py --help
python scripts/profile_dflash_predictor_by_index.py --help
```

## 6) SGLang Evaluation

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python benchmark_sglang.py \
  --target-model Qwen/Qwen3-4B \
  --draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --dataset-name gsm8k \
  --concurrencies 1,4,8,16 \
  --max-samples 30
```

For fixed-window comparisons, include:

```bash
--fixed-question-count 30 --fixed-question-offset 0
```
