# Candidate Verify Performance Tracker

This file tracks throughput changes for candidate verification experiments on the same workload:

- Dataset: `aime25`
- Samples: `30`
- `max_new_tokens=512`
- Model: `Qwen/Qwen3-4B`
- Draft: `z-lab/Qwen3-4B-DFlash-b16`

## Timeline

| run_tag | code state | bs16_dflash tok/s | multi-cand tok/s | multi vs bs16 |
|---|---|---:|---:|---:|
| `aime25_30_threeway_p2_20260307_035143` | pre-optimization baseline | 130.12 | 130.37 | +0.19% |
| `aime25_30_lowrisk_20260307_053907` | low-risk buffer + metadata gating (`9e96b4e`) | 131.23 | 129.60 | -1.25% |
| `aime25_30_shallow_20260307_054825` | + shallow clone option (`e7e69ec`) | 131.06 | 129.59 | -1.12% |
| `aime25_30_opt2_20260307_055245` | + streamlined `sample_multi` metadata/scoring (`aa36f6e`) | 130.27 | 135.19 | +3.78% |
| `aime25_30_opt3_20260307_061401` | + inplace verify cache + static verify shape (`cf1c64c`) | 131.52 | 137.10 | +4.24% |

## Notes

- Multi-candidate wins when candidate-path overhead is kept low.
- Main overhead buckets observed:
  - candidate-path Python/tensor bookkeeping,
  - cache handling around verify (`clone`/repeat/select path),
  - variable-shape verify batches.
- Current next-step implementation targets:
  - in-place cache strategy (`--verify-cache-clone-mode inplace`),
  - static verify shape (`--candidate-verify-static-shape`).

## Verify-Mode Matrix (One Verify Call Per Cycle)

Run tag: `aime25_verifymode_candidates_fix_20260307_074405`  
Artifact summary: `outputs/aime25_verifymode_candidates_fix_20260307_074405/summary.md`

All rows below satisfy:
- `verify_calls_unique=[1]`
- `commit_calls_unique=[0]`

| verify_mode | max_candidates | tokens/s | mean_tau | mean_num_candidates | draft ms/cycle | verify ms/cycle | cycle ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch | 1 | 91.37 | 6.4349 | 1.0000 | 13.4341 | 54.3067 | 69.3614 |
| batch | 2 | 103.12 | 6.7606 | 1.4956 | 14.3186 | 46.7213 | 64.2971 |
| batch | 4 | 98.37 | 6.7192 | 2.1885 | 16.4487 | 46.8793 | 66.9499 |
| batch | 8 | 97.72 | 6.8025 | 2.8826 | 17.5755 | 47.1163 | 68.3005 |
| tree | 1 | 112.19 | 6.4457 | 1.0000 | 11.3604 | 38.9970 | 56.4935 |
| tree | 2 | 127.74 | 6.6754 | 1.5098 | 8.9299 | 38.0491 | 51.3126 |
| tree | 4 | 134.38 | 6.7606 | 2.1298 | 7.8176 | 37.8734 | 49.3795 |
| tree | 8 | 138.24 | 6.9471 | 2.7933 | 7.8385 | 37.5984 | 49.3172 |

## Verify-Mode Matrix (High Candidate Caps: 10-16)

Run tag: `aime25_verifymode_candidates_hi_20260307_081809`  
Artifact summary (pod): `/workspace/DSC291/dflash/outputs/aime25_verifymode_candidates_hi_20260307_081809/summary.md`

All rows below satisfy:
- `verify_calls_unique=[1]`
- `commit_calls_unique=[0]`

| verify_mode | max_candidates | tokens/s | mean_tau | mean_num_candidates | draft ms/cycle | verify ms/cycle | cycle ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch | 10 | 135.938 | 6.8663 | 3.0103 | 8.3987 | 39.6395 | 49.4342 |
| batch | 12 | 136.032 | 6.8879 | 3.0892 | 8.4708 | 39.6639 | 49.5380 |
| batch | 14 | 133.921 | 6.8115 | 3.2758 | 8.5331 | 39.8137 | 49.7810 |
| batch | 16 | 133.196 | 6.8115 | 3.4257 | 8.5965 | 39.9900 | 50.0488 |
| tree | 10 | 137.813 | 6.9660 | 2.9438 | 7.9244 | 37.8756 | 49.6048 |
| tree | 12 | 139.264 | 7.0105 | 3.0602 | 7.8947 | 37.6871 | 49.3970 |
| tree | 14 | 134.749 | 6.9252 | 3.3318 | 8.1009 | 38.4205 | 50.4475 |
| tree | 16 | 138.692 | 6.9691 | 3.4043 | 7.9480 | 37.5145 | 49.3153 |

## What Was Run (Matrix Config)

- Dataset: `aime25`
- Samples: `30`
- `max_new_tokens=512`
- Base speculative block size: `16`
- Verify-call semantics: exactly one target verify pass per cycle (`verify_calls_unique=[1]`)
- `batch` mode:
  - candidate verification done in batched-candidate style.
- `tree` mode:
  - tree-chain verify path with one verify call per cycle.
- Candidate counts tested: `max_candidates in {1,2,4,8}`.

## Key Takeaways From Current Matrix

- Best throughput in this matrix is `tree, max_candidates=8` at `138.24 tok/s`.
- `tree` consistently outperforms `batch` at the same candidate cap.
- As candidate cap increases:
  - `mean_tau` improves in both modes.
  - `tree` keeps cycle-time growth much lower than `batch`.
- The throughput gain over `tree, max_candidates=1` is:
  - `+23.22%` at `max_candidates=8` (`138.24` vs `112.19`),
  - with `cycle ms` effectively flat from `max_candidates=4` to `8` (`49.3795` -> `49.3172`).
- From the high-cap rerun (`10..16`):
  - best observed point is `tree, max_candidates=12` at `139.264 tok/s`,
  - pushing beyond ~`12` does not monotonically help (both modes dip at `14`),
  - `tree` remains better than `batch` for the same candidate cap.

## Repro Command Pattern

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate dflash

CUDA_VISIBLE_DEVICES=0 python -u benchmark_candidate_solutions.py \
  --dataset aime25 \
  --max-samples 30 \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --block-size 16 \
  --max-new-tokens 512 \
  --temperature 0.0 \
  --skip-baseline \
  --collect-profile \
  --branch-depth 6 \
  --branch-top-k 2 \
  --max-candidates <1|2|4|8> \
  --branch-margin-threshold 0.10 \
  --candidate-verify-mode <batch|tree_chain> \
  --candidate-verify-static-shape \
  --verify-cache-clone-mode inplace \
  --save-outputs-path outputs/<run_tag>.jsonl \
  --save-cycle-trace-path outputs/<run_tag>_cycle.jsonl | tee logs/<run_tag>.log
```

## Artifact Notes

- Matrix summary source:
  - `outputs/aime25_verifymode_candidates_fix_20260307_074405/summary.md`
- This tracker is the canonical roll-up for candidate-verify optimization status.
- After each new rerun, append:
  - run tag,
  - mode/candidate cap,
  - `tokens/s`,
  - `mean_tau`,
  - `draft ms/cycle`,
  - `verify ms/cycle`,
  - `cycle ms`.
