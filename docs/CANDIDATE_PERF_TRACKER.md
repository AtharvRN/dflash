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
