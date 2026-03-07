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
