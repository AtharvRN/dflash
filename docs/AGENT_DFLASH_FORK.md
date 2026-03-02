# DFlash + SGLang Fork Workflow (Current)

## Repository Topology
- Main repo branch: `dflash-flex-experiments`
- SGLang submodule path: `third_party/sglang`
- SGLang submodule remote fork: `AtharvRN/sglang`
- SGLang working branch for DFLASH integration: `pr16818-flex-experiments`

## Pull Everything Correctly
```bash
cd /workspace/DSC291/dflash
git pull origin dflash-flex-experiments
git submodule sync --recursive
git submodule update --init --recursive
pip install -e ./third_party/sglang/python --no-deps
```

Why this matters:
- The benchmark launches `sglang` from the active environment.
- If editable install is stale, new DFLASH flags will not be recognized.

## DFLASH Adaptive Runtime Block Size

### Semantics
- `speculative-dflash-block-size`:
  - Configured max DFLASH block size.
  - Defines memory/capture and `speculative_num_draft_tokens`.
- `speculative-dflash-adaptive-k-min`, `k-max`:
  - Runtime adaptation bounds.
  - Must satisfy `k_max <= speculative-dflash-block-size`.
- `speculative-dflash-adaptive-k-start`:
  - Initial block size per request.
  - Allows `k_start < k_max`.

### Practical Pattern (what we wanted)
- Keep capacity/perf headroom with `block_size=16`.
- Start conservatively with `k_start=8`.
- Allow growth up to `k_max=16` when acceptance supports it.

## Canonical c=16 Command
```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_BASELINE=0 \
RUN_TAG=sg_adaptive_c16_kstart8_$(date +%Y%m%d_%H%M%S) \
CONCURRENCY=16 \
QUESTIONS_PER_CONCURRENCY_BASE=8 \
MAX_QUESTIONS_PER_CONFIG=256 \
DFLASH_BLOCK_SIZE=16 \
ADAPTIVE_ENABLED=1 \
ADAPTIVE_RHO=0.30 \
ADAPTIVE_DELTA=1.0 \
ADAPTIVE_K_MIN=1 \
ADAPTIVE_K_MAX=16 \
ADAPTIVE_K_START=8 \
ADAPTIVE_LOW_ACCEPT_THRESHOLD=0.35 \
ADAPTIVE_LOW_ACCEPT_STREAK=2 \
bash run_sglang_dynamic_c16.sh
```

## Logging + Analysis
- Run script output:
  - `logs/<run_tag>/<run_tag>.md`
  - `logs/<run_tag>/<run_tag>_calls.jsonl`
  - `logs/<run_tag>/<run_tag>_calls_summary.md`
  - `logs/<run_tag>/<run_tag>.log`
- Summary helper:
```bash
python scripts/summarize_sglang_calls.py \
  --input logs/<run_tag>/<run_tag>_calls.jsonl \
  --output-md logs/<run_tag>/<run_tag>_calls_summary.md
```

Important call-trace fields:
- `spec_accept_rate`, `spec_accept_length`, `spec_verify_ct`
- `draft_time_s`, `verify_time_s`
- `draft_time_per_cycle_s`, `verify_time_per_cycle_s`
- `spec_runtime_bs_hist`, `spec_runtime_bs_mode`, `spec_runtime_bs_avg`

## Common Setup Errors
- Unknown adaptive args:
  - `pip install -e ./third_party/sglang/python --no-deps`
- Missing dependency after editable install:
  - `pip install imageio`
- `libnuma.so.1` missing for `sgl_kernel`:
  - install `libnuma1` in pod base image/environment

