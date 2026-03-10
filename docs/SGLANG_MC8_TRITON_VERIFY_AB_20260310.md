# SGLang mc8 Verify Backend A/B (2026-03-10)

## Goal

Validate the `packed_tree` multi-candidate (`mc=8`) verify backend A/B after fixing CUDA graph capture for verify backend overrides.

## Code Fix Applied

- Submodule commit (sglang): `8d5a53bec`
- Superproject commit: `d0babf6`
- File changed:
  - `third_party/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`

What was fixed:

1. Ensure CUDA graph state is initialized for any attention backend selected at runtime (including verify backend overrides like `triton`).
2. Use the same backend selector in replay path (`get_attention_backend_for_forward(...)`) instead of always replaying with `model_runner.attn_backend`.
3. Resolve DFLASH verify mask policy against the actual capture-time verify backend (not always the default backend).

## Pod Runs

Pod: `atharv-rwx-pod`  
Env: `conda run -n dflash`  
Model pair: `Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash-b16`  
Config: `DFLASH`, `block_size=16`, `multi-candidate max=8`, `deterministic_prefix_len=2`, `verify_mode=packed_tree`, `c=1`

### Smoke (n=8)

- Triton verify override:
  - run tag: `sg_mc8_triton_smoke_20260310_222549`
  - includes: `--speculative-dflash-multi-candidate-verify-attention-backend triton`
- Default verify backend (no override):
  - run tag: `sg_mc8_flashinfer_smoke_20260310_222729`

### Fixed-seed A/B (n=30, seed=123)

- Triton verify override:
  - run tag: `sg_mc8_triton_q30_20260310_223002`
- Default verify backend (no override):
  - run tag: `sg_mc8_flashinfer_q30_20260310_223141`

## Results

| run | n | tok/s | tau | accept_rate | verify_time/cycle (ms) | draft_time/cycle (ms) | verify_time/call (ms) | draft_time/call (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| triton_q8 | 8 | 524.813 | 10.2774 | 0.6158 | 10.5136 | 1.4615 | 287.8478 | 39.7964 |
| flashinfer_q8 | 8 | 421.591 | 7.8466 | 0.4544 | 10.6279 | 1.4561 | 365.9251 | 50.0596 |
| triton_q30 | 30 | 441.821 | 9.0157 | 0.5320 | 10.6059 | 1.4614 | 347.4036 | 48.1376 |
| flashinfer_q30 | 30 | 401.389 | 7.5481 | 0.4346 | 10.6299 | 1.4536 | 386.9496 | 52.7634 |

## Key Takeaways

1. The original crash is resolved: triton verify override now captures and serves with CUDA graphs.
2. `verify_time_per_cycle` is almost unchanged across the two backends (~10.6 ms).
3. Throughput delta mostly tracks `tau` / accept-rate changes, not per-cycle verify latency change.
4. This indicates remaining backend difference is primarily acceptance behavior (numerics/semantics), not a large raw kernel-time reduction in verify per cycle.

