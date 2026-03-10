# SGLang FlashInfer MC4 Review

Context:
- branch: `dflash-flex-experiments`
- submodule branch: `pr16818-flex-experiments`
- setup under review: DFLASH `sample_multi` with `--speculative-dflash-multi-candidate-max-candidates 4`
- verify mode: `packed_tree`
- attention backend: `flashinfer`

## Findings

1. FlashInfer skips the DFLASH custom tree mask.

- `resolve_dflash_verify_mask_policy()` disables custom-mask building for `FlashInferAttnBackend` and `FlashInferMLAAttnBackend` in [dflash_utils.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_utils.py#L17) and [dflash_utils.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_utils.py#L92).
- The packed mc4 prepare path respects that policy and calls `prepare_for_verify(..., build_custom_mask=build_custom_mask)` in [dflash_worker.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_worker.py#L2788).
- The actual per-candidate isolation mask is only built inside [dflash_info.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_info.py#L241) and specifically for the multi-candidate case at [dflash_info.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_info.py#L256).
- FlashInfer only uses the custom mask when `spec_info.custom_mask` is non-null at [flashinfer_backend.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py#L620) and [flashinfer_backend.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py#L1523).

Risk:
- For FlashInfer mc4, I do not see an alternative DFLASH-specific packed-tree path that encodes candidate isolation from `num_candidates` or `candidate_block_size`.
- That makes the current FlashInfer mc4 path a correctness risk, not only a performance problem.

2. MC4 verify does a full packed target forward before choosing a winner.

- `DFlashVerifyInput.tokens_per_req` is `candidate_block_size * num_candidates` in [dflash_info.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_info.py#L178).
- The worker sets `num_candidates=4` and `candidate_block_size=runtime_block_size` in [dflash_worker.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_worker.py#L2780).
- The verifier reshapes target logits and hidden states to `[bs, num_candidates, block_len, ...]` and only picks the best candidate after the full packed verify at [dflash_info.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_info.py#L355) and [dflash_info.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_info.py#L373).

Consequence:
- With block size `16` and `mc=4`, the target verify width is `64` tokens per request.
- This explains the verify-time blowup even after CUDA graph replay was fixed.

3. The old cycle trace did not expose enough mc4 verify-shape metadata.

- The saved cycle trace only recorded a minimal summary at [dflash_worker.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_worker.py#L3871).
- That was not enough to read off effective packed verify width directly from artifacts.

## Instrumentation Added

To make future runs diagnosable from artifacts alone, I added:
- `candidate_block_size`
- `effective_verify_tokens_per_req`
- `verify_mask_backend`
- `build_custom_mask`

These are now included in:
- top-level per-request `dflash_multi_candidate_last_decision`
- per-cycle `spec_cycle_trace[*].multi_candidate_decision`

Patch location:
- [dflash_worker.py](/Users/atharvramesh/Projects/dflash/third_party/sglang/python/sglang/srt/speculative/dflash_worker.py)

## Bottom Line

- The mc4 slowdown is primarily caused by target-side work inflation.
- There is also a likely FlashInfer semantics hole because the DFLASH multi-candidate tree mask is skipped for FlashInfer without an obvious replacement path.
