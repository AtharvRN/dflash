# SGLang Undocumented Results (Backfill)

Generated on: 2026-03-06

This file backfills run artifacts present under `logs/` that were not explicitly referenced in `docs/*.md` or `results.md` at generation time.

- Undocumented run tags documented here: `19`
- Companion CSV: `docs/SGLANG_UNDOCUMENTED_RESULTS_20260306.csv`

## Runs

| run_tag | dataset | algo | reward | conc | n | baseline toks/s | dflash toks/s | speedup vs baseline | tau | accept_rate | verify/s | draft_tok/s | draft ms/cycle | verify ms/cycle | mode bs | top bs hist |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prefix_len_effect_20260305_summary | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sg_dyn_single_20260302_185139 | gsm8k | - | - | 16 | 128 | 810.71 | 1675.06 | 2.066 | 6.354 | 0.355 | 273.93 | 4108.97 | - | - | 16 | - |
| sg_dynamic_c16_20260302_055145 | gsm8k | - | - | 16 | 256 | - | 2013.78 | - | 6.030 | 0.410 | 348.20 | 4374.12 | 0.229 | 1.296 | 16 | 16:69.5%, 8:30.5% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc025_texp01 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2692.14 | - | 5.137 | 0.519 | 536.36 | 4273.56 | 0.200 | 1.087 | 8 | 8:79.6%, 12:16.7%, 16:3.8% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc025_texp02 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2677.85 | - | 5.137 | 0.519 | 533.52 | 4250.89 | 0.201 | 1.088 | 8 | 8:79.6%, 12:16.7%, 16:3.8% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc025_texp03 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2671.87 | - | 5.137 | 0.519 | 532.32 | 4241.39 | 0.203 | 1.091 | 8 | 8:79.6%, 12:16.7%, 16:3.8% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc025_texp04 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2677.61 | - | 5.137 | 0.519 | 533.47 | 4250.50 | 0.202 | 1.090 | 8 | 8:79.6%, 12:16.7%, 16:3.8% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc05_texp01 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2710.85 | - | 5.296 | 0.482 | 525.16 | 4668.80 | 0.205 | 1.118 | 8 | 8:62.5%, 12:27.8%, 16:9.7% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc05_texp02 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2650.61 | - | 5.347 | 0.478 | 508.98 | 4612.11 | 0.212 | 1.160 | 8 | 8:58.5%, 12:31.5%, 16:10.0% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc05_texp03 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2638.62 | - | 5.332 | 0.483 | 508.27 | 4521.58 | 0.213 | 1.153 | 8 | 8:62.6%, 12:27.3%, 16:10.1% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc05_texp04 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2649.27 | - | 5.347 | 0.478 | 508.72 | 4609.77 | 0.213 | 1.160 | 8 | 8:58.5%, 12:31.5%, 16:10.0% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc10_texp01 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2822.86 | - | 5.464 | 0.457 | 530.36 | 5155.63 | 0.202 | 1.114 | 8 | 8:49.9%, 12:32.2%, 16:17.9% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc10_texp02 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2664.96 | - | 5.524 | 0.456 | 495.58 | 4897.70 | 0.217 | 1.197 | 8 | 8:46.9%, 12:34.2%, 16:18.9% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc10_texp03 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2670.88 | - | 5.522 | 0.456 | 496.62 | 4899.94 | 0.218 | 1.194 | 8 | 8:47.0%, 12:34.4%, 16:18.6% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc10_texp04 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2650.73 | - | 5.524 | 0.456 | 492.93 | 4871.55 | 0.219 | 1.200 | 8 | 8:46.9%, 12:34.2%, 16:18.9% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc15_texp01 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2523.74 | - | 5.597 | 0.451 | 461.42 | 4691.88 | 0.236 | 1.297 | 8 | 8:44.1%, 12:32.7%, 16:23.3% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc15_texp02 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2534.50 | - | 5.603 | 0.448 | 461.98 | 4725.46 | 0.235 | 1.298 | 8 | 8:43.2%, 12:32.8%, 16:23.9% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc15_texp03 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2525.42 | - | 5.603 | 0.448 | 460.33 | 4708.53 | 0.237 | 1.300 | 8 | 8:43.2%, 12:32.8%, 16:23.9% |
| sg_gsm8k_c16_ucb_powerlaw_grid_20260306_003824_ucbc15_texp04 | gsm8k | ucb | throughput_proxy | 16 | 128 | - | 2517.85 | - | 5.603 | 0.448 | 458.95 | 4694.42 | 0.238 | 1.302 | 8 | 8:43.2%, 12:32.8%, 16:23.9% |

## Notes

- Many entries are single-policy runs (no baseline in same log), so speedup is blank by design.
- `mode bs` and cycle-time columns come from call-trace summaries when available.
- Some top-level artifacts (for example `prefix_len_effect_20260305_summary`) are analysis summaries, not benchmark run logs.
