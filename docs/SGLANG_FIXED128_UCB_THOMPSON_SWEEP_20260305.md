# SGLang Fixed-128 Adaptive Sweep (UCB + Thompson)

Date: 2026-03-05  
Run prefix: `sg_ucb_thompson_allc_20260305_035841`  
Setup: TP=1, FlashInfer, fixed GSM8K subset (`128` examples, offset `0`), A100.

## Protocol

- Concurrency sweep: `c in {1,4,8,16,32}`
- Adaptive policies: `ucb`, `thompson`
- Common adaptive config:
  - `ADAPTIVE_K_MIN=8`
  - `ADAPTIVE_K_MAX=16`
  - `ADAPTIVE_K_START=12`
  - `ADAPTIVE_BLOCK_BUCKETS=8,12,16`
  - `ADAPTIVE_REWARD_MODE=throughput_proxy`
  - `ADAPTIVE_PROXY_CYCLE_MS="8:1.777,12:1.936,16:2.065"`
- Baseline disabled in these runs (`RUN_BASELINE=0`)
- Cycle trace and stage timing enabled.

## Artifacts

- Pod logs: `logs/sg_ucb_thompson_allc_20260305_035841_*`
- Pulled archive: `pulled/sg_ucb_thompson_allc_20260305_035841.tar.gz`
- Extracted pulled logs: `pulled/sg_ucb_thompson_allc_20260305_035841/logs/sg_ucb_thompson_allc_20260305_035841_*`
- Static comparison reference: `docs/SGLANG_FIXED128_STATIC_SWEEP_20260304.md`

## Results

| conc | algo | toks/s | latency_s | tau | accept_rate | verify/s | draft_tok/s | spec_verify_ct_sum |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ucb | 439.08 | 89.0 | 5.627 | 0.434 | 81.47 | 860.90 | 7254 |
| 1 | thompson | 448.22 | 87.2 | 5.728 | 0.425 | 81.40 | 893.68 | 7100 |
| 4 | ucb | 1120.14 | 34.2 | 5.502 | 0.465 | 210.10 | 2039.48 | 7187 |
| 4 | thompson | 1118.86 | 34.1 | 5.308 | 0.488 | 215.12 | 1920.33 | 7328 |
| 8 | ucb | 1821.85 | 20.9 | 5.463 | 0.464 | 343.27 | 3313.66 | 7161 |
| 8 | thompson | 1785.67 | 21.6 | 5.168 | 0.521 | 352.21 | 2859.37 | 7615 |
| 16 | ucb | 2764.56 | 13.9 | 5.443 | 0.474 | 521.97 | 4921.08 | 7241 |
| 16 | thompson | 2869.95 | 13.2 | 5.056 | 0.540 | 575.83 | 4367.15 | 7629 |
| 32 | ucb | 3867.76 | 9.8 | 5.262 | 0.497 | 748.35 | 6562.99 | 7311 |
| 32 | thompson | 4025.96 | 9.4 | 4.961 | 0.540 | 826.09 | 6068.87 | 7757 |

## Comparison vs Best Static (same fixed-128 subset)

Best static throughputs from `docs/SGLANG_FIXED128_STATIC_SWEEP_20260304.md`:
- `c=1: 507.08` (`bs=16`)
- `c=4: 1289.83` (`bs=16`)
- `c=8: 1980.43` (`bs=16`)
- `c=16: 2940.08` (`bs=16`)
- `c=32: 4134.44` (`bs=8`)

| conc | UCB tok/s | Thompson tok/s | best static tok/s | UCB vs best static | Thompson vs best static |
|---:|---:|---:|---:|---:|---:|
| 1 | 439.08 | 448.22 | 507.08 | -13.41% | -11.61% |
| 4 | 1120.14 | 1118.86 | 1289.83 | -13.16% | -13.25% |
| 8 | 1821.85 | 1785.67 | 1980.43 | -8.01% | -9.83% |
| 16 | 2764.56 | 2869.95 | 2940.08 | -5.97% | -2.39% |
| 32 | 3867.76 | 4025.96 | 4134.44 | -6.45% | -2.62% |

## Policy-to-Policy (Thompson vs UCB)

| conc | Thompson vs UCB tok/s delta |
|---:|---:|
| 1 | +2.08% |
| 4 | -0.11% |
| 8 | -1.99% |
| 16 | +3.81% |
| 32 | +4.09% |

## Observations

- `thompson` is stronger than `ucb` at high concurrency (`c=16,32`), but still below best static.
- In these runs, both adaptive policies remain below best static across all concurrencies.
- At high concurrency, `thompson` is close to static best (within about `2-3%`).
- From cycle traces:
  - `thompson` tends to concentrate on `bs=8` at higher concurrencies.
  - `ucb` keeps a more mixed usage over `8/12/16`.
