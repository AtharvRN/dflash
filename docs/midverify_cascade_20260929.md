# Confidence-first / L6 correction: protocol and completed results

Authorized 2026-09-29 after the controlled 2k/5k/9,999 scaling study. Question:
does intermediate target computation add useful information beyond already
available drafter confidence and candidate identity, after paying for that
computation? No new collection, model fine-tuning, SGLang integration or monitor.

## Frozen inputs and scope

Use `midverify_scaling_10k_20260929/cache_nonterminal`: 9,999 train cycles,
342 calibration and 1,416 assessment cycles. Preserve prompt membership, saved
B16 candidates, all features and same-forward greedy labels. Verify completion
hashes and the previous scaling result binding before any fitting.

Only L6 is used. Full B16 drafting cost is unchanged. First-stage thresholding
of saved replayed candidate logprob chooses K0 query rows including the anchor.
Second-stage scoring chooses K1 <= K0. Work proxy is
`[6*K0 + 30*K1]/36`, not `[6*16 + 30*K1]/36`. This is not measured latency;
predictor/packing/graph overhead and nonlinear attention costs are unmeasured.

Cached L6 vectors come from full-width verification. Causal prefix invariance
justifies their use for this offline diagnostic in exact arithmetic, but does
not establish numerical parity of actual shortened or segmented GPU forwards.
No deployed correctness or stochastic-sampling guarantee is claimed.

## Three learned feature ablations, three seeds each

All heads have one 128-wide GELU hidden layer, dropout .05, and one binary output
shared across candidate positions. Inputs:

1. `confidence_only`: three draft statistics: candidate logprob, entropy and
   candidate-minus-top logit. All available before target verification.
2. `candidate_confidence`: RMS-normalized frozen candidate LM-head vector
   (2,560 dimensions) plus the same three statistics; also target-free.
3. `target_candidate_confidence`: `[RMS(h_L6), RMS(e), RMS(h_L6)*RMS(e)]`
   (7,680 dimensions) plus the same statistics.

Confidence statistics use means/stds fitted only on training at-risk positions.
Candidate and hidden normalization have no learned parameters. Hidden width is
matched; input dimensionality and parameter counts are NOT matched. The
candidate-only control prevents attributing token-identity gains to L6 work.

Train nine independent trajectories: seeds913/914/915, all 9,999 states,
AdamW3e-4, weight decay.01, clip1, FP32/TF32 off, full128-state shuffled batches,
1,024 updates each. Same first-rejection masked BCE, no auxiliary objective.
Assess calibration every64 updates. No learning-rate or architecture search.

## Joint calibration and selection

Primary operating points: 96% and 99% calibration accepted-token retention.
For each checkpoint, search all distinct prefix-score threshold transitions
jointly, minimizing total row-layer work subject to the aggregate retention
constraint. First threshold uses draft candidate logprob; the second uses the
learned score. Include keep-all and explicit second-stage bypass. Integer
lengths 1–16 remain available, not a fixed four-arm grid.

For target-free controls, BOTH thresholds act before the target, so charge
`K1`, not six layers on K0. For target-feature models, charge the actual cascade
proxy above. Ties prefer greater accepted-token total, then bypass/fewer front
rows and fixed transition ordering. Select checkpoints independently per
operating point using calibration work, then retention, then earlier update.
Assessment never selects checkpoints, thresholds, seeds or feature variants.

Save both chosen checkpoints per trajectory (18 policies), calibration history,
normalization, score arrays, settings and exact-reload checks. Calibration and
assessment scoring use separately fixed batching, preventing a shape change
from altering the scores used for calibration versus assessment application.

## Controls and reporting

- Frozen raw draft-confidence policy from the preceding study, exactly reproduced.
- Frozen L6 target-only MLP from that study, with newly jointly calibrated cascade
  thresholds, no weight changes (three seeds). This isolates staging from fusion.
- Both newly learned target-free controls, allowed the same two-threshold search.
- A clairvoyant second-stage oracle following frozen confidence, preserving its
  exact accepted tokens; diagnostic bound only, never a trainable policy.

For each seed/operating point, choose the strongest target-free family using
calibration only (raw confidence, confidence-only head, candidate-confidence
head). Report every family separately as well as this reference choice.
Report actual assessment retention, K0/K1, work/committed token, held-out risk
BCE, and paired whole-prompt bootstrap intervals (2,000 resamples) versus raw
confidence and versus the calibration-chosen target-free reference.

These are calibration-matched comparisons; assessment retentions can differ.
Do not claim a win from smaller work alone when retention is also lower.
Bootstrap intervals condition on fitted policies and do not cover calibration
selection or training uncertainty. This repeatedly inspected assessment remains
development data, not an untouched paper test. No automatic integration follows.

## Execution

CPU unit tests and a bounded three-model CPU smoke precede the main GPU run.
Use only the idle authorized GPU4, checking immediately before allocation.
Fresh durable run destination, max runtime 1,800 seconds, no overwrite or monitor.
Models/data remain under `/data/scratch/zekaili/atharv/dflash`; deploy via git.

## Completed execution (2026-09-29)

Implementation/launch commit: `db47d71`. Forty focused tests passed, including
exact joint calibration versus brute force, float32 threshold transitions,
prefix availability/no resurrection, cost accounting and target-free feature
independence. The three-model CPU smoke passed six checkpoint reloads.

The initial SSH launch connection reset. Read-only inspection established that
no job or run directory had started and GPU4 was idle before retry. Successful
main launch parent PID 3422004; all nine models / 18 selected checkpoints completed
with exit code 0. Training/scoring/reporting took 19.14 seconds, excluding initial source
hash verification. No new data or target/drafter forward passes were collected.

All 18 checkpoint reloads reproduce calibration and assessment scores exactly.
All 26 completion-bound files (including checkpoints) passed a fresh remote
checksum audit. The eight downloaded bound report/score/plot/metadata files
passed local verification. Raw confidence scores and both frozen operating
points reproduce the preceding run exactly. GPU4 was verified at 0 MiB / 0% after
completion. No integration, additional sweep or recurring monitor was started.

Models actually fitted: confidence-only 3-128-1 (641 parameters),
candidate-confidence 2563-128-1 (328,321 parameters), and
target-candidate-confidence 7683-128-1 (983,681 parameters), three seeds each.
All use GELU/dropout 0.05 and the preregistered masked BCE/update budget.

## Primary assessment results

Arithmetic means across three separately fitted seeds, not an ensemble or
best-seed selection. Raw confidence is one frozen policy. Retention is measured
on assessment; it is not forced to equal the calibration constraint. Work is
full-depth-equivalent target query rows, not measured latency or throughput.

| Calibration target | Policy | Assessment retention | Front rows K0 | Final rows K1 | Work proxy | Work/committed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 96% | Raw confidence | 95.248% | 9.049 | 9.049 | 9.049 | 1.2446 |
| 96% | Learned confidence | 94.805% | 9.107 | 8.896 | 8.896 | 1.2284 |
| 96% | Candidate + confidence | 95.220% | 9.049 | 9.027 | 9.027 | 1.2418 |
| 96% | Confidence → frozen target-only L6 | 94.966% | 9.202 | 8.812 | 8.877 | 1.2240 |
| 96% | Confidence → target/candidate/confidence L6 | 95.012% | 9.069 | 8.730 | 8.787 | 1.2110 |
| 99% | Raw confidence | 98.809% | 11.030 | 11.030 | 11.030 | 1.4696 |
| 99% | Learned confidence | 98.649% | 11.208 | 10.866 | 10.866 | 1.4497 |
| 99% | Candidate + confidence | 98.395% | 11.030 | 10.666 | 10.666 | 1.4262 |
| 99% | Confidence → frozen target-only L6 | 98.495% | 11.119 | 10.079 | 10.252 | 1.3697 |
| 99% | Confidence → target/candidate/confidence L6 | 98.209% | 11.119 | 9.775 | 9.999 | 1.3393 |

"Fused" in the generated plot means the new fusion of L6 target features,
candidate vector and confidence statistics, NOT the earlier DFlash fused-context
vector. No DFlash fused-context input was used in this experiment.

### Compare against the stronger target-free alternatives

At 96%, calibration chooses learned confidence for seeds 913/914 and
candidate-confidence for 915. Their mean assessment is 94.805% retention and
8.895 work rows. The L6 cascade gives 95.012% and 8.787: about 1.22% less work,
with 0.207 percentage points higher mean retention. This is a small effect:
paired work-delta 95% intervals cross zero for seeds 913/914; seed 915 has a clear
work reduction but lower retention than its own selected reference. Do not
describe this as a robust matched-retention win across seeds.

At 99%, calibration chooses candidate-confidence for every seed. Its mean is
98.395% retention and 10.666 work rows. The L6 cascade gives 98.209% and 9.999:
about 6.25% less work, with 0.186 percentage points lower retention. Work per
committed token falls from 1.4262 to 1.3393 (about 6.10%). Both remain proxies.

Paired whole-prompt bootstrap 95% intervals for the relative work saving versus
the corresponding candidate-confidence reference are:

- Seed 913: 5.50–7.74%.
- Seed 914: 5.17–7.32%.
- Seed 915: 4.96–7.00%.

Each retention-delta interval spans zero, but the lower endpoints allow losses
of about 0.55–0.83 percentage points. Thus this does NOT establish retention
equivalence or noninferiority. The work intervals condition on the fitted
policies and do not capture calibration/model-selection uncertainty.

Against raw confidence alone, cascade work savings are 2.90% at the 96%-calibrated
point and 9.35% at the 99%-calibrated point, with retention lower by 0.236pp and
0.601pp respectively. Reporting only those savings would overstate the added
value of target computation relative to the stronger learned target-free heads.

### Optimization and diagnostic findings

The L6 fusion checkpoints selected at 96% are updates 320/704/320; at 99%,
320/192/320. Their calibration BCE rises with prolonged training even as train
BCE falls. The tiny confidence-only model selects updates 1024/1024/1024 at 96%
and 1024/960/960 at 99%, so this remains a fixed-budget comparison, not proof that
every architecture is converged or that all target-free possibilities are exhausted.

Assessment masked BCE at the 99%-selected checkpoints averages 0.2050 for
confidence-only, 0.2442 for candidate-confidence, and 0.2409 for L6 fusion. The
policy/work advantage does not imply universally better probability prediction.
Hidden width is matched, not parameter count, so this is not a complete
capacity-controlled information-theoretic test.

An oracle second-stage boundary after the frozen raw-confidence prefix would
use 7.568 work rows at exactly 95.248% retention, or 8.093 rows at exactly 98.809%.
These bounds preserve the reference accepted tokens and include front-layer
work, but assume clairvoyance and zero probe/compaction overhead. They are not
achieved by the learned cascade and not bounds on measured latency.

## Interpretation / handoff

The cascade is more promising than paying for full-width L6 processing before
any trimming. These results suggest complementary target-side signal at the
higher-retention operating point, but its incremental estimated-work benefit
over learned target-free controls is modest, with a retention tradeoff.

This test does not yet establish a same-retention gain or an online speedup,
and does not justify a large integration or 100k collection automatically. If
continued, the next gate is retention-controlled confirmation against the
strongest target-free controls (including fresh prompt-level evaluation), then
a bounded overhead/segmented-forward feasibility check—not another architecture
or data sweep by default. No such follow-up has been launched.

## Full paths

Remote:
`/data/scratch/zekaili/atharv/dflash/runs/midverify_cascade_20260929`

- `training/`: config, normalization, all 18 checkpoints, frozen scores,
  calibration histories, paired intervals, completion hashes and plots.
- `training.log`, `exit.txt`: execution evidence.
- CPU smoke: `/data/scratch/zekaili/atharv/dflash/runs/midverify_cascade_smoke_20260929`.

Local reports:

- [Policy comparison plot](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_cascade_20260929/training/cascade_comparison.png)
- [Learning curves](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_cascade_20260929/training/learning_curves.png)
- [All metrics, thresholds and paired intervals](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_cascade_20260929/training/summary.json)
- [Generated report](/Users/atharvramesh/Projects/MLSys/dflash-headroom/outputs/midverify_cascade_20260929/training/report.md)
