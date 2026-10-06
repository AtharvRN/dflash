"""Audit frozen candidate-preserving replay; no serving-speedup extrapolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.summarize_midverify_latency import aggregate


def summarize(source):
    if source['config'].get('suite') != 'predraft_verification_trim':
        raise ValueError('Wrong experiment suite')
    cells = source['results']
    totals = aggregate(cells)
    by_key = {(r['C'], r['mode'], r['case'], r['offset']): r for r in cells}
    for cell in cells:
        base = by_key[(cell['C'], cell['mode'], 'fixed16', cell['offset'])]
        if cell['prompt_ids'] != base['prompt_ids']:
            raise ValueError('Unmatched replay cohort')
        if not cell['case'].endswith('_redraft'):
            if cell['observations'][0]['candidate_blocks'] != base['observations'][0]['candidate_blocks']:
                raise ValueError('A candidate-preserving case changed candidates')
    intervals = {}
    rng = np.random.default_rng(1005)
    for key, value in totals.items():
        c, mode, case = value['C'], value['mode'], value['case']
        prefix = f'c{c}_{mode}_'
        value['aggregate_accept_ratio'] = value['accepted_total'] / (value['states']*(value['mean_end']-1)) if value['mean_end'] > 1 else None
        for ref in source['config']['cases']:
            value['speed_ratio_vs_'+ref] = totals[prefix+ref]['ms_per_committed_token']/value['ms_per_committed_token']
        value['best_measured_fixed_case'] = min(('fixed16', 'fixed8_redraft', 'fixed12_redraft'),
                                               key=lambda ref: totals[prefix+ref]['ms_per_committed_token'])
        value['speed_ratio_vs_best_measured_fixed'] = value['speed_ratio_vs_'+value['best_measured_fixed_case']]
        current_cells = [r for r in cells if r['C']==c and r['mode']==mode and r['case']==case]
        value['prefix_invariance_vs_b16'] = {
            field: sum(r['audit'].get('same_candidates_vs_b16', {}).get(field, 0) for r in current_cells)
            for field in ('kept_top1_differences', 'acceptance_differences', 'bonus_differences')
        } if not case.endswith('_redraft') else None
        if not (case.startswith('predraft_') or case == 'raw_confidence'):
            continue
        select = lambda name: sorted((r for r in cells if r['C']==c and r['mode']==mode and r['case']==name), key=lambda r:r['offset'])
        policy, control, base = select(case), select('predraft_hard'), select('fixed16')
        a, b, a16 = [np.array([x for r in rows for x in r['observations'][0]['accepted']]) for rows in (policy, control, base)]
        idx = rng.integers(len(a), size=(2000, len(a)))
        ci = lambda x: np.quantile(x, [.025, .975]).tolist()
        # Timing-only uncertainty conditional on these exact snapshots and their
        # observed committed counts. It excludes workload/fit/calibration uncertainty.
        def timing_samples(rows):
            summed = np.zeros(2000)
            for r in rows:
                samples = np.asarray(r['uninstrumented_cycle_samples_ms'])
                summed += samples[rng.integers(len(samples), size=(2000, len(samples)))].mean(1)
            return summed / sum(r['observations'][0]['committed_tokens'] for r in rows)
        intervals[key] = {
            'prompt_bootstrap_retention_ci95': ci(a[idx].sum(1)/a16[idx].sum(1)),
            'prompt_bootstrap_retention_delta_vs_hard_ci95': ci((a-b)[idx].sum(1)/a16[idx].sum(1)),
            'conditional_timing_speed_vs_b16_ci95': ci(timing_samples(base)/timing_samples(policy)),
            'scope': '128 distinct prompt snapshots; retention resamples prompts. Speed CI resamples timing repeats only, conditional on this cohort; not training/threshold/workload uncertainty.'}
    return {'metrics': totals, 'uncertainty': intervals,
            'limitations': ['Native replay, not HTTP or closed-loop serving throughput',
                'All candidate-preserving policies pay full B16 drafting; only verification is shortened',
                'No prefill, scheduler, future-trajectory cost, graph-bucket misses, or per-cycle allocation/commit/free integration',
                'Target graph is exact-shape; drafter is eager; native numerical audits reported separately',
                'One frozen seed; already-inspected development assessment; no new final test claim']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    complete = json.loads((args.run/'COMPLETE.json').read_text())
    if not complete.get('passed'):
        raise ValueError('Replay not complete')
    for name, digest in complete['binding'].items():
        if Path(name).name != name or sha256(args.run/name) != digest:
            raise ValueError('Replay binding mismatch')
    output = summarize(json.loads((args.run/'summary.json').read_text()))
    output['source_complete_sha256'] = sha256(args.run/'COMPLETE.json')
    atomic_json(args.run/'analysis.json', output)
    print(json.dumps(output['metrics'], indent=2))


if __name__ == '__main__':
    main()
