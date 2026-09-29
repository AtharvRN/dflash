"""Audit and aggregate same-state latency diagnostics; never call this serving TPS."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json


def aggregate(cells):
    groups = defaultdict(list)
    for row in cells:
        groups[(row['C'], row['mode'], row['case'])].append(row)
    result = {}
    for (c, mode, case), rows in groups.items():
        n = sum(r['rows'] for r in rows)
        committed = sum(r['observations'][0]['committed_tokens'] for r in rows)
        elapsed = sum(r['uninstrumented_cycle_ms']['mean'] for r in rows)
        phases = sorted(set(k for r in rows for k in r['phase_stream_ms']))
        key = f'c{c}_{mode}_{case}'
        result[key] = {'C': c, 'mode': mode, 'case': case, 'states': n,
            'cycle_ms': elapsed/len(rows), 'committed_tokens_per_cycle': committed/len(rows),
            'ms_per_committed_token': elapsed/committed,
            'replay_tokens_per_second': 1000*committed/elapsed,
            'accepted_total': committed-n, 'committed_total': committed,
            'mean_front': sum(r['mean_front']*r['rows'] for r in rows)/n,
            'mean_end': sum(r['mean_end']*r['rows'] for r in rows)/n,
            'cycle_repeat_stdev_ms_by_batch': [r['uninstrumented_cycle_ms']['stdev'] for r in rows],
            'phase_stream_ms': {p: sum(r['phase_stream_ms'].get(p, {}).get('mean', 0) for r in rows)/len(rows) for p in phases},
            'audit_top1_disagreements': sum(r['audit']['top1_differences'] for r in rows),
            'audit_acceptance_disagreements': sum(r['audit']['acceptance_differences'] for r in rows),
            'audit_bonus_disagreements': sum(r['audit']['bonus_differences'] for r in rows),
            'anchor_disagreements': sum(r['anchor_mismatches'] for r in rows),
            'max_hidden_relative_l2': max(r['audit']['hidden']['relative_l2'] for r in rows)}
    for value in result.values():
        prefix = f"c{value['C']}_{value['mode']}_"
        for ref in ('fixed16', 'fixed16_redraft', 'fixed8_redraft', 'target_free'):
            if prefix+ref not in result:
                continue
            baseline = result[prefix+ref]
            value['speed_ratio_vs_'+ref] = baseline['ms_per_committed_token']/value['ms_per_committed_token']
        value['retention_vs_same_engine_b16'] = value['accepted_total']/result[prefix+'fixed16']['accepted_total']
    return result


def frozen_policy_audit(source, states):
    import numpy as np
    from dflash.midverify_cascade import apply_cascade
    from scripts.train_midverify_scaling import load_verified

    cache, policies = Path(source['config']['cache']), Path(source['config']['policies'])
    load_verified(cache)
    load_verified(policies)
    saved = json.loads((policies/'summary.json').read_text())
    erows = json.loads((policies/'evaluation_rows.json').read_text())
    lookup = {(str(r['prompt_id']), int(r['cycle'])): i for i, r in enumerate(erows)}
    eval_idx = [lookup[(str(r['prompt_id']), int(r['cycle']))] for r in states]
    data_idx = [r['row'] for r in states]
    confidence = np.load(cache/'draft_stats.npy', mmap_mode='r')[data_idx, :, 0]
    a = np.load(cache/'accepted_len.npy', mmap_mode='r')[data_idx]
    out = {'states': len(states), 'cached_b16_accepted_total': int(a.sum()),
           'note': 'Same snapshot cohort; frozen checkpoints/thresholds, no fitting. Runtime uses freshly computed engine confidence and target features.'}
    with np.load(policies/'scores.npz', allow_pickle=False) as scores:
        for case, key in [('target_free', 'candidate_confidence_seed913_r0.99'),
                          ('cascade', 'target_candidate_confidence_seed913_r0.99')]:
            front, end = apply_cascade(confidence, scores[key][eval_idx], saved['policies'][key]['setting'])
            expected_front = end if case == 'target_free' else front
            item = {'cached_mean_stage0': float(front.mean()),
                    'cached_mean_front': float(expected_front.mean()), 'cached_mean_end': float(end.mean()),
                    'cached_retention': float(np.minimum(a, end-1).sum()/a.sum()), 'native': {}}
            for c in source['config']['concurrencies']:
                for mode in source['config']['modes']:
                    cells = sorted([r for r in source['results'] if r['C'] == c and r['mode'] == mode and r['case'] == case], key=lambda r:r['offset'])
                    nfront = np.array([n for r in cells for n in r['observations'][0]['front']])
                    nend = np.array([n for r in cells for n in r['observations'][0]['end']])
                    item['native'][f'c{c}_{mode}'] = {'front_disagreement_states': int((expected_front != nfront).sum()),
                        'end_disagreement_states': int((end != nend).sum()),
                        'mean_front': float(nfront.mean()), 'mean_end': float(nend.mean())}
            out[case] = item
    return out


def conditional_intervals(cells, draws=4000):
    import numpy as np
    result = {}
    for c, mode in sorted({(r['C'], r['mode']) for r in cells}):
        select = lambda case: sorted([r for r in cells if r['C'] == c and r['mode'] == mode and r['case'] == case], key=lambda r:r['offset'])
        policy, ref, base = select('cascade'), select('target_free'), select('fixed16')
        a = np.array([x for r in policy for x in r['observations'][0]['accepted']])
        b = np.array([x for r in ref for x in r['observations'][0]['accepted']])
        b16 = np.array([x for r in base for x in r['observations'][0]['accepted']])
        rng = np.random.default_rng(929)
        idx = rng.integers(len(a), size=(draws, len(a)))
        out = {'retention_delta_ci95': np.quantile((a-b)[idx].sum(1)/b16[idx].sum(1), [.025, .975]).tolist(),
               'retention_scope': 'Paired resampling of 128 prompt snapshots, conditional on frozen fitted policies; no calibration uncertainty.'}
        if all('uninstrumented_cycle_samples_ms' in r for r in policy+ref):
            ta, tb = np.zeros(draws), np.zeros(draws)
            for x, y in zip(policy, ref):
                ax, by = np.array(x['uninstrumented_cycle_samples_ms']), np.array(y['uninstrumented_cycle_samples_ms'])
                if ax.shape != by.shape:
                    raise ValueError('Unmatched timing repeats')
                take = rng.integers(len(ax), size=(draws, len(ax)))
                ta += ax[take].mean(1)
                tb += by[take].mean(1)
            ratios = tb/ta * (a.sum()+len(a))/(b.sum()+len(b))
            out['replay_speed_ratio_vs_target_free_ci95'] = np.quantile(ratios, [.025, .975]).tolist()
            out['timing_scope'] = 'Paired resampling of repeated timing rounds within each fixed batch; conditional on these snapshots and their observed token counts. Not workload/model/retention uncertainty.'
        result[f'c{c}_{mode}'] = out
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh report directory')
    complete = json.loads((a.run/'COMPLETE.json').read_text())
    if not complete['passed']:
        raise ValueError('Incomplete run')
    for name, expected in complete['binding'].items():
        if Path(name).name != name or sha256(a.run/name) != expected:
            raise ValueError('Source binding mismatch')
    source = json.loads((a.run/'summary.json').read_text())
    result = aggregate(source['results'])
    a.output.mkdir(parents=True)
    report = {'source_completion_sha256': sha256(a.run/'COMPLETE.json'),
        'scope': source['interpretation'], 'aggregation': 'Sum matched batch times / sum actually committed tokens; not mean per-cycle ratios. Repeat SD is not workload uncertainty.',
        'source_commit': source['config']['commit'], 'cells': result,
        'uncertainty': conditional_intervals(source['results'])}
    atomic_json(a.output/'summary.json', report)
    atomic_json(a.output/'frozen_vs_native.json', frozen_policy_audit(source, json.loads((a.run/'states.json').read_text())))
    lines = ['# Native-engine mid-verification latency replay', '', report['scope'], '',
        '| C | Mode | Policy | Cycle ms | Retention | Final rows | ms/committed token | Relative to target-free |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result.values():
        lines.append(f"| {r['C']} | {r['mode']} | {r['case']} | {r['cycle_ms']:.3f} | {r['retention_vs_same_engine_b16']:.5f} | {r['mean_end']:.3f} | {r['ms_per_committed_token']:.5f} | {r['speed_ratio_vs_target_free']:.4f}× |")
    lines += ['', 'Retentions are observed on the selected snapshot cohort, not retuned. ',
        'Numerical differences and full component costs are retained in summary.json.',
        'Fixed8_redraft actually changes draft width/candidates; fixed8_same_candidates only truncates B16.',
        'The explicit layer split and graph captures are a replay harness, not deployed dynamic-serving integration.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = source['config']['cases']
    labels = {'fixed16': 'Saved B16', 'fixed16_redraft': 'Actual B16', 'fixed8_same_candidates': 'B16 → B8 trim', 'fixed8_redraft': 'Actual B8',
              'target_free': 'Target-free', 'target_free_split': 'Split, no pruning', 'cascade': 'L6 cascade'}
    colors = ['#64748b', '#758bab', '#94a3b8', '#b0c4d8', '#2a9d8f', '#e9c46a', '#e76f51']
    for ax, c in zip(axes, source['config']['concurrencies']):
        values = [result[f'c{c}_graph_{n}'] for n in names]
        ax.bar(range(len(names)), [r['ms_per_committed_token'] for r in values], color=colors)
        ax.set_xticks(range(len(names)), [labels[n] for n in names], rotation=30, ha='right')
        ax.set_title(f'C{c}: same saved states, target graphs')
        ax.set_ylabel('Replay ms / committed token (lower is better)')
        ax.grid(axis='y', alpha=.2)
        for i, r in enumerate(values):
            ax.text(i, r['ms_per_committed_token'], f"R={100*r['retention_vs_same_engine_b16']:.1f}%", ha='center', va='bottom', fontsize=8)
    fig.suptitle('Measured replay cost, NOT end-to-end serving speedup; retentions differ')
    fig.tight_layout()
    fig.savefig(a.output/'latency_comparison.png', dpi=180)
    plt.close(fig)
    atomic_json(a.output/'COMPLETE.json', {'passed': True, 'binding': {name: sha256(a.output/name)
        for name in ('summary.json', 'report.md', 'latency_comparison.png', 'frozen_vs_native.json')}})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
