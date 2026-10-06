"""CPU-only frozen-policy audit: cohort shift versus native replay drift.

Assessment outcomes are used only to describe existing decisions. No threshold,
checkpoint, or deployment policy is fitted from assessment rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import apply_setting, calibrate, metrics
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.export_predraft_latency_bundle import verify


def align_rows(states, predictions):
    ids = predictions['row_indices'].astype(int)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Duplicate prediction row identity')
    lookup = {int(row): i for i, row in enumerate(ids)}
    if len({r['row'] for r in states}) != len(states):
        raise ValueError('Duplicate replay state')
    indices = []
    for row in states:
        if row['group'] != 'assessment' or row['row'] not in lookup:
            raise ValueError('Replay state outside frozen assessment')
        i = lookup[row['row']]
        if str(row['prompt_id']) != str(predictions['prompt_ids'][i]):
            raise ValueError('Prompt identity mismatch')
        if row['cached_accepted_len'] != predictions['accepted'][i]:
            raise ValueError('Cached acceptance mismatch')
        indices.append(i)
    return np.asarray(indices, dtype=int)


def describe(accepted, kept, prompts):
    a = np.asarray(accepted)
    if len(a) == 0:
        return {'rows': 0}
    return {**metrics(a, kept), 'prompts': len(set(map(str, prompts))),
            'b16_mean_accepted': float(a.mean()), 'b16_accepted_total': int(a.sum()),
            'b16_histogram': np.bincount(a.astype(int), minlength=16).tolist()}


def replay_audit(run, states, saved, selected, policy_settings):
    complete = json.loads((run/'COMPLETE.json').read_text())
    if not complete.get('passed'):
        raise ValueError('Incomplete native replay')
    for name, digest in complete['binding'].items():
        if Path(name).name != name or sha256(run/name) != digest:
            raise ValueError('Replay binding mismatch')
    if json.loads((run/'states.json').read_text()) != states:
        raise ValueError('Replay cohort differs from frozen bundle')
    source = json.loads((run/'summary.json').read_text())
    if source['config'].get('suite') != 'predraft_verification_trim':
        raise ValueError('Unexpected replay suite')
    if source['config'].get('frozen_policies') != policy_settings:
        raise ValueError('Replay used different frozen policies')
    results = source['results']
    cached_a = saved['accepted'][selected]
    out = {}
    for c in source['config']['concurrencies']:
        audits = []
        for offset in range(0, len(states), c):
            name = f'predraft_feature_audit_c{c}_offset{offset}.json'
            if name not in complete['binding']:
                raise ValueError('Unbound feature audit')
            audits.append(json.loads((run/name).read_text()))
        for mode in source['config']['modes']:
            def cells(case):
                rows = sorted((r for r in results if r['C'] == c and r['mode'] == mode
                               and r['case'] == case), key=lambda r:r['offset'])
                if [r['offset'] for r in rows] != list(range(0, len(states), c)):
                    raise ValueError('Missing/duplicate replay batches')
                if list(map(str, [p for r in rows for p in r['prompt_ids']])) != list(map(str, [r['prompt_id'] for r in states])):
                    raise ValueError('Native row order mismatch')
                return rows
            base = cells('fixed16')
            native_a = np.array([a for r in base for a in r['observations'][0]['accepted']])
            item = {'native_b16_mean_accepted': float(native_a.mean()),
                    'cached_b16_mean_accepted': float(cached_a.mean()),
                    'b16_acceptance_disagreement_states': int((native_a != cached_a).sum()),
                    'candidate_token_differences': sum(a['cached_vs_native_candidate_differences'] for a in audits),
                    'policies': {}}
            for case, setting in policy_settings.items():
                if not case.startswith('predraft_'):
                    continue
                key = Path(setting['checkpoint']).stem
                cached_k = apply_setting(saved[key][selected], setting)
                feature_k = np.array([k for a in audits for k in a['policies'][case]['native_lengths']])
                replay_cached_k = np.array([k for a in audits for k in a['policies'][case]['cached_lengths']])
                rows = cells(case)
                actual_k = np.array([k for r in rows for k in r['observations'][0]['end']])
                actual_a = np.array([a for r in rows for a in r['observations'][0]['accepted']])
                if not np.array_equal(feature_k, actual_k):
                    raise ValueError('Feature audit and measured policy disagree')
                for r, b in zip(rows, base, strict=True):
                    if r['observations'][0]['candidate_blocks'] != b['observations'][0]['candidate_blocks']:
                        raise ValueError('Policy changed candidates')
                expected_a = np.minimum(native_a, actual_k-1)
                item['policies'][case] = {
                    'cached_features_cached_labels': metrics(cached_a, cached_k),
                    'native_features_cached_labels': metrics(cached_a, actual_k),
                    'native_features_native_b16_labels_clipped': metrics(native_a, actual_k),
                    'measured_native_retention': float(actual_a.sum()/native_a.sum()),
                    'native_vs_cached_length_disagreement_states': int((actual_k != cached_k).sum()),
                    'runtime_cached_vs_saved_prediction_length_disagreement_states': int((replay_cached_k != cached_k).sum()),
                    'actual_vs_clipped_native_acceptance_disagreement_states': int((actual_a != expected_a).sum()),
                    'actual_minus_clipped_native_accepted_tokens': int((actual_a-expected_a).sum())}
            out[f'c{c}_{mode}'] = item
    return out


def audit(bundle, training, replay_runs=()):
    verify(bundle)
    verify(training)
    metadata = json.loads((bundle/'bundle.json').read_text())
    if sha256(training/'COMPLETE.json') != metadata['source_training_complete_sha256']:
        raise ValueError('Training provenance mismatch')
    summary = json.loads((training/'summary.json').read_text())
    states = json.loads((bundle/'states.json').read_text())
    saved = dict(np.load(training/'assessment_predictions.npz', allow_pickle=False))
    calibration = dict(np.load(training/'calibration_predictions.npz', allow_pickle=False))
    if set(map(str, saved['prompt_ids'])) & set(map(str, calibration['prompt_ids'])):
        raise ValueError('Calibration/assessment prompt leakage')
    selected = align_rows(states, saved)
    cohort_prompts = {str(r['prompt_id']) for r in states}
    first = np.unique(saved['prompt_ids'], return_index=True)[1]
    calibration_first = np.unique(calibration['prompt_ids'], return_index=True)[1]
    subsets = {
        'calibration_all_cycles': (calibration, np.arange(len(calibration['accepted']))),
        'calibration_first_eligible_per_prompt': (calibration, np.sort(calibration_first)),
        'assessment_all_cycles': (saved, np.arange(len(saved['accepted']))),
        'replay_128_cached': (saved, selected),
        'assessment_all_cycles_replay_prompts': (saved, np.flatnonzero([str(p) in cohort_prompts for p in saved['prompt_ids']])),
        'assessment_first_eligible_per_prompt': (saved, np.sort(first)),
        'assessment_after_first_eligible': (saved, np.setdiff1d(np.arange(len(saved['accepted'])), first)),
        'assessment_other_cycles': (saved, np.setdiff1d(np.arange(len(saved['accepted'])), selected))}
    policies = {}
    for case, setting in metadata['policies'].items():
        key = Path(setting['checkpoint']).stem
        if sha256(bundle/setting['checkpoint']) != sha256(training/setting['checkpoint']):
            raise ValueError('Checkpoint mismatch')
        original = summary['models'][key]['primary']['calibration']
        recalculated = calibrate(calibration[key], calibration['accepted'], (.96,))['0.96']
        if setting['threshold'] != original['threshold'] or setting['threshold'] != recalculated['threshold']:
            raise ValueError('Frozen calibration threshold mismatch')
        reports = {}
        for name, (data, ix) in subsets.items():
            k = apply_setting(data[key][ix], setting)
            reports[name] = describe(data['accepted'][ix], k, data['prompt_ids'][ix])
        for field in ('retention', 'mean_kept_rows', 'aggregate_accept_ratio'):
            if not np.isclose(reports['assessment_all_cycles'][field], summary['models'][key]['primary']['assessment'][field], atol=1e-12, rtol=0):
                raise ValueError('Saved assessment summary cannot be reproduced')
        policies[case] = {'threshold': setting['threshold'], 'subsets': reports}
    return {'schema': 'predraft_retention_audit_v1',
            'source_training_complete_sha256': sha256(training/'COMPLETE.json'),
            'source_bundle_complete_sha256': sha256(bundle/'COMPLETE.json'),
            'cached_policies': policies,
            'replay_cohort_cycles': {str(c): sum(r['cycle'] == c for r in states) for c in sorted({r['cycle'] for r in states})},
            'raw_confidence_saved_full_assessment_control': summary['controls']['raw_confidence_POSTDRAFT_reference_only'],
            'native_replays': {str(run): replay_audit(run, states, saved, selected, metadata['policies']) for run in replay_runs},
            'limitations': ['No assessment fitting or checkpoint changes',
                'Cached metrics clip unchanged candidates; not actual shorter drafting or throughput',
                'Raw-confidence summary is full assessment, not the 128-state replay cohort',
                'Native decomposition is descriptive; different execution paths can change both features and candidates',
                'No native comparison performed unless native_replays is populated']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'training', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--replay', type=Path, action='append', default=[])
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    result = audit(args.bundle, args.training, args.replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result['cached_policies'], indent=2))


if __name__ == '__main__':
    main()
