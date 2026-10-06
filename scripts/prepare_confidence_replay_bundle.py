"""Add a frozen raw-confidence control; never fit on the replay assessment set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import calibrate
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.export_predraft_latency_bundle import verify


def frozen_settings(metadata, summary, calibration, target):
    key = str(float(target))
    policies = {}
    for case, old in metadata['policies'].items():
        name = Path(old['checkpoint']).stem
        entry = summary['models'][name]
        if entry['selected_update'] != old['selected_update']:
            raise ValueError('Checkpoint selection mismatch')
        chosen = entry['operating_points'][key]['calibration']
        recalculated = calibrate(calibration[name], calibration['accepted'], (target,))[key]
        if chosen['threshold'] != recalculated['threshold']:
            raise ValueError('Saved calibration cannot be reproduced')
        policies[case] = {**old, 'threshold': chosen['threshold']}
    raw = summary['controls']['raw_confidence_POSTDRAFT_reference_only'][key]['calibration']
    if raw['retention'] < target - 1e-12:
        raise ValueError('Raw confidence failed calibration constraint')
    policies['raw_confidence'] = {'kind': 'candidate_logprob_threshold',
        'threshold': raw['threshold'], 'calibration': raw,
        'decision': 'Longest prefix with each candidate logprob >= threshold; not a cumulative product',
        'input': 'Current full-B16 draft logits only; no target verification input'}
    return policies


def prepare(bundle, training, output, target=.96):
    if output.exists():
        raise ValueError('Fresh output required')
    verify(bundle)
    verify(training)
    metadata = json.loads((bundle/'bundle.json').read_text())
    if metadata['schema'] != 'predraft_latency_bundle_v1':
        raise ValueError('Expected original frozen replay bundle')
    if sha256(training/'COMPLETE.json') != metadata['source_training_complete_sha256']:
        raise ValueError('Wrong training provenance')
    summary = json.loads((training/'summary.json').read_text())
    calibration = dict(np.load(training/'calibration_predictions.npz', allow_pickle=False))
    policies = frozen_settings(metadata, summary, calibration, target)
    output.mkdir(parents=True)
    for name in ['states.json', 'cached_fused.npy', *[s['checkpoint'] for s in metadata['policies'].values()]]:
        if Path(name).name != name:
            raise ValueError('Invalid bundle path')
        shutil.copy2(bundle/name, output/name)
    atomic_json(output/'bundle.json', {**metadata, 'schema': 'predraft_confidence_replay_bundle_v1',
        'policies': policies, 'calibration_target': float(target),
        'parent_bundle_complete_sha256': sha256(bundle/'COMPLETE.json'),
        'setting': 'Frozen seed913 checkpoints selected at calibration96; thresholds for requested target use calibration only',
        'comparison_scope': 'Calibration-matched, NOT guaranteed assessment-retention-matched; no assessment fitting',
        'raw_calibration_provenance': 'Bound original training summary; raw calibration arrays not present in replay bundle'})
    files = sorted(p.name for p in output.iterdir() if p.is_file())
    atomic_json(output/'COMPLETE.json', {'complete': True, 'binding': {n: sha256(output/n) for n in files}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'training', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--target', type=float, choices=(.90, .95, .96, .98, .99, 1.), default=.96)
    a = p.parse_args()
    prepare(a.bundle, a.training, a.output, a.target)
    print(a.output)


if __name__ == '__main__':
    main()
