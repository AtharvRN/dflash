"""Bind existing post-draft checkpoints to the full assessment replay cohort."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.postdraft_latency import settings
from scripts.export_predraft_latency_bundle import verify
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json


def prepare(bundle, training, output, target=.96):
    if output.exists():
        raise ValueError('Fresh output required')
    verify(bundle)
    verify(training)
    metadata = json.loads((bundle/'bundle.json').read_text())
    config = json.loads((training/'config.json').read_text())
    if metadata['cohort_mode'] != 'all_assessment_cycles':
        raise ValueError('Use all eligible assessment cycles')
    if config['schema'] != 'soft_supervision_training_v1' or config['smoke']:
        raise ValueError('Require completed full post-draft training')
    if config['cache_complete_sha256'] != metadata['source_cache_complete_sha256']:
        raise ValueError('Post-draft training and replay cache differ')
    summary = json.loads((training/'summary.json').read_text())
    policies = settings(summary, target)
    output.mkdir(parents=True)
    for name in ['states.json', 'cached_fused.npy']:
        shutil.copy2(bundle/name, output/name)
    for policy in policies.values():
        if 'checkpoint' in policy:
            shutil.copy2(training/policy['checkpoint'], output/policy['checkpoint'])
    atomic_json(output/'bundle.json', {**metadata, 'schema': 'postdraft_latency_bundle_v1',
        'policies': policies, 'calibration_target': target,
        'parent_bundle_complete_sha256': sha256(bundle/'COMPLETE.json'),
        'source_training': str(training), 'source_training_complete_sha256': sha256(training/'COMPLETE.json'),
        'setting': 'Frozen seed913 post-draft checkpoints and calibration-only thresholds; no refitting',
        'comparison_scope': 'Calibration-matched, not guaranteed assessment-retention-matched'})
    files = sorted(p.name for p in output.iterdir())
    atomic_json(output/'COMPLETE.json', {'complete': True, 'binding': {n:sha256(output/n) for n in files}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('bundle', 'training', 'output'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--target', type=float, choices=(.9,.95,.96,.98,.99,1.), default=.96)
    a=p.parse_args()
    prepare(a.bundle,a.training,a.output,a.target)
    print(a.output)
