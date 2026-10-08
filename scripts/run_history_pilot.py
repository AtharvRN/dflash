"""Bounded NRP history-policy pilot, with durable stage logs and calibration-only tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dflash_history import load_runs, save
from dflash.history_policy import HistoryValueTable, blocks_checked, evaluate_rows


def pilot_blocks(value):
    blocks = blocks_checked(tuple(map(int, value.split(','))))
    if blocks[-1] != 16:
        raise ValueError('this pilot requires B16 as its largest/reference block')
    return blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/workspace/dflashv2_data'))
    p.add_argument('--blocks', type=pilot_blocks, default=tuple(range(2, 17)),
                   help='Allowed collection and policy blocks; e.g. 4,8,12,16. Largest must be 16.')
    args = p.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    blocks = args.blocks
    block_text = ','.join(map(str, blocks))
    save(root/'plan.json', dict(blocks=blocks, train_prompts=64, calibration_prompts=16,
        assessment_prompts=32, collection_max_cycles=24, max_new_tokens=256,
        retention_target=.96, workers=2, models=str(args.models), started_utc=time.time(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        scope='small Transformers history pilot, not SGLang throughput'))

    def run(name, argv):
        print('START', name, flush=True)
        with (root/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable, *map(str, argv)], cwd=repo,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)

    run('unit', ['-m', 'pytest', '-q', 'tests/test_history_policy.py',
                               'tests/test_policy_granularity.py', 'tests/test_postdraft_serving.py'])
    # BF16 width-dependent numerical divergence is recorded, never called exact.
    # Require strict FP32 AR parity as well as probe invariance in both precisions.
    run('pretrained_bf16', ['scripts/test_history_pretrained.py', '--models', args.models,
                      '--output', root/'pretrained_bf16.json', '--allow-ar-mismatch', '--blocks', block_text])
    run('pretrained_fp32', ['scripts/test_history_pretrained.py', '--models', args.models,
                      '--output', root/'pretrained_fp32.json', '--dtype', 'float32', '--blocks', block_text])
    common = ['--manifest', args.data_root/'manifests/qwen3_4b_instruct_100k_messages.jsonl',
              '--split-dir', args.data_root/'splits/qwen3_4b_instruct100k_full_4a100_manifest_seed0_val5pct_20260719',
              '--pilot-manifest', args.data_root/'runs/prefusion_pilot_20260915/cache/manifest.json',
              '--models', args.models, '--max-new-tokens', '256', '--seed', '1007']
    def collect(item):
        group, count = item
        run(group, ['scripts/dflash_history.py', 'collect', '--blocks', block_text,
                    '--group', group, '--limit-prompts', count, '--max-cycles', '24',
                    '--output', root/group, *common])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(collect, [('train', 64), ('calibration', 16), ('assessment', 32)]))
    run('fit', ['scripts/dflash_history.py', 'fit', '--runs', root/'train',
                '--blocks', block_text, '--output', root/'table.json'])
    table = HistoryValueTable(json.loads((root/'table.json').read_text()))
    cal, _ = load_runs([root/'calibration'])
    assess, _ = load_runs([root/'assessment'])
    report = {'scope': 'matched-state pilot; calibration-selected alpha; not throughput', 'arms': {}}
    selections = {}
    for name, history_free in [('history', False), ('history_free', True)]:
        sweep = []
        for i in range(50, 101):
            options = dict(mode='retention', alpha=i/100, history_free=history_free)
            metrics = evaluate_rows(table, cal, **options)
            sweep.append(dict(alpha=i/100, metrics=metrics))
        feasible = [s for s in sweep if s['metrics']['adaptive']['retention_vs_largest'] >= .96]
        if not feasible:
            report['arms'][name] = {'status': 'no calibration-feasible alpha', 'calibration_sweep': sweep}
            continue
        best = max(feasible, key=lambda s: (s['metrics']['adaptive']['aggregate_accept_ratio'], s['alpha']))
        options = dict(mode='retention', alpha=best['alpha'], history_free=history_free)
        selections[name] = options
        report['arms'][name] = dict(alpha=best['alpha'], calibration=best['metrics'],
                                    assessment=evaluate_rows(table, assess, **options), calibration_sweep=sweep)
    save(root/'matched_report.json', report)
    jobs = []
    for name, options in selections.items():
        jobs.append((name, ['--alpha', options['alpha']] + (['--history-free'] if options['history_free'] else [])))
    for b in (b for b in (4, 8, 12, 16) if b in blocks):
        jobs.append((f'fixed{b}', ['--fixed-block', b]))
    def rollout(item):
        name, extra = item
        run('rollout_'+name, ['scripts/dflash_history.py', 'rollout', '--table', root/'table.json',
            '--group', 'assessment', '--limit-prompts', '8', '--max-cycles', '0',
            '--output', root/('rollout_'+name), *common, *extra])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(rollout, jobs))
    summaries = {}
    for name, _ in jobs:
        directory = root/('rollout_'+name)
        summary = json.loads((directory/'summary.json').read_text())
        rows = [json.loads(line) for line in (directory/'cycles.jsonl').read_text().splitlines()]
        eligible = [r for r in rows if r['eligible']]
        n = len(eligible)
        summaries[name] = dict(prompts=len(summary['prompts']), cycles=len(rows), eligible_cycles=n,
            mean_block=sum(r['selected_block'] for r in eligible)/n if n else None,
            mean_accepted=sum(r['selected_accepted'] for r in eligible)/n if n else None,
            aggregate_accept_ratio=sum(r['selected_accepted'] for r in eligible)/sum(r['selected_block']-1 for r in eligible) if n else None,
            outputs={r['prompt_id']: r['output_ids'] for r in summary['prompts'] if 'output_ids' in r})
    reference = summaries['fixed16']['outputs']
    for summary in summaries.values():
        outputs = summary.pop('outputs')
        summary['exact_output_vs_fixed16'] = sum(outputs[k] == reference[k] for k in outputs)
        summary['compared_prompts'] = len(outputs)
    save(root/'rollout_report.json', dict(scope='closed-loop diagnostics; concurrent reference runners, no timing claims', methods=summaries))
    save(root/'complete.json', dict(completed_utc=time.time(), stages='units, pretrained, actual-block collection, fit, calibration, assessment, closed-loop controls'))


if __name__ == '__main__':
    main()
