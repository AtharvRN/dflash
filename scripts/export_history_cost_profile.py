"""Export measured full-batch worker-cycle costs; never infer B4 from other arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.history_policy import HistoryValueTable
from dflash.history_priced import validate_profile
from scripts.dflash_history import digest, save


def checked_run(run):
    config = json.loads((run/'config.json').read_text())
    args = config['args']
    completed = json.loads((run/'COMPLETE.json').read_text())
    expected = {(case, c, r) for case in args['cases'] for c in args['concurrencies']
                for r in range(args['repeats'])}
    cells = [(r['case'], r['concurrency'], r['repeat']) for r in completed]
    if set(cells) != expected or len(cells) != len(expected):
        raise ValueError('Incomplete or duplicate profiling phases')
    return config


def check_supplement(base, extra, config, other):
    """Match runtime/workload before pooling repeats; never select by observed cost."""
    for key in ('models', 'extension', 'gpu', 'environment'):
        if config.get(key) != other.get(key):
            raise ValueError(f'Supplement runtime mismatch: {key}')
    for key in ('cycle_cost_profile', 'eager', 'verify_length_audit', 'requests', 'warmup',
                'max_new_tokens', 'repeats', 'mem_fraction_static', 'max_total_tokens', 'runtime_site_packages'):
        if config['args'].get(key) != other['args'].get(key):
            raise ValueError(f'Supplement settings mismatch: {key}')
    if (not set(other['args']['cases']) <= set(config['args']['cases']) or
            not set(other['args']['concurrencies']) <= set(config['args']['concurrencies']) or
            max(other['args']['concurrencies']) != max(config['args']['concurrencies'])):
        raise ValueError('Supplement must use existing cases/concurrencies and identical max graph batch size')
    if digest(base/'workload.json') != digest(extra/'workload.json'):
        raise ValueError('Supplement workload mismatch')


def export(run, table_path, output, minimum_cycles=20, supplements=()):
    if output.exists():
        raise ValueError('Refusing existing profile')
    run_paths = [run, *supplements]
    if len({p.resolve() for p in run_paths}) != len(run_paths):
        raise ValueError('Duplicate supplement run')
    config = checked_run(run)
    configs = {run: config}
    for extra in supplements:
        configs[extra] = checked_run(extra)
        check_supplement(run, extra, config, configs[extra])
    args = config['args']
    table = HistoryValueTable(json.loads(table_path.read_text()))
    if not args.get('cycle_cost_profile') or args.get('eager'):
        raise ValueError('Requires instrumented graph-enabled fixed-B study')
    if set(args['cases']) != {f'fixed{b}' for b in table.blocks}:
        raise ValueError('Every candidate B must be measured')
    for role, expected in table.payload['provenance']['model_identity'].items():
        model = config['models'][role]
        if any(model[k] != expected[k] for k in ('repo', 'revision')):
            raise ValueError('Pinned model mismatch')
        if digest(Path(model['path'])/'config.json') != expected['config_sha256']:
            raise ValueError('Model config hash mismatch')
    sources = [p/name for p in run_paths for name in ('config.json', 'COMPLETE.json', 'workload.json')]
    stats, costs = {}, {}
    for case in args['cases']:
        b = int(case[5:])
        by_run, hook_identity = {}, None
        for path, cfg in configs.items():
            if case not in cfg['args']['cases']:
                continue
            root = path/case/'cycle_profile'
            hooks = list(root.glob('hook_*.json'))
            paths = list(root.glob('cycles_*.jsonl'))
            if len(hooks) != 1 or len(paths) != 1:
                raise ValueError('Expected one TP1 profiling worker per case')
            hook = json.loads(hooks[0].read_text())
            identity = {k:v for k,v in hook.items() if k != 'source_path'}
            if hook_identity is not None and identity != hook_identity:
                raise ValueError('Supplement worker/hook identity mismatch')
            hook_identity = identity
            sources.extend(hooks + paths)
            by_run[path] = [json.loads(line) for line in paths[0].read_text().splitlines()]
        for c in args['concurrencies']:
            samples, prefixes, graphs, host = [], [], [], []
            per_repeat = {}
            for repeat in range(args['repeats']):
                segments = {str(path): [r for r in records if r['label'] == f'measured_b{b}_c{c}_r{repeat}'
                                       and r['batch_size'] == c]
                            for path, records in by_run.items() if c in configs[path]['args']['concurrencies']}
                rows = [r for segment in segments.values() for r in segment]
                if len(rows) < minimum_cycles:
                    raise ValueError(f'Insufficient full-batch cycles: B{b} C{c} r{repeat}: {len(rows)}')
                times = []
                for row in rows:
                    if row['block_size'] != b:
                        raise ValueError('Wrong runtime block')
                    roots = [s for s in row['spans'] if s['name'] == 'decode_cycle']
                    if len(roots) != 1 or not 0 < roots[0]['stream_elapsed_ms'] < 10000:
                        raise ValueError('Invalid worker-cycle span')
                    times.append(roots[0]['stream_elapsed_ms'])
                    host.append(roots[0]['host_call_ms'])
                    prefixes.extend(row['prefix_lens'])
                    graphs.append(row['target_graph'])
                per_repeat[str(repeat)] = dict(cycles=len(times), mean_ms=statistics.mean(times),
                    segments={path:len(segment) for path,segment in segments.items()})
                samples.extend(times)
            if not all(graphs):
                raise ValueError(f'Not all full batches used target graphs: B{b} C{c}')
            costs.setdefault(str(c), {})[str(b)] = statistics.mean(samples)
            stats[f'B{b}_C{c}'] = dict(cycles=len(samples), mean_ms=statistics.mean(samples),
                median_ms=statistics.median(samples), stdev_ms=statistics.stdev(samples),
                mean_host_call_ms=statistics.mean(host), mean_prefix=statistics.mean(prefixes),
                min_prefix=min(prefixes), max_prefix=max(prefixes), target_graph_fraction=statistics.mean(graphs),
                per_repeat=per_repeat)
    profile = dict(schema_version=1, units='ms', scope='whole_cycle', cost_basis='uniform_batch_cycle',
        model_identity=table.payload['provenance']['model_identity'], costs_ms=costs,
        engine='Recovered SGLang spec-v2 serving worker with fixed-width DFlash',
        engine_revision=json.dumps(config['extension'], sort_keys=True), gpu=config['gpu'],
        dtype='bfloat16; TF32 off', attention_backend='FlashInfer target and drafter',
        graph_mode='target CUDA graph; eager drafter',
        context_workload='GSM8K train calibration prompts; fixed-B closed-loop trajectories; per-cell prefix statistics attached',
        timing_boundaries='Current-stream CUDA events around DFlashWorkerV2.forward_batch_generation, decode only. '
            'Includes draft/verify/setup/acceptance/KV bookkeeping within worker and CPU submission gaps. '
            'Excludes prefill, scheduler outside worker, HTTP and profiler metadata/drain. Not pure kernel time.',
        limitations=['Uniform-B trajectories have different prefix/cycle distributions.',
            'Instrumentation overhead not removed; not clean serving throughput.',
            'Mixed-width cost, controller cost, packing and graph-bucket effects unmeasured.'],
        statistics=stats, source_hashes={str(p.resolve()): digest(p) for p in sources},
        supplements=[str(p) for p in supplements], minimum_cycles_per_pooled_repeat=minimum_cycles,
        pooling='All full-batch cycles from original and supplied matched runs, weighted equally; no timing-based selection')
    for c in args['concurrencies']:
        validate_profile(profile, table, c)
    save(output, profile)
    return profile


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--supplements', type=Path, nargs='*', default=[])
    a = p.parse_args()
    export(a.run, a.table, a.output, supplements=a.supplements)
