"""Supplement undersampled C64 cells, pool provenance-checked costs, freeze/evaluate.

Keeps the original failed-export run intact. Adds two further 256-request repeats
per B12/B16 at C64 on the identical train-only workload. The minimum remains 20
full-batch cycles per pooled repeat; inference repeats are not new unique prompts.
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dflash_history import digest, save
from scripts.export_history_cost_profile import export
from scripts.calibrate_history_priced import run as priced_run


def main():
    p = ArgumentParser(description=__doc__)
    for name in ('original', 'collection', 'output', 'scratch', 'models'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    if a.output.exists() or a.scratch.exists() or a.deadline_unix-time.time() < 1800:
        raise ValueError('Fresh output/scratch and sufficient lifetime required')
    from scripts.gpu_runtime import wait_gpu_runtime
    allocation = wait_gpu_runtime(wait_seconds=30, use_container_gpu=True, require_gpu=True)
    a.output.mkdir(parents=True)
    save(a.output/'allocation.json', allocation)
    repo = Path(__file__).resolve().parents[1]
    source_config = json.loads((a.original/'profile/config.json').read_text())
    source_args = source_config['args']
    command = [sys.executable, '-u', 'scripts/benchmark_postdraft_serving.py',
        '--output', str(a.output/'supplement'), '--scratch', str(a.scratch), '--models', str(a.models),
        '--workload-file', str(a.original/'workload.json'), '--cases','fixed12','fixed16',
        '--concurrencies','64', '--requests',str(source_args['requests']), '--warmup',str(source_args['warmup']),
        '--max-new-tokens',str(source_args['max_new_tokens']), '--repeats',str(source_args['repeats']),
        '--cycle-cost-profile', '--mem-fraction-static',str(source_args['mem_fraction_static']),
        '--max-total-tokens',str(source_args['max_total_tokens']), '--port','22618']
    save(a.output/'plan.json', dict(command=command, original=str(a.original),
        original_config_sha256=digest(a.original/'profile/config.json'),
        workload_sha256=digest(a.original/'workload.json'), deadline_unix=a.deadline_unix,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        scope='Count-driven supplement, no lowering sample threshold; offline modeled policy gains only'))
    child = None

    def phase(name, **kwargs):
        save(a.output/'progress.json', dict(phase=name, unix=time.time(), **kwargs))
        print(name, kwargs, flush=True)

    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        phase('supplementing_c64')
        with (a.output/'profile.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=max(1,a.deadline_unix-time.time()-180))
            if child.returncode:
                raise RuntimeError(f'Supplement failed: {child.returncode}')
            child = None
        phase('exporting_pooled_costs')
        table = a.collection/'frozen_table.json'
        profile = a.output/'cost_profile.json'
        export(a.original/'profile', table, profile, supplements=[a.output/'supplement'])
        for c in (8,16,32,64):
            phase('calibrating', concurrency=c)
            priced_run(Namespace(command='calibrate', table=table, cost_profile=profile, concurrency=c,
                runs=sorted(a.collection.glob('calibration_[0-9][0-9][0-9][0-9]')),
                expected_prompt_ids=a.original/'calibration_ids.json', output=a.output/f'frozen_c{c}.json'))
        for c in (8,16,32,64):
            phase('assessing', concurrency=c)
            priced_run(Namespace(command='assess', table=table, cost_profile=profile,
                frozen=a.output/f'frozen_c{c}.json', runs=sorted(a.collection.glob('assessment_[0-9][0-9][0-9][0-9]')),
                expected_prompt_ids=a.original/'assessment_ids.json', output=a.output/f'assessment_c{c}.json'))
        phase('complete')
        save(a.output/'COMPLETE.json', dict(finished_unix=time.time(),
            files={p.name:digest(p) for p in a.output.glob('*.json')}))
    except BaseException as error:
        save(a.output/'FAILED.json', dict(error=repr(error), unix=time.time()))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
