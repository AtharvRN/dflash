"""One-shot dependency-gated cost profiling, frozen calibration, then assessment.

Does not stop the active collector or create a recurring monitor. Fails if the
collector exits without completion, GPU is occupied, or pod deadline is too near.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dflash_history import digest, save, select_prompts
from scripts.calibrate_history_priced import run as priced_run
from scripts.export_history_cost_profile import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'scratch', 'collection', 'dataset', 'models'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--collector-pid', type=int,
                   help='Required only when collection is still running')
    p.add_argument('--workload-file', type=Path,
                   help='Reuse the previously frozen train-only tokenized cost workload')
    p.add_argument('--runtime-site-packages', nargs='+', default=[])
    p.add_argument('--deadline-unix', type=float, required=True,
                   help='Hard stop before pod deadline, including child cleanup allowance')
    a = p.parse_args()
    if a.output.exists() or a.scratch.exists() or a.deadline_unix <= time.time():
        raise ValueError('Fresh destinations and future bounded deadline required')
    a.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    plan = dict(args={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
        blocks=[4,8,12,16], concurrencies=[8,16,32,64], requests=256, warmup=64, repeats=2,
        max_new_tokens=512, started_unix=time.time(),
        scope='Measured fixed-B instrumented worker-cycle costs; offline priced-policy evaluation. No adaptive speedup claim.')
    save(a.output/'plan.json', plan)
    child = None

    def phase(name, **kwargs):
        print(name, flush=True)
        save(a.output/'progress.json', dict(phase=name, unix=time.time(), **kwargs))

    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        phase('waiting_for_collection')
        while not (a.collection/'complete.json').exists():
            if a.collector_pid is None:
                raise ValueError('Collection is incomplete and no collector PID was supplied')
            proc = Path(f'/proc/{a.collector_pid}/cmdline')
            if not proc.exists() or b'run_gsm8k_history.py' not in proc.read_bytes():
                raise RuntimeError('Collector ended without completion; preserving partial evidence')
            if a.deadline_unix - time.time() < 3600:
                raise TimeoutError('Less than one hour remains for profiling; no GPU job launched')
            time.sleep(30)
        if a.deadline_unix - time.time() < 3600:
            raise TimeoutError('Insufficient remaining lifetime')
        from scripts.gpu_runtime import wait_gpu_runtime
        allocation = wait_gpu_runtime(wait_seconds=60, use_container_gpu=True, require_gpu=True)
        save(a.output/'allocation.json', allocation)
        phase('preparing_calibration_workload')
        metadata = json.loads((a.dataset/'dataset.json').read_text())
        for name, expected_hash in metadata['file_hashes'].items():
            if digest(a.dataset/name) != expected_hash:
                raise ValueError('Prepared GSM8K data binding changed')
        table = a.collection/'frozen_table.json'
        if digest(table) != metadata['table_sha256']:
            raise ValueError('Wrong frozen table')
        expected = {}
        paths = {}
        for group, count in [('calibration',512), ('assessment',1319)]:
            prompts = select_prompts(a.dataset/'messages.jsonl', a.dataset, group,
                                    a.dataset/'evaluation_groups.json', count, 1007)
            if len(prompts) != count:
                raise ValueError('Unexpected GSM8K question count')
            expected[group] = [str(r['manifest_index']) for r in prompts]
            save(a.output/f'{group}_ids.json', expected[group])
            paths[group] = sorted(a.collection.glob(f'{group}_[0-9][0-9][0-9][0-9]'))
        models = json.loads(a.models.read_text())
        cal = select_prompts(a.dataset/'messages.jsonl', a.dataset, 'calibration',
                            a.dataset/'evaluation_groups.json', 512, 1007)
        if a.workload_file:
            workload = json.loads(a.workload_file.read_text())
            validate_frozen_workload(workload, cal, digest(a.dataset/'messages.jsonl'))
            save(a.output/'workload_binding.json', dict(source=str(a.workload_file), sha256=digest(a.workload_file)))
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(models['target']['path'], local_files_only=True)
            prompts = []
            for row in cal[:320]:
                ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                if not 1 <= len(ids) <= 2048:
                    raise ValueError('Unexpected calibration prompt length')
                prompts.append(dict(prompt_id=str(row['manifest_index']), input_ids=ids))
            workload = dict(warmup=prompts[:64], measurement=prompts[64:],
                selection='First 320 seed1007-selected GSM8K train calibration prompts; first64 warmup, next256 measured; no test questions',
                messages_sha256=digest(a.dataset/'messages.jsonl'))
        save(a.output/'workload.json', workload)
        phase('profiling_fixed_blocks')
        command = [sys.executable, '-u', 'scripts/benchmark_postdraft_serving.py',
            '--output', str(a.output/'profile'), '--scratch', str(a.scratch), '--models', str(a.models),
            '--workload-file', str(a.output/'workload.json'), '--cases', 'fixed4','fixed8','fixed12','fixed16',
            '--concurrencies','8','16','32','64', '--requests','256','--warmup','64',
            '--max-new-tokens','512','--repeats','2','--cycle-cost-profile',
            '--mem-fraction-static','0.6','--max-total-tokens','98304','--port','22618']
        if a.runtime_site_packages:
            command += ['--runtime-site-packages', *a.runtime_site_packages]
        save(a.output/'launch.json', command)
        with (a.output/'profile.log').open('x') as log:
            child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=max(1, a.deadline_unix-time.time()-120))
            if child.returncode:
                raise RuntimeError(f'Cost profiler failed: {child.returncode}; inspect profile.log')
            child = None
        phase('exporting_costs')
        profile = a.output/'cost_profile.json'
        export(a.output/'profile', table, profile)
        # Freeze EVERY concurrency before opening assessment labels.
        for c in plan['concurrencies']:
            phase('calibrating', concurrency=c)
            priced_run(Namespace(command='calibrate', table=table, cost_profile=profile, concurrency=c,
                runs=paths['calibration'], expected_prompt_ids=a.output/'calibration_ids.json',
                output=a.output/f'frozen_c{c}.json'))
        for c in plan['concurrencies']:
            phase('assessing', concurrency=c)
            priced_run(Namespace(command='assess', table=table, cost_profile=profile,
                frozen=a.output/f'frozen_c{c}.json', runs=paths['assessment'],
                expected_prompt_ids=a.output/'assessment_ids.json', output=a.output/f'assessment_c{c}.json'))
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


def validate_frozen_workload(workload, cal, messages_hash):
    if workload.get('messages_sha256') != messages_hash:
        raise ValueError('Frozen workload messages binding changed')
    if len(workload['warmup']) != 64 or len(workload['measurement']) != 256:
        raise ValueError('Expected exactly 64 warmup and 256 measurement prompts')
    prompts = workload['warmup'] + workload['measurement']
    if [r['prompt_id'] for r in prompts] != [str(r['manifest_index']) for r in cal[:320]]:
        raise ValueError('Frozen cost workload must match the preselected calibration-only prompts')
    if any(not 1 <= len(r['input_ids']) <= 2048 or
           any(type(t) is not int or t < 0 for t in r['input_ids']) for r in prompts):
        raise ValueError('Invalid frozen input tokens')


if __name__ == '__main__':
    main()
