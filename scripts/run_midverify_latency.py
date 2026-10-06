"""Bounded native-SGLang segmented-forward replay; not closed-loop serving."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify_latency import select_states
from scripts.audit_block_headroom import sha256
from scripts.profile_sglang_latency import ROOT, IMAGE, atomic_json, check_gpu, choose_http_port, command
from scripts.recover_legacy_ragged import restore
from scripts.train_midverify_scaling import load_verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-seconds', type=int, default=1800)
    p.add_argument('--predraft-bundle', type=Path)
    p.add_argument('--use-visible-gpu', action='store_true')
    a = p.parse_args()
    if a.output.exists() or not 60 <= a.max_seconds <= 3600:
        raise ValueError('Fresh output and bounded deadline required')
    if a.predraft_bundle and not a.use_visible_gpu:
        raise ValueError('Predraft workstation replay must use a Slurm GPU allocation')
    if os.environ.get('SLURM_JOB_ID') and not a.use_visible_gpu:
        raise ValueError('Do not override Slurm GPU allocation')
    if a.use_visible_gpu:
        from scripts.gpu_runtime import configure_gpu_runtime
        provenance = configure_gpu_runtime(use_visible_gpu=True, require_gpu=True)
        a.gpu = provenance['nvidia_smi_index']
        gpu = {'uuid': provenance['nvidia_smi_query_id'], 'slurm': provenance}
    else:
        gpu = check_gpu(a.gpu)
    locks = []
    for suffix in ('actual_block', 'midverify'):
        lock = (ROOT / f'gpu_{a.gpu}_{suffix}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    repo = Path(__file__).resolve().parents[1]
    if a.predraft_bundle:
        from scripts.export_predraft_latency_bundle import verify
        cache = policies = a.predraft_bundle.resolve()
        if not cache.is_relative_to(ROOT):
            raise ValueError('Bundle must be under the read-only mounted data root')
        verify(cache)
        collection = json.loads((cache/'bundle.json').read_text())
        if collection['schema'] != 'predraft_latency_bundle_v1':
            raise ValueError('Wrong bundle schema')
        source = cache/'states.json'
        rows = json.loads(source.read_text())
        if len(rows) != 128:
            raise ValueError('Frozen cohort must contain 128 assessment states')
        rows = rows[:4] if a.smoke else rows
    else:
        cache = ROOT / 'runs/midverify_scaling_10k_20260929/cache_nonterminal'
        policies = ROOT / 'runs/midverify_cascade_20260929/training'
        load_verified(cache)
        load_verified(policies)
        collection = json.loads((cache / 'config.json').read_text())
        source = cache / 'source_rows.json'
        if not source.exists():
            source = Path(collection['finalization_source']) / 'source_rows.json'
        rows = select_states(json.loads((cache / 'rows.json').read_text()),
                            json.loads(source.read_text()), 4 if a.smoke else 128)
    models = json.loads((ROOT / 'models.json').read_text())
    from dflash.predraft_latency import same_model_identity
    if not (same_model_identity(models, collection['models']) if a.predraft_bundle else models == collection['models']):
        raise ValueError('Pinned model mismatch')
    if a.predraft_bundle and any(Path(v['path']).name != v['revision'] or not Path(v['path']).is_dir() for v in models.values()):
        raise ValueError('Missing revision-pinned model snapshot')
    a.output.mkdir(parents=True)
    restore(repo / 'vendor/sglang_ragged_20260723', a.output / 'source')
    atomic_json(a.output / 'states.json', rows)
    config = {'gpu': gpu, 'image': IMAGE, 'models': models, 'cache': str(cache),
        'policies': str(policies), 'output': str(a.output), 'smoke': a.smoke,
        'concurrencies': [4] if a.smoke else [64, 128], 'seed': 913, 'target': .99,
        'warmups': 2 if a.smoke else 3, 'repeats': 2 if a.smoke else 12,
        'modes': ['eager', 'graph'], 'max_seconds': a.max_seconds,
        'cases': ['fixed16', 'fixed16_redraft', 'fixed8_same_candidates', 'fixed8_redraft', 'target_free', 'target_free_split', 'cascade'],
        'selection': 'First eligible state of each distinct assessment prompt, canonical order; same 128 states at both C; seed913 fixed in advance, not best-seed selection.',
        'scope': 'Same-state native SGLang layer/FlashInfer replay, eager or exact-shape manually captured target segments; eager drafter. Not scheduler/HTTP/closed-loop throughput. Saved B16 candidates unchanged except explicitly labeled fixed8_redraft/fixed16_redraft controls; fixed8_same_candidates is post-draft truncation only.',
        'limits': 'Fixed allocated prefix/suffix slots; reservation/free microcost reported separately. No prefill, request scheduling, graph bucket misses, allocator commit or future trajectory cost in replay cycle.',
        'commit': command(['git', 'rev-parse', 'HEAD'], cwd=repo).strip(),
        'bindings': {str(path): sha256(path) for path in (source, cache / 'COMPLETE.json', policies / 'COMPLETE.json', repo / 'scripts/midverify_latency_hook/latency_runtime.py')}}
    if a.predraft_bundle:
        config.update(suite='predraft_verification_trim', target=.96,
            cases=['fixed16', 'fixed12_same_candidates', 'fixed8_same_candidates',
                   'fixed12_redraft', 'fixed8_redraft', 'predraft_hard', 'predraft_mixed_tv'],
            selection=collection['selection'],
            scope='Same-state native SGLang replay with full B16 drafting for all candidate-preserving cases. Fresh native B16 generated once per snapshot and preserved across trim policies. Fixed8/12_redraft are separately labeled actual shorter-draft controls. Eager drafter, exact-shape target graph; not serving throughput.',
            frozen_policies=collection['policies'])
    atomic_json(a.output / 'config.json', config)
    runtime = a.output / 'runtime_cache'
    runtime.mkdir()
    name = 'atharv-midverify-latency-' + hashlib.sha256(str(a.output).encode()).hexdigest()[:12]
    env = {'PYTHONPATH': f'{repo}/scripts/midverify_latency_hook:{a.output}/source/python:{repo}',
        'DFLASH_MIDVERIFY_LATENCY_CONFIG': str(a.output / 'config.json'),
        'PYTHONDONTWRITEBYTECODE': '1', 'NVIDIA_TF32_OVERRIDE': '0', 'HF_HUB_OFFLINE': '1',
        'HF_HOME': str(ROOT / 'hf'), 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 'LOGNAME': getpass.getuser(), 'XDG_CACHE_HOME': str(runtime),
        'TRITON_CACHE_DIR': str(runtime / 'triton'), 'CUDA_CACHE_PATH': str(runtime / 'cuda'),
        'TORCHINDUCTOR_CACHE_DIR': str(runtime / 'inductor'), 'FLASHINFER_WORKSPACE_BASE': str(runtime / 'flashinfer'),
        'SGLANG_CACHE_DIR': str(runtime / 'sglang'), 'SGLANG_DG_CACHE_DIR': str(runtime / 'deep_gemm'),
        'TORCH_HOME': str(runtime / 'torch'), 'TORCH_EXTENSIONS_DIR': str(runtime / 'extensions'),
        'SGLANG_ENABLE_SPEC_V2': '1', 'SGLANG_ENABLE_DFLASH_SPEC_V2': '1', 'SGLANG_DFLASH_TIMING': '0'}
    launch = ['docker', 'run', '--name', name, '--network', 'host', '--gpus', 'device=' + gpu['uuid'],
        '--cpus', '12', '--shm-size', '8g', '--cap-drop', 'ALL', '--cap-add', 'DAC_OVERRIDE',
        '--security-opt', 'no-new-privileges', '--workdir', str(repo),
        '-v', f'{repo}:{repo}:ro', '-v', f'{ROOT}:{ROOT}:ro', '-v', f'{a.output}:{a.output}:rw']
    for key, value in env.items():
        launch += ['-e', f'{key}={value}']
    launch += ['--entrypoint', 'python', IMAGE, '-m', 'sglang.launch_server',
        '--model-path', models['target']['path'], '--speculative-algorithm', 'DFLASH',
        '--speculative-draft-model-path', models['draft']['path'], '--speculative-dflash-block-size', '16',
        '--speculative-num-draft-tokens', '16', '--host', '127.0.0.1', '--port', str(choose_http_port()),
        '--tp-size', '1', '--dtype', 'bfloat16', '--random-seed', '934', '--attention-backend', 'flashinfer',
        '--speculative-draft-attention-backend', 'flashinfer', '--mem-fraction-static', '0.50',
        '--max-running-requests', '128', '--max-total-tokens', '262144', '--context-length', '4096',
        '--disable-radix-cache', '--disable-cuda-graph', '--disable-piecewise-cuda-graph']
    atomic_json(a.output / 'launch.json', launch)
    started = time.monotonic()
    process = None
    def stopped(signum, frame):
        raise InterruptedError(f'Launcher received signal {signum}; stopping only its own container')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        with (a.output / 'server.log').open('x') as log:
            process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
            while time.monotonic() - started < a.max_seconds:
                if (a.output / 'COMPLETE.json').exists():
                    print('COMPLETE', str(a.output), flush=True)
                    return
                if (a.output / 'FAILED.json').exists() or process.poll() is not None:
                    raise RuntimeError('Benchmark worker failed; inspect preserved server.log / FAILED.json')
                time.sleep(2)
        raise TimeoutError('Bounded test deadline exceeded')
    except BaseException as error:
        atomic_json(a.output / 'LAUNCH_FAILED.json', {'error': repr(error), 'elapsed_s': time.monotonic()-started})
        raise
    finally:
        # Ignore repeated termination while cleaning up a daemon-owned container.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=25)
        if process:
            process.wait(timeout=20)


if __name__ == '__main__':
    main()
