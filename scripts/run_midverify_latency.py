"""Bounded native-SGLang segmented-forward replay; not closed-loop serving."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
from pathlib import Path
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
    a = p.parse_args()
    if a.output.exists() or not 60 <= a.max_seconds <= 3600:
        raise ValueError('Fresh output and bounded deadline required')
    locks = []
    for suffix in ('actual_block', 'midverify'):
        lock = (ROOT / f'gpu_{a.gpu}_{suffix}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    gpu = check_gpu(a.gpu)
    repo = Path(__file__).resolve().parents[1]
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
    if models != collection['models']:
        raise ValueError('Pinned model mismatch')
    a.output.mkdir(parents=True)
    restore(repo / 'vendor/sglang_ragged_20260723', a.output / 'source')
    atomic_json(a.output / 'states.json', rows)
    config = {'gpu': gpu, 'image': IMAGE, 'models': models, 'cache': str(cache),
        'policies': str(policies), 'output': str(a.output), 'smoke': a.smoke,
        'concurrencies': [4] if a.smoke else [64, 128], 'seed': 913, 'target': .99,
        'warmups': 2 if a.smoke else 3, 'repeats': 2 if a.smoke else 12,
        'modes': ['eager', 'graph'], 'max_seconds': a.max_seconds,
        'cases': ['fixed16', 'fixed8_same_candidates', 'target_free', 'target_free_split', 'cascade'],
        'selection': 'First eligible state of each distinct assessment prompt, canonical order; same 128 states at both C; seed913 fixed in advance, not best-seed selection.',
        'scope': 'Same-state native SGLang layer/FlashInfer replay, eager or exact-shape manually captured target segments; eager drafter. Not scheduler/HTTP/closed-loop throughput. Full B16 candidates unchanged, B8 means post-draft fixed truncation, NOT rerunning B8 drafting.',
        'limits': 'Fixed allocated prefix/suffix slots; reservation/free microcost reported separately. No prefill, request scheduling, graph bucket misses, allocator commit or future trajectory cost in replay cycle.',
        'commit': command(['git', 'rev-parse', 'HEAD'], cwd=repo).strip(),
        'bindings': {str(path): sha256(path) for path in (source, cache / 'COMPLETE.json', policies / 'COMPLETE.json', repo / 'scripts/midverify_latency_hook/latency_runtime.py')}}
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
        subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=25)
        if process:
            process.wait(timeout=20)


if __name__ == '__main__':
    main()
