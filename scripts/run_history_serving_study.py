"""One-shot actual HTTP benchmarks, frozen history policy versus every fixed arm.

No profiling. All 1,319 GSM8K test prompts, two repeats, natural EOS/max512.
High concurrency first; bounded by the current pod's expiry. Partial phases
remain durable and are never reported as a complete benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark_postdraft_serving import save, sha
from score_gsm8k_serving import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','scratch','models','workload','artifacts','table'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--concurrencies',type=int,nargs='+',default=[64,32,16,8])
    a=p.parse_args()
    if a.output.exists() or a.scratch.exists() or time.time()+1200>a.deadline_unix:
        raise ValueError('Need fresh destinations and at least 20 minutes remaining')
    workload=json.loads(a.workload.read_text())
    if len(workload['measurement'])!=1319 or workload['measurement_split']!='test':
        raise ValueError('Expected complete frozen GSM8K test workload')
    from gpu_runtime import wait_gpu_runtime
    allocation=wait_gpu_runtime(wait_seconds=30,use_container_gpu=True,require_gpu=True)
    a.output.mkdir(parents=True)
    a.scratch.mkdir(parents=True)
    save(a.output/'allocation.json',allocation)
    repo=Path(__file__).resolve().parents[1]
    save(a.output/'plan.json',dict(concurrencies=a.concurrencies,requests=1319,repeats=2,
        workload_sha256=sha(a.workload),deadline_unix=a.deadline_unix,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        scope='Actual HTTP output tokens / wall seconds. Profiling disabled. No rho retuning.',
        measurement_includes='prefill, generation, entropy/history/policy, packing, KV, scheduler, detokenization, client',
        graph_mode='target CUDA graphs; eager draft; graph buckets4/8/12/16 for adaptive',
        server_logging='node-local with between-phase durable backup'))
    def interrupted(signum,frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    child=None
    completed=[]
    try:
        for c in a.concurrencies:
            if time.time()+900>a.deadline_unix:
                save(a.output/'PARTIAL.json',dict(completed=completed,reason='pod deadline guard; unstarted phases remain'))
                return
            phase=a.output/f'c{c}'
            command=[sys.executable,'-u','scripts/benchmark_postdraft_serving.py',
                '--output',str(phase),'--scratch',str(a.scratch/f'c{c}'),'--models',str(a.models),
                '--workload-file',str(a.workload),'--cases',f'history{c}','fixed8','fixed12','fixed16','fixed4',
                '--concurrencies',str(c),'--requests','1319','--warmup','128','--repeats','2',
                '--max-new-tokens','512','--history-artifacts',str(a.artifacts),'--history-table',str(a.table),
                '--mem-fraction-static','.6','--max-total-tokens','98304','--port','22528']
            save(a.output/'progress.json',dict(concurrency=c,completed=completed,started_unix=time.time(),command=command))
            print(f'START C{c}',flush=True)
            with (a.output/f'c{c}.driver.log').open('x') as log:
                child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait(timeout=max(1,a.deadline_unix-time.time()-90))
                if child.returncode:
                    raise RuntimeError(f'C{c} failed: {child.returncode}')
                child=None
            save(phase/'gsm8k_summary.json',summarize(phase))
            completed.append(c)
            print(f'COMPLETE C{c}',flush=True)
        save(a.output/'COMPLETE.json',dict(completed=completed,finished_unix=time.time()))
    except BaseException as error:
        save(a.output/'FAILED.json',dict(error=repr(error),completed=completed,unix=time.time()))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)


if __name__=='__main__':
    main()
