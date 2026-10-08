"""Frozen history predictor: train-split calibration, shared-state full GSM8K test assessment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dflash_history import digest, load_runs, save, select_prompts
from dflash.history_policy import HistoryValueTable, evaluate_rows, select_block


def bootstrap(rows, table, selections, draws=2000):
    import numpy as np
    pids = sorted({r['prompt_id'] for r in rows})
    index = {p:i for i,p in enumerate(pids)}
    choices = {**selections, **{f'fixed{b}': {'fixed':b} for b in table.blocks}}
    arrays = {name:np.zeros((len(pids),4)) for name in choices}
    for row in rows:
        for name, options in choices.items():
            b = options['fixed'] if 'fixed' in options else select_block(table,row['history'],**options)[0]
            arrays[name][index[row['prompt_id']]] += (1,row['outcomes'][str(b)]['accepted'],
                                                     b-1,row['outcomes']['16']['accepted'])
    rng = np.random.default_rng(1007)
    weights = rng.multinomial(len(pids),np.full(len(pids),1/len(pids)),size=draws)
    samples, result = {}, {}
    for name, data in arrays.items():
        sums = weights @ data
        metrics = dict(mean_accepted=sums[:,1]/sums[:,0], mean_block=1+sums[:,2]/sums[:,0],
            aggregate_accept_ratio=sums[:,1]/sums[:,2], retention_vs_B16=sums[:,1]/sums[:,3],
            proposed_work_reduction_vs_B16=1-sums[:,2]/(15*sums[:,0]))
        samples[name] = metrics
        result[name] = {k:np.quantile(v,[.025,.975]).tolist() for k,v in metrics.items()}
    paired = {}
    if 'history' in samples and 'history_free' in samples:
        paired = {k:np.quantile(samples['history'][k]-samples['history_free'][k],[.025,.975]).tolist()
                  for k in samples['history']}
    return dict(unit='prompt', prompts=len(pids), draws=draws, seed=1007,
                confidence=.95, conditional_on_frozen_calibration=True,
                intervals=result, history_minus_history_free=paired)


def coverage(paths, expected):
    summaries = [item for path in paths for item in json.loads((path/'summary.json').read_text())['prompts']]
    ids = [r['prompt_id'] for r in summaries]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('incomplete or duplicated question coverage')
    if any('skipped' in r or r.get('cycle_limited') for r in summaries):
        raise ValueError('full-question assessment cannot silently skip/cycle-cap prompts')
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    a = p.parse_args()
    root, data = a.root, a.dataset
    root.mkdir(parents=True,exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    metadata = json.loads((data/'dataset.json').read_text())
    if (metadata['calibration_rows'],metadata['assessment_rows']) != (512,1319):
        raise ValueError('expected 512 calibration and all 1319 test questions')
    if digest(a.table) != metadata['table_sha256']:
        raise ValueError('dataset fitting-overlap audit used a different table')
    for name,h in metadata['file_hashes'].items():
        if digest(data/name) != h:
            raise ValueError(f'prepared dataset changed: {name}')
    shutil.copy2(a.table,root/'frozen_table.json')
    table = HistoryValueTable(json.loads((root/'frozen_table.json').read_text()))
    if table.blocks != (4,8,12,16):
        raise ValueError('this study is the four-arm controller')
    save(root/'plan.json',dict(dataset=metadata, table_sha256=digest(a.table), blocks=list(table.blocks),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        behavior='fixed B16 for all shared-state collection', seed=1007, max_new_tokens=512,
        max_cycles=0, shard_prompts=128, workers=2, retention_target=.96,
        started_utc=time.time(), scope='matched-state policy generalization, not adaptive rollout or throughput'))
    def run(name,argv):
        print('START',name,flush=True)
        with (root/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,*map(str,argv)],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,flush=True)
    run('tests',['-m','pytest','-q','tests/test_history_policy.py','tests/test_gsm8k_history.py','tests/test_gsm8k_serving.py'])
    def collect_group(group,count):
        offsets = list(range(0,count,128))
        paths = [root/f'{group}_{i:04d}' for i in offsets]
        def collect(item):
            offset,path = item
            run(path.name,['scripts/dflash_history.py','collect','--blocks','4,8,12,16',
                '--fixed-block','16','--manifest',data/'messages.jsonl','--split-dir',data,
                '--pilot-manifest',data/'evaluation_groups.json','--models',a.models,'--group',group,
                '--prompt-offset',offset,'--limit-prompts',min(128,count-offset),'--max-new-tokens','512',
                '--max-cycles','0','--seed','1007','--output',path])
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(collect,zip(offsets,paths)))
        expected = [str(r['manifest_index']) for r in select_prompts(data/'messages.jsonl',data,group,
                     data/'evaluation_groups.json',count,1007)]
        if len(expected) != count:
            raise ValueError('question selection dropped rows')
        summaries = coverage(paths,expected)
        rows,provenance = load_runs(paths)
        if provenance['model_identity'] != table.payload['provenance']['model_identity']:
            raise ValueError('frozen predictor/model identity mismatch')
        return rows,provenance,summaries
    cal,cal_provenance,cal_summaries = collect_group('calibration',512)
    selections,cal_reports = {},{}
    for name,history_free in [('history',False),('history_free',True)]:
        sweep = []
        for i in range(50,101):
            options = dict(mode='retention',alpha=i/100,history_free=history_free)
            result = evaluate_rows(table,cal,**options)
            sweep.append(dict(options=options,metrics=result))
        feasible = [s for s in sweep if s['metrics']['adaptive']['retention_vs_largest'] >= .96]
        if feasible:
            best = max(feasible,key=lambda s:(s['metrics']['adaptive']['aggregate_accept_ratio'],s['options']['alpha']))
            selections[name] = best['options']
            cal_reports[name] = best
        else:
            cal_reports[name] = {'status':'no calibration-feasible threshold'}
        save(root/f'calibration_sweep_{name}.json',sweep)
    # Freeze decisions BEFORE collecting or opening test outcomes.
    save(root/'calibration_frozen.json',dict(selections=selections,reports=cal_reports,
        calibration_provenance=cal_provenance,table_sha256=digest(root/'frozen_table.json'),
        questions=512,eligible_cycles=len(cal),frozen_utc=time.time()))
    assessment,provenance,summaries = collect_group('assessment',1319)
    report = dict(scope='all GSM8K test questions; matched B16-behavior states, actual B4/B8/B12/B16 redrafts; no throughput claim',
        questions=1319,eligible_cycles=len(assessment),eligible_prompts=len({r['prompt_id'] for r in assessment}),
        calibration=cal_reports,assessment={name:evaluate_rows(table,assessment,**options) for name,options in selections.items()},
        fixed_controls=evaluate_rows(table,assessment,mode='retention',alpha=1)['fixed'],
        assessment_provenance=provenance, bootstrap=bootstrap(assessment,table,selections),
        generation_cap=512,baseline_length_capped_questions=sum(len(s['output_ids'])>=512 for s in summaries))
    # Score only the actual B16 behavior outputs, not hypothetical adaptive outputs.
    from transformers import AutoTokenizer
    from scripts.score_gsm8k_serving import numeric,predictions
    tokenizer = AutoTokenizer.from_pretrained(json.loads(a.models.read_text())['target']['path'],local_files_only=True)
    gold = json.loads((data/'gold_answers.json').read_text())
    correct = sum(predictions(tokenizer.decode(s['output_ids'],skip_special_tokens=True))[0] == numeric(gold[s['prompt_id']]) for s in summaries)
    report['baseline_B16_strict_boxed_accuracy'] = correct/1319
    report['accuracy_scope'] = 'custom zero-shot chat, thinking off; not standard few-shot lm-eval; no adaptive accuracy claim'
    save(root/'assessment_report.json',report)
    save(root/'complete.json',dict(completed_utc=time.time(),questions=1319))
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
