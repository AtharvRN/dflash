"""Freeze GSM8K train-only calibration and complete test membership for history replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dflash_history import digest, save, select_prompts
from scripts.prepare_gsm8k_serving import REVISION, PROMPT, build_workload


def normalized(text):
    return ' '.join(text.casefold().split())


def prepare(train, test, tokenizer, output, calibration=512, seed=1007, fit_texts=()):
    if output.exists():
        raise FileExistsError(output)
    workload = build_workload(train, test, tokenizer, warmup=calibration, seed=seed)
    if workload['test_unique_questions'] != len(test):
        raise ValueError('test has duplicate questions; require explicit coverage policy')
    groups = {'calibration': workload['warmup'], 'assessment': workload['measurement']}
    fit_texts = [normalized(t) for t in fit_texts]
    overlaps = []
    for group, items in groups.items():
        data = train if group == 'calibration' else test
        for item in items:
            question = normalized(data[item['dataset_index']]['question'])
            if any(question in text for text in fit_texts):
                overlaps.append(item['prompt_id'])
    if overlaps:
        raise ValueError(f'GSM8K questions present in predictor fitting prompts: {overlaps}')
    output.mkdir(parents=True)
    entries, membership, gold = [], [], {}
    for group, items in groups.items():
        data = train if group == 'calibration' else test
        for item in items:
            pid = item['prompt_id']
            entries.append(dict(manifest_index=pid, source='openai/gsm8k',
                dataset_split=item['split'], dataset_index=item['dataset_index'],
                messages=[{'role':'user','content':PROMPT.format(question=data[item['dataset_index']]['question'])}]))
            membership.append(dict(prompt_id=pid, group=group, rows=1))
            gold[pid] = item['gold_answer']
    with (output/'messages.jsonl').open('x') as stream:
        for entry in entries:
            stream.write(json.dumps(entry)+'\n')
    # These are collector partitions, NOT the upstream GSM8K train/test labels.
    # No predictor fitting is done in this run. Both groups are evaluation-only.
    save(output/'train_prompt_ids.json', {'train_prompt_ids': []})
    save(output/'val_prompt_ids.json', {'val_prompt_ids': [e['manifest_index'] for e in entries]})
    save(output/'evaluation_groups.json', {'schema': 'membership only; not a feature cache', 'shards': membership})
    save(output/'gold_answers.json', gold)
    metadata = dict(dataset='openai/gsm8k', revision=REVISION, subset='main', seed=seed,
        source_rows_sha256=workload['source_rows_sha256'], protocol=workload['protocol'],
        upstream_train_rows=len(train), upstream_test_rows=len(test), calibration_rows=calibration,
        assessment_rows=len(test), calibration_source='train', assessment_source='test',
        predictor_fitting='none; previously fitted table is frozen',
        fitting_prompt_overlap_check='normalized question substring in supplied fitting messages; not a model-pretraining contamination audit',
        fitting_messages_checked=len(fit_texts))
    for group, count in [('calibration',calibration), ('assessment',len(test))]:
        selected = select_prompts(output/'messages.jsonl', output, group,
                                  output/'evaluation_groups.json', count, seed)
        if len(selected) != count:
            raise ValueError('collector would drop evaluation questions')
    metadata['file_hashes'] = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
    save(output/'dataset.json', metadata)
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--table', type=Path, required=True)
    p.add_argument('--fit-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.snapshot.name != REVISION:
        raise ValueError('unexpected GSM8K snapshot revision')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    train, test = [pq.read_table(a.snapshot/'main'/f'{s}-00000-of-00001.parquet').to_pylist()
                   for s in ('train','test')]
    if (len(train),len(test)) != (7473,1319):
        raise ValueError('unexpected pinned GSM8K split counts')
    table = json.loads(a.table.read_text())
    fit_ids = set(table['fit_prompt_ids'])
    fit_texts, found = [], set()
    with a.fit_manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            if str(row['manifest_index']) in fit_ids:
                found.add(str(row['manifest_index']))
                fit_texts.extend(m['content'] for m in row['messages'])
    if found != fit_ids:
        raise ValueError('cannot audit all predictor fitting prompts')
    models = json.loads(a.models.read_text())
    tokenizer = AutoTokenizer.from_pretrained(models['target']['path'], local_files_only=True)
    metadata = prepare(train, test, tokenizer, a.output, fit_texts=fit_texts)
    metadata.update(table_sha256=digest(a.table), fitting_manifest_sha256=digest(a.fit_manifest),
                    parquet_hashes={s:digest(a.snapshot/'main'/f'{s}-00000-of-00001.parquet') for s in ('train','test')})
    save(a.output/'dataset.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
