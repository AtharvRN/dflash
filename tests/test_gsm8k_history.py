import json
from pathlib import Path

import pytest

from scripts.prepare_gsm8k_history import prepare
from scripts.dflash_history import select_prompts
from scripts.run_gsm8k_history import bootstrap, coverage
from dflash.history_policy import History, HistoryValueTable


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert 'SECRET_GOLD' not in messages[0]['content']
        return [1,2,3]


def dataset(tmp_path):
    train = [{'question':f'Training question {i}?','answer':'SECRET_GOLD #### 2'} for i in range(8)]
    test = [{'question':f'Test question {i}?','answer':'SECRET_GOLD #### 3'} for i in range(5)]
    output = tmp_path/'dataset'
    prepare(train,test,Tokenizer(),output,calibration=4)
    return output,train,test


def test_train_calibration_full_test_and_shards(tmp_path):
    d,train,test = dataset(tmp_path)
    common = (d/'messages.jsonl',d,'assessment',d/'evaluation_groups.json')
    full = select_prompts(*common,5,1007)
    shards = select_prompts(*common,3,1007,0)+select_prompts(*common,3,1007,3)
    assert shards == full
    assert len({r['manifest_index'] for r in full}) == 5
    assert all(r['dataset_split']=='test' for r in full)
    cal = select_prompts(d/'messages.jsonl',d,'calibration',d/'evaluation_groups.json',4,1007)
    assert all(r['dataset_split']=='train' for r in cal)
    assert not {r['manifest_index'] for r in cal}&{r['manifest_index'] for r in full}
    assert 'SECRET_GOLD' not in (d/'messages.jsonl').read_text()
    with pytest.raises(ValueError):
        select_prompts(*common,3,1007,-1)


def test_fitting_overlap_fails_without_dropping_test(tmp_path):
    _,train,test = dataset(tmp_path)
    with pytest.raises(ValueError,match='fitting prompts'):
        prepare(train,test,Tokenizer(),tmp_path/'overlap',calibration=4,fit_texts=['Answer Test question 2? carefully'])


def test_coverage_requires_every_question_once(tmp_path):
    shard = tmp_path/'shard'
    shard.mkdir()
    p = shard/'summary.json'
    p.write_text(json.dumps({'prompts':[{'prompt_id':'a','cycle_limited':False}]}))
    assert len(coverage([shard],['a'])) == 1
    with pytest.raises(ValueError):
        coverage([shard],['a','b'])
    with pytest.raises(ValueError):
        coverage([shard,shard],['a'])
    p.write_text(json.dumps({'prompts':[{'prompt_id':'a','skipped':'prompt_length'}]}))
    with pytest.raises(ValueError):
        coverage([shard],['a'])


def test_bootstrap_resamples_prompts_with_paired_methods():
    row = dict(schema_version=1,outcome_kind='actual_redraft',prompt_id='fit',cycle=0,group='train',
        eligible=True,terminal_or_capped=False,history=History().snapshot(),
        outcomes={str(b):{'accepted':2} for b in (4,8,12,16)})
    table = HistoryValueTable.fit([row],(4,8,12,16))
    rows = [{**row,'prompt_id':'test','group':'assessment','cycle':i} for i in range(3)]
    options = dict(mode='retention',alpha=.96)
    result = bootstrap(rows,table,{'history':options,'history_free':{**options,'history_free':True}},draws=20)
    assert result['prompts'] == 1
    assert result['intervals']['history']['mean_block'] == [4,4]
    assert result['history_minus_history_free']['aggregate_accept_ratio'] == [0,0]
