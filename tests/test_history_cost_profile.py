import json
import shutil
from pathlib import Path

import pytest

from dflash.history_policy import HistoryValueTable
from scripts.dflash_history import digest
from scripts.export_history_cost_profile import export


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    run = tmp_path/'run'
    models, identity = {}, {}
    for role in ('target', 'draft'):
        path = tmp_path/role
        dump(path/'config.json', {'test': role})
        models[role] = dict(repo=role, revision='test', path=str(path))
        identity[role] = dict(repo=role, revision='test', config_sha256=digest(path/'config.json'))
    table = tmp_path/'table.json'
    dump(table, dict(schema_version=1, blocks=[4,8], certainty_edges=[], global_progress=[2.,4.],
        values={}, counts={}, fit_prompt_ids=['fit'], provenance=dict(model_identity=identity)))
    dump(run/'config.json', dict(args=dict(cycle_cost_profile=True, eager=False, cases=['fixed4','fixed8'],
        concurrencies=[8], repeats=2), models=models, extension={'test': True}, gpu='synthetic GPU'))
    dump(run/'COMPLETE.json', [dict(case=f'fixed{b}', concurrency=8, repeat=r) for b in (4,8) for r in (0,1)])
    dump(run/'workload.json', {'test': True})
    for b in (4,8):
        root = run/f'fixed{b}'/'cycle_profile'
        dump(root/'hook_123.json', {'test': True})
        records = []
        for label, batch, value in [('warmup',8,999), ('measured_b%d_c8_r0'%b,4,999),
            ('measured_b%d_c8_r0'%b,8,b), ('measured_b%d_c8_r1'%b,8,b+2)]:
            records.append(dict(label=label, batch_size=batch, block_size=b, prefix_lens=[100]*batch,
                target_graph=True, spans=[dict(name='decode_cycle', stream_elapsed_ms=value, host_call_ms=value)]))
        (root/'cycles_123.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return run, table


def test_exports_only_measured_full_batches(tmp_path):
    run, table = fixture(tmp_path)
    result = export(run, table, tmp_path/'profile.json', minimum_cycles=1)
    assert result['costs_ms'] == {'8': {'4': 5., '8': 9.}}
    assert result['statistics']['B4_C8']['cycles'] == 2
    assert result['statistics']['B4_C8']['mean_prefix'] == 100
    assert len(result['source_hashes']) == 7


def test_refuses_incomplete_profiles_and_sparse_full_batches(tmp_path):
    run, table = fixture(tmp_path)
    with pytest.raises(ValueError, match='Insufficient'):
        export(run, table, tmp_path/'profile.json')
    cells = json.loads((run/'COMPLETE.json').read_text())
    dump(run/'COMPLETE.json', cells[:-1])
    with pytest.raises(ValueError, match='Incomplete'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=1)


def test_rejects_eager_fallback_and_model_mismatch(tmp_path):
    run, table = fixture(tmp_path)
    path = run/'fixed4/cycle_profile/cycles_123.jsonl'
    path.write_text(path.read_text().replace('"target_graph": true','"target_graph": false'))
    with pytest.raises(ValueError, match='graphs'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=1)
    dump(tmp_path/'draft/config.json', {'changed': True})
    with pytest.raises(ValueError, match='config hash'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=1)


def test_reuse_frozen_train_workload_does_not_retokenize():
    from scripts.run_history_priced_study import validate_frozen_workload
    cal = [dict(manifest_index=f'train_{i}') for i in range(512)]
    prompts = [dict(prompt_id=f'train_{i}', input_ids=[1,2,3]) for i in range(320)]
    workload = dict(warmup=prompts[:64], measurement=prompts[64:], messages_sha256='bound')
    validate_frozen_workload(workload, cal, 'bound')
    with pytest.raises(ValueError, match='binding'):
        validate_frozen_workload(workload, cal, 'changed')
    workload['measurement'][0]['prompt_id'] = 'test_1'
    with pytest.raises(ValueError, match='calibration-only'):
        validate_frozen_workload(workload, cal, 'bound')


def test_pool_extra_cycles_without_overwriting_original(tmp_path):
    run, table = fixture(tmp_path)
    extra = tmp_path/'supplement'
    shutil.copytree(run, extra)
    hashes = {str(p):digest(p) for p in run.rglob('*') if p.is_file()}
    result = export(run, table, tmp_path/'profile.json', minimum_cycles=2, supplements=[extra])
    assert result['statistics']['B8_C8']['cycles'] == 4
    assert result['statistics']['B8_C8']['per_repeat']['0']['segments'] == {str(run):1,str(extra):1}
    assert hashes == {str(p):digest(p) for p in run.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Duplicate'):
        export(run, table, tmp_path/'other.json', supplements=[run])


@pytest.mark.parametrize('key,value', [('gpu','different'), ('environment',['different']),
                                       ('extension',{'different':True})])
def test_reject_incompatible_supplement(tmp_path, key, value):
    run, table = fixture(tmp_path)
    extra = tmp_path/'supplement'
    shutil.copytree(run, extra)
    config = json.loads((extra/'config.json').read_text())
    config[key] = value
    dump(extra/'config.json', config)
    with pytest.raises(ValueError, match='runtime mismatch'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=2, supplements=[extra])


def test_reject_changed_workload_or_hook(tmp_path):
    run, table = fixture(tmp_path)
    extra = tmp_path/'supplement'
    shutil.copytree(run, extra)
    dump(extra/'workload.json', {'changed': True})
    with pytest.raises(ValueError, match='workload mismatch'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=2, supplements=[extra])
    shutil.copy2(run/'workload.json', extra/'workload.json')
    dump(extra/'fixed4/cycle_profile/hook_123.json', {'changed': True})
    with pytest.raises(ValueError, match='hook identity'):
        export(run, table, tmp_path/'profile.json', minimum_cycles=2, supplements=[extra])
