import copy
import json
from argparse import Namespace

import pytest

from dflash.history_policy import History, HistoryValueTable, select_block
from dflash.history_priced import calibrate_priced, candidate_rhos, validate_profile
from scripts.calibrate_history_priced import run
from scripts.dflash_history import digest, selection


def row(pid, group, accepted=(1, 7), cycle=0):
    return dict(schema_version=1, outcome_kind='actual_redraft', prompt_id=pid,
                group=group, cycle=cycle, history=History().snapshot(), eligible=True,
                terminal_or_capped=False,
                outcomes={str(b): dict(accepted=a) for b, a in zip((4, 8), accepted)})


def table():
    return HistoryValueTable.fit([row('fit', 'train')], (4, 8), prior_count=0,
                                 provenance=dict(model_identity={'target': 'test', 'draft': 'test'}))


def test_no_retention_floor_and_not_per_state_ratio():
    t = table()
    costs = {4: 1., 8: 8.}
    # V/T favors B4, but the paper's priced score at rho=0 favors B8.
    assert select_block(t, History().snapshot(), mode='priced', costs_ms=costs, rho=0)[0] == 8
    result = calibrate_priced(t, [row('cal', 'calibration')], costs)
    assert result['calibration']['adaptive']['mean_block'] == 4
    assert result['calibration']['adaptive']['retention_vs_largest'] == pytest.approx(1/7)
    assert result['calibration']['adaptive']['progress_per_modeled_ms'] == 2
    assert result['best_fixed_on_calibration'] == 4


def test_crossings_equal_costs_nonmonotonic_and_deterministic():
    costs = {4: 1., 8: 8.}
    candidates = candidate_rhos([(2., 8.)], (4, 8), costs)
    assert 6/7 in candidates
    assert any(0 < x < 6/7 for x in candidates)
    assert any(x > 6/7 for x in candidates)
    for costs in ({4: 5., 8: 5.}, {4: 8., 8: 1.}):
        result = calibrate_priced(table(), [row('cal', 'calibration')], costs)
        assert result['rho'] == 0
        assert result['calibration']['adaptive']['mean_block'] == 8


@pytest.mark.parametrize('rows', [[row('test', 'assessment')], [row('fit', 'calibration')],
    [row('cal', 'calibration'), row('cal', 'calibration')]])
def test_tuning_rejects_leakage_and_duplicates(rows):
    with pytest.raises(ValueError):
        calibrate_priced(table(), rows, {4: 1., 8: 8.})


def profile():
    p = dict(schema_version=1, units='ms', scope='whole_cycle', cost_basis='uniform_batch_cycle',
             model_identity=table().payload['provenance']['model_identity'],
             source_hashes={'measurement.json': 'a'*64}, costs_ms={'64': {'4': 1., '8': 8.}})
    for field in ('engine', 'engine_revision', 'gpu', 'dtype', 'attention_backend',
                  'graph_mode', 'context_workload', 'timing_boundaries'):
        p[field] = 'synthetic test only'
    return p


def test_profile_requires_all_arms_provenance_and_matching_models():
    p = profile()
    assert validate_profile(p, table(), 64) == {4: 1., 8: 8.}
    for field in ('gpu', 'engine_revision', 'source_hashes', 'timing_boundaries'):
        bad = copy.deepcopy(p)
        bad.pop(field)
        with pytest.raises(ValueError):
            validate_profile(bad, table(), 64)
    for change in ({'costs_ms': {'64': {'8': 8.}}}, {'model_identity': {}},
                   {'cost_basis': 'unknown'}, {'costs_ms': {'64': {'4': 1., '8': float('nan')}}}):
        with pytest.raises(ValueError):
            validate_profile({**p, **change}, table(), 64)


def write_json(path, obj):
    path.write_text(json.dumps(obj))
    return path


def collection(root, group, pid, accepted=(1, 7)):
    root.mkdir()
    write_json(root/'config.json', dict(command='collect', group=group,
               model_identity=table().payload['provenance']['model_identity']))
    (root/'cycles.jsonl').write_text(json.dumps(row(pid, group, accepted))+'\n')
    write_json(root/'summary.json', dict(cycles_sha256=digest(root/'cycles.jsonl'), prompts=[dict(prompt_id=pid)]))
    return root


def test_calibrate_freeze_assess_cli_and_bindings(tmp_path):
    tfile = write_json(tmp_path/'table.json', table().payload)
    measurement = write_json(tmp_path/'measurement.json', {'synthetic': True})
    p = profile()
    p['source_hashes'] = {'measurement.json': digest(measurement)}
    pfile = write_json(tmp_path/'profile.json', p)
    cal = collection(tmp_path/'cal', 'calibration', 'cal')
    args = Namespace(command='calibrate', table=tfile, cost_profile=pfile, concurrency=64,
        expected_prompt_ids=write_json(tmp_path/'cal_ids.json', ['cal']), runs=[cal], output=tmp_path/'frozen.json')
    frozen = run(args)
    assert set(frozen['policies']) == {'history', 'history_free'}
    rho = frozen['policies']['history']['rho']
    rollout_args = Namespace(priced_policy=args.output, table=tfile, cost_profile=pfile,
        concurrency=64, mode='priced', rho=None, history_free=False, policy_name='history')
    assert selection(rollout_args)['rho'] == rho
    rollout_args.concurrency = 32
    with pytest.raises(ValueError, match='concurrency'):
        selection(rollout_args)
    rollout_args.concurrency = 64
    rollout_args.rho = 1.
    with pytest.raises(ValueError, match='override'):
        selection(rollout_args)
    test = collection(tmp_path/'test', 'assessment', 'test', accepted=(0, 7))
    args = Namespace(command='assess', table=tfile, cost_profile=pfile, frozen=tmp_path/'frozen.json',
        expected_prompt_ids=write_json(tmp_path/'test_ids.json', ['test']), runs=[test], output=tmp_path/'report.json')
    report = run(args)
    assert report['policies']['history']['rho'] == rho  # No test retuning.
    assert report['policies']['history']['adaptive']['mean_block'] == 4
    with pytest.raises(ValueError, match='existing'):
        run(args)
    args.output = tmp_path/'new.json'
    write_json(args.expected_prompt_ids, ['test', 'missing'])
    with pytest.raises(ValueError, match='coverage'):
        run(args)
    write_json(args.expected_prompt_ids, ['test'])
    measurement.write_text('{}')
    with pytest.raises(ValueError, match='source hash'):
        run(args)
    write_json(pfile, {**p, 'gpu': 'changed'})
    with pytest.raises(ValueError, match='binding'):
        run(args)
