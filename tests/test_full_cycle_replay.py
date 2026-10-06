import hashlib
import numpy as np
import pytest

from dflash.predraft_latency import validate_replay_cohort
from scripts.export_predraft_latency_bundle import select_all_assessment
from scripts.summarize_predraft_latency import retention_intervals


def test_all_cycles_preserved_and_no_selection_on_acceptance():
    tokens = [10, 20]
    base = {'prefix_token_ids': tokens, 'prefix_length': 1,
            'prefix_sha256': hashlib.sha256(np.array(tokens, dtype=np.int64).tobytes()).hexdigest()}
    rows = [{**base, 'row': i, 'prompt_id': i//3, 'cycle': i%3,
             'group': 'assessment' if i else 'calibration', 'accepted_len': i%16} for i in range(12)]
    mask = np.ones(12, dtype=bool)
    mask[4] = False
    chosen = select_all_assessment(rows, mask)
    assert [r['row'] for r in chosen] == [1,2,3,5,6,7,8,9,10,11]
    validate_replay_cohort(chosen, {'cohort_mode': 'all_assessment_cycles', 'rows': 10})
    rows[2]['accepted_len'] = 15
    assert [r['row'] for r in select_all_assessment(rows, mask)] == [r['row'] for r in chosen]
    with pytest.raises(ValueError, match='Duplicate'):
        select_all_assessment([rows[1], rows[1]], [True, True])


def test_full_cohort_count_and_split_guards():
    rows = [{'prompt_id': 1, 'cycle': i, 'group': 'assessment'} for i in range(7)]
    m = {'cohort_mode': 'all_assessment_cycles', 'rows': 7}
    validate_replay_cohort(rows, m)  # Tail batch allowed; no dropped rows.
    with pytest.raises(ValueError, match='count'):
        validate_replay_cohort(rows[:-1], m)
    with pytest.raises(ValueError, match='128'):
        validate_replay_cohort(rows, {})
    rows[0]['group'] = 'calibration'
    with pytest.raises(ValueError, match='training/calibration'):
        validate_replay_cohort(rows, m)


def test_retention_bootstrap_clusters_all_cycles_of_each_prompt():
    # The only possible ratios with two prompt clusters are 0, .5, 1.
    a = np.array([0,0,2,2]); b = a.copy(); base = np.array([2,2,2,2])
    r = retention_intervals(a, b, base, [1,1,2,2], np.random.default_rng(0))
    assert r['bootstrap_prompts'] == 2
    assert r['prompt_bootstrap_retention_ci95'] == [0,1]
    assert r['prompt_bootstrap_retention_delta_vs_hard_ci95'] == [0,0]
