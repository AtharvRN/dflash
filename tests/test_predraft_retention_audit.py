import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dflash.midverify import calibrate, apply_setting
from scripts.audit_predraft_retention import align_rows, describe
from scripts.prepare_confidence_replay_bundle import frozen_settings


def test_alignment_uses_row_identity_not_position_and_checks_labels():
    predictions = {'row_indices': np.array([31, 42]), 'prompt_ids': np.array([5, 6]),
                   'accepted': np.array([3, 4])}
    rows = [{'row': 42, 'prompt_id': 6, 'group': 'assessment', 'cached_accepted_len': 4},
            {'row': 31, 'prompt_id': 5, 'group': 'assessment', 'cached_accepted_len': 3}]
    assert align_rows(rows, predictions).tolist() == [1, 0]
    for field, value in [('group', 'calibration'), ('prompt_id', 99), ('cached_accepted_len', 9), ('row', 99)]:
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            align_rows(bad, predictions)
    with pytest.raises(ValueError, match='Duplicate'):
        align_rows([rows[0], rows[0]], predictions)


def test_metrics_are_aggregate_and_do_not_mutate_decisions():
    kept = np.array([2, 16])
    result = describe(np.array([1, 10]), kept, [1, 2])
    assert result['aggregate_accept_ratio'] == 11/16
    assert result['retention'] == 1
    assert result['b16_accepted_total'] == 11
    assert kept.tolist() == [2, 16]


def calibration_fixture():
    q = -np.tile(np.arange(1, 16, dtype=float), (3, 1))
    data = {'accepted': np.array([2, 5, 10]), 'hard_seed913': q}
    settings = calibrate(q, data['accepted'], (.9, .96, .99))
    metadata = {'policies': {'predraft_hard': {'checkpoint': 'hard_seed913.pt',
                 'threshold': settings['0.96']['threshold'], 'selected_update': 48}}}
    entry = {'selected_update': 48, 'operating_points': {k: {'calibration': v} for k,v in settings.items()}}
    summary = {'models': {'hard_seed913': entry},
               'controls': {'raw_confidence_POSTDRAFT_reference_only': {k: {'calibration': v} for k,v in settings.items()}}}
    return metadata, summary, data


def test_bundle_settings_need_only_calibration_and_preserve_checkpoint():
    m,s,d = calibration_fixture()
    result = frozen_settings(m,s,d,.99)
    assert result['predraft_hard']['checkpoint'] == 'hard_seed913.pt'
    assert result['predraft_hard']['selected_update'] == 48
    assert result['raw_confidence']['threshold'] == s['controls']['raw_confidence_POSTDRAFT_reference_only']['0.99']['calibration']['threshold']
    # Arbitrary assessment statistics cannot affect exported decisions.
    s['models']['hard_seed913']['assessment'] = {'retention': 0}
    assert result == frozen_settings(m,s,d,.99)
    s['models']['hard_seed913']['operating_points']['0.99']['calibration']['threshold'] = 100.
    with pytest.raises(ValueError, match='reproduced'):
        frozen_settings(m,s,d,.99)


def test_raw_rule_is_first_low_logprob_not_cumulative_probability():
    scores = np.full((1, 15), -.2, dtype=np.float32)
    scores[0, 3] = -2.
    assert apply_setting(scores, {'threshold': -.5}).tolist() == [4]
    assert apply_setting(scores, {'threshold': None}).tolist() == [16]


def test_runtime_confidence_control_computes_only_logprob_and_is_postdraft():
    source = (Path(__file__).resolve().parents[1]/'scripts/midverify_latency_hook/latency_runtime.py').read_text()
    body = source.split('    def run(self, timed=True):',1)[1].split('\ndef correctness',1)[0]
    assert body.index('draft_forward(s, timer') < body.index("elif case == 'raw_confidence':")
    confidence = source.split("if confidence == 'logprob_only':",1)[1].split('\n    return stats',1)[0]
    assert 'stats = candidate_logprob.reshape(snapshot.n, 15)' in confidence.split('else:')[0]
    assert '.exp()' not in confidence.split('else:')[0]


def test_replay_decomposition_rejects_unbound_feature_audits(tmp_path):
    from scripts.audit_block_headroom import sha256
    from scripts.audit_predraft_retention import replay_audit
    states = [{'row': 1, 'prompt_id': 1, 'group': 'assessment', 'cached_accepted_len': 2}]
    (tmp_path/'states.json').write_text(json.dumps(states))
    (tmp_path/'summary.json').write_text(json.dumps({'config': {'suite': 'predraft_verification_trim',
        'concurrencies': [1], 'modes': ['eager'], 'frozen_policies': {}}, 'results': []}))
    binding = {n: sha256(tmp_path/n) for n in ('states.json','summary.json')}
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'passed': True, 'binding': binding}))
    with pytest.raises(ValueError, match='Unbound'):
        replay_audit(tmp_path, states, {'accepted': np.array([2])}, np.array([0]), {})


def test_native_decomposition_separates_length_candidate_and_numerical_effects(tmp_path):
    from scripts.audit_block_headroom import sha256
    from scripts.audit_predraft_retention import replay_audit
    states = [{'prompt_id': 1}, {'prompt_id': 2}]
    scores = -np.tile(np.arange(1, 16, dtype=float), (2, 1))
    settings = {'predraft_hard': {'checkpoint': 'hard.pt', 'threshold': -3.5}}
    blocks = [list(range(16))]*2
    base = {'C': 2, 'mode': 'graph', 'offset': 0, 'prompt_ids': [1, 2],
            'case': 'fixed16', 'observations': [{'accepted': [4, 5], 'candidate_blocks': blocks}]}
    policy = {**base, 'case': 'predraft_hard', 'observations': [{'accepted': [2, 3],
               'candidate_blocks': blocks, 'end': [3, 5]}]}
    source = {'config': {'suite': 'predraft_verification_trim', 'concurrencies': [2],
                         'modes': ['graph'], 'frozen_policies': settings}, 'results': [base, policy]}
    files = {'states.json': states, 'summary.json': source,
             'predraft_feature_audit_c2_offset0.json': {'cached_vs_native_candidate_differences': 2,
                'policies': {'predraft_hard': {'native_lengths': [3, 5], 'cached_lengths': [4, 4]}}}}
    for n,v in files.items():
        (tmp_path/n).write_text(json.dumps(v))
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'passed': True,
         'binding': {n: sha256(tmp_path/n) for n in files}}))
    result = replay_audit(tmp_path, states, {'accepted': np.array([3, 5]), 'hard': scores},
                          np.array([0, 1]), settings)['c2_graph']['policies']['predraft_hard']
    assert result['cached_features_cached_labels']['retention'] == 6/8
    assert result['native_features_cached_labels']['retention'] == 6/8
    assert result['native_features_native_b16_labels_clipped']['retention'] == 6/9
    assert result['measured_native_retention'] == 5/9
    assert result['native_vs_cached_length_disagreement_states'] == 2
    assert result['actual_vs_clipped_native_acceptance_disagreement_states'] == 1
    assert result['actual_minus_clipped_native_accepted_tokens'] == -1


def test_confidence_summary_includes_timing_and_retention_intervals():
    from scripts.summarize_predraft_latency import summarize
    cases = ['fixed16', 'fixed8_redraft', 'fixed12_redraft', 'predraft_hard', 'raw_confidence']
    cells = []
    for case in cases:
        cells.append({'C': 2, 'mode': 'graph', 'case': case, 'rows': 2, 'offset': 0,
            'prompt_ids': [1, 2], 'observations': [{'committed_tokens': 7, 'accepted': [2, 3],
                 'candidate_blocks': [list(range(16))]*2}],
            'uninstrumented_cycle_ms': {'mean': 10., 'stdev': 0.},
            'uninstrumented_cycle_samples_ms': [10., 10.],
            'mean_front': 8., 'mean_end': 8., 'phase_stream_ms': {}, 'anchor_mismatches': 0,
            'audit': {'top1_differences': 0, 'acceptance_differences': 0,
                      'bonus_differences': 0, 'hidden': {'relative_l2': 0.}}})
    r = summarize({'config': {'suite': 'predraft_verification_trim', 'cases': cases}, 'results': cells})
    assert 'c2_graph_raw_confidence' in r['uncertainty']
    assert r['metrics']['c2_graph_raw_confidence']['speed_ratio_vs_predraft_hard'] == 1.
