import hashlib
import numpy as np
import pytest

from dflash.midverify_latency import select_states, packed_indices, compact_indices, accepted_from_top1


def test_all_integer_prefixes_and_compaction():
    front = np.arange(1, 17)
    end = np.maximum(1, front - 3)
    assert np.array_equal(packed_indices(front)[compact_indices(front, end)], packed_indices(end))
    assert len(packed_indices(front)) == front.sum()
    with pytest.raises(ValueError):
        compact_indices(end, front)
    with pytest.raises(ValueError):
        packed_indices([0, 16])


def test_anchor_bonus_and_first_rejection():
    block = np.tile(np.arange(16), (4, 1))
    top = block + 1
    top[2, 2] = 99
    a, bonus = accepted_from_top1(block, top, [1, 16, 8, 4])
    np.testing.assert_array_equal(a, [0, 15, 2, 3])
    np.testing.assert_array_equal(bonus, [1, 16, 99, 4])


def test_state_selection_has_no_label_filter_and_no_duplicate_prompts():
    tokens = [11, 12, 13]
    base = {'prefix_length': 2, 'prefix_token_ids': tokens, 'draft_ids': list(range(15)),
            'prefix_sha256': hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()}
    source = [{**base, 'prompt_id': str(i), 'cycle': j, 'group': 'assessment' if i else 'train'}
              for i in range(4) for j in range(2)]
    rows = [{k: v for k, v in r.items() if k != 'prefix_token_ids'} for r in source]
    picked = select_states(rows, source, 3)
    assert [(r['prompt_id'], r['cycle']) for r in picked] == [('1', 0), ('2', 0), ('3', 0)]
    source[2]['draft_ids'] = [99] * 15
    with pytest.raises(ValueError, match='changed'):
        select_states(rows, source, 3)


def test_latency_aggregation_uses_total_tokens_not_mean_of_ratios():
    from scripts.summarize_midverify_latency import aggregate
    cells = []
    for case in ('fixed16', 'fixed8_redraft', 'target_free'):
        for elapsed, committed in ((10., 10), (20., 30)):
            cells.append({'C': 4, 'mode': 'graph', 'case': case, 'rows': 4,
                'observations': [{'committed_tokens': committed}],
                'uninstrumented_cycle_ms': {'mean': elapsed, 'stdev': .1},
                'mean_front': 8., 'mean_end': 8., 'phase_stream_ms': {},
                'anchor_mismatches': 0, 'audit': {'top1_differences': 0,
                    'acceptance_differences': 0, 'bonus_differences': 0,
                    'hidden': {'relative_l2': 0.}}})
    r = aggregate(cells)['c4_graph_target_free']
    assert r['ms_per_committed_token'] == .75
    assert r['retention_vs_same_engine_b16'] == 1
    assert r['speed_ratio_vs_fixed16'] == 1


def test_identical_conditional_latency_and_retention_intervals():
    from scripts.summarize_midverify_latency import conditional_intervals
    cells = [{'C': 4, 'mode': 'graph', 'case': case, 'offset': 0,
              'observations': [{'accepted': [1, 2, 3, 15]}],
              'uninstrumented_cycle_samples_ms': [1., 2., 3.]}
             for case in ('fixed16', 'target_free', 'cascade')]
    r = conditional_intervals(cells, draws=100)['c4_graph']
    np.testing.assert_array_equal(r['retention_delta_ci95'], [0., 0.])
    np.testing.assert_array_equal(r['replay_speed_ratio_vs_target_free_ci95'], [1., 1.])
