import numpy as np
import pytest

from scripts.analyze_acceptance_geometry import (
    demean_by_prompt, history_features, prompt_balanced_indices,
    recent_tokens, variance_partition,
)


def test_recent_window_never_uses_anchor_or_future():
    trajectory = np.arange(100)
    assert recent_tokens(trajectory, 20, 4).tolist() == [16, 17, 18, 19]
    assert recent_tokens(trajectory, 2, 8).tolist() == [0, 1]
    with pytest.raises(ValueError):
        recent_tokens(trajectory, 100, 4)


def test_history_is_causal_even_if_future_labels_change():
    cycles = np.arange(6)
    first = history_features(np.array([1, 2, 3, 4, 5, 6]), cycles)
    second = history_features(np.array([1, 2, 13, 14, 15, 6]), cycles)
    np.testing.assert_array_equal(first[:3], second[:3])
    assert first[2, 0] == 2
    assert first[2, 1] == 1.5
    assert first[0, 4] == 0


def test_history_gap_is_not_treated_as_previous_cycle():
    result = history_features(np.array([3, 8, 1]), np.array([0, 2, 3]))
    assert result[1, 0] == 0
    assert result[1, 3] == 0
    assert result[2, 0] == 8
    assert result[2, 3] == 1


def test_prompt_balanced_sampling_ignores_labels_and_is_reproducible():
    prompts = np.array([1] * 20 + [2] * 3 + [3] * 8)
    a = prompt_balanced_indices(prompts, 4, 3)
    np.testing.assert_array_equal(a, prompt_balanced_indices(prompts, 4, 3))
    assert len(a) == 11
    assert all((prompts[a] == p).sum() <= 4 for p in set(prompts))


def test_variance_partition_and_demeaning():
    y, p = np.array([0., 2., 10., 12.]), np.array([1, 1, 2, 2])
    np.testing.assert_array_equal(demean_by_prompt(y, p), [-1, 1, -1, 1])
    result = variance_partition(y, p)
    assert result['within_prompt_fraction'] == pytest.approx(1 / 26)
    assert result['within_prompt_fraction'] + result['between_prompt_fraction'] == pytest.approx(1)
