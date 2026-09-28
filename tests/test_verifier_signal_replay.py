import numpy as np
from scripts.collect_verifier_signal_replay import digest_bytes, history_positions


def test_prefill_has_only_one_available_logit():
    positions, valid = history_positions(100, 100)
    assert positions.tolist() == list(range(84, 100))
    assert valid.sum() == 1
    assert positions[valid].tolist() == [99]


def test_history_stops_before_current_anchor_and_masks_prompt_interior():
    positions, valid = history_positions(107, 100)
    assert positions.max() == 106
    assert positions[valid].tolist() == list(range(99, 107))
    assert (positions + 1).max() == 107  # Last distribution predicts known anchor.


def test_short_prefix_is_not_falsely_padded_with_observations():
    positions, valid = history_positions(2, 2)
    assert positions.tolist() == [0, 1]
    assert valid.tolist() == [False, True]


def test_hash_includes_the_known_anchor():
    assert digest_bytes(np.array([1, 2, 3], dtype=np.int64)) != digest_bytes(np.array([1, 2, 4], dtype=np.int64))
