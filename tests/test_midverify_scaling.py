import copy

import numpy as np
import pytest

from scripts.collect_midverify_probe import seed_mapping, array_row_digest
from scripts.train_midverify_scaling import batch_stream, calibration_bce, paired_intervals
from scripts.finalize_midverify_nonterminal import nonterminal_selection


def sample_row(group, pid, cycle, row):
    return {"group": group, "prompt_id": str(pid), "cycle": cycle, "row": row,
            "prefix_sha256": f"{pid}:{cycle}", "draft_ids": [1, 2, 3]}


def test_seed_mapping_preserves_validation_after_training_append():
    base = [sample_row("train", 1, 0, 0), sample_row("calibration", 2, 0, 1), sample_row("assessment", 3, 0, 2)]
    grown = [base[0], sample_row("train", 4, 0, 1), dict(base[1], row=2), dict(base[2], row=3)]
    np.testing.assert_array_equal(seed_mapping(base, grown), [0, 2, 3])
    src = np.arange(3 * 15 * 4).reshape(3, 15, 4).astype(np.float16)
    dst = np.zeros((4, 15, 4), dtype=np.float16)
    dst[[0, 2, 3]] = src
    assert array_row_digest(src, np.arange(3)) == array_row_digest(dst, np.array([0, 2, 3]))
    dst[2, 1, 2] += 1
    assert array_row_digest(src, np.arange(3)) != array_row_digest(dst, np.array([0, 2, 3]))


def test_seed_rejects_changed_candidates_or_validation_membership():
    base = [sample_row("train", 1, 0, 0), sample_row("calibration", 2, 0, 1), sample_row("assessment", 3, 0, 2)]
    changed = copy.deepcopy(base)
    changed[0]["draft_ids"][0] = 99
    with pytest.raises(ValueError, match="contents changed"):
        seed_mapping(base, changed)
    with pytest.raises(ValueError, match="membership changed"):
        seed_mapping(base, base + [sample_row("assessment", 5, 0, 3)])
    with pytest.raises(ValueError, match="Duplicate"):
        seed_mapping(base, base + [base[0]])


def test_seed_rejects_non_nested_training_order():
    base = [sample_row("train", 1, 0, 0), sample_row("train", 1, 1, 1)]
    with pytest.raises(ValueError, match="not nested"):
        seed_mapping(base, list(reversed(base)))


def test_full_batch_stream_preserves_epochs_and_rng():
    import torch
    torch.manual_seed(31)
    rng_before = torch.get_rng_state().clone()
    stream = batch_stream(5, 3, 91)
    batches = [next(stream) for _ in range(5)]
    assert all(len(batch) == 3 for batch in batches)
    assert torch.equal(rng_before, torch.get_rng_state())
    sequence = torch.cat(batches).numpy()
    for start in range(0, 15, 5):
        np.testing.assert_array_equal(np.sort(sequence[start:start + 5]), np.arange(5))
    repeat = batch_stream(5, 3, 91)
    for batch in batches:
        assert torch.equal(batch, next(repeat))


def test_stream_supports_batch_larger_than_dataset_and_rejects_zero():
    stream = batch_stream(2, 9, 4)
    assert len(next(stream)) == 9
    with pytest.raises(ValueError, match="Positive"):
        next(batch_stream(0, 4, 1))


def test_paired_identical_policies_have_zero_deltas():
    result = paired_intervals(np.array([2, 4, 7, 9]), np.array([3, 4, 6, 9]),
                              np.array([3, 4, 6, 9]), np.array(["a", "a", "b", "b"]), 6)
    for key, value in result.items():
        if key.endswith("_ci95"):
            np.testing.assert_array_equal(value, [0., 0.])


def test_calibration_bce_ignores_unobserved_suffix():
    labels = np.array([[1, 0, 0]])
    mask = np.array([[1, 1, 0]])
    loss = calibration_bce(np.array([[.8, .2, .001]]), labels, mask)
    assert loss == pytest.approx(-np.log(.8))
    assert loss == calibration_bce(np.array([[.8, .2, .999]]), labels, mask)


def test_terminal_filter_only_new_unprotected_training():
    rows = [{"group": "train", "draft_ids": [1, 9], "accepted_len": 1, "replayed_accepted_eos": False},
            {"group": "train", "draft_ids": [1, 9], "accepted_len": 2, "replayed_accepted_eos": True}]
    keep, excluded = nonterminal_selection(rows, {9}, [0], protected_train_prefix=1)
    np.testing.assert_array_equal(keep, [0])
    assert excluded == [1]
    for group in ("calibration", "assessment"):
        changed = copy.deepcopy(rows)
        changed[1]["group"] = group
        with pytest.raises(ValueError, match="Cannot exclude"):
            nonterminal_selection(changed, {9}, [], protected_train_prefix=0)
    with pytest.raises(ValueError, match="Cannot exclude"):
        nonterminal_selection(rows, {9}, [1], protected_train_prefix=0)
    with pytest.raises(ValueError, match="Cannot exclude"):
        nonterminal_selection(rows, {9}, [], protected_train_prefix=2)
