import copy

import numpy as np
import pytest

from scripts.collect_midverify_probe import seed_mapping, array_row_digest


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
