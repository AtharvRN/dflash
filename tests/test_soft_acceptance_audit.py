import numpy as np
import pytest
import torch

import json
from scripts.audit_soft_acceptance import label_statistics, summarize, select_training_rows
from scripts.collect_rejected_trace import MODEL_REVISIONS, prefix_hash
from scripts.audit_block_headroom import sha256


def test_tv_can_be_high_while_greedy_rejects():
    q = torch.tensor([[[.51, .49]]])
    p = torch.tensor([[[.49, .51]]])
    out = label_statistics(q.log(), p.log(), torch.tensor([[0]]))
    assert out["tv_overlap"].item() == pytest.approx(.98)
    assert not out["matches"].item()
    assert out["target_margin"].item() < 0
    assert out["target_candidate_prob"].item() == pytest.approx(.49)


def test_suffix_mask_includes_first_rejection_only():
    target = torch.tensor([[[2., 0.], [0., 2.], [2., 0.]]])
    out = label_statistics(target, target, torch.tensor([[0, 0, 0]]))
    assert out["matches"].tolist() == [[True, False, True]]
    assert out["survival"].tolist() == [[1, 0, 0]]
    assert out["at_risk"].tolist() == [[True, True, False]]
    assert out["accepted_len"].item() == 1
    torch.testing.assert_close(out["tv_overlap"], torch.ones(1, 3))


def test_ties_are_not_falsely_assigned_a_positive_margin():
    z = torch.zeros(1, 2, 3)
    out = label_statistics(z, z, torch.tensor([[0, 1]]))
    assert out["ties"].all()
    assert out["matches"].tolist() == [[True, False]]
    assert (out["target_margin"] == 0).all()


def test_tv_matches_rejection_sampling_expectation():
    torch.manual_seed(13)
    qz, pz = torch.randn(3, 15, 17), torch.randn(3, 15, 17)
    q, p = qz.softmax(-1), pz.softmax(-1)
    out = label_statistics(qz, pz, qz.argmax(-1))
    expected = (q * (p/q).clamp(max=1)).sum(-1)
    torch.testing.assert_close(out["tv_overlap"], expected)
    assert torch.all(out["target_margin"][out["matches"]] > 0)


def test_shapes_and_nonfinite_fail_closed():
    with pytest.raises(ValueError, match="shape"):
        label_statistics(torch.zeros(1, 2, 4), torch.zeros(1, 3, 4), torch.zeros(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="Nonfinite"):
        label_statistics(torch.full((1, 2, 4), float("nan")), torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.long))


def test_summary_all_accepted_has_empty_offpath_not_nan():
    z = torch.tensor([[[2., 0.], [2., 0.]]])
    out = {k: v.numpy() for k, v in label_statistics(z, z, z.argmax(-1)).items()}
    out.update(anchor_match=np.array([True]), accepted_eos=np.array([False]), candidate_match=np.ones((1, 2), bool))
    summary = summarize(out, [{"prompt_id": 1, "old_accepted_len": 2}])
    assert summary["off_path_suffix"]["positions"] == 0
    assert summary["off_path_suffix"]["target_margin"]["mean"] is None


def make_source(root):
    def save(name, data):
        (root / name).write_text(json.dumps(data))
    row = {"prompt_id": "1", "group": "train", "cycle": 0, "source": "test", "prefix_length": 2,
           "prefix_token_ids": [2, 3, 4], "prefix_sha256": prefix_hash([2, 3, 4]), "eligible": True,
           "outcomes": {"16": {"draft_ids": list(range(15)), "accepted": 2}}}
    save("prompt_1.json", {"states": [row]})
    save("config.json", {"canonical_train_prompt_ids": [1], "canonical_val_prompt_ids": [2],
                         "models": {name: {"revision": rev, "path": "/models/"+rev}
                                    for name, rev in MODEL_REVISIONS.items()}})
    save("receipts.json", [{"prompt_id": 1, "group": "train", "files": {"prompt_1.json": sha256(root / "prompt_1.json")}},
                            {"prompt_id": 2, "group": "assessment", "files": {}}])
    save("collection_summary.json", {})
    save("COMPLETE.json", {"sample_complete": True, "binding": {name: sha256(root / name)
          for name in ("config.json", "receipts.json", "collection_summary.json")}})


def test_selection_does_not_access_validation_shards(tmp_path):
    make_source(tmp_path)
    rows, binding = select_training_rows(tmp_path, 1)
    assert rows[0]["group"] == "train"
    assert rows[0]["old_accepted_len"] == 2
    assert binding["source_pool_rows"] == 1
    assert select_training_rows(tmp_path, 1)[0] == rows


def test_selection_detects_modified_source_shard(tmp_path):
    make_source(tmp_path)
    (tmp_path / "prompt_1.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        select_training_rows(tmp_path, 1)


def test_selection_requires_completion_and_enough_rows(tmp_path):
    make_source(tmp_path)
    with pytest.raises(ValueError, match="Insufficient"):
        select_training_rows(tmp_path, 2)
    (tmp_path / "COMPLETE.json").write_text('{"sample_complete": false}')
    with pytest.raises(ValueError, match="complete"):
        select_training_rows(tmp_path, 1)
