import json

import numpy as np
import pytest
import torch

from scripts.train_predraft_soft_supervision import ARMS, build_model, objective, policy_scores, predraft_inputs
from scripts.train_soft_supervision import load_cache
from scripts.train_midverify_scaling import parameter_sha, batch_stream
from scripts.audit_block_headroom import sha256
from dflash.midverify import risk_mask, calibrate, apply_setting
from test_soft_supervision import make_cache


@pytest.mark.parametrize("arm", ARMS)
def test_offpath_suffix_has_no_loss_or_gradient(arm):
    output = torch.randn(2, 3, 15, requires_grad=True)
    hard = torch.zeros(2, 15)
    hard[:, :2] = 1
    mask = torch.from_numpy(risk_mask(np.array([2, 2]))).float()
    tv, target = torch.full_like(hard, .7), torch.full_like(hard, .6)
    loss = objective(output, hard, tv, target, mask, arm)
    loss.backward()
    assert torch.count_nonzero(output.grad[:, :, 3:]) == 0
    changed = output.detach().clone()
    changed[:, :, 3:] = 99
    torch.testing.assert_close(objective(changed, hard, tv, target, mask, arm), loss.detach())


@pytest.mark.parametrize("arm,head", [("soft_tv", 1), ("soft_target", 2)])
def test_soft_only_replaces_hard_loss(arm, head):
    output = torch.zeros(2, 3, 15, requires_grad=True)
    hard, mask = torch.zeros(2, 15), torch.ones(2, 15)
    tv, target = torch.full_like(hard, .7), torch.full_like(hard, .8)
    loss = objective(output, hard, tv, target, mask, arm)
    torch.testing.assert_close(loss, objective(output, 1-hard, tv, target, mask, arm))
    loss.backward()
    assert torch.count_nonzero(output.grad[:, 0]) == 0
    assert torch.count_nonzero(output.grad[:, head]) > 0
    # Policy uses the trained soft head, never the untrained hard head.
    changed = output.detach().clone()
    changed[:, 0] += 100
    torch.testing.assert_close(policy_scores(output, arm), policy_scores(changed, arm))


def test_inference_allowlist_excludes_all_current_draft_and_target_fields():
    fused = np.ones((4, 2560), np.float16)
    base = predraft_inputs({"fused": fused})
    polluted = {"fused": fused, "candidate_vectors": None, "draft_stats": np.nan,
                "target_candidate_prob": np.inf, "matches": None, "accepted_len": -1}
    np.testing.assert_array_equal(base, predraft_inputs(polluted))


def test_all_accepted_cap_does_not_invent_a_failure():
    output = torch.zeros(1, 3, 15, requires_grad=True)
    hard = torch.ones(1, 15)
    mask = torch.from_numpy(risk_mask(np.array([15]))).float()
    objective(output, hard, hard, hard, mask, "hard").backward()
    assert (output.grad[:, 0] < 0).all()
    assert torch.count_nonzero(output.grad[:, 1:]) == 0


@pytest.mark.parametrize("arm,head", [("hard_tv", 1), ("hard_target", 2)])
def test_joint_heads_keep_hard_supervision_and_decision(arm, head):
    output = torch.zeros(1, 3, 15, requires_grad=True)
    hard, soft, mask = torch.ones(1, 15), torch.full((1, 15), .7), torch.ones(1, 15)
    base = torch.autograd.grad(objective(output, hard, soft, soft, mask, "hard"), output)[0]
    joint = torch.autograd.grad(objective(output, hard, soft, soft, mask, arm), output)[0]
    torch.testing.assert_close(base[:, 0], joint[:, 0])
    assert torch.count_nonzero(joint[:, head]) > 0
    torch.testing.assert_close(policy_scores(output, arm), policy_scores(output, "hard"))


def test_matched_initialization_minibatches_dropout_and_parameter_count():
    states, batches, outputs = [], [], []
    for _ in ARMS:
        torch.manual_seed(913)
        model = build_model(3)
        states.append(parameter_sha(model.state_dict()))
        batches.append(next(batch_stream(19, 8, 913)))
        outputs.append(model(torch.ones(8, 3)))
    assert len(set(states)) == 1
    for b, o in zip(batches[1:], outputs[1:]):
        torch.testing.assert_close(b, batches[0])
        torch.testing.assert_close(o, outputs[0])


@pytest.mark.parametrize("arm", ARMS)
def test_surrogate_scores_monotone_and_calibration_feasible(arm):
    scores = policy_scores(torch.randn(4, 3, 15)*10, arm).detach().numpy()
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores, axis=1) <= 0)
    a = np.array([0, 3, 9, 15])
    setting = calibrate(scores, a, (.96,))["0.96"]
    k = apply_setting(scores, setting)
    assert np.minimum(a, k-1).sum()/a.sum() >= .96


def make_predraft_cache(root):
    make_cache(root)
    path = root/"batch_0000.npz"
    with np.load(path) as source:
        arrays = {k: source[k] for k in source.files if k != "candidate_vectors"}
    arrays.update(fused=np.ones((24, 2560), np.float16), target_candidate_prob=np.full((24, 15), .6))
    np.savez(path, **arrays)
    receipts = json.loads((root/"receipts.json").read_text())
    receipts[0]["sha256"] = sha256(path)
    (root/"receipts.json").write_text(json.dumps(receipts))
    complete = json.loads((root/"COMPLETE.json").read_text())
    complete["binding"]["receipts.json"] = sha256(root/"receipts.json")
    (root/"COMPLETE.json").write_text(json.dumps(complete))


def test_predraft_cache_view_does_not_require_candidate_vectors(tmp_path):
    make_predraft_cache(tmp_path)
    _, rows, arrays = load_cache(tmp_path, view="predraft")
    assert len(rows) == 24
    assert "candidate_vectors" not in arrays
    assert predraft_inputs(arrays).shape == (24, 2560)


def test_predraft_cache_rejects_corruption(tmp_path):
    make_predraft_cache(tmp_path)
    (tmp_path/"batch_0000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_cache(tmp_path, view="predraft")
