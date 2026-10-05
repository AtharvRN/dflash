import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts.train_predraft_soft_supervision import (
    BCE_ARMS, SOFT_WEIGHT, MARGIN_TEMPERATURE, same_head_targets,
    same_head_objective, policy_scores, objective, prefix_calibration,
)
from dflash.midverify import risk_mask


@pytest.mark.parametrize("arm", BCE_ARMS)
def test_only_primary_output_receives_gradient_and_offpath_is_masked(arm):
    output = torch.randn(2, 3, 15, requires_grad=True)
    hard = torch.zeros(2, 15)
    hard[:, :3] = 1
    tv, margin = torch.full_like(hard, .8), torch.full_like(hard, 2.)
    mask = torch.from_numpy(risk_mask(np.array([3, 3]))).float()
    loss = same_head_objective(output, hard, tv, margin, mask, arm)
    loss.backward()
    assert torch.count_nonzero(output.grad[:, 0, :4]) > 0
    assert torch.count_nonzero(output.grad[:, 0, 4:]) == 0
    assert torch.count_nonzero(output.grad[:, 1:]) == 0
    changed = output.detach().clone()
    changed[:, 1:] = 99
    torch.testing.assert_close(policy_scores(output, arm), policy_scores(changed, arm))


@pytest.mark.parametrize("arm", ["mixed_tv_bce", "mixed_margin_bce"])
def test_blended_target_bce_equals_weighted_hard_and_soft_losses(arm):
    z = torch.randn(2, 3, 15)
    hard = torch.randint(0, 2, (2, 15)).float()
    tv, margin, mask = torch.rand(2, 15), torch.randn(2, 15), torch.ones(2, 15)
    soft = tv if arm == "mixed_tv_bce" else (margin/MARGIN_TEMPERATURE).sigmoid()
    expected = ((1-SOFT_WEIGHT)*F.binary_cross_entropy_with_logits(z[:, 0], hard)
                + SOFT_WEIGHT*F.binary_cross_entropy_with_logits(z[:, 0], soft))
    torch.testing.assert_close(same_head_objective(z, hard, tv, margin, mask, arm), expected)


def test_pure_tv_bce_does_not_use_hard_labels_with_fixed_risk_mask():
    z = torch.randn(2, 3, 15)
    hard, tv, margin, mask = torch.zeros(2, 15), torch.rand(2, 15), torch.randn(2, 15), torch.ones(2, 15)
    first = same_head_objective(z, hard, tv, margin, mask, "soft_tv_bce")
    second = same_head_objective(z, 1-hard, tv, margin*20, mask, "soft_tv_bce")
    torch.testing.assert_close(first, second)


def test_hard_control_is_identical_to_previous_objective():
    z = torch.randn(2, 3, 15, requires_grad=True)
    hard, tv, margin = torch.randint(0, 2, (2, 15)).float(), torch.rand(2, 15), torch.randn(2, 15)
    mask = torch.from_numpy(risk_mask(np.array([0, 15]))).float()
    previous = objective(z, hard, tv, margin, mask, "hard")
    current = same_head_objective(z, hard, tv, margin, mask, "hard")
    torch.testing.assert_close(current, previous, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(current, z, retain_graph=True)[0],
                               torch.autograd.grad(previous, z)[0], rtol=0, atol=0)


def test_margin_mix_preserves_hard_side_and_tie_break():
    hard = torch.tensor([[0., 0., 1., 1., 0., 1.]])
    margin = torch.tensor([[-100., -.01, .01, 100., 0., 0.]])
    labels = same_head_targets(hard, torch.ones_like(hard), margin, "mixed_margin_bce")
    assert (labels[hard == 0] < .5).all()
    assert (labels[hard == 1] > .5).all()
    assert labels[0, 4].item() == pytest.approx(.125)
    assert labels[0, 5].item() == pytest.approx(.875)


def test_prefix_calibration_uses_known_zero_survival_after_rejection():
    a = np.array([0, 3, 15])
    truth = a[:, None] >= np.arange(1, 16)
    scores = np.log(np.where(truth, 1.-1e-9, 1e-9))
    result = prefix_calibration(scores, a)
    assert result["mean_brier"] < 1e-16
    assert result["surrogate_expected_A_mae"] < 1e-6
    assert len(result["positions"]) == 15
    # At position 4, only the all-accepted cycle has nonzero survival.
    assert sum(v["count"]*v["observed"] for v in result["positions"][3]["reliability"]) == 1


def test_prefix_calibration_detects_overconfidence():
    result = prefix_calibration(np.full((2, 15), np.log(.9)), np.array([0, 0]))
    assert result["mean_brier"] == pytest.approx(.81)
    assert result["mean_position_ece"] == pytest.approx(.9)


def test_invalid_arm_fails():
    with pytest.raises(ValueError, match="Unknown"):
        same_head_targets(torch.ones(1), torch.ones(1), torch.ones(1), "unknown")
