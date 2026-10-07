import numpy as np
import pytest
import torch

from scripts.train_postdraft_signal import (
    ARMS, auc_parts, decisions, loss_fn, make_model, policy_scores,
    raw_logits, select_policy,
)
from scripts.train_midverify_scaling import parameter_sha


def features():
    return (torch.randn(4, 15, 3), torch.randn(4, 15, 8),
            torch.randn(4, 15, 8), torch.randint(30, (4, 15)))


def test_residual_starts_exactly_at_frozen_confidence():
    x = features()
    base = make_model("confidence", width=8, vocab=30).eval()
    for arm in ("residual_local", "residual_prefix"):
        model = make_model(arm, width=8, vocab=30, base=base).eval()
        torch.testing.assert_close(model(*x), base(*x), rtol=0, atol=0)
        assert all(not p.requires_grad for p in model.base.parameters())


@pytest.mark.parametrize("arms", [("dspark_form_hard", "dspark_form_tv"), ("residual_local", "residual_prefix")])
def test_paired_arms_identical_parameters(arms):
    base = make_model("confidence", width=8, vocab=30)
    hashes = []
    for arm in arms:
        torch.manual_seed(913)
        model = make_model(arm, width=8, vocab=30, base=base)
        hashes.append(parameter_sha(model.state_dict()))
    assert hashes[0] == hashes[1]


def test_prefix_head_never_reads_future_candidate_inputs():
    x = features()
    model = make_model("residual_prefix", width=8, vocab=30,
                       base=make_model("confidence", width=8, vocab=30)).eval()
    torch.nn.init.normal_(model.net[-1].weight)
    original = model(*x)
    changed = tuple(v.clone() for v in x)
    for v in changed[:3]:
        v[:, 7:] += 100
    torch.testing.assert_close(model(*changed)[:, :7], original[:, :7], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_loss_does_not_supervise_off_path_suffix(arm):
    logits = torch.zeros(2, 15, requires_grad=True)
    risk = torch.zeros_like(logits)
    risk[:, :4] = 1
    hard, tv = torch.ones_like(logits), torch.full_like(logits, .2)
    loss_fn(logits, hard, tv, risk, arm).backward()
    assert torch.count_nonzero(logits.grad[:, 4:]) == 0
    assert (logits.grad[:, :4] > 0).all() if arm == "dspark_form_tv" else (logits.grad[:, :4] < 0).all()


def test_survival_scores_are_monotone_and_match_product():
    q = np.linspace(.1, .99, 60).reshape(4, 15)
    logits = raw_logits(np.log(q))
    survival = policy_scores(logits, "survival")
    np.testing.assert_allclose(np.exp(survival), np.cumprod(q, axis=1))
    assert (np.diff(survival, axis=1) <= 0).all()


def test_calibration_is_feasible_and_uses_no_assessment():
    logits = np.random.default_rng(913).normal(size=(60, 15))
    accepted = np.arange(60) % 16
    settings = select_policy(logits, accepted, (.9, .96, 1.))
    for t, s in settings.items():
        k = decisions(logits, s)
        assert np.minimum(accepted, k-1).sum()/accepted.sum() >= float(t)-1e-12
        assert ((k >= 1) & (k <= 16)).all()


def test_auc_ties_and_perfect_ranking():
    assert auc_parts([0, 1, 2, 3], [0, 0, 1, 1]) == (4., 4)
    assert auc_parts([1, 1, 1, 1], [0, 0, 1, 1]) == (2., 4)
    assert auc_parts([1, 2], [0, 0]) == (0., 0)


def test_all_heads_accept_variable_block_length():
    x = tuple(v[:, :7] for v in features())
    base = make_model("confidence", width=8, vocab=30)
    for arm in ARMS:
        assert make_model(arm, width=8, vocab=30, base=base)(*x).shape == (4, 7)
