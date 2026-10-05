import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_soft_supervision import ARMS, build_model, objective, bootstrap, load_cache
from scripts.audit_block_headroom import sha256
from scripts.train_midverify_scaling import parameter_sha, batch_stream
from dflash.midverify import accepted_lengths, risk_mask, calibrate, apply_setting


@pytest.mark.parametrize("arm", ARMS)
def test_no_loss_or_gradient_after_first_rejection(arm):
    output=torch.randn(2,15,3,requires_grad=True)
    hard=torch.zeros(2,15)
    hard[:,:2]=1
    mask=torch.from_numpy(risk_mask(np.array([2,2]))).float()
    tv=torch.full((2,15),.7)
    margin=torch.zeros(2,15)
    loss=objective(output,hard,tv,margin,mask,arm)
    loss.backward()
    assert torch.count_nonzero(output.grad[:,3:])==0
    changed=output.detach().clone()
    changed[:,3:]=99
    torch.testing.assert_close(objective(changed,hard,tv,margin,mask,arm),loss.detach())


def test_auxiliary_arms_do_not_replace_hard_primary_loss():
    o=torch.zeros(1,15,3,requires_grad=True)
    hard=torch.ones(1,15)
    args=(hard,torch.full_like(hard,.3),torch.ones_like(hard),torch.ones_like(hard))
    gradients=[]
    for arm in ARMS:
        gradients.append(torch.autograd.grad(objective(o,*args,arm),o,retain_graph=True)[0])
    for g in gradients[1:]:
        torch.testing.assert_close(g[...,0],gradients[0][...,0])
    assert torch.count_nonzero(gradients[0][...,1:])==0
    assert torch.count_nonzero(gradients[1][...,1])>0
    assert torch.count_nonzero(gradients[2][...,2])>0


def test_arms_start_with_identical_weights_batches_and_dropout():
    weights,batches,outputs=[],[],[]
    for _ in ARMS:
        torch.manual_seed(913)
        model=build_model(3)
        weights.append(parameter_sha(model.state_dict()))
        stream=batch_stream(19,8,913)
        batches.append(next(stream))
        outputs.append(model(torch.ones(8,15,3)))
    assert len(set(weights))==1
    for i in (1,2):
        torch.testing.assert_close(batches[0],batches[i])
        torch.testing.assert_close(outputs[0],outputs[i])


def test_identical_policies_have_zero_paired_bootstrap_delta():
    a=np.array([1,4,9,15])
    k=np.array([2,4,8,16])
    out=bootstrap(a,k,k,np.array([1,1,2,3]),100)
    for name in ("retention_delta_vs_hard_ci95","mean_budget_delta_vs_hard_ci95","ratio_delta_vs_hard_ci95"):
        np.testing.assert_array_equal(out[name],[0.,0.])


def make_cache(root):
    rows=[{"row":i,"group":g,"prompt_id":j+1} for j,g in enumerate(("train","calibration","assessment"))
          for i in range(j*8,(j+1)*8)]
    matches=np.zeros((24,15),dtype=bool)
    matches[:,:3]=True
    arrays={"matches":matches,"accepted_len":accepted_lengths(matches),"at_risk":risk_mask(accepted_lengths(matches)),
            "tv_overlap":np.full((24,15),.7,dtype=np.float32),"target_margin":np.zeros((24,15),np.float32),
            "draft_stats":np.zeros((24,15,3),np.float32),"candidate_vectors":np.ones((24,15,2560),np.float16),
            "eligible":np.ones(24,bool),"anchor_match":np.ones(24,bool),"accepted_eos":np.zeros(24,bool)}
    np.savez(root/"batch_0000.npz",**arrays)
    (root/"config.json").write_text(json.dumps({"schema":"soft_supervision_v1","smoke":True,"rows":rows}))
    (root/"receipts.json").write_text(json.dumps([{"file":"batch_0000.npz","sha256":sha256(root/"batch_0000.npz"),
         "rows":list(range(24)),"summary":{"canonical_disagreements":0}}]))
    (root/"summary.json").write_text("{}")
    (root/"COMPLETE.json").write_text(json.dumps({"complete":True,"binding":{name:sha256(root/name)
           for name in ("config.json","receipts.json","summary.json")}}))


def test_complete_cache_loads_and_preserves_split(tmp_path):
    make_cache(tmp_path)
    config,rows,arrays=load_cache(tmp_path)
    assert len(rows)==24
    assert arrays["eligible"].all()
    assert config["smoke"]


def test_modified_shard_rejected(tmp_path):
    make_cache(tmp_path)
    (tmp_path/"batch_0000.npz").write_bytes(b"bad")
    with pytest.raises(ValueError,match="checksum"):
        load_cache(tmp_path)


def test_calibration_settings_do_not_require_assessment_labels():
    scores=np.tile(np.arange(15,0,-1),(4,1)).astype(float)
    accepted=np.array([3,5,10,15])
    settings=calibrate(scores,accepted,(.96,))
    k=apply_setting(scores,settings["0.96"])
    assert np.minimum(accepted,k-1).sum()/accepted.sum()>=.96
