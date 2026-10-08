import ast
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_history_serving import prepare


def test_restore_parses(tmp_path):
    dst = tmp_path/'source'
    manifest = prepare(dst)
    for rel in manifest['history_extension']['changes']:
        ast.parse((dst/'python/sglang/srt'/rel).read_text())
    worker = (dst/'python/sglang/srt/speculative/dflash_worker_v2.py').read_text()
    assert worker.count('self.history_policy.observe(') == 2
    assert worker.count('self.history_policy.reset(') == 1


@pytest.fixture
def modules(tmp_path):
    torch = pytest.importorskip('torch')
    import shutil
    pkg = tmp_path/'policy_test_package'
    pkg.mkdir()
    (pkg/'__init__.py').touch()
    for a,b in [('history_policy.py','dflash_history_table.py'),
                ('serving_history_policy.py','policy.py'),
                ('serving_history_entropy.py','dflash_history_entropy.py')]:
        shutil.copy2(ROOT/'dflash'/a, pkg/b)
    sys.path.insert(0, str(tmp_path))
    for name in list(sys.modules):
        if name.startswith('policy_test_package'):
            del sys.modules[name]
    yield (torch, importlib.import_module('policy_test_package.policy'),
           importlib.import_module('policy_test_package.dflash_history_table'))
    sys.path.remove(str(tmp_path))


def table_fixture(table_mod):
    return table_mod.HistoryValueTable(dict(schema_version=1, blocks=[4,8,12,16],
        certainty_edges=[-2., -1.], global_progress=[2.,3.,4.,5.],
        values={'1:1:4:1':[3.,3.,3.,3.], '2:2:8:0':[1.,4.,4.,4.]}, fit_prompt_ids=[]))


def test_lookup_matches_cpu_all_bins_and_cold(modules):
    torch, mod, hm = modules
    table = table_fixture(hm)
    costs = {4:1.,8:2.,12:3.,16:4.}
    policy = mod.DFlashHistoryPolicy(table,costs,.3,capacity=16,device='cpu')
    assert policy.select(torch.tensor([0])).item() == hm.select_block(table,hm.History().snapshot(),mode='priced',costs_ms=costs,rho=.3)[0]
    for n in (1,2):
        for certainty in (-3., -2., -1.5, -1., -.5):
            for b in table.blocks:
                for full in (False,True):
                    policy.count[0]=n
                    policy.entropies[0]=torch.tensor([0., -certainty*n],dtype=torch.float64)
                    policy.previous_block[0]=b
                    policy.previous_full[0]=int(full)
                    h=dict(entropy_count=n,certainty=certainty,previous_block=b,previous_full=full)
                    assert policy.select(torch.tensor([0])).item() == hm.select_block(table,h,mode='priced',costs_ms=costs,rho=.3)[0]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_valid_prefix_entropy_and_slot_reuse(modules,device):
    torch, mod, hm = modules
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('requires GPU')
    table=table_fixture(hm)
    policy=mod.DFlashHistoryPolicy(table,{4:1.,8:2.,12:3.,16:4.},.3,capacity=10,device=device)
    idx=torch.tensor([5,2],device=device)
    lens=torch.tensor([4,8],device=device)
    accepted=torch.tensor([0,6],device=device)
    offsets=torch.tensor([0,4,12],device=device)
    logits=torch.randn(16,10003,device=device,dtype=torch.bfloat16)*4
    real=torch.tensor([0,1,2,3,8,9,10,11,12,13,14,15],device=device)
    lp=logits.float().log_softmax(-1)
    ent=-(lp.exp()*lp).sum(-1)
    torch.testing.assert_close(mod.row_entropy(logits),ent,rtol=2e-5,atol=2e-5)
    policy.observe(idx,lens,accepted,logits,offsets,real)
    expected=torch.stack([ent[0],ent[8:15].mean()]).double()
    torch.testing.assert_close(policy.entropies[idx,1],expected,rtol=2e-5,atol=2e-5)
    policy.observe(idx,lens,accepted,logits,offsets,real)
    assert policy.count[idx].tolist()==[2,2]
    policy.reset(idx[:1])
    assert policy.count[idx].tolist()==[0,2]
    assert policy.entropies[5].tolist()==[0.,0.]


def test_frozen_bindings_fail_closed(modules,tmp_path):
    torch,mod,hm=modules
    table=tmp_path/'table.json'; profile=tmp_path/'profile.json'; frozen=tmp_path/'frozen.json'
    table.write_text('{}');profile.write_text('{}');frozen.write_text(json.dumps(dict(schema_version=1,bindings={})))
    with pytest.raises(ValueError,match='binding mismatch'):
        mod.load_artifacts(table,profile,frozen)


def test_diagnostic_checks_emitted_tokens(modules):
    torch,mod,hm=modules
    policy=mod.DFlashHistoryPolicy(table_fixture(hm),{4:1.,8:2.,12:3.,16:4.},.3,capacity=3,device='cpu')
    logits=torch.eye(4)*10
    # draft proposal 0 accepted, 3 rejected against target 1; emit [0,1].
    args=(torch.tensor([4]),torch.tensor([1]),logits,torch.tensor([0,4]),None,
          torch.tensor([2,0,3,0]))
    policy.check_emission(*args,torch.tensor([0,1]),True)
    with pytest.raises(AssertionError,match='Emitted'):
        policy.check_emission(*args,torch.tensor([0,2]),True)
