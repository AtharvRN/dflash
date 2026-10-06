import copy
from pathlib import Path
import pytest
from dflash.postdraft_latency import settings


def test_settings_use_only_frozen_calibration_and_seed913():
    point={'calibration':{'threshold':-.8,'retention':.961}}
    entry={'selected_update':960,'operating_points':{'0.96':point}}
    summary={'models':{arm+'_seed913':copy.deepcopy(entry) for arm in ('hard','hard_tv','hard_margin')},
             'raw_confidence':{'0.96':copy.deepcopy(point)}}
    original=settings(summary)
    summary['models']['hard_seed913']['operating_points']['0.96']['assessment']={'retention':0}
    assert settings(summary)==original
    assert original['postdraft_hard']['checkpoint']=='hard_seed913.pt'
    summary['raw_confidence']['0.96']['calibration']['retention']=.8
    with pytest.raises(ValueError,match='calibration'):
        settings(summary)


def test_postdraft_features_exactly_match_training_contract():
    torch=pytest.importorskip('torch')
    from dflash.postdraft_latency import features
    e=torch.randn(3,15,2560).half()
    stats=torch.randn(3,15,3)
    mean=torch.tensor([.1,.2,.3],dtype=torch.float64)
    std=torch.tensor([.5,.9,1e-6],dtype=torch.float64)
    expected=torch.cat((e.float()/torch.sqrt(e.float().square().mean(-1,keepdim=True)+1e-6),
                        ((stats.double()-mean)/std).float()),-1)
    assert torch.equal(features(e,stats,mean,std),expected)


def test_same_mode_checks_keep_cross_mode_diagnostics_and_kv_guard():
    s=(Path(__file__).resolve().parents[1]/'scripts/midverify_latency_hook/latency_runtime.py').read_text()
    assert "report['cross_mode_diagnostic']" in s
    assert "raise AssertionError('Same-mode token decision mismatch')" in s
    assert "raise AssertionError('Same-mode sampled KV mismatch')" in s


def test_postdraft_summary_uses_postdraft_hard_control():
    from scripts.summarize_predraft_latency import summarize
    cases=['fixed16','fixed8_redraft','fixed12_redraft','postdraft_hard','postdraft_hard_tv','raw_confidence']
    cells=[]
    for case in cases:
        cells.append({'C':2,'mode':'graph','case':case,'rows':2,'offset':0,'prompt_ids':[1,2],
          'observations':[{'committed_tokens':7,'accepted':[2,3],'candidate_blocks':[list(range(16))]*2}],
          'uninstrumented_cycle_ms':{'mean':10.,'stdev':0.},'uninstrumented_cycle_samples_ms':[10.,10.],
          'mean_front':8.,'mean_end':8.,'phase_stream_ms':{},'anchor_mismatches':0,
          'audit':{'top1_differences':0,'acceptance_differences':0,'bonus_differences':0,
                   'hidden':{'relative_l2':0.},'cross_mode_diagnostic':{'top1_differences':1}}})
    r=summarize({'config':{'suite':'postdraft_verification_trim','cases':cases},'results':cells})
    assert r['metrics']['c2_graph_raw_confidence']['speed_ratio_vs_postdraft_hard']==1.
    assert r['metrics']['c2_graph_raw_confidence']['cross_mode_diagnostic']['top1_differences']==1
    assert 'c2_graph_postdraft_hard_tv' in r['uncertainty']
