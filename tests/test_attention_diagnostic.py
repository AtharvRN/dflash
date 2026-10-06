import pytest
torch = pytest.importorskip('torch')
from scripts.midverify_latency_hook.attention_diagnostic import decode_plan, dense_causal_reference


def test_plan_fields_are_not_guessed_offsets():
    eager = decode_plan([32,384,0,64,0,128,256,0,384,516,0,0,0,0,0])
    graph = decode_plan([43,384,672,128,0,176,352,688,528,660,0,90177536,2240,1,1])
    assert eager['cta_tile_q'] == 64 and not eager['split_kv']
    assert graph['cta_tile_q'] == 128 and graph['split_kv']
    with pytest.raises(ValueError):
        decode_plan([0]*14)


def test_high_precision_reference_uses_bottom_right_causal_and_gqa():
    q = torch.zeros(2,4,2)
    k = torch.zeros(4,2,2)
    v = torch.arange(4.).view(4,1,1).expand(4,2,2).clone()
    out = dense_causal_reference(q,k,v,1.)
    torch.testing.assert_close(out[0], torch.ones(4,2,dtype=torch.float64))
    torch.testing.assert_close(out[1], torch.full((4,2),1.5,dtype=torch.float64))
    v[-1] = 100
    changed = dense_causal_reference(q,k,v,1.)
    torch.testing.assert_close(changed[0],out[0])
    assert not torch.equal(changed[1],out[1])
