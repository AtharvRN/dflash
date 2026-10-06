import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dflash.midverify import apply_setting
from dflash.predraft_latency import fixed_width, frozen_lengths, resolve_concurrencies, same_model_identity
from scripts.export_predraft_latency_bundle import select, verify
from scripts.train_predraft_soft_supervision import build_model, policy_scores


def test_concurrency_defaults_and_matched_low_concurrency_cohorts():
    assert resolve_concurrencies() == [64, 128]
    assert resolve_concurrencies(smoke=True) == [4]
    assert resolve_concurrencies([8, 16, 32]) == [8, 16, 32]
    assert resolve_concurrencies([4], smoke=True) == [4]


@pytest.mark.parametrize('values', [[], [0], [-8], [3], [256], [8, 8], [True], [8.0]])
def test_invalid_concurrencies(values):
    with pytest.raises(ValueError):
        resolve_concurrencies(values)


def test_smoke_remains_c4():
    with pytest.raises(ValueError, match='Smoke'):
        resolve_concurrencies([2], smoke=True)


def test_invalid_cli_concurrency_rejected_before_gpu_or_files(monkeypatch, tmp_path):
    import sys
    from scripts import gpu_runtime, run_midverify_latency
    monkeypatch.setattr(sys, 'argv', ['replay', '--output', str(tmp_path/'out'),
        '--predraft-bundle', str(tmp_path/'bundle'), '--use-container-gpu', '--concurrencies', '3'])
    def unexpected(**kwargs):
        pytest.fail('Invalid concurrency must fail before GPU access')
    monkeypatch.setattr(gpu_runtime, 'configure_gpu_runtime', unexpected)
    with pytest.raises(ValueError, match='Concurrencies'):
        run_midverify_latency.main()
    assert not (tmp_path/'out').exists()


def test_exact_frozen_threshold_and_all_integer_lengths():
    scores = -torch.arange(1, 16).float().repeat(16, 1)
    for count in range(16):
        threshold = np.nextafter(float(-count-1), np.inf)
        actual = frozen_lengths(scores, threshold).numpy()
        np.testing.assert_array_equal(actual, apply_setting(scores.numpy(), {'threshold': threshold}))
        assert np.all(actual == count+1)
    assert frozen_lengths(scores, None).tolist() == [16]*16
    with pytest.raises(ValueError):
        frozen_lengths(scores[:, :3], 0.)


def test_same_inference_head_for_hard_and_mixed():
    torch.manual_seed(913)
    m = build_model().eval()
    with torch.no_grad():
        output = m(torch.randn(4, 2560))
    assert output.shape == (4, 3, 15)
    assert torch.equal(policy_scores(output, 'hard'), policy_scores(output, 'mixed_tv_bce'))
    assert fixed_width('fixed16') == 16
    assert fixed_width('fixed12_same_candidates') == 12
    assert fixed_width('fixed8_redraft') == 8
    with pytest.raises(ValueError):
        fixed_width('predraft_hard')


def test_cohort_selection_uses_membership_not_acceptance():
    tokens = [10, 20, 30]
    base = {'prefix_token_ids': tokens, 'prefix_length': 2,
            'prefix_sha256': hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()}
    rows = [{**base, 'prompt_id': i, 'cycle': j, 'group': 'assessment' if i else 'train',
             'accepted_len': 0 if j == 0 else 15} for i in range(5) for j in range(2)]
    eligible = np.ones(len(rows), bool)
    eligible[2] = False
    got = select(rows, eligible, 3)
    assert [(r['prompt_id'], r['cycle']) for r in got] == [(1,1), (2,0), (3,0)]
    assert select(rows, eligible, 3) == got
    with pytest.raises(ValueError, match='Insufficient'):
        select(rows, eligible, 6)


def test_bundle_detects_mutation(tmp_path):
    p = tmp_path/'data.json'
    p.write_text('{}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'COMPLETE.json').write_text(json.dumps({'complete': True, 'binding': {'data.json': digest}}))
    verify(tmp_path)
    p.write_text('[]')
    with pytest.raises(ValueError, match='binding'):
        verify(tmp_path)


def test_model_identity_allows_only_storage_path_changes():
    a = {k: {'repo': k, 'revision': 'abc', 'path': '/pvc/abc'} for k in ('target', 'draft')}
    b = {k: {**v, 'path': '/workstation/abc'} for k, v in a.items()}
    assert same_model_identity(a, b)
    b['draft']['revision'] = 'changed'
    assert not same_model_identity(a, b)


def test_predraft_computation_precedes_current_draft_and_no_confidence():
    # Runtime requires pinned SGLang/CUDA; CPU guard inspects call ordering, and
    # the scheduled C4 smoke supplies the actual integration/correctness gate.
    source = (Path(__file__).resolve().parents[1]/'scripts/midverify_latency_hook/latency_runtime.py').read_text()
    body = source.split('    def run(self, timed=True):',1)[1].split('\ndef correctness',1)[0]
    assert body.index('self.policies[case].lengths(predraft_fused(s))') < body.index('draft_forward(s, timer')
    assert "confidence = 'logprob_only' if case == 'raw_confidence' else not (case.startswith('fixed') or predraft)" in body
    assert 'max_seconds' in (Path(__file__).resolve().parents[1]/'scripts/run_midverify_latency.py').read_text()


def test_container_launch_validates_allocation_before_files(monkeypatch, tmp_path):
    import sys
    from scripts import gpu_runtime, run_midverify_latency
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(sys, 'argv', ['replay', '--output', str(tmp_path/'out'),
        '--predraft-bundle', str(tmp_path/'bundle'), '--use-container-gpu'])
    def check(**kwargs):
        assert kwargs == {'use_visible_gpu': False, 'use_container_gpu': True, 'require_gpu': True}
        raise RuntimeError('allocation guard reached')
    monkeypatch.setattr(gpu_runtime, 'configure_gpu_runtime', check)
    with pytest.raises(RuntimeError, match='allocation guard reached'):
        run_midverify_latency.main()
    assert not (tmp_path/'out').exists()
