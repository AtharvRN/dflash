import pytest
from scripts import gpu_runtime as g


def test_retry_only_occupancy_without_changing_thresholds(monkeypatch):
    now, calls = [0.], []
    monkeypatch.setattr(g.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(g.time, 'sleep', lambda t: now.__setitem__(0, now[0]+t))
    def configure(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise g.GPUOccupiedError('0 MiB, stale 100%')
        return {'idle': True}
    monkeypatch.setattr(g, 'configure_gpu_runtime', configure)
    args = dict(use_container_gpu=True, require_gpu=True)
    assert g.wait_gpu_runtime(wait_seconds=10, **args) == {'idle': True}
    assert calls == [args]*3
    assert now[0] == 4


def test_persistent_occupancy_remains_failure(monkeypatch):
    now = [0.]
    monkeypatch.setattr(g.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(g.time, 'sleep', lambda t: now.__setitem__(0, now[0]+t))
    def occupied(**kwargs):
        raise g.GPUOccupiedError('busy')
    monkeypatch.setattr(g, 'configure_gpu_runtime', occupied)
    with pytest.raises(g.GPUOccupiedError):
        g.wait_gpu_runtime(wait_seconds=3)
    assert now[0] == 3


@pytest.mark.parametrize('error', [ValueError('bad allocation'), RuntimeError('cannot query GPU')])
def test_bad_allocation_or_query_never_retried(monkeypatch, error):
    def bad(**kwargs):
        raise error
    monkeypatch.setattr(g, 'configure_gpu_runtime', bad)
    monkeypatch.setattr(g.time, 'sleep', lambda _: pytest.fail('Must not retry'))
    with pytest.raises(type(error), match=str(error)):
        g.wait_gpu_runtime()


def test_zero_wait_preserves_original_fail_fast_behavior(monkeypatch):
    def bad(**kwargs):
        raise g.GPUOccupiedError('busy')
    monkeypatch.setattr(g, 'configure_gpu_runtime', bad)
    monkeypatch.setattr(g.time, 'sleep', lambda _: pytest.fail('Must not wait'))
    with pytest.raises(g.GPUOccupiedError):
        g.wait_gpu_runtime(wait_seconds=0)
