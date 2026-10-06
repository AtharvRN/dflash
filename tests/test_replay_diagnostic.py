import ast
from pathlib import Path
import pytest

from dflash.predraft_latency import replay_offsets


def test_single_batch_preserves_original_offset_and_tail():
    assert replay_offsets(1399, 32, 608) == [608]
    assert replay_offsets(1399, 32)[-1] == 1376
    assert replay_offsets(1399, 32, 1376) == [1376]
    for offset in [-1, 609, 1408]:
        with pytest.raises(ValueError, match='aligned'):
            replay_offsets(1399, 32, offset)


def test_diagnostic_persists_report_before_unchanged_guard():
    source = (Path(__file__).resolve().parents[1]/'scripts/midverify_latency_hook/latency_runtime.py').read_text()
    ast.parse(source)
    body = source.split('def correctness(',1)[1].split('@torch.no_grad()',1)[0]
    assert "> .02" in body
    assert body.index('atomic_json(audit_path, report)') < body.index("raise AssertionError('Segmented forward exceeds 2%")
    assert 'same_mode_repeat_hidden' in source
    assert 'uncaptured_graph_wrapper_vs_eager_hidden' in source
    assert 'prefix_slot_mapping_unchanged' in source
