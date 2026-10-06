import ast
import importlib.util
from pathlib import Path
import sys
import pytest
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_postdraft_serving import prepare


def test_extension_restores_and_parses(tmp_path):
    dst = tmp_path / "source"
    manifest = prepare(dst)
    assert len(manifest["changes"]) == 5
    for rel in manifest["changes"]:
        ast.parse((dst / "python/sglang/srt" / rel).read_text())
    with pytest.raises(ValueError):
        prepare(dst)


def test_raw_prefix_policy():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("raw_policy", ROOT / "dflash/serving_raw_confidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    policy = mod.DFlashRawConfidencePolicy(-0.8, 5)
    scalar = torch.zeros((5, 4, 4))
    scalar[..., 3] = torch.tensor([[-1., 0., 0., 0.], [0., 0., -1., 0.],
                                   [0., 0., 0., 0.], [float("nan"), 0., 0., 0.],
                                   [-.5, -.5, -.5, -1.]])
    assert policy.select_verify_lens(hidden=None, prev_token_ids=None, scalar=scalar,
                                    max_block_size=5).tolist() == [1, 3, 5, 1, 4]
    assert policy.arms == (1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        mod.DFlashRawConfidencePolicy(float("nan"))


def test_summary_counts_actual_completions_over_workload_wall_time(tmp_path):
    from summarize_postdraft_serving import summarize
    for case, elapsed in [("fixed16", 2.), ("raw", 1.5)]:
        target = tmp_path / case
        target.mkdir()
        row = {"case": case, "concurrency": 4, "repeat": 0, "requests": 2,
               "workload_sha256": "same", "output_tokens": 30, "wall_time_s": elapsed,
               "throughput_tok_s": 30/elapsed, "results": [
                   {"prompt_id": str(i), "response": {"text": "same", "meta_info": {"completion_tokens": t}}}
                   for i, t in enumerate([10,20])]}
        (target / "c4_r0.json").write_text(json.dumps(row))
    result = summarize(tmp_path)
    assert result["metrics"]["raw_c4"]["throughput_tok_s_mean"] == 20
    assert result["metrics"]["raw_c4"]["speedup_vs_fixed16"] == pytest.approx(4/3)
    assert not result["complete"]
