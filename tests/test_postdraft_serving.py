import ast
import importlib.util
from pathlib import Path
import sys
import pytest
import json
from types import SimpleNamespace, MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_postdraft_serving import prepare


def test_cycle_profile_backup_preserves_local_trace(tmp_path):
    from benchmark_postdraft_serving import backup_cycle_profile
    source, destination = tmp_path/'scratch', tmp_path/'durable'
    source.mkdir()
    (source/'cycles_1.jsonl').write_text('{"cycle":1}\n')
    (source/'control.json').write_text('{"label":"measured"}')
    backup_cycle_profile(source, destination)
    assert (destination/'cycles_1.jsonl').read_bytes() == (source/'cycles_1.jsonl').read_bytes()
    with (source/'cycles_1.jsonl').open('a') as f:
        f.write('{"cycle":2}\n')
    backup_cycle_profile(source, destination)
    assert len((destination/'cycles_1.jsonl').read_text().splitlines()) == 2


def test_frozen_workload_full_copy_and_disjoint_prefix(tmp_path):
    from benchmark_postdraft_serving import workload, sha
    source = tmp_path / "frozen.json"
    saved = {"warmup": [{"prompt_id": str(i)} for i in range(4)],
             "measurement": [{"prompt_id": str(i)} for i in range(4, 12)]}
    source.write_text(json.dumps(saved))
    full = tmp_path / "full"
    full.mkdir()
    args = SimpleNamespace(workload_file=source, output=full, warmup=4, requests=8)
    assert workload(args, None) == (saved["warmup"], saved["measurement"])
    assert (full / "workload.json").read_bytes() == source.read_bytes()
    small = tmp_path / "small"
    small.mkdir()
    args.output, args.warmup, args.requests = small, 2, 3
    warmup, measured = workload(args, None)
    assert warmup == saved["warmup"][:2] and measured == saved["measurement"][:3]
    assert json.loads((small / "workload.json").read_text())["parent_workload_sha256"] == sha(source)
    args.requests = 9
    with pytest.raises(ValueError):
        workload(args, None)


def test_extension_restores_and_parses(tmp_path):
    dst = tmp_path / "source"
    manifest = prepare(dst)
    assert len(manifest["changes"]) == 9
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


def test_logprob_reuse_does_not_change_draft_candidates(tmp_path):
    torch = pytest.importorskip("torch")
    dst = tmp_path / "source"
    prepare(dst)
    source = ast.parse((dst / "python/sglang/srt/speculative/dflash_worker.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_greedy_sample_from_vocab_parallel_head")
    namespace = {"torch": torch, "get_tp_group": lambda: SimpleNamespace(world_size=1)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "projection_test", "exec"), namespace)
    worker = SimpleNamespace(_draft_greedy_local_cap=0, _draft_greedy_local_max_buf=None,
                             _draft_greedy_local_arg_buf=None)
    project = MethodType(namespace[method.name], worker)
    torch.manual_seed(913)
    hidden = torch.randn(31, 8)
    head = SimpleNamespace(weight=torch.randn(23, 8), shard_indices=SimpleNamespace(
        num_org_elements=23, num_org_elements_padded=23, num_added_elements=0,
        org_vocab_start_index=0, added_vocab_start_index=23))
    plain = project(hidden_states=hidden, lm_head=head)
    raw_ids, raw_stats = project(hidden_states=hidden, lm_head=head, return_confidence="logprob_only")
    full_ids, full_stats = project(hidden_states=hidden, lm_head=head, return_confidence=True)
    assert torch.equal(plain, raw_ids) and torch.equal(plain, full_ids)
    assert torch.equal(raw_stats[:,3], full_stats[:,3])
    assert not raw_stats[:,:3].any()


def test_unsupported_raw_policy_cannot_be_silently_ignored(tmp_path):
    dst = tmp_path / "source"
    prepare(dst)
    source = ast.parse((dst / "python/sglang/srt/arg_groups/speculative_hook.py").read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "handle_speculative_decoding")
    namespace = {"_is_spec_v2_enabled": lambda: True}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "arg_guard_test", "exec"), namespace)
    for overrides in [{"speculative_algorithm": None}, {"tp_size": 2},
                      {"attention_backend": "triton"},
                      {"speculative_dflash_postdraft_logprob_threshold": float("nan")}]:
        fields = dict(speculative_algorithm="DFLASH", tp_size=1, attention_backend="flashinfer",
                      speculative_dflash_postdraft_logprob_threshold=-.8)
        with pytest.raises(ValueError):
            namespace[fn.name](SimpleNamespace(**(fields | overrides)))
    namespace["_is_spec_v2_enabled"] = lambda: False
    with pytest.raises(ValueError, match="spec-v2"):
        namespace[fn.name](SimpleNamespace(**fields))


def test_verify_length_audit_excludes_warmup_and_matches_response_cycles(tmp_path):
    from dflash.serving_verify_audit import make_record
    from summarize_verify_lengths import summarize
    dst = tmp_path / "raw"
    dst.mkdir()
    def req(rid, finished=False):
        return SimpleNamespace(rid=rid, is_retracted=False, finished=lambda: finished)
    rid = "pdv:raw_c4_r0:12"
    records = [make_record([req("warmup")], [16], [8], 16, 16, True),
               make_record([req(rid)], [3], [2], 3, 4, True),
               make_record([req(rid)], [5], [3], 5, 8, True),
               make_record([req(rid, True)], [2], [1], 2, 2, False)]
    (dst / "verify_lengths.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    response = {"case":"raw", "concurrency":4, "repeat":0,
                "results":[{"prompt_id":"12", "response":{"meta_info":{
                    "id":rid, "spec_verify_ct":2, "spec_num_correct_drafts":3}}}]}
    (dst / "c4_r0.json").write_text(json.dumps(response))
    out = summarize(tmp_path)["phases"]["pdv:raw_c4_r0"]
    assert out["cycles"] == 2 and out["selected"] == 8
    assert out["average_verification_length_including_anchor"] == 4
    assert out["accepted_over_selected_proposals"] == .5
    assert out["executed_rows_per_counted_cycle"] == 7
    response["results"][0]["response"]["meta_info"]["spec_verify_ct"] = 3
    (dst / "c4_r0.json").write_text(json.dumps(response))
    with pytest.raises(AssertionError):
        summarize(tmp_path)
