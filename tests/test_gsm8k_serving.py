import hashlib
import json
from pathlib import Path
import sys
from fractions import Fraction
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_gsm8k_serving import build_workload
from score_gsm8k_serving import numeric, predictions, summarize


def test_numeric_box_parser():
    assert predictions(r"We had 99. Final: \boxed{1,234.00}")[0] == 1234
    assert predictions(r"\boxed{2} correction \boxed{\frac{1}{2}}")[0] == Fraction(1, 2)
    assert predictions(r"\boxed{\$18}")[0] == 18
    assert predictions(r"Answer is 18")[0] is None
    assert predictions(r"Answer is 18")[1] == 18
    assert predictions(r"\boxed{18 dollars}")[0] is None
    assert predictions(r"\boxed{18")[0] is None
    assert numeric("nan") is None and numeric(r"\frac{1}{0}") is None


def test_workload_is_full_test_and_warmup_disjoint():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert "SECRET_GOLD" not in messages[0]["content"]
            return [1, 2, 3]
    test = [{"question":"test question", "answer":"SECRET_GOLD #### 42"}]
    train = test + [{"question":f"warm {i}", "answer":"SECRET_GOLD #### 3"} for i in range(3)]
    w = build_workload(train, test, Tokenizer(), warmup=2)
    assert len(w["measurement"]) == 1 and len(w["warmup"]) == 2
    assert w["measurement"][0]["gold_answer"] == "42"
    assert not ({r["question_sha256"] for r in w["warmup"]} &
                {r["question_sha256"] for r in w["measurement"]})


def test_accuracy_denominator_includes_unparsed_and_capped_outputs(tmp_path):
    w = {"revision":"pinned", "protocol":"zero-shot", "measurement":[
        {"prompt_id":"a", "gold_answer":"42"}, {"prompt_id":"b", "gold_answer":"8"}]}
    wp = tmp_path/"workload.json"
    wp.write_text(json.dumps(w))
    d = tmp_path/"raw"
    d.mkdir()
    results = [{"prompt_id":pid, "response":{"text":text, "meta_info":{
        "completion_tokens":5, "finish_reason":{"type":finish},
        "spec_verify_ct":2, "spec_num_correct_drafts":2}}}
        for pid,text,finish in [("a",r"\boxed{42}","stop"),("b","answer 8","length")]]
    run = {"workload_sha256":hashlib.sha256(wp.read_bytes()).hexdigest(),
           "case":"raw", "concurrency":4, "repeat":0, "requests":2,
           "results":results, "output_tokens":10, "throughput_tok_s":20}
    path = d/"c4_r0.json"
    path.write_text(json.dumps(run))
    out = summarize(tmp_path)["metrics"]["raw_c4"]
    assert out["strict_boxed_accuracy"] == .5
    assert out["relaxed_numeric_accuracy"] == 1
    assert out["length_cap_fraction"] == .5
    assert out["boxed_parse_failures"] == 1
    run["results"] = results[:1]
    path.write_text(json.dumps(run))
    with pytest.raises(AssertionError):
        summarize(tmp_path)
