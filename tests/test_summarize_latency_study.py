import pytest
from scripts.summarize_latency_study import compare_outputs


def test_output_comparison_counts_requests_not_tokens():
    a = {"requests": [{"prompt_id": 4, "response": {"output_ids": [1, 2, 3]}}]}
    b = {"requests": [{"prompt_id": 4, "response": {"output_ids": [1, 5, 6]}}]}
    assert compare_outputs(a, a)["different_token_sequences"] == 0
    assert compare_outputs(a, b)["different_token_sequences"] == 1
    b["requests"][0]["prompt_id"] = 5
    with pytest.raises(ValueError, match="Unmatched"):
        compare_outputs(a, b)
