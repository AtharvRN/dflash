from scripts.profile_sglang_latency import COMPONENTS, decompose, distribution, load_workload


def test_nested_components_are_disjoint():
    names = ["decode_cycle", "draft_and_verify_prepare", "draft_forward", "draft_projection_argmax",
             "target_verify_prepare", "target_forward", "acceptance_and_target_kv_commit", "draft_kv_upkeep"]
    times = [20, 8, 4, 2, 1, 8, 2, 1]
    parents = [None, 0, 1, 1, 1, 0, 0, 0]
    row = {"spans": [{"name": n, "parent": p, "stream_elapsed_ms": t, "host_call_ms": t/2}
                     for n, p, t in zip(names, parents, times)]}
    values, total = decompose(row)
    assert total == 20 and sum(values.values()) == total
    assert values["draft_setup_allocation"] == 1
    assert values["other_worker"] == 1
    assert sum(decompose(row, "host_call_ms")[0].values()) == 10


def test_distribution_small_samples():
    assert distribution([]) is None
    assert distribution([3])["stdev"] == 0
    assert distribution([3, 1, 2])["median"] == 2
