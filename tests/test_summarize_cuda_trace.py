from scripts.summarize_cuda_trace import analyze, interval_union


def test_union_does_not_double_count_overlap():
    assert interval_union([(1,3),(2,5),(8,9)]) == 5


def test_launch_correlation_and_nested_stage():
    events = [
        {"ph":"X", "cat":"user_annotation", "pid":1,"tid":2,"ts":0,"dur":100,
         "name":"DFLASH/decode_cycle|B=16|C=64|cycle=10"},
        {"ph":"X", "cat":"user_annotation", "pid":1,"tid":2,"ts":10,"dur":40,
         "name":"DFLASH/target_forward"},
        {"ph":"X", "cat":"cuda_runtime", "pid":1,"tid":2,"ts":20,"dur":2,
         "name":"cudaGraphLaunch", "args":{"correlation":123}},
        {"ph":"X", "cat":"kernel", "pid":0,"tid":7,"ts":25,"dur":10,
         "name":"example_gemm", "args":{"correlation":123}},
        {"ph":"X", "cat":"kernel", "pid":0,"tid":7,"ts":37,"dur":5,
         "name":"example_attention", "args":{"correlation":123}},
        {"ph":"X", "cat":"kernel", "pid":0,"tid":7,"ts":110,"dur":20,
         "name":"outside_prefill", "args":{"correlation":999}},
    ]
    out = analyze(events, 64)
    assert out["stage_gpu_sum_ms_per_cycle"] == {"DFLASH/target_forward": .015}
    assert out["gpu_activities_outside_selected_roots_or_unattributed"] == 1
    assert out["cycle_activity"][0]["gaps_inside_envelope_ms"] == .002
