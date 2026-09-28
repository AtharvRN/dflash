"""GPU-resident predictor/selection microbenchmark, NOT integration overhead."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.block_response import BlockResponseMLP
from dflash.policy import HorizonPredictorRuntime


def measure(fn, *, repeats=10, iterations=100):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    gpu_ms, wall_ms = [], []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter()-begin)*1000/iterations)
        gpu_ms.append(start.elapsed_time(end)/iterations)
    return {"stream_ms_mean": statistics.mean(gpu_ms), "stream_ms_stdev": statistics.stdev(gpu_ms),
            "wall_ms_mean": statistics.mean(wall_ms), "wall_ms_stdev": statistics.stdev(wall_ms),
            "repetitions": repeats, "calls_per_repetition": iterations,
            "stream_ms_samples": gpu_ms, "wall_ms_samples": wall_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/scratch/zekaili/atharv/dflash"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing measurements")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old_path = args.root / "checkpoints/context_residual_100k_20260913/last_mlp_best_epoch_4.pt"
    new_path = args.root / "runs/actual_block_predictor_10k_finish_20260928/training/actual_seed_913.pt"
    old = HorizonPredictorRuntime(input_dim=2560, proj_dim=512, hidden_size=256, num_slots=15,
        architecture="last_mlp", num_layers=1, dropout=.05, context_window=16).eval().cuda()
    old.load_state_dict(torch.load(old_path, map_location="cpu", weights_only=False)["model"])
    checkpoint = torch.load(new_path, map_location="cpu", weights_only=False)
    new = BlockResponseMLP().eval().cuda()
    new.load_state_dict(checkpoint["model"])
    setting = checkpoint["calibration_points"]["0.96"]["setting"]
    if setting["kind"] != "penalty":
        raise ValueError("Expected a nontrivial calibrated policy")
    cache = args.root / "runs/policy_granularity_20260927/cache"
    feature_paths = sorted(cache.glob("prompt_*_fused.npy"))
    features = np.concatenate([np.load(p, allow_pickle=False) for p in feature_paths[:32]])[:128]
    results = []
    with torch.inference_mode():
        for count in [1, 16, 32, 64, 128]:
            x = torch.as_tensor(features[:count], device="cuda", dtype=torch.float16)
            mask = torch.ones((count, 1), device="cuda")
            budgets = torch.arange(1, 16, device="cuda", dtype=torch.float32)

            def old_decision():
                survival = old(x[:, None], mask).sigmoid().cumprod(-1)
                cumulative = survival.cumsum(-1)
                return (cumulative >= .99*cumulative[:, -1:]).int().argmax(-1) + 2

            def new_decision():
                return (new(x) - setting["value"]*budgets).argmax(-1) + 2

            for name, fn in [("frozen_100k", old_decision), ("actual_seed_913", new_decision)]:
                expected = fn().clone()
                eager = measure(fn)
                eager_readback = measure(lambda: fn().cpu().tolist())
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(10):
                        fn()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    observed = fn()
                graph.replay()
                torch.cuda.synchronize()
                if not torch.equal(expected, observed):
                    raise ValueError("Graph/eager policy disagreement")
                graphed = measure(graph.replay)
                def replay_and_readback():
                    graph.replay()
                    return observed.cpu().tolist()
                graphed_readback = measure(replay_and_readback)
                results.append({"model": name, "batch_size": count, "eager": eager, "cuda_graph": graphed,
                                "eager_with_cpu_lengths": eager_readback,
                                "cuda_graph_with_cpu_lengths": graphed_readback,
                                "selected_B": expected.cpu().tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "tf32": False, "weights_dtype": "FP32", "input_dtype": "FP16",
        "checkpoint_sha256": {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old_path, new_path]},
        "scope": "GPU-resident feature cast, trained MLP and integer decision, plus separately labeled synchronous CPU-length readback; no feature gather, ragged packing, KV bookkeeping or integrated-serving claim",
        "seed_choice": "913 fixed in advance, not selected by performance; same architecture across seeds"}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
