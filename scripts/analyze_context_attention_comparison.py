"""Paired, prompt-level bootstrap for fixed-calibration offline policy comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.context_attention import choose_budget
from scripts.train_context_attention import atomic_json, buffered_backup


def summarize(totals):
    count, error, accepted, budget, reference = np.moveaxis(totals, -1, 0)
    retention = np.divide(accepted, reference, out=np.full(np.shape(reference), np.nan), where=reference>0)
    return np.stack([error/count, accepted/budget, retention, accepted/count, budget/count], axis=-1)


def paired_intervals(reference, candidate, *, draws=2000, seed=913):
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 5:
        raise ValueError("Expected aligned prompt-level sums")
    if not np.array_equal(reference[:, 0], candidate[:, 0]) or not np.array_equal(reference[:, 4], candidate[:, 4]):
        raise ValueError("Comparisons must use identical prompt rows and labels")
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, draws, 100):
        indices = rng.integers(len(reference), size=(min(100,draws-start),len(reference)))
        deltas.append(summarize(candidate[indices].sum(1))-summarize(reference[indices].sum(1)))
    deltas = np.concatenate(deltas)
    point = summarize(candidate.sum(0))-summarize(reference.sum(0))
    names = ["mae", "accept_ratio", "retention", "mean_accepted", "mean_budget"]
    result = {}
    for i,name in enumerate(names):
        valid = deltas[np.isfinite(deltas[:, i]), i]
        result[name] = {"difference":float(point[i]) if np.isfinite(point[i]) else None,
                        "ci95":np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                        "valid_bootstrap_draws":len(valid)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--persistent-dir",type=Path)
    args=p.parse_args()
    torch.set_num_threads(4)
    config=json.loads((args.run_dir/"config.json").read_text())
    cache=Path(config["cache_dir"])
    rows=np.load(cache/"val/row_index.npy")
    assessment=~np.load(cache/"val/calibration.npy").astype(bool)
    y=np.load(cache/"val/accepted_len.npy")[assessment]
    prompts,groups=np.unique(rows[assessment,2],return_inverse=True)
    results=json.loads((args.run_dir/"summary.json").read_text())
    grouped={}
    for name,result in results.items():
        survival=np.load(args.run_dir/name/"validation_survival.npy")[assessment]
        budget=choose_budget(torch.from_numpy(survival),result["assessment_proxy_policy"]["alpha"]).numpy()
        values=[np.ones(len(y)),np.abs(survival.sum(1)-y),np.minimum(y,budget),budget,y]
        grouped[name]=np.stack([np.bincount(groups,weights=v,minlength=len(prompts)) for v in values],axis=1)
    pairs=[("last_mlp",name) for name in results if name!="last_mlp"]
    if "residual_last_only" in results and "residual_attention" in results:
        pairs.append(("residual_last_only","residual_attention"))
    report={"assessment_rows":len(y),"assessment_prompts":len(prompts),"bootstrap_draws":2000,
            "note":"Candidate minus reference. Resampling unit is prompt; thresholds frozen on calibration. Does not measure training-seed variability. Retention intervals exclude zero-reference draws; valid counts are reported.",
            "comparisons":{f"{candidate}_minus_{reference}":paired_intervals(grouped[reference],grouped[candidate]) for reference,candidate in pairs}}
    path=args.run_dir/"paired_comparison.json"
    atomic_json(path,report)
    if args.persistent_dir:
        buffered_backup(path,args.persistent_dir/path.name)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
