"""Matched post-draft supervision ablation; no target features at inference.

Primary head predicts greedy match. Two auxiliary heads are instantiated in
EVERY arm, but only the designated auxiliary loss is active. All losses mask
the off-path suffix. No assessment-based checkpoint or threshold selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.midverify import accepted_lengths, risk_mask, calibrate, apply_setting, metrics
from scripts.audit_block_headroom import sha256
from scripts.collect_policy_granularity import atomic_json
from scripts.collect_midverify_probe import backup_file
from scripts.gpu_runtime import configure_gpu_runtime
from scripts.train_midverify_scaling import batch_stream, parameter_sha

ARMS = ("hard", "hard_tv", "hard_margin")
TARGETS = (.90, .95, .96, .98, .99, 1.0)
FIELDS = ("matches", "accepted_len", "at_risk", "tv_overlap", "target_margin", "draft_stats",
          "candidate_vectors", "eligible", "anchor_match", "accepted_eos")


def load_cache(root, *, view="postdraft"):
    if view not in ("postdraft", "predraft"):
        raise ValueError("Unknown cache view")
    fields = FIELDS if view == "postdraft" else tuple(k for k in FIELDS if k != "candidate_vectors") + ("fused", "target_candidate_prob")
    complete = json.loads((root/"COMPLETE.json").read_text())
    if not complete.get("complete"):
        raise ValueError("Incomplete collection")
    if set(complete["binding"]) != {"config.json", "receipts.json", "summary.json"}:
        raise ValueError("Unexpected collection bindings")
    for name, digest in complete["binding"].items():
        if sha256(root/name) != digest:
            raise ValueError("Collection binding mismatch")
    config = json.loads((root/"config.json").read_text())
    if config["schema"] != "soft_supervision_v1":
        raise ValueError("Wrong cache schema")
    rows, chunks, seen = config["rows"], [], []
    for receipt in json.loads((root/"receipts.json").read_text()):
        name = receipt["file"]
        if Path(name).name != name or sha256(root/name) != receipt["sha256"]:
            raise ValueError("Shard checksum mismatch")
        if receipt["summary"].get("canonical_disagreements", 0):
            raise ValueError("Canonical disagreement")
        with np.load(root/name, allow_pickle=False) as shard:
            chunk = {k: shard[k] for k in fields}
        if any(len(v) != len(receipt["rows"]) for v in chunk.values()):
            raise ValueError("Shard alignment mismatch")
        seen.extend(receipt["rows"])
        chunks.append(chunk)
    if seen != list(range(len(rows))) or [r["row"] for r in rows] != seen:
        raise ValueError("Row identity/order mismatch")
    arrays = {k: np.concatenate([c[k] for c in chunks]) for k in fields}
    if not np.array_equal(accepted_lengths(arrays["matches"]), arrays["accepted_len"]):
        raise ValueError("Acceptance mismatch")
    if not np.array_equal(risk_mask(arrays["accepted_len"]), arrays["at_risk"]):
        raise ValueError("At-risk mask mismatch")
    if not np.array_equal(arrays["eligible"], arrays["anchor_match"] & ~arrays["accepted_eos"]):
        raise ValueError("Eligibility mismatch")
    n = len(rows)
    shapes = [("draft_stats", (n,15,3)), ("tv_overlap", (n,15)), ("target_margin", (n,15))]
    shapes += ([("candidate_vectors", (n,15,2560))] if view == "postdraft" else
               [("fused", (n,2560)), ("target_candidate_prob", (n,15))])
    for field, shape in shapes:
        if arrays[field].shape != shape or not np.isfinite(arrays[field]).all():
            raise ValueError("Invalid feature or label array")
    if np.any((arrays["tv_overlap"] < -1e-6) | (arrays["tv_overlap"] > 1+1e-6)):
        raise ValueError("Invalid TV labels")
    if view == "predraft" and np.any((arrays["target_candidate_prob"] < 0) | (arrays["target_candidate_prob"] > 1)):
        raise ValueError("Invalid target probability labels")
    groups = {g: {int(r["prompt_id"]) for r in rows if r["group"] == g} for g in ("train", "calibration", "assessment")}
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError("Prompt leakage")
    planned = dict(Counter(r["group"] for r in rows))
    expected = dict.fromkeys(groups, 8) if config["smoke"] else {"train":2000,"calibration":342,"assessment":1416}
    if planned != expected:
        raise ValueError("Frozen source row membership changed")
    return config, rows, arrays


def build_model(width=2563):
    from torch import nn
    return nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(.05), nn.Linear(128, 3))


def objective(output, hard, tv, margin, mask, arm):
    import torch.nn.functional as F
    if arm not in ARMS:
        raise ValueError("Unknown supervision arm")
    mean = lambda x: (x*mask).sum()/mask.sum().clamp_min(1)
    loss = mean(F.binary_cross_entropy_with_logits(output[..., 0], hard, reduction="none"))
    if arm == "hard_tv":
        loss = loss + .1*mean(F.binary_cross_entropy_with_logits(output[..., 1], tv, reduction="none"))
    elif arm == "hard_margin":
        loss = loss + .1*mean(F.smooth_l1_loss(output[..., 2], margin, reduction="none"))
    return loss


def score(model, x):
    import torch
    model.eval()
    with torch.inference_mode():
        # Raw logits retain ordering without sigmoid saturation; same rule in all arms.
        return np.concatenate([model(x[i:i+64])[..., 0].cpu().numpy() for i in range(0,len(x),64)])


def bootstrap(a, k, ref, prompts, draws=2000):
    _, inv = np.unique(prompts, return_inverse=True)
    got, base = np.minimum(a,k-1), np.minimum(a,ref-1)
    totals = np.column_stack([np.bincount(inv, weights=v) for v in
                             (got,base,a,k-1,ref-1,np.ones(len(a)))])
    samples = totals[np.random.default_rng(1005).integers(len(totals),size=(draws,len(totals)))].sum(1)
    ci = lambda x: np.quantile(x,[.025,.975]).tolist()
    return {"retention_ci95":ci(samples[:,0]/samples[:,2]),
            "accept_ratio_ci95":ci(samples[:,0]/np.maximum(samples[:,3],1)),
            "mean_budget_ci95":ci(samples[:,3]/samples[:,5]),
            "retention_delta_vs_hard_ci95":ci((samples[:,0]-samples[:,1])/samples[:,2]),
            "mean_budget_delta_vs_hard_ci95":ci((samples[:,3]-samples[:,4])/samples[:,5]),
            "ratio_delta_vs_hard_ci95":ci(samples[:,0]/np.maximum(samples[:,3],1)-samples[:,1]/np.maximum(samples[:,4],1)),
            "scope":"paired whole prompts; fitted checkpoint/settings fixed; not training/calibration uncertainty",
            "draws":draws}


def descriptive_curve(scores, accepted):
    """Assessment descriptive only; NEVER selects a checkpoint or deployed setting."""
    levels = np.minimum.accumulate(np.asarray(scores,dtype=np.float64),axis=1).ravel()
    thresholds = np.unique(np.quantile(levels,np.linspace(0,1,101)))
    points = [{"threshold":None, **metrics(accepted,np.full(len(accepted),16))}]
    for t in thresholds:
        setting = {"threshold":float(np.nextafter(t,np.inf))}
        points.append({**setting, **metrics(accepted,apply_setting(scores,setting))})
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","output","backup"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.backup.exists() or args.output.resolve()==args.backup.resolve():
        raise ValueError("Fresh independent destinations required")
    collection, rows, arrays = load_cache(args.cache)
    if bool(collection["smoke"]) != args.smoke:
        raise ValueError("Smoke/full cache mismatch")
    group = np.array([r["group"] for r in rows])
    select = {g:np.flatnonzero((group==g)&arrays["eligible"]) for g in ("train","calibration","assessment")}
    tr, ca, ass = (select[g] for g in ("train","calibration","assessment"))
    if min(map(len,select.values()))==0:
        raise ValueError("Empty eligible partition")
    mask, hard = arrays["at_risk"].astype(np.float32), arrays["matches"].astype(np.float32)
    stats = arrays["draft_stats"][tr].astype(np.float64)[mask[tr].astype(bool)]
    mean, std = stats.mean(0), np.maximum(stats.std(0),1e-6)
    margins = arrays["target_margin"][tr][mask[tr].astype(bool)].astype(np.float64)
    mmean,mstd = float(margins.mean()), max(float(margins.std()),1e-6)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    runtime = configure_gpu_runtime(use_container_gpu=True,require_gpu=True)
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device="cuda"
    candidate=torch.as_tensor(arrays["candidate_vectors"].astype(np.float32),device=device)
    candidate=candidate/torch.sqrt(candidate.square().mean(-1,keepdim=True)+1e-6)
    conf=torch.as_tensor(((arrays["draft_stats"].astype(np.float64)-mean)/std).astype(np.float32),device=device)
    x=torch.cat((candidate,conf),-1)
    del candidate,conf
    to_gpu=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    y,tv,margin,risk=(to_gpu(v[tr]) for v in (hard,arrays["tv_overlap"],
                                            ((arrays["target_margin"]-mmean)/mstd).astype(np.float32),mask))
    tx,cx,ax=x[tr],x[ca],x[ass]
    accepted=arrays["accepted_len"]
    prompts=np.array([r["prompt_id"] for r in rows])[ass]
    seeds,updates,every=([913],8,4) if args.smoke else ([913,914,915],1024,64)
    args.output.mkdir(parents=True)
    args.backup.mkdir(parents=True)
    def save_json(name,data):
        atomic_json(args.output/name,data)
        backup_file(args.output/name,args.backup/name)
    config={"schema":"soft_supervision_training_v1","cache_complete_sha256":sha256(args.cache/"COMPLETE.json"),
        "runtime":runtime,"seeds":seeds,"updates":updates,"calibrate_every":every,"batch_size":128,"arms":ARMS,
        "inputs":"RMS-normalized current candidate LM-head vector (2560) + three draft confidence statistics",
        "architecture":"2563 -> Linear128 -> GELU -> Dropout.05 -> Linear3; primary/TV/margin heads instantiated in every arm",
        "loss":"at-risk hard BCE + 0.1 * (TV BCE OR standardized-margin SmoothL1), or hard only",
        "normalization":{"draft_mean":mean.tolist(),"draft_std":std.tolist(),"margin_mean":mmean,"margin_std":mstd},
        "normalization_scope":"eligible training at-risk positions only","optimizer":"AdamW3e-4, wd.01, clip1",
        "checkpoint_selection":"minimum calibration mean budget at >=96% retention; ties higher retention then earlier update",
        "decision":"first primary logit below calibration threshold ends retained prefix; integers0..15 proposals",
        "counts":{g:{"planned":int((group==g).sum()),"eligible":len(v),"prompts":len({rows[i]["prompt_id"] for i in v})} for g,v in select.items()},
        "commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_sha256":sha256(Path(__file__)),"smoke":args.smoke,"auxiliary_weight_selected_using_assessment":False}
    save_json("config.json",config)
    save_json("row_selection.json",{g:v.tolist() for g,v in select.items()})
    summaries,choices,predictions,curves={}, {}, {}, {}
    started=time.monotonic()
    for seed in seeds:
        initial_reference=None
        order_reference=None
        for arm in ARMS:
            torch.manual_seed(seed)
            model=build_model().to(device)
            initial=parameter_sha(model.state_dict())
            if initial_reference is None:
                initial_reference=initial
            if initial!=initial_reference:
                raise ValueError("Unmatched initialization")
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            stream=batch_stream(len(tr),128,seed)
            order_hash=hashlib.sha256()
            history,best,best_key=[],None,None
            for step in range(1,updates+1):
                b=next(stream)
                order_hash.update(b.numpy().tobytes())
                b=b.to(device)
                model.train()
                opt.zero_grad(set_to_none=True)
                loss=objective(model(tx[b]),y[b],tv[b],margin[b],risk[b],arm)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step()
                if step%every:
                    continue
                qc=score(model,cx)
                point=calibrate(qc,accepted[ca],(.96,))["0.96"]
                key=(point["mean_kept_rows"],-point["retention"],step)
                if best_key is None or key<best_key:
                    best_key=key
                    best=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},step,qc.copy())
                record={"update":step,"loss":float(loss.detach()),"calibration":point}
                history.append(record)
                print("UPDATE",arm,seed,json.dumps(record),flush=True)
            if order_reference is None:
                order_reference=order_hash.hexdigest()
            if order_reference!=order_hash.hexdigest():
                raise ValueError("Unmatched minibatch order")
            weights,step,qc=best
            name=f"{arm}_seed{seed}"
            path=args.output/(name+".pt")
            torch.save({"model":weights,"seed":seed,"arm":arm,"selected_update":step,"config":config},path)
            backup_file(path,args.backup/path.name)
            # Reload persisted weights before final scoring; check exact calibration reproduction.
            model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["model"])
            if not np.array_equal(score(model,cx),qc):
                raise ValueError("Checkpoint reload changed calibration scores")
            settings=calibrate(qc,accepted[ca],TARGETS)
            qa=score(model,ax)  # First assessment use for this fitted model.
            points={}
            for target,setting in settings.items():
                k=apply_setting(qa,setting)
                points[target]={"calibration":setting,"assessment":metrics(accepted[ass],k)}
            k=apply_setting(qa,settings["0.96"])
            choices[name]=k
            reference=choices[f"hard_seed{seed}"]
            summaries[name]={"selected_update":step,"initial_parameter_sha256":initial,
                "minibatch_order_sha256":order_hash.hexdigest(),"parameter_count":sum(v.numel() for v in model.parameters()),
                "primary":points["0.96"],"operating_points":points,
                "bootstrap":bootstrap(accepted[ass],k,reference,prompts,100 if args.smoke else 2000)}
            predictions[name]=qa
            curves[name]=descriptive_curve(qa,accepted[ass])
            save_json(name+"_history.json",history)
            save_json("progress.json",{"completed_models":list(summaries),"last":summaries[name],"elapsed_seconds":time.monotonic()-started})
    raw=arrays["draft_stats"][:,:,0]
    raw_settings=calibrate(raw[ca],accepted[ca],TARGETS)
    raw_points={t:{"calibration":s,"assessment":metrics(accepted[ass],apply_setting(raw[ass],s))} for t,s in raw_settings.items()}
    summary={"models":summaries,"raw_confidence":raw_points,"counts":config["counts"],"across_seeds":{},
             "elapsed_seconds":time.monotonic()-started,"limitations":["Fresh replay subset; report exclusions and drift",
              "Post-draft B16 truncation; full drafting cost remains; no throughput claim",
              "Calibration constraint is not an assessment guarantee; do not rank unmatched-retention points as gains",
              "Aux weight0.1 fixed, no sweep; negative result does not exclude all soft-supervision methods",
              "Previously inspected development assessment, not an untouched final test"]}
    for arm in ARMS:
        summary["across_seeds"][arm]={}
        for field in ("aggregate_accept_ratio","retention","mean_kept_rows"):
            values=[v["primary"]["assessment"][field] for name,v in summaries.items() if name.startswith(arm+"_seed")]
            summary["across_seeds"][arm][field]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if len(values)>1 else None}
    save_json("summary.json",summary)
    save_json("assessment_descriptive_curves.json",curves)
    np.savez(args.output/"assessment_predictions.npz",**predictions,**{k+"_kept":v for k,v in choices.items()},
             accepted=accepted[ass],prompt_ids=prompts,raw_confidence=raw[ass])
    backup_file(args.output/"assessment_predictions.npz",args.backup/"assessment_predictions.npz")
    files=sorted(p.name for p in args.output.iterdir() if p.is_file())
    save_json("COMPLETE.json",{"complete":True,"binding":{name:sha256(args.output/name) for name in files}})
    print("COMPLETE",json.dumps({"across_seeds":summary["across_seeds"],"counts":config["counts"]}),flush=True)


if __name__=="__main__":
    main()
