"""Collect actual-block histories, fit a value table, or run greedy DFlash rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash.history_policy import HistoryValueTable, blocks_checked, evaluate_rows, select_block


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_prompts(manifest, split_dir, group, pilot_manifest, limit, seed):
    train = set(map(str, json.loads((split_dir / "train_prompt_ids.json").read_text())["train_prompt_ids"]))
    val = set(map(str, json.loads((split_dir / "val_prompt_ids.json").read_text())["val_prompt_ids"]))
    if train & val:
        raise ValueError("canonical prompt split overlaps")
    memberships = {"calibration": set(), "assessment": set()}
    if pilot_manifest is not None:
        pilot = json.loads(pilot_manifest.read_text())
        memberships = {g: {str(s["prompt_id"]) for s in pilot["shards"]
                           if s["group"] == g and s.get("rows", 0)} for g in memberships}
        if memberships["calibration"] & memberships["assessment"]:
            raise ValueError("pilot groups overlap")
        if not (memberships["calibration"] | memberships["assessment"]) <= val:
            raise ValueError("pilot membership is not canonical validation")
    if group == "train":
        allowed = train
    else:
        if pilot_manifest is None:
            raise ValueError("fixed validation membership requires --pilot-manifest")
        allowed = memberships[group]
    rows, contents, seen = [], {}, set()
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            pid = str(row["manifest_index"])
            if pid in seen:
                raise ValueError("duplicate manifest prompt ID")
            seen.add(pid)
            h = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
            partition = "train" if pid in train else "validation" if pid in val else "other"
            for subgroup, ids in memberships.items():
                if pid in ids:
                    partition = subgroup
            contents.setdefault(h, set()).add(partition)
            if pid in allowed:
                rows.append({**row, "content_sha256": h})
    if not allowed <= seen:
        raise ValueError("manifest is missing split members")
    unique, eligible = set(), []
    for row in rows:
        h = row["content_sha256"]
        if len(contents[h]) == 1 and h not in unique:
            unique.add(h)
            eligible.append(row)
    random.Random(seed).shuffle(eligible)
    return eligible[:limit]


def load_runs(paths):
    rows, identity, hashes, membership = [], None, {}, {}
    for path in paths:
        config = json.loads((path / "config.json").read_text())
        if not (path / "summary.json").exists() or config["command"] != "collect":
            raise ValueError("fit/replay requires a completed counterfactual collection")
        if identity is not None and identity != config["model_identity"]:
            raise ValueError("mixed target/drafter identities")
        identity = config["model_identity"]
        file_hash = digest(path / "cycles.jsonl")
        if json.loads((path / "summary.json").read_text())["cycles_sha256"] != file_hash:
            raise ValueError("collection receipt hash mismatch")
        hashes[str(path / "cycles.jsonl")] = file_hash
        with (path / "cycles.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                pid, group = str(row["prompt_id"]), row["group"]
                if pid in membership and membership[pid] != group:
                    raise ValueError("prompt membership changed across runs")
                membership[pid] = group
                if row["eligible"]:
                    rows.append(row)
    return rows, {"model_identity": identity, "source_hashes": hashes}


def selection(args):
    costs = None
    if args.cost_profile:
        profile = json.loads(args.cost_profile.read_text())
        if profile.get("units") != "ms" or profile.get("scope") != "whole_cycle":
            raise ValueError("cost profile must contain whole-cycle milliseconds")
        costs = {int(b): t for b, t in profile["costs_ms"][str(args.concurrency)].items()}
    return dict(mode=args.mode, alpha=args.alpha, costs_ms=costs, rho=args.rho,
                history_free=args.history_free)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "replay", "collect", "rollout"):
        c = sub.add_parser(name)
        c.add_argument("--output", type=Path, required=True)
        if name in ("fit", "collect"):
            c.add_argument("--blocks", required=True, help="Sorted integers including anchor; e.g. 4,8,12,16")
        if name in ("fit", "replay"):
            c.add_argument("--runs", type=Path, nargs="+", required=True)
        if name == "fit":
            c.add_argument("--bins", type=int, default=8)
            c.add_argument("--prior-count", type=float, default=20.)
        if name in ("replay", "rollout"):
            c.add_argument("--table", type=Path, required=True)
            c.add_argument("--mode", choices=("retention", "priced"), default="retention")
            c.add_argument("--alpha", type=float, default=.96)
            c.add_argument("--rho", type=float)
            c.add_argument("--cost-profile", type=Path)
            c.add_argument("--concurrency", type=int, default=1)
            c.add_argument("--history-free", action="store_true", help="Ablation: use global progress means, ignoring history")
        if name in ("collect", "rollout"):
            c.add_argument("--manifest", type=Path, required=True)
            c.add_argument("--split-dir", type=Path, required=True)
            c.add_argument("--pilot-manifest", type=Path)
            c.add_argument("--models", type=Path, required=True, help="Existing pinned local models.json")
            c.add_argument("--group", choices=("train", "calibration", "assessment"), required=True)
            c.add_argument("--limit-prompts", type=int, default=16)
            c.add_argument("--max-new-tokens", type=int, default=256)
            c.add_argument("--max-prompt-tokens", type=int, default=2048)
            c.add_argument("--max-cycles", type=int, default=32)
            c.add_argument("--seed", type=int, default=1007)
            c.add_argument("--device", default="cuda:0")
            c.add_argument("--fixed-block", type=int, help="Fixed behavior/control instead of random collection or adaptive rollout")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing existing output: preserve previous evidence")
    if args.command in ("fit", "collect"):
        blocks = blocks_checked(tuple(map(int, args.blocks.split(","))))
    else:
        table = HistoryValueTable(json.loads(args.table.read_text()))
        blocks = table.blocks
        options = selection(args)
        select_block(table, {"entropy_count": 0, "certainty": None,
                            "previous_block": None, "previous_full": None}, **options)
    if args.command in ("fit", "replay"):
        rows, provenance = load_runs(args.runs)
        if args.command == "fit":
            train = [r for r in rows if r["group"] == "train"]
            result = HistoryValueTable.fit(train, blocks, bins=args.bins,
                                          prior_count=args.prior_count, provenance=provenance).payload
        else:
            if provenance["model_identity"] != table.payload["provenance"]["model_identity"]:
                raise ValueError("table/model identity mismatch")
            result = {g: evaluate_rows(table, [r for r in rows if r["group"] == g], **options)
                      for g in ("calibration", "assessment") if any(r["group"] == g for r in rows)}
            if not result:
                raise ValueError("no evaluation groups")
            result.update(table_sha256=digest(args.table), selection=options, provenance=provenance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
        print(args.output)
        return

    if min(args.limit_prompts, args.max_new_tokens, args.max_prompt_tokens) < 1 or args.max_cycles < 0:
        raise ValueError("invalid run bounds")
    if args.fixed_block is not None and args.fixed_block not in blocks:
        raise ValueError("fixed block must be one of the candidate sizes")
    prompts = select_prompts(args.manifest, args.split_dir, args.group, args.pilot_manifest,
                             args.limit_prompts, args.seed)
    if not prompts:
        raise ValueError("no eligible prompts")
    models = json.loads(args.models.read_text())
    for k in ("target", "draft"):
        revision = models[k].get("revision", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision) or not models[k].get("repo"):
            raise ValueError("models.json must record repo and immutable 40-character revision for both models")
    identity = {k: {"repo": models[k]["repo"], "revision": models[k]["revision"],
                    "config_sha256": digest(Path(models[k]["path"]) / "config.json")}
                for k in ("target", "draft")}
    if args.command == "rollout":
        if identity != table.payload["provenance"]["model_identity"]:
            raise ValueError("table/model identity mismatch")
        if any(str(r["manifest_index"]) in table.payload["fit_prompt_ids"] for r in prompts):
            raise ValueError("rollout prompts overlap fitting")
    # Lazy GPU imports keep fitting and matched-state analysis CPU-only.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel
    from dflash.history_generate import generate_history
    if transformers.__version__ != "4.57.1":
        raise ValueError("reference implementation requires Transformers 4.57.1")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    target = AutoModelForCausalLM.from_pretrained(models["target"]["path"], local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa").to(args.device).eval().requires_grad_(False)
    draft = DFlashDraftModel.from_pretrained(models["draft"]["path"], local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa").to(args.device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(models["target"]["path"], local_files_only=True)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(blocks=list(blocks), model_identity=identity, model_locations=models, torch=torch.__version__,
                  transformers=transformers.__version__, manifest_sha256=digest(args.manifest),
                  selected_prompt_ids=[str(r["manifest_index"]) for r in prompts],
                  split_hashes={p.name: digest(p) for p in (args.split_dir/"train_prompt_ids.json", args.split_dir/"val_prompt_ids.json")},
                  scope="greedy Transformers reference; no SGLang speedup claim")
    config["source_hashes"] = {str(p): digest(p) for p in
        (Path(__file__), Path(__file__).resolve().parents[1]/"dflash/history_policy.py",
         Path(__file__).resolve().parents[1]/"dflash/history_generate.py")}
    if args.pilot_manifest:
        config["pilot_manifest_sha256"] = digest(args.pilot_manifest)
    if args.command == "rollout":
        config.update(table_sha256=digest(args.table), selection=options)
    save(args.output/"config.json", config)
    summaries = []
    eos = target.generation_config.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    with (args.output/"cycles.jsonl").open("x") as stream:
        for row in prompts:
            pid = str(row["manifest_index"])
            ids = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True,
                enable_thinking=False, tokenize=True, return_tensors="pt").to(args.device)
            if ids.shape[1] > args.max_prompt_tokens:
                summaries.append({"prompt_id": pid, "skipped": "prompt_length"})
                continue
            rng = random.Random(f"{args.seed}:{pid}")

            def choose(h, cycle):
                if args.fixed_block is not None:
                    return args.fixed_block
                return rng.choice(blocks) if args.command == "collect" else select_block(table, h, **options)[0]

            result = generate_history(draft, target, ids, blocks=blocks, choose=choose,
                max_new_tokens=args.max_new_tokens, max_cycles=args.max_cycles,
                stop_token_ids=eos, collect=args.command == "collect", prompt_id=pid, group=args.group)
            for record in result["cycles"]:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            summaries.append({"prompt_id": pid, "output_ids": result["output_ids"][0, ids.shape[1]:].tolist(),
                              "cycles": len(result["cycles"]), "eligible": sum(r["eligible"] for r in result["cycles"]),
                              "elapsed_s": result["elapsed_s"], "cycle_limited": result["cycle_limited"]})
            save(args.output/"progress.json", summaries)
            print(pid, "cycles", summaries[-1]["cycles"], flush=True)
    save(args.output/"summary.json", {"scope": config["scope"], "prompts": summaries,
         "cycles_sha256": digest(args.output/"cycles.jsonl")})


if __name__ == "__main__":
    main()
