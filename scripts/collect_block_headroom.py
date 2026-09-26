"""Collect actual integer-block outcomes on identical canonical training states."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reuse the image's Torch without importing its newer Transformers first.
image_packages = '/opt/sglang/lib/python3.12/site-packages'
if Path(image_packages).exists() and image_packages not in sys.path:
    sys.path.append(image_packages)

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlashDraftModel
from scripts.diagnose_dflash_paired_lengths import atomic_json, run_prompt
from scripts.block_headroom import select_training_prompts, summarize


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class NoPolicies:
    def predict(self, fused):
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'split-dir', 'model', 'draft-model', 'output-dir', 'persistent-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--per-source', type=int, default=32)
    p.add_argument('--states-per-prompt', type=int, default=8)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--max-prompt-tokens', type=int, default=2048)
    p.add_argument('--seed', type=int, default=926)
    p.add_argument('--reverse-check-states', type=int, default=8)
    p.add_argument('--canonical-check-states', type=int, default=16)
    p.add_argument('--max-seconds', type=int, default=10800)
    p.add_argument('--stage-models', type=Path)
    args = p.parse_args()
    args.blocks = list(range(2,21))
    args.max_states = 4*args.per_source*args.states_per_prompt
    if args.max_new_tokens < 32 or min(args.per_source,args.states_per_prompt,args.max_seconds) < 1:
        raise ValueError('Invalid limits')
    for destination in (args.output_dir, args.persistent_dir):
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f'Refusing nonempty destination: {destination}')
        destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if transformers.__version__ != '4.57.1':
        raise RuntimeError('Expected pinned Transformers 4.57.1')
    selected = select_training_prompts(args.manifest, args.split_dir, args.per_source, args.seed)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'prompt_ids':[int(r['manifest_index']) for r in selected],
        'sources': {s:sum(r['source']==s for r in selected) for s in sorted({r['source'] for r in selected})},
        'prompt_content_hashes':{str(r['manifest_index']):r['content_sha256'] for r in selected},
        'split':'canonical train only; cross-split exact message duplicates excluded',
        'temperature':0, 'thinking':False, 'dtype':'bfloat16', 'attention':'sdpa',
        'torch':torch.__version__, 'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(), 'tf32':False,
        'trajectory':'fixed B16; alternative outcomes on identical states, not closed-loop',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{str(path.relative_to(Path.cwd())):sha256(path) for path in
                         [Path.cwd()/'scripts'/name for name in ('collect_block_headroom.py','block_headroom.py','diagnose_dflash_paired_lengths.py')]},
        'input_hashes':{str(path):sha256(path) for path in [args.manifest,args.split_dir/'train_prompt_ids.json',args.split_dir/'val_prompt_ids.json']}})
    atomic_json(args.output_dir/'config.json', config)
    executor = ThreadPoolExecutor(max_workers=1)
    futures = []
    def backup(paths):
        for path in paths:
            temporary = args.persistent_dir/(path.name+'.tmp')
            with path.open('rb') as source, temporary.open('wb') as destination:
                shutil.copyfileobj(source,destination,4*1024*1024)
                destination.flush()
                os.fsync(destination.fileno())
            if sha256(temporary) != sha256(path):
                raise RuntimeError(f'Backup hash mismatch: {path}')
            temporary.replace(args.persistent_dir/path.name)
    def queue(paths):
        for future in futures:
            if future.done():
                future.result()
        futures.append(executor.submit(backup, paths))
    queue([args.output_dir/'config.json'])
    started = time.monotonic()
    try:
        if args.stage_models:
            for key in ('model','draft_model'):
                original = getattr(args,key)
                destination = args.stage_models/key
                if not destination.exists():
                    print(f'STAGING {original} -> {destination}',flush=True)
                    shutil.copytree(original,destination,symlinks=False)
                # Verify checkpoint identity, including staged model weight files.
                for source in original.iterdir():
                    if source.is_file() and (source.suffix in ('.json','.safetensors') or source.name=='merges.txt'):
                        if sha256(source) != sha256(destination/source.name):
                            raise RuntimeError(f'Staged model mismatch: {source}')
                setattr(args,key,destination)
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        target = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
        draft = DFlashDraftModel.from_pretrained(args.draft_model,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
        print('MODELS_READY',flush=True)
        all_rows, progress_records, receipts = [], [], []
        collection_start = time.monotonic()
        for i,row in enumerate(selected):
            if time.monotonic()-started >= args.max_seconds:
                print('TIME_BUDGET_STOP',flush=True)
                break
            batch, progress = run_prompt(args,row,target,draft,tokenizer,NoPolicies(),len(all_rows))
            shard = args.output_dir/f"prompt_{row['manifest_index']}.json"
            atomic_json(shard,{'progress':progress,'states':[r for r,_ in batch]})
            paths = [shard]
            if batch:
                feature = args.output_dir/f"prompt_{row['manifest_index']}_fused.npy"
                np.save(feature,np.stack([f for _,f in batch]),allow_pickle=False)
                paths.append(feature)
            receipt = {'prompt_id':int(row['manifest_index']),'source':row['source'],
                       'states':len(batch),'files':{path.name:sha256(path) for path in paths}}
            receipt_path = args.output_dir/f"receipt_{row['manifest_index']}.json"
            atomic_json(receipt_path,receipt)
            queue(paths+[receipt_path])
            receipts.append(receipt)
            all_rows.extend(r for r,_ in batch)
            progress_records.append(progress)
            status={'prompts_processed':i+1,'prompts_planned':len(selected),'states':len(all_rows),
                    'elapsed_s':time.monotonic()-collection_start,'latest':progress}
            atomic_json(args.output_dir/'progress.json',status)
            print(json.dumps(status),flush=True)
        summary=summarize(all_rows)
        summary.update({'sample_complete':len(progress_records)==len(selected),
                        'prompt_progress':progress_records,'elapsed_s':time.monotonic()-started})
        atomic_json(args.output_dir/'headroom_summary.json',summary)
        atomic_json(args.output_dir/'receipts.json',receipts)
        queue([args.output_dir/'headroom_summary.json',args.output_dir/'receipts.json',args.output_dir/'progress.json'])
        for future in futures:
            future.result()
        atomic_json(args.output_dir/'COMPLETE.json',{'all_backups_verified':True,
                    'sample_complete':summary['sample_complete'],'states':len(all_rows),
                    'summary_sha256':sha256(args.output_dir/'headroom_summary.json')})
        backup([args.output_dir/'COMPLETE.json'])
        print(json.dumps({'complete':True,'sample_complete':summary['sample_complete'],
                          'states':len(all_rows),'eligible':summary['eligible_states'],'checks':summary.get('checks')}),flush=True)
    finally:
        executor.shutdown(wait=True)


if __name__=='__main__':
    main()
