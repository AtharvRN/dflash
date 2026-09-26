"""Read-only integrity/alignment audit of a completed paired headroom run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def validate_states(rows, blocks):
    previous=None
    seen=set()
    for row in rows:
        prefix=row['prefix_token_ids']
        if len(prefix)!=row['prefix_length']+1 or any(not isinstance(t,int) or t<0 for t in prefix):
            raise ValueError('Invalid prefix token IDs/length')
        if hashlib.sha256(np.asarray(prefix,dtype=np.int64).tobytes()).hexdigest()!=row['prefix_sha256']:
            raise ValueError('Prefix hash mismatch')
        if row['cycle'] in seen:
            raise ValueError('Duplicate prompt/cycle')
        seen.add(row['cycle'])
        if sorted(map(int,row['outcomes']))!=blocks:
            raise ValueError('Incomplete block matrix')
        for b in blocks:
            outcome=row['outcomes'][str(b)]
            if len(outcome['draft_ids'])!=b-1 or not isinstance(outcome['accepted'],int) or not 0<=outcome['accepted']<=b-1:
                raise ValueError('Invalid draft/acceptance length')
        if previous:
            if row['cycle']<=previous['cycle'] or row['prefix_length']<=previous['prefix_length']:
                raise ValueError('Nonmonotone reference states')
            outcome=previous['outcomes']['16']
            committed=previous['prefix_token_ids']+outcome['draft_ids'][:outcome['accepted']]
            if prefix[:len(committed)]!=committed:
                raise ValueError('B16 committed prefix changed between sampled states')
        previous=row


def audit(root):
    root=Path(root)
    config=json.loads((root/'config.json').read_text())
    complete=json.loads((root/'COMPLETE.json').read_text())
    summary=json.loads((root/'headroom_summary.json').read_text())
    receipts=json.loads((root/'receipts.json').read_text())
    if sha256(root/'headroom_summary.json')!=complete['summary_sha256']:
        raise ValueError('Summary completion binding mismatch')
    planned=set(map(int,config['prompt_ids']))
    if len(planned)!=len(config['prompt_ids']):
        raise ValueError('Duplicate planned prompt')
    seen=set()
    all_rows=[]
    files_checked=0
    for receipt in receipts:
        pid=receipt['prompt_id']
        if pid not in planned or pid in seen:
            raise ValueError('Unexpected or duplicate receipt prompt')
        seen.add(pid)
        if json.loads((root/f'receipt_{pid}.json').read_text())!=receipt:
            raise ValueError('Receipt manifest mismatch')
        for name,digest in receipt['files'].items():
            if Path(name).name!=name or sha256(root/name)!=digest:
                raise ValueError('Receipt file hash mismatch')
            files_checked+=1
        shard=json.loads((root/f'prompt_{pid}.json').read_text())
        rows=shard['states']
        if len(rows)!=receipt['states'] or any(int(r['prompt_id'])!=pid or r['source']!=receipt['source'] for r in rows):
            raise ValueError('Prompt/source/row-count mismatch')
        validate_states(rows,config['blocks'])
        if rows:
            features=np.load(root/f'prompt_{pid}_fused.npy',allow_pickle=False)
            if features.shape!=(len(rows),2560) or features.dtype!=np.float16 or not np.isfinite(features).all():
                raise ValueError('Invalid paired fused features')
        all_rows.extend(rows)
    if complete['sample_complete'] and seen!=planned:
        raise ValueError('Completed run is missing planned prompt receipts')
    if len(all_rows)!=summary['states'] or len(all_rows)!=complete['states']:
        raise ValueError('Summary row-count mismatch')
    if sum(r['eligible'] for r in all_rows)!=summary['eligible_states']:
        raise ValueError('Summary eligible-count mismatch')
    return {'passed':True,'prompts':len(seen),'states':len(all_rows),
            'eligible_states':sum(r['eligible'] for r in all_rows),'files_hash_checked':files_checked,
            'summary_sha256':sha256(root/'headroom_summary.json'),
            'prefix_hashes_verified':True,'reference_trajectory_continuity_verified':True,
            'feature_shapes_and_finiteness_verified':True,
            'note':'Artifact consistency audit; not a re-execution of GPU outcomes or proof of learnability.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    print(json.dumps(audit(args.run),indent=2))
