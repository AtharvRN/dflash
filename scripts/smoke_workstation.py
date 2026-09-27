"""Small real-model compatibility check; does not launch a research experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gpu',type=int,required=True)
args=p.parse_args()
usage=subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
if int(usage.strip())>1024:
    raise RuntimeError('Selected GPU is occupied; choose an idle GPU')
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from dflash.model import DFlashDraftModel
from scripts.diagnose_dflash_paired_lengths import run_prompt
from scripts.collect_block_headroom import NoPolicies

torch.set_num_threads(4)
torch.manual_seed(926)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
root=Path(os.environ['DFLASH_ROOT'])
models=json.loads((root/'models.json').read_text())
target=AutoModelForCausalLM.from_pretrained(models['target']['path'],torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
draft=DFlashDraftModel.from_pretrained(models['draft']['path'],torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
tokenizer=AutoTokenizer.from_pretrained(models['target']['path'],local_files_only=True)
config=SimpleNamespace(max_prompt_tokens=2048,max_new_tokens=64,states_per_prompt=1,
    blocks=[2,16,20],seed=926,reverse_check_states=1,canonical_check_states=1,max_states=1)
row={'manifest_index':0,'source':'synthetic_setup_smoke','messages':[{'role':'user','content':'Explain in two sentences why the sky appears blue.'}]}
states,progress=run_prompt(config,row,target,draft,tokenizer,NoPolicies(),0)
if len(states)!=1 or not states[0][0]['reverse_order_checked'] or states[0][0]['canonical_disagreements']:
    raise RuntimeError('Real-model smoke check failed')
report={'passed':True,'gpu_index':args.gpu,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'compute_capability':torch.cuda.get_device_capability(),
        'canonical_checks':states[0][0]['canonical_checked'],
        'canonical_disagreements':states[0][0]['canonical_disagreements'],
        'accepted_by_block':{b:r['accepted'] for b,r in states[0][0]['outcomes'].items()},
        'note':'Synthetic one-state compatibility test, not benchmark results.'}
(root/'setup_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
