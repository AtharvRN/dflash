"""Download the exact frozen checkpoint revisions used in the DFlash experiments."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root=Path(os.environ['DFLASH_ROOT'])
specs={
    'target':('Qwen/Qwen3-4B','1cfa9a7208912126459214e8b04321603b3df60c'),
    'draft':('z-lab/Qwen3-4B-DFlash-b16','b74e3a329c4d963783143b1e970d95b002be72bd'),
}
paths={}
for kind,(repo,revision) in specs.items():
    print(f'Downloading {kind}: {repo}@{revision}',flush=True)
    path=snapshot_download(repo_id=repo,revision=revision,
        cache_dir=str(root/'hf'/'hub'),max_workers=4)
    paths[kind]={'repo':repo,'revision':revision,'path':path}
    print(f'Ready {kind}: {path}',flush=True)
destination=root/'models.json'
if destination.exists() and json.loads(destination.read_text())!=paths:
    raise RuntimeError('Refusing conflicting model registry')
destination.write_text(json.dumps(paths,indent=2)+'\n')
