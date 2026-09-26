"""Expose only Torch and its base dependencies from this pod's image venv.

Run with the newly created headroom venv Python. Do not expose the image's newer
Transformers or optional kernels/audio/vision packages to the pinned runtime.
"""
from pathlib import Path
import sys
import sysconfig

source=Path('/opt/sglang/lib/python3.12/site-packages')
destination=Path(sysconfig.get_paths()['purelib'])
if not str(destination).startswith('/tmp/dflash-headroom-venv-'):
    raise RuntimeError(f'Unexpected destination: {destination}')
for name in ('torch','torchgen','functorch','triton','nvidia','sympy','mpmath','networkx','jinja2','markupsafe'):
    paths=[source/name]+list(source.glob(name+'-*.dist-info'))
    if not paths[0].exists():
        raise FileNotFoundError(paths[0])
    for path in paths:
        target=destination/path.name
        if target.exists():
            if target.resolve()!=path.resolve():
                raise RuntimeError(f'Refusing conflicting environment package: {target}')
        else:
            target.symlink_to(path,target_is_directory=True)
        print(target, '->', path)
print('Bootstrap complete for',sys.executable)
