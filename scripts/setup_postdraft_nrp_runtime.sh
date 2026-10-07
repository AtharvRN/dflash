#!/usr/bin/env bash
# Isolated environment; never downgrade packages in the pod's base environment.
set -euo pipefail
task_env="${DFLASH_SETUP_ENV:-/tmp/predraft-latency-env}"
test ! -e "$task_env"
/opt/sglang/bin/python -m venv --system-site-packages "$task_env"
"$task_env/bin/python" -c 'import pathlib,sys; p=pathlib.Path(sys.prefix)/"lib/python3.12/site-packages/sglang_base.pth"; p.write_text("/opt/sglang/lib/python3.12/site-packages\n")'
"$task_env/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'triton==3.6.0'
"$task_env/bin/python" -m pip install 'https://github.com/sgl-project/whl/releases/download/v0.4.3/sglang_kernel-0.4.3+cu130-cp310-abi3-manylinux2014_x86_64.whl#sha256=4aaf19fa0ba3dbdc833587b0d0374d3f1c9db505b8cb114cb681cbc1c1d0b50f'
PYTHONPATH= "$task_env/bin/python" -c 'import torch,torchvision,transformers,sgl_kernel; print(torch.__version__,torchvision.__version__,transformers.__version__); assert torch.__version__ == "2.11.0+cu130"; assert hasattr(sgl_kernel,"fp8_blockwise_scaled_mm")'
