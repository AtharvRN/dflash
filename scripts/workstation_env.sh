# Source this file; it deliberately does not select or reserve a shared GPU.
export DFLASH_REPO=/home/zekaili/atharv/dflash
export DFLASH_ROOT=/data/scratch/zekaili/atharv/dflash
export HF_HOME="$DFLASH_ROOT/hf"
export UV_CACHE_DIR="$DFLASH_ROOT/uv-cache"
export TRITON_CACHE_DIR="$DFLASH_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$DFLASH_ROOT/inductor-cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$DFLASH_REPO"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
source "$DFLASH_ROOT/envs/main/bin/activate"
cd "$DFLASH_REPO"
