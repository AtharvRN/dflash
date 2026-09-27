# UCSD workstation

SSH: `zekaili@tianhaowang-gpu0.ucsd.edu` (host reports `wth-gpu-01`).

- Code: `/home/zekaili/atharv/dflash`
- Scratch root: `/data/scratch/zekaili/atharv/dflash`
- Environment: scratch root `/envs/main`
- Models: scratch root `/hf`; pinned paths recorded in `/models.json`
- Imported data: scratch root `/data/dflashv2_data`
- New runs: scratch root `/runs`; working copies: `/work`

Activate with:

```bash
source /home/zekaili/atharv/dflash/scripts/workstation_env.sh
nvidia-smi
# Select an available GPU explicitly before a new run:
export CUDA_VISIBLE_DEVICES=4
```

GPU 4 was idle at setup; this is not a reservation. GPUs are shared. The host
has eight RTX PRO 6000 Blackwell Server Edition cards (~96 GB each). Home and
scratch share the same /data filesystem; scratch is not an independent backup.
The source PVC data remains unchanged.

Python 3.12.12, PyTorch 2.13.0+cu130, Transformers 4.57.1. Exact installed packages
are in scratch root `/environment.freeze.txt`. All dependency checks passed.
Nine unit tests passed. A synthetic real-model SDPA/BF16 B2/B16/B20 smoke passed
reverse-order checks and all three canonical comparisons on GPU 4 (compute
capability 12.0). Its report is scratch root `/setup_smoke.json`. This is a
compatibility check, not a cross-hardware benchmark or claim of bitwise parity.

Git deployment uses HTTPS because this workstation does not have the author's
GitHub SSH key. Work is on `codex/block-headroom-20260926`. Do not copy credentials
or modify global shell/Git settings to make deployment work.

Imported data:

- Original 99,987-conversation parquet at scratch root `/data/train-00000-of-00001.parquet`.
- Canonical messages manifest and fixed prompt split directory under imported data.
- `runs/prefusion_pilot_20260915` (completed pilot).
- `runs/prefusion_100k_20260915` (original partial expansion, still incomplete).
- `runs/block_headroom_128p_20260926` (completed A100 experiment).

The large historical trace pool and full last-fused cache are not migrated.
Preserve embedded original paths/provenance in imported artifacts; relocation
does not justify changing audit records or treating the partial expansion as
complete. Data-dependent older scripts may need explicit relocated path arguments.

The initial tar stream reset while copying the last results directory. That
directory was retransferred. Its artifact audit passed: 128 prompt receipts,
875 states, 871 eligible states, 245 hashed data files, valid prefix hashes,
finite paired features, and consistent reference trajectories. The parquet
SHA256 matches the original:
`0b66428455f6637c0ad6e6c9bd975f4b7dddff2657e09b3adb4036482cfe9c87`.

No new research collection/training or recurring monitoring was started during
workstation setup. Source the environment and explicitly choose an available GPU
when starting future jobs; the portable launcher refuses missing GPU selection.
