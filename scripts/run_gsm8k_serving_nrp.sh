#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_postdraft_serving_nrp.sh "$@"
"${DFLASH_PYTHON:-/tmp/predraft-latency-env/bin/python}" scripts/score_gsm8k_serving.py "$DFLASH_SERVING_OUTPUT"
