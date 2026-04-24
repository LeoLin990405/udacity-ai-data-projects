#!/bin/bash
# Chain-run: 5-step verify → 100-step full training on M5 MPS
set -e
cd "$(dirname "$0")"

echo "=== [$(date +%H:%M:%S)] STAGE 1: 5-step verify ==="
MAX_STEPS=5 python3 -u run_grpo_mps.py 2>&1 | tee verify.log
echo "=== [$(date +%H:%M:%S)] STAGE 1 done ==="

echo "=== [$(date +%H:%M:%S)] STAGE 2: 100-step full training ==="
MAX_STEPS=100 python3 -u run_grpo_mps.py 2>&1 | tee full.log
echo "=== [$(date +%H:%M:%S)] STAGE 2 done ==="

echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE ==="
