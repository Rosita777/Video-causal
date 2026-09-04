#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-/data/xiaohuang_workspace/ljc/Video-causal-v4}"

GPUS="${GPUS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2}"
LOG_ROOT="${LOG_ROOT:-outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_queue_logs_v2}"
mkdir -p "$LOG_ROOT"

exec env -u PYTHONHOME -u PYTHONPATH \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 \
  TOKENIZERS_PARALLELISM=false \
  models/.wan-runtime/bin/python \
  scripts/run_causal_role_erasure_7mechanism_wan_original_v2.py \
  --gpus "$GPUS" \
  --output-root "$OUTPUT_ROOT" \
  --run \
  > "$LOG_ROOT/queue.log" 2>&1
