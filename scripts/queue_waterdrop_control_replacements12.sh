#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_LOG="$PROJECT_ROOT/logs/waterdrop_control_replacements12_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"
cd "$PROJECT_ROOT"

while true; do
  mapfile -t available < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F, '{gsub(/ /, "", $0); if ($2 >= 45000 && $3 <= 20) print $1}' \
      | head -3
  )
  printf '%s available_gpus=%s\n' "$(date -Iseconds)" "${available[*]:-none}" >>"$QUEUE_LOG"
  if [[ "${#available[@]}" -ge 2 ]]; then break; fi
  sleep 60
done

export GPU_SHARD0="${available[0]}"
export GPU_SHARD1="${available[1]}"
if [[ "${#available[@]}" -ge 3 ]]; then
  export GPU_SHARD2="${available[2]}"
  export TWO_STAGE=0
else
  export GPU_SHARD2="$GPU_SHARD0"
  export TWO_STAGE=1
fi
echo "starting generation on GPUs $GPU_SHARD0 $GPU_SHARD1; two_stage=$TWO_STAGE" >>"$QUEUE_LOG"
bash scripts/run_waterdrop_control_replacements12_wan.sh >>"$QUEUE_LOG" 2>&1

models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py \
  --run-manifest data/waterdrop_control_replacements12_run_manifest.csv \
  --generation-root outputs/waterdrop_control_replacements12_wan \
  --output-csv data/waterdrop_control_replacements12_auto_screen.csv \
  --contact-sheet-dir outputs/waterdrop_control_replacements12_auto_contact_sheets \
  >>"$QUEUE_LOG" 2>&1
echo "auto_screen_complete $(date -Iseconds)" >>"$QUEUE_LOG"
