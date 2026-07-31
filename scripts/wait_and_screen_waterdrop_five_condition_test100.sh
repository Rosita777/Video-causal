#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/waterdrop_five_condition_test100_wan"
LOG_PATH="$PROJECT_ROOT/logs/waterdrop_five_condition_test100_auto_screen.log"
cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG_PATH")"
while true; do
  count="$(find "$OUTPUT_ROOT" -type f -name '*.mp4' 2>/dev/null | wc -l)"
  jobs="$(pgrep -af 'generate_wan_clean.py.*waterdrop_five_condition_test100_shard' | grep -v pgrep | wc -l || true)"
  printf '%s videos=%s/100 generation_jobs=%s\n' "$(date -Iseconds)" "$count" "$jobs" >>"$LOG_PATH"
  if [[ "$count" -eq 100 && "$jobs" -eq 0 ]]; then break; fi
  if [[ "$jobs" -eq 0 && "$count" -lt 100 ]]; then
    echo "Generation stopped before all expected videos were produced" >>"$LOG_PATH"
    exit 1
  fi
  sleep 30
done
models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py \
  --run-manifest data/waterdrop_five_condition_test100_run_manifest.csv \
  --generation-root outputs/waterdrop_five_condition_test100_wan \
  --output-csv data/waterdrop_five_condition_test100_auto_screen.csv \
  --contact-sheet-dir outputs/waterdrop_five_condition_test100_auto_contact_sheets \
  >>"$LOG_PATH" 2>&1
echo "auto_screen_complete $(date -Iseconds)" >>"$LOG_PATH"
