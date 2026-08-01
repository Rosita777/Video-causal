#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/waterdrop_train_pilot40_wan"
LOG_PATH="$PROJECT_ROOT/logs/waterdrop_train_pilot40_watcher.log"
cd "$PROJECT_ROOT"
while true; do
  count="$(find "$OUTPUT_ROOT" -type f -name '*.mp4' 2>/dev/null | wc -l)"
  printf '%s videos=%s/40\n' "$(date -Iseconds)" "$count" >>"$LOG_PATH"
  [[ "$count" -eq 40 ]] && break
  sleep 20
done
sleep 20
models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py \
  --run-manifest data/waterdrop_train_pilot40_run_manifest.csv \
  --generation-root outputs/waterdrop_train_pilot40_wan \
  --output-csv data/waterdrop_train_pilot40_auto_screen.csv \
  --contact-sheet-dir outputs/waterdrop_train_pilot40_auto_contact_sheets \
  >>"$LOG_PATH" 2>&1
echo "auto_screen_complete $(date -Iseconds)" >>"$LOG_PATH"
