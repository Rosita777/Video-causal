#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-250}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_prompt_bank_v2_simple_wan}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/logs/waterdrop_prompt_bank_v2_auto_screen.log}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG_PATH")"

while true; do
  count="$(find "$OUTPUT_ROOT" -type f -name '*.mp4' 2>/dev/null | wc -l)"
  jobs="$(pgrep -af 'generate_wan_clean.py.*waterdrop_prompt_bank_v2_simple_shard' | grep -v pgrep | wc -l || true)"
  printf '%s videos=%s/%s generation_jobs=%s\n' "$(date -Iseconds)" "$count" "$EXPECTED_VIDEOS" "$jobs" >>"$LOG_PATH"
  if [[ "$count" -eq "$EXPECTED_VIDEOS" && "$jobs" -eq 0 ]]; then
    break
  fi
  if [[ "$jobs" -eq 0 && "$count" -lt "$EXPECTED_VIDEOS" ]]; then
    echo "Generation stopped before all expected videos were produced" >>"$LOG_PATH"
    exit 1
  fi
  sleep 30
done

models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py >>"$LOG_PATH" 2>&1
echo "auto_screen_complete $(date -Iseconds)" >>"$LOG_PATH"
