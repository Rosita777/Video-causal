#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCREEN_CSV="${SCREEN_CSV:-$PROJECT_ROOT/data/waterdrop_prompt_bank_v2_auto_screen.csv}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/logs/waterdrop_v2_preliminary_pairs.log}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$LOG_PATH")"
while [[ ! -f "$SCREEN_CSV" ]]; do
  sleep 30
done
models/.wan-runtime/bin/python scripts/build_waterdrop_v2_preliminary_pairs.py >>"$LOG_PATH" 2>&1
echo "preliminary_pairs_complete $(date -Iseconds)" >>"$LOG_PATH"
