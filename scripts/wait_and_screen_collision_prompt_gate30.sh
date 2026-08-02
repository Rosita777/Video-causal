#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while :; do
  count0="$(find outputs/collision_prompt_gate30_wan/shard_0/videos -type f -name '*.mp4' 2>/dev/null | wc -l)"
  count1="$(find outputs/collision_prompt_gate30_wan/shard_1/videos -type f -name '*.mp4' 2>/dev/null | wc -l)"
  echo "$(date -Iseconds) shard0=${count0}/15 shard1=${count1}/15"
  if [[ "$count0" -ge 15 && "$count1" -ge 15 ]]; then
    break
  fi
  sleep 30
done

models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py \
  --run-manifest data/collision_prompt_gate30_run_manifest.csv \
  --generation-root outputs/collision_prompt_gate30_wan \
  --output-csv data/collision_prompt_gate30_auto_screen.csv \
  --contact-sheet-dir outputs/collision_prompt_gate30_contact_sheets
