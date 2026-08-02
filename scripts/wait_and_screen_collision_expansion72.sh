#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/collision_expansion72_wan/shard_0/videos outputs/collision_expansion72_wan/shard_1/videos
while :; do
  count0="$(find outputs/collision_expansion72_wan/shard_0/videos -type f -name '*.mp4' 2>/dev/null | wc -l)"
  count1="$(find outputs/collision_expansion72_wan/shard_1/videos -type f -name '*.mp4' 2>/dev/null | wc -l)"
  echo "$(date -Iseconds) shard0=${count0}/36 shard1=${count1}/36"
  if [[ "$count0" -ge 36 && "$count1" -ge 36 ]]; then break; fi
  sleep 30
done
models/.wan-runtime/bin/python scripts/build_waterdrop_v2_auto_screen.py \
  --run-manifest data/collision_expansion72_run_manifest.csv \
  --generation-root outputs/collision_expansion72_wan \
  --output-csv data/collision_expansion72_auto_screen.csv \
  --contact-sheet-dir outputs/collision_expansion72_contact_sheets
