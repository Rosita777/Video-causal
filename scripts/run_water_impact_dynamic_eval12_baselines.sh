#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHON="models/.wan-runtime/bin/python"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
PROMPTS="prompts/water_impact_dynamic_v1/eval12.prompts"
SEEDS="$($PYTHON - <<'PY'
import csv
rows = list(csv.DictReader(open('data/water_impact_dynamic_v1/eval12.csv')))
print(','.join(row['seed'] for row in rows))
PY
)"
COMMON=(
  --prompts "$PROMPTS" --model "$MODEL" --seeds "$SEEDS"
  --steps 25 --guidance-scale 5 --num-frames 49 --fps 8
  --height 480 --width 832 --dtype bf16 --device cuda
  --vae-slicing --vae-tiling --skip-existing
)

run_negative_prompt() {
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline negative_prompt "${COMMON[@]}" \
    --output-dir outputs/water_impact_dynamic_v1/eval12_negative_prompt
}

run_t2vunlearning() {
  "$PYTHON" scripts/adapters/run_t2vunlearning_wan.py \
    "${COMMON[@]}" \
    --output-dir outputs/water_impact_dynamic_v1/eval12_t2vunlearning
}

run_videoeraser() {
  "$PYTHON" scripts/adapters/run_videoeraser_wan.py \
    "${COMMON[@]}" \
    --output-dir outputs/water_impact_dynamic_v1/eval12_videoeraser
}

case "${1:-}" in
  negative_prompt) run_negative_prompt ;;
  t2vunlearning) run_t2vunlearning ;;
  videoeraser) run_videoeraser ;;
  *) echo "usage: $0 {negative_prompt|t2vunlearning|videoeraser}" >&2; exit 2 ;;
esac
