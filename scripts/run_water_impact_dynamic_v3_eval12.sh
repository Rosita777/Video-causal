#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
PROMPTS="prompts/water_impact_dynamic_v1/eval12.prompts"
BALANCED_NAME="v3_balanced_seeded_ckpt200_scale1p25"
EXPOSURE_NAME="v3_exposure_seeded_ckpt200_scale1p25"

case "${1:-}" in
  balanced)
    NAME="$BALANCED_NAME"
    LORA_PATH="outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_balanced_seeded/checkpoint-000200"
    ;;
  exposure)
    NAME="$EXPOSURE_NAME"
    LORA_PATH="outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_exposure_seeded/checkpoint-000200"
    ;;
  compare)
    exec "$PYTHON" scripts/build_water_impact_dynamic_v3_blind_review.py \
      --original-dir outputs/water_impact_dynamic_v1/eval12_base \
      --balanced-dir "outputs/water_impact_dynamic_v1/eval12_${BALANCED_NAME}" \
      --exposure-dir "outputs/water_impact_dynamic_v1/eval12_${EXPOSURE_NAME}" \
      --output-dir experiments/water_impact_dynamic_eval12/v3_sampling_blind_review \
      --blind-seed 26013
    ;;
  *)
    echo "usage: $0 {balanced|exposure|compare}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="outputs/water_impact_dynamic_v1/eval12_${NAME}"
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
  echo "refusing to reuse or race on output path: $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUTPUT_DIR/.run_reservation"

SEEDS="$("$PYTHON" - <<'PY'
import csv

with open("data/water_impact_dynamic_v1/eval12.csv", newline="", encoding="utf-8") as handle:
    print(",".join(row["seed"] for row in csv.DictReader(handle)))
PY
)"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --seeds "$SEEDS" \
  --steps 25 \
  --guidance-scale 5 \
  --num-frames 49 \
  --fps 8 \
  --height 480 \
  --width 832 \
  --dtype bf16 \
  --device "$DEVICE" \
  --vae-slicing \
  --vae-tiling \
  --lora-path "$LORA_PATH" \
  --lora-scale 1.25
