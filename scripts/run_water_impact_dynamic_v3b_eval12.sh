#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
PROMPTS="prompts/water_impact_dynamic_v1/eval12.prompts"
V3B_CHECKPOINT="outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/checkpoint-000200"
V3B_OUTPUT="outputs/water_impact_dynamic_v3b/eval12_target_prompt_teacher_scale4_v1_ckpt200_scale1p25"
PUBLIC_REVIEW_DIR="experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_public"
PRIVATE_REVIEW_DIR="experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_blind_review_v3_private"
SCORE_DIR="experiments/water_impact_dynamic_eval12/v3b_target_prompt_teacher_scores_v3"

if [[ "$MODEL" != "models/Wan2.1-T2V-1.3B-Diffusers" || "$DEVICE" != "cuda" ]]; then
  echo "v3b eval12 requires the frozen Wan model path and cuda device" >&2
  exit 1
fi

preflight() {
  "$PYTHON" scripts/water_impact_dynamic_v3b_eval_protocol.py
}

case "${1:-}" in
  preflight)
    preflight
    ;;
  v3b)
    preflight
    if ! mkdir "$V3B_OUTPUT" 2>/dev/null; then
      echo "refusing to reuse or race on output path: $V3B_OUTPUT" >&2
      exit 1
    fi
    printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$V3B_OUTPUT/.run_reservation"
    SEEDS="$($PYTHON - <<'PY'
import csv

with open("data/water_impact_dynamic_v1/eval12.csv", newline="", encoding="utf-8") as handle:
    print(",".join(row["seed"] for row in csv.DictReader(handle)))
PY
)"
    "$PYTHON" scripts/generate_wan_clean.py \
      --baseline clean \
      --prompts "$PROMPTS" \
      --output-dir "$V3B_OUTPUT" \
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
      --lora-path "$V3B_CHECKPOINT" \
      --lora-scale 1.25
    PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
from pathlib import Path
from water_impact_dynamic_v3b_eval_protocol import (
    V3B_RUN,
    load_frozen_inputs,
    load_generation_run,
)

root = Path.cwd()
rows, _ = load_frozen_inputs(root)
manifest, _, videos = load_generation_run(root, V3B_RUN, "v3b", rows)
print(f"Validated frozen v3b generation: {manifest} ({len(videos)} videos)")
PY
    ;;
  compare)
    "$PYTHON" scripts/build_water_impact_dynamic_v3b_blind_review.py
    ;;
  score)
    "$PYTHON" scripts/score_water_impact_dynamic_v3b_eval12.py \
      --review "$PUBLIC_REVIEW_DIR/blind_review.csv" \
      --answer-key "$PRIVATE_REVIEW_DIR/answer_key.csv" \
      --review-manifest "$PRIVATE_REVIEW_DIR/review_manifest.json" \
      --output-dir "$SCORE_DIR"
    ;;
  *)
    echo "usage: $0 {preflight|v3b|compare|score}" >&2
    exit 2
    ;;
esac
