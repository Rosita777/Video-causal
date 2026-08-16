#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
PROMPTS="prompts/water_impact_dynamic_v1/v3c_fresh_dev24.prompts"
V3B_CHECKPOINT="outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/checkpoint-000200"
V3C_CHECKPOINT="outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1/checkpoint-000200"
ORIGINAL_OUTPUT="outputs/water_impact_dynamic_v3c/fresh_dev24_base"
V3B_OUTPUT="outputs/water_impact_dynamic_v3b/fresh_dev24_target_prompt_teacher_scale4_v1_ckpt200_scale1p25"
V3C_OUTPUT="outputs/water_impact_dynamic_v3c/fresh_dev24_target_prompt_teacher_sigma2_scale4_v1_ckpt200_scale1p25"
PUBLIC_DIR="experiments/water_impact_dynamic_fresh_dev24/v3b_vs_v3c_blind_review_v1_public"
PRIVATE_DIR="experiments/water_impact_dynamic_fresh_dev24/v3b_vs_v3c_blind_review_v1_private"
REVIEWER_A="experiments/water_impact_dynamic_fresh_dev24/v3c_reviewer_a_blind_scores.csv"
REVIEWER_B="experiments/water_impact_dynamic_fresh_dev24/v3c_reviewer_b_blind_scores.csv"
ADJUDICATOR="experiments/water_impact_dynamic_fresh_dev24/v3c_adjudicator_blind_scores.csv"
SCORE_DIR="experiments/water_impact_dynamic_fresh_dev24/v3b_vs_v3c_scores_v1"

if [[ "$MODEL" != "models/Wan2.1-T2V-1.3B-Diffusers" || "$DEVICE" != "cuda" ]]; then
  echo "v3c fresh-dev24 requires the frozen Wan model path and cuda device" >&2
  exit 1
fi

split_preflight() {
  PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PY'
from pathlib import Path
from water_impact_dynamic_v3c_eval_protocol import validate_model_revision, validate_split_registration

root = Path.cwd()
validate_split_registration(root)
validate_model_revision(root)
print("Validated frozen v3c stage-1 split and model revision")
PY
}

stage2_preflight() {
  split_preflight
  "$PYTHON" scripts/register_water_impact_dynamic_v3c_eval_stage2.py --validate
}

seeds() {
  "$PYTHON" - <<'PY'
import csv
with open("data/water_impact_dynamic_v1/v3c_fresh_dev24.csv", newline="", encoding="utf-8") as handle:
    print(",".join(row["seed"] for row in csv.DictReader(handle)))
PY
}

reserve_run() {
  local output="$1"
  PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$output" <<'PY'
from pathlib import Path
import sys
from water_impact_dynamic_v3c_eval_protocol import (
    SPLIT_REGISTRY_SHA256, file_sha256, load_stage2_registration,
)

output = Path(sys.argv[1])
stage2_path, stage2 = load_stage2_registration(Path.cwd())
output.parent.mkdir(parents=True, exist_ok=True)
try:
    output.mkdir()
except FileExistsError as exc:
    raise SystemExit(f"refusing to reuse or race on output path: {output}") from exc
(output / ".split_registry_sha256").write_text(SPLIT_REGISTRY_SHA256 + "\n", encoding="utf-8")
(output / ".stage2_registration_sha256").write_text(
    file_sha256(stage2_path) + "\n", encoding="utf-8"
)
(output / ".model_inventory_sha256").write_text(
    stage2["generation_spec"]["model_artifact_inventory"]["sha256"] + "\n",
    encoding="utf-8",
)
PY
  printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$output/.run_reservation"
}

generate_arm() {
  local label="$1"
  local output="$2"
  local checkpoint="$3"
  local scale="$4"
  reserve_run "$output"
  local args=(
    --baseline clean
    --prompts "$PROMPTS"
    --output-dir "$output"
    --model "$MODEL"
    --seeds "$(seeds)"
    --steps 25
    --guidance-scale 5
    --num-frames 49
    --fps 8
    --height 480
    --width 832
    --dtype bf16
    --device "$DEVICE"
    --vae-slicing
    --vae-tiling
  )
  if [[ -n "$checkpoint" ]]; then
    args+=(--lora-path "$checkpoint" --lora-scale "$scale")
  fi
  "$PYTHON" scripts/generate_wan_clean.py "${args[@]}"
  PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$label" <<'PY'
from pathlib import Path
import sys
import water_impact_dynamic_v3c_eval_protocol as p

root = Path.cwd()
rows = p.read_csv(root / p.FRESH_DEV_CSV)
stage2_path, stage2 = p.load_stage2_registration(root)
run = {"original": p.ORIGINAL_RUN, "v3b": p.V3B_RUN, "v3c": p.V3C_RUN}[sys.argv[1]]
manifest, _, videos = p.load_generation_run(
    root,
    run,
    sys.argv[1],
    rows,
    stage2_path,
    stage2,
    stage2["generation_spec"]["model_artifact_inventory"],
)
print(f"Validated {sys.argv[1]} generation: {manifest} ({len(videos)} videos)")
PY
}

case "${1:-}" in
  preflight)
    split_preflight
    ;;
  register-stage2)
    split_preflight
    "$PYTHON" scripts/register_water_impact_dynamic_v3c_eval_stage2.py
    ;;
  stage2-preflight)
    stage2_preflight
    ;;
  original)
    generate_arm original "$ORIGINAL_OUTPUT" "" 1.0
    ;;
  v3b)
    generate_arm v3b "$V3B_OUTPUT" "$V3B_CHECKPOINT" 1.25
    ;;
  v3c)
    generate_arm v3c "$V3C_OUTPUT" "$V3C_CHECKPOINT" 1.25
    ;;
  compare)
    stage2_preflight
    "$PYTHON" scripts/build_water_impact_dynamic_v3c_blind_review.py
    ;;
  score)
    stage2_preflight
    "$PYTHON" scripts/score_water_impact_dynamic_v3c_fresh_dev24.py \
      --review-template "$PUBLIC_DIR/blind_review.csv" \
      --reviewer-a "$REVIEWER_A" \
      --reviewer-b "$REVIEWER_B" \
      --adjudicator "$ADJUDICATOR" \
      --answer-key "$PRIVATE_DIR/answer_key.csv" \
      --review-manifest "$PRIVATE_DIR/review_manifest.json" \
      --output-dir "$SCORE_DIR"
    ;;
  *)
    echo "usage: $0 {preflight|register-stage2|stage2-preflight|original|v3b|v3c|compare|score}" >&2
    echo "sealed-final36 intentionally has no launcher command" >&2
    exit 2
    ;;
esac
