#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
CHECKPOINTS="${CHECKPOINTS:-025 050}"

generate_pair() {
  local checkpoint="$1"
  local split="$2"
  local prompts seeds output
  if [[ "$split" == "target" ]]; then
    prompts="data/collision_validation7.prompts"
    seeds="11013,11019"
  else
    prompts="data/collision_specificity8.prompts"
    seeds="12000,12001"
  fi
  output="outputs/collision_causal_gate_ckpt${checkpoint}_${split}2"
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline clean --prompts "$prompts" --limit 2 --seeds "$seeds" \
    --output-dir "$output" --model "$MODEL" \
    --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
    --height 480 --width 832 --dtype bf16 --device cuda \
    --vae-slicing --vae-tiling --skip-existing \
    --lora-path "outputs/adapters/collision_causal_gate_100/checkpoint-000${checkpoint}" \
    --lora-scale 1.0 \
    > "logs/collision_causal_gate_ckpt${checkpoint}_${split}2.log" 2>&1
}

for checkpoint in $CHECKPOINTS; do
  generate_pair "$checkpoint" target
  generate_pair "$checkpoint" control
done

for checkpoint in $CHECKPOINTS; do
  "$PYTHON" scripts/evaluate_collision_validation7.py \
    --base-dir outputs/collision_validation7_base \
    --adapter-dir "outputs/collision_causal_gate_ckpt${checkpoint}_target2" \
    --limit 2 \
    --output-csv "experiments/pilot_week1/collision_causal_gate_ckpt${checkpoint}_target2_metrics.csv" \
    --sheet-dir "outputs/collision_causal_gate_ckpt${checkpoint}_target2_contact_sheets"
  "$PYTHON" scripts/evaluate_collision_validation7.py \
    --base-dir outputs/collision_specificity8_base \
    --adapter-dir "outputs/collision_causal_gate_ckpt${checkpoint}_control2" \
    --limit 2 \
    --output-csv "experiments/pilot_week1/collision_causal_gate_ckpt${checkpoint}_control2_metrics.csv" \
    --sheet-dir "outputs/collision_causal_gate_ckpt${checkpoint}_control2_contact_sheets"
done

"$PYTHON" scripts/evaluate_collision_validation7.py \
  --base-dir outputs/collision_validation7_base \
  --adapter-dir outputs/collision_causal_gate_100_validation7 \
  --limit 2 \
  --output-csv experiments/pilot_week1/collision_causal_gate_ckpt100_target2_metrics.csv \
  --sheet-dir outputs/collision_causal_gate_ckpt100_target2_contact_sheets
"$PYTHON" scripts/evaluate_collision_validation7.py \
  --base-dir outputs/collision_specificity8_base \
  --adapter-dir outputs/collision_causal_gate_100_specificity8 \
  --limit 2 \
  --output-csv experiments/pilot_week1/collision_causal_gate_ckpt100_control2_metrics.csv \
  --sheet-dir outputs/collision_causal_gate_ckpt100_control2_contact_sheets
