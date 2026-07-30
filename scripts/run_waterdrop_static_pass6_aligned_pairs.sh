#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

plan="data/waterdrop_static_pass6_pair_plan.csv"
output_root="outputs/waterdrop_static_pass6_aligned_pairs"

while IFS=, read -r scene_id seed reference_end receiver; do
  if [[ "$scene_id" == "scene_id" ]]; then
    continue
  fi

  input_video=$(find \
    outputs/waterdrop_scene_probe30_wan_seed8300_8329 \
    -path '*/videos/*' \
    -name "*seed${seed}.mp4" \
    -print \
    -quit)

  if [[ -z "$input_video" ]]; then
    echo "Factual video for ${scene_id} seed ${seed} was not found" >&2
    exit 1
  fi

  echo "Building ${scene_id}: ${receiver}"
  models/.wan-runtime/bin/python scripts/build_static_counterfactual_pair.py \
    --input "$input_video" \
    --output-dir "${output_root}/${scene_id}_seed${seed}" \
    --reference-start 0 \
    --reference-end "$reference_end"
done < "$plan"
