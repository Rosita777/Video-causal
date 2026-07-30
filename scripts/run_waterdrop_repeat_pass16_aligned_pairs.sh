#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

review="${REVIEW:-data/waterdrop_scene_probe30_repeat_review.csv}"
source_root="${SOURCE_ROOT:-outputs/waterdrop_scene_probe30_repeat_wan_seed8400_8429}"
output_root="${OUTPUT_ROOT:-outputs/waterdrop_repeat_pass16_aligned_pairs}"

while IFS=, read -r sample_id seed part index receiver decision reference_end notes; do
  if [[ "$sample_id" == "sample_id" || "$decision" != "pass" ]]; then
    continue
  fi

  input_video=$(find "$source_root/$part/videos" -maxdepth 1 -name "*seed${seed}.mp4" -print -quit)
  if [[ -z "$input_video" ]]; then
    echo "Factual video for ${sample_id} seed ${seed} was not found" >&2
    exit 1
  fi

  echo "Building ${sample_id}: ${receiver}"
  models/.wan-runtime/bin/python scripts/build_static_counterfactual_pair.py \
    --input "$input_video" \
    --output-dir "${output_root}/${sample_id}_seed${seed}" \
    --reference-start 0 \
    --reference-end "$reference_end"
done < "$review"
