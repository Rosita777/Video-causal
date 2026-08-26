#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_INDEX="${GPU_INDEX:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="$PROJECT_ROOT/models/.wan-runtime/bin/python"
PREPARER="$PROJECT_ROOT/scripts/prepare_causal_role_erasure_7mechanism_training_cache_v2.py"
INPUT_ROOT="$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/training_inputs_v2"
CACHE_ROOT="$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/training_caches_v2"
LOG_ROOT="$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/cache_queue_logs_v2"

mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue.log"

mechanisms=(
  water_impact
  rigid_collision
  brittle_fracture
  powder_impact
  elastic_deformation
  material_release
  surface_trace
)
registry_files=(
  00_water_impact.json
  01_rigid_collision.json
  02_brittle_fracture.json
  03_powder_impact.json
  04_elastic_deformation.json
  05_material_release.json
  06_surface_trace.json
)
registry_hashes=(
  3e008f6a565bf39b6b938c0d2ebcc057d23dc89bc434bf1df7e2668f3c354a73
  00c62ae3c0c5586b23baddca647f0fbc7948228d538983abf28b15479fb65b41
  f58fb67c0bcef9280b6df43d3ae3fcb28f4df2e081e7755e17a19a964da3b50a
  7cc4c68010cd2ef86df736ae426c846c9214f98c30e60e188fcf693bdcded8a7
  ba0108ba4ef184ac1eb3496e0f5aefa2d59bf4f756b1d455afb5177ff7e4a987
  2442925fdeea8dd01c994706eaa8282c3e5a1980909b3cdafbfb988c96eb5d5e
  d9517ab3bfea7fcccda122c518cef319ee36654a09bd4750f9137a5642e913e4
)

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"
}

wait_for_gpu() {
  while true; do
    IFS=, read -r free_mib utilization < <(
      nvidia-smi --id="$GPU_INDEX" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | tr -d ' '
    )
    if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then
      log "gpu_ready index=$GPU_INDEX free_mib=$free_mib utilization=$utilization"
      return
    fi
    log "gpu_wait index=$GPU_INDEX free_mib=$free_mib utilization=$utilization"
    sleep "$POLL_SECONDS"
  done
}

run_cache() {
  local mechanism="$1"
  local registry_file="$2"
  local registry_hash="$3"
  local mode="$4"
  local output_dir="$5"
  local arm="${6:-}"
  local base_dir="${7:-}"
  local label="${mechanism}_${mode}${arm:+_$arm}"

  if [[ -f "$output_dir/cache_manifest.json" && ! -e "$output_dir/.run_reservation" ]]; then
    log "cache_skip_complete label=$label output=$output_dir"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    log "cache_refuse_nonfresh label=$label output=$output_dir"
    return 1
  fi

  wait_for_gpu
  local command=(
    "$PYTHON" "$PREPARER" "$mode"
    --project-root "$PROJECT_ROOT"
    --registry "$INPUT_ROOT/cache_registries/$registry_file"
    --registry-sha256 "$registry_hash"
    --output-dir "$output_dir"
    --device cuda
    --run
  )
  if [[ -n "$arm" ]]; then
    command+=(--arm "$arm")
  fi
  if [[ -n "$base_dir" ]]; then
    command+=(--base-cache-dir "$base_dir")
  fi
  log "cache_start label=$label output=$output_dir"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${command[@]}" >"$LOG_ROOT/$label.log" 2>&1
  [[ -f "$output_dir/cache_manifest.json" && ! -e "$output_dir/.run_reservation" ]]
  log "cache_complete label=$label output=$output_dir"
}

for index in "${!mechanisms[@]}"; do
  mechanism="${mechanisms[$index]}"
  registry_file="${registry_files[$index]}"
  registry_hash="${registry_hashes[$index]}"
  mechanism_root="$CACHE_ROOT/$mechanism"
  base_dir="$mechanism_root/base"

  run_cache "$mechanism" "$registry_file" "$registry_hash" \
    prepare-base "$base_dir"
  run_cache "$mechanism" "$registry_file" "$registry_hash" \
    prepare-teacher "$mechanism_root/teacher"
  run_cache "$mechanism" "$registry_file" "$registry_hash" \
    prepare-arm "$mechanism_root/arms/matched" matched "$base_dir"
  run_cache "$mechanism" "$registry_file" "$registry_hash" \
    prepare-arm "$mechanism_root/arms/v4" v4

  if [[ "$mechanism" == water_impact || "$mechanism" == brittle_fracture ]]; then
    run_cache "$mechanism" "$registry_file" "$registry_hash" \
      prepare-arm "$mechanism_root/arms/generic_paraphrase" generic_paraphrase
    run_cache "$mechanism" "$registry_file" "$registry_hash" \
      prepare-arm "$mechanism_root/arms/bystander" bystander
  fi
done

log "all_cache_jobs_complete"
