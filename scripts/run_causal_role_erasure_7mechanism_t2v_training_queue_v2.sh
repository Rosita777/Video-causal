#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REGISTRY="${T2V_TRAINING_REGISTRY:-$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2/t2v_training_registry.json}"
PYTHON_BIN="${T2V_PYTHON:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
LOG_ROOT="${T2V_LOG_ROOT:-$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_logs_v2}"
GPU_LIST="${T2V_GPUS:-0,1,2,3}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "T2V_GPUS must contain at least one GPU" >&2
  exit 2
fi
if [[ ! -f "$REGISTRY" ]]; then
  echo "Missing frozen T2V training registry: $REGISTRY" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
mechanisms=(
  water_impact
  rigid_collision
  brittle_fracture
  powder_impact
  elastic_deformation
  material_release
  surface_trace
)

run_one() {
  local mechanism="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py" \
      --project-root "$PROJECT_ROOT" \
      --training-registry "$REGISTRY" \
      --mechanism "$mechanism" \
      > "$LOG_ROOT/${mechanism}.log" 2>&1
}

pending=("${mechanisms[@]}")
while [[ ${#pending[@]} -gt 0 ]]; do
  pids=()
  launched=()
  for gpu in "${GPUS[@]}"; do
    if [[ ${#pending[@]} -eq 0 ]]; then
      break
    fi
    mechanism="${pending[0]}"
    pending=("${pending[@]:1}")
    run_one "$mechanism" "$gpu" &
    pids+=("$!")
    launched+=("$mechanism")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "T2V training failed for ${launched[$index]}" >&2
      exit 1
    fi
  done
done

echo "All seven registry-bound T2V adapted training runs completed."
