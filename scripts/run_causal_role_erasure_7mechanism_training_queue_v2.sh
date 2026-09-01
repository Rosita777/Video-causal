#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/models/.wan-runtime/bin/python"
TRAINER="$PROJECT_ROOT/scripts/train_wan_causal_role_lora_v2.py"
REGISTRY="$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2/run_spec_registry.json"
REGISTRY_SHA256="36c79a862918e436fec59098f052ac6c11e83ce4833dd74b53fc5842285602ca"
LOG_ROOT="$PROJECT_ROOT/outputs/causal_role_erasure_7mechanism_main_v2/training_queue_logs_v2"
MIN_FREE_MIB="${MIN_FREE_MIB:-60000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"

mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue.log"

log() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"
}

actual_registry_sha="$($PYTHON - "$REGISTRY" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
PY
)"
[[ "$actual_registry_sha" == "$REGISTRY_SHA256" ]]

mapfile -t jobs < <(
  "$PYTHON" - "$REGISTRY" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "training_run_specs_frozen_after_cache_validation"
assert payload["run_spec_count"] == 18
for run_id, record in payload["run_specs"].items():
    print("\t".join((run_id, record["path"], record["sha256"])))
PY
)
[[ "${#jobs[@]}" -eq 18 ]]

IFS=',' read -r -a gpu_indices <<<"$GPU_DEVICES"
[[ "${#gpu_indices[@]}" -gt 0 ]]

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local free_mib utilization
    IFS=, read -r free_mib utilization < <(
      nvidia-smi --id="$gpu" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | tr -d ' '
    )
    if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then
      log "gpu_ready index=$gpu free_mib=$free_mib utilization=$utilization"
      return
    fi
    log "gpu_wait index=$gpu free_mib=$free_mib utilization=$utilization"
    sleep "$POLL_SECONDS"
  done
}

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local job_index run_id relative_spec spec_sha spec output_dir
  for ((job_index=worker_index; job_index<${#jobs[@]}; job_index+=${#gpu_indices[@]})); do
    IFS=$'\t' read -r run_id relative_spec spec_sha <<<"${jobs[$job_index]}"
    spec="$PROJECT_ROOT/$relative_spec"
    output_dir="$($PYTHON - "$spec" "$PROJECT_ROOT" <<'PY'
import json, pathlib, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
path = pathlib.Path(payload["output_dir"])
print(path if path.is_absolute() else pathlib.Path(sys.argv[2]) / path)
PY
)"
    if [[ -f "$output_dir/run_receipt.json" && ! -e "$output_dir/.run_reservation" ]]; then
      log "train_skip_complete gpu=$gpu run_id=$run_id"
      continue
    fi
    if [[ -e "$output_dir" ]]; then
      log "train_refuse_nonfresh gpu=$gpu run_id=$run_id output=$output_dir"
      return 1
    fi
    wait_for_gpu "$gpu"
    log "train_start gpu=$gpu run_id=$run_id"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PYTHON" "$TRAINER" \
        --project-root "$PROJECT_ROOT" \
        --run-spec "$spec" \
        --run-spec-sha256 "$spec_sha" \
        --run \
        >"$LOG_ROOT/$run_id.log" 2>&1
    [[ -f "$output_dir/run_receipt.json" && ! -e "$output_dir/.run_reservation" ]]
    log "train_complete gpu=$gpu run_id=$run_id"
  done
}

worker_pids=()
for worker_index in "${!gpu_indices[@]}"; do
  run_worker "$worker_index" "${gpu_indices[$worker_index]}" &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  log "training_queue_failed"
  exit "$status"
fi
log "all_training_runs_complete"
