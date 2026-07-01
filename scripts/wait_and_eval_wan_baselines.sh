#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EXPECTED="${EXPECTED:-1340}"
WAN_PID="${WAN_PID:-${1:-}}"
ROOT="${ROOT:-outputs/wan_v2_yes335_baselines_bf16_step20_f49_480x832_8gpu_s1}"
REVIEW="${REVIEW:-experiments/baseline_review/wan_v2_yes335_baselines_full_review}"
EVAL="${EVAL:-experiments/evaluation/v2_wan_yes335_baselines_gpt5_4_parallel}"
METRICS="${METRICS:-experiments/metrics/v2_wan_yes335_baselines_gpt5_4_prelim}"
TOKEN_FILE="${TOKEN_FILE:-/home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt}"
PY="${PY:-/home/deepseek_VG/.conda/envs/vcecf/bin/python}"
MODEL="${MODEL:-gpt-5.4}"
WORKERS_PER_BASELINE="${WORKERS_PER_BASELINE:-4}"
TIMEOUT="${TIMEOUT:-180}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

count_mp4s() {
  find "$ROOT" -type f -name '*.mp4' 2>/dev/null | wc -l
}

read_gpt_config() {
  "$PY" - "$TOKEN_FILE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
url = ""
key = ""
for index, line in enumerate(lines):
    stripped = line.strip()
    lower = stripped.lower()
    if "987xyz" not in stripped or "url" not in lower:
        continue
    if ":" in stripped:
        url = stripped.split(":", 1)[1].strip()
    for follow in lines[index + 1 :]:
        follow = follow.strip()
        if not follow:
            continue
        if "url" in follow.lower() and "987xyz" not in follow:
            break
        if "sk-" in follow:
            key = follow[follow.find("sk-") :].strip()
            break
    break

if not url or not key:
    raise SystemExit("missing 987xyz url/key")
print(url)
print(key)
PY
}

if [[ -z "$WAN_PID" ]]; then
  WAN_PID="$(pgrep -f '^/home/deepseek_VG/.conda/envs/vcecf/bin/python -u scripts/run_parallel_wan_jobs.py .*wan_v2_yes335_baselines_bf16_step20_f49_480x832_8gpu_s1' | head -n 1 || true)"
fi

log "watcher started; Wan PID=${WAN_PID:-none}; waiting for $EXPECTED Wan baseline mp4s"
while true; do
  count="$(count_mp4s)"
  log "baseline mp4 count: $count/$EXPECTED"
  if [[ "$count" -ge "$EXPECTED" ]]; then
    break
  fi
  if [[ -n "$WAN_PID" ]] && ! kill -0 "$WAN_PID" 2>/dev/null; then
    log "Wan generator PID $WAN_PID is not alive but count is $count/$EXPECTED; stop for manual retry."
    exit 2
  fi
  sleep 120
done

log "building full review artifacts"
"$PY" scripts/build_baseline_review.py \
  --export-manifest benchmarks/causal_footprint_v2/export_wan_clean_yes335_enriched_manifest.json \
  --baseline-root "$ROOT" \
  --output-dir "$REVIEW" \
  --baselines negative_prompt videoeraser t2vunlearning safree_wan \
  --frames-per-video 5 \
  --thumb-width 180 \
  --thumb-height 112

mapfile -t gpt_config < <(read_gpt_config)
export VLM_BASE_URL="${gpt_config[0]}"
export VLM_API_KEY="${gpt_config[1]}"

mkdir -p "$EVAL" "$EVAL/launcher_logs"
log "launching four baseline VLM evaluators with model=$MODEL and workers_per_baseline=$WORKERS_PER_BASELINE"
pids=()
for baseline in negative_prompt videoeraser t2vunlearning safree_wan; do
  mkdir -p "$EVAL/$baseline"
  (
    "$PY" scripts/evaluate_v2_baseline_with_vlm.py \
      --review-csv "$REVIEW/baseline_review.csv" \
      --output-dir "$EVAL/$baseline" \
      --run-api \
      --source-name v2_wan_yes335 \
      --baseline "$baseline" \
      --model "$MODEL" \
      --workers "$WORKERS_PER_BASELINE" \
      --timeout "$TIMEOUT" \
      --continue-on-error
  ) > "$EVAL/$baseline/run.log" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "$EVAL/$baseline/run.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  log "one or more VLM evaluator jobs failed; inspect $EVAL/*/run.log"
  exit "$status"
fi

log "combining VLM predictions"
"$PY" - "$EVAL" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
baselines = ["negative_prompt", "videoeraser", "t2vunlearning", "safree_wan"]
combined = []
fields = None
for baseline in baselines:
    path = root / baseline / "vlm_predictions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fields is None:
            fields = reader.fieldnames
        combined.extend(reader)

out = root / "vlm_predictions_combined.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(combined)

raw_out = root / "vlm_raw_responses_combined.jsonl"
with raw_out.open("w", encoding="utf-8") as out_handle:
    for baseline in baselines:
        raw = root / baseline / "vlm_raw_responses.jsonl"
        if raw.exists():
            out_handle.write(raw.read_text(encoding="utf-8"))

print(f"combined {len(combined)} predictions -> {out}")
PY

log "computing metrics"
"$PY" scripts/compute_v2_baseline_metrics.py \
  --labels "$EVAL/vlm_predictions_combined.csv" \
  --output-dir "$METRICS"
log "done"
