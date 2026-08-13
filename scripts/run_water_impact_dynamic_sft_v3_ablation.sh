#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
MANIFEST="data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
EXPECTED_MANIFEST_SHA256="3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
CACHE_DIR="outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
EXPECTED_CACHE_SHA256="4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"

verify_manifest() {
  local actual
  actual="$("$PYTHON" - "$MANIFEST" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
  if [[ "$actual" != "$EXPECTED_MANIFEST_SHA256" ]]; then
    echo "frozen manifest hash mismatch: $actual" >&2
    exit 1
  fi
  echo "Frozen manifest SHA-256: $actual"
}

verify_cache() {
  "$PYTHON" - "$MANIFEST" "$CACHE_DIR" "$EXPECTED_CACHE_SHA256" <<'PY'
import csv
from hashlib import sha256
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
expected_digest = sys.argv[3]
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected = [
    cache_dir / f"{index:03d}_{row['scene_id']}.pt"
    for index, row in enumerate(rows)
]
actual = sorted(cache_dir.glob("*.pt"))
missing = [path for path in expected if not path.is_file()]
unexpected = sorted(set(actual) - set(expected))
if missing or unexpected or len(actual) != len(expected):
    raise SystemExit(
        f"cache validation failed: expected={len(expected)} actual={len(actual)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
digest = sha256()
for path in expected:
    resolved = path.resolve(strict=True)
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\n")
actual_digest = digest.hexdigest()
if actual_digest != expected_digest:
    raise SystemExit(
        f"frozen cache content hash mismatch: {actual_digest} != {expected_digest}"
    )
print(f"Validated {len(expected)} cache entries; SHA-256={actual_digest}")
PY
}

verify_manifest

case "${1:-}" in
  prepare)
    "$PYTHON" scripts/prepare_water_impact_dynamic_v2_cache.py
    verify_cache
    exit 0
    ;;
  balanced)
    OUTPUT_DIR="outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_balanced_seeded"
    SAMPLING_ARGS=(--balanced-roles)
    ;;
  exposure)
    OUTPUT_DIR="outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_exposure_seeded"
    SAMPLING_ARGS=()
    ;;
  *)
    echo "usage: $0 {prepare|balanced|exposure}" >&2
    exit 2
    ;;
esac

verify_cache
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
  echo "refusing to reuse or race on output path: $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUTPUT_DIR/.run_reservation"

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --max-steps 200 \
  --learning-rate 5e-5 \
  --rank 16 \
  --alpha 16 \
  --save-every 25 \
  --seed 26000 \
  --device "$DEVICE" \
  --role all \
  --objective plain \
  --preserve-weight 4.0 \
  "${SAMPLING_ARGS[@]}"
