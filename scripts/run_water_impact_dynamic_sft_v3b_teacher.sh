#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
if [[ "$MODEL" != "models/Wan2.1-T2V-1.3B-Diffusers" || "$DEVICE" != "cuda" ]]; then
  echo "v3b scale protocol requires the frozen Wan model path and cuda device" >&2
  exit 1
fi
MANIFEST="data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
EXPECTED_MANIFEST_SHA256="3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
BASE_CACHE_DIR="outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
EXPECTED_BASE_CACHE_SHA256="4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
TEACHER_CACHE_DIR="outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
EXPECTED_TEACHER_CACHE_SHA256="6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
EXPECTED_TEACHER_CACHE_MANIFEST_SHA256="c467d7f81ee22b2c4b1ff719537487fbfc808eacc98e730c3d24f0a17aca77cb"
EXPECTED_UNIQUE_EMBEDDING_SHA256="a15f5e910358d5e95bcdd995303abb7eb7e7302fd9ee649c4cfebf3b8f6b6330"
EXPECTED_PROMPT_BINDING_SHA256="9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc"

verify_manifest() {
  "$PYTHON" - "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

actual = sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"frozen manifest hash mismatch: {actual} != {sys.argv[2]}")
print(f"Frozen manifest SHA-256: {actual}")
PY
}

verify_cache() {
  local manifest="$1"
  local cache_dir="$2"
  local expected_digest="$3"
  "$PYTHON" - "$manifest" "$cache_dir" "$expected_digest" <<'PY'
import csv
from hashlib import sha256
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
expected_digest = sys.argv[3]
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected = [cache_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in enumerate(rows)]
actual = sorted(cache_dir.glob("*.pt"))
missing = [path for path in expected if not path.is_file()]
unexpected = sorted(set(actual) - set(expected))
if missing or unexpected or len(actual) != len(expected):
    raise SystemExit(
        f"base cache validation failed: expected={len(expected)} actual={len(actual)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
digest = sha256()
for path in expected:
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with path.resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\n")
actual_digest = digest.hexdigest()
if actual_digest != expected_digest:
    raise SystemExit(f"base cache hash mismatch: {actual_digest} != {expected_digest}")
print(f"Validated {len(expected)} base cache entries; SHA-256={actual_digest}")
PY
}

verify_teacher_cache() {
  if [[ "$EXPECTED_TEACHER_CACHE_SHA256" == "TO_BE_FROZEN_AFTER_PREP" ]]; then
    echo "teacher cache hash has not been frozen in the launcher" >&2
    exit 1
  fi
  "$PYTHON" - \
    "$MANIFEST" \
    "$TEACHER_CACHE_DIR" \
    "$EXPECTED_TEACHER_CACHE_SHA256" \
    "$EXPECTED_PROMPT_BINDING_SHA256" \
    "$EXPECTED_TEACHER_CACHE_MANIFEST_SHA256" \
    "$EXPECTED_UNIQUE_EMBEDDING_SHA256" <<'PY'
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
expected_inventory = sys.argv[3]
expected_binding = sys.argv[4]
expected_manifest_hash = sys.argv[5]
expected_unique_embedding = sys.argv[6]
with manifest_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
erase = [(index, row) for index, row in enumerate(rows) if row["training_role"] == "erase"]
expected = [cache_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in erase]
actual = sorted(cache_dir.glob("*.pt"))
if len(expected) != 178 or set(expected) != set(actual) or any(not path.is_file() for path in expected):
    raise SystemExit(f"teacher cache inventory mismatch: expected={len(expected)} actual={len(actual)}")
binding = sha256()
for _, row in erase:
    binding.update(row["scene_id"].encode("utf-8"))
    binding.update(b"\0")
    binding.update(row["target_generation_prompt"].encode("utf-8"))
    binding.update(b"\n")
if binding.hexdigest() != expected_binding:
    raise SystemExit(f"target-prompt binding mismatch: {binding.hexdigest()}")
digest = sha256()
for path in expected:
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\n")
actual_inventory = digest.hexdigest()
if actual_inventory != expected_inventory:
    raise SystemExit(f"teacher cache hash mismatch: {actual_inventory} != {expected_inventory}")
metadata_path = cache_dir / "cache_manifest.json"
if sha256(metadata_path.read_bytes()).hexdigest() != expected_manifest_hash:
    raise SystemExit("teacher cache manifest byte hash mismatch")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("cache_inventory_sha256") != actual_inventory:
    raise SystemExit("teacher cache manifest does not bind the current inventory")
if metadata.get("prompt_binding_sha256") != expected_binding:
    raise SystemExit("teacher cache manifest prompt binding mismatch")
if metadata.get("unique_embedding_sha256") != expected_unique_embedding:
    raise SystemExit("teacher cache unique-embedding hash mismatch")
print(f"Validated {len(expected)} teacher cache entries; SHA-256={actual_inventory}")
PY
}

write_run_registration() {
  "$PYTHON" - \
    "$OUTPUT_DIR" \
    "$CALIBRATION_ID" \
    "$TEACHER_WEIGHT" \
    "$SANITY_MEAN_MIN" \
    "$SANITY_MEAN_MAX" \
    "$SANITY_SINGLE_MAX" \
    "$EXPECTED_MANIFEST_SHA256" \
    "$EXPECTED_BASE_CACHE_SHA256" \
    "$EXPECTED_TEACHER_CACHE_SHA256" <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

output_dir = Path(sys.argv[1])
pilot = [
    {
        "path": "outputs/water_impact_dynamic_v3b/logs/train_target_prompt_teacher_v1.log",
        "sha256": "c0f35542d9be763ea4a446af773e0e22fe44913b019b89aca51588780f5719ba",
    },
    {
        "path": "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/checkpoint-000025/pytorch_lora_weights.safetensors",
        "sha256": "2ee9f08c83d291630c09efcdf5bf0f8ae082f7b23b4c6be0ed89de791377ff3b",
    },
    {
        "path": "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/checkpoint-000025/training_state.json",
        "sha256": "d51fe90cedc168125e773f4c44ad458cc2baf84f409df6ed29f20cc09bcae854",
    },
]
for record in pilot:
    path = Path(record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise SystemExit(f"lambda=1 pilot artifact mismatch: {path}")

trainer = Path("scripts/train_wan_waterdrop_lora.py")
launcher = Path("scripts/run_water_impact_dynamic_sft_v3b_teacher.sh")
protocol_doc = Path("docs/water_impact_dynamic_v3b_target_prompt_teacher.md")
registration = {
    "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_scale4_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "calibration_id": sys.argv[2],
    "output_dir": str(output_dir),
    "target_prompt_teacher_weight": float(sys.argv[3]),
    "sanity_mean_min": float(sys.argv[4]),
    "sanity_mean_max": float(sys.argv[5]),
    "sanity_single_max": float(sys.argv[6]),
    "sanity_formula": "s_i = weight * sqrt(target_prompt_teacher_loss / flow_loss)",
    "sanity_aggregation": "arithmetic_mean_over_first_16_erase_steps",
    "selection_rule": "nearest_power_of_two(0.30 / mean_i(sqrt(r_i)))",
    "train_manifest_sha256": sys.argv[7],
    "base_cache_sha256": sys.argv[8],
    "teacher_cache_sha256": sys.argv[9],
    "expected_initial_lora_sha256": "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8",
    "lambda1_scale_invalid": True,
    "lambda1_generation_count": 0,
    "lambda1_mean_raw_loss_ratio_first_16": 0.005843,
    "lambda1_logged_mean_sqrt_raw_ratio_first_15": 0.07529824553306262,
    "lambda1_logged_median_sqrt_raw_ratio_first_15": 0.07650179622526487,
    "lambda1_artifacts": pilot,
    "training_config": {
        "model": "models/Wan2.1-T2V-1.3B-Diffusers",
        "height": 480,
        "width": 832,
        "num_frames": 49,
        "max_steps": 200,
        "learning_rate": 5e-5,
        "rank": 16,
        "alpha": 16,
        "grad_accum": 1,
        "seed": 26000,
        "device": "cuda",
        "role": "all",
        "objective": "target_prompt_teacher",
        "balanced_roles": True,
        "preserve_weight": 4.0,
        "target_prompt_calibration_id": "lambda4_from_lambda1_first16_output_gradient_v1",
        "target_prompt_teacher_weight": 4.0,
        "sanity_mean_min": 0.2,
        "sanity_mean_max": 0.5,
        "sanity_single_max": 1.0,
    },
    "trainer_path": str(trainer),
    "trainer_sha256": file_sha256(trainer),
    "launcher_path": str(launcher),
    "launcher_sha256": file_sha256(launcher),
    "protocol_doc_path": str(protocol_doc),
    "protocol_doc_sha256": file_sha256(protocol_doc),
}
path = output_dir / "run_registration.json"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
print(f"Wrote frozen run registration: {path} SHA-256={file_sha256(path)}")
PY
}

verify_manifest

case "${1:-}" in
  prepare)
    exec "$PYTHON" scripts/prepare_water_impact_dynamic_v3b_teacher_cache.py \
      --manifest "$MANIFEST" \
      --model "$MODEL" \
      --output-dir "$TEACHER_CACHE_DIR" \
      --device "$DEVICE"
    ;;
  train)
    echo "the lambda=1 run is frozen scale-invalid; use commit a1598ec to reproduce it" >&2
    exit 1
    ;;
  train-scale4)
    OUTPUT_DIR="outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1"
    TEACHER_WEIGHT="4.0"
    CALIBRATION_ID="lambda4_from_lambda1_first16_output_gradient_v1"
    SANITY_MEAN_MIN="0.2"
    SANITY_MEAN_MAX="0.5"
    SANITY_SINGLE_MAX="1.0"
    ;;
  *)
    echo "usage: $0 {prepare|train-scale4}" >&2
    exit 2
    ;;
esac

verify_cache "$MANIFEST" "$BASE_CACHE_DIR" "$EXPECTED_BASE_CACHE_SHA256"
verify_teacher_cache
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
  echo "refusing to reuse or race on output path: $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUTPUT_DIR/.run_reservation"
write_run_registration

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --cache-dir "$BASE_CACHE_DIR" \
  --target-prompt-cache-dir "$TEACHER_CACHE_DIR" \
  --target-prompt-cache-sha256 "$EXPECTED_TEACHER_CACHE_SHA256" \
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
  --objective target_prompt_teacher \
  --target-prompt-teacher-weight "$TEACHER_WEIGHT" \
  --target-prompt-calibration-id "$CALIBRATION_ID" \
  --target-prompt-sanity-min-output-grad-ratio "$SANITY_MEAN_MIN" \
  --target-prompt-sanity-max-output-grad-ratio "$SANITY_MEAN_MAX" \
  --target-prompt-sanity-max-single-output-grad-ratio "$SANITY_SINGLE_MAX" \
  --preserve-weight 4.0 \
  --balanced-roles
