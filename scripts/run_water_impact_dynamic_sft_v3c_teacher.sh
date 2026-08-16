#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="${WAN_MODEL:-models/Wan2.1-T2V-1.3B-Diffusers}"
DEVICE="${WAN_DEVICE:-cuda}"
if [[ "$MODEL" != "models/Wan2.1-T2V-1.3B-Diffusers" || "$DEVICE" != "cuda" ]]; then
  echo "v3c sigma-weighted protocol requires the frozen Wan model path and cuda device" >&2
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
EXPECTED_TRANSFORMER_INVENTORY_SHA256="fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac"
OUTPUT_DIR="outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1"
TEACHER_WEIGHT="4.0"
CALIBRATION_ID="v3c_two_sigma_mean_one_preregistered_v1"
SANITY_MEAN_MIN="0.2"
SANITY_MEAN_MAX="0.5"
SANITY_SINGLE_MAX="1.0"

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

verify_transformer_inventory() {
  "$PYTHON" - "$MODEL" "$EXPECTED_TRANSFORMER_INVENTORY_SHA256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
expected_aggregate = sys.argv[2]
expected_records = [
    {
        "path": "transformer/config.json",
        "size": 465,
        "sha256": "0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "size": 4998781576,
        "sha256": "6d011927dbd2cc8afe53d57abab04a8fd86f615d83324770d985fb058ece3a24",
    },
    {
        "path": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "size": 677289072,
        "sha256": "b92ec2309b1f239af6f746431815a881afcc938abb26a4f08d9a2fd6c892f872",
    },
    {
        "path": "transformer/diffusion_pytorch_model.safetensors.index.json",
        "size": 73296,
        "sha256": "dcbcf3497134a3f50557ff069dd7d2c84b5c4d8c5932472f6bdb780fb4016589",
    },
]
transformer = model / "transformer"
index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
config_path = transformer / "config.json"
if not config_path.is_file() or not index_path.is_file():
    raise SystemExit("transformer config or safetensors index is missing")
try:
    index = json.loads(index_path.read_text(encoding="utf-8"))
except (json.JSONDecodeError, UnicodeDecodeError) as exc:
    raise SystemExit(f"invalid transformer safetensors index: {exc}") from exc
weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("transformer safetensors index has no non-empty weight_map")
shard_values = list(weight_map.values())
if not shard_values or any(
    not isinstance(name, str)
    or not name.endswith(".safetensors")
    or Path(name).name != name
    for name in shard_values
):
    raise SystemExit("transformer safetensors index contains invalid shard names")
referenced = sorted(set(shard_values))
actual = sorted(path.name for path in transformer.glob("*.safetensors"))
if actual != referenced:
    raise SystemExit(
        f"transformer shard inventory disagrees with index: actual={actual} referenced={referenced}"
    )
relative_names = sorted(
    [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        *(f"transformer/{name}" for name in referenced),
    ]
)
aggregate = sha256()
records = []
for relative_name in relative_names:
    path = model / relative_name
    if not path.is_file():
        raise SystemExit(f"missing transformer inventory file: {path}")
    one = sha256()
    aggregate.update(relative_name.encode("utf-8"))
    aggregate.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            one.update(chunk)
            aggregate.update(chunk)
    aggregate.update(b"\n")
    records.append(
        {"path": relative_name, "size": path.stat().st_size, "sha256": one.hexdigest()}
    )
if records != expected_records:
    raise SystemExit(f"frozen transformer file inventory mismatch: {records!r}")
actual_aggregate = aggregate.hexdigest()
if actual_aggregate != expected_aggregate:
    raise SystemExit(
        f"frozen transformer aggregate SHA-256 mismatch: {actual_aggregate} != {expected_aggregate}"
    )
print(
    f"Validated {len(records)} ordered transformer files; SHA-256={actual_aggregate}"
)
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

verify_v3b_reference() {
  "$PYTHON" - <<'PY'
from hashlib import sha256
from pathlib import Path

def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = {
    Path(
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
        "checkpoint-000200/pytorch_lora_weights.safetensors"
    ): "d3fecf26b7f1ca6c4a8f46c86850a47a7ec5a62762d0e0aa15c49363040875d3",
    Path(
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
        "checkpoint-000200/training_state.json"
    ): "0f9aa26e825f4f6f497b1312c507b685c054bc319f2f9f538e45eeb7a7908bea",
    Path(
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
        "run_registration.json"
    ): "53f0a7c472ba02a38b90b55651f378e5feda0bcd709f86786702de163b3a87f4",
    Path(
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
        "target_prompt_scale_sanity.json"
    ): "26fb8b1ff9e0d446fd186765ba1ff9a9d1a085d75d230cd0a419509ea00bbb12",
}
for path, expected in artifacts.items():
    if not path.is_file():
        raise SystemExit(f"missing frozen v3b reference artifact: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise SystemExit(f"v3b reference hash mismatch: {path}: {actual} != {expected}")
print(f"Validated {len(artifacts)} frozen v3b reference artifacts")
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
    "$EXPECTED_TEACHER_CACHE_SHA256" \
    "$EXPECTED_TRANSFORMER_INVENTORY_SHA256" <<'PY'
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
v3b_reference_artifacts = [
    {
        "path": (
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "checkpoint-000200/pytorch_lora_weights.safetensors"
        ),
        "sha256": "d3fecf26b7f1ca6c4a8f46c86850a47a7ec5a62762d0e0aa15c49363040875d3",
    },
    {
        "path": (
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "checkpoint-000200/training_state.json"
        ),
        "sha256": "0f9aa26e825f4f6f497b1312c507b685c054bc319f2f9f538e45eeb7a7908bea",
    },
    {
        "path": (
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "run_registration.json"
        ),
        "sha256": "53f0a7c472ba02a38b90b55651f378e5feda0bcd709f86786702de163b3a87f4",
    },
    {
        "path": (
            "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
            "target_prompt_scale_sanity.json"
        ),
        "sha256": "26fb8b1ff9e0d446fd186765ba1ff9a9d1a085d75d230cd0a419509ea00bbb12",
    },
]
for record in v3b_reference_artifacts:
    path = Path(record["path"])
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise SystemExit(f"v3b reference artifact mismatch: {path}")

trainer = Path("scripts/train_wan_waterdrop_lora_v3c.py")
launcher = Path("scripts/run_water_impact_dynamic_sft_v3c_teacher.sh")
protocol_doc = Path("docs/water_impact_dynamic_v3c_sigma_weighted_teacher.md")
registration = {
    "protocol": "water_impact_dynamic_v3c_sigma_weighted_target_prompt_teacher_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "calibration_id": sys.argv[2],
    "output_dir": str(output_dir),
    "target_prompt_teacher_base_weight": float(sys.argv[3]),
    "target_prompt_teacher_schedule": "2*sigma",
    "target_prompt_teacher_effective_weight_formula": "4 * (2 * sigma)",
    "target_prompt_teacher_expected_mean_weight": 4.0,
    "sanity_mean_min": float(sys.argv[4]),
    "sanity_mean_max": float(sys.argv[5]),
    "sanity_single_max": float(sys.argv[6]),
    "sanity_formula": (
        "g_i = 8 * sigma_i * sqrt(target_prompt_teacher_loss / flow_loss)"
    ),
    "sanity_aggregation": "arithmetic_mean_over_first_16_erase_steps",
    "checkpoint_policy": "no_checkpoint_before_scale_sanity_passes",
    "train_manifest_sha256": sys.argv[7],
    "base_cache_sha256": sys.argv[8],
    "teacher_cache_sha256": sys.argv[9],
    "teacher_cache_manifest_sha256": "c467d7f81ee22b2c4b1ff719537487fbfc808eacc98e730c3d24f0a17aca77cb",
    "target_prompt_binding_sha256": "9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc",
    "target_prompt_unique_embedding_sha256": "a15f5e910358d5e95bcdd995303abb7eb7e7302fd9ee649c4cfebf3b8f6b6330",
    "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
    "transformer_inventory_algorithm": "sha256_ordered_name_nul_bytes_lf_v1",
    "transformer_inventory": [
        {
            "path": "transformer/config.json",
            "size": 465,
            "sha256": "0b093fa072e9ff28763febe9b964ee582f566733a6d6709deb9dfba1bde16b81",
        },
        {
            "path": "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "size": 4998781576,
            "sha256": "6d011927dbd2cc8afe53d57abab04a8fd86f615d83324770d985fb058ece3a24",
        },
        {
            "path": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "size": 677289072,
            "sha256": "b92ec2309b1f239af6f746431815a881afcc938abb26a4f08d9a2fd6c892f872",
        },
        {
            "path": "transformer/diffusion_pytorch_model.safetensors.index.json",
            "size": 73296,
            "sha256": "dcbcf3497134a3f50557ff069dd7d2c84b5c4d8c5932472f6bdb780fb4016589",
        },
    ],
    "transformer_inventory_sha256": sys.argv[10],
    "expected_initial_lora_sha256": "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8",
    "expected_noise_sigma_rng_initial_sha256": "49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704",
    "expected_noise_sigma_rng_final_sha256": "79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f",
    "expected_sample_order_sha256": "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb",
    "v3b_reference_artifacts": v3b_reference_artifacts,
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
        "save_every": 25,
        "seed": 26000,
        "device": "cuda",
        "role": "all",
        "objective": "target_prompt_teacher_sigma_weighted",
        "balanced_roles": True,
        "preserve_weight": 4.0,
        "target_prompt_calibration_id": "v3c_two_sigma_mean_one_preregistered_v1",
        "target_prompt_teacher_base_weight": 4.0,
        "target_prompt_teacher_schedule": "2*sigma",
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
verify_transformer_inventory
verify_cache "$MANIFEST" "$BASE_CACHE_DIR" "$EXPECTED_BASE_CACHE_SHA256"
verify_teacher_cache
verify_v3b_reference

case "${1:-}" in
  preflight)
    echo "v3c preflight passed; no output directory was created"
    exit 0
    ;;
  train)
    ;;
  *)
    echo "usage: $0 {preflight|train}" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUTPUT_DIR")"
if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
  echo "refusing to reuse or race on output path: $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUTPUT_DIR/.run_reservation"
write_run_registration

exec "$PYTHON" scripts/train_wan_waterdrop_lora_v3c.py \
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
  --objective target_prompt_teacher_sigma_weighted \
  --target-prompt-teacher-weight "$TEACHER_WEIGHT" \
  --target-prompt-calibration-id "$CALIBRATION_ID" \
  --target-prompt-sanity-min-output-grad-ratio "$SANITY_MEAN_MIN" \
  --target-prompt-sanity-max-output-grad-ratio "$SANITY_MEAN_MAX" \
  --target-prompt-sanity-max-single-output-grad-ratio "$SANITY_SINGLE_MAX" \
  --preserve-weight 4.0 \
  --balanced-roles
