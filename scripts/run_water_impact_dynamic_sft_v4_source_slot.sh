#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
MANIFEST="data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
BASE_CACHE_DIR="outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
TEACHER_CACHE_DIR="outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
SOURCE_BANK_REGISTRY="data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json"
HOLDOUT_PUBLIC_COMMITMENT="data/water_impact_dynamic_v4/holdout_public_commitment_v2.json"
CAUSAL_STAGE0_PUBLIC_COMMITMENT="data/water_impact_dynamic_v4/causal_stage0_public_commitment_v2.json"
SOURCE_MAPPING_REGISTRY="data/water_impact_dynamic_v4/source_mapping_v2.json"
PROMPT_SIDECAR_DIR="outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2"
PREFLIGHT_ARTIFACT="outputs/water_impact_dynamic_v4/null_sidecar_preflight_v2.json"
TRAINING_AUTHORIZATION="data/water_impact_dynamic_v4/v4_training_authorization_v2.json"
TRAINING_CODE_REGISTRY="data/water_impact_dynamic_v4/v4_training_code_registry_v2.json"
RUNTIME_REGISTRY="data/water_impact_dynamic_v4/v4_runtime_registry_v2.json"
OUTPUT_DIR="outputs/water_impact_dynamic_v4/adapter_source_slot_randomized_v2"

EXPECTED_MANIFEST_SHA256="3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
EXPECTED_BASE_CACHE_SHA256="4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
EXPECTED_TEACHER_CACHE_SHA256="6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
EXPECTED_SOURCE_BANK_REGISTRY_SHA256="473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814"
EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256="6751a4d3b66491328909853b99bc8e6d06468a30b71f5bb746c7a744692fe84d"
EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256="0d7fab1befdc197a7ae7f864a84c1f1ac3d029d5d72f9a513303892e48ec2477"
EXPECTED_SOURCE_MAPPING_REGISTRY_SHA256="6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2"
EXPECTED_ACTIVE100_MAPPING_SHA256="e8ecf1b10fe6a7787997c17e86d95281a206af60c9e1ce93aa6c28f3040dc8d4"
EXPECTED_FULL178_MAPPING_SHA256="658aa5c01368f63e0076fc31f493370fae7aa4d3a3599218d47642ee66c9ed85"
EXPECTED_PROMPT_SIDECAR_INVENTORY_SHA256="TO_BE_FROZEN_AFTER_PROMPT_PREPARATION"
EXPECTED_PROMPT_SIDECAR_MANIFEST_SHA256="TO_BE_FROZEN_AFTER_PROMPT_PREPARATION"
EXPECTED_MODEL_CONTENT_INVENTORY_SHA256="0a8566eeab29dfbc04303167ce1904b65b964dd1579959645d1f93e19ba15ddf"
EXPECTED_PREFLIGHT_ARTIFACT_SHA256="TO_BE_FROZEN_AFTER_NULL_SIDECAR_PREFLIGHT"
EXPECTED_RUNTIME_REGISTRY_SHA256="TO_BE_FROZEN_AFTER_RUNTIME_REGISTRY"
# The authorization is the final, externally audited root of trust.  Its hash
# cannot be embedded here because the authorization binds a code registry that
# itself binds this launcher.  The formal command must supply the independently
# audited digest via V4_TRAINING_AUTHORIZATION_SHA256.
EXPECTED_TRAINING_AUTHORIZATION_SHA256="${V4_TRAINING_AUTHORIZATION_SHA256:-}"

CALIBRATION_ID="v4_retain_v3b_lambda4_first16_output_gradient_v1"
TEACHER_WEIGHT="4.0"
SANITY_MEAN_MIN="0.2"
SANITY_MEAN_MAX="0.5"
SANITY_SINGLE_MAX="1.0"

if [[ "${1:-}" != "train" || $# -ne 1 ]]; then
  echo "usage: V4_TRAINING_AUTHORIZATION_SHA256=<independently-audited-sha256> $0 train" >&2
  exit 2
fi
if [[ "${WAN_DEVICE:-cuda}" != "cuda" || "${WAN_MODEL:-$MODEL}" != "$MODEL" ]]; then
  echo "v4 requires the registered cuda device and frozen Wan model path" >&2
  exit 1
fi

require_frozen_sha256() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$label is not frozen: $value" >&2
    exit 1
  fi
}

for pair in \
  "source bank|$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \
  "holdout public commitment|$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" \
  "causal Stage-0 public commitment|$EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256" \
  "source mapping|$EXPECTED_SOURCE_MAPPING_REGISTRY_SHA256" \
  "active100 mapping|$EXPECTED_ACTIVE100_MAPPING_SHA256" \
  "full178 mapping|$EXPECTED_FULL178_MAPPING_SHA256" \
  "prompt sidecar inventory|$EXPECTED_PROMPT_SIDECAR_INVENTORY_SHA256" \
  "prompt sidecar manifest|$EXPECTED_PROMPT_SIDECAR_MANIFEST_SHA256" \
  "model content inventory|$EXPECTED_MODEL_CONTENT_INVENTORY_SHA256" \
  "null-sidecar preflight|$EXPECTED_PREFLIGHT_ARTIFACT_SHA256" \
  "runtime registry|$EXPECTED_RUNTIME_REGISTRY_SHA256" \
  "training authorization (independent audit)|$EXPECTED_TRAINING_AUTHORIZATION_SHA256"
do
  require_frozen_sha256 "${pair%%|*}" "${pair#*|}"
done

if [[ ! -x "$PYTHON" ]]; then
  echo "registered v4 Python is missing or non-executable: $PYTHON" >&2
  exit 1
fi
"$PYTHON" scripts/build_water_impact_dynamic_v4_runtime_registry.py validate \
  --output "$RUNTIME_REGISTRY" \
  --expected-sha256 "$EXPECTED_RUNTIME_REGISTRY_SHA256"

verify_file() {
  local path="$1"
  local expected="$2"
  "$PYTHON" - "$path" "$expected" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit(f"missing or symlinked frozen file: {path}")
actual = sha256(path.read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"frozen file hash mismatch: {path}: {actual} != {sys.argv[2]}")
print(f"Validated {path}: {actual}")
PY
}

verify_cache_inventory() {
  local cache_dir="$1"
  local expected_digest="$2"
  local role="$3"
  "$PYTHON" - "$MANIFEST" "$cache_dir" "$expected_digest" "$role" <<'PY'
import csv
from hashlib import sha256
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
expected_digest = sys.argv[3]
role = sys.argv[4]
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
selected = [
    (index, row)
    for index, row in enumerate(rows)
    if role == "all" or row["training_role"] == role
]
expected = [cache_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in selected]
actual = sorted(cache_dir.glob("*.pt"))
if set(actual) != set(expected) or len(actual) != len(expected):
    raise SystemExit(
        f"cache inventory mismatch: {cache_dir}: expected={len(expected)} actual={len(actual)}"
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
    raise SystemExit(f"cache inventory hash mismatch: {actual_digest} != {expected_digest}")
print(f"Validated {len(expected)} cache entries in {cache_dir}: {actual_digest}")
PY
}

verify_model_inventory() {
  "$PYTHON" - "$MODEL" "$EXPECTED_MODEL_CONTENT_INVENTORY_SHA256" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

model = Path(sys.argv[1])
expected = sys.argv[2]
paths = []
excluded_suffixes = (".tmp", ".lock", ".incomplete", "~")
for path in model.rglob("*"):
    relative = path.relative_to(model)
    if ".cache" in relative.parts:
        continue
    if path.is_symlink():
        raise SystemExit(f"model inventory forbids symlinks: {path}")
    if path.is_file() and not path.name.endswith(excluded_suffixes):
        paths.append(path)
paths.sort(key=lambda path: path.relative_to(model).as_posix())
required = {
    "model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer_config.json",
}
names = {path.relative_to(model).as_posix() for path in paths}
if missing := required - names:
    raise SystemExit(f"model content inventory missing required files: {sorted(missing)}")
digest = sha256()
for path in paths:
    digest.update(path.relative_to(model).as_posix().encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\n")
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"model content inventory mismatch: {actual} != {expected}")
print(f"Validated full model content inventory: {actual}")
PY
}

verify_mapping_headers() {
  "$PYTHON" - \
    "$SOURCE_MAPPING_REGISTRY" \
    "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \
    "$HOLDOUT_PUBLIC_COMMITMENT" \
    "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" \
    "$EXPECTED_ACTIVE100_MAPPING_SHA256" \
    "$EXPECTED_FULL178_MAPPING_SHA256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "protocol": "water_impact_dynamic_v4_source_mapping_v2",
    "status": "frozen",
    "dataset_version": "v4_dev72_v2",
    "source_bank_registry_sha256": sys.argv[2],
    "holdout_public_commitment_path": sys.argv[3],
    "holdout_public_commitment_sha256": sys.argv[4],
    "holdout_count": 24,
    "active100_mapping_sha256": sys.argv[5],
    "full178_mapping_sha256": sys.argv[6],
    "sample_order_sha256": "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb",
    "erase_row_count": 178,
    "active_erase_count": 100,
    "active_source_count_min": 1,
    "active_source_count_max": 2,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"mapping registry {key} mismatch")
prompt_builder = Path("scripts/build_water_impact_dynamic_pairs_v1.py")
if prompt_builder.is_symlink() or not prompt_builder.is_file():
    raise SystemExit("canonical prompt builder is missing or symlinked")
if payload.get("canonical_prompt_builder_path") != str(prompt_builder):
    raise SystemExit("mapping canonical prompt-builder path mismatch")
if payload.get("canonical_prompt_builder_sha256") != sha256(prompt_builder.read_bytes()).hexdigest():
    raise SystemExit("mapping canonical prompt-builder hash mismatch")
print("Validated frozen source mapping headers")
PY
}

verify_authorization_chain() {
  "$PYTHON" - \
    "$TRAINING_AUTHORIZATION" \
    "$EXPECTED_TRAINING_AUTHORIZATION_SHA256" \
    "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \
    "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

HEX64 = re.compile(r"[0-9a-f]{64}")
DATASET_VERSION = "v4_dev72_v2"
COMMITMENT_PROTOCOL = "water_impact_dynamic_v4_eval_commitment_registry_v2"
GATE_PROTOCOL = "water_impact_dynamic_v4_machine_gate_registry_v2"
CODE_PROTOCOL = "water_impact_dynamic_v4_training_code_registry_v2"
EXPECTED_GATE_SPEC_SHA256 = "ad0dd71de512572b456f7f46b2d6fbe3eeaf4956bf6d3870de5148b887058b11"
paths = {
    "source_bank_registry": "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json",
    "holdout_public_commitment": "data/water_impact_dynamic_v4/holdout_public_commitment_v2.json",
    "causal_stage0": "data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json",
    "causal_stage1": "data/water_impact_dynamic_v4/causal_stage1_commitment_v2.json",
    "specificity_stage0": "data/water_impact_dynamic_v4/specificity_stage0_commitment_v2.json",
    "specificity_stage1": "data/water_impact_dynamic_v4/specificity_stage1_commitment_v2.json",
    "gate_registry": "data/water_impact_dynamic_v4/v4_machine_gate_registry_v2.json",
    "runtime_registry": "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json",
    "code_registry": "data/water_impact_dynamic_v4/v4_training_code_registry_v2.json",
}
code_paths = {
    "trainer": "scripts/train_wan_waterdrop_lora_v4.py",
    "launcher": "scripts/run_water_impact_dynamic_sft_v4_source_slot.sh",
    "source_mapping": "scripts/build_water_impact_dynamic_v4_source_mapping.py",
    "preparer": "scripts/prepare_water_impact_dynamic_v4_prompt_cache.py",
    "runtime_registry_builder": "scripts/build_water_impact_dynamic_v4_runtime_registry.py",
    "design_doc": "docs/water_impact_dynamic_v4_source_slot_randomization.md",
    "eval_protocol": "scripts/water_impact_dynamic_v4_eval_protocol.py",
    "eval_selector": "scripts/select_water_impact_dynamic_v4_eval.py",
    "eval_blind_builder": "scripts/build_water_impact_dynamic_v4_blind_review.py",
    "eval_scorer": "scripts/score_water_impact_dynamic_v4.py",
    "eval_runner": "scripts/run_water_impact_dynamic_v4_eval.py",
    "generator": "scripts/generate_wan_clean.py",
}
stage_artifacts = {
    ("causal", 0): (
        "candidate_manifest_48", "source_bank_registry_64", "source_ontology_80", "source_split_80",
        "holdout_registry_24", "receiver_ontology_32",
        "canonical_templates", "field_normalization", "raw_root_bundle",
        "raw_render_configuration", "stage0_secrets",
        "screening_seed", "screening_generation_spec", "selector_salt",
        "ranking_formula", "constrained_subset_algorithm", "evaluation_seed_salt",
        "seed_derivation_formula", "forbidden_seed_inventory",
    ),
    ("causal", 1): (
        "screening_generation_manifest", "screening_candidate_binding",
        "screening_review_a", "screening_review_b", "screening_dispute_template",
        "screening_adjudication", "screening_freeze_manifest", "eligibility_table_48",
        "selector_output_24", "selected_case_manifest_24", "unit_manifest_U_72",
    ),
    ("specificity", 0): (
        "candidate_manifest_36", "new_bank_selection_and_receiver_assignment",
        "canonical_templates", "field_normalization", "raw_root_bundle",
        "raw_render_configuration", "stage0_secrets", "screening_seed",
        "screening_generation_spec", "selector_salt", "ranking_formula",
        "constrained_subset_algorithm", "evaluation_seed_salt",
        "seed_derivation_formula", "forbidden_seed_inventory",
    ),
    ("specificity", 1): (
        "screening_generation_manifest", "screening_candidate_binding",
        "screening_review_a", "screening_review_b", "screening_dispute_template",
        "screening_adjudication", "screening_freeze_manifest", "eligibility_table_36",
        "selector_output_18", "selected_case_manifest_18", "unit_manifest_W_36",
        "holdout_mapping_M_6",
    ),
}
expected_row_counts = {
    ("causal", 0, "candidate_manifest_48"): 48,
    ("causal", 0, "source_bank_registry_64"): 64,
    ("causal", 0, "source_ontology_80"): 80,
    ("causal", 0, "source_split_80"): 80,
    ("causal", 0, "holdout_registry_24"): 24,
    ("causal", 0, "receiver_ontology_32"): 32,
    ("causal", 1, "screening_review_a"): 48,
    ("causal", 1, "screening_review_b"): 48,
    ("causal", 1, "screening_candidate_binding"): 48,
    ("causal", 1, "eligibility_table_48"): 48,
    ("causal", 1, "selected_case_manifest_24"): 24,
    ("causal", 1, "unit_manifest_U_72"): 72,
    ("specificity", 0, "candidate_manifest_36"): 36,
    ("specificity", 0, "new_bank_selection_and_receiver_assignment"): 12,
    ("specificity", 1, "screening_review_a"): 36,
    ("specificity", 1, "screening_review_b"): 36,
    ("specificity", 1, "screening_candidate_binding"): 36,
    ("specificity", 1, "eligibility_table_36"): 36,
    ("specificity", 1, "selected_case_manifest_18"): 18,
    ("specificity", 1, "unit_manifest_W_36"): 36,
    ("specificity", 1, "holdout_mapping_M_6"): 6,
}

def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()

def load_public(path, label):
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"{label} is invalid UTF-8 JSON: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"{label} JSON root must be an object")
    return value

def contains_placeholder(value):
    if isinstance(value, dict):
        return any(contains_placeholder(key) or contains_placeholder(child) for key, child in value.items())
    if isinstance(value, list):
        return any(contains_placeholder(child) for child in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(
            token in lowered
            for token in ("placeholder", "todo", "tbd", "fill_me", "to_be_frozen")
        )
    return False

authorization_path = Path(sys.argv[1])
payload = load_public(authorization_path, "training authorization")
if file_hash(authorization_path) != sys.argv[2]:
    raise SystemExit("training authorization differs from independently audited hash")
if payload.get("protocol") != "water_impact_dynamic_v4_training_authorization_v2":
    raise SystemExit("training authorization protocol mismatch")
if payload.get("status") != "authorized":
    raise SystemExit("training is not authorized")
if payload.get("dataset_version") != DATASET_VERSION:
    raise SystemExit("training authorization dataset version mismatch")
if payload.get("sealed_final36_status") != "unopened":
    raise SystemExit("training authorization does not attest final36 is unopened")
if set(payload) != {"protocol", "status", "dataset_version", "sealed_final36_status", *paths}:
    raise SystemExit("training authorization fields mismatch")
resolved = {}
for name, expected_path in paths.items():
    record = payload[name]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SystemExit(f"invalid authorization ref: {name}")
    if record["path"] != expected_path:
        raise SystemExit(f"invalid authorization path: {name}")
    if not isinstance(record["sha256"], str) or not HEX64.fullmatch(record["sha256"]):
        raise SystemExit(f"unfrozen authorization hash: {name}")
    ref_path = Path(expected_path)
    ref_payload = load_public(ref_path, f"authorization ref {name}")
    if file_hash(ref_path) != record["sha256"]:
        raise SystemExit(f"authorization ref byte hash mismatch: {name}")
    if contains_placeholder(ref_payload):
        raise SystemExit(f"authorization ref contains a placeholder: {name}")
    resolved[name] = (ref_path, ref_payload)

for dataset in ("causal", "specificity"):
    stage0_name = f"{dataset}_stage0"
    stage1_name = f"{dataset}_stage1"
    for stage, name in enumerate((stage0_name, stage1_name)):
        stage_payload = resolved[name][1]
        exact = {"protocol", "dataset", "dataset_version", "stage", "status", "sealed_final36_status", "artifacts"}
        if stage == 1:
            exact.add("stage0_registry_sha256")
        if set(stage_payload) != exact:
            raise SystemExit(f"{name} fields mismatch")
        if (
            stage_payload["protocol"] != COMMITMENT_PROTOCOL
            or stage_payload["dataset"] != dataset
            or stage_payload["dataset_version"] != DATASET_VERSION
            or stage_payload["stage"] != stage
            or stage_payload["status"] != "committed"
            or stage_payload["sealed_final36_status"] != "unopened"
        ):
            raise SystemExit(f"{name} public commitment headers mismatch")
        artifacts = stage_payload["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(stage_artifacts[(dataset, stage)]):
            raise SystemExit(f"{name} public artifact commitments are not exact")
        for artifact_name, record in artifacts.items():
            if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes", "row_count"}:
                raise SystemExit(f"{name}/{artifact_name} commitment fields mismatch")
            if not isinstance(record["sha256"], str) or not HEX64.fullmatch(record["sha256"]):
                raise SystemExit(f"{name}/{artifact_name} commitment hash is invalid")
            if not isinstance(record["size_bytes"], int) or record["size_bytes"] <= 0:
                raise SystemExit(f"{name}/{artifact_name} commitment size is invalid")
            if record["row_count"] is not None and (
                not isinstance(record["row_count"], int) or record["row_count"] < 0
            ):
                raise SystemExit(f"{name}/{artifact_name} commitment row count is invalid")
            expected_rows = expected_row_counts.get((dataset, stage, artifact_name))
            if expected_rows is not None and record["row_count"] != expected_rows:
                raise SystemExit(f"{name}/{artifact_name} commitment row count mismatch")
    if resolved[stage1_name][1]["stage0_registry_sha256"] != payload[stage0_name]["sha256"]:
        raise SystemExit(f"{stage1_name} does not bind exact Stage-0 bytes")

if payload["source_bank_registry"]["sha256"] != sys.argv[3]:
    raise SystemExit("training authorization source-bank hash differs from launcher")
if payload["holdout_public_commitment"]["sha256"] != sys.argv[4]:
    raise SystemExit("training authorization holdout hash differs from launcher")
bank = resolved["source_bank_registry"][1]
holdout = resolved["holdout_public_commitment"][1]
causal_stage0 = resolved["causal_stage0"][1]
if causal_stage0["artifacts"]["source_bank_registry_64"]["sha256"] != payload["source_bank_registry"]["sha256"]:
    raise SystemExit("causal Stage-0 source bank differs from training authorization")
if causal_stage0["artifacts"]["holdout_registry_24"]["sha256"] != holdout.get("holdout_registry_file_sha256"):
    raise SystemExit("causal Stage-0 holdout registry differs from public commitment")

gate = resolved["gate_registry"][1]
if set(gate) != {
    "protocol", "status", "dataset_version", "sealed_final36_status", "gate_spec", "gate_spec_sha256", "scorer_sha256"
}:
    raise SystemExit("machine gate registry fields mismatch")
if gate["protocol"] != GATE_PROTOCOL or gate["status"] != "frozen" or gate["dataset_version"] != DATASET_VERSION or gate["sealed_final36_status"] != "unopened":
    raise SystemExit("machine gate registry headers mismatch")
if not isinstance(gate["gate_spec"], dict) or not gate["gate_spec"]:
    raise SystemExit("machine gate registry has no gate spec")
gate_digest = sha256(
    json.dumps(gate["gate_spec"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
).hexdigest()
if gate["gate_spec_sha256"] != gate_digest:
    raise SystemExit("machine gate spec digest mismatch")
if gate["gate_spec_sha256"] != EXPECTED_GATE_SPEC_SHA256:
    raise SystemExit("machine gate spec differs from canonical protocol")

code = resolved["code_registry"][1]
if set(code) != {"protocol", "status", "runtime_registry", "artifacts"}:
    raise SystemExit("training code registry fields mismatch")
if code["protocol"] != CODE_PROTOCOL or code["status"] != "frozen":
    raise SystemExit("training code registry is not frozen")
if code["runtime_registry"] != payload["runtime_registry"]:
    raise SystemExit("training code registry runtime reference mismatch")
if not isinstance(code["artifacts"], dict) or set(code["artifacts"]) != set(code_paths):
    raise SystemExit("training code registry artifact inventory mismatch")
for name, expected_path in code_paths.items():
    record = code["artifacts"][name]
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SystemExit(f"training code registry/{name} fields mismatch")
    if record["path"] != expected_path or not isinstance(record["sha256"], str) or not HEX64.fullmatch(record["sha256"]):
        raise SystemExit(f"training code registry/{name} identity mismatch")
    artifact = Path(expected_path)
    if artifact.is_symlink() or not artifact.is_file() or file_hash(artifact) != record["sha256"]:
        raise SystemExit(f"training code registry/{name} byte hash mismatch")
if gate["scorer_sha256"] != code["artifacts"]["eval_scorer"]["sha256"]:
    raise SystemExit("machine gate scorer differs from training code registry")
if contains_placeholder(payload):
    raise SystemExit("training authorization contains a placeholder")
print("Validated training authorization, public Stage/gate refs, and frozen code registry")
PY
}

read_authorization_git_provenance() {
  "$PYTHON" - "$TRAINING_AUTHORIZATION" "$EXPECTED_TRAINING_AUTHORIZATION_SHA256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

path = Path(sys.argv[1])
expected = sys.argv[2]
try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    upstream = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, upstream],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    authorization = json.loads(path.read_text(encoding="utf-8"))
    registered = {path.as_posix(): expected}
    for name in (
        "source_bank_registry", "holdout_public_commitment",
        "causal_stage0", "causal_stage1", "specificity_stage0",
        "specificity_stage1", "gate_registry", "runtime_registry", "code_registry",
    ):
        record = authorization[name]
        registered[record["path"]] = record["sha256"]
    code_registry_path = Path(authorization["code_registry"]["path"])
    code_registry = json.loads(code_registry_path.read_text(encoding="utf-8"))
    for record in code_registry["artifacts"].values():
        registered[record["path"]] = record["sha256"]
    for registered_path, registered_sha256 in registered.items():
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", registered_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for command in (
            ["git", "diff", "--quiet", "--", registered_path],
            ["git", "diff", "--cached", "--quiet", "--", registered_path],
        ):
            subprocess.run(command, check=True)
        committed = subprocess.check_output(
            ["git", "show", f"{head}:{registered_path}"]
        )
        if sha256(committed).hexdigest() != registered_sha256:
            raise SystemExit(
                f"committed frozen artifact differs from registered hash: {registered_path}"
            )
except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    raise SystemExit(
        "authorization chain must be tracked, unchanged, committed, and present in the configured upstream"
    ) from exc
print(f"{head}\t{upstream}")
PY
}

verify_file "$MANIFEST" "$EXPECTED_MANIFEST_SHA256"
verify_file "$SOURCE_BANK_REGISTRY" "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256"
verify_file "$HOLDOUT_PUBLIC_COMMITMENT" "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256"
verify_file "$CAUSAL_STAGE0_PUBLIC_COMMITMENT" "$EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256"
verify_file "$SOURCE_MAPPING_REGISTRY" "$EXPECTED_SOURCE_MAPPING_REGISTRY_SHA256"
verify_file "$PROMPT_SIDECAR_DIR/cache_manifest_v2.json" "$EXPECTED_PROMPT_SIDECAR_MANIFEST_SHA256"
verify_file "$PREFLIGHT_ARTIFACT" "$EXPECTED_PREFLIGHT_ARTIFACT_SHA256"
verify_file "$RUNTIME_REGISTRY" "$EXPECTED_RUNTIME_REGISTRY_SHA256"
verify_file "$TRAINING_AUTHORIZATION" "$EXPECTED_TRAINING_AUTHORIZATION_SHA256"
verify_cache_inventory "$BASE_CACHE_DIR" "$EXPECTED_BASE_CACHE_SHA256" all
verify_cache_inventory "$TEACHER_CACHE_DIR" "$EXPECTED_TEACHER_CACHE_SHA256" erase
verify_cache_inventory "$PROMPT_SIDECAR_DIR" "$EXPECTED_PROMPT_SIDECAR_INVENTORY_SHA256" erase
verify_model_inventory
verify_mapping_headers
verify_authorization_chain
GIT_PROVENANCE="$(read_authorization_git_provenance)"
IFS=$'\t' read -r REGISTERED_GIT_COMMIT REGISTERED_GIT_UPSTREAM <<< "$GIT_PROVENANCE"

EXPECTED_TRAINING_CODE_REGISTRY_SHA256="$($PYTHON - "$TRAINING_AUTHORIZATION" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["code_registry"]["sha256"])
PY
)"

if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
  echo "refusing to reuse or race on v4 output path: $OUTPUT_DIR" >&2
  exit 1
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) training_authorization_sha256=$EXPECTED_TRAINING_AUTHORIZATION_SHA256" \
  > "$OUTPUT_DIR/.run_reservation"

"$PYTHON" - \
  "$OUTPUT_DIR" \
  "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \
  "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" \
  "$EXPECTED_SOURCE_MAPPING_REGISTRY_SHA256" \
  "$EXPECTED_ACTIVE100_MAPPING_SHA256" \
  "$EXPECTED_FULL178_MAPPING_SHA256" \
  "$EXPECTED_PROMPT_SIDECAR_INVENTORY_SHA256" \
  "$EXPECTED_PROMPT_SIDECAR_MANIFEST_SHA256" \
  "$EXPECTED_MODEL_CONTENT_INVENTORY_SHA256" \
  "$EXPECTED_PREFLIGHT_ARTIFACT_SHA256" \
  "$EXPECTED_TRAINING_AUTHORIZATION_SHA256" \
  "$EXPECTED_TRAINING_CODE_REGISTRY_SHA256" \
  "$REGISTERED_GIT_COMMIT" \
  "$REGISTERED_GIT_UPSTREAM" \
  "$EXPECTED_RUNTIME_REGISTRY_SHA256" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
code_registry_path = Path("data/water_impact_dynamic_v4/v4_training_code_registry_v2.json")
code_artifacts = json.loads(code_registry_path.read_text(encoding="utf-8"))["artifacts"]
mapping_registry = json.loads(
    Path("data/water_impact_dynamic_v4/source_mapping_v2.json").read_text(encoding="utf-8")
)
registration = {
    "protocol": "water_impact_dynamic_v4_source_slot_randomized_teacher_v2",
    "status": "registered",
    "dataset_version": "v4_dev72_v2",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(output),
    "only_training_intervention": (
        "erase factual prompt_embeds replaced by registered augmented source-slot sidecar"
    ),
    "train_manifest_path": "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv",
    "train_manifest_sha256": "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4",
    "base_cache_path": "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2",
    "base_cache_inventory_sha256": "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65",
    "teacher_cache_path": "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1",
    "teacher_cache_inventory_sha256": "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9",
    "source_bank_registry_path": "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json",
    "source_bank_registry_sha256": sys.argv[2],
    "holdout_public_commitment_path": "data/water_impact_dynamic_v4/holdout_public_commitment_v2.json",
    "holdout_public_commitment_sha256": sys.argv[3],
    "holdout_count": 24,
    "source_mapping_registry_path": "data/water_impact_dynamic_v4/source_mapping_v2.json",
    "source_mapping_registry_sha256": sys.argv[4],
    "active100_mapping_sha256": sys.argv[5],
    "full178_mapping_sha256": sys.argv[6],
    "canonical_prompt_builder_path": mapping_registry["canonical_prompt_builder_path"],
    "canonical_prompt_builder_sha256": mapping_registry["canonical_prompt_builder_sha256"],
    "prompt_sidecar_path": "outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2",
    "prompt_sidecar_inventory_sha256": sys.argv[7],
    "prompt_sidecar_manifest_sha256": sys.argv[8],
    "model_content_inventory_sha256": sys.argv[9],
    "transformer_inventory_sha256": "fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac",
    "preflight_artifact_path": "outputs/water_impact_dynamic_v4/null_sidecar_preflight_v2.json",
    "preflight_artifact_sha256": sys.argv[10],
    "training_authorization_path": "data/water_impact_dynamic_v4/v4_training_authorization_v2.json",
    "training_authorization_sha256": sys.argv[11],
    "training_code_registry_path": str(code_registry_path),
    "training_code_registry_sha256": sys.argv[12],
    "runtime_registry_path": "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json",
    "runtime_registry_sha256": sys.argv[15],
    "authorization_source": "independent_audited_committed_and_pushed",
    "git_commit": sys.argv[13],
    "git_upstream": sys.argv[14],
    "expected_initial_lora_sha256": "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8",
    "expected_noise_sigma_rng_initial_sha256": "49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704",
    "expected_noise_sigma_rng_final_sha256": "79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f",
    "expected_sample_order_sha256": "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb",
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
        "save_every": 200,
        "seed": 26000,
        "device": "cuda",
        "role": "all",
        "objective": "source_slot_target_prompt_teacher",
        "balanced_roles": True,
        "preserve_weight": 4.0,
        "target_prompt_teacher_weight": 4.0,
        "target_prompt_calibration_id": "v4_retain_v3b_lambda4_first16_output_gradient_v1",
        "sanity_mean_min": 0.2,
        "sanity_mean_max": 0.5,
        "sanity_single_max": 1.0,
    },
}
for name, record in code_artifacts.items():
    registration[f"{name}_path"] = record["path"]
    registration[f"{name}_sha256"] = record["sha256"]
path = output / "run_registration_v2.json"
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(registration, handle, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

RUN_REGISTRATION_SHA256="$($PYTHON - "$OUTPUT_DIR/run_registration_v2.json" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

exec "$PYTHON" scripts/train_wan_waterdrop_lora_v4.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --cache-dir "$BASE_CACHE_DIR" \
  --target-prompt-cache-dir "$TEACHER_CACHE_DIR" \
  --source-bank-registry "$SOURCE_BANK_REGISTRY" \
  --source-bank-registry-sha256 "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \
  --holdout-public-commitment "$HOLDOUT_PUBLIC_COMMITMENT" \
  --holdout-public-commitment-sha256 "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" \
  --source-mapping-registry "$SOURCE_MAPPING_REGISTRY" \
  --source-mapping-registry-sha256 "$EXPECTED_SOURCE_MAPPING_REGISTRY_SHA256" \
  --prompt-sidecar-dir "$PROMPT_SIDECAR_DIR" \
  --prompt-sidecar-inventory-sha256 "$EXPECTED_PROMPT_SIDECAR_INVENTORY_SHA256" \
  --prompt-sidecar-manifest-sha256 "$EXPECTED_PROMPT_SIDECAR_MANIFEST_SHA256" \
  --model-content-inventory-sha256 "$EXPECTED_MODEL_CONTENT_INVENTORY_SHA256" \
  --runtime-registry "$RUNTIME_REGISTRY" \
  --runtime-registry-sha256 "$EXPECTED_RUNTIME_REGISTRY_SHA256" \
  --preflight-artifact "$PREFLIGHT_ARTIFACT" \
  --preflight-artifact-sha256 "$EXPECTED_PREFLIGHT_ARTIFACT_SHA256" \
  --training-authorization "$TRAINING_AUTHORIZATION" \
  --training-authorization-sha256 "$EXPECTED_TRAINING_AUTHORIZATION_SHA256" \
  --training-code-registry "$TRAINING_CODE_REGISTRY" \
  --training-code-registry-sha256 "$EXPECTED_TRAINING_CODE_REGISTRY_SHA256" \
  --run-registration "$OUTPUT_DIR/run_registration_v2.json" \
  --run-registration-sha256 "$RUN_REGISTRATION_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --max-steps 200 \
  --learning-rate 5e-5 \
  --rank 16 \
  --alpha 16 \
  --grad-accum 1 \
  --save-every 200 \
  --seed 26000 \
  --device cuda \
  --role all \
  --objective source_slot_target_prompt_teacher \
  --balanced-roles \
  --preserve-weight 4.0 \
  --target-prompt-teacher-weight "$TEACHER_WEIGHT" \
  --target-prompt-calibration-id "$CALIBRATION_ID" \
  --target-prompt-sanity-min-output-grad-ratio "$SANITY_MEAN_MIN" \
  --target-prompt-sanity-max-output-grad-ratio "$SANITY_MEAN_MAX" \
  --target-prompt-sanity-max-single-output-grad-ratio "$SANITY_SINGLE_MAX"
