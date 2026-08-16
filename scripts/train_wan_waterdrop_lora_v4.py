#!/usr/bin/env python3
"""Train the registered v4 source-slot-randomized Wan LoRA.

The only treatment relative to v3b is the erase-row factual prompt embedding
loaded from the frozen v4 sidecar.  Counterfactual latents, target-prompt
teacher, preserve branch, loss weights, sample order, RNG, initialization,
optimizer, and step budget remain frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict

from build_water_impact_dynamic_v4_runtime_registry import validate_runtime_registry
from build_water_impact_dynamic_v4_source_mapping import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SAMPLE_ORDER_SHA256,
    EXPECTED_SEED,
    balanced_v3b_schedule,
    canonical_json_sha256,
    file_sha256,
    load_frozen_rows,
    require_sha256,
    sample_order_sha256,
    validate_public_bank_registry,
    validate_public_holdout_commitment,
    validate_public_stage0_commitment,
)
from prepare_water_impact_dynamic_v4_prompt_cache import (
    EXPECTED_BASE_CACHE_SHA256,
    EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT,
    EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256,
    EXPECTED_INITIAL_LORA_SHA256,
    EXPECTED_MODEL_CONTENT_INVENTORY_SHA256,
    EXPECTED_NOISE_RNG_FINAL_SHA256,
    EXPECTED_NOISE_RNG_INITIAL_SHA256,
    EXPECTED_TEACHER_CACHE_SHA256,
    PREFLIGHT_PROTOCOL,
    cache_inventory_sha256,
    exact_cache_paths,
    load_mapping_registry,
    tensor_sha256,
    tokenizer_inventory_binding,
    trainable_state_sha256,
    validate_cache_inventory,
    validate_model_content_inventory,
    validate_prompt_sidecar,
)
from water_impact_dynamic_v4_eval_protocol import GATE_SPEC as EXPECTED_GATE_SPEC


DATASET_VERSION = "v4_dev72_v2"
PROTOCOL = "water_impact_dynamic_v4_source_slot_randomized_teacher_v2"
SANITY_PROTOCOL = "water_impact_dynamic_v4_source_slot_scale_sanity_v2"
ELIGIBILITY_PROTOCOL = "water_impact_dynamic_v4_checkpoint_eligibility_v2"
FINAL_LORA_FINITE_PROTOCOL = "water_impact_dynamic_v4_final_lora_finite_check_v2"
AUTHORIZATION_PROTOCOL = "water_impact_dynamic_v4_training_authorization_v2"
TRAINING_CODE_REGISTRY_PROTOCOL = (
    "water_impact_dynamic_v4_training_code_registry_v2"
)
COMMITMENT_PROTOCOL = "water_impact_dynamic_v4_eval_commitment_registry_v2"
GATE_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_machine_gate_registry_v2"
EXPECTED_GATE_SPEC_SHA256 = canonical_json_sha256(EXPECTED_GATE_SPEC)
FROZEN_TRANSFORMER_INVENTORY_SHA256 = (
    "fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac"
)
EXPECTED_OUTPUT_DIR = Path(
    "outputs/water_impact_dynamic_v4/adapter_source_slot_randomized_v2"
)
EXPECTED_MODEL = Path("models/Wan2.1-T2V-1.3B-Diffusers")
EXPECTED_MANIFEST = Path(
    "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
)
EXPECTED_BASE_CACHE_DIR = Path(
    "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
)
EXPECTED_TEACHER_CACHE_DIR = Path(
    "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
)
EXPECTED_BANK_REGISTRY = Path(
    "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json"
)
EXPECTED_HOLDOUT_PUBLIC_COMMITMENT = Path(
    "data/water_impact_dynamic_v4/holdout_public_commitment_v2.json"
)
EXPECTED_BANK_REGISTRY_SHA256 = (
    "473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814"
)
EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256 = (
    "6751a4d3b66491328909853b99bc8e6d06468a30b71f5bb746c7a744692fe84d"
)
EXPECTED_MAPPING_REGISTRY = Path(
    "data/water_impact_dynamic_v4/source_mapping_v2.json"
)
EXPECTED_PROMPT_SIDECAR_DIR = Path(
    "outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2"
)
EXPECTED_PREFLIGHT = Path(
    "outputs/water_impact_dynamic_v4/null_sidecar_preflight_v2.json"
)
EXPECTED_AUTHORIZATION = Path(
    "data/water_impact_dynamic_v4/v4_training_authorization_v2.json"
)
EXPECTED_TRAINING_CODE_REGISTRY = Path(
    "data/water_impact_dynamic_v4/v4_training_code_registry_v2.json"
)
EXPECTED_RUNTIME_REGISTRY = Path(
    "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json"
)
AUTHORIZATION_REF_PATHS = {
    "source_bank_registry": str(EXPECTED_BANK_REGISTRY),
    "holdout_public_commitment": str(EXPECTED_HOLDOUT_PUBLIC_COMMITMENT),
    "causal_stage0": "data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json",
    "causal_stage1": "data/water_impact_dynamic_v4/causal_stage1_commitment_v2.json",
    "specificity_stage0": "data/water_impact_dynamic_v4/specificity_stage0_commitment_v2.json",
    "specificity_stage1": "data/water_impact_dynamic_v4/specificity_stage1_commitment_v2.json",
    "gate_registry": "data/water_impact_dynamic_v4/v4_machine_gate_registry_v2.json",
    "runtime_registry": "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json",
    "code_registry": "data/water_impact_dynamic_v4/v4_training_code_registry_v2.json",
}
# These names are public commitment interfaces only.  The trainer validates
# their digest records and never follows or opens the committed artifacts.
STAGE_ARTIFACTS: dict[tuple[str, int], tuple[str, ...]] = {
    ("causal", 0): (
        "candidate_manifest_48",
        "source_bank_registry_64",
        "source_ontology_80",
        "source_split_80",
        "holdout_registry_24",
        "receiver_ontology_32",
        "canonical_templates",
        "field_normalization",
        "raw_root_bundle",
        "raw_render_configuration",
        "stage0_secrets",
        "screening_seed",
        "screening_generation_spec",
        "selector_salt",
        "ranking_formula",
        "constrained_subset_algorithm",
        "evaluation_seed_salt",
        "seed_derivation_formula",
        "forbidden_seed_inventory",
    ),
    ("causal", 1): (
        "screening_generation_manifest",
        "screening_candidate_binding",
        "screening_review_a",
        "screening_review_b",
        "screening_dispute_template",
        "screening_adjudication",
        "screening_freeze_manifest",
        "eligibility_table_48",
        "selector_output_24",
        "selected_case_manifest_24",
        "unit_manifest_U_72",
    ),
    ("specificity", 0): (
        "candidate_manifest_36",
        "new_bank_selection_and_receiver_assignment",
        "canonical_templates",
        "field_normalization",
        "raw_root_bundle",
        "raw_render_configuration",
        "stage0_secrets",
        "screening_seed",
        "screening_generation_spec",
        "selector_salt",
        "ranking_formula",
        "constrained_subset_algorithm",
        "evaluation_seed_salt",
        "seed_derivation_formula",
        "forbidden_seed_inventory",
    ),
    ("specificity", 1): (
        "screening_generation_manifest",
        "screening_candidate_binding",
        "screening_review_a",
        "screening_review_b",
        "screening_dispute_template",
        "screening_adjudication",
        "screening_freeze_manifest",
        "eligibility_table_36",
        "selector_output_18",
        "selected_case_manifest_18",
        "unit_manifest_W_36",
        "holdout_mapping_M_6",
    ),
}
EXPECTED_COMMITMENT_ROW_COUNTS: dict[tuple[str, int, str], int] = {
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
EXPECTED_CALIBRATION_ID = "v4_retain_v3b_lambda4_first16_output_gradient_v1"
EXPECTED_CONFIG: dict[str, Any] = {
    "model": str(EXPECTED_MODEL),
    "height": 480,
    "width": 832,
    "num_frames": 49,
    "max_steps": 200,
    "learning_rate": 5e-5,
    "rank": 16,
    "alpha": 16,
    "grad_accum": 1,
    "save_every": 200,
    "seed": EXPECTED_SEED,
    "device": "cuda",
    "role": "all",
    "objective": "source_slot_target_prompt_teacher",
    "balanced_roles": True,
    "preserve_weight": 4.0,
    "target_prompt_teacher_weight": 4.0,
    "target_prompt_calibration_id": EXPECTED_CALIBRATION_ID,
    "sanity_mean_min": 0.2,
    "sanity_mean_max": 0.5,
    "sanity_single_max": 1.0,
}
CODE_ARTIFACT_PATHS = {
    "trainer": Path("scripts/train_wan_waterdrop_lora_v4.py"),
    "launcher": Path("scripts/run_water_impact_dynamic_sft_v4_source_slot.sh"),
    "source_mapping": Path(
        "scripts/build_water_impact_dynamic_v4_source_mapping.py"
    ),
    "preparer": Path(
        "scripts/prepare_water_impact_dynamic_v4_prompt_cache.py"
    ),
    "runtime_registry_builder": Path(
        "scripts/build_water_impact_dynamic_v4_runtime_registry.py"
    ),
    "design_doc": Path(
        "docs/water_impact_dynamic_v4_source_slot_randomization.md"
    ),
    "eval_protocol": Path("scripts/water_impact_dynamic_v4_eval_protocol.py"),
    "eval_selector": Path("scripts/select_water_impact_dynamic_v4_eval.py"),
    "eval_blind_builder": Path(
        "scripts/build_water_impact_dynamic_v4_blind_review.py"
    ),
    "eval_scorer": Path("scripts/score_water_impact_dynamic_v4.py"),
    "eval_runner": Path("scripts/run_water_impact_dynamic_v4_eval.py"),
    "generator": Path("scripts/generate_wan_clean.py"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--target-prompt-cache-dir", type=Path, required=True)
    parser.add_argument("--source-bank-registry", type=Path, required=True)
    parser.add_argument("--source-bank-registry-sha256", required=True)
    parser.add_argument("--holdout-public-commitment", type=Path, required=True)
    parser.add_argument("--holdout-public-commitment-sha256", required=True)
    parser.add_argument("--source-mapping-registry", type=Path, required=True)
    parser.add_argument("--source-mapping-registry-sha256", required=True)
    parser.add_argument("--prompt-sidecar-dir", type=Path, required=True)
    parser.add_argument("--prompt-sidecar-inventory-sha256", required=True)
    parser.add_argument("--prompt-sidecar-manifest-sha256", required=True)
    parser.add_argument("--model-content-inventory-sha256", required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--runtime-registry-sha256", required=True)
    parser.add_argument("--preflight-artifact", type=Path, required=True)
    parser.add_argument("--preflight-artifact-sha256", required=True)
    parser.add_argument("--training-authorization", type=Path, required=True)
    parser.add_argument("--training-authorization-sha256", required=True)
    parser.add_argument("--training-code-registry", type=Path, required=True)
    parser.add_argument("--training-code-registry-sha256", required=True)
    parser.add_argument("--run-registration", type=Path, required=True)
    parser.add_argument("--run-registration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--balanced-roles", action="store_true")
    parser.add_argument("--preserve-weight", type=float, required=True)
    parser.add_argument("--target-prompt-teacher-weight", type=float, required=True)
    parser.add_argument("--target-prompt-calibration-id", required=True)
    parser.add_argument("--target-prompt-sanity-min-output-grad-ratio", type=float, required=True)
    parser.add_argument("--target-prompt-sanity-max-output-grad-ratio", type=float, required=True)
    parser.add_argument(
        "--target-prompt-sanity-max-single-output-grad-ratio", type=float, required=True
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if args.rebuild_cache or args.cache_only:
        parser.error("v4 formal trainer is read-only; cache rebuild/cache-only are forbidden")
    for name, value in vars(args).items():
        if name.endswith("sha256"):
            require_sha256(value, name.replace("_", " "))
    if args.model_content_inventory_sha256 != EXPECTED_MODEL_CONTENT_INVENTORY_SHA256:
        parser.error("v4 requires the frozen v3c Stage-2 full model inventory hash")
    actual_config = training_config(args)
    if actual_config != EXPECTED_CONFIG:
        parser.error(
            "v4 training configuration differs from the registered v3b control: "
            f"actual={actual_config!r} expected={EXPECTED_CONFIG!r}"
        )
    fixed_paths = {
        "manifest": EXPECTED_MANIFEST,
        "model": EXPECTED_MODEL,
        "cache_dir": EXPECTED_BASE_CACHE_DIR,
        "target_prompt_cache_dir": EXPECTED_TEACHER_CACHE_DIR,
        "source_bank_registry": EXPECTED_BANK_REGISTRY,
        "holdout_public_commitment": EXPECTED_HOLDOUT_PUBLIC_COMMITMENT,
        "source_mapping_registry": EXPECTED_MAPPING_REGISTRY,
        "prompt_sidecar_dir": EXPECTED_PROMPT_SIDECAR_DIR,
        "runtime_registry": EXPECTED_RUNTIME_REGISTRY,
        "preflight_artifact": EXPECTED_PREFLIGHT,
        "training_authorization": EXPECTED_AUTHORIZATION,
        "training_code_registry": EXPECTED_TRAINING_CODE_REGISTRY,
        "run_registration": EXPECTED_OUTPUT_DIR / "run_registration_v2.json",
        "output_dir": EXPECTED_OUTPUT_DIR,
    }
    for name, expected in fixed_paths.items():
        if getattr(args, name) != expected:
            parser.error(f"v4 requires frozen {name}={expected}")
    return args


def training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "alpha": args.alpha,
        "grad_accum": args.grad_accum,
        "save_every": args.save_every,
        "seed": args.seed,
        "device": args.device,
        "role": args.role,
        "objective": args.objective,
        "balanced_roles": args.balanced_roles,
        "preserve_weight": args.preserve_weight,
        "target_prompt_teacher_weight": args.target_prompt_teacher_weight,
        "target_prompt_calibration_id": args.target_prompt_calibration_id,
        "sanity_mean_min": args.target_prompt_sanity_min_output_grad_ratio,
        "sanity_mean_max": args.target_prompt_sanity_max_output_grad_ratio,
        "sanity_single_max": args.target_prompt_sanity_max_single_output_grad_ratio,
    }


def validate_canonical_public_hashes(args: argparse.Namespace) -> None:
    """Pin the audited public v2 bytes even when the launcher is bypassed."""

    if args.source_bank_registry_sha256 != EXPECTED_BANK_REGISTRY_SHA256:
        raise ValueError("source-bank registry differs from canonical public v2 bytes")
    if (
        args.holdout_public_commitment_sha256
        != EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256
    ):
        raise ValueError("holdout commitment differs from canonical public v2 bytes")


def validate_output_reservation(args: argparse.Namespace) -> None:
    if args.output_dir.is_symlink() or not args.output_dir.is_dir():
        raise FileNotFoundError("launcher must atomically reserve the v4 output directory")
    allowed = {".run_reservation", "run_registration_v2.json"}
    actual = {path.name for path in args.output_dir.iterdir()}
    if actual != allowed:
        raise ValueError(
            f"formal v4 output directory must contain exactly {sorted(allowed)}, got {sorted(actual)}"
        )
    if not (args.output_dir / ".run_reservation").is_file():
        raise ValueError("v4 output reservation marker is missing")


def _load_public_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a present non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(key) or _contains_placeholder(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_placeholder(child) for child in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(
            token in lowered
            for token in ("placeholder", "todo", "tbd", "fill_me", "to_be_frozen")
        )
    return False


def _validate_stage_commitment(
    payload: dict[str, Any],
    *,
    dataset: str,
    stage: int,
    expected_stage0_sha256: str | None = None,
) -> None:
    exact_fields = {
        "protocol",
        "dataset",
        "dataset_version",
        "stage",
        "status",
        "sealed_final36_status",
        "artifacts",
    }
    if stage == 1:
        exact_fields.add("stage0_registry_sha256")
    if set(payload) != exact_fields:
        raise ValueError(f"authorization {dataset} Stage-{stage} fields are not exact")
    if (
        payload["protocol"] != COMMITMENT_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError(f"authorization {dataset} Stage-{stage} protocol mismatch")
    if payload["stage"] != stage or payload["status"] != "committed":
        raise ValueError(f"authorization {dataset} Stage-{stage} is not committed")
    if payload["sealed_final36_status"] != "unopened":
        raise ValueError("authorization commitment must keep sealed-final36 unopened")
    if stage == 1:
        require_sha256(payload["stage0_registry_sha256"], "Stage-1 Stage-0 hash")
        if payload["stage0_registry_sha256"] != expected_stage0_sha256:
            raise ValueError(f"authorization {dataset} Stage-1 does not bind Stage-0 bytes")
    artifacts = payload["artifacts"]
    required_artifacts = set(STAGE_ARTIFACTS[(dataset, stage)])
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        raise ValueError(
            f"authorization {dataset} Stage-{stage} artifact inventory is not exact"
        )
    for name, record in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"authorization {dataset} Stage-{stage} artifact name is invalid")
        if not isinstance(record, dict) or set(record) != {
            "sha256",
            "size_bytes",
            "row_count",
        }:
            raise ValueError(
                f"authorization {dataset} Stage-{stage}/{name} commitment is not exact"
            )
        require_sha256(record["sha256"], f"authorization {dataset} Stage-{stage}/{name}")
        if not isinstance(record["size_bytes"], int) or record["size_bytes"] <= 0:
            raise ValueError(f"authorization {dataset} Stage-{stage}/{name} size is invalid")
        if record["row_count"] is not None and (
            not isinstance(record["row_count"], int) or record["row_count"] < 0
        ):
            raise ValueError(
                f"authorization {dataset} Stage-{stage}/{name} row count is invalid"
            )
        expected_rows = EXPECTED_COMMITMENT_ROW_COUNTS.get((dataset, stage, name))
        if expected_rows is not None and record["row_count"] != expected_rows:
            raise ValueError(
                f"authorization {dataset} Stage-{stage}/{name} row count must be "
                f"{expected_rows}"
            )
    if _contains_placeholder(payload):
        raise ValueError(f"authorization {dataset} Stage-{stage} contains a placeholder")


def validate_training_code_registry(
    project_root: Path,
    path: Path,
    expected_sha256: str,
    *,
    expected_runtime_registry_sha256: str,
    verify_current_runtime: bool = True,
) -> dict[str, Any]:
    require_sha256(expected_sha256, "training code registry hash")
    payload = _load_public_json(path, "v4 training code registry")
    if file_sha256(path) != expected_sha256:
        raise ValueError("v4 training code registry byte hash mismatch")
    if _contains_placeholder(payload):
        raise ValueError("v4 training code registry contains a placeholder")
    if set(payload) != {"protocol", "status", "runtime_registry", "artifacts"}:
        raise ValueError("training code registry fields are not exact")
    if (
        payload["protocol"] != TRAINING_CODE_REGISTRY_PROTOCOL
        or payload["status"] != "frozen"
    ):
        raise ValueError("training code registry is not frozen")
    runtime_record = payload["runtime_registry"]
    if runtime_record != {
        "path": str(EXPECTED_RUNTIME_REGISTRY),
        "sha256": expected_runtime_registry_sha256,
    }:
        raise ValueError("training code registry runtime reference is not exact")
    validate_runtime_registry(
        project_root / EXPECTED_RUNTIME_REGISTRY,
        expected_runtime_registry_sha256,
        project_root=project_root,
        verify_current_runtime=verify_current_runtime,
    )
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(CODE_ARTIFACT_PATHS):
        raise ValueError("training code registry artifact inventory is not exact")
    for name, expected_path in CODE_ARTIFACT_PATHS.items():
        record = artifacts[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"training code registry/{name} record is not exact")
        if record["path"] != str(expected_path):
            raise ValueError(f"training code registry/{name} path differs from protocol")
        require_sha256(record["sha256"], f"training code registry/{name}")
        artifact = project_root / expected_path
        if artifact.is_symlink() or not artifact.is_file():
            raise FileNotFoundError(
                f"training code registry/{name} artifact is missing or a symlink"
            )
        if file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"training code registry/{name} byte hash mismatch")
    return payload


def validate_training_authorization(
    path: Path,
    expected_sha256: str,
    *,
    project_root: Path = Path("."),
    verify_current_runtime: bool = True,
) -> dict[str, Any]:
    require_sha256(expected_sha256, "training authorization hash")
    payload = _load_public_json(path, "v4 training authorization")
    if file_sha256(path) != expected_sha256:
        raise ValueError("v4 training authorization byte hash mismatch")
    if payload.get("protocol") != AUTHORIZATION_PROTOCOL or payload.get("status") != "authorized":
        raise ValueError("v4 training authorization is not authorized")
    if payload.get("dataset_version") != DATASET_VERSION:
        raise ValueError("training authorization dataset version differs from protocol")
    if payload.get("sealed_final36_status") != "unopened":
        raise ValueError("training authorization must attest sealed-final36 is unopened")
    exact_fields = {
        "protocol",
        "status",
        "dataset_version",
        "sealed_final36_status",
        *AUTHORIZATION_REF_PATHS,
    }
    if set(payload) != exact_fields:
        raise ValueError(
            f"training authorization must contain exact fields {sorted(exact_fields)}"
        )
    resolved: dict[str, Path] = {}
    for name, expected_path in AUTHORIZATION_REF_PATHS.items():
        record = payload[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"authorization ref {name} must contain only path and sha256")
        if record["path"] != expected_path:
            raise ValueError(f"authorization ref {name} path differs from protocol")
        require_sha256(record["sha256"], f"authorization ref {name}")
        ref_path = project_root / expected_path
        if ref_path.is_symlink() or not ref_path.is_file():
            raise FileNotFoundError(
                f"authorization ref {name} is missing or a symlink: {ref_path}"
            )
        if file_sha256(ref_path) != record["sha256"]:
            raise ValueError(f"authorization ref {name} byte hash mismatch")
        ref_payload = _load_public_json(ref_path, f"authorization ref {name}")
        if _contains_placeholder(ref_payload):
            raise ValueError(f"authorization ref {name} contains a placeholder")
        resolved[name] = ref_path

    for dataset in ("causal", "specificity"):
        stage0_name = f"{dataset}_stage0"
        stage1_name = f"{dataset}_stage1"
        stage0 = _load_public_json(resolved[stage0_name], f"authorization ref {stage0_name}")
        stage1 = _load_public_json(resolved[stage1_name], f"authorization ref {stage1_name}")
        _validate_stage_commitment(stage0, dataset=dataset, stage=0)
        _validate_stage_commitment(
            stage1,
            dataset=dataset,
            stage=1,
            expected_stage0_sha256=payload[stage0_name]["sha256"],
        )
    bank, _ = validate_public_bank_registry(
        resolved["source_bank_registry"],
        expected_sha256=payload["source_bank_registry"]["sha256"],
    )
    holdout, _ = validate_public_holdout_commitment(
        resolved["holdout_public_commitment"],
        expected_sha256=payload["holdout_public_commitment"]["sha256"],
        bank_registry=bank,
    )
    causal_stage0 = _load_public_json(
        resolved["causal_stage0"], "authorization ref causal_stage0"
    )
    if causal_stage0["artifacts"]["source_bank_registry_64"]["sha256"] != payload[
        "source_bank_registry"
    ]["sha256"]:
        raise ValueError("causal Stage-0 source bank differs from training authorization")
    if causal_stage0["artifacts"]["holdout_registry_24"]["sha256"] != holdout[
        "holdout_registry_file_sha256"
    ]:
        raise ValueError("causal Stage-0 holdout registry differs from public commitment")
    gate = _load_public_json(resolved["gate_registry"], "authorization gate registry")
    if set(gate) != {
        "protocol",
        "status",
        "dataset_version",
        "sealed_final36_status",
        "gate_spec",
        "gate_spec_sha256",
        "scorer_sha256",
    }:
        raise ValueError("authorization gate registry fields are not exact")
    if (
        gate["protocol"] != GATE_REGISTRY_PROTOCOL
        or gate["status"] != "frozen"
        or gate["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError("authorization gate registry is not frozen")
    if gate["sealed_final36_status"] != "unopened":
        raise ValueError("authorization gate registry must keep sealed-final36 unopened")
    if not isinstance(gate["gate_spec"], dict) or not gate["gate_spec"]:
        raise ValueError("authorization gate registry has no machine gate spec")
    if gate["gate_spec_sha256"] != canonical_json_sha256(gate["gate_spec"]):
        raise ValueError("authorization machine gate spec hash mismatch")
    if (
        gate["gate_spec"] != EXPECTED_GATE_SPEC
        or gate["gate_spec_sha256"] != EXPECTED_GATE_SPEC_SHA256
    ):
        raise ValueError("authorization machine gate spec differs from canonical protocol")
    require_sha256(gate["scorer_sha256"], "authorization machine gate scorer")

    validate_runtime_registry(
        resolved["runtime_registry"],
        payload["runtime_registry"]["sha256"],
        project_root=project_root,
        verify_current_runtime=verify_current_runtime,
    )
    code_registry = validate_training_code_registry(
        project_root,
        resolved["code_registry"],
        payload["code_registry"]["sha256"],
        expected_runtime_registry_sha256=payload["runtime_registry"]["sha256"],
        verify_current_runtime=verify_current_runtime,
    )
    if gate["scorer_sha256"] != code_registry["artifacts"]["eval_scorer"]["sha256"]:
        raise ValueError("machine gate scorer differs from frozen code registry")
    if _contains_placeholder(payload):
        raise ValueError("training authorization contains a placeholder")
    return payload


def validate_preflight(
    args: argparse.Namespace,
    mapping: dict[str, Any],
    *,
    sidecar_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
) -> dict[str, Any]:
    if args.preflight_artifact.is_symlink() or not args.preflight_artifact.is_file():
        raise FileNotFoundError("null-sidecar preflight artifact is missing or symlinked")
    if file_sha256(args.preflight_artifact) != args.preflight_artifact_sha256:
        raise ValueError("null-sidecar preflight artifact byte hash mismatch")
    payload = json.loads(args.preflight_artifact.read_text(encoding="utf-8"))
    exact_fields = {
        "protocol",
        "status",
        "dataset_version",
        "train_manifest_sha256",
        "source_bank_registry_sha256",
        "source_bank_registry_path",
        "holdout_public_commitment_sha256",
        "holdout_public_commitment_path",
        "holdout_count",
        "causal_stage0_public_commitment_path",
        "causal_stage0_public_commitment_sha256",
        "source_mapping_registry_sha256",
        "source_mapping_registry_path",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "prompt_sidecar_inventory_sha256",
        "prompt_sidecar_manifest_sha256",
        "preparer_sha256",
        "transformer_inventory_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "model_content_inventory_sha256",
        "model_artifact_inventory",
        "model_revision",
        "seed",
        "sample_order_sha256",
        "noise_sigma_rng_initial_sha256",
        "noise_sigma_rng_final_sha256",
        "initial_lora_sha256",
        "original_reencode_count",
        "original_reencode_binding_sha256",
        "unique_augmented_reencode_count",
        "augmented_reencode_row_count",
        "augmented_reencode_binding_sha256",
        "augmented_reencode_all_rows_byte_equal",
        "tokenizer_binding",
        "integration_manifest_index",
        "integration_scene_id",
        "v3b_reference_path",
        "v4_null_sidecar_path",
        "null_sidecar_substitution",
        "forward_loss_gradient_equal",
        "rng_restored_between_signatures",
        "trainable_state_restored_between_signatures",
        "v3b_reference_signature",
        "v4_null_sidecar_signature",
        "optimizer_created",
    }
    if not isinstance(payload, dict) or set(payload) != exact_fields:
        raise ValueError("null-sidecar preflight fields are not exact")
    expected = {
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "passed",
        "dataset_version": DATASET_VERSION,
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "source_bank_registry_path": str(EXPECTED_BANK_REGISTRY),
        "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
        "holdout_public_commitment_path": str(EXPECTED_HOLDOUT_PUBLIC_COMMITMENT),
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": str(
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT
        ),
        "causal_stage0_public_commitment_sha256": (
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "source_mapping_registry_path": str(EXPECTED_MAPPING_REGISTRY),
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "canonical_prompt_builder_path": mapping["canonical_prompt_builder_path"],
        "canonical_prompt_builder_sha256": mapping[
            "canonical_prompt_builder_sha256"
        ],
        "base_cache_inventory_sha256": EXPECTED_BASE_CACHE_SHA256,
        "teacher_cache_inventory_sha256": EXPECTED_TEACHER_CACHE_SHA256,
        "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
        "prompt_sidecar_manifest_sha256": args.prompt_sidecar_manifest_sha256,
        "preparer_sha256": file_sha256(CODE_ARTIFACT_PATHS["preparer"]),
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "runtime_registry_path": str(EXPECTED_RUNTIME_REGISTRY),
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "seed": EXPECTED_SEED,
        "sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
        "noise_sigma_rng_initial_sha256": EXPECTED_NOISE_RNG_INITIAL_SHA256,
        "noise_sigma_rng_final_sha256": EXPECTED_NOISE_RNG_FINAL_SHA256,
        "initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
        "original_reencode_count": 178,
        "unique_augmented_reencode_count": sidecar_manifest[
            "unique_augmented_prompt_count"
        ],
        "augmented_reencode_row_count": 178,
        "augmented_reencode_all_rows_byte_equal": True,
        "tokenizer_binding": tokenizer_inventory_binding(model_provenance),
        "forward_loss_gradient_equal": True,
        "rng_restored_between_signatures": True,
        "trainable_state_restored_between_signatures": True,
        "optimizer_created": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"null-sidecar preflight {key} mismatch")
    signature_fields = {
        "prediction_sha256",
        "teacher_prediction_sha256",
        "flow_loss_sha256",
        "teacher_loss_sha256",
        "combined_loss_sha256",
        "gradient_state_sha256",
    }
    v3b_reference_signature = payload.get("v3b_reference_signature")
    v4_null_sidecar_signature = payload.get("v4_null_sidecar_signature")
    if (
        not isinstance(v3b_reference_signature, dict)
        or set(v3b_reference_signature) != signature_fields
        or any(
            not isinstance(value, str)
            for value in v3b_reference_signature.values()
        )
    ):
        raise ValueError("null-sidecar preflight v3b reference signature is not exact")
    for name, value in v3b_reference_signature.items():
        require_sha256(value, f"null-sidecar preflight v3b signature/{name}")
    if v3b_reference_signature != v4_null_sidecar_signature:
        raise ValueError("preflight v3b/v4-null integration signatures are not identical")
    require_sha256(
        payload.get("original_reencode_binding_sha256", ""),
        "null-sidecar original re-encode binding",
    )
    require_sha256(
        payload.get("augmented_reencode_binding_sha256", ""),
        "null-sidecar augmented re-encode binding",
    )
    if payload["original_reencode_binding_sha256"] != sidecar_manifest.get(
        "original_reencode_binding_sha256"
    ):
        raise ValueError(
            "null-sidecar preflight original re-encode binding differs from sidecar manifest"
        )
    for field in (
        "model_artifact_inventory",
        "model_revision",
    ):
        if payload[field] != model_provenance[field]:
            raise ValueError(f"null-sidecar preflight {field} differs from current model")
        if sidecar_manifest.get(field) != model_provenance[field]:
            raise ValueError(f"prompt sidecar {field} differs from current model")
    first_active = next(
        record
        for record in mapping["mapping"]
        if record["active_erase_ordinal"] == 0
    )
    if payload.get("integration_manifest_index") != first_active["manifest_index"]:
        raise ValueError("null-sidecar preflight integration manifest index mismatch")
    if payload.get("integration_scene_id") != first_active["scene_id"]:
        raise ValueError("null-sidecar preflight integration scene mismatch")
    if payload.get("v3b_reference_path") != "frozen_base_cache_prompt_embeds":
        raise ValueError("null-sidecar preflight v3b reference path mismatch")
    if payload.get("v4_null_sidecar_path") != (
        "v4_sidecar_loader_with_fresh_original_augmented_prompt_embeds"
    ):
        raise ValueError("null-sidecar preflight v4 null-sidecar path mismatch")
    if payload.get("null_sidecar_substitution") != (
        "fresh_original_embedding_for_augmented_embedding"
    ):
        raise ValueError("null-sidecar preflight substitution mismatch")
    if _contains_placeholder(payload):
        raise ValueError("null-sidecar preflight contains a placeholder")
    return payload


def validate_run_registration(
    args: argparse.Namespace,
    mapping: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if args.run_registration.is_symlink() or not args.run_registration.is_file():
        raise FileNotFoundError("run registration is missing or symlinked")
    actual_hash = file_sha256(args.run_registration)
    if actual_hash != args.run_registration_sha256:
        raise ValueError(f"run registration byte hash mismatch: {actual_hash}")
    registration = json.loads(args.run_registration.read_text(encoding="utf-8"))
    expected = {
        "protocol": PROTOCOL,
        "status": "registered",
        "dataset_version": DATASET_VERSION,
        "output_dir": str(EXPECTED_OUTPUT_DIR),
        "train_manifest_path": str(EXPECTED_MANIFEST),
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "base_cache_path": str(EXPECTED_BASE_CACHE_DIR),
        "base_cache_inventory_sha256": EXPECTED_BASE_CACHE_SHA256,
        "teacher_cache_path": str(EXPECTED_TEACHER_CACHE_DIR),
        "teacher_cache_inventory_sha256": EXPECTED_TEACHER_CACHE_SHA256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "source_bank_registry_path": str(EXPECTED_BANK_REGISTRY),
        "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
        "holdout_public_commitment_path": str(EXPECTED_HOLDOUT_PUBLIC_COMMITMENT),
        "holdout_count": 24,
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "source_mapping_registry_path": str(EXPECTED_MAPPING_REGISTRY),
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "canonical_prompt_builder_path": mapping["canonical_prompt_builder_path"],
        "canonical_prompt_builder_sha256": mapping[
            "canonical_prompt_builder_sha256"
        ],
        "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
        "prompt_sidecar_manifest_sha256": args.prompt_sidecar_manifest_sha256,
        "prompt_sidecar_path": str(EXPECTED_PROMPT_SIDECAR_DIR),
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "runtime_registry_path": str(EXPECTED_RUNTIME_REGISTRY),
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "preflight_artifact_path": str(EXPECTED_PREFLIGHT),
        "preflight_artifact_sha256": args.preflight_artifact_sha256,
        "training_authorization_path": str(EXPECTED_AUTHORIZATION),
        "training_authorization_sha256": args.training_authorization_sha256,
        "training_code_registry_path": str(EXPECTED_TRAINING_CODE_REGISTRY),
        "training_code_registry_sha256": args.training_code_registry_sha256,
        "authorization_source": "independent_audited_committed_and_pushed",
        "expected_initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
        "expected_sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
        "expected_noise_sigma_rng_initial_sha256": EXPECTED_NOISE_RNG_INITIAL_SHA256,
        "expected_noise_sigma_rng_final_sha256": EXPECTED_NOISE_RNG_FINAL_SHA256,
        "training_config": EXPECTED_CONFIG,
    }
    exact_fields = {
        *expected,
        "created_utc",
        "git_commit",
        "git_upstream",
        "only_training_intervention",
        *(
            field
            for name in CODE_ARTIFACT_PATHS
            for field in (f"{name}_path", f"{name}_sha256")
        ),
    }
    if set(registration) != exact_fields:
        raise ValueError("run registration fields are not exact")
    if not isinstance(registration["created_utc"], str) or not registration["created_utc"]:
        raise ValueError("run registration created_utc is missing")
    git_commit = registration["git_commit"]
    git_upstream = registration["git_upstream"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in git_commit)
        or not isinstance(git_upstream, str)
        or not git_upstream
    ):
        raise ValueError("run registration git provenance is invalid")
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        current_upstream = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed_authorization = subprocess.check_output(
            ["git", "show", f"{git_commit}:{EXPECTED_AUTHORIZATION.as_posix()}"]
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", git_commit, git_upstream],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError("run registration git provenance cannot be verified") from exc
    if current_commit != git_commit or current_upstream != git_upstream:
        raise ValueError("run registration differs from current commit/upstream")
    if hashlib.sha256(committed_authorization).hexdigest() != args.training_authorization_sha256:
        raise ValueError("committed training authorization differs from audited hash")
    for key, value in expected.items():
        if registration.get(key) != value:
            raise ValueError(f"run registration {key} mismatch")
    if registration.get("only_training_intervention") != (
        "erase factual prompt_embeds replaced by registered augmented source-slot sidecar"
    ):
        raise ValueError("run registration does not bind the sole v4 intervention")
    code_registry = validate_training_code_registry(
        Path("."),
        args.training_code_registry,
        args.training_code_registry_sha256,
        expected_runtime_registry_sha256=args.runtime_registry_sha256,
        verify_current_runtime=True,
    )
    for name, expected_path in CODE_ARTIFACT_PATHS.items():
        if registration.get(f"{name}_path") != str(expected_path):
            raise ValueError(f"run registration {name}_path mismatch")
        if registration.get(f"{name}_sha256") != code_registry["artifacts"][name]["sha256"]:
            raise ValueError(f"run registration {name} code hash mismatch")
    return registration, actual_hash


def validate_base_payloads(paths: list[Path], rows: list[dict[str, str]]) -> None:
    if len(paths) != len(rows):
        raise ValueError("base cache path count differs from manifest row count")
    for index, (path, row) in enumerate(zip(paths, rows)):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "scene_id": row["scene_id"],
            "prompt": row["prompt"],
            "training_role": row["training_role"],
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"base cache row {index} has drifted {key}")
        if not isinstance(payload.get("latents"), torch.Tensor):
            raise ValueError(f"base cache row {index} has no latent tensor")


def validate_teacher_payloads(
    paths: list[Path], rows: list[dict[str, str]]
) -> dict[int, Path]:
    erase = [(i, row) for i, row in enumerate(rows) if row["training_role"] == "erase"]
    result: dict[int, Path] = {}
    if len(erase) != len(paths):
        raise ValueError("teacher cache path count differs from erase row count")
    for (index, row), path in zip(erase, paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("scene_id") != row["scene_id"]:
            raise ValueError(f"teacher cache scene mismatch at manifest index {index}")
        if payload.get("target_generation_prompt") != row["target_generation_prompt"]:
            raise ValueError(f"teacher cache prompt mismatch at manifest index {index}")
        embedding = payload.get("teacher_prompt_embeds")
        if not isinstance(embedding, torch.Tensor) or not torch.isfinite(embedding.float()).all():
            raise ValueError(f"teacher cache embedding invalid at manifest index {index}")
        result[index] = path
    return result


def seed_training(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_target_prompt_teacher_pair(
    transformer: torch.nn.Module,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    factual_prompt_embeds: torch.Tensor,
    target_prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the unchanged frozen target-prompt teacher and LoRA student."""

    transformer.disable_adapters()
    try:
        with torch.no_grad():
            teacher_prediction = transformer(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=target_prompt_embeds,
                return_dict=False,
            )[0].detach()
    finally:
        transformer.enable_adapters()
    prediction = transformer(
        hidden_states=noisy,
        timestep=timestep,
        encoder_hidden_states=factual_prompt_embeds,
        return_dict=False,
    )[0]
    return teacher_prediction, prediction


def scale_metrics(raw_ratios: list[float], weight: float) -> dict[str, float]:
    values = np.asarray(raw_ratios, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scale ratios must be finite, nonnegative, and nonempty")
    gradients = weight * np.sqrt(values)
    return {
        "mean_raw_loss_ratio": float(values.mean()),
        "mean_weighted_loss_ratio": float(weight * values.mean()),
        "mean_weighted_output_grad_ratio": float(gradients.mean()),
        "median_weighted_output_grad_ratio": float(np.median(gradients)),
        "max_weighted_output_grad_ratio": float(gradients.max()),
    }


def write_scale_sanity(
    args: argparse.Namespace,
    observations: list[dict[str, Any]],
    *,
    run_registration_sha256: str,
) -> dict[str, Any]:
    if len(observations) != 16:
        raise ValueError("v4 scale sanity requires exactly 16 erase observations")
    metrics = scale_metrics(
        [float(record["raw_loss_ratio"]) for record in observations],
        args.target_prompt_teacher_weight,
    )
    passed = (
        args.target_prompt_sanity_min_output_grad_ratio
        <= metrics["mean_weighted_output_grad_ratio"]
        <= args.target_prompt_sanity_max_output_grad_ratio
        and metrics["max_weighted_output_grad_ratio"]
        <= args.target_prompt_sanity_max_single_output_grad_ratio
    )
    payload = {
        "protocol": SANITY_PROTOCOL,
        "status": "passed" if passed else "registered_scale_sanity_termination",
        "dataset_version": DATASET_VERSION,
        "passed": passed,
        "run_registration_sha256": run_registration_sha256,
        "calibration_id": args.target_prompt_calibration_id,
        "formula": "g_i = 4 * sqrt(target_prompt_teacher_loss / flow_loss)",
        "aggregation": "arithmetic_mean_over_first_16_actual_erase_updates",
        "weight": args.target_prompt_teacher_weight,
        "mean_min": args.target_prompt_sanity_min_output_grad_ratio,
        "mean_max": args.target_prompt_sanity_max_output_grad_ratio,
        "single_max": args.target_prompt_sanity_max_single_output_grad_ratio,
        "observation_count": len(observations),
        **metrics,
        "observations": observations,
    }
    path = args.output_dir / "target_prompt_scale_sanity_v2.json"
    atomic_write_new_json(path, payload)
    return payload


def atomic_write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_save_lora(
    transformer: WanTransformer3DModel,
    checkpoint: Path,
    metadata: dict[str, Any],
    *,
    validated_lora_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        WanPipeline.save_lora_weights(
            str(temporary),
            transformer_lora_layers=convert_state_dict_to_diffusers(
                validated_lora_state
            ),
            safe_serialization=True,
        )
        training_state = temporary / "training_state_v2.json"
        with training_state.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        weights = temporary / "pytorch_lora_weights.safetensors"
        files = {path.name: path for path in temporary.iterdir()}
        if set(files) != {"pytorch_lora_weights.safetensors", "training_state_v2.json"}:
            raise ValueError("checkpoint inventory must contain exactly weights and training state")
        if any(path.is_symlink() or not path.is_file() for path in files.values()):
            raise ValueError("checkpoint inventory contains a symlink or non-file")
        file_records = [
            {"name": weights.name, "sha256": file_sha256(weights)},
            {"name": training_state.name, "sha256": file_sha256(training_state)},
        ]
        os.replace(temporary, checkpoint)
    except BaseException:
        # Retain a failed temporary directory as forensic evidence.  The
        # registered formal run is terminal and cannot be resumed.
        raise
    return {
        "path": str(checkpoint),
        "weights_sha256": file_records[0]["sha256"],
        "training_state_sha256": file_records[1]["sha256"],
    }


def clip_and_validate_grad_norm(
    trainable: list[torch.nn.Parameter], *, max_norm: float, step: int
) -> float:
    """Clip exactly as v3b and fail before optimizer.step on NaN/Inf."""

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    grad_norm_value = float(grad_norm.detach())
    if not math.isfinite(grad_norm_value):
        raise FloatingPointError(f"non-finite clipped gradient norm at step {step}")
    return grad_norm_value


def _named_tensor_state_sha256(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tensor_inventory(
    named_tensors: list[tuple[str, torch.Tensor]], *, label: str
) -> dict[str, int]:
    if not named_tensors:
        raise ValueError(f"final {label} tensor inventory is empty")
    tensor_count = len(named_tensors)
    element_count = 0
    nonfinite_tensor_count = 0
    nonfinite_element_count = 0
    seen: set[str] = set()
    for name, tensor in named_tensors:
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"final {label} tensor name is empty or duplicated")
        seen.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"final {label}/{name} is not a tensor")
        elements = int(tensor.numel())
        if elements <= 0:
            raise ValueError(f"final {label}/{name} tensor is empty")
        finite_elements = int(torch.isfinite(tensor.detach()).sum().item())
        nonfinite = elements - finite_elements
        element_count += elements
        nonfinite_element_count += nonfinite
        nonfinite_tensor_count += int(nonfinite > 0)
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "nonfinite_element_count": nonfinite_element_count,
    }


def validate_final_lora_finiteness(
    transformer: WanTransformer3DModel,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the exact state to save only after all final LoRA values are finite."""

    trainable_named = [
        (name, parameter)
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    ]
    lora_state = get_peft_model_state_dict(transformer)
    if not isinstance(lora_state, dict):
        raise TypeError("final LoRA state must be a tensor dictionary")
    lora_named = list(lora_state.items())
    trainable_inventory = _finite_tensor_inventory(
        trainable_named, label="trainable parameter"
    )
    lora_inventory = _finite_tensor_inventory(lora_named, label="LoRA state")
    if (
        trainable_inventory["nonfinite_tensor_count"]
        or trainable_inventory["nonfinite_element_count"]
        or lora_inventory["nonfinite_tensor_count"]
        or lora_inventory["nonfinite_element_count"]
    ):
        raise FloatingPointError("final LoRA/trainable tensor inventory is non-finite")
    evidence = {
        "protocol": FINAL_LORA_FINITE_PROTOCOL,
        "status": "passed",
        "digest_algorithm": "sha256_sorted_name_shape_dtype_raw_bytes_v1",
        "trainable_parameter_count": trainable_inventory["tensor_count"],
        "trainable_element_count": trainable_inventory["element_count"],
        "nonfinite_trainable_parameter_count": trainable_inventory[
            "nonfinite_tensor_count"
        ],
        "nonfinite_trainable_element_count": trainable_inventory[
            "nonfinite_element_count"
        ],
        "lora_state_tensor_count": lora_inventory["tensor_count"],
        "lora_state_element_count": lora_inventory["element_count"],
        "nonfinite_lora_state_tensor_count": lora_inventory[
            "nonfinite_tensor_count"
        ],
        "nonfinite_lora_state_element_count": lora_inventory[
            "nonfinite_element_count"
        ],
        "trainable_state_sha256": _named_tensor_state_sha256(trainable_named),
        "lora_state_sha256": _named_tensor_state_sha256(lora_named),
    }
    return lora_state, evidence


def _revalidate_frozen_inputs(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    mapping_by_index: dict[int, dict[str, Any]],
) -> None:
    validate_canonical_public_hashes(args)
    validate_cache_inventory(
        rows, args.cache_dir, expected_sha256=EXPECTED_BASE_CACHE_SHA256
    )
    validate_cache_inventory(
        rows,
        args.target_prompt_cache_dir,
        expected_sha256=EXPECTED_TEACHER_CACHE_SHA256,
        role="erase",
    )
    model_provenance = validate_model_content_inventory(
        args.model, args.model_content_inventory_sha256
    )
    validate_prompt_sidecar(
        rows,
        mapping_by_index,
        args.prompt_sidecar_dir,
        expected_inventory_sha256=args.prompt_sidecar_inventory_sha256,
        expected_manifest_sha256=args.prompt_sidecar_manifest_sha256,
        expected_mapping_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        expected_causal_stage0_public_commitment_sha256=(
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        expected_model_inventory_sha256=args.model_content_inventory_sha256,
        expected_runtime_registry_sha256=args.runtime_registry_sha256,
        expected_model_provenance=model_provenance,
    )
    authorization = validate_training_authorization(
        args.training_authorization,
        args.training_authorization_sha256,
        project_root=Path("."),
    )
    if authorization["code_registry"]["sha256"] != args.training_code_registry_sha256:
        raise ValueError("training authorization/code-registry hash changed during training")
    if authorization["runtime_registry"]["sha256"] != args.runtime_registry_sha256:
        raise ValueError("training authorization/runtime-registry hash changed during training")
    for path, expected in (
        (args.source_bank_registry, args.source_bank_registry_sha256),
        (args.holdout_public_commitment, args.holdout_public_commitment_sha256),
        (
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT,
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256,
        ),
        (args.source_mapping_registry, args.source_mapping_registry_sha256),
        (args.preflight_artifact, args.preflight_artifact_sha256),
        (args.training_code_registry, args.training_code_registry_sha256),
        (args.runtime_registry, args.runtime_registry_sha256),
        (args.run_registration, args.run_registration_sha256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"frozen artifact changed during training: {path}")


def train(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    base_paths: list[Path],
    teacher_paths: dict[int, Path],
    sidecar_paths: dict[int, Path],
    mapping: dict[str, Any],
    mapping_by_index: dict[int, dict[str, Any]],
    run_registration_sha256: str,
) -> None:
    seed_training(args.seed)
    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    seed_training(args.seed)
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    initial_lora_sha256 = trainable_state_sha256(transformer)
    if initial_lora_sha256 != EXPECTED_INITIAL_LORA_SHA256:
        raise ValueError(f"initial LoRA digest differs from v3b: {initial_lora_sha256}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rng_initial = tensor_sha256(generator.get_state())
    if rng_initial != EXPECTED_NOISE_RNG_INITIAL_SHA256:
        raise ValueError(f"initial noise/sigma RNG digest drift: {rng_initial}")
    schedule = balanced_v3b_schedule(rows, seed=args.seed, steps=args.max_steps)
    if sample_order_sha256(rows, schedule) != EXPECTED_SAMPLE_ORDER_SHA256:
        raise ValueError("sample order differs from frozen v3b schedule")

    role_counts = Counter()
    active_source_counts = Counter()
    order_digest = hashlib.sha256()
    losses: list[float] = []
    teacher_losses: list[float] = []
    preserve_losses: list[float] = []
    observations: list[dict[str, Any]] = []
    sanity: dict[str, Any] | None = None
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    transformer.train()

    for step, manifest_index in enumerate(schedule, start=1):
        row = rows[manifest_index]
        sample = torch.load(base_paths[manifest_index], map_location="cpu", weights_only=True)
        is_preserve = row["training_role"] == "preserve"
        role_counts[row["training_role"]] += 1
        order_digest.update(
            f"{step}:{row['training_role']}:{row['scene_id']}\n".encode("utf-8")
        )
        clean = sample["latents"].to(device=device, dtype=torch.bfloat16)
        base_prompt_embeds = sample["prompt_embeds"].to(
            device=device, dtype=torch.bfloat16
        )
        assigned_source_id: str | None = None
        if is_preserve:
            prompt_embeds = base_prompt_embeds
            target_prompt_embeds = None
        else:
            mapping_record = mapping_by_index[manifest_index]
            expected_active_ordinal = role_counts["erase"] - 1
            if mapping_record.get("active_erase_ordinal") != expected_active_ordinal:
                raise ValueError(
                    f"active source mapping drift at erase ordinal {expected_active_ordinal}"
                )
            sidecar = torch.load(
                sidecar_paths[manifest_index], map_location="cpu", weights_only=True
            )
            if sidecar["scene_id"] != sample["scene_id"]:
                raise ValueError(f"source sidecar scene mismatch at {manifest_index}")
            prompt_embeds = sidecar["augmented_prompt_embeds"].to(
                device=device, dtype=torch.bfloat16
            )
            assigned_source_id = sidecar["assigned_source_id"]
            if assigned_source_id != mapping_record["assigned_source_id"]:
                raise ValueError(f"source sidecar assignment mismatch at {manifest_index}")
            active_source_counts[assigned_source_id] += 1
            teacher_payload = torch.load(
                teacher_paths[manifest_index], map_location="cpu", weights_only=True
            )
            target_prompt_embeds = teacher_payload["teacher_prompt_embeds"].to(
                device=device, dtype=torch.bfloat16
            )
        if prompt_embeds.shape != base_prompt_embeds.shape:
            raise ValueError(f"v4 prompt shape drift at {manifest_index}")

        noise = torch.randn(
            clean.shape, generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        sigma = torch.rand(
            (clean.shape[0],), generator=generator, dtype=torch.float32
        ).to(device)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        noisy = ((1.0 - sigma) * clean + sigma * noise).to(dtype=torch.bfloat16)
        target = (noise - clean).to(dtype=torch.bfloat16)
        timestep = (sigma.flatten() * 1000.0).to(dtype=torch.bfloat16)

        if is_preserve:
            transformer.disable_adapters()
            try:
                with torch.no_grad():
                    teacher_prediction = transformer(
                        hidden_states=noisy,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0].detach()
            finally:
                transformer.enable_adapters()
            prediction = transformer(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            preserve_loss = torch.nn.functional.mse_loss(
                prediction.float(),
                teacher_prediction.float(),
                reduction="none",
            ).mean()
            flow_loss = torch.zeros((), device=device)
            target_teacher_loss = torch.zeros((), device=device)
            combined_loss = args.preserve_weight * preserve_loss
        else:
            teacher_prediction, prediction = forward_target_prompt_teacher_pair(
                transformer,
                noisy,
                timestep,
                prompt_embeds,
                target_prompt_embeds,
            )
            flow_loss = torch.nn.functional.mse_loss(
                prediction.float(), target.float(), reduction="none"
            ).mean()
            target_teacher_loss = torch.nn.functional.mse_loss(
                prediction.float(), teacher_prediction.float(), reduction="none"
            ).mean()
            preserve_loss = torch.zeros((), device=device)
            combined_loss = flow_loss + args.target_prompt_teacher_weight * target_teacher_loss
            flow_value = float(flow_loss.detach())
            teacher_value = float(target_teacher_loss.detach())
            if (
                not np.isfinite(flow_value)
                or not np.isfinite(teacher_value)
                or flow_value <= 0
                or teacher_value < 0
            ):
                raise FloatingPointError(
                    f"invalid v4 scale losses at step {step}: "
                    f"flow={flow_value} teacher={teacher_value}"
                )
            raw_ratio = teacher_value / flow_value
            if len(observations) < 16:
                observations.append(
                    {
                        "global_step": step,
                        "erase_ordinal": role_counts["erase"] - 1,
                        "manifest_index": manifest_index,
                        "scene_id": row["scene_id"],
                        "assigned_source_id": assigned_source_id,
                        "flow_loss": flow_value,
                        "target_prompt_teacher_loss": teacher_value,
                        "raw_loss_ratio": raw_ratio,
                        "weighted_output_gradient_norm_ratio": (
                            args.target_prompt_teacher_weight * float(np.sqrt(raw_ratio))
                        ),
                    }
                )
                if len(observations) == 16:
                    sanity = write_scale_sanity(
                        args,
                        observations,
                        run_registration_sha256=run_registration_sha256,
                    )
                    if not sanity["passed"]:
                        raise RuntimeError(
                            "registered_scale_sanity_termination: v4 first-16 erase gate failed"
                        )
        if not torch.isfinite(combined_loss):
            raise FloatingPointError(f"non-finite v4 loss at step {step}")
        # The 16th erase sanity artifact is durable before this backward.
        loss = combined_loss / args.grad_accum
        loss.backward()
        grad_norm_value = 0.0
        if step % args.grad_accum == 0:
            grad_norm_value = clip_and_validate_grad_norm(
                trainable, max_norm=1.0, step=step
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_value = float(combined_loss.detach())
        losses.append(loss_value)
        if not is_preserve:
            teacher_losses.append(float(target_teacher_loss.detach()))
        preserve_losses.append(float(preserve_loss.detach()))
        print(
            f"step={step}/{args.max_steps} scene={row['scene_id']} "
            f"role={row['training_role']} assigned_source={assigned_source_id} "
            f"loss={loss_value:.6f} flow={float(flow_loss.detach()):.6f} "
            f"target_teacher={float(target_teacher_loss.detach()):.6f} "
            f"preserve={float(preserve_loss.detach()):.6f} grad_norm={grad_norm_value:.6f} "
            f"mean20={np.mean(losses[-20:]):.6f} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
        del (
            sample,
            clean,
            base_prompt_embeds,
            prompt_embeds,
            noise,
            sigma,
            noisy,
            target,
            timestep,
            teacher_prediction,
            prediction,
            flow_loss,
            target_teacher_loss,
            preserve_loss,
            combined_loss,
            loss,
        )
        if target_prompt_embeds is not None:
            del target_prompt_embeds, teacher_payload, sidecar

    if sanity is None or not sanity.get("passed"):
        raise RuntimeError("v4 scale sanity did not complete; checkpoint forbidden")
    if role_counts != Counter({"erase": 100, "preserve": 100}):
        raise ValueError(f"v4 role count drift: {dict(role_counts)}")
    if dict(sorted(active_source_counts.items())) != mapping["active_source_counts"]:
        raise ValueError("v4 realized active source counts differ from frozen mapping")
    final_order_hash = order_digest.hexdigest()
    final_rng_hash = tensor_sha256(generator.get_state())
    if final_order_hash != EXPECTED_SAMPLE_ORDER_SHA256:
        raise ValueError(f"v4 final sample-order digest drift: {final_order_hash}")
    if final_rng_hash != EXPECTED_NOISE_RNG_FINAL_SHA256:
        raise ValueError(f"v4 final noise/sigma RNG digest drift: {final_rng_hash}")
    _revalidate_frozen_inputs(args, rows, mapping_by_index)
    validated_lora_state, final_lora_finite_check = validate_final_lora_finiteness(
        transformer
    )
    sanity_path = args.output_dir / "target_prompt_scale_sanity_v2.json"
    checkpoint_path = args.output_dir / "checkpoint-000200"
    training_state = {
        "protocol": PROTOCOL,
        "status": "eligible_training_complete",
        "dataset_version": DATASET_VERSION,
        "step": 200,
        "max_steps": 200,
        "only_training_intervention": (
            "erase factual prompt_embeds replaced by registered augmented source-slot sidecar"
        ),
        "training_config": EXPECTED_CONFIG,
        "manifest": str(args.manifest),
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "base_cache_inventory_sha256": EXPECTED_BASE_CACHE_SHA256,
        "teacher_cache_inventory_sha256": EXPECTED_TEACHER_CACHE_SHA256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
        "holdout_public_commitment_path": str(args.holdout_public_commitment),
        "holdout_count": 24,
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "canonical_prompt_builder_path": mapping["canonical_prompt_builder_path"],
        "canonical_prompt_builder_sha256": mapping[
            "canonical_prompt_builder_sha256"
        ],
        "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
        "prompt_sidecar_manifest_sha256": args.prompt_sidecar_manifest_sha256,
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "runtime_registry_path": str(args.runtime_registry),
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "preflight_artifact_sha256": args.preflight_artifact_sha256,
        "training_authorization_path": str(args.training_authorization),
        "training_authorization_sha256": args.training_authorization_sha256,
        "training_code_registry_path": str(args.training_code_registry),
        "training_code_registry_sha256": args.training_code_registry_sha256,
        "run_registration_sha256": run_registration_sha256,
        "scale_sanity_sha256": file_sha256(sanity_path),
        "initial_lora_sha256": initial_lora_sha256,
        "sample_order_sha256": final_order_hash,
        "noise_sigma_rng_initial_sha256": rng_initial,
        "noise_sigma_rng_final_sha256": final_rng_hash,
        "role_step_counts": dict(role_counts),
        "active_source_counts": dict(sorted(active_source_counts.items())),
        "mean_loss_last_20": float(np.mean(losses[-20:])),
        "mean_target_prompt_teacher_loss_last_20": float(
            np.mean(teacher_losses[-20:])
        ),
        "mean_preserve_loss_last_20": float(np.mean(preserve_losses[-20:])),
        "trainer_sha256": file_sha256(Path(__file__)),
        "launcher_sha256": file_sha256(CODE_ARTIFACT_PATHS["launcher"]),
        "final_lora_finite_check": final_lora_finite_check,
    }
    checkpoint = atomic_save_lora(
        transformer,
        checkpoint_path,
        training_state,
        validated_lora_state=validated_lora_state,
    )
    eligibility = {
        "protocol": ELIGIBILITY_PROTOCOL,
        "status": "eligible",
        "dataset_version": DATASET_VERSION,
        "step": 200,
        "checkpoint": checkpoint,
        "run_registration": {
            "path": str(args.run_registration),
            "sha256": run_registration_sha256,
        },
        "preflight": {
            "path": str(args.preflight_artifact),
            "sha256": args.preflight_artifact_sha256,
        },
        "scale_sanity": {
            "path": str(sanity_path),
            "sha256": file_sha256(sanity_path),
        },
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "base_cache_inventory_sha256": EXPECTED_BASE_CACHE_SHA256,
        "teacher_cache_inventory_sha256": EXPECTED_TEACHER_CACHE_SHA256,
        "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
        "role_step_counts": dict(role_counts),
        "sample_order_sha256": final_order_hash,
        "noise_sigma_rng_initial_sha256": rng_initial,
        "noise_sigma_rng_final_sha256": final_rng_hash,
        "initial_lora_sha256": initial_lora_sha256,
        "trainer_sha256": file_sha256(Path(__file__)),
        "launcher_sha256": file_sha256(CODE_ARTIFACT_PATHS["launcher"]),
        "training_authorization_sha256": args.training_authorization_sha256,
        "training_code_registry_sha256": args.training_code_registry_sha256,
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "final_lora_finite_check": final_lora_finite_check,
    }
    atomic_write_new_json(args.output_dir / "checkpoint_eligibility_v2.json", eligibility)
    print(f"Wrote eligible v4 checkpoint: {checkpoint_path}", flush=True)


def main() -> int:
    args = parse_args()
    validate_canonical_public_hashes(args)
    validate_output_reservation(args)
    authorization = validate_training_authorization(
        args.training_authorization,
        args.training_authorization_sha256,
        project_root=Path("."),
    )
    if authorization["code_registry"]["path"] != str(EXPECTED_TRAINING_CODE_REGISTRY):
        raise ValueError("training authorization binds a different code-registry path")
    if authorization["code_registry"]["sha256"] != args.training_code_registry_sha256:
        raise ValueError("training authorization binds a different code-registry hash")
    if authorization["runtime_registry"]["path"] != str(EXPECTED_RUNTIME_REGISTRY):
        raise ValueError("training authorization binds a different runtime-registry path")
    if authorization["runtime_registry"]["sha256"] != args.runtime_registry_sha256:
        raise ValueError("training authorization binds a different runtime-registry hash")
    if authorization["source_bank_registry"] != {
        "path": str(args.source_bank_registry),
        "sha256": args.source_bank_registry_sha256,
    }:
        raise ValueError("training authorization binds a different source-bank registry")
    if authorization["holdout_public_commitment"] != {
        "path": str(args.holdout_public_commitment),
        "sha256": args.holdout_public_commitment_sha256,
    }:
        raise ValueError("training authorization binds a different holdout commitment")
    rows = load_frozen_rows(args.manifest, expected_sha256=EXPECTED_MANIFEST_SHA256)
    bank, _ = validate_public_bank_registry(
        args.source_bank_registry,
        expected_sha256=args.source_bank_registry_sha256,
    )
    holdout, _ = validate_public_holdout_commitment(
        args.holdout_public_commitment,
        expected_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    validate_public_stage0_commitment(
        EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT,
        expected_sha256=EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256,
        bank_registry=bank,
        holdout_commitment=holdout,
    )
    mapping, mapping_by_index = load_mapping_registry(
        args.source_mapping_registry,
        rows,
        expected_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_path=str(args.holdout_public_commitment),
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    base_paths = validate_cache_inventory(
        rows, args.cache_dir, expected_sha256=EXPECTED_BASE_CACHE_SHA256
    )
    validate_base_payloads(base_paths, rows)
    teacher_paths_list = validate_cache_inventory(
        rows,
        args.target_prompt_cache_dir,
        expected_sha256=EXPECTED_TEACHER_CACHE_SHA256,
        role="erase",
    )
    teacher_paths = validate_teacher_payloads(teacher_paths_list, rows)
    model_provenance = validate_model_content_inventory(
        args.model, args.model_content_inventory_sha256
    )
    sidecar_paths, sidecar_manifest = validate_prompt_sidecar(
        rows,
        mapping_by_index,
        args.prompt_sidecar_dir,
        expected_inventory_sha256=args.prompt_sidecar_inventory_sha256,
        expected_manifest_sha256=args.prompt_sidecar_manifest_sha256,
        expected_mapping_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        expected_causal_stage0_public_commitment_sha256=(
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        expected_model_inventory_sha256=args.model_content_inventory_sha256,
        expected_runtime_registry_sha256=args.runtime_registry_sha256,
        expected_model_provenance=model_provenance,
    )
    validate_preflight(
        args,
        mapping,
        sidecar_manifest=sidecar_manifest,
        model_provenance=model_provenance,
    )
    _, registration_sha256 = validate_run_registration(args, mapping)
    train(
        args,
        rows,
        base_paths,
        teacher_paths,
        sidecar_paths,
        mapping,
        mapping_by_index,
        registration_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
