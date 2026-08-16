#!/usr/bin/env python3
"""Frozen protocol constants and provenance checks for the v3c fresh-dev gate."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLIT_PROTOCOL = "water_impact_dynamic_v3c_split_v1"
EVAL_PROTOCOL = "water_impact_dynamic_v3c_fresh_dev24_v1"
STAGE2_PROTOCOL = "water_impact_dynamic_v3c_fresh_dev24_stage2_v1"

TEST_PAIRS = "data/water_impact_dynamic_v1/test_pairs.csv"
TEST_PAIRS_SHA256 = "7a8ad92df03a78e8a972a2df552e61554836e225f2a310efc8e906e9cf2d0036"
EXHAUSTED_EVAL12 = "data/water_impact_dynamic_v1/eval12.csv"
EXHAUSTED_EVAL12_SHA256 = "dca68f8632e10ef83cc5f3867679c9cba54f4cbce96426db5db8c5214ac1ec1a"
FRESH_DEV_CSV = "data/water_impact_dynamic_v1/v3c_fresh_dev24.csv"
SEALED_FINAL_CSV = "data/water_impact_dynamic_v1/v3c_sealed_final36.csv"
FRESH_DEV_PROMPTS = "prompts/water_impact_dynamic_v1/v3c_fresh_dev24.prompts"
SEALED_FINAL_PROMPTS = "prompts/water_impact_dynamic_v1/v3c_sealed_final36.prompts"
SPLIT_REGISTRY = "data/water_impact_dynamic_v1/v3c_eval_split_registry.json"
SPLIT_REGISTRY_SHA256 = "4f31a291e8ffca07da4bf057e9a86df72f656c03aab65bc06d4c3c155b72962a"

STAGE2_REGISTRATION = "data/water_impact_dynamic_v1/v3c_fresh_dev24_stage2_registration.json"
STAGE2_TEMPLATE = (
    "data/water_impact_dynamic_v1/v3c_fresh_dev24_stage2_registration.template.json"
)

SPLIT_SEED = 26016001
GENERALIZATION_GROUPS = (
    "unseen_source",
    "unseen_receiver",
    "unseen_source_and_receiver",
)
PROMPT_VARIANTS = ("direct", "natural")
METHODS = ("v3b", "v3c")
FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)
BLIND_SEED = 26016002

MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
V3B_CHECKPOINT = (
    "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
    "checkpoint-000200"
)
V3B_CHECKPOINT_SHA256 = "f40f15f0a51c840db3e4fa8e2f931bdf89a4e5787f642513161e48d848fd723f"
V3C_TRAINING_ROOT = (
    "outputs/water_impact_dynamic_v3c/adapter_target_prompt_teacher_sigma2_scale4_v1"
)
V3C_CHECKPOINT = f"{V3C_TRAINING_ROOT}/checkpoint-000200"
ORIGINAL_RUN = "outputs/water_impact_dynamic_v3c/fresh_dev24_base"
V3B_RUN = (
    "outputs/water_impact_dynamic_v3b/"
    "fresh_dev24_target_prompt_teacher_scale4_v1_ckpt200_scale1p25"
)
V3C_RUN = (
    "outputs/water_impact_dynamic_v3c/"
    "fresh_dev24_target_prompt_teacher_sigma2_scale4_v1_ckpt200_scale1p25"
)
PUBLIC_REVIEW_DIR = (
    "experiments/water_impact_dynamic_fresh_dev24/"
    "v3b_vs_v3c_blind_review_v1_public"
)
PRIVATE_REVIEW_DIR = (
    "experiments/water_impact_dynamic_fresh_dev24/"
    "v3b_vs_v3c_blind_review_v1_private"
)
SCORE_DIR = "experiments/water_impact_dynamic_fresh_dev24/v3b_vs_v3c_scores_v1"

SCORE_FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)

# This structured rule is included verbatim in the stage-1 registry. It is
# therefore frozen before any eligible v3c generation, independently of the
# later checkpoint hashes in stage 2.
GATE_SPEC: dict[str, Any] = {
    "decision": "all_checks_must_pass",
    "control_method": "v3b",
    "treatment_method": "v3c",
    "control_usable_min": 20,
    "target_suppression_margin_on_control_usable_min": 6,
    "paired_target_improvements_min": 6,
    "clear_to_absent_improvements_min": 2,
    "clear_to_absent_generalization_groups_min": 2,
    "absent_target_count_margin_min": 2,
    "treatment_usable_min": 22,
    "receiver_points_floor": "max(38, v3b_receiver_points - 2)",
    "quality_points_floor": "max(32, v3b_quality_points - 2)",
    "footprint_suppression_on_control_usable": "v3c >= v3b",
    "strict_success_min": 2,
    "usable_definition": "receiver>=1 and quality>=1",
    "strict_definition": "target=0, footprint=0, receiver=2, quality=2",
    "control_usable_set": "C={sample: usable(v3b_sample)}",
    "valid_suppression_on_control_usable": (
        "a v3c target or footprint contribution on C is 0 when that v3c output is unusable"
    ),
    "absent_target_count": "count only usable outputs with target=0",
    "paired_target_improvement": (
        "sample in C with usable v3c and v3c_target < v3b_target"
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_path(project_root: Path, registered: str | Path) -> Path:
    path = Path(registered)
    return path if path.is_absolute() else project_root / path


def _require_file_hash(project_root: Path, registered: str, expected: str, label: str) -> Path:
    path = resolve_path(project_root, registered)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing frozen file: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA-256 mismatch: {actual} != {expected}")
    return path


def _prompt_bytes(rows: list[dict[str, str]]) -> bytes:
    lines = [
        " | ".join(
            (row["training_prompt"], row["source_object"], row["expected_factual_event"])
        )
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _selection_rank(row: dict[str, str]) -> str:
    value = (
        f"{SPLIT_SEED}:{row['generalization_group']}:"
        f"{row['prompt_variant']}:{row['pair_id']}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_expected_partition(
    test_rows: list[dict[str, str]], eval12_rows: list[dict[str, str]]
) -> tuple[list[tuple[int, dict[str, str]]], list[tuple[int, dict[str, str]]]]:
    """Return the only partition permitted by the registered SHA-rank rule."""

    if len(test_rows) != 72 or len(eval12_rows) != 12:
        raise ValueError("source test/eval12 manifests must contain exactly 72/12 rows")
    test_by_pair = {row["pair_id"]: (index, row) for index, row in enumerate(test_rows)}
    if len(test_by_pair) != 72:
        raise ValueError("test_pairs.csv contains duplicate pair_id")
    excluded: set[str] = set()
    for row in eval12_rows:
        pair_id = row["pair_id"]
        if pair_id in excluded or pair_id not in test_by_pair:
            raise ValueError("eval12 is not a unique subset of test_pairs")
        source_index, source = test_by_pair[pair_id]
        if int(row["source_test_index"]) != source_index:
            raise ValueError(f"eval12 source index mismatch: {pair_id}")
        for field, value in source.items():
            if row.get(field) != value:
                raise ValueError(f"eval12 source mismatch for {pair_id}/{field}")
        excluded.add(pair_id)

    expected_strata = {
        (group, variant)
        for group in GENERALIZATION_GROUPS
        for variant in PROMPT_VARIANTS
    }
    strata: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(test_rows):
        if row["pair_id"] in excluded:
            continue
        key = (row["generalization_group"], row["prompt_variant"])
        if key not in expected_strata:
            raise ValueError(f"unexpected split stratum: {key}")
        strata[key].append((index, row))
    if set(strata) != expected_strata or any(len(rows) != 10 for rows in strata.values()):
        raise ValueError("after eval12 exclusion every group/variant stratum must have 10 rows")

    fresh: list[tuple[int, dict[str, str]]] = []
    final: list[tuple[int, dict[str, str]]] = []
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda item: (_selection_rank(item[1]), item[0]))
        fresh.extend(ranked[:4])
        final.extend(ranked[4:])
    fresh.sort(key=lambda item: item[0])
    final.sort(key=lambda item: item[0])
    return fresh, final


def validate_split_registration(project_root: Path) -> dict[str, Any]:
    """Validate the byte-exact, exhaustive 12/24/36 partition and registry."""

    if len(SPLIT_REGISTRY_SHA256) != 64:
        raise RuntimeError(
            "v3c split registry hash is still a fail-closed placeholder; "
            "freeze stage 1 before generation"
        )
    test_path = _require_file_hash(
        project_root, TEST_PAIRS, TEST_PAIRS_SHA256, "test-pairs source"
    )
    eval12_path = _require_file_hash(
        project_root, EXHAUSTED_EVAL12, EXHAUSTED_EVAL12_SHA256, "exhausted eval12"
    )
    registry_path = _require_file_hash(
        project_root, SPLIT_REGISTRY, SPLIT_REGISTRY_SHA256, "v3c split registry"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "protocol": SPLIT_PROTOCOL,
        "status": "frozen_before_v3c_generation",
        "selection_seed": SPLIT_SEED,
        "selection_algorithm": (
            "sha256_rank_within_generalization_group_x_prompt_variant; "
            "lowest_4_of_10_to_fresh_dev"
        ),
        "source_test_pairs": TEST_PAIRS,
        "source_test_pairs_sha256": TEST_PAIRS_SHA256,
        "excluded_eval12": EXHAUSTED_EVAL12,
        "excluded_eval12_sha256": EXHAUSTED_EVAL12_SHA256,
        "fresh_dev_count": 24,
        "sealed_final_count": 36,
        "generalization_groups": list(GENERALIZATION_GROUPS),
        "prompt_variants": list(PROMPT_VARIANTS),
        "fresh_dev_per_group": 8,
        "fresh_dev_per_group_variant": 4,
        "review_semantics": {
            "reviewers": 2,
            "adjudication": "third_blinded_reviewer_for_every_atomic_disagreement",
            "canonical_agreement": "exact_two_reviewer_agreement",
            "canonical_disagreement": "majority_of_three; median_1_for_exact_0_1_2",
            "public_private_packages": "distinct_sibling_directories",
        },
        "gate_spec": GATE_SPEC,
        "sealed_final_policy": (
            "do_not_generate_inspect_or_score_until_fresh_dev_gate_passes_all_checks"
        ),
        "stage2_policy": (
            "v3c_checkpoint_and_training_artifact_hashes_must_be_registered_before_v3c_generation"
        ),
    }
    for field, value in expected_metadata.items():
        if registry.get(field) != value:
            raise ValueError(f"split registry {field} does not match frozen protocol")

    registered_files = registry.get("registered_files")
    expected_paths = (
        FRESH_DEV_CSV,
        SEALED_FINAL_CSV,
        FRESH_DEV_PROMPTS,
        SEALED_FINAL_PROMPTS,
    )
    if not isinstance(registered_files, dict) or set(registered_files) != set(expected_paths):
        raise ValueError("split registry does not bind the exact four split artifacts")
    for registered in expected_paths:
        record = registered_files[registered]
        if not isinstance(record, dict) or set(record) != {"sha256", "row_count"}:
            raise ValueError(f"invalid split artifact record: {registered}")
        _require_file_hash(
            project_root, registered, str(record["sha256"]), f"split artifact {registered}"
        )

    test_rows = read_csv(test_path)
    eval12_rows = read_csv(eval12_path)
    fresh_rows = read_csv(resolve_path(project_root, FRESH_DEV_CSV))
    final_rows = read_csv(resolve_path(project_root, SEALED_FINAL_CSV))
    if len(test_rows) != 72 or len(eval12_rows) != 12 or len(fresh_rows) != 24 or len(final_rows) != 36:
        raise ValueError("v3c split must have source/eval12/fresh/final counts 72/12/24/36")
    if registered_files[FRESH_DEV_CSV]["row_count"] != 24:
        raise ValueError("fresh-dev registered row count mismatch")
    if registered_files[SEALED_FINAL_CSV]["row_count"] != 36:
        raise ValueError("sealed-final registered row count mismatch")
    if registered_files[FRESH_DEV_PROMPTS]["row_count"] != 24:
        raise ValueError("fresh-dev prompt registered row count mismatch")
    if registered_files[SEALED_FINAL_PROMPTS]["row_count"] != 36:
        raise ValueError("sealed-final prompt registered row count mismatch")

    test_by_pair = {row["pair_id"]: (index, row) for index, row in enumerate(test_rows)}
    if len(test_by_pair) != 72:
        raise ValueError("test-pairs source contains duplicate pair_id")
    eval_pairs = {row["pair_id"] for row in eval12_rows}
    fresh_pairs = {row["pair_id"] for row in fresh_rows}
    final_pairs = {row["pair_id"] for row in final_rows}
    if len(eval_pairs) != 12 or len(fresh_pairs) != 24 or len(final_pairs) != 36:
        raise ValueError("a frozen split contains duplicate pair_id")
    if eval_pairs & fresh_pairs or eval_pairs & final_pairs or fresh_pairs & final_pairs:
        raise ValueError("eval12, fresh-dev24, and sealed-final36 are not disjoint")
    if eval_pairs | fresh_pairs | final_pairs != set(test_by_pair):
        raise ValueError("12/24/36 split is not exhaustive over test_pairs")
    expected_fresh, expected_final = derive_expected_partition(test_rows, eval12_rows)
    if [row["pair_id"] for row in fresh_rows] != [row["pair_id"] for _, row in expected_fresh]:
        raise ValueError("fresh-dev rows do not match the registered SHA-rank selection")
    if [row["pair_id"] for row in final_rows] != [row["pair_id"] for _, row in expected_final]:
        raise ValueError("sealed-final rows do not match the registered SHA-rank selection")
    for rows, name in ((fresh_rows, "fresh-dev"), (final_rows, "sealed-final")):
        for index, row in enumerate(rows):
            if int(row["eval_index"]) != index:
                raise ValueError(f"{name} eval_index is not ordered from zero")
            source_index, source = test_by_pair[row["pair_id"]]
            if int(row["source_test_index"]) != source_index:
                raise ValueError(f"{name} source_test_index mismatch")
            for field, expected in source.items():
                if row.get(field) != expected:
                    raise ValueError(f"{name} row differs from source field {field}")

    for group in GENERALIZATION_GROUPS:
        for variant in PROMPT_VARIANTS:
            count = sum(
                row["generalization_group"] == group and row["prompt_variant"] == variant
                for row in fresh_rows
            )
            if count != 4:
                raise ValueError(f"fresh-dev stratum {group}/{variant} has {count}, expected 4")
    for registered, rows in (
        (FRESH_DEV_PROMPTS, fresh_rows),
        (SEALED_FINAL_PROMPTS, final_rows),
    ):
        if resolve_path(project_root, registered).read_bytes() != _prompt_bytes(rows):
            raise ValueError(f"prompt file is not byte-aligned with {registered}")
    assignment = {
        row["pair_id"]: "exhausted_eval12" for row in eval12_rows
    }
    assignment.update({row["pair_id"]: "fresh_dev24" for row in fresh_rows})
    assignment.update({row["pair_id"]: "sealed_final36" for row in final_rows})
    assignment_hash = hashlib.sha256(
        json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if registry.get("assignment_sha256") != assignment_hash:
        raise ValueError("split assignment hash mismatch")
    return registry


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "REQUIRED_" in value or "TO_BE_FROZEN" in value
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _require_artifact_record(project_root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label}: expected exact path/SHA-256 record")
    path = resolve_path(project_root, str(record["path"]))
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing artifact: {path}")
    actual = artifact_sha256(path) if path.is_dir() else file_sha256(path)
    if record["sha256"] != actual or len(str(record["sha256"])) != 64:
        raise ValueError(f"{label}: artifact SHA-256 mismatch")
    return path


def build_stage2_registration(project_root: Path, *, created_utc: str | None = None) -> dict[str, Any]:
    """Hash the completed v3c checkpoint; caller must persist before generation."""

    validate_split_registration(project_root)
    if resolve_path(project_root, V3C_RUN).exists():
        raise RuntimeError("stage-2 registration must be frozen before the v3c run exists")
    checkpoint = resolve_path(project_root, V3C_CHECKPOINT)
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    state = checkpoint / "training_state.json"
    registration = resolve_path(project_root, f"{V3C_TRAINING_ROOT}/run_registration.json")
    sanity = resolve_path(project_root, f"{V3C_TRAINING_ROOT}/target_prompt_scale_sanity.json")
    required = {
        "v3c checkpoint": checkpoint,
        "v3c weights": weights,
        "v3c training state": state,
        "v3c training registration": registration,
        "v3c scale sanity": sanity,
    }
    for label, path in required.items():
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            raise FileNotFoundError(f"{label}: missing completed training artifact: {path}")
    v3b_checkpoint = resolve_path(project_root, V3B_CHECKPOINT)
    if not v3b_checkpoint.is_dir() or artifact_sha256(v3b_checkpoint) != V3B_CHECKPOINT_SHA256:
        raise ValueError("frozen v3b checkpoint artifact is missing or changed")
    return {
        "protocol": STAGE2_PROTOCOL,
        "status": "frozen_after_training_before_v3c_generation",
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "split_registry": {
            "path": SPLIT_REGISTRY,
            "sha256": SPLIT_REGISTRY_SHA256,
        },
        "v3b": {
            "checkpoint": {"path": V3B_CHECKPOINT, "sha256": V3B_CHECKPOINT_SHA256},
        },
        "v3c": {
            "checkpoint": {"path": V3C_CHECKPOINT, "sha256": artifact_sha256(checkpoint)},
            "weights": {
                "path": f"{V3C_CHECKPOINT}/pytorch_lora_weights.safetensors",
                "sha256": file_sha256(weights),
            },
            "training_state": {
                "path": f"{V3C_CHECKPOINT}/training_state.json",
                "sha256": file_sha256(state),
            },
            "run_registration": {
                "path": f"{V3C_TRAINING_ROOT}/run_registration.json",
                "sha256": file_sha256(registration),
            },
            "scale_sanity": {
                "path": f"{V3C_TRAINING_ROOT}/target_prompt_scale_sanity.json",
                "sha256": file_sha256(sanity),
            },
        },
        "generation_spec": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "prompts": FRESH_DEV_PROMPTS,
            "prompts_sha256": validate_split_registration(project_root)["registered_files"][
                FRESH_DEV_PROMPTS
            ]["sha256"],
            "sample_count": 24,
            "num_inference_steps": 25,
            "guidance_scale": 5.0,
            "num_frames": 49,
            "fps": 8,
            "height": 480,
            "width": 832,
            "dtype": "bf16",
            "device": "cuda",
            "lora_scale": 1.25,
            "original_run": ORIGINAL_RUN,
            "v3b_run": V3B_RUN,
            "v3c_run": V3C_RUN,
        },
    }


def validate_stage2_payload(project_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validate_split_registration(project_root)
    if _contains_placeholder(payload):
        raise ValueError("stage-2 registration contains a fail-closed placeholder")
    if payload.get("protocol") != STAGE2_PROTOCOL:
        raise ValueError("stage-2 protocol mismatch")
    if payload.get("status") != "frozen_after_training_before_v3c_generation":
        raise ValueError("stage-2 registration is not frozen")
    split = payload.get("split_registry")
    if split != {"path": SPLIT_REGISTRY, "sha256": SPLIT_REGISTRY_SHA256}:
        raise ValueError("stage-2 registration does not bind the frozen stage-1 split")
    v3b = payload.get("v3b")
    v3c = payload.get("v3c")
    if not isinstance(v3b, dict) or set(v3b) != {"checkpoint"}:
        raise ValueError("stage-2 v3b provenance is incomplete")
    if not isinstance(v3c, dict) or set(v3c) != {
        "checkpoint", "weights", "training_state", "run_registration", "scale_sanity"
    }:
        raise ValueError("stage-2 v3c provenance is incomplete")
    v3b_path = _require_artifact_record(project_root, v3b["checkpoint"], "v3b checkpoint")
    if str(v3b["checkpoint"]["path"]) != V3B_CHECKPOINT or artifact_sha256(v3b_path) != V3B_CHECKPOINT_SHA256:
        raise ValueError("stage-2 registration does not bind the registered v3b checkpoint")
    expected_v3c_paths = {
        "checkpoint": V3C_CHECKPOINT,
        "weights": f"{V3C_CHECKPOINT}/pytorch_lora_weights.safetensors",
        "training_state": f"{V3C_CHECKPOINT}/training_state.json",
        "run_registration": f"{V3C_TRAINING_ROOT}/run_registration.json",
        "scale_sanity": f"{V3C_TRAINING_ROOT}/target_prompt_scale_sanity.json",
    }
    for name, expected_path in expected_v3c_paths.items():
        path = _require_artifact_record(project_root, v3c[name], f"v3c {name}")
        if str(v3c[name]["path"]) != expected_path:
            raise ValueError(f"stage-2 v3c {name} path is not frozen")
        expected_digest = artifact_sha256(path) if name == "checkpoint" else file_sha256(path)
        if str(v3c[name]["sha256"]) != expected_digest:
            raise ValueError(f"stage-2 v3c {name} hash mismatch")
    expected_generation = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "prompts": FRESH_DEV_PROMPTS,
        "prompts_sha256": validate_split_registration(project_root)["registered_files"][
            FRESH_DEV_PROMPTS
        ]["sha256"],
        "sample_count": 24,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device": "cuda",
        "lora_scale": 1.25,
        "original_run": ORIGINAL_RUN,
        "v3b_run": V3B_RUN,
        "v3c_run": V3C_RUN,
    }
    if payload.get("generation_spec") != expected_generation:
        raise ValueError("stage-2 generation specification mismatch")
    return payload


def load_stage2_registration(project_root: Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(project_root, STAGE2_REGISTRATION)
    if not path.is_file():
        raise FileNotFoundError(
            "v3c generation is locked: create the stage-2 checkpoint registration after training"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_stage2_payload(project_root, payload)
    return path, payload


def validate_model_revision(project_root: Path) -> None:
    metadata = resolve_path(
        project_root, f"{MODEL}/.cache/huggingface/download/model_index.json.metadata"
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"missing frozen model revision metadata: {metadata}")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != MODEL_REVISION:
        raise ValueError("frozen model revision mismatch")


def _require_mapping(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{label}: {field}={payload.get(field)!r}, expected {value!r}")


def load_generation_run(
    project_root: Path,
    run_dir: str,
    label: str,
    eval_rows: list[dict[str, str]],
    stage2_path: Path | None = None,
    stage2: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[int, Path]]:
    """Validate one frozen 24-video arm, including pre-generation sidecars."""

    if label not in {"original", *METHODS}:
        raise ValueError(f"unexpected generation arm: {label}")
    if len(eval_rows) != 24:
        raise ValueError("fresh-dev generation requires exactly 24 registered rows")
    run_path = resolve_path(project_root, run_dir)
    manifest_path = run_path / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label}: missing generation manifest: {manifest_path}")
    split_sidecar = run_path / ".split_registry_sha256"
    if not split_sidecar.is_file() or split_sidecar.read_text(encoding="utf-8").strip() != SPLIT_REGISTRY_SHA256:
        raise ValueError(f"{label}: missing or changed pre-generation split binding")
    if label == "v3c":
        if stage2_path is None or stage2 is None:
            raise ValueError("v3c generation validation requires frozen stage-2 provenance")
        stage2_sidecar = run_path / ".stage2_registration_sha256"
        expected_stage2_hash = file_sha256(stage2_path)
        if (
            not stage2_sidecar.is_file()
            or stage2_sidecar.read_text(encoding="utf-8").strip() != expected_stage2_hash
        ):
            raise ValueError("v3c: missing or changed pre-generation stage-2 binding")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_mapping(
        manifest,
        {
            "baseline": "clean",
            "pipeline": "WanPipeline",
            "model": MODEL,
            "dry_run": False,
            "prompts": FRESH_DEV_PROMPTS,
        },
        f"{label} generation manifest",
    )
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: missing generation config")
    _require_mapping(
        generation,
        {
            "baseline": "clean",
            "seed": 42,
            "seeds": [int(row["seed"]) for row in eval_rows],
            "num_inference_steps": 25,
            "guidance_scale": 5.0,
            "num_frames": 49,
            "fps": 8,
            "height": 480,
            "width": 832,
            "dtype": "bf16",
            "device": "cuda",
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "vae_slicing": True,
            "vae_tiling": True,
            "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
            "activation_gate_dir": None,
            "persistent_activation_gate": False,
            "lora_target_phrases": [],
            "attention_gate_dir": None,
            "attention_suppression_phrases": [],
            "attention_suppression_strength": 20.0,
        },
        f"{label} generation config",
    )
    if label == "original":
        expected_lora = {"lora_path": None, "lora_sha256": None, "lora_scale": 1.0}
    elif label == "v3b":
        expected_lora = {
            "lora_path": V3B_CHECKPOINT,
            "lora_sha256": V3B_CHECKPOINT_SHA256,
            "lora_scale": 1.25,
        }
    else:
        assert stage2 is not None
        expected_lora = {
            "lora_path": V3C_CHECKPOINT,
            "lora_sha256": stage2["v3c"]["checkpoint"]["sha256"],
            "lora_scale": 1.25,
        }
    _require_mapping(generation, expected_lora, f"{label} LoRA config")

    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 24:
        raise ValueError(f"{label}: generation manifest must contain exactly 24 items")
    expected_video_root = (run_path / "videos").resolve(strict=True)
    videos: dict[int, Path] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label}: invalid generation item")
        index = int(item.get("index", -1))
        if index in videos or index not in range(24):
            raise ValueError(f"{label}: invalid or duplicate generation index {index}")
        row = eval_rows[index]
        _require_mapping(
            item,
            {
                "prompt": row["training_prompt"],
                "target_concept": row["source_object"],
                "expected_effect": row["expected_factual_event"],
                "seed": int(row["seed"]),
            },
            f"{label} item {index}",
        )
        video_value = item.get("video_path")
        if not isinstance(video_value, str):
            raise ValueError(f"{label}: item {index} has no video path")
        video = resolve_path(project_root, video_value)
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f"{label}: missing video at index {index}: {video}")
        try:
            video.resolve(strict=True).relative_to(expected_video_root)
        except ValueError as exc:
            raise ValueError(f"{label}: video escapes its frozen run directory") from exc
        videos[index] = video
    actual_videos = {path.resolve(strict=True) for path in (run_path / "videos").rglob("*.mp4")}
    if actual_videos != {path.resolve(strict=True) for path in videos.values()}:
        raise ValueError(f"{label}: videos directory contains unregistered or missing MP4 files")
    return manifest_path, manifest, videos


if __name__ == "__main__":
    payload = validate_split_registration(Path.cwd())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
