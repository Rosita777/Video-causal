#!/usr/bin/env python3
"""Frozen provenance checks for the water-impact v3b eval12 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROTOCOL = "water_impact_dynamic_v3b_eval12_v1"
METHODS = ("balanced", "v3b")
FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)

EVAL_CSV = "data/water_impact_dynamic_v1/eval12.csv"
EVAL_CSV_SHA256 = "dca68f8632e10ef83cc5f3867679c9cba54f4cbce96426db5db8c5214ac1ec1a"
PROMPTS = "prompts/water_impact_dynamic_v1/eval12.prompts"
PROMPTS_SHA256 = "06dae57a0202e2d53e32fc02f9b26fd694237755a18f85bdd67c728bf706681c"
TRAIN_MANIFEST = "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
TRAIN_MANIFEST_SHA256 = "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
SEEDS = (26000, 26005, 26010, 26015, 26024, 26031, 26040, 26047, 26048, 26055, 26064, 26071)

ORIGINAL_RUN = "outputs/water_impact_dynamic_v1/eval12_base"
ORIGINAL_GENERATION_MANIFEST_SHA256 = "c29e159f8722728c1e4ce08d67895281518f0f7c0fe6cecb3a28cdc43a39deba"

BALANCED_RUN = "outputs/water_impact_dynamic_v1/eval12_v3_balanced_seeded_ckpt200_scale1p25"
BALANCED_CHECKPOINT = "outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v3_balanced_seeded/checkpoint-000200"
BALANCED_CHECKPOINT_SHA256 = "e61eb33235da3ad68f08e31c451c6690db194bc9b3aa498df58194549955d7f0"
BALANCED_WEIGHTS_SHA256 = "efbd57cc118f05c2ce82a55429057aa03874cc7a615de73063c564cef3f11701"
BALANCED_TRAINING_STATE_SHA256 = "91593e27a0bfde232c7cc344a1579e3f1c203d825bd3252e6160532a833f1142"
BALANCED_GENERATION_MANIFEST_SHA256 = "af1ada55eb56fe28261765e23b4b24b782d403736f6310216602fee069eecf1a"

V3B_RUN = "outputs/water_impact_dynamic_v3b/eval12_target_prompt_teacher_scale4_v1_ckpt200_scale1p25"
V3B_TRAINING_ROOT = "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1"
V3B_CHECKPOINT = f"{V3B_TRAINING_ROOT}/checkpoint-000200"
V3B_CHECKPOINT_SHA256 = "f40f15f0a51c840db3e4fa8e2f931bdf89a4e5787f642513161e48d848fd723f"
V3B_WEIGHTS_SHA256 = "d3fecf26b7f1ca6c4a8f46c86850a47a7ec5a62762d0e0aa15c49363040875d3"
V3B_TRAINING_STATE_SHA256 = "0f9aa26e825f4f6f497b1312c507b685c054bc319f2f9f538e45eeb7a7908bea"
V3B_REGISTRATION_SHA256 = "53f0a7c472ba02a38b90b55651f378e5feda0bcd709f86786702de163b3a87f4"
V3B_SANITY_SHA256 = "26fb8b1ff9e0d446fd186765ba1ff9a9d1a085d75d230cd0a419509ea00bbb12"

BASE_CACHE = "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
BASE_CACHE_SHA256 = "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
TEACHER_CACHE = "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
TEACHER_CACHE_SHA256 = "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
TEACHER_CACHE_MANIFEST_SHA256 = "c467d7f81ee22b2c4b1ff719537487fbfc808eacc98e730c3d24f0a17aca77cb"
TARGET_PROMPT_BINDING_SHA256 = "9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc"
UNIQUE_TARGET_EMBEDDING_SHA256 = "a15f5e910358d5e95bcdd995303abb7eb7e7302fd9ee649c4cfebf3b8f6b6330"
EXPECTED_INITIAL_LORA_SHA256 = "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8"
EXPECTED_SAMPLE_ORDER_SHA256 = "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb"

TRAINER = "scripts/train_wan_waterdrop_lora.py"
TRAINER_SHA256 = "6912d2a2adb4ed659ae2bc95b6882106366b707949ed56be71913f462cfec087"
TRAIN_LAUNCHER = "scripts/run_water_impact_dynamic_sft_v3b_teacher.sh"
TRAIN_LAUNCHER_SHA256 = "d7879c8885f401a3d9972a489d53b48ea68466b6ca86e8922c9b36f458bbc66f"
TRAIN_PROTOCOL_DOC = "docs/water_impact_dynamic_v3b_target_prompt_teacher.md"
TRAIN_PROTOCOL_DOC_SHA256 = "ac96d88327984f91d8c0d1b2075eaa544251382b01ec11075ea16b2d9022422a"

CALIBRATION_ID = "lambda4_from_lambda1_first16_output_gradient_v1"
SCALE_SANITY_PROTOCOL = "water_impact_dynamic_v3b_scale_sanity_v2"

SCORE_FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)


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


def cache_inventory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.resolve(strict=True).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_path(project_root: Path, registered: str | Path) -> Path:
    path = Path(registered)
    return path if path.is_absolute() else project_root / path


def require_file_hash(project_root: Path, registered: str, expected: str, label: str) -> Path:
    path = resolve_path(project_root, registered)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing frozen file: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA-256 mismatch: {actual} != {expected}")
    return path


def require_artifact_hash(project_root: Path, registered: str, expected: str, label: str) -> Path:
    path = resolve_path(project_root, registered)
    if not path.is_dir():
        raise FileNotFoundError(f"{label}: missing frozen directory: {path}")
    actual = artifact_sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: artifact SHA-256 mismatch: {actual} != {expected}")
    return path


def require_mapping(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{label}: {field}={payload.get(field)!r}, expected {value!r}")


def validate_model_revision(project_root: Path) -> dict[str, str]:
    metadata_path = resolve_path(
        project_root,
        f"{MODEL}/.cache/huggingface/download/model_index.json.metadata",
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing frozen model revision metadata: {metadata_path}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != MODEL_REVISION:
        actual = lines[0] if lines else "<empty>"
        raise ValueError(f"frozen model revision mismatch: {actual} != {MODEL_REVISION}")
    return {
        "model": MODEL,
        "model_revision_metadata_path": (
            f"{MODEL}/.cache/huggingface/download/model_index.json.metadata"
        ),
        "model_revision": lines[0],
    }


def load_frozen_inputs(project_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eval_path = require_file_hash(project_root, EVAL_CSV, EVAL_CSV_SHA256, "eval12 CSV")
    prompt_path = require_file_hash(project_root, PROMPTS, PROMPTS_SHA256, "eval12 prompts")
    train_path = require_file_hash(
        project_root, TRAIN_MANIFEST, TRAIN_MANIFEST_SHA256, "training manifest"
    )
    eval_rows = read_csv(eval_path)
    train_rows = read_csv(train_path)
    if len(eval_rows) != 12 or tuple(int(row["eval_index"]) for row in eval_rows) != tuple(range(12)):
        raise ValueError("frozen eval12 must contain ordered indices 0 through 11")
    if tuple(int(row["seed"]) for row in eval_rows) != SEEDS:
        raise ValueError("frozen eval12 seeds do not match the registered seed vector")
    prompt_rows = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pieces = [piece.strip() for piece in line.split("|")]
            if len(pieces) != 3:
                raise ValueError("frozen prompt file contains a malformed row")
            prompt_rows.append(tuple(pieces))
    expected_prompts = [
        (row["training_prompt"], row["source_object"], row["expected_factual_event"])
        for row in eval_rows
    ]
    if prompt_rows != expected_prompts:
        raise ValueError("frozen prompt file is not row-aligned with eval12")
    return eval_rows, train_rows


def validate_training_caches(
    project_root: Path, train_rows: list[dict[str, str]]
) -> dict[str, Any]:
    base_dir = resolve_path(project_root, BASE_CACHE)
    base_paths = [
        base_dir / f"{index:03d}_{row['scene_id']}.pt"
        for index, row in enumerate(train_rows)
    ]
    actual_base = sorted(base_dir.glob("*.pt"))
    if (
        len(base_paths) != 214
        or set(actual_base) != set(base_paths)
        or any(not path.is_file() for path in base_paths)
    ):
        raise ValueError(
            f"frozen base-cache inventory mismatch: expected=214 actual={len(actual_base)}"
        )
    base_hash = cache_inventory_sha256(base_paths)
    if base_hash != BASE_CACHE_SHA256:
        raise ValueError(f"frozen base-cache content hash mismatch: {base_hash}")

    teacher_dir = resolve_path(project_root, TEACHER_CACHE)
    erase_rows = [
        (index, row)
        for index, row in enumerate(train_rows)
        if row["training_role"] == "erase"
    ]
    teacher_paths = [
        teacher_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in erase_rows
    ]
    actual_teacher = sorted(teacher_dir.glob("*.pt"))
    if (
        len(teacher_paths) != 178
        or set(actual_teacher) != set(teacher_paths)
        or any(not path.is_file() for path in teacher_paths)
    ):
        raise ValueError(
            "frozen target-prompt cache inventory mismatch: "
            f"expected=178 actual={len(actual_teacher)}"
        )
    teacher_hash = cache_inventory_sha256(teacher_paths)
    if teacher_hash != TEACHER_CACHE_SHA256:
        raise ValueError(f"frozen target-prompt cache content hash mismatch: {teacher_hash}")
    teacher_manifest_path = require_file_hash(
        project_root,
        f"{TEACHER_CACHE}/cache_manifest.json",
        TEACHER_CACHE_MANIFEST_SHA256,
        "target-prompt cache manifest",
    )
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    require_mapping(
        teacher_manifest,
        {
            "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_v1",
            "source_manifest": TRAIN_MANIFEST,
            "source_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "model": MODEL,
            "dtype": "torch.bfloat16",
            "do_classifier_free_guidance": False,
            "max_sequence_length": 226,
            "model_revision": MODEL_REVISION,
            "erase_row_count": 178,
            "unique_prompt_count": 24,
            "prompt_binding_sha256": TARGET_PROMPT_BINDING_SHA256,
            "unique_embedding_sha256": UNIQUE_TARGET_EMBEDDING_SHA256,
            "cache_inventory_sha256": TEACHER_CACHE_SHA256,
        },
        "target-prompt cache manifest",
    )
    return {
        "base_cache_dir": BASE_CACHE,
        "base_cache_entry_count": 214,
        "base_cache_sha256": base_hash,
        "target_prompt_cache_dir": TEACHER_CACHE,
        "target_prompt_cache_entry_count": 178,
        "target_prompt_cache_sha256": teacher_hash,
        "target_prompt_cache_manifest_path": f"{TEACHER_CACHE}/cache_manifest.json",
        "target_prompt_cache_manifest_sha256": file_sha256(teacher_manifest_path),
    }


def validate_generation_config(
    label: str,
    manifest: dict[str, Any],
    eval_rows: list[dict[str, str]],
) -> None:
    require_mapping(
        manifest,
        {
            "baseline": "clean",
            "pipeline": "WanPipeline",
            "model": MODEL,
            "dry_run": False,
            "prompts": PROMPTS,
        },
        f"{label} generation manifest",
    )
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: missing generation configuration")
    require_mapping(
        generation,
        {
            "baseline": "clean",
            "seed": 42,
            "seeds": list(SEEDS),
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
        require_mapping(
            generation,
            {"lora_path": None, "lora_sha256": None, "lora_scale": 1.0},
            "original generation config",
        )
    else:
        checkpoint = BALANCED_CHECKPOINT if label == "balanced" else V3B_CHECKPOINT
        checkpoint_hash = BALANCED_CHECKPOINT_SHA256 if label == "balanced" else V3B_CHECKPOINT_SHA256
        require_mapping(
            generation,
            {"lora_path": checkpoint, "lora_sha256": checkpoint_hash, "lora_scale": 1.25},
            f"{label} generation config",
        )


def load_generation_run(
    project_root: Path,
    run_dir: str,
    label: str,
    eval_rows: list[dict[str, str]],
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], dict[int, Path]]:
    run_path = resolve_path(project_root, run_dir)
    manifest_path = run_path / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label}: missing generation manifest: {manifest_path}")
    if expected_manifest_sha256 is not None and file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"{label}: frozen generation manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_generation_config(label, manifest, eval_rows)
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != 12:
        raise ValueError(f"{label}: generation manifest must contain exactly 12 items")
    expected_video_root = (run_path / "videos").resolve(strict=True)
    videos: dict[int, Path] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{label}: invalid generation item")
        index = int(item.get("index", -1))
        if index in videos or index not in range(12):
            raise ValueError(f"{label}: invalid or duplicate generation index {index}")
        row = eval_rows[index]
        require_mapping(
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
            raise ValueError(f"{label}: video escapes the frozen run directory: {video}") from exc
        videos[index] = video
    actual_videos = {
        path.resolve(strict=True) for path in (run_path / "videos").rglob("*.mp4")
    }
    registered_videos = {path.resolve(strict=True) for path in videos.values()}
    if actual_videos != registered_videos:
        raise ValueError(
            f"{label}: videos directory contains unregistered or missing mp4 files"
        )
    return manifest_path, manifest, videos


def _validate_balanced_state(state: dict[str, Any]) -> None:
    require_mapping(
        state,
        {
            "step": 200,
            "max_steps": 200,
            "manifest": TRAIN_MANIFEST,
            "manifest_sha256": TRAIN_MANIFEST_SHA256,
            "model": MODEL,
            "cache_dir": BASE_CACHE,
            "cache_entry_count": 214,
            "cache_inventory_sha256": BASE_CACHE_SHA256,
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "grad_accum": 1,
            "device": "cuda",
            "rank": 16,
            "alpha": 16,
            "learning_rate": 5e-5,
            "seed": 26000,
            "initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
            "role": "all",
            "objective": "plain",
            "preserve_weight": 4.0,
            "balanced_roles": True,
            "role_step_counts": {"erase": 100, "preserve": 100},
            "sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
            "causal_gate_dir": None,
            "activation_gate_dir": None,
            "component_gate_dir": None,
            "target_phrase": [],
            "persistent_causal_time": False,
        },
        "balanced training state",
    )


def validate_balanced_checkpoint(project_root: Path) -> dict[str, Any]:
    checkpoint = require_artifact_hash(
        project_root, BALANCED_CHECKPOINT, BALANCED_CHECKPOINT_SHA256, "balanced checkpoint"
    )
    weights = require_file_hash(
        project_root,
        f"{BALANCED_CHECKPOINT}/pytorch_lora_weights.safetensors",
        BALANCED_WEIGHTS_SHA256,
        "balanced LoRA weights",
    )
    state_path = require_file_hash(
        project_root,
        f"{BALANCED_CHECKPOINT}/training_state.json",
        BALANCED_TRAINING_STATE_SHA256,
        "balanced training state",
    )
    _validate_balanced_state(json.loads(state_path.read_text(encoding="utf-8")))
    return {
        "checkpoint_path": BALANCED_CHECKPOINT,
        "checkpoint_sha256": artifact_sha256(checkpoint),
        "weights_path": f"{BALANCED_CHECKPOINT}/pytorch_lora_weights.safetensors",
        "weights_sha256": file_sha256(weights),
        "training_state_path": f"{BALANCED_CHECKPOINT}/training_state.json",
        "training_state_sha256": file_sha256(state_path),
    }


def _validate_registration(project_root: Path, registration: dict[str, Any]) -> None:
    training_config = {
        "model": MODEL,
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
        "target_prompt_calibration_id": CALIBRATION_ID,
        "target_prompt_teacher_weight": 4.0,
        "sanity_mean_min": 0.2,
        "sanity_mean_max": 0.5,
        "sanity_single_max": 1.0,
    }
    require_mapping(
        registration,
        {
            "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_scale4_v1",
            "calibration_id": CALIBRATION_ID,
            "output_dir": V3B_TRAINING_ROOT,
            "target_prompt_teacher_weight": 4.0,
            "sanity_mean_min": 0.2,
            "sanity_mean_max": 0.5,
            "sanity_single_max": 1.0,
            "sanity_formula": "s_i = weight * sqrt(target_prompt_teacher_loss / flow_loss)",
            "sanity_aggregation": "arithmetic_mean_over_first_16_erase_steps",
            "selection_rule": "nearest_power_of_two(0.30 / mean_i(sqrt(r_i)))",
            "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "base_cache_sha256": BASE_CACHE_SHA256,
            "teacher_cache_sha256": TEACHER_CACHE_SHA256,
            "expected_initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
            "lambda1_scale_invalid": True,
            "lambda1_generation_count": 0,
            "lambda1_mean_raw_loss_ratio_first_16": 0.005843,
            "training_config": training_config,
            "trainer_path": TRAINER,
            "trainer_sha256": TRAINER_SHA256,
            "launcher_path": TRAIN_LAUNCHER,
            "launcher_sha256": TRAIN_LAUNCHER_SHA256,
            "protocol_doc_path": TRAIN_PROTOCOL_DOC,
            "protocol_doc_sha256": TRAIN_PROTOCOL_DOC_SHA256,
        },
        "v3b run registration",
    )
    for path, expected, label in (
        (TRAINER, TRAINER_SHA256, "registered trainer"),
        (TRAIN_LAUNCHER, TRAIN_LAUNCHER_SHA256, "registered training launcher"),
        (TRAIN_PROTOCOL_DOC, TRAIN_PROTOCOL_DOC_SHA256, "registered training protocol"),
    ):
        require_file_hash(project_root, path, expected, label)
    expected_pilot = {
        "outputs/water_impact_dynamic_v3b/logs/train_target_prompt_teacher_v1.log": "c0f35542d9be763ea4a446af773e0e22fe44913b019b89aca51588780f5719ba",
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/checkpoint-000025/pytorch_lora_weights.safetensors": "2ee9f08c83d291630c09efcdf5bf0f8ae082f7b23b4c6be0ed89de791377ff3b",
        "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/checkpoint-000025/training_state.json": "d51fe90cedc168125e773f4c44ad458cc2baf84f409df6ed29f20cc09bcae854",
    }
    pilot = registration.get("lambda1_artifacts")
    if not isinstance(pilot, list) or {
        str(row.get("path")): str(row.get("sha256")) for row in pilot if isinstance(row, dict)
    } != expected_pilot:
        raise ValueError("v3b run registration does not contain the frozen lambda=1 evidence")
    for path, digest in expected_pilot.items():
        require_file_hash(project_root, path, digest, "lambda=1 pilot evidence")


def _close(actual: Any, expected: float, label: str) -> None:
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def _validate_sanity(sanity: dict[str, Any]) -> None:
    require_mapping(
        sanity,
        {
            "protocol": SCALE_SANITY_PROTOCOL,
            "calibration_id": CALIBRATION_ID,
            "run_registration_sha256": V3B_REGISTRATION_SHA256,
            "formula": "s_i = weight * sqrt(target_prompt_teacher_loss / flow_loss)",
            "aggregation": "arithmetic_mean_over_first_16_erase_steps",
            "weight": 4.0,
            "mean_output_gradient_norm_ratio_min": 0.2,
            "mean_output_gradient_norm_ratio_max": 0.5,
            "single_output_gradient_norm_ratio_max": 1.0,
            "observation_count": 16,
            "passed": True,
        },
        "v3b scale sanity",
    )
    observations = sanity.get("observations")
    if not isinstance(observations, list) or len(observations) != 16:
        raise ValueError("v3b scale sanity must contain exactly 16 observations")
    raw: list[float] = []
    output_ratios: list[float] = []
    for position, row in enumerate(observations):
        if not isinstance(row, dict) or row.get("global_step") != 2 * position + 1:
            raise ValueError("v3b scale sanity observations are not the first 16 erase steps")
        flow = float(row.get("flow_loss", float("nan")))
        teacher = float(row.get("target_prompt_teacher_loss", float("nan")))
        if not math.isfinite(flow) or flow <= 0 or not math.isfinite(teacher) or teacher < 0:
            raise ValueError("v3b scale sanity contains an invalid loss")
        ratio = teacher / flow
        weighted = 4.0 * ratio
        output_ratio = 4.0 * math.sqrt(ratio)
        _close(row.get("raw_loss_ratio"), ratio, "sanity raw loss ratio")
        _close(row.get("weighted_loss_ratio"), weighted, "sanity weighted loss ratio")
        _close(
            row.get("weighted_output_gradient_norm_ratio"),
            output_ratio,
            "sanity output-gradient ratio",
        )
        raw.append(ratio)
        output_ratios.append(output_ratio)
    derived = {
        "mean_raw_loss_ratio": statistics.fmean(raw),
        "mean_weighted_loss_ratio": 4.0 * statistics.fmean(raw),
        "mean_weighted_output_grad_ratio": statistics.fmean(output_ratios),
        "median_weighted_output_grad_ratio": statistics.median(output_ratios),
        "max_weighted_output_gradient_norm_ratio": max(output_ratios),
    }
    for field, expected in derived.items():
        _close(sanity.get(field), expected, f"v3b scale sanity {field}")
    if not 0.2 <= derived["mean_weighted_output_grad_ratio"] <= 0.5:
        raise ValueError("v3b scale sanity mean output-gradient ratio failed")
    if derived["max_weighted_output_gradient_norm_ratio"] > 1.0:
        raise ValueError("v3b scale sanity maximum output-gradient ratio failed")


def _validate_v3b_state(state: dict[str, Any]) -> None:
    require_mapping(
        state,
        {
            "step": 200,
            "max_steps": 200,
            "manifest": TRAIN_MANIFEST,
            "manifest_sha256": TRAIN_MANIFEST_SHA256,
            "model": MODEL,
            "cache_dir": BASE_CACHE,
            "cache_entry_count": 214,
            "cache_inventory_sha256": BASE_CACHE_SHA256,
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "grad_accum": 1,
            "device": "cuda",
            "rank": 16,
            "alpha": 16,
            "learning_rate": 5e-5,
            "seed": 26000,
            "initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
            "role": "all",
            "objective": "target_prompt_teacher",
            "preserve_weight": 4.0,
            "balanced_roles": True,
            "role_step_counts": {"erase": 100, "preserve": 100},
            "sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
            "causal_gate_dir": None,
            "activation_gate_dir": None,
            "component_gate_dir": None,
            "target_phrase": [],
            "persistent_causal_time": False,
            "target_prompt_teacher_enabled": True,
            "target_prompt_teacher_weight": 4.0,
            "target_prompt_calibration_id": CALIBRATION_ID,
            "teacher_prompt_field": "target_generation_prompt",
            "teacher_adapter_mode": "disabled",
            "teacher_stop_gradient": True,
            "teacher_uses_same_noisy_latent": True,
            "teacher_uses_same_timestep": True,
            "trainer_sha256": TRAINER_SHA256,
            "run_registration_path": f"{V3B_TRAINING_ROOT}/run_registration.json",
            "run_registration_sha256": V3B_REGISTRATION_SHA256,
            "target_prompt_cache_dir": TEACHER_CACHE,
            "target_prompt_cache_entry_count": 178,
            "target_prompt_cache_inventory_sha256": TEACHER_CACHE_SHA256,
            "target_prompt_cache_manifest_sha256": TEACHER_CACHE_MANIFEST_SHA256,
            "target_prompt_binding_sha256": TARGET_PROMPT_BINDING_SHA256,
            "target_prompt_unique_count": 24,
            "target_prompt_unique_embedding_sha256": UNIQUE_TARGET_EMBEDDING_SHA256,
            "teacher_scale_sanity_protocol": SCALE_SANITY_PROTOCOL,
            "teacher_scale_sanity_count": 16,
            "teacher_scale_sanity_mean_output_gradient_norm_ratio_min": 0.2,
            "teacher_scale_sanity_mean_output_gradient_norm_ratio_max": 0.5,
            "teacher_scale_sanity_single_output_gradient_norm_ratio_max": 1.0,
            "teacher_scale_sanity_passed": True,
            "teacher_scale_sanity_path": f"{V3B_TRAINING_ROOT}/target_prompt_scale_sanity.json",
            "teacher_scale_sanity_sha256": V3B_SANITY_SHA256,
        },
        "v3b training state",
    )


def validate_v3b_checkpoint(project_root: Path) -> dict[str, Any]:
    checkpoint = require_artifact_hash(
        project_root, V3B_CHECKPOINT, V3B_CHECKPOINT_SHA256, "v3b checkpoint-200"
    )
    weights = require_file_hash(
        project_root,
        f"{V3B_CHECKPOINT}/pytorch_lora_weights.safetensors",
        V3B_WEIGHTS_SHA256,
        "v3b LoRA weights",
    )
    state_path = require_file_hash(
        project_root,
        f"{V3B_CHECKPOINT}/training_state.json",
        V3B_TRAINING_STATE_SHA256,
        "v3b training state",
    )
    registration_path = require_file_hash(
        project_root,
        f"{V3B_TRAINING_ROOT}/run_registration.json",
        V3B_REGISTRATION_SHA256,
        "v3b run registration",
    )
    sanity_path = require_file_hash(
        project_root,
        f"{V3B_TRAINING_ROOT}/target_prompt_scale_sanity.json",
        V3B_SANITY_SHA256,
        "v3b scale sanity",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    _validate_v3b_state(state)
    _validate_registration(project_root, registration)
    _validate_sanity(sanity)
    _close(
        state.get("teacher_scale_sanity_mean_weighted_output_gradient_norm_ratio"),
        float(sanity["mean_weighted_output_grad_ratio"]),
        "training-state/sanity mean output-gradient binding",
    )
    _close(
        state.get("teacher_scale_sanity_mean_raw_loss_ratio"),
        float(sanity["mean_raw_loss_ratio"]),
        "training-state/sanity raw-ratio binding",
    )
    return {
        "checkpoint_path": V3B_CHECKPOINT,
        "checkpoint_sha256": artifact_sha256(checkpoint),
        "weights_path": f"{V3B_CHECKPOINT}/pytorch_lora_weights.safetensors",
        "weights_sha256": file_sha256(weights),
        "training_state_path": f"{V3B_CHECKPOINT}/training_state.json",
        "training_state_sha256": file_sha256(state_path),
        "run_registration_path": f"{V3B_TRAINING_ROOT}/run_registration.json",
        "run_registration_sha256": file_sha256(registration_path),
        "scale_sanity_path": f"{V3B_TRAINING_ROOT}/target_prompt_scale_sanity.json",
        "scale_sanity_sha256": file_sha256(sanity_path),
        "scale_sanity_mean_output_gradient_norm_ratio": sanity[
            "mean_weighted_output_grad_ratio"
        ],
        "scale_sanity_max_output_gradient_norm_ratio": sanity[
            "max_weighted_output_gradient_norm_ratio"
        ],
    }


def preflight(project_root: Path) -> dict[str, Any]:
    eval_rows, train_rows = load_frozen_inputs(project_root)
    original_manifest, _, _ = load_generation_run(
        project_root,
        ORIGINAL_RUN,
        "original",
        eval_rows,
        expected_manifest_sha256=ORIGINAL_GENERATION_MANIFEST_SHA256,
    )
    balanced_manifest, _, _ = load_generation_run(
        project_root,
        BALANCED_RUN,
        "balanced",
        eval_rows,
        expected_manifest_sha256=BALANCED_GENERATION_MANIFEST_SHA256,
    )
    return {
        "protocol": PROTOCOL,
        "model": validate_model_revision(project_root),
        "original_generation_manifest_sha256": file_sha256(original_manifest),
        "balanced_generation_manifest_sha256": file_sha256(balanced_manifest),
        "training_inputs": validate_training_caches(project_root, train_rows),
        "balanced": validate_balanced_checkpoint(project_root),
        "v3b": validate_v3b_checkpoint(project_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(preflight(args.project_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
