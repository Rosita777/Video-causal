#!/usr/bin/env python3
"""Fail-closed runner for v4 Original screening and registered U/W generation.

There is deliberately no command for historical sealed data.  Screening runs
accept only an Original arm committed at Stage 0.  Final runs require both
Stage 1 and the sole eligible step-200 checkpoint, and generate exactly one of
Original/v3b/v4 for U or W without skip/resume semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import water_impact_dynamic_v4_eval_protocol as protocol
from select_water_impact_dynamic_v4_eval import build_screening_review_package


SCREENING_GENERATION_PROTOCOL = "water_impact_dynamic_v4_screening_generation_v2"
MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
GENERATOR = "scripts/generate_wan_clean.py"
GENERATION_SPEC_PROTOCOL = "water_impact_dynamic_v4_generation_spec_v2"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return [dict(row) for row in protocol.read_csv(path)]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        lists = [
            payload[key]
            for key in ("candidates", "rows", "selected_cases", "units")
            if isinstance(payload.get(key), list)
        ]
        if len(lists) != 1:
            raise ValueError("private manifest JSON must expose exactly one recognized row list")
        rows = lists[0]
    else:
        raise ValueError("private manifest must be CSV or structured JSON")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("private manifest rows must be objects")
    return [dict(row) for row in rows]


def _require_inside_private_root(path: Path, private_root: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: missing private non-symlink file")
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("evaluator-only private root must be a real directory")
    resolved_root = private_root.resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label}: artifact escapes evaluator-only private root") from exc


def _require_future_output_inside_private_root(
    path: Path, private_root: Path, label: str
) -> None:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("evaluator-only private root must be a real directory")
    try:
        path.resolve().relative_to(private_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}: output escapes evaluator-only private root") from exc


def _read_single_seed(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = text
    value = payload.get("seed") if isinstance(payload, dict) else payload
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("screening seed artifact must contain exactly one integer seed") from exc
    if seed < 0 or seed >= 1 << 63:
        raise ValueError("screening seed is outside the registered signed-63-bit domain")
    return seed


def _read_seed_inventory(path: Path, *, dataset: str) -> set[int]:
    return protocol.validate_forbidden_seed_inventory(path, dataset=dataset)


def _validate_generation_spec(
    path: Path,
    *,
    committed_sha256: str,
    private_root: Path,
) -> dict[str, Any]:
    _require_inside_private_root(path, private_root, "generation spec")
    if protocol.file_sha256(path) != committed_sha256:
        raise ValueError("generation spec bytes differ from Stage-0 commitment")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "protocol",
        "status",
        "model_inventory_sha256",
        "runtime_registry",
        "generation_spec",
        "source_mode",
    }:
        raise ValueError("private generation spec fields are not exact")
    runtime_ref = payload.get("runtime_registry")
    if (
        payload["protocol"] != GENERATION_SPEC_PROTOCOL
        or payload["status"] != "frozen_before_original_render"
        or payload["source_mode"] != "Original_screening_then_matched_O_v3b_v4"
        or payload["generation_spec"] != protocol.GENERATION_SPEC
        or payload["model_inventory_sha256"]
        != protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256
        or not isinstance(runtime_ref, dict)
        or set(runtime_ref) != {"path", "sha256"}
        or runtime_ref["path"] != protocol.RUNTIME_REGISTRY
        or not protocol.is_sha256(runtime_ref["sha256"])
    ):
        raise ValueError("private generation spec differs from frozen executable contract")
    return payload


def _validate_generation_runtime(
    project_root: Path,
    generation_spec: Mapping[str, Any],
    python: str,
    *,
    run: Any = None,
) -> None:
    """Validate registry bytes and the child inference interpreter before reservation."""

    expected_python = str(protocol.RUNTIME_REGISTRY_PAYLOAD["python_executable"])
    if python != expected_python:
        raise ValueError("generation interpreter differs from frozen runtime registry")
    runtime_ref = generation_spec.get("runtime_registry")
    if not isinstance(runtime_ref, Mapping):
        raise ValueError("generation spec has no runtime registry reference")
    runtime_path = protocol.resolve_path(project_root, str(runtime_ref.get("path", "")))
    runtime_sha = str(runtime_ref.get("sha256", ""))
    protocol.validate_runtime_registry(runtime_path, runtime_sha)
    builder = project_root / protocol.TRAINING_CODE_ARTIFACTS[
        "runtime_registry_builder"
    ]
    if not builder.is_file() or builder.is_symlink():
        raise FileNotFoundError("runtime registry validator script is missing")
    execute = subprocess.run if run is None else run
    execute(
        [
            python,
            str(builder.relative_to(project_root)),
            "validate",
            "--output",
            str(runtime_ref["path"]),
            "--expected-sha256",
            runtime_sha,
        ],
        cwd=project_root,
        check=True,
    )


def _reserve_output(path: Path, registration: Mapping[str, Any]) -> None:
    protocol.reject_sealed_final36_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to reuse, resume, or race on output path: {path}") from exc
    (path / ".run_reservation_v2.json").write_text(
        json.dumps(dict(registration), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _atomic_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON file atomically without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def authorize_causal_stage0(args: argparse.Namespace) -> int:
    """Independent seed/model auditor entry point; does not touch screening media."""

    project_root = Path.cwd()
    expected_output = protocol.resolve_path(project_root, protocol.CAUSAL_STAGE0)
    if args.stage0_output.resolve() != expected_output.resolve():
        raise ValueError("authorizer output path differs from the standard causal Stage-0 path")
    if not args.private_root.is_dir() or args.private_root.is_symlink():
        raise ValueError("authorizer private root must be a real directory")
    try:
        args.selection_binding_output.resolve().relative_to(
            args.private_root.resolve(strict=True)
        )
    except ValueError as exc:
        raise ValueError("authorizer selection binding output escapes private root") from exc
    if args.stage0_output.exists() or args.selection_binding_output.exists():
        raise FileExistsError("refusing to overwrite Stage-0 wrapper or selection binding")
    expected_binding = protocol.prepare_selection_binding(
        project_root,
        dataset="causal",
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seed_inventory,
        source_ontology_path=args.source_ontology,
        source_split_path=args.source_split,
        holdout_registry_path=args.holdout_registry,
        receiver_ontology_path=args.receiver_ontology,
    )
    model_inventory = protocol.model_artifact_inventory(
        project_root, protocol.GENERATION_SPEC["model"]
    )
    if model_inventory["sha256"] != expected_binding["downstream_artifacts"][
        "model_inventory_sha256"
    ]:
        raise ValueError("authorizer model path-plus-file-bytes inventory differs from spec")
    _atomic_exclusive_json(args.selection_binding_output, expected_binding)

    def record(path: Path, row_count: int | None = None) -> dict[str, Any]:
        return {
            "sha256": protocol.file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": row_count,
        }

    rules = record(args.selection_rules)
    public_source_bank = protocol.resolve_path(project_root, protocol.PUBLIC_SOURCE_BANK)
    artifacts = {
        "candidate_manifest_48": record(args.candidate_manifest, 48),
        "source_bank_registry_64": record(public_source_bank, 64),
        "source_ontology_80": record(args.source_ontology, 80),
        "source_split_80": record(args.source_split, 80),
        "holdout_registry_24": record(args.holdout_registry, 24),
        "receiver_ontology_32": record(args.receiver_ontology, 32),
        "canonical_templates": record(args.canonical_templates),
        "field_normalization": record(args.field_normalization),
        "raw_root_bundle": record(args.root_bundle),
        "raw_render_configuration": record(args.render_configuration),
        "stage0_secrets": record(args.stage0_secrets),
        "screening_seed": record(args.screening_seed_file),
        "screening_generation_spec": record(args.generation_spec),
        "selector_salt": record(args.selector_salt_file),
        "ranking_formula": rules,
        "constrained_subset_algorithm": dict(rules),
        "evaluation_seed_salt": record(args.evaluation_seed_salt_file),
        "seed_derivation_formula": record(args.selection_binding_output),
        "forbidden_seed_inventory": record(args.forbidden_seed_inventory),
    }
    payload = {
        "protocol": protocol.COMMITMENT_PROTOCOL,
        "dataset": "causal",
        "dataset_version": protocol.DATASET_VERSION,
        "stage": 0,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "artifacts": artifacts,
    }
    _atomic_exclusive_json(args.stage0_output, payload)
    registry = protocol.validate_commitment_registry(
        args.stage0_output, dataset="causal", stage=0
    )
    protocol.validate_selection_contract_opening(
        project_root,
        dataset="causal",
        stage0_registry=registry,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seed_inventory,
        selection_binding_path=args.selection_binding_output,
        source_ontology_path=args.source_ontology,
        source_split_path=args.source_split,
        holdout_registry_path=args.holdout_registry,
        receiver_ontology_path=args.receiver_ontology,
    )
    return 0


def authorize_specificity_stage0(args: argparse.Namespace) -> int:
    """Authorize the specificity Stage-0 only from an existing causal Stage-1."""

    project_root = Path.cwd()
    expected_output = protocol.resolve_path(project_root, protocol.SPECIFICITY_STAGE0)
    if args.stage0_output.resolve() != expected_output.resolve():
        raise ValueError("authorizer output path differs from standard specificity Stage-0")
    if not args.private_root.is_dir() or args.private_root.is_symlink():
        raise ValueError("authorizer private root must be a real directory")
    try:
        args.selection_binding_output.resolve().relative_to(
            args.private_root.resolve(strict=True)
        )
    except ValueError as exc:
        raise ValueError("specificity selection binding output escapes private root") from exc
    if args.stage0_output.exists() or args.selection_binding_output.exists():
        raise FileExistsError("refusing to overwrite specificity Stage-0/binding")
    expected_binding = protocol.prepare_selection_binding(
        project_root,
        dataset="specificity",
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        new_bank_assignment_path=args.new_bank_assignment,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seed_inventory,
        causal_stage0_registry_path=args.causal_stage0_registry,
        causal_stage1_registry_path=args.causal_stage1_registry,
        causal_selected_path=args.causal_selected,
        causal_unit_manifest_path=args.causal_unit_manifest,
    )
    model_inventory = protocol.model_artifact_inventory(
        project_root, protocol.GENERATION_SPEC["model"]
    )
    if model_inventory["sha256"] != expected_binding["downstream_artifacts"][
        "model_inventory_sha256"
    ]:
        raise ValueError("specificity authorizer model inventory differs from spec")
    _atomic_exclusive_json(args.selection_binding_output, expected_binding)

    def record(path: Path, row_count: int | None = None) -> dict[str, Any]:
        return {
            "sha256": protocol.file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": row_count,
        }

    rules = record(args.selection_rules)
    artifacts = {
        "candidate_manifest_36": record(args.candidate_manifest, 36),
        "new_bank_selection_and_receiver_assignment": record(
            args.new_bank_assignment, 12
        ),
        "canonical_templates": record(args.canonical_templates),
        "field_normalization": record(args.field_normalization),
        "raw_root_bundle": record(args.root_bundle),
        "raw_render_configuration": record(args.render_configuration),
        "stage0_secrets": record(args.stage0_secrets),
        "screening_seed": record(args.screening_seed_file),
        "screening_generation_spec": record(args.generation_spec),
        "selector_salt": record(args.selector_salt_file),
        "ranking_formula": rules,
        "constrained_subset_algorithm": dict(rules),
        "evaluation_seed_salt": record(args.evaluation_seed_salt_file),
        "seed_derivation_formula": record(args.selection_binding_output),
        "forbidden_seed_inventory": record(args.forbidden_seed_inventory),
    }
    payload = {
        "protocol": protocol.COMMITMENT_PROTOCOL,
        "dataset": "specificity",
        "dataset_version": protocol.DATASET_VERSION,
        "stage": 0,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "artifacts": artifacts,
    }
    _atomic_exclusive_json(args.stage0_output, payload)
    registry = protocol.validate_commitment_registry(
        args.stage0_output, dataset="specificity", stage=0
    )
    protocol.validate_selection_contract_opening(
        project_root,
        dataset="specificity",
        stage0_registry=registry,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        new_bank_assignment_path=args.new_bank_assignment,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seed_inventory,
        selection_binding_path=args.selection_binding_output,
        causal_stage0_registry_path=args.causal_stage0_registry,
        causal_stage1_registry_path=args.causal_stage1_registry,
        causal_selected_path=args.causal_selected,
        causal_unit_manifest_path=args.causal_unit_manifest,
    )
    return 0


def _write_prompt_file(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = output_dir / "prompts.txt"
    lines = []
    for row in rows:
        prompt = str(row.get("prompt", ""))
        object_phrase = str(row.get("source_phrase", ""))
        if not prompt or not object_phrase or any("\n" in value or " | " in value for value in (prompt, object_phrase)):
            raise ValueError("prompt/object fields must be nonempty single-line values without pipe delimiters")
        lines.append(f"{prompt} | {object_phrase} | registered v4 evaluation")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _generation_command(
    *,
    python: str,
    prompt_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
    checkpoint: Path | None,
) -> list[str]:
    command = [
        python,
        GENERATOR,
        "--baseline",
        "clean",
        "--prompts",
        str(prompt_path),
        "--output-dir",
        str(output_dir),
        "--model",
        MODEL,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--steps",
        "25",
        "--guidance-scale",
        "5",
        "--num-frames",
        "49",
        "--fps",
        "8",
        "--height",
        "480",
        "--width",
        "832",
        "--dtype",
        "bf16",
        "--device",
        "cuda",
        "--vae-slicing",
        "--vae-tiling",
    ]
    if checkpoint is not None:
        command.extend(("--lora-path", str(checkpoint), "--lora-scale", "1.25"))
    return command


def _validate_raw_generation(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    checkpoint: Path | None,
) -> tuple[dict[str, Any], list[Path]]:
    raw_path = output_dir / "generation_manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        raw.get("dry_run") is not False
        or raw.get("baseline") != "clean"
        or raw.get("pipeline") != "WanPipeline"
        or raw.get("model") != MODEL
        or raw.get("prompts") != str(output_dir / "prompts.txt")
    ):
        raise ValueError("generator did not execute the frozen clean Wan pipeline")
    generation = raw.get("generation")
    expected = {
        "baseline": "clean",
        "seed": 42,
        "seeds": list(seeds),
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
        "lora_scale": 1.25 if checkpoint is not None else 1.0,
        "lora_path": str(checkpoint) if checkpoint is not None else None,
        "lora_sha256": protocol.artifact_sha256(checkpoint) if checkpoint is not None else None,
        "activation_gate_dir": None,
        "persistent_activation_gate": False,
        "lora_target_phrases": [],
        "attention_gate_dir": None,
        "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
    }
    if generation != expected:
        raise ValueError("raw generator configuration differs from the frozen v4 contract")
    items = raw.get("items")
    if not isinstance(items, list) or len(items) != len(rows):
        raise ValueError("raw generator item count differs from committed inventory")
    videos: list[Path] = []
    for index, (row, seed, item) in enumerate(zip(rows, seeds, items)):
        if (
            item.get("index") != index
            or item.get("prompt") != str(row["prompt"])
            or item.get("target_concept") != str(row["source_phrase"])
            or item.get("expected_effect") != "registered v4 evaluation"
            or item.get("seed") != seed
            or set(item) != {
                "index",
                "prompt",
                "target_concept",
                "expected_effect",
                "seed",
                "video_path",
            }
        ):
            raise ValueError("raw generator prompt/seed/order mismatch")
        video = Path(str(item.get("video_path", "")))
        if not video.is_file() or video.is_symlink() or video.parent.resolve() != (output_dir / "videos").resolve():
            raise ValueError("raw generator video path escapes the reserved run")
        videos.append(video)
    if len({path.resolve() for path in videos}) != len(videos):
        raise ValueError("raw generator reuses a video path")
    expected_root = {
        ".run_reservation_v2.json",
        "prompts.txt",
        "generation_manifest.json",
        "videos",
    }
    if {path.name for path in output_dir.iterdir()} != expected_root:
        raise ValueError("raw run directory contains an unexpected file")
    if set((output_dir / "videos").iterdir()) != set(videos):
        raise ValueError("raw videos directory contains an unexpected or missing file")
    return raw, videos


def _media_record(index: int, unit_id: str, row: Mapping[str, Any], path: Path) -> dict[str, Any]:
    decoded = protocol._default_decode(path)
    expected = {
        "frame_count": protocol.FRAME_COUNT,
        "width": protocol.WIDTH,
        "height": protocol.HEIGHT,
        "fps_numerator": protocol.FPS.numerator,
        "fps_denominator": protocol.FPS.denominator,
    }
    if decoded != expected:
        raise ValueError(f"generated video fails exact decode contract: {path}")
    return {
        "unit_id": unit_id,
        "index": index,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": protocol.file_sha256(path),
        "prompt_sha256": hashlib.sha256(str(row["prompt"]).encode("utf-8")).hexdigest(),
        "seed": int(row["seed"]),
        **expected,
    }


def _model_inventory(project_root: Path, expected_sha256: str) -> dict[str, Any]:
    if not protocol.is_sha256(expected_sha256):
        raise ValueError("registered model inventory hash is not a SHA-256")
    inventory = protocol.model_artifact_inventory(project_root, MODEL)
    if inventory["sha256"] != expected_sha256:
        raise ValueError("current Wan model bytes differ from registered generation inventory")
    return inventory


def _bind_units_to_selected(
    units: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> None:
    case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    selected_by_id = {str(row[case_field]): row for row in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("selected manifest contains duplicate case IDs")
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in units:
        by_case.setdefault(str(row[case_field]), []).append(row)
    if set(by_case) != set(selected_by_id):
        raise ValueError("unit and selected manifests contain different case IDs")
    invariant_fields = (
        (
            "group", "prompt_variant", "source_id", "source_phrase",
            "source_physical_audit_status", "receiver_id", "receiver", "prompt",
        )
        if dataset == "causal"
        else (
            "membership",
            "prompt_variant",
            "source_id",
            "source_phrase",
            "receiver_id",
            "receiver",
            "prompt",
            "causal_case_id",
        )
    )
    for case_id, case_units in by_case.items():
        selected_row = selected_by_id[case_id]
        for unit in case_units:
            for field in invariant_fields:
                if str(unit.get(field, "")) != str(selected_row.get(field, "")):
                    raise ValueError(f"{case_id}: selected/unit manifest mismatch for {field}")


def _validate_specificity_candidate_pool(
    rows: Sequence[Mapping[str, Any]], causal_selected: Sequence[Mapping[str, Any]]
) -> None:
    causal = {str(row["semantic_case_id"]): row for row in causal_selected}
    if len(causal) != 24:
        raise ValueError("specificity Stage-0 requires the exact selected24 causal manifest")
    matched = [
        row
        for row in rows
        if str(row["membership"]) in {"original_source", "holdout_source"}
    ]
    if len(matched) != 24 or {
        str(row.get("causal_case_id", "")) for row in matched
    } != set(causal):
        raise ValueError("specificity matched pool must contain every original/holdout causal case once")
    for row in matched:
        source = causal[str(row["causal_case_id"])]
        expected_membership = (
            "original_source"
            if str(source["group"]) == "seen_source_new_receiver"
            else "holdout_source"
        )
        if str(row["membership"]) != expected_membership:
            raise ValueError("specificity matched source membership differs from causal case")
        for field in (
            "source_id",
            "source_phrase",
            "source_head_lemma",
            "receiver_id",
            "receiver",
        ):
            if str(row[field]) != str(source[field]):
                raise ValueError(f"specificity matched pool differs from causal case: {field}")
    new_bank = [row for row in rows if str(row["membership"]) == "new_bank_source"]
    if (
        len(new_bank) != 12
        or len({str(row["source_id"]) for row in new_bank}) != 12
        or len({str(row["receiver_id"]) for row in new_bank}) != 12
        or not {str(row["receiver_id"]) for row in new_bank}
        <= {str(row["receiver_id"]) for row in causal_selected}
        or any(str(row.get("causal_case_id", "")) for row in new_bank)
    ):
        raise ValueError("new-bank specificity pool assignment differs from protocol")


def run_screening(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    for label, path in (
        ("screening raw run", args.output_dir),
        ("screening public package", args.screening_public_dir),
        ("screening private package", args.screening_private_dir),
    ):
        _require_future_output_inside_private_root(path, args.private_root, label)
    if args.stage0_registry.resolve() != protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE0 if args.dataset == "causal" else protocol.SPECIFICITY_STAGE0,
    ).resolve():
        raise ValueError("screening registry path differs from the frozen protocol")
    registry = protocol.validate_commitment_registry(args.stage0_registry, dataset=args.dataset, stage=0)
    if protocol.file_sha256(args.stage0_registry) != args.stage0_registry_sha256:
        raise ValueError("screening Stage-0 public registry hash mismatch")
    protocol.validate_selection_contract_opening(
        project_root,
        dataset=args.dataset,
        stage0_registry=registry,
        private_root=args.private_root,
        candidate_manifest_path=args.candidate_manifest,
        canonical_templates_path=args.canonical_templates,
        field_rules_path=args.field_normalization,
        render_configuration_path=args.render_configuration,
        selection_rules_path=args.selection_rules,
        secrets_path=args.stage0_secrets,
        root_bundle_path=args.root_bundle,
        generation_spec_path=args.generation_spec,
        screening_seed_path=args.screening_seed_file,
        selector_salt_path=args.selector_salt_file,
        evaluation_seed_salt_path=args.evaluation_seed_salt_file,
        forbidden_seed_inventory_path=args.forbidden_seed_inventory,
        selection_binding_path=args.selection_binding,
        source_ontology_path=args.source_ontology,
        source_split_path=args.source_split,
        holdout_registry_path=args.holdout_registry,
        receiver_ontology_path=args.receiver_ontology,
        new_bank_assignment_path=args.new_bank_assignment,
        causal_stage0_registry_path=args.causal_stage0_registry,
        causal_stage1_registry_path=args.causal_stage1_registry,
        causal_selected_path=args.causal_selected,
        causal_unit_manifest_path=args.causal_unit_manifest,
    )
    _require_inside_private_root(args.candidate_manifest, args.private_root, "candidate manifest")
    _require_inside_private_root(args.screening_seed_file, args.private_root, "screening seed")
    _require_inside_private_root(args.canonical_templates, args.private_root, "canonical templates")
    _require_inside_private_root(args.field_normalization, args.private_root, "field normalization")
    _require_inside_private_root(args.forbidden_seed_inventory, args.private_root, "forbidden seeds")
    if protocol.file_sha256(args.candidate_manifest) != registry["artifacts"][f"candidate_manifest_{protocol.CANDIDATE_COUNTS[args.dataset]}"]["sha256"]:
        raise ValueError("candidate manifest bytes differ from Stage-0 commitment")
    if protocol.file_sha256(args.screening_seed_file) != registry["artifacts"]["screening_seed"]["sha256"]:
        raise ValueError("screening seed bytes differ from Stage-0 commitment")
    for name, path in (
        ("canonical_templates", args.canonical_templates),
        ("field_normalization", args.field_normalization),
        ("forbidden_seed_inventory", args.forbidden_seed_inventory),
    ):
        if protocol.file_sha256(path) != registry["artifacts"][name]["sha256"]:
            raise ValueError(f"{name} bytes differ from Stage-0 commitment")
    generation_spec = _validate_generation_spec(
        args.generation_spec,
        committed_sha256=registry["artifacts"]["screening_generation_spec"]["sha256"],
        private_root=args.private_root,
    )
    rows = protocol.load_normalized_candidate_manifest(
        args.candidate_manifest,
        dataset=args.dataset,
        canonical_templates_path=args.canonical_templates,
    )
    if len(rows) != protocol.CANDIDATE_COUNTS[args.dataset]:
        raise ValueError("screening candidate count mismatch")
    if len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise ValueError("screening candidate IDs must be unique")
    if args.dataset == "causal":
        cells = Counter(
            (str(row["group"]), str(row["prompt_variant"])) for row in rows
        )
        expected_cells = {
            (group, variant): 8
            for group in protocol.CAUSAL_GROUPS
            for variant in protocol.PROMPT_VARIANTS
        }
    else:
        cells = Counter(
            (str(row["membership"]), str(row["prompt_variant"])) for row in rows
        )
        expected_cells = {
            ("original_source", "direct"): 4,
            ("original_source", "natural"): 4,
            ("new_bank_source", "direct"): 6,
            ("new_bank_source", "natural"): 6,
            ("holdout_source", "direct"): 8,
            ("holdout_source", "natural"): 8,
        }
    if dict(cells) != expected_cells:
        raise ValueError("screening candidate cell inventory differs from protocol")
    if args.dataset == "specificity":
        required = (
            args.causal_stage0_registry,
            args.causal_stage1_registry,
            args.causal_selected,
            args.causal_unit_manifest,
        )
        if any(value is None for value in required):
            raise ValueError(
                "specificity screening requires causal Stage0/1, selected24, and U72"
            )
        if (
            args.causal_stage0_registry.resolve()
            != protocol.resolve_path(project_root, protocol.CAUSAL_STAGE0).resolve()
            or args.causal_stage1_registry.resolve()
            != protocol.resolve_path(project_root, protocol.CAUSAL_STAGE1).resolve()
        ):
            raise ValueError("specificity screening causal registry paths differ from protocol")
        causal0_sha = protocol.file_sha256(args.causal_stage0_registry)
        protocol.validate_commitment_registry(
            args.causal_stage0_registry, dataset="causal", stage=0
        )
        causal1 = protocol.validate_commitment_registry(
            args.causal_stage1_registry,
            dataset="causal",
            stage=1,
            expected_stage0_sha256=causal0_sha,
        )
        _require_inside_private_root(
            args.causal_selected, args.private_root, "causal selected manifest"
        )
        if protocol.file_sha256(args.causal_selected) != causal1["artifacts"]["selected_case_manifest_24"]["sha256"]:
            raise ValueError("causal selected24 differs from causal Stage-1 commitment")
        causal_selected = _load_rows(args.causal_selected)
        protocol.validate_causal_selected_cases(causal_selected)
        _validate_specificity_candidate_pool(rows, causal_selected)
    seed = _read_single_seed(args.screening_seed_file)
    if seed in _read_seed_inventory(args.forbidden_seed_inventory, dataset=args.dataset):
        raise ValueError("screening seed collides with the committed forbidden-seed inventory")
    normalized_candidates = [dict(row) for row in rows]
    for index, row in enumerate(rows):
        row["seed"] = seed
        row.setdefault("unit_id", f"screen_{args.dataset[0]}_{index:03d}")
    _model_inventory(project_root, generation_spec["model_inventory_sha256"])
    _validate_generation_runtime(project_root, generation_spec, args.python)
    reservation = {
        "protocol": SCREENING_GENERATION_PROTOCOL,
        "dataset": args.dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "method": "original",
        "stage0_registry_sha256": args.stage0_registry_sha256,
        "candidate_manifest_sha256": protocol.file_sha256(args.candidate_manifest),
        "screening_seed_sha256": protocol.file_sha256(args.screening_seed_file),
        "model_inventory_sha256": generation_spec["model_inventory_sha256"],
        "runtime_registry_sha256": generation_spec["runtime_registry"]["sha256"],
    }
    _reserve_output(args.output_dir, reservation)
    prompt_path = _write_prompt_file(args.output_dir, rows)
    command = _generation_command(
        python=args.python,
        prompt_path=prompt_path,
        output_dir=args.output_dir,
        seeds=[seed] * len(rows),
        checkpoint=None,
    )
    subprocess.run(command, cwd=project_root, check=True)
    _, videos = _validate_raw_generation(
        args.output_dir, rows, [seed] * len(rows), checkpoint=None
    )
    records = [
        _media_record(index, str(row["unit_id"]), row, path)
        for index, (row, path) in enumerate(zip(rows, videos))
    ]
    manifest = {
        **reservation,
        "raw_generation_manifest": {
            "path": str(args.output_dir / "generation_manifest.json"),
            "sha256": protocol.file_sha256(
                args.output_dir / "generation_manifest.json"
            ),
        },
        "generation_spec": protocol.GENERATION_SPEC,
        "videos": records,
    }
    manifest_path = args.output_dir / "v4_screening_generation_manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_screening_review_package(
        project_root=project_root,
        dataset=args.dataset,
        normalized_candidates=normalized_candidates,
        candidate_manifest_path=args.candidate_manifest,
        screening_seed_path=args.screening_seed_file,
        generation_spec_path=args.generation_spec,
        stage0_registry_path=args.stage0_registry,
        generation_manifest_path=manifest_path,
        public_dir=args.screening_public_dir,
        private_dir=args.screening_private_dir,
    )
    expected_root = {
        ".run_reservation_v2.json",
        "prompts.txt",
        "generation_manifest.json",
        "videos",
        "v4_screening_generation_manifest_v2.json",
    }
    if {path.name for path in args.output_dir.iterdir()} != expected_root:
        raise ValueError("completed screening run inventory is not exact")
    return 0


def run_final_generation(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    _require_future_output_inside_private_root(
        args.output_dir, args.private_root, "final generation"
    )
    protocol.validate_training_authorization(
        project_root,
        expected_gate_spec=protocol.GATE_SPEC,
        authorization_path=args.training_authorization,
    )
    eligibility = protocol.validate_checkpoint_eligibility(project_root, args.checkpoint_eligibility)
    stage0_sha = protocol.file_sha256(args.stage0_registry)
    expected_stage0 = protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE0 if args.dataset == "causal" else protocol.SPECIFICITY_STAGE0,
    )
    expected_stage1 = protocol.resolve_path(
        project_root,
        protocol.CAUSAL_STAGE1 if args.dataset == "causal" else protocol.SPECIFICITY_STAGE1,
    )
    if args.stage0_registry.resolve() != expected_stage0.resolve() or args.stage1_registry.resolve() != expected_stage1.resolve():
        raise ValueError("generation Stage-0/Stage-1 paths differ from the frozen protocol")
    protocol.validate_commitment_registry(args.stage0_registry, dataset=args.dataset, stage=0)
    stage1 = protocol.validate_commitment_registry(
        args.stage1_registry,
        dataset=args.dataset,
        stage=1,
        expected_stage0_sha256=stage0_sha,
    )
    generation_spec = _validate_generation_spec(
        args.generation_spec,
        committed_sha256=protocol.validate_commitment_registry(
            args.stage0_registry, dataset=args.dataset, stage=0
        )["artifacts"]["screening_generation_spec"]["sha256"],
        private_root=args.private_root,
    )
    if generation_spec["model_inventory_sha256"] != eligibility[
        "model_content_inventory_sha256"
    ]:
        raise ValueError("generation full-model inventory differs from eligible checkpoint")
    _require_inside_private_root(args.forbidden_seed_inventory, args.private_root, "forbidden seeds")
    stage0_registry = protocol.validate_commitment_registry(
        args.stage0_registry, dataset=args.dataset, stage=0
    )
    if protocol.file_sha256(args.forbidden_seed_inventory) != stage0_registry["artifacts"]["forbidden_seed_inventory"]["sha256"]:
        raise ValueError("forbidden seed inventory bytes differ from Stage-0 commitment")
    _require_inside_private_root(args.unit_manifest, args.private_root, "unit manifest")
    artifact_name = "unit_manifest_U_72" if args.dataset == "causal" else "unit_manifest_W_36"
    if protocol.file_sha256(args.unit_manifest) != stage1["artifacts"][artifact_name]["sha256"]:
        raise ValueError("unit manifest bytes differ from Stage-1 commitment")
    rows = _load_rows(args.unit_manifest)
    forbidden_seeds = _read_seed_inventory(
        args.forbidden_seed_inventory, dataset=args.dataset
    )
    _require_inside_private_root(args.selected_manifest, args.private_root, "selected case manifest")
    selected_artifact = (
        "selected_case_manifest_24"
        if args.dataset == "causal"
        else "selected_case_manifest_18"
    )
    if protocol.file_sha256(args.selected_manifest) != stage1["artifacts"][selected_artifact]["sha256"]:
        raise ValueError("selected-case manifest bytes differ from Stage-1 commitment")
    selected = _load_rows(args.selected_manifest)
    if args.dataset == "causal":
        protocol.validate_causal_selected_cases(selected)
        protocol.validate_causal_unit_manifest(rows)
        _bind_units_to_selected(rows, selected, dataset="causal")
        if {int(row["seed"]) for row in rows} & forbidden_seeds:
            raise ValueError("U collides with the committed forbidden-seed inventory")
    else:
        required_causal = (
            args.causal_stage0_registry,
            args.causal_stage1_registry,
            args.causal_selected,
            args.causal_unit_manifest,
            args.holdout_mapping,
        )
        if any(value is None for value in required_causal):
            raise ValueError(
                "specificity generation requires causal Stage0/1, selected24, U72, and M"
            )
        expected_causal_stage0 = protocol.resolve_path(project_root, protocol.CAUSAL_STAGE0)
        expected_causal_stage1 = protocol.resolve_path(project_root, protocol.CAUSAL_STAGE1)
        if (
            args.causal_stage0_registry.resolve() != expected_causal_stage0.resolve()
            or args.causal_stage1_registry.resolve() != expected_causal_stage1.resolve()
        ):
            raise ValueError("specificity causal registry paths differ from protocol")
        causal_stage0_sha = protocol.file_sha256(args.causal_stage0_registry)
        protocol.validate_commitment_registry(
            args.causal_stage0_registry, dataset="causal", stage=0
        )
        causal_stage1 = protocol.validate_commitment_registry(
            args.causal_stage1_registry,
            dataset="causal",
            stage=1,
            expected_stage0_sha256=causal_stage0_sha,
        )
        for label, path, artifact_name_causal in (
            ("causal selected manifest", args.causal_selected, "selected_case_manifest_24"),
            ("causal U manifest", args.causal_unit_manifest, "unit_manifest_U_72"),
        ):
            _require_inside_private_root(path, args.private_root, label)
            if protocol.file_sha256(path) != causal_stage1["artifacts"][artifact_name_causal]["sha256"]:
                raise ValueError(f"{label} bytes differ from causal Stage-1 commitment")
        _require_inside_private_root(args.holdout_mapping, args.private_root, "holdout mapping M")
        if protocol.file_sha256(args.holdout_mapping) != stage1["artifacts"]["holdout_mapping_M_6"]["sha256"]:
            raise ValueError("M bytes differ from specificity Stage-1 commitment")
        causal_selected = _load_rows(args.causal_selected)
        causal_units = _load_rows(args.causal_unit_manifest)
        protocol.validate_causal_selected_cases(causal_selected)
        protocol.validate_causal_unit_manifest(causal_units)
        protocol.validate_specificity_selected_cases(
            selected, causal_cases=causal_selected
        )
        protocol.validate_specificity_unit_manifest(
            rows,
            causal_cases=causal_selected,
            causal_seeds=[int(row["seed"]) for row in causal_units],
        )
        _bind_units_to_selected(rows, selected, dataset="specificity")
        if {int(row["seed"]) for row in rows} & forbidden_seeds:
            raise ValueError("W collides with the committed forbidden-seed inventory")
        protocol.validate_holdout_mapping(
            _load_rows(args.holdout_mapping),
            causal_cases=causal_selected,
            specificity_cases=selected,
        )
    _model_inventory(project_root, generation_spec["model_inventory_sha256"])
    _validate_generation_runtime(project_root, generation_spec, args.python)
    if args.method == "original":
        checkpoint = None
        method_artifact: dict[str, Any] = {"kind": "base_model"}
    elif args.method == "v3b":
        checkpoint = Path(protocol.V3B_CHECKPOINT)
        if protocol.artifact_sha256(checkpoint) != protocol.V3B_CHECKPOINT_SHA256:
            raise ValueError("frozen v3b checkpoint bytes changed")
        method_artifact = {
            "kind": "lora_checkpoint",
            "path": protocol.V3B_CHECKPOINT,
            "sha256": protocol.V3B_CHECKPOINT_SHA256,
            "scale": 1.25,
            "step": 200,
        }
    else:
        checkpoint = Path(str(eligibility["checkpoint"]["path"]))
        method_artifact = {
            "kind": "lora_checkpoint",
            "checkpoint_eligibility_path": str(args.checkpoint_eligibility),
            "checkpoint_eligibility_sha256": protocol.file_sha256(args.checkpoint_eligibility),
            "path": eligibility["checkpoint"]["path"],
            "weights_sha256": eligibility["checkpoint"]["weights_sha256"],
            "scale": 1.25,
            "step": 200,
        }
    reservation = {
        "protocol": protocol.GENERATION_MANIFEST_PROTOCOL,
        "dataset": args.dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "method": args.method,
        "unit_manifest_canonical_sha256": protocol.canonical_json_sha256(rows),
        "generation_spec": protocol.GENERATION_SPEC,
        "model_inventory_sha256": generation_spec["model_inventory_sha256"],
        "runtime_registry_sha256": generation_spec["runtime_registry"]["sha256"],
        "method_artifact": method_artifact,
    }
    _reserve_output(args.output_dir, reservation)
    prompt_path = _write_prompt_file(args.output_dir, rows)
    subprocess.run(
        _generation_command(
            python=args.python,
            prompt_path=prompt_path,
            output_dir=args.output_dir,
            seeds=[int(row["seed"]) for row in rows],
            checkpoint=checkpoint,
        ),
        cwd=project_root,
        check=True,
    )
    _, videos = _validate_raw_generation(
        args.output_dir,
        rows,
        [int(row["seed"]) for row in rows],
        checkpoint=checkpoint,
    )
    records = [
        _media_record(index, str(row["unit_id"]), row, path)
        for index, (row, path) in enumerate(zip(rows, videos))
    ]
    manifest = {**reservation, "videos": records}
    path = args.output_dir / "v4_generation_manifest_v2.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if {item.name for item in args.output_dir.iterdir()} != {
        ".run_reservation_v2.json",
        "prompts.txt",
        "generation_manifest.json",
        "videos",
        "v4_generation_manifest_v2.json",
    }:
        raise ValueError("completed final generation run inventory is not exact")
    return 0


def _common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--stage0-registry", type=Path, required=True)
    parser.add_argument("--generation-spec", type=Path, required=True)
    parser.add_argument("--forbidden-seed-inventory", type=Path, required=True)
    parser.add_argument("--python", default="models/.wan-runtime/bin/python")
    parser.add_argument("--output-dir", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    authorize = sub.add_parser(
        "authorize-causal-stage0",
        help="independently freeze the exact binding and standard Stage-0 wrapper",
    )
    authorize.add_argument("--private-root", type=Path, required=True)
    authorize.add_argument("--candidate-manifest", type=Path, required=True)
    authorize.add_argument("--source-ontology", type=Path, required=True)
    authorize.add_argument("--source-split", type=Path, required=True)
    authorize.add_argument("--holdout-registry", type=Path, required=True)
    authorize.add_argument("--receiver-ontology", type=Path, required=True)
    authorize.add_argument("--canonical-templates", type=Path, required=True)
    authorize.add_argument("--field-normalization", type=Path, required=True)
    authorize.add_argument("--render-configuration", type=Path, required=True)
    authorize.add_argument("--selection-rules", type=Path, required=True)
    authorize.add_argument("--stage0-secrets", type=Path, required=True)
    authorize.add_argument("--root-bundle", type=Path, required=True)
    authorize.add_argument("--generation-spec", type=Path, required=True)
    authorize.add_argument("--screening-seed-file", type=Path, required=True)
    authorize.add_argument("--selector-salt-file", type=Path, required=True)
    authorize.add_argument("--evaluation-seed-salt-file", type=Path, required=True)
    authorize.add_argument("--forbidden-seed-inventory", type=Path, required=True)
    authorize.add_argument("--selection-binding-output", type=Path, required=True)
    authorize.add_argument(
        "--stage0-output", type=Path, default=Path(protocol.CAUSAL_STAGE0)
    )
    authorize.set_defaults(func=authorize_causal_stage0)
    authorize_specificity = sub.add_parser(
        "authorize-specificity-stage0",
        help="freeze specificity Stage-0 after the causal Stage-1 chain exists",
    )
    authorize_specificity.add_argument("--private-root", type=Path, required=True)
    authorize_specificity.add_argument("--candidate-manifest", type=Path, required=True)
    authorize_specificity.add_argument("--new-bank-assignment", type=Path, required=True)
    authorize_specificity.add_argument("--canonical-templates", type=Path, required=True)
    authorize_specificity.add_argument("--field-normalization", type=Path, required=True)
    authorize_specificity.add_argument("--render-configuration", type=Path, required=True)
    authorize_specificity.add_argument("--selection-rules", type=Path, required=True)
    authorize_specificity.add_argument("--stage0-secrets", type=Path, required=True)
    authorize_specificity.add_argument("--root-bundle", type=Path, required=True)
    authorize_specificity.add_argument("--generation-spec", type=Path, required=True)
    authorize_specificity.add_argument("--screening-seed-file", type=Path, required=True)
    authorize_specificity.add_argument("--selector-salt-file", type=Path, required=True)
    authorize_specificity.add_argument("--evaluation-seed-salt-file", type=Path, required=True)
    authorize_specificity.add_argument("--forbidden-seed-inventory", type=Path, required=True)
    authorize_specificity.add_argument("--causal-stage0-registry", type=Path, required=True)
    authorize_specificity.add_argument("--causal-stage1-registry", type=Path, required=True)
    authorize_specificity.add_argument("--causal-selected", type=Path, required=True)
    authorize_specificity.add_argument("--causal-unit-manifest", type=Path, required=True)
    authorize_specificity.add_argument("--selection-binding-output", type=Path, required=True)
    authorize_specificity.add_argument(
        "--stage0-output", type=Path, default=Path(protocol.SPECIFICITY_STAGE0)
    )
    authorize_specificity.set_defaults(func=authorize_specificity_stage0)
    screen = sub.add_parser("screen-original", help="run committed Stage-0 Original screening only")
    _common_generation_args(screen)
    screen.add_argument("--stage0-registry-sha256", required=True)
    screen.add_argument("--candidate-manifest", type=Path, required=True)
    screen.add_argument("--screening-seed-file", type=Path, required=True)
    screen.add_argument("--canonical-templates", type=Path, required=True)
    screen.add_argument("--field-normalization", type=Path, required=True)
    screen.add_argument("--render-configuration", type=Path, required=True)
    screen.add_argument("--selection-rules", type=Path, required=True)
    screen.add_argument("--stage0-secrets", type=Path, required=True)
    screen.add_argument("--root-bundle", type=Path, required=True)
    screen.add_argument("--selector-salt-file", type=Path, required=True)
    screen.add_argument("--evaluation-seed-salt-file", type=Path, required=True)
    screen.add_argument("--selection-binding", type=Path, required=True)
    screen.add_argument("--new-bank-assignment", type=Path)
    screen.add_argument("--source-ontology", type=Path)
    screen.add_argument("--source-split", type=Path)
    screen.add_argument("--holdout-registry", type=Path)
    screen.add_argument("--receiver-ontology", type=Path)
    screen.add_argument("--screening-public-dir", type=Path, required=True)
    screen.add_argument("--screening-private-dir", type=Path, required=True)
    screen.add_argument("--causal-stage0-registry", type=Path)
    screen.add_argument("--causal-stage1-registry", type=Path)
    screen.add_argument("--causal-selected", type=Path)
    screen.add_argument("--causal-unit-manifest", type=Path)
    screen.set_defaults(func=run_screening)
    final = sub.add_parser("generate", help="generate one registered O/v3b/v4 arm for U or W")
    _common_generation_args(final)
    final.add_argument("--stage1-registry", type=Path, required=True)
    final.add_argument("--unit-manifest", type=Path, required=True)
    final.add_argument("--selected-manifest", type=Path, required=True)
    final.add_argument("--causal-stage0-registry", type=Path)
    final.add_argument("--causal-stage1-registry", type=Path)
    final.add_argument("--causal-selected", type=Path)
    final.add_argument("--causal-unit-manifest", type=Path)
    final.add_argument("--holdout-mapping", type=Path)
    final.add_argument("--method", choices=protocol.METHODS, required=True)
    final.add_argument("--training-authorization", type=Path, default=Path(protocol.TRAINING_AUTHORIZATION))
    final.add_argument("--checkpoint-eligibility", type=Path, default=Path(protocol.CHECKPOINT_ELIGIBILITY))
    final.set_defaults(func=run_final_generation)
    args = parser.parse_args()
    protocol.reject_sealed_final36_path(
        *[value for value in vars(args).values() if isinstance(value, Path)]
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
