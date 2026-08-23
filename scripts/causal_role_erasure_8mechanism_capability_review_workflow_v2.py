#!/usr/bin/env python3
"""Build, adjudicate, finalize, and score the frozen capability-v2 review.

The workflow adapts the audited v1 delivery and adjudication engine while
binding the final v2 manifest, blank template, rubric, and freeze.  The final
generation-manifest SHA-256 is intentionally not hard-coded: ``build``
requires the caller to supply it after generation, then verifies the complete
v2 launcher binding and all 192 video records before creating any delivery.

No command starts generation or review.  Every output is fresh, exclusive,
rollback-protected, and rejected when any path resolves into sealed/final36.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import causal_role_erasure_8mechanism_capability_review_v2 as _review_v2
import causal_role_erasure_8mechanism_capability_review_workflow_v1 as _v1


WORKFLOW_VERSION = "causal_role_erasure_8mechanism_capability_review_workflow_v2"
MANIFEST_PROTOCOL = "causal_role_erasure_8mechanism_capability_v2"
REVIEW_PROTOCOL = "causal_role_erasure_8mechanism_capability_review_v2"
EXPECTED_ROWS = 192

EXPECTED_CANONICAL_SHA256 = (
    "8e8cc14a6d327193cb7f270d2386490dc00ee783d9e7d0f76a697837f218dbee"
)
EXPECTED_PROMPTS_SHA256 = (
    "472eb889d200d9d26c5863f545281468632c0185b8f148f830dfb1589467e1d4"
)
EXPECTED_GENERATOR_SHA256 = (
    "04bc6b8a8f93d885137c6157509b00f46824231560989f7b7f78192b26469b5e"
)
EXPECTED_REVIEW_TEMPLATE_SHA256 = (
    "4f8f3853a25931145d86e663d70ded54da0d162ce2d17174e5c9df7957b2820b"
)
EXPECTED_REVIEW_RUBRIC_SHA256 = (
    "052232152cbf97a070f2479cbea2289955923d144fc48cc1420238080b2547ef"
)
EXPECTED_REVIEW_FREEZE_SHA256 = (
    "3c735ac020c63c62b15bab324258cd3d7a6a5bc457dd2da32793ea5bb29f92f0"
)

DEFAULT_CANONICAL = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_manifest.canonical.json"
)
DEFAULT_REVIEW_TEMPLATE = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_review_template.csv"
)
DEFAULT_REVIEW_RUBRIC = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_review_rubric.json"
)
DEFAULT_REVIEW_FREEZE = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_review_freeze.json"
)

PANEL_RANGES = _v1.PANEL_RANGES
BOOLEAN_FIELDS = _v1.BOOLEAN_FIELDS
ORDINAL_FIELDS = _v1.ORDINAL_FIELDS
SCORE_FIELDS = _v1.SCORE_FIELDS
FROZEN_REVIEW_FIELDS = _v1.FROZEN_REVIEW_FIELDS
ASSIGNMENT_CONTEXT_FIELDS = _v1.ASSIGNMENT_CONTEXT_FIELDS
ASSIGNMENT_FIELDS = _v1.ASSIGNMENT_FIELDS
BINDING_FIELDS = _v1.BINDING_FIELDS
ADJUDICATION_CONTEXT_FIELDS = _v1.ADJUDICATION_CONTEXT_FIELDS
ADJUDICATION_FIELDS = _v1.ADJUDICATION_FIELDS
ADJUDICATION_BINDING_FIELDS = _v1.ADJUDICATION_BINDING_FIELDS

GENERATION_FIELDS = {
    "baseline": "clean",
    "num_inference_steps": 25,
    "guidance_scale": 5.0,
    "num_frames": 49,
    "fps": 8,
    "height": 480,
    "width": 832,
    "dtype": "bf16",
    "device_inside_isolated_process": "cuda",
    "vae_slicing": True,
    "vae_tiling": True,
    "model_cpu_offload": False,
    "sequential_cpu_offload": False,
    "skip_existing": False,
    "per_prompt_seeds": "explicit_from_canonical_manifest",
    "post_generation_media_probe": "decode-count exact frame/fps/resolution validation",
}
GENERATION_MANIFEST_FIELDS = {
    "schema_version",
    "protocol_version",
    "status",
    "stage_registry_sha256",
    "canonical_manifest_sha256",
    "prompts_sha256",
    "generator_sha256",
    "generation",
    "mechanism_manifests",
    "video_binding_key",
    "video_count",
    "items",
}
GENERATION_ITEM_FIELDS = {
    "canonical_row_index",
    "generation_id",
    "mechanism",
    "seed",
    "video_path",
    "video_sha256",
    "size_bytes",
    "media",
}
MECHANISM_MANIFEST_FIELDS = {
    "created_at_utc",
    "baseline",
    "pipeline",
    "model",
    "dry_run",
    "prompts",
    "generation",
    "items",
}
MECHANISM_ITEM_FIELDS = {
    "index",
    "prompt",
    "target_concept",
    "expected_effect",
    "seed",
    "video_path",
}
MECHANISM_GENERATION_FIXED_FIELDS = {
    "baseline": "clean",
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
    "lora_path": None,
    "lora_sha256": None,
    "lora_scale": 1.0,
    "activation_gate_dir": None,
    "persistent_activation_gate": False,
    "lora_target_phrases": [],
    "attention_gate_dir": None,
    "attention_suppression_phrases": [],
    "attention_suppression_strength": 20.0,
}
MECHANISM_ORDER = tuple(_review_v2.MECHANISM_ORDER)

require = _v1.require
sha256_file = _v1.sha256_file
canonical_json_bytes = _v1.canonical_json_bytes
pretty_json_bytes = _v1.pretty_json_bytes
regular_file = _v1.regular_file
require_sha256 = _v1.require_sha256
file_ref = _v1.file_ref
relative_file_ref = _v1.relative_file_ref
read_json_object = _v1.read_json_object
read_json_list = _v1.read_json_list
read_csv_exact = _v1.read_csv_exact
csv_bytes = _v1.csv_bytes
write_bytes_exclusive = _v1.write_bytes_exclusive
prepare_partial_root = _v1.prepare_partial_root
_safe_cleanup_partial = _v1._safe_cleanup_partial
rollback_fresh_partials = _v1.rollback_fresh_partials
expose_partial_root = _v1.expose_partial_root
load_video_frames = _v1.load_video_frames
render_full49_composite = _v1.render_full49_composite
panel_ranges = _v1.panel_ranges
median_of_three = _v1.median_of_three


def reject_sealed_path(*paths: Path) -> None:
    for path in paths:
        text = path.as_posix().casefold()
        require(
            "sealed" not in text and "final36" not in text,
            "sealed/final36 paths are forbidden in capability-v2 review",
        )


def _existing_path_chain(path: Path) -> list[Path]:
    absolute = Path(os.path.abspath(path))
    chain = [absolute]
    chain.extend(absolute.parents)
    return list(reversed(chain))


def validate_path_ancestors_pre_io(path: Path, label: str) -> tuple[Path, Path]:
    """Reject sealed names and symlinks before any content read or mkdir."""

    reject_sealed_path(path)
    lexical = Path(os.path.abspath(path))
    reject_sealed_path(lexical)
    for candidate in _existing_path_chain(lexical):
        if not os.path.lexists(candidate):
            continue
        reject_sealed_path(candidate)
        info = os.lstat(candidate)
        require(
            not stat.S_ISLNK(info.st_mode),
            f"{label} has an existing symlink ancestor: {candidate}",
        )
    resolved = lexical.resolve(strict=False)
    reject_sealed_path(resolved)
    for candidate in _existing_path_chain(resolved):
        if not os.path.lexists(candidate):
            continue
        reject_sealed_path(candidate)
        info = os.lstat(candidate)
        require(
            not stat.S_ISLNK(info.st_mode),
            f"{label} resolved path has an existing symlink ancestor: {candidate}",
        )
    return lexical, resolved


def validate_output_root_pre_io(output_root: Path) -> Path:
    lexical, resolved = validate_path_ancestors_pre_io(output_root, "output root")
    require(
        lexical == resolved,
        f"output root changes after resolution: {output_root}",
    )
    return resolved


def reviewer_instructions_payload(rubric: Mapping[str, Any]) -> dict[str, Any]:
    eligibility = rubric["eligibility"]
    return {
        "schema_version": 2,
        "review_protocol_version": rubric["review_protocol_version"],
        "task": "independent_full_49_frame_atomic_review",
        "independence": {
            "do_not_access_other_reviewer_assignment_or_scores": True,
            "do_not_access_private_bindings_or_canonical_manifest": True,
            "prompt_text_is_context_not_visual_evidence": True,
        },
        "video_contract": rubric["video_contract"],
        "atomic_fields": rubric["atomic_fields"],
        "eligibility_operator": eligibility["operator"],
        "required_atomic_thresholds": eligibility["required_thresholds"],
        "coupled_requirement": eligibility["coupled_requirement"],
        "diagnostic_only_fields": eligibility["diagnostic_only_fields"],
        "notes_policy": "reviewer_notes are private audit evidence and never enter the gate",
    }


@contextmanager
def _v2_workflow_globals() -> Iterator[None]:
    replacements: dict[str, Any] = {
        "WORKFLOW_VERSION": WORKFLOW_VERSION,
        "MANIFEST_PROTOCOL": MANIFEST_PROTOCOL,
        "REVIEW_PROTOCOL": REVIEW_PROTOCOL,
        "EXPECTED_ROWS": EXPECTED_ROWS,
        "EXPECTED_CANONICAL_SHA256": EXPECTED_CANONICAL_SHA256,
        "EXPECTED_REVIEW_TEMPLATE_SHA256": EXPECTED_REVIEW_TEMPLATE_SHA256,
        "EXPECTED_REVIEW_RUBRIC_SHA256": EXPECTED_REVIEW_RUBRIC_SHA256,
        "EXPECTED_REVIEW_FREEZE_SHA256": EXPECTED_REVIEW_FREEZE_SHA256,
        "DEFAULT_CANONICAL": DEFAULT_CANONICAL,
        "DEFAULT_REVIEW_TEMPLATE": DEFAULT_REVIEW_TEMPLATE,
        "DEFAULT_REVIEW_RUBRIC": DEFAULT_REVIEW_RUBRIC,
        "DEFAULT_REVIEW_FREEZE": DEFAULT_REVIEW_FREEZE,
        "reject_sealed_path": reject_sealed_path,
        "validate_generation_manifest": validate_generation_manifest,
        "reviewer_instructions_payload": reviewer_instructions_payload,
        "__file__": str(Path(__file__).resolve()),
    }
    previous = {name: getattr(_v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_v1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v1, name, value)


def deterministic_order(generation_ids: Sequence[str], pass_name: str) -> list[str]:
    with _v2_workflow_globals():
        return _v1.deterministic_order(generation_ids, pass_name)


def validate_frozen_inputs(
    canonical_path: Path,
    template_path: Path,
    rubric_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    with _v2_workflow_globals():
        return _v1.validate_frozen_inputs(
            canonical_path,
            template_path,
            rubric_path,
            freeze_path,
        )


def _read_mechanism_prompt_shard(path: Path, mechanism: str) -> list[dict[str, str]]:
    lexical, resolved = validate_path_ancestors_pre_io(
        path,
        f"{mechanism} prompt shard",
    )
    require(path.is_absolute(), f"{mechanism} prompt shard path is not absolute")
    require(lexical == resolved, f"{mechanism} prompt shard path changes after resolution")
    regular_file(path, f"{mechanism} prompt shard")
    rows: list[dict[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        require(
            len(parts) == 3 and all(parts),
            f"{mechanism} prompt shard line {line_no} is malformed",
        )
        rows.append(
            {
                "prompt": parts[0],
                "target_concept": parts[1],
                "expected_effect": parts[2],
            }
        )
    require(len(rows) == 24, f"{mechanism} prompt shard does not contain 24 items")
    return rows


def _validate_mechanism_manifests(
    value: Any,
    *,
    top_level: Mapping[str, Any],
    canonical: Sequence[Mapping[str, str]],
) -> None:
    require(
        isinstance(value, list) and len(value) == len(MECHANISM_ORDER),
        "generation mechanism-manifest bindings are not exactly eight",
    )
    for index, (record, mechanism) in enumerate(zip(value, MECHANISM_ORDER)):
        require(
            isinstance(record, dict)
            and set(record)
            == {"mechanism", "generation_manifest", "generation_manifest_sha256"},
            f"generation mechanism-manifest binding {index} has wrong schema",
        )
        require(record["mechanism"] == mechanism, "generation mechanism-manifest order mismatch")
        raw_path = Path(str(record["generation_manifest"]))
        require(raw_path.is_absolute(), "generation mechanism-manifest path is not absolute")
        lexical, resolved = validate_path_ancestors_pre_io(
            raw_path,
            f"{mechanism} generation manifest",
        )
        require(
            lexical == resolved,
            f"{mechanism} generation-manifest path changes after resolution",
        )
        regular_file(raw_path, f"{mechanism} generation manifest")
        require_sha256(
            str(record["generation_manifest_sha256"]),
            f"generation mechanism-manifest SHA for {mechanism}",
        )
        before = os.stat(raw_path, follow_symlinks=False)
        require(before.st_size > 0, f"{mechanism} generation manifest is empty")
        raw = raw_path.read_bytes()
        after = os.stat(raw_path, follow_symlinks=False)
        require(
            before.st_size == after.st_size == len(raw)
            and before.st_mtime_ns == after.st_mtime_ns,
            f"{mechanism} generation manifest changed while reading",
        )
        require(
            hashlib.sha256(raw).hexdigest()
            == record["generation_manifest_sha256"],
            f"{mechanism} generation-manifest SHA-256 mismatch",
        )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{mechanism} generation manifest is invalid JSON") from exc
        require(
            isinstance(payload, dict) and set(payload) == MECHANISM_MANIFEST_FIELDS,
            f"{mechanism} generation manifest schema is not exact",
        )

        # The mechanism file is subordinate to these frozen top-level bindings.
        require(top_level["protocol_version"] == MANIFEST_PROTOCOL, "generation protocol mismatch")
        require(top_level["canonical_manifest_sha256"] == EXPECTED_CANONICAL_SHA256, "generation canonical binding mismatch")
        require(top_level["prompts_sha256"] == EXPECTED_PROMPTS_SHA256, "generation prompt binding mismatch")
        require(top_level["generator_sha256"] == EXPECTED_GENERATOR_SHA256, "generation implementation binding mismatch")
        require_sha256(str(top_level["stage_registry_sha256"]), "generation stage-registry SHA-256")
        require(top_level["generation"] == GENERATION_FIELDS, "generation parameter binding mismatch")

        require(isinstance(payload["created_at_utc"], str) and payload["created_at_utc"], f"{mechanism} generation timestamp is invalid")
        require(payload["baseline"] == "clean", f"{mechanism} generation baseline mismatch")
        require(payload["pipeline"] == "WanPipeline", f"{mechanism} generation pipeline mismatch")
        require(isinstance(payload["model"], str) and payload["model"], f"{mechanism} generation model binding is invalid")
        require(payload["dry_run"] is False, f"{mechanism} dry-run manifest cannot enter formal review")

        start = index * 24
        end = start + 24
        canonical_slice = canonical[start:end]
        require(
            len(canonical_slice) == 24
            and {row["mechanism"] for row in canonical_slice} == {mechanism},
            f"{mechanism} canonical slice mismatch",
        )
        top_slice = top_level["items"][start:end]
        require(len(top_slice) == 24, f"{mechanism} top-level item slice mismatch")
        seeds = [int(row["seed"]) for row in canonical_slice]
        expected_generation = {
            **MECHANISM_GENERATION_FIXED_FIELDS,
            "seed": seeds[0],
            "seeds": seeds,
        }
        require(
            payload["generation"] == expected_generation,
            f"{mechanism} generation config mismatch",
        )

        prompt_shard = Path(str(payload["prompts"]))
        prompt_rows = _read_mechanism_prompt_shard(prompt_shard, mechanism)
        expected_prompt_rows = [
            {
                "prompt": row["prompt"],
                "target_concept": row["target_concept"],
                "expected_effect": row["expected_footprint"],
            }
            for row in canonical_slice
        ]
        require(
            prompt_rows == expected_prompt_rows,
            f"{mechanism} prompt shard differs from canonical slice",
        )

        items = payload["items"]
        require(
            isinstance(items, list) and len(items) == 24,
            f"{mechanism} generation manifest does not contain 24 items",
        )
        for local_index, (item, row, prompt_row, top_item) in enumerate(
            zip(items, canonical_slice, prompt_rows, top_slice)
        ):
            require(
                isinstance(item, dict) and set(item) == MECHANISM_ITEM_FIELDS,
                f"{mechanism} generation item {local_index} schema mismatch",
            )
            require(item["index"] == local_index, f"{mechanism} generation item order mismatch")
            require(item["prompt"] == prompt_row["prompt"], f"{mechanism} generation item prompt mismatch")
            require(item["target_concept"] == prompt_row["target_concept"], f"{mechanism} generation item target mismatch")
            require(item["expected_effect"] == prompt_row["expected_effect"], f"{mechanism} generation item effect mismatch")
            require(item["seed"] == int(row["seed"]), f"{mechanism} generation item seed mismatch")
            require(top_item["generation_id"] == row["generation_id"], f"{mechanism} top-level generation ID slice mismatch")
            require(top_item["mechanism"] == mechanism, f"{mechanism} top-level mechanism slice mismatch")
            require(top_item["seed"] == item["seed"], f"{mechanism} top-level seed slice mismatch")
            require(str(top_item["video_path"]) == str(item["video_path"]), f"{mechanism} top-level video path slice mismatch")


def validate_generation_manifest(
    generation_path: Path,
    expected_sha256: str,
    canonical: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reject_sealed_path(generation_path, generation_path.resolve())
    require_sha256(expected_sha256, "expected generation-manifest SHA-256")
    regular_file(generation_path, "generation manifest")
    require(
        sha256_file(generation_path) == expected_sha256,
        "generation-manifest SHA-256 mismatch",
    )
    payload = read_json_object(generation_path, "generation manifest")
    require(set(payload) == GENERATION_MANIFEST_FIELDS, "generation manifest schema is not exact")
    require(payload["schema_version"] == 1, "generation manifest schema version mismatch")
    require(payload["protocol_version"] == MANIFEST_PROTOCOL, "generation protocol mismatch")
    require(
        payload["status"] == "frozen_after_exact_media_validation",
        "generation manifest is not frozen",
    )
    require(payload["video_binding_key"] == "generation_id", "generation video binding key mismatch")
    require(payload["video_count"] == EXPECTED_ROWS, "generation video count mismatch")
    require(
        payload["canonical_manifest_sha256"] == EXPECTED_CANONICAL_SHA256,
        "generation canonical binding mismatch",
    )
    require(
        payload["prompts_sha256"] == EXPECTED_PROMPTS_SHA256,
        "generation prompt binding mismatch",
    )
    require(
        payload["generator_sha256"] == EXPECTED_GENERATOR_SHA256,
        "generation implementation binding mismatch",
    )
    require_sha256(str(payload["stage_registry_sha256"]), "generation stage-registry SHA-256")
    require(payload["generation"] == GENERATION_FIELDS, "generation parameter binding mismatch")
    items = payload["items"]
    require(isinstance(items, list) and len(items) == EXPECTED_ROWS, "generation items are not exactly 192")

    by_id: dict[str, dict[str, Any]] = {}
    for index, (item, row) in enumerate(zip(items, canonical)):
        require(isinstance(item, dict) and set(item) == GENERATION_ITEM_FIELDS, f"generation item {index} schema mismatch")
        generation_id = str(item["generation_id"])
        require(generation_id == row["generation_id"], f"generation item {index} order/ID mismatch")
        require(item["canonical_row_index"] == index, f"generation item {index} row index mismatch")
        require(item["mechanism"] == row["mechanism"], f"generation item {index} mechanism mismatch")
        require(item["seed"] == int(row["seed"]), f"generation item {index} seed mismatch")
        require(generation_id not in by_id, "duplicate generation ID")
        video_path = Path(str(item["video_path"]))
        require(video_path.is_absolute(), f"generation item {index} video path is not absolute")
        reject_sealed_path(video_path, video_path.resolve())
        require_sha256(str(item["video_sha256"]), f"video SHA for {generation_id}")
        require(type(item["size_bytes"]) is int and item["size_bytes"] > 0, f"video size is invalid for {generation_id}")
        require(
            item["media"]
            == {"decoded_frames": 49, "fps": "8/1", "height": 480, "width": 832},
            f"generation media contract mismatch for {generation_id}",
        )
        by_id[generation_id] = dict(item)
    _validate_mechanism_manifests(
        payload["mechanism_manifests"],
        top_level=payload,
        canonical=canonical,
    )
    return payload, by_id


def _call_v1(name: str, *args: Any, **kwargs: Any) -> Any:
    with _v2_workflow_globals():
        return getattr(_v1, name)(*args, **kwargs)


def build_package(**kwargs: Any) -> dict[str, Any]:
    validate_output_root_pre_io(Path(kwargs["output_root"]))
    return _call_v1("build_package", **kwargs)


def verify_package(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call_v1("verify_package", *args, **kwargs)


def validate_completed_review(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
    for path in args[:2]:
        if isinstance(path, Path):
            reject_sealed_path(path, path.resolve())
    return _call_v1("validate_completed_review", *args, **kwargs)


def plan_adjudication(**kwargs: Any) -> dict[str, Any]:
    validate_output_root_pre_io(Path(kwargs["output_root"]))
    return _call_v1("plan_adjudication", **kwargs)


def verify_adjudication_root(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call_v1("verify_adjudication_root", *args, **kwargs)


def validate_third_completed(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
    return _call_v1("validate_third_completed", *args, **kwargs)


def finalize_reviews(**kwargs: Any) -> dict[str, Any]:
    validate_output_root_pre_io(Path(kwargs["output_root"]))
    return _call_v1("finalize_reviews", **kwargs)


def verify_final_root(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    return _call_v1("verify_final_root", *args, **kwargs)


@rollback_fresh_partials
def score_bound_review(
    *,
    final_root: Path,
    expected_review_run_registry_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    validate_output_root_pre_io(output_root)
    registry, canonical_adjudicated, package = verify_final_root(
        final_root,
        expected_review_run_registry_sha256,
    )
    canonical_manifest = _v1._resolve_ref(
        final_root,
        package["frozen_inputs"]["canonical_manifest"],
        "canonical capability manifest",
    )
    review_template = _v1._resolve_ref(
        final_root,
        package["frozen_inputs"]["review_template"],
        "review template",
    )
    review_rubric = _v1._resolve_ref(
        final_root,
        package["frozen_inputs"]["review_rubric"],
        "review rubric",
    )
    review_freeze = _v1._resolve_ref(
        final_root,
        package["frozen_inputs"]["review_freeze"],
        "review freeze",
    )
    manifest_name, manifest_sha256 = _review_v2._manifest_commitment_from_freeze(
        review_freeze
    )
    require(
        canonical_manifest.name == manifest_name,
        "canonical manifest filename differs from review freeze commitment",
    )
    manifest_rows = _review_v2.load_manifest(
        canonical_manifest,
        expected_sha256=manifest_sha256,
    )
    blank_rows = _review_v2.verify_frozen_review_inputs(
        manifest_rows,
        canonical_manifest_name=manifest_name,
        canonical_manifest_sha256=manifest_sha256,
        review_template_path=review_template,
        review_rubric_path=review_rubric,
        review_freeze_path=review_freeze,
    )
    adjudicated = _review_v2._read_csv_exact(
        canonical_adjudicated,
        _review_v2.REVIEW_FIELDS,
        "canonical adjudicated review",
    )
    gate = _review_v2.score_adjudicated_rows(
        adjudicated,
        blank_rows,
        canonical_manifest_sha256=manifest_sha256,
        adjudicated_sha256=sha256_file(canonical_adjudicated),
        review_freeze_sha256=sha256_file(review_freeze),
    )
    partial = prepare_partial_root(output_root)
    gate_path = partial / "aggregate_gate.json"
    write_bytes_exclusive(gate_path, pretty_json_bytes(gate))
    bound_registry = {
        "schema_version": 2,
        "workflow_version": WORKFLOW_VERSION,
        "status": "frozen_bound_capability_gate",
        "workflow_implementation": file_ref(Path(__file__).resolve()),
        "frozen_scorer_implementation": file_ref(Path(_review_v2.__file__).resolve()),
        "review_run_registry": file_ref(final_root / "review_run_registry.json"),
        "canonical_adjudicated": file_ref(canonical_adjudicated),
        "generation_manifest": registry["generation_manifest"],
        "aggregate_gate": relative_file_ref(partial, gate_path),
        "capability_gate_status": gate["status"],
        "training_authorized": False,
        "treatment_generation_authorized": False,
    }
    bound_path = partial / "bound_gate_registry.json"
    write_bytes_exclusive(bound_path, pretty_json_bytes(bound_registry))
    expose_partial_root(partial, output_root)
    return {
        "registry": str(output_root / bound_path.name),
        "sha256": sha256_file(output_root / bound_path.name),
        **bound_registry,
    }


def resolve_project_path(project_root: Path, value: Path) -> Path:
    reject_sealed_path(value)
    resolved = value.resolve() if value.is_absolute() else (project_root / value).resolve()
    reject_sealed_path(resolved)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build two isolated full-49-frame v2 assignments")
    build.add_argument("--generation-manifest", type=Path, required=True)
    build.add_argument("--expected-generation-manifest-sha256", required=True)
    build.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL)
    build.add_argument("--review-template", type=Path, default=DEFAULT_REVIEW_TEMPLATE)
    build.add_argument("--review-rubric", type=Path, default=DEFAULT_REVIEW_RUBRIC)
    build.add_argument("--review-freeze", type=Path, default=DEFAULT_REVIEW_FREEZE)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--dry-run", action="store_true")

    adjudicate = subparsers.add_parser("plan-adjudication", help="build the blind third-review disagreement queue")
    adjudicate.add_argument("--package-root", type=Path, required=True)
    adjudicate.add_argument("--expected-package-manifest-sha256", required=True)
    adjudicate.add_argument("--reviewer-a-completed", type=Path, required=True)
    adjudicate.add_argument("--reviewer-b-completed", type=Path, required=True)
    adjudicate.add_argument("--output-root", type=Path, required=True)

    finalize = subparsers.add_parser("finalize", help="merge A/B/third scores into canonical adjudication")
    finalize.add_argument("--package-root", type=Path, required=True)
    finalize.add_argument("--expected-package-manifest-sha256", required=True)
    finalize.add_argument("--adjudication-root", type=Path, required=True)
    finalize.add_argument("--expected-adjudication-manifest-sha256", required=True)
    finalize.add_argument("--third-completed", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, required=True)

    score = subparsers.add_parser("score", help="run the frozen v2 aggregate scorer on bound adjudication")
    score.add_argument("--final-root", type=Path, required=True)
    score.add_argument("--expected-review-run-registry-sha256", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reject_sealed_path(args.project_root)
        project_root = args.project_root.resolve()
        reject_sealed_path(project_root)
        require(project_root.is_dir(), f"project root is not a directory: {project_root}")

        if args.command == "build":
            raw_output = (
                args.output_root
                if args.output_root.is_absolute()
                else project_root / args.output_root
            )
            validate_output_root_pre_io(raw_output)
            generation = resolve_project_path(project_root, args.generation_manifest)
            canonical = resolve_project_path(project_root, args.canonical_manifest)
            template = resolve_project_path(project_root, args.review_template)
            rubric = resolve_project_path(project_root, args.review_rubric)
            freeze = resolve_project_path(project_root, args.review_freeze)
            output = resolve_project_path(project_root, args.output_root)
            canonical_rows, _, _ = validate_frozen_inputs(
                canonical,
                template,
                rubric,
                freeze,
            )
            validate_generation_manifest(
                generation,
                args.expected_generation_manifest_sha256,
                canonical_rows,
            )
            if args.dry_run:
                ids = [row["generation_id"] for row in canonical_rows]
                print(
                    json.dumps(
                        {
                            "status": "dry_run_validated_no_output_written",
                            "generation_manifest_sha256": args.expected_generation_manifest_sha256,
                            "rows": EXPECTED_ROWS,
                            "panel_ranges_inclusive": PANEL_RANGES,
                            "reviewer_a_order_sha256": hashlib.sha256(canonical_json_bytes(deterministic_order(ids, "reviewer_a"))).hexdigest(),
                            "reviewer_b_order_sha256": hashlib.sha256(canonical_json_bytes(deterministic_order(ids, "reviewer_b"))).hexdigest(),
                        },
                        indent=2,
                    )
                )
                return 0
            result = build_package(
                project_root=project_root,
                generation_manifest=generation,
                expected_generation_sha256=args.expected_generation_manifest_sha256,
                canonical_path=canonical,
                template_path=template,
                rubric_path=rubric,
                freeze_path=freeze,
                output_root=output,
            )
        elif args.command == "plan-adjudication":
            raw_output = (
                args.output_root
                if args.output_root.is_absolute()
                else project_root / args.output_root
            )
            validate_output_root_pre_io(raw_output)
            result = plan_adjudication(
                package_root=resolve_project_path(project_root, args.package_root),
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
                reviewer_a_completed=resolve_project_path(project_root, args.reviewer_a_completed),
                reviewer_b_completed=resolve_project_path(project_root, args.reviewer_b_completed),
                output_root=resolve_project_path(project_root, args.output_root),
            )
        elif args.command == "finalize":
            raw_output = (
                args.output_root
                if args.output_root.is_absolute()
                else project_root / args.output_root
            )
            validate_output_root_pre_io(raw_output)
            result = finalize_reviews(
                package_root=resolve_project_path(project_root, args.package_root),
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
                adjudication_root=resolve_project_path(project_root, args.adjudication_root),
                expected_adjudication_manifest_sha256=args.expected_adjudication_manifest_sha256,
                third_completed=resolve_project_path(project_root, args.third_completed),
                output_root=resolve_project_path(project_root, args.output_root),
            )
        else:
            raw_output = (
                args.output_root
                if args.output_root.is_absolute()
                else project_root / args.output_root
            )
            validate_output_root_pre_io(raw_output)
            result = score_bound_review(
                final_root=resolve_project_path(project_root, args.final_root),
                expected_review_run_registry_sha256=args.expected_review_run_registry_sha256,
                output_root=resolve_project_path(project_root, args.output_root),
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
