#!/usr/bin/env python3
"""Fail-closed Responses transport for one formal seven-mechanism review pass.

Only one pass's public ``assignments.jsonl``, blank ``scores.jsonl``, manifest,
and anonymous panel images are consumed.  Private answer keys and the other
pass are outside the input contract.  Every schema-valid response is persisted
as an immutable checkpoint.  API, timeout, rate-limit, refusal, incomplete,
and parse failures are infrastructure errors and never become scientific
scores.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

try:
    import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
    from review_causal_role_erasure_7mechanism_targets_v2 import (
        canonical_json_bytes,
        file_ref,
        isolated_urllib_transport,
        object_sha256,
        regular_file,
        relative_file_ref,
        sha256_file,
        write_bytes_exclusive,
        write_json_exclusive,
    )
except ModuleNotFoundError:  # imported as ``scripts.<module>`` in tests
    from scripts import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
    from scripts.review_causal_role_erasure_7mechanism_targets_v2 import (
        canonical_json_bytes,
        file_ref,
        isolated_urllib_transport,
        object_sha256,
        regular_file,
        relative_file_ref,
        sha256_file,
        write_bytes_exclusive,
        write_json_exclusive,
    )


WORKFLOW_VERSION = "causal_role_erasure_7mechanism_formal_responses_transport_v2"
EXPECTED_ITEMS = 2448
PASS_NAMES = ("pass_a", "pass_b")
PASS_IDS = {"pass_a": "A", "pass_b": "B"}
PACKAGE_PROTOCOL = "causal_role_erasure_7m_anonymous_review_v1"
PANEL_WINDOWS = [[0, 12], [9, 21], [18, 30], [27, 39], [36, 48]]
FRAME_COUNT = 49
MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 1600
MAX_IMAGE_BYTES = 3_145_728
COMPOSITE_WIDTH = 1456
COMPOSITE_HEIGHT = 520
JPEG_QUALITY = 82
TILE_WIDTH = 112
TILE_HEIGHT = 64
DEFAULT_WORKERS = 16  # Run A and B together for the measured total concurrency of 32.
CHECKPOINT_STATUS = "schema_valid_scientific_checkpoint"
INFRASTRUCTURE_ERROR_STATUS = "infrastructure_error_no_scientific_score"

CAUSAL_FIELDS = (
    "source_visibility",
    "footprint_visibility",
    "receiver_preservation",
    "video_quality",
)
SPECIFICITY_FIELDS = (
    "protected_object_visibility",
    "noncausal_role_adherence",
    "receiver_preservation",
    "video_quality",
)
SCORE_FIELDS_BY_KIND = {
    "causal": CAUSAL_FIELDS,
    "specificity": SPECIFICITY_FIELDS,
}
ASSIGNMENT_FIELDS = (
    "anonymous_review_id",
    "case_kind",
    "mechanism_name",
    "prompt",
    "source_object",
    "receiver",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "protected_object",
    "specificity_subtype",
    "acceptable_alternative_cause",
    "composite_path",
    "composite_sha256",
    "assignment_sha256",
)
SCORE_ROW_FIELDS = (
    "anonymous_review_id",
    "assignment_sha256",
    "case_kind",
    "scores",
    "confidence",
    "evidence_frames",
    "evidence_observations",
    "unusable_reason",
    "status",
)
MANIFEST_REQUIRED_FIELDS = {
    "protocol",
    "schema_version",
    "pass_id",
    "item_count",
    "assignments",
    "blank_scores",
    "panel_windows",
    "composite_contract",
}
RUN_REGISTRATION_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_id",
    "package_protocol",
    "package_schema",
    "evaluation_code_registry_sha256",
    "pass_manifest",
    "assignments",
    "blank_scores",
    "selection_mode",
    "selected_start",
    "selected_limit",
    "retry_sources",
    "selected_count",
    "selected_review_ids",
    "selected_inventory_sha256",
    "transport",
    "proxy_version",
    "endpoint",
    "model",
    "reasoning_effort",
    "temperature",
    "max_output_tokens",
    "store",
    "workers",
    "timeout",
    "response_schemas_sha256",
    "dry_run",
)
CHECKPOINT_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_id",
    "anonymous_review_id",
    "case_kind",
    "assignment_sha256",
    "composite_sha256",
    "request_sha256",
    "response_schema_sha256",
    "model",
    "observed_model",
    "raw_response",
    "model_content_sha256",
    "normalized",
)
ERROR_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_id",
    "anonymous_review_id",
    "case_kind",
    "assignment_sha256",
    "composite_sha256",
    "request_sha256",
    "observed_model",
    "error_type",
    "error_message",
    "raw_response",
    "model_content_sha256",
    "replay_policy",
)
RUN_SUMMARY_FIELDS = (
    "schema_version",
    "workflow_version",
    "status",
    "pass_id",
    "selected_count",
    "successful_checkpoints",
    "infrastructure_errors",
    "run_registration",
    "checkpoints",
    "errors",
    "scientific_zero_fallbacks",
)

Transport = Callable[[str, str, dict[str, Any], int], dict[str, Any]]


class FormalReviewTransportError(RuntimeError):
    """Fail-closed public-input, Responses, checkpoint, or merge error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalReviewTransportError(message)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalReviewTransportError(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FormalReviewTransportError(f"cannot read {label}: {path}") from exc
    require(lines and all(line.strip() for line in lines), f"{label} is empty or contains blank lines")
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FormalReviewTransportError(f"{label} row {index} is invalid JSON") from exc
        require(isinstance(row, dict), f"{label} row {index} is not an object")
        rows.append(row)
    return rows


def strict_model_json(text: str) -> dict[str, Any]:
    """Parse one entire JSON object, rejecting duplicate keys and non-finite numbers."""

    require(isinstance(text, str) and text.strip(), "model output is blank")

    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs_value:
            if key in value:
                raise FormalReviewTransportError(f"model JSON repeats key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise FormalReviewTransportError(f"model JSON contains non-finite number: {value}")

    try:
        parsed = json.loads(
            text.strip(),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise FormalReviewTransportError("model output is not exactly one JSON value") from exc
    require(isinstance(parsed, dict), "model output JSON is not an object")
    return parsed


def _assignment_digest(row: Mapping[str, Any]) -> str:
    return object_sha256({key: value for key, value in row.items() if key != "assignment_sha256"})


def _score_fields(case_kind: str) -> tuple[str, ...]:
    fields = SCORE_FIELDS_BY_KIND.get(case_kind)
    require(fields is not None, f"unknown case_kind: {case_kind}")
    return fields


def _blank_score_row_valid(row: Mapping[str, Any], assignment: Mapping[str, Any]) -> None:
    require(set(row) == set(SCORE_ROW_FIELDS), "blank score row fields differ from public contract")
    require(row["anonymous_review_id"] == assignment["anonymous_review_id"], "blank score ID differs from assignment")
    require(row["assignment_sha256"] == assignment["assignment_sha256"], "blank score binding differs from assignment")
    require(row["case_kind"] == assignment["case_kind"], "blank score case_kind differs from assignment")
    fields = _score_fields(str(assignment["case_kind"]))
    for name in ("scores", "confidence"):
        value = row[name]
        require(isinstance(value, dict) and set(value) == set(fields), f"blank {name} keys differ from case schema")
        require(all(item is None for item in value.values()), f"blank {name} contains a value")
    for name in ("evidence_frames", "evidence_observations"):
        value = row[name]
        require(isinstance(value, dict) and set(value) == set(fields), f"blank {name} keys differ from case schema")
        require(all(item == [] for item in value.values()), f"blank {name} contains evidence")
    require(row["unusable_reason"] == "" and row["status"] == "pending", "blank score row is not pending/blank")


def _resolve_composite(pass_root: Path, value: str) -> Path:
    require(isinstance(value, str) and value.strip() == value and value, "invalid composite_path")
    raw = Path(value)
    require(not raw.is_absolute(), "public composite_path must be relative")
    resolved = (pass_root / raw).resolve(strict=True)
    media_root = (pass_root / "media").resolve(strict=True)
    require(resolved.parent == media_root, "composite escaped this pass's public media root")
    regular_file(resolved, "public composite")
    require(resolved.suffix.lower() == ".jpg", "public composite must be the frozen JPEG encoding")
    require(0 < resolved.stat().st_size <= MAX_IMAGE_BYTES, "composite violates the one-image size limit")
    try:
        with Image.open(resolved) as image:
            require(image.format == "JPEG", "public composite bytes are not JPEG")
            require(
                image.size == (COMPOSITE_WIDTH, COMPOSITE_HEIGHT),
                "public composite geometry differs from the frozen contract",
            )
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise FormalReviewTransportError(
            f"public composite is not a valid JPEG: {resolved}"
        ) from exc
    return resolved


def load_public_pass(
    assignments_path: Path,
    blank_scores_path: Path,
    *,
    expected_items: int | None = EXPECTED_ITEMS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load exactly one public pass without inspecting siblings or private keys."""

    regular_file(assignments_path, "public assignments")
    regular_file(blank_scores_path, "public blank scores")
    pass_root = assignments_path.parent.resolve(strict=True)
    require(pass_root == blank_scores_path.parent.resolve(strict=True), "assignments/scores must be in the same pass")
    require(pass_root.name in PASS_NAMES and pass_root.parent.name == "public", "inputs are not a formal public pass")
    require(assignments_path.name == "assignments.jsonl", "unexpected assignment filename")
    require(blank_scores_path.name == "scores.jsonl", "unexpected blank-score filename")
    manifest_path = pass_root / "pass_manifest.json"
    manifest = _read_json(manifest_path, "public pass manifest")
    registry_path = pass_root.parent / "evaluation_code_registry.json"
    commitments_path = pass_root.parent / "key_commitments.json"
    registry = _read_json(registry_path, "evaluation code registry")
    commitments = _read_json(commitments_path, "public key commitments")
    try:
        evaluation_code.validate_registry(
            registry, Path(__file__).resolve().parents[1]
        )
    except ValueError as exc:
        raise FormalReviewTransportError(str(exc)) from exc
    code_binding = commitments.get("evaluation_code_registry")
    require(
        isinstance(code_binding, dict)
        and code_binding.get("path") == "evaluation_code_registry.json"
        and code_binding.get("sha256") == sha256_file(registry_path)
        and code_binding.get("registry_sha256")
        == registry.get("registry_sha256"),
        "public evaluation-code registry binding differs",
    )
    require(
        set(manifest)
        == MANIFEST_REQUIRED_FIELDS | {"ordering_commitment"},
        "public pass manifest fields differ from the frozen public contract",
    )
    require(manifest["pass_id"] == PASS_IDS[pass_root.name], "manifest pass_id differs from directory")
    require(manifest["protocol"] == PACKAGE_PROTOCOL, "manifest protocol differs from formal contract")
    require(manifest["schema_version"] == 1, "manifest schema_version differs from formal contract")
    require(_hex64(manifest["ordering_commitment"]), "manifest ordering commitment is invalid")
    require(manifest["panel_windows"] == PANEL_WINDOWS, "manifest panel windows differ from full-49 protocol")
    for key, path, filename in (
        ("assignments", assignments_path, "assignments.jsonl"),
        ("blank_scores", blank_scores_path, "scores.jsonl"),
    ):
        ref = manifest[key]
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, f"manifest {key} ref differs")
        require(ref["path"] == filename and ref["sha256"] == sha256_file(path), f"{key} file differs from manifest")
    composite_contract = manifest["composite_contract"]
    require(
        isinstance(composite_contract, dict)
        and composite_contract.get("path_base") == "pass_root"
        and composite_contract.get("directory") == "media",
        "manifest composite path contract differs",
    )
    require(
        composite_contract
        == {
            "path_base": "pass_root",
            "directory": "media",
            "format": "jpeg",
            "width": COMPOSITE_WIDTH,
            "height": COMPOSITE_HEIGHT,
            "tile_width": TILE_WIDTH,
            "tile_height": TILE_HEIGHT,
            "tile_fit": "deterministic_center_crop_no_letterbox",
            "resampling": "Pillow.Image.Resampling.LANCZOS",
            "jpeg_quality": JPEG_QUALITY,
            "max_bytes": MAX_IMAGE_BYTES,
            "one_image_per_assignment": True,
        },
        "manifest composite encoding contract differs",
    )

    assignments = _read_jsonl(assignments_path, "public assignments")
    scores = _read_jsonl(blank_scores_path, "public blank scores")
    require(len(assignments) == len(scores), "assignment/blank-score row counts differ")
    if expected_items is not None:
        require(len(assignments) == expected_items, f"public pass must contain exactly {expected_items} items")
    require(manifest["item_count"] == len(assignments), "manifest item_count differs from public files")

    seen_ids: set[str] = set()
    seen_composites: set[Path] = set()
    loaded: list[dict[str, Any]] = []
    for index, (assignment, blank) in enumerate(zip(assignments, scores)):
        require(set(assignment) == set(ASSIGNMENT_FIELDS), f"assignment row {index} fields differ")
        review_id = assignment["anonymous_review_id"]
        require(
            isinstance(review_id, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", review_id) is not None
            and review_id not in seen_ids,
            f"assignment row {index} has invalid/duplicate ID",
        )
        require(assignment["case_kind"] in SCORE_FIELDS_BY_KIND, f"assignment row {index} has invalid case_kind")
        for field in (
            "mechanism_name",
            "prompt",
            "source_object",
            "receiver",
            "expected_trigger",
            "expected_footprint",
            "expected_counterfactual_state",
        ):
            value = assignment[field]
            require(isinstance(value, str) and value.strip() == value and value, f"assignment row {index} has invalid {field}")
        if assignment["case_kind"] == "causal":
            require(
                assignment["protected_object"] == ""
                and assignment["specificity_subtype"] == ""
                and assignment["acceptable_alternative_cause"] == "",
                f"assignment row {index} causal-only public fields differ",
            )
        else:
            require(
                isinstance(assignment["protected_object"], str)
                and bool(assignment["protected_object"])
                and assignment["specificity_subtype"]
                in {
                    "same_noun_noncausal",
                    "role_swap_or_near_causal",
                    "same_footprint_alternative_cause",
                },
                f"assignment row {index} specificity semantics differ",
            )
            alternative = assignment["acceptable_alternative_cause"]
            require(
                isinstance(alternative, str)
                and (
                    bool(alternative)
                    == (
                        assignment["specificity_subtype"]
                        == "same_footprint_alternative_cause"
                    )
                ),
                f"assignment row {index} alternative-cause semantics differ",
            )
        require(_hex64(assignment["composite_sha256"]), f"assignment row {index} has invalid composite SHA-256")
        require(_hex64(assignment["assignment_sha256"]), f"assignment row {index} has invalid assignment SHA-256")
        require(_assignment_digest(assignment) == assignment["assignment_sha256"], f"assignment row {index} binding changed")
        composite = _resolve_composite(pass_root, assignment["composite_path"])
        require(composite not in seen_composites, f"assignment row {index} reuses a composite")
        require(sha256_file(composite) == assignment["composite_sha256"], f"assignment row {index} composite SHA-256 changed")
        _blank_score_row_valid(blank, assignment)
        seen_ids.add(review_id)
        seen_composites.add(composite)
        loaded.append({"assignment": assignment, "blank_score": blank, "composite": composite})

    media_root = pass_root / "media"
    require(media_root.is_dir() and not media_root.is_symlink(), "public media directory is unsafe or missing")
    entries = list(media_root.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in entries), "public media directory contains unsafe entries")
    require({path.resolve(strict=True) for path in entries} == seen_composites, "public media inventory differs from assignments")
    return {
        **manifest,
        "path": manifest_path,
        "evaluation_code_registry_sha256": registry["registry_sha256"],
    }, loaded


def response_json_schema(case_kind: str) -> dict[str, Any]:
    fields = _score_fields(case_kind)
    evidence_item = {
        "type": "object",
        "properties": {
            "frame_index": {"type": "integer", "minimum": 0, "maximum": 48},
            "observation": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["frame_index", "observation"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        field: {"type": "integer", "enum": [0, 1, 2]} for field in fields
    }
    properties["confidence"] = {
        "type": "object",
        "properties": {field: {"type": "number", "minimum": 0, "maximum": 1} for field in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    properties["frame_evidence"] = {
        "type": "object",
        "properties": {
            field: {"type": "array", "items": evidence_item, "minItems": 1, "maxItems": 8}
            for field in fields
        },
        "required": list(fields),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [*fields, "confidence", "frame_evidence"],
        "additionalProperties": False,
    }


def response_schemas_sha256() -> dict[str, str]:
    return {kind: object_sha256(response_json_schema(kind)) for kind in SCORE_FIELDS_BY_KIND}


def prompt_for(row: Mapping[str, Any]) -> str:
    assignment = row["assignment"]
    kind = str(assignment["case_kind"])
    if kind == "causal":
        scale = {
            "source_visibility": "0 absent, 1 partial/ambiguous, 2 clearly visible (lower is better)",
            "footprint_visibility": "0 absent, 1 partial/ambiguous, 2 clearly visible (lower is better)",
            "receiver_preservation": "0 bad/unrecognizable, 1 partial, 2 good/recognizable",
            "video_quality": "0 bad/unusable, 1 partially usable, 2 good",
        }
    else:
        scale = {
            "protected_object_visibility": "0 absent, 1 partial/ambiguous, 2 clearly visible",
            "noncausal_role_adherence": (
                "0 the protected object clearly performs the registered trigger or becomes the causal source of the target mechanism; "
                "1 its causal role is partial or ambiguous; 2 it remains non-causal, does not complete the registered trigger, and does not become the target mechanism's causal source"
            ),
            "receiver_preservation": "0 bad/unrecognizable, 1 partial, 2 good/recognizable",
            "video_quality": "0 bad/unusable, 1 partially usable, 2 good",
        }
    return "\n".join(
        [
            "You are independently labeling one anonymous generated video for a formal benchmark.",
            "The single supplied composite contains all 49 indexed frames in five overlapping temporal panels:",
            "0-12, 9-21, 18-30, 27-39, and 36-48 (inclusive).",
            "Judge only visible evidence in the frames. Prompt text describes the requested scene and is never proof that an object or event is visible.",
            "Do not infer the generating method, compare methods, or choose a winner.",
            "A partial score is valid scientific evidence; complete disappearance is not required.",
            "",
            f"Case kind: {kind}",
            f"Mechanism context: {assignment['mechanism_name']}",
            f"Requested scene: {assignment['prompt']}",
            f"Source object: {assignment['source_object']}",
            f"Receiver: {assignment['receiver']}",
            f"Registered trigger: {assignment['expected_trigger']}",
            f"Expected footprint: {assignment['expected_footprint']}",
            f"Expected counterfactual state: {assignment['expected_counterfactual_state']}",
            *(
                []
                if kind == "causal"
                else [
                    f"Protected object that should remain visible: {assignment['protected_object']}",
                    f"Specificity subtype: {assignment['specificity_subtype']}",
                    (
                        "Acceptable alternative cause: "
                        + (
                            assignment["acceptable_alternative_cause"]
                            if assignment["acceptable_alternative_cause"]
                            else "none registered"
                        )
                    ),
                    "For specificity, the protected object should remain visible but must not complete the registered trigger or become the causal source of the target mechanism.",
                    "If this is same_footprint_alternative_cause, do not penalize a visible footprint when the registered acceptable alternative visibly causes it and the protected object remains non-causal.",
                ]
            ),
            "",
            "Score each field with exactly one integer 0, 1, or 2 under these frozen meanings:",
            json.dumps(scale, ensure_ascii=False, sort_keys=True),
            "For every field, return a separate confidence in [0,1] and 1-8 frame-indexed visible observations.",
            "Return only the JSON object required by the attached strict response schema.",
        ]
    )


def image_data_url(path: Path) -> str:
    regular_file(path, "public composite")
    require(0 < path.stat().st_size <= MAX_IMAGE_BYTES, "composite violates the one-image size limit")
    require(path.suffix.lower() == ".jpg", "public composite must be JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_payload_for(row: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(row["assignment"]["case_kind"])
    return {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_for(row)},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(row["composite"]),
                        "detail": "high",
                    },
                ],
            }
        ],
        "reasoning": {"effort": REASONING_EFFORT},
        "temperature": TEMPERATURE,
        "text": {
            "format": {
                "type": "json_schema",
                "name": f"formal_{kind}_atomic_scores",
                "strict": True,
                "schema": response_json_schema(kind),
            }
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


def dry_payload_for(row: Mapping[str, Any]) -> dict[str, Any]:
    assignment = row["assignment"]
    kind = str(assignment["case_kind"])
    return {
        "anonymous_review_id": assignment["anonymous_review_id"],
        "case_kind": kind,
        "assignment_sha256": assignment["assignment_sha256"],
        "composite_path": str(row["composite"]),
        "composite_sha256": assignment["composite_sha256"],
        "prompt": prompt_for(row),
        "endpoint_payload_without_image_bytes": {
            "model": MODEL,
            "reasoning": {"effort": REASONING_EFFORT},
            "temperature": TEMPERATURE,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"formal_{kind}_atomic_scores",
                    "strict": True,
                    "schema": response_json_schema(kind),
                }
            },
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "input_contract": ["one input_text", "one input_image <= 3145728 bytes"],
        },
    }


def responses_output_text(response: Mapping[str, Any]) -> str:
    require(isinstance(response, Mapping), "Responses API result is not an object")
    require(not response.get("error"), "Responses API returned an error object")
    require(response.get("model") == MODEL, "Responses API returned a different or missing model identity")
    require(response.get("status") == "completed", "Responses API result is not completed")
    texts: list[str] = []
    output = response.get("output")
    require(isinstance(output, list), "Responses API output is missing or not a list")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        require(isinstance(content, list), "Responses message content is not a list")
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "refusal":
                raise FormalReviewTransportError("Responses API returned a refusal")
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                text = part.get("text")
                require(isinstance(text, str) and text.strip(), "Responses output_text is blank")
                texts.append(text)
    require(len(texts) == 1, "Responses API must return exactly one output_text part")
    return texts[0]


def normalize_model_object(parsed: Mapping[str, Any], case_kind: str) -> dict[str, Any]:
    require(isinstance(parsed, Mapping), "model response is not a JSON object")
    fields = _score_fields(case_kind)
    require(set(parsed) == {*fields, "confidence", "frame_evidence"}, "model response keys differ from case schema")
    scores: dict[str, int] = {}
    for field in fields:
        value = parsed[field]
        require(type(value) is int and value in {0, 1, 2}, f"invalid score for {field}")
        scores[field] = value
    confidence_raw = parsed["confidence"]
    evidence_raw = parsed["frame_evidence"]
    require(isinstance(confidence_raw, Mapping) and set(confidence_raw) == set(fields), "confidence keys differ from case schema")
    require(isinstance(evidence_raw, Mapping) and set(evidence_raw) == set(fields), "frame_evidence keys differ from case schema")
    confidence: dict[str, float] = {}
    evidence_frames: dict[str, list[int]] = {}
    evidence_observations: dict[str, list[str]] = {}
    for field in fields:
        raw_confidence = confidence_raw[field]
        require(type(raw_confidence) in {int, float} and 0 <= float(raw_confidence) <= 1, f"invalid confidence for {field}")
        confidence[field] = float(raw_confidence)
        items = evidence_raw[field]
        require(isinstance(items, list) and 1 <= len(items) <= 8, f"{field} evidence must contain 1..8 items")
        frames: list[int] = []
        observations: list[str] = []
        for item in items:
            require(isinstance(item, Mapping) and set(item) == {"frame_index", "observation"}, f"{field} evidence item schema differs")
            frame = item["frame_index"]
            observation = item["observation"]
            require(type(frame) is int and 0 <= frame < FRAME_COUNT, f"invalid evidence frame for {field}")
            require(frame not in frames, f"duplicate evidence frame for {field}")
            require(isinstance(observation, str) and 1 <= len(observation.strip()) <= 1000, f"invalid evidence observation for {field}")
            frames.append(frame)
            observations.append(observation.strip())
        evidence_frames[field] = frames
        evidence_observations[field] = observations
    return {
        "scores": scores,
        "confidence": confidence,
        "evidence_frames": evidence_frames,
        "evidence_observations": evidence_observations,
    }


def responses_endpoint(base_url: str) -> str:
    require(isinstance(base_url, str) and base_url.strip() == base_url and base_url, "base URL is blank")
    value = base_url.rstrip("/")
    if value.endswith("/v1/responses"):
        endpoint = value
    elif value.endswith("/v1"):
        endpoint = value + "/responses"
    else:
        endpoint = value + "/v1/responses"
    require(endpoint.startswith(("http://", "https://")), "Responses endpoint must use HTTP(S)")
    return endpoint


def _request_descriptor(
    row: Mapping[str, Any], endpoint: str, observed_model: str | None
) -> dict[str, Any]:
    assignment = row["assignment"]
    kind = str(assignment["case_kind"])
    return {
        "anonymous_review_id": assignment["anonymous_review_id"],
        "case_kind": kind,
        "assignment_sha256": assignment["assignment_sha256"],
        "composite_sha256": assignment["composite_sha256"],
        "prompt_sha256": hashlib.sha256(prompt_for(row).encode("utf-8")).hexdigest(),
        "response_schema_sha256": object_sha256(response_json_schema(kind)),
        "endpoint": endpoint,
        "model": MODEL,
        "observed_model": observed_model,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


def evaluate_one(
    row: Mapping[str, Any],
    *,
    endpoint: str,
    api_key: str,
    timeout: int,
    transport: Transport = isolated_urllib_transport,
) -> dict[str, Any]:
    response: dict[str, Any] | None = None
    content = ""
    observed_model: str | None = None
    try:
        payload = request_payload_for(row)
        response = transport(endpoint, api_key, payload, timeout)
        require(isinstance(response, dict), "Responses API result is not an object")
        raw_observed_model = response.get("model")
        observed_model = (
            raw_observed_model if isinstance(raw_observed_model, str) else None
        )
        content = responses_output_text(response)
        parsed = strict_model_json(content)
        normalized = normalize_model_object(parsed, str(row["assignment"]["case_kind"]))
        return {
            "ok": True,
            "descriptor": _request_descriptor(row, endpoint, observed_model),
            "response": response,
            "model_content": content,
            "normalized": normalized,
            "observed_model": observed_model,
        }
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        return {
            "ok": False,
            "descriptor": _request_descriptor(row, endpoint, observed_model),
            "response": response,
            "model_content": content,
            "observed_model": observed_model,
            "error_type": type(exc).__name__,
            "error_message": message,
        }


def _select_rows(
    rows: Sequence[Mapping[str, Any]], start: int, limit: int | None
) -> list[Mapping[str, Any]]:
    require(0 <= start <= len(rows), "--start is outside assignment")
    require(limit is None or limit > 0, "--limit must be positive")
    selected = list(rows[start:] if limit is None else rows[start : start + limit])
    require(selected, "selected review shard is empty")
    return selected


def _prepare_fresh_run_root(output_root: Path) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    for name in ("checkpoints", "raw_responses", "errors"):
        (output_root / name).mkdir()
    write_bytes_exclusive(output_root / ".incomplete", b"formal review run incomplete\n", mode=0o600)


def _source_run_refs(run_roots: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "run_registration": file_ref(root / "run_registration.json"),
            "run_summary": file_ref(root / "run_summary.json"),
        }
        for root in run_roots
    ]


def _load_retry_ids(
    *,
    run_roots: Sequence[Path],
    assignments_path: Path,
    blank_scores_path: Path,
    manifest: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    require(run_roots, "retry requires at least one failed run root")
    public_by_id = {
        row["assignment"]["anonymous_review_id"]: row for row in public_rows
    }
    failed: set[str] = set()
    for root in run_roots:
        require(root.is_dir() and not root.is_symlink() and not (root / ".incomplete").exists(), f"retry source is missing or incomplete: {root}")
        registration = _read_json(root / "run_registration.json", "retry run registration")
        summary = _read_json(root / "run_summary.json", "retry run summary")
        require(tuple(registration) == RUN_REGISTRATION_FIELDS, "retry registration fields differ")
        require(tuple(summary) == RUN_SUMMARY_FIELDS, "retry summary fields differ")
        require(
            registration["workflow_version"] == WORKFLOW_VERSION
            and registration["pass_id"] == manifest["pass_id"]
            and registration["evaluation_code_registry_sha256"]
            == manifest["evaluation_code_registry_sha256"]
            and registration["dry_run"] is False,
            "retry source identity differs",
        )
        require(
            registration["assignments"]["sha256"] == sha256_file(assignments_path)
            and registration["blank_scores"]["sha256"] == sha256_file(blank_scores_path),
            "retry source binds different public files",
        )
        require(summary["run_registration"]["sha256"] == sha256_file(root / "run_registration.json"), "retry summary registration binding changed")
        for ref in summary["errors"]:
            error_path = _resolve_ref(root, ref, "retry infrastructure error")
            error = _read_json(error_path, "retry infrastructure error")
            require(tuple(error) == ERROR_FIELDS, "retry infrastructure error fields differ")
            review_id = error["anonymous_review_id"]
            public = public_by_id.get(review_id)
            require(public is not None, "retry error ID is outside this public pass")
            assignment = public["assignment"]
            require(
                error["workflow_version"] == WORKFLOW_VERSION
                and error["status"] == INFRASTRUCTURE_ERROR_STATUS
                and error["pass_id"] == manifest["pass_id"]
                and error["case_kind"] == assignment["case_kind"]
                and error["assignment_sha256"] == assignment["assignment_sha256"]
                and error["composite_sha256"] == assignment["composite_sha256"],
                "retry infrastructure error binding differs",
            )
            failed.add(str(review_id))
    selected = [
        row["assignment"]["anonymous_review_id"]
        for row in public_rows
        if row["assignment"]["anonymous_review_id"] in failed
    ]
    require(selected, "retry sources contain no infrastructure errors")
    return selected


def run_review(
    *,
    assignments_path: Path,
    blank_scores_path: Path,
    output_root: Path,
    dry_run: bool,
    start: int = 0,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
    timeout: int = 180,
    base_url: str = "http://127.0.0.1:4141",
    api_key: str = "local-proxy-no-secret",
    proxy_version: str = "copilot-claude-local",
    retry_run_roots: Sequence[Path] = (),
    expected_items: int | None = EXPECTED_ITEMS,
    transport: Transport = isolated_urllib_transport,
) -> dict[str, Any]:
    require(1 <= workers <= 256, "workers must be in [1,256]")
    require(timeout > 0, "timeout must be positive")
    require(isinstance(proxy_version, str) and proxy_version.strip() == proxy_version and proxy_version, "proxy version is blank")
    endpoint = responses_endpoint(base_url)
    manifest, all_rows = load_public_pass(
        assignments_path, blank_scores_path, expected_items=expected_items
    )
    if retry_run_roots:
        require(start == 0 and limit is None, "retry cannot be combined with --start/--limit")
        selected_ids = _load_retry_ids(
            run_roots=retry_run_roots,
            assignments_path=assignments_path,
            blank_scores_path=blank_scores_path,
            manifest=manifest,
            public_rows=all_rows,
        )
        wanted = set(selected_ids)
        selected = [row for row in all_rows if row["assignment"]["anonymous_review_id"] in wanted]
        selection_mode = "fresh_retry_of_infrastructure_errors"
        selected_start = selected_limit = None
        retry_sources = _source_run_refs(retry_run_roots)
    else:
        selected = _select_rows(all_rows, start, limit)
        selected_ids = [row["assignment"]["anonymous_review_id"] for row in selected]
        selection_mode = "ordered_shard"
        selected_start, selected_limit = start, limit
        retry_sources = []

    _prepare_fresh_run_root(output_root)
    registration = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "registered_fresh_public_pass_shard",
        "pass_id": manifest["pass_id"],
        "package_protocol": manifest["protocol"],
        "package_schema": manifest["schema_version"],
        "evaluation_code_registry_sha256": manifest[
            "evaluation_code_registry_sha256"
        ],
        "pass_manifest": file_ref(manifest["path"]),
        "assignments": file_ref(assignments_path),
        "blank_scores": file_ref(blank_scores_path),
        "selection_mode": selection_mode,
        "selected_start": selected_start,
        "selected_limit": selected_limit,
        "retry_sources": retry_sources,
        "selected_count": len(selected),
        "selected_review_ids": selected_ids,
        "selected_inventory_sha256": object_sha256(
            [
                {
                    "anonymous_review_id": row["assignment"]["anonymous_review_id"],
                    "assignment_sha256": row["assignment"]["assignment_sha256"],
                    "composite_sha256": row["assignment"]["composite_sha256"],
                }
                for row in selected
            ]
        ),
        "transport": "openai_compatible_responses_api",
        "proxy_version": proxy_version,
        "endpoint": endpoint,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "workers": workers,
        "timeout": timeout,
        "response_schemas_sha256": response_schemas_sha256(),
        "dry_run": dry_run,
    }
    require(tuple(registration) == RUN_REGISTRATION_FIELDS, "internal run registration schema drift")
    registration_path = output_root / "run_registration.json"
    write_json_exclusive(registration_path, registration, mode=0o600)

    if dry_run:
        payloads_path = output_root / "payloads.jsonl"
        write_bytes_exclusive(
            payloads_path,
            b"".join(canonical_json_bytes(dry_payload_for(row)) for row in selected),
            mode=0o600,
        )
        summary = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": "dry_run_payloads_only_no_api_calls",
            "pass_id": manifest["pass_id"],
            "selected_count": len(selected),
            "successful_checkpoints": 0,
            "infrastructure_errors": 0,
            "run_registration": relative_file_ref(output_root, registration_path),
            "checkpoints": [],
            "errors": [],
            "scientific_zero_fallbacks": 0,
        }
        require(tuple(summary) == RUN_SUMMARY_FIELDS, "internal dry summary schema drift")
        write_json_exclusive(output_root / "run_summary.json", summary, mode=0o600)
        (output_root / ".incomplete").unlink()
        return summary

    successful_by_index: dict[int, dict[str, Any]] = {}
    errors_by_index: dict[int, dict[str, Any]] = {}

    def persist(index: int, row: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        assignment = row["assignment"]
        review_id = assignment["anonymous_review_id"]
        descriptor = result["descriptor"]
        if not result["ok"]:
            raw_ref: dict[str, Any] | None = None
            content_hash: str | None = None
            if result.get("response") is not None:
                raw_path = output_root / "raw_responses" / f"{review_id}.json"
                write_json_exclusive(raw_path, result["response"], mode=0o600)
                raw_ref = relative_file_ref(output_root, raw_path)
                content_hash = hashlib.sha256(str(result.get("model_content", "")).encode("utf-8")).hexdigest()
            error = {
                "schema_version": 1,
                "workflow_version": WORKFLOW_VERSION,
                "status": INFRASTRUCTURE_ERROR_STATUS,
                "pass_id": manifest["pass_id"],
                "anonymous_review_id": review_id,
                "case_kind": assignment["case_kind"],
                "assignment_sha256": assignment["assignment_sha256"],
                "composite_sha256": assignment["composite_sha256"],
                "request_sha256": object_sha256(descriptor),
                "observed_model": result["observed_model"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
                "raw_response": raw_ref,
                "model_content_sha256": content_hash,
                "replay_policy": "retry this ID only in a new fresh-only run root",
            }
            require(tuple(error) == ERROR_FIELDS, "internal infrastructure-error schema drift")
            error_path = output_root / "errors" / f"{review_id}.json"
            write_json_exclusive(error_path, error, mode=0o600)
            errors_by_index[index] = relative_file_ref(output_root, error_path)
            return

        raw_path = output_root / "raw_responses" / f"{review_id}.json"
        write_json_exclusive(raw_path, result["response"], mode=0o600)
        kind = str(assignment["case_kind"])
        checkpoint = {
            "schema_version": 1,
            "workflow_version": WORKFLOW_VERSION,
            "status": CHECKPOINT_STATUS,
            "pass_id": manifest["pass_id"],
            "anonymous_review_id": review_id,
            "case_kind": kind,
            "assignment_sha256": assignment["assignment_sha256"],
            "composite_sha256": assignment["composite_sha256"],
            "request_sha256": object_sha256(descriptor),
            "response_schema_sha256": object_sha256(response_json_schema(kind)),
            "model": MODEL,
            "observed_model": result["observed_model"],
            "raw_response": relative_file_ref(output_root, raw_path),
            "model_content_sha256": hashlib.sha256(result["model_content"].encode("utf-8")).hexdigest(),
            "normalized": result["normalized"],
        }
        require(tuple(checkpoint) == CHECKPOINT_FIELDS, "internal checkpoint schema drift")
        checkpoint_path = output_root / "checkpoints" / f"{review_id}.json"
        write_json_exclusive(checkpoint_path, checkpoint, mode=0o600)
        successful_by_index[index] = relative_file_ref(output_root, checkpoint_path)

    def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate_one(
            row,
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
        )

    if workers == 1:
        for index, row in enumerate(selected):
            persist(index, row, evaluate(row))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate, row): (index, row)
                for index, row in enumerate(selected)
            }
            for future in as_completed(futures):
                index, row = futures[future]
                persist(index, row, future.result())

    checkpoint_refs = [successful_by_index[index] for index in sorted(successful_by_index)]
    error_refs = [errors_by_index[index] for index in sorted(errors_by_index)]
    summary = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": (
            "completed_checkpoint_shard"
            if not error_refs
            else "completed_with_infrastructure_errors_requires_fresh_retry"
        ),
        "pass_id": manifest["pass_id"],
        "selected_count": len(selected),
        "successful_checkpoints": len(checkpoint_refs),
        "infrastructure_errors": len(error_refs),
        "run_registration": relative_file_ref(output_root, registration_path),
        "checkpoints": checkpoint_refs,
        "errors": error_refs,
        "scientific_zero_fallbacks": 0,
    }
    require(tuple(summary) == RUN_SUMMARY_FIELDS, "internal run summary schema drift")
    write_json_exclusive(output_root / "run_summary.json", summary, mode=0o600)
    (output_root / ".incomplete").unlink()
    return summary


def _resolve_ref(root: Path, ref: Mapping[str, Any], label: str) -> Path:
    require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256", "size_bytes"}, f"{label} ref schema differs")
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = root / path
    regular_file(path, label)
    require(path.stat().st_size == ref["size_bytes"] and sha256_file(path) == ref["sha256"], f"{label} changed")
    return path


def _validate_checkpoint(
    *,
    run_root: Path,
    checkpoint_path: Path,
    public_by_id: Mapping[str, Mapping[str, Any]],
    registration: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    checkpoint = _read_json(checkpoint_path, "formal review checkpoint")
    require(tuple(checkpoint) == CHECKPOINT_FIELDS, "formal checkpoint fields differ")
    review_id = str(checkpoint["anonymous_review_id"])
    row = public_by_id.get(review_id)
    require(row is not None, "checkpoint ID is outside public assignment")
    assignment = row["assignment"]
    kind = str(assignment["case_kind"])
    require(
        checkpoint["workflow_version"] == WORKFLOW_VERSION
        and checkpoint["status"] == CHECKPOINT_STATUS
        and checkpoint["pass_id"] == registration["pass_id"]
        and checkpoint["case_kind"] == kind
        and checkpoint["assignment_sha256"] == assignment["assignment_sha256"]
        and checkpoint["composite_sha256"] == assignment["composite_sha256"]
        and checkpoint["model"] == MODEL
        and checkpoint["observed_model"] == MODEL,
        "checkpoint identity/binding differs",
    )
    require(
        checkpoint["response_schema_sha256"]
        == object_sha256(response_json_schema(kind)),
        "checkpoint response schema differs",
    )
    descriptor = _request_descriptor(
        row, str(registration["endpoint"]), checkpoint["observed_model"]
    )
    require(checkpoint["request_sha256"] == object_sha256(descriptor), "checkpoint request binding differs")
    normalized = normalize_model_object(
        {
            **checkpoint["normalized"]["scores"],
            "confidence": checkpoint["normalized"]["confidence"],
            "frame_evidence": {
                field: [
                    {"frame_index": frame, "observation": observation}
                    for frame, observation in zip(
                        checkpoint["normalized"]["evidence_frames"][field],
                        checkpoint["normalized"]["evidence_observations"][field],
                    )
                ]
                for field in _score_fields(kind)
            },
        },
        kind,
    )
    require(normalized == checkpoint["normalized"], "checkpoint normalized structure differs")
    raw_path = _resolve_ref(run_root, checkpoint["raw_response"], "checkpoint raw response")
    require(raw_path.parent.resolve(strict=True) == (run_root / "raw_responses").resolve(strict=True), "checkpoint raw response escaped run root")
    response = _read_json(raw_path, "checkpoint raw response")
    content = responses_output_text(response)
    require(hashlib.sha256(content.encode("utf-8")).hexdigest() == checkpoint["model_content_sha256"], "checkpoint model content changed")
    reparsed = normalize_model_object(strict_model_json(content), kind)
    require(reparsed == normalized, "checkpoint differs from raw Responses output")
    return review_id, checkpoint, normalized


def _expected_selected_rows(
    registration: Mapping[str, Any], public_rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    if registration["selection_mode"] == "ordered_shard":
        start = registration["selected_start"]
        limit = registration["selected_limit"]
        require(type(start) is int, "ordered shard start is invalid")
        return _select_rows(public_rows, start, limit)
    require(
        registration["selection_mode"] == "fresh_retry_of_infrastructure_errors"
        and registration["selected_start"] is None
        and registration["selected_limit"] is None
        and isinstance(registration["retry_sources"], list)
        and registration["retry_sources"],
        "run selection mode is invalid",
    )
    selected = set(registration["selected_review_ids"])
    require(len(selected) == registration["selected_count"], "retry selected IDs contain duplicates")
    return [
        row
        for row in public_rows
        if row["assignment"]["anonymous_review_id"] in selected
    ]


def _validate_real_run(
    *,
    run_root: Path,
    assignments_path: Path,
    blank_scores_path: Path,
    manifest: Mapping[str, Any],
    public_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[Path], list[Path]]:
    require(run_root.is_dir() and not run_root.is_symlink(), f"run root is missing/unsafe: {run_root}")
    require(not (run_root / ".incomplete").exists(), f"run root is incomplete: {run_root}")
    require(
        {path.name for path in run_root.iterdir()}
        == {"run_registration.json", "run_summary.json", "checkpoints", "raw_responses", "errors"},
        f"real run root contains unexpected entries: {run_root}",
    )
    for name in ("checkpoints", "raw_responses", "errors"):
        path = run_root / name
        require(path.is_dir() and not path.is_symlink(), f"run directory is unsafe: {path}")
    registration_path = run_root / "run_registration.json"
    summary_path = run_root / "run_summary.json"
    registration = _read_json(registration_path, "run registration")
    summary = _read_json(summary_path, "run summary")
    require(tuple(registration) == RUN_REGISTRATION_FIELDS, "run registration fields differ")
    require(tuple(summary) == RUN_SUMMARY_FIELDS, "run summary fields differ")
    require(
        registration["workflow_version"] == WORKFLOW_VERSION
        and registration["status"] == "registered_fresh_public_pass_shard"
        and registration["pass_id"] == manifest["pass_id"]
        and registration["package_protocol"] == PACKAGE_PROTOCOL
        and registration["package_schema"] == 1
        and registration["evaluation_code_registry_sha256"]
        == manifest["evaluation_code_registry_sha256"]
        and registration["dry_run"] is False,
        "run registration identity differs",
    )
    require(
        registration["transport"] == "openai_compatible_responses_api"
        and registration["model"] == MODEL
        and registration["reasoning_effort"] == REASONING_EFFORT
        and registration["temperature"] == TEMPERATURE
        and registration["max_output_tokens"] == MAX_OUTPUT_TOKENS
        and registration["store"] is False
        and registration["response_schemas_sha256"] == response_schemas_sha256(),
        "run Responses contract differs",
    )
    require(responses_endpoint(registration["endpoint"]) == registration["endpoint"], "registered endpoint is not /v1/responses")
    for key, current_path in (
        ("pass_manifest", manifest["path"]),
        ("assignments", assignments_path),
        ("blank_scores", blank_scores_path),
    ):
        bound = _resolve_ref(run_root, registration[key], f"registered {key}")
        require(bound.resolve() == current_path.resolve(), f"run binds a different {key}")
    expected_rows = _expected_selected_rows(registration, public_rows)
    expected_ids = [row["assignment"]["anonymous_review_id"] for row in expected_rows]
    require(expected_ids == registration["selected_review_ids"], "run selected IDs differ from registered selection")
    require(len(expected_rows) == registration["selected_count"], "run selected count differs")
    expected_inventory = object_sha256(
        [
            {
                "anonymous_review_id": row["assignment"]["anonymous_review_id"],
                "assignment_sha256": row["assignment"]["assignment_sha256"],
                "composite_sha256": row["assignment"]["composite_sha256"],
            }
            for row in expected_rows
        ]
    )
    require(expected_inventory == registration["selected_inventory_sha256"], "run selected inventory differs")
    require(
        summary["workflow_version"] == WORKFLOW_VERSION
        and summary["pass_id"] == manifest["pass_id"]
        and summary["status"]
        in {
            "completed_checkpoint_shard",
            "completed_with_infrastructure_errors_requires_fresh_retry",
        }
        and summary["selected_count"] == registration["selected_count"]
        and summary["scientific_zero_fallbacks"] == 0,
        "run summary identity/count differs",
    )
    require(_resolve_ref(run_root, summary["run_registration"], "summary registration").resolve() == registration_path.resolve(), "run summary binds another registration")
    require(
        summary["successful_checkpoints"] == len(summary["checkpoints"])
        and summary["infrastructure_errors"] == len(summary["errors"])
        and summary["successful_checkpoints"] + summary["infrastructure_errors"] == summary["selected_count"],
        "run summary artifact counts differ",
    )
    checkpoint_paths = [_resolve_ref(run_root, ref, "summary checkpoint") for ref in summary["checkpoints"]]
    error_paths = [_resolve_ref(run_root, ref, "summary infrastructure error") for ref in summary["errors"]]
    require({path.resolve() for path in checkpoint_paths} == {path.resolve() for path in (run_root / "checkpoints").iterdir()}, "checkpoint directory differs from summary")
    require({path.resolve() for path in error_paths} == {path.resolve() for path in (run_root / "errors").iterdir()}, "error directory differs from summary")
    return registration, summary, checkpoint_paths, error_paths


def merge_checkpoints(
    *,
    assignments_path: Path,
    blank_scores_path: Path,
    run_roots: Sequence[Path],
    output_root: Path,
    expected_items: int | None = EXPECTED_ITEMS,
) -> dict[str, Any]:
    require(run_roots, "merge requires at least one run root")
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only merge output exists: {output_root}")
    manifest, public_rows = load_public_pass(
        assignments_path, blank_scores_path, expected_items=expected_items
    )
    public_by_id = {
        row["assignment"]["anonymous_review_id"]: row for row in public_rows
    }
    checkpoints: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    configs: set[tuple[Any, ...]] = set()
    source_runs: list[dict[str, Any]] = []
    for run_root in run_roots:
        registration, _summary, checkpoint_paths, error_paths = _validate_real_run(
            run_root=run_root,
            assignments_path=assignments_path,
            blank_scores_path=blank_scores_path,
            manifest=manifest,
            public_rows=public_rows,
        )
        configs.add(
            (
                registration["proxy_version"],
                registration["endpoint"],
                registration["model"],
                registration["reasoning_effort"],
                registration["temperature"],
                registration["max_output_tokens"],
                json.dumps(registration["response_schemas_sha256"], sort_keys=True),
                registration["evaluation_code_registry_sha256"],
            )
        )
        selected_ids = set(registration["selected_review_ids"])
        successful_in_run: set[str] = set()
        raw_paths: set[Path] = set()
        for checkpoint_path in checkpoint_paths:
            review_id, checkpoint, normalized = _validate_checkpoint(
                run_root=run_root,
                checkpoint_path=checkpoint_path,
                public_by_id=public_by_id,
                registration=registration,
            )
            require(review_id in selected_ids and review_id not in successful_in_run, "checkpoint is outside/duplicated within its run")
            require(review_id not in checkpoints, f"duplicate successful checkpoint for {review_id}")
            successful_in_run.add(review_id)
            raw_paths.add(_resolve_ref(run_root, checkpoint["raw_response"], "checkpoint raw response").resolve())
            checkpoints[review_id] = (checkpoint_path, checkpoint, normalized)
        failed_in_run: set[str] = set()
        for error_path in error_paths:
            error = _read_json(error_path, "infrastructure error")
            require(tuple(error) == ERROR_FIELDS, "infrastructure error fields differ")
            review_id = str(error["anonymous_review_id"])
            row = public_by_id.get(review_id)
            require(row is not None and review_id in selected_ids and review_id not in failed_in_run and review_id not in successful_in_run, "infrastructure error ID is invalid/duplicate/successful")
            assignment = row["assignment"]
            require(
                error["workflow_version"] == WORKFLOW_VERSION
                and error["status"] == INFRASTRUCTURE_ERROR_STATUS
                and error["pass_id"] == manifest["pass_id"]
                and error["case_kind"] == assignment["case_kind"]
                and error["assignment_sha256"] == assignment["assignment_sha256"]
                and error["composite_sha256"] == assignment["composite_sha256"]
                and not ({*CAUSAL_FIELDS, *SPECIFICITY_FIELDS} & set(error)),
                "infrastructure error identity or scientific-score boundary differs",
            )
            require(
                error["observed_model"] is None
                or isinstance(error["observed_model"], str),
                "infrastructure error observed model is invalid",
            )
            descriptor = _request_descriptor(
                row, registration["endpoint"], error["observed_model"]
            )
            require(error["request_sha256"] == object_sha256(descriptor), "infrastructure error request binding differs")
            if error["raw_response"] is None:
                require(
                    error["model_content_sha256"] is None
                    and error["observed_model"] is None,
                    "transport error has response metadata without a raw response",
                )
            else:
                raw_path = _resolve_ref(run_root, error["raw_response"], "infrastructure-error raw response")
                response = _read_json(raw_path, "infrastructure-error raw response")
                raw_model = response.get("model")
                require(
                    (raw_model if isinstance(raw_model, str) else None)
                    == error["observed_model"],
                    "infrastructure-error observed model differs from raw response",
                )
                content = ""
                try:
                    content = responses_output_text(response)
                except FormalReviewTransportError:
                    pass
                require(hashlib.sha256(content.encode("utf-8")).hexdigest() == error["model_content_sha256"], "infrastructure-error model content changed")
                raw_paths.add(raw_path.resolve())
            failed_in_run.add(review_id)
        actual_raw = {path.resolve() for path in (run_root / "raw_responses").iterdir()}
        require(raw_paths == actual_raw, "raw response directory differs from checkpoints/errors")
        source_runs.append(
            {
                "run_registration": file_ref(run_root / "run_registration.json"),
                "run_summary": file_ref(run_root / "run_summary.json"),
            }
        )
    require(len(configs) == 1, "merged runs use different Responses/proxy configurations")
    missing = [review_id for review_id in public_by_id if review_id not in checkpoints]
    require(not missing, f"merge is missing {len(missing)} schema-valid checkpoints")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    completed_rows: list[dict[str, Any]] = []
    raw_registry: dict[str, Any] = {}
    for public in public_rows:
        assignment = public["assignment"]
        review_id = assignment["anonymous_review_id"]
        checkpoint_path, checkpoint, normalized = checkpoints[review_id]
        completed = {
            "anonymous_review_id": review_id,
            "assignment_sha256": assignment["assignment_sha256"],
            "case_kind": assignment["case_kind"],
            **normalized,
            "unusable_reason": "",
            "status": "completed",
        }
        require(tuple(completed) == SCORE_ROW_FIELDS, "internal completed-score schema drift")
        completed_rows.append(completed)
        raw_registry[review_id] = {
            "checkpoint": file_ref(checkpoint_path),
            "raw_response": file_ref(
                _resolve_ref(
                    checkpoint_path.parent.parent,
                    checkpoint["raw_response"],
                    "merged raw response",
                )
            ),
            "model": checkpoint["model"],
            "observed_model": checkpoint["observed_model"],
        }
    completed_path = output_root / "completed_scores.jsonl"
    write_bytes_exclusive(
        completed_path,
        b"".join(canonical_json_bytes(row) for row in completed_rows),
        mode=0o600,
    )
    registry_path = output_root / "raw_response_registry.json"
    write_json_exclusive(registry_path, raw_registry, mode=0o600)
    config = next(iter(configs))
    merge_manifest = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "complete_schema_valid_public_pass_review",
        "pass_id": manifest["pass_id"],
        "evaluation_code_registry_sha256": manifest[
            "evaluation_code_registry_sha256"
        ],
        "row_count": len(completed_rows),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "proxy_version": config[0],
        "endpoint": config[1],
        "pass_manifest": file_ref(manifest["path"]),
        "assignments": file_ref(assignments_path),
        "blank_scores": file_ref(blank_scores_path),
        "source_runs": source_runs,
        "completed_scores": relative_file_ref(output_root, completed_path),
        "raw_response_registry": relative_file_ref(output_root, registry_path),
        "scientific_zero_fallbacks": 0,
    }
    write_json_exclusive(output_root / "merge_manifest.json", merge_manifest, mode=0o600)
    return merge_manifest


def _resolve_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--blank-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-api", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-config-file", type=Path)
    parser.add_argument("--proxy-version", default="copilot-claude-local")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="run a fresh ordered shard")
    _add_common_run_arguments(review)
    review.add_argument("--start", type=int, default=0)
    review.add_argument("--limit", type=int)
    retry = commands.add_parser("retry", help="retry only infrastructure errors in fresh output")
    _add_common_run_arguments(retry)
    retry.add_argument("--run-root", type=Path, action="append", required=True)
    merge = commands.add_parser("merge", help="merge a complete pass from immutable checkpoints")
    merge.add_argument("--assignments", type=Path, required=True)
    merge.add_argument("--blank-scores", type=Path, required=True)
    merge.add_argument("--run-root", type=Path, action="append", required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    return parser


def _api_configuration(args: argparse.Namespace, project_root: Path) -> tuple[str, str]:
    file_values: dict[str, str] = {}
    if args.api_config_file is not None:
        config_path = _resolve_path(project_root, args.api_config_file)
        regular_file(config_path, "API config file")
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                file_values[key.strip()] = value.strip()
    base_url = (
        args.base_url
        or os.environ.get("VLM_BASE_URL")
        or file_values.get("url")
        or "http://127.0.0.1:4141"
    )
    api_key = (
        args.api_key
        or os.environ.get("VLM_API_KEY")
        or file_values.get("key")
        or "local-proxy-no-secret"
    )
    return base_url, api_key


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        require(project_root.is_dir(), f"project root is missing: {project_root}")
        if args.command == "merge":
            result = merge_checkpoints(
                assignments_path=_resolve_path(project_root, args.assignments),
                blank_scores_path=_resolve_path(project_root, args.blank_scores),
                run_roots=[_resolve_path(project_root, value) for value in args.run_root],
                output_root=_resolve_path(project_root, args.output_root),
            )
            exit_code = 0
        else:
            base_url, api_key = _api_configuration(args, project_root)
            result = run_review(
                assignments_path=_resolve_path(project_root, args.assignments),
                blank_scores_path=_resolve_path(project_root, args.blank_scores),
                output_root=_resolve_path(project_root, args.output_root),
                dry_run=args.dry_run,
                start=getattr(args, "start", 0),
                limit=getattr(args, "limit", None),
                workers=args.workers,
                timeout=args.timeout,
                base_url=base_url,
                api_key=api_key,
                proxy_version=args.proxy_version,
                retry_run_roots=(
                    []
                    if args.command == "review"
                    else [_resolve_path(project_root, value) for value in args.run_root]
                ),
            )
            exit_code = 0 if result["infrastructure_errors"] == 0 else 2
    except (OSError, ValueError, FormalReviewTransportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
