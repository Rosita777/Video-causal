#!/usr/bin/env python3
"""Freeze and score capability-review protocol v2.

Protocol v2 is a post-hoc, pre-treatment amendment.  It preserves the full
49-frame, eight-field atomic review, but aligns row eligibility with the
formal Original eligibility and continuous CES design: partial causal evidence
is admissible, while doubly ambiguous trigger/footprint evidence is not.

The freeze command binds a caller-supplied SHA-256 commitment to the canonical
v2 manifest.  The score command recovers that commitment from the immutable
freeze and verifies every deterministic review artifact before scoring.  This
module never generates or opens video and never emits row-level review data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


MANIFEST_PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v2"
REVIEW_PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_review_v2"
ARTIFACT_STEM = "causal_role_erasure_8mechanism_capability_v2"
EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "8e8cc14a6d327193cb7f270d2386490dc00ee783d9e7d0f76a697837f218dbee"
)

MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "field_mediated_response",
    "material_release",
    "surface_trace",
)
PROMPT_STYLES = ("direct", "natural")

EXPECTED_ROWS = 192
ROWS_PER_MECHANISM = 24
ROWS_PER_STYLE_PER_MECHANISM = 12
MIN_ELIGIBLE_PER_MECHANISM = 12
MIN_ELIGIBLE_PER_STYLE = 6
MIN_ELIGIBLE_SOURCE_COVERAGE = 2
MIN_ELIGIBLE_RECEIVER_COVERAGE = 2
MIN_TRIGGER_FOOTPRINT_SUM = 3

METADATA_FIELDS = (
    "review_id",
    "generation_id",
    "case_id",
    "mechanism",
    "source_id",
    "receiver_id",
    "prompt_style",
    "seed",
    "manifest_row_sha256",
    "video_binding_key",
)
BOOLEAN_SCORE_FIELDS = ("decodable",)
ORDINAL_SCORE_FIELDS = (
    "clean_prefix",
    "source_after16",
    "trigger_visible",
    "footprint_after_trigger",
    "receiver_recognizable",
    "fixed_camera",
    "quality",
)
SCORE_FIELDS = BOOLEAN_SCORE_FIELDS + ORDINAL_SCORE_FIELDS
REVIEW_FIELDS = METADATA_FIELDS + SCORE_FIELDS + ("reviewer_notes",)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "case_id",
        "expected_footprint",
        "fps",
        "generation_id",
        "intended_use",
        "mechanism",
        "method_arm",
        "num_frames",
        "prompt_style",
        "protocol_version",
        "receiver_id",
        "reference_end_exclusive",
        "reference_start_inclusive",
        "seed",
        "source_id",
        "treatment_status",
    }
)

TRIGGER_DEFINITIONS = {
    "water_impact": "visible entry of the source through the water surface",
    "rigid_collision": "visible horizontal contact between the source and the upright receiver",
    "brittle_fracture": "visible downward strike of the intact glass receiver",
    "powder_impact": "visible downward contact with the flat powder bed",
    "elastic_deformation": "visible downward contact that starts receiver deformation",
    "field_mediated_response": "visible fan blades begin spinning while a clear gap remains",
    "material_release": "visible single puncture of the sealed paper pouch",
    "surface_trace": "visible single vertical press into the unmarked surface",
}

MECHANISM_SUBTYPES = {
    "field_mediated_response": "airflow_non_contact_motion",
}

MECHANISM_RECEIVER_SCOPES = {
    "elastic_deformation": "black_round_and_blue_square_trampolines",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def normalize_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    require(
        len(normalized) == 64
        and all(character in "0123456789abcdef" for character in normalized),
        f"{label} must be exactly 64 lowercase hexadecimal characters",
    )
    return normalized


def _require_regular_nonsymlink(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing, non-regular, or symlinked: {path}")


def load_manifest(path: Path, *, expected_sha256: str) -> list[dict[str, str]]:
    _require_regular_nonsymlink(path, "canonical capability v2 manifest")
    expected = normalize_sha256(expected_sha256, "canonical manifest SHA-256")
    require(
        expected == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "canonical capability v2 manifest hash mismatch",
    )
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected, "canonical capability v2 manifest hash mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("canonical capability v2 manifest is invalid JSON") from exc
    require(isinstance(payload, list), "canonical capability v2 manifest root must be a list")
    rows: list[dict[str, str]] = []
    for index, value in enumerate(payload):
        require(isinstance(value, dict), f"manifest row {index} is not an object")
        require(
            REQUIRED_MANIFEST_FIELDS.issubset(value),
            f"manifest row {index} lacks required fields",
        )
        require(
            all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()),
            f"manifest row {index} is not string-valued",
        )
        rows.append(dict(value))
    validate_manifest_rows(rows)
    require(canonical_json_bytes(rows) == raw, "capability v2 manifest is not canonical JSON")
    return rows


def validate_manifest_rows(rows: Sequence[Mapping[str, str]]) -> None:
    require(len(rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} manifest rows")
    require(
        len({row["generation_id"] for row in rows}) == EXPECTED_ROWS,
        "manifest generation IDs are not unique",
    )
    require(
        len({row["seed"] for row in rows}) == EXPECTED_ROWS,
        "manifest generation seeds are not unique",
    )
    by_mechanism: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        require(
            row["protocol_version"] == MANIFEST_PROTOCOL_VERSION,
            "manifest protocol version mismatch",
        )
        require(row["method_arm"] == "original", "capability review is Original-only")
        require(
            row["intended_use"] == "original_capability_screening_only",
            "manifest use is not capability screening",
        )
        require(
            row["treatment_status"] == "pre_method_original_only",
            "manifest is not pre-method Original-only",
        )
        require(row["num_frames"] == "49" and row["fps"] == "8", "video contract drift")
        require(
            row["reference_start_inclusive"] == "0"
            and row["reference_end_exclusive"] == "16",
            "clean-prefix diagnostic interval drift",
        )
        require(row["prompt_style"] in PROMPT_STYLES, "unknown prompt style")
        by_mechanism[row["mechanism"]].append(row)
    require(tuple(by_mechanism) == MECHANISM_ORDER, "manifest mechanism order mismatch")
    for mechanism in MECHANISM_ORDER:
        mechanism_rows = by_mechanism[mechanism]
        require(
            len(mechanism_rows) == ROWS_PER_MECHANISM,
            f"{mechanism}: expected {ROWS_PER_MECHANISM} rows",
        )
        require(
            Counter(row["prompt_style"] for row in mechanism_rows)
            == Counter({"direct": 12, "natural": 12}),
            f"{mechanism}: prompt-style balance mismatch",
        )


def manifest_row_sha256(row: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(row)))


def build_expected_metadata(review_id: str, manifest: Mapping[str, str]) -> dict[str, str]:
    return {
        "review_id": review_id,
        "generation_id": manifest["generation_id"],
        "case_id": manifest["case_id"],
        "mechanism": manifest["mechanism"],
        "source_id": manifest["source_id"],
        "receiver_id": manifest["receiver_id"],
        "prompt_style": manifest["prompt_style"],
        "seed": manifest["seed"],
        "manifest_row_sha256": manifest_row_sha256(manifest),
        "video_binding_key": manifest["generation_id"],
    }


def build_review_rows(
    manifest_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    validate_manifest_rows(manifest_rows)
    output: list[dict[str, str]] = []
    for index, source in enumerate(manifest_rows):
        review_id = f"capv2rev{index:03d}"
        row = {
            **build_expected_metadata(review_id, source),
            **{field: "" for field in SCORE_FIELDS},
            "reviewer_notes": "",
        }
        require(tuple(row) == REVIEW_FIELDS, "internal review field order changed")
        output.append(row)
    validate_blank_review_rows(output, manifest_rows)
    return output


def validate_blank_review_rows(
    rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
) -> None:
    require(len(rows) == EXPECTED_ROWS, "review template row count mismatch")
    require(all(tuple(row) == REVIEW_FIELDS for row in rows), "review template columns are not exact")
    require(
        [row["review_id"] for row in rows]
        == [f"capv2rev{index:03d}" for index in range(EXPECTED_ROWS)],
        "review IDs/order are not exact",
    )
    for row, manifest in zip(rows, manifest_rows):
        expected = build_expected_metadata(row["review_id"], manifest)
        require(
            all(row[field] == expected[field] for field in METADATA_FIELDS),
            "review template metadata differs from canonical manifest",
        )
        require(
            all(row[field] == "" for field in (*SCORE_FIELDS, "reviewer_notes")),
            "review template is not blank",
        )


def review_csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _field_specifications() -> list[dict[str, Any]]:
    return [
        {
            "name": "decodable",
            "type": "boolean",
            "allowed_values": {"0": "false", "1": "true"},
            "eligibility_role": "required_equal_1",
            "instruction": (
                "Score 1 only when exactly 49 sequential frames decode and every frame can be "
                "inspected; otherwise score 0."
            ),
        },
        {
            "name": "clean_prefix",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "clear violation",
                "1": "ambiguous or partly clean",
                "2": "all frames 0-15 clean",
            },
            "eligibility_role": "diagnostic_only_value_does_not_enter_gate",
            "instruction": (
                "Across every frame 0-15, score whether the source and causal footprint are "
                "absent and the receiver is in its specified clean state. This value is public "
                "diagnostic evidence and does not change v2 eligibility."
            ),
        },
        {
            "name": "source_after16",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "source absent after frame 15",
                "1": "source partial or ambiguous after frame 15",
                "2": "source clearly visible after frame 15",
            },
            "eligibility_role": "required_equal_2",
            "instruction": "Judge source visibility across all frames 16-48.",
        },
        {
            "name": "trigger_visible",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "trigger absent or contradicted",
                "1": "trigger partial, occluded, or temporally ambiguous",
                "2": "trigger clearly visible",
            },
            "eligibility_role": "required_at_least_1_and_coupled_with_footprint",
            "instruction": "Use the frozen mechanism-specific trigger definition.",
        },
        {
            "name": "footprint_after_trigger",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "footprint absent or present before the trigger",
                "1": "footprint weak, partial, or onset order ambiguous",
                "2": "expected footprint clear and first visible only after the trigger",
            },
            "eligibility_role": "required_at_least_1_and_coupled_with_trigger",
            "instruction": (
                "Inspect all 49 frames and use the frozen expected footprint. A genuine partial "
                "footprint is accepted by v2 when the coupled trigger/footprint evidence sum is "
                "at least 3."
            ),
        },
        {
            "name": "receiver_recognizable",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "receiver missing, wrong, or unrecognizable",
                "1": "receiver partial or ambiguous",
                "2": "intended receiver clear and recognizable",
            },
            "eligibility_role": "required_at_least_1",
            "instruction": "The intended receiver must remain identifiable throughout the event.",
        },
        {
            "name": "fixed_camera",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "cut or substantial camera motion",
                "1": "minor drift or jitter",
                "2": "locked camera with no cut",
            },
            "eligibility_role": "required_at_least_1",
            "instruction": "Inspect all frames for cuts, reframing, pan, tilt, zoom, or drift.",
        },
        {
            "name": "quality",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "severe corruption or not judgeable",
                "1": "judgeable with material artifacts or temporal incoherence",
                "2": "coherent and fully judgeable",
            },
            "eligibility_role": "required_at_least_1",
            "instruction": "Judge visual and temporal integrity across the complete video.",
        },
    ]


def build_rubric_payload(
    manifest_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    validate_manifest_rows(manifest_rows)
    mechanism_rubrics: dict[str, dict[str, Any]] = {}
    for mechanism in MECHANISM_ORDER:
        footprint_by_receiver: dict[str, str] = {}
        for row in manifest_rows:
            if row["mechanism"] != mechanism:
                continue
            receiver_id = row["receiver_id"]
            footprint = row["expected_footprint"]
            prior = footprint_by_receiver.setdefault(receiver_id, footprint)
            require(
                prior == footprint,
                f"{mechanism}: receiver footprint definition changed across rows",
            )
        require(
            len(footprint_by_receiver) == 2,
            f"{mechanism}: expected exactly two receiver footprint bindings",
        )
        distinct_footprints = set(footprint_by_receiver.values())
        mechanism_rubric: dict[str, Any] = {
            "trigger_definition": TRIGGER_DEFINITIONS[mechanism],
        }
        if len(distinct_footprints) == 1:
            mechanism_rubric["expected_footprint"] = next(iter(distinct_footprints))
        else:
            mechanism_rubric["expected_footprint_by_receiver_id"] = footprint_by_receiver
        if mechanism in MECHANISM_SUBTYPES:
            mechanism_rubric["subtype"] = MECHANISM_SUBTYPES[mechanism]
        if mechanism in MECHANISM_RECEIVER_SCOPES:
            mechanism_rubric["receiver_scope"] = MECHANISM_RECEIVER_SCOPES[mechanism]
        mechanism_rubrics[mechanism] = mechanism_rubric
    return {
        "schema_version": 2,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "manifest_protocol_version": MANIFEST_PROTOCOL_VERSION,
        "protocol_amendment": {
            "status": "post_hoc_pre_treatment_amendment",
            "v1_is_unchanged": True,
            "treatment_outputs_may_not_be_inspected_before_freeze": True,
            "reason": (
                "The primary formal result is continuous CES, where clear-to-partial change is "
                "valid evidence; capability eligibility therefore accepts partial footprint or "
                "trigger evidence without requiring all eight atomic fields to equal 2."
            ),
            "anti_guarantee_policy": (
                "One uniform row rule and one uniform mechanism rule apply to all eight "
                "mechanisms; rows, identities, prompt styles, and seeds may not be deleted or "
                "replaced to manufacture a pass."
            ),
            "field_mediated_subtype": "airflow_non_contact_motion",
            "elastic_receiver_scope": "black_round_and_blue_square_trampolines",
        },
        "scope": {
            "purpose": "pre_method_original_capability_gate_only",
            "method_comparison_authorized": False,
            "training_authorized": False,
            "treatment_generation_authorized": False,
        },
        "video_contract": {
            "required_frame_count": 49,
            "required_fps": 8,
            "review_frame_indices_inclusive": [0, 48],
            "all_49_frames_must_be_inspected": True,
            "clean_prefix_frame_indices_inclusive": [0, 15],
            "post_prefix_frame_indices_inclusive": [16, 48],
            "missing_or_decode_failure_policy": "ineligible",
        },
        "video_binding": {
            "template_field": "video_binding_key",
            "frozen_value": "generation_id",
            "path_and_video_sha256_source": "later_frozen_generation_manifest",
            "unbound_template_does_not_authorize_review": True,
        },
        "atomic_fields": _field_specifications(),
        "eligibility": {
            "operator": "all_required_thresholds_and_coupled_evidence",
            "complete_eight_field_record_required": True,
            "required_thresholds": {
                "decodable": {"operator": "equal", "value": 1},
                "source_after16": {"operator": "equal", "value": 2},
                "trigger_visible": {"operator": "at_least", "value": 1},
                "footprint_after_trigger": {"operator": "at_least", "value": 1},
                "receiver_recognizable": {"operator": "at_least", "value": 1},
                "fixed_camera": {"operator": "at_least", "value": 1},
                "quality": {"operator": "at_least", "value": 1},
            },
            "coupled_requirement": {
                "fields": ["trigger_visible", "footprint_after_trigger"],
                "operator": "sum_at_least",
                "value": MIN_TRIGGER_FOOTPRINT_SUM,
                "purpose": "exclude rows where both causal trigger and effect are ambiguous",
            },
            "diagnostic_only_fields": ["clean_prefix"],
            "partial_effect_accepted": True,
            "all_fields_equal_2_required": False,
            "blank_atomic_value": "ineligible_incomplete_record",
            "unknown_atomic_value": "invalid_input",
        },
        "formal_metric_alignment": {
            "source_after16": "formal Original source_visibility = 2",
            "footprint_after_trigger": "formal Original footprint_visibility >= 1",
            "receiver_recognizable": "formal Original receiver_preservation >= 1",
            "quality": "formal Original video_quality >= 1",
            "capability_only_structural_guards": [
                "decodable",
                "trigger_visible",
                "fixed_camera",
            ],
            "clean_prefix": "reported diagnostic only; not a v2 gate input",
            "formal_E_b_unchanged": True,
            "formal_CES_unchanged": True,
            "formal_cases_recompute_E_b_from_same_backbone_original": True,
            "capability_rows_never_substitute_for_formal_case_E_b": True,
            "continuous_primary_endpoint": True,
            "strict_success_is_secondary_only": True,
        },
        "mechanisms": mechanism_rubrics,
        "review_workflow": {
            "reviewer_a": "independent_full_49_frame_atomic_review",
            "reviewer_b": "independent_full_49_frame_atomic_review",
            "reviewers_must_not_share_scores": True,
            "third_reviewer": "blinded_atomic_adjudication_for_every_disagreement",
            "ordinal_resolution": "median_of_reviewer_a_reviewer_b_and_third_reviewer",
            "boolean_resolution": "majority_of_reviewer_a_reviewer_b_and_third_reviewer",
            "canonical_adjudicated_csv_required_before_scoring": True,
            "free_text_is_never_used_by_the_gate": True,
        },
        "mechanism_gate": {
            "total_rows_required": ROWS_PER_MECHANISM,
            "minimum_eligible_rows": MIN_ELIGIBLE_PER_MECHANISM,
            "minimum_eligible_direct_rows": MIN_ELIGIBLE_PER_STYLE,
            "minimum_eligible_natural_rows": MIN_ELIGIBLE_PER_STYLE,
            "minimum_distinct_eligible_source_ids": MIN_ELIGIBLE_SOURCE_COVERAGE,
            "minimum_distinct_eligible_receiver_ids": MIN_ELIGIBLE_RECEIVER_COVERAGE,
            "all_eight_mechanisms_must_pass": True,
        },
        "gate_output": {
            "granularity": "aggregate_only",
            "row_level_output": False,
            "free_text_output": False,
            "clean_prefix_distribution_is_diagnostic_only": True,
            "mechanism_weighting": "equal_one_eighth",
        },
    }


def artifact_paths(data_output_dir: Path) -> dict[str, Path]:
    return {
        "review_template": data_output_dir / f"{ARTIFACT_STEM}_review_template.csv",
        "review_rubric": data_output_dir / f"{ARTIFACT_STEM}_review_rubric.json",
        "review_freeze": data_output_dir / f"{ARTIFACT_STEM}_review_freeze.json",
    }


def build_freeze_payload(
    *,
    canonical_manifest_name: str,
    canonical_manifest_sha256: str,
    review_template_sha256: str,
    review_rubric_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "status": "frozen_blank_before_treatment_generation_and_review",
        "protocol_amendment_status": "post_hoc_pre_treatment_amendment",
        "counts": {
            "rows": EXPECTED_ROWS,
            "mechanisms": len(MECHANISM_ORDER),
            "rows_per_mechanism": ROWS_PER_MECHANISM,
        },
        "artifacts": {
            "canonical_manifest": {
                "name": canonical_manifest_name,
                "sha256": canonical_manifest_sha256,
            },
            "review_template": {
                "name": f"{ARTIFACT_STEM}_review_template.csv",
                "sha256": review_template_sha256,
            },
            "review_rubric": {
                "name": f"{ARTIFACT_STEM}_review_rubric.json",
                "sha256": review_rubric_sha256,
            },
        },
        "checks": {
            "template_exact_192_rows": True,
            "template_scores_and_notes_blank": True,
            "metadata_bound_to_canonical_manifest": True,
            "canonical_manifest_sha256_bound": True,
            "video_binding_deferred_to_generation_manifest": True,
            "all_49_frames_required": True,
            "eight_atomic_fields_required": True,
            "clean_prefix_diagnostic_only": True,
            "partial_effect_accepted": True,
            "aggregate_only_gate": True,
            "all_eight_mechanisms_required": True,
        },
    }


def build_artifact_payloads(
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    canonical_manifest_name: str,
    canonical_manifest_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    manifest_digest = normalize_sha256(canonical_manifest_sha256, "canonical manifest SHA-256")
    require(
        manifest_digest == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "canonical capability v2 manifest hash mismatch",
    )
    template_raw = review_csv_bytes(build_review_rows(manifest_rows))
    rubric_raw = pretty_json_bytes(build_rubric_payload(manifest_rows))
    freeze = build_freeze_payload(
        canonical_manifest_name=canonical_manifest_name,
        canonical_manifest_sha256=manifest_digest,
        review_template_sha256=sha256_bytes(template_raw),
        review_rubric_sha256=sha256_bytes(rubric_raw),
    )
    return {
        "review_template": template_raw,
        "review_rubric": rubric_raw,
        "review_freeze": pretty_json_bytes(freeze),
    }, freeze


def _plain_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent is unsafe")


def write_bytes_exclusive_atomic(path: Path, raw: bytes) -> tuple[int, int]:
    """Publish bytes with an atomic no-replace hard link and return inode identity."""

    _plain_output_parent(path)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
        info = os.lstat(path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return info.st_dev, info.st_ino
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if (info.st_dev, info.st_ino) == identity:
        path.unlink()


def write_artifacts_exclusive_atomic(
    paths: Mapping[str, Path], payloads: Mapping[str, bytes]
) -> None:
    require(set(paths) == set(payloads), "artifact path/payload mismatch")
    for path in paths.values():
        _plain_output_parent(path)
    collisions = [str(path) for path in paths.values() if os.path.lexists(path)]
    if collisions:
        raise FileExistsError(
            "refusing to overwrite existing artifact(s): " + ", ".join(sorted(collisions))
        )
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for name in ("review_template", "review_rubric", "review_freeze"):
            identity = write_bytes_exclusive_atomic(paths[name], payloads[name])
            created.append((paths[name], identity))
    except BaseException:
        for path, identity in reversed(created):
            _unlink_if_identity(path, identity)
        raise


def _read_csv_exact(
    path: Path, expected_header: Sequence[str], label: str
) -> list[dict[str, str]]:
    _require_regular_nonsymlink(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == tuple(expected_header), f"{label} header is not exact")
        return [dict(row) for row in reader]


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular_nonsymlink(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    require(isinstance(payload, dict), f"{label} root must be an object")
    return payload


def _manifest_commitment_from_freeze(path: Path) -> tuple[str, str]:
    freeze = _load_json_object(path, "review freeze")
    require(
        freeze.get("review_protocol_version") == REVIEW_PROTOCOL_VERSION,
        "review freeze protocol version mismatch",
    )
    try:
        manifest = freeze["artifacts"]["canonical_manifest"]
        name = manifest["name"]
        digest = manifest["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("review freeze lacks canonical manifest commitment") from exc
    require(isinstance(name, str), "review freeze canonical manifest name is invalid")
    require(isinstance(digest, str), "review freeze canonical manifest hash is invalid")
    normalized = normalize_sha256(digest, "review freeze canonical manifest SHA-256")
    require(
        normalized == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "review freeze canonical manifest hash differs from v2 protocol commitment",
    )
    return name, normalized


def verify_frozen_review_inputs(
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    canonical_manifest_name: str,
    canonical_manifest_sha256: str,
    review_template_path: Path,
    review_rubric_path: Path,
    review_freeze_path: Path,
) -> list[dict[str, str]]:
    expected_payloads, expected_freeze = build_artifact_payloads(
        manifest_rows,
        canonical_manifest_name=canonical_manifest_name,
        canonical_manifest_sha256=canonical_manifest_sha256,
    )
    _require_regular_nonsymlink(review_template_path, "review template")
    _require_regular_nonsymlink(review_rubric_path, "review rubric")
    require(
        review_template_path.read_bytes() == expected_payloads["review_template"],
        "review template differs from deterministic frozen bytes",
    )
    require(
        review_rubric_path.read_bytes() == expected_payloads["review_rubric"],
        "review rubric differs from deterministic frozen bytes",
    )
    freeze = _load_json_object(review_freeze_path, "review freeze")
    require(freeze == expected_freeze, "review freeze differs from deterministic commitment")
    require(
        sha256_file(review_template_path) == freeze["artifacts"]["review_template"]["sha256"],
        "review template hash mismatch",
    )
    require(
        sha256_file(review_rubric_path) == freeze["artifacts"]["review_rubric"]["sha256"],
        "review rubric hash mismatch",
    )
    rows = _read_csv_exact(review_template_path, REVIEW_FIELDS, "review template")
    validate_blank_review_rows(rows, manifest_rows)
    return rows


def validate_adjudicated_rows(
    rows: Sequence[Mapping[str, str]],
    blank_rows: Sequence[Mapping[str, str]],
) -> None:
    require(len(rows) == EXPECTED_ROWS, "canonical adjudicated review must cover exactly 192 rows")
    require(
        all(tuple(row) == REVIEW_FIELDS for row in rows),
        "canonical adjudicated review columns are not exact",
    )
    expected_ids = [row["review_id"] for row in blank_rows]
    actual_ids = [row["review_id"] for row in rows]
    require(actual_ids == expected_ids, "canonical adjudicated review IDs/order are not exact")
    require(len(set(actual_ids)) == EXPECTED_ROWS, "canonical adjudicated review IDs duplicate")
    for row, blank in zip(rows, blank_rows):
        require(
            all(row[field] == blank[field] for field in METADATA_FIELDS),
            f"{row.get('review_id', '<unknown>')}: frozen metadata changed",
        )
        for field in BOOLEAN_SCORE_FIELDS:
            require(
                row[field] in {"", "0", "1"},
                f"{row['review_id']}: invalid boolean score in {field}",
            )
        for field in ORDINAL_SCORE_FIELDS:
            require(
                row[field] in {"", "0", "1", "2"},
                f"{row['review_id']}: invalid ordinal score in {field}",
            )


def row_is_eligible(row: Mapping[str, str]) -> bool:
    if any(row[field] == "" for field in SCORE_FIELDS):
        return False
    trigger = int(row["trigger_visible"])
    footprint = int(row["footprint_after_trigger"])
    return (
        row["decodable"] == "1"
        and row["source_after16"] == "2"
        and trigger >= 1
        and footprint >= 1
        and trigger + footprint >= MIN_TRIGGER_FOOTPRINT_SUM
        and int(row["receiver_recognizable"]) >= 1
        and int(row["fixed_camera"]) >= 1
        and int(row["quality"]) >= 1
    )


def _score_distribution(
    rows: Sequence[Mapping[str, str]], field: str
) -> dict[str, int]:
    return {
        "0": sum(row[field] == "0" for row in rows),
        "1": sum(row[field] == "1" for row in rows),
        "2": sum(row[field] == "2" for row in rows),
        "missing": sum(row[field] == "" for row in rows),
    }


def score_adjudicated_rows(
    rows: Sequence[Mapping[str, str]],
    blank_rows: Sequence[Mapping[str, str]],
    *,
    canonical_manifest_sha256: str,
    adjudicated_sha256: str,
    review_freeze_sha256: str,
) -> dict[str, Any]:
    validate_adjudicated_rows(rows, blank_rows)
    per_mechanism: dict[str, Any] = {}
    overall_eligible = 0
    overall_missing = 0
    overall_decode_fail = 0
    mechanisms_passing = 0
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mechanism"]].append(row)
    require(tuple(grouped) == MECHANISM_ORDER, "adjudicated mechanism order mismatch")
    for mechanism in MECHANISM_ORDER:
        mechanism_rows = grouped[mechanism]
        require(len(mechanism_rows) == ROWS_PER_MECHANISM, "mechanism row count mismatch")
        eligible_rows = [row for row in mechanism_rows if row_is_eligible(row)]
        missing_rows = [
            row for row in mechanism_rows if any(row[field] == "" for field in SCORE_FIELDS)
        ]
        decode_fail_rows = [row for row in mechanism_rows if row["decodable"] == "0"]
        style_counts = {
            style: sum(
                row_is_eligible(row)
                for row in mechanism_rows
                if row["prompt_style"] == style
            )
            for style in PROMPT_STYLES
        }
        source_coverage = len({row["source_id"] for row in eligible_rows})
        receiver_coverage = len({row["receiver_id"] for row in eligible_rows})
        gates = {
            "eligible_rows_at_least_12": len(eligible_rows) >= MIN_ELIGIBLE_PER_MECHANISM,
            "eligible_direct_rows_at_least_6": style_counts["direct"] >= MIN_ELIGIBLE_PER_STYLE,
            "eligible_natural_rows_at_least_6": style_counts["natural"] >= MIN_ELIGIBLE_PER_STYLE,
            "eligible_source_coverage_at_least_2": source_coverage >= MIN_ELIGIBLE_SOURCE_COVERAGE,
            "eligible_receiver_coverage_at_least_2": (
                receiver_coverage >= MIN_ELIGIBLE_RECEIVER_COVERAGE
            ),
        }
        mechanism_pass = all(gates.values())
        mechanisms_passing += int(mechanism_pass)
        overall_eligible += len(eligible_rows)
        overall_missing += len(missing_rows)
        overall_decode_fail += len(decode_fail_rows)
        per_mechanism[mechanism] = {
            "total_rows": len(mechanism_rows),
            "eligible_rows": len(eligible_rows),
            "ineligible_rows": len(mechanism_rows) - len(eligible_rows),
            "missing_atomic_rows": len(missing_rows),
            "decode_fail_rows": len(decode_fail_rows),
            "eligible_by_prompt_style": style_counts,
            "distinct_eligible_source_ids": source_coverage,
            "distinct_eligible_receiver_ids": receiver_coverage,
            "diagnostics": {
                "clean_prefix_distribution": _score_distribution(
                    mechanism_rows, "clean_prefix"
                ),
                "clean_prefix_enters_gate": False,
            },
            "gates": gates,
            "pass": mechanism_pass,
        }
    overall_pass = mechanisms_passing == len(MECHANISM_ORDER)
    return {
        "schema_version": 2,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "status": "pass" if overall_pass else "fail",
        "scope": "aggregate_only_original_capability_gate",
        "protocol_amendment_status": "post_hoc_pre_treatment_amendment",
        "input_sha256": {
            "canonical_manifest": normalize_sha256(
                canonical_manifest_sha256, "canonical manifest SHA-256"
            ),
            "review_freeze": normalize_sha256(review_freeze_sha256, "review freeze SHA-256"),
            "canonical_adjudicated_review": normalize_sha256(
                adjudicated_sha256, "canonical adjudicated review SHA-256"
            ),
        },
        "thresholds": {
            "eligible_rows_per_mechanism": MIN_ELIGIBLE_PER_MECHANISM,
            "eligible_rows_per_prompt_style_per_mechanism": MIN_ELIGIBLE_PER_STYLE,
            "distinct_eligible_source_ids_per_mechanism": MIN_ELIGIBLE_SOURCE_COVERAGE,
            "distinct_eligible_receiver_ids_per_mechanism": MIN_ELIGIBLE_RECEIVER_COVERAGE,
            "row_eligibility": {
                "decodable_equal": 1,
                "source_after16_equal": 2,
                "trigger_visible_at_least": 1,
                "footprint_after_trigger_at_least": 1,
                "trigger_plus_footprint_at_least": MIN_TRIGGER_FOOTPRINT_SUM,
                "receiver_recognizable_at_least": 1,
                "fixed_camera_at_least": 1,
                "quality_at_least": 1,
                "clean_prefix": "diagnostic_only",
            },
            "all_atomic_gates_required_per_row": False,
            "complete_eight_field_record_required": True,
            "all_mechanisms_required": True,
        },
        "aggregate": {
            "total_rows": EXPECTED_ROWS,
            "eligible_rows": overall_eligible,
            "ineligible_rows": EXPECTED_ROWS - overall_eligible,
            "missing_atomic_rows": overall_missing,
            "decode_fail_rows": overall_decode_fail,
            "mechanisms_total": len(MECHANISM_ORDER),
            "mechanisms_passing": mechanisms_passing,
            "equal_mechanism_weight": "1/8",
            "pass": overall_pass,
        },
        "per_mechanism": per_mechanism,
        "authorization": {
            "capability_gate_passed": overall_pass,
            "training_authorized": False,
            "treatment_generation_authorized": False,
        },
    }


def freeze_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha256 = normalize_sha256(
        args.canonical_manifest_sha256, "canonical manifest SHA-256"
    )
    rows = load_manifest(args.canonical_manifest, expected_sha256=manifest_sha256)
    payloads, freeze = build_artifact_payloads(
        rows,
        canonical_manifest_name=args.canonical_manifest.name,
        canonical_manifest_sha256=manifest_sha256,
    )
    paths = artifact_paths(args.data_output_dir)
    write_artifacts_exclusive_atomic(paths, payloads)
    return freeze


def score_command(args: argparse.Namespace) -> dict[str, Any]:
    manifest_name, manifest_sha256 = _manifest_commitment_from_freeze(args.review_freeze)
    require(
        args.canonical_manifest.name == manifest_name,
        "canonical manifest filename differs from review freeze commitment",
    )
    rows = load_manifest(args.canonical_manifest, expected_sha256=manifest_sha256)
    blank_rows = verify_frozen_review_inputs(
        rows,
        canonical_manifest_name=manifest_name,
        canonical_manifest_sha256=manifest_sha256,
        review_template_path=args.review_template,
        review_rubric_path=args.review_rubric,
        review_freeze_path=args.review_freeze,
    )
    adjudicated = _read_csv_exact(
        args.canonical_adjudicated_review,
        REVIEW_FIELDS,
        "canonical adjudicated review",
    )
    payload = score_adjudicated_rows(
        adjudicated,
        blank_rows,
        canonical_manifest_sha256=manifest_sha256,
        adjudicated_sha256=sha256_file(args.canonical_adjudicated_review),
        review_freeze_sha256=sha256_file(args.review_freeze),
    )
    write_bytes_exclusive_atomic(args.output, pretty_json_bytes(payload))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze blank v2 template and public rubric")
    freeze.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_manifest.canonical.json",
    )
    freeze.add_argument(
        "--canonical-manifest-sha256",
        default=EXPECTED_CANONICAL_MANIFEST_SHA256,
        help="precommitted SHA-256 of the canonical v2 manifest",
    )
    freeze.add_argument("--data-output-dir", type=Path, default=Path("data"))

    score = subparsers.add_parser("score", help="score one canonical adjudicated v2 CSV")
    score.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_manifest.canonical.json",
    )
    score.add_argument(
        "--review-template",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_review_template.csv",
    )
    score.add_argument(
        "--review-rubric",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_review_rubric.json",
    )
    score.add_argument(
        "--review-freeze",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_review_freeze.json",
    )
    score.add_argument("--canonical-adjudicated-review", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = freeze_command(args) if args.command == "freeze" else score_command(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
