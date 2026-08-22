#!/usr/bin/env python3
"""Freeze and score the eight-mechanism Original capability review.

The review template is intentionally independent of generated media paths.  Its
``video_binding_key`` is the frozen ``generation_id``; a later generation
manifest must bind that key to a regular video file and its digest before a
review package is assembled.  This module never generates or opens video.
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


MANIFEST_PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v1"
REVIEW_PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_review_v1"
ARTIFACT_STEM = "causal_role_erasure_8mechanism_capability_v1"
EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "6d425076a7156aabc9695e6ab4fbe7cf85922261d1282c067fd56d176d05031d"
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
MIN_ELIGIBLE_PER_MECHANISM = 15
MIN_ELIGIBLE_PER_STYLE = 6
MIN_ELIGIBLE_SOURCE_COVERAGE = 2
MIN_ELIGIBLE_RECEIVER_COVERAGE = 2

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
    "rigid_collision": "visible physical contact between source and receiver",
    "brittle_fracture": "visible strike of the intact brittle receiver by the source",
    "powder_impact": "visible source contact with the initially undisturbed powder bed",
    "elastic_deformation": "visible source contact initiating receiver deformation",
    "field_mediated_response": (
        "visible closest approach followed by receiver motion while a gap remains; "
        "direct contact is a trigger failure"
    ),
    "material_release": "visible impact or puncture that opens the intact receiver",
    "surface_trace": "visible source contact and motion along the initially unmarked surface",
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


def _require_regular_nonsymlink(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is missing, non-regular, or symlinked: {path}")


def load_manifest(path: Path) -> list[dict[str, str]]:
    _require_regular_nonsymlink(path, "canonical capability manifest")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    require(
        digest == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "canonical capability manifest hash mismatch",
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("canonical capability manifest is invalid JSON") from exc
    require(isinstance(payload, list), "canonical capability manifest root must be a list")
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
    require(canonical_json_bytes(rows) == raw, "capability manifest is not canonical JSON")
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
            "clean-prefix contract drift",
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


def build_review_rows(
    manifest_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    validate_manifest_rows(manifest_rows)
    output: list[dict[str, str]] = []
    for index, source in enumerate(manifest_rows):
        row = {
            "review_id": f"caprev{index:03d}",
            "generation_id": source["generation_id"],
            "case_id": source["case_id"],
            "mechanism": source["mechanism"],
            "source_id": source["source_id"],
            "receiver_id": source["receiver_id"],
            "prompt_style": source["prompt_style"],
            "seed": source["seed"],
            "manifest_row_sha256": manifest_row_sha256(source),
            "video_binding_key": source["generation_id"],
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
    require(
        all(tuple(row) == REVIEW_FIELDS for row in rows),
        "review template columns are not exact",
    )
    require(
        [row["review_id"] for row in rows]
        == [f"caprev{index:03d}" for index in range(EXPECTED_ROWS)],
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
            "pass_value": "1",
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
            "pass_value": "2",
            "instruction": (
                "Across every frame 0-15, the source and causal footprint are absent and the "
                "receiver is in its specified clean state."
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
            "pass_value": "2",
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
            "pass_value": "2",
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
            "pass_value": "2",
            "instruction": "Inspect all 49 frames and use the frozen expected footprint.",
        },
        {
            "name": "receiver_recognizable",
            "type": "ordinal_0_1_2",
            "allowed_values": {
                "0": "receiver missing, wrong, or unrecognizable",
                "1": "receiver partial or ambiguous",
                "2": "intended receiver clear and recognizable",
            },
            "pass_value": "2",
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
            "pass_value": "2",
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
            "pass_value": "2",
            "instruction": "Judge visual and temporal integrity across the complete video.",
        },
    ]


def build_rubric_payload(
    manifest_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    validate_manifest_rows(manifest_rows)
    footprints: dict[str, str] = {}
    for mechanism in MECHANISM_ORDER:
        values = {
            row["expected_footprint"]
            for row in manifest_rows
            if row["mechanism"] == mechanism
        }
        require(len(values) == 1, f"{mechanism}: footprint definition is not unique")
        footprints[mechanism] = next(iter(values))
    return {
        "schema_version": 1,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "manifest_protocol_version": MANIFEST_PROTOCOL_VERSION,
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
            "operator": "all",
            "required_values": {
                "decodable": "1",
                **{field: "2" for field in ORDINAL_SCORE_FIELDS},
            },
            "blank_atomic_value": "ineligible",
            "unknown_atomic_value": "invalid_input",
        },
        "mechanisms": {
            mechanism: {
                "trigger_definition": TRIGGER_DEFINITIONS[mechanism],
                "expected_footprint": footprints[mechanism],
            }
            for mechanism in MECHANISM_ORDER
        },
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
    canonical_manifest_sha256: str,
    review_template_sha256: str,
    review_rubric_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "status": "frozen_blank_before_generation_and_review",
        "counts": {
            "rows": EXPECTED_ROWS,
            "mechanisms": len(MECHANISM_ORDER),
            "rows_per_mechanism": ROWS_PER_MECHANISM,
        },
        "artifacts": {
            "canonical_manifest": {
                "name": f"{ARTIFACT_STEM}_manifest.canonical.json",
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
            "video_binding_deferred_to_generation_manifest": True,
            "all_49_frames_required": True,
            "aggregate_only_gate": True,
        },
    }


def build_artifact_payloads(
    manifest_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    template_raw = review_csv_bytes(build_review_rows(manifest_rows))
    rubric_raw = pretty_json_bytes(build_rubric_payload(manifest_rows))
    freeze = build_freeze_payload(
        canonical_manifest_sha256=EXPECTED_CANONICAL_MANIFEST_SHA256,
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


def _read_csv_exact(path: Path, expected_header: Sequence[str], label: str) -> list[dict[str, str]]:
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


def verify_frozen_review_inputs(
    manifest_rows: Sequence[Mapping[str, str]],
    *,
    review_template_path: Path,
    review_rubric_path: Path,
    review_freeze_path: Path,
) -> list[dict[str, str]]:
    expected_payloads, expected_freeze = build_artifact_payloads(manifest_rows)
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
        sha256_file(review_template_path)
        == freeze["artifacts"]["review_template"]["sha256"],
        "review template hash mismatch",
    )
    require(
        sha256_file(review_rubric_path)
        == freeze["artifacts"]["review_rubric"]["sha256"],
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
    return row["decodable"] == "1" and all(
        row[field] == "2" for field in ORDINAL_SCORE_FIELDS
    )


def score_adjudicated_rows(
    rows: Sequence[Mapping[str, str]],
    blank_rows: Sequence[Mapping[str, str]],
    *,
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
        missing_rows = [row for row in mechanism_rows if any(row[field] == "" for field in SCORE_FIELDS)]
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
            "eligible_rows_at_least_15": len(eligible_rows) >= MIN_ELIGIBLE_PER_MECHANISM,
            "eligible_direct_rows_at_least_6": style_counts["direct"] >= MIN_ELIGIBLE_PER_STYLE,
            "eligible_natural_rows_at_least_6": style_counts["natural"] >= MIN_ELIGIBLE_PER_STYLE,
            "eligible_source_coverage_at_least_2": source_coverage >= MIN_ELIGIBLE_SOURCE_COVERAGE,
            "eligible_receiver_coverage_at_least_2": receiver_coverage >= MIN_ELIGIBLE_RECEIVER_COVERAGE,
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
            "gates": gates,
            "pass": mechanism_pass,
        }
    overall_pass = mechanisms_passing == len(MECHANISM_ORDER)
    return {
        "schema_version": 1,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "status": "pass" if overall_pass else "fail",
        "scope": "aggregate_only_original_capability_gate",
        "input_sha256": {
            "canonical_manifest": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "review_freeze": review_freeze_sha256,
            "canonical_adjudicated_review": adjudicated_sha256,
        },
        "thresholds": {
            "eligible_rows_per_mechanism": MIN_ELIGIBLE_PER_MECHANISM,
            "eligible_rows_per_prompt_style_per_mechanism": MIN_ELIGIBLE_PER_STYLE,
            "distinct_eligible_source_ids_per_mechanism": MIN_ELIGIBLE_SOURCE_COVERAGE,
            "distinct_eligible_receiver_ids_per_mechanism": MIN_ELIGIBLE_RECEIVER_COVERAGE,
            "all_atomic_gates_required_per_row": True,
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
    rows = load_manifest(args.canonical_manifest)
    payloads, freeze = build_artifact_payloads(rows)
    paths = artifact_paths(args.data_output_dir)
    write_artifacts_exclusive_atomic(paths, payloads)
    return freeze


def score_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_manifest(args.canonical_manifest)
    blank_rows = verify_frozen_review_inputs(
        rows,
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
        adjudicated_sha256=sha256_file(args.canonical_adjudicated_review),
        review_freeze_sha256=sha256_file(args.review_freeze),
    )
    write_bytes_exclusive_atomic(args.output, pretty_json_bytes(payload))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze blank template and rubric")
    freeze.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path("data") / f"{ARTIFACT_STEM}_manifest.canonical.json",
    )
    freeze.add_argument("--data-output-dir", type=Path, default=Path("data"))

    score = subparsers.add_parser("score", help="score one canonical adjudicated CSV")
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
