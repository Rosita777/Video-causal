#!/usr/bin/env python3
"""Build the blinded human audit and freeze canonical seven-mechanism scores.

The workflow has four fresh-only stages: ``build-audit`` creates a public
treatment-blind human template plus a private VLM/audit key, ``expand-audit``
applies the frozen >5% high-confidence-error expansion rule, and ``freeze``
resolves audited atoms while every answer key remains sealed.  Only then does
``freeze-eligibility`` open the committed Original-only key to freeze Original
eligibility and shared-capability subsets.  The full method key remains sealed
until the separate metrics stage.  Method identities never enter the public
human template.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
    import review_causal_role_erasure_7mechanism_formal_v2 as transport
except ModuleNotFoundError:
    from scripts import causal_role_erasure_7mechanism_evaluation_code_registry_v1 as evaluation_code
    from scripts import review_causal_role_erasure_7mechanism_formal_v2 as transport


PROTOCOL = "causal_role_erasure_7m_human_canonicalization_v1"
REVIEW_PROTOCOL = "causal_role_erasure_7m_anonymous_review_v1"
TRANSPORT_WORKFLOW = "causal_role_erasure_7mechanism_formal_responses_transport_v2"
EXPECTED_ITEMS = 2448
EXPECTED_ORIGINAL_KEY_ITEMS = 588
MAIN_STREAMS = (
    "wan_original",
    "matched_control",
    "V4",
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
IDENTIFICATION_STREAMS = ("generic_paraphrase", "bystander_token")
FIELDS_BY_KIND = {
    "causal": (
        "source_visibility",
        "footprint_visibility",
        "receiver_preservation",
        "video_quality",
    ),
    "specificity": (
        "protected_object_visibility",
        "noncausal_role_adherence",
        "receiver_preservation",
        "video_quality",
    ),
}
AUDIT_SEED = 8_202_600
CALIBRATION_FRACTION = 0.10
HIGH_CONFIDENCE_FRACTION = 0.10
LOW_CONFIDENCE_THRESHOLD = 0.75
EXPANSION_ERROR_THRESHOLD = 0.05
SCORE_ROW_FIELDS = {
    "anonymous_review_id",
    "assignment_sha256",
    "case_kind",
    "scores",
    "confidence",
    "evidence_frames",
    "evidence_observations",
    "unusable_reason",
    "status",
}
MERGE_MANIFEST_FIELDS = {
    "schema_version",
    "workflow_version",
    "status",
    "pass_id",
    "evaluation_code_registry_sha256",
    "row_count",
    "model",
    "reasoning_effort",
    "temperature",
    "max_output_tokens",
    "store",
    "proxy_version",
    "endpoint",
    "pass_manifest",
    "assignments",
    "blank_scores",
    "source_runs",
    "completed_scores",
    "raw_response_registry",
    "scientific_zero_fallbacks",
}
PUBLIC_CONTEXT_FIELDS = (
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
)
HUMAN_QUEUE_FIELDS = (
    "audit_id",
    "anonymous_review_id",
    "assignment_sha256",
    "field",
    *PUBLIC_CONTEXT_FIELDS,
    "reviewer_1_score",
    "reviewer_1_notes",
    "reviewer_2_score",
    "reviewer_2_notes",
    "adjudicator_score",
    "adjudicator_notes",
)


class CanonicalizationError(ValueError):
    """The blinded audit or canonical freeze violates the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanonicalizationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalizationError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    regular_file(path, label)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        require(line.strip() == line and line, f"{label} row {index} is blank/noncanonical")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalizationError(f"{label} row {index} is invalid JSON") from exc
        require(isinstance(value, dict), f"{label} row {index} is not an object")
        rows.append(value)
    return rows


def load_csv(path: Path, fields: Sequence[str], label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == tuple(fields), f"{label} header changed")
        return [dict(row) for row in reader]


def write_bytes(path: Path, value: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(mode)


def write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    write_bytes(path, canonical_json_bytes(value), mode)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], mode: int = 0o600) -> None:
    write_bytes(path, b"".join(canonical_json_bytes(dict(row)) for row in rows), mode)


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def file_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def relative_ref(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _resolve_ref(root: Path, ref: Any, label: str, expected: Path | None = None) -> Path:
    require(isinstance(ref, dict) and set(ref) >= {"path", "sha256"}, f"{label} reference malformed")
    value = Path(str(ref["path"]))
    path = (value if value.is_absolute() else root / value)
    regular_file(path, label)
    path = path.resolve()
    if expected is not None:
        require(path == expected.resolve(), f"{label} path mismatch")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA mismatch")
    return path


def _score_fields(kind: str) -> tuple[str, ...]:
    fields = FIELDS_BY_KIND.get(kind)
    require(fields is not None, f"unknown case_kind: {kind}")
    return fields


def validate_score_row(row: Mapping[str, Any], label: str) -> None:
    require(set(row) == SCORE_ROW_FIELDS, f"{label}: score fields changed")
    kind = str(row.get("case_kind", ""))
    fields = _score_fields(kind)
    require(row.get("status") == "completed", f"{label}: score is not completed")
    require(isinstance(row.get("unusable_reason"), str), f"{label}: unusable_reason is invalid")
    for name in ("scores", "confidence", "evidence_frames", "evidence_observations"):
        require(isinstance(row.get(name), dict) and set(row[name]) == set(fields), f"{label}: {name} keys changed")
    require(all(type(value) is int and value in (0, 1, 2) for value in row["scores"].values()), f"{label}: invalid ordinal score")
    require(all(type(value) in (int, float) and 0 <= value <= 1 for value in row["confidence"].values()), f"{label}: invalid confidence")
    for field in fields:
        frames = row["evidence_frames"][field]
        observations = row["evidence_observations"][field]
        require(isinstance(frames, list) and all(type(value) is int and 0 <= value <= 48 for value in frames), f"{label}: invalid evidence frames")
        require(len(frames) == len(set(frames)), f"{label}: duplicate evidence frames")
        require(isinstance(observations, list) and all(isinstance(value, str) and value.strip() for value in observations), f"{label}: invalid evidence observations")


def load_merged_scores(
    path: Path,
    pass_id: str,
    expected_items: int = EXPECTED_ITEMS,
    *,
    expected_assignments_path: Path | None = None,
) -> list[dict[str, Any]]:
    require(path.name == "completed_scores.jsonl", f"pass {pass_id}: unexpected completed-score filename")
    manifest = load_json(path.parent / "merge_manifest.json", f"pass {pass_id} merge manifest")
    require(
        set(manifest) == MERGE_MANIFEST_FIELDS
        and manifest.get("schema_version") == 1
        and manifest.get("workflow_version") == TRANSPORT_WORKFLOW
        and manifest.get("status") == "complete_schema_valid_public_pass_review"
        and manifest.get("pass_id") == pass_id,
        f"pass {pass_id}: merge manifest identity/status mismatch",
    )
    current_registry = evaluation_code.build_registry(
        Path(__file__).resolve().parents[1]
    )
    require(
        manifest.get("evaluation_code_registry_sha256")
        == current_registry["registry_sha256"],
        f"pass {pass_id}: evaluation code registry changed",
    )
    require(
        manifest.get("row_count") == expected_items
        and manifest.get("model") == "gpt-5.6-luna"
        and manifest.get("reasoning_effort") == "low"
        and manifest.get("temperature") == 1.0
        and manifest.get("max_output_tokens") == 1600
        and manifest.get("store") is False
        and isinstance(manifest.get("proxy_version"), str)
        and str(manifest.get("endpoint", "")).endswith("/v1/responses"),
        f"pass {pass_id}: frozen model/transport configuration changed",
    )
    _resolve_ref(path.parent, manifest.get("completed_scores"), f"pass {pass_id} completed scores", path)
    require(manifest.get("scientific_zero_fallbacks") == 0, f"pass {pass_id}: scientific fallback detected")
    assignments_path = _resolve_ref(
        path.parent,
        manifest.get("assignments"),
        f"pass {pass_id} assignments",
        expected_assignments_path,
    )
    blank_scores_path = _resolve_ref(path.parent, manifest.get("blank_scores"), f"pass {pass_id} blank scores")
    pass_manifest_path = _resolve_ref(path.parent, manifest.get("pass_manifest"), f"pass {pass_id} public manifest")
    public_manifest, public_rows = transport.load_public_pass(
        assignments_path,
        blank_scores_path,
        expected_items=expected_items,
    )
    require(Path(public_manifest["path"]).resolve() == pass_manifest_path, f"pass {pass_id}: public manifest binding differs")
    source_runs = manifest.get("source_runs")
    require(isinstance(source_runs, list) and source_runs, f"pass {pass_id}: source run registry is empty")
    for index, source in enumerate(source_runs):
        require(isinstance(source, dict) and set(source) == {"run_registration", "run_summary"}, f"pass {pass_id}: source run {index} schema changed")
        _resolve_ref(path.parent, source["run_registration"], f"pass {pass_id} run {index} registration")
        _resolve_ref(path.parent, source["run_summary"], f"pass {pass_id} run {index} summary")
    registry_path = _resolve_ref(path.parent, manifest.get("raw_response_registry"), f"pass {pass_id} raw response registry")
    registry = load_json(registry_path, f"pass {pass_id} raw response registry")
    rows = load_jsonl(path, f"pass {pass_id} completed scores")
    require(len(rows) == expected_items, f"pass {pass_id}: expected exactly {expected_items} scores")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        validate_score_row(row, f"pass {pass_id} row {index}")
        review_id = str(row.get("anonymous_review_id", ""))
        require(review_id and review_id not in seen, f"pass {pass_id}: duplicate/blank anonymous ID")
        seen.add(review_id)
    require(set(registry) == seen, f"pass {pass_id}: raw-response registry IDs differ from completed scores")
    require(
        all(
            isinstance(value, dict)
            and value.get("model") == "gpt-5.6-luna"
            and value.get("observed_model") == "gpt-5.6-luna"
            for value in registry.values()
        ),
        f"pass {pass_id}: raw-response registry model identity changed",
    )
    require(
        [row["assignment"]["anonymous_review_id"] for row in public_rows]
        == [row["anonymous_review_id"] for row in rows],
        f"pass {pass_id}: public/completed anonymous order or inventory differs",
    )
    completed_by_id = {row["anonymous_review_id"]: row for row in rows}
    for public in public_rows:
        assignment = public["assignment"]
        completed = completed_by_id[assignment["anonymous_review_id"]]
        require(
            completed["assignment_sha256"] == assignment["assignment_sha256"]
            and completed["case_kind"] == assignment["case_kind"],
            f"pass {pass_id}: completed/public assignment binding differs",
        )
    return rows


def load_assignments(path: Path, expected_items: int = EXPECTED_ITEMS) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path, "public assignments")
    require(len(rows) == expected_items, f"public assignments must contain {expected_items} rows")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        review_id = str(row.get("anonymous_review_id", ""))
        require(review_id and review_id not in by_id, f"assignment row {index}: duplicate/blank ID")
        require(set(row) == set(transport.ASSIGNMENT_FIELDS), f"assignment row {index}: field schema changed")
        for field in ("assignment_sha256", *PUBLIC_CONTEXT_FIELDS):
            require(field in row, f"assignment row {index}: missing {field}")
        claimed = str(row["assignment_sha256"])
        payload = {key: value for key, value in row.items() if key != "assignment_sha256"}
        require(hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == claimed, f"assignment row {index}: assignment SHA mismatch")
        composite = Path(str(row["composite_path"]))
        require(
            not composite.is_absolute()
            and composite.parts == ("media", f"{review_id}.jpg"),
            f"assignment row {index}: composite path escaped/fails ID binding",
        )
        composite = (path.parent / composite).resolve()
        require(composite.parent == (path.parent / "media").resolve(), f"assignment row {index}: composite escaped media")
        regular_file(composite, f"assignment row {index} composite")
        require(sha256_file(composite) == row["composite_sha256"], f"assignment row {index}: composite SHA mismatch")
        row = dict(row)
        row["_resolved_composite_path"] = str(composite)
        by_id[review_id] = row
    return by_id


def load_key_commitments(path: Path) -> dict[str, Any]:
    value = load_json(path, "public key commitments")
    require(
        value.get("protocol") == REVIEW_PROTOCOL
        and value.get("schema_version") == 1
        and value.get("commitment_scheme") == "sha256(canonical-jsonl-bytes)",
        "public key commitment protocol changed",
    )
    code_ref = value.get("evaluation_code_registry")
    require(
        isinstance(code_ref, dict)
        and code_ref.get("path") == "evaluation_code_registry.json",
        "public evaluation-code registry reference changed",
    )
    code_path = path.parent / "evaluation_code_registry.json"
    regular_file(code_path, "evaluation code registry")
    require(
        code_ref.get("sha256") == sha256_file(code_path),
        "evaluation code registry file hash changed",
    )
    registry = load_json(code_path, "evaluation code registry")
    try:
        evaluation_code.validate_registry(
            registry, Path(__file__).resolve().parents[1]
        )
    except ValueError as exc:
        raise CanonicalizationError(str(exc)) from exc
    require(
        code_ref.get("registry_sha256") == registry["registry_sha256"],
        "evaluation code registry self-commitment changed",
    )
    for name, count in (
        ("tier_0_audit_strata", EXPECTED_ITEMS),
        ("tier_1_original_only", EXPECTED_ORIGINAL_KEY_ITEMS),
        ("tier_2_full", EXPECTED_ITEMS),
    ):
        entry = value.get(name)
        require(
            isinstance(entry, dict)
            and entry.get("row_count") == count
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64,
            f"public {name} commitment changed",
        )
    return value


def load_audit_strata_key(
    path: Path,
    commitments_path: Path,
    expected_items: int = EXPECTED_ITEMS,
) -> dict[str, dict[str, Any]]:
    commitments = load_key_commitments(commitments_path)
    require(
        commitments["tier_0_audit_strata"]["row_count"] == expected_items
        and commitments["tier_0_audit_strata"]["sha256"] == sha256_file(path),
        "opaque audit-strata key differs from its precommitted SHA/count",
    )
    rows = load_jsonl(path, "opaque audit-strata key")
    require(len(rows) == expected_items, f"opaque audit-strata key must contain {expected_items} rows")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(
            set(row) == {"anonymous_review_id", "case_kind", "mechanism", "stream_stratum"},
            f"opaque audit-strata row {index} schema changed",
        )
        review_id = str(row["anonymous_review_id"])
        require(review_id and review_id not in by_id, f"opaque audit-strata row {index} ID repeats")
        require(row["case_kind"] in FIELDS_BY_KIND, f"opaque audit-strata row {index} case kind changed")
        require(
            isinstance(row["stream_stratum"], str)
            and row["stream_stratum"].startswith("ss_")
            and len(row["stream_stratum"]) == 35,
            f"opaque audit-strata row {index} token changed",
        )
        by_id[review_id] = row
    require(len({row["stream_stratum"] for row in rows}) == 10, "opaque stream-stratum inventory must contain ten unlabeled streams")
    return by_id


def _rank(seed: int, purpose: str, atom: Mapping[str, Any]) -> str:
    value = f"{PROTOCOL}|{seed}|{purpose}|{atom['anonymous_review_id']}|{atom['field']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample_by_stratum(
    atoms: Sequence[dict[str, Any]],
    fraction: float,
    purpose: str,
    *,
    excluded: set[str] | None = None,
) -> set[str]:
    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        by_stratum[(atom["mechanism"], atom["stream_stratum"], atom["field"])].append(atom)
    selected: set[str] = set()
    excluded = excluded or set()
    for rows in by_stratum.values():
        target = math.ceil(len(rows) * fraction)
        eligible = [row for row in rows if row["audit_id"] not in excluded]
        ranked = sorted(eligible, key=lambda row: _rank(AUDIT_SEED, purpose, row))
        if len(ranked) < target:
            ranked.extend(
                sorted(
                    (row for row in rows if row["audit_id"] in excluded),
                    key=lambda row: _rank(AUDIT_SEED, purpose, row),
                )[: target - len(ranked)]
            )
        require(len(ranked) >= target, f"{purpose}: cannot select frozen 10% stratum sample")
        selected.update(row["audit_id"] for row in ranked[:target])
    return selected


def build_atoms(
    scores_a: Sequence[Mapping[str, Any]],
    scores_b: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    audit_strata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    a_by_id = {str(row["anonymous_review_id"]): row for row in scores_a}
    b_by_id = {str(row["anonymous_review_id"]): row for row in scores_b}
    require(set(a_by_id) == set(b_by_id) == set(assignments) == set(audit_strata), "A/B/assignment/audit-strata anonymous inventories differ")
    atoms: list[dict[str, Any]] = []
    for review_id in sorted(a_by_id):
        a, b, assignment, stratum = a_by_id[review_id], b_by_id[review_id], assignments[review_id], audit_strata[review_id]
        require(a["assignment_sha256"] == b["assignment_sha256"] == assignment["assignment_sha256"], f"{review_id}: assignment binding mismatch")
        require(a["case_kind"] == b["case_kind"] == assignment["case_kind"] == stratum["case_kind"], f"{review_id}: case_kind mismatch")
        require(assignment["mechanism_name"].casefold().replace(" ", "_") == stratum["mechanism"], f"{review_id}: public/opaque mechanism mismatch")
        fields = _score_fields(str(a["case_kind"]))
        item_unusable = bool(
            a["unusable_reason"]
            or b["unusable_reason"]
            or a["scores"]["video_quality"] == 0
            or b["scores"]["video_quality"] == 0
            or a["scores"]["receiver_preservation"] == 0
            or b["scores"]["receiver_preservation"] == 0
        )
        for field in fields:
            audit_id = hashlib.sha256(f"{review_id}\0{field}".encode("utf-8")).hexdigest()[:32]
            atoms.append(
                {
                    "audit_id": audit_id,
                    "anonymous_review_id": review_id,
                    "assignment_sha256": a["assignment_sha256"],
                    "case_kind": a["case_kind"],
                    "field": field,
                    "mechanism": stratum["mechanism"],
                    "stream_stratum": stratum["stream_stratum"],
                    "a_score": a["scores"][field],
                    "b_score": b["scores"][field],
                    "a_confidence": a["confidence"][field],
                    "b_confidence": b["confidence"][field],
                    "a_evidence_frames": a["evidence_frames"][field],
                    "b_evidence_frames": b["evidence_frames"][field],
                    "a_evidence_observations": a["evidence_observations"][field],
                    "b_evidence_observations": b["evidence_observations"][field],
                    "item_unusable": item_unusable,
                    "context": {field: assignment[field] for field in PUBLIC_CONTEXT_FIELDS},
                }
            )
    require(len(atoms) == expected_atom_count(audit_strata), "atomic score inventory is incomplete")
    require(len({atom["audit_id"] for atom in atoms}) == len(atoms), "audit ID collision")
    return atoms


def expected_atom_count(full_key: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(len(_score_fields(str(row["case_kind"]))) for row in full_key.values())


def select_audit_atoms(atoms: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    calibration = _sample_by_stratum(atoms, CALIBRATION_FRACTION, "calibration")
    high_conf_pool = [
        atom
        for atom in atoms
        if atom["a_score"] == atom["b_score"]
        and atom["a_score"] != 1
        and atom["a_confidence"] >= LOW_CONFIDENCE_THRESHOLD
        and atom["b_confidence"] >= LOW_CONFIDENCE_THRESHOLD
        and not atom["item_unusable"]
    ]
    high_conf_audit = _sample_by_stratum(
        high_conf_pool,
        HIGH_CONFIDENCE_FRACTION,
        "high_confidence_agreement",
        excluded=calibration,
    )
    selected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for atom in atoms:
        reasons: list[str] = []
        if atom["audit_id"] in calibration:
            reasons.append("calibration_10pct")
        if atom["a_score"] != atom["b_score"]:
            reasons.append("vlm_disagreement")
        if atom["a_score"] == 1 or atom["b_score"] == 1:
            reasons.append("partial")
        if atom["item_unusable"]:
            reasons.append("unusable_output")
        if atom["a_confidence"] < LOW_CONFIDENCE_THRESHOLD or atom["b_confidence"] < LOW_CONFIDENCE_THRESHOLD:
            reasons.append("low_confidence")
        if atom["audit_id"] in high_conf_audit:
            reasons.append("high_confidence_agreement_audit_10pct")
        if reasons:
            value = dict(atom)
            value["reason_codes"] = reasons
            selected.append(value)
            reason_counts.update(reasons)
    require(calibration <= {row["audit_id"] for row in selected}, "calibration sample disappeared")
    require(high_conf_audit <= {row["audit_id"] for row in selected}, "high-confidence audit disappeared")
    return selected, dict(sorted(reason_counts.items()))


def _human_row(atom: Mapping[str, Any], existing: Mapping[str, str] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "audit_id": atom["audit_id"],
        "anonymous_review_id": atom["anonymous_review_id"],
        "assignment_sha256": atom["assignment_sha256"],
        "field": atom["field"],
        **atom["context"],
        "reviewer_1_score": "",
        "reviewer_1_notes": "",
        "reviewer_2_score": "",
        "reviewer_2_notes": "",
        "adjudicator_score": "",
        "adjudicator_notes": "",
    }
    if existing:
        for field in HUMAN_QUEUE_FIELDS[-6:]:
            row[field] = existing[field]
    require(tuple(row) == HUMAN_QUEUE_FIELDS, "internal human queue schema changed")
    return row


def _private_atom(atom: Mapping[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in atom.items() if key != "context"}
    value["public_context_sha256"] = hashlib.sha256(
        canonical_json_bytes(atom["context"])
    ).hexdigest()
    return value


def _materialize_audit_media(
    public_root: Path,
    review_ids: Iterable[str],
    assignments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    media_root = public_root / "media"
    media_root.mkdir(mode=0o755)
    inventory: list[dict[str, Any]] = []
    for review_id in sorted(set(review_ids)):
        assignment = assignments[review_id]
        source = Path(str(assignment["_resolved_composite_path"]))
        regular_file(source, f"audit source composite {review_id}")
        target = media_root / f"{review_id}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        require(
            sha256_file(target) == assignment["composite_sha256"],
            f"audit composite copy changed: {review_id}",
        )
        inventory.append(
            {
                "anonymous_review_id": review_id,
                "path": f"media/{review_id}.jpg",
                "sha256": assignment["composite_sha256"],
                "size_bytes": target.stat().st_size,
            }
        )
    return inventory


def _fresh_root(output_root: Path) -> Path:
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    staging.chmod(0o700)
    return staging


def _expose(staging: Path, output_root: Path) -> None:
    require(not output_root.exists(), f"output appeared during build: {output_root}")
    os.replace(staging, output_root)


def build_audit_package(
    *,
    scores_a_path: Path,
    scores_b_path: Path,
    assignments_path: Path,
    audit_strata_key_path: Path,
    key_commitments_path: Path,
    output_root: Path,
    expected_items: int = EXPECTED_ITEMS,
) -> dict[str, Any]:
    scores_a = load_merged_scores(
        scores_a_path,
        "A",
        expected_items,
        expected_assignments_path=assignments_path,
    )
    scores_b = load_merged_scores(scores_b_path, "B", expected_items)
    require(
        [row["anonymous_review_id"] for row in scores_a]
        != [row["anonymous_review_id"] for row in scores_b],
        "A/B review orders are not independently randomized",
    )
    assignments = load_assignments(assignments_path, expected_items)
    audit_strata = load_audit_strata_key(
        audit_strata_key_path, key_commitments_path, expected_items
    )
    evaluation_code_registry_sha256 = load_key_commitments(
        key_commitments_path
    )["evaluation_code_registry"]["registry_sha256"]
    atoms = build_atoms(scores_a, scores_b, assignments, audit_strata)
    selected, reason_counts = select_audit_atoms(atoms)
    human_rows = [_human_row(atom) for atom in selected]
    staging = _fresh_root(output_root)
    try:
        public = staging / "public"
        private = staging / "private"
        public.mkdir(mode=0o755)
        private.mkdir(mode=0o700)
        media_inventory = _materialize_audit_media(
            public,
            (row["anonymous_review_id"] for row in selected),
            assignments,
        )
        media_manifest_path = public / "media_manifest.json"
        write_json(
            media_manifest_path,
            {"count": len(media_inventory), "items": media_inventory},
            mode=0o644,
        )
        queue_path = public / "human_audit_queue.csv"
        write_bytes(queue_path, csv_bytes(human_rows, HUMAN_QUEUE_FIELDS), mode=0o644)
        key_path = private / "audit_key.jsonl"
        all_atoms_path = private / "all_atoms.jsonl"
        write_jsonl(key_path, (_private_atom(atom) for atom in selected))
        write_jsonl(all_atoms_path, (_private_atom(atom) for atom in atoms))
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "initial_human_audit_frozen",
            "evaluation_code_registry_sha256": evaluation_code_registry_sha256,
            "audit_seed": AUDIT_SEED,
            "rules": {
                "calibration_fraction_per_mechanism_stream_field": CALIBRATION_FRACTION,
                "all_vlm_disagreements": True,
                "all_partial_atoms": True,
                "all_unusable_output_atoms": True,
                "confidence_below": LOW_CONFIDENCE_THRESHOLD,
                "separate_high_confidence_agreement_fraction": HIGH_CONFIDENCE_FRACTION,
                "expand_remaining_mechanism_field_if_error_rate_above": EXPANSION_ERROR_THRESHOLD,
                "two_independent_humans": True,
                "third_adjudicator_for_every_human_disagreement": True,
            },
            "counts": {
                "videos": expected_items,
                "atoms": len(atoms),
                "initial_audit_atoms": len(selected),
                "initial_audit_videos": len({row["anonymous_review_id"] for row in selected}),
                "reasons": reason_counts,
            },
            "inputs": {
                "scores_a": file_ref(scores_a_path),
                "scores_b": file_ref(scores_b_path),
                "assignments": file_ref(assignments_path),
                "audit_strata_key": file_ref(audit_strata_key_path),
                "key_commitments": file_ref(key_commitments_path),
            },
            "artifacts": {
                "human_queue": relative_ref(staging, queue_path),
                "media_manifest": relative_ref(staging, media_manifest_path),
                "audit_key": relative_ref(staging, key_path),
                "all_atoms": relative_ref(staging, all_atoms_path),
            },
            "human_delivery": "Provide independent copies to reviewer 1 and reviewer 2; neither may inspect the private directory or the other's labels.",
        }
        write_json(staging / "audit_manifest.json", manifest)
        _expose(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_audit_root(audit_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_json(audit_root / "audit_manifest.json", "audit manifest")
    require(manifest.get("protocol") == PROTOCOL, "audit manifest protocol mismatch")
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict), "audit manifest input bindings are missing")
    for name in ("scores_a", "scores_b", "assignments", "audit_strata_key", "key_commitments"):
        _resolve_ref(audit_root, inputs.get(name), f"audit input {name}")
    commitments_path = _resolve_ref(
        audit_root, inputs["key_commitments"], "audit input key_commitments"
    )
    commitments = load_key_commitments(commitments_path)
    require(
        manifest.get("evaluation_code_registry_sha256")
        == commitments["evaluation_code_registry"]["registry_sha256"],
        "audit manifest evaluation-code binding changed",
    )
    key_path = _resolve_ref(audit_root, manifest["artifacts"]["audit_key"], "audit key")
    all_atoms_path = _resolve_ref(audit_root, manifest["artifacts"]["all_atoms"], "all atoms")
    media_manifest_path = _resolve_ref(
        audit_root,
        manifest["artifacts"]["media_manifest"],
        "audit media manifest",
    )
    audit_rows = load_jsonl(key_path, "audit key")
    all_rows = load_jsonl(all_atoms_path, "all atoms")
    media_manifest = load_json(media_manifest_path, "audit media manifest")
    items = media_manifest.get("items")
    expected_ids = {row["anonymous_review_id"] for row in audit_rows}
    require(
        isinstance(items, list)
        and media_manifest.get("count") == len(items)
        and {item.get("anonymous_review_id") for item in items if isinstance(item, dict)}
        == expected_ids,
        "audit media manifest inventory differs from audit rows",
    )
    expected_media_paths: set[Path] = set()
    for item in items:
        media_path = audit_root / "public" / str(item["path"])
        regular_file(media_path, "audit composite")
        require(sha256_file(media_path) == item["sha256"], "audit composite differs from media manifest")
        expected_media_paths.add(media_path.resolve())
    media_root = audit_root / "public" / "media"
    require(
        media_root.is_dir()
        and not media_root.is_symlink()
        and {path.resolve() for path in media_root.iterdir()} == expected_media_paths
        and all(path.is_file() and not path.is_symlink() for path in media_root.iterdir()),
        "audit public media directory differs from the committed inventory",
    )
    return manifest, audit_rows, all_rows


def _human_score(text: str, label: str, *, allow_blank: bool = False) -> int | None:
    if text == "" and allow_blank:
        return None
    require(text in ("0", "1", "2"), f"{label} must be 0, 1, or 2")
    return int(text)


def validate_completed_human_rows(
    path: Path,
    audit_atoms: Mapping[str, Mapping[str, Any]],
    *,
    require_adjudication: bool,
) -> dict[str, dict[str, str]]:
    rows = load_csv(path, HUMAN_QUEUE_FIELDS, "completed human audit")
    by_id = {row["audit_id"]: row for row in rows}
    require(len(by_id) == len(rows) == len(audit_atoms) and set(by_id) == set(audit_atoms), "completed human audit inventory differs")
    for audit_id, row in by_id.items():
        atom = audit_atoms[audit_id]
        require(
            row["anonymous_review_id"] == atom["anonymous_review_id"]
            and row["assignment_sha256"] == atom["assignment_sha256"]
            and row["field"] == atom["field"],
            f"{audit_id}: completed human row binding changed",
        )
        observed_context = {field: row[field] for field in PUBLIC_CONTEXT_FIELDS}
        require(
            hashlib.sha256(canonical_json_bytes(observed_context)).hexdigest()
            == atom["public_context_sha256"],
            f"{audit_id}: completed human public context changed",
        )
        first = _human_score(row["reviewer_1_score"], f"{audit_id} reviewer 1")
        second = _human_score(row["reviewer_2_score"], f"{audit_id} reviewer 2")
        third = _human_score(row["adjudicator_score"], f"{audit_id} adjudicator", allow_blank=True)
        if first == second:
            require(third is None, f"{audit_id}: unnecessary adjudicator score")
        elif require_adjudication:
            require(third is not None, f"{audit_id}: human disagreement requires adjudication")
    return by_id


def resolved_human_score(row: Mapping[str, str]) -> int:
    values = [int(row["reviewer_1_score"]), int(row["reviewer_2_score"])]
    if values[0] == values[1]:
        return values[0]
    return int(row["adjudicator_score"])


def _is_high_conf_agreement(atom: Mapping[str, Any]) -> bool:
    return (
        atom["a_score"] == atom["b_score"]
        and atom["a_score"] != 1
        and atom["a_confidence"] >= LOW_CONFIDENCE_THRESHOLD
        and atom["b_confidence"] >= LOW_CONFIDENCE_THRESHOLD
        and not atom["item_unusable"]
    )


def expansion_requirements(
    audit_atoms: Mapping[str, Mapping[str, Any]],
    all_atoms: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, Mapping[str, str]],
) -> tuple[set[tuple[str, str]], set[str], list[dict[str, Any]]]:
    audited_by_stratum: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, str]]]] = defaultdict(list)
    for audit_id, atom in audit_atoms.items():
        if _is_high_conf_agreement(atom):
            audited_by_stratum[(atom["mechanism"], atom["field"])].append((atom, completed[audit_id]))
    expanded_strata: set[tuple[str, str]] = set()
    diagnostics: list[dict[str, Any]] = []
    for stratum, pairs in sorted(audited_by_stratum.items()):
        errors = sum(resolved_human_score(row) != atom["a_score"] for atom, row in pairs)
        error_rate = errors / len(pairs)
        expand = error_rate > EXPANSION_ERROR_THRESHOLD
        if expand:
            expanded_strata.add(stratum)
        diagnostics.append(
            {
                "mechanism": stratum[0],
                "field": stratum[1],
                "audited_high_confidence_atoms": len(pairs),
                "errors": errors,
                "error_rate": error_rate,
                "expansion_required": expand,
            }
        )
    # Once an initial audited sample crosses 5%, expansion is irreversible even
    # if the added labels subsequently dilute the observed aggregate error.
    persistent_strata = {
        (atom["mechanism"], atom["field"])
        for atom in audit_atoms.values()
        if "expanded_high_confidence_error_gt_5pct" in atom.get("reason_codes", [])
    }
    expanded_strata.update(persistent_strata)
    for diagnostic in diagnostics:
        if (diagnostic["mechanism"], diagnostic["field"]) in persistent_strata:
            diagnostic["expansion_required"] = True
            diagnostic["expansion_trigger_persisted"] = True
    required = {
        audit_id
        for audit_id, atom in all_atoms.items()
        if (atom["mechanism"], atom["field"]) in expanded_strata
    }
    return expanded_strata, required, diagnostics


def expand_audit_package(
    *,
    audit_root: Path,
    completed_human_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_manifest, audit_rows, all_rows = _load_audit_root(audit_root)
    audit_atoms = {row["audit_id"]: row for row in audit_rows}
    all_atoms = {row["audit_id"]: row for row in all_rows}
    completed = validate_completed_human_rows(completed_human_path, audit_atoms, require_adjudication=True)
    strata, required, diagnostics = expansion_requirements(audit_atoms, all_atoms, completed)
    added = sorted(required - set(audit_atoms))
    combined_atoms = [*audit_rows]
    for audit_id in added:
        atom = dict(all_atoms[audit_id])
        atom["reason_codes"] = ["expanded_high_confidence_error_gt_5pct"]
        combined_atoms.append(atom)
    assignment_context = load_assignments(Path(source_manifest["inputs"]["assignments"]["path"]), source_manifest["counts"]["videos"])
    queue_rows = [
        _human_row(
            {**atom, "context": {field: assignment_context[atom["anonymous_review_id"]][field] for field in PUBLIC_CONTEXT_FIELDS}},
            completed.get(atom["audit_id"]),
        )
        for atom in combined_atoms
    ]
    staging = _fresh_root(output_root)
    try:
        public, private = staging / "public", staging / "private"
        public.mkdir(mode=0o755)
        private.mkdir(mode=0o700)
        media_inventory = _materialize_audit_media(
            public,
            (row["anonymous_review_id"] for row in combined_atoms),
            assignment_context,
        )
        media_manifest_path = public / "media_manifest.json"
        write_json(
            media_manifest_path,
            {"count": len(media_inventory), "items": media_inventory},
            mode=0o644,
        )
        queue = public / "human_audit_queue.csv"
        key = private / "audit_key.jsonl"
        all_path = private / "all_atoms.jsonl"
        diagnostics_path = private / "expansion_diagnostics.json"
        write_bytes(queue, csv_bytes(queue_rows, HUMAN_QUEUE_FIELDS), mode=0o644)
        write_jsonl(key, combined_atoms)
        write_jsonl(all_path, all_rows)
        write_json(diagnostics_path, diagnostics)
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "expanded_human_audit_frozen" if added else "no_expansion_required_human_audit_frozen",
            "evaluation_code_registry_sha256": source_manifest[
                "evaluation_code_registry_sha256"
            ],
            "audit_seed": AUDIT_SEED,
            "source_audit_manifest": file_ref(audit_root / "audit_manifest.json"),
            "completed_source_human_audit": file_ref(completed_human_path),
            "counts": {
                "videos": source_manifest["counts"]["videos"],
                "atoms": len(all_rows),
                "initial_audit_atoms": len(audit_rows),
                "expanded_strata": len(strata),
                "added_audit_atoms": len(added),
                "total_audit_atoms": len(combined_atoms),
            },
            "inputs": source_manifest["inputs"],
            "artifacts": {
                "human_queue": relative_ref(staging, queue),
                "media_manifest": relative_ref(staging, media_manifest_path),
                "audit_key": relative_ref(staging, key),
                "all_atoms": relative_ref(staging, all_path),
                "expansion_diagnostics": relative_ref(staging, diagnostics_path),
            },
        }
        write_json(staging / "audit_manifest.json", manifest)
        _expose(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _macro_f1(truth: Sequence[int], predicted: Sequence[int]) -> float:
    values = []
    for label in (0, 1, 2):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(values) / 3


def _quadratic_weighted_kappa(truth: Sequence[int], predicted: Sequence[int]) -> float:
    n = len(truth)
    require(n > 0 and len(predicted) == n, "kappa inputs are empty/misaligned")
    observed = [[0] * 3 for _ in range(3)]
    for left, right in zip(truth, predicted):
        observed[left][right] += 1
    truth_counts = Counter(truth)
    pred_counts = Counter(predicted)
    observed_weighted = sum(((i - j) ** 2 / 4) * observed[i][j] for i in range(3) for j in range(3)) / n
    expected_weighted = sum(((i - j) ** 2 / 4) * truth_counts[i] * pred_counts[j] for i in range(3) for j in range(3)) / (n * n)
    return 1.0 if expected_weighted == 0 and observed_weighted == 0 else 0.0 if expected_weighted == 0 else 1 - observed_weighted / expected_weighted


def freeze_canonical_scores(
    *,
    audit_root: Path,
    completed_human_path: Path,
    scores_a_path: Path,
    scores_b_path: Path,
    output_root: Path,
    expected_items: int = EXPECTED_ITEMS,
) -> dict[str, Any]:
    audit_manifest, audit_rows, all_rows = _load_audit_root(audit_root)
    _resolve_ref(audit_root, audit_manifest["inputs"]["scores_a"], "frozen pass A scores", scores_a_path)
    _resolve_ref(audit_root, audit_manifest["inputs"]["scores_b"], "frozen pass B scores", scores_b_path)
    audit_atoms = {row["audit_id"]: row for row in audit_rows}
    all_atoms = {row["audit_id"]: row for row in all_rows}
    completed = validate_completed_human_rows(completed_human_path, audit_atoms, require_adjudication=True)
    strata, required, diagnostics = expansion_requirements(audit_atoms, all_atoms, completed)
    require(required <= set(audit_atoms), f"audit expansion is incomplete: {len(required - set(audit_atoms))} atoms remain")
    scores_a = load_merged_scores(scores_a_path, "A", expected_items)
    scores_b = load_merged_scores(scores_b_path, "B", expected_items)
    a_by_id = {row["anonymous_review_id"]: row for row in scores_a}
    b_by_id = {row["anonymous_review_id"]: row for row in scores_b}
    require(set(a_by_id) == set(b_by_id), "A/B score inventories differ at freeze")
    human_by_atom = {audit_id: resolved_human_score(row) for audit_id, row in completed.items()}
    canonical: list[dict[str, Any]] = []
    for review_id in sorted(a_by_id):
        a, b = a_by_id[review_id], b_by_id[review_id]
        require(
            a["assignment_sha256"] == b["assignment_sha256"]
            and a["case_kind"] == b["case_kind"],
            f"{review_id}: A/B binding changed at canonical freeze",
        )
        fields = _score_fields(str(a["case_kind"]))
        values: dict[str, int] = {}
        sources: dict[str, str] = {}
        for field in fields:
            audit_id = hashlib.sha256(f"{review_id}\0{field}".encode("utf-8")).hexdigest()[:32]
            if audit_id in human_by_atom:
                values[field] = human_by_atom[audit_id]
                sources[field] = "human_consensus_or_adjudication"
            else:
                require(a["scores"][field] == b["scores"][field], f"{review_id}/{field}: unaudited VLM disagreement")
                require(a["scores"][field] != 1 and a["confidence"][field] >= LOW_CONFIDENCE_THRESHOLD and b["confidence"][field] >= LOW_CONFIDENCE_THRESHOLD, f"{review_id}/{field}: required human audit missing")
                values[field] = a["scores"][field]
                sources[field] = "high_confidence_vlm_agreement_not_selected_for_audit"
        canonical.append(
            {
                "anonymous_review_id": review_id,
                "assignment_sha256": a["assignment_sha256"],
                "case_kind": a["case_kind"],
                "scores": values,
                "canonical_sources": sources,
                "status": "canonical_frozen",
            }
        )

    truth = [human_by_atom[audit_id] for audit_id in sorted(human_by_atom)]
    pred_a = [all_atoms[audit_id]["a_score"] for audit_id in sorted(human_by_atom)]
    pred_b = [all_atoms[audit_id]["b_score"] for audit_id in sorted(human_by_atom)]
    calibration_ids = sorted(
        audit_id
        for audit_id, atom in audit_atoms.items()
        if "calibration_10pct" in atom.get("reason_codes", [])
    )
    require(calibration_ids, "frozen calibration sample is empty")
    calibration_truth = [human_by_atom[audit_id] for audit_id in calibration_ids]
    calibration_a = [all_atoms[audit_id]["a_score"] for audit_id in calibration_ids]
    calibration_b = [all_atoms[audit_id]["b_score"] for audit_id in calibration_ids]
    diagnostics_by_stratum = diagnostics
    quality = {
        "vlm_vlm_atomic_disagreement_rate": sum(row["a_score"] != row["b_score"] for row in all_rows) / len(all_rows),
        "calibration_vlm_a_human_macro_f1": _macro_f1(calibration_truth, calibration_a),
        "calibration_vlm_b_human_macro_f1": _macro_f1(calibration_truth, calibration_b),
        "calibration_vlm_a_human_quadratic_weighted_kappa": _quadratic_weighted_kappa(calibration_truth, calibration_a),
        "calibration_vlm_b_human_quadratic_weighted_kappa": _quadratic_weighted_kappa(calibration_truth, calibration_b),
        "all_targeted_audits_vlm_a_human_macro_f1": _macro_f1(truth, pred_a),
        "all_targeted_audits_vlm_b_human_macro_f1": _macro_f1(truth, pred_b),
        "all_targeted_audits_vlm_a_human_quadratic_weighted_kappa": _quadratic_weighted_kappa(truth, pred_a),
        "all_targeted_audits_vlm_b_human_quadratic_weighted_kappa": _quadratic_weighted_kappa(truth, pred_b),
        "human_review_atom_coverage": len(human_by_atom) / len(all_rows),
        "human_review_video_coverage": len({audit_atoms[audit_id]["anonymous_review_id"] for audit_id in human_by_atom}) / expected_items,
        "human_disagreement_atoms": sum(row["reviewer_1_score"] != row["reviewer_2_score"] for row in completed.values()),
        "expanded_strata": len(strata),
        "expanded_atoms": sum("expanded_high_confidence_error_gt_5pct" in atom.get("reason_codes", []) for atom in audit_rows),
    }
    staging = _fresh_root(output_root)
    try:
        canonical_path = staging / "canonical_scores.jsonl"
        diagnostics_path = staging / "audit_diagnostics.json"
        write_jsonl(canonical_path, canonical)
        write_json(diagnostics_path, {"quality": quality, "expansion_strata": diagnostics_by_stratum})
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "canonical_anonymous_scores_frozen_before_answer_key_opening",
            "evaluation_code_registry_sha256": audit_manifest[
                "evaluation_code_registry_sha256"
            ],
            "row_count": expected_items,
            "atomic_count": len(all_rows),
            "inputs": {
                "audit_manifest": file_ref(audit_root / "audit_manifest.json"),
                "completed_human_audit": file_ref(completed_human_path),
                "scores_a": file_ref(scores_a_path),
                "scores_b": file_ref(scores_b_path),
                "audit_strata_key": audit_manifest["inputs"]["audit_strata_key"],
                "key_commitments": audit_manifest["inputs"]["key_commitments"],
            },
            "artifacts": {
                "canonical_scores": relative_ref(staging, canonical_path),
                "audit_diagnostics": relative_ref(staging, diagnostics_path),
            },
            "quality": quality,
            "full_method_key_opened_for_scoring": False,
        }
        write_json(staging / "canonical_manifest.json", manifest)
        _expose(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def freeze_original_eligibility(
    *,
    canonical_root: Path,
    original_key_path: Path,
    key_commitments_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    canonical_manifest = load_json(canonical_root / "canonical_manifest.json", "canonical manifest")
    require(
        canonical_manifest.get("protocol") == PROTOCOL
        and canonical_manifest.get("status")
        == "canonical_anonymous_scores_frozen_before_answer_key_opening"
        and canonical_manifest.get("row_count") == EXPECTED_ITEMS,
        "canonical score freeze is not complete",
    )
    canonical_path = _resolve_ref(
        canonical_root,
        canonical_manifest["artifacts"]["canonical_scores"],
        "canonical scores",
    )
    canonical_rows = load_jsonl(canonical_path, "canonical scores")
    canonical_by_id = {row["anonymous_review_id"]: row for row in canonical_rows}
    require(len(canonical_rows) == len(canonical_by_id) == EXPECTED_ITEMS, "canonical score IDs changed")
    _resolve_ref(
        canonical_root,
        canonical_manifest["inputs"]["key_commitments"],
        "pre-canonical key commitments",
        key_commitments_path,
    )
    commitments = load_key_commitments(key_commitments_path)
    require(
        canonical_manifest.get("evaluation_code_registry_sha256")
        == commitments["evaluation_code_registry"]["registry_sha256"],
        "canonical evaluation-code binding changed before eligibility freeze",
    )
    require(
        commitments["tier_1_original_only"]["sha256"] == sha256_file(original_key_path),
        "Original-only key differs from its precommitted SHA",
    )
    original_rows = load_jsonl(original_key_path, "Original-only key")
    require(len(original_rows) == EXPECTED_ORIGINAL_KEY_ITEMS, "Original-only key count changed")
    original_by_id = {row["anonymous_review_id"]: row for row in original_rows}
    require(len(original_by_id) == EXPECTED_ORIGINAL_KEY_ITEMS and set(original_by_id) <= set(canonical_by_id), "Original-only key IDs changed")
    require(
        Counter(row.get("stream") for row in original_rows)
        == {"wan_original": 294, "cogvideox_original": 294},
        "Original-only key stream inventory changed",
    )
    eligibility: list[dict[str, Any]] = []
    by_case_backbone: dict[tuple[str, str], bool] = {}
    for review_id, key in sorted(original_by_id.items()):
        expected_backbone = "wan" if key["stream"] == "wan_original" else "cogvideox"
        require(key.get("backbone_family") == expected_backbone, f"Original key backbone mismatch: {review_id}")
        canonical = canonical_by_id[review_id]
        require(canonical["case_kind"] == key["case_kind"], f"Original key kind mismatch: {review_id}")
        if key["case_kind"] != "causal":
            continue
        scores = canonical["scores"]
        eligible = (
            scores["source_visibility"] == 2
            and scores["footprint_visibility"] >= 1
            and scores["receiver_preservation"] >= 1
            and scores["video_quality"] >= 1
        )
        backbone = str(key["backbone_family"])
        pair = (str(key["case_id"]), backbone)
        require(pair not in by_case_backbone, f"duplicate Original eligibility row: {pair}")
        by_case_backbone[pair] = eligible
        eligibility.append(
            {
                "anonymous_review_id": review_id,
                "backbone_family": backbone,
                "case_id": key["case_id"],
                "mechanism": key["mechanism"],
                "eligible": int(eligible),
                **{field: scores[field] for field in FIELDS_BY_KIND["causal"]},
            }
        )
    require(len(eligibility) == 336, "Original eligibility must contain 2x168 causal rows")
    case_meta = {(row["case_id"], row["mechanism"]) for row in eligibility}
    require(len(case_meta) == 168, "Original causal case inventory changed")
    shared = []
    for case_id, mechanism in sorted(case_meta):
        wan = by_case_backbone[(case_id, "wan")]
        cog = by_case_backbone[(case_id, "cogvideox")]
        shared.append(
            {
                "case_id": case_id,
                "mechanism": mechanism,
                "wan_original_eligible": int(wan),
                "cogvideox_original_eligible": int(cog),
                "shared_eligible": int(wan and cog),
            }
        )
    staging = _fresh_root(output_root)
    try:
        eligibility_path = staging / "original_eligibility.csv"
        shared_path = staging / "shared_capability_subsets.csv"
        write_bytes(eligibility_path, csv_bytes(eligibility, tuple(eligibility[0])))
        write_bytes(shared_path, csv_bytes(shared, tuple(shared[0])))
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "original_eligibility_and_shared_subsets_frozen_before_full_key_opening",
            "evaluation_code_registry_sha256": canonical_manifest[
                "evaluation_code_registry_sha256"
            ],
            "inputs": {
                "canonical_manifest": file_ref(canonical_root / "canonical_manifest.json"),
                "canonical_scores": file_ref(canonical_path),
                "original_only_key": file_ref(original_key_path),
                "key_commitments": file_ref(key_commitments_path),
            },
            "artifacts": {
                "original_eligibility": relative_ref(staging, eligibility_path),
                "shared_capability_subsets": relative_ref(staging, shared_path),
            },
            "counts": {
                "original_causal_rows": 336,
                "semantic_causal_cases": 168,
                "shared_eligible_cases": sum(row["shared_eligible"] for row in shared),
            },
            "full_method_key_opened": False,
        }
        write_json(staging / "eligibility_manifest.json", manifest)
        _expose(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-audit")
    build.add_argument("--scores-a", type=Path, required=True)
    build.add_argument("--scores-b", type=Path, required=True)
    build.add_argument("--assignments", type=Path, required=True)
    build.add_argument("--audit-strata-key", type=Path, required=True)
    build.add_argument("--key-commitments", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    expand = commands.add_parser("expand-audit")
    expand.add_argument("--audit-root", type=Path, required=True)
    expand.add_argument("--completed-human", type=Path, required=True)
    expand.add_argument("--output-root", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--audit-root", type=Path, required=True)
    freeze.add_argument("--completed-human", type=Path, required=True)
    freeze.add_argument("--scores-a", type=Path, required=True)
    freeze.add_argument("--scores-b", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    eligibility = commands.add_parser("freeze-eligibility")
    eligibility.add_argument("--canonical-root", type=Path, required=True)
    eligibility.add_argument("--original-key", type=Path, required=True)
    eligibility.add_argument("--key-commitments", type=Path, required=True)
    eligibility.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.command == "build-audit":
            result = build_audit_package(
                scores_a_path=_resolve(root, args.scores_a),
                scores_b_path=_resolve(root, args.scores_b),
                assignments_path=_resolve(root, args.assignments),
                audit_strata_key_path=_resolve(root, args.audit_strata_key),
                key_commitments_path=_resolve(root, args.key_commitments),
                output_root=_resolve(root, args.output_root),
            )
        elif args.command == "expand-audit":
            result = expand_audit_package(
                audit_root=_resolve(root, args.audit_root),
                completed_human_path=_resolve(root, args.completed_human),
                output_root=_resolve(root, args.output_root),
            )
        elif args.command == "freeze":
            result = freeze_canonical_scores(
                audit_root=_resolve(root, args.audit_root),
                completed_human_path=_resolve(root, args.completed_human),
                scores_a_path=_resolve(root, args.scores_a),
                scores_b_path=_resolve(root, args.scores_b),
                output_root=_resolve(root, args.output_root),
            )
        else:
            result = freeze_original_eligibility(
                canonical_root=_resolve(root, args.canonical_root),
                original_key_path=_resolve(root, args.original_key),
                key_commitments_path=_resolve(root, args.key_commitments),
                output_root=_resolve(root, args.output_root),
            )
    except CanonicalizationError as exc:
        parser.exit(2, f"canonicalization refused: {exc}\n")
    print(json.dumps({"status": result["status"], "counts": result.get("counts", {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
