#!/usr/bin/env python3
"""Finalize the one-pass-plus-targeted-audit 7-mechanism target screen.

This is a deliberately narrow replacement for the two-full-review selection
stage.  Reviewer A remains canonical on fields that were not targeted for an
independent audit.  A targeted audit agreement is canonical directly.  An
audited disagreement is never silently resolved: a blind field-level
adjudication may resolve it, otherwise it remains explicitly unresolved.

``plan-adjudication`` publishes only unresolved disagreements that can change
the selected-178 membership under some completion of the other unresolved
fields.  The public queue contains neither Reviewer-A nor audit scores.

``finalize`` fails if a selection-affecting disagreement is unresolved.  A
selection-inert disagreement may remain unresolved (and is emitted with a
blank canonical score plus its two-value domain); it is never relabelled as a
Reviewer-A score.  The selected membership is then invariant to either value.
The frozen training row order remains selection-hash order for compatibility,
while membership is determined by the score ranking specified below.

The tool performs no model call, generation, cache preparation, or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

try:  # Package import in tests and ``python -m``.
    from scripts import screen_causal_role_erasure_7mechanism_targets_v2 as screening
except ModuleNotFoundError:  # Direct ``python scripts/<name>.py`` execution.
    import screen_causal_role_erasure_7mechanism_targets_v2 as screening


WORKFLOW_VERSION = "causal_role_erasure_7mechanism_targeted_audit_selection_v1"
SCHEMA_VERSION = 1
SELECTED_PER_MECHANISM = screening.SELECTED_PER_MECHANISM
MECHANISM_ORDER = screening.MECHANISM_ORDER
NEW_MECHANISMS = screening.NEW_MECHANISMS
WATER_MECHANISM = screening.WATER_MECHANISM
SCORE_FIELDS = screening.SCORE_FIELDS
CORE_FIELDS = ("source_absent", "trigger_absent", "footprint_absent")
HARD_REJECT_FIELDS = (
    "receiver_recognizable",
    "source_absent",
    "trigger_absent",
    "footprint_absent",
    "quality",
)
SELECTION_HASH_CONTEXT = screening.SELECTION_HASH_CONTEXT
MEDIA_CONTRACT = {
    "decoded_frames": screening.FRAME_COUNT,
    "fps": f"{screening.FPS.numerator}/{screening.FPS.denominator}",
    "height": screening.HEIGHT,
    "width": screening.WIDTH,
}

AUDIT_TOP_FIELDS = ("reviewer", "blind_to_a", "items")
AUDIT_ITEM_FIELDS = ("anonymous_review_id", "scores", "evidence")
EVIDENCE_FIELDS = ("frame_index", "observation")
ADJUDICATION_TOP_FIELDS = (
    "reviewer",
    "blind_to_reviewer_a_and_agent_audit_scores",
    "items",
)
ADJUDICATION_ITEM_FIELDS = ("anonymous_review_id", "scores", "evidence")

CANONICAL_FIELDS = (
    "anonymous_review_id",
    "candidate_id",
    "mechanism",
    *SCORE_FIELDS,
    "canonical_status",
    "unresolved_fields",
    "score_domains_json",
    "hard_reject",
    "selected",
    "screening_rank",
)
PROVENANCE_FIELDS = (
    "anonymous_review_id",
    "candidate_id",
    "mechanism",
    "field",
    "reviewer_a_score",
    "audit_status",
    "audit_reviewer",
    "audit_score",
    "audit_source_sha256",
    "adjudication_status",
    "adjudicator",
    "adjudication_score",
    "adjudication_source_sha256",
    "canonical_score",
    "resolution",
)
SELECTED_APPEND_FIELDS = (
    *SCORE_FIELDS,
    "canonical_status",
    "unresolved_fields",
    "score_domains_json",
    "eligible",
    "membership_rank",
    "selection_hash",
    "selection_rank",
    "selected_index",
    "selected_video_path",
    "selected_video_sha256",
    "selected_size_bytes",
    "selected_media",
)


class TargetedAuditError(ValueError):
    """A fail-closed targeted-audit protocol error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetedAuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def require_hex64(value: str, label: str) -> str:
    require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256",
    )
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    regular_file(path, "CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        require(fields, f"CSV has no header: {path}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"CSV contains overflow columns: {path}")
    return fields, [{field: str(row[field]) for field in fields} for row in rows]


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: str(row.get(field, "")) for field in fields})
    return stream.getvalue().encode("utf-8")


def write_bytes_exclusive(path: Path, raw: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced input")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_file_ref(root: Path, path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    regular_file(path, "output artifact")
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if row_count is not None:
        result["row_count"] = row_count
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetedAuditError(f"cannot parse {label}: {path}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _score(value: Any, label: str) -> int:
    require(type(value) in (int, str), f"{label} is not an integer score")
    try:
        result = int(value)
    except ValueError as exc:
        raise TargetedAuditError(f"{label} is not an integer score") from exc
    require(str(result) == str(value), f"{label} is not a canonical integer score")
    require(result in {0, 1, 2}, f"{label} is outside 0,1,2")
    return result


def selection_hash(subject_id: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_HASH_CONTEXT}|{subject_id}".encode()
    ).hexdigest()


def hard_reject(scores: Mapping[str, int]) -> bool:
    """Natural motion is intentionally absent: it is a soft rank signal."""

    return any(int(scores[field]) == 0 for field in HARD_REJECT_FIELDS)


def score_rank_tuple(scores: Mapping[str, int]) -> tuple[int, ...]:
    """Return the frozen descending membership-ranking tuple."""

    return (
        min(int(scores[field]) for field in CORE_FIELDS),
        sum(int(scores[field]) for field in CORE_FIELDS),
        int(scores["receiver_recognizable"]),
        int(scores["quality"]),
        int(scores["natural_motion"]),
        sum(int(scores[field]) for field in SCORE_FIELDS),
    )


def priority(scores: Mapping[str, int], subject_id: str) -> tuple[int, ...] | None:
    """Larger tuples are better; the negated hash implements ascending tie-break."""

    if hard_reject(scores):
        return None
    return (*score_rank_tuple(scores), -int(selection_hash(subject_id), 16))


def ranking_sort_key(scores: Mapping[str, int], subject_id: str) -> tuple[Any, ...]:
    values = score_rank_tuple(scores)
    return (*(-value for value in values), selection_hash(subject_id), subject_id)


def _validate_evidence(value: Any, label: str) -> None:
    require(isinstance(value, list) and value, f"{label} evidence must be non-empty")
    for index, evidence in enumerate(value):
        require(isinstance(evidence, dict), f"{label} evidence {index} is not an object")
        require(set(evidence) == set(EVIDENCE_FIELDS), f"{label} evidence {index} fields differ")
        require(
            type(evidence["frame_index"]) is int
            and 0 <= evidence["frame_index"] < screening.FRAME_COUNT,
            f"{label} evidence {index} frame is invalid",
        )
        require(
            isinstance(evidence["observation"], str)
            and evidence["observation"].strip(),
            f"{label} evidence {index} observation is blank",
        )


def load_assignment(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    fields, rows = read_csv(path)
    require(tuple(fields) == screening.ASSIGNMENT_FIELDS, "Reviewer-A assignment columns differ")
    require(len(rows) == screening.EXPECTED_NEW_VIDEOS, "Reviewer-A assignment must contain 1152 rows")
    require(
        [int(row["assignment_position"]) for row in rows] == list(range(len(rows))),
        "Reviewer-A assignment positions are not canonical",
    )
    identifiers = [row["anonymous_review_id"] for row in rows]
    require(len(set(identifiers)) == len(rows), "Reviewer-A assignment IDs repeat")
    require(
        Counter(row["mechanism"] for row in rows)
        == Counter({mechanism: screening.ROWS_PER_MECHANISM for mechanism in NEW_MECHANISMS}),
        "Reviewer-A assignment mechanism inventory differs",
    )
    return rows, {row["anonymous_review_id"]: row for row in rows}


def load_reviewer_a(
    path: Path,
    assignment_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    fields, rows = read_csv(path)
    require(tuple(fields) == screening.SCORE_TEMPLATE_FIELDS, "Reviewer-A completed columns differ")
    require(len(rows) == len(assignment_rows), "Reviewer-A completed row count differs")
    output: dict[str, dict[str, int]] = {}
    for index, (score_row, assignment) in enumerate(zip(rows, assignment_rows)):
        anonymous_id = assignment["anonymous_review_id"]
        require(
            score_row["anonymous_review_id"] == anonymous_id,
            f"Reviewer-A row {index} ID/order changed",
        )
        require(anonymous_id not in output, "Reviewer-A completed IDs repeat")
        output[anonymous_id] = {
            field: _score(score_row[field], f"Reviewer-A {anonymous_id}/{field}")
            for field in SCORE_FIELDS
        }
    return output


def load_agent_audits(
    paths: Sequence[Path],
    assignment_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    require(paths, "at least one targeted agent-audit JSON is required")
    scores: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, str]] = {}
    reviewers: set[str] = set()
    for path in paths:
        payload = _load_json_object(path, "targeted agent audit")
        require(set(payload) == set(AUDIT_TOP_FIELDS), f"audit top-level fields differ: {path}")
        reviewer = payload["reviewer"]
        require(isinstance(reviewer, str) and reviewer.strip(), f"audit reviewer is blank: {path}")
        require(reviewer not in reviewers, f"audit reviewer name repeats: {reviewer}")
        reviewers.add(reviewer)
        require(payload["blind_to_a"] is True, f"audit was not declared blind to Reviewer A: {path}")
        require(isinstance(payload["items"], list) and payload["items"], f"audit items are empty: {path}")
        source_sha256 = sha256_file(path)
        for item_index, item in enumerate(payload["items"]):
            label = f"audit {path.name} item {item_index}"
            require(isinstance(item, dict) and set(item) == set(AUDIT_ITEM_FIELDS), f"{label} fields differ")
            anonymous_id = item["anonymous_review_id"]
            require(
                isinstance(anonymous_id, str) and anonymous_id in assignment_by_id,
                f"{label} references an unknown assignment ID",
            )
            require(anonymous_id not in scores, f"targeted audit ID repeats: {anonymous_id}")
            require(isinstance(item["scores"], dict) and item["scores"], f"{label} scores are empty")
            require(set(item["scores"]) <= set(SCORE_FIELDS), f"{label} contains an unknown score field")
            scores[anonymous_id] = {
                field: _score(raw_score, f"{label}/{field}")
                for field, raw_score in item["scores"].items()
            }
            _validate_evidence(item["evidence"], label)
            metadata[anonymous_id] = {
                "reviewer": reviewer,
                "source_path": str(path),
                "source_sha256": source_sha256,
            }
    return scores, metadata


def load_adjudication(
    path: Path | None,
    *,
    reviewer_a: Mapping[str, Mapping[str, int]],
    audit_scores: Mapping[str, Mapping[str, int]],
) -> tuple[dict[tuple[str, str], int], dict[str, str]]:
    if path is None:
        return {}, {}
    payload = _load_json_object(path, "field-level adjudication")
    require(set(payload) == set(ADJUDICATION_TOP_FIELDS), "adjudication top-level fields differ")
    reviewer = payload["reviewer"]
    require(isinstance(reviewer, str) and reviewer.strip(), "adjudication reviewer is blank")
    require(
        payload["blind_to_reviewer_a_and_agent_audit_scores"] is True,
        "adjudication was not declared blind to both parent scores",
    )
    require(isinstance(payload["items"], list), "adjudication items must be a list")
    output: dict[tuple[str, str], int] = {}
    for item_index, item in enumerate(payload["items"]):
        label = f"adjudication item {item_index}"
        require(isinstance(item, dict) and set(item) == set(ADJUDICATION_ITEM_FIELDS), f"{label} fields differ")
        anonymous_id = item["anonymous_review_id"]
        require(anonymous_id in audit_scores, f"{label} references a non-audited ID")
        require(isinstance(item["scores"], dict) and item["scores"], f"{label} scores are empty")
        require(set(item["scores"]) <= set(SCORE_FIELDS), f"{label} contains an unknown field")
        for field, raw_score in item["scores"].items():
            require(field in audit_scores[anonymous_id], f"{label} field was not independently audited")
            require(
                reviewer_a[anonymous_id][field] != audit_scores[anonymous_id][field],
                f"{label} attempts to adjudicate a field without A/audit disagreement",
            )
            key = (anonymous_id, field)
            require(key not in output, f"adjudication field repeats: {anonymous_id}/{field}")
            output[key] = _score(raw_score, f"{label}/{field}")
        _validate_evidence(item["evidence"], label)
    return output, {
        "reviewer": reviewer,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
    }


def load_binding(
    path: Path,
    *,
    assignment_rows: Sequence[Mapping[str, str]],
    candidate_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    fields, rows = read_csv(path)
    require(tuple(fields) == screening.BINDING_FIELDS, "Reviewer-A binding columns differ")
    require(len(rows) == len(assignment_rows), "Reviewer-A binding row count differs")
    output: dict[str, dict[str, str]] = {}
    candidate_ids: set[str] = set()
    for index, (binding, assignment) in enumerate(zip(rows, assignment_rows)):
        anonymous_id = assignment["anonymous_review_id"]
        require(binding["anonymous_review_id"] == anonymous_id, f"binding row {index} ID/order changed")
        require(binding["pass_name"] == "reviewer_a", f"binding row {index} pass differs")
        candidate_id = binding["candidate_id"]
        require(candidate_id in candidate_by_id, f"binding row {index} candidate is unknown")
        require(candidate_id not in candidate_ids, f"binding candidate repeats: {candidate_id}")
        candidate_ids.add(candidate_id)
        candidate = candidate_by_id[candidate_id]
        require(candidate["mechanism"] == assignment["mechanism"] == binding["mechanism"], f"binding row {index} mechanism differs")
        require(binding["composite_path"] == assignment["composite_path"], f"binding row {index} composite path differs")
        require(
            binding["candidate_row_sha256"] == screening.canonical_row_sha256(candidate),
            f"binding row {index} candidate-row hash differs",
        )
        require_hex64(binding["video_sha256"], f"binding row {index} video hash")
        require_hex64(binding["composite_sha256"], f"binding row {index} composite hash")
        require(binding["video_path"].strip(), f"binding row {index} video path is blank")
        output[anonymous_id] = binding
    expected = {
        candidate_id
        for candidate_id, row in candidate_by_id.items()
        if row["mechanism"] in NEW_MECHANISMS
    }
    require(candidate_ids == expected, "Reviewer-A binding/candidate inventories differ")
    return output, {anonymous_id: row["candidate_id"] for anonymous_id, row in output.items()}


def derive_candidate_mapping_from_public_assignment(
    *,
    assignment_rows: Sequence[Mapping[str, str]],
    candidate_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Recover candidate IDs by an exact, unique public-field join.

    This permits score-blind adjudication planning before the private package
    is locally available without ever changing the candidate-ID hash tie-break.
    Finalization still requires the private binding for video paths and hashes.
    """

    join_fields = (
        "mechanism",
        "mechanism_name",
        "target_prompt",
        "receiver",
        "source_object",
        "expected_trigger",
        "expected_footprint",
        "expected_counterfactual_state",
        "excluded_content_phrase",
    )
    candidates = [
        row for row in candidate_by_id.values() if row["mechanism"] in NEW_MECHANISMS
    ]
    by_public_key: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for candidate in candidates:
        by_public_key[tuple(candidate.get(field, "") for field in join_fields)].append(
            candidate["candidate_id"]
        )
    require(
        all(len(candidate_ids) == 1 for candidate_ids in by_public_key.values()),
        "candidate manifest is not unique under the public assignment join",
    )
    output: dict[str, str] = {}
    for index, assignment in enumerate(assignment_rows):
        key = tuple(assignment[field] for field in join_fields)
        matches = by_public_key.get(key, [])
        require(len(matches) == 1, f"assignment row {index} has no unique candidate join")
        output[assignment["anonymous_review_id"]] = matches[0]
    require(
        len(set(output.values())) == len(candidates) == len(output),
        "public assignment/candidate join is not one-to-one and complete",
    )
    return output


def _field_disagreements(
    reviewer_a: Mapping[str, Mapping[str, int]],
    audit_scores: Mapping[str, Mapping[str, int]],
) -> set[tuple[str, str]]:
    return {
        (anonymous_id, field)
        for anonymous_id, scores in audit_scores.items()
        for field in scores
        if reviewer_a[anonymous_id][field] != scores[field]
    }


def _score_domain(
    anonymous_id: str,
    *,
    reviewer_a: Mapping[str, Mapping[str, int]],
    audit_scores: Mapping[str, Mapping[str, int]],
    adjudicated: Mapping[tuple[str, str], int],
    forced: Mapping[str, int] | None = None,
) -> list[dict[str, int]]:
    base = dict(reviewer_a[anonymous_id])
    choices: list[tuple[str, tuple[int, ...]]] = []
    for field in SCORE_FIELDS:
        key = (anonymous_id, field)
        if forced is not None and field in forced:
            choices.append((field, (forced[field],)))
        elif key in adjudicated:
            choices.append((field, (adjudicated[key],)))
        elif (
            field in audit_scores.get(anonymous_id, {})
            and audit_scores[anonymous_id][field] != base[field]
        ):
            choices.append((field, (base[field], audit_scores[anonymous_id][field])))
        else:
            choices.append((field, (base[field],)))
    output: dict[tuple[int, ...], dict[str, int]] = {}
    for values in itertools.product(*(value for _, value in choices)):
        scores = {field: score for (field, _), score in zip(choices, values)}
        output[tuple(scores[field] for field in SCORE_FIELDS)] = scores
    return list(output.values())


def _priority_domain(
    anonymous_id: str,
    *,
    subject_id: str,
    reviewer_a: Mapping[str, Mapping[str, int]],
    audit_scores: Mapping[str, Mapping[str, int]],
    adjudicated: Mapping[tuple[str, str], int],
) -> set[tuple[int, ...] | None]:
    return {
        priority(scores, subject_id)
        for scores in _score_domain(
            anonymous_id,
            reviewer_a=reviewer_a,
            audit_scores=audit_scores,
            adjudicated=adjudicated,
        )
    }


def _priority_pair_can_change_membership(
    left: tuple[int, ...] | None,
    right: tuple[int, ...] | None,
    other_domains: Sequence[set[tuple[int, ...] | None]],
    selected_count: int,
) -> bool:
    """Exact top-k influence test for one candidate-state pair.

    Each other candidate independently chooses one state from its domain.  The
    category argument proves whether the lower state can be out while the
    higher state is in under one common assignment of all other states.
    """

    if left == right:
        return False
    valid = [value for value in (left, right) if value is not None]
    if len(valid) == 1:
        high = valid[0]
        forced_above = sum(
            all(state is not None and state > high for state in domain)
            for domain in other_domains
        )
        return forced_above < selected_count
    if not valid:
        return False
    low, high = sorted(valid)
    forced_high = 0
    middle_available = 0
    optional_high_without_middle = 0
    for domain in other_domains:
        has_high = any(state is not None and state > high for state in domain)
        has_middle = any(
            state is not None and low < state <= high for state in domain
        )
        has_low = any(state is None or state <= low for state in domain)
        if has_high and not has_middle and not has_low:
            forced_high += 1
        elif has_middle:
            middle_available += 1
        elif has_high:
            optional_high_without_middle += 1
    if forced_high >= selected_count:
        return False
    high_budget = selected_count - 1 - forced_high
    maximum_above_low = (
        forced_high
        + middle_available
        + min(high_budget, optional_high_without_middle)
    )
    return maximum_above_low >= selected_count


def selection_affecting_disagreements(
    *,
    assignment_rows: Sequence[Mapping[str, str]],
    reviewer_a: Mapping[str, Mapping[str, int]],
    audit_scores: Mapping[str, Mapping[str, int]],
    adjudicated: Mapping[tuple[str, str], int],
    subject_by_anonymous_id: Mapping[str, str],
    selected_count: int = SELECTED_PER_MECHANISM,
) -> set[tuple[str, str]]:
    """Return unresolved field disagreements with possible top-k influence."""

    assignment_by_id = {row["anonymous_review_id"]: row for row in assignment_rows}
    unresolved = _field_disagreements(reviewer_a, audit_scores) - set(adjudicated)
    by_mechanism: dict[str, list[str]] = defaultdict(list)
    for row in assignment_rows:
        by_mechanism[row["mechanism"]].append(row["anonymous_review_id"])
    domains = {
        anonymous_id: _priority_domain(
            anonymous_id,
            subject_id=subject_by_anonymous_id[anonymous_id],
            reviewer_a=reviewer_a,
            audit_scores=audit_scores,
            adjudicated=adjudicated,
        )
        for anonymous_id in assignment_by_id
    }
    affecting: set[tuple[str, str]] = set()
    for anonymous_id, field in sorted(unresolved):
        other_fields = [
            candidate_field
            for candidate_field in SCORE_FIELDS
            if candidate_field != field
            and (anonymous_id, candidate_field) in unresolved
        ]
        pairs: set[tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = set()
        value_options = [
            (reviewer_a[anonymous_id][candidate_field], audit_scores[anonymous_id][candidate_field])
            for candidate_field in other_fields
        ]
        for values in itertools.product(*value_options):
            forced = dict(zip(other_fields, values))
            left_scores = dict(reviewer_a[anonymous_id])
            right_scores = dict(reviewer_a[anonymous_id])
            for fixed_field, fixed_value in adjudicated.items():
                if fixed_field[0] == anonymous_id:
                    left_scores[fixed_field[1]] = fixed_value
                    right_scores[fixed_field[1]] = fixed_value
            left_scores.update(forced)
            right_scores.update(forced)
            left_scores[field] = reviewer_a[anonymous_id][field]
            right_scores[field] = audit_scores[anonymous_id][field]
            subject_id = subject_by_anonymous_id[anonymous_id]
            pairs.add((priority(left_scores, subject_id), priority(right_scores, subject_id)))
        mechanism = assignment_by_id[anonymous_id]["mechanism"]
        other_domains = [
            domains[other_id]
            for other_id in by_mechanism[mechanism]
            if other_id != anonymous_id
        ]
        if any(
            _priority_pair_can_change_membership(left, right, other_domains, selected_count)
            for left, right in pairs
        ):
            affecting.add((anonymous_id, field))
    return affecting


def _load_common(
    *,
    reviewer_a_completed: Path,
    reviewer_a_assignment: Path,
    agent_audit_paths: Sequence[Path],
    adjudication_path: Path | None,
    reviewer_a_binding: Path | None,
    candidate_manifest: Path | None,
    require_private_binding: bool,
) -> dict[str, Any]:
    assignment_rows, assignment_by_id = load_assignment(reviewer_a_assignment)
    reviewer_a = load_reviewer_a(reviewer_a_completed, assignment_rows)
    audit_scores, audit_metadata = load_agent_audits(agent_audit_paths, assignment_by_id)
    adjudicated, adjudication_metadata = load_adjudication(
        adjudication_path,
        reviewer_a=reviewer_a,
        audit_scores=audit_scores,
    )

    require(candidate_manifest is not None, "candidate manifest is required for candidate-ID selection hashes")
    candidate_fields, candidate_rows = screening.load_and_validate_candidates(candidate_manifest)
    candidate_by_id = {row["candidate_id"]: row for row in candidate_rows}
    binding_by_id: dict[str, dict[str, str]] = {}
    if reviewer_a_binding is not None:
        binding_by_id, subject_by_id = load_binding(
            reviewer_a_binding,
            assignment_rows=assignment_rows,
            candidate_by_id=candidate_by_id,
        )
        tie_break_basis = "candidate_id_from_private_binding"
    else:
        require(not require_private_binding, "finalize requires Reviewer-A binding and candidate manifest")
        subject_by_id = derive_candidate_mapping_from_public_assignment(
            assignment_rows=assignment_rows,
            candidate_by_id=candidate_by_id,
        )
        tie_break_basis = "candidate_id_from_exact_public_manifest_join"

    return {
        "assignment_rows": assignment_rows,
        "assignment_by_id": assignment_by_id,
        "reviewer_a": reviewer_a,
        "audit_scores": audit_scores,
        "audit_metadata": audit_metadata,
        "adjudicated": adjudicated,
        "adjudication_metadata": adjudication_metadata,
        "binding_by_id": binding_by_id,
        "candidate_by_id": candidate_by_id,
        "candidate_fields": candidate_fields,
        "subject_by_id": subject_by_id,
        "tie_break_basis": tie_break_basis,
    }


def _fresh_partial(output_root: Path) -> Path:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = output_root.parent / f".{output_root.name}.partial-{os.getpid()}"
    require(not partial.exists(), f"partial output already exists: {partial}")
    partial.mkdir()
    return partial


def _publish_partial(partial: Path, output_root: Path) -> None:
    os.replace(partial, output_root)


def plan_adjudication(
    *,
    reviewer_a_completed: Path,
    reviewer_a_assignment: Path,
    agent_audit_paths: Sequence[Path],
    reviewer_a_binding: Path | None,
    candidate_manifest: Path | None,
    output_root: Path,
) -> dict[str, Any]:
    common = _load_common(
        reviewer_a_completed=reviewer_a_completed,
        reviewer_a_assignment=reviewer_a_assignment,
        agent_audit_paths=agent_audit_paths,
        adjudication_path=None,
        reviewer_a_binding=reviewer_a_binding,
        candidate_manifest=candidate_manifest,
        require_private_binding=False,
    )
    affecting = selection_affecting_disagreements(
        assignment_rows=common["assignment_rows"],
        reviewer_a=common["reviewer_a"],
        audit_scores=common["audit_scores"],
        adjudicated={},
        subject_by_anonymous_id=common["subject_by_id"],
    )
    fields_by_id: dict[str, list[str]] = defaultdict(list)
    for anonymous_id, field in affecting:
        fields_by_id[anonymous_id].append(field)
    mechanism_index = {mechanism: index for index, mechanism in enumerate(NEW_MECHANISMS)}
    ordered_ids = sorted(
        fields_by_id,
        key=lambda anonymous_id: (
            mechanism_index[common["assignment_by_id"][anonymous_id]["mechanism"]],
            selection_hash(common["subject_by_id"][anonymous_id]),
            anonymous_id,
        ),
    )
    public_items: list[dict[str, Any]] = []
    template_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    for anonymous_id in ordered_ids:
        assignment = common["assignment_by_id"][anonymous_id]
        requested_fields = [field for field in SCORE_FIELDS if field in fields_by_id[anonymous_id]]
        public_items.append(
            {
                "anonymous_review_id": anonymous_id,
                "mechanism": assignment["mechanism"],
                "mechanism_name": assignment["mechanism_name"],
                "composite_path": assignment["composite_path"],
                "target_prompt": assignment["target_prompt"],
                "receiver": assignment["receiver"],
                "source_object": assignment["source_object"],
                "expected_trigger": assignment["expected_trigger"],
                "expected_footprint": assignment["expected_footprint"],
                "expected_counterfactual_state": assignment["expected_counterfactual_state"],
                "excluded_content_phrase": assignment["excluded_content_phrase"],
                "requested_fields": requested_fields,
            }
        )
        template_items.append(
            {
                "anonymous_review_id": anonymous_id,
                "scores": {field: None for field in requested_fields},
                "evidence": [],
            }
        )
        private_items.append(
            {
                "anonymous_review_id": anonymous_id,
                "candidate_id": common["subject_by_id"][anonymous_id],
                "requested_fields": requested_fields,
                "reviewer_a_scores": {
                    field: common["reviewer_a"][anonymous_id][field]
                    for field in requested_fields
                },
                "agent_audit_scores": {
                    field: common["audit_scores"][anonymous_id][field]
                    for field in requested_fields
                },
            }
        )

    partial = _fresh_partial(output_root)
    try:
        public = partial / "public"
        private = partial / "private"
        public.mkdir()
        private.mkdir(mode=0o700)
        assignment_path = public / "adjudication_assignment.json"
        template_path = public / "adjudication_template.json"
        instructions_path = public / "instructions.json"
        binding_path = private / "adjudication_binding.json"
        assignment_payload = {
            "schema_version": SCHEMA_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "task": "blind_field_level_adjudication_of_selection_affecting_disagreements",
            "blind_to_reviewer_a_and_agent_audit_scores": True,
            "items": public_items,
        }
        template_payload = {
            "reviewer": "",
            "blind_to_reviewer_a_and_agent_audit_scores": True,
            "items": template_items,
        }
        instructions = screening.reviewer_instructions()
        instructions.update(
            {
                "workflow_version": WORKFLOW_VERSION,
                "task": "score_only_requested_fields_without_accessing_parent_scores",
                "fill_only_requested_fields": True,
                "return_schema": list(ADJUDICATION_TOP_FIELDS),
            }
        )
        write_bytes_exclusive(assignment_path, canonical_json_bytes(assignment_payload))
        write_bytes_exclusive(template_path, canonical_json_bytes(template_payload))
        write_bytes_exclusive(instructions_path, canonical_json_bytes(instructions))
        write_bytes_exclusive(
            binding_path,
            canonical_json_bytes({"items": private_items}),
            mode=0o600,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "status": "committed_before_blind_selection_affecting_adjudication",
            "selection_analysis_tie_break_basis": common["tie_break_basis"],
            "reviewer_a_completed": file_ref(reviewer_a_completed),
            "reviewer_a_assignment": file_ref(reviewer_a_assignment),
            "agent_audits": [file_ref(path) for path in agent_audit_paths],
            "reviewer_a_binding": None if reviewer_a_binding is None else file_ref(reviewer_a_binding),
            "candidate_manifest": file_ref(candidate_manifest),
            "audited_items": len(common["audit_scores"]),
            "atomic_audit_disagreements": len(_field_disagreements(common["reviewer_a"], common["audit_scores"])),
            "selection_affecting_items": len(public_items),
            "selection_affecting_atomic_disagreements": len(affecting),
            "public_assignment": relative_file_ref(partial, assignment_path),
            "public_template": relative_file_ref(partial, template_path),
            "public_instructions": relative_file_ref(partial, instructions_path),
            "private_binding": relative_file_ref(partial, binding_path),
        }
        manifest_path = partial / "adjudication_plan_manifest.json"
        write_bytes_exclusive(manifest_path, canonical_json_bytes(manifest))
        _publish_partial(partial, output_root)
        return {
            **manifest,
            "manifest": str(output_root / manifest_path.name),
            "sha256": sha256_file(output_root / manifest_path.name),
        }
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _canonicalize(
    common: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, int | None]],
    dict[str, dict[str, list[int]]],
    list[dict[str, Any]],
]:
    reviewer_a = common["reviewer_a"]
    audit_scores = common["audit_scores"]
    adjudicated = common["adjudicated"]
    score_by_id: dict[str, dict[str, int | None]] = {}
    domains_by_id: dict[str, dict[str, list[int]]] = {}
    provenance: list[dict[str, Any]] = []
    for assignment in common["assignment_rows"]:
        anonymous_id = assignment["anonymous_review_id"]
        candidate_id = common["subject_by_id"][anonymous_id]
        canonical: dict[str, int | None] = {}
        domains: dict[str, list[int]] = {}
        for field in SCORE_FIELDS:
            a_score = reviewer_a[anonymous_id][field]
            audit_score = audit_scores.get(anonymous_id, {}).get(field)
            was_audited = field in audit_scores.get(anonymous_id, {})
            key = (anonymous_id, field)
            if not was_audited:
                value: int | None = a_score
                resolution = "reviewer_a_unreviewed"
                audit_status = "not_audited"
            elif audit_score == a_score:
                value = a_score
                resolution = "reviewer_a_agent_agreement"
                audit_status = "agreement"
            elif key in adjudicated:
                value = adjudicated[key]
                resolution = "blind_field_adjudication"
                audit_status = "disagreement"
            else:
                value = None
                resolution = "unresolved"
                audit_status = "disagreement"
                domains[field] = sorted({a_score, int(audit_score)})
            canonical[field] = value
            audit_meta = common["audit_metadata"].get(anonymous_id, {}) if was_audited else {}
            adjudication_meta = common["adjudication_metadata"]
            provenance.append(
                {
                    "anonymous_review_id": anonymous_id,
                    "candidate_id": candidate_id,
                    "mechanism": assignment["mechanism"],
                    "field": field,
                    "reviewer_a_score": a_score,
                    "audit_status": audit_status,
                    "audit_reviewer": audit_meta.get("reviewer", ""),
                    "audit_score": "" if audit_score is None else audit_score,
                    "audit_source_sha256": audit_meta.get("source_sha256", ""),
                    "adjudication_status": "resolved" if key in adjudicated else "not_adjudicated",
                    "adjudicator": adjudication_meta.get("reviewer", "") if key in adjudicated else "",
                    "adjudication_score": adjudicated.get(key, ""),
                    "adjudication_source_sha256": adjudication_meta.get("source_sha256", "") if key in adjudicated else "",
                    "canonical_score": "" if value is None else value,
                    "resolution": resolution,
                }
            )
        score_by_id[anonymous_id] = canonical
        domains_by_id[anonymous_id] = domains
    return score_by_id, domains_by_id, provenance


def _representative_scores(
    anonymous_id: str,
    canonical: Mapping[str, int | None],
    reviewer_a: Mapping[str, Mapping[str, int]],
) -> dict[str, int]:
    return {
        field: reviewer_a[anonymous_id][field] if canonical[field] is None else int(canonical[field])
        for field in SCORE_FIELDS
    }


def _load_water_screen(
    path: Path,
    *,
    candidate_by_id: Mapping[str, Mapping[str, str]],
    project_root: Path,
    media_probe: Callable[[Path], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields, rows = read_csv(path)
    require(
        {"pair_id", "video_path", "final_status"} <= set(fields),
        "historical Water screen is missing pair_id/video_path/final_status",
    )
    require(len(rows) == screening.ROWS_PER_MECHANISM, "historical Water screen must contain exactly 192 rows")
    require(
        Counter(row["final_status"] for row in rows)
        == Counter({"accept": SELECTED_PER_MECHANISM, "reject": 14}),
        "historical Water screen must freeze exactly 178 accept and 14 reject rows",
    )
    water_candidates = {
        row["candidate_id"]: row
        for row in candidate_by_id.values()
        if row["mechanism"] == WATER_MECHANISM
    }
    by_pair = {row["historical_pair_id"]: row for row in water_candidates.values()}
    expected = {
        row["candidate_id"]
        for row in water_candidates.values()
        if row.get("historical_screen_status") == "accept"
    }
    require(len(expected) == SELECTED_PER_MECHANISM, "candidate manifest does not freeze 178 accepted Water rows")
    screen_by_pair = {row["pair_id"]: row for row in rows}
    require(len(screen_by_pair) == screening.ROWS_PER_MECHANISM, "historical Water screen pair IDs repeat")
    require(set(screen_by_pair) == set(by_pair), "historical Water screen/candidate pair inventories differ")
    for pair_id, candidate in by_pair.items():
        screen_row = screen_by_pair[pair_id]
        require(
            screen_row["final_status"] == candidate["historical_screen_status"],
            f"Water pair {pair_id} screen status differs from candidate manifest",
        )
        require(
            screen_row["video_path"] == candidate["target_video_path"],
            f"Water pair {pair_id} target path differs from candidate manifest",
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        if source["final_status"] != "accept":
            continue
        pair_id = source["pair_id"]
        candidate = by_pair[pair_id]
        candidate_id = candidate["candidate_id"]
        require(candidate_id not in seen, f"Water candidate repeats: {candidate_id}")
        seen.add(candidate_id)
        declared_path = source["video_path"]
        video_path = Path(declared_path)
        resolved = video_path if video_path.is_absolute() else project_root / video_path
        regular_file(resolved, f"Water row {index} target video")
        require(resolved.suffix.lower() == ".mp4", f"Water row {index} target is not MP4")
        actual_sha256 = sha256_file(resolved)
        media = dict(media_probe(resolved))
        require(media == MEDIA_CONTRACT, f"Water row {index} decoded media contract differs")
        selected.append(
            {
                "candidate": candidate,
                "video_path": declared_path,
                "video_sha256": actual_sha256,
                "size_bytes": resolved.stat().st_size,
                "media": media,
            }
        )
    require(seen == expected, "historical Water screen accepts differ from candidate manifest")
    return selected


def probe_video_media(path: Path) -> dict[str, Any]:
    """Decode the complete video and return its exact media contract."""

    try:
        import av
    except ImportError as exc:
        raise TargetedAuditError("PyAV is required to verify selected target media") from exc
    with av.open(str(path)) as container:
        videos = [stream for stream in container.streams if stream.type == "video"]
        audios = [stream for stream in container.streams if stream.type == "audio"]
        require(len(videos) == 1 and not audios, f"selected video stream inventory differs: {path}")
        stream = videos[0]
        rate = stream.average_rate or stream.guessed_rate
        decoded_frames = sum(1 for _ in container.decode(video=stream.index))
        return {
            "decoded_frames": decoded_frames,
            "fps": "" if rate is None else f"{Fraction(rate).numerator}/{Fraction(rate).denominator}",
            "height": int(stream.height),
            "width": int(stream.width),
        }


def _verify_bound_video(
    *,
    project_root: Path,
    path_value: str,
    expected_sha256: str,
    label: str,
    media_probe: Callable[[Path], Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    require_hex64(expected_sha256, f"{label} video hash")
    path = Path(path_value)
    resolved = path if path.is_absolute() else project_root / path
    regular_file(resolved, f"{label} video")
    require(resolved.suffix.lower() == ".mp4", f"{label} video is not MP4")
    require(sha256_file(resolved) == expected_sha256, f"{label} video bytes differ from binding")
    require(resolved.stat().st_size > 0, f"{label} video is empty")
    media = dict(media_probe(resolved))
    require(media == MEDIA_CONTRACT, f"{label} decoded media contract differs")
    return resolved.stat().st_size, media


def finalize_selection(
    *,
    project_root: Path,
    reviewer_a_completed: Path,
    reviewer_a_assignment: Path,
    agent_audit_paths: Sequence[Path],
    adjudication_path: Path | None,
    reviewer_a_binding: Path,
    candidate_manifest: Path,
    water_screen_csv: Path,
    output_root: Path,
    media_probe: Callable[[Path], Mapping[str, Any]] = probe_video_media,
) -> dict[str, Any]:
    common = _load_common(
        reviewer_a_completed=reviewer_a_completed,
        reviewer_a_assignment=reviewer_a_assignment,
        agent_audit_paths=agent_audit_paths,
        adjudication_path=adjudication_path,
        reviewer_a_binding=reviewer_a_binding,
        candidate_manifest=candidate_manifest,
        require_private_binding=True,
    )
    candidate_build_registry = candidate_manifest.parent / "build_registry.json"
    if candidate_build_registry.is_file() and not candidate_build_registry.is_symlink():
        build_registry = _load_json_object(candidate_build_registry, "candidate build registry")
        require(
            build_registry.get("input_sha256", {}).get("water_screen")
            == sha256_file(water_screen_csv),
            "historical Water screen bytes differ from candidate build registry",
        )
    unresolved = _field_disagreements(common["reviewer_a"], common["audit_scores"]) - set(common["adjudicated"])
    affecting = selection_affecting_disagreements(
        assignment_rows=common["assignment_rows"],
        reviewer_a=common["reviewer_a"],
        audit_scores=common["audit_scores"],
        adjudicated=common["adjudicated"],
        subject_by_anonymous_id=common["subject_by_id"],
    )
    require(
        not affecting,
        "selection-affecting A/audit disagreements remain unresolved: "
        + ", ".join(f"{anonymous_id}/{field}" for anonymous_id, field in sorted(affecting)),
    )
    canonical_by_id, domains_by_id, provenance = _canonicalize(common)
    representative = {
        anonymous_id: _representative_scores(
            anonymous_id,
            canonical_by_id[anonymous_id],
            common["reviewer_a"],
        )
        for anonymous_id in canonical_by_id
    }
    assignment_by_id = common["assignment_by_id"]
    selected_ids_by_mechanism: dict[str, list[str]] = {}
    screening_rank_by_id: dict[str, int] = {}
    nonhard_counts: dict[str, int] = {}
    for mechanism in NEW_MECHANISMS:
        identifiers = [
            row["anonymous_review_id"]
            for row in common["assignment_rows"]
            if row["mechanism"] == mechanism
        ]
        ranked = sorted(
            (
                anonymous_id
                for anonymous_id in identifiers
                if not hard_reject(representative[anonymous_id])
            ),
            key=lambda anonymous_id: ranking_sort_key(
                representative[anonymous_id], common["subject_by_id"][anonymous_id]
            ),
        )
        nonhard_counts[mechanism] = len(ranked)
        require(
            len(ranked) >= SELECTED_PER_MECHANISM,
            f"{mechanism}: only {len(ranked)} non-hard-reject targets; need {SELECTED_PER_MECHANISM}",
        )
        for rank, anonymous_id in enumerate(ranked, start=1):
            screening_rank_by_id[anonymous_id] = rank
        selected_ids_by_mechanism[mechanism] = ranked[:SELECTED_PER_MECHANISM]

    water_selected = _load_water_screen(
        water_screen_csv,
        candidate_by_id=common["candidate_by_id"],
        project_root=project_root,
        media_probe=media_probe,
    )
    selected_candidate_ids = {
        common["subject_by_id"][anonymous_id]
        for identifiers in selected_ids_by_mechanism.values()
        for anonymous_id in identifiers
    }

    canonical_rows: list[dict[str, Any]] = []
    for assignment in common["assignment_rows"]:
        anonymous_id = assignment["anonymous_review_id"]
        candidate_id = common["subject_by_id"][anonymous_id]
        domains = domains_by_id[anonymous_id]
        canonical = canonical_by_id[anonymous_id]
        unresolved_fields = [field for field in SCORE_FIELDS if canonical[field] is None]
        possible_hard_reject = {
            hard_reject(scores)
            for scores in _score_domain(
                anonymous_id,
                reviewer_a=common["reviewer_a"],
                audit_scores=common["audit_scores"],
                adjudicated=common["adjudicated"],
            )
        }
        hard_reject_status = (
            "unresolved"
            if len(possible_hard_reject) > 1
            else "yes" if True in possible_hard_reject else "no"
        )
        canonical_rows.append(
            {
                "anonymous_review_id": anonymous_id,
                "candidate_id": candidate_id,
                "mechanism": assignment["mechanism"],
                **{field: "" if canonical[field] is None else canonical[field] for field in SCORE_FIELDS},
                "canonical_status": "resolved" if not unresolved_fields else "partially_unresolved_selection_inert",
                "unresolved_fields": ";".join(unresolved_fields),
                "score_domains_json": json.dumps(domains, sort_keys=True, separators=(",", ":")),
                "hard_reject": hard_reject_status,
                "selected": "yes" if candidate_id in selected_candidate_ids else "no",
                "screening_rank": "" if unresolved_fields else screening_rank_by_id.get(anonymous_id, ""),
            }
        )

    partial = _fresh_partial(output_root)
    try:
        selected_dir = partial / "selected_by_mechanism"
        selected_dir.mkdir()
        canonical_path = partial / "canonical_nonwater_scores.csv"
        provenance_path = partial / "audit_provenance.csv"
        write_bytes_exclusive(canonical_path, csv_bytes(canonical_rows, CANONICAL_FIELDS), mode=0o600)
        write_bytes_exclusive(provenance_path, csv_bytes(provenance, PROVENANCE_FIELDS), mode=0o600)

        selected_fields = (*common["candidate_fields"], *SELECTED_APPEND_FIELDS)
        all_selected: list[dict[str, Any]] = []
        mechanism_refs: dict[str, Any] = {}
        for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
            output_rows: list[dict[str, Any]] = []
            if mechanism == WATER_MECHANISM:
                records = [
                    {
                        "candidate": record["candidate"],
                        "scores": {field: "" for field in SCORE_FIELDS},
                        "canonical_status": "historical_water_accept",
                        "unresolved_fields": "",
                        "score_domains_json": "{}",
                        "membership_rank": "",
                        "video_path": record["video_path"],
                        "video_sha256": record["video_sha256"],
                        "size_bytes": record["size_bytes"],
                        "media": record["media"],
                    }
                    for record in water_selected
                ]
            else:
                records = []
                for anonymous_id in selected_ids_by_mechanism[mechanism]:
                    candidate_id = common["subject_by_id"][anonymous_id]
                    binding = common["binding_by_id"][anonymous_id]
                    canonical = canonical_by_id[anonymous_id]
                    unresolved_fields = [field for field in SCORE_FIELDS if canonical[field] is None]
                    size_bytes, media = _verify_bound_video(
                        project_root=project_root,
                        path_value=binding["video_path"],
                        expected_sha256=binding["video_sha256"],
                        label=candidate_id,
                        media_probe=media_probe,
                    )
                    records.append(
                        {
                            "candidate": common["candidate_by_id"][candidate_id],
                            "scores": {field: "" if canonical[field] is None else canonical[field] for field in SCORE_FIELDS},
                            "canonical_status": "resolved" if not unresolved_fields else "partially_unresolved_selection_inert",
                            "unresolved_fields": ";".join(unresolved_fields),
                            "score_domains_json": json.dumps(domains_by_id[anonymous_id], sort_keys=True, separators=(",", ":")),
                            "membership_rank": "" if unresolved_fields else screening_rank_by_id[anonymous_id],
                            "video_path": binding["video_path"],
                            "video_sha256": binding["video_sha256"],
                            "size_bytes": size_bytes,
                            "media": media,
                        }
                    )
            records.sort(
                key=lambda record: (
                    selection_hash(record["candidate"]["candidate_id"]),
                    record["candidate"]["candidate_id"],
                )
            )
            require(len(records) == SELECTED_PER_MECHANISM, f"{mechanism}: selected count differs")
            for selected_index, record in enumerate(records):
                candidate = record["candidate"]
                output_rows.append(
                    {
                        **candidate,
                        **record["scores"],
                        "canonical_status": record["canonical_status"],
                        "unresolved_fields": record["unresolved_fields"],
                        "score_domains_json": record["score_domains_json"],
                        "eligible": "yes",
                        "membership_rank": record["membership_rank"],
                        "selection_hash": selection_hash(candidate["candidate_id"]),
                        "selection_rank": selected_index + 1,
                        "selected_index": selected_index,
                        "selected_video_path": record["video_path"],
                        "selected_video_sha256": record["video_sha256"],
                        "selected_size_bytes": record["size_bytes"],
                        "selected_media": json.dumps(record["media"], sort_keys=True, separators=(",", ":")),
                    }
                )
            path = selected_dir / f"{mechanism_index:02d}_{mechanism}.csv"
            write_bytes_exclusive(path, csv_bytes(output_rows, selected_fields), mode=0o600)
            mechanism_refs[mechanism] = relative_file_ref(partial, path, row_count=len(output_rows))
            all_selected.extend(output_rows)

        require(len(all_selected) == len(MECHANISM_ORDER) * SELECTED_PER_MECHANISM, "selected total differs")
        all_path = partial / "selected_targets.csv"
        write_bytes_exclusive(all_path, csv_bytes(all_selected, selected_fields), mode=0o600)
        unresolved_by_mechanism = Counter(
            assignment_by_id[anonymous_id]["mechanism"]
            for anonymous_id, _ in unresolved
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "status": "frozen_targeted_audit_selection",
            "protocol_version": screening.PROTOCOL_VERSION,
            "selection_rule": {
                "hard_reject_if_any_zero": list(HARD_REJECT_FIELDS),
                "natural_motion": "soft_ranking_only",
                "membership_rank_descending": [
                    "min(source_absent,trigger_absent,footprint_absent)",
                    "sum(source_absent,trigger_absent,footprint_absent)",
                    "receiver_recognizable",
                    "quality",
                    "natural_motion",
                    "sum(all_six_scores)",
                ],
                "membership_tie_break": "ascending sha256('target-select-v1|' + candidate_id)",
                "published_row_order": "ascending selection hash for frozen training compatibility",
                "selected_per_mechanism": SELECTED_PER_MECHANISM,
                "unresolved_selection_inert_computation": "Reviewer-A endpoint used only as a representative; membership was proven invariant and the value is not canonicalized",
            },
            "reviewer_a_completed": file_ref(reviewer_a_completed),
            "reviewer_a_assignment": file_ref(reviewer_a_assignment),
            "agent_audits": [file_ref(path) for path in agent_audit_paths],
            "adjudication": None if adjudication_path is None else file_ref(adjudication_path),
            "reviewer_a_binding": file_ref(reviewer_a_binding),
            "candidate_manifest": file_ref(candidate_manifest),
            "water_historical_screen": file_ref(water_screen_csv),
            "audited_items": len(common["audit_scores"]),
            "atomic_audit_disagreements": len(_field_disagreements(common["reviewer_a"], common["audit_scores"])),
            "adjudicated_atomic_disagreements": len(common["adjudicated"]),
            "unresolved_selection_inert_atomic_disagreements": len(unresolved),
            "unresolved_by_mechanism": dict(sorted(unresolved_by_mechanism.items())),
            "nonhard_counts": {WATER_MECHANISM: SELECTED_PER_MECHANISM, **nonhard_counts},
            "selected_counts": {mechanism: SELECTED_PER_MECHANISM for mechanism in MECHANISM_ORDER},
            "canonical_nonwater_scores": relative_file_ref(partial, canonical_path, row_count=len(canonical_rows)),
            "audit_provenance": relative_file_ref(partial, provenance_path, row_count=len(provenance)),
            "selected_targets": relative_file_ref(partial, all_path, row_count=len(all_selected)),
            "selected_by_mechanism": mechanism_refs,
            "selected_total": len(all_selected),
        }
        registry_path = partial / "selection_registry.json"
        write_bytes_exclusive(registry_path, canonical_json_bytes(manifest), mode=0o600)
        _publish_partial(partial, output_root)
        return {
            **manifest,
            "registry": str(output_root / registry_path.name),
            "sha256": sha256_file(output_root / registry_path.name),
        }
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _resolve(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path


def _add_common_arguments(parser: argparse.ArgumentParser, *, private_optional: bool) -> None:
    parser.add_argument("--reviewer-a-completed", type=Path, required=True)
    parser.add_argument("--reviewer-a-assignment", type=Path, required=True)
    parser.add_argument("--agent-audit", type=Path, action="append", required=True)
    parser.add_argument("--reviewer-a-binding", type=Path, required=not private_optional)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan-adjudication",
        help="emit a score-blind queue of only selection-affecting disagreements",
    )
    _add_common_arguments(plan, private_optional=True)
    finalize = commands.add_parser(
        "finalize",
        help="freeze canonical/provenance tables and exactly 178 targets per mechanism",
    )
    _add_common_arguments(finalize, private_optional=False)
    finalize.add_argument("--adjudication-json", type=Path)
    finalize.add_argument(
        "--water-screen-csv",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        require(project_root.is_dir(), f"project root missing: {project_root}")
        common = {
            "reviewer_a_completed": _resolve(project_root, args.reviewer_a_completed),
            "reviewer_a_assignment": _resolve(project_root, args.reviewer_a_assignment),
            "agent_audit_paths": [_resolve(project_root, path) for path in args.agent_audit],
            "reviewer_a_binding": _resolve(project_root, args.reviewer_a_binding),
            "candidate_manifest": _resolve(project_root, args.candidate_manifest),
            "output_root": _resolve(project_root, args.output_root),
        }
        if args.command == "plan-adjudication":
            result = plan_adjudication(**common)
        elif args.command == "finalize":
            result = finalize_selection(
                project_root=project_root,
                adjudication_path=_resolve(project_root, args.adjudication_json),
                water_screen_csv=_resolve(project_root, args.water_screen_csv),
                **common,
            )
        else:  # pragma: no cover - argparse prevents this
            raise TargetedAuditError("unknown command")
    except (OSError, TargetedAuditError, screening.ScreeningError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
