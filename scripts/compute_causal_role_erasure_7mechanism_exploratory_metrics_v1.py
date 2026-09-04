#!/usr/bin/env python3
"""Compute post-unblinding exploratory metrics from blinded VLM A/B means.

This adapter is intentionally outside the formal human-canonical workflow.  It
first materializes Original-only mean-rule eligibility and shared-capability
subsets, and only then opens the full method key in this process.  Every output
is labelled NOT_CANONICAL and is unsuitable for final scientific claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import build_causal_role_erasure_7mechanism_provisional_consensus_v1 as consensus
    import compute_causal_role_erasure_7mechanism_formal_metrics_v1 as registered
except ModuleNotFoundError:  # imported as ``scripts.<module>`` in tests
    from scripts import build_causal_role_erasure_7mechanism_provisional_consensus_v1 as consensus
    from scripts import compute_causal_role_erasure_7mechanism_formal_metrics_v1 as registered


PROTOCOL = "causal_role_erasure_7m_post_unblinding_exploratory_metrics_v1"
STATUS = "post_unblinding_exploratory_vlm_ab_mean_not_human_calibrated"
CANONICAL_STATUS = "NOT_CANONICAL"
SCIENTIFIC_LABEL = "POST_UNBLINDING_EXPLORATORY_PRELIMINARY"
USE_RESTRICTION = "WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS"
FINAL_AUTHORITY = "FINAL_HUMAN_CANONICAL_SCORES_REMAIN_PRIMARY"
EXPECTED_ITEMS = 2448
EXPECTED_ORIGINAL_ITEMS = 588
COMPLETION_STATUS = "complete_post_unblinding_exploratory_metrics"
NONLINEAR_ORDER = (
    "fieldwise A/B mean first; then eligibility/usable thresholds; then CES/SU"
)
PROTOCOL_DEVIATION = {
    "occurred": True,
    "code": "full_method_key_access_preceded_provisional_decision_materialization",
    "scope": "a_separate_read_only_audit_process_opened_the_full_method_key",
    "consequence": (
        "this_analysis_is_not_a_pre_unblinding_preregistration_and_cannot_be_"
        "used_for_confirmatory_or_final_scientific_claims"
    ),
}
ANNOTATION = {
    "analysis_status": STATUS,
    "canonical_status": CANONICAL_STATUS,
    "scientific_label": SCIENTIFIC_LABEL,
    "use_restriction": USE_RESTRICTION,
    "final_authority": FINAL_AUTHORITY,
    "formal_claims_permitted": False,
}


class ExploratoryMetricsError(ValueError):
    """An input or output violates the exploratory analysis boundary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExploratoryMetricsError(message)


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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExploratoryMetricsError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    regular_file(path, label)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        require(line and line.strip() == line, f"{label} row {index} is blank/noncanonical")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExploratoryMetricsError(f"{label} row {index} is invalid JSON") from exc
        require(isinstance(row, dict), f"{label} row {index} is not an object")
        rows.append(row)
    return rows


def load_csv(path: Path, label: str) -> list[dict[str, str]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} has no header")
        return [dict(row) for row in reader]


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(0o444)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_json_bytes(value))


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    require(bool(rows), "cannot serialize an empty table")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def file_ref(path: Path) -> dict[str, Any]:
    regular_file(path, "referenced artifact")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def relative_ref(
    root: Path, path: Path, *, row_count: int | None = None
) -> dict[str, Any]:
    regular_file(path, "output artifact")
    ref = {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if row_count is not None:
        ref["row_count"] = row_count
    return ref


def _resolve_ref(root: Path, ref: Any, label: str, expected: Path) -> None:
    require(isinstance(ref, dict) and set(ref) >= {"path", "sha256"}, f"{label} reference malformed")
    candidate = Path(str(ref["path"]))
    candidate = candidate if candidate.is_absolute() else root / candidate
    regular_file(candidate, label)
    require(candidate.resolve() == expected.resolve(), f"{label} path mismatch")
    require(ref["sha256"] == sha256_file(candidate), f"{label} SHA mismatch")


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _verify_external_file_ref(ref: Any, label: str) -> Path:
    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256", "size_bytes"}
        and _valid_sha(ref.get("sha256"))
        and type(ref.get("size_bytes")) is int,
        f"{label} external file reference malformed",
    )
    path = Path(str(ref["path"]))
    require(path.is_absolute(), f"{label} external path is not absolute")
    regular_file(path, label)
    require(
        path.stat().st_size == ref["size_bytes"]
        and sha256_file(path) == ref["sha256"],
        f"{label} external input SHA/size changed",
    )
    return path.resolve()


def _resolve_output_file_ref(root: Path, ref: Any, label: str) -> Path:
    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256", "size_bytes"}
        and _valid_sha(ref.get("sha256"))
        and type(ref.get("size_bytes")) is int,
        f"{label} output reference malformed",
    )
    value = Path(str(ref["path"]))
    path = value if value.is_absolute() else root / value
    regular_file(path, label)
    require(
        path.stat().st_size == ref["size_bytes"]
        and sha256_file(path) == ref["sha256"],
        f"{label} output SHA/size changed",
    )
    return path.resolve()


def _load_bound_ab_sources(
    receipt: Mapping[str, Any], receipt_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Re-validate receipt inputs and load the authoritative A/B score rows."""

    inputs = receipt.get("inputs")
    require(isinstance(inputs, dict), "provisional receipt inputs missing")
    expected = {
        "scores_a",
        "scores_a_merge_manifest",
        "scores_b",
        "scores_b_merge_manifest",
        "public_assignments_a",
        "public_assignments_b",
        "public_key_commitments",
        "evaluation_code_registry",
        "provisional_implementation",
    }
    require(set(inputs) == expected, "provisional receipt input inventory changed")
    resolved: dict[str, Path] = {}
    for name, ref in inputs.items():
        if name != "provisional_implementation":
            resolved[name] = _verify_external_file_ref(ref, f"provisional {name}")

    snapshot_ref = receipt.get("artifacts", {}).get("implementation_snapshot")
    snapshot_path = _resolve_output_file_ref(
        receipt_root, snapshot_ref, "provisional implementation snapshot"
    )
    implementation_ref = inputs["provisional_implementation"]
    require(
        isinstance(implementation_ref, dict)
        and set(implementation_ref) == {"path", "sha256", "size_bytes"}
        and snapshot_path.stat().st_size == implementation_ref["size_bytes"]
        and sha256_file(snapshot_path) == implementation_ref["sha256"],
        "authoritative provisional implementation snapshot differs from committed source",
    )
    code_provenance = receipt.get("code_provenance")
    require(
        isinstance(code_provenance, dict)
        and code_provenance.get("live_implementation") == implementation_ref
        and code_provenance.get("implementation_snapshot") == snapshot_ref
        and code_provenance.get("source_git", {}).get(
            "implementation_snapshot_is_authoritative"
        )
        is True,
        "provisional authoritative code provenance changed",
    )
    require(
        resolved["scores_a_merge_manifest"]
        == resolved["scores_a"].parent / "merge_manifest.json"
        and resolved["scores_b_merge_manifest"]
        == resolved["scores_b"].parent / "merge_manifest.json",
        "provisional A/B merge-manifest paths changed",
    )
    try:
        rows_a = consensus.canonical.load_merged_scores(
            resolved["scores_a"],
            "A",
            EXPECTED_ITEMS,
            expected_assignments_path=resolved["public_assignments_a"],
        )
        rows_b = consensus.canonical.load_merged_scores(
            resolved["scores_b"],
            "B",
            EXPECTED_ITEMS,
            expected_assignments_path=resolved["public_assignments_b"],
        )
    except ValueError as exc:
        raise ExploratoryMetricsError(str(exc)) from exc
    by_a = {str(row["anonymous_review_id"]): row for row in rows_a}
    by_b = {str(row["anonymous_review_id"]): row for row in rows_b}
    require(
        len(by_a) == len(by_b) == EXPECTED_ITEMS and set(by_a) == set(by_b),
        "authoritative A/B anonymous inventories differ",
    )
    return by_a, by_b


def load_provisional_scores(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Load and independently verify the provisional A/B-mean receipt and rows."""

    receipt_path = path.parent / "provisional_receipt.json"
    receipt = load_json(receipt_path, "provisional receipt")
    require(
        receipt.get("protocol") == consensus.PROTOCOL
        and receipt.get("status") == STATUS
        and receipt.get("completion_status")
        == "complete_post_unblinding_exploratory_vlm_ab_mean"
        and receipt.get("scientific_label") == SCIENTIFIC_LABEL
        and receipt.get("use_restriction") == USE_RESTRICTION
        and receipt.get("final_authority") == FINAL_AUTHORITY,
        "provisional receipt identity/status changed",
    )
    require(receipt.get("protocol_deviation", {}).get("occurred") is True, "protocol deviation is not disclosed")
    require(
        receipt.get("key_and_metric_boundaries", {}).get("full_method_key_read_by_this_builder") is False
        and receipt.get("key_and_metric_boundaries", {}).get("method_metrics_computed") is False
        and receipt.get("key_and_metric_boundaries", {}).get("canonical_scores_read_or_written") is False,
        "provisional consensus crossed its key/metric boundary",
    )
    _resolve_ref(path.parent, receipt.get("artifacts", {}).get("provisional_scores"), "provisional scores", path)
    counts = receipt.get("counts", {})
    require(
        counts.get("videos") == EXPECTED_ITEMS
        and counts.get("atomic_scores") == EXPECTED_ITEMS * 4
        and counts.get("pass_a_rows") == EXPECTED_ITEMS
        and counts.get("pass_b_rows") == EXPECTED_ITEMS,
        "provisional receipt counts changed",
    )
    decision_ref = receipt.get("artifacts", {}).get("provisional_decision")
    require(isinstance(decision_ref, dict), "provisional decision reference missing")
    decision_path = Path(str(decision_ref.get("path", "")))
    decision_path = decision_path if decision_path.is_absolute() else path.parent / decision_path
    _resolve_ref(path.parent, decision_ref, "provisional decision", decision_path)
    decision = load_json(decision_path, "provisional decision")
    require(
        decision.get("decision", {}).get("rule")
        == "fieldwise_arithmetic_mean_of_blinded_pass_a_and_pass_b"
        and decision.get("decision", {}).get("rounding") == "none"
        and decision.get("status") == STATUS,
        "provisional mean decision changed",
    )
    require(
        decision.get("code_provenance") == receipt.get("code_provenance"),
        "provisional decision/receipt code provenance differs",
    )
    source_a, source_b = _load_bound_ab_sources(receipt, path.parent)

    rows = load_jsonl(path, "provisional scores")
    require(len(rows) == EXPECTED_ITEMS, "provisional score inventory must be exactly 2448")
    ids = [str(row.get("anonymous_review_id", "")) for row in rows]
    require(len(set(ids)) == EXPECTED_ITEMS and all(ids), "provisional anonymous IDs repeat or are empty")
    for index, row in enumerate(rows):
        require(set(row) == consensus.PROVISIONAL_ROW_FIELDS, f"provisional row {index} schema changed")
        kind = row.get("case_kind")
        fields = consensus.canonical.FIELDS_BY_KIND.get(kind)
        require(fields is not None and set(row.get("scores", {})) == set(fields), f"provisional row {index} score fields changed")
        require(
            row.get("status") == STATUS
            and row.get("scientific_label") == SCIENTIFIC_LABEL
            and row.get("use_restriction") == USE_RESTRICTION
            and row.get("final_authority") == FINAL_AUTHORITY
            and row.get("consensus_rule") == "fieldwise_arithmetic_mean_of_blinded_pass_a_and_pass_b",
            f"provisional row {index} labels changed",
        )
        sources = row.get("source_scores")
        require(isinstance(sources, dict) and set(sources) == {"pass_a", "pass_b"}, f"provisional row {index} source scores changed")
        require(
            set(sources["pass_a"]) == set(fields)
            and set(sources["pass_b"]) == set(fields),
            f"provisional row {index} source score fields changed",
        )
        hashes = row.get("source_score_record_sha256")
        require(
            isinstance(hashes, dict)
            and set(hashes) == {"pass_a", "pass_b"}
            and all(_valid_sha(value) for value in hashes.values()),
            f"provisional row {index} source hashes changed",
        )
        for field in fields:
            left, right = sources["pass_a"].get(field), sources["pass_b"].get(field)
            require(type(left) is int and left in (0, 1, 2), f"provisional row {index}/{field} pass A invalid")
            require(type(right) is int and right in (0, 1, 2), f"provisional row {index}/{field} pass B invalid")
            require(row["scores"][field] == (left + right) / 2, f"provisional row {index}/{field} is not the A/B mean")
        review_id = str(row["anonymous_review_id"])
        left_row, right_row = source_a.get(review_id), source_b.get(review_id)
        require(left_row is not None and right_row is not None, f"provisional row {index} lacks authoritative A/B rows")
        require(
            row["assignment_sha256"]
            == left_row.get("assignment_sha256")
            == right_row.get("assignment_sha256")
            and row["case_kind"]
            == left_row.get("case_kind")
            == right_row.get("case_kind"),
            f"provisional row {index} A/B assignment or case-kind binding changed",
        )
        require(
            row["source_scores"]["pass_a"] == left_row.get("scores")
            and row["source_scores"]["pass_b"] == right_row.get("scores"),
            f"provisional row {index} source scores differ from authoritative A/B rows",
        )
        require(
            row["source_score_record_sha256"]["pass_a"]
            == hashlib.sha256(canonical_json_bytes(left_row)).hexdigest()
            and row["source_score_record_sha256"]["pass_b"]
            == hashlib.sha256(canonical_json_bytes(right_row)).hexdigest(),
            f"provisional row {index} source row commitment changed",
        )
    return rows, receipt, receipt_path


def load_formal_cases(path: Path) -> list[dict[str, str]]:
    rows = load_csv(path, "formal cases")
    require(len(rows) == 294, "formal case inventory must be exactly 294")
    require(len({row.get("case_id") for row in rows}) == 294, "formal case IDs repeat")
    require({row.get("protocol_version") for row in rows} == {registered.PROTOCOL_VERSION}, "formal case protocol changed")
    require(
        Counter((row.get("mechanism"), row.get("case_kind")) for row in rows)
        == Counter(
            {
                **{(mechanism, "causal"): 24 for mechanism in registered.MECHANISMS},
                **{(mechanism, "specificity"): 18 for mechanism in registered.MECHANISMS},
            }
        ),
        "formal case mechanism/kind balance changed",
    )
    return rows


def _validate_key_case(binding: Mapping[str, Any], case: Mapping[str, str], review_id: str) -> None:
    for field in (
        "mechanism",
        "case_kind",
        "generalization_group",
        "source_membership",
        "prompt_style",
        "footprint_lexicalization",
        "specificity_subtype",
        "m6_pair_id",
        "source_id",
        "receiver_id",
    ):
        require(str(binding.get(field, "")) == str(case.get(field, "")), f"{review_id}: key/formal {field} mismatch")
    require(
        int(binding.get("formal_global_index", -1)) == int(case["global_case_index"])
        and int(binding.get("seed", -1)) == int(case["seed"]),
        f"{review_id}: key/formal index or seed mismatch",
    )


def load_original_key(
    path: Path,
    commitments: Mapping[str, Any],
    provisional_by_id: Mapping[str, Mapping[str, Any]],
    cases: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    regular_file(path, "Original-only key")
    require(sha256_file(path) == commitments["tier_1_original_only"]["sha256"], "Original-only key differs from its commitment")
    rows = load_jsonl(path, "Original-only key")
    require(len(rows) == EXPECTED_ORIGINAL_ITEMS, "Original-only key must contain exactly 588 rows")
    by_id = {str(row.get("anonymous_review_id", "")): row for row in rows}
    require(len(by_id) == EXPECTED_ORIGINAL_ITEMS and set(by_id) <= set(provisional_by_id), "Original-only key IDs repeat or differ")
    require(Counter(row.get("stream") for row in rows) == {"wan_original": 294, "cogvideox_original": 294}, "Original-only stream inventory changed")
    case_by_id = {row["case_id"]: row for row in cases}
    for review_id, binding in by_id.items():
        expected_backbone = "wan" if binding.get("stream") == "wan_original" else "cogvideox"
        require(binding.get("backbone_family") == expected_backbone, f"{review_id}: Original backbone mismatch")
        case = case_by_id.get(str(binding.get("case_id", "")))
        require(case is not None, f"{review_id}: Original key case missing from formal cases")
        _validate_key_case(binding, case, review_id)
        require(provisional_by_id[review_id]["case_kind"] == binding["case_kind"], f"{review_id}: score/key kind mismatch")
    return rows


def build_mean_rule_eligibility(
    provisional_by_id: Mapping[str, Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply source==2 and every other causal field>=1 to Original means."""

    eligibility: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], int] = {}
    for binding in sorted(original_rows, key=lambda row: str(row["anonymous_review_id"])):
        if binding["case_kind"] != "causal":
            continue
        review_id = str(binding["anonymous_review_id"])
        scores = provisional_by_id[review_id]["scores"]
        eligible = int(
            scores["source_visibility"] == 2
            and scores["footprint_visibility"] >= 1
            and scores["receiver_preservation"] >= 1
            and scores["video_quality"] >= 1
        )
        pair = (str(binding["case_id"]), str(binding["backbone_family"]))
        require(pair not in by_pair, f"duplicate Original causal pair: {pair}")
        by_pair[pair] = eligible
        eligibility.append(
            {
                **ANNOTATION,
                "anonymous_review_id": review_id,
                "backbone_family": binding["backbone_family"],
                "case_id": binding["case_id"],
                "mechanism": binding["mechanism"],
                "eligible": eligible,
                "source_visibility": scores["source_visibility"],
                "footprint_visibility": scores["footprint_visibility"],
                "receiver_preservation": scores["receiver_preservation"],
                "video_quality": scores["video_quality"],
            }
        )
    require(len(eligibility) == 336, "Original eligibility must contain exactly 336 causal rows")
    case_meta = {(str(row["case_id"]), str(row["mechanism"])) for row in eligibility}
    require(len(case_meta) == 168, "Original causal semantic inventory changed")
    shared: list[dict[str, Any]] = []
    for case_id, mechanism in sorted(case_meta):
        wan = by_pair.get((case_id, "wan"))
        cog = by_pair.get((case_id, "cogvideox"))
        require(wan is not None and cog is not None, f"{case_id}: missing Original backbone")
        shared.append(
            {
                **ANNOTATION,
                "case_id": case_id,
                "mechanism": mechanism,
                "wan_original_eligible": wan,
                "cogvideox_original_eligible": cog,
                "shared_eligible": int(wan == 1 and cog == 1),
            }
        )
    return eligibility, shared


def _open_full_key_after_checkpoint(
    path: Path,
    commitments: Mapping[str, Any],
    checkpoint_path: Path,
    provisional_by_id: Mapping[str, Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """This is the only full-key opening point; the stage-1 file must exist."""

    checkpoint = load_json(checkpoint_path, "Original-only checkpoint")
    require(
        checkpoint.get("status") == STATUS
        and checkpoint.get("sequence_checkpoint")
        == "original_mean_rule_eligibility_and_shared_subsets_materialized_before_this_process_opened_full_method_key"
        and checkpoint.get("full_method_key_opened_by_this_process") is False,
        "Original-only checkpoint is absent or invalid before full-key opening",
    )
    regular_file(path, "full method key")
    observed_sha = sha256_file(path)
    require(observed_sha == commitments["tier_2_full"]["sha256"], "full method key differs from its commitment")
    rows = load_jsonl(path, "full method key")
    require(len(rows) == EXPECTED_ITEMS, "full key must contain exactly 2448 rows")
    by_id = {str(row.get("anonymous_review_id", "")): row for row in rows}
    require(len(by_id) == EXPECTED_ITEMS and set(by_id) == set(provisional_by_id), "full-key/provisional ID inventories differ")
    counts = Counter((row.get("evaluation_partition"), row.get("stream")) for row in rows)
    require(
        counts
        == Counter(
            {
                **{("main", stream): 294 for stream in registered.MAIN_STREAMS},
                **{("identification", stream): 48 for stream in registered.IDENTIFICATION_STREAMS},
            }
        ),
        "full-key stream inventory changed",
    )
    for row in rows:
        expected_backbone = "cogvideox" if row.get("stream") in registered.COG_STREAMS else "wan"
        require(row.get("backbone_family") == expected_backbone, f"{row.get('anonymous_review_id')}: backbone mismatch")
    require(
        all(dict(row) == by_id.get(str(row.get("anonymous_review_id", ""))) for row in original_rows),
        "Original-only key is not the exact committed subset of the full key",
    )
    return rows, observed_sha


def _annotate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**ANNOTATION, **dict(row)} for row in rows]


def _drop_partial_label(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "eligible_source_clear_to_partial_rate"}
        for row in rows
    ]


def _exploratory_headline(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        observed = int(row.pop("registered_win"))
        if row["estimable"] == 0:
            for field in (
                "point_estimate",
                "ci95_lower",
                "ci95_upper",
                "one_sided_p",
                "holm_adjusted_p",
            ):
                row[field] = ""
            row["estimability_status"] = "not_estimable_fail_closed"
        else:
            row["estimability_status"] = "estimable"
        row["exploratory_positive_threshold_pattern"] = observed if row["estimable"] == 1 else 0
        row["formal_win_claim"] = "NOT_PERMITTED"
        output.append(row)
    return output


def _exploratory_headline_mechanism(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        if row["estimable"] == 0:
            row["point_estimate"] = ""
            row["estimability_status"] = "not_estimable_fail_closed"
        else:
            row["estimability_status"] = "estimable"
        output.append(row)
    return output


def _rename_flag(rows: Sequence[Mapping[str, Any]], old: str, new: str) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        row[new] = int(row.pop(old))
        output.append(row)
    return output


def build_ab_order_sensitivity_macro(
    provisional_rows: Sequence[Mapping[str, Any]],
    full_key_rows: Sequence[Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, str]],
    primary_macro: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Contrast primary mean-before-threshold order with passwise thresholding."""

    pass_macros: dict[str, dict[str, Mapping[str, Any]]] = {}
    for pass_name in ("pass_a", "pass_b"):
        pass_scores = [
            {
                "anonymous_review_id": row["anonymous_review_id"],
                "case_kind": row["case_kind"],
                "scores": dict(row["source_scores"][pass_name]),
            }
            for row in provisional_rows
        ]
        pass_by_id = {str(row["anonymous_review_id"]): row for row in pass_scores}
        pass_eligibility, _ = build_mean_rule_eligibility(pass_by_id, original_rows)
        pass_case_metrics = registered.build_case_metrics(
            pass_scores, full_key_rows, pass_eligibility, cases
        )
        _, macro = registered.mechanism_and_macro_tables(pass_case_metrics)
        pass_macros[pass_name] = {str(row["stream"]): row for row in macro}

    primary_by_stream = {str(row["stream"]): row for row in primary_macro}
    require(
        set(primary_by_stream)
        == set(pass_macros["pass_a"])
        == set(pass_macros["pass_b"])
        == set(registered.MAIN_STREAMS),
        "A/B sensitivity macro stream inventory changed",
    )
    metrics = (
        "original_capability_coverage",
        "causal_usable_fraction",
        "specificity_usable_fraction",
        "ces",
        "specificity_utility",
    )
    output: list[dict[str, Any]] = []
    for stream in registered.MAIN_STREAMS:
        left = pass_macros["pass_a"][stream]
        right = pass_macros["pass_b"][stream]
        primary = primary_by_stream[stream]
        row: dict[str, Any] = {
            "stream": stream,
            "sensitivity_only": True,
            "replaces_primary": False,
            "primary_nonlinear_order": NONLINEAR_ORDER,
            "sensitivity_order": (
                "within each pass eligibility/usable thresholds then CES/SU and macro; "
                "then arithmetic mean of pass A and pass B macros"
            ),
        }
        for metric in metrics:
            a_value = float(left[metric])
            b_value = float(right[metric])
            pass_mean = (a_value + b_value) / 2
            primary_value = float(primary[metric])
            row[f"pass_a_{metric}"] = a_value
            row[f"pass_b_{metric}"] = b_value
            row[f"mean_of_pass_macros_{metric}"] = pass_mean
            row[f"primary_atomic_mean_{metric}"] = primary_value
            row[f"mean_of_pass_macros_minus_primary_{metric}"] = (
                pass_mean - primary_value
            )
        output.append(row)
    require(len(output) == 8, "A/B order sensitivity macro must contain eight streams")
    return output


def _write_table_pair(staging: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = _annotate(rows)
    csv_path = staging / f"{name}.csv"
    json_path = staging / f"{name}.json"
    write_bytes(csv_path, csv_bytes(annotated))
    write_json(json_path, annotated)
    return {
        "csv": relative_ref(staging, csv_path, row_count=len(annotated)),
        "json": relative_ref(staging, json_path, row_count=len(annotated)),
    }


def _verify_table_refs(
    staging: Path,
    table_refs: Mapping[str, Mapping[str, Any]],
    expected_counts: Mapping[str, int],
) -> None:
    for name, formats in table_refs.items():
        require(set(formats) == {"csv", "json"}, f"{name} table formats changed")
        for format_name, ref in formats.items():
            require(
                isinstance(ref, dict)
                and set(ref) == {"path", "sha256", "size_bytes", "row_count"}
                and ref["row_count"] == expected_counts[name],
                f"{name}/{format_name} table reference/count changed",
            )
            value = Path(str(ref["path"]))
            require(not value.is_absolute() and ".." not in value.parts, f"{name}/{format_name} path escaped output")
            path = staging / value
            regular_file(path, f"{name}/{format_name} table")
            require(
                path.stat().st_size == ref["size_bytes"]
                and sha256_file(path) == ref["sha256"],
                f"{name}/{format_name} table changed before publish",
            )
        csv_rows = load_csv(staging / str(formats["csv"]["path"]), f"{name} CSV")
        json_value = json.loads(
            (staging / str(formats["json"]["path"])).read_text(encoding="utf-8")
        )
        require(
            isinstance(json_value, list)
            and len(csv_rows) == len(json_value) == expected_counts[name],
            f"{name} CSV/JSON row counts differ",
        )


def _revalidate_before_publish(
    *,
    provisional_scores_path: Path,
    original_scores: Sequence[Mapping[str, Any]],
    original_receipt: Mapping[str, Any],
    external_inputs: Mapping[str, Mapping[str, Any]],
    implementation_snapshot_path: Path,
    table_refs: Mapping[str, Mapping[str, Any]],
    expected_table_counts: Mapping[str, int],
    staging: Path,
) -> None:
    """Re-hash every input and output binding as the final publication gate."""

    scores, receipt, _ = load_provisional_scores(provisional_scores_path)
    require(list(scores) == list(original_scores), "provisional scores changed during metrics")
    require(dict(receipt) == dict(original_receipt), "provisional receipt changed during metrics")
    for name, ref in external_inputs.items():
        _verify_external_file_ref(ref, f"exploratory {name}")
    implementation_ref = external_inputs["exploratory_metrics_implementation"]
    require(
        implementation_snapshot_path.stat().st_size == implementation_ref["size_bytes"]
        and sha256_file(implementation_snapshot_path) == implementation_ref["sha256"],
        "exploratory implementation snapshot differs from executed source",
    )
    _verify_table_refs(staging, table_refs, expected_table_counts)


def compute_exploratory_metrics(
    *,
    provisional_scores_path: Path,
    original_key_path: Path,
    full_key_path: Path,
    key_commitments_path: Path,
    formal_cases_path: Path,
    output_root: Path,
    bootstrap_replicates: int = registered.BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    require(bootstrap_replicates == registered.BOOTSTRAP_REPLICATES, "exploratory adapter reuses the frozen 10,000-replicate computation")
    output_root = output_root.resolve()
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    staging.chmod(0o700)
    try:
        implementation_path = Path(__file__).resolve()
        implementation_ref = file_ref(implementation_path)
        implementation_snapshot_path = staging / "implementation_snapshot.py"
        write_bytes(implementation_snapshot_path, implementation_path.read_bytes())
        code_provenance = {
            "live_implementation": implementation_ref,
            "implementation_snapshot": relative_ref(
                staging, implementation_snapshot_path
            ),
            "source_git": consensus.capture_git_state(implementation_path.parents[1]),
        }
        scores, provisional_receipt, provisional_receipt_path = load_provisional_scores(provisional_scores_path)
        try:
            commitments, registry_path, registry = consensus.load_public_commitments(key_commitments_path)
        except consensus.ProvisionalConsensusError as exc:
            raise ExploratoryMetricsError(str(exc)) from exc
        _resolve_ref(
            provisional_scores_path.parent,
            provisional_receipt.get("inputs", {}).get("public_key_commitments"),
            "provisional key commitments",
            key_commitments_path,
        )
        require(
            provisional_receipt.get("evaluation_code_registry_sha256") == registry["registry_sha256"],
            "provisional/evaluation-code registry binding changed",
        )
        cases = load_formal_cases(formal_cases_path)
        provisional_by_id = {str(row["anonymous_review_id"]): row for row in scores}

        # Stage 1: use only the committed Original-only key.  Do not stat, hash,
        # read, or reference the full key before this checkpoint is on disk.
        original_rows = load_original_key(original_key_path, commitments, provisional_by_id, cases)
        eligibility, shared = build_mean_rule_eligibility(provisional_by_id, original_rows)
        eligibility_path = staging / "original_mean_rule_eligibility.csv"
        eligibility_json_path = staging / "original_mean_rule_eligibility.json"
        shared_path = staging / "shared_capability_subsets.csv"
        shared_json_path = staging / "shared_capability_subsets.json"
        write_bytes(eligibility_path, csv_bytes(eligibility))
        write_json(eligibility_json_path, eligibility)
        write_bytes(shared_path, csv_bytes(shared))
        write_json(shared_json_path, shared)
        checkpoint = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "completion_status": "original_only_checkpoint_complete",
            "canonical_status": CANONICAL_STATUS,
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
            "sequence_checkpoint": "original_mean_rule_eligibility_and_shared_subsets_materialized_before_this_process_opened_full_method_key",
            "full_method_key_opened_by_this_process": False,
            "full_method_key_public_commitment_sha256": commitments["tier_2_full"]["sha256"],
            "protocol_deviation": PROTOCOL_DEVIATION,
            "eligibility_rule": "source_visibility_mean==2 and every_other_causal_mean>=1",
            "nonlinear_order": NONLINEAR_ORDER,
            "counts": {
                "original_key_rows": len(original_rows),
                "original_causal_rows": len(eligibility),
                "shared_semantic_cases": len(shared),
                "shared_eligible_cases": sum(int(row["shared_eligible"]) for row in shared),
            },
            "inputs": {
                "provisional_scores": file_ref(provisional_scores_path),
                "provisional_receipt": file_ref(provisional_receipt_path),
                "original_only_key": file_ref(original_key_path),
                "key_commitments": file_ref(key_commitments_path),
                "formal_cases": file_ref(formal_cases_path),
                "evaluation_code_registry": file_ref(registry_path),
                "exploratory_metrics_implementation": implementation_ref,
            },
            "code_provenance": code_provenance,
            "artifacts": {
                "original_mean_rule_eligibility_csv": relative_ref(
                    staging, eligibility_path, row_count=len(eligibility)
                ),
                "original_mean_rule_eligibility_json": relative_ref(
                    staging, eligibility_json_path, row_count=len(eligibility)
                ),
                "shared_capability_subsets_csv": relative_ref(
                    staging, shared_path, row_count=len(shared)
                ),
                "shared_capability_subsets_json": relative_ref(
                    staging, shared_json_path, row_count=len(shared)
                ),
            },
        }
        checkpoint_path = staging / "original_only_checkpoint.json"
        write_json(checkpoint_path, checkpoint)

        # Stage 2: the full key is opened only after the preceding file exists.
        full_rows, full_key_sha = _open_full_key_after_checkpoint(
            full_key_path, commitments, checkpoint_path, provisional_by_id, original_rows
        )
        try:
            case_metrics = registered.build_case_metrics(scores, full_rows, eligibility, cases)
            mechanism, macro = registered.mechanism_and_macro_tables(case_metrics)
            headline, headline_mechanism, _ = registered.headline_contrasts(
                case_metrics, shared, replicates=bootstrap_replicates
            )
            noninferiority, _ = registered.noninferiority_table(
                case_metrics, replicates=bootstrap_replicates
            )
            identification, _ = registered.identification_table(
                case_metrics, replicates=bootstrap_replicates
            )
            role, _ = registered.main_role_conditioned_evidence(
                case_metrics, replicates=bootstrap_replicates
            )
            m6 = registered.m6_table(case_metrics, cases)
            sensitivity = build_ab_order_sensitivity_macro(
                scores, full_rows, original_rows, cases, macro
            )
        except registered.FormalMetricsError as exc:
            raise ExploratoryMetricsError(str(exc)) from exc

        mechanism = _drop_partial_label(mechanism)
        macro = _drop_partial_label(macro)
        headline = _exploratory_headline(headline)
        headline_mechanism = _exploratory_headline_mechanism(headline_mechanism)
        noninferiority = _rename_flag(
            noninferiority, "noninferior", "exploratory_noninferiority_threshold_pattern"
        )
        identification = _rename_flag(
            identification, "novelty_condition_pass", "exploratory_novelty_threshold_pattern"
        )
        role = _rename_flag(role, "registered_positive", "exploratory_positive_point_pattern")

        table_values = {
            "per_case_metrics": case_metrics,
            "mechanism_metrics": mechanism,
            "macro_metrics": macro,
            "headline_contrasts": headline,
            "headline_mechanism_contrasts": headline_mechanism,
            "noninferiority": noninferiority,
            "identification_contrasts": identification,
            "role_conditioned_evidence": role,
            "m6_pairs": m6,
            "ab_order_sensitivity_macro": sensitivity,
        }
        table_refs = {
            name: _write_table_pair(staging, name, values)
            for name, values in table_values.items()
        }
        cross_backbone = [row for row in headline if row["contrast"] != "matched_control"]
        results = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "completion_status": COMPLETION_STATUS,
            "canonical_status": CANONICAL_STATUS,
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
            "formal_claims_permitted": False,
            "human_calibrated": False,
            "protocol_deviation": PROTOCOL_DEVIATION,
            "sequence": {
                "original_only_checkpoint_materialized_before_full_key_opened_by_this_process": True,
                "global_pre_unblinding_preregistration": False,
            },
            "score_semantics": "each_atomic_score_is_the_unrounded_arithmetic_mean_of_two_blinded_vlm_passes",
            "eligibility_rule": "source_visibility_mean==2 and every_other_causal_mean>=1",
            "nonlinear_order": NONLINEAR_ORDER,
            "aggregation": "case_then_mechanism_mean_then_equal_weight_7_mechanism_macro",
            "omitted_metrics": {
                "eligible_source_clear_to_partial_rate": "omitted because a mean score of 1 is not an exact partial label",
                "secondary_metrics_table": "omitted duplicate projection; only partial-rate removed for ambiguity",
            },
            "bootstrap": {
                "replicates": registered.BOOTSTRAP_REPLICATES,
                "headline_pcg64_seed": registered.HEADLINE_SEED,
                "identification_pcg64_seed": registered.IDENTIFICATION_SEED,
                "interpretation": "exploratory_only_not_confirmatory",
            },
            "estimability": {
                "minimum_shared_cases_per_mechanism": registered.MIN_SHARED_CASES_PER_MECHANISM,
                "cross_backbone_contrasts_estimable": all(row["estimable"] == 1 for row in cross_backbone),
                "fail_closed": True,
            },
            "ab_order_sensitivity": {
                "table": "ab_order_sensitivity_macro",
                "row_count": len(sensitivity),
                "sensitivity_only": True,
                "replaces_primary": False,
                "description": (
                    "pass A and pass B independently apply eligibility/usable thresholds "
                    "and CES/SU aggregation; their macro mean is compared with the primary "
                    "atomic-mean-first result"
                ),
            },
            "exploratory_threshold_patterns": {
                "headline_positive_count": sum(int(row["exploratory_positive_threshold_pattern"]) for row in headline),
                "headline_total": len(headline),
                "primary_noninferiority_pattern_count": sum(
                    int(row["exploratory_noninferiority_threshold_pattern"])
                    for row in noninferiority
                    if row["role"] == "primary"
                ),
                "primary_noninferiority_total": sum(row["role"] == "primary" for row in noninferiority),
                "identification_pattern_count": sum(int(row["exploratory_novelty_threshold_pattern"]) for row in identification),
                "identification_total": len(identification),
                "role_conditioned_positive_count": sum(int(row["exploratory_positive_point_pattern"]) for row in role),
                "role_conditioned_total": len(role),
                "formal_win_claim": "NOT_PERMITTED",
            },
            "headline_contrasts": headline,
            "noninferiority": noninferiority,
            "identification_contrasts": identification,
            "role_conditioned_evidence": role,
            "tables": table_refs,
        }
        results_path = staging / "results.json"
        write_json(results_path, results)
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "completion_status": COMPLETION_STATUS,
            "canonical_status": CANONICAL_STATUS,
            "scientific_label": SCIENTIFIC_LABEL,
            "use_restriction": USE_RESTRICTION,
            "final_authority": FINAL_AUTHORITY,
            "formal_claims_permitted": False,
            "protocol_deviation": PROTOCOL_DEVIATION,
            "nonlinear_order": NONLINEAR_ORDER,
            "full_method_key_opened_by_this_process_after_original_only_checkpoint": True,
            "code_provenance": code_provenance,
            "inputs": {
                **checkpoint["inputs"],
                "full_method_key": {
                    "path": str(full_key_path.resolve()),
                    "sha256": full_key_sha,
                    "size_bytes": full_key_path.stat().st_size,
                },
                "original_only_checkpoint": relative_ref(staging, checkpoint_path),
            },
            "results": relative_ref(staging, results_path),
            "tables": table_refs,
            "all_inputs_revalidated_immediately_before_publish": True,
        }
        manifest_path = staging / "exploratory_metrics_manifest.json"
        write_json(manifest_path, manifest)
        _revalidate_before_publish(
            provisional_scores_path=provisional_scores_path,
            original_scores=scores,
            original_receipt=provisional_receipt,
            external_inputs={
                **checkpoint["inputs"],
                "full_method_key": manifest["inputs"]["full_method_key"],
            },
            implementation_snapshot_path=implementation_snapshot_path,
            table_refs=table_refs,
            expected_table_counts={
                name: len(values) for name, values in table_values.items()
            },
            staging=staging,
        )
        require(not output_root.exists(), f"output appeared during build: {output_root}")
        os.replace(staging, output_root)
        return results
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--provisional-scores", type=Path, required=True)
    parser.add_argument("--original-key", type=Path, required=True)
    parser.add_argument("--full-key", type=Path, required=True)
    parser.add_argument("--key-commitments", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        results = compute_exploratory_metrics(
            provisional_scores_path=_resolve(root, args.provisional_scores),
            original_key_path=_resolve(root, args.original_key),
            full_key_path=_resolve(root, args.full_key),
            key_commitments_path=_resolve(root, args.key_commitments),
            formal_cases_path=_resolve(root, args.formal_cases),
            output_root=_resolve(root, args.output_root),
        )
    except ExploratoryMetricsError as exc:
        parser.exit(2, f"exploratory metrics refused: {exc}\n")
    print(
        json.dumps(
            {
                "status": results["status"],
                "canonical_status": results["canonical_status"],
                "formal_claims_permitted": results["formal_claims_permitted"],
                "estimability": results["estimability"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
