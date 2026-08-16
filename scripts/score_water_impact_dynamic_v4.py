#!/usr/bin/env python3
"""Freeze blind reviews, unblind, and score the registered v4 gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import water_impact_dynamic_v4_eval_protocol as protocol
import select_water_impact_dynamic_v4_eval as selector
from build_water_impact_dynamic_v4_blind_review import (
    review_binding_sha256,
    validate_blocked_assignment,
    validate_review_package,
    validate_review_package_commitment,
)


CAUSAL_SHORT_FIELDS = {
    "target": protocol.CAUSAL_SCORE_FIELDS[0],
    "footprint": protocol.CAUSAL_SCORE_FIELDS[1],
    "receiver": protocol.CAUSAL_SCORE_FIELDS[2],
    "quality": protocol.CAUSAL_SCORE_FIELDS[3],
    "causal_link": protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0],
}
SPECIFICITY_SHORT_FIELDS = {
    "protected": protocol.SPECIFICITY_SCORE_FIELDS[0],
    "receiver": protocol.SPECIFICITY_SCORE_FIELDS[1],
    "quality": protocol.SPECIFICITY_SCORE_FIELDS[2],
    "adherence": protocol.SPECIFICITY_SCORE_FIELDS[3],
}


def _score(row: Mapping[str, Any], field: str, label: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid score for {field}") from exc
    if value not in {0, 1, 2}:
        raise ValueError(f"{label}: {field} must be 0, 1, or 2")
    return value


def _scored_fields(dataset: str, arm_code: str) -> dict[str, str]:
    if dataset == "causal":
        return {
            key: value
            for key, value in CAUSAL_SHORT_FIELDS.items()
            if key != "causal_link" or arm_code == "O"
        }
    if dataset == "specificity":
        return SPECIFICITY_SHORT_FIELDS
    raise ValueError("unknown review dataset")


def derive_dispute_template(
    dataset: str,
    template_rows: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    template, a, b = _validate_reviewer_inputs(dataset, template_rows, reviewer_a, reviewer_b)
    output: list[dict[str, str]] = []
    for review_id in sorted(template):
        arm = template[review_id]["arm_code"]
        for short, field in _scored_fields(dataset, arm).items():
            left = _score(a[review_id], field, f"reviewer A/{review_id}")
            right = _score(b[review_id], field, f"reviewer B/{review_id}")
            if left != right:
                output.append({"review_id": review_id, "field": short})
    return output


def _validate_reviewer_inputs(
    dataset: str,
    template_rows: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Mapping[str, str]], dict[str, Mapping[str, str]], dict[str, Mapping[str, str]]]:
    expected_n = 3 * protocol.UNIT_COUNTS[dataset]
    if any(len(rows) != expected_n for rows in (template_rows, reviewer_a, reviewer_b)):
        raise ValueError(f"{dataset}: template and two reviews must each contain {expected_n} rows")
    mappings = []
    for label, rows in (("template", template_rows), ("reviewer A", reviewer_a), ("reviewer B", reviewer_b)):
        mapping = {str(row.get("review_id", "")): row for row in rows}
        if len(mapping) != expected_n or "" in mapping:
            raise ValueError(f"{dataset}: duplicate/blank review ID in {label}")
        mappings.append(mapping)
    template, a, b = mappings
    if set(template) != set(a) or set(template) != set(b):
        raise ValueError("reviewer IDs differ from frozen public template")
    columns = set(template_rows[0])
    if any(set(row) != columns for row in [*template_rows, *reviewer_a, *reviewer_b]):
        raise ValueError("review sheets must retain exact public columns")
    protocol.validate_public_review_columns(template_rows)
    all_score_fields = (
        {*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS}
        if dataset == "causal"
        else set(protocol.SPECIFICITY_SCORE_FIELDS)
    )
    metadata = columns - all_score_fields - {"notes"}
    for review_id, frozen in template.items():
        arm = str(frozen.get("arm_code", ""))
        if arm not in protocol.ARM_CODES:
            raise ValueError(f"{review_id}: unexpected arm code")
        expected_scored = set(_scored_fields(dataset, arm).values())
        for field in all_score_fields:
            if str(frozen.get(field, "")) != "":
                raise ValueError("frozen public template must contain blank score cells")
            for label, reviewer in (("A", a[review_id]), ("B", b[review_id])):
                if field in expected_scored:
                    _score(reviewer, field, f"reviewer {label}/{review_id}")
                elif str(reviewer.get(field, "")) != "":
                    raise ValueError(f"{review_id}: candidate-only-inapplicable field must remain blank")
        for reviewer in (a[review_id], b[review_id]):
            for field in metadata:
                if str(reviewer.get(field, "")) != str(frozen.get(field, "")):
                    raise ValueError(f"{review_id}: reviewer changed frozen metadata {field}")
    return template, a, b


def merge_blind_reviews(
    dataset: str,
    template_rows: Sequence[Mapping[str, str]],
    reviewer_a: Sequence[Mapping[str, str]],
    reviewer_b: Sequence[Mapping[str, str]],
    dispute_rows: Sequence[Mapping[str, str]],
    adjudication_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use agreement, else three-reviewer majority with 0/1/2 median semantics."""

    template, a, b = _validate_reviewer_inputs(dataset, template_rows, reviewer_a, reviewer_b)
    expected_disputes = derive_dispute_template(dataset, template_rows, reviewer_a, reviewer_b)
    if [dict(row) for row in dispute_rows] != expected_disputes:
        raise ValueError("blank dispute template is not the exact atomic disagreement set")
    expected_keys = {(row["review_id"], row["field"]) for row in expected_disputes}
    adjudication: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in adjudication_rows:
        if set(row) != {"review_id", "field", "score", "brief_reason"}:
            raise ValueError("adjudication sheet columns are not exact")
        key = (str(row["review_id"]), str(row["field"]))
        if key not in expected_keys or key in adjudication:
            raise ValueError(f"unexpected or duplicate adjudication: {key}")
        _score(row, "score", f"adjudicator/{key[0]}/{key[1]}")
        if not str(row["brief_reason"]).strip():
            raise ValueError("every adjudication requires a brief blinded reason")
        adjudication[key] = row
    if set(adjudication) != expected_keys:
        raise ValueError("every atomic disagreement requires blinded adjudication")

    canonical: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    all_fields = (
        (*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS)
        if dataset == "causal"
        else protocol.SPECIFICITY_SCORE_FIELDS
    )
    short_by_long = {long: short for short, long in {**CAUSAL_SHORT_FIELDS, **SPECIFICITY_SHORT_FIELDS}.items()}
    for frozen in template_rows:
        review_id = str(frozen["review_id"])
        output = {field: value for field, value in frozen.items() if field not in {*all_fields, "notes"}}
        adjudicated: list[str] = []
        scored = set(_scored_fields(dataset, str(frozen["arm_code"])).values())
        for field in all_fields:
            if field not in scored:
                output[field] = ""
                continue
            left = _score(a[review_id], field, f"reviewer A/{review_id}")
            right = _score(b[review_id], field, f"reviewer B/{review_id}")
            if left == right:
                output[field] = left
                continue
            short = short_by_long[field]
            third = _score(adjudication[(review_id, short)], "score", f"adjudicator/{review_id}/{short}")
            final = int(statistics.median((left, right, third)))
            output[field] = final
            adjudicated.append(short)
            audit.append(
                {
                    "review_id": review_id,
                    "field": short,
                    "reviewer_a": left,
                    "reviewer_b": right,
                    "adjudicator": third,
                    "canonical": final,
                }
            )
        output["notes"] = "two_reviewer_agreement" if not adjudicated else f"adjudicated:{','.join(adjudicated)}"
        canonical.append(output)
    return canonical, audit


def freeze_review_artifacts(
    *,
    project_root: Path,
    dataset: str,
    package_commitment_path: Path,
    template_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    dispute_path: Path,
    adjudication_path: Path,
    canonical_path: Path,
    audit_path: Path,
    freeze_manifest_path: Path,
) -> dict[str, Any]:
    for path in (canonical_path, audit_path, freeze_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen review artifact: {path}")
    validate_review_package_commitment(
        project_root,
        dataset=dataset,
        commitment_path=package_commitment_path,
        template_path=template_path,
    )
    template = protocol.read_csv(template_path)
    reviewer_a = protocol.read_csv(reviewer_a_path)
    reviewer_b = protocol.read_csv(reviewer_b_path)
    expected_disputes = derive_dispute_template(dataset, template, reviewer_a, reviewer_b)
    if not dispute_path.is_file():
        protocol.write_csv(dispute_path, expected_disputes, fieldnames=("review_id", "field"))
    canonical, audit = merge_blind_reviews(
        dataset,
        template,
        reviewer_a,
        reviewer_b,
        protocol.read_csv(dispute_path),
        protocol.read_csv(adjudication_path),
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=False)
    protocol.write_csv(canonical_path, canonical)
    protocol.write_csv(
        audit_path,
        audit,
        fieldnames=("review_id", "field", "reviewer_a", "reviewer_b", "adjudicator", "canonical"),
    )
    manifest = {
        "protocol": protocol.FINAL_REVIEW_FREEZE_PROTOCOL,
        "dataset": dataset,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen_before_answer_key_opening",
        "review_binding_sha256": review_binding_sha256(template),
        "review_package_commitment": {
            "path": str(package_commitment_path),
            "sha256": protocol.file_sha256(package_commitment_path),
        },
        "artifacts": {
            "public_template": {"path": str(template_path), "sha256": protocol.file_sha256(template_path)},
            "reviewer_a": {"path": str(reviewer_a_path), "sha256": protocol.file_sha256(reviewer_a_path)},
            "reviewer_b": {"path": str(reviewer_b_path), "sha256": protocol.file_sha256(reviewer_b_path)},
            "dispute_template": {"path": str(dispute_path), "sha256": protocol.file_sha256(dispute_path)},
            "adjudication": {"path": str(adjudication_path), "sha256": protocol.file_sha256(adjudication_path)},
            "canonical_anonymous": {"path": str(canonical_path), "sha256": protocol.file_sha256(canonical_path)},
            "adjudication_audit": {"path": str(audit_path), "sha256": protocol.file_sha256(audit_path)},
        },
    }
    freeze_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def validate_review_freeze(
    project_root: Path,
    path: Path,
    *,
    dataset: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "protocol",
        "dataset",
        "dataset_version",
        "status",
        "review_binding_sha256",
        "review_package_commitment",
        "artifacts",
    }:
        raise ValueError("final review-freeze manifest fields are not exact")
    if (
        payload["protocol"] != protocol.FINAL_REVIEW_FREEZE_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != protocol.DATASET_VERSION
        or payload["status"] != "frozen_before_answer_key_opening"
    ):
        raise ValueError("final review artifacts were not frozen before unblinding")
    required = {
        "public_template",
        "reviewer_a",
        "reviewer_b",
        "dispute_template",
        "adjudication",
        "canonical_anonymous",
        "adjudication_audit",
    }
    if not isinstance(payload["artifacts"], dict) or set(payload["artifacts"]) != required:
        raise ValueError("review-freeze artifact inventory is not exact")
    commitment_ref = payload["review_package_commitment"]
    if not isinstance(commitment_ref, dict) or set(commitment_ref) != {"path", "sha256"}:
        raise ValueError("review-freeze package commitment ref is not exact")
    commitment_path = protocol.resolve_path(project_root, str(commitment_ref["path"]))
    if (
        not commitment_path.is_file()
        or commitment_path.is_symlink()
        or protocol.file_sha256(commitment_path) != commitment_ref["sha256"]
    ):
        raise ValueError("review-freeze package commitment byte hash mismatch")
    paths: dict[str, Path] = {}
    for name, record in payload["artifacts"].items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"review-freeze/{name}: record is not exact")
        artifact = protocol.resolve_path(project_root, str(record["path"]))
        if not artifact.is_file() or artifact.is_symlink() or protocol.file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"review-freeze/{name}: byte hash mismatch")
        paths[name] = artifact
    template = protocol.read_csv(paths["public_template"])
    validate_review_package_commitment(
        project_root,
        dataset=dataset,
        commitment_path=commitment_path,
        template_path=paths["public_template"],
    )
    paths["review_package_commitment"] = commitment_path
    if payload["review_binding_sha256"] != review_binding_sha256(template):
        raise ValueError("review-freeze public binding mismatch")
    canonical, audit = merge_blind_reviews(
        dataset,
        template,
        protocol.read_csv(paths["reviewer_a"]),
        protocol.read_csv(paths["reviewer_b"]),
        protocol.read_csv(paths["dispute_template"]),
        protocol.read_csv(paths["adjudication"]),
    )
    if canonical != _typed_canonical(protocol.read_csv(paths["canonical_anonymous"]), dataset):
        raise ValueError("frozen canonical anonymous scores do not recompute")
    if audit != _typed_audit(protocol.read_csv(paths["adjudication_audit"])):
        raise ValueError("frozen adjudication audit does not recompute")
    return payload, paths


def _typed_canonical(rows: Sequence[Mapping[str, str]], dataset: str) -> list[dict[str, Any]]:
    score_fields = (
        (*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS)
        if dataset == "causal"
        else protocol.SPECIFICITY_SCORE_FIELDS
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in score_fields:
            if item[field] != "":
                item[field] = int(item[field])
        output.append(item)
    return output


def _typed_audit(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            **{field: int(row[field]) for field in ("reviewer_a", "reviewer_b", "adjudicator", "canonical")},
        }
        for row in rows
    ]


def _usable_causal(row: Mapping[str, Any]) -> bool:
    return int(row[protocol.CAUSAL_SCORE_FIELDS[2]]) >= 1 and int(row[protocol.CAUSAL_SCORE_FIELDS[3]]) >= 1


def _strict_causal(row: Mapping[str, Any]) -> bool:
    return tuple(int(row[field]) for field in protocol.CAUSAL_SCORE_FIELDS) == (0, 0, 2, 2)


def _usable_specificity(row: Mapping[str, Any]) -> bool:
    return int(row[protocol.SPECIFICITY_SCORE_FIELDS[1]]) >= 1 and int(row[protocol.SPECIFICITY_SCORE_FIELDS[2]]) >= 1


def _index_scored_rows(
    rows: Sequence[Mapping[str, Any]], dataset: str
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], dict[str, Mapping[str, Any]]]:
    expected_n = protocol.UNIT_COUNTS[dataset]
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    metadata: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        unit_id = str(row.get("unit_id", ""))
        method = str(row.get("method", ""))
        if not unit_id or method not in protocol.METHODS or method in by_unit[unit_id]:
            raise ValueError(f"{dataset}: duplicate/invalid unit-method row")
        by_unit[unit_id][method] = row
        metadata.setdefault(unit_id, row)
    if len(rows) != 3 * expected_n or len(by_unit) != expected_n or any(set(arms) != set(protocol.METHODS) for arms in by_unit.values()):
        raise ValueError(f"{dataset}: scoring requires exact paired O/v3b/v4 inventory")
    case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    invariants = (
        case_field,
        "group",
        "membership",
        "prompt_variant",
        "replicate",
        "seed",
        "source_id",
        "source_phrase",
        "receiver_id",
        "receiver",
    )
    for unit_id, arms in by_unit.items():
        first = next(iter(arms.values()))
        for row in arms.values():
            if any(str(row.get(field, "")) != str(first.get(field, "")) for field in invariants):
                raise ValueError(f"{dataset}/{unit_id}: semantic metadata differs across arms")
        fields = (
            protocol.CAUSAL_SCORE_FIELDS
            if dataset == "causal"
            else protocol.SPECIFICITY_SCORE_FIELDS
        )
        for method, row in arms.items():
            for field in fields:
                _score(row, field, f"{dataset}/{unit_id}/{method}")
        if dataset == "causal":
            _score(arms["original"], protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0], f"causal/{unit_id}/original")
    return by_unit, metadata


def _cluster_bootstrap(gains: Sequence[int]) -> dict[str, Any]:
    if not gains:
        return {"n_complete_cases": 0, "mean": None, "percentile_95": [None, None]}
    rng = random.Random(protocol.GATE_SPEC["secondary_cluster_bootstrap"]["seed"])
    iterations = int(protocol.GATE_SPEC["secondary_cluster_bootstrap"]["iterations"])
    means = sorted(sum(rng.choice(gains) for _ in gains) / len(gains) for _ in range(iterations))
    lower = means[math.floor(0.025 * (iterations - 1))]
    upper = means[math.ceil(0.975 * (iterations - 1))]
    return {
        "n_complete_cases": len(gains),
        "mean": sum(gains) / len(gains),
        "percentile_95": [lower, upper],
        "iterations": iterations,
        "seed": protocol.GATE_SPEC["secondary_cluster_bootstrap"]["seed"],
        "gate_override": False,
    }


def compute_causal_gate(
    rows: Sequence[Mapping[str, Any]], *, provenance_valid: bool = True
) -> dict[str, Any]:
    by_unit, metadata = _index_scored_rows(rows, "causal")
    target, footprint, receiver, quality = protocol.CAUSAL_SCORE_FIELDS
    causal_link = protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]
    E = {
        unit_id
        for unit_id, arms in by_unit.items()
        if int(arms["original"][target]) == 2
        and int(arms["original"][footprint]) >= 1
        and int(arms["original"][receiver]) >= 1
        and int(arms["original"][quality]) >= 1
        and int(arms["original"][causal_link]) == 2
    }
    C = {unit_id for unit_id in E if _usable_causal(by_unit[unit_id]["v3b"])}
    case_units: dict[str, set[str]] = defaultdict(set)
    for unit_id, row in metadata.items():
        case_units[str(row["semantic_case_id"])].add(unit_id)
    K = {case_id for case_id, units in case_units.items() if len(units) == 3 and units <= C}
    C_hold = {unit_id for unit_id in C if str(metadata[unit_id]["group"]) in protocol.HOLDOUT_GROUPS}

    def suppression(unit_id: str, method: str, field: str) -> int:
        row = by_unit[unit_id][method]
        return 2 - int(row[field]) if _usable_causal(row) else 0

    def delta(units: Iterable[str], field: str) -> int:
        return sum(suppression(unit_id, "v4", field) - suppression(unit_id, "v3b", field) for unit_id in units)

    def subset(**criteria: Any) -> set[str]:
        return {
            unit_id
            for unit_id in C
            if all(str(metadata[unit_id].get(field)) == str(value) for field, value in criteria.items())
        }

    paired_improvements = {
        unit_id
        for unit_id in C
        if _usable_causal(by_unit[unit_id]["v4"])
        and int(by_unit[unit_id]["v4"][target]) < int(by_unit[unit_id]["v3b"][target])
    }
    clear_to_absent = {
        unit_id
        for unit_id in C
        if _usable_causal(by_unit[unit_id]["v4"])
        and int(by_unit[unit_id]["v3b"][target]) == 2
        and int(by_unit[unit_id]["v4"][target]) == 0
    }
    strict_v4 = {unit_id for unit_id in C if _usable_causal(by_unit[unit_id]["v4"]) and _strict_causal(by_unit[unit_id]["v4"])}
    paired_strict = {unit_id for unit_id in strict_v4 if not _strict_causal(by_unit[unit_id]["v3b"])}
    case_gain = {case_id: delta(case_units[case_id], target) for case_id in K}
    case_clear_counts = {
        case_id: len(case_units[case_id] & clear_to_absent) for case_id in K
    }
    k_cell_counts = Counter(
        (str(metadata[next(iter(case_units[case]))]["group"]), str(metadata[next(iter(case_units[case]))]["prompt_variant"]))
        for case in K
    )
    target_by_replicate = {str(rep): delta(subset(replicate=rep), target) for rep in range(3)}
    target_by_group = {group: delta(subset(group=group), target) for group in protocol.CAUSAL_GROUPS}
    target_by_variant = {variant: delta(subset(prompt_variant=variant), target) for variant in protocol.PROMPT_VARIANTS}
    target_by_cell = {
        f"{group}|{variant}": delta(subset(group=group, prompt_variant=variant), target)
        for group in protocol.CAUSAL_GROUPS
        for variant in protocol.PROMPT_VARIANTS
    }
    footprint_by_group = {group: delta(subset(group=group), footprint) for group in protocol.CAUSAL_GROUPS}
    footprint_by_cell = {
        f"{group}|{variant}": delta(subset(group=group, prompt_variant=variant), footprint)
        for group in protocol.CAUSAL_GROUPS
        for variant in protocol.PROMPT_VARIANTS
    }
    c2a_hold = clear_to_absent & C_hold
    c2a_hold_cases = {str(metadata[unit_id]["semantic_case_id"]) for unit_id in c2a_hold}
    strict_hold = {unit_id for unit_id in strict_v4 if unit_id in C_hold}
    strict_cases = {str(metadata[unit_id]["semantic_case_id"]) for unit_id in strict_v4}
    paired_strict_hold_cases = {
        str(metadata[unit_id]["semantic_case_id"]) for unit_id in paired_strict if unit_id in C_hold
    }
    v3b_absent = sum(int(by_unit[unit_id]["v3b"][target]) == 0 for unit_id in C)
    v4_absent = sum(_usable_causal(by_unit[unit_id]["v4"]) and int(by_unit[unit_id]["v4"][target]) == 0 for unit_id in C)
    v3b_receiver = sum(int(arms["v3b"][receiver]) for arms in by_unit.values())
    v4_receiver = sum(int(arms["v4"][receiver]) for arms in by_unit.values())
    v3b_quality = sum(int(arms["v3b"][quality]) for arms in by_unit.values())
    v4_quality = sum(int(arms["v4"][quality]) for arms in by_unit.values())
    v4_usable = sum(_usable_causal(arms["v4"]) for arms in by_unit.values())
    positive_holdout_cases = {
        case_id
        for case_id, gain in case_gain.items()
        if gain > 0 and str(metadata[next(iter(case_units[case_id]))]["group"]) in protocol.HOLDOUT_GROUPS
    }
    strict_groups = {str(metadata[unit_id]["group"]) for unit_id in strict_v4}
    strict_variants = {str(metadata[unit_id]["prompt_variant"]) for unit_id in strict_v4}
    checks = {
        "01_all_provenance_valid": bool(provenance_valid),
        "02_E_at_least_66_and_C_at_least_64": len(E) >= 66 and len(C) >= 64,
        "03_K_at_least_20_and_each_cell_at_least_3": len(K) >= 20 and all(k_cell_counts[(g, v)] >= 3 for g in protocol.CAUSAL_GROUPS for v in protocol.PROMPT_VARIANTS),
        "04_target_delta_C_at_least_18": delta(C, target) >= 18,
        "05_each_replicate_target_delta_at_least_3": all(value >= 3 for value in target_by_replicate.values()),
        "06_each_group_and_variant_target_delta_at_least_3": all(value >= 3 for value in [*target_by_group.values(), *target_by_variant.values()]),
        "07_each_cell_target_delta_nonnegative": all(value >= 0 for value in target_by_cell.values()),
        "08_paired_usable_target_improvements_at_least_18": len(paired_improvements) >= 18,
        "09_positive_case_gain_at_least_10": sum(gain > 0 for gain in case_gain.values()) >= 10,
        "10_holdout_target_delta_at_least_12_and_improvements_at_least_12": delta(C_hold, target) >= 12 and len(paired_improvements & C_hold) >= 12,
        "11_positive_holdout_case_gain_at_least_7": len(positive_holdout_cases) >= 7,
        "12_clear_to_absent_at_least_6": len(clear_to_absent) >= 6,
        "13_holdout_clear_to_absent_four_cases_cover_groups_variants": len(c2a_hold) >= 4 and len(c2a_hold_cases) >= 4 and {str(metadata[u]["group"]) for u in c2a_hold} == set(protocol.HOLDOUT_GROUPS) and {str(metadata[u]["prompt_variant"]) for u in c2a_hold} == set(protocol.PROMPT_VARIANTS),
        "14_two_holdout_cases_clear_to_absent_on_two_replicates": sum(count >= 2 and str(metadata[next(iter(case_units[case]))]["group"]) in protocol.HOLDOUT_GROUPS for case, count in case_clear_counts.items()) >= 2,
        "15_usable_absent_target_margin_at_least_6": v4_absent >= v3b_absent + 6,
        "16_v4_usable_at_least_68": v4_usable >= 68,
        "17_receiver_floor": v4_receiver >= max(114, v3b_receiver - 6),
        "18_quality_floor": v4_quality >= max(96, v3b_quality - 6),
        "19_footprint_nonworse_global_holdout_groups_and_cell_minus1": delta(C, footprint) >= 0 and delta(C_hold, footprint) >= 0 and all(value >= 0 for value in footprint_by_group.values()) and all(value >= -1 for value in footprint_by_cell.values()),
        "20_strict_success_count_holdout_and_coverage": len(strict_v4) >= 6 and len(strict_hold) >= 4 and strict_groups == set(protocol.CAUSAL_GROUPS) and strict_variants == set(protocol.PROMPT_VARIANTS) and len(strict_cases) >= 4,
        "21_paired_strict_gain_and_holdout_case_floors": len(paired_strict) >= 4 and len(paired_strict_hold_cases) >= 3,
    }
    return {
        "protocol": protocol.PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "set_sizes": {"U": 72, "E": len(E), "C": len(C), "C_hold": len(C_hold), "K": len(K)},
        "sets": {"E": sorted(E), "C": sorted(C), "K": sorted(K)},
        "target_delta": {
            "C": delta(C, target),
            "C_hold": delta(C_hold, target),
            "by_replicate": target_by_replicate,
            "by_group": target_by_group,
            "by_variant": target_by_variant,
            "by_cell": target_by_cell,
        },
        "footprint_delta": {"C": delta(C, footprint), "C_hold": delta(C_hold, footprint), "by_group": footprint_by_group, "by_cell": footprint_by_cell},
        "paired_target_improvement_units": sorted(paired_improvements),
        "clear_to_absent_units": sorted(clear_to_absent),
        "strict_v4_units": sorted(strict_v4),
        "paired_strict_gain_units": sorted(paired_strict),
        "case_target_gain": case_gain,
        "case_clear_to_absent_count": case_clear_counts,
        "usable_absent_target": {"v3b_on_C": v3b_absent, "v4_on_C": v4_absent},
        "all_U_totals": {
            "v3b_receiver": v3b_receiver,
            "v4_receiver": v4_receiver,
            "v3b_quality": v3b_quality,
            "v4_quality": v4_quality,
            "v4_usable": v4_usable,
        },
        "cluster_bootstrap": _cluster_bootstrap(list(case_gain.values())),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _partition_ids(
    base: set[str], metadata: Mapping[str, Mapping[str, Any]], **criteria: str
) -> set[str]:
    return {
        unit_id for unit_id in base if all(str(metadata[unit_id].get(field)) == value for field, value in criteria.items())
    }


def compute_specificity_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_unit, metadata = _index_scored_rows(rows, "specificity")
    protected, receiver, quality, adherence = protocol.SPECIFICITY_SCORE_FIELDS
    H = {
        unit_id
        for unit_id, arms in by_unit.items()
        if int(arms["original"][protected]) == 2
        and int(arms["original"][receiver]) >= 1
        and int(arms["original"][quality]) >= 1
        and int(arms["original"][adherence]) == 2
    }
    D = {unit_id for unit_id in H if _usable_specificity(by_unit[unit_id]["v3b"])}
    case_units: dict[str, set[str]] = defaultdict(set)
    for unit_id, row in metadata.items():
        case_units[str(row["specificity_case_id"])].add(unit_id)
    K_D = {case_id for case_id, units in case_units.items() if len(units) == 2 and units <= D}

    def points(units: Iterable[str], method: str, field: str) -> int:
        return sum(int(by_unit[unit_id][method][field]) if _usable_specificity(by_unit[unit_id][method]) else 0 for unit_id in units)

    def absent(units: Iterable[str], method: str) -> int:
        return sum(_usable_specificity(by_unit[unit_id][method]) and int(by_unit[unit_id][method][protected]) == 0 for unit_id in units)

    partition_specs: dict[str, dict[str, str]] = {}
    for membership in protocol.SPECIFICITY_MEMBERSHIPS:
        partition_specs[f"membership:{membership}"] = {"membership": membership}
    for variant in protocol.PROMPT_VARIANTS:
        partition_specs[f"variant:{variant}"] = {"prompt_variant": variant}
    for membership in protocol.SPECIFICITY_MEMBERSHIPS:
        for variant in protocol.PROMPT_VARIANTS:
            partition_specs[f"cell:{membership}|{variant}"] = {"membership": membership, "prompt_variant": variant}

    def floor(units: set[str]) -> int:
        return math.ceil(1.5 * len(units))

    def compare_metric(field: str) -> tuple[dict[str, Any], dict[str, bool], dict[str, bool]]:
        details: dict[str, Any] = {}
        baseline: dict[str, bool] = {}
        treatment: dict[str, bool] = {}
        b_global = points(D, "v3b", field)
        v_global = points(D, "v4", field)
        details["global_D"] = {"n": len(D), "floor": floor(D), "v3b": b_global, "v4": v_global}
        baseline["global_D"] = b_global >= floor(D)
        treatment["global_D"] = v_global >= max(b_global - 3, floor(D))
        for label, criteria in partition_specs.items():
            units = _partition_ids(D, metadata, **criteria)
            b_value = points(units, "v3b", field)
            v_value = points(units, "v4", field)
            margin = 1 if label.startswith("cell:") else 2
            details[label] = {"n": len(units), "floor": floor(units), "v3b": b_value, "v4": v_value, "margin": margin}
            baseline[label] = b_value >= floor(units)
            treatment[label] = v_value >= floor(units) and v_value >= b_value - margin
        return details, baseline, treatment

    pv_details, pv_baseline, pv_treatment = compare_metric(protected)
    nr_details, nr_baseline, nr_treatment = compare_metric(adherence)
    h_floor_checks: dict[str, bool] = {}
    for label, criteria in {"global_H": {}, **partition_specs}.items():
        units = H if not criteria else _partition_ids(H, metadata, **criteria)
        h_floor_checks[f"PV/{label}"] = points(units, "v4", protected) >= floor(units)
        h_floor_checks[f"NR/{label}"] = points(units, "v4", adherence) >= floor(units)
    membership_h = {membership: len(_partition_ids(H, metadata, membership=membership)) for membership in protocol.SPECIFICITY_MEMBERSHIPS}
    membership_d = {membership: len(_partition_ids(D, metadata, membership=membership)) for membership in protocol.SPECIFICITY_MEMBERSHIPS}
    cell_d = {
        f"{membership}|{variant}": len(_partition_ids(D, metadata, membership=membership, prompt_variant=variant))
        for membership in protocol.SPECIFICITY_MEMBERSHIPS
        for variant in protocol.PROMPT_VARIANTS
    }
    v3b_absent_membership = {membership: absent(_partition_ids(D, metadata, membership=membership), "v3b") for membership in protocol.SPECIFICITY_MEMBERSHIPS}
    v4_absent_d_membership = {membership: absent(_partition_ids(D, metadata, membership=membership), "v4") for membership in protocol.SPECIFICITY_MEMBERSHIPS}
    v4_absent_h_membership = {membership: absent(_partition_ids(H, metadata, membership=membership), "v4") for membership in protocol.SPECIFICITY_MEMBERSHIPS}
    absent_baseline = absent(D, "v3b") <= 3 and all(value <= 1 for value in v3b_absent_membership.values())
    absent_treatment = (
        absent(D, "v4") <= min(absent(D, "v3b") + 1, 3)
        and absent(H, "v4") <= 3
        and all(value <= 1 for value in v4_absent_d_membership.values())
        and all(value <= 1 for value in v4_absent_h_membership.values())
    )
    v4_usable = sum(_usable_specificity(arms["v4"]) for arms in by_unit.values())
    v3b_receiver = sum(int(arms["v3b"][receiver]) for arms in by_unit.values())
    v4_receiver = sum(int(arms["v4"][receiver]) for arms in by_unit.values())
    v3b_quality = sum(int(arms["v3b"][quality]) for arms in by_unit.values())
    v4_quality = sum(int(arms["v4"][quality]) for arms in by_unit.values())
    checks = {
        "01_H_at_least_33_and_D_at_least_32": len(H) >= 33 and len(D) >= 32,
        "02_membership_and_cell_validity_floors": all(value >= 11 for value in membership_h.values()) and all(value >= 10 for value in membership_d.values()) and all(value >= 5 for value in cell_d.values()),
        "03_K_D_at_least_15": len(K_D) >= 15,
        "04_global_PV_baseline_floor_and_v4_margin": pv_baseline["global_D"] and pv_treatment["global_D"],
        "05_partition_PV_baseline_floors_and_v4_margins": all(value for key, value in pv_baseline.items() if key != "global_D") and all(value for key, value in pv_treatment.items() if key != "global_D"),
        "06_NR_baseline_floors_and_v4_margins": all(nr_baseline.values()) and all(nr_treatment.values()),
        "07_v4_absolute_PV_NR_floors_on_H_partitions": all(h_floor_checks.values()),
        "08_usable_absent_object_caps": absent_baseline and absent_treatment,
        "09_v4_usable_at_least_33": v4_usable >= 33,
        "10_receiver_floor": v4_receiver >= max(57, v3b_receiver - 3),
        "11_quality_floor": v4_quality >= max(48, v3b_quality - 3),
    }
    baseline_validity = {
        "condition_04_v3b_global_PV_floor": pv_baseline["global_D"],
        "condition_05_v3b_partition_PV_floors": all(value for key, value in pv_baseline.items() if key != "global_D"),
        "condition_06_v3b_global_and_partition_NR_floors": all(nr_baseline.values()),
        "condition_08_v3b_absent_caps": absent_baseline,
    }
    return {
        "protocol": protocol.PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "set_sizes": {"W": 36, "H": len(H), "D": len(D), "K_D": len(K_D)},
        "sets": {"H": sorted(H), "D": sorted(D), "K_D": sorted(K_D)},
        "membership_H": membership_h,
        "membership_D": membership_d,
        "cell_D": cell_d,
        "PV": pv_details,
        "NR": nr_details,
        "H_absolute_floor_checks": h_floor_checks,
        "usable_absent_protected": {
            "v3b_D": absent(D, "v3b"),
            "v4_D": absent(D, "v4"),
            "v4_H": absent(H, "v4"),
            "v3b_D_by_membership": v3b_absent_membership,
            "v4_D_by_membership": v4_absent_d_membership,
            "v4_H_by_membership": v4_absent_h_membership,
        },
        "all_W_totals": {
            "v4_usable": v4_usable,
            "v3b_receiver": v3b_receiver,
            "v4_receiver": v4_receiver,
            "v3b_quality": v3b_quality,
            "v4_quality": v4_quality,
        },
        "baseline_validity_checks": baseline_validity,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compute_role_selectivity_gate(
    causal_rows: Sequence[Mapping[str, Any]],
    specificity_rows: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    *,
    causal_gate: Mapping[str, Any] | None = None,
    specificity_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    causal_by_unit, causal_meta = _index_scored_rows(causal_rows, "causal")
    spec_by_unit, spec_meta = _index_scored_rows(specificity_rows, "specificity")
    causal_cases: dict[str, list[str]] = defaultdict(list)
    spec_cases: dict[str, list[str]] = defaultdict(list)
    for unit_id, row in causal_meta.items():
        causal_cases[str(row["semantic_case_id"])].append(unit_id)
    for unit_id, row in spec_meta.items():
        spec_cases[str(row["specificity_case_id"])].append(unit_id)
    protocol.validate_holdout_mapping(
        mapping_rows,
        causal_cases=[causal_meta[units[0]] for units in causal_cases.values()],
        specificity_cases=[spec_meta[units[0]] for units in spec_cases.values()],
    )
    causal_gate = causal_gate or compute_causal_gate(causal_rows)
    specificity_gate = specificity_gate or compute_specificity_gate(specificity_rows)
    K = set(causal_gate["sets"]["K"])
    K_D = set(specificity_gate["sets"]["K_D"])
    target = protocol.CAUSAL_SCORE_FIELDS[0]
    protected, _, _, adherence = protocol.SPECIFICITY_SCORE_FIELDS
    records: list[dict[str, Any]] = []
    for mapping in mapping_rows:
        causal_id = str(mapping["causal_case_id"])
        spec_id = str(mapping["specificity_case_id"])
        complete = causal_id in K and spec_id in K_D
        causal_units = causal_cases[causal_id]
        spec_units = spec_cases[spec_id]
        gain = int(causal_gate["case_target_gain"].get(causal_id, 0))
        all_causal_usable = all(_usable_causal(causal_by_unit[unit]["v4"]) for unit in causal_units)
        all_spec_usable = all(_usable_specificity(spec_by_unit[unit]["v4"]) for unit in spec_units)
        pv_v4 = sum(int(spec_by_unit[unit]["v4"][protected]) for unit in spec_units) if all_spec_usable else 0
        pv_v3b = sum(int(spec_by_unit[unit]["v3b"][protected]) for unit in spec_units)
        nr_v4 = sum(int(spec_by_unit[unit]["v4"][adherence]) for unit in spec_units) if all_spec_usable else 0
        nr_v3b = sum(int(spec_by_unit[unit]["v3b"][adherence]) for unit in spec_units)
        no_absent = all(int(spec_by_unit[unit]["v4"][protected]) != 0 for unit in spec_units)
        adherence_two = all(int(spec_by_unit[unit]["v4"][adherence]) == 2 for unit in spec_units)
        clear_to_absent = any(
            int(causal_by_unit[unit]["v3b"][target]) == 2
            and _usable_causal(causal_by_unit[unit]["v4"])
            and int(causal_by_unit[unit]["v4"][target]) == 0
            for unit in causal_units
        )
        role_selective = (
            complete
            and gain > 0
            and all_causal_usable
            and pv_v4 >= pv_v3b
            and nr_v4 >= nr_v3b
            and all_spec_usable
            and no_absent
            and adherence_two
        )
        records.append(
            {
                "causal_case_id": causal_id,
                "specificity_case_id": spec_id,
                "group": str(causal_meta[causal_units[0]]["group"]),
                "prompt_variant": str(causal_meta[causal_units[0]]["prompt_variant"]),
                "complete": complete,
                "causal_target_gain": gain,
                "v4_causal_all_usable": all_causal_usable,
                "v3b_specificity_PV": pv_v3b,
                "v4_specificity_PV": pv_v4,
                "v3b_specificity_NR": nr_v3b,
                "v4_specificity_NR": nr_v4,
                "v4_specificity_all_usable": all_spec_usable,
                "v4_specificity_no_absent": no_absent,
                "v4_specificity_adherence_all_2": adherence_two,
                "clear_to_absent_replicate": clear_to_absent,
                "role_selective": role_selective,
            }
        )
    complete = [row for row in records if row["complete"]]
    selective = [row for row in records if row["role_selective"]]
    checks = {
        "mapping_exactly_6_one_to_one": len(records) == 6,
        "complete_pairs_at_least_5": len(complete) >= 5,
        "role_selective_pairs_at_least_3": len(selective) >= 3,
        "role_selective_covers_both_groups_and_variants": {row["group"] for row in selective} == set(protocol.HOLDOUT_GROUPS) and {row["prompt_variant"] for row in selective} == set(protocol.PROMPT_VARIANTS),
        "role_selective_clear_to_absent_at_least_2": sum(bool(row["clear_to_absent_replicate"]) for row in selective) >= 2,
    }
    return {"mapping_pairs": records, "checks": checks, "passed": all(checks.values())}


def classify_post_checkpoint_outcome(
    causal_gate: Mapping[str, Any],
    specificity_gate: Mapping[str, Any],
    role_gate: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = bool(causal_gate["checks"]["01_all_provenance_valid"])
    if not provenance:
        outcome = "invalid_run"
    else:
        evaluation_valid = (
            bool(causal_gate["checks"]["02_E_at_least_66_and_C_at_least_64"])
            and bool(causal_gate["checks"]["03_K_at_least_20_and_each_cell_at_least_3"])
            and all(bool(specificity_gate["checks"][key]) for key in (
                "01_H_at_least_33_and_D_at_least_32",
                "02_membership_and_cell_validity_floors",
                "03_K_D_at_least_15",
            ))
            and bool(role_gate["checks"]["complete_pairs_at_least_5"])
            and all(bool(value) for value in specificity_gate["baseline_validity_checks"].values())
        )
        if not evaluation_valid:
            outcome = "inconclusive_invalid_evaluation"
        elif causal_gate["passed"] and specificity_gate["passed"] and role_gate["passed"]:
            outcome = "eligible_for_separate_main_experiment_preregistration"
        else:
            outcome = "valid_negative_ablation"
    return {
        "outcome": outcome,
        "promote_v4": outcome == "eligible_for_separate_main_experiment_preregistration",
        "sealed_final36_action": "remain_unopened; no scorer-side generation or access",
    }


def classify_precheckpoint_outcome(reason: str) -> dict[str, Any]:
    allowed = {
        "preflight_dataset_invalid",
        "registered_scale_sanity_termination",
        "invalid_training_run",
    }
    if reason not in allowed:
        raise ValueError("unregistered pre-checkpoint terminal outcome")
    return {
        "outcome": reason,
        "promote_v4": False,
        "sealed_final36_action": "remain_unopened",
    }


def _require_private_file(path: Path, *, private_root: Path, label: str) -> None:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("scorer private root must be a real evaluator-only directory")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: private artifact is missing")
    try:
        path.resolve(strict=True).relative_to(private_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}: artifact escapes evaluator-only private root") from exc


def _require_private_committed_file(
    path: Path,
    *,
    private_root: Path,
    commitment: Mapping[str, Any],
    label: str,
) -> None:
    _require_private_file(path, private_root=private_root, label=label)
    if (
        path.stat().st_size != commitment["size_bytes"]
        or protocol.file_sha256(path) != commitment["sha256"]
    ):
        raise ValueError(f"{label}: bytes differ from Stage-1 commitment")


def _validate_selected_unit_binding(
    selected_rows: Sequence[Mapping[str, Any]],
    unit_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> None:
    """Require U/W to be the exact registered replication of selected cases."""

    case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    expected_replicates = set(range(protocol.REPLICATES[dataset]))
    selected = {str(row.get(case_field, "")): row for row in selected_rows}
    if len(selected) != protocol.CASE_COUNTS[dataset] or "" in selected:
        raise ValueError(f"{dataset}: selected-case identity inventory is not exact")
    unit_fields = set(selected_rows[0]) | {"unit_id", "replicate", "seed"}
    if any(set(row) != set(selected_rows[0]) for row in selected_rows):
        raise ValueError(f"{dataset}: selected-case columns are inconsistent")
    if any(set(row) != unit_fields for row in unit_rows):
        raise ValueError(f"{dataset}: unit columns do not exactly extend selected-case columns")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in unit_rows:
        case_id = str(unit.get(case_field, ""))
        case = selected.get(case_id)
        if case is None:
            raise ValueError(f"{dataset}: unit references an unselected case")
        if any(str(unit[field]) != str(case[field]) for field in case):
            raise ValueError(f"{dataset}/{case_id}: unit scientific fields differ from selected case")
        by_case[case_id].append(unit)
    if set(by_case) != set(selected) or any(
        {int(row["replicate"]) for row in rows} != expected_replicates
        for rows in by_case.values()
    ):
        raise ValueError(f"{dataset}: unit replication differs from selected-case inventory")


def _validate_scores_against_units(
    scored_rows: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> None:
    unit_by_id = {str(row["unit_id"]): row for row in units}
    if len(unit_by_id) != protocol.UNIT_COUNTS[dataset]:
        raise ValueError(f"{dataset}: committed unit inventory is not exact")
    case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    fields = (
        case_field,
        "group",
        "membership",
        "prompt_variant",
        "replicate",
        "seed",
        "source_id",
        "source_phrase",
        "receiver_id",
        "receiver",
    )
    for row in scored_rows:
        unit = unit_by_id.get(str(row["unit_id"]))
        if unit is None:
            raise ValueError(f"{dataset}: scored row references an unknown committed unit")
        if any(str(row.get(field, "")) != str(unit.get(field, "")) for field in fields):
            raise ValueError(f"{dataset}/{row['unit_id']}: score metadata differs from committed unit")


def _validate_cross_dataset_video_isolation(
    causal_package: Mapping[str, Any],
    specificity_package: Mapping[str, Any],
) -> None:
    causal_hashes = set(causal_package["_validated_source_video_sha256"])
    specificity_hashes = set(specificity_package["_validated_source_video_sha256"])
    if causal_hashes & specificity_hashes:
        raise ValueError("causal and specificity generation inventories reuse video bytes")
def _unblind(
    dataset: str,
    canonical_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    expected_n = 3 * protocol.UNIT_COUNTS[dataset]
    key = {row["review_id"]: row for row in key_rows}
    if len(canonical_rows) != expected_n or len(key) != expected_n or set(key) != {str(row["review_id"]) for row in canonical_rows}:
        raise ValueError("answer key does not match exact canonical anonymous inventory")
    output: list[dict[str, Any]] = []
    for row in canonical_rows:
        record = key[str(row["review_id"])]
        if str(row["arm_code"]) != record["arm_code"]:
            raise ValueError("answer-key arm mismatch")
        method = record["method"]
        if method not in protocol.METHODS:
            raise ValueError("answer key contains an unexpected method")
        common = {
            "review_id": str(row["review_id"]),
            "unit_id": record["unit_id"],
            "method": method,
            "group": record["group"],
            "membership": record["membership"],
            "prompt_variant": record["prompt_variant"],
            "replicate": int(record["replicate"]),
            "seed": int(record["seed"]),
            "source_id": record["source_id"],
            "source_phrase": record["source_phrase"],
            "receiver_id": record["receiver_id"],
            "receiver": record["receiver"],
        }
        if dataset == "causal":
            common["semantic_case_id"] = record["case_id"]
            for field in protocol.CAUSAL_SCORE_FIELDS:
                common[field] = int(row[field])
            common[protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]] = (
                int(row[protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]]) if method == "original" else ""
            )
        else:
            common["specificity_case_id"] = record["case_id"]
            for field in protocol.SPECIFICITY_SCORE_FIELDS:
                common[field] = int(row[field])
        output.append(common)
    return output


def freeze_gate_registry(output_path: Path, scorer_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite machine gate registry: {output_path}")
    payload = {
        "protocol": protocol.GATE_REGISTRY_PROTOCOL,
        "status": "frozen",
        "dataset_version": protocol.DATASET_VERSION,
        "sealed_final36_status": "unopened",
        "gate_spec": protocol.GATE_SPEC,
        "gate_spec_sha256": protocol.canonical_json_sha256(protocol.GATE_SPEC),
        "scorer_sha256": protocol.file_sha256(scorer_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _cmd_freeze(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(*[value for value in vars(args).values() if isinstance(value, Path)])
    freeze_review_artifacts(
        project_root=Path.cwd(),
        dataset=args.dataset,
        package_commitment_path=args.review_package_commitment,
        template_path=args.review_template,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        dispute_path=args.dispute_template,
        adjudication_path=args.adjudication,
        canonical_path=args.canonical_anonymous,
        audit_path=args.adjudication_audit,
        freeze_manifest_path=args.freeze_manifest,
    )
    return 0


def _cmd_freeze_gate(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(args.output)
    freeze_gate_registry(args.output, Path(__file__))
    return 0


def _cmd_score_impl(args: argparse.Namespace) -> int:
    protocol.reject_sealed_final36_path(*[value for value in vars(args).values() if isinstance(value, Path)])
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite score directory: {args.output_dir}")
    project_root = Path.cwd()
    expected_authorization = protocol.resolve_path(project_root, protocol.TRAINING_AUTHORIZATION)
    expected_eligibility = protocol.resolve_path(project_root, protocol.CHECKPOINT_ELIGIBILITY)
    if args.training_authorization.resolve() != expected_authorization.resolve():
        raise ValueError("scorer training-authorization path differs from protocol")
    if args.checkpoint_eligibility.resolve() != expected_eligibility.resolve():
        raise ValueError("scorer checkpoint-eligibility path differs from protocol")
    protocol.validate_training_authorization(
        project_root,
        expected_gate_spec=protocol.GATE_SPEC,
        authorization_path=args.training_authorization,
    )
    protocol.validate_checkpoint_eligibility(project_root, args.checkpoint_eligibility)

    causal_stage0_path = protocol.resolve_path(project_root, protocol.CAUSAL_STAGE0)
    causal_stage1_path = protocol.resolve_path(project_root, protocol.CAUSAL_STAGE1)
    specificity_stage0_path = protocol.resolve_path(project_root, protocol.SPECIFICITY_STAGE0)
    specificity_stage1_path = protocol.resolve_path(project_root, protocol.SPECIFICITY_STAGE1)
    causal_stage0 = protocol.validate_commitment_registry(
        causal_stage0_path, dataset="causal", stage=0
    )
    causal_stage1 = protocol.validate_commitment_registry(
        causal_stage1_path,
        dataset="causal",
        stage=1,
        expected_stage0_sha256=protocol.file_sha256(causal_stage0_path),
    )
    specificity_stage0 = protocol.validate_commitment_registry(
        specificity_stage0_path, dataset="specificity", stage=0
    )
    specificity_stage1 = protocol.validate_commitment_registry(
        specificity_stage1_path,
        dataset="specificity",
        stage=1,
        expected_stage0_sha256=protocol.file_sha256(specificity_stage0_path),
    )
    causal_opening_payload, causal_opened = protocol.validate_commitment_opening(
        project_root,
        args.causal_commitment_opening,
        dataset="causal",
        stage0_path=causal_stage0_path,
        stage1_path=causal_stage1_path,
        private_root=args.private_root,
    )
    specificity_opening_payload, specificity_opened = protocol.validate_commitment_opening(
        project_root,
        args.specificity_commitment_opening,
        dataset="specificity",
        stage0_path=specificity_stage0_path,
        stage1_path=specificity_stage1_path,
        private_root=args.private_root,
    )
    for path, commitment, label in (
        (
            args.causal_selected_manifest,
            causal_stage1["artifacts"]["selected_case_manifest_24"],
            "causal selected24",
        ),
        (
            args.causal_unit_manifest,
            causal_stage1["artifacts"]["unit_manifest_U_72"],
            "causal U72",
        ),
        (
            args.specificity_selected_manifest,
            specificity_stage1["artifacts"]["selected_case_manifest_18"],
            "specificity selected18",
        ),
        (
            args.specificity_unit_manifest,
            specificity_stage1["artifacts"]["unit_manifest_W_36"],
            "specificity W36",
        ),
        (
            args.holdout_mapping,
            specificity_stage1["artifacts"]["holdout_mapping_M_6"],
            "specificity holdout mapping M6",
        ),
    ):
        _require_private_committed_file(
            path,
            private_root=args.private_root,
            commitment=commitment,
            label=label,
        )
    expected_opened_paths = {
        args.causal_selected_manifest: causal_opened["selected_case_manifest_24"],
        args.causal_unit_manifest: causal_opened["unit_manifest_U_72"],
        args.specificity_selected_manifest: specificity_opened[
            "selected_case_manifest_18"
        ],
        args.specificity_unit_manifest: specificity_opened["unit_manifest_W_36"],
        args.holdout_mapping: specificity_opened["holdout_mapping_M_6"],
    }
    if any(
        supplied.resolve(strict=True) != opened.resolve(strict=True)
        for supplied, opened in expected_opened_paths.items()
    ):
        raise ValueError("scorer manifest path differs from the commitment opening")
    for path, label in (
        (args.causal_answer_key, "causal answer key"),
        (args.specificity_answer_key, "specificity answer key"),
        (args.causal_review_manifest, "causal review manifest"),
        (args.specificity_review_manifest, "specificity review manifest"),
        (args.causal_assignment_salt, "causal assignment salt"),
        (args.specificity_assignment_salt, "specificity assignment salt"),
    ):
        _require_private_file(path, private_root=args.private_root, label=label)

    causal_assignment_salt = args.causal_assignment_salt.read_text(
        encoding="utf-8"
    ).strip()
    specificity_assignment_salt = args.specificity_assignment_salt.read_text(
        encoding="utf-8"
    ).strip()
    if (
        not causal_assignment_salt
        or not specificity_assignment_salt
        or "\x00" in causal_assignment_salt
        or "\x00" in specificity_assignment_salt
    ):
        raise ValueError("assignment-salt openings must be nonempty NUL-free text")

    causal_selected = protocol.read_csv(args.causal_selected_manifest)
    causal_units = protocol.read_csv(args.causal_unit_manifest)
    specificity_selected = protocol.read_csv(args.specificity_selected_manifest)
    specificity_units = protocol.read_csv(args.specificity_unit_manifest)
    mapping = protocol.read_csv(args.holdout_mapping)
    protocol.validate_causal_selected_cases(causal_selected)
    protocol.validate_causal_unit_manifest(causal_units)
    _validate_selected_unit_binding(causal_selected, causal_units, dataset="causal")
    protocol.validate_specificity_selected_cases(
        specificity_selected, causal_cases=causal_selected
    )
    protocol.validate_specificity_unit_manifest(
        specificity_units,
        causal_cases=causal_selected,
        causal_seeds={int(row["seed"]) for row in causal_units},
    )
    _validate_selected_unit_binding(
        specificity_selected, specificity_units, dataset="specificity"
    )
    protocol.validate_holdout_mapping(
        mapping,
        causal_cases=causal_selected,
        specificity_cases=specificity_selected,
    )
    protocol.validate_selection_contract_opening(
        project_root,
        dataset="causal",
        stage0_registry=causal_stage0,
        private_root=args.private_root,
        candidate_manifest_path=causal_opened["candidate_manifest_48"],
        source_ontology_path=causal_opened["source_ontology_80"],
        source_split_path=causal_opened["source_split_80"],
        holdout_registry_path=causal_opened["holdout_registry_24"],
        receiver_ontology_path=causal_opened["receiver_ontology_32"],
        canonical_templates_path=causal_opened["canonical_templates"],
        field_rules_path=causal_opened["field_normalization"],
        render_configuration_path=causal_opened["raw_render_configuration"],
        selection_rules_path=causal_opened["ranking_formula"],
        secrets_path=causal_opened["stage0_secrets"],
        root_bundle_path=causal_opened["raw_root_bundle"],
        generation_spec_path=causal_opened["screening_generation_spec"],
        screening_seed_path=causal_opened["screening_seed"],
        selector_salt_path=causal_opened["selector_salt"],
        evaluation_seed_salt_path=causal_opened["evaluation_seed_salt"],
        forbidden_seed_inventory_path=causal_opened["forbidden_seed_inventory"],
        selection_binding_path=causal_opened["seed_derivation_formula"],
    )
    protocol.validate_selection_contract_opening(
        project_root,
        dataset="specificity",
        stage0_registry=specificity_stage0,
        private_root=args.private_root,
        candidate_manifest_path=specificity_opened["candidate_manifest_36"],
        new_bank_assignment_path=specificity_opened[
            "new_bank_selection_and_receiver_assignment"
        ],
        canonical_templates_path=specificity_opened["canonical_templates"],
        field_rules_path=specificity_opened["field_normalization"],
        render_configuration_path=specificity_opened["raw_render_configuration"],
        selection_rules_path=specificity_opened["ranking_formula"],
        secrets_path=specificity_opened["stage0_secrets"],
        root_bundle_path=specificity_opened["raw_root_bundle"],
        generation_spec_path=specificity_opened["screening_generation_spec"],
        screening_seed_path=specificity_opened["screening_seed"],
        selector_salt_path=specificity_opened["selector_salt"],
        evaluation_seed_salt_path=specificity_opened["evaluation_seed_salt"],
        forbidden_seed_inventory_path=specificity_opened["forbidden_seed_inventory"],
        selection_binding_path=specificity_opened["seed_derivation_formula"],
        causal_stage0_registry_path=causal_stage0_path,
        causal_stage1_registry_path=causal_stage1_path,
        causal_selected_path=args.causal_selected_manifest,
        causal_unit_manifest_path=args.causal_unit_manifest,
    )
    selector.revalidate_stage1_derivation(
        project_root,
        dataset="causal",
        private_root=args.private_root,
        opened_paths=causal_opened,
    )
    selector.revalidate_stage1_derivation(
        project_root,
        dataset="specificity",
        private_root=args.private_root,
        opened_paths=specificity_opened,
        causal_cases=causal_selected,
        causal_units=causal_units,
    )

    _, causal_paths = validate_review_freeze(
        project_root, args.causal_review_freeze, dataset="causal"
    )
    _, spec_paths = validate_review_freeze(
        project_root, args.specificity_review_freeze, dataset="specificity"
    )
    causal_package = validate_review_package(
        project_root,
        dataset="causal",
        unit_rows=causal_units,
        template_path=causal_paths["public_template"],
        answer_key_path=args.causal_answer_key,
        review_manifest_path=args.causal_review_manifest,
        package_commitment_path=causal_paths["review_package_commitment"],
        assignment_salt=causal_assignment_salt,
        unit_manifest_path=args.causal_unit_manifest,
        checkpoint_eligibility_path=args.checkpoint_eligibility,
    )
    specificity_package = validate_review_package(
        project_root,
        dataset="specificity",
        unit_rows=specificity_units,
        template_path=spec_paths["public_template"],
        answer_key_path=args.specificity_answer_key,
        review_manifest_path=args.specificity_review_manifest,
        package_commitment_path=spec_paths["review_package_commitment"],
        assignment_salt=specificity_assignment_salt,
        unit_manifest_path=args.specificity_unit_manifest,
        checkpoint_eligibility_path=args.checkpoint_eligibility,
    )
    if (
        causal_package["_validated_model_inventory_sha256"]
        != specificity_package["_validated_model_inventory_sha256"]
    ):
        raise ValueError("causal and specificity evaluations use different model inventories")
    if (
        causal_package["_validated_runtime_registry_sha256"]
        != specificity_package["_validated_runtime_registry_sha256"]
    ):
        raise ValueError("causal and specificity evaluations use different runtime registries")
    _validate_cross_dataset_video_isolation(causal_package, specificity_package)
    causal_key = protocol.read_csv(args.causal_answer_key)
    spec_key = protocol.read_csv(args.specificity_answer_key)
    causal_rows = _unblind("causal", _typed_canonical(protocol.read_csv(causal_paths["canonical_anonymous"]), "causal"), causal_key)
    spec_rows = _unblind("specificity", _typed_canonical(protocol.read_csv(spec_paths["canonical_anonymous"]), "specificity"), spec_key)
    _validate_scores_against_units(causal_rows, causal_units, dataset="causal")
    _validate_scores_against_units(spec_rows, specificity_units, dataset="specificity")
    provenance_checks = {
        "causal_stage1_binds_stage0": causal_stage1["stage0_registry_sha256"]
        == protocol.file_sha256(causal_stage0_path),
        "specificity_stage1_binds_stage0": specificity_stage1[
            "stage0_registry_sha256"
        ]
        == protocol.file_sha256(specificity_stage0_path),
        "opened_manifests_are_exact_paths": all(
            supplied.resolve(strict=True) == opened.resolve(strict=True)
            for supplied, opened in expected_opened_paths.items()
        ),
        "review_model_inventory_identical": causal_package[
            "_validated_model_inventory_sha256"
        ]
        == specificity_package["_validated_model_inventory_sha256"],
        "review_runtime_registry_identical": causal_package[
            "_validated_runtime_registry_sha256"
        ]
        == specificity_package["_validated_runtime_registry_sha256"],
        "causal_commitment_opening_bound": causal_opening_payload[
            "stage1_registry_sha256"
        ]
        == protocol.file_sha256(causal_stage1_path),
        "specificity_commitment_opening_bound": specificity_opening_payload[
            "stage1_registry_sha256"
        ]
        == protocol.file_sha256(specificity_stage1_path),
    }
    provenance_valid = all(provenance_checks.values())
    causal_gate = compute_causal_gate(causal_rows, provenance_valid=provenance_valid)
    specificity_gate = compute_specificity_gate(spec_rows)
    role_gate = compute_role_selectivity_gate(
        causal_rows,
        spec_rows,
        mapping,
        causal_gate=causal_gate,
        specificity_gate=specificity_gate,
    )
    outcome = classify_post_checkpoint_outcome(causal_gate, specificity_gate, role_gate)
    gate = {
        "protocol": protocol.PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "gate_spec": protocol.GATE_SPEC,
        "input_provenance": {
            "checks": provenance_checks,
            "training_authorization_sha256": protocol.file_sha256(args.training_authorization),
            "checkpoint_eligibility_sha256": protocol.file_sha256(args.checkpoint_eligibility),
            "causal_review_freeze_sha256": protocol.file_sha256(args.causal_review_freeze),
            "specificity_review_freeze_sha256": protocol.file_sha256(args.specificity_review_freeze),
            "causal_answer_key_sha256": protocol.file_sha256(args.causal_answer_key),
            "specificity_answer_key_sha256": protocol.file_sha256(args.specificity_answer_key),
            "causal_review_manifest_sha256": protocol.file_sha256(args.causal_review_manifest),
            "specificity_review_manifest_sha256": protocol.file_sha256(args.specificity_review_manifest),
            "causal_stage0_registry_sha256": protocol.file_sha256(causal_stage0_path),
            "causal_stage1_registry_sha256": protocol.file_sha256(causal_stage1_path),
            "specificity_stage0_registry_sha256": protocol.file_sha256(
                specificity_stage0_path
            ),
            "specificity_stage1_registry_sha256": protocol.file_sha256(
                specificity_stage1_path
            ),
            "causal_selected_manifest_sha256": protocol.file_sha256(
                args.causal_selected_manifest
            ),
            "causal_unit_manifest_sha256": protocol.file_sha256(
                args.causal_unit_manifest
            ),
            "specificity_selected_manifest_sha256": protocol.file_sha256(
                args.specificity_selected_manifest
            ),
            "specificity_unit_manifest_sha256": protocol.file_sha256(
                args.specificity_unit_manifest
            ),
            "holdout_mapping_sha256": protocol.file_sha256(args.holdout_mapping),
        },
        "causal": causal_gate,
        "specificity": specificity_gate,
        "role_selectivity": role_gate,
        "decision": outcome,
    }
    args.output_dir.mkdir(parents=True)
    protocol.write_csv(args.output_dir / "causal_unblinded_scores_v2.csv", causal_rows)
    protocol.write_csv(args.output_dir / "specificity_unblinded_scores_v2.csv", spec_rows)
    (args.output_dir / "gate_v2.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "outcome_v2.json").write_text(
        json.dumps(
            {
                "protocol": protocol.PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "terminal",
                **outcome,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 0


def _record_invalid_run(output_dir: Path, error: Exception) -> None:
    """Write a non-promotable terminal record without exposing private error text."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "outcome_v2.json"
    temporary = output_dir / f".outcome_v2.json.tmp.{os.getpid()}"
    if path.exists() or path.is_symlink() or temporary.exists():
        return
    payload = {
        "protocol": protocol.PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "terminal",
        "outcome": "invalid_run",
        "promote_v4": False,
        "sealed_final36_action": "remain_unopened; no scorer-side generation or access",
        "reason_code": "provenance_or_evaluation_validation_failed",
        "error_type": type(error).__name__,
    }
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cmd_score(args: argparse.Namespace) -> int:
    # Keep this guard outside the exception-to-invalid-record path: a rejected
    # sealed-final36 destination must never be created merely to report that it
    # was rejected.
    protocol.reject_sealed_final36_path(args.output_dir)
    output_preexisted = args.output_dir.exists()
    try:
        return _cmd_score_impl(args)
    except Exception as exc:
        if not output_preexisted:
            _record_invalid_run(args.output_dir, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-reviews", help="freeze anonymous scores before answer-key opening")
    freeze.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    freeze.add_argument("--review-template", type=Path, required=True)
    freeze.add_argument("--review-package-commitment", type=Path, required=True)
    freeze.add_argument("--reviewer-a", type=Path, required=True)
    freeze.add_argument("--reviewer-b", type=Path, required=True)
    freeze.add_argument("--dispute-template", type=Path, required=True)
    freeze.add_argument("--adjudication", type=Path, required=True)
    freeze.add_argument("--canonical-anonymous", type=Path, required=True)
    freeze.add_argument("--adjudication-audit", type=Path, required=True)
    freeze.add_argument("--freeze-manifest", type=Path, required=True)
    freeze.set_defaults(func=_cmd_freeze)
    gate = sub.add_parser("freeze-gate-registry", help="freeze executable machine gate before training")
    gate.add_argument("--output", type=Path, required=True)
    gate.set_defaults(func=_cmd_freeze_gate)
    score = sub.add_parser("score", help="validate frozen reviews, unblind, and apply all gates")
    score.add_argument("--training-authorization", type=Path, default=Path(protocol.TRAINING_AUTHORIZATION))
    score.add_argument("--checkpoint-eligibility", type=Path, default=Path(protocol.CHECKPOINT_ELIGIBILITY))
    score.add_argument("--private-root", type=Path, required=True)
    score.add_argument("--causal-commitment-opening", type=Path, required=True)
    score.add_argument("--specificity-commitment-opening", type=Path, required=True)
    score.add_argument("--causal-selected-manifest", type=Path, required=True)
    score.add_argument("--causal-unit-manifest", type=Path, required=True)
    score.add_argument("--specificity-selected-manifest", type=Path, required=True)
    score.add_argument("--specificity-unit-manifest", type=Path, required=True)
    score.add_argument("--causal-review-freeze", type=Path, required=True)
    score.add_argument("--specificity-review-freeze", type=Path, required=True)
    score.add_argument("--causal-answer-key", type=Path, required=True)
    score.add_argument("--specificity-answer-key", type=Path, required=True)
    score.add_argument("--causal-review-manifest", type=Path, required=True)
    score.add_argument("--specificity-review-manifest", type=Path, required=True)
    score.add_argument("--causal-assignment-salt", type=Path, required=True)
    score.add_argument("--specificity-assignment-salt", type=Path, required=True)
    score.add_argument("--holdout-mapping", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(func=_cmd_score)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
