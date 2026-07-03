#!/usr/bin/env python3
"""Score C0.1 factorial-gate human review labels."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence


THRESHOLDS = {
    "original": 4,
    "remove_target": 4,
    "target_only": 4,
    "footprint_only": 3,
}

EXPECTED_VARIANTS = ["original", "remove_target", "footprint_only", "target_only"]

CELL_GATE_FIELDS = [
    "review_id",
    "pair_id",
    "item_index",
    "seed_index",
    "variant",
    "expected_target_visible",
    "expected_footprint_visible",
    "observed_target_visible",
    "observed_footprint_visible",
    "target_match",
    "footprint_match",
    "scene_structure_preserved",
    "cells_distinguishable",
    "generation_failure",
    "mode_collapse",
    "cell_success",
    "rejection_reasons",
    "notes",
]

ITEM_GATE_FIELDS = [
    "pair_id",
    "item_index",
    "total_cells",
    "missing_variants",
    "original_total",
    "original_successes",
    "original_threshold",
    "remove_target_total",
    "remove_target_successes",
    "remove_target_threshold",
    "footprint_only_total",
    "footprint_only_successes",
    "footprint_only_threshold",
    "target_only_total",
    "target_only_successes",
    "target_only_threshold",
    "gate_status",
    "rejection_reasons",
]

FALSE_FLAGS = {"no", "false", "0"}
TRUE_FLAGS = {"yes", "true", "1"}


def normalize_presence(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "yes": "present",
        "true": "present",
        "1": "present",
        "strong": "present",
        "present": "present",
        "no": "absent",
        "false": "absent",
        "0": "absent",
        "absent": "absent",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "": "uncertain",
    }
    if value not in aliases:
        raise ValueError(f"unknown presence label: {value}")
    return aliases[value]


def expected_to_presence(value: str) -> str:
    return "present" if str(value).strip().lower() == "yes" else "absent"


def is_false_flag(value: str) -> bool:
    return str(value).strip().lower() in FALSE_FLAGS


def is_true_flag(value: str) -> bool:
    return str(value).strip().lower() in TRUE_FLAGS


def cell_success(review: dict[str, str], key: dict[str, str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    target = normalize_presence(review.get("target_visible", ""))
    footprint = normalize_presence(review.get("footprint_visible", ""))
    if target == "uncertain" or footprint == "uncertain":
        reasons.append("review_uncertain")
    if is_false_flag(review.get("scene_structure_preserved", "yes")):
        reasons.append("scene_drift")
    if is_false_flag(review.get("cells_distinguishable", "yes")):
        reasons.append("cells_indistinguishable")
    if is_true_flag(review.get("generation_failure", "no")):
        reasons.append("generation_failure")
    if is_true_flag(review.get("mode_collapse", "no")):
        reasons.append("mode_collapse")
    if target != "uncertain" and target != expected_to_presence(key["expected_target_visible"]):
        if key["variant"] == "original":
            reasons.append("original_unreliable")
        elif key["variant"] == "remove_target":
            reasons.append("remove_target_failed")
        else:
            reasons.append(f"{key['variant']}_target_mismatch")
    if footprint != "uncertain" and footprint != expected_to_presence(
        key["expected_footprint_visible"]
    ):
        if key["variant"] == "target_only":
            reasons.append("target_only_preserves_footprint")
        elif key["variant"] == "footprint_only":
            reasons.append("footprint_only_incoherent")
        elif key["variant"] == "original":
            reasons.append("original_unreliable")
        elif key["variant"] == "remove_target":
            reasons.append("remove_target_failed")
    return not reasons, sorted(set(reasons))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_to_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def csv_row(row: dict[str, Any], fields: Sequence[str]) -> dict[str, str]:
    return {field: bool_to_text(row.get(field, "")) for field in fields}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row, fields))


def key_by_review_id(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        review_id = str(row.get("review_id", "")).strip()
        if not review_id:
            raise ValueError("answer key row missing review_id")
        if review_id in by_id:
            raise ValueError(f"duplicate answer key review_id: {review_id}")
        by_id[review_id] = row
    return by_id


def score_cell_rows(
    review_rows: Sequence[dict[str, str]],
    key_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    keys = key_by_review_id(key_rows)
    scored: list[dict[str, Any]] = []
    for review in review_rows:
        review_id = str(review.get("review_id", "")).strip()
        if not review_id:
            raise ValueError("review row missing review_id")
        if review_id not in keys:
            raise ValueError(f"review_id not found in answer key: {review_id}")
        key = keys[review_id]
        variant = str(key.get("variant", "")).strip()
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"unknown variant in answer key: {variant}")
        success, reasons = cell_success(review, key)
        target = normalize_presence(review.get("target_visible", ""))
        footprint = normalize_presence(review.get("footprint_visible", ""))
        scored.append(
            {
                "review_id": review_id,
                "pair_id": str(key.get("pair_id", "")).strip(),
                "item_index": str(key.get("item_index", "")).strip(),
                "seed_index": str(key.get("seed_index", "")).strip(),
                "variant": variant,
                "expected_target_visible": expected_to_presence(
                    key.get("expected_target_visible", "")
                ),
                "expected_footprint_visible": expected_to_presence(
                    key.get("expected_footprint_visible", "")
                ),
                "observed_target_visible": target,
                "observed_footprint_visible": footprint,
                "target_match": target
                != "uncertain"
                and target == expected_to_presence(key.get("expected_target_visible", "")),
                "footprint_match": footprint
                != "uncertain"
                and footprint
                == expected_to_presence(key.get("expected_footprint_visible", "")),
                "scene_structure_preserved": str(
                    review.get("scene_structure_preserved", "")
                ),
                "cells_distinguishable": str(review.get("cells_distinguishable", "")),
                "generation_failure": str(review.get("generation_failure", "")),
                "mode_collapse": str(review.get("mode_collapse", "")),
                "cell_success": success,
                "rejection_reasons": ",".join(reasons),
                "notes": str(review.get("notes", "")),
            }
        )
    missing_review_ids = sorted(set(keys) - {str(row.get("review_id", "")).strip() for row in review_rows})
    if missing_review_ids:
        raise ValueError(
            "answer key review_id missing from review CSV: " + ",".join(missing_review_ids)
        )
    return sorted(scored, key=cell_sort_key)


def item_key(row: dict[str, Any]) -> str:
    return str(row.get("pair_id", "")).strip() or str(row.get("item_index", "")).strip()


def cell_sort_key(row: dict[str, Any]) -> tuple[int, str, int, int, str]:
    item_index = str(row.get("item_index", "")).strip()
    seed_index = str(row.get("seed_index", "")).strip()
    variant = str(row.get("variant", ""))
    return (
        int(item_index) if item_index.isdigit() else 10**9,
        str(row.get("pair_id", "")),
        int(seed_index) if seed_index.isdigit() else 10**9,
        EXPECTED_VARIANTS.index(variant) if variant in EXPECTED_VARIANTS else 10**9,
        str(row.get("review_id", "")),
    )


def aggregate_item_scores(cell_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in sorted(cell_rows, key=cell_sort_key):
        groups.setdefault(item_key(row), []).append(row)

    item_scores: list[dict[str, Any]] = []
    for _key, rows in groups.items():
        first = rows[0]
        by_variant = {
            variant: [row for row in rows if row.get("variant") == variant]
            for variant in EXPECTED_VARIANTS
        }
        missing_variants = [
            variant for variant in EXPECTED_VARIANTS if not by_variant[variant]
        ]
        item_reasons = sorted(
            {
                reason
                for row in rows
                for reason in str(row.get("rejection_reasons", "")).split(",")
                if reason
            }
        )
        summary: dict[str, Any] = {
            "pair_id": first.get("pair_id", ""),
            "item_index": first.get("item_index", ""),
            "total_cells": len(rows),
            "missing_variants": ",".join(missing_variants),
        }
        threshold_failures: list[str] = []
        for variant in EXPECTED_VARIANTS:
            variant_rows = by_variant[variant]
            successes = sum(bool(row.get("cell_success")) for row in variant_rows)
            threshold = THRESHOLDS[variant]
            summary[f"{variant}_total"] = len(variant_rows)
            summary[f"{variant}_successes"] = successes
            summary[f"{variant}_threshold"] = threshold
            if successes < threshold:
                threshold_failures.append(f"{variant}_below_threshold")
        all_reasons = sorted(set(item_reasons + threshold_failures + missing_variants))
        summary["gate_status"] = "pass" if not all_reasons else "fail"
        summary["rejection_reasons"] = ",".join(all_reasons)
        item_scores.append(summary)
    return sorted(item_scores, key=item_sort_key)


def item_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    index = str(row.get("item_index", "")).strip()
    if index.isdigit():
        return int(index), str(row.get("pair_id", ""))
    return 10**9, str(row.get("pair_id", ""))


def write_outputs(
    output_dir: Path,
    cell_rows: Sequence[dict[str, Any]],
    item_rows: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cell_gate_summary.csv", cell_rows, CELL_GATE_FIELDS)
    write_csv(output_dir / "item_gate_summary.csv", item_rows, ITEM_GATE_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.review_csv.exists():
        parser.exit(2, f"{args.review_csv}: file not found\n")
    if not args.answer_key.exists():
        parser.exit(2, f"{args.answer_key}: file not found\n")
    review_rows = read_csv(args.review_csv)
    key_rows = read_csv(args.answer_key)
    cell_rows = score_cell_rows(review_rows, key_rows)
    item_rows = aggregate_item_scores(cell_rows)
    write_outputs(args.output_dir, cell_rows, item_rows)
    passed = sum(1 for row in item_rows if row.get("gate_status") == "pass")
    print(
        f"Wrote {len(cell_rows)} cell rows and {len(item_rows)} item rows to {args.output_dir}"
    )
    print(f"C0.1 gate pass: {passed}/{len(item_rows)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
