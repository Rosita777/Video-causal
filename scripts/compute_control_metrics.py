#!/usr/bin/env python3
"""Compute causal-footprint control-set metrics from annotated controls."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRIC_FIELDS = [
    "total_controls",
    "strict_valid_count",
    "borderline_valid_count",
    "invalid_count",
    "lenient_valid_count",
    "causal_specificity_pass_count",
    "causal_specificity_fail_count",
    "main_usable_count",
    "lenient_usable_count",
    "strict_valid_rate",
    "lenient_valid_rate",
    "causal_specificity_pass_rate",
    "main_usable_rate",
    "lenient_usable_rate",
]

TYPE_FIELDS = [
    "total_controls",
    "specificity_success_count",
    "specificity_failure_count",
    "strict_valid_count",
    "lenient_valid_count",
    "specificity_success_rate",
    "strict_valid_rate",
    "lenient_valid_rate",
]


def norm(value: str | None) -> str:
    return (value or "").strip().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / denominator:.4f}"


def label_yes_or_borderline(value: str) -> bool:
    return norm(value) in {"yes", "borderline"}


def specificity_pass(row: dict[str, str]) -> bool:
    control_type = norm(row.get("control_type"))
    target_absent = norm(row.get("target_visible")) == "no"
    effect_present = label_yes_or_borderline(row.get("effect_visible", ""))
    alternative_present = label_yes_or_borderline(row.get("alternative_cause_visible", ""))
    alternative_absent = norm(row.get("alternative_cause_visible")) == "no"

    if control_type == "no_cause":
        return target_absent and norm(row.get("effect_visible")) == "no" and alternative_absent
    if control_type == "effect_only":
        return target_absent and effect_present and alternative_absent
    if control_type == "alternative_cause":
        return target_absent and effect_present and alternative_present
    return False


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        output = dict(row)
        valid = norm(row.get("control_valid"))
        passed = specificity_pass(row)
        output["causal_specificity_pass"] = "yes" if passed else "no"
        output["strict_valid"] = "yes" if valid == "yes" else "no"
        output["lenient_valid"] = "yes" if valid in {"yes", "borderline"} else "no"
        output["main_usable"] = "yes" if valid == "yes" and passed else "no"
        output["lenient_usable"] = "yes" if valid in {"yes", "borderline"} and passed else "no"
        enriched.append(output)
    return enriched


def summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    strict_valid = sum(norm(row.get("control_valid")) == "yes" for row in rows)
    borderline_valid = sum(norm(row.get("control_valid")) == "borderline" for row in rows)
    invalid = sum(norm(row.get("control_valid")) == "no" for row in rows)
    lenient_valid = strict_valid + borderline_valid
    specificity = sum(norm(row.get("causal_specificity_pass")) == "yes" for row in rows)
    main_usable = sum(norm(row.get("main_usable")) == "yes" for row in rows)
    lenient_usable = sum(norm(row.get("lenient_usable")) == "yes" for row in rows)
    return {
        "total_controls": str(total),
        "strict_valid_count": str(strict_valid),
        "borderline_valid_count": str(borderline_valid),
        "invalid_count": str(invalid),
        "lenient_valid_count": str(lenient_valid),
        "causal_specificity_pass_count": str(specificity),
        "causal_specificity_fail_count": str(total - specificity),
        "main_usable_count": str(main_usable),
        "lenient_usable_count": str(lenient_usable),
        "strict_valid_rate": rate(strict_valid, total),
        "lenient_valid_rate": rate(lenient_valid, total),
        "causal_specificity_pass_rate": rate(specificity, total),
        "main_usable_rate": rate(main_usable, total),
        "lenient_usable_rate": rate(lenient_usable, total),
    }


def summarize_type(rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    success = sum(norm(row.get("causal_specificity_pass")) == "yes" for row in rows)
    strict_valid = sum(norm(row.get("control_valid")) == "yes" for row in rows)
    lenient_valid = sum(norm(row.get("control_valid")) in {"yes", "borderline"} for row in rows)
    return {
        "total_controls": str(total),
        "specificity_success_count": str(success),
        "specificity_failure_count": str(total - success),
        "strict_valid_count": str(strict_valid),
        "lenient_valid_count": str(lenient_valid),
        "specificity_success_rate": rate(success, total),
        "strict_valid_rate": rate(strict_valid, total),
        "lenient_valid_rate": rate(lenient_valid, total),
    }


def group_rows(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "")].append(row)
    return dict(sorted(grouped.items()))


def write_group_csv(
    path: Path,
    group_field: str,
    table: dict[str, dict[str, str]],
    metric_fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[group_field, *metric_fields], lineterminator="\n")
        writer.writeheader()
        for group, metrics in table.items():
            writer.writerow({group_field: group, **metrics})


def table_for_group(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    table = {"ALL": summarize(rows)}
    for group, group_rows_ in group_rows(rows, field).items():
        table[group] = summarize(group_rows_)
    return table


def type_table(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    table = {"ALL": summarize_type(rows)}
    for group, group_rows_ in group_rows(rows, "control_type").items():
        table[group] = summarize_type(group_rows_)
    return table


def write_enriched(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    rows: list[dict[str, str]],
    by_backbone: dict[str, dict[str, str]],
    by_type: dict[str, dict[str, str]],
    by_mechanism: dict[str, dict[str, str]],
) -> None:
    overall = by_backbone["ALL"]
    total = overall["total_controls"]
    lines = [
        "# Causal Footprint Controls v1 Metrics",
        "",
        f"- Total controls: {total}",
        f"- Strict usable controls: {overall['strict_valid_count']}/{total} ({overall['strict_valid_rate']})",
        f"- Lenient usable controls: {overall['lenient_valid_count']}/{total} ({overall['lenient_valid_rate']})",
        f"- Causal-specificity pass: {overall['causal_specificity_pass_count']}/{total} ({overall['causal_specificity_pass_rate']})",
        f"- Main usable controls (strict valid and specificity-pass): {overall['main_usable_count']}/{total} ({overall['main_usable_rate']})",
        f"- Lenient usable controls with specificity pass: {overall['lenient_usable_count']}/{total} ({overall['lenient_usable_rate']})",
        "",
        "## By Backbone",
        "",
        "| Backbone | Controls | Strict valid | Lenient valid | Specificity pass | Main usable | Lenient usable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backbone, metrics in by_backbone.items():
        lines.append(
            "| "
            f"{backbone} | {metrics['total_controls']} | {metrics['strict_valid_count']} | "
            f"{metrics['lenient_valid_count']} | {metrics['causal_specificity_pass_count']} | "
            f"{metrics['main_usable_count']} | {metrics['lenient_usable_count']} |"
        )

    lines.extend(
        [
            "",
            "## By Control Type",
            "",
            "| Control type | Controls | Specificity success | Specificity rate | Strict valid | Lenient valid |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for control_type, metrics in by_type.items():
        lines.append(
            "| "
            f"{control_type} | {metrics['total_controls']} | {metrics['specificity_success_count']} | "
            f"{metrics['specificity_success_rate']} | {metrics['strict_valid_count']} | "
            f"{metrics['lenient_valid_count']} |"
        )

    lines.extend(
        [
            "",
            "## By Mechanism",
            "",
            "| Mechanism | Controls | Strict valid | Lenient valid | Specificity pass | Main usable |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mechanism, metrics in by_mechanism.items():
        lines.append(
            "| "
            f"{mechanism} | {metrics['total_controls']} | {metrics['strict_valid_count']} | "
            f"{metrics['lenient_valid_count']} | {metrics['causal_specificity_pass_count']} | "
            f"{metrics['main_usable_count']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = read_csv(args.review)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    enriched = enrich_rows(rows)
    by_backbone = table_for_group(enriched, "backbone")
    by_mechanism = table_for_group(enriched, "mechanism_type")
    by_type = type_table(enriched)

    prefix = "causal_footprint_v0_controls_v1"
    write_enriched(args.output_dir / f"{prefix}_derived_rows.csv", enriched)
    write_group_csv(args.output_dir / f"{prefix}_metrics_by_backbone.csv", "backbone", by_backbone, METRIC_FIELDS)
    write_group_csv(args.output_dir / f"{prefix}_metrics_by_mechanism.csv", "mechanism_type", by_mechanism, METRIC_FIELDS)
    write_group_csv(args.output_dir / f"{prefix}_metrics_by_control_type.csv", "control_type", by_type, TYPE_FIELDS)
    write_summary(
        args.output_dir / f"{prefix}_metrics_summary.md",
        enriched,
        by_backbone,
        by_type,
        by_mechanism,
    )
    print(f"Wrote control metrics for {len(enriched)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
