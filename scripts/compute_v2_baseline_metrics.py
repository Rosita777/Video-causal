#!/usr/bin/env python3
"""Compute v2 target/footprint erasure metrics from verified baseline labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "baseline",
    "mechanism_type",
    "target_visible",
    "footprint_visible",
    "footprint_match",
    "separation_clear",
    "video_quality",
    "final_label",
}

LABELS = [
    "strict_causal_footprint_leakage",
    "erased_clean",
    "target_leakage",
    "borderline",
    "other_failure",
]

METRIC_FIELDS = [
    "total_outputs",
    "target_erased_count",
    "target_erasure_rate",
    "strict_leakage_count",
    "strict_causal_footprint_leakage_rate",
    "footprint_visible_given_erased_count",
    "footprint_retention_given_erased",
    "strict_leakage_given_erased",
    "erased_clean_count",
    "erased_clean_rate",
    "target_leakage_count",
    "target_leakage_rate",
    "borderline_count",
    "borderline_rate",
    "other_failure_count",
    "other_failure_rate",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / denominator:.4f}"


def count_label(rows: list[dict[str, str]], label: str) -> int:
    return sum(row.get("final_label") == label for row in rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    target_erased = sum(row.get("target_visible") == "no" for row in rows)
    strict = count_label(rows, "strict_causal_footprint_leakage")
    erased_clean = count_label(rows, "erased_clean")
    target_leakage = count_label(rows, "target_leakage")
    borderline = count_label(rows, "borderline")
    other_failure = count_label(rows, "other_failure")
    footprint_visible_given_erased = sum(
        row.get("target_visible") == "no" and row.get("footprint_visible") == "yes" for row in rows
    )
    return {
        "total_outputs": str(total),
        "target_erased_count": str(target_erased),
        "target_erasure_rate": rate(target_erased, total),
        "strict_leakage_count": str(strict),
        "strict_causal_footprint_leakage_rate": rate(strict, total),
        "footprint_visible_given_erased_count": str(footprint_visible_given_erased),
        "footprint_retention_given_erased": rate(footprint_visible_given_erased, target_erased),
        "strict_leakage_given_erased": rate(strict, target_erased),
        "erased_clean_count": str(erased_clean),
        "erased_clean_rate": rate(erased_clean, total),
        "target_leakage_count": str(target_leakage),
        "target_leakage_rate": rate(target_leakage, total),
        "borderline_count": str(borderline),
        "borderline_rate": rate(borderline, total),
        "other_failure_count": str(other_failure),
        "other_failure_rate": rate(other_failure, total),
    }


def group_rows(rows: list[dict[str, str]], field: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "")].append(row)
    return dict(sorted(grouped.items()))


def table_for_group(rows: list[dict[str, str]], group_field: str) -> dict[str, dict[str, str]]:
    table = {"ALL": summarize(rows)}
    for group, group_rows_ in group_rows(rows, group_field).items():
        table[group] = summarize(group_rows_)
    return table


def write_group_csv(path: Path, group_field: str, table: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[group_field, *METRIC_FIELDS], lineterminator="\n")
        writer.writeheader()
        for group, metrics in table.items():
            writer.writerow({group_field: group, **metrics})


def write_summary(path: Path, by_baseline: dict[str, dict[str, str]]) -> None:
    overall = by_baseline["ALL"]
    total = overall["total_outputs"]
    lines = [
        "# V2 Baseline Metrics",
        "",
        "These metrics use human-verified v2 target/footprint labels.",
        "",
        f"- Total outputs: {total}",
        f"- Target erased: {overall['target_erased_count']}/{total} ({overall['target_erasure_rate']})",
        (
            "- Footprint retained given target erased: "
            f"{overall['footprint_visible_given_erased_count']}/{overall['target_erased_count']} "
            f"({overall['footprint_retention_given_erased']})"
        ),
        (
            "- Strict causal-footprint leakage: "
            f"{overall['strict_leakage_count']}/{total} "
            f"({overall['strict_causal_footprint_leakage_rate']})"
        ),
        f"- Strict leakage given target erased: {overall['strict_leakage_given_erased']}",
        f"- Erased clean: {overall['erased_clean_count']}/{total} ({overall['erased_clean_rate']})",
        f"- Target leakage: {overall['target_leakage_count']}/{total} ({overall['target_leakage_rate']})",
        f"- Borderline: {overall['borderline_count']}/{total} ({overall['borderline_rate']})",
        f"- Other failure: {overall['other_failure_count']}/{total} ({overall['other_failure_rate']})",
        "",
        "## By Baseline",
        "",
        "| Baseline | Outputs | Target erased | Strict | Erased clean | Target leakage | Borderline | Other failure | Strict rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, metrics in by_baseline.items():
        lines.append(
            "| "
            f"`{baseline}` | {metrics['total_outputs']} | {metrics['target_erased_count']} | "
            f"{metrics['strict_leakage_count']} | {metrics['erased_clean_count']} | "
            f"{metrics['target_leakage_count']} | {metrics['borderline_count']} | "
            f"{metrics['other_failure_count']} | {metrics['strict_causal_footprint_leakage_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = read_csv(args.labels)
        by_baseline = table_for_group(rows, "baseline")
        by_mechanism = table_for_group(rows, "mechanism_type")
        write_group_csv(args.output_dir / "v2_metrics_by_baseline.csv", "baseline", by_baseline)
        write_group_csv(args.output_dir / "v2_metrics_by_mechanism.csv", "mechanism_type", by_mechanism)
        write_summary(args.output_dir / "v2_metrics_summary.md", by_baseline)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote v2 baseline metrics for {len(rows)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
