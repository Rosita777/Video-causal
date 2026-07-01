#!/usr/bin/env python3
"""Compute causal-footprint metrics from an adjudicated evaluation manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRIC_FIELDS = [
    "total_outputs",
    "target_erased_count",
    "strict_leakage_count",
    "borderline_count",
    "strict_or_borderline_count",
    "target_leakage_count",
    "other_failure_count",
    "strict_leakage_rate",
    "borderline_rate",
    "strict_or_borderline_rate",
    "target_leakage_rate",
    "strict_leakage_given_target_erased",
]


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


def gated_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("control_gate", "pass") == "pass"]


def summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    target_erased = sum(row.get("target_visible", "") == "no" for row in rows)
    strict = sum(row.get("adjudicated_label", "") == "strict_leakage" for row in rows)
    borderline = sum(row.get("adjudicated_label", "") == "borderline" for row in rows)
    target_leakage = sum(row.get("adjudicated_label", "") == "target_leakage" for row in rows)
    other_failure = sum(row.get("adjudicated_label", "") == "other_failure" for row in rows)
    strict_or_borderline = strict + borderline
    return {
        "total_outputs": str(total),
        "target_erased_count": str(target_erased),
        "strict_leakage_count": str(strict),
        "borderline_count": str(borderline),
        "strict_or_borderline_count": str(strict_or_borderline),
        "target_leakage_count": str(target_leakage),
        "other_failure_count": str(other_failure),
        "strict_leakage_rate": rate(strict, total),
        "borderline_rate": rate(borderline, total),
        "strict_or_borderline_rate": rate(strict_or_borderline, total),
        "target_leakage_rate": rate(target_leakage, total),
        "strict_leakage_given_target_erased": rate(strict, target_erased),
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
        writer = csv.DictWriter(writerow_handle := handle, fieldnames=[group_field, *METRIC_FIELDS], lineterminator="\n")
        writer.writeheader()
        for group, metrics in table.items():
            writer.writerow({group_field: group, **metrics})
    del writerow_handle


def write_summary(path: Path, by_baseline: dict[str, dict[str, str]]) -> None:
    overall = by_baseline["ALL"]
    total = overall["total_outputs"]
    lines = [
        "# Causal Footprint Eval Pipeline Metrics",
        "",
        f"- Total gated outputs: {total}",
        f"- Strict leakage: {overall['strict_leakage_count']}/{total} ({overall['strict_leakage_rate']})",
        f"- Borderline: {overall['borderline_count']}/{total} ({overall['borderline_rate']})",
        f"- Strict or borderline: {overall['strict_or_borderline_count']}/{total} ({overall['strict_or_borderline_rate']})",
        f"- Target leakage: {overall['target_leakage_count']}/{total} ({overall['target_leakage_rate']})",
        f"- Strict leakage given target erased: {overall['strict_leakage_given_target_erased']}",
        "",
        "## By Baseline",
        "",
        "| Baseline | Outputs | Strict | Borderline | Target leakage | Strict rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, metrics in by_baseline.items():
        lines.append(
            "| "
            f"{baseline} | {metrics['total_outputs']} | {metrics['strict_leakage_count']} | "
            f"{metrics['borderline_count']} | {metrics['target_leakage_count']} | "
            f"{metrics['strict_leakage_rate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = gated_rows(read_csv(args.manifest))
        by_baseline = table_for_group(rows, "baseline")
        by_mechanism = table_for_group(rows, "mechanism_type")
        write_group_csv(args.output_dir / "causal_eval_metrics_by_baseline.csv", "baseline", by_baseline)
        write_group_csv(args.output_dir / "causal_eval_metrics_by_mechanism.csv", "mechanism_type", by_mechanism)
        write_summary(args.output_dir / "causal_eval_metrics_summary.md", by_baseline)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote causal eval metrics for {len(rows)} gated rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
