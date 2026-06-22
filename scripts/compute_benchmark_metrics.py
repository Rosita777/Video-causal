#!/usr/bin/env python3
"""Compute causal-footprint benchmark metrics from unified benchmark items."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    "strict_leakage_given_target_erased",
]


def read_items(path: Path) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            items.append(json.loads(stripped))
    return items


def flatten_outputs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for output in item.get("baseline_outputs", []):
            row = {
                "item_id": item.get("item_id", ""),
                "source_name": item.get("source_name", ""),
                "pair_id": item.get("pair_id", ""),
                "mechanism_type": item.get("mechanism_type", ""),
                "target_concept": item.get("target_concept", ""),
                "expected_effect": item.get("expected_effect", ""),
            }
            row.update(output)
            rows.append(row)
    return rows


def is_strict_leakage(row: dict[str, Any]) -> bool:
    return row.get("usable_for_claim") == "yes"


def is_borderline(row: dict[str, Any]) -> bool:
    return row.get("usable_for_claim") == "borderline"


def is_target_erased(row: dict[str, Any]) -> bool:
    return row.get("target_visible") == "no"


def is_target_leakage(row: dict[str, Any]) -> bool:
    return row.get("failure_mode") == "target_leakage"


def rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / denominator:.4f}"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    total = len(rows)
    target_erased = sum(is_target_erased(row) for row in rows)
    strict = sum(is_strict_leakage(row) for row in rows)
    borderline = sum(is_borderline(row) for row in rows)
    strict_or_borderline = strict + borderline
    target_leakage = sum(is_target_leakage(row) for row in rows)
    other_failure = total - strict_or_borderline - target_leakage
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
        "strict_leakage_given_target_erased": rate(strict, target_erased),
    }


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(sorted(grouped.items()))


def write_metrics_csv(path: Path, group_field: str, rows_by_group: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[group_field, *METRIC_FIELDS])
        writer.writeheader()
        for group, metrics in rows_by_group.items():
            writer.writerow({group_field: group, **metrics})


def table_for_group(rows: list[dict[str, Any]], group_field: str) -> dict[str, dict[str, str]]:
    table = {"ALL": summarize_rows(rows)}
    for group, group_rows_ in group_rows(rows, group_field).items():
        table[group] = summarize_rows(group_rows_)
    return table


def write_summary(path: Path, rows: list[dict[str, Any]], baseline_table: dict[str, dict[str, str]]) -> None:
    overall = baseline_table["ALL"]
    total = overall["total_outputs"]
    strict = overall["strict_leakage_count"]
    borderline = overall["borderline_count"]
    strict_rate = overall["strict_leakage_rate"]
    conditional = overall["strict_leakage_given_target_erased"]
    lines = [
        "# Causal Footprint Benchmark Metrics",
        "",
        f"- Total erasure outputs: {total}",
        f"- Strict causal-footprint leakage: {strict}/{total} ({strict_rate})",
        f"- Borderline causal-footprint cases: {borderline}/{total} ({overall['borderline_rate']})",
        f"- Strict leakage given target erased: {conditional}",
        "",
        "## By Baseline",
        "",
        "| Baseline | Outputs | Strict | Borderline | Target leakage | Strict rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, metrics in baseline_table.items():
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
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    items = read_items(args.items)
    rows = flatten_outputs(items)
    baseline_table = table_for_group(rows, "baseline")
    mechanism_table = table_for_group(rows, "mechanism_type")

    write_metrics_csv(args.output_dir / "causal_footprint_v0_metrics_by_baseline.csv", "baseline", baseline_table)
    write_metrics_csv(args.output_dir / "causal_footprint_v0_metrics_by_mechanism.csv", "mechanism_type", mechanism_table)
    write_summary(args.output_dir / "causal_footprint_v0_metrics_summary.md", rows, baseline_table)
    print(f"Wrote metrics for {len(rows)} outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
