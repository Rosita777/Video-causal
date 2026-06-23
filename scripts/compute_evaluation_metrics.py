#!/usr/bin/env python3
"""Compute causal-footprint evaluation metrics from a v1 manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LABELS = ["strict_leakage", "borderline", "target_leakage", "other_failure"]
METRIC_FIELDS = [
    "total_outputs",
    "strict_leakage_count",
    "borderline_count",
    "relaxed_leakage_count",
    "target_leakage_count",
    "other_failure_count",
    "strict_leakage_rate",
    "borderline_rate",
    "relaxed_leakage_rate",
    "target_leakage_rate",
    "other_failure_rate",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
    return reader.fieldnames, rows


def rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / denominator:.4f}"


def summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    counts = {label: sum(row.get("human_label", "") == label for row in rows) for label in LABELS}
    relaxed = counts["strict_leakage"] + counts["borderline"]
    return {
        "total_outputs": str(total),
        "strict_leakage_count": str(counts["strict_leakage"]),
        "borderline_count": str(counts["borderline"]),
        "relaxed_leakage_count": str(relaxed),
        "target_leakage_count": str(counts["target_leakage"]),
        "other_failure_count": str(counts["other_failure"]),
        "strict_leakage_rate": rate(counts["strict_leakage"], total),
        "borderline_rate": rate(counts["borderline"], total),
        "relaxed_leakage_rate": rate(relaxed, total),
        "target_leakage_rate": rate(counts["target_leakage"], total),
        "other_failure_rate": rate(counts["other_failure"], total),
    }


def grouped(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(field, "")].append(row)
    table = {"ALL": summarize(rows)}
    for name in sorted(groups):
        table[name] = summarize(groups[name])
    return table


def write_group_csv(path: Path, group_field: str, table: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[group_field, *METRIC_FIELDS], lineterminator="\n")
        writer.writeheader()
        for group, metrics in table.items():
            writer.writerow({group_field: group, **metrics})


def model_prefixes(fieldnames: list[str]) -> list[str]:
    return sorted(
        field[: -len("_label")]
        for field in fieldnames
        if field.endswith("_label") and field != "human_label" and not field.startswith("review_")
    )


def model_agreement_rows(fieldnames: list[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for model in model_prefixes(fieldnames):
        compared = [row for row in rows if row.get(f"{model}_label", "")]
        agreements = sum(row.get(f"{model}_label", "") == row.get("human_label", "") for row in compared)
        disagreements = len(compared) - agreements
        output.append(
            {
                "model": model,
                "compared_outputs": str(len(compared)),
                "agreement_count": str(agreements),
                "disagreement_count": str(disagreements),
                "agreement_rate": rate(agreements, len(compared)),
            }
        )
    return output


def write_model_agreement(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "compared_outputs", "agreement_count", "disagreement_count", "agreement_rate"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    total_rows: int,
    baseline_table: dict[str, dict[str, str]],
    agreement_rows: list[dict[str, str]],
) -> None:
    overall = baseline_table["ALL"]
    lines = [
        "# Causal Footprint Evaluation Metrics",
        "",
        f"- Total outputs: {total_rows}",
        f"- Strict leakage: {overall['strict_leakage_count']}/{total_rows} ({overall['strict_leakage_rate']})",
        f"- Borderline: {overall['borderline_count']}/{total_rows} ({overall['borderline_rate']})",
        f"- Relaxed leakage: {overall['relaxed_leakage_count']}/{total_rows} ({overall['relaxed_leakage_rate']})",
        f"- Target leakage: {overall['target_leakage_count']}/{total_rows} ({overall['target_leakage_rate']})",
        "",
        "## By Baseline",
        "",
        "| Baseline | Outputs | Strict | Borderline | Relaxed | Target leakage | Other failure | Relaxed rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline, metrics in baseline_table.items():
        lines.append(
            "| "
            f"{baseline} | {metrics['total_outputs']} | {metrics['strict_leakage_count']} | "
            f"{metrics['borderline_count']} | {metrics['relaxed_leakage_count']} | "
            f"{metrics['target_leakage_count']} | {metrics['other_failure_count']} | "
            f"{metrics['relaxed_leakage_rate']} |"
        )
    if agreement_rows:
        lines.extend(
            [
                "",
                "## Model Agreement",
                "",
                "| Model | Compared | Agree | Disagree | Agreement rate |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in agreement_rows:
            lines.append(
                "| "
                f"{row['model']} | {row['compared_outputs']} | {row['agreement_count']} | "
                f"{row['disagreement_count']} | {row['agreement_rate']} |"
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
        fieldnames, rows = read_csv(args.manifest)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    baseline_table = grouped(rows, "baseline")
    mechanism_table = grouped(rows, "mechanism_type")
    agreement = model_agreement_rows(fieldnames, rows)
    write_group_csv(args.output_dir / "metrics_by_baseline.csv", "baseline", baseline_table)
    write_group_csv(args.output_dir / "metrics_by_mechanism.csv", "mechanism_type", mechanism_table)
    write_model_agreement(args.output_dir / "model_agreement.csv", agreement)
    write_summary(args.output_dir / "metrics_summary.md", len(rows), baseline_table, agreement)
    print(f"Wrote evaluation metrics for {len(rows)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
