#!/usr/bin/env python3
"""Build paper-facing tables and example galleries from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter, defaultdict
from pathlib import Path


BASELINE_LABELS = {
    "negative_prompt": "Negative Prompt",
    "safree_cogvideox": "SAFREE-CogVideoX",
    "videoeraser": "VideoEraser local",
    "t2vunlearning": "T2V proxy",
}

MECHANISM_ORDER = [
    "fluid_impact",
    "surface_trace",
    "fracture_damage",
    "elastic_deformation",
    "field_mediated",
    "particle_dispersion",
]

SOURCE_PRIORITY = {
    "round6_yes23": 0,
    "round5_yes10": 1,
    "round4_valid9": 2,
    "valid5": 3,
}

EXAMPLE_FIELDS = [
    "rank",
    "output_id",
    "item_id",
    "source_name",
    "pair_id",
    "mechanism_type",
    "baseline",
    "target_concept",
    "expected_effect",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "failure_mode",
    "notes",
    "video_path",
    "reference_sheet_path",
    "reference_sheet_exists",
    "sheet_path",
    "sheet_exists",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(row: dict[str, str], field: str) -> str:
    return row.get(field, "0.0000")


def int_text(row: dict[str, str], field: str) -> str:
    return row.get(field, "0")


def table_rows_without_all(rows: list[dict[str, str]], group_field: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get(group_field) != "ALL"]


def markdown_metric_table(rows: list[dict[str, str]], group_field: str, title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"| {group_field} | Outputs | Strict | Borderline | Target leakage | Strict rate | Strict given erased |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows_without_all(rows, group_field):
        group = row.get(group_field, "")
        if group_field == "baseline":
            group = BASELINE_LABELS.get(group, group)
        lines.append(
            "| "
            f"{group} | {int_text(row, 'total_outputs')} | {int_text(row, 'strict_leakage_count')} | "
            f"{int_text(row, 'borderline_count')} | {int_text(row, 'target_leakage_count')} | "
            f"{pct(row, 'strict_leakage_rate')} | {pct(row, 'strict_leakage_given_target_erased')} |"
        )
    return lines


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_latex_table(path: Path, rows: list[dict[str, str]], group_field: str, label: str) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        f"{label} & Outputs & Strict & Borderline & Strict Rate \\\\",
        "\\midrule",
    ]
    for row in table_rows_without_all(rows, group_field):
        group = row.get(group_field, "")
        if group_field == "baseline":
            group = BASELINE_LABELS.get(group, group)
        lines.append(
            f"{latex_escape(group)} & {int_text(row, 'total_outputs')} & "
            f"{int_text(row, 'strict_leakage_count')} & {int_text(row, 'borderline_count')} & "
            f"{pct(row, 'strict_leakage_rate')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def score_example(row: dict[str, str]) -> tuple[int, int, int, int, str, str]:
    source_rank = SOURCE_PRIORITY.get(row.get("source_name", ""), 99)
    quality_rank = 0 if row.get("video_quality") == "good" else 1
    target_rank = 0 if row.get("target_visible") == "no" else 1
    causeless_rank = 0 if row.get("causeless_effect") == "yes" else 1
    return (source_rank, quality_rank, target_rank, causeless_rank, row.get("baseline", ""), row.get("output_id", ""))


def select_examples(rows: list[dict[str, str]], max_examples: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("human_label") == "strict_leakage"
        and row.get("sheet_exists") == "true"
        and row.get("reference_sheet_exists") == "true"
    ]
    candidates.sort(key=lambda row: (MECHANISM_ORDER.index(row["mechanism_type"]) if row.get("mechanism_type") in MECHANISM_ORDER else 99, score_example(row)))

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    for mechanism in MECHANISM_ORDER:
        if len(selected) >= max_examples:
            break
        mechanism_candidates = [row for row in candidates if row.get("mechanism_type") == mechanism and row.get("output_id") not in selected_ids]
        if mechanism_candidates:
            row = mechanism_candidates[0]
            selected.append(row)
            selected_ids.add(row["output_id"])

    for row in sorted(candidates, key=score_example):
        if len(selected) >= max_examples:
            break
        if row.get("output_id") not in selected_ids:
            selected.append(row)
            selected_ids.add(row["output_id"])
    return selected


def enriched_gold_rows(gold_rows: list[dict[str, str]], vlm_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    vlm_by_id = {row.get("output_id", ""): row for row in vlm_rows}
    enriched = []
    for row in gold_rows:
        merged = dict(row)
        vlm = vlm_by_id.get(row.get("output_id", ""), {})
        for field in ["reference_sheet_path", "reference_sheet_exists", "sheet_path", "sheet_exists"]:
            merged[field] = vlm.get(field, "")
        enriched.append(merged)
    return enriched


def write_selected_examples_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output_rows = []
    for index, row in enumerate(rows, start=1):
        output = {field: row.get(field, "") for field in EXAMPLE_FIELDS}
        output["rank"] = str(index)
        output_rows.append(output)
    write_csv(path, EXAMPLE_FIELDS, output_rows)


def rel_ref(path_text: str, output_dir: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return path_text


def write_examples_html(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for index, row in enumerate(rows, start=1):
        ref = html.escape(rel_ref(row.get("reference_sheet_path", ""), path.parent))
        sheet = html.escape(rel_ref(row.get("sheet_path", ""), path.parent))
        baseline = BASELINE_LABELS.get(row.get("baseline", ""), row.get("baseline", ""))
        cards.append(
            "\n".join(
                [
                    "<section class='case'>",
                    f"<h2>{index}. {html.escape(row.get('pair_id', ''))} <span>{html.escape(row.get('mechanism_type', ''))}</span></h2>",
                    "<div class='meta'>",
                    f"<b>Baseline:</b> {html.escape(baseline)}<br>",
                    f"<b>Target:</b> {html.escape(row.get('target_concept', ''))}<br>",
                    f"<b>Footprint:</b> {html.escape(row.get('expected_effect', ''))}<br>",
                    f"<b>Reason:</b> {html.escape(row.get('notes', ''))}",
                    "</div>",
                    "<div class='sheets'>",
                    f"<div><h3>Clean reference</h3><img src='{ref}' alt='clean reference'></div>",
                    f"<div><h3>Erased output</h3><img src='{sheet}' alt='erased output'></div>",
                    "</div>",
                    "</section>",
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>Causal Footprint Paper Examples</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:18px;background:#f7f7f7;color:#222}",
                ".case{background:white;border:1px solid #ddd;margin:14px 0;padding:12px}",
                "h1{margin-bottom:4px}h2{font-size:18px}h2 span{color:#666;font-size:13px;font-weight:400}",
                ".meta{font-size:13px;line-height:1.45}.sheets{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}",
                "img{max-width:100%;border:1px solid #ccc;background:#eee}h3{font-size:13px;margin:4px 0}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Causal Footprint Paper Examples</h1>",
                "<p>Strict leakage examples selected for mechanism diversity and clean-reference availability.</p>",
                *cards,
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def failure_taxonomy(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if row.get("human_label") == "strict_leakage":
            continue
        counts[(row.get("human_label", ""), row.get("failure_mode", ""), row.get("baseline", ""))] += 1
    return [
        {
            "human_label": human_label,
            "failure_mode": failure_mode,
            "baseline": baseline,
            "count": str(count),
        }
        for (human_label, failure_mode, baseline), count in sorted(counts.items())
    ]


def write_failure_summary(path: Path, taxonomy_rows: list[dict[str, str]]) -> None:
    by_label: defaultdict[str, int] = defaultdict(int)
    by_mode: defaultdict[str, int] = defaultdict(int)
    for row in taxonomy_rows:
        count = int(row["count"])
        by_label[row["human_label"]] += count
        by_mode[row["failure_mode"]] += count
    lines = ["# Failure Taxonomy", ""]
    lines.append("## By Human Label")
    lines.append("")
    for label, count in sorted(by_label.items()):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## By Failure Mode", ""])
    for mode, count in sorted(by_mode.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {mode}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, baseline_rows: list[dict[str, str]], mechanism_rows: list[dict[str, str]]) -> None:
    overall = next((row for row in baseline_rows if row.get("baseline") == "ALL"), {})
    total = int_text(overall, "total_outputs")
    strict = int_text(overall, "strict_leakage_count")
    borderline = int_text(overall, "borderline_count")
    target_erased = int_text(overall, "target_erased_count")
    lines = [
        "# Causal Footprint Benchmark Paper Assets",
        "",
        f"- Total erasure outputs: {total}",
        f"- Strict causal-footprint leakage: {strict}/{total} ({pct(overall, 'strict_leakage_rate')})",
        f"- Borderline causal-footprint cases: {borderline}/{total} ({pct(overall, 'borderline_rate')})",
        f"- Strict leakage given target erased: {strict}/{target_erased} ({pct(overall, 'strict_leakage_given_target_erased')})",
        "",
        *markdown_metric_table(baseline_rows, "baseline", "By Baseline"),
        "",
        *markdown_metric_table(mechanism_rows, "mechanism_type", "By Mechanism"),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_paper_metric_csv(path: Path, rows: list[dict[str, str]], group_field: str) -> None:
    fields = [
        group_field,
        "total_outputs",
        "strict_leakage_count",
        "borderline_count",
        "strict_or_borderline_count",
        "target_leakage_count",
        "strict_leakage_rate",
        "strict_leakage_given_target_erased",
    ]
    write_csv(path, fields, [{field: row.get(field, "") for field in fields} for row in rows])


def build_assets(args: argparse.Namespace) -> None:
    baseline_rows = read_csv(args.metrics_by_baseline)
    mechanism_rows = read_csv(args.metrics_by_mechanism)
    gold_rows = read_csv(args.gold)
    vlm_rows = read_csv(args.vlm_inputs)
    enriched = enriched_gold_rows(gold_rows, vlm_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "benchmark_results_summary.md", baseline_rows, mechanism_rows)
    copy_paper_metric_csv(args.output_dir / "paper_table_by_baseline.csv", baseline_rows, "baseline")
    copy_paper_metric_csv(args.output_dir / "paper_table_by_mechanism.csv", mechanism_rows, "mechanism_type")
    write_latex_table(args.output_dir / "paper_table_by_baseline.tex", baseline_rows, "baseline", "Baseline")
    write_latex_table(args.output_dir / "paper_table_by_mechanism.tex", mechanism_rows, "mechanism_type", "Mechanism")

    taxonomy = failure_taxonomy(enriched)
    write_csv(args.output_dir / "failure_taxonomy.csv", ["human_label", "failure_mode", "baseline", "count"], taxonomy)
    write_failure_summary(args.output_dir / "failure_taxonomy_summary.md", taxonomy)

    selected = select_examples(enriched, args.max_examples)
    write_selected_examples_csv(args.output_dir / "selected_strict_examples.csv", selected)
    write_examples_html(args.output_dir / "selected_strict_examples.html", selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--vlm-inputs", type=Path, required=True)
    parser.add_argument("--metrics-by-baseline", type=Path, required=True)
    parser.add_argument("--metrics-by-mechanism", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        build_assets(args)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n")
    print(f"Wrote paper assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
