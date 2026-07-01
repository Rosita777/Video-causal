#!/usr/bin/env python3
"""Build a static annotation review page and queue CSV from an evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path


REVIEW_FIELDS = [
    "review_label",
    "review_target_visible",
    "review_effect_visible",
    "review_separation_clear",
    "review_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    for field in REVIEW_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in REVIEW_FIELDS:
                output.setdefault(field, "")
            writer.writerow(output)


def model_prefixes(fieldnames: list[str]) -> list[str]:
    prefixes = []
    for field in fieldnames:
        if field.endswith("_label") and field != "human_label" and not field.startswith("review_"):
            prefixes.append(field[: -len("_label")])
    return prefixes


def media_ref(path_text: str, *, project_root: Path, output_dir: Path) -> str:
    if not path_text:
        return ""
    source = Path(path_text)
    if not source.is_absolute():
        source = project_root / source
    try:
        return os.path.relpath(source.resolve(), output_dir.resolve())
    except OSError:
        return path_text


def image_cell(path_text: str, *, project_root: Path, output_dir: Path, label: str) -> str:
    if not path_text:
        return "<span class='muted'>missing</span>"
    ref = media_ref(path_text, project_root=project_root, output_dir=output_dir)
    escaped_ref = html.escape(ref)
    raw = html.escape(path_text)
    return (
        f"<a href='{escaped_ref}' target='_blank' data-source-path='{raw}'>"
        f"<img src='{escaped_ref}' alt='{html.escape(label)}'></a>"
    )


def render_model_badges(row: dict[str, str], prefixes: list[str]) -> str:
    badges = []
    for prefix in prefixes:
        label = row.get(f"{prefix}_label", "")
        if not label:
            continue
        disagrees = row.get(f"{prefix}_disagrees", "")
        css = "model disagree" if disagrees == "yes" else "model"
        suffix = " <b>Disagreement</b>" if disagrees == "yes" else ""
        reason = row.get(f"{prefix}_reason", "")
        badges.append(
            f"<div class='{css}'><span>{html.escape(prefix)}: {html.escape(label)}</span>{suffix}"
            f"<br><span class='muted'>{html.escape(reason)}</span></div>"
        )
    if not badges:
        return "<span class='muted'>no model predictions</span>"
    return "".join(badges)


def write_html(path: Path, rows: list[dict[str, str]], *, project_root: Path) -> None:
    output_dir = path.parent
    prefixes = model_prefixes(list(rows[0]) if rows else [])
    cards = []
    for idx, row in enumerate(rows):
        cards.append(
            "\n".join(
                [
                    "<section class='case'>",
                    f"<h2>{idx:03d} <code>{html.escape(row.get('sample_id', ''))}</code></h2>",
                    "<div class='meta'>",
                    f"<b>mechanism:</b> {html.escape(row.get('mechanism_id', ''))} "
                    f"({html.escape(row.get('mechanism_type', ''))})<br>",
                    f"<b>baseline:</b> {html.escape(row.get('baseline', ''))}<br>",
                    f"<b>target:</b> {html.escape(row.get('target_concept', ''))}<br>",
                    f"<b>effect:</b> {html.escape(row.get('causal_effect', ''))}<br>",
                    f"<b>prompt:</b> {html.escape(row.get('clean_prompt', ''))}",
                    "</div>",
                    "<div class='sheets'>",
                    "<div><h3>Clean reference</h3>"
                    + image_cell(row.get("reference_sheet_path", ""), project_root=project_root, output_dir=output_dir, label="reference")
                    + "</div>",
                    "<div><h3>Erased output</h3>"
                    + image_cell(row.get("contact_sheet_path", ""), project_root=project_root, output_dir=output_dir, label="output")
                    + "</div>",
                    "</div>",
                    "<div class='labels'>",
                    f"<b>human:</b> {html.escape(row.get('human_label', ''))} "
                    f"(target={html.escape(row.get('human_target_visible', ''))}, "
                    f"effect={html.escape(row.get('human_effect_visible', ''))}, "
                    f"separation={html.escape(row.get('human_separation_clear', ''))}, "
                    f"quality={html.escape(row.get('human_video_quality', ''))})",
                    f"<p>{html.escape(row.get('human_notes', ''))}</p>",
                    render_model_badges(row, prefixes),
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
                "<title>Causal Footprint Annotation Review</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:18px;color:#222;background:#f7f7f7}",
                "h1{margin:0 0 4px}.muted{color:#666}.case{background:#fff;border:1px solid #ddd;margin:14px 0;padding:12px}",
                ".meta{font-size:13px;line-height:1.45}.sheets{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}",
                "img{max-width:100%;border:1px solid #ccc;background:#eee}h2{font-size:16px}h3{font-size:13px;margin:4px 0}",
                ".labels{font-size:13px;margin-top:10px}.model{border-left:4px solid #9bbcff;background:#eef4ff;margin:6px 0;padding:6px}",
                ".disagree{border-left-color:#c43;background:#fff0ed}code{white-space:normal}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Causal Footprint Annotation Review</h1>",
                "<p class='muted'>Static review page for human audit. Use annotation_queue.csv for manual label updates.</p>",
                *cards,
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_queue(args.output_dir / "annotation_queue.csv", rows)
    write_html(args.output_dir / "review.html", rows, project_root=args.project_root)
    print(f"Wrote annotation review for {len(rows)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
