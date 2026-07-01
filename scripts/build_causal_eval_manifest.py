#!/usr/bin/env python3
"""Build a controls-gated causal-footprint evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if value in indexed:
            raise ValueError(f"duplicate {key}: {value}")
        indexed[value] = row
    return indexed


def group_by(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get(key, ""), []).append(row)
    return grouped


def control_gate_for(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "fail"
    return "pass" if all(row.get("control_valid", "") == "yes" for row in rows) else "fail"


def build_rows(
    items: list[dict[str, object]],
    controls: list[dict[str, str]],
    baseline_review: list[dict[str, str]],
) -> list[dict[str, str]]:
    controls_by_item = group_by(controls, "item_id")
    review_by_key = {f"{row.get('item_id', '')}::{row.get('baseline', '')}": row for row in baseline_review}
    rows: list[dict[str, str]] = []

    for item in items:
        item_id = str(item.get("item_id", ""))
        control_rows = controls_by_item.get(item_id, [])
        for output in item.get("baseline_outputs", []):
            baseline = str(output.get("baseline", ""))
            review = review_by_key.get(f"{item_id}::{baseline}", {})
            rows.append(
                {
                    "item_id": item_id,
                    "source_name": str(item.get("source_name", "")),
                    "pair_id": str(item.get("pair_id", "")),
                    "mechanism_type": str(item.get("mechanism_type", "")),
                    "baseline": baseline,
                    "video_path": str(output.get("video_path", "")),
                    "control_gate": control_gate_for(control_rows),
                    "target_visible": str(output.get("target_visible", "")),
                    "causal_effect_visible": str(output.get("causal_effect_visible", "")),
                    "causeless_effect": str(output.get("causeless_effect", "")),
                    "video_quality": str(output.get("video_quality", "")),
                    "usable_for_claim": str(output.get("usable_for_claim", "")),
                    "failure_mode": str(output.get("failure_mode", "")),
                    "vlm_label": str(review.get("vlm_label", "")),
                    "human_label": str(review.get("review_label", output.get("usable_for_claim", ""))),
                    "adjudicated_label": str(review.get("review_label", output.get("usable_for_claim", ""))),
                    "review_notes": str(review.get("review_notes", "")),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--baseline-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = build_rows(read_jsonl(args.items), read_csv(args.controls), read_csv(args.baseline_review))
        write_csv(args.output, rows)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {len(rows)} causal eval manifest rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
