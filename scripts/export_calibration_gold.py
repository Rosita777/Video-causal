#!/usr/bin/env python3
"""Export human-labeled benchmark outputs as evaluator calibration gold CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GOLD_FIELDS = [
    "output_id",
    "item_id",
    "source_name",
    "pair_id",
    "mechanism_type",
    "baseline",
    "video_path",
    "seed",
    "target_concept",
    "expected_effect",
    "source_prompt",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "usable_for_claim",
    "failure_mode",
    "human_label",
    "notes",
]


def read_items(path: Path) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            items.append(json.loads(stripped))
    return items


def derive_human_label(output: dict[str, Any]) -> str:
    if output.get("usable_for_claim") == "yes":
        return "strict_leakage"
    if output.get("usable_for_claim") == "borderline":
        return "borderline"
    if output.get("failure_mode") == "target_leakage":
        return "target_leakage"
    return "other_failure"


def flatten_gold_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        item_id = str(item.get("item_id", ""))
        for output in item.get("baseline_outputs", []):
            baseline = str(output.get("baseline", ""))
            rows.append(
                {
                    "output_id": f"{item_id}::{baseline}",
                    "item_id": item_id,
                    "source_name": str(item.get("source_name", "")),
                    "pair_id": str(item.get("pair_id", "")),
                    "mechanism_type": str(item.get("mechanism_type", "")),
                    "baseline": baseline,
                    "video_path": str(output.get("video_path", "")),
                    "seed": "" if output.get("seed") is None else str(output.get("seed")),
                    "target_concept": str(item.get("target_concept", "")),
                    "expected_effect": str(item.get("expected_effect", "")),
                    "source_prompt": str(item.get("source_prompt", "")),
                    "target_visible": str(output.get("target_visible", "")),
                    "causal_effect_visible": str(output.get("causal_effect_visible", "")),
                    "causeless_effect": str(output.get("causeless_effect", "")),
                    "video_quality": str(output.get("video_quality", "")),
                    "usable_for_claim": str(output.get("usable_for_claim", "")),
                    "failure_mode": str(output.get("failure_mode", "")),
                    "human_label": derive_human_label(output),
                    "notes": str(output.get("notes", "")),
                }
            )
    return rows


def write_gold_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = flatten_gold_rows(read_items(args.items))
    write_gold_csv(rows, args.output)
    print(f"Wrote {len(rows)} gold rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
