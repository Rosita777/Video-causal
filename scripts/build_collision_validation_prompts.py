#!/usr/bin/env python3
"""Build the pipe-delimited Wan prompt file for strict collision validation cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path("data/collision_feasible72_semantic_review.csv"))
    parser.add_argument("--source", type=Path, default=Path("data/collision_feasible72.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/collision_validation7.prompts"))
    args = parser.parse_args()

    with args.review.open(newline="", encoding="utf-8") as handle:
        selected = {
            row["scene_id"]: row
            for row in csv.DictReader(handle)
            if row.get("human_decision") == "accept" and row.get("selection_split") == "validation"
        }
    with args.source.open(newline="", encoding="utf-8") as handle:
        source = {row["scene_id"]: row for row in csv.DictReader(handle)}

    missing = sorted(set(selected) - set(source))
    if missing:
        raise SystemExit(f"Missing source rows: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for scene_id in sorted(selected):
            row = source[scene_id]
            handle.write(
                f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n"
            )
    print(f"Wrote {len(selected)} validation prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
