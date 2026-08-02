#!/usr/bin/env python3
"""Select a balanced 20-case held-out causal evaluation from frozen test100."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


GROUPS = [
    ("explicit_causal", "liquid_surface"),
    ("explicit_causal", "hard_surface"),
    ("implicit_causal", "liquid_surface"),
    ("implicit_causal", "hard_surface"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/waterdrop_test100_final.csv"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("data/waterdrop_dual_traj_eval20.csv")
    )
    parser.add_argument(
        "--output-prompts", type=Path, default=Path("prompts/waterdrop_dual_traj_eval20.txt")
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[dict[str, str]] = []
    for condition, family in GROUPS:
        candidates = [
            row
            for row in rows
            if row["condition"] == condition
            and row["family"] == family
            and row["semantic_status"] == "pass"
        ]
        selected.extend(candidates[:5])

    if len(selected) != 20:
        raise RuntimeError(f"Expected 20 selected cases, found {len(selected)}")
    if {row["fixed_seed"] for row in selected} != {"9100"}:
        raise RuntimeError("Expanded eval20 expects every frozen case to use seed 9100")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["eval_index", *selected[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(selected):
            writer.writerow({"eval_index": f"{index:03d}", **row})

    args.output_prompts.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prompts.open("w", encoding="utf-8") as handle:
        handle.write("# Balanced held-out causal eval20: explicit/implicit x liquid/hard.\n\n")
        for row in selected:
            handle.write(f"{row['prompt']} | single water droplet | {row['scene_id']}\n")


if __name__ == "__main__":
    main()
