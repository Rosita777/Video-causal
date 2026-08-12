#!/usr/bin/env python3
"""Build a balanced 12-prompt checkpoint-selection evaluation subset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SELECTED_INDICES = [0, 5, 10, 15, 24, 31, 40, 47, 48, 55, 64, 71]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/water_impact_dynamic_v1/test_pairs.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/water_impact_dynamic_v1/eval12.csv"))
    parser.add_argument("--output-prompts", type=Path, default=Path("prompts/water_impact_dynamic_v1/eval12.prompts"))
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [dict(rows[index], eval_index=str(position), source_test_index=str(index)) for position, index in enumerate(SELECTED_INDICES)]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["eval_index", "source_test_index", *rows[0].keys()]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    args.output_prompts.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prompts.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                f"{row['training_prompt']} | {row['source_object']} | "
                f"{row['expected_factual_event']}\n"
            )
    print(f"Wrote {len(selected)} evaluation prompts")
    print("seeds=" + ",".join(row["seed"] for row in selected))


if __name__ == "__main__":
    main()
