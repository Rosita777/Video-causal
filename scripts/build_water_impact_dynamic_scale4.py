#!/usr/bin/env python3
"""Build four representative prompts for the checkpoint-200 LoRA scale sweep."""

from __future__ import annotations

import csv
from pathlib import Path


INDICES = [0, 2, 4, 8]


def main() -> None:
    rows = list(csv.DictReader(Path("data/water_impact_dynamic_v1/eval12.csv").open()))
    selected = [rows[index] for index in INDICES]
    output = Path("prompts/water_impact_dynamic_v1/scale4.prompts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(
                f"{row['training_prompt']} | {row['source_object']} | "
                f"{row['expected_factual_event']}\n"
            )
    seeds = Path("data/water_impact_dynamic_v1/scale4_seeds.txt")
    seeds.write_text(",".join(row["seed"] for row in selected) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} prompts")


if __name__ == "__main__":
    main()
