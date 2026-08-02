#!/usr/bin/env python3
"""Select 16 balanced, non-test waterdrop candidates for training expansion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SELECTION = {
    "liquid_surface": [
        "a transparent measuring jug filled with water",
        "a stainless-steel saucepan filled with water",
        "an open rain barrel filled with water",
        "a round stone birdbath filled with water",
    ],
    "hard_surface": [
        "dark slate tile",
        "sealed hardwood tabletop",
        "clear acrylic sheet",
        "flat cast-iron griddle",
    ],
    "absorbent_surface": [
        "white paper towel",
        "unfinished pine board",
        "dense felt sheet fixed flat",
        "natural cellulose sponge",
    ],
    "granular_surface": [
        "loose dry garden soil",
        "dry beach sand",
        "yellow cornmeal",
        "fine sawdust",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/waterdrop_prompt_bank_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/waterdrop_generalization_expansion16.csv"))
    parser.add_argument("--prompt-prefix", type=Path, default=Path("prompts/waterdrop_generalization_expansion16"))
    args = parser.parse_args()

    with args.source.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    by_key = {(row["family"], row["receiver"], row["variant"]): row for row in source}

    records: list[dict[str, str]] = []
    for family, receivers in SELECTION.items():
        for receiver in receivers:
            row = by_key[(family, receiver, "0")]
            records.append(
                {
                    "expansion_id": f"wdgen{len(records):03d}",
                    "source_scene_id": row["scene_id"],
                    "family": row["family"],
                    "mechanism": row["mechanism"],
                    "receiver_id": row["receiver_id"],
                    "receiver": row["receiver"],
                    "footprint_family": row["causal_footprint"],
                    "fixed_seed": str(9800 + len(records)),
                    "prompt": row["prompt"],
                    "target_concept": "single falling water droplet",
                    "expected_effect": row["causal_footprint"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    for shard in range(2):
        selected = records[shard::2]
        path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Waterdrop generalization expansion16, balanced two-way shard.\n")
            for row in selected:
                handle.write(f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n")

    print(f"Wrote {len(records)} candidates to {args.output} and two balanced prompt shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
