#!/usr/bin/env python3
"""Build collision candidates only from receiver types Wan generated reliably."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FAMILIES = {
    "wood_block": [
        "three pale balsa-wood blocks", "three light pine blocks", "three tan cork blocks",
        "three white wooden blocks", "three blue foam blocks", "three yellow toy blocks",
        "three small cardboard blocks", "three light oak blocks", "three green foam blocks",
        "three beige cork blocks", "three blue wooden blocks", "three white toy blocks",
    ],
    "paper_cup": [
        "three white paper cups", "three blue paper cups", "three yellow paper cups",
        "three green paper cups", "three plain cardboard cups", "three small white cups",
        "three light blue paper cups", "three pale yellow paper cups", "three gray paper cups",
        "three orange paper cups", "three tan cardboard cups", "three purple paper cups",
    ],
    "short_tin": [
        "three short silver metal tins", "three short blue aluminum cans", "three short white tins",
        "three short gold metal tins", "three short green aluminum cans", "three short gray tins",
        "three small silver cans", "three small blue metal tins", "three small white cans",
        "three low gold tins", "three low green cans", "three low gray metal tins",
    ],
    "wood_peg": [
        "three tan wooden pegs", "three pale wooden pegs", "three blue toy pegs",
        "three yellow wooden pegs", "three white toy pegs", "three green wooden pegs",
        "three short tan wooden pins", "three short pale wooden pins", "three short blue toy pins",
        "three short yellow wooden pins", "three short white toy pins", "three short green wooden pins",
    ],
    "toy_pawn": [
        "three yellow toy pawns", "three blue toy pawns", "three white toy pawns",
        "three green toy pawns", "three orange toy pawns", "three gray toy pawns",
        "three small yellow game pieces", "three small blue game pieces", "three small white game pieces",
        "three small green game pieces", "three small orange game pieces", "three small gray game pieces",
    ],
    "wide_domino": [
        "three wide white domino blocks", "three wide blue domino blocks", "three wide yellow domino blocks",
        "three wide green domino blocks", "three wide tan wooden blocks", "three wide gray toy blocks",
        "three short white domino blocks", "three short blue domino blocks", "three short yellow domino blocks",
        "three short green domino blocks", "three short tan domino blocks", "three short gray domino blocks",
    ],
}


def prompt_for(receiver: str) -> str:
    return (
        "A simple realistic side-view tabletop video, fixed camera, one continuous shot. "
        f"Exactly {receiver} stand upright in one loose row with wide gaps. "
        "At first the three objects are completely still and no ball is visible. "
        "After two seconds, one small red rubber ball rolls slowly on the tabletop from left to right. "
        "The ball stays on the tabletop, strikes the leftmost object once, and that object falls onto its side. "
        "The other two objects remain still. Only one red ball exists in the entire video. "
        "No bouncing, flying, extra balls, hands, people, tools, cuts, or camera movement."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/collision_feasible72.csv"))
    parser.add_argument("--run-manifest", type=Path, default=Path("data/collision_feasible72_run_manifest.csv"))
    parser.add_argument("--prompt-prefix", type=Path, default=Path("prompts/collision_feasible72"))
    args = parser.parse_args()

    rows = []
    for family, receivers in FAMILIES.items():
        for receiver in receivers:
            index = len(rows)
            rows.append({
                "scene_id": f"collisionf{index:03d}",
                "family": family,
                "receiver": receiver,
                "fixed_seed": str(11000 + index),
                "prompt": prompt_for(receiver),
                "target_concept": "one small red rubber ball",
                "expected_effect": "the leftmost receiver falls only after ball contact",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for shard in range(2):
        selected = rows[shard::2]
        prompt_path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}.txt"
        seed_path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}_seeds.txt"
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write("# Collision prompts mined from Wan's feasible generation region.\n")
            for row in selected:
                handle.write(f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n")
        seed_path.write_text(",".join(row["fixed_seed"] for row in selected) + "\n", encoding="utf-8")

    with args.run_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["shard", "shard_index", "scene_id", "family", "receiver", "fixed_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "shard": index % 2,
                "shard_index": index // 2,
                **{key: row[key] for key in fields[2:]},
            })
    print(f"Wrote {len(rows)} feasibility-mined collision candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
