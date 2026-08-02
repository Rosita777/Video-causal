#!/usr/bin/env python3
"""Build 72 improved collision candidates from the successful gate families."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RECEIVERS = {
    "block_stack": [
        "three light pine cubes stacked in a short tower", "four red foam blocks stacked loosely",
        "five blue wooden blocks standing in a close row", "three small cardboard cartons stacked two levels high",
        "six white plastic domino blocks standing in a straight row", "four tan cork cubes standing upright",
        "three hollow plastic bricks stacked on a tabletop", "five small balsa-wood blocks in a tight row",
        "four yellow foam cubes arranged as a low tower", "three lightweight toy bricks stacked vertically",
        "six short wooden domino blocks standing in a row", "four small paper boxes stacked loosely",
        "three red wooden cubes in a short tower", "five white foam blocks standing side by side",
        "four blue plastic bricks arranged as a small wall", "three light cork blocks stacked at the center",
        "six tan toy blocks standing in a straight line", "four hollow cardboard cubes in a low stack",
        "three green foam cubes stacked vertically", "five narrow wooden blocks standing upright",
        "four lightweight building bricks in a short tower", "three small white cartons stacked loosely",
        "six red domino blocks standing in a close row", "four plain wooden cubes forming a low wall",
    ],
    "container_row": [
        "three empty silver cans standing close together", "four lightweight white paper cups in a row",
        "three small clear plastic bottles standing upright", "five empty red aluminum cans in a straight row",
        "four short metal tins standing close together", "three lightweight blue plastic cups in a row",
        "five small cardboard cups standing upright", "four empty green drink cans in a close row",
        "three clear plastic cups standing side by side", "five short paper cups in a straight line",
        "four lightweight orange cans standing upright", "three small empty metal tins in a row",
        "five blue disposable cups standing close together", "four short plastic bottles standing upright",
        "three empty white cans in a straight row", "five lightweight cardboard tubes standing upright",
        "four small red paper cups in a close row", "three empty gold aluminum cans standing together",
        "five clear disposable cups in a straight row", "four small cylindrical tins standing upright",
        "three lightweight juice cans in a close row", "five short blue paper cups standing upright",
        "four clear empty bottles in a straight row", "three small white plastic cups standing together",
    ],
    "upright_pieces": [
        "five light wooden pegs standing in a straight row", "four blue plastic game pieces standing upright",
        "six short white toy pins standing close together", "five red wooden bowling pins in a row",
        "four small yellow game pawns standing upright", "six tan wooden dowels standing on end",
        "five lightweight white domino pieces in a row", "four short blue toy posts standing upright",
        "six small wooden cylinders standing in a line", "five red plastic pawns standing close together",
        "four light cork pegs standing on end", "six short yellow toy pins in a straight row",
        "five blue wooden posts standing upright", "four small white bowling pins in a row",
        "six lightweight game pieces standing close together", "five tan wooden pegs standing in a line",
        "four red plastic domino pieces standing upright", "six short blue wooden cylinders on end",
        "five yellow toy pawns standing close together", "four lightweight wooden pins in a row",
        "six white plastic posts standing upright", "five small red wooden pegs in a line",
        "four blue toy bowling pins standing together", "six light cork cylinders standing on end",
    ],
}


def prompt_for(receiver: str) -> str:
    return (
        "A realistic fixed-camera tabletop video in one continuous shot. "
        f"For the first two seconds, {receiver} remains completely still and the ball is not visible. "
        "Then exactly one small red rubber ball rolls along the tabletop from the left, stays on the table, "
        "and visibly contacts the nearest object. Only after that contact, the contacted object and nearby objects "
        "tip over, slide, or move in a short physical collision chain. Before contact no object moves. "
        "There is only one ball and no hands, people, tools, extra balls, bouncing, or flying objects. "
        "The camera, table, lighting, and background remain fixed throughout."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/collision_expansion72.csv"))
    parser.add_argument("--run-manifest", type=Path, default=Path("data/collision_expansion72_run_manifest.csv"))
    parser.add_argument("--prompt-prefix", type=Path, default=Path("prompts/collision_expansion72"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for family, receivers in RECEIVERS.items():
        for receiver in receivers:
            index = len(rows)
            rows.append(
                {
                    "scene_id": f"collisionx{index:03d}", "family": family, "receiver": receiver,
                    "fixed_seed": str(10000 + index), "prompt": prompt_for(receiver),
                    "target_concept": "exactly one small red rubber ball",
                    "expected_effect": "contact-triggered tipping sliding or movement of the contacted objects",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for shard in range(2):
        selected = rows[shard::2]
        prompt_path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}.txt"
        seed_path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}_seeds.txt"
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write("# Improved collision expansion72.\n")
            for row in selected:
                handle.write(f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n")
        seed_path.write_text(",".join(row["fixed_seed"] for row in selected) + "\n", encoding="utf-8")
    with args.run_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["shard", "shard_index", "scene_id", "family", "receiver", "fixed_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"shard": index % 2, "shard_index": index // 2, **{key: row[key] for key in fields[2:]}})
    print(f"Wrote {len(rows)} improved collision candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
