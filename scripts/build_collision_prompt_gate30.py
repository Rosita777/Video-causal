#!/usr/bin/env python3
"""Build a small, balanced prompt gate for a ball-collision mechanism."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RECEIVERS = {
    "block_stack": [
        "a short stack of three lightweight wooden blocks",
        "four colorful foam cubes standing in a small tower",
        "three plain cardboard boxes stacked on a table",
        "a row of five small wooden blocks",
        "a small tower of white plastic building bricks",
        "three yellow toy blocks arranged in a row",
        "a short stack of red and blue wooden cubes",
        "four small cork blocks standing upright",
        "a neat row of six white domino blocks",
        "three small stone-like toy blocks on a tabletop",
    ],
    "container_row": [
        "three empty aluminum cans standing in a row",
        "four clear plastic cups standing upright",
        "three small paper cups arranged in a row",
        "two empty glass bottles standing apart",
        "four blue plastic bottles standing upright",
        "three small metal tins on a wooden table",
        "a row of five white disposable cups",
        "three short cardboard tubes standing upright",
        "four empty orange juice cans in a row",
        "three small ceramic cups standing on a counter",
    ],
    "toy_row": [
        "five upright wooden toy pins in a row",
        "three black chess pieces standing on a board",
        "four small red toy cones standing upright",
        "a row of five white plastic dominoes",
        "three wooden bowling pins on a tabletop",
        "four small toy traffic cones in a line",
        "three upright blue game pieces",
        "a row of five short wooden pegs",
        "four small yellow toy pins standing apart",
        "three lightweight figurines standing on a table",
    ],
}


def make_prompt(receiver: str) -> str:
    return (
        "A realistic fixed-camera tabletop video in one continuous shot. "
        f"During the first two seconds, {receiver} remains completely still and upright. "
        "Then one small red rubber ball rolls in from the left, visibly contacts the nearest object, "
        "and only after contact the contacted objects tip over or move away in a short, clear collision chain. "
        "The ball, the contact, and the resulting movement remain visible. "
        "The camera, table, lighting, and background remain fixed throughout."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/collision_prompt_gate30.csv"))
    parser.add_argument("--prompt-prefix", type=Path, default=Path("prompts/collision_prompt_gate30"))
    parser.add_argument("--run-manifest", type=Path, default=Path("data/collision_prompt_gate30_run_manifest.csv"))
    args = parser.parse_args()

    records: list[dict[str, str]] = []
    for family, receivers in RECEIVERS.items():
        for receiver in receivers:
            index = len(records)
            records.append(
                {
                    "scene_id": f"collision{index:03d}",
                    "family": family,
                    "receiver": receiver,
                    "fixed_seed": str(9900 + index),
                    "prompt": make_prompt(receiver),
                    "target_concept": "one small red rubber ball",
                    "expected_effect": "contact-triggered tipping or movement of the contacted objects",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    for shard in range(2):
        path = args.prompt_prefix.parent / f"{args.prompt_prefix.name}_shard_{shard}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Collision mechanism prompt gate30.\n")
            for row in records[shard::2]:
                handle.write(f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n")

    with args.run_manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["shard", "shard_index", "scene_id", "family", "receiver", "fixed_seed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(records):
            writer.writerow(
                {
                    "shard": str(index % 2),
                    "shard_index": str(index // 2),
                    "scene_id": row["scene_id"],
                    "family": row["family"],
                    "receiver": row["receiver"],
                    "fixed_seed": row["fixed_seed"],
                }
            )
    print(f"Wrote {len(records)} collision prompt-gate candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
