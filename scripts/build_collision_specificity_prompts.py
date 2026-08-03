#!/usr/bin/env python3
"""Build prompts for testing whether the collision adapter is specific to its target."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STATIC_CONTROLS = [
    "A realistic fixed-camera tabletop video in one continuous shot. Three blue plastic cups stand upright in a row on a wooden table and remain completely still for the entire video. No ball, hands, people, tools, or other moving objects are present. The camera, lighting, table, and background remain unchanged. | one small red rubber ball | no target motion",
    "A realistic fixed-camera tabletop video in one continuous shot. Four silver metal cans stand upright with clear gaps on a white tabletop and remain completely still for the entire video. No ball, hands, people, tools, or other moving objects are present. The camera, lighting, table, and background remain unchanged. | one small red rubber ball | no target motion",
    "A realistic fixed-camera tabletop video in one continuous shot. Three green wooden pegs stand upright on a pale table and remain completely still for the entire video. No ball, hands, people, tools, or other moving objects are present. The camera, lighting, table, and background remain unchanged. | one small red rubber ball | no target motion",
    "A realistic fixed-camera tabletop video in one continuous shot. Three white chess pawns stand on a dark tabletop and remain completely still for the entire video. No ball, hands, people, tools, or other moving objects are present. The camera, lighting, table, and background remain unchanged. | one small red rubber ball | no target motion",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterdrop", type=Path, default=Path("data/waterdrop_dual_traj_eval20.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/collision_specificity8.prompts"))
    args = parser.parse_args()

    with args.waterdrop.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:4]
    if len(rows) != 4:
        raise SystemExit("Expected at least four waterdrop evaluation rows")

    lines = list(STATIC_CONTROLS)
    for row in rows:
        lines.append(
            f"{row['prompt']} | single falling water droplet | {row['causal_footprint']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} specificity prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
