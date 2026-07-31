#!/usr/bin/env python3
"""Build 30 additional resting-bead candidates for target-only replenishment."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARD_COUNT = 6
FIXED_SEED = 9300
SOURCE_SCENE_IDS = [
    "wdsimple0158",
    "wdsimple0164",
    "wdsimple0174",
    "wdsimple0176",
    "wdsimple0184",
    "wdsimple0198",
    "wdsimple0202",
    "wdsimple0214",
    "wdsimple0218",
    "wdsimple0220",
    "wdsimple0226",
    "wdsimple0228",
    "wdsimple0238",
    "wdsimple0242",
    "wdsimple0246",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_prompt(receiver: str, variant: int) -> str:
    opening = (
        "A locked-off scientific still-life recording, not an action scene."
        if variant == 0
        else "A static product-shot video with a fixed macro camera; absolutely nothing happens."
    )
    return (
        f"{opening} Before recording begins, exactly one compact clear sessile water bead is "
        f"already resting at the center of a {receiver}. The same single rounded bead is fully "
        "visible in the first frame, every middle frame, and the final frame, unchanged in size, "
        "shape, and position. The surrounding hard surface stays completely dry. No object enters "
        "the frame and no event occurs: no falling drop, formation, growth, impact, splash, ripple, "
        "ring, spreading, trail, or evaporation. Camera, lighting, surface, and background remain "
        "identical throughout."
    )


def main() -> int:
    source = {row["scene_id"]: row for row in read_csv(ROOT / "data/waterdrop_prompt_bank_v2_simple.csv")}
    rows: list[dict[str, str]] = []
    for source_scene_id in SOURCE_SCENE_IDS:
        source_row = source[source_scene_id]
        for variant in range(2):
            index = len(rows)
            rows.append(
                {
                    "scene_id": f"wdreplenish{index:03d}",
                    "source_scene_id": source_scene_id,
                    "receiver_id": source_row["receiver_id"],
                    "family": "hard_surface",
                    "receiver": source_row["receiver"],
                    "prompt_variant": str(variant),
                    "condition": "target_only_resting_bead",
                    "split": "frozen_test_replenishment_candidate",
                    "expected_base_target": "yes",
                    "expected_base_footprint": "no",
                    "expected_erased_target": "no",
                    "expected_erased_footprint": "no",
                    "erase_instruction": "Remove the water droplet.",
                    "fixed_seed": str(FIXED_SEED),
                    "prompt": make_prompt(source_row["receiver"], variant),
                }
            )

    csv_path = ROOT / "data/waterdrop_target_only_replenish30.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shards = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows = []
    for shard, shard_rows in enumerate(shards):
        prompt_path = ROOT / f"prompts/waterdrop_target_only_replenish30_shard_{shard}.txt"
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Resting-bead replenishment shard {shard}; fixed seed {FIXED_SEED}.\n")
            handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
            for shard_index, row in enumerate(shard_rows):
                handle.write(f"{row['prompt']} | single water droplet | no causal footprint\n")
                manifest_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "prompt"},
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                    }
                )

    manifest_path = ROOT / "data/waterdrop_target_only_replenish30_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Wrote 30 replenishment prompts; shards=[5, 5, 5, 5, 5, 5]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
