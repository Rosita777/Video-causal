#!/usr/bin/env python3
"""Build four dry-step clean controls to replace the unstable puddle control."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 9500
OPENINGS = [
    "A locked-camera still-life video; absolutely nothing happens.",
    "A static scientific control recording, unchanged from beginning to end.",
    "A fixed macro video that looks like the same still photograph in every frame.",
    "A completely uneventful control video with no temporal change.",
]


def main() -> None:
    rows = []
    for index, opening in enumerate(OPENINGS):
        prompt = (
            f"{opening} A clean dry flat gray stone step is centered in frame. "
            "The stone remains completely empty, dry, and unchanged in the first frame, every "
            "middle frame, and the final frame. No water, droplet, puddle, wet mark, ripple, ring, "
            "splash, object, person, shadow movement, or lighting change appears. The camera, stone "
            "texture, background, and lighting remain fixed throughout."
        )
        rows.append(
            {
                "scene_id": f"wdcleanfix{index:03d}",
                "replaced_scene_id": "wdfive0049",
                "receiver_group_id": "wdreceiver09",
                "source_scene_id": "wdsimple0141",
                "receiver_id": "clean_dry_stone_step",
                "family": "hard_surface",
                "receiver": "a clean dry flat gray stone step",
                "condition": "clean_control",
                "prompt_variant": str(index),
                "split": "frozen_test_replacement_candidate",
                "causal_footprint": "none",
                "expected_base_target": "no",
                "expected_base_footprint": "no",
                "expected_erased_target": "no",
                "expected_erased_footprint": "no",
                "erase_instruction": "Remove the water droplet.",
                "fixed_seed": str(SEED),
                "prompt": prompt,
            }
        )

    data_path = ROOT / "data/waterdrop_clean_dry_step4.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    prompt_path = ROOT / "prompts/waterdrop_clean_dry_step4.txt"
    with prompt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Dry-step clean-control replacements; fixed seed {SEED}.\n")
        handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
        for row in rows:
            handle.write(f"{row['prompt']} | single water droplet | preserve clean scene\n")

    manifest_rows = []
    for index, row in enumerate(rows):
        manifest_rows.append(
            {
                **{key: value for key, value in row.items() if key != "prompt"},
                "shard": "0",
                "shard_index": str(index),
            }
        )
    manifest_path = ROOT / "data/waterdrop_clean_dry_step4_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Wrote 4 dry-step clean controls")


if __name__ == "__main__":
    main()
