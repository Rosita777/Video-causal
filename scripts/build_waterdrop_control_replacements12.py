#!/usr/bin/env python3
"""Build candidates for the three failed non-target controls in test100."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARD_COUNT = 3
FIXED_SEED = 9400
SOURCE_IDS = ["wdfive0008", "wdfive0018", "wdfive0049"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unrelated_prompt(receiver: str, variant: int) -> str:
    motions = [
        "Gentle low-amplitude ripples are already traveling across the water surface in the first frame and continue throughout, caused only by a hidden vibration source outside the shot.",
        "The water is flat at first; halfway through, a gentle breeze from outside the frame creates clearly visible natural ripples that continue to the end.",
        "A hidden low-power fan outside the shot creates several small visible waves across the water surface throughout the recording.",
        "A concealed vibration source beneath the receiver creates gentle repeating ripples on the water surface, while the receiver itself stays fixed.",
    ]
    return (
        "A realistic locked-camera macro video in one continuous shot. "
        f"A {receiver} remains centered, intact, and completely unobstructed. {motions[variant]} "
        "No falling water, water bead, splash, hand, tool, or foreign object enters the frame. "
        "Only the water-surface ripples move; the receiver, camera, lighting, and background remain unchanged."
    )


def clean_prompt(receiver: str, variant: int) -> str:
    openings = [
        "A locked-camera still-life video; absolutely nothing happens.",
        "A static scientific control recording, visually identical from beginning to end.",
        "A fixed macro video that looks like an unchanged still photograph for the full clip.",
        "A completely uneventful control video with no temporal change.",
    ]
    return (
        f"{openings[variant]} A {receiver} is already present and remains perfectly still. "
        "Its water surface stays flat and mirror-smooth in the first frame, every middle frame, "
        "and the final frame. No droplet, ripple, wave, ring, splash, wind, vibration, object, or "
        "lighting change occurs. The receiver, camera, background, and reflections remain fixed."
    )


def main() -> int:
    source = {
        row["scene_id"]: row
        for row in read_csv(ROOT / "data/waterdrop_five_condition_test100.csv")
    }
    rows: list[dict[str, str]] = []
    for replaced_scene_id in SOURCE_IDS:
        base = source[replaced_scene_id]
        for variant in range(4):
            index = len(rows)
            prompt = (
                unrelated_prompt(base["receiver"], variant)
                if base["condition"] == "unrelated_footprint"
                else clean_prompt(base["receiver"], variant)
            )
            rows.append(
                {
                    "scene_id": f"wdcontrolfix{index:03d}",
                    "replaced_scene_id": replaced_scene_id,
                    "receiver_group_id": base["receiver_group_id"],
                    "source_scene_id": base["source_scene_id"],
                    "receiver_id": base["receiver_id"],
                    "family": base["family"],
                    "receiver": base["receiver"],
                    "condition": base["condition"],
                    "prompt_variant": str(variant),
                    "split": "frozen_test_replacement_candidate",
                    "causal_footprint": base["causal_footprint"],
                    "expected_base_target": base["expected_base_target"],
                    "expected_base_footprint": base["expected_base_footprint"],
                    "expected_erased_target": base["expected_erased_target"],
                    "expected_erased_footprint": base["expected_erased_footprint"],
                    "erase_instruction": base["erase_instruction"],
                    "fixed_seed": str(FIXED_SEED),
                    "prompt": prompt,
                }
            )

    csv_path = ROOT / "data/waterdrop_control_replacements12.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shards = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows = []
    for shard, shard_rows in enumerate(shards):
        path = ROOT / f"prompts/waterdrop_control_replacements12_shard_{shard}.txt"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Control replacement candidates, shard {shard}; seed {FIXED_SEED}.\n")
            handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
            for shard_index, row in enumerate(shard_rows):
                handle.write(f"{row['prompt']} | single water droplet | preserve scene\n")
                manifest_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "prompt"},
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                    }
                )
    manifest = ROOT / "data/waterdrop_control_replacements12_run_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Wrote 12 replacement candidates; shards=[4, 4, 4]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
