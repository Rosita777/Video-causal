#!/usr/bin/env python3
"""Build the first 10-receiver, four-condition waterdrop training pilot."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXED_SEED = 9600
SHARD_COUNT = 4
SOURCE_SCENE_IDS = [
    "wdsimple0150",
    "wdsimple0156",
    "wdsimple0162",
    "wdsimple0166",
    "wdsimple0168",
    "wdsimple0170",
    "wdsimple0180",
    "wdsimple0188",
    "wdsimple0190",
    "wdsimple0200",
]
CONDITIONS = ["explicit_causal", "target_only", "unrelated_footprint", "clean_control"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prompt_for(receiver: str, condition: str, group_index: int) -> str:
    prefix = f"A realistic locked-camera macro video of a {receiver} in one continuous shot. "
    if condition == "explicit_causal":
        footprint = (
            "a compact circular wet patch and several tiny radial splash marks"
            if group_index % 2 == 0
            else "a small wet ring and a few short outward splash streaks"
        )
        return (
            prefix
            + "The clean dry surface is still for the first two seconds. Then exactly one clear "
            "water droplet falls from above, strikes the center, and visibly creates "
            + footprint
            + ". The bead and its wet causal footprint remain visible. No other object enters. "
            "The camera, receiver, lighting, and background remain fixed."
        )
    if condition == "target_only":
        return (
            prefix
            + "Exactly one compact clear water bead is already resting at the center in the first "
            "frame and remains in the same place throughout. The surrounding surface stays dry. "
            "No falling water, impact, splash, wet patch, ring, trail, or spreading occurs. Nothing "
            "else changes."
        )
    if condition == "unrelated_footprint":
        mark = (
            "a thin white chalk circle"
            if group_index % 2 == 0
            else "a small ring made of fine pale sand grains"
        )
        return (
            prefix
            + f"A clearly visible {mark} is already centered on the otherwise clean dry surface "
            "and remains unchanged throughout. It is a dry solid marking, not water. No water, "
            "droplet, wet patch, splash, or ripple appears. The camera, receiver, marking, lighting, "
            "and background remain fixed."
        )
    return (
        prefix
        + "The surface is completely empty, clean, and dry in every frame. No water, droplet, wet "
        "mark, ring, splash, object, person, shadow movement, or lighting change appears. The "
        "camera, receiver, texture, and background remain fixed throughout."
    )


def main() -> None:
    source = {
        row["scene_id"]: row
        for row in read_csv(ROOT / "data/waterdrop_prompt_bank_v2_simple.csv")
    }
    frozen = read_csv(ROOT / "data/waterdrop_test100_final.csv")
    frozen_receiver_ids = {row["receiver_id"] for row in frozen}
    frozen_receiver_names = {row["receiver"].strip().lower() for row in frozen}

    selected = [source[scene_id] for scene_id in SOURCE_SCENE_IDS]
    leakage = [
        row["receiver_id"]
        for row in selected
        if row["receiver_id"] in frozen_receiver_ids
        or row["receiver"].strip().lower() in frozen_receiver_names
    ]
    if leakage:
        raise ValueError(f"training/test receiver leakage: {leakage}")

    rows = []
    for group_index, source_row in enumerate(selected):
        scene_ids = {
            condition: f"wdtrain{group_index:02d}_{condition}"
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            target_condition = (
                "clean_control"
                if condition in {"explicit_causal", "target_only", "clean_control"}
                else "unrelated_footprint"
            )
            rows.append(
                {
                    "scene_id": scene_ids[condition],
                    "train_group_id": f"wdtrain{group_index:02d}",
                    "source_scene_id": source_row["scene_id"],
                    "receiver_id": source_row["receiver_id"],
                    "family": source_row["family"],
                    "receiver": source_row["receiver"],
                    "condition": condition,
                    "split": "pilot_train",
                    "expected_generated_target": "yes" if condition in {"explicit_causal", "target_only"} else "no",
                    "expected_generated_footprint": "yes" if condition in {"explicit_causal", "unrelated_footprint"} else "no",
                    "sft_target_condition": target_condition,
                    "sft_target_scene_id": scene_ids[target_condition],
                    "erase_instruction": "Remove the water droplet.",
                    "fixed_seed": str(FIXED_SEED),
                    "prompt": prompt_for(source_row["receiver"], condition, group_index),
                }
            )

    counts = Counter(row["condition"] for row in rows)
    if len(rows) != 40 or any(counts[condition] != 10 for condition in CONDITIONS):
        raise ValueError(f"invalid pilot counts: {dict(counts)}")

    data_path = ROOT / "data/waterdrop_train_pilot40.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    shards = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows = []
    for shard, shard_rows in enumerate(shards):
        prompt_path = ROOT / f"prompts/waterdrop_train_pilot40_shard_{shard}.txt"
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Waterdrop train pilot40, shard {shard}; fixed seed {FIXED_SEED}.\n")
            handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
            for shard_index, row in enumerate(shard_rows):
                handle.write(f"{row['prompt']} | single water droplet | {row['condition']}\n")
                manifest_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "prompt"},
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                    }
                )
    manifest_path = ROOT / "data/waterdrop_train_pilot40_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    heldout_path = ROOT / "data/waterdrop_train_pilot40_receivers.txt"
    heldout_path.write_text(
        "\n".join(row["receiver_id"] for row in selected) + "\n", encoding="utf-8"
    )
    print(f"Wrote pilot40; conditions={dict(counts)}; test leakage=0")


if __name__ == "__main__":
    main()
