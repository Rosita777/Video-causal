#!/usr/bin/env python3
"""Build dynamic counterfactual training and generalization prompt pairs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path


TRAIN_SOURCES = [
    ("water_droplet", "one large clear water droplet"),
    ("ice_cube", "one small transparent ice cube"),
    ("red_apple", "one small red apple"),
    ("green_lime", "one small green lime"),
    ("blue_marble", "one blue glass marble"),
    ("wooden_cube", "one small light wooden cube"),
    ("steel_ball", "one polished steel ball bearing"),
    ("plastic_block", "one small red plastic toy block"),
]

TEST_SOURCES = [
    ("gray_stone", "one smooth gray stone"),
    ("yellow_ball", "one small yellow rubber ball"),
    ("walnut", "one whole brown walnut"),
    ("strawberry", "one ripe red strawberry"),
    ("ceramic_bead", "one white ceramic bead"),
    ("pine_cone", "one small brown pine cone"),
]

TRAIN_RECEIVERS = [
    ("shallow_pond", "a calm shallow pond"),
    ("glass_bowl", "a transparent glass mixing bowl filled with water"),
    ("white_basin", "a wide white ceramic basin filled with water"),
    ("glass_tank", "a clear rectangular glass tank filled with water"),
    ("metal_bucket", "a clean metal bucket filled with water"),
    ("kitchen_sink", "a stainless-steel kitchen sink basin filled with water"),
    ("porcelain_bowl", "a plain porcelain soup bowl filled with water"),
    ("cooking_pot", "a black cooking pot filled with water"),
    ("garden_birdbath", "a round stone birdbath filled with water"),
    ("glass_dish", "a rectangular glass baking dish filled with water"),
    ("plastic_tub", "a blue plastic storage tub filled with water"),
    ("laboratory_beaker", "a laboratory beaker filled with clear water"),
]

TEST_RECEIVERS = [
    ("copper_bowl", "a polished copper bowl filled with water"),
    ("bathtub", "a white bathtub filled with water"),
    ("rain_barrel", "an open rain barrel filled with water"),
    ("stone_basin", "a carved granite basin filled with water"),
    ("acrylic_tank", "a clear acrylic tank filled with water"),
    ("water_trough", "a small rectangular water trough"),
    ("pet_bowl", "a stainless-steel pet water bowl"),
    ("mason_jar", "a wide-mouth mason jar filled with water"),
]

FIELDS = [
    "protocol_version",
    "pair_id",
    "split",
    "generalization_group",
    "mechanism",
    "source_id",
    "source_object",
    "source_seen",
    "receiver_id",
    "receiver",
    "receiver_seen",
    "prompt_variant",
    "training_prompt",
    "target_generation_prompt",
    "expected_factual_event",
    "expected_counterfactual_state",
    "seed",
    "num_frames",
    "fps",
    "factual_video",
    "desired_target_video",
]


def factual_prompt(source: str, receiver: str, variant: str) -> str:
    if variant == "direct":
        event = (
            f"{source.capitalize()} falls naturally from above, enters the center of the water in "
            f"{receiver}, and makes contact. The contact produces a visible brief splash "
            "followed by circular ripples spreading across the water."
        )
    else:
        event = (
            f"In a single natural motion, {source} drops into the center of the water in "
            f"{receiver}. After it touches the water, a short splash rises and expanding "
            "ripples travel outward."
        )
    return (
        "A simple realistic locked-camera video in one continuous shot. "
        f"{event} The receiver, camera, lighting, and background remain consistent."
    )


def counterfactual_prompt(receiver: str, variant: str) -> str:
    if variant == "direct":
        motion = (
            "The water moves gently throughout the shot with subtle natural surface "
            "undulations and slowly changing reflections."
        )
    else:
        motion = (
            "Soft ambient air creates mild continuous movement on the water surface, "
            "while reflected light shifts naturally over time."
        )
    return (
        "A simple realistic locked-camera video in one continuous shot showing "
        f"{receiver}. {motion} Nothing falls into or strikes the water. No impact splash, "
        "impact cavity, or impact-generated circular wave appears. The receiver, camera, "
        "lighting, and background remain consistent."
    )


def make_row(
    *,
    index: int,
    split: str,
    group: str,
    source: tuple[str, str],
    receiver: tuple[str, str],
    variant: str,
    source_seen: bool,
    receiver_seen: bool,
) -> dict[str, str]:
    source_id, source_name = source
    receiver_id, receiver_name = receiver
    pair_id = f"widyn_{split}_{index:04d}_{source_id}_{receiver_id}_{variant}"
    return {
        "protocol_version": "water_impact_dynamic_v1",
        "pair_id": pair_id,
        "split": split,
        "generalization_group": group,
        "mechanism": "object_enters_water",
        "source_id": source_id,
        "source_object": source_name,
        "source_seen": "yes" if source_seen else "no",
        "receiver_id": receiver_id,
        "receiver": receiver_name,
        "receiver_seen": "yes" if receiver_seen else "no",
        "prompt_variant": variant,
        "training_prompt": factual_prompt(source_name, receiver_name, variant),
        "target_generation_prompt": counterfactual_prompt(receiver_name, variant),
        "expected_factual_event": "source enters water, then splash and expanding ripples",
        "expected_counterfactual_state": (
            "same receiver with natural water motion but no source or impact footprint"
        ),
        "seed": str(26000 + index),
        "num_frames": "49",
        "fps": "8",
        "factual_video": "",
        "desired_target_video": "",
    }


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    for index, (source, receiver, variant) in enumerate(
        product(TRAIN_SOURCES, TRAIN_RECEIVERS, ("direct", "natural"))
    ):
        train_rows.append(
            make_row(
                index=index,
                split="train",
                group="seen_source_seen_receiver",
                source=source,
                receiver=receiver,
                variant=variant,
                source_seen=True,
                receiver_seen=True,
            )
        )

    test_specs: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    test_specs.extend(
        ("unseen_source", source, receiver)
        for source, receiver in product(TEST_SOURCES, TRAIN_RECEIVERS[:4])
    )
    test_specs.extend(
        ("unseen_receiver", source, receiver)
        for source, receiver in product(TRAIN_SOURCES[:3], TEST_RECEIVERS)
    )
    both_pairs = list(product(TEST_SOURCES, TEST_RECEIVERS))
    test_specs.extend(
        ("unseen_source_and_receiver", source, receiver)
        for source, receiver in both_pairs
        if (TEST_SOURCES.index(source) + TEST_RECEIVERS.index(receiver)) % 2 == 0
    )
    test_rows = [
        make_row(
            index=index,
            split="test",
            group=group,
            source=source,
            receiver=receiver,
            variant="direct" if index % 2 == 0 else "natural",
            source_seen=group == "unseen_receiver",
            receiver_seen=group == "unseen_source",
        )
        for index, (group, source, receiver) in enumerate(test_specs)
    ]
    return train_rows, test_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_prompt_file(path: Path, rows: list[dict[str, str]], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row[key]} | Water impact | "
                f"{row['expected_counterfactual_state']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/water_impact_dynamic_v1"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("prompts/water_impact_dynamic_v1"))
    args = parser.parse_args()

    train_rows, test_rows = build_rows()
    write_csv(args.data_dir / "train_pairs.csv", train_rows)
    write_csv(args.data_dir / "test_pairs.csv", test_rows)
    write_prompt_file(args.prompt_dir / "train_factual.prompts", train_rows, "training_prompt")
    write_prompt_file(
        args.prompt_dir / "train_counterfactual.prompts", train_rows, "target_generation_prompt"
    )
    write_prompt_file(args.prompt_dir / "test_factual.prompts", test_rows, "training_prompt")
    write_prompt_file(
        args.prompt_dir / "test_counterfactual.prompts", test_rows, "target_generation_prompt"
    )
    print(f"Wrote {len(train_rows)} training pairs and {len(test_rows)} test pairs")


if __name__ == "__main__":
    main()
