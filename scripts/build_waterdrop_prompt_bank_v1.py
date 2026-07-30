#!/usr/bin/env python3
"""Build 800 unique one-seed waterdrop scenes within the scoped task."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


LIQUID_RECEIVERS = [
    "a calm shallow pond",
    "a transparent glass mixing bowl filled with water",
    "a plain white ceramic cup filled with water",
    "a clean metal bucket filled with water",
    "a shallow rain puddle on level pavement",
    "a rectangular glass aquarium filled with water",
    "a stainless-steel kitchen sink basin filled with water",
    "a white enamel washbasin filled with water",
    "a round stone birdbath filled with water",
    "a small garden fountain basin with the pump turned off",
    "a laboratory beaker filled with clear water",
    "a transparent measuring jug filled with water",
    "a stainless-steel saucepan filled with water",
    "a black cooking pot filled with water",
    "a porcelain soup bowl filled with water",
    "a small porcelain teacup filled with water",
    "a clear stemmed wine glass filled with water",
    "a plain glass tumbler filled with water",
    "a wide-mouth mason jar filled with water",
    "a short glass flower vase filled with water",
    "a wide porcelain basin filled with water",
    "a white bathtub filled with still water",
    "a shallow white shower tray holding a thin layer of water",
    "an open rain barrel filled with water",
    "an open metal pail filled with water",
    "a blue plastic storage tub filled with water",
    "one compartment of a white ice-cube tray filled with water",
    "a stainless-steel pet water bowl",
    "a carved granite basin filled with water",
    "a polished copper bowl filled with water",
    "a polished brass bowl filled with water",
    "a shallow aluminum pan filled with water",
    "a rectangular glass baking dish filled with water",
    "a clear acrylic tank filled with water",
    "a shallow laboratory glass tray filled with water",
    "a glazed ceramic baking dish filled with water",
    "a black stone bowl filled with water",
    "a sealed wooden bowl filled with water",
    "a clean coconut-shell cup filled with water",
    "a large seashell holding a pool of water",
    "a natural rock depression holding rainwater",
    "a leaf-shaped ceramic dish filled with water",
    "a laboratory petri dish filled with water",
    "a concave laboratory watch glass holding water",
    "a shallow puddle on a microscope glass slide",
    "a rooftop puddle on a flat membrane surface",
    "a rain-filled road pothole",
    "a shallow puddle on a flat stone step",
    "a still indoor decorative pool",
    "a small rectangular water trough",
]

HARD_RECEIVERS = [
    "dark-gray ceramic tile",
    "stainless-steel tray",
    "transparent glass tabletop",
    "polished dark marble slab",
    "plain white ceramic plate",
    "blue plastic cutting board",
    "granite countertop",
    "white quartz countertop",
    "dark slate tile",
    "polished concrete slab",
    "smooth epoxy floor sample",
    "vinyl flooring sample",
    "porcelain sink ledge",
    "white enamel stove surface",
    "flat chrome-plated shelf",
    "brushed aluminum sheet",
    "polished copper plate",
    "polished brass plate",
    "steel workbench top",
    "flat cast-iron griddle",
    "black nonstick baking tray",
    "clear glass plate",
    "clear acrylic sheet",
    "polycarbonate panel",
    "rigid white PVC board",
    "sealed hardwood tabletop",
    "laminated office desk",
    "lacquered wood panel",
    "painted metal cabinet top",
    "tempered-glass refrigerator shelf",
    "flat wall mirror",
    "glazed brick",
    "glazed pottery lid",
    "polished stone coaster",
    "glazed ceramic coaster",
    "clear glass coaster",
    "stainless-steel coaster",
    "rigid plastic cafeteria tray",
    "porcelain bathtub ledge",
    "flat section of a painted car hood",
    "flat smartphone glass screen",
    "laptop glass trackpad",
    "camera filter glass",
    "laboratory resin worktop",
    "smooth whiteboard panel",
    "polished obsidian slab",
    "polished jade slab",
    "terrazzo tile",
    "glazed porcelain floor tile",
    "sealed concrete paver",
]

ABSORBENT_RECEIVERS = [
    "white paper towel",
    "white cotton towel fixed flat",
    "unfinished pine board",
    "white facial tissue supported flat",
    "brown corrugated cardboard",
    "white blotting paper",
    "cold-pressed watercolor paper",
    "uncoated printer paper",
    "brown kraft paper",
    "newsprint paper",
    "laboratory filter paper",
    "paper coffee filter supported flat",
    "plain cotton fabric fixed flat",
    "linen fabric fixed flat",
    "denim fabric fixed flat",
    "artist canvas fixed flat",
    "dense felt sheet fixed flat",
    "wool felt sheet fixed flat",
    "microfiber cleaning cloth fixed flat",
    "natural chamois fixed flat",
    "natural cellulose sponge",
    "yellow kitchen sponge supported flat",
    "dense foam cleaning sponge",
    "unsealed cork sheet",
    "unglazed terracotta tile",
    "unglazed ceramic test tile",
    "unpainted gypsum board",
    "dry plaster surface",
    "raw concrete sample",
    "unsealed cement board",
    "porous sandstone slab",
    "unsealed limestone slab",
    "dry red brick",
    "dry mortar sample",
    "unfinished bamboo board",
    "unfinished oak board",
    "unfinished maple board",
    "unfinished birch board",
    "unfinished cedar board",
    "unfinished beech board",
    "unfinished poplar board",
    "unfinished walnut board",
    "unfinished ash board",
    "untreated suede fixed flat",
    "untreated nubuck fixed flat",
    "gray paperboard",
    "molded paper-pulp tray",
    "white rice paper supported flat",
    "cotton drawing paper",
    "uncoated paper egg carton surface",
]

GRANULAR_RECEIVERS = [
    "fine dry sand",
    "coarse dry sand",
    "dry beach sand",
    "dry silica sand",
    "loose dry garden soil",
    "dry potting soil",
    "dry clay-rich soil",
    "dry silt",
    "dry garden loam",
    "white wheat flour",
    "yellow cornmeal",
    "white cornstarch",
    "powdered sugar",
    "granulated white sugar",
    "dry brown sugar",
    "fine table salt",
    "coarse rock salt",
    "dry sea salt",
    "dry kosher salt",
    "baking soda",
    "unsweetened cocoa powder",
    "dry ground coffee",
    "dry loose black tea leaves",
    "fine dry breadcrumbs",
    "fine sawdust",
    "wood flour",
    "pale chalk dust",
    "dry cement powder",
    "dry plaster powder",
    "talcum powder",
    "graphite powder",
    "charcoal powder",
    "fine iron filings",
    "tiny clear glass beads",
    "tiny plastic beads",
    "uncooked white rice grains",
    "dry red lentils",
    "dry quinoa grains",
    "dry couscous grains",
    "white sesame seeds",
    "black poppy seeds",
    "dry chia seeds",
    "coarsely crushed black pepper",
    "ground turmeric powder",
    "ground paprika powder",
    "dry rolled oats",
    "fine clay cat litter",
    "fine horticultural perlite",
    "fine horticultural vermiculite",
    "small aquarium gravel",
]

LIQUID_VARIANTS = [
    (
        "completely still, with a perfectly flat water surface",
        "the center of the water surface",
        "a small crown splash rises and concentric ripples spread outward",
    ),
    (
        "completely still, with a calm level water surface",
        "slightly left of the center of the water surface",
        "a brief impact cavity closes and circular ripples cross the surface",
    ),
    (
        "completely still, with a shallow motionless water surface",
        "slightly right of center",
        "a low radial splash forms and several shallow ripples expand outward",
    ),
    (
        "completely still, with the water surface and its nearest edge clearly visible",
        "near the visible inner edge of the water surface",
        "an asymmetric splash forms and curved ripples travel toward the opposite edge",
    ),
]

HARD_VARIANTS = [
    (
        "clean, completely dry, horizontal, and motionless",
        "the center",
        "a brief radial splash appears and several clear water beads remain",
    ),
    (
        "clean, completely dry, horizontal, and motionless",
        "slightly left of center",
        "a small splash appears and a thin localized wet spot remains",
    ),
    (
        "clean, dry, horizontal, and covered by a hydrophobic coating",
        "slightly right of center",
        "the water recoils after a tiny splash and gathers into one compact bead",
    ),
    (
        "clean, completely dry, horizontal, and framed with one edge clearly visible",
        "near the visible edge",
        "an asymmetric splash appears and a small amount of water gathers against the edge",
    ),
    (
        "clean, dry, and held at a gentle fixed incline",
        "the upper-center area",
        "a small splash appears and one bead slides a short distance downhill leaving a wet trail",
    ),
]

ABSORBENT_VARIANTS = [
    (
        "clean, completely dry, flat, and motionless",
        "the center",
        "the water is absorbed and a small dark wet patch gradually expands",
    ),
    (
        "clean, completely dry, flat, and supported on a rigid surface",
        "slightly left of center",
        "the water soaks in and leaves a compact dark damp spot with little outward spread",
    ),
    (
        "clean, completely dry, flat, and showing a clear grain or fiber direction",
        "slightly right of center",
        "an elongated wet mark gradually spreads along the visible grain or fibers",
    ),
    (
        "clean, completely dry, flat, and firmly supported so it cannot move",
        "near one visible edge",
        "an asymmetric wet patch appears and slowly spreads inward from the contact point",
    ),
]

GRANULAR_VARIANTS = [
    (
        "arranged as a smooth level bed and completely undisturbed",
        "the center of the bed",
        "a small crater forms, a few particles move outward, and the center becomes darker and damp",
    ),
    (
        "arranged as a small dry mound and completely undisturbed",
        "the top of the mound",
        "a small indentation forms, several particles scatter down the sides, and a damp clump remains",
    ),
    (
        "arranged as a lightly compacted dry layer and completely undisturbed",
        "slightly off-center",
        "a shallow dent forms and nearby particles bind into a small darker damp cluster",
    ),
]


def make_prompt(
    receiver: str,
    condition: str,
    location: str,
    effect: str,
    clean_prefix: str,
) -> str:
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"During the first two seconds, {receiver} is {condition}. "
        f"{clean_prefix} "
        "Then exactly one large clear water droplet enters from the top of the frame, "
        f"falls visibly downward, and contacts {location}. "
        f"Only after contact, {effect}. "
        "The camera, lighting, background, receiver geometry, and all other objects remain unchanged."
    )


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs = [
        ("liquid_surface", LIQUID_RECEIVERS, LIQUID_VARIANTS, "surface_flow"),
        ("hard_surface", HARD_RECEIVERS, HARD_VARIANTS, "surface_residue"),
        ("absorbent_surface", ABSORBENT_RECEIVERS, ABSORBENT_VARIANTS, "absorption"),
        ("granular_surface", GRANULAR_RECEIVERS, GRANULAR_VARIANTS, "particle_displacement"),
    ]
    for family, receivers, variants, mechanism in specs:
        clean_prefix = (
            "No falling droplet, splash, impact cavity, or ripple is visible during "
            "these first two seconds."
            if family == "liquid_surface"
            else "No water, falling droplet, or wet footprint is visible during these first two seconds."
        )
        for receiver_index, receiver in enumerate(receivers):
            for variant_index, (condition, location, effect) in enumerate(variants):
                rows.append(
                    {
                        "scene_id": f"wdp{len(rows):04d}",
                        "family": family,
                        "mechanism": mechanism,
                        "receiver_id": f"{family}_{receiver_index:02d}",
                        "receiver": receiver,
                        "variant": str(variant_index),
                        "surface_condition": condition,
                        "impact_location": location,
                        "causal_footprint": effect,
                        "seed_policy": "one_seed_only",
                        "prompt": make_prompt(
                            receiver, condition, location, effect, clean_prefix
                        ),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, default=Path("data/waterdrop_prompt_bank_v1.csv"))
    parser.add_argument("--txt", type=Path, default=Path("prompts/waterdrop_prompt_bank_v1.txt"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    rows = build_rows()

    expected = {
        "liquid_surface": 200,
        "hard_surface": 250,
        "absorbent_surface": 200,
        "granular_surface": 150,
    }
    counts = Counter(row["family"] for row in rows)
    if counts != expected or len(rows) != 800:
        raise ValueError(f"unexpected family counts: {counts}")
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("duplicate prompts found")

    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    txt_path = args.txt if args.txt.is_absolute() else root / args.txt
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Waterdrop prompt bank v1: 800 unique scenes, one seed per prompt.\n")
        handle.write("# Format: <prompt> | <target> | <effect>\n\n")
        for row in rows:
            handle.write(
                f"{row['prompt']} | single falling water droplet | "
                f"{row['causal_footprint']}\n"
            )
    print(f"Wrote {len(rows)} prompts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
