#!/usr/bin/env python3
"""Append manually accepted expansion pairs to the frozen waterdrop split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key_for(receiver: str) -> str:
    value = receiver.lower()
    if "saucepan" in value or "jug" in value or "barrel" in value or "birdbath" in value:
        return "liquid_container"
    if "hardwood" in value:
        return "wood"
    if "acrylic" in value or "slate" in value or "griddle" in value:
        return "hard_surface"
    if "towel" in value or "felt" in value or "sponge" in value or "pine" in value:
        return "absorbent_fabric"
    if "cornmeal" in value or "sawdust" in value or "sand" in value or "soil" in value:
        return "particulate"
    return value.replace(" ", "_")


def footprint_for(family: str, receiver: str) -> str:
    if family == "liquid_surface":
        return "splash_ripple"
    if family == "absorbent_surface":
        return "spreading_wet_patch"
    if family == "granular_surface":
        return "crater_or_damp_particle_mark"
    return "splash_wet_mark"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/waterdrop_generalization_split_v1.csv"))
    parser.add_argument("--expansion", type=Path, default=Path("data/waterdrop_generalization_expansion16.csv"))
    parser.add_argument("--screen", type=Path, default=Path("data/waterdrop_generalization_expansion16_auto_screen.csv"))
    parser.add_argument("--review", type=Path, default=Path("data/waterdrop_generalization_expansion16_review.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/waterdrop_generalization_split_v2.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/waterdrop_generalization_split_v2_summary.json"))
    args = parser.parse_args()

    base = read(args.base)
    expansion = {row["expansion_id"]: row for row in read(args.expansion)}
    screen = {row["scene_id"]: row for row in read(args.screen)}
    accepted = [row for row in read(args.review) if row["decision"] == "accepted"]
    records = list(base)
    for index, review in enumerate(accepted, start=1):
        candidate = expansion[review["scene_id"]]
        observed = screen[review["scene_id"]]
        records.append(
            {
                "pair_id": f"waterdrop_expansion_{review['scene_id']}",
                "sample_id": review["scene_id"],
                "batch": "generalization_expansion16",
                "seed": candidate["fixed_seed"],
                "receiver": candidate["receiver"],
                "receiver_key": key_for(candidate["receiver"]),
                "footprint_family": footprint_for(candidate["family"], candidate["receiver"]),
                "split": "train_candidate",
                "prompt": candidate["prompt"],
                "target_concept": candidate["target_concept"],
                "expected_effect": candidate["expected_effect"],
                "factual_video": observed["video_path"],
                "target_video": f"outputs/waterdrop_generalization_expansion16_pairs/{review['scene_id']}/counterfactual_static.mp4",
                "clean_reference": f"outputs/waterdrop_generalization_expansion16_pairs/{review['scene_id']}/clean_reference.png",
                "reference_end_exclusive": review["reference_end_exclusive"],
                "first_frame_reference_mae_255": "",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    counts = Counter(row["split"] for row in records)
    summary = {
        "version": "v2",
        "base_pairs": len(base),
        "accepted_expansion_pairs": len(accepted),
        "total_pairs": len(records),
        "counts_by_split": dict(counts),
        "train_candidate_by_footprint": dict(Counter(row["footprint_family"] for row in records if row["split"] == "train_candidate")),
        "accepted_expansion_ids": [row["scene_id"] for row in accepted],
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
