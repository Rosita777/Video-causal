#!/usr/bin/env python3
"""Build a mixed collision-erasure and non-target-preservation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision", type=Path, default=Path("data/collision_train31.csv"))
    parser.add_argument("--waterdrop", type=Path, default=Path("data/waterdrop_generalization_eval16.csv"))
    parser.add_argument(
        "--static-manifest",
        type=Path,
        default=Path("outputs/collision_specificity8_base/generation_manifest.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/collision_mixed_preserve51.csv"))
    args = parser.parse_args()

    collision = read_csv(args.collision)
    if len(collision) != 31:
        raise SystemExit(f"Expected 31 collision rows, found {len(collision)}")
    waterdrop = read_csv(args.waterdrop)
    if len(waterdrop) != 16:
        raise SystemExit(f"Expected 16 waterdrop rows, found {len(waterdrop)}")
    static_manifest = json.loads(args.static_manifest.read_text(encoding="utf-8"))
    static_items = list(static_manifest["items"][:4])
    if len(static_items) != 4:
        raise SystemExit("Expected four static-control generation items")

    records = list(collision)
    for index, row in enumerate(waterdrop):
        records.append(
            {
                "scene_id": f"preserve_waterdrop_{index:03d}",
                "train_group_id": row["pair_id"],
                "receiver_id": row["receiver_key"],
                "receiver": row["receiver"],
                "condition": "non_target_causal_preserve",
                "prompt": row["prompt"],
                "generated_video": row["factual_video"],
                "desired_target_video": row["factual_video"],
                "training_role": "preserve",
                "training_objective": "identity_noise_prediction",
                "residual_mask_enabled": "no",
                "residual_mask_factual_video": "",
                "residual_mask_target_video": "",
                "reference_end_exclusive": "",
                "source_split": "waterdrop_generalization_eval16",
                "source_index": str(index),
            }
        )
    for index, item in enumerate(static_items):
        video_path = str(item["video_path"])
        records.append(
            {
                "scene_id": f"preserve_static_{index:03d}",
                "train_group_id": f"specificity_static_{index:03d}",
                "receiver_id": "static_control",
                "receiver": "non-target static tabletop scene",
                "condition": "non_target_static_preserve",
                "prompt": str(item["prompt"]),
                "generated_video": video_path,
                "desired_target_video": video_path,
                "training_role": "preserve",
                "training_objective": "identity_noise_prediction",
                "residual_mask_enabled": "no",
                "residual_mask_factual_video": "",
                "residual_mask_target_video": "",
                "reference_end_exclusive": "",
                "source_split": "collision_specificity8_base",
                "source_index": str(index),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print("Wrote 51 rows: erase=31 preserve_waterdrop=16 preserve_static=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
