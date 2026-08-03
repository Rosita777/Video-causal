#!/usr/bin/env python3
"""Combine collision erasure rows with category-agnostic teacher-preservation rows."""

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
    parser.add_argument("--target-only", type=Path, default=Path("data/collision_object_only5.csv"))
    parser.add_argument(
        "--generic-root", type=Path, default=Path("outputs/generic_preservation32_base")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/collision_general_preserve68.csv")
    )
    args = parser.parse_args()

    erase_rows = read_csv(args.collision) + read_csv(args.target_only)
    if len(erase_rows) != 36:
        raise SystemExit(f"Expected 36 erase rows, found {len(erase_rows)}")

    items = []
    for shard in range(2):
        manifest = args.generic_root / f"shard_{shard}" / "generation_manifest.json"
        items.extend(json.loads(manifest.read_text(encoding="utf-8"))["items"])
    if len(items) != 32:
        raise SystemExit(f"Expected 32 generic generations, found {len(items)}")

    records = list(erase_rows)
    for index, item in enumerate(items):
        video = str(item["video_path"])
        records.append(
            {
                "scene_id": f"generic_preserve_{index:03d}",
                "train_group_id": f"generic_preserve_{index:03d}",
                "receiver_id": "generic_non_target",
                "receiver": "generic non-target scene",
                "condition": "teacher_preserve",
                "prompt": str(item["prompt"]),
                "generated_video": video,
                "desired_target_video": video,
                "training_role": "preserve",
                "training_objective": "frozen_teacher_prediction",
                "residual_mask_enabled": "no",
                "residual_mask_factual_video": "",
                "residual_mask_target_video": "",
                "reference_end_exclusive": "",
                "source_split": "generic_preservation32",
                "source_index": str(index),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} rows: erase={len(erase_rows)} preserve={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
