#!/usr/bin/env python3
"""Build cache manifests for other-ball collisions and red-ball negation controls."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDS = [
    "scene_id", "train_group_id", "receiver_id", "receiver", "condition", "prompt",
    "generated_video", "desired_target_video", "training_role", "training_objective",
    "residual_mask_enabled", "residual_mask_factual_video", "residual_mask_target_video",
    "reference_end_exclusive", "source_split", "source_index",
]


def generation_items(root: Path) -> list[dict[str, object]]:
    items = []
    for shard in range(2):
        manifest = root / f"shard_{shard}" / "generation_manifest.json"
        items.extend(json.loads(manifest.read_text(encoding="utf-8"))["items"])
    return items


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--other-ball-root", type=Path, default=Path("outputs/non_target_ball_collision10"))
    parser.add_argument("--negation-root", type=Path, default=Path("outputs/red_ball_negation8"))
    parser.add_argument("--pair-root", type=Path, default=Path("outputs/non_target_ball_collision10_pairs"))
    parser.add_argument("--accepted-other-ball", default="0,1,3,5,6,7")
    parser.add_argument("--accepted-negation", default="1,3")
    parser.add_argument("--other-ball-output", type=Path, default=Path("data/non_target_ball_collision10_pairs.csv"))
    parser.add_argument("--negation-output", type=Path, default=Path("data/red_ball_negation8.csv"))
    args = parser.parse_args()

    accepted_other = {int(value) for value in args.accepted_other_ball.split(",") if value.strip()}
    accepted_negation = {int(value) for value in args.accepted_negation.split(",") if value.strip()}
    builder = Path("scripts/build_static_counterfactual_pair.py").resolve()

    other_rows = []
    for index, item in enumerate(generation_items(args.other_ball_root)):
        if index not in accepted_other:
            continue
        factual = Path(str(item["video_path"]))
        pair_dir = args.pair_root / f"other_ball_{index:03d}"
        pair_manifest = pair_dir / "pair_manifest.json"
        if not pair_manifest.exists():
            subprocess.run(
                [sys.executable, str(builder), "--input", str(factual), "--output-dir", str(pair_dir), "--reference-end", "16"],
                check=True,
            )
        target = json.loads(pair_manifest.read_text(encoding="utf-8"))["output_video"]
        other_rows.append({
            "scene_id": f"other_ball_{index:03d}", "train_group_id": f"other_ball_{index:03d}",
            "receiver_id": "other_ball_collision", "receiver": "other-colored-ball receiver",
            "condition": "non_target_object_collision", "prompt": item["prompt"],
            "generated_video": str(factual), "desired_target_video": str(target),
            "training_role": "erase", "training_objective": "analysis_pair",
            "residual_mask_enabled": "yes", "residual_mask_factual_video": str(factual),
            "residual_mask_target_video": str(target), "reference_end_exclusive": "16",
            "source_split": "non_target_ball_collision10", "source_index": str(index),
        })

    negation_rows = []
    for index, item in enumerate(generation_items(args.negation_root)):
        if index not in accepted_negation:
            continue
        video = str(item["video_path"])
        negation_rows.append({
            "scene_id": f"red_ball_negation_{index:03d}", "train_group_id": f"red_ball_negation_{index:03d}",
            "receiver_id": "red_ball_negation", "receiver": "static non-target receiver",
            "condition": "target_text_negation", "prompt": item["prompt"],
            "generated_video": video, "desired_target_video": video,
            "training_role": "preserve", "training_objective": "analysis_only",
            "residual_mask_enabled": "no", "residual_mask_factual_video": "",
            "residual_mask_target_video": "", "reference_end_exclusive": "",
            "source_split": "red_ball_negation8", "source_index": str(index),
        })

    write_csv(args.other_ball_output, other_rows)
    write_csv(args.negation_output, negation_rows)
    print(f"Wrote other_ball={len(other_rows)} negation={len(negation_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
