#!/usr/bin/env python3
"""Build a unified annotation table for the dynamic water-impact eval12."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHODS = {
    "original": "outputs/water_impact_dynamic_v1/eval12_base/videos",
    "negative_prompt": "outputs/water_impact_dynamic_v1/eval12_negative_prompt/videos",
    "t2vunlearning": "outputs/water_impact_dynamic_v1/eval12_t2vunlearning/videos",
    "videoeraser": "outputs/water_impact_dynamic_v1/eval12_videoeraser/videos",
    "ours_v2": "outputs/water_impact_dynamic_v1/eval12_preserve_v2_ckpt200_scale1p25/videos",
}

FIELDS = [
    "sample_index",
    "pair_id",
    "generalization_group",
    "source_object",
    "receiver",
    "method",
    "video_path",
    "video_exists",
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
    "strict_success",
    "notes",
]


def one_video(root: Path, index: int) -> Path | None:
    matches = sorted(root.glob(f"{index:03d}_*.mp4"))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple videos for index {index} in {root}")
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, default=Path("data/water_impact_dynamic_v1/eval12.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/water_impact_dynamic_eval12/review.csv"),
    )
    args = parser.parse_args()

    with args.eval_csv.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    rows = []
    for index, sample in enumerate(samples):
        for method, directory in METHODS.items():
            video = one_video(Path(directory), index)
            rows.append(
                {
                    "sample_index": index,
                    "pair_id": sample["pair_id"],
                    "generalization_group": sample["generalization_group"],
                    "source_object": sample["source_object"],
                    "receiver": sample["receiver"],
                    "method": method,
                    "video_path": str(video) if video else "",
                    "video_exists": "yes" if video else "no",
                    "target_visibility_0_absent_2_clear": "",
                    "footprint_visibility_0_absent_2_clear": "",
                    "receiver_preservation_0_bad_2_good": "",
                    "video_quality_0_bad_2_good": "",
                    "strict_success": "",
                    "notes": "",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
