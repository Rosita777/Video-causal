#!/usr/bin/env python3
"""Record the manual semantic review of the resting-bead target-only set."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/waterdrop_target_only_resting20.csv"
OUTPUT = ROOT / "data/waterdrop_target_only_resting20_semantic_review.csv"

PASS_IDS = {
    "wdresting000",
    "wdresting001",
    "wdresting002",
    "wdresting005",
    "wdresting006",
    "wdresting007",
    "wdresting010",
    "wdresting011",
    "wdresting016",
    "wdresting017",
}

FAIL_REASONS = {
    "wdresting003": "bead appears late and spreads into a ring",
    "wdresting004": "bead is absent initially and grows later",
    "wdresting008": "wet ring is present and bead forms later",
    "wdresting009": "bead appears late and then fades",
    "wdresting012": "bead appears late with a surrounding wet ring",
    "wdresting013": "impact-like concentric rings and late bead formation",
    "wdresting014": "large wet footprint and changing bead",
    "wdresting015": "bead appears late and spreads",
    "wdresting018": "bead changes shape substantially instead of remaining motionless",
    "wdresting019": "bead is absent initially and appears late",
}


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    reviewed = []
    for row in rows:
        scene_id = row["scene_id"]
        passed = scene_id in PASS_IDS
        reviewed.append(
            {
                "scene_id": scene_id,
                "receiver": row["receiver"],
                "semantic_pass": "yes" if passed else "no",
                "failure_reason": "" if passed else FAIL_REASONS[scene_id],
                "review_basis": "12-frame contact sheet",
            }
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reviewed[0].keys())
        writer.writeheader()
        writer.writerows(reviewed)

    passed = sum(row["semantic_pass"] == "yes" for row in reviewed)
    print(f"wrote {OUTPUT}: {passed}/{len(reviewed)} pass")


if __name__ == "__main__":
    main()
