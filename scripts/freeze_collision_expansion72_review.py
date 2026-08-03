#!/usr/bin/env python3
"""Freeze the human semantic review of collision expansion72."""

from __future__ import annotations

import csv
from pathlib import Path


ACCEPTED = {
    "collisionx007": (24, "single ball contacts a balsa block before the block tips"),
    "collisionx012": (28, "single ball contacts the left wooden block before it tilts"),
    "collisionx027": (8, "single ball contacts a short can before the can tips"),
    "collisionx033": (16, "single ball reaches a paper cup before the cup moves"),
    "collisionx035": (24, "single ball reaches the metal tins before one tin moves"),
    "collisionx063": (24, "single ball contacts a wooden peg before the peg falls"),
    "collisionx066": (12, "single ball reaches the toy pawns before they fall"),
}

HANDS = {"collisionx020", "collisionx029", "collisionx030"}
NO_CLEAN_PREFIX = {
    "collisionx017", "collisionx044", "collisionx046", "collisionx047",
    "collisionx056", "collisionx064", "collisionx067", "collisionx069",
}
PRE_MOTION = {"collisionx051"}
MULTIPLE_BALLS = {
    "collisionx000", "collisionx002", "collisionx004", "collisionx005",
    "collisionx008", "collisionx010", "collisionx013", "collisionx016",
    "collisionx018", "collisionx019", "collisionx021", "collisionx022",
    "collisionx023", "collisionx024", "collisionx025", "collisionx026",
    "collisionx028", "collisionx031", "collisionx034", "collisionx037",
    "collisionx038", "collisionx039", "collisionx040", "collisionx041",
    "collisionx042", "collisionx043", "collisionx045", "collisionx048",
    "collisionx049", "collisionx052", "collisionx053", "collisionx054",
    "collisionx055", "collisionx058", "collisionx059", "collisionx060",
    "collisionx062", "collisionx068", "collisionx071",
}
NO_TARGET_MOTION = {
    "collisionx001", "collisionx003", "collisionx006", "collisionx009",
    "collisionx011", "collisionx014", "collisionx015", "collisionx032",
    "collisionx036", "collisionx050", "collisionx061", "collisionx070",
}
AMBIGUOUS = {"collisionx057", "collisionx065"}


def rejection_reason(scene_id: str) -> str:
    if scene_id in HANDS:
        return "hand or person enters the generated video"
    if scene_id in NO_CLEAN_PREFIX:
        return "ball is visible from the beginning, so no clean counterfactual prefix exists"
    if scene_id in PRE_MOTION:
        return "receiver objects move before the ball contact"
    if scene_id in MULTIPLE_BALLS:
        return "the ball duplicates or extra ball-like objects appear"
    if scene_id in NO_TARGET_MOTION:
        return "the ball moves but the intended receiver does not visibly move after contact"
    if scene_id in AMBIGUOUS:
        return "contact order or target identity is visually ambiguous"
    raise KeyError(f"Missing manual decision for {scene_id}")


def main() -> int:
    source = Path("data/collision_expansion72_auto_screen.csv")
    output = Path("data/collision_expansion72_semantic_review.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    reviewed = []
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id in ACCEPTED:
            reference_end, reason = ACCEPTED[scene_id]
            decision = "accept"
        else:
            reference_end, reason = "", rejection_reason(scene_id)
            decision = "reject"
        reviewed.append({
            **row,
            "human_decision": decision,
            "human_reason": reason,
            "reference_end_exclusive": reference_end,
        })

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(reviewed)
    print(f"Wrote {len(reviewed)} reviews: {len(ACCEPTED)} accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
