#!/usr/bin/env python3
"""Freeze human review and the train/validation selection for feasible72."""

from __future__ import annotations

import csv
from pathlib import Path


ACCEPTED = {
    "collisionf008": (8, "train"),
    "collisionf009": (20, "train"),
    "collisionf013": (32, "validation"),
    "collisionf014": (24, "train"),
    "collisionf017": (36, "train"),
    "collisionf019": (32, "validation"),
    "collisionf024": (32, "validation"),
    "collisionf025": (28, "validation"),
    "collisionf027": (20, "train"),
    "collisionf029": (8, "train"),
    "collisionf041": (32, "validation"),
    "collisionf045": (40, "validation"),
    "collisionf046": (40, "train"),
    "collisionf048": (16, "train"),
    "collisionf049": (28, "train"),
    "collisionf050": (24, "validation"),
    "collisionf053": (24, "train"),
    "collisionf054": (24, "train"),
    "collisionf057": (32, "train"),
    "collisionf062": (20, "train"),
    "collisionf069": (24, "train"),
}


def main() -> int:
    source = Path("data/collision_feasible72_auto_screen.csv")
    output = Path("data/collision_feasible72_semantic_review.csv")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    reviewed = []
    for row in rows:
        scene_id = row["scene_id"]
        accepted = ACCEPTED.get(scene_id)
        if accepted:
            reference_end, split = accepted
            decision = "accept"
            reason = "one ball contacts a receiver before visible receiver motion"
        else:
            reference_end, split = "", "reject"
            decision = "reject"
            reason = "strict clean-prefix, single-ball, contact-order, or receiver-motion requirement fails"
        reviewed.append({
            **row,
            "human_decision": decision,
            "human_reason": reason,
            "reference_end_exclusive": reference_end,
            "selection_split": split,
        })

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviewed[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(reviewed)

    train_count = sum(row["selection_split"] == "train" for row in reviewed)
    validation_count = sum(row["selection_split"] == "validation" for row in reviewed)
    print(f"Wrote {len(reviewed)} reviews: train={train_count}, validation={validation_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
