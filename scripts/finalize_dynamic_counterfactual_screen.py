#!/usr/bin/env python3
"""Apply the recorded semantic review to the dynamic-target screen."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REJECTIONS = {
    4: "receiver is not recognizable as a water-filled white basin",
    21: "receiver collapses into an abstract blue strip",
    36: "receiver is a plate-like shape rather than a porcelain water bowl",
    39: "a human hand enters and stirs the water",
    56: "receiver is a blue tank-like box rather than a metal bucket",
    57: "large white surface patches resemble an unwanted footprint",
    60: "target is nearly static",
    61: "receiver is not visible",
    81: "a human hand enters the bucket",
    86: "a large white region appears and changes across the water surface",
    92: "a white object enters the plastic tub",
    128: "water visibly pours into the bucket during the shot",
    153: "a human hand appears beside the bucket",
    158: "large white patches abruptly appear across the pot surface",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    for row in rows:
        index = int(row["pair_index"])
        if index in REJECTIONS:
            row["semantic_status"] = "reject"
            row["final_status"] = "reject"
            row["notes"] = REJECTIONS[index]
        else:
            row["semantic_status"] = "accept"
            row["final_status"] = "accept"
            if row["technical_status"] != "candidate":
                row["notes"] = "visual review accepted despite technical warning"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Finalized {len(rows)} rows: {dict(Counter(row['final_status'] for row in rows))}")


if __name__ == "__main__":
    main()
