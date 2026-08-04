#!/usr/bin/env python3
"""Rank generic preservation videos by visible temporal change."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/generic_preservation32_base"))
    parser.add_argument("--output", type=Path, default=Path("data/generic_preservation32_motion_scores.csv"))
    parser.add_argument("--select", type=int, default=20)
    args = parser.parse_args()

    items = []
    for shard in range(2):
        manifest = args.root / f"shard_{shard}" / "generation_manifest.json"
        for item in json.loads(manifest.read_text(encoding="utf-8"))["items"]:
            frames = iio.imread(item["video_path"], plugin="pyav").astype(np.float32) / 255.0
            early_late = float(np.abs(frames[-8:].mean(0) - frames[:8].mean(0)).mean())
            adjacent = float(np.abs(frames[1:] - frames[:-1]).mean())
            items.append(
                {
                    "prompt": item["prompt"],
                    "seed": item["seed"],
                    "video_path": item["video_path"],
                    "early_late_mae": early_late,
                    "adjacent_mae": adjacent,
                }
            )
    items.sort(key=lambda row: row["early_late_mae"], reverse=True)
    for index, item in enumerate(items):
        item["motion_rank"] = index + 1
        item["selected"] = "yes" if index < args.select else "no"
        item["early_late_mae"] = f"{item['early_late_mae']:.8f}"
        item["adjacent_mae"] = f"{item['adjacent_mae']:.8f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(items[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)
    print(f"Wrote {len(items)} rows; selected top {args.select} by early-late MAE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
