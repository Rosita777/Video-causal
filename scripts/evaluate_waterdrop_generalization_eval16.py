#!/usr/bin/env python3
"""Measure adapter change suppression on receiver-held-out waterdrop cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav").astype(np.float32) / 255.0
    if frames.ndim != 4 or len(frames) < 40:
        raise ValueError(f"Unexpected video shape: {path} -> {frames.shape}")
    return frames


def one_video(root: Path, output_dir: Path, index: str) -> Path:
    matches = sorted((root / output_dir).glob(f"{index}_*.mp4"))
    if len(matches) != 1:
        raise ValueError(f"Expected one generated video for index {index}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/waterdrop_generalization_eval16.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    root = Path(".").resolve()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    rows = []
    for case in cases:
        base = load(root / case["factual_video"])
        candidate_path = one_video(root, args.output_dir, case["eval_index"])
        candidate = load(candidate_path)
        base_change = float(np.abs(base[32:49].mean(axis=0) - base[:8].mean(axis=0)).mean())
        candidate_change = float(np.abs(candidate[32:49].mean(axis=0) - candidate[:8].mean(axis=0)).mean())
        suppression = 100.0 * (1.0 - candidate_change / base_change) if base_change else 0.0
        rows.append(
            {
                "eval_index": case["eval_index"],
                "scene_id": case.get("scene_id") or case["pair_id"],
                "receiver": case["receiver"],
                "footprint_family": case["footprint_family"],
                "base_post_change_mae": f"{base_change:.8f}",
                "candidate_post_change_mae": f"{candidate_change:.8f}",
                "post_change_suppression_percent": f"{suppression:.2f}",
                "early_base_mae": f"{np.abs(candidate[:17] - base[:17]).mean():.8f}",
                "video_path": str(candidate_path.relative_to(root)),
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    suppression = np.mean([float(row["post_change_suppression_percent"]) for row in rows])
    early = np.mean([float(row["early_base_mae"]) for row in rows])
    print(f"eval16 count={len(rows)} mean_suppression={suppression:.2f}% mean_early_base_mae={early:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
