#!/usr/bin/env python3
"""Compute paired automatic metrics for Protocol v1 Wan videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav").astype(np.float32) / 255.0
    if frames.ndim != 4 or len(frames) < 49:
        raise ValueError(f"Unexpected video shape {frames.shape}: {path}")
    return frames[:49]


def index_videos(root: Path, mechanism: str) -> dict[str, Path]:
    manifest = json.loads((root / mechanism / "generation_manifest.json").read_text())
    return {str(item["prompt"]): Path(str(item["video_path"])) for item in manifest["items"]}


def mean_change(frames: np.ndarray, start: int, end: int) -> float:
    return float(np.abs(frames[start:end].mean(axis=0) - frames[:8].mean(axis=0)).mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    by_mech = {m: index_videos(args.original_root, m) for m in sorted({r["mechanism"] for r in rows})}
    ours_by_mech = {m: index_videos(args.ours_root, m) for m in by_mech}
    output = []
    for row in rows:
        mechanism = row["mechanism"]
        prompt = row["prompt"]
        original_path = by_mech[mechanism][prompt]
        ours_path = ours_by_mech[mechanism][prompt]
        original = load_video(original_path)
        ours = load_video(ours_path)
        original_post = mean_change(original, 32, 49)
        ours_post = mean_change(ours, 32, 49)
        suppression = 100.0 * (1.0 - ours_post / original_post) if original_post else 0.0
        output.append({
            **row,
            "original_video": str(original_path),
            "ours_video": str(ours_path),
            "original_post_change_mae": f"{original_post:.8f}",
            "ours_post_change_mae": f"{ours_post:.8f}",
            "footprint_suppression_percent": f"{suppression:.2f}",
            "early_divergence_mae": f"{np.abs(ours[:17] - original[:17]).mean():.8f}",
            "ours_late_frame_mae": f"{np.abs(ours[32:49] - original[32:49]).mean():.8f}",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with (args.output_dir / "pairwise_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(output)

    groups = {"ALL": output}
    for field in ("mechanism", "generalization_group", "source_seen", "receiver_seen"):
        groups.update({f"{field}={value}": [r for r in output if r[field] == value] for value in sorted({r[field] for r in output})})
    summary = []
    for name, items in groups.items():
        summary.append({
            "group": name,
            "count": len(items),
            "mean_footprint_suppression_percent": f"{np.mean([float(r['footprint_suppression_percent']) for r in items]):.2f}",
            "mean_early_divergence_mae": f"{np.mean([float(r['early_divergence_mae']) for r in items]):.8f}",
            "mean_ours_late_frame_mae": f"{np.mean([float(r['ours_late_frame_mae']) for r in items]):.8f}",
        })
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(f"Wrote pairwise metrics for {len(output)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
