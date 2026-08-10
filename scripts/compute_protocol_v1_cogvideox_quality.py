#!/usr/bin/env python3
"""Compute non-semantic quality diagnostics for Protocol v1 CogVideoX videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np


BASELINE_ROOTS = {
    "original": "eval_cogvideox_original",
    "negative_prompt": "eval_cogvideox_negative_prompt",
    "t2vunlearning_adapted": "eval_t2vunlearning_adapted_ckpt100",
    "videoeraser_official": "videoeraser_official",
}


def load_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(frames) != 49:
        raise ValueError(f"Expected 49 frames, found {len(frames)}: {path}")
    return np.stack(frames).astype(np.float32)


def index_videos(root: Path, mechanism: str) -> dict[str, Path]:
    manifest = json.loads((root / mechanism / "generation_manifest.json").read_text())
    return {item["sample_id"]: Path(item["video_path"]) for item in manifest["items"]}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--t2v-root", default=BASELINE_ROOTS["t2vunlearning_adapted"]
    )
    args = parser.parse_args()
    BASELINE_ROOTS["t2vunlearning_adapted"] = args.t2v_root

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    mechanisms = sorted({row["mechanism"] for row in manifest_rows})
    indexed = {
        baseline: {
            mechanism: index_videos(args.protocol_root / root, mechanism)
            for mechanism in mechanisms
        }
        for baseline, root in BASELINE_ROOTS.items()
    }

    originals = {}
    rows = []
    for manifest_row in manifest_rows:
        sample_id = manifest_row["sample_id"]
        mechanism = manifest_row["mechanism"]
        original = load_video(indexed["original"][mechanism][sample_id])
        originals[sample_id] = original
        for baseline in BASELINE_ROOTS:
            video_path = indexed[baseline][mechanism][sample_id]
            frames = original if baseline == "original" else load_video(video_path)
            rows.append(
                {
                    "sample_id": sample_id,
                    "mechanism": mechanism,
                    "generalization_group": manifest_row["generalization_group"],
                    "baseline": baseline,
                    "mean_rgb": f"{frames.mean():.4f}",
                    "spatial_std": f"{frames.std():.4f}",
                    "dark_pixel_fraction": f"{(frames < 16).mean():.6f}",
                    "saturated_pixel_fraction": f"{(frames > 240).mean():.6f}",
                    "temporal_frame_mae": f"{np.abs(np.diff(frames, axis=0)).mean():.4f}",
                    "early_mae_vs_original": f"{np.abs(frames[:17] - original[:17]).mean():.4f}",
                    "late_mae_vs_original": f"{np.abs(frames[32:] - original[32:]).mean():.4f}",
                    "quality_collapse": str(
                        bool(frames.mean() < 30 or (frames < 16).mean() > 0.5)
                    ).lower(),
                    "video_path": str(video_path),
                }
            )

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["baseline"]), str(row["mechanism"]))].append(row)
    summary = []
    for (baseline, mechanism), items in sorted(groups.items()):
        summary.append(
            {
                "baseline": baseline,
                "mechanism": mechanism,
                "count": len(items),
                "quality_collapse_count": sum(
                    row["quality_collapse"] == "true" for row in items
                ),
                "mean_rgb": f"{np.mean([float(row['mean_rgb']) for row in items]):.4f}",
                "mean_temporal_frame_mae": f"{np.mean([float(row['temporal_frame_mae']) for row in items]):.4f}",
                "mean_early_mae_vs_original": f"{np.mean([float(row['early_mae_vs_original']) for row in items]):.4f}",
                "mean_late_mae_vs_original": f"{np.mean([float(row['late_mae_vs_original']) for row in items]):.4f}",
            }
        )
    write_csv(args.output_dir / "per_video_quality.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    print(f"Wrote {len(rows)} video diagnostics and {len(summary)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
