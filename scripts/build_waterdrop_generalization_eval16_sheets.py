#!/usr/bin/env python3
"""Build base/scale-0.75/scale-1.0 frame sheets for eval16."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


FRAMES = [0, 8, 16, 24, 32, 40, 48]


def load(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.stack([frame[:, :, :3] for frame in reader])
    finally:
        reader.close()


def one(root: Path, index: str) -> Path:
    matches = sorted(root.glob(f"{index}_*.mp4"))
    if len(matches) != 1:
        raise ValueError(f"Expected one video for {index} in {root}, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/waterdrop_generalization_eval16.csv"))
    parser.add_argument("--scale075", type=Path, default=Path("outputs/waterdrop_generalization_v2_dual_traj_scale075_eval16/videos"))
    parser.add_argument("--scale100", type=Path, default=Path("outputs/waterdrop_generalization_v2_dual_traj_scale100_eval16/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/waterdrop_generalization_v2_eval16_sheets"))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        paths = [
            ("base", Path(case["factual_video"])),
            ("scale 0.75", one(args.scale075, case["eval_index"])),
            ("scale 1.00", one(args.scale100, case["eval_index"])),
        ]
        videos = [(label, load(path)) for label, path in paths]
        cell_w = 208
        cell_h = round(videos[0][1].shape[1] * cell_w / videos[0][1].shape[2])
        label_w = 90
        header_h = 48
        sheet = Image.new("RGB", (label_w + cell_w * len(FRAMES), header_h + cell_h * 3), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 7), f"{case['eval_index']} | {case['receiver']} | {case['footprint_family']}", fill="black")
        for column, frame_index in enumerate(FRAMES):
            draw.text((label_w + column * cell_w + 5, 29), f"frame {frame_index}", fill="black")
        for row, (label, frames) in enumerate(videos):
            y = header_h + row * cell_h
            draw.text((5, y + 7), label, fill="black")
            for column, frame_index in enumerate(FRAMES):
                frame = Image.fromarray(frames[frame_index]).resize((cell_w, cell_h))
                sheet.paste(frame, (label_w + column * cell_w, y))
        sheet.save(args.output_dir / f"{case['eval_index']}_{case['sample_id']}.jpg", quality=92)
    print(f"Wrote {len(cases)} sheets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
