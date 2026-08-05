#!/usr/bin/env python3
"""Build frozen-base/plain/dual frame sheets for held-out waterdrop eval20."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from evaluate_waterdrop_expanded_eval20 import one_match


FRAME_INDICES = [0, 8, 16, 24, 32, 40, 48]


def load_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.stack([frame[:, :, :3] for frame in reader])
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/waterdrop_dual_traj_eval20.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    base_dir = args.base_dir if args.base_dir.is_absolute() else root / args.base_dir
    adapter_dir = args.adapter_dir if args.adapter_dir.is_absolute() else root / args.adapter_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))

    for case in cases:
        paths = [
            ("frozen_base", one_match(base_dir, f"{case['eval_index']}_*.mp4")),
            ("method_v1", one_match(adapter_dir, f"{case['eval_index']}_*.mp4")),
        ]
        videos = [(label, load_video(path)) for label, path in paths]
        cell_width = 208
        cell_height = round(videos[0][1].shape[1] * cell_width / videos[0][1].shape[2])
        label_width = 100
        header_height = 48
        sheet = Image.new(
            "RGB",
            (label_width + cell_width * len(FRAME_INDICES), header_height + cell_height * len(videos)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        title = f"{case['eval_index']} | {case['condition']} | {case['family']} | {case['receiver']}"
        draw.text((6, 7), title[:170], fill="black")
        for column, frame_index in enumerate(FRAME_INDICES):
            draw.text((label_width + column * cell_width + 5, 29), f"frame {frame_index}", fill="black")
        for row, (label, frames) in enumerate(videos):
            y = header_height + row * cell_height
            draw.text((5, y + 7), label, fill="black")
            for column, frame_index in enumerate(FRAME_INDICES):
                image = Image.fromarray(frames[frame_index]).resize((cell_width, cell_height))
                sheet.paste(image, (label_width + column * cell_width, y))
        sheet.save(output_dir / f"{case['eval_index']}_{case['scene_id']}.jpg", quality=92)


if __name__ == "__main__":
    main()
