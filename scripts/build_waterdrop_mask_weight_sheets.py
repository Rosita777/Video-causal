#!/usr/bin/env python3
"""Build side-by-side frame sheets for the mask-weight quick evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compare_waterdrop_mask_weights import CASES, METHODS, one_match


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_id, _, _, _, index in CASES:
        videos = []
        for method, method_dir in METHODS.items():
            path = one_match(root / method_dir, f"{index:03d}_*.mp4")
            videos.append((method, load_video(path)))

        cell_width = 208
        cell_height = round(videos[0][1].shape[1] * cell_width / videos[0][1].shape[2])
        label_width = 110
        header_height = 44
        sheet = Image.new(
            "RGB",
            (label_width + cell_width * len(FRAME_INDICES), header_height + cell_height * len(videos)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), case_id, fill="black")
        for column, frame_index in enumerate(FRAME_INDICES):
            draw.text((label_width + column * cell_width + 6, 25), f"frame {frame_index}", fill="black")
        for row, (method, frames) in enumerate(videos):
            y = header_height + row * cell_height
            draw.text((6, y + 8), method, fill="black")
            for column, frame_index in enumerate(FRAME_INDICES):
                image = Image.fromarray(frames[frame_index]).resize((cell_width, cell_height))
                sheet.paste(image, (label_width + column * cell_width, y))
        sheet.save(output_dir / f"{case_id}.jpg", quality=92)


if __name__ == "__main__":
    main()
