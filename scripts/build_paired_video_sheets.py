#!/usr/bin/env python3
"""Build frame contact sheets for two aligned video directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_pilot import parse_prompt_file  # noqa: E402


FRAME_INDICES = [0, 8, 16, 24, 32, 40, 48]


def one_match(root: Path, index: int) -> Path:
    matches = sorted(root.glob(f"{index:03d}_*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one video for index {index} in {root}, found {len(matches)}")
    return matches[0]


def load_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.stack([frame[:, :, :3] for frame in reader])
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prompts = parse_prompt_file(args.prompts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        videos = [
            ("frozen_base", load_video(one_match(args.base_dir, index))),
            (args.candidate_label, load_video(one_match(args.candidate_dir, index))),
        ]
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
        draw.text((6, 7), f"{index:03d} | {prompt['target_concept']} | {prompt['expected_effect']}", fill="black")
        for column, frame_index in enumerate(FRAME_INDICES):
            draw.text((label_width + column * cell_width + 5, 29), f"frame {frame_index}", fill="black")
        for row, (label, frames) in enumerate(videos):
            y = header_height + row * cell_height
            draw.text((5, y + 7), label, fill="black")
            for column, frame_index in enumerate(FRAME_INDICES):
                image = Image.fromarray(frames[frame_index]).resize((cell_width, cell_height))
                sheet.paste(image, (label_width + column * cell_width, y))
        sheet.save(args.output_dir / f"{index:03d}.jpg", quality=92)


if __name__ == "__main__":
    main()
