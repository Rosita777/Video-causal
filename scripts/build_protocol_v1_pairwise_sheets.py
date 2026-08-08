#!/usr/bin/env python3
"""Build Original/Ours frame sheets from Protocol v1 pairwise metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


FRAME_INDICES = [0, 8, 16, 24, 32, 40, 48]


def load_video(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        return np.stack([frame[:, :, :3] for frame in reader])
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mechanism", action="append", default=[])
    args = parser.parse_args()
    rows = list(csv.DictReader(args.metrics.open(newline="", encoding="utf-8")))
    if args.mechanism:
        rows = [row for row in rows if row["mechanism"] in set(args.mechanism)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        videos = [
            ("Original", load_video(Path(row["original_video"]))),
            ("Ours", load_video(Path(row["ours_video"]))),
        ]
        cell_width = 208
        cell_height = round(videos[0][1].shape[1] * cell_width / videos[0][1].shape[2])
        label_width = 92
        header_height = 52
        sheet = Image.new("RGB", (label_width + cell_width * len(FRAME_INDICES), header_height + 2 * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        title = f"{row['sample_id']} | {row['generalization_group']} | suppression {row['footprint_suppression_percent']}%"
        draw.text((6, 6), title, fill="black")
        for column, frame_index in enumerate(FRAME_INDICES):
            draw.text((label_width + column * cell_width + 5, 31), f"frame {frame_index}", fill="black")
        for video_row, (label, frames) in enumerate(videos):
            y = header_height + video_row * cell_height
            draw.text((6, y + 8), label, fill="black")
            for column, frame_index in enumerate(FRAME_INDICES):
                frame = Image.fromarray(frames[frame_index]).resize((cell_width, cell_height))
                sheet.paste(frame, (label_width + column * cell_width, y))
        destination = args.output_dir / row["mechanism"] / f"{row['sample_id']}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, quality=90)
    print(f"Wrote {len(rows)} sheets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
