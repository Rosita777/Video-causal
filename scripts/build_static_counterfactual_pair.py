#!/usr/bin/env python3
"""Build an aligned static counterfactual from clean pre-event frames."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def build_reference_frame(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape [time, height, width, 3]")
    if start < 0 or end <= start or end > len(frames):
        raise ValueError("reference range must be inside the video and contain at least one frame")
    return np.median(frames[start:end], axis=0).astype(np.uint8)


def load_video(path: Path) -> tuple[np.ndarray, float]:
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frames = np.stack([frame[:, :, :3] for frame in reader], axis=0)
    finally:
        reader.close()
    return frames, float(metadata.get("fps", 8.0))


def write_static_video(path: Path, frame: np.ndarray, frame_count: int, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    try:
        for _ in range(frame_count):
            writer.append_data(frame)
    finally:
        writer.close()


def make_contact_sheet(frames: np.ndarray, reference: np.ndarray, output_path: Path) -> None:
    indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
    width = 320
    height = round(frames.shape[1] * width / frames.shape[2])
    label_height = 28
    sheet = Image.new("RGB", (width * 3, (height + label_height) * 4), "white")
    draw = ImageDraw.Draw(sheet)

    for position, frame_index in enumerate(indices):
        column = position % 3
        row = position // 3
        frame_image = Image.fromarray(frames[frame_index]).resize((width, height))
        y = row * (height + label_height)
        sheet.paste(frame_image, (column * width, y + label_height))
        draw.text((column * width + 8, y + 7), f"factual frame {frame_index}", fill="black")

    reference_image = Image.fromarray(reference).resize((width, height))
    for position, frame_index in enumerate(indices):
        column = position % 3
        row = position // 3 + 2
        y = row * (height + label_height)
        sheet.paste(reference_image, (column * width, y + label_height))
        draw.text((column * width + 8, y + 7), f"counterfactual frame {frame_index}", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-start", type=int, default=0)
    parser.add_argument("--reference-end", type=int, required=True)
    parser.add_argument("--fps", type=float)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.reference_start < 0:
        parser.error("--reference-start must be non-negative")
    if args.reference_end <= args.reference_start:
        parser.error("--reference-end must be greater than --reference-start")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")

    frames, source_fps = load_video(args.input)
    try:
        reference = build_reference_frame(frames, args.reference_start, args.reference_end)
    except ValueError as exc:
        parser.error(str(exc))

    fps = args.fps or source_fps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "counterfactual_static.mp4"
    reference_path = args.output_dir / "clean_reference.png"
    contact_sheet_path = args.output_dir / "factual_counterfactual_contact_sheet.jpg"

    write_static_video(video_path, reference, len(frames), fps)
    Image.fromarray(reference).save(reference_path)
    make_contact_sheet(frames, reference, contact_sheet_path)

    first_frame_mae = float(np.abs(frames[0].astype(np.int16) - reference.astype(np.int16)).mean())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "median_of_clean_pre_event_frames_repeated_as_static_video",
        "input": str(args.input),
        "output_video": str(video_path),
        "clean_reference": str(reference_path),
        "contact_sheet": str(contact_sheet_path),
        "frame_count": int(len(frames)),
        "fps": fps,
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "reference_start_inclusive": args.reference_start,
        "reference_end_exclusive": args.reference_end,
        "first_frame_reference_mae": first_frame_mae,
    }
    manifest_path = args.output_dir / "pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Static counterfactual written: {video_path}")
    print(f"Contact sheet written: {contact_sheet_path}")
    print(f"Manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
