#!/usr/bin/env python3
"""Build contact sheets and blank annotation CSVs for baseline video review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ANNOTATION_FIELDS = [
    "prompt_id",
    "baseline",
    "video_path",
    "prompt",
    "target_concept",
    "expected_effect",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "usable_for_claim",
    "notes",
]


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count <= 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _read_video_frames(path: Path, frames_per_video: int):
    import av
    from PIL import Image

    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    return [frames[i] for i in evenly_spaced_indices(len(frames), frames_per_video)]


def make_contact_sheet(
    videos: list[tuple[str, Path]],
    output_path: Path,
    *,
    frames_per_video: int = 7,
    thumb_width: int = 216,
    thumb_height: int = 144,
) -> None:
    from PIL import Image, ImageDraw

    rows: list[tuple[str, list[Image.Image]]] = []
    for label, path in videos:
        thumbs = []
        for frame in _read_video_frames(path, frames_per_video):
            thumbs.append(frame.resize((thumb_width, thumb_height)))
        rows.append((label, thumbs))

    label_width = max(140, max((len(label) for label, _ in rows), default=0) * 8 + 16)
    width = label_width + frames_per_video * thumb_width
    height = max(1, len(rows)) * thumb_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, (label, thumbs) in enumerate(rows):
        y = row_idx * thumb_height
        draw.text((8, y + 8), label, fill="black")
        for col_idx, thumb in enumerate(thumbs):
            sheet.paste(thumb, (label_width + col_idx * thumb_width, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_video_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--video must be formatted as label=path")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--video must be formatted as label=path")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-concept", required=True)
    parser.add_argument("--expected-effect", required=True)
    parser.add_argument("--video", action="append", type=parse_video_arg, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=7)
    parser.add_argument("--thumb-width", type=int, default=216)
    parser.add_argument("--thumb-height", type=int, default=144)
    args = parser.parse_args()

    make_contact_sheet(
        args.video,
        args.contact_sheet,
        frames_per_video=args.frames_per_video,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )

    rows = []
    for label, path in args.video:
        rows.append(
            {
                "prompt_id": args.prompt_id,
                "baseline": label,
                "video_path": str(path),
                "prompt": args.prompt,
                "target_concept": args.target_concept,
                "expected_effect": args.expected_effect,
                "target_visible": "",
                "causal_effect_visible": "",
                "causeless_effect": "",
                "video_quality": "",
                "usable_for_claim": "",
                "notes": "",
            }
        )
    args.annotation_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.annotation_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.contact_sheet} and {args.annotation_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
