#!/usr/bin/env python3
"""Build VLM-ready contact sheets and input CSV rows from calibration gold."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


INPUT_FIELDS = [
    "output_id",
    "item_id",
    "source_name",
    "pair_id",
    "mechanism_type",
    "baseline",
    "video_path",
    "video_exists",
    "sheet_path",
    "sheet_exists",
    "sheet_error",
    "seed",
    "target_concept",
    "expected_effect",
    "source_prompt",
    "human_label",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "usable_for_claim",
    "failure_mode",
    "notes",
]

REQUIRED_GOLD_FIELDS = {"item_id", "baseline", "video_path", "target_concept", "expected_effect"}


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug[:180] or "output"


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def read_gold(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(REQUIRED_GOLD_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def read_video_frames(path: Path, frame_count: int) -> list[Any]:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    return [frames[index] for index in evenly_spaced_indices(len(frames), frame_count)]


def write_contact_sheet(
    frames: list[Any],
    output_path: Path,
    *,
    thumb_width: int,
    thumb_height: int,
) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (len(frames) * thumb_width, thumb_height), "white")
    for index, frame in enumerate(frames):
        thumb = frame.resize((thumb_width, thumb_height))
        sheet.paste(thumb, (index * thumb_width, 0))
    sheet.save(output_path, quality=92)


def output_id_for(row: dict[str, str]) -> str:
    if row.get("output_id"):
        return row["output_id"]
    return f"{row.get('item_id', '')}::{row.get('baseline', '')}"


def build_input_row(
    row: dict[str, str],
    *,
    project_root: Path,
    sheet_dir: Path,
    frames_per_video: int,
    thumb_width: int,
    thumb_height: int,
) -> dict[str, str]:
    output_id = output_id_for(row)
    video_path = resolve_path(row.get("video_path", ""), project_root)
    base = {field: row.get(field, "") for field in INPUT_FIELDS}
    base["output_id"] = output_id
    base["video_path"] = row.get("video_path", "")

    if not video_path.exists():
        base.update({"video_exists": "false", "sheet_path": "", "sheet_exists": "false", "sheet_error": "missing video"})
        return base

    sheet_path = sheet_dir / f"{safe_slug(output_id)}.jpg"
    try:
        frames = read_video_frames(video_path, frames_per_video)
        if not frames:
            raise ValueError("video has no decodable frames")
        write_contact_sheet(frames, sheet_path, thumb_width=thumb_width, thumb_height=thumb_height)
    except Exception as exc:  # keep the row auditable rather than dropping it
        base.update(
            {
                "video_exists": "true",
                "sheet_path": "",
                "sheet_exists": "false",
                "sheet_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return base

    base.update(
        {
            "video_exists": "true",
            "sheet_path": str(sheet_path),
            "sheet_exists": "true",
            "sheet_error": "",
        }
    )
    return base


def write_inputs(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--thumb-height", type=int, default=128)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        gold_rows = read_gold(args.gold)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    project_root = Path.cwd()
    input_rows = [
        build_input_row(
            row,
            project_root=project_root,
            sheet_dir=args.sheet_dir,
            frames_per_video=args.frames_per_video,
            thumb_width=args.thumb_width,
            thumb_height=args.thumb_height,
        )
        for row in gold_rows
    ]
    write_inputs(input_rows, args.output)
    sheet_count = sum(row["sheet_exists"] == "true" for row in input_rows)
    print(f"Wrote {len(input_rows)} VLM input rows to {args.output} ({sheet_count} sheets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
