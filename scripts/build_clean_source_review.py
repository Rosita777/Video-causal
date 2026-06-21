#!/usr/bin/env python3
"""Build review artifacts for clean-source screening manifests."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import re


CLEAN_FIELDS = [
    "prompt_id",
    "pair_id",
    "baseline",
    "mechanism_type",
    "video_path",
    "prompt",
    "target_concept",
    "expected_effect",
    "target_visible",
    "effect_visible",
    "temporal_order_clear",
    "effect_depends_on_target",
    "video_quality",
    "clean_source_valid",
    "notes",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        rows = list(reader)
    normalized = []
    for row in rows:
        item = dict(row)
        if "pair_id" not in item and "round4_id" in item:
            item["pair_id"] = item["round4_id"]
        normalized.append(item)
    return normalized


def safe_stem(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-")
    return cleaned[:80] or "case"


def resolve_media_ref(path_text: str, *, project_root: Path, output_dir: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root / path
    try:
        return os.path.relpath(path.resolve(), output_dir.resolve())
    except OSError:
        return path_text


def metadata_for_index(metadata_items: list[dict], item: dict, row_index: int) -> dict:
    source_index = item.get("source_prompt_index", item.get("index", row_index))
    try:
        source_index = int(source_index)
    except (TypeError, ValueError):
        source_index = row_index
    if 0 <= source_index < len(metadata_items):
        return metadata_items[source_index]
    return {}


def normalize_rows(data: dict, metadata_items: list[dict]) -> list[dict]:
    baseline = data.get("baseline") or "clean"
    items = data.get("items") or data.get("outputs") or []
    rows = []
    for idx, item in enumerate(items):
        metadata = metadata_for_index(metadata_items, item, idx)
        prompt_id = item.get("prompt_id") or item.get("pair_job_id") or f"case_{idx:03d}"
        expected_effect = (
            item.get("expected_effect")
            or item.get("causal_footprint")
            or metadata.get("causal_footprint")
            or ""
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "pair_id": metadata.get("pair_id", item.get("pair_id", "")),
                "baseline": baseline,
                "mechanism_type": metadata.get("mechanism_type", item.get("mechanism_type", "")),
                "video_path": item.get("video_path") or item.get("output_path", ""),
                "prompt": item.get("prompt", ""),
                "target_concept": item.get("target_concept") or metadata.get("target_concept", ""),
                "expected_effect": expected_effect,
                "target_visible": "",
                "effect_visible": "",
                "temporal_order_clear": "",
                "effect_depends_on_target": "",
                "video_quality": "",
                "clean_source_valid": "",
                "notes": "",
            }
        )
    return rows


def baseline_label(baseline: str) -> str:
    return "Clean reference" if baseline == "clean" else baseline


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    output_path = output_dir / "clean_source_screening.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_frame_strip(
    video_path: str,
    *,
    output_path: Path,
    project_root: Path,
    frames_per_video: int,
    thumb_width: int,
    thumb_height: int,
) -> Path | None:
    if not video_path:
        return None
    source_path = Path(video_path)
    if not source_path.is_absolute():
        source_path = project_root / source_path
    if not source_path.exists():
        return None

    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(source_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        positions = [
            round(i * (frame_count - 1) / max(frames_per_video - 1, 1))
            for i in range(frames_per_video)
        ]
        thumbs = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((thumb_width, thumb_height))
            canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
            offset = ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2)
            canvas.paste(image, offset)
            thumbs.append(canvas)
        if not thumbs:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strip = Image.new("RGB", (thumb_width * len(thumbs), thumb_height), "white")
        for idx, thumb in enumerate(thumbs):
            strip.paste(thumb, (idx * thumb_width, 0))
        strip.save(output_path, quality=92)
        return output_path
    finally:
        cap.release()


def write_html(
    rows: list[dict],
    *,
    output_dir: Path,
    project_root: Path,
    frames_per_video: int,
    thumb_width: int,
    thumb_height: int,
    skip_frame_extraction: bool,
) -> Path:
    strip_dir = output_dir / "frame_strips"
    rendered_rows = []
    for idx, row in enumerate(rows):
        strip_path = None
        if not skip_frame_extraction:
            strip_path = build_frame_strip(
                row["video_path"],
                output_path=strip_dir / f"{idx:03d}_{safe_stem(row['prompt_id'])}.jpg",
                project_root=project_root,
                frames_per_video=frames_per_video,
                thumb_width=thumb_width,
                thumb_height=thumb_height,
            )
        video_ref = resolve_media_ref(row["video_path"], project_root=project_root, output_dir=output_dir)
        strip_ref = (
            resolve_media_ref(str(strip_path), project_root=project_root, output_dir=output_dir)
            if strip_path
            else ""
        )
        preview_html = (
            f"<img class='strip' src='{html.escape(strip_ref)}' alt='frame preview'>"
            if strip_ref
            else "<span class='muted'>Frame preview skipped or unavailable.</span>"
        )
        video_html = (
            f"<a href='{html.escape(video_ref)}' target='_blank'>open video</a>"
            if video_ref
            else "<span class='muted'>missing video path</span>"
        )
        rendered_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(row['pair_id'] or row['prompt_id'])}</code>"
            f"<br><span class='muted'>{html.escape(row['mechanism_type'])}</span></td>"
            f"<td><span class='badge'>{html.escape(baseline_label(row['baseline']))}</span></td>"
            f"<td><b>target:</b> {html.escape(row['target_concept'])}"
            f"<br><b>causal footprint:</b> {html.escape(row['expected_effect'])}</td>"
            f"<td class='prompt'>{html.escape(row['prompt'])}</td>"
            f"<td>{video_html}</td>"
            f"<td>{preview_html}</td>"
            "</tr>"
        )

    output_path = output_dir / "clean_gallery.html"
    output_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset='utf-8'>",
                "<title>Clean Source Review</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:18px;color:#222}",
                "h1{margin-bottom:4px}.muted{color:#666}.badge{background:#eaf3ff;border:1px solid #9cc7ff;border-radius:999px;padding:2px 8px;white-space:nowrap}",
                "table{border-collapse:collapse;font-size:12px;width:100%;table-layout:fixed}",
                "td,th{border:1px solid #ddd;padding:6px;vertical-align:top}",
                "th{background:#f2f2f2;text-align:left}",
                ".prompt{width:30%;line-height:1.35}.strip{max-width:100%;border:1px solid #ccc}",
                "code{white-space:normal}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>Clean Source Review</h1>",
                "<p class='muted'>This page screens source videos before applying erasure baselines. Rows labeled <b>Clean reference</b> are original generated videos, not erasure outputs.</p>",
                "<table>",
                "<tr><th>#</th><th>pair</th><th>baseline</th><th>target / footprint</th><th>source prompt</th><th>video</th><th>preview</th></tr>",
                *rendered_rows,
                "</table>",
                "</body>",
                "</html>",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path)
    parser.add_argument("--metadata-tsv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--videos-per-sheet", type=int, default=5)
    parser.add_argument("--frames-per-video", type=int, default=9)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--thumb-height", type=int, default=160)
    parser.add_argument("--skip-frame-extraction", action="store_true")
    args = parser.parse_args()
    if args.metadata_manifest and args.metadata_tsv:
        parser.error("use only one of --metadata-manifest or --metadata-tsv")

    data = load_json(args.manifest)
    metadata_items = []
    if args.metadata_manifest:
        metadata_items = load_json(args.metadata_manifest).get("items", [])
    if args.metadata_tsv:
        try:
            metadata_items = load_metadata_tsv(args.metadata_tsv)
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(data, metadata_items)
    csv_path = write_csv(rows, args.output_dir)
    html_path = write_html(
        rows,
        output_dir=args.output_dir,
        project_root=args.project_root,
        frames_per_video=args.frames_per_video,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        skip_frame_extraction=args.skip_frame_extraction,
    )
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote review gallery to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
