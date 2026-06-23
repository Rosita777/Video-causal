#!/usr/bin/env python3
"""Build grouped review artifacts for baseline erasure outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_BASELINES = ["negative_prompt", "safree_cogvideox", "videoeraser", "t2vunlearning"]
BASELINE_LABELS = {
    "clean_reference": "Clean reference",
    "negative_prompt": "Negative prompt",
    "safree_cogvideox": "SAFREE-CogVideoX",
    "videoeraser": "VideoEraser local",
    "t2vunlearning": "T2V proxy",
}
REVIEW_FIELDS = [
    "item_index",
    "slice_index",
    "source_index",
    "pair_id",
    "mechanism_type",
    "baseline",
    "baseline_label",
    "video_path",
    "video_exists",
    "strip_path",
    "strip_exists",
    "seed",
    "target_concept",
    "expected_effect",
    "source_prompt",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "usable_for_claim",
    "failure_mode",
    "notes",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def html_ref(path_text: str, *, output_dir: Path, project_root: Path) -> str:
    if not path_text:
        return ""
    path = resolve_path(path_text, project_root)
    try:
        return os.path.relpath(path, output_dir)
    except ValueError:
        return str(path)


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def read_video_frames(path: Path, frame_count: int) -> list[Any]:
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    return [frames[index] for index in evenly_spaced_indices(len(frames), frame_count)]


def write_strip(
    video_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    from PIL import Image, ImageDraw

    frames = read_video_frames(video_path, frame_count)
    if not frames:
        raise ValueError("video has no decodable frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_height = 18
    sheet = Image.new("RGB", (len(frames) * thumb_width, thumb_height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        thumb = frame.resize((thumb_width, thumb_height))
        x = index * thumb_width
        sheet.paste(thumb, (x, label_height))
        draw.text((x + 4, 3), f"t={index + 1}/{len(frames)}", fill=(80, 80, 80))
    sheet.save(output_path, quality=92)


def load_export_items(export_manifest: Path) -> list[dict[str, str]]:
    data = read_json(export_manifest)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{export_manifest}: missing list field 'items'")
    normalized: list[dict[str, str]] = []
    for fallback_index, item in enumerate(items):
        slice_index = str(item.get("slice_index", fallback_index))
        normalized.append(
            {
                "item_index": str(fallback_index),
                "slice_index": slice_index,
                "source_index": str(item.get("source_index", "")),
                "pair_id": str(item.get("pair_id") or item.get("round5_id") or f"item_{fallback_index:03d}"),
                "mechanism_type": str(item.get("mechanism_type", "")),
                "target_concept": str(item.get("target_concept", "")),
                "expected_effect": str(item.get("causal_footprint") or item.get("expected_effect", "")),
                "source_prompt": str(item.get("source_prompt") or item.get("prompt", "")),
                "clean_video_path": str(item.get("clean_video_path", "")),
            }
        )
    return normalized


def baseline_manifest_path(baseline_root: Path, baseline: str, slice_index: str) -> Path:
    return baseline_root / f"{baseline}_shards" / f"prompt_{int(slice_index):03d}" / "generation_manifest.json"


def load_baseline_video(baseline_root: Path, baseline: str, slice_index: str) -> tuple[str, str, str]:
    manifest_path = baseline_manifest_path(baseline_root, baseline, slice_index)
    if not manifest_path.exists():
        return "", "", "missing generation manifest"
    manifest = read_json(manifest_path)
    items = manifest.get("items") or []
    if not items:
        return "", "", "generation manifest has no items"
    item = items[0]
    return str(item.get("video_path", "")), str(item.get("seed", "")), ""


def base_row(item: dict[str, str], baseline: str) -> dict[str, str]:
    return {
        "item_index": item["item_index"],
        "slice_index": item["slice_index"],
        "source_index": item["source_index"],
        "pair_id": item["pair_id"],
        "mechanism_type": item["mechanism_type"],
        "baseline": baseline,
        "baseline_label": BASELINE_LABELS.get(baseline, baseline),
        "video_path": "",
        "video_exists": "false",
        "strip_path": "",
        "strip_exists": "false",
        "seed": "",
        "target_concept": item["target_concept"],
        "expected_effect": item["expected_effect"],
        "source_prompt": item["source_prompt"],
        "target_visible": "",
        "causal_effect_visible": "",
        "causeless_effect": "",
        "video_quality": "",
        "usable_for_claim": "",
        "failure_mode": "",
        "notes": "",
    }


def attach_video_and_strip(
    row: dict[str, str],
    *,
    video_path_text: str,
    seed: str,
    strip_path: Path,
    project_root: Path,
    frame_count: int,
    thumb_width: int,
    thumb_height: int,
    skip_frame_extraction: bool,
) -> None:
    row["video_path"] = video_path_text
    row["seed"] = seed
    if not video_path_text:
        row["video_exists"] = "false"
        return
    video_path = resolve_path(video_path_text, project_root)
    row["video_exists"] = "true" if video_path.exists() else "false"
    if row["video_exists"] != "true" or skip_frame_extraction:
        return
    try:
        write_strip(
            video_path,
            strip_path,
            frame_count=frame_count,
            thumb_width=thumb_width,
            thumb_height=thumb_height,
        )
    except Exception as exc:
        row["notes"] = f"frame extraction failed: {type(exc).__name__}: {exc}"
        return
    row["strip_path"] = display_path(strip_path, project_root)
    row["strip_exists"] = "true"


def build_rows(
    export_items: list[dict[str, str]],
    *,
    baseline_root: Path,
    output_dir: Path,
    baselines: list[str],
    project_root: Path,
    frame_count: int,
    thumb_width: int,
    thumb_height: int,
    skip_frame_extraction: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    strip_dir = output_dir / "frame_strips"
    for item in export_items:
        item_index = int(item["item_index"])
        clean_row = base_row(item, "clean_reference")
        attach_video_and_strip(
            clean_row,
            video_path_text=item["clean_video_path"],
            seed="",
            strip_path=strip_dir / f"{item_index:03d}_clean_reference.jpg",
            project_root=project_root,
            frame_count=frame_count,
            thumb_width=thumb_width,
            thumb_height=thumb_height,
            skip_frame_extraction=skip_frame_extraction,
        )
        rows.append(clean_row)

        for baseline in baselines:
            row = base_row(item, baseline)
            video_path_text, seed, note = load_baseline_video(baseline_root, baseline, item["slice_index"])
            row["notes"] = note
            attach_video_and_strip(
                row,
                video_path_text=video_path_text,
                seed=seed,
                strip_path=strip_dir / f"{item_index:03d}_{baseline}.jpg",
                project_root=project_root,
                frame_count=frame_count,
                thumb_width=thumb_width,
                thumb_height=thumb_height,
                skip_frame_extraction=skip_frame_extraction,
            )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row_media_html(row: dict[str, str], *, output_dir: Path, project_root: Path) -> str:
    video = ""
    if row["video_path"]:
        ref = html.escape(html_ref(row["video_path"], output_dir=output_dir, project_root=project_root))
        video = f"<a href='{ref}' target='_blank'>{html.escape(Path(row['video_path']).name)}</a>"
    else:
        video = "<span class='missing'>missing video</span>"
    if row["strip_path"]:
        strip_ref = html.escape(html_ref(row["strip_path"], output_dir=output_dir, project_root=project_root))
        strip = f"<a href='{strip_ref}' target='_blank'><img src='{strip_ref}' alt='frame strip'></a>"
    else:
        strip = "<span class='muted'>no frame strip</span>"
    note = f"<div class='note'>{html.escape(row['notes'])}</div>" if row["notes"] else ""
    exists = "ok" if row["video_exists"] == "true" else "missing"
    return (
        f"<div class='baseline-cell {exists}'>"
        f"<div class='baseline-name'>{html.escape(row['baseline_label'])}</div>"
        f"<div>{video}</div>"
        f"{strip}"
        f"{note}"
        "</div>"
    )


def write_html(path: Path, rows: list[dict[str, str]], *, baselines: list[str], project_root: Path) -> None:
    output_dir = path.parent
    by_item: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_item.setdefault(row["item_index"], []).append(row)
    baseline_order = ["clean_reference", *baselines]
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Baseline Review</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:16px;background:#f8f8f8;color:#202124}",
        "table{border-collapse:collapse;width:100%;background:white}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top}",
        "th{background:#eee;position:sticky;top:0;z-index:1}",
        ".meta{width:260px;font-size:13px;line-height:1.35}",
        ".prompt{color:#555;margin-top:6px}",
        ".baseline-cell{min-width:220px;font-size:12px}",
        ".baseline-name{font-weight:700;margin-bottom:4px}",
        ".baseline-cell img{width:100%;max-width:520px;border:1px solid #ccc;margin-top:4px}",
        ".muted{color:#777}",
        ".missing{color:#9a3412;font-weight:700}",
        ".note{color:#9a3412;margin-top:4px}",
        "code{font-size:12px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Round Baseline Review</h1>",
        f"<p class='muted'>{len(by_item)} prompts, {len(rows)} rows. Labels are blank for manual review.</p>",
        "<table>",
        "<tr><th>prompt</th>" + "".join(f"<th>{html.escape(BASELINE_LABELS.get(b, b))}</th>" for b in baseline_order) + "</tr>",
    ]
    for item_index in sorted(by_item, key=lambda value: int(value)):
        item_rows = {row["baseline"]: row for row in by_item[item_index]}
        first = by_item[item_index][0]
        meta = (
            f"<td class='meta'><b>{html.escape(item_index)}. <code>{html.escape(first['pair_id'])}</code></b>"
            f"<br>{html.escape(first['mechanism_type'])}"
            f"<br><b>target:</b> {html.escape(first['target_concept'])}"
            f"<br><b>effect:</b> {html.escape(first['expected_effect'])}"
            f"<div class='prompt'>{html.escape(first['source_prompt'])}</div></td>"
        )
        cells = []
        for baseline in baseline_order:
            row = item_rows.get(baseline)
            cell_html = (
                row_media_html(row, output_dir=output_dir, project_root=project_root)
                if row
                else "<span class='missing'>missing row</span>"
            )
            cells.append(f"<td>{cell_html}</td>")
        lines.append("<tr>" + meta + "".join(cells) + "</tr>")
    lines.extend(["</table>", "</body>", "</html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--thumb-height", type=int, default=128)
    parser.add_argument("--skip-frame-extraction", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path.cwd()
    try:
        export_items = load_export_items(args.export_manifest)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        export_items,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        baselines=args.baselines,
        project_root=project_root,
        frame_count=args.frames_per_video,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        skip_frame_extraction=args.skip_frame_extraction,
    )
    csv_path = args.output_dir / "baseline_review.csv"
    html_path = args.output_dir / "baseline_gallery.html"
    write_csv(csv_path, rows)
    write_html(html_path, rows, baselines=args.baselines, project_root=project_root)
    existing = sum(row["video_exists"] == "true" for row in rows)
    strips = sum(row["strip_exists"] == "true" for row in rows)
    print(f"Wrote {len(rows)} rows to {csv_path} ({existing} videos, {strips} frame strips)")
    print(f"Wrote gallery to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
