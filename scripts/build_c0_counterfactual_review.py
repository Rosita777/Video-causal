#!/usr/bin/env python3
"""Build review rows for Method C0 counterfactual-grid outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_baseline_review import attach_video_and_strip  # noqa: E402


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
    "expected_target_visible",
    "expected_footprint_visible",
    "variant_role",
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


def base_row(item: dict[str, object], baseline: str) -> dict[str, str]:
    return {
        "item_index": str(item.get("probe_index", "")),
        "slice_index": str(item.get("slice_index", "")),
        "source_index": str(item.get("source_index", "")),
        "pair_id": str(item.get("pair_id", "")),
        "mechanism_type": str(item.get("mechanism_type", "")),
        "baseline": baseline,
        "baseline_label": str(item.get("variant_label", baseline)),
        "video_path": "",
        "video_exists": "false",
        "strip_path": "",
        "strip_exists": "false",
        "seed": str(item.get("seed", "")),
        "target_concept": str(item.get("target_concept", "")),
        "expected_effect": str(item.get("causal_footprint", "")),
        "source_prompt": str(item.get("prompt", "")),
        "expected_target_visible": str(item.get("expected_target_visible", "")),
        "expected_footprint_visible": str(item.get("expected_footprint_visible", "")),
        "variant_role": str(item.get("variant_role") or item.get("variant", "")),
        "target_visible": "",
        "causal_effect_visible": "",
        "causeless_effect": "",
        "video_quality": "",
        "usable_for_claim": "",
        "failure_mode": "",
        "notes": "",
    }


def grouped_items(items: Sequence[dict[str, object]]) -> list[list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for item in items:
        key = str(item.get("probe_index", item.get("pair_id", "")))
        groups.setdefault(key, []).append(item)
    return [groups[key] for key in sorted(groups, key=lambda value: int(value) if value.isdigit() else value)]


def build_rows(
    items: Sequence[dict[str, object]],
    *,
    output_dir: Path,
    project_root: Path,
    frame_count: int,
    thumb_width: int,
    thumb_height: int,
    skip_frame_extraction: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    strip_dir = output_dir / "frame_strips"
    for group in grouped_items(items):
        by_variant = {str(item.get("variant", "")): item for item in group}
        original = by_variant.get("original", group[0])
        clean_row = base_row(original, "clean_reference")
        clean_row["baseline_label"] = "Clean reference"
        clean_row["expected_target_visible"] = "yes"
        clean_row["expected_footprint_visible"] = "yes"
        attach_video_and_strip(
            clean_row,
            video_path_text=str(original.get("video_path", "")),
            seed=str(original.get("seed", "")),
            strip_path=strip_dir / f"{int(clean_row['item_index']):03d}_clean_reference.jpg",
            project_root=project_root,
            frame_count=frame_count,
            thumb_width=thumb_width,
            thumb_height=thumb_height,
            skip_frame_extraction=skip_frame_extraction,
        )
        rows.append(clean_row)

        for item in group:
            variant = str(item.get("variant", ""))
            row = base_row(item, variant)
            attach_video_and_strip(
                row,
                video_path_text=str(item.get("video_path", "")),
                seed=str(item.get("seed", "")),
                strip_path=strip_dir / f"{int(row['item_index']):03d}_{variant}.jpg",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--thumb-height", type=int, default=128)
    parser.add_argument("--skip-frame-extraction", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames_per_video <= 0:
        parser.error("--frames-per-video must be positive")
    if args.thumb_width <= 0 or args.thumb_height <= 0:
        parser.error("--thumb-width and --thumb-height must be positive")
    manifest = read_json(args.generation_manifest)
    items = manifest.get("items")
    if not isinstance(items, list):
        parser.exit(2, f"{args.generation_manifest}: missing list field 'items'\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        items,
        output_dir=args.output_dir,
        project_root=Path.cwd(),
        frame_count=args.frames_per_video,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        skip_frame_extraction=args.skip_frame_extraction,
    )
    out = args.output_dir / "review.csv"
    write_csv(out, rows)
    strips = sum(row["strip_exists"] == "true" for row in rows)
    print(f"Wrote {len(rows)} C0 review rows to {out} ({strips} frame strips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
