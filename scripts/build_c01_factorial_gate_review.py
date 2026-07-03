#!/usr/bin/env python3
"""Build blinded human-review artifacts for the C0.1 factorial gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_baseline_review import attach_video_and_strip  # noqa: E402


REVIEW_FIELDS = [
    "review_id",
    "item_index",
    "seed_index",
    "video_path",
    "video_exists",
    "strip_path",
    "strip_exists",
    "target_concept",
    "footprint_definition",
    "source_prompt",
    "target_visible",
    "footprint_visible",
    "scene_structure_preserved",
    "cells_distinguishable",
    "generation_failure",
    "mode_collapse",
    "reviewer_id",
    "notes",
]

KEY_FIELDS = [
    "review_id",
    "pair_id",
    "item_index",
    "seed_index",
    "seed",
    "variant",
    "expected_target_visible",
    "expected_footprint_visible",
    "prompt",
    "video_path",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def review_id_for(item: dict[str, object]) -> str:
    key = "|".join(
        [
            str(item.get("pair_id", "")),
            str(item.get("probe_index", "")),
            str(item.get("seed_index", "")),
            str(item.get("variant", "")),
            str(item.get("seed", "")),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return (
        f"c01_{int(item.get('probe_index', 0)):03d}_"
        f"s{int(item.get('seed_index', 0)):02d}_"
        f"{digest}"
    )


def text_field(item: dict[str, object], *names: str) -> str:
    for name in names:
        value = item.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def item_index_for(item: dict[str, object]) -> str:
    return text_field(item, "probe_index", "item_index")


def base_review_row(item: dict[str, object]) -> dict[str, str]:
    return {
        "review_id": review_id_for(item),
        "item_index": item_index_for(item),
        "seed_index": text_field(item, "seed_index"),
        "video_path": "",
        "video_exists": "false",
        "strip_path": "",
        "strip_exists": "false",
        "target_concept": text_field(item, "target_concept"),
        "footprint_definition": text_field(item, "causal_footprint", "expected_effect"),
        "source_prompt": text_field(item, "source_prompt", "generation_prompt", "prompt"),
        "target_visible": "",
        "footprint_visible": "",
        "scene_structure_preserved": "",
        "cells_distinguishable": "",
        "generation_failure": "",
        "mode_collapse": "",
        "reviewer_id": "",
        "notes": "",
    }


def answer_key_row(item: dict[str, object]) -> dict[str, str]:
    return {
        "review_id": review_id_for(item),
        "pair_id": text_field(item, "pair_id"),
        "item_index": item_index_for(item),
        "seed_index": text_field(item, "seed_index"),
        "seed": text_field(item, "seed"),
        "variant": text_field(item, "variant"),
        "expected_target_visible": text_field(item, "expected_target_visible"),
        "expected_footprint_visible": text_field(item, "expected_footprint_visible"),
        "prompt": text_field(item, "prompt"),
        "video_path": text_field(item, "video_path"),
    }


def only_fields(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in fieldnames}


def build_rows(
    items: Sequence[dict[str, object]],
    *,
    output_dir: Path,
    project_root: Path,
    frame_count: int,
    thumb_width: int,
    thumb_height: int,
    skip_frame_extraction: bool,
    shuffle_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    strip_dir = output_dir / "frame_strips"
    for item in items:
        review_id = review_id_for(item)
        row = base_review_row(item)
        attach_video_and_strip(
            row,
            video_path_text=text_field(item, "video_path"),
            seed=text_field(item, "seed"),
            strip_path=strip_dir / f"{review_id}.jpg",
            project_root=project_root,
            frame_count=frame_count,
            thumb_width=thumb_width,
            thumb_height=thumb_height,
            skip_frame_extraction=skip_frame_extraction,
        )
        row["video_path"] = ""
        review_rows.append(only_fields(row, REVIEW_FIELDS))
        key_rows.append(only_fields(answer_key_row(item), KEY_FIELDS))

    random.Random(shuffle_seed).shuffle(review_rows)
    key_rows.sort(key=lambda row: row["review_id"])
    return review_rows, key_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(only_fields(row, fieldnames) for row in rows)


def write_review_manifest(
    path: Path,
    *,
    generation_manifest: Path,
    blind_review_csv: Path,
    answer_key_csv: Path,
    review_rows: list[dict[str, str]],
    key_rows: list[dict[str, str]],
    shuffle_seed: int,
) -> None:
    strip_count = sum(row["strip_exists"] == "true" for row in review_rows)
    data = {
        "generation_manifest": str(generation_manifest),
        "blind_review_csv": str(blind_review_csv),
        "answer_key_csv": str(answer_key_csv),
        "review_row_count": len(review_rows),
        "answer_key_row_count": len(key_rows),
        "frame_strip_count": strip_count,
        "shuffle_seed": shuffle_seed,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=192)
    parser.add_argument("--thumb-height", type=int, default=128)
    parser.add_argument("--skip-frame-extraction", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=17)
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
    review_rows, key_rows = build_rows(
        items,
        output_dir=args.output_dir,
        project_root=Path.cwd(),
        frame_count=args.frames_per_video,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        skip_frame_extraction=args.skip_frame_extraction,
        shuffle_seed=args.shuffle_seed,
    )

    blind_review_csv = args.output_dir / "blind_review.csv"
    answer_key_csv = args.output_dir / "answer_key.csv"
    write_csv(blind_review_csv, review_rows, REVIEW_FIELDS)
    write_csv(answer_key_csv, key_rows, KEY_FIELDS)
    write_review_manifest(
        args.output_dir / "review_manifest.json",
        generation_manifest=args.generation_manifest,
        blind_review_csv=blind_review_csv,
        answer_key_csv=answer_key_csv,
        review_rows=review_rows,
        key_rows=key_rows,
        shuffle_seed=args.shuffle_seed,
    )

    strips = sum(row["strip_exists"] == "true" for row in review_rows)
    print(
        f"Wrote {len(review_rows)} C0.1 blinded review rows to {blind_review_csv} "
        f"({strips} frame strips); answer key at {answer_key_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
