#!/usr/bin/env python3
"""Build item-level C0.2 spot-check contact sheets from frame strips."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence


VARIANTS = ["original", "remove_target", "footprint_only", "target_only"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")
    return slug or "item"


def numeric_key(text: str) -> tuple[int, str]:
    try:
        return (int(text), text)
    except ValueError:
        return (10**9, text)


def strip_path_for(row: dict[str, str], frame_strip_dir: Path) -> Path:
    return frame_strip_dir / f"{row['review_id']}.jpg"


def grouped_by_item(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("item_index", "")].append(row)
    return dict(groups)


def label_text(row: dict[str, str]) -> str:
    target = row.get("expected_target_visible", "")
    footprint = row.get("expected_footprint_visible", "")
    return (
        f"s{row.get('seed_index', '')} | {row.get('variant', '')} | "
        f"T:{target} F:{footprint}"
    )


def draw_sheet(
    rows: Sequence[dict[str, str]],
    *,
    frame_strip_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    from PIL import Image, ImageDraw

    row_by_key = {
        (row.get("seed_index", ""), row.get("variant", "")): row
        for row in rows
    }
    seed_indices = sorted({row.get("seed_index", "") for row in rows}, key=numeric_key)
    strip_size = (1, 1)
    opened: dict[tuple[str, str], Image.Image] = {}
    missing: list[str] = []
    for seed_index in seed_indices:
        for variant in VARIANTS:
            row = row_by_key.get((seed_index, variant))
            if row is None:
                missing.append(f"seed={seed_index} variant={variant}")
                continue
            path = strip_path_for(row, frame_strip_dir)
            if not path.exists():
                missing.append(str(path))
                continue
            image = Image.open(path).convert("RGB")
            opened[(seed_index, variant)] = image
            strip_size = (max(strip_size[0], image.width), max(strip_size[1], image.height))

    padding = 10
    label_height = 22
    header_height = 30
    cell_width = strip_size[0] + padding * 2
    cell_height = strip_size[1] + label_height + padding * 2
    width = len(VARIANTS) * cell_width
    height = header_height + max(1, len(seed_indices)) * cell_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    item_index = rows[0].get("item_index", "") if rows else ""
    pair_id = rows[0].get("pair_id", "") if rows else ""
    draw.text((padding, padding), f"C0.2 item {item_index} | {pair_id}", fill=(0, 0, 0))

    for row_index, seed_index in enumerate(seed_indices):
        y = header_height + row_index * cell_height
        for col_index, variant in enumerate(VARIANTS):
            x = col_index * cell_width
            row = row_by_key.get((seed_index, variant), {})
            draw.text((x + padding, y + padding), label_text(row), fill=(0, 0, 0))
            image = opened.get((seed_index, variant))
            if image is None:
                draw.rectangle(
                    [
                        x + padding,
                        y + label_height + padding,
                        x + padding + strip_size[0] - 1,
                        y + label_height + padding + strip_size[1] - 1,
                    ],
                    outline=(180, 0, 0),
                )
                draw.text(
                    (x + padding + 4, y + label_height + padding + 4),
                    "missing strip",
                    fill=(180, 0, 0),
                )
                continue
            sheet.paste(image, (x + padding, y + label_height + padding))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return {
        "sheet_path": str(output_path),
        "item_index": item_index,
        "pair_id": pair_id,
        "row_count": len(rows),
        "seed_indices": seed_indices,
        "variants": VARIANTS,
        "missing_strips": missing,
    }


def build_sheets(
    rows: Sequence[dict[str, str]],
    *,
    frame_strip_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    items = []
    for item_index, item_rows in sorted(grouped_by_item(rows).items(), key=lambda pair: numeric_key(pair[0])):
        sheet_path = output_dir / f"item_{slugify(item_index)}_all_seeds_four_cells.jpg"
        items.append(
            draw_sheet(
                item_rows,
                frame_strip_dir=frame_strip_dir,
                output_path=sheet_path,
            )
        )

    missing_strip_count = sum(len(item["missing_strips"]) for item in items)
    return {
        "item_count": len(items),
        "sheet_count": len(items),
        "missing_strip_count": missing_strip_count,
        "items": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--frame-strip-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = read_csv(args.answer_key)
    if not rows:
        parser.exit(2, f"{args.answer_key}: no answer-key rows\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_sheets(
        rows,
        frame_strip_dir=args.frame_strip_dir,
        output_dir=args.output_dir,
    )
    manifest_path = args.output_dir / "spotcheck_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {manifest['sheet_count']} C0.2 spot-check sheets to {args.output_dir} "
        f"({manifest['missing_strip_count']} missing strips)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
