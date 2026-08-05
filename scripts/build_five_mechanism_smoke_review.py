#!/usr/bin/env python3
"""Build contact sheets and a joint Wan/CogVideoX smoke review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from build_clean_source_review import build_frame_strip
except ModuleNotFoundError:
    from scripts.build_clean_source_review import build_frame_strip


FIELDS = [
    "backbone",
    "candidate_id",
    "mechanism",
    "target_concept",
    "receiver_family",
    "receiver",
    "expected_footprint",
    "prompt",
    "seed",
    "video_path",
    "contact_sheet",
    "target_visible",
    "footprint_visible",
    "temporal_order_clear",
    "video_quality_ok",
    "clean_source_valid",
    "notes",
]


def read_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["prompt"]: row for row in rows}


def read_manifest(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing list field 'items'")
    return items


def build_rows(
    candidates: dict[str, dict[str, str]], manifests: list[tuple[str, Path]]
) -> list[dict[str, str]]:
    rows = []
    for backbone, manifest_path in manifests:
        for item in read_manifest(manifest_path):
            prompt = str(item.get("prompt", ""))
            metadata = candidates.get(prompt)
            if metadata is None:
                raise ValueError(f"{manifest_path}: prompt is absent from candidate CSV: {prompt[:80]}")
            rows.append(
                {
                    "backbone": backbone,
                    "candidate_id": metadata["candidate_id"],
                    "mechanism": metadata["mechanism"],
                    "target_concept": metadata["target_concept"],
                    "receiver_family": metadata["receiver_family"],
                    "receiver": metadata["receiver"],
                    "expected_footprint": metadata["expected_footprint"],
                    "prompt": prompt,
                    "seed": str(item.get("seed", "")),
                    "video_path": str(item.get("video_path", "")),
                    "contact_sheet": "",
                    "target_visible": "",
                    "footprint_visible": "",
                    "temporal_order_clear": "",
                    "video_quality_ok": "",
                    "clean_source_valid": "",
                    "notes": "",
                }
            )
    return rows


def attach_sheets(rows: list[dict[str, str]], output_dir: Path, project_root: Path) -> None:
    sheets = output_dir / "contact_sheets"
    for row in rows:
        sheet_path = sheets / f"{row['backbone']}_{row['candidate_id']}.jpg"
        built = build_frame_strip(
            row["video_path"],
            output_path=sheet_path,
            project_root=project_root,
            frames_per_video=12,
            thumb_width=160,
            thumb_height=96,
        )
        if built is None:
            row["notes"] = "contact sheet unavailable"
            continue
        row["contact_sheet"] = str(built)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--manifest must use backbone=path")
    backbone, path = value.split("=", 1)
    return backbone, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/five_mechanism_eval_candidates_v0.csv"))
    parser.add_argument("--manifest", action="append", type=parse_manifest, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        rows = build_rows(read_candidates(args.candidates), args.manifest)
        attach_sheets(rows, args.output_dir, args.project_root)
        write_csv(args.output_dir / "smoke_review.csv", rows)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {len(rows)} joint review rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
