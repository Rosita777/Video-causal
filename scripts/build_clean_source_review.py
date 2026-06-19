#!/usr/bin/env python3
"""Build review artifacts for clean-source screening manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLEAN_FIELDS = [
    "prompt_id",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos-per-sheet", type=int, default=5)
    parser.add_argument("--frames-per-video", type=int, default=9)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--thumb-height", type=int, default=160)
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    outputs = data.get("outputs", [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, item in enumerate(outputs):
        rows.append(
            {
                "prompt_id": item.get("prompt_id") or f"case_{idx:03d}",
                "video_path": item.get("output_path", ""),
                "prompt": item.get("prompt", ""),
                "target_concept": item.get("target_concept", ""),
                "expected_effect": item.get("expected_effect", ""),
                "target_visible": "",
                "effect_visible": "",
                "temporal_order_clear": "",
                "effect_depends_on_target": "",
                "video_quality": "",
                "clean_source_valid": "",
                "notes": "",
            }
        )
    with (args.output_dir / "clean_source_screening.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_dir / 'clean_source_screening.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
