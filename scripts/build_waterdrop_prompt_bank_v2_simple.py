#!/usr/bin/env python3
"""Build the evidence-supported simple subset of waterdrop prompt bank v1."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


KEEP_VARIANTS = {
    "liquid_surface": {"0", "1", "2"},
    "hard_surface": {"0", "1"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, default=Path("data/waterdrop_prompt_bank_v1.csv"))
    parser.add_argument("--csv", type=Path, default=Path("data/waterdrop_prompt_bank_v2_simple.csv"))
    parser.add_argument("--txt", type=Path, default=Path("prompts/waterdrop_prompt_bank_v2_simple.txt"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    with source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, str]] = []
    for source_row in source_rows:
        variants = KEEP_VARIANTS.get(source_row["family"])
        if variants is None or source_row["variant"] not in variants:
            continue
        rows.append(
            {
                "scene_id": f"wdsimple{len(rows):04d}",
                "source_scene_id": source_row["scene_id"],
                **{key: value for key, value in source_row.items() if key != "scene_id"},
            }
        )

    counts = Counter(row["family"] for row in rows)
    expected = Counter({"liquid_surface": 150, "hard_surface": 100})
    if counts != expected or len(rows) != 250:
        raise ValueError(f"unexpected v2 counts: {counts}")
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("duplicate prompts found")

    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    txt_path = args.txt if args.txt.is_absolute() else root / args.txt
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Waterdrop prompt bank v2 simple: 250 scenes, one seed per prompt.\n")
        handle.write("# Families: liquid surface and simple hard surface only.\n")
        handle.write("# Format: <prompt> | <target> | <effect>\n\n")
        for row in rows:
            handle.write(
                f"{row['prompt']} | single falling water droplet | "
                f"{row['causal_footprint']}\n"
            )
    print(f"Wrote {len(rows)} simple prompts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
