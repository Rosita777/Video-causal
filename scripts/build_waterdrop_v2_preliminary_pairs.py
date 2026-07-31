#!/usr/bin/env python3
"""Build provisional static targets for technically clean auto-screen candidates."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--screen-csv",
        type=Path,
        default=Path("data/waterdrop_prompt_bank_v2_auto_screen.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/waterdrop_prompt_bank_v2_preliminary_pairs")
    )
    parser.add_argument("--limit", type=int, help="Build at most this many candidates")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    screen_csv = args.screen_csv if args.screen_csv.is_absolute() else root / args.screen_csv
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    builder = root / "scripts/build_static_counterfactual_pair.py"

    with screen_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if row["auto_status"] == "candidate" and row["estimated_clean_end_exclusive"].isdigit()
    ]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        candidates = candidates[: args.limit]
    built = 0
    for row in candidates:
        video = Path(row["video_path"])
        if not video.is_absolute():
            video = root / video
        output_dir = output_root / f"{row['scene_id']}_shard{row['shard']}_{row['shard_index']}"
        subprocess.run(
            [
                "models/.wan-runtime/bin/python",
                str(builder),
                "--input",
                str(video),
                "--output-dir",
                str(output_dir),
                "--reference-start",
                "0",
                "--reference-end",
                row["estimated_clean_end_exclusive"],
            ],
            cwd=root,
            check=True,
        )
        built += 1
    print(f"Built {built} provisional pairs from {len(rows)} auto-screened videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
