#!/usr/bin/env python3
"""Compare full-strength and conservative-scale dual trajectory eval20."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_case = {(row["eval_index"], row["method"]): row for row in rows}
    cases = sorted({row["eval_index"] for row in rows})
    output = []
    for eval_index in cases:
        full = by_case[(eval_index, "dual_traj")]
        conservative = by_case[(eval_index, "dual_traj_scale075")]
        output.append(
            {
                "eval_index": eval_index,
                "scene_id": full["scene_id"],
                "condition": full["condition"],
                "family": full["family"],
                "receiver": full["receiver"],
                "full_scale_suppression_percent": full["post_change_suppression_percent"],
                "scale075_suppression_percent": conservative["post_change_suppression_percent"],
                "scale075_minus_full_points": f"{float(conservative['post_change_suppression_percent']) - float(full['post_change_suppression_percent']):.2f}",
                "full_scale_early_base_mae": full["early_base_mae"],
                "scale075_early_base_mae": conservative["early_base_mae"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
