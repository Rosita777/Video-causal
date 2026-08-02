#!/usr/bin/env python3
"""Summarize removal and preservation metrics for redirect-weight quick evals."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SWEEP_METHODS = {
    "plain",
    "dual_traj_rw0025",
    "dual_traj_bg1",
    "dual_traj_rw0100",
    "dual_traj_scale050",
    "dual_traj_scale075",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] in SWEEP_METHODS]

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary: list[dict[str, str]] = []
    for method, method_rows in sorted(by_method.items()):
        removal = [row for row in method_rows if row["split"] != "control"]
        controls = [row for row in method_rows if row["split"] == "control"]
        summary.append(
            {
                "method": method,
                "removal_cases": str(len(removal)),
                "mean_removal_suppression_percent": f"{np.mean([float(row['post_change_suppression_percent']) for row in removal]):.2f}",
                "mean_removal_early_base_mae": f"{np.mean([float(row['early_base_mae']) for row in removal]):.8f}",
                "mean_control_early_base_mae": f"{np.mean([float(row['early_base_mae']) for row in controls]):.8f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
