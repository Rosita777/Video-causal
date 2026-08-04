#!/usr/bin/env python3
"""Build all four seen-scene prompts used by the collision overfit audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/collision_overfit4.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/collision_overfit4_eval_all.prompts"))
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["training_role"] == "erase"]
    if len(rows) != 4:
        raise ValueError(f"Expected four erase rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row['prompt']} | one small red rubber ball | "
                "the receiver remains upright when the ball and collision chain are erased\n"
            )
    print(f"Wrote {len(rows)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
