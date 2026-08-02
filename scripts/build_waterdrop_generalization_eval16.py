#!/usr/bin/env python3
"""Export the receiver-held-out portion of the waterdrop v2 split for inference."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("data/waterdrop_generalization_split_v2.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/waterdrop_generalization_eval16.csv"))
    parser.add_argument("--output-prompts", type=Path, default=Path("prompts/waterdrop_generalization_eval16.txt"))
    parser.add_argument("--output-seeds", type=Path, default=Path("prompts/waterdrop_generalization_eval16_seeds.txt"))
    args = parser.parse_args()

    with args.split.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "internal_receiver_holdout"]
    if len(rows) != 16:
        raise ValueError(f"Expected 16 internal holdout rows, found {len(rows)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["eval_index", *rows[0].keys()], lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"eval_index": f"{index:03d}", **row})

    args.output_prompts.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prompts.open("w", encoding="utf-8") as handle:
        handle.write("# Receiver-held-out waterdrop generalization eval16.\n")
        for row in rows:
            handle.write(f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}\n")
    args.output_seeds.write_text(",".join(row["seed"] for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} holdout prompts and CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
