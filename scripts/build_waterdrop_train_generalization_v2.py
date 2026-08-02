#!/usr/bin/env python3
"""Convert the waterdrop v2 split into the Wan LoRA training manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("data/waterdrop_generalization_split_v2.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/waterdrop_train_generalization_v2.csv"))
    args = parser.parse_args()

    with args.split.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train_rows = [row for row in rows if row["split"] == "train_candidate"]
    if not train_rows:
        raise ValueError("v2 split has no train_candidate rows")

    records = []
    for index, row in enumerate(train_rows):
        records.append(
            {
                "scene_id": row["pair_id"],
                "train_group_id": row["sample_id"],
                "receiver_id": row["receiver_key"],
                "receiver": row["receiver"],
                "condition": "explicit_causal",
                "prompt": row["prompt"],
                "generated_video": row["factual_video"],
                "desired_target_video": row["target_video"],
                "training_role": "erase",
                "training_objective": "counterfactual_noise_prediction",
                "residual_mask_enabled": "yes",
                "residual_mask_factual_video": row["factual_video"],
                "residual_mask_target_video": row["target_video"],
                "reference_end_exclusive": row["reference_end_exclusive"],
                "source_split": row["split"],
                "source_index": str(index),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} erase rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
