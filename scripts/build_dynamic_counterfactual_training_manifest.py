#!/usr/bin/env python3
"""Join dynamic prompt pairs with generated target videos for LoRA training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXTRA_FIELDS = [
    "scene_id",
    "prompt",
    "training_role",
    "training_objective",
    "generated_video",
    "residual_mask_enabled",
    "residual_mask_factual_video",
    "residual_mask_target_video",
]


def load_generated_items(paths: list[Path]) -> dict[int, dict[str, object]]:
    items: dict[int, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dry_run"):
            raise ValueError(f"Generation manifest is a dry run: {path}")
        for item in payload["items"]:
            index = int(item["index"])
            if index in items:
                raise ValueError(f"Duplicate generated index {index}")
            items[index] = item
    return items


def build_training_rows(
    pair_rows: list[dict[str, str]],
    generated: dict[int, dict[str, object]],
    accepted_indices: set[int] | None = None,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for index, pair in enumerate(pair_rows):
        if accepted_indices is not None and index not in accepted_indices:
            continue
        if index not in generated:
            raise ValueError(f"Missing target video for pair index {index}")
        item = generated[index]
        if item["prompt"] != pair["target_generation_prompt"]:
            raise ValueError(f"Target prompt mismatch at pair index {index}")
        if int(item["seed"]) != int(pair["seed"]):
            raise ValueError(f"Target seed mismatch at pair index {index}")
        video = Path(str(item["video_path"]))
        if not video.exists():
            raise FileNotFoundError(video)
        try:
            portable_video = video.relative_to(Path.cwd())
        except ValueError:
            portable_video = video
        row = dict(pair)
        row.update(
            {
                "scene_id": pair["pair_id"],
                "prompt": pair["training_prompt"],
                "training_role": "erase",
                "training_objective": "dynamic_counterfactual_sft",
                "generated_video": "",
                "desired_target_video": str(portable_video),
                "residual_mask_enabled": "no",
                "residual_mask_factual_video": "",
                "residual_mask_target_video": "",
            }
        )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--screen-csv",
        type=Path,
        help="Optional screening CSV; only rows with final_status=accept are retained.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.pairs.open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
        pair_fields = list(pair_rows[0]) if pair_rows else []
    accepted_indices = None
    if args.screen_csv is not None:
        with args.screen_csv.open(newline="", encoding="utf-8") as handle:
            screen_rows = list(csv.DictReader(handle))
        accepted_indices = {
            int(row["pair_index"])
            for row in screen_rows
            if row.get("final_status") == "accept"
        }
        if not accepted_indices:
            raise ValueError("Screening CSV contains no final_status=accept rows")
    rows = build_training_rows(
        pair_rows,
        load_generated_items(args.generation_manifest),
        accepted_indices,
    )
    fieldnames = pair_fields + [field for field in EXTRA_FIELDS if field not in pair_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} dynamic SFT rows to {args.output}")


if __name__ == "__main__":
    main()
