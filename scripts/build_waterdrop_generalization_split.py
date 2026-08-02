#!/usr/bin/env python3
"""Build a receiver-held-out candidate split from reviewed aligned waterdrop pairs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


BATCHES = (
    ("seed8300", "data/waterdrop_scene_probe30_review.csv"),
    ("seed8400", "data/waterdrop_scene_probe30_repeat_review.csv"),
    ("seed8500", "data/waterdrop_scene_probe30_seed8500_review.csv"),
    ("seed8600", "data/waterdrop_scene_probe30_seed8600_review.csv"),
    ("seed8700", "data/waterdrop_scene_probe30_seed8700_review.csv"),
)
PROMPTS = {
    "part_a": Path("prompts/waterdrop_scene_probe30_part_a.txt"),
    "part_b": Path("prompts/waterdrop_scene_probe30_part_b.txt"),
}

# These receiver keys occur in the frozen eval20, including obvious synonyms.
EXTERNAL_EVAL_KEYS = {"pond", "cup", "tray", "cutting_board"}
# Internal holdout adds unseen receiver/footprint families to the training audit.
INTERNAL_HOLDOUT_KEYS = {"glass_tabletop", "bucket", "cardboard", "sponge", "chalk"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompt, target, effect = [part.strip() for part in line.split("|", 2)]
        rows.append({"prompt": prompt, "target_concept": target, "expected_effect": effect})
    return rows


def receiver_key(receiver: str) -> str:
    value = receiver.lower()
    if "pond" in value:
        return "pond"
    if "cup" in value:
        return "cup"
    if "tray" in value:
        return "tray"
    if "cutting board" in value:
        return "cutting_board"
    if "tabletop" in value:
        return "glass_tabletop"
    if "bucket" in value:
        return "bucket"
    if "cardboard" in value:
        return "cardboard"
    if "sponge" in value:
        return "sponge"
    if "chalk" in value:
        return "chalk"
    if "towel" in value or "cloth" in value or "tissue" in value:
        return "absorbent_fabric"
    if "soil" in value:
        return "soil"
    if "sand" in value or "flour" in value or "salt" in value or "coffee" in value:
        return "particulate"
    if "aquarium" in value or "puddle" in value or "water" in value or "bowl" in value:
        return "liquid_container"
    if "wood" in value:
        return "wood"
    if "tile" in value or "marble" in value or "plate" in value:
        return "hard_surface"
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def footprint_family(receiver: str) -> str:
    value = receiver.lower()
    if any(token in value for token in ("pond", "water", "bowl", "bucket", "puddle", "aquarium", "cup")):
        return "splash_ripple"
    if any(token in value for token in ("towel", "cloth", "tissue", "sponge", "cardboard", "soil", "wood")):
        return "spreading_wet_patch"
    if any(token in value for token in ("sand", "flour", "salt", "coffee", "chalk")):
        return "crater_or_damp_particle_mark"
    return "splash_wet_mark"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--aligned", type=Path, default=Path("data/waterdrop_aligned_pairs_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/waterdrop_generalization_split_v1.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/waterdrop_generalization_split_v1_summary.json"))
    args = parser.parse_args()
    root = args.repo_root.resolve()

    review_by_id: dict[str, dict[str, str]] = {}
    for batch, review_path in BATCHES:
        for row in read_csv(root / review_path):
            item = dict(row)
            item["batch"] = batch
            review_id = row.get("sample_id") or row.get("global_id")
            if not review_id:
                raise ValueError(f"Review row in {review_path} has no sample_id/global_id")
            review_by_id[review_id] = item

    prompts = {name: parse_prompts(root / path) for name, path in PROMPTS.items()}
    aligned_path = args.aligned if args.aligned.is_absolute() else root / args.aligned
    aligned = read_csv(aligned_path)
    records: list[dict[str, str]] = []
    for row in aligned:
        review = review_by_id.get(row["sample_id"])
        if review is None:
            raise ValueError(f"Missing review row for {row['sample_id']}")
        part_name = review["part"]
        if part_name in {"a", "b"}:
            part_name = f"part_{part_name}"
        prompt_rows = prompts[part_name]
        prompt = prompt_rows[int(review["index"])]
        key = receiver_key(row["receiver"])
        if key in EXTERNAL_EVAL_KEYS:
            split = "reserved_external_eval_overlap"
        elif key in INTERNAL_HOLDOUT_KEYS:
            split = "internal_receiver_holdout"
        else:
            split = "train_candidate"
        records.append(
            {
                "pair_id": row["pair_id"],
                "sample_id": row["sample_id"],
                "batch": row["batch"],
                "seed": row["seed"],
                "receiver": row["receiver"],
                "receiver_key": key,
                "footprint_family": footprint_family(row["receiver"]),
                "split": split,
                "prompt": prompt["prompt"],
                "target_concept": prompt["target_concept"],
                "expected_effect": prompt["expected_effect"],
                "factual_video": row["factual_video"],
                "target_video": row["target_video"],
                "clean_reference": row["clean_reference"],
                "reference_end_exclusive": row["reference_end_exclusive"],
                "first_frame_reference_mae_255": row["first_frame_reference_mae_255"],
            }
        )

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(row["split"] for row in records)
    summary = {
        "total_aligned_pairs": len(records),
        "counts_by_split": dict(counts),
        "train_candidate_by_footprint": dict(
            Counter(row["footprint_family"] for row in records if row["split"] == "train_candidate")
        ),
        "internal_holdout_by_footprint": dict(
            Counter(row["footprint_family"] for row in records if row["split"] == "internal_receiver_holdout")
        ),
        "minimum_new_train_pairs_needed_for_30": max(0, 30 - counts["train_candidate"]),
        "external_eval_receiver_keys": sorted(EXTERNAL_EVAL_KEYS),
        "internal_holdout_receiver_keys": sorted(INTERNAL_HOLDOUT_KEYS),
    }
    summary_path = args.summary if args.summary.is_absolute() else root / args.summary
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
