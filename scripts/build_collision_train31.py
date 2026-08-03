#!/usr/bin/env python3
"""Build aligned counterfactual pairs and a 31-row collision training manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


BATCHES = (
    {
        "name": "gate30",
        "prompt_bank": "data/collision_prompt_gate30.csv",
        "auto_screen": "data/collision_prompt_gate30_auto_screen.csv",
        "review": "data/collision_prompt_gate30_semantic_review.csv",
        "selected": lambda row: row["decision"] == "accept",
    },
    {
        "name": "expansion72",
        "prompt_bank": "data/collision_expansion72.csv",
        "auto_screen": "data/collision_expansion72_auto_screen.csv",
        "review": "data/collision_expansion72_semantic_review.csv",
        "selected": lambda row: row["human_decision"] == "accept",
    },
    {
        "name": "feasible72",
        "prompt_bank": "data/collision_feasible72.csv",
        "auto_screen": "data/collision_feasible72_auto_screen.csv",
        "review": "data/collision_feasible72_semantic_review.csv",
        "selected": lambda row: row["selection_split"] == "train",
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/collision_train31_aligned_pairs"))
    parser.add_argument("--manifest", type=Path, default=Path("data/collision_train31.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_root = root / args.output_root
    selected_rows: list[dict[str, str]] = []

    for batch in BATCHES:
        prompts = {row["scene_id"]: row for row in read_rows(root / batch["prompt_bank"])}
        screens = {row["scene_id"]: row for row in read_rows(root / batch["auto_screen"])}
        for review in read_rows(root / batch["review"]):
            if not batch["selected"](review):
                continue
            scene_id = review["scene_id"]
            prompt = prompts[scene_id]
            screen = screens[scene_id]
            factual = root / screen["video_path"]
            if not factual.is_file():
                raise FileNotFoundError(f"{scene_id}: factual video not found: {factual}")
            reference_end = int(review["reference_end_exclusive"])
            if reference_end <= 0:
                raise ValueError(f"{scene_id}: invalid reference_end_exclusive")
            selected_rows.append({
                "batch": batch["name"],
                "scene_id": scene_id,
                "family": prompt["family"],
                "receiver": prompt["receiver"],
                "seed": prompt["fixed_seed"],
                "prompt": prompt["prompt"],
                "factual_video": screen["video_path"],
                "reference_end_exclusive": str(reference_end),
            })

    counts = {batch["name"]: sum(row["batch"] == batch["name"] for row in selected_rows) for batch in BATCHES}
    if counts != {"gate30": 10, "expansion72": 7, "feasible72": 14}:
        raise ValueError(f"unexpected selected counts: {counts}")
    if len(selected_rows) != 31:
        raise ValueError(f"expected exactly 31 selected rows, got {len(selected_rows)}")
    print(f"Validated 31 factual videos: {counts}")
    if args.dry_run:
        return 0

    records = []
    builder = root / "scripts/build_static_counterfactual_pair.py"
    for index, row in enumerate(selected_rows):
        pair_dir = output_root / f"{row['scene_id']}_seed{row['seed']}"
        pair_manifest = pair_dir / "pair_manifest.json"
        if not pair_manifest.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(builder),
                    "--input",
                    row["factual_video"],
                    "--output-dir",
                    str(pair_dir.relative_to(root)),
                    "--reference-end",
                    row["reference_end_exclusive"],
                ],
                cwd=root,
                check=True,
            )
        with pair_manifest.open(encoding="utf-8") as handle:
            pair = json.load(handle)
        target = root / pair["output_video"]
        if not target.is_file():
            raise FileNotFoundError(f"{row['scene_id']}: target video missing: {target}")
        factual = row["factual_video"]
        target_rel = str(target.relative_to(root))
        records.append({
            "scene_id": f"collision_{index:03d}",
            "train_group_id": row["scene_id"],
            "receiver_id": row["family"],
            "receiver": row["receiver"],
            "condition": "explicit_causal",
            "prompt": row["prompt"],
            "generated_video": factual,
            "desired_target_video": target_rel,
            "training_role": "erase",
            "training_objective": "counterfactual_noise_prediction",
            "residual_mask_enabled": "yes",
            "residual_mask_factual_video": factual,
            "residual_mask_target_video": target_rel,
            "reference_end_exclusive": row["reference_end_exclusive"],
            "source_split": row["batch"],
            "source_index": str(index),
        })

    manifest = root / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} training rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
