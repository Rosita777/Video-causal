#!/usr/bin/env python3
"""Build erase rows from target-only videos and their static counterfactual pairs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, default=Path("outputs/collision_target_only8_base/generation_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/collision_target_only8_pairs"))
    parser.add_argument("--output", type=Path, default=Path("data/collision_object_only8.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.generation.read_text(encoding="utf-8"))
    items = list(manifest["items"])
    if len(items) != 8:
        raise SystemExit(f"Expected 8 generated items, found {len(items)}")
    root = Path(".").resolve()
    records = []
    builder = root / "scripts/build_static_counterfactual_pair.py"
    for index, item in enumerate(items):
        factual = Path(str(item["video_path"]))
        pair_dir = args.output_root / f"targetonly_{index:03d}"
        pair_manifest = pair_dir / "pair_manifest.json"
        if not args.dry_run and not pair_manifest.exists():
            subprocess.run(
                [sys.executable, str(builder), "--input", str(factual), "--output-dir", str(pair_dir), "--reference-end", "16"],
                cwd=root,
                check=True,
            )
        if args.dry_run:
            continue
        pair = json.loads(pair_manifest.read_text(encoding="utf-8"))
        target = str(pair["output_video"])
        records.append({
            "scene_id": f"targetonly_{index:03d}",
            "train_group_id": f"targetonly_{index:03d}",
            "receiver_id": "target_only",
            "receiver": "target-only tabletop scene",
            "condition": "target_only",
            "prompt": item["prompt"],
            "generated_video": str(factual),
            "desired_target_video": target,
            "training_role": "erase",
            "training_objective": "counterfactual_noise_prediction",
            "residual_mask_enabled": "yes",
            "residual_mask_factual_video": str(factual),
            "residual_mask_target_video": target,
            "reference_end_exclusive": "16",
            "source_split": "collision_target_only8",
            "source_index": str(index),
        })
    if args.dry_run:
        print("Validated 8 target-only generation items")
        return 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} target-only erase rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
