#!/usr/bin/env python3
"""Build one training manifest from reviewed aligned waterdrop batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BATCHES = (
    {
        "name": "seed8300",
        "results": "data/waterdrop_static_pass6_pair_results.csv",
        "review": None,
        "root": "outputs/waterdrop_static_pass6_aligned_pairs",
    },
    {
        "name": "seed8400",
        "results": "data/waterdrop_repeat_pass16_pair_results.csv",
        "review": "data/waterdrop_scene_probe30_repeat_review.csv",
        "root": "outputs/waterdrop_repeat_pass16_aligned_pairs",
    },
    {
        "name": "seed8500",
        "results": "data/waterdrop_seed8500_pair_results.csv",
        "review": "data/waterdrop_scene_probe30_seed8500_review.csv",
        "root": "outputs/waterdrop_seed8500_pass13_aligned_pairs",
    },
    {
        "name": "seed8600",
        "results": "data/waterdrop_seed8600_pair_results.csv",
        "review": "data/waterdrop_scene_probe30_seed8600_review.csv",
        "root": "outputs/waterdrop_seed8600_pass12_aligned_pairs",
    },
    {
        "name": "seed8700",
        "results": "data/waterdrop_seed8700_pair_results.csv",
        "review": "data/waterdrop_scene_probe30_seed8700_review.csv",
        "root": "outputs/waterdrop_seed8700_pass11_aligned_pairs",
    },
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("data/waterdrop_aligned_pairs_v1.csv")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    records: list[dict[str, object]] = []

    for batch in BATCHES:
        review_by_id: dict[str, dict[str, str]] = {}
        if batch["review"]:
            for row in read_rows(repo_root / str(batch["review"])):
                review_by_id[row["sample_id"]] = row

        for result in read_rows(repo_root / str(batch["results"])):
            if result["decision"] != "pass":
                continue
            sample_id = result.get("sample_id") or result["scene_id"]
            seed = result["seed"]
            review = review_by_id.get(sample_id, {})
            receiver = result.get("receiver") or review.get("receiver")
            if not receiver:
                raise ValueError(f"receiver missing for {sample_id}")

            pair_dir = repo_root / str(batch["root"]) / f"{sample_id}_seed{seed}"
            manifest_path = pair_dir / "pair_manifest.json"
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)

            required = {
                "factual_video": repo_root / manifest["input"],
                "target_video": repo_root / manifest["output_video"],
                "clean_reference": repo_root / manifest["clean_reference"],
            }
            for kind, path in required.items():
                if not path.is_file():
                    raise FileNotFoundError(f"{sample_id}: {kind} not found: {path}")

            records.append(
                {
                    "pair_id": f"waterdrop_{len(records):03d}",
                    "batch": batch["name"],
                    "sample_id": sample_id,
                    "seed": seed,
                    "receiver": receiver,
                    "factual_video": required["factual_video"].relative_to(repo_root),
                    "target_video": required["target_video"].relative_to(repo_root),
                    "clean_reference": required["clean_reference"].relative_to(repo_root),
                    "reference_end_exclusive": result["reference_end_exclusive"],
                    "first_frame_reference_mae_255": result[
                        "first_frame_reference_mae_255"
                    ],
                }
            )

    fieldnames = list(records[0])
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} accepted aligned pairs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
