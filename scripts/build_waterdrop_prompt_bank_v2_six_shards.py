#!/usr/bin/env python3
"""Split waterdrop prompt bank v2 into six balanced Wan input shards."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


SHARD_COUNT = 6
FIXED_SEED = 9000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source", type=Path, default=Path("data/waterdrop_prompt_bank_v2_simple.csv")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/waterdrop_prompt_bank_v2_simple_run_manifest.csv"),
    )
    parser.add_argument(
        "--prompt-prefix",
        type=Path,
        default=Path("prompts/waterdrop_prompt_bank_v2_simple_shard"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_family: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    shards: list[list[dict[str, str]]] = [[] for _ in range(SHARD_COUNT)]
    cursor = 0
    for family in ("liquid_surface", "hard_surface"):
        for row in by_family[family]:
            shards[cursor % SHARD_COUNT].append(row)
            cursor += 1

    manifest_rows: list[dict[str, str]] = []
    for shard_index, shard_rows in enumerate(shards):
        prefix = args.prompt_prefix if args.prompt_prefix.is_absolute() else root / args.prompt_prefix
        prompt_path = Path(f"{prefix}_{shard_index}.txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# Waterdrop prompt bank v2 simple, shard {shard_index}/{SHARD_COUNT - 1}; "
                f"fixed seed {FIXED_SEED}.\n"
            )
            handle.write("# Format: <prompt> | <target> | <effect>\n\n")
            for local_index, row in enumerate(shard_rows):
                handle.write(
                    f"{row['prompt']} | single falling water droplet | "
                    f"{row['causal_footprint']}\n"
                )
                manifest_rows.append(
                    {
                        "scene_id": row["scene_id"],
                        "source_scene_id": row["source_scene_id"],
                        "family": row["family"],
                        "receiver": row["receiver"],
                        "variant": row["variant"],
                        "shard": str(shard_index),
                        "shard_index": str(local_index),
                        "fixed_seed": str(FIXED_SEED),
                    }
                )

    sizes = [len(shard) for shard in shards]
    if sum(sizes) != 250 or max(sizes) - min(sizes) > 1:
        raise ValueError(f"unbalanced shards: {sizes}")
    if len({row["scene_id"] for row in manifest_rows}) != 250:
        raise ValueError("duplicate or missing scene IDs in run manifest")
    family_counts = Counter(row["family"] for row in manifest_rows)
    if family_counts != Counter({"liquid_surface": 150, "hard_surface": 100}):
        raise ValueError(f"unexpected family counts: {family_counts}")

    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote six shards with sizes {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
