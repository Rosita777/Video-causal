#!/usr/bin/env python3
"""Link existing erase and preservation latent caches into the v2 row order."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def cached_by_scene(cache_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in cache_dir.glob("*.pt"):
        scene_id = path.stem.split("_", 1)[1]
        if scene_id in result:
            raise ValueError(f"Duplicate cache for {scene_id}")
        result[scene_id] = path.resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"),
    )
    parser.add_argument(
        "--erase-cache",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v1/cache_dynamic_sft_v1"),
    )
    parser.add_argument(
        "--preserve-cache",
        type=Path,
        default=Path("outputs/protocol_v1/cache_water_impact"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"),
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sources = {
        "erase": cached_by_scene(args.erase_cache),
        "preserve": cached_by_scene(args.preserve_cache),
    }
    args.output_cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        role = row["training_role"]
        scene_id = row["scene_id"]
        source = sources[role].get(scene_id)
        if source is None:
            raise FileNotFoundError(f"Missing {role} cache for {scene_id}")
        destination = args.output_cache / f"{index:03d}_{scene_id}.pt"
        if destination.is_symlink() and destination.resolve() == source:
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
    linked = list(args.output_cache.glob("*.pt"))
    if len(linked) != len(rows):
        raise ValueError(f"Expected {len(rows)} cache links, found {len(linked)}")
    print(f"Prepared {len(linked)} cache links in {args.output_cache}")


if __name__ == "__main__":
    main()
