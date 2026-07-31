#!/usr/bin/env python3
"""Build a 100-scene waterdrop pilot whose prompts omit causal outcomes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


SHARD_COUNT = 4
FIXED_SEED = 9000


def make_implicit_prompt(row: dict[str, str]) -> str:
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"During the first two seconds, {row['receiver']} is {row['surface_condition']}. "
        "No water droplet is visible during these first two seconds. "
        "Then exactly one large clear water droplet enters from the top of the frame, "
        f"falls visibly downward, and contacts {row['impact_location']}. "
        "The camera and lighting remain fixed throughout."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source", type=Path, default=Path("data/waterdrop_prompt_bank_v2_simple.csv")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("data/waterdrop_implicit_pilot100.csv")
    )
    parser.add_argument(
        "--txt", type=Path, default=Path("prompts/waterdrop_implicit_pilot100.txt")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/waterdrop_implicit_pilot100_run_manifest.csv"),
    )
    parser.add_argument(
        "--prompt-prefix",
        type=Path,
        default=Path("prompts/waterdrop_implicit_pilot100_shard"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    with source.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    selected = [
        row
        for row in source_rows
        if row["family"] in {"liquid_surface", "hard_surface"} and row["variant"] == "0"
    ]
    rows: list[dict[str, str]] = []
    for index, source_row in enumerate(selected):
        rows.append(
            {
                "scene_id": f"wdimplicit{index:04d}",
                "source_scene_id": source_row["scene_id"],
                "family": source_row["family"],
                "mechanism": source_row["mechanism"],
                "receiver_id": source_row["receiver_id"],
                "receiver": source_row["receiver"],
                "variant": source_row["variant"],
                "surface_condition": source_row["surface_condition"],
                "impact_location": source_row["impact_location"],
                "causal_footprint": source_row["causal_footprint"],
                "generation_prompt_type": "implicit_outcome",
                "result_mentioned_in_prompt": "false",
                "erase_instruction": "Remove the water droplet.",
                "seed_policy": "one_fixed_seed",
                "prompt": make_implicit_prompt(source_row),
            }
        )

    counts = Counter(row["family"] for row in rows)
    expected = Counter({"liquid_surface": 50, "hard_surface": 50})
    if len(rows) != 100 or counts != expected:
        raise ValueError(f"unexpected pilot counts: {counts}")
    if len({row["prompt"] for row in rows}) != len(rows):
        raise ValueError("duplicate prompts found")
    forbidden = ("splash", "ripple", "wet spot", "water bead", "impact cavity")
    for row in rows:
        lowered = row["prompt"].lower()
        if any(term in lowered for term in forbidden):
            raise ValueError(f"causal outcome leaked into prompt: {row['scene_id']}")

    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    txt_path = args.txt if args.txt.is_absolute() else root / args.txt
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Waterdrop implicit-outcome pilot: 100 scenes, fixed seed 9000.\n")
        handle.write("# The generation prompt does not mention the causal footprint.\n")
        handle.write("# Format: <prompt> | <target> | <hidden expected effect>\n\n")
        for row in rows:
            handle.write(
                f"{row['prompt']} | single falling water droplet | {row['causal_footprint']}\n"
            )

    shards: list[list[dict[str, str]]] = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows: list[dict[str, str]] = []
    prefix = args.prompt_prefix if args.prompt_prefix.is_absolute() else root / args.prompt_prefix
    for shard, shard_rows in enumerate(shards):
        prompt_path = Path(f"{prefix}_{shard}.txt")
        with prompt_path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# Waterdrop implicit-outcome pilot, shard {shard}/{SHARD_COUNT - 1}; "
                f"fixed seed {FIXED_SEED}.\n"
            )
            handle.write("# Format: <prompt> | <target> | <hidden expected effect>\n\n")
            for shard_index, row in enumerate(shard_rows):
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
                        "generation_prompt_type": row["generation_prompt_type"],
                        "result_mentioned_in_prompt": row["result_mentioned_in_prompt"],
                        "erase_instruction": row["erase_instruction"],
                        "causal_footprint": row["causal_footprint"],
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                        "fixed_seed": str(FIXED_SEED),
                    }
                )

    sizes = [len(shard) for shard in shards]
    if sizes != [25, 25, 25, 25]:
        raise ValueError(f"unexpected shard sizes: {sizes}")
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote 100 implicit prompts: {dict(counts)}; shards={sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
