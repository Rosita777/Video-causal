#!/usr/bin/env python3
"""Build a 20-scene target-only control with a resting water bead."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SHARD_COUNT = 4
FIXED_SEED = 9200


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evenly_select(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError(f"non-unique selection indices: {indices}")
    return [rows[index] for index in indices]


def make_prompt(row: dict[str, str]) -> str:
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"For the entire clip, exactly one single clear spherical water bead rests "
        f"motionless at the center of a {row['receiver']}. "
        "The compact rounded water bead is already present in the first frame and remains "
        "clearly visible in the same place. The hard surface is dry everywhere else. "
        "There is no falling water, impact, splash, spreading wet patch, trail, or ripple. "
        "The camera, lighting, surface, and background remain fixed throughout."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source = read_csv(root / "data/waterdrop_prompt_bank_v2_simple.csv")
    explicit = read_csv(root / "data/waterdrop_prompt_bank_v2_auto_screen.csv")
    implicit = read_csv(root / "data/waterdrop_implicit_pilot100_auto_screen.csv")
    explicit_status = {row["scene_id"]: row["auto_status"] for row in explicit}
    implicit_status = {row["source_scene_id"]: row["auto_status"] for row in implicit}
    eligible = sorted(
        (
            row
            for row in source
            if row["family"] == "hard_surface"
            and row["variant"] == "0"
            and explicit_status.get(row["scene_id"]) == "candidate"
            and implicit_status.get(row["scene_id"]) == "candidate"
        ),
        key=lambda row: row["receiver_id"],
    )
    selected = evenly_select(eligible, 20)
    rows = []
    for index, row in enumerate(selected):
        rows.append(
            {
                "scene_id": f"wdresting{index:03d}",
                "source_scene_id": row["scene_id"],
                "receiver_id": row["receiver_id"],
                "family": row["family"],
                "receiver": row["receiver"],
                "condition": "target_only_resting_bead",
                "split": "frozen_test_candidate",
                "expected_base_target": "yes",
                "expected_base_footprint": "no",
                "expected_erased_target": "no",
                "expected_erased_footprint": "no",
                "erase_instruction": "Remove the water droplet.",
                "fixed_seed": str(FIXED_SEED),
                "prompt": make_prompt(row),
            }
        )
    if len(rows) != 20 or len({row["receiver_id"] for row in rows}) != 20:
        raise ValueError("expected 20 unique hard-surface receivers")

    csv_path = root / "data/waterdrop_target_only_resting20.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    txt_path = root / "prompts/waterdrop_target_only_resting20.txt"
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Target-only control: one resting water bead on a hard surface.\n")
        handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
        for row in rows:
            handle.write(
                f"{row['prompt']} | single water droplet | no causal footprint\n"
            )

    shards: list[list[dict[str, str]]] = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows = []
    for shard, shard_rows in enumerate(shards):
        path = root / f"prompts/waterdrop_target_only_resting20_shard_{shard}.txt"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# Resting-bead target-only control, shard {shard}/{SHARD_COUNT - 1}; "
                f"fixed seed {FIXED_SEED}.\n"
            )
            handle.write("# Format: <prompt> | <target> | <expected effect>\n\n")
            for shard_index, row in enumerate(shard_rows):
                handle.write(
                    f"{row['prompt']} | single water droplet | no causal footprint\n"
                )
                manifest_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "prompt"},
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                    }
                )
    manifest = root / "data/waterdrop_target_only_resting20_run_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    heldout = root / "data/waterdrop_target_only_resting20_heldout_receivers.txt"
    heldout.write_text(
        "\n".join(row["receiver_id"] for row in rows) + "\n", encoding="utf-8"
    )
    print("Wrote 20 resting-bead target-only prompts; shards=[5, 5, 5, 5]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
