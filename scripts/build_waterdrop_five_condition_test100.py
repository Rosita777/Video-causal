#!/usr/bin/env python3
"""Build a frozen 20-receiver, five-condition waterdrop evaluation set."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CONDITIONS = (
    "explicit_causal",
    "implicit_causal",
    "target_only",
    "unrelated_footprint",
    "clean_control",
)
SHARD_COUNT = 4
FIXED_SEED = 9100


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evenly_select(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) < count:
        raise ValueError(f"need {count} rows, found {len(rows)}")
    if count == 1:
        return [rows[0]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError(f"selection indices are not unique: {indices}")
    return [rows[index] for index in indices]


def target_only_prompt(row: dict[str, str]) -> str:
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"During the first two seconds, {row['receiver']} is {row['surface_condition']}. "
        "No water droplet is visible during these first two seconds. "
        "Then exactly one large clear water droplet enters from the top of the frame and "
        "stops while still visibly separated from the receiver. It remains suspended, "
        "never makes contact, and never changes the receiver. "
        "The camera and lighting remain fixed throughout."
    )


def unrelated_footprint_prompt(row: dict[str, str]) -> str:
    if row["family"] == "liquid_surface":
        event = (
            "Then a gentle breeze from outside the frame begins and creates several small "
            "natural ripples across the water surface. No falling water droplet appears at "
            "any time."
        )
    else:
        event = (
            "Then several small clear condensation beads gradually become visible on the "
            "surface. No falling water droplet appears and nothing impacts the surface."
        )
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"During the first two seconds, {row['receiver']} is {row['surface_condition']}. "
        f"{event} The camera and lighting remain fixed throughout."
    )


def clean_control_prompt(row: dict[str, str]) -> str:
    return (
        "A realistic fixed-camera macro video in one continuous shot. "
        f"For the entire clip, {row['receiver']} remains {row['surface_condition']}. "
        "Nothing enters, touches, or changes the receiver. The camera and lighting remain "
        "fixed throughout."
    )


def expected_states(condition: str) -> tuple[str, str, str, str]:
    states = {
        "explicit_causal": ("yes", "yes", "no", "no"),
        "implicit_causal": ("yes", "yes", "no", "no"),
        "target_only": ("yes", "no", "no", "no"),
        "unrelated_footprint": ("no", "yes", "no", "yes"),
        "clean_control": ("no", "no", "no", "no"),
    }
    return states[condition]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()

    source_rows = read_csv(root / "data/waterdrop_prompt_bank_v2_simple.csv")
    explicit_screen = read_csv(root / "data/waterdrop_prompt_bank_v2_auto_screen.csv")
    implicit_rows = read_csv(root / "data/waterdrop_implicit_pilot100.csv")
    implicit_screen = read_csv(root / "data/waterdrop_implicit_pilot100_auto_screen.csv")
    source_by_id = {row["scene_id"]: row for row in source_rows}
    implicit_by_source = {row["source_scene_id"]: row for row in implicit_rows}
    explicit_status = {row["scene_id"]: row["auto_status"] for row in explicit_screen}
    implicit_status = {
        row["source_scene_id"]: row["auto_status"] for row in implicit_screen
    }

    eligible: dict[str, list[dict[str, str]]] = {
        "liquid_surface": [],
        "hard_surface": [],
    }
    for source_id, implicit_row in implicit_by_source.items():
        source_row = source_by_id[source_id]
        if (
            explicit_status.get(source_id) == "candidate"
            and implicit_status.get(source_id) == "candidate"
        ):
            eligible[source_row["family"]].append(source_row)
    for family in eligible:
        eligible[family].sort(key=lambda row: row["receiver_id"])

    selected = evenly_select(eligible["liquid_surface"], 10) + evenly_select(
        eligible["hard_surface"], 10
    )
    rows: list[dict[str, str]] = []
    for receiver_index, source_row in enumerate(selected):
        implicit_row = implicit_by_source[source_row["scene_id"]]
        prompts = {
            "explicit_causal": source_row["prompt"],
            "implicit_causal": implicit_row["prompt"],
            "target_only": target_only_prompt(source_row),
            "unrelated_footprint": unrelated_footprint_prompt(source_row),
            "clean_control": clean_control_prompt(source_row),
        }
        for condition in CONDITIONS:
            base_target, base_footprint, erased_target, erased_footprint = expected_states(
                condition
            )
            rows.append(
                {
                    "scene_id": f"wdfive{len(rows):04d}",
                    "receiver_group_id": f"wdreceiver{receiver_index:02d}",
                    "source_scene_id": source_row["scene_id"],
                    "receiver_id": source_row["receiver_id"],
                    "family": source_row["family"],
                    "receiver": source_row["receiver"],
                    "condition": condition,
                    "split": "frozen_test",
                    "causal_footprint": source_row["causal_footprint"],
                    "expected_base_target": base_target,
                    "expected_base_footprint": base_footprint,
                    "expected_erased_target": erased_target,
                    "expected_erased_footprint": erased_footprint,
                    "erase_instruction": "Remove the water droplet.",
                    "fixed_seed": str(FIXED_SEED),
                    "prompt": prompts[condition],
                }
            )

    if len(rows) != 100:
        raise ValueError(f"expected 100 rows, found {len(rows)}")
    if Counter(row["condition"] for row in rows) != Counter({name: 20 for name in CONDITIONS}):
        raise ValueError("condition counts are not balanced")
    if Counter(row["family"] for row in rows) != Counter(
        {"liquid_surface": 50, "hard_surface": 50}
    ):
        raise ValueError("family counts are not balanced")
    if len({row["receiver_id"] for row in rows}) != 20:
        raise ValueError("expected 20 held-out receivers")

    csv_path = root / "data/waterdrop_five_condition_test100.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = root / "prompts/waterdrop_five_condition_test100.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("# Frozen waterdrop five-condition evaluation set.\n")
        handle.write("# Format: <prompt> | <target> | <hidden expected effect>\n\n")
        for row in rows:
            handle.write(
                f"{row['prompt']} | single falling water droplet | {row['causal_footprint']}\n"
            )

    shards: list[list[dict[str, str]]] = [[] for _ in range(SHARD_COUNT)]
    for index, row in enumerate(rows):
        shards[index % SHARD_COUNT].append(row)
    manifest_rows: list[dict[str, str]] = []
    for shard, shard_rows in enumerate(shards):
        path = root / f"prompts/waterdrop_five_condition_test100_shard_{shard}.txt"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# Frozen five-condition test, shard {shard}/{SHARD_COUNT - 1}; "
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
                        **{key: value for key, value in row.items() if key != "prompt"},
                        "shard": str(shard),
                        "shard_index": str(shard_index),
                    }
                )
    manifest_path = root / "data/waterdrop_five_condition_test100_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    held_out = root / "data/waterdrop_five_condition_test100_heldout_receivers.txt"
    held_out.write_text(
        "\n".join(row["receiver_id"] for row in selected) + "\n", encoding="utf-8"
    )
    print("Wrote frozen test100: 20 receivers x 5 conditions; shards=[25, 25, 25, 25]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
