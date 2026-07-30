#!/usr/bin/env python3
"""Select a balanced 40-prompt capability gate from waterdrop prompt bank v1."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


FAMILIES = (
    "liquid_surface",
    "hard_surface",
    "absorbent_surface",
    "granular_surface",
)
RECEIVER_INDICES = (0, 5, 11, 16, 22, 27, 33, 38, 44, 49)
FIXED_SEED = 9000


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--bank", type=Path, default=Path("data/waterdrop_prompt_bank_v1.csv")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("data/waterdrop_prompt_gate40_v1.csv")
    )
    parser.add_argument(
        "--part-a", type=Path, default=Path("prompts/waterdrop_prompt_gate40_v1_part_a.txt")
    )
    parser.add_argument(
        "--part-b", type=Path, default=Path("prompts/waterdrop_prompt_gate40_v1_part_b.txt")
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    bank_path = args.bank if args.bank.is_absolute() else root / args.bank
    bank = read_rows(bank_path)
    by_receiver: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in bank:
        receiver_index = int(row["receiver_id"].rsplit("_", 1)[1])
        by_receiver.setdefault((row["family"], receiver_index), []).append(row)

    selected: list[dict[str, str]] = []
    for family in FAMILIES:
        family_rows = []
        for position, receiver_index in enumerate(RECEIVER_INDICES):
            candidates = sorted(
                by_receiver[(family, receiver_index)], key=lambda item: int(item["variant"])
            )
            family_rows.append(candidates[position % len(candidates)])
        for position, row in enumerate(family_rows):
            selected.append(
                {
                    "gate_id": "",
                    "shard": "part_a" if position < 5 else "part_b",
                    "fixed_seed": str(FIXED_SEED),
                    **row,
                }
            )

    selected.sort(key=lambda row: (row["shard"], FAMILIES.index(row["family"])))
    for index, row in enumerate(selected):
        row["gate_id"] = f"wdg{index:03d}"

    counts = Counter((row["shard"], row["family"]) for row in selected)
    expected = {(shard, family): 5 for shard in ("part_a", "part_b") for family in FAMILIES}
    if len(selected) != 40 or counts != expected:
        raise ValueError(f"unbalanced gate: {counts}")
    if len({row["prompt"] for row in selected}) != 40:
        raise ValueError("duplicate gate prompts found")

    csv_path = args.csv if args.csv.is_absolute() else root / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    for shard, output_arg in (("part_a", args.part_a), ("part_b", args.part_b)):
        output = output_arg if output_arg.is_absolute() else root / output_arg
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write(f"# Waterdrop prompt gate40 v1, {shard}; fixed seed {FIXED_SEED}.\n")
            handle.write("# Format: <prompt> | <target> | <effect>\n\n")
            for row in selected:
                if row["shard"] == shard:
                    handle.write(
                        f"{row['prompt']} | single falling water droplet | "
                        f"{row['causal_footprint']}\n"
                    )
    print(f"Wrote {len(selected)} prompts with balanced counts: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
