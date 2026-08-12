#!/usr/bin/env python3
"""Combine dynamic water-impact erase rows with generic preservation rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--erase-manifest",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_manifest.csv"),
    )
    parser.add_argument(
        "--preserve-manifest",
        type=Path,
        default=Path("data/protocol_v1/wan_train_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"),
    )
    args = parser.parse_args()

    erase_rows = [row for row in read_rows(args.erase_manifest) if row["training_role"] == "erase"]
    preserve_rows = [
        row for row in read_rows(args.preserve_manifest) if row["training_role"] == "preserve"
    ]
    if not erase_rows or not preserve_rows:
        raise ValueError("Both erase and preserve rows are required")

    fields: list[str] = []
    for row in erase_rows + preserve_rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(erase_rows + preserve_rows)

    print(
        f"Wrote {args.output}: {len(erase_rows)} erase + "
        f"{len(preserve_rows)} preserve = {len(erase_rows) + len(preserve_rows)} rows"
    )


if __name__ == "__main__":
    main()
