#!/usr/bin/env python3
"""Extract unresolved rows into a human adjudication queue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def build_queue(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    queue: list[dict[str, str]] = []
    for row in rows:
        if row.get("adjudicated_label", ""):
            continue
        output = dict(row)
        output["review_status"] = "needs_human_review"
        queue.append(output)
    return queue


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = build_queue(read_csv(args.manifest))
        write_csv(args.output, rows)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {len(rows)} human adjudication queue rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
