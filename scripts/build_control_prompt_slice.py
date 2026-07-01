#!/usr/bin/env python3
"""Export causal-footprint control specs into prompt and manifest files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "control_id",
    "source_name",
    "source_pair_id",
    "source_baseline",
    "mechanism_type",
    "target_concept",
    "expected_effect",
    "control_type",
    "prompt",
    "purpose",
}


def read_specs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        missing = sorted(field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip())
        if missing:
            raise ValueError(f"{path}:{line_no}: missing required field(s): {', '.join(missing)}")
        control_id = str(row["control_id"])
        if control_id in seen:
            raise ValueError(f"duplicate control_id: {control_id}")
        seen.add(control_id)
        rows.append(row)
    return rows


def prompt_line(row: dict[str, Any]) -> str:
    return f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}"


def manifest_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "pair_id": row["control_id"],
        "control_id": row["control_id"],
        "source_name": row["source_name"],
        "source_pair_id": row["source_pair_id"],
        "source_baseline": row["source_baseline"],
        "mechanism_type": row["mechanism_type"],
        "target_concept": row["target_concept"],
        "causal_footprint": row["expected_effect"],
        "expected_effect": row["expected_effect"],
        "control_type": row["control_type"],
        "source_prompt": row["prompt"],
        "prompt": row["prompt"],
        "purpose": row["purpose"],
    }


def write_prompts(rows: list[dict[str, Any]], output_path: Path, slice_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Exported control slice: {slice_name}",
        f"# Source format: {', '.join(sorted(REQUIRED_FIELDS))}",
        "# Format: <prompt> | <target> | <effect>",
        "",
        *[prompt_line(row) for row in rows],
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    specs_path: Path,
    prompts_path: Path,
    slice_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slice_name": slice_name,
        "specs": str(specs_path),
        "output_prompts": str(prompts_path),
        "count": len(rows),
        "items": [manifest_item(row, index) for index, row in enumerate(rows)],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-prompts", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--slice-name", default="controls")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        rows = read_specs(args.specs)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    write_prompts(rows, args.output_prompts, args.slice_name)
    if args.output_manifest:
        write_manifest(
            rows,
            output_path=args.output_manifest,
            specs_path=args.specs,
            prompts_path=args.output_prompts,
            slice_name=args.slice_name,
        )
    print(f"Exported {len(rows)} control prompts to {args.output_prompts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
