#!/usr/bin/env python3
"""Export benchmark candidate pairs into generation prompt files."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CANDIDATES = Path("benchmarks/causal_footprint_v0/candidate_pairs.tsv")
DEFAULT_OUTPUT_PROMPTS = Path("prompts/causal_footprint_v0_accepted24.txt")
DEFAULT_STATUS = "accepted_v0_slice"

REQUIRED_COLUMNS = {
    "pair_id",
    "target_concept",
    "causal_footprint",
    "mechanism_type",
    "temporal_type",
    "exclusivity_score",
    "counterfactual_clarity",
    "generatability_score",
    "erasure_targetability",
    "status",
    "source_prompt",
    "counterfactual_prompt",
    "control_prompt",
}


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)

    seen: set[str] = set()
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)
    return rows


def filter_rows(rows: list[dict[str, str]], status: str) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == status]


def read_clean_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted({"pair_id", "clean_source_valid"} - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)

    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in labels:
            raise ValueError(f"duplicate clean-label pair_id: {pair_id}")
        labels[pair_id] = row
    return labels


def filter_by_clean_labels(
    rows: list[dict[str, str]],
    labels: dict[str, dict[str, str]],
    allowed_values: list[str],
) -> list[dict[str, str]]:
    allowed = set(allowed_values)
    return [row for row in rows if labels.get(row["pair_id"], {}).get("clean_source_valid") in allowed]


def prompt_line(row: dict[str, str]) -> str:
    return f"{row['source_prompt']} | {row['target_concept']} | {row['causal_footprint']}"


def write_prompts(rows: list[dict[str, str]], output_path: Path, status: str, clean_source_valid: list[str] | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exported from candidate_pairs.tsv",
        f"# Status filter: {status}",
        *([f"# Clean-source label filter: {', '.join(clean_source_valid)}"] if clean_source_valid else []),
        "# Format: <prompt> | <target> | <effect>",
        "",
        *[prompt_line(row) for row in rows],
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    rows: list[dict[str, str]],
    output_path: Path,
    candidates: Path,
    prompts: Path,
    status: str,
    clean_labels: Path | None,
    clean_source_valid: list[str] | None,
    label_map: dict[str, dict[str, str]] | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for row in rows:
        label = label_map.get(row["pair_id"], {}) if label_map else {}
        item = {
            "pair_id": row["pair_id"],
            "target_concept": row["target_concept"],
            "causal_footprint": row["causal_footprint"],
            "mechanism_type": row["mechanism_type"],
            "temporal_type": row["temporal_type"],
            "exclusivity_score": int(row["exclusivity_score"]),
            "counterfactual_clarity": int(row["counterfactual_clarity"]),
            "generatability_score": int(row["generatability_score"]),
            "erasure_targetability": int(row["erasure_targetability"]),
            "counterfactual_prompt": row["counterfactual_prompt"],
            "control_prompt": row["control_prompt"],
        }
        if label:
            item["clean_source_valid"] = label.get("clean_source_valid", "")
            item["clean_source_notes"] = label.get("notes", "")
        items.append(item)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": str(candidates),
        "output_prompts": str(prompts),
        "status": status,
        "count": len(rows),
        "items": items,
    }
    if clean_labels is not None:
        manifest["clean_labels"] = str(clean_labels)
    if clean_source_valid is not None:
        manifest["clean_source_valid"] = clean_source_valid
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-prompts", type=Path, default=DEFAULT_OUTPUT_PROMPTS)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--status", default=DEFAULT_STATUS)
    parser.add_argument("--clean-labels", type=Path)
    parser.add_argument("--clean-source-valid", action="append")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.clean_source_valid and args.clean_labels is None:
        parser.error("--clean-source-valid requires --clean-labels")

    try:
        rows = filter_rows(read_candidates(args.candidates), args.status)
        label_map = read_clean_labels(args.clean_labels) if args.clean_labels else None
        if args.clean_source_valid:
            rows = filter_by_clean_labels(rows, label_map or {}, args.clean_source_valid)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    write_prompts(rows, args.output_prompts, args.status, args.clean_source_valid)
    if args.output_manifest is not None:
        write_manifest(
            rows,
            args.output_manifest,
            args.candidates,
            args.output_prompts,
            args.status,
            args.clean_labels,
            args.clean_source_valid,
            label_map,
        )
    print(f"Exported {len(rows)} prompts to {args.output_prompts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
