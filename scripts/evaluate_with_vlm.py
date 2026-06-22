#!/usr/bin/env python3
"""Build dry-run VLM request payloads from contact-sheet evaluator inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_INPUT_FIELDS = {
    "output_id",
    "item_id",
    "baseline",
    "video_path",
    "sheet_path",
    "sheet_exists",
    "sheet_error",
    "target_concept",
    "expected_effect",
    "source_prompt",
}

RESPONSE_SCHEMA = {
    "target_absent": ["yes", "no", "partial"],
    "effect_visible": ["yes", "no", "partial"],
    "quality_ok": ["yes", "no"],
    "pred_label": ["strict_leakage", "borderline", "target_leakage", "other_failure"],
    "confidence": "float in [0.0, 1.0]",
    "reason": "short explanation",
}


def read_inputs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(REQUIRED_INPUT_FIELDS - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def prompt_for(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are judging a video concept-erasure result from a 5-frame contact sheet.",
            f"Target concept to erase: {row['target_concept']}",
            f"Expected downstream causal effect: {row['expected_effect']}",
            f"Original source prompt: {row.get('source_prompt', '')}",
            "",
            "Inspect the frames in temporal order from left to right.",
            "Answer these questions: whether the target concept is absent, whether the expected effect is visible, and whether the image evidence is good enough to judge.",
            "Then choose the final prediction label using the provided response schema.",
            "Return only valid JSON. Keep the reason short and visual.",
        ]
    )


def payload_for(row: dict[str, str]) -> dict[str, object]:
    return {
        "output_id": row["output_id"],
        "item_id": row["item_id"],
        "baseline": row["baseline"],
        "video_path": row["video_path"],
        "image_path": row["sheet_path"],
        "sheet_available": row["sheet_exists"] == "true",
        "sheet_error": row.get("sheet_error", ""),
        "target_concept": row["target_concept"],
        "expected_effect": row["expected_effect"],
        "prompt": prompt_for(row),
        "response_schema": RESPONSE_SCHEMA,
    }


def filter_rows(rows: list[dict[str, str]], *, include_missing: bool, limit: int | None) -> list[dict[str, str]]:
    filtered = [row for row in rows if include_missing or row.get("sheet_exists") == "true"]
    if limit is not None:
        return filtered[:limit]
    return filtered


def write_jsonl(payloads: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is implemented; real VLM API calls will use the same payload schema later")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    try:
        rows = read_inputs(args.inputs)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    selected = filter_rows(rows, include_missing=args.include_missing, limit=args.limit)
    payloads = [payload_for(row) for row in selected]
    write_jsonl(payloads, args.output_jsonl)
    skipped = len(rows) - len(filter_rows(rows, include_missing=args.include_missing, limit=None))
    print(f"Wrote {len(payloads)} dry-run payloads to {args.output_jsonl} (skipped_missing={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
