#!/usr/bin/env python3
"""Export a model-specific v2 clean-source slice for baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def source_index_from_sample_id(sample_id: str) -> int:
    match = re.fullmatch(r"case_(\d+)", sample_id.strip())
    if not match:
        raise ValueError(f"unsupported sample_id format: {sample_id!r}")
    return int(match.group(1))


def clean_generation_by_source_index(path: Path) -> dict[int, dict[str, Any]]:
    data = read_json(path)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing list field 'items'")
    by_source: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(items):
        source_index = int(item.get("source_prompt_index", item.get("index", fallback_index)))
        if source_index in by_source:
            raise ValueError(f"{path}: duplicate clean generation source index {source_index}")
        by_source[source_index] = item
    return by_source


def candidate_items(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing list field 'items'")
    return items


def clean_source_flag(row: dict[str, str]) -> str:
    for field in ("human_clean_source_valid", "clean_source_valid", "rule_clean_source_candidate"):
        value = row.get(field, "").strip().lower()
        if value:
            return value
    return ""


def yes_prediction_rows(path: Path, *, accepted_value: str) -> list[tuple[int, dict[str, str]]]:
    rows = read_csv(path)
    selected: list[tuple[int, dict[str, str]]] = []
    seen: set[int] = set()
    for row in rows:
        sample_id = row.get("sample_id") or row.get("prompt_id")
        if not sample_id:
            raise ValueError(f"{path}: clean prediction row is missing sample_id/prompt_id")
        source_index = source_index_from_sample_id(sample_id)
        if source_index in seen:
            raise ValueError(f"{path}: duplicate clean prediction source index {source_index}")
        seen.add(source_index)
        if clean_source_flag(row) == accepted_value:
            selected.append((source_index, row))
    return selected


def score_value(item: dict[str, Any], key: str) -> Any:
    scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
    return item.get(key, scores.get(key, ""))


def exported_item(
    *,
    slice_index: int,
    source_index: int,
    candidate: dict[str, Any],
    clean_generation: dict[str, Any],
    clean_prediction: dict[str, str],
    clean_note: str,
) -> dict[str, Any]:
    expected_effect = str(candidate.get("causal_footprint") or candidate.get("expected_effect") or "")
    return {
        "pair_id": str(candidate.get("pair_id", "")),
        "target_concept": str(candidate.get("target_concept", "")),
        "causal_footprint": expected_effect,
        "mechanism_type": str(candidate.get("mechanism_type", "")),
        "temporal_type": str(candidate.get("temporal_type", "")),
        "exclusivity_score": score_value(candidate, "exclusivity_score"),
        "counterfactual_clarity": score_value(candidate, "counterfactual_clarity"),
        "generatability_score": score_value(candidate, "generatability_score"),
        "erasure_targetability": score_value(candidate, "erasure_targetability"),
        "counterfactual_prompt": str(candidate.get("counterfactual_prompt", "")),
        "control_prompt": str(candidate.get("control_prompt", "")),
        "clean_source_valid": "yes",
        "clean_source_notes": clean_note,
        "slice_index": slice_index,
        "source_index": str(source_index),
        "source_prompt": str(candidate.get("source_prompt") or candidate.get("prompt") or clean_generation.get("prompt", "")),
        "clean_video_path": str(clean_generation.get("video_path", "")),
        "clean_prompt_id": clean_prediction.get("sample_id") or clean_prediction.get("prompt_id") or f"case_{source_index:03d}",
        "clean_vlm_confidence": clean_prediction.get("confidence", ""),
        "clean_vlm_reason": clean_prediction.get("reason", ""),
    }


def prompt_line(item: dict[str, Any]) -> str:
    return f"{item['source_prompt']} | {item['target_concept']} | {item['causal_footprint']}"


def write_prompts(path: Path, items: list[dict[str, Any]], *, slice_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {slice_name}",
        "# Format: <prompt> | <target> | <expected_effect>",
        "# Exported from model-specific clean-source screening.",
        "",
    ]
    lines.extend(prompt_line(item) for item in items)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    slice_name: str,
    output_prompts: Path,
    candidate_manifest: Path,
    clean_predictions: Path,
    clean_generation_manifest: Path,
    items: list[dict[str, Any]],
    accepted_value: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slice_name": slice_name,
        "output_prompts": str(output_prompts),
        "status": "model_specific_clean_source_slice",
        "count": len(items),
        "items": items,
        "clean_label_source": str(clean_predictions),
        "clean_generation_manifest": str(clean_generation_manifest),
        "source_manifest": str(candidate_manifest),
        "clean_source_valid": [accepted_value],
        "manifest_role": "model_specific_enriched_clean_source_manifest_for_baseline_review",
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_slice(args: argparse.Namespace) -> int:
    candidates = candidate_items(args.candidate_manifest)
    clean_by_source = clean_generation_by_source_index(args.clean_generation_manifest)
    selected = yes_prediction_rows(args.clean_predictions, accepted_value=args.accepted_value)
    items: list[dict[str, Any]] = []
    for slice_index, (source_index, prediction) in enumerate(selected):
        if source_index >= len(candidates):
            raise ValueError(
                f"{args.clean_predictions}: source index {source_index} exceeds candidate count {len(candidates)}"
            )
        clean_generation = clean_by_source.get(source_index)
        if clean_generation is None:
            raise ValueError(f"{args.clean_generation_manifest}: missing clean generation for source index {source_index}")
        items.append(
            exported_item(
                slice_index=slice_index,
                source_index=source_index,
                candidate=candidates[source_index],
                clean_generation=clean_generation,
                clean_prediction=prediction,
                clean_note=args.clean_label_source_note,
            )
        )

    write_prompts(args.output_prompts, items, slice_name=args.slice_name)
    write_manifest(
        args.output_manifest,
        slice_name=args.slice_name,
        output_prompts=args.output_prompts,
        candidate_manifest=args.candidate_manifest,
        clean_predictions=args.clean_predictions,
        clean_generation_manifest=args.clean_generation_manifest,
        items=items,
        accepted_value=args.accepted_value,
    )
    return len(items)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--clean-predictions", type=Path, required=True)
    parser.add_argument("--clean-generation-manifest", type=Path, required=True)
    parser.add_argument("--output-prompts", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--slice-name", required=True)
    parser.add_argument("--accepted-value", default="yes")
    parser.add_argument(
        "--clean-label-source-note",
        default="Clean-source validity follows chunked VLM prelabeling for this model-specific source run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        count = export_slice(args)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {count} clean-source items to {args.output_manifest}")
    print(f"Wrote prompts to {args.output_prompts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
