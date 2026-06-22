#!/usr/bin/env python3
"""Build a unified JSONL benchmark item file from manifests and review CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASELINE_OUTPUT_FIELDS = [
    "run_id",
    "prompt_index",
    "baseline",
    "seed",
    "video_path",
    "video_exists",
    "video_bytes",
    "target_visible",
    "causal_effect_visible",
    "causeless_effect",
    "video_quality",
    "usable_for_claim",
    "failure_mode",
    "notes",
]


def parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_source_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split(",", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("--source must be formatted as name,manifest.json,summary.csv")
    return parts[0], Path(parts[1]), Path(parts[2])


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"{path}: missing list field 'items'")
    return data


def load_source_prompts(manifest: dict[str, Any], manifest_path: Path) -> list[str]:
    prompt_path_value = manifest.get("output_prompts")
    if not prompt_path_value:
        return []

    prompt_path = Path(prompt_path_value)
    if not prompt_path.exists() and not prompt_path.is_absolute():
        prompt_path = manifest_path.parent / prompt_path
    if not prompt_path.exists():
        return []

    prompts: list[str] = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        prompt, *_ = stripped.rsplit(" | ", 2)
        prompts.append(prompt)
    return prompts


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted({"pair_id", "baseline"} - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def row_to_output(row: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id", ""),
        "prompt_index": parse_int(row.get("prompt_index")),
        "baseline": row.get("baseline", ""),
        "seed": parse_int(row.get("seed")),
        "video_path": row.get("video_path", ""),
        "video_exists": parse_bool(row.get("video_exists")),
        "video_bytes": parse_int(row.get("video_bytes")),
        "target_visible": row.get("target_visible", ""),
        "causal_effect_visible": row.get("causal_effect_visible", ""),
        "causeless_effect": row.get("causeless_effect", ""),
        "video_quality": row.get("video_quality", ""),
        "usable_for_claim": row.get("usable_for_claim", ""),
        "failure_mode": row.get("failure_mode", ""),
        "notes": row.get("notes", ""),
    }


def clean_reference_from(item: dict[str, Any], clean_row: dict[str, str] | None) -> dict[str, Any]:
    reference = {
        "clean_source_valid": item.get("clean_source_valid", ""),
        "clean_source_notes": item.get("clean_source_notes", ""),
        "run_id": "",
        "prompt_index": None,
        "seed": None,
        "video_path": "",
        "video_exists": None,
        "video_bytes": None,
        "video_quality": "",
        "notes": "",
    }
    if clean_row:
        output = row_to_output(clean_row)
        reference.update(
            {
                "run_id": output["run_id"],
                "prompt_index": output["prompt_index"],
                "seed": output["seed"],
                "video_path": output["video_path"],
                "video_exists": output["video_exists"],
                "video_bytes": output["video_bytes"],
                "video_quality": output["video_quality"],
                "notes": output["notes"],
            }
        )
    return reference


def source_prompt_for(item: dict[str, Any], prompts: list[str], item_index: int) -> str:
    if item.get("prompt"):
        return str(item["prompt"])
    if item.get("source_prompt"):
        return str(item["source_prompt"])
    if item_index < len(prompts):
        return prompts[item_index]
    return ""


def build_items(source_name: str, manifest_path: Path, summary_path: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    prompts = load_source_prompts(manifest, manifest_path)
    summary_rows = load_summary_rows(summary_path)

    rows_by_pair: dict[str, list[dict[str, str]]] = {}
    for row in summary_rows:
        rows_by_pair.setdefault(row["pair_id"], []).append(row)

    manifest_pair_ids = [item["pair_id"] for item in manifest["items"]]
    duplicates = sorted({pair_id for pair_id in manifest_pair_ids if manifest_pair_ids.count(pair_id) > 1})
    if duplicates:
        raise ValueError(f"{manifest_path}: duplicate pair_id(s): {', '.join(duplicates)}")

    unknown_pairs = sorted(set(rows_by_pair) - set(manifest_pair_ids))
    if unknown_pairs:
        raise ValueError(f"{summary_path}: summary contains pair_id(s) not in manifest: {', '.join(unknown_pairs)}")

    items = []
    for item_index, manifest_item in enumerate(manifest["items"]):
        pair_id = manifest_item["pair_id"]
        rows = rows_by_pair.get(pair_id, [])
        clean_rows = [row for row in rows if row.get("baseline") == "clean_reference"]
        baseline_rows = [row for row in rows if row.get("baseline") != "clean_reference"]
        if len(clean_rows) > 1:
            raise ValueError(f"{summary_path}: multiple clean_reference rows for {pair_id}")

        items.append(
            {
                "item_id": f"{source_name}:{pair_id}",
                "source_name": source_name,
                "source_index": manifest_item.get("source_index"),
                "pair_id": pair_id,
                "mechanism_type": manifest_item.get("mechanism_type", ""),
                "temporal_type": manifest_item.get("temporal_type", ""),
                "target_concept": manifest_item.get("target_concept", ""),
                "expected_effect": manifest_item.get("causal_footprint", manifest_item.get("expected_effect", "")),
                "source_prompt": source_prompt_for(manifest_item, prompts, item_index),
                "counterfactual_prompt": manifest_item.get("counterfactual_prompt", ""),
                "control_prompt": manifest_item.get("control_prompt", ""),
                "clean_reference": clean_reference_from(manifest_item, clean_rows[0] if clean_rows else None),
                "baseline_outputs": [row_to_output(row) for row in baseline_rows],
            }
        )
    return items


def write_jsonl(items: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        all_items: list[dict[str, Any]] = []
        seen_item_ids: set[str] = set()
        for source_name, manifest_path, summary_path in args.source:
            source_items = build_items(source_name, manifest_path, summary_path)
            for item in source_items:
                if item["item_id"] in seen_item_ids:
                    raise ValueError(f"duplicate item_id: {item['item_id']}")
                seen_item_ids.add(item["item_id"])
            all_items.extend(source_items)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    write_jsonl(all_items, args.output)
    total_outputs = sum(len(item["baseline_outputs"]) for item in all_items)
    print(f"Wrote {len(all_items)} benchmark items and {total_outputs} baseline outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
