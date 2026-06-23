#!/usr/bin/env python3
"""Build a unified causal-footprint evaluation manifest CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BASE_FIELDS = [
    "sample_id",
    "item_id",
    "mechanism_id",
    "mechanism_type",
    "source_name",
    "target_concept",
    "causal_effect",
    "clean_prompt",
    "erasure_target",
    "baseline",
    "seed",
    "reference_video_path",
    "output_video_path",
    "reference_sheet_path",
    "contact_sheet_path",
    "expected_target_absent",
    "expected_effect_visible",
    "human_target_visible",
    "human_effect_visible",
    "human_separation_clear",
    "human_video_quality",
    "human_label",
    "human_failure_mode",
    "human_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    output_id = row.get("output_id", "")
    if output_id:
        return output_id
    return f"{row.get('item_id', '')}::{row.get('baseline', '')}"


def index_rows(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed = {}
    for row in rows:
        key = row_key(row)
        if key in indexed:
            raise ValueError(f"{path}: duplicate row for {key}")
        indexed[key] = row
    return indexed


def normalize_model_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("prediction model name cannot be empty")
    return normalized


def parse_prediction_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--prediction must use model=path")
    name, path_text = text.split("=", 1)
    return normalize_model_name(name), Path(path_text)


def load_predictions(prediction_args: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    predictions = {}
    for arg in prediction_args:
        model, path = parse_prediction_arg(arg)
        rows = read_csv(path)
        predictions[model] = index_rows(rows, path)
    return predictions


def prediction_fields(models: list[str]) -> list[str]:
    fields = []
    for model in models:
        fields.extend([f"{model}_label", f"{model}_confidence", f"{model}_reason", f"{model}_disagrees"])
    return fields


def build_manifest_rows(
    gold_rows: list[dict[str, str]],
    vlm_rows: list[dict[str, str]],
    predictions_by_model: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    gold_by_key = index_rows(gold_rows, Path("gold"))
    vlm_by_key = index_rows(vlm_rows, Path("vlm_inputs"))
    rows = []

    for key, gold in gold_by_key.items():
        vlm = vlm_by_key.get(key, {})
        row = {
            "sample_id": key,
            "item_id": gold.get("item_id", ""),
            "mechanism_id": gold.get("pair_id", ""),
            "mechanism_type": gold.get("mechanism_type", ""),
            "source_name": gold.get("source_name", ""),
            "target_concept": gold.get("target_concept", ""),
            "causal_effect": gold.get("expected_effect", ""),
            "clean_prompt": gold.get("source_prompt", ""),
            "erasure_target": gold.get("target_concept", ""),
            "baseline": gold.get("baseline", ""),
            "seed": gold.get("seed", ""),
            "reference_video_path": gold.get("reference_video_path", ""),
            "output_video_path": gold.get("video_path", ""),
            "reference_sheet_path": vlm.get("reference_sheet_path", ""),
            "contact_sheet_path": vlm.get("sheet_path", ""),
            "expected_target_absent": "yes",
            "expected_effect_visible": "yes",
            "human_target_visible": gold.get("target_visible", ""),
            "human_effect_visible": gold.get("causal_effect_visible", ""),
            "human_separation_clear": gold.get("causeless_effect", ""),
            "human_video_quality": gold.get("video_quality", ""),
            "human_label": gold.get("human_label", ""),
            "human_failure_mode": gold.get("failure_mode", ""),
            "human_notes": gold.get("notes", ""),
        }
        for model, predictions in predictions_by_model.items():
            prediction = predictions.get(key)
            label = prediction.get("pred_label", "") if prediction else ""
            row[f"{model}_label"] = label
            row[f"{model}_confidence"] = prediction.get("confidence", "") if prediction else ""
            row[f"{model}_reason"] = prediction.get("reason", "") if prediction else ""
            row[f"{model}_disagrees"] = "yes" if label and label != row["human_label"] else ("no" if label else "")
        rows.append(row)

    gold_keys = set(gold_by_key)
    for model, predictions in predictions_by_model.items():
        extra = sorted(set(predictions) - gold_keys)
        if extra:
            raise ValueError(f"{model}: predictions without gold rows: {', '.join(extra[:10])}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--vlm-inputs", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[], help="Optional model=prediction_csv")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        predictions = load_predictions(args.prediction)
        rows = build_manifest_rows(read_csv(args.gold), read_csv(args.vlm_inputs), predictions)
        fields = [*BASE_FIELDS, *prediction_fields(list(predictions))]
        write_csv(args.output, fields, rows)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Wrote {len(rows)} evaluation manifest rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
