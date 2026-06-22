#!/usr/bin/env python3
"""Calibrate evaluator predictions against human causal-footprint labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LABELS = ["strict_leakage", "borderline", "target_leakage", "other_failure"]
PREDICTION_FIELDS = {
    "item_id",
    "baseline",
    "video_path",
    "target_absent",
    "effect_visible",
    "quality_ok",
    "pred_label",
    "confidence",
    "reason",
}
GOLD_FIELDS = {"item_id", "baseline", "human_label"}
LABEL_METRIC_FIELDS = ["label", "support", "tp", "fp", "fn", "precision", "recall", "f1"]


def read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = sorted(required_fields - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        return list(reader)


def output_key(row: dict[str, str]) -> str:
    return f"{row.get('item_id', '')}::{row.get('baseline', '')}"


def index_rows(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = output_key(row)
        if key in indexed:
            raise ValueError(f"{path}: duplicate row for {key}")
        indexed[key] = row
    return indexed


def validate_labels(rows: list[dict[str, str]], field: str, path: Path) -> None:
    allowed = set(LABELS)
    invalid = sorted({row.get(field, "") for row in rows if row.get(field, "") not in allowed})
    if invalid:
        raise ValueError(f"{path}: invalid {field} value(s): {', '.join(invalid)}")


def join_gold_predictions(
    gold_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    prediction_path: Path,
    *,
    allow_partial: bool = False,
) -> list[tuple[str, str]]:
    predictions_by_key = index_rows(prediction_rows, prediction_path)
    joined = []
    missing = []
    for gold in gold_rows:
        key = output_key(gold)
        prediction = predictions_by_key.get(key)
        if prediction is None:
            missing.append(key)
        else:
            joined.append((gold["human_label"], prediction["pred_label"]))
    if missing and not allow_partial:
        raise ValueError(f"missing predictions for: {', '.join(missing[:10])}")

    extra = sorted(set(predictions_by_key) - {output_key(row) for row in gold_rows})
    if extra:
        raise ValueError(f"predictions without gold rows: {', '.join(extra[:10])}")
    return joined


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def label_metrics(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows = []
    for label in LABELS:
        support = sum(gold == label for gold, _ in pairs)
        tp = sum(gold == label and pred == label for gold, pred in pairs)
        fp = sum(gold != label and pred == label for gold, pred in pairs)
        fn = sum(gold == label and pred != label for gold, pred in pairs)
        precision = safe_rate(tp, tp + fp)
        recall = safe_rate(tp, tp + fn)
        f1 = f1_score(precision, recall)
        rows.append(
            {
                "label": label,
                "support": str(support),
                "tp": str(tp),
                "fp": str(fp),
                "fn": str(fn),
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
            }
        )
    return rows


def binary_metrics(pairs: list[tuple[str, str]], positives: set[str]) -> dict[str, float | int]:
    tp = sum(gold in positives and pred in positives for gold, pred in pairs)
    fp = sum(gold not in positives and pred in positives for gold, pred in pairs)
    fn = sum(gold in positives and pred not in positives for gold, pred in pairs)
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def confusion_rows(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for gold, pred in pairs:
        counts[(gold, pred)] += 1
    rows = []
    for gold in LABELS:
        for pred in LABELS:
            count = counts[(gold, pred)]
            if count:
                rows.append({"gold_label": gold, "pred_label": pred, "count": str(count)})
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    total: int,
    label_rows: list[dict[str, str]],
    strict_binary: dict[str, float | int],
    relaxed_binary: dict[str, float | int],
    confusion: list[dict[str, str]],
) -> None:
    macro_f1 = safe_rate(sum(float(row["f1"]) for row in label_rows), len(label_rows))
    lines = [
        "# Evaluator Calibration Summary",
        "",
        f"- Matched predictions: {total}",
        f"- Strict leakage binary F1: {strict_binary['f1']:.4f}",
        f"- Relaxed leakage binary F1: {relaxed_binary['f1']:.4f}",
        f"- Macro F1: {macro_f1:.4f}",
        "",
        "## Label Metrics",
        "",
        "| Label | Support | TP | FP | FN | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in label_rows:
        lines.append(
            "| "
            f"{row['label']} | {row['support']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']} | {row['recall']} | {row['f1']} |"
        )
    lines.extend(["", "## Confusion Matrix", "", "| Gold | Predicted | Count |", "| --- | --- | ---: |"])
    for row in confusion:
        lines.append(f"| {row['gold_label']} | {row['pred_label']} | {row['count']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def calibrate(gold_path: Path, prediction_path: Path, output_dir: Path, *, allow_partial: bool = False) -> int:
    gold_rows = read_csv(gold_path, GOLD_FIELDS)
    prediction_rows = read_csv(prediction_path, PREDICTION_FIELDS)
    validate_labels(gold_rows, "human_label", gold_path)
    validate_labels(prediction_rows, "pred_label", prediction_path)

    pairs = join_gold_predictions(gold_rows, prediction_rows, prediction_path, allow_partial=allow_partial)
    label_rows = label_metrics(pairs)
    confusion = confusion_rows(pairs)
    strict_binary = binary_metrics(pairs, {"strict_leakage"})
    relaxed_binary = binary_metrics(pairs, {"strict_leakage", "borderline"})

    write_csv(output_dir / "calibration_metrics_by_label.csv", LABEL_METRIC_FIELDS, label_rows)
    write_csv(output_dir / "calibration_confusion_matrix.csv", ["gold_label", "pred_label", "count"], confusion)
    write_summary(output_dir / "calibration_metrics_summary.md", len(pairs), label_rows, strict_binary, relaxed_binary, confusion)
    return len(pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        count = calibrate(args.gold, args.predictions, args.output_dir, allow_partial=args.allow_partial)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(f"Calibrated {count} predictions into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
