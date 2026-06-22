from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_calibrate_evaluator_computes_label_and_binary_metrics(tmp_path):
    gold = tmp_path / "gold.csv"
    predictions = tmp_path / "predictions.csv"
    out_dir = tmp_path / "out"
    gold_fields = [
        "item_id",
        "baseline",
        "video_path",
        "mechanism_type",
        "human_label",
    ]
    prediction_fields = [
        "item_id",
        "baseline",
        "video_path",
        "target_absent",
        "effect_visible",
        "quality_ok",
        "pred_label",
        "confidence",
        "reason",
    ]
    write_csv(
        gold,
        gold_fields,
        [
            {
                "item_id": "i1",
                "baseline": "negative_prompt",
                "video_path": "v1.mp4",
                "mechanism_type": "fluid_impact",
                "human_label": "strict_leakage",
            },
            {
                "item_id": "i2",
                "baseline": "negative_prompt",
                "video_path": "v2.mp4",
                "mechanism_type": "fracture_damage",
                "human_label": "borderline",
            },
            {
                "item_id": "i3",
                "baseline": "videoeraser",
                "video_path": "v3.mp4",
                "mechanism_type": "elastic_deformation",
                "human_label": "target_leakage",
            },
            {
                "item_id": "i4",
                "baseline": "videoeraser",
                "video_path": "v4.mp4",
                "mechanism_type": "field_mediated",
                "human_label": "other_failure",
            },
        ],
    )
    write_csv(
        predictions,
        prediction_fields,
        [
            {
                "item_id": "i1",
                "baseline": "negative_prompt",
                "video_path": "v1.mp4",
                "target_absent": "yes",
                "effect_visible": "yes",
                "quality_ok": "yes",
                "pred_label": "strict_leakage",
                "confidence": "0.9",
                "reason": "matches",
            },
            {
                "item_id": "i2",
                "baseline": "negative_prompt",
                "video_path": "v2.mp4",
                "target_absent": "partial",
                "effect_visible": "yes",
                "quality_ok": "yes",
                "pred_label": "strict_leakage",
                "confidence": "0.6",
                "reason": "too aggressive",
            },
            {
                "item_id": "i3",
                "baseline": "videoeraser",
                "video_path": "v3.mp4",
                "target_absent": "no",
                "effect_visible": "yes",
                "quality_ok": "yes",
                "pred_label": "target_leakage",
                "confidence": "0.8",
                "reason": "target visible",
            },
            {
                "item_id": "i4",
                "baseline": "videoeraser",
                "video_path": "v4.mp4",
                "target_absent": "yes",
                "effect_visible": "no",
                "quality_ok": "no",
                "pred_label": "other_failure",
                "confidence": "0.7",
                "reason": "effect gone",
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "calibrate_evaluator.py"),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Calibrated 4 predictions" in result.stdout
    with (out_dir / "calibration_metrics_by_label.csv").open(newline="", encoding="utf-8") as handle:
        by_label = {row["label"]: row for row in csv.DictReader(handle)}

    assert by_label["strict_leakage"]["tp"] == "1"
    assert by_label["strict_leakage"]["fp"] == "1"
    assert by_label["strict_leakage"]["fn"] == "0"
    assert by_label["strict_leakage"]["precision"] == "0.5000"
    assert by_label["borderline"]["fn"] == "1"

    with (out_dir / "calibration_confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
        confusion = list(csv.DictReader(handle))
    assert {
        "gold_label": "borderline",
        "pred_label": "strict_leakage",
        "count": "1",
    } in confusion

    summary = (out_dir / "calibration_metrics_summary.md").read_text(encoding="utf-8")
    assert "Strict leakage binary F1: 0.6667" in summary
    assert "Relaxed leakage binary F1: 1.0000" in summary
    assert "Macro F1: 0.6667" in summary


def test_calibrate_evaluator_rejects_missing_prediction_rows(tmp_path):
    gold = tmp_path / "gold.csv"
    predictions = tmp_path / "predictions.csv"
    out_dir = tmp_path / "out"
    write_csv(
        gold,
        ["item_id", "baseline", "video_path", "human_label"],
        [
            {
                "item_id": "i1",
                "baseline": "negative_prompt",
                "video_path": "v1.mp4",
                "human_label": "strict_leakage",
            }
        ],
    )
    write_csv(
        predictions,
        [
            "item_id",
            "baseline",
            "video_path",
            "target_absent",
            "effect_visible",
            "quality_ok",
            "pred_label",
            "confidence",
            "reason",
        ],
        [],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "calibrate_evaluator.py"),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing predictions for: i1::negative_prompt" in result.stderr


def test_calibrate_evaluator_allows_partial_prediction_rows(tmp_path):
    gold = tmp_path / "gold.csv"
    predictions = tmp_path / "predictions.csv"
    out_dir = tmp_path / "out"
    write_csv(
        gold,
        ["item_id", "baseline", "video_path", "human_label"],
        [
            {
                "item_id": "i1",
                "baseline": "negative_prompt",
                "video_path": "v1.mp4",
                "human_label": "strict_leakage",
            },
            {
                "item_id": "i2",
                "baseline": "videoeraser",
                "video_path": "v2.mp4",
                "human_label": "target_leakage",
            },
        ],
    )
    write_csv(
        predictions,
        [
            "item_id",
            "baseline",
            "video_path",
            "target_absent",
            "effect_visible",
            "quality_ok",
            "pred_label",
            "confidence",
            "reason",
        ],
        [
            {
                "item_id": "i1",
                "baseline": "negative_prompt",
                "video_path": "v1.mp4",
                "target_absent": "yes",
                "effect_visible": "yes",
                "quality_ok": "yes",
                "pred_label": "strict_leakage",
                "confidence": "0.9",
                "reason": "matches",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "calibrate_evaluator.py"),
            "--gold",
            str(gold),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(out_dir),
            "--allow-partial",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Calibrated 1 predictions" in result.stdout
    summary = (out_dir / "calibration_metrics_summary.md").read_text(encoding="utf-8")
    assert "Matched predictions: 1" in summary
