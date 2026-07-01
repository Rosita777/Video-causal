from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compute_control_metrics_reports_validity_and_specificity(tmp_path):
    review = tmp_path / "control_review.csv"
    output_dir = tmp_path / "metrics"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "backbone",
                "mechanism_type",
                "control_type",
                "target_visible",
                "effect_visible",
                "alternative_cause_visible",
                "video_quality",
                "control_valid",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "row_index": "0",
                    "backbone": "cogvideox2b",
                    "mechanism_type": "fluid_impact",
                    "control_type": "no_cause",
                    "target_visible": "no",
                    "effect_visible": "no",
                    "alternative_cause_visible": "no",
                    "video_quality": "good",
                    "control_valid": "yes",
                },
                {
                    "row_index": "1",
                    "backbone": "cogvideox2b",
                    "mechanism_type": "fluid_impact",
                    "control_type": "effect_only",
                    "target_visible": "no",
                    "effect_visible": "yes",
                    "alternative_cause_visible": "no",
                    "video_quality": "good",
                    "control_valid": "yes",
                },
                {
                    "row_index": "2",
                    "backbone": "cogvideox2b",
                    "mechanism_type": "fluid_impact",
                    "control_type": "alternative_cause",
                    "target_visible": "no",
                    "effect_visible": "yes",
                    "alternative_cause_visible": "yes",
                    "video_quality": "good",
                    "control_valid": "yes",
                },
                {
                    "row_index": "3",
                    "backbone": "cogvideox5b",
                    "mechanism_type": "surface_trace",
                    "control_type": "no_cause",
                    "target_visible": "no",
                    "effect_visible": "yes",
                    "alternative_cause_visible": "no",
                    "video_quality": "good",
                    "control_valid": "no",
                },
                {
                    "row_index": "4",
                    "backbone": "cogvideox5b",
                    "mechanism_type": "surface_trace",
                    "control_type": "effect_only",
                    "target_visible": "no",
                    "effect_visible": "borderline",
                    "alternative_cause_visible": "no",
                    "video_quality": "ok",
                    "control_valid": "borderline",
                },
            ]
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_control_metrics.py"),
            "--review",
            str(review),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote control metrics for 5 rows" in result.stdout
    with (output_dir / "causal_footprint_v0_controls_v1_metrics_by_backbone.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        table = {row["backbone"]: row for row in csv.DictReader(handle)}

    assert table["ALL"]["total_controls"] == "5"
    assert table["ALL"]["strict_valid_count"] == "3"
    assert table["ALL"]["lenient_valid_count"] == "4"
    assert table["ALL"]["strict_valid_rate"] == "0.6000"
    assert table["ALL"]["lenient_valid_rate"] == "0.8000"
    assert table["ALL"]["causal_specificity_pass_count"] == "4"
    assert table["ALL"]["causal_specificity_pass_rate"] == "0.8000"
    assert table["cogvideox2b"]["strict_valid_count"] == "3"
    assert table["cogvideox5b"]["strict_valid_count"] == "0"

    with (output_dir / "causal_footprint_v0_controls_v1_metrics_by_control_type.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        by_type = {row["control_type"]: row for row in csv.DictReader(handle)}
    assert by_type["no_cause"]["specificity_success_count"] == "1"
    assert by_type["no_cause"]["specificity_failure_count"] == "1"
    assert by_type["effect_only"]["specificity_success_count"] == "2"
    assert by_type["alternative_cause"]["specificity_success_count"] == "1"

    summary = (output_dir / "causal_footprint_v0_controls_v1_metrics_summary.md").read_text(encoding="utf-8")
    assert "Strict usable controls: 3/5 (0.6000)" in summary
    assert "Causal-specificity pass: 4/5 (0.8000)" in summary
