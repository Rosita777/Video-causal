from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compute_evaluation_metrics_reports_groups_and_model_agreement(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "metrics"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "mechanism_type",
                "baseline",
                "human_label",
                "claude_label",
                "qwen_label",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "a::negative_prompt",
                    "mechanism_type": "fluid_impact",
                    "baseline": "negative_prompt",
                    "human_label": "strict_leakage",
                    "claude_label": "borderline",
                    "qwen_label": "strict_leakage",
                },
                {
                    "sample_id": "b::negative_prompt",
                    "mechanism_type": "fluid_impact",
                    "baseline": "negative_prompt",
                    "human_label": "borderline",
                    "claude_label": "borderline",
                    "qwen_label": "strict_leakage",
                },
                {
                    "sample_id": "c::videoeraser",
                    "mechanism_type": "fracture_damage",
                    "baseline": "videoeraser",
                    "human_label": "target_leakage",
                    "claude_label": "target_leakage",
                    "qwen_label": "target_leakage",
                },
                {
                    "sample_id": "d::videoeraser",
                    "mechanism_type": "fracture_damage",
                    "baseline": "videoeraser",
                    "human_label": "other_failure",
                    "claude_label": "",
                    "qwen_label": "",
                },
            ]
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_evaluation_metrics.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote evaluation metrics for 4 rows" in result.stdout

    with (output_dir / "metrics_by_baseline.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["baseline"]: row for row in csv.DictReader(handle)}
    assert rows["ALL"]["total_outputs"] == "4"
    assert rows["ALL"]["strict_leakage_count"] == "1"
    assert rows["ALL"]["borderline_count"] == "1"
    assert rows["ALL"]["relaxed_leakage_count"] == "2"
    assert rows["ALL"]["target_leakage_count"] == "1"
    assert rows["ALL"]["other_failure_count"] == "1"
    assert rows["ALL"]["relaxed_leakage_rate"] == "0.5000"
    assert rows["negative_prompt"]["relaxed_leakage_count"] == "2"
    assert rows["videoeraser"]["target_leakage_count"] == "1"

    with (output_dir / "model_agreement.csv").open(newline="", encoding="utf-8") as handle:
        agreement = {row["model"]: row for row in csv.DictReader(handle)}
    assert agreement["claude"]["compared_outputs"] == "3"
    assert agreement["claude"]["agreement_count"] == "2"
    assert agreement["claude"]["disagreement_count"] == "1"
    assert agreement["claude"]["agreement_rate"] == "0.6667"
    assert agreement["qwen"]["compared_outputs"] == "3"
    assert agreement["qwen"]["agreement_count"] == "2"

    summary = (output_dir / "metrics_summary.md").read_text(encoding="utf-8")
    assert "Relaxed leakage: 2/4 (0.5000)" in summary
    assert "| claude | 3 | 2 | 1 | 0.6667 |" in summary
