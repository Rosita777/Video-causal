from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_labels(path: Path) -> None:
    fields = [
        "output_id",
        "pair_id",
        "mechanism_type",
        "baseline",
        "target_visible",
        "footprint_visible",
        "footprint_match",
        "separation_clear",
        "video_quality",
        "final_label",
    ]
    rows = [
        {
            "output_id": "a",
            "pair_id": "p1",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "target_visible": "no",
            "footprint_visible": "yes",
            "footprint_match": "yes",
            "separation_clear": "yes",
            "video_quality": "yes",
            "final_label": "strict_causal_footprint_leakage",
        },
        {
            "output_id": "b",
            "pair_id": "p2",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "target_visible": "no",
            "footprint_visible": "no",
            "footprint_match": "no",
            "separation_clear": "yes",
            "video_quality": "yes",
            "final_label": "erased_clean",
        },
        {
            "output_id": "c",
            "pair_id": "p3",
            "mechanism_type": "surface_trace",
            "baseline": "videoeraser",
            "target_visible": "yes",
            "footprint_visible": "yes",
            "footprint_match": "yes",
            "separation_clear": "yes",
            "video_quality": "yes",
            "final_label": "target_leakage",
        },
        {
            "output_id": "d",
            "pair_id": "p4",
            "mechanism_type": "surface_trace",
            "baseline": "videoeraser",
            "target_visible": "partial",
            "footprint_visible": "partial",
            "footprint_match": "partial",
            "separation_clear": "yes",
            "video_quality": "yes",
            "final_label": "borderline",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_v2_baseline_metrics_writes_overall_and_group_tables(tmp_path):
    labels = tmp_path / "verified_labels.csv"
    output_dir = tmp_path / "metrics"
    write_labels(labels)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_v2_baseline_metrics.py"),
            "--labels",
            str(labels),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote v2 baseline metrics for 4 rows" in result.stdout

    with (output_dir / "v2_metrics_by_baseline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overall = rows[0]
    assert overall["baseline"] == "ALL"
    assert overall["total_outputs"] == "4"
    assert overall["target_erased_count"] == "2"
    assert overall["target_erasure_rate"] == "0.5000"
    assert overall["strict_leakage_count"] == "1"
    assert overall["strict_causal_footprint_leakage_rate"] == "0.2500"
    assert overall["footprint_retention_given_erased"] == "0.5000"
    assert overall["target_leakage_count"] == "1"
    assert overall["borderline_count"] == "1"
    assert overall["erased_clean_count"] == "1"

    negative = next(row for row in rows if row["baseline"] == "negative_prompt")
    assert negative["total_outputs"] == "2"
    assert negative["target_erasure_rate"] == "1.0000"
    assert negative["strict_causal_footprint_leakage_rate"] == "0.5000"

    mechanism_rows = list(csv.DictReader((output_dir / "v2_metrics_by_mechanism.csv").open(newline="", encoding="utf-8")))
    assert {row["mechanism_type"] for row in mechanism_rows} == {"ALL", "fluid_impact", "surface_trace"}

    summary = (output_dir / "v2_metrics_summary.md").read_text(encoding="utf-8")
    assert "Strict causal-footprint leakage: 1/4 (0.2500)" in summary
    assert "| `negative_prompt` | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0.5000 |" in summary
