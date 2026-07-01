from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compute_causal_eval_metrics_uses_adjudicated_label(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "metrics"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "baseline", "adjudicated_label", "target_visible", "control_gate"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "item_id": "i1",
                    "baseline": "negative_prompt",
                    "adjudicated_label": "strict_leakage",
                    "target_visible": "no",
                    "control_gate": "pass",
                },
                {
                    "item_id": "i1",
                    "baseline": "videoeraser",
                    "adjudicated_label": "borderline",
                    "target_visible": "no",
                    "control_gate": "pass",
                },
                {
                    "item_id": "i2",
                    "baseline": "negative_prompt",
                    "adjudicated_label": "target_leakage",
                    "target_visible": "yes",
                    "control_gate": "pass",
                },
                {
                    "item_id": "i3",
                    "baseline": "negative_prompt",
                    "adjudicated_label": "strict_leakage",
                    "target_visible": "no",
                    "control_gate": "fail",
                },
            ]
        )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_causal_eval_metrics.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote causal eval metrics for 3 gated rows" in result.stdout
    summary = (output_dir / "causal_eval_metrics_summary.md").read_text(encoding="utf-8")
    assert "Strict leakage: 1/3 (0.3333)" in summary
    assert "Borderline: 1/3 (0.3333)" in summary
    assert "Target leakage: 1/3 (0.3333)" in summary
    assert "Strict leakage given target erased: 0.5000" in summary

    by_baseline = {
        row["baseline"]: row
        for row in csv.DictReader(
            (output_dir / "causal_eval_metrics_by_baseline.csv").open(newline="", encoding="utf-8")
        )
    }
    assert by_baseline["negative_prompt"]["strict_leakage_count"] == "1"
    assert by_baseline["videoeraser"]["borderline_count"] == "1"
