from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compute_benchmark_metrics_reports_strict_and_conditional_rates(tmp_path):
    items = tmp_path / "items.jsonl"
    out_dir = tmp_path / "metrics"
    rows = [
        {
            "item_id": "s:p1",
            "source_name": "s",
            "pair_id": "p1",
            "mechanism_type": "fluid_impact",
            "baseline_outputs": [
                {
                    "baseline": "negative_prompt",
                    "target_visible": "no",
                    "causal_effect_visible": "yes",
                    "causeless_effect": "yes",
                    "usable_for_claim": "yes",
                    "failure_mode": "causal_footprint_leakage",
                },
                {
                    "baseline": "videoeraser",
                    "target_visible": "yes",
                    "causal_effect_visible": "yes",
                    "causeless_effect": "no",
                    "usable_for_claim": "no",
                    "failure_mode": "target_leakage",
                },
            ],
        },
        {
            "item_id": "s:p2",
            "source_name": "s",
            "pair_id": "p2",
            "mechanism_type": "fracture_damage",
            "baseline_outputs": [
                {
                    "baseline": "negative_prompt",
                    "target_visible": "no",
                    "causal_effect_visible": "partial",
                    "causeless_effect": "partial",
                    "usable_for_claim": "borderline",
                    "failure_mode": "borderline_residual_cause",
                },
                {
                    "baseline": "videoeraser",
                    "target_visible": "no",
                    "causal_effect_visible": "no",
                    "causeless_effect": "no",
                    "usable_for_claim": "no",
                    "failure_mode": "effect_erased",
                },
            ],
        },
    ]
    items.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_benchmark_metrics.py"),
            "--items",
            str(items),
            "--output-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote metrics for 4 outputs" in result.stdout
    with (out_dir / "causal_footprint_v0_metrics_by_baseline.csv").open(newline="", encoding="utf-8") as handle:
        table = {row["baseline"]: row for row in csv.DictReader(handle)}

    assert table["ALL"]["total_outputs"] == "4"
    assert table["ALL"]["strict_leakage_count"] == "1"
    assert table["ALL"]["borderline_count"] == "1"
    assert table["ALL"]["target_leakage_count"] == "1"
    assert table["ALL"]["strict_leakage_rate"] == "0.2500"
    assert table["ALL"]["strict_leakage_given_target_erased"] == "0.3333"
    assert table["negative_prompt"]["total_outputs"] == "2"
    assert table["negative_prompt"]["strict_or_borderline_count"] == "2"
    assert table["videoeraser"]["other_failure_count"] == "1"

    summary = (out_dir / "causal_footprint_v0_metrics_summary.md").read_text(encoding="utf-8")
    assert "Strict causal-footprint leakage: 1/4 (0.2500)" in summary
