from pathlib import Path
import csv
import subprocess
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sheet(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 32), color).save(path)


def metric_row(group_field: str, group: str, total: str, strict: str, borderline: str) -> dict[str, str]:
    return {
        group_field: group,
        "total_outputs": total,
        "target_erased_count": strict,
        "strict_leakage_count": strict,
        "borderline_count": borderline,
        "strict_or_borderline_count": str(int(strict) + int(borderline)),
        "target_leakage_count": "1",
        "other_failure_count": "0",
        "strict_leakage_rate": f"{int(strict) / int(total):.4f}",
        "borderline_rate": f"{int(borderline) / int(total):.4f}",
        "strict_or_borderline_rate": f"{(int(strict) + int(borderline)) / int(total):.4f}",
        "strict_leakage_given_target_erased": "1.0000",
    }


def test_build_paper_assets_writes_summary_examples_and_failure_taxonomy(tmp_path):
    baseline_metrics = tmp_path / "metrics_by_baseline.csv"
    mechanism_metrics = tmp_path / "metrics_by_mechanism.csv"
    gold = tmp_path / "gold.csv"
    vlm = tmp_path / "vlm_inputs.csv"
    out_dir = tmp_path / "paper_assets"

    metric_fields = [
        "total_outputs",
        "target_erased_count",
        "strict_leakage_count",
        "borderline_count",
        "strict_or_borderline_count",
        "target_leakage_count",
        "other_failure_count",
        "strict_leakage_rate",
        "borderline_rate",
        "strict_or_borderline_rate",
        "strict_leakage_given_target_erased",
    ]
    write_csv(
        baseline_metrics,
        ["baseline", *metric_fields],
        [
            metric_row("baseline", "ALL", "4", "2", "1"),
            metric_row("baseline", "negative_prompt", "2", "1", "1"),
            metric_row("baseline", "videoeraser", "2", "1", "0"),
        ],
    )
    write_csv(
        mechanism_metrics,
        ["mechanism_type", *metric_fields],
        [
            metric_row("mechanism_type", "ALL", "4", "2", "1"),
            metric_row("mechanism_type", "fluid_impact", "2", "1", "1"),
            metric_row("mechanism_type", "surface_trace", "2", "1", "0"),
        ],
    )

    gold_fields = [
        "output_id",
        "item_id",
        "source_name",
        "pair_id",
        "mechanism_type",
        "baseline",
        "video_path",
        "reference_video_path",
        "seed",
        "target_concept",
        "expected_effect",
        "source_prompt",
        "target_visible",
        "causal_effect_visible",
        "causeless_effect",
        "video_quality",
        "usable_for_claim",
        "failure_mode",
        "human_label",
        "notes",
    ]
    gold_rows = [
        {
            "output_id": "round6:p1::negative_prompt",
            "item_id": "round6:p1",
            "source_name": "round6_yes23",
            "pair_id": "p1",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "video_path": "v1.mp4",
            "reference_video_path": "r1.mp4",
            "seed": "1",
            "target_concept": "pebble",
            "expected_effect": "ripples remain",
            "source_prompt": "A pebble falls into water.",
            "target_visible": "no",
            "causal_effect_visible": "yes",
            "causeless_effect": "yes",
            "video_quality": "good",
            "usable_for_claim": "yes",
            "failure_mode": "causal_footprint_leakage",
            "human_label": "strict_leakage",
            "notes": "Pebble absent while ripples remain.",
        },
        {
            "output_id": "round6:p2::videoeraser",
            "item_id": "round6:p2",
            "source_name": "round6_yes23",
            "pair_id": "p2",
            "mechanism_type": "surface_trace",
            "baseline": "videoeraser",
            "video_path": "v2.mp4",
            "reference_video_path": "r2.mp4",
            "seed": "2",
            "target_concept": "sneaker",
            "expected_effect": "footprint remains",
            "source_prompt": "A sneaker leaves a footprint.",
            "target_visible": "no",
            "causal_effect_visible": "yes",
            "causeless_effect": "yes",
            "video_quality": "good",
            "usable_for_claim": "yes",
            "failure_mode": "causal_footprint_leakage",
            "human_label": "strict_leakage",
            "notes": "Sneaker absent while footprint remains.",
        },
        {
            "output_id": "round6:p3::negative_prompt",
            "item_id": "round6:p3",
            "source_name": "round6_yes23",
            "pair_id": "p3",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "video_path": "v3.mp4",
            "reference_video_path": "r3.mp4",
            "seed": "3",
            "target_concept": "coin",
            "expected_effect": "rings remain",
            "source_prompt": "A coin falls into water.",
            "target_visible": "partial",
            "causal_effect_visible": "yes",
            "causeless_effect": "partial",
            "video_quality": "ok",
            "usable_for_claim": "borderline",
            "failure_mode": "borderline_residual_cause",
            "human_label": "borderline",
            "notes": "Small residual source cue remains.",
        },
        {
            "output_id": "round6:p4::videoeraser",
            "item_id": "round6:p4",
            "source_name": "round6_yes23",
            "pair_id": "p4",
            "mechanism_type": "surface_trace",
            "baseline": "videoeraser",
            "video_path": "v4.mp4",
            "reference_video_path": "r4.mp4",
            "seed": "4",
            "target_concept": "tire",
            "expected_effect": "track remains",
            "source_prompt": "A tire leaves a track.",
            "target_visible": "yes",
            "causal_effect_visible": "yes",
            "causeless_effect": "no",
            "video_quality": "good",
            "usable_for_claim": "no",
            "failure_mode": "target_leakage",
            "human_label": "target_leakage",
            "notes": "Tire remains visible.",
        },
    ]
    write_csv(gold, gold_fields, gold_rows)

    vlm_fields = [
        "output_id",
        "reference_sheet_path",
        "reference_sheet_exists",
        "sheet_path",
        "sheet_exists",
    ]
    vlm_rows = []
    for row in gold_rows:
        ref = tmp_path / "sheets" / f"{row['output_id'].replace(':', '-')}-ref.jpg"
        sheet = tmp_path / "sheets" / f"{row['output_id'].replace(':', '-')}.jpg"
        write_sheet(ref, (20, 20, 20))
        write_sheet(sheet, (200, 200, 240))
        vlm_rows.append(
            {
                "output_id": row["output_id"],
                "reference_sheet_path": str(ref),
                "reference_sheet_exists": "true",
                "sheet_path": str(sheet),
                "sheet_exists": "true",
            }
        )
    write_csv(vlm, vlm_fields, vlm_rows)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_paper_assets.py"),
            "--gold",
            str(gold),
            "--vlm-inputs",
            str(vlm),
            "--metrics-by-baseline",
            str(baseline_metrics),
            "--metrics-by-mechanism",
            str(mechanism_metrics),
            "--output-dir",
            str(out_dir),
            "--max-examples",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Wrote paper assets" in result.stdout

    summary = (out_dir / "benchmark_results_summary.md").read_text(encoding="utf-8")
    assert "Strict causal-footprint leakage: 2/4 (0.5000)" in summary
    assert "| Negative Prompt | 2 | 1 | 1 |" in summary

    with (out_dir / "selected_strict_examples.csv").open(newline="", encoding="utf-8") as handle:
        examples = list(csv.DictReader(handle))
    assert [row["mechanism_type"] for row in examples] == ["fluid_impact", "surface_trace"]
    assert examples[0]["reference_sheet_exists"] == "true"
    assert examples[0]["sheet_exists"] == "true"

    with (out_dir / "failure_taxonomy.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert {
        "human_label": "target_leakage",
        "failure_mode": "target_leakage",
        "baseline": "videoeraser",
        "count": "1",
    } in failures

    html = (out_dir / "selected_strict_examples.html").read_text(encoding="utf-8")
    assert "Pebble absent while ripples remain." in html
    assert "paper_table_by_baseline.tex" in [path.name for path in out_dir.iterdir()]
