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


def test_build_evaluation_manifest_merges_gold_sheets_and_predictions(tmp_path):
    gold = tmp_path / "gold.csv"
    vlm_inputs = tmp_path / "vlm_inputs.csv"
    predictions = tmp_path / "claude.csv"
    output = tmp_path / "manifest.csv"

    gold_fields = [
        "output_id",
        "item_id",
        "source_name",
        "pair_id",
        "mechanism_type",
        "baseline",
        "video_path",
        "reference_video_path",
        "reference_video_exists",
        "reference_video_quality",
        "reference_notes",
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
    write_csv(
        gold,
        gold_fields,
        [
            {
                "output_id": "item_a::videoeraser",
                "item_id": "item_a",
                "source_name": "round4_valid9",
                "pair_id": "round4_fluid_a",
                "mechanism_type": "fluid_impact",
                "baseline": "videoeraser",
                "video_path": "outputs/a.mp4",
                "reference_video_path": "outputs/ref.mp4",
                "reference_video_exists": "True",
                "reference_video_quality": "good",
                "reference_notes": "clean source",
                "seed": "4100",
                "target_concept": "stone",
                "expected_effect": "ripples spread outward",
                "source_prompt": "A stone falls into water and makes ripples.",
                "target_visible": "no",
                "causal_effect_visible": "yes",
                "causeless_effect": "yes",
                "video_quality": "good",
                "usable_for_claim": "yes",
                "failure_mode": "causal_footprint_leakage",
                "human_label": "strict_leakage",
                "notes": "stone gone, ripples remain",
            }
        ],
    )
    write_csv(
        vlm_inputs,
        [
            "output_id",
            "item_id",
            "baseline",
            "sheet_path",
            "sheet_exists",
            "reference_sheet_path",
            "reference_sheet_exists",
        ],
        [
            {
                "output_id": "item_a::videoeraser",
                "item_id": "item_a",
                "baseline": "videoeraser",
                "sheet_path": "experiments/eval/frame_sheets/a.jpg",
                "sheet_exists": "true",
                "reference_sheet_path": "experiments/eval/frame_sheets/ref.jpg",
                "reference_sheet_exists": "true",
            }
        ],
    )
    write_csv(
        predictions,
        ["item_id", "baseline", "pred_label", "confidence", "reason"],
        [
            {
                "item_id": "item_a",
                "baseline": "videoeraser",
                "pred_label": "borderline",
                "confidence": "0.72",
                "reason": "The effect is visible but the source region is ambiguous.",
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_evaluation_manifest.py"),
            "--gold",
            str(gold),
            "--vlm-inputs",
            str(vlm_inputs),
            "--prediction",
            f"claude={predictions}",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "item_a::videoeraser"
    assert row["mechanism_id"] == "round4_fluid_a"
    assert row["causal_effect"] == "ripples spread outward"
    assert row["clean_prompt"] == "A stone falls into water and makes ripples."
    assert row["erasure_target"] == "stone"
    assert row["contact_sheet_path"] == "experiments/eval/frame_sheets/a.jpg"
    assert row["reference_sheet_path"] == "experiments/eval/frame_sheets/ref.jpg"
    assert row["expected_target_absent"] == "yes"
    assert row["expected_effect_visible"] == "yes"
    assert row["human_target_visible"] == "no"
    assert row["human_effect_visible"] == "yes"
    assert row["human_separation_clear"] == "yes"
    assert row["human_label"] == "strict_leakage"
    assert row["claude_label"] == "borderline"
    assert row["claude_confidence"] == "0.72"
    assert row["claude_disagrees"] == "yes"


def test_build_evaluation_manifest_rejects_prediction_without_gold_key(tmp_path):
    gold = tmp_path / "gold.csv"
    vlm_inputs = tmp_path / "vlm_inputs.csv"
    predictions = tmp_path / "claude.csv"
    output = tmp_path / "manifest.csv"

    write_csv(
        gold,
        [
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
            "failure_mode",
            "human_label",
            "notes",
        ],
        [
            {
                "output_id": "item_a::videoeraser",
                "item_id": "item_a",
                "source_name": "round4_valid9",
                "pair_id": "round4_fluid_a",
                "mechanism_type": "fluid_impact",
                "baseline": "videoeraser",
                "video_path": "outputs/a.mp4",
                "reference_video_path": "",
                "seed": "1",
                "target_concept": "stone",
                "expected_effect": "ripples",
                "source_prompt": "prompt",
                "target_visible": "no",
                "causal_effect_visible": "yes",
                "causeless_effect": "yes",
                "video_quality": "good",
                "failure_mode": "causal_footprint_leakage",
                "human_label": "strict_leakage",
                "notes": "",
            }
        ],
    )
    write_csv(vlm_inputs, ["output_id", "item_id", "baseline"], [])
    write_csv(
        predictions,
        ["item_id", "baseline", "pred_label", "confidence", "reason"],
        [{"item_id": "missing", "baseline": "videoeraser", "pred_label": "strict_leakage", "confidence": "1", "reason": ""}],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_evaluation_manifest.py"),
            "--gold",
            str(gold),
            "--vlm-inputs",
            str(vlm_inputs),
            "--prediction",
            f"claude={predictions}",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "predictions without gold rows" in result.stderr
