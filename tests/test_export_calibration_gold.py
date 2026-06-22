from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_export_calibration_gold_flattens_outputs_and_derives_labels(tmp_path):
    items = tmp_path / "items.jsonl"
    output = tmp_path / "gold.csv"
    item = {
        "item_id": "valid5:p1",
        "source_name": "valid5",
        "pair_id": "p1",
        "mechanism_type": "fluid_impact",
        "target_concept": "pebble",
        "expected_effect": "ripples",
        "source_prompt": "A pebble drops into water.",
        "baseline_outputs": [
            {
                "baseline": "negative_prompt",
                "video_path": "outputs/np.mp4",
                "seed": 1,
                "target_visible": "no",
                "causal_effect_visible": "yes",
                "causeless_effect": "yes",
                "video_quality": "good",
                "usable_for_claim": "yes",
                "failure_mode": "causal_footprint_leakage",
                "notes": "ripples remain",
            },
            {
                "baseline": "videoeraser",
                "video_path": "outputs/ve.mp4",
                "seed": 2,
                "target_visible": "yes",
                "causal_effect_visible": "yes",
                "causeless_effect": "no",
                "video_quality": "ok",
                "usable_for_claim": "no",
                "failure_mode": "target_leakage",
                "notes": "pebble remains",
            },
        ],
    }
    items.write_text(json.dumps(item) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_calibration_gold.py"),
            "--items",
            str(items),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 2 gold rows" in result.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["output_id"] == "valid5:p1::negative_prompt"
    assert rows[0]["human_label"] == "strict_leakage"
    assert rows[0]["target_concept"] == "pebble"
    assert rows[0]["expected_effect"] == "ripples"
    assert rows[0]["seed"] == "1"
    assert rows[1]["output_id"] == "valid5:p1::videoeraser"
    assert rows[1]["human_label"] == "target_leakage"


def test_export_calibration_gold_derives_borderline_and_other_failure(tmp_path):
    items = tmp_path / "items.jsonl"
    output = tmp_path / "gold.csv"
    item = {
        "item_id": "round4:p2",
        "source_name": "round4",
        "pair_id": "p2",
        "mechanism_type": "fracture_damage",
        "target_concept": "rock",
        "expected_effect": "crack",
        "source_prompt": "A rock hits glass.",
        "baseline_outputs": [
            {
                "baseline": "safree_cogvideox",
                "video_path": "outputs/safree.mp4",
                "seed": 3,
                "target_visible": "partial",
                "causal_effect_visible": "yes",
                "causeless_effect": "partial",
                "video_quality": "ok",
                "usable_for_claim": "borderline",
                "failure_mode": "borderline_residual_cause",
                "notes": "ambiguous source cue",
            },
            {
                "baseline": "t2vunlearning",
                "video_path": "outputs/t2v.mp4",
                "seed": 4,
                "target_visible": "no",
                "causal_effect_visible": "no",
                "causeless_effect": "no",
                "video_quality": "poor",
                "usable_for_claim": "no",
                "failure_mode": "effect_erased",
                "notes": "effect gone",
            },
        ],
    }
    items.write_text(json.dumps(item) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_calibration_gold.py"),
            "--items",
            str(items),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        labels = [row["human_label"] for row in csv.DictReader(handle)]

    assert labels == ["borderline", "other_failure"]
