from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_causal_eval_manifest_joins_valid_items_controls_and_baseline_rows(tmp_path):
    items = tmp_path / "items.jsonl"
    controls = tmp_path / "controls.csv"
    baseline_review = tmp_path / "baseline_review.csv"
    output = tmp_path / "manifest.csv"

    items.write_text(
        json.dumps(
            {
                "item_id": "round6:pair1",
                "source_name": "round6",
                "pair_id": "pair1",
                "mechanism_type": "fluid_impact",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
                "source_prompt": "A pebble drops into still water.",
                "baseline_outputs": [
                    {
                        "baseline": "negative_prompt",
                        "video_path": "neg.mp4",
                        "target_visible": "no",
                        "causal_effect_visible": "yes",
                        "causeless_effect": "yes",
                        "video_quality": "good",
                        "usable_for_claim": "yes",
                        "failure_mode": "causal_footprint_leakage",
                        "notes": "ripples remain",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    controls.write_text(
        "item_id,backbone,control_type,control_valid,target_visible,effect_visible,alternative_cause_visible\n"
        "round6:pair1,cogvideox2b,no_cause,yes,no,no,no\n"
        "round6:pair1,cogvideox2b,effect_only,yes,no,yes,no\n"
        "round6:pair1,cogvideox2b,alternative_cause,yes,no,yes,yes\n",
        encoding="utf-8",
    )
    baseline_review.write_text(
        "item_id,baseline,review_label,review_target_visible,review_effect_visible,review_separation_clear,review_notes\n"
        "round6:pair1,negative_prompt,strict_leakage,no,yes,yes,looks leaked\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_causal_eval_manifest.py"),
            "--items",
            str(items),
            "--controls",
            str(controls),
            "--baseline-review",
            str(baseline_review),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Wrote 1 causal eval manifest rows" in result.stdout
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert rows == [
        {
            "item_id": "round6:pair1",
            "source_name": "round6",
            "pair_id": "pair1",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "video_path": "neg.mp4",
            "control_gate": "pass",
            "target_visible": "no",
            "causal_effect_visible": "yes",
            "causeless_effect": "yes",
            "video_quality": "good",
            "usable_for_claim": "yes",
            "failure_mode": "causal_footprint_leakage",
            "vlm_label": "",
            "human_label": "strict_leakage",
            "adjudicated_label": "strict_leakage",
            "review_notes": "looks leaked",
        }
    ]
