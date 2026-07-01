from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_review_csv(path: Path) -> None:
    fieldnames = [
        "prompt_id",
        "pair_id",
        "baseline",
        "mechanism_type",
        "video_path",
        "prompt",
        "target_concept",
        "expected_effect",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "case_000",
                "pair_id": "fluid_pebble__effect_only",
                "baseline": "clean",
                "mechanism_type": "fluid_impact",
                "video_path": "outputs/demo.mp4",
                "prompt": "Ripples spread with no pebble visible.",
                "target_concept": "pebble",
                "expected_effect": "ripples spread outward",
            }
        )


def test_clean_control_vlm_dry_run_uses_existing_frame_strip(tmp_path):
    review = tmp_path / "clean_source_screening.csv"
    write_review_csv(review)
    strip_dir = tmp_path / "frame_strips"
    strip_dir.mkdir()
    (strip_dir / "000_case_000.jpg").write_bytes(b"fake-jpeg")
    output = tmp_path / "payloads.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_clean_controls_with_vlm.py"),
            "--review-csv",
            str(review),
            "--frame-strip-dir",
            str(strip_dir),
            "--output-jsonl",
            str(output),
            "--dry-run",
            "--task",
            "control",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["sample_id"] == "case_000"
    assert payload["pair_id"] == "fluid_pebble__effect_only"
    assert payload["image_path"].endswith("000_case_000.jpg")
    assert payload["task"] == "control"
    assert payload["response_schema"]["target_visible"] == ["yes", "no", "partial"]
    assert payload["response_schema"]["effect_visible"] == ["yes", "no", "partial"]
    assert "control_type" in payload["response_schema"]
    assert "clean_source_valid" in payload["response_schema"]
    assert "Do not infer from the text prompt alone" in payload["prompt"]


def test_clean_control_normalize_prediction_derives_validity():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_controls_with_vlm import normalize_prediction

    pred = normalize_prediction(
        {
            "target_visible": "yes",
            "effect_visible": "yes",
            "temporal_order_clear": "yes",
            "effect_depends_on_target": "yes",
            "video_quality": "yes",
            "confidence": 0.8,
            "reason": "target appears before the ripples",
        },
        task="clean",
    )

    assert pred["clean_source_valid"] == "yes"
    assert pred["confidence"] == "0.8000"


def test_control_normalize_prediction_keeps_control_validity():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_clean_controls_with_vlm import normalize_prediction

    pred = normalize_prediction(
        {
            "target_visible": "no",
            "effect_visible": "yes",
            "temporal_order_clear": "no",
            "effect_depends_on_target": "no",
            "video_quality": "yes",
            "control_valid": "yes",
            "control_type": "effect_only",
            "confidence": "0.5",
            "reason": "ripples are visible with no pebble",
        },
        task="control",
    )

    assert pred["control_valid"] == "yes"
    assert pred["control_type"] == "effect_only"
    assert pred["clean_source_valid"] == ""
