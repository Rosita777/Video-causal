from pathlib import Path
import csv
import json
import subprocess
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REVIEW_FIELDS = [
    "item_index",
    "slice_index",
    "source_index",
    "pair_id",
    "mechanism_type",
    "baseline",
    "baseline_label",
    "video_path",
    "video_exists",
    "strip_path",
    "strip_exists",
    "seed",
    "target_concept",
    "expected_effect",
    "source_prompt",
    "target_visible",
    "footprint_visible",
    "footprint_match",
    "separation_clear",
    "video_quality",
    "confidence",
    "final_label",
    "notes",
]


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 48), color).save(path)


def write_review(path: Path, clean_strip: Path, erased_strip: Path) -> None:
    rows = [
        {
            "item_index": "0",
            "slice_index": "0",
            "source_index": "4",
            "pair_id": "fluid_impact_pebble_pond_002",
            "mechanism_type": "fluid_impact",
            "baseline": "clean_reference",
            "baseline_label": "Clean reference",
            "video_path": "outputs/clean.mp4",
            "video_exists": "true",
            "strip_path": str(clean_strip),
            "strip_exists": "true",
            "seed": "",
            "target_concept": "pebble",
            "expected_effect": "circular ripples spread outward",
            "source_prompt": "A pebble falls into a pond and ripples spread outward.",
        },
        {
            "item_index": "0",
            "slice_index": "0",
            "source_index": "4",
            "pair_id": "fluid_impact_pebble_pond_002",
            "mechanism_type": "fluid_impact",
            "baseline": "negative_prompt",
            "baseline_label": "Negative prompt",
            "video_path": "outputs/erased.mp4",
            "video_exists": "true",
            "strip_path": str(erased_strip),
            "strip_exists": "true",
            "seed": "9300",
            "target_concept": "pebble",
            "expected_effect": "circular ripples spread outward",
            "source_prompt": "A pebble falls into a pond and ripples spread outward.",
        },
    ]
    for row in rows:
        for field in REVIEW_FIELDS:
            row.setdefault(field, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_dry_run_pairs_clean_reference_with_erased_output(tmp_path):
    clean_strip = tmp_path / "strips" / "clean.jpg"
    erased_strip = tmp_path / "strips" / "erased.jpg"
    review = tmp_path / "baseline_review.csv"
    output_dir = tmp_path / "vlm"
    write_image(clean_strip, (0, 0, 255))
    write_image(erased_strip, (0, 255, 255))
    write_review(review, clean_strip, erased_strip)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_v2_baseline_with_vlm.py"),
            "--review-csv",
            str(review),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 VLM input rows" in result.stdout
    inputs = list(csv.DictReader((output_dir / "vlm_inputs.csv").open(newline="", encoding="utf-8")))
    assert len(inputs) == 1
    assert inputs[0]["baseline"] == "negative_prompt"
    assert inputs[0]["reference_strip_path"] == str(clean_strip)
    assert inputs[0]["strip_path"] == str(erased_strip)
    assert inputs[0]["output_id"] == "v2_yes100:fluid_impact_pebble_pond_002::negative_prompt"

    payloads = [json.loads(line) for line in (output_dir / "vlm_payloads.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["reference_image_path"] == str(clean_strip)
    assert payload["image_path"] == str(erased_strip)
    assert "first image is the clean reference" in payload["prompt"]
    assert "second image is the erased output" in payload["prompt"]
    assert "5-frame" not in payload["prompt"]
    assert payload["response_schema"]["footprint_visible"] == ["yes", "no", "partial"]
    assert "final_label" not in payload["response_schema"]


def test_v2_label_derivation_from_atomic_fields():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from evaluate_v2_baseline_with_vlm import normalize_prediction

    cases = [
        (
            {
                "target_visible": "no",
                "footprint_visible": "yes",
                "footprint_match": "yes",
                "separation_clear": "yes",
                "video_quality": "yes",
            },
            "strict_causal_footprint_leakage",
        ),
        (
            {
                "target_visible": "no",
                "footprint_visible": "no",
                "footprint_match": "no",
                "separation_clear": "yes",
                "video_quality": "yes",
            },
            "erased_clean",
        ),
        (
            {
                "target_visible": "yes",
                "footprint_visible": "yes",
                "footprint_match": "yes",
                "separation_clear": "yes",
                "video_quality": "yes",
            },
            "target_leakage",
        ),
        (
            {
                "target_visible": "partial",
                "footprint_visible": "yes",
                "footprint_match": "yes",
                "separation_clear": "yes",
                "video_quality": "yes",
            },
            "borderline",
        ),
        (
            {
                "target_visible": "no",
                "footprint_visible": "yes",
                "footprint_match": "yes",
                "separation_clear": "yes",
                "video_quality": "no",
            },
            "other_failure",
        ),
    ]

    for parsed, label in cases:
        normalized = normalize_prediction({**parsed, "confidence": 0.75, "reason": "visual evidence"})
        assert normalized["final_label"] == label
        assert normalized["confidence"] == "0.7500"
        assert normalized["notes"] == "visual evidence"
