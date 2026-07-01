from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_clean_source_review_html_names_baseline_prompt_and_pair_metadata(tmp_path):
    generation_manifest = tmp_path / "generation_manifest.json"
    metadata_manifest = tmp_path / "export_manifest.json"
    output_dir = tmp_path / "review"

    prompt = (
        "A realistic close-up video of a red ball hitting a glass pane, "
        "causing cracks to spread outward."
    )
    generation_manifest.write_text(
        json.dumps(
            {
                "baseline": "clean",
                "items": [
                    {
                        "index": 0,
                        "prompt": prompt,
                        "target_concept": "red ball",
                        "expected_effect": "cracks spread outward",
                        "video_path": "outputs/example/videos/red_ball.mp4",
                        "source_prompt_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fracture_damage_red_ball_glass_001",
                        "mechanism_type": "fracture_damage",
                        "causal_footprint": "cracks spread outward",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_clean_source_review.py"),
            "--manifest",
            str(generation_manifest),
            "--metadata-manifest",
            str(metadata_manifest),
            "--output-dir",
            str(output_dir),
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    html = (output_dir / "clean_gallery.html").read_text(encoding="utf-8")
    assert "Clean reference" in html
    assert "fracture_damage_red_ball_glass_001" in html
    assert "fracture_damage" in html
    assert "red ball" in html
    assert "cracks spread outward" in html
    assert prompt in html
    assert "red_ball.mp4" in html

    with (output_dir / "clean_source_screening.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["baseline"] == "clean"
    assert rows[0]["pair_id"] == "fracture_damage_red_ball_glass_001"
    assert rows[0]["mechanism_type"] == "fracture_damage"


def test_clean_source_review_accepts_jsonl_metadata_manifest(tmp_path):
    generation_manifest = tmp_path / "generation_manifest.json"
    metadata_manifest = tmp_path / "candidate_items.jsonl"
    output_dir = tmp_path / "review"

    generation_manifest.write_text(
        json.dumps(
            {
                "baseline": "clean",
                "items": [
                    {
                        "index": 0,
                        "prompt": "A realistic video of a pebble falling into a pond, causing ripples.",
                        "video_path": "outputs/example/videos/pebble.mp4",
                        "source_prompt_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_manifest.write_text(
        json.dumps(
            {
                "pair_id": "fluid_impact_pebble_pond_002",
                "mechanism_type": "fluid_impact",
                "target_concept": "pebble",
                "expected_effect": "circular ripples spread outward",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_clean_source_review.py"),
            "--manifest",
            str(generation_manifest),
            "--metadata-manifest",
            str(metadata_manifest),
            "--output-dir",
            str(output_dir),
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with (output_dir / "clean_source_screening.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pair_id"] == "fluid_impact_pebble_pond_002"
    assert rows[0]["target_concept"] == "pebble"
    assert rows[0]["expected_effect"] == "circular ripples spread outward"
