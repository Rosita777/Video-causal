from pathlib import Path
import csv
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_manifest(path: Path, video_path: str, baseline: str = "negative_prompt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "baseline": baseline,
                "items": [
                    {
                        "index": 0,
                        "prompt": "A close-up video of a coin dropping into water, causing ripples.",
                        "target_concept": "coin",
                        "expected_effect": "ripples spread outward",
                        "video_path": video_path,
                        "seed": 6100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_baseline_review_aligns_clean_reference_and_baselines(tmp_path):
    export_manifest = tmp_path / "export_manifest.json"
    baseline_root = tmp_path / "suite"
    output_dir = tmp_path / "review"

    clean_video = tmp_path / "clean.mp4"
    negative_video = tmp_path / "negative.mp4"
    clean_video.write_bytes(b"not-a-real-video")
    negative_video.write_bytes(b"not-a-real-video")

    export_manifest.write_text(
        json.dumps(
            {
                "slice_name": "yes1",
                "items": [
                    {
                        "slice_index": 0,
                        "source_index": 4,
                        "pair_id": "round5_fluid_coin_fountain_005",
                        "mechanism_type": "fluid_impact",
                        "target_concept": "coin",
                        "causal_footprint": "small splash and ripple rings appear",
                        "source_prompt": "A close-up video of a coin dropping into water, causing ripples.",
                        "clean_video_path": str(clean_video),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_manifest(
        baseline_root / "negative_prompt_shards" / "prompt_000" / "generation_manifest.json",
        str(negative_video),
        baseline="negative_prompt",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_baseline_review.py"),
            "--export-manifest",
            str(export_manifest),
            "--baseline-root",
            str(baseline_root),
            "--output-dir",
            str(output_dir),
            "--baselines",
            "negative_prompt",
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    html = (output_dir / "baseline_gallery.html").read_text(encoding="utf-8")
    assert "round5_fluid_coin_fountain_005" in html
    assert "Clean reference" in html
    assert "Negative prompt" in html
    assert "small splash and ripple rings appear" in html
    assert "clean.mp4" in html
    assert "negative.mp4" in html

    with (output_dir / "baseline_review.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["baseline"] for row in rows] == ["clean_reference", "negative_prompt"]
    assert {row["video_exists"] for row in rows} == {"true"}
    assert rows[0]["pair_id"] == "round5_fluid_coin_fountain_005"
    assert rows[1]["seed"] == "6100"


def test_baseline_review_keeps_missing_baseline_rows(tmp_path):
    export_manifest = tmp_path / "export_manifest.json"
    baseline_root = tmp_path / "suite"
    output_dir = tmp_path / "review"

    clean_video = tmp_path / "clean.mp4"
    clean_video.write_bytes(b"not-a-real-video")
    export_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "slice_index": 0,
                        "pair_id": "round5_surface_tire_puddle_010",
                        "mechanism_type": "surface_trace",
                        "target_concept": "bicycle tire",
                        "causal_footprint": "wet tire track remains on pavement",
                        "source_prompt": "A tire rolls out of a puddle and leaves a wet track.",
                        "clean_video_path": str(clean_video),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_baseline_review.py"),
            "--export-manifest",
            str(export_manifest),
            "--baseline-root",
            str(baseline_root),
            "--output-dir",
            str(output_dir),
            "--baselines",
            "videoeraser",
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    with (output_dir / "baseline_review.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["baseline"] for row in rows] == ["clean_reference", "videoeraser"]
    assert rows[0]["video_exists"] == "true"
    assert rows[1]["video_exists"] == "false"
    assert rows[1]["video_path"] == ""
    assert rows[1]["notes"] == "missing generation manifest"
