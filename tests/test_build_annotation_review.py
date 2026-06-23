from pathlib import Path
import csv
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_annotation_review_writes_static_html_and_queue(tmp_path):
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "review"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "item_id",
                "mechanism_id",
                "mechanism_type",
                "target_concept",
                "causal_effect",
                "clean_prompt",
                "erasure_target",
                "baseline",
                "reference_sheet_path",
                "contact_sheet_path",
                "human_label",
                "human_target_visible",
                "human_effect_visible",
                "human_separation_clear",
                "human_video_quality",
                "human_notes",
                "claude_label",
                "claude_disagrees",
                "claude_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "item_a::videoeraser",
                "item_id": "item_a",
                "mechanism_id": "round4_fluid_a",
                "mechanism_type": "fluid_impact",
                "target_concept": "stone",
                "causal_effect": "ripples spread outward",
                "clean_prompt": "A stone falls into water and makes ripples.",
                "erasure_target": "stone",
                "baseline": "videoeraser",
                "reference_sheet_path": "frame_sheets/ref.jpg",
                "contact_sheet_path": "frame_sheets/out.jpg",
                "human_label": "strict_leakage",
                "human_target_visible": "no",
                "human_effect_visible": "yes",
                "human_separation_clear": "yes",
                "human_video_quality": "good",
                "human_notes": "stone gone, ripples remain",
                "claude_label": "borderline",
                "claude_disagrees": "yes",
                "claude_reason": "source/effect separation is ambiguous",
            }
        )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_annotation_review.py"),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    html = (output_dir / "review.html").read_text(encoding="utf-8")
    assert "Causal Footprint Annotation Review" in html
    assert "round4_fluid_a" in html
    assert "videoeraser" in html
    assert "strict_leakage" in html
    assert "claude: borderline" in html
    assert "Disagreement" in html
    assert "frame_sheets/ref.jpg" in html
    assert "frame_sheets/out.jpg" in html

    with (output_dir / "annotation_queue.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "item_a::videoeraser"
    assert rows[0]["review_label"] == ""
    assert rows[0]["review_target_visible"] == ""
    assert rows[0]["review_effect_visible"] == ""
    assert rows[0]["review_separation_clear"] == ""
    assert rows[0]["review_notes"] == ""
