from pathlib import Path
import importlib.util
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / "scripts" / "build_c0_counterfactual_review.py"


def load_review_module():
    spec = importlib.util.spec_from_file_location("build_c0_counterfactual_review", REVIEW_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_generation_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "generation_manifest.json"
    items = []
    for variant in ["original", "remove_target", "footprint_only", "target_only"]:
        items.append(
            {
                "probe_index": 0,
                "pair_id": "fluid_impact_pebble_pond_002",
                "slice_index": 5,
                "source_index": "12",
                "mechanism_type": "fluid_impact",
                "variant": variant,
                "variant_label": variant.replace("_", " "),
                "prompt": f"prompt for {variant}",
                "source_prompt": "A pebble drops into a pond and causes circular ripples.",
                "target_concept": "pebble",
                "causal_footprint": "circular ripples spread outward",
                "expected_target_visible": "yes" if variant in {"original", "target_only"} else "no",
                "expected_footprint_visible": "yes" if variant in {"original", "footprint_only"} else "no",
                "seed": 30000,
                "video_path": f"experiments/c0/videos/{variant}.mp4",
            }
        )
    path.write_text(
        json.dumps(
            {
                "baseline": "c0_counterfactual_grid",
                "dry_run": False,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_review_builder_creates_reference_and_variant_rows_without_frame_extraction(tmp_path):
    module = load_review_module()
    manifest = json.loads(write_generation_manifest(tmp_path).read_text(encoding="utf-8"))

    rows = module.build_rows(
        manifest["items"],
        output_dir=tmp_path / "review",
        project_root=PROJECT_ROOT,
        frame_count=5,
        thumb_width=192,
        thumb_height=128,
        skip_frame_extraction=True,
    )

    assert [row["baseline"] for row in rows] == [
        "clean_reference",
        "original",
        "remove_target",
        "footprint_only",
        "target_only",
    ]
    clean = rows[0]
    assert clean["video_path"].endswith("original.mp4")
    assert clean["strip_exists"] == "false"
    assert rows[2]["baseline_label"] == "remove target"
    assert rows[2]["expected_target_visible"] == "no"
    assert rows[3]["expected_footprint_visible"] == "yes"
    assert rows[4]["source_prompt"] == "prompt for target_only"


def test_review_builder_cli_writes_review_csv(tmp_path):
    manifest_path = write_generation_manifest(tmp_path)
    output_dir = tmp_path / "review"

    result = subprocess.run(
        [
            sys.executable,
            str(REVIEW_PATH),
            "--generation-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    review_csv = output_dir / "review.csv"
    assert review_csv.exists()
    text = review_csv.read_text(encoding="utf-8")
    assert "clean_reference" in text
    assert "footprint_only" in text
