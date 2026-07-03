import csv
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_c01_factorial_gate_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_c01_factorial_gate_review", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_generation_manifest(tmp_path: Path) -> Path:
    items = []
    for seed_index, seed in enumerate([40000, 40001]):
        for variant, expected in [
            ("original", ("yes", "yes")),
            ("remove_target", ("no", "no")),
            ("footprint_only", ("no", "yes")),
            ("target_only", ("yes", "no")),
        ]:
            video_path = tmp_path / f"{seed_index}_{variant}.mp4"
            video_path.write_bytes(b"fake")
            items.append(
                {
                    "probe_index": 0,
                    "pair_id": "fluid_impact_pebble_pond_002",
                    "slice_index": 5,
                    "source_index": "12",
                    "mechanism_type": "fluid_impact",
                    "seed_index": seed_index,
                    "seed": seed,
                    "variant": variant,
                    "variant_label": variant,
                    "variant_role": variant,
                    "video_path": str(video_path),
                    "target_concept": "pebble",
                    "causal_footprint": "circular ripples",
                    "source_prompt": "A pebble drops into a pond.",
                    "prompt": f"{variant} prompt",
                    "expected_target_visible": expected[0],
                    "expected_footprint_visible": expected[1],
                }
            )
    path = tmp_path / "generation_manifest.json"
    path.write_text(
        json.dumps({"baseline": "c0_counterfactual_grid", "items": items}),
        encoding="utf-8",
    )
    return path


def test_build_review_outputs_blind_rows_and_answer_key(tmp_path):
    module = load_module()
    manifest = write_generation_manifest(tmp_path)
    output_dir = tmp_path / "review"

    result = module.main(
        [
            "--generation-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--skip-frame-extraction",
            "--shuffle-seed",
            "7",
        ]
    )

    assert result == 0
    review_rows = list(csv.DictReader((output_dir / "blind_review.csv").open(encoding="utf-8")))
    key_rows = list(csv.DictReader((output_dir / "answer_key.csv").open(encoding="utf-8")))
    assert len(review_rows) == 8
    assert len(key_rows) == 8
    assert "variant" not in review_rows[0]
    assert "expected_target_visible" not in review_rows[0]
    assert not any("original" in row["review_id"] for row in review_rows)
    assert not any("target_only" in row["review_id"] for row in review_rows)
    assert all(row["video_path"] == "" for row in review_rows)
    assert {
        "target_visible",
        "footprint_visible",
        "scene_structure_preserved",
        "cells_distinguishable",
    }.issubset(review_rows[0])
    assert {"variant", "expected_target_visible", "expected_footprint_visible"}.issubset(key_rows[0])
    assert {row["review_id"] for row in review_rows} == {row["review_id"] for row in key_rows}
    assert key_rows == sorted(key_rows, key=lambda row: row["review_id"])

    run_manifest = json.loads((output_dir / "review_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["generation_manifest"] == str(manifest)
    assert run_manifest["blind_review_csv"] == str(output_dir / "blind_review.csv")
    assert run_manifest["answer_key_csv"] == str(output_dir / "answer_key.csv")
