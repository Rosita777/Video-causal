from pathlib import Path
import importlib.util
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "adapters" / "run_c0_counterfactual_grid.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_c0_counterfactual_grid", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_probe_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "probe_manifest.json"
    path.write_text(
        json.dumps(
            {
                "probe_name": "zeroscope_mvp0_causal_chain_probe",
                "items": [
                    {
                        "probe_index": 2,
                        "pair_id": "fluid_impact_pebble_pond_002",
                        "slice_index": 5,
                        "source_index": "12",
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "source_prompt": "A pebble drops into a pond and causes circular ripples.",
                        "generation_prompt": "A still pond. pebble causes circular ripples spread outward.",
                        "counterfactual_prompt": "A still pond. No pebble is present.",
                        "control_prompt": "A still pond with circular ripples and no visible object.",
                        "clean_video_path": "outputs/clean.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_items_creates_four_expected_counterfactual_variants(tmp_path):
    module = load_runner_module()
    probe_manifest = write_probe_manifest(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "30000",
        ]
    )

    rows = module.build_items(args, json.loads(probe_manifest.read_text())["items"])

    assert [row["variant"] for row in rows] == [
        "original",
        "remove_target",
        "footprint_only",
        "target_only",
    ]
    assert {row["seed"] for row in rows} == {30002}
    expected = {
        row["variant"]: (row["expected_target_visible"], row["expected_footprint_visible"])
        for row in rows
    }
    assert expected == {
        "original": ("yes", "yes"),
        "remove_target": ("no", "no"),
        "footprint_only": ("no", "yes"),
        "target_only": ("yes", "no"),
    }
    prompts = {row["variant"]: row["prompt"] for row in rows}
    assert prompts["original"] == "A still pond. pebble causes circular ripples spread outward."
    assert "No pebble is present" in prompts["remove_target"]
    assert "no circular ripples spread outward" in prompts["remove_target"]
    assert "A still pond with circular ripples and no visible object" in prompts["footprint_only"]
    assert "no pebble" in prompts["footprint_only"]
    assert "pebble is clearly visible" in prompts["target_only"]
    assert "no circular ripples spread outward" in prompts["target_only"]
    assert rows[0]["video_path"].endswith("_original_seed30002.mp4")


def test_c0_counterfactual_grid_dry_run_writes_manifest(tmp_path):
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "c0"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--seed",
            "31000",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "c0_counterfactual_grid"
    assert manifest["dry_run"] is True
    assert manifest["variant_grid"] == ["original", "remove_target", "footprint_only", "target_only"]
    assert len(manifest["items"]) == 4


def test_c0_counterfactual_grid_original_only_dry_run_writes_screening_manifest(tmp_path):
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "c0_screen"

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--seed",
            "33000",
            "--variant-set",
            "original",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant_grid"] == ["original"]
    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["variant"] == "original"
    assert manifest["items"][0]["expected_target_visible"] == "yes"
    assert manifest["items"][0]["expected_footprint_visible"] == "yes"


def test_build_items_expands_each_probe_item_over_multiple_seeds(tmp_path):
    module = load_runner_module()
    probe_manifest = write_probe_manifest(tmp_path)
    args = module.build_parser().parse_args(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "40000",
            "--seeds-per-item",
            "3",
        ]
    )

    rows = module.build_items(args, json.loads(probe_manifest.read_text())["items"])

    assert len(rows) == 12
    assert [row["seed_index"] for row in rows[:4]] == [0, 0, 0, 0]
    assert [row["seed_index"] for row in rows[4:8]] == [1, 1, 1, 1]
    assert [row["seed_index"] for row in rows[8:12]] == [2, 2, 2, 2]
    assert {row["seed"] for row in rows[:4]} == {40002}
    assert {row["seed"] for row in rows[4:8]} == {40003}
    assert {row["seed"] for row in rows[8:12]} == {40004}
    assert rows[4]["video_path"].endswith("_seed01_original_seed40003.mp4")


def test_c0_counterfactual_grid_real_mode_calls_generator(tmp_path, monkeypatch):
    module = load_runner_module()
    probe_manifest = write_probe_manifest(tmp_path)
    output_dir = tmp_path / "c0"
    calls = []

    def fake_generate(args, rows):
        calls.append({"dry_run": args.dry_run, "rows": rows})

    monkeypatch.setattr(module, "generate_counterfactual_videos", fake_generate)

    result = module.main(
        [
            "--probe-manifest",
            str(probe_manifest),
            "--output-dir",
            str(output_dir),
            "--seed",
            "32000",
            "--limit-items",
            "1",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert len(calls[0]["rows"]) == 4
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["variant"] == "original"
