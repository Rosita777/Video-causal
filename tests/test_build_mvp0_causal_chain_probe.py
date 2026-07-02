from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_probe_builder_writes_minimal_pair_contracts(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fluid_impact_pebble_pond_002",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "source_prompt": "A realistic close-up video of a small pebble dropping into a still pond, causing circular ripples to spread outward across the water.",
                        "counterfactual_prompt": "A realistic close-up video of a still pond with a calm, undisturbed surface. No pebble is present.",
                        "control_prompt": "A realistic close-up video of a still pond where gentle background ripples move across the surface with no impact point.",
                        "clean_video_path": "outputs/clean0.mp4",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["probe_name"] == "zeroscope_mvp0_causal_chain_probe"
    assert manifest["dry_run"] is True
    assert manifest["items"][0]["pair_id"] == "fluid_impact_pebble_pond_002"
    pairs = manifest["items"][0]["minimal_pairs"]
    assert set(pairs) == {"cause", "mechanism", "footprint"}
    assert pairs["cause"]["positive"].startswith("A realistic close-up video")
    assert "pebble" in pairs["cause"]["positive"]
    assert "without pebble" in pairs["cause"]["negative"]
    assert "impact" in pairs["mechanism"]["positive"].lower()
    assert "no impact" in pairs["mechanism"]["negative"].lower()
    assert "circular ripples" in pairs["footprint"]["positive"]
    assert "no circular ripples" in pairs["footprint"]["negative"]
    assert (output_dir / "prompts" / "source_prompts.txt").exists()
    assert (output_dir / "prompts" / "counterfactual_prompts.txt").exists()


def test_probe_builder_prioritizes_strict_leakage_and_priority_mechanisms(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "surface_trace_a",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "surface_trace",
                        "target_concept": "shoe",
                        "causal_footprint": "footprint trace",
                        "source_prompt": "A shoe presses into mud leaving a footprint trace.",
                        "counterfactual_prompt": "A smooth mud surface with no shoe.",
                        "control_prompt": "A smooth mud surface.",
                    },
                    {
                        "pair_id": "fracture_damage_b",
                        "source_index": "1",
                        "slice_index": 1,
                        "mechanism_type": "fracture_damage",
                        "target_concept": "hammer",
                        "causal_footprint": "cracks in glass",
                        "source_prompt": "A hammer hits glass and cracks spread.",
                        "counterfactual_prompt": "An intact glass pane with no hammer.",
                        "control_prompt": "An intact glass pane.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "pair_id,final_label\nsurface_trace_a,strict_causal_footprint_leakage\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--baseline-labels",
            str(labels),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert [item["pair_id"] for item in manifest["items"]] == ["surface_trace_a"]


def test_probe_builder_balances_strict_leakage_across_mechanisms(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fluid_a",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "ripples",
                        "source_prompt": "A pebble drops into water and ripples appear.",
                        "counterfactual_prompt": "Still water with no pebble.",
                        "control_prompt": "Still water.",
                    },
                    {
                        "pair_id": "fluid_b",
                        "source_index": "1",
                        "slice_index": 1,
                        "mechanism_type": "fluid_impact",
                        "target_concept": "coin",
                        "causal_footprint": "splash",
                        "source_prompt": "A coin drops into water and a splash appears.",
                        "counterfactual_prompt": "Still water with no coin.",
                        "control_prompt": "Still water.",
                    },
                    {
                        "pair_id": "surface_c",
                        "source_index": "2",
                        "slice_index": 2,
                        "mechanism_type": "surface_trace",
                        "target_concept": "shoe",
                        "causal_footprint": "footprint trace",
                        "source_prompt": "A shoe presses into mud leaving a footprint trace.",
                        "counterfactual_prompt": "Smooth mud with no shoe.",
                        "control_prompt": "Smooth mud.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "pair_id,final_label\n"
        "fluid_a,strict_causal_footprint_leakage\n"
        "fluid_b,strict_causal_footprint_leakage\n"
        "surface_c,strict_causal_footprint_leakage\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--baseline-labels",
            str(labels),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    assert [item["mechanism_type"] for item in manifest["items"]] == [
        "fluid_impact",
        "surface_trace",
    ]


def test_probe_builder_uses_neutral_context_not_footprint_only_control(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fracture_damage_puck_mirror",
                        "source_index": "47",
                        "slice_index": 20,
                        "mechanism_type": "fracture_damage",
                        "target_concept": "black hockey puck",
                        "causal_footprint": "a star-shaped crack spreads across the mirror",
                        "source_prompt": (
                            "A realistic fixed-camera close-up video from a side close-up view. "
                            "The scene starts with an intact mirror tile, with no pre-existing mirror crack. "
                            "A clearly visible black hockey puck enters the frame and remains visible before contact. "
                            "The black hockey puck slides into the mirror tile, causing a star-shaped crack spreads across the mirror."
                        ),
                        "counterfactual_prompt": (
                            "A realistic fixed-camera close-up video of the same scene before the event, "
                            "with no black hockey puck, no visible cause, and no a star-shaped crack spreads across the mirror."
                        ),
                        "control_prompt": (
                            "A realistic fixed-camera close-up video showing a star-shaped crack spreads across the mirror, "
                            "with no black hockey puck or other visible cause in the frame."
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    pairs = manifest["items"][0]["minimal_pairs"]
    assert "intact mirror tile" in pairs["cause"]["positive"]
    assert "showing a star-shaped crack spreads" not in pairs["cause"]["positive"]
    assert "no black hockey puck" not in pairs["cause"]["positive"]
    assert "no a star-shaped crack" not in pairs["footprint"]["negative"]
    assert "no star-shaped crack spreads across the mirror" in pairs["footprint"]["negative"]


def test_probe_builder_sanitizes_counterfactual_context_before_minimal_pairs(tmp_path):
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fluid_impact_pebble_pond_002",
                        "source_index": "0",
                        "slice_index": 0,
                        "mechanism_type": "fluid_impact",
                        "target_concept": "pebble",
                        "causal_footprint": "circular ripples spread outward",
                        "source_prompt": (
                            "A realistic close-up video of a small pebble dropping into a still pond, "
                            "causing circular ripples to spread outward across the water."
                        ),
                        "counterfactual_prompt": (
                            "A realistic close-up video of a still pond with a calm, undisturbed surface. "
                            "No pebble is present."
                        ),
                        "control_prompt": (
                            "A realistic close-up video of a still pond where gentle background ripples "
                            "move across the surface with no impact point."
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    pairs = manifest["items"][0]["minimal_pairs"]
    assert pairs["cause"]["positive"] == (
        "A realistic close-up video of a still pond with a calm, undisturbed surface, with pebble."
    )
    assert "No pebble is present, with pebble" not in pairs["cause"]["positive"]
    assert "No pebble is present" not in pairs["mechanism"]["positive"]


def test_probe_builder_adds_compact_generation_prompt_without_losing_source(tmp_path):
    source_prompt = (
        "A realistic fixed-camera close-up video from a side close-up view. "
        "The scene starts with an intact mirror tile, with no pre-existing mirror crack. "
        "A clearly visible black hockey puck enters the frame and remains visible before contact. "
        "The black hockey puck slides into the mirror tile after the target is clearly visible, "
        "causing a star-shaped crack spreads across the mirror; the effect begins only after "
        "contact and remains visible for several frames."
    )
    clean_manifest = tmp_path / "clean_valid.json"
    clean_manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "pair_id": "fracture_damage_puck_mirror",
                        "source_index": "47",
                        "slice_index": 20,
                        "mechanism_type": "fracture_damage",
                        "target_concept": "black hockey puck",
                        "causal_footprint": "a star-shaped crack spreads across the mirror",
                        "source_prompt": source_prompt,
                        "counterfactual_prompt": "An intact mirror tile with no black hockey puck.",
                        "control_prompt": "An intact mirror tile.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mvp0_causal_chain_probe.py"),
            "--clean-valid-manifest",
            str(clean_manifest),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "probe_manifest.json").read_text(encoding="utf-8"))
    item = manifest["items"][0]
    assert item["source_prompt"] == source_prompt
    assert item["generation_prompt"] != source_prompt
    assert "black hockey puck" in item["generation_prompt"]
    assert "star-shaped crack spreads across the mirror" in item["generation_prompt"]
    assert "remains visible before contact" not in item["generation_prompt"]
    assert "effect begins only after contact" not in item["generation_prompt"]
