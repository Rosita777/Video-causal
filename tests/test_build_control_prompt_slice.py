from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_control_prompt_slice_writes_prompts_and_manifest(tmp_path):
    specs = tmp_path / "controls.jsonl"
    prompts = tmp_path / "controls.txt"
    manifest = tmp_path / "controls_manifest.json"
    specs.write_text(
        json.dumps(
            {
                "control_id": "round6_pebble__no_cause",
                "source_name": "round6_yes23",
                "source_pair_id": "round6_fluid_pebble_fountain_014",
                "source_baseline": "t2vunlearning",
                "mechanism_type": "fluid_impact",
                "target_concept": "pebble",
                "expected_effect": "ripple rings spread in fountain water",
                "control_type": "no_cause",
                "prompt": "A close-up video of a calm shallow fountain basin with still water, no pebble, no falling object, and no ripple rings.",
                "purpose": "Verify that the model does not add ripple rings without an impact cause.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_control_prompt_slice.py"),
            "--specs",
            str(specs),
            "--output-prompts",
            str(prompts),
            "--output-manifest",
            str(manifest),
            "--slice-name",
            "controls_v1",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Exported 1 control prompts" in result.stdout
    assert prompts.read_text(encoding="utf-8").splitlines()[-1] == (
        "A close-up video of a calm shallow fountain basin with still water, no pebble, "
        "no falling object, and no ripple rings. | pebble | ripple rings spread in fountain water"
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["slice_name"] == "controls_v1"
    assert data["count"] == 1
    assert data["items"][0]["pair_id"] == "round6_pebble__no_cause"
    assert data["items"][0]["source_pair_id"] == "round6_fluid_pebble_fountain_014"
    assert data["items"][0]["control_type"] == "no_cause"
    assert data["items"][0]["causal_footprint"] == "ripple rings spread in fountain water"


def test_build_control_prompt_slice_rejects_duplicate_control_ids(tmp_path):
    specs = tmp_path / "controls.jsonl"
    specs.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "control_id": "duplicate",
                        "source_name": "round6_yes23",
                        "source_pair_id": "p1",
                        "source_baseline": "negative_prompt",
                        "mechanism_type": "fluid_impact",
                        "target_concept": "raindrop",
                        "expected_effect": "ripples",
                        "control_type": "no_cause",
                        "prompt": "Still water with no droplet and no ripples.",
                        "purpose": "first",
                    }
                ),
                json.dumps(
                    {
                        "control_id": "duplicate",
                        "source_name": "round6_yes23",
                        "source_pair_id": "p1",
                        "source_baseline": "negative_prompt",
                        "mechanism_type": "fluid_impact",
                        "target_concept": "raindrop",
                        "expected_effect": "ripples",
                        "control_type": "effect_only",
                        "prompt": "Ripples spread with no visible droplet.",
                        "purpose": "second",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_control_prompt_slice.py"),
            "--specs",
            str(specs),
            "--output-prompts",
            str(tmp_path / "controls.txt"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "duplicate control_id: duplicate" in result.stderr
