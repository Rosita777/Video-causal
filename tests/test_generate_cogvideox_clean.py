from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cogvideox_clean_dry_run_writes_generation_manifest(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_dir = tmp_path / "outputs"
    prompt_file.write_text(
        "A realistic video of a red ball rolling into wooden blocks, and the blocks fall over after impact. | ball | wooden blocks fall over\n"
        "A realistic close-up video of an ice cube dropping into cola, and bubbles rise after impact. | ice cube | bubbles rise\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_cogvideox_clean.py"),
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--model",
            "local/CogVideoX-2b",
            "--seed",
            "123",
            "--steps",
            "4",
            "--guidance-scale",
            "6.5",
            "--num-frames",
            "49",
            "--fps",
            "8",
            "--limit",
            "1",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run generation manifest written" in result.stdout
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "clean"
    assert manifest["model"] == "local/CogVideoX-2b"
    assert manifest["dry_run"] is True
    assert manifest["generation"]["seed"] == 123
    assert manifest["generation"]["num_inference_steps"] == 4
    assert manifest["generation"]["guidance_scale"] == 6.5
    assert len(manifest["items"]) == 1
    assert manifest["items"][0]["target_concept"] == "ball"
    assert manifest["items"][0]["expected_effect"] == "wooden blocks fall over"
    assert manifest["items"][0]["video_path"].endswith("_seed123.mp4")


def test_cogvideox_clean_rejects_non_positive_limit(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "A realistic video of a red ball rolling into wooden blocks, and the blocks fall over after impact. | ball | wooden blocks fall over\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_cogvideox_clean.py"),
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--limit",
            "0",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--limit must be positive" in result.stderr
