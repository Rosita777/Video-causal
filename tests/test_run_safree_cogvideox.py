from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_safree_cogvideox_dry_run_records_concept_injection(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_dir = tmp_path / "safree"
    prompt_file.write_text(
        "A realistic close-up video of an ice cube dropping into cola, and bubbles rise after impact. | ice cube | bubbles rise\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / "run_safree_cogvideox.py"),
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/CogVideoX-2b",
            "--safree-root",
            str(tmp_path / "missing_safree"),
            "--seed",
            "200",
            "--steps",
            "20",
            "--guidance-scale",
            "6.0",
            "--num-frames",
            "49",
            "--fps",
            "8",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run SAFREE manifest written" in result.stdout
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "safree_cogvideox"
    assert manifest["dry_run"] is True
    assert manifest["generation"]["seed"] == 200
    assert manifest["safree"]["concept_injection"] == "target_concept_as_single_concept_dict_entry"
    assert manifest["safree"]["external_pipeline_present"] is False
    assert manifest["items"][0]["target_concept"] == "ice cube"
    assert manifest["items"][0]["safree_concept_key"] == "ice cube"
    assert manifest["items"][0]["safree_concept_terms"] == ["ice cube"]
    assert manifest["items"][0]["video_path"].endswith("_seed200.mp4")


def test_safree_cogvideox_rejects_non_positive_steps(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "A realistic close-up video of a stone falling into calm water, and circular ripples spread outward. | stone | circular ripples spread outward\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / "run_safree_cogvideox.py"),
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--steps",
            "0",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--steps must be positive" in result.stderr
