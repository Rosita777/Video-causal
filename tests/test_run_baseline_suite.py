from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_suite_dry_run_lists_required_baselines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    prompt_file.write_text(
        "A realistic close-up video of a stone falling into calm water, and circular ripples spread outward. | stone | circular ripples spread outward\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baseline_suite.py"),
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--model",
            "models/CogVideoX-2b",
            "--seed",
            "200",
            "--steps",
            "20",
            "--parallel",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Baseline suite manifest written" in result.stdout
    manifest = json.loads((output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parallel"] is True
    jobs = {job["baseline"]: job for job in manifest["jobs"]}

    assert list(jobs) == [
        "negative_prompt",
        "safree_cogvideox",
        "videoeraser",
        "t2vunlearning",
    ]
    assert jobs["negative_prompt"]["status"] == "ready"
    assert jobs["negative_prompt"]["command"][:4] == [
        sys.executable,
        "scripts/generate_cogvideox_clean.py",
        "--baseline",
        "negative_prompt",
    ]
    assert jobs["negative_prompt"]["output_dir"] == str(output_root / "negative_prompt")
    assert jobs["safree_cogvideox"]["status"] == "blocked_missing_adapter"
    assert jobs["videoeraser"]["status"] == "blocked_missing_adapter"
    assert jobs["t2vunlearning"]["status"] == "blocked_missing_adapter"


def test_baseline_suite_can_select_single_baseline(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    prompt_file.write_text(
        "A realistic close-up video of an ice cube dropping into cola, and bubbles rise. | ice cube | bubbles rise\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baseline_suite.py"),
            "--baseline",
            "negative_prompt",
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert [job["baseline"] for job in manifest["jobs"]] == ["negative_prompt"]
