from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_zeroscope_jobs_dry_run_expands_matrix(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "\n".join(
            [
                "Prompt zero. | pebble | ripples",
                "Prompt one. | baseball | cracks",
                "Prompt two. | stamp | ink mark",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "zeroscope_jobs"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_parallel_zeroscope_jobs.py"),
            "--baseline",
            "clean",
            "--baseline",
            "videoeraser",
            "--prompts",
            str(prompts),
            "--output-root",
            str(output_root),
            "--source-indices",
            "0,2",
            "--gpus",
            "0,1",
            "--slots-per-gpu",
            "2",
            "--seed",
            "500",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_root / "parallel_job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "models/zeroscope_v2_576w"
    assert manifest["baselines"] == ["clean", "videoeraser"]
    assert manifest["max_concurrent_jobs"] == 4
    assert [(row["baseline"], row["source_prompt_index"], row["seed"]) for row in manifest["jobs"]] == [
        ("clean", 0, 500),
        ("videoeraser", 0, 500),
        ("clean", 2, 502),
        ("videoeraser", 2, 502),
    ]
    assert manifest["jobs"][0]["command"][:4] == [
        sys.executable,
        "scripts/generate_zeroscope_clean.py",
        "--baseline",
        "clean",
    ]
    assert manifest["jobs"][1]["command"][:2] == [
        sys.executable,
        "scripts/adapters/run_videoeraser_zeroscope.py",
    ]
