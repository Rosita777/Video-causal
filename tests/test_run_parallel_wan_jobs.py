from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_wan_jobs_dry_run_expands_clean_jobs(tmp_path):
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
    output_root = tmp_path / "wan_jobs"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_parallel_wan_jobs.py"),
            "--prompts",
            str(prompts),
            "--output-root",
            str(output_root),
            "--source-indices",
            "0,2",
            "--gpus",
            "3,5",
            "--slots-per-gpu",
            "1",
            "--seed",
            "9300",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_root / "parallel_job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "models/Wan2.1-T2V-1.3B-Diffusers"
    assert manifest["baselines"] == ["clean"]
    assert manifest["max_concurrent_jobs"] == 2
    assert [(row["source_prompt_index"], row["seed"], row["gpu"]) for row in manifest["jobs"]] == [
        (0, 9300, 3),
        (2, 9302, 5),
    ]
    assert manifest["jobs"][0]["command"][:4] == [
        sys.executable,
        "scripts/generate_wan_clean.py",
        "--baseline",
        "clean",
    ]
    assert "--enable-sequential-cpu-offload" in manifest["jobs"][0]["command"]
    assert "--vae-tiling" in manifest["jobs"][0]["command"]
