from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parallel_jobs_dry_run_expands_prompt_baseline_matrix(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "jobs"
    prompt_file.write_text(
        "\n".join(
            [
                "A realistic close-up video of a pebble dropping into water, and ripples spread outward. | pebble | ripples spread outward",
                "A realistic close-up video of a baseball striking glass, and cracks spread outward. | baseball | cracks spread outward",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_parallel_baseline_jobs.py"),
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--model",
            "models/CogVideoX-2b",
            "--seed",
            "700",
            "--steps",
            "20",
            "--gpus",
            "0,1",
            "--slots-per-gpu",
            "2",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Parallel baseline job manifest written" in result.stdout
    manifest = json.loads((output_root / "parallel_job_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["gpus"] == [0, 1]
    assert manifest["slots_per_gpu"] == 2
    assert len(manifest["jobs"]) == 8
    assert [job["slot"] for job in manifest["jobs"][:4]] == ["gpu0_slot0", "gpu0_slot1", "gpu1_slot0", "gpu1_slot1"]

    first = manifest["jobs"][0]
    assert first["baseline"] == "negative_prompt"
    assert first["prompt_index"] == 0
    assert first["source_prompt_index"] == 0
    assert first["seed"] == 700
    assert first["prompt_shard"].endswith("prompt_shards/prompt_000.txt")
    assert first["output_dir"].endswith("negative_prompt_shards/prompt_000")
    assert first["command"][:4] == [
        sys.executable,
        "scripts/generate_cogvideox_clean.py",
        "--baseline",
        "negative_prompt",
    ]
    assert "--device" in first["command"]
    assert "cuda" in first["command"]


def test_parallel_jobs_can_select_baselines_and_source_indices(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "jobs"
    prompt_file.write_text(
        "\n".join(
            [
                "Prompt zero. | zero | effect zero",
                "Prompt one. | one | effect one",
                "Prompt two. | two | effect two",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_parallel_baseline_jobs.py"),
            "--baseline",
            "videoeraser",
            "--baseline",
            "t2vunlearning",
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--source-indices",
            "0,2",
            "--seed",
            "900",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_root / "parallel_job_manifest.json").read_text(encoding="utf-8"))
    assert [(job["baseline"], job["source_prompt_index"], job["seed"]) for job in manifest["jobs"]] == [
        ("videoeraser", 0, 900),
        ("t2vunlearning", 0, 900),
        ("videoeraser", 2, 902),
        ("t2vunlearning", 2, 902),
    ]
    assert all("--mode" in job["command"] and "local" in job["command"] for job in manifest["jobs"])
