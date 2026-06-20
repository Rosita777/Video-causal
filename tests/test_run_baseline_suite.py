from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_suite_dry_run_lists_required_baselines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    missing_safree_root = tmp_path / "missing_safree"
    missing_videoeraser_root = tmp_path / "missing_videoeraser"
    missing_t2v_root = tmp_path / "missing_t2v"
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
            "--safree-root",
            str(missing_safree_root),
            "--videoeraser-root",
            str(missing_videoeraser_root),
            "--t2vunlearning-root",
            str(missing_t2v_root),
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
    assert jobs["safree_cogvideox"]["status"] == "blocked_missing_external"
    assert jobs["safree_cogvideox"]["missing"] == [
        str(missing_safree_root / "cogvideox" / "cogvideox_pipeline.py")
    ]
    assert jobs["videoeraser"]["status"] == "ready"
    assert jobs["videoeraser"]["implementation"] == "local_reimplementation"
    assert "missing" not in jobs["videoeraser"]
    assert "--mode" in jobs["videoeraser"]["command"]
    assert "local" in jobs["videoeraser"]["command"]
    assert jobs["t2vunlearning"]["status"] == "blocked_missing_external"
    assert jobs["t2vunlearning"]["missing"] == [
        str(missing_t2v_root / "test_cogvideo.py"),
        str(missing_t2v_root / "receler" / "concept_reg_cogvideo.py"),
    ]


def test_baseline_suite_marks_safree_ready_when_external_pipeline_exists(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    safree_root = tmp_path / "SAFREE"
    (safree_root / "cogvideox").mkdir(parents=True)
    (safree_root / "cogvideox" / "cogvideox_pipeline.py").write_text("# fake external pipeline\n", encoding="utf-8")
    prompt_file.write_text(
        "A realistic close-up video of an ice cube dropping into cola, and bubbles rise. | ice cube | bubbles rise\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baseline_suite.py"),
            "--baseline",
            "safree_cogvideox",
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--model",
            "models/CogVideoX-2b",
            "--safree-root",
            str(safree_root),
            "--dtype",
            "fp32",
            "--seed",
            "200",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 1
    assert manifest["external"]["safree_root"] == str(safree_root)
    job = manifest["jobs"][0]
    assert job["baseline"] == "safree_cogvideox"
    assert job["status"] == "ready"
    assert job["command"][:2] == [sys.executable, "scripts/adapters/run_safree_cogvideox.py"]
    assert "--safree-root" in job["command"]
    assert str(safree_root) in job["command"]
    assert "--dtype" in job["command"]
    assert "fp32" in job["command"]



def test_baseline_suite_marks_videoeraser_ready_when_external_runner_exists(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    videoeraser_root = tmp_path / "VideoEraser"
    (videoeraser_root / "ModelScope").mkdir(parents=True)
    (videoeraser_root / "ModelScope" / "inference.py").write_text("# fake runner\n", encoding="utf-8")
    prompt_file.write_text(
        "A realistic close-up video of a stone falling into calm water, and circular ripples spread outward. | stone | circular ripples spread outward\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baseline_suite.py"),
            "--baseline",
            "videoeraser",
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--videoeraser-root",
            str(videoeraser_root),
            "--videoeraser-mode",
            "external",
            "--seed",
            "200",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["external"]["videoeraser_root"] == str(videoeraser_root)
    job = manifest["jobs"][0]
    assert job["baseline"] == "videoeraser"
    assert job["status"] == "ready"
    assert job["command"][:2] == [sys.executable, "scripts/adapters/run_videoeraser_cogvideox.py"]
    assert "--videoeraser-root" in job["command"]
    assert str(videoeraser_root) in job["command"]


def test_baseline_suite_marks_t2vunlearning_ready_when_external_runner_exists(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_root = tmp_path / "suite"
    t2v_root = tmp_path / "T2VUnlearning"
    (t2v_root / "receler").mkdir(parents=True)
    (t2v_root / "test_cogvideo.py").write_text("# fake runner\n", encoding="utf-8")
    (t2v_root / "receler" / "concept_reg_cogvideo.py").write_text("# fake training\n", encoding="utf-8")
    prompt_file.write_text(
        "A realistic close-up video of an ice cube dropping into cola, and bubbles rise. | ice cube | bubbles rise\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_baseline_suite.py"),
            "--baseline",
            "t2vunlearning",
            "--prompts",
            str(prompt_file),
            "--output-root",
            str(output_root),
            "--t2vunlearning-root",
            str(t2v_root),
            "--seed",
            "200",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["external"]["t2vunlearning_root"] == str(t2v_root)
    job = manifest["jobs"][0]
    assert job["baseline"] == "t2vunlearning"
    assert job["status"] == "ready"
    assert job["command"][:2] == [sys.executable, "scripts/adapters/run_t2vunlearning_cogvideox.py"]
    assert "--t2vunlearning-root" in job["command"]
    assert str(t2v_root) in job["command"]


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
