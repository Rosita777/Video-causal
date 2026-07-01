from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generate_zeroscope_clean_dry_run_writes_manifest(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "# prompt | target | effect\n"
        "A pebble drops into still water, causing ripples to spread outward. | pebble | ripples spread outward\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "zeroscope_clean"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_zeroscope_clean.py"),
            "--baseline",
            "negative_prompt",
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/zeroscope_v2_576w",
            "--seed",
            "123",
            "--steps",
            "8",
            "--num-frames",
            "16",
            "--height",
            "320",
            "--width",
            "576",
            "--dtype",
            "fp16",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "negative_prompt"
    assert manifest["pipeline"] == "TextToVideoSDPipeline"
    assert manifest["model"] == "models/zeroscope_v2_576w"
    assert manifest["generation"]["height"] == 320
    assert manifest["generation"]["width"] == 576
    assert manifest["items"][0]["negative_prompt"] == "pebble"
    assert manifest["items"][0]["video_path"].endswith("_seed123.mp4")
