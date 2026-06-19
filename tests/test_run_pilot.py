from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def test_parse_causal_prompt_file_skips_comments_and_blank_lines(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n"
        "# comment\n"
        "a red ball knocks over blocks | ball | blocks fall\n"
        "a person opens a door | person | door opens\n",
        encoding="utf-8",
    )

    from run_pilot import parse_prompt_file

    items = parse_prompt_file(prompt_file)

    assert items == [
        {
            "prompt": "a red ball knocks over blocks",
            "target_concept": "ball",
            "expected_effect": "blocks fall",
        },
        {
            "prompt": "a person opens a door",
            "target_concept": "person",
            "expected_effect": "door opens",
        },
    ]


def test_dry_run_writes_manifest_with_all_prompts(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_dir = tmp_path / "out"
    prompt_file.write_text(
        "a red ball knocks over blocks | ball | blocks fall\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pilot.py"),
            "--baseline",
            "negative_prompt",
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Dry-run manifest written" in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "negative_prompt"
    assert manifest["dry_run"] is True
    assert manifest["items"][0]["target_concept"] == "ball"
