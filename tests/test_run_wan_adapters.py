from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_prompts(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "A pebble drops into still water, causing ripples to spread outward. | pebble | ripples spread outward\n",
        encoding="utf-8",
    )
    return prompts


def run_adapter(script_name: str, tmp_path: Path) -> dict:
    prompts = write_prompts(tmp_path)
    output_dir = tmp_path / script_name.removesuffix(".py")
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / script_name),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/Wan2.1-T2V-1.3B-Diffusers",
            "--seed",
            "321",
            "--steps",
            "8",
            "--num-frames",
            "17",
            "--height",
            "480",
            "--width",
            "832",
            "--dtype",
            "bf16",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))


def test_videoeraser_wan_dry_run_records_proxy_contract(tmp_path):
    manifest = run_adapter("run_videoeraser_wan.py", tmp_path)

    assert manifest["baseline"] == "videoeraser"
    assert manifest["pipeline"] == "WanPipeline"
    assert manifest["implementation"]["local_method"] == "spea_arng_wan_proxy_v0"
    item = manifest["items"][0]
    assert item["videoeraser"]["negative_prompt"] == "pebble"
    assert "pebble" not in item["videoeraser"]["erased_prompt"].lower()


def test_t2vunlearning_wan_dry_run_records_proxy_contract(tmp_path):
    manifest = run_adapter("run_t2vunlearning_wan.py", tmp_path)

    assert manifest["baseline"] == "t2vunlearning"
    assert manifest["pipeline"] == "WanPipeline"
    assert manifest["implementation"]["local_method"] == "receler_wan_proxy_v0"
    item = manifest["items"][0]
    assert item["t2vunlearning"]["unlearn_concept"] == "pebble"
    assert "pebble" not in item["t2vunlearning"]["unlearn_prompt"].lower()


def test_safree_wan_dry_run_records_proxy_contract(tmp_path):
    manifest = run_adapter("run_safree_wan.py", tmp_path)

    assert manifest["baseline"] == "safree_wan"
    assert manifest["pipeline"] == "WanPipeline"
    assert manifest["implementation"]["local_method"] == "safree_wan_proxy_v0"
    item = manifest["items"][0]
    assert item["safree"]["unsafe_concept"] == "pebble"
    assert item["safree"]["method"] == "concept_direction_subtraction"
