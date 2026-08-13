from pathlib import Path
import json
from types import SimpleNamespace
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_wan_clean import select_prompt_encode_device  # noqa: E402


def test_generate_wan_clean_dry_run_writes_manifest(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "# prompt | target | effect\n"
        "A pebble drops into still water, causing ripples to spread outward. | pebble | ripples spread outward\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "wan_clean"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_wan_clean.py"),
            "--baseline",
            "negative_prompt",
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/Wan2.1-T2V-1.3B-Diffusers",
            "--seed",
            "123",
            "--steps",
            "8",
            "--num-frames",
            "17",
            "--height",
            "320",
            "--width",
            "576",
            "--dtype",
            "bf16",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == "negative_prompt"
    assert manifest["pipeline"] == "WanPipeline"
    assert manifest["model"] == "models/Wan2.1-T2V-1.3B-Diffusers"
    assert manifest["generation"]["height"] == 320
    assert manifest["generation"]["width"] == 576
    assert manifest["items"][0]["negative_prompt"] == "pebble"
    assert manifest["items"][0]["video_path"].endswith("_seed123.mp4")


def test_generate_wan_clean_accepts_explicit_seed_list(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "First static scene. | droplet | still surface\n"
        "Second static scene. | droplet | dry surface\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "wan_explicit_seeds"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_wan_clean.py"),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "8300,8316",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in manifest["items"]] == [8300, 8316]
    assert manifest["generation"]["seeds"] == [8300, 8316]


def test_generate_wan_clean_rejects_wrong_explicit_seed_count(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "First static scene. | droplet | still surface\n"
        "Second static scene. | droplet | dry surface\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_wan_clean.py"),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(tmp_path / "bad_seed_count"),
            "--seeds",
            "8300",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "1 values but 2 prompts were selected" in result.stderr


def test_wan_clean_uses_cpu_prompt_encoding_when_offloaded():
    offloaded = SimpleNamespace(enable_sequential_cpu_offload=True, enable_model_cpu_offload=False)
    model_offloaded = SimpleNamespace(enable_sequential_cpu_offload=False, enable_model_cpu_offload=True)
    not_offloaded = SimpleNamespace(enable_sequential_cpu_offload=False, enable_model_cpu_offload=False)

    assert select_prompt_encode_device(offloaded, selected_device="cuda", cuda_available=True) == "cpu"
    assert select_prompt_encode_device(model_offloaded, selected_device="cuda", cuda_available=True) == "cpu"
    assert select_prompt_encode_device(not_offloaded, selected_device="cuda", cuda_available=True) == "cuda"
    assert select_prompt_encode_device(offloaded, selected_device="cpu", cuda_available=False) == "cpu"


def test_dry_run_records_lora_artifact_fingerprint(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A calm bowl of water. | stone | ripples\n", encoding="utf-8")
    lora = tmp_path / "checkpoint"
    lora.mkdir()
    (lora / "weights.safetensors").write_bytes(b"controlled weights")
    output_dir = tmp_path / "wan_lora"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_wan_clean.py"),
            "--prompts",
            str(prompts),
            "--output-dir",
            str(output_dir),
            "--lora-path",
            str(lora),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["lora_sha256"]
    assert len(manifest["generation"]["lora_sha256"]) == 64
