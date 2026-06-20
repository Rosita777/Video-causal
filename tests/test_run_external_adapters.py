from pathlib import Path
import json
import os
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_adapter(script_name, tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    output_dir = tmp_path / "outputs"
    prompt_file.write_text(
        "A realistic close-up video of a stone falling into calm water, and circular ripples spread outward. | stone | circular ripples spread outward\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adapters" / script_name),
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(output_dir),
            "--model",
            "models/CogVideoX-2b",
            "--external-root",
            str(tmp_path / "missing_external"),
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
    return result, json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))


def test_videoeraser_adapter_dry_run_records_local_reimplementation_contract(tmp_path):
    result, manifest = run_adapter("run_videoeraser_cogvideox.py", tmp_path)

    assert "Dry-run videoeraser manifest written" in result.stdout
    assert manifest["baseline"] == "videoeraser"
    assert manifest["dry_run"] is True
    assert manifest["implementation"]["selected_mode"] == "local_reimplementation"
    assert manifest["implementation"]["local_method"] == "spea_arng_cogvideox_v0"
    assert manifest["external"]["required_files_present"] is False
    item = manifest["items"][0]
    assert item["target_concept"] == "stone"
    assert item["videoeraser"]["negative_prompt"] == "stone"
    assert item["videoeraser"]["erased_prompt"] != item["prompt"]
    assert item["video_path"].endswith("_seed200.mp4")


def test_t2vunlearning_adapter_dry_run_records_local_reimplementation_contract(tmp_path):
    result, manifest = run_adapter("run_t2vunlearning_cogvideox.py", tmp_path)

    assert "Dry-run t2vunlearning manifest written" in result.stdout
    assert manifest["baseline"] == "t2vunlearning"
    assert manifest["dry_run"] is True
    assert manifest["implementation"]["selected_mode"] == "local_reimplementation"
    assert manifest["implementation"]["local_method"] == "receler_cogvideox_proxy_v0"
    assert manifest["implementation"]["eraser_rank"] == 128
    assert manifest["external"]["required_files_present"] is False
    item = manifest["items"][0]
    assert item["target_concept"] == "stone"
    assert item["t2vunlearning"]["unlearn_concept"] == "stone"
    assert item["t2vunlearning"]["negative_prompt"] == "stone"
    assert item["video_path"].endswith("_seed200.mp4")


def write_fake_runner(path: Path, marker_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
import argparse
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--prompts', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--seed', required=True)
parser.add_argument('--steps', required=True)
parser.add_argument('--guidance-scale', required=True)
parser.add_argument('--limit')
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / '{marker_name}').write_text(
    json.dumps(
        {{
            'cwd': os.getcwd(),
            'prompts': args.prompts,
            'output_dir': args.output_dir,
            'model': args.model,
        }},
        indent=2,
    )
    + '\\n',
    encoding='utf-8',
)
""".format(marker_name=marker_name),
        encoding="utf-8",
    )


def run_real_adapter_with_fake_external(script_name: str, external_root: Path, tmp_path: Path, extra_args=None):
    prompt_file = tmp_path / "prompts.txt"
    output_dir = tmp_path / "outputs"
    prompt = "A realistic close-up video of an ice cube dropping into cola, and bubbles rise. | ice cube | bubbles rise"
    prompt_file.write_text(prompt + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "adapters" / script_name),
        "--prompts",
        os.path.relpath(prompt_file, PROJECT_ROOT),
        "--output-dir",
        os.path.relpath(output_dir, PROJECT_ROOT),
        "--model",
        "models/CogVideoX-2b",
        "--external-root",
        os.path.relpath(external_root, PROJECT_ROOT),
        "--seed",
        "200",
        "--steps",
        "20",
        "--guidance-scale",
        "6.0",
    ]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_dir


def test_videoeraser_adapter_real_run_uses_absolute_paths_for_external_runner(tmp_path):
    external_root = tmp_path / "VideoEraser"
    write_fake_runner(external_root / "ModelScope" / "inference.py", "external_call.json")

    output_dir = run_real_adapter_with_fake_external("run_videoeraser_cogvideox.py", external_root, tmp_path, ["--mode", "external"])

    call = json.loads((output_dir / "external_call.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert Path(call["cwd"]).resolve() == external_root.resolve()
    assert Path(call["prompts"]).is_absolute()
    assert Path(call["output_dir"]).is_absolute()
    assert Path(call["model"]).is_absolute()
    assert manifest["dry_run"] is False


def test_t2vunlearning_adapter_real_run_uses_absolute_paths_for_external_runner(tmp_path):
    external_root = tmp_path / "T2VUnlearning"
    write_fake_runner(external_root / "test_cogvideo.py", "external_call.json")
    (external_root / "receler").mkdir(parents=True, exist_ok=True)
    training_file = external_root / "receler" / "concept_reg_cogvideo.py"
    training_file.write_text("# fake training" + "\n", encoding="utf-8")

    output_dir = run_real_adapter_with_fake_external("run_t2vunlearning_cogvideox.py", external_root, tmp_path, ["--mode", "external"])

    call = json.loads((output_dir / "external_call.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    assert Path(call["cwd"]).resolve() == external_root.resolve()
    assert Path(call["prompts"]).is_absolute()
    assert Path(call["output_dir"]).is_absolute()
    assert Path(call["model"]).is_absolute()
    assert manifest["dry_run"] is False

