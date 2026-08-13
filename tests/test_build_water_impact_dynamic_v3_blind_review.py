from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_water_impact_dynamic_v3_blind_review import (  # noqa: E402
    EXPECTED_MODEL,
    EXPECTED_PROMPTS,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_TRAIN_MANIFEST,
    artifact_sha256,
    cache_inventory_sha256,
    expected_training_schedule,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_run(
    tmp_path: Path,
    label: str,
    eval_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    cache_dir: Path,
) -> Path:
    run_dir = tmp_path / label
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True)
    items = []
    for index, row in enumerate(eval_rows):
        video = video_dir / f"{index:03d}_{label}_seed{row['seed']}.mp4"
        video.write_bytes(b"video")
        items.append(
            {
                "index": index,
                "prompt": row["training_prompt"],
                "seed": int(row["seed"]),
                "video_path": str(video),
            }
        )

    lora_path: Path | None = None
    lora_hash: str | None = None
    lora_scale = 1.0
    if label in {"balanced", "exposure"}:
        lora_path = tmp_path / f"checkpoint_{label}"
        lora_path.mkdir()
        (lora_path / "adapter.safetensors").write_bytes(label.encode("utf-8"))
        balanced = label == "balanced"
        counts, order_hash = expected_training_schedule(train_rows, balanced=balanced)
        state = {
            "step": 200,
            "max_steps": 200,
            "manifest": EXPECTED_TRAIN_MANIFEST,
            "manifest_sha256": EXPECTED_TRAIN_SHA256,
            "model": EXPECTED_MODEL,
            "cache_dir": str(cache_dir),
            "cache_entry_count": len(train_rows),
            "cache_inventory_sha256": cache_inventory_sha256(
                sorted(cache_dir.glob("*.pt"))
            ),
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "grad_accum": 1,
            "device": "cuda",
            "rank": 16,
            "alpha": 16,
            "learning_rate": 5e-5,
            "seed": 26000,
            "initial_lora_sha256": "a" * 64,
            "role": "all",
            "objective": "plain",
            "mask_weight": 4.0,
            "background_weight": 1.0,
            "pair_weight": 1.0,
            "pair_margin": 0.05,
            "redirect_weight": 1.0,
            "object_weight": 1.0,
            "receiver_weight": 1.0,
            "preserve_weight": 4.0,
            "balanced_roles": balanced,
            "role_step_counts": counts,
            "sample_order_sha256": order_hash,
            "causal_gate_dir": None,
            "gate_floor": 0.0,
            "activation_gate_dir": None,
            "component_gate_dir": None,
            "target_phrase": [],
            "persistent_causal_time": False,
        }
        (lora_path / "training_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        lora_hash = artifact_sha256(lora_path)
        lora_scale = 1.25

    generation = {
        "baseline": "clean",
        "seeds": [int(row["seed"]) for row in eval_rows],
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "activation_gate_dir": None,
        "persistent_activation_gate": False,
        "lora_target_phrases": [],
        "attention_gate_dir": None,
        "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
        "lora_path": str(lora_path) if lora_path else None,
        "lora_sha256": lora_hash,
        "lora_scale": lora_scale,
    }
    (run_dir / "generation_manifest.json").write_text(
        json.dumps(
            {
                "baseline": "clean",
                "pipeline": "WanPipeline",
                "model": EXPECTED_MODEL,
                "dry_run": False,
                "prompts": EXPECTED_PROMPTS,
                "generation": generation,
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    eval_csv = PROJECT_ROOT / "data" / "water_impact_dynamic_v1" / "eval12.csv"
    train_csv = PROJECT_ROOT / EXPECTED_TRAIN_MANIFEST
    eval_rows = read_csv(eval_csv)
    train_rows = read_csv(train_csv)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for index, row in enumerate(train_rows):
        (cache_dir / f"{index:03d}_{row['scene_id']}.pt").write_bytes(b"cache")
    original = write_run(tmp_path, "original", eval_rows, train_rows, cache_dir)
    balanced = write_run(tmp_path, "balanced", eval_rows, train_rows, cache_dir)
    exposure = write_run(tmp_path, "exposure", eval_rows, train_rows, cache_dir)
    return eval_csv, original, balanced, exposure, cache_dir


def run_builder(
    tmp_path: Path,
    eval_csv: Path,
    original: Path,
    balanced: Path,
    exposure: Path,
    cache_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_water_impact_dynamic_v3_blind_review.py"),
            "--eval-csv",
            str(eval_csv),
            "--original-dir",
            str(original),
            "--balanced-dir",
            str(balanced),
            "--exposure-dir",
            str(exposure),
            "--cache-dir",
            str(cache_dir),
            "--expected-cache-sha256",
            cache_inventory_sha256(sorted(cache_dir.glob("*.pt"))),
            "--output-dir",
            str(tmp_path / "review"),
            "--blind-seed",
            "7",
            "--skip-frame-extraction",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )


def test_builds_blinded_two_arm_review_and_answer_key(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)
    result = run_builder(tmp_path, eval_csv, original, balanced, exposure, cache_dir)

    assert result.returncode == 0, result.stderr
    output = tmp_path / "review"
    review = read_csv(output / "blind_review.csv")
    key = read_csv(output / "answer_key.csv")
    assert len(review) == 24
    assert len(key) == 24
    assert "method" not in review[0]
    assert "video_path" not in review[0]
    assert {row["method"] for row in key} == {"balanced", "exposure"}
    assert {row["review_id"] for row in review} == {row["review_id"] for row in key}
    assert all(row["target_visibility_0_absent_2_clear"] == "" for row in review)
    assert all(row["generalization_group"] for row in key)
    manifest = json.loads((output / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_rows"] == 24
    assert manifest["sample_count"] == 12
    assert (
        manifest["training_provenance"]["balanced"]["initial_lora_sha256"]
        == manifest["training_provenance"]["exposure"]["initial_lora_sha256"]
    )


def test_rejects_seed_mismatch(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)
    manifest_path = exposure / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_builder(tmp_path, eval_csv, original, balanced, exposure, cache_dir)

    assert result.returncode != 0
    assert "seed mismatch" in result.stderr


def test_rejects_wrong_training_role_counts(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)
    manifest_path = exposure / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lora_path = Path(manifest["generation"]["lora_path"])
    state_path = lora_path / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["role_step_counts"] = {"erase": 167, "preserve": 33}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest["generation"]["lora_sha256"] = artifact_sha256(lora_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_builder(tmp_path, eval_csv, original, balanced, exposure, cache_dir)

    assert result.returncode != 0
    assert "role_step_counts" in result.stderr


def test_rejects_swapped_training_arms(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)

    result = run_builder(tmp_path, eval_csv, original, exposure, balanced, cache_dir)

    assert result.returncode != 0
    assert "balanced_roles" in result.stderr


def test_rejects_lora_artifact_changed_after_generation(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)
    manifest = json.loads(
        (exposure / "generation_manifest.json").read_text(encoding="utf-8")
    )
    lora_path = Path(manifest["generation"]["lora_path"])
    (lora_path / "adapter.safetensors").write_bytes(b"changed")

    result = run_builder(tmp_path, eval_csv, original, balanced, exposure, cache_dir)

    assert result.returncode != 0
    assert "LoRA artifact hash mismatch" in result.stderr


def test_rejects_generation_setting_mismatch(tmp_path: Path) -> None:
    eval_csv, original, balanced, exposure, cache_dir = build_fixture(tmp_path)
    manifest_path = exposure / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["guidance_scale"] = 6.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_builder(tmp_path, eval_csv, original, balanced, exposure, cache_dir)

    assert result.returncode != 0
    assert "generation guidance_scale mismatch" in result.stderr
