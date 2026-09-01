from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline  # noqa: E402
import run_causal_role_erasure_7mechanism_t2v_adapted_v2 as runner  # noqa: E402


FAKE_RUNTIME = {
    "python": platform.python_version(),
    "implementation": platform.python_implementation(),
    "packages": {"torch": "fixture", "diffusers": "fixture", "transformers": "fixture"},
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_json_bytes(value))


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "build_causal_role_erasure_7mechanism_baseline_registry_v2.py",
        "train_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
        "run_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
    ):
        shutil.copy2(SCRIPTS / name, scripts / name)

    formal = project / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
    formal.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", formal)

    model = project / "models/CogVideoX-2b"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name":"CogVideoXPipeline"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"model-weights")
    inventory = baseline.inventory_tree(model)

    baseline_root = project / "outputs/baseline_registry"
    inventory_path = baseline_root / "model_inventory.json"
    _write_json(inventory_path, inventory)

    t2v_root = project / "outputs/t2v_specs"
    training_runs = []
    checkpoint_refs = []
    for mechanism in baseline.MECHANISMS:
        checkpoint = project / "outputs/t2v_training" / mechanism / "checkpoint-000100"
        checkpoint.mkdir(parents=True)
        weights = checkpoint / "eraser_weights.pt"
        config = checkpoint / "eraser_config.json"
        state = checkpoint / "training_state.json"
        weights.write_bytes(f"weights::{mechanism}".encode("utf-8"))
        _write_json(
            config,
            {
                "eraser_type": "adapter",
                "eraser_rank": 128,
                "adapter_target": "every CogVideoX transformer block attn1",
            },
        )
        _write_json(state, {"step": 100})
        run_spec_sha = hashlib.sha256(f"spec::{mechanism}".encode()).hexdigest()
        training_runs.append(
            {
                "run_id": f"t2v7m_{mechanism}_adapted",
                "mechanism": mechanism,
                "eligible_checkpoint": checkpoint.relative_to(project).as_posix(),
                "run_spec_sha256": run_spec_sha,
            }
        )
        training_receipt = {
            "protocol": "causal_role_erasure_7mechanism_t2v_training_registry_v2",
            "protocol_version": baseline.PROTOCOL_VERSION,
            "status": "eligible",
            "run_id": f"t2v7m_{mechanism}_adapted",
            "mechanism": mechanism,
            "step": 100,
            "run_spec_sha256": run_spec_sha,
            "weights_sha256": runner.sha256_file(weights),
            "config_sha256": runner.sha256_file(config),
            "training_state_sha256": runner.sha256_file(state),
            "formal_output_inspected": False,
        }
        receipt_path = checkpoint / "training_receipt.json"
        _write_json(receipt_path, training_receipt)
        checkpoint_refs.append(
            {
                "mechanism": mechanism,
                "eligible_checkpoint": checkpoint.relative_to(project).as_posix(),
                "status": "ready",
                "weights_sha256": runner.sha256_file(weights),
                "config_sha256": runner.sha256_file(config),
                "training_state_sha256": runner.sha256_file(state),
                "receipt_sha256": runner.sha256_file(receipt_path),
            }
        )
    training_registry = {
        "protocol": "causal_role_erasure_7mechanism_t2v_training_registry_v2",
        "protocol_version": baseline.PROTOCOL_VERSION,
        "status": "t2v_training_specs_frozen_pre_training",
        "formal_output_inspected": False,
        "model_root": model.relative_to(project).as_posix(),
        "model_inventory": inventory_path.relative_to(project).as_posix(),
        "model_inventory_sha256": runner.sha256_file(inventory_path),
        "runtime_python": str(Path(sys.executable).resolve()),
        "runtime_python_sha256": runner.sha256_file(Path(sys.executable).resolve()),
        "code_sha256": {
            runner.TRAINER_RELATIVE: runner.sha256_file(project / runner.TRAINER_RELATIVE),
        },
        "runs": training_runs,
    }
    training_registry_path = t2v_root / "t2v_training_registry.json"
    _write_json(training_registry_path, training_registry)

    runtime_python = Path(sys.executable).resolve()
    registry = {
        "schema_version": 1,
        "protocol": baseline.REGISTRY_PROTOCOL,
        "protocol_version": baseline.PROTOCOL_VERSION,
        "status": "baseline_stack_frozen_ready",
        "formal_generation_authorized": True,
        "formal_cases": {
            "path": formal.relative_to(project).as_posix(),
            "sha256": runner.sha256_file(formal),
            "row_count": 294,
            "same_case_same_seed_across_streams": True,
        },
        "model": {
            "root": model.relative_to(project).as_posix(),
            "inventory_path": inventory_path.relative_to(project).as_posix(),
            "inventory_sha256": runner.sha256_file(inventory_path),
            "file_count": inventory["file_count"],
        },
        "runtime": {
            **FAKE_RUNTIME,
            "python_executable": str(runtime_python),
            "python_executable_sha256": runner.sha256_file(runtime_python),
        },
        "generation": {
            "shared": dict(baseline.COG_GENERATION),
            "streams": {runner.BASELINE: {"dtype": "bf16"}},
        },
        "implementations": {
            runner.BASELINE: {
                "status": "ready",
                "training_registry": training_registry_path.relative_to(project).as_posix(),
                "training_registry_sha256": runner.sha256_file(training_registry_path),
                "checkpoints": checkpoint_refs,
            }
        },
        "code_sha256": {
            runner.BASELINE_BUILDER_RELATIVE: runner.sha256_file(project / runner.BASELINE_BUILDER_RELATIVE),
            runner.TRAINER_RELATIVE: runner.sha256_file(project / runner.TRAINER_RELATIVE),
            runner.RUNNER_RELATIVE: runner.sha256_file(project / runner.RUNNER_RELATIVE),
        },
    }
    registry_path = baseline_root / "baseline_registry.json"
    _write_json(registry_path, registry)
    receipt_path = baseline_root / "build_receipt.json"
    _write_json(receipt_path, {"registry_sha256": runner.sha256_file(registry_path)})
    return {
        "project": project,
        "formal": formal,
        "registry": registry_path,
        "receipt": receipt_path,
    }


def _load_and_plan(fixture: dict[str, Path], output: Path, mechanism: str = "brittle_fracture"):
    registry = runner.load_registry(
        fixture["project"],
        fixture["registry"],
        fixture["receipt"],
        runtime_probe=lambda _python: dict(FAKE_RUNTIME),
    )
    plan = runner.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism=mechanism,
        output_dir=output,
    )
    return registry, plan


def test_plan_consumes_frozen_42_case_seeds_and_one_eligible_checkpoint(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_t2v/brittle_fracture"
    _registry, plan = _load_and_plan(fixture, output)
    with fixture["formal"].open(newline="", encoding="utf-8") as handle:
        expected = [
            row for row in csv.DictReader(handle) if row["mechanism"] == "brittle_fracture"
        ]
    assert len(plan["items"]) == 42
    assert [item["case_id"] for item in plan["items"]] == [row["case_id"] for row in expected]
    assert [item["seed"] for item in plan["items"]] == [int(row["seed"]) for row in expected]
    assert plan["generation"]["num_frames"] == 49
    assert plan["generation"]["fps"] == 8
    assert plan["generation"]["height"] == 480
    assert plan["generation"]["width"] == 720
    assert plan["checkpoint"]["path"].endswith(
        "outputs/t2v_training/brittle_fracture/checkpoint-000100"
    )


def test_per_video_receipts_allow_only_exact_resume(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_t2v/water_impact"
    registry, full_plan = _load_and_plan(fixture, output, "water_impact")
    plan = {**full_plan, "items": full_plan["items"][:2]}

    def fake_render(_torch, _export, _pipe, item, _generation):
        path = Path(item["video_path"])
        path.write_bytes(f"video::{item['case_id']}".encode("utf-8"))

    monkeypatch.setattr(runner, "render_item", fake_render)
    result = runner.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(runner.MEDIA),
        pipeline_loader=lambda _registry, _root, _checkpoint: (object(), object(), object()),
    )
    assert result == {"status": "complete", "completed": 2, "expected": 2}
    receipts = sorted((output / "receipts").glob("*.json"))
    assert len(receipts) == 2
    first = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert first["generation_plan_sha256"] == runner.sha256_file(output / "generation_plan.json")
    assert first["media"] == runner.MEDIA

    resumed = runner.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=True,
        dry_run=False,
        media_probe=lambda _python, _video: dict(runner.MEDIA),
        pipeline_loader=lambda *_args: (_ for _ in ()).throw(AssertionError("pipeline loaded on exact resume")),
    )
    assert resumed["completed"] == 2

    first_video = Path(plan["items"][0]["video_path"])
    first_video.write_bytes(b"tampered")
    with pytest.raises(runner.T2VGenerationError, match="receipt mismatch"):
        runner.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(runner.MEDIA),
        )


def test_fresh_only_and_unreceipted_video_are_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_t2v/surface_trace"
    registry, full_plan = _load_and_plan(fixture, output, "surface_trace")
    plan = {**full_plan, "items": full_plan["items"][:1]}

    def fake_render(_torch, _export, _pipe, item, _generation):
        Path(item["video_path"]).write_bytes(b"video")

    monkeypatch.setattr(runner, "render_item", fake_render)
    runner.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(runner.MEDIA),
        pipeline_loader=lambda *_args: (object(), object(), object()),
    )
    with pytest.raises(runner.T2VGenerationError, match="fresh-only"):
        runner.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=False,
            dry_run=False,
        )
    Path(plan["items"][0]["receipt_path"]).unlink()
    with pytest.raises(runner.T2VGenerationError, match="unreceipted formal video"):
        runner.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
        )


def test_media_contract_and_checkpoint_hashes_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"video")
    with pytest.raises(runner.T2VGenerationError, match="media contract mismatch"):
        runner.validate_video(
            Path(sys.executable).resolve(),
            video,
            lambda _python, _video: {
                "decoded_frames": 48,
                "fps": "8/1",
                "height": 480,
                "width": 720,
            },
        )

    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    checkpoint = fixture["project"] / registry["implementations"][runner.BASELINE]["checkpoints"][0]["eligible_checkpoint"]
    (checkpoint / "eraser_weights.pt").write_bytes(b"tampered")
    with pytest.raises(runner.T2VGenerationError, match="weights changed"):
        runner.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            runtime_probe=lambda _python: dict(FAKE_RUNTIME),
        )
