from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline  # noqa: E402
import build_causal_role_erasure_7mechanism_t2v_training_registry_v2 as t2v_registry  # noqa: E402
import run_causal_role_erasure_7mechanism_cogvideox_controls_v2 as controls  # noqa: E402


CODE_FILES = (
    "build_causal_role_erasure_7mechanism_baseline_registry_v2.py",
    "run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py",
    "build_causal_role_erasure_7mechanism_t2v_training_registry_v2.py",
    "train_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
    "run_causal_role_erasure_7mechanism_t2v_training_queue_v2.sh",
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(SCRIPTS / name, scripts / name)

    formal = project / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
    formal.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", formal)

    model = project / "models/CogVideoX-2b"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name":"CogVideoXPipeline"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"cogvideox-weights")
    inventory = baseline.inventory_tree(model)
    inventory_path = project / "data/cog_model_inventory.json"
    inventory_path.write_bytes(baseline.canonical_json_bytes(inventory))

    selected_root = project / "outputs/training_inputs/selected_by_mechanism"
    mechanism_refs = {}
    for mechanism_index, mechanism in enumerate(baseline.MECHANISMS):
        rows = [
            {
                "protocol_version": baseline.PROTOCOL_VERSION,
                "mechanism": mechanism,
                "selected_index": index,
                "candidate_id": f"{mechanism}_candidate_{index:03d}",
                "factual_prompt": f"Prompt {index} for {mechanism}.",
            }
            for index in range(178)
        ]
        path = selected_root / f"{mechanism_index:02d}_{mechanism}_selected178.csv"
        _write_csv(path, rows)
        mechanism_refs[mechanism] = {
            "selected178": {
                "path": path.relative_to(project).as_posix(),
                "row_count": 178,
                "sha256": baseline.sha256_file(path),
            }
        }
    training_inputs = project / "outputs/training_inputs/training_input_registry.json"
    training_inputs.write_bytes(
        baseline.canonical_json_bytes(
            {
                "protocol_version": baseline.PROTOCOL_VERSION,
                "counts": {"selected_per_mechanism": 178},
                "mechanisms": mechanism_refs,
            }
        )
    )

    videoeraser = project / "recovered/VideoEraser"
    (videoeraser / "CogVideoX").mkdir(parents=True)
    (videoeraser / "CogVideoX/cogvideox_pipeline.py").write_text("# official pipeline fixture\n", encoding="utf-8")
    (videoeraser / "OFFICIAL_COMMIT").write_text(baseline.EXPECTED_VIDEOERASER_COMMIT + "\n", encoding="utf-8")

    safree = project / "baselines/external/SAFREE"
    (safree / "cogvideox").mkdir(parents=True)
    (safree / "cogvideox/cogvideox_pipeline.py").write_text("# safree pipeline fixture\n", encoding="utf-8")
    (safree / "SAFREE_COMMIT").write_text("1" * 40 + "\n", encoding="utf-8")
    return {
        "project": project,
        "formal": formal,
        "model": model,
        "model_inventory": inventory_path,
        "training_inputs": training_inputs,
        "videoeraser": videoeraser,
        "safree": safree,
        "runtime_python": Path(sys.executable).resolve(),
        "t2v_specs": project / "outputs/t2v_specs",
        "t2v_checkpoints": project / "outputs/t2v_checkpoints",
        "baseline_registry": project / "outputs/baseline_registry",
    }


def _build_t2v(f: dict[str, Path], train_rows: int = 36) -> dict:
    return t2v_registry.build_training_registry(
        project_root=f["project"],
        training_input_registry=f["training_inputs"],
        model_root=f["model"],
        model_inventory=f["model_inventory"],
        runtime_python=f["runtime_python"],
        output_root=f["t2v_specs"],
        checkpoint_root=f["t2v_checkpoints"],
        train_rows_per_mechanism=train_rows,
        max_steps=100,
        eligible_checkpoint_step=100,
    )


def _materialize_t2v_checkpoints(f: dict[str, Path], registry: dict) -> None:
    for run in registry["runs"]:
        checkpoint = f["project"] / run["eligible_checkpoint"]
        checkpoint.mkdir(parents=True)
        weights = checkpoint / "eraser_weights.pt"
        config = checkpoint / "eraser_config.json"
        state = checkpoint / "training_state.json"
        weights.write_bytes(f"weights::{run['mechanism']}".encode())
        config.write_text('{"eraser_rank":128}\n', encoding="utf-8")
        state.write_text('{"step":100}\n', encoding="utf-8")
        receipt = {
            "status": "eligible",
            "mechanism": run["mechanism"],
            "run_spec_sha256": run["run_spec_sha256"],
            "weights_sha256": baseline.sha256_file(weights),
            "config_sha256": baseline.sha256_file(config),
        }
        (checkpoint / "training_receipt.json").write_bytes(baseline.canonical_json_bytes(receipt))


def _fake_runtime(_python: Path) -> dict:
    return {
        "python": "3.11.0",
        "implementation": "CPython",
        "executable": str(_python),
        "packages": {"torch": "2.6.0", "diffusers": "0.33.1", "transformers": "4.51.3"},
    }


def test_t2v_registry_replaces_hardcoded_row_contract_and_freezes_seven_runs(tmp_path):
    f = _fixture(tmp_path)
    registry = _build_t2v(f, train_rows=41)
    assert registry["status"] == "t2v_training_specs_frozen_pre_training"
    assert registry["train_rows_per_mechanism"] == 41
    assert len(registry["runs"]) == 7
    for run in registry["runs"]:
        spec_path = f["project"] / run["run_spec"]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        rows_path = f["project"] / spec["training_rows"]["path"]
        with rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 41
        assert len({row["candidate_id"] for row in rows}) == 41
        assert spec["parameters"]["eligible_checkpoint_step"] == 100
        assert spec["checkpoint_selection"].startswith("precommitted")


def test_baseline_registry_snapshots_official_videoeraser_and_binds_ready_stack(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    t2v = _build_t2v(f)
    _materialize_t2v_checkpoints(f, t2v)
    fixture_pipeline = f["videoeraser"] / "CogVideoX/cogvideox_pipeline.py"
    monkeypatch.setattr(baseline, "EXPECTED_VIDEOERASER_PIPELINE_SHA256", baseline.sha256_file(fixture_pipeline))
    registry = baseline.build_registry(
        project_root=f["project"],
        formal_cases=f["formal"],
        model_root=f["model"],
        runtime_python=f["runtime_python"],
        videoeraser_source_root=f["videoeraser"],
        safree_root=f["safree"],
        t2v_registry=f["t2v_specs"] / "t2v_training_registry.json",
        output_root=f["baseline_registry"],
        safree_expected_commit="1" * 40,
        safree_expected_pipeline_sha256=baseline.sha256_file(f["safree"] / "cogvideox/cogvideox_pipeline.py"),
        require_complete=True,
        runtime_probe=_fake_runtime,
    )
    assert registry["formal_generation_authorized"] is True
    assert registry["implementations"]["videoeraser_official"]["commit"] == baseline.EXPECTED_VIDEOERASER_COMMIT
    copied = f["baseline_registry"] / "external_snapshot/VideoEraser/CogVideoX/cogvideox_pipeline.py"
    assert baseline.sha256_file(copied) == baseline.sha256_file(fixture_pipeline)
    assert registry["implementations"]["safree_cogvideox"]["commit"] == "1" * 40
    assert all(item["status"] == "ready" for item in registry["implementations"]["t2vunlearning_adapted"]["checkpoints"])


def test_baseline_registry_records_missing_safree_as_blocker(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    shutil.rmtree(f["safree"])
    fixture_pipeline = f["videoeraser"] / "CogVideoX/cogvideox_pipeline.py"
    monkeypatch.setattr(baseline, "EXPECTED_VIDEOERASER_PIPELINE_SHA256", baseline.sha256_file(fixture_pipeline))
    registry = baseline.build_registry(
        project_root=f["project"],
        formal_cases=f["formal"],
        model_root=f["model"],
        runtime_python=f["runtime_python"],
        videoeraser_source_root=f["videoeraser"],
        safree_root=f["safree"],
        t2v_registry=None,
        output_root=f["baseline_registry"],
        runtime_probe=_fake_runtime,
    )
    assert registry["formal_generation_authorized"] is False
    assert registry["blocked_reasons"] == ["safree_cogvideox", "t2vunlearning_adapted"]
    assert registry["implementations"]["safree_cogvideox"]["status"] == "blocked_missing_external_clone"


def test_controls_keep_arbitrary_formal_seeds_and_checkpoint_full_media(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    t2v = _build_t2v(f)
    _materialize_t2v_checkpoints(f, t2v)
    fixture_pipeline = f["videoeraser"] / "CogVideoX/cogvideox_pipeline.py"
    monkeypatch.setattr(baseline, "EXPECTED_VIDEOERASER_PIPELINE_SHA256", baseline.sha256_file(fixture_pipeline))
    baseline.build_registry(
        project_root=f["project"],
        formal_cases=f["formal"],
        model_root=f["model"],
        runtime_python=f["runtime_python"],
        videoeraser_source_root=f["videoeraser"],
        safree_root=f["safree"],
        t2v_registry=f["t2v_specs"] / "t2v_training_registry.json",
        output_root=f["baseline_registry"],
        safree_expected_commit="1" * 40,
        safree_expected_pipeline_sha256=baseline.sha256_file(f["safree"] / "cogvideox/cogvideox_pipeline.py"),
        require_complete=True,
        runtime_probe=_fake_runtime,
    )
    registry_path = f["baseline_registry"] / "baseline_registry.json"
    receipt_path = f["baseline_registry"] / "build_receipt.json"
    registry = controls.load_registry(f["project"], registry_path, receipt_path)
    output = f["project"] / "outputs/formal_controls/water_original"
    plan = controls.build_plan(
        project_root=f["project"],
        registry_path=registry_path,
        registry=registry,
        baseline="cogvideox_original",
        mechanism="water_impact",
        output_dir=output,
    )
    assert len(plan["items"]) == 42
    with f["formal"].open(newline="", encoding="utf-8") as handle:
        expected = [int(row["seed"]) for row in csv.DictReader(handle) if row["mechanism"] == "water_impact"]
    assert [item["seed"] for item in plan["items"]] == expected
    assert any(right - left != 1 for left, right in zip(expected, expected[1:]))

    small_plan = {**plan, "items": plan["items"][:2]}

    def fake_render(_torch, _export, _pipe, item, _generation):
        path = Path(item["video_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video::{item['case_id']}".encode())

    monkeypatch.setattr(controls, "render_item", fake_render)
    result = controls.run_plan(
        project_root=f["project"],
        registry=registry,
        plan=small_plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(controls.MEDIA),
        pipeline_loader=lambda _registry, _root: (object(), object(), object()),
    )
    assert result == {"status": "complete", "completed": 2, "expected": 2}
    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["outputs"]) == 2
    first_video = Path(small_plan["items"][0]["video_path"])
    first_video.write_bytes(b"tampered")
    try:
        controls.run_plan(
            project_root=f["project"],
            registry=registry,
            plan=small_plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(controls.MEDIA),
            pipeline_loader=lambda _registry, _root: (object(), object(), object()),
        )
    except controls.ControlError as exc:
        assert "video changed after receipt" in str(exc)
    else:
        raise AssertionError("tampered formal video was accepted")


def test_t2v_registry_rejects_eligible_checkpoint_selection(tmp_path):
    f = _fixture(tmp_path)
    try:
        t2v_registry.build_training_registry(
            project_root=f["project"],
            training_input_registry=f["training_inputs"],
            model_root=f["model"],
            model_inventory=f["model_inventory"],
            runtime_python=f["runtime_python"],
            output_root=f["t2v_specs"],
            checkpoint_root=f["t2v_checkpoints"],
            train_rows_per_mechanism=36,
            max_steps=100,
            eligible_checkpoint_step=50,
        )
    except t2v_registry.TrainingRegistryError as exc:
        assert "final training step" in str(exc)
    else:
        raise AssertionError("post-hoc checkpoint selection was accepted")
