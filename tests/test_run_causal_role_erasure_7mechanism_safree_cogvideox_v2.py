from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline  # noqa: E402
import run_causal_role_erasure_7mechanism_safree_cogvideox_v2 as safree  # noqa: E402


RUNTIME_PACKAGES = {
    "torch": "2.6.0",
    "diffusers": "0.33.1",
    "transformers": "4.51.3",
}


def _runtime_probe(runtime_python: Path) -> dict[str, object]:
    return {
        "executable": str(runtime_python),
        "packages": dict(RUNTIME_PACKAGES),
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    runner = project / safree.RUNNER_RELATIVE
    runner.parent.mkdir(parents=True)
    shutil.copy2(SCRIPTS / Path(safree.RUNNER_RELATIVE).name, runner)
    builder = project / safree.BUILDER_RELATIVE
    shutil.copy2(SCRIPTS / builder.name, builder)

    formal = project / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
    formal.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", formal)

    model = project / "models/CogVideoX-2b"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text('{"_class_name":"CogVideoXPipeline"}\n', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"frozen-cogvideox-weights")
    inventory = baseline.inventory_tree(model)

    registry_root = project / "outputs/baseline_registry"
    registry_root.mkdir(parents=True)
    inventory_path = registry_root / "model_inventory.json"
    inventory_path.write_bytes(baseline.canonical_json_bytes(inventory))

    safree_root = project / "baselines/external/SAFREE"
    pipeline = safree_root / "cogvideox/cogvideox_pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text(
        "class CogVideoXPipeline:\n    pass\n\nCONCEPT_DICT = {}\n",
        encoding="utf-8",
    )
    commit = "1" * 40
    (safree_root / "SAFREE_COMMIT").write_text(commit + "\n", encoding="utf-8")

    runtime_python = Path(sys.executable).resolve()
    registry = {
        "schema_version": 1,
        "protocol": baseline.REGISTRY_PROTOCOL,
        "protocol_version": baseline.PROTOCOL_VERSION,
        "status": "baseline_stack_frozen_ready",
        "formal_generation_authorized": True,
        "formal_cases": {
            "path": formal.relative_to(project).as_posix(),
            "sha256": baseline.sha256_file(formal),
            "row_count": baseline.FORMAL_CASES,
            "same_case_same_seed_across_streams": True,
        },
        "model": {
            "family": "CogVideoX-2B",
            "root": model.relative_to(project).as_posix(),
            "inventory_path": inventory_path.relative_to(project).as_posix(),
            "inventory_sha256": baseline.sha256_file(inventory_path),
            "file_count": inventory["file_count"],
            "total_size_bytes": inventory["total_size_bytes"],
        },
        "runtime": {
            "python_executable": str(runtime_python),
            "python_executable_sha256": baseline.sha256_file(runtime_python),
            "packages": dict(RUNTIME_PACKAGES),
        },
        "mechanism_concepts": baseline.MECHANISM_CONCEPTS,
        "generation": {
            "shared": baseline.COG_GENERATION,
            "streams": {safree.BASELINE: dict(safree.SAFREE_STREAM)},
        },
        "implementations": {
            safree.BASELINE: {
                "status": "ready",
                "repository": "https://github.com/jaehong31/SAFREE",
                "commit": commit,
                "pipeline_path": pipeline.relative_to(project).as_posix(),
                "pipeline_sha256": baseline.sha256_file(pipeline),
                "concept_injection": safree.SAFREE_CONCEPT_INJECTION,
                "dtype": "fp32",
            }
        },
        "streams": [safree.BASELINE],
        "code_sha256": {
            safree.BUILDER_RELATIVE: baseline.sha256_file(builder),
            safree.RUNNER_RELATIVE: baseline.sha256_file(runner),
        },
    }
    registry_path = registry_root / "baseline_registry.json"
    registry_path.write_bytes(baseline.canonical_json_bytes(registry))
    receipt_path = registry_root / "build_receipt.json"
    receipt_path.write_bytes(
        baseline.canonical_json_bytes(
            {
                "protocol": baseline.REGISTRY_PROTOCOL,
                "status": registry["status"],
                "registry_sha256": baseline.sha256_file(registry_path),
                "model_inventory_sha256": baseline.sha256_file(inventory_path),
            }
        )
    )
    return {
        "project": project,
        "formal": formal,
        "model": model,
        "pipeline": pipeline,
        "builder": builder,
        "registry": registry_path,
        "receipt": receipt_path,
    }


def _load(fixture: dict[str, Path]) -> dict:
    return safree.load_registry(
        fixture["project"],
        fixture["registry"],
        fixture["receipt"],
        runtime_probe=_runtime_probe,
    )


def _rewrite_registry(fixture: dict[str, Path], registry: dict) -> None:
    fixture["registry"].write_bytes(baseline.canonical_json_bytes(registry))
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt["registry_sha256"] = baseline.sha256_file(fixture["registry"])
    fixture["receipt"].write_bytes(baseline.canonical_json_bytes(receipt))


def test_runner_help_works_with_pythonpath_cleared():
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / Path(safree.RUNNER_RELATIVE).name), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline-registry" in result.stdout


def test_plan_consumes_all_frozen_seeds_and_supports_mechanism_shards(tmp_path):
    fixture = _fixture(tmp_path)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/formal_safree/water_impact"
    plan = safree.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )

    with fixture["formal"].open(newline="", encoding="utf-8") as handle:
        expected_rows = [row for row in csv.DictReader(handle) if row["mechanism"] == "water_impact"]
    assert len(plan["items"]) == 42
    assert [item["seed"] for item in plan["items"]] == [int(row["seed"]) for row in expected_rows]
    assert any(
        right["seed"] - left["seed"] != 1
        for left, right in zip(plan["items"], plan["items"][1:])
    )
    assert {item["safree_concept_key"] for item in plan["items"]} == {"Water impact"}
    assert all(item["safree_concept_terms"] == ["Water impact"] for item in plan["items"])
    assert plan["generation"] == {**baseline.COG_GENERATION, **safree.SAFREE_STREAM}
    assert plan["safree"]["pipeline_sha256"] == baseline.sha256_file(fixture["pipeline"])

    all_plan = safree.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism=None,
        output_dir=fixture["project"] / "outputs/formal_safree/all",
    )
    assert len(all_plan["items"]) == 294
    assert [item["global_case_index"] for item in all_plan["items"]] == list(range(294))


def test_run_is_fresh_only_and_strictly_resumes_hash_bound_receipts(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/formal_safree/water_impact"
    full_plan = safree.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    plan = {**full_plan, "items": full_plan["items"][:2]}

    def fake_render(_module, _torch, _export, _pipe, item, _generation):
        video = Path(item["video_path"])
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video::{item['case_id']}::{item['seed']}".encode())

    monkeypatch.setattr(safree, "render_item", fake_render)
    fake_runtime = (object(), object(), object(), object())
    result = safree.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(safree.MEDIA),
        pipeline_loader=lambda _registry, _root: fake_runtime,
    )
    assert result == {"status": "complete", "completed": 2, "expected": 2}
    assert len(list((output / "receipts").glob("*.json"))) == 2
    assert (output / ".complete").read_text(encoding="utf-8") == safree.sha256_file(output / "generation_manifest.json") + "\n"

    with pytest.raises(safree.SAFREEError, match="already exists"):
        safree.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=False,
            dry_run=False,
            media_probe=lambda _python, _video: dict(safree.MEDIA),
            pipeline_loader=lambda _registry, _root: fake_runtime,
        )

    resumed = safree.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=True,
        dry_run=False,
        media_probe=lambda _python, _video: dict(safree.MEDIA),
        pipeline_loader=lambda _registry, _root: (_ for _ in ()).throw(AssertionError("pipeline loaded during complete resume")),
    )
    assert resumed == result

    first_receipt_path = Path(plan["items"][0]["receipt_path"])
    first_receipt = json.loads(first_receipt_path.read_text(encoding="utf-8"))
    first_receipt["seed"] += 1
    first_receipt_path.write_bytes(safree.canonical_json_bytes(first_receipt))
    with pytest.raises(safree.SAFREEError, match="receipt changed"):
        safree.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(safree.MEDIA),
            pipeline_loader=lambda _registry, _root: fake_runtime,
        )


def test_resume_rejects_any_plan_change(tmp_path):
    fixture = _fixture(tmp_path)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/formal_safree/dry"
    plan = safree.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="surface_trace",
        output_dir=output,
    )
    assert safree.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=True,
    )["status"] == "dry_run_plan_frozen"

    changed = {**plan, "generation": {**plan["generation"], "guidance_scale": 7.0}}
    with pytest.raises(safree.SAFREEError, match="resume plan differs"):
        safree.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=changed,
            output_dir=output,
            resume=True,
            dry_run=True,
        )


def test_resume_rejects_unreceipted_existing_video_without_backfill(tmp_path):
    fixture = _fixture(tmp_path)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/formal_safree/water_impact"
    full_plan = safree.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    plan = {**full_plan, "items": full_plan["items"][:1]}
    assert safree.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=True,
    )["status"] == "dry_run_plan_frozen"
    video_path = Path(plan["items"][0]["video_path"])
    receipt_path = Path(plan["items"][0]["receipt_path"])
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"orphaned-video")

    with pytest.raises(safree.SAFREEError, match="refusing receipt backfill"):
        safree.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(safree.MEDIA),
            pipeline_loader=lambda _registry, _root: pytest.fail("orphan resume loaded pipeline"),
        )
    assert not receipt_path.exists()


def test_render_invokes_official_concept_adapter_with_case_seed_and_frozen_media(tmp_path):
    concept_dict: dict[str, list[str]] = {"existing": ["existing term"]}
    module = SimpleNamespace(CONCEPT_DICT=concept_dict)
    calls: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *, device: str):
            calls["generator_device"] = device

        def manual_seed(self, seed: int):
            calls["seed"] = seed
            return self

    class FakeTorch:
        Generator = FakeGenerator

    class FakePipe:
        def __call__(self, **kwargs):
            calls["pipeline"] = kwargs
            assert concept_dict["Water impact"] == ["Water impact"]
            return SimpleNamespace(frames=[["frame-0", "frame-1"]])

    def fake_export(frames, path, *, fps):
        calls["frames"] = frames
        calls["fps"] = fps
        Path(path).write_bytes(b"encoded-video")

    video = tmp_path / "video.mp4"
    item = {
        "prompt": "A frozen formal prompt.",
        "seed": 987654321,
        "safree_concept_key": "Water impact",
        "safree_concept_terms": ["Water impact"],
        "video_path": video.as_posix(),
    }
    safree.render_item(
        module,
        FakeTorch,
        fake_export,
        FakePipe(),
        item,
        {**baseline.COG_GENERATION, **safree.SAFREE_STREAM},
    )

    pipeline_call = calls["pipeline"]
    assert pipeline_call["prompt"] == item["prompt"]
    assert pipeline_call["concept"] == "Water impact"
    assert pipeline_call["num_frames"] == 49
    assert pipeline_call["height"] == 480
    assert pipeline_call["width"] == 720
    assert pipeline_call["num_inference_steps"] == 50
    assert calls["seed"] == item["seed"]
    assert calls["generator_device"] == "cuda"
    assert calls["fps"] == 8
    assert video.read_bytes() == b"encoded-video"
    assert concept_dict == {"existing": ["existing term"]}


def test_registry_and_media_validation_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["pipeline"].write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(safree.SAFREEError, match="pipeline changed"):
        _load(fixture)

    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not-decoded-in-this-unit-test")
    media_with_audio = {**safree.MEDIA, "audio_streams": 1}
    with pytest.raises(safree.SAFREEError, match="media contract mismatch"):
        safree.validate_video(
            Path(sys.executable),
            video,
            media_probe=lambda _python, _video: media_with_audio,
        )


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        (safree.BUILDER_RELATIVE, "registry builder is not bound"),
        (safree.RUNNER_RELATIVE, "SAFREE formal runner is not bound"),
    ],
)
def test_registry_requires_builder_and_runner_code_bindings(tmp_path, relative, message):
    fixture = _fixture(tmp_path)
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["code_sha256"].pop(relative)
    _rewrite_registry(fixture, registry)
    with pytest.raises(safree.SAFREEError, match=message):
        _load(fixture)


def test_registry_rejects_mutated_builder(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["builder"].write_text("# mutated after registry freeze\n", encoding="utf-8")
    with pytest.raises(safree.SAFREEError, match="registry builder changed"):
        _load(fixture)


def test_registry_reprobes_exact_runtime_versions_and_current_python(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    with pytest.raises(safree.SAFREEError, match="package versions changed"):
        safree.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            runtime_probe=lambda runtime_python: {
                "executable": str(runtime_python),
                "packages": {**RUNTIME_PACKAGES, "transformers": "9.9.9"},
            },
        )

    monkeypatch.setattr(safree.sys, "executable", str(tmp_path / "wrong-python"))
    with pytest.raises(safree.SAFREEError, match="wrong runtime Python"):
        _load(fixture)


def test_registered_runtime_probe_uses_isolation_and_builder_version_semantics(monkeypatch):
    runtime_python = Path(sys.executable).resolve()
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": True, "text": True, "capture_output": True}
        return SimpleNamespace(
            stdout=json.dumps(
                {"executable": str(runtime_python), "packages": RUNTIME_PACKAGES}
            )
        )

    monkeypatch.setattr(safree.subprocess, "run", fake_run)
    assert safree.probe_registered_runtime(runtime_python)["packages"] == RUNTIME_PACKAGES
    assert calls[0][:3] == [str(runtime_python), "-I", "-c"]
    assert "metadata.version(name)" in calls[0][3]
