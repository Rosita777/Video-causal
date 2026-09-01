from __future__ import annotations

import csv
import hashlib
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

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_registry  # noqa: E402
import run_causal_role_erasure_7mechanism_videoeraser_official_v2 as videoeraser  # noqa: E402


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


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / Path(videoeraser.RUNNER_RELATIVE).name
    shutil.copy2(SCRIPTS / runner.name, runner)
    builder = project / videoeraser.BUILDER_RELATIVE
    shutil.copy2(SCRIPTS / builder.name, builder)

    formal = project / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
    formal.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", formal)

    model = project / "models/CogVideoX-2b"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text(
        '{"_class_name":"CogVideoXPipeline"}\n',
        encoding="utf-8",
    )
    (model / "weights.bin").write_bytes(b"frozen-cogvideox-weights")
    inventory = baseline_registry.inventory_tree(model)

    registry_root = project / "outputs/baseline_registry_v2"
    snapshot = registry_root / "external_snapshot/VideoEraser"
    pipeline = snapshot / "CogVideoX/cogvideox_pipeline.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("class CogVideoXPipeline: pass\n", encoding="utf-8")
    (snapshot / "OFFICIAL_COMMIT").write_text(
        baseline_registry.EXPECTED_VIDEOERASER_COMMIT + "\n",
        encoding="utf-8",
    )
    pipeline_digest = videoeraser.sha256_file(pipeline)
    monkeypatch.setattr(
        baseline_registry,
        "EXPECTED_VIDEOERASER_PIPELINE_SHA256",
        pipeline_digest,
    )

    inventory_path = registry_root / "model_inventory.json"
    inventory_path.write_bytes(videoeraser.canonical_json_bytes(inventory))
    runtime_python = Path(sys.executable).resolve()
    registry = {
        "schema_version": 1,
        "protocol": baseline_registry.REGISTRY_PROTOCOL,
        "protocol_version": baseline_registry.PROTOCOL_VERSION,
        "status": "baseline_stack_frozen_ready",
        "formal_generation_authorized": True,
        "formal_cases": {
            "path": formal.relative_to(project).as_posix(),
            "sha256": videoeraser.sha256_file(formal),
            "row_count": 294,
            "same_case_same_seed_across_streams": True,
        },
        "model": {
            "family": "CogVideoX-2B",
            "root": model.relative_to(project).as_posix(),
            "inventory_path": inventory_path.relative_to(project).as_posix(),
            "inventory_sha256": videoeraser.sha256_file(inventory_path),
            "file_count": inventory["file_count"],
            "total_size_bytes": inventory["total_size_bytes"],
        },
        "runtime": {
            "python_executable": str(runtime_python),
            "python_executable_sha256": videoeraser.sha256_file(runtime_python),
            "packages": dict(RUNTIME_PACKAGES),
        },
        "mechanism_concepts": dict(baseline_registry.MECHANISM_CONCEPTS),
        "generation": {
            "shared": dict(baseline_registry.COG_GENERATION),
            "streams": {videoeraser.BASELINE: dict(videoeraser.EXPECTED_STREAM)},
        },
        "implementations": {
            videoeraser.BASELINE: {
                "status": "ready",
                "label": "VideoEraser (official CogVideoX)",
                "repository": "https://github.com/bluedream02/VideoEraser",
                "commit": baseline_registry.EXPECTED_VIDEOERASER_COMMIT,
                "pipeline_path": pipeline.relative_to(project).as_posix(),
                "pipeline_sha256": pipeline_digest,
                "official_pipeline_unmodified": True,
                "dtype": "bf16",
            }
        },
        "streams": [videoeraser.BASELINE],
        "code_sha256": {
            videoeraser.BUILDER_RELATIVE: videoeraser.sha256_file(builder),
            videoeraser.RUNNER_RELATIVE: videoeraser.sha256_file(runner),
        },
    }
    registry_path = registry_root / "baseline_registry.json"
    registry_path.write_bytes(videoeraser.canonical_json_bytes(registry))
    receipt_path = registry_root / "build_receipt.json"
    receipt_path.write_bytes(
        videoeraser.canonical_json_bytes(
            {
                "protocol": baseline_registry.REGISTRY_PROTOCOL,
                "status": "baseline_stack_frozen_ready",
                "registry_sha256": videoeraser.sha256_file(registry_path),
                "model_inventory_sha256": videoeraser.sha256_file(inventory_path),
            }
        )
    )
    return {
        "project": project,
        "formal": formal,
        "pipeline": pipeline,
        "builder": builder,
        "registry": registry_path,
        "receipt": receipt_path,
    }


def _load(fixture: dict[str, Path]) -> dict:
    return videoeraser.load_registry(
        fixture["project"],
        fixture["registry"],
        fixture["receipt"],
        runtime_probe=_runtime_probe,
    )


def _rewrite_registry(fixture: dict[str, Path], registry: dict) -> None:
    fixture["registry"].write_bytes(videoeraser.canonical_json_bytes(registry))
    receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    receipt["registry_sha256"] = videoeraser.sha256_file(fixture["registry"])
    fixture["receipt"].write_bytes(videoeraser.canonical_json_bytes(receipt))


def test_runner_help_works_with_pythonpath_cleared():
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = ""
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPTS / Path(videoeraser.RUNNER_RELATIVE).name), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline-registry" in result.stdout


def test_plan_uses_all_frozen_cases_and_supports_mechanism_shards(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = _load(fixture)
    all_output = fixture["project"] / "outputs/videoeraser/all"
    all_plan = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism=None,
        output_dir=all_output,
    )
    assert len(all_plan["items"]) == 294
    assert all_plan["baseline"] == "videoeraser_official"
    assert all_plan["implementation"]["official_pipeline_unmodified"] is True
    assert all_plan["generation"]["num_frames"] == 49
    assert all_plan["generation"]["fps"] == 8
    assert all_plan["generation"]["height"] == 480
    assert all_plan["generation"]["width"] == 720

    shard_output = fixture["project"] / "outputs/videoeraser/water"
    shard = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=shard_output,
    )
    assert len(shard["items"]) == 42
    with fixture["formal"].open(newline="", encoding="utf-8") as handle:
        expected = [
            row
            for row in csv.DictReader(handle)
            if row["mechanism"] == "water_impact"
        ]
    assert [item["case_id"] for item in shard["items"]] == [row["case_id"] for row in expected]
    assert [item["seed"] for item in shard["items"]] == [int(row["seed"]) for row in expected]
    assert any(
        right - left != 1
        for left, right in zip(
            [item["seed"] for item in shard["items"]],
            [item["seed"] for item in shard["items"]][1:],
        )
    )
    assert {item["target_concept"] for item in shard["items"]} == {"Water impact"}


def test_run_writes_strict_receipts_and_revalidates_resume(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/videoeraser/water"
    plan = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    small_plan = {**plan, "items": plan["items"][:2]}

    def fake_render(_torch, _export, _pipe, item, _generation):
        path = Path(item["video_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video::{item['case_id']}::{item['seed']}".encode())

    monkeypatch.setattr(videoeraser, "render_item", fake_render)
    result = videoeraser.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=small_plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
        pipeline_loader=lambda _registry, _root: (object(), object(), object()),
    )
    assert result == {"status": "complete", "completed": 2, "expected": 2}
    receipts = sorted((output / "receipts").glob("*.json"))
    assert len(receipts) == 2
    first_receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert first_receipt["official_commit"] == baseline_registry.EXPECTED_VIDEOERASER_COMMIT
    assert first_receipt["official_pipeline_sha256"] == videoeraser.sha256_file(fixture["pipeline"])
    assert first_receipt["media"] == videoeraser.MEDIA
    assert first_receipt["generation_plan_sha256"] == hashlib.sha256(
        videoeraser.canonical_json_bytes(small_plan)
    ).hexdigest()

    resumed = videoeraser.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=small_plan,
        output_dir=output,
        resume=True,
        dry_run=False,
        media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
        pipeline_loader=lambda _registry, _root: pytest.fail("completed resume loaded the GPU pipeline"),
    )
    assert resumed == result

    first_video = Path(small_plan["items"][0]["video_path"])
    first_video.write_bytes(b"tampered")
    with pytest.raises(videoeraser.VideoEraserError, match="formal receipt mismatch"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=small_plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
            pipeline_loader=lambda _registry, _root: pytest.fail("tampered resume loaded pipeline"),
        )


def test_resume_rejects_a_different_plan_and_bad_media(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/videoeraser/water"
    plan = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    small_plan = {**plan, "items": plan["items"][:1]}
    assert videoeraser.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=small_plan,
        output_dir=output,
        resume=False,
        dry_run=True,
    ) == {"status": "dry_run_plan_frozen", "completed": 0, "expected": 1}

    with pytest.raises(videoeraser.VideoEraserError, match="use --resume"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=small_plan,
            output_dir=output,
            resume=False,
            dry_run=True,
        )

    changed_plan = {
        **small_plan,
        "generation": {**small_plan["generation"], "guidance_scale": 5.5},
    }
    with pytest.raises(videoeraser.VideoEraserError, match="resume plan differs"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=changed_plan,
            output_dir=output,
            resume=True,
            dry_run=True,
        )

    def fake_render(_torch, _export, _pipe, item, _generation):
        path = Path(item["video_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bad-media")

    monkeypatch.setattr(videoeraser, "render_item", fake_render)
    with pytest.raises(videoeraser.VideoEraserError, match="media contract mismatch"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=small_plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: {
                **videoeraser.MEDIA,
                "decoded_frames": 48,
            },
            pipeline_loader=lambda _registry, _root: (object(), object(), object()),
        )


def test_resume_rejects_unreceipted_existing_video_without_backfill(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/videoeraser/water"
    full_plan = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    plan = {**full_plan, "items": full_plan["items"][:1]}
    assert videoeraser.run_plan(
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

    with pytest.raises(videoeraser.VideoEraserError, match="refusing receipt backfill"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
            pipeline_loader=lambda _registry, _root: pytest.fail("orphan resume loaded pipeline"),
        )
    assert not receipt_path.exists()


def test_completed_resume_validates_manifest_marker_pair_without_overwriting(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = _load(fixture)
    output = fixture["project"] / "outputs/videoeraser/water"
    full_plan = videoeraser.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        registry=registry,
        mechanism="water_impact",
        output_dir=output,
    )
    plan = {**full_plan, "items": full_plan["items"][:1]}

    def fake_render(_torch, _export, _pipe, item, _generation):
        video = Path(item["video_path"])
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"validated-video")

    monkeypatch.setattr(videoeraser, "render_item", fake_render)
    videoeraser.run_plan(
        project_root=fixture["project"],
        registry=registry,
        plan=plan,
        output_dir=output,
        resume=False,
        dry_run=False,
        media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
        pipeline_loader=lambda _registry, _root: (object(), object(), object()),
    )
    manifest_path = output / "generation_manifest.json"
    complete_path = output / ".complete"

    bad_marker = "0" * 64 + "\n"
    complete_path.write_text(bad_marker, encoding="utf-8")
    with pytest.raises(videoeraser.VideoEraserError, match="does not bind the manifest"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
            pipeline_loader=lambda _registry, _root: pytest.fail("bad completion marker loaded pipeline"),
        )
    assert complete_path.read_text(encoding="utf-8") == bad_marker

    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["outputs"][0]["seed"] += 1
    tampered_bytes = videoeraser.canonical_json_bytes(tampered_manifest)
    manifest_path.write_bytes(tampered_bytes)
    paired_marker = videoeraser.sha256_file(manifest_path) + "\n"
    complete_path.write_text(paired_marker, encoding="utf-8")
    with pytest.raises(videoeraser.VideoEraserError, match="manifest differs from validated receipts"):
        videoeraser.run_plan(
            project_root=fixture["project"],
            registry=registry,
            plan=plan,
            output_dir=output,
            resume=True,
            dry_run=False,
            media_probe=lambda _python, _video: dict(videoeraser.MEDIA),
            pipeline_loader=lambda _registry, _root: pytest.fail("tampered completed resume loaded pipeline"),
        )
    assert manifest_path.read_bytes() == tampered_bytes
    assert complete_path.read_text(encoding="utf-8") == paired_marker


def test_render_calls_official_concept_pipeline_with_frozen_seed(tmp_path):
    calls: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *, device):
            calls["generator_device"] = device

        def manual_seed(self, seed):
            calls["seed"] = seed
            return self

    class FakeTorch:
        Generator = FakeGenerator

    class FakeResult:
        frames = [["frame-0", "frame-1"]]

    class FakePipe:
        def __call__(self, **kwargs):
            calls["pipeline"] = kwargs
            return FakeResult()

    def fake_export(frames, path, *, fps):
        calls["export"] = {"frames": frames, "path": path, "fps": fps}
        Path(path).write_bytes(b"video")

    video_path = tmp_path / "formal.mp4"
    item = {
        "prompt": "A stone strikes a pane and cracks spread.",
        "target_concept": "Brittle fracture",
        "seed": 991337,
        "video_path": str(video_path),
    }
    generation = {**baseline_registry.COG_GENERATION, **videoeraser.EXPECTED_STREAM}
    videoeraser.render_item(FakeTorch, fake_export, FakePipe(), item, generation)
    pipeline_call = calls["pipeline"]
    assert pipeline_call["prompt"] == item["prompt"]
    assert pipeline_call["concept"] == "Brittle fracture"
    assert pipeline_call["generator"] is not None
    assert calls["generator_device"] == "cuda"
    assert calls["seed"] == 991337
    assert pipeline_call["num_inference_steps"] == 50
    assert pipeline_call["num_frames"] == 49
    assert pipeline_call["height"] == 480
    assert pipeline_call["width"] == 720
    assert calls["export"]["fps"] == 8


def test_registry_loader_rejects_mutated_official_snapshot(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["pipeline"].write_text("# mutated after registry freeze\n", encoding="utf-8")
    with pytest.raises(videoeraser.VideoEraserError, match="snapshot changed"):
        _load(fixture)


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        (videoeraser.BUILDER_RELATIVE, "registry builder is not bound"),
        (videoeraser.RUNNER_RELATIVE, "VideoEraser runner is not bound"),
    ],
)
def test_registry_requires_builder_and_runner_code_bindings(tmp_path, monkeypatch, relative, message):
    fixture = _fixture(tmp_path, monkeypatch)
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["code_sha256"].pop(relative)
    _rewrite_registry(fixture, registry)
    with pytest.raises(videoeraser.VideoEraserError, match=message):
        _load(fixture)


def test_registry_rejects_mutated_builder(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["builder"].write_text("# mutated after registry freeze\n", encoding="utf-8")
    with pytest.raises(videoeraser.VideoEraserError, match="registry builder changed"):
        _load(fixture)


def test_registry_reprobes_exact_runtime_versions_and_current_python(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    with pytest.raises(videoeraser.VideoEraserError, match="package versions changed"):
        videoeraser.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            runtime_probe=lambda runtime_python: {
                "executable": str(runtime_python),
                "packages": {**RUNTIME_PACKAGES, "torch": "9.9.9"},
            },
        )

    monkeypatch.setattr(videoeraser.sys, "executable", str(tmp_path / "wrong-python"))
    with pytest.raises(videoeraser.VideoEraserError, match="wrong runtime Python"):
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

    monkeypatch.setattr(videoeraser.subprocess, "run", fake_run)
    assert videoeraser.probe_registered_runtime(runtime_python)["packages"] == RUNTIME_PACKAGES
    assert calls[0][:3] == [str(runtime_python), "-I", "-c"]
    assert "metadata.version(name)" in calls[0][3]
