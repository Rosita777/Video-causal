from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_causal_role_erasure_8mechanism_capability_v2 as runner  # noqa: E402


CANONICAL = PROJECT_ROOT / runner.DEFAULT_CANONICAL_MANIFEST
PROMPTS = PROJECT_ROOT / runner.DEFAULT_PROMPTS
GENERATOR = PROJECT_ROOT / "scripts/generate_wan_clean.py"


def test_frozen_v2_inputs_are_exactly_192_fresh_ordered_rows():
    rows, prompt_items = runner.load_frozen_inputs(CANONICAL, PROMPTS)

    assert len(rows) == len(prompt_items) == 192
    assert Counter(row["mechanism"] for row in rows) == {
        mechanism: 24 for mechanism in runner.MECHANISM_ORDER
    }
    assert len({row["case_id"] for row in rows}) == 64
    assert len({row["generation_id"] for row in rows}) == 192
    assert len({row["seed"] for row in rows}) == 192
    assert all(row["case_id"].startswith("cap8v2") for row in rows)
    assert all(row["source_id"].startswith("v2_") for row in rows)
    assert all(row["receiver_id"].startswith("v2_") for row in rows)

    for mechanism_index, mechanism in enumerate(runner.MECHANISM_ORDER):
        shard = rows[24 * mechanism_index : 24 * (mechanism_index + 1)]
        assert {row["mechanism"] for row in shard} == {mechanism}
        assert [int(row["seed"]) for row in shard] == [
            runner.expected_seed(
                mechanism_index,
                combination_index,
                repetition_index,
            )
            for combination_index in range(8)
            for repetition_index in range(3)
        ]
        assert Counter(row["prompt_style"] for row in shard) == {
            "direct": 12,
            "natural": 12,
        }
        assert len({row["source_id"] for row in shard}) == 2
        assert len({row["receiver_id"] for row in shard}) == 2
        assert len({(row["source_id"], row["receiver_id"]) for row in shard}) == 4
        assert prompt_items[24 * mechanism_index : 24 * (mechanism_index + 1)] == [
            {
                "prompt": row["prompt"],
                "target_concept": row["target_concept"],
                "expected_effect": row["expected_footprint"],
            }
            for row in shard
        ]


def test_smoke_decision_and_all_eleven_prompt_files_are_hash_bound():
    amendment_relative = (
        "docs/causal_role_erasure_8mechanism_capability_v2_amendment.md"
    )
    decision_path = (
        PROJECT_ROOT
        / "data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    registered = decision["prompt_file_sha256"]
    assert len(registered) == 11
    assert set(registered).issubset(runner.FROZEN_CODE_AND_DATA_PATHS)
    assert (
        amendment_relative in runner.FROZEN_CODE_AND_DATA_PATHS
    )
    assert runner.EXPECTED_FROZEN_ARTIFACT_SHA256[amendment_relative] == (
        "9bc99bbf22988edc4e156195c266807b91fea2305ffee2dc639441686e9a4b7c"
    )
    assert runner.sha256_file(PROJECT_ROOT / amendment_relative) == (
        runner.EXPECTED_FROZEN_ARTIFACT_SHA256[amendment_relative]
    )
    assert (
        "data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json"
        in runner.FROZEN_CODE_AND_DATA_PATHS
    )
    for relative, expected in registered.items():
        assert runner.EXPECTED_FROZEN_ARTIFACT_SHA256[relative] == expected
        assert runner.sha256_file(PROJECT_ROOT / relative) == expected
    assert runner.EXPECTED_FROZEN_ARTIFACT_SHA256[
        decision_path.relative_to(PROJECT_ROOT).as_posix()
    ] == runner.sha256_file(decision_path)


def test_input_loader_rejects_reordered_manifest_even_with_updated_hash(
    tmp_path: Path,
    monkeypatch,
):
    rows = json.loads(CANONICAL.read_text(encoding="utf-8"))
    rows[0], rows[1] = rows[1], rows[0]
    changed = tmp_path / "reordered.canonical.json"
    changed.write_bytes(runner.canonical_json_bytes(rows))
    monkeypatch.setattr(
        runner,
        "EXPECTED_CANONICAL_MANIFEST_SHA256",
        runner.sha256_file(changed),
    )

    with pytest.raises(ValueError, match="repetition order mismatch"):
        runner.load_frozen_inputs(changed, PROMPTS)


def test_dry_run_writes_four_gpu_two_wave_plan_without_starting_gpu(
    tmp_path: Path,
):
    output_root = tmp_path / "capability-v2-plan"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_causal_role_erasure_8mechanism_capability_v2.py"),
            "--output-root",
            str(output_root),
            "--gpus",
            "4,5,6,7",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    plan = json.loads(
        (output_root / "capability_run_manifest.json").read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (output_root / "capability_run_aggregate.json").read_text(encoding="utf-8")
    )
    assert plan["protocol_version"] == runner.PROTOCOL_VERSION
    assert plan["runner_version"] == runner.RUNNER_VERSION
    assert plan["dry_run"] is True
    assert plan["stage_binding"] is None
    assert plan["inputs"]["canonical_manifest_sha256"] == (
        runner.EXPECTED_CANONICAL_MANIFEST_SHA256
    )
    assert plan["inputs"]["csv_manifest_sha256"] == runner.EXPECTED_CSV_SHA256
    assert plan["inputs"]["summary_sha256"] == runner.EXPECTED_SUMMARY_SHA256
    assert plan["inputs"]["prompts_sha256"] == runner.EXPECTED_PROMPTS_SHA256
    assert plan["inputs"]["seed_formula"] == runner.SEED_FORMULA
    assert plan["generation"] == {
        "baseline": "clean",
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device_inside_isolated_process": "cuda",
        "vae_slicing": True,
        "vae_tiling": True,
        "model_cpu_offload": False,
        "sequential_cpu_offload": False,
        "skip_existing": False,
        "per_prompt_seeds": "explicit_from_canonical_manifest",
        "post_generation_media_probe": (
            "decode-count exact frame/fps/resolution validation"
        ),
    }
    assert plan["scheduler"]["layout"] == "4_gpus_x_2_waves"
    assert [
        (job["wave_index"], job["gpu"], job["mechanism"])
        for job in plan["jobs"]
    ] == [
        (0, 4, "water_impact"),
        (0, 5, "rigid_collision"),
        (0, 6, "brittle_fracture"),
        (0, 7, "powder_impact"),
        (1, 4, "elastic_deformation"),
        (1, 5, "field_mediated_response"),
        (1, 6, "material_release"),
        (1, 7, "surface_trace"),
    ]
    for job in plan["jobs"]:
        assert job["unset_environment"] == ["PYTHONHOME", "PYTHONPATH"]
        assert job["environment"]["PYTHONSAFEPATH"] == "1"
        assert len(job["seeds"]) == 24
        assert job["command"][job["command"].index("--seeds") + 1] == ",".join(
            str(seed) for seed in job["seeds"]
        )
        assert "--skip-existing" not in job["command"]
        assert "--dry-run" not in job["command"]
        assert not Path(job["output_dir"]).exists()
    assert aggregate["status"] == "planned"
    assert aggregate["status_counts"] == {"planned": 8}
    assert not any((output_root / "mechanisms").iterdir())

    repeated = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_causal_role_erasure_8mechanism_capability_v2.py"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "output root must not exist" in repeated.stderr


def test_formal_cli_requires_attestation_and_stage_hash_before_output(
    tmp_path: Path,
):
    output_root = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_causal_role_erasure_8mechanism_capability_v2.py"),
            "--output-root",
            str(output_root),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "attest-sealed-final36-unopened" in result.stderr
    assert not output_root.exists()


def test_sealed_path_is_rejected_before_manifest_read(tmp_path: Path, monkeypatch):
    called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("sealed path reached the manifest loader")

    monkeypatch.setattr(runner, "load_frozen_inputs", forbidden_loader)
    return_code = runner.main(
        [
            "--canonical-manifest",
            "data/sealed-final36-do-not-read.json",
            "--output-root",
            str(tmp_path / "plan"),
            "--dry-run",
        ]
    )
    assert return_code == 1
    assert called is False
    assert not (tmp_path / "plan").exists()


def _copy_stage_files(destination: Path) -> None:
    for relative in runner.FROZEN_CODE_AND_DATA_PATHS:
        source = PROJECT_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_tiny_model_and_runtime(project_root: Path) -> tuple[Path, Path]:
    model_root = project_root / "models/Wan2.1-T2V-1.3B-Diffusers"
    model_root.mkdir(parents=True)
    model_file = model_root / "config.json"
    model_file.write_text('{"tiny":true}\n', encoding="utf-8")
    model_files = [
        {
            "path": model_file.relative_to(project_root).as_posix(),
            "sha256": runner.sha256_file(model_file),
            "size_bytes": model_file.stat().st_size,
        }
    ]
    model_payload = {
        "protocol": runner._v1.MODEL_INVENTORY_PROTOCOL,
        "status": "frozen",
        "dataset_version": runner._v1.FROZEN_INVENTORY_DATASET_VERSION,
        "model_root": "models/Wan2.1-T2V-1.3B-Diffusers",
        "file_count": 1,
        "files": model_files,
        "inventory_sha256": hashlib.sha256(
            runner.canonical_json_bytes(model_files)
        ).hexdigest(),
    }
    model_inventory = project_root / runner.DEFAULT_MODEL_INVENTORY
    model_inventory.parent.mkdir(parents=True, exist_ok=True)
    model_inventory.write_text(json.dumps(model_payload) + "\n", encoding="utf-8")

    runtime_python = project_root / "models/.wan-runtime/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runtime_python.chmod(0o755)
    live = runner.runtime_content_inventory(project_root / "models/.wan-runtime")
    runtime_payload = {
        "protocol": runner._v1.RUNTIME_REGISTRY_PROTOCOL,
        "status": "frozen",
        "dataset_version": runner._v1.FROZEN_INVENTORY_DATASET_VERSION,
        "runtime_root": "models/.wan-runtime",
        "python_executable": "models/.wan-runtime/bin/python",
        "sys_prefix_policy": "test",
        "python": {},
        "torch": {},
        "cuda": {},
        "packages": {},
        **live,
        "module_origins": {},
    }
    runtime_registry = project_root / runner.DEFAULT_RUNTIME_REGISTRY
    runtime_registry.write_text(json.dumps(runtime_payload) + "\n", encoding="utf-8")
    return model_inventory, runtime_registry


def _git(project_root: Path, *arguments: str) -> None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_stage_binds_exact_public_evidence_and_v1_stage_registry(
    tmp_path: Path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    _copy_stage_files(project)
    (project / ".gitignore").write_text("models/\n", encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Capability V2 Test")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "frozen v2 stage inputs")

    model_inventory, runtime_registry = _write_tiny_model_and_runtime(project)
    for relative in runner.PUBLIC_EVIDENCE_PATHS:
        evidence = project / relative
        if not evidence.exists():
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text("{}\n", encoding="utf-8")
    predecessor = project / runner.DEFAULT_V1_STAGE_REGISTRY
    predecessor.write_text('{"v1_stage":"frozen"}\n', encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "EXPECTED_V1_STAGE_REGISTRY_SHA256",
        runner.sha256_file(predecessor),
    )

    rows, _ = runner.load_frozen_inputs(
        project / runner.DEFAULT_CANONICAL_MANIFEST,
        project / runner.DEFAULT_PROMPTS,
    )
    stage_registry = project / runner.DEFAULT_STAGE_REGISTRY
    payload = runner.prepare_stage_registry(
        project_root=project,
        stage_registry=stage_registry,
        model_inventory=model_inventory,
        runtime_registry=runtime_registry,
        rows=rows,
    )
    digest = runner.sha256_file(stage_registry)
    assert payload["status"] == "authorized_for_original_capability_v2_generation"
    assert payload["sealed_final36_status"] == "unopened"
    assert payload["predecessor_v1_stage_registry"]["sha256"] == (
        runner.sha256_file(predecessor)
    )
    assert len(payload["public_evidence"]) == 8
    assert payload["seed_registry"]["count"] == 192
    assert payload["generation"] == runner.generation_contract()
    assert payload["scheduler"]["gpu_count"] == 4
    assert payload["scheduler"]["waves"] == 2
    assert payload["media_validation"]["final_rehash_all_192"] is True

    reopened = runner.reopen_and_validate_stage_registry(
        project_root=project,
        stage_registry=stage_registry,
        expected_sha256=digest,
        rows=rows,
    )
    assert reopened == payload
    runner.quick_revalidate_stage_registry(
        project_root=project,
        stage_registry=stage_registry,
        expected_sha256=digest,
    )

    extra = project / "unexpected.txt"
    extra.write_text("not allowlisted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked set"):
        runner.reopen_and_validate_stage_registry(
            project_root=project,
            stage_registry=stage_registry,
            expected_sha256=digest,
            rows=rows,
        )
    extra.unlink()

    original_predecessor = predecessor.read_bytes()
    predecessor.write_bytes(original_predecessor + b"drift\n")
    with pytest.raises(ValueError, match="predecessor v1 stage-registry SHA-256"):
        runner.quick_revalidate_stage_registry(
            project_root=project,
            stage_registry=stage_registry,
            expected_sha256=digest,
        )
    predecessor.write_bytes(original_predecessor)


def _make_execution_plan(tmp_path: Path):
    rows, prompt_items = runner.load_frozen_inputs(CANONICAL, PROMPTS)
    runtime_python = tmp_path / "runtime/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runtime_python.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    stage_registry = tmp_path / "stage.json"
    stage_registry.write_text('{"frozen":true}\n', encoding="utf-8")
    output_root = tmp_path / "run"
    plan = runner.build_plan(
        rows=rows,
        prompt_items=prompt_items,
        project_root=PROJECT_ROOT,
        canonical_manifest=CANONICAL,
        prompts=PROMPTS,
        output_root=output_root,
        generator=GENERATOR,
        python_executable=runtime_python,
        model=model,
        gpus=[0, 1, 2, 3],
        dry_run=False,
        stage_registry=stage_registry,
        stage_registry_sha256=runner.sha256_file(stage_registry),
    )
    runner.prepare_output_root(output_root, plan, prompt_items)
    return plan, output_root


def _fake_generation(job_command: list[str], return_code: int) -> None:
    if return_code:
        return
    output_dir = Path(job_command[job_command.index("--output-dir") + 1])
    prompts_path = Path(job_command[job_command.index("--prompts") + 1])
    model = job_command[job_command.index("--model") + 1]
    seeds = [
        int(value)
        for value in job_command[job_command.index("--seeds") + 1].split(",")
    ]
    prompt_items = runner.parse_prompt_file_strict(prompts_path)
    videos = output_dir / "videos"
    videos.mkdir(parents=True)
    items = []
    for index, (prompt_item, seed) in enumerate(zip(prompt_items, seeds)):
        video = videos / f"{index:03d}_fake_seed{seed}.mp4"
        video.write_bytes(b"fake-mp4")
        items.append(
            {
                "index": index,
                **prompt_item,
                "seed": seed,
                "video_path": str(video),
            }
        )
    generation = {
        **runner._expected_generation_fields(),
        "seed": seeds[0],
        "seeds": seeds,
    }
    payload = {
        "baseline": "clean",
        "pipeline": "WanPipeline",
        "model": model,
        "dry_run": False,
        "prompts": str(prompts_path),
        "generation": generation,
        "items": items,
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


class _ImmediateProcess:
    def __init__(self, return_code: int):
        self.return_code = return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = -15

    def kill(self):
        self.return_code = -9


def _media_probe(_runtime_python: Path, _video: Path):
    return {
        "decoded_frames": 49,
        "fps": "8/1",
        "height": 480,
        "width": 832,
    }


def test_execution_isolated_two_wave_and_freezes_exact_192_sha_bound_videos(
    tmp_path: Path,
):
    plan, output_root = _make_execution_plan(tmp_path)
    launched = []
    stage_revalidations = []
    process_environments = []
    media_probe_calls = []

    def popen(command, **kwargs):
        mechanism = Path(command[command.index("--output-dir") + 1]).name.split(
            "_", 1
        )[1]
        launched.append(mechanism)
        process_environments.append(kwargs["env"])
        _fake_generation(command, 0)
        return _ImmediateProcess(0)

    def media_probe(runtime_python: Path, video: Path):
        media_probe_calls.append((runtime_python, video))
        return _media_probe(runtime_python, video)

    runner.execute_plan(
        plan,
        output_root,
        poll_interval=0.001,
        popen_factory=popen,
        sleep_fn=lambda _seconds: None,
        media_probe=media_probe,
        stage_revalidator=lambda: stage_revalidations.append("checked"),
    )
    assert launched == list(runner.MECHANISM_ORDER)
    assert stage_revalidations == ["checked", "checked", "checked"]
    assert all(
        "PYTHONHOME" not in env and "PYTHONPATH" not in env
        for env in process_environments
    )
    assert all(env["PYTHONSAFEPATH"] == "1" for env in process_environments)
    assert len(media_probe_calls) == 384

    aggregate = json.loads(
        (output_root / "capability_run_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "completed"
    assert aggregate["validated_videos"] == 192
    assert aggregate["status_counts"] == {"completed": 8}
    generation_manifest_path = output_root / "capability_generation_manifest.json"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    assert generation_manifest["protocol_version"] == runner.PROTOCOL_VERSION
    assert generation_manifest["video_binding_key"] == "generation_id"
    assert generation_manifest["video_count"] == 192
    assert [
        item["canonical_row_index"] for item in generation_manifest["items"]
    ] == list(range(192))
    assert all(
        len(item["video_sha256"]) == 64 for item in generation_manifest["items"]
    )
    assert all(item["media"] == _media_probe(Path(), Path()) for item in generation_manifest["items"])
    assert aggregate["frozen_generation_manifest"]["sha256"] == (
        runner.sha256_file(generation_manifest_path)
    )


def test_final_freeze_rehashes_all_192_videos_and_rejects_drift(tmp_path: Path):
    plan, output_root = _make_execution_plan(tmp_path)
    revalidation_count = 0

    def popen(command, **_kwargs):
        _fake_generation(command, 0)
        return _ImmediateProcess(0)

    def revalidate():
        nonlocal revalidation_count
        revalidation_count += 1
        if revalidation_count == 3:
            first_video = sorted(
                Path(plan["jobs"][0]["output_dir"]).joinpath("videos").glob("*.mp4")
            )[0]
            first_video.write_bytes(first_video.read_bytes() + b"post-wave-drift")

    with pytest.raises(ValueError, match="drifted after initial validation"):
        runner.execute_plan(
            plan,
            output_root,
            poll_interval=0.001,
            popen_factory=popen,
            sleep_fn=lambda _seconds: None,
            media_probe=_media_probe,
            stage_revalidator=revalidate,
        )
    aggregate = json.loads(
        (output_root / "capability_run_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "failed"
    assert "could not freeze generation manifest" in aggregate["error"]
    assert not (output_root / "capability_generation_manifest.json").exists()


def test_wave_zero_failure_blocks_wave_one(tmp_path: Path):
    plan, output_root = _make_execution_plan(tmp_path)
    launched = []

    def popen(command, **_kwargs):
        mechanism = Path(command[command.index("--output-dir") + 1]).name.split(
            "_", 1
        )[1]
        launched.append(mechanism)
        return_code = 17 if mechanism == "rigid_collision" else 0
        _fake_generation(command, return_code)
        return _ImmediateProcess(return_code)

    with pytest.raises(RuntimeError, match="rigid_collision"):
        runner.execute_plan(
            plan,
            output_root,
            poll_interval=0.001,
            popen_factory=popen,
            sleep_fn=lambda _seconds: None,
            media_probe=_media_probe,
            stage_revalidator=lambda: None,
        )
    assert launched == list(runner.MECHANISM_ORDER[:4])
    aggregate = json.loads(
        (output_root / "capability_run_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["status"] == "failed"
    assert aggregate["status_counts"] == {
        "completed": 3,
        "failed": 1,
        "planned": 4,
    }


def test_pyav_probe_requires_exact_decode_fps_and_resolution(
    monkeypatch,
    tmp_path: Path,
):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    runtime_python = tmp_path / "python"
    runtime_python.write_text("fake", encoding="utf-8")

    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "streams": 1,
                "width": 832,
                "height": 480,
                "fps": "8/1",
                "decoded_frames": 49,
            }
        )

    captured_environments = []

    def fake_run(*_args, **kwargs):
        captured_environments.append(kwargs["env"])
        return Result()

    monkeypatch.setattr(runner._v1.subprocess, "run", fake_run)
    assert runner.probe_video_media(runtime_python, video) == {
        "decoded_frames": 49,
        "fps": "8/1",
        "height": 480,
        "width": 832,
    }
    assert "PYTHONHOME" not in captured_environments[0]
    assert "PYTHONPATH" not in captured_environments[0]
    assert captured_environments[0]["PYTHONSAFEPATH"] == "1"

    Result.stdout = json.dumps(
        {
            "streams": 1,
            "width": 832,
            "height": 480,
            "fps": "8/1",
            "decoded_frames": 48,
        }
    )
    with pytest.raises(ValueError, match="expected 49 decoded frames"):
        runner.probe_video_media(runtime_python, video)
