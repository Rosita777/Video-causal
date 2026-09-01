from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline  # noqa: E402
import run_causal_role_erasure_7mechanism_baseline_queue_v2 as queue  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(queue.canonical_json_bytes(payload))


def _fixture(tmp_path: Path) -> dict[str, Any]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPTS / Path(queue.RUNNER_RELATIVE).name, project / queue.RUNNER_RELATIVE)
    shutil.copy2(SCRIPTS / Path(queue.BUILDER_RELATIVE).name, project / queue.BUILDER_RELATIVE)
    for relative in set(queue.RUNNER_BY_STREAM.values()):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen fixture runner: {relative}\n", encoding="utf-8")

    formal = project / "data/formal_cases.csv"
    formal.parent.mkdir(parents=True)
    formal_rows = ["case_id,mechanism,seed"]
    for mechanism_index, mechanism in enumerate(queue.MECHANISMS):
        for case_index in range(queue.CASES_PER_JOB):
            formal_rows.append(
                f"{mechanism}_case_{case_index:02d},{mechanism},{500_000 + mechanism_index * queue.CASES_PER_JOB + case_index}"
            )
    formal.write_text("\n".join(formal_rows) + "\n", encoding="utf-8")
    runtime = Path(sys.executable).resolve()
    runtime_contract = {
        "python": "fixture-python",
        "implementation": "CPython",
        "executable": str(runtime),
        "packages": {
            "torch": "fixture-torch",
            "diffusers": "fixture-diffusers",
            "transformers": "fixture-transformers",
        },
    }
    registry = {
        "schema_version": 1,
        "protocol": baseline.REGISTRY_PROTOCOL,
        "protocol_version": baseline.PROTOCOL_VERSION,
        "status": "baseline_stack_frozen_ready",
        "formal_generation_authorized": True,
        "streams": list(queue.STREAMS),
        "formal_cases": {
            "path": formal.relative_to(project).as_posix(),
            "sha256": queue.sha256_file(formal),
            "row_count": 294,
        },
        "runtime": {
            **runtime_contract,
            "python_executable": str(runtime),
            "python_executable_sha256": queue.sha256_file(runtime),
        },
        "implementations": {
            stream: {"status": "ready"} for stream in queue.STREAMS
        },
        "code_sha256": {
            relative: queue.sha256_file(project / relative)
            for relative in set(queue.RUNNER_BY_STREAM.values())
            | {queue.RUNNER_RELATIVE, queue.BUILDER_RELATIVE}
        },
    }
    registry_root = project / "outputs/baseline_registry"
    registry_path = registry_root / "baseline_registry.json"
    receipt_path = registry_root / "build_receipt.json"
    _write_json(registry_path, registry)
    _write_json(
        receipt_path,
        {
            "protocol": baseline.REGISTRY_PROTOCOL,
            "status": "baseline_stack_frozen_ready",
            "registry_sha256": queue.sha256_file(registry_path),
        },
    )
    return {
        "project": project,
        "registry": registry_path,
        "receipt": receipt_path,
        "runtime_contract": runtime_contract,
    }


def _build(
    fixture: dict[str, Any],
    output: Path,
    streams=queue.STREAMS,
    gpus=(0, 1, 2, 3),
    safree_limit=4,
):
    registry = queue.load_registry(
        fixture["project"],
        fixture["registry"],
        fixture["receipt"],
        streams,
        runtime_probe=lambda _python: fixture["runtime_contract"],
    )
    return queue.build_plan(
        project_root=fixture["project"],
        registry_path=fixture["registry"],
        receipt_path=fixture["receipt"],
        registry=registry,
        output_root=output,
        selected_streams=streams,
        gpus=gpus,
        safree_max_concurrency=safree_limit,
    )


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class _FinishedProcess:
    def __init__(self, return_code: int = 0):
        self.return_code = return_code

    def poll(self):
        return self.return_code

    def terminate(self):
        return None

    def kill(self):
        return None


class _FakeFormalRunner:
    def __init__(self, registry_sha256: str):
        self.registry_sha256 = registry_sha256
        self.calls = 0

    def __call__(self, command, **_kwargs):
        self.calls += 1
        command = list(command)
        output = Path(_command_value(command, "--output-dir"))
        mechanism = _command_value(command, "--mechanism")
        if "--baseline" in command:
            stream = _command_value(command, "--baseline")
        else:
            name = Path(command[1]).name
            stream = next(
                candidate
                for candidate, relative in queue.RUNNER_BY_STREAM.items()
                if Path(relative).name == name and candidate not in queue.CONTROL_STREAMS
            )
        output.mkdir(parents=True)
        videos = output / "videos"
        receipts = output / "receipts"
        videos.mkdir()
        receipts.mkdir()
        items = []
        output_receipts = []
        for index in range(queue.CASES_PER_JOB):
            case_id = f"{mechanism}_case_{index:02d}"
            seed = (
                500_000
                + queue.MECHANISMS.index(mechanism) * queue.CASES_PER_JOB
                + index
            )
            video = videos / f"{case_id}.mp4"
            receipt_path = receipts / f"{case_id}.json"
            video.write_bytes(f"video::{stream}::{case_id}".encode("utf-8"))
            item = {
                "case_id": case_id,
                "mechanism": mechanism,
                "seed": seed,
                "video_path": str(video),
                "receipt_path": str(receipt_path),
            }
            receipt = {
                "case_id": case_id,
                "baseline": stream,
                "seed": seed,
                "video_path": str(video),
                "video_sha256": queue.sha256_file(video),
                "media": dict(queue.MEDIA),
            }
            _write_json(receipt_path, receipt)
            items.append(item)
            output_receipts.append(receipt)
        child_plan = {
            "schema_version": 1,
            "baseline": stream,
            "mechanism": mechanism,
            "baseline_registry_sha256": self.registry_sha256,
            "items": items,
        }
        child_manifest = {**child_plan, "status": "complete", "outputs": output_receipts}
        _write_json(output / "generation_plan.json", child_plan)
        _write_json(output / "generation_manifest.json", child_manifest)
        (output / ".complete").write_text(
            queue.sha256_file(output / "generation_manifest.json") + "\n",
            encoding="utf-8",
        )
        return _FinishedProcess()


def test_full_35_job_plan_uses_four_gpu_waves_and_safree_limit(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_baselines"
    plan = _build(fixture, output, safree_limit=1)
    assert plan["job_count"] == 35
    assert plan["expected_videos"] == 1470
    assert Counter(job["stream"] for job in plan["jobs"]) == {
        stream: 7 for stream in queue.STREAMS
    }
    assert sum(plan["scheduler"]["jobs_per_wave"]) == 35
    for wave in range(plan["scheduler"]["wave_count"]):
        jobs = [job for job in plan["jobs"] if job["wave_index"] == wave]
        assert len(jobs) <= 4
        assert len({job["gpu"] for job in jobs}) == len(jobs)
        assert sum(job["stream"] == "safree_cogvideox" for job in jobs) <= 1
    control = next(job for job in plan["jobs"] if job["stream"] == "negative_prompt")
    assert _command_value(control["command"], "--baseline") == "negative_prompt"
    t2v = next(job for job in plan["jobs"] if job["stream"] == "t2vunlearning_adapted")
    assert Path(t2v["command"][1]).name == Path(queue.RUNNER_BY_STREAM["t2vunlearning_adapted"]).name


def test_selected_stream_execution_and_exact_resume_are_cpu_testable(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_baselines"
    streams = ("cogvideox_original", "videoeraser_official")
    plan = _build(fixture, output, streams=streams, safree_limit=4)
    queue.prepare_output_root(output, plan)
    fake = _FakeFormalRunner(plan["inputs"]["baseline_registry_sha256"])
    queue.execute_plan(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=fake,
        sleep_fn=lambda _seconds: None,
        media_probe=lambda _python, _video: dict(queue.MEDIA),
    )
    assert fake.calls == 14
    final = json.loads((output / "baseline_generation_manifest.json").read_text(encoding="utf-8"))
    assert final["video_count"] == 588
    assert final["status"] == "frozen_after_full_media_validation"
    assert Counter(item["stream"] for item in final["items"]) == {
        "cogvideox_original": 294,
        "videoeraser_official": 294,
    }
    master = queue.read_master_formal_cases(Path(plan["inputs"]["formal_cases"]))
    for stream in streams:
        stream_items = [item for item in final["items"] if item["stream"] == stream]
        assert {item["case_id"] for item in stream_items} == set(master)
        assert all(item["seed"] == master[item["case_id"]]["seed"] for item in stream_items)
    aggregate = json.loads((output / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["status"] == "completed"
    assert aggregate["status_counts"] == {"completed": 14}

    queue.execute_plan(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed job relaunched")
        ),
        media_probe=lambda _python, _video: dict(queue.MEDIA),
    )
    first_job = plan["jobs"][0]
    first_child = json.loads(
        (Path(first_job["output_dir"]) / "generation_plan.json").read_text(encoding="utf-8")
    )
    Path(first_child["items"][0]["video_path"]).write_bytes(b"tampered")
    with pytest.raises(queue.QueueError, match="video hash mismatch"):
        queue.preflight_resume(
            plan, media_probe=lambda _python, _video: dict(queue.MEDIA)
        )


def test_fresh_only_plan_and_strict_status_resume(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_baselines"
    plan = _build(
        fixture,
        output,
        streams=("safree_cogvideox",),
        gpus=(2, 3),
        safree_limit=1,
    )
    assert plan["scheduler"]["jobs_per_wave"] == [1] * 7
    queue.prepare_output_root(output, plan)
    with pytest.raises(queue.QueueError, match="must be fresh"):
        queue.prepare_output_root(output, plan)
    first = plan["jobs"][0]
    _write_json(Path(first["status_path"]), queue.status_payload(first, "running"))
    with pytest.raises(queue.QueueError, match="refusing non-completed status running"):
        queue.preflight_resume(
            plan, media_probe=lambda _python, _video: dict(queue.MEDIA)
        )


def test_registry_must_authorize_formal_generation(tmp_path):
    fixture = _fixture(tmp_path)
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["formal_generation_authorized"] = False
    _write_json(fixture["registry"], registry)
    _write_json(
        fixture["receipt"],
        {
            "protocol": baseline.REGISTRY_PROTOCOL,
            "registry_sha256": queue.sha256_file(fixture["registry"]),
        },
    )
    with pytest.raises(queue.QueueError, match="does not authorize"):
        queue.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            queue.STREAMS,
        )


def test_runtime_reprobe_and_builder_self_hash_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    calls = []

    def observed_runtime(python: Path):
        calls.append(python)
        return fixture["runtime_contract"]

    queue.load_registry(
        fixture["project"],
        fixture["registry"],
        fixture["receipt"],
        ("negative_prompt",),
        runtime_probe=observed_runtime,
    )
    assert calls == [Path(sys.executable).resolve()]
    changed_runtime = {
        **fixture["runtime_contract"],
        "packages": {**fixture["runtime_contract"]["packages"], "torch": "changed"},
    }
    with pytest.raises(queue.QueueError, match="runtime packages changed"):
        queue.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            ("negative_prompt",),
            runtime_probe=lambda _python: changed_runtime,
        )

    builder = fixture["project"] / queue.BUILDER_RELATIVE
    builder.write_text("# changed builder\n", encoding="utf-8")
    with pytest.raises(queue.QueueError, match="builder changed"):
        queue.load_registry(
            fixture["project"],
            fixture["registry"],
            fixture["receipt"],
            ("negative_prompt",),
            runtime_probe=lambda _python: fixture["runtime_contract"],
        )


def test_job_validation_rejects_seed_that_differs_from_master(tmp_path):
    fixture = _fixture(tmp_path)
    output = fixture["project"] / "outputs/formal_baselines"
    plan = _build(fixture, output, streams=("negative_prompt",), safree_limit=4)
    job = plan["jobs"][0]
    fake = _FakeFormalRunner(plan["inputs"]["baseline_registry_sha256"])
    fake(job["command"])
    child_root = Path(job["output_dir"])
    child_plan_path = child_root / "generation_plan.json"
    child_manifest_path = child_root / "generation_manifest.json"
    child_plan = json.loads(child_plan_path.read_text(encoding="utf-8"))
    child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
    item = child_plan["items"][0]
    item["seed"] += 1
    child_manifest["items"][0]["seed"] = item["seed"]
    receipt_path = Path(item["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["seed"] = item["seed"]
    child_manifest["outputs"][0]["seed"] = item["seed"]
    _write_json(receipt_path, receipt)
    _write_json(child_plan_path, child_plan)
    _write_json(child_manifest_path, child_manifest)
    (child_root / ".complete").write_text(
        queue.sha256_file(child_manifest_path) + "\n", encoding="utf-8"
    )
    with pytest.raises(queue.QueueError, match="seed differs from master formal case"):
        queue.validate_job_outputs(
            job,
            plan,
            media_probe=lambda _python, _video: dict(queue.MEDIA),
        )


def test_registered_runners_bootstrap_sibling_imports_in_sanitized_subprocess(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PYTHONSAFEPATH="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    scripts = [ROOT / queue.RUNNER_RELATIVE]
    scripts.extend(ROOT / relative for relative in sorted(set(queue.RUNNER_BY_STREAM.values())))
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"
