from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_role_erasure_7mechanism_wan_original_v2 as runner  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()

    runner_code = project / runner.RUNNER_RELATIVE
    runner_code.parent.mkdir(parents=True)
    runner_code.write_bytes(Path(runner.__file__).read_bytes())
    queue_code = project / runner.QUEUE_RELATIVE
    queue_code.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")

    generator = project / runner.DEFAULT_GENERATOR
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator.write_text("# frozen fake Wan generator\n", encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_GENERATOR_SHA256", runner.sha256_file(generator))

    python = project / runner.DEFAULT_PYTHON
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    python.chmod(0o755)

    def fake_runtime_probe(executable: Path) -> dict[str, object]:
        return {
            "executable": str(executable.resolve()),
            "binary_format": "ELF",
            "python_executable_sha256": runner.sha256_file(executable),
            "python": {"implementation": "CPython", "version": "3.11.15"},
            "package_versions": {
                "torch": "2.6.0",
                "diffusers": "0.33.1",
                "transformers": "4.51.3",
            },
            "module_versions": {
                "torch": "2.6.0+cu124",
                "diffusers": "0.33.1",
                "transformers": "4.51.3",
            },
        }

    monkeypatch.setattr(runner, "probe_runtime_process", fake_runtime_probe)

    model = project / runner.DEFAULT_MODEL
    model.mkdir(parents=True)
    model_file = model / "model.bin"
    model_file.write_bytes(b"frozen base model")
    model_inventory = project / runner.DEFAULT_MODEL_INVENTORY
    _write_json(
        model_inventory,
        {
            "protocol": "water_impact_dynamic_v4_model_content_inventory_v3",
            "status": "frozen",
            "model_root": runner.DEFAULT_MODEL.as_posix(),
            "files": [
                {
                    "path": (runner.DEFAULT_MODEL / model_file.name).as_posix(),
                    "sha256": runner.sha256_file(model_file),
                    "size_bytes": model_file.stat().st_size,
                }
            ],
        },
    )
    monkeypatch.setattr(
        runner, "EXPECTED_MODEL_INVENTORY_SHA256", runner.sha256_file(model_inventory)
    )

    runtime_registry = project / runner.DEFAULT_RUNTIME_REGISTRY
    _write_json(
        runtime_registry,
        {
            "protocol": "water_impact_dynamic_v4_runtime_registry_v3",
            "status": "frozen",
            "python_executable": runner.DEFAULT_PYTHON.as_posix(),
            "python": {"implementation": "CPython", "version": "3.11.15"},
            "torch": {"distribution_version": "2.6.0", "module_version": "2.6.0+cu124"},
            "packages": {
                "torch": "2.6.0",
                "diffusers": "0.33.1",
                "transformers": "4.51.3",
            },
        },
    )
    monkeypatch.setattr(
        runner, "EXPECTED_RUNTIME_REGISTRY_SHA256", runner.sha256_file(runtime_registry)
    )

    formal: list[dict[str, str]] = []
    for mechanism_index, mechanism in enumerate(runner.MECHANISMS):
        for local_index in range(runner.JOB_ROWS):
            global_index = mechanism_index * runner.JOB_ROWS + local_index
            formal.append(
                {
                    "protocol_version": runner.PROTOCOL_ID,
                    "case_id": f"eval7m_m{mechanism_index:02d}_c{local_index:02d}",
                    "global_case_index": str(global_index),
                    "mechanism_index": str(mechanism_index),
                    "mechanism": mechanism,
                    "case_kind": "causal" if local_index < 24 else "specificity",
                    "source_object": f"source object {global_index}",
                    "prompt": f"Formal prompt {global_index}.",
                    "expected_counterfactual_state": f"clean state {global_index}",
                    "seed": str(1_000_000 + global_index),
                    "num_frames": "49",
                    "fps": "8",
                    "height": "480",
                    "width": "832",
                }
            )
    formal_path = project / runner.DEFAULT_FORMAL
    _write_csv(formal_path, formal)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_SHA256", runner.sha256_file(formal_path))

    output = project / runner.DEFAULT_OUTPUT_ROOT
    plan = runner.build_plan(
        project_root=project,
        formal_path=formal_path,
        output_root=output,
        runner_code=runner_code,
        queue_code=queue_code,
        generator=generator,
        python=python,
        model=model,
        model_inventory=model_inventory,
        runtime_registry=runtime_registry,
        gpus=(0, 1, 2, 3),
    )
    return project, output, plan


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class _FinishedProcess:
    def poll(self):
        return 0

    def terminate(self):
        return None

    def kill(self):
        return None


class _FakeGenerator:
    def __init__(self):
        self.calls = 0

    def __call__(self, command, **_kwargs):
        self.calls += 1
        command = list(command)
        assert "--lora-path" not in command
        assert "--skip-existing" not in command
        output = Path(_value(command, "--output-dir"))
        shard = Path(_value(command, "--prompts"))
        output.mkdir()
        videos = output / "videos"
        videos.mkdir()
        seeds = [int(value) for value in _value(command, "--seeds").split(",")]
        prompt_items = [line.split(" | ") for line in shard.read_text(encoding="utf-8").splitlines()]
        items = []
        for index, ((prompt, target, effect), seed) in enumerate(zip(prompt_items, seeds)):
            video = videos / f"{index:03d}_seed{seed}.mp4"
            video.write_bytes(f"fake-video-{seed}".encode("utf-8"))
            items.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "target_concept": target,
                    "expected_effect": effect,
                    "seed": seed,
                    "video_path": str(video),
                }
            )
        generation = {
            "baseline": "clean",
            "seed": int(_value(command, "--seed")),
            "seeds": seeds,
            "num_inference_steps": int(_value(command, "--steps")),
            "guidance_scale": float(_value(command, "--guidance-scale")),
            "num_frames": int(_value(command, "--num-frames")),
            "fps": int(_value(command, "--fps")),
            "height": int(_value(command, "--height")),
            "width": int(_value(command, "--width")),
            "dtype": _value(command, "--dtype"),
            "device": _value(command, "--device"),
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "vae_slicing": "--vae-slicing" in command,
            "vae_tiling": "--vae-tiling" in command,
            "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
            "lora_path": None,
            "lora_sha256": None,
            "lora_scale": 1.0,
            "activation_gate_dir": None,
            "persistent_activation_gate": False,
            "lora_target_phrases": [],
            "attention_gate_dir": None,
            "attention_suppression_phrases": [],
            "attention_suppression_strength": 20.0,
        }
        _write_json(
            output / "generation_manifest.json",
            {
                "baseline": "clean",
                "pipeline": "WanPipeline",
                "model": _value(command, "--model"),
                "dry_run": False,
                "prompts": str(shard),
                "generation": generation,
                "items": items,
            },
        )
        return _FinishedProcess()


def _fake_media(_python: Path, _video: Path) -> dict[str, object]:
    return dict(runner.MEDIA_CONTRACT)


def test_plan_is_exactly_seven_no_lora_jobs_in_four_plus_three_waves(
    tmp_path: Path, monkeypatch
):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    assert output.name == "formal_wan_original_v2"
    assert plan["stream"] == "wan_original"
    assert plan["job_count"] == 7
    assert plan["expected_videos"] == 294
    assert [sum(job["wave_index"] == wave for job in plan["jobs"]) for wave in range(2)] == [4, 3]
    assert [sum(job["row_count"] for job in plan["jobs"] if job["wave_index"] == wave) for wave in range(2)] == [168, 126]
    assert [job["mechanism"] for job in plan["jobs"]] == list(runner.MECHANISMS)
    assert all("--lora-path" not in job["command"] for job in plan["jobs"])
    assert all("--skip-existing" not in job["command"] for job in plan["jobs"])
    assert all(_value(job["command"], "--steps") == "25" for job in plan["jobs"])
    assert all(_value(job["command"], "--guidance-scale") == "5.0" for job in plan["jobs"])


def test_cpu_fake_run_freezes_294_receipted_videos_and_exact_resume_skips_all(
    tmp_path: Path, monkeypatch
):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    fake = _FakeGenerator()
    runner.execute_plan(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=fake,
        sleep_fn=lambda _seconds: None,
        media_probe=_fake_media,
    )
    assert fake.calls == 7
    final = json.loads((output / "wan_original_generation_manifest.json").read_text())
    assert final["protocol_id"] == runner.PROTOCOL_ID
    assert final["runner_id"] == runner.RUNNER_ID
    assert final["status"] == "frozen_after_exact_294_video_validation"
    assert final["video_count"] == 294
    assert [item["formal_global_index"] for item in final["items"]] == list(range(294))
    assert {item["stream"] for item in final["items"]} == {"wan_original"}
    assert all(item["media"] == runner.MEDIA_CONTRACT for item in final["items"])
    aggregate = json.loads((output / "wan_original_aggregate.json").read_text())
    assert aggregate["status"] == "completed"
    assert aggregate["validated_videos"] == 294
    assert aggregate["status_counts"] == {"completed": 7}
    assert len(list((output / "receipts").glob("*.json"))) == 7

    runner.execute_plan(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=fake,
        sleep_fn=lambda _seconds: None,
        media_probe=_fake_media,
    )
    assert fake.calls == 7


def test_resume_refuses_a_planned_job_with_partial_output(tmp_path: Path, monkeypatch):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    Path(plan["jobs"][0]["output_dir"]).mkdir()
    with pytest.raises(ValueError, match="refusing partial output"):
        runner.preflight_resume(output, plan, media_probe=_fake_media)


def test_media_audio_is_fail_closed(tmp_path: Path, monkeypatch):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    job = plan["jobs"][0]
    _FakeGenerator()(job["command"])
    bad_media = dict(runner.MEDIA_CONTRACT)
    bad_media["audio_streams"] = 1
    with pytest.raises(ValueError, match="media contract mismatch"):
        runner.validate_job_outputs(job, plan, media_probe=lambda *_args: bad_media)


def test_completed_receipt_drift_is_fail_closed(tmp_path: Path, monkeypatch):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    runner.execute_plan(
        plan,
        output,
        poll_interval=0.001,
        popen_factory=_FakeGenerator(),
        sleep_fn=lambda _seconds: None,
        media_probe=_fake_media,
    )
    receipt = Path(plan["jobs"][0]["receipt_path"])
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt/output drift"):
        runner.preflight_resume(output, plan, media_probe=_fake_media)


def test_live_model_content_must_match_frozen_inventory(tmp_path: Path, monkeypatch):
    project, _, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(Path(plan["output_root"]), plan)
    model_file = project / runner.DEFAULT_MODEL / "model.bin"
    model_file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size mismatch|SHA mismatch"):
        runner.validate_bound_inputs(plan, runtime=True, verify_model_content=True)


@pytest.mark.parametrize("field", ["runner", "queue"])
def test_runner_and_queue_code_tamper_are_rejected(
    tmp_path: Path, monkeypatch, field: str
):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    code = Path(plan["implementation"][field])
    code.write_bytes(code.read_bytes() + b"# changed\n")
    with pytest.raises(ValueError, match=f"Wan Original {field} changed"):
        runner.validate_bound_inputs(plan, runtime=False)


def test_runtime_binary_or_package_drift_is_rejected(tmp_path: Path, monkeypatch):
    _, output, plan = _fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    runtime_python = Path(plan["implementation"]["python_executable"])
    runtime_python.write_bytes(runtime_python.read_bytes() + b"# changed\n")
    runtime_python.chmod(0o755)
    with pytest.raises(ValueError, match="runtime changed after planning"):
        runner.validate_bound_inputs(plan, runtime=True)

    # Restore the planned bytes, then independently prove version drift fails.
    runtime_python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    runtime_python.chmod(0o755)
    observed = dict(plan["inputs"]["runtime_registry"]["observation"])
    observed["package_versions"] = dict(observed["package_versions"])
    observed["package_versions"]["diffusers"] = "999.0"
    monkeypatch.setattr(runner, "probe_runtime_process", lambda _python: observed)
    with pytest.raises(ValueError, match="diffusers distribution version"):
        runner.validate_bound_inputs(plan, runtime=True)


def test_shell_queue_targets_only_the_independent_output_root():
    script = (
        PROJECT_ROOT
        / "scripts/run_causal_role_erasure_7mechanism_wan_original_queue_v2.sh"
    ).read_text(encoding="utf-8")
    assert "formal_wan_original_v2" in script
    assert "run_causal_role_erasure_7mechanism_wan_original_v2.py" in script
    assert "formal_eval_v2" not in script
