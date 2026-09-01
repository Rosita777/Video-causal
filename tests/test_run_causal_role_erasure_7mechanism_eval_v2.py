from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_role_erasure_7mechanism_eval_v2 as runner  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    generator = project / runner.DEFAULT_GENERATOR
    generator.parent.mkdir(parents=True)
    generator.write_text("# fake frozen generator\n", encoding="utf-8")
    runtime = project / runner.DEFAULT_PYTHON
    runtime.parent.mkdir(parents=True)
    runtime.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    runtime.chmod(0o755)
    model = project / runner.DEFAULT_MODEL
    model.mkdir(parents=True)

    model_registry = project / "data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json"
    runtime_registry = project / "data/water_impact_dynamic_v4/v4_runtime_registry_v3.json"
    model_registry.parent.mkdir(parents=True)
    model_registry.write_text('{"status":"frozen-model"}\n', encoding="utf-8")
    runtime_registry.write_text('{"status":"frozen-runtime"}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_MODEL_REGISTRY_SHA256", runner.sha256_file(model_registry))
    monkeypatch.setattr(runner, "EXPECTED_RUNTIME_REGISTRY_SHA256", runner.sha256_file(runtime_registry))
    monkeypatch.setattr(runner, "EXPECTED_GENERATOR_SHA256", runner.sha256_file(generator))

    formal: list[dict[str, str]] = []
    for mechanism_index, mechanism in enumerate(runner.MECHANISMS):
        for local in range(42):
            global_index = mechanism_index * 42 + local
            formal.append(
                {
                    "protocol_version": runner.PROTOCOL_ID,
                    "case_id": f"eval7m_m{mechanism_index:02d}_c{local:02d}",
                    "global_case_index": str(global_index),
                    "mechanism_index": str(mechanism_index),
                    "historical_capability_index": str(mechanism_index),
                    "mechanism": mechanism,
                    "mechanism_name": mechanism,
                    "case_kind": "causal" if local < 24 else "specificity",
                    "generalization_group": "group",
                    "source_membership": "membership",
                    "prompt_style": "direct" if local % 2 == 0 else "natural",
                    "footprint_lexicalization": "explicit" if local < 24 else "",
                    "specificity_subtype": "" if local < 24 else "bystander",
                    "semantic_replicate": str(local % 2) if local < 24 else "",
                    "source_id": f"source_{mechanism_index}_{local}",
                    "source_object": f"source object {mechanism_index} {local}",
                    "receiver_id": f"receiver_{mechanism_index}_{local}",
                    "receiver": f"receiver {mechanism_index} {local}",
                    "prompt": f"Formal prompt {mechanism_index} {local}.",
                    "expected_trigger": "trigger",
                    "expected_footprint": "footprint",
                    "expected_counterfactual_state": f"counterfactual state {mechanism_index} {local}",
                    "protected_object": "",
                    "acceptable_alternative_cause": "",
                    "m6_pair_id": "",
                    "seed": str(100_000 + global_index),
                    "seed_nonce": "0",
                    "num_frames": "49",
                    "fps": "8",
                    "height": "480",
                    "width": "832",
                }
            )
    formal_path = project / runner.DEFAULT_FORMAL
    _write_csv(formal_path, formal)
    monkeypatch.setattr(runner, "EXPECTED_FORMAL_SHA256", runner.sha256_file(formal_path))

    identification: list[dict[str, str]] = []
    for mechanism in runner.IDENTIFICATION_MECHANISMS:
        for formal_row in [row for row in formal if row["mechanism"] == mechanism][:24]:
            identification.append(
                {
                    "protocol_version": runner.PROTOCOL_ID,
                    "identification_case_index": str(len(identification)),
                    "mechanism": mechanism,
                    "case_id": formal_row["case_id"],
                    "case_kind": formal_row["case_kind"],
                    "generalization_group": formal_row["generalization_group"],
                    "source_membership": formal_row["source_membership"],
                    "prompt_style": formal_row["prompt_style"],
                    "footprint_lexicalization": formal_row["footprint_lexicalization"],
                    "specificity_subtype": formal_row["specificity_subtype"],
                    "seed": formal_row["seed"],
                    "included_streams": "matched_control,V4",
                    "additional_streams": "generic_paraphrase,bystander_token",
                }
            )
    identification_path = project / runner.DEFAULT_IDENTIFICATION
    _write_csv(identification_path, identification)
    monkeypatch.setattr(runner, "EXPECTED_IDENTIFICATION_SHA256", runner.sha256_file(identification_path))

    specs: list[tuple[str, str, str]] = []
    specs.extend((mechanism, "matched_control", "main") for mechanism in runner.MECHANISMS)
    specs.extend((mechanism, "V4", "main") for mechanism in runner.MECHANISMS)
    for mechanism in runner.IDENTIFICATION_MECHANISMS:
        specs.extend((mechanism, arm, "identification") for arm in runner.IDENTIFICATION_ARMS)
    matrix: list[dict[str, str]] = []
    run_spec_refs: dict[str, dict[str, object]] = {}
    receipt_inputs: list[tuple[dict[str, str], Path, str]] = []
    for index, (mechanism, arm, group) in enumerate(specs):
        run_id = f"train7m_{mechanism}_{arm.lower()}"
        output_dir = f"outputs/causal_role_erasure_7mechanism_main_v2/training/{run_id}"
        matrix_row = {
            "protocol_version": runner.PROTOCOL_ID,
            "run_index": str(index),
            "run_id": run_id,
            "mechanism_index": str(runner.MECHANISMS.index(mechanism)),
            "mechanism": mechanism,
            "arm": arm,
            "run_group": group,
            "training_seed": "26000",
            "updates": "200",
            "erase_updates": "100",
            "preserve_updates": "100",
            "selected_erase_rows_required": "178",
            "preserve_rows": "36",
            "lora_rank": "16",
            "lora_alpha": "16",
            "learning_rate": "5e-5",
            "target_teacher_weight": "4",
            "preservation_weight": "4",
            "checkpoint": "200",
            "inference_scale": "1.25",
            "authorization_state": "requires_selected178_and_frozen_caches",
            "output_dir": output_dir,
        }
        matrix.append(matrix_row)
        spec_rel = Path("outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2/run_specs") / f"{index:02d}_{run_id}.json"
        spec_path = project / spec_rel
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7mechanism_training_run_spec_v2",
            "protocol_version": runner.PROTOCOL_ID,
            "status": "frozen",
            "run_id": run_id,
            "mechanism": mechanism,
            "arm": arm,
            "output_dir": output_dir,
            "model_root": str(runner.DEFAULT_MODEL),
            "python_executable": str(runner.DEFAULT_PYTHON),
            "model_inventory": {"path": str(model_registry.relative_to(project)), "sha256": runner.sha256_file(model_registry)},
            "runtime_registry": {"path": str(runtime_registry.relative_to(project)), "sha256": runner.sha256_file(runtime_registry)},
            "training_config": {"checkpoint_step": 200, "inference_lora_scale": 1.25},
        }
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        spec_sha = runner.sha256_file(spec_path)
        run_spec_refs[run_id] = {"path": str(spec_rel), "row_count": 1, "sha256": spec_sha}
        receipt_inputs.append((matrix_row, spec_path, spec_sha))

    matrix_path = project / runner.DEFAULT_RUN_MATRIX
    _write_csv(matrix_path, matrix)
    monkeypatch.setattr(runner, "EXPECTED_RUN_MATRIX_SHA256", runner.sha256_file(matrix_path))
    registry_path = project / runner.DEFAULT_RUN_SPEC_REGISTRY
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "training_run_specs_frozen_after_cache_validation",
                "run_spec_count": 18,
                "run_specs": run_spec_refs,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry_sha = runner.sha256_file(registry_path)
    monkeypatch.setattr(runner, "EXPECTED_RUN_SPEC_REGISTRY_SHA256", registry_sha)

    for matrix_row, spec_path, spec_sha in receipt_inputs:
        training_dir = project / matrix_row["output_dir"]
        checkpoint = training_dir / "checkpoint-000200"
        checkpoint.mkdir(parents=True)
        weights = checkpoint / "pytorch_lora_weights.safetensors"
        state = checkpoint / "training_state.json"
        weights.write_bytes(f"weights:{matrix_row['run_id']}".encode())
        state.write_text('{"step":200}\n', encoding="utf-8")
        receipt = {
            "protocol": "wan_causal_role_lora_training_v2",
            "protocol_version": runner.PROTOCOL_ID,
            "status": "eligible",
            "step": 200,
            "run_id": matrix_row["run_id"],
            "mechanism": matrix_row["mechanism"],
            "arm": matrix_row["arm"],
            "run_spec_sha256": spec_sha,
            "inference_lora_scale": 1.25,
            "finite_receipt": {"status": "passed", "nonfinite_tensor_count": 0},
            "role_step_counts": {"erase": 100, "preserve": 100},
            "cache_validation_basis": {"run_spec_registry": {"sha256": registry_sha}},
            "checkpoint": {
                "path": str(checkpoint),
                "weights_sha256": runner.sha256_file(weights),
                "training_state_sha256": runner.sha256_file(state),
            },
        }
        (training_dir / "run_receipt.json").write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    output = project / runner.DEFAULT_OUTPUT_ROOT
    plan = runner.build_plan(
        project_root=project,
        formal_path=formal_path,
        identification_path=identification_path,
        run_matrix_path=matrix_path,
        registry_path=registry_path,
        output_root=output,
        generator=generator,
        python=runtime,
        model=model,
        gpus=[0, 1, 2, 3],
    )
    return project, output, plan


def _fake_media(_runtime: Path, _video: Path) -> dict[str, object]:
    return {
        "video_streams": 1,
        "audio_streams": 0,
        "decoded_frames": 49,
        "fps": "8/1",
        "width": 832,
        "height": 480,
    }


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
        values = {command[index]: command[index + 1] for index in range(2, len(command) - 1, 2) if command[index].startswith("--") and not command[index + 1].startswith("--")}
        flags = {item for item in command if item.startswith("--")}
        output = Path(values["--output-dir"])
        prompts = Path(values["--prompts"])
        checkpoint = Path(values["--lora-path"])
        output.mkdir(parents=True)
        videos = output / "videos"
        videos.mkdir()
        parsed = [line.split(" | ") for line in prompts.read_text(encoding="utf-8").splitlines()]
        seeds = [int(seed) for seed in values["--seeds"].split(",")]
        items = []
        for index, ((prompt, target, effect), seed) in enumerate(zip(parsed, seeds)):
            video = videos / f"{index:03d}_fake_seed{seed}.mp4"
            video.write_bytes(f"fake-video-{index}-{seed}".encode())
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
            "seed": 42,
            "seeds": seeds,
            "num_inference_steps": int(values["--steps"]),
            "guidance_scale": float(values["--guidance-scale"]),
            "num_frames": int(values["--num-frames"]),
            "fps": int(values["--fps"]),
            "height": int(values["--height"]),
            "width": int(values["--width"]),
            "dtype": values["--dtype"],
            "device": values["--device"],
            "enable_model_cpu_offload": "--enable-model-cpu-offload" in flags,
            "enable_sequential_cpu_offload": "--enable-sequential-cpu-offload" in flags,
            "vae_slicing": "--vae-slicing" in flags,
            "vae_tiling": "--vae-tiling" in flags,
            "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
            "lora_path": str(checkpoint),
            "lora_sha256": runner.artifact_sha256(checkpoint),
            "lora_scale": float(values["--lora-scale"]),
            "activation_gate_dir": None,
            "persistent_activation_gate": False,
            "lora_target_phrases": [],
            "attention_gate_dir": None,
            "attention_suppression_phrases": [],
            "attention_suppression_strength": 20.0,
        }
        (output / "generation_manifest.json").write_text(
            json.dumps(
                {
                    "baseline": "clean",
                    "pipeline": "WanPipeline",
                    "model": values["--model"],
                    "dry_run": False,
                    "prompts": str(prompts),
                    "generation": generation,
                    "items": items,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return _FinishedProcess()


def test_plan_binds_exact_18_jobs_and_684_outputs(tmp_path: Path, monkeypatch):
    _, _, plan = _build_fixture(tmp_path, monkeypatch)
    assert len(plan["jobs"]) == 18
    assert plan["expected_videos"] == 684
    assert [sum(job["row_count"] for job in plan["jobs"] if job["wave_index"] == wave) for wave in range(5)] == [168, 168, 168, 132, 48]
    assert [sum(job["wave_index"] == wave for job in plan["jobs"]) for wave in range(5)] == [4, 4, 4, 4, 2]
    assert sum(job["row_count"] for job in plan["jobs"][:14]) == 588
    assert sum(job["row_count"] for job in plan["jobs"][14:]) == 96
    assert all("--skip-existing" not in job["command"] for job in plan["jobs"])


def test_fake_generation_freezes_684_and_completed_resume_skips_all(tmp_path: Path, monkeypatch):
    _, output, plan = _build_fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    fake = _FakeGenerator()
    runner.execute_plan(
        plan,
        output,
        poll_interval=0.01,
        popen_factory=fake,
        sleep_fn=lambda _seconds: None,
        media_probe=_fake_media,
    )
    assert fake.calls == 18
    final = json.loads((output / "eval_generation_manifest.json").read_text())
    assert final["status"] == "frozen_after_exact_684_video_validation"
    assert final["video_count"] == 684
    assert len(final["items"]) == 684
    runner.execute_plan(
        plan,
        output,
        poll_interval=0.01,
        popen_factory=fake,
        sleep_fn=lambda _seconds: None,
        media_probe=_fake_media,
    )
    assert fake.calls == 18


def test_resume_refuses_any_planned_job_with_partial_output(tmp_path: Path, monkeypatch):
    _, output, plan = _build_fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output, plan)
    partial = Path(plan["jobs"][0]["output_dir"])
    partial.mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing partial output"):
        runner.preflight_resume(plan, media_probe=_fake_media)


def test_probe_contract_rejects_audio(tmp_path: Path, monkeypatch):
    _, output_root, plan = _build_fixture(tmp_path, monkeypatch)
    runner.prepare_output_root(output_root, plan)
    job = plan["jobs"][0]
    bad = dict(_fake_media(Path(), Path()))
    bad["audio_streams"] = 1
    with pytest.raises(ValueError, match="media contract mismatch"):
        # Build one fake output, then route validation through an audio-bearing probe.
        output = Path(job["output_dir"])
        _FakeGenerator()(job["command"])
        runner.validate_job_outputs(job, plan, media_probe=lambda *_args: bad)
