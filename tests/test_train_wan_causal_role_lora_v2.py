from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_wan_causal_role_lora_v2 as trainer  # noqa: E402


def _write_cache(project: Path, name: str, mode: str, count: int, *, arm=None):
    root = project / f"caches/{name}"
    root.mkdir(parents=True)
    (root / "cache_plan.json").write_text("{}\n", encoding="utf-8")
    files = []
    for index in range(count):
        if mode == "prepare-base":
            erase = index < trainer.ERASE_ROWS
            row_id = f"candidate_{index:03d}" if erase else f"preserve_{index - trainer.ERASE_ROWS:03d}"
            payload = {
                "row_id": row_id,
                "training_role": "erase" if erase else "preserve",
                "latents": {"shape": list(trainer.LATENT_SHAPE), "dtype": trainer.DTYPE, "value": index},
                "prompt_embeds": {"shape": list(trainer.PROMPT_SHAPE), "dtype": trainer.DTYPE, "value": f"base-{index}"},
            }
            prompt_hash = f"{index:064x}"[-64:]
            record = {"index": index, "row_id": row_id, "prompt_embeds_sha256": prompt_hash}
        else:
            row_id = f"candidate_{index:03d}"
            key = "teacher_prompt_embeds" if mode == "prepare-teacher" else "student_prompt_embeds"
            value = f"base-{index}" if arm == "matched" else f"{arm}-{index}"
            payload = {
                "candidate_id": row_id,
                key: {"shape": list(trainer.PROMPT_SHAPE), "dtype": trainer.DTYPE, "value": value},
            }
            prompt_hash = f"{index:064x}"[-64:] if arm == "matched" else f"{index + 1000:064x}"[-64:]
            record = {"index": index, "row_id": row_id, "prompt_embeds_sha256": prompt_hash}
        path = root / f"{index:03d}_{row_id}.pt"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        record.update(file=path.name, sha256=trainer.sha256_file(path))
        files.append(record)
    paths = [root / item["file"] for item in files]
    manifest = {
        "protocol_version": trainer.PROTOCOL_VERSION,
        "mode": mode,
        "arm": arm,
        "mechanism": "rigid_collision",
        "row_count": count,
        "ordered_inventory_sha256": trainer.inventory_sha256(paths),
        "files": files,
    }
    manifest_path = root / "cache_manifest.json"
    manifest_path.write_bytes(trainer.canonical_json_bytes(manifest))
    return root, manifest, {
        "path": root.relative_to(project).as_posix(),
        "manifest_sha256": trainer.sha256_file(manifest_path),
        "ordered_inventory_sha256": manifest["ordered_inventory_sha256"],
        "row_count": count,
    }


def _artifact(project: Path, path: Path, rows: int):
    return {"path": path.relative_to(project).as_posix(), "sha256": trainer.sha256_file(path), "row_count": rows}


def _fixture(tmp_path: Path, *, arm="V4"):
    project = tmp_path / "project"
    project.mkdir()
    base_root, base_manifest, base_record = _write_cache(project, "base", "prepare-base", 214)
    teacher_root, teacher_manifest, teacher_record = _write_cache(project, "teacher", "prepare-teacher", 178)
    internal_arm = trainer.ARM_CACHE_NAMES[arm]
    arm_root, arm_manifest, arm_record = _write_cache(project, "arm", "prepare-arm", 178, arm=internal_arm)

    model = project / "models/Wan"
    model.mkdir(parents=True)
    model_file = model / "weights.bin"
    model_file.write_bytes(b"model")
    model_inventory = project / "registries/model.json"
    model_inventory.parent.mkdir(parents=True)
    model_inventory.write_text(
        json.dumps({"files": [{"path": "weights.bin", "sha256": trainer.sha256_file(model_file), "size_bytes": 5}]}) + "\n",
        encoding="utf-8",
    )
    runtime_root = project / "models/runtime"
    python = runtime_root / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    runtime_registry = project / "registries/runtime.json"
    runtime_registry.write_text(json.dumps({"status": "frozen"}) + "\n", encoding="utf-8")
    model_record = _artifact(project, model_inventory, 1)
    runtime_record = _artifact(project, runtime_registry, 1)
    selected_record = {"path": "data/selected178.csv", "sha256": "a" * 64, "row_count": 178}
    for root, manifest, record in (
        (base_root, base_manifest, base_record),
        (teacher_root, teacher_manifest, teacher_record),
        (arm_root, arm_manifest, arm_record),
    ):
        manifest["model_inventory"] = model_record
        manifest["runtime_registry"] = runtime_record
        manifest["selected178"] = selected_record
        manifest_path = root / "cache_manifest.json"
        manifest_path.write_bytes(trainer.canonical_json_bytes(manifest))
        record["manifest_sha256"] = trainer.sha256_file(manifest_path)

    receipt_record = None
    if arm == "matched_control":
        items = [
            {
                "selected_index": index,
                "candidate_id": base_manifest["files"][index]["row_id"],
                "base_prompt_embeds_sha256": base_manifest["files"][index]["prompt_embeds_sha256"],
                "arm_prompt_embeds_sha256": arm_manifest["files"][index]["prompt_embeds_sha256"],
                "tensor_equal": True,
            }
            for index in range(178)
        ]
        receipt = project / "registries/matched_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "protocol_version": trainer.PROTOCOL_VERSION,
                    "status": "passed",
                    "mechanism": "rigid_collision",
                    "arm": "matched_control",
                    "row_count": 178,
                    "base_cache_manifest_sha256": base_record["manifest_sha256"],
                    "arm_cache_manifest_sha256": arm_record["manifest_sha256"],
                    "items": items,
                }
            ) + "\n",
            encoding="utf-8",
        )
        receipt_record = _artifact(project, receipt, 178)

    spec = project / "registries/run.json"
    output = project / f"outputs/{arm}"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": trainer.RUN_SPEC_PROTOCOL,
                "protocol_version": trainer.PROTOCOL_VERSION,
                "status": "frozen",
                "run_id": f"rigid_{arm}",
                "mechanism": "rigid_collision",
                "arm": arm,
                "base_cache": base_record,
                "teacher_cache": teacher_record,
                "arm_cache": arm_record,
                "model_root": model.relative_to(project).as_posix(),
                "model_inventory": model_record,
                "runtime_root": runtime_root.relative_to(project).as_posix(),
                "runtime_registry": runtime_record,
                "python_executable": python.relative_to(project).as_posix(),
                "training_config": trainer.EXPECTED_CONFIG,
                "output_dir": output.relative_to(project).as_posix(),
                "matched_tensor_equality_receipt": receipt_record,
            }
        ) + "\n",
        encoding="utf-8",
    )
    return project, spec, output


class FakeBackend:
    def __init__(self, *, preflight_pass=True):
        self.steps = []
        self.preflight_pass = preflight_pass

    def load_payload(self, path):
        return json.loads(path.read_text())

    def validate_base_payload(self, payload, role, row_id):
        assert payload["training_role"] == role
        assert payload["row_id"] == row_id
        assert tuple(payload["latents"]["shape"]) == trainer.LATENT_SHAPE
        assert payload["latents"]["dtype"] == trainer.DTYPE
        assert tuple(payload["prompt_embeds"]["shape"]) == trainer.PROMPT_SHAPE

    def validate_prompt_payload(self, payload, key, candidate_id):
        assert payload["candidate_id"] == candidate_id
        assert tuple(payload[key]["shape"]) == trainer.PROMPT_SHAPE
        assert payload[key]["dtype"] == trainer.DTYPE

    def initialize(self):
        return trainer.EXPECTED_INITIAL_LORA_SHA256

    def null_preflight(self, base, teacher, arm):
        return {
            "status": "passed" if self.preflight_pass else "failed",
            "forward_exact": self.preflight_pass,
            "loss_exact": self.preflight_pass,
            "gradient_comparison": "per_parameter_numeric_allclose",
            "gradient_parameter_count": 4,
            "legacy_byte_gradient_gate_used": False,
        }

    def begin_training(self):
        return trainer.EXPECTED_NOISE_RNG_INITIAL_SHA256

    def train_step(self, role, base, teacher, arm):
        self.steps.append(role)
        if role == "erase":
            assert teacher is not None and arm is not None
            return {"loss": 1.25, "flow_loss": 1.0, "teacher_loss": 0.0625, "preserve_loss": 0.0, "grad_norm": 0.5}
        assert teacher is None and arm is None
        return {"loss": 0.4, "flow_loss": 0.0, "teacher_loss": 0.0, "preserve_loss": 0.1, "grad_norm": 0.25}

    def noise_rng_final_sha256(self):
        return trainer.EXPECTED_NOISE_RNG_FINAL_SHA256

    def finite_receipt(self):
        return {"status": "passed", "tensor_count": 4, "nonfinite_tensor_count": 0, "trainable_state_sha256": "f" * 64}

    def save_checkpoint(self, path, metadata):
        path.mkdir()
        weights = path / "pytorch_lora_weights.safetensors"
        state = path / "training_state.json"
        weights.write_bytes(b"finite-weights")
        state.write_bytes(trainer.canonical_json_bytes(metadata))
        return {"path": str(path), "weights_sha256": trainer.sha256_file(weights), "training_state_sha256": trainer.sha256_file(state)}

    def close(self):
        pass


def _loaded(project: Path, spec_path: Path):
    spec = trainer.load_run_spec(project, spec_path, trainer.sha256_file(spec_path))
    base = trainer.validate_cache(spec["base_cache"], mode="prepare-base", mechanism=spec["mechanism"], arm=None)
    teacher = trainer.validate_cache(spec["teacher_cache"], mode="prepare-teacher", mechanism=spec["mechanism"], arm=None)
    arm = trainer.validate_cache(spec["arm_cache"], mode="prepare-arm", mechanism=spec["mechanism"], arm=trainer.ARM_CACHE_NAMES[spec["arm"]])
    trainer.validate_cache_stack(spec, base, teacher, arm)
    trainer.validate_matched_receipt(spec, base, arm)
    return spec, base, teacher, arm


def test_schedule_is_exact_alternating_100_100_and_hash_bound():
    schedule = trainer.build_schedule()
    assert len(schedule) == 200
    assert [role for role, _ in schedule][::2] == ["erase"] * 100
    assert [role for role, _ in schedule][1::2] == ["preserve"] * 100
    assert trainer.schedule_sha256(schedule) == trainer.EXPECTED_SCHEDULE_SHA256


def test_all_four_arms_share_identical_config_and_determinism_digests(tmp_path: Path):
    observed = []
    for arm_name in trainer.ARMS:
        arm_root = tmp_path / arm_name
        arm_root.mkdir()
        project, spec_path, _ = _fixture(arm_root, arm=arm_name)
        spec, _, _, _ = _loaded(project, spec_path)
        plan = trainer.build_plan(spec, dry_run=True)
        observed.append(
            (
                plan["training_config"], plan["schedule_sha256"],
                plan["training_config"]["expected_initial_lora_sha256"],
                plan["training_config"]["expected_noise_rng_initial_sha256"],
                plan["training_config"]["expected_noise_rng_final_sha256"],
            )
        )
    assert all(value == observed[0] for value in observed)


def test_dry_run_is_fresh_and_does_not_construct_backend(tmp_path: Path, monkeypatch):
    project, spec, output = _fixture(tmp_path)

    class Forbidden:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run loaded backend")

    monkeypatch.setattr(trainer, "RealBackend", Forbidden)
    assert trainer.main(
        ["--project-root", str(project), "--run-spec", str(spec), "--run-spec-sha256", trainer.sha256_file(spec), "--dry-run"]
    ) == 0
    plan = json.loads((output / "run_plan.json").read_text())
    assert plan["dry_run"] is True
    assert plan["training_config"] == trainer.EXPECTED_CONFIG
    assert plan["null_preflight"] == "numeric_forward_loss_exact_per_parameter_gradient_allclose_A_A"
    assert not list(output.glob("checkpoint-*"))
    assert trainer.main(
        ["--project-root", str(project), "--run-spec", str(spec), "--run-spec-sha256", trainer.sha256_file(spec), "--dry-run"]
    ) == 1


def test_run_spec_external_hash_and_unified_config_are_fail_closed(tmp_path: Path):
    project, spec_path, output = _fixture(tmp_path)
    with pytest.raises(ValueError, match="external expected"):
        trainer.load_run_spec(project, spec_path, "0" * 64)
    payload = json.loads(spec_path.read_text())
    payload["training_config"]["learning_rate"] = 1e-4
    spec_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unified config"):
        trainer.load_run_spec(project, spec_path, trainer.sha256_file(spec_path))
    assert not output.exists()


def test_cache_inventory_byte_drift_is_rejected(tmp_path: Path):
    project, spec_path, _ = _fixture(tmp_path)
    spec = trainer.load_run_spec(project, spec_path, trainer.sha256_file(spec_path))
    first = Path(spec["arm_cache"]["path"]) / "000_candidate_000.pt"
    first.write_bytes(first.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="binding mismatch"):
        trainer.validate_cache(spec["arm_cache"], mode="prepare-arm", mechanism="rigid_collision", arm="v4")


def test_matched_receipt_binds_all_178_equal_tensors(tmp_path: Path):
    project, spec_path, _ = _fixture(tmp_path, arm="matched_control")
    spec, base, _, arm = _loaded(project, spec_path)
    receipt = trainer.validate_matched_receipt(spec, base, arm)
    assert receipt["row_count"] == 178
    assert all(item["tensor_equal"] for item in receipt["items"])
    receipt_path = Path(spec["matched_tensor_equality_receipt"]["path"])
    payload = json.loads(receipt_path.read_text())
    payload["items"][0]["tensor_equal"] = False
    receipt_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte hash mismatch"):
        trainer.load_run_spec(project, spec_path, trainer.sha256_file(spec_path))


def test_fake_backend_executes_exact_200_steps_and_only_checkpoint200(tmp_path: Path):
    project, spec_path, output = _fixture(tmp_path)
    spec, base, teacher, arm = _loaded(project, spec_path)
    plan = trainer.build_plan(spec, dry_run=False)
    trainer.reserve_output(output, plan)
    backend = FakeBackend()
    receipt = trainer.run_training(plan, spec, backend, base, teacher, arm)
    assert backend.steps == ["erase", "preserve"] * 100
    assert receipt["role_step_counts"] == {"erase": 100, "preserve": 100}
    assert receipt["initial_lora_sha256"] == trainer.EXPECTED_INITIAL_LORA_SHA256
    assert receipt["noise_rng_final_sha256"] == trainer.EXPECTED_NOISE_RNG_FINAL_SHA256
    assert receipt["inference_lora_scale"] == 1.25
    assert [path.name for path in output.glob("checkpoint-*")] == ["checkpoint-000200"]
    null = json.loads((output / "null_preflight.json").read_text())
    assert null["forward_exact"] is True and null["loss_exact"] is True
    assert null["legacy_byte_gradient_gate_used"] is False
    assert set(output.iterdir()) == {
        output / "run_plan.json", output / "null_preflight.json",
        output / "run_receipt.json", output / "checkpoint-000200",
    }


def test_failed_numeric_aa_preflight_forbids_checkpoint(tmp_path: Path):
    project, spec_path, output = _fixture(tmp_path)
    spec, base, teacher, arm = _loaded(project, spec_path)
    plan = trainer.build_plan(spec, dry_run=False)
    trainer.reserve_output(output, plan)
    with pytest.raises(ValueError, match="numeric A-A"):
        trainer.run_training(plan, spec, FakeBackend(preflight_pass=False), base, teacher, arm)
    assert not list(output.glob("checkpoint-*"))
