#!/usr/bin/env python3
"""Train one registry-bound causal-role LoRA run from frozen v2 caches.

Dry-run is standard-library-only.  A real run validates the exact run-spec,
cache manifests and ordered PT inventories before importing the ML backend.
The output is fresh-only and contains one eligible checkpoint at step 200.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
RUN_SPEC_PROTOCOL = "causal_role_erasure_7mechanism_training_run_spec_v2"
TRAINING_PROTOCOL = "wan_causal_role_lora_training_v2"
NULL_PREFLIGHT_PROTOCOL = "wan_causal_role_lora_numeric_aa_preflight_v3"
ARMS = ("matched_control", "V4", "generic_paraphrase", "bystander_token")
ARM_CACHE_NAMES = {
    "matched_control": "matched",
    "V4": "v4",
    "generic_paraphrase": "generic_paraphrase",
    "bystander_token": "bystander",
}
MECHANISMS = (
    "water_impact", "rigid_collision", "brittle_fracture", "powder_impact",
    "elastic_deformation", "material_release", "surface_trace",
)
ERASE_ROWS = 178
PRESERVE_ROWS = 36
BASE_ROWS = 214
STEPS = 200
SEED = 26000
PROMPT_SHAPE = (1, 226, 4096)
LATENT_SHAPE = (1, 16, 13, 60, 104)
DTYPE = "torch.bfloat16"
EXPECTED_INITIAL_LORA_SHA256 = "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8"
EXPECTED_NOISE_RNG_INITIAL_SHA256 = "49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704"
EXPECTED_NOISE_RNG_FINAL_SHA256 = "79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f"
EXPECTED_SCHEDULE_SHA256 = "e006d4d730b807e699a033973537032bad41ea484a5a0808b80e9f9b5136be6e"
RUN_SPEC_REGISTRY_RELATIVE = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2/run_spec_registry.json"
)
EXPECTED_RUN_SPEC_REGISTRY_SHA256 = "36c79a862918e436fec59098f052ac6c11e83ce4833dd74b53fc5842285602ca"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
SEALED = ("final36", "sealed-final", "sealed_final")
GRADIENT_MAX_ABS_TOL = 1e-4
GRADIENT_MAX_PARAMETER_L2_RELATIVE = 0.15
GRADIENT_MAX_GLOBAL_L2_RELATIVE = 0.05

EXPECTED_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "max_steps": STEPS,
    "learning_rate": 5e-5,
    "rank": 16,
    "alpha": 16,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "optimizer": "AdamW",
    "betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "grad_accum": 1,
    "max_grad_norm": 1.0,
    "teacher_weight": 4.0,
    "preserve_weight": 4.0,
    "schedule": "strict_alternating_100_erase_100_preserve_index_shuffle_v2",
    "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    "expected_initial_lora_sha256": EXPECTED_INITIAL_LORA_SHA256,
    "expected_noise_rng_initial_sha256": EXPECTED_NOISE_RNG_INITIAL_SHA256,
    "expected_noise_rng_final_sha256": EXPECTED_NOISE_RNG_FINAL_SHA256,
    "checkpoint_step": 200,
    "inference_lora_scale": 1.25,
    "dtype": DTYPE,
    "device": "cuda",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def reject_sealed(*values: str | Path) -> None:
    for value in values:
        require(not any(token in str(value).casefold() for token in SEALED), f"sealed/final36 path is forbidden: {value}")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")


def resolve_path(project_root: Path, value: str | Path, *, strict: bool = True) -> Path:
    reject_sealed(value)
    path = Path(value)
    path = path if path.is_absolute() else project_root / path
    result = path.resolve(strict=strict)
    reject_sealed(result)
    return result


def inventory_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_record(project_root: Path, value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "row_count"}, f"run-spec {label} fields invalid")
    expected = require_sha256(value["sha256"], f"run-spec {label}.sha256")
    require(type(value["row_count"]) is int and value["row_count"] >= 0, f"run-spec {label}.row_count invalid")
    path = resolve_path(project_root, value["path"])
    regular_file(path, label)
    require(sha256_file(path) == expected, f"run-spec {label} byte hash mismatch")
    return {"path": str(path), "sha256": expected, "row_count": value["row_count"]}


def cache_record(project_root: Path, value: Any, label: str, expected_rows: int) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"path", "manifest_sha256", "ordered_inventory_sha256", "row_count"}, f"run-spec {label} fields invalid")
    require_sha256(value["manifest_sha256"], f"run-spec {label}.manifest")
    require_sha256(value["ordered_inventory_sha256"], f"run-spec {label}.inventory")
    require(value["row_count"] == expected_rows, f"run-spec {label}.row_count must be {expected_rows}")
    path = resolve_path(project_root, value["path"])
    require(path.is_dir() and not path.is_symlink(), f"run-spec {label} path invalid")
    return {**value, "path": str(path)}


def load_run_spec(project_root: Path, path: Path, expected_sha256: str) -> dict[str, Any]:
    require_sha256(expected_sha256, "external run-spec hash")
    regular_file(path, "run-spec")
    require(sha256_file(path) == expected_sha256, "run-spec differs from external expected SHA-256")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol", "protocol_version", "status", "run_id",
        "mechanism", "arm", "base_cache", "teacher_cache", "arm_cache",
        "model_root", "model_inventory", "runtime_root", "runtime_registry",
        "python_executable", "training_config", "output_dir",
        "matched_tensor_equality_receipt",
    }


    require(isinstance(payload, dict) and set(payload) == required, "run-spec fields are not exact")
    require(payload["schema_version"] == 1 and payload["protocol"] == RUN_SPEC_PROTOCOL, "run-spec schema/protocol mismatch")
    require(payload["protocol_version"] == PROTOCOL_VERSION and payload["status"] == "frozen", "run-spec is not frozen")
    require(SAFE_ID.fullmatch(str(payload["run_id"])) is not None, "run_id is unsafe")
    require(payload["mechanism"] in MECHANISMS, "run-spec mechanism invalid")
    require(payload["arm"] in ARMS, "run-spec arm invalid")
    require(payload["training_config"] == EXPECTED_CONFIG, "run-spec training_config differs from unified config")
    base = cache_record(project_root, payload["base_cache"], "base_cache", BASE_ROWS)
    teacher = cache_record(project_root, payload["teacher_cache"], "teacher_cache", ERASE_ROWS)
    arm_cache = cache_record(project_root, payload["arm_cache"], "arm_cache", ERASE_ROWS)
    model_inventory = artifact_record(project_root, payload["model_inventory"], "model_inventory")
    runtime_registry = artifact_record(project_root, payload["runtime_registry"], "runtime_registry")
    receipt_value = payload["matched_tensor_equality_receipt"]
    if payload["arm"] == "matched_control":
        receipt = artifact_record(project_root, receipt_value, "matched_tensor_equality_receipt")
        require(receipt["row_count"] == ERASE_ROWS, "matched receipt must contain 178 rows")
    else:
        require(receipt_value is None, "non-matched run must not carry a matched receipt")
        receipt = None
    output = resolve_path(project_root, payload["output_dir"], strict=False)
    return {
        **payload,
        "project_root": str(project_root), "run_spec_path": str(path),
        "run_spec_sha256": expected_sha256, "base_cache": base,
        "teacher_cache": teacher, "arm_cache": arm_cache,
        "model_root": str(resolve_path(project_root, payload["model_root"], strict=False)),
        "model_inventory": model_inventory,
        "runtime_root": str(resolve_path(project_root, payload["runtime_root"], strict=False)),
        "runtime_registry": runtime_registry,
        "python_executable": str(resolve_path(project_root, payload["python_executable"], strict=False)),
        "output_dir": str(output), "matched_tensor_equality_receipt": receipt,
    }


def validate_frozen_run_spec_registry(
    project_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    path = resolve_path(project_root, RUN_SPEC_REGISTRY_RELATIVE)
    regular_file(path, "frozen run-spec registry")
    require(
        sha256_file(path) == EXPECTED_RUN_SPEC_REGISTRY_SHA256,
        "frozen run-spec registry SHA-256 mismatch",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("status") == "training_run_specs_frozen_after_cache_validation"
        and payload.get("run_spec_count") == 18,
        "frozen run-spec registry status/count mismatch",
    )
    record = payload.get("run_specs", {}).get(spec["run_id"])
    require(isinstance(record, dict), "run is absent from frozen run-spec registry")
    require(
        resolve_path(project_root, record.get("path", ""))
        == Path(str(spec["run_spec_path"])).resolve()
        and record.get("sha256") == spec["run_spec_sha256"],
        "run-spec registry binding mismatch",
    )
    return {
        "path": str(path),
        "sha256": EXPECTED_RUN_SPEC_REGISTRY_SHA256,
        "row_count": 18,
    }


def validate_cache(
    record: Mapping[str, Any], *, mode: str, mechanism: str,
    arm: str | None, verify_payload_bytes: bool = True,
) -> dict[str, Any]:
    root = Path(str(record["path"])); manifest_path = root / "cache_manifest.json"
    regular_file(manifest_path, f"{mode} cache manifest")
    require(sha256_file(manifest_path) == record["manifest_sha256"], f"{mode} cache manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("protocol_version") == PROTOCOL_VERSION, f"{mode} cache protocol mismatch")
    require(manifest.get("mode") == mode and manifest.get("mechanism") == mechanism, f"{mode} cache identity mismatch")
    if arm is not None:
        require(manifest.get("arm") == arm, f"{mode} cache arm mismatch")
    require(manifest.get("row_count") == record["row_count"], f"{mode} cache row count mismatch")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) == record["row_count"], f"{mode} cache files invalid")
    paths = [root / str(item.get("file", "")) for item in files]
    require(len(set(paths)) == len(paths), f"{mode} cache paths are not unique")
    for index, (item, path) in enumerate(zip(files, paths)):
        regular_file(path, f"{mode} cache row {index}")
        require(item.get("index") == index, f"{mode} cache row {index} index mismatch")
        require_sha256(item.get("sha256"), f"{mode} cache row {index}")
        if verify_payload_bytes:
            require(item["sha256"] == sha256_file(path), f"{mode} cache row {index} binding mismatch")
    if verify_payload_bytes:
        require(inventory_sha256(paths) == record["ordered_inventory_sha256"], f"{mode} ordered inventory mismatch")
    require(manifest.get("ordered_inventory_sha256") == record["ordered_inventory_sha256"], f"{mode} manifest inventory mismatch")
    allowed = {manifest_path, root / "cache_plan.json", *paths}
    require(set(root.iterdir()) == allowed, f"{mode} cache directory contains unexpected entries")
    return {"manifest": manifest, "paths": paths, "manifest_path": manifest_path}


def validate_matched_receipt(spec: Mapping[str, Any], base: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any] | None:
    record = spec["matched_tensor_equality_receipt"]
    if record is None:
        return None
    payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    required = {"protocol_version", "status", "mechanism", "arm", "row_count", "base_cache_manifest_sha256", "arm_cache_manifest_sha256", "items"}
    require(isinstance(payload, dict) and set(payload) == required, "matched equality receipt fields invalid")
    require(payload["protocol_version"] == PROTOCOL_VERSION and payload["status"] == "passed", "matched equality receipt not passed")
    require(payload["mechanism"] == spec["mechanism"] and payload["arm"] == "matched_control", "matched receipt identity mismatch")
    require(payload["row_count"] == ERASE_ROWS, "matched receipt row count mismatch")
    require(payload["base_cache_manifest_sha256"] == spec["base_cache"]["manifest_sha256"], "matched receipt base cache mismatch")
    require(payload["arm_cache_manifest_sha256"] == spec["arm_cache"]["manifest_sha256"], "matched receipt arm cache mismatch")
    items = payload["items"]
    require(isinstance(items, list) and len(items) == ERASE_ROWS, "matched receipt items invalid")
    for index, (item, base_file, arm_file) in enumerate(zip(items, base["manifest"]["files"][:ERASE_ROWS], arm["manifest"]["files"])):
        require(
            isinstance(item, dict)
            and item == {
                "selected_index": index,
                "candidate_id": base_file["row_id"],
                "base_prompt_embeds_sha256": base_file["prompt_embeds_sha256"],
                "arm_prompt_embeds_sha256": arm_file["prompt_embeds_sha256"],
                "tensor_equal": True,
            },
            f"matched receipt row {index} mismatch",
        )
        require(item["base_prompt_embeds_sha256"] == item["arm_prompt_embeds_sha256"], f"matched receipt row {index} tensors differ")
    return payload


def validate_cache_stack(
    spec: Mapping[str, Any],
    base: Mapping[str, Any],
    teacher: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> None:
    manifests = [base["manifest"], teacher["manifest"], arm["manifest"]]
    expected_model = spec["model_inventory"]["sha256"]
    expected_runtime = spec["runtime_registry"]["sha256"]
    for label, manifest in zip(("base", "teacher", "arm"), manifests):
        require(
            manifest.get("model_inventory", {}).get("sha256") == expected_model,
            f"{label} cache model inventory differs from run-spec",
        )
        require(
            manifest.get("runtime_registry", {}).get("sha256") == expected_runtime,
            f"{label} cache runtime registry differs from run-spec",
        )
    selected_hashes = {
        manifest.get("selected178", {}).get("sha256") for manifest in manifests
    }
    require(len(selected_hashes) == 1 and None not in selected_hashes, "cache stack selected178 binding mismatch")
    base_ids = [item.get("row_id") for item in base["manifest"]["files"][:ERASE_ROWS]]
    require(
        base_ids == [item.get("row_id") for item in teacher["manifest"]["files"]]
        == [item.get("row_id") for item in arm["manifest"]["files"]],
        "cache stack candidate row order differs",
    )


def build_schedule() -> list[tuple[str, int]]:
    rng = random.Random(SEED)
    pools = {"erase": list(range(ERASE_ROWS)), "preserve": list(range(PRESERVE_ROWS))}
    for pool in pools.values(): rng.shuffle(pool)
    cursors = {"erase": 0, "preserve": 0}; result: list[tuple[str, int]] = []
    for step in range(1, STEPS + 1):
        role = "erase" if step % 2 else "preserve"
        if cursors[role] >= len(pools[role]):
            rng.shuffle(pools[role]); cursors[role] = 0
        index = pools[role][cursors[role]]; cursors[role] += 1
        result.append((role, index))
    return result


def schedule_sha256(schedule: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for step, (role, index) in enumerate(schedule, 1): digest.update(f"{step}:{role}:{index}\n".encode())
    return digest.hexdigest()


def build_plan(
    spec: Mapping[str, Any], *, dry_run: bool,
    run_spec_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schedule = build_schedule()
    require(Counter(role for role, _ in schedule) == Counter({"erase": 100, "preserve": 100}), "schedule role counts invalid")
    require(all(role == ("erase" if step % 2 else "preserve") for step, (role, _) in enumerate(schedule, 1)), "schedule does not strictly alternate")
    digest = schedule_sha256(schedule); require(digest == EXPECTED_SCHEDULE_SHA256, "schedule digest differs from unified config")
    return {
        "schema_version": 1, "protocol": TRAINING_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION, "status": "planned" if dry_run else "reserved_for_run",
        "dry_run": dry_run, "run_id": spec["run_id"], "mechanism": spec["mechanism"],
        "arm": spec["arm"], "output_dir": spec["output_dir"],
        "run_spec": {"path": spec["run_spec_path"], "sha256": spec["run_spec_sha256"]},
        "base_cache": spec["base_cache"], "teacher_cache": spec["teacher_cache"],
        "arm_cache": spec["arm_cache"], "model_root": spec["model_root"],
        "model_inventory": spec["model_inventory"], "runtime_root": spec["runtime_root"],
        "runtime_registry": spec["runtime_registry"], "python_executable": spec["python_executable"],
        "matched_tensor_equality_receipt": spec["matched_tensor_equality_receipt"],
        "training_config": EXPECTED_CONFIG, "schedule_sha256": digest,
        "role_step_counts": {"erase": 100, "preserve": 100},
        "null_preflight": "numeric_forward_loss_exact_numeric_gradient_l2_bounded_A_A_v3",
        "cache_validation_basis": {
            "mode": (
                "full_per_file_hash_revalidation"
                if run_spec_registry is None
                else "frozen_run_spec_registry_plus_live_tensor_contract"
            ),
            "run_spec_registry": run_spec_registry,
        },
        "checkpoint_policy": "only_checkpoint_000200",
    }


def validate_live_model_runtime(
    spec: Mapping[str, Any], *, verify_model_bytes: bool = True
) -> None:
    # Reuse the generic cache preparer's inventory algorithms.  That module is
    # standard-library-only until live=True reaches its explicit torch check.
    import prepare_causal_role_erasure_7mechanism_training_cache_v2 as contract
    binding = {
        "project_root": spec["project_root"],
        "model_root": spec["model_root"],
        "model_inventory": spec["model_inventory"],
        "runtime_root": spec["runtime_root"],
        "runtime_registry": spec["runtime_registry"],
        "python_executable": spec["python_executable"],
    }
    contract.validate_model_inventory(binding, live=verify_model_bytes)
    contract.validate_runtime(binding, live=True)


def write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"): flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())


def reserve_output(output: Path, plan: Mapping[str, Any]) -> None:
    require(not output.exists() and not output.is_symlink(), f"output directory must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True); require(not output.parent.is_symlink(), "output parent is symlinked")
    os.mkdir(output, 0o755); write_exclusive(output / "run_plan.json", canonical_json_bytes(plan))
    if not plan["dry_run"]: write_exclusive(output / ".run_reservation", b"training in progress\n")


class TrainingBackend(Protocol):
    def load_payload(self, path: Path) -> Mapping[str, Any]: ...
    def validate_base_payload(self, payload: Mapping[str, Any], role: str, row_id: str) -> None: ...
    def validate_prompt_payload(self, payload: Mapping[str, Any], key: str, candidate_id: str) -> None: ...
    def initialize(self) -> str: ...
    def null_preflight(self, base: Mapping[str, Any], teacher: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]: ...
    def begin_training(self) -> str: ...
    def train_step(self, role: str, base: Mapping[str, Any], teacher: Mapping[str, Any] | None, arm: Mapping[str, Any] | None) -> dict[str, float]: ...
    def noise_rng_final_sha256(self) -> str: ...
    def finite_receipt(self) -> dict[str, Any]: ...
    def save_checkpoint(self, path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


def validate_payloads(backend: TrainingBackend, base: Mapping[str, Any], teacher: Mapping[str, Any], arm: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    base_payloads = []
    for index, path in enumerate(base["paths"]):
        payload = backend.load_payload(path); role = "erase" if index < ERASE_ROWS else "preserve"
        row_id = base["manifest"]["files"][index]["row_id"]
        backend.validate_base_payload(payload, role, row_id); base_payloads.append(payload)
    teacher_payloads = []; arm_payloads = []
    for index in range(ERASE_ROWS):
        candidate_id = base["manifest"]["files"][index]["row_id"]
        t = backend.load_payload(teacher["paths"][index]); a = backend.load_payload(arm["paths"][index])
        backend.validate_prompt_payload(t, "teacher_prompt_embeds", candidate_id)
        backend.validate_prompt_payload(a, "student_prompt_embeds", candidate_id)
        teacher_payloads.append(t); arm_payloads.append(a)
    return base_payloads, teacher_payloads, arm_payloads


def run_training(
    plan: Mapping[str, Any], spec: Mapping[str, Any], backend: TrainingBackend,
    base: Mapping[str, Any], teacher: Mapping[str, Any], arm: Mapping[str, Any],
    *, frozen_revalidator: Callable[[], None] | None = None,
) -> dict[str, Any]:
    output = Path(str(plan["output_dir"])); base_p, teacher_p, arm_p = validate_payloads(backend, base, teacher, arm)
    initial = backend.initialize(); require(initial == EXPECTED_INITIAL_LORA_SHA256, "initial LoRA digest mismatch")
    preflight = backend.null_preflight(base_p[0], teacher_p[0], arm_p[0])
    require(preflight.get("status") == "passed", "numeric A-A null preflight failed")
    require(preflight.get("forward_exact") is True, "numeric A-A forward equality failed")
    require(preflight.get("loss_exact") is True, "numeric A-A loss equality failed")
    require(
        isinstance(preflight.get("gradient_comparison"), str)
        and preflight["gradient_comparison"].startswith("per_parameter_numeric_l2_relative"),
        "numeric A-A per-parameter gradient comparison missing",
    )
    require(type(preflight.get("gradient_parameter_count")) is int and preflight["gradient_parameter_count"] > 0, "numeric A-A gradient inventory empty")
    require(preflight.get("legacy_byte_gradient_gate_used") is False, "legacy byte-gradient gate is forbidden")
    preflight_payload = {
        "protocol": NULL_PREFLIGHT_PROTOCOL, "protocol_version": PROTOCOL_VERSION,
        "run_spec_sha256": spec["run_spec_sha256"], **preflight,
    }
    write_exclusive(output / "null_preflight.json", canonical_json_bytes(preflight_payload))
    rng_initial = backend.begin_training(); require(rng_initial == EXPECTED_NOISE_RNG_INITIAL_SHA256, "initial noise RNG digest mismatch")
    schedule = build_schedule(); losses: list[float] = []; role_counts = Counter()
    for step, (role, index) in enumerate(schedule, 1):
        role_counts[role] += 1
        metrics = backend.train_step(
            role, base_p[index if role == "erase" else ERASE_ROWS + index],
            teacher_p[index] if role == "erase" else None,
            arm_p[index] if role == "erase" else None,
        )
        require(set(metrics) >= {"loss", "flow_loss", "teacher_loss", "preserve_loss", "grad_norm"}, f"step {step}: metrics incomplete")
        require(all(math.isfinite(float(value)) for value in metrics.values()), f"step {step}: non-finite metric")
        losses.append(float(metrics["loss"]))
    require(role_counts == Counter({"erase": 100, "preserve": 100}), "realized role counts invalid")
    rng_final = backend.noise_rng_final_sha256(); require(rng_final == EXPECTED_NOISE_RNG_FINAL_SHA256, "final noise RNG digest mismatch")
    if frozen_revalidator is not None:
        frozen_revalidator()
    finite = backend.finite_receipt(); require(finite.get("status") == "passed", "final LoRA is non-finite")
    require(finite.get("nonfinite_tensor_count") == 0, "final LoRA finite receipt reports non-finite tensors")
    require_sha256(finite.get("trainable_state_sha256"), "final trainable state")
    checkpoint = output / "checkpoint-000200"
    metadata = {
        "protocol": TRAINING_PROTOCOL, "protocol_version": PROTOCOL_VERSION,
        "status": "eligible_training_complete", "run_id": plan["run_id"],
        "mechanism": plan["mechanism"], "arm": plan["arm"], "step": 200,
        "run_spec_sha256": spec["run_spec_sha256"], "training_config": EXPECTED_CONFIG,
        "cache_validation_basis": plan["cache_validation_basis"],
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256, "role_step_counts": dict(role_counts),
        "initial_lora_sha256": initial, "noise_rng_initial_sha256": rng_initial,
        "noise_rng_final_sha256": rng_final, "mean_loss_last20": sum(losses[-20:]) / 20,
        "null_preflight_sha256": sha256_file(output / "null_preflight.json"),
        "finite_receipt": finite,
    }
    checkpoint_record = backend.save_checkpoint(checkpoint, metadata)
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), "checkpoint-000200 was not created")
    require(not list(output.glob("checkpoint-*")) or list(output.glob("checkpoint-*")) == [checkpoint], "only checkpoint-000200 is allowed")
    require(
        isinstance(checkpoint_record, dict)
        and set(checkpoint_record) == {"path", "weights_sha256", "training_state_sha256"},
        "checkpoint record fields invalid",
    )
    require(Path(checkpoint_record["path"]).resolve() == checkpoint.resolve(), "checkpoint record path mismatch")
    weights_path = checkpoint / "pytorch_lora_weights.safetensors"
    state_path = checkpoint / "training_state.json"
    regular_file(weights_path, "checkpoint weights"); regular_file(state_path, "checkpoint training state")
    require(checkpoint_record["weights_sha256"] == sha256_file(weights_path), "checkpoint weights hash mismatch")
    require(checkpoint_record["training_state_sha256"] == sha256_file(state_path), "checkpoint state hash mismatch")
    receipt = {
        **metadata, "status": "eligible", "checkpoint": checkpoint_record,
        "base_cache_manifest_sha256": spec["base_cache"]["manifest_sha256"],
        "teacher_cache_manifest_sha256": spec["teacher_cache"]["manifest_sha256"],
        "arm_cache_manifest_sha256": spec["arm_cache"]["manifest_sha256"],
        "inference_lora_scale": 1.25,
    }
    write_exclusive(output / "run_receipt.json", canonical_json_bytes(receipt)); (output / ".run_reservation").unlink()
    require(
        set(output.iterdir())
        == {
            output / "run_plan.json", output / "null_preflight.json",
            output / "run_receipt.json", checkpoint,
        },
        "final run output inventory contains unexpected entries",
    )
    return receipt


class RealBackend:
    def __init__(self, spec: Mapping[str, Any]):
        import torch
        self.torch = torch; self.spec = spec; self.device = torch.device("cuda")
        self.transformer = None; self.optimizer = None; self.generator = None

    def load_payload(self, path: Path) -> Mapping[str, Any]: return self.torch.load(path, map_location="cpu", weights_only=True)
    def _tensor(self, payload: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> Any:
        value = payload.get(key); require(isinstance(value, self.torch.Tensor), f"payload missing {key}")
        require(tuple(value.shape) == shape and value.dtype == self.torch.bfloat16, f"payload {key} tensor contract mismatch")
        require(bool(self.torch.isfinite(value.float()).all()), f"payload {key} is non-finite"); return value
    def validate_base_payload(self, payload: Mapping[str, Any], role: str, row_id: str) -> None:
        require(payload.get("training_role") == role and payload.get("row_id") == row_id, "base payload identity mismatch")
        self._tensor(payload, "latents", LATENT_SHAPE); self._tensor(payload, "prompt_embeds", PROMPT_SHAPE)
    def validate_prompt_payload(self, payload: Mapping[str, Any], key: str, candidate_id: str) -> None:
        require(payload.get("candidate_id") == candidate_id, "prompt payload candidate mismatch"); self._tensor(payload, key, PROMPT_SHAPE)
    def _state_sha(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.transformer.named_parameters()):
            if parameter.requires_grad:
                value = parameter.detach().contiguous().cpu(); digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode()); digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    def _tensor_sha(self, value: Any) -> str:
        value = value.detach().contiguous().cpu(); digest = hashlib.sha256(); digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode()); digest.update(value.view(self.torch.uint8).numpy().tobytes()); return digest.hexdigest()
    def initialize(self) -> str:
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig
        self.torch.manual_seed(SEED); self.torch.cuda.manual_seed_all(SEED)
        self.transformer = WanTransformer3DModel.from_pretrained(self.spec["model_root"], subfolder="transformer", torch_dtype=self.torch.bfloat16).to(self.device)
        self.transformer.requires_grad_(False); self.transformer.enable_gradient_checkpointing()
        self.torch.manual_seed(SEED); self.torch.cuda.manual_seed_all(SEED)
        self.transformer.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian", target_modules=EXPECTED_CONFIG["target_modules"]))
        self.transformer.train(); return self._state_sha()
    def _forward_pair(self, noisy: Any, timestep: Any, student_prompt: Any, teacher_prompt: Any) -> tuple[Any, Any]:
        self.transformer.disable_adapters()
        try:
            with self.torch.no_grad(): teacher = self.transformer(hidden_states=noisy, timestep=timestep, encoder_hidden_states=teacher_prompt, return_dict=False)[0].detach()
        finally: self.transformer.enable_adapters()
        student = self.transformer(hidden_states=noisy, timestep=timestep, encoder_hidden_states=student_prompt, return_dict=False)[0]
        return teacher, student
    def _erase_forward(self, base: Mapping[str, Any], teacher: Mapping[str, Any], arm: Mapping[str, Any], generator: Any) -> tuple[Any, dict[str, Any]]:
        clean = base["latents"].to(self.device, dtype=self.torch.bfloat16); noise = self.torch.randn(clean.shape, generator=generator, dtype=self.torch.float32).to(self.device, dtype=self.torch.bfloat16)
        sigma = self.torch.rand((clean.shape[0],), generator=generator, dtype=self.torch.float32).to(self.device).view(-1,1,1,1,1)
        noisy = ((1-sigma)*clean + sigma*noise).to(self.torch.bfloat16); target=(noise-clean).to(self.torch.bfloat16); timestep=(sigma.flatten()*1000).to(self.torch.bfloat16)
        frozen, student = self._forward_pair(noisy, timestep, arm["student_prompt_embeds"].to(self.device), teacher["teacher_prompt_embeds"].to(self.device))
        flow=self.torch.nn.functional.mse_loss(student.float(),target.float()); distill=self.torch.nn.functional.mse_loss(student.float(),frozen.float()); loss=flow+4*distill
        return loss, {"flow":flow,"teacher":distill,"student":student,"frozen":frozen}
    def null_preflight(self, base: Mapping[str, Any], teacher: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
        observations=[]; gradients=[]
        for _ in range(2):
            generator=self.torch.Generator(device="cpu").manual_seed(SEED+991)
            self.transformer.zero_grad(set_to_none=True); loss, values=self._erase_forward(base,teacher,arm,generator); loss.backward()
            observations.append({"loss":float(loss.detach()),"flow":float(values["flow"].detach()),"teacher":float(values["teacher"].detach()),"student":values["student"].detach().cpu(),"frozen":values["frozen"].detach().cpu()})
            gradients.append({name:param.grad.detach().float().cpu().clone() for name,param in self.transformer.named_parameters() if param.requires_grad})
        self.transformer.zero_grad(set_to_none=True)
        require(self.torch.equal(observations[0]["student"], observations[1]["student"]) and self.torch.equal(observations[0]["frozen"], observations[1]["frozen"]), "A-A forward tensors differ")
        require(all(observations[0][key] == observations[1][key] for key in ("loss","flow","teacher")), "A-A losses differ")
        require(set(gradients[0]) == set(gradients[1]) and gradients[0], "A-A gradient inventory invalid")
        max_abs=0.0; max_parameter_l2_relative=0.0
        global_difference_squared=0.0; global_reference_squared=0.0
        for name in gradients[0]:
            left,right=gradients[0][name].double(),gradients[1][name].double(); require(bool(self.torch.isfinite(left).all() and self.torch.isfinite(right).all()), f"non-finite gradient {name}")
            difference=left-right; difference_norm=float(self.torch.linalg.vector_norm(difference)); reference_norm=max(float(self.torch.linalg.vector_norm(left)),float(self.torch.linalg.vector_norm(right)),1e-30); parameter_relative=difference_norm/reference_norm; parameter_max_abs=float(difference.abs().max())
            require(parameter_relative <= GRADIENT_MAX_PARAMETER_L2_RELATIVE, f"numeric gradient L2 mismatch {name}"); require(parameter_max_abs <= GRADIENT_MAX_ABS_TOL, f"numeric gradient absolute mismatch {name}")
            max_parameter_l2_relative=max(max_parameter_l2_relative,parameter_relative); max_abs=max(max_abs,parameter_max_abs); global_difference_squared+=difference_norm*difference_norm; global_reference_squared+=reference_norm*reference_norm
        global_l2_relative=math.sqrt(global_difference_squared/global_reference_squared); require(global_l2_relative <= GRADIENT_MAX_GLOBAL_L2_RELATIVE,"numeric global gradient L2 mismatch")
        return {"status":"passed","forward_exact":True,"loss_exact":True,"gradient_comparison":"per_parameter_numeric_l2_relative_le_0p15_max_abs_le_1e-4_global_l2_relative_le_0p05","gradient_parameter_count":len(gradients[0]),"gradient_max_abs_difference":max_abs,"gradient_max_parameter_l2_relative":max_parameter_l2_relative,"gradient_global_l2_relative":global_l2_relative,"legacy_byte_gradient_gate_used":False}
    def begin_training(self) -> str:
        trainable=[parameter for parameter in self.transformer.parameters() if parameter.requires_grad]
        self.optimizer=self.torch.optim.AdamW(trainable,lr=5e-5,betas=(0.9,0.999),weight_decay=0.01); self.generator=self.torch.Generator(device="cpu").manual_seed(SEED); return self._tensor_sha(self.generator.get_state())
    def train_step(self, role: str, base: Mapping[str, Any], teacher: Mapping[str, Any] | None, arm: Mapping[str, Any] | None) -> dict[str,float]:
        clean=base["latents"].to(self.device,dtype=self.torch.bfloat16); noise=self.torch.randn(clean.shape,generator=self.generator,dtype=self.torch.float32).to(self.device,dtype=self.torch.bfloat16); sigma=self.torch.rand((clean.shape[0],),generator=self.generator,dtype=self.torch.float32).to(self.device).view(-1,1,1,1,1); noisy=((1-sigma)*clean+sigma*noise).to(self.torch.bfloat16); timestep=(sigma.flatten()*1000).to(self.torch.bfloat16)
        flow=distill=preserve=self.torch.zeros((),device=self.device)
        if role=="erase":
            target=(noise-clean).to(self.torch.bfloat16); frozen,student=self._forward_pair(noisy,timestep,arm["student_prompt_embeds"].to(self.device),teacher["teacher_prompt_embeds"].to(self.device)); flow=self.torch.nn.functional.mse_loss(student.float(),target.float()); distill=self.torch.nn.functional.mse_loss(student.float(),frozen.float()); loss=flow+4*distill
        else:
            prompt=base["prompt_embeds"].to(self.device); self.transformer.disable_adapters()
            try:
                with self.torch.no_grad(): frozen=self.transformer(hidden_states=noisy,timestep=timestep,encoder_hidden_states=prompt,return_dict=False)[0].detach()
            finally:self.transformer.enable_adapters()
            student=self.transformer(hidden_states=noisy,timestep=timestep,encoder_hidden_states=prompt,return_dict=False)[0]; preserve=self.torch.nn.functional.mse_loss(student.float(),frozen.float()); loss=4*preserve
        require(bool(self.torch.isfinite(loss)),"non-finite loss"); self.optimizer.zero_grad(set_to_none=True); loss.backward(); grad=self.torch.nn.utils.clip_grad_norm_([p for p in self.transformer.parameters() if p.requires_grad],1.0); require(math.isfinite(float(grad)),"non-finite grad norm"); self.optimizer.step()
        return {"loss":float(loss.detach()),"flow_loss":float(flow.detach()),"teacher_loss":float(distill.detach()),"preserve_loss":float(preserve.detach()),"grad_norm":float(grad)}
    def noise_rng_final_sha256(self) -> str:return self._tensor_sha(self.generator.get_state())
    def finite_receipt(self) -> dict[str,Any]:
        tensors=[(name,p) for name,p in self.transformer.named_parameters() if p.requires_grad]; nonfinite=sum(int(not bool(self.torch.isfinite(p).all())) for _,p in tensors); return {"status":"passed" if nonfinite==0 else "failed","tensor_count":len(tensors),"nonfinite_tensor_count":nonfinite,"trainable_state_sha256":self._state_sha()}
    def save_checkpoint(self,path:Path,metadata:Mapping[str,Any])->dict[str,Any]:
        from diffusers import WanPipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import get_peft_model_state_dict
        path.mkdir(parents=False,exist_ok=False); WanPipeline.save_lora_weights(str(path),transformer_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(self.transformer)),safe_serialization=True); state=path/"training_state.json"; write_exclusive(state,canonical_json_bytes(metadata)); weights=path/"pytorch_lora_weights.safetensors"; require(set(path.iterdir())=={state,weights},"checkpoint inventory invalid"); return {"path":str(path),"weights_sha256":sha256_file(weights),"training_state_sha256":sha256_file(state)}
    def close(self)->None:
        self.transformer=self.optimizer=self.generator=None
        if hasattr(self,"torch") and self.torch.cuda.is_available():self.torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False); parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument("--run-spec",type=Path,required=True); parser.add_argument("--run-spec-sha256",required=True); action=parser.add_mutually_exclusive_group(required=True); action.add_argument("--dry-run",action="store_true"); action.add_argument("--run",action="store_true"); return parser


def main(argv:Sequence[str]|None=None)->int:
    parser=build_parser();args=parser.parse_args(argv)
    try:
        reject_sealed(args.project_root,args.run_spec);project=args.project_root.resolve(strict=True);spec_path=resolve_path(project,args.run_spec);spec=load_run_spec(project,spec_path,args.run_spec_sha256)
        trusted_registry=None if args.dry_run else validate_frozen_run_spec_registry(project,spec)
        verify_payload_bytes=args.dry_run
        base=validate_cache(spec["base_cache"],mode="prepare-base",mechanism=spec["mechanism"],arm=None,verify_payload_bytes=verify_payload_bytes);teacher=validate_cache(spec["teacher_cache"],mode="prepare-teacher",mechanism=spec["mechanism"],arm=None,verify_payload_bytes=verify_payload_bytes);arm=validate_cache(spec["arm_cache"],mode="prepare-arm",mechanism=spec["mechanism"],arm=ARM_CACHE_NAMES[spec["arm"]],verify_payload_bytes=verify_payload_bytes);validate_cache_stack(spec,base,teacher,arm);validate_matched_receipt(spec,base,arm)
        plan=build_plan(spec,dry_run=args.dry_run,run_spec_registry=trusted_registry);output=Path(spec["output_dir"])
        if args.dry_run:reserve_output(output,plan);print(f"Planned training run {spec['run_id']} at {output}");return 0
        require(Path(sys.executable).resolve()==Path(spec["python_executable"]).resolve(),"run must use registered Python")
        validate_live_model_runtime(spec,verify_model_bytes=False)
        backend=RealBackend(spec)
        try:
            # Tensor payload validation precedes model initialization inside run_training.
            def revalidate_frozen() -> None:
                reopened = load_run_spec(project, spec_path, args.run_spec_sha256)
                validate_frozen_run_spec_registry(project,reopened)
                checked_base = validate_cache(reopened["base_cache"], mode="prepare-base", mechanism=reopened["mechanism"], arm=None,verify_payload_bytes=False)
                checked_teacher = validate_cache(reopened["teacher_cache"], mode="prepare-teacher", mechanism=reopened["mechanism"], arm=None,verify_payload_bytes=False)
                checked_arm = validate_cache(reopened["arm_cache"], mode="prepare-arm", mechanism=reopened["mechanism"], arm=ARM_CACHE_NAMES[reopened["arm"]],verify_payload_bytes=False)
                validate_cache_stack(reopened, checked_base, checked_teacher, checked_arm)
                validate_matched_receipt(reopened, checked_base, checked_arm)
                validate_live_model_runtime(reopened,verify_model_bytes=False)
            reserve_output(output,plan);receipt=run_training(plan,spec,backend,base,teacher,arm,frozen_revalidator=revalidate_frozen)
        finally:backend.close()
        require(sha256_file(spec_path)==spec["run_spec_sha256"],"run-spec changed during training");print(f"Completed {spec['run_id']}: {receipt['checkpoint']['path']}");return 0
    except (OSError,ValueError,RuntimeError,ImportError,json.JSONDecodeError) as exc:print(f"ERROR: {exc}",file=sys.stderr);return 1


if __name__=="__main__":raise SystemExit(main())
