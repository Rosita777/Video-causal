#!/usr/bin/env python3
"""Fail-closed generation runner for the 684 trained-Wan formal outputs.

The runner binds the frozen 294-case manifest, 48-case identification subset,
18 frozen training run specs, and eligible step-200 receipts before launching
one fresh Wan generation job per checkpoint.  A stopped run may continue only
when every existing job is already fully validated and marked completed;
partial, running, or failed jobs are never retried in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_ID = "causal_role_erasure_7m_single_seed_v2"
RUNNER_ID = "causal_role_erasure_7mechanism_trained_wan_eval_v2"
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
MAIN_ARMS = ("matched_control", "V4")
IDENTIFICATION_ARMS = ("generic_paraphrase", "bystander_token")
IDENTIFICATION_MECHANISMS = ("water_impact", "brittle_fracture")

FORMAL_ROWS = 294
IDENTIFICATION_ROWS = 48
MAIN_JOB_ROWS = 42
IDENTIFICATION_JOB_ROWS = 24
JOB_COUNT = 18
EXPECTED_VIDEOS = 684

STEPS = 25
GUIDANCE_SCALE = 5.0
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
DTYPE = "bf16"
LORA_SCALE = 1.25

EXPECTED_FORMAL_SHA256 = "555ec79a1f084a8b75c802cd570801faf98531fc063fa0eb5170ab8c1d710a04"
EXPECTED_IDENTIFICATION_SHA256 = "07e102ec70f3957b7d6cface7826ccdde428c4a436560a6cc3631c01bbd6c897"
EXPECTED_RUN_MATRIX_SHA256 = "0fb9427d987041ea3dbac6eaeb1a5cf49760b774738e31c371f1fe3b19fc59a4"
EXPECTED_RUN_SPEC_REGISTRY_SHA256 = "36c79a862918e436fec59098f052ac6c11e83ce4833dd74b53fc5842285602ca"
EXPECTED_MODEL_REGISTRY_SHA256 = "51e7199b99ee206934924ee043bd01b40ba413dfb60ef1e72682d36a10b46290"
EXPECTED_RUNTIME_REGISTRY_SHA256 = "9043adf7f823022b20711267ab9e28b9dfb72452273d26e27c131b92e65eff01"
EXPECTED_GENERATOR_SHA256 = "04bc6b8a8f93d885137c6157509b00f46824231560989f7b7f78192b26469b5e"

DEFAULT_FORMAL = Path("data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv")
DEFAULT_IDENTIFICATION = Path(
    "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv"
)
DEFAULT_RUN_MATRIX = Path("data/causal_role_erasure_7mechanism_main_v2/run_matrix.csv")
DEFAULT_RUN_SPEC_REGISTRY = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/"
    "training_run_specs_v2/run_spec_registry.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2"
)
DEFAULT_GENERATOR = Path("scripts/generate_wan_clean.py")
DEFAULT_PYTHON = Path("models/.wan-runtime/bin/python")
DEFAULT_MODEL = Path("models/Wan2.1-T2V-1.3B-Diffusers")

PYAV_PROBE_CODE = r"""
import av, json, sys
container = av.open(sys.argv[1])
video = [s for s in container.streams if s.type == "video"]
audio = [s for s in container.streams if s.type == "audio"]
if len(video) != 1:
    raise SystemExit("expected exactly one video stream")
stream = video[0]
frames = list(container.decode(stream))
rate = stream.average_rate
print(json.dumps({
    "video_streams": len(video),
    "audio_streams": len(audio),
    "decoded_frames": len(frames),
    "fps": None if rate is None else str(rate),
    "width": stream.width,
    "height": stream.height,
}))
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def real_directory(path: Path, label: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    regular_file(path, path.name)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{path}: missing CSV header")
        return [dict(row) for row in reader]


def resolve(project_root: Path, value: Path, *, must_exist: bool = True) -> Path:
    path = value if value.is_absolute() else project_root / value
    return path.resolve(strict=must_exist)


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must contain four integers") from exc
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("--gpus requires exactly four distinct non-negative GPUs")
    return gpus


def validate_formal_rows(rows: Sequence[Mapping[str, str]]) -> None:
    require(len(rows) == FORMAL_ROWS, "formal case count must be 294")
    require([int(row["global_case_index"]) for row in rows] == list(range(FORMAL_ROWS)), "formal indices are not exact")
    require(len({row["case_id"] for row in rows}) == FORMAL_ROWS, "formal case IDs repeat")
    require(len({int(row["seed"]) for row in rows}) == FORMAL_ROWS, "formal seeds repeat")
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        block = [row for row in rows if row["mechanism"] == mechanism]
        require(len(block) == MAIN_JOB_ROWS, f"{mechanism}: expected 42 formal rows")
        require({int(row["mechanism_index"]) for row in block} == {mechanism_index}, f"{mechanism}: mechanism index mismatch")
        require(Counter(row["case_kind"] for row in block) == {"causal": 24, "specificity": 18}, f"{mechanism}: 24/18 split mismatch")
        for row in block:
            require(row["protocol_version"] == PROTOCOL_ID, "formal protocol mismatch")
            require((int(row["num_frames"]), int(row["fps"]), int(row["height"]), int(row["width"])) == (49, 8, 480, 832), f"{row['case_id']}: media contract mismatch")
            for field in ("prompt", "source_object", "expected_counterfactual_state"):
                value = str(row[field]).strip()
                require(value and "|" not in value and "\n" not in value, f"{row['case_id']}: invalid {field}")


def validate_identification_rows(
    rows: Sequence[Mapping[str, str]], formal_rows: Sequence[Mapping[str, str]]
) -> dict[tuple[str, str], Mapping[str, str]]:
    require(len(rows) == IDENTIFICATION_ROWS, "identification subset count must be 48")
    by_case = {(row["mechanism"], row["case_id"]): row for row in formal_rows}
    require(len(by_case) == FORMAL_ROWS, "formal join key repeats")
    joined: dict[tuple[str, str], Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        require(int(row["identification_case_index"]) == index, "identification indices are not exact")
        require(row["protocol_version"] == PROTOCOL_ID, "identification protocol mismatch")
        require(row["mechanism"] in IDENTIFICATION_MECHANISMS, "identification mechanism mismatch")
        require(row["included_streams"] == "matched_control,V4", "included-stream binding mismatch")
        require(row["additional_streams"] == "generic_paraphrase,bystander_token", "additional-stream binding mismatch")
        key = (row["mechanism"], row["case_id"])
        require(key in by_case and key not in joined, "identification/formal join mismatch")
        formal = by_case[key]
        for field in (
            "case_kind", "generalization_group", "source_membership", "prompt_style",
            "footprint_lexicalization", "specificity_subtype", "seed",
        ):
            require(row[field] == formal[field], f"{row['case_id']}: identification {field} mismatch")
        joined[key] = formal
    require(Counter(row["mechanism"] for row in rows) == {"water_impact": 24, "brittle_fracture": 24}, "identification balance mismatch")
    return joined


def validate_run_matrix(rows: Sequence[Mapping[str, str]]) -> None:
    require(len(rows) == JOB_COUNT, "run matrix must contain 18 rows")
    require([int(row["run_index"]) for row in rows] == list(range(JOB_COUNT)), "run indices are not exact")
    require(len({row["run_id"] for row in rows}) == JOB_COUNT, "run IDs repeat")
    require(Counter(row["arm"] for row in rows) == {"matched_control": 7, "V4": 7, "generic_paraphrase": 2, "bystander_token": 2}, "run arm inventory mismatch")
    for row in rows:
        require(row["protocol_version"] == PROTOCOL_ID, "run-matrix protocol mismatch")
        require(row["mechanism"] in MECHANISMS, "run-matrix mechanism mismatch")
        expected_group = "main" if row["arm"] in MAIN_ARMS else "identification"
        require(row["run_group"] == expected_group, f"{row['run_id']}: run group mismatch")
        require((int(row["checkpoint"]), float(row["inference_scale"])) == (200, LORA_SCALE), f"{row['run_id']}: inference contract mismatch")


def validate_checkpoint(
    project_root: Path,
    matrix_row: Mapping[str, str],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = matrix_row["run_id"]
    refs = registry.get("run_specs")
    require(isinstance(refs, dict), "run-spec registry malformed")
    require(run_id in refs and isinstance(refs[run_id], dict), f"{run_id}: run spec missing")
    ref = refs[run_id]
    spec_path = resolve(project_root, Path(str(ref.get("path", ""))))
    regular_file(spec_path, f"{run_id} run spec")
    spec_sha = sha256_file(spec_path)
    require(spec_sha == ref.get("sha256"), f"{run_id}: run-spec SHA mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for field in ("run_id", "mechanism", "arm", "output_dir"):
        require(str(spec.get(field)) == matrix_row[field], f"{run_id}: run-spec {field} mismatch")
    require(spec.get("protocol_version") == PROTOCOL_ID and spec.get("status") == "frozen", f"{run_id}: run spec not frozen")
    require(spec.get("model_root") == str(DEFAULT_MODEL), f"{run_id}: model root mismatch")
    require(spec.get("python_executable") == str(DEFAULT_PYTHON), f"{run_id}: runtime path mismatch")
    require(spec.get("training_config", {}).get("checkpoint_step") == 200, f"{run_id}: checkpoint step mismatch")
    require(spec.get("training_config", {}).get("inference_lora_scale") == LORA_SCALE, f"{run_id}: LoRA scale mismatch")
    for field, expected in (("model_inventory", EXPECTED_MODEL_REGISTRY_SHA256), ("runtime_registry", EXPECTED_RUNTIME_REGISTRY_SHA256)):
        binding = spec.get(field)
        require(isinstance(binding, dict) and binding.get("sha256") == expected, f"{run_id}: {field} binding mismatch")
        path = resolve(project_root, Path(str(binding.get("path", ""))))
        regular_file(path, f"{run_id} {field}")
        require(sha256_file(path) == expected, f"{run_id}: {field} bytes changed")

    training_dir = resolve(project_root, Path(matrix_row["output_dir"]))
    receipt_path = training_dir / "run_receipt.json"
    regular_file(receipt_path, f"{run_id} receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "eligible" and receipt.get("step") == 200, f"{run_id}: checkpoint not eligible at step 200")
    for field in ("run_id", "mechanism", "arm"):
        require(str(receipt.get(field)) == matrix_row[field], f"{run_id}: receipt {field} mismatch")
    require(receipt.get("run_spec_sha256") == spec_sha, f"{run_id}: receipt/run-spec mismatch")
    require(receipt.get("inference_lora_scale") == LORA_SCALE, f"{run_id}: receipt scale mismatch")
    require(receipt.get("finite_receipt", {}).get("status") == "passed" and receipt.get("finite_receipt", {}).get("nonfinite_tensor_count") == 0, f"{run_id}: finite checkpoint receipt failed")
    require(receipt.get("role_step_counts") == {"erase": 100, "preserve": 100}, f"{run_id}: training role counts mismatch")
    basis = receipt.get("cache_validation_basis", {}).get("run_spec_registry", {})
    require(basis.get("sha256") == EXPECTED_RUN_SPEC_REGISTRY_SHA256, f"{run_id}: receipt registry binding mismatch")

    checkpoint = training_dir / "checkpoint-000200"
    real_directory(checkpoint, f"{run_id} checkpoint")
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    state = checkpoint / "training_state.json"
    regular_file(weights, f"{run_id} weights")
    regular_file(state, f"{run_id} training state")
    require(set(checkpoint.iterdir()) == {weights, state}, f"{run_id}: checkpoint inventory is not exact")
    checkpoint_ref = receipt.get("checkpoint", {})
    require(sha256_file(weights) == checkpoint_ref.get("weights_sha256"), f"{run_id}: weights SHA mismatch")
    require(sha256_file(state) == checkpoint_ref.get("training_state_sha256"), f"{run_id}: training-state SHA mismatch")
    require(Path(str(checkpoint_ref.get("path", ""))).resolve() == checkpoint.resolve(), f"{run_id}: receipt checkpoint path mismatch")
    return {
        "run_spec": str(spec_path),
        "run_spec_sha256": spec_sha,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checkpoint": str(checkpoint),
        "checkpoint_artifact_sha256": artifact_sha256(checkpoint),
        "weights_sha256": checkpoint_ref["weights_sha256"],
        "training_state_sha256": checkpoint_ref["training_state_sha256"],
    }


def prompt_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        f"{row['prompt']} | {row['source_object']} | {row['expected_counterfactual_state']}"
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_command(
    python: Path, generator: Path, shard: Path, output_dir: Path, model: Path,
    checkpoint: Path, seeds: Sequence[int],
) -> list[str]:
    return [
        str(python), str(generator), "--baseline", "clean", "--prompts", str(shard),
        "--output-dir", str(output_dir), "--model", str(model), "--lora-path",
        str(checkpoint), "--lora-scale", "1.25", "--seeds",
        ",".join(str(seed) for seed in seeds), "--steps", "25", "--guidance-scale",
        "5", "--num-frames", "49", "--fps", "8", "--height", "480",
        "--width", "832", "--dtype", "bf16", "--device", "cuda",
        "--vae-slicing", "--vae-tiling",
    ]


def build_plan(
    *, project_root: Path, formal_path: Path, identification_path: Path,
    run_matrix_path: Path, registry_path: Path, output_root: Path,
    generator: Path, python: Path, model: Path, gpus: Sequence[int],
) -> dict[str, Any]:
    require(sha256_file(formal_path) == EXPECTED_FORMAL_SHA256, "formal_cases.csv SHA mismatch")
    require(sha256_file(identification_path) == EXPECTED_IDENTIFICATION_SHA256, "identification_subset.csv SHA mismatch")
    require(sha256_file(run_matrix_path) == EXPECTED_RUN_MATRIX_SHA256, "run_matrix.csv SHA mismatch")
    require(sha256_file(registry_path) == EXPECTED_RUN_SPEC_REGISTRY_SHA256, "run-spec registry SHA mismatch")
    require(sha256_file(generator) == EXPECTED_GENERATOR_SHA256, "Wan generator SHA mismatch")
    formal = read_csv(formal_path)
    identification = read_csv(identification_path)
    matrix = read_csv(run_matrix_path)
    validate_formal_rows(formal)
    identification_join = validate_identification_rows(identification, formal)
    validate_run_matrix(matrix)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    require(registry.get("status") == "training_run_specs_frozen_after_cache_validation", "run-spec registry not frozen")
    require(registry.get("run_spec_count") == JOB_COUNT, "run-spec registry count mismatch")
    require(set(registry.get("run_specs", {})) == {row["run_id"] for row in matrix}, "run-spec registry IDs mismatch")

    formal_by_mechanism = {mechanism: [row for row in formal if row["mechanism"] == mechanism] for mechanism in MECHANISMS}
    jobs: list[dict[str, Any]] = []
    for job_index, matrix_row in enumerate(matrix):
        binding = validate_checkpoint(project_root, matrix_row, registry)
        if matrix_row["run_group"] == "main":
            selected = formal_by_mechanism[matrix_row["mechanism"]]
        else:
            selected = [
                identification_join[(matrix_row["mechanism"], row["case_id"])]
                for row in identification
                if row["mechanism"] == matrix_row["mechanism"]
            ]
        expected_rows = MAIN_JOB_ROWS if matrix_row["run_group"] == "main" else IDENTIFICATION_JOB_ROWS
        require(len(selected) == expected_rows, f"{matrix_row['run_id']}: selected case count mismatch")
        wave = job_index // 4
        gpu = int(gpus[job_index % 4])
        run_id = matrix_row["run_id"]
        shard = output_root / "prompt_shards" / f"{job_index:02d}_{run_id}.prompts"
        output_dir = output_root / matrix_row["run_group"] / matrix_row["arm"].lower() / matrix_row["mechanism"]
        shard_raw = prompt_bytes(selected)
        contract = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "runner_id": RUNNER_ID,
            "job_index": job_index,
            "job_id": run_id,
            "wave_index": wave,
            "gpu": gpu,
            "run_group": matrix_row["run_group"],
            "arm": matrix_row["arm"],
            "mechanism": matrix_row["mechanism"],
            "row_count": expected_rows,
            "case_ids": [row["case_id"] for row in selected],
            "formal_global_indices": [int(row["global_case_index"]) for row in selected],
            "prompts": [row["prompt"] for row in selected],
            "target_concepts": [row["source_object"] for row in selected],
            "expected_effects": [row["expected_counterfactual_state"] for row in selected],
            "seeds": [int(row["seed"]) for row in selected],
            "prompt_shard": str(shard),
            "prompt_shard_sha256": hashlib.sha256(shard_raw).hexdigest(),
            "output_dir": str(output_dir),
            "log_path": str(output_root / "logs" / f"{job_index:02d}_{run_id}.log"),
            "status_path": str(output_root / "statuses" / f"{job_index:02d}_{run_id}.json"),
            "job_manifest_path": str(output_root / "job_manifests" / f"{job_index:02d}_{run_id}.json"),
            **binding,
        }
        contract["command"] = build_command(
            python, generator, shard, output_dir, model,
            Path(binding["checkpoint"]), contract["seeds"],
        )
        contract["environment"] = {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        contract["unset_environment"] = ["PYTHONHOME", "PYTHONPATH"]
        jobs.append(contract)
    require(Counter(job["wave_index"] for job in jobs) == {0: 4, 1: 4, 2: 4, 3: 4, 4: 2}, "scheduler must be 4+4+4+4+2")
    require(sum(job["row_count"] for job in jobs) == EXPECTED_VIDEOS, "expected output count is not 684")
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "inputs": {
            "formal_cases": str(formal_path), "formal_cases_sha256": EXPECTED_FORMAL_SHA256,
            "identification_subset": str(identification_path), "identification_subset_sha256": EXPECTED_IDENTIFICATION_SHA256,
            "run_matrix": str(run_matrix_path), "run_matrix_sha256": EXPECTED_RUN_MATRIX_SHA256,
            "run_spec_registry": str(registry_path), "run_spec_registry_sha256": EXPECTED_RUN_SPEC_REGISTRY_SHA256,
        },
        "implementation": {
            "generator": str(generator), "generator_sha256": EXPECTED_GENERATOR_SHA256,
            "python_executable": str(python), "media_probe_python": str(python),
            "model": str(model),
        },
        "generation": {
            "baseline": "clean", "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE, "num_frames": NUM_FRAMES,
            "fps": FPS, "height": HEIGHT, "width": WIDTH, "dtype": DTYPE,
            "lora_scale": LORA_SCALE, "scientific_retry": False,
            "skip_existing": False, "per_case_seed": "formal_cases.csv",
        },
        "scheduler": {
            "gpus": list(gpus), "waves": 5, "jobs_per_wave": [4, 4, 4, 4, 2],
            "resume_policy": "skip_only_fully_revalidated_completed_jobs_refuse_all_partials",
        },
        "expected_videos": EXPECTED_VIDEOS,
        "jobs": jobs,
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_exclusive(temporary, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def status_payload(job: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1, "job_id": job["job_id"], "job_index": job["job_index"],
        "wave_index": job["wave_index"], "gpu": job["gpu"], "status": status,
        "expected_videos": job["row_count"], "updated_at_utc": utc_now(),
    }
    payload.update(extra)
    return payload


def prepare_output_root(output_root: Path, plan: Mapping[str, Any]) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must be fresh: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    require(not output_root.parent.is_symlink(), "output-root parent is symlinked")
    os.mkdir(output_root)
    for name in ("prompt_shards", "job_manifests", "statuses", "logs"):
        (output_root / name).mkdir()
    for job in plan["jobs"]:
        shard_raw = (
            "\n".join(
                f"{prompt} | {target} | {effect}"
                for prompt, target, effect in zip(job["prompts"], job["target_concepts"], job["expected_effects"])
            ) + "\n"
        ).encode("utf-8")
        require(hashlib.sha256(shard_raw).hexdigest() == job["prompt_shard_sha256"], "prompt shard hash construction mismatch")
        write_exclusive(Path(job["prompt_shard"]), shard_raw)
        write_exclusive(Path(job["job_manifest_path"]), canonical_json_bytes(job))
        write_json_atomic(Path(job["status_path"]), status_payload(job, "planned"))
    write_exclusive(output_root / "eval_run_manifest.json", canonical_json_bytes(plan))
    write_aggregate(output_root, plan, "planned")


def validate_bound_inputs(plan: Mapping[str, Any], *, runtime: bool) -> None:
    for name in ("formal_cases", "identification_subset", "run_matrix", "run_spec_registry"):
        path = Path(plan["inputs"][name])
        regular_file(path, name)
        require(sha256_file(path) == plan["inputs"][f"{name}_sha256"], f"{name} changed after planning")
    generator = Path(plan["implementation"]["generator"])
    regular_file(generator, "Wan generator")
    require(sha256_file(generator) == plan["implementation"]["generator_sha256"], "Wan generator changed after planning")
    for job in plan["jobs"]:
        shard = Path(job["prompt_shard"])
        manifest = Path(job["job_manifest_path"])
        regular_file(shard, f"{job['job_id']} prompt shard")
        regular_file(manifest, f"{job['job_id']} job manifest")
        require(sha256_file(shard) == job["prompt_shard_sha256"], f"{job['job_id']}: prompt shard changed")
        require(manifest.read_bytes() == canonical_json_bytes(job), f"{job['job_id']}: job manifest changed")
        for field in ("run_spec", "receipt"):
            path = Path(job[field]); regular_file(path, f"{job['job_id']} {field}")
            require(sha256_file(path) == job[f"{field}_sha256"], f"{job['job_id']}: {field} changed")
        checkpoint = Path(job["checkpoint"])
        real_directory(checkpoint, f"{job['job_id']} checkpoint")
        require(artifact_sha256(checkpoint) == job["checkpoint_artifact_sha256"], f"{job['job_id']}: checkpoint changed")
    if runtime:
        python = Path(plan["implementation"]["python_executable"])
        model = Path(plan["implementation"]["model"])
        regular_file(python, "Wan Python")
        require(os.access(python, os.X_OK), "Wan Python is not executable")
        real_directory(model, "Wan model")


def probe_video_media(runtime_python: Path, video_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None); environment.pop("PYTHONPATH", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    result = subprocess.run(
        [str(runtime_python), "-c", PYAV_PROBE_CODE, str(video_path)],
        capture_output=True, text=True, check=False, env=environment,
    )
    require(result.returncode == 0, f"PyAV could not decode {video_path}: {result.stderr.strip()}")
    raw = json.loads(result.stdout)
    require(raw.get("video_streams") == 1 and raw.get("audio_streams") == 0, f"{video_path}: stream inventory mismatch")
    require(raw.get("decoded_frames") == NUM_FRAMES, f"{video_path}: decoded-frame mismatch")
    require((raw.get("width"), raw.get("height")) == (WIDTH, HEIGHT), f"{video_path}: resolution mismatch")
    require(Fraction(str(raw.get("fps"))) == Fraction(FPS, 1), f"{video_path}: fps mismatch")
    return {"video_streams": 1, "audio_streams": 0, "decoded_frames": 49, "fps": "8/1", "width": 832, "height": 480}


def expected_generation(job: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": "clean", "seed": 42, "seeds": job["seeds"],
        "num_inference_steps": 25, "guidance_scale": 5.0, "num_frames": 49,
        "fps": 8, "height": 480, "width": 832, "dtype": "bf16", "device": "cuda",
        "enable_model_cpu_offload": False, "enable_sequential_cpu_offload": False,
        "vae_slicing": True, "vae_tiling": True,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "lora_path": job["checkpoint"], "lora_sha256": job["checkpoint_artifact_sha256"],
        "lora_scale": 1.25, "activation_gate_dir": None,
        "persistent_activation_gate": False, "lora_target_phrases": [],
        "attention_gate_dir": None, "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
    }


def validate_job_outputs(
    job: Mapping[str, Any], plan: Mapping[str, Any], *,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> dict[str, Any]:
    output = Path(job["output_dir"])
    real_directory(output, f"{job['job_id']} output")
    manifest = output / "generation_manifest.json"
    regular_file(manifest, f"{job['job_id']} generation manifest")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    require(raw.get("baseline") == "clean" and raw.get("pipeline") == "WanPipeline" and raw.get("dry_run") is False, f"{job['job_id']}: generator identity mismatch")
    require(raw.get("model") == plan["implementation"]["model"], f"{job['job_id']}: model mismatch")
    require(raw.get("prompts") == job["prompt_shard"], f"{job['job_id']}: prompt binding mismatch")
    require(raw.get("generation") == expected_generation(job, plan), f"{job['job_id']}: generation config mismatch")
    items = raw.get("items")
    require(isinstance(items, list) and len(items) == job["row_count"], f"{job['job_id']}: item count mismatch")
    videos_dir = output / "videos"
    real_directory(videos_dir, f"{job['job_id']} videos")
    outputs: list[dict[str, Any]] = []
    paths: set[Path] = set()
    expected_keys = {"index", "prompt", "target_concept", "expected_effect", "seed", "video_path"}
    for index, item in enumerate(items):
        require(isinstance(item, dict) and set(item) == expected_keys, f"{job['job_id']} item {index}: fields mismatch")
        require(item["index"] == index and item["prompt"] == job["prompts"][index], f"{job['job_id']} item {index}: order/prompt mismatch")
        require(item["target_concept"] == job["target_concepts"][index] and item["expected_effect"] == job["expected_effects"][index], f"{job['job_id']} item {index}: semantic binding mismatch")
        require(item["seed"] == job["seeds"][index], f"{job['job_id']} item {index}: seed mismatch")
        video = Path(item["video_path"])
        regular_file(video, f"{job['job_id']} video {index}")
        require(video.parent.resolve() == videos_dir.resolve() and video.suffix.lower() == ".mp4" and video.stat().st_size > 0, f"{job['job_id']} item {index}: video path/bytes invalid")
        require(video not in paths, f"{job['job_id']}: duplicate video path")
        paths.add(video)
        media = media_probe(Path(plan["implementation"]["media_probe_python"]), video)
        require(media == {"video_streams": 1, "audio_streams": 0, "decoded_frames": 49, "fps": "8/1", "width": 832, "height": 480}, f"{job['job_id']} item {index}: media contract mismatch")
        outputs.append({
            "job_id": job["job_id"], "arm": job["arm"], "mechanism": job["mechanism"],
            "case_id": job["case_ids"][index], "formal_global_index": job["formal_global_indices"][index],
            "seed": item["seed"], "video_path": str(video), "video_sha256": sha256_file(video),
            "size_bytes": video.stat().st_size, "media": media,
        })
    require(set(videos_dir.iterdir()) == paths, f"{job['job_id']}: videos directory inventory mismatch")
    require(set(output.iterdir()) == {manifest, videos_dir}, f"{job['job_id']}: output inventory mismatch")
    return {
        "generation_manifest_sha256": sha256_file(manifest),
        "validated_video_count": len(outputs),
        "validated_video_bytes": sum(item["size_bytes"] for item in outputs),
        "outputs": outputs,
    }


def read_status(job: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(job["status_path"]); regular_file(path, f"{job['job_id']} status")
    return json.loads(path.read_text(encoding="utf-8"))


def write_aggregate(output_root: Path, plan: Mapping[str, Any], state: str, error: str | None = None) -> None:
    statuses = [read_status(job) for job in plan["jobs"]]
    payload: dict[str, Any] = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "runner_id": RUNNER_ID,
        "status": state, "updated_at_utc": utc_now(), "expected_videos": EXPECTED_VIDEOS,
        "validated_videos": sum(int(status.get("validated_video_count", 0)) for status in statuses),
        "status_counts": dict(sorted(Counter(status["status"] for status in statuses).items())),
        "run_manifest": {"path": str(output_root / "eval_run_manifest.json"), "sha256": sha256_file(output_root / "eval_run_manifest.json")},
    }
    final = output_root / "eval_generation_manifest.json"
    if final.is_file():
        payload["generation_manifest"] = {"path": str(final), "sha256": sha256_file(final)}
    if error:
        payload["error"] = error
    write_json_atomic(output_root / "eval_aggregate.json", payload)


def load_existing_plan(output_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / "eval_run_manifest.json"
    regular_file(path, "existing eval run manifest")
    actual = json.loads(path.read_text(encoding="utf-8"))
    comparison = dict(expected); comparison["created_at_utc"] = actual.get("created_at_utc")
    require(actual == comparison, "existing run manifest differs from current frozen plan")
    return actual


def preflight_resume(plan: Mapping[str, Any], *, media_probe: Callable[[Path, Path], dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for job in plan["jobs"]:
        status = read_status(job)
        require(status.get("job_id") == job["job_id"] and status.get("job_index") == job["job_index"], f"{job['job_id']}: status identity mismatch")
        output = Path(job["output_dir"]); log = Path(job["log_path"])
        if status.get("status") == "completed":
            observed = validate_job_outputs(job, plan, media_probe=media_probe)
            require(observed == {key: status.get(key) for key in observed}, f"{job['job_id']}: completed status/output drift")
            completed.add(job["job_id"])
        else:
            require(status.get("status") == "planned", f"{job['job_id']}: refusing non-completed status {status.get('status')}")
            require(not output.exists() and not output.is_symlink(), f"{job['job_id']}: refusing partial output")
            require(not log.exists() and not log.is_symlink(), f"{job['job_id']}: refusing previously launched log")
    return completed


def terminate(running: Mapping[str, tuple[Mapping[str, Any], Any, Any]]) -> None:
    for _, process, _ in running.values():
        if process.poll() is None: process.terminate()
    deadline = time.monotonic() + 10
    for _, process, _ in running.values():
        while process.poll() is None and time.monotonic() < deadline: time.sleep(0.1)
        if process.poll() is None: process.kill()


def freeze_final(output_root: Path, plan: Mapping[str, Any], *, media_probe: Callable[[Path, Path], dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        status = read_status(job)
        require(status.get("status") == "completed", f"{job['job_id']}: not completed")
        observed = validate_job_outputs(job, plan, media_probe=media_probe)
        require(observed == {key: status.get(key) for key in observed}, f"{job['job_id']}: final validation drift")
        items.extend(observed["outputs"])
    require(len(items) == EXPECTED_VIDEOS and len({(item["job_id"], item["case_id"]) for item in items}) == EXPECTED_VIDEOS, "final 684-output inventory mismatch")
    payload = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "runner_id": RUNNER_ID,
        "status": "frozen_after_exact_684_video_validation", "video_count": EXPECTED_VIDEOS,
        "inputs": plan["inputs"], "generation": plan["generation"], "items": items,
    }
    final = output_root / "eval_generation_manifest.json"
    raw = canonical_json_bytes(payload)
    if final.exists():
        regular_file(final, "existing final manifest")
        require(final.read_bytes() == raw, "existing final manifest differs from revalidation")
    else:
        write_exclusive(final, raw)


def execute_plan(
    plan: Mapping[str, Any], output_root: Path, *, poll_interval: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> None:
    validate_bound_inputs(plan, runtime=True)
    completed = preflight_resume(plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "running")
    for wave in range(5):
        jobs = [job for job in plan["jobs"] if job["wave_index"] == wave and job["job_id"] not in completed]
        if not jobs: continue
        running: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        failures: list[str] = []
        try:
            for job in jobs:
                validate_bound_inputs(plan, runtime=True)
                write_json_atomic(Path(job["status_path"]), status_payload(job, "running", started_at_utc=utc_now()))
                log = Path(job["log_path"]).open("xb")
                environment = os.environ.copy()
                for key in job["unset_environment"]: environment.pop(key, None)
                environment.update(job["environment"])
                process = popen_factory(job["command"], cwd=plan["project_root"], stdout=log, stderr=subprocess.STDOUT, env=environment)
                running[job["job_id"]] = (job, process, log)
            while running:
                progressed = False
                for job_id, (job, process, log) in list(running.items()):
                    code = process.poll()
                    if code is None: continue
                    progressed = True; log.close(); running.pop(job_id)
                    if code != 0:
                        failures.append(f"{job_id}: generator exited {code}")
                        write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", return_code=code))
                        continue
                    try:
                        observed = validate_job_outputs(job, plan, media_probe=media_probe)
                    except BaseException as exc:
                        failures.append(f"{job_id}: {type(exc).__name__}: {exc}")
                        write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", return_code=0, error=str(exc)))
                        continue
                    write_json_atomic(Path(job["status_path"]), status_payload(job, "completed", return_code=0, **observed))
                    completed.add(job_id)
                write_aggregate(output_root, plan, "running")
                if running and not progressed: sleep_fn(poll_interval)
        except BaseException:
            terminate(running)
            for job, _, log in running.values():
                log.close()
                write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", error="launcher interrupted"))
            write_aggregate(output_root, plan, "failed", "launcher interrupted")
            raise
        if failures:
            message = "; ".join(failures); write_aggregate(output_root, plan, "failed", message)
            raise RuntimeError(message)
    validate_bound_inputs(plan, runtime=True)
    freeze_final(output_root, plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--formal-cases", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--identification-subset", type=Path, default=DEFAULT_IDENTIFICATION)
    parser.add_argument("--run-matrix", type=Path, default=DEFAULT_RUN_MATRIX)
    parser.add_argument("--run-spec-registry", type=Path, default=DEFAULT_RUN_SPEC_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpus", type=parse_gpus, required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require(args.poll_interval > 0, "poll interval must be positive")
        project = args.project_root.resolve(strict=True)
        formal = resolve(project, args.formal_cases)
        identification = resolve(project, args.identification_subset)
        matrix = resolve(project, args.run_matrix)
        registry = resolve(project, args.run_spec_registry)
        output = resolve(project, args.output_root, must_exist=False)
        generator = resolve(project, DEFAULT_GENERATOR)
        python = resolve(project, args.python_executable, must_exist=args.run)
        model = resolve(project, args.model, must_exist=args.run)
        require(generator == (project / DEFAULT_GENERATOR).resolve(strict=True), "only the frozen Wan generator is allowed")
        plan = build_plan(
            project_root=project, formal_path=formal, identification_path=identification,
            run_matrix_path=matrix, registry_path=registry, output_root=output,
            generator=generator, python=python, model=model, gpus=args.gpus,
        )
        if args.plan:
            prepare_output_root(output, plan)
            print(f"Planned 18 jobs / 684 videos at {output}")
            return 0
        if output.exists():
            plan = load_existing_plan(output, plan)
        else:
            prepare_output_root(output, plan)
        execute_plan(plan, output, poll_interval=args.poll_interval)
        print(f"Completed and froze 684 trained-Wan videos at {output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
