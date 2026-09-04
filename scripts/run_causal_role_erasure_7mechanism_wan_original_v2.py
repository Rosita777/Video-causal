#!/usr/bin/env python3
"""Generate the 294-case formal Wan Original stream, fail closed.

This runner is deliberately independent from ``formal_eval_v2``.  It binds the
frozen 294-row formal-case table, the frozen Wan base-model/runtime registries,
and the frozen clean generator, then launches one no-LoRA process per mechanism
in deterministic 4+3 GPU waves.  Resume is allowed only for jobs whose complete
outputs, media, SHA-256 digests, and immutable receipts revalidate exactly;
partial, running, or failed jobs are never retried in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROTOCOL_ID = "causal_role_erasure_7m_single_seed_v2"
RUNNER_ID = "causal_role_erasure_7mechanism_wan_original_v2"
STREAM = "wan_original"
RUNNER_RELATIVE = Path(
    "scripts/run_causal_role_erasure_7mechanism_wan_original_v2.py"
)
QUEUE_RELATIVE = Path(
    "scripts/run_causal_role_erasure_7mechanism_wan_original_queue_v2.sh"
)
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)

FORMAL_ROWS = 294
JOB_ROWS = 42
JOB_COUNT = 7
WAVE_COUNT = 2
STEPS = 25
GUIDANCE_SCALE = 5.0
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
DTYPE = "bf16"

EXPECTED_FORMAL_SHA256 = "555ec79a1f084a8b75c802cd570801faf98531fc063fa0eb5170ab8c1d710a04"
EXPECTED_GENERATOR_SHA256 = "04bc6b8a8f93d885137c6157509b00f46824231560989f7b7f78192b26469b5e"
EXPECTED_MODEL_INVENTORY_SHA256 = "51e7199b99ee206934924ee043bd01b40ba413dfb60ef1e72682d36a10b46290"
EXPECTED_RUNTIME_REGISTRY_SHA256 = "9043adf7f823022b20711267ab9e28b9dfb72452273d26e27c131b92e65eff01"

DEFAULT_FORMAL = Path("data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv")
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2"
)
DEFAULT_GENERATOR = Path("scripts/generate_wan_clean.py")
DEFAULT_PYTHON = Path("models/.wan-runtime/bin/python")
DEFAULT_MODEL = Path("models/Wan2.1-T2V-1.3B-Diffusers")
DEFAULT_MODEL_INVENTORY = Path(
    "data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json"
)
DEFAULT_RUNTIME_REGISTRY = Path(
    "data/water_impact_dynamic_v4/v4_runtime_registry_v3.json"
)

MEDIA_CONTRACT = {
    "video_streams": 1,
    "audio_streams": 0,
    "decoded_frames": NUM_FRAMES,
    "fps": f"{FPS}/1",
    "width": WIDTH,
    "height": HEIGHT,
}

PYAV_PROBE_CODE = r"""
import av, json, sys
from fractions import Fraction
with av.open(sys.argv[1]) as container:
    videos = [stream for stream in container.streams if stream.type == "video"]
    audios = [stream for stream in container.streams if stream.type == "audio"]
    if len(videos) != 1:
        raise SystemExit("expected exactly one video stream")
    stream = videos[0]
    rate = stream.average_rate or stream.guessed_rate
    frames = sum(1 for _ in container.decode(video=stream.index))
    print(json.dumps({
        "video_streams": len(videos),
        "audio_streams": len(audios),
        "decoded_frames": frames,
        "fps": None if rate is None else f"{Fraction(rate).numerator}/{Fraction(rate).denominator}",
        "width": int(stream.width),
        "height": int(stream.height),
    }, sort_keys=True))
""".strip()

RUNTIME_PROBE_CODE = r"""
import importlib
import importlib.metadata
import json
import platform
import sys

names = ("torch", "diffusers", "transformers")
modules = {name: importlib.import_module(name) for name in names}
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "python": {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    },
    "package_versions": {
        name: importlib.metadata.version(name) for name in names
    },
    "module_versions": {
        name: str(getattr(modules[name], "__version__", "")) for name in names
    },
    "module_origins": {
        name: str(getattr(modules[name], "__file__", "")) for name in names
    },
}, sort_keys=True))
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


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def real_directory(path: Path, label: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def resolve(project_root: Path, value: Path, *, must_exist: bool = True) -> Path:
    path = value if value.is_absolute() else project_root / value
    return path.resolve(strict=must_exist)


def parse_gpus(value: str) -> tuple[int, int, int, int]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must contain four integers") from exc
    if len(parsed) != 4 or len(set(parsed)) != 4 or any(gpu < 0 for gpu in parsed):
        raise argparse.ArgumentTypeError(
            "--gpus requires exactly four distinct non-negative GPU indices"
        )
    return parsed  # type: ignore[return-value]


def read_formal_rows(path: Path) -> list[dict[str, str]]:
    regular_file(path, "formal cases")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "formal cases have no header")
        rows = [dict(row) for row in reader]
    require(len(rows) == FORMAL_ROWS, "formal case count must be exactly 294")
    require(
        [int(row["global_case_index"]) for row in rows] == list(range(FORMAL_ROWS)),
        "formal global indices are not exact/canonical",
    )
    require(len({row["case_id"] for row in rows}) == FORMAL_ROWS, "formal case IDs repeat")
    require(len({int(row["seed"]) for row in rows}) == FORMAL_ROWS, "formal seeds repeat")
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        selected = [row for row in rows if row["mechanism"] == mechanism]
        require(len(selected) == JOB_ROWS, f"{mechanism}: expected exactly 42 rows")
        require(
            {int(row["mechanism_index"]) for row in selected} == {mechanism_index},
            f"{mechanism}: mechanism index mismatch",
        )
        require(
            Counter(row["case_kind"] for row in selected)
            == {"causal": 24, "specificity": 18},
            f"{mechanism}: causal/specificity balance changed",
        )
        for row in selected:
            require(row["protocol_version"] == PROTOCOL_ID, "formal protocol mismatch")
            require(
                (
                    int(row["num_frames"]),
                    int(row["fps"]),
                    int(row["height"]),
                    int(row["width"]),
                )
                == (NUM_FRAMES, FPS, HEIGHT, WIDTH),
                f"{row['case_id']}: media contract mismatch",
            )
            for field in ("prompt", "source_object", "expected_counterfactual_state"):
                value = str(row[field]).strip()
                require(
                    value and "|" not in value and "\n" not in value,
                    f"{row['case_id']}: invalid {field}",
                )
    return rows


def probe_runtime_process(python: Path) -> dict[str, Any]:
    regular_file(python, "Wan runtime Python")
    require(os.access(python, os.X_OK), "Wan runtime Python is not executable")
    with python.open("rb") as handle:
        require(handle.read(4) == b"\x7fELF", "Wan runtime Python is not an ELF executable")
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1"
    )
    result = subprocess.run(
        [str(python), "-c", RUNTIME_PROBE_CODE],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    require(result.returncode == 0, f"Wan runtime probe failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    require(isinstance(payload, dict), "Wan runtime probe returned a non-object")
    payload["executable"] = str(Path(str(payload.get("executable", ""))).resolve())
    payload["binary_format"] = "ELF"
    payload["python_executable_sha256"] = sha256_file(python)
    return payload


def runtime_content_inventory(runtime_root: Path) -> dict[str, Any]:
    """Reproduce the frozen v3 ordered path/NUL/bytes/newline inventory."""

    real_directory(runtime_root, "Wan runtime root")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(runtime_root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"runtime contains symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"runtime contains non-regular or hardlinked file: {path}",
        )
        relative = path.relative_to(runtime_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        observed_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                observed_size += len(chunk)
        require(observed_size == info.st_size, f"runtime file changed while hashing: {path}")
        digest.update(b"\n")
        file_count += 1
        total_bytes += observed_size
    require(file_count > 0 and total_bytes > 0, "Wan runtime is empty")
    return {
        "content_inventory_algorithm": "sha256_ordered_relative_path_nul_raw_bytes_newline_v1",
        "content_file_count": file_count,
        "content_total_bytes": total_bytes,
        "content_inventory_sha256": digest.hexdigest(),
    }


def runtime_origin_record(project_root: Path, runtime_root: Path, origin: str) -> dict[str, Any]:
    path = Path(origin)
    require(path.is_absolute(), "runtime module origin is not absolute")
    resolved = path.resolve(strict=True)
    runtime_resolved = runtime_root.resolve(strict=True)
    require(runtime_resolved in resolved.parents, "runtime module origin escaped runtime root")
    regular_file(resolved, "runtime module origin")
    return {
        "path": resolved.relative_to(project_root.resolve(strict=True)).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_runtime_live(
    project_root: Path,
    python: Path,
    runtime_registry: Path,
    *,
    verify_content: bool,
) -> dict[str, Any]:
    regular_file(runtime_registry, "Wan runtime registry")
    require(
        sha256_file(runtime_registry) == EXPECTED_RUNTIME_REGISTRY_SHA256,
        "Wan runtime registry SHA mismatch",
    )
    registered = json.loads(runtime_registry.read_text(encoding="utf-8"))
    require(
        registered.get("protocol") == "water_impact_dynamic_v4_runtime_registry_v3"
        and registered.get("status") == "frozen",
        "Wan runtime registry is not the frozen v3 registry",
    )
    require(
        registered.get("python_executable") == DEFAULT_PYTHON.as_posix(),
        "Wan runtime Python binding mismatch",
    )
    require(
        registered.get("runtime_root") == DEFAULT_PYTHON.parents[1].as_posix(),
        "Wan runtime root binding mismatch",
    )
    runtime_root = project_root / str(registered["runtime_root"])
    real_directory(runtime_root, "Wan runtime root")
    observed = probe_runtime_process(python)
    require(
        Path(str(observed.get("executable", ""))).resolve() == python.resolve(),
        "runtime probe executed a different Python",
    )
    require(
        Path(str(observed.get("prefix", ""))).resolve() == runtime_root.resolve(),
        "runtime probe used a different sys.prefix",
    )
    registered_python = registered.get("python")
    require(
        isinstance(registered_python, dict)
        and observed.get("python")
        == {
            "implementation": registered_python.get("implementation"),
            "version": registered_python.get("version"),
        },
        "live Python identity/version differs from frozen runtime registry",
    )
    registered_packages = registered.get("packages")
    require(isinstance(registered_packages, dict), "runtime package registry is malformed")
    for name in ("torch", "diffusers", "transformers"):
        require(
            observed.get("package_versions", {}).get(name)
            == registered_packages.get(name),
            f"live {name} distribution version differs from frozen runtime registry",
        )
    registered_torch = registered.get("torch")
    require(isinstance(registered_torch, dict), "runtime torch registry is malformed")
    require(
        observed.get("module_versions", {}).get("torch")
        == registered_torch.get("module_version"),
        "live torch module version differs from frozen runtime registry",
    )
    for name in ("diffusers", "transformers"):
        require(
            observed.get("module_versions", {}).get(name)
            == registered_packages.get(name),
            f"live {name} module version differs from frozen runtime registry",
        )
    registered_origins = registered.get("module_origins")
    require(isinstance(registered_origins, dict), "runtime module-origin registry is malformed")
    for name in ("torch", "diffusers", "transformers"):
        origin = observed.get("module_origins", {}).get(name)
        require(isinstance(origin, str) and origin, f"live {name} module origin is missing")
        require(
            runtime_origin_record(project_root, runtime_root, origin)
            == registered_origins.get(name),
            f"live {name} module origin/bytes differ from frozen runtime registry",
        )
    if verify_content:
        live_content = runtime_content_inventory(runtime_root)
        require(
            all(registered.get(key) == value for key, value in live_content.items()),
            "live runtime content inventory differs from frozen runtime registry",
        )
        observed["verified_content_inventory"] = live_content
    return observed


def load_frozen_bindings(
    *,
    project_root: Path,
    model_inventory: Path,
    runtime_registry: Path,
    python: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    regular_file(model_inventory, "Wan model inventory")
    regular_file(runtime_registry, "Wan runtime registry")
    require(
        sha256_file(model_inventory) == EXPECTED_MODEL_INVENTORY_SHA256,
        "Wan model inventory SHA mismatch",
    )
    require(
        sha256_file(runtime_registry) == EXPECTED_RUNTIME_REGISTRY_SHA256,
        "Wan runtime registry SHA mismatch",
    )
    model_payload = json.loads(model_inventory.read_text(encoding="utf-8"))
    require(
        model_payload.get("protocol") == "water_impact_dynamic_v4_model_content_inventory_v3"
        and model_payload.get("status") == "frozen",
        "Wan model inventory is not the frozen v3 inventory",
    )
    require(
        model_payload.get("model_root", model_payload.get("root"))
        == DEFAULT_MODEL.as_posix(),
        "Wan model inventory root mismatch",
    )
    files = model_payload.get("files")
    require(isinstance(files, list) and files, "Wan model inventory has no files")
    runtime_observation = validate_runtime_live(
        project_root, python, runtime_registry, verify_content=True
    )
    model_binding = {
        "path": str(model_inventory),
        "sha256": EXPECTED_MODEL_INVENTORY_SHA256,
        "file_count": len(files),
    }
    runtime_binding = {
        "path": str(runtime_registry),
        "sha256": EXPECTED_RUNTIME_REGISTRY_SHA256,
        "observation": runtime_observation,
    }
    return model_binding, runtime_binding


def validate_model_bytes(project_root: Path, model_inventory: Path, model: Path) -> None:
    """Rehash the exact frozen base model once before a real launch."""

    payload = json.loads(model_inventory.read_text(encoding="utf-8"))
    files = payload["files"]
    real_directory(model, "Wan base model")
    expected_paths: list[Path] = []
    for index, item in enumerate(files):
        require(isinstance(item, dict), f"model inventory row {index} malformed")
        relative = item.get("path")
        require(isinstance(relative, str) and relative, f"model inventory row {index} path missing")
        relative_path = Path(relative)
        require(not relative_path.is_absolute(), f"model inventory row {index} path is absolute")
        target = project_root / relative_path
        if not str(relative).startswith(DEFAULT_MODEL.as_posix() + "/"):
            target = model / relative_path
        target = target.resolve(strict=True)
        require(model.resolve(strict=True) in target.parents, f"model inventory row {index} escaped model root")
        regular_file(target, f"model inventory row {index}")
        size = item.get("size_bytes", item.get("size"))
        if isinstance(size, int):
            require(target.stat().st_size == size, f"model inventory row {index} size mismatch")
        require(sha256_file(target) == item.get("sha256"), f"model inventory row {index} SHA mismatch")
        expected_paths.append(target)
    actual_paths = sorted(path.resolve() for path in model.rglob("*") if path.is_file())
    require(sorted(expected_paths) == actual_paths, "live Wan model inventory has missing or extra files")


def prompt_shard_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    return (
        "\n".join(
            f"{row['prompt']} | {row['source_object']} | {row['expected_counterfactual_state']}"
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def build_command(
    *,
    python: Path,
    generator: Path,
    shard: Path,
    output_dir: Path,
    model: Path,
    seeds: Sequence[int],
) -> list[str]:
    require(len(seeds) == JOB_ROWS, "Wan Original job must contain exactly 42 seeds")
    command = [
        str(python),
        str(generator),
        "--baseline",
        "clean",
        "--prompts",
        str(shard),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--seed",
        "42",
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--steps",
        str(STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--num-frames",
        str(NUM_FRAMES),
        "--fps",
        str(FPS),
        "--height",
        str(HEIGHT),
        "--width",
        str(WIDTH),
        "--dtype",
        DTYPE,
        "--device",
        "cuda",
        "--vae-slicing",
        "--vae-tiling",
    ]
    require("--lora-path" not in command, "Wan Original must not load a LoRA")
    require("--skip-existing" not in command, "fresh formal generation forbids skip-existing")
    return command


def build_plan(
    *,
    project_root: Path,
    formal_path: Path,
    output_root: Path,
    runner_code: Path,
    queue_code: Path,
    generator: Path,
    python: Path,
    model: Path,
    model_inventory: Path,
    runtime_registry: Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    require(len(gpus) == 4 and len(set(gpus)) == 4, "exactly four GPUs are required")
    regular_file(runner_code, "Wan Original runner")
    regular_file(queue_code, "Wan Original queue")
    require(sha256_file(formal_path) == EXPECTED_FORMAL_SHA256, "formal_cases.csv SHA mismatch")
    require(sha256_file(generator) == EXPECTED_GENERATOR_SHA256, "Wan generator SHA mismatch")
    rows = read_formal_rows(formal_path)
    model_binding, runtime_binding = load_frozen_bindings(
        project_root=project_root,
        model_inventory=model_inventory,
        runtime_registry=runtime_registry,
        python=python,
    )
    jobs: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        selected = [row for row in rows if row["mechanism"] == mechanism]
        start = mechanism_index * JOB_ROWS
        require(
            [int(row["global_case_index"]) for row in selected]
            == list(range(start, start + JOB_ROWS)),
            f"{mechanism}: rows are not the canonical contiguous block",
        )
        job_id = f"wan_original_{mechanism}"
        shard = output_root / "prompt_shards" / f"{mechanism_index:02d}_{mechanism}.prompts"
        output_dir = output_root / "mechanisms" / f"{mechanism_index:02d}_{mechanism}"
        shard_raw = prompt_shard_bytes(selected)
        job: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "runner_id": RUNNER_ID,
            "stream": STREAM,
            "job_index": mechanism_index,
            "job_id": job_id,
            "wave_index": mechanism_index // 4,
            "gpu": int(gpus[mechanism_index % 4]),
            "mechanism": mechanism,
            "row_count": JOB_ROWS,
            "case_ids": [row["case_id"] for row in selected],
            "formal_global_indices": [int(row["global_case_index"]) for row in selected],
            "prompts": [row["prompt"] for row in selected],
            "target_concepts": [row["source_object"] for row in selected],
            "expected_effects": [row["expected_counterfactual_state"] for row in selected],
            "seeds": [int(row["seed"]) for row in selected],
            "prompt_shard": str(shard),
            "prompt_shard_sha256": hashlib.sha256(shard_raw).hexdigest(),
            "output_dir": str(output_dir),
            "log_path": str(output_root / "logs" / f"{mechanism_index:02d}_{job_id}.log"),
            "status_path": str(output_root / "statuses" / f"{mechanism_index:02d}_{job_id}.json"),
            "job_manifest_path": str(output_root / "job_manifests" / f"{mechanism_index:02d}_{job_id}.json"),
            "receipt_path": str(output_root / "receipts" / f"{mechanism_index:02d}_{job_id}.json"),
            "environment": {
                "CUDA_VISIBLE_DEVICES": str(gpus[mechanism_index % 4]),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "unset_environment": ["PYTHONHOME", "PYTHONPATH"],
        }
        job["command"] = build_command(
            python=python,
            generator=generator,
            shard=shard,
            output_dir=output_dir,
            model=model,
            seeds=job["seeds"],
        )
        jobs.append(job)
    require(
        Counter(job["wave_index"] for job in jobs) == {0: 4, 1: 3},
        "Wan Original scheduler must be exactly 4+3",
    )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "stream": STREAM,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "inputs": {
            "formal_cases": str(formal_path),
            "formal_cases_sha256": EXPECTED_FORMAL_SHA256,
            "model_inventory": model_binding,
            "runtime_registry": runtime_binding,
        },
        "implementation": {
            "runner": str(runner_code),
            "runner_sha256": sha256_file(runner_code),
            "queue": str(queue_code),
            "queue_sha256": sha256_file(queue_code),
            "generator": str(generator),
            "generator_sha256": EXPECTED_GENERATOR_SHA256,
            "python_executable": str(python),
            "media_probe_python": str(python),
            "model": str(model),
        },
        "generation": {
            "baseline": "clean",
            "method": STREAM,
            "base_model_only": True,
            "lora_path": None,
            "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "height": HEIGHT,
            "width": WIDTH,
            "dtype": DTYPE,
            "device": "cuda",
            "vae_slicing": True,
            "vae_tiling": True,
            "skip_existing": False,
            "per_case_seed": "formal_cases.csv",
        },
        "scheduler": {
            "gpus": list(gpus),
            "waves": WAVE_COUNT,
            "jobs_per_wave": [4, 3],
            "resume_policy": "skip_only_fully_revalidated_completed_jobs_refuse_all_partials",
        },
        "job_count": JOB_COUNT,
        "expected_videos": FORMAL_ROWS,
        "jobs": jobs,
    }


def write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_exclusive(
        temporary,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    os.replace(temporary, path)


def status_payload(job: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "stream": STREAM,
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "mechanism": job["mechanism"],
        "wave_index": job["wave_index"],
        "gpu": job["gpu"],
        "status": status,
        "expected_videos": JOB_ROWS,
        "updated_at_utc": utc_now(),
    }
    payload.update(extra)
    return payload


def prepare_output_root(output_root: Path, plan: Mapping[str, Any]) -> None:
    require(
        not output_root.exists() and not output_root.is_symlink(),
        f"output root must be fresh: {output_root}",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    require(not output_root.parent.is_symlink(), "output-root parent is symlinked")
    os.mkdir(output_root)
    for name in (
        "prompt_shards",
        "job_manifests",
        "statuses",
        "receipts",
        "logs",
        "mechanisms",
    ):
        (output_root / name).mkdir()
    for job in plan["jobs"]:
        raw = (
            "\n".join(
                f"{prompt} | {target} | {effect}"
                for prompt, target, effect in zip(
                    job["prompts"], job["target_concepts"], job["expected_effects"]
                )
            )
            + "\n"
        ).encode("utf-8")
        require(
            hashlib.sha256(raw).hexdigest() == job["prompt_shard_sha256"],
            f"{job['job_id']}: prompt shard construction drift",
        )
        write_exclusive(Path(job["prompt_shard"]), raw)
        write_exclusive(Path(job["job_manifest_path"]), canonical_json_bytes(job))
        write_json_atomic(Path(job["status_path"]), status_payload(job, "planned"))
    write_exclusive(output_root / "wan_original_run_manifest.json", canonical_json_bytes(plan))
    write_aggregate(output_root, plan, "planned")


def validate_bound_inputs(
    plan: Mapping[str, Any],
    *,
    runtime: bool,
    verify_model_content: bool = False,
) -> None:
    formal = Path(plan["inputs"]["formal_cases"])
    model_inventory = Path(plan["inputs"]["model_inventory"]["path"])
    runtime_registry = Path(plan["inputs"]["runtime_registry"]["path"])
    runner_code = Path(plan["implementation"]["runner"])
    queue_code = Path(plan["implementation"]["queue"])
    generator = Path(plan["implementation"]["generator"])
    for path, expected, label in (
        (formal, plan["inputs"]["formal_cases_sha256"], "formal cases"),
        (model_inventory, plan["inputs"]["model_inventory"]["sha256"], "model inventory"),
        (runtime_registry, plan["inputs"]["runtime_registry"]["sha256"], "runtime registry"),
        (runner_code, plan["implementation"]["runner_sha256"], "Wan Original runner"),
        (queue_code, plan["implementation"]["queue_sha256"], "Wan Original queue"),
        (generator, plan["implementation"]["generator_sha256"], "Wan generator"),
    ):
        regular_file(path, label)
        require(sha256_file(path) == expected, f"{label} changed after planning")
    for job in plan["jobs"]:
        shard = Path(job["prompt_shard"])
        manifest = Path(job["job_manifest_path"])
        regular_file(shard, f"{job['job_id']} prompt shard")
        regular_file(manifest, f"{job['job_id']} job manifest")
        require(sha256_file(shard) == job["prompt_shard_sha256"], f"{job['job_id']}: prompt shard changed")
        require(manifest.read_bytes() == canonical_json_bytes(job), f"{job['job_id']}: job manifest changed")
    if runtime:
        python = Path(plan["implementation"]["python_executable"])
        model = Path(plan["implementation"]["model"])
        regular_file(python, "Wan runtime Python")
        require(os.access(python, os.X_OK), "Wan runtime Python is not executable")
        real_directory(model, "Wan base model")
        observed_runtime = validate_runtime_live(
            Path(plan["project_root"]),
            python,
            runtime_registry,
            verify_content=False,
        )
        planned_runtime = dict(plan["inputs"]["runtime_registry"]["observation"])
        planned_runtime.pop("verified_content_inventory", None)
        require(
            observed_runtime == planned_runtime,
            "live Wan runtime changed after planning",
        )
        if verify_model_content:
            validate_model_bytes(Path(plan["project_root"]), model_inventory, model)


def probe_video_media(runtime_python: Path, video_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1"
    )
    result = subprocess.run(
        [str(runtime_python), "-c", PYAV_PROBE_CODE, str(video_path)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    require(
        result.returncode == 0,
        f"PyAV could not decode {video_path}: {result.stderr.strip()}",
    )
    raw = json.loads(result.stdout)
    require(raw.get("video_streams") == 1, f"{video_path}: video stream mismatch")
    require(raw.get("audio_streams") == 0, f"{video_path}: audio stream forbidden")
    require(raw.get("decoded_frames") == NUM_FRAMES, f"{video_path}: frame count mismatch")
    require((raw.get("width"), raw.get("height")) == (WIDTH, HEIGHT), f"{video_path}: resolution mismatch")
    require(Fraction(str(raw.get("fps"))) == Fraction(FPS, 1), f"{video_path}: fps mismatch")
    return dict(MEDIA_CONTRACT)


def expected_generation(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": "clean",
        "seed": 42,
        "seeds": job["seeds"],
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "height": HEIGHT,
        "width": WIDTH,
        "dtype": DTYPE,
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
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


def validate_job_outputs(
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> dict[str, Any]:
    output = Path(job["output_dir"])
    real_directory(output, f"{job['job_id']} output")
    generation_manifest = output / "generation_manifest.json"
    regular_file(generation_manifest, f"{job['job_id']} generation manifest")
    payload = json.loads(generation_manifest.read_text(encoding="utf-8"))
    require(
        payload.get("baseline") == "clean"
        and payload.get("pipeline") == "WanPipeline"
        and payload.get("dry_run") is False,
        f"{job['job_id']}: generator identity mismatch",
    )
    require(payload.get("model") == plan["implementation"]["model"], f"{job['job_id']}: model mismatch")
    require(payload.get("prompts") == job["prompt_shard"], f"{job['job_id']}: prompt binding mismatch")
    require(payload.get("generation") == expected_generation(job), f"{job['job_id']}: generation config mismatch")
    items = payload.get("items")
    require(isinstance(items, list) and len(items) == JOB_ROWS, f"{job['job_id']}: item count mismatch")
    videos_dir = output / "videos"
    real_directory(videos_dir, f"{job['job_id']} videos")
    expected_keys = {"index", "prompt", "target_concept", "expected_effect", "seed", "video_path"}
    listed: set[Path] = set()
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        require(isinstance(item, dict) and set(item) == expected_keys, f"{job['job_id']} item {index}: fields mismatch")
        require(item["index"] == index and item["prompt"] == job["prompts"][index], f"{job['job_id']} item {index}: order/prompt mismatch")
        require(item["target_concept"] == job["target_concepts"][index], f"{job['job_id']} item {index}: target mismatch")
        require(item["expected_effect"] == job["expected_effects"][index], f"{job['job_id']} item {index}: effect mismatch")
        require(item["seed"] == job["seeds"][index], f"{job['job_id']} item {index}: seed mismatch")
        video = Path(str(item["video_path"]))
        regular_file(video, f"{job['job_id']} video {index}")
        require(video.parent.resolve() == videos_dir.resolve(), f"{job['job_id']} item {index}: path escaped videos dir")
        require(video.suffix.lower() == ".mp4" and video.stat().st_size > 0, f"{job['job_id']} item {index}: invalid video")
        require(video not in listed, f"{job['job_id']}: duplicate video path")
        listed.add(video)
        media = media_probe(Path(plan["implementation"]["media_probe_python"]), video)
        require(media == MEDIA_CONTRACT, f"{job['job_id']} item {index}: media contract mismatch")
        outputs.append(
            {
                "stream": STREAM,
                "job_id": job["job_id"],
                "case_id": job["case_ids"][index],
                "mechanism": job["mechanism"],
                "formal_global_index": job["formal_global_indices"][index],
                "seed": item["seed"],
                "video_path": str(video),
                "video_sha256": sha256_file(video),
                "size_bytes": video.stat().st_size,
                "media": media,
            }
        )
    require(set(videos_dir.iterdir()) == listed, f"{job['job_id']}: videos inventory mismatch")
    require(set(output.iterdir()) == {generation_manifest, videos_dir}, f"{job['job_id']}: output inventory mismatch")
    return {
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "validated_video_count": len(outputs),
        "validated_video_bytes": sum(item["size_bytes"] for item in outputs),
        "outputs": outputs,
    }


def receipt_payload(
    output_root: Path,
    job: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "stream": STREAM,
        "status": "validated_complete",
        "job_id": job["job_id"],
        "job_index": job["job_index"],
        "mechanism": job["mechanism"],
        "expected_videos": JOB_ROWS,
        "run_manifest": {
            "path": str(output_root / "wan_original_run_manifest.json"),
            "sha256": sha256_file(output_root / "wan_original_run_manifest.json"),
        },
        "job_manifest": {
            "path": job["job_manifest_path"],
            "sha256": sha256_file(Path(job["job_manifest_path"])),
        },
        "prompt_shard": {
            "path": job["prompt_shard"],
            "sha256": job["prompt_shard_sha256"],
        },
        "generation_manifest": {
            "path": str(Path(job["output_dir"]) / "generation_manifest.json"),
            "sha256": validation["generation_manifest_sha256"],
        },
        "validated_video_count": validation["validated_video_count"],
        "validated_video_bytes": validation["validated_video_bytes"],
        "outputs": validation["outputs"],
    }


def read_status(job: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(job["status_path"])
    regular_file(path, f"{job['job_id']} status")
    return json.loads(path.read_text(encoding="utf-8"))


def write_aggregate(
    output_root: Path,
    plan: Mapping[str, Any],
    state: str,
    error: str | None = None,
) -> None:
    statuses = [read_status(job) for job in plan["jobs"]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "stream": STREAM,
        "status": state,
        "updated_at_utc": utc_now(),
        "expected_jobs": JOB_COUNT,
        "expected_videos": FORMAL_ROWS,
        "validated_videos": sum(int(status.get("validated_video_count", 0)) for status in statuses),
        "status_counts": dict(sorted(Counter(status["status"] for status in statuses).items())),
        "run_manifest": {
            "path": str(output_root / "wan_original_run_manifest.json"),
            "sha256": sha256_file(output_root / "wan_original_run_manifest.json"),
        },
        "jobs": [
            {
                "job_id": status["job_id"],
                "mechanism": status["mechanism"],
                "wave_index": status["wave_index"],
                "gpu": status["gpu"],
                "status": status["status"],
                "validated_video_count": status.get("validated_video_count", 0),
                "receipt_sha256": status.get("receipt_sha256"),
                "error": status.get("error"),
            }
            for status in statuses
        ],
    }
    final = output_root / "wan_original_generation_manifest.json"
    if final.is_file():
        payload["generation_manifest"] = {"path": str(final), "sha256": sha256_file(final)}
    if error is not None:
        payload["error"] = error
    write_json_atomic(output_root / "wan_original_aggregate.json", payload)


def load_existing_plan(output_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / "wan_original_run_manifest.json"
    regular_file(path, "existing Wan Original run manifest")
    actual = json.loads(path.read_text(encoding="utf-8"))
    comparison = dict(expected)
    comparison["created_at_utc"] = actual.get("created_at_utc")
    require(actual == comparison, "existing run manifest differs from current frozen plan")
    return actual


def validate_completed_job(
    output_root: Path,
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]],
) -> dict[str, Any]:
    validation = validate_job_outputs(job, plan, media_probe=media_probe)
    expected_receipt = receipt_payload(output_root, job, validation)
    receipt = Path(job["receipt_path"])
    regular_file(receipt, f"{job['job_id']} receipt")
    require(receipt.read_bytes() == canonical_json_bytes(expected_receipt), f"{job['job_id']}: receipt/output drift")
    return {**validation, "receipt_sha256": sha256_file(receipt)}


def preflight_resume(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]],
) -> set[str]:
    completed: set[str] = set()
    for job in plan["jobs"]:
        status = read_status(job)
        require(
            status.get("job_id") == job["job_id"]
            and status.get("job_index") == job["job_index"],
            f"{job['job_id']}: status identity mismatch",
        )
        output = Path(job["output_dir"])
        log = Path(job["log_path"])
        receipt = Path(job["receipt_path"])
        if status.get("status") == "completed":
            observed = validate_completed_job(
                output_root, job, plan, media_probe=media_probe
            )
            for field in (
                "generation_manifest_sha256",
                "validated_video_count",
                "validated_video_bytes",
                "receipt_sha256",
            ):
                require(status.get(field) == observed[field], f"{job['job_id']}: completed status drift in {field}")
            completed.add(job["job_id"])
        else:
            require(
                status.get("status") == "planned",
                f"{job['job_id']}: refusing non-completed status {status.get('status')}",
            )
            require(not output.exists() and not output.is_symlink(), f"{job['job_id']}: refusing partial output")
            require(not log.exists() and not log.is_symlink(), f"{job['job_id']}: refusing previously launched log")
            require(not receipt.exists() and not receipt.is_symlink(), f"{job['job_id']}: refusing stray receipt")
    return completed


def terminate(running: Mapping[str, tuple[Mapping[str, Any], Any, Any]]) -> None:
    for _, process, _ in running.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for _, process, _ in running.values():
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def freeze_final(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]],
) -> None:
    items: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        status = read_status(job)
        require(status.get("status") == "completed", f"{job['job_id']}: not completed")
        observed = validate_completed_job(output_root, job, plan, media_probe=media_probe)
        require(status.get("receipt_sha256") == observed["receipt_sha256"], f"{job['job_id']}: final receipt drift")
        receipt = json.loads(Path(job["receipt_path"]).read_text(encoding="utf-8"))
        items.extend(receipt["outputs"])
        jobs.append(
            {
                "job_id": job["job_id"],
                "mechanism": job["mechanism"],
                "generation_manifest": receipt["generation_manifest"],
                "receipt": {"path": job["receipt_path"], "sha256": observed["receipt_sha256"]},
            }
        )
    items.sort(key=lambda item: int(item["formal_global_index"]))
    require(
        [item["formal_global_index"] for item in items] == list(range(FORMAL_ROWS)),
        "final formal index inventory is not exact",
    )
    require(len({item["case_id"] for item in items}) == FORMAL_ROWS, "final case IDs repeat")
    require(len({item["video_path"] for item in items}) == FORMAL_ROWS, "final video paths repeat")
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "runner_id": RUNNER_ID,
        "status": "frozen_after_exact_294_video_validation",
        "video_count": FORMAL_ROWS,
        "inputs": plan["inputs"],
        "generation": plan["generation"],
        "jobs": jobs,
        "items": items,
    }
    final = output_root / "wan_original_generation_manifest.json"
    raw = canonical_json_bytes(payload)
    if final.exists():
        regular_file(final, "existing final Wan Original manifest")
        require(final.read_bytes() == raw, "existing final Wan Original manifest drifted")
    else:
        write_exclusive(final, raw)


def execute_plan(
    plan: Mapping[str, Any],
    output_root: Path,
    *,
    poll_interval: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
    verify_model_content: bool = True,
) -> None:
    validate_bound_inputs(
        plan, runtime=True, verify_model_content=verify_model_content
    )
    completed = preflight_resume(output_root, plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "running")
    for wave in range(WAVE_COUNT):
        jobs = [
            job
            for job in plan["jobs"]
            if job["wave_index"] == wave and job["job_id"] not in completed
        ]
        if not jobs:
            continue
        validate_bound_inputs(plan, runtime=True)
        running: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        failures: list[str] = []
        try:
            for job in jobs:
                write_json_atomic(
                    Path(job["status_path"]),
                    status_payload(job, "running", started_at_utc=utc_now()),
                )
                log = Path(job["log_path"]).open("xb")
                environment = os.environ.copy()
                for key in job["unset_environment"]:
                    environment.pop(key, None)
                environment.update(job["environment"])
                try:
                    process = popen_factory(
                        job["command"],
                        cwd=plan["project_root"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                except BaseException as exc:
                    log.close()
                    message = f"launch failed: {type(exc).__name__}: {exc}"
                    write_json_atomic(Path(job["status_path"]), status_payload(job, "failed", error=message))
                    failures.append(f"{job['job_id']}: {message}")
                    continue
                running[job["job_id"]] = (job, process, log)
            while running:
                progressed = False
                for job_id, (job, process, log) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    progressed = True
                    log.close()
                    running.pop(job_id)
                    if code != 0:
                        message = f"generator exited {code}"
                        failures.append(f"{job_id}: {message}")
                        write_json_atomic(
                            Path(job["status_path"]),
                            status_payload(job, "failed", return_code=code, error=message),
                        )
                        continue
                    try:
                        validation = validate_job_outputs(job, plan, media_probe=media_probe)
                        receipt = receipt_payload(output_root, job, validation)
                        write_exclusive(Path(job["receipt_path"]), canonical_json_bytes(receipt))
                        receipt_sha = sha256_file(Path(job["receipt_path"]))
                    except BaseException as exc:
                        message = f"post-generation validation failed: {type(exc).__name__}: {exc}"
                        failures.append(f"{job_id}: {message}")
                        write_json_atomic(
                            Path(job["status_path"]),
                            status_payload(job, "failed", return_code=code, error=message),
                        )
                        continue
                    write_json_atomic(
                        Path(job["status_path"]),
                        status_payload(
                            job,
                            "completed",
                            return_code=0,
                            generation_manifest_sha256=validation["generation_manifest_sha256"],
                            validated_video_count=validation["validated_video_count"],
                            validated_video_bytes=validation["validated_video_bytes"],
                            receipt_path=job["receipt_path"],
                            receipt_sha256=receipt_sha,
                        ),
                    )
                    completed.add(job_id)
                write_aggregate(output_root, plan, "running")
                if running and not progressed:
                    sleep_fn(poll_interval)
        except BaseException:
            terminate(running)
            for job, _, log in running.values():
                try:
                    log.close()
                except OSError:
                    pass
                write_json_atomic(
                    Path(job["status_path"]),
                    status_payload(job, "failed", error="launcher interrupted"),
                )
            write_aggregate(output_root, plan, "failed", "launcher interrupted")
            raise
        if failures:
            message = "; ".join(failures)
            write_aggregate(output_root, plan, "failed", message)
            raise RuntimeError(message)
    validate_bound_inputs(plan, runtime=True)
    freeze_final(output_root, plan, media_probe=media_probe)
    write_aggregate(output_root, plan, "completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--formal-cases", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--runtime-registry", type=Path, default=DEFAULT_RUNTIME_REGISTRY)
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
        output = resolve(project, args.output_root, must_exist=False)
        runner_code = resolve(project, RUNNER_RELATIVE)
        queue_code = resolve(project, QUEUE_RELATIVE)
        generator = resolve(project, DEFAULT_GENERATOR)
        python = resolve(project, args.python_executable)
        model = resolve(project, args.model, must_exist=args.run)
        model_inventory = resolve(project, args.model_inventory)
        runtime_registry = resolve(project, args.runtime_registry)
        require(
            generator == (project / DEFAULT_GENERATOR).resolve(strict=True),
            "only the frozen Wan generator is allowed",
        )
        require(
            model == (project / DEFAULT_MODEL).resolve(strict=args.run),
            "only the frozen Wan base model is allowed",
        )
        require(
            python == (project / DEFAULT_PYTHON).resolve(strict=args.run),
            "only the frozen Wan runtime Python is allowed",
        )
        require(
            output == (project / DEFAULT_OUTPUT_ROOT).resolve(strict=False),
            "only the independent formal_wan_original_v2 output root is allowed",
        )
        plan = build_plan(
            project_root=project,
            formal_path=formal,
            output_root=output,
            runner_code=runner_code,
            queue_code=queue_code,
            generator=generator,
            python=python,
            model=model,
            model_inventory=model_inventory,
            runtime_registry=runtime_registry,
            gpus=args.gpus,
        )
        if args.plan:
            prepare_output_root(output, plan)
            print(f"Planned 7 Wan Original jobs / 294 videos at {output}")
            return 0
        if output.exists():
            plan = load_existing_plan(output, plan)
        else:
            prepare_output_root(output, plan)
        execute_plan(plan, output, poll_interval=args.poll_interval)
        print(f"Completed and froze 294 Wan Original videos at {output}")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
