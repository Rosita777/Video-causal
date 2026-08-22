#!/usr/bin/env python3
"""Plan or run the frozen 192-video Wan capability batch.

The launcher is intentionally narrow.  It accepts only the frozen v1
capability manifest/prompt bytes, makes one 24-prompt Wan process per
mechanism, and schedules the eight processes as two fail-closed waves over
exactly four GPUs.  A dry run writes the complete commands and immutable input
shards but never imports or starts a GPU pipeline.
"""

from __future__ import annotations

import argparse
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


PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v1"
RUNNER_VERSION = "causal_role_erasure_8mechanism_capability_runner_v1"

EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "6d425076a7156aabc9695e6ab4fbe7cf85922261d1282c067fd56d176d05031d"
)
EXPECTED_PROMPTS_SHA256 = (
    "7a7902b1ee183c122014f3b9fa2ef1199fabb8f15293fe8d3c6ec468f963046e"
)
EXPECTED_CSV_SHA256 = (
    "dbb315e6ab7815798d0ffc3d625550e6937b55b5522e683a189f5bab609ea189"
)
EXPECTED_SUMMARY_SHA256 = (
    "2bf018a35ce993cc3496aafaee3f9dcb647f5e7cbc923c8573abade60ea2658e"
)
EXPECTED_GENERATOR_SHA256 = (
    "04bc6b8a8f93d885137c6157509b00f46824231560989f7b7f78192b26469b5e"
)

MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "field_mediated_response",
    "material_release",
    "surface_trace",
)
ROWS_PER_MECHANISM = 24
EXPECTED_ROWS = len(MECHANISM_ORDER) * ROWS_PER_MECHANISM
GPUS_REQUIRED = 4
WAVES = 2

STEPS = 25
GUIDANCE_SCALE = 5.0
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
DTYPE = "bf16"
PYAV_PROBE_CODE = """\
import av, json, sys
container = av.open(sys.argv[1])
streams = [stream for stream in container.streams if stream.type == 'video']
if len(streams) != 1:
    raise RuntimeError(f'expected one video stream, got {len(streams)}')
stream = streams[0]
frames = list(container.decode(video=stream.index))
rate = stream.average_rate or stream.guessed_rate
print(json.dumps({
    'streams': len(streams),
    'decoded_frames': len(frames),
    'width': stream.width,
    'height': stream.height,
    'fps': None if rate is None else f'{rate.numerator}/{rate.denominator}',
}))
"""
RUNTIME_CONTENT_INVENTORY_ALGORITHM = (
    "sha256_ordered_relative_path_nul_raw_bytes_newline_v1"
)
STAT_SEAL_ALGORITHM = (
    "sha256_ordered_relative_path_dev_inode_mode_nlink_size_mtime_ns_ctime_ns_v1"
)
MODEL_INVENTORY_PROTOCOL = "water_impact_dynamic_v4_model_content_inventory_v3"
RUNTIME_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_runtime_registry_v3"
FROZEN_INVENTORY_DATASET_VERSION = "v4_dev72_v3"

DEFAULT_CANONICAL_MANIFEST = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_manifest.canonical.json"
)
DEFAULT_CSV_MANIFEST = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_manifest.csv"
)
DEFAULT_SUMMARY = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_summary.json"
)
DEFAULT_PROMPTS = Path("prompts/causal_role_erasure_8mechanism_capability_v1.prompts")
DEFAULT_STAGE_REGISTRY = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_stage_registry.json"
)
DEFAULT_MODEL_INVENTORY = Path(
    "data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json"
)
DEFAULT_RUNTIME_REGISTRY = Path(
    "data/water_impact_dynamic_v4/v4_runtime_registry_v3.json"
)

ALLOWED_UNTRACKED_EVIDENCE_PATHS = (
    "data/water_impact_dynamic_v4/v4_causal_capacity_confirm_v3.json",
    "data/water_impact_dynamic_v4/v4_causal_capacity_model_v3.json",
    "data/water_impact_dynamic_v4/v4_causal_capacity_search_v3.json",
    "data/water_impact_dynamic_v4/v4_causal_forbidden_seed_source_audit_v3.json",
    "data/water_impact_dynamic_v4/v4_causal_static_graph_audit_v3.json",
    "data/water_impact_dynamic_v4/v4_eval_code_registry_v3.json",
    "data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json",
    "data/water_impact_dynamic_v4/v4_runtime_registry_v3.json",
)

FROZEN_CODE_AND_DATA_PATHS = (
    "docs/causal_role_erasure_8mechanism_protocol_v1.md",
    "scripts/build_causal_role_erasure_8mechanism_capability_v1.py",
    "scripts/causal_role_erasure_8mechanism_capability_review_v1.py",
    "scripts/run_causal_role_erasure_8mechanism_capability_v1.py",
    "tests/test_build_causal_role_erasure_8mechanism_capability_v1.py",
    "tests/test_causal_role_erasure_8mechanism_capability_review_v1.py",
    "tests/test_run_causal_role_erasure_8mechanism_capability_v1.py",
    DEFAULT_CANONICAL_MANIFEST.as_posix(),
    DEFAULT_CSV_MANIFEST.as_posix(),
    DEFAULT_SUMMARY.as_posix(),
    DEFAULT_PROMPTS.as_posix(),
    "data/causal_role_erasure_8mechanism_capability_v1_review_template.csv",
    "data/causal_role_erasure_8mechanism_capability_v1_review_rubric.json",
    "data/causal_role_erasure_8mechanism_capability_v1_review_freeze.json",
    "scripts/generate_wan_clean.py",
)

EXPECTED_ROW_FIELDS = {
    "protocol_version",
    "generation_id",
    "case_id",
    "mechanism_index",
    "combination_index",
    "repetition_index",
    "mechanism",
    "mechanism_name",
    "ontology_status",
    "ontology_provenance",
    "intended_use",
    "method_arm",
    "treatment_status",
    "prompt_style",
    "source_id",
    "source_object",
    "source_family",
    "source_motion",
    "receiver_id",
    "receiver",
    "receiver_family",
    "receiver_clean_state",
    "compatibility_rule",
    "prompt",
    "target_concept",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "seed_formula",
    "num_frames",
    "fps",
    "reference_start_inclusive",
    "reference_end_exclusive",
}


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


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular non-symlink file: {path}")


def require_sha256(value: str, label: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256 digest",
    )


def require_git_oid(value: str) -> None:
    require(
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value),
        "git HEAD is not a canonical object ID",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def reject_sealed_path(*paths: Path) -> None:
    for path in paths:
        lowered = path.as_posix().casefold()
        require(
            "final36" not in lowered and "sealed-final" not in lowered,
            "sealed-final36 paths are forbidden in the capability stage",
        )


def project_relative(project_root: Path, path: Path, label: str) -> str:
    resolved_root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    require(
        resolved != resolved_root and resolved_root in resolved.parents,
        f"{label} escapes the project root: {path}",
    )
    return resolved.relative_to(resolved_root).as_posix()


def require_no_symlink_components(project_root: Path, path: Path, label: str) -> None:
    root = Path(os.path.abspath(project_root))
    candidate = Path(os.path.abspath(path))
    require(candidate == root or root in candidate.parents, f"{label} escapes project root")
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        info = os.lstat(current)
        require(not stat.S_ISLNK(info.st_mode), f"{label} contains a symlink component: {current}")


def file_record(project_root: Path, path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    return {
        "path": project_relative(project_root, path, label),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {label}: {path}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def git_snapshot(
    project_root: Path,
    *,
    allowed_stage_registry: Path | None,
) -> dict[str, str]:
    def invoke(arguments: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        require(result.returncode == 0, f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
        return result.stdout

    head = invoke(["rev-parse", "HEAD"]).strip()
    require_git_oid(head)
    status_lines = [
        line
        for line in invoke(["status", "--porcelain=v1", "--untracked-files=all"]).splitlines()
        if line
    ]
    require(
        all(line.startswith("?? ") for line in status_lines),
        "tracked worktree changes are forbidden at capability stage freeze/run",
    )
    observed_untracked = {line[3:] for line in status_lines}
    expected_untracked = set(ALLOWED_UNTRACKED_EVIDENCE_PATHS)
    if allowed_stage_registry is not None:
        relative = allowed_stage_registry.resolve(strict=True).relative_to(
            project_root.resolve(strict=True)
        ).as_posix()
        expected_untracked.add(relative)
    require(
        observed_untracked == expected_untracked,
        "worktree untracked set must be exactly the eight frozen public evidence files"
        + (" plus this stage registry" if allowed_stage_registry is not None else ""),
    )
    return {
        "head": head,
        "worktree_policy": "no tracked changes; exact eight public evidence files, plus the stage registry after freeze",
    }


def require_git_ignored_output(project_root: Path, output_root: Path) -> None:
    root = project_root.resolve(strict=True)
    candidate = Path(os.path.abspath(output_root))
    lexical_root = Path(os.path.abspath(project_root))
    require(lexical_root in candidate.parents, "formal output root must be inside the project")
    relative = candidate.relative_to(lexical_root).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", relative],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, "formal output root must be covered by the frozen git-ignore policy")


def untracked_evidence_records(project_root: Path) -> list[dict[str, Any]]:
    records = [
        file_record(project_root, project_root / relative, f"public evidence {relative}")
        for relative in ALLOWED_UNTRACKED_EVIDENCE_PATHS
    ]
    require(
        [record["path"] for record in records]
        == list(ALLOWED_UNTRACKED_EVIDENCE_PATHS),
        "public evidence order changed",
    )
    return records


def validate_model_inventory_live(
    project_root: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    payload = load_json_object(inventory_path, "model content inventory")
    required = {
        "protocol",
        "status",
        "dataset_version",
        "model_root",
        "file_count",
        "files",
        "inventory_sha256",
    }
    require(set(payload) == required, "model content inventory schema mismatch")
    require(
        payload["protocol"] == MODEL_INVENTORY_PROTOCOL
        and payload["status"] == "frozen"
        and payload["dataset_version"] == FROZEN_INVENTORY_DATASET_VERSION,
        "model content inventory identity/status mismatch",
    )
    require(payload["model_root"] == "models/Wan2.1-T2V-1.3B-Diffusers", "model root mismatch")
    files = payload["files"]
    require(isinstance(files, list) and files, "model content inventory files are invalid")
    require(payload["file_count"] == len(files), "model content inventory file count mismatch")
    require(
        payload["inventory_sha256"]
        == hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
        "model content inventory internal digest mismatch",
    )
    model_root = project_root / str(payload["model_root"])
    require(model_root.is_dir() and not model_root.is_symlink(), "live model root is missing or symlinked")
    require_no_symlink_components(project_root, model_root, "live model root")
    actual_paths: list[str] = []
    total_bytes = 0
    for candidate in sorted(model_root.rglob("*")):
        info = os.lstat(candidate)
        require(not stat.S_ISLNK(info.st_mode), "live model tree contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "live model tree contains a non-regular file or hardlink")
        actual_paths.append(candidate.relative_to(project_root).as_posix())
        total_bytes += info.st_size
    registered_paths: list[str] = []
    for index, record in enumerate(files):
        require(isinstance(record, dict) and set(record) == {"path", "sha256", "size_bytes"}, f"model file record {index} is invalid")
        relative = record["path"]
        require(isinstance(relative, str) and relative.startswith(str(payload["model_root"]) + "/"), f"model file record {index} escaped model root")
        path = project_root / relative
        regular_file(path, f"model file {index}")
        require(
            project_relative(project_root, path, f"model file {index}") == relative,
            f"model file record {index} is not a canonical project-relative path",
        )
        require_no_symlink_components(project_root, path, f"model file {index}")
        require(path.stat().st_nlink == 1, f"model file {index} is hardlinked")
        require(path.stat().st_size == record["size_bytes"], f"model file {index} size mismatch")
        require(sha256_file(path) == record["sha256"], f"model file {index} SHA-256 mismatch")
        registered_paths.append(relative)
    require(registered_paths == sorted(set(registered_paths)), "model inventory paths are not unique and sorted")
    require(registered_paths == actual_paths, "model inventory does not cover the exact live model tree")
    return {
        "registry": file_record(project_root, inventory_path, "model content inventory"),
        "model_root": str(payload["model_root"]),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "content_inventory_sha256": str(payload["inventory_sha256"]),
        "stat_seal": tree_stat_seal(model_root),
    }


def tree_stat_seal(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"stat-seal root is missing or symlinked: {root}")
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), f"stat-seal tree contains a symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, f"stat-seal tree contains a non-regular file or hardlink: {path}")
        record = {
            "path": path.relative_to(root).as_posix(),
            "device": info.st_dev,
            "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
            "size_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
        digest.update(canonical_json_bytes(record))
        file_count += 1
    require(file_count > 0, "stat-seal tree is empty")
    return {
        "algorithm": STAT_SEAL_ALGORITHM,
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def runtime_content_inventory(runtime_root: Path) -> dict[str, Any]:
    require(runtime_root.is_dir() and not runtime_root.is_symlink(), "live runtime root is missing or symlinked")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(runtime_root.rglob("*")):
        info = os.lstat(path)
        require(not stat.S_ISLNK(info.st_mode), "live runtime contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "live runtime contains a non-regular file or hardlink")
        relative = path.relative_to(runtime_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            observed = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                observed += len(chunk)
        require(observed == info.st_size, "runtime file changed while hashing")
        digest.update(b"\n")
        file_count += 1
        total_bytes += info.st_size
    require(file_count > 0 and total_bytes > 0, "live runtime is empty")
    return {
        "content_inventory_algorithm": RUNTIME_CONTENT_INVENTORY_ALGORITHM,
        "content_file_count": file_count,
        "content_total_bytes": total_bytes,
        "content_inventory_sha256": digest.hexdigest(),
    }


def validate_runtime_registry_live(
    project_root: Path,
    registry_path: Path,
) -> dict[str, Any]:
    payload = load_json_object(registry_path, "runtime registry")
    required = {
        "protocol",
        "status",
        "dataset_version",
        "runtime_root",
        "python_executable",
        "sys_prefix_policy",
        "python",
        "torch",
        "cuda",
        "packages",
        "content_inventory_algorithm",
        "content_file_count",
        "content_total_bytes",
        "content_inventory_sha256",
        "module_origins",
    }
    require(set(payload) == required, "runtime registry schema mismatch")
    require(
        payload["protocol"] == RUNTIME_REGISTRY_PROTOCOL
        and payload["status"] == "frozen"
        and payload["dataset_version"] == FROZEN_INVENTORY_DATASET_VERSION,
        "runtime registry identity/status mismatch",
    )
    require(payload["runtime_root"] == "models/.wan-runtime", "runtime root mismatch")
    require(payload["python_executable"] == "models/.wan-runtime/bin/python", "runtime Python path mismatch")
    live_runtime_root = project_root / str(payload["runtime_root"])
    require_no_symlink_components(project_root, live_runtime_root, "live runtime root")
    live = runtime_content_inventory(live_runtime_root)
    for field, value in live.items():
        require(payload[field] == value, f"runtime registry live {field} mismatch")
    executable = project_root / str(payload["python_executable"])
    regular_file(executable, "runtime Python")
    require_no_symlink_components(project_root, executable, "runtime Python")
    require(executable.stat().st_nlink == 1 and os.access(executable, os.X_OK), "runtime Python is hardlinked or non-executable")
    return {
        "registry": file_record(project_root, registry_path, "runtime registry"),
        "runtime_root": str(payload["runtime_root"]),
        "python_executable": str(payload["python_executable"]),
        **live,
        "stat_seal": tree_stat_seal(live_runtime_root),
    }


def generation_contract() -> dict[str, Any]:
    return {
        "baseline": "clean",
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
        "model_cpu_offload": False,
        "sequential_cpu_offload": False,
        "skip_existing": False,
        "seed_source": "192 explicit seeds in canonical order",
    }


def stage_artifact_records(project_root: Path) -> list[dict[str, Any]]:
    records = [
        file_record(project_root, project_root / relative, f"frozen stage artifact {relative}")
        for relative in FROZEN_CODE_AND_DATA_PATHS
    ]
    require([record["path"] for record in records] == list(FROZEN_CODE_AND_DATA_PATHS), "stage artifact order changed")
    hashes = {record["path"]: record["sha256"] for record in records}
    require(hashes[DEFAULT_CANONICAL_MANIFEST.as_posix()] == EXPECTED_CANONICAL_MANIFEST_SHA256, "canonical manifest hash changed")
    require(hashes[DEFAULT_CSV_MANIFEST.as_posix()] == EXPECTED_CSV_SHA256, "CSV manifest hash changed")
    require(hashes[DEFAULT_SUMMARY.as_posix()] == EXPECTED_SUMMARY_SHA256, "summary hash changed")
    require(hashes[DEFAULT_PROMPTS.as_posix()] == EXPECTED_PROMPTS_SHA256, "prompt hash changed")
    require(hashes["scripts/generate_wan_clean.py"] == EXPECTED_GENERATOR_SHA256, "Wan generator hash changed")
    return records


def seed_registry(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    seeds = [int(row["seed"]) for row in rows]
    require(len(seeds) == EXPECTED_ROWS and len(set(seeds)) == EXPECTED_ROWS, "seed registry is not 192 unique seeds")
    return {
        "source": DEFAULT_CANONICAL_MANIFEST.as_posix(),
        "count": EXPECTED_ROWS,
        "order": "canonical row order",
        "algorithm": "840000 + 1000*mechanism_index + 10*combination_index + repetition_index",
        "ordered_seed_list_sha256": hashlib.sha256(canonical_json_bytes(seeds)).hexdigest(),
    }


def build_stage_registry_payload(
    *,
    project_root: Path,
    stage_registry: Path,
    model_inventory: Path,
    runtime_registry: Path,
    rows: Sequence[Mapping[str, str]],
    prepared_at_utc: str,
    registry_already_exists: bool,
) -> dict[str, Any]:
    reject_sealed_path(stage_registry, model_inventory, runtime_registry)
    git = git_snapshot(
        project_root,
        allowed_stage_registry=stage_registry if registry_already_exists else None,
    )
    model_binding = validate_model_inventory_live(project_root, model_inventory)
    runtime_binding = validate_runtime_registry_live(project_root, runtime_registry)
    return {
        "schema_version": 1,
        "registry_type": "causal_role_erasure_8mechanism_capability_stage_v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": "authorized_for_original_capability_generation",
        "prepared_at_utc": prepared_at_utc,
        "sealed_final36_status": "unopened",
        "sealed_final36_access_policy": "operator-attested; no sealed/final36 path is accepted or read",
        "git": git,
        "untracked_evidence": untracked_evidence_records(project_root),
        "frozen_artifacts": stage_artifact_records(project_root),
        "model_content_inventory": model_binding,
        "runtime_registry": runtime_binding,
        "seed_registry": seed_registry(rows),
        "generation": generation_contract(),
        "scheduler": {
            "mechanism_order": list(MECHANISM_ORDER),
            "rows_per_mechanism": ROWS_PER_MECHANISM,
            "total_rows": EXPECTED_ROWS,
            "gpu_count": GPUS_REQUIRED,
            "waves": WAVES,
            "one_long_lived_generator_process_per_mechanism": True,
            "wave_1_requires_wave_0_success": True,
        },
        "media_validation": {
            "implementation": "PyAV full decode in the frozen Wan runtime",
            "runtime_python": runtime_binding["python_executable"],
            "decode_count_frames": True,
            "required_video_streams": 1,
            "required_frames": NUM_FRAMES,
            "required_fps": FPS,
            "required_height": HEIGHT,
            "required_width": WIDTH,
        },
        "authorization": {
            "original_only": True,
            "training_authorized": False,
            "treatment_generation_authorized": False,
            "output_root_must_not_exist": True,
            "skip_existing_forbidden": True,
        },
    }


def prepare_stage_registry(
    *,
    project_root: Path,
    stage_registry: Path,
    model_inventory: Path,
    runtime_registry: Path,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    require(not stage_registry.exists() and not stage_registry.is_symlink(), f"stage registry already exists: {stage_registry}")
    payload = build_stage_registry_payload(
        project_root=project_root,
        stage_registry=stage_registry,
        model_inventory=model_inventory,
        runtime_registry=runtime_registry,
        rows=rows,
        prepared_at_utc=utc_now(),
        registry_already_exists=False,
    )
    stage_registry.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_exclusive(
        stage_registry,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return payload


def reopen_and_validate_stage_registry(
    *,
    project_root: Path,
    stage_registry: Path,
    expected_sha256: str,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    require_sha256(expected_sha256, "expected stage-registry SHA-256")
    regular_file(stage_registry, "capability stage registry")
    require(sha256_file(stage_registry) == expected_sha256, "capability stage-registry SHA-256 mismatch")
    payload = load_json_object(stage_registry, "capability stage registry")
    require(
        set(payload)
        == {
            "schema_version",
            "registry_type",
            "protocol_version",
            "status",
            "prepared_at_utc",
            "sealed_final36_status",
            "sealed_final36_access_policy",
            "git",
            "untracked_evidence",
            "frozen_artifacts",
            "model_content_inventory",
            "runtime_registry",
            "seed_registry",
            "generation",
            "scheduler",
            "media_validation",
            "authorization",
        },
        "capability stage-registry schema mismatch",
    )
    require(payload["schema_version"] == 1, "stage-registry schema version mismatch")
    require(payload["protocol_version"] == PROTOCOL_VERSION, "stage-registry protocol mismatch")
    require(payload["status"] == "authorized_for_original_capability_generation", "stage registry is not authorized")
    require(payload["sealed_final36_status"] == "unopened", "stage registry does not keep sealed-final36 unopened")
    model_inventory = project_root / payload["model_content_inventory"]["registry"]["path"]
    runtime_registry = project_root / payload["runtime_registry"]["registry"]["path"]
    require(
        model_inventory.resolve(strict=True)
        == (project_root / DEFAULT_MODEL_INVENTORY).resolve(strict=True),
        "stage registry model-inventory path is not standard",
    )
    require(
        runtime_registry.resolve(strict=True)
        == (project_root / DEFAULT_RUNTIME_REGISTRY).resolve(strict=True),
        "stage registry runtime-registry path is not standard",
    )
    expected = build_stage_registry_payload(
        project_root=project_root,
        stage_registry=stage_registry,
        model_inventory=model_inventory,
        runtime_registry=runtime_registry,
        rows=rows,
        prepared_at_utc=str(payload["prepared_at_utc"]),
        registry_already_exists=True,
    )
    require(payload == expected, "stage registry no longer matches live code/data/model/runtime/git state")
    return payload


def quick_revalidate_stage_registry(
    *,
    project_root: Path,
    stage_registry: Path,
    expected_sha256: str,
) -> None:
    """Check wave-boundary TOCTOU seals without rereading all model/runtime bytes."""

    require_sha256(expected_sha256, "expected stage-registry SHA-256")
    regular_file(stage_registry, "capability stage registry")
    require(sha256_file(stage_registry) == expected_sha256, "capability stage-registry SHA-256 mismatch")
    payload = load_json_object(stage_registry, "capability stage registry")
    require(
        payload.get("status") == "authorized_for_original_capability_generation"
        and payload.get("sealed_final36_status") == "unopened",
        "capability stage registry is not authorized/unopened",
    )
    require(
        git_snapshot(project_root, allowed_stage_registry=stage_registry)
        == payload.get("git"),
        "git HEAD/worktree changed after stage freeze",
    )
    require(
        untracked_evidence_records(project_root)
        == payload.get("untracked_evidence"),
        "public evidence changed after stage freeze",
    )
    require(
        stage_artifact_records(project_root) == payload.get("frozen_artifacts"),
        "frozen code/data/review artifact changed after stage freeze",
    )

    model_binding = payload.get("model_content_inventory")
    runtime_binding = payload.get("runtime_registry")
    require(isinstance(model_binding, dict) and isinstance(runtime_binding, dict), "stage live bindings are malformed")
    model_registry = project_root / model_binding["registry"]["path"]
    runtime_registry = project_root / runtime_binding["registry"]["path"]
    require(
        file_record(project_root, model_registry, "model content inventory")
        == model_binding["registry"],
        "model content-inventory registry changed after stage freeze",
    )
    require(
        file_record(project_root, runtime_registry, "runtime registry")
        == runtime_binding["registry"],
        "runtime registry changed after stage freeze",
    )
    model_root = project_root / model_binding["model_root"]
    runtime_root = project_root / runtime_binding["runtime_root"]
    require_no_symlink_components(project_root, model_root, "live model root")
    require_no_symlink_components(project_root, runtime_root, "live runtime root")
    require(
        tree_stat_seal(model_root) == model_binding["stat_seal"],
        "live model stat seal changed after stage freeze",
    )
    require(
        tree_stat_seal(runtime_root) == runtime_binding["stat_seal"],
        "live runtime stat seal changed after stage freeze",
    )


def parse_prompt_file_strict(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        require(
            len(parts) == 3 and all(parts),
            f"{path}:{line_no}: expected '<prompt> | <target> | <effect>'",
        )
        prompt, target_concept, expected_effect = parts
        items.append(
            {
                "prompt": prompt,
                "target_concept": target_concept,
                "expected_effect": expected_effect,
            }
        )
    return items


def expected_seed(mechanism_index: int, combination_index: int, repetition_index: int) -> int:
    return 840000 + 1000 * mechanism_index + 10 * combination_index + repetition_index


def load_frozen_inputs(
    canonical_manifest: Path,
    prompts: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    regular_file(canonical_manifest, "canonical manifest")
    regular_file(prompts, "prompt file")
    require(
        sha256_file(canonical_manifest) == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "canonical manifest SHA-256 mismatch; refusing to plan or run",
    )
    require(
        sha256_file(prompts) == EXPECTED_PROMPTS_SHA256,
        "prompt file SHA-256 mismatch; refusing to plan or run",
    )
    try:
        raw_rows = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse canonical manifest: {canonical_manifest}") from exc
    require(isinstance(raw_rows, list), "canonical manifest must be a JSON array")
    require(len(raw_rows) == EXPECTED_ROWS, f"canonical manifest must contain exactly {EXPECTED_ROWS} rows")
    require(all(isinstance(row, dict) for row in raw_rows), "canonical manifest rows must be objects")
    rows: list[dict[str, str]] = [dict(row) for row in raw_rows]
    prompt_items = parse_prompt_file_strict(prompts)
    require(len(prompt_items) == EXPECTED_ROWS, f"prompt file must contain exactly {EXPECTED_ROWS} items")

    seen_generation_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for global_index, (row, prompt_item) in enumerate(zip(rows, prompt_items)):
        require(set(row) == EXPECTED_ROW_FIELDS, f"row {global_index}: canonical field schema mismatch")
        require(all(isinstance(value, str) for value in row.values()), f"row {global_index}: every field must be a string")
        mechanism_index = global_index // ROWS_PER_MECHANISM
        local_index = global_index % ROWS_PER_MECHANISM
        combination_index = local_index // 3
        repetition_index = local_index % 3
        mechanism = MECHANISM_ORDER[mechanism_index]
        case_id = f"cap8m{mechanism_index:02d}c{combination_index:02d}"
        generation_id = f"{case_id}r{repetition_index:02d}"
        seed = expected_seed(mechanism_index, combination_index, repetition_index)

        require(row["protocol_version"] == PROTOCOL_VERSION, f"row {global_index}: protocol mismatch")
        require(row["mechanism_index"] == str(mechanism_index), f"row {global_index}: mechanism index/order mismatch")
        require(row["mechanism"] == mechanism, f"row {global_index}: mechanism order is not contiguous and canonical")
        require(row["combination_index"] == str(combination_index), f"row {global_index}: combination order mismatch")
        require(row["repetition_index"] == str(repetition_index), f"row {global_index}: repetition order mismatch")
        require(row["case_id"] == case_id, f"row {global_index}: case ID mismatch")
        require(row["generation_id"] == generation_id, f"row {global_index}: generation ID mismatch")
        require(int(row["seed"]) == seed, f"row {global_index}: explicit seed mismatch")
        require(row["method_arm"] == "original", f"row {global_index}: non-Original arm is forbidden")
        require(row["treatment_status"] == "pre_method_original_only", f"row {global_index}: treatment row is forbidden")
        require(row["intended_use"] == "original_capability_screening_only", f"row {global_index}: use mismatch")
        require(row["num_frames"] == str(NUM_FRAMES), f"row {global_index}: frame count mismatch")
        require(row["fps"] == str(FPS), f"row {global_index}: FPS mismatch")
        require(row["reference_start_inclusive"] == "0", f"row {global_index}: reference start mismatch")
        require(row["reference_end_exclusive"] == "16", f"row {global_index}: reference end mismatch")
        require(
            prompt_item
            == {
                "prompt": row["prompt"],
                "target_concept": row["target_concept"],
                "expected_effect": row["expected_footprint"],
            },
            f"row {global_index}: prompt-file order/content differs from canonical manifest",
        )
        require(generation_id not in seen_generation_ids, f"row {global_index}: duplicate generation ID")
        require(seed not in seen_seeds, f"row {global_index}: duplicate seed")
        seen_generation_ids.add(generation_id)
        seen_seeds.add(seed)

    require(
        Counter(row["mechanism"] for row in rows)
        == Counter({mechanism: ROWS_PER_MECHANISM for mechanism in MECHANISM_ORDER}),
        "manifest is not balanced at 24 contiguous rows per mechanism",
    )
    return rows, prompt_items


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must contain four comma-separated integers") from exc
    if len(gpus) != GPUS_REQUIRED or len(set(gpus)) != GPUS_REQUIRED or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("--gpus must contain exactly four distinct non-negative GPU indices")
    return gpus


def prompt_shard_bytes(mechanism: str, prompt_items: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        f"# {PROTOCOL_VERSION}: frozen 24-row shard for {mechanism}",
        "# Order and values are bound to the canonical 192-row manifest.",
        "",
    ]
    lines.extend(
        f"{item['prompt']} | {item['target_concept']} | {item['expected_effect']}"
        for item in prompt_items
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_generator_command(
    *,
    python_executable: Path,
    generator: Path,
    prompt_shard: Path,
    output_dir: Path,
    model: Path,
    seeds: Sequence[int],
) -> list[str]:
    require(len(seeds) == ROWS_PER_MECHANISM, "each generator process must receive exactly 24 explicit seeds")
    return [
        str(python_executable),
        str(generator),
        "--baseline",
        "clean",
        "--prompts",
        str(prompt_shard),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--seed",
        str(seeds[0]),
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


def build_plan(
    *,
    rows: Sequence[Mapping[str, str]],
    prompt_items: Sequence[Mapping[str, str]],
    project_root: Path,
    canonical_manifest: Path,
    prompts: Path,
    output_root: Path,
    generator: Path,
    python_executable: Path,
    model: Path,
    gpus: Sequence[int],
    dry_run: bool,
    stage_registry: Path | None,
    stage_registry_sha256: str | None,
) -> dict[str, Any]:
    require(len(gpus) == GPUS_REQUIRED and len(set(gpus)) == GPUS_REQUIRED, "exactly four unique GPUs are required")
    jobs: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        start = mechanism_index * ROWS_PER_MECHANISM
        end = start + ROWS_PER_MECHANISM
        mechanism_rows = rows[start:end]
        mechanism_prompts = prompt_items[start:end]
        shard_path = output_root / "prompt_shards" / f"{mechanism_index:02d}_{mechanism}.prompts"
        raw_shard = prompt_shard_bytes(mechanism, mechanism_prompts)
        seeds = [int(row["seed"]) for row in mechanism_rows]
        output_dir = output_root / "mechanisms" / f"{mechanism_index:02d}_{mechanism}"
        wave_index = mechanism_index // GPUS_REQUIRED
        gpu = int(gpus[mechanism_index % GPUS_REQUIRED])
        job_id = f"wave{wave_index}_{mechanism_index:02d}_{mechanism}"
        command = build_generator_command(
            python_executable=python_executable,
            generator=generator,
            prompt_shard=shard_path,
            output_dir=output_dir,
            model=model,
            seeds=seeds,
        )
        require("--skip-existing" not in command, "capability generation must never skip existing outputs")
        jobs.append(
            {
                "job_id": job_id,
                "wave_index": wave_index,
                "gpu": gpu,
                "mechanism_index": mechanism_index,
                "mechanism": mechanism,
                "canonical_row_start_inclusive": start,
                "canonical_row_end_exclusive": end,
                "row_count": ROWS_PER_MECHANISM,
                "generation_ids": [str(row["generation_id"]) for row in mechanism_rows],
                "seeds": seeds,
                "prompt_shard": str(shard_path),
                "prompt_shard_sha256": hashlib.sha256(raw_shard).hexdigest(),
                "output_dir": str(output_dir),
                "log_path": str(output_root / "logs" / f"{job_id}.log"),
                "status_path": str(output_root / "statuses" / f"{job_id}.json"),
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONSAFEPATH": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                },
                "unset_environment": ["PYTHONHOME", "PYTHONPATH"],
                "command": command,
                "planned_status": "planned",
            }
        )
    require(
        Counter(job["wave_index"] for job in jobs) == Counter({0: 4, 1: 4}),
        "scheduler must contain exactly two four-process waves",
    )
    return {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "dry_run": dry_run,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "stage_binding": (
            None
            if stage_registry is None
            else {
                "path": str(stage_registry),
                "sha256": stage_registry_sha256,
                "status": "reopened_and_live_state_rehashed_before_output_reservation",
            }
        ),
        "inputs": {
            "canonical_manifest": str(canonical_manifest),
            "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "prompts": str(prompts),
            "prompts_sha256": EXPECTED_PROMPTS_SHA256,
            "rows": EXPECTED_ROWS,
            "mechanism_order": list(MECHANISM_ORDER),
        },
        "implementation": {
            "generator": str(generator),
            "generator_sha256": EXPECTED_GENERATOR_SHA256,
            "python_executable": str(python_executable),
            "model": str(model),
            "media_probe_python": str(python_executable),
        },
        "generation": {
            "baseline": "clean",
            "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "height": HEIGHT,
            "width": WIDTH,
            "dtype": DTYPE,
            "device_inside_isolated_process": "cuda",
            "vae_slicing": True,
            "vae_tiling": True,
            "model_cpu_offload": False,
            "sequential_cpu_offload": False,
            "skip_existing": False,
            "per_prompt_seeds": "explicit_from_canonical_manifest",
            "post_generation_media_probe": "decode-count exact frame/fps/resolution validation",
        },
        "scheduler": {
            "gpus": list(gpus),
            "waves": WAVES,
            "processes_per_wave": GPUS_REQUIRED,
            "prompts_per_process": ROWS_PER_MECHANISM,
            "pipeline_lifetime": "one_generate_wan_clean_process_per_mechanism",
            "wave_barrier": "next_wave_requires_all_prior_wave_jobs_valid",
            "failure_policy": "fail_closed_no_resume_no_skip",
        },
        "jobs": jobs,
    }


def write_bytes_exclusive(path: Path, raw: bytes) -> None:
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
    raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_bytes_exclusive(temporary, raw)
    os.replace(temporary, path)


def prepare_output_root(
    output_root: Path,
    plan: Mapping[str, Any],
    prompt_items: Sequence[Mapping[str, str]],
) -> None:
    require(not output_root.exists() and not output_root.is_symlink(), f"output root must not exist: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    require(output_root.parent.is_dir() and not output_root.parent.is_symlink(), "output-root parent must be a regular directory")
    os.mkdir(output_root, 0o755)
    for name in ("prompt_shards", "logs", "statuses", "mechanisms"):
        (output_root / name).mkdir()

    for job in plan["jobs"]:
        start = int(job["canonical_row_start_inclusive"])
        end = int(job["canonical_row_end_exclusive"])
        shard_path = Path(str(job["prompt_shard"]))
        raw = prompt_shard_bytes(str(job["mechanism"]), prompt_items[start:end])
        require(hashlib.sha256(raw).hexdigest() == job["prompt_shard_sha256"], "planned shard hash mismatch")
        write_bytes_exclusive(shard_path, raw)
        write_json_atomic(
            Path(str(job["status_path"])),
            {
                "schema_version": 1,
                "job_id": job["job_id"],
                "mechanism": job["mechanism"],
                "wave_index": job["wave_index"],
                "gpu": job["gpu"],
                "status": "planned",
                "updated_at_utc": utc_now(),
                "expected_videos": ROWS_PER_MECHANISM,
            },
        )
    write_bytes_exclusive(
        output_root / "capability_run_manifest.json",
        (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    write_aggregate(output_root, plan, overall_status="planned")


def read_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(job["status_path"])).read_text(encoding="utf-8"))


def write_aggregate(
    output_root: Path,
    plan: Mapping[str, Any],
    *,
    overall_status: str,
    error: str | None = None,
) -> None:
    statuses = [read_status(job) for job in plan["jobs"]]
    counts = Counter(str(status["status"]) for status in statuses)
    completed_videos = sum(int(status.get("validated_video_count", 0)) for status in statuses)
    completed_bytes = sum(int(status.get("validated_video_bytes", 0)) for status in statuses)
    frozen_generation_manifest = output_root / "capability_generation_manifest.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "runner_version": RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "updated_at_utc": utc_now(),
        "status": overall_status,
        "dry_run": bool(plan["dry_run"]),
        "expected_mechanisms": len(MECHANISM_ORDER),
        "expected_videos": EXPECTED_ROWS,
        "validated_videos": completed_videos,
        "validated_video_bytes": completed_bytes,
        "status_counts": dict(sorted(counts.items())),
        "input_hashes": {
            "canonical_manifest": EXPECTED_CANONICAL_MANIFEST_SHA256,
            "prompts": EXPECTED_PROMPTS_SHA256,
            "generator": EXPECTED_GENERATOR_SHA256,
            "stage_registry": (
                None
                if plan.get("stage_binding") is None
                else plan["stage_binding"]["sha256"]
            ),
        },
        "media_validation": {
            "probe": "PyAV full decode via " + plan["implementation"]["media_probe_python"],
            "decoded_frames": NUM_FRAMES,
            "fps": FPS,
            "height": HEIGHT,
            "width": WIDTH,
        },
        "frozen_generation_manifest": (
            None
            if not frozen_generation_manifest.is_file()
            else {
                "path": str(frozen_generation_manifest),
                "sha256": sha256_file(frozen_generation_manifest),
            }
        ),
        "mechanisms": [
            {
                "mechanism": status["mechanism"],
                "wave_index": status["wave_index"],
                "gpu": status["gpu"],
                "status": status["status"],
                "return_code": status.get("return_code"),
                "validated_video_count": status.get("validated_video_count", 0),
                "generation_manifest_sha256": status.get("generation_manifest_sha256"),
                "error": status.get("error"),
            }
            for status in statuses
        ],
    }
    if error is not None:
        payload["error"] = error
    write_json_atomic(output_root / "capability_run_aggregate.json", payload)


def validate_bound_files(plan: Mapping[str, Any], *, require_runtime: bool) -> None:
    inputs = plan["inputs"]
    implementation = plan["implementation"]
    canonical_manifest = Path(str(inputs["canonical_manifest"]))
    prompts = Path(str(inputs["prompts"]))
    generator = Path(str(implementation["generator"]))
    regular_file(canonical_manifest, "canonical manifest")
    regular_file(prompts, "prompt file")
    regular_file(generator, "Wan generator")
    require(sha256_file(canonical_manifest) == EXPECTED_CANONICAL_MANIFEST_SHA256, "canonical manifest changed after planning")
    require(sha256_file(prompts) == EXPECTED_PROMPTS_SHA256, "prompt file changed after planning")
    require(sha256_file(generator) == EXPECTED_GENERATOR_SHA256, "Wan generator changed after planning")
    stage_binding = plan.get("stage_binding")
    if stage_binding is not None:
        stage_registry = Path(str(stage_binding["path"]))
        regular_file(stage_registry, "capability stage registry")
        require(
            sha256_file(stage_registry) == stage_binding["sha256"],
            "capability stage registry changed after planning",
        )
    for job in plan["jobs"]:
        shard = Path(str(job["prompt_shard"]))
        regular_file(shard, f"{job['mechanism']} prompt shard")
        require(sha256_file(shard) == job["prompt_shard_sha256"], f"{job['mechanism']}: prompt shard changed")
    if require_runtime:
        validate_runtime_dependencies(plan)


def validate_runtime_dependencies(plan: Mapping[str, Any]) -> None:
    implementation = plan["implementation"]
    python_executable = Path(str(implementation["python_executable"]))
    model = Path(str(implementation["model"]))
    regular_file(python_executable, "Wan runtime Python")
    require(os.access(python_executable, os.X_OK), f"Wan runtime Python is not executable: {python_executable}")
    require(model.is_dir() and not model.is_symlink(), f"Wan model root is not a regular non-symlink directory: {model}")


def _expected_generation_fields() -> dict[str, Any]:
    return {
        "baseline": "clean",
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
        "lora_path": None,
        "activation_gate_dir": None,
        "attention_gate_dir": None,
    }


def probe_video_media(runtime_python: Path, video_path: Path) -> dict[str, Any]:
    command = [
        str(runtime_python),
        "-c",
        PYAV_PROBE_CODE,
        str(video_path),
    ]
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    require(result.returncode == 0, f"PyAV could not fully decode {video_path.name}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"PyAV returned invalid JSON for {video_path.name}") from exc
    require(isinstance(payload, dict), f"{video_path.name}: invalid PyAV probe payload")
    require(payload.get("streams") == 1, f"{video_path.name}: expected exactly one video stream")
    require(payload.get("width") == WIDTH and payload.get("height") == HEIGHT, f"{video_path.name}: expected {WIDTH}x{HEIGHT}")
    frames = payload.get("decoded_frames")
    require(type(frames) is int, f"{video_path.name}: PyAV did not report a frame count")
    require(frames == NUM_FRAMES, f"{video_path.name}: expected {NUM_FRAMES} decoded frames, got {frames}")
    raw_rate = payload.get("fps")
    try:
        rate = Fraction(str(raw_rate))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{video_path.name}: PyAV reported an invalid frame rate") from exc
    require(rate == Fraction(FPS, 1), f"{video_path.name}: expected {FPS} fps, got {raw_rate}")
    return {
        "decoded_frames": frames,
        "fps": f"{rate.numerator}/{rate.denominator}",
        "height": int(payload["height"]),
        "width": int(payload["width"]),
    }


def validate_job_outputs(
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"]))
    generation_manifest = output_dir / "generation_manifest.json"
    regular_file(generation_manifest, f"{job['mechanism']} generation manifest")
    try:
        payload = json.loads(generation_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{job['mechanism']}: invalid generation manifest") from exc
    require(isinstance(payload, dict), f"{job['mechanism']}: generation manifest must be an object")
    require(payload.get("baseline") == "clean", f"{job['mechanism']}: generation baseline mismatch")
    require(payload.get("pipeline") == "WanPipeline", f"{job['mechanism']}: pipeline mismatch")
    require(payload.get("model") == plan["implementation"]["model"], f"{job['mechanism']}: model path mismatch")
    require(payload.get("dry_run") is False, f"{job['mechanism']}: dry-run output cannot satisfy a real run")
    require(payload.get("prompts") == job["prompt_shard"], f"{job['mechanism']}: prompt shard binding mismatch")
    generation = payload.get("generation")
    require(isinstance(generation, dict), f"{job['mechanism']}: generation config is missing")
    for field, expected in _expected_generation_fields().items():
        require(generation.get(field) == expected, f"{job['mechanism']}: generation.{field} mismatch")
    require(generation.get("seeds") == job["seeds"], f"{job['mechanism']}: explicit seed list mismatch")

    items = payload.get("items")
    require(isinstance(items, list) and len(items) == ROWS_PER_MECHANISM, f"{job['mechanism']}: expected 24 generated items")
    shard_items = parse_prompt_file_strict(Path(str(job["prompt_shard"])))
    outputs: list[dict[str, Any]] = []
    listed_paths: set[Path] = set()
    videos_dir = output_dir / "videos"
    require(videos_dir.is_dir() and not videos_dir.is_symlink(), f"{job['mechanism']}: videos directory is missing or symlinked")
    for local_index, (item, prompt_item) in enumerate(zip(items, shard_items)):
        require(isinstance(item, dict), f"{job['mechanism']}: item {local_index} is not an object")
        require(item.get("index") == local_index, f"{job['mechanism']}: output item order mismatch")
        require(item.get("prompt") == prompt_item["prompt"], f"{job['mechanism']}: output prompt mismatch")
        require(item.get("target_concept") == prompt_item["target_concept"], f"{job['mechanism']}: output target mismatch")
        require(item.get("expected_effect") == prompt_item["expected_effect"], f"{job['mechanism']}: output effect mismatch")
        require(item.get("seed") == job["seeds"][local_index], f"{job['mechanism']}: output seed mismatch")
        video_path = Path(str(item.get("video_path", "")))
        regular_file(video_path, f"{job['mechanism']} video {local_index}")
        require(video_path.parent == videos_dir, f"{job['mechanism']}: video path escaped its output directory")
        require(video_path.suffix.lower() == ".mp4", f"{job['mechanism']}: output is not MP4")
        size = video_path.stat().st_size
        require(size > 0, f"{job['mechanism']}: empty video {local_index}")
        require(video_path not in listed_paths, f"{job['mechanism']}: duplicate output video path")
        listed_paths.add(video_path)
        media = media_probe(
            Path(str(plan["implementation"]["media_probe_python"])),
            video_path,
        )
        require(
            media
            == {
                "decoded_frames": NUM_FRAMES,
                "fps": f"{FPS}/1",
                "height": HEIGHT,
                "width": WIDTH,
            },
            f"{job['mechanism']}: video {local_index} media probe contract mismatch",
        )
        outputs.append(
            {
                "canonical_row_index": int(job["canonical_row_start_inclusive"]) + local_index,
                "generation_id": job["generation_ids"][local_index],
                "seed": item["seed"],
                "video_path": str(video_path),
                "video_sha256": sha256_file(video_path),
                "bytes": size,
                "media": media,
            }
        )
    actual_video_entries = set(videos_dir.iterdir())
    require(actual_video_entries == listed_paths, f"{job['mechanism']}: videos directory contains missing or extra entries")
    require(set(output_dir.iterdir()) == {generation_manifest, videos_dir}, f"{job['mechanism']}: output directory contains unexpected entries")
    return {
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "validated_video_count": len(outputs),
        "validated_video_bytes": sum(int(output["bytes"]) for output in outputs),
        "outputs": outputs,
    }


def _status_payload(job: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "mechanism": job["mechanism"],
        "wave_index": job["wave_index"],
        "gpu": job["gpu"],
        "status": status,
        "updated_at_utc": utc_now(),
        "expected_videos": ROWS_PER_MECHANISM,
    }
    payload.update(extra)
    return payload


def write_generation_manifest(output_root: Path, plan: Mapping[str, Any]) -> Path:
    items: list[dict[str, Any]] = []
    mechanism_manifests: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        status = read_status(job)
        require(status["status"] == "completed", "cannot freeze generation manifest before all mechanisms complete")
        mechanism_manifests.append(
            {
                "mechanism": job["mechanism"],
                "generation_manifest": str(Path(str(job["output_dir"])) / "generation_manifest.json"),
                "generation_manifest_sha256": status["generation_manifest_sha256"],
            }
        )
        for output in status["outputs"]:
            items.append(
                {
                    "canonical_row_index": output["canonical_row_index"],
                    "generation_id": output["generation_id"],
                    "mechanism": job["mechanism"],
                    "seed": output["seed"],
                    "video_path": output["video_path"],
                    "video_sha256": output["video_sha256"],
                    "size_bytes": output["bytes"],
                    "media": output["media"],
                }
            )
    items.sort(key=lambda item: int(item["canonical_row_index"]))
    require([item["canonical_row_index"] for item in items] == list(range(EXPECTED_ROWS)), "generation manifest row order is not canonical")
    require(len({item["generation_id"] for item in items}) == EXPECTED_ROWS, "generation manifest IDs are not unique")
    require(len({item["video_path"] for item in items}) == EXPECTED_ROWS, "generation manifest paths are not unique")
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_after_exact_media_validation",
        "stage_registry_sha256": plan["stage_binding"]["sha256"],
        "canonical_manifest_sha256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        "prompts_sha256": EXPECTED_PROMPTS_SHA256,
        "generator_sha256": EXPECTED_GENERATOR_SHA256,
        "generation": plan["generation"],
        "mechanism_manifests": mechanism_manifests,
        "video_binding_key": "generation_id",
        "video_count": EXPECTED_ROWS,
        "items": items,
    }
    path = output_root / "capability_generation_manifest.json"
    write_bytes_exclusive(path, canonical_json_bytes(payload))
    return path


def revalidate_completed_outputs(
    plan: Mapping[str, Any],
    *,
    media_probe: Callable[[Path, Path], dict[str, Any]],
) -> None:
    for job in plan["jobs"]:
        status = read_status(job)
        require(status["status"] == "completed", f"{job['mechanism']}: status is not completed at final freeze")
        observed = validate_job_outputs(job, plan, media_probe=media_probe)
        frozen = {
            key: status[key]
            for key in (
                "generation_manifest_sha256",
                "validated_video_count",
                "validated_video_bytes",
                "outputs",
            )
        }
        require(
            observed == frozen,
            f"{job['mechanism']}: generated media/manifest drifted after initial validation",
        )


def terminate_processes(running: Mapping[str, tuple[Mapping[str, Any], Any, Any]]) -> None:
    for _, process, _ in running.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for _, process, _ in running.values():
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def execute_plan(
    plan: Mapping[str, Any],
    output_root: Path,
    *,
    poll_interval: float,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    media_probe: Callable[[Path, Path], dict[str, Any]] = probe_video_media,
    stage_revalidator: Callable[[], None] | None = None,
) -> None:
    require(plan["dry_run"] is False, "cannot execute a dry-run plan")
    require(plan.get("stage_binding") is not None, "formal execution requires a frozen stage binding")
    require(stage_revalidator is not None, "formal execution requires a live stage revalidator")
    validate_bound_files(plan, require_runtime=True)
    write_aggregate(output_root, plan, overall_status="running")
    for wave_index in range(WAVES):
        wave_jobs = [job for job in plan["jobs"] if job["wave_index"] == wave_index]
        require(len(wave_jobs) == GPUS_REQUIRED, f"wave {wave_index}: expected four jobs")
        try:
            stage_revalidator()
            validate_bound_files(plan, require_runtime=True)
        except BaseException as exc:
            message = f"stage revalidation failed before wave {wave_index}: {type(exc).__name__}: {exc}"
            write_aggregate(output_root, plan, overall_status="failed", error=message)
            raise RuntimeError(message) from exc
        running: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        failures: list[str] = []
        try:
            for job in wave_jobs:
                output_dir = Path(str(job["output_dir"]))
                require(not output_dir.exists() and not output_dir.is_symlink(), f"{job['mechanism']}: output directory already exists")
                status_path = Path(str(job["status_path"]))
                write_json_atomic(status_path, _status_payload(job, "running", started_at_utc=utc_now()))
                log_path = Path(str(job["log_path"]))
                log_handle = log_path.open("xb")
                env = os.environ.copy()
                for key in job["unset_environment"]:
                    env.pop(str(key), None)
                env.update({str(key): str(value) for key, value in job["environment"].items()})
                try:
                    process = popen_factory(
                        job["command"],
                        cwd=str(plan["project_root"]),
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                except BaseException as exc:
                    log_handle.close()
                    message = f"process launch failed: {type(exc).__name__}: {exc}"
                    write_json_atomic(status_path, _status_payload(job, "failed", error=message))
                    failures.append(f"{job['mechanism']}: {message}")
                    continue
                running[str(job["job_id"])] = (job, process, log_handle)

            while running:
                progressed = False
                for job_id, (job, process, log_handle) in list(running.items()):
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    progressed = True
                    log_handle.close()
                    running.pop(job_id)
                    status_path = Path(str(job["status_path"]))
                    if return_code != 0:
                        message = f"generate_wan_clean exited with code {return_code}"
                        write_json_atomic(
                            status_path,
                            _status_payload(job, "failed", return_code=return_code, error=message, finished_at_utc=utc_now()),
                        )
                        failures.append(f"{job['mechanism']}: {message}")
                        continue
                    try:
                        validation = validate_job_outputs(
                            job,
                            plan,
                            media_probe=media_probe,
                        )
                    except BaseException as exc:
                        message = f"post-generation validation failed: {type(exc).__name__}: {exc}"
                        write_json_atomic(
                            status_path,
                            _status_payload(job, "failed", return_code=return_code, error=message, finished_at_utc=utc_now()),
                        )
                        failures.append(f"{job['mechanism']}: {message}")
                        continue
                    write_json_atomic(
                        status_path,
                        _status_payload(job, "completed", return_code=return_code, finished_at_utc=utc_now(), **validation),
                    )
                write_aggregate(output_root, plan, overall_status="running")
                if running and not progressed:
                    sleep_fn(poll_interval)
        except BaseException:
            terminate_processes(running)
            for job, _, log_handle in running.values():
                try:
                    log_handle.close()
                except OSError:
                    pass
                write_json_atomic(
                    Path(str(job["status_path"])),
                    _status_payload(job, "failed", error="launcher interrupted while process was running"),
                )
            write_aggregate(output_root, plan, overall_status="failed", error="launcher interrupted")
            raise

        try:
            validate_bound_files(plan, require_runtime=True)
        except BaseException as exc:
            failures.append(f"bound input/code changed during wave {wave_index}: {exc}")
        if failures:
            message = "; ".join(failures)
            write_aggregate(output_root, plan, overall_status="failed", error=message)
            raise RuntimeError(message)

    statuses = [read_status(job) for job in plan["jobs"]]
    require(all(status["status"] == "completed" for status in statuses), "not all mechanism jobs completed")
    require(sum(int(status["validated_video_count"]) for status in statuses) == EXPECTED_ROWS, "validated video count is not 192")
    try:
        stage_revalidator()
        validate_bound_files(plan, require_runtime=True)
        revalidate_completed_outputs(plan, media_probe=media_probe)
        write_generation_manifest(output_root, plan)
    except BaseException as exc:
        message = f"could not freeze generation manifest: {type(exc).__name__}: {exc}"
        write_aggregate(output_root, plan, overall_status="failed", error=message)
        raise
    write_aggregate(output_root, plan, overall_status="completed")


def resolve_project_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    default_project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=DEFAULT_CANONICAL_MANIFEST,
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generator", type=Path, default=Path("scripts/generate_wan_clean.py"))
    parser.add_argument("--python-executable", type=Path, default=Path("models/.wan-runtime/bin/python"))
    parser.add_argument("--model", type=Path, default=Path("models/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--model-content-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--runtime-registry", type=Path, default=DEFAULT_RUNTIME_REGISTRY)
    parser.add_argument("--stage-registry", type=Path, default=DEFAULT_STAGE_REGISTRY)
    parser.add_argument("--expected-stage-registry-sha256")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--prepare-stage", action="store_true")
    parser.add_argument("--attest-sealed-final36-unopened", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prepare_stage and args.dry_run:
        parser.error("--prepare-stage and --dry-run are mutually exclusive")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    if not args.prepare_stage and args.output_root is None:
        parser.error("--output-root is required for dry-run planning or execution")
    if args.prepare_stage and args.output_root is not None:
        parser.error("--output-root is not accepted by --prepare-stage")
    formal_action = args.prepare_stage or not args.dry_run
    if formal_action and not args.attest_sealed_final36_unopened:
        parser.error("formal stage preparation/execution requires --attest-sealed-final36-unopened")
    if not args.prepare_stage and not args.dry_run and args.expected_stage_registry_sha256 is None:
        parser.error("formal execution requires --expected-stage-registry-sha256")
    raw_paths = [
        args.project_root,
        args.canonical_manifest,
        args.prompts,
        args.generator,
        args.model_content_inventory,
        args.runtime_registry,
        args.stage_registry,
    ]
    if args.output_root is not None:
        raw_paths.append(args.output_root)
    try:
        reject_sealed_path(*raw_paths)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        parser.error(f"--project-root is not a directory: {project_root}")
    canonical_manifest = resolve_project_path(project_root, args.canonical_manifest)
    prompts = resolve_project_path(project_root, args.prompts)
    generator = resolve_project_path(project_root, args.generator)
    python_executable = resolve_project_path(project_root, args.python_executable)
    model = resolve_project_path(project_root, args.model)
    model_inventory = resolve_project_path(project_root, args.model_content_inventory)
    runtime_registry = resolve_project_path(project_root, args.runtime_registry)
    stage_registry = resolve_project_path(project_root, args.stage_registry)
    output_root = (
        None
        if args.output_root is None
        else resolve_project_path(project_root, args.output_root)
    )

    try:
        resolved_paths = [
            project_root,
            canonical_manifest,
            prompts,
            generator,
            model_inventory,
            runtime_registry,
            stage_registry,
        ]
        if output_root is not None:
            resolved_paths.append(output_root)
        reject_sealed_path(*resolved_paths)
        if formal_action:
            require(
                canonical_manifest == (project_root / DEFAULT_CANONICAL_MANIFEST).resolve(),
                "formal stage/run requires the standard canonical-manifest path",
            )
            require(
                prompts == (project_root / DEFAULT_PROMPTS).resolve(),
                "formal stage/run requires the standard prompt path",
            )
            require(
                generator == (project_root / "scripts/generate_wan_clean.py").resolve(),
                "formal stage/run requires the standard Wan generator path",
            )
            require(
                model_inventory == (project_root / DEFAULT_MODEL_INVENTORY).resolve(),
                "formal stage/run requires the standard model-inventory path",
            )
            require(
                runtime_registry == (project_root / DEFAULT_RUNTIME_REGISTRY).resolve(),
                "formal stage/run requires the standard runtime-registry path",
            )
        rows, prompt_items = load_frozen_inputs(canonical_manifest, prompts)
        regular_file(generator, "Wan generator")
        require(sha256_file(generator) == EXPECTED_GENERATOR_SHA256, "Wan generator SHA-256 mismatch; refusing to plan or run")
        if args.prepare_stage:
            require(
                stage_registry == (project_root / DEFAULT_STAGE_REGISTRY).resolve(),
                f"--prepare-stage must write the standard registry path: {DEFAULT_STAGE_REGISTRY}",
            )
            payload = prepare_stage_registry(
                project_root=project_root,
                stage_registry=stage_registry,
                model_inventory=model_inventory,
                runtime_registry=runtime_registry,
                rows=rows,
            )
            digest = sha256_file(stage_registry)
            print(json.dumps({"stage_registry": str(stage_registry), "sha256": digest, "status": payload["status"]}, indent=2))
            return 0

        require(output_root is not None, "internal error: output root was not resolved")
        stage_payload: dict[str, Any] | None = None
        if not args.dry_run:
            require(
                stage_registry == (project_root / DEFAULT_STAGE_REGISTRY).resolve(),
                f"formal run requires the standard registry path: {DEFAULT_STAGE_REGISTRY}",
            )
            stage_payload = reopen_and_validate_stage_registry(
                project_root=project_root,
                stage_registry=stage_registry,
                expected_sha256=str(args.expected_stage_registry_sha256),
                rows=rows,
            )
            python_executable = project_root / stage_payload["runtime_registry"]["python_executable"]
            model = project_root / stage_payload["model_content_inventory"]["model_root"]
            require_git_ignored_output(project_root, output_root)
        plan = build_plan(
            rows=rows,
            prompt_items=prompt_items,
            project_root=project_root,
            canonical_manifest=canonical_manifest,
            prompts=prompts,
            output_root=output_root,
            generator=generator,
            python_executable=python_executable,
            model=model,
            gpus=args.gpus,
            dry_run=args.dry_run,
            stage_registry=None if args.dry_run else stage_registry,
            stage_registry_sha256=(
                None if args.dry_run else str(args.expected_stage_registry_sha256)
            ),
        )
        if not args.dry_run:
            validate_runtime_dependencies(plan)
        prepare_output_root(output_root, plan, prompt_items)
        validate_bound_files(plan, require_runtime=not args.dry_run)
        if args.dry_run:
            print(f"Dry-run capability plan written without starting GPU work: {output_root / 'capability_run_manifest.json'}")
            return 0
        def revalidate_stage_live() -> None:
            quick_revalidate_stage_registry(
                project_root=project_root,
                stage_registry=stage_registry,
                expected_sha256=str(args.expected_stage_registry_sha256),
            )

        execute_plan(
            plan,
            output_root,
            poll_interval=args.poll_interval,
            stage_revalidator=revalidate_stage_live,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Validated all {EXPECTED_ROWS} capability videos: {output_root / 'capability_run_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
