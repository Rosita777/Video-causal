#!/usr/bin/env python3
"""Plan or run the hash-bound 192-video Original capability-v2 batch.

The launcher adapts the audited v1 execution engine while replacing every
semantic input, ID, seed, stage registry, and protocol binding with v2 values.
It never reads a sealed/final36 path, never resumes an existing output, and
never starts GPU work in dry-run mode.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import run_causal_role_erasure_8mechanism_capability_v1 as _v1


PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v2"
RUNNER_VERSION = "causal_role_erasure_8mechanism_capability_runner_v2"

EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "8e8cc14a6d327193cb7f270d2386490dc00ee783d9e7d0f76a697837f218dbee"
)
EXPECTED_CSV_SHA256 = (
    "e92d68e23687ec72642e3620b4254e0e1e22e0d1483f84718e6dd4b1dd4ea5c3"
)
EXPECTED_SUMMARY_SHA256 = (
    "f52dcc9dbef02e62fd4e3c82d0e9726c1d49fd19e47bec61a5a55a7676a0894c"
)
EXPECTED_PROMPTS_SHA256 = (
    "472eb889d200d9d26c5863f545281468632c0185b8f148f830dfb1589467e1d4"
)
EXPECTED_GENERATOR_SHA256 = (
    "04bc6b8a8f93d885137c6157509b00f46824231560989f7b7f78192b26469b5e"
)
EXPECTED_V1_STAGE_REGISTRY_SHA256 = (
    "f74034ce3cd08db3aa19bd516170c3d0af53b9202e840385e798c13e0e1b29c5"
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
BASE_SEED = 940000
SEED_FORMULA = (
    "940000 + 1000*mechanism_index + 10*combination_index + repetition_index"
)

STEPS = 25
GUIDANCE_SCALE = 5.0
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
DTYPE = "bf16"
PYAV_PROBE_CODE = _v1.PYAV_PROBE_CODE

DEFAULT_CANONICAL_MANIFEST = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_manifest.canonical.json"
)
DEFAULT_CSV_MANIFEST = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_manifest.csv"
)
DEFAULT_SUMMARY = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_summary.json"
)
DEFAULT_PROMPTS = Path(
    "prompts/causal_role_erasure_8mechanism_capability_v2.prompts"
)
DEFAULT_STAGE_REGISTRY = Path(
    "data/causal_role_erasure_8mechanism_capability_v2_stage_registry.json"
)
DEFAULT_V1_STAGE_REGISTRY = Path(
    "data/causal_role_erasure_8mechanism_capability_v1_stage_registry.json"
)
DEFAULT_MODEL_INVENTORY = _v1.DEFAULT_MODEL_INVENTORY
DEFAULT_RUNTIME_REGISTRY = _v1.DEFAULT_RUNTIME_REGISTRY

PUBLIC_EVIDENCE_PATHS = tuple(_v1.ALLOWED_UNTRACKED_EVIDENCE_PATHS)
ALLOWED_PREEXISTING_UNTRACKED_PATHS = (
    *PUBLIC_EVIDENCE_PATHS,
    DEFAULT_V1_STAGE_REGISTRY.as_posix(),
)

FROZEN_CODE_AND_DATA_PATHS = (
    "docs/causal_role_erasure_8mechanism_capability_v2.md",
    "docs/causal_role_erasure_8mechanism_capability_v2_amendment.md",
    "data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json",
    "prompts/capability_v2_smoke16/00_water_impact.prompts",
    "prompts/capability_v2_smoke16/01_rigid_collision.prompts",
    "prompts/capability_v2_smoke16/02_brittle_fracture.prompts",
    "prompts/capability_v2_smoke16/03_powder_impact.prompts",
    "prompts/capability_v2_smoke16/04_elastic_deformation.prompts",
    "prompts/capability_v2_smoke16/05_field_mediated_response.prompts",
    "prompts/capability_v2_smoke16/06_material_release.prompts",
    "prompts/capability_v2_smoke16/07_surface_trace.prompts",
    "prompts/capability_v2_smoke_patch4/04_elastic_deformation.prompts",
    "prompts/capability_v2_smoke_patch4/05_field_mediated_response.prompts",
    "prompts/capability_v2_smoke_airflow2/05_field_mediated_response.prompts",
    "scripts/build_causal_role_erasure_8mechanism_capability_v2.py",
    "scripts/causal_role_erasure_8mechanism_capability_review_v2.py",
    "scripts/run_causal_role_erasure_8mechanism_capability_v2.py",
    "scripts/run_causal_role_erasure_8mechanism_capability_v1.py",
    "tests/test_build_causal_role_erasure_8mechanism_capability_v2.py",
    "tests/test_causal_role_erasure_8mechanism_capability_review_v2.py",
    "tests/test_run_causal_role_erasure_8mechanism_capability_v2.py",
    DEFAULT_CANONICAL_MANIFEST.as_posix(),
    DEFAULT_CSV_MANIFEST.as_posix(),
    DEFAULT_SUMMARY.as_posix(),
    DEFAULT_PROMPTS.as_posix(),
    "data/causal_role_erasure_8mechanism_capability_v2_review_template.csv",
    "data/causal_role_erasure_8mechanism_capability_v2_review_rubric.json",
    "data/causal_role_erasure_8mechanism_capability_v2_review_freeze.json",
    "scripts/generate_wan_clean.py",
)

# Self-referential launcher/test bytes are bound by the newly written v2 stage
# registry. All other newly frozen bytes are checked against these constants
# before that registry can be created.
EXPECTED_FROZEN_ARTIFACT_SHA256 = {
    "docs/causal_role_erasure_8mechanism_capability_v2.md": (
        "1afea1079ef56da4480edc75b0791420b429edfc470d594f9d0dc13f620787ab"
    ),
    "docs/causal_role_erasure_8mechanism_capability_v2_amendment.md": (
        "9bc99bbf22988edc4e156195c266807b91fea2305ffee2dc639441686e9a4b7c"
    ),
    "data/causal_role_erasure_8mechanism_capability_v2_development_smoke_decision.json": (
        "60a3377c74ea01c777b865efd3f8e5d4c3d34a33089e4825770fe5a7e5dd8010"
    ),
    "prompts/capability_v2_smoke16/00_water_impact.prompts": (
        "27822ee594d6cc69e78a64d8ed24dfd4735866431843f0cd26b6b8efcb099692"
    ),
    "prompts/capability_v2_smoke16/01_rigid_collision.prompts": (
        "83edafe831339ef2175f04b70d82b3f7d69534381081af19029d0c296de07439"
    ),
    "prompts/capability_v2_smoke16/02_brittle_fracture.prompts": (
        "8b268bcf85fd61bd078cefd095fa50b2c795f3a3cc4202edeaf741351c49ff6c"
    ),
    "prompts/capability_v2_smoke16/03_powder_impact.prompts": (
        "098073f7954e42c14a626c395cefcc16b78360e31a0fb82b86fe006c75c68a00"
    ),
    "prompts/capability_v2_smoke16/04_elastic_deformation.prompts": (
        "0c8b1614d91dc99984abf0f20749a7381bf8b12a427c4b37bd90a1656a51cd59"
    ),
    "prompts/capability_v2_smoke16/05_field_mediated_response.prompts": (
        "de29d0f3bbf38e7ba0c309a6b138ecb06ad9e2b523770fd72c3e2fe50f55d5d6"
    ),
    "prompts/capability_v2_smoke16/06_material_release.prompts": (
        "06a75d71e9534941cfb0a20a061aa7fbbe9a05f90e48bc843ac5340603b425be"
    ),
    "prompts/capability_v2_smoke16/07_surface_trace.prompts": (
        "da1343c6f0d344a06639ee8ef3dc749a320cd5efbeb3112eec885030a935b7fa"
    ),
    "prompts/capability_v2_smoke_patch4/04_elastic_deformation.prompts": (
        "9be05ad3eabb09eb5ec63da035355bec2ccd82246168979302f153b4a1049741"
    ),
    "prompts/capability_v2_smoke_patch4/05_field_mediated_response.prompts": (
        "2bac65750f5928620c1e758873057399fd46120609bf54e09ff19ce2d9cad9ef"
    ),
    "prompts/capability_v2_smoke_airflow2/05_field_mediated_response.prompts": (
        "fce181a4c2664a2054e5ffa08f419d5e19892e70005086be014ad729c99edb24"
    ),
    "scripts/build_causal_role_erasure_8mechanism_capability_v2.py": (
        "9516b52b6c6de3a87a8ede037e0d59d3ef35bebfd1a435874acc701e02a2427b"
    ),
    "scripts/causal_role_erasure_8mechanism_capability_review_v2.py": (
        "16a4e9a6f83925bdcd326c3ada3786f362983a42435d3ae376929d2892767c40"
    ),
    "scripts/run_causal_role_erasure_8mechanism_capability_v1.py": (
        "32faf629a5797a27d14559be9fc7266387e8c45da67a81fb63fbc1e5c86a2569"
    ),
    "tests/test_build_causal_role_erasure_8mechanism_capability_v2.py": (
        "161c1b1b8b2261afb39965e348d6f48f1f84a6b26f54a620d2a7ee7a345ab9fa"
    ),
    "tests/test_causal_role_erasure_8mechanism_capability_review_v2.py": (
        "4d4289dfe2af10d2d137098f776db0049018d637ca4c9bcab01f5c453f3ea868"
    ),
    DEFAULT_CANONICAL_MANIFEST.as_posix(): EXPECTED_CANONICAL_MANIFEST_SHA256,
    DEFAULT_CSV_MANIFEST.as_posix(): EXPECTED_CSV_SHA256,
    DEFAULT_SUMMARY.as_posix(): EXPECTED_SUMMARY_SHA256,
    DEFAULT_PROMPTS.as_posix(): EXPECTED_PROMPTS_SHA256,
    "data/causal_role_erasure_8mechanism_capability_v2_review_template.csv": (
        "4f8f3853a25931145d86e663d70ded54da0d162ce2d17174e5c9df7957b2820b"
    ),
    "data/causal_role_erasure_8mechanism_capability_v2_review_rubric.json": (
        "052232152cbf97a070f2479cbea2289955923d144fc48cc1420238080b2547ef"
    ),
    "data/causal_role_erasure_8mechanism_capability_v2_review_freeze.json": (
        "3c735ac020c63c62b15bab324258cd3d7a6a5bc457dd2da32793ea5bb29f92f0"
    ),
    "scripts/generate_wan_clean.py": EXPECTED_GENERATOR_SHA256,
}

EXPECTED_ROW_FIELDS = set(_v1.EXPECTED_ROW_FIELDS)
PROMPT_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
FORBIDDEN_PROMPT_TERMS = re.compile(
    r"\b(?:hand|hands|person|people|human|man|woman|no|not|without|never)\b",
    re.IGNORECASE,
)

require = _v1.require
utc_now = _v1.utc_now
sha256_file = _v1.sha256_file
regular_file = _v1.regular_file
require_sha256 = _v1.require_sha256
canonical_json_bytes = _v1.canonical_json_bytes
reject_sealed_path = _v1.reject_sealed_path
file_record = _v1.file_record
load_json_object = _v1.load_json_object
require_no_symlink_components = _v1.require_no_symlink_components
tree_stat_seal = _v1.tree_stat_seal
runtime_content_inventory = _v1.runtime_content_inventory
validate_model_inventory_live = _v1.validate_model_inventory_live
validate_runtime_registry_live = _v1.validate_runtime_registry_live
write_bytes_exclusive = _v1.write_bytes_exclusive


@contextmanager
def _v2_execution_globals() -> Iterator[None]:
    """Temporarily bind the audited v1 execution engine to the v2 contract."""

    replacements = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "RUNNER_VERSION": RUNNER_VERSION,
        "EXPECTED_CANONICAL_MANIFEST_SHA256": EXPECTED_CANONICAL_MANIFEST_SHA256,
        "EXPECTED_PROMPTS_SHA256": EXPECTED_PROMPTS_SHA256,
        "EXPECTED_CSV_SHA256": EXPECTED_CSV_SHA256,
        "EXPECTED_SUMMARY_SHA256": EXPECTED_SUMMARY_SHA256,
        "EXPECTED_GENERATOR_SHA256": EXPECTED_GENERATOR_SHA256,
        "MECHANISM_ORDER": MECHANISM_ORDER,
        "ROWS_PER_MECHANISM": ROWS_PER_MECHANISM,
        "EXPECTED_ROWS": EXPECTED_ROWS,
        "GPUS_REQUIRED": GPUS_REQUIRED,
        "WAVES": WAVES,
        "STEPS": STEPS,
        "GUIDANCE_SCALE": GUIDANCE_SCALE,
        "NUM_FRAMES": NUM_FRAMES,
        "FPS": FPS,
        "HEIGHT": HEIGHT,
        "WIDTH": WIDTH,
        "DTYPE": DTYPE,
        "DEFAULT_CANONICAL_MANIFEST": DEFAULT_CANONICAL_MANIFEST,
        "DEFAULT_CSV_MANIFEST": DEFAULT_CSV_MANIFEST,
        "DEFAULT_SUMMARY": DEFAULT_SUMMARY,
        "DEFAULT_PROMPTS": DEFAULT_PROMPTS,
        "DEFAULT_STAGE_REGISTRY": DEFAULT_STAGE_REGISTRY,
        "EXPECTED_ROW_FIELDS": EXPECTED_ROW_FIELDS,
    }
    previous = {name: getattr(_v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(_v1, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v1, name, value)


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
        "seed_source": "192 explicit v2 seeds in canonical order",
    }


def git_snapshot(
    project_root: Path,
    *,
    allowed_stage_registry: Path | None,
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    require((project_root / ".git").exists(), "project root is not a git worktree")

    def invoke(arguments: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
            },
        )
        require(result.returncode == 0, f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
        return result.stdout

    head = invoke(["rev-parse", "HEAD"]).strip()
    _v1.require_git_oid(head)
    status_lines = [
        line
        for line in invoke(["status", "--porcelain=v1", "--untracked-files=all"]).splitlines()
        if line
    ]
    require(
        all(line.startswith("?? ") for line in status_lines),
        "tracked worktree changes are forbidden at capability-v2 stage freeze/run",
    )
    observed_untracked = {line[3:] for line in status_lines}
    expected_untracked = set(ALLOWED_PREEXISTING_UNTRACKED_PATHS)
    if allowed_stage_registry is not None:
        relative = allowed_stage_registry.resolve(strict=True).relative_to(
            project_root
        ).as_posix()
        expected_untracked.add(relative)
    require(
        observed_untracked == expected_untracked,
        "worktree untracked set must be exactly the eight public evidence files "
        "plus the hash-bound v1 stage registry"
        + (" plus this v2 stage registry" if allowed_stage_registry is not None else ""),
    )
    return {
        "head": head,
        "worktree_policy": (
            "no tracked changes; exact eight public evidence files and v1 stage "
            "registry, plus the v2 stage registry after freeze"
        ),
    }


def public_evidence_records(project_root: Path) -> list[dict[str, Any]]:
    records = [
        file_record(project_root, project_root / relative, f"public evidence {relative}")
        for relative in PUBLIC_EVIDENCE_PATHS
    ]
    require(
        [record["path"] for record in records] == list(PUBLIC_EVIDENCE_PATHS),
        "public evidence order changed",
    )
    return records


def predecessor_stage_record(project_root: Path) -> dict[str, Any]:
    record = file_record(
        project_root,
        project_root / DEFAULT_V1_STAGE_REGISTRY,
        "predecessor v1 capability stage registry",
    )
    require(
        record["sha256"] == EXPECTED_V1_STAGE_REGISTRY_SHA256,
        "predecessor v1 stage-registry SHA-256 mismatch",
    )
    return record


def stage_artifact_records(project_root: Path) -> list[dict[str, Any]]:
    records = [
        file_record(
            project_root,
            project_root / relative,
            f"frozen v2 stage artifact {relative}",
        )
        for relative in FROZEN_CODE_AND_DATA_PATHS
    ]
    require(
        [record["path"] for record in records] == list(FROZEN_CODE_AND_DATA_PATHS),
        "v2 stage artifact order changed",
    )
    hashes = {record["path"]: record["sha256"] for record in records}
    for relative, expected in EXPECTED_FROZEN_ARTIFACT_SHA256.items():
        require(hashes.get(relative) == expected, f"frozen artifact hash changed: {relative}")
    return records


def expected_seed(
    mechanism_index: int,
    combination_index: int,
    repetition_index: int,
) -> int:
    return BASE_SEED + 1000 * mechanism_index + 10 * combination_index + repetition_index


def seed_registry(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    seeds = [int(row["seed"]) for row in rows]
    require(
        len(seeds) == EXPECTED_ROWS and len(set(seeds)) == EXPECTED_ROWS,
        "v2 seed registry is not 192 unique seeds",
    )
    return {
        "source": DEFAULT_CANONICAL_MANIFEST.as_posix(),
        "count": EXPECTED_ROWS,
        "order": "canonical row order",
        "algorithm": SEED_FORMULA,
        "ordered_seed_list_sha256": hashlib.sha256(
            canonical_json_bytes(seeds)
        ).hexdigest(),
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
        "schema_version": 2,
        "registry_type": "causal_role_erasure_8mechanism_capability_stage_v2",
        "protocol_version": PROTOCOL_VERSION,
        "status": "authorized_for_original_capability_v2_generation",
        "prepared_at_utc": prepared_at_utc,
        "sealed_final36_status": "unopened",
        "sealed_final36_access_policy": (
            "operator-attested; sealed/final36 paths are rejected before input read"
        ),
        "git": git,
        "public_evidence": public_evidence_records(project_root),
        "predecessor_v1_stage_registry": predecessor_stage_record(project_root),
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
            "required_video_streams": 1,
            "required_frames": NUM_FRAMES,
            "required_fps": FPS,
            "required_height": HEIGHT,
            "required_width": WIDTH,
            "final_rehash_all_192": True,
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
    require(
        not stage_registry.exists() and not stage_registry.is_symlink(),
        f"v2 stage registry already exists: {stage_registry}",
    )
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
    require_sha256(expected_sha256, "expected v2 stage-registry SHA-256")
    regular_file(stage_registry, "capability-v2 stage registry")
    require(
        sha256_file(stage_registry) == expected_sha256,
        "capability-v2 stage-registry SHA-256 mismatch",
    )
    payload = load_json_object(stage_registry, "capability-v2 stage registry")
    require(payload.get("schema_version") == 2, "v2 stage-registry schema mismatch")
    require(payload.get("protocol_version") == PROTOCOL_VERSION, "v2 stage-registry protocol mismatch")
    require(
        payload.get("status") == "authorized_for_original_capability_v2_generation",
        "v2 stage registry is not authorized",
    )
    require(
        payload.get("sealed_final36_status") == "unopened",
        "v2 stage registry does not keep sealed-final36 unopened",
    )
    model_inventory = project_root / payload["model_content_inventory"]["registry"]["path"]
    runtime_registry = project_root / payload["runtime_registry"]["registry"]["path"]
    require(
        model_inventory.resolve(strict=True)
        == (project_root / DEFAULT_MODEL_INVENTORY).resolve(strict=True),
        "v2 stage model-inventory path is not standard",
    )
    require(
        runtime_registry.resolve(strict=True)
        == (project_root / DEFAULT_RUNTIME_REGISTRY).resolve(strict=True),
        "v2 stage runtime-registry path is not standard",
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
    require(
        payload == expected,
        "v2 stage registry no longer matches live code/data/model/runtime/git state",
    )
    return payload


def quick_revalidate_stage_registry(
    *,
    project_root: Path,
    stage_registry: Path,
    expected_sha256: str,
) -> None:
    require_sha256(expected_sha256, "expected v2 stage-registry SHA-256")
    regular_file(stage_registry, "capability-v2 stage registry")
    require(
        sha256_file(stage_registry) == expected_sha256,
        "capability-v2 stage-registry SHA-256 mismatch",
    )
    payload = load_json_object(stage_registry, "capability-v2 stage registry")
    require(
        payload.get("status") == "authorized_for_original_capability_v2_generation"
        and payload.get("sealed_final36_status") == "unopened",
        "capability-v2 stage registry is not authorized/unopened",
    )
    require(
        git_snapshot(project_root, allowed_stage_registry=stage_registry)
        == payload.get("git"),
        "git HEAD/worktree changed after v2 stage freeze",
    )
    require(
        public_evidence_records(project_root) == payload.get("public_evidence"),
        "public evidence changed after v2 stage freeze",
    )
    require(
        predecessor_stage_record(project_root)
        == payload.get("predecessor_v1_stage_registry"),
        "predecessor v1 stage registry changed after v2 stage freeze",
    )
    require(
        stage_artifact_records(project_root) == payload.get("frozen_artifacts"),
        "frozen v2 artifact changed after stage freeze",
    )
    model_binding = payload["model_content_inventory"]
    runtime_binding = payload["runtime_registry"]
    model_root = project_root / model_binding["model_root"]
    runtime_root = project_root / runtime_binding["runtime_root"]
    require_no_symlink_components(project_root, model_root, "live model root")
    require_no_symlink_components(project_root, runtime_root, "live runtime root")
    require(
        tree_stat_seal(model_root) == model_binding["stat_seal"],
        "live model stat seal changed after v2 stage freeze",
    )
    require(
        tree_stat_seal(runtime_root) == runtime_binding["stat_seal"],
        "live runtime stat seal changed after v2 stage freeze",
    )


def parse_prompt_file_strict(path: Path) -> list[dict[str, str]]:
    return _v1.parse_prompt_file_strict(path)


def load_frozen_inputs(
    canonical_manifest: Path,
    prompts: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    regular_file(canonical_manifest, "v2 canonical manifest")
    regular_file(prompts, "v2 prompt file")
    require(
        sha256_file(canonical_manifest) == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "v2 canonical manifest SHA-256 mismatch; refusing to plan or run",
    )
    require(
        sha256_file(prompts) == EXPECTED_PROMPTS_SHA256,
        "v2 prompt file SHA-256 mismatch; refusing to plan or run",
    )
    try:
        raw_rows = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse v2 canonical manifest: {canonical_manifest}") from exc
    require(isinstance(raw_rows, list), "v2 canonical manifest must be a JSON array")
    require(len(raw_rows) == EXPECTED_ROWS, "v2 canonical manifest must contain exactly 192 rows")
    require(all(isinstance(row, dict) for row in raw_rows), "v2 canonical rows must be objects")
    rows: list[dict[str, str]] = [dict(row) for row in raw_rows]
    prompt_items = parse_prompt_file_strict(prompts)
    require(len(prompt_items) == EXPECTED_ROWS, "v2 prompt file must contain exactly 192 items")

    seen_generation_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for global_index, (row, prompt_item) in enumerate(zip(rows, prompt_items)):
        require(set(row) == EXPECTED_ROW_FIELDS, f"row {global_index}: v2 field schema mismatch")
        require(all(isinstance(value, str) for value in row.values()), f"row {global_index}: every field must be a string")
        mechanism_index = global_index // ROWS_PER_MECHANISM
        local_index = global_index % ROWS_PER_MECHANISM
        combination_index = local_index // 3
        repetition_index = local_index % 3
        mechanism = MECHANISM_ORDER[mechanism_index]
        case_id = f"cap8v2m{mechanism_index:02d}c{combination_index:02d}"
        generation_id = f"{case_id}r{repetition_index:02d}"
        seed = expected_seed(mechanism_index, combination_index, repetition_index)

        require(row["protocol_version"] == PROTOCOL_VERSION, f"row {global_index}: v2 protocol mismatch")
        require(row["mechanism_index"] == str(mechanism_index), f"row {global_index}: mechanism index/order mismatch")
        require(row["mechanism"] == mechanism, f"row {global_index}: mechanism order mismatch")
        require(row["combination_index"] == str(combination_index), f"row {global_index}: combination order mismatch")
        require(row["repetition_index"] == str(repetition_index), f"row {global_index}: repetition order mismatch")
        require(row["case_id"] == case_id, f"row {global_index}: fresh v2 case ID mismatch")
        require(row["generation_id"] == generation_id, f"row {global_index}: fresh v2 generation ID mismatch")
        require(int(row["seed"]) == seed, f"row {global_index}: explicit v2 seed mismatch")
        require(row["seed_formula"] == SEED_FORMULA, f"row {global_index}: v2 seed formula mismatch")
        require(row["method_arm"] == "original", f"row {global_index}: non-Original arm is forbidden")
        require(row["treatment_status"] == "pre_method_original_only", f"row {global_index}: treatment row is forbidden")
        require(row["intended_use"] == "original_capability_screening_only", f"row {global_index}: use mismatch")
        require(row["prompt_style"] == ("direct" if combination_index < 4 else "natural"), f"row {global_index}: prompt-style order mismatch")
        require(row["num_frames"] == "49" and row["fps"] == "8", f"row {global_index}: video contract mismatch")
        require(row["reference_start_inclusive"] == "0" and row["reference_end_exclusive"] == "16", f"row {global_index}: diagnostic reference metadata mismatch")
        sentences = [part.strip() for part in row["prompt"].split(".") if part.strip()]
        require(len(sentences) == 2, f"row {global_index}: prompt must have exactly two sentences")
        require(len(PROMPT_WORD.findall(row["prompt"])) <= 55, f"row {global_index}: prompt exceeds 55 English words")
        require(FORBIDDEN_PROMPT_TERMS.search(row["prompt"]) is None, f"row {global_index}: prompt contains a forbidden human/negative term")
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
        == Counter({mechanism: 24 for mechanism in MECHANISM_ORDER}),
        "v2 manifest is not balanced at 24 contiguous rows per mechanism",
    )
    for mechanism in MECHANISM_ORDER:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        require(Counter(row["prompt_style"] for row in mechanism_rows) == {"direct": 12, "natural": 12}, f"{mechanism}: v2 style balance mismatch")
        require(len({row["source_id"] for row in mechanism_rows}) == 2, f"{mechanism}: v2 source count mismatch")
        require(len({row["receiver_id"] for row in mechanism_rows}) == 2, f"{mechanism}: v2 receiver count mismatch")
        require(len({(row["source_id"], row["receiver_id"]) for row in mechanism_rows}) == 4, f"{mechanism}: v2 physical crossing mismatch")
    return rows, prompt_items


def parse_gpus(value: str) -> list[int]:
    return _v1.parse_gpus(value)


def _call_execution_engine(name: str, *args: Any, **kwargs: Any) -> Any:
    with _v2_execution_globals():
        return getattr(_v1, name)(*args, **kwargs)


def build_plan(**kwargs: Any) -> dict[str, Any]:
    plan = _call_execution_engine("build_plan", **kwargs)
    plan["inputs"].update(
        {
            "csv_manifest": str(Path(plan["project_root"]) / DEFAULT_CSV_MANIFEST),
            "csv_manifest_sha256": EXPECTED_CSV_SHA256,
            "summary": str(Path(plan["project_root"]) / DEFAULT_SUMMARY),
            "summary_sha256": EXPECTED_SUMMARY_SHA256,
            "seed_formula": SEED_FORMULA,
        }
    )
    plan["scheduler"]["layout"] = "4_gpus_x_2_waves"
    return plan


def prepare_output_root(*args: Any, **kwargs: Any) -> None:
    _call_execution_engine("prepare_output_root", *args, **kwargs)


def read_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call_execution_engine("read_status", *args, **kwargs)


def validate_bound_files(*args: Any, **kwargs: Any) -> None:
    _call_execution_engine("validate_bound_files", *args, **kwargs)


def validate_runtime_dependencies(*args: Any, **kwargs: Any) -> None:
    _call_execution_engine("validate_runtime_dependencies", *args, **kwargs)


def _expected_generation_fields() -> dict[str, Any]:
    return _call_execution_engine("_expected_generation_fields")


def probe_video_media(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call_execution_engine("probe_video_media", *args, **kwargs)


def validate_job_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _call_execution_engine("validate_job_outputs", *args, **kwargs)


def write_generation_manifest(*args: Any, **kwargs: Any) -> Path:
    return _call_execution_engine("write_generation_manifest", *args, **kwargs)


def revalidate_completed_outputs(*args: Any, **kwargs: Any) -> None:
    _call_execution_engine("revalidate_completed_outputs", *args, **kwargs)


def execute_plan(*args: Any, **kwargs: Any) -> None:
    _call_execution_engine("execute_plan", *args, **kwargs)


def resolve_project_path(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    default_project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generator", type=Path, default=Path("scripts/generate_wan_clean.py"))
    parser.add_argument("--python-executable", type=Path, default=Path("models/.wan-runtime/bin/python"))
    parser.add_argument("--model", type=Path, default=Path("models/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--model-content-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--runtime-registry", type=Path, default=DEFAULT_RUNTIME_REGISTRY)
    parser.add_argument("--stage-registry", type=Path, default=DEFAULT_STAGE_REGISTRY)
    parser.add_argument("--v1-stage-registry", type=Path, default=DEFAULT_V1_STAGE_REGISTRY)
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
        parser.error("formal v2 stage preparation/execution requires --attest-sealed-final36-unopened")
    if not args.prepare_stage and not args.dry_run and args.expected_stage_registry_sha256 is None:
        parser.error("formal v2 execution requires --expected-stage-registry-sha256")

    raw_paths = [
        args.project_root,
        args.canonical_manifest,
        args.prompts,
        args.generator,
        args.model_content_inventory,
        args.runtime_registry,
        args.stage_registry,
        args.v1_stage_registry,
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
    v1_stage_registry = resolve_project_path(project_root, args.v1_stage_registry)
    output_root = None if args.output_root is None else resolve_project_path(project_root, args.output_root)

    try:
        resolved_paths = [
            project_root,
            canonical_manifest,
            prompts,
            generator,
            model_inventory,
            runtime_registry,
            stage_registry,
            v1_stage_registry,
        ]
        if output_root is not None:
            resolved_paths.append(output_root)
        reject_sealed_path(*resolved_paths)
        if formal_action:
            require(canonical_manifest == (project_root / DEFAULT_CANONICAL_MANIFEST).resolve(), "formal v2 stage/run requires the standard canonical-manifest path")
            require(prompts == (project_root / DEFAULT_PROMPTS).resolve(), "formal v2 stage/run requires the standard prompt path")
            require(generator == (project_root / "scripts/generate_wan_clean.py").resolve(), "formal v2 stage/run requires the standard Wan generator path")
            require(model_inventory == (project_root / DEFAULT_MODEL_INVENTORY).resolve(), "formal v2 stage/run requires the standard model-inventory path")
            require(runtime_registry == (project_root / DEFAULT_RUNTIME_REGISTRY).resolve(), "formal v2 stage/run requires the standard runtime-registry path")
            require(v1_stage_registry == (project_root / DEFAULT_V1_STAGE_REGISTRY).resolve(), "formal v2 stage/run requires the standard v1 stage-registry path")

        rows, prompt_items = load_frozen_inputs(canonical_manifest, prompts)
        regular_file(generator, "Wan generator")
        require(sha256_file(generator) == EXPECTED_GENERATOR_SHA256, "Wan generator SHA-256 mismatch; refusing to plan or run")

        if args.prepare_stage:
            require(stage_registry == (project_root / DEFAULT_STAGE_REGISTRY).resolve(), f"--prepare-stage must write the standard v2 registry path: {DEFAULT_STAGE_REGISTRY}")
            regular_file(v1_stage_registry, "predecessor v1 stage registry")
            require(sha256_file(v1_stage_registry) == EXPECTED_V1_STAGE_REGISTRY_SHA256, "predecessor v1 stage-registry SHA-256 mismatch")
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
            require(stage_registry == (project_root / DEFAULT_STAGE_REGISTRY).resolve(), f"formal v2 run requires the standard registry path: {DEFAULT_STAGE_REGISTRY}")
            stage_payload = reopen_and_validate_stage_registry(
                project_root=project_root,
                stage_registry=stage_registry,
                expected_sha256=str(args.expected_stage_registry_sha256),
                rows=rows,
            )
            python_executable = project_root / stage_payload["runtime_registry"]["python_executable"]
            model = project_root / stage_payload["model_content_inventory"]["model_root"]
            _v1.require_git_ignored_output(project_root, output_root)

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
            stage_registry_sha256=None if args.dry_run else str(args.expected_stage_registry_sha256),
        )
        if not args.dry_run:
            validate_runtime_dependencies(plan)
        prepare_output_root(output_root, plan, prompt_items)
        validate_bound_files(plan, require_runtime=not args.dry_run)
        if args.dry_run:
            print(f"Dry-run capability-v2 plan written without starting GPU work: {output_root / 'capability_run_manifest.json'}")
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
    print(f"Validated all {EXPECTED_ROWS} capability-v2 videos: {output_root / 'capability_run_aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
