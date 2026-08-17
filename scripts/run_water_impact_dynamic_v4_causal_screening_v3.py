#!/usr/bin/env python3
"""Run the one-shot v4_dev72_v3 Original screen and freeze its review package.

This entry point is deliberately independent from every v1/v2 evaluation
implementation.  It accepts only the standard v3 Stage-0 paths, invokes the
code-registry-bound generic Wan generator without a shell, and never performs
review, adjudication, selection, treatment generation, or scoring.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import build_water_impact_dynamic_v4_causal_candidates_v3 as builder
    import select_water_impact_dynamic_v4_causal_v3 as selector
    import authorize_water_impact_dynamic_v4_causal_stage0_v3 as authorizer
except ModuleNotFoundError:  # imported as scripts.run_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import build_water_impact_dynamic_v4_causal_candidates_v3 as builder
    from scripts import select_water_impact_dynamic_v4_causal_v3 as selector
    from scripts import authorize_water_impact_dynamic_v4_causal_stage0_v3 as authorizer


SCREENING_RUN_PROTOCOL = "water_impact_dynamic_v4_causal_screening_run_v3"
RAW_INVENTORY_PROTOCOL = "water_impact_dynamic_v4_causal_raw_video_inventory_v3"
GENERATION_MANIFEST_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_generation_manifest_v3"
)
PUBLIC_PACKAGE_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_public_package_v3"
)
PRIVATE_PACKAGE_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_private_package_v3"
)
CANDIDATE_BINDING_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_candidate_binding_v3"
)
ANONYMOUS_INVENTORY_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_anonymous_video_inventory_v3"
)
COMPOSITE_INVENTORY_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_composite_inventory_v3"
)
PACKAGE_COMMITMENT_PROTOCOL = (
    "water_impact_dynamic_v4_causal_screening_package_commitment_v3"
)
STATUS_PROTOCOL = "water_impact_dynamic_v4_causal_screening_status_v3"
LOCK_PROTOCOL = "water_impact_dynamic_v4_causal_screening_cuda_lock_v3"

GENERATION_DIRNAME = "causal_original_screening_generation_v3"
PUBLIC_PACKAGE_DIRNAME = "causal_original_screening_review_public_v3"
PRIVATE_PACKAGE_DIRNAME = "causal_original_screening_review_private_v3"
CUDA_LOCK_BASENAME = "causal_original_screening_cuda_lock_v3.json"
INVALID_REASON_GENERATION = "screening_generation_incomplete"
INVALID_REASON_PACKAGE = "screening_package_integrity_failure"

GENERIC_MANIFEST_BASENAME = "generation_manifest.json"
RAW_INVENTORY_BASENAME = "screening_raw_video_inventory_576_v3.json"
GENERATION_MANIFEST_BASENAME = "screening_generation_manifest_576_v3.json"
REVIEW_TEMPLATE_BASENAME = "screening_review_template_576_v3.csv"
ANSWER_KEY_BASENAME = "screening_answer_key_576_v3.csv"
CANDIDATE_BINDING_BASENAME = "screening_candidate_binding_576_v3.json"
ANONYMOUS_INVENTORY_BASENAME = (
    "screening_anonymous_video_inventory_576_v3.json"
)
COMPOSITE_INVENTORY_BASENAME = "screening_composite_inventory_576_v3.json"
PUBLIC_MANIFEST_BASENAME = "screening_public_package_manifest_576_v3.json"
PRIVATE_MANIFEST_BASENAME = "screening_private_package_manifest_576_v3.json"
PACKAGE_COMMITMENT_BASENAME = "screening_package_commitment_v3.json"

FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)
REVIEW_FIELDS = (
    "source_visibility",
    "footprint_visibility",
    "receiver",
    "quality",
    "causal_link",
)
REVIEW_HEADER = (
    "review_id",
    "candidate_video_path",
    "candidate_video_sha256",
    "composite_path",
    "composite_sha256",
    *REVIEW_FIELDS,
    "notes",
)
EXPECTED_GENERATION = {
    "steps": 25,
    "cfg": 5,
    "frames": 49,
    "width": 832,
    "height": 480,
    "fps": 8,
    "dtype": "bf16",
    "adapter": None,
    "skip_existing": False,
    "resume": False,
    "worker_count": 1,
}
MAX_SCREENING_GENERATION_SECONDS = 576 * 600
GENERATOR_BOOTSTRAP = """
import os, runpy, sys, types
import torch, diffusers, diffusers.utils
scripts = os.path.realpath("scripts")
for name in (
    "run_pilot",
    "generate_cogvideox_clean",
    "causal_lora_activation_gate",
    "target_token_attention_suppression",
):
    path = os.path.join(scripts, name + ".py")
    module = types.ModuleType(name)
    module.__file__ = path
    module.__package__ = ""
    sys.modules[name] = module
    with open(path, "rb") as handle:
        source = handle.read()
    exec(compile(source, path, "exec"), module.__dict__)
generator = os.path.join(scripts, "generate_wan_clean.py")
sys.argv = [generator, *sys.argv[1:]]
runpy.run_path(generator, run_name="__main__")
""".strip()


class TerminalScreeningFailure(RuntimeError):
    """A terminal one-shot failure safe to report without private detail."""

    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


class ConsumedReservationFailure(RuntimeError):
    """Reservation failed only after this invocation acquired the one-shot lock."""


@dataclass(frozen=True)
class Stage0Context:
    project_root: Path
    private_root: Path
    stage0_path: Path
    stage0_sha256: str
    stage0: Mapping[str, Any]
    pending_sha256: str
    binding_sha256: str
    opening_paths: Mapping[str, Path]
    opening_records: Mapping[str, Mapping[str, Any]]
    candidate_payload: Mapping[str, Any]
    graph_payload: Mapping[str, Any]
    generation_spec: Mapping[str, Any]
    screening_seed: int
    model_inventory_sha256: str
    runtime_registry_sha256: str
    code_registry_sha256: str
    generator_path: Path
    generator_sha256: str
    generator_dependency_closure_sha256: str
    media_runtime_packages: Mapping[str, str]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return protocol.canonical_json_bytes(dict(payload))


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    return protocol.write_json_exclusive_atomic(path, payload, mode=mode)


def _write_bytes_exclusive(path: Path, raw: bytes, mode: int) -> str:
    protocol.reject_forbidden_path(path)
    protocol._require_no_symlink_components(path.parent)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite frozen output: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return protocol.sha256_bytes(raw)


def _write_csv_exclusive(
    path: Path, header: Sequence[str], rows: Sequence[Mapping[str, Any]], mode: int
) -> str:
    protocol.reject_forbidden_path(path)
    protocol._require_no_symlink_components(path.parent)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite frozen CSV: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(header),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                if set(row) != set(header):
                    raise ValueError("review CSV row fields are not exact")
                writer.writerow({name: row[name] for name in header})
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return protocol.sha256_file(path)


def _record(path: Path, row_count: int | None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink artifact: {path}")
    info = path.stat()
    protocol.require(info.st_nlink == 1 and info.st_size > 0, "artifact link/size invalid")
    return {
        "sha256": protocol.sha256_file(path),
        "size_bytes": info.st_size,
        "row_count": row_count,
    }


def _load_public(project_root: Path, relative: Path) -> dict[str, Any]:
    return protocol.load_json(
        project_root / relative, project_root=project_root, allow_v2=False
    )


def _load_private(private_root: Path, basename: str) -> dict[str, Any]:
    return protocol.load_json(private_root / basename, private_root=private_root)


def _require_private_root(project_root: Path, private_root: Path) -> Path:
    private_root = protocol._canonical_lexical_absolute(private_root)
    protocol.reject_forbidden_path(private_root)
    protocol._require_no_symlink_components(private_root)
    info = private_root.stat()
    protocol.require(
        private_root.is_dir()
        and not private_root.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "PRIVATE_V3_ROOT must be a real mode-700 directory",
    )
    resolved = private_root.resolve(strict=True)
    protocol.require(
        resolved != project_root
        and resolved not in project_root.parents
        and project_root not in resolved.parents,
        "project and PRIVATE_V3_ROOT must be distinct and nonnested",
    )
    return resolved


def _require_no_public_terminal_or_stage1(project_root: Path) -> None:
    for relative in (protocol.INVALID_OUTCOME, protocol.STAGE1_REGISTRY):
        target = project_root / relative
        protocol.require(
            not os.path.lexists(target),
            f"v3 screening conflicts with terminal/Stage-1 artifact: {target}",
        )


@contextlib.contextmanager
def _screening_mutex(private_root: Path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(private_root, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(
                "another v3 screening invocation owns the private-root mutex"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _require_initial_private_inventory(private_root: Path) -> None:
    expected = set(authorizer.PRIVATE_INPUTS.values()) | {
        "causal_selection_binding_v3.json"
    }
    observed = {entry.name for entry in private_root.iterdir()}
    protocol.require(
        observed == expected,
        "pre-screening PRIVATE_V3_ROOT inventory is not the exact 20 artifacts",
    )
    for name in authorizer.PRIVATE_INPUTS.values():
        protocol.validate_private_path(private_root, private_root / name)
    protocol.validate_private_path(
        private_root, private_root / "causal_selection_binding_v3.json"
    )


def _require_runtime_private_inventory(private_root: Path) -> None:
    base = set(authorizer.PRIVATE_INPUTS.values()) | {
        "causal_selection_binding_v3.json",
        GENERATION_DIRNAME,
        CUDA_LOCK_BASENAME,
    }
    observed = {entry.name for entry in private_root.iterdir()}
    transient = {
        name
        for name in observed
        if name.startswith(".causal-screening-package-v3-")
    }
    protocol.require(
        len(transient) <= 1 and observed == base | transient,
        "runtime PRIVATE_V3_ROOT inventory contains an unexpected artifact",
    )
    generation_dir = private_root / GENERATION_DIRNAME
    protocol.require(
        generation_dir.is_dir()
        and not generation_dir.is_symlink()
        and stat.S_IMODE(generation_dir.stat().st_mode) == 0o700,
        "reserved screening generation directory is invalid",
    )
    protocol.validate_private_path(
        private_root, private_root / CUDA_LOCK_BASENAME
    )
    for name in transient:
        candidate = private_root / name
        protocol.require(
            candidate.is_dir()
            and not candidate.is_symlink()
            and stat.S_IMODE(candidate.stat().st_mode) == 0o700,
            "screening package staging directory is invalid",
        )


def _compare_wrapper_artifacts(
    *,
    project_root: Path,
    private_root: Path,
    stage0: Mapping[str, Any],
    opening_records: Mapping[str, Mapping[str, Any]],
) -> None:
    artifacts = stage0["artifacts"]
    for name, record in opening_records.items():
        if name == "selection_rules":
            for alias in (
                "ranking_formula",
                "constrained_subset_algorithm",
                "seed_derivation_formula",
            ):
                protocol.require(
                    artifacts[alias] == record,
                    f"Stage-0 selection-rules alias mismatch: {alias}",
                )
        else:
            protocol.require(
                artifacts[name] == record,
                f"Stage-0 opening record mismatch: {name}",
            )
    protocol.require(
        artifacts["upstream_source_bank_registry_64_v2"]
        == authorizer._file_record(project_root / protocol.V2_BANK, 64)
        and artifacts["upstream_source_mapping_178_v2"]
        == authorizer._file_record(project_root / protocol.V2_MAPPING, 178)
        and artifacts["preregistration"]
        == authorizer._file_record(
            project_root / authorizer.PREREG_PATH, None
        )
        and artifacts["v2_public_aggregate_design_input"]
        == authorizer._file_record(project_root / protocol.V2_TERMINATION, 6)
        and artifacts["selection_binding"]
        == authorizer._file_record(
            private_root / "causal_selection_binding_v3.json", None
        ),
        "Stage-0 derived/upstream artifact mismatch",
    )


def validate_runner_process_environment(
    project_root: Path, media_runtime_packages: Mapping[str, str]
) -> None:
    expected_executable = (
        project_root / "models/.wan-runtime/bin/python"
    ).resolve(strict=True)
    expected_prefix = (project_root / "models/.wan-runtime").resolve(strict=True)
    protocol.require(
        Path(sys.executable).resolve(strict=True) == expected_executable
        and Path(sys.prefix).resolve(strict=True) == expected_prefix,
        "screening runner is not executing in the frozen interpreter",
    )
    protocol.require(
        sys.flags.isolated == 1,
        "screening runner must itself execute in isolated Python mode",
    )
    observed = {
        name: importlib.metadata.version(name)
        for name in authorizer.MEDIA_RUNTIME_DISTRIBUTIONS
    }
    protocol.require(
        observed == dict(media_runtime_packages),
        "screening runner media package versions differ from generation spec",
    )
    scripts_root = (project_root / "scripts").resolve(strict=True)
    for module_name in ("av", "PIL"):
        spec = importlib.util.find_spec(module_name)
        protocol.require(
            spec is not None and isinstance(spec.origin, str) and spec.origin,
            f"media runtime module has no concrete origin: {module_name}",
        )
        origin = Path(spec.origin).resolve(strict=True)
        protocol.require(
            scripts_root not in origin.parents and origin != scripts_root,
            f"media runtime module is shadowed by repository code: {module_name}",
        )


def _validate_selection_binding(
    payload: Mapping[str, Any],
    *,
    private_root: Path,
    pending_sha256: str,
    opening_records: Mapping[str, Mapping[str, Any]],
    secret_commitments: Mapping[str, str],
    graph_payload: Mapping[str, Any],
    historical_inventory_sha256: str,
    historical_inventory_count: int,
    forbidden_count: int,
    v2_hashes: Mapping[str, str],
    project_root: Path,
    generator_dependency_closure_sha256: str,
    media_runtime_packages: Mapping[str, str],
) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "public_pending_sha256",
            "private_root_basename",
            "opening_artifacts",
            "secret_commitments",
            "graph_contract",
            "historical_receiver_contract",
            "seed_contract",
            "registries",
            "capacity",
            "upstream_public_sha256",
            "authorizer_sha256",
        },
        "selection binding",
    )
    protocol.require(
        payload["protocol"] == authorizer.SELECTION_BINDING_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "authorized_before_original_screening"
        and payload["public_pending_sha256"] == pending_sha256
        and payload["private_root_basename"] == private_root.name
        and payload["opening_artifacts"] == dict(opening_records)
        and payload["secret_commitments"] == dict(secret_commitments),
        "selection binding identity/openings mismatch",
    )
    protocol.require(
        payload["graph_contract"]
        == {
            "graph_sha256": graph_payload["graph_sha256"],
            "graph_file_sha256": opening_records["candidate_graph_576"][
                "sha256"
            ],
            "graph_assignment_salt_sha256": opening_records[
                "graph_assignment_salt"
            ]["sha256"],
            "r1_permutation_sha256": graph_payload["r1"]["permutation_sha256"],
            "r3_permutation_sha256": graph_payload["r3"]["permutation_sha256"],
        }
        and payload["historical_receiver_contract"]
        == {
            "inventory_sha256": historical_inventory_sha256,
            "inventory_count": historical_inventory_count,
            "selected_anchor_count": 8,
        },
        "selection binding graph/historical contract mismatch",
    )
    protocol.require(
        payload["seed_contract"]
        == {
            "preselection_seed_audit_sha256": opening_records[
                "preselection_seed_audit_1728"
            ]["sha256"],
            "seed_count": 1728,
            "unique_seed_count": 1728,
            "screening_collision_count": 0,
            "forbidden_collision_count": 0,
            "forbidden_seed_source_audit_sha256": opening_records[
                "forbidden_seed_source_audit"
            ]["sha256"],
            "forbidden_seed_count": forbidden_count,
        },
        "selection binding seed contract mismatch",
    )
    protocol.require(
        payload["registries"]
        == {
            "model_content_inventory_sha256": opening_records[
                "model_content_inventory"
            ]["sha256"],
            "runtime_registry_sha256": opening_records["runtime_registry"][
                "sha256"
            ],
            "eval_code_registry_sha256": opening_records[
                "eval_code_registry"
            ]["sha256"],
            "screening_cost_calibration_sha256": opening_records[
                "screening_cost_calibration"
            ]["sha256"],
            "generator_dependency_closure_sha256": (
                generator_dependency_closure_sha256
            ),
            "media_runtime_packages": dict(media_runtime_packages),
        }
        and payload["capacity"]
        == {
            "model_sha256": opening_records["capacity_model_spec"]["sha256"],
            "search_sha256": opening_records[
                "capacity_search_result_200000"
            ]["sha256"],
            "confirmation_sha256": opening_records[
                "capacity_confirm_result_1000000"
            ]["sha256"],
            "static_graph_sha256": opening_records[
                "static_graph_robustness_report"
            ]["sha256"],
        }
        and payload["upstream_public_sha256"] == dict(v2_hashes)
        and payload["authorizer_sha256"]
        == protocol.sha256_file(
            project_root / protocol.CODE_ARTIFACT_PATHS["stage0_authorizer"]
        ),
        "selection binding provenance mismatch",
    )


def validate_stage0_for_screening(
    *,
    project_root: Path,
    private_root: Path,
    require_initial_inventory: bool,
) -> Stage0Context:
    project_root = protocol.validate_project_root(project_root)
    private_root = _require_private_root(project_root, private_root)
    _require_no_public_terminal_or_stage1(project_root)
    if require_initial_inventory:
        _require_initial_private_inventory(private_root)
    else:
        _require_runtime_private_inventory(private_root)

    stage0_path = project_root / protocol.STAGE0_REGISTRY
    stage0 = _load_public(project_root, protocol.STAGE0_REGISTRY)
    protocol.validate_commitment_registry(stage0, stage=0)
    stage0_sha256 = protocol.sha256_file(stage0_path)
    v2_hashes = protocol.validate_v2_public_inputs(project_root)

    code_payload = _load_public(project_root, protocol.CODE_REGISTRY)
    expected_code = authorizer.validate_code_registry_full(code_payload, project_root)
    protocol.require(code_payload == expected_code, "current code registry drift")

    pending_path = project_root / protocol.STAGE0_PUBLIC
    pending = _load_public(project_root, protocol.STAGE0_PUBLIC)
    authorizer.validate_pending(
        pending, project_root=project_root, pending_path=pending_path
    )
    pending_sha256 = protocol.sha256_file(pending_path)

    opening_paths = authorizer._opening_paths(project_root, private_root)
    for name in authorizer.PRIVATE_INPUTS:
        protocol.validate_private_path(private_root, opening_paths[name])
    opening_records = authorizer._records_for_openings(opening_paths)
    protocol.require(
        opening_records == pending["component_commitments"],
        "pending/opening records differ before screening",
    )
    _compare_wrapper_artifacts(
        project_root=project_root,
        private_root=private_root,
        stage0=stage0,
        opening_records=opening_records,
    )

    source_payload = _load_private(
        private_root,
        authorizer.PRIVATE_INPUTS["eval_holdout_source_ontology_48"],
    )
    sources = builder.validate_holdout_ontology(source_payload)
    holdout_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["holdout_registry_48"]
    )
    authorizer._validate_holdout_registry(holdout_payload, sources)
    receiver_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["receiver_ontology_56"]
    )
    builder.validate_receiver_ontology(receiver_payload)
    historical_payload = _load_private(
        private_root,
        authorizer.PRIVATE_INPUTS["historical_receiver_anchors_8"],
    )
    builder.validate_historical_anchors(historical_payload)
    mapping_payload = protocol.load_json(
        project_root / protocol.V2_MAPPING,
        project_root=project_root,
        allow_v2=True,
    )
    historical_inventory, historical_inventory_sha256 = (
        authorizer._historical_receiver_inventory(
            mapping_payload, historical_payload
        )
    )

    candidate_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["candidate_manifest_576"]
    )
    graph_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["candidate_graph_576"]
    )
    builder.validate_candidate_projection(graph_payload, candidate_payload)
    builder.validate_templates_and_fields(
        opening_paths["canonical_templates"],
        opening_paths["field_normalization"],
        private_root=private_root,
    )

    graph_salt = authorizer._read_text_secret(opening_paths["graph_assignment_salt"])
    selector_salt = authorizer._read_text_secret(opening_paths["selector_salt"])
    evaluation_salt = authorizer._read_text_secret(
        opening_paths["evaluation_seed_salt"]
    )
    screening_seed = authorizer._read_text_secret(
        opening_paths["screening_seed"], integer=True
    )
    assert isinstance(graph_salt, str) and isinstance(selector_salt, str)
    assert isinstance(evaluation_salt, str) and isinstance(screening_seed, int)
    secrets_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["stage0_secrets"]
    )
    secret_commitments = authorizer._validate_secrets(
        secrets_payload,
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
    )
    protocol.require(
        pending["public_metadata"]
        == authorizer._expected_public_metadata(
            opening_records, secret_commitments
        ),
        "pending secret commitments drifted before screening",
    )
    source_bank_payload = protocol.load_json(
        project_root / protocol.V2_BANK,
        project_root=project_root,
        allow_v2=True,
    )
    builder.validate_graph_against_inputs(
        graph_payload,
        candidate_payload,
        holdout_payload=source_payload,
        receiver_payload=receiver_payload,
        historical_payload=historical_payload,
        source_bank_payload=source_bank_payload,
        graph_assignment_salt=graph_salt,
    )

    model_payload = _load_public(project_root, authorizer.MODEL_INVENTORY_PATH)
    model_inventory_sha256 = authorizer._validate_model_inventory(
        model_payload, project_root
    )
    runtime_payload = _load_public(project_root, authorizer.RUNTIME_REGISTRY_PATH)
    live_hardware = authorizer._validate_runtime_registry(
        runtime_payload, project_root
    )
    runtime_registry_sha256 = opening_records["runtime_registry"]["sha256"]
    _, generator_closure_sha256 = authorizer.generator_dependency_closure(
        project_root
    )
    media_runtime_packages = authorizer.probe_media_runtime_packages(project_root)
    validate_runner_process_environment(project_root, media_runtime_packages)

    render_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["raw_render_configuration"]
    )
    authorizer._validate_render(render_payload, model_inventory_sha256)
    rules_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["selection_rules"]
    )
    authorizer._validate_rules(rules_payload)
    generation_spec = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["screening_generation_spec"]
    )
    authorizer._validate_generation_spec(
        generation_spec,
        candidate_sha=opening_records["candidate_manifest_576"]["sha256"],
        graph_sha=opening_records["candidate_graph_576"]["sha256"],
        render_sha=opening_records["raw_render_configuration"]["sha256"],
        screening_seed_sha=opening_records["screening_seed"]["sha256"],
        graph_salt_sha=opening_records["graph_assignment_salt"]["sha256"],
        model_sha=model_inventory_sha256,
        runtime_sha=runtime_registry_sha256,
        generator_dependency_closure_sha256=generator_closure_sha256,
        media_runtime_packages=media_runtime_packages,
    )
    protocol.require(
        generation_spec["generation"] == EXPECTED_GENERATION,
        "screening generation spec is not the single-worker contract",
    )

    forbidden_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["forbidden_seed_inventory"]
    )
    forbidden = selector.validate_forbidden_seed_inventory(forbidden_payload)
    protocol.require(
        screening_seed not in forbidden,
        "screening seed collides with forbidden inventory",
    )
    forbidden_audit = _load_public(
        project_root, protocol.FORBIDDEN_SEED_SOURCE_AUDIT
    )
    authorizer._validate_forbidden_seed_source_audit(
        forbidden_audit,
        v3_inventory_sha256=opening_records["forbidden_seed_inventory"][
            "sha256"
        ],
        v3_seed_count=len(forbidden),
    )
    seed_payload = _load_private(
        private_root,
        authorizer.PRIVATE_INPUTS["preselection_seed_audit_1728"],
    )
    authorizer._validate_seed_audit(
        seed_payload,
        candidates=candidate_payload["candidates"],
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
        forbidden=forbidden,
        candidate_sha=opening_records["candidate_manifest_576"]["sha256"],
        evaluation_salt_sha=opening_records["evaluation_seed_salt"]["sha256"],
        screening_seed_sha=opening_records["screening_seed"]["sha256"],
        forbidden_sha=opening_records["forbidden_seed_inventory"]["sha256"],
    )

    identity_payload = _load_public(project_root, protocol.IDENTITY_REPORT)
    protocol.validate_identity_disjointness_report(identity_payload)
    ontology_bundle_sha256 = protocol.sha256_bytes(
        protocol.canonical_json_bytes(
            {
                authorizer.PRIVATE_INPUTS[
                    "eval_holdout_source_ontology_48"
                ]: opening_records["eval_holdout_source_ontology_48"]["sha256"],
                authorizer.PRIVATE_INPUTS[
                    "receiver_ontology_56"
                ]: opening_records["receiver_ontology_56"]["sha256"],
                authorizer.PRIVATE_INPUTS[
                    "historical_receiver_anchors_8"
                ]: opening_records["historical_receiver_anchors_8"]["sha256"],
            }
        )
    )
    protocol.require(
        identity_payload["v3_candidate_graph_sha256"]
        == opening_records["candidate_graph_576"]["sha256"]
        and identity_payload["v3_ontology_bundle_sha256"]
        == ontology_bundle_sha256,
        "identity report binding drifted before screening",
    )
    construct_payload = _load_public(project_root, protocol.CONSTRUCT_REPORT)
    protocol.validate_construct_equivalence_report(construct_payload)
    protocol.require(
        construct_payload["v3_file_sha256"]
        == {
            "templates": opening_records["canonical_templates"]["sha256"],
            "field_rules": opening_records["field_normalization"]["sha256"],
            "selection_rules": opening_records["selection_rules"]["sha256"],
        }
        and construct_payload["qualification_sha256"]["v3"]
        == protocol.sha256_bytes(
            protocol.canonical_json_bytes(rules_payload["qualification"])
        )
        and construct_payload["cell_quota_sha256"]["v3"]
        == protocol.sha256_bytes(
            protocol.canonical_json_bytes(rules_payload["cell_quota"])
        ),
        "construct report binding drifted before screening",
    )

    capacity_model = _load_public(project_root, authorizer.CAPACITY_MODEL_PATH)
    capacity_search = _load_public(project_root, authorizer.CAPACITY_SEARCH_PATH)
    capacity_confirm = _load_public(project_root, authorizer.CAPACITY_CONFIRM_PATH)
    static_graph = _load_public(project_root, authorizer.STATIC_GRAPH_PATH)
    authorizer._validate_capacity_artifacts(
        capacity_model, capacity_search, capacity_confirm, static_graph
    )
    cost_payload = _load_public(project_root, authorizer.COST_CALIBRATION_PATH)
    authorizer._validate_cost_calibration(
        cost_payload,
        model_sha=model_inventory_sha256,
        runtime_sha=runtime_registry_sha256,
        render_sha=opening_records["raw_render_configuration"]["sha256"],
        live_hardware=live_hardware,
    )
    bundle_payload = _load_private(
        private_root, authorizer.PRIVATE_INPUTS["raw_root_bundle"]
    )
    authorizer._validate_bundle(
        bundle_payload,
        private_records={
            name: opening_records[name] for name in authorizer.PRIVATE_INPUTS
        },
    )

    binding_path = private_root / "causal_selection_binding_v3.json"
    binding_payload = protocol.load_json(binding_path, private_root=private_root)
    _validate_selection_binding(
        binding_payload,
        private_root=private_root,
        pending_sha256=pending_sha256,
        opening_records=opening_records,
        secret_commitments=secret_commitments,
        graph_payload=graph_payload,
        historical_inventory_sha256=historical_inventory_sha256,
        historical_inventory_count=len(historical_inventory),
        forbidden_count=len(forbidden),
        v2_hashes=v2_hashes,
        project_root=project_root,
        generator_dependency_closure_sha256=generator_closure_sha256,
        media_runtime_packages=media_runtime_packages,
    )

    code_registry_sha256 = opening_records["eval_code_registry"]["sha256"]
    generator_record = code_payload["artifacts"]["generator"]
    generator_path = project_root / generator_record["path"]
    protocol.require(
        generator_record["path"]
        == protocol.CODE_ARTIFACT_PATHS["generator"]
        and protocol.sha256_file(generator_path) == generator_record["sha256"],
        "generic generator code differs from code registry",
    )
    return Stage0Context(
        project_root=project_root,
        private_root=private_root,
        stage0_path=stage0_path,
        stage0_sha256=stage0_sha256,
        stage0=stage0,
        pending_sha256=pending_sha256,
        binding_sha256=protocol.sha256_file(binding_path),
        opening_paths=dict(opening_paths),
        opening_records=dict(opening_records),
        candidate_payload=candidate_payload,
        graph_payload=graph_payload,
        generation_spec=generation_spec,
        screening_seed=screening_seed,
        model_inventory_sha256=model_inventory_sha256,
        runtime_registry_sha256=runtime_registry_sha256,
        code_registry_sha256=code_registry_sha256,
        generator_path=generator_path,
        generator_sha256=generator_record["sha256"],
        generator_dependency_closure_sha256=generator_closure_sha256,
        media_runtime_packages=dict(media_runtime_packages),
    )


def _require_same_context(before: Stage0Context, after: Stage0Context) -> None:
    protocol.require(
        after.stage0_sha256 == before.stage0_sha256
        and after.pending_sha256 == before.pending_sha256
        and after.binding_sha256 == before.binding_sha256
        and after.opening_records == before.opening_records
        and after.model_inventory_sha256 == before.model_inventory_sha256
        and after.runtime_registry_sha256 == before.runtime_registry_sha256
        and after.code_registry_sha256 == before.code_registry_sha256
        and after.generator_sha256 == before.generator_sha256
        and after.generator_dependency_closure_sha256
        == before.generator_dependency_closure_sha256
        and after.media_runtime_packages == before.media_runtime_packages,
        "frozen Stage-0/model/runtime/code bytes changed during screening",
    )


def _standard_outputs(private_root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        private_root / GENERATION_DIRNAME,
        private_root / PUBLIC_PACKAGE_DIRNAME,
        private_root / PRIVATE_PACKAGE_DIRNAME,
        private_root / CUDA_LOCK_BASENAME,
    )


def _reserve_execution(context: Stage0Context, worker_count: int) -> tuple[Path, Path, Path, Path]:
    generation_dir, public_dir, private_dir, lock_path = _standard_outputs(
        context.private_root
    )
    protocol.require(worker_count == 1, "v3 screening permits exactly one GPU worker")
    for output in (generation_dir, public_dir, private_dir, lock_path):
        protocol.reject_forbidden_path(output)
        if os.path.lexists(output):
            raise FileExistsError(f"one-shot screening target already exists: {output}")
    lock_payload = {
        "protocol": LOCK_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "consumed_for_one_shot_screening",
        "stage0_registry_sha256": context.stage0_sha256,
        "worker_count": 1,
    }
    _write_bytes_exclusive(lock_path, _json_bytes(lock_payload), 0o600)
    try:
        generation_dir.mkdir(mode=0o700)
        reservation = {
            "protocol": SCREENING_RUN_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "reserved_one_shot",
            "stage0_registry_sha256": context.stage0_sha256,
            "selection_binding_sha256": context.binding_sha256,
            "candidate_manifest_sha256": context.opening_records[
                "candidate_manifest_576"
            ]["sha256"],
            "generation_spec_sha256": context.opening_records[
                "screening_generation_spec"
            ]["sha256"],
            "model_content_inventory_sha256": context.model_inventory_sha256,
            "runtime_registry_sha256": context.runtime_registry_sha256,
            "code_registry_sha256": context.code_registry_sha256,
        "generator_sha256": context.generator_sha256,
        "generator_dependency_closure_sha256": (
            context.generator_dependency_closure_sha256
        ),
        "media_runtime_packages": dict(context.media_runtime_packages),
        "worker_count": 1,
        }
        _write_json_exclusive(
            generation_dir / ".run_reservation_v3.json", reservation, 0o600
        )
        _write_json_exclusive(
            generation_dir / "execution_started_v3.json",
            {
                "protocol": STATUS_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "started_terminal_one_shot",
                "stage0_registry_sha256": context.stage0_sha256,
                "worker_count": 1,
            },
            0o600,
        )
    except BaseException as exc:
        # The permanent lock intentionally survives every post-consumption failure.
        raise ConsumedReservationFailure(
            "one-shot output reservation failed after lock acquisition"
        ) from exc
    return generation_dir, public_dir, private_dir, lock_path


def _write_prompt_file(
    generation_dir: Path, candidates: Sequence[Mapping[str, Any]]
) -> Path:
    lines: list[str] = []
    for row in candidates:
        prompt = row["canonical_prompt"]
        source = row["source_phrase"]
        protocol.require(
            isinstance(prompt, str)
            and isinstance(source, str)
            and prompt
            and source
            and not any(character in prompt for character in "\r\n")
            and not any(character in source for character in "\r\n")
            and "|" not in prompt
            and "|" not in source,
            "candidate prompt/source cannot be serialized for generation",
        )
        lines.append(
            f"{prompt} | {source} | registered v4 causal Original screening"
        )
    path = generation_dir / "prompts.txt"
    _write_bytes_exclusive(path, ("\n".join(lines) + "\n").encode("utf-8"), 0o600)
    return path


def generation_command(
    *,
    python_executable: str,
    generator_relative: str,
    prompt_path: Path,
    generation_dir: Path,
    screening_seed: int,
) -> list[str]:
    protocol.require(
        generator_relative == protocol.CODE_ARTIFACT_PATHS["generator"],
        "generator bootstrap target differs from code registry",
    )
    return [
        python_executable,
        "-I",
        "-c",
        GENERATOR_BOOTSTRAP,
        "--baseline",
        "clean",
        "--prompts",
        os.fspath(prompt_path),
        "--output-dir",
        os.fspath(generation_dir),
        "--model",
        "models/Wan2.1-T2V-1.3B-Diffusers",
        "--seeds",
        ",".join([str(screening_seed)] * protocol.CANDIDATE_COUNT),
        "--steps",
        "25",
        "--guidance-scale",
        "5",
        "--num-frames",
        "49",
        "--fps",
        "8",
        "--height",
        "480",
        "--width",
        "832",
        "--dtype",
        "bf16",
        "--device",
        "cuda",
        "--vae-slicing",
        "--vae-tiling",
    ]


def sanitized_worker_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    base = os.environ if source is None else source
    runtime_bin = os.path.realpath("models/.wan-runtime/bin")
    output = {
        "PATH": os.pathsep.join((runtime_bin, "/usr/bin", "/bin")),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if key in base:
            output[key] = base[key]
    return output


def _validate_generic_manifest(
    *,
    context: Stage0Context,
    generation_dir: Path,
    prompt_path: Path,
) -> tuple[Mapping[str, Any], list[Path]]:
    manifest_path = generation_dir / GENERIC_MANIFEST_BASENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("generic generator manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol.require_exact_keys(
        manifest,
        {
            "created_at_utc",
            "baseline",
            "pipeline",
            "model",
            "dry_run",
            "prompts",
            "generation",
            "items",
        },
        "generic generator manifest",
    )
    created = manifest["created_at_utc"]
    protocol.require(isinstance(created, str), "generator timestamp is invalid")
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generator timestamp is invalid") from exc
    protocol.require(
        parsed_created.tzinfo is not None,
        "generator timestamp must be timezone-aware",
    )
    expected_generation = {
        "baseline": "clean",
        "seed": 42,
        "seeds": [context.screening_seed] * protocol.CANDIDATE_COUNT,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
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
    protocol.require(
        manifest["baseline"] == "clean"
        and manifest["pipeline"] == "WanPipeline"
        and manifest["model"] == "models/Wan2.1-T2V-1.3B-Diffusers"
        and manifest["dry_run"] is False
        and manifest["prompts"] == os.fspath(prompt_path)
        and manifest["generation"] == expected_generation,
        "generic generator configuration differs from frozen Original contract",
    )
    candidates = context.candidate_payload["candidates"]
    items = manifest["items"]
    protocol.require(
        isinstance(items, list) and len(items) == protocol.CANDIDATE_COUNT,
        "generic generator item count mismatch",
    )
    videos: list[Path] = []
    expected_item_keys = {
        "index",
        "prompt",
        "target_concept",
        "expected_effect",
        "seed",
        "video_path",
    }
    videos_dir = generation_dir / "videos"
    protocol.require(
        videos_dir.is_dir() and not videos_dir.is_symlink(),
        "generic generator videos directory is missing",
    )
    for index, (candidate, item) in enumerate(zip(candidates, items)):
        protocol.require_exact_keys(item, expected_item_keys, "generator item")
        path = protocol._canonical_lexical_absolute(Path(item["video_path"]))
        protocol.require(
            item["index"] == index
            and item["prompt"] == candidate["canonical_prompt"]
            and item["target_concept"] == candidate["source_phrase"]
            and item["expected_effect"]
            == "registered v4 causal Original screening"
            and item["seed"] == context.screening_seed
            and path.parent == videos_dir
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_nlink == 1
            and path.stat().st_size > 0,
            "generic generator item/video binding mismatch",
        )
        videos.append(path)
    protocol.require(
        len({path.resolve(strict=True) for path in videos})
        == protocol.CANDIDATE_COUNT
        and set(videos_dir.iterdir()) == set(videos),
        "raw video inventory is duplicate or inexact",
    )
    protocol.require(
        {entry.name for entry in generation_dir.iterdir()}
        == {
            ".run_reservation_v3.json",
            "execution_started_v3.json",
            "generator_output_v3.log",
            "prompts.txt",
            GENERIC_MANIFEST_BASENAME,
            "videos",
        },
        "generic generator created an unexpected raw-run artifact",
    )
    return manifest, videos


def _decode_video(
    path: Path, *, collect_composite_frames: bool
) -> tuple[dict[str, int], list[Any]]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for exact screening decode") from exc
    selected: dict[int, Any] = {}
    count = 0
    width: int | None = None
    height: int | None = None
    with av.open(os.fspath(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
            raise ValueError(
                "screening video must contain one video stream and no audio"
            )
        stream = container.streams.video[0]
        rate = stream.average_rate
        protocol.require(
            rate is not None and rate.numerator == 8 * rate.denominator,
            "screening video fps mismatch",
        )
        for index, frame in enumerate(container.decode(video=0)):
            protocol.require(
                frame.width == 832
                and frame.height == 480
                and bool(frame.planes)
                and all(plane.buffer_size > 0 for plane in frame.planes),
                "screening frame dimensions mismatch",
            )
            width = frame.width
            height = frame.height
            if collect_composite_frames and index in FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            count += 1
    protocol.require(
        count == 49 and width == 832 and height == 480,
        "screening video does not decode to exact 49 nonempty frames",
    )
    if collect_composite_frames:
        protocol.require(
            set(selected) == set(FRAME_INDICES),
            "screening composite frame inventory is incomplete",
        )
    return {
        "frame_count": 49,
        "width": 832,
        "height": 480,
        "fps_numerator": 8,
        "fps_denominator": 1,
    }, [selected[index] for index in FRAME_INDICES] if selected else []


def _raw_video_records(
    context: Stage0Context, videos: Sequence[Path]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (candidate, video) in enumerate(
        zip(context.candidate_payload["candidates"], videos)
    ):
        decode, _ = _decode_video(video, collect_composite_frames=False)
        records.append(
            {
                "index": index,
                "case_id": candidate["case_id"],
                "video_name": video.name,
                "size_bytes": video.stat().st_size,
                "sha256": protocol.sha256_file(video),
                "prompt_sha256": hashlib.sha256(
                    candidate["canonical_prompt"].encode("utf-8")
                ).hexdigest(),
                "screening_seed_sha256": context.opening_records[
                    "screening_seed"
                ]["sha256"],
                **decode,
            }
        )
    return records


def validate_raw_video_inventory(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "stage0_registry_sha256",
            "candidate_manifest_sha256",
            "generation_spec_sha256",
            "videos",
        },
        "raw video inventory",
    )
    protocol.require(
        payload["protocol"] == RAW_INVENTORY_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "complete"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT
        and all(
            protocol.is_hex64(payload[key])
            for key in (
                "stage0_registry_sha256",
                "candidate_manifest_sha256",
                "generation_spec_sha256",
            )
        ),
        "raw video inventory identity mismatch",
    )
    videos = payload["videos"]
    protocol.require(
        isinstance(videos, list) and len(videos) == protocol.CANDIDATE_COUNT,
        "raw video inventory row count mismatch",
    )
    row_keys = {
        "index",
        "case_id",
        "video_name",
        "size_bytes",
        "sha256",
        "prompt_sha256",
        "screening_seed_sha256",
        "frame_count",
        "width",
        "height",
        "fps_numerator",
        "fps_denominator",
    }
    for index, row in enumerate(videos):
        protocol.require_exact_keys(row, row_keys, "raw video row")
        protocol.require(
            row["index"] == index
            and isinstance(row["case_id"], str)
            and row["case_id"]
            and isinstance(row["video_name"], str)
            and Path(row["video_name"]).name == row["video_name"]
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0
            and all(
                protocol.is_hex64(row[key])
                for key in (
                    "sha256",
                    "prompt_sha256",
                    "screening_seed_sha256",
                )
            )
            and {
                key: row[key]
                for key in (
                    "frame_count",
                    "width",
                    "height",
                    "fps_numerator",
                    "fps_denominator",
                )
            }
            == {
                "frame_count": 49,
                "width": 832,
                "height": 480,
                "fps_numerator": 8,
                "fps_denominator": 1,
            },
            "raw video row contract mismatch",
        )
    protocol.require(
        len({row["case_id"] for row in videos}) == protocol.CANDIDATE_COUNT
        and len({row["video_name"] for row in videos})
        == protocol.CANDIDATE_COUNT,
        "raw video IDs/names are duplicate",
    )
    return payload


def validate_generation_manifest(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    hash_keys = {
        "stage0_registry_sha256",
        "selection_binding_sha256",
        "candidate_manifest_sha256",
        "candidate_graph_sha256",
        "generation_spec_sha256",
        "screening_seed_sha256",
        "model_content_inventory_sha256",
        "runtime_registry_sha256",
        "code_registry_sha256",
        "generator_sha256",
        "generator_dependency_closure_sha256",
        "cuda_lock_sha256",
        "run_reservation_sha256",
        "execution_started_sha256",
        "generator_log_sha256",
        "prompt_file_sha256",
        "generic_generation_manifest_sha256",
        "raw_video_inventory_sha256",
    }
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "worker_count",
            *hash_keys,
            "media_runtime_packages",
            "videos",
        },
        "screening generation manifest",
    )
    protocol.require(
        payload["protocol"] == GENERATION_MANIFEST_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "complete_original_screening_generation"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT
        and payload["worker_count"] == 1
        and all(protocol.is_hex64(payload[key]) for key in hash_keys)
        and isinstance(payload["media_runtime_packages"], dict)
        and set(payload["media_runtime_packages"])
        == set(authorizer.MEDIA_RUNTIME_DISTRIBUTIONS)
        and all(
            isinstance(value, str) and value
            for value in payload["media_runtime_packages"].values()
        ),
        "screening generation manifest identity/provenance mismatch",
    )
    validate_raw_video_inventory(
        {
            "protocol": RAW_INVENTORY_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "complete",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "stage0_registry_sha256": payload["stage0_registry_sha256"],
            "candidate_manifest_sha256": payload[
                "candidate_manifest_sha256"
            ],
            "generation_spec_sha256": payload["generation_spec_sha256"],
            "videos": payload["videos"],
        }
    )
    return payload


def _write_raw_manifests(
    *,
    context: Stage0Context,
    generation_dir: Path,
    prompt_path: Path,
    videos: Sequence[Path],
) -> tuple[dict[str, Any], Path, Path]:
    records = _raw_video_records(context, videos)
    raw_inventory = {
        "protocol": RAW_INVENTORY_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "complete",
        "candidate_count": protocol.CANDIDATE_COUNT,
        "stage0_registry_sha256": context.stage0_sha256,
        "candidate_manifest_sha256": context.opening_records[
            "candidate_manifest_576"
        ]["sha256"],
        "generation_spec_sha256": context.opening_records[
            "screening_generation_spec"
        ]["sha256"],
        "videos": records,
    }
    validate_raw_video_inventory(raw_inventory)
    raw_path = generation_dir / RAW_INVENTORY_BASENAME
    _write_json_exclusive(raw_path, raw_inventory, 0o600)
    generic_path = generation_dir / GENERIC_MANIFEST_BASENAME
    generation_manifest = {
        "protocol": GENERATION_MANIFEST_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "complete_original_screening_generation",
        "candidate_count": protocol.CANDIDATE_COUNT,
        "worker_count": 1,
        "stage0_registry_sha256": context.stage0_sha256,
        "selection_binding_sha256": context.binding_sha256,
        "candidate_manifest_sha256": context.opening_records[
            "candidate_manifest_576"
        ]["sha256"],
        "candidate_graph_sha256": context.opening_records[
            "candidate_graph_576"
        ]["sha256"],
        "generation_spec_sha256": context.opening_records[
            "screening_generation_spec"
        ]["sha256"],
        "screening_seed_sha256": context.opening_records["screening_seed"][
            "sha256"
        ],
        "model_content_inventory_sha256": context.model_inventory_sha256,
        "runtime_registry_sha256": context.runtime_registry_sha256,
        "code_registry_sha256": context.code_registry_sha256,
        "generator_sha256": context.generator_sha256,
        "generator_dependency_closure_sha256": (
            context.generator_dependency_closure_sha256
        ),
        "media_runtime_packages": dict(context.media_runtime_packages),
        "cuda_lock_sha256": protocol.sha256_file(
            context.private_root / CUDA_LOCK_BASENAME
        ),
        "run_reservation_sha256": protocol.sha256_file(
            generation_dir / ".run_reservation_v3.json"
        ),
        "execution_started_sha256": protocol.sha256_file(
            generation_dir / "execution_started_v3.json"
        ),
        "generator_log_sha256": protocol.sha256_file(
            generation_dir / "generator_output_v3.log"
        ),
        "prompt_file_sha256": protocol.sha256_file(prompt_path),
        "generic_generation_manifest_sha256": protocol.sha256_file(generic_path),
        "raw_video_inventory_sha256": protocol.sha256_file(raw_path),
        "videos": records,
    }
    validate_generation_manifest(generation_manifest)
    manifest_path = generation_dir / GENERATION_MANIFEST_BASENAME
    _write_json_exclusive(manifest_path, generation_manifest, 0o600)
    return generation_manifest, manifest_path, raw_path


def _build_composite(path: Path, frames: Sequence[Any]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for screening composites") from exc
    protocol.require(len(frames) == len(FRAME_INDICES), "composite frame count mismatch")
    frame_width = 208
    frame_height = 120
    header = 28
    image = Image.new(
        "RGB", (frame_width * len(frames), header + frame_height), "white"
    )
    draw = ImageDraw.Draw(image)
    for column, (frame_index, frame) in enumerate(zip(FRAME_INDICES, frames)):
        x = column * frame_width
        draw.text((x + 6, 6), f"frame {frame_index}", fill="black")
        image.paste(frame.resize((frame_width, frame_height)), (x, header))
    image.save(path, format="JPEG", quality=92)
    path.chmod(0o600)


def _tree_snapshot(root: Path) -> tuple[dict[str, Any], ...]:
    protocol.require(root.is_dir() and not root.is_symlink(), "snapshot root invalid")
    records: list[dict[str, Any]] = []
    for entry in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        protocol.require(not entry.is_symlink(), "package snapshot contains symlink")
        relative = entry.relative_to(root).as_posix()
        info = entry.stat()
        if entry.is_dir():
            records.append(
                {
                    "path": relative,
                    "kind": "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        elif entry.is_file():
            protocol.require(info.st_nlink == 1, "package snapshot contains hardlink")
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "size_bytes": info.st_size,
                    "sha256": protocol.sha256_file(entry),
                }
            )
        else:
            raise ValueError("package snapshot contains non-regular entry")
    return tuple(records)


def _validate_public_package_payload(payload: Mapping[str, Any]) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "stage0_registry_sha256",
            "generation_manifest_sha256",
            "review_order_sha256",
            "review_template_sha256",
            "anonymous_video_inventory_sha256",
            "composite_inventory_sha256",
        },
        "public screening package",
    )
    protocol.require(
        payload["protocol"] == PUBLIC_PACKAGE_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_screening_review"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT,
        "public screening package identity mismatch",
    )
    for key in (
        "stage0_registry_sha256",
        "generation_manifest_sha256",
        "review_order_sha256",
        "review_template_sha256",
        "anonymous_video_inventory_sha256",
        "composite_inventory_sha256",
    ):
        protocol.require(protocol.is_hex64(payload[key]), f"public {key} invalid")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    protocol.require(
        not any(
            token in encoded.casefold()
            for token in (
                '"path"',
                '"seed"',
                '"identity"',
                '"prompt"',
                '"case_id"',
                "source_phrase",
                "receiver_phrase",
            )
        ),
        "public screening package leaks identity/seed/path fields",
    )


def _validate_public_commitment_payload(payload: Mapping[str, Any]) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "stage0_registry_sha256",
            "pending_commitment_sha256",
            "selection_binding_sha256",
            "candidate_manifest_sha256",
            "candidate_graph_sha256",
            "generation_spec_sha256",
            "generation_manifest_sha256",
            "raw_video_inventory_sha256",
            "model_content_inventory_sha256",
            "runtime_registry_sha256",
            "code_registry_sha256",
            "generator_sha256",
            "generator_dependency_closure_sha256",
            "media_runtime_packages",
            "cuda_lock_sha256",
            "run_reservation_sha256",
            "execution_started_sha256",
            "generator_log_sha256",
            "prompt_file_sha256",
            "generic_generation_manifest_sha256",
            "review_order_sha256",
            "review_template_sha256",
            "answer_key_sha256",
            "candidate_binding_sha256",
            "anonymous_video_inventory_sha256",
            "composite_inventory_sha256",
            "public_manifest_sha256",
            "private_manifest_sha256",
            "raw_media",
            "anonymous_media",
            "composites",
        },
        "screening package commitment",
    )
    protocol.require(
        payload["protocol"] == PACKAGE_COMMITMENT_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "committed_before_any_screening_review"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT,
        "screening package commitment identity mismatch",
    )
    for key, value in payload.items():
        if key.endswith("_sha256"):
            protocol.require(protocol.is_hex64(value), f"commitment {key} invalid")
    protocol.require(
        isinstance(payload["media_runtime_packages"], dict)
        and set(payload["media_runtime_packages"])
        == set(authorizer.MEDIA_RUNTIME_DISTRIBUTIONS)
        and all(
            isinstance(value, str) and value
            for value in payload["media_runtime_packages"].values()
        ),
        "commitment media runtime packages invalid",
    )
    expected_ids = {f"s{index:03d}" for index in range(protocol.CANDIDATE_COUNT)}
    for name in ("raw_media", "anonymous_media", "composites"):
        inventory = payload[name]
        protocol.require(
            isinstance(inventory, dict) and set(inventory) == expected_ids,
            f"commitment {name} inventory mismatch",
        )
        for review_id, record in inventory.items():
            protocol.require_exact_keys(
                record,
                {"sha256", "size_bytes"},
                f"commitment {name}/{review_id}",
            )
            protocol.require(
                protocol.is_hex64(record["sha256"])
                and type(record["size_bytes"]) is int
                and record["size_bytes"] > 0,
                f"commitment {name}/{review_id} record invalid",
            )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    protocol.require(
        not any(
            token in encoded.casefold()
            for token in (
                '"path"',
                '"seed"',
                '"identity"',
                '"prompt"',
                '"case_id"',
                "source_phrase",
                "receiver_phrase",
            )
        ),
        "public package commitment leaks identity/seed/path fields",
    )


def validate_candidate_binding(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {"protocol", "dataset_version", "status", "candidate_count", "rows"},
        "screening candidate binding",
    )
    protocol.require(
        payload["protocol"] == CANDIDATE_BINDING_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_candidate_binding"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT,
        "screening candidate binding identity mismatch",
    )
    rows = payload["rows"]
    protocol.require(
        isinstance(rows, list) and len(rows) == protocol.CANDIDATE_COUNT,
        "screening candidate binding row count mismatch",
    )
    for index, row in enumerate(rows):
        protocol.require_exact_keys(
            row,
            {
                "review_id",
                "candidate",
                "raw_video_sha256",
                "anonymous_video_sha256",
                "composite_sha256",
            },
            "screening candidate binding row",
        )
        protocol.require(
            row["review_id"] == f"s{index:03d}"
            and isinstance(row["candidate"], dict)
            and all(
                protocol.is_hex64(row[key])
                for key in (
                    "raw_video_sha256",
                    "anonymous_video_sha256",
                    "composite_sha256",
                )
            ),
            "screening candidate binding row mismatch",
        )
        protocol.candidate_record_bytes(row["candidate"])
    return payload


def validate_anonymous_inventory(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "review_order_sha256",
            "videos",
        },
        "anonymous video inventory",
    )
    protocol.require(
        payload["protocol"] == ANONYMOUS_INVENTORY_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_screening_review"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT
        and protocol.is_hex64(payload["review_order_sha256"])
        and isinstance(payload["videos"], list)
        and len(payload["videos"]) == protocol.CANDIDATE_COUNT,
        "anonymous video inventory identity/count mismatch",
    )
    for index, row in enumerate(payload["videos"]):
        protocol.require_exact_keys(
            row,
            {
                "review_id",
                "sha256",
                "size_bytes",
                "frame_count",
                "width",
                "height",
                "fps_numerator",
                "fps_denominator",
            },
            "anonymous video row",
        )
        protocol.require(
            row["review_id"] == f"s{index:03d}"
            and protocol.is_hex64(row["sha256"])
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0
            and {
                key: row[key]
                for key in (
                    "frame_count",
                    "width",
                    "height",
                    "fps_numerator",
                    "fps_denominator",
                )
            }
            == {
                "frame_count": 49,
                "width": 832,
                "height": 480,
                "fps_numerator": 8,
                "fps_denominator": 1,
            },
            "anonymous video row mismatch",
        )
    return payload


def validate_composite_inventory(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "review_order_sha256",
            "frame_indices",
            "composites",
        },
        "composite inventory",
    )
    protocol.require(
        payload["protocol"] == COMPOSITE_INVENTORY_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_screening_review"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT
        and protocol.is_hex64(payload["review_order_sha256"])
        and payload["frame_indices"] == list(FRAME_INDICES)
        and isinstance(payload["composites"], list)
        and len(payload["composites"]) == protocol.CANDIDATE_COUNT,
        "composite inventory identity/count mismatch",
    )
    for index, row in enumerate(payload["composites"]):
        protocol.require_exact_keys(
            row, {"review_id", "sha256", "size_bytes"}, "composite row"
        )
        protocol.require(
            row["review_id"] == f"s{index:03d}"
            and protocol.is_hex64(row["sha256"])
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0,
            "composite row mismatch",
        )
    return payload


def validate_private_package_manifest(
    payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    hash_keys = {
        "stage0_registry_sha256",
        "selection_binding_sha256",
        "candidate_manifest_sha256",
        "candidate_graph_sha256",
        "generation_spec_sha256",
        "generation_manifest_sha256",
        "raw_video_inventory_sha256",
        "review_order_sha256",
        "review_template_sha256",
        "answer_key_sha256",
        "candidate_binding_sha256",
        "anonymous_video_inventory_sha256",
        "composite_inventory_sha256",
        "public_manifest_sha256",
        "generator_dependency_closure_sha256",
    }
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            *hash_keys,
            "media_runtime_packages",
            "raw_media",
            "anonymous_media",
            "composites",
        },
        "private screening package manifest",
    )
    protocol.require(
        payload["protocol"] == PRIVATE_PACKAGE_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_screening_review"
        and payload["candidate_count"] == protocol.CANDIDATE_COUNT
        and all(protocol.is_hex64(payload[key]) for key in hash_keys)
        and isinstance(payload["media_runtime_packages"], dict)
        and set(payload["media_runtime_packages"])
        == set(authorizer.MEDIA_RUNTIME_DISTRIBUTIONS)
        and all(
            isinstance(value, str) and value
            for value in payload["media_runtime_packages"].values()
        ),
        "private screening package manifest identity/provenance mismatch",
    )
    expected_ids = {f"s{index:03d}" for index in range(protocol.CANDIDATE_COUNT)}
    for name in ("raw_media", "anonymous_media", "composites"):
        inventory = payload[name]
        protocol.require(
            isinstance(inventory, dict) and set(inventory) == expected_ids,
            f"private package {name} inventory mismatch",
        )
        for record in inventory.values():
            protocol.require_exact_keys(
                record, {"sha256", "size_bytes"}, f"private package {name} row"
            )
            protocol.require(
                protocol.is_hex64(record["sha256"])
                and type(record["size_bytes"]) is int
                and record["size_bytes"] > 0,
                f"private package {name} record invalid",
            )
    return payload


def validate_package_commitment(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _validate_public_commitment_payload(payload)
    return payload


def build_screening_package(
    *,
    context: Stage0Context,
    generation_manifest: Mapping[str, Any],
    generation_manifest_path: Path,
    raw_inventory_path: Path,
    videos: Sequence[Path],
    public_dir: Path,
    private_dir: Path,
    revalidate: Callable[[], None],
) -> dict[str, Any]:
    loaded_generation = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    loaded_raw = json.loads(raw_inventory_path.read_text(encoding="utf-8"))
    protocol.require(
        loaded_generation == dict(generation_manifest)
        and loaded_generation.get("protocol") == GENERATION_MANIFEST_PROTOCOL
        and loaded_generation.get("status")
        == "complete_original_screening_generation"
        and loaded_generation.get("candidate_count") == protocol.CANDIDATE_COUNT
        and isinstance(loaded_generation.get("videos"), list)
        and len(loaded_generation["videos"]) == protocol.CANDIDATE_COUNT
        and loaded_raw.get("protocol") == RAW_INVENTORY_PROTOCOL
        and loaded_raw.get("status") == "complete"
        and loaded_raw.get("videos") == loaded_generation["videos"],
        "screening package input manifests are not exact and cross-bound",
    )
    for index, (candidate, video, record) in enumerate(
        zip(
            context.candidate_payload["candidates"],
            videos,
            loaded_generation["videos"],
        )
    ):
        protocol.require(
            record["index"] == index
            and record["case_id"] == candidate["case_id"]
            and record["video_name"] == video.name
            and record["sha256"] == protocol.sha256_file(video)
            and record["size_bytes"] == video.stat().st_size,
            "screening package video differs from generation manifest",
        )
    for target in (public_dir, private_dir):
        protocol.validate_private_path(
            context.private_root, target, must_exist=False
        )
        if os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite screening package: {target}")
    protocol.require(
        public_dir == context.private_root / PUBLIC_PACKAGE_DIRNAME
        and private_dir == context.private_root / PRIVATE_PACKAGE_DIRNAME,
        "screening public/private package paths are not exact siblings",
    )
    staging_root = Path(
        tempfile.mkdtemp(prefix=".causal-screening-package-v3-", dir=context.private_root)
    )
    staging_root.chmod(0o700)
    work_public = staging_root / "public"
    work_private = staging_root / "private"
    published: list[Path] = []
    reserved_targets: list[Path] = []
    try:
        work_public.mkdir(mode=0o700)
        work_private.mkdir(mode=0o700)
        media_dir = work_public / "media"
        composite_dir = work_public / "composites"
        media_dir.mkdir(mode=0o700)
        composite_dir.mkdir(mode=0o700)

        public_rows: list[dict[str, Any]] = []
        answer_rows: list[dict[str, Any]] = []
        projection_rows: list[dict[str, Any]] = []
        raw_hashes: dict[str, dict[str, Any]] = {}
        anonymous_hashes: dict[str, dict[str, Any]] = {}
        composite_hashes: dict[str, dict[str, Any]] = {}
        review_ids: list[str] = []
        for index, (candidate, raw_video) in enumerate(
            zip(context.candidate_payload["candidates"], videos)
        ):
            review_id = f"s{index:03d}"
            review_ids.append(review_id)
            anonymous = media_dir / f"{review_id}.mp4"
            shutil.copyfile(raw_video, anonymous)
            anonymous.chmod(0o600)
            protocol.require(
                not anonymous.is_symlink()
                and not anonymous.samefile(raw_video)
                and anonymous.stat().st_nlink == 1
                and protocol.sha256_file(anonymous)
                == protocol.sha256_file(raw_video),
                "anonymous video is not an independent byte-exact copy",
            )
            _, frames = _decode_video(
                anonymous, collect_composite_frames=True
            )
            composite = composite_dir / f"{review_id}.jpg"
            _build_composite(composite, frames)
            protocol.require(
                composite.is_file()
                and not composite.is_symlink()
                and composite.stat().st_nlink == 1
                and composite.stat().st_size > 0,
                "screening composite creation failed",
            )
            raw_record = {
                "sha256": protocol.sha256_file(raw_video),
                "size_bytes": raw_video.stat().st_size,
            }
            anonymous_record = {
                "sha256": protocol.sha256_file(anonymous),
                "size_bytes": anonymous.stat().st_size,
            }
            composite_record = {
                "sha256": protocol.sha256_file(composite),
                "size_bytes": composite.stat().st_size,
            }
            raw_hashes[review_id] = raw_record
            anonymous_hashes[review_id] = anonymous_record
            composite_hashes[review_id] = composite_record
            public_rows.append(
                {
                    "review_id": review_id,
                    "candidate_video_path": f"media/{review_id}.mp4",
                    "candidate_video_sha256": anonymous_record["sha256"],
                    "composite_path": f"composites/{review_id}.jpg",
                    "composite_sha256": composite_record["sha256"],
                    **{name: "" for name in REVIEW_FIELDS},
                    "notes": "",
                }
            )
            answer_rows.append(
                {
                    "review_id": review_id,
                    "candidate_index": index,
                    "case_id": candidate["case_id"],
                    "raw_video_sha256": raw_record["sha256"],
                    "anonymous_video_sha256": anonymous_record["sha256"],
                    "composite_sha256": composite_record["sha256"],
                }
            )
            projection_rows.append(
                {
                    "review_id": review_id,
                    "candidate": dict(candidate),
                    "raw_video_sha256": raw_record["sha256"],
                    "anonymous_video_sha256": anonymous_record["sha256"],
                    "composite_sha256": composite_record["sha256"],
                }
            )

        review_path = work_public / REVIEW_TEMPLATE_BASENAME
        _write_csv_exclusive(review_path, REVIEW_HEADER, public_rows, 0o600)
        answer_header = (
            "review_id",
            "candidate_index",
            "case_id",
            "raw_video_sha256",
            "anonymous_video_sha256",
            "composite_sha256",
        )
        answer_path = work_private / ANSWER_KEY_BASENAME
        _write_csv_exclusive(answer_path, answer_header, answer_rows, 0o600)
        candidate_binding_payload = {
            "protocol": CANDIDATE_BINDING_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_candidate_binding",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "rows": projection_rows,
        }
        validate_candidate_binding(candidate_binding_payload)
        candidate_binding_path = work_private / CANDIDATE_BINDING_BASENAME
        _write_json_exclusive(
            candidate_binding_path, candidate_binding_payload, 0o600
        )

        review_order_sha256 = protocol.sha256_bytes(
            protocol.canonical_json_bytes(review_ids)
        )
        anonymous_inventory = {
            "protocol": ANONYMOUS_INVENTORY_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_before_screening_review",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "review_order_sha256": review_order_sha256,
            "videos": [
                {
                    "review_id": review_id,
                    **anonymous_hashes[review_id],
                    "frame_count": 49,
                    "width": 832,
                    "height": 480,
                    "fps_numerator": 8,
                    "fps_denominator": 1,
                }
                for review_id in review_ids
            ],
        }
        validate_anonymous_inventory(anonymous_inventory)
        anonymous_inventory_path = (
            work_public / ANONYMOUS_INVENTORY_BASENAME
        )
        _write_json_exclusive(
            anonymous_inventory_path, anonymous_inventory, 0o600
        )
        composite_inventory = {
            "protocol": COMPOSITE_INVENTORY_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_before_screening_review",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "review_order_sha256": review_order_sha256,
            "frame_indices": list(FRAME_INDICES),
            "composites": [
                {"review_id": review_id, **composite_hashes[review_id]}
                for review_id in review_ids
            ],
        }
        validate_composite_inventory(composite_inventory)
        composite_inventory_path = work_public / COMPOSITE_INVENTORY_BASENAME
        _write_json_exclusive(
            composite_inventory_path, composite_inventory, 0o600
        )
        public_manifest = {
            "protocol": PUBLIC_PACKAGE_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_before_screening_review",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "stage0_registry_sha256": context.stage0_sha256,
            "generation_manifest_sha256": protocol.sha256_file(
                generation_manifest_path
            ),
            "review_order_sha256": review_order_sha256,
            "review_template_sha256": protocol.sha256_file(review_path),
            "anonymous_video_inventory_sha256": protocol.sha256_file(
                anonymous_inventory_path
            ),
            "composite_inventory_sha256": protocol.sha256_file(
                composite_inventory_path
            ),
        }
        _validate_public_package_payload(public_manifest)
        public_manifest_path = work_public / PUBLIC_MANIFEST_BASENAME
        _write_json_exclusive(public_manifest_path, public_manifest, 0o600)

        private_manifest = {
            "protocol": PRIVATE_PACKAGE_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen_before_screening_review",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "stage0_registry_sha256": context.stage0_sha256,
            "selection_binding_sha256": context.binding_sha256,
            "candidate_manifest_sha256": context.opening_records[
                "candidate_manifest_576"
            ]["sha256"],
            "candidate_graph_sha256": context.opening_records[
                "candidate_graph_576"
            ]["sha256"],
            "generation_spec_sha256": context.opening_records[
                "screening_generation_spec"
            ]["sha256"],
            "generation_manifest_sha256": protocol.sha256_file(
                generation_manifest_path
            ),
            "raw_video_inventory_sha256": protocol.sha256_file(
                raw_inventory_path
            ),
            "review_order_sha256": review_order_sha256,
            "review_template_sha256": protocol.sha256_file(review_path),
            "answer_key_sha256": protocol.sha256_file(answer_path),
            "candidate_binding_sha256": protocol.sha256_file(
                candidate_binding_path
            ),
            "anonymous_video_inventory_sha256": protocol.sha256_file(
                anonymous_inventory_path
            ),
            "composite_inventory_sha256": protocol.sha256_file(
                composite_inventory_path
            ),
            "public_manifest_sha256": protocol.sha256_file(
                public_manifest_path
            ),
            "generator_dependency_closure_sha256": (
                context.generator_dependency_closure_sha256
            ),
            "media_runtime_packages": dict(context.media_runtime_packages),
            "raw_media": raw_hashes,
            "anonymous_media": anonymous_hashes,
            "composites": composite_hashes,
        }
        validate_private_package_manifest(private_manifest)
        private_manifest_path = work_private / PRIVATE_MANIFEST_BASENAME
        _write_json_exclusive(private_manifest_path, private_manifest, 0o600)

        commitment = {
            "protocol": PACKAGE_COMMITMENT_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "committed_before_any_screening_review",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "stage0_registry_sha256": context.stage0_sha256,
            "pending_commitment_sha256": context.pending_sha256,
            "selection_binding_sha256": context.binding_sha256,
            "candidate_manifest_sha256": context.opening_records[
                "candidate_manifest_576"
            ]["sha256"],
            "candidate_graph_sha256": context.opening_records[
                "candidate_graph_576"
            ]["sha256"],
            "generation_spec_sha256": context.opening_records[
                "screening_generation_spec"
            ]["sha256"],
            "generation_manifest_sha256": protocol.sha256_file(
                generation_manifest_path
            ),
            "raw_video_inventory_sha256": protocol.sha256_file(
                raw_inventory_path
            ),
            "model_content_inventory_sha256": context.model_inventory_sha256,
            "runtime_registry_sha256": context.runtime_registry_sha256,
            "code_registry_sha256": context.code_registry_sha256,
            "generator_sha256": context.generator_sha256,
            "generator_dependency_closure_sha256": (
                context.generator_dependency_closure_sha256
            ),
            "media_runtime_packages": dict(context.media_runtime_packages),
            "cuda_lock_sha256": generation_manifest["cuda_lock_sha256"],
            "run_reservation_sha256": generation_manifest[
                "run_reservation_sha256"
            ],
            "execution_started_sha256": generation_manifest[
                "execution_started_sha256"
            ],
            "generator_log_sha256": generation_manifest[
                "generator_log_sha256"
            ],
            "prompt_file_sha256": generation_manifest["prompt_file_sha256"],
            "generic_generation_manifest_sha256": generation_manifest[
                "generic_generation_manifest_sha256"
            ],
            "review_order_sha256": review_order_sha256,
            "review_template_sha256": protocol.sha256_file(review_path),
            "answer_key_sha256": protocol.sha256_file(answer_path),
            "candidate_binding_sha256": protocol.sha256_file(
                candidate_binding_path
            ),
            "anonymous_video_inventory_sha256": protocol.sha256_file(
                anonymous_inventory_path
            ),
            "composite_inventory_sha256": protocol.sha256_file(
                composite_inventory_path
            ),
            "public_manifest_sha256": protocol.sha256_file(
                public_manifest_path
            ),
            "private_manifest_sha256": protocol.sha256_file(
                private_manifest_path
            ),
            "raw_media": raw_hashes,
            "anonymous_media": anonymous_hashes,
            "composites": composite_hashes,
        }
        validate_package_commitment(commitment)
        private_commitment_path = work_private / PACKAGE_COMMITMENT_BASENAME
        commitment_raw = _json_bytes(commitment)
        _write_bytes_exclusive(private_commitment_path, commitment_raw, 0o600)

        protocol.require(
            json.loads(candidate_binding_path.read_text(encoding="utf-8"))
            == candidate_binding_payload
            and json.loads(anonymous_inventory_path.read_text(encoding="utf-8"))
            == anonymous_inventory
            and json.loads(composite_inventory_path.read_text(encoding="utf-8"))
            == composite_inventory
            and json.loads(public_manifest_path.read_text(encoding="utf-8"))
            == public_manifest
            and json.loads(private_manifest_path.read_text(encoding="utf-8"))
            == private_manifest
            and private_commitment_path.read_bytes() == commitment_raw,
            "staged screening package JSON bytes changed before publication",
        )
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            observed_review = list(reader)
            protocol.require(
                tuple(reader.fieldnames or ()) == REVIEW_HEADER,
                "staged review template header mismatch",
            )
        with answer_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            observed_answer = list(reader)
            protocol.require(
                tuple(reader.fieldnames or ()) == answer_header,
                "staged answer-key header mismatch",
            )
        protocol.require(
            len(observed_review) == protocol.CANDIDATE_COUNT
            and all(
                row["review_id"] == f"s{index:03d}"
                and row["candidate_video_path"]
                == f"media/s{index:03d}.mp4"
                and row["candidate_video_sha256"]
                == anonymous_hashes[f"s{index:03d}"]["sha256"]
                and row["composite_path"]
                == f"composites/s{index:03d}.jpg"
                and row["composite_sha256"]
                == composite_hashes[f"s{index:03d}"]["sha256"]
                and all(row[field] == "" for field in (*REVIEW_FIELDS, "notes"))
                for index, row in enumerate(observed_review)
            )
            and len(observed_answer) == protocol.CANDIDATE_COUNT
            and all(
                row["review_id"] == f"s{index:03d}"
                and row["candidate_index"] == str(index)
                and row["case_id"]
                == context.candidate_payload["candidates"][index]["case_id"]
                for index, row in enumerate(observed_answer)
            ),
            "staged screening CSV rows changed before publication",
        )
        protocol.require(
            {entry.name for entry in media_dir.iterdir()}
            == {f"s{index:03d}.mp4" for index in range(protocol.CANDIDATE_COUNT)}
            and {entry.name for entry in composite_dir.iterdir()}
            == {f"s{index:03d}.jpg" for index in range(protocol.CANDIDATE_COUNT)}
            and all(
                raw_hashes[review_id]
                == {
                    "sha256": protocol.sha256_file(videos[index]),
                    "size_bytes": videos[index].stat().st_size,
                }
                and anonymous_hashes[review_id]
                == {
                    "sha256": protocol.sha256_file(
                        media_dir / f"{review_id}.mp4"
                    ),
                    "size_bytes": (media_dir / f"{review_id}.mp4").stat().st_size,
                }
                and composite_hashes[review_id]
                == {
                    "sha256": protocol.sha256_file(
                        composite_dir / f"{review_id}.jpg"
                    ),
                    "size_bytes": (
                        composite_dir / f"{review_id}.jpg"
                    ).stat().st_size,
                }
                for index, review_id in enumerate(review_ids)
            ),
            "staged anonymous/composite file inventory changed",
        )
        staged_public_snapshot = _tree_snapshot(work_public)
        staged_private_snapshot = _tree_snapshot(work_private)
        revalidate()
        if os.path.lexists(private_dir) or os.path.lexists(public_dir):
            raise FileExistsError("screening package target appeared before publication")
        try:
            private_dir.mkdir(mode=0o700)
            reserved_targets.append(private_dir)
            public_dir.mkdir(mode=0o700)
            reserved_targets.append(public_dir)
        except BaseException:
            for reserved in reversed(reserved_targets):
                if reserved.is_dir() and not reserved.is_symlink():
                    reserved.rmdir()
            raise
        protocol.require(
            _tree_snapshot(work_public) == staged_public_snapshot
            and _tree_snapshot(work_private) == staged_private_snapshot,
            "staged screening package changed after final provenance revalidation",
        )
        os.rename(work_private, private_dir)
        reserved_targets.remove(private_dir)
        published.append(private_dir)
        os.rename(work_public, public_dir)
        reserved_targets.remove(public_dir)
        published.append(public_dir)
        return commitment
    except BaseException:
        for published_path in reversed(published):
            if published_path.is_dir() and not published_path.is_symlink():
                shutil.rmtree(published_path)
        for reserved in reversed(reserved_targets):
            if reserved.is_dir() and not reserved.is_symlink():
                reserved.rmdir()
        raise
    finally:
        if staging_root.is_dir() and not staging_root.is_symlink():
            shutil.rmtree(staging_root)


def _publish_invalid_outcome(
    *,
    project_root: Path,
    stage0_sha256: str,
    failure_phase: str,
    reason_code: str,
    generation_manifest_sha256: str | None,
) -> None:
    payload = {
        "protocol": protocol.INVALID_OUTCOME_PROTOCOL,
        "dataset": protocol.DATASET,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "preflight_dataset_invalid",
        "failure_phase": failure_phase,
        "reason_code": reason_code,
        "stage0_registry_sha256": stage0_sha256,
        "candidate_count": protocol.CANDIDATE_COUNT,
        "eligible_count": None,
        "cell_eligible_counts": None,
        "selector_output_created": False,
        "unit_manifest_created": False,
        "stage1_registry_created": False,
        "sealed_final36_status": "unopened",
        "bound_artifacts": {
            "stage0_registry": stage0_sha256,
            "screening_generation_manifest": generation_manifest_sha256,
            "screening_package_commitment": None,
            "screening_freeze_manifest": None,
            "canonical_eligibility": None,
            "selector_stderr": None,
        },
    }
    protocol.validate_invalid_outcome(
        payload, expected_stage0_sha256=stage0_sha256
    )
    _write_json_exclusive(
        project_root / protocol.INVALID_OUTCOME, payload, 0o644
    )


def _write_terminal_status(
    *,
    generation_dir: Path,
    context: Stage0Context,
    status: str,
    failure_phase: str | None,
    reason_code: str | None,
    generation_manifest_sha256: str | None,
    package_commitment_sha256: str | None,
) -> None:
    payload = {
        "protocol": STATUS_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": status,
        "stage0_registry_sha256": context.stage0_sha256,
        "failure_phase": failure_phase,
        "reason_code": reason_code,
        "generation_manifest_sha256": generation_manifest_sha256,
        "package_commitment_sha256": package_commitment_sha256,
    }
    basename = (
        "execution_succeeded_v3.json"
        if status == "succeeded"
        else "execution_failed_v3.json"
    )
    _write_json_exclusive(generation_dir / basename, payload, 0o600)


def _normalize_private_tree_modes(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        return
    root.chmod(0o700)
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError("screening output contains a symlink")
        if entry.is_dir():
            entry.chmod(0o700)
        elif entry.is_file():
            protocol.require(
                entry.stat().st_nlink == 1,
                "screening output contains a hardlink",
            )
            entry.chmod(0o600)
        else:
            raise ValueError("screening output contains a non-regular entry")


def _validate_success_inventory(
    *,
    context: Stage0Context,
    generation_dir: Path,
    public_dir: Path,
    private_dir: Path,
    expected_commitment: Mapping[str, Any],
    expected_commitment_sha256: str,
) -> None:
    expected_generation = {
        ".run_reservation_v3.json",
        "execution_started_v3.json",
        "generator_output_v3.log",
        "prompts.txt",
        GENERIC_MANIFEST_BASENAME,
        RAW_INVENTORY_BASENAME,
        GENERATION_MANIFEST_BASENAME,
        "videos",
    }
    protocol.require(
        {entry.name for entry in generation_dir.iterdir()} == expected_generation,
        "successful screening generation inventory is not exact",
    )
    protocol.require(
        {entry.name for entry in public_dir.iterdir()}
        == {
            "media",
            "composites",
            REVIEW_TEMPLATE_BASENAME,
            ANONYMOUS_INVENTORY_BASENAME,
            COMPOSITE_INVENTORY_BASENAME,
            PUBLIC_MANIFEST_BASENAME,
        }
        and {entry.name for entry in private_dir.iterdir()}
        == {
            ANSWER_KEY_BASENAME,
            CANDIDATE_BINDING_BASENAME,
            PRIVATE_MANIFEST_BASENAME,
            PACKAGE_COMMITMENT_BASENAME,
        },
        "successful screening package inventory is not exact",
    )
    for directory in (
        generation_dir,
        generation_dir / "videos",
        public_dir,
        public_dir / "media",
        public_dir / "composites",
        private_dir,
    ):
        protocol.require(
            directory.is_dir()
            and not directory.is_symlink()
            and stat.S_IMODE(directory.stat().st_mode) == 0o700,
            "successful screening directory mode is not 700",
        )
    regular_files = [
        entry
        for directory in (generation_dir, public_dir, private_dir)
        for entry in directory.rglob("*")
        if entry.is_file()
    ]
    protocol.require(
        all(
            not entry.is_symlink()
            and entry.stat().st_nlink == 1
            and stat.S_IMODE(entry.stat().st_mode) == 0o600
            for entry in regular_files
        ),
        "successful screening file mode/link contract failed",
    )
    private_commitment = private_dir / PACKAGE_COMMITMENT_BASENAME
    commitment = json.loads(private_commitment.read_text(encoding="utf-8"))
    validate_package_commitment(commitment)
    protocol.require(
        commitment == dict(expected_commitment)
        and private_commitment.read_bytes() == _json_bytes(expected_commitment)
        and protocol.sha256_file(private_commitment)
        == expected_commitment_sha256,
        "published package commitment differs from in-memory committed bytes",
    )
    raw_path = generation_dir / RAW_INVENTORY_BASENAME
    generation_path = generation_dir / GENERATION_MANIFEST_BASENAME
    candidate_binding_path = private_dir / CANDIDATE_BINDING_BASENAME
    anonymous_inventory_path = public_dir / ANONYMOUS_INVENTORY_BASENAME
    composite_inventory_path = public_dir / COMPOSITE_INVENTORY_BASENAME
    public_manifest_path = public_dir / PUBLIC_MANIFEST_BASENAME
    private_manifest_path = private_dir / PRIVATE_MANIFEST_BASENAME
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    candidate_binding = json.loads(
        candidate_binding_path.read_text(encoding="utf-8")
    )
    anonymous_inventory = json.loads(
        anonymous_inventory_path.read_text(encoding="utf-8")
    )
    composite_inventory = json.loads(
        composite_inventory_path.read_text(encoding="utf-8")
    )
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    private_manifest = json.loads(
        private_manifest_path.read_text(encoding="utf-8")
    )
    validate_raw_video_inventory(raw)
    validate_generation_manifest(generation)
    validate_candidate_binding(candidate_binding)
    validate_anonymous_inventory(anonymous_inventory)
    validate_composite_inventory(composite_inventory)
    _validate_public_package_payload(public_manifest)
    validate_private_package_manifest(private_manifest)
    protocol.require(
        generation["videos"] == raw["videos"]
        and commitment["generation_manifest_sha256"]
        == protocol.sha256_file(generation_path)
        and commitment["raw_video_inventory_sha256"]
        == protocol.sha256_file(raw_path)
        and commitment["candidate_binding_sha256"]
        == protocol.sha256_file(candidate_binding_path)
        and commitment["anonymous_video_inventory_sha256"]
        == protocol.sha256_file(anonymous_inventory_path)
        and commitment["composite_inventory_sha256"]
        == protocol.sha256_file(composite_inventory_path)
        and commitment["public_manifest_sha256"]
        == protocol.sha256_file(public_manifest_path)
        and commitment["private_manifest_sha256"]
        == protocol.sha256_file(private_manifest_path)
        and commitment["review_template_sha256"]
        == protocol.sha256_file(public_dir / REVIEW_TEMPLATE_BASENAME)
        and commitment["answer_key_sha256"]
        == protocol.sha256_file(private_dir / ANSWER_KEY_BASENAME)
        and public_manifest["anonymous_video_inventory_sha256"]
        == commitment["anonymous_video_inventory_sha256"]
        and public_manifest["composite_inventory_sha256"]
        == commitment["composite_inventory_sha256"]
        and private_manifest["candidate_binding_sha256"]
        == commitment["candidate_binding_sha256"]
        and generation["cuda_lock_sha256"]
        == protocol.sha256_file(context.private_root / CUDA_LOCK_BASENAME)
        and generation["run_reservation_sha256"]
        == protocol.sha256_file(generation_dir / ".run_reservation_v3.json")
        and generation["execution_started_sha256"]
        == protocol.sha256_file(generation_dir / "execution_started_v3.json")
        and generation["generator_log_sha256"]
        == protocol.sha256_file(generation_dir / "generator_output_v3.log")
        and generation["prompt_file_sha256"]
        == protocol.sha256_file(generation_dir / "prompts.txt")
        and generation["generic_generation_manifest_sha256"]
        == protocol.sha256_file(generation_dir / GENERIC_MANIFEST_BASENAME)
        and generation["raw_video_inventory_sha256"]
        == protocol.sha256_file(raw_path)
        and generation["stage0_registry_sha256"] == context.stage0_sha256
        and generation["selection_binding_sha256"] == context.binding_sha256
        and generation["model_content_inventory_sha256"]
        == context.model_inventory_sha256
        and generation["runtime_registry_sha256"]
        == context.runtime_registry_sha256
        and generation["code_registry_sha256"] == context.code_registry_sha256
        and generation["generator_sha256"] == context.generator_sha256
        and commitment["stage0_registry_sha256"] == context.stage0_sha256
        and commitment["pending_commitment_sha256"] == context.pending_sha256
        and commitment["selection_binding_sha256"] == context.binding_sha256
        and commitment["candidate_manifest_sha256"]
        == context.opening_records["candidate_manifest_576"]["sha256"]
        and commitment["candidate_graph_sha256"]
        == context.opening_records["candidate_graph_576"]["sha256"]
        and commitment["generation_spec_sha256"]
        == context.opening_records["screening_generation_spec"]["sha256"]
        and commitment["model_content_inventory_sha256"]
        == context.model_inventory_sha256
        and commitment["runtime_registry_sha256"]
        == context.runtime_registry_sha256
        and commitment["code_registry_sha256"] == context.code_registry_sha256
        and commitment["generator_sha256"] == context.generator_sha256
        and generation["generator_dependency_closure_sha256"]
        == context.generator_dependency_closure_sha256
        and generation["media_runtime_packages"]
        == dict(context.media_runtime_packages)
        and private_manifest["generator_dependency_closure_sha256"]
        == context.generator_dependency_closure_sha256
        and private_manifest["media_runtime_packages"]
        == dict(context.media_runtime_packages)
        and commitment["generator_dependency_closure_sha256"]
        == context.generator_dependency_closure_sha256
        and commitment["media_runtime_packages"]
        == dict(context.media_runtime_packages)
        and private_manifest["raw_media"] == commitment["raw_media"]
        and private_manifest["anonymous_media"] == commitment["anonymous_media"]
        and private_manifest["composites"] == commitment["composites"],
        "screening package commitment/manifest binding mismatch",
    )
    raw_by_review = {
        f"s{index:03d}": record for index, record in enumerate(raw["videos"])
    }
    anonymous_by_review = {
        record["review_id"]: record for record in anonymous_inventory["videos"]
    }
    composite_by_review = {
        record["review_id"]: record
        for record in composite_inventory["composites"]
    }
    for index in range(protocol.CANDIDATE_COUNT):
        review_id = f"s{index:03d}"
        raw_video = generation_dir / "videos" / raw_by_review[review_id]["video_name"]
        anonymous_video = public_dir / "media" / f"{review_id}.mp4"
        composite = public_dir / "composites" / f"{review_id}.jpg"
        protocol.require(
            commitment["raw_media"][review_id]
            == {
                "sha256": protocol.sha256_file(raw_video),
                "size_bytes": raw_video.stat().st_size,
            }
            and commitment["anonymous_media"][review_id]
            == {
                "sha256": protocol.sha256_file(anonymous_video),
                "size_bytes": anonymous_video.stat().st_size,
            }
            and commitment["composites"][review_id]
            == {
                "sha256": protocol.sha256_file(composite),
                "size_bytes": composite.stat().st_size,
            }
            and raw_by_review[review_id]["sha256"]
            == commitment["raw_media"][review_id]["sha256"]
            and anonymous_by_review[review_id]["sha256"]
            == commitment["anonymous_media"][review_id]["sha256"]
            and composite_by_review[review_id]["sha256"]
            == commitment["composites"][review_id]["sha256"],
            "published screening media differs from package commitment",
        )
    review_path = public_dir / REVIEW_TEMPLATE_BASENAME
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        review_rows = list(reader)
        protocol.require(
            tuple(reader.fieldnames or ()) == REVIEW_HEADER,
            "published review template header mismatch",
        )
    protocol.require(
        len(review_rows) == protocol.CANDIDATE_COUNT
        and all(
            row["review_id"] == f"s{index:03d}"
            and row["candidate_video_path"] == f"media/s{index:03d}.mp4"
            and row["candidate_video_sha256"]
            == commitment["anonymous_media"][f"s{index:03d}"]["sha256"]
            and row["composite_path"] == f"composites/s{index:03d}.jpg"
            and row["composite_sha256"]
            == commitment["composites"][f"s{index:03d}"]["sha256"]
            and all(row[field] == "" for field in (*REVIEW_FIELDS, "notes"))
            for index, row in enumerate(review_rows)
        ),
        "published review template rows differ from package commitment",
    )
    answer_header = (
        "review_id",
        "candidate_index",
        "case_id",
        "raw_video_sha256",
        "anonymous_video_sha256",
        "composite_sha256",
    )
    with (private_dir / ANSWER_KEY_BASENAME).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        answer_rows = list(reader)
        protocol.require(
            tuple(reader.fieldnames or ()) == answer_header,
            "published answer-key header mismatch",
        )
    protocol.require(
        len(answer_rows) == protocol.CANDIDATE_COUNT
        and len(candidate_binding["rows"]) == protocol.CANDIDATE_COUNT
        and all(
            answer_rows[index]["review_id"] == f"s{index:03d}"
            and answer_rows[index]["candidate_index"] == str(index)
            and answer_rows[index]["case_id"]
            == context.candidate_payload["candidates"][index]["case_id"]
            and answer_rows[index]["raw_video_sha256"]
            == commitment["raw_media"][f"s{index:03d}"]["sha256"]
            and answer_rows[index]["anonymous_video_sha256"]
            == commitment["anonymous_media"][f"s{index:03d}"]["sha256"]
            and answer_rows[index]["composite_sha256"]
            == commitment["composites"][f"s{index:03d}"]["sha256"]
            and candidate_binding["rows"][index]["review_id"]
            == f"s{index:03d}"
            and candidate_binding["rows"][index]["candidate"]
            == context.candidate_payload["candidates"][index]
            for index in range(protocol.CANDIDATE_COUNT)
        ),
        "published answer-key/candidate binding differs from frozen candidates",
    )
    lock_path = context.private_root / CUDA_LOCK_BASENAME
    protocol.validate_private_path(context.private_root, lock_path)
    lock = protocol.load_json(lock_path, private_root=context.private_root)
    protocol.require(
        lock
        == {
            "protocol": LOCK_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "consumed_for_one_shot_screening",
            "stage0_registry_sha256": context.stage0_sha256,
            "worker_count": 1,
        },
        "CUDA consumption lock content mismatch",
    )


def _minimal_stage0_boundary(
    project_root: Path, private_root: Path
) -> tuple[Path, Path, str]:
    project_root = protocol.validate_project_root(project_root)
    private_root = _require_private_root(project_root, private_root)
    generation_dir, public_dir, private_dir, lock_path = _standard_outputs(
        private_root
    )
    invalid_path = project_root / protocol.INVALID_OUTCOME
    for target in (
        generation_dir,
        public_dir,
        private_dir,
        lock_path,
        invalid_path,
        project_root / protocol.STAGE1_REGISTRY,
    ):
        if os.path.lexists(target):
            raise FileExistsError(
                f"v3 screening is already consumed or terminal: {target}"
            )
    stage0_path = project_root / protocol.STAGE0_REGISTRY
    protocol.validate_runtime_read_path(
        project_root, stage0_path, allow_v2=False
    )
    return project_root, private_root, protocol.sha256_file(stage0_path)


def _run_screening_locked(
    *,
    project_root: Path,
    private_root: Path,
    python_executable: str,
    worker_count: int,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    project_root, private_root, boundary_stage0_sha256 = _minimal_stage0_boundary(
        project_root, private_root
    )
    try:
        context = validate_stage0_for_screening(
            project_root=project_root,
            private_root=private_root,
            require_initial_inventory=True,
        )
        protocol.require(
            context.stage0_sha256 == boundary_stage0_sha256,
            "Stage-0 bytes changed during initial screening validation",
        )
        expected_python = "models/.wan-runtime/bin/python"
        protocol.require(
            python_executable == expected_python,
            "screening interpreter must be the exact frozen runtime executable",
        )
        protocol.require(
            worker_count == 1
            and context.generation_spec["generation"]["worker_count"] == 1,
            "screening worker count differs from the frozen single-worker spec",
        )
        before_reservation = validate_stage0_for_screening(
            project_root=context.project_root,
            private_root=context.private_root,
            require_initial_inventory=True,
        )
        _require_same_context(context, before_reservation)
    except BaseException as exc:
        generation_dir, public_dir, private_dir, lock_path = _standard_outputs(
            private_root
        )
        if any(
            os.path.lexists(path)
            for path in (
                generation_dir,
                public_dir,
                private_dir,
                lock_path,
                project_root / protocol.INVALID_OUTCOME,
                project_root / protocol.STAGE1_REGISTRY,
            )
        ):
            raise FileExistsError(
                "screening preflight lost ownership to another invocation"
            ) from exc
        try:
            _publish_invalid_outcome(
                project_root=project_root,
                stage0_sha256=boundary_stage0_sha256,
                failure_phase="original_generation",
                reason_code=INVALID_REASON_GENERATION,
                generation_manifest_sha256=None,
            )
        except BaseException as publication_error:
            raise TerminalScreeningFailure(
                "terminal_outcome_publication_failure"
            ) from publication_error
        raise TerminalScreeningFailure(
            "screening_preflight_integrity_failure"
        ) from exc
    generation_dir, public_dir, private_dir, lock_path = _standard_outputs(
        context.private_root
    )
    try:
        generation_dir, public_dir, private_dir, lock_path = _reserve_execution(
            context, worker_count
        )
    except ConsumedReservationFailure as exc:
        invalid_error: BaseException | None = None
        try:
            _publish_invalid_outcome(
                project_root=context.project_root,
                stage0_sha256=context.stage0_sha256,
                failure_phase="original_generation",
                reason_code=INVALID_REASON_GENERATION,
                generation_manifest_sha256=None,
            )
        except BaseException as publication_error:
            invalid_error = publication_error
        status_error: BaseException | None = None
        try:
            if generation_dir.is_dir() and not generation_dir.is_symlink():
                _normalize_private_tree_modes(generation_dir)
                _write_terminal_status(
                    generation_dir=generation_dir,
                    context=context,
                    status="failed_terminal",
                    failure_phase="original_generation",
                    reason_code=INVALID_REASON_GENERATION,
                    generation_manifest_sha256=None,
                    package_commitment_sha256=None,
                )
        except BaseException as private_status_error:
            status_error = private_status_error
        if invalid_error is not None:
            raise TerminalScreeningFailure(
                "terminal_outcome_publication_failure"
            ) from invalid_error
        if status_error is not None:
            raise TerminalScreeningFailure(
                "terminal_private_status_failure"
            ) from status_error
        raise TerminalScreeningFailure(
            "original_generation_reservation_failure"
        ) from exc
    generation_manifest_path: Path | None = None
    generation_manifest_sha256: str | None = None
    commitment: dict[str, Any] | None = None
    failure_phase = "original_generation"
    reason_code = INVALID_REASON_GENERATION
    try:
        log_path = generation_dir / "generator_output_v3.log"
        log_descriptor = os.open(
            log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(log_descriptor, "wb") as log_handle:
            prompt_path = _write_prompt_file(
                generation_dir, context.candidate_payload["candidates"]
            )
            command = generation_command(
                python_executable=python_executable,
                generator_relative=protocol.CODE_ARTIFACT_PATHS["generator"],
                prompt_path=prompt_path,
                generation_dir=generation_dir,
                screening_seed=context.screening_seed,
            )
            completed = run(
                command,
                cwd=context.project_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                check=False,
                timeout=MAX_SCREENING_GENERATION_SECONDS,
                env=sanitized_worker_environment(),
            )
        protocol.require(
            getattr(completed, "returncode", None) == 0,
            "generic generator process failed",
        )
        generic_manifest, videos = _validate_generic_manifest(
            context=context,
            generation_dir=generation_dir,
            prompt_path=prompt_path,
        )
        del generic_manifest
        (generation_dir / "videos").chmod(0o700)
        (generation_dir / GENERIC_MANIFEST_BASENAME).chmod(0o600)
        for video in videos:
            video.chmod(0o600)
        generation_manifest, generation_manifest_path, raw_inventory_path = (
            _write_raw_manifests(
                context=context,
                generation_dir=generation_dir,
                prompt_path=prompt_path,
                videos=videos,
            )
        )
        generation_manifest_sha256 = protocol.sha256_file(
            generation_manifest_path
        )

        post_generation = validate_stage0_for_screening(
            project_root=context.project_root,
            private_root=context.private_root,
            require_initial_inventory=False,
        )
        _require_same_context(context, post_generation)
        failure_phase = "screening_package"
        reason_code = INVALID_REASON_PACKAGE

        def revalidate_before_publish() -> None:
            observed = validate_stage0_for_screening(
                project_root=context.project_root,
                private_root=context.private_root,
                require_initial_inventory=False,
            )
            _require_same_context(context, observed)

        commitment = build_screening_package(
            context=context,
            generation_manifest=generation_manifest,
            generation_manifest_path=generation_manifest_path,
            raw_inventory_path=raw_inventory_path,
            videos=videos,
            public_dir=public_dir,
            private_dir=private_dir,
            revalidate=revalidate_before_publish,
        )
        commitment_sha256 = protocol.sha256_bytes(_json_bytes(commitment))
        assert generation_manifest_sha256 is not None
        _validate_success_inventory(
            context=context,
            generation_dir=generation_dir,
            public_dir=public_dir,
            private_dir=private_dir,
            expected_commitment=commitment,
            expected_commitment_sha256=commitment_sha256,
        )
        _require_no_public_terminal_or_stage1(context.project_root)
        _write_terminal_status(
            generation_dir=generation_dir,
            context=context,
            status="succeeded",
            failure_phase=None,
            reason_code=None,
            generation_manifest_sha256=generation_manifest_sha256,
            package_commitment_sha256=commitment_sha256,
        )
        return {
            "status": "succeeded",
            "candidate_count": protocol.CANDIDATE_COUNT,
            "stage0_registry_sha256": context.stage0_sha256,
            "generation_manifest_sha256": generation_manifest_sha256,
            "package_commitment_sha256": commitment_sha256,
        }
    except BaseException as exc:
        try:
            _normalize_private_tree_modes(generation_dir)
        except Exception:
            pass
        invalid_error: BaseException | None = None
        try:
            _publish_invalid_outcome(
                project_root=context.project_root,
                stage0_sha256=context.stage0_sha256,
                failure_phase=failure_phase,
                reason_code=reason_code,
                generation_manifest_sha256=(
                    generation_manifest_sha256
                    if failure_phase == "screening_package"
                    else None
                ),
            )
        except BaseException as publication_error:
            invalid_error = publication_error
        status_error: BaseException | None = None
        try:
            _write_terminal_status(
                generation_dir=generation_dir,
                context=context,
                status="failed_terminal",
                failure_phase=failure_phase,
                reason_code=reason_code,
                generation_manifest_sha256=generation_manifest_sha256,
                package_commitment_sha256=None,
            )
        except BaseException as private_status_error:
            status_error = private_status_error
        if invalid_error is not None:
            raise TerminalScreeningFailure(
                "terminal_outcome_publication_failure"
            ) from invalid_error
        if status_error is not None:
            raise TerminalScreeningFailure(
                "terminal_private_status_failure"
            ) from status_error
        raise TerminalScreeningFailure(
            "original_generation_failure"
            if failure_phase == "original_generation"
            else "screening_package_failure"
        ) from exc


def run_screening(
    *,
    project_root: Path,
    private_root: Path,
    python_executable: str,
    worker_count: int,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    validated_project = protocol.validate_project_root(project_root)
    validated_private = _require_private_root(validated_project, private_root)
    with _screening_mutex(validated_private):
        return _run_screening_locked(
            project_root=validated_project,
            private_root=validated_private,
            python_executable=python_executable,
            worker_count=worker_count,
            run=run,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    parser.add_argument(
        "--python",
        required=True,
        help="Must be exactly models/.wan-runtime/bin/python",
    )
    parser.add_argument("--worker-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_screening(
            project_root=args.project_root,
            private_root=args.private_root,
            python_executable=args.python,
            worker_count=args.worker_count,
        )
    except TerminalScreeningFailure as exc:
        print(f"terminal screening failure: {exc.category}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"screening preflight failed closed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2
    print(protocol.canonical_json_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
