#!/usr/bin/env python3
"""Fail-closed Stage-0 authorizer for the preregistered v4_dev72_v3 graph.

This program opens and recomputes every Stage-0 input, then writes the private
selection binding followed by the public standard wrapper.  It never renders
media or invokes an evaluation runner; its only accelerator action is the
read-only child-runtime CUDA identity probe required by the frozen contract.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import os
import secrets as pysecrets
import stat
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import build_water_impact_dynamic_v4_causal_candidates_v3 as builder
    import select_water_impact_dynamic_v4_causal_v3 as selector
    import validate_water_impact_dynamic_v4_causal_capacity_v3 as capacity
except ModuleNotFoundError:  # imported as scripts.authorize_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import build_water_impact_dynamic_v4_causal_candidates_v3 as builder
    from scripts import select_water_impact_dynamic_v4_causal_v3 as selector
    from scripts import validate_water_impact_dynamic_v4_causal_capacity_v3 as capacity


PENDING_PROTOCOL = "water_impact_dynamic_v4_causal_stage0_public_commitment_v3"
PENDING_SCHEMA = "water_impact_dynamic_v4_source_slot_registry_v3"
PENDING_REGISTRY = "causal_stage0_public_commitment_v3"
SELECTION_BINDING_PROTOCOL = "water_impact_dynamic_v4_selection_binding_v3"
MODEL_INVENTORY_PROTOCOL = "water_impact_dynamic_v4_model_content_inventory_v3"
RUNTIME_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_runtime_registry_v3"
COST_PROTOCOL = "water_impact_dynamic_v4_screening_cost_calibration_v3"
COST_RUN_MANIFEST_PROTOCOL = (
    "water_impact_dynamic_v4_screening_cost_calibration_run_manifest_v3"
)
COST_CALIBRATION_RUN_DIRNAME = "v4_screening_cost_calibration_run_v3"
COST_RUN_MANIFEST_BASENAME = "calibration_run_manifest_v3.json"
COST_FAILED_BASENAME = "execution_failed_v3.json"
GENERATION_SPEC_PROTOCOL = "water_impact_dynamic_v4_generation_spec_v3"
SEED_AUDIT_PROTOCOL = "water_impact_dynamic_v4_preselection_seed_audit_v3"
HOLDOUT_PUBLIC_PROTOCOL = "water_impact_dynamic_v4_holdout_public_commitment_v3"
HOLDOUT_FROZEN_RECORDS: dict[str, dict[str, Any]] = {
    "eval_holdout_source_ontology_48": {
        "sha256": "e9cfceed714c52ea14a834c5bd6070da798885c3f2b7048bd1f1207ea30c46a4",
        "size_bytes": 80175,
        "row_count": 48,
    },
    "holdout_registry_48": {
        "sha256": "423a3ae68d27ecf03cb3a72375fca5e282de4c3b45efe9f3a3bb9b6059bc70a0",
        "size_bytes": 79649,
        "row_count": 48,
    },
    "receiver_ontology_56": {
        "sha256": "7336adfd55aafd2dc5024092a8f0c20bad3083238b4d7af871def6399db9051b",
        "size_bytes": 38088,
        "row_count": 56,
    },
    "historical_receiver_anchors_8": {
        "sha256": "06e28e0eb35d85a5bc1d36de4856e30d477a3efd91ce89f3c242e9fa66cff202",
        "size_bytes": 3420,
        "row_count": 8,
    },
    "curation_manifest": {
        "sha256": "c3da1aaa3784e033e0d106a5123d05e228b21057f297066c8b4f936b3f96b16f",
        "size_bytes": 3143,
        "row_count": None,
    },
    "curation_public_aggregate": {
        "sha256": "98d242c13fdc4be93a92cf0d9d97eba90b0170869529731f812abc446bad6e46",
        "size_bytes": 3617,
        "row_count": None,
    },
    "curation_semantic_audit": {
        "sha256": "a6f498be9fe6a85f4fa94b350570b503b7e64d5b086d1fad9a9ff012f30ae2dd",
        "size_bytes": 11842,
        "row_count": None,
    },
    "curation_validator": {
        "sha256": "eb415329092e39cfbc56c6d33637b098a8b109e28ff26eb6a211dd8109fdfbc3",
        "size_bytes": 21641,
        "row_count": None,
    },
    "curation_tests": {
        "sha256": "cf0870aa6870e34da0683b6bdf95e8ae9b1e5a7094f3e7abe25a54fb262e2f13",
        "size_bytes": 3925,
        "row_count": None,
    },
}
HOLDOUT_EVIDENCE_BASENAMES = {
    "eval_holdout_source_ontology_48": (
        "eval_holdout_source_ontology_private48_v3.json"
    ),
    "holdout_registry_48": "holdout_registry_private48_v3.json",
    "receiver_ontology_56": "receiver_ontology_private56_v3.json",
    "historical_receiver_anchors_8": (
        "historical_receiver_anchors_private8_v3.json"
    ),
    "curation_manifest": "curation_private_manifest_v3.json",
    "curation_public_aggregate": "curation_public_aggregate_staging_v3.json",
    "curation_semantic_audit": (
        "public_allowlist_semantic_audit_private_v3.json"
    ),
    "curation_validator": "validate_clean_room_v3_r6.py",
    "curation_tests": "test_clean_room_curation_v3_r6.py",
}
CURATION_PUBLIC_AGGREGATE_PROTOCOL = (
    "water_impact_dynamic_v4_curation_public_aggregate_v3"
)
V2_FORBIDDEN_SEED_INVENTORY_SHA256 = (
    "f2f72728a83c7e3ec54735a58f3f2e0a5afd1c132822eeecad7dc2006cb5ecd4"
)
SECRETS_PROTOCOL = "water_impact_dynamic_v4_causal_stage0_secrets_v3"
HISTORICAL_SECRET_AUDIT_PROTOCOL = (
    "water_impact_dynamic_v4_v3_historical_secret_disjointness_audit_v1"
)
HISTORICAL_SECRET_AUDIT_BASENAME = "historical_secret_audit_private_v3.json"
SECRET_SAMPLING_REQUEST_PROTOCOL = (
    "water_impact_dynamic_v4_v3_secret_sampling_request_v1"
)
SECRET_SAMPLING_AGGREGATE_PROTOCOL = (
    "water_impact_dynamic_v4_v3_secret_sampling_public_aggregate_v1"
)
SECRET_SAMPLING_REQUEST_BASENAME = "secret_sampling_request_private_v3.json"
HISTORICAL_ACCESSIBLE_RAW_COUNT = 6
HISTORICAL_ACCESSIBLE_RAW_ALLOWLIST_SHA256 = (
    "1eb6e1d850fa2698c4f8f8a9213bbd4f712a4cc9c2683b6f94a87520f670c222"
)
HISTORICAL_COMMITMENT_ONLY_COUNT = 4
HISTORICAL_COMMITMENT_ALLOWLIST_SHA256 = (
    "c1eace9cafb26723c49d3476581111369a9930b83e10ed6fe2537f17ab671400"
)
HISTORICAL_RAW_SOURCE_FILES = {
    "clean": {
        "salts_private_v2.json": {
            "sha256": "65ee2c78d92134f6af882923101149428d92b743e34a1304f24258a2bcecadbf",
            "size_bytes": 751,
            "raw_hex64_count": 6,
        },
        "causal_stage0_secrets_private_v2.json": {
            "sha256": "9675329f4c133af9d712c9dca34c437f326eedf6e7c35a41fd5a019cd72920ad",
            "size_bytes": 501,
            "raw_hex64_count": 2,
        },
    },
    "authorizer": {
        "causal_stage0_secrets_private_v2.json": {
            "sha256": "9675329f4c133af9d712c9dca34c437f326eedf6e7c35a41fd5a019cd72920ad",
            "size_bytes": 501,
            "raw_hex64_count": 2,
        },
        "causal_stage0_selector_salt_v2.txt": {
            "sha256": "b5d0def1f40f57192774b673ba29ee3e32859b96e26428b66bfe57efade7a325",
            "size_bytes": 65,
            "raw_hex64_count": 1,
        },
        "causal_evaluation_seed_salt_v2.txt": {
            "sha256": "768e04dcc46f482dace4ce72c443a846028f5263e25627a787979375350f2e74",
            "size_bytes": 65,
            "raw_hex64_count": 1,
        },
    },
}
HISTORICAL_COMMITMENT_PUBLIC_SOURCES = {
    "data/water_impact_dynamic_v4/causal_stage0_public_commitment_v2.json": {
        "sha256": "0d7fab1befdc197a7ae7f864a84c1f1ac3d029d5d72f9a513303892e48ec2477",
        "fields": {
            "evaluation_seed_salt_commitment_sha256": "6c943c09a6429df0cc3d1c098e685911110f7510abd8a53e25f307881b9e0dfc",
            "selector_salt_commitment_sha256": "a944183ed47f0024b65f49bb1af172964be934f0aa9151034d7f11fec54ea893",
        },
    },
    "data/water_impact_dynamic_v4/source_mapping_v2.json": {
        "sha256": "6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2",
        "fields": {
            "source_assignment_salt_sha256": "9b0534a57ea63dbc5747a1fc628e7b840e63cfbc1858d7cf6d158aab76ab547a",
        },
    },
    "data/water_impact_dynamic_v4/holdout_public_commitment_v2.json": {
        "sha256": "6751a4d3b66491328909853b99bc8e6d06468a30b71f5bb746c7a744692fe84d",
        "fields": {
            "split_salt_commitment_sha256": "d22b37311f8360a8ba4ae17efb9055097fd7413a867435418f3108358b33fd5a",
        },
    },
}
RULES_PROTOCOL = "water_impact_dynamic_v4_causal_selection_rules_v3"
RENDER_PROTOCOL = "water_impact_dynamic_v4_causal_render_configuration_v3"
BUNDLE_PROTOCOL = "water_impact_dynamic_v4_causal_stage0_bundle_v3"
RUNTIME_PACKAGE_NAMES = {
    "accelerate",
    "diffusers",
    "huggingface-hub",
    "numpy",
    "peft",
    "protobuf",
    "safetensors",
    "sentencepiece",
    "tokenizers",
    "torch",
    "transformers",
}
EXPECTED_RUNTIME_PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "diffusers": "0.33.1",
    "huggingface-hub": "0.36.2",
    "numpy": "2.4.6",
    "peft": "0.15.2",
    "protobuf": "7.35.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.21.4",
    "torch": "2.6.0",
    "transformers": "4.51.3",
}
EXPECTED_RUNTIME_PYTHON = {"implementation": "CPython", "version": "3.11.15"}
EXPECTED_RUNTIME_TORCH = {
    "distribution_version": "2.6.0",
    "module_version": "2.6.0+cu124",
}
assert set(EXPECTED_RUNTIME_PACKAGE_VERSIONS) == RUNTIME_PACKAGE_NAMES
GENERATOR_DEPENDENCY_PATHS = (
    "scripts/generate_wan_clean.py",
    "scripts/generate_cogvideox_clean.py",
    "scripts/run_pilot.py",
    "scripts/causal_lora_activation_gate.py",
    "scripts/target_token_attention_suppression.py",
)
MEDIA_RUNTIME_DISTRIBUTIONS = ("av", "Pillow")
RUNTIME_CONTENT_INVENTORY_ALGORITHM = (
    "sha256_ordered_relative_path_nul_raw_bytes_newline_v1"
)
RUNTIME_ORIGIN_MODULES = (
    "torch",
    "diffusers",
    "transformers",
    "peft",
    "safetensors",
    "numpy",
    "av",
    "PIL",
)
CALIBRATION_SEEDS = protocol.CALIBRATION_SEEDS
CALIBRATION_PROMPT_SHA256 = protocol.CALIBRATION_PROMPT_SHA256

EXPECTED_PREREG_SHA256 = (
    "11d1bff22ec1a3938007be86a29824b046c8303aee2304394b4bb767d1923448"
)
PREREG_PATH = Path("docs/water_impact_dynamic_v4_dev72_v3_preregistration.md")
MODEL_INVENTORY_PATH = protocol.DATA_ROOT / "v4_model_content_inventory_v3.json"
RUNTIME_REGISTRY_PATH = protocol.DATA_ROOT / "v4_runtime_registry_v3.json"
COST_CALIBRATION_PATH = protocol.DATA_ROOT / "v4_screening_cost_calibration_v3.json"
CAPACITY_MODEL_PATH = protocol.DATA_ROOT / "v4_causal_capacity_model_v3.json"
CAPACITY_SEARCH_PATH = protocol.DATA_ROOT / "v4_causal_capacity_search_v3.json"
CAPACITY_CONFIRM_PATH = protocol.DATA_ROOT / "v4_causal_capacity_confirm_v3.json"
STATIC_GRAPH_PATH = protocol.DATA_ROOT / "v4_causal_static_graph_audit_v3.json"

PRIVATE_INPUTS = {
    "candidate_manifest_576": "causal_stage0_candidates_private576_v3.json",
    "eval_holdout_source_ontology_48": "eval_holdout_source_ontology_private48_v3.json",
    "holdout_registry_48": "holdout_registry_private48_v3.json",
    "receiver_ontology_56": "receiver_ontology_private56_v3.json",
    "historical_receiver_anchors_8": "historical_receiver_anchors_private8_v3.json",
    "candidate_graph_576": "causal_stage0_candidate_graph_private576_v3.json",
    "canonical_templates": "causal_stage0_templates_private_v3.json",
    "field_normalization": "causal_stage0_field_rules_private_v3.json",
    "raw_root_bundle": "causal_stage0_bundle_private_v3.json",
    "raw_render_configuration": "causal_stage0_render_config_private_v3.json",
    "stage0_secrets": "causal_stage0_secrets_private_v3.json",
    "screening_seed": "causal_screening_seed_v3.txt",
    "graph_assignment_salt": "causal_graph_assignment_salt_v3.txt",
    "screening_generation_spec": "causal_generation_spec_v3.json",
    "selection_rules": "causal_stage0_selection_rules_private_v3.json",
    "selector_salt": "causal_stage0_selector_salt_v3.txt",
    "evaluation_seed_salt": "causal_evaluation_seed_salt_v3.txt",
    "forbidden_seed_inventory": "causal_forbidden_seed_inventory_v3.json",
    "preselection_seed_audit_1728": "causal_preselection_seed_audit_1728_v3.json",
}
PRIVATE_EXTERNAL_INPUT_NAMES = {
    PRIVATE_INPUTS["eval_holdout_source_ontology_48"],
    PRIVATE_INPUTS["holdout_registry_48"],
    PRIVATE_INPUTS["receiver_ontology_56"],
    PRIVATE_INPUTS["historical_receiver_anchors_8"],
    PRIVATE_INPUTS["canonical_templates"],
    PRIVATE_INPUTS["field_normalization"],
    PRIVATE_INPUTS["forbidden_seed_inventory"],
}
PUBLIC_OPENINGS = {
    "holdout_public_commitment": protocol.HOLDOUT_PUBLIC_COMMITMENT,
    "model_content_inventory": MODEL_INVENTORY_PATH,
    "runtime_registry": RUNTIME_REGISTRY_PATH,
    "eval_code_registry": protocol.CODE_REGISTRY,
    "screening_cost_calibration": COST_CALIBRATION_PATH,
    "capacity_model_spec": CAPACITY_MODEL_PATH,
    "capacity_search_result_200000": CAPACITY_SEARCH_PATH,
    "capacity_confirm_result_1000000": CAPACITY_CONFIRM_PATH,
    "static_graph_robustness_report": STATIC_GRAPH_PATH,
    "identity_disjointness_report": protocol.IDENTITY_REPORT,
    "v2_construct_equivalence_report": protocol.CONSTRUCT_REPORT,
    "forbidden_seed_source_audit": protocol.FORBIDDEN_SEED_SOURCE_AUDIT,
}
OPENING_NAMES = tuple(PRIVATE_INPUTS) + tuple(PUBLIC_OPENINGS)
assert len(OPENING_NAMES) == 31

PHYSICAL_ROW_COUNTS: dict[str, int | None | str] = {
    "candidate_manifest_576": 576,
    "eval_holdout_source_ontology_48": 48,
    "holdout_registry_48": 48,
    "receiver_ontology_56": 56,
    "historical_receiver_anchors_8": 8,
    "candidate_graph_576": 576,
    "canonical_templates": None,
    "field_normalization": None,
    "raw_root_bundle": None,
    "raw_render_configuration": None,
    "stage0_secrets": None,
    "screening_seed": None,
    "graph_assignment_salt": None,
    "screening_generation_spec": None,
    "selection_rules": None,
    "selector_salt": None,
    "evaluation_seed_salt": None,
    "forbidden_seed_inventory": "positive",
    "preselection_seed_audit_1728": 1728,
    "holdout_public_commitment": None,
    "model_content_inventory": None,
    "runtime_registry": None,
    "eval_code_registry": None,
    "screening_cost_calibration": 5,
    "capacity_model_spec": None,
    "capacity_search_result_200000": None,
    "capacity_confirm_result_1000000": None,
    "static_graph_robustness_report": None,
    "identity_disjointness_report": None,
    "v2_construct_equivalence_report": None,
    "forbidden_seed_source_audit": None,
}

RANK_FORMULA = (
    "sha256(utf8(\"causal-selector-v3\") || NUL || utf8(selector_salt) || "
    "NUL || canonical_candidate_record_bytes)"
)
SEED_FORMULA = (
    "uint32(first_4_bytes(sha256(utf8(\"causal-eval-seed-v3\") || NUL || "
    "utf8(evaluation_salt) || NUL || utf8(case_id) || NUL || "
    "utf8(decimal_replicate))), big_endian)"
)
GRAPH_FORMULA = (
    "sha256(utf8(\"causal-graph-receiver-permutation-v3\") || NUL || "
    "utf8(graph_assignment_salt) || NUL || utf8(pool) || NUL || "
    "utf8(receiver_id))"
)
QUALIFICATION = {
    "source_visibility": 2,
    "footprint_visibility_min": 1,
    "receiver_min": 1,
    "quality_min": 1,
    "causal_link": 2,
}


def _load_public_json(project_root: Path, relative: Path) -> dict[str, Any]:
    path = project_root / relative
    protocol.validate_runtime_read_path(project_root, path, allow_v2=False)
    return protocol.load_json(path, project_root=project_root, allow_v2=False)


def _load_private_json(private_root: Path, name: str) -> dict[str, Any]:
    return protocol.load_json(private_root / name, private_root=private_root)


def _file_record(path: Path, row_count: int | None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink artifact: {path}")
    info = path.stat()
    protocol.require(info.st_nlink == 1, f"artifact is hardlinked: {path}")
    return {
        "sha256": protocol.sha256_file(path),
        "size_bytes": info.st_size,
        "row_count": row_count,
    }


def _physical_record(path: Path, rule: int | None | str) -> dict[str, Any]:
    if rule == "positive":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("seeds")
        protocol.require(isinstance(rows, list) and rows, "forbidden seed rows missing")
        row_count: int | None = len(rows)
    else:
        row_count = rule
    return _file_record(path, row_count)


def _read_text_secret(path: Path, *, integer: bool = False) -> str | int:
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("secret opening is not ASCII") from exc
    protocol.require(text.endswith("\n") and text.count("\n") == 1, "secret opening must have one trailing LF")
    value = text[:-1]
    if integer:
        protocol.require(value == str(int(value)) and 0 <= int(value) < 2**32, "screening seed opening invalid")
        return int(value)
    return protocol.validate_lower_hex_salt(value, "secret opening")


def _secret_commitment(name: str, value: str | int) -> str:
    return protocol.sha256_bytes(
        name.encode("utf-8") + b"\x00" + str(value).encode("utf-8")
    )


def _json_normalized(value: Any) -> Any:
    return json.loads(protocol.canonical_json_bytes(value))


def generator_dependency_closure(
    project_root: Path,
) -> tuple[dict[str, str], str]:
    project_root = protocol.validate_project_root(project_root)
    paths = [project_root / relative for relative in GENERATOR_DEPENDENCY_PATHS]
    records: dict[str, str] = {}
    for relative, path in zip(GENERATOR_DEPENDENCY_PATHS, paths):
        protocol._require_no_symlink_components(path)
        protocol._require_regular_file(path, single_link=True)
        records[relative] = protocol.sha256_file(path)
    protocol.validate_no_v2_imports(paths, project_root)
    allowed_paths = {path.resolve(strict=True) for path in paths}
    observed_local_dependencies: set[Path] = set()
    scripts_root = (project_root / "scripts").resolve(strict=True)
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_tops: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_tops.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_tops.add(node.module.split(".", 1)[0])
        for top in imported_tops:
            candidates = [
                scripts_root / f"{top}.py",
                scripts_root / f"{top}.pyc",
                scripts_root / top,
                scripts_root / f"{top}.so",
                scripts_root / f"{top}.pyd",
                *scripts_root.glob(f"{top}.*.so"),
                *scripts_root.glob(f"{top}.*.pyd"),
                *(scripts_root / "__pycache__").glob(f"{top}*.pyc"),
            ]
            for candidate in candidates:
                if not os.path.lexists(candidate):
                    continue
                protocol.require(
                    not candidate.is_symlink(),
                    f"generator import shadow is symlinked: {candidate}",
                )
                resolved = candidate.resolve(strict=True)
                protocol.require(
                    resolved in allowed_paths and resolved.is_file(),
                    f"unbound generator import shadow exists: {candidate}",
                )
                observed_local_dependencies.add(resolved)
    for top in ("torch", "diffusers", "av", "PIL"):
        shadow_candidates = [
            scripts_root / f"{top}.py",
            scripts_root / f"{top}.pyc",
            scripts_root / top,
            scripts_root / f"{top}.so",
            scripts_root / f"{top}.pyd",
            *scripts_root.glob(f"{top}.*.so"),
            *scripts_root.glob(f"{top}.*.pyd"),
            *(scripts_root / "__pycache__").glob(f"{top}*.pyc"),
        ]
        protocol.require(
            not any(os.path.lexists(candidate) for candidate in shadow_candidates),
            f"unbound media runtime import shadow exists: scripts/{top}",
        )
    protocol.require(
        observed_local_dependencies
        == {path.resolve(strict=True) for path in paths[1:]},
        "generator local dependency closure is not the exact fixed five paths",
    )
    digest = protocol.sha256_bytes(protocol.canonical_json_bytes(records))
    return records, digest


def probe_media_runtime_packages(project_root: Path) -> dict[str, str]:
    project_root = protocol.validate_project_root(project_root)
    executable = project_root / "models/.wan-runtime/bin/python"
    runtime_root = project_root / "models/.wan-runtime"
    protocol._require_no_symlink_components(runtime_root)
    protocol._require_no_symlink_components(executable)
    protocol._require_regular_file(executable, single_link=True)
    source = """
import importlib.metadata, json, os, sys
names = json.loads(sys.argv[1])
print(json.dumps({
    "executable_realpath": os.path.realpath(sys.executable),
    "prefix_realpath": os.path.realpath(sys.prefix),
    "packages": {name: importlib.metadata.version(name) for name in names},
}, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            os.fspath(executable),
            "-I",
            "-c",
            source,
            json.dumps(MEDIA_RUNTIME_DISTRIBUTIONS, separators=(",", ":")),
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
    )
    protocol.require(completed.returncode == 0, "media runtime child probe failed")
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("media runtime child probe output invalid") from exc
    protocol.require_exact_keys(
        observed,
        {"executable_realpath", "prefix_realpath", "packages"},
        "media runtime child probe",
    )
    packages = observed["packages"]
    protocol.require(
        observed["executable_realpath"] == os.path.realpath(executable)
        and observed["prefix_realpath"] == os.path.realpath(runtime_root)
        and isinstance(packages, dict)
        and set(packages) == set(MEDIA_RUNTIME_DISTRIBUTIONS)
        and all(isinstance(value, str) and value for value in packages.values()),
        "media runtime child probe mismatch",
    )
    return {name: packages[name] for name in MEDIA_RUNTIME_DISTRIBUTIONS}


def build_code_registry_payload(project_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    artifacts: dict[str, Any] = {}
    paths: list[Path] = []
    for name, relative in protocol.CODE_ARTIFACT_PATHS.items():
        path = project_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"required v3 code artifact is missing: {name}"
            )
        protocol._require_no_symlink_components(path)
        protocol._require_regular_file(path, single_link=True)
        paths.append(path)
        artifacts[name] = {"path": relative, "sha256": protocol.sha256_file(path)}
    runtime_names = {
        "protocol",
        "candidate_builder",
        "stage0_authorizer",
        "screening_runner",
        "screening_freezer",
        "selector",
        "validator",
        "capacity_validator",
    }
    protocol.validate_no_v2_imports(
        [project_root / protocol.CODE_ARTIFACT_PATHS[name] for name in runtime_names],
        project_root,
    )
    generator_dependency_closure(project_root)
    return {
        "protocol": protocol.CODE_REGISTRY_PROTOCOL,
        "status": "frozen",
        "dataset_version": protocol.DATASET_VERSION,
        "v2_read_allowlist": dict(protocol.V2_RUNTIME_READ_ALLOWLIST),
        "artifacts": artifacts,
    }


def validate_code_registry_full(
    payload: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    expected = build_code_registry_payload(project_root)
    protocol.require(payload == expected, "code registry differs from current exact 13-file inventory")
    protocol.validate_code_registry(payload, project_root)
    return expected


def _expected_sizing_rule(
    components: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "protocol": "water_impact_dynamic_v4_capacity_binding_v3",
        "candidate_count": 576,
        "cell_counts": {
            f"{group}:{variant}": protocol.CELL_COUNTS[(group, variant)]
            for group, variant in protocol.CELL_ORDER
        },
        "m0_familywise_shortage_probability": 0.0366161849,
        "m0_ceiling": 0.05,
        "m1_familywise_shortage_probability": 0.0971766186,
        "m1_ceiling": 0.15,
        "m2_rho010": {
            "iterations": 1_000_000,
            "failures": 143_547,
            "failure_rate": 0.143547,
            "wilson_upper_one_sided_95": 0.1441246991,
            "ceiling": 0.15,
            "passes": True,
        },
        "capacity_model_sha256": components["capacity_model_spec"]["sha256"],
        "capacity_search_sha256": components[
            "capacity_search_result_200000"
        ]["sha256"],
        "capacity_confirmation_sha256": components[
            "capacity_confirm_result_1000000"
        ]["sha256"],
        "static_graph_sha256": components["static_graph_robustness_report"][
            "sha256"
        ],
    }


def _expected_curation_audit(
    components: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "protocol": "water_impact_dynamic_v4_curation_binding_v3",
        "source_count": 48,
        "receiver_count": 56,
        "historical_anchor_count": 8,
        "strict_source_physical_pass": 48,
        "unique_source_ids": 48,
        "unique_source_heads": 48,
        "unique_receiver_ids": 56,
        "unique_receiver_heads": 56,
        "unique_historical_receiver_ids": 8,
        "source_ontology_sha256": components[
            "eval_holdout_source_ontology_48"
        ]["sha256"],
        "receiver_ontology_sha256": components["receiver_ontology_56"][
            "sha256"
        ],
        "historical_anchors_sha256": components[
            "historical_receiver_anchors_8"
        ]["sha256"],
        "holdout_public_commitment_sha256": components[
            "holdout_public_commitment"
        ]["sha256"],
        "identity_report_sha256": components["identity_disjointness_report"][
            "sha256"
        ],
        "construct_report_sha256": components[
            "v2_construct_equivalence_report"
        ]["sha256"],
        "forbidden_seed_source_audit_sha256": components[
            "forbidden_seed_source_audit"
        ]["sha256"],
    }


def _expected_public_metadata(
    components: Mapping[str, Mapping[str, Any]],
    secret_commitments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    secrets = (
        dict(secret_commitments)
        if secret_commitments is not None
        else {
            "screening_seed": components["screening_seed"]["sha256"],
            "graph_assignment_salt": components["graph_assignment_salt"][
                "sha256"
            ],
            "selector_salt": components["selector_salt"]["sha256"],
            "evaluation_seed_salt": components["evaluation_seed_salt"][
                "sha256"
            ],
        }
    )
    return {
        "sealed_final36_status": "unopened",
        "source_bank_registry_sha256": protocol.V2_RUNTIME_READ_ALLOWLIST[
            protocol.V2_BANK.as_posix()
        ],
        "source_mapping_sha256": protocol.V2_RUNTIME_READ_ALLOWLIST[
            protocol.V2_MAPPING.as_posix()
        ],
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "v2_termination_sha256": protocol.V2_RUNTIME_READ_ALLOWLIST[
            protocol.V2_TERMINATION.as_posix()
        ],
        "candidate_manifest_sha256": components["candidate_manifest_576"][
            "sha256"
        ],
        "candidate_graph_sha256": components["candidate_graph_576"]["sha256"],
        "model_inventory_sha256": components["model_content_inventory"][
            "sha256"
        ],
        "runtime_registry_sha256": components["runtime_registry"]["sha256"],
        "code_registry_sha256": components["eval_code_registry"]["sha256"],
        "cost_calibration_sha256": components["screening_cost_calibration"][
            "sha256"
        ],
        "forbidden_seed_source_audit_sha256": components[
            "forbidden_seed_source_audit"
        ]["sha256"],
        "secret_commitments": secrets,
    }


def validate_pending(
    payload: Mapping[str, Any], *, project_root: Path, pending_path: Path
) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "schema",
            "registry",
            "dataset_version",
            "stage",
            "status",
            "authorization_status",
            "candidate_count",
            "cell_counts",
            "sizing_rule",
            "design_input",
            "curation_audit",
            "public_metadata",
            "component_commitments",
            "remaining_blockers",
        },
        "pending Stage-0 commitment",
    )
    protocol.require(
        payload["protocol"] == PENDING_PROTOCOL
        and payload["schema"] == PENDING_SCHEMA
        and payload["registry"] == PENDING_REGISTRY
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and type(payload["stage"]) is int
        and payload["stage"] == 0,
        "pending Stage-0 identity mismatch",
    )
    protocol.require(
        payload["status"] == "frozen_components_pending_authorization"
        and payload["authorization_status"] == "not_authorized"
        and payload["remaining_blockers"] == [],
        "pending Stage-0 is not ready for authorization",
    )
    protocol.require(payload["candidate_count"] == 576, "pending candidate count mismatch")
    protocol.require(
        payload["cell_counts"]
        == {f"{g}:{v}": protocol.CELL_COUNTS[(g, v)] for g, v in protocol.CELL_ORDER},
        "pending cell counts mismatch",
    )
    design = protocol.require_exact_keys(
        payload["design_input"], {"preregistration", "v2_termination"}, "pending design input"
    )
    expected_design = {
        "preregistration": {
            "path": PREREG_PATH.as_posix(),
            "sha256": EXPECTED_PREREG_SHA256,
        },
        "v2_termination": {
            "path": protocol.V2_TERMINATION.as_posix(),
            "sha256": protocol.V2_RUNTIME_READ_ALLOWLIST[
                protocol.V2_TERMINATION.as_posix()
            ],
        },
    }
    protocol.require(design == expected_design, "pending design input mismatch")
    protocol._require_no_symlink_components(project_root / PREREG_PATH)
    protocol._require_regular_file(project_root / PREREG_PATH, single_link=True)
    protocol.require(
        protocol.sha256_file(project_root / PREREG_PATH) == EXPECTED_PREREG_SHA256,
        "preregistration byte hash mismatch",
    )
    components = payload["component_commitments"]
    protocol.require(
        isinstance(components, dict) and set(components) == set(OPENING_NAMES),
        "pending component inventory must be the exact 31 openings",
    )
    for name, record in components.items():
        protocol.require_exact_keys(
            record, {"sha256", "size_bytes", "row_count"}, f"pending/{name}"
        )
        protocol.require(
            protocol.is_hex64(record["sha256"])
            and type(record["size_bytes"]) is int
            and record["size_bytes"] > 0,
            f"pending/{name} byte record invalid",
        )
        expected_rows = PHYSICAL_ROW_COUNTS[name]
        if expected_rows == "positive":
            protocol.require(type(record["row_count"]) is int and record["row_count"] > 0, f"pending/{name} rows invalid")
        else:
            protocol.require(record["row_count"] == expected_rows, f"pending/{name} rows mismatch")
    protocol.require(
        payload["sizing_rule"] == _expected_sizing_rule(components),
        "pending sizing_rule differs from preregistered values/hashes",
    )
    protocol.require(
        payload["curation_audit"] == _expected_curation_audit(components),
        "pending curation_audit differs from exact counts/hashes",
    )
    metadata = payload["public_metadata"]
    expected_metadata = _expected_public_metadata(components)
    protocol.require(
        isinstance(metadata, dict) and set(metadata) == set(expected_metadata),
        "pending public_metadata fields are not exact",
    )
    for key, value in expected_metadata.items():
        if key != "secret_commitments":
            protocol.require(
                metadata[key] == value,
                f"pending public_metadata mismatch: {key}",
            )
    secret_values = metadata["secret_commitments"]
    protocol.require(
        isinstance(secret_values, dict)
        and set(secret_values)
        == {
            "screening_seed",
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        }
        and all(protocol.is_hex64(value) for value in secret_values.values()),
        "pending secret commitment fields are not exact",
    )
    protocol.require(
        not protocol.contains_placeholder(payload["sizing_rule"])
        and not protocol.contains_placeholder(payload["curation_audit"])
        and not protocol.contains_placeholder(payload["public_metadata"]),
        "pending exact metadata contains placeholder",
    )
    protocol.require(
        pending_path == project_root / protocol.STAGE0_PUBLIC,
        "pending Stage-0 path is not standard",
    )
    return payload


def _validate_holdout_registry(
    payload: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "holdout_count",
            "status",
            "ordered_entries_sha256",
            "entries",
        },
        "holdout registry",
    )
    protocol.require(
        payload["protocol"] == "water_impact_dynamic_v4_holdout_registry_v3"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["holdout_count"] == 48
        and payload["status"] == "frozen",
        "holdout registry identity mismatch",
    )
    protocol.require(
        payload["entries"] == list(sources),
        "holdout registry/source ontology drift",
    )
    protocol.require(
        payload["ordered_entries_sha256"]
        == protocol.sha256_bytes(protocol.canonical_json_bytes(payload["entries"])),
        "holdout ordered entry hash mismatch",
    )


def _historical_receiver_inventory(
    mapping_payload: Mapping[str, Any], historical_payload: Mapping[str, Any]
) -> tuple[list[dict[str, str]], str]:
    mapping = mapping_payload.get("mapping")
    protocol.require(
        isinstance(mapping, list) and len(mapping) == 178,
        "frozen source mapping must contain exactly 178 rows",
    )
    pairs = sorted(
        {
            (str(row.get("receiver_id", "")), str(row.get("receiver", "")))
            for row in mapping
        }
    )
    protocol.require(
        pairs
        and all(receiver_id and receiver_phrase for receiver_id, receiver_phrase in pairs),
        "source mapping receiver inventory contains blank values",
    )
    inventory = [
        {"receiver_id": receiver_id, "receiver_phrase": receiver_phrase}
        for receiver_id, receiver_phrase in pairs
    ]
    digest = protocol.sha256_bytes(protocol.canonical_json_bytes(inventory))
    protocol.require(
        historical_payload["training_receiver_inventory_sha256"] == digest,
        "historical receiver inventory hash mismatch",
    )
    allowed = {(row["receiver_id"], row["receiver_phrase"]) for row in inventory}
    anchors = historical_payload["anchors"]
    protocol.require(
        len(anchors) == 8
        and len({row["receiver_id"] for row in anchors}) == 8,
        "historical anchor receiver inventory invalid",
    )
    for row in anchors:
        pair = (row["receiver_id"], row["receiver_phrase"])
        protocol.require(pair in allowed, "historical receiver pair is absent from mapping")
        expected = protocol.sha256_bytes(
            protocol.canonical_json_bytes(
                {
                    "receiver_id": row["receiver_id"],
                    "receiver_phrase": row["receiver_phrase"],
                }
            )
        )
        protocol.require(
            row["historical_training_binding_sha256"] == expected,
            "historical receiver row binding mismatch",
        )
    return inventory, digest


def _validate_render(payload: Mapping[str, Any], model_inventory_sha256: str) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "arm",
            "model_family",
            "model_content_inventory_sha256",
            "steps",
            "cfg",
            "frames",
            "width",
            "height",
            "fps",
            "dtype",
            "adapter",
            "screening_scope",
        },
        "render configuration",
    )
    protocol.require(
        payload
        == {
            "protocol": RENDER_PROTOCOL,
            "dataset_version": protocol.DATASET_VERSION,
            "status": "frozen",
            "arm": "Original_only",
            "model_family": "Wan 2.1 T2V 1.3B",
            "model_content_inventory_sha256": model_inventory_sha256,
            "steps": 25,
            "cfg": 5,
            "frames": 49,
            "width": 832,
            "height": 480,
            "fps": 8,
            "dtype": "bf16",
            "adapter": None,
            "screening_scope": "all 49 frames for every candidate",
        },
        "render configuration differs from preregistration",
    )


def _validate_rules(payload: Mapping[str, Any]) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "qualification",
            "cell_quota",
            "graph_permutation_domain",
            "graph_permutation_formula",
            "ranking_domain",
            "ranking_formula",
            "subset_algorithm",
            "evaluation_seed_domain",
            "evaluation_seed_formula",
            "replicates",
            "required_selected_cases",
            "required_evaluation_units",
        },
        "selection rules",
    )
    protocol.require(
        payload["protocol"] == RULES_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen"
        and payload["qualification"] == QUALIFICATION
        and payload["cell_quota"] == {
            "per_group_prompt_variant": 4,
            "selected_per_group": 8,
            "selected_total": 24,
        }
        and payload["graph_permutation_domain"]
        == "causal-graph-receiver-permutation-v3"
        and payload["graph_permutation_formula"] == GRAPH_FORMULA
        and payload["ranking_domain"] == protocol.RANK_DOMAIN
        and payload["ranking_formula"] == RANK_FORMULA
        and payload["subset_algorithm"]
        == {
            "algorithm": "rank_order_greedy_include_if_exact_completion_exists",
            "groups": builder.graph_topology(),
            "rank_tie_policy": "invalidate_data_version",
        }
        and payload["evaluation_seed_domain"] == protocol.SEED_DOMAIN
        and payload["evaluation_seed_formula"] == SEED_FORMULA
        and payload["replicates"] == [0, 1, 2]
        and payload["required_selected_cases"] == 24
        and payload["required_evaluation_units"] == 72,
        "selection rules differ from preregistration",
    )


def _secret_commitments(
    *,
    screening_seed: int,
    graph_salt: str,
    selector_salt: str,
    evaluation_salt: str,
) -> dict[str, str]:
    return {
        "screening_seed": _secret_commitment(
            "causal_screening_seed_v3", screening_seed
        ),
        "graph_assignment_salt": _secret_commitment(
            "causal_graph_assignment_salt_v3", graph_salt
        ),
        "selector_salt": _secret_commitment(
            "causal_stage0_selector_salt_v3", selector_salt
        ),
        "evaluation_seed_salt": _secret_commitment(
            "causal_evaluation_seed_salt_v3", evaluation_salt
        ),
    }


def validate_secret_sampling_request(
    payload: Mapping[str, Any],
    *,
    forbidden_seed_inventory_sha256: str,
    forbidden_numeric_seed_count: int,
    screening_seed: int | None = None,
    graph_salt: str | None = None,
    selector_salt: str | None = None,
    evaluation_salt: str | None = None,
) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "sampling_provenance",
            "raw_secret_values_emitted",
        },
        "secret sampling request",
    )
    provenance = payload["sampling_provenance"]
    protocol.require_exact_keys(
        provenance,
        {
            "entropy_source",
            "independent_draws",
            "salt_draw_count",
            "salt_bytes_per_draw",
            "salt_encoding",
            "salt_draw_attempts",
            "screening_seed_draw_count",
            "screening_seed_bytes_per_draw",
            "screening_seed_byte_order",
            "screening_seed_encoding",
            "screening_seed_draw_attempts",
            "new_secret_commitments",
            "forbidden_seed_inventory_sha256",
            "forbidden_numeric_seed_count",
            "screening_seed_forbidden_intersection_count",
        },
        "secret sampling provenance",
    )
    attempts = provenance["salt_draw_attempts"]
    protocol.require_exact_keys(
        attempts,
        {
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        },
        "salt draw attempts",
    )
    commitments = provenance["new_secret_commitments"]
    protocol.require_exact_keys(
        commitments,
        {
            "screening_seed",
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        },
        "new secret commitments",
    )
    protocol.require(
        payload["protocol"] == SECRET_SAMPLING_REQUEST_PROTOCOL
        and payload["status"] == "sampled_pending_historical_audit"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["raw_secret_values_emitted"] is False
        and provenance["entropy_source"] == "operating_system_csprng"
        and provenance["independent_draws"] is True
        and provenance["salt_draw_count"] == 3
        and provenance["salt_bytes_per_draw"] == 32
        and provenance["salt_encoding"] == "lower_hex64"
        and all(type(attempts[name]) is int and attempts[name] > 0 for name in attempts)
        and provenance["screening_seed_draw_count"] == 1
        and provenance["screening_seed_bytes_per_draw"] == 4
        and provenance["screening_seed_byte_order"] == "big_endian"
        and provenance["screening_seed_encoding"]
        == "canonical_unsigned_decimal_uint32"
        and type(provenance["screening_seed_draw_attempts"]) is int
        and provenance["screening_seed_draw_attempts"] > 0
        and all(protocol.is_hex64(value) for value in commitments.values())
        and provenance["forbidden_seed_inventory_sha256"]
        == forbidden_seed_inventory_sha256
        and type(provenance["forbidden_numeric_seed_count"]) is int
        and provenance["forbidden_numeric_seed_count"]
        == forbidden_numeric_seed_count
        and forbidden_numeric_seed_count > 0
        and provenance["screening_seed_forbidden_intersection_count"] == 0,
        "secret sampling request differs from the frozen contract",
    )
    raw = (screening_seed, graph_salt, selector_salt, evaluation_salt)
    if any(value is not None for value in raw):
        protocol.require(
            all(value is not None for value in raw),
            "all four raw secret openings are required together",
        )
        assert isinstance(screening_seed, int)
        assert isinstance(graph_salt, str)
        assert isinstance(selector_salt, str)
        assert isinstance(evaluation_salt, str)
        protocol.validate_secret_separation(
            graph_assignment_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
            screening_seed=screening_seed,
        )
        protocol.require(
            commitments
            == _secret_commitments(
                screening_seed=screening_seed,
                graph_salt=graph_salt,
                selector_salt=selector_salt,
                evaluation_salt=evaluation_salt,
            ),
            "secret sampling request does not bind the raw openings",
        )
    protocol.require(
        not protocol.contains_placeholder(payload),
        "secret sampling request contains placeholder",
    )
    return payload


def validate_historical_secret_audit(
    payload: Mapping[str, Any],
    *,
    forbidden_seed_inventory_sha256: str | None = None,
    forbidden_numeric_seed_count: int | None = None,
    new_salt_commitments: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    keys = {
        "protocol",
        "status",
        "v3_hex_salt_count",
        "new_salt_commitments",
        "accessible_historical_raw_allowlist_sha256",
        "accessible_historical_raw_hex_secret_count",
        "accessible_historical_raw_comparison_count",
        "accessible_historical_raw_intersection_count",
        "commitment_only_historical_allowlist_sha256",
        "commitment_only_historical_hex_secret_count",
        "commitment_only_comparison_count",
        "commitment_only_collision_union_bound_numerator",
        "commitment_only_collision_union_bound_denominator_power",
        "forbidden_seed_inventory_sha256",
        "forbidden_numeric_seed_count",
        "screening_seed_forbidden_intersection_count",
        "raw_historical_secret_values_emitted",
    }
    protocol.require_exact_keys(payload, keys, "historical secret audit")
    integer_fields = (
        "accessible_historical_raw_hex_secret_count",
        "accessible_historical_raw_comparison_count",
        "accessible_historical_raw_intersection_count",
        "commitment_only_historical_hex_secret_count",
        "commitment_only_comparison_count",
        "commitment_only_collision_union_bound_numerator",
        "commitment_only_collision_union_bound_denominator_power",
        "forbidden_numeric_seed_count",
        "screening_seed_forbidden_intersection_count",
    )
    protocol.require(
        payload["protocol"] == HISTORICAL_SECRET_AUDIT_PROTOCOL
        and payload["status"] == "passed"
        and payload["v3_hex_salt_count"] == 3
        and isinstance(payload["new_salt_commitments"], Mapping)
        and set(payload["new_salt_commitments"])
        == {
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        }
        and all(
            protocol.is_hex64(value)
            for value in payload["new_salt_commitments"].values()
        )
        and all(
            type(payload[name]) is int and payload[name] >= 0
            for name in integer_fields
        )
        and protocol.is_hex64(
            payload["accessible_historical_raw_allowlist_sha256"]
        )
        and protocol.is_hex64(
            payload["commitment_only_historical_allowlist_sha256"]
        )
        and protocol.is_hex64(payload["forbidden_seed_inventory_sha256"])
        and payload["accessible_historical_raw_comparison_count"]
        == 3 * payload["accessible_historical_raw_hex_secret_count"]
        and payload["accessible_historical_raw_allowlist_sha256"]
        == HISTORICAL_ACCESSIBLE_RAW_ALLOWLIST_SHA256
        and payload["accessible_historical_raw_hex_secret_count"]
        == HISTORICAL_ACCESSIBLE_RAW_COUNT
        and payload["accessible_historical_raw_hex_secret_count"] > 0
        and payload["accessible_historical_raw_intersection_count"] == 0
        and payload["commitment_only_comparison_count"]
        == 3 * payload["commitment_only_historical_hex_secret_count"]
        and payload["commitment_only_historical_allowlist_sha256"]
        == HISTORICAL_COMMITMENT_ALLOWLIST_SHA256
        and payload["commitment_only_historical_hex_secret_count"]
        == HISTORICAL_COMMITMENT_ONLY_COUNT
        and payload["commitment_only_historical_hex_secret_count"] > 0
        and payload["commitment_only_collision_union_bound_numerator"]
        == payload["commitment_only_comparison_count"]
        and payload["commitment_only_collision_union_bound_denominator_power"]
        == 256
        and payload["forbidden_numeric_seed_count"] > 0
        and payload["screening_seed_forbidden_intersection_count"] == 0
        and payload["raw_historical_secret_values_emitted"] is False,
        "historical secret audit contract failed",
    )
    if forbidden_seed_inventory_sha256 is not None:
        protocol.require(
            payload["forbidden_seed_inventory_sha256"]
            == forbidden_seed_inventory_sha256,
            "historical audit forbidden inventory hash mismatch",
        )
    if forbidden_numeric_seed_count is not None:
        protocol.require(
            payload["forbidden_numeric_seed_count"]
            == forbidden_numeric_seed_count,
            "historical audit forbidden seed count mismatch",
        )
    if new_salt_commitments is not None:
        protocol.require(
            payload["new_salt_commitments"] == dict(new_salt_commitments),
            "historical audit is bound to a different salt triplet",
        )
    protocol.require(
        not protocol.contains_placeholder(payload),
        "historical secret audit contains placeholder",
    )
    return payload


def _validate_secrets(
    payload: Mapping[str, Any], *, graph_salt: str, selector_salt: str,
    evaluation_salt: str, screening_seed: int,
    sampling_request: Mapping[str, Any] | None = None,
    sampling_request_sha256: str | None = None,
) -> dict[str, str]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "screening_seed_namespace",
            "screening_seed",
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_namespace",
            "evaluation_seed_salt",
            "sampling_request_sha256",
            "sampling_provenance",
            "historical_secret_audit",
        },
        "Stage-0 secrets",
    )
    protocol.require(
        payload["protocol"] == SECRETS_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen"
        and payload["screening_seed_namespace"] == protocol.SCREENING_NAMESPACE
        and payload["evaluation_seed_namespace"] == protocol.EVALUATION_NAMESPACE
        and payload["screening_seed"] == screening_seed
        and payload["graph_assignment_salt"] == graph_salt
        and payload["selector_salt"] == selector_salt
        and payload["evaluation_seed_salt"] == evaluation_salt
        and protocol.is_hex64(payload["sampling_request_sha256"]),
        "Stage-0 secret openings do not match",
    )
    provenance = payload["sampling_provenance"]
    request_from_secrets = {
        "protocol": SECRET_SAMPLING_REQUEST_PROTOCOL,
        "status": "sampled_pending_historical_audit",
        "dataset_version": protocol.DATASET_VERSION,
        "sampling_provenance": provenance,
        "raw_secret_values_emitted": False,
    }
    reconstructed_request_sha256 = protocol.sha256_bytes(
        protocol.canonical_json_bytes(request_from_secrets)
    )
    protocol.require(
        payload["sampling_request_sha256"] == reconstructed_request_sha256,
        "Stage-0 secrets do not bind the canonical retained sampling request",
    )
    validate_secret_sampling_request(
        request_from_secrets,
        forbidden_seed_inventory_sha256=provenance.get(
            "forbidden_seed_inventory_sha256"
        ) if isinstance(provenance, Mapping) else "",
        forbidden_numeric_seed_count=provenance.get(
            "forbidden_numeric_seed_count"
        ) if isinstance(provenance, Mapping) else -1,
        screening_seed=screening_seed,
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
    )
    if sampling_request is not None:
        protocol.require(
            sampling_request == request_from_secrets,
            "Stage-0 secrets do not embed the retained sampling request",
        )
    if sampling_request_sha256 is not None:
        protocol.require(
            reconstructed_request_sha256 == sampling_request_sha256,
            "Stage-0 secrets sampling request hash mismatch",
        )
    salt_commitments = {
        key: provenance["new_secret_commitments"][key]
        for key in (
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        )
    }
    validate_historical_secret_audit(
        payload["historical_secret_audit"],
        forbidden_seed_inventory_sha256=provenance[
            "forbidden_seed_inventory_sha256"
        ],
        forbidden_numeric_seed_count=provenance["forbidden_numeric_seed_count"],
        new_salt_commitments=salt_commitments,
    )
    protocol.validate_secret_separation(
        graph_assignment_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
    )
    return _secret_commitments(
        screening_seed=screening_seed,
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
    )


def _validate_model_inventory(
    payload: Mapping[str, Any], project_root: Path
) -> str:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "model_root",
            "file_count",
            "files",
            "inventory_sha256",
        },
        "model inventory",
    )
    protocol.require(
        payload["protocol"] == MODEL_INVENTORY_PROTOCOL
        and payload["status"] == "frozen"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["model_root"]
        == "models/Wan2.1-T2V-1.3B-Diffusers",
        "model inventory identity mismatch",
    )
    files = payload["files"]
    protocol.require(
        isinstance(files, list)
        and files
        and payload["file_count"] == len(files),
        "model file inventory invalid",
    )
    paths: list[str] = []
    model_root_path = project_root / payload["model_root"]
    protocol._require_no_symlink_components(model_root_path)
    protocol.require(model_root_path.is_dir(), "model root is not a directory")
    actual_paths: list[str] = []
    for candidate in model_root_path.rglob("*"):
        info = os.lstat(candidate)
        protocol.require(
            not stat.S_ISLNK(info.st_mode),
            "model inventory contains a symlink",
        )
        if stat.S_ISDIR(info.st_mode):
            continue
        protocol.require(
            stat.S_ISREG(info.st_mode),
            "model inventory contains a non-regular entry",
        )
        protocol.require(
            info.st_nlink == 1,
            "model inventory contains a hardlink",
        )
        actual_paths.append(candidate.relative_to(project_root).as_posix())
    actual_paths.sort()
    for record in files:
        protocol.require_exact_keys(record, {"path", "sha256", "size_bytes"}, "model file")
        path = project_root / record["path"]
        relative = protocol.validate_runtime_read_path(
            project_root, path, allow_v2=False
        )
        protocol.require(
            relative == record["path"]
            and record["path"].startswith(payload["model_root"].rstrip("/") + "/")
            and protocol.sha256_file(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            "model file byte mismatch",
        )
        paths.append(record["path"])
    protocol.require(paths == sorted(paths) and len(set(paths)) == len(paths), "model paths must be unique and sorted")
    protocol.require(paths == actual_paths, "model registry does not cover the exact on-disk inventory")
    protocol.require(
        payload["inventory_sha256"]
        == protocol.sha256_bytes(protocol.canonical_json_bytes(files)),
        "model inventory digest mismatch",
    )
    return payload["inventory_sha256"]


def _runtime_content_inventory(
    project_root: Path, runtime_root: Path
) -> dict[str, Any]:
    protocol._require_no_symlink_components(runtime_root)
    protocol.require(
        runtime_root.is_dir() and not runtime_root.is_symlink(),
        "runtime root is not a real directory",
    )
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(runtime_root.rglob("*")):
        info = os.lstat(path)
        protocol.require(
            not stat.S_ISLNK(info.st_mode),
            "runtime tree contains a symlink",
        )
        if stat.S_ISDIR(info.st_mode):
            continue
        protocol.require(
            stat.S_ISREG(info.st_mode),
            "runtime tree contains a special/non-regular entry",
        )
        protocol.require(
            info.st_nlink == 1,
            "runtime tree contains a hardlinked file",
        )
        relative = path.relative_to(runtime_root).as_posix()
        encoded = relative.encode("utf-8")
        protocol.require(
            b"\x00" not in encoded and relative not in {"", "."},
            "runtime inventory path is not canonical UTF-8",
        )
        digest.update(encoded)
        digest.update(b"\x00")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            protocol.require(
                stat.S_ISREG(opened.st_mode)
                and opened.st_nlink == 1
                and (opened.st_dev, opened.st_ino)
                == (info.st_dev, info.st_ino),
                "runtime file changed during inventory",
            )
            observed_size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                observed_size += len(chunk)
            protocol.require(
                observed_size == opened.st_size == info.st_size,
                "runtime file size changed during inventory",
            )
        finally:
            os.close(descriptor)
        digest.update(b"\n")
        file_count += 1
        total_bytes += info.st_size
    protocol.require(file_count > 0 and total_bytes > 0, "runtime tree is empty")
    return {
        "content_inventory_algorithm": RUNTIME_CONTENT_INVENTORY_ALGORITHM,
        "content_file_count": file_count,
        "content_total_bytes": total_bytes,
        "content_inventory_sha256": digest.hexdigest(),
    }


def _module_origin_record(
    project_root: Path, runtime_root: Path, origin: str
) -> dict[str, Any]:
    protocol.require(
        isinstance(origin, str) and origin,
        "runtime module origin is missing",
    )
    origin_path = Path(origin)
    protocol.require(origin_path.is_absolute(), "module origin must be absolute")
    protocol.require(
        origin_path == protocol._canonical_lexical_absolute(origin_path),
        "module origin contains a lexical alias",
    )
    protocol._require_no_symlink_components(origin_path)
    resolved_origin = origin_path.resolve(strict=True)
    resolved_runtime = runtime_root.resolve(strict=True)
    protocol.require(
        resolved_origin != resolved_runtime
        and resolved_runtime in resolved_origin.parents,
        "module origin escapes the frozen runtime root",
    )
    protocol._require_regular_file(origin_path, single_link=True)
    relative = origin_path.relative_to(project_root).as_posix()
    protocol.require(
        relative.startswith("models/.wan-runtime/"),
        "module origin path is not project-relative runtime content",
    )
    return {
        "path": relative,
        "sha256": protocol.sha256_file(origin_path),
        "size_bytes": origin_path.stat().st_size,
    }


def _validate_runtime_registry(
    payload: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
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
        },
        "runtime registry",
    )
    protocol.require(
        payload["protocol"] == RUNTIME_REGISTRY_PROTOCOL
        and payload["status"] == "frozen"
        and payload["dataset_version"] == protocol.DATASET_VERSION,
        "runtime registry identity mismatch",
    )
    protocol.require(
        payload["runtime_root"] == "models/.wan-runtime"
        and payload["python_executable"] == "models/.wan-runtime/bin/python"
        and payload["sys_prefix_policy"]
        == "realpath(sys.prefix)==realpath(runtime_root)",
        "runtime path/prefix policy mismatch",
    )
    runtime_root = project_root / payload["runtime_root"]
    executable = project_root / payload["python_executable"]
    protocol._require_no_symlink_components(runtime_root)
    protocol.require(
        runtime_root.is_dir() and not runtime_root.is_symlink(),
        "runtime root is not a real directory",
    )
    protocol._require_no_symlink_components(executable)
    protocol._require_regular_file(executable, single_link=True)
    protocol.require(
        {
            key: payload[key]
            for key in (
                "content_inventory_algorithm",
                "content_file_count",
                "content_total_bytes",
                "content_inventory_sha256",
            )
        }
        == _runtime_content_inventory(project_root, runtime_root),
        "runtime content inventory differs from all on-disk bytes",
    )
    protocol.require_exact_keys(
        payload["torch"], {"distribution_version", "module_version"}, "runtime torch"
    )
    protocol.require_exact_keys(
        payload["cuda"],
        {
            "available_required",
            "torch_cuda_version",
            "cudnn_version",
            "device_count",
            "device_models",
        },
        "runtime CUDA",
    )
    packages = payload["packages"]
    protocol.require(
        payload["python"] == EXPECTED_RUNTIME_PYTHON
        and payload["torch"] == EXPECTED_RUNTIME_TORCH
        and payload["cuda"]["available_required"] is True
        and payload["cuda"]["torch_cuda_version"] == "12.4"
        and payload["cuda"]["cudnn_version"] == 90100
        and packages == EXPECTED_RUNTIME_PACKAGE_VERSIONS,
        "runtime package inventory is not exact",
    )
    probe_source = """
import importlib, importlib.metadata, json, os, platform, sys
import torch
package_names = json.loads(sys.argv[1])
module_names = json.loads(sys.argv[2])
payload = {
    "executable_realpath": os.path.realpath(sys.executable),
    "prefix_realpath": os.path.realpath(sys.prefix),
    "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
    "torch": {"distribution_version": importlib.metadata.version("torch"), "module_version": torch.__version__},
    "cuda": {
        "available_required": torch.cuda.is_available() is True,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "device_models": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    },
    "packages": {name: importlib.metadata.version(name) for name in package_names},
    "module_origins": {name: importlib.import_module(name).__file__ for name in module_names},
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            os.fspath(executable),
            "-I",
            "-c",
            probe_source,
            json.dumps(sorted(RUNTIME_PACKAGE_NAMES), separators=(",", ":")),
            json.dumps(list(RUNTIME_ORIGIN_MODULES), separators=(",", ":")),
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    protocol.require(
        completed.returncode == 0,
        "frozen runtime child probe failed",
    )
    try:
        observed = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("frozen runtime child probe output is invalid") from exc
    protocol.require_exact_keys(
        observed,
        {
            "executable_realpath",
            "prefix_realpath",
            "python",
            "torch",
            "cuda",
            "packages",
            "module_origins",
        },
        "runtime child probe",
    )
    protocol.require(
        observed["executable_realpath"] == os.path.realpath(executable)
        and observed["prefix_realpath"] == os.path.realpath(runtime_root),
        "runtime child interpreter/prefix mismatch",
    )
    protocol.require(
        observed["python"] == payload["python"]
        and observed["torch"] == payload["torch"]
        and observed["cuda"] == payload["cuda"]
        and observed["packages"] == packages,
        "runtime child metadata differs from registry",
    )
    observed_origins = {
        name: _module_origin_record(
            project_root, runtime_root, observed["module_origins"][name]
        )
        for name in RUNTIME_ORIGIN_MODULES
    }
    protocol.require(
        set(observed["module_origins"]) == set(RUNTIME_ORIGIN_MODULES)
        and payload["module_origins"] == observed_origins,
        "runtime critical module origins/bytes differ from registry",
    )
    device_count = observed["cuda"]["device_count"]
    device_models = observed["cuda"]["device_models"]
    protocol.require(
        observed["cuda"]["available_required"] is True
        and type(device_count) is int
        and device_count > 0
        and isinstance(device_models, list)
        and len(device_models) == device_count
        and all(isinstance(name, str) and name for name in device_models),
        "frozen runtime child has no usable CUDA device",
    )
    protocol.require(
        {
            key: payload[key]
            for key in (
                "content_inventory_algorithm",
                "content_file_count",
                "content_total_bytes",
                "content_inventory_sha256",
            )
        }
        == _runtime_content_inventory(project_root, runtime_root),
        "runtime content changed during child validation",
    )
    return {
        "accelerator_type": "CUDA",
        "device_count": device_count,
        "device_models": device_models,
    }


def _cost_tree_records(run_dir: Path) -> list[dict[str, Any]]:
    protocol._require_no_symlink_components(run_dir)
    info = run_dir.stat()
    protocol.require(
        run_dir.is_dir()
        and not run_dir.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "cost calibration run directory must be real mode-700",
    )
    records: list[dict[str, Any]] = []
    directories: set[str] = set()
    for path in sorted(
        run_dir.rglob("*"),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        entry = os.lstat(path)
        protocol.require(
            not stat.S_ISLNK(entry.st_mode),
            "cost calibration tree contains symlink",
        )
        if stat.S_ISDIR(entry.st_mode):
            protocol.require(
                stat.S_IMODE(entry.st_mode) == 0o700,
                "cost calibration tree directory mode differs",
            )
            directories.add(path.relative_to(run_dir).as_posix())
            continue
        protocol.require(
            stat.S_ISREG(entry.st_mode)
            and entry.st_nlink == 1
            and stat.S_IMODE(entry.st_mode) == 0o600,
            "cost calibration tree file contract differs",
        )
        relative = path.relative_to(run_dir).as_posix()
        protocol.require(
            relative not in {"", "."}
            and ".." not in Path(relative).parts,
            "cost calibration tree path is not canonical",
        )
        records.append(
            {
                "path": relative,
                "sha256": protocol.sha256_file(path),
                "size_bytes": entry.st_size,
            }
        )
    protocol.require(
        directories
        == {
            relative
            for index in range(5)
            for relative in (
                f"render_{index:03d}",
                f"render_{index:03d}/videos",
            )
        },
        "cost calibration tree directory inventory differs",
    )
    protocol.require(
        [record["path"] for record in records]
        == sorted(record["path"] for record in records),
        "cost calibration tree records are not sorted",
    )
    return records


def _validate_calibration_video_decode(
    project_root: Path, video_paths: Sequence[Path]
) -> None:
    executable = project_root / "models/.wan-runtime/bin/python"
    protocol._require_no_symlink_components(executable)
    protocol._require_regular_file(executable, single_link=True)
    source = r'''import av, json, sys
paths = json.loads(sys.argv[1])
rows = []
for path in paths:
    container = av.open(path)
    stream = container.streams.video[0]
    count = 0
    width = None
    height = None
    for frame in container.decode(video=0):
        count += 1
        width = frame.width
        height = frame.height
    rate = stream.average_rate
    rows.append({"frames": count, "width": width, "height": height,
                 "fps_numerator": rate.numerator if rate is not None else None,
                 "fps_denominator": rate.denominator if rate is not None else None})
    container.close()
print(json.dumps(rows, sort_keys=True, separators=(",", ":")))
'''
    completed = subprocess.run(
        [
            os.fspath(executable),
            "-I",
            "-c",
            source,
            json.dumps([os.fspath(path) for path in video_paths], separators=(",", ":")),
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    protocol.require(completed.returncode == 0, "cost video decode probe failed")
    try:
        rows = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("cost video decode probe output is invalid") from exc
    protocol.require(
        isinstance(rows, list)
        and len(rows) == 5
        and all(
            row
            == {
                "frames": 49,
                "width": 832,
                "height": 480,
                "fps_numerator": 8,
                "fps_denominator": 1,
            }
            for row in rows
        ),
        "cost calibration videos do not decode as exact 49/832x480/8fps",
    )


def _expected_calibration_generic_generation(seed: int) -> dict[str, Any]:
    return {
        "baseline": "clean",
        "seed": 42,
        "seeds": [seed],
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


def _validate_cost_generic_manifest(
    *,
    project_root: Path,
    run_dir: Path,
    index: int,
    prompt_path: Path,
    generic_path: Path,
    expected_video_path: Path,
) -> None:
    generic = protocol.load_json(
        generic_path, project_root=project_root, allow_v2=False
    )
    protocol.require_exact_keys(
        generic,
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
        "cost generic generation manifest",
    )
    created = generic["created_at_utc"]
    protocol.require(isinstance(created, str), "cost timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("cost timestamp is invalid") from exc
    protocol.require(
        parsed.tzinfo is not None
        and generic["baseline"] == "clean"
        and generic["pipeline"] == "WanPipeline"
        and generic["model"] == "models/Wan2.1-T2V-1.3B-Diffusers"
        and generic["dry_run"] is False
        and generic["prompts"] == os.fspath(prompt_path)
        and generic["generation"]
        == _expected_calibration_generic_generation(CALIBRATION_SEEDS[index]),
        "cost generic generation contract differs",
    )
    items = generic["items"]
    protocol.require(
        isinstance(items, list) and len(items) == 1,
        "cost generic item inventory differs",
    )
    item = items[0]
    protocol.require_exact_keys(
        item,
        {
            "index",
            "prompt",
            "target_concept",
            "expected_effect",
            "seed",
            "video_path",
        },
        "cost generic item",
    )
    prompt = item["prompt"]
    target = item["target_concept"]
    video_path = protocol._canonical_lexical_absolute(Path(item["video_path"]))
    protocol.require(
        item["index"] == 0
        and isinstance(prompt, str)
        and hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        == CALIBRATION_PROMPT_SHA256[index]
        and isinstance(target, str)
        and target.strip() == target
        and target != ""
        and item["expected_effect"] == "public pre-Stage-0 cost calibration"
        and item["seed"] == CALIBRATION_SEEDS[index]
        and video_path == expected_video_path
        and prompt_path.read_bytes()
        == (
            f"{prompt} | {target} | public pre-Stage-0 cost calibration\n"
        ).encode("utf-8")
        and expected_video_path.parent.name == "videos"
        and expected_video_path.parent.parent == generic_path.parent
        and {entry.name for entry in generic_path.parent.iterdir()}
        == {generic_path.name, "videos"}
        and {entry.name for entry in expected_video_path.parent.iterdir()}
        == {expected_video_path.name},
        "cost generic prompt/seed/video binding differs",
    )


def _validate_cost_run_evidence(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    run_dir: Path,
    model_sha: str,
    runtime_sha: str,
    render_sha: str,
    live_hardware: Mapping[str, Any],
    code_registry_sha256: str,
    generator_sha256: str,
    generator_dependency_closure_sha256: str,
    media_runtime_packages: Mapping[str, str],
) -> None:
    expected_run_dir = project_root / protocol.DATA_ROOT / COST_CALIBRATION_RUN_DIRNAME
    protocol.require(
        protocol._canonical_lexical_absolute(run_dir) == expected_run_dir,
        "cost calibration run directory is not the standard path",
    )
    protocol.require(
        not os.path.lexists(run_dir / COST_FAILED_BASENAME),
        "failed cost calibration cannot be authorized",
    )
    manifest_path = run_dir / COST_RUN_MANIFEST_BASENAME
    protocol.validate_runtime_read_path(project_root, manifest_path, allow_v2=False)
    manifest = protocol.load_json(
        manifest_path, project_root=project_root, allow_v2=False
    )
    protocol.require_exact_keys(
        manifest,
        {
            "protocol",
            "status",
            "dataset_version",
            "model_content_inventory_sha256",
            "runtime_registry_sha256",
            "render_configuration_sha256",
            "code_registry_sha256",
            "generator_sha256",
            "generator_dependency_closure_sha256",
            "media_runtime_packages",
            "hardware",
            "generation_configuration",
            "calibration_count",
            "items",
        },
        "cost calibration run manifest",
    )
    generation = {
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
    protocol.require(
        manifest["protocol"] == COST_RUN_MANIFEST_PROTOCOL
        and manifest["status"] == "completed_before_cost_publication"
        and manifest["dataset_version"] == protocol.DATASET_VERSION
        and manifest["model_content_inventory_sha256"] == model_sha
        and manifest["runtime_registry_sha256"] == runtime_sha
        and manifest["render_configuration_sha256"] == render_sha
        and manifest["code_registry_sha256"] == code_registry_sha256
        and manifest["generator_sha256"] == generator_sha256
        and manifest["generator_dependency_closure_sha256"]
        == generator_dependency_closure_sha256
        and manifest["media_runtime_packages"] == dict(media_runtime_packages)
        and manifest["hardware"] == dict(live_hardware)
        and manifest["generation_configuration"] == generation
        and manifest["calibration_count"] == 5
        and isinstance(manifest["items"], list)
        and len(manifest["items"]) == 5,
        "cost calibration run manifest context differs",
    )
    item_keys = {
        "index",
        "prompt_sha256",
        "seed",
        "prompt_path",
        "prompt_file_sha256",
        "render_log_path",
        "render_log_sha256",
        "generic_manifest_path",
        "generic_manifest_sha256",
        "video_path",
        "video_sha256",
        "video_size_bytes",
        "wall_time_seconds",
        "frames",
        "width",
        "height",
        "fps",
    }
    video_paths: list[Path] = []
    expected_paths = {
        "calibration_reservation_v3.json",
        "execution_started_v3.json",
        COST_RUN_MANIFEST_BASENAME,
    }
    for index, item in enumerate(manifest["items"]):
        protocol.require_exact_keys(item, item_keys, "cost calibration item")
        paths = {
            "prompt": item["prompt_path"],
            "log": item["render_log_path"],
            "generic": item["generic_manifest_path"],
            "video": item["video_path"],
        }
        protocol.require(
            item["index"] == index
            and item["prompt_sha256"] == CALIBRATION_PROMPT_SHA256[index]
            and item["seed"] == CALIBRATION_SEEDS[index]
            and paths["prompt"] == f"prompt_{index:03d}.txt"
            and paths["log"] == f"render_{index:03d}.log"
            and paths["generic"]
            == f"render_{index:03d}/generation_manifest.json"
            and isinstance(paths["video"], str)
            and paths["video"].startswith(f"render_{index:03d}/videos/")
            and paths["video"].endswith(".mp4")
            and item["video_sha256"] == payload["video_sha256"][index]
            and item["video_size_bytes"] == payload["video_size_bytes"][index]
            and item["wall_time_seconds"] == payload["wall_time_seconds"][index]
            and item["frames"] == 49
            and item["width"] == 832
            and item["height"] == 480
            and item["fps"] == 8,
            "cost calibration item differs from fixed5 contract",
        )
        for label, relative in paths.items():
            path = run_dir / relative
            protocol.validate_runtime_read_path(project_root, path, allow_v2=False)
            protocol.require(
                path.relative_to(run_dir).as_posix() == relative,
                f"cost {label} path is not canonical",
            )
            expected_paths.add(relative)
        for relative, hash_key in (
            (paths["prompt"], "prompt_file_sha256"),
            (paths["log"], "render_log_sha256"),
            (paths["generic"], "generic_manifest_sha256"),
            (paths["video"], "video_sha256"),
        ):
            path = run_dir / relative
            protocol.require(
                protocol.sha256_file(path) == item[hash_key],
                f"cost evidence hash mismatch: {relative}",
            )
        protocol.require(
            (run_dir / paths["video"]).stat().st_size
            == item["video_size_bytes"],
            "cost video size mismatch",
        )
        _validate_cost_generic_manifest(
            project_root=project_root,
            run_dir=run_dir,
            index=index,
            prompt_path=run_dir / paths["prompt"],
            generic_path=run_dir / paths["generic"],
            expected_video_path=run_dir / paths["video"],
        )
        video_paths.append(run_dir / paths["video"])
    records = _cost_tree_records(run_dir)
    protocol.require(
        len(records) == 23
        and {record["path"] for record in records} == expected_paths,
        "cost calibration run tree is not the exact 23-file inventory",
    )
    protocol.require(
        protocol.sha256_file(manifest_path)
        == payload["calibration_run_manifest_sha256"]
        and protocol.sha256_bytes(protocol.canonical_json_bytes(records))
        == payload["calibration_tree_sha256"],
        "cost calibration manifest/tree commitment mismatch",
    )
    _validate_calibration_video_decode(project_root, video_paths)


def _validate_cost_calibration(
    payload: Mapping[str, Any], *, model_sha: str, runtime_sha: str,
    render_sha: str, live_hardware: Mapping[str, Any],
    code_registry_sha256: str, generator_sha256: str,
    generator_dependency_closure_sha256: str,
    media_runtime_packages: Mapping[str, str],
    project_root: Path | None = None,
    calibration_run_dir: Path | None = None,
) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "hardware",
            "model_content_inventory_sha256",
            "runtime_registry_sha256",
            "render_configuration_sha256",
            "code_registry_sha256",
            "generator_sha256",
            "generator_dependency_closure_sha256",
            "media_runtime_packages",
            "generation_configuration",
            "public_prompt_sha256",
            "calibration_seeds",
            "video_sha256",
            "video_size_bytes",
            "wall_time_seconds",
            "maximum_wall_time_seconds",
            "maximum_allowed_seconds",
            "candidate_count",
            "gpu_hour_cap",
            "passes",
            "calibration_run_manifest_sha256",
            "calibration_tree_sha256",
        },
        "screening cost calibration",
    )
    times = payload["wall_time_seconds"]
    prompts = payload["public_prompt_sha256"]
    seeds = payload["calibration_seeds"]
    video_hashes = payload["video_sha256"]
    video_sizes = payload["video_size_bytes"]
    protocol.require(
        payload["protocol"] == COST_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["hardware"] == dict(live_hardware)
        and payload["model_content_inventory_sha256"] == model_sha
        and payload["runtime_registry_sha256"] == runtime_sha
        and payload["render_configuration_sha256"] == render_sha
        and payload["code_registry_sha256"] == code_registry_sha256
        and payload["generator_sha256"] == generator_sha256
        and payload["generator_dependency_closure_sha256"]
        == generator_dependency_closure_sha256
        and payload["media_runtime_packages"] == dict(media_runtime_packages)
        and payload["generation_configuration"]
        == {
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
        and isinstance(prompts, list)
        and prompts == list(CALIBRATION_PROMPT_SHA256)
        and seeds == list(CALIBRATION_SEEDS)
        and isinstance(video_hashes, list)
        and len(video_hashes) == 5
        and len(set(video_hashes)) == 5
        and all(protocol.is_hex64(item) for item in video_hashes)
        and isinstance(video_sizes, list)
        and len(video_sizes) == 5
        and all(type(item) is int and item > 0 for item in video_sizes)
        and isinstance(times, list)
        and len(times) == 5
        and all(type(item) in (int, float) and item > 0 for item in times)
        and payload["maximum_wall_time_seconds"] == max(times)
        and payload["maximum_allowed_seconds"] == 600
        and max(times) <= 600
        and payload["candidate_count"] == 576
        and payload["gpu_hour_cap"] == 100
        and payload["passes"] is True
        and protocol.is_hex64(payload["calibration_run_manifest_sha256"])
        and protocol.is_hex64(payload["calibration_tree_sha256"]),
        "screening cost calibration failed",
    )
    protocol.require(
        (project_root is None) == (calibration_run_dir is None),
        "cost evidence project/run roots must be supplied together",
    )
    if project_root is not None and calibration_run_dir is not None:
        _validate_cost_run_evidence(
            payload,
            project_root=project_root,
            run_dir=calibration_run_dir,
            model_sha=model_sha,
            runtime_sha=runtime_sha,
            render_sha=render_sha,
            live_hardware=live_hardware,
            code_registry_sha256=code_registry_sha256,
            generator_sha256=generator_sha256,
            generator_dependency_closure_sha256=(
                generator_dependency_closure_sha256
            ),
            media_runtime_packages=media_runtime_packages,
        )


def _validate_capacity_common(payload: Mapping[str, Any], status: str) -> None:
    base_keys = {
        "protocol",
        "dataset_version",
        "status",
        "numpy_version",
        "bit_generator",
        "posterior",
        "anchor_model",
        "draw_order",
        "graph",
        "graph_robustness",
        "analytic_models",
        "oracle",
    }
    protocol.require(base_keys <= set(payload), "capacity artifact base fields missing")
    protocol.require(
        payload["protocol"] == capacity.PROTOCOL
        and payload["dataset_version"] == capacity.DATASET_VERSION
        and payload["status"] == status
        and payload["numpy_version"] == capacity.REQUIRED_NUMPY_VERSION
        and payload["bit_generator"] == capacity.BIT_GENERATOR
        and payload["posterior"] == "M0 Beta(x+1,9-x)"
        and payload["anchor_model"]
        == "Beta(p*kappa,(1-p)*kappa), kappa=(1-rho)/rho"
        and payload["draw_order"]
        == (
            "per 5000-row batch: one (B,6) posterior-p call; six cell-ordered "
            "(B,A_c) theta calls; optional G1/G2/G3 frailty calls; then separate "
            "uniform calls in cell, anchor, iteration, edge order"
        )
        and payload["graph"] == _json_normalized(capacity.graph_specification())
        and payload["graph_robustness"]
        == _json_normalized(capacity.graph_robustness_report())
        and payload["analytic_models"]
        == _json_normalized(capacity.analytic_capacity_report())
        and payload["oracle"]
        == {
            "engine": capacity.CompiledOracle.name,
            "embedded_c_source_sha256": capacity.ORACLE_C_SOURCE_SHA256,
            "self_test_against_python_reference": True,
        },
        "capacity common contract mismatch",
    )


def _validate_capacity_artifacts(
    model: Mapping[str, Any], search: Mapping[str, Any], confirm: Mapping[str, Any],
    graph_report: Mapping[str, Any]
) -> None:
    expected_base = {
        "protocol",
        "dataset_version",
        "status",
        "numpy_version",
        "bit_generator",
        "posterior",
        "anchor_model",
        "draw_order",
        "graph",
        "graph_robustness",
        "analytic_models",
        "oracle",
    }
    protocol.require(set(model) == expected_base, "capacity model spec fields are not exact")
    _validate_capacity_common(model, "frozen_capacity_model_spec")
    protocol.require(
        set(search) == expected_base | {"profile", "seed", "result", "reference_match", "decision"},
        "capacity search fields are not exact",
    )
    _validate_capacity_common(search, "exact_frozen_capacity_search_result")
    protocol.require(search["profile"] == "search" and search["seed"] == capacity.seed_record(capacity.SEARCH_DOMAIN) and search["reference_match"] is True, "capacity search identity mismatch")
    capacity.validate_reference_result("search", search["result"])
    protocol.require(
        search["decision"]
        == {
            "search_ceiling": capacity.SEARCH_WILSON_CEILING,
            "passes": True,
            "first_lattice_point": True,
            "larger_lattice_points_inspected": 0,
        },
        "capacity search decision mismatch",
    )
    protocol.require(
        set(confirm)
        == expected_base
        | {"profile", "scenario_order", "scenarios", "reference_match", "decision"},
        "capacity confirmation fields are not exact",
    )
    _validate_capacity_common(confirm, "exact_frozen_capacity_confirmation_result")
    protocol.require(
        confirm["profile"] == "combined_confirmation"
        and confirm["scenario_order"] == list(capacity.CONFIRMATION_PROFILE_ORDER)
        and set(confirm["scenarios"]) == set(capacity.CONFIRMATION_PROFILE_ORDER)
        and confirm["reference_match"] is True,
        "capacity confirmation identity mismatch",
    )
    for profile in capacity.CONFIRMATION_PROFILE_ORDER:
        scenario = confirm["scenarios"][profile]
        protocol.require_exact_keys(scenario, {"seed", "result", "reference_match"}, f"capacity/{profile}")
        protocol.require(
            scenario["seed"] == capacity.seed_record(capacity.REFERENCE_RESULTS[profile]["domain"])
            and scenario["reference_match"] is True,
            f"capacity/{profile} seed/reference mismatch",
        )
        capacity.validate_reference_result(profile, scenario["result"])
    protocol.require(
        confirm["decision"]
        == {
            "authorization_scenario": "rho010",
            "confirmation_ceiling": capacity.CONFIRM_WILSON_CEILING,
            "passes": True,
            "rho020_report_only": True,
            "shared_frailty_report_only": True,
        },
        "capacity confirmation decision mismatch",
    )
    protocol.require(
        graph_report == _json_normalized(capacity.graph_robustness_report()),
        "static graph report mismatch",
    )


def _validate_seed_audit(
    payload: Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]],
    evaluation_salt: str, screening_seed: int, forbidden: set[int],
    candidate_sha: str, evaluation_salt_sha: str, screening_seed_sha: str,
    forbidden_sha: str
) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_manifest_sha256",
            "evaluation_seed_salt_sha256",
            "screening_seed_sha256",
            "forbidden_seed_inventory_sha256",
            "seed_count",
            "unique_seed_count",
            "screening_collision_count",
            "forbidden_collision_count",
            "ordered_seed_records_sha256",
            "records",
        },
        "preselection seed audit",
    )
    records = payload["records"]
    expected: list[dict[str, Any]] = []
    for candidate in candidates:
        for replicate in protocol.REPLICATES:
            expected.append(
                {
                    "case_id": candidate["case_id"],
                    "replicate": replicate,
                    "seed": protocol.derive_evaluation_seed(
                        evaluation_salt, candidate["case_id"], replicate
                    ),
                }
            )
    seeds = [row["seed"] for row in expected]
    protocol.require(
        payload["protocol"] == SEED_AUDIT_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "passed"
        and payload["candidate_manifest_sha256"] == candidate_sha
        and payload["evaluation_seed_salt_sha256"] == evaluation_salt_sha
        and payload["screening_seed_sha256"] == screening_seed_sha
        and payload["forbidden_seed_inventory_sha256"] == forbidden_sha
        and payload["seed_count"] == 1728
        and payload["unique_seed_count"] == 1728
        and payload["screening_collision_count"] == 0
        and payload["forbidden_collision_count"] == 0
        and records == expected
        and len(set(seeds)) == 1728
        and screening_seed not in seeds
        and screening_seed not in forbidden
        and not (set(seeds) & forbidden)
        and payload["ordered_seed_records_sha256"]
        == protocol.sha256_bytes(protocol.canonical_json_bytes(records)),
        "preselection seed audit mismatch",
    )


def _validate_forbidden_seed_source_audit(
    payload: Mapping[str, Any],
    *,
    v3_inventory_sha256: str,
    v3_seed_count: int,
) -> None:
    """Validate the isolated aggregate proof without opening any v2 input."""

    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "v2_stage0_registry_sha256",
            "v2_forbidden_seed_inventory_sha256",
            "v3_forbidden_seed_inventory_sha256",
            "v2_seed_count",
            "v3_seed_count",
            "intersection_seed_count",
            "v2_missing_from_v3_count",
            "v3_additional_seed_count",
            "set_relation",
        },
        "forbidden seed source audit",
    )
    count_names = {
        "v2_seed_count",
        "v3_seed_count",
        "intersection_seed_count",
        "v2_missing_from_v3_count",
        "v3_additional_seed_count",
    }
    protocol.require(
        all(
            type(payload[name]) is int and payload[name] >= 0
            for name in count_names
        ),
        "forbidden seed source audit counts are invalid",
    )
    relation = payload["set_relation"]
    protocol.require(
        payload["protocol"] == protocol.FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["v2_stage0_registry_sha256"] == protocol.V2_STAGE0_SHA256
        and payload["v2_forbidden_seed_inventory_sha256"]
        == V2_FORBIDDEN_SEED_INVENTORY_SHA256
        and payload["v3_forbidden_seed_inventory_sha256"]
        == v3_inventory_sha256
        and payload["v2_seed_count"] > 0
        and payload["v3_seed_count"] == v3_seed_count
        and v3_seed_count > 0
        and payload["v2_missing_from_v3_count"] == 0
        and payload["intersection_seed_count"] == payload["v2_seed_count"]
        and payload["intersection_seed_count"]
        + payload["v2_missing_from_v3_count"]
        == payload["v2_seed_count"]
        and payload["intersection_seed_count"]
        + payload["v3_additional_seed_count"]
        == payload["v3_seed_count"]
        and isinstance(relation, str)
        and relation == "strict_superset"
        and payload["v3_additional_seed_count"] >= len(CALIBRATION_SEEDS),
        "forbidden seed source audit does not prove v2 coverage",
    )
    protocol.require(
        payload["v3_seed_count"] > payload["v2_seed_count"]
        and payload["v3_additional_seed_count"] > 0,
        "strict-superset forbidden-seed relation/count mismatch",
    )


def _validate_generation_spec(
    payload: Mapping[str, Any], *, candidate_sha: str, graph_sha: str,
    render_sha: str, screening_seed_sha: str, graph_salt_sha: str,
    model_sha: str, runtime_sha: str,
    generator_dependency_closure_sha256: str,
    media_runtime_packages: Mapping[str, str],
) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_manifest_sha256",
            "candidate_graph_sha256",
            "render_configuration_sha256",
            "screening_seed_sha256",
            "graph_assignment_salt_sha256",
            "model_content_inventory_sha256",
            "runtime_registry_sha256",
            "generator_dependency_closure_sha256",
            "media_runtime_packages",
            "candidate_count",
            "generation",
        },
        "generation spec",
    )
    protocol.require(
        payload["protocol"] == GENERATION_SPEC_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_original_screening"
        and payload["candidate_manifest_sha256"] == candidate_sha
        and payload["candidate_graph_sha256"] == graph_sha
        and payload["render_configuration_sha256"] == render_sha
        and payload["screening_seed_sha256"] == screening_seed_sha
        and payload["graph_assignment_salt_sha256"] == graph_salt_sha
        and payload["model_content_inventory_sha256"] == model_sha
        and payload["runtime_registry_sha256"] == runtime_sha
        and payload["generator_dependency_closure_sha256"]
        == generator_dependency_closure_sha256
        and payload["media_runtime_packages"] == dict(media_runtime_packages)
        and payload["candidate_count"] == 576
        and payload["generation"]
        == {
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
        },
        "generation spec mismatch",
    )


def _validate_bundle(
    payload: Mapping[str, Any], *,
    private_records: Mapping[str, Mapping[str, Any]]
) -> None:
    component_names = set(PRIVATE_INPUTS) - {"raw_root_bundle"}
    protocol.require_exact_keys(
        payload,
        {"protocol", "dataset_version", "status", "components"},
        "private root bundle",
    )
    protocol.require(
        payload["protocol"] == BUNDLE_PROTOCOL
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["status"] == "frozen_before_pending_commitment",
        "private root bundle identity mismatch",
    )
    components = payload["components"]
    protocol.require(
        isinstance(components, dict) and set(components) == component_names,
        "private root bundle component inventory mismatch",
    )
    for name in component_names:
        protocol.require(
            components[name] == private_records[name]["sha256"],
            f"private root bundle component mismatch: {name}",
        )


def _opening_paths(project_root: Path, private_root: Path) -> dict[str, Path]:
    output = {name: private_root / basename for name, basename in PRIVATE_INPUTS.items()}
    output.update({name: project_root / relative for name, relative in PUBLIC_OPENINGS.items()})
    return output


def _validate_private_inventory(private_root: Path) -> None:
    protocol._require_no_symlink_components(private_root)
    info = private_root.stat()
    protocol.require(
        private_root.is_dir()
        and not private_root.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "PRIVATE_V3_ROOT must be a real mode-700 directory",
    )
    entries = list(private_root.iterdir())
    protocol.require(
        {entry.name for entry in entries} == set(PRIVATE_INPUTS.values()),
        "PRIVATE_V3_ROOT inventory must contain exactly the 19 Stage-0 inputs",
    )
    for entry in entries:
        protocol.validate_private_path(private_root, entry)


def _records_for_openings(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: _physical_record(path, PHYSICAL_ROW_COUNTS[name])
        for name, path in paths.items()
    }


def _require_records_unchanged(
    paths: Mapping[str, Path], before: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    observed = _records_for_openings(paths)
    protocol.require(observed == dict(before), "Stage-0 opening bytes changed during authorization")
    return observed


def _wrapper_artifacts(
    *, project_root: Path, private_root: Path, opening_records: Mapping[str, Mapping[str, Any]],
    binding_path: Path
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    artifacts.update(opening_records)
    artifacts["upstream_source_bank_registry_64_v2"] = _file_record(project_root / protocol.V2_BANK, 64)
    artifacts["upstream_source_mapping_178_v2"] = _file_record(project_root / protocol.V2_MAPPING, 178)
    selection_record = opening_records["selection_rules"]
    artifacts["ranking_formula"] = dict(selection_record)
    artifacts["constrained_subset_algorithm"] = dict(selection_record)
    artifacts["seed_derivation_formula"] = dict(selection_record)
    del artifacts["selection_rules"]
    artifacts["selection_binding"] = _file_record(binding_path, None)
    artifacts["preregistration"] = _file_record(project_root / PREREG_PATH, None)
    artifacts["v2_public_aggregate_design_input"] = _file_record(
        project_root / protocol.V2_TERMINATION, 6
    )
    protocol.require(set(artifacts) == set(protocol.STAGE0_ARTIFACT_ROWS), "wrapper artifact inventory mismatch")
    return artifacts


def _write_public_wrapper_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    protocol.write_json_exclusive_atomic(path, payload, mode=0o644)


def _write_bytes_owned_exclusive(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    ownership: list[tuple[Path, tuple[int, int]]] | None = None,
) -> tuple[int, int]:
    protocol._require_no_symlink_components(path.parent)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite owned output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    opened_temporary = os.fstat(descriptor)
    owned_inode: tuple[int, int] | None = (
        opened_temporary.st_dev,
        opened_temporary.st_ino,
    )

    def remove_owned_leaf() -> None:
        if owned_inode is None or not os.path.lexists(path):
            return
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            return
        if (info.st_dev, info.st_ino) == owned_inode:
            path.unlink()

    def remove_owned_temporary() -> None:
        if owned_inode is None or not os.path.lexists(temporary):
            return
        info = os.lstat(temporary)
        if (
            stat.S_ISREG(info.st_mode)
            and (info.st_dev, info.st_ino) == owned_inode
        ):
            temporary.unlink()

    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_info = os.lstat(temporary)
        protocol.require(
            stat.S_ISREG(temporary_info.st_mode)
            and (temporary_info.st_dev, temporary_info.st_ino) == owned_inode
            and temporary_info.st_nlink == 1
            and stat.S_IMODE(temporary_info.st_mode) == mode,
            "owned output temporary inode changed before publication",
        )
        if ownership is not None:
            ownership.append((path, owned_inode))
        os.link(temporary, path)
        remove_owned_temporary()
        protocol.require(
            not os.path.lexists(temporary),
            "owned output temporary path was replaced during publication",
        )
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        remove_owned_leaf()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        remove_owned_temporary()
    assert owned_inode is not None
    return owned_inode


def _write_json_owned_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    mode: int,
    ownership: list[tuple[Path, tuple[int, int]]] | None = None,
) -> tuple[int, int]:
    return _write_bytes_owned_exclusive(
        path,
        protocol.canonical_json_bytes(dict(payload)),
        mode=mode,
        ownership=ownership,
    )


def _write_private_binding_exclusive(
    path: Path, payload: Mapping[str, Any]
) -> tuple[int, int]:
    return _write_json_owned_exclusive(path, payload, mode=0o600)


def _remove_owned_inode_if_present(path: Path, owned_inode: tuple[int, int]) -> None:
    if not os.path.lexists(path):
        return
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or (info.st_dev, info.st_ino) != owned_inode:
        raise RuntimeError(f"cannot safely roll back drifted artifact: {path}")
    path.unlink()


def _rollback_owned_outputs(
    owned: Sequence[tuple[Path, tuple[int, int]]]
) -> None:
    failures: list[BaseException] = []
    parents: set[Path] = set()
    for path, inode in reversed(owned):
        parents.add(path.parent)
        try:
            _remove_owned_inode_if_present(path, inode)
        except BaseException as exc:
            failures.append(exc)
    for parent in sorted(parents, key=os.fspath):
        try:
            _fsync_directory(parent)
        except BaseException as exc:
            failures.append(exc)
    if failures:
        raise RuntimeError("owned-output rollback was incomplete") from failures[0]


def _validate_public_output_path(
    project_root: Path, path: Path, expected_relative: Path
) -> None:
    protocol.require(
        protocol._canonical_lexical_absolute(path)
        == project_root / expected_relative,
        "public output path is not exact",
    )
    protocol.reject_forbidden_path(path)
    protocol._require_no_symlink_components(path.parent)
    protocol.require(path.parent.is_dir(), "public output parent is missing")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite public output: {path}")


class StaticAuthorizationFailure(RuntimeError):
    """A pre-wrapper path/permission/serialization failure that may be repaired."""


def _reject_existing_terminal_or_stage1(project_root: Path) -> None:
    for relative in (protocol.INVALID_OUTCOME, protocol.STAGE1_REGISTRY):
        target = project_root / relative
        if os.path.lexists(target):
            raise FileExistsError(
                f"v3 authorization is already terminal or Stage-1 exists: {target}"
            )


def _reject_existing_preparation_boundary(project_root: Path) -> None:
    """Reject every scientific/publication boundary before producer work.

    Producer entry points call this immediately after validating the lexical
    project root, before opening a private root, probing a runtime/model, or
    constructing an output payload.  A pending commitment freezes the
    scientific inputs; a standard Stage-0 wrapper, terminal invalid outcome,
    or Stage-1 wrapper is an even later one-shot boundary.
    """

    for relative in (
        protocol.STAGE0_PUBLIC,
        protocol.STAGE0_REGISTRY,
        protocol.INVALID_OUTCOME,
        protocol.STAGE1_REGISTRY,
    ):
        target = project_root / relative
        if os.path.lexists(target):
            raise FileExistsError(
                f"v3 preparation boundary already exists: {target}"
            )


def _require_disjoint_project_private_roots(
    project_root: Path, private_root: Path
) -> Path:
    """Return the canonical private root after proving roots are non-nested."""

    private_root = protocol._canonical_lexical_absolute(private_root)
    project_root = protocol._canonical_lexical_absolute(project_root)
    lexical_nested = (
        private_root == project_root
        or private_root in project_root.parents
        or project_root in private_root.parents
    )
    protocol.require(not lexical_nested, "project and private roots must be disjoint")
    protocol._require_no_symlink_components(private_root)
    protocol.require(
        private_root.is_dir() and not private_root.is_symlink(),
        "private root must be a real directory",
    )
    resolved_project = project_root.resolve(strict=True)
    resolved_private = private_root.resolve(strict=True)
    resolved_nested = (
        resolved_private == resolved_project
        or resolved_private in resolved_project.parents
        or resolved_project in resolved_private.parents
    )
    protocol.require(
        not resolved_nested,
        "resolved project and private roots must be disjoint",
    )
    return private_root


def _require_distinct_mode700_root(
    root: Path, *, other_roots: Sequence[Path]
) -> Path:
    root = protocol._canonical_lexical_absolute(root)
    protocol.reject_forbidden_path(root)
    protocol._require_no_symlink_components(root)
    info = root.stat()
    protocol.require(
        root.is_dir()
        and not root.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "isolated secret root must be a real mode-700 directory",
    )
    resolved = root.resolve(strict=True)
    for other in other_roots:
        other_lexical = protocol._canonical_lexical_absolute(other)
        other_resolved = other_lexical.resolve(strict=True)
        protocol.require(
            root != other_lexical
            and root not in other_lexical.parents
            and other_lexical not in root.parents
            and resolved != other_resolved
            and resolved not in other_resolved.parents
            and other_resolved not in resolved.parents,
            "secret, private, and project roots must be distinct and nonnested",
        )
    return root


@contextlib.contextmanager
def _multi_root_mutex(*roots: Path):
    canonical = sorted(
        {protocol._canonical_lexical_absolute(root) for root in roots},
        key=os.fspath,
    )
    protocol.require(
        len(canonical) == len(roots),
        "mutex roots must be distinct",
    )
    descriptors: list[int] = []
    try:
        for root in canonical:
            protocol._require_no_symlink_components(root)
            info = root.stat()
            protocol.require(
                root.is_dir()
                and not root.is_symlink()
                and stat.S_IMODE(info.st_mode) == 0o700,
                "secret transaction requires real mode-700 roots",
            )
            descriptor = os.open(
                root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)
        yield
    except BlockingIOError as exc:
        raise FileExistsError("another secret transaction owns a required root") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_complete_secret_audit_root(
    project_root: Path,
    private_root: Path,
    secret_audit_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    secret_audit_root = _require_distinct_mode700_root(
        secret_audit_root,
        other_roots=(project_root, private_root),
    )
    expected = {
        SECRET_SAMPLING_REQUEST_BASENAME,
        HISTORICAL_SECRET_AUDIT_BASENAME,
    }
    protocol.require(
        {entry.name for entry in secret_audit_root.iterdir()} == expected,
        "secret audit root must retain exactly request and historical audit",
    )
    request_path = secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME
    audit_path = secret_audit_root / HISTORICAL_SECRET_AUDIT_BASENAME
    for path in (request_path, audit_path):
        protocol.validate_private_path(secret_audit_root, path)
    request = protocol.load_json(
        request_path, private_root=secret_audit_root
    )
    audit = protocol.load_json(audit_path, private_root=secret_audit_root)
    return secret_audit_root, request, audit, protocol.sha256_file(request_path)


def _require_historical_projection_root(
    root: Path, *, expected_names: set[str], other_roots: Sequence[Path]
) -> Path:
    root = protocol._canonical_lexical_absolute(root)
    lowered = root.as_posix().casefold()
    protocol.require(
        not any(marker in lowered for marker in ("sealed", "final36", "quarantine")),
        "historical projection root enters a forbidden category",
    )
    protocol._require_no_symlink_components(root)
    info = root.stat()
    protocol.require(
        root.is_dir()
        and not root.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "historical projection root must be a real mode-700 directory",
    )
    resolved = root.resolve(strict=True)
    for other in other_roots:
        other_path = protocol._canonical_lexical_absolute(other)
        other_resolved = other_path.resolve(strict=True)
        protocol.require(
            root != other_path
            and root not in other_path.parents
            and other_path not in root.parents
            and resolved != other_resolved
            and resolved not in other_resolved.parents
            and other_resolved not in resolved.parents,
            "historical projection roots must be distinct and nonnested",
        )
    protocol.require(
        {entry.name for entry in root.iterdir()} == expected_names,
        "historical projection inventory is not exact",
    )
    for name in expected_names:
        _validate_historical_projection_file(root, root / name)
    return root


def _validate_historical_projection_file(root: Path, path: Path) -> None:
    protocol.require(
        protocol._canonical_lexical_absolute(path) == root / path.name
        and path.parent == root,
        "historical projection file is not an exact root child",
    )
    protocol._require_no_symlink_components(path)
    info = os.lstat(path)
    protocol.require(
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600,
        "historical projection file mode/link/type contract failed",
    )


def _load_historical_raw_projection(root: Path, kind: str) -> set[str]:
    expected = HISTORICAL_RAW_SOURCE_FILES[kind]
    protocol.require(
        {entry.name for entry in root.iterdir()} == set(expected),
        f"{kind} historical projection inventory changed",
    )
    values: list[str] = []
    salts_keys = {
        "schema",
        "protocol",
        "dataset_version",
        "source_ontology_salt",
        "source_split_salt",
        "receiver_ontology_salt",
        "causal_stage0_selector_salt",
        "causal_evaluation_seed_salt",
        "causal_screening_seed_token",
    }
    secrets_keys = {
        "schema",
        "protocol",
        "dataset_version",
        "screening_seed_namespace",
        "screening_seed",
        "evaluation_seed_namespace",
        "evaluation_seed_salt",
        "selector_salt",
    }
    for name, record in expected.items():
        path = root / name
        _validate_historical_projection_file(root, path)
        protocol.require(
            protocol.sha256_file(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            f"historical raw source bytes changed: {kind}/{name}",
        )
        if name == "salts_private_v2.json":
            payload = json.loads(path.read_bytes())
            protocol.require_exact_keys(payload, salts_keys, "v2 clean salt source")
            protocol.require(
                payload["schema"] == "water_impact_dynamic_v4_source_slot_registry_v2"
                and payload["protocol"]
                == "water_impact_dynamic_v4_source_slot_registry_v2"
                and payload["dataset_version"] == "v4_dev72_v2",
                "v2 clean salt source identity mismatch",
            )
            names = (
                "source_ontology_salt",
                "source_split_salt",
                "receiver_ontology_salt",
                "causal_stage0_selector_salt",
                "causal_evaluation_seed_salt",
                "causal_screening_seed_token",
            )
            extracted = [payload[field] for field in names]
        elif name == "causal_stage0_secrets_private_v2.json":
            payload = json.loads(path.read_bytes())
            protocol.require_exact_keys(payload, secrets_keys, "v2 causal secrets source")
            protocol.require(
                payload["schema"] == "water_impact_dynamic_v4_source_slot_registry_v2"
                and payload["protocol"]
                == "water_impact_dynamic_v4_source_slot_registry_v2"
                and payload["dataset_version"] == "v4_dev72_v2"
                and payload["screening_seed_namespace"]
                == "v4-causal-stage0-screening-v2"
                and payload["evaluation_seed_namespace"]
                == "v4-causal-evaluation-v2",
                "v2 causal secrets source identity mismatch",
            )
            extracted = [payload["evaluation_seed_salt"], payload["selector_salt"]]
        else:
            raw = path.read_bytes()
            protocol.require(
                len(raw) == 65
                and raw.endswith(b"\n")
                and protocol.is_hex64(raw[:-1].decode("ascii")),
                f"historical raw salt opening is not canonical: {name}",
            )
            extracted = [raw[:-1].decode("ascii")]
        protocol.require(
            len(extracted) == record["raw_hex64_count"]
            and all(protocol.is_hex64(value) for value in extracted),
            f"historical raw source extraction count mismatch: {kind}/{name}",
        )
        values.extend(extracted)
    return set(values)


def _load_commitment_only_historical_allowlist(project_root: Path) -> set[str]:
    commitments: set[str] = set()
    for relative, spec in HISTORICAL_COMMITMENT_PUBLIC_SOURCES.items():
        path = project_root / relative
        protocol._require_no_symlink_components(path)
        protocol._require_regular_file(path, single_link=True)
        protocol.require(
            protocol.sha256_file(path) == spec["sha256"],
            f"public v2 commitment source bytes changed: {relative}",
        )
        payload = json.loads(path.read_bytes())
        for field, expected in spec["fields"].items():
            protocol.require(
                payload.get(field) == expected and protocol.is_hex64(expected),
                f"public v2 commitment field changed: {relative}/{field}",
            )
            commitments.add(expected)
    protocol.require(
        len(commitments) == HISTORICAL_COMMITMENT_ONLY_COUNT
        and protocol.sha256_bytes(
            protocol.canonical_json_bytes(sorted(commitments))
        )
        == HISTORICAL_COMMITMENT_ALLOWLIST_SHA256,
        "commitment-only historical allowlist digest/count changed",
    )
    return commitments


@contextlib.contextmanager
def _authorization_mutex(private_root: Path):
    private_root = protocol._canonical_lexical_absolute(private_root)
    protocol._require_no_symlink_components(private_root)
    info = private_root.stat()
    protocol.require(
        private_root.is_dir()
        and not private_root.is_symlink()
        and stat.S_IMODE(info.st_mode) == 0o700,
        "authorization mutex requires a real mode-700 PRIVATE_V3_ROOT",
    )
    descriptor = os.open(
        private_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(
                "another Stage-0 authorizer owns PRIVATE_V3_ROOT"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _is_exact_published_json(
    path: Path, payload: Mapping[str, Any], *, mode: int
) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    info = path.stat()
    return (
        info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == mode
        and protocol.sha256_file(path)
        == protocol.sha256_bytes(protocol.canonical_json_bytes(dict(payload)))
    )


def _publish_authorization_invalid(
    project_root: Path, *, stage0_sha256: str | None
) -> dict[str, Any]:
    after_wrapper = stage0_sha256 is not None
    payload = {
        "protocol": protocol.INVALID_OUTCOME_PROTOCOL,
        "dataset": protocol.DATASET,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "preflight_dataset_invalid",
        "failure_phase": (
            "original_generation" if after_wrapper else "stage0_authorization"
        ),
        "reason_code": (
            "screening_generation_incomplete"
            if after_wrapper
            else "stage0_authorization_integrity_failure"
        ),
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
            "screening_generation_manifest": None,
            "screening_package_commitment": None,
            "screening_freeze_manifest": None,
            "canonical_eligibility": None,
            "selector_stderr": None,
        },
    }
    protocol.validate_invalid_outcome(
        payload, expected_stage0_sha256=stage0_sha256
    )
    protocol.write_json_exclusive_atomic(
        project_root / protocol.INVALID_OUTCOME, payload, mode=0o644
    )
    return payload


def _authorize_impl(
    *, project_root: Path, private_root: Path, pending_path: Path,
    binding_path: Path, wrapper_path: Path,
    state: dict[str, Any],
    preflight_only: bool = False,
    pending_payload_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    private_root = protocol._canonical_lexical_absolute(private_root)
    protocol._require_no_symlink_components(private_root)
    private_resolved = private_root.resolve(strict=True)
    protocol.require(
        private_resolved != project_root
        and private_resolved not in project_root.parents
        and project_root not in private_resolved.parents,
        "project and PRIVATE_V3_ROOT must be distinct and nonnested",
    )
    protocol.require(pending_path == project_root / protocol.STAGE0_PUBLIC, "pending path is not standard")
    protocol.require(wrapper_path == project_root / protocol.STAGE0_REGISTRY, "wrapper path is not standard")
    protocol.require(binding_path == private_root / "causal_selection_binding_v3.json", "selection binding path is not standard")
    protocol.validate_private_output_path(private_root, binding_path)
    _validate_public_output_path(
        project_root, wrapper_path, protocol.STAGE0_REGISTRY
    )

    v2_before = protocol.validate_v2_public_inputs(project_root)
    expected_code = build_code_registry_payload(project_root)  # missing code fails before private opening
    code_payload = _load_public_json(project_root, protocol.CODE_REGISTRY)
    validate_code_registry_full(code_payload, project_root)

    _validate_private_inventory(private_root)

    pending = (
        dict(pending_payload_override)
        if pending_payload_override is not None
        else _load_public_json(project_root, protocol.STAGE0_PUBLIC)
    )
    validate_pending(pending, project_root=project_root, pending_path=pending_path)
    state["pending_validated"] = True
    pending_sha = (
        protocol.sha256_bytes(protocol.canonical_json_bytes(pending))
        if pending_payload_override is not None
        else protocol.sha256_file(pending_path)
    )
    paths = _opening_paths(project_root, private_root)
    for name in PRIVATE_INPUTS:
        protocol.validate_private_path(private_root, paths[name])
    opening_records = _records_for_openings(paths)
    protocol.require(opening_records == pending["component_commitments"], "pending commitments do not match exact 31 opening bytes")

    source_payload = _load_private_json(private_root, PRIVATE_INPUTS["eval_holdout_source_ontology_48"])
    sources = builder.validate_holdout_ontology(source_payload)
    holdout_payload = _load_private_json(private_root, PRIVATE_INPUTS["holdout_registry_48"])
    _validate_holdout_registry(holdout_payload, sources)
    receiver_payload = _load_private_json(private_root, PRIVATE_INPUTS["receiver_ontology_56"])
    builder.validate_receiver_ontology(receiver_payload)
    historical_payload = _load_private_json(private_root, PRIVATE_INPUTS["historical_receiver_anchors_8"])
    builder.validate_historical_anchors(historical_payload)
    mapping_payload = protocol.load_json(
        project_root / protocol.V2_MAPPING,
        project_root=project_root,
        allow_v2=True,
    )
    historical_inventory, historical_inventory_sha = _historical_receiver_inventory(
        mapping_payload, historical_payload
    )
    candidate_payload = _load_private_json(private_root, PRIVATE_INPUTS["candidate_manifest_576"])
    graph_payload = _load_private_json(private_root, PRIVATE_INPUTS["candidate_graph_576"])
    builder.validate_candidate_projection(graph_payload, candidate_payload)
    builder.validate_templates_and_fields(
        paths["canonical_templates"], paths["field_normalization"], private_root=private_root
    )

    graph_salt = _read_text_secret(paths["graph_assignment_salt"])
    selector_salt = _read_text_secret(paths["selector_salt"])
    evaluation_salt = _read_text_secret(paths["evaluation_seed_salt"])
    screening_seed = _read_text_secret(paths["screening_seed"], integer=True)
    assert isinstance(graph_salt, str) and isinstance(selector_salt, str)
    assert isinstance(evaluation_salt, str) and isinstance(screening_seed, int)
    secrets_payload = _load_private_json(private_root, PRIVATE_INPUTS["stage0_secrets"])
    secret_commitments = _validate_secrets(
        secrets_payload,
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
    )
    protocol.require(
        pending["public_metadata"]
        == _expected_public_metadata(opening_records, secret_commitments),
        "pending public metadata/secret commitments mismatch",
    )
    protocol.require(
        graph_payload["graph_assignment_salt_sha256"]
        == opening_records["graph_assignment_salt"]["sha256"],
        "graph does not bind exact graph-assignment salt file",
    )
    source_bank_payload = protocol.load_json(
        project_root / protocol.V2_BANK, project_root=project_root, allow_v2=True
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

    model_payload = _load_public_json(project_root, MODEL_INVENTORY_PATH)
    model_sha = _validate_model_inventory(model_payload, project_root)
    runtime_payload = _load_public_json(project_root, RUNTIME_REGISTRY_PATH)
    runtime_sha = opening_records["runtime_registry"]["sha256"]
    live_hardware = _validate_runtime_registry(runtime_payload, project_root)
    _, generator_closure_sha256 = generator_dependency_closure(project_root)
    media_runtime_packages = probe_media_runtime_packages(project_root)
    render_payload = _load_private_json(private_root, PRIVATE_INPUTS["raw_render_configuration"])
    _validate_render(render_payload, model_sha)
    render_sha = opening_records["raw_render_configuration"]["sha256"]
    rules_payload = _load_private_json(private_root, PRIVATE_INPUTS["selection_rules"])
    _validate_rules(rules_payload)
    generation_payload = _load_private_json(private_root, PRIVATE_INPUTS["screening_generation_spec"])
    _validate_generation_spec(
        generation_payload,
        candidate_sha=opening_records["candidate_manifest_576"]["sha256"],
        graph_sha=opening_records["candidate_graph_576"]["sha256"],
        render_sha=render_sha,
        screening_seed_sha=opening_records["screening_seed"]["sha256"],
        graph_salt_sha=opening_records["graph_assignment_salt"]["sha256"],
        model_sha=model_sha,
        runtime_sha=runtime_sha,
        generator_dependency_closure_sha256=generator_closure_sha256,
        media_runtime_packages=media_runtime_packages,
    )
    forbidden_payload = _load_private_json(private_root, PRIVATE_INPUTS["forbidden_seed_inventory"])
    forbidden = selector.validate_forbidden_seed_inventory(forbidden_payload)
    protocol.require(
        set(CALIBRATION_SEEDS) <= forbidden,
        "v3 forbidden inventory omits fixed calibration seeds",
    )
    protocol.require(
        {
            "name": "v3_screening_cost_calibration_seeds",
            "sha256": code_payload["artifacts"]["screening_runner"]["sha256"],
            "seed_count": 5,
        }
        in forbidden_payload["source_commitments"],
        "forbidden inventory lacks code-bound calibration seed source",
    )
    validate_historical_secret_audit(
        secrets_payload["historical_secret_audit"],
        forbidden_seed_inventory_sha256=opening_records[
            "forbidden_seed_inventory"
        ]["sha256"],
        forbidden_numeric_seed_count=len(forbidden),
    )
    forbidden_source_audit = _load_public_json(
        project_root, protocol.FORBIDDEN_SEED_SOURCE_AUDIT
    )
    _validate_forbidden_seed_source_audit(
        forbidden_source_audit,
        v3_inventory_sha256=opening_records["forbidden_seed_inventory"][
            "sha256"
        ],
        v3_seed_count=len(forbidden),
    )
    seed_payload = _load_private_json(private_root, PRIVATE_INPUTS["preselection_seed_audit_1728"])
    _validate_seed_audit(
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

    identity_payload = _load_public_json(project_root, protocol.IDENTITY_REPORT)
    protocol.validate_identity_disjointness_report(identity_payload)
    holdout_public_payload = _load_public_json(
        project_root, protocol.HOLDOUT_PUBLIC_COMMITMENT
    )
    validate_holdout_public_commitment(
        holdout_public_payload,
        identity_report_sha256=opening_records["identity_disjointness_report"][
            "sha256"
        ],
        private_records=opening_records,
    )
    ontology_bundle_sha = protocol.sha256_bytes(
        protocol.canonical_json_bytes(
            {
                PRIVATE_INPUTS["eval_holdout_source_ontology_48"]: opening_records[
                    "eval_holdout_source_ontology_48"
                ]["sha256"],
                PRIVATE_INPUTS["receiver_ontology_56"]: opening_records[
                    "receiver_ontology_56"
                ]["sha256"],
                PRIVATE_INPUTS["historical_receiver_anchors_8"]: opening_records[
                    "historical_receiver_anchors_8"
                ]["sha256"],
            }
        )
    )
    protocol.require(
        identity_payload["v3_candidate_graph_sha256"]
        == opening_records["candidate_graph_576"]["sha256"]
        and identity_payload["v3_ontology_bundle_sha256"]
        == ontology_bundle_sha,
        "identity report graph/ontology binding mismatch",
    )
    construct_payload = _load_public_json(project_root, protocol.CONSTRUCT_REPORT)
    protocol.validate_construct_equivalence_report(construct_payload)
    protocol.require(
        construct_payload["v3_file_sha256"]
        == {
            "templates": opening_records["canonical_templates"]["sha256"],
            "field_rules": opening_records["field_normalization"]["sha256"],
            "selection_rules": opening_records["selection_rules"]["sha256"],
        },
        "construct report v3 byte binding mismatch",
    )
    protocol.require(
        construct_payload["qualification_sha256"]["v3"]
        == protocol.sha256_bytes(
            protocol.canonical_json_bytes(rules_payload["qualification"])
        )
        and construct_payload["cell_quota_sha256"]["v3"]
        == protocol.sha256_bytes(
            protocol.canonical_json_bytes(rules_payload["cell_quota"])
        ),
        "construct report v3 subobject binding mismatch",
    )
    capacity_model = _load_public_json(project_root, CAPACITY_MODEL_PATH)
    capacity_search = _load_public_json(project_root, CAPACITY_SEARCH_PATH)
    capacity_confirm = _load_public_json(project_root, CAPACITY_CONFIRM_PATH)
    static_graph = _load_public_json(project_root, STATIC_GRAPH_PATH)
    _validate_capacity_artifacts(
        capacity_model, capacity_search, capacity_confirm, static_graph
    )
    cost_payload = _load_public_json(project_root, COST_CALIBRATION_PATH)
    _validate_cost_calibration(
        cost_payload,
        model_sha=model_sha,
        runtime_sha=runtime_sha,
        render_sha=render_sha,
        live_hardware=live_hardware,
        code_registry_sha256=opening_records["eval_code_registry"]["sha256"],
        generator_sha256=code_payload["artifacts"]["generator"]["sha256"],
        generator_dependency_closure_sha256=generator_closure_sha256,
        media_runtime_packages=media_runtime_packages,
        project_root=project_root,
        calibration_run_dir=(
            project_root / protocol.DATA_ROOT / COST_CALIBRATION_RUN_DIRNAME
        ),
    )
    bundle_payload = _load_private_json(private_root, PRIVATE_INPUTS["raw_root_bundle"])
    _validate_bundle(
        bundle_payload,
        private_records={name: opening_records[name] for name in PRIVATE_INPUTS},
    )

    def revalidate_deep(label: str) -> None:
        protocol.require(
            (
                protocol.sha256_bytes(protocol.canonical_json_bytes(pending))
                if pending_payload_override is not None
                else protocol.sha256_file(pending_path)
            )
            == pending_sha,
            f"pending Stage-0 bytes changed {label}",
        )
        protocol.require(
            protocol.sha256_file(project_root / PREREG_PATH)
            == EXPECTED_PREREG_SHA256,
            f"preregistration bytes changed {label}",
        )
        _require_records_unchanged(paths, opening_records)
        protocol.require(
            build_code_registry_payload(project_root) == expected_code,
            f"code bytes changed {label}",
        )
        protocol.require(
            protocol.validate_v2_public_inputs(project_root) == v2_before,
            f"v2 bytes changed {label}",
        )
        protocol.require(
            _validate_model_inventory(model_payload, project_root) == model_sha,
            f"model bytes changed {label}",
        )
        protocol.require(
            _validate_runtime_registry(runtime_payload, project_root)
            == live_hardware,
            f"runtime changed {label}",
        )
        protocol.require(
            generator_dependency_closure(project_root)[1]
            == generator_closure_sha256
            and probe_media_runtime_packages(project_root)
            == media_runtime_packages,
            f"generator/media runtime changed {label}",
        )
        _validate_cost_calibration(
            cost_payload,
            model_sha=model_sha,
            runtime_sha=runtime_sha,
            render_sha=render_sha,
            live_hardware=live_hardware,
            code_registry_sha256=opening_records["eval_code_registry"]["sha256"],
            generator_sha256=code_payload["artifacts"]["generator"]["sha256"],
            generator_dependency_closure_sha256=generator_closure_sha256,
            media_runtime_packages=media_runtime_packages,
            project_root=project_root,
            calibration_run_dir=(
                project_root / protocol.DATA_ROOT / COST_CALIBRATION_RUN_DIRNAME
            ),
        )

    revalidate_deep("before binding publication")
    if preflight_only:
        return {
            "status": "preflight_valid_not_authorized",
            "opening_count": len(opening_records),
            "pending_sha256": pending_sha,
        }

    binding_payload: dict[str, Any] = {
        "protocol": SELECTION_BINDING_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "authorized_before_original_screening",
        "public_pending_sha256": pending_sha,
        "private_root_basename": private_root.name,
        "opening_artifacts": dict(opening_records),
        "secret_commitments": secret_commitments,
        "graph_contract": {
            "graph_sha256": graph_payload["graph_sha256"],
            "graph_file_sha256": opening_records["candidate_graph_576"]["sha256"],
            "graph_assignment_salt_sha256": opening_records["graph_assignment_salt"]["sha256"],
            "r1_permutation_sha256": graph_payload["r1"]["permutation_sha256"],
            "r3_permutation_sha256": graph_payload["r3"]["permutation_sha256"],
        },
        "historical_receiver_contract": {
            "inventory_sha256": historical_inventory_sha,
            "inventory_count": len(historical_inventory),
            "selected_anchor_count": 8,
        },
        "seed_contract": {
            "preselection_seed_audit_sha256": opening_records["preselection_seed_audit_1728"]["sha256"],
            "seed_count": 1728,
            "unique_seed_count": 1728,
            "screening_collision_count": 0,
            "forbidden_collision_count": 0,
            "forbidden_seed_source_audit_sha256": opening_records[
                "forbidden_seed_source_audit"
            ]["sha256"],
            "forbidden_seed_count": len(forbidden),
        },
        "registries": {
            "model_content_inventory_sha256": opening_records["model_content_inventory"]["sha256"],
            "runtime_registry_sha256": opening_records["runtime_registry"]["sha256"],
            "eval_code_registry_sha256": opening_records["eval_code_registry"]["sha256"],
            "screening_cost_calibration_sha256": opening_records["screening_cost_calibration"]["sha256"],
            "generator_dependency_closure_sha256": generator_closure_sha256,
            "media_runtime_packages": media_runtime_packages,
        },
        "capacity": {
            "model_sha256": opening_records["capacity_model_spec"]["sha256"],
            "search_sha256": opening_records["capacity_search_result_200000"]["sha256"],
            "confirmation_sha256": opening_records["capacity_confirm_result_1000000"]["sha256"],
            "static_graph_sha256": opening_records["static_graph_robustness_report"]["sha256"],
        },
        "upstream_public_sha256": dict(v2_before),
        "authorizer_sha256": protocol.sha256_file(
            project_root / protocol.CODE_ARTIFACT_PATHS["stage0_authorizer"]
        ),
    }
    binding_owned_inode: tuple[int, int] | None = None
    wrapper: dict[str, Any] | None = None
    try:
        try:
            binding_owned_inode = _write_private_binding_exclusive(
                binding_path, binding_payload
            )
        except BaseException as exc:
            if not os.path.lexists(binding_path):
                raise StaticAuthorizationFailure(
                    "selection-binding publication failed before creation"
                ) from exc
            raise
        revalidate_deep("after binding publication")
        wrapper = {
            "protocol": protocol.COMMITMENT_PROTOCOL,
            "dataset": protocol.DATASET,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "artifacts": _wrapper_artifacts(
                project_root=project_root,
                private_root=private_root,
                opening_records=opening_records,
                binding_path=binding_path,
            ),
        }
        protocol.validate_commitment_registry(wrapper, stage=0)
        state["wrapper_sha256"] = protocol.sha256_bytes(
            protocol.canonical_json_bytes(wrapper)
        )

        revalidate_deep("before wrapper publication")
        protocol.require(
            not wrapper_path.exists() and not wrapper_path.is_symlink(),
            "standard Stage-0 wrapper target appeared during authorization",
        )
        try:
            _write_public_wrapper_exclusive(wrapper_path, wrapper)  # final I/O
        except BaseException as exc:
            if _is_exact_published_json(wrapper_path, wrapper, mode=0o644):
                state["wrapper_published"] = True
                raise
            raise StaticAuthorizationFailure(
                "standard wrapper publication failed before boundary"
            ) from exc
        state["wrapper_published"] = True
        return wrapper
    except BaseException:
        if state["wrapper_published"]:
            raise
        rollback_error: Exception | None = None
        if binding_owned_inode is not None:
            try:
                _remove_owned_inode_if_present(binding_path, binding_owned_inode)
            except Exception as exc:
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError("Stage-0 publication rollback was not safe") from rollback_error
        raise


def _authorize_with_terminal_boundary(
    *,
    project_root: Path,
    private_root: Path,
    pending_path: Path,
    binding_path: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "pending_validated": False,
        "wrapper_published": False,
        "wrapper_sha256": None,
    }
    try:
        return _authorize_impl(
            project_root=project_root,
            private_root=private_root,
            pending_path=pending_path,
            binding_path=binding_path,
            wrapper_path=wrapper_path,
            state=state,
        )
    except StaticAuthorizationFailure:
        raise
    except BaseException as exc:
        if not state["pending_validated"]:
            raise
        stage0_sha256 = (
            state["wrapper_sha256"] if state["wrapper_published"] else None
        )
        try:
            _publish_authorization_invalid(
                protocol.validate_project_root(project_root),
                stage0_sha256=stage0_sha256,
            )
        except BaseException as publication_error:
            raise RuntimeError(
                "terminal Stage-0 authorization outcome publication failed"
            ) from publication_error
        raise exc


def authorize(
    *,
    project_root: Path,
    private_root: Path,
    pending_path: Path,
    binding_path: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_terminal_or_stage1(project_root)
    private_root = protocol._canonical_lexical_absolute(private_root)
    with _authorization_mutex(private_root):
        _reject_existing_terminal_or_stage1(project_root)
        return _authorize_with_terminal_boundary(
            project_root=project_root,
            private_root=private_root,
            pending_path=pending_path,
            binding_path=binding_path,
            wrapper_path=wrapper_path,
        )


def build_model_inventory_payload(project_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    model_root_relative = "models/Wan2.1-T2V-1.3B-Diffusers"
    model_root = project_root / model_root_relative
    protocol._require_no_symlink_components(model_root)
    protocol.require(model_root.is_dir(), "model root is missing")
    files: list[dict[str, Any]] = []
    for path in sorted(model_root.rglob("*")):
        info = os.lstat(path)
        protocol.require(
            not stat.S_ISLNK(info.st_mode),
            "model inventory contains symlink",
        )
        if stat.S_ISDIR(info.st_mode):
            continue
        protocol.require(
            stat.S_ISREG(info.st_mode),
            "model inventory contains non-regular entry",
        )
        protocol.require(info.st_nlink == 1, "model file is hardlinked")
        files.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": protocol.sha256_file(path),
                "size_bytes": info.st_size,
            }
        )
    protocol.require(files, "model inventory is empty")
    payload = {
        "protocol": MODEL_INVENTORY_PROTOCOL,
        "status": "frozen",
        "dataset_version": protocol.DATASET_VERSION,
        "model_root": model_root_relative,
        "file_count": len(files),
        "files": files,
        "inventory_sha256": protocol.sha256_bytes(
            protocol.canonical_json_bytes(files)
        ),
    }
    _validate_model_inventory(payload, project_root)
    return payload


def build_runtime_registry_payload(project_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    executable = project_root / "models/.wan-runtime/bin/python"
    runtime_root = project_root / "models/.wan-runtime"
    protocol._require_no_symlink_components(runtime_root)
    protocol._require_no_symlink_components(executable)
    protocol.require(
        runtime_root.is_dir() and not runtime_root.is_symlink(),
        "runtime root is invalid",
    )
    protocol._require_regular_file(executable, single_link=True)
    content_inventory = _runtime_content_inventory(project_root, runtime_root)
    source = """
import importlib, importlib.metadata, json, platform, torch
names = json.loads(__import__('sys').argv[1])
module_names = json.loads(__import__('sys').argv[2])
print(json.dumps({
  "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
  "torch": {"distribution_version": importlib.metadata.version("torch"), "module_version": torch.__version__},
  "cuda": {"available_required": torch.cuda.is_available() is True, "torch_cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(), "device_count": torch.cuda.device_count(), "device_models": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
  "packages": {name: importlib.metadata.version(name) for name in names},
  "module_origins": {name: importlib.import_module(name).__file__ for name in module_names},
}, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            os.fspath(executable),
            "-I",
            "-c",
            source,
            json.dumps(sorted(RUNTIME_PACKAGE_NAMES), separators=(",", ":")),
            json.dumps(list(RUNTIME_ORIGIN_MODULES), separators=(",", ":")),
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    protocol.require(completed.returncode == 0, "runtime registry probe failed")
    observed = json.loads(completed.stdout)
    protocol.require_exact_keys(
        observed,
        {"python", "torch", "cuda", "packages", "module_origins"},
        "runtime observation",
    )
    raw_origins = observed.pop("module_origins")
    protocol.require(
        isinstance(raw_origins, Mapping)
        and set(raw_origins) == set(RUNTIME_ORIGIN_MODULES),
        "runtime module origin inventory is incomplete",
    )
    module_origins = {
        name: _module_origin_record(
            project_root, runtime_root, raw_origins[name]
        )
        for name in RUNTIME_ORIGIN_MODULES
    }
    payload = {
        "protocol": RUNTIME_REGISTRY_PROTOCOL,
        "status": "frozen",
        "dataset_version": protocol.DATASET_VERSION,
        "runtime_root": "models/.wan-runtime",
        "python_executable": "models/.wan-runtime/bin/python",
        "sys_prefix_policy": "realpath(sys.prefix)==realpath(runtime_root)",
        **observed,
        **content_inventory,
        "module_origins": module_origins,
    }
    _validate_runtime_registry(payload, project_root)
    return payload


def build_capacity_model_payload() -> dict[str, Any]:
    with capacity.compiled_oracle() as oracle:
        self_test = capacity.compiled_oracle_self_test(oracle)
    protocol.require(self_test is True, "capacity compiled oracle self-test failed")
    return capacity._exact_artifact_base(
        status="frozen_capacity_model_spec",
        self_test=self_test,
        graph_robustness=capacity.graph_robustness_report(),
        analytic_models=capacity.analytic_capacity_report(),
    )


def prepare_static(project_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    outputs = {
        "model_content_inventory": (
            project_root / MODEL_INVENTORY_PATH,
            build_model_inventory_payload(project_root),
        ),
        "runtime_registry": (
            project_root / RUNTIME_REGISTRY_PATH,
            build_runtime_registry_payload(project_root),
        ),
        "eval_code_registry": (
            project_root / protocol.CODE_REGISTRY,
            build_code_registry_payload(project_root),
        ),
        "capacity_model_spec": (
            project_root / CAPACITY_MODEL_PATH,
            build_capacity_model_payload(),
        ),
        "static_graph_robustness_report": (
            project_root / STATIC_GRAPH_PATH,
            _json_normalized(capacity.graph_robustness_report()),
        ),
    }
    for path, _ in outputs.values():
        protocol.validate_runtime_read_path(
            project_root, path.parent / path.name, allow_v2=False
        ) if path.exists() else protocol._require_no_symlink_components(path.parent)
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite static artifact: {path}")
    owned: list[tuple[Path, tuple[int, int]]] = []
    try:
        for path, payload in outputs.values():
            _write_json_owned_exclusive(
                path, payload, mode=0o644, ownership=owned
            )
        observed: dict[str, dict[str, Any]] = {}
        for name, (path, expected) in outputs.items():
            loaded = protocol.load_json(
                path, project_root=project_root, allow_v2=False
            )
            protocol.require(
                loaded == expected
                and protocol.sha256_file(path)
                == protocol.sha256_bytes(protocol.canonical_json_bytes(expected)),
                f"static artifact readback mismatch: {name}",
            )
            observed[name] = loaded
        _validate_model_inventory(observed["model_content_inventory"], project_root)
        _validate_runtime_registry(observed["runtime_registry"], project_root)
        validate_code_registry_full(observed["eval_code_registry"], project_root)
        _validate_capacity_common(
            observed["capacity_model_spec"], "frozen_capacity_model_spec"
        )
        protocol.require(
            observed["static_graph_robustness_report"]
            == _json_normalized(capacity.graph_robustness_report()),
            "static graph report drifted during publication",
        )
        result = {
            "status": "prepared_static_not_authorized",
            "artifacts": {
                name: _file_record(path, PHYSICAL_ROW_COUNTS[name])
                for name, (path, _) in outputs.items()
            },
        }
    except BaseException:
        _rollback_owned_outputs(owned)
        raise
    return result


def build_selection_rules_payload() -> dict[str, Any]:
    return {
        "protocol": RULES_PROTOCOL,
        "dataset_version": protocol.DATASET_VERSION,
        "status": "frozen",
        "qualification": dict(QUALIFICATION),
        "cell_quota": {
            "per_group_prompt_variant": 4,
            "selected_per_group": 8,
            "selected_total": 24,
        },
        "graph_permutation_domain": protocol.GRAPH_ASSIGNMENT_DOMAIN,
        "graph_permutation_formula": GRAPH_FORMULA,
        "ranking_domain": protocol.RANK_DOMAIN,
        "ranking_formula": RANK_FORMULA,
        "subset_algorithm": {
            "algorithm": "rank_order_greedy_include_if_exact_completion_exists",
            "groups": builder.graph_topology(),
            "rank_tie_policy": "invalidate_data_version",
        },
        "evaluation_seed_domain": protocol.SEED_DOMAIN,
        "evaluation_seed_formula": SEED_FORMULA,
        "replicates": list(protocol.REPLICATES),
        "required_selected_cases": 24,
        "required_evaluation_units": 72,
    }


def audit_historical_secrets(
    project_root: Path,
    private_root: Path,
    secret_audit_root: Path,
    clean_projection_root: Path,
    authorizer_projection_root: Path,
) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    private_root = _require_disjoint_project_private_roots(
        project_root, private_root
    )
    secret_audit_root = _require_distinct_mode700_root(
        secret_audit_root, other_roots=(project_root, private_root)
    )
    clean_projection_root = _require_historical_projection_root(
        clean_projection_root,
        expected_names=set(HISTORICAL_RAW_SOURCE_FILES["clean"]),
        other_roots=(project_root, private_root, secret_audit_root),
    )
    authorizer_projection_root = _require_historical_projection_root(
        authorizer_projection_root,
        expected_names=set(HISTORICAL_RAW_SOURCE_FILES["authorizer"]),
        other_roots=(
            project_root,
            private_root,
            secret_audit_root,
            clean_projection_root,
        ),
    )
    retained_raw_names = {
        PRIVATE_INPUTS["screening_seed"],
        PRIVATE_INPUTS["graph_assignment_salt"],
        PRIVATE_INPUTS["selector_salt"],
        PRIVATE_INPUTS["evaluation_seed_salt"],
    }
    with _multi_root_mutex(
        private_root,
        secret_audit_root,
        clean_projection_root,
        authorizer_projection_root,
    ):
        _reject_existing_preparation_boundary(project_root)
        protocol.require(
            {entry.name for entry in private_root.iterdir()}
            == PRIVATE_EXTERNAL_INPUT_NAMES | retained_raw_names,
            "historical audit requires the exact post-sampling private inventory",
        )
        protocol.require(
            {entry.name for entry in secret_audit_root.iterdir()}
            == {SECRET_SAMPLING_REQUEST_BASENAME},
            "historical audit requires exactly the retained sampling request",
        )
        for name in retained_raw_names:
            protocol.validate_private_path(private_root, private_root / name)
        request_path = secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME
        protocol.validate_private_path(secret_audit_root, request_path)
        request = protocol.load_json(
            request_path, private_root=secret_audit_root
        )
        provenance = request.get("sampling_provenance", {})
        validate_secret_sampling_request(
            request,
            forbidden_seed_inventory_sha256=provenance.get(
                "forbidden_seed_inventory_sha256", ""
            ),
            forbidden_numeric_seed_count=provenance.get(
                "forbidden_numeric_seed_count", -1
            ),
        )
        graph_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["graph_assignment_salt"]
        )
        selector_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["selector_salt"]
        )
        evaluation_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
        )
        assert isinstance(graph_salt, str)
        assert isinstance(selector_salt, str)
        assert isinstance(evaluation_salt, str)
        protocol.require(
            len({graph_salt, selector_salt, evaluation_salt}) == 3,
            "sampled v3 salts are not pairwise distinct",
        )
        new_commitments = {
            "graph_assignment_salt": _secret_commitment(
                "causal_graph_assignment_salt_v3", graph_salt
            ),
            "selector_salt": _secret_commitment(
                "causal_stage0_selector_salt_v3", selector_salt
            ),
            "evaluation_seed_salt": _secret_commitment(
                "causal_evaluation_seed_salt_v3", evaluation_salt
            ),
        }
        protocol.require(
            all(
                provenance["new_secret_commitments"][name] == commitment
                for name, commitment in new_commitments.items()
            ),
            "historical auditor raw salts do not match the retained request",
        )
        raw_allowlist = _load_historical_raw_projection(
            clean_projection_root, "clean"
        ) | _load_historical_raw_projection(
            authorizer_projection_root, "authorizer"
        )
        protocol.require(
            len(raw_allowlist) == HISTORICAL_ACCESSIBLE_RAW_COUNT
            and protocol.sha256_bytes(
                protocol.canonical_json_bytes(sorted(raw_allowlist))
            )
            == HISTORICAL_ACCESSIBLE_RAW_ALLOWLIST_SHA256,
            "accessible historical raw allowlist digest/count changed",
        )
        intersections = len(
            {graph_salt, selector_salt, evaluation_salt} & raw_allowlist
        )
        protocol.require(
            intersections == 0,
            "new v3 salt intersects accessible historical raw secrets",
        )
        commitment_only = _load_commitment_only_historical_allowlist(
            project_root
        )
        commitment_comparisons = 3 * len(commitment_only)
        audit = {
            "protocol": HISTORICAL_SECRET_AUDIT_PROTOCOL,
            "status": "passed",
            "v3_hex_salt_count": 3,
            "new_salt_commitments": new_commitments,
            "accessible_historical_raw_allowlist_sha256": (
                HISTORICAL_ACCESSIBLE_RAW_ALLOWLIST_SHA256
            ),
            "accessible_historical_raw_hex_secret_count": len(raw_allowlist),
            "accessible_historical_raw_comparison_count": 3 * len(raw_allowlist),
            "accessible_historical_raw_intersection_count": intersections,
            "commitment_only_historical_allowlist_sha256": (
                HISTORICAL_COMMITMENT_ALLOWLIST_SHA256
            ),
            "commitment_only_historical_hex_secret_count": len(commitment_only),
            "commitment_only_comparison_count": commitment_comparisons,
            "commitment_only_collision_union_bound_numerator": (
                commitment_comparisons
            ),
            "commitment_only_collision_union_bound_denominator_power": 256,
            "forbidden_seed_inventory_sha256": provenance[
                "forbidden_seed_inventory_sha256"
            ],
            "forbidden_numeric_seed_count": provenance[
                "forbidden_numeric_seed_count"
            ],
            "screening_seed_forbidden_intersection_count": provenance[
                "screening_seed_forbidden_intersection_count"
            ],
            "raw_historical_secret_values_emitted": False,
        }
        validate_historical_secret_audit(
            audit,
            forbidden_seed_inventory_sha256=provenance[
                "forbidden_seed_inventory_sha256"
            ],
            forbidden_numeric_seed_count=provenance[
                "forbidden_numeric_seed_count"
            ],
            new_salt_commitments=new_commitments,
        )
        output = secret_audit_root / HISTORICAL_SECRET_AUDIT_BASENAME
        protocol.validate_private_output_path(secret_audit_root, output)
        owned: list[tuple[Path, tuple[int, int]]] = []
        try:
            _write_json_owned_exclusive(
                output, audit, mode=0o600, ownership=owned
            )
            observed = protocol.load_json(
                output, private_root=secret_audit_root
            )
            protocol.require(
                observed == audit
                and {entry.name for entry in secret_audit_root.iterdir()}
                == {
                    SECRET_SAMPLING_REQUEST_BASENAME,
                    HISTORICAL_SECRET_AUDIT_BASENAME,
                },
                "historical audit publication readback/inventory mismatch",
            )
            validate_historical_secret_audit(
                observed,
                forbidden_seed_inventory_sha256=provenance[
                    "forbidden_seed_inventory_sha256"
                ],
                forbidden_numeric_seed_count=provenance[
                    "forbidden_numeric_seed_count"
                ],
                new_salt_commitments=new_commitments,
            )
            protocol.require(
                _load_historical_raw_projection(clean_projection_root, "clean")
                | _load_historical_raw_projection(
                    authorizer_projection_root, "authorizer"
                )
                == raw_allowlist
                and _load_commitment_only_historical_allowlist(project_root)
                == commitment_only
                and protocol.sha256_file(request_path)
                == protocol.sha256_bytes(protocol.canonical_json_bytes(request))
                and _read_text_secret(
                    private_root / PRIVATE_INPUTS["graph_assignment_salt"]
                )
                == graph_salt
                and _read_text_secret(
                    private_root / PRIVATE_INPUTS["selector_salt"]
                )
                == selector_salt
                and _read_text_secret(
                    private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
                )
                == evaluation_salt,
                "historical audit sources/request changed during publication",
            )
            result = dict(audit)
        except BaseException:
            _rollback_owned_outputs(owned)
            raise
        return result


def sample_secrets(
    project_root: Path,
    private_root: Path,
    secret_audit_root: Path,
) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    private_root = _require_disjoint_project_private_roots(
        project_root, private_root
    )
    secret_audit_root = _require_distinct_mode700_root(
        secret_audit_root,
        other_roots=(project_root, private_root),
    )
    with _multi_root_mutex(private_root, secret_audit_root):
        _reject_existing_preparation_boundary(project_root)
        protocol.require(
            {entry.name for entry in private_root.iterdir()}
            == PRIVATE_EXTERNAL_INPUT_NAMES,
            "sample-secrets requires exactly the seven frozen external inputs",
        )
        protocol.require(
            not any(secret_audit_root.iterdir()),
            "secret audit root must be empty before sampling",
        )
        for name in PRIVATE_EXTERNAL_INPUT_NAMES:
            protocol.validate_private_path(private_root, private_root / name)
        forbidden_path = private_root / PRIVATE_INPUTS["forbidden_seed_inventory"]
        forbidden_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["forbidden_seed_inventory"]
        )
        forbidden = selector.validate_forbidden_seed_inventory(forbidden_payload)
        protocol.require(
            set(CALIBRATION_SEEDS) <= forbidden,
            "v3 forbidden inventory omits fixed calibration seeds",
        )

        salts: dict[str, str] = {}
        salt_attempts: dict[str, int] = {}
        for name in (
            "graph_assignment_salt",
            "selector_salt",
            "evaluation_seed_salt",
        ):
            attempts = 0
            while True:
                attempts += 1
                draw = pysecrets.token_bytes(32)
                protocol.require(
                    type(draw) is bytes and len(draw) == 32,
                    "OS CSPRNG salt draw is not exactly 32 bytes",
                )
                value = draw.hex()
                if value not in salts.values():
                    salts[name] = value
                    salt_attempts[name] = attempts
                    break
        screening_attempts = 0
        while True:
            screening_attempts += 1
            draw = pysecrets.token_bytes(4)
            protocol.require(
                type(draw) is bytes and len(draw) == 4,
                "OS CSPRNG screening draw is not exactly four bytes",
            )
            screening_seed = int.from_bytes(draw, "big")
            if screening_seed not in forbidden:
                break
        graph_salt = salts["graph_assignment_salt"]
        selector_salt = salts["selector_salt"]
        evaluation_salt = salts["evaluation_seed_salt"]
        protocol.validate_secret_separation(
            graph_assignment_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
            screening_seed=screening_seed,
        )
        commitments = _secret_commitments(
            screening_seed=screening_seed,
            graph_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
        )
        sampling_provenance = {
            "entropy_source": "operating_system_csprng",
            "independent_draws": True,
            "salt_draw_count": 3,
            "salt_bytes_per_draw": 32,
            "salt_encoding": "lower_hex64",
            "salt_draw_attempts": salt_attempts,
            "screening_seed_draw_count": 1,
            "screening_seed_bytes_per_draw": 4,
            "screening_seed_byte_order": "big_endian",
            "screening_seed_encoding": "canonical_unsigned_decimal_uint32",
            "screening_seed_draw_attempts": screening_attempts,
            "new_secret_commitments": commitments,
            "forbidden_seed_inventory_sha256": protocol.sha256_file(
                forbidden_path
            ),
            "forbidden_numeric_seed_count": len(forbidden),
            "screening_seed_forbidden_intersection_count": 0,
        }
        request = {
            "protocol": SECRET_SAMPLING_REQUEST_PROTOCOL,
            "status": "sampled_pending_historical_audit",
            "dataset_version": protocol.DATASET_VERSION,
            "sampling_provenance": sampling_provenance,
            "raw_secret_values_emitted": False,
        }
        validate_secret_sampling_request(
            request,
            forbidden_seed_inventory_sha256=protocol.sha256_file(forbidden_path),
            forbidden_numeric_seed_count=len(forbidden),
            screening_seed=screening_seed,
            graph_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
        )
        targets: list[tuple[Path, bytes]] = [
            (
                private_root / PRIVATE_INPUTS["graph_assignment_salt"],
                f"{graph_salt}\n".encode("ascii"),
            ),
            (
                private_root / PRIVATE_INPUTS["selector_salt"],
                f"{selector_salt}\n".encode("ascii"),
            ),
            (
                private_root / PRIVATE_INPUTS["evaluation_seed_salt"],
                f"{evaluation_salt}\n".encode("ascii"),
            ),
            (
                private_root / PRIVATE_INPUTS["screening_seed"],
                f"{screening_seed}\n".encode("ascii"),
            ),
            (
                secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME,
                protocol.canonical_json_bytes(request),
            ),
        ]
        for path, _ in targets:
            protocol.validate_private_output_path(
                private_root if path.parent == private_root else secret_audit_root,
                path,
            )
        owned: list[tuple[Path, tuple[int, int]]] = []
        try:
            for path, raw in targets:
                _write_bytes_owned_exclusive(
                    path, raw, mode=0o600, ownership=owned
                )
            observed_screening = _read_text_secret(
                private_root / PRIVATE_INPUTS["screening_seed"], integer=True
            )
            observed_graph = _read_text_secret(
                private_root / PRIVATE_INPUTS["graph_assignment_salt"]
            )
            observed_selector = _read_text_secret(
                private_root / PRIVATE_INPUTS["selector_salt"]
            )
            observed_evaluation = _read_text_secret(
                private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
            )
            observed_request = protocol.load_json(
                secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME,
                private_root=secret_audit_root,
            )
            validate_secret_sampling_request(
                observed_request,
                forbidden_seed_inventory_sha256=protocol.sha256_file(
                    forbidden_path
                ),
                forbidden_numeric_seed_count=len(forbidden),
                screening_seed=observed_screening,
                graph_salt=observed_graph,
                selector_salt=observed_selector,
                evaluation_salt=observed_evaluation,
            )
            protocol.require(
                observed_request == request
                and {entry.name for entry in private_root.iterdir()}
                == PRIVATE_EXTERNAL_INPUT_NAMES
                | {
                    PRIVATE_INPUTS["screening_seed"],
                    PRIVATE_INPUTS["graph_assignment_salt"],
                    PRIVATE_INPUTS["selector_salt"],
                    PRIVATE_INPUTS["evaluation_seed_salt"],
                }
                and {entry.name for entry in secret_audit_root.iterdir()}
                == {SECRET_SAMPLING_REQUEST_BASENAME},
                "secret sampling transaction inventory/readback mismatch",
            )
            _fsync_directory(private_root)
            _fsync_directory(secret_audit_root)
            request_sha = protocol.sha256_file(
                secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME
            )
            result = {
                "protocol": SECRET_SAMPLING_AGGREGATE_PROTOCOL,
                "status": "sampled_pending_historical_audit",
                "dataset_version": protocol.DATASET_VERSION,
                "sampling_request_sha256": request_sha,
                "salt_draw_attempts": dict(salt_attempts),
                "screening_seed_draw_attempts": screening_attempts,
                "salt_draw_count": 3,
                "screening_seed_draw_count": 1,
                "raw_secret_values_emitted": False,
            }
            protocol.require_exact_keys(
                result,
                {
                    "protocol",
                    "status",
                    "dataset_version",
                    "sampling_request_sha256",
                    "salt_draw_attempts",
                    "screening_seed_draw_attempts",
                    "salt_draw_count",
                    "screening_seed_draw_count",
                    "raw_secret_values_emitted",
                },
                "secret sampling public aggregate",
            )
        except BaseException:
            _rollback_owned_outputs(owned)
            raise
        return result


def prepare_private(
    project_root: Path,
    private_root: Path,
    secret_audit_root: Path,
) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    private_root = _require_disjoint_project_private_roots(
        project_root, private_root
    )
    secret_audit_root = _require_distinct_mode700_root(
        secret_audit_root,
        other_roots=(project_root, private_root),
    )
    retained_raw_names = {
        PRIVATE_INPUTS["screening_seed"],
        PRIVATE_INPUTS["graph_assignment_salt"],
        PRIVATE_INPUTS["selector_salt"],
        PRIVATE_INPUTS["evaluation_seed_salt"],
    }
    with _multi_root_mutex(private_root, secret_audit_root):
        _reject_existing_preparation_boundary(project_root)
        for target in (
            project_root / protocol.STAGE0_PUBLIC,
            project_root / protocol.STAGE0_REGISTRY,
            private_root / "causal_selection_binding_v3.json",
            private_root / "causal_original_screening_generation_v3",
            private_root / "causal_original_screening_review_public_v3",
            private_root / "causal_original_screening_review_private_v3",
        ):
            if os.path.lexists(target):
                raise FileExistsError(
                    f"prepare-private boundary already exists: {target}"
                )
        v2_before = protocol.validate_v2_public_inputs(project_root)
        observed = {entry.name for entry in private_root.iterdir()}
        protocol.require(
            observed == PRIVATE_EXTERNAL_INPUT_NAMES | retained_raw_names,
            "prepare-private requires the seven inputs plus four sampled secrets",
        )
        for name in PRIVATE_EXTERNAL_INPUT_NAMES | retained_raw_names:
            protocol.validate_private_path(private_root, private_root / name)

        source_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["eval_holdout_source_ontology_48"]
        )
        sources = builder.validate_holdout_ontology(source_payload)
        holdout_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["holdout_registry_48"]
        )
        _validate_holdout_registry(holdout_payload, sources)
        receiver_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["receiver_ontology_56"]
        )
        builder.validate_receiver_ontology(receiver_payload)
        historical_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["historical_receiver_anchors_8"]
        )
        builder.validate_historical_anchors(historical_payload)
        for name in (
            "eval_holdout_source_ontology_48",
            "holdout_registry_48",
            "receiver_ontology_56",
            "historical_receiver_anchors_8",
        ):
            protocol.require(
                _physical_record(
                    private_root / PRIVATE_INPUTS[name],
                    PHYSICAL_ROW_COUNTS[name],
                )
                == HOLDOUT_FROZEN_RECORDS[name],
                f"r6 frozen byte record mismatch: {name}",
            )
        builder.validate_templates_and_fields(
            private_root / PRIVATE_INPUTS["canonical_templates"],
            private_root / PRIVATE_INPUTS["field_normalization"],
            private_root=private_root,
        )
        forbidden_payload = _load_private_json(
            private_root, PRIVATE_INPUTS["forbidden_seed_inventory"]
        )
        forbidden = selector.validate_forbidden_seed_inventory(forbidden_payload)
        protocol.require(
            set(CALIBRATION_SEEDS) <= forbidden,
            "v3 forbidden inventory omits fixed calibration seeds",
        )
        (
            _,
            sampling_request,
            historical_secret_audit,
            sampling_request_sha256,
        ) = _load_complete_secret_audit_root(
            project_root, private_root, secret_audit_root
        )
        historical_audit_sha256 = protocol.sha256_file(
            secret_audit_root / HISTORICAL_SECRET_AUDIT_BASENAME
        )
        graph_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["graph_assignment_salt"]
        )
        selector_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["selector_salt"]
        )
        evaluation_salt = _read_text_secret(
            private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
        )
        screening_seed = _read_text_secret(
            private_root / PRIVATE_INPUTS["screening_seed"], integer=True
        )
        assert isinstance(graph_salt, str) and isinstance(selector_salt, str)
        assert isinstance(evaluation_salt, str) and isinstance(screening_seed, int)
        validate_secret_sampling_request(
            sampling_request,
            forbidden_seed_inventory_sha256=protocol.sha256_file(
                private_root / PRIVATE_INPUTS["forbidden_seed_inventory"]
            ),
            forbidden_numeric_seed_count=len(forbidden),
            screening_seed=screening_seed,
            graph_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
        )
        protocol.require(
            screening_seed not in forbidden,
            "retained screening seed collides with forbidden inventory",
        )
        salt_commitments = {
            name: sampling_request["sampling_provenance"][
                "new_secret_commitments"
            ][name]
            for name in (
                "graph_assignment_salt",
                "selector_salt",
                "evaluation_seed_salt",
            )
        }
        validate_historical_secret_audit(
            historical_secret_audit,
            forbidden_seed_inventory_sha256=protocol.sha256_file(
                private_root / PRIVATE_INPUTS["forbidden_seed_inventory"]
            ),
            forbidden_numeric_seed_count=len(forbidden),
            new_salt_commitments=salt_commitments,
        )
        mapping_payload = protocol.load_json(
            project_root / protocol.V2_MAPPING,
            project_root=project_root,
            allow_v2=True,
        )
        _historical_receiver_inventory(mapping_payload, historical_payload)
        bank_payload = protocol.load_json(
            project_root / protocol.V2_BANK,
            project_root=project_root,
            allow_v2=True,
        )

        protocol.validate_secret_separation(
            graph_assignment_salt=graph_salt,
            selector_salt=selector_salt,
            evaluation_salt=evaluation_salt,
            screening_seed=screening_seed,
        )
        graph, manifest = builder.build_candidate_graph(
            holdout_payload=source_payload,
            receiver_payload=receiver_payload,
            historical_payload=historical_payload,
            source_bank_payload=bank_payload,
            graph_assignment_salt=graph_salt,
        )
        records = [
            {
                "case_id": candidate["case_id"],
                "replicate": replicate,
                "seed": protocol.derive_evaluation_seed(
                    evaluation_salt, candidate["case_id"], replicate
                ),
            }
            for candidate in manifest["candidates"]
            for replicate in protocol.REPLICATES
        ]
        derived = [record["seed"] for record in records]
        protocol.require(
            len(derived) == 1728
            and len(set(derived)) == 1728
            and screening_seed not in derived
            and not (set(derived) & forbidden),
            "sampled evaluation-seed inventory collides",
        )

        model_payload = _load_public_json(project_root, MODEL_INVENTORY_PATH)
        model_sha = _validate_model_inventory(model_payload, project_root)
        runtime_payload = _load_public_json(project_root, RUNTIME_REGISTRY_PATH)
        _validate_runtime_registry(runtime_payload, project_root)
        runtime_sha = protocol.sha256_file(project_root / RUNTIME_REGISTRY_PATH)
        code_payload = _load_public_json(project_root, protocol.CODE_REGISTRY)
        validate_code_registry_full(code_payload, project_root)
        protocol.require(
            {
                "name": "v3_screening_cost_calibration_seeds",
                "sha256": code_payload["artifacts"]["screening_runner"]["sha256"],
                "seed_count": 5,
            }
            in forbidden_payload["source_commitments"],
            "forbidden inventory lacks code-bound calibration seed source",
        )
        _, closure_sha = generator_dependency_closure(project_root)
        media_packages = probe_media_runtime_packages(project_root)

        output_paths = {
            name: private_root / basename
            for name, basename in PRIVATE_INPUTS.items()
            if name not in {
                "eval_holdout_source_ontology_48",
                "holdout_registry_48",
                "receiver_ontology_56",
                "historical_receiver_anchors_8",
                "canonical_templates",
                "field_normalization",
                "forbidden_seed_inventory",
                "raw_root_bundle",
                "screening_seed",
                "graph_assignment_salt",
                "selector_salt",
                "evaluation_seed_salt",
            }
        }
        bundle_path = private_root / PRIVATE_INPUTS["raw_root_bundle"]
        for path in (*output_paths.values(), bundle_path):
            protocol.validate_private_output_path(private_root, path)

        owned: list[tuple[Path, tuple[int, int]]] = []
        try:
            render = {
                "protocol": RENDER_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "frozen",
                "arm": "Original_only",
                "model_family": "Wan 2.1 T2V 1.3B",
                "model_content_inventory_sha256": model_sha,
                "steps": 25,
                "cfg": 5,
                "frames": 49,
                "width": 832,
                "height": 480,
                "fps": 8,
                "dtype": "bf16",
                "adapter": None,
                "screening_scope": "all 49 frames for every candidate",
            }
            rules = build_selection_rules_payload()
            secrets_payload = {
                "protocol": SECRETS_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "frozen",
                "screening_seed_namespace": protocol.SCREENING_NAMESPACE,
                "screening_seed": screening_seed,
                "graph_assignment_salt": graph_salt,
                "selector_salt": selector_salt,
                "evaluation_seed_namespace": protocol.EVALUATION_NAMESPACE,
                "evaluation_seed_salt": evaluation_salt,
                "sampling_request_sha256": sampling_request_sha256,
                "sampling_provenance": sampling_request["sampling_provenance"],
                "historical_secret_audit": historical_secret_audit,
            }
            for name, payload in (
                ("raw_render_configuration", render),
                ("selection_rules", rules),
                ("stage0_secrets", secrets_payload),
            ):
                path = output_paths[name]
                _write_json_owned_exclusive(
                    path, payload, mode=0o600, ownership=owned
                )

            graph_path = output_paths["candidate_graph_576"]
            manifest_path = output_paths["candidate_manifest_576"]
            builder._write_graph_manifest_transaction(
                graph_path,
                graph,
                manifest_path,
                manifest,
                post_link_check=lambda: builder._require_v2_hashes_unchanged(
                    project_root, v2_before
                ),
                ownership_sink=owned,
            )

            seed_audit = {
                "protocol": SEED_AUDIT_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "passed",
                "candidate_manifest_sha256": protocol.sha256_file(manifest_path),
                "evaluation_seed_salt_sha256": protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
                ),
                "screening_seed_sha256": protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["screening_seed"]
                ),
                "forbidden_seed_inventory_sha256": protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["forbidden_seed_inventory"]
                ),
                "seed_count": 1728,
                "unique_seed_count": 1728,
                "screening_collision_count": 0,
                "forbidden_collision_count": 0,
                "ordered_seed_records_sha256": protocol.sha256_bytes(
                    protocol.canonical_json_bytes(records)
                ),
                "records": records,
            }
            generation_spec = {
                "protocol": GENERATION_SPEC_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "frozen_before_original_screening",
                "candidate_manifest_sha256": protocol.sha256_file(manifest_path),
                "candidate_graph_sha256": protocol.sha256_file(graph_path),
                "render_configuration_sha256": protocol.sha256_file(
                    output_paths["raw_render_configuration"]
                ),
                "screening_seed_sha256": protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["screening_seed"]
                ),
                "graph_assignment_salt_sha256": protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["graph_assignment_salt"]
                ),
                "model_content_inventory_sha256": model_sha,
                "runtime_registry_sha256": runtime_sha,
                "generator_dependency_closure_sha256": closure_sha,
                "media_runtime_packages": media_packages,
                "candidate_count": 576,
                "generation": {
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
                },
            }
            for name, payload in (
                ("preselection_seed_audit_1728", seed_audit),
                ("screening_generation_spec", generation_spec),
            ):
                path = output_paths[name]
                _write_json_owned_exclusive(
                    path, payload, mode=0o600, ownership=owned
                )

            private_records = {
                name: _physical_record(
                    private_root / basename, PHYSICAL_ROW_COUNTS[name]
                )
                for name, basename in PRIVATE_INPUTS.items()
                if name != "raw_root_bundle"
            }
            bundle = {
                "protocol": BUNDLE_PROTOCOL,
                "dataset_version": protocol.DATASET_VERSION,
                "status": "frozen_before_pending_commitment",
                "components": {
                    name: record["sha256"] for name, record in private_records.items()
                },
            }
            _write_json_owned_exclusive(
                bundle_path, bundle, mode=0o600, ownership=owned
            )
            _validate_render(render, model_sha)
            _validate_rules(rules)
            _validate_secrets(
                secrets_payload,
                graph_salt=graph_salt,
                selector_salt=selector_salt,
                evaluation_salt=evaluation_salt,
                screening_seed=screening_seed,
                sampling_request=sampling_request,
                sampling_request_sha256=sampling_request_sha256,
            )
            _validate_seed_audit(
                seed_audit,
                candidates=manifest["candidates"],
                evaluation_salt=evaluation_salt,
                screening_seed=screening_seed,
                forbidden=forbidden,
                candidate_sha=protocol.sha256_file(manifest_path),
                evaluation_salt_sha=protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["evaluation_seed_salt"]
                ),
                screening_seed_sha=protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["screening_seed"]
                ),
                forbidden_sha=protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["forbidden_seed_inventory"]
                ),
            )
            _validate_generation_spec(
                generation_spec,
                candidate_sha=protocol.sha256_file(manifest_path),
                graph_sha=protocol.sha256_file(graph_path),
                render_sha=protocol.sha256_file(
                    output_paths["raw_render_configuration"]
                ),
                screening_seed_sha=protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["screening_seed"]
                ),
                graph_salt_sha=protocol.sha256_file(
                    private_root / PRIVATE_INPUTS["graph_assignment_salt"]
                ),
                model_sha=model_sha,
                runtime_sha=runtime_sha,
                generator_dependency_closure_sha256=closure_sha,
                media_runtime_packages=media_packages,
            )
            _validate_bundle(
                bundle,
                private_records={
                    name: _physical_record(
                        private_root / basename, PHYSICAL_ROW_COUNTS[name]
                    )
                    for name, basename in PRIVATE_INPUTS.items()
                },
            )
            builder._require_v2_hashes_unchanged(project_root, v2_before)
            protocol.require(
                {
                    entry.name for entry in secret_audit_root.iterdir()
                }
                == {
                    SECRET_SAMPLING_REQUEST_BASENAME,
                    HISTORICAL_SECRET_AUDIT_BASENAME,
                }
                and protocol.sha256_file(
                    secret_audit_root / SECRET_SAMPLING_REQUEST_BASENAME
                )
                == sampling_request_sha256
                and protocol.sha256_file(
                    secret_audit_root / HISTORICAL_SECRET_AUDIT_BASENAME
                )
                == historical_audit_sha256,
                "secret request/audit root changed during private preparation",
            )
            result = {
                "status": "prepared_private_not_authorized",
                "created_artifact_count": len(owned),
                "candidate_count": 576,
                "seed_audit_count": 1728,
                "private_inventory_sha256": protocol.sha256_bytes(
                    protocol.canonical_json_bytes(
                        {
                            entry.name: protocol.sha256_file(entry)
                            for entry in sorted(private_root.iterdir())
                        }
                    )
                ),
            }
        except BaseException:
            _rollback_owned_outputs(owned)
            raise
        return result


def _validate_curation_public_aggregate(payload: Mapping[str, Any]) -> None:
    protocol.require_exact_keys(
        payload,
        {
            "artifact_commitments",
            "blockers",
            "counts",
            "dataset_version",
            "preservation_aggregate",
            "protocol",
            "quality_aggregate",
            "rejected_intermediate",
            "rejected_predecessor",
            "rejected_r5",
            "rejected_validator_intermediate",
            "status",
        },
        "r6 curation public aggregate",
    )
    expected_commitments = {
        HOLDOUT_EVIDENCE_BASENAMES[name]: HOLDOUT_FROZEN_RECORDS[name]
        for name in (
            "eval_holdout_source_ontology_48",
            "holdout_registry_48",
            "receiver_ontology_56",
            "historical_receiver_anchors_8",
            "curation_semantic_audit",
        )
    }
    protocol.require(
        payload["protocol"] == CURATION_PUBLIC_AGGREGATE_PROTOCOL
        and payload["status"] == protocol.R6_PUBLIC_AGGREGATE_STATUS
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["counts"]
        == {"sources": 48, "new_receivers": 56, "historical_anchors": 8}
        and payload["artifact_commitments"] == expected_commitments
        and payload["blockers"]
        == [protocol.R6_IDENTITY_AUDIT_BLOCKER]
        and payload["preservation_aggregate"]
        == {
            "historical_phrase_mutations": 0,
            "quantitative_source_physical_changes": 0,
            "receiver_record_ids_preserved": 56,
            "receiver_surface_mutations": 46,
            "source_record_ids_preserved": 48,
            "source_state_semantic_repairs": 1,
            "source_surface_mutations": 19,
        }
        and payload["quality_aggregate"]
        == {
            "historical_phrase_mutations": 0,
            "material_core_token_failures": 0,
            "protocol_lineage_mismatch_count": 0,
            "receiver_at_most_25_tokens": 56,
            "receiver_head_whole_span_once": 56,
            "receiver_natural_grammar_pass": 56,
            "receiver_note_pairs_ge_0p85": 0,
            "receiver_note_similarity_max_x10000": 7232,
            "receiver_semantic_suitability_concerns": 0,
            "receiver_static_open_bounded_water": 56,
            "receiver_surface_pairs_ge_0p85": 0,
            "receiver_surface_similarity_max_x10000": 8496,
            "source_material_redundancy_failures": 0,
            "source_natural_grammar_pass": 48,
            "source_note_pairs_ge_0p85": 0,
            "source_note_similarity_max_x10000": 6347,
            "source_state_semantic_concerns": 0,
            "source_surface_pairs_ge_0p85": 0,
            "source_surface_similarity_max_x10000": 8046,
            "strict_source_physical_pass": 48,
        },
        "r6 aggregate quality/preservation contract failed",
    )
    for name in (
        "rejected_intermediate",
        "rejected_predecessor",
        "rejected_r5",
        "rejected_validator_intermediate",
    ):
        rejected = payload[name]
        protocol.require(
            isinstance(rejected, Mapping)
            and rejected.get("status") == "REJECTED_NO_FALLBACK",
            f"{name} is not terminally rejected",
        )
        for key, value in rejected.items():
            if "sha256" in key or "digest" in key:
                protocol.require(
                    protocol.is_hex64(value),
                    f"{name} contains malformed frozen hash",
                )
    protocol.require(
        not protocol.contains_placeholder(payload),
        "r6 aggregate contains placeholder",
    )


def validate_holdout_public_commitment(
    payload: Mapping[str, Any],
    *,
    identity_report_sha256: str,
    private_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "counts",
            "artifacts",
            "identity_disjointness_report_sha256",
            "independent_language_review_status",
            "remaining_blockers",
        },
        "holdout public commitment",
    )
    protocol.require(
        payload["protocol"] == HOLDOUT_PUBLIC_PROTOCOL
        and payload["status"] == "committed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["counts"]
        == {
            "source_count": 48,
            "receiver_count": 56,
            "historical_anchor_count": 8,
        }
        and payload["artifacts"] == HOLDOUT_FROZEN_RECORDS
        and payload["identity_disjointness_report_sha256"]
        == identity_report_sha256
        and payload["independent_language_review_status"] == "passed"
        and payload["remaining_blockers"] == [],
        "holdout public commitment differs from frozen r6 records",
    )
    if private_records is not None:
        for name in (
            "eval_holdout_source_ontology_48",
            "holdout_registry_48",
            "receiver_ontology_56",
            "historical_receiver_anchors_8",
        ):
            protocol.require(
                private_records[name] == HOLDOUT_FROZEN_RECORDS[name],
                f"holdout/private opening mismatch: {name}",
            )
    protocol.require(
        not protocol.contains_placeholder(payload),
        "holdout public commitment contains placeholder",
    )
    return payload


def prepare_holdout(project_root: Path, evidence_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    evidence_root = _require_distinct_mode700_root(
        evidence_root, other_roots=(project_root,)
    )
    with _multi_root_mutex(evidence_root):
        _reject_existing_preparation_boundary(project_root)
        protocol.require(
            {entry.name for entry in evidence_root.iterdir()}
            == set(HOLDOUT_EVIDENCE_BASENAMES.values()),
            "r6 evidence root must contain exactly the nine frozen files",
        )
        evidence_records: dict[str, dict[str, Any]] = {}
        for name, basename in HOLDOUT_EVIDENCE_BASENAMES.items():
            path = evidence_root / basename
            protocol.validate_private_path(evidence_root, path)
            evidence_records[name] = _physical_record(
                path, HOLDOUT_FROZEN_RECORDS[name]["row_count"]
            )
        protocol.require(
            evidence_records == HOLDOUT_FROZEN_RECORDS,
            "r6 evidence bytes differ from the frozen nine-file inventory",
        )
        source_payload = protocol.load_json(
            evidence_root
            / HOLDOUT_EVIDENCE_BASENAMES[
                "eval_holdout_source_ontology_48"
            ],
            private_root=evidence_root,
        )
        sources = builder.validate_holdout_ontology(source_payload)
        holdout_payload = protocol.load_json(
            evidence_root / HOLDOUT_EVIDENCE_BASENAMES["holdout_registry_48"],
            private_root=evidence_root,
        )
        _validate_holdout_registry(holdout_payload, sources)
        receiver_payload = protocol.load_json(
            evidence_root / HOLDOUT_EVIDENCE_BASENAMES["receiver_ontology_56"],
            private_root=evidence_root,
        )
        builder.validate_receiver_ontology(receiver_payload)
        historical_payload = protocol.load_json(
            evidence_root
            / HOLDOUT_EVIDENCE_BASENAMES["historical_receiver_anchors_8"],
            private_root=evidence_root,
        )
        builder.validate_historical_anchors(historical_payload)
        aggregate = protocol.load_json(
            evidence_root
            / HOLDOUT_EVIDENCE_BASENAMES["curation_public_aggregate"],
            private_root=evidence_root,
        )
        _validate_curation_public_aggregate(aggregate)
        identity_path = project_root / protocol.IDENTITY_REPORT
        identity = _load_public_json(project_root, protocol.IDENTITY_REPORT)
        protocol.validate_identity_disjointness_report(identity)
        identity_sha = protocol.sha256_file(identity_path)
        private_records = {
            name: evidence_records[name]
            for name in (
                "eval_holdout_source_ontology_48",
                "holdout_registry_48",
                "receiver_ontology_56",
                "historical_receiver_anchors_8",
            )
        }
        payload = {
            "protocol": HOLDOUT_PUBLIC_PROTOCOL,
            "status": "committed",
            "dataset_version": protocol.DATASET_VERSION,
            "counts": {
                "source_count": 48,
                "receiver_count": 56,
                "historical_anchor_count": 8,
            },
            "artifacts": _json_normalized(evidence_records),
            "identity_disjointness_report_sha256": identity_sha,
            "independent_language_review_status": "passed",
            "remaining_blockers": [],
        }
        validate_holdout_public_commitment(
            payload,
            identity_report_sha256=identity_sha,
            private_records=private_records,
        )
        output = project_root / protocol.HOLDOUT_PUBLIC_COMMITMENT
        protocol._require_no_symlink_components(output.parent)
        output_ownership: list[tuple[Path, tuple[int, int]]] = []
        try:
            _write_json_owned_exclusive(
                output, payload, mode=0o644, ownership=output_ownership
            )
            observed = _load_public_json(
                project_root, protocol.HOLDOUT_PUBLIC_COMMITMENT
            )
            validate_holdout_public_commitment(
                observed,
                identity_report_sha256=identity_sha,
                private_records=private_records,
            )
            after_records = {
                name: _physical_record(
                    evidence_root / basename,
                    HOLDOUT_FROZEN_RECORDS[name]["row_count"],
                )
                for name, basename in HOLDOUT_EVIDENCE_BASENAMES.items()
            }
            protocol.require(
                after_records == evidence_records,
                "r6 evidence changed during holdout publication",
            )
            result = {
                "status": "prepared_holdout_not_authorized",
                "sha256": protocol.sha256_file(output),
            }
        except BaseException:
            _rollback_owned_outputs(output_ownership)
            raise
        return result


def _build_pending_payload(
    project_root: Path, private_root: Path
) -> dict[str, Any]:
    paths = _opening_paths(project_root, private_root)
    opening_records = _records_for_openings(paths)
    graph_salt = _read_text_secret(paths["graph_assignment_salt"])
    selector_salt = _read_text_secret(paths["selector_salt"])
    evaluation_salt = _read_text_secret(paths["evaluation_seed_salt"])
    screening_seed = _read_text_secret(paths["screening_seed"], integer=True)
    assert isinstance(graph_salt, str) and isinstance(selector_salt, str)
    assert isinstance(evaluation_salt, str) and isinstance(screening_seed, int)
    secret_payload = _load_private_json(
        private_root, PRIVATE_INPUTS["stage0_secrets"]
    )
    secret_commitments = _validate_secrets(
        secret_payload,
        graph_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
    )
    return {
        "protocol": PENDING_PROTOCOL,
        "schema": PENDING_SCHEMA,
        "registry": PENDING_REGISTRY,
        "dataset_version": protocol.DATASET_VERSION,
        "stage": 0,
        "status": "frozen_components_pending_authorization",
        "authorization_status": "not_authorized",
        "candidate_count": 576,
        "cell_counts": {
            f"{group}:{variant}": protocol.CELL_COUNTS[(group, variant)]
            for group, variant in protocol.CELL_ORDER
        },
        "sizing_rule": _expected_sizing_rule(opening_records),
        "design_input": {
            "preregistration": {
                "path": PREREG_PATH.as_posix(),
                "sha256": EXPECTED_PREREG_SHA256,
            },
            "v2_termination": {
                "path": protocol.V2_TERMINATION.as_posix(),
                "sha256": protocol.V2_RUNTIME_READ_ALLOWLIST[
                    protocol.V2_TERMINATION.as_posix()
                ],
            },
        },
        "curation_audit": _expected_curation_audit(opening_records),
        "public_metadata": _expected_public_metadata(
            opening_records, secret_commitments
        ),
        "component_commitments": opening_records,
        "remaining_blockers": [],
    }


def prepare_pending(project_root: Path, private_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    _reject_existing_preparation_boundary(project_root)
    private_root = _require_disjoint_project_private_roots(
        project_root, private_root
    )
    pending_path = project_root / protocol.STAGE0_PUBLIC
    binding_path = private_root / "causal_selection_binding_v3.json"
    wrapper_path = project_root / protocol.STAGE0_REGISTRY
    for path in (pending_path, binding_path, wrapper_path):
        if os.path.lexists(path):
            raise FileExistsError(f"prepare-pending target/boundary exists: {path}")
    with _authorization_mutex(private_root):
        _reject_existing_preparation_boundary(project_root)
        payload = _build_pending_payload(project_root, private_root)
        state: dict[str, Any] = {
            "pending_validated": False,
            "wrapper_published": False,
            "wrapper_sha256": None,
        }
        result = _authorize_impl(
            project_root=project_root,
            private_root=private_root,
            pending_path=pending_path,
            binding_path=binding_path,
            wrapper_path=wrapper_path,
            state=state,
            preflight_only=True,
            pending_payload_override=payload,
        )
        pending_sha = protocol.sha256_bytes(protocol.canonical_json_bytes(payload))
        pending_ownership: list[tuple[Path, tuple[int, int]]] = []
        try:
            _write_json_owned_exclusive(
                pending_path,
                payload,
                mode=0o644,
                ownership=pending_ownership,
            )
            observed = _load_public_json(project_root, protocol.STAGE0_PUBLIC)
            protocol.require(
                observed == payload
                and protocol.sha256_file(pending_path) == pending_sha,
                "pending commitment readback mismatch",
            )
            prepared = {
                "status": "prepared_pending_not_authorized",
                "opening_count": result["opening_count"],
                "pending_sha256": pending_sha,
            }
        except BaseException:
            _rollback_owned_outputs(pending_ownership)
            raise
        return prepared


def preflight(project_root: Path, private_root: Path) -> dict[str, Any]:
    project_root = protocol.validate_project_root(project_root)
    private_root = protocol._canonical_lexical_absolute(private_root)
    _reject_existing_terminal_or_stage1(project_root)
    with _authorization_mutex(private_root):
        _reject_existing_terminal_or_stage1(project_root)
        state: dict[str, Any] = {
            "pending_validated": False,
            "wrapper_published": False,
            "wrapper_sha256": None,
        }
        return _authorize_impl(
            project_root=project_root,
            private_root=private_root,
            pending_path=project_root / protocol.STAGE0_PUBLIC,
            binding_path=private_root / "causal_selection_binding_v3.json",
            wrapper_path=project_root / protocol.STAGE0_REGISTRY,
            state=state,
            preflight_only=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    static = subparsers.add_parser("prepare-static")
    static.add_argument("--project-root", required=True, type=Path)

    sample = subparsers.add_parser("sample-secrets")
    sample.add_argument("--project-root", required=True, type=Path)
    sample.add_argument("--private-root", required=True, type=Path)
    sample.add_argument("--secret-audit-root", required=True, type=Path)

    historical_audit = subparsers.add_parser("audit-historical-secrets")
    historical_audit.add_argument("--project-root", required=True, type=Path)
    historical_audit.add_argument("--private-root", required=True, type=Path)
    historical_audit.add_argument("--secret-audit-root", required=True, type=Path)
    historical_audit.add_argument(
        "--clean-projection-root", required=True, type=Path
    )
    historical_audit.add_argument(
        "--authorizer-projection-root", required=True, type=Path
    )

    private = subparsers.add_parser("prepare-private")
    private.add_argument("--project-root", required=True, type=Path)
    private.add_argument("--private-root", required=True, type=Path)
    private.add_argument("--secret-audit-root", required=True, type=Path)

    holdout = subparsers.add_parser("prepare-holdout")
    holdout.add_argument("--project-root", required=True, type=Path)
    holdout.add_argument("--r6-evidence-root", required=True, type=Path)

    pending = subparsers.add_parser("prepare-pending")
    pending.add_argument("--project-root", required=True, type=Path)
    pending.add_argument("--private-root", required=True, type=Path)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--project-root", required=True, type=Path)
    preflight_parser.add_argument("--private-root", required=True, type=Path)

    authorize_parser = subparsers.add_parser("authorize")
    authorize_parser.add_argument("--project-root", required=True, type=Path)
    authorize_parser.add_argument("--private-root", required=True, type=Path)
    authorize_parser.add_argument(
        "--pending",
        default=protocol.STAGE0_PUBLIC.as_posix(),
    )
    authorize_parser.add_argument(
        "--selection-binding-output",
        default="causal_selection_binding_v3.json",
    )
    authorize_parser.add_argument(
        "--stage0-output",
        default=protocol.STAGE0_REGISTRY.as_posix(),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-static":
        result = prepare_static(args.project_root)
    elif args.command == "sample-secrets":
        result = sample_secrets(
            args.project_root,
            args.private_root,
            args.secret_audit_root,
        )
    elif args.command == "audit-historical-secrets":
        result = audit_historical_secrets(
            args.project_root,
            args.private_root,
            args.secret_audit_root,
            args.clean_projection_root,
            args.authorizer_projection_root,
        )
    elif args.command == "prepare-private":
        result = prepare_private(
            args.project_root,
            args.private_root,
            args.secret_audit_root,
        )
    elif args.command == "prepare-holdout":
        result = prepare_holdout(args.project_root, args.r6_evidence_root)
    elif args.command == "prepare-pending":
        result = prepare_pending(args.project_root, args.private_root)
    elif args.command == "preflight":
        result = preflight(args.project_root, args.private_root)
    else:
        project_root = protocol.validate_project_root(args.project_root)
        private_root = protocol._canonical_lexical_absolute(args.private_root)
        protocol.require(
            args.pending == protocol.STAGE0_PUBLIC.as_posix()
            and args.stage0_output == protocol.STAGE0_REGISTRY.as_posix()
            and args.selection_binding_output == "causal_selection_binding_v3.json",
            "authorizer paths must use exact canonical defaults",
        )
        result = authorize(
            project_root=project_root,
            private_root=private_root,
            pending_path=project_root / args.pending,
            binding_path=private_root / args.selection_binding_output,
            wrapper_path=project_root / args.stage0_output,
        )
    print(protocol.canonical_json_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
