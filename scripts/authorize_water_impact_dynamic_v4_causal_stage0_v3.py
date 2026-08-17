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
import json
import os
import stat
import subprocess
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
GENERATION_SPEC_PROTOCOL = "water_impact_dynamic_v4_generation_spec_v3"
SEED_AUDIT_PROTOCOL = "water_impact_dynamic_v4_preselection_seed_audit_v3"
V2_FORBIDDEN_SEED_INVENTORY_SHA256 = (
    "f2f72728a83c7e3ec54735a58f3f2e0a5afd1c132822eeecad7dc2006cb5ecd4"
)
SECRETS_PROTOCOL = "water_impact_dynamic_v4_causal_stage0_secrets_v3"
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

EXPECTED_PREREG_SHA256 = (
    "fc3ee25586037099440476392305a3be34e8fecfda8740ae7c6eea201c3c3b7d"
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
PUBLIC_OPENINGS = {
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
assert len(OPENING_NAMES) == 30

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
        "pending component inventory must be the exact 30 openings",
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
        payload["dataset_version"] == protocol.DATASET_VERSION
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


def _validate_secrets(
    payload: Mapping[str, Any], *, graph_salt: str, selector_salt: str,
    evaluation_salt: str, screening_seed: int
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
        and payload["evaluation_seed_salt"] == evaluation_salt,
        "Stage-0 secret openings do not match",
    )
    protocol.validate_secret_separation(
        graph_assignment_salt=graph_salt,
        selector_salt=selector_salt,
        evaluation_salt=evaluation_salt,
        screening_seed=screening_seed,
    )
    return {
        "screening_seed": _secret_commitment("causal_screening_seed_v3", screening_seed),
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
        protocol.require(not candidate.is_symlink(), "model inventory contains a symlink")
        if candidate.is_file():
            protocol.require(candidate.stat().st_nlink == 1, "model inventory contains a hardlink")
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
import importlib.metadata, json, os, platform, sys
import torch
package_names = json.loads(sys.argv[1])
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
        ],
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=60,
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
    return {
        "accelerator_type": "CUDA",
        "device_count": device_count,
        "device_models": device_models,
    }


def _validate_cost_calibration(
    payload: Mapping[str, Any], *, model_sha: str, runtime_sha: str,
    render_sha: str, live_hardware: Mapping[str, Any]
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
            "public_prompt_sha256",
            "wall_time_seconds",
            "maximum_wall_time_seconds",
            "maximum_allowed_seconds",
            "candidate_count",
            "gpu_hour_cap",
            "passes",
        },
        "screening cost calibration",
    )
    times = payload["wall_time_seconds"]
    prompts = payload["public_prompt_sha256"]
    protocol.require(
        payload["protocol"] == COST_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["hardware"] == dict(live_hardware)
        and payload["model_content_inventory_sha256"] == model_sha
        and payload["runtime_registry_sha256"] == runtime_sha
        and payload["render_configuration_sha256"] == render_sha
        and isinstance(prompts, list)
        and len(prompts) == 5
        and len(set(prompts)) == 5
        and all(protocol.is_hex64(item) for item in prompts)
        and isinstance(times, list)
        and len(times) == 5
        and all(type(item) in (int, float) and item > 0 for item in times)
        and payload["maximum_wall_time_seconds"] == max(times)
        and payload["maximum_allowed_seconds"] == 600
        and max(times) <= 600
        and payload["candidate_count"] == 576
        and payload["gpu_hour_cap"] == 100
        and payload["passes"] is True,
        "screening cost calibration failed",
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
        and relation in {"equal", "strict_superset"},
        "forbidden seed source audit does not prove v2 coverage",
    )
    if relation == "equal":
        protocol.require(
            payload["v3_seed_count"] == payload["v2_seed_count"]
            and payload["v3_additional_seed_count"] == 0,
            "equal forbidden-seed relation/count mismatch",
        )
    else:
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


def _write_private_binding_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    expected = protocol.canonical_json_bytes(dict(payload))
    try:
        protocol.write_json_exclusive_atomic(path, payload, mode=0o600)
    except Exception:
        if path.is_file() and not path.is_symlink():
            info = path.stat()
            if (
                info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o600
                and protocol.sha256_file(path) == protocol.sha256_bytes(expected)
            ):
                path.unlink()
        raise


def _remove_owned_json_if_present(
    path: Path, payload: Mapping[str, Any], *, mode: int
) -> None:
    """Remove only the exact single-link artifact created by this invocation."""

    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"cannot safely roll back replaced artifact: {path}")
    info = path.stat()
    expected_sha = protocol.sha256_bytes(protocol.canonical_json_bytes(dict(payload)))
    if (
        info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != mode
        or protocol.sha256_file(path) != expected_sha
    ):
        raise RuntimeError(f"cannot safely roll back drifted artifact: {path}")
    path.unlink()


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

    pending = _load_public_json(project_root, protocol.STAGE0_PUBLIC)
    validate_pending(pending, project_root=project_root, pending_path=pending_path)
    state["pending_validated"] = True
    pending_sha = protocol.sha256_file(pending_path)
    paths = _opening_paths(project_root, private_root)
    for name in PRIVATE_INPUTS:
        protocol.validate_private_path(private_root, paths[name])
    opening_records = _records_for_openings(paths)
    protocol.require(opening_records == pending["component_commitments"], "pending commitments do not match exact 30 opening bytes")

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
    )
    bundle_payload = _load_private_json(private_root, PRIVATE_INPUTS["raw_root_bundle"])
    _validate_bundle(
        bundle_payload,
        private_records={name: opening_records[name] for name in PRIVATE_INPUTS},
    )

    def revalidate_deep(label: str) -> None:
        protocol.require(
            protocol.sha256_file(pending_path) == pending_sha,
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

    revalidate_deep("before binding publication")

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
    binding_created = False
    wrapper: dict[str, Any] | None = None
    try:
        try:
            _write_private_binding_exclusive(binding_path, binding_payload)
        except BaseException as exc:
            if not os.path.lexists(binding_path):
                raise StaticAuthorizationFailure(
                    "selection-binding publication failed before creation"
                ) from exc
            raise
        binding_created = True

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
        if binding_created:
            try:
                _remove_owned_json_if_present(
                    binding_path, binding_payload, mode=0o600
                )
            except Exception as exc:
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError("Stage-0 publication rollback was not safe") from rollback_error
        raise


def authorize(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument(
        "--pending",
        default=protocol.STAGE0_PUBLIC.as_posix(),
    )
    parser.add_argument(
        "--selection-binding-output",
        default="causal_selection_binding_v3.json",
    )
    parser.add_argument(
        "--stage0-output",
        default=protocol.STAGE0_REGISTRY.as_posix(),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = protocol.validate_project_root(args.project_root)
    private_root = protocol._canonical_lexical_absolute(args.private_root)
    protocol.require(
        args.pending == protocol.STAGE0_PUBLIC.as_posix()
        and args.stage0_output == protocol.STAGE0_REGISTRY.as_posix()
        and args.selection_binding_output == "causal_selection_binding_v3.json",
        "authorizer paths must use exact canonical defaults",
    )
    authorize(
        project_root=project_root,
        private_root=private_root,
        pending_path=project_root / args.pending,
        binding_path=private_root / args.selection_binding_output,
        wrapper_path=project_root / args.stage0_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
