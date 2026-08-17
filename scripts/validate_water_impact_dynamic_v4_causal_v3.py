#!/usr/bin/env python3
"""Fail-closed validation interfaces for v4_dev72_v3 causal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import select_water_impact_dynamic_v4_causal_v3 as selector
except ModuleNotFoundError:  # imported as scripts.validate_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import select_water_impact_dynamic_v4_causal_v3 as selector


SELECTED_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "status",
    "selected_count",
    "selected",
}
UNIT_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "status",
    "unit_count",
    "units",
}
UNIT_ROW_KEYS = {
    "unit_id",
    "semantic_case_id",
    "replicate",
    "seed",
    "group",
    "prompt_variant",
    "canonical_prompt",
}


def _load_exact_public_input(
    project_root: Path, supplied: Path, expected_relative: Path
) -> Mapping[str, Any]:
    root = protocol.validate_project_root(project_root)
    protocol.reject_forbidden_path(supplied)
    expected = root / expected_relative
    protocol.require(
        supplied.absolute() == expected.absolute(),
        f"public input path must be exactly {expected_relative.as_posix()}",
    )
    current = root
    for part in expected_relative.parts:
        current = current / part
        protocol.require(not current.is_symlink(), "public input has a symlink component")
    protocol.validate_runtime_read_path(root, supplied, allow_v2=False)
    return protocol.load_json(supplied, project_root=root, allow_v2=False)


def validate_registry_file(
    path: Path,
    *,
    stage: int,
    expected_stage0_sha256: str | None = None,
) -> Mapping[str, Any]:
    del path, expected_stage0_sha256
    stage_name = "Stage0" if stage == 0 else "Stage1"
    raise RuntimeError(
        f"formal {stage_name} provenance validation not implemented"
    )


def verify_registry_artifact(
    registry: Mapping[str, Any], name: str, path: Path
) -> None:
    protocol.require(name in registry["artifacts"], f"registry artifact missing: {name}")
    record = registry["artifacts"][name]
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink artifact: {path}")
    protocol.require(record["sha256"] == protocol.sha256_file(path), f"artifact byte hash mismatch: {name}")
    protocol.require(record["size_bytes"] == path.stat().st_size, f"artifact size mismatch: {name}")


def validate_stage0_core(
    *,
    registry: Mapping[str, Any],
    graph: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    holdout_ontology: Mapping[str, Any],
    receiver_ontology: Mapping[str, Any],
    historical_anchors: Mapping[str, Any],
    source_bank: Mapping[str, Any],
    graph_assignment_salt: str,
    identity_report: Mapping[str, Any],
    construct_report: Mapping[str, Any],
    graph_file_sha256: str,
    template_file_sha256: str,
    field_rules_file_sha256: str,
    selection_rules_file_sha256: str,
) -> None:
    del (
        registry,
        graph,
        candidate_manifest,
        holdout_ontology,
        receiver_ontology,
        historical_anchors,
        source_bank,
        graph_assignment_salt,
        identity_report,
        construct_report,
        graph_file_sha256,
        template_file_sha256,
        field_rules_file_sha256,
        selection_rules_file_sha256,
    )
    raise RuntimeError("formal Stage0 provenance validation not implemented")


def validate_selected_payload(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    protocol.require_exact_keys(payload, SELECTED_TOP_KEYS, "selected manifest")
    protocol.require(payload["protocol"] == selector.SELECTED_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["status"] == "selected" and payload["selected_count"] == protocol.SELECTED_COUNT, "selected manifest protocol/count mismatch")
    rows = payload["selected"]
    protocol.require(isinstance(rows, list) and len(rows) == protocol.SELECTED_COUNT, "selected rows invalid")
    expected_keys = set(protocol.GRAPH_EDGE_KEYS) | {"selection_rank_sha256"}
    for row in rows:
        protocol.require_exact_keys(row, expected_keys, "selected row")
        protocol.candidate_record_bytes({key: row[key] for key in protocol.GRAPH_EDGE_KEYS})
        protocol.require(protocol.is_hex64(row["selection_rank_sha256"]), "selected rank invalid")
    selector.validate_selected_rows(rows)
    protocol.require([row["selection_rank_sha256"] for row in rows] == sorted(row["selection_rank_sha256"] for row in rows), "selected rows not in rank order")
    return tuple(rows)


def validate_unit_payload(
    payload: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    evaluation_salt: str,
    screening_seed: int,
    forbidden_seeds: set[int],
) -> tuple[Mapping[str, Any], ...]:
    protocol.require_exact_keys(payload, UNIT_TOP_KEYS, "unit manifest")
    protocol.require(payload["protocol"] == selector.UNIT_PROTOCOL and payload["dataset_version"] == protocol.DATASET_VERSION and payload["status"] == "frozen" and payload["unit_count"] == protocol.UNIT_COUNT, "unit manifest protocol/count mismatch")
    units = payload["units"]
    protocol.require(isinstance(units, list) and len(units) == protocol.UNIT_COUNT, "unit rows invalid")
    selected_by_id = {str(row["case_id"]): row for row in selected_rows}
    seen_pairs: set[tuple[str, int]] = set()
    seen_seeds: set[int] = set()
    for row in units:
        protocol.require_exact_keys(row, UNIT_ROW_KEYS, "unit row")
        case_id = row["semantic_case_id"]
        replicate = row["replicate"]
        protocol.require(case_id in selected_by_id and replicate in protocol.REPLICATES and (case_id, replicate) not in seen_pairs, "unit case/replicate binding invalid")
        selected = selected_by_id[case_id]
        expected_seed = protocol.derive_evaluation_seed(evaluation_salt, case_id, replicate)
        protocol.require(row["unit_id"] == f"{case_id}:r{replicate}" and row["seed"] == expected_seed, "unit ID/seed derivation mismatch")
        protocol.require(expected_seed not in seen_seeds and expected_seed not in forbidden_seeds and expected_seed != screening_seed, "unit seed collision")
        protocol.require(row["group"] == selected["group"] and row["prompt_variant"] == selected["prompt_variant"] and row["canonical_prompt"] == selected["canonical_prompt"], "unit semantic fields drift")
        seen_pairs.add((case_id, replicate))
        seen_seeds.add(expected_seed)
    protocol.require(seen_pairs == {(case_id, replicate) for case_id in selected_by_id for replicate in protocol.REPLICATES}, "unit manifest coverage incomplete")
    return tuple(units)


def validate_stage1_core(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise RuntimeError("formal Stage1 provenance validation not implemented")


def validate_static_code_boundary(project_root: Path) -> None:
    project_root = protocol.validate_project_root(project_root)
    nonruntime = {
        "generator",
        "tests",
        "identity_disjointness_auditor",
        "construct_equivalence_auditor",
    }
    all_paths = {
        key: project_root / value
        for key, value in protocol.CODE_ARTIFACT_PATHS.items()
    }
    for key, path in all_paths.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required v3 code artifact is missing: {key}")
        protocol._require_no_symlink_components(path)
        if path.stat().st_nlink != 1:
            raise PermissionError(f"required v3 code artifact is hardlinked: {key}")
    protocol.validate_no_v2_imports(
        [path for key, path in all_paths.items() if key not in nonruntime],
        project_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static-code")
    static.add_argument("--project-root", type=Path, required=True)
    registry = subparsers.add_parser("registry")
    registry.add_argument("--project-root", type=Path, required=True)
    registry.add_argument("--path", type=Path, required=True)
    registry.add_argument("--stage", type=int, choices=(0, 1), required=True)
    registry.add_argument("--stage0-sha256")
    invalid = subparsers.add_parser("invalid-outcome")
    invalid.add_argument("--project-root", type=Path, required=True)
    invalid.add_argument("--path", type=Path, required=True)
    reports = subparsers.add_parser("reports")
    reports.add_argument("--project-root", type=Path, required=True)
    reports.add_argument("--identity", type=Path, required=True)
    reports.add_argument("--construct", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "static-code":
        validate_static_code_boundary(args.project_root)
        output = {"status": "v3_static_code_boundary_valid"}
    elif args.command == "registry":
        stage_name = "Stage0" if args.stage == 0 else "Stage1"
        raise RuntimeError(
            f"formal {stage_name} provenance validation not implemented"
        )
    elif args.command == "invalid-outcome":
        root = protocol.validate_project_root(args.project_root)
        invalid_payload = _load_exact_public_input(
            root, args.path, protocol.INVALID_OUTCOME
        )
        expected_stage0_sha256 = None
        if invalid_payload.get("failure_phase") != "stage0_authorization":
            stage0_path = root / protocol.STAGE0_REGISTRY
            _load_exact_public_input(root, stage0_path, protocol.STAGE0_REGISTRY)
            expected_stage0_sha256 = protocol.sha256_file(stage0_path)
        protocol.validate_invalid_outcome(
            invalid_payload,
            expected_stage0_sha256=expected_stage0_sha256,
        )
        output = {"status": "invalid_outcome_valid"}
    else:
        protocol.validate_identity_disjointness_report(
            _load_exact_public_input(
                args.project_root, args.identity, protocol.IDENTITY_REPORT
            )
        )
        protocol.validate_construct_equivalence_report(
            _load_exact_public_input(
                args.project_root, args.construct, protocol.CONSTRUCT_REPORT
            )
        )
        output = {"status": "audit_reports_valid"}
    print(protocol.canonical_json_bytes(output).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
