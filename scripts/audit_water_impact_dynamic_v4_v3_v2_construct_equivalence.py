#!/usr/bin/env python3
"""Isolated aggregate-only v2/v3 construct-equivalence auditor."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import audit_water_impact_dynamic_v4_v3_v2_disjointness as secure
except ModuleNotFoundError:  # imported as scripts.audit_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import audit_water_impact_dynamic_v4_v3_v2_disjointness as secure


V2_TEMPLATE_BASENAME = "causal_stage0_templates_private_v2.json"
V2_FIELD_BASENAME = "causal_stage0_field_rules_private_v2.json"
V2_SELECTION_BASENAME = "causal_stage0_selection_rules_private_v2.json"
V3_TEMPLATE_BASENAME = "causal_stage0_templates_private_v3.json"
V3_FIELD_BASENAME = "causal_stage0_field_rules_private_v3.json"
V3_SELECTION_BASENAME = "causal_stage0_selection_rules_private_v3.json"
V2_PRIVATE_ALLOWLIST = frozenset(
    {V2_TEMPLATE_BASENAME, V2_FIELD_BASENAME, V2_SELECTION_BASENAME}
)
V3_PRIVATE_ALLOWLIST = frozenset(
    {V3_TEMPLATE_BASENAME, V3_FIELD_BASENAME, V3_SELECTION_BASENAME}
)
STANDARD_OUTPUT_RELATIVE = protocol.CONSTRUCT_REPORT


@dataclass(frozen=True)
class ConstructAuditContract:
    v2_stage0_sha256: str = protocol.V2_STAGE0_SHA256
    v2_template_sha256: str = protocol.V2_TEMPLATE_SHA256
    v2_field_rules_sha256: str = protocol.V2_FIELD_RULES_SHA256
    v2_selection_rules_sha256: str = protocol.V2_SELECTION_RULES_SHA256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _validate_construct_report(
    payload: Mapping[str, Any], contract: ConstructAuditContract
) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "v2_stage0_registry_sha256",
            "v2_file_sha256",
            "v3_file_sha256",
            "qualification_sha256",
            "cell_quota_sha256",
            "exact_equal",
        },
        "construct audit report",
    )
    _require(
        payload["protocol"] == protocol.CONSTRUCT_REPORT_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["v2_stage0_registry_sha256"] == contract.v2_stage0_sha256,
        "construct audit protocol/status mismatch",
    )
    expected_v2 = {
        "templates": contract.v2_template_sha256,
        "field_rules": contract.v2_field_rules_sha256,
        "selection_rules": contract.v2_selection_rules_sha256,
    }
    _require(payload["v2_file_sha256"] == expected_v2, "construct v2 hashes mismatch")
    for name in ("v2_file_sha256", "v3_file_sha256"):
        values = protocol.require_exact_keys(
            payload[name], {"templates", "field_rules", "selection_rules"}, name
        )
        _require(all(protocol.is_hex64(value) for value in values.values()), "construct file hash invalid")
    for name in ("qualification_sha256", "cell_quota_sha256"):
        values = protocol.require_exact_keys(payload[name], {"v2", "v3"}, name)
        _require(
            all(protocol.is_hex64(value) for value in values.values())
            and values["v2"] == values["v3"],
            f"construct {name} mismatch",
        )
    _require(
        payload["exact_equal"]
        == {
            "templates": True,
            "field_rules": True,
            "qualification": True,
            "cell_quota": True,
        },
        "construct equality flags failed",
    )
    _require(not protocol.contains_placeholder(payload), "construct audit contains placeholder")
    secure._assert_aggregate_only(payload)
    return payload


def build_construct_report(
    *,
    wrapper: Mapping[str, Any],
    v2_files: Mapping[str, bytes],
    v3_files: Mapping[str, bytes],
    contract: ConstructAuditContract,
) -> dict[str, Any]:
    commitments = {
        "templates": secure._artifact_record(wrapper, "canonical_templates", None),
        "field_rules": secure._artifact_record(wrapper, "field_normalization", None),
        "selection_rules_rank": secure._artifact_record(wrapper, "ranking_formula", None),
        "selection_rules_subset": secure._artifact_record(
            wrapper, "constrained_subset_algorithm", None
        ),
    }
    _require(
        commitments["selection_rules_rank"]
        == commitments["selection_rules_subset"],
        "v2 selection-rule commitments differ",
    )
    by_role = {
        "templates": (V2_TEMPLATE_BASENAME, V3_TEMPLATE_BASENAME),
        "field_rules": (V2_FIELD_BASENAME, V3_FIELD_BASENAME),
        "selection_rules": (V2_SELECTION_BASENAME, V3_SELECTION_BASENAME),
    }
    v2_hashes: dict[str, str] = {}
    v3_hashes: dict[str, str] = {}
    for role, (v2_name, v3_name) in by_role.items():
        v2_raw = v2_files[v2_name]
        commitment = (
            commitments["selection_rules_rank"]
            if role == "selection_rules"
            else commitments[role]
        )
        secure._verify_committed_bytes(v2_raw, commitment, f"v2 {role}")
        v2_hashes[role] = secure.sha256_bytes(v2_raw)
        v3_hashes[role] = secure.sha256_bytes(v3_files[v3_name])
    _require(
        v2_hashes
        == {
            "templates": contract.v2_template_sha256,
            "field_rules": contract.v2_field_rules_sha256,
            "selection_rules": contract.v2_selection_rules_sha256,
        },
        "v2 construct bytes differ from frozen hashes",
    )
    _require(v2_files[V2_TEMPLATE_BASENAME] == v3_files[V3_TEMPLATE_BASENAME], "template files differ")
    _require(v2_files[V2_FIELD_BASENAME] == v3_files[V3_FIELD_BASENAME], "field-rule files differ")

    v2_rules = _json_object(v2_files[V2_SELECTION_BASENAME], "v2 selection rules")
    v3_rules = _json_object(v3_files[V3_SELECTION_BASENAME], "v3 selection rules")
    _require(
        not protocol.contains_placeholder(v2_rules)
        and not protocol.contains_placeholder(v3_rules),
        "selection-rule input contains placeholder",
    )
    for key in ("qualification", "cell_quota"):
        _require(key in v2_rules and key in v3_rules, f"selection rules omit {key}")
    qualification = {
        "v2": secure.sha256_bytes(secure.canonical_json_bytes(v2_rules["qualification"])),
        "v3": secure.sha256_bytes(secure.canonical_json_bytes(v3_rules["qualification"])),
    }
    cell_quota = {
        "v2": secure.sha256_bytes(secure.canonical_json_bytes(v2_rules["cell_quota"])),
        "v3": secure.sha256_bytes(secure.canonical_json_bytes(v3_rules["cell_quota"])),
    }
    _require(qualification["v2"] == qualification["v3"], "qualification objects differ")
    _require(cell_quota["v2"] == cell_quota["v3"], "cell-quota objects differ")
    report = {
        "protocol": protocol.CONSTRUCT_REPORT_PROTOCOL,
        "status": "passed",
        "dataset_version": protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": contract.v2_stage0_sha256,
        "v2_file_sha256": v2_hashes,
        "v3_file_sha256": v3_hashes,
        "qualification_sha256": qualification,
        "cell_quota_sha256": cell_quota,
        "exact_equal": {
            "templates": True,
            "field_rules": True,
            "qualification": True,
            "cell_quota": True,
        },
    }
    _validate_construct_report(report, contract)
    if contract == ConstructAuditContract():
        protocol.validate_construct_equivalence_report(report)
    return report


def run_construct_audit(
    *,
    project_root: Path,
    private_v2_root: Path,
    private_v3_root: Path,
    contract: ConstructAuditContract = ConstructAuditContract(),
    publish: bool = True,
) -> tuple[dict[str, Any], str | None]:
    secure.validate_distinct_roots(project_root, private_v2_root, private_v3_root)
    wrapper, _ = secure.load_v2_wrapper(
        project_root,
        secure.IdentityAuditContract(v2_stage0_sha256=contract.v2_stage0_sha256),
    )
    with secure.SecurePrivateRoot(private_v2_root, V2_PRIVATE_ALLOWLIST) as v2_root:
        v2_files = {name: v2_root.read_exact(name) for name in sorted(V2_PRIVATE_ALLOWLIST)}
    with secure.SecurePrivateRoot(private_v3_root, V3_PRIVATE_ALLOWLIST) as v3_root:
        v3_files = {name: v3_root.read_exact(name) for name in sorted(V3_PRIVATE_ALLOWLIST)}
    report = build_construct_report(
        wrapper=wrapper,
        v2_files=v2_files,
        v3_files=v3_files,
        contract=contract,
    )
    digest = (
        secure.write_report_to_relative(project_root, STANDARD_OUTPUT_RELATIVE, report)
        if publish
        else None
    )
    return report, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--private-v2-root", type=Path, required=True)
    parser.add_argument("--private-v3-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, digest = run_construct_audit(
        project_root=args.project_root,
        private_v2_root=args.private_v2_root,
        private_v3_root=args.private_v3_root,
    )
    print(
        secure.canonical_json_bytes(
            {
                "status": report["status"],
                "output": STANDARD_OUTPUT_RELATIVE.as_posix(),
                "sha256": digest,
                "exact_equal": report["exact_equal"],
            }
        ).decode("ascii"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
