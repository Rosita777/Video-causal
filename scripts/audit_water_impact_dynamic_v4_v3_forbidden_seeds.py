#!/usr/bin/env python3
"""Isolated aggregate-only v2/v3 forbidden-seed coverage auditor."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    import audit_water_impact_dynamic_v4_v3_v2_disjointness as secure
except ModuleNotFoundError:  # imported as scripts.audit_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol
    from scripts import audit_water_impact_dynamic_v4_v3_v2_disjointness as secure


V2_INVENTORY_BASENAME = "causal_forbidden_seed_inventory_v2.json"
V3_INVENTORY_BASENAME = "causal_forbidden_seed_inventory_v3.json"
V2_PRIVATE_ALLOWLIST = frozenset({V2_INVENTORY_BASENAME})
V3_PRIVATE_ALLOWLIST = frozenset({V3_INVENTORY_BASENAME})
STANDARD_OUTPUT_RELATIVE = protocol.FORBIDDEN_SEED_SOURCE_AUDIT
V2_INVENTORY_PROTOCOL = "water_impact_dynamic_v4_forbidden_seed_inventory_v2"
V3_INVENTORY_PROTOCOL = "water_impact_dynamic_v4_forbidden_seed_inventory_v3"
SEED_ENCODING = "nonnegative JSON integer below 2^63"
DEFAULT_V2_INVENTORY_SHA256 = (
    "f2f72728a83c7e3ec54735a58f3f2e0a5afd1c132822eeecad7dc2006cb5ecd4"
)
REPORT_KEYS = {
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
}


@dataclass(frozen=True)
class ForbiddenSeedAuditContract:
    v2_stage0_sha256: str = protocol.V2_STAGE0_SHA256
    v2_inventory_sha256: str = DEFAULT_V2_INVENTORY_SHA256


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


def _read_single_private_input(
    root: Path, allowlist: frozenset[str], basename: str
) -> bytes:
    with secure.SecurePrivateRoot(root, allowlist) as private_root:
        _require(
            set(os.listdir(private_root.fd)) == set(allowlist),
            "isolated forbidden-seed root inventory is not exact",
        )
        return private_root.read_exact(basename)


def _validate_inventory(
    raw: bytes, *, protocol_name: str
) -> tuple[dict[str, Any], tuple[int, ...]]:
    payload = _json_object(raw, "forbidden seed inventory")
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "dataset",
            "status",
            "seed_encoding",
            "source_commitments",
            "seeds",
        },
        "forbidden seed inventory",
    )
    _require(
        payload["protocol"] == protocol_name
        and payload["dataset"] == protocol.DATASET
        and payload["status"] == "frozen_by_independent_seed_auditor"
        and payload["seed_encoding"] == SEED_ENCODING,
        "forbidden seed inventory protocol/status mismatch",
    )
    sources = payload["source_commitments"]
    _require(isinstance(sources, list) and sources, "seed source commitments missing")
    source_names: list[str] = []
    for source in sources:
        protocol.require_exact_keys(
            source, {"name", "sha256", "seed_count"}, "seed source commitment"
        )
        _require(
            isinstance(source["name"], str)
            and source["name"].strip() == source["name"]
            and source["name"]
            and protocol.is_hex64(source["sha256"])
            and type(source["seed_count"]) is int
            and source["seed_count"] >= 0,
            "seed source commitment is invalid",
        )
        source_names.append(source["name"])
    _require(
        source_names == sorted(source_names)
        and len(set(source_names)) == len(source_names),
        "seed source commitments must be sorted and unique",
    )
    seeds = payload["seeds"]
    _require(
        isinstance(seeds, list)
        and seeds
        and all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds),
        "forbidden seeds must be positive-length signed63 integers",
    )
    _require(
        seeds == sorted(seeds) and len(set(seeds)) == len(seeds),
        "forbidden seeds must be sorted and unique",
    )
    _require(
        sum(source["seed_count"] for source in sources) >= len(seeds),
        "seed source counts do not cover the inventory",
    )
    _require(
        not protocol.contains_placeholder(payload),
        "forbidden seed inventory contains placeholder content",
    )
    return payload, tuple(seeds)


def validate_report(
    payload: Mapping[str, Any], contract: ForbiddenSeedAuditContract
) -> Mapping[str, Any]:
    protocol.require_exact_keys(payload, REPORT_KEYS, "forbidden seed source report")
    _require(
        payload["protocol"] == protocol.FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["v2_stage0_registry_sha256"] == contract.v2_stage0_sha256
        and payload["v2_forbidden_seed_inventory_sha256"]
        == contract.v2_inventory_sha256,
        "forbidden seed source report protocol/hash mismatch",
    )
    for name in (
        "v3_forbidden_seed_inventory_sha256",
        "v2_forbidden_seed_inventory_sha256",
        "v2_stage0_registry_sha256",
    ):
        _require(protocol.is_hex64(payload[name]), "forbidden seed report hash invalid")
    count_names = (
        "v2_seed_count",
        "v3_seed_count",
        "intersection_seed_count",
        "v2_missing_from_v3_count",
        "v3_additional_seed_count",
    )
    _require(
        all(type(payload[name]) is int and payload[name] >= 0 for name in count_names),
        "forbidden seed report counts are invalid",
    )
    _require(
        payload["v2_seed_count"] > 0
        and payload["v3_seed_count"] > 0
        and payload["v2_missing_from_v3_count"] == 0
        and payload["intersection_seed_count"] == payload["v2_seed_count"]
        and payload["intersection_seed_count"]
        + payload["v2_missing_from_v3_count"]
        == payload["v2_seed_count"]
        and payload["intersection_seed_count"]
        + payload["v3_additional_seed_count"]
        == payload["v3_seed_count"],
        "forbidden seed report arithmetic does not prove coverage",
    )
    relation = payload["set_relation"]
    _require(relation in {"equal", "strict_superset"}, "set relation is invalid")
    if relation == "equal":
        _require(
            payload["v3_seed_count"] == payload["v2_seed_count"]
            and payload["v3_additional_seed_count"] == 0,
            "equal relation counts are inconsistent",
        )
    else:
        _require(
            payload["v3_seed_count"] > payload["v2_seed_count"]
            and payload["v3_additional_seed_count"] > 0,
            "strict-superset relation counts are inconsistent",
        )
    _require(not protocol.contains_placeholder(payload), "forbidden seed report contains placeholder")
    _require(
        all(not isinstance(value, (list, dict)) for value in payload.values()),
        "forbidden seed report may contain aggregate scalars only",
    )
    return payload


def build_report(
    *,
    wrapper: Mapping[str, Any],
    v2_raw: bytes,
    v3_raw: bytes,
    contract: ForbiddenSeedAuditContract,
) -> dict[str, Any]:
    record = secure._artifact_record(wrapper, "forbidden_seed_inventory", None)
    secure._verify_committed_bytes(v2_raw, record, "v2 forbidden seed inventory")
    _require(
        secure.sha256_bytes(v2_raw) == contract.v2_inventory_sha256,
        "v2 forbidden seed inventory differs from the frozen hash",
    )
    _, v2_seeds = _validate_inventory(
        v2_raw,
        protocol_name=V2_INVENTORY_PROTOCOL,
    )
    _, v3_seeds = _validate_inventory(
        v3_raw,
        protocol_name=V3_INVENTORY_PROTOCOL,
    )
    v2_set = set(v2_seeds)
    v3_set = set(v3_seeds)
    missing = v2_set - v3_set
    _require(not missing, "v3 forbidden inventory omits a v2 seed")
    intersection_count = len(v2_set & v3_set)
    additional_count = len(v3_set - v2_set)
    relation = "equal" if not additional_count else "strict_superset"
    report = {
        "protocol": protocol.FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL,
        "status": "passed",
        "dataset_version": protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": contract.v2_stage0_sha256,
        "v2_forbidden_seed_inventory_sha256": secure.sha256_bytes(v2_raw),
        "v3_forbidden_seed_inventory_sha256": secure.sha256_bytes(v3_raw),
        "v2_seed_count": len(v2_set),
        "v3_seed_count": len(v3_set),
        "intersection_seed_count": intersection_count,
        "v2_missing_from_v3_count": len(missing),
        "v3_additional_seed_count": additional_count,
        "set_relation": relation,
    }
    validate_report(report, contract)
    return report


def run_audit(
    *,
    project_root: Path,
    private_v2_root: Path,
    private_v3_root: Path,
    contract: ForbiddenSeedAuditContract = ForbiddenSeedAuditContract(),
    publish: bool = True,
) -> tuple[dict[str, Any], str | None]:
    secure.validate_distinct_roots(project_root, private_v2_root, private_v3_root)
    wrapper, _ = secure.load_v2_wrapper(
        project_root,
        secure.IdentityAuditContract(v2_stage0_sha256=contract.v2_stage0_sha256),
    )
    v2_raw = _read_single_private_input(
        private_v2_root, V2_PRIVATE_ALLOWLIST, V2_INVENTORY_BASENAME
    )
    v3_raw = _read_single_private_input(
        private_v3_root, V3_PRIVATE_ALLOWLIST, V3_INVENTORY_BASENAME
    )
    report = build_report(
        wrapper=wrapper,
        v2_raw=v2_raw,
        v3_raw=v3_raw,
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
    report, digest = run_audit(
        project_root=args.project_root,
        private_v2_root=args.private_v2_root,
        private_v3_root=args.private_v3_root,
    )
    print(
        secure.canonical_json_bytes(
            {
                "status": report["status"],
                "sha256": digest,
                "v2_seed_count": report["v2_seed_count"],
                "v3_seed_count": report["v3_seed_count"],
                "set_relation": report["set_relation"],
            }
        ).decode("ascii"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
