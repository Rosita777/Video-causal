#!/usr/bin/env python3
"""Isolated aggregate-only v3/v2 identity-disjointness auditor.

The executable opens only the exact public v2 Stage-0 wrapper, one committed
v2 private candidate manifest, and four preregistered v3 private inputs.  It
never opens media, reviews, eligibility, salts, seeds, or sealed data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import water_impact_dynamic_v4_eval_protocol_v3 as protocol
except ModuleNotFoundError:  # imported as scripts.audit_...
    from scripts import water_impact_dynamic_v4_eval_protocol_v3 as protocol


V2_STAGE0_RELATIVE = Path(
    "data/water_impact_dynamic_v4/causal_stage0_commitment_v2.json"
)
V2_CANDIDATE_BASENAME = "causal_stage0_candidates_private_v2.json"
V3_SOURCE_BASENAME = "eval_holdout_source_ontology_private48_v3.json"
V3_RECEIVER_BASENAME = "receiver_ontology_private56_v3.json"
V3_HISTORICAL_BASENAME = "historical_receiver_anchors_private8_v3.json"
V3_GRAPH_BASENAME = "causal_stage0_candidate_graph_private576_v3.json"
V2_PRIVATE_ALLOWLIST = frozenset({V2_CANDIDATE_BASENAME})
V3_PRIVATE_ALLOWLIST = frozenset(
    {
        V3_SOURCE_BASENAME,
        V3_RECEIVER_BASENAME,
        V3_HISTORICAL_BASENAME,
        V3_GRAPH_BASENAME,
    }
)
STANDARD_OUTPUT_RELATIVE = protocol.IDENTITY_REPORT
FORBIDDEN_PATH_TOKENS = ("sealed", "final36", "quarantine")
FORBIDDEN_OUTPUT_KEYS = {
    "case_id",
    "source_id",
    "receiver_id",
    "source_phrase",
    "receiver_phrase",
    "prompt",
    "seed",
    "score",
    "media",
    "path",
    "row",
    "rows",
    "candidate",
    "candidates",
    "edge",
    "edges",
}

V2_CANDIDATE_TOP_KEYS = {
    "schema",
    "protocol",
    "dataset_version",
    "stage",
    "candidate_count",
    "candidates",
}
V2_CANDIDATE_ROW_KEYS = {
    "case_id",
    "group",
    "prompt_variant",
    "source_membership",
    "source_id",
    "source_phrase",
    "source_head_lemma",
    "source_physical_audit_status",
    "receiver_membership",
    "receiver_id",
    "receiver_phrase",
    "canonical_prompt",
    "canonical_record_sha256",
}
V3_SOURCE_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "source_count",
    "sources",
    "curation_audit",
    "disjointness_commitment",
}
V3_RECEIVER_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "receiver_count",
    "pools",
    "receivers",
    "curation_audit",
    "disjointness_commitment",
}
V3_HISTORICAL_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "anchor_count",
    "training_receiver_inventory_sha256",
    "v2_disjointness_commitment",
    "anchors",
}
V3_GRAPH_TOP_KEYS = {
    "protocol",
    "dataset_version",
    "status",
    "candidate_count",
    "cell_counts",
    "topology",
    "graph_assignment_salt_sha256",
    "r1",
    "r3",
    "anchors",
    "edges",
    "graph_sha256",
}
V3_SOURCE_ROW_KEYS = {
    "source_id",
    "source_phrase",
    "normalized_phrase",
    "head_lemma",
    "origin",
    "food_status",
    "shape_class",
    "color_family",
    "material_family",
    "texture_class",
    "impact_plausibility",
    "physical_audit_status",
    "curator",
    "curation_stratum",
    "group_pool",
    "head_ordinal",
}
V3_RECEIVER_ROW_KEYS = {
    "receiver_id",
    "receiver_phrase",
    "normalized_phrase",
    "head_lemma",
    "receiver_type",
    "pool",
    "receiver_ordinal",
    "curator_note",
    "curator",
}
V3_HISTORICAL_ROW_KEYS = {
    "anchor_id",
    "receiver_id",
    "receiver_phrase",
    "normalized_phrase",
    "head_lemma",
    "historical_training_binding_sha256",
}


@dataclass(frozen=True)
class IdentityAuditContract:
    v2_stage0_sha256: str = protocol.V2_STAGE0_SHA256
    v2_dataset_version: str = "v4_dev72_v2"
    v2_commitment_protocol: str = "water_impact_dynamic_v4_eval_commitment_registry_v2"
    v2_registry_schema: str = "water_impact_dynamic_v4_source_slot_registry_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_phrase(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def reject_forbidden_path(*paths: Path) -> None:
    for path in paths:
        lexical = _absolute(path)
        resolved = lexical.resolve(strict=False)
        for candidate in (lexical, resolved):
            if any(
                token in candidate.as_posix().casefold()
                for token in FORBIDDEN_PATH_TOKENS
            ):
                raise ValueError("auditor path references forbidden data")


def _require_real_components(path: Path) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        current = current / component
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"path contains a symlink component: {current}")
        if index < len(absolute.parts[1:]) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"path ancestor is not a directory: {current}")
    reject_forbidden_path(absolute)
    return absolute


def validate_distinct_roots(project_root: Path, v2_root: Path, v3_root: Path) -> None:
    roots = [_require_real_components(path).resolve(strict=True) for path in (project_root, v2_root, v3_root)]
    _require(len(set(roots)) == 3, "project/v2/v3 roots must be distinct")
    for left in roots:
        for right in roots:
            if left == right:
                continue
            try:
                left.relative_to(right)
            except ValueError:
                pass
            else:
                raise ValueError("project/v2/v3 roots may not be nested")


class SecurePrivateRoot(AbstractContextManager["SecurePrivateRoot"]):
    """Descriptor-rooted reader that refuses every nonallowlisted basename."""

    def __init__(self, root: Path, allowlist: Iterable[str]):
        self.path = _require_real_components(root)
        info = os.lstat(self.path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("private auditor root must be a real mode-700 directory")
        self.allowlist = frozenset(allowlist)
        self.fd = os.open(
            self.path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )

    def __exit__(self, *args: object) -> None:
        os.close(self.fd)

    def read_exact(self, basename: str) -> bytes:
        if basename not in self.allowlist or Path(basename).name != basename:
            raise PermissionError("private auditor attempted a nonallowlisted open")
        before = os.stat(basename, dir_fd=self.fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise PermissionError("private input must be mode-600 regular nlink-1")
        descriptor = os.open(
            basename,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self.fd,
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("private input changed while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            final_opened = os.fstat(descriptor)
            after = os.stat(basename, dir_fd=self.fd, follow_symlinks=False)
            if (
                (opened.st_dev, opened.st_ino, opened.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
                or (opened.st_dev, opened.st_ino, opened.st_size)
                != (final_opened.st_dev, final_opened.st_ino, final_opened.st_size)
                or after.st_nlink != 1
                or stat.S_IMODE(after.st_mode) != 0o600
            ):
                raise RuntimeError("private input changed while reading")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


def _open_relative_file(root: Path, relative: Path) -> bytes:
    """Read one exact public file without following any directory symlink."""

    root_path = _require_real_components(root)
    root_fd = os.open(
        root_path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptors = [root_fd]
    try:
        current = root_fd
        for component in relative.parts[:-1]:
            current = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current,
            )
            descriptors.append(current)
        basename = relative.name
        before = os.stat(basename, dir_fd=current, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise FileNotFoundError("public wrapper is not a regular file")
        descriptor = os.open(
            basename,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("public wrapper changed while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical readable JSON") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _artifact_record(
    wrapper: Mapping[str, Any], name: str, expected_rows: int | None
) -> Mapping[str, Any]:
    artifacts = wrapper.get("artifacts")
    _require(isinstance(artifacts, dict), "v2 wrapper artifact inventory missing")
    record = artifacts.get(name)
    _require(
        isinstance(record, dict)
        and set(record) == {"sha256", "size_bytes", "row_count"},
        f"v2 wrapper commitment is not exact: {name}",
    )
    _require(protocol.is_hex64(record["sha256"]), "v2 committed hash invalid")
    _require(
        isinstance(record["size_bytes"], int)
        and not isinstance(record["size_bytes"], bool)
        and record["size_bytes"] > 0,
        "v2 committed size invalid",
    )
    _require(record["row_count"] == expected_rows, "v2 committed row count mismatch")
    return record


def load_v2_wrapper(
    project_root: Path, contract: IdentityAuditContract
) -> tuple[dict[str, Any], bytes]:
    raw = _open_relative_file(project_root, V2_STAGE0_RELATIVE)
    _require(
        sha256_bytes(raw) == contract.v2_stage0_sha256,
        "v2 public Stage-0 wrapper hash mismatch",
    )
    wrapper = _json_object(raw, "v2 Stage-0 wrapper")
    expected = {
        "protocol",
        "dataset",
        "dataset_version",
        "stage",
        "status",
        "sealed_final36_status",
        "artifacts",
    }
    _require(set(wrapper) == expected, "v2 wrapper fields are not exact")
    _require(
        wrapper["protocol"] == contract.v2_commitment_protocol
        and wrapper["dataset"] == "causal"
        and wrapper["dataset_version"] == contract.v2_dataset_version
        and wrapper["stage"] == 0
        and wrapper["status"] == "committed"
        and wrapper["sealed_final36_status"] == "unopened",
        "v2 wrapper protocol/status mismatch",
    )
    return wrapper, raw


def _verify_committed_bytes(
    raw: bytes, record: Mapping[str, Any], label: str
) -> None:
    _require(len(raw) == record["size_bytes"], f"{label} committed size mismatch")
    _require(
        sha256_bytes(raw) == record["sha256"],
        f"{label} committed byte hash mismatch",
    )


def _validate_v2_candidates(raw: bytes) -> tuple[Mapping[str, Any], ...]:
    payload = _json_object(raw, "v2 candidate manifest")
    _require(not protocol.contains_placeholder(payload), "v2 candidate manifest contains placeholder")
    _require(set(payload) == V2_CANDIDATE_TOP_KEYS, "v2 candidate fields are not exact")
    _require(
        payload["schema"] == "water_impact_dynamic_v4_source_slot_registry_v2"
        and payload["protocol"] == payload["schema"]
        and payload["dataset_version"] == "v4_dev72_v2"
        and payload["stage"] == 0
        and payload["candidate_count"] == 48,
        "v2 candidate protocol/count mismatch",
    )
    rows = payload["candidates"]
    _require(isinstance(rows, list) and len(rows) == 48, "v2 candidate rows missing")
    for row in rows:
        _require(
            isinstance(row, dict) and set(row) == V2_CANDIDATE_ROW_KEYS,
            "v2 candidate row fields are not exact",
        )
        base = dict(row)
        digest = base.pop("canonical_record_sha256")
        _require(
            protocol.is_hex64(digest)
            and sha256_bytes(canonical_json_bytes(base)) == digest,
            "v2 candidate canonical record mismatch",
        )
    return tuple(rows)


def _validate_v3_sources(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _require(not protocol.contains_placeholder(payload), "v3 source ontology contains placeholder")
    _require(set(payload) == V3_SOURCE_TOP_KEYS, "v3 source ontology fields are not exact")
    rows = payload.get("sources")
    _require(
        payload.get("protocol")
        == "water_impact_dynamic_v4_eval_holdout_source_ontology_v3"
        and payload.get("dataset_version") == protocol.DATASET_VERSION
        and payload.get("source_count") == 48
        and isinstance(rows, list)
        and len(rows) == 48,
        "v3 source ontology count mismatch",
    )
    _require(
        all(isinstance(row, dict) and set(row) == V3_SOURCE_ROW_KEYS for row in rows),
        "v3 source row fields are not exact",
    )
    _require(len({row["source_id"] for row in rows}) == 48, "v3 source IDs repeat")
    _require(
        all(
            row["normalized_phrase"] == normalize_phrase(row["source_phrase"])
            and row["normalized_phrase"].split()[-1] == row["head_lemma"]
            for row in rows
        ),
        "v3 source normalization/head mismatch",
    )
    return tuple(rows)


def _validate_v3_receivers(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _require(not protocol.contains_placeholder(payload), "v3 receiver ontology contains placeholder")
    _require(set(payload) == V3_RECEIVER_TOP_KEYS, "v3 receiver ontology fields are not exact")
    rows = payload.get("receivers")
    _require(
        payload.get("protocol")
        == "water_impact_dynamic_v4_eval_receiver_ontology_v3"
        and payload.get("dataset_version") == protocol.DATASET_VERSION
        and payload.get("receiver_count") == 56
        and payload.get("pools") == {"R1": 24, "R3": 32}
        and isinstance(rows, list)
        and len(rows) == 56,
        "v3 receiver ontology count mismatch",
    )
    _require(
        all(isinstance(row, dict) and set(row) == V3_RECEIVER_ROW_KEYS for row in rows),
        "v3 receiver row fields are not exact",
    )
    _require(len({row["receiver_id"] for row in rows}) == 56, "v3 receiver IDs repeat")
    _require(
        all(
            row["normalized_phrase"] == normalize_phrase(row["receiver_phrase"])
            and row["normalized_phrase"].split()[-1] == row["head_lemma"]
            for row in rows
        ),
        "v3 receiver normalization/head mismatch",
    )
    return tuple(rows)


def _validate_v3_historical(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _require(not protocol.contains_placeholder(payload), "v3 historical anchors contain placeholder")
    _require(set(payload) == V3_HISTORICAL_TOP_KEYS, "v3 historical fields are not exact")
    rows = payload.get("anchors")
    _require(
        payload.get("protocol")
        == "water_impact_dynamic_v4_historical_receiver_anchors_v3"
        and payload.get("dataset_version") == protocol.DATASET_VERSION
        and payload.get("anchor_count") == 8
        and isinstance(rows, list)
        and len(rows) == 8,
        "v3 historical anchor count mismatch",
    )
    _require(
        all(
            isinstance(row, dict) and set(row) == V3_HISTORICAL_ROW_KEYS
            for row in rows
        ),
        "v3 historical row fields are not exact",
    )
    _require(len({row["receiver_id"] for row in rows}) == 8, "v3 historical receivers repeat")
    _require(
        all(
            row["normalized_phrase"] == normalize_phrase(row["receiver_phrase"])
            and row["normalized_phrase"].split()[-1] == row["head_lemma"]
            for row in rows
        ),
        "v3 historical normalization/head mismatch",
    )
    return tuple(rows)


def _validate_v3_graph(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    _require(not protocol.contains_placeholder(payload), "v3 graph contains placeholder")
    _require(set(payload) == V3_GRAPH_TOP_KEYS, "v3 graph fields are not exact")
    rows = payload.get("edges")
    _require(
        payload.get("protocol") == protocol.GRAPH_PROTOCOL
        and payload.get("dataset_version") == protocol.DATASET_VERSION
        and payload.get("candidate_count") == 576
        and isinstance(rows, list)
        and len(rows) == 576,
        "v3 graph protocol/count mismatch",
    )
    _require(
        all(isinstance(row, dict) and set(row) == set(protocol.GRAPH_EDGE_KEYS) for row in rows),
        "v3 graph edge fields are not exact",
    )
    for row in rows:
        candidate = dict(row)
        record_digest = candidate.pop("canonical_record_sha256")
        _require(
            protocol.is_hex64(record_digest)
            and sha256_bytes(canonical_json_bytes(candidate)) == record_digest,
            "v3 graph canonical record mismatch",
        )
    base = dict(payload)
    digest = base.pop("graph_sha256")
    _require(
        protocol.is_hex64(digest)
        and sha256_bytes(canonical_json_bytes(base)) == digest,
        "v3 graph self-hash mismatch",
    )
    _require(len({row["case_id"] for row in rows}) == 576, "v3 graph case IDs repeat")
    return tuple(rows)


def _ontology_bundle_sha256(files: Mapping[str, bytes]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {name: sha256_bytes(raw) for name, raw in sorted(files.items())}
        )
    )


def validate_identity_report(
    payload: Mapping[str, Any], contract: IdentityAuditContract
) -> Mapping[str, Any]:
    protocol.require_exact_keys(
        payload,
        {
            "protocol",
            "status",
            "dataset_version",
            "v2_stage0_registry_sha256",
            "v2_candidate_manifest_sha256",
            "v3_candidate_graph_sha256",
            "v3_ontology_bundle_sha256",
            "compared_counts",
            "allowed_identity_exceptions",
            "intersection_counts",
        },
        "identity audit report",
    )
    _require(
        payload["protocol"] == protocol.IDENTITY_REPORT_PROTOCOL
        and payload["status"] == "passed"
        and payload["dataset_version"] == protocol.DATASET_VERSION
        and payload["v2_stage0_registry_sha256"] == contract.v2_stage0_sha256,
        "identity audit protocol/status mismatch",
    )
    for key in (
        "v2_candidate_manifest_sha256",
        "v3_candidate_graph_sha256",
        "v3_ontology_bundle_sha256",
    ):
        _require(protocol.is_hex64(payload[key]), "identity audit hash invalid")
    compared = protocol.require_exact_keys(
        payload["compared_counts"],
        {
            "v2_candidates",
            "v3_graph_edges",
            "v3_fresh_sources",
            "v3_fresh_receivers",
            "v3_historical_receivers",
            "v3_original_source_nodes",
        },
        "identity compared counts",
    )
    _require(
        compared
        == {
            "v2_candidates": 48,
            "v3_graph_edges": 576,
            "v3_fresh_sources": 48,
            "v3_fresh_receivers": 56,
            "v3_historical_receivers": 8,
            "v3_original_source_nodes": 8,
        },
        "identity compared counts mismatch",
    )
    _require(
        payload["allowed_identity_exceptions"]
        == {"original_source_nodes": 8, "historical_receiver_nodes": 8},
        "identity exception counts mismatch",
    )
    intersections = protocol.require_exact_keys(
        payload["intersection_counts"],
        {
            "case_id",
            "canonical_record",
            "fresh_source_id",
            "fresh_receiver_id",
            "source_receiver_pair",
            "source_receiver_variant_triple",
        },
        "identity intersections",
    )
    _require(
        all(type(value) is int and value == 0 for value in intersections.values()),
        "identity audit intersection is nonzero",
    )
    _require(not protocol.contains_placeholder(payload), "identity audit contains placeholder")
    return payload


def _assert_aggregate_only(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            aggregate_intersection_label = (
                path == ("intersection_counts",)
                and key
                in {
                    "case_id",
                    "fresh_source_id",
                    "fresh_receiver_id",
                    "source_receiver_pair",
                    "source_receiver_variant_triple",
                }
                and type(item) is int
            )
            if key.casefold() in FORBIDDEN_OUTPUT_KEYS and not aggregate_intersection_label:
                raise ValueError("audit output contains a private-content field")
            _assert_aggregate_only(item, (*path, key))
    elif isinstance(value, list):
        for item in value:
            _assert_aggregate_only(item, path)


def build_identity_report(
    *,
    wrapper: Mapping[str, Any],
    v2_candidates_raw: bytes,
    v3_files: Mapping[str, bytes],
    contract: IdentityAuditContract,
) -> dict[str, Any]:
    candidate_record = _artifact_record(wrapper, "candidate_manifest_48", 48)
    _verify_committed_bytes(v2_candidates_raw, candidate_record, "v2 candidates")
    v2_rows = _validate_v2_candidates(v2_candidates_raw)
    sources = _validate_v3_sources(_json_object(v3_files[V3_SOURCE_BASENAME], "v3 sources"))
    receivers = _validate_v3_receivers(_json_object(v3_files[V3_RECEIVER_BASENAME], "v3 receivers"))
    historical = _validate_v3_historical(_json_object(v3_files[V3_HISTORICAL_BASENAME], "v3 historical anchors"))
    graph_rows = _validate_v3_graph(_json_object(v3_files[V3_GRAPH_BASENAME], "v3 candidate graph"))

    v2_case = {str(row["case_id"]) for row in v2_rows}
    v2_records = {str(row["canonical_record_sha256"]) for row in v2_rows}
    v2_sources = {str(row["source_id"]) for row in v2_rows}
    v2_original_sources = {
        str(row["source_id"])
        for row in v2_rows
        if row["source_membership"] == "original_source"
    }
    v2_source_phrases = {normalize_phrase(str(row["source_phrase"])) for row in v2_rows}
    v2_source_heads = {str(row["source_head_lemma"]) for row in v2_rows}
    v2_receivers = {str(row["receiver_id"]) for row in v2_rows}
    v2_receiver_phrases = {
        normalize_phrase(str(row["receiver_phrase"])) for row in v2_rows
    }
    v2_receiver_heads = {
        normalize_phrase(str(row["receiver_phrase"])).split()[-1]
        for row in v2_rows
    }
    v2_pairs = {(str(row["source_id"]), str(row["receiver_id"])) for row in v2_rows}
    v2_triples = {
        (str(row["source_id"]), str(row["receiver_id"]), str(row["prompt_variant"]))
        for row in v2_rows
    }
    fresh_sources = {str(row["source_id"]) for row in sources}
    fresh_receivers = {str(row["receiver_id"]) for row in receivers}
    historical_receivers = {str(row["receiver_id"]) for row in historical}
    source_by_id = {str(row["source_id"]): row for row in sources}
    receiver_by_id = {str(row["receiver_id"]): row for row in receivers}
    historical_by_id = {str(row["receiver_id"]): row for row in historical}
    v2_source_by_id: dict[str, Mapping[str, Any]] = {}
    for row in v2_rows:
        source_id = str(row["source_id"])
        previous = v2_source_by_id.setdefault(source_id, row)
        _require(
            previous["source_phrase"] == row["source_phrase"]
            and previous["source_head_lemma"] == row["source_head_lemma"],
            "v2 source identity is internally rebound",
        )
    graph_case = {str(row["case_id"]) for row in graph_rows}
    graph_records = {str(row["canonical_record_sha256"]) for row in graph_rows}
    graph_sources = {str(row["source_id"]) for row in graph_rows}
    graph_fresh_receivers = {
        str(row["receiver_id"])
        for row in graph_rows
        if row["receiver_membership"] == "new_receiver"
    }
    graph_pairs = {(str(row["source_id"]), str(row["receiver_id"])) for row in graph_rows}
    graph_triples = {
        (str(row["source_id"]), str(row["receiver_id"]), str(row["prompt_variant"]))
        for row in graph_rows
    }

    _require(fresh_sources <= graph_sources, "v3 graph omits a fresh source")
    original_sources = graph_sources - fresh_sources
    _require(
        len(original_sources) == 8 and original_sources <= v2_original_sources,
        "v3 original-source exception count is not eight",
    )
    _require(
        graph_fresh_receivers == fresh_receivers,
        "v3 fresh receiver graph binding mismatch",
    )
    _require(
        historical_receivers
        == {
            str(row["receiver_id"])
            for row in graph_rows
            if row["receiver_membership"] == "seen_receiver"
        },
        "v3 historical receiver graph binding mismatch",
    )
    for row in graph_rows:
        source_id = str(row["source_id"])
        if source_id in source_by_id:
            source = source_by_id[source_id]
            _require(
                row["source_phrase"] == source["source_phrase"]
                and row["source_head_lemma"] == source["head_lemma"],
                "v3 graph fresh source identity is rebound",
            )
        else:
            source = v2_source_by_id.get(source_id)
            _require(
                source is not None
                and row["source_phrase"] == source["source_phrase"]
                and row["source_head_lemma"] == source["source_head_lemma"],
                "v3 graph original source identity is rebound",
            )
        receiver_id = str(row["receiver_id"])
        if receiver_id in receiver_by_id:
            _require(
                row["receiver_phrase"] == receiver_by_id[receiver_id]["receiver_phrase"],
                "v3 graph fresh receiver identity is rebound",
            )
        else:
            historical_receiver = historical_by_id.get(receiver_id)
            _require(
                historical_receiver is not None
                and row["receiver_phrase"]
                == historical_receiver["receiver_phrase"],
                "v3 graph historical receiver identity is rebound",
            )
    intersections = {
        "case_id": len(v2_case & graph_case),
        "canonical_record": len(v2_records & graph_records),
        "fresh_source_id": sum(
            1
            for row in sources
            if row["source_id"] in v2_sources
            or row["normalized_phrase"] in v2_source_phrases
            or row["head_lemma"] in v2_source_heads
        ),
        "fresh_receiver_id": sum(
            1
            for row in receivers
            if row["receiver_id"] in v2_receivers
            or row["normalized_phrase"] in v2_receiver_phrases
            or row["head_lemma"] in v2_receiver_heads
        ),
        "source_receiver_pair": len(v2_pairs & graph_pairs),
        "source_receiver_variant_triple": len(v2_triples & graph_triples),
    }
    _require(all(value == 0 for value in intersections.values()), "v3/v2 identity intersection is nonzero")
    report = {
        "protocol": protocol.IDENTITY_REPORT_PROTOCOL,
        "status": "passed",
        "dataset_version": protocol.DATASET_VERSION,
        "v2_stage0_registry_sha256": contract.v2_stage0_sha256,
        "v2_candidate_manifest_sha256": sha256_bytes(v2_candidates_raw),
        "v3_candidate_graph_sha256": sha256_bytes(v3_files[V3_GRAPH_BASENAME]),
        "v3_ontology_bundle_sha256": _ontology_bundle_sha256(
            {
                name: v3_files[name]
                for name in (
                    V3_SOURCE_BASENAME,
                    V3_RECEIVER_BASENAME,
                    V3_HISTORICAL_BASENAME,
                )
            }
        ),
        "compared_counts": {
            "v2_candidates": 48,
            "v3_graph_edges": 576,
            "v3_fresh_sources": 48,
            "v3_fresh_receivers": 56,
            "v3_historical_receivers": 8,
            "v3_original_source_nodes": 8,
        },
        "allowed_identity_exceptions": {
            "original_source_nodes": 8,
            "historical_receiver_nodes": 8,
        },
        "intersection_counts": intersections,
    }
    validate_identity_report(report, contract)
    if contract == IdentityAuditContract():
        protocol.validate_identity_disjointness_report(report)
    _assert_aggregate_only(report)
    return report


def _output_parent_fd(
    project_root: Path, relative: Path
) -> tuple[int, list[int]]:
    root = _require_real_components(project_root)
    descriptors = [
        os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    ]
    current = descriptors[0]
    for component in relative.parts[:-1]:
        current = os.open(
            component,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        descriptors.append(current)
    return current, descriptors


def write_report_to_relative(
    project_root: Path, relative: Path, payload: Mapping[str, Any]
) -> str:
    _require(
        not relative.is_absolute() and relative.name and ".." not in relative.parts,
        "audit output relative path is invalid",
    )
    reject_forbidden_path(project_root, _absolute(project_root) / relative)
    _assert_aggregate_only(payload)
    raw = canonical_json_bytes(dict(payload))
    parent_fd, descriptors = _output_parent_fd(project_root, relative)
    basename = relative.name
    temporary = f".{basename}.tmp.{os.getpid()}"
    temporary_created = False
    try:
        try:
            os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("refusing to overwrite frozen audit report")
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
        temporary_created = True
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
            temporary_info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary,
            basename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        if (published.st_dev, published.st_ino) != (
            temporary_info.st_dev,
            temporary_info.st_ino,
        ):
            raise RuntimeError("audit report publication inode mismatch")
        os.fsync(parent_fd)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return sha256_bytes(raw)


def write_standard_report(project_root: Path, payload: Mapping[str, Any]) -> str:
    return write_report_to_relative(project_root, STANDARD_OUTPUT_RELATIVE, payload)


def run_identity_audit(
    *,
    project_root: Path,
    private_v2_root: Path,
    private_v3_root: Path,
    contract: IdentityAuditContract = IdentityAuditContract(),
    publish: bool = True,
) -> tuple[dict[str, Any], str | None]:
    validate_distinct_roots(project_root, private_v2_root, private_v3_root)
    wrapper, _ = load_v2_wrapper(project_root, contract)
    with SecurePrivateRoot(private_v2_root, V2_PRIVATE_ALLOWLIST) as v2_root:
        v2_candidates = v2_root.read_exact(V2_CANDIDATE_BASENAME)
    with SecurePrivateRoot(private_v3_root, V3_PRIVATE_ALLOWLIST) as v3_root:
        v3_files = {name: v3_root.read_exact(name) for name in sorted(V3_PRIVATE_ALLOWLIST)}
    report = build_identity_report(
        wrapper=wrapper,
        v2_candidates_raw=v2_candidates,
        v3_files=v3_files,
        contract=contract,
    )
    digest = write_standard_report(project_root, report) if publish else None
    return report, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--private-v2-root", type=Path, required=True)
    parser.add_argument("--private-v3-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report, digest = run_identity_audit(
        project_root=args.project_root,
        private_v2_root=args.private_v2_root,
        private_v3_root=args.private_v3_root,
    )
    print(
        canonical_json_bytes(
            {
                "status": report["status"],
                "output": STANDARD_OUTPUT_RELATIVE.as_posix(),
                "sha256": digest,
                "intersection_counts": report["intersection_counts"],
            }
        ).decode("ascii"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
