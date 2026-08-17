#!/usr/bin/env python3
"""Public, fail-closed interfaces for the v4_dev72_v3 causal data freeze.

This module is deliberately independent from every v2 Python entry point.  It
contains schemas and byte-level validation only; it never opens v2 private
data, media, reviews, seeds, salts, sealed data, or final36.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET = "causal"
DATASET_VERSION = "v4_dev72_v3"
DATA_ROOT = Path("data/water_impact_dynamic_v4")

PROTOCOL = "water_impact_dynamic_v4_eval_protocol_v3"
COMMITMENT_PROTOCOL = "water_impact_dynamic_v4_eval_commitment_registry_v3"
GRAPH_PROTOCOL = "water_impact_dynamic_v4_causal_candidate_graph_v3"
CANDIDATE_PROTOCOL = "water_impact_dynamic_v4_causal_candidate_manifest_v3"
SELECTOR_SUMMARY_PROTOCOL = "water_impact_dynamic_v4_causal_selector_summary_v3"
INVALID_OUTCOME_PROTOCOL = "water_impact_dynamic_v4_preflight_dataset_outcome_v3"
IDENTITY_REPORT_PROTOCOL = (
    "water_impact_dynamic_v4_v3_v2_identity_disjointness_audit_v1"
)
CONSTRUCT_REPORT_PROTOCOL = (
    "water_impact_dynamic_v4_v3_v2_construct_equivalence_audit_v1"
)
FORBIDDEN_SEED_SOURCE_AUDIT_PROTOCOL = (
    "water_impact_dynamic_v4_v3_forbidden_seed_source_audit_v1"
)
CODE_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_eval_code_registry_v3"

STAGE0_PUBLIC = DATA_ROOT / "causal_stage0_public_commitment_v3.json"
STAGE0_REGISTRY = DATA_ROOT / "causal_stage0_commitment_v3.json"
STAGE1_REGISTRY = DATA_ROOT / "causal_stage1_commitment_v3.json"
INVALID_OUTCOME = DATA_ROOT / "causal_preflight_dataset_invalid_v3.json"
CODE_REGISTRY = DATA_ROOT / "v4_eval_code_registry_v3.json"
IDENTITY_REPORT = DATA_ROOT / "v4_causal_identity_disjointness_v3.json"
CONSTRUCT_REPORT = DATA_ROOT / "v4_causal_v2_construct_equivalence_v3.json"
FORBIDDEN_SEED_SOURCE_AUDIT = (
    DATA_ROOT / "v4_causal_forbidden_seed_source_audit_v3.json"
)
V3_PUBLIC_PATHS_WITH_V2_LITERAL = {CONSTRUCT_REPORT.as_posix()}

V2_TERMINATION = Path("results/water_impact_dynamic_v4_causal_screening_termination_v2.md")
V2_BANK = DATA_ROOT / "source_bank_public64_registry_v2.json"
V2_MAPPING = DATA_ROOT / "source_mapping_v2.json"
V2_RUNTIME_READ_ALLOWLIST = {
    V2_TERMINATION.as_posix(): (
        "fc6171711a73f4a6eeb30d1f2d005439b7ff7fb7a91d064642fe5da02461ad77"
    ),
    V2_BANK.as_posix(): (
        "473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814"
    ),
    V2_MAPPING.as_posix(): (
        "6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2"
    ),
}
V2_STAGE0_SHA256 = "29696ad8031bb164fe1c6819c8c382d7e4e828835f750f0d245e4877d4167b38"
V2_FREEZE_SHA256 = "52ac630edd16d234742f4b3cdd75b840e2c2b3a3d2f73f23797fae91ecd59cb5"
V2_TEMPLATE_SHA256 = "76d3b2be61389a26cc5feb9b1211c5e7b0830a85369e27783fb56e5286ce0559"
V2_FIELD_RULES_SHA256 = "a1e23230b199a96e9f458c135e6ce2d18bf377966a6867dc5a2cca88d124e2ce"
V2_SELECTION_RULES_SHA256 = "aa41a6da40ae107fafa36ef96c18db6fc7446b9a504123aaaff9a0465f53ed36"

RANK_DOMAIN = "causal-selector-v3"
SEED_DOMAIN = "causal-eval-seed-v3"
GRAPH_ASSIGNMENT_DOMAIN = "causal-graph-receiver-permutation-v3"
GRAPH_ASSIGNMENT_COMMITMENT_NAME = "causal_graph_assignment_salt_v3"
SCREENING_COMMITMENT_NAME = "causal_screening_seed_v3"
SELECTOR_COMMITMENT_NAME = "causal_stage0_selector_salt_v3"
EVALUATION_COMMITMENT_NAME = "causal_evaluation_seed_salt_v3"
SCREENING_NAMESPACE = "v4-causal-stage0-screening-v3"
EVALUATION_NAMESPACE = "v4-causal-evaluation-v3"
PROMPT_VARIANTS = ("direct", "natural")
GROUPS = (
    "holdout_source_new_receiver",
    "holdout_source_seen_receiver",
    "seen_source_new_receiver",
)
CELL_ORDER = tuple((group, variant) for group in GROUPS for variant in PROMPT_VARIANTS)
CELL_COUNTS = {
    ("holdout_source_new_receiver", "direct"): 48,
    ("holdout_source_new_receiver", "natural"): 168,
    ("holdout_source_seen_receiver", "direct"): 24,
    ("holdout_source_seen_receiver", "natural"): 24,
    ("seen_source_new_receiver", "direct"): 96,
    ("seen_source_new_receiver", "natural"): 216,
}
CANDIDATE_COUNT = 576
SELECTED_COUNT = 24
UNIT_COUNT = 72
REPLICATES = (0, 1, 2)

R1_DIRECT_OFFSETS = (0, 11)
R1_NATURAL_OFFSETS = (0, 3, 7, 11, 15, 19, 22)

GRAPH_EDGE_KEYS = (
    "case_id",
    "group",
    "prompt_variant",
    "physical_anchor_id",
    "edge_ordinal",
    "source_membership",
    "source_id",
    "source_phrase",
    "source_head_lemma",
    "receiver_membership",
    "receiver_id",
    "receiver_phrase",
    "canonical_prompt",
    "canonical_record_sha256",
)
ELIGIBILITY_FIELDS = (
    "source_visibility",
    "footprint_visibility",
    "receiver",
    "quality",
    "causal_link",
)

STAGE0_ARTIFACT_ROWS: dict[str, int | None | str] = {
    "candidate_manifest_576": 576,
    "upstream_source_bank_registry_64_v2": 64,
    "upstream_source_mapping_178_v2": 178,
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
    "selector_salt": None,
    "ranking_formula": None,
    "constrained_subset_algorithm": None,
    "evaluation_seed_salt": None,
    "seed_derivation_formula": None,
    "forbidden_seed_inventory": "positive",
    "forbidden_seed_source_audit": None,
    "preselection_seed_audit_1728": 1728,
    "selection_binding": None,
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
    "preregistration": None,
    "v2_public_aggregate_design_input": 6,
}
STAGE1_ARTIFACT_ROWS: dict[str, int | None | str] = {
    "screening_generation_manifest_576": 576,
    "screening_raw_video_inventory_576": 576,
    "screening_candidate_binding_576": 576,
    "screening_anonymous_video_inventory_576": 576,
    "screening_composite_inventory_576": 576,
    "screening_public_package_manifest_576": 576,
    "screening_private_package_manifest_576": 576,
    "screening_package_commitment": None,
    "screening_review_template_576": 576,
    "screening_review_a_576": 576,
    "screening_review_b_576": 576,
    "screening_dispute_manifest": "disputes",
    "screening_adjudication": "disputes",
    "screening_adjudication_audit": "disputes",
    "screening_freeze_manifest": None,
    "eligibility_table_576": 576,
    "selector_summary": None,
    "selected_case_manifest_24": 24,
    "unit_manifest_U_72": 72,
}

INVALID_PHASES = {
    "stage0_authorization",
    "original_generation",
    "screening_package",
    "screening_review",
    "screening_freeze",
    "selection",
    "stage1_publication",
}
INVALID_REASON_CODES = {
    "stage0_authorization_integrity_failure",
    "screening_generation_incomplete",
    "screening_package_integrity_failure",
    "screening_review_coverage_failure",
    "screening_adjudication_integrity_failure",
    "screening_cell_quota_infeasible",
    "screening_anchor_coverage_infeasible",
    "selection_rank_tie",
    "global_subset_infeasible",
    "seed_contract_failure",
    "stage1_publication_failure",
}
INVALID_BOUND_ARTIFACT_KEYS = {
    "stage0_registry",
    "screening_generation_manifest",
    "screening_package_commitment",
    "screening_freeze_manifest",
    "canonical_eligibility",
    "selector_stderr",
}

CODE_ARTIFACT_PATHS = {
    "protocol": "scripts/water_impact_dynamic_v4_eval_protocol_v3.py",
    "candidate_builder": "scripts/build_water_impact_dynamic_v4_causal_candidates_v3.py",
    "stage0_authorizer": "scripts/authorize_water_impact_dynamic_v4_causal_stage0_v3.py",
    "screening_runner": "scripts/run_water_impact_dynamic_v4_causal_screening_v3.py",
    "screening_freezer": "scripts/freeze_water_impact_dynamic_v4_causal_screening_v3.py",
    "selector": "scripts/select_water_impact_dynamic_v4_causal_v3.py",
    "validator": "scripts/validate_water_impact_dynamic_v4_causal_v3.py",
    "capacity_validator": "scripts/validate_water_impact_dynamic_v4_causal_capacity_v3.py",
    "identity_disjointness_auditor": (
        "scripts/audit_water_impact_dynamic_v4_v3_v2_disjointness.py"
    ),
    "construct_equivalence_auditor": (
        "scripts/audit_water_impact_dynamic_v4_v3_v2_construct_equivalence.py"
    ),
    "forbidden_seed_auditor": (
        "scripts/audit_water_impact_dynamic_v4_v3_forbidden_seeds.py"
    ),
    "tests": "tests/test_water_impact_dynamic_v4_causal_v3.py",
    "generator": "scripts/generate_wan_clean.py",
}

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PLACEHOLDER_TOKENS = ("placeholder", "todo", "tbd", "fill_me", "to_be_frozen")
FORBIDDEN_PATH_TOKENS = ("sealed", "final36", "quarantine")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_exact_keys(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label}: must be an object")
    require(set(value) == set(keys), f"{label}: fields are not exact")
    return value


def contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(contains_placeholder(key) or contains_placeholder(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        return not lowered or any(token in lowered for token in PLACEHOLDER_TOKENS)
    return False


def reject_forbidden_path(*paths: Path) -> None:
    for path in paths:
        lexical = Path(os.path.abspath(os.fspath(path)))
        resolved = lexical.resolve(strict=False)
        for candidate in (lexical, resolved):
            lowered = candidate.as_posix().casefold()
            if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
                raise ValueError("sealed/final36/quarantine paths are forbidden")


def _canonical_lexical_absolute(path: Path) -> Path:
    raw = os.fspath(path)
    require(isinstance(raw, str) and raw, "path must be a nonempty string")
    require(
        raw == os.path.normpath(raw),
        "path contains a noncanonical lexical alias",
    )
    parts = Path(raw).parts
    require(".." not in parts, "path contains a parent-directory alias")
    return Path(os.path.abspath(raw))


def _require_no_symlink_components(path: Path, *, leaf_may_be_absent: bool = False) -> Path:
    """Require every existing lexical path component to be a real directory/file."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        current = current / component
        is_leaf = index == len(parts) - 1
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if leaf_may_be_absent and is_leaf:
                return absolute
            raise
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"path contains a symlink component: {current}")
        if not is_leaf and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"path ancestor is not a directory: {current}")
    return absolute


def validate_project_root(project_root: Path) -> Path:
    """Resolve an existing project root only after checking every ancestor."""

    reject_forbidden_path(project_root)
    lexical = _require_no_symlink_components(
        _canonical_lexical_absolute(project_root)
    )
    info = os.lstat(lexical)
    require(stat.S_ISDIR(info.st_mode), "project root must be a real directory")
    return lexical.resolve(strict=True)


def _relative_path(project_root: Path, path: Path) -> str:
    root = validate_project_root(project_root)
    lexical = _canonical_lexical_absolute(path)
    try:
        return lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path is lexically outside the project root") from exc


def validate_runtime_read_path(project_root: Path, path: Path, *, allow_v2: bool = False) -> str:
    reject_forbidden_path(path)
    root = validate_project_root(project_root)
    lexical = _canonical_lexical_absolute(path)
    relative = _relative_path(project_root, path)
    if "_v2" in relative.casefold():
        if relative in V3_PUBLIC_PATHS_WITH_V2_LITERAL:
            require(not allow_v2, "v3 construct report is not a v2 upstream")
        elif not allow_v2 or relative not in V2_RUNTIME_READ_ALLOWLIST:
            raise ValueError("v3 runtime attempted a nonallowlisted v2 read")
    require(
        lexical == root / Path(relative),
        "runtime path is not the exact project-root-relative lexical path",
    )
    _require_no_symlink_components(lexical)
    require(
        lexical.resolve(strict=True) == lexical,
        "runtime path resolves through a noncanonical alias",
    )
    _require_regular_file(lexical, single_link=True)
    return relative


def _require_regular_file(path: Path, *, mode: int | None = None, single_link: bool = False) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink regular file: {path}")
    info = path.stat()
    if mode is not None and stat.S_IMODE(info.st_mode) != mode:
        raise PermissionError(f"file mode must be {mode:o}: {path}")
    if single_link and info.st_nlink != 1:
        raise PermissionError(f"hardlinks are forbidden: {path}")


def validate_private_path(private_root: Path, path: Path, *, must_exist: bool = True) -> Path:
    reject_forbidden_path(private_root, path)
    root_lexical = _require_no_symlink_components(private_root)
    path_lexical = Path(os.path.abspath(os.fspath(path)))
    if "_v2" in path_lexical.as_posix().casefold():
        raise ValueError("v3 runtime private paths may not reference v2")
    root_info = os.lstat(root_lexical)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o700:
        raise PermissionError("PRIVATE_V3_ROOT must be a mode-700 non-symlink directory")
    try:
        path_lexical.relative_to(root_lexical)
    except ValueError as exc:
        raise ValueError("private path is lexically outside PRIVATE_V3_ROOT") from exc
    _require_no_symlink_components(path_lexical, leaf_may_be_absent=not must_exist)
    root = root_lexical.resolve(strict=True)
    resolved = path_lexical.resolve(strict=must_exist)
    reject_forbidden_path(root, resolved)
    if "_v2" in resolved.as_posix().casefold():
        raise ValueError("v3 runtime private paths may not resolve through v2")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("private path resolves outside PRIVATE_V3_ROOT") from exc
    if must_exist:
        relative = path_lexical.relative_to(root_lexical)
        current = root_lexical
        for component in relative.parts[:-1]:
            current = current / component
            info = os.lstat(current)
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
                raise PermissionError(f"private directory mode must be 700: {current}")
        _require_regular_file(path_lexical, mode=0o600, single_link=True)
    return resolved


def validate_private_output_path(private_root: Path, path: Path) -> Path:
    """Validate one absent private output below real mode-700 directories."""

    resolved = validate_private_path(private_root, path, must_exist=False)
    lexical = Path(os.path.abspath(os.fspath(path)))
    if os.path.lexists(lexical):
        raise FileExistsError(f"refusing to overwrite private output: {lexical}")
    parent = lexical.parent
    _require_no_symlink_components(parent)
    root_lexical = Path(os.path.abspath(os.fspath(private_root)))
    relative_parent = parent.relative_to(root_lexical)
    current = root_lexical
    for component in relative_parent.parts:
        current = current / component
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError(
                f"private output directory mode must be 700: {current}"
            )
    reject_forbidden_path(lexical, resolved, parent.resolve(strict=True))
    return resolved


def load_json(
    path: Path,
    *,
    project_root: Path | None = None,
    allow_v2: bool = False,
    private_root: Path | None = None,
) -> dict[str, Any]:
    if project_root is not None:
        validate_runtime_read_path(project_root, path, allow_v2=allow_v2)
    if private_root is not None:
        validate_private_path(private_root, path)
    else:
        _require_regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def write_json_exclusive_atomic(path: Path, payload: Mapping[str, Any], *, mode: int = 0o600) -> str:
    reject_forbidden_path(path)
    absolute = Path(os.path.abspath(os.fspath(path)))
    _require_no_symlink_components(absolute.parent)
    if os.path.lexists(absolute):
        raise FileExistsError(f"refusing to overwrite: {path}")
    raw = canonical_json_bytes(dict(payload))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{absolute.name}.", dir=absolute.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(absolute):
            raise FileExistsError(f"refusing to overwrite: {path}")
        os.link(temporary, absolute)
        temporary.unlink()
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return sha256_bytes(raw)


def validate_v2_public_inputs(project_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in V2_RUNTIME_READ_ALLOWLIST.items():
        path = project_root / relative
        validate_runtime_read_path(project_root, path, allow_v2=True)
        _require_regular_file(path)
        actual = sha256_file(path)
        require(actual == expected, f"v2 public input hash mismatch: {relative}")
        observed[relative] = actual
    termination = (project_root / V2_TERMINATION).read_text(encoding="utf-8")
    require(V2_STAGE0_SHA256 in termination, "termination record does not bind v2 Stage-0")
    require(V2_FREEZE_SHA256 in termination, "termination record does not bind v2 freeze")
    return observed


def candidate_record_bytes(row: Mapping[str, Any]) -> bytes:
    require(set(row) == set(GRAPH_EDGE_KEYS), "candidate record fields are not exact")
    base = dict(row)
    digest = base.pop("canonical_record_sha256")
    require(is_hex64(digest), "candidate canonical record hash is invalid")
    raw = canonical_json_bytes(base)
    require(sha256_bytes(raw) == digest, "candidate canonical record hash mismatch")
    return raw


def validate_lower_hex_salt(value: str, label: str) -> str:
    require(is_hex64(value), f"{label} must be lower hex64")
    return value


def validate_secret_separation(
    *,
    graph_assignment_salt: str,
    selector_salt: str,
    evaluation_salt: str,
    screening_seed: int,
) -> None:
    salts = (
        validate_lower_hex_salt(graph_assignment_salt, "graph assignment salt"),
        validate_lower_hex_salt(selector_salt, "selector salt"),
        validate_lower_hex_salt(evaluation_salt, "evaluation seed salt"),
    )
    require(len(set(salts)) == 3, "graph/selector/evaluation salts must be pairwise distinct")
    require(
        isinstance(screening_seed, int)
        and not isinstance(screening_seed, bool)
        and 0 <= screening_seed < 2**32,
        "screening seed must be uint32",
    )
    require(str(screening_seed) not in salts, "screening seed must differ from every salt")


def selection_rank(row: Mapping[str, Any], selector_salt: str) -> str:
    salt = validate_lower_hex_salt(selector_salt, "selector salt")
    payload = (
        RANK_DOMAIN.encode("utf-8")
        + b"\x00"
        + salt.encode("utf-8")
        + b"\x00"
        + candidate_record_bytes(row)
    )
    return sha256_bytes(payload)


def derive_evaluation_seed(evaluation_salt: str, case_id: str, replicate: int) -> int:
    salt = validate_lower_hex_salt(evaluation_salt, "evaluation seed salt")
    require(isinstance(case_id, str) and case_id and case_id.strip() == case_id, "case_id invalid")
    require(isinstance(replicate, int) and not isinstance(replicate, bool) and replicate in REPLICATES, "replicate invalid")
    payload = (
        SEED_DOMAIN.encode("utf-8")
        + b"\x00"
        + salt.encode("utf-8")
        + b"\x00"
        + case_id.encode("utf-8")
        + b"\x00"
        + str(replicate).encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _validate_artifact_records(
    artifacts: Any, expected: Mapping[str, int | None | str], label: str
) -> dict[str, Mapping[str, Any]]:
    require(isinstance(artifacts, dict) and set(artifacts) == set(expected), f"{label}: artifact inventory is not exact")
    dispute_count: int | None = None
    output: dict[str, Mapping[str, Any]] = {}
    for name, expected_rows in expected.items():
        record = require_exact_keys(artifacts[name], {"sha256", "size_bytes", "row_count"}, f"{label}/{name}")
        require(is_hex64(record["sha256"]), f"{label}/{name}: sha256 invalid")
        require(isinstance(record["size_bytes"], int) and not isinstance(record["size_bytes"], bool) and record["size_bytes"] > 0, f"{label}/{name}: size invalid")
        rows = record["row_count"]
        if expected_rows is None:
            require(rows is None, f"{label}/{name}: row_count must be null")
        elif expected_rows == "positive":
            require(isinstance(rows, int) and not isinstance(rows, bool) and rows > 0, f"{label}/{name}: row_count must be positive")
        elif expected_rows == "disputes":
            require(isinstance(rows, int) and not isinstance(rows, bool) and 0 <= rows <= 2880, f"{label}/{name}: dispute count invalid")
            if dispute_count is None:
                dispute_count = rows
            require(rows == dispute_count, f"{label}: dispute/adjudication counts differ")
        else:
            require(rows == expected_rows, f"{label}/{name}: row_count mismatch")
        output[name] = record
    return output


def validate_commitment_registry(
    payload: Mapping[str, Any], *, stage: int, expected_stage0_sha256: str | None = None
) -> Mapping[str, Any]:
    require(
        type(stage) is int and stage in (0, 1),
        "commitment stage must be exactly 0 or 1",
    )
    keys = {
        "protocol",
        "dataset",
        "dataset_version",
        "stage",
        "status",
        "sealed_final36_status",
        "artifacts",
    }
    if stage == 1:
        keys.add("stage0_registry_sha256")
    require_exact_keys(payload, keys, f"causal Stage-{stage}")
    require(payload["protocol"] == COMMITMENT_PROTOCOL, "commitment protocol mismatch")
    require(payload["dataset"] == DATASET and payload["dataset_version"] == DATASET_VERSION, "commitment dataset/version mismatch")
    require(
        type(payload["stage"]) is int
        and payload["stage"] == stage
        and payload["status"] == "committed",
        "commitment stage/status mismatch",
    )
    require(payload["sealed_final36_status"] == "unopened", "sealed-final36 must remain unopened")
    if stage == 1:
        require(is_hex64(payload["stage0_registry_sha256"]), "Stage-1 Stage-0 hash invalid")
        require(expected_stage0_sha256 is not None and payload["stage0_registry_sha256"] == expected_stage0_sha256, "Stage-1 does not bind exact Stage-0 bytes")
    _validate_artifact_records(
        payload["artifacts"], STAGE0_ARTIFACT_ROWS if stage == 0 else STAGE1_ARTIFACT_ROWS, f"Stage-{stage}"
    )
    require(not contains_placeholder(payload), "commitment contains placeholder content")
    return payload


def validate_identity_disjointness_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    require_exact_keys(
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
        "identity disjointness report",
    )
    require(payload["protocol"] == IDENTITY_REPORT_PROTOCOL and payload["status"] == "passed" and payload["dataset_version"] == DATASET_VERSION, "identity report protocol/status mismatch")
    for key in (
        "v2_stage0_registry_sha256",
        "v2_candidate_manifest_sha256",
        "v3_candidate_graph_sha256",
        "v3_ontology_bundle_sha256",
    ):
        require(is_hex64(payload[key]), f"identity report {key} invalid")
    require(payload["v2_stage0_registry_sha256"] == V2_STAGE0_SHA256, "identity report v2 root mismatch")
    counts = require_exact_keys(
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
    require(
        counts
        == {
            "v2_candidates": 48,
            "v3_graph_edges": 576,
            "v3_fresh_sources": 48,
            "v3_fresh_receivers": 56,
            "v3_historical_receivers": 8,
            "v3_original_source_nodes": 8,
        },
        "identity compared counts invalid",
    )
    exceptions = require_exact_keys(payload["allowed_identity_exceptions"], {"original_source_nodes", "historical_receiver_nodes"}, "identity exceptions")
    require(exceptions == {"original_source_nodes": 8, "historical_receiver_nodes": 8}, "identity exceptions mismatch")
    intersections = require_exact_keys(
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
    require(all(value == 0 and isinstance(value, int) and not isinstance(value, bool) for value in intersections.values()), "identity intersection is nonzero")
    require(not contains_placeholder(payload), "identity report contains placeholder")
    return payload


def validate_construct_equivalence_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    require_exact_keys(
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
        "construct equivalence report",
    )
    require(payload["protocol"] == CONSTRUCT_REPORT_PROTOCOL and payload["status"] == "passed" and payload["dataset_version"] == DATASET_VERSION, "construct report protocol/status mismatch")
    require(payload["v2_stage0_registry_sha256"] == V2_STAGE0_SHA256, "construct report v2 root mismatch")
    for name in ("v2_file_sha256", "v3_file_sha256"):
        values = require_exact_keys(payload[name], {"templates", "field_rules", "selection_rules"}, name)
        require(all(is_hex64(value) for value in values.values()), f"{name}: hash invalid")
    require(payload["v2_file_sha256"]["templates"] == V2_TEMPLATE_SHA256, "v2 template hash mismatch")
    require(payload["v2_file_sha256"]["field_rules"] == V2_FIELD_RULES_SHA256, "v2 field-rules hash mismatch")
    require(payload["v2_file_sha256"]["selection_rules"] == V2_SELECTION_RULES_SHA256, "v2 selection-rules hash mismatch")
    for name in ("qualification_sha256", "cell_quota_sha256"):
        values = require_exact_keys(payload[name], {"v2", "v3"}, name)
        require(all(is_hex64(value) for value in values.values()) and values["v2"] == values["v3"], f"{name}: equivalence mismatch")
    equal = require_exact_keys(payload["exact_equal"], {"templates", "field_rules", "qualification", "cell_quota"}, "construct equality flags")
    require(all(value is True for value in equal.values()), "construct equality flag failed")
    require(payload["v3_file_sha256"]["templates"] == V2_TEMPLATE_SHA256, "v3 templates are not byte-equal")
    require(payload["v3_file_sha256"]["field_rules"] == V2_FIELD_RULES_SHA256, "v3 field rules are not byte-equal")
    require(not contains_placeholder(payload), "construct report contains placeholder")
    return payload


def validate_selector_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    require_exact_keys(
        payload,
        {
            "protocol",
            "dataset_version",
            "status",
            "candidate_count",
            "eligible_count",
            "cell_eligible_counts",
            "selected_count",
            "unit_count",
            "stage0_registry_sha256",
            "screening_freeze_sha256",
            "eligibility_table_sha256",
            "selected_case_manifest_sha256",
            "unit_manifest_sha256",
            "selection_rank_tuple_sha256",
            "constraints",
        },
        "selector summary",
    )
    require(payload["protocol"] == SELECTOR_SUMMARY_PROTOCOL and payload["dataset_version"] == DATASET_VERSION and payload["status"] == "selected", "selector summary protocol/status mismatch")
    require(payload["candidate_count"] == CANDIDATE_COUNT and payload["selected_count"] == SELECTED_COUNT and payload["unit_count"] == UNIT_COUNT, "selector summary counts mismatch")
    require(isinstance(payload["eligible_count"], int) and not isinstance(payload["eligible_count"], bool) and SELECTED_COUNT <= payload["eligible_count"] <= CANDIDATE_COUNT, "selector eligible count invalid")
    cells = payload["cell_eligible_counts"]
    require(isinstance(cells, dict) and set(cells) == {f"{group}:{variant}" for group, variant in CELL_ORDER}, "selector cell counts not exact")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 4 for value in cells.values()), "selector cell eligibility below quota")
    for key in (
        "stage0_registry_sha256",
        "screening_freeze_sha256",
        "eligibility_table_sha256",
        "selected_case_manifest_sha256",
        "unit_manifest_sha256",
        "selection_rank_tuple_sha256",
    ):
        require(is_hex64(payload[key]), f"selector summary {key} invalid")
    constraints = require_exact_keys(
        payload["constraints"],
        {
            "cell_quota_pass",
            "g1_distinct_head_pass",
            "g2_anchor_coverage_pass",
            "g3_anchor_coverage_pass",
            "original_source_coverage_pass",
            "holdout_head_uniqueness_pass",
            "receiver_uniqueness_pass",
            "rank_tie_free",
            "seed_contract_pass",
        },
        "selector constraints",
    )
    require(all(value is True for value in constraints.values()), "selector constraint failed")
    require(not contains_placeholder(payload), "selector summary contains placeholder")
    return payload


def validate_invalid_outcome(
    payload: Mapping[str, Any], *, expected_stage0_sha256: str | None = None
) -> Mapping[str, Any]:
    require_exact_keys(
        payload,
        {
            "protocol",
            "dataset",
            "dataset_version",
            "status",
            "failure_phase",
            "reason_code",
            "stage0_registry_sha256",
            "candidate_count",
            "eligible_count",
            "cell_eligible_counts",
            "selector_output_created",
            "unit_manifest_created",
            "stage1_registry_created",
            "sealed_final36_status",
            "bound_artifacts",
        },
        "invalid outcome",
    )
    require(payload["protocol"] == INVALID_OUTCOME_PROTOCOL and payload["dataset"] == DATASET and payload["dataset_version"] == DATASET_VERSION and payload["status"] == "preflight_dataset_invalid", "invalid outcome protocol/status mismatch")
    require(payload["failure_phase"] in INVALID_PHASES and payload["reason_code"] in INVALID_REASON_CODES, "invalid outcome phase/reason invalid")
    stage0_hash = payload["stage0_registry_sha256"]
    if stage0_hash is None:
        require(payload["failure_phase"] == "stage0_authorization" and payload["reason_code"] == "stage0_authorization_integrity_failure", "null Stage-0 hash allowed only for scientific authorization failure")
        require(expected_stage0_sha256 is None, "pre-wrapper failure may not receive a Stage-0 hash")
    else:
        require(is_hex64(stage0_hash), "invalid outcome Stage-0 hash invalid")
        require(
            is_hex64(expected_stage0_sha256)
            and stage0_hash == expected_stage0_sha256,
            "invalid outcome does not bind exact standard Stage-0 bytes",
        )
    require(payload["candidate_count"] == CANDIDATE_COUNT, "invalid outcome candidate count mismatch")
    eligible = payload["eligible_count"]
    cells = payload["cell_eligible_counts"]
    if eligible is None or cells is None:
        require(eligible is None and cells is None, "invalid outcome eligibility fields must be both null")
    else:
        require(isinstance(eligible, int) and not isinstance(eligible, bool) and 0 <= eligible <= CANDIDATE_COUNT, "invalid outcome eligible count invalid")
        require(isinstance(cells, dict) and set(cells) == {f"{group}:{variant}" for group, variant in CELL_ORDER}, "invalid outcome cell inventory invalid")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in cells.values()) and sum(cells.values()) == eligible, "invalid outcome cell counts invalid")
    require(payload["selector_output_created"] is False and payload["unit_manifest_created"] is False and payload["stage1_registry_created"] is False, "invalid outcome may not claim outputs")
    require(payload["sealed_final36_status"] == "unopened", "invalid outcome sealed status invalid")
    bound = require_exact_keys(payload["bound_artifacts"], INVALID_BOUND_ARTIFACT_KEYS, "invalid outcome bound artifacts")
    require(bound["stage0_registry"] == stage0_hash, "invalid outcome Stage-0 cross-binding mismatch")
    require(all(value is None or is_hex64(value) for value in bound.values()), "invalid outcome bound artifact hash invalid")
    phase = payload["failure_phase"]
    reasons_by_phase = {
        "stage0_authorization": {"stage0_authorization_integrity_failure"},
        "original_generation": {"screening_generation_incomplete"},
        "screening_package": {"screening_package_integrity_failure"},
        "screening_review": {"screening_review_coverage_failure"},
        "screening_freeze": {"screening_adjudication_integrity_failure"},
        "selection": {
            "screening_cell_quota_infeasible",
            "screening_anchor_coverage_infeasible",
            "selection_rank_tie",
            "global_subset_infeasible",
            "seed_contract_failure",
        },
        "stage1_publication": {"stage1_publication_failure"},
    }
    require(
        payload["reason_code"] in reasons_by_phase[phase],
        "invalid outcome reason does not match failure phase",
    )
    required_by_phase = {
        "stage0_authorization": set(),
        "original_generation": {"stage0_registry"},
        "screening_package": {
            "stage0_registry",
            "screening_generation_manifest",
        },
        "screening_review": {
            "stage0_registry",
            "screening_generation_manifest",
            "screening_package_commitment",
        },
        "screening_freeze": {
            "stage0_registry",
            "screening_generation_manifest",
            "screening_package_commitment",
        },
        "selection": set(INVALID_BOUND_ARTIFACT_KEYS),
        "stage1_publication": set(INVALID_BOUND_ARTIFACT_KEYS),
    }
    required = required_by_phase[phase]
    require(
        {name for name, value in bound.items() if value is not None} == required,
        "invalid outcome bound-artifact phase matrix mismatch",
    )
    after_freeze = phase in {"selection", "stage1_publication"}
    require(
        (eligible is not None and cells is not None) == after_freeze,
        "invalid outcome eligibility fields violate failure phase",
    )
    require(not contains_placeholder(payload), "invalid outcome contains placeholder/free text")
    return payload


def validate_code_registry(payload: Mapping[str, Any], project_root: Path) -> Mapping[str, Any]:
    project_root = validate_project_root(project_root)
    require_exact_keys(payload, {"protocol", "status", "dataset_version", "v2_read_allowlist", "artifacts"}, "v3 code registry")
    require(payload["protocol"] == CODE_REGISTRY_PROTOCOL and payload["status"] == "frozen" and payload["dataset_version"] == DATASET_VERSION, "code registry protocol/status mismatch")
    reads = payload["v2_read_allowlist"]
    require(isinstance(reads, dict) and reads == V2_RUNTIME_READ_ALLOWLIST, "code registry v2 read allowlist mismatch")
    artifacts = payload["artifacts"]
    require(isinstance(artifacts, dict) and set(artifacts) == set(CODE_ARTIFACT_PATHS), "code registry artifact inventory mismatch")
    for name, expected_path in CODE_ARTIFACT_PATHS.items():
        record = require_exact_keys(artifacts[name], {"path", "sha256"}, f"code registry/{name}")
        require(record["path"] == expected_path and is_hex64(record["sha256"]), f"code registry/{name} mismatch")
        path = project_root / expected_path
        _require_no_symlink_components(path)
        _require_regular_file(path, single_link=True)
        require(sha256_file(path) == record["sha256"], f"code registry/{name} byte drift")
    require(not contains_placeholder(payload), "code registry contains placeholder")
    return payload


def validate_no_v2_imports(paths: Sequence[Path], project_root: Path) -> None:
    project_root = validate_project_root(project_root)
    allowed_literals = set(V2_RUNTIME_READ_ALLOWLIST) | set(V2_RUNTIME_READ_ALLOWLIST.values()) | {
        V2_STAGE0_SHA256,
        V2_FREEZE_SHA256,
        V2_TEMPLATE_SHA256,
        V2_FIELD_RULES_SHA256,
        V2_SELECTION_RULES_SHA256,
        "_v2",
        "validate_v2_public_inputs",
        "validate_no_v2_imports",
        IDENTITY_REPORT_PROTOCOL,
        CONSTRUCT_REPORT_PROTOCOL,
        CONSTRUCT_REPORT.name,
    }
    allowed_literals.update(Path(value).name for value in V2_RUNTIME_READ_ALLOWLIST)
    allowed_literals.update(name for name in STAGE0_ARTIFACT_ROWS if "_v2" in name)
    allowed_literals.update(value for value in CODE_ARTIFACT_PATHS.values() if "_v2" in value)

    def static_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = static_string(node.left)
            right = static_string(node.right)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.JoinedStr) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in node.values
        ):
            return "".join(item.value for item in node.values)
        return None

    local_prefixes = (
        "scripts",
        "tests",
        "water_impact_dynamic_v4",
        "build_water_impact_dynamic_v4",
        "select_water_impact_dynamic_v4",
        "validate_water_impact_dynamic_v4",
        "authorize_water_impact_dynamic_v4",
        "run_water_impact_dynamic_v4",
        "freeze_water_impact_dynamic_v4",
        "audit_water_impact_dynamic_v4",
    )

    def module_candidates(current: Path, module: str) -> tuple[Path, ...]:
        parts = tuple(item for item in module.split(".") if item)
        if not parts:
            return ()
        roots = (project_root, current.parent)
        output: list[Path] = []
        for root in roots:
            base = root.joinpath(*parts)
            output.extend((base.with_suffix(".py"), base / "__init__.py"))
        unique: list[Path] = []
        seen: set[Path] = set()
        for candidate in output:
            lexical = _canonical_lexical_absolute(candidate)
            if lexical not in seen:
                seen.add(lexical)
                unique.append(lexical)
        return tuple(unique)

    def resolve_local_module(current: Path, module: str) -> Path | None:
        for candidate in module_candidates(current, module):
            try:
                candidate.relative_to(project_root)
            except ValueError:
                continue
            if candidate.is_file():
                _require_no_symlink_components(candidate)
                _require_regular_file(candidate, single_link=True)
                return candidate
        return None

    queue = [_canonical_lexical_absolute(path) for path in paths]
    visited: set[Path] = set()
    while queue:
        path = queue.pop(0)
        if path in visited:
            continue
        visited.add(path)
        _require_no_symlink_components(path)
        _require_regular_file(path, single_link=True)
        relative = _relative_path(project_root, path)
        require("_v2" not in Path(relative).name.casefold(), "v3 code path aliases a v2 entry point")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            require(not any("_v2" in name.casefold() for name in names), f"v2 import forbidden in {relative}")
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = []
                base = node.module or ""
                if node.level:
                    current_parts = list(path.relative_to(project_root).with_suffix("").parts[:-1])
                    keep = len(current_parts) - node.level + 1
                    require(keep >= 0, f"relative import escapes project root in {relative}")
                    prefix = ".".join(current_parts[:keep])
                    base = ".".join(item for item in (prefix, base) if item)
                if base:
                    modules.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        child = ".".join(item for item in (base, alias.name) if item)
                        modules.append(child)
            else:
                modules = []
            for module in modules:
                local = resolve_local_module(path, module)
                if local is not None:
                    if local not in visited:
                        queue.append(local)
                    continue
                top = module.split(".", 1)[0]
                local_directory = project_root / top
                if module.startswith(local_prefixes) or local_directory.exists():
                    if not local_directory.exists():
                        raise FileNotFoundError(
                            f"repo-local import is missing: {module} from {relative}"
                        )
                    require(
                        local_directory.is_dir() and not local_directory.is_symlink(),
                        f"repo-local import root invalid: {module}",
                    )
                    if module == top:
                        continue
                    if not (
                        local_directory.is_dir()
                        and any(
                            candidate.is_file()
                            for candidate in module_candidates(path, module)
                        )
                    ):
                        raise FileNotFoundError(
                            f"repo-local import is missing: {module} from {relative}"
                        )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"}:
                    raise ValueError(f"dynamic import/evaluation forbidden in {relative}")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    raise ValueError(f"dynamic import forbidden in {relative}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if "_v2" in value.casefold():
                    require(value in allowed_literals, f"nonallowlisted v2 literal in {relative}")
            if isinstance(node, (ast.BinOp, ast.JoinedStr)):
                value = static_string(node)
                if value is not None and "_v2" in value.casefold():
                    require(value in allowed_literals, f"constructed nonallowlisted v2 literal in {relative}")


def verify_file_record(path: Path, record: Mapping[str, Any], *, expected_rows: int | None = None) -> None:
    require_exact_keys(record, {"path", "sha256", "size_bytes", "row_count"}, "file record")
    _require_no_symlink_components(path)
    _require_regular_file(path, single_link=True)
    raw_size = path.stat().st_size
    require(record["path"] == path.as_posix(), "file record path mismatch")
    require(record["sha256"] == sha256_file(path) and record["size_bytes"] == raw_size, "file record byte mismatch")
    require(record["row_count"] == expected_rows, "file record row count mismatch")


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "is_hex64",
    "require",
    "require_exact_keys",
    "contains_placeholder",
    "reject_forbidden_path",
    "validate_project_root",
    "validate_runtime_read_path",
    "validate_private_path",
    "validate_private_output_path",
    "load_json",
    "write_json_exclusive_atomic",
    "validate_v2_public_inputs",
    "candidate_record_bytes",
    "validate_secret_separation",
    "selection_rank",
    "derive_evaluation_seed",
    "validate_commitment_registry",
    "validate_identity_disjointness_report",
    "validate_construct_equivalence_report",
    "validate_selector_summary",
    "validate_invalid_outcome",
    "validate_code_registry",
    "validate_no_v2_imports",
    "verify_file_record",
]
