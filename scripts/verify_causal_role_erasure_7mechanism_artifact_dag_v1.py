#!/usr/bin/env python3
"""Verify the declared artifact DAG and pre-metric code-freeze manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DAG_PROTOCOL = "causal_role_erasure_7m_artifact_dag_v1"
PRE_METRIC_PROTOCOL = "causal_role_erasure_7m_pre_metric_code_freeze_v2"
PRE_METRIC_STATUS = "clean_tree_components_frozen_before_human_canonical_metrics"
PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
ENVIRONMENT_PROTOCOL = "causal_role_erasure_7m_release_evaluation_environment_v1"
ENVIRONMENT_STATUS = "release_time_environment_validated_not_historical_execution_receipt"
INVENTORY_PROTOCOL = "causal_role_erasure_7m_wan_training_receipt_inventory_v1"
INVENTORY_STATUS = "18_receipts_indexed_from_individually_validated_eval_run_manifest"
LEDGER_PROTOCOL = "causal_role_erasure_7m_executable_command_ledger_v1"
LEDGER_STATUS = "full_reproduction_commands_frozen_before_canonical_metrics"
FINAL_DAG_STATUS = "complete_human_canonical_release_bound_to_pre_metric_prefix"
RESERVED_EXTRA_ROOT_IDS = {"repo_root", "pre_metric_root"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
EVAL_PYTHON_PLACEHOLDER = {
    "kind": "python_interpreter",
    "must_be_supplied": True,
    "executable_sha256_binding": {
        "artifact_binding": "evaluation_environment_receipt",
        "json_pointer": "/python/executable_sha256",
        "comparison": "sha256_of_resolved_executable",
    },
}
LEDGER_SCOPE = {
    "commands_are_sanitized": True,
    "commands_are_argument_vectors_not_shell_strings": True,
    "post_hoc_reproduction_commands_are_not_claimed_as_historical_invocations": True,
    "realized_expanded_generation_commands_are_bound_by_evidence": True,
    "manual_stage_inventory_includes_non_executable_human_work": True,
    "manual_stage_released_input_refs_are_partial_not_complete_delivery_receipts": True,
    "target_selection_private_outputs_and_exact_delivery_receipts_are_not_released_here": True,
}
MANUAL_STAGE_FIELDS = {
    "stage_id",
    "phase",
    "completion_state_at_ledger_freeze",
    "parent_stage_ids",
    "instruction",
    "released_input_refs",
    "unreleased_or_pending_input_roles",
}
MANUAL_STAGE_IDS = (
    "target_selection_reviewer_a",
    "target_selection_blind_audit_1",
    "target_selection_blind_audit_2",
    "target_selection_blind_audit_3",
    "target_selection_blind_audit_4",
    "target_selection_blind_adjudication",
    "initial_reviewer_1",
    "initial_reviewer_2",
    "initial_adjudication",
    "final_reviewer_1",
    "final_reviewer_2",
    "final_adjudication",
)
TARGET_SELECTION_MANUAL_STAGE_IDS = set(MANUAL_STAGE_IDS[:6])
TARGET_SELECTION_RELEASED_INPUT_PATHS = {
    "static_build_registry": "data/causal_role_erasure_7mechanism_main_v2/build_registry.json",
    "target_candidate_registry": "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv",
}
HUMAN_EVALUATION_RELEASED_INPUT_PATHS = {
    "reviewer_instructions": "docs/causal_role_erasure_7mechanism_human_review_v2.md",
}
MANUAL_STAGE_PARENTS = {
    "target_selection_reviewer_a": (),
    "target_selection_blind_audit_1": (),
    "target_selection_blind_audit_2": (),
    "target_selection_blind_audit_3": (),
    "target_selection_blind_audit_4": (),
    "target_selection_blind_adjudication": (
        "target_selection_reviewer_a",
        "target_selection_blind_audit_1",
        "target_selection_blind_audit_2",
        "target_selection_blind_audit_3",
        "target_selection_blind_audit_4",
    ),
    "initial_reviewer_1": (),
    "initial_reviewer_2": (),
    "initial_adjudication": ("initial_reviewer_1", "initial_reviewer_2"),
    "final_reviewer_1": ("initial_reviewer_1", "initial_adjudication"),
    "final_reviewer_2": ("initial_reviewer_2", "initial_adjudication"),
    "final_adjudication": ("final_reviewer_1", "final_reviewer_2"),
}
REQUIRED_COMPONENT_PATHS = {
    "evaluation_provenance_amendment": "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json",
    "canonicalization_wrapper": "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py",
    "formal_metrics_wrapper": "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py",
    "final_table_exporter": "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
    "reviewer_instructions": "docs/causal_role_erasure_7mechanism_human_review_v2.md",
    "reproducibility_builder": "scripts/build_causal_role_erasure_7mechanism_reproducibility_v1.py",
    "review_process_receipt_builder": "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py",
    "pre_metric_freeze_builder": "scripts/build_causal_role_erasure_7mechanism_pre_metric_freeze_v2.py",
    "artifact_dag_verifier": "scripts/verify_causal_role_erasure_7mechanism_artifact_dag_v1.py",
    "evaluation_environment_lock": "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock",
    "reproducibility_guide": "reproducibility/causal_role_erasure_7m/README.md",
}
RELEASE_TEST_PATHS = (
    "tests/test_build_causal_role_erasure_7mechanism_review_package_v1.py",
    "tests/test_review_causal_role_erasure_7mechanism_formal_v2.py",
    "tests/test_run_causal_role_erasure_7mechanism_formal_review_v1.py",
    "tests/test_causal_role_erasure_7mechanism_evaluation_code_registry_v1.py",
    "tests/test_causal_role_erasure_7mechanism_formal_metrics_v1.py",
    "tests/test_causal_role_erasure_7mechanism_evaluation_v2.py",
    "tests/test_export_causal_role_erasure_7mechanism_paper_tables_v1.py",
    "tests/test_causal_role_erasure_7mechanism_reproducibility_v1.py",
    "tests/test_causal_role_erasure_7mechanism_review_process_receipt_v2.py",
)
FORBIDDEN_RELEASE_TOKENS = (
    "post_unblinding_preliminary",
    "not_canonical",
    "writing_preview",
    "preview_label",
    "tables/preliminary",
    "/private/",
)
FINAL_PENDING_NODE_IDS = {
    "reviewer_deliveries",
    "human_labels",
    "adjudication",
    "audit_expansion",
    "final_review_process_receipt",
    "canonical_scores",
    "original_eligibility",
    "formal_metrics",
    "paper_tables",
}
FINAL_UNAVAILABLE_NODE_IDS = {
    "target_generation_aggregate",
    "target_selection_registry",
    "selected_targets",
    "training_input_registry",
    "wan_run_spec_registry",
    "t2v_training_registry",
}
FINAL_ARTIFACT_LOCATIONS = {
    "reviewer_deliveries": ("final_review_process_root", "review_process_receipt_v2.json"),
    "human_labels": ("final_review_process_root", "review_process_receipt_v2.json"),
    "adjudication": ("final_review_process_root", "review_process_receipt_v2.json"),
    "audit_expansion": ("final_audit_root", "audit_manifest.json"),
    "final_review_process_receipt": ("final_review_process_root", "review_process_receipt_v2.json"),
    "canonical_scores": ("canonical_root", "canonical_manifest.json"),
    "original_eligibility": ("eligibility_root", "eligibility_manifest.json"),
    "formal_metrics": ("metrics_root", "score_manifest.json"),
    "paper_tables": ("paper_table_root", "paper_tables_manifest.json"),
}
FINAL_SEMANTIC_CHECKS = {
    "reviewer_deliveries": [{"pointer": "/status", "equals": "independent_public_only_reviews_and_complete_adjudication_frozen"}],
    "human_labels": [{"pointer": "/assertions/two_independent_reviews", "equals": True}],
    "adjudication": [{"pointer": "/counts/unresolved_disagreements", "equals": 0}],
    "audit_expansion": [],
    "final_review_process_receipt": [{"pointer": "/status", "equals": "independent_public_only_reviews_and_complete_adjudication_frozen"}],
    "canonical_scores": [{"pointer": "/status", "equals": "canonical_anonymous_scores_frozen_after_disclosed_global_full_key_access"}],
    "original_eligibility": [{"pointer": "/status", "equals": "original_eligibility_and_shared_subsets_frozen_after_disclosed_global_full_key_access"}],
    "formal_metrics": [{"pointer": "/status", "equals": "formal_metrics_frozen_after_disclosed_global_full_key_access"}],
    "paper_tables": [{"pointer": "/status", "equals": "human_canonical_paper_tables_complete"}],
}


class ArtifactDagError(ValueError):
    """The artifact graph is malformed, stale, cyclic, or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactDagError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular non-symlink file")


def safe_relative(value: str, label: str) -> Path:
    path = Path(value)
    require(value != "" and not path.is_absolute(), f"{label} path must be relative")
    require(".." not in path.parts and "." not in path.parts, f"{label} path is unsafe")
    lowered = path.as_posix().lower()
    require(not any(token in lowered for token in FORBIDDEN_RELEASE_TOKENS), f"{label} reaches forbidden preliminary/private material")
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = value.get(field)
    require(isinstance(observed, str) and HEX64.fullmatch(observed) is not None, f"{label} self digest missing")
    body = dict(value)
    del body[field]
    require(hashlib.sha256(canonical_json_bytes(body)).hexdigest() == observed, f"{label} self digest mismatch")


def git_bytes(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    require(completed.returncode == 0, f"git {' '.join(arguments)} failed")
    return completed.stdout


def git_text(repo_root: Path, *arguments: str) -> str:
    return git_bytes(repo_root, *arguments).decode("utf-8").strip()


def committed_blob(repo_root: Path, commit: str, relative: str, label: str) -> bytes:
    safe_relative(relative, label)
    line = git_text(repo_root, "ls-tree", commit, "--", relative)
    fields = line.split(None, 3)
    require(
        len(fields) == 4
        and fields[0] in ("100644", "100755")
        and fields[1] == "blob"
        and HEX40.fullmatch(fields[2]) is not None
        and fields[3] == relative,
        f"{label} is absent or not a regular blob in recorded implementation commit",
    )
    require(git_text(repo_root, "cat-file", "-t", fields[2]) == "blob", f"{label} Git object is not a blob")
    return git_bytes(repo_root, "cat-file", "blob", fields[2])


def verify_ref_matches_commit(
    repo_root: Path,
    commit: str,
    ref: Mapping[str, Any],
    label: str,
) -> None:
    blob = committed_blob(repo_root, commit, str(ref["path"]), label)
    require(
        len(blob) == ref["size_bytes"]
        and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        f"{label} bytes differ from recorded implementation commit",
    )


def validate_ref(ref: Mapping[str, Any], roots: Mapping[str, Path], label: str) -> Path:
    require(
        set(ref) == {"root_id", "path", "sha256", "size_bytes"},
        f"{label} reference schema changed",
    )
    root_id = ref.get("root_id")
    require(isinstance(root_id, str) and root_id in roots, f"{label} root is not supplied")
    relative = safe_relative(str(ref.get("path", "")), label)
    resolved_root = roots[root_id].resolve(strict=True)
    path = (resolved_root / relative).resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactDagError(f"{label} escapes root {root_id}") from exc
    regular_file(path, label)
    require(isinstance(ref.get("sha256"), str) and HEX64.fullmatch(ref["sha256"]) is not None, f"{label} SHA malformed")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA mismatch")
    require(path.stat().st_size == ref.get("size_bytes"), f"{label} size mismatch")
    return path


def file_ref(root_id: str, relative: str, roots: Mapping[str, Path]) -> dict[str, Any]:
    require(root_id in roots, f"final artifact root is not supplied: {root_id}")
    safe = safe_relative(relative, f"{root_id} artifact")
    root = roots[root_id].resolve(strict=True)
    path = (root / safe).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArtifactDagError(f"final artifact escapes root {root_id}") from exc
    regular_file(path, f"{root_id}/{relative}")
    return {
        "root_id": root_id,
        "path": safe.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "output parent is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def sealed(value: Mapping[str, Any], field: str = "dag_sha256") -> dict[str, Any]:
    body = dict(value)
    require(field not in body, f"{field} already exists")
    return {**body, field: hashlib.sha256(canonical_json_bytes(body)).hexdigest()}


def json_pointer(value: Any, pointer: str) -> Any:
    require(pointer.startswith("/"), "JSON pointer must begin with /")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            require(token in current, f"JSON pointer missing key {token}")
            current = current[token]
        elif isinstance(current, list):
            require(token.isdigit() and int(token) < len(current), "JSON pointer index invalid")
            current = current[int(token)]
        else:
            raise ArtifactDagError("JSON pointer descends through a scalar")
    return current


def validate_semantics(path: Path, checks: Sequence[Mapping[str, Any]], label: str) -> None:
    if not checks:
        return
    payload = load_json(path, label)
    for check in checks:
        require(set(check) == {"pointer", "equals"}, f"{label} semantic check malformed")
        require(json_pointer(payload, str(check["pointer"])) == check["equals"], f"{label} semantic check failed at {check['pointer']}")


def validate_acyclic(nodes: Mapping[str, Mapping[str, Any]]) -> None:
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        marker = state.get(node_id, 0)
        require(marker != 1, "artifact DAG contains a cycle")
        if marker == 2:
            return
        state[node_id] = 1
        parents = nodes[node_id].get("parents")
        require(isinstance(parents, list) and all(isinstance(parent, str) for parent in parents), f"{node_id} parents malformed")
        for parent in parents:
            require(parent in nodes, f"{node_id} has unknown parent {parent}")
            visit(parent)
        state[node_id] = 2

    for node_id in nodes:
        visit(node_id)


def verify_dag(value: Mapping[str, Any], roots: Mapping[str, Path], mode: str) -> dict[str, int]:
    require(mode in ("pre-metric", "final"), "artifact DAG verification mode invalid")
    require(
        set(value)
        == {"schema_version", "protocol", "protocol_version", "status", "nodes", "counts", "dag_sha256"},
        "artifact DAG top-level schema mismatch",
    )
    require(
        value.get("schema_version") == 1
        and value.get("protocol") == DAG_PROTOCOL
        and value.get("protocol_version") == PROTOCOL_VERSION,
        "artifact DAG protocol mismatch",
    )
    require(isinstance(value.get("status"), str) and value["status"].strip(), "artifact DAG status missing")
    serialized = json.dumps(value, sort_keys=True).lower()
    require(
        not any(token in serialized for token in FORBIDDEN_RELEASE_TOKENS),
        "artifact DAG contains forbidden preliminary/private material",
    )
    validate_self_digest(value, "dag_sha256", "artifact DAG")
    raw_nodes = value.get("nodes")
    require(isinstance(raw_nodes, list) and raw_nodes, "artifact DAG has no nodes")
    nodes = {str(node.get("id")): node for node in raw_nodes if isinstance(node, dict)}
    require(
        len(nodes) == len(raw_nodes)
        and all(ROLE.fullmatch(node_id) is not None for node_id in nodes),
        "artifact DAG node IDs repeat, are blank, or are unsafe",
    )
    validate_acyclic(nodes)
    counts = Counter(str(node.get("status")) for node in raw_nodes)
    require(set(counts) <= {"materialized", "pending", "private_commitment", "declared_unavailable"}, "artifact DAG node status invalid")
    if mode == "final":
        require(value["status"] == FINAL_DAG_STATUS, "final artifact DAG status changed")
        require(
            FINAL_PENDING_NODE_IDS
            | {"pre_metric_artifact_dag", "pre_metric_code_freeze"}
            <= set(nodes),
            "final artifact DAG omits mandatory completion/ancestry nodes",
        )
        require(counts.get("pending", 0) == 0, "final artifact DAG still has pending nodes")
    for node_id, node in nodes.items():
        status = node.get("status")
        parents = node.get("parents")
        require(
            isinstance(parents, list)
            and len(parents) == len(set(parents))
            and node_id not in parents,
            f"{node_id} parents repeat or contain itself",
        )
        if status == "materialized":
            require(
                set(node) == {"id", "status", "parents", "artifact", "semantic_checks"},
                f"{node_id} materialized-node schema changed",
            )
            ref = node.get("artifact")
            require(isinstance(ref, dict), f"{node_id} materialized artifact missing")
            path = validate_ref(ref, roots, node_id)
            checks = node.get("semantic_checks", [])
            require(isinstance(checks, list), f"{node_id} semantic checks malformed")
            validate_semantics(path, checks, node_id)
        elif status == "private_commitment":
            require(
                set(node) == {"id", "status", "parents", "commitment_sha256", "release_statement"},
                f"{node_id} private-commitment schema changed",
            )
            require(
                isinstance(node.get("commitment_sha256"), str)
                and HEX64.fullmatch(node["commitment_sha256"]) is not None
                and isinstance(node.get("release_statement"), str)
                and node["release_statement"].strip(),
                f"{node_id} private commitment incomplete",
            )
            require("artifact" not in node, f"{node_id} private node exposes a path")
        elif status == "declared_unavailable":
            expected_fields = {
                "id",
                "status",
                "parents",
                "access_conditions",
                "reproduction_limit",
                "expected_sha256" if mode == "final" else "expected_sha256_prefix",
            }
            require(set(node) == expected_fields, f"{node_id} unavailable-node schema changed")
            digest = node.get("expected_sha256")
            prefix = node.get("expected_sha256_prefix")
            require(
                (
                    isinstance(digest, str)
                    and HEX64.fullmatch(digest) is not None
                    and prefix is None
                )
                or (
                    mode == "pre-metric"
                    and digest is None
                    and isinstance(prefix, str)
                    and re.fullmatch(r"[0-9a-f]{12}", prefix) is not None
                ),
                f"{node_id} unavailable artifact digest is incomplete",
            )
            require(
                isinstance(node.get("access_conditions"), str)
                and node["access_conditions"].strip()
                and isinstance(node.get("reproduction_limit"), str)
                and node["reproduction_limit"].strip(),
                f"{node_id} unavailable artifact lacks access/limit",
            )
        else:
            require(
                set(node) == {"id", "status", "parents", "pending_reason"},
                f"{node_id} pending-node schema changed",
            )
            require(isinstance(node.get("pending_reason"), str) and node["pending_reason"].strip(), f"{node_id} pending reason missing")
            require(
                "artifact" not in node
                and "expected_sha256" not in node
                and "expected_sha256_prefix" not in node,
                f"{node_id} has a prospective hash",
            )
    expected = value.get("counts")
    require(isinstance(expected, dict) and expected == dict(sorted(counts.items())), "artifact DAG status counts mismatch")
    return dict(counts)


def parse_unavailable_digests(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        require("=" in value, "--unavailable-sha256 must be NODE_ID=SHA256")
        node_id, digest = value.split("=", 1)
        require(node_id not in result, "unavailable SHA node repeats")
        require(
            node_id in FINAL_UNAVAILABLE_NODE_IDS and HEX64.fullmatch(digest) is not None,
            "unavailable SHA declaration is invalid",
        )
        result[node_id] = digest
    require(
        set(result) == FINAL_UNAVAILABLE_NODE_IDS,
        "final DAG requires all six full unavailable-artifact SHA-256 values",
    )
    return result


def _self_hash_matches(value: Mapping[str, Any], field: str, label: str) -> None:
    require(isinstance(value.get(field), str), f"{label} self hash missing")
    body = {key: item for key, item in value.items() if key != field}
    require(
        value[field] == hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        f"{label} self hash mismatch",
    )


def validate_final_authorities(
    *, roots: Mapping[str, Path], pre_metric: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    refs = {
        node_id: file_ref(root_id, relative, roots)
        for node_id, (root_id, relative) in FINAL_ARTIFACT_LOCATIONS.items()
    }
    pre_metric_ref = file_ref(
        "pre_metric_root", "pre_metric_code_freeze_v2.json", roots
    )
    process_path = validate_ref(
        refs["final_review_process_receipt"], roots, "final review process receipt"
    )
    process = load_json(process_path, "final review process receipt")
    require(
        process.get("protocol") == "causal_role_erasure_7m_human_review_process_receipt_v2"
        and process.get("protocol_version") == PROTOCOL_VERSION
        and process.get("status")
        == "independent_public_only_reviews_and_complete_adjudication_frozen",
        "final review process receipt identity/status changed",
    )
    _self_hash_matches(process, "receipt_sha256", "final review process receipt")
    require(
        process.get("pre_metric_authority")
        == {
            "pre_metric_code_freeze": pre_metric_ref,
            "implementation_commit": pre_metric["implementation_commit"],
            "evaluation_environment_receipt": pre_metric["reproducibility_bindings"][
                "evaluation_environment_receipt"
            ],
        }
        and process.get("review_round_ancestry", {}).get("round")
        == "expanded_or_no_expansion",
        "final review process is not bound to the pre-metric/initial-review ancestry",
    )
    audit = load_json(
        validate_ref(refs["audit_expansion"], roots, "final audit manifest"),
        "final audit manifest",
    )
    require(
        audit.get("protocol") == "causal_role_erasure_7m_human_canonicalization_v1"
        and audit.get("status")
        in {
            "expanded_human_audit_frozen",
            "no_expansion_required_human_audit_frozen",
        },
        "final audit manifest identity/status changed",
    )
    canonical = load_json(
        validate_ref(refs["canonical_scores"], roots, "canonical manifest"),
        "canonical manifest",
    )
    eligibility = load_json(
        validate_ref(refs["original_eligibility"], roots, "eligibility manifest"),
        "eligibility manifest",
    )
    metrics = load_json(
        validate_ref(refs["formal_metrics"], roots, "score manifest"),
        "score manifest",
    )
    paper = load_json(
        validate_ref(refs["paper_tables"], roots, "paper-table manifest"),
        "paper-table manifest",
    )
    for value, field, status, label in (
        (
            canonical,
            "manifest_sha256",
            "canonical_anonymous_scores_frozen_after_disclosed_global_full_key_access",
            "canonical manifest",
        ),
        (
            eligibility,
            "manifest_sha256",
            "original_eligibility_and_shared_subsets_frozen_after_disclosed_global_full_key_access",
            "eligibility manifest",
        ),
        (
            metrics,
            "manifest_sha256",
            "formal_metrics_frozen_after_disclosed_global_full_key_access",
            "score manifest",
        ),
        (
            paper,
            "manifest_sha256",
            "human_canonical_paper_tables_complete",
            "paper-table manifest",
        ),
    ):
        require(value.get("status") == status, f"{label} status changed")
        _self_hash_matches(value, field, label)
    expected_process_ref = refs["final_review_process_receipt"]
    expected_canonical_ref = refs["canonical_scores"]
    expected_eligibility_ref = refs["original_eligibility"]
    expected_metrics_ref = refs["formal_metrics"]
    require(
        canonical.get("pre_metric_code_freeze") == pre_metric_ref
        and canonical.get("inputs", {}).get("review_process_receipt")
        == expected_process_ref
        and canonical.get("inputs", {}).get("completed_human_audit")
        == process.get("inputs", {}).get("completed_human"),
        "canonical manifest differs from final review/pre-metric authority",
    )
    require(
        eligibility.get("pre_metric_code_freeze") == pre_metric_ref
        and eligibility.get("inputs", {}).get("canonical_manifest")
        == expected_canonical_ref,
        "eligibility manifest differs from canonical/pre-metric authority",
    )
    require(
        metrics.get("pre_metric_code_freeze") == pre_metric_ref
        and metrics.get("inputs", {}).get("canonical_manifest")
        == expected_canonical_ref
        and metrics.get("inputs", {}).get("eligibility_manifest")
        == expected_eligibility_ref,
        "score manifest differs from canonical/eligibility/pre-metric authority",
    )
    require(
        paper.get("pre_metric_code_freeze") == pre_metric_ref
        and {
            key: paper.get("score_manifest", {}).get(key)
            for key in ("root_id", "path", "sha256", "size_bytes")
        }
        == expected_metrics_ref,
        "paper-table manifest differs from score/pre-metric authority",
    )
    return refs


def build_final_dag(
    *,
    prefix: Mapping[str, Any],
    roots: Mapping[str, Path],
    pre_metric: Mapping[str, Any],
    unavailable_sha256: Mapping[str, str],
) -> dict[str, Any]:
    verify_dag(prefix, roots, "pre-metric")
    prefix_nodes = prefix["nodes"]
    pending_ids = {
        str(node.get("id"))
        for node in prefix_nodes
        if isinstance(node, dict) and node.get("status") == "pending"
    }
    unavailable_ids = {
        str(node.get("id"))
        for node in prefix_nodes
        if isinstance(node, dict) and node.get("status") == "declared_unavailable"
    }
    require(pending_ids == FINAL_PENDING_NODE_IDS, "pre-metric pending branch changed")
    require(
        unavailable_ids == FINAL_UNAVAILABLE_NODE_IDS,
        "pre-metric unavailable upstream inventory changed",
    )
    refs = validate_final_authorities(roots=roots, pre_metric=pre_metric)
    final_nodes: list[dict[str, Any]] = []
    for source in prefix_nodes:
        node = dict(source)
        node_id = str(node["id"])
        if node["status"] == "pending":
            parents = list(node["parents"])
            if node_id == "reviewer_deliveries":
                parents.append("pre_metric_code_freeze")
            node = {
                "id": node_id,
                "status": "materialized",
                "parents": parents,
                "artifact": refs[node_id],
                "semantic_checks": FINAL_SEMANTIC_CHECKS[node_id],
            }
        elif node["status"] == "declared_unavailable":
            digest = unavailable_sha256[node_id]
            require(
                digest.startswith(str(node["expected_sha256_prefix"])),
                f"{node_id} full SHA does not extend the frozen prefix",
            )
            node = {
                "id": node_id,
                "status": "declared_unavailable",
                "parents": list(node["parents"]),
                "expected_sha256": digest,
                "access_conditions": node["access_conditions"],
                "reproduction_limit": node["reproduction_limit"],
            }
        final_nodes.append(node)
    prefix_ref = dict(pre_metric["reproducibility_bindings"]["pre_metric_artifact_dag"])
    final_nodes.extend(
        [
            {
                "id": "pre_metric_artifact_dag",
                "status": "materialized",
                "parents": [],
                "artifact": prefix_ref,
                "semantic_checks": [
                    {
                        "pointer": "/status",
                        "equals": "completed_prefix_bound_with_explicit_pending_human_branch",
                    }
                ],
            },
            {
                "id": "pre_metric_code_freeze",
                "status": "materialized",
                "parents": ["pre_metric_artifact_dag"],
                "artifact": file_ref(
                    "pre_metric_root", "pre_metric_code_freeze_v2.json", roots
                ),
                "semantic_checks": [{"pointer": "/status", "equals": PRE_METRIC_STATUS}],
            },
        ]
    )
    counts = dict(sorted(Counter(str(node["status"]) for node in final_nodes).items()))
    return sealed(
        {
            "schema_version": 1,
            "protocol": DAG_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": FINAL_DAG_STATUS,
            "nodes": final_nodes,
            "counts": counts,
        }
    )


def verify_final_prefix_binding(
    *, final: Mapping[str, Any], prefix: Mapping[str, Any], pre_metric: Mapping[str, Any], roots: Mapping[str, Path]
) -> None:
    require(final.get("status") == FINAL_DAG_STATUS, "final artifact DAG status changed")
    final_nodes = {str(node["id"]): node for node in final["nodes"]}
    prefix_nodes = {str(node["id"]): node for node in prefix["nodes"]}
    require(
        set(final_nodes) == set(prefix_nodes) | {"pre_metric_artifact_dag", "pre_metric_code_freeze"},
        "final artifact DAG mandatory node set changed",
    )
    refs = validate_final_authorities(roots=roots, pre_metric=pre_metric)
    for node_id, before in prefix_nodes.items():
        after = final_nodes[node_id]
        if before["status"] in {"materialized", "private_commitment"}:
            require(after == before, f"final DAG changed frozen prefix node {node_id}")
        elif before["status"] == "declared_unavailable":
            require(
                set(after)
                == {"id", "status", "parents", "expected_sha256", "access_conditions", "reproduction_limit"}
                and after["status"] == "declared_unavailable"
                and after["parents"] == before["parents"]
                and after["expected_sha256"].startswith(before["expected_sha256_prefix"])
                and after["access_conditions"] == before["access_conditions"]
                and after["reproduction_limit"] == before["reproduction_limit"],
                f"final DAG changed unavailable prefix node {node_id}",
            )
        else:
            expected_parents = list(before["parents"])
            if node_id == "reviewer_deliveries":
                expected_parents.append("pre_metric_code_freeze")
            require(
                after
                == {
                    "id": node_id,
                    "status": "materialized",
                    "parents": expected_parents,
                    "artifact": refs[node_id],
                    "semantic_checks": FINAL_SEMANTIC_CHECKS[node_id],
                },
                f"final DAG completion node changed: {node_id}",
            )
    expected_prefix_ref = pre_metric["reproducibility_bindings"][
        "pre_metric_artifact_dag"
    ]
    require(
        final_nodes["pre_metric_artifact_dag"]
        == {
            "id": "pre_metric_artifact_dag",
            "status": "materialized",
            "parents": [],
            "artifact": expected_prefix_ref,
            "semantic_checks": [
                {
                    "pointer": "/status",
                    "equals": "completed_prefix_bound_with_explicit_pending_human_branch",
                }
            ],
        }
        and final_nodes["pre_metric_code_freeze"]
        == {
            "id": "pre_metric_code_freeze",
            "status": "materialized",
            "parents": ["pre_metric_artifact_dag"],
            "artifact": file_ref(
                "pre_metric_root", "pre_metric_code_freeze_v2.json", roots
            ),
            "semantic_checks": [{"pointer": "/status", "equals": PRE_METRIC_STATUS}],
        },
        "final DAG pre-metric ancestry changed",
    )


def verify_pre_metric_freeze(value: Mapping[str, Any], roots: Mapping[str, Path]) -> None:
    required_top = {
        "schema_version", "protocol", "protocol_version", "status", "implementation_commit",
        "deviation", "v1_authority", "source_bindings", "components", "reproducibility_bindings",
        "pending_outputs", "manifest_sha256",
    }
    require(set(value) == required_top, "pre-metric freeze top-level schema mismatch")
    require(
        value.get("schema_version") == 1
        and value.get("protocol") == PRE_METRIC_PROTOCOL
        and value.get("protocol_version") == PROTOCOL_VERSION
        and value.get("status") == PRE_METRIC_STATUS,
        "pre-metric freeze protocol/status mismatch",
    )
    validate_self_digest(value, "manifest_sha256", "pre-metric freeze")
    require(
        value.get("deviation")
        == {
            "global_full_key_opened_before_canonical_freeze": True,
            "human_review_process_local_answer_key_blindness_preserved": True,
            "v1_files_byte_identical": True,
            "scientific_rules_changed": False,
            "unchanged_v1_final_stages_forbidden": True,
        },
        "pre-metric deviation declaration changed",
    )
    commit = value.get("implementation_commit")
    require(
        isinstance(commit, dict)
        and set(commit) == {"commit", "tree", "worktree_clean"}
        and isinstance(commit.get("commit"), str)
        and HEX40.fullmatch(commit["commit"]) is not None
        and isinstance(commit.get("tree"), str)
        and HEX40.fullmatch(commit["tree"]) is not None
        and commit.get("worktree_clean") is True,
        "pre-metric implementation commit binding invalid",
    )
    components = value.get("components")
    require(isinstance(components, list) and components, "pre-metric components missing")
    roles: set[str] = set()
    component_paths: set[str] = set()
    repo_root = roots.get("repo_root")
    require(repo_root is not None, "repo_root is required for component verification")
    recorded_commit = str(commit["commit"])
    require(
        git_text(repo_root, "rev-parse", f"{recorded_commit}^{{commit}}") == recorded_commit
        and git_text(repo_root, "cat-file", "-t", recorded_commit) == "commit",
        "recorded implementation commit is unavailable or is not a commit object",
    )
    require(
        git_text(repo_root, "rev-parse", f"{recorded_commit}^{{tree}}") == commit["tree"]
        and git_text(repo_root, "cat-file", "-t", str(commit["tree"])) == "tree",
        "recorded implementation tree differs from commit or is not a tree object",
    )

    authority = value.get("v1_authority")
    require(isinstance(authority, dict) and set(authority) == {"evaluation_code_registry", "files"}, "v1 authority schema changed")
    registry_ref = authority["evaluation_code_registry"]
    require(
        isinstance(registry_ref, dict)
        and set(registry_ref) == {"path", "sha256", "size_bytes", "registry_sha256"}
        and registry_ref["path"] == "review_package/public/evaluation_code_registry.json"
        and isinstance(registry_ref["registry_sha256"], str)
        and HEX64.fullmatch(registry_ref["registry_sha256"]) is not None,
        "v1 evaluation registry authority changed",
    )
    registry_path = validate_ref(
        {
            "root_id": "review_root",
            "path": registry_ref["path"],
            "sha256": registry_ref["sha256"],
            "size_bytes": registry_ref["size_bytes"],
        },
        roots,
        "v1 evaluation registry",
    )
    registry_value = load_json(registry_path, "v1 evaluation registry")
    require(
        registry_value.get("registry_sha256") == registry_ref["registry_sha256"]
        and registry_value.get("files") == authority["files"],
        "v1 evaluation registry body differs from authority",
    )
    v1_files = authority["files"]
    require(isinstance(v1_files, list) and len(v1_files) == 5, "v1 authority must bind five files")
    for index, ref in enumerate(v1_files):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}, f"v1 file {index} ref malformed")
        validate_ref({"root_id": "repo_root", **ref}, roots, f"v1 file {index}")
        verify_ref_matches_commit(repo_root, recorded_commit, ref, f"v1 file {index}")

    source_bindings = value.get("source_bindings")
    expected_source_names = {
        "evaluation_code_registry", "key_commitments", "formal_launch_receipt",
        "pass_a_merge_manifest", "pass_b_merge_manifest", "human_audit_manifest",
        "human_audit_queue", "formal_cases", "identification_subset", "private_key_commitments",
    }
    require(isinstance(source_bindings, dict) and set(source_bindings) == expected_source_names, "pre-metric source-binding inventory changed")
    for name in expected_source_names - {"private_key_commitments"}:
        ref = source_bindings[name]
        require(isinstance(ref, dict), f"source binding {name} malformed")
        validate_ref(ref, roots, f"source binding {name}")
    require(
        source_bindings["evaluation_code_registry"]["sha256"] == registry_ref["sha256"],
        "source binding and v1 registry authority disagree",
    )
    private = source_bindings["private_key_commitments"]
    require(
        isinstance(private, dict)
        and set(private) == {"tier_0_audit_strata", "tier_1_original_only", "tier_2_full"},
        "private commitment inventory changed",
    )
    for name, ref in private.items():
        require(
            isinstance(ref, dict)
            and set(ref) == {"sha256", "row_count"}
            and isinstance(ref["sha256"], str)
            and HEX64.fullmatch(ref["sha256"]) is not None
            and type(ref["row_count"]) is int
            and ref["row_count"] > 0,
            f"private commitment {name} malformed",
        )
    for component in components:
        require(isinstance(component, dict) and set(component) == {"path", "sha256", "size_bytes", "role"}, "component record malformed")
        role = component.get("role")
        require(
            isinstance(role, str)
            and ROLE.fullmatch(role) is not None
            and role not in roles,
            "component roles repeat or are unsafe",
        )
        roles.add(role)
        require(component["path"] not in component_paths, "component paths repeat")
        component_paths.add(str(component["path"]))
        ref = {"root_id": "repo_root", "path": component["path"], "sha256": component["sha256"], "size_bytes": component["size_bytes"]}
        validate_ref(ref, roots, f"component {role}")
        verify_ref_matches_commit(repo_root, recorded_commit, ref, f"component {role}")
    require(
        set(REQUIRED_COMPONENT_PATHS) <= roles,
        "pre-metric component roles incomplete",
    )
    by_role = {str(component["role"]): component for component in components}
    for role, expected_path in REQUIRED_COMPONENT_PATHS.items():
        require(by_role[role]["path"] == expected_path, f"pre-metric component path changed: {role}")
    require(
        [str(component["role"]) for component in components]
        == sorted(str(component["role"]) for component in components),
        "pre-metric components are not in deterministic role order",
    )
    bindings = value.get("reproducibility_bindings")
    require(
        isinstance(bindings, dict)
        and set(bindings)
        == {"evaluation_environment_lock", "evaluation_environment_receipt", "wan_training_receipt_inventory", "reproduction_command_ledger", "pre_metric_artifact_dag"},
        "reproducibility bindings missing or changed",
    )
    for name, ref in bindings.items():
        require(isinstance(ref, dict), f"reproducibility binding {name} malformed")
        validate_ref(ref, roots, f"reproducibility binding {name}")
    expected_repro_locations = {
        "evaluation_environment_lock": (
            "repo_root",
            "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock",
        ),
        "evaluation_environment_receipt": (
            "pre_metric_root",
            "evaluation_environment_receipt_v1.json",
        ),
        "wan_training_receipt_inventory": (
            "pre_metric_root",
            "wan_training_receipt_inventory_v1.json",
        ),
        "reproduction_command_ledger": (
            "pre_metric_root",
            "reproduction_command_ledger_v1.json",
        ),
        "pre_metric_artifact_dag": (
            "pre_metric_root",
            "pre_metric_artifact_dag_v1.json",
        ),
    }
    for name, (root_id, path) in expected_repro_locations.items():
        require(
            bindings[name]["root_id"] == root_id and bindings[name]["path"] == path,
            f"reproducibility binding location changed: {name}",
        )

    environment_receipt_path = validate_ref(
        bindings["evaluation_environment_receipt"], roots, "release evaluation environment receipt"
    )
    environment_receipt = load_json(
        environment_receipt_path, "release evaluation environment receipt"
    )
    require(
        environment_receipt.get("schema_version") == 1
        and environment_receipt.get("protocol") == ENVIRONMENT_PROTOCOL
        and environment_receipt.get("protocol_version") == PROTOCOL_VERSION
        and environment_receipt.get("status") == ENVIRONMENT_STATUS
        and environment_receipt.get("tests", {}).get("status") == "passed"
        and environment_receipt.get("tests", {}).get("return_code") == 0,
        "release evaluation environment receipt is incomplete",
    )
    validate_self_digest(
        environment_receipt,
        "receipt_sha256",
        "release evaluation environment receipt",
    )
    require(
        environment_receipt.get("lock") == bindings["evaluation_environment_lock"],
        "environment receipt and pre-metric lock binding differ",
    )
    test_refs = environment_receipt.get("tests", {}).get("files")
    require(
        isinstance(test_refs, list)
        and [ref.get("path") for ref in test_refs if isinstance(ref, dict)]
        == list(RELEASE_TEST_PATHS),
        "environment receipt test-file inventory changed",
    )
    for index, ref in enumerate(test_refs):
        require(isinstance(ref, dict), f"environment receipt test {index} reference malformed")
        validate_ref(ref, roots, f"environment receipt test {index}")
        require(ref["root_id"] == "repo_root", f"environment receipt test {index} is outside the repository")
        verify_ref_matches_commit(
            repo_root, recorded_commit, ref, f"environment receipt test {index}"
        )
        require(ref["path"] in component_paths, f"environment receipt test {index} is not a frozen component")

    inventory_path = validate_ref(
        bindings["wan_training_receipt_inventory"], roots, "Wan training receipt inventory"
    )
    inventory = load_json(inventory_path, "Wan training receipt inventory")
    require(
        set(inventory)
        == {
            "schema_version",
            "protocol",
            "protocol_version",
            "status",
            "verification_basis",
            "sources",
            "counts",
            "items",
            "inventory_sha256",
        }
        and
        inventory.get("schema_version") == 1
        and inventory.get("protocol") == INVENTORY_PROTOCOL
        and inventory.get("protocol_version") == PROTOCOL_VERSION
        and inventory.get("status") == INVENTORY_STATUS
        and inventory.get("counts", {}).get("receipts") == 18
        and inventory.get("counts", {}).get("validated_generation_rows") == 684
        and isinstance(inventory.get("items"), list)
        and len(inventory["items"]) == 18,
        "Wan training receipt inventory is incomplete",
    )
    validate_self_digest(inventory, "inventory_sha256", "Wan training receipt inventory")
    inventory_items = inventory["items"]
    require(
        [item.get("run_index") for item in inventory_items if isinstance(item, dict)]
        == list(range(18)),
        "Wan training receipt order changed",
    )
    receipt_hashes: set[str] = set()
    run_spec_hashes: set[str] = set()
    for index, item in enumerate(inventory_items):
        require(
            isinstance(item, dict)
            and set(item)
            == {
                "run_index",
                "run_id",
                "mechanism",
                "arm",
                "run_group",
                "step",
                "erase_updates",
                "preserve_updates",
                "validated_generation_rows",
                "run_spec",
                "receipt",
                "checkpoint",
            },
            f"Wan training receipt item {index} schema changed",
        )
        require(
            item["step"] == 200
            and item["erase_updates"] == 100
            and item["preserve_updates"] == 100
            and item["validated_generation_rows"] in (24, 42),
            f"Wan training receipt item {index} training/count contract changed",
        )
        for name in ("run_spec", "receipt"):
            ref = item[name]
            require(
                isinstance(ref, dict)
                and set(ref) == {"root_id", "path", "sha256"}
                and ref["root_id"] == "a100_project_root"
                and isinstance(ref["sha256"], str)
                and HEX64.fullmatch(ref["sha256"]) is not None,
                f"Wan training receipt item {index} {name} ref invalid",
            )
            safe_relative(str(ref["path"]), f"Wan training receipt item {index} {name}")
        checkpoint = item["checkpoint"]
        require(
            isinstance(checkpoint, dict)
            and set(checkpoint)
            == {
                "root_id",
                "path",
                "artifact_sha256",
                "weights_sha256",
                "training_state_sha256",
            }
            and checkpoint["root_id"] == "a100_project_root"
            and all(
                isinstance(checkpoint[name], str)
                and HEX64.fullmatch(checkpoint[name]) is not None
                for name in (
                    "artifact_sha256",
                    "weights_sha256",
                    "training_state_sha256",
                )
            ),
            f"Wan training receipt item {index} checkpoint ref invalid",
        )
        safe_relative(str(checkpoint["path"]), f"Wan training receipt item {index} checkpoint")
        receipt_hashes.add(str(item["receipt"]["sha256"]))
        run_spec_hashes.add(str(item["run_spec"]["sha256"]))
    require(
        len(receipt_hashes) == len(run_spec_hashes) == 18,
        "Wan training receipt or run-spec hashes repeat",
    )

    ledger_path = validate_ref(
        bindings["reproduction_command_ledger"], roots, "reproduction command ledger"
    )
    ledger = load_json(ledger_path, "reproduction command ledger")
    commands = ledger.get("commands")
    stage_ids = {
        str(command.get("stage_id"))
        for command in commands
        if isinstance(command, dict)
    } if isinstance(commands, list) else set()
    required_review_stages = {
        "reviewer_1_initial_delivery",
        "reviewer_2_initial_delivery",
        "initial_human_review_process_receipt",
        "audit_expansion",
        "reviewer_1_final_delivery",
        "reviewer_2_final_delivery",
        "final_human_review_process_receipt",
        "paper_tables",
        "artifact_dag_final_materialize",
        "artifact_dag_final",
    }
    require(
        set(ledger)
        == {
            "schema_version",
            "protocol",
            "protocol_version",
            "status",
            "scope",
            "placeholders",
            "commands",
            "entry_points",
            "counts",
            "manual_stages",
            "realized_command_evidence",
            "ledger_sha256",
        }
        and
        ledger.get("schema_version") == 1
        and ledger.get("protocol") == LEDGER_PROTOCOL
        and ledger.get("protocol_version") == PROTOCOL_VERSION
        and ledger.get("status") == LEDGER_STATUS
        and isinstance(commands, list)
        and len(stage_ids) == len(commands)
        and required_review_stages <= stage_ids
        and ledger.get("counts", {}).get("command_records") == len(commands)
        and ledger.get("counts", {}).get("realized_expanded_generation_commands") == 60
        and isinstance(ledger.get("entry_points"), dict)
        and ledger["entry_points"],
        "reproduction command ledger is incomplete",
    )
    validate_self_digest(ledger, "ledger_sha256", "reproduction command ledger")
    require(ledger.get("scope") == LEDGER_SCOPE, "reproduction command ledger scope changed")
    placeholders = ledger.get("placeholders")
    require(isinstance(placeholders, dict) and placeholders, "reproduction command placeholders missing")
    require(
        placeholders.get("EVAL_PYTHON") == EVAL_PYTHON_PLACEHOLDER
        and HEX64.fullmatch(
            str(
                json_pointer(
                    environment_receipt,
                    EVAL_PYTHON_PLACEHOLDER["executable_sha256_binding"]["json_pointer"],
                )
            )
        )
        is not None,
        "EVAL_PYTHON is not typed and bound to the environment-receipt interpreter hash",
    )
    entry_points = ledger["entry_points"]
    observed_entry_points: set[str] = set()
    for index, command in enumerate(commands):
        require(
            isinstance(command, dict)
            and set(command)
            == {
                "stage_id",
                "environment_id",
                "cwd",
                "argv",
                "environment",
                "private_inputs",
                "command_kind",
            }
            and command["cwd"] == "${PROJECT_ROOT}"
            and isinstance(command["argv"], list)
            and len(command["argv"]) >= 2
            and all(isinstance(item, str) and item for item in command["argv"])
            and isinstance(command["environment"], dict)
            and isinstance(command["private_inputs"], list),
            f"reproduction command {index} schema changed",
        )
        serialized_command = json.dumps(command, sort_keys=True)
        used_placeholders = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", serialized_command))
        require(used_placeholders <= set(placeholders), f"reproduction command {index} uses an undefined placeholder")
        require(
            set(command["private_inputs"])
            == {name for name in used_placeholders if name.startswith("PRIVATE_")},
            f"reproduction command {index} private-input declaration changed",
        )
        entry_point = str(command["argv"][1])
        safe_relative(entry_point, f"reproduction command {index} entry point")
        observed_entry_points.add(entry_point)
    require(set(entry_points) == observed_entry_points, "reproduction entry-point inventory changed")
    for path, ref in entry_points.items():
        require(
            isinstance(ref, dict) and ref.get("path") == path,
            f"reproduction entry point {path} ref invalid",
        )
        validate_ref(ref, roots, f"reproduction entry point {path}")
        require(ref["root_id"] == "repo_root", f"reproduction entry point {path} is outside repository")
        verify_ref_matches_commit(repo_root, recorded_commit, ref, f"reproduction entry point {path}")
    manual_stages = ledger.get("manual_stages")
    require(
        isinstance(manual_stages, list)
        and len(manual_stages) == ledger["counts"].get("manual_stages") == 12
        and [stage.get("stage_id") for stage in manual_stages if isinstance(stage, dict)]
        == list(MANUAL_STAGE_IDS),
        "reproduction manual-stage inventory changed",
    )
    for index, stage in enumerate(manual_stages):
        stage_id = MANUAL_STAGE_IDS[index]
        target_selection = stage_id in TARGET_SELECTION_MANUAL_STAGE_IDS
        expected_paths = (
            TARGET_SELECTION_RELEASED_INPUT_PATHS
            if target_selection
            else HUMAN_EVALUATION_RELEASED_INPUT_PATHS
        )
        require(
            isinstance(stage, dict)
            and set(stage) == MANUAL_STAGE_FIELDS
            and stage.get("phase")
            == ("target_selection" if target_selection else "human_canonical_evaluation")
            and stage.get("completion_state_at_ledger_freeze")
            == (
                "completed_before_ledger_freeze"
                if target_selection
                else "required_after_ledger_freeze"
            )
            and stage.get("parent_stage_ids") == list(MANUAL_STAGE_PARENTS[stage_id])
            and isinstance(stage.get("instruction"), str)
            and stage["instruction"].strip()
            and isinstance(stage.get("released_input_refs"), dict)
            and set(stage["released_input_refs"]) == set(expected_paths)
            and isinstance(stage.get("unreleased_or_pending_input_roles"), list)
            and stage["unreleased_or_pending_input_roles"]
            and all(
                isinstance(role, str) and ROLE.fullmatch(role) is not None
                for role in stage["unreleased_or_pending_input_roles"]
            )
            and len(stage["unreleased_or_pending_input_roles"])
            == len(set(stage["unreleased_or_pending_input_roles"])),
            f"reproduction manual stage {stage_id} is malformed or incorrectly scoped",
        )
        for role, path in expected_paths.items():
            ref = stage["released_input_refs"][role]
            require(
                isinstance(ref, dict)
                and ref.get("root_id") == "repo_root"
                and ref.get("path") == path,
                f"reproduction manual stage {stage_id} released input {role} is invalid",
            )
            validate_ref(ref, roots, f"reproduction manual stage {stage_id} input {role}")
            verify_ref_matches_commit(
                repo_root,
                recorded_commit,
                ref,
                f"reproduction manual stage {stage_id} input {role}",
            )

    bound_dag_path = validate_ref(
        bindings["pre_metric_artifact_dag"], roots, "pre-metric artifact DAG"
    )
    bound_dag = load_json(bound_dag_path, "pre-metric artifact DAG")
    verify_dag(bound_dag, roots, "pre-metric")
    dag_nodes = {
        str(node["id"]): node
        for node in bound_dag["nodes"]
        if isinstance(node, dict) and "id" in node
    }
    dag_binding_nodes = {
        "evaluation_environment_lock": "evaluation_environment_lock",
        "evaluation_environment_receipt": "release_evaluation_environment_receipt",
        "wan_training_receipt_inventory": "wan_receipt_inventory",
        "reproduction_command_ledger": "executable_command_ledger",
    }
    for binding_name, node_id in dag_binding_nodes.items():
        require(
            node_id in dag_nodes
            and dag_nodes[node_id].get("status") == "materialized"
            and dag_nodes[node_id].get("artifact") == bindings[binding_name],
            f"pre-metric DAG differs from reproducibility binding: {binding_name}",
        )
    for role, component in by_role.items():
        node_id = f"component_{role}"
        expected_ref = {
            "root_id": "repo_root",
            "path": component["path"],
            "sha256": component["sha256"],
            "size_bytes": component["size_bytes"],
        }
        require(
            node_id in dag_nodes
            and dag_nodes[node_id].get("status") == "materialized"
            and dag_nodes[node_id].get("artifact") == expected_ref,
            f"pre-metric DAG differs from component binding: {role}",
        )
    require(
        all(
            name in dag_nodes and dag_nodes[name].get("status") == "pending"
            for name in (
                "human_labels",
                "adjudication",
                "canonical_scores",
                "original_eligibility",
                "formal_metrics",
                "paper_tables",
            )
        ),
        "pre-metric DAG omits a required pending output",
    )
    require(
        value.get("pending_outputs")
        == [
            {"name": name, "status": "not_materialized"}
            for name in ("human_labels", "adjudication", "canonical_scores", "original_eligibility", "formal_metrics", "paper_tables")
        ],
        "pre-metric pending output inventory changed",
    )


def parse_extra_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        require("=" in value, "--root must be ROOT_ID=PATH")
        root_id, raw = value.split("=", 1)
        require(root_id and root_id not in roots, "root ID is blank or repeated")
        require(
            root_id not in RESERVED_EXTRA_ROOT_IDS,
            f"--root may not override reserved root ID: {root_id}",
        )
        root = Path(raw).resolve(strict=True)
        require(root.is_dir() and not root.is_symlink(), f"root {root_id} is not a real directory")
        roots[root_id] = root
    return roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument("--media-snapshot-root", type=Path)
    parser.add_argument("--wan-snapshot-root", type=Path)
    parser.add_argument("--pre-metric-root", type=Path, required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pre-metric-freeze", type=Path, required=True)
    parser.add_argument("--mode", choices=("pre-metric", "final"), required=True)
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--unavailable-sha256", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        roots = {
            "repo_root": args.repo_root.resolve(strict=True),
            "pre_metric_root": args.pre_metric_root.resolve(strict=True),
            **parse_extra_roots(args.root),
        }
        require(
            args.pre_metric_freeze.resolve(strict=True)
            == roots["pre_metric_root"] / "pre_metric_code_freeze_v2.json",
            "pre-metric freeze path is not the bound pre-metric-root manifest",
        )
        if args.mode == "pre-metric":
            require(not args.build_final, "pre-metric mode cannot build a final DAG")
            require(
                args.manifest.resolve(strict=True)
                == roots["pre_metric_root"] / "pre_metric_artifact_dag_v1.json",
                "pre-metric DAG path is not the bound pre-metric-root DAG",
            )
        for root_id, raw in (
            ("review_root", args.review_root),
            ("media_snapshot_root", args.media_snapshot_root),
            ("wan_snapshot_root", args.wan_snapshot_root),
        ):
            if raw is not None:
                roots[root_id] = raw.resolve(strict=True)
        pre_metric = load_json(args.pre_metric_freeze, "pre-metric freeze")
        verify_pre_metric_freeze(pre_metric, roots)
        prefix_path = validate_ref(
            pre_metric["reproducibility_bindings"]["pre_metric_artifact_dag"],
            roots,
            "bound pre-metric artifact DAG",
        )
        prefix = load_json(prefix_path, "bound pre-metric artifact DAG")
        verify_dag(prefix, roots, "pre-metric")
        if args.mode == "final":
            required_final_roots = {
                "final_release_root",
                "final_audit_root",
                "final_review_process_root",
                "canonical_root",
                "eligibility_root",
                "metrics_root",
                "paper_table_root",
            }
            require(
                required_final_roots <= set(roots),
                "final artifact DAG roots are incomplete",
            )
            require(
                args.manifest.resolve()
                == roots["final_release_root"] / "artifact_dag_release_v1.json",
                "final DAG path is not the bound final-release-root manifest",
            )
            if args.build_final:
                final = build_final_dag(
                    prefix=prefix,
                    roots=roots,
                    pre_metric=pre_metric,
                    unavailable_sha256=parse_unavailable_digests(
                        args.unavailable_sha256
                    ),
                )
                write_json_exclusive(args.manifest, final)
            else:
                require(
                    not args.unavailable_sha256,
                    "unavailable SHA arguments are accepted only while building",
                )
        else:
            require(not args.unavailable_sha256, "pre-metric mode does not accept final SHA values")
        manifest = load_json(args.manifest, "artifact DAG")
        counts = verify_dag(manifest, roots, args.mode)
        if args.mode == "final":
            verify_final_prefix_binding(
                final=manifest, prefix=prefix, pre_metric=pre_metric, roots=roots
            )
    except (OSError, json.JSONDecodeError, ArtifactDagError) as exc:
        parser.exit(2, f"artifact-DAG verification refused: {exc}\n")
    print(json.dumps({"status": "verified", "mode": args.mode, "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
