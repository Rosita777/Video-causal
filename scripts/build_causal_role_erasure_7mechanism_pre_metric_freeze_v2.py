#!/usr/bin/env python3
"""Freeze the clean pre-metric implementation and its current artifact DAG.

This command must run from a clean implementation commit.  It writes a fresh
out-of-tree root atomically; the tracked repository never self-references a
commit that contains a generated freeze manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import build_causal_role_erasure_7mechanism_reproducibility_v1 as repro
    import verify_causal_role_erasure_7mechanism_artifact_dag_v1 as dag_verifier
except ModuleNotFoundError:  # imported as scripts.<module> in tests
    from scripts import build_causal_role_erasure_7mechanism_reproducibility_v1 as repro
    from scripts import verify_causal_role_erasure_7mechanism_artifact_dag_v1 as dag_verifier


PROTOCOL = "causal_role_erasure_7m_pre_metric_code_freeze_v2"
STATUS = "clean_tree_components_frozen_before_human_canonical_metrics"
PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
AMENDMENT_PROTOCOL = "causal_role_erasure_7m_evaluation_provenance_amendment_v2"
AMENDMENT_STATUS = "frozen_after_disclosed_global_full_key_access_before_human_labeling"

EXPECTED = {
    "build_registry": "4f7dd07e0ac25d0523e310e8e3c790a5979c685282e047c47dc3e10ecb1f6602",
    "ontology_registry": "37ef4920665e0cbcaae86e0b31257708facc882bb6a8478c6b9e66ecd50e2036",
    "formal_cases": "555ec79a1f084a8b75c802cd570801faf98531fc063fa0eb5170ab8c1d710a04",
    "identification_subset": "07e102ec70f3957b7d6cface7826ccdde428c4a436560a6cc3631c01bbd6c897",
    "baseline_registry": "8a9d92b86bd1ae2f08be37ad6e2d57b6ac1d4877397e080b3046807cfd9228ef",
    "wan_original_aggregate": "42cb825655bffd8137c7dfb3a32d8262487a4e51c6fa59bc82d3401ad7a1568c",
    "trained_wan_aggregate": "a5b4822234806da8e2e41b4ab42da1ba0f44d48f15bfea92d618e454f09d0f25",
    "cog_core_aggregate": "fc25120ba70531fa67ce5b1c5d46fcfba54cea5ccfb56e28e4f4b9bdd00ad6f9",
    "safree_aggregate": "2e73eb99b9352b33703bf922b378c3628226fce0f2ef5aabdbae3efc46f0c484",
    "rebound_registry": "a13682bf3efba4b140480cba184a9ed7c33cc70a7671d2a7cc6086c9e2721ae4",
    "evaluation_registry_file": "c4f10b7129493fd2aa646ab4a38f0b141332dfa8d9da1e64f2c737c33861528f",
    "evaluation_registry_identity": "bafc045ba94a6a5a8771ea8a2dd879c5f42aa219e3b5228b43594494b24eedf4",
    "key_commitments": "825d0d3c207b75469db17aecf712e67c957605c2f1dec74475d10e975f19859c",
    "pass_a_merge": "98b7f1b61070b789cb8755dab56829be06608ddda966dc44d7c11b32fbbae07c",
    "pass_b_merge": "a5cc4a44c4eb017f2bd0f81a67959f5d8848ef460c412d1758521450ac30c457",
    "formal_launch": "833254ac9b38a609f233b0d09d9507d316326ab46a6956a0d07b4c1fdf0d3ff1",
    "audit_manifest": "d8f40e586ce33869508029dd705296e5b7de179851f9a5bf11134898500f1538",
    "audit_queue": "e8a8f9dbd5b45df978aa73a59978b68826634c52d9b94e7d8d3e50c889e64e7b",
}
UNAVAILABLE_UPSTREAM = {
    "target_generation_aggregate": "3b7bc97cd4d2",
    "target_selection_registry": "efe43d540f7a",
    "selected_targets": "d5d97dfb9868",
    "training_input_registry": "cd97ea21e7aa",
    "wan_run_spec_registry": "36c79a862918",
    "t2v_training_registry": "cd71f08fdb9d",
}


class PreMetricFreezeError(ValueError):
    """The implementation tree or pre-metric evidence is not freeze-ready."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreMetricFreezeError(message)


def git_output(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(completed.returncode == 0, f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def clean_commit(project_root: Path) -> dict[str, Any]:
    require(git_output(project_root, "rev-parse", "--is-inside-work-tree") == "true", "project root is not a Git worktree")
    require(git_output(project_root, "status", "--porcelain", "--untracked-files=all") == "", "implementation worktree is not clean")
    commit = git_output(project_root, "rev-parse", "HEAD^{commit}")
    require(git_output(project_root, "cat-file", "-t", commit) == "commit", "HEAD is not a Git commit object")
    tree = git_output(project_root, "rev-parse", f"{commit}^{{tree}}")
    require(git_output(project_root, "cat-file", "-t", tree) == "tree", "HEAD tree is not a Git tree object")
    require(len(commit) == len(tree) == 40, "implementation commit/tree is not SHA-1 shaped")
    return {"commit": commit, "tree": tree, "worktree_clean": True}


def checked_ref(root_id: str, logical: str, actual: Path, expected_sha: str) -> dict[str, Any]:
    require(repro.sha256_file(actual) == expected_sha, f"{logical} SHA mismatch")
    return repro.file_ref(root_id, logical, actual)


def validate_amendment(path: Path) -> dict[str, Any]:
    value = repro.load_json(path, "evaluation provenance amendment")
    require(
        value.get("protocol") == AMENDMENT_PROTOCOL
        and value.get("protocol_version") == PROTOCOL_VERSION
        and value.get("status") == AMENDMENT_STATUS,
        "evaluation provenance amendment identity changed",
    )
    repro.validate_self_digest(value, "amendment_sha256", "evaluation provenance amendment")
    return value


def component(project_root: Path, path: Path, role: str) -> dict[str, Any]:
    require(re.fullmatch(r"[a-z][a-z0-9_]*", role) is not None, f"component role is unsafe: {role}")
    ref = repro.repo_ref(project_root, path)
    return {"path": ref["path"], "sha256": ref["sha256"], "size_bytes": ref["size_bytes"], "role": role}


def materialized_node(node_id: str, artifact: Mapping[str, Any], parents: Sequence[str] = (), checks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return {
        "id": node_id,
        "status": "materialized",
        "parents": list(parents),
        "artifact": dict(artifact),
        "semantic_checks": list(checks),
    }


def pending_node(node_id: str, parents: Sequence[str], reason: str) -> dict[str, Any]:
    return {"id": node_id, "status": "pending", "parents": list(parents), "pending_reason": reason}


def build_dag(
    *,
    sources: Mapping[str, Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    inventory_ref: Mapping[str, Any],
    ledger_ref: Mapping[str, Any],
    lock_ref: Mapping[str, Any],
    environment_receipt_ref: Mapping[str, Any],
) -> dict[str, Any]:
    nodes = [
        materialized_node("static_build", sources["build_registry"], checks=[{"pointer": "/counts/mechanisms", "equals": 7}, {"pointer": "/counts/formal_cases", "equals": 294}, {"pointer": "/counts/training_runs", "equals": 18}]),
        materialized_node("ontology", sources["ontology_registry"], parents=["static_build"]),
        materialized_node("formal_cases", sources["formal_cases"], parents=["static_build"]),
        materialized_node("identification_subset", sources["identification_subset"], parents=["formal_cases"]),
    ]
    unavailable_parents = {
        "target_generation_aggregate": ["static_build"],
        "target_selection_registry": ["target_generation_aggregate"],
        "selected_targets": ["target_selection_registry"],
        "training_input_registry": ["selected_targets", "ontology"],
        "wan_run_spec_registry": ["training_input_registry"],
        "t2v_training_registry": ["training_input_registry"],
    }
    for node_id, prefix in UNAVAILABLE_UPSTREAM.items():
        nodes.append(
            {
                "id": node_id,
                "status": "declared_unavailable",
                "parents": unavailable_parents[node_id],
                "expected_sha256_prefix": prefix,
                "access_conditions": "The small authority file remains in the unmounted persistent A100 project and must be copied or released before final verification.",
                "reproduction_limit": "Only its published digest prefix and downstream binding can be checked from this local pre-metric snapshot.",
            }
        )
    nodes.extend(
        [
            materialized_node("baseline_registry", sources["baseline_registry"], parents=["t2v_training_registry"], checks=[{"pointer": "/expected_cogvideox_outputs", "equals": 1470}, {"pointer": "/formal_generation_authorized", "equals": True}]),
            materialized_node("wan_original_generation", sources["wan_original_aggregate"], parents=["formal_cases"], checks=[{"pointer": "/validated_videos", "equals": 294}, {"pointer": "/status", "equals": "completed"}]),
            materialized_node("wan_receipt_inventory", inventory_ref, parents=["wan_run_spec_registry"]),
            materialized_node("trained_wan_generation", sources["trained_wan_aggregate"], parents=["wan_receipt_inventory", "formal_cases", "identification_subset"], checks=[{"pointer": "/validated_videos", "equals": 684}, {"pointer": "/status", "equals": "completed"}]),
            materialized_node("cog_core_generation", sources["cog_core_aggregate"], parents=["baseline_registry", "formal_cases"], checks=[{"pointer": "/validated_videos", "equals": 1176}, {"pointer": "/status", "equals": "completed"}]),
            materialized_node("safree_generation", sources["safree_aggregate"], parents=["baseline_registry", "formal_cases"], checks=[{"pointer": "/validated_videos", "equals": 294}, {"pointer": "/status", "equals": "completed"}]),
            materialized_node("rebound_manifests", sources["rebound_registry"], parents=["wan_original_generation", "trained_wan_generation", "cog_core_generation", "safree_generation"]),
            {
                "id": "anonymous_package_receipt",
                "status": "private_commitment",
                "parents": ["rebound_manifests", "evaluation_code_registry"],
                "commitment_sha256": "63933393ea04883bbec1f3d383f5ef49d4b933adf66040755bb622ad0170837c",
                "release_statement": "A sanitized derivative must bind this complete private-parent digest without releasing the blind secret.",
            },
            materialized_node("evaluation_code_registry", sources["evaluation_registry"], parents=[]),
            materialized_node("key_commitments", sources["key_commitments"], parents=["anonymous_package_receipt"]),
            materialized_node("pass_a_merge", sources["pass_a_merge"], parents=["key_commitments"], checks=[{"pointer": "/row_count", "equals": 2448}, {"pointer": "/scientific_zero_fallbacks", "equals": 0}]),
            materialized_node("pass_b_merge", sources["pass_b_merge"], parents=["key_commitments"], checks=[{"pointer": "/row_count", "equals": 2448}, {"pointer": "/scientific_zero_fallbacks", "equals": 0}]),
            materialized_node("formal_launch", sources["formal_launch"], parents=["pass_a_merge", "pass_b_merge"]),
            materialized_node("initial_human_audit", sources["audit_manifest"], parents=["pass_a_merge", "pass_b_merge"], checks=[{"pointer": "/counts/initial_audit_atoms", "equals": 3633}, {"pointer": "/counts/initial_audit_videos", "equals": 2009}]),
            materialized_node("evaluation_environment_lock", lock_ref, parents=[]),
            materialized_node(
                "release_evaluation_environment_receipt",
                environment_receipt_ref,
                parents=["evaluation_environment_lock", "component_reproducibility_builder"],
                checks=[
                    {"pointer": "/status", "equals": repro.ENVIRONMENT_STATUS},
                    {"pointer": "/tests/status", "equals": "passed"},
                ],
            ),
            materialized_node("executable_command_ledger", ledger_ref, parents=[]),
        ]
    )
    for item in components:
        nodes.append(
            materialized_node(
                f"component_{item['role']}",
                {"root_id": "repo_root", "path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]},
            )
        )
    nodes.extend(
        [
            pending_node("reviewer_deliveries", ["initial_human_audit", "release_evaluation_environment_receipt", "component_reviewer_instructions", "component_review_process_receipt_builder"], "two isolated reviewer deliveries have not been frozen"),
            pending_node("human_labels", ["reviewer_deliveries"], "two independent reviews have not been completed"),
            pending_node("adjudication", ["human_labels"], "human disagreements have not been adjudicated"),
            pending_node("audit_expansion", ["adjudication", "component_canonicalization_wrapper"], "the registered expansion decision and final reviewer projections have not been materialized"),
            pending_node("final_review_process_receipt", ["audit_expansion", "component_review_process_receipt_builder"], "the final projected review round and process receipt have not been frozen"),
            pending_node("canonical_scores", ["final_review_process_receipt", "component_canonicalization_wrapper"], "canonical scores have not been materialized"),
            pending_node("original_eligibility", ["canonical_scores"], "Original eligibility has not been materialized"),
            pending_node("formal_metrics", ["original_eligibility", "component_formal_metrics_wrapper"], "formal metrics have not been materialized"),
            pending_node("paper_tables", ["formal_metrics", "component_final_table_exporter"], "final paper tables have not been materialized"),
        ]
    )
    counts = dict(sorted(Counter(str(node["status"]) for node in nodes).items()))
    return repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": dag_verifier.DAG_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": "completed_prefix_bound_with_explicit_pending_human_branch",
            "nodes": nodes,
            "counts": counts,
        },
        "dag_sha256",
    )


def build_freeze(
    *,
    project_root: Path,
    review_root: Path,
    media_snapshot_root: Path,
    wan_snapshot_root: Path,
    output_root: Path,
    environment_lock: Path,
    environment_receipt: Path,
    reviewer_instructions: Path,
    amendment: Path,
    canonicalization_wrapper: Path,
    formal_metrics_wrapper: Path,
    final_table_exporter: Path,
    extra_components: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    review_root = review_root.resolve(strict=True)
    media_snapshot_root = media_snapshot_root.resolve(strict=True)
    wan_snapshot_root = wan_snapshot_root.resolve(strict=True)
    output_root = output_root.resolve()
    try:
        output_root.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise PreMetricFreezeError("pre-metric freeze root must be outside the Git checkout")
    require(not output_root.exists() and not output_root.is_symlink(), "pre-metric freeze root must be fresh")
    implementation_commit = clean_commit(project_root)
    amendment_value = validate_amendment(amendment)
    repro.validate_environment_receipt(
        project_root=project_root,
        lock_path=environment_lock,
        receipt_path=environment_receipt,
    )

    paths = {
        "build_registry": project_root / "data/causal_role_erasure_7mechanism_main_v2/build_registry.json",
        "ontology_registry": project_root / "data/causal_role_erasure_7mechanism_main_v2/ontology_registry.json",
        "formal_cases": project_root / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv",
        "identification_subset": project_root / "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv",
        "baseline_registry": media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json",
        "wan_original_aggregate": wan_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2/wan_original_aggregate.json",
        "trained_wan_aggregate": media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_aggregate.json",
        "cog_core_aggregate": media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2/aggregate.json",
        "safree_aggregate": media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2/aggregate.json",
        "rebound_registry": review_root / "rebound_manifests/registry.json",
        "evaluation_registry": review_root / "review_package/public/evaluation_code_registry.json",
        "key_commitments": review_root / "review_package/public/key_commitments.json",
        "pass_a_merge": review_root / "merged/pass_a/merge_manifest.json",
        "pass_b_merge": review_root / "merged/pass_b/merge_manifest.json",
        "formal_launch": review_root / "formal_launch_receipt.json",
        "audit_manifest": review_root / "human_audit_stage0/audit_manifest.json",
        "audit_queue": review_root / "human_audit_stage0/public/human_audit_queue.csv",
    }
    logical = {
        "build_registry": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/build_registry.json"),
        "ontology_registry": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/ontology_registry.json"),
        "formal_cases": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"),
        "identification_subset": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv"),
        "baseline_registry": ("media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json"),
        "wan_original_aggregate": ("wan_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2/wan_original_aggregate.json"),
        "trained_wan_aggregate": ("media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_aggregate.json"),
        "cog_core_aggregate": ("media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2/aggregate.json"),
        "safree_aggregate": ("media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2/aggregate.json"),
        "rebound_registry": ("review_root", "rebound_manifests/registry.json"),
        "evaluation_registry": ("review_root", "review_package/public/evaluation_code_registry.json"),
        "key_commitments": ("review_root", "review_package/public/key_commitments.json"),
        "pass_a_merge": ("review_root", "merged/pass_a/merge_manifest.json"),
        "pass_b_merge": ("review_root", "merged/pass_b/merge_manifest.json"),
        "formal_launch": ("review_root", "formal_launch_receipt.json"),
        "audit_manifest": ("review_root", "human_audit_stage0/audit_manifest.json"),
        "audit_queue": ("review_root", "human_audit_stage0/public/human_audit_queue.csv"),
    }
    expected_key = {
        "evaluation_registry": "evaluation_registry_file",
        "pass_a_merge": "pass_a_merge",
        "pass_b_merge": "pass_b_merge",
    }
    sources = {
        name: checked_ref(
            logical[name][0], logical[name][1], paths[name], EXPECTED[expected_key.get(name, name)]
        )
        for name in paths
    }

    registry = repro.load_json(paths["evaluation_registry"], "evaluation code registry")
    require(registry.get("registry_sha256") == EXPECTED["evaluation_registry_identity"], "evaluation registry identity changed")
    v1_files = registry.get("files")
    require(isinstance(v1_files, list) and len(v1_files) == 5, "evaluation registry must bind five v1 files")
    for ref in v1_files:
        require(isinstance(ref, dict), "evaluation registry file ref malformed")
        path = project_root / repro.safe_relative(str(ref.get("path", "")), "v1 source")
        repro.regular_file(path, "v1 source")
        require(repro.sha256_file(path) == ref.get("sha256") and path.stat().st_size == ref.get("size_bytes"), "v1 source changed")

    roles_and_paths = [
        ("evaluation_provenance_amendment", amendment),
        ("canonicalization_wrapper", canonicalization_wrapper),
        ("formal_metrics_wrapper", formal_metrics_wrapper),
        ("final_table_exporter", final_table_exporter),
        ("reviewer_instructions", reviewer_instructions),
        ("reproducibility_builder", project_root / "scripts/build_causal_role_erasure_7mechanism_reproducibility_v1.py"),
        ("review_process_receipt_builder", project_root / "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py"),
        ("pre_metric_freeze_builder", Path(__file__)),
        ("artifact_dag_verifier", project_root / "scripts/verify_causal_role_erasure_7mechanism_artifact_dag_v1.py"),
        ("evaluation_environment_lock", environment_lock),
        ("reproducibility_guide", project_root / "reproducibility/causal_role_erasure_7m/README.md"),
        *extra_components,
    ]
    require(len({role for role, _ in roles_and_paths}) == len(roles_and_paths), "component roles repeat")
    components = sorted((component(project_root, path, role) for role, path in roles_and_paths), key=lambda item: item["role"])

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    require(parent.is_dir() and not parent.is_symlink(), "pre-metric parent is not a real directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=parent))
    try:
        inventory_path = stage / "wan_training_receipt_inventory_v1.json"
        ledger_path = stage / "reproduction_command_ledger_v1.json"
        environment_receipt_path = stage / "evaluation_environment_receipt_v1.json"
        dag_path = stage / "pre_metric_artifact_dag_v1.json"
        manifest_path = stage / "pre_metric_code_freeze_v2.json"
        repro.freeze_wan_receipt_inventory(
            project_root=project_root,
            run_matrix_path=paths["formal_cases"].parent / "run_matrix.csv",
            eval_run_manifest_path=media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_run_manifest.json",
            eval_aggregate_path=paths["trained_wan_aggregate"],
            output_path=inventory_path,
        )
        repro.freeze_command_ledger(
            project_root=project_root,
            wan_original_run_manifest=wan_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2/wan_original_run_manifest.json",
            trained_wan_run_manifest=media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_run_manifest.json",
            core_queue_plan=media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2/queue_plan.json",
            safree_queue_plan=media_snapshot_root / "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2/queue_plan.json",
            formal_launch_receipt=paths["formal_launch"],
            output_path=ledger_path,
        )
        with environment_receipt.open("rb") as source, environment_receipt_path.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        inventory_ref = repro.file_ref("pre_metric_root", inventory_path.name, inventory_path)
        ledger_ref = repro.file_ref("pre_metric_root", ledger_path.name, ledger_path)
        environment_receipt_ref = repro.file_ref(
            "pre_metric_root", environment_receipt_path.name, environment_receipt_path
        )
        lock_ref = repro.repo_ref(project_root, environment_lock)
        dag = build_dag(
            sources=sources,
            components=components,
            inventory_ref=inventory_ref,
            ledger_ref=ledger_ref,
            lock_ref=lock_ref,
            environment_receipt_ref=environment_receipt_ref,
        )
        repro.write_json_exclusive(dag_path, dag)
        dag_ref = repro.file_ref("pre_metric_root", dag_path.name, dag_path)

        source_bindings = {
            "evaluation_code_registry": sources["evaluation_registry"],
            "key_commitments": sources["key_commitments"],
            "formal_launch_receipt": sources["formal_launch"],
            "pass_a_merge_manifest": sources["pass_a_merge"],
            "pass_b_merge_manifest": sources["pass_b_merge"],
            "human_audit_manifest": sources["audit_manifest"],
            "human_audit_queue": sources["audit_queue"],
            "formal_cases": sources["formal_cases"],
            "identification_subset": sources["identification_subset"],
            "private_key_commitments": {
                name: {"sha256": amendment_value["bindings"][name]["sha256"], "row_count": amendment_value["bindings"][name]["row_count"]}
                for name in ("tier_0_audit_strata", "tier_1_original_only", "tier_2_full")
            },
        }
        registry_authority = {
            "path": "review_package/public/evaluation_code_registry.json",
            "sha256": sources["evaluation_registry"]["sha256"],
            "size_bytes": sources["evaluation_registry"]["size_bytes"],
            "registry_sha256": registry["registry_sha256"],
        }
        manifest = repro.with_self_digest(
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "protocol_version": PROTOCOL_VERSION,
                "status": STATUS,
                "implementation_commit": implementation_commit,
                "deviation": {
                    "global_full_key_opened_before_canonical_freeze": True,
                    "human_review_process_local_answer_key_blindness_preserved": True,
                    "v1_files_byte_identical": True,
                    "scientific_rules_changed": False,
                    "unchanged_v1_final_stages_forbidden": True,
                },
                "v1_authority": {"evaluation_code_registry": registry_authority, "files": v1_files},
                "source_bindings": source_bindings,
                "components": components,
                "reproducibility_bindings": {
                    "evaluation_environment_lock": lock_ref,
                    "evaluation_environment_receipt": environment_receipt_ref,
                    "wan_training_receipt_inventory": inventory_ref,
                    "reproduction_command_ledger": ledger_ref,
                    "pre_metric_artifact_dag": dag_ref,
                },
                "pending_outputs": [
                    {"name": name, "status": "not_materialized"}
                    for name in ("human_labels", "adjudication", "canonical_scores", "original_eligibility", "formal_metrics", "paper_tables")
                ],
            },
            "manifest_sha256",
        )
        repro.write_json_exclusive(manifest_path, manifest)
        roots = {
            "repo_root": project_root,
            "review_root": review_root,
            "media_snapshot_root": media_snapshot_root,
            "wan_snapshot_root": wan_snapshot_root,
            "pre_metric_root": stage,
        }
        dag_verifier.verify_dag(dag, roots, "pre-metric")
        dag_verifier.verify_pre_metric_freeze(manifest, roots)
        require(clean_commit(project_root) == implementation_commit, "implementation tree changed during freeze")
        os.replace(stage, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_extra_component(value: str) -> tuple[str, Path]:
    require("=" in value, "--extra-component must be ROLE=PATH")
    role, raw = value.split("=", 1)
    require(re.fullmatch(r"[a-z][a-z0-9_]*", role) is not None and raw, "--extra-component role/path blank or unsafe")
    return role, Path(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--media-snapshot-root", type=Path, required=True)
    parser.add_argument("--wan-snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--reviewer-instructions", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--canonicalization-wrapper", type=Path, required=True)
    parser.add_argument("--formal-metrics-wrapper", type=Path, required=True)
    parser.add_argument("--final-table-exporter", type=Path, required=True)
    parser.add_argument("--extra-component", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_freeze(
            project_root=args.project_root,
            review_root=args.review_root,
            media_snapshot_root=args.media_snapshot_root,
            wan_snapshot_root=args.wan_snapshot_root,
            output_root=args.output_root,
            environment_lock=args.environment_lock,
            environment_receipt=args.environment_receipt,
            reviewer_instructions=args.reviewer_instructions,
            amendment=args.amendment,
            canonicalization_wrapper=args.canonicalization_wrapper,
            formal_metrics_wrapper=args.formal_metrics_wrapper,
            final_table_exporter=args.final_table_exporter,
            extra_components=[parse_extra_component(value) for value in args.extra_component],
        )
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError, repro.ReproducibilityError, dag_verifier.ArtifactDagError, PreMetricFreezeError) as exc:
        parser.exit(2, f"pre-metric freeze refused: {exc}\n")
    print(json.dumps({"status": result["status"], "manifest_sha256": result["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
