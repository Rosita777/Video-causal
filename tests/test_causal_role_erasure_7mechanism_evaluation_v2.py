from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v1 as legacy_canonical
from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v2 as canonical_v2
from scripts import build_causal_role_erasure_7mechanism_review_process_receipt_v2 as review_process
from scripts import build_causal_role_erasure_7mechanism_reproducibility_v1 as reproducibility
from scripts import compute_causal_role_erasure_7mechanism_formal_metrics_v1 as legacy_metrics
from scripts import compute_causal_role_erasure_7mechanism_formal_metrics_v2 as metrics_v2
from tests.test_causal_role_erasure_7mechanism_formal_metrics_v1 import (
    complete_human_queue,
    synthetic_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    PROJECT_ROOT
    / "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json"
)
FORMAL_CASES = (
    PROJECT_ROOT / "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"
)


def _file_binding(source_path: Path, **extra):
    return {
        "sha256": canonical_v2.sha256_file(source_path),
        "size_bytes": source_path.stat().st_size,
        **extra,
    }


def _logical(root_id: str, path: str, source: Path):
    return {
        "root_id": root_id,
        "path": path,
        "sha256": canonical_v2.sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_v2.canonical_json_bytes(value))


def _seal(value, field: str):
    body = dict(value)
    body.pop(field, None)
    return {**body, field: canonical_v2.object_sha256(body)}


def _synthetic_amendment(
    path: Path,
    inputs,
    initial_audit: Path,
    formal_launch_receipt: Path,
) -> Path:
    value = copy.deepcopy(json.loads(AMENDMENT.read_text(encoding="utf-8")))
    bindings = value["bindings"]
    commitments = inputs["key_commitments"]
    audit_manifest = initial_audit / "audit_manifest.json"
    queue = initial_audit / "public/human_audit_queue.csv"
    audit_value = json.loads(audit_manifest.read_text(encoding="utf-8"))
    bindings["key_commitments"] = _file_binding(commitments)
    bindings["formal_launch_receipt"] = _file_binding(formal_launch_receipt)
    bindings["pass_a_merge_manifest"] = _file_binding(
        inputs["scores_a"].parent / "merge_manifest.json"
    )
    bindings["pass_b_merge_manifest"] = _file_binding(
        inputs["scores_b"].parent / "merge_manifest.json"
    )
    bindings["human_audit_manifest"] = _file_binding(
        audit_manifest,
        videos=audit_value["counts"]["videos"],
        atoms=audit_value["counts"]["atoms"],
        initial_audit_atoms=audit_value["counts"]["initial_audit_atoms"],
        initial_audit_videos=audit_value["counts"]["initial_audit_videos"],
    )
    bindings["human_audit_queue"] = _file_binding(
        queue,
        rows=audit_value["counts"]["initial_audit_atoms"],
        review_fields_blank=True,
    )
    commitments_value = json.loads(commitments.read_text(encoding="utf-8"))
    for name in (
        "tier_0_audit_strata",
        "tier_1_original_only",
        "tier_2_full",
    ):
        bindings[name] = commitments_value[name]
    value = _seal(value, "amendment_sha256")
    _write_json(path, value)
    return path


def _git_snapshot_commit(paths: list[Path], index_path: Path) -> tuple[str, str]:
    environment = {
        **os.environ,
        "GIT_INDEX_FILE": str(index_path),
        "GIT_AUTHOR_NAME": "Evaluation v2 test",
        "GIT_AUTHOR_EMAIL": "evaluation-v2-test@example.invalid",
        "GIT_COMMITTER_NAME": "Evaluation v2 test",
        "GIT_COMMITTER_EMAIL": "evaluation-v2-test@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    }

    def run(*arguments: str, input_text: str | None = None) -> str:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    run("read-tree", "HEAD")
    relatives = [path.resolve().relative_to(PROJECT_ROOT).as_posix() for path in paths]
    run("add", "-f", "--", *relatives)
    tree = run("write-tree")
    parent = run("rev-parse", "HEAD")
    commit = run(
        "commit-tree",
        tree,
        "-p",
        parent,
        input_text="synthetic evaluation-v2 authority\n",
    )
    assert run("cat-file", "-t", commit) == "commit"
    assert run("cat-file", "-t", tree) == "tree"
    return commit, tree


def _fake_reproducibility_artifacts(pre_metric_root: Path) -> dict[str, Path]:
    lock_path = (
        PROJECT_ROOT
        / canonical_v2.CRITICAL_COMPONENT_PATHS["evaluation_environment_lock"]
    )
    test_paths = list(reproducibility.RELEASE_TEST_PATHS)
    environment_receipt_path = (
        pre_metric_root / "evaluation_environment_receipt_v1.json"
    )
    pins = reproducibility.parse_lock(lock_path)
    installed = reproducibility.installed_distributions()
    assert all(installed.get(name) == version for name, version in pins.items())
    assert not (
        set(installed)
        - set(pins)
        - reproducibility.BOOTSTRAP_DISTRIBUTIONS
    )
    codec_versions = {
        "pyav_libraries": reproducibility.isolated_json_probe(
            "import av,json; print(json.dumps({k:list(v) for k,v in sorted(av.library_versions.items())},sort_keys=True))",
            "PyAV",
        ),
        "opencv": reproducibility.isolated_json_probe(
            "import cv2,json; print(json.dumps(cv2.__version__))", "OpenCV"
        ),
    }
    codec_versions.update(
        reproducibility.isolated_json_probe(
            "import json; from PIL import Image,features; print(json.dumps({'pillow':Image.__version__,'libjpeg':features.version_codec('jpg')},sort_keys=True))",
            "Pillow",
        )
    )
    environment_receipt = _seal(
        {
            "schema_version": 1,
            "protocol": reproducibility.ENVIRONMENT_PROTOCOL,
            "protocol_version": canonical_v2.PROTOCOL_VERSION,
            "status": reproducibility.ENVIRONMENT_STATUS,
            "scope": {
                "release_time_reproduction_environment": True,
                "historical_execution_environment_claim": False,
                "hosted_model_bitwise_replay_claim": False,
                "frozen_completed_scores_are_downstream_authority": True,
            },
            "lock": _logical(
                "repo_root",
                canonical_v2.CRITICAL_COMPONENT_PATHS[
                    "evaluation_environment_lock"
                ],
                lock_path,
            ),
            "python": {
                "implementation": "CPython",
                "version": platform.python_version(),
                "executable_sha256": canonical_v2.sha256_file(
                    Path(sys.executable).resolve()
                ),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "distributions": dict(sorted(pins.items())),
            "bootstrap_distributions": {
                name: installed[name]
                for name in sorted(reproducibility.BOOTSTRAP_DISTRIBUTIONS)
                if name in installed
            },
            "codec_versions": codec_versions,
            "external_tools": {
                "latexmk": reproducibility.external_tool_receipt(
                    "latexmk", ("-v",)
                )
            },
            "tests": {
                "command": [
                    "${EVAL_PYTHON}",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    *test_paths,
                ],
                "files": [
                    _logical("repo_root", path, PROJECT_ROOT / path)
                    for path in test_paths
                ],
                "status": "passed",
                "return_code": 0,
                "summary": f"{reproducibility.EXPECTED_RELEASE_TEST_PASSES} passed",
            },
        },
        "receipt_sha256",
    )
    _write_json(environment_receipt_path, environment_receipt)

    inventory_path = pre_metric_root / "wan_training_receipt_inventory_v1.json"
    inventory = _seal(
        {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7m_wan_training_receipt_inventory_v1",
            "protocol_version": canonical_v2.PROTOCOL_VERSION,
            "status": "18_receipts_indexed_from_individually_validated_eval_run_manifest",
            "verification_basis": {
                "receipt_bytes_reopened_by_this_builder": False,
                "receipt_bytes_and_semantics_validated_before_frozen_eval_run_manifest": True,
                "source_validator": "synthetic-test-validator",
            },
            "sources": {},
            "counts": {
                "receipts": 18,
                "main_runs": 14,
                "identification_runs": 4,
                "erase_updates_per_run": 100,
                "preserve_updates_per_run": 100,
                "eligible_checkpoint_step": 200,
                "validated_generation_rows": 684,
            },
            "items": [{"run_id": f"synthetic_{index:02d}"} for index in range(18)],
        },
        "inventory_sha256",
    )
    _write_json(inventory_path, inventory)

    ledger_path = pre_metric_root / "reproduction_command_ledger_v1.json"
    ledger = _seal(
        {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7m_executable_command_ledger_v1",
            "protocol_version": canonical_v2.PROTOCOL_VERSION,
            "status": "full_reproduction_commands_frozen_before_canonical_metrics",
            "scope": {
                "commands_are_sanitized": True,
                "commands_are_argument_vectors_not_shell_strings": True,
            },
            "placeholders": {},
            "commands": [
                {
                    "stage_id": "synthetic_evaluation_v2",
                    "environment_id": "evaluation",
                    "cwd": "${PROJECT_ROOT}",
                    "argv": ["${EVAL_PYTHON}", "synthetic.py"],
                    "environment": {},
                    "private_inputs": [],
                    "command_kind": "release_reproduction_command",
                }
            ],
            "manual_stages": [],
            "realized_command_evidence": {},
        },
        "ledger_sha256",
    )
    _write_json(ledger_path, ledger)

    dag_path = pre_metric_root / "pre_metric_artifact_dag_v1.json"
    nodes = [
        {
            "id": name,
            "status": "pending",
            "parents": [],
            "pending_reason": "synthetic pre-metric test fixture",
        }
        for name in canonical_v2.PENDING_OUTPUTS
    ]
    dag = _seal(
        {
            "schema_version": 1,
            "protocol": "causal_role_erasure_7m_artifact_dag_v1",
            "protocol_version": canonical_v2.PROTOCOL_VERSION,
            "status": "completed_prefix_bound_with_explicit_pending_human_branch",
            "nodes": nodes,
            "counts": {"pending": len(nodes)},
        },
        "dag_sha256",
    )
    _write_json(dag_path, dag)
    return {
        "environment_receipt": environment_receipt_path,
        "inventory": inventory_path,
        "ledger": ledger_path,
        "dag": dag_path,
    }


def _pre_metric_manifest(
    tmp_path: Path,
    amendment: Path,
    review_root: Path,
    initial_audit: Path,
    implementation_commit: str,
    implementation_tree: str,
) -> Path:
    amendment_value = json.loads(amendment.read_text(encoding="utf-8"))
    source_paths = {
        "evaluation_code_registry": review_root
        / "review_package/public/evaluation_code_registry.json",
        "key_commitments": review_root
        / "review_package/public/key_commitments.json",
        "formal_launch_receipt": review_root / "formal_launch_receipt.json",
        "pass_a_merge_manifest": review_root / "merged/pass_a/merge_manifest.json",
        "pass_b_merge_manifest": review_root / "merged/pass_b/merge_manifest.json",
        "human_audit_manifest": initial_audit / "audit_manifest.json",
        "human_audit_queue": initial_audit / "public/human_audit_queue.csv",
        "formal_cases": FORMAL_CASES,
        "identification_subset": PROJECT_ROOT
        / "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv",
    }
    source_bindings = {
        name: _logical(root_id, path, source_paths[name])
        for name, (root_id, path) in canonical_v2.SOURCE_BINDING_PATHS.items()
    }
    source_bindings["private_key_commitments"] = {
        name: dict(amendment_value["bindings"][name])
        for name in (
            "tier_0_audit_strata",
            "tier_1_original_only",
            "tier_2_full",
        )
    }
    components = []
    for role, relative in canonical_v2.CRITICAL_COMPONENT_PATHS.items():
        source = PROJECT_ROOT / relative
        components.append(
            {
                "path": relative,
                "sha256": canonical_v2.sha256_file(source),
                "size_bytes": source.stat().st_size,
                "role": role,
            }
        )
    components.sort(key=lambda row: row["role"])
    pre_metric_root = tmp_path / "pre_metric"
    repro = _fake_reproducibility_artifacts(pre_metric_root)
    value = {
        "schema_version": 1,
        "protocol": canonical_v2.PRE_METRIC_PROTOCOL,
        "protocol_version": canonical_v2.PROTOCOL_VERSION,
        "status": canonical_v2.PRE_METRIC_STATUS,
        "deviation": dict(canonical_v2.PRE_METRIC_DEVIATION),
        "v1_authority": {
            "evaluation_code_registry": {
                "path": "review_package/public/evaluation_code_registry.json",
                "sha256": canonical_v2.REGISTRY_FILE_SHA256,
                "size_bytes": canonical_v2.REGISTRY_FILE_SIZE,
                "registry_sha256": canonical_v2.REGISTRY_SHA256,
            },
            "files": [dict(row) for row in canonical_v2.V1_FILES],
        },
        "source_bindings": source_bindings,
        "components": components,
        "reproducibility_bindings": {
            "evaluation_environment_lock": _logical(
                "repo_root",
                canonical_v2.CRITICAL_COMPONENT_PATHS["evaluation_environment_lock"],
                PROJECT_ROOT
                / canonical_v2.CRITICAL_COMPONENT_PATHS[
                    "evaluation_environment_lock"
                ],
            ),
            "evaluation_environment_receipt": _logical(
                "pre_metric_root",
                repro["environment_receipt"].name,
                repro["environment_receipt"],
            ),
            "wan_training_receipt_inventory": _logical(
                "pre_metric_root", repro["inventory"].name, repro["inventory"]
            ),
            "reproduction_command_ledger": _logical(
                "pre_metric_root", repro["ledger"].name, repro["ledger"]
            ),
            "pre_metric_artifact_dag": _logical(
                "pre_metric_root", repro["dag"].name, repro["dag"]
            ),
        },
        "pending_outputs": [
            {"name": name, "status": "not_materialized"}
            for name in canonical_v2.PENDING_OUTPUTS
        ],
        "implementation_commit": {
            "commit": implementation_commit,
            "tree": implementation_tree,
            "worktree_clean": True,
        },
    }
    value = _seal(value, "manifest_sha256")
    path = pre_metric_root / "pre_metric_code_freeze_v2.json"
    _write_json(path, value)
    return path


def _write_reviewer_completion(
    source_queue: Path, completed_human: Path, reviewer_id: str, output: Path
) -> None:
    with source_queue.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or ())
        source_rows = list(reader)
    with completed_human.open(newline="", encoding="utf-8") as handle:
        completed = {
            row["audit_id"]: row for row in csv.DictReader(handle)
        }
    own = {f"{reviewer_id}_score", f"{reviewer_id}_notes"}
    rows = []
    for source in source_rows:
        row = dict(source)
        for field in review_process.SCORE_FIELDS:
            row[field] = completed[source["audit_id"]][field] if field in own else ""
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(legacy_canonical.csv_bytes(rows, header))


def _complete_new_expansion_atoms(
    queue_path: Path, audit_key_path: Path, output_path: Path
) -> None:
    atoms = {
        row["audit_id"]: row
        for row in legacy_canonical.load_jsonl(audit_key_path, "audit key")
    }
    rows = legacy_canonical.load_csv(
        queue_path, legacy_canonical.HUMAN_QUEUE_FIELDS, "expanded human queue"
    )
    for row in rows:
        score = str(atoms[row["audit_id"]]["a_score"])
        for reviewer_id in ("reviewer_1", "reviewer_2"):
            score_field = f"{reviewer_id}_score"
            notes_field = f"{reviewer_id}_notes"
            if row[score_field] == "":
                row[score_field] = score
                row[notes_field] = f"independent {reviewer_id} expansion review"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        legacy_canonical.csv_bytes(rows, legacy_canonical.HUMAN_QUEUE_FIELDS)
    )


def _review_process_receipt(
    tmp_path: Path,
    review_root: Path,
    audit: Path,
    completed_human: Path,
    reviewer_instructions: Path,
    pre_metric: Path,
    initial_process_receipt: Path | None = None,
) -> Path:
    process_root = tmp_path / "review_process"
    process_root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    deliveries = {}
    for reviewer_id in ("reviewer_1", "reviewer_2"):
        deliveries[reviewer_id] = tmp_path / f"{reviewer_id}_delivery"
        receipts[reviewer_id] = process_root / f"{reviewer_id}_delivery.json"
        review_process.prepare_delivery(
            project_root=PROJECT_ROOT,
            pre_metric_path=pre_metric,
            review_root=review_root,
            audit_manifest_path=audit / "audit_manifest.json",
            public_root=audit / "public",
            reviewer_id=reviewer_id,
            reviewer_instructions=reviewer_instructions,
            delivery_root=deliveries[reviewer_id],
            output_receipt=receipts[reviewer_id],
        )
    reviewer_files = {
        reviewer_id: process_root / f"{reviewer_id}_completed.csv"
        for reviewer_id in ("reviewer_1", "reviewer_2")
    }
    for reviewer_id, output in reviewer_files.items():
        _write_reviewer_completion(
            audit / "public/human_audit_queue.csv",
            completed_human,
            reviewer_id,
            output,
        )
    attestation = process_root / "independence_attestation.json"
    _write_json(
        attestation,
        {
            "schema_version": 1,
            "protocol": review_process.ATTESTATION_PROTOCOL,
            "protocol_version": review_process.PROTOCOL_VERSION,
            "status": review_process.ATTESTATION_STATUS,
            "reviewer_1_completed_sha256": canonical_v2.sha256_file(
                reviewer_files["reviewer_1"]
            ),
            "reviewer_2_completed_sha256": canonical_v2.sha256_file(
                reviewer_files["reviewer_2"]
            ),
            "reviewer_1_delivery_receipt_sha256": canonical_v2.sha256_file(
                receipts["reviewer_1"]
            ),
            "reviewer_2_delivery_receipt_sha256": canonical_v2.sha256_file(
                receipts["reviewer_2"]
            ),
            "reviewer_declarations": {
                reviewer_id: {
                    "did_not_access_peer_labels": True,
                    "did_not_access_answer_key": True,
                    "did_not_access_method_mapping": True,
                    "did_not_access_private_directory": True,
                    "did_not_access_parent_workspace": True,
                }
                for reviewer_id in ("reviewer_1", "reviewer_2")
            },
            "both_reviews_frozen_before_merge": True,
            "coordinator_merged_only_after_both_frozen": True,
        },
    )
    receipt = process_root / "review_process_receipt_v2.json"
    review_process.freeze_process(
        project_root=PROJECT_ROOT,
        pre_metric_path=pre_metric,
        review_root=review_root,
        audit_manifest_path=audit / "audit_manifest.json",
        public_root=audit / "public",
        reviewer_instructions=reviewer_instructions,
        reviewer_1_delivery_receipt=receipts["reviewer_1"],
        reviewer_2_delivery_receipt=receipts["reviewer_2"],
        reviewer_1_completed=reviewer_files["reviewer_1"],
        reviewer_2_completed=reviewer_files["reviewer_2"],
        independence_attestation=attestation,
        completed_human=completed_human,
        output_receipt=receipt,
        initial_process_receipt=initial_process_receipt,
    )
    return receipt


@pytest.fixture(scope="module")
def synthetic_v2_pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("causal7m_v2")
    authority_root = Path(
        tempfile.mkdtemp(prefix=".pytest-evaluation-v2-", dir=PROJECT_ROOT)
    )
    original_component_paths = dict(canonical_v2.CRITICAL_COMPONENT_PATHS)
    original_metrics_provenance_paths = dict(
        metrics_v2.provenance.CRITICAL_COMPONENT_PATHS
    )
    original_canonical_file = canonical_v2.__file__
    original_metrics_provenance_file = metrics_v2.provenance.__file__
    original_metrics_file = metrics_v2.__file__
    try:
        copied_component_paths = {}
        for role, relative in original_component_paths.items():
            source = PROJECT_ROOT / relative
            if role == "evaluation_provenance_amendment":
                suffix = source.suffix or ".artifact"
                target = authority_root / "components" / f"{role}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                copied_component_paths[role] = target.relative_to(
                    PROJECT_ROOT
                ).as_posix()
            else:
                copied_component_paths[role] = relative
        canonical_v2.CRITICAL_COMPONENT_PATHS.clear()
        canonical_v2.CRITICAL_COMPONENT_PATHS.update(copied_component_paths)
        metrics_v2.provenance.CRITICAL_COMPONENT_PATHS.clear()
        metrics_v2.provenance.CRITICAL_COMPONENT_PATHS.update(
            copied_component_paths
        )

        review_root = tmp_path / "review"
        package_root = review_root / "review_package"
        inputs = synthetic_inputs(package_root)
        for pass_name, old_name in (("pass_a", "merge_a"), ("pass_b", "merge_b")):
            destination = review_root / "merged" / pass_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package_root / old_name, destination)
            inputs[f"scores_{pass_name[-1]}"] = destination / "completed_scores.jsonl"

        formal_launch = review_root / "formal_launch_receipt.json"
        _write_json(
            formal_launch,
            {
                "schema_version": 1,
                "workflow_version": "causal_role_erasure_7mechanism_formal_review_launch_v1",
                "status": "completed_two_independent_schema_valid_passes",
                "evaluation_code_registry_sha256": canonical_v2.REGISTRY_SHA256,
                "model": "gpt-5.6-luna",
                "scientific_zero_fallbacks": 0,
                "merged": {
                    "pass_a": _file_binding(
                        review_root / "merged/pass_a/merge_manifest.json",
                        path=str(review_root / "merged/pass_a/merge_manifest.json"),
                    ),
                    "pass_b": _file_binding(
                        review_root / "merged/pass_b/merge_manifest.json",
                        path=str(review_root / "merged/pass_b/merge_manifest.json"),
                    ),
                },
            },
        )

        initial_audit = review_root / "human_audit_stage0"
        legacy_canonical.build_audit_package(
            scores_a_path=inputs["scores_a"],
            scores_b_path=inputs["scores_b"],
            assignments_path=inputs["assignments"],
            audit_strata_key_path=inputs["audit_strata_key"],
            key_commitments_path=inputs["key_commitments"],
            output_root=initial_audit,
        )
        completed_initial = (
            tmp_path / "initial_round/review_process/completed_human.csv"
        )
        completed_initial.parent.mkdir(parents=True)
        complete_human_queue(
            initial_audit / "public/human_audit_queue.csv",
            initial_audit / "private/audit_key.jsonl",
            completed_initial,
        )
        audit = review_root / "human_audit_stage1"
        child = legacy_canonical.expand_audit_package(
            audit_root=initial_audit,
            completed_human_path=completed_initial,
            output_root=audit,
        )
        assert child["status"] == "no_expansion_required_human_audit_frozen"
        completed = tmp_path / "review_process/completed_human.csv"
        completed.parent.mkdir(parents=True)
        complete_human_queue(
            audit / "public/human_audit_queue.csv",
            audit / "private/audit_key.jsonl",
            completed,
        )

        amendment = _synthetic_amendment(
            PROJECT_ROOT
            / copied_component_paths["evaluation_provenance_amendment"],
            inputs,
            initial_audit,
            formal_launch,
        )
        reviewer_instructions = (
            PROJECT_ROOT / copied_component_paths["reviewer_instructions"]
        )
        component_files = [
            PROJECT_ROOT / relative for relative in copied_component_paths.values()
        ]
        implementation_commit, implementation_tree = _git_snapshot_commit(
            component_files,
            tmp_path / "synthetic-git-index",
        )
        pre_metric = _pre_metric_manifest(
            tmp_path,
            amendment,
            review_root,
            initial_audit,
            implementation_commit,
            implementation_tree,
        )
        initial_review_receipt = _review_process_receipt(
            tmp_path / "initial_round",
            review_root,
            initial_audit,
            completed_initial,
            reviewer_instructions,
            pre_metric,
        )
        review_receipt = _review_process_receipt(
            tmp_path,
            review_root,
            audit,
            completed,
            reviewer_instructions,
            pre_metric,
            initial_review_receipt,
        )

        legacy_canonical_root = tmp_path / "legacy_canonical"
        legacy_canonical.freeze_canonical_scores(
            audit_root=audit,
            completed_human_path=completed,
            scores_a_path=inputs["scores_a"],
            scores_b_path=inputs["scores_b"],
            output_root=legacy_canonical_root,
        )
        canonical_root = tmp_path / "canonical_v2"
        canonical_v2.freeze_canonical_scores_v2(
            project_root=PROJECT_ROOT,
            amendment_path=amendment,
            pre_metric_path=pre_metric,
            review_process_receipt_path=review_receipt,
            audit_root=audit,
            completed_human_path=completed,
            scores_a_path=inputs["scores_a"],
            scores_b_path=inputs["scores_b"],
            output_root=canonical_root,
        )

        legacy_eligibility_root = tmp_path / "legacy_eligibility"
        legacy_canonical.freeze_original_eligibility(
            canonical_root=legacy_canonical_root,
            original_key_path=inputs["original_key"],
            key_commitments_path=inputs["key_commitments"],
            output_root=legacy_eligibility_root,
        )
        eligibility_root = tmp_path / "eligibility_v2"
        canonical_v2.freeze_original_eligibility_v2(
            project_root=PROJECT_ROOT,
            amendment_path=amendment,
            pre_metric_path=pre_metric,
            canonical_root=canonical_root,
            original_key_path=inputs["original_key"],
            key_commitments_path=inputs["key_commitments"],
            output_root=eligibility_root,
        )

        legacy_metrics_root = tmp_path / "legacy_metrics"
        legacy_metrics.compute_formal_metrics(
            canonical_scores_path=legacy_canonical_root / "canonical_scores.jsonl",
            full_key_path=inputs["full_key"],
            key_commitments_path=inputs["key_commitments"],
            original_eligibility_path=legacy_eligibility_root
            / "original_eligibility.csv",
            shared_subsets_path=legacy_eligibility_root
            / "shared_capability_subsets.csv",
            formal_cases_path=FORMAL_CASES,
            output_root=legacy_metrics_root,
        )
        metrics_root = tmp_path / "metrics_v2"
        metrics_v2.compute_formal_metrics_v2(
            project_root=PROJECT_ROOT,
            amendment_path=amendment,
            pre_metric_path=pre_metric,
            canonical_root=canonical_root,
            eligibility_root=eligibility_root,
            full_key_path=inputs["full_key"],
            key_commitments_path=inputs["key_commitments"],
            formal_cases_path=FORMAL_CASES,
            output_root=metrics_root,
        )
        yield {
            "tmp": tmp_path,
            "review_root": review_root,
            "inputs": inputs,
            "initial_audit": initial_audit,
            "audit": audit,
            "completed": completed,
            "completed_initial": completed_initial,
            "amendment": amendment,
            "pre_metric": pre_metric,
            "review_receipt": review_receipt,
            "legacy_canonical": legacy_canonical_root,
            "canonical": canonical_root,
            "legacy_eligibility": legacy_eligibility_root,
            "eligibility": eligibility_root,
            "legacy_metrics": legacy_metrics_root,
            "metrics": metrics_root,
            "components": copied_component_paths,
            "implementation_commit": implementation_commit,
            "implementation_tree": implementation_tree,
        }
    finally:
        canonical_v2.__file__ = original_canonical_file
        metrics_v2.__file__ = original_metrics_file
        metrics_v2.provenance.__file__ = original_metrics_provenance_file
        canonical_v2.CRITICAL_COMPONENT_PATHS.clear()
        canonical_v2.CRITICAL_COMPONENT_PATHS.update(original_component_paths)
        metrics_v2.provenance.CRITICAL_COMPONENT_PATHS.clear()
        metrics_v2.provenance.CRITICAL_COMPONENT_PATHS.update(
            original_metrics_provenance_paths
        )
        shutil.rmtree(authority_root, ignore_errors=True)


def test_static_amendment_and_five_file_registry_are_self_consistent():
    value = canonical_v2.validate_amendment(AMENDMENT, PROJECT_ROOT)
    assert value["deviation"] == canonical_v2.AMENDMENT_DEVIATION
    assert value["bindings"]["evaluation_code_registry"]["registry_sha256"] == canonical_v2.REGISTRY_SHA256
    for expected in canonical_v2.V1_FILES:
        path = PROJECT_ROOT / expected["path"]
        assert path.stat().st_size == expected["size_bytes"]
        assert canonical_v2.sha256_file(path) == expected["sha256"]

    # A same-name preloaded module with a forged on-disk origin must not become
    # the metrics wrapper's provenance implementation.
    module_name = "canonicalize_causal_role_erasure_7mechanism_formal_v2"
    previous = sys.modules.get(module_name)
    fake = types.ModuleType(module_name)
    fake.__file__ = str(
        PROJECT_ROOT
        / "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py"
    )
    fake.ambient_attack_marker = True
    sys.modules[module_name] = fake
    try:
        probe_name = "_evaluation_v2_metrics_preload_probe"
        spec = importlib.util.spec_from_file_location(
            probe_name,
            PROJECT_ROOT
            / "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py",
        )
        assert spec is not None and spec.loader is not None
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        assert probe.provenance is not fake
        assert not hasattr(probe.provenance, "ambient_attack_marker")
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_v2_payloads_are_byte_identical_and_only_manifests_change(
    synthetic_v2_pipeline,
    monkeypatch,
):
    item = synthetic_v2_pipeline
    for name in ("canonical_scores.jsonl", "audit_diagnostics.json"):
        assert (item["canonical"] / name).read_bytes() == (
            item["legacy_canonical"] / name
        ).read_bytes()
    for name in ("original_eligibility.csv", "shared_capability_subsets.csv"):
        assert (item["eligibility"] / name).read_bytes() == (
            item["legacy_eligibility"] / name
        ).read_bytes()
    legacy_metric_files = {
        path.name
        for path in item["legacy_metrics"].iterdir()
        if path.is_file() and path.name != "score_manifest.json"
    }
    assert len(legacy_metric_files) == 21
    for name in legacy_metric_files:
        assert (item["metrics"] / name).read_bytes() == (
            item["legacy_metrics"] / name
        ).read_bytes()
    assert json.loads((item["metrics"] / "results.json").read_text())["protocol"] == legacy_metrics.PROTOCOL

    # Even a swap-and-restore mutation after validation cannot affect v1: it
    # receives only the already-captured private snapshot.
    original_run = canonical_v2._run_registered_stage_isolated
    original_snapshot = canonical_v2._snapshot_input
    original_capture = canonical_v2._capture_validated_json
    original_logical_ref = canonical_v2.logical_ref
    completed_path = item["completed"]
    completed_bytes = completed_path.read_bytes()
    manifest_paths = {
        (item["audit"] / "audit_manifest.json").resolve(): (
            item["audit"] / "audit_manifest.json"
        ).read_bytes(),
        (
            item["inputs"]["scores_a"].parent / "merge_manifest.json"
        ).resolve(): (
            item["inputs"]["scores_a"].parent / "merge_manifest.json"
        ).read_bytes(),
        (
            item["inputs"]["scores_b"].parent / "merge_manifest.json"
        ).resolve(): (
            item["inputs"]["scores_b"].parent / "merge_manifest.json"
        ).read_bytes(),
    }
    canonical_authority_paths = {
        item["amendment"].resolve(): item["amendment"].read_bytes(),
        item["pre_metric"].resolve(): item["pre_metric"].read_bytes(),
        item["review_receipt"].resolve(): item["review_receipt"].read_bytes(),
        (
            item["review_root"]
            / "review_package/public/evaluation_code_registry.json"
        ).resolve(): (
            item["review_root"]
            / "review_package/public/evaluation_code_registry.json"
        ).read_bytes(),
    }
    canonical_authority_swaps = set()

    def snapshot_then_swap(source, target, seals, label, **kwargs):
        result = original_snapshot(source, target, seals, label, **kwargs)
        resolved = source.resolve()
        if resolved in manifest_paths:
            source.write_bytes(b'{"attacker_selected_manifest":true}\n')
        return result

    def capture_then_swap_canonical(path, seals, label, validated, **kwargs):
        seal = original_capture(path, seals, label, validated, **kwargs)
        resolved = path.resolve()
        if resolved in canonical_authority_paths:
            path.write_bytes(b'{"attacker_selected_authority":true}\n')
            canonical_authority_swaps.add(resolved)
        return seal

    def reject_canonical_live_authority_ref(root_id, root, path):
        if (
            canonical_authority_swaps == set(canonical_authority_paths)
            and path.resolve() in canonical_authority_paths
        ):
            raise AssertionError("canonical stage reopened a sealed authority path")
        return original_logical_ref(root_id, root, path)

    def mutate_then_restore(**kwargs):
        for path, payload in manifest_paths.items():
            path.write_bytes(payload)
        for path, payload in canonical_authority_paths.items():
            path.write_bytes(payload)
        completed_path.write_bytes(b"transient malicious completed-human input\n")
        try:
            return original_run(**kwargs)
        finally:
            completed_path.write_bytes(completed_bytes)
            for path, payload in manifest_paths.items():
                path.write_bytes(payload)
            for path, payload in canonical_authority_paths.items():
                path.write_bytes(payload)

    monkeypatch.setattr(canonical_v2, "_snapshot_input", snapshot_then_swap)
    monkeypatch.setattr(
        canonical_v2, "_capture_validated_json", capture_then_swap_canonical
    )
    monkeypatch.setattr(
        canonical_v2, "logical_ref", reject_canonical_live_authority_ref
    )
    monkeypatch.setattr(
        canonical_v2, "_run_registered_stage_isolated", mutate_then_restore
    )
    mutation_output = item["tmp"] / "snapshot_mutation_canonical"
    canonical_v2.freeze_canonical_scores_v2(
        project_root=PROJECT_ROOT,
        amendment_path=item["amendment"],
        pre_metric_path=item["pre_metric"],
        review_process_receipt_path=item["review_receipt"],
        audit_root=item["audit"],
        completed_human_path=completed_path,
        scores_a_path=item["inputs"]["scores_a"],
        scores_b_path=item["inputs"]["scores_b"],
        output_root=mutation_output,
    )
    for name in ("canonical_scores.jsonl", "audit_diagnostics.json"):
        assert (mutation_output / name).read_bytes() == (
            item["legacy_canonical"] / name
        ).read_bytes()
    assert canonical_authority_swaps == set(canonical_authority_paths)

    monkeypatch.setattr(canonical_v2, "_snapshot_input", original_snapshot)
    monkeypatch.setattr(canonical_v2, "_capture_validated_json", original_capture)
    monkeypatch.setattr(canonical_v2, "logical_ref", original_logical_ref)

    def mutate_without_restore(**kwargs):
        completed_path.write_bytes(b"persistent malicious completed-human input\n")
        return original_run(**kwargs)

    monkeypatch.setattr(
        canonical_v2, "_run_registered_stage_isolated", mutate_without_restore
    )
    rejected_output = item["tmp"] / "persistent_mutation_canonical"
    try:
        with pytest.raises(
            canonical_v2.ProvenanceV2Error, match="pre-publish input"
        ):
            canonical_v2.freeze_canonical_scores_v2(
                project_root=PROJECT_ROOT,
                amendment_path=item["amendment"],
                pre_metric_path=item["pre_metric"],
                review_process_receipt_path=item["review_receipt"],
                audit_root=item["audit"],
                completed_human_path=completed_path,
                scores_a_path=item["inputs"]["scores_a"],
                scores_b_path=item["inputs"]["scores_b"],
                output_root=rejected_output,
            )
    finally:
        completed_path.write_bytes(completed_bytes)
    assert not rejected_output.exists()

    # Exercise the same authority swap against the metrics stage.  Every
    # validated JSON is replaced immediately after capture; the isolated run
    # restores the originals only after all captures are complete.  A trap on
    # logical_ref proves common authority fields are never reopened live.
    metric_provenance = metrics_v2.provenance
    metric_capture = metric_provenance._capture_validated_json
    metric_run = metric_provenance._run_registered_stage_isolated
    metric_logical_ref = metric_provenance.logical_ref
    registry_path = (
        item["review_root"]
        / "review_package/public/evaluation_code_registry.json"
    )
    metric_authority_paths = {
        item["amendment"].resolve(): item["amendment"].read_bytes(),
        item["pre_metric"].resolve(): item["pre_metric"].read_bytes(),
        (item["canonical"] / "canonical_manifest.json").resolve(): (
            item["canonical"] / "canonical_manifest.json"
        ).read_bytes(),
        (item["eligibility"] / "eligibility_manifest.json").resolve(): (
            item["eligibility"] / "eligibility_manifest.json"
        ).read_bytes(),
        registry_path.resolve(): registry_path.read_bytes(),
    }
    metric_swaps = set()

    def capture_then_swap_metric(path, seals, label, validated, **kwargs):
        seal = metric_capture(path, seals, label, validated, **kwargs)
        resolved = path.resolve()
        if resolved in metric_authority_paths:
            path.write_bytes(b'{"attacker_selected_authority":true}\n')
            metric_swaps.add(resolved)
        return seal

    def restore_then_run_metric(**kwargs):
        for path, payload in metric_authority_paths.items():
            path.write_bytes(payload)
        return metric_run(**kwargs)

    def reject_live_authority_ref(root_id, root, path):
        if path.resolve() in metric_authority_paths:
            raise AssertionError("metrics reopened a sealed authority path")
        return metric_logical_ref(root_id, root, path)

    monkeypatch.setattr(
        metric_provenance,
        "_capture_validated_json",
        capture_then_swap_metric,
    )
    monkeypatch.setattr(
        metric_provenance,
        "_run_registered_stage_isolated",
        restore_then_run_metric,
    )
    monkeypatch.setattr(
        metric_provenance, "logical_ref", reject_live_authority_ref
    )
    metric_swap_output = item["tmp"] / "authority_swap_metrics"
    try:
        metrics_v2.compute_formal_metrics_v2(
            project_root=PROJECT_ROOT,
            amendment_path=item["amendment"],
            pre_metric_path=item["pre_metric"],
            canonical_root=item["canonical"],
            eligibility_root=item["eligibility"],
            full_key_path=item["inputs"]["full_key"],
            key_commitments_path=item["inputs"]["key_commitments"],
            formal_cases_path=FORMAL_CASES,
            output_root=metric_swap_output,
        )
    finally:
        for path, payload in metric_authority_paths.items():
            path.write_bytes(payload)
    assert metric_swaps == set(metric_authority_paths)
    for path in item["metrics"].iterdir():
        if path.is_file() and path.name != "score_manifest.json":
            assert (metric_swap_output / path.name).read_bytes() == path.read_bytes()


def test_expansion_required_path_projects_added_rows_and_freezes_final_receipt(
    synthetic_v2_pipeline,
):
    item = synthetic_v2_pipeline
    case_root = item["tmp"] / "expansion_required_case"
    initial_atoms = legacy_canonical.load_jsonl(
        item["initial_audit"] / "private/audit_key.jsonl", "initial audit key"
    )
    first = next(
        atom
        for atom in initial_atoms
        if legacy_canonical._is_high_conf_agreement(atom)
    )
    forced = {
        atom["audit_id"]
        for atom in initial_atoms
        if atom["mechanism"] == first["mechanism"]
        and atom["field"] == first["field"]
        and legacy_canonical._is_high_conf_agreement(atom)
    }
    assert forced
    triggering_completion = (
        case_root / "initial_round/review_process/completed_human.csv"
    )
    triggering_completion.parent.mkdir(parents=True)
    complete_human_queue(
        item["initial_audit"] / "public/human_audit_queue.csv",
        item["initial_audit"] / "private/audit_key.jsonl",
        triggering_completion,
        force_error_audit_ids=forced,
    )
    expanded = item["review_root"] / "human_audit_expansion_required_test"
    expansion_manifest = legacy_canonical.expand_audit_package(
        audit_root=item["initial_audit"],
        completed_human_path=triggering_completion,
        output_root=expanded,
    )
    assert expansion_manifest["status"] == "expanded_human_audit_frozen"
    assert expansion_manifest["counts"]["expanded_strata"] == 1
    assert expansion_manifest["counts"]["added_audit_atoms"] > 0

    final_completed = case_root / "review_process/completed_human.csv"
    _complete_new_expansion_atoms(
        expanded / "public/human_audit_queue.csv",
        expanded / "private/audit_key.jsonl",
        final_completed,
    )
    reviewer_instructions = (
        PROJECT_ROOT / item["components"]["reviewer_instructions"]
    )
    triggering_initial_receipt = _review_process_receipt(
        case_root / "initial_round",
        item["review_root"],
        item["initial_audit"],
        triggering_completion,
        reviewer_instructions,
        item["pre_metric"],
    )
    process_receipt = _review_process_receipt(
        case_root,
        item["review_root"],
        expanded,
        final_completed,
        reviewer_instructions,
        item["pre_metric"],
        triggering_initial_receipt,
    )
    process_root = case_root / "review_process"
    for reviewer_id in ("reviewer_1", "reviewer_2"):
        delivery_receipt = json.loads(
            (process_root / f"{reviewer_id}_delivery.json").read_text()
        )
        assert (
            delivery_receipt["projection_mode"]
            == "reviewer_specific_expanded_queue_peer_and_adjudicator_columns_blank"
        )
        with (case_root / f"{reviewer_id}_delivery/human_audit_queue.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            delivered = list(csv.DictReader(handle))
        own = {f"{reviewer_id}_score", f"{reviewer_id}_notes"}
        added = [
            row
            for row in delivered
            if row["audit_id"] not in {atom["audit_id"] for atom in initial_atoms}
        ]
        assert added
        assert all(
            row[field] == ""
            for row in added
            for field in review_process.SCORE_FIELDS
        )
        inherited = next(row for row in delivered if row["audit_id"] in forced)
        assert all(
            inherited[field] == ""
            for field in review_process.SCORE_FIELDS
            if field not in own
        )
        assert inherited[f"{reviewer_id}_score"] in ("0", "1", "2")

    output = case_root / "canonical_v2"
    manifest = canonical_v2.freeze_canonical_scores_v2(
        project_root=PROJECT_ROOT,
        amendment_path=item["amendment"],
        pre_metric_path=item["pre_metric"],
        review_process_receipt_path=process_receipt,
        audit_root=expanded,
        completed_human_path=final_completed,
        scores_a_path=item["inputs"]["scores_a"],
        scores_b_path=item["inputs"]["scores_b"],
        output_root=output,
    )
    assert manifest["status"] == canonical_v2.CANONICAL_STATUS
    assert manifest["quality"]["expanded_strata"] == 1


def test_published_v2_manifests_are_truthful_relative_and_fully_chained(
    synthetic_v2_pipeline,
):
    item = synthetic_v2_pipeline
    canonical_manifest = json.loads(
        (item["canonical"] / "canonical_manifest.json").read_text()
    )
    eligibility_manifest = json.loads(
        (item["eligibility"] / "eligibility_manifest.json").read_text()
    )
    score_manifest = json.loads(
        (item["metrics"] / "score_manifest.json").read_text()
    )
    assert canonical_manifest["status"] == canonical_v2.CANONICAL_STATUS
    assert eligibility_manifest["status"] == canonical_v2.ELIGIBILITY_STATUS
    assert score_manifest["status"] == metrics_v2.STATUS
    assert score_manifest["inputs"]["canonical_manifest"]["sha256"] == canonical_v2.sha256_file(
        item["canonical"] / "canonical_manifest.json"
    )
    assert score_manifest["inputs"]["eligibility_manifest"]["sha256"] == canonical_v2.sha256_file(
        item["eligibility"] / "eligibility_manifest.json"
    )
    assert score_manifest["tables"]["identification"]["csv"]["path"] == "identification_contrasts.csv"
    assert set(
        json.loads(item["pre_metric"].read_text())["reproducibility_bindings"]
    ) == canonical_v2.REPRODUCIBILITY_BINDING_NAMES
    for manifest in (canonical_manifest, eligibility_manifest, score_manifest):
        serialized = canonical_v2.canonical_json_bytes(manifest).decode()
        assert not any(
            forbidden in "".join(serialized.split())
            for forbidden in canonical_v2.FORBIDDEN_PUBLISHED_TEXT
        )
        for value in _walk_strings(manifest):
            assert not value.startswith("/")


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_tampered_review_receipt_fails_before_output(
    synthetic_v2_pipeline,
):
    item = synthetic_v2_pipeline
    receipt = json.loads(item["review_receipt"].read_text())
    receipt["assertions"]["global_answer_key_blindness_claimed"] = True
    receipt = _seal(receipt, "receipt_sha256")
    tampered = item["tmp"] / "review_process/tampered_receipt.json"
    _write_json(tampered, receipt)
    output = item["tmp"] / "tampered_canonical"
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="assertions"):
        canonical_v2.freeze_canonical_scores_v2(
            project_root=PROJECT_ROOT,
            amendment_path=item["amendment"],
            pre_metric_path=item["pre_metric"],
            review_process_receipt_path=tampered,
            audit_root=item["audit"],
            completed_human_path=item["completed"],
            scores_a_path=item["inputs"]["scores_a"],
            scores_b_path=item["inputs"]["scores_b"],
            output_root=output,
        )
    assert not output.exists()


def test_audit_ancestry_rejects_direct_initial_and_repeated_decision(
    synthetic_v2_pipeline,
):
    item = synthetic_v2_pipeline
    amendment = json.loads(item["amendment"].read_text())
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="exactly one"):
        canonical_v2.validate_audit_ancestry(
            audit_root=item["initial_audit"],
            amendment=amendment,
            legacy=legacy_canonical,
        )
    repeated = item["review_root"] / "human_audit_repeated_no_expansion_test"
    legacy_canonical.expand_audit_package(
        audit_root=item["audit"],
        completed_human_path=item["completed"],
        output_root=repeated,
    )
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="exactly one"):
        canonical_v2.validate_audit_ancestry(
            audit_root=repeated,
            amendment=amendment,
            legacy=legacy_canonical,
        )


def test_premetric_authority_rejects_fake_git_id_and_dirty_critical_blob(
    synthetic_v2_pipeline,
    monkeypatch,
):
    item = synthetic_v2_pipeline
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "cat-file",
            "-t",
            item["implementation_commit"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "cat-file",
            "-t",
            item["implementation_tree"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert commit.stdout.strip() == "commit"
    assert tree.stdout.strip() == "tree"

    premetric_value = json.loads(item["pre_metric"].read_text())
    receipt_ref = premetric_value["reproducibility_bindings"][
        "evaluation_environment_receipt"
    ]
    receipt_path = item["pre_metric"].parent / receipt_ref["path"]
    receipt_bytes = receipt_path.read_bytes()
    original_seal_input = canonical_v2._seal_input
    swapped_receipt = False

    def seal_then_swap_receipt(path, label, **kwargs):
        nonlocal swapped_receipt
        payload, seal = original_seal_input(path, label, **kwargs)
        if path.resolve() == receipt_path.resolve() and not swapped_receipt:
            receipt_path.write_bytes(b'{"attacker_selected_receipt":true}\n')
            swapped_receipt = True
        return payload, seal

    monkeypatch.setattr(canonical_v2, "_seal_input", seal_then_swap_receipt)
    receipt_seals = {}
    try:
        captured_receipt = canonical_v2._environment_receipt_for_stage(
            pre_metric_path=item["pre_metric"],
            pre_metric=premetric_value,
            seals=receipt_seals,
        )
        assert captured_receipt["protocol"] == reproducibility.ENVIRONMENT_PROTOCOL
        assert json.loads(receipt_path.read_text()) == {
            "attacker_selected_receipt": True
        }
    finally:
        receipt_path.write_bytes(receipt_bytes)
        monkeypatch.setattr(canonical_v2, "_seal_input", original_seal_input)

    manifest = json.loads(item["pre_metric"].read_text())
    fake = copy.deepcopy(manifest)
    fake["implementation_commit"]["commit"] = "0" * 40
    fake = _seal(fake, "manifest_sha256")
    fake_path = item["tmp"] / "fake_git_authority.json"
    _write_json(fake_path, fake)
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="git rev-parse"):
        canonical_v2.validate_pre_metric_freeze(
            fake_path,
            item["amendment"],
            json.loads(item["amendment"].read_text()),
            PROJECT_ROOT,
        )

    component = PROJECT_ROOT / item["components"]["final_table_exporter"]
    original = component.read_bytes()
    try:
        component.write_bytes(original + b"\n# dirty-after-freeze\n")
        with pytest.raises(
            canonical_v2.ProvenanceV2Error, match="live component differs"
        ):
            canonical_v2.validate_execution_authority(
                item["amendment"], item["pre_metric"], PROJECT_ROOT
            )
    finally:
        component.write_bytes(original)

    class ForgedLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.__file__ = str(
                PROJECT_ROOT / canonical_v2.LOCAL_IMPORT_SOURCES[module.__name__]
            )
            module.forged_by_meta_path = True

    class ForgedFinder:
        def __init__(self):
            self.hits = []

        def find_spec(self, fullname, path=None, target=None):
            if fullname in canonical_v2.LOCAL_IMPORT_SOURCES:
                self.hits.append(fullname)
                origin = str(
                    PROJECT_ROOT / canonical_v2.LOCAL_IMPORT_SOURCES[fullname]
                )
                return importlib.util.spec_from_loader(
                    fullname, ForgedLoader(), origin=origin
                )
            return None

    finder = ForgedFinder()
    sys.meta_path.insert(0, finder)
    try:
        loaded = canonical_v2.load_registered_canonicalizer(
            PROJECT_ROOT, json.loads(item["pre_metric"].read_text())
        )
        assert not hasattr(loaded.evaluation_code, "forged_by_meta_path")
        assert finder.hits == []
    finally:
        sys.meta_path.remove(finder)

    real_av = sys.modules.get("av")
    fake_av = types.ModuleType("av")
    fake_av.__file__ = str(
        Path(importlib.util.find_spec("av").origin).resolve()
    )
    fake_av.__version__ = reproducibility.parse_lock(
        PROJECT_ROOT
        / canonical_v2.CRITICAL_COMPONENT_PATHS["evaluation_environment_lock"]
    )["av"]
    sys.modules["av"] = fake_av
    try:
        with pytest.raises(
            canonical_v2.ProvenanceV2Error, match="preloaded PyAV"
        ):
            canonical_v2.load_registered_canonicalizer(
                PROJECT_ROOT, premetric_value
            )
    finally:
        if real_av is None:
            sys.modules.pop("av", None)
        else:
            sys.modules["av"] = real_av

    real_numpy = sys.modules.get("numpy")
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.__file__ = str(
        Path(importlib.util.find_spec("numpy").origin).resolve()
    )
    fake_numpy.__version__ = reproducibility.parse_lock(
        PROJECT_ROOT
        / canonical_v2.CRITICAL_COMPONENT_PATHS["evaluation_environment_lock"]
    )["numpy"]
    fake_numpy.ambient_attack_marker = True
    sys.modules["numpy"] = fake_numpy
    try:
        with pytest.raises(
            canonical_v2.ProvenanceV2Error, match="preloaded numpy"
        ):
            canonical_v2.load_registered_metrics(
                PROJECT_ROOT, json.loads(item["pre_metric"].read_text())
            )
    finally:
        if real_numpy is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = real_numpy

    shadow = PROJECT_ROOT / "numpy.py"
    assert not shadow.exists()
    try:
        shadow.write_text("raise RuntimeError('shadow import executed')\n")
        with pytest.raises(canonical_v2.ProvenanceV2Error, match="shadow-capable"):
            canonical_v2.validate_execution_authority(
                item["amendment"], item["pre_metric"], PROJECT_ROOT
            )
    finally:
        shadow.unlink(missing_ok=True)


def test_strict_nested_manifest_and_named_source_inventory_are_fail_closed(
    synthetic_v2_pipeline,
):
    item = synthetic_v2_pipeline
    premetric = json.loads(item["pre_metric"].read_text())
    del premetric["source_bindings"]["formal_launch_receipt"]
    premetric = _seal(premetric, "manifest_sha256")
    bad_premetric = item["tmp"] / "missing_named_source.json"
    _write_json(bad_premetric, premetric)
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="named source"):
        canonical_v2.validate_pre_metric_freeze(
            bad_premetric,
            item["amendment"],
            json.loads(item["amendment"].read_text()),
            PROJECT_ROOT,
        )

    copied = item["tmp"] / "tampered_nested_canonical"
    shutil.copytree(item["canonical"], copied)
    manifest_path = copied / "canonical_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["completed_human_audit"]["path"] = (
        "reviewer_1_completed.csv"
    )
    manifest = _seal(manifest, "manifest_sha256")
    manifest_path.write_bytes(canonical_v2.canonical_json_bytes(manifest))
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="input roots/locations"):
        canonical_v2.load_canonical_v2(
            canonical_root=copied,
            project_root=PROJECT_ROOT,
            amendment_path=item["amendment"],
            pre_metric_path=item["pre_metric"],
            key_commitments_path=item["inputs"]["key_commitments"],
        )


def test_v2_scanner_rejects_legacy_false_provenance(tmp_path: Path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"status":"canonical_anonymous_scores_frozen_before_answer_key_opening"}',
        encoding="utf-8",
    )
    with pytest.raises(canonical_v2.ProvenanceV2Error, match="legacy false"):
        canonical_v2.scan_published_tree(root)
