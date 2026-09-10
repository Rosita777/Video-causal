from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import build_causal_role_erasure_7mechanism_reproducibility_v1 as repro
from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v2 as canonical_v2
from scripts import verify_causal_role_erasure_7mechanism_artifact_dag_v1 as verifier


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_environment_lock_is_exact_and_truthfully_scoped(tmp_path: Path):
    lock = tmp_path / "eval.lock"
    lock.write_text(
        "\n".join(f"{name}==1.0" for name in sorted(repro.LOCK_DISTRIBUTIONS)) + "\n",
        encoding="utf-8",
    )
    assert set(repro.parse_lock(lock)) == repro.LOCK_DISTRIBUTIONS
    lock.write_text(lock.read_text().replace("av==1.0", "av>=1.0"), encoding="utf-8")
    with pytest.raises(repro.ReproducibilityError, match="exact pin"):
        repro.parse_lock(lock)


def test_environment_receipt_requires_passed_exact_release_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = tmp_path / "project"
    project.mkdir()
    lock = project / "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        "\n".join(f"{name}==1.0" for name in sorted(repro.LOCK_DISTRIBUTIONS)) + "\n",
        encoding="utf-8",
    )
    test_refs = []
    for relative in repro.RELEASE_TEST_PATHS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        test_refs.append(repro.repo_ref(project, path))
    value = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": repro.ENVIRONMENT_PROTOCOL,
            "protocol_version": repro.PROTOCOL_VERSION,
            "status": repro.ENVIRONMENT_STATUS,
            "scope": {
                "release_time_reproduction_environment": True,
                "historical_execution_environment_claim": False,
                "hosted_model_bitwise_replay_claim": False,
                "frozen_completed_scores_are_downstream_authority": True,
            },
            "lock": repro.repo_ref(project, lock),
            "python": {
                "implementation": "CPython",
                "version": "3.12.14",
                "executable_sha256": "a" * 64,
            },
            "platform": {"system": "test", "release": "test", "machine": "test"},
            "distributions": dict(sorted(repro.parse_lock(lock).items())),
            "bootstrap_distributions": {"pip": "25.0"},
            "codec_versions": {
                "pyav_libraries": {"libavcodec": [1, 2, 3]},
                "opencv": "1.0",
                "pillow": "1.0",
                "libjpeg": "1.0",
            },
            "external_tools": {
                "latexmk": {
                    "command": "latexmk",
                    "executable_sha256": "b" * 64,
                    "version_output_sha256": "c" * 64,
                    "version_first_line": "Latexmk synthetic",
                }
            },
            "tests": {
                "execution_model": repro.RELEASE_TEST_EXECUTION_MODEL,
                "files": test_refs,
                "per_file": [
                    {
                        "path": relative,
                        "command": repro.release_test_command(
                            "${EVAL_PYTHON}", relative
                        ),
                        "status": "passed",
                        "return_code": 0,
                        "result": {
                            "passed": repro.EXPECTED_RELEASE_TEST_PASSES_BY_PATH[
                                relative
                            ],
                            "failed": 0,
                            "skipped": 0,
                        },
                        "summary": (
                            f"{repro.EXPECTED_RELEASE_TEST_PASSES_BY_PATH[relative]} "
                            "passed, 0 failed, 0 skipped"
                        ),
                    }
                    for relative in repro.RELEASE_TEST_PATHS
                ],
                "aggregate": {
                    "passed": repro.EXPECTED_RELEASE_TEST_PASSES,
                    "failed": 0,
                    "skipped": 0,
                },
                "status": "passed",
                "return_code": 0,
                "summary": (
                    f"{repro.EXPECTED_RELEASE_TEST_PASSES} passed, "
                    "0 failed, 0 skipped"
                ),
            },
        },
        "receipt_sha256",
    )
    receipt = tmp_path / "environment_receipt.json"
    receipt.write_bytes(repro.canonical_json_bytes(value))
    with pytest.raises(repro.ReproducibilityError, match="live"):
        repro.validate_environment_receipt(
            project_root=project, lock_path=lock, receipt_path=receipt
        )
    validated = repro.validate_environment_receipt(
        project_root=project, lock_path=lock, receipt_path=receipt, live=False
    )
    assert validated["tests"]["execution_model"] == (
        "one_fresh_python_pytest_subprocess_per_test_file"
    )
    assert validated["tests"]["aggregate"] == {
        "passed": 64,
        "failed": 0,
        "skipped": 0,
    }
    assert [row["path"] for row in validated["tests"]["per_file"]] == list(
        repro.RELEASE_TEST_PATHS
    )
    assert all(len(row["command"]) == 7 for row in validated["tests"]["per_file"])

    value["tests"]["status"] = "not_run"
    del value["receipt_sha256"]
    value = repro.with_self_digest(value, "receipt_sha256")
    receipt.write_bytes(repro.canonical_json_bytes(value))
    with pytest.raises(repro.ReproducibilityError, match="test evidence"):
        repro.validate_environment_receipt(
            project_root=project, lock_path=lock, receipt_path=receipt, live=False
        )

    observed_commands = []

    def fake_run(command, **_kwargs):
        observed_commands.append(command)
        relative = command[-1]
        count = repro.EXPECTED_RELEASE_TEST_PASSES_BY_PATH[relative]
        return subprocess.CompletedProcess(command, 0, f"{count} passed in 0.01s\n", "")

    monkeypatch.setattr(repro.subprocess, "run", fake_run)
    evidence = repro.run_release_test_files(
        project_root=project, test_relative_paths=repro.RELEASE_TEST_PATHS
    )
    assert observed_commands == [
        repro.release_test_command(repro.sys.executable, relative)
        for relative in repro.RELEASE_TEST_PATHS
    ]
    assert all(command[-1] in repro.RELEASE_TEST_PATHS for command in observed_commands)
    assert evidence["aggregate"] == {"passed": 64, "failed": 0, "skipped": 0}


def test_command_ledger_uses_v2_final_stages_and_keeps_only_v1_audit_expansion(
    tmp_path: Path,
):
    commands = repro.command_records()
    by_id = {row["stage_id"]: row for row in commands}
    assert len(by_id) == len(commands) == 37
    assert by_id["freeze_canonical_scores"]["argv"][1].endswith("formal_v2.py")
    assert by_id["freeze_original_eligibility"]["argv"][1].endswith("formal_v2.py")
    assert by_id["formal_metrics"]["argv"][1].endswith("formal_metrics_v2.py")
    assert by_id["audit_expansion"]["argv"][1].endswith("formal_v1.py")
    assert "expand-audit" in by_id["audit_expansion"]["argv"]
    assert by_id["release_evaluation_environment_receipt"]["argv"].index("freeze-environment") == 2
    assert "--environment-receipt" in by_id["pre_metric_code_freeze"]["argv"]
    assert [
        row["stage_id"] for row in commands
    ].index("release_evaluation_environment_receipt") < [
        row["stage_id"] for row in commands
    ].index("pre_metric_code_freeze")
    assert {
        "reviewer_1_initial_delivery",
        "reviewer_2_initial_delivery",
        "initial_human_review_process_receipt",
        "reviewer_1_final_delivery",
        "reviewer_2_final_delivery",
        "final_human_review_process_receipt",
        "paper_tables",
        "artifact_dag_final_materialize",
        "artifact_dag_final",
    } <= set(by_id)
    assert "${EXPANDED_AUDIT_ROOT}" in by_id["freeze_canonical_scores"]["argv"]
    assert by_id["paper_tables"]["argv"] == [
        "${EVAL_PYTHON}",
        "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
        "--repo-root",
        "${PROJECT_ROOT}",
        "--review-root",
        "${REVIEW_ROOT}",
        "--eligibility-root",
        "${ELIGIBILITY_ROOT}",
        "--score-manifest",
        "${METRICS_ROOT}/score_manifest.json",
        "--pre-metric-freeze",
        "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json",
        "--amendment",
        "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json",
        "--output-root",
        "${PROJECT_ROOT}/paper/tables/human_canonical_v1",
    ]
    assert [row["stage_id"] for row in commands].index(
        "artifact_dag_final_materialize"
    ) < [row["stage_id"] for row in commands].index("artifact_dag_final")
    assert "--build-final" in by_id["artifact_dag_final_materialize"]["argv"]
    for stage in (
        "reviewer_1_initial_delivery",
        "reviewer_2_initial_delivery",
        "initial_human_review_process_receipt",
        "reviewer_1_final_delivery",
        "reviewer_2_final_delivery",
        "final_human_review_process_receipt",
    ):
        assert by_id[stage]["argv"][
            by_id[stage]["argv"].index("--pre-metric-freeze") + 1
        ] == "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json"
    assert "--initial-process-receipt" in by_id[
        "final_human_review_process_receipt"
    ]["argv"]
    forbidden = []
    for row in commands:
        argv = row["argv"]
        if argv[1].endswith("canonicalize_causal_role_erasure_7mechanism_formal_v1.py"):
            forbidden.extend(value for value in argv if value in ("freeze", "freeze-eligibility"))
        assert not argv[1].endswith("compute_causal_role_erasure_7mechanism_formal_metrics_v1.py")
    assert forbidden == []
    project = Path(__file__).resolve().parents[1]
    stages = repro.manual_stage_records(project)
    assert [stage["stage_id"] for stage in stages] == list(verifier.MANUAL_STAGE_IDS)
    assert len(stages) == 12
    assert all(set(stage) == repro.MANUAL_STAGE_FIELDS for stage in stages)
    assert all(
        stage["completion_state_at_ledger_freeze"]
        == (
            "completed_before_ledger_freeze"
            if stage["phase"] == "target_selection"
            else "required_after_ledger_freeze"
        )
        for stage in stages
    )
    for stage in stages:
        for ref in stage["released_input_refs"].values():
            path = project / ref["path"]
            assert ref["root_id"] == "repo_root"
            assert ref["sha256"] == sha(path)
            assert ref["size_bytes"] == path.stat().st_size
    assert repro.EVAL_PYTHON_PLACEHOLDER == verifier.EVAL_PYTHON_PLACEHOLDER
    for root_id in sorted(verifier.RESERVED_EXTRA_ROOT_IDS):
        with pytest.raises(verifier.ArtifactDagError, match="reserved root ID"):
            verifier.parse_extra_roots([f"{root_id}={tmp_path}"])
    assert verifier.parse_extra_roots([f"final_release_root={tmp_path}"]) == {
        "final_release_root": tmp_path.resolve()
    }


def _wan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "project"
    data = project / "data/causal_role_erasure_7mechanism_main_v2"
    data.mkdir(parents=True)
    matrix = data / "run_matrix.csv"
    fieldnames = ["run_index", "run_id", "mechanism", "arm", "output_dir"]
    rows = []
    jobs = []
    mechanisms = ["water_impact", "rigid_collision", "brittle_fracture", "powder_impact", "elastic_deformation", "material_release", "surface_trace"]
    arms = ["matched_control"] * 7 + ["V4"] * 7 + ["generic_paraphrase"] * 2 + ["bystander_token"] * 2
    for index, arm in enumerate(arms):
        mechanism = mechanisms[index % 7] if index < 14 else ("water_impact" if index < 16 else "brittle_fracture")
        run_id = f"run_{index:02d}"
        output_dir = f"outputs/causal_role_erasure_7mechanism_main_v2/training/{run_id}"
        rows.append({"run_index": index, "run_id": run_id, "mechanism": mechanism, "arm": arm, "output_dir": output_dir})
        group = "main" if index < 14 else "identification"
        jobs.append(
            {
                "job_index": index,
                "job_id": run_id,
                "mechanism": mechanism,
                "arm": arm,
                "run_group": group,
                "row_count": 42 if group == "main" else 24,
                "receipt": f"/host/{output_dir}/run_receipt.json",
                "receipt_sha256": f"{index + 1:064x}",
                "run_spec": f"/host/outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2/run_specs/{index:02d}_{run_id}.json",
                "run_spec_sha256": f"{index + 101:064x}",
                "checkpoint": f"/host/{output_dir}/checkpoint-000200",
                "checkpoint_artifact_sha256": f"{index + 201:064x}",
                "weights_sha256": f"{index + 301:064x}",
                "training_state_sha256": f"{index + 401:064x}",
            }
        )
    with matrix.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    run_manifest = tmp_path / "eval_run_manifest.json"
    write_json(
        run_manifest,
        {
            "protocol_id": repro.PROTOCOL_VERSION,
            "expected_videos": 684,
            "jobs": jobs,
        },
    )
    aggregate = tmp_path / "eval_aggregate.json"
    write_json(
        aggregate,
        {
            "status": "completed",
            "expected_videos": 684,
            "validated_videos": 684,
            "status_counts": {"completed": 18},
            "run_manifest": {"sha256": sha(run_manifest)},
        },
    )
    monkeypatch.setattr(repro, "EXPECTED_RUN_MATRIX_SHA256", sha(matrix))
    return project, matrix, run_manifest, aggregate


def test_wan_inventory_derives_18_sanitized_refs_without_opening_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project, matrix, run_manifest, aggregate = _wan_fixture(tmp_path, monkeypatch)
    output = tmp_path / "inventory.json"
    result = repro.freeze_wan_receipt_inventory(
        project_root=project,
        run_matrix_path=matrix,
        eval_run_manifest_path=run_manifest,
        eval_aggregate_path=aggregate,
        output_path=output,
        expected_run_manifest_sha256=sha(run_manifest),
        expected_aggregate_sha256=sha(aggregate),
    )
    assert result["counts"] == {
        "receipts": 18,
        "main_runs": 14,
        "identification_runs": 4,
        "erase_updates_per_run": 100,
        "preserve_updates_per_run": 100,
        "eligible_checkpoint_step": 200,
        "validated_generation_rows": 684,
    }
    assert all(item["receipt"]["root_id"] == "a100_project_root" for item in result["items"])
    assert all(not Path(item["receipt"]["path"]).is_absolute() for item in result["items"])
    repro.validate_self_digest(result, "inventory_sha256", "inventory")


def _dag(nodes: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for node in nodes:
        status = str(node["status"])
        counts[status] = counts.get(status, 0) + 1
    return repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.DAG_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": "test",
            "nodes": nodes,
            "counts": dict(sorted(counts.items())),
        },
        "dag_sha256",
    )


def test_dag_rejects_hash_drift_cycles_and_pending_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "a.json"
    write_json(artifact, {"count": 1})
    ref = {"root_id": "repo_root", "path": "a.json", "sha256": sha(artifact), "size_bytes": artifact.stat().st_size}
    value = _dag(
        [
            {"id": "a", "status": "materialized", "parents": [], "artifact": ref, "semantic_checks": [{"pointer": "/count", "equals": 1}]},
            {"id": "future", "status": "pending", "parents": ["a"], "pending_reason": "not produced"},
        ]
    )
    assert verifier.verify_dag(value, {"repo_root": root}, "pre-metric")["pending"] == 1
    with pytest.raises(verifier.ArtifactDagError, match="final artifact DAG"):
        verifier.verify_dag(value, {"repo_root": root}, "final")
    artifact.write_text("drift\n", encoding="utf-8")
    with pytest.raises(verifier.ArtifactDagError, match="SHA mismatch"):
        verifier.verify_dag(value, {"repo_root": root}, "pre-metric")
    cycle = _dag(
        [
            {"id": "a", "status": "pending", "parents": ["b"], "pending_reason": "x"},
            {"id": "b", "status": "pending", "parents": ["a"], "pending_reason": "x"},
        ]
    )
    with pytest.raises(verifier.ArtifactDagError, match="cycle"):
        verifier.verify_dag(cycle, {}, "pre-metric")

    one_node = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.DAG_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": verifier.FINAL_DAG_STATUS,
            "nodes": [
                {
                    "id": "a",
                    "status": "materialized",
                    "parents": [],
                    "artifact": ref,
                    "semantic_checks": [],
                }
            ],
            "counts": {"materialized": 1},
        },
        "dag_sha256",
    )
    with pytest.raises(verifier.ArtifactDagError, match="mandatory"):
        verifier.verify_dag(one_node, {"repo_root": root}, "final")

    premetric = tmp_path / "premetric-final"
    premetric.mkdir()
    freeze_path = premetric / "pre_metric_code_freeze_v2.json"
    write_json(freeze_path, {"status": verifier.PRE_METRIC_STATUS})
    pending_nodes = [
        {
            "id": node_id,
            "status": "pending",
            "parents": [],
            "pending_reason": "synthetic pending branch",
        }
        for node_id in sorted(verifier.FINAL_PENDING_NODE_IDS)
    ]
    unavailable_nodes = []
    unavailable_sha256 = {}
    for index, node_id in enumerate(sorted(verifier.FINAL_UNAVAILABLE_NODE_IDS), 1):
        prefix = f"{index:012x}"
        unavailable_sha256[node_id] = prefix + "0" * 52
        unavailable_nodes.append(
            {
                "id": node_id,
                "status": "declared_unavailable",
                "parents": [],
                "expected_sha256_prefix": prefix,
                "access_conditions": "synthetic release condition",
                "reproduction_limit": "synthetic release limit",
            }
        )
    prefix_dag = _dag([*unavailable_nodes, *pending_nodes])
    prefix_path = premetric / "pre_metric_artifact_dag_v1.json"
    prefix_path.write_bytes(repro.canonical_json_bytes(prefix_dag))
    prefix_ref = {
        "root_id": "pre_metric_root",
        "path": prefix_path.name,
        "sha256": sha(prefix_path),
        "size_bytes": prefix_path.stat().st_size,
    }
    final_artifact = premetric / "final_authority.json"
    write_json(final_artifact, {"status": "synthetic"})
    final_ref = {
        "root_id": "pre_metric_root",
        "path": final_artifact.name,
        "sha256": sha(final_artifact),
        "size_bytes": final_artifact.stat().st_size,
    }
    final_refs = {
        node_id: dict(final_ref) for node_id in verifier.FINAL_PENDING_NODE_IDS
    }
    monkeypatch.setattr(
        verifier,
        "validate_final_authorities",
        lambda **_kwargs: final_refs,
    )
    pre_metric = {
        "implementation_commit": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "worktree_clean": True,
        },
        "reproducibility_bindings": {
            "pre_metric_artifact_dag": prefix_ref,
            "evaluation_environment_receipt": {},
        },
    }
    roots = {"pre_metric_root": premetric}
    final_1 = verifier.build_final_dag(
        prefix=prefix_dag,
        roots=roots,
        pre_metric=pre_metric,
        unavailable_sha256=unavailable_sha256,
    )
    final_2 = verifier.build_final_dag(
        prefix=prefix_dag,
        roots=roots,
        pre_metric=pre_metric,
        unavailable_sha256=unavailable_sha256,
    )
    assert final_1 == final_2
    assert final_1["status"] == verifier.FINAL_DAG_STATUS
    assert "pending" not in final_1["counts"]
    verifier.verify_final_prefix_binding(
        final=final_1,
        prefix=prefix_dag,
        pre_metric=pre_metric,
        roots=roots,
    )


def test_dag_rejects_preview_and_private_paths(tmp_path: Path):
    root = tmp_path / "root"
    path = root / "post_unblinding_preliminary_v2/a.json"
    path.parent.mkdir(parents=True)
    write_json(path, {})
    value = _dag(
        [
            {
                "id": "bad",
                "status": "materialized",
                "parents": [],
                "artifact": {"root_id": "repo_root", "path": "post_unblinding_preliminary_v2/a.json", "sha256": sha(path), "size_bytes": path.stat().st_size},
                "semantic_checks": [],
            }
        ]
    )
    with pytest.raises(verifier.ArtifactDagError, match="forbidden"):
        verifier.verify_dag(value, {"repo_root": root}, "pre-metric")


def test_pre_metric_manifest_contract_matches_v2_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = Path(__file__).resolve().parents[1]
    amendment_path = project / "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json"
    amendment = canonical_v2.validate_amendment(amendment_path, project)
    roles = {
        "evaluation_provenance_amendment": amendment_path,
        "canonicalization_wrapper": project / "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py",
        "formal_metrics_wrapper": project / "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py",
        "final_table_exporter": project / "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
        "reviewer_instructions": project / "docs/causal_role_erasure_7mechanism_human_review_v2.md",
        "reproducibility_builder": project / "scripts/build_causal_role_erasure_7mechanism_reproducibility_v1.py",
        "review_process_receipt_builder": project / "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py",
        "pre_metric_freeze_builder": project / "scripts/build_causal_role_erasure_7mechanism_pre_metric_freeze_v2.py",
        "artifact_dag_verifier": project / "scripts/verify_causal_role_erasure_7mechanism_artifact_dag_v1.py",
        "evaluation_environment_lock": project / "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock",
    }
    components = []
    for role, path in sorted(roles.items()):
        components.append({"role": role, "path": path.relative_to(project).as_posix(), "sha256": sha(path), "size_bytes": path.stat().st_size})
    premetric = tmp_path / "premetric"
    premetric.mkdir()
    bindings = {}
    for name, filename in (
        ("evaluation_environment_receipt", "evaluation_environment_receipt_v1.json"),
        ("wan_training_receipt_inventory", "wan_training_receipt_inventory_v1.json"),
        ("reproduction_command_ledger", "reproduction_command_ledger_v1.json"),
        ("pre_metric_artifact_dag", "pre_metric_artifact_dag_v1.json"),
    ):
        path = premetric / filename
        write_json(path, {"name": name})
        bindings[name] = {"root_id": "pre_metric_root", "path": path.name, "sha256": sha(path), "size_bytes": path.stat().st_size}
    lock = roles["evaluation_environment_lock"]
    bindings["evaluation_environment_lock"] = {"root_id": "repo_root", "path": lock.relative_to(project).as_posix(), "sha256": sha(lock), "size_bytes": lock.stat().st_size}
    source_bindings = {}
    for name, (root_id, logical_path) in canonical_v2.SOURCE_BINDING_PATHS.items():
        ref = amendment["bindings"][name]
        source_bindings[name] = {
            "root_id": root_id,
            "path": logical_path,
            "sha256": ref["sha256"],
            "size_bytes": ref["size_bytes"],
        }
    source_bindings["private_key_commitments"] = {
        name: dict(amendment["bindings"][name])
        for name in ("tier_0_audit_strata", "tier_1_original_only", "tier_2_full")
    }
    value = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": canonical_v2.PRE_METRIC_PROTOCOL,
            "protocol_version": canonical_v2.PROTOCOL_VERSION,
            "status": canonical_v2.PRE_METRIC_STATUS,
            "implementation_commit": {"commit": "a" * 40, "tree": "b" * 40, "worktree_clean": True},
            "deviation": {
                "global_full_key_opened_before_canonical_freeze": True,
                "human_review_process_local_answer_key_blindness_preserved": True,
                "v1_files_byte_identical": True,
                "scientific_rules_changed": False,
                "unchanged_v1_final_stages_forbidden": True,
            },
            "v1_authority": {
                "evaluation_code_registry": {"path": "review_package/public/evaluation_code_registry.json", "sha256": canonical_v2.REGISTRY_FILE_SHA256, "size_bytes": canonical_v2.REGISTRY_FILE_SIZE, "registry_sha256": canonical_v2.REGISTRY_SHA256},
                "files": [dict(row) for row in canonical_v2.V1_FILES],
            },
            "source_bindings": source_bindings,
            "components": components,
            "reproducibility_bindings": bindings,
            "pending_outputs": [{"name": name, "status": "not_materialized"} for name in canonical_v2.PENDING_OUTPUTS],
        },
        "manifest_sha256",
    )
    path = premetric / "pre_metric_code_freeze_v2.json"
    path.write_bytes(repro.canonical_json_bytes(value))
    monkeypatch.setattr(canonical_v2, "_validate_git_implementation", lambda **_kwargs: None)
    # This test targets the manifest schema.  Commit/blob binding and shadow-path
    # refusal are exercised separately with real synthetic Git authorities.
    monkeypatch.setattr(canonical_v2, "_reject_shadow_capable_files", lambda *_args: None)
    monkeypatch.setattr(canonical_v2, "_validate_pre_metric_local_artifacts", lambda **_kwargs: None)
    canonical_v2.validate_pre_metric_freeze(path, amendment_path, amendment, project)


def test_pre_metric_verifier_binds_every_component_to_recorded_git_blob(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    component_sources = list(verifier.REQUIRED_COMPONENT_PATHS.items()) + [
        (f"release_test_{index:02d}", relative)
        for index, relative in enumerate(verifier.RELEASE_TEST_PATHS)
    ]
    components: list[dict[str, object]] = []
    for role, relative in component_sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}\n", encoding="utf-8")
        components.append({"path": path.relative_to(repo).as_posix(), "sha256": sha(path), "size_bytes": path.stat().st_size, "role": role})
    components.sort(key=lambda row: str(row["role"]))
    dummy_entry_point = repo / "scripts/dummy_release_stage.py"
    dummy_entry_point.write_text("# synthetic release entry point\n", encoding="utf-8")
    for role, relative in verifier.TARGET_SELECTION_RELEASED_INPUT_PATHS.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}\n", encoding="utf-8")
    v1_files = []
    for index in range(5):
        path = repo / f"scripts/frozen_v1_{index}.py"
        path.write_text(f"# v1 {index}\n", encoding="utf-8")
        v1_files.append({"path": path.relative_to(repo).as_posix(), "sha256": sha(path), "size_bytes": path.stat().st_size})
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "freeze"], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
    premetric = tmp_path / "premetric"
    premetric.mkdir()
    lock = repo / verifier.REQUIRED_COMPONENT_PATHS["evaluation_environment_lock"]
    lock_ref = {"root_id": "repo_root", "path": lock.relative_to(repo).as_posix(), "sha256": sha(lock), "size_bytes": lock.stat().st_size}
    test_refs = [
        {
            "root_id": "repo_root",
            "path": relative,
            "sha256": sha(repo / relative),
            "size_bytes": (repo / relative).stat().st_size,
        }
        for relative in verifier.RELEASE_TEST_PATHS
    ]
    environment_receipt = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.ENVIRONMENT_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": verifier.ENVIRONMENT_STATUS,
            "scope": {},
            "lock": lock_ref,
            "python": {"executable_sha256": "a" * 64},
            "platform": {},
            "distributions": {},
            "bootstrap_distributions": {},
            "codec_versions": {},
            "tests": {"status": "passed", "return_code": 0, "files": test_refs},
        },
        "receipt_sha256",
    )
    environment_receipt_path = premetric / "evaluation_environment_receipt_v1.json"
    environment_receipt_path.write_bytes(repro.canonical_json_bytes(environment_receipt))
    inventory = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.INVENTORY_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": verifier.INVENTORY_STATUS,
            "verification_basis": {},
            "sources": {},
            "counts": {"receipts": 18, "validated_generation_rows": 684},
            "items": [
                {
                    "run_index": index,
                    "run_id": f"run_{index:02d}",
                    "mechanism": "test",
                    "arm": "test",
                    "run_group": "main" if index < 14 else "identification",
                    "step": 200,
                    "erase_updates": 100,
                    "preserve_updates": 100,
                    "validated_generation_rows": 42 if index < 14 else 24,
                    "run_spec": {"root_id": "a100_project_root", "path": f"run_specs/{index:02d}.json", "sha256": f"{index + 1:064x}"},
                    "receipt": {"root_id": "a100_project_root", "path": f"receipts/{index:02d}.json", "sha256": f"{index + 101:064x}"},
                    "checkpoint": {
                        "root_id": "a100_project_root",
                        "path": f"checkpoints/{index:02d}",
                        "artifact_sha256": f"{index + 201:064x}",
                        "weights_sha256": f"{index + 301:064x}",
                        "training_state_sha256": f"{index + 401:064x}",
                    },
                }
                for index in range(18)
            ],
        },
        "inventory_sha256",
    )
    inventory_path = premetric / "wan_training_receipt_inventory_v1.json"
    inventory_path.write_bytes(repro.canonical_json_bytes(inventory))
    required_stages = [
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
    ]
    commands = [
        {
            "stage_id": stage,
            "environment_id": "evaluation",
            "cwd": "${PROJECT_ROOT}",
            "argv": ["${EVAL_PYTHON}", "scripts/dummy_release_stage.py"],
            "environment": {},
            "private_inputs": [],
            "command_kind": "release_reproduction_command",
        }
        for stage in [*required_stages, *[f"synthetic_stage_{index:02d}" for index in range(27)]]
    ]
    ledger = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.LEDGER_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": verifier.LEDGER_STATUS,
            "scope": dict(verifier.LEDGER_SCOPE),
            "placeholders": {
                "EVAL_PYTHON": dict(verifier.EVAL_PYTHON_PLACEHOLDER),
                "PROJECT_ROOT": {"kind": "path_or_value", "must_be_supplied": True},
            },
            "commands": commands,
            "entry_points": {"scripts/dummy_release_stage.py": {"root_id": "repo_root", "path": "scripts/dummy_release_stage.py", "sha256": sha(dummy_entry_point), "size_bytes": dummy_entry_point.stat().st_size}},
            "counts": {"command_records": 37, "manual_stages": 12, "realized_expanded_generation_commands": 60},
            "manual_stages": repro.manual_stage_records(repo),
            "realized_command_evidence": {},
        },
        "ledger_sha256",
    )
    ledger_path = premetric / "reproduction_command_ledger_v1.json"
    ledger_path.write_bytes(repro.canonical_json_bytes(ledger))
    bindings = {
        "evaluation_environment_lock": lock_ref,
        "evaluation_environment_receipt": {"root_id": "pre_metric_root", "path": environment_receipt_path.name, "sha256": sha(environment_receipt_path), "size_bytes": environment_receipt_path.stat().st_size},
        "wan_training_receipt_inventory": {"root_id": "pre_metric_root", "path": inventory_path.name, "sha256": sha(inventory_path), "size_bytes": inventory_path.stat().st_size},
        "reproduction_command_ledger": {"root_id": "pre_metric_root", "path": ledger_path.name, "sha256": sha(ledger_path), "size_bytes": ledger_path.stat().st_size},
    }
    review = tmp_path / "review"
    registry_path = review / "review_package/public/evaluation_code_registry.json"
    write_json(registry_path, {"registry_sha256": "f" * 64, "files": v1_files})
    binding_file = premetric / "source.json"
    write_json(binding_file, {"source": True})
    binding_ref = {"root_id": "pre_metric_root", "path": binding_file.name, "sha256": sha(binding_file), "size_bytes": binding_file.stat().st_size}
    registry_source_ref = {"root_id": "review_root", "path": "review_package/public/evaluation_code_registry.json", "sha256": sha(registry_path), "size_bytes": registry_path.stat().st_size}
    source_bindings = {
        "evaluation_code_registry": registry_source_ref,
        "key_commitments": dict(binding_ref),
        "formal_launch_receipt": dict(binding_ref),
        "pass_a_merge_manifest": dict(binding_ref),
        "pass_b_merge_manifest": dict(binding_ref),
        "human_audit_manifest": dict(binding_ref),
        "human_audit_queue": dict(binding_ref),
        "formal_cases": dict(binding_ref),
        "identification_subset": dict(binding_ref),
        "private_key_commitments": {
            "tier_0_audit_strata": {"sha256": "1" * 64, "row_count": 1},
            "tier_1_original_only": {"sha256": "2" * 64, "row_count": 1},
            "tier_2_full": {"sha256": "3" * 64, "row_count": 1},
        },
    }
    dag_nodes = [
        {"id": "evaluation_environment_lock", "status": "materialized", "parents": [], "artifact": lock_ref, "semantic_checks": []},
        {"id": "release_evaluation_environment_receipt", "status": "materialized", "parents": ["evaluation_environment_lock"], "artifact": bindings["evaluation_environment_receipt"], "semantic_checks": []},
        {"id": "wan_receipt_inventory", "status": "materialized", "parents": [], "artifact": bindings["wan_training_receipt_inventory"], "semantic_checks": []},
        {"id": "executable_command_ledger", "status": "materialized", "parents": [], "artifact": bindings["reproduction_command_ledger"], "semantic_checks": []},
    ]
    for component in components:
        dag_nodes.append(
            {
                "id": f"component_{component['role']}",
                "status": "materialized",
                "parents": [],
                "artifact": {"root_id": "repo_root", "path": component["path"], "sha256": component["sha256"], "size_bytes": component["size_bytes"]},
                "semantic_checks": [],
            }
        )
    dag_nodes.extend(
        {"id": name, "status": "pending", "parents": [], "pending_reason": "not materialized"}
        for name in ("human_labels", "adjudication", "canonical_scores", "original_eligibility", "formal_metrics", "paper_tables")
    )
    dag = _dag(dag_nodes)
    dag_path = premetric / "pre_metric_artifact_dag_v1.json"
    dag_path.write_bytes(repro.canonical_json_bytes(dag))
    bindings["pre_metric_artifact_dag"] = {"root_id": "pre_metric_root", "path": dag_path.name, "sha256": sha(dag_path), "size_bytes": dag_path.stat().st_size}
    value = repro.with_self_digest(
        {
            "schema_version": 1,
            "protocol": verifier.PRE_METRIC_PROTOCOL,
            "protocol_version": verifier.PROTOCOL_VERSION,
            "status": verifier.PRE_METRIC_STATUS,
            "implementation_commit": {"commit": commit, "tree": tree, "worktree_clean": True},
            "deviation": {
                "global_full_key_opened_before_canonical_freeze": True,
                "human_review_process_local_answer_key_blindness_preserved": True,
                "v1_files_byte_identical": True,
                "scientific_rules_changed": False,
                "unchanged_v1_final_stages_forbidden": True,
            },
            "v1_authority": {
                "evaluation_code_registry": {"path": "review_package/public/evaluation_code_registry.json", "sha256": sha(registry_path), "size_bytes": registry_path.stat().st_size, "registry_sha256": "f" * 64},
                "files": v1_files,
            },
            "source_bindings": source_bindings,
            "components": components,
            "reproducibility_bindings": bindings,
            "pending_outputs": [{"name": name, "status": "not_materialized"} for name in ("human_labels", "adjudication", "canonical_scores", "original_eligibility", "formal_metrics", "paper_tables")],
        },
        "manifest_sha256",
    )
    roots = {"repo_root": repo, "pre_metric_root": premetric, "review_root": review}
    verifier.verify_pre_metric_freeze(value, roots)
    changed_component = next(
        component for component in components if component["role"] == "reviewer_instructions"
    )
    changed = repo / str(changed_component["path"])
    changed.write_text("post-commit drift\n", encoding="utf-8")
    changed_component["sha256"] = sha(changed)
    changed_component["size_bytes"] = changed.stat().st_size
    changed_node = next(
        node for node in dag["nodes"] if node["id"] == "component_reviewer_instructions"
    )
    changed_node["artifact"]["sha256"] = changed_component["sha256"]
    changed_node["artifact"]["size_bytes"] = changed_component["size_bytes"]
    del dag["dag_sha256"]
    dag = repro.with_self_digest(dag, "dag_sha256")
    dag_path.write_bytes(repro.canonical_json_bytes(dag))
    bindings["pre_metric_artifact_dag"]["sha256"] = sha(dag_path)
    bindings["pre_metric_artifact_dag"]["size_bytes"] = dag_path.stat().st_size
    del value["manifest_sha256"]
    value = repro.with_self_digest(value, "manifest_sha256")
    with pytest.raises(verifier.ArtifactDagError, match="recorded implementation commit"):
        verifier.verify_pre_metric_freeze(value, roots)
