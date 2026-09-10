#!/usr/bin/env python3
"""Run frozen v1 formal metrics under truthful v2 provenance.

Inputs must be authoritative schema-v2 canonical and eligibility roots.  They
are projected into a private temporary v1 compatibility tree, the byte-frozen
v1 metric implementation is executed there, and only its scientific payloads
are copied into the published v2 root.  The legacy manifest is never
published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

_EXPECTED_PROVENANCE_ORIGIN = Path(__file__).resolve().with_name(
    "canonicalize_causal_role_erasure_7mechanism_formal_v2.py"
)


def _load_private_provenance() -> tuple[types.ModuleType, str]:
    """Capture the sibling wrapper bytes without consulting ambient imports."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(_EXPECTED_PROVENANCE_ORIGIN, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ImportError("provenance wrapper changed while it was captured")
    source = b"".join(chunks)
    if len(source) != after.st_size:
        raise ImportError("provenance wrapper size changed while it was captured")
    module_name = "_causal7m_private_provenance_v2"
    module = types.ModuleType(module_name)
    module.__file__ = str(_EXPECTED_PROVENANCE_ORIGIN)
    module.__package__ = ""
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(
            compile(
                source.decode("utf-8"),
                str(_EXPECTED_PROVENANCE_ORIGIN),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module, hashlib.sha256(source).hexdigest()


provenance, _CAPTURED_PROVENANCE_SHA256 = _load_private_provenance()


PROTOCOL = "causal_role_erasure_7m_formal_metrics_v2"
STATUS = "formal_metrics_frozen_after_disclosed_global_full_key_access"
SCORE_MANIFEST_FIELDS = {
    "schema_version",
    "protocol",
    "status",
    "provenance_amendment",
    "pre_metric_code_freeze",
    "evaluation_code_registry",
    "wrapper_implementation",
    "registered_v1_derivation",
    "key_access",
    "inputs",
    "results",
    "tables",
    "manifest_sha256",
}
TABLE_NAMES = (
    "per_case_metrics",
    "mechanism_metrics",
    "macro_metrics",
    "headline_contrasts",
    "headline_mechanism_contrasts",
    "noninferiority",
    "identification",
    "role_conditioned_evidence",
    "m6_pairs",
    "secondary_metrics",
)
TABLE_BASENAMES = {
    **{name: name for name in TABLE_NAMES},
    "identification": "identification_contrasts",
}


def _require_private_provenance_authority(pre_metric: Mapping[str, Any]) -> None:
    components = provenance._component_by_role(pre_metric["components"])
    expected = components["canonicalization_wrapper"]
    provenance.require(
        _CAPTURED_PROVENANCE_SHA256 == expected["sha256"],
        "formal-metrics v2 provenance code differs from the pre-metric component",
    )
SCORE_INPUTS = {
    "canonical_manifest": ("canonical_root", "canonical_manifest.json"),
    "canonical_scores": ("canonical_root", "canonical_scores.jsonl"),
    "eligibility_manifest": ("eligibility_root", "eligibility_manifest.json"),
    "original_eligibility": ("eligibility_root", "original_eligibility.csv"),
    "shared_capability_subsets": (
        "eligibility_root",
        "shared_capability_subsets.csv",
    ),
    "full_method_key": ("review_root", "review_package/private/full_key.jsonl"),
    "key_commitments": (
        "review_root",
        "review_package/public/key_commitments.json",
    ),
    "formal_cases": (
        "repo_root",
        "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv",
    ),
}


def _copy_payloads(
    legacy_root: Path, staging: Path, legacy_manifest: Mapping[str, Any]
) -> tuple[Path, dict[str, dict[str, dict[str, Any]]], dict[str, Path]]:
    expected_names = {"results.json", "score_manifest.json"}
    payloads: dict[str, Path] = {}
    table_refs: dict[str, dict[str, dict[str, Any]]] = {}
    results_ref = legacy_manifest.get("results")
    provenance.require(
        isinstance(results_ref, dict) and results_ref.get("path") == "results.json",
        "legacy metric results reference changed",
    )
    legacy_results = legacy_root / "results.json"
    provenance.require(
        provenance.sha256_file(legacy_results) == results_ref.get("sha256"),
        "legacy metric results SHA changed",
    )
    published_results = staging / "results.json"
    provenance._copy_bytes(legacy_results, published_results)
    payloads["results"] = legacy_results
    tables = legacy_manifest.get("tables")
    provenance.require(
        isinstance(tables, dict) and set(tables) == set(TABLE_NAMES),
        "legacy metric table inventory changed",
    )
    for name in TABLE_NAMES:
        pair = tables[name]
        provenance.require(
            isinstance(pair, dict) and set(pair) == {"csv", "json"},
            f"legacy metric table ref changed: {name}",
        )
        table_refs[name] = {}
        for kind in ("csv", "json"):
            ref = pair[kind]
            provenance.require(
                isinstance(ref, dict)
                and set(ref) >= {"path", "sha256"}
                and Path(str(ref["path"])).name
                == f"{TABLE_BASENAMES[name]}.{'csv' if kind == 'csv' else 'json'}",
                f"legacy metric {name}/{kind} ref changed",
            )
            source = legacy_root / str(ref["path"])
            provenance.regular_file(source, f"legacy metric {name}/{kind}")
            provenance.require(
                provenance.sha256_file(source) == ref["sha256"],
                f"legacy metric {name}/{kind} SHA changed",
            )
            target = staging / source.name
            provenance._copy_bytes(source, target)
            expected_names.add(source.name)
            payloads[f"{name}.{kind}"] = source
            table_refs[name][kind] = provenance.logical_ref(
                "metrics_root", staging, target
            )
    provenance.require(
        {path.name for path in legacy_root.iterdir() if path.is_file()}
        == expected_names,
        "legacy metric root contains an unexpected file",
    )
    return published_results, table_refs, payloads


def _load_eligibility_v2(
    *,
    eligibility_root: Path,
    canonical_root: Path,
    project_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    key_commitments_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    path = eligibility_root / "eligibility_manifest.json"
    manifest = provenance.load_json(path, "eligibility-v2 manifest")
    provenance.require(
        set(manifest) == provenance.ELIGIBILITY_MANIFEST_FIELDS,
        "eligibility-v2 manifest schema changed",
    )
    provenance.require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == provenance.CANONICAL_PROTOCOL
        and manifest["status"] == provenance.ELIGIBILITY_STATUS,
        "eligibility-v2 manifest identity/status changed",
    )
    provenance._validate_self_hash(
        manifest, "manifest_sha256", "eligibility-v2 manifest"
    )
    provenance.require(
        isinstance(manifest["inputs"], dict)
        and set(manifest["inputs"])
        == {
            "canonical_manifest",
            "canonical_scores",
            "original_only_key",
            "key_commitments",
        }
        and isinstance(manifest["artifacts"], dict)
        and set(manifest["artifacts"])
        == {"original_eligibility", "shared_capability_subsets"},
        "eligibility-v2 nested inventory changed",
    )
    review_root = key_commitments_path.parent.parent.parent
    roots = {
        "repo_root": project_root,
        "amendment_root": amendment_path.parent,
        "pre_metric_root": pre_metric_path.parent,
        "canonical_root": canonical_root,
        "eligibility_root": eligibility_root,
        "review_root": review_root,
    }
    provenance.resolve_logical_ref(
        manifest["provenance_amendment"],
        roots,
        "eligibility amendment",
        amendment_path,
    )
    provenance._validate_registry_ref(
        manifest["evaluation_code_registry"],
        "eligibility-v2 evaluation registry",
    )
    provenance._validate_authority_ref(
        manifest["provenance_amendment"],
        root_id="amendment_root",
        path=amendment_path.name,
        sha256=provenance.sha256_file(amendment_path),
        size_bytes=amendment_path.stat().st_size,
        label="eligibility-v2 amendment",
    )
    provenance._validate_authority_ref(
        manifest["pre_metric_code_freeze"],
        root_id="pre_metric_root",
        path=pre_metric_path.name,
        sha256=provenance.sha256_file(pre_metric_path),
        size_bytes=pre_metric_path.stat().st_size,
        label="eligibility-v2 pre-metric freeze",
    )
    wrapper_source = project_root / provenance.CRITICAL_COMPONENT_PATHS[
        "canonicalization_wrapper"
    ]
    provenance._validate_authority_ref(
        manifest["wrapper_implementation"],
        root_id="repo_root",
        path=provenance.CRITICAL_COMPONENT_PATHS["canonicalization_wrapper"],
        sha256=provenance.sha256_file(wrapper_source),
        size_bytes=wrapper_source.stat().st_size,
        label="eligibility-v2 wrapper",
    )
    provenance.resolve_logical_ref(
        manifest["pre_metric_code_freeze"],
        roots,
        "eligibility pre-metric freeze",
        pre_metric_path,
    )
    provenance.resolve_logical_ref(
        manifest["wrapper_implementation"],
        roots,
        "eligibility wrapper",
        Path(provenance.__file__),
    )
    provenance.resolve_logical_ref(
        manifest["inputs"]["canonical_manifest"],
        roots,
        "eligibility canonical manifest",
        canonical_root / "canonical_manifest.json",
    )
    provenance.resolve_logical_ref(
        manifest["inputs"]["canonical_scores"],
        roots,
        "eligibility canonical scores",
        canonical_root / "canonical_scores.jsonl",
    )
    provenance.resolve_logical_ref(
        manifest["inputs"]["key_commitments"],
        roots,
        "eligibility key commitments",
        key_commitments_path,
    )
    provenance.resolve_logical_ref(
        manifest["inputs"]["original_only_key"],
        roots,
        "eligibility Original-only key",
    )
    provenance.require(
        manifest["inputs"]["canonical_manifest"]["root_id"]
        == "canonical_root"
        and manifest["inputs"]["canonical_manifest"]["path"]
        == "canonical_manifest.json"
        and manifest["inputs"]["canonical_scores"]["root_id"]
        == "canonical_root"
        and manifest["inputs"]["canonical_scores"]["path"]
        == "canonical_scores.jsonl"
        and manifest["inputs"]["original_only_key"]["root_id"]
        == "review_root"
        and manifest["inputs"]["original_only_key"]["path"]
        == "review_package/private/original_only_key.jsonl"
        and manifest["inputs"]["key_commitments"]["root_id"]
        == "review_root"
        and manifest["inputs"]["key_commitments"]["path"]
        == "review_package/public/key_commitments.json",
        "eligibility-v2 input roots/locations changed",
    )
    for name in ("original_eligibility", "shared_capability_subsets"):
        provenance.resolve_logical_ref(
            manifest["artifacts"][name], roots, f"eligibility artifact {name}"
        )
        provenance.require(
            manifest["artifacts"][name]["root_id"] == "eligibility_root"
            and manifest["artifacts"][name]["path"]
            == {
                "original_eligibility": "original_eligibility.csv",
                "shared_capability_subsets": "shared_capability_subsets.csv",
            }[name],
            f"eligibility-v2 artifact location changed: {name}",
        )
    provenance._validate_derivation(
        value=manifest["registered_v1_derivation"],
        implementation_path="scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
        implementation_sha256=provenance.V1_FILES[3]["sha256"],
        entry_point="freeze_original_eligibility",
        payload_refs=manifest["artifacts"],
        label="eligibility-v2",
    )
    counts = manifest["counts"]
    provenance.require(
        isinstance(counts, dict)
        and set(counts)
        == {
            "original_causal_rows",
            "semantic_causal_cases",
            "shared_eligible_cases",
        }
        and counts["original_causal_rows"] == 336
        and counts["semantic_causal_cases"] == 168
        and type(counts["shared_eligible_cases"]) is int
        and 0 <= counts["shared_eligible_cases"] <= 168,
        "eligibility-v2 counts changed",
    )
    provenance.require(
        manifest["key_access"]
        == {
            "global_full_method_key_opened_before_human_canonicalization": True,
            "tier_1_original_only_key_read_by_this_stage": True,
            "tier_2_full_method_key_read_by_this_stage": False,
            "global_pre_unblinding_claim": False,
        },
        "eligibility-v2 key-access record changed",
    )
    provenance.scan_published_tree(eligibility_root)
    return manifest, roots


def _legacy_eligibility_projection(
    *,
    destination: Path,
    manifest: Mapping[str, Any],
    roots: Mapping[str, Path],
    compatibility_canonical_manifest: Path,
    compatibility_canonical_scores: Path,
    original_key_path: Path,
    key_commitments_path: Path,
    seals: dict[Path, provenance.InputSeal] | None = None,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True)
    destination.chmod(0o700)
    original_source = provenance.resolve_logical_ref(
        manifest["artifacts"]["original_eligibility"],
        roots,
        "Original eligibility",
    )
    shared_source = provenance.resolve_logical_ref(
        manifest["artifacts"]["shared_capability_subsets"],
        roots,
        "shared capability subsets",
    )
    original_path = destination / "original_eligibility.csv"
    shared_path = destination / "shared_capability_subsets.csv"
    if seals is None:
        provenance._copy_bytes(original_source, original_path)
        provenance._copy_bytes(shared_source, shared_path)
    else:
        provenance._snapshot_input(
            original_source,
            original_path,
            seals,
            "Original-eligibility compatibility snapshot",
            expected_sha256=str(
                manifest["artifacts"]["original_eligibility"]["sha256"]
            ),
            expected_size=int(
                manifest["artifacts"]["original_eligibility"]["size_bytes"]
            ),
        )
        provenance._snapshot_input(
            shared_source,
            shared_path,
            seals,
            "shared-subsets compatibility snapshot",
            expected_sha256=str(
                manifest["artifacts"]["shared_capability_subsets"]["sha256"]
            ),
            expected_size=int(
                manifest["artifacts"]["shared_capability_subsets"]["size_bytes"]
            ),
        )
    legacy_manifest = {
        "schema_version": 1,
        "protocol": "causal_role_erasure_7m_human_canonicalization_v1",
        "status": "original_eligibility_and_shared_subsets_frozen_before_full_key_opening",
        "evaluation_code_registry_sha256": provenance.REGISTRY_SHA256,
        "inputs": {
            "canonical_manifest": provenance.legacy_ref(
                compatibility_canonical_manifest
            ),
            "canonical_scores": provenance.legacy_ref(
                compatibility_canonical_scores
            ),
            "original_only_key": provenance.legacy_ref(original_key_path),
            "key_commitments": provenance.legacy_ref(key_commitments_path),
        },
        "artifacts": {
            "original_eligibility": provenance.legacy_ref(original_path),
            "shared_capability_subsets": provenance.legacy_ref(shared_path),
        },
        "counts": manifest["counts"],
        "full_method_key_opened": False,
    }
    provenance._write_json(destination / "eligibility_manifest.json", legacy_manifest)
    return original_path, shared_path


def validate_score_manifest_v2(
    *,
    score_manifest_path: Path,
    project_root: Path,
    review_root: Path,
    canonical_root: Path,
    eligibility_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = provenance.load_json(score_manifest_path, "formal score manifest v2")
    provenance.require(
        set(manifest) == SCORE_MANIFEST_FIELDS,
        "formal score manifest v2 schema changed",
    )
    provenance.require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == PROTOCOL
        and manifest["status"] == STATUS,
        "formal score manifest v2 identity/status changed",
    )
    provenance._validate_self_hash(
        manifest, "manifest_sha256", "formal score manifest v2"
    )
    roots = {
        "repo_root": project_root,
        "review_root": review_root,
        "canonical_root": canonical_root,
        "eligibility_root": eligibility_root,
        "metrics_root": score_manifest_path.parent,
        "amendment_root": amendment_path.parent,
        "pre_metric_root": pre_metric_path.parent,
    }
    amendment, pre_metric = provenance.validate_execution_authority(
        amendment_path, pre_metric_path, project_root
    )
    provenance._validate_authority_ref(
        manifest["provenance_amendment"],
        root_id="amendment_root",
        path=amendment_path.name,
        sha256=provenance.sha256_file(amendment_path),
        size_bytes=amendment_path.stat().st_size,
        label="score-manifest amendment",
    )
    provenance._validate_authority_ref(
        manifest["pre_metric_code_freeze"],
        root_id="pre_metric_root",
        path=pre_metric_path.name,
        sha256=provenance.sha256_file(pre_metric_path),
        size_bytes=pre_metric_path.stat().st_size,
        label="score-manifest pre-metric freeze",
    )
    provenance._validate_registry_ref(
        manifest["evaluation_code_registry"],
        "score-manifest evaluation registry",
    )
    wrapper_path = project_root / provenance.CRITICAL_COMPONENT_PATHS[
        "formal_metrics_wrapper"
    ]
    provenance._validate_authority_ref(
        manifest["wrapper_implementation"],
        root_id="repo_root",
        path=provenance.CRITICAL_COMPONENT_PATHS["formal_metrics_wrapper"],
        sha256=provenance.sha256_file(wrapper_path),
        size_bytes=wrapper_path.stat().st_size,
        label="score-manifest wrapper",
    )
    inputs = manifest["inputs"]
    provenance.require(
        isinstance(inputs, dict) and set(inputs) == set(SCORE_INPUTS),
        "score-manifest input inventory changed",
    )
    paths: dict[str, Path] = {}
    for name, (root_id, expected_path) in SCORE_INPUTS.items():
        ref = provenance.validate_logical_ref(
            inputs[name], f"score-manifest input {name}"
        )
        provenance.require(
            ref["root_id"] == root_id and ref["path"] == expected_path,
            f"score-manifest input location changed: {name}",
        )
        paths[name] = provenance.resolve_logical_ref(
            ref, roots, f"score-manifest input {name}"
        )
    provenance.require(
        paths["canonical_manifest"]
        == (canonical_root / "canonical_manifest.json").resolve()
        and paths["eligibility_manifest"]
        == (eligibility_root / "eligibility_manifest.json").resolve(),
        "score manifest binds different upstream roots",
    )
    registry_path, _ = provenance.validate_review_authority(
        project_root=project_root,
        amendment=amendment,
        pre_metric=pre_metric,
        key_commitments_path=paths["key_commitments"],
    )
    provenance.require(
        registry_path
        == provenance.resolve_logical_ref(
            {
                key: manifest["evaluation_code_registry"][key]
                for key in provenance.LOGICAL_REF_FIELDS
            },
            roots,
            "score-manifest registry",
        ),
        "score manifest registry path differs from authority chain",
    )
    provenance._binding_matches_file(
        amendment["bindings"]["tier_2_full"],
        paths["full_method_key"],
        "score-manifest full method key",
    )
    provenance._binding_matches_file(
        amendment["bindings"]["formal_cases"],
        paths["formal_cases"],
        "score-manifest formal cases",
    )
    provenance.load_canonical_v2(
        canonical_root=canonical_root,
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric_path=pre_metric_path,
        key_commitments_path=paths["key_commitments"],
    )
    _load_eligibility_v2(
        eligibility_root=eligibility_root,
        canonical_root=canonical_root,
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric_path=pre_metric_path,
        key_commitments_path=paths["key_commitments"],
    )
    provenance.require(
        manifest["key_access"]
        == {
            "global_full_method_key_opened_before_human_canonicalization": True,
            "tier_2_full_method_key_read_by_this_stage": True,
            "global_pre_unblinding_claim": False,
        },
        "score-manifest key-access disclosure changed",
    )
    results_ref = provenance.validate_logical_ref(
        manifest["results"], "score-manifest results"
    )
    provenance.require(
        results_ref["root_id"] == "metrics_root"
        and results_ref["path"] == "results.json",
        "score-manifest results location changed",
    )
    paths["results"] = provenance.resolve_logical_ref(
        results_ref, roots, "score-manifest results"
    )
    tables = manifest["tables"]
    provenance.require(
        isinstance(tables, dict) and set(tables) == set(TABLE_NAMES),
        "score-manifest table inventory changed",
    )
    payload_refs: dict[str, Mapping[str, Any]] = {"results": results_ref}
    for name in TABLE_NAMES:
        pair = tables[name]
        provenance.require(
            isinstance(pair, dict) and set(pair) == {"csv", "json"},
            f"score-manifest table pair changed: {name}",
        )
        for kind in ("csv", "json"):
            ref = provenance.validate_logical_ref(
                pair[kind], f"score-manifest table {name}/{kind}"
            )
            expected = f"{TABLE_BASENAMES[name]}.{kind}"
            provenance.require(
                ref["root_id"] == "metrics_root" and ref["path"] == expected,
                f"score-manifest table location changed: {name}/{kind}",
            )
            paths[f"{name}_{kind}"] = provenance.resolve_logical_ref(
                ref, roots, f"score-manifest table {name}/{kind}"
            )
            payload_refs[f"{name}.{kind}"] = ref
    provenance._validate_derivation(
        value=manifest["registered_v1_derivation"],
        implementation_path="scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
        implementation_sha256=provenance.V1_FILES[4]["sha256"],
        entry_point="compute_formal_metrics",
        payload_refs=payload_refs,
        label="formal-score-v2",
    )
    results = provenance.load_json(paths["results"], "legacy scientific results")
    provenance.require(
        results.get("schema_version") == 1
        and results.get("protocol")
        == "causal_role_erasure_7m_formal_metrics_v1"
        and results.get("status") == "formal_metrics_complete"
        and results.get("evaluation_code_registry_sha256")
        == provenance.REGISTRY_SHA256,
        "legacy scientific results identity/status changed",
    )
    result_tables = results.get("tables")
    provenance.require(
        isinstance(result_tables, dict) and set(result_tables) == set(TABLE_NAMES),
        "legacy scientific result table inventory changed",
    )
    for name in TABLE_NAMES:
        for kind in ("csv", "json"):
            provenance.require(
                result_tables[name][kind]["sha256"]
                == tables[name][kind]["sha256"],
                f"legacy results/score-manifest table SHA differs: {name}/{kind}",
            )
    provenance.scan_published_tree(score_manifest_path.parent)
    return manifest, paths


def compute_formal_metrics_v2(
    *,
    project_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    canonical_root: Path,
    eligibility_root: Path,
    full_key_path: Path,
    key_commitments_path: Path,
    formal_cases_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    amendment, pre_metric = provenance.validate_execution_authority(
        amendment_path, pre_metric_path, project_root
    )
    _require_private_provenance_authority(pre_metric)
    registry_path, registry = provenance.validate_review_authority(
        project_root=project_root,
        amendment=amendment,
        pre_metric=pre_metric,
        key_commitments_path=key_commitments_path,
    )
    provenance._binding_matches_file(
        amendment["bindings"]["tier_2_full"], full_key_path, "full method key"
    )
    provenance._binding_matches_file(
        amendment["bindings"]["formal_cases"], formal_cases_path, "formal cases"
    )
    canonical_manifest, canonical_roots = provenance.load_canonical_v2(
        canonical_root=canonical_root,
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric_path=pre_metric_path,
        key_commitments_path=key_commitments_path,
    )
    eligibility_manifest, eligibility_roots = _load_eligibility_v2(
        eligibility_root=eligibility_root,
        canonical_root=canonical_root,
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric_path=pre_metric_path,
        key_commitments_path=key_commitments_path,
    )
    original_key_path = provenance.resolve_logical_ref(
        eligibility_manifest["inputs"]["original_only_key"],
        eligibility_roots,
        "eligibility Original-only key",
    )
    seals: dict[Path, provenance.InputSeal] = {}
    authority_payloads: dict[Path, bytes] = {}
    amendment_seal = provenance._capture_validated_json(
        amendment_path,
        seals,
        "formal-metrics provenance amendment",
        amendment,
    )
    pre_metric_seal = provenance._capture_validated_json(
        pre_metric_path,
        seals,
        "formal-metrics pre-metric code freeze",
        pre_metric,
    )
    registry_seal = provenance._capture_validated_json(
        registry_path,
        seals,
        "formal-metrics evaluation code registry",
        registry,
        expected_sha256=provenance.REGISTRY_FILE_SHA256,
        expected_size=provenance.REGISTRY_FILE_SIZE,
        payloads=authority_payloads,
    )
    canonical_manifest_seal = provenance._capture_validated_json(
        canonical_root / "canonical_manifest.json",
        seals,
        "formal-metrics canonical manifest",
        canonical_manifest,
    )
    eligibility_manifest_seal = provenance._capture_validated_json(
        eligibility_root / "eligibility_manifest.json",
        seals,
        "formal-metrics eligibility manifest",
        eligibility_manifest,
    )
    wrapper_path = project_root / provenance.CRITICAL_COMPONENT_PATHS[
        "formal_metrics_wrapper"
    ]
    wrapper_component = provenance._component_by_role(pre_metric["components"])[
        "formal_metrics_wrapper"
    ]
    wrapper_seal = provenance._record_input(
        wrapper_path,
        seals,
        "formal-metrics wrapper",
        expected_sha256=str(wrapper_component["sha256"]),
        expected_size=int(wrapper_component["size_bytes"]),
    )
    private_parent = Path(tempfile.mkdtemp(prefix="causal7m-metrics-v2-"))
    private_parent.chmod(0o700)
    staging: Path | None = None
    try:
        snapshot_root = private_parent / "verified_inputs"
        snapshot_root.mkdir(mode=0o700)
        key_snapshot_root = snapshot_root / "keys"
        key_commitments_seal = provenance._snapshot_input(
            key_commitments_path,
            key_snapshot_root / "key_commitments.json",
            seals,
            "formal-metrics key-commitments snapshot",
            expected_sha256=str(amendment["bindings"]["key_commitments"]["sha256"]),
            expected_size=int(amendment["bindings"]["key_commitments"]["size_bytes"]),
        )
        provenance._write_bytes(
            key_snapshot_root / "evaluation_code_registry.json",
            authority_payloads[registry_seal.path],
        )
        full_key_seal = provenance._snapshot_input(
            full_key_path,
            snapshot_root / "full_key.jsonl",
            seals,
            "full method-key snapshot",
            expected_sha256=str(amendment["bindings"]["tier_2_full"]["sha256"]),
        )
        formal_cases_seal = provenance._snapshot_input(
            formal_cases_path,
            snapshot_root / "formal_cases.csv",
            seals,
            "formal-case snapshot",
            expected_sha256=str(amendment["bindings"]["formal_cases"]["sha256"]),
            expected_size=int(amendment["bindings"]["formal_cases"]["size_bytes"]),
        )
        original_key_seal = provenance._snapshot_input(
            original_key_path,
            snapshot_root / "original_only_key.jsonl",
            seals,
            "formal-metrics Original-only key snapshot",
            expected_sha256=str(amendment["bindings"]["tier_1_original_only"]["sha256"]),
        )
        audit_strata_source = provenance.resolve_logical_ref(
            canonical_manifest["inputs"]["audit_strata_key"],
            canonical_roots,
            "formal-metrics audit-strata key",
        )
        audit_strata_snapshot = snapshot_root / "audit_strata_key.jsonl"
        provenance._snapshot_input(
            audit_strata_source,
            audit_strata_snapshot,
            seals,
            "formal-metrics audit-strata snapshot",
            expected_sha256=str(canonical_manifest["inputs"]["audit_strata_key"]["sha256"]),
            expected_size=int(canonical_manifest["inputs"]["audit_strata_key"]["size_bytes"]),
        )
        compatibility_canonical = private_parent / "canonical"
        compatibility_scores, compatibility_manifest = (
            provenance._legacy_canonical_projection(
                destination=compatibility_canonical,
                manifest=canonical_manifest,
                roots=canonical_roots,
                seals=seals,
                key_commitments_override=key_snapshot_root
                / "key_commitments.json",
                audit_strata_override=audit_strata_snapshot,
            )
        )
        compatibility_eligibility = private_parent / "eligibility"
        compatibility_original, compatibility_shared = (
            _legacy_eligibility_projection(
                destination=compatibility_eligibility,
                manifest=eligibility_manifest,
                roots=eligibility_roots,
                compatibility_canonical_manifest=compatibility_manifest,
                compatibility_canonical_scores=compatibility_scores,
                original_key_path=snapshot_root / "original_only_key.jsonl",
                key_commitments_path=key_snapshot_root / "key_commitments.json",
                seals=seals,
            )
        )
        legacy_root = private_parent / "registered_v1"
        provenance._run_registered_stage_isolated(
            project_root=project_root,
            pre_metric_path=pre_metric_path,
            pre_metric=pre_metric,
            private_parent=private_parent,
            stage="metrics",
            seals=seals,
            arguments={
                "canonical_scores_path": compatibility_scores,
                "full_key_path": snapshot_root / "full_key.jsonl",
                "key_commitments_path": key_snapshot_root / "key_commitments.json",
                "original_eligibility_path": compatibility_original,
                "shared_subsets_path": compatibility_shared,
                "formal_cases_path": snapshot_root / "formal_cases.csv",
                "output_root": legacy_root,
            },
        )
        legacy_manifest_path = legacy_root / "score_manifest.json"
        legacy_manifest = provenance.load_json(
            legacy_manifest_path, "legacy score manifest"
        )
        provenance.require(
            legacy_manifest.get("status")
            == "frozen_after_registered_10000_bootstrap_scoring",
            "registered v1 metric stage did not emit its expected status",
        )
        staging = provenance._fresh_staging(output_root)
        published_results, table_refs, payloads = _copy_payloads(
            legacy_root, staging, legacy_manifest
        )
        review_root = key_commitments_path.parent.parent.parent
        body = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "status": STATUS,
            "provenance_amendment": provenance.logical_ref_from_seal(
                "amendment_root", amendment_path.parent, amendment_seal
            ),
            "pre_metric_code_freeze": provenance.logical_ref_from_seal(
                "pre_metric_root", pre_metric_path.parent, pre_metric_seal
            ),
            "evaluation_code_registry": {
                **provenance.logical_ref_from_seal(
                    "review_root", review_root, registry_seal
                ),
                "registry_sha256": provenance.REGISTRY_SHA256,
            },
            "wrapper_implementation": provenance.logical_ref_from_seal(
                "repo_root", project_root, wrapper_seal
            ),
            "registered_v1_derivation": provenance._derivation(
                project_root=project_root,
                implementation=project_root
                / "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
                entry_point="compute_formal_metrics",
                legacy_manifest=legacy_manifest_path,
                payloads=payloads,
            ),
            "key_access": {
                "global_full_method_key_opened_before_human_canonicalization": True,
                "tier_2_full_method_key_read_by_this_stage": True,
                "global_pre_unblinding_claim": False,
            },
            "inputs": {
                "canonical_manifest": provenance.logical_ref_from_seal(
                    "canonical_root", canonical_root, canonical_manifest_seal
                ),
                "canonical_scores": provenance.logical_ref_from_seal(
                    "canonical_root",
                    canonical_root,
                    seals[(canonical_root / "canonical_scores.jsonl").resolve()],
                ),
                "eligibility_manifest": provenance.logical_ref_from_seal(
                    "eligibility_root", eligibility_root, eligibility_manifest_seal
                ),
                "original_eligibility": provenance.logical_ref_from_seal(
                    "eligibility_root",
                    eligibility_root,
                    seals[(eligibility_root / "original_eligibility.csv").resolve()],
                ),
                "shared_capability_subsets": provenance.logical_ref_from_seal(
                    "eligibility_root",
                    eligibility_root,
                    seals[
                        (eligibility_root / "shared_capability_subsets.csv").resolve()
                    ],
                ),
                "full_method_key": provenance.logical_ref_from_seal(
                    "review_root", review_root, full_key_seal
                ),
                "key_commitments": provenance.logical_ref_from_seal(
                    "review_root", review_root, key_commitments_seal
                ),
                "formal_cases": provenance.logical_ref_from_seal(
                    "repo_root", project_root, formal_cases_seal
                ),
            },
            "results": provenance.logical_ref(
                "metrics_root", staging, published_results
            ),
            "tables": table_refs,
        }
        manifest = provenance._sealed(body)
        provenance.require(
            set(manifest) == SCORE_MANIFEST_FIELDS,
            "internal score-manifest-v2 schema drift",
        )
        provenance._write_json(staging / "score_manifest.json", manifest)
        provenance.scan_published_tree(staging)
        provenance.validate_current_v1_sources(project_root)
        provenance._assert_inputs_unchanged(seals)
        provenance._publish(staging, output_root)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(private_parent, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--pre-metric-freeze-manifest", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--eligibility-root", type=Path, required=True)
    parser.add_argument("--full-key", type=Path, required=True)
    parser.add_argument("--key-commitments", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        manifest = compute_formal_metrics_v2(
            project_root=project_root,
            amendment_path=_resolve(project_root, args.amendment),
            pre_metric_path=_resolve(project_root, args.pre_metric_freeze_manifest),
            canonical_root=_resolve(project_root, args.canonical_root),
            eligibility_root=_resolve(project_root, args.eligibility_root),
            full_key_path=_resolve(project_root, args.full_key),
            key_commitments_path=_resolve(project_root, args.key_commitments),
            formal_cases_path=_resolve(project_root, args.formal_cases),
            output_root=_resolve(project_root, args.output_root),
        )
    except (OSError, provenance.ProvenanceV2Error) as exc:
        parser.exit(2, f"v2 formal metrics refused: {exc}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
