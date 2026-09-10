#!/usr/bin/env python3
"""Export fixed human-canonical paper tables from the deviation-aware v2 envelope.

This program is a presentation projection, not a metric evaluator.  It accepts
only the schema-2 formal-metrics manifest, verifies its complete provenance
chain before opening metric values, and emits a fresh deterministic artifact
tree.  The post-unblinding preview workflow is intentionally incompatible.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import canonicalize_causal_role_erasure_7mechanism_formal_v2 as provenance_v2
except ModuleNotFoundError:
    from scripts import canonicalize_causal_role_erasure_7mechanism_formal_v2 as provenance_v2


PROTOCOL = "causal_role_erasure_7m_human_canonical_paper_tables_v1"
STATUS = "human_canonical_paper_tables_complete"
OUTPUT_RELATIVE = Path("paper/tables/human_canonical_v1")
SCORE_PROTOCOL = "causal_role_erasure_7m_formal_metrics_v2"
SCORE_STATUS = "formal_metrics_frozen_after_disclosed_global_full_key_access"
PRE_METRIC_PROTOCOL = "causal_role_erasure_7m_pre_metric_code_freeze_v2"
PRE_METRIC_STATUS = "clean_tree_components_frozen_before_human_canonical_metrics"
CANONICAL_PROTOCOL = "causal_role_erasure_7m_human_canonicalization_v2"
CANONICAL_STATUS = "canonical_anonymous_scores_frozen_after_disclosed_global_full_key_access"
ELIGIBILITY_STATUS = "original_eligibility_and_shared_subsets_frozen_after_disclosed_global_full_key_access"
AMENDMENT_PROTOCOL = "causal_role_erasure_7m_evaluation_provenance_amendment_v2"
AMENDMENT_STATUS = "frozen_after_disclosed_global_full_key_access_before_human_labeling"
LEGACY_RESULTS_PROTOCOL = "causal_role_erasure_7m_formal_metrics_v1"
LEGACY_RESULTS_STATUS = "formal_metrics_complete"
REGISTRY_PROTOCOL = "causal_role_erasure_7m_evaluation_code_registry_v1"
REGISTRY_STATUS = "five_evaluation_implementations_frozen"
REGISTRY_FILE_SHA256 = "c4f10b7129493fd2aa646ab4a38f0b141332dfa8d9da1e64f2c737c33861528f"
REGISTRY_SHA256 = "bafc045ba94a6a5a8771ea8a2dd879c5f42aa219e3b5228b43594494b24eedf4"
REGISTRY_HELPER_SHA256 = "c65278de7a357c4f7db2f12d4927ffe46afe5e1a504524194ab9efcfbefd7536"
FORMAL_METRICS_V1_SHA256 = "5b1a4b823082df9ae2d7e922adbd1bedec4c279593a71208228498b052f40114"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_COMMENT_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
SAFE_COMMENT_ROOT = re.compile(r"[a-z][a-z0-9_]*\Z")
SAFE_REF_PATH = re.compile(r"[A-Za-z0-9_./-]+\Z")
BOOTSTRAP_REPLICATES = 10_000
HEADLINE_SEED = 8_202_601
IDENTIFICATION_SEED = 8_202_602

STREAMS = (
    "wan_original",
    "matched_control",
    "V4",
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
WAN_STREAMS = STREAMS[:3]
COG_STREAMS = STREAMS[3:]
BACKBONE_NAMES = {"wan": "Wan", "cogvideox": "CogVideoX"}
DISPLAY_NAMES = {
    "wan_original": "Wan Original",
    "matched_control": "Matched Control",
    "V4": r"\textsc{SRCD} (ours)",
    "cogvideox_original": "CogVideoX Original",
    "negative_prompt": "Negative Prompt",
    "videoeraser_official": "VideoEraser (official CogVideoX)",
    "t2vunlearning_adapted": "T2VUnlearning-adapted (ours)",
    "safree_cogvideox": "SAFREE-CogVideoX",
}
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
MECHANISM_NAMES = {
    "water_impact": "Water",
    "rigid_collision": "Collision",
    "brittle_fracture": "Fracture",
    "powder_impact": "Powder",
    "elastic_deformation": "Elastic",
    "material_release": "Release",
    "surface_trace": "Trace",
}
HEADLINE_CONTRASTS = (
    "matched_control",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
IDENTIFICATION_CONTROLS = ("generic_paraphrase", "bystander_token")
CONTROL_NAMES = {
    "generic_paraphrase": "Generic Paraphrase",
    "bystander_token": "Bystander Token",
}
ROLE_SUBSETS = ("implicit_footprint", "held_out_source")
NI_METRICS = (
    "specificity_utility",
    "receiver_preservation",
    "video_quality",
    "usable_fraction",
    "specificity_receiver_preservation",
    "specificity_video_quality",
    "specificity_usable_fraction",
)
SPECIFICITY_SUBTYPES = (
    "same_noun_noncausal",
    "role_swap_or_near_causal",
    "same_footprint_alternative_cause",
)
M6_ORDER = (
    ("original_training", "direct"),
    ("original_training", "natural"),
    ("augmentation_bank", "direct"),
    ("augmentation_bank", "natural"),
    ("eval_holdout", "direct"),
    ("eval_holdout", "natural"),
)

PREVIEW_SENTINELS = (
    "NOT_CANONICAL",
    "POST_UNBLINDING_EXPLORATORY_PRELIMINARY",
    "WRITING_PREVIEW_ONLY_NOT_FOR_FINAL_SCIENTIFIC_CLAIMS",
    "paper_preview",
    "post_unblinding_preliminary",
    "ab_order_sensitivity",
)
PREVIEW_FIELDS = {
    "analysis_status",
    "canonical_status",
    "scientific_label",
    "use_restriction",
    "final_authority",
    "formal_claims_permitted",
}

REF_FIELDS = {"root_id", "path", "sha256", "size_bytes"}
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
PRE_METRIC_FIELDS = {
    "schema_version",
    "protocol",
    "protocol_version",
    "status",
    "deviation",
    "v1_authority",
    "source_bindings",
    "components",
    "reproducibility_bindings",
    "pending_outputs",
    "implementation_commit",
    "manifest_sha256",
}
INPUT_NAMES = {
    "canonical_manifest",
    "canonical_scores",
    "eligibility_manifest",
    "original_eligibility",
    "shared_capability_subsets",
    "full_method_key",
    "key_commitments",
    "formal_cases",
}
INPUT_LOCATIONS = {
    "canonical_manifest": ("canonical_root", "canonical_manifest.json"),
    "canonical_scores": ("canonical_root", "canonical_scores.jsonl"),
    "eligibility_manifest": ("eligibility_root", "eligibility_manifest.json"),
    "original_eligibility": ("eligibility_root", "original_eligibility.csv"),
    "shared_capability_subsets": ("eligibility_root", "shared_capability_subsets.csv"),
    "full_method_key": ("review_root", "review_package/private/full_key.jsonl"),
    "key_commitments": ("review_root", "review_package/public/key_commitments.json"),
    "formal_cases": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"),
}
DECLARATION_ONLY_INPUTS = {"canonical_manifest", "canonical_scores", "full_method_key"}
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
TABLE_BASENAMES = {**{name: name for name in TABLE_NAMES}, "identification": "identification_contrasts"}

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "per_case_metrics": (
        "anonymous_review_id", "evaluation_partition", "stream", "backbone_family",
        "case_id", "mechanism", "case_kind", "seed", "m6_pair_id",
        "footprint_lexicalization", "source_membership", "generalization_group",
        "usable", "receiver_preservation_norm", "video_quality_norm",
        "original_eligible", "ces", "su", "source_absent", "footprint_absent",
        "strict_success", "source_visibility", "footprint_visibility",
        "receiver_preservation", "video_quality", "protected_object_visibility",
        "noncausal_role_adherence",
    ),
    "mechanism_metrics": (
        "stream", "mechanism", "causal_n", "specificity_n", "ces",
        "specificity_utility", "original_capability_coverage",
        "causal_usable_fraction", "specificity_usable_fraction",
        "causal_receiver_preservation", "causal_video_quality",
        "specificity_receiver_preservation", "specificity_video_quality",
        "source_absent_rate", "footprint_absent_rate", "strict_success_rate",
        "eligible_source_clear_to_partial_rate", "eligible_source_clear_to_absent_rate",
    ),
    "macro_metrics": (
        "stream", "mechanism_count", "ces", "specificity_utility",
        "original_capability_coverage", "causal_usable_fraction",
        "specificity_usable_fraction", "causal_receiver_preservation",
        "causal_video_quality", "specificity_receiver_preservation",
        "specificity_video_quality", "source_absent_rate", "footprint_absent_rate",
        "strict_success_rate", "eligible_source_clear_to_partial_rate",
        "eligible_source_clear_to_absent_rate",
    ),
    "headline_contrasts": (
        "contrast", "estimable", "point_estimate", "ci95_lower", "ci95_upper",
        "one_sided_p", "bootstrap_replicates", "bootstrap_seed",
        "holm_adjusted_p", "registered_win",
    ),
    "headline_mechanism_contrasts": (
        "contrast", "mechanism", "n", "estimable", "point_estimate",
    ),
    "noninferiority": (
        "metric", "role", "margin", "point_estimate", "ci95_lower",
        "ci95_upper", "one_sided_p", "bootstrap_replicates", "bootstrap_seed",
        "noninferior",
    ),
    "identification": (
        "control", "point_estimate", "ci95_lower", "ci95_upper", "one_sided_p",
        "bootstrap_replicates", "bootstrap_seed",
        "implicit_two_mechanism_point_estimate", "implicit_water_point_estimate",
        "implicit_fracture_point_estimate", "specificity_margin",
        "specificity_point_estimate", "specificity_ci95_lower",
        "specificity_ci95_upper", "specificity_noninferior", "holm_adjusted_p",
        "novelty_condition_pass",
    ),
    "role_conditioned_evidence": (
        "subset", "per_mechanism_n", "point_estimate", "ci95_lower", "ci95_upper",
        "one_sided_p", "bootstrap_replicates", "bootstrap_seed", "registered_positive",
    ),
    "m6_pairs": (
        "stream", "mechanism", "m6_pair_id", "causal_case_id",
        "specificity_case_id", "causal_ces", "specificity_utility", "pair_mean",
    ),
    "secondary_metrics": (
        "stream", "mechanism", "original_capability_coverage",
        "causal_usable_fraction", "specificity_usable_fraction",
        "causal_receiver_preservation", "causal_video_quality",
        "specificity_receiver_preservation", "specificity_video_quality",
        "source_absent_rate", "footprint_absent_rate", "strict_success_rate",
        "eligible_source_clear_to_partial_rate", "eligible_source_clear_to_absent_rate",
    ),
}
CSV_SCHEMAS = {
    **TABLE_SCHEMAS,
    # canonical_scores.jsonl is canonical-key-sorted.  Consequently the first
    # causal score mapping contributes its fields to the registered v1 CSV in
    # this physical order, followed by the two specificity-only fields.
    "per_case_metrics": (
        "anonymous_review_id", "evaluation_partition", "stream", "backbone_family",
        "case_id", "mechanism", "case_kind", "seed", "m6_pair_id",
        "footprint_lexicalization", "source_membership", "generalization_group",
        "usable", "receiver_preservation_norm", "video_quality_norm",
        "original_eligible", "ces", "su", "source_absent", "footprint_absent",
        "strict_success", "footprint_visibility", "receiver_preservation",
        "source_visibility", "video_quality", "noncausal_role_adherence",
        "protected_object_visibility",
    ),
}
EXPECTED_ROWS = {
    "per_case_metrics": 2448,
    "mechanism_metrics": 56,
    "macro_metrics": 8,
    "headline_contrasts": 5,
    "headline_mechanism_contrasts": 35,
    "noninferiority": 7,
    "identification": 2,
    "role_conditioned_evidence": 2,
    "m6_pairs": 336,
    "secondary_metrics": 56,
}
ELIGIBILITY_SCHEMA = (
    "anonymous_review_id", "backbone_family", "case_id", "mechanism", "eligible",
    "source_visibility", "footprint_visibility", "receiver_preservation", "video_quality",
)
SHARED_SCHEMA = (
    "case_id", "mechanism", "wan_original_eligible",
    "cogvideox_original_eligible", "shared_eligible",
)
FORMAL_CASES_SCHEMA = (
    "protocol_version", "case_id", "global_case_index", "mechanism_index",
    "historical_capability_index", "mechanism", "mechanism_name", "case_kind",
    "generalization_group", "source_membership", "prompt_style",
    "footprint_lexicalization", "specificity_subtype", "semantic_replicate",
    "source_id", "source_object", "receiver_id", "receiver", "prompt",
    "expected_trigger", "expected_footprint", "expected_counterfactual_state",
    "protected_object", "acceptable_alternative_cause", "m6_pair_id", "seed",
    "seed_nonce", "num_frames", "fps", "height", "width",
)


class FinalTableError(ValueError):
    """The final-table trust chain, schema, or deterministic projection failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalTableError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")


def _reject_json_constant(value: str) -> None:
    raise FinalTableError(f"JSON contains non-finite number {value!r}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"JSON repeats key {key!r}")
        value[key] = item
    return value


def _check_finite_json(value: Any, label: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_json(item, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_json(item, f"{label}/{index}")


def parse_json_bytes(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalTableError(f"{label} is not valid UTF-8 JSON") from exc
    _check_finite_json(value, label)
    return value


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    regular_file(path, label)
    raw = path.read_bytes()
    value = parse_json_bytes(raw, label)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def self_digest(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(body))


def digest_without(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha256_bytes(canonical_json_bytes(body))


def check_self_digest(value: Mapping[str, Any], label: str) -> None:
    claimed = value.get("manifest_sha256")
    require(isinstance(claimed, str) and HEX64.fullmatch(claimed) is not None, f"{label} self digest is missing or invalid")
    require(self_digest(value) == claimed, f"{label} self digest mismatch")


def check_no_preview(value: Any, label: str) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    for sentinel in PREVIEW_SENTINELS:
        require(sentinel not in text, f"{label} contains forbidden preview sentinel {sentinel!r}")


def safe_relative_path(value: Any, label: str) -> PurePosixPath:
    require(isinstance(value, str) and value, f"{label} path is empty")
    require(SAFE_REF_PATH.fullmatch(value) is not None, f"{label} path contains unsafe/comment-control characters")
    require("\\" not in value, f"{label} path must use POSIX separators")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"{label} path must be relative")
    require(all(part not in ("", ".", "..") for part in path.parts), f"{label} path is noncanonical")
    require(str(path) == value, f"{label} path is not canonical POSIX form")
    return path


def validate_ref_declaration(ref: Any, label: str, *, extra: set[str] | None = None) -> dict[str, Any]:
    extra = extra or set()
    require(isinstance(ref, dict) and set(ref) == REF_FIELDS | extra, f"{label} reference schema changed")
    root_id = ref["root_id"]
    safe_relative_path(ref["path"], label)
    sha = ref["sha256"]
    size = ref["size_bytes"]
    require(isinstance(root_id, str) and SAFE_COMMENT_ROOT.fullmatch(root_id) is not None, f"{label} root_id is invalid")
    require(isinstance(sha, str) and HEX64.fullmatch(sha) is not None, f"{label} SHA is not lowercase SHA-256")
    require(type(size) is int and size >= 0, f"{label} size must be a nonnegative integer")
    return dict(ref)


def validate_ref(ref: Any, roots: Mapping[str, Path], label: str, *, extra: set[str] | None = None) -> Path:
    declared = validate_ref_declaration(ref, label, extra=extra)
    root_id = declared["root_id"]
    require(root_id in roots, f"{label} has unknown root_id {root_id!r}")
    rel = safe_relative_path(declared["path"], label)
    sha = declared["sha256"]
    size = declared["size_bytes"]
    root = roots[root_id].resolve()
    path = (root / Path(*rel.parts)).resolve()
    require(path == root or root in path.parents, f"{label} escapes its declared root")
    regular_file(path, label)
    require(path.stat().st_size == size, f"{label} size mismatch")
    require(sha256_file(path) == sha, f"{label} SHA mismatch")
    return path


def snapshot_file(seals: dict[Path, tuple[int, str, str]], path: Path, label: str) -> None:
    path = path.resolve()
    regular_file(path, label)
    observed = (path.stat().st_size, sha256_file(path), label)
    if path in seals:
        require(seals[path][:2] == observed[:2], f"{label} changed while collecting the authority snapshot")
    else:
        seals[path] = observed


def snapshot_logical_refs(
    value: Any,
    roots: Mapping[str, Path],
    seals: dict[Path, tuple[int, str, str]],
    label: str,
) -> None:
    if isinstance(value, dict):
        if REF_FIELDS <= set(value) and isinstance(value.get("root_id"), str):
            root_id = value["root_id"]
            # Some upstream manifests retain private review-process roots that
            # this presentation-only stage neither receives nor opens.  The
            # strict v2 validator checks their declarations; snapshot every
            # reference whose concrete root is available to this process.
            if root_id not in roots:
                return
            path = provenance_v2.resolve_logical_ref(
                {key: value[key] for key in REF_FIELDS}, roots, label
            )
            snapshot_file(seals, path, label)
            return
        for key, item in value.items():
            snapshot_logical_refs(item, roots, seals, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            snapshot_logical_refs(item, roots, seals, f"{label}/{index}")


def verify_snapshot(seals: Mapping[Path, tuple[int, str, str]]) -> None:
    for path, (size, digest, label) in seals.items():
        regular_file(path, label)
        require(path.stat().st_size == size and sha256_file(path) == digest, f"{label} changed during export")


def collect_authority_snapshot(
    *,
    roots: Mapping[str, Path],
    exporter_path: Path,
    score_path: Path,
    score: Mapping[str, Any],
    pre_metric_path: Path,
    pre_metric: Mapping[str, Any],
    amendment_path: Path,
) -> dict[Path, tuple[int, str, str]]:
    """Seal every concrete authority file available to the exporter."""

    seals: dict[Path, tuple[int, str, str]] = {}
    for path, label in (
        (exporter_path, "live final table exporter"),
        (score_path, "v2 score manifest"),
        (pre_metric_path, "pre-metric code freeze"),
        (amendment_path, "provenance amendment"),
    ):
        snapshot_file(seals, path, label)
    score_snapshot = dict(score)
    score_snapshot["inputs"] = {
        name: ref
        for name, ref in score["inputs"].items()
        if name not in DECLARATION_ONLY_INPUTS
    }
    snapshot_logical_refs(score_snapshot, roots, seals, "v2 score manifest")
    snapshot_logical_refs(pre_metric, roots, seals, "pre-metric code freeze")

    for index, component in enumerate(pre_metric.get("components", [])):
        if isinstance(component, dict) and isinstance(component.get("path"), str):
            relative = safe_relative_path(component["path"], f"pre-metric component {index}")
            snapshot_file(
                seals,
                roots["repo_root"] / Path(*relative.parts),
                f"pre-metric component {index}",
            )

    amendment, _ = load_json(amendment_path, "provenance amendment snapshot")
    bindings = amendment.get("bindings")
    if isinstance(bindings, dict):
        for name in ("formal_cases", "identification_subset", "evaluation_code_registry_validator"):
            binding = bindings.get(name)
            if isinstance(binding, dict) and isinstance(binding.get("path"), str):
                relative = safe_relative_path(binding["path"], f"amendment binding {name}")
                snapshot_file(
                    seals,
                    roots["repo_root"] / Path(*relative.parts),
                    f"amendment binding {name}",
                )

    registry_path = validate_ref(
        score["evaluation_code_registry"], roots, "evaluation registry snapshot", extra={"registry_sha256"}
    )
    registry, _ = load_json(registry_path, "evaluation registry snapshot")
    for index, row in enumerate(registry.get("files", [])):
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            relative = safe_relative_path(row["path"], f"registry file {index}")
            snapshot_file(
                seals,
                roots["repo_root"] / Path(*relative.parts),
                f"registry file {index}",
            )

    inputs = score.get("inputs")
    if isinstance(inputs, dict):
        for name in ("eligibility_manifest",):
            ref = inputs.get(name)
            if isinstance(ref, dict):
                manifest_path = validate_ref(ref, roots, f"{name} snapshot")
                manifest, _ = load_json(manifest_path, f"{name} snapshot")
                public_manifest = dict(manifest)
                if name == "eligibility_manifest" and isinstance(manifest.get("inputs"), dict):
                    public_manifest["inputs"] = {
                        input_name: input_ref
                        for input_name, input_ref in manifest["inputs"].items()
                        if input_name == "key_commitments"
                    }
                snapshot_logical_refs(public_manifest, roots, seals, f"{name} snapshot")
    verify_snapshot(seals)
    return seals


def validate_pre_metric_freeze(path: Path, exporter_path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = load_json(path, "pre-metric code freeze")
    require(set(value) == PRE_METRIC_FIELDS, "pre-metric freeze top-level schema changed")
    require(value["schema_version"] == 1, "pre-metric freeze schema version changed")
    require(value["protocol"] == PRE_METRIC_PROTOCOL, "pre-metric freeze protocol changed")
    require(value["status"] == PRE_METRIC_STATUS, "pre-metric freeze status changed")
    check_self_digest(value, "pre-metric freeze")
    check_no_preview(value, "pre-metric freeze")
    commit = value["implementation_commit"]
    require(
        isinstance(commit, dict)
        and set(commit) == {"commit", "tree", "worktree_clean"}
        and isinstance(commit["commit"], str) and len(commit["commit"]) == 40
        and isinstance(commit["tree"], str) and len(commit["tree"]) == 40
        and commit["worktree_clean"] is True,
        "pre-metric implementation_commit is not a clean frozen commit/tree",
    )
    components = value["components"]
    require(isinstance(components, list), "pre-metric components must be a list")
    matches = [row for row in components if isinstance(row, dict) and row.get("role") == "final_table_exporter"]
    require(len(matches) == 1, "pre-metric freeze must bind exactly one final table exporter")
    binding = matches[0]
    require(set(binding) == {"path", "sha256", "size_bytes", "role"}, "final table exporter binding schema changed")
    require(binding["path"] == "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py", "final table exporter path changed")
    regular_file(exporter_path, "live final table exporter")
    require(binding["size_bytes"] == exporter_path.stat().st_size, "final table exporter size differs from pre-metric freeze")
    require(binding["sha256"] == sha256_file(exporter_path), "final table exporter SHA differs from pre-metric freeze")
    return value, raw


def validate_score_manifest(
    path: Path,
    roots: Mapping[str, Path],
    pre_metric_path: Path,
    pre_metric: Mapping[str, Any],
    amendment_expected_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Path]]:
    value, raw = load_json(path, "v2 score manifest")
    require(set(value) == SCORE_MANIFEST_FIELDS, "v2 score manifest top-level schema changed")
    require(value["schema_version"] == 2, "v2 score manifest schema version changed")
    require(value["protocol"] == SCORE_PROTOCOL, "v2 score manifest protocol changed")
    require(value["status"] == SCORE_STATUS, "v2 score manifest status changed")
    check_self_digest(value, "v2 score manifest")
    check_no_preview(value, "v2 score manifest")

    amendment_ref = value["provenance_amendment"]
    amendment_path = validate_ref(amendment_ref, roots, "provenance amendment")
    require(amendment_path == amendment_expected_path.resolve(), "score manifest binds a different provenance amendment")
    amendment, _ = load_json(amendment_path, "provenance amendment")
    require(amendment.get("protocol") == AMENDMENT_PROTOCOL and amendment.get("status") == AMENDMENT_STATUS, "provenance amendment identity/status changed")
    require(amendment.get("amendment_sha256") == digest_without(amendment, "amendment_sha256"), "provenance amendment semantic digest mismatch")

    freeze_ref = value["pre_metric_code_freeze"]
    resolved_freeze = validate_ref(freeze_ref, roots, "pre-metric code freeze reference")
    require(resolved_freeze == pre_metric_path.resolve(), "score manifest binds a different pre-metric freeze")

    registry_ref = value["evaluation_code_registry"]
    registry_path = validate_ref(registry_ref, roots, "evaluation code registry", extra={"registry_sha256"})
    require(registry_ref["sha256"] == REGISTRY_FILE_SHA256 and registry_ref["registry_sha256"] == REGISTRY_SHA256, "evaluation registry frozen digests changed")
    registry, _ = load_json(registry_path, "evaluation code registry")
    require(registry.get("protocol") == REGISTRY_PROTOCOL and registry.get("status") == REGISTRY_STATUS and registry.get("registry_sha256") == REGISTRY_SHA256, "evaluation registry identity/status changed")
    require(isinstance(registry.get("files"), list) and len(registry["files"]) == 5, "evaluation registry must bind five files")
    for index, row in enumerate(registry["files"]):
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"}, f"registry file {index} schema changed")
        rel = safe_relative_path(row["path"], f"registry file {index}")
        candidate = (roots["repo_root"] / Path(*rel.parts)).resolve()
        regular_file(candidate, f"registry file {index}")
        require(candidate.stat().st_size == row["size_bytes"] and sha256_file(candidate) == row["sha256"], f"registry file {index} bytes changed")
    helper = roots["repo_root"] / "scripts" / "causal_role_erasure_7mechanism_evaluation_code_registry_v1.py"
    regular_file(helper, "evaluation registry helper")
    require(sha256_file(helper) == REGISTRY_HELPER_SHA256, "evaluation registry helper bytes changed")

    wrapper_path = validate_ref(value["wrapper_implementation"], roots, "formal-metrics v2 wrapper")
    require(wrapper_path == roots["repo_root"] / "scripts" / "compute_causal_role_erasure_7mechanism_formal_metrics_v2.py", "formal-metrics v2 wrapper path changed")
    derivation = value["registered_v1_derivation"]
    require(
        isinstance(derivation, dict)
        and set(derivation) == {"implementation", "entry_point", "legacy_manifest_sha256", "legacy_manifest_authoritative", "scientific_payload_byte_identical", "payload_sha256"}
        and derivation["legacy_manifest_authoritative"] is False
        and derivation["scientific_payload_byte_identical"] is True,
        "registered v1 derivation schema changed",
    )
    legacy_implementation = validate_ref(derivation["implementation"], roots, "registered v1 metrics implementation")
    require(legacy_implementation == roots["repo_root"] / "scripts" / "compute_causal_role_erasure_7mechanism_formal_metrics_v1.py" and derivation["implementation"]["sha256"] == FORMAL_METRICS_V1_SHA256, "registered v1 metrics implementation changed")
    require(derivation["entry_point"] == "compute_formal_metrics" and isinstance(derivation["legacy_manifest_sha256"], str) and len(derivation["legacy_manifest_sha256"]) == 64, "registered v1 derivation values changed")
    access = value["key_access"]
    expected_access = {
        "global_full_method_key_opened_before_human_canonicalization": True,
        "global_pre_unblinding_claim": False,
        "tier_2_full_method_key_read_by_this_stage": True,
    }
    require(access == expected_access, "formal-metrics v2 key-access disclosure changed")

    inputs = value["inputs"]
    require(isinstance(inputs, dict) and set(inputs) == INPUT_NAMES, "v2 score manifest input inventory changed")
    paths: dict[str, Path] = {}
    for name in sorted(INPUT_NAMES):
        declared = validate_ref_declaration(inputs[name], f"score input {name}")
        expected_root, expected_path = INPUT_LOCATIONS[name]
        require(
            declared["root_id"] == expected_root and declared["path"] == expected_path,
            f"score input {name} location changed",
        )
        if name not in DECLARATION_ONLY_INPUTS:
            paths[name] = validate_ref(inputs[name], roots, f"score input {name}")
    results_path = validate_ref(value["results"], roots, "legacy scientific results")
    require(results_path.parent == roots["metrics_root"].resolve(), "results.json must be a direct metrics-root child")
    paths["results"] = results_path

    tables = value["tables"]
    require(isinstance(tables, dict) and set(tables) == set(TABLE_NAMES), "v2 score manifest table inventory changed")
    for name in TABLE_NAMES:
        pair = tables[name]
        require(isinstance(pair, dict) and set(pair) == {"csv", "json"}, f"{name} table pair schema changed")
        for suffix in ("csv", "json"):
            table_path = validate_ref(pair[suffix], roots, f"{name} {suffix}")
            require(table_path.parent == roots["metrics_root"].resolve(), f"{name} {suffix} must be a direct metrics-root child")
            require(table_path.name == f"{TABLE_BASENAMES[name]}.{suffix}", f"{name} {suffix} basename changed")
            paths[f"{name}_{suffix}"] = table_path

    payload_hashes = derivation["payload_sha256"]
    expected_payload_names = {"results", *(f"{name}.{suffix}" for name in TABLE_NAMES for suffix in ("csv", "json"))}
    require(isinstance(payload_hashes, dict) and set(payload_hashes) == expected_payload_names, "registered v1 payload inventory changed")
    require(payload_hashes["results"] == value["results"]["sha256"], "registered v1 results parity hash changed")
    for name in TABLE_NAMES:
        for suffix in ("csv", "json"):
            require(payload_hashes[f"{name}.{suffix}"] == value["tables"][name][suffix]["sha256"], f"registered v1 payload parity changed: {name}/{suffix}")

    eligibility_manifest, _ = load_json(paths["eligibility_manifest"], "eligibility v2 manifest")
    require(eligibility_manifest.get("schema_version") == 2 and eligibility_manifest.get("protocol") == CANONICAL_PROTOCOL and eligibility_manifest.get("status") == ELIGIBILITY_STATUS, "eligibility v2 manifest identity/status changed")
    return value, raw, paths


def validate_v2_authority_envelope(
    *,
    repo_root: Path,
    review_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    key_commitments_path: Path,
    expected_pre_metric: Mapping[str, Any],
    expected_registry_path: Path,
) -> None:
    """Validate the v2 authority without opening Tier-2 or canonical scores."""

    try:
        amendment, strict_pre_metric = provenance_v2.validate_execution_authority(
            amendment_path, pre_metric_path, repo_root
        )
        registry_path, _ = provenance_v2.validate_review_authority(
            project_root=repo_root,
            amendment=amendment,
            pre_metric=strict_pre_metric,
            key_commitments_path=key_commitments_path,
        )
    except Exception as exc:
        if isinstance(exc, FinalTableError):
            raise
        raise FinalTableError(f"strict v2 authority-envelope validation refused: {exc}") from exc
    _scientific_equal(strict_pre_metric, expected_pre_metric, "strict-v2/custom pre-metric freeze")
    require(
        registry_path.resolve() == expected_registry_path.resolve()
        and registry_path.resolve().is_relative_to(review_root.resolve()),
        "strict v2 review authority resolves a different public registry",
    )


def load_json_rows(path: Path, name: str) -> list[dict[str, Any]]:
    regular_file(path, f"{name} JSON")
    raw = path.read_bytes()
    value = parse_json_bytes(raw, f"{name} JSON")
    require(isinstance(value, list), f"{name} JSON must be an array")
    require(raw == canonical_json_bytes(value), f"{name} JSON is not canonical")
    require(len(value) == EXPECTED_ROWS[name], f"{name} row count changed")
    schema = set(TABLE_SCHEMAS[name])
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        require(isinstance(row, dict) and set(row) == schema, f"{name} row {index} schema changed")
        require(not (set(row) & PREVIEW_FIELDS), f"{name} row {index} contains preview fields")
        rows.append(dict(row))
    return rows


def load_csv_rows(path: Path, fields: Sequence[str], expected_rows: int, label: str) -> list[dict[str, str]]:
    regular_file(path, f"{label} CSV")
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise FinalTableError(f"{label} CSV is invalid") from exc
    require(text.endswith("\n"), f"{label} CSV must end with one newline")
    require("\r" not in text and "\x00" not in text, f"{label} CSV contains a noncanonical line ending or NUL")
    canonical = io.StringIO(newline="")
    csv.writer(canonical, lineterminator="\n").writerows(records)
    require(canonical.getvalue() == text, f"{label} CSV uses noncanonical quoting or spacing")
    require(bool(records) and tuple(records[0]) == tuple(fields), f"{label} CSV header changed")
    require(all(len(record) == len(fields) for record in records[1:]), f"{label} CSV has an extra or missing cell")
    require(all(record and any(cell != "" for cell in record) for record in records[1:]), f"{label} CSV contains a blank row")
    rows = [dict(zip(fields, record, strict=True)) for record in records[1:]]
    require(len(rows) == expected_rows, f"{label} CSV row count changed")
    return rows


def csv_matches_json(csv_rows: Sequence[Mapping[str, str]], json_rows: Sequence[Mapping[str, Any]], label: str) -> None:
    require(len(csv_rows) == len(json_rows), f"{label} CSV/JSON length differs")
    for row_index, (csv_row, json_row) in enumerate(zip(csv_rows, json_rows)):
        for field, value in json_row.items():
            cell = csv_row[field]
            if isinstance(value, dict):
                try:
                    parsed = ast.literal_eval(cell)
                except (ValueError, SyntaxError) as exc:
                    raise FinalTableError(f"{label} row {row_index}/{field} structured CSV value is invalid") from exc
                require(parsed == value, f"{label} row {row_index}/{field} CSV/JSON differs")
                require(cell == repr(parsed), f"{label} row {row_index}/{field} CSV structured value is noncanonical")
            elif isinstance(value, bool):
                require(cell == str(value), f"{label} row {row_index}/{field} CSV/JSON differs")
            elif isinstance(value, (int, float)):
                try:
                    parsed = float(cell)
                except ValueError as exc:
                    raise FinalTableError(f"{label} row {row_index}/{field} CSV number is invalid") from exc
                require(math.isfinite(float(value)) and math.isfinite(parsed) and math.isclose(parsed, float(value), rel_tol=0, abs_tol=0), f"{label} row {row_index}/{field} CSV/JSON differs")
                require(cell == str(value), f"{label} row {row_index}/{field} CSV number is noncanonical")
            else:
                require(cell == str(value), f"{label} row {row_index}/{field} CSV/JSON differs")


def number(value: Any, label: str, *, allow_blank: bool = False) -> float | None:
    if value == "" and allow_blank:
        return None
    require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def binary(value: Any, label: str) -> int:
    result = int(number(value, label) or 0)
    require(type(value) is int and result in (0, 1) and value == result, f"{label} is not binary")
    return result


def integer(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    require(type(value) is int, f"{label} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{label} is above {maximum}")
    return value


def unit(value: Any, label: str, *, allow_blank: bool = False) -> float | None:
    result = number(value, label, allow_blank=allow_blank)
    require(result is None or 0 <= result <= 1, f"{label} is outside [0,1]")
    return result


def signed_unit(value: Any, label: str, *, allow_blank: bool = False) -> float | None:
    result = number(value, label, allow_blank=allow_blank)
    require(result is None or -1 <= result <= 1, f"{label} is outside [-1,1]")
    return result


def validate_inventories(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    mechanism = tables["mechanism_metrics"]
    require(Counter((r["stream"], r["mechanism"]) for r in mechanism) == Counter((s, m) for s in STREAMS for m in MECHANISMS), "mechanism inventory is not complete 8x7")
    for index, row in enumerate(mechanism):
        require(integer(row["causal_n"], f"mechanism row {index}/causal_n") == 24 and integer(row["specificity_n"], f"mechanism row {index}/specificity_n") == 18, f"mechanism row {index} counts changed")
        for field in TABLE_SCHEMAS["mechanism_metrics"][4:]:
            unit(row[field], f"mechanism row {index}/{field}", allow_blank=field.startswith("eligible_source_clear_to_"))

    macro = tables["macro_metrics"]
    require([r["stream"] for r in macro] == list(STREAMS), "macro stream order/inventory changed")
    by_mech = {(r["stream"], r["mechanism"]): r for r in mechanism}
    for row in macro:
        require(integer(row["mechanism_count"], f"{row['stream']}/mechanism_count") == 7, f"{row['stream']}: macro mechanism count changed")
        for field in TABLE_SCHEMAS["macro_metrics"][2:]:
            observed = unit(row[field], f"macro {row['stream']}/{field}", allow_blank=field.startswith("eligible_source_clear_to_"))
            children = [number(by_mech[(row["stream"], m)][field], field, allow_blank=True) for m in MECHANISMS]
            expected = sum(float(v) for v in children) / 7 if all(v is not None for v in children) else None
            require((observed is None and expected is None) or (observed is not None and expected is not None and math.isclose(observed, expected, abs_tol=1e-12)), f"{row['stream']}/{field}: macro is not equal-weight seven-mechanism mean")

    headline = tables["headline_contrasts"]
    require([r["contrast"] for r in headline] == list(HEADLINE_CONTRASTS), "headline contrast order/inventory changed")
    for row in headline:
        estimable = binary(row["estimable"], f"headline {row['contrast']} estimable")
        require(integer(row["bootstrap_replicates"], "headline bootstrap replicates") == BOOTSTRAP_REPLICATES and integer(row["bootstrap_seed"], "headline bootstrap seed") == HEADLINE_SEED, "headline bootstrap contract changed")
        registered_win = binary(row["registered_win"], f"headline {row['contrast']} registered win")
        if estimable:
            for field in ("point_estimate", "ci95_lower", "ci95_upper"):
                signed_unit(row[field], f"headline {row['contrast']}/{field}")
            require(float(row["ci95_lower"]) <= float(row["ci95_upper"]), f"headline {row['contrast']} CI is reversed")
        else:
            require(row["point_estimate"] == "" and row["ci95_lower"] == "" and row["ci95_upper"] == "", f"headline {row['contrast']} NE values must be blank")
        for field in ("one_sided_p", "holm_adjusted_p"):
            unit(row[field], f"headline {row['contrast']}/{field}")
        if not estimable:
            require(row["one_sided_p"] == 1.0 and row["holm_adjusted_p"] == 1.0 and registered_win == 0, f"headline {row['contrast']} NE sentinels changed")
        expected_win = int(
            estimable == 1
            and float(row["point_estimate"]) > 0
            and float(row["ci95_lower"]) > 0
            and float(row["holm_adjusted_p"]) < 0.05
        )
        require(registered_win == expected_win, f"headline {row['contrast']} registered-win rule changed")

    hm = tables["headline_mechanism_contrasts"]
    require(Counter((r["contrast"], r["mechanism"]) for r in hm) == Counter((c, m) for c in HEADLINE_CONTRASTS for m in MECHANISMS), "headline mechanism inventory changed")
    for row in hm:
        integer(row["n"], "headline mechanism n", minimum=0, maximum=24)
        estimable = binary(row["estimable"], "headline mechanism estimable")
        require(
            estimable == (1 if row["contrast"] == "matched_control" else int(row["n"] >= 12)),
            "headline mechanism minimum-shared-case rule changed",
        )
        if row["contrast"] == "matched_control":
            require(row["n"] == 24, "Matched headline mechanism count changed")
        if estimable:
            signed_unit(row["point_estimate"], "headline mechanism point")
        else:
            require(row["point_estimate"] == "", "headline mechanism NE point must be blank")
    hm_by = {(row["contrast"], row["mechanism"]): row for row in hm}
    for row in headline:
        expected_estimable = int(all(int(hm_by[(row["contrast"], mechanism)]["estimable"]) == 1 for mechanism in MECHANISMS))
        require(int(row["estimable"]) == expected_estimable, f"headline {row['contrast']} overall estimability changed")

    ni = tables["noninferiority"]
    require([r["metric"] for r in ni] == list(NI_METRICS), "noninferiority order/inventory changed")
    expected_margins = (-0.10, -0.10, -0.10, -0.05, -0.10, -0.10, -0.05)
    for row, margin in zip(ni, expected_margins):
        require(row["role"] == ("primary" if row["metric"] in NI_METRICS[:4] else "secondary"), "noninferiority role changed")
        require(math.isclose(float(row["margin"]), margin, abs_tol=0), f"{row['metric']}: NI margin changed")
        require(integer(row["bootstrap_replicates"], "NI bootstrap replicates") == BOOTSTRAP_REPLICATES and integer(row["bootstrap_seed"], "NI bootstrap seed") == HEADLINE_SEED, "NI bootstrap contract changed")
        for field in ("point_estimate", "ci95_lower", "ci95_upper"):
            signed_unit(row[field], f"NI {row['metric']}/{field}")
        unit(row["one_sided_p"], f"NI {row['metric']}/one_sided_p")
        require(float(row["ci95_lower"]) <= float(row["ci95_upper"]), f"{row['metric']}: NI CI is reversed")
        decision = binary(row["noninferior"], f"NI {row['metric']} decision")
        require(decision == int(float(row["ci95_lower"]) > margin), f"{row['metric']}: NI decision differs from lower-bound rule")

    identification = tables["identification"]
    require([r["control"] for r in identification] == list(IDENTIFICATION_CONTROLS), "identification order/inventory changed")
    for row in identification:
        require(integer(row["bootstrap_replicates"], "identification bootstrap replicates") == BOOTSTRAP_REPLICATES and integer(row["bootstrap_seed"], "identification bootstrap seed") == IDENTIFICATION_SEED, "identification bootstrap contract changed")
        for field in TABLE_SCHEMAS["identification"]:
            if field not in {"control", "bootstrap_replicates", "bootstrap_seed", "specificity_noninferior", "novelty_condition_pass"}:
                signed_unit(row[field], f"identification {row['control']}/{field}")
        for field in ("one_sided_p", "holm_adjusted_p"):
            unit(row[field], f"identification {row['control']}/{field}")
        require(float(row["ci95_lower"]) <= float(row["ci95_upper"]) and float(row["specificity_ci95_lower"]) <= float(row["specificity_ci95_upper"]), f"identification {row['control']} CI is reversed")
        spec = binary(row["specificity_noninferior"], "identification specificity decision")
        require(spec == int(float(row["specificity_ci95_lower"]) > -0.10) and math.isclose(float(row["specificity_margin"]), -0.10), "identification specificity NI rule changed")
        novelty = binary(row["novelty_condition_pass"], "identification novelty decision")
        expected_novelty = int(
            float(row["point_estimate"]) > 0
            and float(row["ci95_lower"]) > 0
            and float(row["holm_adjusted_p"]) < 0.05
            and float(row["implicit_two_mechanism_point_estimate"]) > 0
            and spec == 1
        )
        require(novelty == expected_novelty, f"identification {row['control']} novelty gate changed")

    role = tables["role_conditioned_evidence"]
    require([r["subset"] for r in role] == list(ROLE_SUBSETS), "role-conditioned order/inventory changed")
    expected_counts = {"implicit_footprint": 12, "held_out_source": 16}
    for row in role:
        counts = row["per_mechanism_n"]
        require(isinstance(counts, dict) and set(counts) == set(MECHANISMS) and set(counts.values()) == {expected_counts[row["subset"]]}, f"{row['subset']}: per-mechanism counts changed")
        require(integer(row["bootstrap_replicates"], "role bootstrap replicates") == BOOTSTRAP_REPLICATES and integer(row["bootstrap_seed"], "role bootstrap seed") == HEADLINE_SEED, "role-conditioned bootstrap contract changed")
        for field in ("point_estimate", "ci95_lower", "ci95_upper"):
            signed_unit(row[field], f"role {row['subset']}/{field}")
        unit(row["one_sided_p"], f"role {row['subset']}/one_sided_p")
        require(float(row["ci95_lower"]) <= float(row["ci95_upper"]), f"role {row['subset']} CI is reversed")
        positive = binary(row["registered_positive"], f"role {row['subset']} registered positive")
        require(positive == int(float(row["point_estimate"]) > 0), f"role {row['subset']} positive-point rule changed")

    m6 = tables["m6_pairs"]
    require(Counter((r["stream"], r["mechanism"]) for r in m6) == Counter((s, m) for s in STREAMS for m in MECHANISMS for _ in range(6)), "M6 inventory changed")
    require(len({(r["stream"], r["m6_pair_id"]) for r in m6}) == 336, "M6 rows repeat")
    for row in m6:
        unit(row["causal_ces"], "M6 causal CES")
        unit(row["specificity_utility"], "M6 SU")
        unit(row["pair_mean"], "M6 pair mean")
        require(math.isclose(float(row["pair_mean"]), (float(row["causal_ces"]) + float(row["specificity_utility"])) / 2, abs_tol=1e-12), "M6 pair mean changed")

    secondary = tables["secondary_metrics"]
    require(Counter((r["stream"], r["mechanism"]) for r in secondary) == Counter((s, m) for s in STREAMS for m in MECHANISMS), "secondary inventory changed")
    projection_fields = TABLE_SCHEMAS["secondary_metrics"]
    mech_by_pair = {(r["stream"], r["mechanism"]): r for r in mechanism}
    for row in secondary:
        source = mech_by_pair[(row["stream"], row["mechanism"])]
        require(all(row[field] == source[field] for field in projection_fields), "secondary table is not an exact mechanism projection")


def validate_per_case(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = Counter({**{("main", stream): 294 for stream in STREAMS}, **{("identification", stream): 48 for stream in IDENTIFICATION_CONTROLS}})
    require(Counter((r["evaluation_partition"], r["stream"]) for r in rows) == expected, "per-case stream inventory differs from 8x294+2x48")
    require(len({(r["evaluation_partition"], r["stream"], r["case_id"]) for r in rows}) == 2448, "per-case identities repeat")
    for index, row in enumerate(rows):
        for field in ("anonymous_review_id", "evaluation_partition", "stream", "backbone_family", "case_id", "mechanism", "case_kind", "m6_pair_id", "footprint_lexicalization", "source_membership", "generalization_group"):
            safe_text(row[field], f"per-case row {index}/{field}")
        require(row["evaluation_partition"] in ("main", "identification"), f"per-case row {index} partition changed")
        expected_streams = STREAMS if row["evaluation_partition"] == "main" else IDENTIFICATION_CONTROLS
        require(row["stream"] in expected_streams, f"per-case row {index} stream changed")
        expected_backbone = "cogvideox" if row["stream"] in COG_STREAMS else "wan"
        require(row["backbone_family"] == expected_backbone, f"per-case row {index} backbone differs from its registered stream")
        require(row["mechanism"] in MECHANISMS and row["case_kind"] in ("causal", "specificity"), f"per-case row {index} identity changed")
        integer(row["seed"], f"per-case row {index}/seed", minimum=0)
        binary(row["usable"], f"per-case row {index}/usable")
        unit(row["receiver_preservation_norm"], f"per-case row {index}/receiver norm")
        unit(row["video_quality_norm"], f"per-case row {index}/quality norm")
        for field in ("receiver_preservation", "video_quality"):
            require(number(row[field], field) in (0, 1, 2), f"per-case row {index}/{field} outside 0/1/2")
        if row["case_kind"] == "causal":
            for field in ("original_eligible", "source_absent", "footprint_absent", "strict_success"):
                binary(row[field], f"per-case row {index}/{field}")
            unit(row["ces"], f"per-case row {index}/CES")
            require(row["su"] == "" and row["protected_object_visibility"] == "" and row["noncausal_role_adherence"] == "", f"per-case causal row {index} has specificity values")
            for field in ("source_visibility", "footprint_visibility"):
                require(number(row[field], field) in (0, 1, 2), f"per-case row {index}/{field} outside 0/1/2")
            require(int(row["source_absent"]) == int(row["source_visibility"] == 0), f"per-case row {index} source-absence derivation changed")
            require(int(row["footprint_absent"]) == int(row["footprint_visibility"] == 0), f"per-case row {index} footprint-absence derivation changed")
        else:
            unit(row["su"], f"per-case row {index}/SU")
            require(all(row[field] == "" for field in ("original_eligible", "ces", "source_absent", "footprint_absent", "strict_success", "source_visibility", "footprint_visibility")), f"per-case specificity row {index} has causal values")
            for field in ("protected_object_visibility", "noncausal_role_adherence"):
                require(number(row[field], field) in (0, 1, 2), f"per-case row {index}/{field} outside 0/1/2")


def validate_shared_sample_sizes(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    shared: Sequence[Mapping[str, str]],
) -> None:
    shared_counts = Counter(
        row["mechanism"] for row in shared if row["shared_eligible"] == "1"
    )
    rows = tables["headline_mechanism_contrasts"]
    for row in rows:
        expected_n = 24 if row["contrast"] == "matched_control" else shared_counts[row["mechanism"]]
        require(row["n"] == expected_n, f"headline {row['contrast']}/{row['mechanism']} n differs from the manifest-bound shared subset")


def validate_per_case_public_bindings(
    rows: Sequence[Mapping[str, Any]],
    formal_cases: Sequence[Mapping[str, str]],
    eligibility: Sequence[Mapping[str, str]],
) -> None:
    """Bind de-anonymized case metrics to nonprivate case/eligibility data."""

    case_by_id = {row["case_id"]: row for row in formal_cases}
    for index, row in enumerate(rows):
        case = case_by_id.get(row["case_id"])
        require(case is not None, f"per-case row {index} is absent from formal cases")
        for field in (
            "mechanism",
            "case_kind",
            "m6_pair_id",
            "footprint_lexicalization",
            "source_membership",
            "generalization_group",
        ):
            require(row[field] == case[field], f"per-case row {index}/{field} differs from formal cases")
        require(row["seed"] == int(case["seed"]), f"per-case row {index}/seed differs from formal cases")

    per_case_by_id = {row["anonymous_review_id"]: row for row in rows}
    require(len(per_case_by_id) == len(rows), "per-case anonymous review IDs repeat")
    for index, eligibility_row in enumerate(eligibility):
        row = per_case_by_id.get(eligibility_row["anonymous_review_id"])
        require(row is not None, f"Original eligibility row {index} is absent from per-case metrics")
        expected_stream = "wan_original" if eligibility_row["backbone_family"] == "wan" else "cogvideox_original"
        require(
            row["evaluation_partition"] == "main"
            and row["stream"] == expected_stream
            and row["case_id"] == eligibility_row["case_id"]
            and row["mechanism"] == eligibility_row["mechanism"]
            and row["case_kind"] == "causal",
            f"Original eligibility row {index} identity differs from per-case metrics",
        )
        for field in ("original_eligible", "source_visibility", "footprint_visibility", "receiver_preservation", "video_quality"):
            eligibility_field = "eligible" if field == "original_eligible" else field
            require(
                int(row[field]) == int(eligibility_row[eligibility_field]),
                f"Original eligibility row {index}/{eligibility_field} differs from per-case metrics",
            )


def load_formal_cases(path: Path) -> list[dict[str, str]]:
    rows = load_csv_rows(path, FORMAL_CASES_SCHEMA, 294, "formal cases")
    require(Counter(r["mechanism"] for r in rows) == Counter(m for m in MECHANISMS for _ in range(42)), "formal case mechanism inventory changed")
    require(len({r["case_id"] for r in rows}) == 294, "formal case IDs repeat")
    expected_kinds = Counter({**{(m, "causal"): 24 for m in MECHANISMS}, **{(m, "specificity"): 18 for m in MECHANISMS}})
    require(Counter((r["mechanism"], r["case_kind"]) for r in rows) == expected_kinds, "formal causal/specificity balance changed")
    for index, row in enumerate(rows):
        for field, value in row.items():
            safe_text(value, f"formal case row {index}/{field}")
        require(row["protocol_version"] == provenance_v2.PROTOCOL_VERSION, f"formal case row {index} protocol changed")
        require(row["mechanism"] in MECHANISMS, f"formal case row {index} mechanism changed")
        require(row["case_kind"] in ("causal", "specificity"), f"formal case row {index} kind changed")
        require(row["source_membership"] in {item[0] for item in M6_ORDER}, f"formal case row {index} source membership changed")
        require(row["prompt_style"] in {item[1] for item in M6_ORDER}, f"formal case row {index} prompt style changed")
        for field in ("global_case_index", "mechanism_index", "historical_capability_index", "seed", "seed_nonce", "num_frames", "fps", "height", "width"):
            try:
                parsed = int(row[field])
            except ValueError as exc:
                raise FinalTableError(f"formal case row {index}/{field} is not an integer") from exc
            require(str(parsed) == row[field], f"formal case row {index}/{field} is not canonical integer text")
        if row["case_kind"] == "causal":
            try:
                semantic_replicate = int(row["semantic_replicate"])
            except ValueError as exc:
                raise FinalTableError(
                    f"formal case row {index}/semantic_replicate is not an integer for a causal case"
                ) from exc
            require(
                str(semantic_replicate) == row["semantic_replicate"]
                and semantic_replicate in (0, 1),
                f"formal case row {index}/semantic_replicate must be canonical 0/1 for a causal case",
            )
        else:
            require(
                row["semantic_replicate"] == "",
                f"formal case row {index}/semantic_replicate must be blank for a specificity case",
            )
        require(row["specificity_subtype"] in (("",) if row["case_kind"] == "causal" else SPECIFICITY_SUBTYPES), f"formal case row {index} specificity subtype changed")
    return rows


def load_eligibility(path: Path) -> list[dict[str, str]]:
    rows = load_csv_rows(path, ELIGIBILITY_SCHEMA, 336, "Original eligibility")
    require(Counter((r["backbone_family"], r["mechanism"]) for r in rows) == Counter((b, m) for b in ("wan", "cogvideox") for m in MECHANISMS for _ in range(24)), "Original eligibility inventory changed")
    require(len({(r["backbone_family"], r["case_id"]) for r in rows}) == 336, "Original eligibility rows repeat")
    for index, row in enumerate(rows):
        for field, value in row.items():
            safe_text(value, f"Original eligibility row {index}/{field}")
        require(row["backbone_family"] in ("wan", "cogvideox"), f"Original eligibility row {index} backbone changed")
        require(row["mechanism"] in MECHANISMS, f"Original eligibility row {index} mechanism changed")
        require(row["eligible"] in ("0", "1"), f"Original eligibility row {index} is not binary")
        require(all(row[field] in ("0", "1", "2") for field in ("source_visibility", "footprint_visibility", "receiver_preservation", "video_quality")), f"Original eligibility row {index} raw score is outside 0/1/2")
    return rows


def load_shared(path: Path, eligibility: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = load_csv_rows(path, SHARED_SCHEMA, 168, "shared capability")
    require(Counter(r["mechanism"] for r in rows) == Counter(m for m in MECHANISMS for _ in range(24)), "shared capability inventory changed")
    require(len({r["case_id"] for r in rows}) == 168, "shared capability case IDs repeat")
    original = {(r["backbone_family"], r["case_id"]): int(r["eligible"]) for r in eligibility}
    for index, row in enumerate(rows):
        for field, value in row.items():
            safe_text(value, f"shared capability row {index}/{field}")
        require(row["mechanism"] in MECHANISMS, f"shared capability row {index} mechanism changed")
        require(all(row[field] in ("0", "1") for field in ("wan_original_eligible", "cogvideox_original_eligible", "shared_eligible")), f"shared capability row {index} is not binary")
        wan, cog, shared = int(row["wan_original_eligible"]), int(row["cogvideox_original_eligible"]), int(row["shared_eligible"])
        require(("wan", row["case_id"]) in original and ("cogvideox", row["case_id"]) in original, f"{row['case_id']}: shared capability case is absent from Original eligibility")
        require(wan == original[("wan", row["case_id"])] and cog == original[("cogvideox", row["case_id"])] and shared == wan * cog, f"{row['case_id']}: shared capability mismatch")
    return rows


def _scientific_equal(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(observed, dict) and set(observed) == set(expected), f"{label} schema differs from registered recomputation")
        for key in expected:
            _scientific_equal(observed[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        require(isinstance(observed, list) and len(observed) == len(expected), f"{label} length differs from registered recomputation")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _scientific_equal(left, right, f"{label}/{index}")
    elif type(expected) in (int, float):
        require(type(observed) in (int, float) and not isinstance(observed, bool), f"{label} numeric type differs from registered recomputation")
        require(math.isfinite(float(observed)) and float(observed) == float(expected), f"{label} differs from registered recomputation")
    else:
        require(type(observed) is type(expected) and observed == expected, f"{label} differs from registered recomputation")


def recompute_registered_tables(
    *,
    repo_root: Path,
    pre_metric: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, str]],
    formal_cases: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, bool]]:
    metrics_path = repo_root / "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py"
    regular_file(metrics_path, "registered v1 metrics implementation")
    require(sha256_file(metrics_path) == FORMAL_METRICS_V1_SHA256, "registered v1 metrics implementation changed")
    try:
        legacy = provenance_v2.load_registered_metrics(repo_root, pre_metric)
        mechanism, macro = legacy.mechanism_and_macro_tables(per_case)
        headline, headline_mechanism, all_baselines = legacy.headline_contrasts(per_case, shared, replicates=BOOTSTRAP_REPLICATES)
        noninferiority, retention = legacy.noninferiority_table(per_case, replicates=BOOTSTRAP_REPLICATES)
        identification, identification_novelty = legacy.identification_table(per_case, replicates=BOOTSTRAP_REPLICATES)
        role, role_positive = legacy.main_role_conditioned_evidence(per_case, replicates=BOOTSTRAP_REPLICATES)
        m6 = legacy.m6_table(per_case, formal_cases)
    except Exception as exc:
        if isinstance(exc, FinalTableError):
            raise
        raise FinalTableError(f"registered v1 scientific recomputation refused: {exc}") from exc
    secondary_fields = TABLE_SCHEMAS["secondary_metrics"]
    secondary = [{field: row[field] for field in secondary_fields} for row in mechanism]
    matched_win = next(row["registered_win"] == 1 for row in headline if row["contrast"] == "matched_control")
    claims = {
        "outperforms_all_registered_baselines": bool(all_baselines),
        "retaining_allowed_by_all_primary_noninferiority_margins": bool(retention),
        "role_conditioned_novelty_supported": bool(identification_novelty and role_positive and retention and matched_win),
    }
    return {
        "per_case_metrics": per_case,
        "mechanism_metrics": mechanism,
        "macro_metrics": macro,
        "headline_contrasts": headline,
        "headline_mechanism_contrasts": headline_mechanism,
        "noninferiority": noninferiority,
        "identification": identification,
        "role_conditioned_evidence": role,
        "m6_pairs": m6,
        "secondary_metrics": secondary,
    }, claims


def validate_legacy_results(
    results: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    recomputed_claims: Mapping[str, bool],
    score: Mapping[str, Any],
) -> None:
    expected_fields = {
        "schema_version",
        "protocol",
        "status",
        "evaluation_code_registry_sha256",
        "aggregation",
        "metric_definitions",
        "bootstrap",
        "holm_families",
        "claims",
        "headline_contrasts",
        "noninferiority",
        "identification_contrasts",
        "role_conditioned_main_evidence",
        "tables",
    }
    require(set(results) == expected_fields, "legacy scientific results schema changed")
    require(
        results["schema_version"] == 1
        and results["protocol"] == LEGACY_RESULTS_PROTOCOL
        and results["status"] == LEGACY_RESULTS_STATUS,
        "legacy scientific results identity/status changed",
    )
    require(results["evaluation_code_registry_sha256"] == REGISTRY_SHA256, "legacy results registry binding changed")
    require(results["aggregation"] == "case_then_mechanism_mean_then_equal_weight_7_mechanism_macro", "legacy aggregation declaration changed")
    require(
        results["metric_definitions"]
        == {
            "original_eligibility": "source_visibility==2 and footprint_visibility>=1 and receiver_preservation>=1 and video_quality>=1",
            "usable": "receiver_preservation>=1 and video_quality>=1",
            "CES": "E_b * usable * ((2-source_visibility)+(2-footprint_visibility))/4",
            "SU": "usable * (protected_object_visibility+noncausal_role_adherence)/4",
            "external_headline": "(V4-WanOriginal)-(baseline-CogVideoXOriginal) on pre-frozen shared-capability cases",
            "fixed_denominator_policy": "Original-ineligible causal cases remain present and contribute zero CES",
        },
        "legacy metric definitions changed",
    )
    require(
        results["bootstrap"]
        == {
            "replicates": BOOTSTRAP_REPLICATES,
            "headline_pcg64_seed": HEADLINE_SEED,
            "identification_pcg64_seed": IDENTIFICATION_SEED,
            "interval": "unadjusted_percentile_2.5_97.5",
        }
        and results["holm_families"] == {"headline": 5, "identification": 2},
        "legacy inference declaration changed",
    )
    _scientific_equal(results["claims"], dict(recomputed_claims), "legacy results/claims")
    _scientific_equal(results["headline_contrasts"], list(tables["headline_contrasts"]), "legacy results/headline")
    _scientific_equal(results["noninferiority"], list(tables["noninferiority"]), "legacy results/noninferiority")
    _scientific_equal(results["identification_contrasts"], list(tables["identification"]), "legacy results/identification")
    _scientific_equal(results["role_conditioned_main_evidence"], list(tables["role_conditioned_evidence"]), "legacy results/role-conditioned")
    refs = results["tables"]
    require(isinstance(refs, dict) and set(refs) == set(TABLE_NAMES), "legacy results table-ref inventory changed")
    for name in TABLE_NAMES:
        require(isinstance(refs[name], dict) and set(refs[name]) == {"csv", "json"}, f"legacy results {name} ref pair changed")
        for kind in ("csv", "json"):
            legacy_ref = refs[name][kind]
            authoritative_ref = score["tables"][name][kind]
            require(
                isinstance(legacy_ref, dict)
                and set(legacy_ref) == {"path", "sha256", "size_bytes"}
                and legacy_ref
                == {
                    "path": authoritative_ref["path"],
                    "sha256": authoritative_ref["sha256"],
                    "size_bytes": authoritative_ref["size_bytes"],
                },
                f"legacy results {name}/{kind} ref differs from score manifest",
            )


def decimal_text(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return "--"
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise FinalTableError(f"display value is not decimal: {value!r}") from exc
    quantum = Decimal(1).scaleb(-digits)
    rounded = decimal.quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return f"{rounded:.{digits}f}"


def p_text(value: Any) -> str:
    if value in ("", None):
        return "--"
    p = float(value)
    require(0 <= p <= 1, "display p-value outside [0,1]")
    return r"$<0.001$" if p < 0.001 else decimal_text(p)


def ci_text(lower: Any, upper: Any) -> str:
    if lower in ("", None) or upper in ("", None):
        return "--"
    require(float(lower) <= float(upper), "confidence interval is reversed")
    return f"[{decimal_text(lower)}, {decimal_text(upper)}]"


def decision(value: Any) -> str:
    return "Pass" if binary(value, "decision") else "Fail"


def safe_text(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} must be text")
    forbidden = [
        char
        for char in value
        if ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F or char in ("\u2028", "\u2029")
    ]
    require(not forbidden, f"{label} contains a control or line-separator character")
    return value


def tex_escape(value: Any) -> str:
    text = safe_text(value, "TeX text")
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def comment_token(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    text = safe_text(value, label)
    require(pattern.fullmatch(text) is not None, f"{label} is unsafe for a TeX provenance comment")
    return text


def table_tex(caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[str] | str], *, wide: bool = True, size: str = r"\scriptsize") -> str:
    environment = "table*" if wide else "table"
    columns = "l" + "r" * (len(headers) - 1)
    resize = len(headers) >= 8
    lines = [
        rf"\begin{{{environment}}}[t]", r"  \centering", f"  {size}",
        r"  \setlength{\tabcolsep}{3pt}", rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
    ]
    if resize:
        lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.extend([
        rf"  \begin{{tabular}}{{{columns}}}", r"    \toprule",
        "    " + " & ".join(headers) + r" \\", r"    \midrule",
    ])
    for row in rows:
        if isinstance(row, str):
            require(row in {
                rf"\multicolumn{{{len(headers)}}}{{l}}{{\textit{{Wan backbone}}}}",
                rf"\multicolumn{{{len(headers)}}}{{l}}{{\textit{{CogVideoX backbone}}}}",
            }, "unregistered raw TeX group row")
            lines.append("    " + row + r" \\")
        else:
            require(len(row) == len(headers), "rendered table row width changed")
            lines.append("    " + " & ".join(row) + r" \\")
    lines.extend([r"    \bottomrule", r"  \end{tabular}"])
    if resize:
        lines.append(r"  }")
    lines.extend([rf"\end{{{environment}}}", ""])
    return "\n".join(lines)


def chunked_table_tex(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    chunk_size: int = 14,
) -> str:
    require(chunk_size > 0 and rows, "chunked table requires rows and positive chunk size")
    parts = []
    for index in range(0, len(rows), chunk_size):
        part_number = index // chunk_size + 1
        suffix = "" if part_number == 1 else f"-part-{part_number}"
        part_caption = caption if part_number == 1 else f"{caption} (continued)."
        parts.append(table_tex(part_caption, label + suffix, headers, rows[index:index + chunk_size]))
    return "".join(parts)


def provenance_header(
    *, score_manifest_sha: str, pre_metric_sha: str, pre_metric: Mapping[str, Any],
    exporter_sha: str, logical_rows: int, sources: Sequence[tuple[str, Mapping[str, Any]]],
    amendment_sha: str, registry_sha: str,
) -> str:
    score_manifest_sha = comment_token(score_manifest_sha, "score-manifest SHA", HEX64)
    pre_metric_sha = comment_token(pre_metric_sha, "pre-metric SHA", HEX64)
    exporter_sha = comment_token(exporter_sha, "exporter SHA", HEX64)
    amendment_sha = comment_token(amendment_sha, "amendment SHA", HEX64)
    registry_sha = comment_token(registry_sha, "registry SHA", HEX64)
    commit = comment_token(pre_metric["implementation_commit"]["commit"], "implementation commit", HEX40)
    tree = comment_token(pre_metric["implementation_commit"]["tree"], "implementation tree", HEX40)
    integer(logical_rows, "logical row count", minimum=0)
    lines = [
        "% AUTO-GENERATED HUMAN-CANONICAL TABLE; DO NOT EDIT.",
        f"% exporter_protocol: {PROTOCOL}",
        "% canonical_status: HUMAN_CANONICAL",
        f"% score_manifest: score_manifest.json sha256={score_manifest_sha}",
        f"% pre_metric_code_freeze: pre_metric_code_freeze_v2.json sha256={pre_metric_sha}",
        f"% implementation_commit: {commit}",
        f"% implementation_tree: {tree}",
        f"% evaluation_registry_sha256: {registry_sha}",
        f"% provenance_amendment_sha256: {amendment_sha}",
        f"% exporter: scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py sha256={exporter_sha}",
        f"% logical_row_count: {logical_rows}",
    ]
    for name, ref in sources:
        source_name = comment_token(name, "source-map name", SAFE_COMMENT_NAME)
        root_id = comment_token(ref["root_id"], f"{source_name} root_id", SAFE_COMMENT_ROOT)
        path = comment_token(ref["path"], f"{source_name} path", SAFE_REF_PATH)
        sha = comment_token(ref["sha256"], f"{source_name} SHA", HEX64)
        lines.append(f"% source: {source_name}={root_id}:{path} sha256={sha}")
    return "\n".join(lines) + "\n"


def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    require(bool(rows), f"cannot average empty {field}")
    return sum(float(row[field]) for row in rows) / len(rows)


def fixed_index(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], label: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {tuple(row[key] for key in keys): row for row in rows}
    require(len(result) == len(rows), f"{label} keys repeat")
    return result


def build_outputs(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    eligibility: Sequence[Mapping[str, str]],
    shared: Sequence[Mapping[str, str]],
    formal_cases: Sequence[Mapping[str, str]],
) -> tuple[dict[str, bytes], dict[str, int]]:
    mechanism = fixed_index(tables["mechanism_metrics"], ("stream", "mechanism"), "mechanism")
    macro = {row["stream"]: row for row in tables["macro_metrics"]}
    headline = {row["contrast"]: row for row in tables["headline_contrasts"]}
    headline_mech = fixed_index(tables["headline_mechanism_contrasts"], ("contrast", "mechanism"), "headline mechanism")
    ni = {row["metric"]: row for row in tables["noninferiority"]}
    identification = {row["control"]: row for row in tables["identification"]}
    role = {row["subset"]: row for row in tables["role_conditioned_evidence"]}
    per_case = tables["per_case_metrics"]
    case_meta = {row["case_id"]: row for row in formal_cases}
    per_case_index = fixed_index(per_case, ("evaluation_partition", "stream", "case_id"), "per-case")
    outputs: dict[str, bytes] = {}
    row_counts: dict[str, int] = {}

    # Main T1: values tied with the within-backbone maximum at displayed
    # precision are bolded.  No global cross-backbone maximum is defined.
    rendered = {(s, m): decimal_text(mechanism[(s, m)]["ces"]) for s in STREAMS for m in MECHANISMS}
    rendered.update({(s, "macro"): decimal_text(macro[s]["ces"]) for s in STREAMS})
    bold: set[tuple[str, str]] = set()
    for block in (WAN_STREAMS, COG_STREAMS):
        for column in (*MECHANISMS, "macro"):
            high = max(Decimal(rendered[(s, column)]) for s in block)
            bold.update((s, column) for s in block if Decimal(rendered[(s, column)]) == high)
    t1_rows: list[Sequence[str] | str] = []
    for stream in STREAMS:
        if stream == WAN_STREAMS[0]:
            t1_rows.append(r"\multicolumn{9}{l}{\textit{Wan backbone}}")
        elif stream == COG_STREAMS[0]:
            t1_rows.append(r"\multicolumn{9}{l}{\textit{CogVideoX backbone}}")
        values = []
        for column in (*MECHANISMS, "macro"):
            value = rendered[(stream, column)]
            values.append(rf"\textbf{{{value}}}" if (stream, column) in bold else value)
        t1_rows.append([DISPLAY_NAMES[stream], *values])
    outputs["main/tab_main_mechanisms.tex"] = table_tex(
        "Human-canonical CES across seven mechanisms. Bold marks the highest observed value at displayed precision within each backbone block; it is not an inferential claim.",
        "tab:main-mechanisms", ["Method", *[MECHANISM_NAMES[m] for m in MECHANISMS], "Macro"], t1_rows,
    ).encode()
    row_counts["main/tab_main_mechanisms.tex"] = 8

    # Main T2: fixed six-row Wan panel followed by the four external contrasts.
    metric_map = (
        ("CES", "ces", "matched_control"),
        ("Original capability", "original_capability_coverage", None),
        ("Specificity utility", "specificity_utility", "specificity_utility"),
        ("Receiver preservation", "causal_receiver_preservation", "receiver_preservation"),
        ("Video quality", "causal_video_quality", "video_quality"),
        ("Usable fraction", "causal_usable_fraction", "usable_fraction"),
    )
    t2a = []
    for label, macro_field, inference_key in metric_map:
        ours, matched = macro["V4"][macro_field], macro["matched_control"][macro_field]
        if inference_key is None:
            t2a.append([label, decimal_text(ours), decimal_text(matched), "--", "--", "Original eligibility", "--"])
        elif inference_key == "matched_control":
            inf = headline["matched_control"]
            t2a.append([label, decimal_text(ours), decimal_text(matched), decimal_text(inf["point_estimate"]), ci_text(inf["ci95_lower"], inf["ci95_upper"]), f"Holm p={p_text(inf['holm_adjusted_p'])}", decision(inf["registered_win"])])
        else:
            inf = ni[inference_key]
            t2a.append([label, decimal_text(ours), decimal_text(matched), decimal_text(inf["point_estimate"]), ci_text(inf["ci95_lower"], inf["ci95_upper"]), f"margin {decimal_text(inf['margin'])}", decision(inf["noninferior"])])
    t2 = table_tex(
        "Wan single-factor efficacy and registered preservation guardrails. Non-inferiority decisions use the lower confidence bound and the stated margin.",
        "tab:main-macro-guardrails", ["Outcome", r"\textsc{SRCD}", "Matched", r"$\Delta$", r"95\% CI", "Criterion", "Decision"], t2a,
    )
    external_rows = []
    for contrast in HEADLINE_CONTRASTS[1:]:
        row = headline[contrast]
        if not int(row["estimable"]):
            external_rows.append([DISPLAY_NAMES[contrast], "NE", "--", "--", "Not estimable"])
        else:
            external_rows.append([DISPLAY_NAMES[contrast], decimal_text(row["point_estimate"]), ci_text(row["ci95_lower"], row["ci95_upper"]), p_text(row["holm_adjusted_p"]), decision(row["registered_win"])])
    t2 += table_tex(
        "Original-normalized external benchmark contrasts on the frozen shared-capability subset.",
        "tab:main-external-contrasts", ["External baseline", r"Normalized $\Delta$CES", r"95\% CI", "Holm p", "Status"], external_rows,
    )
    outputs["main/tab_main_macro_guardrails.tex"] = t2.encode()
    row_counts["main/tab_main_macro_guardrails.tex"] = 10

    t3_rows = []
    for control in IDENTIFICATION_CONTROLS:
        row = identification[control]
        t3_rows.append([
            CONTROL_NAMES[control], decimal_text(row["point_estimate"]), ci_text(row["ci95_lower"], row["ci95_upper"]),
            p_text(row["holm_adjusted_p"]), decimal_text(row["implicit_two_mechanism_point_estimate"]),
            decimal_text(row["specificity_point_estimate"]), ci_text(row["specificity_ci95_lower"], row["specificity_ci95_upper"]),
            decimal_text(row["specificity_margin"]), decision(row["specificity_noninferior"]), decision(row["novelty_condition_pass"]),
        ])
    outputs["main/tab_identification_controls.tex"] = table_tex(
        "Water/Fracture identification controls. Causal and specificity contrasts compare SRCD against each control; the implicit result is a prespecified point-estimate condition.",
        "tab:identification-controls", ["Control", r"$\Delta$CES", r"95\% CI", "Holm p", r"Implicit $\Delta$", r"$\Delta$SU", r"SU 95\% CI", "Margin", "SU NI", "All conditions"], t3_rows,
    ).encode()
    row_counts["main/tab_identification_controls.tex"] = 2

    overlap_by_mechanism = Counter(
        row["mechanism"]
        for row in formal_cases
        if row["case_kind"] == "causal"
        and row["footprint_lexicalization"] == "implicit"
        and row["source_membership"] == "eval_holdout"
    )
    require(
        overlap_by_mechanism == Counter({mechanism: 8 for mechanism in MECHANISMS}),
        "formal-case implicit/held-out overlap differs from 8 cases per mechanism",
    )
    overlap_per_mechanism = next(iter(overlap_by_mechanism.values()))
    overlap_total = sum(overlap_by_mechanism.values())
    t4_rows = []
    for subset in ROLE_SUBSETS:
        row = role[subset]
        t4_rows.append([
            "Implicit footprint" if subset == "implicit_footprint" else "Held-out source",
            str(next(iter(row["per_mechanism_n"].values()))), decimal_text(row["point_estimate"]),
            ci_text(row["ci95_lower"], row["ci95_upper"]), p_text(row["one_sided_p"]), decision(row["registered_positive"]),
        ])
    outputs["main/tab_generalization.tex"] = table_tex(
        f"Prespecified role-conditioned subsets. Implicit-footprint and held-out-source subsets overlap in {overlap_per_mechanism} cases per mechanism ({overlap_total} total) and are not independent replications.",
        "tab:generalization", ["Subset", r"$n$/mechanism", r"SRCD$-$Matched $\Delta$CES", r"95\% CI", "Unadjusted p", "Positive point"], t4_rows,
    ).encode()
    row_counts["main/tab_generalization.tex"] = 2

    # E1 absolute mechanism decomposition and per-mechanism headline diagnostics.
    component_rows = []
    for stream in STREAMS:
        for mech in MECHANISMS:
            block = [r for r in per_case if r["evaluation_partition"] == "main" and r["stream"] == stream and r["mechanism"] == mech and r["case_kind"] == "causal"]
            source_component = sum(int(r["original_eligible"]) * int(r["usable"]) * (2 - int(r["source_visibility"])) / 4 for r in block) / 24
            footprint_component = sum(int(r["original_eligible"]) * int(r["usable"]) * (2 - int(r["footprint_visibility"])) / 4 for r in block) / 24
            require(math.isclose(source_component + footprint_component, float(mechanism[(stream, mech)]["ces"]), abs_tol=1e-12), "gated CES decomposition does not sum to CES")
            backbone = "cogvideox" if stream in COG_STREAMS else "wan"
            component_rows.append([BACKBONE_NAMES[backbone], DISPLAY_NAMES[stream], MECHANISM_NAMES[mech], "24/18", decimal_text(mechanism[(stream, mech)]["ces"]), decimal_text(source_component), decimal_text(footprint_component), decimal_text(mechanism[(stream, mech)]["original_capability_coverage"])])
    e1 = chunked_table_tex(
        "Per-mechanism human-canonical CES decomposition. Source and footprint components are the two fixed additive terms of CES; counts are causal/specificity.",
        "tab:e1-mechanism-decomposition", ["Backbone", "Method", "Mechanism", r"$n_c/n_s$", "CES", "Source component", "Footprint component", "Original capability"], component_rows,
    )
    diagnostic_rows = []
    for contrast in HEADLINE_CONTRASTS:
        for mech in MECHANISMS:
            row = headline_mech[(contrast, mech)]
            estimate = decimal_text(row["point_estimate"]) if int(row["estimable"]) else "NE"
            diagnostic_rows.append([DISPLAY_NAMES[contrast], MECHANISM_NAMES[mech], str(row["n"]), estimate, "Estimable" if int(row["estimable"]) else "NE"])
    e1 += chunked_table_tex(
        "Per-mechanism headline diagnostics. These point estimates have no separate registered confidence intervals or hypothesis tests.",
        "tab:e1-headline-mechanism-diagnostics", ["Comparison baseline", "Mechanism", "n", r"$\Delta$CES", "Status"], diagnostic_rows,
    )
    outputs["appendix/tab_e1_mechanism_decomposition.tex"] = e1.encode()
    row_counts["appendix/tab_e1_mechanism_decomposition.tex"] = 91

    e2_rows = []
    for stream in STREAMS:
        for mech in MECHANISMS:
            row = mechanism[(stream, mech)]
            backbone = "cogvideox" if stream in COG_STREAMS else "wan"
            e2_rows.append([BACKBONE_NAMES[backbone], DISPLAY_NAMES[stream], MECHANISM_NAMES[mech], decimal_text(row["specificity_utility"]), decimal_text(row["causal_receiver_preservation"]), decimal_text(row["causal_video_quality"]), decimal_text(row["causal_usable_fraction"]), decimal_text(row["specificity_receiver_preservation"]), decimal_text(row["specificity_video_quality"]), decimal_text(row["specificity_usable_fraction"])])
    e2 = chunked_table_tex(
        "Complete per-mechanism preservation and specificity breakdown on a 0--1 scale. Registered non-inferiority decisions are reported separately.",
        "tab:e2-guardrail-breakdown", ["Backbone", "Method", "Mechanism", "SU", r"$R_c$", r"$Q_c$", r"$U_c$", r"$R_s$", r"$Q_s$", r"$U_s$"], e2_rows,
    )
    subtype_rows = []
    for stream in STREAMS:
        for subtype in SPECIFICITY_SUBTYPES:
            per_mechanism_values = []
            for mech in MECHANISMS:
                block = [r for r in per_case if r["evaluation_partition"] == "main" and r["stream"] == stream and r["mechanism"] == mech and r["case_kind"] == "specificity" and case_meta[r["case_id"]]["specificity_subtype"] == subtype]
                require(len(block) == 6, f"{stream}/{mech}/{subtype}: expected six specificity rows")
                per_mechanism_values.append((mean(block, "su"), mean(block, "receiver_preservation_norm"), mean(block, "video_quality_norm"), mean(block, "usable")))
            backbone = "cogvideox" if stream in COG_STREAMS else "wan"
            subtype_rows.append([BACKBONE_NAMES[backbone], DISPLAY_NAMES[stream], tex_escape(subtype.replace("_", " ").title()), "6", *[decimal_text(sum(values[index] for values in per_mechanism_values) / 7) for index in range(4)]])
    e2 += chunked_table_tex(
        "Specificity-subtype macro breakdown, averaging six cases within each mechanism and then weighting mechanisms equally.",
        "tab:e2-specificity-subtypes", ["Backbone", "Method", "Subtype", r"$n$/mechanism", "SU", "Receiver", "Quality", "Usable"], subtype_rows, chunk_size=12,
    )
    outputs["appendix/tab_e2_guardrail_breakdown.tex"] = e2.encode()
    row_counts["appendix/tab_e2_guardrail_breakdown.tex"] = 80

    eligible_by = {(r["backbone_family"], r["mechanism"]): 0 for r in eligibility}
    for row in eligibility:
        eligible_by[(row["backbone_family"], row["mechanism"])] += int(row["eligible"])
    shared_by = Counter(row["mechanism"] for row in shared if row["shared_eligible"] == "1")
    e3_rows = []
    for row_id, name in (("wan", "Wan Original"), ("cogvideox", "CogVideoX Original"), ("shared", "Shared (both Originals)")):
        counts = [eligible_by[(row_id, m)] if row_id != "shared" else shared_by[m] for m in MECHANISMS]
        e3_rows.append([name, *[f"{value}/24" for value in counts], f"{sum(counts)}/168"])
    outputs["appendix/tab_e3_capability.tex"] = table_tex(
        "Original eligibility and shared-capability counts. External contrasts require the same semantic case to be eligible under both Originals.",
        "tab:e3-capability", ["Capability set", *[MECHANISM_NAMES[m] for m in MECHANISMS], "All"], e3_rows,
    ).encode()
    row_counts["appendix/tab_e3_capability.tex"] = 3

    e4_rows = []
    for contrast in HEADLINE_CONTRASTS:
        row = headline[contrast]
        if int(row["estimable"]):
            e4_rows.append([DISPLAY_NAMES[contrast], decimal_text(row["point_estimate"]), ci_text(row["ci95_lower"], row["ci95_upper"]), p_text(row["one_sided_p"]), p_text(row["holm_adjusted_p"]), decision(row["registered_win"])])
        else:
            e4_rows.append([DISPLAY_NAMES[contrast], "NE", "--", "--", "--", "Not estimable"])
    outputs["appendix/tab_e4_headline_contrasts.tex"] = table_tex(
        "Complete registered headline CES contrasts. Matched is a Wan single-factor comparison; external rows are Original-normalized shared-capability comparisons.",
        "tab:e4-headline-contrasts", ["Baseline", r"$\Delta$CES", r"95\% CI", "Raw p", "Holm p", "Decision"], e4_rows,
    ).encode()
    row_counts["appendix/tab_e4_headline_contrasts.tex"] = 5

    e5_rows = [[row["metric"].replace("_", " ").title(), row["role"].title(), decimal_text(row["point_estimate"]), ci_text(row["ci95_lower"], row["ci95_upper"]), decimal_text(row["margin"]), decision(row["noninferior"])] for row in (ni[m] for m in NI_METRICS)]
    outputs["appendix/tab_e5_noninferiority.tex"] = table_tex(
        "Registered non-inferiority tests for SRCD minus Matched. Decisions compare the lower confidence bound with the margin; superiority-to-zero p-values are intentionally not shown.",
        "tab:e5-noninferiority", ["Metric", "Role", r"$\Delta$", r"95\% CI", "Margin", "Decision"], e5_rows,
    ).encode()
    row_counts["appendix/tab_e5_noninferiority.tex"] = 7

    secondary = fixed_index(tables["secondary_metrics"], ("stream", "mechanism"), "secondary")
    e6_rows = []
    for stream in STREAMS:
        for mech in MECHANISMS:
            row = secondary[(stream, mech)]
            backbone = "cogvideox" if stream in COG_STREAMS else "wan"
            e6_rows.append([BACKBONE_NAMES[backbone], DISPLAY_NAMES[stream], MECHANISM_NAMES[mech], decimal_text(row["source_absent_rate"]), decimal_text(row["footprint_absent_rate"]), decimal_text(row["strict_success_rate"]), decimal_text(row["eligible_source_clear_to_partial_rate"]), decimal_text(row["eligible_source_clear_to_absent_rate"])])
    outputs["appendix/tab_e6_secondary_outcomes.tex"] = chunked_table_tex(
        "Descriptive secondary causal outcomes. Complete absence and strict success are not the primary task definition.",
        "tab:e6-secondary-outcomes", ["Backbone", "Method", "Mechanism", "Source absent", "Footprint absent", "Strict success", r"Clear$\to$partial", r"Clear$\to$absent"], e6_rows,
    ).encode()
    row_counts["appendix/tab_e6_secondary_outcomes.tex"] = 56

    m6_by = fixed_index(tables["m6_pairs"], ("stream", "m6_pair_id"), "M6")
    m6_meta: dict[tuple[str, str, str], str] = {}
    for row in formal_cases:
        if row["m6_pair_id"]:
            m6_meta[(row["mechanism"], row["source_membership"], row["prompt_style"])] = row["m6_pair_id"]
    e7_rows = []
    fig1_rows = []
    for mech in MECHANISMS:
        for membership, style in M6_ORDER:
            pair_id = m6_meta[(mech, membership, style)]
            matched, ours = m6_by[("matched_control", pair_id)], m6_by[("V4", pair_id)]
            label = tex_escape(f"{membership}/{style}")
            e7_rows.append([MECHANISM_NAMES[mech], label, decimal_text(matched["causal_ces"]), decimal_text(ours["causal_ces"]), decimal_text(matched["specificity_utility"]), decimal_text(ours["specificity_utility"])])
            fig1_rows.append({"mechanism": mech, "pair_order": len(fig1_rows), "source_membership": membership, "prompt_style": style, "matched_causal_ces": matched["causal_ces"], "srcd_causal_ces": ours["causal_ces"], "matched_specificity_utility": matched["specificity_utility"], "srcd_specificity_utility": ours["specificity_utility"]})
    outputs["appendix/tab_e7_m6_pairs.tex"] = chunked_table_tex(
        "All 42 registered M6 causal--specificity pairs for the focal Wan comparison.",
        "tab:e7-m6-pairs", ["Mechanism", "Pair", "Matched CES", "SRCD CES", "Matched SU", "SRCD SU"], e7_rows,
    ).encode()
    row_counts["appendix/tab_e7_m6_pairs.tex"] = 42

    f1_rows = [[CONTROL_NAMES[c], decimal_text(identification[c]["point_estimate"]), ci_text(identification[c]["ci95_lower"], identification[c]["ci95_upper"]), p_text(identification[c]["one_sided_p"]), p_text(identification[c]["holm_adjusted_p"]), decimal_text(identification[c]["implicit_two_mechanism_point_estimate"]), decimal_text(identification[c]["implicit_water_point_estimate"]), decimal_text(identification[c]["implicit_fracture_point_estimate"]), decision(identification[c]["novelty_condition_pass"])] for c in IDENTIFICATION_CONTROLS]
    outputs["appendix/tab_f1_identification_causal.tex"] = table_tex(
        "Complete causal and implicit-footprint identification results on Water and Fracture.",
        "tab:f1-identification-causal", ["Control", r"$\Delta$CES", r"95\% CI", "Raw p", "Holm p", "Implicit macro", "Water", "Fracture", "All conditions"], f1_rows,
    ).encode()
    row_counts["appendix/tab_f1_identification_causal.tex"] = 2

    f2_rows = [[CONTROL_NAMES[c], decimal_text(identification[c]["specificity_point_estimate"]), ci_text(identification[c]["specificity_ci95_lower"], identification[c]["specificity_ci95_upper"]), decimal_text(identification[c]["specificity_margin"]), decision(identification[c]["specificity_noninferior"])] for c in IDENTIFICATION_CONTROLS]
    outputs["appendix/tab_f2_identification_specificity.tex"] = table_tex(
        "Specificity component of the two identification-control gates.",
        "tab:f2-identification-specificity", ["Control", r"$\Delta$SU", r"95\% CI", "Margin", "Decision"], f2_rows,
    ).encode()
    row_counts["appendix/tab_f2_identification_specificity.tex"] = 2

    group_defs = (
        ("explicit", lambda r: r["footprint_lexicalization"] == "explicit", 12),
        ("implicit", lambda r: r["footprint_lexicalization"] == "implicit", 12),
        ("seen source", lambda r: r["source_membership"] != "eval_holdout", 8),
        ("held-out source", lambda r: r["source_membership"] == "eval_holdout", 16),
    )
    f3_rows = []
    overlap = 0
    for label, predicate, expected_n in group_defs:
        for mech in MECHANISMS:
            ours = [r for r in per_case if r["evaluation_partition"] == "main" and r["stream"] == "V4" and r["mechanism"] == mech and r["case_kind"] == "causal" and predicate(r)]
            matched = [per_case_index[("main", "matched_control", r["case_id"])] for r in ours]
            require(len(ours) == expected_n, f"{label}/{mech}: subset count changed")
            f3_rows.append([label.title(), MECHANISM_NAMES[mech], str(expected_n), decimal_text(mean(ours, "ces")), decimal_text(mean(matched, "ces")), decimal_text(mean(ours, "ces") - mean(matched, "ces"))])
    overlap = sum(r["stream"] == "V4" and r["evaluation_partition"] == "main" and r["case_kind"] == "causal" and r["footprint_lexicalization"] == "implicit" and r["source_membership"] == "eval_holdout" for r in per_case)
    require(overlap == 56, "implicit/held-out overlap changed")
    f3 = chunked_table_tex(
        "Descriptive role-conditioned breakdown. The implicit and held-out subsets overlap in 8 cases per mechanism (56 total).",
        "tab:f3-role-conditioned", ["Subset", "Mechanism", "n", "SRCD CES", "Matched CES", r"$\Delta$CES"], f3_rows,
    )
    macro_role_rows = [["Implicit footprint" if s == "implicit_footprint" else "Held-out source", str(next(iter(role[s]["per_mechanism_n"].values()))), decimal_text(role[s]["point_estimate"]), ci_text(role[s]["ci95_lower"], role[s]["ci95_upper"]), p_text(role[s]["one_sided_p"])] for s in ROLE_SUBSETS]
    f3 += table_tex(
        "Registered equal-weight macro role-conditioned contrasts.",
        "tab:f3-role-conditioned-macro", ["Subset", r"$n$/mechanism", r"$\Delta$CES", r"95\% CI", "Unadjusted p"], macro_role_rows,
    )
    outputs["appendix/tab_f3_role_conditioned.tex"] = f3.encode()
    row_counts["appendix/tab_f3_role_conditioned.tex"] = 30

    fig2_rows = []
    for mech in MECHANISMS:
        row = headline_mech[("matched_control", mech)]
        fig2_rows.append({"mechanism": mech, "mechanism_order": MECHANISMS.index(mech), "n": row["n"], "estimable": row["estimable"], "point_estimate": row["point_estimate"], "ci95_lower": "", "ci95_upper": "", "inference_scope": "DESCRIPTIVE_POINT_ONLY_NO_REGISTERED_PER_MECHANISM_INTERVAL"})
    outputs["figure_data/fig_f1_m6_pairs.csv"] = csv_bytes(fig1_rows, tuple(fig1_rows[0]))
    outputs["figure_data/fig_f2_mechanism_contrasts.csv"] = csv_bytes(fig2_rows, tuple(fig2_rows[0]))
    row_counts["figure_data/fig_f1_m6_pairs.csv"] = 42
    row_counts["figure_data/fig_f2_mechanism_contrasts.csv"] = 7
    return outputs, row_counts


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: canonical_scalar(row[field]) for field in fields})
    return output.getvalue().encode("utf-8")


def canonical_scalar(value: Any) -> Any:
    if isinstance(value, float):
        require(math.isfinite(value), "figure-data value is not finite")
        return format(value, ".17g")
    return value


def export_tables(
    *, repo_root: Path, review_root: Path, eligibility_root: Path, score_manifest_path: Path,
    pre_metric_path: Path, amendment_path: Path, output_root: Path,
) -> dict[str, Any]:
    roots = {
        "repo_root": repo_root.resolve(),
        "review_root": review_root.resolve(),
        "eligibility_root": eligibility_root.resolve(),
        "metrics_root": score_manifest_path.resolve().parent,
        "pre_metric_root": pre_metric_path.resolve().parent,
        "amendment_root": amendment_path.resolve().parent,
    }
    for name, root in roots.items():
        require(root.is_dir() and not root.is_symlink(), f"{name} is not a regular directory")
    output_root = output_root.resolve()
    required_output_root = (roots["repo_root"] / OUTPUT_RELATIVE).resolve()
    require(
        output_root == required_output_root,
        f"output root must be the versioned leaf {OUTPUT_RELATIVE.as_posix()}",
    )
    expected_exporter = roots["repo_root"] / "scripts" / "export_causal_role_erasure_7mechanism_paper_tables_v1.py"
    require(Path(__file__).resolve() == expected_exporter.resolve(), "exporter must run from its frozen repo-relative path")

    pre_metric, pre_metric_raw = validate_pre_metric_freeze(pre_metric_path.resolve(), expected_exporter)
    score, score_raw, paths = validate_score_manifest(
        score_manifest_path.resolve(), roots, pre_metric_path.resolve(), pre_metric,
        amendment_path.resolve(),
    )
    authority_seals = collect_authority_snapshot(
        roots=roots,
        exporter_path=expected_exporter,
        score_path=score_manifest_path.resolve(),
        score=score,
        pre_metric_path=pre_metric_path.resolve(),
        pre_metric=pre_metric,
        amendment_path=amendment_path.resolve(),
    )

    # Validate the complete public v2 authority envelope before opening any
    # aggregate metric.  Tier-2 full-key and canonical-score refs remain sealed
    # declarations and are never resolved by this presentation stage.
    validate_v2_authority_envelope(
        repo_root=roots["repo_root"],
        review_root=roots["review_root"],
        amendment_path=amendment_path.resolve(),
        pre_metric_path=pre_metric_path.resolve(),
        key_commitments_path=paths["key_commitments"],
        expected_pre_metric=pre_metric,
        expected_registry_path=validate_ref(
            score["evaluation_code_registry"], roots, "score-manifest public evaluation registry", extra={"registry_sha256"}
        ),
    )
    verify_snapshot(authority_seals)

    tables: dict[str, list[dict[str, Any]]] = {}
    for name in TABLE_NAMES:
        json_rows = load_json_rows(paths[f"{name}_json"], name)
        csv_rows = load_csv_rows(paths[f"{name}_csv"], CSV_SCHEMAS[name], EXPECTED_ROWS[name], name)
        csv_matches_json(csv_rows, json_rows, name)
        tables[name] = json_rows
    validate_inventories(tables)
    validate_per_case(tables["per_case_metrics"])
    formal_cases = load_formal_cases(paths["formal_cases"])
    eligibility = load_eligibility(paths["original_eligibility"])
    shared = load_shared(paths["shared_capability_subsets"], eligibility)
    validate_per_case_public_bindings(tables["per_case_metrics"], formal_cases, eligibility)
    recomputed_tables, recomputed_claims = recompute_registered_tables(
        repo_root=roots["repo_root"],
        pre_metric=pre_metric,
        per_case=tables["per_case_metrics"],
        shared=shared,
        formal_cases=formal_cases,
    )
    for name in TABLE_NAMES:
        _scientific_equal(tables[name], recomputed_tables[name], f"{name} complete recomputation")
    validate_shared_sample_sizes(tables, shared)
    legacy_results, _ = load_json(paths["results"], "legacy scientific results")
    validate_legacy_results(legacy_results, tables, recomputed_claims, score)

    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    exporter_bytes = expected_exporter.read_bytes()
    exporter_sha = sha256_bytes(exporter_bytes)
    outputs, row_counts = build_outputs(tables, eligibility, shared, formal_cases)
    source_refs = score["tables"]

    def table_sources(*names: str) -> list[tuple[str, Mapping[str, Any]]]:
        return [
            (f"{name}.{kind}", source_refs[name][kind])
            for name in names
            for kind in ("csv", "json")
        ]

    def input_sources(*names: str) -> list[tuple[str, Mapping[str, Any]]]:
        return [(name, score["inputs"][name]) for name in names]

    # Direct, per-output dependency map.  Authority manifests are recorded in
    # dedicated header/manifest fields; this map names only the metric payloads
    # and nonmetric scientific inputs actually read for each projection.
    source_map: dict[str, list[tuple[str, Mapping[str, Any]]]] = {
        "main/tab_main_mechanisms.tex": table_sources("mechanism_metrics", "macro_metrics"),
        "main/tab_main_macro_guardrails.tex": table_sources("macro_metrics", "headline_contrasts", "noninferiority"),
        "main/tab_identification_controls.tex": table_sources("identification"),
        "main/tab_generalization.tex": table_sources("role_conditioned_evidence") + input_sources("formal_cases"),
        "appendix/tab_e1_mechanism_decomposition.tex": table_sources("mechanism_metrics", "headline_mechanism_contrasts", "per_case_metrics"),
        "appendix/tab_e2_guardrail_breakdown.tex": table_sources("mechanism_metrics", "per_case_metrics") + input_sources("formal_cases"),
        "appendix/tab_e3_capability.tex": input_sources("original_eligibility", "shared_capability_subsets"),
        "appendix/tab_e4_headline_contrasts.tex": table_sources("headline_contrasts"),
        "appendix/tab_e5_noninferiority.tex": table_sources("noninferiority"),
        "appendix/tab_e6_secondary_outcomes.tex": table_sources("secondary_metrics"),
        "appendix/tab_e7_m6_pairs.tex": table_sources("m6_pairs") + input_sources("formal_cases"),
        "appendix/tab_f1_identification_causal.tex": table_sources("identification"),
        "appendix/tab_f2_identification_specificity.tex": table_sources("identification"),
        "appendix/tab_f3_role_conditioned.tex": table_sources("per_case_metrics", "role_conditioned_evidence"),
        "figure_data/fig_f1_m6_pairs.csv": table_sources("m6_pairs") + input_sources("formal_cases"),
        "figure_data/fig_f2_mechanism_contrasts.csv": table_sources("headline_mechanism_contrasts"),
    }
    require(set(source_map) == set(outputs), "internal per-output source map is incomplete")
    amendment, _ = load_json(amendment_path.resolve(), "provenance amendment for output binding")
    amendment_sha = amendment["amendment_sha256"]
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        artifact_refs: dict[str, dict[str, Any]] = {}
        for relative, body in sorted(outputs.items()):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative.endswith(".tex"):
                body = provenance_header(
                    score_manifest_sha=sha256_bytes(score_raw),
                    pre_metric_sha=sha256_bytes(pre_metric_raw),
                    pre_metric=pre_metric,
                    exporter_sha=exporter_sha,
                    logical_rows=row_counts[relative],
                    sources=source_map[relative],
                    amendment_sha=amendment_sha,
                    registry_sha=REGISTRY_SHA256,
                ).encode() + body
            with path.open("xb") as handle:
                handle.write(body)
            path.chmod(0o444)
            artifact_refs[relative] = {
                "root_id": "paper_table_root",
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "logical_row_count": row_counts[relative],
                "sources": [
                    {"name": name, **dict(ref)} for name, ref in source_map[relative]
                ],
            }
        snapshot = staging / "implementation_snapshot.py"
        snapshot.write_bytes(exporter_bytes)
        snapshot.chmod(0o444)
        artifact_refs["implementation_snapshot.py"] = {
            "root_id": "paper_table_root",
            "path": "implementation_snapshot.py",
            "sha256": exporter_sha,
            "size_bytes": len(exporter_bytes),
            "sources": [
                {
                    "name": "final_table_exporter",
                    "root_id": "repo_root",
                    "path": "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
                    "sha256": exporter_sha,
                    "size_bytes": len(exporter_bytes),
                }
            ],
        }
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": STATUS,
            "canonical_status": "HUMAN_CANONICAL",
            "manifest_location": {
                "root_id": "paper_table_root",
                "path": "paper_tables_manifest.json",
            },
            "score_manifest": {
                "root_id": "metrics_root",
                "path": score_manifest_path.resolve().relative_to(roots["metrics_root"]).as_posix(),
                "sha256": sha256_bytes(score_raw),
                "size_bytes": len(score_raw),
                "manifest_sha256": score["manifest_sha256"],
            },
            "pre_metric_code_freeze": {
                "root_id": "pre_metric_root",
                "path": pre_metric_path.resolve().relative_to(roots["pre_metric_root"]).as_posix(),
                "sha256": sha256_bytes(pre_metric_raw),
                "size_bytes": len(pre_metric_raw),
                "manifest_sha256": pre_metric["manifest_sha256"],
            },
            "evaluation_code_registry": dict(score["evaluation_code_registry"]),
            "provenance_amendment": dict(score["provenance_amendment"]),
            "exporter": {
                "root_id": "repo_root",
                "path": "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py",
                "sha256": exporter_sha,
                "size_bytes": len(exporter_bytes),
            },
            "implementation_commit": pre_metric["implementation_commit"],
            "format": {"metric_digits": 3, "p_digits": 3, "p_lt_threshold": 0.001, "missing": "--", "not_estimable": "NE", "emphasis": "within_backbone_display_precision_maxima_only", "backbone_partition": ["wan", "cogvideox"]},
            "artifacts": artifact_refs,
        }
        manifest["manifest_sha256"] = self_digest(manifest)
        manifest_path = staging / "paper_tables_manifest.json"
        manifest_path.write_bytes(pretty_json_bytes(manifest))
        manifest_path.chmod(0o444)

        # Close all read/build/publish TOCTOU windows.
        verify_snapshot(authority_seals)
        require(expected_exporter.read_bytes() == exporter_bytes, "exporter changed during build")
        require(score_manifest_path.read_bytes() == score_raw, "score manifest changed during build")
        require(pre_metric_path.read_bytes() == pre_metric_raw, "pre-metric freeze changed during build")
        for label, path in paths.items():
            ref = None
            if label in score["inputs"]:
                ref = score["inputs"][label]
            elif label == "results":
                ref = score["results"]
            elif label.endswith("_csv") or label.endswith("_json"):
                name, suffix = label.rsplit("_", 1)
                ref = score["tables"][name][suffix]
            if ref is not None:
                require(path.stat().st_size == ref["size_bytes"] and sha256_file(path) == ref["sha256"], f"{label} changed during export")
        require(not output_root.exists(), "output root appeared during build")
        os.replace(staging, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--eligibility-root", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--pre-metric-freeze", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = export_tables(
            repo_root=args.repo_root,
            review_root=args.review_root,
            eligibility_root=args.eligibility_root,
            score_manifest_path=args.score_manifest,
            pre_metric_path=args.pre_metric_freeze,
            amendment_path=args.amendment,
            output_root=args.output_root,
        )
    except FinalTableError as exc:
        raise SystemExit(f"final table export refused: {exc}") from exc
    print(json.dumps({"status": manifest["status"], "artifact_count": len(manifest["artifacts"]), "canonical_status": manifest["canonical_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
