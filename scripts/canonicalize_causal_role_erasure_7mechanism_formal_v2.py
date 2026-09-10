#!/usr/bin/env python3
"""Deviation-aware wrappers around the frozen v1 canonicalization stages.

The registered v1 implementation remains the sole scientific implementation.
This module validates the frozen authority chain, runs the v1 stage in a
private temporary compatibility root, copies scientific payloads byte for
byte, discards the legacy manifest, and publishes a truthful v2 manifest.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
AMENDMENT_PROTOCOL = "causal_role_erasure_7m_evaluation_provenance_amendment_v2"
AMENDMENT_STATUS = (
    "frozen_after_disclosed_global_full_key_access_before_human_labeling"
)
PRE_METRIC_PROTOCOL = "causal_role_erasure_7m_pre_metric_code_freeze_v2"
PRE_METRIC_STATUS = "clean_tree_components_frozen_before_human_canonical_metrics"
PRE_METRIC_DEVIATION = {
    "global_full_key_opened_before_canonical_freeze": True,
    "human_review_process_local_answer_key_blindness_preserved": True,
    "v1_files_byte_identical": True,
    "scientific_rules_changed": False,
    "unchanged_v1_final_stages_forbidden": True,
}
CANONICAL_PROTOCOL = "causal_role_erasure_7m_human_canonicalization_v2"
CANONICAL_STATUS = (
    "canonical_anonymous_scores_frozen_after_disclosed_global_full_key_access"
)
ELIGIBILITY_STATUS = (
    "original_eligibility_and_shared_subsets_frozen_after_disclosed_global_full_key_access"
)
REVIEW_PROCESS_PROTOCOL = "causal_role_erasure_7m_human_review_process_receipt_v2"
REVIEW_PROCESS_STATUS = (
    "independent_public_only_reviews_and_complete_adjudication_frozen"
)
INITIAL_AUDIT_RULES = {
    "calibration_fraction_per_mechanism_stream_field": 0.10,
    "all_vlm_disagreements": True,
    "all_partial_atoms": True,
    "all_unusable_output_atoms": True,
    "confidence_below": 0.75,
    "separate_high_confidence_agreement_fraction": 0.10,
    "expand_remaining_mechanism_field_if_error_rate_above": 0.05,
    "two_independent_humans": True,
    "third_adjudicator_for_every_human_disagreement": True,
}

REGISTRY_PROTOCOL = "causal_role_erasure_7m_evaluation_code_registry_v1"
REGISTRY_STATUS = "five_evaluation_implementations_frozen"
REGISTRY_SHA256 = "bafc045ba94a6a5a8771ea8a2dd879c5f42aa219e3b5228b43594494b24eedf4"
REGISTRY_FILE_SHA256 = (
    "c4f10b7129493fd2aa646ab4a38f0b141332dfa8d9da1e64f2c737c33861528f"
)
REGISTRY_FILE_SIZE = 1081
REGISTRY_HELPER = {
    "path": "scripts/causal_role_erasure_7mechanism_evaluation_code_registry_v1.py",
    "sha256": "c65278de7a357c4f7db2f12d4927ffe46afe5e1a504524194ab9efcfbefd7536",
    "size_bytes": 2746,
}
V1_FILES = (
    {
        "path": "scripts/build_causal_role_erasure_7mechanism_review_package_v1.py",
        "sha256": "376548ca484e60b6923857325c5e49998f5cbc34a46fcedb36aee94c0da2d88f",
        "size_bytes": 69575,
    },
    {
        "path": "scripts/review_causal_role_erasure_7mechanism_formal_v2.py",
        "sha256": "3452c2259da25087cb017881404b4636e4ecb977c303f5f7af5b1de4d0c02616",
        "size_bytes": 71124,
    },
    {
        "path": "scripts/run_causal_role_erasure_7mechanism_formal_review_v1.py",
        "sha256": "1ce5ef0234e6c1afc891c1848f34e1b6ddda7c6dc583ce3e8e65447a519aafa4",
        "size_bytes": 63152,
    },
    {
        "path": "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
        "sha256": "ecf5bcffc60b3e5fe732f061544fe5480346c6d203be697e31fbae595887ce70",
        "size_bytes": 61365,
    },
    {
        "path": "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
        "sha256": "5b1a4b823082df9ae2d7e922adbd1bedec4c279593a71208228498b052f40114",
        "size_bytes": 50909,
    },
)

AMENDMENT_FIELDS = {
    "schema_version",
    "protocol",
    "protocol_version",
    "status",
    "deviation",
    "preserved_contract",
    "bindings",
    "amendment_sha256",
}
AMENDMENT_BINDING_FIELDS = {
    "evaluation_code_registry",
    "evaluation_code_registry_validator",
    "key_commitments",
    "formal_launch_receipt",
    "pass_a_merge_manifest",
    "pass_b_merge_manifest",
    "human_audit_manifest",
    "human_audit_queue",
    "formal_cases",
    "identification_subset",
    "tier_0_audit_strata",
    "tier_1_original_only",
    "tier_2_full",
}
AMENDMENT_DEVIATION = {
    "global_full_method_key_opened": True,
    "access_mode": "separate_read_only_audit",
    "opened_before_human_canonicalization": True,
    "opened_before_exploratory_ab_mean_materialization": True,
    "global_pre_unblinding_state_restorable": False,
    "vlm_review_process_used_public_only_inputs": True,
    "human_review_process_answer_key_blind_required": True,
    "post_unblinding_preview_admissible_for_final_claims": False,
}
PRESERVED_RULES = {
    "rubric",
    "audit_sampling",
    "canonical_substitution",
    "audit_expansion",
    "original_eligibility",
    "ces_and_su",
    "bootstrap",
    "estimability",
    "multiplicity",
    "noninferiority_margins",
    "claim_gates",
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
PENDING_OUTPUTS = (
    "human_labels",
    "adjudication",
    "canonical_scores",
    "original_eligibility",
    "formal_metrics",
    "paper_tables",
)
SOURCE_BINDING_NAMES = {
    "evaluation_code_registry",
    "key_commitments",
    "formal_launch_receipt",
    "pass_a_merge_manifest",
    "pass_b_merge_manifest",
    "human_audit_manifest",
    "human_audit_queue",
    "formal_cases",
    "identification_subset",
    "private_key_commitments",
}
REPRODUCIBILITY_BINDING_NAMES = {
    "evaluation_environment_lock",
    "evaluation_environment_receipt",
    "wan_training_receipt_inventory",
    "reproduction_command_ledger",
    "pre_metric_artifact_dag",
}
CRITICAL_COMPONENT_PATHS = {
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
}
SOURCE_BINDING_PATHS = {
    "evaluation_code_registry": ("review_root", "review_package/public/evaluation_code_registry.json"),
    "key_commitments": ("review_root", "review_package/public/key_commitments.json"),
    "formal_launch_receipt": ("review_root", "formal_launch_receipt.json"),
    "pass_a_merge_manifest": ("review_root", "merged/pass_a/merge_manifest.json"),
    "pass_b_merge_manifest": ("review_root", "merged/pass_b/merge_manifest.json"),
    "human_audit_manifest": ("review_root", "human_audit_stage0/audit_manifest.json"),
    "human_audit_queue": ("review_root", "human_audit_stage0/public/human_audit_queue.csv"),
    "formal_cases": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"),
    "identification_subset": ("repo_root", "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv"),
}
LOGICAL_REF_FIELDS = {"root_id", "path", "sha256", "size_bytes"}
CANONICAL_MANIFEST_FIELDS = {
    "schema_version",
    "protocol",
    "status",
    "provenance_amendment",
    "pre_metric_code_freeze",
    "evaluation_code_registry",
    "wrapper_implementation",
    "registered_v1_derivation",
    "key_access",
    "row_count",
    "atomic_count",
    "inputs",
    "artifacts",
    "quality",
    "manifest_sha256",
}
ELIGIBILITY_MANIFEST_FIELDS = {
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
    "artifacts",
    "counts",
    "manifest_sha256",
}
FORBIDDEN_PUBLISHED_TEXT = (
    "canonical_anonymous_scores_frozen_before_answer_key_opening",
    "original_eligibility_and_shared_subsets_frozen_before_full_key_opening",
    '"full_method_key_opened":false',
    '"full_method_key_opened_for_scoring":false',
)
REVIEW_PROCESS_FIELDS = {
    "schema_version",
    "protocol",
    "protocol_version",
    "status",
    "pre_metric_authority",
    "review_round_ancestry",
    "inputs",
    "counts",
    "assertions",
    "audit_manifest_status",
    "receipt_sha256",
}

LOCAL_IMPORT_SOURCES = {
    "causal_role_erasure_7mechanism_evaluation_code_registry_v1": (
        "scripts/causal_role_erasure_7mechanism_evaluation_code_registry_v1.py"
    ),
    "evaluate_v2_baseline_with_vlm": "scripts/evaluate_v2_baseline_with_vlm.py",
    "screen_causal_role_erasure_7mechanism_targets_v2": (
        "scripts/screen_causal_role_erasure_7mechanism_targets_v2.py"
    ),
    "review_causal_role_erasure_7mechanism_targets_v2": (
        "scripts/review_causal_role_erasure_7mechanism_targets_v2.py"
    ),
    "review_causal_role_erasure_7mechanism_formal_v2": (
        "scripts/review_causal_role_erasure_7mechanism_formal_v2.py"
    ),
}
LOCAL_IMPORT_ORDER = tuple(LOCAL_IMPORT_SOURCES)
SHADOW_IMPORT_NAMES = {
    *LOCAL_IMPORT_SOURCES,
    "canonicalize_causal_role_erasure_7mechanism_formal_v2",
    "numpy",
    "PIL",
    "av",
    "cv2",
}
_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True)
class InputSeal:
    """One immutable read of a mutable input used by a scientific stage."""

    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int


ISOLATED_STAGE_RUNNER = r'''#!/usr/bin/env python3
import importlib
import importlib.machinery
import importlib.metadata
import hashlib
import json
import os
import sys
from pathlib import Path


def fail(message):
    raise RuntimeError(message)


payload_path = Path(sys.argv[1])
payload = json.loads(payload_path.read_text(encoding="utf-8"))
code_root = Path(payload["code_root"]).resolve(strict=True)
scripts_root = code_root / "scripts"
site_roots = [str(Path(value).resolve(strict=True)) for value in payload["site_roots"]]
sys.meta_path[:] = [
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    importlib.machinery.PathFinder,
]
sys.path[:] = [str(scripts_root), *site_roots, *payload["stdlib_roots"]]

for name, expected_version in payload["distributions"].items():
    module_name = {"pillow": "PIL", "opencv-python-headless": "cv2"}.get(name, name)
    if module_name not in payload["required_modules"]:
        continue
    module = importlib.import_module(module_name)
    distribution = importlib.metadata.distribution(name)
    if distribution.version != expected_version:
        fail(f"isolated {name} distribution version differs from environment receipt")
    observed_version = getattr(module, "__version__", None)
    if module_name == "PIL":
        from PIL import Image
        observed_version = Image.__version__
    if str(observed_version) != expected_version:
        fail(f"isolated {module_name} version differs from environment receipt")
    origin = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if not any(origin == root or root in origin.parents for root in map(Path, site_roots)):
        fail(f"isolated {module_name} loaded outside the receipt-bound site roots")
    owned_files = {
        Path(distribution.locate_file(item)).resolve()
        for item in (distribution.files or ())
    }
    if origin not in owned_files:
        fail(f"isolated {module_name} origin is not owned by its locked distribution")

stage = payload["stage"]
if stage in {"canonical", "eligibility"}:
    legacy = importlib.import_module(
        "canonicalize_causal_role_erasure_7mechanism_formal_v1"
    )
else:
    legacy = importlib.import_module(
        "compute_causal_role_erasure_7mechanism_formal_metrics_v1"
    )

arguments = {name: Path(value) for name, value in payload["arguments"].items()}
if stage == "canonical":
    manifest = legacy.load_json(arguments.pop("snapshot_audit_manifest"), "snapshot audit manifest")
    audit_rows = legacy.load_jsonl(arguments.pop("snapshot_audit_key"), "snapshot audit key")
    all_rows = legacy.load_jsonl(arguments.pop("snapshot_all_atoms"), "snapshot all atoms")
    scores = {
        "A": legacy.load_jsonl(arguments["scores_a_path"], "snapshot pass A scores"),
        "B": legacy.load_jsonl(arguments["scores_b_path"], "snapshot pass B scores"),
    }
    manifest = dict(manifest)
    manifest["inputs"] = dict(manifest["inputs"])
    for input_name, argument_name in (
        ("scores_a", "scores_a_path"),
        ("scores_b", "scores_b_path"),
    ):
        snapshot_path = arguments[argument_name]
        snapshot_bytes = snapshot_path.read_bytes()
        manifest["inputs"][input_name] = {
            "path": str(snapshot_path.resolve()),
            "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "size_bytes": len(snapshot_bytes),
        }
    for pass_id, rows in scores.items():
        if len(rows) != legacy.EXPECTED_ITEMS:
            fail(f"snapshot pass {pass_id} score count changed")
        seen = set()
        for index, row in enumerate(rows):
            legacy.validate_score_row(row, f"snapshot pass {pass_id} row {index}")
            review_id = row["anonymous_review_id"]
            if not review_id or review_id in seen:
                fail(f"snapshot pass {pass_id} score IDs changed")
            seen.add(review_id)

    def load_audit_root(_root):
        return manifest, audit_rows, all_rows

    def load_merged_scores(_path, pass_id, expected_items=legacy.EXPECTED_ITEMS, **_kwargs):
        rows = scores[pass_id]
        if len(rows) != expected_items:
            fail(f"snapshot pass {pass_id} expected count changed")
        return rows

    legacy._load_audit_root = load_audit_root
    legacy.load_merged_scores = load_merged_scores
    legacy.freeze_canonical_scores(**arguments)
elif stage == "eligibility":
    legacy.freeze_original_eligibility(**arguments)
elif stage == "metrics":
    legacy.compute_formal_metrics(**arguments)
else:
    fail("unknown isolated registered stage")
'''


class ProvenanceV2Error(ValueError):
    """A v2 authority, compatibility, or publication invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceV2Error(message)


def _read_regular_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    """Read one regular file through one no-follow descriptor.

    The descriptor is opened once, and both metadata checks bracket the read.
    This avoids validating one inode and hashing/executing another.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProvenanceV2Error(f"{label} cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label} changed while it was read: {path}",
        )
        payload = b"".join(chunks)
        require(len(payload) == after.st_size, f"{label} size changed while read: {path}")
        return payload, after
    finally:
        os.close(descriptor)


def _seal_input(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[bytes, InputSeal]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    payload, metadata = _read_regular_bytes(absolute, label)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"{label} SHA differs")
    if expected_size is not None:
        require(len(payload) == expected_size, f"{label} size differs")
    return payload, InputSeal(
        path=absolute,
        sha256=digest,
        size_bytes=len(payload),
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _remember_seal(seals: dict[Path, InputSeal], seal: InputSeal) -> None:
    previous = seals.get(seal.path)
    require(
        previous is None or previous == seal,
        f"input changed between stage snapshots: {seal.path}",
    )
    seals[seal.path] = seal


def _snapshot_input(
    source: Path,
    target: Path,
    seals: dict[Path, InputSeal],
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> InputSeal:
    payload, seal = _seal_input(
        source,
        label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    _remember_seal(seals, seal)
    _write_bytes(target, payload)
    return seal


def _record_input(
    path: Path,
    seals: dict[Path, InputSeal],
    label: str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> InputSeal:
    _payload, seal = _seal_input(
        path,
        label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    _remember_seal(seals, seal)
    return seal


def _capture_validated_json(
    path: Path,
    seals: dict[Path, InputSeal],
    label: str,
    validated: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    payloads: dict[Path, bytes] | None = None,
) -> InputSeal:
    payload, seal = _seal_input(
        path,
        label,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    )
    captured = _load_json_bytes(payload, f"captured {label}")
    require(
        captured == dict(validated),
        f"{label} changed after validation and before capture",
    )
    _remember_seal(seals, seal)
    if payloads is not None:
        payloads[seal.path] = payload
    return seal


def _snapshot_legacy_ref(
    *,
    root: Path,
    ref: Mapping[str, Any],
    target: Path,
    seals: dict[Path, InputSeal],
    label: str,
) -> tuple[Path, InputSeal]:
    candidate = Path(str(ref["path"]))
    source = Path(
        os.path.abspath(os.fspath(candidate if candidate.is_absolute() else root / candidate))
    )
    seal = _snapshot_input(
        source,
        target,
        seals,
        label,
        expected_sha256=str(ref["sha256"]),
        expected_size=(
            int(ref["size_bytes"]) if "size_bytes" in ref else None
        ),
    )
    return source, seal


def _assert_inputs_unchanged(seals: Mapping[Path, InputSeal]) -> None:
    for path, expected in sorted(seals.items(), key=lambda item: str(item[0])):
        _payload, observed = _seal_input(
            path,
            f"pre-publish input {path}",
            expected_sha256=expected.sha256,
            expected_size=expected.size_bytes,
        )
        require(
            (observed.device, observed.inode) == (expected.device, expected.inode),
            f"pre-publish input inode changed: {path}",
        )


def logical_ref_from_seal(root_id: str, root: Path, seal: InputSeal) -> dict[str, Any]:
    resolved_root = root.resolve()
    try:
        relative = seal.path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ProvenanceV2Error(
            f"snapshotted artifact is outside {root_id}: {seal.path}"
        ) from exc
    _safe_relative_text(relative, f"{root_id} snapshotted path")
    return {
        "root_id": root_id,
        "path": relative,
        "sha256": seal.sha256,
        "size_bytes": seal.size_bytes,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ascii_object_sha256(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    payload, _metadata = _read_regular_bytes(path, "SHA-256 input")
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    require(
        completed.returncode == 0,
        f"git {' '.join(arguments)} failed: {completed.stderr.decode('utf-8', 'replace').strip()}",
    )
    return completed.stdout


def _git_text(project_root: Path, *arguments: str) -> str:
    return _git_bytes(project_root, *arguments).decode("utf-8").strip()


def _committed_blob(project_root: Path, commit: str, relative: str) -> bytes:
    _safe_relative_text(relative, "committed component path")
    entry = _git_text(project_root, "ls-tree", commit, "--", relative)
    match = re.fullmatch(
        r"(100644|100755) blob ([0-9a-f]{40})\t(.+)", entry
    )
    require(
        match is not None and match.group(3) == relative,
        f"committed component is not one regular Git blob: {relative}",
    )
    object_id = match.group(2)
    require(
        _git_text(project_root, "rev-parse", f"{commit}:{relative}") == object_id,
        f"committed component blob identity changed: {relative}",
    )
    return _git_bytes(project_root, "cat-file", "blob", object_id)


def _verify_committed_path(
    *,
    project_root: Path,
    commit: str,
    relative: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[str, int]:
    payload = _committed_blob(project_root, commit, relative)
    observed_sha = hashlib.sha256(payload).hexdigest()
    observed_size = len(payload)
    if expected_sha256 is not None:
        require(observed_sha == expected_sha256, f"committed blob SHA changed: {relative}")
    if expected_size is not None:
        require(observed_size == expected_size, f"committed blob size changed: {relative}")
    current = project_root / relative
    current_payload, _metadata = _read_regular_bytes(
        current, f"live committed component {relative}"
    )
    require(
        hashlib.sha256(current_payload).hexdigest() == observed_sha
        and len(current_payload) == observed_size,
        f"live component differs from committed blob: {relative}",
    )
    return observed_sha, observed_size


def _trusted_runtime_paths() -> tuple[list[str], list[str]]:
    stdlib_candidates = {
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
    }
    site_candidates = {
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    }
    stdlib = sorted(
        str(Path(value).resolve()) for value in stdlib_candidates if value
    )
    destination_shared = sysconfig.get_config_var("DESTSHARED")
    if destination_shared:
        stdlib.append(str(Path(destination_shared).resolve()))
    base_prefix = Path(sys.base_prefix).resolve()
    for value in sys.path:
        if not value:
            continue
        candidate = Path(value).resolve()
        if base_prefix == candidate or base_prefix in candidate.parents:
            stdlib.append(str(candidate))
    site = sorted(str(Path(value).resolve()) for value in site_candidates if value)
    require(stdlib and site, "cannot resolve trusted interpreter import roots")
    return sorted(set(stdlib) - set(site)), site


def _module_from_committed_blob(
    *,
    project_root: Path,
    commit: str,
    relative: str,
    module_name: str,
) -> types.ModuleType:
    payload = _committed_blob(project_root, commit, relative)
    _verify_committed_path(
        project_root=project_root,
        commit=commit,
        relative=relative,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
    )
    filename = str((project_root / relative).resolve())
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceV2Error(f"registered source is not UTF-8: {relative}") from exc
    module = types.ModuleType(module_name)
    module.__file__ = filename
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name, loader=None, origin=filename
    )
    sys.modules[module_name] = module
    try:
        exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_verified_source_module(
    *,
    project_root: Path,
    relative: str,
    module_name: str,
    pre_metric: Mapping[str, Any],
    shadow_names: Sequence[str] = (),
):
    commit = str(pre_metric["implementation_commit"]["commit"])
    path = (project_root / relative).resolve()
    names_to_replace = {
        *shadow_names,
        module_name,
        "numpy",
        "PIL",
        "av",
        "cv2",
    }
    with _IMPORT_LOCK:
        saved_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if any(name == root or name.startswith(f"{root}.") for root in names_to_replace)
        }
        for name in list(saved_modules):
            sys.modules.pop(name, None)
        saved_path = list(sys.path)
        saved_meta = list(sys.meta_path)
        stdlib_roots, site_roots = _trusted_runtime_paths()
        sys.path[:] = [
            str((project_root / "scripts").resolve()),
            *site_roots,
            *stdlib_roots,
        ]
        sys.meta_path[:] = [
            importlib.machinery.BuiltinImporter,
            importlib.machinery.FrozenImporter,
            importlib.machinery.PathFinder,
        ]
        loaded_names: set[str] = set()
        try:
            for name in LOCAL_IMPORT_ORDER:
                if name not in shadow_names:
                    continue
                _module_from_committed_blob(
                    project_root=project_root,
                    commit=commit,
                    relative=LOCAL_IMPORT_SOURCES[name],
                    module_name=name,
                )
                loaded_names.add(name)
            module = _module_from_committed_blob(
                project_root=project_root,
                commit=commit,
                relative=relative,
                module_name=module_name,
            )
            loaded_names.add(module_name)
            require(
                Path(str(module.__file__)).resolve() == path,
                f"verified module origin changed: {relative}",
            )
            for name in shadow_names:
                dependency = sys.modules.get(name)
                require(
                    dependency is not None and getattr(dependency, "__file__", None),
                    f"verified local dependency has no file origin: {name}",
                )
                expected_relative = LOCAL_IMPORT_SOURCES[name]
                expected_origin = (project_root / expected_relative).resolve()
                require(
                    Path(str(dependency.__file__)).resolve() == expected_origin,
                    f"bare import shadowed outside its verified source: {name}",
                )
                _verify_committed_path(
                    project_root=project_root,
                    commit=commit,
                    relative=expected_relative,
                )
            for module_name_value, distribution_name in (
                ("numpy", "numpy"),
                ("PIL", "pillow"),
                ("av", "av"),
                ("cv2", "opencv-python-headless"),
            ):
                dependency = sys.modules.get(module_name_value)
                if dependency is None:
                    continue
                origin_text = getattr(dependency, "__file__", None)
                require(
                    isinstance(origin_text, str) and origin_text,
                    f"{module_name_value} has no import origin",
                )
                origin = Path(origin_text).resolve()
                require(
                    any(origin == Path(root) or Path(root) in origin.parents for root in site_roots),
                    f"{module_name_value} loaded outside trusted site-packages roots",
                )
                distribution = importlib.metadata.distribution(distribution_name)
                owned_files = {
                    Path(distribution.locate_file(item)).resolve()
                    for item in (distribution.files or ())
                }
                require(
                    origin in owned_files,
                    f"{module_name_value} origin is not owned by its locked distribution",
                )
                observed_version = getattr(dependency, "__version__", None)
                if module_name_value == "PIL":
                    observed_version = getattr(sys.modules.get("PIL.Image"), "__version__", observed_version)
                require(
                    str(observed_version)
                    == distribution.version,
                    f"{module_name_value} module/distribution versions differ",
                )
            return module
        finally:
            for name in list(sys.modules):
                if (
                    name in loaded_names
                    or any(
                        name == root or name.startswith(f"{root}.")
                        for root in ("numpy", "PIL", "av", "cv2")
                    )
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path
            sys.meta_path[:] = saved_meta


def _reject_shadow_capable_files(project_root: Path, commit: str) -> None:
    """Reject repository-local import candidates outside the frozen code tree."""

    roots = (project_root, project_root / "scripts")
    allowed = {
        (project_root / relative).resolve()
        for relative in LOCAL_IMPORT_SOURCES.values()
    }
    allowed.update(
        {
            (project_root / row["path"]).resolve()
            for row in (*V1_FILES, REGISTRY_HELPER)
        }
    )
    allowed.update(
        (project_root / relative).resolve()
        for relative in CRITICAL_COMPONENT_PATHS.values()
        if str(relative).endswith(".py")
    )
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for entry in root.iterdir():
            stem = entry.name[:-3] if entry.name.endswith(".py") else entry.name
            if stem not in SHADOW_IMPORT_NAMES:
                continue
            resolved = entry.resolve()
            if resolved in allowed and entry.is_file() and not entry.is_symlink():
                relative = resolved.relative_to(project_root.resolve()).as_posix()
                _verify_committed_path(
                    project_root=project_root,
                    commit=commit,
                    relative=relative,
                )
                continue
            raise ProvenanceV2Error(
                f"untracked, ignored, or unexpected shadow-capable import path: {entry}"
            )


def _validate_live_environment_without_rerunning_tests(
    *,
    reproducibility: Any,
    project_root: Path,
    lock_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Bind this process to the exact receipt while avoiding recursive pytest."""

    try:
        receipt = reproducibility.validate_environment_receipt(
            project_root=project_root,
            lock_path=lock_path,
            receipt_path=receipt_path,
            live=False,
        )
        reproducibility.require(
            sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12),
            "live release evaluator is not CPython 3.12.x",
        )
        pins = reproducibility.parse_lock(lock_path)
        installed = reproducibility.installed_distributions()
        reproducibility.require(
            all(installed.get(name) == version for name, version in pins.items()),
            "live installed distributions differ from the lock",
        )
        reproducibility.require(
            set(installed) - set(pins) - reproducibility.BOOTSTRAP_DISTRIBUTIONS
            == set(),
            "live release environment has unbound distributions",
        )
        live_python = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_sha256": sha256_file(Path(sys.executable).resolve(strict=True)),
        }
        reproducibility.require(
            receipt["python"] == live_python,
            "environment receipt Python differs from the live interpreter",
        )
        live_platform = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
        reproducibility.require(
            receipt["platform"] == live_platform,
            "environment receipt platform differs from the live host",
        )
        reproducibility.require(
            receipt["bootstrap_distributions"]
            == {
                name: installed[name]
                for name in sorted(reproducibility.BOOTSTRAP_DISTRIBUTIONS)
                if name in installed
            },
            "environment receipt bootstrap distributions differ from the live environment",
        )
        live_codecs = {
            "pyav_libraries": reproducibility.isolated_json_probe(
                "import av,json; print(json.dumps({k:list(v) for k,v in sorted(av.library_versions.items())},sort_keys=True))",
                "PyAV",
            ),
            "opencv": reproducibility.isolated_json_probe(
                "import cv2,json; print(json.dumps(cv2.__version__))", "OpenCV"
            ),
        }
        live_pillow = reproducibility.isolated_json_probe(
            "import json; from PIL import Image,features; print(json.dumps({'pillow':Image.__version__,'libjpeg':features.version_codec('jpg')},sort_keys=True))",
            "Pillow",
        )
        reproducibility.require(
            isinstance(live_pillow, dict), "live Pillow codec probe malformed"
        )
        live_codecs.update(live_pillow)
        reproducibility.require(
            receipt["codec_versions"] == live_codecs,
            "environment receipt codecs differ from the live environment",
        )
        reproducibility.require(
            receipt["external_tools"]
            == {
                "latexmk": reproducibility.external_tool_receipt(
                    "latexmk", ("-v",)
                )
            },
            "environment receipt external tools differ from the live environment",
        )
    except reproducibility.ReproducibilityError as exc:
        raise ProvenanceV2Error(str(exc)) from exc
    return receipt


def _materialize_registered_code(
    *, project_root: Path, pre_metric: Mapping[str, Any], destination: Path
) -> Path:
    """Create a private source-only tree from the recorded implementation commit."""

    commit = str(pre_metric["implementation_commit"]["commit"])
    destination.mkdir(mode=0o700)
    paths = {
        *(str(row["path"]) for row in V1_FILES),
        str(REGISTRY_HELPER["path"]),
        *LOCAL_IMPORT_SOURCES.values(),
    }
    for relative in sorted(paths):
        payload = _committed_blob(project_root, commit, relative)
        _verify_committed_path(
            project_root=project_root,
            commit=commit,
            relative=relative,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
        )
        _write_bytes(destination / relative, payload)
    runner = destination / "isolated_stage_runner.py"
    _write_bytes(runner, ISOLATED_STAGE_RUNNER.encode("utf-8"))
    return runner


def _environment_receipt_for_stage(
    *,
    pre_metric_path: Path,
    pre_metric: Mapping[str, Any],
    seals: dict[Path, InputSeal],
) -> dict[str, Any]:
    ref = validate_logical_ref(
        pre_metric["reproducibility_bindings"]["evaluation_environment_receipt"],
        "stage evaluation environment receipt",
    )
    require(
        ref["root_id"] == "pre_metric_root",
        "stage evaluation environment receipt root changed",
    )
    root = pre_metric_path.parent.resolve()
    receipt_path = Path(os.path.abspath(os.fspath(root / ref["path"])))
    try:
        receipt_path.relative_to(root)
    except ValueError as exc:
        raise ProvenanceV2Error(
            "stage evaluation environment receipt escapes pre_metric_root"
        ) from exc
    payload, seal = _seal_input(
        receipt_path,
        "stage evaluation environment receipt",
        expected_sha256=str(ref["sha256"]),
        expected_size=int(ref["size_bytes"]),
    )
    _remember_seal(seals, seal)
    return _load_json_bytes(payload, "stage evaluation environment receipt")


def _run_registered_stage_isolated(
    *,
    project_root: Path,
    pre_metric_path: Path,
    pre_metric: Mapping[str, Any],
    private_parent: Path,
    stage: str,
    arguments: Mapping[str, Path],
    seals: dict[Path, InputSeal],
) -> None:
    code_root = private_parent / "registered_code"
    runner = _materialize_registered_code(
        project_root=project_root,
        pre_metric=pre_metric,
        destination=code_root,
    )
    receipt = _environment_receipt_for_stage(
        pre_metric_path=pre_metric_path,
        pre_metric=pre_metric,
        seals=seals,
    )
    stdlib_roots, site_roots = _trusted_runtime_paths()
    required_modules = {
        "canonical": ["PIL", "av"],
        "eligibility": ["PIL", "av"],
        "metrics": ["numpy"],
    }
    require(stage in required_modules, f"unknown registered stage: {stage}")
    payload = {
        "stage": stage,
        "code_root": str(code_root),
        "arguments": {name: str(path) for name, path in arguments.items()},
        "stdlib_roots": stdlib_roots,
        "site_roots": site_roots,
        "distributions": dict(receipt["distributions"]),
        "required_modules": required_modules[stage],
    }
    payload_path = private_parent / f"{stage}_isolated_payload.json"
    _write_json(payload_path, payload)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(runner), str(payload_path)],
        cwd=private_parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    require(
        completed.returncode == 0,
        "isolated registered v1 stage failed: "
        + (completed.stderr.strip() or completed.stdout.strip() or "unknown failure"),
    )


def load_registered_canonicalizer(
    project_root: Path, pre_metric: Mapping[str, Any]
):
    require(
        not any(
            name == root or name.startswith(f"{root}.")
            for name in sys.modules
            for root in ("av", "cv2")
        ),
        "registered canonicalizer refuses a preloaded PyAV/OpenCV module",
    )
    module = _load_verified_source_module(
        project_root=project_root,
        relative="scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
        module_name="_causal7m_verified_canonical_v1",
        pre_metric=pre_metric,
        shadow_names=(
            "causal_role_erasure_7mechanism_evaluation_code_registry_v1",
            "review_causal_role_erasure_7mechanism_formal_v2",
            "review_causal_role_erasure_7mechanism_targets_v2",
            "evaluate_v2_baseline_with_vlm",
            "screen_causal_role_erasure_7mechanism_targets_v2",
        ),
    )
    require(
        Path(str(module.evaluation_code.__file__)).resolve()
        == (project_root / REGISTRY_HELPER["path"]).resolve(),
        "registered canonicalizer loaded a shadowed registry helper",
    )
    require(
        Path(str(module.transport.__file__)).resolve()
        == (
            project_root
            / "scripts/review_causal_role_erasure_7mechanism_formal_v2.py"
        ).resolve(),
        "registered canonicalizer loaded a shadowed transport module",
    )
    return module


def load_registered_metrics(project_root: Path, pre_metric: Mapping[str, Any]):
    require(
        not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules),
        "registered metric loader refuses a preloaded numpy module",
    )
    module = _load_verified_source_module(
        project_root=project_root,
        relative="scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
        module_name="_causal7m_verified_metrics_v1",
        pre_metric=pre_metric,
        shadow_names=("causal_role_erasure_7mechanism_evaluation_code_registry_v1",),
    )
    require(
        Path(str(module.evaluation_code.__file__)).resolve()
        == (project_root / REGISTRY_HELPER["path"]).resolve(),
        "registered metric builder loaded a shadowed registry helper",
    )
    return module


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def _reject_constant(value: str) -> None:
    raise ProvenanceV2Error(f"JSON contains non-finite number: {value}")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"JSON repeats key: {key}")
        value[key] = item
    return value


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceV2Error(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_json(path: Path, label: str) -> dict[str, Any]:
    payload, _metadata = _read_regular_bytes(path, label)
    return _load_json_bytes(payload, label)


def _sealed(value: Mapping[str, Any], field: str = "manifest_sha256") -> dict[str, Any]:
    body = dict(value)
    require(field not in body, f"self-hash field already present: {field}")
    return {**body, field: object_sha256(body)}


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    require(isinstance(value.get(field), str), f"{label} self-hash is missing")
    body = {key: item for key, item in value.items() if key != field}
    require(value[field] == object_sha256(body), f"{label} self-hash changed")


def _write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    path.chmod(mode)


def _write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    _write_bytes(path, canonical_json_bytes(value), mode)


def _copy_bytes(source: Path, target: Path) -> None:
    payload, _metadata = _read_regular_bytes(
        source, f"scientific payload {source.name}"
    )
    _write_bytes(target, payload)
    copied, _copied_metadata = _read_regular_bytes(
        target, f"copied scientific payload {source.name}"
    )
    require(copied == payload, f"byte parity failed: {source.name}")


def _safe_relative_text(value: Any, label: str) -> str:
    text = str(value)
    path = Path(text)
    require(text and not path.is_absolute(), f"{label} must be relative")
    require(".." not in path.parts and "." not in path.parts, f"{label} escapes its root")
    require(path.as_posix() == text, f"{label} is not a canonical POSIX path")
    return text


def logical_ref(root_id: str, root: Path, path: Path) -> dict[str, Any]:
    regular_file(path, f"{root_id} artifact")
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProvenanceV2Error(f"artifact is outside {root_id}: {path}") from exc
    relative_text = relative.as_posix()
    _safe_relative_text(relative_text, f"{root_id} path")
    return {
        "root_id": root_id,
        "path": relative_text,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_logical_ref(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == LOGICAL_REF_FIELDS, f"{label} ref schema changed")
    require(re.fullmatch(r"[a-z][a-z0-9_]*", str(value["root_id"])) is not None, f"{label} root_id invalid")
    _safe_relative_text(value["path"], f"{label} path")
    require(re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is not None, f"{label} SHA invalid")
    require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0, f"{label} size invalid")
    return dict(value)


def resolve_logical_ref(
    value: Any,
    roots: Mapping[str, Path],
    label: str,
    expected: Path | None = None,
) -> Path:
    ref = validate_logical_ref(value, label)
    require(ref["root_id"] in roots, f"{label} uses unavailable root_id {ref['root_id']}")
    root = roots[ref["root_id"]].resolve()
    path = (root / ref["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProvenanceV2Error(f"{label} escapes {ref['root_id']}") from exc
    regular_file(path, label)
    if expected is not None:
        require(path == expected.resolve(), f"{label} path differs")
    require(path.stat().st_size == ref["size_bytes"], f"{label} size differs")
    require(sha256_file(path) == ref["sha256"], f"{label} SHA differs")
    return path


def legacy_ref(path: Path) -> dict[str, Any]:
    regular_file(path, f"legacy compatibility artifact {path.name}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _fresh_staging(output_root: Path) -> Path:
    require(not output_root.exists() and not output_root.is_symlink(), f"fresh-only output exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.v2tmp-", dir=output_root.parent))
    staging.chmod(0o700)
    return staging


def _publish(staging: Path, output_root: Path) -> None:
    require(
        not output_root.exists() and not output_root.is_symlink(),
        f"output appeared during v2 stage: {output_root}",
    )
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    target_bytes = os.fsencode(output_root)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 0x1)  # NOREPLACE
    else:
        raise ProvenanceV2Error(
            "atomic no-replace directory publication is unavailable on this platform"
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise ProvenanceV2Error(
                f"output appeared during v2 stage: {output_root}"
            )
        raise OSError(error, os.strerror(error), str(output_root))


def _registry_body() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": REGISTRY_PROTOCOL,
        "status": REGISTRY_STATUS,
        "files": [dict(row) for row in V1_FILES],
    }


def expected_registry() -> dict[str, Any]:
    body = _registry_body()
    require(object_sha256(body) == REGISTRY_SHA256, "internal frozen registry identity drift")
    return {**body, "registry_sha256": REGISTRY_SHA256}


def validate_current_v1_sources(project_root: Path) -> None:
    for row in (*V1_FILES, REGISTRY_HELPER):
        path = project_root / row["path"]
        regular_file(path, f"frozen implementation {row['path']}")
        require(path.stat().st_size == row["size_bytes"], f"frozen implementation size changed: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"frozen implementation SHA changed: {row['path']}")


def validate_registry_file(path: Path, project_root: Path) -> dict[str, Any]:
    regular_file(path, "evaluation code registry")
    require(path.stat().st_size == REGISTRY_FILE_SIZE, "evaluation code registry size changed")
    require(sha256_file(path) == REGISTRY_FILE_SHA256, "evaluation code registry file SHA changed")
    value = load_json(path, "evaluation code registry")
    require(value == expected_registry(), "evaluation code registry body changed")
    validate_current_v1_sources(project_root)
    return value


def validate_amendment(path: Path, project_root: Path) -> dict[str, Any]:
    value = load_json(path, "evaluation provenance amendment")
    require(set(value) == AMENDMENT_FIELDS, "evaluation provenance amendment schema changed")
    require(
        value["schema_version"] == 2
        and value["protocol"] == AMENDMENT_PROTOCOL
        and value["protocol_version"] == PROTOCOL_VERSION
        and value["status"] == AMENDMENT_STATUS,
        "evaluation provenance amendment identity/status changed",
    )
    _validate_self_hash(value, "amendment_sha256", "evaluation provenance amendment")
    require(value["deviation"] == AMENDMENT_DEVIATION, "evaluation provenance deviation changed")
    preserved = value["preserved_contract"]
    require(
        isinstance(preserved, dict)
        and set(preserved) == {"permitted_changes", "scientific_payload_byte_parity_required", "rules"}
        and preserved["permitted_changes"] == ["manifest_schema", "provenance_statements", "wrapper_orchestration"]
        and preserved["scientific_payload_byte_parity_required"] is True,
        "amendment change scope changed",
    )
    rules = preserved["rules"]
    require(
        isinstance(rules, dict)
        and set(rules) == PRESERVED_RULES
        and set(rules.values()) == {"unchanged_from_evaluation_code_registry_v1"},
        "amendment scientific contract changed",
    )
    bindings = value["bindings"]
    require(isinstance(bindings, dict) and set(bindings) == AMENDMENT_BINDING_FIELDS, "amendment bindings changed")
    require(
        bindings["evaluation_code_registry"]
        == {
            "sha256": REGISTRY_FILE_SHA256,
            "size_bytes": REGISTRY_FILE_SIZE,
            "registry_sha256": REGISTRY_SHA256,
        },
        "amendment registry binding changed",
    )
    require(bindings["evaluation_code_registry_validator"] == REGISTRY_HELPER, "amendment registry-helper binding changed")
    for name in ("formal_cases", "identification_subset"):
        ref = bindings[name]
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}, f"{name} binding schema changed")
        relative = _safe_relative_text(ref["path"], f"{name} binding")
        artifact = project_root / relative
        regular_file(artifact, name)
        require(artifact.stat().st_size == ref["size_bytes"] and sha256_file(artifact) == ref["sha256"], f"{name} binding changed")
    validate_current_v1_sources(project_root)
    return value


def _all_sha256(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sha256" and isinstance(item, str):
                values.add(item)
            values.update(_all_sha256(item))
    elif isinstance(value, list):
        for item in value:
            values.update(_all_sha256(item))
    return values


def _component_by_role(value: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(value, list) and value, "pre-metric component inventory is empty")
    by_role: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for index, row in enumerate(value):
        require(
            isinstance(row, dict)
            and set(row) == {"path", "sha256", "size_bytes", "role"},
            f"pre-metric component {index} schema changed",
        )
        _safe_relative_text(row["path"], f"pre-metric component {index}")
        require(re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is not None, f"pre-metric component {index} SHA invalid")
        require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0, f"pre-metric component {index} size invalid")
        role = str(row["role"])
        require(
            re.fullmatch(r"[a-z][a-z0-9_]*", role) is not None
            and role not in by_role,
            f"duplicate/unsafe pre-metric role: {role}",
        )
        require(
            row["path"] not in paths,
            f"duplicate pre-metric component path: {row['path']}",
        )
        paths.add(str(row["path"]))
        by_role[role] = dict(row)
    return by_role


def _validate_git_implementation(
    *,
    project_root: Path,
    amendment_path: Path,
    pre_metric: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
) -> None:
    commit = pre_metric["implementation_commit"]
    commit_id = str(commit["commit"])
    require(
        _git_text(project_root, "rev-parse", f"{commit_id}^{{commit}}") == commit_id,
        "pre-metric implementation commit does not resolve to the recorded commit",
    )
    require(
        _git_text(project_root, "cat-file", "-t", commit_id) == "commit",
        "pre-metric implementation authority is not a Git commit object",
    )
    require(
        _git_text(project_root, "rev-parse", f"{commit_id}^{{tree}}")
        == commit["tree"],
        "pre-metric implementation tree differs from the recorded commit tree",
    )
    require(
        _git_text(project_root, "cat-file", "-t", str(commit["tree"])) == "tree",
        "pre-metric implementation tree is not a Git tree object",
    )
    require(
        amendment_path.resolve()
        == (project_root / CRITICAL_COMPONENT_PATHS["evaluation_provenance_amendment"]).resolve(),
        "CLI amendment is not the committed canonical amendment path",
    )
    require(
        Path(__file__).resolve()
        == (project_root / CRITICAL_COMPONENT_PATHS["canonicalization_wrapper"]).resolve(),
        "canonicalization wrapper is not running from the committed project path",
    )
    for role, expected_path in CRITICAL_COMPONENT_PATHS.items():
        require(role in components, f"pre-metric component missing: {role}")
        row = components[role]
        require(row["path"] == expected_path, f"pre-metric component path changed: {role}")
    for role, row in components.items():
        _verify_committed_path(
            project_root=project_root,
            commit=commit_id,
            relative=str(row["path"]),
            expected_sha256=str(row["sha256"]),
            expected_size=int(row["size_bytes"]),
        )
    for row in (*V1_FILES, REGISTRY_HELPER):
        _verify_committed_path(
            project_root=project_root,
            commit=commit_id,
            relative=str(row["path"]),
            expected_sha256=str(row["sha256"]),
            expected_size=int(row["size_bytes"]),
        )


def _validate_source_bindings(
    value: Any, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        isinstance(value, dict) and set(value) == SOURCE_BINDING_NAMES,
        "pre-metric named source-binding inventory changed",
    )
    for name, (root_id, path) in SOURCE_BINDING_PATHS.items():
        ref = validate_logical_ref(value[name], f"pre-metric source {name}")
        require(
            ref["root_id"] == root_id and ref["path"] == path,
            f"pre-metric source location changed: {name}",
        )
        amendment_ref = amendment["bindings"][name]
        require(
            ref["sha256"] == amendment_ref["sha256"]
            and ref["size_bytes"] == amendment_ref["size_bytes"],
            f"pre-metric source differs from amendment: {name}",
        )
    private = value["private_key_commitments"]
    require(
        isinstance(private, dict)
        and set(private)
        == {"tier_0_audit_strata", "tier_1_original_only", "tier_2_full"},
        "pre-metric private-key commitment inventory changed",
    )
    for name, ref in private.items():
        require(
            isinstance(ref, dict)
            and set(ref) == {"sha256", "row_count"}
            and ref == amendment["bindings"][name],
            f"pre-metric private-key commitment changed: {name}",
        )
    return dict(value)


def _validate_reproducibility_bindings(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and set(value) == REPRODUCIBILITY_BINDING_NAMES,
        "pre-metric reproducibility-binding inventory changed",
    )
    locations = {
        "evaluation_environment_lock": (
            "repo_root",
            CRITICAL_COMPONENT_PATHS["evaluation_environment_lock"],
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
    for name, (root_id, path) in locations.items():
        ref = validate_logical_ref(value[name], f"pre-metric reproducibility {name}")
        require(
            ref["root_id"] == root_id and ref["path"] == path,
            f"pre-metric reproducibility location changed: {name}",
        )
    return dict(value)


def _validate_pre_metric_local_artifacts(
    *, project_root: Path, pre_metric_path: Path, pre_metric: Mapping[str, Any]
) -> None:
    roots = {
        "repo_root": project_root,
        "pre_metric_root": pre_metric_path.parent,
    }
    refs = pre_metric["reproducibility_bindings"]
    lock_path = resolve_logical_ref(
        refs["evaluation_environment_lock"], roots, "evaluation environment lock"
    )
    component = _component_by_role(pre_metric["components"])[
        "evaluation_environment_lock"
    ]
    require(
        component["sha256"] == sha256_file(lock_path)
        and component["size_bytes"] == lock_path.stat().st_size,
        "environment-lock component and reproducibility binding differ",
    )
    environment_receipt_path = resolve_logical_ref(
        refs["evaluation_environment_receipt"],
        roots,
        "release evaluation environment receipt",
    )
    reproducibility = _load_verified_source_module(
        project_root=project_root,
        relative=CRITICAL_COMPONENT_PATHS["reproducibility_builder"],
        module_name="_causal7m_verified_reproducibility_v1",
        pre_metric=pre_metric,
    )
    _validate_live_environment_without_rerunning_tests(
        reproducibility=reproducibility,
        project_root=project_root,
        lock_path=lock_path,
        receipt_path=environment_receipt_path,
    )
    inventory_path = resolve_logical_ref(
        refs["wan_training_receipt_inventory"],
        roots,
        "Wan training receipt inventory",
    )
    inventory = load_json(inventory_path, "Wan training receipt inventory")
    require(
        inventory.get("schema_version") == 1
        and inventory.get("protocol")
        == "causal_role_erasure_7m_wan_training_receipt_inventory_v1"
        and inventory.get("protocol_version") == PROTOCOL_VERSION
        and inventory.get("status")
        == "18_receipts_indexed_from_individually_validated_eval_run_manifest",
        "Wan training receipt inventory identity/status changed",
    )
    _validate_self_hash(inventory, "inventory_sha256", "Wan receipt inventory")
    require(
        inventory.get("counts", {}).get("receipts") == 18
        and inventory.get("counts", {}).get("validated_generation_rows") == 684
        and isinstance(inventory.get("items"), list)
        and len(inventory["items"]) == 18,
        "Wan receipt inventory counts changed",
    )
    ledger_path = resolve_logical_ref(
        refs["reproduction_command_ledger"], roots, "reproduction command ledger"
    )
    ledger = load_json(ledger_path, "reproduction command ledger")
    require(
        ledger.get("schema_version") == 1
        and ledger.get("protocol")
        == "causal_role_erasure_7m_executable_command_ledger_v1"
        and ledger.get("protocol_version") == PROTOCOL_VERSION
        and ledger.get("status")
        == "full_reproduction_commands_frozen_before_canonical_metrics",
        "reproduction command ledger identity/status changed",
    )
    _validate_self_hash(ledger, "ledger_sha256", "reproduction command ledger")
    require(
        isinstance(ledger.get("commands"), list) and ledger["commands"],
        "reproduction command ledger is empty",
    )
    dag_path = resolve_logical_ref(
        refs["pre_metric_artifact_dag"], roots, "pre-metric artifact DAG"
    )
    dag = load_json(dag_path, "pre-metric artifact DAG")
    require(
        dag.get("schema_version") == 1
        and dag.get("protocol") == "causal_role_erasure_7m_artifact_dag_v1"
        and dag.get("protocol_version") == PROTOCOL_VERSION
        and dag.get("status")
        == "completed_prefix_bound_with_explicit_pending_human_branch",
        "pre-metric artifact DAG identity/status changed",
    )
    _validate_self_hash(dag, "dag_sha256", "pre-metric artifact DAG")
    nodes = dag.get("nodes")
    require(isinstance(nodes, list) and nodes, "pre-metric artifact DAG is empty")
    pending = {
        str(node.get("id"))
        for node in nodes
        if isinstance(node, dict) and node.get("status") == "pending"
    }
    require(
        set(PENDING_OUTPUTS) <= pending,
        "pre-metric artifact DAG omits a required pending output",
    )


def validate_pre_metric_freeze(
    path: Path,
    amendment_path: Path,
    amendment: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    value = load_json(path, "pre-metric code freeze")
    require(set(value) == PRE_METRIC_FIELDS, "pre-metric code-freeze schema changed")
    require(
        value["schema_version"] == 1
        and value["protocol"] == PRE_METRIC_PROTOCOL
        and value["protocol_version"] == PROTOCOL_VERSION
        and value["status"] == PRE_METRIC_STATUS,
        "pre-metric code-freeze identity/status changed",
    )
    _validate_self_hash(value, "manifest_sha256", "pre-metric code freeze")
    require(
        value["deviation"] == PRE_METRIC_DEVIATION,
        "pre-metric deviation declaration changed",
    )
    commit = value["implementation_commit"]
    require(
        isinstance(commit, dict)
        and set(commit) == {"commit", "tree", "worktree_clean"}
        and re.fullmatch(r"[0-9a-f]{40}", str(commit["commit"])) is not None
        and re.fullmatch(r"[0-9a-f]{40}", str(commit["tree"])) is not None
        and commit["worktree_clean"] is True,
        "pre-metric implementation commit binding changed",
    )
    authority = value["v1_authority"]
    require(isinstance(authority, dict) and set(authority) == {"evaluation_code_registry", "files"}, "pre-metric v1 authority schema changed")
    registry_ref = authority["evaluation_code_registry"]
    require(
        isinstance(registry_ref, dict)
        and set(registry_ref) == {"path", "sha256", "size_bytes", "registry_sha256"}
        and _safe_relative_text(registry_ref["path"], "pre-metric registry path")
        and registry_ref["sha256"] == REGISTRY_FILE_SHA256
        and registry_ref["size_bytes"] == REGISTRY_FILE_SIZE
        and registry_ref["registry_sha256"] == REGISTRY_SHA256,
        "pre-metric registry authority changed",
    )
    require(authority["files"] == [dict(row) for row in V1_FILES], "pre-metric five-file authority changed")
    components = _component_by_role(value["components"])
    _validate_source_bindings(value["source_bindings"], amendment)
    _validate_reproducibility_bindings(value["reproducibility_bindings"])
    pending = value["pending_outputs"]
    require(
        isinstance(pending, list)
        and pending
        == [{"name": name, "status": "not_materialized"} for name in PENDING_OUTPUTS],
        "pre-metric pending-output inventory changed",
    )
    _validate_git_implementation(
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric=value,
        components=components,
    )
    _reject_shadow_capable_files(
        project_root, str(value["implementation_commit"]["commit"])
    )
    _validate_pre_metric_local_artifacts(
        project_root=project_root, pre_metric_path=path, pre_metric=value
    )
    validate_current_v1_sources(project_root)
    return value


def validate_execution_authority(
    amendment_path: Path,
    pre_metric_path: Path,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    amendment = validate_amendment(amendment_path, project_root)
    pre_metric = validate_pre_metric_freeze(
        pre_metric_path, amendment_path, amendment, project_root
    )
    return amendment, pre_metric


def _binding_matches_file(binding: Mapping[str, Any], path: Path, label: str) -> None:
    regular_file(path, label)
    require(binding.get("sha256") == sha256_file(path), f"{label} SHA differs from amendment")
    if "size_bytes" in binding:
        require(binding["size_bytes"] == path.stat().st_size, f"{label} size differs from amendment")


def _resolve_legacy_ref(root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, dict) and set(value) >= {"path", "sha256"}, f"{label} ref malformed")
    candidate = Path(str(value["path"]))
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    regular_file(path, label)
    require(sha256_file(path) == value["sha256"], f"{label} SHA differs")
    return path


def _key_commitments_from_audit(audit_root: Path) -> Path:
    manifest = load_json(audit_root / "audit_manifest.json", "audit manifest")
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict), "audit manifest inputs missing")
    return _resolve_legacy_ref(audit_root, inputs.get("key_commitments"), "audit key commitments")


def _initial_audit_manifest_path(audit_root: Path) -> Path:
    current = (audit_root / "audit_manifest.json").resolve()
    seen: set[Path] = set()
    for _ in range(8):
        require(current not in seen, "audit-manifest ancestry cycle")
        seen.add(current)
        manifest = load_json(current, "audit manifest ancestry")
        if manifest.get("status") == "initial_human_audit_frozen":
            return current
        ref = manifest.get("source_audit_manifest")
        require(ref is not None, "expanded audit does not bind its initial ancestor")
        current = _resolve_legacy_ref(current.parent, ref, "source audit manifest")
    raise ProvenanceV2Error("audit-manifest ancestry is too deep")


def _load_audit_payloads(
    legacy: Any, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = manifest_path.parent
    try:
        manifest, audit_rows, all_rows = legacy._load_audit_root(root)
    except legacy.CanonicalizationError as exc:
        raise ProvenanceV2Error(str(exc)) from exc
    require(manifest_path == (root / "audit_manifest.json").resolve(), "audit manifest path changed")
    return manifest, audit_rows, all_rows


def validate_audit_ancestry(
    *,
    audit_root: Path,
    amendment: Mapping[str, Any],
    legacy: Any,
) -> dict[str, Any]:
    final_path = (audit_root / "audit_manifest.json").resolve()
    chain: list[
        tuple[
            Path,
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]
    ] = []
    current = final_path
    seen: set[Path] = set()
    for _ in range(8):
        require(current not in seen, "audit-manifest ancestry cycle")
        seen.add(current)
        manifest, audit_rows, all_rows = _load_audit_payloads(legacy, current)
        require(
            manifest.get("protocol")
            == "causal_role_erasure_7m_human_canonicalization_v1"
            and manifest.get("evaluation_code_registry_sha256") == REGISTRY_SHA256
            and manifest.get("audit_seed") == 8_202_600,
            "audit ancestry identity changed",
        )
        chain.append((current, manifest, audit_rows, all_rows))
        if manifest.get("status") == "initial_human_audit_frozen":
            break
        require(
            manifest.get("status")
            in {
                "expanded_human_audit_frozen",
                "no_expansion_required_human_audit_frozen",
            },
            "audit ancestry contains an unknown status",
        )
        source = manifest.get("source_audit_manifest")
        require(source is not None, "expanded audit omits its source manifest")
        current = _resolve_legacy_ref(
            current.parent, source, "source audit manifest"
        )
    else:
        raise ProvenanceV2Error("audit-manifest ancestry is too deep")
    require(
        chain[-1][1].get("status") == "initial_human_audit_frozen",
        "audit ancestry does not terminate at the initial audit",
    )
    require(
        len(chain) == 2,
        "final audit must be exactly one deterministic expansion/no-expansion child of the initial audit",
    )
    initial_path, initial, initial_rows, initial_all = chain[-1]
    require(
        set(initial)
        == {
            "schema_version",
            "protocol",
            "status",
            "evaluation_code_registry_sha256",
            "audit_seed",
            "rules",
            "counts",
            "inputs",
            "artifacts",
            "human_delivery",
        },
        "initial audit manifest schema changed",
    )
    require(
        initial["schema_version"] == 1
        and initial["status"] == "initial_human_audit_frozen"
        and initial["rules"] == INITIAL_AUDIT_RULES
        and isinstance(initial["inputs"], dict)
        and set(initial["inputs"])
        == {
            "scores_a",
            "scores_b",
            "assignments",
            "audit_strata_key",
            "key_commitments",
        }
        and isinstance(initial["artifacts"], dict)
        and set(initial["artifacts"])
        == {"human_queue", "media_manifest", "audit_key", "all_atoms"},
        "initial audit nested contract changed",
    )
    _binding_matches_file(
        amendment["bindings"]["human_audit_manifest"],
        initial_path,
        "initial human-audit manifest",
    )
    initial_queue = _resolve_legacy_ref(
        initial_path.parent,
        initial["artifacts"]["human_queue"],
        "initial human-audit queue",
    )
    _binding_matches_file(
        amendment["bindings"]["human_audit_queue"],
        initial_queue,
        "initial human-audit queue",
    )
    expected_initial, reason_counts = legacy.select_audit_atoms(initial_all)
    require(
        initial_rows == expected_initial,
        "initial audit atom selection is not the exact deterministic v1 selection",
    )
    require(
        initial.get("counts")
        == {
            "videos": 2448,
            "atoms": len(initial_all),
            "initial_audit_atoms": len(initial_rows),
            "initial_audit_videos": len(
                {row["anonymous_review_id"] for row in initial_rows}
            ),
            "reasons": reason_counts,
        },
        "initial audit counts changed",
    )
    forward = list(reversed(chain))
    for (
        source_path,
        source_manifest,
        source_rows,
        source_all,
    ), (
        child_path,
        child_manifest,
        child_rows,
        child_all,
    ) in zip(forward, forward[1:]):
        require(
            set(child_manifest)
            == {
                "schema_version",
                "protocol",
                "status",
                "evaluation_code_registry_sha256",
                "audit_seed",
                "source_audit_manifest",
                "completed_source_human_audit",
                "counts",
                "inputs",
                "artifacts",
            },
            "expanded/no-expansion audit manifest schema changed",
        )
        require(
            child_manifest["schema_version"] == 1
            and isinstance(child_manifest["artifacts"], dict)
            and set(child_manifest["artifacts"])
            == {
                "human_queue",
                "media_manifest",
                "audit_key",
                "all_atoms",
                "expansion_diagnostics",
            },
            "expanded/no-expansion audit nested contract changed",
        )
        require(
            _resolve_legacy_ref(
                child_path.parent,
                child_manifest["source_audit_manifest"],
                "child source-audit manifest",
            )
            == source_path,
            "audit child binds a different source manifest",
        )
        completed_path = _resolve_legacy_ref(
            child_path.parent,
            child_manifest["completed_source_human_audit"],
            "source completed-human audit",
        )
        source_by_id = {row["audit_id"]: row for row in source_rows}
        completed = legacy.validate_completed_human_rows(
            completed_path, source_by_id, require_adjudication=True
        )
        strata, required, _diagnostics = legacy.expansion_requirements(
            source_by_id,
            {row["audit_id"]: row for row in source_all},
            completed,
        )
        expected_added_ids = sorted(required - set(source_by_id))
        expected_added = []
        all_by_id = {row["audit_id"]: row for row in source_all}
        for audit_id in expected_added_ids:
            row = dict(all_by_id[audit_id])
            row["reason_codes"] = [
                "expanded_high_confidence_error_gt_5pct"
            ]
            expected_added.append(row)
        require(
            child_all == source_all,
            "audit child changed or added atoms to the frozen all-atom inventory",
        )
        require(
            child_rows == [*source_rows, *expected_added],
            "audit child has missing, extra, reordered, or altered audit atoms",
        )
        expected_status = (
            "expanded_human_audit_frozen"
            if expected_added
            else "no_expansion_required_human_audit_frozen"
        )
        require(
            child_manifest["status"] == expected_status,
            "audit child status disagrees with deterministic expansion",
        )
        require(
            child_manifest["inputs"] == source_manifest["inputs"],
            "audit child changed frozen score/package inputs",
        )
        counts = child_manifest["counts"]
        require(
            isinstance(counts, dict)
            and set(counts)
            == {
                "videos",
                "atoms",
                "initial_audit_atoms",
                "expanded_strata",
                "added_audit_atoms",
                "total_audit_atoms",
            }
            and counts["videos"] == 2448
            and counts["atoms"] == len(source_all)
            and counts["initial_audit_atoms"] == len(source_rows)
            and counts["expanded_strata"] == len(strata)
            and counts["added_audit_atoms"] == len(expected_added)
            and counts["total_audit_atoms"] == len(child_rows),
            "audit child counts changed",
        )
    return {
        "initial_manifest": initial_path,
        "initial_queue": initial_queue,
        "final_manifest": final_path,
        "chain_length": len(chain),
        "final_status": chain[0][1]["status"],
        "final_audit_atoms": len(chain[0][2]),
    }


def validate_review_process_receipt(
    *,
    path: Path,
    audit_manifest_path: Path,
    completed_human_path: Path,
    project_root: Path,
    review_root: Path,
    pre_metric_path: Path,
    pre_metric: Mapping[str, Any],
) -> dict[str, Any]:
    value = load_json(path, "human-review process receipt")
    require(set(value) == REVIEW_PROCESS_FIELDS, "human-review process receipt schema changed")
    require(
        value["schema_version"] == 1
        and value["protocol"] == REVIEW_PROCESS_PROTOCOL
        and value["protocol_version"] == PROTOCOL_VERSION
        and value["status"] == REVIEW_PROCESS_STATUS,
        "human-review process receipt identity/status changed",
    )
    receipt_body = {
        key: item for key, item in value.items() if key != "receipt_sha256"
    }
    require(
        value["receipt_sha256"] == _ascii_object_sha256(receipt_body),
        "human-review process receipt self-hash changed",
    )
    expected_pre_metric_authority = {
        "pre_metric_code_freeze": logical_ref(
            "pre_metric_root", pre_metric_path.parent, pre_metric_path
        ),
        "implementation_commit": dict(pre_metric["implementation_commit"]),
        "evaluation_environment_receipt": dict(
            pre_metric["reproducibility_bindings"][
                "evaluation_environment_receipt"
            ]
        ),
    }
    require(
        value["pre_metric_authority"] == expected_pre_metric_authority,
        "human-review receipt binds a different pre-metric authority",
    )
    inputs = value["inputs"]
    expected_input_names = {
        "audit_manifest",
        "reviewer_instructions",
        "reviewer_1_delivery_receipt",
        "reviewer_2_delivery_receipt",
        "reviewer_1_completed",
        "reviewer_2_completed",
        "independence_attestation",
        "completed_human",
    }
    require(
        isinstance(inputs, dict) and set(inputs) == expected_input_names,
        "human-review process input inventory changed",
    )
    for name, ref in inputs.items():
        validate_logical_ref(ref, f"human-review process input {name}")
    roots = {
        "repo_root": project_root,
        "review_root": review_root,
        "review_process_root": path.parent,
    }
    expected_roots = {
        "audit_manifest": "review_root",
        "reviewer_instructions": "repo_root",
        "reviewer_1_delivery_receipt": "review_process_root",
        "reviewer_2_delivery_receipt": "review_process_root",
        "reviewer_1_completed": "review_process_root",
        "reviewer_2_completed": "review_process_root",
        "independence_attestation": "review_process_root",
        "completed_human": "review_process_root",
    }
    evidence: dict[str, Path] = {}
    for name, ref in inputs.items():
        require(
            ref["root_id"] == expected_roots[name],
            f"human-review process input root changed: {name}",
        )
        evidence[name] = resolve_logical_ref(
            ref,
            roots,
            f"human-review process evidence {name}",
            audit_manifest_path
            if name == "audit_manifest"
            else completed_human_path
            if name == "completed_human"
            else None,
        )
        if ref["root_id"] == "review_process_root":
            require(
                evidence[name].parent == path.parent.resolve(),
                f"human-review process evidence is not a direct receipt-root child: {name}",
            )
    require(
        evidence["reviewer_instructions"]
        == (project_root / CRITICAL_COMPONENT_PATHS["reviewer_instructions"]).resolve(),
        "human-review receipt binds different reviewer instructions",
    )
    require(
        value["assertions"]
        == {
            "two_independent_reviews": True,
            "reviews_frozen_before_merge": True,
            "reviewers_received_public_only_deliveries": True,
            "reviewers_did_not_access_peer_labels": True,
            "all_disagreements_adjudicated": True,
            "process_local_answer_key_blindness_attested": True,
            "global_answer_key_blindness_claimed": False,
        },
        "human-review process assertions changed",
    )
    audit_manifest = load_json(audit_manifest_path, "human-review-bound audit manifest")
    ancestry = value["review_round_ancestry"]
    require(
        isinstance(ancestry, dict)
        and set(ancestry) == {"round", "initial_process_receipt"},
        "human-review round ancestry schema changed",
    )
    if audit_manifest.get("status") == "initial_human_audit_frozen":
        require(
            ancestry == {"round": "initial", "initial_process_receipt": None},
            "initial human-review receipt has invalid ancestry",
        )
    else:
        binding = ancestry.get("initial_process_receipt")
        require(
            ancestry.get("round") == "expanded_or_no_expansion"
            and isinstance(binding, dict)
            and set(binding)
            == {
                "file_sha256",
                "size_bytes",
                "receipt_sha256",
                "audit_manifest_sha256",
                "audit_manifest_size_bytes",
                "completed_human_sha256",
                "completed_human_size_bytes",
            }
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(binding[name])) is not None
                for name in (
                    "file_sha256",
                    "receipt_sha256",
                    "audit_manifest_sha256",
                    "completed_human_sha256",
                )
            ),
            "final human-review receipt lacks the initial-process binding",
        )
        source_audit = audit_manifest.get("source_audit_manifest")
        source_completed = audit_manifest.get("completed_source_human_audit")
        require(
            isinstance(source_audit, dict)
            and isinstance(source_completed, dict)
            and binding["audit_manifest_sha256"] == source_audit.get("sha256")
            and binding["audit_manifest_size_bytes"] == source_audit.get("size_bytes")
            and binding["completed_human_sha256"] == source_completed.get("sha256")
            and binding["completed_human_size_bytes"] == source_completed.get("size_bytes"),
            "final human-review receipt does not bind its expansion source",
        )
    counts = value["counts"]
    require(
        isinstance(counts, dict)
        and set(counts)
        == {
            "audit_atoms",
            "human_agreements",
            "human_disagreements",
            "adjudicated_disagreements",
            "unresolved_disagreements",
        }
        and all(type(item) is int and item >= 0 for item in counts.values())
        and counts["unresolved_disagreements"] == 0
        and counts["adjudicated_disagreements"] == counts["human_disagreements"]
        and counts["human_agreements"] + counts["human_disagreements"]
        == counts["audit_atoms"],
        "human-review process counts changed",
    )
    require(
        value["audit_manifest_status"] == audit_manifest.get("status"),
        "human-review process audit status changed",
    )
    require(
        "full_key" not in path.read_text(encoding="utf-8"),
        "human-review process receipt exposes a full-key reference",
    )
    builder = _load_verified_source_module(
        project_root=project_root,
        relative=CRITICAL_COMPONENT_PATHS["review_process_receipt_builder"],
        module_name="_causal7m_verified_review_process_v2",
        pre_metric=pre_metric,
    )
    public_root = audit_manifest_path.parent / "public"
    try:
        audit, header, source_rows, public_inventory = builder.load_public_audit(
            audit_manifest_path, public_root
        )
        deliveries = {
            reviewer_id: builder.validate_delivery_receipt(
                evidence[f"{reviewer_id}_delivery_receipt"],
                reviewer_id,
                expected_pre_metric_authority,
            )
            for reviewer_id in ("reviewer_1", "reviewer_2")
        }
        expected_audit_ref = logical_ref(
            "review_root", review_root, audit_manifest_path
        )
        expected_instruction_ref = logical_ref(
            "repo_root", project_root, evidence["reviewer_instructions"]
        )
        source_inventory_sha = builder.inventory_digest(public_inventory)
        queue_path = public_root / "human_audit_queue.csv"
        source_queue_bytes = queue_path.read_bytes()
        source_queue_sha = hashlib.sha256(source_queue_bytes).hexdigest()
        source_by_path = {item["path"]: dict(item) for item in public_inventory}
        require(
            "human_audit_queue.csv" in source_by_path,
            "human-review public inventory omits its queue",
        )
        for reviewer_id, delivery in deliveries.items():
            require(
                delivery["audit_manifest"] == expected_audit_ref
                and delivery["reviewer_instructions"] == expected_instruction_ref,
                f"{reviewer_id} delivery binds different audit/instructions",
            )
            require(
                delivery["source_public_inventory_sha256"]
                == source_inventory_sha
                and delivery["source_queue_sha256"] == source_queue_sha
                and delivery["file_count"] == len(public_inventory)
                and delivery["audit_atom_count"] == len(source_rows),
                f"{reviewer_id} delivery source evidence changed",
            )
            if audit["status"] == "initial_human_audit_frozen":
                projection_mode = "byte_exact_initial_public_copy"
                projected_bytes = source_queue_bytes
            else:
                projection_mode = (
                    "reviewer_specific_expanded_queue_peer_and_adjudicator_columns_blank"
                )
                own = {
                    f"{reviewer_id}_score",
                    f"{reviewer_id}_notes",
                }
                projected_rows = []
                for row in source_rows:
                    projected = dict(row)
                    for field in builder.SCORE_FIELDS:
                        if field not in own:
                            projected[field] = ""
                    projected_rows.append(projected)
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=list(header))
                writer.writeheader()
                writer.writerows(projected_rows)
                projected_bytes = output.getvalue().encode("utf-8")
            projected_sha = hashlib.sha256(projected_bytes).hexdigest()
            projected_inventory = [dict(item) for item in public_inventory]
            for item in projected_inventory:
                if item["path"] == "human_audit_queue.csv":
                    item["sha256"] = projected_sha
                    item["size_bytes"] = len(projected_bytes)
            require(
                delivery["projection_mode"] == projection_mode
                and delivery["delivery_queue_sha256"] == projected_sha
                and delivery["delivery_inventory_sha256"]
                == builder.inventory_digest(projected_inventory),
                f"{reviewer_id} deterministic delivery projection changed",
            )
        rows_1 = builder.validate_reviewer_file(
            path=evidence["reviewer_1_completed"],
            reviewer_id="reviewer_1",
            header=header,
            blank_rows=source_rows,
        )
        rows_2 = builder.validate_reviewer_file(
            path=evidence["reviewer_2_completed"],
            reviewer_id="reviewer_2",
            header=header,
            blank_rows=source_rows,
        )
        builder.validate_attestation(
            evidence["independence_attestation"],
            sha256_file(evidence["reviewer_1_completed"]),
            sha256_file(evidence["reviewer_2_completed"]),
            sha256_file(evidence["reviewer_1_delivery_receipt"]),
            sha256_file(evidence["reviewer_2_delivery_receipt"]),
        )
        merged_header, merged = builder.read_csv(
            completed_human_path, "completed human audit"
        )
    except builder.ReviewProcessError as exc:
        raise ProvenanceV2Error(str(exc)) from exc
    require(
        merged_header == header and len(merged) == len(source_rows),
        "completed human audit inventory changed",
    )
    immutable = [field for field in header if field not in builder.SCORE_FIELDS]
    disagreements = 0
    for index, (before, first, second, row) in enumerate(
        zip(source_rows, rows_1, rows_2, merged)
    ):
        require(
            all(row[field] == before[field] for field in immutable),
            f"completed human row {index} changed immutable evidence",
        )
        require(
            all(
                row[field] == first[field]
                for field in ("reviewer_1_score", "reviewer_1_notes")
            )
            and all(
                row[field] == second[field]
                for field in ("reviewer_2_score", "reviewer_2_notes")
            ),
            f"completed human row {index} differs from independent reviewers",
        )
        if first["reviewer_1_score"] == second["reviewer_2_score"]:
            require(
                row["adjudicator_score"] == ""
                and row["adjudicator_notes"] == "",
                f"completed human row {index} has unnecessary adjudication",
            )
        else:
            disagreements += 1
            require(
                row["adjudicator_score"] in ("0", "1", "2"),
                f"completed human row {index} lacks adjudication",
            )
    require(
        counts
        == {
            "audit_atoms": len(merged),
            "human_agreements": len(merged) - disagreements,
            "human_disagreements": disagreements,
            "adjudicated_disagreements": disagreements,
            "unresolved_disagreements": 0,
        },
        "human-review process receipt counts do not reproduce",
    )
    return value


def validate_review_authority(
    *,
    project_root: Path,
    amendment: Mapping[str, Any],
    pre_metric: Mapping[str, Any],
    audit_root: Path | None = None,
    scores_a_path: Path | None = None,
    scores_b_path: Path | None = None,
    key_commitments_path: Path,
) -> tuple[Path, dict[str, Any]]:
    review_root = key_commitments_path.parent.parent.parent.resolve()
    roots = {"repo_root": project_root, "review_root": review_root}
    sources = pre_metric["source_bindings"]
    resolved_sources = {
        name: resolve_logical_ref(ref, roots, f"pre-metric source {name}")
        for name, ref in sources.items()
        if name != "private_key_commitments"
    }
    require(
        resolved_sources["key_commitments"] == key_commitments_path.resolve(),
        "CLI key commitments differ from pre-metric source binding",
    )
    _binding_matches_file(amendment["bindings"]["key_commitments"], key_commitments_path, "key commitments")
    commitments = load_json(key_commitments_path, "key commitments")
    require(
        commitments.get("protocol") == "causal_role_erasure_7m_anonymous_review_v1"
        and commitments.get("schema_version") == 1
        and commitments.get("commitment_scheme") == "sha256(canonical-jsonl-bytes)",
        "key-commitment identity changed",
    )
    for name, amendment_name in (
        ("tier_0_audit_strata", "tier_0_audit_strata"),
        ("tier_1_original_only", "tier_1_original_only"),
        ("tier_2_full", "tier_2_full"),
    ):
        require(commitments.get(name) == amendment["bindings"][amendment_name], f"{name} commitment changed")
    registry_path = resolved_sources["evaluation_code_registry"]
    registry = validate_registry_file(registry_path, project_root)
    require(
        commitments.get("evaluation_code_registry")
        == {
            "path": "evaluation_code_registry.json",
            "sha256": REGISTRY_FILE_SHA256,
            "registry_sha256": REGISTRY_SHA256,
        },
        "key-commitment registry binding changed",
    )
    if scores_a_path is not None:
        require(
            resolved_sources["pass_a_merge_manifest"]
            == (scores_a_path.parent / "merge_manifest.json").resolve(),
            "pass-A scores differ from the pre-metric merge binding",
        )
        _binding_matches_file(
            amendment["bindings"]["pass_a_merge_manifest"],
            scores_a_path.parent / "merge_manifest.json",
            "pass-A merge manifest",
        )
    if scores_b_path is not None:
        require(
            resolved_sources["pass_b_merge_manifest"]
            == (scores_b_path.parent / "merge_manifest.json").resolve(),
            "pass-B scores differ from the pre-metric merge binding",
        )
        _binding_matches_file(
            amendment["bindings"]["pass_b_merge_manifest"],
            scores_b_path.parent / "merge_manifest.json",
            "pass-B merge manifest",
        )
    if audit_root is not None:
        initial_manifest_path = _initial_audit_manifest_path(audit_root)
        _binding_matches_file(
            amendment["bindings"]["human_audit_manifest"],
            initial_manifest_path,
            "initial human-audit manifest",
        )
        initial = load_json(initial_manifest_path, "initial human-audit manifest")
        queue_path = _resolve_legacy_ref(
            initial_manifest_path.parent,
            initial["artifacts"]["human_queue"],
            "initial human-audit queue",
        )
        _binding_matches_file(
            amendment["bindings"]["human_audit_queue"],
            queue_path,
            "initial human-audit queue",
        )
        require(
            initial_manifest_path == resolved_sources["human_audit_manifest"],
            "audit ancestry begins from a different pre-metric audit manifest",
        )
        require(
            queue_path == resolved_sources["human_audit_queue"],
            "initial audit queue differs from the pre-metric source binding",
        )
    launch_path = resolved_sources["formal_launch_receipt"]
    _binding_matches_file(
        amendment["bindings"]["formal_launch_receipt"],
        launch_path,
        "formal launch receipt",
    )
    launch = load_json(launch_path, "formal launch receipt")
    require(
        launch.get("schema_version") == 1
        and launch.get("workflow_version")
        == "causal_role_erasure_7mechanism_formal_review_launch_v1"
        and launch.get("status")
        == "completed_two_independent_schema_valid_passes"
        and launch.get("evaluation_code_registry_sha256") == REGISTRY_SHA256
        and launch.get("scientific_zero_fallbacks") == 0,
        "formal launch receipt identity/status changed",
    )
    merged = launch.get("merged")
    require(
        isinstance(merged, dict) and set(merged) == {"pass_a", "pass_b"},
        "formal launch merge inventory changed",
    )
    for pass_name, source_name in (
        ("pass_a", "pass_a_merge_manifest"),
        ("pass_b", "pass_b_merge_manifest"),
    ):
        resolved = _resolve_legacy_ref(
            launch_path.parent, merged[pass_name], f"formal launch {pass_name} merge"
        )
        require(
            resolved == resolved_sources[source_name],
            f"formal launch {pass_name} binds a different merge",
        )
    return registry_path, registry


def _review_root(paths: Iterable[Path]) -> Path:
    resolved = [str(path.resolve()) for path in paths]
    require(bool(resolved), "cannot infer review root")
    return Path(os.path.commonpath(resolved)).resolve()


def _common_authority_fields(
    *,
    project_root: Path,
    amendment_root: Path,
    pre_metric_root: Path,
    review_root: Path,
    amendment_seal: InputSeal,
    pre_metric_seal: InputSeal,
    registry_seal: InputSeal,
    wrapper_seal: InputSeal,
) -> dict[str, Any]:
    return {
        "provenance_amendment": logical_ref_from_seal(
            "amendment_root", amendment_root, amendment_seal
        ),
        "pre_metric_code_freeze": logical_ref_from_seal(
            "pre_metric_root", pre_metric_root, pre_metric_seal
        ),
        "evaluation_code_registry": {
            **logical_ref_from_seal("review_root", review_root, registry_seal),
            "registry_sha256": REGISTRY_SHA256,
        },
        "wrapper_implementation": logical_ref_from_seal(
            "repo_root", project_root, wrapper_seal
        ),
    }


def _derivation(
    *,
    project_root: Path,
    implementation: Path,
    entry_point: str,
    legacy_manifest: Path,
    payloads: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "implementation": logical_ref("repo_root", project_root, implementation),
        "entry_point": entry_point,
        "legacy_manifest_sha256": sha256_file(legacy_manifest),
        "legacy_manifest_authoritative": False,
        "scientific_payload_byte_identical": True,
        "payload_sha256": {
            name: sha256_file(path) for name, path in sorted(payloads.items())
        },
    }


def scan_published_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"published v2 tree contains symlink: {path}")
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        compact = "".join(text.split())
        for forbidden in FORBIDDEN_PUBLISHED_TEXT:
            require(forbidden not in compact, f"published v2 tree contains legacy false provenance: {forbidden}")


def freeze_canonical_scores_v2(
    *,
    project_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    review_process_receipt_path: Path,
    audit_root: Path,
    completed_human_path: Path,
    scores_a_path: Path,
    scores_b_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    amendment, pre_metric = validate_execution_authority(
        amendment_path, pre_metric_path, project_root
    )
    key_commitments_path = _key_commitments_from_audit(audit_root)
    review_process_receipt = validate_review_process_receipt(
        path=review_process_receipt_path,
        audit_manifest_path=audit_root / "audit_manifest.json",
        completed_human_path=completed_human_path,
        project_root=project_root,
        review_root=key_commitments_path.parent.parent.parent,
        pre_metric_path=pre_metric_path,
        pre_metric=pre_metric,
    )
    registry_path, registry = validate_review_authority(
        project_root=project_root,
        amendment=amendment,
        pre_metric=pre_metric,
        audit_root=audit_root,
        scores_a_path=scores_a_path,
        scores_b_path=scores_b_path,
        key_commitments_path=key_commitments_path,
    )
    legacy = load_registered_canonicalizer(project_root, pre_metric)
    validate_audit_ancestry(
        audit_root=audit_root, amendment=amendment, legacy=legacy
    )
    seals: dict[Path, InputSeal] = {}
    amendment_seal = _capture_validated_json(
        amendment_path, seals, "provenance amendment", amendment
    )
    pre_metric_seal = _capture_validated_json(
        pre_metric_path, seals, "pre-metric code freeze", pre_metric
    )
    review_receipt_seal = _capture_validated_json(
        review_process_receipt_path,
        seals,
        "final review-process receipt",
        review_process_receipt,
    )
    registry_seal = _capture_validated_json(
        registry_path,
        seals,
        "evaluation code registry",
        registry,
        expected_sha256=REGISTRY_FILE_SHA256,
        expected_size=REGISTRY_FILE_SIZE,
    )
    wrapper_component = _component_by_role(pre_metric["components"])[
        "canonicalization_wrapper"
    ]
    wrapper_seal = _record_input(
        project_root / str(wrapper_component["path"]),
        seals,
        "canonicalization wrapper",
        expected_sha256=str(wrapper_component["sha256"]),
        expected_size=int(wrapper_component["size_bytes"]),
    )

    private_parent = Path(tempfile.mkdtemp(prefix="causal7m-canonical-v2-"))
    private_parent.chmod(0o700)
    staging: Path | None = None
    try:
        snapshot_root = private_parent / "verified_inputs"
        snapshot_root.mkdir(mode=0o700)
        audit_manifest_path = audit_root / "audit_manifest.json"
        completed_ref = review_process_receipt["inputs"]["completed_human"]
        completed_seal = _snapshot_input(
            completed_human_path,
            snapshot_root / "completed_human.csv",
            seals,
            "completed human audit snapshot",
            expected_sha256=str(completed_ref["sha256"]),
            expected_size=int(completed_ref["size_bytes"]),
        )
        audit_ref = review_process_receipt["inputs"]["audit_manifest"]
        audit_manifest_snapshot = snapshot_root / "audit" / "audit_manifest.json"
        audit_manifest_seal = _snapshot_input(
            audit_manifest_path,
            audit_manifest_snapshot,
            seals,
            "final audit manifest snapshot",
            expected_sha256=str(audit_ref["sha256"]),
            expected_size=int(audit_ref["size_bytes"]),
        )
        audit_manifest = load_json(
            audit_manifest_snapshot, "captured final audit manifest"
        )
        _audit_key_source, audit_key_seal = _snapshot_legacy_ref(
            root=audit_root,
            ref=audit_manifest["artifacts"]["audit_key"],
            target=snapshot_root / "audit" / "audit_key.jsonl",
            seals=seals,
            label="final audit key snapshot",
        )
        _all_atoms_source, all_atoms_seal = _snapshot_legacy_ref(
            root=audit_root,
            ref=audit_manifest["artifacts"]["all_atoms"],
            target=snapshot_root / "audit" / "all_atoms.jsonl",
            seals=seals,
            label="final all-atoms snapshot",
        )
        score_snapshots: dict[str, Path] = {}
        score_seals: dict[str, InputSeal] = {}
        for pass_name, pass_id, source in (
            ("pass_a", "A", scores_a_path),
            ("pass_b", "B", scores_b_path),
        ):
            merge_path = source.parent / "merge_manifest.json"
            merge_ref = amendment["bindings"][f"{pass_name}_merge_manifest"]
            merge_snapshot = snapshot_root / pass_name / "merge_manifest.json"
            _snapshot_input(
                merge_path,
                merge_snapshot,
                seals,
                f"{pass_name} merge manifest",
                expected_sha256=str(merge_ref["sha256"]),
                expected_size=int(merge_ref["size_bytes"]),
            )
            merge = load_json(
                merge_snapshot, f"captured {pass_name} merge manifest"
            )
            completed_score_ref = merge["completed_scores"]
            target = snapshot_root / pass_name / "completed_scores.jsonl"
            score_seals[pass_id] = _snapshot_input(
                source,
                target,
                seals,
                f"{pass_name} completed-score snapshot",
                expected_sha256=str(completed_score_ref["sha256"]),
                expected_size=(
                    int(completed_score_ref["size_bytes"])
                    if "size_bytes" in completed_score_ref
                    else None
                ),
            )
            score_snapshots[pass_id] = target
        review_root = key_commitments_path.parent.parent.parent.resolve()
        audit_strata_path = _resolve_legacy_ref(
            audit_root,
            audit_manifest["inputs"]["audit_strata_key"],
            "audit-strata key",
        )
        audit_strata_seal = _record_input(
            audit_strata_path,
            seals,
            "audit-strata key",
            expected_sha256=str(audit_manifest["inputs"]["audit_strata_key"]["sha256"]),
            expected_size=int(audit_manifest["inputs"]["audit_strata_key"]["size_bytes"]),
        )
        key_commitments_seal = _record_input(
            key_commitments_path,
            seals,
            "key commitments",
            expected_sha256=str(amendment["bindings"]["key_commitments"]["sha256"]),
            expected_size=int(amendment["bindings"]["key_commitments"]["size_bytes"]),
        )
        legacy_root = private_parent / "registered_v1"
        _run_registered_stage_isolated(
            project_root=project_root,
            pre_metric_path=pre_metric_path,
            pre_metric=pre_metric,
            private_parent=private_parent,
            stage="canonical",
            seals=seals,
            arguments={
                "audit_root": snapshot_root / "audit",
                "completed_human_path": snapshot_root / "completed_human.csv",
                "scores_a_path": score_snapshots["A"],
                "scores_b_path": score_snapshots["B"],
                "output_root": legacy_root,
                "snapshot_audit_manifest": snapshot_root
                / "audit"
                / "audit_manifest.json",
                "snapshot_audit_key": snapshot_root / "audit" / "audit_key.jsonl",
                "snapshot_all_atoms": snapshot_root / "audit" / "all_atoms.jsonl",
            },
        )
        legacy_manifest_path = legacy_root / "canonical_manifest.json"
        legacy_manifest = load_json(legacy_manifest_path, "legacy canonical manifest")
        require(
            legacy_manifest.get("status")
            == "canonical_anonymous_scores_frozen_before_answer_key_opening",
            "registered v1 canonical stage did not emit its expected compatibility status",
        )
        payloads = {
            "canonical_scores": legacy_root / "canonical_scores.jsonl",
            "audit_diagnostics": legacy_root / "audit_diagnostics.json",
        }
        staging = _fresh_staging(output_root)
        published = {
            name: staging / path.name for name, path in payloads.items()
        }
        for name in payloads:
            _copy_bytes(payloads[name], published[name])
        common = _common_authority_fields(
            project_root=project_root,
            amendment_root=amendment_path.parent,
            pre_metric_root=pre_metric_path.parent,
            review_root=review_root,
            amendment_seal=amendment_seal,
            pre_metric_seal=pre_metric_seal,
            registry_seal=registry_seal,
            wrapper_seal=wrapper_seal,
        )
        body = {
            "schema_version": 2,
            "protocol": CANONICAL_PROTOCOL,
            "status": CANONICAL_STATUS,
            **common,
            "registered_v1_derivation": _derivation(
                project_root=project_root,
                implementation=project_root
                / "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
                entry_point="freeze_canonical_scores",
                legacy_manifest=legacy_manifest_path,
                payloads=payloads,
            ),
            "key_access": {
                "global_full_method_key_opened_before_human_canonicalization": True,
                "full_method_key_read_by_this_stage": False,
                "human_review_process_local_answer_key_blindness_preserved": True,
                "global_pre_unblinding_claim": False,
            },
            "row_count": legacy_manifest["row_count"],
            "atomic_count": legacy_manifest["atomic_count"],
            "inputs": {
                "audit_manifest": logical_ref_from_seal(
                    "review_root", review_root, audit_manifest_seal
                ),
                "completed_human_audit": logical_ref_from_seal(
                    "review_process_root",
                    review_process_receipt_path.parent,
                    completed_seal,
                ),
                "scores_a": logical_ref_from_seal(
                    "review_root", review_root, score_seals["A"]
                ),
                "scores_b": logical_ref_from_seal(
                    "review_root", review_root, score_seals["B"]
                ),
                "audit_strata_key": logical_ref_from_seal(
                    "review_root", review_root, audit_strata_seal
                ),
                "key_commitments": logical_ref_from_seal(
                    "review_root", review_root, key_commitments_seal
                ),
                "review_process_receipt": logical_ref_from_seal(
                    "review_process_root",
                    review_process_receipt_path.parent,
                    review_receipt_seal,
                ),
            },
            "artifacts": {
                name: logical_ref("canonical_root", staging, path)
                for name, path in published.items()
            },
            "quality": legacy_manifest["quality"],
        }
        require(
            body["inputs"]["completed_human_audit"]
            == review_process_receipt["inputs"]["completed_human"],
            "canonical completed-human input differs from the final process receipt",
        )
        manifest = _sealed(body)
        require(set(manifest) == CANONICAL_MANIFEST_FIELDS, "internal canonical-v2 manifest schema drift")
        _write_json(staging / "canonical_manifest.json", manifest)
        scan_published_tree(staging)
        validate_current_v1_sources(project_root)
        _assert_inputs_unchanged(seals)
        _publish(staging, output_root)
        staging = None
        return manifest
    except legacy.CanonicalizationError as exc:
        raise ProvenanceV2Error(str(exc)) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(private_parent, ignore_errors=True)


def _roots_for_canonical(
    *,
    project_root: Path,
    pre_metric_path: Path,
    amendment_path: Path,
    canonical_root: Path,
    key_commitments_path: Path,
) -> dict[str, Path]:
    review_root = key_commitments_path.parent.parent.parent
    return {
        "repo_root": project_root,
        "amendment_root": amendment_path.parent,
        "pre_metric_root": pre_metric_path.parent,
        "canonical_root": canonical_root,
        "review_root": review_root,
    }


def _validate_registry_ref(value: Any, label: str) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and set(value) == {*LOGICAL_REF_FIELDS, "registry_sha256"},
        f"{label} registry-ref schema changed",
    )
    base = {key: value[key] for key in LOGICAL_REF_FIELDS}
    validate_logical_ref(base, label)
    require(
        value["root_id"] == "review_root"
        and value["path"] == "review_package/public/evaluation_code_registry.json"
        and value["sha256"] == REGISTRY_FILE_SHA256
        and value["size_bytes"] == REGISTRY_FILE_SIZE
        and value["registry_sha256"] == REGISTRY_SHA256,
        f"{label} registry authority changed",
    )
    return dict(value)


def _validate_derivation(
    *,
    value: Any,
    implementation_path: str,
    implementation_sha256: str,
    entry_point: str,
    payload_refs: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "implementation",
            "entry_point",
            "legacy_manifest_sha256",
            "legacy_manifest_authoritative",
            "scientific_payload_byte_identical",
            "payload_sha256",
        },
        f"{label} derivation schema changed",
    )
    implementation = validate_logical_ref(
        value["implementation"], f"{label} implementation"
    )
    require(
        implementation["root_id"] == "repo_root"
        and implementation["path"] == implementation_path
        and implementation["sha256"] == implementation_sha256
        and value["entry_point"] == entry_point
        and re.fullmatch(r"[0-9a-f]{64}", str(value["legacy_manifest_sha256"]))
        is not None
        and value["legacy_manifest_authoritative"] is False
        and value["scientific_payload_byte_identical"] is True,
        f"{label} derivation authority changed",
    )
    hashes = value["payload_sha256"]
    require(
        isinstance(hashes, dict) and set(hashes) == set(payload_refs),
        f"{label} parity-payload inventory changed",
    )
    for name, ref in payload_refs.items():
        require(
            hashes[name] == ref["sha256"],
            f"{label} parity SHA differs from published artifact: {name}",
        )


def _validate_authority_ref(
    value: Any,
    *,
    root_id: str,
    path: str,
    sha256: str,
    size_bytes: int,
    label: str,
) -> None:
    ref = validate_logical_ref(value, label)
    require(
        ref
        == {
            "root_id": root_id,
            "path": path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        },
        f"{label} authority changed",
    )


def load_canonical_v2(
    *,
    canonical_root: Path,
    project_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    key_commitments_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    path = canonical_root / "canonical_manifest.json"
    manifest = load_json(path, "canonical-v2 manifest")
    require(set(manifest) == CANONICAL_MANIFEST_FIELDS, "canonical-v2 manifest schema changed")
    require(
        manifest["schema_version"] == 2
        and manifest["protocol"] == CANONICAL_PROTOCOL
        and manifest["status"] == CANONICAL_STATUS,
        "canonical-v2 manifest identity/status changed",
    )
    _validate_self_hash(manifest, "manifest_sha256", "canonical-v2 manifest")
    require(
        isinstance(manifest["inputs"], dict)
        and set(manifest["inputs"])
        == {
            "audit_manifest",
            "completed_human_audit",
            "scores_a",
            "scores_b",
            "audit_strata_key",
            "key_commitments",
            "review_process_receipt",
        },
        "canonical-v2 input inventory changed",
    )
    require(
        isinstance(manifest["artifacts"], dict)
        and set(manifest["artifacts"])
        == {"canonical_scores", "audit_diagnostics"},
        "canonical-v2 artifact inventory changed",
    )
    for name, ref in manifest["inputs"].items():
        validate_logical_ref(ref, f"canonical-v2 input {name}")
    expected_input_roots = {
        "audit_manifest": "review_root",
        "completed_human_audit": "review_process_root",
        "scores_a": "review_root",
        "scores_b": "review_root",
        "audit_strata_key": "review_root",
        "key_commitments": "review_root",
        "review_process_receipt": "review_process_root",
    }
    require(
        all(
            manifest["inputs"][name]["root_id"] == root_id
            for name, root_id in expected_input_roots.items()
        )
        and Path(manifest["inputs"]["audit_manifest"]["path"]).name
        == "audit_manifest.json"
        and manifest["inputs"]["completed_human_audit"]["path"]
        == "completed_human.csv"
        and manifest["inputs"]["scores_a"]["path"]
        == "merged/pass_a/completed_scores.jsonl"
        and manifest["inputs"]["scores_b"]["path"]
        == "merged/pass_b/completed_scores.jsonl"
        and manifest["inputs"]["audit_strata_key"]["path"]
        == "review_package/private/audit_strata_key.jsonl"
        and manifest["inputs"]["key_commitments"]["path"]
        == "review_package/public/key_commitments.json"
        and manifest["inputs"]["review_process_receipt"]["path"]
        == "review_process_receipt_v2.json",
        "canonical-v2 input roots/locations changed",
    )
    _validate_registry_ref(
        manifest["evaluation_code_registry"], "canonical-v2 evaluation registry"
    )
    _validate_authority_ref(
        manifest["provenance_amendment"],
        root_id="amendment_root",
        path=amendment_path.name,
        sha256=sha256_file(amendment_path),
        size_bytes=amendment_path.stat().st_size,
        label="canonical-v2 amendment",
    )
    _validate_authority_ref(
        manifest["pre_metric_code_freeze"],
        root_id="pre_metric_root",
        path=pre_metric_path.name,
        sha256=sha256_file(pre_metric_path),
        size_bytes=pre_metric_path.stat().st_size,
        label="canonical-v2 pre-metric freeze",
    )
    wrapper_path = project_root / CRITICAL_COMPONENT_PATHS["canonicalization_wrapper"]
    _validate_authority_ref(
        manifest["wrapper_implementation"],
        root_id="repo_root",
        path=CRITICAL_COMPONENT_PATHS["canonicalization_wrapper"],
        sha256=sha256_file(wrapper_path),
        size_bytes=wrapper_path.stat().st_size,
        label="canonical-v2 wrapper",
    )
    roots = _roots_for_canonical(
        project_root=project_root,
        pre_metric_path=pre_metric_path,
        amendment_path=amendment_path,
        canonical_root=canonical_root,
        key_commitments_path=key_commitments_path,
    )
    resolve_logical_ref(manifest["provenance_amendment"], roots, "canonical amendment", amendment_path)
    resolve_logical_ref(manifest["pre_metric_code_freeze"], roots, "canonical pre-metric freeze", pre_metric_path)
    resolve_logical_ref(manifest["wrapper_implementation"], roots, "canonical wrapper", Path(__file__))
    for name in ("canonical_scores", "audit_diagnostics"):
        resolve_logical_ref(manifest["artifacts"][name], roots, f"canonical artifact {name}")
        require(
            manifest["artifacts"][name]["root_id"] == "canonical_root"
            and manifest["artifacts"][name]["path"]
            == {
                "canonical_scores": "canonical_scores.jsonl",
                "audit_diagnostics": "audit_diagnostics.json",
            }[name],
            f"canonical-v2 artifact location changed: {name}",
        )
    _validate_derivation(
        value=manifest["registered_v1_derivation"],
        implementation_path="scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
        implementation_sha256=V1_FILES[3]["sha256"],
        entry_point="freeze_canonical_scores",
        payload_refs=manifest["artifacts"],
        label="canonical-v2",
    )
    require(
        manifest["row_count"] == 2448 and manifest["atomic_count"] == 9792,
        "canonical-v2 row/atom counts changed",
    )
    quality = manifest["quality"]
    quality_fields = {
        "vlm_vlm_atomic_disagreement_rate",
        "calibration_vlm_a_human_macro_f1",
        "calibration_vlm_b_human_macro_f1",
        "calibration_vlm_a_human_quadratic_weighted_kappa",
        "calibration_vlm_b_human_quadratic_weighted_kappa",
        "all_targeted_audits_vlm_a_human_macro_f1",
        "all_targeted_audits_vlm_b_human_macro_f1",
        "all_targeted_audits_vlm_a_human_quadratic_weighted_kappa",
        "all_targeted_audits_vlm_b_human_quadratic_weighted_kappa",
        "human_review_atom_coverage",
        "human_review_video_coverage",
        "human_disagreement_atoms",
        "expanded_strata",
        "expanded_atoms",
    }
    require(
        isinstance(quality, dict)
        and set(quality) == quality_fields
        and all(
            type(quality[name]) in (int, float)
            for name in quality_fields
        ),
        "canonical-v2 quality schema changed",
    )
    expected_access = {
        "global_full_method_key_opened_before_human_canonicalization": True,
        "full_method_key_read_by_this_stage": False,
        "human_review_process_local_answer_key_blindness_preserved": True,
        "global_pre_unblinding_claim": False,
    }
    require(manifest["key_access"] == expected_access, "canonical-v2 key-access record changed")
    scan_published_tree(canonical_root)
    return manifest, roots


def _legacy_canonical_projection(
    *,
    destination: Path,
    manifest: Mapping[str, Any],
    roots: Mapping[str, Path],
    seals: dict[Path, InputSeal] | None = None,
    key_commitments_override: Path | None = None,
    audit_strata_override: Path | None = None,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True)
    destination.chmod(0o700)
    canonical_source = resolve_logical_ref(
        manifest["artifacts"]["canonical_scores"], roots, "canonical scores"
    )
    diagnostics_source = resolve_logical_ref(
        manifest["artifacts"]["audit_diagnostics"], roots, "audit diagnostics"
    )
    canonical_path = destination / "canonical_scores.jsonl"
    diagnostics_path = destination / "audit_diagnostics.json"
    if seals is None:
        _copy_bytes(canonical_source, canonical_path)
        _copy_bytes(diagnostics_source, diagnostics_path)
    else:
        _snapshot_input(
            canonical_source,
            canonical_path,
            seals,
            "canonical-score compatibility snapshot",
            expected_sha256=str(manifest["artifacts"]["canonical_scores"]["sha256"]),
            expected_size=int(manifest["artifacts"]["canonical_scores"]["size_bytes"]),
        )
        _snapshot_input(
            diagnostics_source,
            diagnostics_path,
            seals,
            "audit-diagnostics compatibility snapshot",
            expected_sha256=str(manifest["artifacts"]["audit_diagnostics"]["sha256"]),
            expected_size=int(manifest["artifacts"]["audit_diagnostics"]["size_bytes"]),
        )
    legacy_input_names = (
        "audit_manifest",
        "completed_human_audit",
        "scores_a",
        "scores_b",
        "audit_strata_key",
        "key_commitments",
    )
    resolved_inputs = {}
    for name in legacy_input_names:
        ref = manifest["inputs"][name]
        if name in {"audit_strata_key", "key_commitments"}:
            override = (
                audit_strata_override
                if name == "audit_strata_key"
                else key_commitments_override
            )
            resolved_inputs[name] = legacy_ref(
                override
                if override is not None
                else resolve_logical_ref(ref, roots, f"canonical input {name}")
            )
        else:
            resolved_inputs[name] = {
                "path": ref["path"],
                "sha256": ref["sha256"],
                "size_bytes": ref["size_bytes"],
            }
    legacy_manifest = {
        "schema_version": 1,
        "protocol": "causal_role_erasure_7m_human_canonicalization_v1",
        "status": "canonical_anonymous_scores_frozen_before_answer_key_opening",
        "evaluation_code_registry_sha256": REGISTRY_SHA256,
        "row_count": manifest["row_count"],
        "atomic_count": manifest["atomic_count"],
        "inputs": resolved_inputs,
        "artifacts": {
            "canonical_scores": legacy_ref(canonical_path),
            "audit_diagnostics": legacy_ref(diagnostics_path),
        },
        "quality": manifest["quality"],
        "full_method_key_opened_for_scoring": False,
    }
    _write_json(destination / "canonical_manifest.json", legacy_manifest)
    return canonical_path, destination / "canonical_manifest.json"


def freeze_original_eligibility_v2(
    *,
    project_root: Path,
    amendment_path: Path,
    pre_metric_path: Path,
    canonical_root: Path,
    original_key_path: Path,
    key_commitments_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    amendment, pre_metric = validate_execution_authority(
        amendment_path, pre_metric_path, project_root
    )
    registry_path, registry = validate_review_authority(
        project_root=project_root,
        amendment=amendment,
        pre_metric=pre_metric,
        key_commitments_path=key_commitments_path,
    )
    _binding_matches_file(
        amendment["bindings"]["tier_1_original_only"],
        original_key_path,
        "Original-only key",
    )
    canonical_manifest, roots = load_canonical_v2(
        canonical_root=canonical_root,
        project_root=project_root,
        amendment_path=amendment_path,
        pre_metric_path=pre_metric_path,
        key_commitments_path=key_commitments_path,
    )
    seals: dict[Path, InputSeal] = {}
    authority_payloads: dict[Path, bytes] = {}
    amendment_seal = _capture_validated_json(
        amendment_path,
        seals,
        "eligibility provenance amendment",
        amendment,
    )
    pre_metric_seal = _capture_validated_json(
        pre_metric_path,
        seals,
        "eligibility pre-metric code freeze",
        pre_metric,
    )
    registry_seal = _capture_validated_json(
        registry_path,
        seals,
        "eligibility evaluation code registry",
        registry,
        expected_sha256=REGISTRY_FILE_SHA256,
        expected_size=REGISTRY_FILE_SIZE,
        payloads=authority_payloads,
    )
    canonical_manifest_seal = _capture_validated_json(
        canonical_root / "canonical_manifest.json",
        seals,
        "eligibility canonical-v2 manifest",
        canonical_manifest,
    )
    wrapper_component = _component_by_role(pre_metric["components"])[
        "canonicalization_wrapper"
    ]
    wrapper_seal = _record_input(
        project_root / str(wrapper_component["path"]),
        seals,
        "eligibility canonicalization wrapper",
        expected_sha256=str(wrapper_component["sha256"]),
        expected_size=int(wrapper_component["size_bytes"]),
    )
    legacy = load_registered_canonicalizer(project_root, pre_metric)

    private_parent = Path(tempfile.mkdtemp(prefix="causal7m-eligibility-v2-"))
    private_parent.chmod(0o700)
    staging: Path | None = None
    try:
        snapshot_root = private_parent / "verified_inputs"
        snapshot_root.mkdir(mode=0o700)
        key_snapshot_root = snapshot_root / "keys"
        key_commitments_seal = _snapshot_input(
            key_commitments_path,
            key_snapshot_root / "key_commitments.json",
            seals,
            "eligibility key-commitments snapshot",
            expected_sha256=str(amendment["bindings"]["key_commitments"]["sha256"]),
            expected_size=int(amendment["bindings"]["key_commitments"]["size_bytes"]),
        )
        registry_snapshot = key_snapshot_root / "evaluation_code_registry.json"
        _write_bytes(registry_snapshot, authority_payloads[registry_seal.path])
        original_key_seal = _snapshot_input(
            original_key_path,
            snapshot_root / "original_only_key.jsonl",
            seals,
            "Original-only key snapshot",
            expected_sha256=str(amendment["bindings"]["tier_1_original_only"]["sha256"]),
        )
        audit_strata_source = resolve_logical_ref(
            canonical_manifest["inputs"]["audit_strata_key"],
            roots,
            "eligibility audit-strata key",
        )
        audit_strata_snapshot = snapshot_root / "audit_strata_key.jsonl"
        _snapshot_input(
            audit_strata_source,
            audit_strata_snapshot,
            seals,
            "eligibility audit-strata snapshot",
            expected_sha256=str(canonical_manifest["inputs"]["audit_strata_key"]["sha256"]),
            expected_size=int(canonical_manifest["inputs"]["audit_strata_key"]["size_bytes"]),
        )
        compat_canonical = private_parent / "canonical"
        canonical_scores_path, _ = _legacy_canonical_projection(
            destination=compat_canonical,
            manifest=canonical_manifest,
            roots=roots,
            seals=seals,
            key_commitments_override=key_snapshot_root / "key_commitments.json",
            audit_strata_override=audit_strata_snapshot,
        )
        legacy_root = private_parent / "registered_v1"
        _run_registered_stage_isolated(
            project_root=project_root,
            pre_metric_path=pre_metric_path,
            pre_metric=pre_metric,
            private_parent=private_parent,
            stage="eligibility",
            seals=seals,
            arguments={
                "canonical_root": compat_canonical,
                "original_key_path": snapshot_root / "original_only_key.jsonl",
                "key_commitments_path": key_snapshot_root / "key_commitments.json",
                "output_root": legacy_root,
            },
        )
        legacy_manifest_path = legacy_root / "eligibility_manifest.json"
        legacy_manifest = load_json(legacy_manifest_path, "legacy eligibility manifest")
        require(
            legacy_manifest.get("status")
            == "original_eligibility_and_shared_subsets_frozen_before_full_key_opening",
            "registered v1 eligibility stage did not emit its expected compatibility status",
        )
        payloads = {
            "original_eligibility": legacy_root / "original_eligibility.csv",
            "shared_capability_subsets": legacy_root / "shared_capability_subsets.csv",
        }
        staging = _fresh_staging(output_root)
        published = {
            name: staging / path.name for name, path in payloads.items()
        }
        for name in payloads:
            _copy_bytes(payloads[name], published[name])
        review_root = key_commitments_path.parent.parent.parent
        common = _common_authority_fields(
            project_root=project_root,
            amendment_root=amendment_path.parent,
            pre_metric_root=pre_metric_path.parent,
            review_root=review_root,
            amendment_seal=amendment_seal,
            pre_metric_seal=pre_metric_seal,
            registry_seal=registry_seal,
            wrapper_seal=wrapper_seal,
        )
        body = {
            "schema_version": 2,
            "protocol": CANONICAL_PROTOCOL,
            "status": ELIGIBILITY_STATUS,
            **common,
            "registered_v1_derivation": _derivation(
                project_root=project_root,
                implementation=project_root
                / "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
                entry_point="freeze_original_eligibility",
                legacy_manifest=legacy_manifest_path,
                payloads=payloads,
            ),
            "key_access": {
                "global_full_method_key_opened_before_human_canonicalization": True,
                "tier_1_original_only_key_read_by_this_stage": True,
                "tier_2_full_method_key_read_by_this_stage": False,
                "global_pre_unblinding_claim": False,
            },
            "inputs": {
                "canonical_manifest": logical_ref_from_seal(
                    "canonical_root", canonical_root, canonical_manifest_seal
                ),
                "canonical_scores": logical_ref_from_seal(
                    "canonical_root",
                    canonical_root,
                    seals[(canonical_root / "canonical_scores.jsonl").resolve()],
                ),
                "original_only_key": logical_ref_from_seal(
                    "review_root", review_root, original_key_seal
                ),
                "key_commitments": logical_ref_from_seal(
                    "review_root", review_root, key_commitments_seal
                ),
            },
            "artifacts": {
                name: logical_ref("eligibility_root", staging, path)
                for name, path in published.items()
            },
            "counts": legacy_manifest["counts"],
        }
        manifest = _sealed(body)
        require(set(manifest) == ELIGIBILITY_MANIFEST_FIELDS, "internal eligibility-v2 manifest schema drift")
        _write_json(staging / "eligibility_manifest.json", manifest)
        scan_published_tree(staging)
        validate_current_v1_sources(project_root)
        require(
            sha256_file(canonical_scores_path)
            == seals[(canonical_root / "canonical_scores.jsonl").resolve()].sha256,
            "canonical compatibility copy changed",
        )
        _assert_inputs_unchanged(seals)
        _publish(staging, output_root)
        staging = None
        return manifest
    except legacy.CanonicalizationError as exc:
        raise ProvenanceV2Error(str(exc)) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(private_parent, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-amendment")
    validate.add_argument("--amendment", type=Path, required=True)
    validate.add_argument("--pre-metric-freeze-manifest", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--amendment", type=Path, required=True)
    freeze.add_argument("--pre-metric-freeze-manifest", type=Path, required=True)
    freeze.add_argument("--review-process-receipt", type=Path, required=True)
    freeze.add_argument("--audit-root", type=Path, required=True)
    freeze.add_argument("--completed-human", type=Path, required=True)
    freeze.add_argument("--scores-a", type=Path, required=True)
    freeze.add_argument("--scores-b", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    eligibility = commands.add_parser("freeze-eligibility")
    eligibility.add_argument("--amendment", type=Path, required=True)
    eligibility.add_argument("--pre-metric-freeze-manifest", type=Path, required=True)
    eligibility.add_argument("--canonical-root", type=Path, required=True)
    eligibility.add_argument("--original-key", type=Path, required=True)
    eligibility.add_argument("--key-commitments", type=Path, required=True)
    eligibility.add_argument("--output-root", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        amendment = _resolve(project_root, args.amendment)
        pre_metric = _resolve(project_root, args.pre_metric_freeze_manifest)
        if args.command == "validate-amendment":
            amendment_value, pre_metric_value = validate_execution_authority(
                amendment, pre_metric, project_root
            )
            result = {
                "status": "validated_v2_evaluation_authority",
                "amendment_sha256": amendment_value["amendment_sha256"],
                "pre_metric_manifest_sha256": pre_metric_value["manifest_sha256"],
            }
        elif args.command == "freeze":
            result = freeze_canonical_scores_v2(
                project_root=project_root,
                amendment_path=amendment,
                pre_metric_path=pre_metric,
                review_process_receipt_path=_resolve(
                    project_root, args.review_process_receipt
                ),
                audit_root=_resolve(project_root, args.audit_root),
                completed_human_path=_resolve(project_root, args.completed_human),
                scores_a_path=_resolve(project_root, args.scores_a),
                scores_b_path=_resolve(project_root, args.scores_b),
                output_root=_resolve(project_root, args.output_root),
            )
        else:
            result = freeze_original_eligibility_v2(
                project_root=project_root,
                amendment_path=amendment,
                pre_metric_path=pre_metric,
                canonical_root=_resolve(project_root, args.canonical_root),
                original_key_path=_resolve(project_root, args.original_key),
                key_commitments_path=_resolve(project_root, args.key_commitments),
                output_root=_resolve(project_root, args.output_root),
            )
    except (OSError, ProvenanceV2Error) as exc:
        parser.exit(2, f"v2 canonicalization refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
