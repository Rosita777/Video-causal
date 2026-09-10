#!/usr/bin/env python3
"""Freeze public-only delivery and human-review process receipts.

The delivery check reads only the public audit package.  The process check is
intended to run after both independent reviewer files and adjudication exist;
it validates their schema and bindings but emits only aggregate counts, never
individual scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DELIVERY_PROTOCOL = "causal_role_erasure_7m_public_review_delivery_v2"
DELIVERY_STATUS = "isolated_public_only_delivery_frozen"
ATTESTATION_PROTOCOL = "causal_role_erasure_7m_human_review_independence_attestation_v2"
ATTESTATION_STATUS = "both_reviews_frozen_before_merge"
PROCESS_PROTOCOL = "causal_role_erasure_7m_human_review_process_receipt_v2"
PROCESS_STATUS = "independent_public_only_reviews_and_complete_adjudication_frozen"
PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
PRE_METRIC_PROTOCOL = "causal_role_erasure_7m_pre_metric_code_freeze_v2"
PRE_METRIC_STATUS = "clean_tree_components_frozen_before_human_canonical_metrics"
AUDIT_PROTOCOL = "causal_role_erasure_7m_human_canonicalization_v1"
AUDIT_STATUSES = (
    "initial_human_audit_frozen",
    "expanded_human_audit_frozen",
    "no_expansion_required_human_audit_frozen",
)
DELIVERY_FIELDS = {
    "schema_version",
    "protocol",
    "protocol_version",
    "status",
    "reviewer_id",
    "pre_metric_authority",
    "audit_manifest",
    "reviewer_instructions",
    "source_public_inventory_sha256",
    "delivery_inventory_sha256",
    "projection_mode",
    "source_queue_sha256",
    "delivery_queue_sha256",
    "file_count",
    "audit_atom_count",
    "private_files_present",
    "contains_only_frozen_public_media_and_sanitized_queue",
    "other_reviewer_labels_present",
    "receipt_sha256",
}
PRE_METRIC_AUTHORITY_FIELDS = {
    "pre_metric_code_freeze",
    "implementation_commit",
    "evaluation_environment_receipt",
}
PROCESS_FIELDS = {
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
INITIAL_PROCESS_BINDING_FIELDS = {
    "file_sha256",
    "size_bytes",
    "receipt_sha256",
    "audit_manifest_sha256",
    "audit_manifest_size_bytes",
    "completed_human_sha256",
    "completed_human_size_bytes",
}

SCORE_FIELDS = (
    "reviewer_1_score",
    "reviewer_1_notes",
    "reviewer_2_score",
    "reviewer_2_notes",
    "adjudicator_score",
    "adjudicator_notes",
)
REQUIRED_QUEUE_FIELDS = {
    "audit_id",
    "anonymous_review_id",
    "assignment_sha256",
    "field",
    "case_kind",
    "composite_path",
    "composite_sha256",
    *SCORE_FIELDS,
}
CAUSAL_FIELDS = {
    "source_visibility",
    "footprint_visibility",
    "receiver_preservation",
    "video_quality",
}
SPECIFICITY_FIELDS = {
    "protected_object_visibility",
    "noncausal_role_adherence",
    "receiver_preservation",
    "video_quality",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ReviewProcessError(ValueError):
    """The public delivery or human-review process is not fully bound."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewProcessError(message)


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


def real_directory(path: Path, label: str) -> None:
    require(path.is_dir() and not path.is_symlink(), f"{label} must be a real directory")


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def safe_relative(path: str, label: str) -> Path:
    candidate = Path(path)
    require(path != "" and not candidate.is_absolute(), f"{label} path must be relative")
    require(".." not in candidate.parts and "." not in candidate.parts, f"{label} path is unsafe")
    return candidate


def repo_ref(project_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(project_root.resolve(strict=True))
    except ValueError as exc:
        raise ReviewProcessError(f"repository artifact is outside project root: {path}") from exc
    return {
        "root_id": "repo_root",
        "path": relative.as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def named_ref(root_id: str, path: str, source: Path) -> dict[str, Any]:
    safe_relative(path, f"{root_id} artifact")
    regular_file(source, f"{root_id} artifact")
    return {
        "root_id": root_id,
        "path": path,
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def root_relative_ref(root_id: str, root: Path, source: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved_source = source.resolve(strict=True)
    try:
        relative = resolved_source.relative_to(resolved_root)
    except ValueError as exc:
        raise ReviewProcessError(f"{source} is outside {root_id}") from exc
    return named_ref(root_id, relative.as_posix(), resolved_source)


def load_pre_metric_authority(
    *, project_root: Path, pre_metric_path: Path, reviewer_instructions: Path
) -> dict[str, Any]:
    """Bind a review operation to the exact clean code/environment freeze.

    The binding proves content-consistent ordering.  It does not claim a
    trusted clock or an externally signed timestamp.
    """
    project_root = project_root.resolve(strict=True)
    pre_metric_path = pre_metric_path.resolve(strict=True)
    require(
        pre_metric_path.name == "pre_metric_code_freeze_v2.json",
        "pre-metric freeze filename changed",
    )
    value = load_json(pre_metric_path, "pre-metric code freeze")
    require(
        value.get("schema_version") == 1
        and value.get("protocol") == PRE_METRIC_PROTOCOL
        and value.get("protocol_version") == PROTOCOL_VERSION
        and value.get("status") == PRE_METRIC_STATUS,
        "pre-metric code-freeze identity/status changed",
    )
    validate_self_digest(value, "manifest_sha256", "pre-metric code freeze")
    implementation = value.get("implementation_commit")
    require(
        isinstance(implementation, dict)
        and set(implementation) == {"commit", "tree", "worktree_clean"}
        and re.fullmatch(r"[0-9a-f]{40}", str(implementation.get("commit", "")))
        is not None
        and re.fullmatch(r"[0-9a-f]{40}", str(implementation.get("tree", "")))
        is not None
        and implementation.get("worktree_clean") is True,
        "pre-metric implementation binding is invalid",
    )
    bindings = value.get("reproducibility_bindings")
    require(isinstance(bindings, dict), "pre-metric reproducibility bindings missing")
    environment_ref = bindings.get("evaluation_environment_receipt")
    require(
        isinstance(environment_ref, dict)
        and set(environment_ref) == {"root_id", "path", "sha256", "size_bytes"}
        and environment_ref.get("root_id") == "pre_metric_root"
        and environment_ref.get("path") == "evaluation_environment_receipt_v1.json"
        and HEX64.fullmatch(str(environment_ref.get("sha256", ""))) is not None
        and type(environment_ref.get("size_bytes")) is int,
        "pre-metric environment-receipt binding changed",
    )
    environment_path = pre_metric_path.parent / str(environment_ref["path"])
    regular_file(environment_path, "bound evaluation environment receipt")
    require(
        sha256_file(environment_path) == environment_ref["sha256"]
        and environment_path.stat().st_size == environment_ref["size_bytes"],
        "bound evaluation environment receipt changed",
    )
    components = value.get("components")
    require(isinstance(components, list), "pre-metric component inventory missing")
    by_role = {
        str(item.get("role")): item for item in components if isinstance(item, dict)
    }
    expected_components = {
        "review_process_receipt_builder": Path(__file__).resolve(),
        "reviewer_instructions": reviewer_instructions.resolve(strict=True),
    }
    for role, live_source in expected_components.items():
        item = by_role.get(role)
        require(isinstance(item, dict), f"pre-metric {role} component missing")
        expected_path = str(item.get("path", ""))
        safe_relative(expected_path, f"pre-metric {role}")
        if role == "reviewer_instructions":
            require(
                expected_path
                == reviewer_instructions.resolve(strict=True).relative_to(project_root).as_posix(),
                "pre-metric reviewer-instructions path changed",
            )
        frozen_source = project_root / expected_path
        regular_file(frozen_source, f"frozen {role}")
        require(
            isinstance(item, dict)
            and item.get("path") == expected_path
            and item.get("sha256") == sha256_file(frozen_source)
            and item.get("size_bytes") == frozen_source.stat().st_size
            and frozen_source.read_bytes() == live_source.read_bytes(),
            f"current {role} differs from the pre-metric freeze",
        )
    return {
        "pre_metric_code_freeze": root_relative_ref(
            "pre_metric_root", pre_metric_path.parent, pre_metric_path
        ),
        "implementation_commit": dict(implementation),
        "evaluation_environment_receipt": dict(environment_ref),
    }


def validate_pre_metric_authority(value: Any, expected: Mapping[str, Any]) -> None:
    require(
        isinstance(value, dict)
        and set(value) == PRE_METRIC_AUTHORITY_FIELDS
        and value == expected,
        "pre-metric authority binding changed",
    )


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), "output parent may not be a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
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


def with_self_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    require(field not in value, f"{field} already present")
    body = dict(value)
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return {**body, field: digest}


def walk_regular_files(root: Path) -> list[Path]:
    real_directory(root, "inventory root")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            require(not (current_path / name).is_symlink(), "inventory contains a symlinked directory")
        for name in names:
            path = current_path / name
            regular_file(path, "inventory member")
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in walk_regular_files(root)
    ]


def inventory_digest(items: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(items))).hexdigest()


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    regular_file(path, label)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} has no header")
        rows = list(reader)
        return list(reader.fieldnames), rows


def csv_bytes(header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(header))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def projected_rows(
    rows: Sequence[Mapping[str, str]], reviewer_id: str
) -> list[dict[str, str]]:
    own = {f"{reviewer_id}_score", f"{reviewer_id}_notes"}
    projected: list[dict[str, str]] = []
    for row in rows:
        value = dict(row)
        for field in SCORE_FIELDS:
            if field not in own:
                value[field] = ""
        projected.append(value)
    return projected


def projected_inventory(
    source_inventory: Sequence[Mapping[str, Any]], queue_bytes: bytes
) -> list[dict[str, Any]]:
    items = [dict(item) for item in source_inventory]
    queue_items = [item for item in items if item["path"] == "human_audit_queue.csv"]
    require(len(queue_items) == 1, "public inventory must contain one top-level human queue")
    queue_items[0]["sha256"] = hashlib.sha256(queue_bytes).hexdigest()
    queue_items[0]["size_bytes"] = len(queue_bytes)
    return items


def load_public_audit(
    audit_manifest_path: Path, public_root: Path
) -> tuple[dict[str, Any], list[str], list[dict[str, str]], list[dict[str, Any]]]:
    real_directory(public_root, "public audit root")
    require(
        public_root.resolve() == (audit_manifest_path.parent / "public").resolve(),
        "public audit root does not match the audit manifest",
    )
    audit = load_json(audit_manifest_path, "audit manifest")
    require(
        audit.get("protocol") == AUDIT_PROTOCOL
        and audit.get("status") in AUDIT_STATUSES,
        "audit manifest protocol/status mismatch",
    )
    artifacts = audit.get("artifacts")
    require(isinstance(artifacts, dict), "audit manifest artifacts missing")
    for name in ("human_queue", "media_manifest"):
        ref = artifacts.get(name)
        require(isinstance(ref, dict), f"audit manifest lacks {name}")
        relative = safe_relative(str(ref.get("path", "")), f"audit {name}")
        require(relative.parts[0] == "public", f"audit {name} is not public")
        source = audit_manifest_path.parent / relative
        regular_file(source, f"audit {name}")
        require(sha256_file(source) == ref.get("sha256"), f"audit {name} SHA mismatch")
        require(source.stat().st_size == ref.get("size_bytes"), f"audit {name} size mismatch")
        require(source.resolve().is_relative_to(public_root.resolve()), f"audit {name} is outside public root")

    queue_path = audit_manifest_path.parent / safe_relative(
        str(artifacts["human_queue"]["path"]), "human queue"
    )
    header, rows = read_csv(queue_path, "public human queue")
    require(REQUIRED_QUEUE_FIELDS <= set(header), "human queue required columns missing")
    require(len(header) == len(set(header)), "human queue header repeats a column")
    require(len({row.get("audit_id", "") for row in rows}) == len(rows), "human queue audit IDs repeat")
    require(all(row.get("audit_id", "") for row in rows), "human queue has a blank audit ID")
    for index, row in enumerate(rows):
        require(row["case_kind"] in ("causal", "specificity"), f"human queue row {index} case kind invalid")
        allowed = CAUSAL_FIELDS if row["case_kind"] == "causal" else SPECIFICITY_FIELDS
        require(row["field"] in allowed, f"human queue row {index} field/case-kind mismatch")
        require(
            HEX64.fullmatch(row["assignment_sha256"]) is not None
            and HEX64.fullmatch(row["composite_sha256"]) is not None,
            f"human queue row {index} binding SHA invalid",
        )
    if audit.get("status") == "initial_human_audit_frozen":
        require(all(row[field] == "" for row in rows for field in SCORE_FIELDS), "initial public queue is not blank")
    else:
        for index, row in enumerate(rows):
            for field in ("reviewer_1_score", "reviewer_2_score", "adjudicator_score"):
                require(row[field] in ("", "0", "1", "2"), f"expanded public queue row {index} score invalid")
            first = row["reviewer_1_score"]
            second = row["reviewer_2_score"]
            adjudicator = row["adjudicator_score"]
            require((first == "") == (second == ""), f"expanded public queue row {index} reviewer completion differs")
            if first == "":
                require(
                    all(row[field] == "" for field in SCORE_FIELDS),
                    f"expanded public queue row {index} partially prefills a new atom",
                )
            elif first == second:
                require(
                    adjudicator == "" and row["adjudicator_notes"] == "",
                    f"expanded public queue row {index} has unnecessary prior adjudication",
                )
            else:
                require(adjudicator in ("0", "1", "2"), f"expanded public queue row {index} lacks prior adjudication")
    counts = audit.get("counts")
    expected_count_key = (
        "initial_audit_atoms"
        if audit.get("status") == "initial_human_audit_frozen"
        else "total_audit_atoms"
    )
    require(isinstance(counts, dict) and counts.get(expected_count_key) == len(rows), "audit manifest/queue count mismatch")

    media_path = audit_manifest_path.parent / safe_relative(
        str(artifacts["media_manifest"]["path"]), "media manifest"
    )
    media = load_json(media_path, "media manifest")
    items = media.get("items")
    require(isinstance(items, list) and media.get("count") == len(items), "media manifest count mismatch")
    expected_paths: set[Path] = set()
    media_sha256: dict[Path, str] = {}
    for item in items:
        require(isinstance(item, dict), "media manifest item malformed")
        relative = safe_relative(str(item.get("path", "")), "media item")
        source = public_root / relative
        regular_file(source, "public audit media")
        require(sha256_file(source) == item.get("sha256"), "public audit media SHA mismatch")
        require(source.stat().st_size == item.get("size_bytes"), "public audit media size mismatch")
        require(relative not in expected_paths, "media manifest repeats a path")
        expected_paths.add(relative)
        media_sha256[relative] = str(item.get("sha256"))
    actual_media = {
        path.relative_to(public_root) for path in walk_regular_files(public_root / "media")
    }
    require(actual_media == expected_paths, "public audit media inventory differs from manifest")
    queue_paths: set[Path] = set()
    for index, row in enumerate(rows):
        relative = safe_relative(row["composite_path"], f"human queue row {index} composite")
        require(relative in expected_paths, f"human queue row {index} composite is absent from media manifest")
        require(row["composite_sha256"] == media_sha256[relative], f"human queue row {index} composite SHA mismatch")
        queue_paths.add(relative)
    require(queue_paths == expected_paths, "human queue and media manifest cover different videos")
    return audit, header, rows, inventory(public_root)


def freeze_delivery(
    *,
    project_root: Path,
    pre_metric_path: Path,
    review_root: Path,
    audit_manifest_path: Path,
    public_root: Path,
    reviewer_id: str,
    reviewer_instructions: Path,
    delivery_root: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    require(reviewer_id in ("reviewer_1", "reviewer_2"), "reviewer ID must be reviewer_1 or reviewer_2")
    pre_metric_authority = load_pre_metric_authority(
        project_root=project_root,
        pre_metric_path=pre_metric_path,
        reviewer_instructions=reviewer_instructions,
    )
    audit, header, rows, source_inventory = load_public_audit(audit_manifest_path, public_root)
    delivery_inventory = inventory(delivery_root)
    source_by_path = {item["path"]: item for item in source_inventory}
    delivery_by_path = {item["path"]: item for item in delivery_inventory}
    require(set(source_by_path) == set(delivery_by_path), "delivery file inventory differs from public audit")
    queue_relative = "human_audit_queue.csv"
    require(queue_relative in source_by_path, "public audit queue is not at the public-root top level")
    if audit.get("status") == "initial_human_audit_frozen":
        require(delivery_inventory == source_inventory, "initial delivery is not an exact public-only copy")
        projection_mode = "byte_exact_initial_public_copy"
    else:
        for relative in set(source_by_path) - {queue_relative}:
            require(source_by_path[relative] == delivery_by_path[relative], f"expanded delivery changed public artifact {relative}")
        delivered_header, delivered_rows = read_csv(delivery_root / queue_relative, "expanded reviewer delivery queue")
        require(delivered_header == header and len(delivered_rows) == len(rows), "expanded reviewer delivery queue inventory changed")
        immutable = [field for field in header if field not in SCORE_FIELDS]
        own_score = f"{reviewer_id}_score"
        own_notes = f"{reviewer_id}_notes"
        forbidden = [field for field in SCORE_FIELDS if field not in (own_score, own_notes)]
        for index, (source, delivered) in enumerate(zip(rows, delivered_rows)):
            require(all(delivered[field] == source[field] for field in immutable), f"expanded delivery row {index} binding changed")
            if source[own_score] == "":
                require(delivered[own_score] == delivered[own_notes] == "", f"expanded delivery row {index} prefills a new review")
            else:
                require(delivered[own_score] == source[own_score] and delivered[own_notes] == source[own_notes], f"expanded delivery row {index} changed the recipient's frozen prior label")
            require(all(delivered[field] == "" for field in forbidden), f"expanded delivery row {index} exposes peer/adjudicator labels")
        projection_mode = "reviewer_specific_expanded_queue_peer_and_adjudicator_columns_blank"
    require(all("private" not in Path(item["path"]).parts for item in delivery_inventory), "delivery contains a private path")
    try:
        output_receipt.resolve().relative_to(delivery_root.resolve())
    except ValueError:
        pass
    else:
        raise ReviewProcessError("delivery receipt must be outside the public-only delivery")
    receipt = with_self_digest(
        {
            "schema_version": 1,
            "protocol": DELIVERY_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": DELIVERY_STATUS,
            "reviewer_id": reviewer_id,
            "pre_metric_authority": pre_metric_authority,
            "audit_manifest": root_relative_ref("review_root", review_root, audit_manifest_path),
            "reviewer_instructions": repo_ref(project_root, reviewer_instructions),
            "source_public_inventory_sha256": inventory_digest(source_inventory),
            "delivery_inventory_sha256": inventory_digest(delivery_inventory),
            "projection_mode": projection_mode,
            "source_queue_sha256": source_by_path[queue_relative]["sha256"],
            "delivery_queue_sha256": delivery_by_path[queue_relative]["sha256"],
            "file_count": len(delivery_inventory),
            "audit_atom_count": len(rows),
            "private_files_present": 0,
            "contains_only_frozen_public_media_and_sanitized_queue": True,
            "other_reviewer_labels_present": False,
        },
        "receipt_sha256",
    )
    write_json_exclusive(output_receipt, receipt)
    return receipt


def write_csv_exclusive(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def prepare_delivery(
    *,
    project_root: Path,
    pre_metric_path: Path,
    review_root: Path,
    audit_manifest_path: Path,
    public_root: Path,
    reviewer_id: str,
    reviewer_instructions: Path,
    delivery_root: Path,
    output_receipt: Path,
) -> dict[str, Any]:
    require(reviewer_id in ("reviewer_1", "reviewer_2"), "reviewer ID must be reviewer_1 or reviewer_2")
    require(not delivery_root.exists() and not delivery_root.is_symlink(), "delivery root must be fresh")
    require(not output_receipt.exists() and not output_receipt.is_symlink(), "delivery receipt must be fresh")
    audit, header, rows, _source_inventory = load_public_audit(audit_manifest_path, public_root)
    parent = delivery_root.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    require(parent.is_dir() and not parent.is_symlink(), "delivery parent must be a real directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{delivery_root.name}.stage-", dir=parent))
    try:
        queue_relative = Path("human_audit_queue.csv")
        for source in walk_regular_files(public_root):
            relative = source.relative_to(public_root)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == queue_relative and audit.get("status") != "initial_human_audit_frozen":
                write_csv_exclusive(target, header, projected_rows(rows, reviewer_id))
            else:
                with source.open("rb") as source_handle, target.open("xb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
        receipt = freeze_delivery(
            project_root=project_root,
            pre_metric_path=pre_metric_path,
            review_root=review_root,
            audit_manifest_path=audit_manifest_path,
            public_root=public_root,
            reviewer_id=reviewer_id,
            reviewer_instructions=reviewer_instructions,
            delivery_root=stage,
            output_receipt=output_receipt,
        )
        os.replace(stage, delivery_root)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            output_receipt.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = value.get(field)
    require(
        isinstance(observed, str) and HEX64.fullmatch(observed) is not None,
        f"{label} self digest missing",
    )
    body = dict(value)
    del body[field]
    require(hashlib.sha256(canonical_json_bytes(body)).hexdigest() == observed, f"{label} self digest mismatch")


def validate_delivery_receipt(
    path: Path,
    reviewer_id: str,
    expected_pre_metric_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = load_json(path, f"{reviewer_id} delivery receipt")
    require(set(receipt) == DELIVERY_FIELDS, f"{reviewer_id} delivery receipt schema changed")
    validate_self_digest(receipt, "receipt_sha256", f"{reviewer_id} delivery receipt")
    require(
        receipt.get("protocol") == DELIVERY_PROTOCOL
        and receipt.get("protocol_version") == PROTOCOL_VERSION
        and receipt.get("status") == DELIVERY_STATUS
        and receipt.get("reviewer_id") == reviewer_id,
        f"{reviewer_id} delivery receipt contract mismatch",
    )
    require(
        isinstance(receipt.get("pre_metric_authority"), dict)
        and set(receipt["pre_metric_authority"]) == PRE_METRIC_AUTHORITY_FIELDS,
        f"{reviewer_id} delivery pre-metric authority malformed",
    )
    if expected_pre_metric_authority is not None:
        validate_pre_metric_authority(
            receipt["pre_metric_authority"], expected_pre_metric_authority
        )
    require(receipt.get("private_files_present") == 0, f"{reviewer_id} delivery includes private files")
    require(receipt.get("contains_only_frozen_public_media_and_sanitized_queue") is True, f"{reviewer_id} delivery not public-only")
    require(receipt.get("other_reviewer_labels_present") is False, f"{reviewer_id} delivery contains peer labels")
    require(
        type(receipt.get("file_count")) is int
        and receipt["file_count"] >= 3
        and type(receipt.get("audit_atom_count")) is int
        and receipt["audit_atom_count"] > 0,
        f"{reviewer_id} delivery counts invalid",
    )
    for field in (
        "source_public_inventory_sha256",
        "delivery_inventory_sha256",
        "source_queue_sha256",
        "delivery_queue_sha256",
    ):
        require(
            isinstance(receipt.get(field), str)
            and HEX64.fullmatch(receipt[field]) is not None,
            f"{reviewer_id} delivery {field} invalid",
        )
    for field in ("audit_manifest", "reviewer_instructions"):
        ref = receipt.get(field)
        require(
            isinstance(ref, dict)
            and set(ref) == {"root_id", "path", "sha256", "size_bytes"}
            and isinstance(ref.get("root_id"), str)
            and isinstance(ref.get("path"), str)
            and HEX64.fullmatch(str(ref.get("sha256", ""))) is not None
            and type(ref.get("size_bytes")) is int
            and ref["size_bytes"] >= 0,
            f"{reviewer_id} delivery {field} reference invalid",
        )
        safe_relative(ref["path"], f"{reviewer_id} delivery {field}")
    mode = receipt.get("projection_mode")
    require(mode in ("byte_exact_initial_public_copy", "reviewer_specific_expanded_queue_peer_and_adjudicator_columns_blank"), f"{reviewer_id} delivery projection mode invalid")
    if mode == "byte_exact_initial_public_copy":
        require(receipt.get("source_public_inventory_sha256") == receipt.get("delivery_inventory_sha256"), f"{reviewer_id} initial delivery inventory differs")
    return receipt


def validate_reviewer_file(
    *,
    path: Path,
    reviewer_id: str,
    header: Sequence[str],
    blank_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    observed_header, rows = read_csv(path, f"{reviewer_id} completed CSV")
    require(observed_header == list(header) and len(rows) == len(blank_rows), f"{reviewer_id} CSV inventory changed")
    own_score = f"{reviewer_id}_score"
    own_notes = f"{reviewer_id}_notes"
    forbidden = [field for field in SCORE_FIELDS if field not in (own_score, own_notes)]
    immutable = [field for field in header if field not in SCORE_FIELDS]
    for index, (before, after) in enumerate(zip(blank_rows, rows)):
        require(all(after[field] == before[field] for field in immutable), f"{reviewer_id} row {index} binding changed")
        require(after[own_score] in ("0", "1", "2"), f"{reviewer_id} row {index} score invalid")
        if before[own_score] != "":
            require(after[own_score] == before[own_score] and after[own_notes] == before[own_notes], f"{reviewer_id} row {index} changed a frozen prior label")
        require(all(after[field] == "" for field in forbidden), f"{reviewer_id} row {index} edited another role")
    return rows


def validate_attestation(
    path: Path,
    reviewer_1_sha256: str,
    reviewer_2_sha256: str,
    delivery_1_sha256: str,
    delivery_2_sha256: str,
) -> dict[str, Any]:
    value = load_json(path, "independence attestation")
    required = {
        "schema_version": 1,
        "protocol": ATTESTATION_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "status": ATTESTATION_STATUS,
        "reviewer_1_completed_sha256": reviewer_1_sha256,
        "reviewer_2_completed_sha256": reviewer_2_sha256,
        "reviewer_1_delivery_receipt_sha256": delivery_1_sha256,
        "reviewer_2_delivery_receipt_sha256": delivery_2_sha256,
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
    }
    require(value == required, "independence attestation is incomplete or has extra fields")
    return value


def bind_initial_process_receipt(
    *,
    path: Path,
    expected_pre_metric_authority: Mapping[str, Any],
    final_audit_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = load_json(path, "initial human-review process receipt")
    require(set(value) == PROCESS_FIELDS, "initial process receipt schema changed")
    validate_self_digest(value, "receipt_sha256", "initial process receipt")
    require(
        value.get("protocol") == PROCESS_PROTOCOL
        and value.get("protocol_version") == PROTOCOL_VERSION
        and value.get("status") == PROCESS_STATUS
        and value.get("audit_manifest_status") == "initial_human_audit_frozen",
        "initial process receipt identity/status changed",
    )
    validate_pre_metric_authority(
        value.get("pre_metric_authority"), expected_pre_metric_authority
    )
    require(
        value.get("review_round_ancestry")
        == {"round": "initial", "initial_process_receipt": None},
        "initial process receipt has invalid round ancestry",
    )
    inputs = value.get("inputs")
    require(isinstance(inputs, dict), "initial process receipt inputs missing")
    initial_audit = inputs.get("audit_manifest")
    initial_completed = inputs.get("completed_human")
    source_audit = final_audit_manifest.get("source_audit_manifest")
    source_completed = final_audit_manifest.get("completed_source_human_audit")
    for label, ref in (
        ("initial audit", initial_audit),
        ("initial completed human", initial_completed),
        ("final source audit", source_audit),
        ("final source completed human", source_completed),
    ):
        require(
            isinstance(ref, dict)
            and HEX64.fullmatch(str(ref.get("sha256", ""))) is not None
            and type(ref.get("size_bytes")) is int,
            f"{label} binding is invalid",
        )
    require(
        initial_audit["sha256"] == source_audit["sha256"]
        and initial_audit["size_bytes"] == source_audit["size_bytes"],
        "final audit does not bind the initial process receipt audit manifest",
    )
    require(
        initial_completed["sha256"] == source_completed["sha256"]
        and initial_completed["size_bytes"] == source_completed["size_bytes"],
        "final audit completed_source_human_audit differs from the initial process receipt",
    )
    return {
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "receipt_sha256": value["receipt_sha256"],
        "audit_manifest_sha256": initial_audit["sha256"],
        "audit_manifest_size_bytes": initial_audit["size_bytes"],
        "completed_human_sha256": initial_completed["sha256"],
        "completed_human_size_bytes": initial_completed["size_bytes"],
    }


def freeze_process(
    *,
    project_root: Path,
    pre_metric_path: Path,
    review_root: Path,
    audit_manifest_path: Path,
    public_root: Path,
    reviewer_instructions: Path,
    reviewer_1_delivery_receipt: Path,
    reviewer_2_delivery_receipt: Path,
    reviewer_1_completed: Path,
    reviewer_2_completed: Path,
    independence_attestation: Path,
    completed_human: Path,
    output_receipt: Path,
    initial_process_receipt: Path | None = None,
) -> dict[str, Any]:
    pre_metric_authority = load_pre_metric_authority(
        project_root=project_root,
        pre_metric_path=pre_metric_path,
        reviewer_instructions=reviewer_instructions,
    )
    process_root = output_receipt.parent.resolve()
    for label, path in (
        ("reviewer-1 delivery receipt", reviewer_1_delivery_receipt),
        ("reviewer-2 delivery receipt", reviewer_2_delivery_receipt),
        ("reviewer-1 completed CSV", reviewer_1_completed),
        ("reviewer-2 completed CSV", reviewer_2_completed),
        ("independence attestation", independence_attestation),
        ("completed human CSV", completed_human),
    ):
        require(path.resolve().parent == process_root, f"{label} must be a direct review-process-root child")
    audit, header, blank_rows, source_inventory = load_public_audit(audit_manifest_path, public_root)
    delivery_1 = validate_delivery_receipt(
        reviewer_1_delivery_receipt, "reviewer_1", pre_metric_authority
    )
    delivery_2 = validate_delivery_receipt(
        reviewer_2_delivery_receipt, "reviewer_2", pre_metric_authority
    )
    for field in ("audit_manifest", "reviewer_instructions", "source_public_inventory_sha256"):
        require(delivery_1.get(field) == delivery_2.get(field), f"delivery receipts disagree on {field}")
    require(delivery_1["audit_atom_count"] == delivery_2["audit_atom_count"] == len(blank_rows), "delivery atom counts differ")
    expected_audit_ref = root_relative_ref("review_root", review_root, audit_manifest_path)
    expected_instruction_ref = repo_ref(project_root, reviewer_instructions)
    require(delivery_1["audit_manifest"] == expected_audit_ref, "delivery receipts bind a different audit manifest")
    require(delivery_1["reviewer_instructions"] == expected_instruction_ref, "delivery receipts bind different reviewer instructions")

    source_inventory_sha = inventory_digest(source_inventory)
    source_queue = public_root / "human_audit_queue.csv"
    source_queue_bytes = source_queue.read_bytes()
    source_queue_sha = hashlib.sha256(source_queue_bytes).hexdigest()
    for reviewer_id, delivery in (
        ("reviewer_1", delivery_1),
        ("reviewer_2", delivery_2),
    ):
        if audit["status"] == "initial_human_audit_frozen":
            mode = "byte_exact_initial_public_copy"
            delivery_queue_bytes = source_queue_bytes
        else:
            mode = "reviewer_specific_expanded_queue_peer_and_adjudicator_columns_blank"
            delivery_queue_bytes = csv_bytes(header, projected_rows(blank_rows, reviewer_id))
        expected_delivery_inventory = projected_inventory(source_inventory, delivery_queue_bytes)
        require(
            delivery["projection_mode"] == mode
            and delivery["source_public_inventory_sha256"] == source_inventory_sha
            and delivery["source_queue_sha256"] == source_queue_sha
            and delivery["delivery_queue_sha256"]
            == hashlib.sha256(delivery_queue_bytes).hexdigest()
            and delivery["delivery_inventory_sha256"]
            == inventory_digest(expected_delivery_inventory)
            and delivery["file_count"] == len(source_inventory)
            and delivery["audit_atom_count"] == len(blank_rows),
            f"{reviewer_id} delivery receipt differs from deterministic projection",
        )

    if audit["status"] == "initial_human_audit_frozen":
        require(
            initial_process_receipt is None,
            "initial review round may not bind an earlier process receipt",
        )
        review_round_ancestry = {
            "round": "initial",
            "initial_process_receipt": None,
        }
    else:
        require(
            initial_process_receipt is not None,
            "expanded/no-expansion review round requires the initial process receipt",
        )
        binding = bind_initial_process_receipt(
            path=initial_process_receipt,
            expected_pre_metric_authority=pre_metric_authority,
            final_audit_manifest=audit,
        )
        require(
            set(binding) == INITIAL_PROCESS_BINDING_FIELDS,
            "initial process binding schema changed",
        )
        review_round_ancestry = {
            "round": "expanded_or_no_expansion",
            "initial_process_receipt": binding,
        }

    rows_1 = validate_reviewer_file(path=reviewer_1_completed, reviewer_id="reviewer_1", header=header, blank_rows=blank_rows)
    rows_2 = validate_reviewer_file(path=reviewer_2_completed, reviewer_id="reviewer_2", header=header, blank_rows=blank_rows)
    validate_attestation(
        independence_attestation,
        sha256_file(reviewer_1_completed),
        sha256_file(reviewer_2_completed),
        sha256_file(reviewer_1_delivery_receipt),
        sha256_file(reviewer_2_delivery_receipt),
    )

    merged_header, merged = read_csv(completed_human, "completed human CSV")
    require(merged_header == header and len(merged) == len(blank_rows), "completed human CSV inventory changed")
    immutable = [field for field in header if field not in SCORE_FIELDS]
    disagreements = 0
    for index, (before, first, second, row) in enumerate(zip(blank_rows, rows_1, rows_2, merged)):
        require(all(row[field] == before[field] for field in immutable), f"completed row {index} binding changed")
        for field in ("reviewer_1_score", "reviewer_1_notes"):
            require(row[field] == first[field], f"completed row {index} reviewer-1 columns changed")
        for field in ("reviewer_2_score", "reviewer_2_notes"):
            require(row[field] == second[field], f"completed row {index} reviewer-2 columns changed")
        if before["adjudicator_score"] != "":
            require(
                row["adjudicator_score"] == before["adjudicator_score"]
                and row["adjudicator_notes"] == before["adjudicator_notes"],
                f"completed row {index} changed frozen prior adjudication",
            )
        if first["reviewer_1_score"] == second["reviewer_2_score"]:
            require(row["adjudicator_score"] == "" and row["adjudicator_notes"] == "", f"completed row {index} has unnecessary adjudication")
        else:
            disagreements += 1
            require(row["adjudicator_score"] in ("0", "1", "2"), f"completed row {index} disagreement lacks adjudication")

    receipt = with_self_digest(
        {
            "schema_version": 1,
            "protocol": PROCESS_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": PROCESS_STATUS,
            "pre_metric_authority": pre_metric_authority,
            "review_round_ancestry": review_round_ancestry,
            "inputs": {
                "audit_manifest": root_relative_ref("review_root", review_root, audit_manifest_path),
                "reviewer_instructions": repo_ref(project_root, reviewer_instructions),
                "reviewer_1_delivery_receipt": root_relative_ref("review_process_root", output_receipt.parent, reviewer_1_delivery_receipt),
                "reviewer_2_delivery_receipt": root_relative_ref("review_process_root", output_receipt.parent, reviewer_2_delivery_receipt),
                "reviewer_1_completed": root_relative_ref("review_process_root", output_receipt.parent, reviewer_1_completed),
                "reviewer_2_completed": root_relative_ref("review_process_root", output_receipt.parent, reviewer_2_completed),
                "independence_attestation": root_relative_ref("review_process_root", output_receipt.parent, independence_attestation),
                "completed_human": root_relative_ref("review_process_root", output_receipt.parent, completed_human),
            },
            "counts": {
                "audit_atoms": len(merged),
                "human_agreements": len(merged) - disagreements,
                "human_disagreements": disagreements,
                "adjudicated_disagreements": disagreements,
                "unresolved_disagreements": 0,
            },
            "assertions": {
                "two_independent_reviews": True,
                "reviews_frozen_before_merge": True,
                "reviewers_received_public_only_deliveries": True,
                "reviewers_did_not_access_peer_labels": True,
                "all_disagreements_adjudicated": True,
                "process_local_answer_key_blindness_attested": True,
                "global_answer_key_blindness_claimed": False,
            },
            "audit_manifest_status": audit["status"],
        },
        "receipt_sha256",
    )
    write_json_exclusive(output_receipt, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    delivery = commands.add_parser("freeze-delivery")
    delivery.add_argument("--project-root", type=Path, required=True)
    delivery.add_argument("--pre-metric-freeze", type=Path, required=True)
    delivery.add_argument("--review-root", type=Path, required=True)
    delivery.add_argument("--audit-manifest", type=Path, required=True)
    delivery.add_argument("--public-root", type=Path, required=True)
    delivery.add_argument("--reviewer-id", choices=("reviewer_1", "reviewer_2"), required=True)
    delivery.add_argument("--reviewer-instructions", type=Path, required=True)
    delivery.add_argument("--delivery-root", type=Path, required=True)
    delivery.add_argument("--output-receipt", type=Path, required=True)

    prepare = commands.add_parser("prepare-delivery")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--pre-metric-freeze", type=Path, required=True)
    prepare.add_argument("--review-root", type=Path, required=True)
    prepare.add_argument("--audit-manifest", type=Path, required=True)
    prepare.add_argument("--public-root", type=Path, required=True)
    prepare.add_argument("--reviewer-id", choices=("reviewer_1", "reviewer_2"), required=True)
    prepare.add_argument("--reviewer-instructions", type=Path, required=True)
    prepare.add_argument("--delivery-root", type=Path, required=True)
    prepare.add_argument("--output-receipt", type=Path, required=True)

    process = commands.add_parser("freeze-process")
    process.add_argument("--project-root", type=Path, required=True)
    process.add_argument("--pre-metric-freeze", type=Path, required=True)
    process.add_argument("--review-root", type=Path, required=True)
    process.add_argument("--audit-manifest", type=Path, required=True)
    process.add_argument("--public-root", type=Path, required=True)
    process.add_argument("--reviewer-instructions", type=Path, required=True)
    process.add_argument("--reviewer-1-delivery-receipt", type=Path, required=True)
    process.add_argument("--reviewer-2-delivery-receipt", type=Path, required=True)
    process.add_argument("--reviewer-1-completed", type=Path, required=True)
    process.add_argument("--reviewer-2-completed", type=Path, required=True)
    process.add_argument("--independence-attestation", type=Path, required=True)
    process.add_argument("--completed-human", type=Path, required=True)
    process.add_argument("--initial-process-receipt", type=Path)
    process.add_argument("--output-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-delivery":
            result = freeze_delivery(
                project_root=args.project_root,
                pre_metric_path=args.pre_metric_freeze,
                review_root=args.review_root,
                audit_manifest_path=args.audit_manifest,
                public_root=args.public_root,
                reviewer_id=args.reviewer_id,
                reviewer_instructions=args.reviewer_instructions,
                delivery_root=args.delivery_root,
                output_receipt=args.output_receipt,
            )
        elif args.command == "prepare-delivery":
            result = prepare_delivery(
                project_root=args.project_root,
                pre_metric_path=args.pre_metric_freeze,
                review_root=args.review_root,
                audit_manifest_path=args.audit_manifest,
                public_root=args.public_root,
                reviewer_id=args.reviewer_id,
                reviewer_instructions=args.reviewer_instructions,
                delivery_root=args.delivery_root,
                output_receipt=args.output_receipt,
            )
        else:
            result = freeze_process(
                project_root=args.project_root,
                pre_metric_path=args.pre_metric_freeze,
                review_root=args.review_root,
                audit_manifest_path=args.audit_manifest,
                public_root=args.public_root,
                reviewer_instructions=args.reviewer_instructions,
                reviewer_1_delivery_receipt=args.reviewer_1_delivery_receipt,
                reviewer_2_delivery_receipt=args.reviewer_2_delivery_receipt,
                reviewer_1_completed=args.reviewer_1_completed,
                reviewer_2_completed=args.reviewer_2_completed,
                independence_attestation=args.independence_attestation,
                completed_human=args.completed_human,
                output_receipt=args.output_receipt,
                initial_process_receipt=args.initial_process_receipt,
            )
    except (OSError, json.JSONDecodeError, ReviewProcessError) as exc:
        parser.exit(2, f"review-process receipt refused: {exc}\n")
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
