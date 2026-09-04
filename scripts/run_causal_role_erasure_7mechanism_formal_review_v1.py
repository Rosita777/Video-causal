#!/usr/bin/env python3
"""Prepare and launch the blinded seven-mechanism formal VLM review.

The launcher joins the downloaded 2,154-video authority snapshot with the
later 294-video Wan Original snapshot, creates path-rebound (never overwritten)
manifest views, builds the 2,448-item anonymous package, and runs reviewer A
and B concurrently.  Each pass uses 16 workers, so the two-pass ceiling is 32.

Formal execution starts with ten realistic items per pass.  Only after both
smokes are schema-valid does it run fresh ordered shards, retry infrastructure
failures in new roots, and merge immutable checkpoints.  The Copilot proxy key
is read into memory from a local file and is never placed in argv, the process
environment, a log, a registration, or a child-process command line.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import secrets
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import build_causal_role_erasure_7mechanism_review_package_v1 as package
    import review_causal_role_erasure_7mechanism_formal_v2 as transport
except ModuleNotFoundError:  # imported as ``scripts.<module>`` in tests
    from scripts import build_causal_role_erasure_7mechanism_review_package_v1 as package
    from scripts import review_causal_role_erasure_7mechanism_formal_v2 as transport


WORKFLOW_VERSION = "causal_role_erasure_7mechanism_formal_review_launch_v1"
REMOTE_PROJECT_ROOT = Path("/data/xiaohuang_workspace/ljc/Video-causal-v4")
LOCAL_RESPONSES_BASE_URL = "http://127.0.0.1:4141"
LOCAL_RESPONSES_ENDPOINT = "http://127.0.0.1:4141/v1/responses"
PASS_WORKERS = 16
TOTAL_CONCURRENCY = 32
SMOKE_ITEMS = 10
DEFAULT_SHARD_SIZE = 256
DEFAULT_MAX_RETRY_ROUNDS = 3
TOTAL_ITEMS_PER_PASS = 2448
DEFAULT_PREFLIGHT_ITEMS = 20
PREFLIGHT_STATUS = "successful_copilot_responses_vision_structured_preflight"

FORMAL_CASES_REL = Path("data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv")
IDENTIFICATION_REL = Path(
    "data/causal_role_erasure_7mechanism_main_v2/identification_subset.csv"
)
TRAINED_WAN_MANIFEST_REL = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_generation_manifest.json"
)
COG_CORE_MANIFEST_REL = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2/baseline_generation_manifest.json"
)
SAFREE_MANIFEST_REL = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2/baseline_generation_manifest.json"
)
WAN_ORIGINAL_MANIFEST_REL = Path(
    "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2/wan_original_generation_manifest.json"
)

MANIFEST_SPECS = {
    "trained_wan": ("authority", TRAINED_WAN_MANIFEST_REL, 684),
    "cog_core": ("authority", COG_CORE_MANIFEST_REL, 1176),
    "safree": ("authority", SAFREE_MANIFEST_REL, 294),
    "wan_original": ("wan_original", WAN_ORIGINAL_MANIFEST_REL, 294),
}

ReviewRunner = Callable[..., dict[str, Any]]
Merger = Callable[..., dict[str, Any]]
PackageBuilder = Callable[..., dict[str, Any]]
ProxyProbe = Callable[[str, str, int], dict[str, Any]]


class FormalLaunchError(RuntimeError):
    """The local snapshots or launch state violate the formal protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalLaunchError(message)


def _regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")


def _directory(path: Path, label: str) -> Path:
    require(path.is_dir() and not path.is_symlink(), f"{label} missing or symlinked: {path}")
    return path.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalLaunchError(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def _safe_under(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FormalLaunchError(f"{label} escaped its snapshot root") from exc
    return resolved


def _snapshot_file(root: Path, relative: Path, label: str) -> Path:
    root = _directory(root, f"{label} snapshot")
    candidate = root / relative
    _regular_file(candidate, label)
    return _safe_under(root, candidate, label)


def formal_shards(shard_size: int = DEFAULT_SHARD_SIZE) -> list[dict[str, int | str]]:
    require(shard_size > 0, "shard_size must be positive")
    shards: list[dict[str, int | str]] = []
    start = SMOKE_ITEMS
    index = 0
    while start < TOTAL_ITEMS_PER_PASS:
        limit = min(shard_size, TOTAL_ITEMS_PER_PASS - start)
        shards.append({"name": f"full_{index:03d}", "start": start, "limit": limit})
        start += limit
        index += 1
    require(
        SMOKE_ITEMS + sum(int(row["limit"]) for row in shards)
        == TOTAL_ITEMS_PER_PASS,
        "internal shard coverage differs from 2,448",
    )
    return shards


def inspect_snapshot_readiness(
    *, authority_snapshot: Path, wan_original_snapshot: Path
) -> dict[str, Any]:
    expected = {
        "formal_cases": (authority_snapshot, FORMAL_CASES_REL),
        "identification_subset": (authority_snapshot, IDENTIFICATION_REL),
        "trained_wan_manifest": (authority_snapshot, TRAINED_WAN_MANIFEST_REL),
        "cog_core_manifest": (authority_snapshot, COG_CORE_MANIFEST_REL),
        "safree_manifest": (authority_snapshot, SAFREE_MANIFEST_REL),
        "wan_original_manifest": (wan_original_snapshot, WAN_ORIGINAL_MANIFEST_REL),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for label, (root, relative) in expected.items():
        candidate = root / relative
        ready = (
            root.is_dir()
            and not root.is_symlink()
            and candidate.is_file()
            and not candidate.is_symlink()
        )
        artifacts[label] = {
            "snapshot": str(root),
            "relative_path": relative.as_posix(),
            "ready": ready,
        }
        if not ready:
            missing.append(label)
    return {
        "ready": not missing,
        "missing": missing,
        "artifacts": artifacts,
    }


def _public_root_from_copy(path: Path) -> Path:
    candidate = _directory(path, "copied public review package")
    if candidate.name == "public":
        public_root = candidate
    else:
        public_root = _directory(candidate / "public", "copied package public root")
    return public_root


def validate_public_package_copy(
    path: Path, *, expected_items: int = TOTAL_ITEMS_PER_PASS
) -> tuple[Path, dict[str, Any]]:
    """Validate the copied public-only package without opening any private key."""

    public_root = _public_root_from_copy(path)
    require(
        {entry.name for entry in public_root.iterdir()}
        == {"pass_a", "pass_b", "key_commitments.json"},
        "copied public package inventory differs",
    )
    by_pass: dict[str, list[dict[str, Any]]] = {}
    manifest_refs: dict[str, Any] = {}
    for pass_name in transport.PASS_NAMES:
        pass_root = public_root / pass_name
        manifest, rows = transport.load_public_pass(
            pass_root / "assignments.jsonl",
            pass_root / "scores.jsonl",
            expected_items=expected_items,
        )
        by_pass[pass_name] = rows
        manifest_refs[pass_name] = transport.file_ref(manifest["path"])
    inventory_by_pass = {
        pass_name: {
            row["assignment"]["anonymous_review_id"]: {
                key: value
                for key, value in row["assignment"].items()
                if key != "composite_path"
            }
            for row in rows
        }
        for pass_name, rows in by_pass.items()
    }
    require(
        inventory_by_pass["pass_a"] == inventory_by_pass["pass_b"],
        "copied A/B public semantic inventories differ",
    )
    orders = {
        pass_name: [
            row["assignment"]["anonymous_review_id"] for row in rows
        ]
        for pass_name, rows in by_pass.items()
    }
    require(orders["pass_a"] != orders["pass_b"], "copied A/B orders are not independent")
    commitments_path = public_root / "key_commitments.json"
    commitments = _load_json(commitments_path, "public key commitments")
    require(
        commitments.get("protocol") == package.PACKAGE_PROTOCOL
        and commitments.get("schema_version") == 1
        and commitments.get("commitment_scheme")
        == "sha256(canonical-jsonl-bytes)"
        and commitments.get("tier_0_audit_strata", {}).get("row_count")
        == expected_items
        and commitments.get("tier_1_original_only", {}).get("row_count") == 588
        and commitments.get("tier_2_full", {}).get("row_count")
        == expected_items
        and commitments.get("generation_ledger", {}).get("row_count")
        == expected_items,
        "copied public key commitments differ from the frozen protocol",
    )
    for name in (
        "tier_0_audit_strata",
        "tier_1_original_only",
        "tier_2_full",
        "generation_ledger",
    ):
        sha = commitments[name].get("sha256")
        require(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha),
            f"copied public {name} commitment is invalid",
        )
    blind_sha = commitments.get("blind_key_sha256")
    require(
        isinstance(blind_sha, str) and re.fullmatch(r"[0-9a-f]{64}", blind_sha),
        "copied public blind-key commitment is invalid",
    )
    inventory = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "validated_copied_public_only_package",
        "public_root": str(public_root),
        "item_count_per_pass": expected_items,
        "pass_manifests": manifest_refs,
        "key_commitments": transport.file_ref(commitments_path),
        "semantic_inventory_sha256": transport.object_sha256(
            inventory_by_pass["pass_a"]
        ),
        "orders_independent": True,
        "private_files_opened": 0,
    }
    return public_root, inventory


def build_dry_run_plan(
    *,
    authority_snapshot: Path | None,
    wan_original_snapshot: Path | None,
    review_package_public: Path | None,
    launch_root: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
    proxy_version: str = "copilot-claude-local",
) -> dict[str, Any]:
    shards = formal_shards(shard_size)
    using_copy = review_package_public is not None
    require(
        using_copy
        or (authority_snapshot is not None and wan_original_snapshot is not None),
        "dry run requires a copied public package or both snapshots",
    )
    require(
        not using_copy
        or (authority_snapshot is None and wan_original_snapshot is None),
        "choose copied public package or snapshot-build mode, not both",
    )
    if using_copy:
        assert review_package_public is not None
        public_candidate = (
            review_package_public
            if review_package_public.name == "public"
            else review_package_public / "public"
        )
        required = [
            public_candidate / "pass_a" / "assignments.jsonl",
            public_candidate / "pass_a" / "scores.jsonl",
            public_candidate / "pass_b" / "assignments.jsonl",
            public_candidate / "pass_b" / "scores.jsonl",
            public_candidate / "key_commitments.json",
        ]
        missing = [str(path) for path in required if not path.is_file() or path.is_symlink()]
        readiness = {
            "ready": not missing,
            "missing": missing,
            "mode": "copied_public_package",
        }
    else:
        assert authority_snapshot is not None and wan_original_snapshot is not None
        readiness = inspect_snapshot_readiness(
            authority_snapshot=authority_snapshot,
            wan_original_snapshot=wan_original_snapshot,
        )
        readiness["mode"] = "snapshot_build"
    return {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": (
            "dry_run_ready_no_api_calls"
            if readiness["ready"]
            else "dry_run_waiting_for_snapshot_no_api_calls"
        ),
        "source_mode": readiness["mode"],
        "authority_snapshot": (
            None if authority_snapshot is None else str(authority_snapshot)
        ),
        "wan_original_snapshot": (
            None if wan_original_snapshot is None else str(wan_original_snapshot)
        ),
        "review_package_public": (
            None if review_package_public is None else str(review_package_public)
        ),
        "launch_root": str(launch_root),
        "snapshot_readiness": readiness,
        "package": {
            "expected_main_videos": 2352,
            "expected_identification_videos": 96,
            "expected_total_videos": TOTAL_ITEMS_PER_PASS,
            "anonymous_passes": ["A", "B"],
            "private_keys_opened_during_review": False,
        },
        "transport": {
            "endpoint": LOCAL_RESPONSES_ENDPOINT,
            "model": transport.MODEL,
            "reasoning_effort": transport.REASONING_EFFORT,
            "temperature": transport.TEMPERATURE,
            "max_output_tokens": transport.MAX_OUTPUT_TOKENS,
            "store": False,
            "proxy_version": proxy_version,
            "workers_per_pass": PASS_WORKERS,
            "simultaneous_passes": 2,
            "maximum_total_concurrency": TOTAL_CONCURRENCY,
            "key_transport": "memory_to_isolated_child_stdin_only",
        },
        "smoke": {"start": 0, "limit": SMOKE_ITEMS, "both_passes": True},
        "full_shards": shards,
        "primary_requests_per_pass": TOTAL_ITEMS_PER_PASS,
        "primary_requests_total": TOTAL_ITEMS_PER_PASS * 2,
        "failure_policy": {
            "scientific_zero_fallbacks": 0,
            "fresh_attempt_root_after_interruption": True,
            "fresh_retry_root_for_infrastructure_errors": True,
            "merge_only_after_exact_complete_pass": True,
        },
        "dry_run": True,
        "api_calls": 0,
    }


def _remap_remote_file(
    *, remote_value: Any, snapshot_root: Path, remote_project_root: Path, label: str
) -> Path:
    remote = Path(str(remote_value))
    require(remote.is_absolute(), f"{label} was not an absolute authority path")
    try:
        relative = remote.relative_to(remote_project_root)
    except ValueError as exc:
        raise FormalLaunchError(f"{label} escaped the frozen remote project root") from exc
    local = snapshot_root / relative
    _regular_file(local, f"relocated {label}")
    return _safe_under(snapshot_root, local, f"relocated {label}")


def rebound_manifest(
    *,
    source_manifest: Path,
    source_snapshot: Path,
    authority_snapshot: Path,
    remote_project_root: Path = REMOTE_PROJECT_ROOT,
    expected_items: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a derivative manifest changing only frozen local path fields."""

    original = _load_json(source_manifest, "source generation manifest")
    rebound = copy.deepcopy(original)
    items = rebound.get("items")
    require(isinstance(items, list) and len(items) == expected_items, "source manifest item count differs")
    rewrites: list[dict[str, str]] = []
    for index, item in enumerate(items):
        require(isinstance(item, dict) and "video_path" in item, f"manifest item {index} lacks video_path")
        remote = str(item["video_path"])
        local = _remap_remote_file(
            remote_value=remote,
            snapshot_root=source_snapshot,
            remote_project_root=remote_project_root,
            label=f"item {index} video_path",
        )
        declared_size = item.get("size_bytes")
        require(type(declared_size) is int and local.stat().st_size == declared_size, f"item {index} relocated size differs")
        item["video_path"] = str(local)
        rewrites.append({"field": f"items[{index}].video_path", "from": remote, "to": str(local)})

    inputs = rebound.get("inputs")
    require(isinstance(inputs, dict), "source manifest lacks inputs")
    formal_path = _snapshot_file(
        authority_snapshot, FORMAL_CASES_REL, "formal cases"
    )
    old_formal = str(inputs.get("formal_cases", ""))
    require(Path(old_formal).is_absolute(), "manifest formal_cases path is not absolute")
    _remap_remote_file(
        remote_value=old_formal,
        snapshot_root=authority_snapshot,
        remote_project_root=remote_project_root,
        label="formal_cases",
    )
    inputs["formal_cases"] = str(formal_path)
    rewrites.append({"field": "inputs.formal_cases", "from": old_formal, "to": str(formal_path)})
    if "identification_subset" in inputs:
        identification_path = _snapshot_file(
            authority_snapshot, IDENTIFICATION_REL, "identification subset"
        )
        old_identification = str(inputs["identification_subset"])
        _remap_remote_file(
            remote_value=old_identification,
            snapshot_root=authority_snapshot,
            remote_project_root=remote_project_root,
            label="identification_subset",
        )
        inputs["identification_subset"] = str(identification_path)
        rewrites.append(
            {
                "field": "inputs.identification_subset",
                "from": old_identification,
                "to": str(identification_path),
            }
        )
    receipt = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "path_rebound_view_only_original_unchanged",
        "source_manifest": transport.file_ref(source_manifest),
        "source_snapshot": str(source_snapshot.resolve()),
        "remote_project_root": str(remote_project_root),
        "item_count": len(items),
        "rewrite_count": len(rewrites),
        "rewrites_sha256": transport.object_sha256(rewrites),
        "allowed_mutations": ["items[*].video_path", "inputs.formal_cases", "inputs.identification_subset"],
    }
    return rebound, receipt


def _write_or_verify_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    raw = transport.canonical_json_bytes(value)
    if path.exists() or path.is_symlink():
        _regular_file(path, "existing launch artifact")
        require(path.read_bytes() == raw, f"existing launch artifact differs: {path}")
        return
    transport.write_bytes_exclusive(path, raw, mode=mode)


def prepare_rebound_manifests(
    *,
    authority_snapshot: Path,
    wan_original_snapshot: Path,
    output_root: Path,
    remote_project_root: Path = REMOTE_PROJECT_ROOT,
) -> dict[str, Path]:
    authority_snapshot = _directory(authority_snapshot, "authority snapshot")
    wan_original_snapshot = _directory(wan_original_snapshot, "Wan Original snapshot")
    output_root.mkdir(parents=True, exist_ok=True)
    require(output_root.is_dir() and not output_root.is_symlink(), "rebound manifest root is unsafe")
    rebound_paths: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    for label, (snapshot_name, relative, expected_items) in MANIFEST_SPECS.items():
        snapshot = (
            authority_snapshot
            if snapshot_name == "authority"
            else wan_original_snapshot
        )
        source = _snapshot_file(snapshot, relative, f"{label} manifest")
        rebound, receipt = rebound_manifest(
            source_manifest=source,
            source_snapshot=snapshot,
            authority_snapshot=authority_snapshot,
            remote_project_root=remote_project_root,
            expected_items=expected_items,
        )
        target = output_root / f"{label}.json"
        _write_or_verify_json(target, rebound)
        receipt = {**receipt, "rebound_manifest": transport.file_ref(target)}
        receipt_path = output_root / f"{label}.relocation_receipt.json"
        _write_or_verify_json(receipt_path, receipt)
        rebound_paths[label] = target
        receipts[label] = transport.file_ref(receipt_path)
    registry = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "four_path_rebound_manifest_views_frozen",
        "manifests": {
            label: transport.file_ref(path) for label, path in rebound_paths.items()
        },
        "receipts": receipts,
    }
    _write_or_verify_json(output_root / "registry.json", registry)
    return rebound_paths


def read_copilot_key(path: Path) -> str:
    """Read the proxy credential without copying it into process-global state."""

    _regular_file(path, "Copilot key file")
    require(path.stat().st_size <= 65536, "Copilot key file is unexpectedly large")
    raw = path.read_text(encoding="utf-8").strip()
    require(raw, "Copilot key file is empty")
    value: Any = None
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FormalLaunchError("Copilot key JSON is invalid") from exc
        require(isinstance(payload, dict), "Copilot key JSON is not an object")
        value = payload.get("key", payload.get("api_key"))
    else:
        candidates = []
        for line in raw.splitlines():
            if ":" in line:
                name, candidate = line.split(":", 1)
                if name.strip() in {"key", "api_key"}:
                    candidates.append(candidate.strip())
        value = candidates[0] if len(candidates) == 1 else raw
    require(isinstance(value, str) and value and not any(char.isspace() for char in value), "Copilot key is malformed")
    return value


def _http_get_json(url: str, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
        },
        method="GET",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
            headers = {key.casefold(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        raise FormalLaunchError(f"proxy probe returned HTTP {exc.code} for {url}") from None
    except Exception as exc:
        message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        raise FormalLaunchError(
            f"proxy probe failed for {url}: {type(exc).__name__}: {message}"
        ) from None
    require(status == 200, f"proxy probe returned HTTP {status} for {url}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalLaunchError(f"proxy probe returned invalid JSON for {url}") from exc
    require(isinstance(payload, dict), f"proxy probe result is not an object for {url}")
    return {
        "http_status": status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "body_sha256": transport.object_sha256(payload),
        "headers": headers,
        "payload": payload,
    }


def _recursive_values(value: Any, wanted_key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == wanted_key:
                found.append(item)
            found.extend(_recursive_values(item, wanted_key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_values(item, wanted_key))
    return found


def _metadata_has_true(metadata: Mapping[str, Any], key: str) -> bool:
    return any(value is True for value in _recursive_values(metadata, key))


def _metadata_maximum(metadata: Mapping[str, Any], key: str) -> int:
    values = [
        value
        for value in _recursive_values(metadata, key)
        if type(value) is int
    ]
    return max(values, default=-1)


def validate_luna_metadata(models_payload: Mapping[str, Any]) -> dict[str, Any]:
    data = models_payload.get("data")
    require(isinstance(data, list), "/v1/models payload has no data list")
    candidates = [
        item
        for item in data
        if isinstance(item, dict) and item.get("id") == transport.MODEL
    ]
    require(len(candidates) == 1, "/v1/models does not contain exactly one gpt-5.6-luna")
    metadata = candidates[0]
    endpoint_values = _recursive_values(metadata, "supported_endpoints")
    endpoints = {
        str(endpoint)
        for value in endpoint_values
        if isinstance(value, list)
        for endpoint in value
    }
    reasoning_values = [
        *_recursive_values(metadata, "reasoning_efforts"),
        *_recursive_values(metadata, "supported_reasoning_efforts"),
    ]
    efforts = {
        str(effort)
        for value in reasoning_values
        if isinstance(value, list)
        for effort in value
    }
    max_images = _metadata_maximum(metadata, "max_images")
    max_image_size = max(
        _metadata_maximum(metadata, "max_image_size"),
        _metadata_maximum(metadata, "max_image_bytes"),
    )
    structured = _metadata_has_true(
        metadata, "structured_outputs"
    ) or _metadata_has_true(metadata, "supports_structured_outputs")
    require(
        bool({"/responses", "/v1/responses", "responses"} & endpoints),
        "Luna metadata does not authorize Responses",
    )
    require(max_images >= 1, "Luna metadata does not authorize one image")
    require(
        max_image_size >= transport.MAX_IMAGE_BYTES,
        "Luna image-size capability is below the frozen composite limit",
    )
    require(structured, "Luna metadata does not authorize structured outputs")
    require("low" in efforts, "Luna metadata does not authorize low reasoning")
    return {
        "model": transport.MODEL,
        "metadata_sha256": transport.object_sha256(metadata),
        "responses_supported": True,
        "vision_supported": True,
        "max_images": max_images,
        "max_image_size": max_image_size,
        "structured_outputs": True,
        "low_reasoning_supported": True,
    }


def probe_local_proxy(base_url: str, api_key: str, timeout: int) -> dict[str, Any]:
    require(
        transport.responses_endpoint(base_url) == LOCAL_RESPONSES_ENDPOINT,
        "preflight proxy endpoint differs from frozen localhost endpoint",
    )
    models = _http_get_json(base_url.rstrip("/") + "/v1/models", api_key, timeout)
    luna = validate_luna_metadata(models["payload"])
    health: dict[str, Any]
    try:
        observed = _http_get_json(base_url.rstrip("/") + "/health", api_key, timeout)
        payload = observed["payload"]
        observed_version = payload.get("version")
        if not isinstance(observed_version, str):
            observed_version = observed["headers"].get("x-proxy-version")
        health = {
            "status": "healthy_via_health_endpoint",
            "http_status": observed["http_status"],
            "latency_ms": observed["latency_ms"],
            "body_sha256": observed["body_sha256"],
            "observed_version": observed_version,
        }
    except FormalLaunchError:
        health = {
            "status": "healthy_via_models_endpoint",
            "http_status": models["http_status"],
            "latency_ms": models["latency_ms"],
            "body_sha256": models["body_sha256"],
            "observed_version": models["headers"].get("x-proxy-version"),
        }
    return {
        "health": health,
        "models_http_status": models["http_status"],
        "models_latency_ms": models["latency_ms"],
        "models_body_sha256": models["body_sha256"],
        "luna": luna,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    require(values, "cannot calculate latency percentile without measurements")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return round(float(ordered[index]), 3)


def run_preflight(
    *,
    review_package_public: Path,
    output_root: Path,
    copilot_key_file: Path,
    proxy_version: str,
    pass_name: str = "pass_a",
    start: int = 0,
    limit: int = DEFAULT_PREFLIGHT_ITEMS,
    workers: int = PASS_WORKERS,
    timeout: int = 180,
    expected_items: int = TOTAL_ITEMS_PER_PASS,
    proxy_probe: ProxyProbe = probe_local_proxy,
    review_runner: ReviewRunner = transport.run_review,
    api_transport: transport.Transport = transport.isolated_urllib_transport,
) -> dict[str, Any]:
    require(pass_name in transport.PASS_NAMES, "preflight pass is invalid")
    require(1 <= workers <= TOTAL_CONCURRENCY, "preflight workers exceed the measured ceiling")
    require(limit > 0 and start >= 0 and start + limit <= expected_items, "preflight shard is invalid")
    require(not output_root.exists() and not output_root.is_symlink(), "preflight output must be fresh")
    public_root, inventory = validate_public_package_copy(
        review_package_public, expected_items=expected_items
    )
    api_key = read_copilot_key(copilot_key_file)
    probe = proxy_probe(LOCAL_RESPONSES_BASE_URL, api_key, timeout)
    require(
        probe.get("health", {}).get("status")
        in {"healthy_via_health_endpoint", "healthy_via_models_endpoint"}
        and probe.get("models_http_status") == 200
        and probe.get("luna", {}).get("metadata_sha256"),
        "proxy capability probe did not pass",
    )
    measurements: list[dict[str, Any]] = []
    measurement_lock = threading.Lock()

    def measured_api_transport(
        url: str, key: str, payload: dict[str, Any], request_timeout: int
    ) -> dict[str, Any]:
        require(key == api_key, "preflight runner changed the in-memory credential")
        started = time.perf_counter()
        try:
            response = api_transport(url, key, payload, request_timeout)
        except Exception as exc:
            code = getattr(exc, "code", None)
            with measurement_lock:
                measurements.append(
                    {
                        "ok": False,
                        "http_status": code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    }
                )
            raise
        with measurement_lock:
            measurements.append(
                {
                    "ok": True,
                    "http_status": 200,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
        return response

    run_root = output_root / "transport_run"
    try:
        summary = review_runner(
            assignments_path=public_root / pass_name / "assignments.jsonl",
            blank_scores_path=public_root / pass_name / "scores.jsonl",
            output_root=run_root,
            dry_run=False,
            start=start,
            limit=limit,
            workers=workers,
            timeout=timeout,
            base_url=LOCAL_RESPONSES_BASE_URL,
            api_key=api_key,
            proxy_version=proxy_version,
            retry_run_roots=(),
            expected_items=expected_items,
            transport=measured_api_transport,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        raise FormalLaunchError(
            f"preflight transport failed: {type(exc).__name__}: {message}"
        ) from None
    require(len(measurements) == limit, "preflight measurement count differs from requested items")
    successes = sum(item["ok"] is True for item in measurements)
    latencies = [float(item["latency_ms"]) for item in measurements]
    http_counts: dict[str, int] = {}
    for item in measurements:
        name = "none" if item["http_status"] is None else str(item["http_status"])
        http_counts[name] = http_counts.get(name, 0) + 1
    require(
        summary.get("successful_checkpoints") == limit
        and summary.get("infrastructure_errors") == 0
        and summary.get("scientific_zero_fallbacks") == 0
        and successes == limit,
        "preflight review shard was not 100% schema-valid",
    )
    receipt = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": PREFLIGHT_STATUS,
        "public_package_inventory": inventory,
        "proxy": probe,
        "configured_proxy_version": proxy_version,
        "endpoint": LOCAL_RESPONSES_ENDPOINT,
        "model": transport.MODEL,
        "reasoning_effort": transport.REASONING_EFFORT,
        "temperature": transport.TEMPERATURE,
        "max_output_tokens": transport.MAX_OUTPUT_TOKENS,
        "store": False,
        "structured_outputs": True,
        "vision_input": True,
        "pass_name": pass_name,
        "start": start,
        "limit": limit,
        "workers": workers,
        "attempted": len(measurements),
        "successful": successes,
        "success_rate": successes / len(measurements),
        "http_status_counts": http_counts,
        "latency_ms": {
            "minimum": round(min(latencies), 3),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "maximum": round(max(latencies), 3),
        },
        "transport_run_summary": transport.file_ref(
            run_root / "run_summary.json"
        ),
        "credential_persisted": False,
    }
    serialized = json.dumps(receipt, ensure_ascii=False)
    require(api_key not in serialized, "credential leaked into preflight receipt")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(output_root / "preflight_receipt.json", receipt)
    return receipt


def validate_preflight_receipt(
    *, path: Path, public_inventory: Mapping[str, Any], proxy_version: str
) -> dict[str, Any]:
    receipt = _load_json(path, "formal preflight receipt")
    require(
        receipt.get("workflow_version") == WORKFLOW_VERSION
        and receipt.get("status") == PREFLIGHT_STATUS
        and receipt.get("endpoint") == LOCAL_RESPONSES_ENDPOINT
        and receipt.get("model") == transport.MODEL
        and receipt.get("reasoning_effort") == transport.REASONING_EFFORT
        and receipt.get("temperature") == transport.TEMPERATURE
        and receipt.get("max_output_tokens") == transport.MAX_OUTPUT_TOKENS
        and receipt.get("store") is False
        and receipt.get("structured_outputs") is True
        and receipt.get("vision_input") is True
        and receipt.get("configured_proxy_version") == proxy_version
        and receipt.get("attempted") == receipt.get("successful")
        and receipt.get("success_rate") == 1.0
        and receipt.get("credential_persisted") is False,
        "formal preflight receipt is not successful or configuration-matched",
    )
    bound_inventory = receipt.get("public_package_inventory")
    require(
        isinstance(bound_inventory, dict)
        and bound_inventory.get("semantic_inventory_sha256")
        == public_inventory.get("semantic_inventory_sha256")
        and bound_inventory.get("pass_manifests")
        == public_inventory.get("pass_manifests")
        and bound_inventory.get("key_commitments")
        == public_inventory.get("key_commitments"),
        "preflight receipt binds a different copied public package",
    )
    luna = receipt.get("proxy", {}).get("luna", {})
    require(
        luna.get("responses_supported") is True
        and luna.get("vision_supported") is True
        and luna.get("structured_outputs") is True
        and luna.get("low_reasoning_supported") is True
        and luna.get("max_images", 0) >= 1
        and luna.get("max_image_size", 0) >= transport.MAX_IMAGE_BYTES,
        "preflight receipt lacks Luna Responses/vision/structured capability",
    )
    return receipt


def ensure_blind_key(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        _regular_file(path, "blind key")
        require(len(path.read_bytes()) >= 32, "existing blind key is too short")
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    transport.write_bytes_exclusive(path, secrets.token_bytes(64), mode=0o600)
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, "blind key mode is not 0600")
    return path.resolve(strict=True)


def _require_under_workspace(workspace_root: Path, path: Path, label: str) -> None:
    root = workspace_root.resolve(strict=True)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FormalLaunchError(f"{label} is outside workspace_root") from exc


def prepare_review_package(
    *,
    workspace_root: Path,
    authority_snapshot: Path,
    wan_original_snapshot: Path,
    launch_root: Path,
    blind_key_path: Path | None = None,
    package_builder: PackageBuilder = package.build_review_package,
) -> Path:
    workspace_root = _directory(workspace_root, "workspace root")
    authority_snapshot = _directory(authority_snapshot, "authority snapshot")
    wan_original_snapshot = _directory(wan_original_snapshot, "Wan Original snapshot")
    for label, path in (
        ("authority snapshot", authority_snapshot),
        ("Wan Original snapshot", wan_original_snapshot),
        ("launch root", launch_root),
    ):
        _require_under_workspace(workspace_root, path, label)
    readiness = inspect_snapshot_readiness(
        authority_snapshot=authority_snapshot,
        wan_original_snapshot=wan_original_snapshot,
    )
    require(readiness["ready"], f"snapshots are incomplete: {', '.join(readiness['missing'])}")
    launch_root.mkdir(parents=True, exist_ok=True)
    require(launch_root.is_dir() and not launch_root.is_symlink(), "launch root is unsafe")
    package_root = launch_root / "review_package"
    if package_root.exists() or package_root.is_symlink():
        _directory(package_root, "existing review package")
        for pass_name in transport.PASS_NAMES:
            pass_root = package_root / "public" / pass_name
            transport.load_public_pass(
                pass_root / "assignments.jsonl",
                pass_root / "scores.jsonl",
                expected_items=TOTAL_ITEMS_PER_PASS,
            )
        return package_root.resolve(strict=True)

    rebound = prepare_rebound_manifests(
        authority_snapshot=authority_snapshot,
        wan_original_snapshot=wan_original_snapshot,
        output_root=launch_root / "rebound_manifests",
    )
    blind_key = ensure_blind_key(
        blind_key_path or (launch_root / "private" / "review_blind.key")
    )
    receipt = package_builder(
        project_root=workspace_root,
        formal_cases_path=_snapshot_file(
            authority_snapshot, FORMAL_CASES_REL, "formal cases"
        ),
        identification_path=_snapshot_file(
            authority_snapshot, IDENTIFICATION_REL, "identification subset"
        ),
        wan_original_manifest_path=rebound["wan_original"],
        trained_wan_manifest_path=rebound["trained_wan"],
        cog_core_manifest_path=rebound["cog_core"],
        safree_manifest_path=rebound["safree"],
        blind_key_path=blind_key,
        output_dir=package_root,
        upstream_manifest_paths={
            label: _snapshot_file(
                authority_snapshot if snapshot_name == "authority" else wan_original_snapshot,
                relative,
                f"{label} upstream source manifest",
            )
            for label, (snapshot_name, relative, _expected_items) in MANIFEST_SPECS.items()
        },
        upstream_rebase_roots={
            label: (
                authority_snapshot
                if snapshot_name == "authority"
                else wan_original_snapshot
            )
            for label, (snapshot_name, _relative, _expected_items) in MANIFEST_SPECS.items()
        },
        registered_project_root=REMOTE_PROJECT_ROOT,
    )
    require(
        receipt.get("status")
        == "frozen_after_exact_2448_media_and_blinding_validation"
        and receipt.get("total_video_count") == TOTAL_ITEMS_PER_PASS,
        "review package builder did not return the frozen 2,448-item receipt",
    )
    for pass_name in transport.PASS_NAMES:
        pass_root = package_root / "public" / pass_name
        transport.load_public_pass(
            pass_root / "assignments.jsonl",
            pass_root / "scores.jsonl",
            expected_items=TOTAL_ITEMS_PER_PASS,
        )
    return package_root.resolve(strict=True)


def _completed_attempt(stage_root: Path) -> tuple[Path, dict[str, Any]] | None:
    if not stage_root.exists():
        return None
    require(stage_root.is_dir() and not stage_root.is_symlink(), f"unsafe review stage root: {stage_root}")
    completed: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(stage_root.iterdir()):
        require(path.is_dir() and not path.is_symlink(), f"unexpected review stage entry: {path}")
        require(re.fullmatch(r"attempt_[0-9]{3}", path.name) is not None, f"unexpected attempt name: {path.name}")
        if (path / ".incomplete").exists() or not (path / "run_summary.json").is_file():
            continue
        summary = _load_json(path / "run_summary.json", "completed attempt summary")
        require(
            summary.get("workflow_version") == transport.WORKFLOW_VERSION
            and summary.get("status")
            in {
                "completed_checkpoint_shard",
                "completed_with_infrastructure_errors_requires_fresh_retry",
            },
            f"attempt summary is not a real completed transport run: {path}",
        )
        completed.append((path, summary))
    require(len(completed) <= 1, f"stage contains multiple completed attempts: {stage_root}")
    return completed[0] if completed else None


def _next_attempt_root(stage_root: Path) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    require(stage_root.is_dir() and not stage_root.is_symlink(), f"unsafe review stage root: {stage_root}")
    indices: list[int] = []
    for path in stage_root.iterdir():
        match = re.fullmatch(r"attempt_([0-9]{3})", path.name)
        require(match is not None and path.is_dir() and not path.is_symlink(), f"unexpected review stage entry: {path}")
        indices.append(int(match.group(1)))
    index = 0 if not indices else max(indices) + 1
    require(index <= 999, f"too many attempts in {stage_root}")
    return stage_root / f"attempt_{index:03d}"


def _run_or_reuse_attempt(
    *,
    stage_root: Path,
    assignments_path: Path,
    blank_scores_path: Path,
    api_key: str,
    proxy_version: str,
    timeout: int,
    start: int = 0,
    limit: int | None = None,
    retry_run_roots: Sequence[Path] = (),
    review_runner: ReviewRunner = transport.run_review,
) -> tuple[Path, dict[str, Any]]:
    existing = _completed_attempt(stage_root)
    if existing is not None:
        return existing
    output_root = _next_attempt_root(stage_root)
    try:
        summary = review_runner(
            assignments_path=assignments_path,
            blank_scores_path=blank_scores_path,
            output_root=output_root,
            dry_run=False,
            start=start,
            limit=limit,
            workers=PASS_WORKERS,
            timeout=timeout,
            base_url=LOCAL_RESPONSES_BASE_URL,
            api_key=api_key,
            proxy_version=proxy_version,
            retry_run_roots=retry_run_roots,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        raise FormalLaunchError(
            f"formal transport attempt failed: {type(exc).__name__}: {message}"
        ) from None
    require(
        isinstance(summary, dict)
        and summary.get("workflow_version") == transport.WORKFLOW_VERSION
        and summary.get("scientific_zero_fallbacks") == 0,
        "formal transport returned an invalid summary",
    )
    return output_root, summary


def _complete_stage_with_retries(
    *,
    stage_root: Path,
    assignments_path: Path,
    blank_scores_path: Path,
    api_key: str,
    proxy_version: str,
    timeout: int,
    start: int,
    limit: int,
    max_retry_rounds: int,
    review_runner: ReviewRunner,
) -> list[Path]:
    initial_root, summary = _run_or_reuse_attempt(
        stage_root=stage_root / "initial",
        assignments_path=assignments_path,
        blank_scores_path=blank_scores_path,
        api_key=api_key,
        proxy_version=proxy_version,
        timeout=timeout,
        start=start,
        limit=limit,
        review_runner=review_runner,
    )
    roots = [initial_root]
    latest = initial_root
    round_index = 0
    while int(summary.get("infrastructure_errors", -1)) > 0:
        require(round_index < max_retry_rounds, f"stage still has infrastructure errors after {max_retry_rounds} fresh retries: {stage_root}")
        retry_root, summary = _run_or_reuse_attempt(
            stage_root=stage_root / f"retry_{round_index:02d}",
            assignments_path=assignments_path,
            blank_scores_path=blank_scores_path,
            api_key=api_key,
            proxy_version=proxy_version,
            timeout=timeout,
            retry_run_roots=[latest],
            review_runner=review_runner,
        )
        roots.append(retry_root)
        latest = retry_root
        round_index += 1
    require(summary.get("infrastructure_errors") == 0, "stage ended without a valid infrastructure-error count")
    return roots


def _run_pass_pair(
    *,
    public_root: Path,
    runs_root: Path,
    stage_name: str,
    start: int,
    limit: int,
    api_key: str,
    proxy_version: str,
    timeout: int,
    max_retry_rounds: int,
    review_runner: ReviewRunner,
) -> dict[str, list[Path]]:
    outputs: dict[str, list[Path]] = {}

    def one(pass_name: str) -> tuple[str, list[Path]]:
        pass_root = public_root / pass_name
        roots = _complete_stage_with_retries(
            stage_root=runs_root / pass_name / stage_name,
            assignments_path=pass_root / "assignments.jsonl",
            blank_scores_path=pass_root / "scores.jsonl",
            api_key=api_key,
            proxy_version=proxy_version,
            timeout=timeout,
            start=start,
            limit=limit,
            max_retry_rounds=max_retry_rounds,
            review_runner=review_runner,
        )
        return pass_name, roots

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(one, pass_name): pass_name for pass_name in transport.PASS_NAMES}
        for future in as_completed(futures):
            pass_name, roots = future.result()
            outputs[pass_name] = roots
    require(set(outputs) == set(transport.PASS_NAMES), "paired pass execution did not return A and B")
    return outputs


def _merge_or_reuse(
    *,
    pass_root: Path,
    run_roots: Sequence[Path],
    output_root: Path,
    merger: Merger,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        _directory(output_root, "existing merge output")
        manifest = _load_json(output_root / "merge_manifest.json", "existing merge manifest")
        require(
            manifest.get("workflow_version") == transport.WORKFLOW_VERSION
            and manifest.get("status") == "complete_schema_valid_public_pass_review"
            and manifest.get("row_count") == TOTAL_ITEMS_PER_PASS,
            "existing merge output is not complete",
        )
        return manifest
    return merger(
        assignments_path=pass_root / "assignments.jsonl",
        blank_scores_path=pass_root / "scores.jsonl",
        run_roots=list(run_roots),
        output_root=output_root,
    )


def execute_formal_review(
    *,
    workspace_root: Path | None,
    authority_snapshot: Path | None,
    wan_original_snapshot: Path | None,
    review_package_public: Path | None,
    launch_root: Path,
    copilot_key_file: Path,
    preflight_receipt: Path,
    proxy_version: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
    timeout: int = 180,
    max_retry_rounds: int = DEFAULT_MAX_RETRY_ROUNDS,
    blind_key_path: Path | None = None,
    package_builder: PackageBuilder = package.build_review_package,
    review_runner: ReviewRunner = transport.run_review,
    merger: Merger = transport.merge_checkpoints,
) -> dict[str, Any]:
    require(proxy_version.strip() == proxy_version and proxy_version, "proxy_version is blank")
    require(timeout > 0 and max_retry_rounds >= 0, "timeout/retry policy is invalid")
    shards = formal_shards(shard_size)
    if review_package_public is not None:
        require(
            workspace_root is None
            and authority_snapshot is None
            and wan_original_snapshot is None,
            "copied-package mode cannot also use snapshot-build inputs",
        )
        public_root, package_inventory = validate_public_package_copy(
            review_package_public
        )
        try:
            launch_root.resolve().relative_to(public_root)
        except ValueError:
            pass
        else:
            raise FormalLaunchError(
                "launch_root must not be inside the immutable copied public package"
            )
    else:
        require(
            workspace_root is not None
            and authority_snapshot is not None
            and wan_original_snapshot is not None,
            "snapshot-build mode requires workspace and both snapshots",
        )
        package_root = prepare_review_package(
            workspace_root=workspace_root,
            authority_snapshot=authority_snapshot,
            wan_original_snapshot=wan_original_snapshot,
            launch_root=launch_root,
            blind_key_path=blind_key_path,
            package_builder=package_builder,
        )
        public_root, package_inventory = validate_public_package_copy(package_root)
    validated_preflight = validate_preflight_receipt(
        path=preflight_receipt,
        public_inventory=package_inventory,
        proxy_version=proxy_version,
    )
    api_key = read_copilot_key(copilot_key_file)
    runs_root = launch_root / "runs"
    roots_by_pass: dict[str, list[Path]] = {
        pass_name: [] for pass_name in transport.PASS_NAMES
    }

    smoke = _run_pass_pair(
        public_root=public_root,
        runs_root=runs_root,
        stage_name="smoke",
        start=0,
        limit=SMOKE_ITEMS,
        api_key=api_key,
        proxy_version=proxy_version,
        timeout=timeout,
        max_retry_rounds=max_retry_rounds,
        review_runner=review_runner,
    )
    for pass_name in transport.PASS_NAMES:
        roots_by_pass[pass_name].extend(smoke[pass_name])

    for shard in shards:
        pair = _run_pass_pair(
            public_root=public_root,
            runs_root=runs_root,
            stage_name=str(shard["name"]),
            start=int(shard["start"]),
            limit=int(shard["limit"]),
            api_key=api_key,
            proxy_version=proxy_version,
            timeout=timeout,
            max_retry_rounds=max_retry_rounds,
            review_runner=review_runner,
        )
        for pass_name in transport.PASS_NAMES:
            roots_by_pass[pass_name].extend(pair[pass_name])

    merged: dict[str, dict[str, Any]] = {}
    for pass_name in transport.PASS_NAMES:
        merged[pass_name] = _merge_or_reuse(
            pass_root=public_root / pass_name,
            run_roots=roots_by_pass[pass_name],
            output_root=launch_root / "merged" / pass_name,
            merger=merger,
        )
        require(
            merged[pass_name].get("status")
            == "complete_schema_valid_public_pass_review"
            and merged[pass_name].get("row_count") == TOTAL_ITEMS_PER_PASS,
            f"{pass_name} merge is not exact/complete",
        )

    result = {
        "schema_version": 1,
        "workflow_version": WORKFLOW_VERSION,
        "status": "completed_two_independent_schema_valid_passes",
        "public_package_root": str(public_root),
        "public_package_inventory": package_inventory,
        "pass_workers": PASS_WORKERS,
        "maximum_total_concurrency": TOTAL_CONCURRENCY,
        "primary_requests_per_pass": TOTAL_ITEMS_PER_PASS,
        "endpoint": LOCAL_RESPONSES_ENDPOINT,
        "model": transport.MODEL,
        "proxy_version": proxy_version,
        "preflight_receipt": transport.file_ref(preflight_receipt),
        "preflight_luna_metadata_sha256": validated_preflight["proxy"]["luna"]
        ["metadata_sha256"],
        "source_run_counts": {
            pass_name: len(roots_by_pass[pass_name])
            for pass_name in transport.PASS_NAMES
        },
        "merged": {
            pass_name: transport.file_ref(
                launch_root / "merged" / pass_name / "merge_manifest.json"
            )
            for pass_name in transport.PASS_NAMES
        },
        "scientific_zero_fallbacks": 0,
    }
    require(api_key not in json.dumps(result, ensure_ascii=False), "credential leaked into launch result")
    _write_or_verify_json(launch_root / "formal_launch_receipt.json", result)
    return result


def _resolve(project_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--authority-snapshot", type=Path)
    parser.add_argument("--wan-original-snapshot", type=Path)
    parser.add_argument("--review-package-public", type=Path)
    parser.add_argument("--launch-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare-package", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    parser.add_argument("--copilot-key-file", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--preflight-pass", choices=transport.PASS_NAMES, default="pass_a")
    parser.add_argument("--preflight-start", type=int, default=0)
    parser.add_argument("--preflight-limit", type=int, default=DEFAULT_PREFLIGHT_ITEMS)
    parser.add_argument("--preflight-workers", type=int, default=PASS_WORKERS)
    parser.add_argument("--blind-key-file", type=Path)
    parser.add_argument("--proxy-version", default="copilot-claude-local")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retry-rounds", type=int, default=DEFAULT_MAX_RETRY_ROUNDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        if args.dry_run:
            result = build_dry_run_plan(
                authority_snapshot=(
                    None
                    if args.authority_snapshot is None
                    else _resolve(project_root, args.authority_snapshot)
                ),
                wan_original_snapshot=(
                    None
                    if args.wan_original_snapshot is None
                    else _resolve(project_root, args.wan_original_snapshot)
                ),
                review_package_public=(
                    None
                    if args.review_package_public is None
                    else _resolve(project_root, args.review_package_public)
                ),
                launch_root=_resolve(project_root, args.launch_root),
                shard_size=args.shard_size,
                proxy_version=args.proxy_version,
            )
        elif args.prepare_package:
            require(
                args.workspace_root is not None
                and args.authority_snapshot is not None
                and args.wan_original_snapshot is not None,
                "--prepare-package requires --workspace-root, --authority-snapshot, and --wan-original-snapshot",
            )
            require(
                args.review_package_public is None,
                "--prepare-package builds from snapshots and does not accept --review-package-public",
            )
            package_root = prepare_review_package(
                workspace_root=_resolve(project_root, args.workspace_root),
                authority_snapshot=_resolve(project_root, args.authority_snapshot),
                wan_original_snapshot=_resolve(project_root, args.wan_original_snapshot),
                launch_root=_resolve(project_root, args.launch_root),
                blind_key_path=(
                    None
                    if args.blind_key_file is None
                    else _resolve(project_root, args.blind_key_file)
                ),
            )
            public_root, inventory = validate_public_package_copy(package_root / "public")
            result = {
                "schema_version": 1,
                "workflow_version": WORKFLOW_VERSION,
                "status": "review_package_prepared_no_api_calls",
                "package_root": str(package_root),
                "public_root": str(public_root),
                "public_inventory": inventory,
                "next_stage": "run --preflight with --review-package-public set to public_root",
                "api_calls": 0,
            }
        elif args.preflight:
            require(
                args.review_package_public is not None,
                "--preflight requires --review-package-public",
            )
            require(
                args.copilot_key_file is not None,
                "--preflight requires --copilot-key-file",
            )
            result = run_preflight(
                review_package_public=_resolve(
                    project_root, args.review_package_public
                ),
                output_root=_resolve(project_root, args.launch_root) / "preflight",
                copilot_key_file=_resolve(project_root, args.copilot_key_file),
                proxy_version=args.proxy_version,
                pass_name=args.preflight_pass,
                start=args.preflight_start,
                limit=args.preflight_limit,
                workers=args.preflight_workers,
                timeout=args.timeout,
            )
        else:
            require(args.copilot_key_file is not None, "--run-formal requires --copilot-key-file")
            require(args.preflight_receipt is not None, "--run-formal requires --preflight-receipt")
            result = execute_formal_review(
                workspace_root=(
                    None
                    if args.workspace_root is None
                    else _resolve(project_root, args.workspace_root)
                ),
                authority_snapshot=(
                    None
                    if args.authority_snapshot is None
                    else _resolve(project_root, args.authority_snapshot)
                ),
                wan_original_snapshot=(
                    None
                    if args.wan_original_snapshot is None
                    else _resolve(project_root, args.wan_original_snapshot)
                ),
                review_package_public=(
                    None
                    if args.review_package_public is None
                    else _resolve(project_root, args.review_package_public)
                ),
                launch_root=_resolve(project_root, args.launch_root),
                copilot_key_file=_resolve(project_root, args.copilot_key_file),
                preflight_receipt=_resolve(project_root, args.preflight_receipt),
                proxy_version=args.proxy_version,
                shard_size=args.shard_size,
                timeout=args.timeout,
                max_retry_rounds=args.max_retry_rounds,
                blind_key_path=(
                    None
                    if args.blind_key_file is None
                    else _resolve(project_root, args.blind_key_file)
                ),
            )
    except (OSError, ValueError, FormalLaunchError) as exc:
        parser.exit(2, f"formal review launch refused: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
