#!/usr/bin/env python3
"""Build and validate the five-script formal-evaluation code registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "causal_role_erasure_7m_evaluation_code_registry_v1"
SCRIPT_PATHS = (
    "scripts/build_causal_role_erasure_7mechanism_review_package_v1.py",
    "scripts/review_causal_role_erasure_7mechanism_formal_v2.py",
    "scripts/run_causal_role_erasure_7mechanism_formal_review_v1.py",
    "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py",
    "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v1.py",
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_registry(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    files = []
    for relative in SCRIPT_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"evaluation implementation missing or symlinked: {path}")
        files.append(
            {
                "path": relative,
                "sha256": _sha(path),
                "size_bytes": path.stat().st_size,
            }
        )
    body = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "five_evaluation_implementations_frozen",
        "files": files,
    }
    return {**body, "registry_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_registry(value: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "protocol",
        "status",
        "files",
        "registry_sha256",
    }:
        raise ValueError("evaluation code registry schema changed")
    body = {key: value[key] for key in ("schema_version", "protocol", "status", "files")}
    if (
        body["schema_version"] != 1
        or body["protocol"] != PROTOCOL
        or body["status"] != "five_evaluation_implementations_frozen"
        or value["registry_sha256"] != hashlib.sha256(_canonical(body)).hexdigest()
    ):
        raise ValueError("evaluation code registry identity or self-commitment changed")
    observed = build_registry(project_root)
    if observed != dict(value):
        raise ValueError("current evaluation implementations differ from the frozen registry")
    return observed

