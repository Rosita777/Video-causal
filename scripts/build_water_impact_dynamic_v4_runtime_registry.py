#!/usr/bin/env python3
"""Build or validate the frozen public v4 Python/runtime registry.

The registry contains only public software/runtime provenance.  Building it
must happen with the registered ``models/.wan-runtime`` interpreter; it never
reads evaluation data or any sealed/private artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any


PROTOCOL = "water_impact_dynamic_v4_runtime_registry_v2"
DATASET_VERSION = "v4_dev72_v2"
EXPECTED_RUNTIME_ROOT = Path("models/.wan-runtime")
EXPECTED_PYTHON_EXECUTABLE = Path("models/.wan-runtime/bin/python")
EXPECTED_PYTHON_VERSION = "3.11.15"
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "diffusers": "0.33.1",
    "huggingface-hub": "0.36.2",
    "numpy": "2.4.6",
    "peft": "0.15.2",
    "protobuf": "7.35.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.21.4",
    "torch": "2.6.0",
    "transformers": "4.51.3",
}
EXPECTED_TORCH_CUDA_VERSION = "12.4"
EXPECTED_CUDNN_VERSION = 90100
EXPECTED_TORCH_MODULE_VERSION = "2.6.0+cu124"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_registry_payload() -> dict[str, Any]:
    """Return the deterministic, reviewable public runtime contract."""

    return {
        "protocol": PROTOCOL,
        "status": "frozen",
        "dataset_version": DATASET_VERSION,
        "runtime_root": str(EXPECTED_RUNTIME_ROOT),
        "python_executable": str(EXPECTED_PYTHON_EXECUTABLE),
        "sys_prefix_policy": "realpath(sys.prefix)==realpath(runtime_root)",
        "python": {
            "implementation": "CPython",
            "version": EXPECTED_PYTHON_VERSION,
        },
        "torch": {
            "distribution_version": EXPECTED_PACKAGE_VERSIONS["torch"],
            "module_version": EXPECTED_TORCH_MODULE_VERSION,
        },
        "cuda": {
            "available_required": True,
            "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
            "cudnn_version": EXPECTED_CUDNN_VERSION,
        },
        "packages": dict(EXPECTED_PACKAGE_VERSIONS),
    }


def validate_current_runtime(project_root: Path = Path(".")) -> None:
    """Fail unless this process is the exact registered GPU environment."""

    runtime_root = (project_root / EXPECTED_RUNTIME_ROOT).resolve(strict=True)
    current_prefix = Path(sys.prefix).resolve(strict=True)
    if current_prefix != runtime_root:
        raise ValueError(
            f"runtime sys.prefix mismatch: {current_prefix} != {runtime_root}"
        )
    expected_executable = (
        project_root / EXPECTED_PYTHON_EXECUTABLE
    ).resolve(strict=True)
    current_executable = Path(sys.executable).resolve(strict=True)
    if current_executable != expected_executable:
        raise ValueError(
            "runtime interpreter mismatch: "
            f"{current_executable} != {expected_executable}"
        )
    if platform.python_implementation() != "CPython":
        raise ValueError("v4 runtime requires CPython")
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise ValueError(
            f"Python version mismatch: {platform.python_version()} != "
            f"{EXPECTED_PYTHON_VERSION}"
        )
    actual_packages = {
        distribution: importlib.metadata.version(distribution)
        for distribution in EXPECTED_PACKAGE_VERSIONS
    }
    if actual_packages != EXPECTED_PACKAGE_VERSIONS:
        raise ValueError(
            f"runtime package versions differ from protocol: {actual_packages!r}"
        )

    import torch

    if torch.__version__ != EXPECTED_TORCH_MODULE_VERSION:
        raise ValueError(f"runtime torch version mismatch: {torch.__version__}")
    if not torch.cuda.is_available():
        raise ValueError("registered v4 runtime requires CUDA availability")
    if torch.version.cuda != EXPECTED_TORCH_CUDA_VERSION:
        raise ValueError(f"runtime CUDA version mismatch: {torch.version.cuda}")
    cudnn_version = torch.backends.cudnn.version()
    if cudnn_version != EXPECTED_CUDNN_VERSION:
        raise ValueError(f"runtime cuDNN version mismatch: {cudnn_version}")


def validate_runtime_registry(
    path: Path,
    expected_sha256: str,
    *,
    project_root: Path = Path("."),
    verify_current_runtime: bool = True,
) -> dict[str, Any]:
    """Validate exact registry bytes/schema and, formally, the live runtime."""

    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise ValueError("runtime registry hash must be a lowercase SHA-256")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"runtime registry is missing or symlinked: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError("runtime registry byte hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("runtime registry is not valid UTF-8 JSON") from exc
    if payload != expected_registry_payload():
        raise ValueError("runtime registry differs from the frozen runtime contract")
    if verify_current_runtime:
        validate_current_runtime(project_root)
    return payload


def atomic_write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite runtime registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "validate"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/v4_runtime_registry_v2.json"),
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.action == "build":
        if args.expected_sha256 is not None:
            parser.error("--expected-sha256 is valid only for validate")
        validate_current_runtime(Path("."))
        atomic_write_new_json(args.output, expected_registry_payload())
        print(f"Wrote frozen v4 runtime registry: {args.output} sha256={file_sha256(args.output)}")
        return 0
    if args.expected_sha256 is None:
        parser.error("validate requires --expected-sha256")
    validate_runtime_registry(
        args.output,
        args.expected_sha256,
        project_root=Path("."),
        verify_current_runtime=True,
    )
    print(f"Validated frozen v4 runtime registry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
