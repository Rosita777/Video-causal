#!/usr/bin/env python3
"""Build sanitized reproducibility receipts without opening scores or keys."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
ENVIRONMENT_PROTOCOL = "causal_role_erasure_7m_release_evaluation_environment_v1"
ENVIRONMENT_STATUS = "release_time_environment_validated_not_historical_execution_receipt"
INVENTORY_PROTOCOL = "causal_role_erasure_7m_wan_training_receipt_inventory_v1"
INVENTORY_STATUS = "18_receipts_indexed_from_individually_validated_eval_run_manifest"
LEDGER_PROTOCOL = "causal_role_erasure_7m_executable_command_ledger_v1"
LEDGER_STATUS = "full_reproduction_commands_frozen_before_canonical_metrics"

EXPECTED_RUN_MATRIX_SHA256 = "0fb9427d987041ea3dbac6eaeb1a5cf49760b774738e31c371f1fe3b19fc59a4"
EXPECTED_EVAL_RUN_MANIFEST_SHA256 = "36f7f6d5dadf165c6e3aa6705638fe404d51b8f6f2fc69ea8f929f2604b0b146"
EXPECTED_EVAL_AGGREGATE_SHA256 = "a5b4822234806da8e2e41b4ab42da1ba0f44d48f15bfea92d618e454f09d0f25"
EXPECTED_WAN_ORIGINAL_RUN_MANIFEST_SHA256 = "b2c3d61697c8baad27574e690585d328c29790425756e381b41c4c45a3d76197"
EXPECTED_CORE_QUEUE_PLAN_SHA256 = "d5bf15267d5cb6e38515eb617547cec7449ef29f91878a0b5855581360c00782"
EXPECTED_SAFREE_QUEUE_PLAN_SHA256 = "73a4f59713e9cc230947b2208814df095d2d013c14017e7e8f76fbf257f7cfd7"
EXPECTED_FORMAL_LAUNCH_RECEIPT_SHA256 = "833254ac9b38a609f233b0d09d9507d316326ab46a6956a0d07b4c1fdf0d3ff1"

LOCK_DISTRIBUTIONS = {
    "av",
    "iniconfig",
    "numpy",
    "opencv-python-headless",
    "packaging",
    "pillow",
    "pluggy",
    "pygments",
    "pytest",
}
BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools", "wheel"}
RELEASE_TEST_PATHS = (
    "tests/test_build_causal_role_erasure_7mechanism_review_package_v1.py",
    "tests/test_review_causal_role_erasure_7mechanism_formal_v2.py",
    "tests/test_run_causal_role_erasure_7mechanism_formal_review_v1.py",
    "tests/test_causal_role_erasure_7mechanism_evaluation_code_registry_v1.py",
    "tests/test_causal_role_erasure_7mechanism_formal_metrics_v1.py",
    "tests/test_causal_role_erasure_7mechanism_evaluation_v2.py",
    "tests/test_export_causal_role_erasure_7mechanism_paper_tables_v1.py",
    "tests/test_causal_role_erasure_7mechanism_reproducibility_v1.py",
    "tests/test_causal_role_erasure_7mechanism_review_process_receipt_v2.py",
)
EXPECTED_RELEASE_TEST_PASSES = 64
EXPECTED_RELEASE_TEST_PASSES_BY_PATH = dict(
    zip(RELEASE_TEST_PATHS, (10, 9, 9, 2, 3, 9, 11, 8, 3), strict=True)
)
RELEASE_TEST_EXECUTION_MODEL = "one_fresh_python_pytest_subprocess_per_test_file"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMAND_FIELDS = {
    "stage_id",
    "environment_id",
    "cwd",
    "argv",
    "environment",
    "private_inputs",
    "command_kind",
}
COMMAND_ENVIRONMENTS = {"a100", "cpu", "evaluation", "evaluation_hosted"}
EVAL_PYTHON_PLACEHOLDER = {
    "kind": "python_interpreter",
    "must_be_supplied": True,
    "executable_sha256_binding": {
        "artifact_binding": "evaluation_environment_receipt",
        "json_pointer": "/python/executable_sha256",
        "comparison": "sha256_of_resolved_executable",
    },
}
LEDGER_SCOPE = {
    "commands_are_sanitized": True,
    "commands_are_argument_vectors_not_shell_strings": True,
    "post_hoc_reproduction_commands_are_not_claimed_as_historical_invocations": True,
    "realized_expanded_generation_commands_are_bound_by_evidence": True,
    "manual_stage_inventory_includes_non_executable_human_work": True,
    "manual_stage_released_input_refs_are_partial_not_complete_delivery_receipts": True,
    "target_selection_private_outputs_and_exact_delivery_receipts_are_not_released_here": True,
}
MANUAL_STAGE_FIELDS = {
    "stage_id",
    "phase",
    "completion_state_at_ledger_freeze",
    "parent_stage_ids",
    "instruction",
    "released_input_refs",
    "unreleased_or_pending_input_roles",
}
TARGET_SELECTION_RELEASED_INPUT_PATHS = {
    "static_build_registry": "data/causal_role_erasure_7mechanism_main_v2/build_registry.json",
    "target_candidate_registry": "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv",
}


class ReproducibilityError(ValueError):
    """A reproducibility artifact is incomplete, unsafe, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReproducibilityError(message)


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


def load_json(path: Path, label: str) -> dict[str, Any]:
    regular_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def safe_relative(value: str, label: str) -> str:
    path = Path(value)
    require(value != "" and not path.is_absolute(), f"{label} path must be relative")
    require(".." not in path.parts and "." not in path.parts, f"{label} path is unsafe")
    return path.as_posix()


def file_ref(root_id: str, logical_path: str, actual_path: Path) -> dict[str, Any]:
    regular_file(actual_path, logical_path)
    return {
        "root_id": root_id,
        "path": safe_relative(logical_path, logical_path),
        "sha256": sha256_file(actual_path),
        "size_bytes": actual_path.stat().st_size,
    }


def repo_ref(project_root: Path, path: Path) -> dict[str, Any]:
    resolved_root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReproducibilityError(f"component outside repository: {path}") from exc
    return file_ref("repo_root", relative.as_posix(), resolved)


def with_self_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    require(field not in value, f"{field} already present")
    body = dict(value)
    return {**body, field: hashlib.sha256(canonical_json_bytes(body)).hexdigest()}


def validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = value.get(field)
    require(isinstance(observed, str) and HEX64.fullmatch(observed) is not None, f"{label} self digest missing")
    body = dict(value)
    del body[field]
    require(hashlib.sha256(canonical_json_bytes(body)).hexdigest() == observed, f"{label} self digest mismatch")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.parent.is_symlink(), "output parent may not be a symlink")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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


def parse_lock(path: Path) -> dict[str, str]:
    regular_file(path, "evaluation environment lock")
    pins: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        require(match is not None, f"environment lock line {number} is not an exact pin")
        name = match.group(1).lower().replace("_", "-")
        require(name not in pins, f"environment lock repeats {name}")
        pins[name] = match.group(2)
    require(set(pins) == LOCK_DISTRIBUTIONS, "environment lock distribution inventory changed")
    return pins


def installed_distributions() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).lower().replace("_", "-")
        if name:
            values[name] = distribution.version
    return values


def isolated_json_probe(source: str, label: str) -> Any:
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    require(completed.returncode == 0, f"{label} isolated import probe failed")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ReproducibilityError(f"{label} isolated import probe returned invalid JSON") from exc


def external_tool_receipt(command: str, version_arguments: Sequence[str]) -> dict[str, Any]:
    located = shutil.which(command)
    require(located is not None, f"required external tool is absent from PATH: {command}")
    executable = Path(located).resolve(strict=True)
    regular_file(executable, f"{command} executable")
    completed = subprocess.run(
        [str(executable), *version_arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    require(completed.returncode == 0, f"{command} version probe failed")
    output = (completed.stdout + completed.stderr).strip()
    require(output, f"{command} version probe returned no output")
    return {
        "command": command,
        "executable_sha256": sha256_file(executable),
        "version_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "version_first_line": output.splitlines()[0].strip(),
    }


def release_test_command(python: str, relative_path: str) -> list[str]:
    return [
        python,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        relative_path,
    ]


def _pytest_outcome_counts(output: str, label: str) -> tuple[dict[str, int], str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    summary = lines[-1] if lines else ""
    counts = {
        outcome: sum(
            int(match.group(1))
            for match in re.finditer(
                rf"(?:^|[\s,])(\d+) {outcome}(?:s)?(?:[\s,]|$)", summary
            )
        )
        for outcome in ("passed", "failed", "skipped")
    }
    unexpected = sum(
        int(match.group(1))
        for outcome in ("error", "xfailed", "xpassed")
        for match in re.finditer(
            rf"(?:^|[\s,])(\d+) {outcome}(?:s)?(?:[\s,]|$)", summary
        )
    )
    require(summary != "", f"{label} returned no pytest summary")
    require(unexpected == 0, f"{label} produced a non-release pytest outcome")
    return counts, summary


def run_release_test_files(
    *, project_root: Path, test_relative_paths: Sequence[str]
) -> dict[str, Any]:
    require(
        list(test_relative_paths) == list(RELEASE_TEST_PATHS),
        "release evaluation test inventory/order changed",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_ADDOPTS": "",
        }
    )
    per_file: list[dict[str, Any]] = []
    for relative_path in test_relative_paths:
        completed = subprocess.run(
            release_test_command(sys.executable, relative_path),
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        counts, raw_summary = _pytest_outcome_counts(
            completed.stdout + "\n" + completed.stderr,
            f"release test {relative_path}",
        )
        expected_passes = EXPECTED_RELEASE_TEST_PASSES_BY_PATH[relative_path]
        require(
            completed.returncode == 0
            and counts
            == {"passed": expected_passes, "failed": 0, "skipped": 0},
            f"release test did not reproduce its exact all-passing count: "
            f"{relative_path}: {raw_summary}",
        )
        per_file.append(
            {
                "path": relative_path,
                "command": release_test_command("${EVAL_PYTHON}", relative_path),
                "status": "passed",
                "return_code": 0,
                "result": counts,
                "summary": f"{expected_passes} passed, 0 failed, 0 skipped",
            }
        )
    aggregate = {
        outcome: sum(record["result"][outcome] for record in per_file)
        for outcome in ("passed", "failed", "skipped")
    }
    require(
        aggregate
        == {
            "passed": EXPECTED_RELEASE_TEST_PASSES,
            "failed": 0,
            "skipped": 0,
        },
        "release evaluation suite did not aggregate to exactly 64 passed/0 failed/0 skipped",
    )
    return {
        "execution_model": RELEASE_TEST_EXECUTION_MODEL,
        "per_file": per_file,
        "aggregate": aggregate,
        "status": "passed",
        "return_code": 0,
        "summary": f"{EXPECTED_RELEASE_TEST_PASSES} passed, 0 failed, 0 skipped",
    }


def freeze_environment_receipt(
    *,
    project_root: Path,
    lock_path: Path,
    output_path: Path,
    test_files: Sequence[Path],
    run_tests: bool,
) -> dict[str, Any]:
    require(sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12), "release evaluator requires CPython 3.12.x")
    pins = parse_lock(lock_path)
    installed = installed_distributions()
    for name, version in pins.items():
        require(installed.get(name) == version, f"installed {name} does not match lock")
    unexpected = set(installed) - set(pins) - BOOTSTRAP_DISTRIBUTIONS
    require(not unexpected, f"release environment has unbound distributions: {sorted(unexpected)}")

    test_relative_paths: list[str] = []
    for path in test_files:
        regular_file(path, "evaluation test")
        try:
            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise ReproducibilityError(f"evaluation test is outside project root: {path}") from exc
        test_relative_paths.append(relative)
    require(
        test_relative_paths == list(RELEASE_TEST_PATHS),
        "release evaluation test inventory/order changed",
    )
    result: dict[str, Any]
    if run_tests:
        result = run_release_test_files(
            project_root=project_root, test_relative_paths=test_relative_paths
        )
    else:
        result = {
            "execution_model": RELEASE_TEST_EXECUTION_MODEL,
            "per_file": [],
            "aggregate": {"passed": 0, "failed": 0, "skipped": 0},
            "status": "not_run",
            "return_code": None,
            "summary": "test execution explicitly disabled",
        }

    codec_versions = {
        "pyav_libraries": isolated_json_probe(
            "import av,json; print(json.dumps({k:list(v) for k,v in sorted(av.library_versions.items())},sort_keys=True))",
            "PyAV",
        ),
        "opencv": isolated_json_probe(
            "import cv2,json; print(json.dumps(cv2.__version__))",
            "OpenCV",
        ),
    }
    pillow_probe = isolated_json_probe(
        "import json; from PIL import Image,features; print(json.dumps({'pillow':Image.__version__,'libjpeg':features.version_codec('jpg')},sort_keys=True))",
        "Pillow",
    )
    require(
        isinstance(pillow_probe, dict) and set(pillow_probe) == {"pillow", "libjpeg"},
        "Pillow isolated import probe schema changed",
    )
    codec_versions.update(pillow_probe)

    receipt = with_self_digest(
        {
            "schema_version": 1,
            "protocol": ENVIRONMENT_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": ENVIRONMENT_STATUS if run_tests else "release_time_environment_inventory_only_tests_not_run",
            "scope": {
                "release_time_reproduction_environment": True,
                "historical_execution_environment_claim": False,
                "hosted_model_bitwise_replay_claim": False,
                "frozen_completed_scores_are_downstream_authority": True,
            },
            "lock": repo_ref(project_root, lock_path),
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "executable_sha256": sha256_file(Path(sys.executable).resolve()),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "distributions": dict(sorted(pins.items())),
            "bootstrap_distributions": {
                name: installed[name] for name in sorted(BOOTSTRAP_DISTRIBUTIONS) if name in installed
            },
            "codec_versions": codec_versions,
            "external_tools": {
                "latexmk": external_tool_receipt("latexmk", ("-v",)),
            },
            "tests": {
                "files": [repo_ref(project_root, path) for path in test_files],
                **result,
            },
        },
        "receipt_sha256",
    )
    write_json_exclusive(output_path, receipt)
    return receipt


def validate_environment_receipt(
    *,
    project_root: Path,
    lock_path: Path,
    receipt_path: Path,
    expected_test_paths: Sequence[str] = RELEASE_TEST_PATHS,
    live: bool = True,
) -> dict[str, Any]:
    receipt = load_json(receipt_path, "release evaluation environment receipt")
    require(
        set(receipt)
        == {
            "schema_version",
            "protocol",
            "protocol_version",
            "status",
            "scope",
            "lock",
            "python",
            "platform",
            "distributions",
            "bootstrap_distributions",
            "codec_versions",
            "external_tools",
            "tests",
            "receipt_sha256",
        },
        "release evaluation environment receipt schema changed",
    )
    validate_self_digest(receipt, "receipt_sha256", "release evaluation environment receipt")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("protocol") == ENVIRONMENT_PROTOCOL
        and receipt.get("protocol_version") == PROTOCOL_VERSION
        and receipt.get("status") == ENVIRONMENT_STATUS,
        "release evaluation environment receipt identity/status changed",
    )
    require(
        receipt.get("scope")
        == {
            "release_time_reproduction_environment": True,
            "historical_execution_environment_claim": False,
            "hosted_model_bitwise_replay_claim": False,
            "frozen_completed_scores_are_downstream_authority": True,
        },
        "release evaluation environment receipt scope changed",
    )
    require(receipt.get("lock") == repo_ref(project_root, lock_path), "environment receipt binds a different lock")
    require(receipt.get("distributions") == dict(sorted(parse_lock(lock_path).items())), "environment receipt pins changed")
    bootstrap = receipt.get("bootstrap_distributions")
    require(
        isinstance(bootstrap, dict)
        and set(bootstrap) <= BOOTSTRAP_DISTRIBUTIONS
        and all(isinstance(value, str) and value for value in bootstrap.values()),
        "environment receipt bootstrap distribution inventory invalid",
    )
    python = receipt.get("python")
    require(
        isinstance(python, dict)
        and set(python) == {"implementation", "version", "executable_sha256"}
        and python.get("implementation") == "CPython"
        and re.fullmatch(r"3\.12\.\d+", str(python.get("version", ""))) is not None
        and HEX64.fullmatch(str(python.get("executable_sha256", ""))) is not None,
        "environment receipt Python identity invalid",
    )
    platform_value = receipt.get("platform")
    require(
        isinstance(platform_value, dict)
        and set(platform_value) == {"system", "release", "machine"}
        and all(isinstance(value, str) and value for value in platform_value.values()),
        "environment receipt platform identity invalid",
    )
    codecs = receipt.get("codec_versions")
    require(
        isinstance(codecs, dict)
        and set(codecs) == {"pyav_libraries", "opencv", "pillow", "libjpeg"}
        and isinstance(codecs["pyav_libraries"], dict)
        and codecs["pyav_libraries"],
        "environment receipt codec inventory invalid",
    )
    external_tools = receipt.get("external_tools")
    require(
        isinstance(external_tools, dict)
        and set(external_tools) == {"latexmk"}
        and isinstance(external_tools["latexmk"], dict)
        and set(external_tools["latexmk"])
        == {
            "command",
            "executable_sha256",
            "version_output_sha256",
            "version_first_line",
        }
        and external_tools["latexmk"].get("command") == "latexmk"
        and HEX64.fullmatch(str(external_tools["latexmk"].get("executable_sha256", "")))
        is not None
        and HEX64.fullmatch(str(external_tools["latexmk"].get("version_output_sha256", "")))
        is not None
        and isinstance(external_tools["latexmk"].get("version_first_line"), str)
        and external_tools["latexmk"]["version_first_line"].strip(),
        "environment receipt external-tool inventory invalid",
    )
    tests = receipt.get("tests")
    expected_paths = [safe_relative(path, "release test") for path in expected_test_paths]
    require(
        expected_paths == list(RELEASE_TEST_PATHS),
        "release evaluation test inventory/order changed",
    )
    expected_files = [repo_ref(project_root, project_root / path) for path in expected_paths]
    strict_test_fields = {
        "execution_model",
        "files",
        "per_file",
        "aggregate",
        "status",
        "return_code",
        "summary",
    }
    legacy_test_fields = {"command", "files", "status", "return_code", "summary"}
    has_strict_per_file_evidence = isinstance(tests, dict) and set(tests) == strict_test_fields
    if has_strict_per_file_evidence:
        expected_per_file = [
            {
                "path": path,
                "command": release_test_command("${EVAL_PYTHON}", path),
                "status": "passed",
                "return_code": 0,
                "result": {
                    "passed": EXPECTED_RELEASE_TEST_PASSES_BY_PATH[path],
                    "failed": 0,
                    "skipped": 0,
                },
                "summary": (
                    f"{EXPECTED_RELEASE_TEST_PASSES_BY_PATH[path]} passed, "
                    "0 failed, 0 skipped"
                ),
            }
            for path in expected_paths
        ]
        require(
            tests
            == {
                "execution_model": RELEASE_TEST_EXECUTION_MODEL,
                "files": expected_files,
                "per_file": expected_per_file,
                "aggregate": {
                    "passed": EXPECTED_RELEASE_TEST_PASSES,
                    "failed": 0,
                    "skipped": 0,
                },
                "status": "passed",
                "return_code": 0,
                "summary": (
                    f"{EXPECTED_RELEASE_TEST_PASSES} passed, 0 failed, 0 skipped"
                ),
            },
            "environment receipt per-file test evidence changed or is incomplete",
        )
    else:
        # Compatibility is limited to non-live validation of synthetic/downstream
        # v1 fixtures.  A release freeze or live revalidation always requires the
        # per-file subprocess evidence above.
        require(
            not live
            and isinstance(tests, dict)
            and set(tests) == legacy_test_fields
            and tests.get("command")
            == [
                "${EVAL_PYTHON}",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *expected_paths,
            ]
            and tests.get("files") == expected_files
            and tests.get("status") == "passed"
            and tests.get("return_code") == 0
            and tests.get("summary") == f"{EXPECTED_RELEASE_TEST_PASSES} passed",
            "live validation requires deterministic per-file test evidence",
        )
    if live:
        require(
            sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12),
            "live release evaluator is not CPython 3.12.x",
        )
        pins = parse_lock(lock_path)
        installed = installed_distributions()
        require(
            all(installed.get(name) == version for name, version in pins.items()),
            "live installed distributions differ from the lock",
        )
        require(
            set(installed) - set(pins) - BOOTSTRAP_DISTRIBUTIONS == set(),
            "live release environment has unbound distributions",
        )
        live_python = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_sha256": sha256_file(Path(sys.executable).resolve(strict=True)),
        }
        require(python == live_python, "environment receipt Python differs from the live interpreter")
        live_platform = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
        require(platform_value == live_platform, "environment receipt platform differs from the live host")
        require(
            bootstrap
            == {
                name: installed[name]
                for name in sorted(BOOTSTRAP_DISTRIBUTIONS)
                if name in installed
            },
            "environment receipt bootstrap distributions differ from the live environment",
        )
        live_codecs = {
            "pyav_libraries": isolated_json_probe(
                "import av,json; print(json.dumps({k:list(v) for k,v in sorted(av.library_versions.items())},sort_keys=True))",
                "PyAV",
            ),
            "opencv": isolated_json_probe(
                "import cv2,json; print(json.dumps(cv2.__version__))", "OpenCV"
            ),
        }
        live_pillow = isolated_json_probe(
            "import json; from PIL import Image,features; print(json.dumps({'pillow':Image.__version__,'libjpeg':features.version_codec('jpg')},sort_keys=True))",
            "Pillow",
        )
        require(isinstance(live_pillow, dict), "live Pillow codec probe malformed")
        live_codecs.update(live_pillow)
        require(codecs == live_codecs, "environment receipt codecs differ from the live environment")
        require(
            external_tools
            == {"latexmk": external_tool_receipt("latexmk", ("-v",))},
            "environment receipt external tools differ from the live environment",
        )
        live_test_evidence = run_release_test_files(
            project_root=project_root, test_relative_paths=expected_paths
        )
        require(
            {key: value for key, value in tests.items() if key != "files"}
            == live_test_evidence,
            "live release suite did not reproduce the recorded per-file 64-pass gate",
        )
    return receipt


def read_run_matrix(path: Path) -> list[dict[str, str]]:
    regular_file(path, "run matrix")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 18 and [int(row["run_index"]) for row in rows] == list(range(18)), "run matrix is not the frozen 18-run order")
    require(len({row["run_id"] for row in rows}) == 18, "run matrix IDs repeat")
    return rows


def logical_a100_path(value: str, expected_suffix: str, label: str) -> str:
    normalized = Path(value).as_posix()
    marker = "/outputs/causal_role_erasure_7mechanism_main_v2/"
    require(marker in normalized, f"{label} lacks the formal output-root marker")
    logical = "outputs/causal_role_erasure_7mechanism_main_v2/" + normalized.split(marker, 1)[1]
    require(logical == expected_suffix, f"{label} path differs from frozen run matrix")
    return logical


def freeze_wan_receipt_inventory(
    *,
    project_root: Path,
    run_matrix_path: Path,
    eval_run_manifest_path: Path,
    eval_aggregate_path: Path,
    output_path: Path,
    expected_run_manifest_sha256: str = EXPECTED_EVAL_RUN_MANIFEST_SHA256,
    expected_aggregate_sha256: str = EXPECTED_EVAL_AGGREGATE_SHA256,
) -> dict[str, Any]:
    require(sha256_file(run_matrix_path) == EXPECTED_RUN_MATRIX_SHA256, "run matrix SHA mismatch")
    require(sha256_file(eval_run_manifest_path) == expected_run_manifest_sha256, "trained-Wan run manifest SHA mismatch")
    require(sha256_file(eval_aggregate_path) == expected_aggregate_sha256, "trained-Wan aggregate SHA mismatch")
    matrix = read_run_matrix(run_matrix_path)
    manifest = load_json(eval_run_manifest_path, "trained-Wan run manifest")
    aggregate = load_json(eval_aggregate_path, "trained-Wan aggregate")
    require(
        aggregate.get("status") == "completed"
        and aggregate.get("expected_videos") == aggregate.get("validated_videos") == 684,
        "trained-Wan aggregate is incomplete",
    )
    require(aggregate.get("status_counts") == {"completed": 18}, "trained-Wan aggregate job counts changed")
    require(aggregate.get("run_manifest", {}).get("sha256") == expected_run_manifest_sha256, "aggregate does not bind run manifest")
    jobs = manifest.get("jobs")
    require(
        manifest.get("protocol_id") == PROTOCOL_VERSION
        and manifest.get("expected_videos") == 684
        and isinstance(jobs, list)
        and len(jobs) == 18,
        "trained-Wan run manifest identity/counts changed",
    )
    by_id = {str(job.get("job_id")): job for job in jobs if isinstance(job, dict)}
    require(set(by_id) == {row["run_id"] for row in matrix}, "trained-Wan jobs differ from run matrix")

    items: list[dict[str, Any]] = []
    for row in matrix:
        job = by_id[row["run_id"]]
        run_id = row["run_id"]
        output_dir = safe_relative(row["output_dir"], f"{run_id} output directory")
        require(
            output_dir.startswith("outputs/causal_role_erasure_7mechanism_main_v2/"),
            f"{run_id} output directory is outside the formal output root",
        )
        expected_receipt = f"{output_dir}/run_receipt.json"
        expected_checkpoint = f"{output_dir}/checkpoint-000200"
        expected_spec = f"outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2/run_specs/{int(row['run_index']):02d}_{run_id}.json"
        receipt_path = logical_a100_path(str(job.get("receipt", "")), expected_receipt, f"{run_id} receipt")
        checkpoint_path = logical_a100_path(str(job.get("checkpoint", "")), expected_checkpoint, f"{run_id} checkpoint")
        spec_path = logical_a100_path(str(job.get("run_spec", "")), expected_spec, f"{run_id} run spec")
        for field in ("receipt_sha256", "run_spec_sha256", "checkpoint_artifact_sha256", "weights_sha256", "training_state_sha256"):
            require(isinstance(job.get(field), str) and HEX64.fullmatch(job[field]) is not None, f"{run_id} {field} invalid")
        expected_group = "main" if row["arm"] in ("matched_control", "V4") else "identification"
        expected_rows = 42 if expected_group == "main" else 24
        require(
            job.get("job_index") == int(row["run_index"])
            and job.get("mechanism") == row["mechanism"]
            and job.get("arm") == row["arm"]
            and job.get("run_group") == expected_group
            and job.get("row_count") == expected_rows,
            f"{run_id} job identity/count mismatch",
        )
        items.append(
            {
                "run_index": int(row["run_index"]),
                "run_id": run_id,
                "mechanism": row["mechanism"],
                "arm": row["arm"],
                "run_group": expected_group,
                "step": 200,
                "erase_updates": 100,
                "preserve_updates": 100,
                "validated_generation_rows": expected_rows,
                "run_spec": {"root_id": "a100_project_root", "path": spec_path, "sha256": job["run_spec_sha256"]},
                "receipt": {"root_id": "a100_project_root", "path": receipt_path, "sha256": job["receipt_sha256"]},
                "checkpoint": {
                    "root_id": "a100_project_root",
                    "path": checkpoint_path,
                    "artifact_sha256": job["checkpoint_artifact_sha256"],
                    "weights_sha256": job["weights_sha256"],
                    "training_state_sha256": job["training_state_sha256"],
                },
            }
        )
    require(len({item["receipt"]["sha256"] for item in items}) == 18, "Wan receipt hashes repeat")
    require(len({item["run_spec"]["sha256"] for item in items}) == 18, "Wan run-spec hashes repeat")
    require(sum(item["validated_generation_rows"] for item in items) == 684, "Wan job rows do not sum to 684")
    arms = Counter(item["arm"] for item in items)
    require(arms == {"matched_control": 7, "V4": 7, "generic_paraphrase": 2, "bystander_token": 2}, "Wan arm inventory changed")
    inventory = with_self_digest(
        {
            "schema_version": 1,
            "protocol": INVENTORY_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": INVENTORY_STATUS,
            "verification_basis": {
                "receipt_bytes_reopened_by_this_builder": False,
                "receipt_bytes_and_semantics_validated_before_frozen_eval_run_manifest": True,
                "source_validator": "scripts/run_causal_role_erasure_7mechanism_eval_v2.py::validate_checkpoint",
            },
            "sources": {
                "run_matrix": repo_ref(project_root, run_matrix_path),
                "eval_run_manifest": file_ref(
                    "media_snapshot_root",
                    "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_run_manifest.json",
                    eval_run_manifest_path,
                ),
                "eval_aggregate": file_ref(
                    "media_snapshot_root",
                    "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_aggregate.json",
                    eval_aggregate_path,
                ),
            },
            "counts": {
                "receipts": 18,
                "main_runs": 14,
                "identification_runs": 4,
                "erase_updates_per_run": 100,
                "preserve_updates_per_run": 100,
                "eligible_checkpoint_step": 200,
                "validated_generation_rows": 684,
            },
            "items": items,
        },
        "inventory_sha256",
    )
    write_json_exclusive(output_path, inventory)
    return inventory


def _command(
    stage_id: str,
    environment_id: str,
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    private_inputs: Sequence[str] = (),
    realized: bool = False,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "environment_id": environment_id,
        "cwd": "${PROJECT_ROOT}",
        "argv": list(argv),
        "environment": dict(sorted((environment or {}).items())),
        "private_inputs": list(private_inputs),
        "command_kind": "realized_launcher" if realized else "release_reproduction_command",
    }


def manual_stage_records(project_root: Path) -> list[dict[str, Any]]:
    """Inventory non-executable review work without claiming complete delivery provenance."""

    target_inputs = {
        role: repo_ref(project_root, project_root / path)
        for role, path in sorted(TARGET_SELECTION_RELEASED_INPUT_PATHS.items())
    }
    evaluation_inputs = {
        "reviewer_instructions": repo_ref(
            project_root,
            project_root / "docs/causal_role_erasure_7mechanism_human_review_v2.md",
        )
    }

    def record(
        stage_id: str,
        phase: str,
        completion_state: str,
        parents: Sequence[str],
        instruction: str,
        released_inputs: Mapping[str, Mapping[str, Any]],
        unavailable_inputs: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "stage_id": stage_id,
            "phase": phase,
            "completion_state_at_ledger_freeze": completion_state,
            "parent_stage_ids": list(parents),
            "instruction": instruction,
            "released_input_refs": {
                role: dict(ref) for role, ref in sorted(released_inputs.items())
            },
            "unreleased_or_pending_input_roles": list(unavailable_inputs),
        }

    target_state = "completed_before_ledger_freeze"
    target_phase = "target_selection"
    target_unreleased = (
        "anonymous_reviewer_assignment",
        "anonymous_49_frame_media_delivery",
        "completed_private_score_file",
        "exact_historical_delivery_receipt",
    )
    records = [
        record(
            "target_selection_reviewer_a",
            target_phase,
            target_state,
            (),
            "complete the full anonymous Reviewer-A six-field target-candidate labels",
            target_inputs,
            target_unreleased,
        )
    ]
    for index in range(1, 5):
        records.append(
            record(
                f"target_selection_blind_audit_{index}",
                target_phase,
                target_state,
                (),
                "complete the independently assigned targeted field audit without access to Reviewer-A or other audit scores",
                target_inputs,
                (
                    "targeted_anonymous_audit_assignment",
                    "anonymous_49_frame_media_delivery",
                    "completed_private_audit_file",
                    "exact_historical_delivery_receipt",
                ),
            )
        )
    target_parents = [
        "target_selection_reviewer_a",
        *(f"target_selection_blind_audit_{index}" for index in range(1, 5)),
    ]
    records.append(
        record(
            "target_selection_blind_adjudication",
            target_phase,
            target_state,
            target_parents,
            "score only selection-affecting disputed fields without access to either parent score",
            target_inputs,
            (
                "blind_disagreement_projection",
                "anonymous_49_frame_media_delivery",
                "completed_private_adjudication_file",
                "exact_historical_delivery_receipt",
            ),
        )
    )

    pending_state = "required_after_ledger_freeze"
    evaluation_phase = "human_canonical_evaluation"
    records.extend(
        [
            record(
                "initial_reviewer_1",
                evaluation_phase,
                pending_state,
                (),
                "complete only reviewer_1 score/note columns in the initial isolated public-only copy",
                evaluation_inputs,
                ("initial_reviewer_1_projection_and_media",),
            ),
            record(
                "initial_reviewer_2",
                evaluation_phase,
                pending_state,
                (),
                "complete only reviewer_2 score/note columns independently in the initial isolated public-only copy",
                evaluation_inputs,
                ("initial_reviewer_2_projection_and_media",),
            ),
            record(
                "initial_adjudication",
                evaluation_phase,
                pending_state,
                ("initial_reviewer_1", "initial_reviewer_2"),
                "adjudicate every initial-round disagreement after both reviewer files are frozen",
                evaluation_inputs,
                ("initial_completed_reviews_and_disagreement_projection",),
            ),
            record(
                "final_reviewer_1",
                evaluation_phase,
                pending_state,
                ("initial_reviewer_1", "initial_adjudication"),
                "in the reviewer-specific final projection, preserve reviewer_1 prior labels and score only newly blank rows",
                evaluation_inputs,
                ("final_reviewer_1_projection_and_media",),
            ),
            record(
                "final_reviewer_2",
                evaluation_phase,
                pending_state,
                ("initial_reviewer_2", "initial_adjudication"),
                "in the reviewer-specific final projection, preserve reviewer_2 prior labels and score only newly blank rows",
                evaluation_inputs,
                ("final_reviewer_2_projection_and_media",),
            ),
            record(
                "final_adjudication",
                evaluation_phase,
                pending_state,
                ("final_reviewer_1", "final_reviewer_2"),
                "adjudicate every new final-round disagreement after both final reviewer files are frozen",
                evaluation_inputs,
                ("final_completed_reviews_and_disagreement_projection",),
            ),
        ]
    )
    require(
        len(records) == 12
        and len({row["stage_id"] for row in records}) == len(records)
        and all(set(row) == MANUAL_STAGE_FIELDS for row in records),
        "manual-stage inventory construction failed",
    )
    return records


def command_records() -> list[dict[str, Any]]:
    py = "${A100_PYTHON}"
    ev = "${EVAL_PYTHON}"
    environment_receipt_command = [
        ev,
        "scripts/build_causal_role_erasure_7mechanism_reproducibility_v1.py",
        "freeze-environment",
        "--project-root",
        "${PROJECT_ROOT}",
        "--lock",
        "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock",
    ]
    for test_path in RELEASE_TEST_PATHS:
        environment_receipt_command.extend(("--test-file", test_path))
    environment_receipt_command.extend(
        ("--output", "${ENVIRONMENT_RECEIPT_ROOT}/evaluation_environment_receipt_v1.json")
    )
    commands = [
        _command("static_build", "a100", [py, "scripts/build_causal_role_erasure_7mechanism_main_v2.py", "--eval-seed-salt", "${PRIVATE_EVAL_SEED_SALT}", "--output-dir", "data/causal_role_erasure_7mechanism_main_v2", "--prompt-dir", "prompts/causal_role_erasure_7mechanism_main_v2", "--water-source-bank", "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json", "--water-train-pairs", "data/water_impact_dynamic_v1/train_pairs.csv", "--water-screen", "data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv", "--preserve-manifest", "data/protocol_v1/preserve_manifest.csv"], private_inputs=["PRIVATE_EVAL_SEED_SALT"]),
        _command("target_generation", "a100", [py, "scripts/run_causal_role_erasure_7mechanism_targets_v2.py", "--project-root", "${PROJECT_ROOT}", "--target-candidates", "data/causal_role_erasure_7mechanism_main_v2/target_candidates.csv", "--water-reuse-manifest", "outputs/water_impact_dynamic_v1/train_targets_v1/generation_manifest.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/target_generation", "--python-executable", "models/.wan-runtime/bin/python", "--model", "models/Wan2.1-T2V-1.3B-Diffusers", "--gpus", "0,1,2,3", "--poll-interval", "5", "--run"]),
        _command("target_selection", "cpu", [ev, "scripts/finalize_causal_role_erasure_7mechanism_targeted_audit_v1.py", "--project-root", "${PROJECT_ROOT}", "finalize", "--reviewer-a-completed", "${TARGET_REVIEWER_A_COMPLETED}", "--reviewer-a-assignment", "${TARGET_REVIEWER_A_ASSIGNMENT}", "--agent-audit", "${TARGET_AGENT_AUDIT_1}", "--agent-audit", "${TARGET_AGENT_AUDIT_2}", "--agent-audit", "${TARGET_AGENT_AUDIT_3}", "--agent-audit", "${TARGET_AGENT_AUDIT_4}", "--reviewer-a-binding", "${PRIVATE_TARGET_REVIEWER_BINDING}", "--candidate-manifest", "${TARGET_CANDIDATE_MANIFEST}", "--adjudication-json", "${TARGET_ADJUDICATION}", "--water-screen-csv", "data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv", "--output-root", "${TARGET_SELECTION_OUTPUT_ROOT}"], private_inputs=["PRIVATE_TARGET_REVIEWER_BINDING"]),
        _command("training_inputs", "a100", [py, "scripts/build_causal_role_erasure_7mechanism_training_registries_v2.py", "--project-root", "${PROJECT_ROOT}", "build-inputs", "--selected-targets", "${SELECTED_TARGETS}", "--selected-targets-sha256", "${SELECTED_TARGETS_SHA256}", "--ontology-registry", "data/causal_role_erasure_7mechanism_main_v2/ontology_registry.json", "--ontology-registry-sha256", "37ef4920665e0cbcaae86e0b31257708facc882bb6a8478c6b9e66ecd50e2036", "--run-matrix", "data/causal_role_erasure_7mechanism_main_v2/run_matrix.csv", "--run-matrix-sha256", EXPECTED_RUN_MATRIX_SHA256, "--preserve-manifest", "data/protocol_v1/preserve_manifest.csv", "--preserve-manifest-sha256", "${PRESERVE_MANIFEST_SHA256}", "--model-inventory", "data/water_impact_dynamic_v4/v4_model_content_inventory_v3.json", "--model-inventory-sha256", "51e7199b99ee206934924ee043bd01b40ba413dfb60ef1e72682d36a10b46290", "--runtime-registry", "data/water_impact_dynamic_v4/v4_runtime_registry_v3.json", "--runtime-registry-sha256", "9043adf7f823022b20711267ab9e28b9dfb72452273d26e27c131b92e65eff01", "--preserve-media-root", "${PRESERVE_MEDIA_ROOT}", "--model-root", "models/Wan2.1-T2V-1.3B-Diffusers", "--runtime-root", "models/.wan-runtime", "--python-executable", "models/.wan-runtime/bin/python", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/training_inputs_v2", "--cache-output-root", "outputs/causal_role_erasure_7mechanism_main_v2/training_caches_v2", "--run-spec-output-root", "outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2"]),
        _command("training_run_specs", "a100", [py, "scripts/build_causal_role_erasure_7mechanism_training_registries_v2.py", "--project-root", "${PROJECT_ROOT}", "finalize-runs", "--input-registry", "outputs/causal_role_erasure_7mechanism_main_v2/training_inputs_v2/training_input_registry.json", "--input-registry-sha256", "${TRAINING_INPUT_REGISTRY_SHA256}", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/training_run_specs_v2"]),
        _command("wan_cache", "a100", ["bash", "scripts/run_causal_role_erasure_7mechanism_cache_queue_v2.sh"], environment={"GPU_DEVICES": "0,1,2,3", "MAX_UTILIZATION": "20", "MIN_FREE_MIB": "60000", "POLL_SECONDS": "60"}),
        _command("wan_training", "a100", ["bash", "scripts/run_causal_role_erasure_7mechanism_training_queue_v2.sh"], environment={"GPU_DEVICES": "0,1,2,3", "MAX_UTILIZATION": "20", "MIN_FREE_MIB": "60000", "POLL_SECONDS": "60", "SLOTS_PER_GPU": "2"}),
        _command("t2v_registry", "a100", [py, "scripts/build_causal_role_erasure_7mechanism_t2v_training_registry_v2.py", "--project-root", "${PROJECT_ROOT}", "--training-input-registry", "outputs/causal_role_erasure_7mechanism_main_v2/training_inputs_v2/training_input_registry.json", "--model-root", "models/CogVideoX-2b", "--model-inventory", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_pre_t2v/model_inventory.json", "--runtime-python", "models/.wan-runtime/bin/python", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2", "--checkpoint-root", "outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_v2", "--train-rows-per-mechanism", "36", "--max-steps", "100", "--eligible-checkpoint-step", "100"]),
        _command("t2v_training", "a100", ["bash", "scripts/run_causal_role_erasure_7mechanism_t2v_training_queue_v2.sh"], environment={"PROJECT_ROOT": "${PROJECT_ROOT}", "T2V_GPUS": "0,1,2,3", "T2V_PYTHON": "${A100_PYTHON}", "T2V_TRAINING_REGISTRY": "${PROJECT_ROOT}/outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2/t2v_training_registry.json"}),
        _command("baseline_registry", "a100", [py, "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py", "--project-root", "${PROJECT_ROOT}", "--formal-cases", "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", "--model-root", "models/CogVideoX-2b", "--runtime-python", "models/.wan-runtime/bin/python", "--videoeraser-source-root", "${VIDEOERASER_SOURCE_ROOT}", "--safree-root", "${SAFREE_SOURCE_ROOT}", "--safree-expected-commit", "b8b2c3fa9d7f51c46f5a570170503fc98bd9c7ec", "--safree-expected-pipeline-sha256", "55185f5972ec0942e9ba653176c14c9ecd5ac0e19ef308d92e31439a10ce2609", "--t2v-registry", "outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2/t2v_training_registry.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final", "--require-complete"]),
        _command("wan_original_generation", "a100", ["bash", "scripts/run_causal_role_erasure_7mechanism_wan_original_queue_v2.sh"], environment={"GPUS": "0,1,2,3", "PROJECT_ROOT": "${PROJECT_ROOT}", "OUTPUT_ROOT": "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2", "LOG_ROOT": "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_queue_logs_v2"}, realized=True),
        _command("trained_wan_generation", "a100", ["bash", "scripts/run_causal_role_erasure_7mechanism_eval_queue_v2.sh"], environment={"GPUS": "0,1,2,3", "PROJECT_ROOT": "${PROJECT_ROOT}", "OUTPUT_ROOT": "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2", "LOG_ROOT": "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_queue_logs_v2"}, realized=True),
        _command("cogvideox_core_plan", "a100", [py, "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py", "--project-root", "${PROJECT_ROOT}", "--baseline-registry", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json", "--baseline-receipt", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/build_receipt.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2", "--streams", "cogvideox_original,negative_prompt,videoeraser_official,t2vunlearning_adapted", "--gpus", "0,1,2,3", "--poll-interval", "5", "--dry-run"]),
        _command("cogvideox_core_generation", "a100", [py, "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py", "--project-root", "${PROJECT_ROOT}", "--baseline-registry", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json", "--baseline-receipt", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/build_receipt.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2", "--streams", "cogvideox_original,negative_prompt,videoeraser_official,t2vunlearning_adapted", "--gpus", "0,1,2,3", "--poll-interval", "5", "--resume"], realized=True),
        _command("safree_plan", "a100", [py, "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py", "--project-root", "${PROJECT_ROOT}", "--baseline-registry", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json", "--baseline-receipt", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/build_receipt.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2", "--streams", "safree_cogvideox", "--gpus", "0,1,2,3", "--safree-max-concurrency", "1", "--poll-interval", "5", "--dry-run"]),
        _command("safree_generation", "a100", [py, "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py", "--project-root", "${PROJECT_ROOT}", "--baseline-registry", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/baseline_registry.json", "--baseline-receipt", "outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2_final/build_receipt.json", "--output-root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2", "--streams", "safree_cogvideox", "--gpus", "0,1,2,3", "--safree-max-concurrency", "1", "--poll-interval", "5", "--resume"], realized=True),
        _command("anonymous_review_package", "evaluation", [ev, "scripts/run_causal_role_erasure_7mechanism_formal_review_v1.py", "--project-root", "${PROJECT_ROOT}", "--workspace-root", "${WORKSPACE_ROOT}", "--authority-snapshot", "${MEDIA_SNAPSHOT_ROOT}", "--wan-original-snapshot", "${WAN_ORIGINAL_SNAPSHOT_ROOT}", "--launch-root", "${REVIEW_ROOT}", "--blind-key-file", "${PRIVATE_REVIEW_BLIND_KEY}", "--prepare-package"], private_inputs=["PRIVATE_REVIEW_BLIND_KEY"]),
        _command("vlm_preflight", "evaluation_hosted", [ev, "scripts/run_causal_role_erasure_7mechanism_formal_review_v1.py", "--project-root", "${PROJECT_ROOT}", "--review-package-public", "${REVIEW_ROOT}/review_package/public", "--launch-root", "${REVIEW_ROOT}", "--copilot-key-file", "${PRIVATE_COPILOT_KEY_FILE}", "--preflight", "--preflight-pass", "pass_a", "--preflight-start", "0", "--preflight-limit", "20", "--preflight-workers", "16", "--proxy-version", "2.2.15", "--timeout", "180"], private_inputs=["PRIVATE_COPILOT_KEY_FILE"]),
        _command("vlm_two_pass_review", "evaluation_hosted", [ev, "scripts/run_causal_role_erasure_7mechanism_formal_review_v1.py", "--project-root", "${PROJECT_ROOT}", "--review-package-public", "${REVIEW_ROOT}/review_package/public", "--launch-root", "${REVIEW_ROOT}", "--copilot-key-file", "${PRIVATE_COPILOT_KEY_FILE}", "--preflight-receipt", "${REVIEW_ROOT}/preflight/preflight_receipt.json", "--run-formal", "--proxy-version", "2.2.15", "--shard-size", "256", "--timeout", "180", "--max-retry-rounds", "3"], private_inputs=["PRIVATE_COPILOT_KEY_FILE"]),
        _command("initial_human_audit_queue", "evaluation", [ev, "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py", "--project-root", "${PROJECT_ROOT}", "build-audit", "--scores-a", "${REVIEW_ROOT}/merged/pass_a/completed_scores.jsonl", "--scores-b", "${REVIEW_ROOT}/merged/pass_b/completed_scores.jsonl", "--assignments", "${REVIEW_ROOT}/review_package/public/pass_a/assignments.jsonl", "--audit-strata-key", "${PRIVATE_AUDIT_STRATA_KEY}", "--key-commitments", "${REVIEW_ROOT}/review_package/public/key_commitments.json", "--output-root", "${INITIAL_AUDIT_ROOT}"], private_inputs=["PRIVATE_AUDIT_STRATA_KEY"], realized=True),
        _command("release_evaluation_environment_receipt", "evaluation", environment_receipt_command),
        _command("pre_metric_code_freeze", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_pre_metric_freeze_v2.py", "--project-root", "${PROJECT_ROOT}", "--review-root", "${REVIEW_ROOT}", "--media-snapshot-root", "${MEDIA_SNAPSHOT_ROOT}", "--wan-snapshot-root", "${WAN_ORIGINAL_SNAPSHOT_ROOT}", "--output-root", "${PRE_METRIC_ROOT}", "--environment-lock", "reproducibility/causal_role_erasure_7m/evaluation-reproduction-v1.lock", "--environment-receipt", "${ENVIRONMENT_RECEIPT_ROOT}/evaluation_environment_receipt_v1.json", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--canonicalization-wrapper", "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py", "--formal-metrics-wrapper", "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py", "--final-table-exporter", "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py", "--extra-component", "review_package_tests=tests/test_build_causal_role_erasure_7mechanism_review_package_v1.py", "--extra-component", "review_schema_tests=tests/test_review_causal_role_erasure_7mechanism_formal_v2.py", "--extra-component", "formal_review_tests=tests/test_run_causal_role_erasure_7mechanism_formal_review_v1.py", "--extra-component", "evaluation_registry_tests=tests/test_causal_role_erasure_7mechanism_evaluation_code_registry_v1.py", "--extra-component", "formal_metrics_v1_tests=tests/test_causal_role_erasure_7mechanism_formal_metrics_v1.py", "--extra-component", "evaluation_v2_tests=tests/test_causal_role_erasure_7mechanism_evaluation_v2.py", "--extra-component", "final_exporter_tests=tests/test_export_causal_role_erasure_7mechanism_paper_tables_v1.py", "--extra-component", "reproducibility_tests=tests/test_causal_role_erasure_7mechanism_reproducibility_v1.py", "--extra-component", "review_process_tests=tests/test_causal_role_erasure_7mechanism_review_process_receipt_v2.py"]),
        _command("validate_provenance_amendment", "evaluation", [ev, "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py", "--project-root", "${PROJECT_ROOT}", "validate-amendment", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--pre-metric-freeze-manifest", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json"]),
        _command("reviewer_1_initial_delivery", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "prepare-delivery", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${INITIAL_AUDIT_ROOT}", "--audit-manifest", "${INITIAL_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${INITIAL_AUDIT_ROOT}/public", "--reviewer-id", "reviewer_1", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--delivery-root", "${INITIAL_REVIEWER_1_DELIVERY_ROOT}", "--output-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_1_delivery.json"]),
        _command("reviewer_2_initial_delivery", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "prepare-delivery", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${INITIAL_AUDIT_ROOT}", "--audit-manifest", "${INITIAL_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${INITIAL_AUDIT_ROOT}/public", "--reviewer-id", "reviewer_2", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--delivery-root", "${INITIAL_REVIEWER_2_DELIVERY_ROOT}", "--output-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_2_delivery.json"]),
        _command("initial_human_review_process_receipt", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "freeze-process", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${INITIAL_AUDIT_ROOT}", "--audit-manifest", "${INITIAL_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${INITIAL_AUDIT_ROOT}/public", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--reviewer-1-delivery-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_1_delivery.json", "--reviewer-2-delivery-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_2_delivery.json", "--reviewer-1-completed", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_1_completed.csv", "--reviewer-2-completed", "${INITIAL_REVIEW_PROCESS_ROOT}/reviewer_2_completed.csv", "--independence-attestation", "${INITIAL_REVIEW_PROCESS_ROOT}/independence_attestation.json", "--completed-human", "${INITIAL_REVIEW_PROCESS_ROOT}/completed_human.csv", "--output-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/review_process_receipt_v2.json"]),
        _command("audit_expansion", "evaluation", [ev, "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v1.py", "--project-root", "${PROJECT_ROOT}", "expand-audit", "--audit-root", "${INITIAL_AUDIT_ROOT}", "--completed-human", "${INITIAL_REVIEW_PROCESS_ROOT}/completed_human.csv", "--output-root", "${EXPANDED_AUDIT_ROOT}"]),
        _command("reviewer_1_final_delivery", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "prepare-delivery", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${EXPANDED_AUDIT_ROOT}", "--audit-manifest", "${EXPANDED_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${EXPANDED_AUDIT_ROOT}/public", "--reviewer-id", "reviewer_1", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--delivery-root", "${FINAL_REVIEWER_1_DELIVERY_ROOT}", "--output-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_1_delivery.json"]),
        _command("reviewer_2_final_delivery", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "prepare-delivery", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${EXPANDED_AUDIT_ROOT}", "--audit-manifest", "${EXPANDED_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${EXPANDED_AUDIT_ROOT}/public", "--reviewer-id", "reviewer_2", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--delivery-root", "${FINAL_REVIEWER_2_DELIVERY_ROOT}", "--output-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_2_delivery.json"]),
        _command("final_human_review_process_receipt", "evaluation", [ev, "scripts/build_causal_role_erasure_7mechanism_review_process_receipt_v2.py", "freeze-process", "--project-root", "${PROJECT_ROOT}", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--review-root", "${EXPANDED_AUDIT_ROOT}", "--audit-manifest", "${EXPANDED_AUDIT_ROOT}/audit_manifest.json", "--public-root", "${EXPANDED_AUDIT_ROOT}/public", "--reviewer-instructions", "docs/causal_role_erasure_7mechanism_human_review_v2.md", "--reviewer-1-delivery-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_1_delivery.json", "--reviewer-2-delivery-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_2_delivery.json", "--reviewer-1-completed", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_1_completed.csv", "--reviewer-2-completed", "${FINAL_REVIEW_PROCESS_ROOT}/reviewer_2_completed.csv", "--independence-attestation", "${FINAL_REVIEW_PROCESS_ROOT}/independence_attestation.json", "--completed-human", "${FINAL_REVIEW_PROCESS_ROOT}/completed_human.csv", "--initial-process-receipt", "${INITIAL_REVIEW_PROCESS_ROOT}/review_process_receipt_v2.json", "--output-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/review_process_receipt_v2.json"]),
        _command("freeze_canonical_scores", "evaluation", [ev, "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py", "--project-root", "${PROJECT_ROOT}", "freeze", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--pre-metric-freeze-manifest", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--audit-root", "${EXPANDED_AUDIT_ROOT}", "--completed-human", "${FINAL_REVIEW_PROCESS_ROOT}/completed_human.csv", "--review-process-receipt", "${FINAL_REVIEW_PROCESS_ROOT}/review_process_receipt_v2.json", "--scores-a", "${REVIEW_ROOT}/merged/pass_a/completed_scores.jsonl", "--scores-b", "${REVIEW_ROOT}/merged/pass_b/completed_scores.jsonl", "--output-root", "${CANONICAL_ROOT}"]),
        _command("freeze_original_eligibility", "evaluation", [ev, "scripts/canonicalize_causal_role_erasure_7mechanism_formal_v2.py", "--project-root", "${PROJECT_ROOT}", "freeze-eligibility", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--pre-metric-freeze-manifest", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--canonical-root", "${CANONICAL_ROOT}", "--original-key", "${PRIVATE_ORIGINAL_KEY}", "--key-commitments", "${REVIEW_ROOT}/review_package/public/key_commitments.json", "--output-root", "${ELIGIBILITY_ROOT}"], private_inputs=["PRIVATE_ORIGINAL_KEY"]),
        _command("formal_metrics", "evaluation", [ev, "scripts/compute_causal_role_erasure_7mechanism_formal_metrics_v2.py", "--project-root", "${PROJECT_ROOT}", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--pre-metric-freeze-manifest", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--canonical-root", "${CANONICAL_ROOT}", "--eligibility-root", "${ELIGIBILITY_ROOT}", "--full-key", "${PRIVATE_FULL_KEY}", "--key-commitments", "${REVIEW_ROOT}/review_package/public/key_commitments.json", "--formal-cases", "data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv", "--output-root", "${METRICS_ROOT}"], private_inputs=["PRIVATE_FULL_KEY"]),
        _command("paper_tables", "evaluation", [ev, "scripts/export_causal_role_erasure_7mechanism_paper_tables_v1.py", "--repo-root", "${PROJECT_ROOT}", "--review-root", "${REVIEW_ROOT}", "--eligibility-root", "${ELIGIBILITY_ROOT}", "--score-manifest", "${METRICS_ROOT}/score_manifest.json", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--amendment", "data/causal_role_erasure_7mechanism_main_v2/evaluation_provenance_amendment_v2.json", "--output-root", "${PROJECT_ROOT}/paper/tables/human_canonical_v1"]),
        _command("artifact_dag_final_materialize", "evaluation", [ev, "scripts/verify_causal_role_erasure_7mechanism_artifact_dag_v1.py", "--repo-root", "${PROJECT_ROOT}", "--review-root", "${REVIEW_ROOT}", "--media-snapshot-root", "${MEDIA_SNAPSHOT_ROOT}", "--wan-snapshot-root", "${WAN_ORIGINAL_SNAPSHOT_ROOT}", "--pre-metric-root", "${PRE_METRIC_ROOT}", "--root", "final_release_root=${FINAL_RELEASE_ROOT}", "--root", "final_audit_root=${EXPANDED_AUDIT_ROOT}", "--root", "final_review_process_root=${FINAL_REVIEW_PROCESS_ROOT}", "--root", "canonical_root=${CANONICAL_ROOT}", "--root", "eligibility_root=${ELIGIBILITY_ROOT}", "--root", "metrics_root=${METRICS_ROOT}", "--root", "paper_table_root=${PROJECT_ROOT}/paper/tables/human_canonical_v1", "--manifest", "${FINAL_RELEASE_ROOT}/artifact_dag_release_v1.json", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--mode", "final", "--build-final", "--unavailable-sha256", "target_generation_aggregate=${TARGET_GENERATION_AGGREGATE_SHA256}", "--unavailable-sha256", "target_selection_registry=${TARGET_SELECTION_REGISTRY_SHA256}", "--unavailable-sha256", "selected_targets=${SELECTED_TARGETS_FULL_SHA256}", "--unavailable-sha256", "training_input_registry=${TRAINING_INPUT_REGISTRY_FULL_SHA256}", "--unavailable-sha256", "wan_run_spec_registry=${WAN_RUN_SPEC_REGISTRY_SHA256}", "--unavailable-sha256", "t2v_training_registry=${T2V_TRAINING_REGISTRY_SHA256}"]),
        _command("artifact_dag_final", "evaluation", [ev, "scripts/verify_causal_role_erasure_7mechanism_artifact_dag_v1.py", "--repo-root", "${PROJECT_ROOT}", "--review-root", "${REVIEW_ROOT}", "--media-snapshot-root", "${MEDIA_SNAPSHOT_ROOT}", "--wan-snapshot-root", "${WAN_ORIGINAL_SNAPSHOT_ROOT}", "--pre-metric-root", "${PRE_METRIC_ROOT}", "--root", "final_release_root=${FINAL_RELEASE_ROOT}", "--root", "final_audit_root=${EXPANDED_AUDIT_ROOT}", "--root", "final_review_process_root=${FINAL_REVIEW_PROCESS_ROOT}", "--root", "canonical_root=${CANONICAL_ROOT}", "--root", "eligibility_root=${ELIGIBILITY_ROOT}", "--root", "metrics_root=${METRICS_ROOT}", "--root", "paper_table_root=${PROJECT_ROOT}/paper/tables/human_canonical_v1", "--manifest", "${FINAL_RELEASE_ROOT}/artifact_dag_release_v1.json", "--pre-metric-freeze", "${PRE_METRIC_ROOT}/pre_metric_code_freeze_v2.json", "--mode", "final"]),
        _command("submission_gate", "evaluation", [ev, "paper/scripts/check_submission.py"]),
    ]
    return commands


def validate_commands(project_root: Path, commands: Sequence[Mapping[str, Any]], placeholders: Mapping[str, Any]) -> None:
    require(len({record.get("stage_id") for record in commands}) == len(commands), "command stage IDs repeat")
    known = set(placeholders)
    for record in commands:
        require(set(record) == COMMAND_FIELDS, "command record schema changed")
        require(
            isinstance(record.get("stage_id"), str)
            and re.fullmatch(r"[a-z][a-z0-9_]*", record["stage_id"]) is not None,
            "command stage ID is unsafe",
        )
        require(record.get("environment_id") in COMMAND_ENVIRONMENTS, "command environment ID is invalid")
        require(record.get("cwd") == "${PROJECT_ROOT}", "command working directory changed")
        require(record.get("command_kind") in ("realized_launcher", "release_reproduction_command"), "command kind is invalid")
        argv = record.get("argv")
        require(isinstance(argv, list) and len(argv) >= 2 and all(isinstance(value, str) and value for value in argv), "command argv malformed")
        environment = record.get("environment")
        require(
            isinstance(environment, dict)
            and all(
                isinstance(name, str)
                and re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is not None
                and isinstance(value, str)
                and value
                for name, value in environment.items()
            ),
            "command environment mapping malformed",
        )
        private_inputs = record.get("private_inputs")
        require(
            isinstance(private_inputs, list)
            and len(private_inputs) == len(set(private_inputs))
            and all(isinstance(value, str) and value.startswith("PRIVATE_") for value in private_inputs),
            "command private-input inventory malformed",
        )
        serialized = json.dumps(record, sort_keys=True)
        require("/Users/" not in serialized and "/data/" not in serialized, "command contains a machine-specific absolute path")
        used = set(PLACEHOLDER.findall(serialized))
        require(used <= known, f"command uses undefined placeholders: {sorted(used-known)}")
        require(
            set(private_inputs) == {name for name in used if name.startswith("PRIVATE_")},
            "command private-input declaration differs from its placeholders",
        )
        script = argv[1]
        require(script.startswith("scripts/") or script.startswith("paper/"), "command does not name a repository entry point")
        regular_file(project_root / safe_relative(script, "command script"), "command script")


def freeze_command_ledger(
    *,
    project_root: Path,
    wan_original_run_manifest: Path,
    trained_wan_run_manifest: Path,
    core_queue_plan: Path,
    safree_queue_plan: Path,
    formal_launch_receipt: Path,
    output_path: Path,
) -> dict[str, Any]:
    evidence_specs = (
        ("wan_original", "wan_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_wan_original_v2/wan_original_run_manifest.json", wan_original_run_manifest, EXPECTED_WAN_ORIGINAL_RUN_MANIFEST_SHA256, 7, 294),
        ("trained_wan", "media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_eval_v2/eval_run_manifest.json", trained_wan_run_manifest, EXPECTED_EVAL_RUN_MANIFEST_SHA256, 18, 684),
        ("cogvideox_core", "media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_core4_v2/queue_plan.json", core_queue_plan, EXPECTED_CORE_QUEUE_PLAN_SHA256, 28, 1176),
        ("safree", "media_snapshot_root", "outputs/causal_role_erasure_7mechanism_main_v2/formal_baselines_safree_v2/queue_plan.json", safree_queue_plan, EXPECTED_SAFREE_QUEUE_PLAN_SHA256, 7, 294),
    )
    evidence: dict[str, Any] = {}
    for name, root_id, logical, path, expected_sha, jobs, videos in evidence_specs:
        require(sha256_file(path) == expected_sha, f"{name} realized command evidence SHA mismatch")
        payload = load_json(path, f"{name} command evidence")
        require(isinstance(payload.get("jobs"), list) and len(payload["jobs"]) == jobs, f"{name} job count mismatch")
        require(payload.get("expected_videos") == videos, f"{name} video count mismatch")
        if "job_count" in payload:
            require(payload["job_count"] == jobs, f"{name} declared job count mismatch")
        require(
            all(
                isinstance(job, dict)
                and isinstance(job.get("command"), list)
                and len(job["command"]) >= 2
                and all(isinstance(value, str) and value for value in job["command"])
                for job in payload["jobs"]
            ),
            f"{name} lacks complete expanded commands",
        )
        per_job_counts = [
            job.get("row_count", job.get("expected_videos")) for job in payload["jobs"]
        ]
        require(
            all(type(count) is int and count > 0 for count in per_job_counts)
            and sum(per_job_counts) == videos,
            f"{name} expanded command video counts mismatch",
        )
        evidence[name] = {**file_ref(root_id, logical, path), "expanded_jobs": jobs, "expected_videos": videos}
    require(sha256_file(formal_launch_receipt) == EXPECTED_FORMAL_LAUNCH_RECEIPT_SHA256, "formal launch receipt SHA mismatch")
    launch = load_json(formal_launch_receipt, "formal launch receipt")
    require(launch.get("status") == "completed_two_independent_schema_valid_passes", "formal launch incomplete")
    evidence["formal_vlm_launch"] = file_ref("review_root", "formal_launch_receipt.json", formal_launch_receipt)

    placeholders = {
        name: {"kind": "private" if name.startswith("PRIVATE_") else "path_or_value", "must_be_supplied": True}
        for name in (
            "A100_PYTHON", "CANONICAL_ROOT", "ELIGIBILITY_ROOT", "ENVIRONMENT_RECEIPT_ROOT", "EVAL_PYTHON",
            "EXPANDED_AUDIT_ROOT", "FINAL_RELEASE_ROOT", "FINAL_REVIEW_PROCESS_ROOT", "FINAL_REVIEWER_1_DELIVERY_ROOT",
            "FINAL_REVIEWER_2_DELIVERY_ROOT", "INITIAL_AUDIT_ROOT", "INITIAL_REVIEW_PROCESS_ROOT",
            "INITIAL_REVIEWER_1_DELIVERY_ROOT", "INITIAL_REVIEWER_2_DELIVERY_ROOT", "MEDIA_SNAPSHOT_ROOT", "METRICS_ROOT",
            "PRE_METRIC_ROOT", "PRESERVE_MANIFEST_SHA256",
            "PRESERVE_MEDIA_ROOT", "PRIVATE_AUDIT_STRATA_KEY", "PRIVATE_COPILOT_KEY_FILE", "PRIVATE_EVAL_SEED_SALT", "PRIVATE_FULL_KEY",
            "PRIVATE_ORIGINAL_KEY", "PRIVATE_REVIEW_BLIND_KEY", "PRIVATE_TARGET_REVIEWER_BINDING", "PROJECT_ROOT",
            "REVIEWER_1_DELIVERY_ROOT", "REVIEWER_2_DELIVERY_ROOT", "REVIEW_ROOT", "SAFREE_SOURCE_ROOT", "SELECTED_TARGETS", "SELECTED_TARGETS_SHA256",
            "TARGET_ADJUDICATION", "TARGET_AGENT_AUDIT_1", "TARGET_AGENT_AUDIT_2", "TARGET_AGENT_AUDIT_3",
            "TARGET_AGENT_AUDIT_4", "TARGET_CANDIDATE_MANIFEST", "TARGET_REVIEWER_A_ASSIGNMENT",
            "TARGET_REVIEWER_A_COMPLETED", "TARGET_SELECTION_OUTPUT_ROOT", "TARGET_GENERATION_AGGREGATE_SHA256",
            "TARGET_SELECTION_REGISTRY_SHA256", "SELECTED_TARGETS_FULL_SHA256", "TRAINING_INPUT_REGISTRY_FULL_SHA256",
            "TRAINING_INPUT_REGISTRY_SHA256", "WAN_RUN_SPEC_REGISTRY_SHA256", "T2V_TRAINING_REGISTRY_SHA256",
            "VIDEOERASER_SOURCE_ROOT", "WAN_ORIGINAL_SNAPSHOT_ROOT", "WORKSPACE_ROOT",
        )
    }
    placeholders["EVAL_PYTHON"] = EVAL_PYTHON_PLACEHOLDER
    commands = command_records()
    validate_commands(project_root, commands, placeholders)
    entry_points = {
        path: repo_ref(project_root, project_root / path)
        for path in sorted({str(record["argv"][1]) for record in commands})
    }
    ledger = with_self_digest(
        {
            "schema_version": 1,
            "protocol": LEDGER_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": LEDGER_STATUS,
            "scope": LEDGER_SCOPE,
            "placeholders": placeholders,
            "commands": commands,
            "entry_points": entry_points,
            "counts": {
                "command_records": len(commands),
                "manual_stages": 12,
                "realized_expanded_generation_commands": sum(
                    item[5] for item in evidence_specs
                ),
            },
            "manual_stages": manual_stage_records(project_root),
            "realized_command_evidence": evidence,
        },
        "ledger_sha256",
    )
    write_json_exclusive(output_path, ledger)
    return ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser("freeze-environment")
    environment.add_argument("--project-root", type=Path, required=True)
    environment.add_argument("--lock", type=Path, required=True)
    environment.add_argument("--test-file", type=Path, action="append", required=True)
    environment.add_argument("--skip-tests", action="store_true")
    environment.add_argument("--output", type=Path, required=True)

    inventory = commands.add_parser("freeze-wan-receipts")
    inventory.add_argument("--project-root", type=Path, required=True)
    inventory.add_argument("--run-matrix", type=Path, required=True)
    inventory.add_argument("--eval-run-manifest", type=Path, required=True)
    inventory.add_argument("--eval-aggregate", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)

    ledger = commands.add_parser("freeze-command-ledger")
    ledger.add_argument("--project-root", type=Path, required=True)
    ledger.add_argument("--wan-original-run-manifest", type=Path, required=True)
    ledger.add_argument("--trained-wan-run-manifest", type=Path, required=True)
    ledger.add_argument("--core-queue-plan", type=Path, required=True)
    ledger.add_argument("--safree-queue-plan", type=Path, required=True)
    ledger.add_argument("--formal-launch-receipt", type=Path, required=True)
    ledger.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-environment":
            result = freeze_environment_receipt(
                project_root=args.project_root,
                lock_path=args.lock,
                output_path=args.output,
                test_files=args.test_file,
                run_tests=not args.skip_tests,
            )
            digest_field = "receipt_sha256"
        elif args.command == "freeze-wan-receipts":
            result = freeze_wan_receipt_inventory(
                project_root=args.project_root,
                run_matrix_path=args.run_matrix,
                eval_run_manifest_path=args.eval_run_manifest,
                eval_aggregate_path=args.eval_aggregate,
                output_path=args.output,
            )
            digest_field = "inventory_sha256"
        else:
            result = freeze_command_ledger(
                project_root=args.project_root,
                wan_original_run_manifest=args.wan_original_run_manifest,
                trained_wan_run_manifest=args.trained_wan_run_manifest,
                core_queue_plan=args.core_queue_plan,
                safree_queue_plan=args.safree_queue_plan,
                formal_launch_receipt=args.formal_launch_receipt,
                output_path=args.output,
            )
            digest_field = "ledger_sha256"
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError, ReproducibilityError) as exc:
        parser.exit(2, f"reproducibility freeze refused: {exc}\n")
    print(json.dumps({"status": result["status"], digest_field: result[digest_field]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
