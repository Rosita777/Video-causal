#!/usr/bin/env python3
"""Freeze the seven-mechanism CogVideoX baseline implementation stack.

This command performs no model inference.  It inventories the frozen
CogVideoX model/runtime, snapshots the already-recovered official VideoEraser
pipeline, binds an optional official SAFREE checkout, and records the state of
the seven T2VUnlearning-adapted checkpoints.  An incomplete registry is useful
for exposing blockers, but formal generation is authorized only when every
required implementation and checkpoint is present and hash-bound.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import build_causal_role_erasure_7mechanism_main_v2 as main_data


PROTOCOL_VERSION = main_data.PROTOCOL_VERSION
REGISTRY_PROTOCOL = "causal_role_erasure_7mechanism_baseline_registry_v2"
MECHANISMS = main_data.MECHANISM_ORDER
FORMAL_CASES = 294
CASES_PER_MECHANISM = 42
EXPECTED_VIDEOERASER_COMMIT = "ba19cceb561dda916614e609759eb5c5b54f1c83"
EXPECTED_VIDEOERASER_PIPELINE_SHA256 = (
    "bd9e4052740eba1b37fbd31b879c4079f7139d2d23ff3a5811cab09b8321c0f4"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_RUNTIME_PACKAGES = ("torch", "diffusers", "transformers")
VIDEO = {"num_frames": 49, "fps": 8, "height": 480, "width": 720}
COG_GENERATION = {
    **VIDEO,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
    "scheduler": "CogVideoXDPMScheduler",
    "scheduler_timestep_spacing": "trailing",
    "use_dynamic_cfg": True,
}
STREAMS = (
    "cogvideox_original",
    "negative_prompt",
    "videoeraser_official",
    "t2vunlearning_adapted",
    "safree_cogvideox",
)
MECHANISM_CONCEPTS = {
    "water_impact": "Water impact",
    "rigid_collision": "Rigid collision",
    "brittle_fracture": "Brittle fracture",
    "powder_impact": "Powder impact",
    "elastic_deformation": "Elastic deformation",
    "material_release": "Material release",
    "surface_trace": "Surface trace",
}


class RegistryError(ValueError):
    """A fail-closed baseline registration error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_path(project_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def validate_formal_cases(path: Path) -> list[dict[str, str]]:
    require(path.is_file() and not path.is_symlink(), f"formal cases missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == FORMAL_CASES, f"formal case count must be {FORMAL_CASES}")
    require(len({row.get("case_id", "") for row in rows}) == FORMAL_CASES, "formal case IDs repeat")
    require({row.get("mechanism") for row in rows} == set(MECHANISMS), "formal mechanisms changed")
    for mechanism in MECHANISMS:
        selected = [row for row in rows if row["mechanism"] == mechanism]
        require(len(selected) == CASES_PER_MECHANISM, f"{mechanism} must have 42 cases")
        require(sum(row["case_kind"] == "causal" for row in selected) == 24, f"{mechanism} causal count changed")
        require(sum(row["case_kind"] == "specificity" for row in selected) == 18, f"{mechanism} specificity count changed")
        require({row["mechanism_name"] for row in selected} == {MECHANISM_CONCEPTS[mechanism]}, f"{mechanism} concept phrase changed")
    for row in rows:
        require(row.get("protocol_version") == PROTOCOL_VERSION, "formal protocol changed")
        require(int(row.get("seed", "0")) > 0, f"invalid frozen seed for {row.get('case_id')}")
    return rows


def inventory_tree(root: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"model root missing or symlinked: {root}")
    symlinks = sorted(path for path in root.rglob("*") if path.is_symlink())
    require(not symlinks, f"model inventory forbids symlinks: {symlinks[:3]}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    require(files, f"model root contains no regular files: {root}")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "files": rows,
    }


def probe_runtime(runtime_python: Path) -> dict[str, Any]:
    require(runtime_python.is_file() and not runtime_python.is_symlink(), f"runtime Python missing: {runtime_python}")
    code = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=('torch','diffusers','transformers');"
        "versions={n:m.version(n) for n in names};"
        "print(json.dumps({'python':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'executable':sys.executable,'packages':versions},sort_keys=True))"
    )
    result = subprocess.run(
        [str(runtime_python), "-I", "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    require(set(payload.get("packages", {})) == set(REQUIRED_RUNTIME_PACKAGES), "runtime package inventory incomplete")
    return payload


def observed_git_commit(root: Path) -> str | None:
    marker = root / "SAFREE_COMMIT"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip().casefold()
        return value if HEX40.fullmatch(value) else None
    git_entry = root / ".git"
    git_dir = git_entry
    if git_entry.is_file():
        line = git_entry.read_text(encoding="utf-8").strip()
        if line.startswith("gitdir:"):
            candidate = Path(line.split(":", 1)[1].strip())
            git_dir = candidate if candidate.is_absolute() else (root / candidate).resolve()
    head = git_dir / "HEAD"
    if head.is_file():
        value = head.read_text(encoding="utf-8").strip()
        if HEX40.fullmatch(value.casefold()):
            return value.casefold()
        if value.startswith("ref:"):
            ref = value.split(":", 1)[1].strip()
            loose_ref = git_dir / ref
            if loose_ref.is_file():
                commit = loose_ref.read_text(encoding="utf-8").strip().casefold()
                if HEX40.fullmatch(commit):
                    return commit
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for raw in packed.read_text(encoding="utf-8").splitlines():
                    fields = raw.split()
                    if len(fields) == 2 and fields[1] == ref and HEX40.fullmatch(fields[0].casefold()):
                        return fields[0].casefold()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().casefold()
    return value if HEX40.fullmatch(value) else None


def inspect_safree(
    project_root: Path,
    root: Path,
    expected_commit: str | None,
    expected_pipeline_sha256: str | None,
) -> dict[str, Any]:
    pipeline = root / "cogvideox" / "cogvideox_pipeline.py"
    if not pipeline.is_file() or pipeline.is_symlink():
        return {
            "status": "blocked_missing_external_clone",
            "repository": "https://github.com/jaehong31/SAFREE",
            "expected_pipeline": canonical_path(project_root, pipeline),
        }
    commit = observed_git_commit(root)
    require(commit is not None, "SAFREE checkout has no exact 40-hex commit binding")
    digest = sha256_file(pipeline)
    if expected_commit is not None:
        require(HEX40.fullmatch(expected_commit) is not None, "expected SAFREE commit is invalid")
        require(commit == expected_commit, "SAFREE commit does not match expected commit")
    if expected_pipeline_sha256 is not None:
        require(HEX64.fullmatch(expected_pipeline_sha256) is not None, "expected SAFREE SHA-256 is invalid")
        require(digest == expected_pipeline_sha256, "SAFREE pipeline SHA-256 mismatch")
    return {
        "status": "ready",
        "repository": "https://github.com/jaehong31/SAFREE",
        "commit": commit,
        "pipeline_path": canonical_path(project_root, pipeline),
        "pipeline_sha256": digest,
        "concept_injection": "mechanism_name_as_single_CONCEPT_DICT_entry",
        "dtype": "fp32",
    }


def inspect_t2v(
    project_root: Path,
    registry_path: Path | None,
    *,
    expected_model_root: Path,
    expected_model_inventory_sha256: str,
    expected_runtime_python: Path,
    expected_runtime_python_sha256: str,
) -> dict[str, Any]:
    if registry_path is None or not registry_path.is_file():
        return {
            "status": "blocked_missing_training_registry",
            "label": "T2VUnlearning-adapted (ours)",
            "reason": "public matching training code and mechanism checkpoints are unavailable",
        }
    raw = registry_path.read_bytes()
    registry = json.loads(raw)
    require(registry.get("protocol") == "causal_role_erasure_7mechanism_t2v_training_registry_v2", "unexpected T2V registry protocol")
    require(registry.get("protocol_version") == PROTOCOL_VERSION, "T2V registry protocol version changed")
    require(registry.get("status") == "t2v_training_specs_frozen_pre_training", "T2V training registry status changed")
    require(registry.get("formal_output_inspected") is False, "T2V training registry was contaminated by formal outputs")
    registered_model_root = Path(str(registry.get("model_root", "")))
    registered_model_root = registered_model_root if registered_model_root.is_absolute() else project_root / registered_model_root
    require(registered_model_root.resolve() == expected_model_root.resolve(), "T2V training base model differs from formal base model")
    inventory_path = Path(str(registry.get("model_inventory", "")))
    inventory_path = inventory_path if inventory_path.is_absolute() else project_root / inventory_path
    require(inventory_path.is_file() and not inventory_path.is_symlink(), "T2V model inventory is missing")
    require(sha256_file(inventory_path) == registry.get("model_inventory_sha256"), "T2V model inventory changed")
    require(
        registry.get("model_inventory_sha256") == expected_model_inventory_sha256,
        "T2V training model inventory differs from formal base model inventory",
    )
    registered_runtime = Path(str(registry.get("runtime_python", "")))
    registered_runtime = registered_runtime if registered_runtime.is_absolute() else project_root / registered_runtime
    require(registered_runtime.resolve() == expected_runtime_python.resolve(), "T2V training runtime differs from formal runtime")
    require(sha256_file(registered_runtime) == registry.get("runtime_python_sha256"), "T2V training runtime changed")
    require(
        registry.get("runtime_python_sha256") == expected_runtime_python_sha256,
        "T2V training runtime hash differs from formal runtime",
    )
    for relative, digest in registry.get("code_sha256", {}).items():
        path = project_root / str(relative)
        require(path.is_file() and not path.is_symlink(), f"T2V registered code artifact missing: {path}")
        require(sha256_file(path) == digest, f"T2V registered code artifact changed: {relative}")
    runs = registry.get("runs")
    require(isinstance(runs, list) and len(runs) == len(MECHANISMS), "T2V registry must contain seven runs")
    require([run.get("mechanism") for run in runs] == list(MECHANISMS), "T2V run order changed")
    bound = []
    ready = True
    for run in runs:
        spec_path = Path(str(run.get("run_spec", "")))
        spec_path = spec_path if spec_path.is_absolute() else project_root / spec_path
        require(spec_path.is_file() and not spec_path.is_symlink(), f"T2V run spec missing: {spec_path}")
        require(sha256_file(spec_path) == run.get("run_spec_sha256"), f"T2V run spec changed: {spec_path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        require(spec.get("protocol") == registry["protocol"], f"T2V run-spec protocol changed: {spec_path}")
        require(spec.get("protocol_version") == PROTOCOL_VERSION, f"T2V run-spec version changed: {spec_path}")
        require(spec.get("run_id") == run.get("run_id"), f"T2V run ID mismatch: {spec_path}")
        require(spec.get("mechanism") == run.get("mechanism"), f"T2V run-spec mechanism mismatch: {spec_path}")
        require(spec.get("eligible_checkpoint") == run.get("eligible_checkpoint"), f"T2V eligible checkpoint mismatch: {spec_path}")
        spec_model_root = Path(str(spec.get("model_root", "")))
        spec_model_root = spec_model_root if spec_model_root.is_absolute() else project_root / spec_model_root
        require(spec_model_root.resolve() == expected_model_root.resolve(), f"T2V run-spec base model mismatch: {spec_path}")
        spec_inventory = spec.get("model_inventory", {})
        require(
            spec_inventory.get("sha256") == expected_model_inventory_sha256,
            f"T2V run-spec model inventory mismatch: {spec_path}",
        )
        spec_runtime = Path(str(spec.get("runtime_python", "")))
        spec_runtime = spec_runtime if spec_runtime.is_absolute() else project_root / spec_runtime
        require(spec_runtime.resolve() == expected_runtime_python.resolve(), f"T2V run-spec runtime mismatch: {spec_path}")
        require(
            spec.get("runtime_python_sha256") == expected_runtime_python_sha256,
            f"T2V run-spec runtime hash mismatch: {spec_path}",
        )
        training_rows = Path(str(spec.get("training_rows", {}).get("path", "")))
        training_rows = training_rows if training_rows.is_absolute() else project_root / training_rows
        require(training_rows.is_file() and not training_rows.is_symlink(), f"T2V training rows missing: {training_rows}")
        require(
            sha256_file(training_rows) == spec.get("training_rows", {}).get("sha256"),
            f"T2V training rows changed: {training_rows}",
        )
        checkpoint = Path(str(run["eligible_checkpoint"]))
        checkpoint = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
        weights = checkpoint / "eraser_weights.pt"
        config = checkpoint / "eraser_config.json"
        state = checkpoint / "training_state.json"
        receipt = checkpoint / "training_receipt.json"
        present = all(path.is_file() and not path.is_symlink() for path in (weights, config, state, receipt))
        ready = ready and present
        item = {
            "mechanism": run["mechanism"],
            "eligible_checkpoint": canonical_path(project_root, checkpoint),
            "status": "ready" if present else "blocked_untrained",
        }
        if present:
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            weights_sha256 = sha256_file(weights)
            config_sha256 = sha256_file(config)
            state_sha256 = sha256_file(state)
            require(receipt_payload.get("status") == "eligible", f"T2V checkpoint is not eligible: {checkpoint}")
            require(receipt_payload.get("protocol") == registry["protocol"], f"T2V receipt protocol mismatch: {checkpoint}")
            require(receipt_payload.get("protocol_version") == PROTOCOL_VERSION, f"T2V receipt version mismatch: {checkpoint}")
            require(receipt_payload.get("run_id") == run.get("run_id"), f"T2V receipt run ID mismatch: {checkpoint}")
            require(receipt_payload.get("mechanism") == run["mechanism"], f"T2V receipt mechanism mismatch: {checkpoint}")
            require(
                receipt_payload.get("step") == spec.get("parameters", {}).get("eligible_checkpoint_step"),
                f"T2V receipt step mismatch: {checkpoint}",
            )
            require(receipt_payload.get("run_spec_sha256") == run["run_spec_sha256"], f"T2V receipt run-spec mismatch: {checkpoint}")
            require(
                receipt_payload.get("training_rows_sha256") == spec.get("training_rows", {}).get("sha256"),
                f"T2V receipt training rows mismatch: {checkpoint}",
            )
            require(receipt_payload.get("weights_sha256") == weights_sha256, f"T2V weights receipt mismatch: {checkpoint}")
            require(receipt_payload.get("config_sha256") == config_sha256, f"T2V config receipt mismatch: {checkpoint}")
            require(receipt_payload.get("training_state_sha256") == state_sha256, f"T2V state receipt mismatch: {checkpoint}")
            require(receipt_payload.get("formal_output_inspected") is False, f"T2V checkpoint used formal outputs: {checkpoint}")
            item.update(
                {
                    "weights_sha256": weights_sha256,
                    "config_sha256": config_sha256,
                    "training_state_sha256": state_sha256,
                    "receipt_sha256": sha256_file(receipt),
                }
            )
        bound.append(item)
    return {
        "status": "ready" if ready else "blocked_untrained",
        "label": "T2VUnlearning-adapted (ours)",
        "training_registry": canonical_path(project_root, registry_path),
        "training_registry_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoints": bound,
    }


def snapshot_videoeraser(
    source_root: Path,
    stage_snapshot_root: Path,
    final_snapshot_root: Path,
) -> dict[str, Any]:
    source_pipeline = source_root / "CogVideoX" / "cogvideox_pipeline.py"
    source_commit = source_root / "OFFICIAL_COMMIT"
    require(source_pipeline.is_file() and not source_pipeline.is_symlink(), f"official VideoEraser pipeline missing: {source_pipeline}")
    require(source_commit.is_file() and not source_commit.is_symlink(), f"VideoEraser commit marker missing: {source_commit}")
    commit = source_commit.read_text(encoding="utf-8").strip().casefold()
    require(commit == EXPECTED_VIDEOERASER_COMMIT, "official VideoEraser commit changed")
    digest = sha256_file(source_pipeline)
    require(digest == EXPECTED_VIDEOERASER_PIPELINE_SHA256, "official VideoEraser pipeline SHA-256 changed")
    target_pipeline = stage_snapshot_root / "CogVideoX" / "cogvideox_pipeline.py"
    target_pipeline.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_pipeline, target_pipeline)
    shutil.copy2(source_commit, stage_snapshot_root / "OFFICIAL_COMMIT")
    require(sha256_file(target_pipeline) == digest, "VideoEraser snapshot copy changed bytes")
    return {
        "status": "ready",
        "label": "VideoEraser (official CogVideoX)",
        "repository": "https://github.com/bluedream02/VideoEraser",
        "commit": commit,
        "pipeline_path": (final_snapshot_root / "CogVideoX" / "cogvideox_pipeline.py").as_posix(),
        "pipeline_sha256": digest,
        "official_pipeline_unmodified": True,
        "dtype": "bf16",
    }


def build_registry(
    *,
    project_root: Path,
    formal_cases: Path,
    model_root: Path,
    runtime_python: Path,
    videoeraser_source_root: Path,
    safree_root: Path,
    t2v_registry: Path | None,
    output_root: Path,
    safree_expected_commit: str | None = None,
    safree_expected_pipeline_sha256: str | None = None,
    require_complete: bool = False,
    runtime_probe: Callable[[Path], Mapping[str, Any]] = probe_runtime,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    formal_cases = formal_cases.resolve()
    model_root = model_root.resolve()
    runtime_python = runtime_python.resolve()
    videoeraser_source_root = videoeraser_source_root.resolve()
    safree_root = safree_root.resolve(strict=False)
    t2v_registry = t2v_registry.resolve() if t2v_registry is not None and t2v_registry.exists() else t2v_registry
    output_root = output_root.resolve(strict=False)
    require(not output_root.exists(), f"output root already exists: {output_root}")
    validate_formal_cases(formal_cases)
    model_inventory = inventory_tree(model_root)
    runtime = dict(runtime_probe(runtime_python))
    for package in REQUIRED_RUNTIME_PACKAGES:
        require(bool(runtime.get("packages", {}).get(package)), f"runtime missing {package}")
    safree = inspect_safree(
        project_root,
        safree_root,
        safree_expected_commit.casefold() if safree_expected_commit else None,
        safree_expected_pipeline_sha256.casefold() if safree_expected_pipeline_sha256 else None,
    )
    t2v = inspect_t2v(
        project_root,
        t2v_registry,
        expected_model_root=model_root,
        expected_model_inventory_sha256=hashlib.sha256(canonical_json_bytes(model_inventory)).hexdigest(),
        expected_runtime_python=runtime_python,
        expected_runtime_python_sha256=sha256_file(runtime_python),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    final_snapshot = output_root / "external_snapshot" / "VideoEraser"
    try:
        videoeraser = snapshot_videoeraser(
            videoeraser_source_root,
            stage / "external_snapshot" / "VideoEraser",
            Path(canonical_path(project_root, final_snapshot)),
        )
        (stage / "model_inventory.json").write_bytes(canonical_json_bytes(model_inventory))
        code_paths = (
            "scripts/build_causal_role_erasure_7mechanism_baseline_registry_v2.py",
            "scripts/run_causal_role_erasure_7mechanism_cogvideox_controls_v2.py",
            "scripts/run_causal_role_erasure_7mechanism_videoeraser_official_v2.py",
            "scripts/run_causal_role_erasure_7mechanism_safree_cogvideox_v2.py",
            "scripts/run_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
            "scripts/run_causal_role_erasure_7mechanism_baseline_queue_v2.py",
            "scripts/build_causal_role_erasure_7mechanism_t2v_training_registry_v2.py",
            "scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
        )
        code = {}
        for relative in code_paths:
            path = project_root / relative
            require(path.is_file() and not path.is_symlink(), f"baseline code artifact missing: {path}")
            code[relative] = sha256_file(path)
        complete = safree["status"] == "ready" and t2v["status"] == "ready"
        registry = {
            "schema_version": 1,
            "protocol": REGISTRY_PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": "baseline_stack_frozen_ready" if complete else "baseline_stack_incomplete",
            "formal_generation_authorized": complete,
            "formal_cases": {
                "path": canonical_path(project_root, formal_cases),
                "sha256": sha256_file(formal_cases),
                "row_count": FORMAL_CASES,
                "same_case_same_seed_across_streams": True,
            },
            "model": {
                "family": "CogVideoX-2B",
                "root": canonical_path(project_root, model_root),
                "inventory_path": canonical_path(project_root, output_root / "model_inventory.json"),
                "inventory_sha256": hashlib.sha256(canonical_json_bytes(model_inventory)).hexdigest(),
                "file_count": model_inventory["file_count"],
                "total_size_bytes": model_inventory["total_size_bytes"],
            },
            "runtime": {
                **runtime,
                "python_executable": canonical_path(project_root, runtime_python),
                "python_executable_sha256": sha256_file(runtime_python),
            },
            "mechanism_concepts": MECHANISM_CONCEPTS,
            "generation": {
                "shared": COG_GENERATION,
                "streams": {
                    "cogvideox_original": {"dtype": "bf16", "method": "unmodified_positive_prompt"},
                    "negative_prompt": {"dtype": "bf16", "method": "mechanism_concept_as_negative_prompt"},
                    "videoeraser_official": {"dtype": "bf16", "method": "official_SPEA_ARNG_pipeline"},
                    "t2vunlearning_adapted": {"dtype": "bf16", "method": "paper_guided_internal_adaptation"},
                    "safree_cogvideox": {"dtype": "fp32", "method": "official_pipeline_with_mechanism_concept_injection"},
                },
            },
            "implementations": {
                "videoeraser_official": videoeraser,
                "safree_cogvideox": safree,
                "t2vunlearning_adapted": t2v,
                "negative_prompt": {"status": "ready", "label": "Negative Prompt"},
                "cogvideox_original": {"status": "ready", "label": "CogVideoX Original"},
            },
            "streams": list(STREAMS),
            "expected_cogvideox_outputs": FORMAL_CASES * len(STREAMS),
            "code_sha256": code,
            "blocked_reasons": [
                name
                for name, item in (("safree_cogvideox", safree), ("t2vunlearning_adapted", t2v))
                if item["status"] != "ready"
            ],
        }
        if require_complete:
            require(complete, "baseline stack is incomplete: " + ", ".join(registry["blocked_reasons"]))
        (stage / "baseline_registry.json").write_bytes(canonical_json_bytes(registry))
        receipt = {
            "protocol": REGISTRY_PROTOCOL,
            "status": registry["status"],
            "registry_sha256": sha256_file(stage / "baseline_registry.json"),
            "model_inventory_sha256": sha256_file(stage / "model_inventory.json"),
        }
        (stage / "build_receipt.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(stage, output_root)
        return registry
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--formal-cases", type=Path, default=Path("data/causal_role_erasure_7mechanism_main_v2/formal_cases.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("models/CogVideoX-2b"))
    parser.add_argument("--runtime-python", type=Path, default=Path("models/.wan-runtime/bin/python"))
    parser.add_argument("--videoeraser-source-root", type=Path, default=Path("/data/xiaohuang_workspace/ljc/Video-causal/baselines/external/VideoEraser"))
    parser.add_argument("--safree-root", type=Path, default=Path("baselines/external/SAFREE"))
    parser.add_argument("--safree-expected-commit")
    parser.add_argument("--safree-expected-pipeline-sha256")
    parser.add_argument("--t2v-registry", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/causal_role_erasure_7mechanism_main_v2/baseline_registry_v2"))
    parser.add_argument("--require-complete", action="store_true")
    return parser


def resolve_under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    registry = build_registry(
        project_root=project_root,
        formal_cases=resolve_under(project_root, args.formal_cases),
        model_root=resolve_under(project_root, args.model_root),
        runtime_python=resolve_under(project_root, args.runtime_python),
        videoeraser_source_root=resolve_under(project_root, args.videoeraser_source_root),
        safree_root=resolve_under(project_root, args.safree_root),
        t2v_registry=resolve_under(project_root, args.t2v_registry) if args.t2v_registry else None,
        output_root=resolve_under(project_root, args.output_root),
        safree_expected_commit=args.safree_expected_commit,
        safree_expected_pipeline_sha256=args.safree_expected_pipeline_sha256,
        require_complete=args.require_complete,
    )
    print(json.dumps({"status": registry["status"], "formal_generation_authorized": registry["formal_generation_authorized"], "blocked_reasons": registry["blocked_reasons"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
