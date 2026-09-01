#!/usr/bin/env python3
"""Freeze seven T2VUnlearning-adapted CogVideoX training specifications.

This is a pre-training, CPU-only registry builder.  It deterministically draws
the registered training-prompt count from each mechanism's already-frozen 178
rows and emits one explicit run spec per mechanism.  The trainer consumes the
recorded row count; no historical 36-row or 20-eval-row assertion remains in
the executable path.
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
from typing import Any, Mapping, Sequence

import build_causal_role_erasure_7mechanism_baseline_registry_v2 as baseline_contract


PROTOCOL = "causal_role_erasure_7mechanism_t2v_training_registry_v2"
PROTOCOL_VERSION = baseline_contract.PROTOCOL_VERSION
MECHANISMS = baseline_contract.MECHANISMS
DEFAULT_TRAIN_ROWS = 36
DEFAULT_MAX_STEPS = 100
TRAIN_FIELDS = (
    "protocol_version",
    "mechanism",
    "training_row_index",
    "candidate_id",
    "selected_index",
    "factual_prompt",
    "mechanism_concept",
    "selection_sha256",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TrainingRegistryError(ValueError):
    """A fail-closed adapted-baseline registration error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingRegistryError(message)


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


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return stream.getvalue().encode("utf-8")


def canonical_path(project_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_training_inputs(project_root: Path, registry_path: Path) -> dict[str, Any]:
    require(registry_path.is_file() and not registry_path.is_symlink(), f"training input registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    require(registry.get("protocol_version") == PROTOCOL_VERSION, "training input protocol changed")
    require(registry.get("counts", {}).get("selected_per_mechanism") == 178, "selected178 contract changed")
    require(set(registry.get("mechanisms", {})) == set(MECHANISMS), "training input mechanisms changed")
    for mechanism in MECHANISMS:
        ref = registry["mechanisms"][mechanism]["selected178"]
        path = Path(ref["path"])
        path = path if path.is_absolute() else project_root / path
        require(path.is_file() and not path.is_symlink(), f"selected178 missing for {mechanism}: {path}")
        require(ref.get("row_count") == 178, f"selected178 row count changed for {mechanism}")
        require(HEX64.fullmatch(str(ref.get("sha256", ""))) is not None, f"selected178 SHA invalid for {mechanism}")
        require(sha256_file(path) == ref["sha256"], f"selected178 changed for {mechanism}")
    return registry


def load_model_inventory(model_root: Path, inventory_path: Path) -> dict[str, Any]:
    require(inventory_path.is_file() and not inventory_path.is_symlink(), f"model inventory missing: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = inventory.get("files")
    require(isinstance(files, list) and files, "model inventory is empty")
    require(inventory.get("file_count") == len(files), "model inventory count mismatch")
    for item in files:
        path = model_root / item["path"]
        require(path.is_file() and not path.is_symlink(), f"model file missing: {path}")
        require(path.stat().st_size == item["size_bytes"], f"model size changed: {path}")
        require(sha256_file(path) == item["sha256"], f"model hash changed: {path}")
    return inventory


def selected_rows(
    project_root: Path,
    inputs: Mapping[str, Any],
    mechanism: str,
    count: int,
) -> tuple[Path, list[dict[str, str]]]:
    ref = inputs["mechanisms"][mechanism]["selected178"]
    path = Path(ref["path"])
    path = path if path.is_absolute() else project_root / path
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 178, f"selected178 CSV count changed for {mechanism}")
    require(len({row["candidate_id"] for row in rows}) == 178, f"selected178 IDs repeat for {mechanism}")
    ranked = []
    for row in rows:
        digest = hashlib.sha256(
            b"t2v-adapted-train-v2\0"
            + mechanism.encode("utf-8")
            + b"\0"
            + row["candidate_id"].encode("utf-8")
        ).hexdigest()
        ranked.append((digest, row["candidate_id"], row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    output = []
    for index, (digest, _candidate, row) in enumerate(ranked[:count]):
        require(row["mechanism"] == mechanism, f"selected row mechanism mismatch for {mechanism}")
        require(bool(row["factual_prompt"].strip()), f"empty factual prompt for {row['candidate_id']}")
        output.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "mechanism": mechanism,
                "training_row_index": index,
                "candidate_id": row["candidate_id"],
                "selected_index": row["selected_index"],
                "factual_prompt": row["factual_prompt"],
                "mechanism_concept": baseline_contract.MECHANISM_CONCEPTS[mechanism],
                "selection_sha256": digest,
            }
        )
    require(len(output) == count, f"could not select {count} T2V rows for {mechanism}")
    return path, output


def build_training_registry(
    *,
    project_root: Path,
    training_input_registry: Path,
    model_root: Path,
    model_inventory: Path,
    runtime_python: Path,
    output_root: Path,
    checkpoint_root: Path,
    train_rows_per_mechanism: int = DEFAULT_TRAIN_ROWS,
    max_steps: int = DEFAULT_MAX_STEPS,
    eligible_checkpoint_step: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    training_input_registry = training_input_registry.resolve()
    model_root = model_root.resolve()
    model_inventory = model_inventory.resolve()
    runtime_python = runtime_python.resolve()
    output_root = output_root.resolve(strict=False)
    checkpoint_root = checkpoint_root.resolve(strict=False)
    require(not output_root.exists(), f"output root already exists: {output_root}")
    require(1 <= train_rows_per_mechanism <= 178, "train rows per mechanism must be within 1..178")
    require(max_steps > 0, "max steps must be positive")
    require(eligible_checkpoint_step == max_steps, "only the precommitted final training step may be eligible")
    require(runtime_python.is_file(), f"runtime Python missing: {runtime_python}")
    inputs = load_training_inputs(project_root, training_input_registry)
    inventory = load_model_inventory(model_root, model_inventory)
    code_paths = (
        "scripts/build_causal_role_erasure_7mechanism_t2v_training_registry_v2.py",
        "scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py",
        "scripts/run_causal_role_erasure_7mechanism_t2v_training_queue_v2.sh",
    )
    code_sha256 = {}
    for relative in code_paths:
        path = project_root / relative
        require(path.is_file() and not path.is_symlink(), f"T2V code artifact missing: {path}")
        code_sha256[relative] = sha256_file(path)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        runs = []
        for mechanism_index, mechanism in enumerate(MECHANISMS):
            selected_path, rows = selected_rows(
                project_root, inputs, mechanism, train_rows_per_mechanism
            )
            final_rows_path = output_root / "training_rows" / f"{mechanism_index:02d}_{mechanism}.csv"
            stage_rows_path = stage / "training_rows" / f"{mechanism_index:02d}_{mechanism}.csv"
            stage_rows_path.parent.mkdir(parents=True, exist_ok=True)
            stage_rows_path.write_bytes(csv_bytes(rows, TRAIN_FIELDS))
            run_id = f"t2v7m_{mechanism}_adapted"
            output_dir = checkpoint_root / mechanism
            eligible_checkpoint = output_dir / f"checkpoint-{eligible_checkpoint_step:06d}"
            spec = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "protocol_version": PROTOCOL_VERSION,
                "run_id": run_id,
                "mechanism_index": mechanism_index,
                "mechanism": mechanism,
                "label": "T2VUnlearning-adapted (ours)",
                "training_rows": {
                    "path": canonical_path(project_root, final_rows_path),
                    "sha256": sha256_file(stage_rows_path),
                    "row_count": len(rows),
                    "source_selected178": canonical_path(project_root, selected_path),
                    "selection_rule": "ascending SHA256(t2v-adapted-train-v2 NUL mechanism NUL candidate_id)",
                },
                "model_root": canonical_path(project_root, model_root),
                "model_inventory": {
                    "path": canonical_path(project_root, model_inventory),
                    "sha256": sha256_file(model_inventory),
                    "file_count": inventory["file_count"],
                },
                "runtime_python": canonical_path(project_root, runtime_python),
                "runtime_python_sha256": sha256_file(runtime_python),
                "output_dir": canonical_path(project_root, output_dir),
                "eligible_checkpoint": canonical_path(project_root, eligible_checkpoint),
                "parameters": {
                    "rank": 128,
                    "learning_rate": "1e-4",
                    "max_steps": max_steps,
                    "negative_scale": 7.0,
                    "localization_weight": 1.0,
                    "preservation_weight": 0.0,
                    "num_frames": 49,
                    "height": 480,
                    "width": 720,
                    "dtype": "bf16",
                    "training_seed": 12000,
                    "save_every": eligible_checkpoint_step,
                    "eligible_checkpoint_step": eligible_checkpoint_step,
                },
                "checkpoint_selection": "precommitted final step only; no formal-output tuning",
            }
            spec_path = stage / "run_specs" / f"{mechanism_index:02d}_{mechanism}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_bytes(canonical_json_bytes(spec))
            runs.append(
                {
                    "run_id": run_id,
                    "mechanism": mechanism,
                    "run_spec": canonical_path(
                        project_root,
                        output_root / "run_specs" / f"{mechanism_index:02d}_{mechanism}.json",
                    ),
                    "run_spec_sha256": sha256_file(spec_path),
                    "eligible_checkpoint": canonical_path(project_root, eligible_checkpoint),
                }
            )
        registry = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "status": "t2v_training_specs_frozen_pre_training",
            "training_input_registry": canonical_path(project_root, training_input_registry),
            "training_input_registry_sha256": sha256_file(training_input_registry),
            "model_root": canonical_path(project_root, model_root),
            "model_inventory": canonical_path(project_root, model_inventory),
            "model_inventory_sha256": sha256_file(model_inventory),
            "runtime_python": canonical_path(project_root, runtime_python),
            "runtime_python_sha256": sha256_file(runtime_python),
            "code_sha256": code_sha256,
            "train_rows_per_mechanism": train_rows_per_mechanism,
            "runs": runs,
            "formal_output_inspected": False,
        }
        (stage / "t2v_training_registry.json").write_bytes(canonical_json_bytes(registry))
        receipt = {
            "protocol": PROTOCOL,
            "registry_sha256": sha256_file(stage / "t2v_training_registry.json"),
            "run_count": len(runs),
            "status": registry["status"],
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
    parser.add_argument("--training-input-registry", type=Path, default=Path("outputs/causal_role_erasure_7mechanism_main_v2/training_inputs_v2/training_input_registry.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/CogVideoX-2b"))
    parser.add_argument("--model-inventory", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, default=Path("models/.wan-runtime/bin/python"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_specs_v2"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("outputs/causal_role_erasure_7mechanism_main_v2/t2v_training_v2"))
    parser.add_argument("--train-rows-per-mechanism", type=int, default=DEFAULT_TRAIN_ROWS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eligible-checkpoint-step", type=int, default=DEFAULT_MAX_STEPS)
    return parser


def under(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    registry = build_training_registry(
        project_root=project_root,
        training_input_registry=under(project_root, args.training_input_registry),
        model_root=under(project_root, args.model_root),
        model_inventory=under(project_root, args.model_inventory),
        runtime_python=under(project_root, args.runtime_python),
        output_root=under(project_root, args.output_root),
        checkpoint_root=under(project_root, args.checkpoint_root),
        train_rows_per_mechanism=args.train_rows_per_mechanism,
        max_steps=args.max_steps,
        eligible_checkpoint_step=args.eligible_checkpoint_step,
    )
    print(json.dumps({"status": registry["status"], "runs": len(registry["runs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
