#!/usr/bin/env python3
"""Prepare registry-bound base, teacher, or arm caches for one 7m mechanism.

The module intentionally imports no ML package at module import time.  Dry-run
validates and freezes a plan without loading Wan, torch, CUDA, or video media.
Each real mode reserves a fresh output directory and writes an exact ordered
PT inventory plus a canonical JSON manifest; no cache can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
CACHE_PROTOCOL = "causal_role_erasure_7mechanism_training_cache_v2"
REGISTRY_PROTOCOL = "causal_role_erasure_7mechanism_training_cache_registry_v2"
MODES = ("prepare-base", "prepare-teacher", "prepare-arm")
ARMS = ("matched", "v4", "generic_paraphrase", "bystander")
MECHANISMS = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
ERASE_ROWS = 178
PRESERVE_ROWS = 36
BASE_ROWS = ERASE_ROWS + PRESERVE_ROWS
PROMPT_SHAPE = (1, 226, 4096)
LATENT_SHAPE = (1, 16, 13, 60, 104)
TENSOR_DTYPE = "torch.bfloat16"
NUM_FRAMES = 49
FPS = 8
HEIGHT = 480
WIDTH = 832
MAX_SEQUENCE_LENGTH = 226
PROMPT_BATCH_SIZE = 16
PROMPT_DEVICE_MAP = "balanced"
PROMPT_SHARD_MAX_MEMORY_GIB = 18
PROMPT_MIN_CUDA_DEVICES = 4
INVENTORY_ALGORITHM = "sha256_ordered_filename_nul_file_bytes_newline_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
SEALED = ("final36", "sealed-final", "sealed_final")

SELECTED_FIELDS = {
    "protocol_version", "mechanism", "selected_index", "candidate_id",
    "global_index", "source_id", "source_object", "receiver_id", "receiver",
    "prompt_style", "factual_prompt", "target_prompt", "seed",
    "target_video_path", "target_video_sha256",
}
PRESERVE_FIELDS = {
    "protocol_version", "preserve_index", "preserve_id", "prompt",
    "target_video_path", "target_video_sha256",
}
ARM_FIELDS = {
    "protocol_version", "mechanism", "arm", "selected_index", "candidate_id",
    "student_prompt", "assigned_source_id", "assigned_source_object",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fixed_prompt_batch(prompts: Sequence[str]) -> list[str]:
    require(1 <= len(prompts) <= PROMPT_BATCH_SIZE, "prompt batch size is invalid")
    values = list(prompts)
    return values + [values[-1]] * (PROMPT_BATCH_SIZE - len(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")


def reject_sealed(*values: str | Path) -> None:
    for value in values:
        require(
            not any(token in str(value).casefold() for token in SEALED),
            f"sealed/final36 path is forbidden: {value}",
        )


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def resolve_path(project_root: Path, value: str | Path, *, strict: bool = True) -> Path:
    reject_sealed(value)
    path = Path(value)
    path = path if path.is_absolute() else project_root / path
    result = path.resolve(strict=strict)
    reject_sealed(result)
    return result


def artifact_record(
    project_root: Path,
    value: Any,
    label: str,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    require(isinstance(value, dict), f"registry {label} must be an object")
    require(set(value) == {"path", "sha256", "row_count"}, f"registry {label} fields are not exact")
    require_sha256(value["sha256"], f"registry {label}.sha256")
    require(type(value["row_count"]) is int and value["row_count"] >= 0, f"registry {label}.row_count invalid")
    if expected_rows is not None:
        require(value["row_count"] == expected_rows, f"registry {label}.row_count must be {expected_rows}")
    path = resolve_path(project_root, str(value["path"]))
    regular_file(path, f"registry {label}")
    require(sha256_file(path) == value["sha256"], f"registry {label} byte hash mismatch")
    return {"path": str(path), "sha256": value["sha256"], "row_count": value["row_count"]}


def load_registry(project_root: Path, path: Path, expected_sha256: str) -> dict[str, Any]:
    require_sha256(expected_sha256, "external cache registry hash")
    regular_file(path, "cache registry")
    require(sha256_file(path) == expected_sha256, "cache registry differs from external expected SHA-256")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "cache registry root must be an object")
    required = {
        "schema_version", "protocol", "protocol_version", "status", "mechanism",
        "selected178", "preserve36", "model_root", "model_inventory",
        "runtime_root", "runtime_registry", "python_executable", "arms",
    }
    require(set(payload) == required, "cache registry fields are not exact")
    require(payload["schema_version"] == 1, "cache registry schema version mismatch")
    require(payload["protocol"] == REGISTRY_PROTOCOL, "cache registry protocol mismatch")
    require(payload["protocol_version"] == PROTOCOL_VERSION, "cache registry protocol_version mismatch")
    require(payload["status"] == "frozen", "cache registry is not frozen")
    mechanism = payload["mechanism"]
    require(mechanism in MECHANISMS, "cache registry mechanism is invalid")
    selected = artifact_record(project_root, payload["selected178"], "selected178", expected_rows=ERASE_ROWS)
    preserve = artifact_record(project_root, payload["preserve36"], "preserve36", expected_rows=PRESERVE_ROWS)
    model_inventory = artifact_record(project_root, payload["model_inventory"], "model_inventory")
    runtime_registry = artifact_record(project_root, payload["runtime_registry"], "runtime_registry")
    model_root = resolve_path(project_root, payload["model_root"], strict=False)
    runtime_root = resolve_path(project_root, payload["runtime_root"], strict=False)
    python_executable = resolve_path(project_root, payload["python_executable"], strict=False)
    arms = payload["arms"]
    require(isinstance(arms, dict) and set(arms).issubset(ARMS), "registry arms contain an unknown key")
    arm_records: dict[str, dict[str, Any] | None] = {}
    for arm, value in arms.items():
        arm_records[arm] = (
            None
            if value is None
            else artifact_record(project_root, value, f"arms.{arm}", expected_rows=ERASE_ROWS)
        )
    return {
        **payload,
        "project_root": str(project_root),
        "registry_path": str(path),
        "registry_sha256": expected_sha256,
        "selected178": selected,
        "preserve36": preserve,
        "model_root": str(model_root),
        "model_inventory": model_inventory,
        "runtime_root": str(runtime_root),
        "runtime_registry": runtime_registry,
        "python_executable": str(python_executable),
        "arms": arm_records,
    }


def read_csv_exact(record: Mapping[str, Any], required: set[str], label: str) -> list[dict[str, str]]:
    path = Path(str(record["path"]))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"{label} has no header")
        require(required.issubset(reader.fieldnames), f"{label} missing canonical columns: {sorted(required - set(reader.fieldnames))}")
        rows = list(reader)
    require(len(rows) == record["row_count"], f"{label} row count mismatch")
    return rows


def _integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    return result


def load_selected(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv_exact(registry["selected178"], SELECTED_FIELDS, "selected178")
    mechanism = str(registry["mechanism"])
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        require(raw["protocol_version"] == PROTOCOL_VERSION, f"selected row {index}: protocol mismatch")
        require(raw["mechanism"] == mechanism, f"selected row {index}: mechanism mismatch")
        require(_integer(raw["selected_index"], f"selected row {index} index") == index, "selected_index must be 0..177 in file order")
        require(SAFE_ID.fullmatch(raw["candidate_id"]) is not None, f"selected row {index}: unsafe candidate_id")
        require(raw["factual_prompt"].strip() and raw["target_prompt"].strip(), f"selected row {index}: empty prompt")
        require_sha256(raw["target_video_sha256"], f"selected row {index} target video")
        path = resolve_path(Path(str(registry["project_root"])), raw["target_video_path"], strict=False)
        result.append({**raw, "selected_index": index, "global_index": _integer(raw["global_index"], "global_index"), "seed": _integer(raw["seed"], "seed"), "target_video_path": str(path)})
    require(len({row["candidate_id"] for row in result}) == ERASE_ROWS, "selected candidate_id values are not unique")
    require(len({row["global_index"] for row in result}) == ERASE_ROWS, "selected global_index values are not unique")
    return result


def load_preserve(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv_exact(registry["preserve36"], PRESERVE_FIELDS, "preserve36")
    result: list[dict[str, Any]] = []
    project_root = Path(str(registry["project_root"]))
    for index, raw in enumerate(rows):
        require(raw["protocol_version"] == PROTOCOL_VERSION, f"preserve row {index}: protocol mismatch")
        require(_integer(raw["preserve_index"], f"preserve row {index} index") == index, "preserve_index must be 0..35")
        require(SAFE_ID.fullmatch(raw["preserve_id"]) is not None, f"preserve row {index}: unsafe preserve_id")
        require(raw["prompt"].strip(), f"preserve row {index}: empty prompt")
        require_sha256(raw["target_video_sha256"], f"preserve row {index} target video")
        path = resolve_path(project_root, raw["target_video_path"], strict=False)
        result.append({**raw, "preserve_index": index, "target_video_path": str(path)})
    require(len({row["preserve_id"] for row in result}) == PRESERVE_ROWS, "preserve_id values are not unique")
    return result


def load_arm_mapping(registry: Mapping[str, Any], arm: str, selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    record = registry["arms"].get(arm)
    if record is None:
        require(arm == "matched", f"registry has no mapping for arm {arm}")
        return [
            {
                "protocol_version": PROTOCOL_VERSION,
                "mechanism": registry["mechanism"],
                "arm": arm,
                "selected_index": index,
                "candidate_id": row["candidate_id"],
                "student_prompt": row["factual_prompt"],
                "assigned_source_id": "",
                "assigned_source_object": "",
            }
            for index, row in enumerate(selected)
        ]
    rows = read_csv_exact(record, ARM_FIELDS, f"{arm} mapping")
    result: list[dict[str, Any]] = []
    require(len(rows) == len(selected), f"{arm} mapping/selected row count mismatch")
    for index, (raw, selected_row) in enumerate(zip(rows, selected)):
        require(raw["protocol_version"] == PROTOCOL_VERSION, f"arm row {index}: protocol mismatch")
        require(raw["mechanism"] == registry["mechanism"], f"arm row {index}: mechanism mismatch")
        require(raw["arm"] == arm, f"arm row {index}: arm mismatch")
        require(_integer(raw["selected_index"], f"arm row {index} index") == index, "arm selected_index mismatch")
        require(raw["candidate_id"] == selected_row["candidate_id"], f"arm row {index}: candidate binding mismatch")
        require(raw["student_prompt"].strip(), f"arm row {index}: empty student prompt")
        if arm == "matched":
            require(raw["student_prompt"] == selected_row["factual_prompt"], f"matched row {index}: student prompt differs from factual prompt")
        if arm == "v4":
            require(raw["assigned_source_id"].strip() and raw["assigned_source_object"].strip(), f"v4 row {index}: assigned source is missing")
            require(raw["assigned_source_id"] != selected_row["source_id"], f"v4 row {index}: assigned source equals original source")
        result.append({**raw, "selected_index": index})
    return result


def validate_model_inventory(registry: Mapping[str, Any], *, live: bool) -> dict[str, Any]:
    path = Path(str(registry["model_inventory"]["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "model inventory must be an object")
    files = payload.get("files")
    require(isinstance(files, list) and len(files) == registry["model_inventory"]["row_count"], "model inventory file count mismatch")
    root = Path(str(registry["model_root"]))
    registered_root = payload.get("model_root", payload.get("root"))
    if isinstance(registered_root, str):
        require(
            resolve_path(Path(str(registry["project_root"])), registered_root, strict=False)
            == root.resolve(),
            "model inventory root differs from cache registry",
        )
    if live:
        require(root.is_dir() and not root.is_symlink(), "model root is missing or symlinked")
        project_root = Path(str(registry["project_root"])).resolve(strict=True)
        root = root.resolve(strict=True)
        try:
            registered_root_relative = root.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("model root escaped project root") from exc
        expected_paths: list[Path] = []
        for index, item in enumerate(files):
            require(isinstance(item, dict), f"model inventory row {index} invalid")
            relative = item.get("path")
            require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), f"model inventory row {index} path invalid")
            relative_path = Path(relative)
            # The frozen v3 inventory records canonical project-relative
            # paths (including the model-root prefix).  Unit-sized/legacy
            # registries used by this tool recorded model-root-relative
            # paths.  Resolve both representations to the same live file;
            # never concatenate the model root twice.
            if relative_path.parts[: len(registered_root_relative.parts)] == registered_root_relative.parts:
                target = project_root / relative_path
            else:
                target = root / relative_path
            resolved_target = target.resolve(strict=True)
            require(
                resolved_target == root or root in resolved_target.parents,
                f"model inventory row {index} escaped model root",
            )
            require(resolved_target not in expected_paths, f"model inventory row {index} path repeats")
            expected_paths.append(resolved_target)
            expected = require_sha256(item.get("sha256"), f"model inventory row {index}")
            regular_file(resolved_target, f"model inventory row {index}")
            if type(item.get("size_bytes", item.get("size"))) is int:
                require(resolved_target.stat().st_size == item.get("size_bytes", item.get("size")), f"model inventory row {index} size mismatch")
            require(sha256_file(resolved_target) == expected, f"model inventory row {index} byte mismatch")
        actual_paths: list[Path] = []
        for candidate in sorted(root.rglob("*")):
            info = os.lstat(candidate)
            require(not stat.S_ISLNK(info.st_mode), "live model tree contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                continue
            require(
                stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                "live model tree contains a non-regular file or hardlink",
            )
            actual_paths.append(candidate.resolve(strict=True))
        require(
            expected_paths == sorted(set(expected_paths)),
            "model inventory paths are not unique and sorted by live path",
        )
        require(actual_paths == expected_paths, "live model file inventory has missing or extra files")
    return payload


def validate_runtime(registry: Mapping[str, Any], *, live: bool) -> dict[str, Any]:
    path = Path(str(registry["runtime_registry"]["path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "runtime registry must be an object")
    project_root = Path(str(registry["project_root"]))
    if isinstance(payload.get("runtime_root"), str):
        require(resolve_path(project_root, payload["runtime_root"], strict=False) == Path(str(registry["runtime_root"])), "runtime root differs from registry")
    if isinstance(payload.get("python_executable"), str):
        require(resolve_path(project_root, payload["python_executable"], strict=False) == Path(str(registry["python_executable"])), "runtime Python differs from registry")
    if not live:
        return payload
    runtime_root = Path(str(registry["runtime_root"]))
    python = Path(str(registry["python_executable"]))
    require(runtime_root.is_dir() and not runtime_root.is_symlink(), "runtime root missing or symlinked")
    regular_file(python, "registered Python executable")
    require(os.access(python, os.X_OK), "registered Python is not executable")
    require(Path(sys.executable).resolve() == python.resolve(), "current interpreter differs from registry")
    expected_python = payload.get("python", {}).get("version")
    if expected_python is not None:
        require(platform.python_version() == expected_python, "Python version differs from runtime registry")
    packages = payload.get("packages")
    if isinstance(packages, dict):
        actual = {name: importlib.metadata.version(name) for name in packages}
        require(actual == packages, "package versions differ from runtime registry")
    import torch
    require(torch.cuda.is_available(), "registered cache preparation requires CUDA")
    expected_torch = payload.get("torch", {}).get("module_version")
    if expected_torch is not None:
        require(torch.__version__ == expected_torch, "torch module version differs from registry")
    return payload


def input_binding_sha256(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for field in fields:
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def build_plan(
    registry: Mapping[str, Any], mode: str, output_dir: Path, *, arm: str | None,
    base_cache_dir: Path | None, dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
    require(mode in MODES, "invalid cache mode")
    selected = load_selected(registry)
    preserve = load_preserve(registry)
    mapping = None
    if mode == "prepare-arm":
        require(arm in ARMS, "prepare-arm requires a valid --arm")
        mapping = load_arm_mapping(registry, str(arm), selected)
        if arm == "matched":
            require(base_cache_dir is not None, "matched arm requires --base-cache-dir")
    else:
        require(arm is None and base_cache_dir is None, "--arm/--base-cache-dir are valid only for prepare-arm")
    row_count = BASE_ROWS if mode == "prepare-base" else ERASE_ROWS
    plan = {
        "schema_version": 1,
        "protocol": CACHE_PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "status": "planned" if dry_run else "reserved_for_run",
        "dry_run": dry_run,
        "mode": mode,
        "arm": arm,
        "mechanism": registry["mechanism"],
        "output_dir": str(output_dir),
        "registry": {"path": registry["registry_path"], "sha256": registry["registry_sha256"]},
        "selected178": registry["selected178"],
        "preserve36": registry["preserve36"],
        "model_root": registry["model_root"],
        "model_inventory": registry["model_inventory"],
        "runtime_root": registry["runtime_root"],
        "runtime_registry": registry["runtime_registry"],
        "python_executable": registry["python_executable"],
        "base_cache_dir": None if base_cache_dir is None else str(base_cache_dir),
        "row_count": row_count,
        "tensor_contract": {
            "dtype": TENSOR_DTYPE,
            "prompt_shape": list(PROMPT_SHAPE),
            "latent_shape": list(LATENT_SHAPE) if mode == "prepare-base" else None,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "height": HEIGHT,
            "width": WIDTH,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "prompt_batch_size": PROMPT_BATCH_SIZE,
            "prompt_device_map": PROMPT_DEVICE_MAP,
            "prompt_shard_max_memory_gib": PROMPT_SHARD_MAX_MEMORY_GIB,
            "prompt_min_cuda_devices": PROMPT_MIN_CUDA_DEVICES,
        },
        "input_binding_sha256": input_binding_sha256(
            mapping if mapping is not None else selected,
            ("candidate_id", "student_prompt") if mapping is not None else ("candidate_id", "factual_prompt", "target_prompt", "target_video_sha256"),
        ),
        "matched_requires_fresh_encode_and_tensor_equality_to_base": mode == "prepare-arm" and arm == "matched",
    }
    return plan, selected, preserve, mapping


def write_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def reserve_output(output_dir: Path, plan: Mapping[str, Any]) -> None:
    require(not output_dir.exists() and not output_dir.is_symlink(), f"output directory must be fresh: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    require(not output_dir.parent.is_symlink(), "output parent is symlinked")
    os.mkdir(output_dir, 0o755)
    write_exclusive(output_dir / "cache_plan.json", canonical_json_bytes(plan))
    if not plan["dry_run"]:
        write_exclusive(output_dir / ".run_reservation", b"cache preparation in progress\n")


class CacheBackend(Protocol):
    dtype_name: str
    def encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]: ...
    def encode_latent(self, video_path: Path) -> Any: ...
    def tensor_shape(self, tensor: Any) -> tuple[int, ...]: ...
    def tensor_dtype(self, tensor: Any) -> str: ...
    def tensor_sha256(self, tensor: Any) -> str: ...
    def tensor_equal(self, left: Any, right: Any) -> bool: ...
    def save(self, payload: Mapping[str, Any], path: Path) -> None: ...
    def load(self, path: Path) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class RealBackend:
    dtype_name = TENSOR_DTYPE
    def __init__(self, model: Path, device: str):
        import torch
        self.torch = torch
        self.model = model
        self.device = torch.device(device)
        self._vae = None
        self._processor = None
        self._mean = None
        self._std = None

    def encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from transformers import AutoTokenizer, UMT5EncoderModel

        device_count = self.torch.cuda.device_count()
        require(
            device_count >= PROMPT_MIN_CUDA_DEVICES,
            f"prompt encoding requires at least {PROMPT_MIN_CUDA_DEVICES} visible CUDA devices",
        )
        max_memory = {
            index: f"{PROMPT_SHARD_MAX_MEMORY_GIB}GiB"
            for index in range(device_count)
        }
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(self.model / "text_encoder"),
            torch_dtype=self.torch.bfloat16,
            device_map=PROMPT_DEVICE_MAP,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        device_map = getattr(text_encoder, "hf_device_map", None)
        require(isinstance(device_map, dict) and device_map, "prompt encoder device map is missing")
        mapped_devices = set(device_map.values())
        require(
            all(type(value) is int for value in mapped_devices)
            and len(mapped_devices) >= 2,
            "prompt encoder was not sharded across CUDA devices",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(self.model / "tokenizer"))
        text_encoder.eval()
        unique_prompts = list(dict.fromkeys(prompts))
        for index, prompt in enumerate(unique_prompts, 1):
            tokenized = tokenizer(prompt, add_special_tokens=True)
            require(len(tokenized.input_ids) <= MAX_SEQUENCE_LENGTH, f"prompt {index} exceeds 226 tokens")
        result: dict[str, Any] = {}
        for start in range(0, len(unique_prompts), PROMPT_BATCH_SIZE):
            batch = unique_prompts[start : start + PROMPT_BATCH_SIZE]
            encoded_batch = fixed_prompt_batch(batch)
            cleaned = [prompt_clean(prompt) for prompt in encoded_batch]
            text_inputs = tokenizer(
                cleaned,
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            mask = text_inputs.attention_mask
            sequence_lengths = mask.gt(0).sum(dim=1).long()
            input_device = text_encoder.shared.weight.device
            with self.torch.inference_mode():
                embeddings = text_encoder(
                    text_inputs.input_ids.to(input_device),
                    mask.to(input_device),
                ).last_hidden_state
            embeddings = embeddings.to(dtype=self.torch.bfloat16, device=self.device)
            unpadded = [value[:length] for value, length in zip(embeddings, sequence_lengths)]
            embeddings = self.torch.stack(
                [
                    self.torch.cat(
                        [
                            value,
                            value.new_zeros(
                                MAX_SEQUENCE_LENGTH - value.size(0), value.size(1)
                            ),
                        ]
                    )
                    for value in unpadded
                ],
                dim=0,
            )
            embeddings = embeddings.detach().contiguous().cpu()
            require(
                tuple(embeddings.shape)
                == (PROMPT_BATCH_SIZE, PROMPT_SHAPE[1], PROMPT_SHAPE[2])
                and embeddings.dtype == self.torch.bfloat16,
                f"prompt batch {start // PROMPT_BATCH_SIZE}: tensor contract mismatch",
            )
            require(
                bool(self.torch.isfinite(embeddings.float()).all()),
                f"prompt batch {start // PROMPT_BATCH_SIZE}: non-finite tensor",
            )
            for offset, prompt in enumerate(batch):
                embedding = embeddings[offset : offset + 1].clone().contiguous()
                self._validate(embedding, PROMPT_SHAPE, f"prompt {start + offset + 1}")
                result[prompt] = embedding
            del embeddings
        del text_encoder, tokenizer
        self._clear()
        return result

    def _init_vae(self) -> None:
        if self._vae is not None:
            return
        from diffusers import AutoencoderKLWan, WanPipeline
        self._vae = AutoencoderKLWan.from_pretrained(str(self.model), subfolder="vae", torch_dtype=self.torch.bfloat16).to(self.device)
        self._vae.eval()
        self._vae.requires_grad_(False)
        if hasattr(self._vae, "enable_tiling"):
            self._vae.enable_tiling()
        self._processor = WanPipeline.from_pretrained(str(self.model), transformer=None, text_encoder=None, tokenizer=None, vae=None).video_processor
        self._mean = self.torch.tensor(self._vae.config.latents_mean, device=self.device, dtype=self.torch.bfloat16).view(1, -1, 1, 1, 1)
        self._std = self.torch.tensor(self._vae.config.latents_std, device=self.device, dtype=self.torch.bfloat16).view(1, -1, 1, 1, 1)

    def encode_latent(self, video_path: Path) -> Any:
        import av
        self._init_vae()
        with av.open(str(video_path)) as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            require(len(streams) == 1, f"{video_path}: expected one video stream")
            stream = streams[0]
            frames = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
            from fractions import Fraction
            require(len(frames) == NUM_FRAMES, f"{video_path}: expected exactly 49 frames")
            require(Fraction(str(stream.average_rate)) == Fraction(FPS, 1), f"{video_path}: expected 8 fps")
            require(stream.width == WIDTH and stream.height == HEIGHT, f"{video_path}: expected 832x480")
        video = self._processor.preprocess_video(frames, height=HEIGHT, width=WIDTH).to(device=self.device, dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            raw = self._vae.encode(video).latent_dist.mode()
            latent = ((raw - self._mean) / self._std).detach().contiguous().cpu()
        self._validate(latent, LATENT_SHAPE, str(video_path))
        del video, raw
        self._clear()
        return latent

    def _validate(self, tensor: Any, shape: tuple[int, ...], label: str) -> None:
        require(tuple(tensor.shape) == shape and tensor.dtype == self.torch.bfloat16, f"{label}: tensor contract mismatch")
        require(bool(self.torch.isfinite(tensor.float()).all()), f"{label}: non-finite tensor")

    def tensor_shape(self, tensor: Any) -> tuple[int, ...]: return tuple(tensor.shape)
    def tensor_dtype(self, tensor: Any) -> str: return str(tensor.dtype)
    def tensor_sha256(self, tensor: Any) -> str:
        value = tensor.detach().contiguous().cpu()
        digest = hashlib.sha256()
        digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(str(value.dtype).encode("ascii")); digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    def tensor_equal(self, left: Any, right: Any) -> bool: return bool(self.torch.equal(left, right))
    def save(self, payload: Mapping[str, Any], path: Path) -> None: self.torch.save(dict(payload), path)
    def load(self, path: Path) -> Mapping[str, Any]: return self.torch.load(path, map_location="cpu", weights_only=True)
    def _clear(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available(): self.torch.cuda.empty_cache()
    def close(self) -> None:
        self._vae = self._processor = self._mean = self._std = None
        self._clear()


def inventory_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def validate_matched_base_cache(base_dir: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = base_dir / "cache_manifest.json"
    plan_path = base_dir / "cache_plan.json"
    regular_file(manifest_path, "matched base cache manifest")
    regular_file(plan_path, "matched base cache plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("protocol") == CACHE_PROTOCOL, "matched base cache protocol mismatch")
    require(manifest.get("protocol_version") == PROTOCOL_VERSION, "matched base cache protocol_version mismatch")
    require(manifest.get("mode") == "prepare-base", "matched base cache mode mismatch")
    require(manifest.get("mechanism") == plan["mechanism"], "matched base cache mechanism mismatch")
    require(manifest.get("row_count") == BASE_ROWS, "matched base cache row count mismatch")
    require(manifest.get("registry", {}).get("sha256") == plan["registry"]["sha256"], "matched base cache registry mismatch")
    require(manifest.get("model_inventory", {}).get("sha256") == plan["model_inventory"]["sha256"], "matched base cache model mismatch")
    records = manifest.get("files")
    require(isinstance(records, list) and len(records) == BASE_ROWS, "matched base cache file registry invalid")
    paths = [base_dir / str(record.get("file", "")) for record in records]
    require(all(path.is_file() and not path.is_symlink() for path in paths), "matched base cache PT file missing/symlinked")
    require(len(set(paths)) == BASE_ROWS, "matched base cache PT paths are not unique")
    require(set(base_dir.iterdir()) == {manifest_path, plan_path, *paths}, "matched base cache directory has unexpected files")
    for record, path in zip(records, paths):
        require(record.get("sha256") == sha256_file(path), f"matched base cache file changed: {path.name}")
    require(manifest.get("ordered_inventory_sha256") == inventory_sha256(paths), "matched base cache inventory hash mismatch")
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}


def _validate_input_video(path: Path, expected_sha: str, label: str) -> None:
    regular_file(path, label)
    require(sha256_file(path) == expected_sha, f"{label} byte hash mismatch")


def _cache_filename(index: int, row_id: str) -> str:
    require(SAFE_ID.fullmatch(row_id) is not None, f"unsafe cache row ID: {row_id}")
    return f"{index:03d}_{row_id}.pt"


def prepare_cache(
    plan: Mapping[str, Any], selected: Sequence[Mapping[str, Any]],
    preserve: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, Any]] | None,
    backend: CacheBackend,
) -> dict[str, Any]:
    output = Path(str(plan["output_dir"]))
    mode = str(plan["mode"])
    arm = plan["arm"]
    files: list[dict[str, Any]] = []
    paths: list[Path] = []
    if mode == "prepare-base":
        prompts = [str(row["factual_prompt"]) for row in selected] + [str(row["prompt"]) for row in preserve]
        embeddings = backend.encode_prompts(prompts)
        work = [
            (index, "erase", str(row["candidate_id"]), str(row["factual_prompt"]), row)
            for index, row in enumerate(selected)
        ] + [
            (ERASE_ROWS + index, "preserve", str(row["preserve_id"]), str(row["prompt"]), row)
            for index, row in enumerate(preserve)
        ]
        for manifest_index, role, row_id, prompt, row in work:
            video = Path(str(row["target_video_path"]))
            _validate_input_video(video, str(row["target_video_sha256"]), f"{role} video {row_id}")
            latent = backend.encode_latent(video)
            embedding = embeddings[prompt]
            require(backend.tensor_shape(latent) == LATENT_SHAPE, f"{row_id}: latent shape mismatch")
            require(backend.tensor_shape(embedding) == PROMPT_SHAPE, f"{row_id}: prompt shape mismatch")
            require(backend.tensor_dtype(latent) == TENSOR_DTYPE, f"{row_id}: latent dtype mismatch")
            require(backend.tensor_dtype(embedding) == TENSOR_DTYPE, f"{row_id}: prompt dtype mismatch")
            payload = {
                "protocol": CACHE_PROTOCOL, "protocol_version": PROTOCOL_VERSION,
                "mode": mode, "mechanism": plan["mechanism"], "manifest_index": manifest_index,
                "row_id": row_id, "training_role": role, "canonical_factual_prompt": prompt,
                "target_video_path": str(video), "target_video_sha256": row["target_video_sha256"],
                "latents": latent, "latents_sha256": backend.tensor_sha256(latent),
                "prompt_embeds": embedding,
                "canonical_factual_prompt_embeds_sha256": backend.tensor_sha256(embedding),
                "registry_sha256": plan["registry"]["sha256"],
                "model_inventory_sha256": plan["model_inventory"]["sha256"],
                "runtime_registry_sha256": plan["runtime_registry"]["sha256"],
            }
            path = output / _cache_filename(manifest_index, row_id)
            backend.save(payload, path); paths.append(path)
            files.append({"index": manifest_index, "row_id": row_id, "file": path.name, "sha256": sha256_file(path), "latents_sha256": payload["latents_sha256"], "prompt_embeds_sha256": payload["canonical_factual_prompt_embeds_sha256"]})
    else:
        if mode == "prepare-teacher":
            prompts = [str(row["target_prompt"]) for row in selected]
        else:
            require(mapping is not None, "arm mapping missing")
            prompts = [str(row["student_prompt"]) for row in mapping]
        embeddings = backend.encode_prompts(prompts)
        base_dir = None if plan["base_cache_dir"] is None else Path(str(plan["base_cache_dir"]))
        base_manifest_record = None
        if arm == "matched":
            require(base_dir is not None and base_dir.is_dir() and not base_dir.is_symlink(), "matched base cache is invalid")
            base_manifest_record = validate_matched_base_cache(base_dir, plan)
        for index, selected_row in enumerate(selected):
            row_id = str(selected_row["candidate_id"])
            prompt = str(selected_row["target_prompt"]) if mode == "prepare-teacher" else str(mapping[index]["student_prompt"])
            embedding = embeddings[prompt]
            require(backend.tensor_shape(embedding) == PROMPT_SHAPE, f"{row_id}: prompt shape mismatch")
            require(backend.tensor_dtype(embedding) == TENSOR_DTYPE, f"{row_id}: prompt dtype mismatch")
            tensor_key = "teacher_prompt_embeds" if mode == "prepare-teacher" else "student_prompt_embeds"
            payload = {
                "protocol": CACHE_PROTOCOL, "protocol_version": PROTOCOL_VERSION,
                "mode": mode, "arm": arm, "mechanism": plan["mechanism"],
                "manifest_index": index, "selected_index": index, "candidate_id": row_id,
                "prompt": prompt, tensor_key: embedding,
                f"{tensor_key}_sha256": backend.tensor_sha256(embedding),
                "registry_sha256": plan["registry"]["sha256"],
                "model_inventory_sha256": plan["model_inventory"]["sha256"],
                "runtime_registry_sha256": plan["runtime_registry"]["sha256"],
            }
            if mode == "prepare-arm":
                payload["assigned_source_id"] = mapping[index]["assigned_source_id"]
                payload["assigned_source_object"] = mapping[index]["assigned_source_object"]
            if arm == "matched":
                candidates = list(base_dir.glob(f"{index:03d}_*.pt"))
                require(len(candidates) == 1, f"matched row {index}: base cache file not unique")
                base = backend.load(candidates[0])
                require(base.get("row_id") == row_id, f"matched row {index}: base row ID mismatch")
                require(base.get("canonical_factual_prompt") == prompt, f"matched row {index}: base prompt mismatch")
                require(backend.tensor_equal(embedding, base.get("prompt_embeds")), f"matched row {index}: freshly encoded tensor differs from base canonical tensor")
                payload["base_cache_file_sha256"] = sha256_file(candidates[0])
                payload["base_cache_manifest_sha256"] = base_manifest_record["sha256"]
                payload["matched_base_tensor_equal"] = True
            path = output / _cache_filename(index, row_id)
            backend.save(payload, path); paths.append(path)
            files.append({"index": index, "row_id": row_id, "file": path.name, "sha256": sha256_file(path), "prompt_embeds_sha256": payload[f"{tensor_key}_sha256"]})
    expected = BASE_ROWS if mode == "prepare-base" else ERASE_ROWS
    require(len(paths) == expected and len(list(output.glob("*.pt"))) == expected, "cache PT inventory count mismatch")
    manifest = {
        **{key: plan[key] for key in ("schema_version", "protocol", "protocol_version", "mode", "arm", "mechanism", "registry", "selected178", "preserve36", "model_root", "model_inventory", "runtime_root", "runtime_registry", "row_count", "tensor_contract", "input_binding_sha256")},
        "status": "complete", "inventory_algorithm": INVENTORY_ALGORITHM,
        "ordered_inventory_sha256": inventory_sha256(paths), "files": files,
    }
    if mode == "prepare-arm" and arm == "matched":
        manifest["matched_base_cache_manifest"] = base_manifest_record
    write_exclusive(output / "cache_manifest.json", canonical_json_bytes(manifest))
    (output / ".run_reservation").unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--base-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(); args = parser.parse_args(argv)
    if args.run and args.device != "cuda": parser.error("formal cache preparation requires --device cuda")
    try:
        reject_sealed(args.project_root, args.registry, args.output_dir, args.base_cache_dir or "")
        project_root = args.project_root.resolve(strict=True)
        registry_path = resolve_path(project_root, args.registry)
        output = resolve_path(project_root, args.output_dir, strict=False)
        base = None if args.base_cache_dir is None else resolve_path(project_root, args.base_cache_dir, strict=args.run)
        registry = load_registry(project_root, registry_path, args.registry_sha256)
        validate_model_inventory(registry, live=False)
        validate_runtime(registry, live=False)
        plan, selected, preserve, mapping = build_plan(registry, args.mode, output, arm=args.arm, base_cache_dir=base, dry_run=args.dry_run)
        if args.dry_run:
            reserve_output(output, plan)
            print(f"Planned {args.mode} cache at {output}")
            return 0
        require(Path(sys.executable).resolve() == Path(registry["python_executable"]).resolve(), "run must use registry Python executable")
        validate_model_inventory(registry, live=True)
        validate_runtime(registry, live=True)
        reserve_output(output, plan)
        backend = RealBackend(Path(registry["model_root"]), args.device)
        try:
            manifest = prepare_cache(plan, selected, preserve, mapping, backend)
        finally:
            backend.close()
        reopened = load_registry(project_root, registry_path, args.registry_sha256)
        require(reopened["registry_sha256"] == registry["registry_sha256"], "registry changed during cache preparation")
        print(f"Prepared {manifest['row_count']} {args.mode} entries; inventory={manifest['ordered_inventory_sha256']}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
