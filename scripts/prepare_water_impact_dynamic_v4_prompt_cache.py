#!/usr/bin/env python3
"""Prepare and preflight the read-only v4 augmented factual-prompt sidecar.

The two commands are intentionally separate processes:

* ``prepare-cache`` writes the 178 augmented erase-row embeddings after a
  full original-prompt re-encode comparison.
* ``null-preflight`` independently re-encodes the original prompts and proves
  null-sidecar forward/loss/LoRA-gradient, schedule, and RNG equivalence.

Neither command reads an evaluator-private source registry.  Both consume the
public, already-frozen source mapping only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import WanPipeline, WanTransformer3DModel
from peft import LoraConfig
from transformers import AutoTokenizer

from build_water_impact_dynamic_v4_runtime_registry import (
    EXPECTED_PACKAGE_VERSIONS,
    validate_runtime_registry,
)
from build_water_impact_dynamic_v4_source_mapping import (
    EXPECTED_ACTIVE_ERASE_ROWS,
    EXPECTED_ERASE_ROWS,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_ROWS,
    EXPECTED_SAMPLE_ORDER_SHA256,
    EXPECTED_SEED,
    PROMPT_BUILDER_FILE,
    PROMPT_BUILDER_PATH,
    PROTOCOL as MAPPING_PROTOCOL,
    balanced_v3b_schedule,
    build_mapping,
    canonical_json_sha256,
    factual_prompt,
    file_sha256,
    load_frozen_rows,
    require_sha256,
    sample_order_sha256,
    validate_public_bank_registry,
    validate_public_holdout_commitment,
    validate_public_stage0_commitment,
)


CACHE_PROTOCOL = "water_impact_dynamic_v4_source_prompt_sidecar_v2"
PREFLIGHT_PROTOCOL = "water_impact_dynamic_v4_null_sidecar_preflight_v2"
DATASET_VERSION = "v4_dev72_v2"
MODEL_INVENTORY_ALGORITHM = (
    "sha256_ordered_relative_path_nul_bytes_newline_with_file_records_v1"
)
MODEL_INVENTORY_EXCLUDED = [
    "any .cache directory",
    "*.tmp",
    "*.lock",
    "*.incomplete",
    "*~",
]
EXPECTED_BASE_CACHE_SHA256 = (
    "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
)
EXPECTED_TEACHER_CACHE_SHA256 = (
    "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
)
EXPECTED_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
EXPECTED_MODEL_CONTENT_INVENTORY_SHA256 = (
    "0a8566eeab29dfbc04303167ce1904b65b964dd1579959645d1f93e19ba15ddf"
)
EXPECTED_PROMPT_SHAPE = (1, 226, 4096)
EXPECTED_PROMPT_DTYPE = torch.bfloat16
EXPECTED_SIDECAR_RUNTIME_VERSIONS = {
    package: EXPECTED_PACKAGE_VERSIONS[package]
    for package in (
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
    )
}
EXPECTED_NOISE_RNG_INITIAL_SHA256 = (
    "49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704"
)
EXPECTED_NOISE_RNG_FINAL_SHA256 = (
    "79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f"
)
EXPECTED_INITIAL_LORA_SHA256 = (
    "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8"
)
EXPECTED_HOLDOUT_PUBLIC_COMMITMENT = Path(
    "data/water_impact_dynamic_v4/holdout_public_commitment_v2.json"
)
EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT = Path(
    "data/water_impact_dynamic_v4/causal_stage0_public_commitment_v2.json"
)
EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256 = (
    "0d7fab1befdc197a7ae7f864a84c1f1ac3d029d5d72f9a513303892e48ec2477"
)
EXPECTED_RUNTIME_REGISTRY = Path(
    "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json"
)
FROZEN_TRANSFORMER_INVENTORY_SHA256 = (
    "fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac"
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            _contains_placeholder(key) or _contains_placeholder(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(child) for child in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return any(
            token in lowered
            for token in ("placeholder", "todo", "tbd", "fill_me", "to_be_frozen")
        )
    return False


def trainable_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise ValueError(f"trainable parameter has no null-preflight gradient: {name}")
        value = parameter.grad.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_model_content_inventory(
    model: Path,
) -> dict[str, Any]:
    """Replicate the frozen v3c Stage-2 full-pipeline inventory algorithm."""

    if model.is_symlink() or not model.is_dir():
        raise FileNotFoundError(f"model directory is missing: {model}")
    paths: list[Path] = []
    excluded_suffixes = (".tmp", ".lock", ".incomplete", "~")
    for path in model.rglob("*"):
        relative = path.relative_to(model)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"model inventory forbids symlinks: {path}")
        if path.is_file() and not path.name.endswith(excluded_suffixes):
            paths.append(path)
    paths.sort(key=lambda path: path.relative_to(model).as_posix())
    if not paths:
        raise ValueError("model content inventory is empty")
    required = {
        "model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
    }
    relative_names = {path.relative_to(model).as_posix() for path in paths}
    missing = required - relative_names
    if missing:
        raise FileNotFoundError(f"model content inventory is missing: {sorted(missing)}")
    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(model).as_posix()
        per_file = hashlib.sha256()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                per_file.update(chunk)
                aggregate.update(chunk)
        aggregate.update(b"\n")
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": per_file.hexdigest()}
        )
    return {
        "algorithm": MODEL_INVENTORY_ALGORITHM,
        "root": str(model),
        "excluded": list(MODEL_INVENTORY_EXCLUDED),
        "file_count": len(paths),
        "sha256": aggregate.hexdigest(),
        "files": records,
    }


def validate_model_content_inventory(
    model: Path, expected_sha256: str
) -> dict[str, Any]:
    require_sha256(expected_sha256, "model content inventory hash")
    if expected_sha256 != EXPECTED_MODEL_CONTENT_INVENTORY_SHA256:
        raise ValueError(
            "model content inventory hash differs from frozen v3c Stage-2 inventory"
        )
    inventory = compute_model_content_inventory(model)
    if inventory["sha256"] != expected_sha256:
        raise ValueError(
            f"model content inventory mismatch: {inventory['sha256']} != {expected_sha256}"
        )
    revision_path = model / ".cache/huggingface/download/model_index.json.metadata"
    if revision_path.is_symlink() or not revision_path.is_file():
        raise ValueError(f"cannot validate a non-file or symlinked model revision: {revision_path}")
    try:
        revision = revision_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError) as exc:
        raise ValueError(f"cannot validate model revision at {revision_path}: {exc}") from exc
    if revision != EXPECTED_MODEL_REVISION:
        raise ValueError(f"model revision mismatch: {revision} != {EXPECTED_MODEL_REVISION}")
    return {
        "model_artifact_inventory": inventory,
        "model_revision": revision,
    }


def cache_inventory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.resolve(strict=True).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def exact_cache_paths(
    rows: list[dict[str, str]], cache_dir: Path, *, role: str | None = None
) -> list[Path]:
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise FileNotFoundError(f"cache directory is missing or symlinked: {cache_dir}")
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if role is None or row["training_role"] == role
    ]
    expected = [cache_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in selected]
    actual = sorted(cache_dir.glob("*.pt"))
    missing = [path for path in expected if path.is_symlink() or not path.is_file()]
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected or len(actual) != len(expected):
        raise ValueError(
            f"cache inventory mismatch for {cache_dir}: expected={len(expected)} "
            f"actual={len(actual)} missing={len(missing)} unexpected={len(unexpected)}"
        )
    return expected


def validate_cache_inventory(
    rows: list[dict[str, str]],
    cache_dir: Path,
    *,
    expected_sha256: str,
    role: str | None = None,
) -> list[Path]:
    require_sha256(expected_sha256, f"cache inventory hash for {cache_dir}")
    paths = exact_cache_paths(rows, cache_dir, role=role)
    actual = cache_inventory_sha256(paths)
    if actual != expected_sha256:
        raise ValueError(f"cache inventory hash mismatch for {cache_dir}: {actual}")
    return paths


def load_mapping_registry(
    path: Path,
    rows: list[dict[str, str]],
    *,
    expected_sha256: str,
    expected_bank_sha256: str,
    expected_holdout_commitment_path: str,
    expected_holdout_commitment_sha256: str,
    bank_registry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require_sha256(expected_sha256, "source mapping registry hash")
    require_sha256(expected_bank_sha256, "source bank registry hash")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"source mapping registry is missing or symlinked: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"source mapping registry hash mismatch: {actual} != {expected_sha256}")
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid source mapping registry: {exc}") from exc
    expected_scalars = {
        "protocol": MAPPING_PROTOCOL,
        "status": "frozen",
        "dataset_version": DATASET_VERSION,
        "source_bank_registry_sha256": expected_bank_sha256,
        "holdout_public_commitment_path": expected_holdout_commitment_path,
        "holdout_public_commitment_sha256": expected_holdout_commitment_sha256,
        "holdout_count": 24,
        "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "seed": EXPECTED_SEED,
        "sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
        "canonical_prompt_builder_path": str(PROMPT_BUILDER_PATH),
        "canonical_prompt_builder_sha256": file_sha256(PROMPT_BUILDER_FILE),
        "erase_row_count": EXPECTED_ERASE_ROWS,
        "active_erase_count": EXPECTED_ACTIVE_ERASE_ROWS,
        "active_source_count_min": 1,
        "active_source_count_max": 2,
    }
    for key, value in expected_scalars.items():
        if registry.get(key) != value:
            raise ValueError(f"source mapping {key} mismatch: {registry.get(key)!r} != {value!r}")
    if bank_registry is None:
        raise ValueError("source mapping validation requires the exact public bank registry")
    reconstructed = build_mapping(
        rows,
        bank_registry,
        bank_registry_sha256=expected_bank_sha256,
        holdout_commitment_path=expected_holdout_commitment_path,
        holdout_commitment_sha256=expected_holdout_commitment_sha256,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
    )
    if registry != reconstructed:
        raise ValueError(
            "source mapping registry is not the unique reconstruction from bank, salt, and manifest"
        )
    records = registry.get("mapping")
    if not isinstance(records, list) or len(records) != EXPECTED_ERASE_ROWS:
        raise ValueError("source mapping must contain exactly 178 records")
    if canonical_json_sha256(records) != registry.get("full178_mapping_sha256"):
        raise ValueError("full178 mapping digest mismatch")
    active = sorted(
        (record for record in records if record.get("active_erase_ordinal") is not None),
        key=lambda record: record["active_erase_ordinal"],
    )
    if len(active) != EXPECTED_ACTIVE_ERASE_ROWS:
        raise ValueError("source mapping must contain exactly 100 active records")
    if canonical_json_sha256(active) != registry.get("active100_mapping_sha256"):
        raise ValueError("active100 mapping digest mismatch")
    if [record.get("active_erase_ordinal") for record in active] != list(
        range(EXPECTED_ACTIVE_ERASE_ROWS)
    ):
        raise ValueError("active erase ordinals must be the exact range 0..99")
    active_counts = Counter(record.get("assigned_source_id") for record in active)
    if (
        len(active_counts) != 64
        or set(active_counts.values()) - {1, 2}
        or dict(sorted(active_counts.items())) != registry.get("active_source_counts")
    ):
        raise ValueError("active source counts do not match the balanced registry")
    bank_by_id = {source["source_id"]: source for source in bank_registry["entries"]}
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        index = record.get("manifest_index")
        if not isinstance(index, int) or index in by_index or not (0 <= index < len(rows)):
            raise ValueError(f"invalid or duplicate mapping manifest_index: {index!r}")
        row = rows[index]
        if row["training_role"] != "erase" or record.get("scene_id") != row["scene_id"]:
            raise ValueError(f"mapping row does not bind erase manifest index {index}")
        expected_original = {
            "original_source_id": row["source_id"],
            "original_source_phrase": row["source_object"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
            "prompt_variant": row["prompt_variant"],
            "original_factual_prompt": row["prompt"],
        }
        for key, value in expected_original.items():
            if record.get(key) != value:
                raise ValueError(f"mapping record {index} has drifted {key}")
        if record.get("assigned_source_phrase") == row["source_object"]:
            raise ValueError(f"mapping record {index} violates source derangement")
        rebuilt = factual_prompt(
            record.get("assigned_source_phrase", ""),
            row["receiver"],
            row["prompt_variant"],
        )
        if record.get("augmented_factual_prompt") != rebuilt:
            raise ValueError(f"mapping record {index} is not a canonical source-slot-only edit")
        source = bank_by_id.get(record.get("assigned_source_id"))
        if source is None:
            raise ValueError(f"mapping record {index} uses a source outside the public bank")
        if (
            record.get("assigned_source_phrase") != source["source_phrase"]
            or record.get("assigned_source_membership") != source["membership"]
        ):
            raise ValueError(f"mapping record {index} disagrees with its public bank source")
        by_index[index] = record
    if len(by_index) != EXPECTED_ERASE_ROWS:
        raise ValueError("mapping does not cover every erase manifest row")
    return registry, by_index


def token_length(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


@torch.no_grad()
def encode_prompts(
    pipe: WanPipeline,
    prompts: list[str],
    *,
    device: torch.device,
    label: str,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for index, prompt in enumerate(prompts, start=1):
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=device,
            dtype=torch.bfloat16,
        )
        value = prompt_embeds.detach().contiguous().cpu()
        if tuple(value.shape) != EXPECTED_PROMPT_SHAPE or value.dtype != EXPECTED_PROMPT_DTYPE:
            raise ValueError(
                f"{label} prompt embedding has unexpected shape/dtype: "
                f"{tuple(value.shape)} {value.dtype}"
            )
        if not torch.isfinite(value.float()).all():
            raise FloatingPointError(f"{label} prompt embedding is non-finite")
        embeddings[prompt] = value
        print(f"Encoded {label} prompt {index}/{len(prompts)}", flush=True)
    return embeddings


def _validate_original_reencodes(
    rows: list[dict[str, str]],
    base_paths: list[Path],
    embeddings: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["training_role"] != "erase":
            continue
        payload = torch.load(base_paths[index], map_location="cpu", weights_only=True)
        cached = payload.get("prompt_embeds")
        encoded = embeddings[row["prompt"]]
        if not isinstance(cached, torch.Tensor):
            raise ValueError(f"base cache has no prompt_embeds: {base_paths[index]}")
        if (
            tuple(cached.shape) != EXPECTED_PROMPT_SHAPE
            or cached.dtype != EXPECTED_PROMPT_DTYPE
            or not torch.equal(cached, encoded)
        ):
            raise ValueError(
                f"fresh original prompt encoding differs from frozen base cache at {index}"
            )
        digest = tensor_sha256(encoded)
        records.append({"manifest_index": index, "scene_id": row["scene_id"], "sha256": digest})
    return records


def _validate_augmented_reencodes(
    rows: list[dict[str, str]],
    mapping_by_index: dict[int, dict[str, Any]],
    sidecar_paths: dict[int, Path],
    fresh_embeddings: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Independently bind every stored sidecar row to a fresh text encoding."""

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["training_role"] != "erase":
            continue
        mapping_record = mapping_by_index[index]
        prompt = mapping_record["augmented_factual_prompt"]
        fresh = fresh_embeddings[prompt]
        payload = torch.load(
            sidecar_paths[index], map_location="cpu", weights_only=True
        )
        stored = payload["augmented_prompt_embeds"]
        if not torch.equal(stored, fresh):
            raise ValueError(
                f"fresh augmented prompt encoding differs from sidecar at {index}"
            )
        records.append(
            {
                "manifest_index": index,
                "scene_id": row["scene_id"],
                "assigned_source_id": mapping_record["assigned_source_id"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "embedding_sha256": tensor_sha256(fresh),
            }
        )
    if len(records) != EXPECTED_ERASE_ROWS:
        raise ValueError("augmented re-encode did not cover all erase rows")
    return records


def tokenizer_inventory_binding(model_provenance: dict[str, Any]) -> dict[str, Any]:
    records = [
        record
        for record in model_provenance["model_artifact_inventory"]["files"]
        if record["path"].startswith("tokenizer/")
    ]
    if not records:
        raise ValueError("model inventory contains no tokenizer files")
    return {
        "path": "models/Wan2.1-T2V-1.3B-Diffusers/tokenizer",
        "file_count": len(records),
        "inventory_sha256": canonical_json_sha256(records),
    }


def validate_prompt_sidecar_payload(
    payload: Any, *, scalar_expected: dict[str, Any], path: Path
) -> tuple[torch.Tensor, int]:
    exact_fields = {
        *scalar_expected,
        "augmented_prompt_embeds",
        "augmented_prompt_embeds_sha256",
        "registered_token_length",
    }
    if not isinstance(payload, dict) or set(payload) != exact_fields:
        raise ValueError(f"prompt sidecar {path} fields are not exact")
    if _contains_placeholder(payload):
        raise ValueError(f"prompt sidecar {path} contains a placeholder")
    for key, value in scalar_expected.items():
        if payload[key] != value:
            raise ValueError(f"prompt sidecar {path} has drifted {key}")
    embedding = payload["augmented_prompt_embeds"]
    if (
        not isinstance(embedding, torch.Tensor)
        or tuple(embedding.shape) != EXPECTED_PROMPT_SHAPE
        or embedding.dtype != EXPECTED_PROMPT_DTYPE
        or not torch.isfinite(embedding.float()).all()
    ):
        raise ValueError(f"prompt sidecar {path} has invalid embedding tensor")
    if tensor_sha256(embedding) != payload["augmented_prompt_embeds_sha256"]:
        raise ValueError(f"prompt sidecar {path} embedding digest mismatch")
    length = payload["registered_token_length"]
    if not isinstance(length, int) or isinstance(length, bool) or not (1 <= length <= 226):
        raise ValueError(f"prompt sidecar {path} has invalid token length")
    return embedding, length


def reserve_new_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to reuse {label} directory: {path}") from exc
    (path / ".run_reservation").write_text(
        f"pid={os.getpid()} status=building\n", encoding="utf-8"
    )


def atomic_write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
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


def prepare_cache(args: argparse.Namespace) -> int:
    validate_runtime_registry(
        args.runtime_registry,
        args.runtime_registry_sha256,
        project_root=Path("."),
        verify_current_runtime=True,
    )
    rows = load_frozen_rows(args.manifest, expected_sha256=args.manifest_sha256)
    bank, _ = validate_public_bank_registry(
        args.source_bank_registry,
        expected_sha256=args.source_bank_registry_sha256,
    )
    holdout, _ = validate_public_holdout_commitment(
        args.holdout_public_commitment,
        expected_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    validate_public_stage0_commitment(
        args.causal_stage0_public_commitment,
        expected_sha256=args.causal_stage0_public_commitment_sha256,
        bank_registry=bank,
        holdout_commitment=holdout,
    )
    mapping, by_index = load_mapping_registry(
        args.source_mapping_registry,
        rows,
        expected_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_path=str(args.holdout_public_commitment),
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    base_paths = validate_cache_inventory(
        rows,
        args.base_cache_dir,
        expected_sha256=args.base_cache_sha256,
    )
    validate_cache_inventory(
        rows,
        args.teacher_cache_dir,
        expected_sha256=args.teacher_cache_sha256,
        role="erase",
    )
    model_provenance = validate_model_content_inventory(
        args.model, args.model_content_inventory_sha256
    )
    reserve_new_directory(args.output_dir, "v4 prompt sidecar")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), subfolder="tokenizer")
    original_prompts = list(
        dict.fromkeys(row["prompt"] for row in rows if row["training_role"] == "erase")
    )
    augmented_prompts = list(
        dict.fromkeys(
            by_index[index]["augmented_factual_prompt"]
            for index, row in enumerate(rows)
            if row["training_role"] == "erase"
        )
    )
    lengths = {
        prompt: token_length(tokenizer, prompt)
        for prompt in dict.fromkeys(original_prompts + augmented_prompts)
    }
    too_long = [(prompt, length) for prompt, length in lengths.items() if length > 226]
    if too_long:
        raise ValueError(f"v4 prompt would be truncated at 226 tokens: {too_long[0][1]}")

    pipe = WanPipeline.from_pretrained(
        str(args.model), transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.text_encoder.eval()
    original_embeddings = encode_prompts(
        pipe, original_prompts, device=device, label="original factual"
    )
    reencode_records = _validate_original_reencodes(rows, base_paths, original_embeddings)
    augmented_embeddings = encode_prompts(
        pipe, augmented_prompts, device=device, label="augmented factual"
    )
    del pipe, tokenizer
    clear_memory()

    cache_paths: list[Path] = []
    sidecar_records: list[dict[str, Any]] = []
    for manifest_index, row in enumerate(rows):
        if row["training_role"] != "erase":
            continue
        record = by_index[manifest_index]
        prompt = record["augmented_factual_prompt"]
        embedding = augmented_embeddings[prompt]
        embedding_sha256 = tensor_sha256(embedding)
        cache_path = args.output_dir / f"{manifest_index:03d}_{row['scene_id']}.pt"
        torch.save(
            {
                "protocol": CACHE_PROTOCOL,
                "dataset_version": DATASET_VERSION,
                "manifest_index": manifest_index,
                "scene_id": row["scene_id"],
                "training_role": "erase",
                "original_source_id": record["original_source_id"],
                "original_source_phrase": record["original_source_phrase"],
                "assigned_source_id": record["assigned_source_id"],
                "assigned_source_phrase": record["assigned_source_phrase"],
                "augmented_factual_prompt": prompt,
                "augmented_prompt_embeds": embedding,
                "augmented_prompt_embeds_sha256": embedding_sha256,
                "registered_token_length": lengths[prompt],
                "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
                "source_bank_registry_sha256": args.source_bank_registry_sha256,
                "holdout_public_commitment_sha256": (
                    args.holdout_public_commitment_sha256
                ),
                "train_manifest_sha256": args.manifest_sha256,
                "model_content_inventory_sha256": args.model_content_inventory_sha256,
                "runtime_registry_sha256": args.runtime_registry_sha256,
            },
            cache_path,
        )
        cache_paths.append(cache_path)
        sidecar_records.append(
            {
                "file": cache_path.name,
                "manifest_index": manifest_index,
                "scene_id": row["scene_id"],
                "assigned_source_id": record["assigned_source_id"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "embedding_sha256": embedding_sha256,
                "registered_token_length": lengths[prompt],
            }
        )
    inventory_hash = cache_inventory_sha256(cache_paths)
    manifest = {
        "protocol": CACHE_PROTOCOL,
        "status": "prepared",
        "dataset_version": DATASET_VERSION,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": args.manifest_sha256,
        "source_mapping_registry": str(args.source_mapping_registry),
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "holdout_public_commitment_path": str(args.holdout_public_commitment),
        "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": str(
            args.causal_stage0_public_commitment
        ),
        "causal_stage0_public_commitment_sha256": (
            args.causal_stage0_public_commitment_sha256
        ),
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "canonical_prompt_builder_path": mapping["canonical_prompt_builder_path"],
        "canonical_prompt_builder_sha256": mapping[
            "canonical_prompt_builder_sha256"
        ],
        "base_cache_dir": str(args.base_cache_dir),
        "base_cache_inventory_sha256": args.base_cache_sha256,
        "teacher_cache_dir": str(args.teacher_cache_dir),
        "teacher_cache_inventory_sha256": args.teacher_cache_sha256,
        "model": str(args.model),
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "runtime_registry_path": str(args.runtime_registry),
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        **model_provenance,
        "dtype": str(EXPECTED_PROMPT_DTYPE),
        "shape": list(EXPECTED_PROMPT_SHAPE),
        "max_sequence_length": 226,
        "truncation_allowed": False,
        "erase_row_count": len(cache_paths),
        "unique_original_prompt_count": len(original_prompts),
        "unique_augmented_prompt_count": len(augmented_prompts),
        "max_registered_token_length": max(
            lengths[prompt] for prompt in augmented_prompts
        ),
        "original_reencode_binding_sha256": canonical_json_sha256(reencode_records),
        "sidecar_record_binding_sha256": canonical_json_sha256(sidecar_records),
        "cache_inventory_sha256": inventory_hash,
        "files": [path.name for path in cache_paths],
        "runtime_versions": dict(EXPECTED_SIDECAR_RUNTIME_VERSIONS),
    }
    atomic_write_new_json(args.output_dir / "cache_manifest_v2.json", manifest)
    (args.output_dir / ".run_reservation").unlink()
    print(
        f"Prepared {len(cache_paths)} augmented factual embeddings; "
        f"inventory_sha256={inventory_hash}",
        flush=True,
    )
    return 0


def validate_prompt_sidecar(
    rows: list[dict[str, str]],
    mapping_by_index: dict[int, dict[str, Any]],
    sidecar_dir: Path,
    *,
    expected_inventory_sha256: str,
    expected_manifest_sha256: str,
    expected_mapping_sha256: str,
    expected_bank_sha256: str,
    expected_holdout_commitment_sha256: str,
    expected_causal_stage0_public_commitment_sha256: str,
    expected_model_inventory_sha256: str,
    expected_runtime_registry_sha256: str,
    expected_model_provenance: dict[str, Any],
) -> tuple[dict[int, Path], dict[str, Any]]:
    require_sha256(expected_inventory_sha256, "prompt sidecar inventory hash")
    require_sha256(expected_manifest_sha256, "prompt sidecar manifest hash")
    if (
        not isinstance(expected_model_provenance, dict)
        or set(expected_model_provenance)
        != {"model_artifact_inventory", "model_revision"}
        or expected_model_provenance["model_revision"] != EXPECTED_MODEL_REVISION
    ):
        raise ValueError("prompt sidecar expected model provenance is not exact")
    if sidecar_dir.is_symlink() or not sidecar_dir.is_dir():
        raise FileNotFoundError("prompt sidecar directory is missing or symlinked")
    erase = [
        (index, row)
        for index, row in enumerate(rows)
        if row["training_role"] == "erase"
    ]
    expected = [sidecar_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in erase]
    actual = sorted(sidecar_dir.glob("*.pt"))
    if (
        set(actual) != set(expected)
        or len(actual) != EXPECTED_ERASE_ROWS
        or any(path.is_symlink() or not path.is_file() for path in expected)
    ):
        raise ValueError("prompt sidecar must contain exactly the 178 registered .pt entries")
    expected_children = {path.name for path in expected} | {"cache_manifest_v2.json"}
    if {path.name for path in sidecar_dir.iterdir()} != expected_children:
        raise ValueError("prompt sidecar directory contains an unregistered artifact")
    inventory = cache_inventory_sha256(expected)
    if inventory != expected_inventory_sha256:
        raise ValueError(f"prompt sidecar inventory mismatch: {inventory}")
    manifest_path = sidecar_dir / "cache_manifest_v2.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("prompt sidecar cache_manifest_v2.json is missing or symlinked")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("prompt sidecar cache_manifest_v2.json byte hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_manifest_fields = {
        "protocol",
        "status",
        "dataset_version",
        "source_manifest",
        "source_manifest_sha256",
        "source_mapping_registry",
        "source_mapping_registry_sha256",
        "source_bank_registry_sha256",
        "holdout_public_commitment_path",
        "holdout_public_commitment_sha256",
        "holdout_count",
        "causal_stage0_public_commitment_path",
        "causal_stage0_public_commitment_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "base_cache_dir",
        "base_cache_inventory_sha256",
        "teacher_cache_dir",
        "teacher_cache_inventory_sha256",
        "model",
        "model_content_inventory_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "transformer_inventory_sha256",
        "model_artifact_inventory",
        "model_revision",
        "dtype",
        "shape",
        "max_sequence_length",
        "truncation_allowed",
        "erase_row_count",
        "unique_original_prompt_count",
        "unique_augmented_prompt_count",
        "max_registered_token_length",
        "original_reencode_binding_sha256",
        "sidecar_record_binding_sha256",
        "cache_inventory_sha256",
        "files",
        "runtime_versions",
    }
    if not isinstance(manifest, dict) or set(manifest) != exact_manifest_fields:
        raise ValueError("prompt sidecar manifest fields are not exact")
    if _contains_placeholder(manifest):
        raise ValueError("prompt sidecar manifest contains a placeholder")
    full_mapping = sorted(
        mapping_by_index.values(), key=lambda record: record["erase_ordinal"]
    )
    active_mapping = sorted(
        (
            record
            for record in mapping_by_index.values()
            if record["active_erase_ordinal"] is not None
        ),
        key=lambda record: record["active_erase_ordinal"],
    )
    original_prompts = {
        row["prompt"] for row in rows if row["training_role"] == "erase"
    }
    augmented_prompts = {
        record["augmented_factual_prompt"] for record in mapping_by_index.values()
    }
    expected_manifest_values = {
        "protocol": CACHE_PROTOCOL,
        "status": "prepared",
        "dataset_version": DATASET_VERSION,
        "source_manifest": str(Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv")),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_mapping_registry": str(Path("data/water_impact_dynamic_v4/source_mapping_v2.json")),
        "source_mapping_registry_sha256": expected_mapping_sha256,
        "active100_mapping_sha256": canonical_json_sha256(active_mapping),
        "full178_mapping_sha256": canonical_json_sha256(full_mapping),
        "canonical_prompt_builder_path": str(PROMPT_BUILDER_PATH),
        "canonical_prompt_builder_sha256": file_sha256(PROMPT_BUILDER_FILE),
        "source_bank_registry_sha256": expected_bank_sha256,
        "holdout_public_commitment_path": str(EXPECTED_HOLDOUT_PUBLIC_COMMITMENT),
        "holdout_public_commitment_sha256": expected_holdout_commitment_sha256,
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": str(
            EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT
        ),
        "causal_stage0_public_commitment_sha256": (
            expected_causal_stage0_public_commitment_sha256
        ),
        "base_cache_dir": str(Path("outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2")),
        "base_cache_inventory_sha256": EXPECTED_BASE_CACHE_SHA256,
        "teacher_cache_dir": str(Path("outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1")),
        "teacher_cache_inventory_sha256": EXPECTED_TEACHER_CACHE_SHA256,
        "model": str(Path("models/Wan2.1-T2V-1.3B-Diffusers")),
        "model_content_inventory_sha256": expected_model_inventory_sha256,
        "runtime_registry_path": str(EXPECTED_RUNTIME_REGISTRY),
        "runtime_registry_sha256": expected_runtime_registry_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "model_revision": EXPECTED_MODEL_REVISION,
        "model_artifact_inventory": expected_model_provenance[
            "model_artifact_inventory"
        ],
        "cache_inventory_sha256": expected_inventory_sha256,
        "erase_row_count": EXPECTED_ERASE_ROWS,
        "dtype": str(EXPECTED_PROMPT_DTYPE),
        "shape": list(EXPECTED_PROMPT_SHAPE),
        "max_sequence_length": 226,
        "truncation_allowed": False,
        "unique_original_prompt_count": len(original_prompts),
        "unique_augmented_prompt_count": len(augmented_prompts),
        "runtime_versions": EXPECTED_SIDECAR_RUNTIME_VERSIONS,
    }
    for key, value in expected_manifest_values.items():
        if manifest.get(key) != value:
            raise ValueError(f"prompt sidecar manifest {key} mismatch")
    if manifest.get("files") != [path.name for path in expected]:
        raise ValueError("prompt sidecar manifest file order/inventory mismatch")
    require_sha256(
        manifest.get("original_reencode_binding_sha256"),
        "prompt sidecar original re-encode binding",
    )
    require_sha256(
        manifest.get("sidecar_record_binding_sha256"),
        "prompt sidecar record binding",
    )
    by_index: dict[int, Path] = {}
    sidecar_records: list[dict[str, Any]] = []
    for (index, row), path in zip(erase, expected):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        record = mapping_by_index[index]
        scalar_expected = {
            "protocol": CACHE_PROTOCOL,
            "dataset_version": DATASET_VERSION,
            "manifest_index": index,
            "scene_id": row["scene_id"],
            "training_role": "erase",
            "original_source_id": record["original_source_id"],
            "original_source_phrase": record["original_source_phrase"],
            "assigned_source_id": record["assigned_source_id"],
            "assigned_source_phrase": record["assigned_source_phrase"],
            "augmented_factual_prompt": record["augmented_factual_prompt"],
            "source_mapping_registry_sha256": expected_mapping_sha256,
            "source_bank_registry_sha256": expected_bank_sha256,
            "holdout_public_commitment_sha256": expected_holdout_commitment_sha256,
            "train_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "model_content_inventory_sha256": expected_model_inventory_sha256,
            "runtime_registry_sha256": expected_runtime_registry_sha256,
        }
        embedding, length = validate_prompt_sidecar_payload(
            payload, scalar_expected=scalar_expected, path=path
        )
        by_index[index] = path
        sidecar_records.append(
            {
                "file": path.name,
                "manifest_index": index,
                "scene_id": row["scene_id"],
                "assigned_source_id": record["assigned_source_id"],
                "prompt_sha256": hashlib.sha256(
                    record["augmented_factual_prompt"].encode("utf-8")
                ).hexdigest(),
                "embedding_sha256": tensor_sha256(embedding),
                "registered_token_length": length,
            }
        )
    if manifest.get("sidecar_record_binding_sha256") != canonical_json_sha256(
        sidecar_records
    ):
        raise ValueError("prompt sidecar manifest record binding mismatch")
    if manifest["max_registered_token_length"] != max(
        record["registered_token_length"] for record in sidecar_records
    ):
        raise ValueError("prompt sidecar manifest max registered token length mismatch")
    return by_index, manifest


def compute_noise_rng_digests(
    base_paths: list[Path], schedule: list[int], *, seed: int
) -> tuple[str, str]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = tensor_sha256(generator.get_state())
    for manifest_index in schedule:
        payload = torch.load(base_paths[manifest_index], map_location="cpu", weights_only=True)
        shape = tuple(payload["latents"].shape)
        torch.randn(shape, generator=generator, dtype=torch.float32)
        torch.rand((shape[0],), generator=generator, dtype=torch.float32)
    return initial, tensor_sha256(generator.get_state())


def _v3b_reference_forward_signature(
    transformer: WanTransformer3DModel,
    *,
    clean: torch.Tensor,
    factual_prompt_embeds: torch.Tensor,
    target_prompt_embeds: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> dict[str, Any]:
    """Execute the frozen v3b base-cache factual-conditioning path."""

    transformer.zero_grad(set_to_none=True)
    noisy = ((1.0 - sigma) * clean + sigma * noise).to(dtype=torch.bfloat16)
    target = (noise - clean).to(dtype=torch.bfloat16)
    timestep = (sigma.flatten() * 1000.0).to(dtype=torch.bfloat16)
    transformer.disable_adapters()
    try:
        with torch.no_grad():
            teacher = transformer(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=target_prompt_embeds,
                return_dict=False,
            )[0].detach()
    finally:
        transformer.enable_adapters()
    prediction = transformer(
        hidden_states=noisy,
        timestep=timestep,
        encoder_hidden_states=factual_prompt_embeds,
        return_dict=False,
    )[0]
    flow_loss = torch.nn.functional.mse_loss(
        prediction.float(), target.float(), reduction="none"
    ).mean()
    teacher_loss = torch.nn.functional.mse_loss(
        prediction.float(), teacher.float(), reduction="none"
    ).mean()
    combined_loss = flow_loss + 4.0 * teacher_loss
    combined_loss.backward()
    signature = {
        "prediction_sha256": tensor_sha256(prediction),
        "teacher_prediction_sha256": tensor_sha256(teacher),
        "flow_loss_sha256": tensor_sha256(flow_loss),
        "teacher_loss_sha256": tensor_sha256(teacher_loss),
        "combined_loss_sha256": tensor_sha256(combined_loss),
        "gradient_state_sha256": gradient_state_sha256(transformer),
    }
    transformer.zero_grad(set_to_none=True)
    return signature


def _v4_null_sidecar_forward_signature(
    transformer: WanTransformer3DModel,
    *,
    clean: torch.Tensor,
    null_sidecar: dict[str, torch.Tensor],
    target_prompt_embeds: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> dict[str, Any]:
    """Execute the v4 sidecar-loader path with an original-prompt null entry."""

    transformer.zero_grad(set_to_none=True)
    factual_prompt_embeds = null_sidecar["augmented_prompt_embeds"]
    noisy = ((1.0 - sigma) * clean + sigma * noise).to(dtype=torch.bfloat16)
    target = (noise - clean).to(dtype=torch.bfloat16)
    timestep = (sigma.flatten() * 1000.0).to(dtype=torch.bfloat16)
    transformer.disable_adapters()
    try:
        with torch.no_grad():
            teacher = transformer(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=target_prompt_embeds,
                return_dict=False,
            )[0].detach()
    finally:
        transformer.enable_adapters()
    prediction = transformer(
        hidden_states=noisy,
        timestep=timestep,
        encoder_hidden_states=factual_prompt_embeds,
        return_dict=False,
    )[0]
    flow_loss = torch.nn.functional.mse_loss(
        prediction.float(), target.float(), reduction="none"
    ).mean()
    teacher_loss = torch.nn.functional.mse_loss(
        prediction.float(), teacher.float(), reduction="none"
    ).mean()
    combined_loss = flow_loss + 4.0 * teacher_loss
    combined_loss.backward()
    signature = {
        "prediction_sha256": tensor_sha256(prediction),
        "teacher_prediction_sha256": tensor_sha256(teacher),
        "flow_loss_sha256": tensor_sha256(flow_loss),
        "teacher_loss_sha256": tensor_sha256(teacher_loss),
        "combined_loss_sha256": tensor_sha256(combined_loss),
        "gradient_state_sha256": gradient_state_sha256(transformer),
    }
    transformer.zero_grad(set_to_none=True)
    return signature


def run_null_preflight(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite null-sidecar preflight: {args.output}")
    validate_runtime_registry(
        args.runtime_registry,
        args.runtime_registry_sha256,
        project_root=Path("."),
        verify_current_runtime=True,
    )
    rows = load_frozen_rows(args.manifest, expected_sha256=args.manifest_sha256)
    bank, _ = validate_public_bank_registry(
        args.source_bank_registry,
        expected_sha256=args.source_bank_registry_sha256,
    )
    holdout, _ = validate_public_holdout_commitment(
        args.holdout_public_commitment,
        expected_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    validate_public_stage0_commitment(
        args.causal_stage0_public_commitment,
        expected_sha256=args.causal_stage0_public_commitment_sha256,
        bank_registry=bank,
        holdout_commitment=holdout,
    )
    mapping, by_index = load_mapping_registry(
        args.source_mapping_registry,
        rows,
        expected_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_path=str(args.holdout_public_commitment),
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    base_paths = validate_cache_inventory(
        rows, args.base_cache_dir, expected_sha256=args.base_cache_sha256
    )
    teacher_paths_list = validate_cache_inventory(
        rows,
        args.teacher_cache_dir,
        expected_sha256=args.teacher_cache_sha256,
        role="erase",
    )
    teacher_paths = {
        index: path
        for (index, row), path in zip(
            ((i, row) for i, row in enumerate(rows) if row["training_role"] == "erase"),
            teacher_paths_list,
        )
    }
    model_provenance = validate_model_content_inventory(
        args.model, args.model_content_inventory_sha256
    )
    sidecar_paths, sidecar_manifest = validate_prompt_sidecar(
        rows,
        by_index,
        args.prompt_sidecar_dir,
        expected_inventory_sha256=args.prompt_sidecar_inventory_sha256,
        expected_manifest_sha256=args.prompt_sidecar_manifest_sha256,
        expected_mapping_sha256=args.source_mapping_registry_sha256,
        expected_bank_sha256=args.source_bank_registry_sha256,
        expected_holdout_commitment_sha256=args.holdout_public_commitment_sha256,
        expected_causal_stage0_public_commitment_sha256=(
            args.causal_stage0_public_commitment_sha256
        ),
        expected_model_inventory_sha256=args.model_content_inventory_sha256,
        expected_runtime_registry_sha256=args.runtime_registry_sha256,
        expected_model_provenance=model_provenance,
    )
    schedule = balanced_v3b_schedule(rows, seed=EXPECTED_SEED, steps=200)
    order_hash = sample_order_sha256(rows, schedule)
    if order_hash != EXPECTED_SAMPLE_ORDER_SHA256:
        raise ValueError("null-sidecar preflight sample-order digest differs from v3b")
    rng_initial, rng_final = compute_noise_rng_digests(
        base_paths, schedule, seed=EXPECTED_SEED
    )
    if rng_initial != EXPECTED_NOISE_RNG_INITIAL_SHA256:
        raise ValueError(f"initial noise/sigma RNG digest drift: {rng_initial}")
    if rng_final != EXPECTED_NOISE_RNG_FINAL_SHA256:
        raise ValueError(f"final noise/sigma RNG digest drift: {rng_final}")

    # Fresh text-encoder process check: all original embeddings must still be
    # byte-identical to the frozen v3b base cache.
    device = torch.device(args.device)
    pipe = WanPipeline.from_pretrained(
        str(args.model), transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.text_encoder.eval()
    original_prompts = list(
        dict.fromkeys(row["prompt"] for row in rows if row["training_role"] == "erase")
    )
    fresh_original = encode_prompts(
        pipe, original_prompts, device=device, label="null-preflight original factual"
    )
    reencode_records = _validate_original_reencodes(rows, base_paths, fresh_original)
    augmented_prompts = list(
        dict.fromkeys(
            by_index[index]["augmented_factual_prompt"]
            for index, row in enumerate(rows)
            if row["training_role"] == "erase"
        )
    )
    fresh_augmented = encode_prompts(
        pipe,
        augmented_prompts,
        device=device,
        label="null-preflight augmented factual",
    )
    augmented_reencode_records = _validate_augmented_reencodes(
        rows, by_index, sidecar_paths, fresh_augmented
    )
    del pipe
    clear_memory()

    first_erase_index = next(index for index in schedule if rows[index]["training_role"] == "erase")
    base_payload = torch.load(base_paths[first_erase_index], map_location="cpu", weights_only=True)
    teacher_payload = torch.load(
        teacher_paths[first_erase_index], map_location="cpu", weights_only=True
    )
    base_prompt = base_payload["prompt_embeds"]
    null_prompt = fresh_original[rows[first_erase_index]["prompt"]]
    if not torch.equal(base_prompt, null_prompt):
        raise ValueError("selected null-sidecar embedding is not byte-identical to base prompt")

    torch.manual_seed(EXPECTED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXPECTED_SEED)
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    torch.manual_seed(EXPECTED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXPECTED_SEED)
    transformer.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    initial_lora = trainable_state_sha256(transformer)
    if initial_lora != EXPECTED_INITIAL_LORA_SHA256:
        raise ValueError(f"null-preflight initial LoRA digest drift: {initial_lora}")
    transformer.train()
    clean = base_payload["latents"].to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(EXPECTED_SEED)
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    sigma = torch.rand((clean.shape[0],), generator=generator, dtype=torch.float32).to(device)
    sigma = sigma.view(-1, 1, 1, 1, 1)
    target_prompt = teacher_payload["teacher_prompt_embeds"].to(
        device=device, dtype=torch.bfloat16
    )
    initial_trainable_state = {
        name: parameter.detach().clone()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    v3b_reference_signature = _v3b_reference_forward_signature(
        transformer,
        clean=clean,
        factual_prompt_embeds=base_prompt.to(device=device, dtype=torch.bfloat16),
        target_prompt_embeds=target_prompt,
        noise=noise,
        sigma=sigma,
    )
    with torch.no_grad():
        for name, parameter in transformer.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(initial_trainable_state[name])
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    v4_null_sidecar = {
        "augmented_prompt_embeds": null_prompt.to(
            device=device, dtype=torch.bfloat16
        )
    }
    v4_null_sidecar_signature = _v4_null_sidecar_forward_signature(
        transformer,
        clean=clean,
        null_sidecar=v4_null_sidecar,
        target_prompt_embeds=target_prompt,
        noise=noise,
        sigma=sigma,
    )
    if v3b_reference_signature != v4_null_sidecar_signature:
        raise ValueError(
            "null-sidecar forward/loss/LoRA-gradient signature differs from v3b base prompt"
        )
    del transformer, clean, noise, sigma, target_prompt, initial_trainable_state
    clear_memory()

    artifact = {
        "protocol": PREFLIGHT_PROTOCOL,
        "status": "passed",
        "dataset_version": DATASET_VERSION,
        "train_manifest_sha256": args.manifest_sha256,
        "source_bank_registry_sha256": args.source_bank_registry_sha256,
        "source_bank_registry_path": str(args.source_bank_registry),
        "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
        "holdout_public_commitment_path": str(args.holdout_public_commitment),
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": str(
            args.causal_stage0_public_commitment
        ),
        "causal_stage0_public_commitment_sha256": (
            args.causal_stage0_public_commitment_sha256
        ),
        "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
        "source_mapping_registry_path": str(args.source_mapping_registry),
        "active100_mapping_sha256": mapping["active100_mapping_sha256"],
        "full178_mapping_sha256": mapping["full178_mapping_sha256"],
        "canonical_prompt_builder_path": mapping["canonical_prompt_builder_path"],
        "canonical_prompt_builder_sha256": mapping[
            "canonical_prompt_builder_sha256"
        ],
        "base_cache_inventory_sha256": args.base_cache_sha256,
        "teacher_cache_inventory_sha256": args.teacher_cache_sha256,
        "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
        "prompt_sidecar_manifest_sha256": args.prompt_sidecar_manifest_sha256,
        "preparer_sha256": file_sha256(Path(__file__)),
        "model_content_inventory_sha256": args.model_content_inventory_sha256,
        "runtime_registry_path": str(args.runtime_registry),
        "runtime_registry_sha256": args.runtime_registry_sha256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        **model_provenance,
        "seed": EXPECTED_SEED,
        "sample_order_sha256": order_hash,
        "noise_sigma_rng_initial_sha256": rng_initial,
        "noise_sigma_rng_final_sha256": rng_final,
        "initial_lora_sha256": initial_lora,
        "original_reencode_count": len(reencode_records),
        "original_reencode_binding_sha256": canonical_json_sha256(reencode_records),
        "unique_augmented_reencode_count": len(augmented_prompts),
        "augmented_reencode_row_count": len(augmented_reencode_records),
        "augmented_reencode_binding_sha256": canonical_json_sha256(
            augmented_reencode_records
        ),
        "augmented_reencode_all_rows_byte_equal": True,
        "tokenizer_binding": tokenizer_inventory_binding(model_provenance),
        "integration_manifest_index": first_erase_index,
        "integration_scene_id": rows[first_erase_index]["scene_id"],
        "v3b_reference_path": "frozen_base_cache_prompt_embeds",
        "v4_null_sidecar_path": (
            "v4_sidecar_loader_with_fresh_original_augmented_prompt_embeds"
        ),
        "null_sidecar_substitution": "fresh_original_embedding_for_augmented_embedding",
        "forward_loss_gradient_equal": True,
        "rng_restored_between_signatures": True,
        "trainable_state_restored_between_signatures": True,
        "v3b_reference_signature": v3b_reference_signature,
        "v4_null_sidecar_signature": v4_null_sidecar_signature,
        "optimizer_created": False,
    }
    atomic_write_new_json(args.output, artifact)
    print(
        f"Null-sidecar preflight passed; artifact={args.output} "
        f"sha256={file_sha256(args.output)}",
        flush=True,
    )
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"),
    )
    parser.add_argument("--manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument(
        "--model", type=Path, default=Path("models/Wan2.1-T2V-1.3B-Diffusers")
    )
    parser.add_argument("--model-content-inventory-sha256", required=True)
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"),
    )
    parser.add_argument("--base-cache-sha256", default=EXPECTED_BASE_CACHE_SHA256)
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"),
    )
    parser.add_argument("--teacher-cache-sha256", default=EXPECTED_TEACHER_CACHE_SHA256)
    parser.add_argument(
        "--source-mapping-registry",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/source_mapping_v2.json"),
    )
    parser.add_argument("--source-mapping-registry-sha256", required=True)
    parser.add_argument(
        "--source-bank-registry",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json"),
    )
    parser.add_argument("--source-bank-registry-sha256", required=True)
    parser.add_argument(
        "--holdout-public-commitment",
        type=Path,
        default=EXPECTED_HOLDOUT_PUBLIC_COMMITMENT,
    )
    parser.add_argument("--holdout-public-commitment-sha256", required=True)
    parser.add_argument(
        "--causal-stage0-public-commitment",
        type=Path,
        default=EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT,
    )
    parser.add_argument(
        "--causal-stage0-public-commitment-sha256",
        default=EXPECTED_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256,
    )
    parser.add_argument(
        "--runtime-registry",
        type=Path,
        default=EXPECTED_RUNTIME_REGISTRY,
    )
    parser.add_argument("--runtime-registry-sha256", required=True)
    parser.add_argument("--device", default="cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-cache")
    add_common_arguments(prepare)
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2"),
    )
    preflight = commands.add_parser("null-preflight")
    add_common_arguments(preflight)
    preflight.add_argument(
        "--prompt-sidecar-dir",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2"),
    )
    preflight.add_argument("--prompt-sidecar-inventory-sha256", required=True)
    preflight.add_argument("--prompt-sidecar-manifest-sha256", required=True)
    preflight.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v4/null_sidecar_preflight_v2.json"),
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if name.endswith("sha256"):
            require_sha256(value, name.replace("_", " "))
    if args.device != "cuda":
        parser.error("v4 prompt preparation/preflight requires the registered cuda device")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "prepare-cache":
        return prepare_cache(args)
    if args.command == "null-preflight":
        return run_null_preflight(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
