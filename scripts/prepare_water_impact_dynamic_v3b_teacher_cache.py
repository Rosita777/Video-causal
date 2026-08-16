#!/usr/bin/env python3
"""Build the frozen target-prompt embedding sidecar for water-impact v3b."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from diffusers import WanPipeline


EXPECTED_MANIFEST_SHA256 = "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
EXPECTED_PROMPT_BINDING_SHA256 = "9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc"
EXPECTED_ERASE_ROWS = 178
EXPECTED_UNIQUE_PROMPTS = 24
EXPECTED_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_binding_sha256(rows: list[tuple[int, dict[str, str]]]) -> str:
    digest = hashlib.sha256()
    for _, row in rows:
        digest.update(row["scene_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["target_generation_prompt"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_inventory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def encode_unique_prompts(
    model: Path, prompts: list[str], device: torch.device
) -> dict[str, torch.Tensor]:
    pipe = WanPipeline.from_pretrained(
        str(model), transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.text_encoder.eval()
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
        embeddings[prompt] = prompt_embeds.detach().contiguous().cpu()
        print(f"Encoded target prompt {index}/{len(prompts)}", flush=True)
    del pipe
    clear_memory()
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_hash = file_sha256(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        parser.error(f"frozen manifest hash mismatch: {manifest_hash}")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    erase_rows = [
        (index, row)
        for index, row in enumerate(all_rows)
        if row["training_role"] == "erase"
    ]
    if len(erase_rows) != EXPECTED_ERASE_ROWS:
        parser.error(f"expected {EXPECTED_ERASE_ROWS} erase rows, found {len(erase_rows)}")
    if any(not row.get("target_generation_prompt", "").strip() for _, row in erase_rows):
        parser.error("every erase row must have a non-empty target_generation_prompt")
    binding_hash = prompt_binding_sha256(erase_rows)
    if binding_hash != EXPECTED_PROMPT_BINDING_SHA256:
        parser.error(f"frozen target-prompt binding mismatch: {binding_hash}")

    revision_path = (
        args.model
        / ".cache/huggingface/download/model_index.json.metadata"
    )
    try:
        model_revision = revision_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError) as exc:
        parser.error(f"cannot verify frozen model revision from {revision_path}: {exc}")
    if model_revision != EXPECTED_MODEL_REVISION:
        parser.error(
            f"frozen model revision mismatch: {model_revision} != {EXPECTED_MODEL_REVISION}"
        )

    unique_prompts = list(dict.fromkeys(row["target_generation_prompt"] for _, row in erase_rows))
    if len(unique_prompts) != EXPECTED_UNIQUE_PROMPTS:
        parser.error(
            f"expected {EXPECTED_UNIQUE_PROMPTS} unique target prompts, found {len(unique_prompts)}"
        )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"refusing to reuse teacher cache directory: {args.output_dir}")
    (args.output_dir / ".run_reservation").write_text(
        "teacher cache construction in progress\n", encoding="utf-8"
    )

    embeddings = encode_unique_prompts(args.model, unique_prompts, torch.device(args.device))
    cache_paths: list[Path] = []
    for manifest_index, row in erase_rows:
        cache_path = args.output_dir / f"{manifest_index:03d}_{row['scene_id']}.pt"
        embedding = embeddings[row["target_generation_prompt"]]
        torch.save(
            {
                "manifest_index": manifest_index,
                "scene_id": row["scene_id"],
                "training_role": "erase",
                "target_generation_prompt": row["target_generation_prompt"],
                "teacher_prompt_embeds": embedding,
                "teacher_prompt_embeds_sha256": tensor_sha256(embedding),
                "model": str(args.model),
                "manifest_sha256": manifest_hash,
            },
            cache_path,
        )
        cache_paths.append(cache_path)
    inventory_hash = cache_inventory_sha256(cache_paths)
    embedding_digest = hashlib.sha256()
    for prompt in unique_prompts:
        embedding_digest.update(prompt.encode("utf-8"))
        embedding_digest.update(b"\0")
        embedding_digest.update(tensor_sha256(embeddings[prompt]).encode("ascii"))
        embedding_digest.update(b"\n")
    cache_manifest = {
        "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_v1",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": manifest_hash,
        "model": str(args.model),
        "device": args.device,
        "dtype": "torch.bfloat16",
        "do_classifier_free_guidance": False,
        "max_sequence_length": 226,
        "model_revision": model_revision,
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            )
        },
        "erase_row_count": len(erase_rows),
        "unique_prompt_count": len(unique_prompts),
        "prompt_binding_sha256": binding_hash,
        "unique_embedding_sha256": embedding_digest.hexdigest(),
        "cache_inventory_sha256": inventory_hash,
        "files": [path.name for path in cache_paths],
    }
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / ".run_reservation").unlink()
    print(
        f"Prepared {len(cache_paths)} teacher embeddings from {len(unique_prompts)} "
        f"unique prompts; SHA-256={inventory_hash}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
