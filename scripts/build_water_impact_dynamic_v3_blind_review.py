#!/usr/bin/env python3
"""Build a blinded, paired review package for the seeded v3 sampling ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw


FRAME_INDICES = (0, 8, 16, 24, 32, 40, 48)
METHODS = ("balanced", "exposure")
EXPECTED_EVAL_SHA256 = "dca68f8632e10ef83cc5f3867679c9cba54f4cbce96426db5db8c5214ac1ec1a"
EXPECTED_TRAIN_SHA256 = "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
EXPECTED_MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
EXPECTED_PROMPTS = "prompts/water_impact_dynamic_v1/eval12.prompts"
EXPECTED_PROMPTS_SHA256 = "06dae57a0202e2d53e32fc02f9b26fd694237755a18f85bdd67c728bf706681c"
EXPECTED_TRAIN_MANIFEST = "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
EXPECTED_CACHE_DIR = "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
EXPECTED_CACHE_SHA256 = "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
SCORE_FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_inventory_sha256(cache_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in cache_paths:
        resolved = path.resolve(strict=True)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def review_binding_sha256(rows: list[dict[str, object]]) -> str:
    fields = (
        "review_id",
        "sample_index",
        "pair_id",
        "generalization_group",
        "candidate_code",
        "composite_path",
        "source_object",
        "receiver",
    )
    canonical = [
        {field: str(row[field]) for field in fields}
        for row in sorted(rows, key=lambda item: str(item["review_id"]))
    ]
    return sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_run(
    run_dir: Path,
    eval_rows: list[dict[str, str]],
    label: str,
) -> tuple[Path, dict[str, object], dict[int, Path]]:
    manifest_path = run_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dry_run"):
        raise ValueError(f"{label}: generation manifest is a dry run")
    items = manifest.get("items", [])
    if len(items) != len(eval_rows):
        raise ValueError(f"{label}: expected {len(eval_rows)} items, found {len(items)}")
    videos: dict[int, Path] = {}
    expected_video_root = (run_dir / "videos").resolve(strict=True)
    for item in items:
        index = int(item["index"])
        if index in videos or not 0 <= index < len(eval_rows):
            raise ValueError(f"{label}: invalid or duplicate item index {index}")
        expected = eval_rows[index]
        if int(item["seed"]) != int(expected["seed"]):
            raise ValueError(f"{label}: seed mismatch at index {index}")
        if item["prompt"] != expected["training_prompt"]:
            raise ValueError(f"{label}: prompt mismatch at index {index}")
        video = Path(str(item["video_path"]))
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f"{label}: missing video at index {index}: {video}")
        try:
            video.resolve(strict=True).relative_to(expected_video_root)
        except ValueError as exc:
            raise ValueError(f"{label}: video escapes its run directory: {video}") from exc
        videos[index] = video
    return manifest_path, manifest, videos


def expected_training_schedule(
    train_rows: list[dict[str, str]], *, balanced: bool, seed: int = 26000
) -> tuple[dict[str, int], str]:
    rng = random.Random(seed)
    counts = {"erase": 0, "preserve": 0}
    digest = hashlib.sha256()
    if balanced:
        role_indices = {
            role: [index for index, row in enumerate(train_rows) if row["training_role"] == role]
            for role in ("erase", "preserve")
        }
        for role in role_indices:
            rng.shuffle(role_indices[role])
        cursors = {"erase": 0, "preserve": 0}
        sample_indices: list[int] = []
        for step in range(1, 201):
            role = "erase" if step % 2 else "preserve"
            cursor = cursors[role]
            if cursor >= len(role_indices[role]):
                rng.shuffle(role_indices[role])
                cursor = 0
            sample_indices.append(role_indices[role][cursor])
            cursors[role] = cursor + 1
    else:
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        sample_indices = order[:200]
    for step, sample_index in enumerate(sample_indices, start=1):
        row = train_rows[sample_index]
        role = row["training_role"]
        counts[role] += 1
        digest.update(f"{step}:{role}:{row['scene_id']}\n".encode("utf-8"))
    return counts, digest.hexdigest()


def validate_training_provenance(
    label: str,
    generation_manifest: dict[str, object],
    train_rows: list[dict[str, str]],
    cache_dir: Path,
    expected_cache_sha256: str,
) -> dict[str, object]:
    generation = generation_manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: missing generation config")
    lora_value = generation.get("lora_path")
    if not isinstance(lora_value, str):
        raise ValueError(f"{label}: missing LoRA path")
    lora_path = Path(lora_value)
    if not lora_path.is_dir():
        raise FileNotFoundError(f"{label}: missing LoRA checkpoint: {lora_path}")
    recorded_lora_hash = generation.get("lora_sha256")
    actual_lora_hash = artifact_sha256(lora_path)
    if recorded_lora_hash != actual_lora_hash:
        raise ValueError(f"{label}: LoRA artifact hash mismatch")
    state_path = lora_path / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_balanced = label == "balanced"
    expected_counts, expected_order_hash = expected_training_schedule(
        train_rows, balanced=expected_balanced
    )
    expected = {
        "step": 200,
        "max_steps": 200,
        "manifest": EXPECTED_TRAIN_MANIFEST,
        "manifest_sha256": EXPECTED_TRAIN_SHA256,
        "model": EXPECTED_MODEL,
        "cache_dir": str(cache_dir),
        "cache_entry_count": len(train_rows),
        "height": 480,
        "width": 832,
        "num_frames": 49,
        "grad_accum": 1,
        "device": "cuda",
        "rank": 16,
        "alpha": 16,
        "learning_rate": 5e-5,
        "seed": 26000,
        "role": "all",
        "objective": "plain",
        "mask_weight": 4.0,
        "background_weight": 1.0,
        "pair_weight": 1.0,
        "pair_margin": 0.05,
        "redirect_weight": 1.0,
        "object_weight": 1.0,
        "receiver_weight": 1.0,
        "preserve_weight": 4.0,
        "balanced_roles": expected_balanced,
        "role_step_counts": expected_counts,
        "sample_order_sha256": expected_order_hash,
        "causal_gate_dir": None,
        "gate_floor": 0.0,
        "activation_gate_dir": None,
        "component_gate_dir": None,
        "target_phrase": [],
        "persistent_causal_time": False,
    }
    cache_paths = [
        cache_dir / f"{index:03d}_{row['scene_id']}.pt"
        for index, row in enumerate(train_rows)
    ]
    current_cache_sha256 = cache_inventory_sha256(cache_paths)
    if current_cache_sha256 != expected_cache_sha256:
        raise ValueError(f"{label}: frozen cache content hash mismatch")
    expected["cache_inventory_sha256"] = expected_cache_sha256
    for field, expected_value in expected.items():
        if state.get(field) != expected_value:
            raise ValueError(
                f"{label}: training_state {field}={state.get(field)!r}, expected {expected_value!r}"
            )
    initial_hash = state.get("initial_lora_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError(f"{label}: invalid initial LoRA hash")
    return {
        "path": str(state_path),
        "sha256": file_sha256(state_path),
        "initial_lora_sha256": initial_hash,
        "role_step_counts": expected_counts,
        "sample_order_sha256": expected_order_hash,
        "cache_inventory_sha256": state["cache_inventory_sha256"],
        "lora_sha256": actual_lora_hash,
    }


def validate_generation_config(
    label: str, manifest: dict[str, object], eval_rows: list[dict[str, str]]
) -> None:
    expected_top = {
        "baseline": "clean",
        "pipeline": "WanPipeline",
        "model": EXPECTED_MODEL,
        "dry_run": False,
        "prompts": EXPECTED_PROMPTS,
    }
    for field, expected_value in expected_top.items():
        if manifest.get(field) != expected_value:
            raise ValueError(f"{label}: manifest {field} mismatch")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError(f"{label}: missing generation config")
    expected_generation = {
        "baseline": "clean",
        "seeds": [int(row["seed"]) for row in eval_rows],
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "activation_gate_dir": None,
        "persistent_activation_gate": False,
        "lora_target_phrases": [],
        "attention_gate_dir": None,
        "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
    }
    for field, expected_value in expected_generation.items():
        if generation.get(field) != expected_value:
            raise ValueError(f"{label}: generation {field} mismatch")
    expected_scale = 1.0 if label == "original" else 1.25
    if generation.get("lora_scale") != expected_scale:
        raise ValueError(f"{label}: generation lora_scale mismatch")
    if label == "original" and generation.get("lora_path") is not None:
        raise ValueError("original: unexpected LoRA path")


def load_frames(path: Path) -> list[Image.Image]:
    import av

    selected: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in FRAME_INDICES:
                selected[index] = frame.to_image().convert("RGB")
            if index >= FRAME_INDICES[-1]:
                break
    missing = [index for index in FRAME_INDICES if index not in selected]
    if missing:
        raise ValueError(f"Missing frames {missing}: {path}")
    return [selected[index] for index in FRAME_INDICES]


def make_strip(frames: list[Image.Image], label: str) -> Image.Image:
    frame_width = 208
    frame_height = round(frames[0].height * frame_width / frames[0].width)
    label_width = 120
    header_height = 24
    strip = Image.new(
        "RGB",
        (label_width + frame_width * len(frames), header_height + frame_height),
        "white",
    )
    draw = ImageDraw.Draw(strip)
    draw.text((8, header_height + 10), label, fill="black")
    for column, (frame_index, frame) in enumerate(
        zip(FRAME_INDICES, frames, strict=True)
    ):
        x = label_width + column * frame_width
        draw.text((x + 5, 5), f"frame {frame_index}", fill="black")
        strip.paste(frame.resize((frame_width, frame_height)), (x, header_height))
    return strip


def build_composite(
    output_path: Path,
    pair_id: str,
    generalization_group: str,
    ordered_methods: list[str],
    paths: dict[str, Path],
) -> None:
    rows = [("Original reference", load_frames(paths["original"]))]
    rows.extend(
        (f"Candidate {chr(ord('A') + index)}", load_frames(paths[method]))
        for index, method in enumerate(ordered_methods)
    )
    strips = [make_strip(frames, label) for label, frames in rows]
    title_height = 42
    composite = Image.new(
        "RGB",
        (strips[0].width, title_height + sum(strip.height for strip in strips)),
        "white",
    )
    draw = ImageDraw.Draw(composite)
    draw.text((8, 7), pair_id, fill="black")
    draw.text((8, 23), f"Generalization: {generalization_group}", fill="black")
    y = title_height
    for strip in strips:
        composite.paste(strip, (0, y))
        y += strip.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/eval12.csv"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(EXPECTED_TRAIN_MANIFEST),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(EXPECTED_CACHE_DIR))
    parser.add_argument(
        "--expected-cache-sha256",
        default=EXPECTED_CACHE_SHA256,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--balanced-dir", type=Path, required=True)
    parser.add_argument("--exposure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=26013)
    parser.add_argument("--skip-frame-extraction", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"refusing to overwrite review directory: {args.output_dir}")
    if file_sha256(args.eval_csv) != EXPECTED_EVAL_SHA256:
        parser.error("frozen eval12 manifest hash mismatch")
    if file_sha256(args.train_manifest) != EXPECTED_TRAIN_SHA256:
        parser.error("frozen training manifest hash mismatch")
    if file_sha256(Path(EXPECTED_PROMPTS)) != EXPECTED_PROMPTS_SHA256:
        parser.error("frozen eval12 prompt file hash mismatch")
    eval_rows = read_csv(args.eval_csv)
    train_rows = read_csv(args.train_manifest)
    if len(eval_rows) != 12:
        parser.error(f"expected 12 eval rows, found {len(eval_rows)}")
    run_dirs = {
        "original": args.original_dir,
        "balanced": args.balanced_dir,
        "exposure": args.exposure_dir,
    }
    manifest_paths: dict[str, Path] = {}
    manifests: dict[str, dict[str, object]] = {}
    videos: dict[str, dict[int, Path]] = {}
    for label, directory in run_dirs.items():
        manifest_paths[label], manifests[label], videos[label] = load_run(
            directory, eval_rows, label
        )
        validate_generation_config(label, manifests[label], eval_rows)
    training_provenance = {
        label: validate_training_provenance(
            label,
            manifests[label],
            train_rows,
            args.cache_dir,
            args.expected_cache_sha256,
        )
        for label in METHODS
    }
    if (
        training_provenance["balanced"]["initial_lora_sha256"]
        != training_provenance["exposure"]["initial_lora_sha256"]
    ):
        raise ValueError("controlled arms have different initial LoRA hashes")
    balanced_paths = {path.resolve(strict=True) for path in videos["balanced"].values()}
    exposure_paths = {path.resolve(strict=True) for path in videos["exposure"].values()}
    if balanced_paths & exposure_paths:
        raise ValueError("controlled arms reference overlapping video paths")

    args.output_dir.mkdir(parents=True)
    sample_order = list(range(len(eval_rows)))
    random.Random(args.blind_seed).shuffle(sample_order)
    review_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for review_position, sample_index in enumerate(sample_order):
        sample = eval_rows[sample_index]
        pair_id = sample["pair_id"]
        ordered_methods = list(METHODS)
        random.Random(f"{args.blind_seed}:{pair_id}").shuffle(ordered_methods)
        composite_path = args.output_dir / "composites" / f"r{review_position:03d}.jpg"
        sample_paths = {label: paths[sample_index] for label, paths in videos.items()}
        if not args.skip_frame_extraction:
            build_composite(
                composite_path,
                pair_id,
                sample["generalization_group"],
                ordered_methods,
                sample_paths,
            )
        for candidate_index, method in enumerate(ordered_methods):
            candidate_code = chr(ord("A") + candidate_index)
            review_id = f"r{review_position:03d}_{candidate_code}"
            review_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": pair_id,
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": candidate_code,
                    "composite_path": str(composite_path),
                    "source_object": sample["source_object"],
                    "receiver": sample["receiver"],
                    **{field: "" for field in SCORE_FIELDS},
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_index": sample_index,
                    "pair_id": pair_id,
                    "generalization_group": sample["generalization_group"],
                    "candidate_code": candidate_code,
                    "method": method,
                    "video_path": str(sample_paths[method]),
                }
            )

    review_fields = list(review_rows[0])
    key_fields = list(key_rows[0])
    write_csv(args.output_dir / "blind_review.csv", review_rows, review_fields)
    answer_key_path = args.output_dir / "answer_key.csv"
    write_csv(
        answer_key_path,
        sorted(key_rows, key=lambda row: str(row["review_id"])),
        key_fields,
    )
    review_manifest = {
        "eval_csv": str(args.eval_csv),
        "eval_csv_sha256": file_sha256(args.eval_csv),
        "prompts_sha256": file_sha256(Path(EXPECTED_PROMPTS)),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "blind_seed": args.blind_seed,
        "frame_indices": list(FRAME_INDICES),
        "review_rows": len(review_rows),
        "sample_count": len(eval_rows),
        "answer_key_sha256": file_sha256(answer_key_path),
        "review_binding_sha256": review_binding_sha256(review_rows),
        "training_provenance": training_provenance,
        "video_sha256": {
            label: {
                str(index): file_content_sha256(path)
                for index, path in sorted(run_videos.items())
            }
            for label, run_videos in videos.items()
        },
        "composite_sha256": {
            path.name: file_sha256(path)
            for path in sorted((args.output_dir / "composites").glob("*.jpg"))
        },
        "generation_manifests": {
            label: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for label, path in manifest_paths.items()
        },
    }
    (args.output_dir / "review_manifest.json").write_text(
        json.dumps(review_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(review_rows)} blinded rows for {len(eval_rows)} eval samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
