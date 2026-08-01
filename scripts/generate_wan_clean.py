#!/usr/bin/env python3
"""Generate or plan Wan clean and negative-prompt videos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import resolve_torch_dtype, slugify  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


DEFAULT_MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_OUTPUT_DIR = Path("outputs/wan_clean")
PIPELINE_NAME = "WanPipeline"


def build_generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "baseline": args.baseline,
        "seed": args.seed,
        "seeds": args.seeds,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "device": args.device,
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "enable_sequential_cpu_offload": args.enable_sequential_cpu_offload,
        "vae_slicing": args.vae_slicing,
        "vae_tiling": args.vae_tiling,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "lora_scale": args.lora_scale,
    }


def build_manifest_items(
    prompts: list[dict[str, str]],
    output_dir: Path,
    base_seed: int,
    limit: int | None,
    baseline: str,
    explicit_seeds: list[int] | None = None,
) -> list[dict[str, object]]:
    selected = prompts[:limit] if limit is not None else prompts
    items: list[dict[str, object]] = []
    video_dir = output_dir / "videos"
    for index, item in enumerate(selected):
        seed = explicit_seeds[index] if explicit_seeds is not None else base_seed + index
        prompt_slug = slugify(item["prompt"])
        manifest_item: dict[str, object] = {
            "index": index,
            "prompt": item["prompt"],
            "target_concept": item["target_concept"],
            "expected_effect": item["expected_effect"],
            "seed": seed,
            "video_path": str(video_dir / f"{index:03d}_{prompt_slug}_seed{seed}.mp4"),
        }
        if baseline == "negative_prompt":
            manifest_item["negative_prompt"] = item["target_concept"]
        items.append(manifest_item)
    return items


def parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def write_manifest(
    *,
    output_dir: Path,
    baseline: str,
    model: str,
    prompts_path: Path,
    generation: dict[str, object],
    items: list[dict[str, object]],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "pipeline": PIPELINE_NAME,
        "model": model,
        "dry_run": dry_run,
        "prompts": str(prompts_path),
        "generation": generation,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def output_frames(result: object) -> object:
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, tuple):
        frames = result[0]
    if frames is None:
        raise RuntimeError("WanPipeline result does not contain frames")
    return frames[0]


def select_prompt_encode_device(args: argparse.Namespace, *, selected_device: str, cuda_available: bool) -> str:
    if selected_device.startswith("cuda") and cuda_available:
        if args.enable_sequential_cpu_offload or args.enable_model_cpu_offload:
            return "cpu"
    return selected_device


def move_prompt_embeds(torch_module, embeds, negative_embeds, *, device: str, dtype):
    prompt_embeds = embeds.to(device=torch_module.device(device), dtype=dtype)
    negative_prompt_embeds = None
    if negative_embeds is not None:
        negative_prompt_embeds = negative_embeds.to(device=torch_module.device(device), dtype=dtype)
    return prompt_embeds, negative_prompt_embeds


def selected_generation_device(args: argparse.Namespace, torch_module) -> str:
    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch_module.cuda.is_available() else "cpu"
    if (args.enable_sequential_cpu_offload or args.enable_model_cpu_offload) and torch_module.cuda.is_available():
        selected_device = "cuda"
    return selected_device


def apply_device_strategy(args: argparse.Namespace, pipe, selected_device: str) -> None:
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(selected_device)


def generate_videos(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    try:
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise SystemExit(
            "Wan generation requires torch and diffusers with WanPipeline. "
            "Install heavy generation dependencies before running without --dry-run."
        ) from exc

    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    if args.lora_path is not None:
        pipe.load_lora_weights(str(args.lora_path), adapter_name="waterdrop")
        pipe.set_adapters("waterdrop", adapter_weights=args.lora_scale)

    if args.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    selected_device = selected_generation_device(args, torch)
    encode_device = select_prompt_encode_device(
        args,
        selected_device=selected_device,
        cuda_available=torch.cuda.is_available(),
    )
    encoded_items: dict[int, tuple[object, object]] = {}
    if encode_device != selected_device:
        for item in items:
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=str(item["prompt"]),
                negative_prompt=item.get("negative_prompt"),
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                num_videos_per_prompt=1,
                device=torch.device(encode_device),
                dtype=torch_dtype,
            )
            encoded_items[int(item["index"])] = (prompt_embeds, negative_prompt_embeds)

    apply_device_strategy(args, pipe, selected_device)

    for item in items:
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator_device = "cuda" if selected_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(item["seed"]))
        prompt = str(item["prompt"])
        negative_prompt = item.get("negative_prompt")
        prompt_embeds = None
        negative_prompt_embeds = None
        if int(item["index"]) in encoded_items:
            prompt_embeds, negative_prompt_embeds = encoded_items[int(item["index"])]
            prompt_embeds, negative_prompt_embeds = move_prompt_embeds(
                torch,
                prompt_embeds,
                negative_prompt_embeds,
                device=selected_device if selected_device.startswith("cuda") else "cpu",
                dtype=torch_dtype,
            )
            prompt = None
            negative_prompt = None
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        export_to_video(output_frames(result), str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--baseline", choices=["clean", "negative_prompt"], default="clean")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        help="Comma-separated per-prompt seeds; overrides --seed when provided",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.lora_scale < 0:
        parser.error("--lora-scale must be non-negative")
    if args.lora_path is not None and not args.dry_run and not args.lora_path.exists():
        parser.error(f"--lora-path does not exist: {args.lora_path}")

    prompts = parse_prompt_file(args.prompts)
    selected_count = len(prompts[: args.limit] if args.limit is not None else prompts)
    if args.seeds is not None and len(args.seeds) != selected_count:
        parser.error(f"--seeds contains {len(args.seeds)} values but {selected_count} prompts were selected")
    generation = build_generation_config(args)
    items = build_manifest_items(
        prompts,
        args.output_dir,
        args.seed,
        args.limit,
        args.baseline,
        explicit_seeds=args.seeds,
    )

    if not args.dry_run:
        generate_videos(args, items)

    manifest = write_manifest(
        output_dir=args.output_dir,
        baseline=args.baseline,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation,
        items=items,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry-run Wan manifest written: {manifest}")
    else:
        print(f"Wan generation manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
