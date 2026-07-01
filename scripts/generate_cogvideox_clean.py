#!/usr/bin/env python3
"""Generate or plan clean CogVideoX-2B causal-source videos."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from run_pilot import parse_prompt_file


DEFAULT_MODEL = "zai-org/CogVideoX-2b"
DEFAULT_PROMPTS = Path("prompts/cogvideox_causal_screening.txt")
DEFAULT_OUTPUT_DIR = Path("outputs/cogvideox_clean")


def slugify(text: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "prompt"
    return slug[:max_length].rstrip("-") or "prompt"


def build_generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
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
    }


def build_manifest_items(
    prompts: list[dict[str, str]],
    output_dir: Path,
    base_seed: int,
    limit: int | None,
) -> list[dict[str, object]]:
    selected = prompts[:limit] if limit is not None else prompts
    items: list[dict[str, object]] = []
    video_dir = output_dir / "videos"
    for index, item in enumerate(selected):
        seed = base_seed + index
        prompt_slug = slugify(item["prompt"])
        video_path = video_dir / f"{index:03d}_{prompt_slug}_seed{seed}.mp4"
        items.append(
            {
                "index": index,
                "prompt": item["prompt"],
                "target_concept": item["target_concept"],
                "expected_effect": item["expected_effect"],
                "seed": seed,
                "video_path": str(video_path),
            }
        )
    return items


def write_manifest(
    *,
    output_dir: Path,
    model: str,
    prompts_path: Path,
    generation: dict[str, object],
    items: list[dict[str, object]],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "clean",
        "model": model,
        "dry_run": dry_run,
        "prompts": str(prompts_path),
        "generation": generation,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def resolve_torch_dtype(torch_module, dtype: str):
    if dtype == "fp16":
        return torch_module.float16
    if dtype == "bf16":
        return torch_module.bfloat16
    if dtype == "fp32":
        return torch_module.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def generate_videos(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    try:
        import torch
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise SystemExit(
            "CogVideoX generation requires torch and diffusers. "
            "Install baseline-specific heavy dependencies before running without --dry-run."
        ) from exc

    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    if args.vae_slicing:
        pipe.vae.enable_slicing()
    if args.vae_tiling:
        pipe.vae.enable_tiling()

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected_device = args.device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(selected_device)

    for item in items:
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator_device = "cuda" if selected_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(item["seed"]))
        result = pipe(
            prompt=str(item["prompt"]),
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        export_to_video(result.frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="auto")
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

    prompts = parse_prompt_file(args.prompts)
    generation = build_generation_config(args)
    items = build_manifest_items(prompts, args.output_dir, args.seed, args.limit)

    if not args.dry_run:
        generate_videos(args, items)

    manifest = write_manifest(
        output_dir=args.output_dir,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation,
        items=items,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry-run generation manifest written: {manifest}")
    else:
        print(f"Generation manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
