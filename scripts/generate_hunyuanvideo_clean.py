#!/usr/bin/env python3
"""Generate clean causal-source videos with HunyuanVideo."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from run_pilot import parse_prompt_file


DEFAULT_MODEL = "hunyuanvideo-community/HunyuanVideo"
DEFAULT_PROMPTS = Path("prompts/backbone_capability_probe_v0.prompts")
DEFAULT_OUTPUT_DIR = Path("outputs/backbone_capability_probe_v0/hunyuanvideo")


def slugify(text: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_length].rstrip("-") or "prompt")


def resolve_torch_dtype(torch_module, dtype: str):
    return {
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "fp32": torch_module.float32,
    }[dtype]


def build_items(prompts: list[dict[str, str]], output_dir: Path, seed: int, limit: int | None):
    selected = prompts[:limit] if limit is not None else prompts
    video_dir = output_dir / "videos"
    return [
        {
            "index": index,
            "prompt": item["prompt"],
            "target_concept": item["target_concept"],
            "expected_effect": item["expected_effect"],
            "seed": seed + index,
            "video_path": str(video_dir / f"{index:03d}_{slugify(item['prompt'])}_seed{seed + index}.mp4"),
        }
        for index, item in enumerate(selected)
    ]


def generate(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    import torch
    from diffusers import HunyuanVideoPipeline
    from diffusers.utils import export_to_video

    pipe = HunyuanVideoPipeline.from_pretrained(
        args.model,
        torch_dtype=resolve_torch_dtype(torch, args.dtype),
    )
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        generator_device = "cuda"
    else:
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        pipe.to(device)
        generator_device = device

    for item in items:
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=generator_device).manual_seed(int(item["seed"]))
        result = pipe(
            prompt=str(item["prompt"]),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        export_to_video(result.frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=16000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.steps <= 0 or args.num_frames <= 0 or args.fps <= 0:
        parser.error("steps, num-frames, and fps must be positive")

    prompts = parse_prompt_file(args.prompts)
    items = build_items(prompts, args.output_dir, args.seed, args.limit)
    if not args.dry_run:
        generate(args, items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "clean",
        "model": args.model,
        "dry_run": args.dry_run,
        "prompts": str(args.prompts),
        "generation": {
            "seed": args.seed,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "dtype": args.dtype,
            "enable_model_cpu_offload": args.enable_model_cpu_offload,
            "vae_tiling": args.vae_tiling,
        },
        "items": items,
    }
    manifest_path = args.output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generation manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
