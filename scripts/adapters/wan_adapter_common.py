"""Shared helpers for Wan-family baseline adapters."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import resolve_torch_dtype, slugify  # noqa: E402
from generate_wan_clean import output_frames  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


DEFAULT_MODEL = "models/Wan2.1-T2V-1.3B-Diffusers"
PIPELINE_NAME = "WanPipeline"


def erase_concept_from_prompt(prompt: str, target: str, replacement: str) -> str:
    target = target.strip()
    if not target:
        return prompt
    pattern = re.compile(r"\b" + re.escape(target) + r"\b", flags=re.IGNORECASE)
    erased = pattern.sub(replacement, prompt)
    erased = re.sub(r"\s+([,.;:!?])", r"\1", erased)
    erased = re.sub(r"\s+", " ", erased).strip()
    return erased or prompt


def generation_config(args: argparse.Namespace) -> dict[str, object]:
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
        "limit": args.limit,
    }


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--enable-sequential-cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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


def selected_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts = parse_prompt_file(args.prompts)
    return prompts[: args.limit] if args.limit is not None else prompts


def base_item(index: int, prompt_row: dict[str, str], output_dir: Path, seed: int) -> dict[str, object]:
    prompt_slug = slugify(prompt_row["prompt"])
    return {
        "index": index,
        "prompt": prompt_row["prompt"],
        "target_concept": prompt_row["target_concept"],
        "expected_effect": prompt_row["expected_effect"],
        "seed": seed,
        "video_path": str(output_dir / "videos" / f"{index:03d}_{prompt_slug}_seed{seed}.mp4"),
    }


def write_manifest(
    *,
    output_dir: Path,
    baseline: str,
    model: str,
    prompts_path: Path,
    generation: dict[str, object],
    implementation: dict[str, object],
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
        "implementation": implementation,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_wan_pipe(args: argparse.Namespace):
    try:
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise SystemExit("Wan adapters require torch and diffusers with WanPipeline.") from exc

    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    if args.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        pipe.to(selected_device)

    return torch, torch_dtype, export_to_video, pipe, selected_device


def encode_cfg(
    pipe,
    torch_module,
    torch_dtype,
    *,
    prompt: str,
    negative_prompt: str,
    device: str,
):
    return pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch_module.device(device),
        dtype=torch_dtype,
    )


def generate_encoded_videos(
    args: argparse.Namespace,
    items: list[dict[str, object]],
    encode_item: Callable,
) -> None:
    torch_module, torch_dtype, export_to_video, pipe, selected_device = load_wan_pipe(args)
    generator_device = "cuda" if str(selected_device).startswith("cuda") and torch_module.cuda.is_available() else "cpu"

    for item in items:
        prompt_embeds, negative_prompt_embeds = encode_item(pipe, torch_module, torch_dtype, selected_device, item)
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch_module.Generator(device=generator_device).manual_seed(int(item["seed"]))
        result = pipe(
            prompt=None,
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
