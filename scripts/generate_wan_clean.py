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
    baseline: str,
) -> list[dict[str, object]]:
    selected = prompts[:limit] if limit is not None else prompts
    items: list[dict[str, object]] = []
    video_dir = output_dir / "videos"
    for index, item in enumerate(selected):
        seed = base_seed + index
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

    if args.vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if args.vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

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
            negative_prompt=item.get("negative_prompt"),
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
    parser.add_argument("--steps", type=int, default=25)
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

    prompts = parse_prompt_file(args.prompts)
    generation = build_generation_config(args)
    items = build_manifest_items(prompts, args.output_dir, args.seed, args.limit, args.baseline)

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
