#!/usr/bin/env python3
"""Run the SAFREE CogVideoX adapter for project prompt files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import resolve_torch_dtype, slugify  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


DEFAULT_MODEL = "models/CogVideoX-2b"
DEFAULT_SAFREE_ROOT = Path("baselines/external/SAFREE")
SAFREE_PIPELINE_RELATIVE = Path("cogvideox") / "cogvideox_pipeline.py"


def safree_pipeline_path(safree_root: Path) -> Path:
    return safree_root / SAFREE_PIPELINE_RELATIVE


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
        "scheduler": "CogVideoXDPMScheduler",
        "scheduler_timestep_spacing": "trailing",
        "use_dynamic_cfg": True,
    }


def build_safree_config(args: argparse.Namespace) -> dict[str, object]:
    pipeline_path = safree_pipeline_path(args.safree_root)
    return {
        "safree_root": str(args.safree_root),
        "external_pipeline": str(pipeline_path),
        "external_pipeline_present": pipeline_path.is_file(),
        "concept_injection": "target_concept_as_single_concept_dict_entry",
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
        target = item["target_concept"]
        items.append(
            {
                "index": index,
                "prompt": item["prompt"],
                "target_concept": target,
                "expected_effect": item["expected_effect"],
                "seed": seed,
                "video_path": str(video_path),
                "safree_concept_key": target,
                "safree_concept_terms": [target],
            }
        )
    return items


def write_manifest(
    *,
    output_dir: Path,
    model: str,
    prompts_path: Path,
    generation: dict[str, object],
    safree: dict[str, object],
    items: list[dict[str, object]],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "safree_cogvideox",
        "model": model,
        "dry_run": dry_run,
        "prompts": str(prompts_path),
        "generation": generation,
        "safree": safree,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_safree_module(pipeline_path: Path):
    if not pipeline_path.is_file():
        raise SystemExit(
            f"Missing SAFREE CogVideoX pipeline: {pipeline_path}. "
            "Download or clone https://github.com/jaehong31/SAFREE into baselines/external/SAFREE."
        )
    module_dir = str(pipeline_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("safree_cogvideox_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import SAFREE CogVideoX pipeline from {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_videos(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    try:
        import torch
        from diffusers import CogVideoXDPMScheduler
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise SystemExit(
            "SAFREE-CogVideoX requires torch and diffusers. "
            "Install heavy generation dependencies before running without --dry-run."
        ) from exc

    module = load_safree_module(safree_pipeline_path(args.safree_root))
    if not hasattr(module, "CogVideoXPipeline") or not hasattr(module, "CONCEPT_DICT"):
        raise SystemExit("SAFREE CogVideoX pipeline must define CogVideoXPipeline and CONCEPT_DICT")

    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    pipe = module.CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

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
        concept_key = str(item["safree_concept_key"])
        module.CONCEPT_DICT[concept_key] = list(item["safree_concept_terms"])
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator_device = "cuda" if selected_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(item["seed"]))
        result = pipe(
            prompt=str(item["prompt"]),
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
            concept=concept_key,
        )
        export_to_video(result.frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--safree-root", type=Path, default=DEFAULT_SAFREE_ROOT)
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
    safree = build_safree_config(args)
    items = build_manifest_items(prompts, args.output_dir, args.seed, args.limit)

    if not args.dry_run:
        generate_videos(args, items)

    manifest = write_manifest(
        output_dir=args.output_dir,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation,
        safree=safree,
        items=items,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry-run SAFREE manifest written: {manifest}")
    else:
        print(f"SAFREE generation manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
