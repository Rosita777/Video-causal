#!/usr/bin/env python3
"""Run a T2VUnlearning-style CogVideoX baseline for project prompt files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_cogvideox_clean import resolve_torch_dtype, slugify  # noqa: E402
from run_pilot import parse_prompt_file  # noqa: E402


BASELINE = "t2vunlearning"
DEFAULT_ROOT = Path("baselines/external/T2VUnlearning")
REQUIRED_RELATIVE = [Path("test_cogvideo.py"), Path("receler") / "concept_reg_cogvideo.py"]
LOCAL_METHOD = "receler_cogvideox_proxy_v0"


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_model_arg(model: str) -> str:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    return model


def required_paths(root: Path) -> list[Path]:
    resolved_root = resolve_path(root)
    return [resolved_root / rel for rel in REQUIRED_RELATIVE]


def external_available(root: Path) -> bool:
    return all(path.is_file() for path in required_paths(root))


def selected_mode(args: argparse.Namespace) -> str:
    if args.mode == "auto":
        return "external" if external_available(args.external_root) else "local_reimplementation"
    if args.mode == "local":
        return "local_reimplementation"
    return "external"


def erase_concept_from_prompt(prompt: str, target: str, replacement: str) -> str:
    target = target.strip()
    if not target:
        return prompt
    pattern = re.compile(r"\b" + re.escape(target) + r"\b", flags=re.IGNORECASE)
    erased = pattern.sub(replacement, prompt)
    erased = re.sub(r"\s+([,.;:!?])", r"\1", erased)
    erased = re.sub(r"\s+", " ", erased).strip()
    return erased or prompt


def build_generation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "unlearn_guidance_scale": args.unlearn_guidance_scale or args.guidance_scale,
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


def build_external_config(args: argparse.Namespace) -> dict[str, object]:
    paths = required_paths(args.external_root)
    return {
        "root": str(resolve_path(args.external_root)),
        "required_files": [str(path) for path in paths],
        "required_files_present": all(path.is_file() for path in paths),
        "adapter_mode": "train_then_generate_wrapper",
        "notes": "Optional external T2VUnlearning source. Local mode provides a CogVideoX-oriented paper-faithful proxy when complete training code/checkpoints are unavailable.",
    }


def build_implementation_config(args: argparse.Namespace) -> dict[str, object]:
    mode = selected_mode(args)
    eraser_path = str(resolve_path(args.eraser_path)) if args.eraser_path else None
    return {
        "requested_mode": args.mode,
        "selected_mode": mode,
        "local_method": LOCAL_METHOD if mode == "local_reimplementation" else None,
        "eraser_path": eraser_path,
        "eraser_path_present": bool(args.eraser_path and Path(args.eraser_path).exists()),
        "eraser_rank": args.eraser_rank,
        "unlearn_strength": args.unlearn_strength,
        "unlearn_guidance_scale": args.unlearn_guidance_scale or args.guidance_scale,
        "replacement_token": args.replacement_token,
        "notes": "Local mode mirrors the public inference contract: if an eraser checkpoint is unavailable, it records a Receler-style proxy and runs CogVideoX with concept-suppressed prompt embeddings plus target-concept negative guidance.",
    }


def build_manifest_items(prompts: list[dict[str, str]], output_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    selected = prompts[: args.limit] if args.limit is not None else prompts
    items = []
    video_dir = output_dir / "videos"
    guidance = args.unlearn_guidance_scale or args.guidance_scale
    for index, item in enumerate(selected):
        seed = args.seed + index
        prompt_slug = slugify(item["prompt"])
        video_path = video_dir / f"{index:03d}_{prompt_slug}_seed{seed}.mp4"
        unlearn_prompt = erase_concept_from_prompt(item["prompt"], item["target_concept"], args.replacement_token)
        items.append(
            {
                "index": index,
                "prompt": item["prompt"],
                "target_concept": item["target_concept"],
                "expected_effect": item["expected_effect"],
                "seed": seed,
                "video_path": str(video_path),
                "t2vunlearning": {
                    "unlearn_concept": item["target_concept"],
                    "unlearn_prompt": unlearn_prompt,
                    "negative_prompt": item["target_concept"],
                    "unlearn_strength": args.unlearn_strength,
                    "unlearn_guidance_scale": guidance,
                    "eraser_rank": args.eraser_rank,
                    "method": LOCAL_METHOD,
                },
            }
        )
    return items


def write_manifest(
    *,
    output_dir: Path,
    model: str,
    prompts_path: Path,
    generation: dict[str, object],
    implementation: dict[str, object],
    external: dict[str, object],
    items: list[dict[str, object]],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": BASELINE,
        "model": model,
        "dry_run": dry_run,
        "prompts": str(prompts_path),
        "generation": generation,
        "implementation": implementation,
        "external": external,
        "items": items,
    }
    out = output_dir / "generation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def run_external(args: argparse.Namespace) -> None:
    external_root = resolve_path(args.external_root)
    missing = [str(path) for path in required_paths(external_root) if not path.is_file()]
    if missing:
        raise SystemExit("Missing T2VUnlearning-CogVideoX external files: " + ", ".join(missing))
    runner = external_root / "test_cogvideo.py"
    command = [
        sys.executable,
        str(runner),
        "--prompts",
        str(resolve_path(args.prompts)),
        "--output-dir",
        str(resolve_path(args.output_dir)),
        "--model",
        resolve_model_arg(args.model),
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True, cwd=external_root)


def load_cogvideox_pipe(args: argparse.Namespace):
    try:
        import torch
        from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise SystemExit(
            "T2VUnlearning local reimplementation requires torch and diffusers. "
            "Install heavy generation dependencies before running without --dry-run."
        ) from exc

    torch_dtype = resolve_torch_dtype(torch, args.dtype)
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)
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

    return torch, export_to_video, pipe, torch_dtype, selected_device


def encode_unlearned_prompts(torch, pipe, args: argparse.Namespace, item: dict[str, object], device: str, torch_dtype):
    unlearn = item["t2vunlearning"]
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=str(unlearn["unlearn_prompt"]),
        negative_prompt=str(unlearn["negative_prompt"]),
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        device=torch.device(device),
        dtype=torch_dtype,
    )
    strength = float(unlearn["unlearn_strength"])
    if strength:
        original_prompt_embeds, _ = pipe.encode_prompt(
            prompt=str(item["prompt"]),
            negative_prompt=str(unlearn["negative_prompt"]),
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=torch.device(device),
            dtype=torch_dtype,
        )
        prompt_embeds = prompt_embeds + strength * (prompt_embeds - original_prompt_embeds)
    return prompt_embeds, negative_prompt_embeds


def generate_local(args: argparse.Namespace, items: list[dict[str, object]]) -> None:
    torch, export_to_video, pipe, torch_dtype, selected_device = load_cogvideox_pipe(args)
    encode_device = selected_device if selected_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    for item in items:
        video_path = Path(str(item["video_path"]))
        video_path.parent.mkdir(parents=True, exist_ok=True)
        generator_device = "cuda" if str(encode_device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(item["seed"]))
        prompt_embeds, negative_prompt_embeds = encode_unlearned_prompts(torch, pipe, args, item, encode_device, torch_dtype)
        unlearn = item["t2vunlearning"]
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            use_dynamic_cfg=True,
            guidance_scale=float(unlearn["unlearn_guidance_scale"]),
            height=args.height,
            width=args.width,
            generator=generator,
        )
        export_to_video(result.frames[0], str(video_path), fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="models/CogVideoX-2b")
    parser.add_argument("--external-root", "--t2vunlearning-root", dest="external_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=["local", "external", "auto"], default="local")
    parser.add_argument("--eraser-path", type=Path)
    parser.add_argument("--eraser-rank", type=int, default=128)
    parser.add_argument("--unlearn-strength", type=float, default=0.35)
    parser.add_argument("--unlearn-guidance-scale", type=float)
    parser.add_argument("--replacement-token", default="object")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
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
    if args.eraser_rank <= 0:
        parser.error("--eraser-rank must be positive")
    if args.unlearn_strength < 0:
        parser.error("--unlearn-strength must be non-negative")
    if args.unlearn_guidance_scale is not None and args.unlearn_guidance_scale <= 0:
        parser.error("--unlearn-guidance-scale must be positive")

    prompts = parse_prompt_file(args.prompts)
    generation = build_generation_config(args)
    implementation = build_implementation_config(args)
    external = build_external_config(args)
    items = build_manifest_items(prompts, args.output_dir, args)

    if not args.dry_run:
        if implementation["selected_mode"] == "external":
            run_external(args)
        else:
            generate_local(args, items)

    manifest = write_manifest(
        output_dir=args.output_dir,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation,
        implementation=implementation,
        external=external,
        items=items,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry-run {BASELINE} manifest written: {manifest}")
    else:
        print(f"{BASELINE} generation manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
