#!/usr/bin/env python3
"""Train a Wan LoRA on aligned waterdrop counterfactual targets.

The v0 baseline uses only rows whose training role is ``erase``. Text prompts
condition the frozen Wan backbone while the desired clean videos provide the
flow-matching target. Video latents and prompt embeddings are cached once so
the large VAE and text encoder are not resident during transformer training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--role", choices=["erase", "preserve", "all"], default="erase")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be divisible by 16")
    if args.num_frames % 4 != 1:
        parser.error("--num-frames must be 4n+1 for the Wan VAE")
    if min(args.max_steps, args.rank, args.alpha, args.grad_accum) <= 0:
        parser.error("step, rank, alpha, and accumulation values must be positive")
    return args


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.role != "all":
        rows = [row for row in rows if row["training_role"] == args.role]
    if not rows:
        raise ValueError(f"No {args.role!r} rows found in {args.manifest}")
    required = {"scene_id", "prompt", "desired_target_video", "training_role"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def read_video(path: Path, num_frames: int) -> list[Image.Image]:
    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    if len(frames) < num_frames:
        raise ValueError(f"{path} has {len(frames)} frames; expected at least {num_frames}")
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return [frames[index] for index in indices]


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def cache_prompt_embeddings(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    device = torch.device(args.device)
    pipe = WanPipeline.from_pretrained(
        str(args.model), transformer=None, vae=None, torch_dtype=torch.bfloat16
    ).to(device)
    embeddings: dict[str, torch.Tensor] = {}
    for row in rows:
        prompt = row["prompt"]
        if prompt not in embeddings:
            prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                device=device,
                dtype=torch.bfloat16,
            )
            embeddings[prompt] = prompt_embeds.cpu()
    del pipe
    clear_memory()
    return embeddings


@torch.no_grad()
def build_cache(args: argparse.Namespace, rows: list[dict[str, str]], project_root: Path) -> list[Path]:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [args.cache_dir / f"{index:03d}_{row['scene_id']}.pt" for index, row in enumerate(rows)]
    if not args.rebuild_cache and all(path.exists() for path in cache_paths):
        print(f"Using {len(cache_paths)} existing cache files in {args.cache_dir}", flush=True)
        return cache_paths

    prompt_embeddings = cache_prompt_embeddings(args, rows)
    device = torch.device(args.device)
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model), subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()
    vae.enable_tiling()
    processor = WanPipeline.from_pretrained(
        str(args.model), transformer=None, text_encoder=None, tokenizer=None, vae=None
    ).video_processor
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)

    for index, (row, cache_path) in enumerate(zip(rows, cache_paths, strict=True)):
        target_path = resolve_path(project_root, row["desired_target_video"])
        frames = read_video(target_path, args.num_frames)
        video = processor.preprocess_video(frames, height=args.height, width=args.width).to(
            device=device, dtype=torch.bfloat16
        )
        raw_latents = vae.encode(video).latent_dist.mode()
        latents = ((raw_latents - mean) / std).cpu()
        payload = {
            "scene_id": row["scene_id"],
            "prompt": row["prompt"],
            "training_role": row["training_role"],
            "target_video": str(target_path),
            "latents": latents,
            "prompt_embeds": prompt_embeddings[row["prompt"]],
        }
        torch.save(payload, cache_path)
        print(f"Cached {index + 1}/{len(rows)}: {row['scene_id']} {tuple(latents.shape)}", flush=True)
        del video, raw_latents, latents, payload
        clear_memory()

    del vae, processor
    clear_memory()
    return cache_paths


def save_lora(transformer: WanTransformer3DModel, output_dir: Path, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(transformer)
    WanPipeline.save_lora_weights(
        str(output_dir), transformer_lora_layers=convert_state_dict_to_diffusers(state), safe_serialization=True
    )
    (output_dir / "training_state.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def train(args: argparse.Namespace, cache_paths: list[Path]) -> None:
    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(f"Trainable LoRA parameters: {trainable_count:,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    order = list(range(len(cache_paths)))
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    transformer.train()
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(order) == 0:
            order_rng.shuffle(order)
        sample_index = order[(step - 1) % len(order)]
        sample = torch.load(cache_paths[sample_index], map_location="cpu", weights_only=True)
        clean = sample["latents"].to(device=device, dtype=torch.bfloat16)
        prompt_embeds = sample["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device, dtype=torch.bfloat16)
        sigma = torch.rand((clean.shape[0],), generator=generator, dtype=torch.float32).to(device)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * clean + sigma * noise
        target = noise - clean
        timestep = (sigma.flatten() * 1000.0).to(dtype=torch.bfloat16)

        prediction = transformer(
            hidden_states=noisy,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float()) / args.grad_accum
        loss.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach()) * args.grad_accum
        losses.append(loss_value)
        elapsed = time.time() - started
        print(
            f"step={step}/{args.max_steps} scene={sample['scene_id']} loss={loss_value:.6f} "
            f"mean20={np.mean(losses[-20:]):.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del sample, clean, prompt_embeds, noise, noisy, target, prediction, loss

        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint = args.output_dir / f"checkpoint-{step:06d}"
            save_lora(
                transformer,
                checkpoint,
                {
                    "step": step,
                    "max_steps": args.max_steps,
                    "mean_loss_last_20": float(np.mean(losses[-20:])),
                    "manifest": str(args.manifest),
                    "model": str(args.model),
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "learning_rate": args.learning_rate,
                    "seed": args.seed,
                    "role": args.role,
                },
            )
            print(f"Saved {checkpoint}", flush=True)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    rows = load_rows(args)
    print(f"Selected {len(rows)} manifest rows with role={args.role}", flush=True)
    if args.dry_run:
        for row in rows:
            target = resolve_path(project_root, row["desired_target_video"])
            if not target.exists():
                raise FileNotFoundError(target)
        print("Dry run passed: manifest and target videos are valid.")
        return 0
    cache_paths = build_cache(args, rows, project_root)
    if not args.cache_only:
        train(args, cache_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
