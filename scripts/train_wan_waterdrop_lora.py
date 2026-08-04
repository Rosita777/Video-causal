#!/usr/bin/env python3
"""Train a Wan LoRA on aligned waterdrop counterfactual targets.

The plain baseline uses counterfactual flow matching. Mask-bg reweights that
loss on the observed causal residual. Paired-separation additionally pushes the
prediction away from the factual causal target inside the residual. Both masked
objectives distill the frozen base prediction outside the residual. Preserve
rows distill the frozen model over the complete latent, without requiring a
category-specific preservation label. Video latents and prompt embeddings are
cached once so the large VAE and text encoder are not resident during training.
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
from transformers import AutoTokenizer

from causal_lora_activation_gate import (
    CausalLoRAActivationGate,
    make_temporally_persistent_gate,
)
from target_token_attention_suppression import find_token_mask


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
    parser.add_argument(
        "--objective",
        choices=[
            "plain",
            "mask_bg",
            "paired_sep",
            "dual_traj",
            "causal_gate",
            "counterfactual_sft",
            "target_conditioned_sft",
            "target_conditioned_redirect",
        ],
        default="plain",
    )
    parser.add_argument(
        "--causal-gate-dir",
        type=Path,
        help="Per-scene causal gates exported by the dual-gate evaluator.",
    )
    parser.add_argument(
        "--gate-floor",
        type=float,
        default=0.0,
        help="Minimum erase weight outside the exported causal gate.",
    )
    parser.add_argument(
        "--activation-gate-dir",
        type=Path,
        help="Gate LoRA residual activations using per-scene patch-token masks.",
    )
    parser.add_argument(
        "--target-phrase",
        action="append",
        default=[],
        help="Exact target phrase used to gate cross-attention text LoRA tokens.",
    )
    parser.add_argument(
        "--persistent-causal-time",
        action="store_true",
        help="Keep the full causal-chain spatial union active after its first frame.",
    )
    parser.add_argument("--mask-weight", type=float, default=4.0)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--redirect-weight", type=float, default=1.0)
    parser.add_argument(
        "--preserve-weight",
        type=float,
        default=1.0,
        help="Weight for frozen-teacher matching on training_role=preserve rows.",
    )
    parser.add_argument(
        "--balanced-roles",
        action="store_true",
        help="Alternate erase and preserve rows when training with --role all.",
    )
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
    if min(
        args.mask_weight,
        args.background_weight,
        args.pair_weight,
        args.pair_margin,
        args.redirect_weight,
        args.preserve_weight,
        args.gate_floor,
    ) < 0:
        parser.error("all loss weights and margins must be non-negative")
    if args.gate_floor > 1:
        parser.error("--gate-floor must be at most 1")
    gated_objectives = {
        "causal_gate",
        "counterfactual_sft",
        "target_conditioned_sft",
        "target_conditioned_redirect",
    }
    if args.objective in gated_objectives and args.causal_gate_dir is None:
        parser.error(f"--causal-gate-dir is required for --objective {args.objective}")
    if args.objective in {"target_conditioned_sft", "target_conditioned_redirect"}:
        if args.activation_gate_dir is None:
            parser.error("--activation-gate-dir is required for target-conditioned objectives")
        if not args.target_phrase:
            parser.error("at least one --target-phrase is required for target-conditioned objectives")
        if args.role != "erase":
            parser.error("target-conditioned objectives currently support --role erase only")
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
        raise ValueError(f"No rows found for role={args.role!r}, objective={args.objective!r}")
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


def causal_residual_mask(factual: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Build a robust soft spatiotemporal mask in Wan latent resolution."""
    residual = (factual.float() - clean.float()).abs().mean(dim=1, keepdim=True)
    low = torch.quantile(residual, 0.50)
    high = torch.quantile(residual, 0.95)
    mask = ((residual - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    mask = torch.nn.functional.avg_pool3d(mask, kernel_size=3, stride=1, padding=1)
    return mask.to(dtype=torch.bfloat16)


def paired_separation_loss(
    prediction: torch.Tensor,
    counterfactual_target: torch.Tensor,
    factual_target: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require the prediction to prefer the counterfactual target in the causal region."""
    positive_error = (prediction.float() - counterfactual_target.float()).square().mean(
        dim=1, keepdim=True
    )
    negative_error = (prediction.float() - factual_target.float()).square().mean(
        dim=1, keepdim=True
    )
    hinge = torch.relu(margin + positive_error - negative_error)
    weights = mask.float()
    return (hinge * weights).sum() / weights.sum().clamp_min(1e-6)


def factual_redirect_loss(
    prediction: torch.Tensor,
    noisy_factual: torch.Tensor,
    counterfactual_clean: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pull the clean endpoint predicted from a factual trajectory toward the counterfactual."""
    predicted_clean = noisy_factual.float() - sigma.float() * prediction.float()
    error = (predicted_clean - counterfactual_clean.float()).square().mean(dim=1, keepdim=True)
    weights = mask.float()
    return (error * weights).sum() / weights.sum().clamp_min(1e-6)


def gated_flow_loss(element_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = element_loss.float().mean(dim=1, keepdim=True)
    weights = mask.float()
    return (error * weights).sum() / weights.sum().clamp_min(1e-6)


def load_causal_gate(args: argparse.Namespace, scene_id: str, mask: torch.Tensor) -> torch.Tensor:
    gate_path = args.causal_gate_dir / f"{scene_id}.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing causal gate for {scene_id}: {gate_path}")
    payload = torch.load(gate_path, map_location="cpu", weights_only=True)
    gate = payload["gate"].float()[None, None]
    gate = torch.nn.functional.interpolate(
        gate,
        size=mask.shape[-3:],
        mode="trilinear",
        align_corners=False,
    ).to(device=mask.device)
    gate = args.gate_floor + (1.0 - args.gate_floor) * gate.clamp(0.0, 1.0)
    combined = mask.float() * gate
    return make_temporally_persistent_gate(combined) if args.persistent_causal_time else combined


def load_activation_gate(
    gate_dir: Path,
    scene_id: str,
    device: torch.device,
    *,
    persistent_time: bool = False,
) -> torch.Tensor:
    gate_path = gate_dir / f"{scene_id}.pt"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing activation gate for {scene_id}: {gate_path}")
    payload = torch.load(gate_path, map_location="cpu", weights_only=True)
    gate = payload["gate"].float().unsqueeze(0)
    if persistent_time:
        gate = make_temporally_persistent_gate(gate)
    return gate.to(device=device)


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
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    processor = WanPipeline.from_pretrained(
        str(args.model), transformer=None, text_encoder=None, tokenizer=None, vae=None
    ).video_processor
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)

    for index, (row, cache_path) in enumerate(zip(rows, cache_paths, strict=True)):
        target_path = resolve_path(project_root, row["desired_target_video"])
        target_frames = read_video(target_path, args.num_frames)
        target_video = processor.preprocess_video(target_frames, height=args.height, width=args.width).to(
            device=device, dtype=torch.bfloat16
        )
        raw_latents = vae.encode(target_video).latent_dist.mode()
        latents = ((raw_latents - mean) / std).cpu()
        payload = {
            "scene_id": row["scene_id"],
            "prompt": row["prompt"],
            "training_role": row["training_role"],
            "target_video": str(target_path),
            "latents": latents,
            "prompt_embeds": prompt_embeddings[row["prompt"]],
        }
        if args.objective in {
            "mask_bg",
            "paired_sep",
            "dual_traj",
            "causal_gate",
            "counterfactual_sft",
            "target_conditioned_sft",
            "target_conditioned_redirect",
        } and row.get("residual_mask_enabled") == "yes":
            factual_path = resolve_path(project_root, row["residual_mask_factual_video"])
            factual_frames = read_video(factual_path, args.num_frames)
            factual_video = processor.preprocess_video(
                factual_frames, height=args.height, width=args.width
            ).to(device=device, dtype=torch.bfloat16)
            factual_raw_latents = vae.encode(factual_video).latent_dist.mode()
            factual_latents = (factual_raw_latents - mean) / std
            mask = causal_residual_mask(factual_latents, latents.to(device))
            payload["factual_video"] = str(factual_path)
            payload["factual_latents"] = factual_latents.cpu()
            payload["residual_mask"] = mask.cpu()
            payload["mask_mean"] = float(mask.float().mean())
            payload["mask_peak"] = float(mask.float().max())
            del factual_video, factual_raw_latents, factual_latents, mask
        torch.save(payload, cache_path)
        mask_note = f" mask_mean={payload['mask_mean']:.4f}" if "mask_mean" in payload else ""
        print(
            f"Cached {index + 1}/{len(rows)}: {row['scene_id']} {tuple(latents.shape)}{mask_note}",
            flush=True,
        )
        del target_video, raw_latents, latents, payload
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
    activation_controller = (
        CausalLoRAActivationGate(transformer)
        if args.activation_gate_dir is not None
        else None
    )
    if activation_controller is not None:
        print(
            f"Activation-gated LoRA modules: {activation_controller.module_count}",
            flush=True,
        )
    target_tokenizer = (
        AutoTokenizer.from_pretrained(str(args.model), subfolder="tokenizer")
        if args.objective in {"target_conditioned_sft", "target_conditioned_redirect"}
        else None
    )
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(f"Trainable LoRA parameters: {trainable_count:,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    order = list(range(len(cache_paths)))
    role_indices = {"erase": [], "preserve": []}
    if args.balanced_roles and args.role == "all":
        for index, cache_path in enumerate(cache_paths):
            metadata = torch.load(cache_path, map_location="cpu", weights_only=True)
            role_indices[metadata["training_role"]].append(index)
        if not all(role_indices.values()):
            raise ValueError("--balanced-roles requires both erase and preserve rows")
        for role in role_indices:
            order_rng.shuffle(role_indices[role])
        role_cursors = {"erase": 0, "preserve": 0}
    losses: list[float] = []
    remove_losses: list[float] = []
    background_losses: list[float] = []
    pair_losses: list[float] = []
    redirect_losses: list[float] = []
    preserve_losses: list[float] = []
    gate_means: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    transformer.train()
    for step in range(1, args.max_steps + 1):
        if args.balanced_roles and args.role == "all":
            role = "erase" if step % 2 else "preserve"
            cursor = role_cursors[role]
            if cursor >= len(role_indices[role]):
                order_rng.shuffle(role_indices[role])
                cursor = 0
            sample_index = role_indices[role][cursor]
            role_cursors[role] = cursor + 1
        else:
            if (step - 1) % len(order) == 0:
                order_rng.shuffle(order)
            sample_index = order[(step - 1) % len(order)]
        sample = torch.load(cache_paths[sample_index], map_location="cpu", weights_only=True)
        is_preserve = sample["training_role"] == "preserve"
        clean = sample["latents"].to(device=device, dtype=torch.bfloat16)
        prompt_embeds = sample["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        if activation_controller is not None:
            if is_preserve:
                activation_controller.clear_gate()
            else:
                activation_controller.set_gate(
                    load_activation_gate(
                        args.activation_gate_dir,
                        sample["scene_id"],
                        device,
                        persistent_time=args.persistent_causal_time,
                    )
                )
                if target_tokenizer is not None:
                    activation_controller.set_text_gate(
                        find_token_mask(
                            target_tokenizer,
                            sample["prompt"],
                            args.target_phrase,
                            max_length=prompt_embeds.shape[1],
                        ).to(device=device)
                    )
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device, dtype=torch.bfloat16)
        sigma = torch.rand((clean.shape[0],), generator=generator, dtype=torch.float32).to(device)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        noisy = ((1.0 - sigma) * clean + sigma * noise).to(dtype=torch.bfloat16)
        target = (noise - clean).to(dtype=torch.bfloat16)
        timestep = (sigma.flatten() * 1000.0).to(dtype=torch.bfloat16)

        teacher_prediction = None
        teacher_factual_prediction = None
        has_residual_mask = (
            args.objective in {
                "mask_bg",
                "paired_sep",
                "dual_traj",
                "causal_gate",
                "counterfactual_sft",
                "target_conditioned_sft",
                "target_conditioned_redirect",
            }
            and "residual_mask" in sample
        )
        uses_factual = has_residual_mask and args.objective in {
            "paired_sep",
            "dual_traj",
            "causal_gate",
            "target_conditioned_redirect",
        }
        if uses_factual:
            factual = sample["factual_latents"].to(device=device, dtype=torch.bfloat16)
            factual_target = (noise - factual).to(dtype=torch.bfloat16)
        if has_residual_mask and args.objective in {
            "dual_traj",
            "causal_gate",
            "target_conditioned_redirect",
        }:
            noisy_factual = ((1.0 - sigma) * factual + sigma * noise).to(dtype=torch.bfloat16)
        needs_teacher = is_preserve or (has_residual_mask and args.background_weight > 0)
        if needs_teacher:
            if activation_controller is not None:
                activation_controller.disable()
            transformer.disable_adapters()
            with torch.no_grad():
                teacher_prediction = transformer(
                    hidden_states=noisy,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if has_residual_mask and args.objective in {
                    "dual_traj",
                    "causal_gate",
                    "target_conditioned_redirect",
                }:
                    teacher_factual_prediction = transformer(
                        hidden_states=noisy_factual,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
            transformer.enable_adapters()
            if activation_controller is not None:
                activation_controller.enable()

        prediction = transformer(
            hidden_states=noisy,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        element_loss = torch.nn.functional.mse_loss(
            prediction.float(), target.float(), reduction="none"
        )
        if is_preserve:
            preserve_loss = torch.nn.functional.mse_loss(
                prediction.float(), teacher_prediction.float()
            )
            remove_loss = torch.zeros((), device=device)
            background_loss = torch.zeros((), device=device)
            pair_loss = torch.zeros((), device=device)
            redirect_loss = torch.zeros((), device=device)
            combined_loss = args.preserve_weight * preserve_loss
        elif has_residual_mask:
            preserve_loss = torch.zeros((), device=device)
            mask = sample["residual_mask"].to(device=device, dtype=torch.float32)
            if args.objective in {
                "causal_gate",
                "counterfactual_sft",
                "target_conditioned_sft",
                "target_conditioned_redirect",
            }:
                mask = load_causal_gate(args, sample["scene_id"], mask)
                remove_loss = gated_flow_loss(element_loss, mask)
            elif args.objective == "mask_bg":
                remove_loss = (element_loss * (1.0 + args.mask_weight * mask)).mean()
            else:
                remove_loss = element_loss.mean()
            if teacher_prediction is not None:
                background_loss = (
                    (prediction.float() - teacher_prediction.float()).square() * (1.0 - mask)
                ).mean()
            else:
                background_loss = torch.zeros((), device=device)
            if args.objective in {"paired_sep", "dual_traj", "causal_gate"}:
                pair_loss = paired_separation_loss(
                    prediction, target, factual_target, mask, args.pair_margin
                )
            else:
                pair_loss = torch.zeros((), device=device)
            if args.objective in {
                "dual_traj",
                "causal_gate",
                "target_conditioned_redirect",
            }:
                factual_prediction = transformer(
                    hidden_states=noisy_factual,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                redirect_loss = factual_redirect_loss(
                    factual_prediction, noisy_factual, clean, sigma, mask
                )
                if teacher_factual_prediction is not None:
                    factual_background_loss = (
                        (factual_prediction.float() - teacher_factual_prediction.float()).square()
                        * (1.0 - mask)
                    ).mean()
                    background_loss = 0.5 * (background_loss + factual_background_loss)
            else:
                redirect_loss = torch.zeros((), device=device)
            combined_loss = (
                remove_loss
                + args.background_weight * background_loss
                + args.pair_weight * pair_loss
                + args.redirect_weight * redirect_loss
            )
        else:
            preserve_loss = torch.zeros((), device=device)
            remove_loss = element_loss.mean()
            background_loss = torch.zeros((), device=device)
            pair_loss = torch.zeros((), device=device)
            redirect_loss = torch.zeros((), device=device)
            combined_loss = remove_loss
        loss = combined_loss / args.grad_accum
        loss.backward()
        if activation_controller is not None:
            activation_controller.clear_gate()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach()) * args.grad_accum
        remove_value = float(remove_loss.detach())
        background_value = float(background_loss.detach())
        pair_value = float(pair_loss.detach())
        redirect_value = float(redirect_loss.detach())
        preserve_value = float(preserve_loss.detach())
        gate_mean = float(mask.mean()) if has_residual_mask else 0.0
        losses.append(loss_value)
        remove_losses.append(remove_value)
        background_losses.append(background_value)
        pair_losses.append(pair_value)
        redirect_losses.append(redirect_value)
        preserve_losses.append(preserve_value)
        gate_means.append(gate_mean)
        elapsed = time.time() - started
        print(
            f"step={step}/{args.max_steps} scene={sample['scene_id']} "
            f"role={sample['training_role']} masked={has_residual_mask} "
            f"loss={loss_value:.6f} "
            f"remove={remove_value:.6f} bg={background_value:.6f} "
            f"pair={pair_value:.6f} redirect={redirect_value:.6f} preserve={preserve_value:.6f} "
            f"gate={gate_mean:.6f} "
            f"mean20={np.mean(losses[-20:]):.6f} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del sample, clean, prompt_embeds, noise, noisy, target, prediction, loss, element_loss
        if teacher_prediction is not None:
            del teacher_prediction
        if has_residual_mask:
            del mask
        if uses_factual:
            del factual, factual_target, pair_loss
        if has_residual_mask and args.objective in {
            "dual_traj",
            "causal_gate",
            "target_conditioned_redirect",
        }:
            del noisy_factual, factual_prediction, redirect_loss
            if teacher_factual_prediction is not None:
                del teacher_factual_prediction, factual_background_loss

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
                    "objective": args.objective,
                    "mask_weight": args.mask_weight,
                    "background_weight": args.background_weight,
                    "pair_weight": args.pair_weight,
                    "pair_margin": args.pair_margin,
                    "redirect_weight": args.redirect_weight,
                    "preserve_weight": args.preserve_weight,
                    "balanced_roles": args.balanced_roles,
                    "causal_gate_dir": str(args.causal_gate_dir) if args.causal_gate_dir else None,
                    "gate_floor": args.gate_floor,
                    "activation_gate_dir": (
                        str(args.activation_gate_dir) if args.activation_gate_dir else None
                    ),
                    "target_phrase": args.target_phrase,
                    "persistent_causal_time": args.persistent_causal_time,
                    "mean_remove_loss_last_20": float(np.mean(remove_losses[-20:])),
                    "mean_background_loss_last_20": float(np.mean(background_losses[-20:])),
                    "mean_pair_loss_last_20": float(np.mean(pair_losses[-20:])),
                    "mean_redirect_loss_last_20": float(np.mean(redirect_losses[-20:])),
                    "mean_preserve_loss_last_20": float(np.mean(preserve_losses[-20:])),
                    "mean_gate_last_20": float(np.mean(gate_means[-20:])),
                },
            )
            print(f"Saved {checkpoint}", flush=True)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    rows = load_rows(args)
    print(
        f"Selected {len(rows)} manifest rows with role={args.role}, objective={args.objective}",
        flush=True,
    )
    if args.dry_run:
        for row in rows:
            target = resolve_path(project_root, row["desired_target_video"])
            if not target.exists():
                raise FileNotFoundError(target)
            if args.objective in {
                "mask_bg",
                "paired_sep",
                "dual_traj",
                "causal_gate",
                "counterfactual_sft",
                "target_conditioned_sft",
                "target_conditioned_redirect",
            } and row.get("residual_mask_enabled") == "yes":
                factual = resolve_path(project_root, row["residual_mask_factual_video"])
                if not factual.exists():
                    raise FileNotFoundError(factual)
                if args.objective in {
                    "causal_gate",
                    "counterfactual_sft",
                    "target_conditioned_sft",
                    "target_conditioned_redirect",
                }:
                    gate = args.causal_gate_dir / f"{row['scene_id']}.pt"
                    if not gate.exists():
                        raise FileNotFoundError(gate)
                if args.activation_gate_dir is not None and row["training_role"] == "erase":
                    activation_gate = args.activation_gate_dir / f"{row['scene_id']}.pt"
                    if not activation_gate.exists():
                        raise FileNotFoundError(activation_gate)
        print("Dry run passed: manifest and target videos are valid.")
        return 0
    cache_paths = build_cache(args, rows, project_root)
    if not args.cache_only:
        train(args, cache_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
