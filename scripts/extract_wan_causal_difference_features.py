#!/usr/bin/env python3
"""Extract paired Wan hidden-state differences for causal-subspace feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collision-cache", type=Path, required=True)
    parser.add_argument("--generic-cache", type=Path, required=True)
    parser.add_argument("--waterdrop-cache", type=Path, required=True)
    parser.add_argument("--other-ball-cache", type=Path)
    parser.add_argument("--negation-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--tokens-per-sample", type=int, default=64)
    parser.add_argument("--tokens-per-frame", type=int, default=0)
    parser.add_argument("--background-per-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def cache_files(root: Path, role: str | None = None) -> list[Path]:
    paths = sorted(root.glob("*.pt"))
    if role is None:
        return paths
    selected = []
    for path in paths:
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if sample["training_role"] == role:
            selected.append(path)
    return selected


def patch_mask(mask: torch.Tensor) -> torch.Tensor:
    pooled = torch.nn.functional.avg_pool3d(
        mask.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2)
    )
    return pooled.flatten()


def generic_pair(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    factual = latents
    counterfactual = factual[:, :, :1].expand_as(factual).clone()
    residual = (factual.float() - counterfactual.float()).abs().mean(dim=1, keepdim=True)
    low = torch.quantile(residual, 0.50)
    high = torch.quantile(residual, 0.95)
    mask = ((residual - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    return factual, counterfactual, mask


@torch.no_grad()
def extract_group(
    *,
    name: str,
    paths: list[Path],
    transformer: WanTransformer3DModel,
    layer: int,
    sigma_value: float,
    tokens_per_sample: int,
    tokens_per_frame: int,
    background_per_frame: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    hook = transformer.blocks[layer].register_forward_hook(capture)
    records = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        for index, path in enumerate(paths):
            sample = torch.load(path, map_location="cpu", weights_only=True)
            if name in {"generic", "negation"}:
                factual, counterfactual, mask = generic_pair(sample["latents"])
            else:
                if "factual_latents" not in sample or "residual_mask" not in sample:
                    continue
                factual = sample["factual_latents"]
                counterfactual = sample["latents"]
                mask = sample["residual_mask"]

            factual = factual.to(device=device, dtype=torch.bfloat16)
            counterfactual = counterfactual.to(device=device, dtype=torch.bfloat16)
            prompt = sample["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
            noise = torch.randn(factual.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(device=device, dtype=torch.bfloat16)
            sigma = torch.full((2, 1, 1, 1, 1), sigma_value, device=device)
            clean = torch.cat([factual, counterfactual], dim=0)
            noisy = ((1.0 - sigma) * clean + sigma * noise.expand_as(clean)).to(torch.bfloat16)
            timestep = torch.full((2,), sigma_value * 1000.0, device=device, dtype=torch.bfloat16)
            captured.clear()
            transformer(
                hidden_states=noisy,
                timestep=timestep,
                encoder_hidden_states=prompt.expand(2, -1, -1),
                return_dict=False,
            )
            hidden = captured[-1].float().cpu()
            delta = hidden[0] - hidden[1]
            weights = patch_mask(mask)
            if len(weights) != len(delta):
                raise ValueError(
                    f"Token/mask mismatch for {path}: hidden={len(delta)} mask={len(weights)}"
                )
            if tokens_per_frame > 0:
                frame_weights = weights.view(mask.shape[-3], -1)
                causal_indices = []
                background_indices = []
                for frame_index, values in enumerate(frame_weights):
                    offset = frame_index * len(values)
                    causal_indices.append(
                        torch.topk(values, k=min(tokens_per_frame, len(values)), sorted=False).indices
                        + offset
                    )
                    background_indices.append(
                        torch.topk(
                            values,
                            k=min(background_per_frame, len(values)),
                            largest=False,
                            sorted=False,
                        ).indices
                        + offset
                    )
                indices = torch.cat(causal_indices)
                background_indices_tensor = torch.cat(background_indices)
            else:
                count = min(tokens_per_sample, len(weights))
                indices = torch.topk(weights, k=count, sorted=False).indices
                background_indices_tensor = torch.topk(
                    weights,
                    k=min(background_per_frame * mask.shape[-3], len(weights)),
                    largest=False,
                    sorted=False,
                ).indices
            token_features = delta[indices]
            factual_features = hidden[0][indices]
            background_features = hidden[0][background_indices_tensor]
            grid_height, grid_width = mask.shape[-2] // 2, mask.shape[-1] // 2
            positions = torch.stack(
                [
                    indices // (grid_height * grid_width),
                    (indices % (grid_height * grid_width)) // grid_width,
                    indices % grid_width,
                ],
                dim=1,
            )
            records.append(
                {
                    "scene_id": sample["scene_id"],
                    "path": str(path),
                    "video_path": sample["target_video"],
                    "features": token_features.to(torch.float16),
                    "pooled": token_features.mean(dim=0).to(torch.float16),
                    "factual_features": factual_features.to(torch.float16),
                    "background_features": background_features.to(torch.float16),
                    "positions": positions.to(torch.int16),
                    "mask_weights": weights[indices].to(torch.float16),
                    "mask_weight_mean": float(weights[indices].mean()),
                }
            )
            print(
                f"{name} {index + 1}/{len(paths)} scene={sample['scene_id']} "
                f"tokens={len(indices)} delta_norm={token_features.norm(dim=1).mean():.4f}",
                flush=True,
            )
    finally:
        hook.remove()
    return {"name": name, "records": records}


def main() -> int:
    args = parse_args()
    if not 0.0 < args.sigma < 1.0:
        raise ValueError("--sigma must be between 0 and 1")
    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model), subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval().requires_grad_(False)
    if not 0 <= args.layer < len(transformer.blocks):
        raise ValueError(f"Invalid layer {args.layer}; model has {len(transformer.blocks)} blocks")

    erase_paths = cache_files(args.collision_cache, role="erase")
    groups = [
        ("collision", erase_paths[:31]),
        ("target_object", erase_paths[31:36]),
        ("generic", cache_files(args.generic_cache, role="preserve")),
        ("waterdrop", cache_files(args.waterdrop_cache, role="erase")),
    ]
    if args.other_ball_cache:
        groups.append(("other_ball", cache_files(args.other_ball_cache, role="erase")))
    if args.negation_cache:
        groups.append(("negation", cache_files(args.negation_cache, role="preserve")))
    payload = {
        "config": {
            "model": str(args.model),
            "layer": args.layer,
            "sigma": args.sigma,
            "tokens_per_sample": args.tokens_per_sample,
            "tokens_per_frame": args.tokens_per_frame,
            "background_per_frame": args.background_per_frame,
            "seed": args.seed,
        },
        "groups": {},
    }
    for offset, (name, paths) in enumerate(groups):
        payload["groups"][name] = extract_group(
            name=name,
            paths=paths,
            transformer=transformer,
            layer=args.layer,
            sigma_value=args.sigma,
            tokens_per_sample=args.tokens_per_sample,
            tokens_per_frame=args.tokens_per_frame,
            background_per_frame=args.background_per_frame,
            seed=args.seed + offset * 10000,
            device=device,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "output": str(args.output),
        "groups": {name: len(group["records"]) for name, group in payload["groups"].items()},
        **payload["config"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
