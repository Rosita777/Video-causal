#!/usr/bin/env python3
"""Paper-guided CogVideoX adaptation of T2VUnlearning training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--negative-scale", type=float, default=7.0)
    parser.add_argument("--localization-weight", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12000)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention import Attention
    from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock

    rows = [
        row
        for row in csv.DictReader(args.manifest.open(newline="", encoding="utf-8"))
        if row["mechanism"] == args.mechanism and row["training_role"] == "erase"
    ]
    if len(rows) != 36:
        raise ValueError(f"Expected 36 erase prompts for {args.mechanism}, found {len(rows)}")
    prompts = [row["prompt"] for row in rows]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    pipe = CogVideoXPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    prompt_cache = {}
    for prompt in ["", *prompts]:
        embeddings, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            device=device,
            dtype=torch.bfloat16,
        )
        prompt_cache[prompt] = embeddings.detach().cpu()
    transformer = pipe.transformer
    pipe.transformer = None
    del pipe
    torch.cuda.empty_cache()

    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()

    class AdapterEraser(nn.Module):
        def __init__(self, dim: int, rank: int):
            super().__init__()
            self.down = nn.Linear(dim, rank)
            self.act = nn.GELU()
            self.up = nn.Linear(rank, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, hidden_states):
            dtype = hidden_states.dtype
            return self.up(self.act(self.down(hidden_states.float()))).to(dtype)

    class CogVideoXWithEraser(nn.Module):
        def __init__(self, attention: Attention, rank: int):
            super().__init__()
            self.attn = attention
            self.adapter = AdapterEraser(attention.to_v.weight.shape[-1], rank)
            self.use_eraser = True
            self.last_output = None

        def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            hidden_states, encoder_hidden_states = self.attn(
                hidden_states, encoder_hidden_states, attention_mask, **kwargs
            )
            self.last_output = None
            if self.use_eraser:
                residual = self.adapter(hidden_states)
                self.last_output = residual
                hidden_states = hidden_states + residual
            return hidden_states, encoder_hidden_states

    erasers = []
    for block in transformer.transformer_blocks:
        if not isinstance(block, CogVideoXBlock):
            raise TypeError(f"Unexpected transformer block: {type(block)}")
        block.attn1 = CogVideoXWithEraser(block.attn1, args.rank)
        erasers.append(block.attn1)
    transformer.to(device=device, dtype=torch.bfloat16)
    for eraser in erasers:
        eraser.adapter.to(device=device, dtype=torch.float32)
    parameters = [parameter for eraser in erasers for parameter in eraser.adapter.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_height = args.height // 8
    latent_width = args.width // 8
    generator = torch.Generator(device=device).manual_seed(args.seed)
    losses = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def set_enabled(enabled: bool) -> None:
        for eraser in erasers:
            eraser.use_eraser = enabled

    def save(step: int) -> None:
        checkpoint = args.output_dir / f"checkpoint-{step:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        weights = {
            f"transformer_blocks.{index}.attn1.adapter": eraser.adapter.state_dict()
            for index, eraser in enumerate(erasers)
        }
        torch.save(weights, checkpoint / "eraser_weights.pt")
        (checkpoint / "eraser_config.json").write_text(
            json.dumps({"eraser_type": "adapter", "eraser_rank": args.rank}, indent=2) + "\n"
        )
        (checkpoint / "training_state.json").write_text(
            json.dumps({"step": step, "losses": losses[-20:], "config": vars(args)}, indent=2, default=str) + "\n"
        )

    transformer.train()
    for step in range(1, args.max_steps + 1):
        prompt = prompts[(step - 1) % len(prompts)]
        prompt_embeds = prompt_cache[prompt].to(device=device, dtype=torch.bfloat16)
        uncond_embeds = prompt_cache[""].to(device=device, dtype=torch.bfloat16)
        noisy = torch.randn(
            (1, latent_frames, transformer.config.in_channels, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        timesteps = torch.randint(
            0, 1000, (1,), generator=generator, device=device, dtype=torch.long
        )
        set_enabled(False)
        with torch.no_grad():
            velocity_uncond = transformer(
                hidden_states=noisy,
                encoder_hidden_states=uncond_embeds,
                timestep=timesteps,
                return_dict=False,
            )[0]
            velocity_target = transformer(
                hidden_states=noisy,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                return_dict=False,
            )[0]
            negative_velocity = velocity_uncond - args.negative_scale * (
                velocity_target - velocity_uncond
            )
        set_enabled(True)
        prediction = transformer(
            hidden_states=noisy,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            return_dict=False,
        )[0]
        unlearn_loss = F.mse_loss(prediction.float(), negative_velocity.float())
        localization_loss = torch.stack(
            [eraser.last_output.float().square().mean() for eraser in erasers]
        ).mean()
        loss = unlearn_loss + args.localization_weight * localization_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "unlearn": float(unlearn_loss.detach()),
            "localization": float(localization_loss.detach()),
        }
        losses.append(record)
        print(
            f"step={step}/{args.max_steps} loss={record['loss']:.6f} "
            f"unlearn={record['unlearn']:.6f} loc={record['localization']:.6f}",
            flush=True,
        )
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
