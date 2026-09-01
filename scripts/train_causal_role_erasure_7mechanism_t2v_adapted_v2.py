#!/usr/bin/env python3
"""Train one registry-bound T2VUnlearning-adapted CogVideoX operator.

The implementation intentionally keeps the project's disclosed paper-guided
adaptation: negatively guided velocity matching plus an adapter-localization
penalty, with no claim of official training-code parity.  Formal outputs are
not inputs to this trainer and only the single precommitted final checkpoint is
emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import build_causal_role_erasure_7mechanism_t2v_training_registry_v2 as contract


def require(condition: bool, message: str) -> None:
    if not condition:
        raise contract.TrainingRegistryError(message)


def sha256_file(path: Path) -> str:
    return contract.sha256_file(path)


def resolve_registered(project_root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def load_run_spec(
    project_root: Path,
    registry_path: Path,
    mechanism: str,
) -> tuple[dict[str, Any], Path, str]:
    require(registry_path.is_file() and not registry_path.is_symlink(), f"T2V registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    require(registry.get("protocol") == contract.PROTOCOL, "unexpected T2V registry protocol")
    require(registry.get("protocol_version") == contract.PROTOCOL_VERSION, "T2V protocol version changed")
    require(registry.get("status") == "t2v_training_specs_frozen_pre_training", "T2V registry is not pre-training frozen")
    trainer_relative = "scripts/train_causal_role_erasure_7mechanism_t2v_adapted_v2.py"
    trainer_path = project_root / trainer_relative
    require(trainer_relative in registry.get("code_sha256", {}), "T2V trainer is not code-bound")
    require(trainer_path.is_file() and not trainer_path.is_symlink(), f"T2V trainer missing: {trainer_path}")
    require(sha256_file(trainer_path) == registry["code_sha256"][trainer_relative], "T2V trainer changed after registry freeze")
    matches = [run for run in registry.get("runs", []) if run.get("mechanism") == mechanism]
    require(len(matches) == 1, f"expected one T2V run for {mechanism}")
    ref = matches[0]
    spec_path = resolve_registered(project_root, ref["run_spec"])
    require(spec_path.is_file() and not spec_path.is_symlink(), f"T2V run spec missing: {spec_path}")
    spec_sha = sha256_file(spec_path)
    require(spec_sha == ref["run_spec_sha256"], "T2V run spec changed after registration")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    require(spec.get("protocol") == contract.PROTOCOL, "unexpected T2V run-spec protocol")
    require(spec.get("mechanism") == mechanism, "T2V run-spec mechanism mismatch")
    require(spec.get("eligible_checkpoint") == ref.get("eligible_checkpoint"), "eligible checkpoint reference mismatch")
    rows_path = resolve_registered(project_root, spec["training_rows"]["path"])
    require(sha256_file(rows_path) == spec["training_rows"]["sha256"], "T2V training rows changed")
    model_root = resolve_registered(project_root, spec["model_root"])
    inventory_path = resolve_registered(project_root, spec["model_inventory"]["path"])
    contract.load_model_inventory(model_root, inventory_path)
    current_python = Path(sys.executable).resolve()
    registered_python = resolve_registered(project_root, spec["runtime_python"])
    require(current_python == registered_python, f"wrong runtime Python: {current_python} != {registered_python}")
    require(sha256_file(current_python) == spec["runtime_python_sha256"], "runtime Python changed")
    return spec, rows_path, spec_sha


def read_prompts(rows_path: Path, spec: Mapping[str, Any]) -> list[str]:
    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(spec["training_rows"]["row_count"])
    require(len(rows) == expected, "T2V training row count differs from run spec")
    require([int(row["training_row_index"]) for row in rows] == list(range(expected)), "T2V training row order changed")
    require(all(row["mechanism"] == spec["mechanism"] for row in rows), "T2V training row mechanism mismatch")
    require(len({row["candidate_id"] for row in rows}) == expected, "T2V candidate IDs repeat")
    prompts = [row["factual_prompt"] for row in rows]
    require(all(prompt.strip() for prompt in prompts), "T2V training prompt is empty")
    return prompts


def train(project_root: Path, spec: Mapping[str, Any], rows_path: Path, spec_sha: str) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention import Attention
    from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock

    parameters = spec["parameters"]
    rank = int(parameters["rank"])
    learning_rate = float(parameters["learning_rate"])
    max_steps = int(parameters["max_steps"])
    negative_scale = float(parameters["negative_scale"])
    localization_weight = float(parameters["localization_weight"])
    seed = int(parameters["training_seed"])
    checkpoint_step = int(parameters["eligible_checkpoint_step"])
    require(checkpoint_step == max_steps, "eligible T2V checkpoint must be final step")
    prompts = read_prompts(rows_path, spec)
    model_root = resolve_registered(project_root, spec["model_root"])
    output_dir = resolve_registered(project_root, spec["output_dir"])
    require(not output_dir.exists(), f"fresh-only T2V output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    pipe = CogVideoXPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
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
        def __init__(self, dim: int, adapter_rank: int):
            super().__init__()
            self.down = nn.Linear(dim, adapter_rank)
            self.act = nn.GELU()
            self.up = nn.Linear(adapter_rank, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, hidden_states):
            dtype = hidden_states.dtype
            return self.up(self.act(self.down(hidden_states.float()))).to(dtype)

    class CogVideoXWithEraser(nn.Module):
        def __init__(self, attention: Attention, adapter_rank: int):
            super().__init__()
            self.attn = attention
            self.adapter = AdapterEraser(attention.to_v.weight.shape[-1], adapter_rank)
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
        require(isinstance(block, CogVideoXBlock), f"unexpected CogVideoX block: {type(block)}")
        block.attn1 = CogVideoXWithEraser(block.attn1, rank)
        erasers.append(block.attn1)
    transformer.to(device=device, dtype=torch.bfloat16)
    for eraser in erasers:
        eraser.adapter.to(device=device, dtype=torch.float32)
    trainable = [parameter for eraser in erasers for parameter in eraser.adapter.parameters()]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    latent_frames = (int(parameters["num_frames"]) - 1) // 4 + 1
    latent_height = int(parameters["height"]) // 8
    latent_width = int(parameters["width"]) // 8
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []

    def set_enabled(enabled: bool) -> None:
        for eraser in erasers:
            eraser.use_eraser = enabled

    try:
        transformer.train()
        for step in range(1, max_steps + 1):
            prompt = prompts[(step - 1) % len(prompts)]
            prompt_embeds = prompt_cache[prompt].to(device=device, dtype=torch.bfloat16)
            uncond_embeds = prompt_cache[""].to(device=device, dtype=torch.bfloat16)
            noisy = torch.randn(
                (1, latent_frames, transformer.config.in_channels, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            timesteps = torch.randint(0, 1000, (1,), generator=generator, device=device, dtype=torch.long)
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
                negative_velocity = velocity_uncond - negative_scale * (
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
            loss = unlearn_loss + localization_weight * localization_loss
            require(bool(torch.isfinite(loss).item()), f"non-finite T2V loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "unlearn": float(unlearn_loss.detach()),
                "localization": float(localization_loss.detach()),
            }
            require(all(math.isfinite(value) for key, value in record.items() if key != "step"), f"non-finite T2V record at step {step}")
            losses.append(record)
            print(
                f"step={step}/{max_steps} loss={record['loss']:.6f} "
                f"unlearn={record['unlearn']:.6f} loc={record['localization']:.6f}",
                flush=True,
            )
        checkpoint = stage / f"checkpoint-{checkpoint_step:06d}"
        checkpoint.mkdir(parents=True)
        weights = {
            f"transformer_blocks.{index}.attn1.adapter": eraser.adapter.state_dict()
            for index, eraser in enumerate(erasers)
        }
        weights_path = checkpoint / "eraser_weights.pt"
        torch.save(weights, weights_path)
        config = {
            "eraser_type": "adapter",
            "eraser_rank": rank,
            "adapter_target": "every CogVideoX transformer block attn1",
            "label": "T2VUnlearning-adapted (ours)",
        }
        config_path = checkpoint / "eraser_config.json"
        config_path.write_bytes(contract.canonical_json_bytes(config))
        state = {
            "step": checkpoint_step,
            "losses_last20": losses[-20:],
            "run_spec_sha256": spec_sha,
            "training_rows_sha256": spec["training_rows"]["sha256"],
        }
        state_path = checkpoint / "training_state.json"
        state_path.write_bytes(contract.canonical_json_bytes(state))
        receipt = {
            "protocol": contract.PROTOCOL,
            "protocol_version": contract.PROTOCOL_VERSION,
            "status": "eligible",
            "run_id": spec["run_id"],
            "mechanism": spec["mechanism"],
            "step": checkpoint_step,
            "run_spec_sha256": spec_sha,
            "training_rows_sha256": spec["training_rows"]["sha256"],
            "weights_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "training_state_sha256": sha256_file(state_path),
            "formal_output_inspected": False,
        }
        (checkpoint / "training_receipt.json").write_bytes(contract.canonical_json_bytes(receipt))
        os.replace(stage, output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--training-registry", type=Path, required=True)
    parser.add_argument("--mechanism", choices=contract.MECHANISMS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    registry_path = args.training_registry if args.training_registry.is_absolute() else project_root / args.training_registry
    spec, rows_path, spec_sha = load_run_spec(project_root, registry_path.resolve(), args.mechanism)
    prompts = read_prompts(rows_path, spec)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_validated",
                    "mechanism": args.mechanism,
                    "training_rows": len(prompts),
                    "max_steps": spec["parameters"]["max_steps"],
                    "eligible_checkpoint": spec["eligible_checkpoint"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = train(project_root, spec, rows_path, spec_sha)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
