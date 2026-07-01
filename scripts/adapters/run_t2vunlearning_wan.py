#!/usr/bin/env python3
"""Run a T2VUnlearning-style Wan baseline proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

from wan_adapter_common import (
    add_common_args,
    base_item,
    encode_cfg,
    erase_concept_from_prompt,
    generate_encoded_videos,
    generation_config,
    selected_prompts,
    validate_common_args,
    write_manifest,
)


BASELINE = "t2vunlearning"
LOCAL_METHOD = "receler_wan_proxy_v0"


def implementation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "local_method": LOCAL_METHOD,
        "eraser_path": str(args.eraser_path) if args.eraser_path else None,
        "eraser_path_present": bool(args.eraser_path and args.eraser_path.exists()),
        "eraser_rank": args.eraser_rank,
        "unlearn_strength": args.unlearn_strength,
        "replacement_token": args.replacement_token,
        "notes": (
            "Wan proxy for the public T2VUnlearning inference contract. When no "
            "trained adapter is available, it suppresses the target with an erased "
            "prompt, target negative guidance, and Receler-style prompt embedding "
            "displacement."
        ),
    }


def build_items(args: argparse.Namespace) -> list[dict[str, object]]:
    items = []
    for index, prompt_row in enumerate(selected_prompts(args)):
        seed = args.seed + index
        item = base_item(index, prompt_row, args.output_dir, seed)
        unlearn_prompt = erase_concept_from_prompt(
            str(item["prompt"]), str(item["target_concept"]), args.replacement_token
        )
        item["t2vunlearning"] = {
            "unlearn_concept": item["target_concept"],
            "unlearn_prompt": unlearn_prompt,
            "negative_prompt": item["target_concept"],
            "unlearn_strength": args.unlearn_strength,
            "eraser_rank": args.eraser_rank,
            "method": LOCAL_METHOD,
        }
        items.append(item)
    return items


def encode_item(pipe, torch_module, torch_dtype, selected_device: str, item: dict[str, object]):
    unlearn = item["t2vunlearning"]
    prompt_embeds, negative_prompt_embeds = encode_cfg(
        pipe,
        torch_module,
        torch_dtype,
        prompt=str(unlearn["unlearn_prompt"]),
        negative_prompt=str(unlearn["negative_prompt"]),
        device=selected_device,
    )
    strength = float(unlearn["unlearn_strength"])
    if strength:
        original_prompt_embeds, _ = encode_cfg(
            pipe,
            torch_module,
            torch_dtype,
            prompt=str(item["prompt"]),
            negative_prompt=str(unlearn["negative_prompt"]),
            device=selected_device,
        )
        prompt_embeds = prompt_embeds + strength * (prompt_embeds - original_prompt_embeds)
    return prompt_embeds, negative_prompt_embeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_args(parser)
    parser.add_argument("--eraser-path", type=Path)
    parser.add_argument("--eraser-rank", type=int, default=128)
    parser.add_argument("--unlearn-strength", type=float, default=0.35)
    parser.add_argument("--replacement-token", default="object")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_args(parser, args)
    if args.eraser_rank <= 0:
        parser.error("--eraser-rank must be positive")
    if args.unlearn_strength < 0:
        parser.error("--unlearn-strength must be non-negative")

    items = build_items(args)
    if not args.dry_run:
        generate_encoded_videos(args, items, encode_item)

    manifest = write_manifest(
        output_dir=args.output_dir,
        baseline=BASELINE,
        model=args.model,
        prompts_path=args.prompts,
        generation=generation_config(args),
        implementation=implementation_config(args),
        items=items,
        dry_run=args.dry_run,
    )
    print(f"Wan {BASELINE} manifest written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
