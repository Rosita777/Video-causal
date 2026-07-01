#!/usr/bin/env python3
"""Run a VideoEraser-style Wan baseline proxy."""

from __future__ import annotations

import argparse

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


BASELINE = "videoeraser"
LOCAL_METHOD = "spea_arng_wan_proxy_v0"


def implementation_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "local_method": LOCAL_METHOD,
        "spea_strength": args.spea_strength,
        "replacement_token": args.replacement_token,
        "notes": (
            "Wan proxy for VideoEraser-style inference: replace the target concept "
            "in the positive prompt, use the target as negative guidance, and push "
            "prompt embeddings away from the original concept-bearing prompt."
        ),
    }


def build_items(args: argparse.Namespace) -> list[dict[str, object]]:
    items = []
    for index, prompt_row in enumerate(selected_prompts(args)):
        seed = args.seed + index
        item = base_item(index, prompt_row, args.output_dir, seed)
        erased_prompt = erase_concept_from_prompt(
            str(item["prompt"]), str(item["target_concept"]), args.replacement_token
        )
        item["videoeraser"] = {
            "erased_prompt": erased_prompt,
            "negative_prompt": item["target_concept"],
            "spea_strength": args.spea_strength,
            "method": LOCAL_METHOD,
        }
        items.append(item)
    return items


def encode_item(pipe, torch_module, torch_dtype, selected_device: str, item: dict[str, object]):
    eraser = item["videoeraser"]
    prompt_embeds, negative_prompt_embeds = encode_cfg(
        pipe,
        torch_module,
        torch_dtype,
        prompt=str(eraser["erased_prompt"]),
        negative_prompt=str(eraser["negative_prompt"]),
        device=selected_device,
    )
    strength = float(eraser["spea_strength"])
    if strength:
        original_prompt_embeds, _ = encode_cfg(
            pipe,
            torch_module,
            torch_dtype,
            prompt=str(item["prompt"]),
            negative_prompt=str(eraser["negative_prompt"]),
            device=selected_device,
        )
        prompt_embeds = prompt_embeds + strength * (prompt_embeds - original_prompt_embeds)
    return prompt_embeds, negative_prompt_embeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_args(parser)
    parser.add_argument("--spea-strength", type=float, default=0.4)
    parser.add_argument("--replacement-token", default="object")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_args(parser, args)
    if args.spea_strength < 0:
        parser.error("--spea-strength must be non-negative")

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
