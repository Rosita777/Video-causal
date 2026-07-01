#!/usr/bin/env python3
"""Run a SAFREE-style Wan inference-time erasure proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

from wan_adapter_common import (
    add_common_args,
    base_item,
    encode_cfg,
    generate_encoded_videos,
    generation_config,
    selected_prompts,
    validate_common_args,
    write_manifest,
)


BASELINE = "safree_wan"
LOCAL_METHOD = "safree_wan_proxy_v0"


def implementation_config(args: argparse.Namespace) -> dict[str, object]:
    official_path = args.safree_root / "pipeline.py"
    return {
        "local_method": LOCAL_METHOD,
        "official_external_available": official_path.exists(),
        "safree_root": str(args.safree_root),
        "concept_strength": args.concept_strength,
        "notes": (
            "No official SAFREE Wan adapter is present locally. This proxy keeps "
            "the original prompt, uses the target concept as negative guidance, "
            "and subtracts a target-concept embedding direction from the positive "
            "prompt embedding."
        ),
    }


def build_items(args: argparse.Namespace) -> list[dict[str, object]]:
    items = []
    for index, prompt_row in enumerate(selected_prompts(args)):
        seed = args.seed + index
        item = base_item(index, prompt_row, args.output_dir, seed)
        item["safree"] = {
            "unsafe_concept": item["target_concept"],
            "negative_prompt": item["target_concept"],
            "concept_strength": args.concept_strength,
            "method": "concept_direction_subtraction",
        }
        items.append(item)
    return items


def encode_item(pipe, torch_module, torch_dtype, selected_device: str, item: dict[str, object]):
    safree = item["safree"]
    prompt_embeds, negative_prompt_embeds = encode_cfg(
        pipe,
        torch_module,
        torch_dtype,
        prompt=str(item["prompt"]),
        negative_prompt=str(safree["negative_prompt"]),
        device=selected_device,
    )
    strength = float(safree["concept_strength"])
    if strength:
        concept_embeds, _ = encode_cfg(
            pipe,
            torch_module,
            torch_dtype,
            prompt=str(safree["unsafe_concept"]),
            negative_prompt="",
            device=selected_device,
        )
        prompt_embeds = prompt_embeds - strength * concept_embeds
    return prompt_embeds, negative_prompt_embeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_args(parser)
    parser.add_argument("--safree-root", type=Path, default=Path("baselines/external/SAFREE"))
    parser.add_argument("--concept-strength", type=float, default=0.25)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_common_args(parser, args)
    if args.concept_strength < 0:
        parser.error("--concept-strength must be non-negative")

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
