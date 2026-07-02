#!/usr/bin/env python3
"""Build a Phase B paraphrase-averaged variant of the MVP-0 probe manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PHASE_B_PROBE_NAME = "zeroscope_mvp0_causal_chain_probe_phase_b_paraphrase"
ORTHOGONAL_SEMANTIC_PAIRS = [
    {
        "positive": "A realistic video with birds flying across the sky.",
        "negative": "A realistic video with no birds in the sky.",
    },
    {
        "positive": "A realistic video with a candle flame flickering on a table.",
        "negative": "A realistic video with no candle flame on the table.",
    },
    {
        "positive": "A realistic video with a red car driving along a road.",
        "negative": "A realistic video with no red car on the road.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.rstrip(".")


def context_from_generation_prompt(item: dict[str, Any]) -> str:
    prompt = normalize_sentence(str(item.get("generation_prompt") or item.get("source_prompt", "")))
    target = normalize_sentence(str(item.get("target_concept", "")))
    footprint = normalize_sentence(str(item.get("causal_footprint", "")))
    marker = f"{target} causes {footprint}"
    if marker and marker in prompt:
        prefix = prompt.split(marker, 1)[0].strip().rstrip(".")
        if prefix:
            return prefix
    first = prompt.split(".", 1)[0].strip()
    return first or prompt


def mechanism_variant(item: dict[str, Any]) -> str:
    target = normalize_sentence(str(item.get("target_concept", "target"))) or "target"
    mechanism = str(item.get("mechanism_type", ""))
    if mechanism == "fluid_impact":
        return f"{target} strikes the water surface"
    if mechanism == "fracture_damage":
        return f"{target} hits the mirror surface"
    if mechanism == "elastic_deformation":
        return f"{target} collides with the elastic net"
    if mechanism == "surface_trace":
        return f"{target} contacts the surface"
    if mechanism == "particle_dispersion":
        return f"{target} scatters nearby particles"
    if mechanism == "field_mediated":
        return f"{target} interacts through a visible field"
    return f"{target} creates a causal disturbance"


def original_pair(item: dict[str, Any], link: str) -> dict[str, str]:
    pair = item.get("minimal_pairs", {}).get(link)
    if isinstance(pair, list):
        return dict(pair[0])
    return dict(pair)


def cause_pairs(item: dict[str, Any]) -> list[dict[str, str]]:
    context = context_from_generation_prompt(item)
    target = normalize_sentence(str(item.get("target_concept", "target"))) or "target"
    return [
        original_pair(item, "cause"),
        {
            "positive": f"{context}, {target} is visible in the scene.",
            "negative": f"{context}, {target} is absent from the scene.",
        },
        {
            "positive": f"{context}, the scene includes {target}.",
            "negative": f"{context}, the scene includes no {target}.",
        },
    ]


def mechanism_pairs(item: dict[str, Any]) -> list[dict[str, str]]:
    context = context_from_generation_prompt(item)
    mechanism = mechanism_variant(item)
    return [
        original_pair(item, "mechanism"),
        {
            "positive": f"{context}, {mechanism}.",
            "negative": f"{context}, no impact or causal contact occurs.",
        },
        {
            "positive": f"{context}, the cause physically triggers the event.",
            "negative": f"{context}, the scene stays undisturbed with no causal trigger.",
        },
    ]


def footprint_pairs(item: dict[str, Any]) -> list[dict[str, str]]:
    context = context_from_generation_prompt(item)
    footprint = normalize_sentence(str(item.get("causal_footprint", "effect"))) or "effect"
    return [
        original_pair(item, "footprint"),
        {
            "positive": f"{context}, the downstream effect is visible: {footprint}.",
            "negative": f"{context}, the downstream effect is absent.",
        },
        {
            "positive": f"{context}, visible evidence shows {footprint}.",
            "negative": f"{context}, visible evidence shows no {footprint}.",
        },
    ]


def orthogonal_pairs() -> list[dict[str, str]]:
    return [dict(pair) for pair in ORTHOGONAL_SEMANTIC_PAIRS]


def expand_item(item: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(item)
    expanded["minimal_pairs"] = {
        "cause": cause_pairs(item),
        "mechanism": mechanism_pairs(item),
        "footprint": footprint_pairs(item),
        "orthogonal_semantic": orthogonal_pairs(),
    }
    expanded["phase_b_pair_count_per_link"] = 3
    return expanded


def build_phase_b_manifest(source: dict[str, Any]) -> dict[str, Any]:
    items = source.get("items")
    if not isinstance(items, list):
        raise ValueError("source manifest missing list field 'items'")
    return {
        **source,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": PHASE_B_PROBE_NAME,
        "source_probe_name": source.get("probe_name", ""),
        "phase_b_method": "paraphrase_averaged_minimal_pairs",
        "phase_b_control_method": "paraphrase_averaged_norm_matched_orthogonal",
        "phase_b_pair_count_per_link": 3,
        "items": [expand_item(item) for item in items],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_phase_b_manifest(read_json(args.probe_manifest))
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote Phase B paraphrase probe manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
