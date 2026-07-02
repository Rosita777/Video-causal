#!/usr/bin/env python3
"""Build the ZeroScope MVP-0 causal-chain probe manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE_NAME = "zeroscope_mvp0_causal_chain_probe"
PRIORITY_MECHANISMS = [
    "fluid_impact",
    "fracture_damage",
    "elastic_deformation",
    "particle_dispersion",
    "surface_trace",
    "field_mediated",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def leakage_pair_ids(labels_path: Path | None) -> set[str]:
    if labels_path is None:
        return set()
    return {
        row["pair_id"]
        for row in read_csv(labels_path)
        if row.get("final_label") == "strict_causal_footprint_leakage"
    }


def scene_context(item: dict[str, Any]) -> str:
    source = str(item.get("source_prompt", "")).strip()
    start_match = re.search(
        r"(The scene starts with .*?)(?:\. A clearly visible|\. The [^.]+ enters|\. [A-Z]|$)",
        source,
    )
    if start_match:
        prefix = source.split("The scene starts with", 1)[0].strip()
        start = start_match.group(1).strip()
        return f"{prefix} {start}.".strip()

    counterfactual = str(item.get("counterfactual_prompt", "")).strip()
    if counterfactual:
        sanitized = sanitize_context(counterfactual, item)
        if sanitized:
            return sanitized
    control = str(item.get("control_prompt", "")).strip()
    if control:
        sanitized = sanitize_context(control, item)
        if sanitized:
            return sanitized
    return source


def sanitize_context(text: str, item: dict[str, Any]) -> str:
    target = str(item.get("target_concept", "")).strip()
    footprint = str(item.get("causal_footprint") or item.get("expected_effect") or "").strip()
    sentences = [part.strip() for part in re.split(r"\.\s+", text.strip().rstrip(".")) if part.strip()]
    banned: list[str] = ["no visible cause", "other visible cause", "no impact point"]
    if target:
        banned.extend([f"no {target}", f"without {target}", f"{target} is present"])
    if footprint:
        banned.extend([f"no {footprint}", footprint])
    kept = [
        sentence
        for sentence in sentences
        if not any(needle.lower() in sentence.lower() for needle in banned)
    ]
    return ". ".join(kept).strip() + "." if kept else ""


def mechanism_phrase(item: dict[str, Any]) -> str:
    mechanism = str(item.get("mechanism_type", ""))
    target = str(item.get("target_concept", "target")).strip() or "target"
    if mechanism == "fluid_impact":
        return f"{target} impact with water"
    if mechanism == "fracture_damage":
        return f"{target} impact causing fracture"
    if mechanism == "surface_trace":
        return f"{target} contact leaving a surface trace"
    if mechanism == "elastic_deformation":
        return f"{target} collision causing elastic deformation"
    if mechanism == "field_mediated":
        return f"{target} field interaction"
    if mechanism == "particle_dispersion":
        return f"{target} collision dispersing particles"
    return f"{target} causal interaction"


def no_phrase(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return f"no {cleaned}"


def minimal_pairs_for(item: dict[str, Any]) -> dict[str, dict[str, str]]:
    context = scene_context(item).rstrip(".")
    target = str(item.get("target_concept", "")).strip()
    footprint = str(item.get("causal_footprint") or item.get("expected_effect") or "").strip()
    mechanism = mechanism_phrase(item)
    return {
        "cause": {
            "positive": f"{context}, with {target}.",
            "negative": f"{context}, without {target}.",
        },
        "mechanism": {
            "positive": f"{context}, with {mechanism}.",
            "negative": f"{context}, with no impact or causal disturbance.",
        },
        "footprint": {
            "positive": f"{context}, with {footprint}.",
            "negative": f"{context}, with {no_phrase(footprint)}.",
        },
    }


def compact_generation_prompt(item: dict[str, Any]) -> str:
    context = scene_context(item).rstrip(".")
    target = str(item.get("target_concept", "")).strip()
    footprint = str(item.get("causal_footprint") or item.get("expected_effect") or "").strip()
    if context and target and footprint:
        return re.sub(r"\s+", " ", f"{context}. {target} causes {footprint}.").strip()
    return str(item.get("source_prompt", "")).strip()


def priority_key(item: dict[str, Any], strict_pairs: set[str]) -> tuple[int, int, str]:
    pair_id = str(item.get("pair_id", ""))
    mechanism = str(item.get("mechanism_type", ""))
    strict_rank = 0 if pair_id in strict_pairs else 1
    mechanism_rank = (
        PRIORITY_MECHANISMS.index(mechanism)
        if mechanism in PRIORITY_MECHANISMS
        else len(PRIORITY_MECHANISMS)
    )
    return strict_rank, mechanism_rank, pair_id


def round_robin_by_mechanism(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=lambda item: priority_key(item, set()))
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in sorted_items:
        buckets.setdefault(str(item.get("mechanism_type", "")), []).append(item)

    mechanism_order = sorted(
        buckets,
        key=lambda mechanism: (
            PRIORITY_MECHANISMS.index(mechanism)
            if mechanism in PRIORITY_MECHANISMS
            else len(PRIORITY_MECHANISMS),
            mechanism,
        ),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets[mechanism] for mechanism in mechanism_order):
        for mechanism in mechanism_order:
            if buckets[mechanism]:
                selected.append(buckets[mechanism].pop(0))
                if len(selected) == limit:
                    break
    return selected


def select_items(items: list[dict[str, Any]], strict_pairs: set[str], limit: int) -> list[dict[str, Any]]:
    strict_items = [item for item in items if str(item.get("pair_id", "")) in strict_pairs]
    non_strict_items = [item for item in items if str(item.get("pair_id", "")) not in strict_pairs]
    selected = round_robin_by_mechanism(strict_items, limit)
    if len(selected) < limit:
        selected.extend(round_robin_by_mechanism(non_strict_items, limit - len(selected)))
    return selected


def manifest_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "probe_index": index,
        "pair_id": str(item.get("pair_id", "")),
        "source_index": str(item.get("source_index", "")),
        "slice_index": int(item.get("slice_index", index)),
        "mechanism_type": str(item.get("mechanism_type", "")),
        "target_concept": str(item.get("target_concept", "")),
        "causal_footprint": str(item.get("causal_footprint") or item.get("expected_effect") or ""),
        "source_prompt": str(item.get("source_prompt", "")),
        "generation_prompt": compact_generation_prompt(item),
        "counterfactual_prompt": str(item.get("counterfactual_prompt", "")),
        "control_prompt": str(item.get("control_prompt", "")),
        "clean_video_path": str(item.get("clean_video_path", "")),
        "minimal_pairs": minimal_pairs_for(item),
    }


def write_prompt_file(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_probe_prompts(output_dir: Path, items: list[dict[str, Any]]) -> None:
    prompts_dir = output_dir / "prompts"
    write_prompt_file(
        prompts_dir / "source_prompts.txt",
        [
            "# source prompts",
            "# Format: <prompt> | <target> | <expected_effect>",
            "",
            *[
                f"{item['generation_prompt']} | {item['target_concept']} | {item['causal_footprint']}"
                for item in items
            ],
        ],
    )
    write_prompt_file(
        prompts_dir / "counterfactual_prompts.txt",
        [
            "# counterfactual prompt-only controls",
            "# Format: <prompt> | <target> | <expected_effect>",
            "",
            *[
                f"{item['counterfactual_prompt']} | {item['target_concept']} | {item['causal_footprint']}"
                for item in items
            ],
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--clean-valid-manifest", type=Path, required=True)
    parser.add_argument("--baseline-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    data = read_json(args.clean_valid_manifest)
    items = data.get("items")
    if not isinstance(items, list):
        parser.exit(2, f"{args.clean_valid_manifest}: missing list field 'items'\n")

    strict_pairs = leakage_pair_ids(args.baseline_labels)
    selected = [
        manifest_item(item, index)
        for index, item in enumerate(select_items(items, strict_pairs, args.limit))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_probe_prompts(args.output_dir, selected)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": PROBE_NAME,
        "dry_run": args.dry_run,
        "source_manifest": str(args.clean_valid_manifest),
        "baseline_labels": str(args.baseline_labels) if args.baseline_labels else "",
        "count": len(selected),
        "items": selected,
    }
    out = args.output_dir / "probe_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} MVP-0 probe items to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
