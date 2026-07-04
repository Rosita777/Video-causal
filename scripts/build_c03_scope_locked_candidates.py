#!/usr/bin/env python3
"""Build the pre-registered C0.3 scope-locked candidate manifest."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCOPE = "low_entanglement_rigid_object_surface_trace"
PREDICATES = [
    "rigid_or_tool_like_target",
    "simple_static_surface",
    "localized_persistent_footprint",
    "footprint_only_plausible",
    "target_footprint_text_separable",
    "stable_background_and_camera",
]
FACTORIAL_CELLS = {
    "original": {"target_visible": "yes", "footprint_visible": "yes"},
    "remove_target": {"target_visible": "no", "footprint_visible": "no"},
    "footprint_only": {"target_visible": "no", "footprint_visible": "yes"},
    "target_only": {"target_visible": "yes", "footprint_visible": "no"},
}
FIELD_NOTES = {
    "causal_footprint_absence_phrase": (
        "Grammar-safe noun phrase used after 'no' in absence prompts; "
        "not a factorial cell or a distinct experimental condition."
    ),
}

CANDIDATES = [
    (
        "metal comb",
        "smooth sand tray",
        "parallel grooves in the sand",
        "parallel grooves in the sand",
    ),
    (
        "toy car",
        "soft clay slab",
        "two tire tracks in the soft clay",
        "tire tracks in the soft clay",
    ),
    (
        "rubber stamp",
        "blank paper sheet",
        "a square ink stamp mark on the paper",
        "square ink stamp mark on the paper",
    ),
    (
        "piece of chalk",
        "clean blackboard",
        "a white chalk line on the blackboard",
        "white chalk line on the blackboard",
    ),
    (
        "wooden block",
        "smooth clay slab",
        "a square imprint in the clay",
        "square imprint in the clay",
    ),
    (
        "hiking boot",
        "wet sand patch",
        "a single boot print in the wet sand",
        "boot print in the wet sand",
    ),
    (
        "paint roller",
        "white paper strip",
        "a blue paint stripe on the paper",
        "blue paint stripe on the paper",
    ),
    (
        "wooden stylus",
        "smooth wax tablet",
        "a thin carved line in the wax",
        "thin carved line in the wax",
    ),
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


def pair_id_for(index: int, target: str, footprint: str) -> str:
    return f"c03_surface_trace_{index:02d}_{slugify(target)}_{slugify(footprint)}"


def item_row(
    index: int,
    target: str,
    surface: str,
    footprint: str,
    absence_footprint: str,
) -> dict[str, object]:
    pair_id = pair_id_for(index, target, footprint)
    source_prompt = (
        f"A realistic fixed-camera close-up video of a {target} contacting a "
        f"{surface}, leaving {footprint}."
    )
    return {
        "probe_index": index,
        "slice_index": index,
        "source_index": f"c03_{index:02d}",
        "pair_id": pair_id,
        "mechanism_type": "scope_locked_surface_trace",
        "target_concept": target,
        "surface_or_object": surface,
        "causal_footprint": footprint,
        "causal_footprint_absence": absence_footprint,
        "causal_footprint_absence_phrase": absence_footprint,
        "source_prompt": source_prompt,
        "generation_prompt": source_prompt,
        "counterfactual_prompt": (
            f"A realistic fixed-camera close-up video of a clean {surface}. "
            f"No {target} is present. The scene shows no {absence_footprint}."
        ),
        "control_prompt": (
            f"A realistic fixed-camera close-up video of a {surface} with "
            f"{footprint}. No {target} is present and no visible cause appears "
            "in the frame."
        ),
        "prompt_template_id": "c03_scope_locked",
        "prior_seen": False,
        "candidate_scope": SCOPE,
        "scope_predicates_met": list(PREDICATES),
    }


def build_manifest() -> dict[str, object]:
    items = [item_row(index, *candidate) for index, candidate in enumerate(CANDIDATES)]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_name": "c03_scope_locked_surface_trace_candidates",
        "dry_run": False,
        "candidate_scope": SCOPE,
        "factorial_cells": FACTORIAL_CELLS,
        "field_notes": FIELD_NOTES,
        "count": len(items),
        "items": items,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest['count']} C0.3 candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
