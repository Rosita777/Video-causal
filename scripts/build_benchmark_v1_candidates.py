#!/usr/bin/env python3
"""Build the causal-footprint benchmark-v1 candidate pool and controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_V0_ITEMS = Path("benchmarks/causal_footprint_v0/items.jsonl")
DEFAULT_ROUND6 = Path("benchmarks/causal_footprint_v0/round6_taxonomy_expansion_prompts.tsv")
DEFAULT_LEGACY = Path("benchmarks/causal_footprint_v0/candidate_pairs.tsv")
DEFAULT_OUTPUT_DIR = Path("benchmarks/causal_footprint_v1")
DEFAULT_PROMPT_DIR = Path("prompts")
DEFAULT_TARGET_COUNT = 150

SUPPLEMENTAL_LEGACY_PAIR_IDS = [
    "surface_trace_tire_mud_002",
    "surface_trace_hand_clay_003",
    "fracture_damage_hammer_tile_002",
    "fracture_damage_dropped_cup_floor_004",
    "fluid_impact_ink_droplet_glass_004",
    "elastic_deformation_bowling_ball_mattress_003",
]

TSV_COLUMNS = [
    "pair_id",
    "target_concept",
    "causal_footprint",
    "mechanism_type",
    "temporal_type",
    "exclusivity_score",
    "counterfactual_clarity",
    "generatability_score",
    "erasure_targetability",
    "status",
    "pair_source",
    "causal_chain",
    "source_prompt",
    "counterfactual_prompt",
    "control_prompt",
    "notes",
]

TEMPORAL_DEFAULTS = {
    "fluid_impact": "delayed",
    "fracture_damage": "persistent",
    "elastic_deformation": "synchronous",
    "surface_trace": "persistent",
    "field_mediated": "delayed",
    "particle_dispersion": "delayed",
}

ALTERNATIVE_CAUSES = {
    "fluid_impact": "a small visible bead",
    "fracture_damage": "a small metal hammer",
    "elastic_deformation": "a gloved hand",
    "surface_trace": "a smooth wooden block",
    "field_mediated": "a thin wooden stick",
    "particle_dispersion": "a small paper cup",
}


@dataclass(frozen=True)
class Candidate:
    pair_id: str
    target_concept: str
    causal_footprint: str
    mechanism_type: str
    temporal_type: str
    exclusivity_score: int
    counterfactual_clarity: int
    generatability_score: int
    erasure_targetability: int
    source_collection: str
    source_item_id: str
    pair_source: str
    causal_chain: str
    source_prompt: str
    counterfactual_prompt: str
    control_prompt: str
    notes: str


def clean(value: Any) -> str:
    return str(value or "").strip()


def default_temporal_type(mechanism_type: str) -> str:
    return TEMPORAL_DEFAULTS.get(mechanism_type, "delayed")


def generic_causal_chain(target: str, footprint: str) -> str:
    return f"{target} appears and acts on the scene -> {footprint}"


def no_cause_prompt(target: str, footprint: str) -> str:
    return (
        "A realistic fixed-camera close-up video of the same scene before the event, "
        f"with no {target}, no visible cause, and no {footprint}."
    )


def effect_only_prompt(target: str, footprint: str) -> str:
    return (
        "A realistic fixed-camera close-up video showing "
        f"{footprint}, with no {target} or other visible cause in the frame."
    )


def alternative_cause_prompt(mechanism: str, target: str, footprint: str) -> str:
    cause = ALTERNATIVE_CAUSES.get(mechanism, "a different visible object")
    return (
        "A realistic fixed-camera close-up video of "
        f"{cause} producing a similar footprint: {footprint}, with no {target} visible."
    )


def candidate_from_v0(item: dict[str, Any]) -> Candidate:
    mechanism = clean(item.get("mechanism_type"))
    target = clean(item.get("target_concept"))
    footprint = clean(item.get("expected_effect"))
    pair_id = clean(item.get("pair_id"))
    return Candidate(
        pair_id=pair_id,
        target_concept=target,
        causal_footprint=footprint,
        mechanism_type=mechanism,
        temporal_type=clean(item.get("temporal_type")) or default_temporal_type(mechanism),
        exclusivity_score=5,
        counterfactual_clarity=5,
        generatability_score=5,
        erasure_targetability=5,
        source_collection="causal_footprint_v0",
        source_item_id=clean(item.get("item_id")),
        pair_source="clean_source_gated_v0",
        causal_chain=generic_causal_chain(target, footprint),
        source_prompt=clean(item.get("source_prompt")),
        counterfactual_prompt=clean(item.get("counterfactual_prompt")) or no_cause_prompt(target, footprint),
        control_prompt=clean(item.get("control_prompt")) or effect_only_prompt(target, footprint),
        notes=f"Imported from benchmark-v0 item {clean(item.get('item_id'))}.",
    )


def candidate_from_round6(row: dict[str, str]) -> Candidate:
    mechanism = clean(row.get("mechanism_type"))
    target = clean(row.get("target_concept"))
    footprint = clean(row.get("causal_footprint"))
    pair_id = clean(row.get("round6_id"))
    return Candidate(
        pair_id=pair_id,
        target_concept=target,
        causal_footprint=footprint,
        mechanism_type=mechanism,
        temporal_type=default_temporal_type(mechanism),
        exclusivity_score=4,
        counterfactual_clarity=4,
        generatability_score=4,
        erasure_targetability=4,
        source_collection="round6_taxonomy_pool",
        source_item_id=pair_id,
        pair_source="round6_taxonomy_expansion",
        causal_chain=generic_causal_chain(target, footprint),
        source_prompt=clean(row.get("source_prompt")),
        counterfactual_prompt=no_cause_prompt(target, footprint),
        control_prompt=effect_only_prompt(target, footprint),
        notes=clean(row.get("notes")),
    )


def candidate_from_legacy(row: dict[str, str]) -> Candidate:
    mechanism = clean(row.get("mechanism_type"))
    target = clean(row.get("target_concept"))
    footprint = clean(row.get("causal_footprint"))
    return Candidate(
        pair_id=clean(row.get("pair_id")),
        target_concept=target,
        causal_footprint=footprint,
        mechanism_type=mechanism,
        temporal_type=clean(row.get("temporal_type")) or default_temporal_type(mechanism),
        exclusivity_score=int(clean(row.get("exclusivity_score")) or 4),
        counterfactual_clarity=int(clean(row.get("counterfactual_clarity")) or 4),
        generatability_score=int(clean(row.get("generatability_score")) or 4),
        erasure_targetability=int(clean(row.get("erasure_targetability")) or 4),
        source_collection="legacy_candidate_pairs",
        source_item_id=clean(row.get("pair_id")),
        pair_source=clean(row.get("pair_source")) or "legacy_candidate_pairs",
        causal_chain=clean(row.get("causal_chain")) or generic_causal_chain(target, footprint),
        source_prompt=clean(row.get("source_prompt")),
        counterfactual_prompt=clean(row.get("counterfactual_prompt")) or no_cause_prompt(target, footprint),
        control_prompt=clean(row.get("control_prompt")) or effect_only_prompt(target, footprint),
        notes=clean(row.get("notes")),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        return list(reader)


def add_unique(target: list[Candidate], seen: set[str], candidates: list[Candidate]) -> None:
    for candidate in candidates:
        if not candidate.pair_id or candidate.pair_id in seen:
            continue
        target.append(candidate)
        seen.add(candidate.pair_id)


def build_candidates(
    *,
    v0_items: Path,
    round6_candidates: Path,
    legacy_candidates: Path,
    target_count: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    seen: set[str] = set()

    add_unique(selected, seen, [candidate_from_v0(row) for row in read_jsonl(v0_items)])
    add_unique(selected, seen, [candidate_from_round6(row) for row in read_tsv(round6_candidates)])

    legacy_rows = [
        row
        for row in read_tsv(legacy_candidates)
        if clean(row.get("status")).lower() != "rejected"
        and clean(row.get("mechanism_type")) != "agent_or_object_response"
    ]
    legacy_by_pair_id = {clean(row.get("pair_id")): row for row in legacy_rows}
    preferred = [
        candidate_from_legacy(legacy_by_pair_id[pair_id])
        for pair_id in SUPPLEMENTAL_LEGACY_PAIR_IDS
        if pair_id in legacy_by_pair_id
    ]
    fallback = [
        candidate_from_legacy(row)
        for row in legacy_rows
        if clean(row.get("pair_id")) not in SUPPLEMENTAL_LEGACY_PAIR_IDS
    ]
    add_unique(selected, seen, preferred)
    add_unique(selected, seen, fallback)

    if len(selected) < target_count:
        raise ValueError(f"only {len(selected)} unique candidate rows available; requested {target_count}")
    return selected[:target_count]


def candidate_to_json(candidate: Candidate, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "item_id": f"v1:{candidate.pair_id}",
        "pair_id": candidate.pair_id,
        "source_collection": candidate.source_collection,
        "source_item_id": candidate.source_item_id,
        "pair_source": candidate.pair_source,
        "mechanism_type": candidate.mechanism_type,
        "temporal_type": candidate.temporal_type,
        "target_concept": candidate.target_concept,
        "expected_effect": candidate.causal_footprint,
        "causal_footprint": candidate.causal_footprint,
        "causal_chain": candidate.causal_chain,
        "source_prompt": candidate.source_prompt,
        "counterfactual_prompt": candidate.counterfactual_prompt,
        "control_prompt": candidate.control_prompt,
        "candidate_status": "candidate_v1",
        "scores": {
            "exclusivity_score": candidate.exclusivity_score,
            "counterfactual_clarity": candidate.counterfactual_clarity,
            "generatability_score": candidate.generatability_score,
            "erasure_targetability": candidate.erasure_targetability,
        },
        "notes": candidate.notes,
    }


def candidate_to_tsv_row(candidate: Candidate) -> dict[str, str]:
    return {
        "pair_id": candidate.pair_id,
        "target_concept": candidate.target_concept,
        "causal_footprint": candidate.causal_footprint,
        "mechanism_type": candidate.mechanism_type,
        "temporal_type": candidate.temporal_type,
        "exclusivity_score": str(candidate.exclusivity_score),
        "counterfactual_clarity": str(candidate.counterfactual_clarity),
        "generatability_score": str(candidate.generatability_score),
        "erasure_targetability": str(candidate.erasure_targetability),
        "status": "candidate_v1",
        "pair_source": candidate.pair_source,
        "causal_chain": candidate.causal_chain,
        "source_prompt": candidate.source_prompt,
        "counterfactual_prompt": candidate.counterfactual_prompt,
        "control_prompt": candidate.control_prompt,
        "notes": candidate.notes,
    }


def prompt_line(candidate: Candidate) -> str:
    return f"{candidate.source_prompt} | {candidate.target_concept} | {candidate.causal_footprint}"


def control_rows(candidate: Candidate) -> list[dict[str, str]]:
    base = {
        "source_name": "causal_footprint_v1",
        "source_pair_id": candidate.pair_id,
        "source_baseline": "candidate_v1_clean_screening",
        "mechanism_type": candidate.mechanism_type,
        "target_concept": candidate.target_concept,
        "expected_effect": candidate.causal_footprint,
    }
    return [
        {
            **base,
            "control_id": f"{candidate.pair_id}__no_cause",
            "control_type": "no_cause",
            "prompt": no_cause_prompt(candidate.target_concept, candidate.causal_footprint),
            "purpose": "Verify that the model does not add the target or footprint when both are absent.",
            "expected_target_presence": "no",
            "expected_footprint_presence": "no",
            "expected_alternative_cause_presence": "no",
        },
        {
            **base,
            "control_id": f"{candidate.pair_id}__effect_only",
            "control_type": "effect_only",
            "prompt": effect_only_prompt(candidate.target_concept, candidate.causal_footprint),
            "purpose": "Measure the model prior for the footprint when the target cause is absent.",
            "expected_target_presence": "no",
            "expected_footprint_presence": "yes",
            "expected_alternative_cause_presence": "no",
        },
        {
            **base,
            "control_id": f"{candidate.pair_id}__alternative_cause",
            "control_type": "alternative_cause",
            "prompt": alternative_cause_prompt(
                candidate.mechanism_type,
                candidate.target_concept,
                candidate.causal_footprint,
            ),
            "purpose": "Check whether a different visible cause can explain a similar footprint.",
            "expected_target_presence": "no",
            "expected_footprint_presence": "yes",
            "expected_alternative_cause_presence": "yes",
        },
    ]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_candidate_tsv(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=TSV_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate_to_tsv_row(candidate))


def write_prompts(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exported benchmark-v1 candidate prompts",
        "# Status: candidate_v1; not clean-source-gated yet",
        "# Format: <prompt> | <target> | <effect>",
        "",
        *[prompt_line(candidate) for candidate in candidates],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_control_prompts(controls: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exported benchmark-v1 control prompts",
        "# Format: <prompt> | <target> | <effect>",
        "",
        *[f"{row['prompt']} | {row['target_concept']} | {row['expected_effect']}" for row in controls],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def control_manifest_item(row: dict[str, str], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "pair_id": row["control_id"],
        "control_id": row["control_id"],
        "source_name": row["source_name"],
        "source_pair_id": row["source_pair_id"],
        "source_baseline": row["source_baseline"],
        "mechanism_type": row["mechanism_type"],
        "target_concept": row["target_concept"],
        "causal_footprint": row["expected_effect"],
        "expected_effect": row["expected_effect"],
        "control_type": row["control_type"],
        "source_prompt": row["prompt"],
        "prompt": row["prompt"],
        "purpose": row["purpose"],
        "expected_target_presence": row["expected_target_presence"],
        "expected_footprint_presence": row["expected_footprint_presence"],
        "expected_alternative_cause_presence": row["expected_alternative_cause_presence"],
    }


def write_manifests(
    candidates: list[Candidate],
    controls: list[dict[str, str]],
    *,
    output_dir: Path,
    candidate_prompts: Path,
    control_prompts: Path,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    candidate_manifest = {
        "created_at_utc": created_at,
        "slice_name": "causal_footprint_v1_candidates",
        "output_prompts": str(candidate_prompts),
        "count": len(candidates),
        "items": [candidate_to_json(candidate, index) for index, candidate in enumerate(candidates)],
    }
    (output_dir / "export_candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    controls_manifest = {
        "created_at_utc": created_at,
        "slice_name": "causal_footprint_v1_controls",
        "output_prompts": str(control_prompts),
        "count": len(controls),
        "items": [control_manifest_item(row, index) for index, row in enumerate(controls)],
    }
    (output_dir / "export_controls_manifest.json").write_text(
        json.dumps(controls_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_candidate(candidate: Candidate) -> None:
    missing = [
        name
        for name in [
            "pair_id",
            "target_concept",
            "causal_footprint",
            "mechanism_type",
            "source_prompt",
        ]
        if not getattr(candidate, name)
    ]
    if missing:
        raise ValueError(f"{candidate.pair_id or '<missing pair_id>'}: missing {', '.join(missing)}")
    if not re.fullmatch(r"[A-Za-z0-9_:-]+", candidate.pair_id):
        raise ValueError(f"{candidate.pair_id}: pair_id contains unsupported characters")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0-items", type=Path, default=DEFAULT_V0_ITEMS)
    parser.add_argument("--round6-candidates", type=Path, default=DEFAULT_ROUND6)
    parser.add_argument("--legacy-candidates", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        candidates = build_candidates(
            v0_items=args.v0_items,
            round6_candidates=args.round6_candidates,
            legacy_candidates=args.legacy_candidates,
            target_count=args.target_count,
        )
        for candidate in candidates:
            validate_candidate(candidate)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_prompts = args.prompt_dir / "causal_footprint_v1_candidates.txt"
    control_prompts = args.prompt_dir / "causal_footprint_v1_controls.txt"
    controls = [row for candidate in candidates for row in control_rows(candidate)]

    write_jsonl(
        [candidate_to_json(candidate, index) for index, candidate in enumerate(candidates)],
        args.output_dir / "candidate_items.jsonl",
    )
    write_candidate_tsv(candidates, args.output_dir / "candidate_pairs.tsv")
    write_jsonl(controls, args.output_dir / "controls_specs.jsonl")
    write_prompts(candidates, candidate_prompts)
    write_control_prompts(controls, control_prompts)
    write_manifests(
        candidates,
        controls,
        output_dir=args.output_dir,
        candidate_prompts=candidate_prompts,
        control_prompts=control_prompts,
    )

    print(
        f"Wrote {len(candidates)} candidate items and {len(controls)} controls "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
