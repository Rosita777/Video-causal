#!/usr/bin/env python3
"""Build the causal-footprint benchmark-v2 candidate pool and controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_V1_ITEMS = Path("benchmarks/causal_footprint_v1/candidate_items.jsonl")
DEFAULT_OUTPUT_DIR = Path("benchmarks/causal_footprint_v2")
DEFAULT_PROMPT_DIR = Path("prompts")
DEFAULT_TARGET_COUNT = 360

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

MECHANISM_TARGETS = {
    "fracture_damage": 80,
    "surface_trace": 80,
    "elastic_deformation": 80,
    "fluid_impact": 60,
    "field_mediated": 30,
    "particle_dispersion": 30,
}

MECHANISM_ORDER = [
    "fracture_damage",
    "surface_trace",
    "elastic_deformation",
    "fluid_impact",
    "particle_dispersion",
    "field_mediated",
]

ALTERNATIVE_CAUSES = {
    "fluid_impact": "a small visible bead",
    "fracture_damage": "a small metal hammer",
    "elastic_deformation": "a gloved hand",
    "surface_trace": "a smooth wooden block",
    "field_mediated": "a thin wooden stick",
    "particle_dispersion": "a small paper cup",
}

VARIANTS = [
    {
        "slug": "plain",
        "shot": "with a plain high-contrast background",
        "timing": "in the middle of the clip",
    },
    {
        "slug": "slow",
        "shot": "in slow motion with a locked camera",
        "timing": "after a short still beginning",
    },
    {
        "slug": "top",
        "shot": "from a top-down macro view",
        "timing": "after the surface is shown clean",
    },
    {
        "slug": "side",
        "shot": "from a side close-up view",
        "timing": "after the target is clearly visible",
    },
    {
        "slug": "bright",
        "shot": "under bright studio lighting",
        "timing": "only after the visible contact moment",
    },
]


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


@dataclass(frozen=True)
class Scenario:
    mechanism_type: str
    temporal_type: str
    target_concept: str
    scene_before: str
    action: str
    causal_footprint: str
    no_preexisting: str
    exclusivity_score: int = 5
    counterfactual_clarity: int = 5
    generatability_score: int = 4
    erasure_targetability: int = 5


def clean(value: Any) -> str:
    return str(value or "").strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
    return rows


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


def candidate_from_v1(row: dict[str, Any]) -> Candidate:
    target = clean(row.get("target_concept"))
    footprint = clean(row.get("expected_effect") or row.get("causal_footprint"))
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    return Candidate(
        pair_id=clean(row.get("pair_id")),
        target_concept=target,
        causal_footprint=footprint,
        mechanism_type=clean(row.get("mechanism_type")),
        temporal_type=clean(row.get("temporal_type")) or "delayed",
        exclusivity_score=int(scores.get("exclusivity_score", 5)),
        counterfactual_clarity=int(scores.get("counterfactual_clarity", 5)),
        generatability_score=int(scores.get("generatability_score", 4)),
        erasure_targetability=int(scores.get("erasure_targetability", 5)),
        source_collection="causal_footprint_v1",
        source_item_id=clean(row.get("item_id")),
        pair_source=clean(row.get("pair_source")) or "causal_footprint_v1",
        causal_chain=clean(row.get("causal_chain")) or generic_causal_chain(target, footprint),
        source_prompt=clean(row.get("source_prompt")),
        counterfactual_prompt=clean(row.get("counterfactual_prompt")) or no_cause_prompt(target, footprint),
        control_prompt=clean(row.get("control_prompt")) or effect_only_prompt(target, footprint),
        notes=f"Imported from benchmark-v1 item {clean(row.get('item_id'))}.",
    )


def expansion_scenarios() -> list[Scenario]:
    fracture = [
        ("red rubber ball", "an intact thin sugar-glass pane", "hits the center of the pane", "radial cracks spread outward across the glass", "crack"),
        ("white ceramic mug", "a clean gray tile floor", "drops onto the tile", "ceramic shards and thin cracks spread from the impact point", "crack or shard"),
        ("small metal nut", "an intact phone screen on a white table", "falls onto the screen", "a spiderweb crack appears on the phone screen", "screen crack"),
        ("yellow golf ball", "a clear acrylic sheet", "strikes the sheet", "a circular impact crack appears and spreads", "impact crack"),
        ("wooden mallet", "a smooth clay pot", "taps the side of the pot", "a small chip and crack line remain on the clay", "chip or crack"),
        ("ice cube", "a thin frozen puddle surface", "lands on the ice", "branching white cracks spread through the ice", "ice crack"),
        ("black hockey puck", "an intact mirror tile", "slides into the mirror tile", "a star-shaped crack spreads across the mirror", "mirror crack"),
        ("small hammer", "a plain white ceramic plate", "strikes the plate edge", "a visible chip and hairline crack appear", "chip or crack"),
        ("marble ball", "a brittle plaster slab", "drops onto the slab", "a round dent and branching cracks appear", "dent or crack"),
        ("stone cube", "a dry mud tile", "falls onto the tile", "dry mud cracks radiate from the impact point", "mud crack"),
        ("metal spoon", "a thin chocolate shell", "taps the shell", "jagged cracks spread across the chocolate", "chocolate crack"),
        ("small brick", "a clean windshield test panel", "hits the panel", "spiderweb cracks spread from the contact point", "windshield crack"),
        ("clear glass bead", "a thin ice sheet", "drops onto the ice sheet", "concentric cracks spread across the ice", "ice crack"),
        ("steel ball bearing", "a brittle black slate tile", "falls onto the slate", "pale branching cracks appear on the slate", "slate crack"),
    ]
    surface = [
        ("blue sneaker", "a flat patch of wet tan sand", "steps down and lifts away", "a sharp shoeprint remains in the wet sand", "print or mark"),
        ("bare hand", "a clean fogged glass panel", "presses flat against the glass", "a handprint remains on the fogged glass", "handprint"),
        ("bicycle tire", "smooth wet mud", "rolls straight through the mud", "a dark tire track remains in the mud", "track"),
        ("rubber stamp", "a blank white paper sheet", "presses onto the paper", "a square ink stamp mark remains on the paper", "ink mark"),
        ("wooden pencil", "a smooth clay tablet", "drags across the clay", "a thin groove remains in the clay", "groove"),
        ("toy car wheel", "a flat layer of white flour", "rolls across the flour", "two parallel wheel tracks remain in the flour", "track"),
        ("dog paw", "fresh white snow", "steps down and lifts away", "a paw print remains in the snow", "paw print"),
        ("paintbrush", "a blank white canvas", "swipes across the canvas", "a red paint stroke remains on the canvas", "paint stroke"),
        ("marker pen", "a clean whiteboard", "draws a line across the board", "a black line remains on the whiteboard", "marker line"),
        ("coin edge", "a smooth wax tablet", "presses into the wax", "a round coin impression remains in the wax", "impression"),
        ("garden rake", "a smooth soil bed", "pulls across the soil", "parallel grooves remain in the soil", "groove"),
        ("ice skate blade", "a smooth ice surface", "scrapes across the ice", "a bright scratch line remains in the ice", "scratch"),
        ("boot heel", "soft gray clay", "steps down and lifts away", "a deep heel print remains in the clay", "print"),
        ("chalk stick", "a clean blackboard", "moves across the board", "a white chalk line remains on the blackboard", "chalk line"),
    ]
    elastic = [
        ("orange basketball", "a black trampoline fabric surface", "drops onto the fabric", "the trampoline dips downward and rebounds", "dip or dent"),
        ("yellow tennis ball", "the center of a tennis racket", "hits the strings", "the racket strings bend inward and rebound", "string bend"),
        ("soccer ball", "a white goal net", "hits the center of the net", "the net stretches backward and ripples", "net stretch"),
        ("gloved fist", "a red punching bag", "punches the bag", "the bag dents inward and swings back", "dent"),
        ("metal weight", "a vertical coil spring", "drops onto the spring", "the spring coils compress downward", "compression"),
        ("rubber ball", "a soft blue gel block", "presses into the gel", "a round dent forms and slowly rebounds", "dent"),
        ("book", "a square foam block", "falls onto the block", "the foam block compresses flat then rises", "compression"),
        ("hand", "a white pillow", "presses into the pillow", "a deep pillow dent forms and remains briefly", "pillow dent"),
        ("ping pong ball", "a hanging thin curtain", "hits the curtain", "the curtain bows backward and ripples", "curtain bow"),
        ("foot", "a yellow sponge block", "steps onto the sponge", "the sponge compresses flat under the foot", "compression"),
        ("wooden block", "a rubber membrane", "falls onto the membrane", "the membrane stretches downward into a bowl shape", "membrane stretch"),
    ]
    fluid = [
        ("blue ink droplet", "a clear glass of still water", "falls into the water", "a blue plume blooms downward through the water", "plume or ripple"),
        ("red dye droplet", "a clear bowl of still water", "enters the water surface", "a red cloud blooms and spreads through the water", "dye cloud"),
        ("black pebble", "a shallow tray of still blue water", "drops into the water", "circular ripples spread outward from the impact point", "ripple"),
        ("silver coin", "a shallow fountain basin", "falls into the water", "a small splash crown and ripple rings appear", "splash or ripple"),
        ("milk droplet", "a cup of black coffee", "falls into the coffee", "a white swirl blooms through the coffee", "white swirl"),
        ("green grape", "a clear bowl of water", "drops into the water", "a splash crown rises and ripples spread", "splash or ripple"),
        ("ice cube", "a glass of tea", "falls into the tea", "brown ripples and bubbles spread from the impact point", "ripple or bubble"),
    ]
    particle = [
        ("chalk eraser", "a clean blackboard ledge with loose chalk dust", "taps the ledge", "a white chalk dust cloud rises", "dust cloud"),
        ("flour scoop", "a dark tabletop with a small flour pile", "drops flour onto the table", "white flour dust spreads outward", "flour dust"),
        ("makeup brush", "a compact of pink powder", "taps the powder", "a pink powder cloud blooms upward", "powder cloud"),
        ("salt shaker", "a black table surface", "shakes once above the table", "white salt grains scatter across the black table", "salt grains"),
        ("sandbag", "a clean blue floor", "tips over", "tan sand spills into a spreading pile", "sand pile"),
        ("cracker", "a dark plate", "breaks against the plate", "small crumbs scatter across the plate", "crumbs"),
    ]
    field = [
        ("red balloon", "loose hair strands on a white table", "moves close to the hair", "hair strands lift upward and cling toward the balloon", "lifted hair"),
        ("black bar magnet", "iron filings scattered on white paper", "slides near the filings", "iron filings gather into curved field lines", "field lines"),
        ("yellow plastic comb", "tiny paper scraps on a blue table", "moves near the scraps", "paper scraps lift and cling toward the comb", "lifted paper"),
        ("desk fan", "hanging paper streamers", "turns on facing the streamers", "paper streamers bend backward and flutter", "bent streamers"),
        ("hair dryer", "a pile of white powder on a black table", "blows toward the powder", "white powder spreads into a sideways cloud", "powder cloud"),
        ("charged plastic ruler", "small foam beads on a dark tray", "moves near the beads", "foam beads roll together toward the ruler", "moving beads"),
    ]

    rows: list[Scenario] = []
    for mechanism, temporal, raw_rows in [
        ("fracture_damage", "persistent", fracture),
        ("surface_trace", "persistent", surface),
        ("elastic_deformation", "synchronous", elastic),
        ("fluid_impact", "delayed", fluid),
        ("particle_dispersion", "delayed", particle),
        ("field_mediated", "delayed", field),
    ]:
        for target, scene, action, footprint, no_preexisting in raw_rows:
            rows.append(
                Scenario(
                    mechanism_type=mechanism,
                    temporal_type=temporal,
                    target_concept=target,
                    scene_before=scene,
                    action=action,
                    causal_footprint=footprint,
                    no_preexisting=no_preexisting,
                    generatability_score=5 if mechanism in {"particle_dispersion", "field_mediated"} else 4,
                )
            )
    return rows


def source_prompt_for_scenario(scenario: Scenario, variant: dict[str, str]) -> str:
    return (
        f"A realistic fixed-camera close-up video {variant['shot']}. "
        f"The scene starts with {scenario.scene_before}, with no pre-existing {scenario.no_preexisting}. "
        f"A clearly visible {scenario.target_concept} enters the frame and remains visible before contact. "
        f"The {scenario.target_concept} {scenario.action} {variant['timing']}, causing {scenario.causal_footprint}; "
        "the effect begins only after contact and remains visible for several frames."
    )


def expansion_candidate(scenario: Scenario, variant: dict[str, str], serial: int) -> Candidate:
    pair_id = (
        f"v2_{scenario.mechanism_type}_{slugify(scenario.target_concept)}_"
        f"{slugify(scenario.causal_footprint)[:36]}_{variant['slug']}_{serial:03d}"
    )
    source_prompt = source_prompt_for_scenario(scenario, variant)
    return Candidate(
        pair_id=pair_id,
        target_concept=scenario.target_concept,
        causal_footprint=scenario.causal_footprint,
        mechanism_type=scenario.mechanism_type,
        temporal_type=scenario.temporal_type,
        exclusivity_score=scenario.exclusivity_score,
        counterfactual_clarity=scenario.counterfactual_clarity,
        generatability_score=scenario.generatability_score,
        erasure_targetability=scenario.erasure_targetability,
        source_collection="causal_footprint_v2_targeted_expansion",
        source_item_id=pair_id,
        pair_source="v2_targeted_expansion",
        causal_chain=generic_causal_chain(scenario.target_concept, scenario.causal_footprint),
        source_prompt=source_prompt,
        counterfactual_prompt=no_cause_prompt(scenario.target_concept, scenario.causal_footprint),
        control_prompt=effect_only_prompt(scenario.target_concept, scenario.causal_footprint),
        notes=(
            "Targeted v2 expansion generated after CogVideoX-2B v1 clean-source failure audit; "
            "prompt enforces target visibility, no pre-existing footprint, and post-contact effect onset."
        ),
    )


def all_expansion_candidates() -> list[Candidate]:
    candidates = []
    serial = 0
    for mechanism in MECHANISM_ORDER:
        scenarios = [row for row in expansion_scenarios() if row.mechanism_type == mechanism]
        for scenario in scenarios:
            for variant in VARIANTS:
                candidates.append(expansion_candidate(scenario, variant, serial))
                serial += 1
    return candidates


def add_unique(target: list[Candidate], seen: set[str], candidates: list[Candidate], limit: int | None = None) -> None:
    for candidate in candidates:
        if limit is not None and len(target) >= limit:
            return
        if not candidate.pair_id or candidate.pair_id in seen:
            continue
        target.append(candidate)
        seen.add(candidate.pair_id)


def build_candidates(v1_items: Path, target_count: int) -> list[Candidate]:
    selected: list[Candidate] = []
    seen: set[str] = set()
    add_unique(selected, seen, [candidate_from_v1(row) for row in read_jsonl(v1_items)])

    expansion = all_expansion_candidates()
    by_mechanism: dict[str, list[Candidate]] = {mechanism: [] for mechanism in MECHANISM_ORDER}
    for candidate in expansion:
        by_mechanism.setdefault(candidate.mechanism_type, []).append(candidate)

    current_counts = Counter(candidate.mechanism_type for candidate in selected)
    for mechanism in MECHANISM_ORDER:
        desired = MECHANISM_TARGETS.get(mechanism, current_counts[mechanism])
        deficit = max(0, desired - current_counts[mechanism])
        add_unique(selected, seen, by_mechanism.get(mechanism, [])[:deficit], target_count)
        if len(selected) >= target_count:
            return selected[:target_count]

    add_unique(selected, seen, expansion, target_count)
    if len(selected) < target_count:
        raise ValueError(f"only {len(selected)} unique candidate rows available; requested {target_count}")
    return selected[:target_count]


def candidate_to_json(candidate: Candidate, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "item_id": f"v2:{candidate.pair_id}",
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
        "candidate_status": "candidate_v2",
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
        "status": "candidate_v2",
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
        "source_name": "causal_footprint_v2",
        "source_pair_id": candidate.pair_id,
        "source_baseline": "candidate_v2_clean_screening",
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
        "# Exported benchmark-v2 candidate prompts",
        "# Status: candidate_v2; not clean-source-gated yet",
        "# Format: <prompt> | <target> | <effect>",
        "",
        *[prompt_line(candidate) for candidate in candidates],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_control_prompts(controls: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exported benchmark-v2 control prompts",
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
        "slice_name": "causal_footprint_v2_candidates",
        "output_prompts": str(candidate_prompts),
        "target_mechanism_counts": MECHANISM_TARGETS,
        "actual_mechanism_counts": dict(Counter(candidate.mechanism_type for candidate in candidates)),
        "count": len(candidates),
        "items": [candidate_to_json(candidate, index) for index, candidate in enumerate(candidates)],
    }
    (output_dir / "export_candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    controls_manifest = {
        "created_at_utc": created_at,
        "slice_name": "causal_footprint_v2_controls",
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


def write_readme(path: Path, candidates: list[Candidate]) -> None:
    counts = Counter(candidate.mechanism_type for candidate in candidates)
    lines = [
        "# Causal Footprint Benchmark V2",
        "",
        "V2 is a larger clean-source candidate pool for CogVideoX-2B screening.",
        "It keeps all v1 candidate rows and appends targeted expansions after the v1",
        "chunked GPT-5.4 audit showed that 150 rows yielded only 72 clean-source candidates.",
        "",
        "The expansion is intentionally biased toward weak v1 buckets:",
        "`fracture_damage`, `surface_trace`, and `elastic_deformation`.",
        "",
        "Files:",
        "",
        "- `candidate_items.jsonl`: machine-readable candidate rows.",
        "- `candidate_pairs.tsv`: tabular candidate metadata.",
        "- `controls_specs.jsonl`: three controls per candidate.",
        "- `export_candidate_manifest.json`: prompt export manifest.",
        "- `export_controls_manifest.json`: control export manifest.",
        "",
        "Mechanism counts:",
        "",
        *[f"- `{mechanism}`: {counts[mechanism]}" for mechanism in sorted(counts)],
        "",
        "V2 rows are not final benchmark rows. They must pass clean-source chunked VLM",
        "triage and human adjudication before controls and erasure baselines are run.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-items", type=Path, default=DEFAULT_V1_ITEMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.target_count <= 0:
        parser.error("--target-count must be positive")
    try:
        candidates = build_candidates(args.v1_items, args.target_count)
        for candidate in candidates:
            validate_candidate(candidate)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_prompts = args.prompt_dir / "causal_footprint_v2_candidates.txt"
    control_prompts = args.prompt_dir / "causal_footprint_v2_controls.txt"
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
    write_readme(args.output_dir / "README.md", candidates)

    counts = Counter(candidate.mechanism_type for candidate in candidates)
    print(f"Wrote {len(candidates)} candidate items to {args.output_dir / 'candidate_items.jsonl'}")
    print(f"Wrote {len(controls)} control specs to {args.output_dir / 'controls_specs.jsonl'}")
    print("Mechanism counts: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
