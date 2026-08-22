#!/usr/bin/env python3
"""Build the Original-only eight-mechanism capability batch.

This is a pre-method capability screen.  It does not define training data, select
evaluation cases, or authorize treatment generation.  The first four mechanism
ontologies are read from the frozen Protocol v1 registry.  The additional four
ontologies are intentionally labelled capability-only drafts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_protocol_v1_manifests import (  # noqa: E402
    causal_action as protocol_v1_causal_action,
    clean_prefix as protocol_v1_clean_prefix,
    fixed_context as protocol_v1_fixed_context,
)


PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v1"
ARTIFACT_STEM = PROTOCOL_VERSION
EXPECTED_PROTOCOL_V1_REGISTRY_SHA256 = (
    "d0e24777e73da74d38875ddb997c6ffc098fdea86f9807d799ed787303f4280a"
)

MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "field_mediated_response",
    "material_release",
    "surface_trace",
)
PROTOCOL_V1_MECHANISMS = MECHANISM_ORDER[:4]
DRAFT_MECHANISMS = MECHANISM_ORDER[4:]

ONTOLOGY_STATUS_PROTOCOL_V1 = "protocol_v1_reused_for_capability_screen"
ONTOLOGY_STATUS_DRAFT = "draft_capability_only_not_training_ready"
INTENDED_USE = "original_capability_screening_only"
METHOD_ARM = "original"
TREATMENT_STATUS = "pre_method_original_only"

BASE_SEED = 840000
SEED_FORMULA = (
    "840000 + 1000*mechanism_index + 10*combination_index + repetition_index"
)
COMBINATIONS_PER_MECHANISM = 8
REPETITIONS_PER_COMBINATION = 3
ROWS_PER_MECHANISM = COMBINATIONS_PER_MECHANISM * REPETITIONS_PER_COMBINATION
VIDEO_FRAMES = 49
FPS = 8
CLEAN_PREFIX_FRAMES = 16
REFERENCE_START_INCLUSIVE = 0
REFERENCE_END_EXCLUSIVE = CLEAN_PREFIX_FRAMES

PROMPT_STYLES = ("direct", "natural")

MANIFEST_FIELDS = (
    "protocol_version",
    "generation_id",
    "case_id",
    "mechanism_index",
    "combination_index",
    "repetition_index",
    "mechanism",
    "mechanism_name",
    "ontology_status",
    "ontology_provenance",
    "intended_use",
    "method_arm",
    "treatment_status",
    "prompt_style",
    "source_id",
    "source_object",
    "source_family",
    "source_motion",
    "receiver_id",
    "receiver",
    "receiver_family",
    "receiver_clean_state",
    "compatibility_rule",
    "prompt",
    "target_concept",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "seed_formula",
    "num_frames",
    "fps",
    "reference_start_inclusive",
    "reference_end_exclusive",
)


# These four definitions are capability-probe candidates only.  Passing this
# batch is necessary, but not sufficient, for constructing a training ontology.
DRAFT_MECHANISM_SPECS: dict[str, dict[str, Any]] = {
    "elastic_deformation": {
        "name": "Elastic deformation",
        "footprint": "a clearly visible temporary indentation or stretching of the receiver followed by elastic rebound",
        "counterfactual_state": "the source object is absent and the elastic receiver keeps its original undeformed shape",
        "fixed_context": "The camera, supports, lighting, and background remain fixed throughout.",
        "sources": [
            {
                "id": "orange_basketball",
                "name": "one small orange basketball",
                "motion": "falls vertically from above",
                "family": "vertical_deflection",
            },
            {
                "id": "red_rubber_ball",
                "name": "one small red rubber ball",
                "motion": "falls vertically from above",
                "family": "vertical_deflection",
            },
            {
                "id": "green_tennis_ball",
                "name": "one green tennis ball",
                "motion": "flies horizontally from left to right",
                "family": "horizontal_stretch",
            },
            {
                "id": "soccer_ball",
                "name": "one small black-and-white soccer ball",
                "motion": "flies horizontally from left to right",
                "family": "horizontal_stretch",
            },
        ],
        "receivers": [
            {
                "id": "black_trampoline",
                "name": "a small black trampoline stretched horizontally",
                "clean_state": "its elastic surface is flat, taut, and motionless",
                "family": "vertical_deflection",
            },
            {
                "id": "blue_foam_cushion",
                "name": "a thick blue foam cushion resting on a table",
                "clean_state": "its top surface is level, undeformed, and motionless",
                "family": "vertical_deflection",
            },
            {
                "id": "tennis_practice_net",
                "name": "a taut square tennis practice net held vertically",
                "clean_state": "the net is straight, evenly tensioned, and motionless",
                "family": "horizontal_stretch",
            },
            {
                "id": "soccer_goal_net",
                "name": "a taut white soccer goal net held vertically",
                "clean_state": "the net hangs in its original shape and is completely still",
                "family": "horizontal_stretch",
            },
        ],
        "pair_indices": ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (2, 3), (3, 2)),
    },
    "field_mediated_response": {
        "name": "Field-mediated response",
        "footprint": "clear receiver motion toward or away from the source while a visible gap remains between them",
        "counterfactual_state": "the source is absent and the lightweight receiver remains in its original position without induced motion",
        "fixed_context": "The camera, support, lighting, and background remain fixed, and no direct contact occurs.",
        "sources": [
            {
                "id": "charged_blue_balloon",
                "name": "one charged blue balloon",
                "motion": "moves slowly into view from the left and stops nearby without touching",
                "family": "electrostatic",
            },
            {
                "id": "charged_black_comb",
                "name": "one charged black plastic comb",
                "motion": "moves slowly into view from the left and stops nearby without touching",
                "family": "electrostatic",
            },
            {
                "id": "charged_clear_acrylic_rod",
                "name": "one charged clear acrylic rod",
                "motion": "moves slowly into view from the left and stops nearby without touching",
                "family": "electrostatic",
            },
            {
                "id": "charged_red_plastic_ruler",
                "name": "one charged red plastic ruler",
                "motion": "moves slowly into view from the left and stops nearby without touching",
                "family": "electrostatic",
            },
        ],
        "receivers": [
            {
                "id": "paper_confetti",
                "name": "several tiny colored paper pieces resting on a dark tabletop",
                "clean_state": "every paper piece is flat and completely motionless",
                "family": "electrostatic",
            },
            {
                "id": "foam_beads",
                "name": "several tiny white foam beads resting on a black tray",
                "clean_state": "every bead is separated and completely motionless",
                "family": "electrostatic",
            },
            {
                "id": "tissue_squares",
                "name": "several tiny white tissue-paper squares resting on a black tray",
                "clean_state": "every tissue square lies flat and is completely motionless",
                "family": "electrostatic",
            },
            {
                "id": "foil_strips",
                "name": "several narrow lightweight aluminum-foil strips resting on a dark tray",
                "clean_state": "every foil strip lies flat and is completely motionless",
                "family": "electrostatic",
            },
        ],
        "pair_indices": ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (2, 3), (3, 2)),
    },
    "material_release": {
        "name": "Material release and particle dispersion",
        "footprint": "opening of the receiver followed by a visible outward pour or scatter of its contained particles",
        "counterfactual_state": "the source is absent, the receiver remains closed and intact, and no contained material is released",
        "fixed_context": "The camera, receiver support, lighting, and background remain fixed throughout.",
        "sources": [
            {
                "id": "pointed_wood_dowel",
                "name": "one pointed wooden dowel",
                "motion": "descends vertically from above",
                "family": "puncture_release",
            },
            {
                "id": "pointed_metal_rod",
                "name": "one pointed silver metal rod",
                "motion": "descends vertically from above",
                "family": "puncture_release",
            },
            {
                "id": "small_wood_mallet",
                "name": "one small wooden mallet",
                "motion": "moves vertically into frame from above",
                "family": "impact_release",
            },
            {
                "id": "polished_steel_ball",
                "name": "one polished steel ball",
                "motion": "falls vertically from above",
                "family": "impact_release",
            },
        ],
        "receivers": [
            {
                "id": "lentil_paper_pouch",
                "name": "a thin suspended paper pouch filled with yellow lentils",
                "clean_state": "the pouch is sealed, intact, and motionless with no material outside",
                "family": "puncture_release",
            },
            {
                "id": "bead_paper_sachet",
                "name": "a thin suspended paper sachet filled with blue beads",
                "clean_state": "the sachet is sealed, intact, and motionless with no material outside",
                "family": "puncture_release",
            },
            {
                "id": "rice_clay_capsule",
                "name": "a brittle hollow clay capsule filled with white rice grains",
                "clean_state": "the capsule is closed, intact, and motionless with no grains outside",
                "family": "impact_release",
            },
            {
                "id": "sand_chalk_shell",
                "name": "a brittle hollow chalk shell filled with red sand",
                "clean_state": "the shell is closed, intact, and motionless with no sand outside",
                "family": "impact_release",
            },
        ],
        "pair_indices": ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (2, 3), (3, 2)),
    },
    "surface_trace": {
        "name": "Surface trace",
        "footprint": "a new localized imprint or groove confined to the source contact path",
        "counterfactual_state": "the source is absent and the receiver surface remains smooth, clean, and completely unmarked",
        "fixed_context": "The camera, surface support, lighting, and background remain fixed throughout.",
        "sources": [
            {
                "id": "boot_shaped_stamp",
                "name": "one small boot-shaped rubber stamp",
                "motion": "descends vertically from above",
                "family": "press_trace",
            },
            {
                "id": "round_rubber_stamp",
                "name": "one round rubber stamp",
                "motion": "descends vertically from above",
                "family": "press_trace",
            },
            {
                "id": "rounded_wood_stylus",
                "name": "one rounded wooden stylus",
                "motion": "moves into view from the left",
                "family": "drag_trace",
            },
            {
                "id": "blunt_metal_stylus",
                "name": "one blunt silver metal stylus",
                "motion": "moves into view from the left",
                "family": "drag_trace",
            },
        ],
        "receivers": [
            {
                "id": "damp_sand_bed",
                "name": "a flat bed of fine damp sand",
                "clean_state": "its surface is smooth, level, motionless, and completely unmarked",
                "family": "press_trace",
            },
            {
                "id": "soft_clay_pad",
                "name": "a flat pad of soft gray clay",
                "clean_state": "its surface is smooth, level, motionless, and completely unmarked",
                "family": "press_trace",
            },
            {
                "id": "modeling_wax_slab",
                "name": "a flat slab of soft pale modeling wax",
                "clean_state": "its surface is smooth, level, motionless, and completely unmarked",
                "family": "drag_trace",
            },
            {
                "id": "red_clay_strip",
                "name": "a long flat strip of soft red clay",
                "clean_state": "its surface is smooth, level, motionless, and completely unmarked",
                "family": "drag_trace",
            },
        ],
        "pair_indices": ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (2, 3), (3, 2)),
    },
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol_v1_registry(path: Path, *, enforce_frozen_hash: bool = True) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Protocol v1 registry is not a regular file: {path}")
    if enforce_frozen_hash:
        require(
            sha256_file(path) == EXPECTED_PROTOCOL_V1_REGISTRY_SHA256,
            "Protocol v1 registry hash mismatch; refusing to construct a changed capability batch",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse Protocol v1 registry: {path}") from exc
    require(isinstance(payload, dict), "Protocol v1 registry must be an object")
    require(payload.get("protocol_version") == "v1", "Protocol v1 version mismatch")
    require(payload.get("video_frames") == VIDEO_FRAMES, "Protocol v1 frame count mismatch")
    require(payload.get("fps") == FPS, "Protocol v1 fps mismatch")
    require(payload.get("clean_prefix_frames") == CLEAN_PREFIX_FRAMES, "Protocol v1 clean-prefix mismatch")
    mechanisms = payload.get("mechanisms")
    require(isinstance(mechanisms, dict), "Protocol v1 mechanisms must be an object")
    require(
        all(mechanism in mechanisms for mechanism in PROTOCOL_V1_MECHANISMS),
        "Protocol v1 is missing a required mechanism",
    )
    return payload


def _validate_item(item: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    require(set(item) >= set(fields), f"{label} is missing required fields")
    for field in fields:
        require(isinstance(item[field], str) and item[field].strip() == item[field] and item[field], f"{label}.{field} is invalid")
        require("|" not in item[field] and "\n" not in item[field] and "\r" not in item[field], f"{label}.{field} contains a forbidden delimiter")


def _protocol_v1_spec(registry: Mapping[str, Any], mechanism: str) -> dict[str, Any]:
    raw = registry["mechanisms"][mechanism]
    require(isinstance(raw, dict), f"{mechanism}: registry mechanism must be an object")
    sources = list(raw.get("train_sources", [])) + list(raw.get("test_sources", []))
    receivers = list(raw.get("train_receivers", [])) + list(raw.get("test_receivers", []))
    require(len(sources) >= 4 and len(receivers) >= 4, f"{mechanism}: insufficient Protocol v1 ontology")
    sources = [dict(source, family=f"{mechanism}_compatible") for source in sources[:4]]
    receivers = [dict(receiver, family=f"{mechanism}_compatible") for receiver in receivers[:4]]
    return {
        "name": raw["name"],
        "footprint": raw["footprint"],
        "counterfactual_state": raw["counterfactual_state"],
        "sources": sources,
        "receivers": receivers,
        "pair_indices": ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 2), (2, 3), (3, 0)),
        "ontology_status": ONTOLOGY_STATUS_PROTOCOL_V1,
        "ontology_provenance": "data/protocol_v1/registry.json",
    }


def mechanism_specs(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        mechanism: _protocol_v1_spec(registry, mechanism)
        for mechanism in PROTOCOL_V1_MECHANISMS
    }
    for mechanism in DRAFT_MECHANISMS:
        spec = dict(DRAFT_MECHANISM_SPECS[mechanism])
        spec["sources"] = [dict(item) for item in spec["sources"]]
        spec["receivers"] = [dict(item) for item in spec["receivers"]]
        spec["ontology_status"] = ONTOLOGY_STATUS_DRAFT
        spec["ontology_provenance"] = f"{PROTOCOL_VERSION}:embedded_capability_draft"
        specs[mechanism] = spec
    require(tuple(specs) == MECHANISM_ORDER, "mechanism order is not canonical")
    validate_mechanism_specs(specs)
    return specs


def validate_mechanism_specs(specs: Mapping[str, Mapping[str, Any]]) -> None:
    require(tuple(specs) == MECHANISM_ORDER, "expected exactly the eight canonical mechanisms")
    for mechanism, spec in specs.items():
        for field in ("name", "footprint", "counterfactual_state", "ontology_status", "ontology_provenance"):
            require(isinstance(spec.get(field), str) and str(spec[field]).strip(), f"{mechanism}.{field} is invalid")
            require("|" not in str(spec[field]) and "\n" not in str(spec[field]) and "\r" not in str(spec[field]), f"{mechanism}.{field} contains a forbidden delimiter")
        sources = spec.get("sources")
        receivers = spec.get("receivers")
        pairs = spec.get("pair_indices")
        require(isinstance(sources, list) and len(sources) == 4, f"{mechanism}: expected four capability sources")
        require(isinstance(receivers, list) and len(receivers) == 4, f"{mechanism}: expected four capability receivers")
        require(isinstance(pairs, (tuple, list)) and len(pairs) == COMBINATIONS_PER_MECHANISM, f"{mechanism}: expected eight capability pairs")
        for index, source in enumerate(sources):
            require(isinstance(source, dict), f"{mechanism}: source {index} must be an object")
            _validate_item(source, ("id", "name", "motion", "family"), f"{mechanism}.source[{index}]")
        for index, receiver in enumerate(receivers):
            require(isinstance(receiver, dict), f"{mechanism}: receiver {index} must be an object")
            _validate_item(receiver, ("id", "name", "clean_state", "family"), f"{mechanism}.receiver[{index}]")
        require(len({source["id"] for source in sources}) == 4, f"{mechanism}: source IDs repeat")
        require(len({receiver["id"] for receiver in receivers}) == 4, f"{mechanism}: receiver IDs repeat")
        if mechanism == "field_mediated_response":
            require(
                {source["family"] for source in sources}
                == {receiver["family"] for receiver in receivers}
                == {"electrostatic"},
                "field-mediated capability ontology must remain purely electrostatic",
            )
        require(len(set(tuple(pair) for pair in pairs)) == 8, f"{mechanism}: capability pairs repeat")
        for pair in pairs:
            require(isinstance(pair, (tuple, list)) and len(pair) == 2, f"{mechanism}: malformed capability pair")
            source_index, receiver_index = pair
            require(type(source_index) is int and 0 <= source_index < 4, f"{mechanism}: invalid source index")
            require(type(receiver_index) is int and 0 <= receiver_index < 4, f"{mechanism}: invalid receiver index")
            if mechanism in DRAFT_MECHANISMS:
                require(
                    sources[source_index]["family"] == receivers[receiver_index]["family"],
                    f"{mechanism}: incompatible draft source/receiver pair",
                )
        require(Counter(pair[0] for pair in pairs) == Counter({index: 2 for index in range(4)}), f"{mechanism}: source coverage is not balanced")
        require(Counter(pair[1] for pair in pairs) == Counter({index: 2 for index in range(4)}), f"{mechanism}: receiver coverage is not balanced")
        if mechanism in DRAFT_MECHANISMS:
            require(spec["ontology_status"] == ONTOLOGY_STATUS_DRAFT, f"{mechanism}: draft status is not fail-closed")
            require(isinstance(spec.get("fixed_context"), str) and str(spec["fixed_context"]).strip(), f"{mechanism}: fixed context is invalid")
            require("|" not in str(spec["fixed_context"]) and "\n" not in str(spec["fixed_context"]) and "\r" not in str(spec["fixed_context"]), f"{mechanism}: fixed context contains a forbidden delimiter")


def _draft_action(mechanism: str, source: Mapping[str, str], receiver: Mapping[str, str]) -> str:
    if mechanism == "elastic_deformation":
        return f"{source['name']} {source['motion']} and contacts {receiver['name']} once"
    if mechanism == "field_mediated_response":
        return f"{source['name']} {source['motion']}; while a clear gap remains, it causes {receiver['name']} to move"
    if mechanism == "material_release":
        if source["family"] == "puncture_release":
            return f"{source['name']} {source['motion']} and punctures {receiver['name']} once"
        return f"{source['name']} {source['motion']} and strikes {receiver['name']} once"
    if mechanism == "surface_trace":
        if source["family"] == "press_trace":
            return f"{source['name']} {source['motion']} and presses once into {receiver['name']}"
        return f"{source['name']} {source['motion']} and drags once across {receiver['name']}"
    raise KeyError(mechanism)


def capability_prompt(
    mechanism: str,
    spec: Mapping[str, Any],
    source: Mapping[str, str],
    receiver: Mapping[str, str],
    prompt_style: str,
) -> str:
    require(prompt_style in PROMPT_STYLES, f"unsupported prompt style: {prompt_style}")
    prefix = protocol_v1_clean_prefix(mechanism, dict(source), dict(receiver))
    if mechanism in PROTOCOL_V1_MECHANISMS:
        action = protocol_v1_causal_action(mechanism, dict(source), dict(receiver))
        fixed = protocol_v1_fixed_context(mechanism)
    else:
        action = _draft_action(mechanism, source, receiver)
        fixed = str(spec["fixed_context"])
    footprint = str(spec["footprint"])
    if prompt_style == "direct":
        event = f"Then {action}. Only after this causal event, the scene shows {footprint}."
    else:
        event = f"After the quiet opening, {action}. The event naturally produces {footprint}."
    return (
        "A realistic locked-camera close-up video in one continuous shot. "
        f"{prefix} {event} {fixed} "
        "No cuts, camera motion, people, hands, text, or additional moving objects appear."
    )


def seed_for(mechanism_index: int, combination_index: int, repetition_index: int) -> int:
    require(0 <= mechanism_index < len(MECHANISM_ORDER), "mechanism index outside seed domain")
    require(0 <= combination_index < COMBINATIONS_PER_MECHANISM, "combination index outside seed domain")
    require(0 <= repetition_index < REPETITIONS_PER_COMBINATION, "repetition index outside seed domain")
    return BASE_SEED + 1000 * mechanism_index + 10 * combination_index + repetition_index


def _compatibility_rule(mechanism: str, source: Mapping[str, str], receiver: Mapping[str, str]) -> str:
    if mechanism in PROTOCOL_V1_MECHANISMS:
        return "protocol_v1_physical_pair_selected_from_frozen_ontology"
    return f"draft_family_match:{source['family']}"


def build_rows(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    specs = mechanism_specs(registry)
    rows: list[dict[str, str]] = []
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        spec = specs[mechanism]
        for combination_index, (source_index, receiver_index) in enumerate(spec["pair_indices"]):
            source = spec["sources"][source_index]
            receiver = spec["receivers"][receiver_index]
            prompt_style = "direct" if combination_index < 4 else "natural"
            case_id = f"cap8m{mechanism_index:02d}c{combination_index:02d}"
            prompt = capability_prompt(mechanism, spec, source, receiver, prompt_style)
            for repetition_index in range(REPETITIONS_PER_COMBINATION):
                generation_id = f"{case_id}r{repetition_index:02d}"
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "generation_id": generation_id,
                    "case_id": case_id,
                    "mechanism_index": str(mechanism_index),
                    "combination_index": str(combination_index),
                    "repetition_index": str(repetition_index),
                    "mechanism": mechanism,
                    "mechanism_name": str(spec["name"]),
                    "ontology_status": str(spec["ontology_status"]),
                    "ontology_provenance": str(spec["ontology_provenance"]),
                    "intended_use": INTENDED_USE,
                    "method_arm": METHOD_ARM,
                    "treatment_status": TREATMENT_STATUS,
                    "prompt_style": prompt_style,
                    "source_id": source["id"],
                    "source_object": source["name"],
                    "source_family": source["family"],
                    "source_motion": source["motion"],
                    "receiver_id": receiver["id"],
                    "receiver": receiver["name"],
                    "receiver_family": receiver["family"],
                    "receiver_clean_state": receiver["clean_state"],
                    "compatibility_rule": _compatibility_rule(mechanism, source, receiver),
                    "prompt": prompt,
                    "target_concept": source["name"],
                    "expected_footprint": str(spec["footprint"]),
                    "expected_counterfactual_state": str(spec["counterfactual_state"]),
                    "seed": str(seed_for(mechanism_index, combination_index, repetition_index)),
                    "seed_formula": SEED_FORMULA,
                    "num_frames": str(VIDEO_FRAMES),
                    "fps": str(FPS),
                    "reference_start_inclusive": str(REFERENCE_START_INCLUSIVE),
                    "reference_end_exclusive": str(REFERENCE_END_EXCLUSIVE),
                }
                require(tuple(row) == MANIFEST_FIELDS, "internal manifest field order changed")
                rows.append(row)
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    expected_total = len(MECHANISM_ORDER) * ROWS_PER_MECHANISM
    require(len(rows) == expected_total, f"expected {expected_total} rows, got {len(rows)}")
    require(all(tuple(row) == MANIFEST_FIELDS for row in rows), "manifest field schema mismatch")
    require(len({row["generation_id"] for row in rows}) == expected_total, "generation IDs are not unique")
    require(len({int(row["seed"]) for row in rows}) == expected_total, "generation seeds are not unique")
    require(len({row["case_id"] for row in rows}) == len(MECHANISM_ORDER) * COMBINATIONS_PER_MECHANISM, "case ID count mismatch")
    require(len({row["prompt"] for row in rows}) == len(MECHANISM_ORDER) * COMBINATIONS_PER_MECHANISM, "semantic-case prompts are not unique")

    by_mechanism: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_mechanism[row["mechanism"]].append(row)
        by_case[row["case_id"]].append(row)
        require(row["protocol_version"] == PROTOCOL_VERSION, "row protocol version mismatch")
        require(row["intended_use"] == INTENDED_USE, "row use expanded beyond capability screening")
        require(row["method_arm"] == METHOD_ARM and row["treatment_status"] == TREATMENT_STATUS, "non-Original row found in capability batch")
        require(row["num_frames"] == str(VIDEO_FRAMES), "row frame count mismatch")
        require(row["fps"] == str(FPS), "row fps mismatch")
        require(row["reference_start_inclusive"] == "0", "row clean-prefix start mismatch")
        require(row["reference_end_exclusive"] == str(CLEAN_PREFIX_FRAMES), "row clean-prefix end mismatch")
        require(row["seed_formula"] == SEED_FORMULA, "row seed formula mismatch")
        require("During the first two seconds" in row["prompt"], "prompt does not state the clean opening")
        require("The source object is not visible" in row["prompt"], "prompt does not exclude the source from the clean opening")
        require(row["source_object"] in row["prompt"], "prompt omits its source")
        require(row["receiver"] in row["prompt"], "prompt omits its receiver")
        require(row["expected_footprint"] in row["prompt"], "prompt omits its footprint")
        require(not re.search(r"\b(?:todo|tbd|placeholder)\b", row["prompt"], flags=re.IGNORECASE), "prompt contains placeholder language")
        require("|" not in row["prompt"] and "\n" not in row["prompt"] and "\r" not in row["prompt"], "prompt contains a forbidden delimiter")

    require(tuple(by_mechanism) == MECHANISM_ORDER, "manifest mechanism order mismatch")
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        mechanism_rows = by_mechanism[mechanism]
        require(len(mechanism_rows) == ROWS_PER_MECHANISM, f"{mechanism}: row count mismatch")
        styles = Counter(row["prompt_style"] for row in mechanism_rows)
        require(styles == Counter({"direct": 12, "natural": 12}), f"{mechanism}: prompt-style quota mismatch")
        for prompt_style in PROMPT_STYLES:
            style_rows = [row for row in mechanism_rows if row["prompt_style"] == prompt_style]
            require(Counter(row["source_id"] for row in style_rows) == Counter({row["source_id"]: 3 for row in style_rows}), f"{mechanism}: source/style coverage is not balanced")
            require(Counter(row["receiver_id"] for row in style_rows) == Counter({row["receiver_id"]: 3 for row in style_rows}), f"{mechanism}: receiver/style coverage is not balanced")
        pairs = {(row["source_id"], row["receiver_id"]) for row in mechanism_rows}
        require(len(pairs) == COMBINATIONS_PER_MECHANISM, f"{mechanism}: source-receiver quota mismatch")
        require({row["case_id"] for row in mechanism_rows} == {f"cap8m{mechanism_index:02d}c{index:02d}" for index in range(8)}, f"{mechanism}: case IDs mismatch")
        expected_status = ONTOLOGY_STATUS_PROTOCOL_V1 if mechanism in PROTOCOL_V1_MECHANISMS else ONTOLOGY_STATUS_DRAFT
        require({row["ontology_status"] for row in mechanism_rows} == {expected_status}, f"{mechanism}: ontology status mismatch")
        if mechanism in DRAFT_MECHANISMS:
            require(all(row["source_family"] == row["receiver_family"] for row in mechanism_rows), f"{mechanism}: compatibility family mismatch")

    for case_id, case_rows in by_case.items():
        require(len(case_rows) == REPETITIONS_PER_COMBINATION, f"{case_id}: repetition count mismatch")
        require({int(row["repetition_index"]) for row in case_rows} == {0, 1, 2}, f"{case_id}: repetition indices mismatch")
        invariant_fields = set(MANIFEST_FIELDS) - {"generation_id", "repetition_index", "seed"}
        for field in invariant_fields:
            require(len({row[field] for row in case_rows}) == 1, f"{case_id}: {field} changes across repetitions")
        for row in case_rows:
            require(
                int(row["seed"])
                == seed_for(int(row["mechanism_index"]), int(row["combination_index"]), int(row["repetition_index"])),
                f"{case_id}: seed formula mismatch",
            )


def csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def prompts_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        f"# {PROTOCOL_VERSION}: Original-only capability screen; order matches the canonical manifest",
        "",
    ]
    lines.extend(
        f"{row['prompt']} | {row['target_concept']} | {row['expected_footprint']}"
        for row in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def summary_payload(
    rows: Sequence[Mapping[str, str]],
    *,
    protocol_v1_registry_sha256: str,
    artifact_hashes: Mapping[str, str],
    canonical_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scope": {
            "intended_use": INTENDED_USE,
            "method_arm": METHOD_ARM,
            "treatment_status": TREATMENT_STATUS,
            "training_authorized": False,
            "evaluation_selection_authorized": False,
            "treatment_generation_authorized": False,
        },
        "counts": {
            "mechanisms": len(MECHANISM_ORDER),
            "source_receiver_combinations_per_mechanism": COMBINATIONS_PER_MECHANISM,
            "repetitions_per_combination": REPETITIONS_PER_COMBINATION,
            "rows_per_mechanism": ROWS_PER_MECHANISM,
            "total_rows": len(rows),
            "direct_per_mechanism": 12,
            "natural_per_mechanism": 12,
        },
        "video": {
            "num_frames": VIDEO_FRAMES,
            "fps": FPS,
            "clean_prefix_frames": CLEAN_PREFIX_FRAMES,
            "clean_prefix_seconds": CLEAN_PREFIX_FRAMES / FPS,
            "reference_interval_half_open": [REFERENCE_START_INCLUSIVE, REFERENCE_END_EXCLUSIVE],
        },
        "seed": {"base_seed": BASE_SEED, "formula": SEED_FORMULA, "globally_unique": True},
        "mechanism_order": list(MECHANISM_ORDER),
        "ontology_status": {
            mechanism: (ONTOLOGY_STATUS_PROTOCOL_V1 if mechanism in PROTOCOL_V1_MECHANISMS else ONTOLOGY_STATUS_DRAFT)
            for mechanism in MECHANISM_ORDER
        },
        "provenance": {
            "protocol_v1_registry": "data/protocol_v1/registry.json",
            "protocol_v1_registry_sha256": protocol_v1_registry_sha256,
            "protocol_v1_reused_mechanisms": list(PROTOCOL_V1_MECHANISMS),
            "draft_capability_only_mechanisms": list(DRAFT_MECHANISMS),
        },
        "checks": {
            "eight_mechanisms_equal_weight": True,
            "eight_unique_pairs_per_mechanism": True,
            "direct_natural_balanced": True,
            "three_fixed_repetitions_per_case": True,
            "first_16_frames_prompted_clean": True,
            "draft_compatibility_family_matched": True,
            "original_only": True,
            "unique_generation_ids": True,
            "unique_generation_seeds": True,
        },
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "artifact_sha256": dict(artifact_hashes),
    }


def artifact_paths(data_output_dir: Path, prompts_output_dir: Path) -> dict[str, Path]:
    return {
        "manifest_csv": data_output_dir / f"{ARTIFACT_STEM}_manifest.csv",
        "canonical_manifest_json": data_output_dir / f"{ARTIFACT_STEM}_manifest.canonical.json",
        "summary_json": data_output_dir / f"{ARTIFACT_STEM}_summary.json",
        "prompts": prompts_output_dir / f"{ARTIFACT_STEM}.prompts",
    }


def write_bytes_exclusive(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_artifacts_exclusive(
    paths: Mapping[str, Path],
    payloads: Mapping[str, bytes],
) -> None:
    require(set(paths) == set(payloads), "artifact path/payload mismatch")
    parents = {path.parent for path in paths.values()}
    for parent in parents:
        parent.mkdir(parents=True, exist_ok=True)
        require(parent.is_dir() and not parent.is_symlink(), f"artifact parent is not a regular directory: {parent}")
    collisions = sorted(str(path) for path in paths.values() if path.exists() or path.is_symlink())
    if collisions:
        raise FileExistsError("refusing to overwrite existing artifact(s): " + ", ".join(collisions))
    created: list[Path] = []
    try:
        for name in ("manifest_csv", "canonical_manifest_json", "prompts", "summary_json"):
            write_bytes_exclusive(paths[name], payloads[name])
            created.append(paths[name])
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def build_artifact_payloads(
    rows: Sequence[Mapping[str, str]],
    *,
    protocol_v1_registry_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    canonical = canonical_json_bytes(list(rows))
    primary_payloads = {
        "manifest_csv": csv_bytes(rows),
        "canonical_manifest_json": canonical,
        "prompts": prompts_bytes(rows),
    }
    artifact_hashes = {name: sha256_bytes(raw) for name, raw in primary_payloads.items()}
    summary = summary_payload(
        rows,
        protocol_v1_registry_sha256=protocol_v1_registry_sha256,
        artifact_hashes=artifact_hashes,
        canonical_manifest_sha256=sha256_bytes(canonical),
    )
    payloads = dict(primary_payloads)
    payloads["summary_json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payloads, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-v1-registry",
        type=Path,
        default=Path("data/protocol_v1/registry.json"),
    )
    parser.add_argument("--data-output-dir", type=Path, default=Path("data"))
    parser.add_argument("--prompts-output-dir", type=Path, default=Path("prompts"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_protocol_v1_registry(args.protocol_v1_registry)
    rows = build_rows(registry)
    payloads, summary = build_artifact_payloads(
        rows,
        protocol_v1_registry_sha256=sha256_file(args.protocol_v1_registry),
    )
    paths = artifact_paths(args.data_output_dir, args.prompts_output_dir)
    write_artifacts_exclusive(paths, payloads)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
