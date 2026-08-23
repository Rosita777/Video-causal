#!/usr/bin/env python3
"""Build the fresh, simplified eight-mechanism capability-v2 batch.

Protocol amendment from v1:

* all eight mechanism ontologies are fresh v2 registrations;
* each mechanism crosses two sources with two receivers in both direct and
  natural wording styles, with three deterministic generations per case;
* every prompt is exactly two sentences and at most 55 English words;
* sentence one gives only a simple receiver state, while sentence two gives
  one source action and one visible result;
* human actors and negative-prompt lists are forbidden;
* field-mediated response uses airflow-mediated non-contact motion; and
* frames 0--15 remain in the manifest only as diagnostic metadata, not as a
  v2 semantic eligibility requirement.

This builder is Original-only and writes new artifacts exclusively.  It does
not read, import, modify, or overwrite the v1 capability ontology or artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_8mechanism_capability_v2"
ARTIFACT_STEM = PROTOCOL_VERSION
ONTOLOGY_STATUS = "fresh_capability_v2"
ONTOLOGY_PROVENANCE = f"{PROTOCOL_VERSION}:fresh_simplified_ontology"
INTENDED_USE = "original_capability_screening_only"
METHOD_ARM = "original"
TREATMENT_STATUS = "pre_method_original_only"

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
PROMPT_STYLES = ("direct", "natural")
PHYSICAL_PAIR_INDICES = ((0, 0), (0, 1), (1, 0), (1, 1))

BASE_SEED = 940000
SEED_FORMULA = (
    "940000 + 1000*mechanism_index + 10*combination_index + repetition_index"
)
SOURCES_PER_MECHANISM = 2
RECEIVERS_PER_MECHANISM = 2
COMBINATIONS_PER_STYLE = 4
COMBINATIONS_PER_MECHANISM = 8
REPETITIONS_PER_COMBINATION = 3
ROWS_PER_MECHANISM = COMBINATIONS_PER_MECHANISM * REPETITIONS_PER_COMBINATION
VIDEO_FRAMES = 49
FPS = 8
DIAGNOSTIC_REFERENCE_START_INCLUSIVE = 0
DIAGNOSTIC_REFERENCE_END_EXCLUSIVE = 16
PROMPT_SENTENCE_COUNT = 2
PROMPT_MAX_ENGLISH_WORDS = 55

PROTOCOL_AMENDMENT: dict[str, Any] = {
    "amends": "causal_role_erasure_8mechanism_capability_v1",
    "applies_to": PROTOCOL_VERSION,
    "v1_artifacts_modified": False,
    "fresh_ontology_for_all_eight_mechanisms": True,
    "protocol_v1_registry_reused": False,
    "sources_per_mechanism": SOURCES_PER_MECHANISM,
    "receivers_per_mechanism": RECEIVERS_PER_MECHANISM,
    "physical_crossing": "2_sources_x_2_receivers",
    "wording_styles": list(PROMPT_STYLES),
    "cases_per_mechanism": COMBINATIONS_PER_MECHANISM,
    "repetitions_per_case": REPETITIONS_PER_COMBINATION,
    "prompt_sentence_count_exact": PROMPT_SENTENCE_COUNT,
    "prompt_max_english_words": PROMPT_MAX_ENGLISH_WORDS,
    "sentence_one": "simple_receiver_initial_state",
    "sentence_two": "one_source_action_and_one_visible_result",
    "exact_initial_timing_required": False,
    "source_offscreen_clause_required": False,
    "human_actor_terms_forbidden": True,
    "negative_prompt_list_forbidden": True,
    "field_mediated_subtype": "airflow_non_contact_motion",
    "reference_frames_0_15": "diagnostic_only_not_v2_eligibility",
}

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

FORBIDDEN_HUMAN_TERMS = re.compile(
    r"\b(?:hand|hands|person|people|human|man|woman)\b", re.IGNORECASE
)
FORBIDDEN_NEGATION_TERMS = re.compile(
    r"\b(?:no|not|without|never)\b", re.IGNORECASE
)
ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def _source(source_id: str, name: str, motion: str, family: str) -> dict[str, str]:
    return {"id": source_id, "name": name, "motion": motion, "family": family}


def _receiver(
    receiver_id: str,
    name: str,
    clean_state: str,
    initial_sentence_clause: str,
    family: str,
    footprint: str | None = None,
) -> dict[str, str]:
    item = {
        "id": receiver_id,
        "name": name,
        "clean_state": clean_state,
        "initial": initial_sentence_clause,
        "family": family,
    }
    if footprint is not None:
        item["footprint"] = footprint
    return item


MECHANISM_SPECS: dict[str, dict[str, Any]] = {
    "water_impact": {
        "name": "Water impact",
        "trigger": "visible entry of the source through the water surface",
        "footprint": "one splash and expanding circular ripples appear",
        "counterfactual_state": "the water stays flat and still",
        "sources": [
            _source(
                "v2_water_red_apple",
                "one small red apple",
                "falls straight down",
                "v2_water_impact",
            ),
            _source(
                "v2_water_gray_stone",
                "one smooth gray stone",
                "falls straight down",
                "v2_water_impact",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_water_white_basin",
                "a wide white ceramic basin filled with water",
                "the water surface is flat and still",
                "a wide white ceramic basin filled with water has a flat, still surface",
                "v2_water_impact",
            ),
            _receiver(
                "v2_water_steel_sink",
                "a stainless-steel sink filled with water",
                "the water surface is flat and still",
                "a stainless-steel sink filled with water has a flat, still surface",
                "v2_water_impact",
            ),
        ],
    },
    "rigid_collision": {
        "name": "Rigid collision",
        "trigger": "visible horizontal contact between the source and the upright receiver",
        "footprint": "the receiver tips over and slides a short distance",
        "counterfactual_state": "the receiver stays upright and in its original position",
        "sources": [
            _source(
                "v2_collision_red_ball",
                "one small red rubber ball",
                "rolls once from left to right",
                "v2_single_rigid_target",
            ),
            _source(
                "v2_collision_blue_ball",
                "one small blue rubber ball",
                "rolls once from left to right",
                "v2_single_rigid_target",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_collision_yellow_block",
                "one tall yellow wooden block",
                "the block stands upright and still",
                "one tall yellow wooden block stands upright and still",
                "v2_single_rigid_target",
            ),
            _receiver(
                "v2_collision_brown_box",
                "one small brown cardboard box",
                "the box stands upright and still",
                "one small brown cardboard box stands upright and still",
                "v2_single_rigid_target",
            ),
        ],
    },
    "brittle_fracture": {
        "name": "Brittle fracture",
        "trigger": "visible downward strike of the intact glass receiver",
        "footprint": "the glass cracks and breaks into several separate pieces",
        "counterfactual_state": "the glass receiver stays intact",
        "sources": [
            _source(
                "v2_fracture_steel_ball",
                "one polished steel ball",
                "falls straight down",
                "v2_gravity_fracture",
            ),
            _source(
                "v2_fracture_gray_stone",
                "one smooth gray stone",
                "falls straight down",
                "v2_gravity_fracture",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_fracture_glass_bowl",
                "one clear glass bowl on a dark tabletop",
                "the bowl is intact and still",
                "one clear glass bowl on a dark tabletop is intact and still",
                "v2_gravity_fracture",
            ),
            _receiver(
                "v2_fracture_glass_pane",
                "one thin clear glass pane lying flat on two dark supports",
                "the pane is intact and still",
                "one thin clear glass pane lying flat on two dark supports is intact and still",
                "v2_gravity_fracture",
            ),
        ],
    },
    "powder_impact": {
        "name": "Powder impact",
        "trigger": "visible downward contact with the flat powder bed",
        "footprint": "one short dust puff leaves a clear impact crater",
        "counterfactual_state": "the powder bed stays flat and undisturbed",
        "sources": [
            _source(
                "v2_powder_steel_ball",
                "one polished steel ball",
                "falls straight down",
                "v2_powder_impact",
            ),
            _source(
                "v2_powder_blue_cube",
                "one small blue wooden cube",
                "falls straight down",
                "v2_powder_impact",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_powder_white_flour",
                "a shallow black tray filled with white flour",
                "the flour surface is flat and still",
                "a shallow black tray filled with white flour has a flat, still surface",
                "v2_powder_impact",
            ),
            _receiver(
                "v2_powder_tan_sand",
                "a shallow black tray filled with fine tan sand",
                "the sand surface is flat and still",
                "a shallow black tray filled with fine tan sand has a flat, still surface",
                "v2_powder_impact",
            ),
        ],
    },
    "elastic_deformation": {
        "name": "Elastic deformation",
        "trigger": "visible downward contact that starts receiver deformation",
        "footprint": "the surface dips deeply and springs back",
        "counterfactual_state": "the elastic surface keeps its flat original shape",
        "sources": [
            _source(
                "v2_elastic_red_ball",
                "one red rubber ball",
                "falls straight down",
                "v2_vertical_elastic",
            ),
            _source(
                "v2_elastic_orange_ball",
                "one small orange basketball",
                "falls straight down",
                "v2_vertical_elastic",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_elastic_black_trampoline",
                "a small black trampoline",
                "the surface is flat and still",
                "a small black trampoline has a flat, still surface",
                "v2_vertical_elastic",
            ),
            _receiver(
                "v2_elastic_blue_square_trampoline",
                "a small square blue trampoline",
                "the surface is flat and still",
                "a small square blue trampoline has a flat, still surface",
                "v2_vertical_elastic",
            ),
        ],
    },
    "field_mediated_response": {
        "name": "Field-mediated airflow response",
        "trigger": "visible fan blades begin spinning while a clear gap remains",
        "footprint": "the lightweight receiver moves strongly in the airflow while a clear gap remains",
        "counterfactual_state": "the lightweight receiver stays in its original position",
        "sources": [
            _source(
                "v2_field_blue_desk_fan",
                "one small blue desk fan",
                "starts spinning its blades across a clear gap",
                "v2_airflow_non_contact",
            ),
            _source(
                "v2_field_black_desk_fan",
                "one small black desk fan",
                "starts spinning its blades across a clear gap",
                "v2_airflow_non_contact",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_field_red_pinwheel",
                "one red paper pinwheel",
                "the pinwheel is still",
                "one red paper pinwheel rests still",
                "v2_airflow_non_contact",
                "the red pinwheel spins rapidly in the airflow while a clear gap remains",
            ),
            _receiver(
                "v2_field_yellow_ribbon",
                "one long yellow ribbon",
                "the ribbon hangs straight and still",
                "one long yellow ribbon hangs straight and still",
                "v2_airflow_non_contact",
                "the yellow ribbon flutters strongly in the airflow while a clear gap remains",
            ),
        ],
    },
    "material_release": {
        "name": "Material release",
        "trigger": "visible single puncture of the sealed paper pouch",
        "footprint": "the pouch tears and its contents pour downward",
        "counterfactual_state": "the pouch stays sealed and retains its contents",
        "sources": [
            _source(
                "v2_release_wood_dowel",
                "one pointed wooden dowel",
                "drops straight down",
                "v2_puncture_release",
            ),
            _source(
                "v2_release_steel_spike",
                "one pointed silver steel spike",
                "drops straight down",
                "v2_puncture_release",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_release_blue_bead_pouch",
                "a thin white paper pouch filled with blue beads",
                "the pouch is sealed and still",
                "a thin white paper pouch filled with blue beads hangs sealed and still",
                "v2_puncture_release",
            ),
            _receiver(
                "v2_release_yellow_lentil_pouch",
                "a thin brown paper pouch filled with yellow lentils",
                "the pouch is sealed and still",
                "a thin brown paper pouch filled with yellow lentils hangs sealed and still",
                "v2_puncture_release",
            ),
        ],
    },
    "surface_trace": {
        "name": "Surface trace",
        "trigger": "visible single vertical press into the unmarked surface",
        "footprint": "a crisp geometric imprint remains",
        "counterfactual_state": "the surface stays smooth and unmarked",
        "sources": [
            _source(
                "v2_trace_round_stamp",
                "one round black rubber stamp",
                "presses straight down once and rises",
                "v2_press_trace",
            ),
            _source(
                "v2_trace_square_stamp",
                "one square red rubber stamp",
                "presses straight down once and rises",
                "v2_press_trace",
            ),
        ],
        "receivers": [
            _receiver(
                "v2_trace_damp_sand",
                "a flat tray of damp beige sand",
                "the sand is smooth and unmarked",
                "a flat tray of damp beige sand is smooth and unmarked",
                "v2_press_trace",
            ),
            _receiver(
                "v2_trace_gray_clay",
                "a flat light-gray clay pad",
                "the clay is smooth and unmarked",
                "a flat light-gray clay pad is smooth and unmarked",
                "v2_press_trace",
            ),
        ],
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def english_word_count(prompt: str) -> int:
    return len(ENGLISH_WORD.findall(prompt))


def prompt_sentences(prompt: str) -> list[str]:
    return [part.strip() for part in prompt.split(".") if part.strip()]


def _validate_text(value: Any, label: str) -> None:
    require(isinstance(value, str) and value.strip() == value and value, f"{label} is invalid")
    require("|" not in value and "\n" not in value and "\r" not in value, f"{label} contains a forbidden delimiter")


def validate_mechanism_specs(specs: Mapping[str, Mapping[str, Any]]) -> None:
    require(tuple(specs) == MECHANISM_ORDER, "expected exactly the eight canonical v2 mechanisms")
    all_source_ids: list[str] = []
    all_receiver_ids: list[str] = []
    for mechanism, spec in specs.items():
        for field in ("name", "trigger", "footprint", "counterfactual_state"):
            _validate_text(spec.get(field), f"{mechanism}.{field}")
        sources = spec.get("sources")
        receivers = spec.get("receivers")
        require(isinstance(sources, list) and len(sources) == 2, f"{mechanism}: expected two fresh v2 sources")
        require(isinstance(receivers, list) and len(receivers) == 2, f"{mechanism}: expected two fresh v2 receivers")
        for index, source in enumerate(sources):
            require(isinstance(source, dict), f"{mechanism}: source {index} must be an object")
            require(set(source) == {"id", "name", "motion", "family"}, f"{mechanism}: source schema mismatch")
            for field in source:
                _validate_text(source[field], f"{mechanism}.source[{index}].{field}")
            require(source["id"].startswith("v2_"), f"{mechanism}: source ID is not fresh v2")
            all_source_ids.append(source["id"])
        for index, receiver in enumerate(receivers):
            require(isinstance(receiver, dict), f"{mechanism}: receiver {index} must be an object")
            required_receiver_fields = {"id", "name", "clean_state", "initial", "family"}
            require(
                set(receiver) in (required_receiver_fields, required_receiver_fields | {"footprint"}),
                f"{mechanism}: receiver schema mismatch",
            )
            for field in receiver:
                _validate_text(receiver[field], f"{mechanism}.receiver[{index}].{field}")
            require(receiver["id"].startswith("v2_"), f"{mechanism}: receiver ID is not fresh v2")
            require(receiver["name"] in receiver["initial"], f"{mechanism}: receiver initial clause omits receiver name")
            all_receiver_ids.append(receiver["id"])
        require(len({source["id"] for source in sources}) == 2, f"{mechanism}: source IDs repeat")
        require(len({receiver["id"] for receiver in receivers}) == 2, f"{mechanism}: receiver IDs repeat")
        require(
            {source["family"] for source in sources}
            == {receiver["family"] for receiver in receivers},
            f"{mechanism}: fresh v2 compatibility family mismatch",
        )
    require(len(set(all_source_ids)) == 16, "fresh v2 source IDs must be globally unique")
    require(len(set(all_receiver_ids)) == 16, "fresh v2 receiver IDs must be globally unique")
    field = specs["field_mediated_response"]
    require(
        {item["family"] for item in field["sources"] + field["receivers"]}
        == {"v2_airflow_non_contact"},
        "field-mediated v2 ontology must use only airflow non-contact motion",
    )
    require(
        all("footprint" in receiver for receiver in field["receivers"]),
        "field-mediated v2 receivers must register receiver-specific airflow footprints",
    )


def expected_footprint(
    mechanism: str,
    spec: Mapping[str, Any],
    receiver: Mapping[str, str],
) -> str:
    if mechanism == "field_mediated_response":
        return receiver["footprint"]
    return str(spec["footprint"])


def _action(
    mechanism: str,
    source: Mapping[str, str],
    receiver: Mapping[str, str],
) -> str:
    source_name = source["name"]
    receiver_name = receiver["name"]
    if mechanism == "water_impact":
        return f"{source_name} falls straight down into the center of {receiver_name}"
    if mechanism == "rigid_collision":
        return f"{source_name} {source['motion']} and strikes {receiver_name} once"
    if mechanism == "brittle_fracture":
        return f"{source_name} falls straight down and strikes {receiver_name} once"
    if mechanism == "powder_impact":
        return f"{source_name} falls straight down into the center of {receiver_name}"
    if mechanism == "elastic_deformation":
        return f"{source_name} falls onto the center of {receiver_name} once"
    if mechanism == "field_mediated_response":
        return f"the blades of {source_name} begin spinning across a clear gap"
    if mechanism == "material_release":
        return f"{source_name} drops straight down and punctures {receiver_name} once"
    if mechanism == "surface_trace":
        return f"{source_name} presses straight down once into {receiver_name} and rises"
    raise KeyError(mechanism)


def capability_prompt(
    mechanism: str,
    spec: Mapping[str, Any],
    source: Mapping[str, str],
    receiver: Mapping[str, str],
    prompt_style: str,
) -> str:
    require(prompt_style in PROMPT_STYLES, f"unsupported prompt style: {prompt_style}")
    initial = f"Locked-camera close-up: {receiver['initial']}."
    action = _action(mechanism, source, receiver)
    footprint = expected_footprint(mechanism, spec, receiver)
    if prompt_style == "direct":
        event = f"Then {action}; {footprint}."
    else:
        event = f"{action[0].upper() + action[1:]}; as a natural result, {footprint}."
    prompt = f"{initial} {event}"
    validate_prompt(prompt, source=source, receiver=receiver, footprint=footprint)
    return prompt


def validate_prompt(
    prompt: str,
    *,
    source: Mapping[str, str],
    receiver: Mapping[str, str],
    footprint: str,
) -> None:
    _validate_text(prompt, "prompt")
    sentences = prompt_sentences(prompt)
    require(prompt.endswith("."), "prompt must end with a period")
    require(len(sentences) == PROMPT_SENTENCE_COUNT, "prompt must contain exactly two sentences")
    require(english_word_count(prompt) <= PROMPT_MAX_ENGLISH_WORDS, "prompt exceeds 55 English words")
    require(FORBIDDEN_HUMAN_TERMS.search(prompt) is None, "prompt contains a forbidden human actor term")
    require(FORBIDDEN_NEGATION_TERMS.search(prompt) is None, "prompt contains a forbidden negative clause")
    require("first two seconds" not in prompt.lower(), "prompt reintroduces exact initial timing")
    require("offscreen" not in prompt.lower(), "prompt reintroduces an offscreen clause")
    require(source["name"].lower() not in sentences[0].lower(), "source appears in the initial-state sentence")
    require(source["name"].lower() in sentences[1].lower(), "event sentence omits the source")
    require(receiver["name"].lower() in prompt.lower(), "prompt omits the receiver")
    require(footprint.lower() in sentences[1].lower(), "event sentence omits the registered footprint")


def seed_for(mechanism_index: int, combination_index: int, repetition_index: int) -> int:
    require(0 <= mechanism_index < len(MECHANISM_ORDER), "mechanism index outside seed domain")
    require(0 <= combination_index < COMBINATIONS_PER_MECHANISM, "combination index outside seed domain")
    require(0 <= repetition_index < REPETITIONS_PER_COMBINATION, "repetition index outside seed domain")
    return BASE_SEED + 1000 * mechanism_index + 10 * combination_index + repetition_index


def build_rows() -> list[dict[str, str]]:
    validate_mechanism_specs(MECHANISM_SPECS)
    rows: list[dict[str, str]] = []
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        spec = MECHANISM_SPECS[mechanism]
        for style_index, prompt_style in enumerate(PROMPT_STYLES):
            for pair_index, (source_index, receiver_index) in enumerate(PHYSICAL_PAIR_INDICES):
                combination_index = style_index * COMBINATIONS_PER_STYLE + pair_index
                source = spec["sources"][source_index]
                receiver = spec["receivers"][receiver_index]
                case_id = f"cap8v2m{mechanism_index:02d}c{combination_index:02d}"
                prompt = capability_prompt(mechanism, spec, source, receiver, prompt_style)
                for repetition_index in range(REPETITIONS_PER_COMBINATION):
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "generation_id": f"{case_id}r{repetition_index:02d}",
                        "case_id": case_id,
                        "mechanism_index": str(mechanism_index),
                        "combination_index": str(combination_index),
                        "repetition_index": str(repetition_index),
                        "mechanism": mechanism,
                        "mechanism_name": str(spec["name"]),
                        "ontology_status": ONTOLOGY_STATUS,
                        "ontology_provenance": ONTOLOGY_PROVENANCE,
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
                        "compatibility_rule": f"fresh_v2_family_match:{source['family']}",
                        "prompt": prompt,
                        "target_concept": source["name"],
                        "expected_footprint": expected_footprint(mechanism, spec, receiver),
                        "expected_counterfactual_state": str(spec["counterfactual_state"]),
                        "seed": str(seed_for(mechanism_index, combination_index, repetition_index)),
                        "seed_formula": SEED_FORMULA,
                        "num_frames": str(VIDEO_FRAMES),
                        "fps": str(FPS),
                        "reference_start_inclusive": str(DIAGNOSTIC_REFERENCE_START_INCLUSIVE),
                        "reference_end_exclusive": str(DIAGNOSTIC_REFERENCE_END_EXCLUSIVE),
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
    require(len({row["seed"] for row in rows}) == expected_total, "generation seeds are not unique")
    require(len({row["case_id"] for row in rows}) == 64, "case ID count mismatch")

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mechanism"]].append(row)
        by_case[row["case_id"]].append(row)
        require(row["protocol_version"] == PROTOCOL_VERSION, "row protocol version mismatch")
        require(row["generation_id"].startswith("cap8v2"), "generation ID is not fresh v2")
        require(row["case_id"].startswith("cap8v2"), "case ID is not fresh v2")
        require(row["ontology_status"] == ONTOLOGY_STATUS, "row ontology status mismatch")
        require(row["ontology_provenance"] == ONTOLOGY_PROVENANCE, "row ontology provenance mismatch")
        require(row["method_arm"] == METHOD_ARM, "capability v2 must remain Original-only")
        require(row["intended_use"] == INTENDED_USE, "row use expanded beyond capability screening")
        require(row["treatment_status"] == TREATMENT_STATUS, "treatment row found in capability v2")
        require(row["num_frames"] == "49" and row["fps"] == "8", "video contract mismatch")
        require(row["reference_start_inclusive"] == "0", "diagnostic reference start mismatch")
        require(row["reference_end_exclusive"] == "16", "diagnostic reference end mismatch")
        require(row["source_id"].startswith("v2_"), "source ID is not fresh v2")
        require(row["receiver_id"].startswith("v2_"), "receiver ID is not fresh v2")
        require(row["source_family"] == row["receiver_family"], "row compatibility family mismatch")
        require(row["seed_formula"] == SEED_FORMULA, "seed formula mismatch")
        require(
            int(row["seed"])
            == seed_for(
                int(row["mechanism_index"]),
                int(row["combination_index"]),
                int(row["repetition_index"]),
            ),
            "row seed does not match the v2 formula",
        )
        source = {
            "name": row["source_object"],
        }
        receiver = {
            "name": row["receiver"],
        }
        validate_prompt(
            row["prompt"],
            source=source,
            receiver=receiver,
            footprint=row["expected_footprint"],
        )

    require(tuple(grouped) == MECHANISM_ORDER, "manifest mechanism order mismatch")
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        mechanism_rows = grouped[mechanism]
        require(len(mechanism_rows) == ROWS_PER_MECHANISM, f"{mechanism}: row count mismatch")
        require(Counter(row["prompt_style"] for row in mechanism_rows) == {"direct": 12, "natural": 12}, f"{mechanism}: style quota mismatch")
        require(len({row["source_id"] for row in mechanism_rows}) == 2, f"{mechanism}: source count mismatch")
        require(len({row["receiver_id"] for row in mechanism_rows}) == 2, f"{mechanism}: receiver count mismatch")
        require(len({(row["source_id"], row["receiver_id"]) for row in mechanism_rows}) == 4, f"{mechanism}: physical crossing mismatch")
        for style in PROMPT_STYLES:
            style_rows = [row for row in mechanism_rows if row["prompt_style"] == style]
            require(len({(row["source_id"], row["receiver_id"]) for row in style_rows}) == 4, f"{mechanism}: {style} does not cover the 2x2 crossing")
            require(Counter(row["source_id"] for row in style_rows) == Counter({source_id: 6 for source_id in {row["source_id"] for row in style_rows}}), f"{mechanism}: {style} source balance mismatch")
            require(Counter(row["receiver_id"] for row in style_rows) == Counter({receiver_id: 6 for receiver_id in {row["receiver_id"] for row in style_rows}}), f"{mechanism}: {style} receiver balance mismatch")
        require({row["case_id"] for row in mechanism_rows} == {f"cap8v2m{mechanism_index:02d}c{index:02d}" for index in range(8)}, f"{mechanism}: case IDs mismatch")

    for case_id, case_rows in by_case.items():
        require(len(case_rows) == 3, f"{case_id}: repetition count mismatch")
        require({row["repetition_index"] for row in case_rows} == {"0", "1", "2"}, f"{case_id}: repetition indices mismatch")
        require(len({row["prompt"] for row in case_rows}) == 1, f"{case_id}: prompt changes across repetitions")
        invariant_fields = set(MANIFEST_FIELDS) - {"generation_id", "repetition_index", "seed"}
        for field in invariant_fields:
            require(len({row[field] for row in case_rows}) == 1, f"{case_id}: {field} changes across repetitions")


def csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def prompts_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        f"# {PROTOCOL_VERSION}: fresh simplified Original-only capability screen",
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
    artifact_hashes: Mapping[str, str],
    canonical_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_amendment": PROTOCOL_AMENDMENT,
        "scope": {
            "intended_use": INTENDED_USE,
            "method_arm": METHOD_ARM,
            "treatment_status": TREATMENT_STATUS,
            "training_authorized": False,
            "evaluation_selection_authorized": False,
            "treatment_generation_authorized": False,
        },
        "counts": {
            "mechanisms": 8,
            "sources_per_mechanism": 2,
            "receivers_per_mechanism": 2,
            "physical_pairs_per_mechanism": 4,
            "cases_per_mechanism": 8,
            "repetitions_per_case": 3,
            "rows_per_mechanism": 24,
            "total_cases": 64,
            "total_rows": len(rows),
            "direct_rows_per_mechanism": 12,
            "natural_rows_per_mechanism": 12,
        },
        "video": {
            "num_frames": VIDEO_FRAMES,
            "fps": FPS,
            "diagnostic_reference_interval_half_open": [
                DIAGNOSTIC_REFERENCE_START_INCLUSIVE,
                DIAGNOSTIC_REFERENCE_END_EXCLUSIVE,
            ],
            "diagnostic_reference_role": "diagnostic_only_not_v2_eligibility",
        },
        "prompt_contract": {
            "sentences_exact": PROMPT_SENTENCE_COUNT,
            "max_english_words": PROMPT_MAX_ENGLISH_WORDS,
            "max_observed_english_words": max(english_word_count(row["prompt"]) for row in rows),
            "simple_initial_state": True,
            "one_action_one_visible_result": True,
            "human_actor_terms_forbidden": True,
            "negative_prompt_list_forbidden": True,
        },
        "ontology": {
            "status": ONTOLOGY_STATUS,
            "provenance": ONTOLOGY_PROVENANCE,
            "all_eight_mechanisms_fresh": True,
            "protocol_v1_registry_reused": False,
            "field_mediated_subtype": "airflow_non_contact_motion",
        },
        "seed": {
            "base_seed": BASE_SEED,
            "formula": SEED_FORMULA,
            "globally_unique": True,
        },
        "mechanism_order": list(MECHANISM_ORDER),
        "recommended_aggregate_gate": {
            "rows_per_mechanism": 24,
            "minimum_eligible_rows": 12,
            "minimum_eligible_direct_rows": 6,
            "minimum_eligible_natural_rows": 6,
            "minimum_distinct_eligible_source_ids": 2,
            "minimum_distinct_eligible_receiver_ids": 2,
            "all_eight_mechanisms_must_pass": True,
            "clean_prefix_is_eligibility_requirement": False,
        },
        "checks": {
            "eight_mechanisms_equal_weight": True,
            "fresh_two_by_two_crossing_per_style": True,
            "direct_natural_balanced": True,
            "three_fixed_repetitions_per_case": True,
            "exactly_two_prompt_sentences": True,
            "prompt_word_cap_enforced": True,
            "original_only": True,
            "unique_generation_ids": True,
            "unique_generation_seeds": True,
            "exclusive_no_overwrite_output": True,
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
    for parent in {path.parent for path in paths.values()}:
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
) -> tuple[dict[str, bytes], dict[str, Any]]:
    validate_rows(rows)
    canonical = canonical_json_bytes(list(rows))
    primary_payloads = {
        "manifest_csv": csv_bytes(rows),
        "canonical_manifest_json": canonical,
        "prompts": prompts_bytes(rows),
    }
    artifact_hashes = {name: sha256_bytes(raw) for name, raw in primary_payloads.items()}
    summary = summary_payload(
        rows,
        artifact_hashes=artifact_hashes,
        canonical_manifest_sha256=sha256_bytes(canonical),
    )
    payloads = dict(primary_payloads)
    payloads["summary_json"] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return payloads, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-output-dir", type=Path, default=Path("data"))
    parser.add_argument("--prompts-output-dir", type=Path, default=Path("prompts"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_rows()
    payloads, summary = build_artifact_payloads(rows)
    paths = artifact_paths(args.data_output_dir, args.prompts_output_dir)
    write_artifacts_exclusive(paths, payloads)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
