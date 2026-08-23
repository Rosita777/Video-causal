#!/usr/bin/env python3
"""Build the complete seven-mechanism main-experiment data registration.

This is a static, fail-closed builder.  It creates ontologies, target-candidate
and formal-evaluation manifests, the Water/Fracture identification subset, the
18-run matrix, and six non-Water target-prompt shards.  It never generates or
inspects media.
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
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "causal_role_erasure_7m_single_seed_v2"
DATASET_VERSION = "causal_role_erasure_7mechanism_main_v2"
VIDEO = {"num_frames": 49, "fps": 8, "height": 480, "width": 832}
TARGET_BASE_SEED = 1_100_000
TRAINING_SEED = 26_000
WATER_TARGET_GENERATION_MANIFEST_SHA256 = (
    "406e4b2d06c80e415dca05bda2262924e07c3aea67398b5caca3687fca7c140d"
)
EXPECTED_INPUT_SHA256 = {
    "water_source_bank": "473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814",
    "water_train_pairs": "672fed568bb514561d882ef1f6c897563540299bfe7bd72e9550df8e259f3936",
    "water_screen": "f46bcc3d6d13d0cbce006119645786e868c48789131c76f9951f3323dc561ed3",
    "preserve_manifest": "05f1f2617ef1bb8a0706924856ebddb673de8b8ed32473c5b27cb4a4cfdd82c3",
}

MECHANISM_ORDER = (
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
    "material_release",
    "surface_trace",
)
HISTORICAL_CAPABILITY_INDEX = {
    "water_impact": 0,
    "rigid_collision": 1,
    "brittle_fracture": 2,
    "powder_impact": 3,
    "elastic_deformation": 4,
    "material_release": 6,
    "surface_trace": 7,
}
PROMPT_STYLES = ("direct", "natural")
FOOTPRINT_LEXICALIZATIONS = ("explicit", "implicit")
GENERALIZATION_GROUPS = (
    "holdout_source_fresh_receiver",
    "holdout_source_seen_receiver",
    "seen_source_fresh_receiver",
)
SOURCE_MEMBERSHIPS = ("original_training", "augmentation_bank", "eval_holdout")
SPECIFICITY_SUBTYPES = (
    "same_noun_noncausal",
    "role_swap_or_near_causal",
    "same_footprint_alternative_cause",
)
COMPACT_MECHANISMS = {
    "water_impact",
    "rigid_collision",
    "brittle_fracture",
    "powder_impact",
    "elastic_deformation",
}

TARGET_FIELDS = (
    "protocol_version",
    "candidate_id",
    "global_index",
    "mechanism_index",
    "historical_capability_index",
    "mechanism",
    "mechanism_name",
    "source_index",
    "source_id",
    "source_object",
    "receiver_index",
    "receiver_id",
    "receiver",
    "prompt_style",
    "factual_prompt",
    "target_prompt",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "seed",
    "num_frames",
    "fps",
    "target_origin",
    "prompt_shard",
    "prompt_shard_index",
    "height",
    "width",
    "seed_strategy",
    "intended_use",
    "historical_pair_id",
    "historical_screen_status",
    "target_video_path",
    "excluded_content_phrase",
)
FORMAL_FIELDS = (
    "protocol_version",
    "case_id",
    "global_case_index",
    "mechanism_index",
    "historical_capability_index",
    "mechanism",
    "mechanism_name",
    "case_kind",
    "generalization_group",
    "source_membership",
    "prompt_style",
    "footprint_lexicalization",
    "specificity_subtype",
    "semantic_replicate",
    "source_id",
    "source_object",
    "receiver_id",
    "receiver",
    "prompt",
    "expected_trigger",
    "expected_footprint",
    "expected_counterfactual_state",
    "protected_object",
    "acceptable_alternative_cause",
    "m6_pair_id",
    "seed",
    "seed_nonce",
    "num_frames",
    "fps",
    "height",
    "width",
)
IDENTIFICATION_FIELDS = (
    "protocol_version",
    "identification_case_index",
    "mechanism",
    "case_id",
    "case_kind",
    "generalization_group",
    "source_membership",
    "prompt_style",
    "footprint_lexicalization",
    "specificity_subtype",
    "seed",
    "included_streams",
    "additional_streams",
)
RUN_FIELDS = (
    "protocol_version",
    "run_index",
    "run_id",
    "mechanism_index",
    "mechanism",
    "arm",
    "run_group",
    "training_seed",
    "updates",
    "erase_updates",
    "preserve_updates",
    "selected_erase_rows_required",
    "preserve_rows",
    "lora_rank",
    "lora_alpha",
    "learning_rate",
    "target_teacher_weight",
    "preservation_weight",
    "checkpoint",
    "inference_scale",
    "authorization_state",
    "output_dir",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def slug(value: str) -> str:
    return normalize(value).replace(" ", "_")


def csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_frozen_input(path: Path, logical_name: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{logical_name} is not a regular file")
    require(
        sha256_file(path) == EXPECTED_INPUT_SHA256[logical_name],
        f"{logical_name} SHA-256 mismatch",
    )


def _source(
    mechanism: str,
    source_id: str,
    phrase: str,
    membership: str,
    semantic_family: str,
    *,
    bank_index: int | None,
    provenance: str,
    shared_pool_id: str = "",
) -> dict[str, Any]:
    head = normalize(phrase).split()[-1]
    return {
        "source_id": source_id,
        "canonical_phrase": phrase,
        "normalized_phrase": normalize(phrase),
        "head_lemma": head,
        "semantic_family": semantic_family,
        "membership": membership,
        "bank_index": bank_index,
        "physical_tags": list(MECHANISM_SPECS[mechanism]["source_physical_tags"]),
        "motion_phrase": MECHANISM_SPECS[mechanism]["source_motion"],
        "provenance": provenance,
        "shared_pool_id": shared_pool_id,
    }


def _receiver(
    mechanism: str,
    receiver_id: str,
    phrase: str,
    clean_state: str,
    membership: str,
    receiver_family: str,
    provenance: str,
) -> dict[str, Any]:
    return {
        "receiver_id": receiver_id,
        "canonical_phrase": phrase,
        "normalized_phrase": normalize(phrase),
        "head_lemma": normalize(phrase).split()[-1],
        "receiver_family": receiver_family,
        "clean_state": clean_state,
        "natural_motion": MECHANISM_SPECS[mechanism]["receiver_natural_motion"],
        "membership": membership,
        "physical_tags": list(MECHANISM_SPECS[mechanism]["receiver_physical_tags"]),
        "provenance": provenance,
    }


MECHANISM_SPECS: dict[str, dict[str, Any]] = {
    "water_impact": {
        "name": "Water impact",
        "trigger": "visible entry of the source through the water surface",
        "footprint": "a brief splash and expanding circular ripples",
        "prompt_footprint": "a brief splash appears and circular ripples expand across the water",
        "counterfactual": "the water remains smooth, level, and undisturbed",
        "source_end_state": "the source remains visible in or beside the receiver",
        "source_motion": "falls from above into the water",
        "receiver_natural_motion": "soft reflected light moves slowly across the water",
        "source_physical_tags": ("compact", "visible", "water_entry_compatible"),
        "receiver_physical_tags": ("open", "water_filled", "bounded"),
        "target_forbidden": ("fall", "drop", "enter", "impact", "splash", "ripple"),
        "footprint_forbidden": ("splash", "ripple"),
        "excluded": "source object, water entry or impact, splash, expanding circular ripples",
        "alternative_cause": "a small mechanical paddle briefly strikes the water",
    },
    "rigid_collision": {
        "name": "Rigid collision",
        "trigger": "visible contact between the moving source and one upright receiver",
        "footprint": "the receiver tips over and slides a short distance",
        "prompt_footprint": "the receiver tips over, slides a short distance, and remains there",
        "counterfactual": "the receiver remains upright and in its original position",
        "source_end_state": "the source stops visibly beside the receiver",
        "source_motion": "slides or rolls horizontally into the receiver",
        "receiver_natural_motion": "a soft highlight shifts gradually across the scene",
        "source_physical_tags": ("compact", "rigid", "slide_or_roll_compatible"),
        "receiver_physical_tags": ("single", "upright", "freestanding"),
        "target_forbidden": ("strike", "contact", "collid", "tip", "fall", "slide"),
        "footprint_forbidden": ("tip", "topple", "fall", "slide"),
        "excluded": "moving source, collision or contact, receiver tipping, receiver displacement",
        "alternative_cause": "a separate black ball rolls in and strikes the receiver",
    },
    "brittle_fracture": {
        "name": "Brittle fracture",
        "trigger": "visible downward impact on the intact brittle receiver",
        "footprint": "separate broken fragments remain around the impact point",
        "prompt_footprint": "the receiver breaks and separate fragments remain around the impact point",
        "counterfactual": "the brittle receiver remains intact and uncracked",
        "source_end_state": "the source rests visibly among or beside the receiver",
        "source_motion": "falls from above onto the brittle receiver",
        "receiver_natural_motion": "a reflected highlight moves gently across the intact surface",
        "source_physical_tags": ("compact", "rigid", "impact_compatible"),
        "receiver_physical_tags": ("brittle", "intact_initially", "supported"),
        "target_forbidden": ("strike", "impact", "crack", "break", "fragment"),
        "footprint_forbidden": ("crack", "break", "broken", "fragment"),
        "excluded": "falling impactor, impact or strike, cracks, broken fragments",
        "alternative_cause": "a small mechanical press descends onto the receiver",
    },
    "powder_impact": {
        "name": "Powder impact",
        "trigger": "visible downward contact with the flat powder bed",
        "footprint": "a deep round crater remains around the half-embedded source",
        "prompt_footprint": "the source stays half embedded and a deep round crater remains around it",
        "counterfactual": "the powder bed remains flat, smooth, and undisturbed",
        "source_end_state": "the source remains visible and half embedded",
        "source_motion": "falls from above into the powder bed",
        "receiver_natural_motion": "soft illumination changes gradually across the powder surface",
        "source_physical_tags": ("compact", "rigid", "powder_impact_compatible"),
        "receiver_physical_tags": ("open_tray", "loose_powder", "flat_initially"),
        "target_forbidden": ("fall", "drop", "impact", "contact", "crater", "embedded"),
        "footprint_forbidden": ("crater", "embedded", "dust", "plume"),
        "excluded": "falling source, powder impact, dust plume, crater, embedded object",
        "alternative_cause": "a round mechanical plunger presses once into the powder",
    },
    "elastic_deformation": {
        "name": "Elastic deformation",
        "trigger": "visible downward loading of the flat elastic bed",
        "footprint": "the elastic bed remains deeply bowed beneath the resting source",
        "prompt_footprint": "the source comes to rest and the elastic bed remains deeply bowed beneath it",
        "counterfactual": "the elastic bed remains flat, taut, and unloaded",
        "source_end_state": "the source rests visibly on the elastic bed",
        "source_motion": "falls from above onto the elastic bed",
        "receiver_natural_motion": "a soft highlight moves gradually across the taut bed",
        "source_physical_tags": ("compact", "visible", "elastic_loading_compatible"),
        "receiver_physical_tags": ("elastic", "taut_initially", "supported_frame"),
        "target_forbidden": ("fall", "drop", "load", "contact", "bow", "indent"),
        "footprint_forbidden": ("bow", "bowed", "indent", "deform", "sag"),
        "excluded": "falling source, elastic loading or contact, bowed bed, indentation",
        "alternative_cause": "a separate dark sandbag is lowered onto the elastic bed",
    },
    "material_release": {
        "name": "Material release",
        "trigger": "visible puncture of the sealed particle-filled pouch",
        "footprint": "the pouch tears and newly released particles collect below it",
        "prompt_footprint": "the pouch tears and newly released particles collect below it",
        "counterfactual": "the pouch remains sealed and retains all of its contents",
        "source_end_state": "the pointed source remains visible beside the puncture location",
        "source_motion": "moves downward and punctures the pouch",
        "receiver_natural_motion": "the sealed pouch sways almost imperceptibly in place",
        "source_physical_tags": ("rigid", "pointed", "puncture_compatible"),
        "receiver_physical_tags": ("sealed", "particle_filled", "suspended"),
        "target_forbidden": ("puncture", "pierce", "tear", "rupture", "release", "spill"),
        "footprint_forbidden": ("tear", "release", "particle", "spill", "pour"),
        "excluded": "pointed source, puncture or rupture, pouch tear, released particles",
        "alternative_cause": "a separate pair of scissors cuts the lower corner of the pouch",
    },
    "surface_trace": {
        "name": "Surface trace",
        "trigger": "visible vertical press into the initially smooth surface",
        "footprint": "a deep crisp shape-specific indentation remains in the surface",
        "prompt_footprint": "a deep crisp shape-specific indentation appears and remains in the surface",
        "counterfactual": "the susceptible surface remains smooth and completely unmarked",
        "source_end_state": "the stamp lifts slightly and remains visible in frame",
        "source_motion": "presses vertically into the surface and lifts slightly",
        "receiver_natural_motion": "a soft highlight moves gradually across the smooth surface",
        "source_physical_tags": ("rigid", "imprinting_face", "press_compatible"),
        "receiver_physical_tags": ("susceptible_surface", "smooth_initially", "bounded"),
        "target_forbidden": ("press", "contact", "stamp", "imprint", "indent", "groove"),
        "footprint_forbidden": ("imprint", "indent", "groove", "mark", "trace"),
        "excluded": "stamp or die, pressing contact, imprint, groove, indentation",
        "alternative_cause": "a separate embossing die presses once into the surface",
    },
}


ORIGINAL_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "water_impact": (
        ("water_droplet", "one large clear water droplet"),
        ("ice_cube", "one small transparent ice cube"),
        ("red_apple", "one small red apple"),
        ("green_lime", "one small green lime"),
        ("blue_marble", "one blue glass marble"),
        ("wooden_cube", "one small light wooden cube"),
        ("steel_ball", "one polished steel ball bearing"),
        ("plastic_block", "one small red plastic toy block"),
    ),
    "rigid_collision": (
        ("red_ball", "one small red rubber ball"),
        ("blue_ball", "one small blue rubber ball"),
        ("wood_cylinder", "one short wooden cylinder"),
        ("metal_puck", "one silver metal puck"),
        ("yellow_toy_car", "one small yellow toy car"),
        ("green_cube", "one small green wooden cube"),
        ("orange_roller", "one short orange plastic roller"),
        ("steel_bearing", "one polished steel bearing"),
    ),
    "brittle_fracture": (
        ("steel_ball", "one polished steel ball"),
        ("gray_stone", "one smooth gray stone"),
        ("wood_mallet", "one small wooden mallet"),
        ("metal_hammer", "one small metal hammer"),
        ("wood_baton", "one short wooden baton"),
        ("brass_weight", "one compact brass weight"),
        ("ceramic_pestle", "one solid ceramic pestle"),
        ("iron_block", "one compact iron block"),
    ),
    "powder_impact": (
        ("red_ball", "one large glossy red ball"),
        ("blue_ball", "one large glossy blue ball"),
        ("steel_ball", "one polished steel ball"),
        ("wood_cube", "one small wooden cube"),
        ("gray_stone", "one smooth gray stone"),
        ("brass_weight", "one compact brass weight"),
        ("orange_ball", "one large orange rubber ball"),
        ("black_puck", "one thick black metal puck"),
    ),
    "elastic_deformation": (
        ("orange_basketball", "one large orange basketball"),
        ("red_medicine_ball", "one heavy red medicine ball"),
        ("blue_ball", "one large blue rubber ball"),
        ("steel_sphere", "one polished steel sphere"),
        ("black_weight", "one compact black training weight"),
        ("yellow_ball", "one large yellow rubber ball"),
        ("wood_sphere", "one solid wooden sphere"),
        ("gray_stone", "one smooth gray stone"),
    ),
    "material_release": (
        ("wood_dowel", "one pointed wooden dowel"),
        ("steel_spike", "one pointed silver steel spike"),
        ("brass_awl", "one pointed brass awl"),
        ("iron_nail", "one long iron nail"),
        ("bamboo_skewer", "one rigid bamboo skewer"),
        ("steel_punch", "one narrow steel punch"),
        ("stone_point", "one sharp stone point"),
        ("metal_needle", "one thick metal needle"),
    ),
    "surface_trace": (
        ("round_black_stamp", "one large round black rubber stamp"),
        ("square_red_stamp", "one large square red rubber stamp"),
        ("triangle_blue_stamp", "one large triangular blue rubber stamp"),
        ("star_green_stamp", "one large star-shaped green rubber stamp"),
        ("hexagon_brass_die", "one large hexagonal brass face die"),
        ("oval_wood_stamp", "one large oval wooden stamp"),
        ("diamond_steel_die", "one large diamond-shaped steel die"),
        ("crescent_clay_stamp", "one large crescent-shaped clay stamp"),
    ),
}


TRAIN_RECEIVERS: dict[str, tuple[str, ...]] = {
    "water_impact": (
        "a calm shallow pond", "a transparent glass mixing bowl filled with water",
        "a wide white ceramic basin filled with water", "a clear rectangular glass tank filled with water",
        "a clean metal bucket filled with water", "a stainless-steel kitchen sink basin filled with water",
        "a plain porcelain soup bowl filled with water", "a black cooking pot filled with water",
        "a round stone birdbath filled with water", "a rectangular glass baking dish filled with water",
        "a blue plastic storage tub filled with water", "a laboratory beaker filled with clear water",
    ),
    "rigid_collision": (
        "one tall yellow wooden domino", "one small brown cardboard box", "one tall blue foam block",
        "one upright white plastic pin", "one narrow green wooden block", "one empty red aluminum can",
        "one tall cork cylinder", "one upright black game piece", "one narrow orange carton",
        "one tall purple plastic cup", "one upright silver metal tin", "one freestanding beige toy brick",
    ),
    "brittle_fracture": (
        "one clear glass bowl on a dark tabletop", "one thin clear glass pane on dark supports",
        "one empty clear wine glass", "one empty clear glass bottle", "one empty clear glass jar",
        "one clear glass tile on two supports", "one thin ceramic plate on dark supports",
        "one brittle plaster tile", "one hollow ceramic cup", "one clear laboratory vial",
        "one thin ice sheet on dark supports", "one brittle clay tablet",
    ),
    "powder_impact": (
        "a shallow black tray filled with white flour", "a shallow white tray filled with cocoa powder",
        "a shallow dark tray filled with cornstarch", "a shallow steel tray filled with fine tan sand",
        "a shallow blue tray filled with powdered sugar", "a shallow glass tray filled with chalk powder",
        "a shallow red tray filled with dry soil", "a shallow ceramic tray filled with fine ash",
        "a shallow brass tray filled with talcum powder", "a shallow gray tray filled with baking soda",
        "a shallow green tray filled with ground clay", "a shallow white tray filled with fine sawdust",
    ),
    "elastic_deformation": (
        "one small black trampoline with a taut bed", "one small round blue trampoline with a taut bed",
        "one square red elastic rebounder", "one circular green elastic training net",
        "one rectangular yellow stretch membrane", "one oval gray elastic exercise bed",
        "one small purple trampoline", "one compact orange rebound net", "one square white elastic mesh",
        "one round teal stretch bed", "one rectangular black elastic platform", "one oval blue rebounder",
    ),
    "material_release": (
        "a sealed white paper pouch filled with blue beads", "a sealed brown paper pouch filled with yellow lentils",
        "a sealed red fabric pouch filled with white pellets", "a sealed blue paper pouch filled with black seeds",
        "a sealed green fiber pouch filled with orange grains", "a sealed gray paper pouch filled with red beads",
        "a sealed yellow fabric pouch filled with brown pellets", "a sealed black paper pouch filled with white grains",
        "a sealed purple fiber pouch filled with silver beads", "a sealed orange paper pouch filled with green seeds",
        "a sealed beige fabric pouch filled with blue pellets", "a sealed teal paper pouch filled with yellow grains",
    ),
    "surface_trace": (
        "one smooth pale clay pad", "one smooth light-gray clay pad", "one flat tray of damp beige sand",
        "one smooth red modeling-clay slab", "one flat blue wax tablet", "one smooth green kinetic-sand pad",
        "one flat white plaster-clay slab", "one smooth brown soft-clay tablet", "one flat purple wax pad",
        "one smooth orange molding-sand tray", "one flat black putty slab", "one smooth yellow clay tile",
    ),
}


COMPACT_HOLDOUT_HEADS = (
    "orb", "sphere", "capsule", "prism", "tetrahedron", "octahedron", "dodecahedron", "icosahedron",
    "ellipsoid", "ovoid", "torus", "hemisphere", "pyramid", "wedge", "ingot", "billet",
    "slug", "brick", "tile", "cobble", "pebble", "stone", "baton", "rod",
    "dowel", "peg", "pin", "key", "lock", "buckle", "badge", "seal",
    "stud", "nail", "awl", "rasp", "file", "drillbit", "mandrel", "arbor",
    "shaft", "axle", "drum", "flywheel", "impeller", "manifold", "nozzle", "valve",
)
MATERIALS = ("steel", "brass", "iron", "bronze", "titanium", "ceramic", "stone", "alloy")
SHAPES = ("round", "square", "triangular", "oval", "hexagonal", "diamond-shaped", "star-shaped", "crescent-shaped")


def _formula_sources(mechanism: str, membership: str, count: int) -> list[dict[str, Any]]:
    require(mechanism in {"material_release", "surface_trace"}, "formula pool mechanism invalid")
    rows: list[dict[str, Any]] = []
    if mechanism == "material_release":
        kinds = (
            ("conical piercer", "fluted puncture tool", "tapered perforator", "needle-point probe", "triangular piercing bit", "faceted puncture pin", "hollow piercing tip")
            if membership == "augmentation_bank"
            else ("barbed opening tool", "lance-point cutter", "pyramidal pouch pick", "thorn-shaped opener", "arrow-point perforator", "beveled rupture pin")
        )
    else:
        kinds = (
            ("raised-face tool", "rigid-faced block", "handled face plate", "geometric face die", "patterned press tool", "sculpted-face block", "hard-faced seal")
            if membership == "augmentation_bank"
            else ("emblem face tool", "recessed face die", "symbolic face block", "textured-face press", "outlined face plate", "carved-face seal")
        )
    product = [(a, b) for a in (MATERIALS if mechanism == "material_release" else SHAPES) for b in kinds]
    require(len(product) == count, f"{mechanism} {membership} formula count mismatch")
    for index, (descriptor, kind) in enumerate(product):
        phrase = f"one palm-sized {descriptor} {kind}"
        prefix = "aug" if membership == "augmentation_bank" else "holdout"
        rows.append(
            _source(
                mechanism,
                f"{mechanism}_{prefix}_{index:02d}_{slug(descriptor + '_' + kind)}",
                phrase,
                membership,
                f"{mechanism}_{prefix}_family_{index:02d}",
                bank_index=(8 + index if membership == "augmentation_bank" else None),
                provenance=f"{DATASET_VERSION}:formulaic_physically_compatible_{prefix}_pool_v1",
            )
        )
    return rows


def _fresh_receivers(mechanism: str) -> list[dict[str, Any]]:
    descriptors = ("white", "black", "red", "blue", "green", "yellow", "gray", "orange")
    nouns = {
        "water_impact": ("basin", "bowl", "trough", "tank", "dish", "vat", "fountain bowl"),
        "rigid_collision": ("domino", "carton", "block", "pin", "canister", "cup", "game piece"),
        "brittle_fracture": ("glass bowl", "glass pane", "ceramic tile", "glass jar", "plaster plate", "clay tablet", "glass vial"),
        "powder_impact": ("flour tray", "sand tray", "cocoa tray", "chalk-powder tray", "soil tray", "ash tray", "clay-powder tray"),
        "elastic_deformation": ("trampoline", "rebounder", "elastic net", "stretch membrane", "exercise bed", "elastic platform", "rebound mesh"),
        "material_release": ("bead pouch", "grain pouch", "pellet pouch", "seed pouch", "lentil pouch", "sand pouch", "token pouch"),
        "surface_trace": ("clay pad", "wax tablet", "damp-sand tray", "putty slab", "molding-sand tile", "soft-clay plate", "kinetic-sand bed"),
    }[mechanism]
    rows: list[dict[str, Any]] = []
    for index, (color, noun) in enumerate((a, b) for a in descriptors for b in nouns):
        if mechanism == "water_impact":
            phrase = f"a {color} {noun} filled with clear water"
            clean = "the water surface is smooth, level, and calm"
        elif mechanism == "rigid_collision":
            phrase = f"one tall {color} {noun} standing upright"
            clean = "the single receiver is upright, freestanding, and still"
        elif mechanism == "brittle_fracture":
            phrase = f"one intact {color} {noun} on dark supports"
            clean = "the brittle receiver is intact, uncracked, and still"
        elif mechanism == "powder_impact":
            phrase = f"a shallow {color} {noun} with a flat surface"
            clean = "the loose surface is flat, smooth, and still"
        elif mechanism == "elastic_deformation":
            phrase = f"one small {color} {noun} with a taut flat bed"
            clean = "the elastic bed is flat, taut, unloaded, and still"
        elif mechanism == "material_release":
            phrase = f"a sealed {color} {noun} hanging above a clean tray"
            clean = "the pouch is sealed, intact, and retains all contents"
        else:
            phrase = f"one smooth {color} {noun} with its full border visible"
            clean = "the susceptible surface is smooth, flat, and unmarked"
        rows.append(
            _receiver(
                mechanism,
                f"{mechanism}_fresh_receiver_{index:02d}",
                phrase,
                clean,
                "fresh_eval",
                f"{mechanism}_fresh_{slug(noun)}",
                f"{DATASET_VERSION}:formulaic_fresh_receiver_pool_v1",
            )
        )
    require(len(rows) == 56, f"{mechanism}: fresh receiver count mismatch")
    return rows


def _train_receiver_rows(mechanism: str) -> list[dict[str, Any]]:
    rows = []
    for index, phrase in enumerate(TRAIN_RECEIVERS[mechanism]):
        if mechanism == "water_impact":
            clean = "the water surface is smooth, level, and calm"
        elif mechanism == "rigid_collision":
            clean = "the single receiver is upright, freestanding, and still"
        elif mechanism == "brittle_fracture":
            clean = "the brittle receiver is intact, uncracked, and still"
        elif mechanism == "powder_impact":
            clean = "the loose surface is flat, smooth, and still"
        elif mechanism == "elastic_deformation":
            clean = "the elastic bed is flat, taut, unloaded, and still"
        elif mechanism == "material_release":
            clean = "the pouch is sealed, intact, and retains all contents"
        else:
            clean = "the susceptible surface is smooth, flat, and unmarked"
        rows.append(
            _receiver(
                mechanism,
                f"{mechanism}_train_receiver_{index:02d}",
                phrase,
                clean,
                "training",
                f"{mechanism}_train_family_{index:02d}",
                f"{DATASET_VERSION}:registered_training_receiver_v1",
            )
        )
    return rows


def _event_clause(mechanism: str, source: str, receiver: str, style: str, *, eval_wording: bool) -> str:
    require(style in PROMPT_STYLES, "prompt style invalid")
    s = source.capitalize()
    if style == "natural":
        if mechanism == "water_impact":
            return (f"In one natural motion, {source} drops into the center of {receiver} and reaches the water surface" if eval_wording else f"As the shot continues naturally, {source} falls into the middle of {receiver} and touches the water")
        if mechanism == "rigid_collision":
            return (f"In one continuous motion, {source} travels across the floor into {receiver} with visible contact" if eval_wording else f"As the scene unfolds, {source} rolls or slides naturally across the floor and reaches {receiver}")
        if mechanism == "brittle_fracture":
            return (f"In a single natural descent, {source} drops onto {receiver} and makes visible impact" if eval_wording else f"As the scene unfolds, {source} falls naturally onto {receiver} and hits its center")
        if mechanism == "powder_impact":
            return (f"In one natural descent, {source} drops into the center of {receiver} and reaches the loose surface" if eval_wording else f"As the scene unfolds, {source} falls naturally into {receiver} and touches the powder bed")
        if mechanism == "elastic_deformation":
            return (f"In one natural descent, {source} drops onto the center of {receiver} and loads the elastic bed" if eval_wording else f"As the scene unfolds, {source} falls naturally onto {receiver} and settles on its elastic bed")
        if mechanism == "material_release":
            return (f"In one continuous downward motion, {source} reaches and visibly pierces {receiver}" if eval_wording else f"As the scene unfolds, {source} descends naturally and makes a single opening in {receiver}")
        if mechanism == "surface_trace":
            return (f"In one continuous motion, {source} lowers onto {receiver}, makes visible contact, and rises slightly" if eval_wording else f"As the scene unfolds, {source} lowers naturally onto {receiver}, touches once, and lifts a little")
        raise KeyError(mechanism)
    if mechanism == "water_impact":
        return (f"{s} descends into the center of {receiver} and visibly enters the water" if eval_wording else f"{s} falls from above into the center of {receiver} and visibly crosses the water surface")
    if mechanism == "rigid_collision":
        return (f"{s} moves across the floor and makes visible contact with {receiver}" if eval_wording else f"{s} slides or rolls horizontally and visibly strikes {receiver}")
    if mechanism == "brittle_fracture":
        return (f"{s} descends and makes visible impact with {receiver}" if eval_wording else f"{s} falls from above and visibly strikes {receiver}")
    if mechanism == "powder_impact":
        return (f"{s} descends into the center of {receiver} and visibly reaches the loose surface" if eval_wording else f"{s} falls from above into {receiver} and visibly contacts the powder bed")
    if mechanism == "elastic_deformation":
        return (f"{s} descends onto the center of {receiver} and visibly loads the elastic bed" if eval_wording else f"{s} falls from above onto {receiver} and makes visible downward contact")
    if mechanism == "material_release":
        return (f"{s} moves downward and visibly pierces {receiver}" if eval_wording else f"{s} descends and visibly punctures {receiver}")
    if mechanism == "surface_trace":
        return (f"{s} moves vertically into {receiver}, makes visible contact, and lifts slightly" if eval_wording else f"{s} presses straight down into {receiver} once and rises slightly")
    raise KeyError(mechanism)


def factual_prompt(
    mechanism: str,
    source: str,
    receiver: Mapping[str, Any],
    style: str,
    *,
    eval_wording: bool,
    footprint_lexicalization: str = "explicit",
) -> str:
    spec = MECHANISM_SPECS[mechanism]
    opener = "Locked-camera close-up" if eval_wording else "A simple realistic locked-camera video"
    event = _event_clause(mechanism, source, receiver["canonical_phrase"], style, eval_wording=eval_wording)
    prompt = f"{opener}: {receiver['canonical_phrase']} is visible; {receiver['clean_state']}. {event}."
    if footprint_lexicalization == "explicit":
        prompt += f" Afterward, {spec['prompt_footprint']}; {spec['source_end_state']}."
    else:
        require(footprint_lexicalization == "implicit", "footprint lexicalization invalid")
        prompt += f" The source stays in frame after the registered interaction."
    return prompt


def target_prompt(mechanism: str, receiver: Mapping[str, Any], style: str) -> str:
    spec = MECHANISM_SPECS[mechanism]
    if style == "direct":
        motion = spec["receiver_natural_motion"]
    else:
        motion = f"Throughout the shot, {spec['receiver_natural_motion']}"
    return (
        f"A simple realistic locked-camera close-up of {receiver['canonical_phrase']}. "
        f"{spec['counterfactual'].capitalize()}. {motion.capitalize()}. "
        "The viewpoint, receiver identity, lighting, and background geometry remain stable."
    )


def _specificity_prompt(
    mechanism: str,
    source: str,
    receiver: Mapping[str, Any],
    style: str,
    subtype: str,
) -> tuple[str, str]:
    spec = MECHANISM_SPECS[mechanism]
    prefix = "Locked-camera direct view" if style == "direct" else "In one natural continuous locked-camera shot"
    if subtype == "same_noun_noncausal":
        prompt = (
            f"{prefix}, {source} remains clearly visible and motionless on a display stand beside "
            f"{receiver['canonical_phrase']}. The protected object never acts on the receiver, which stays in its registered clean state."
        )
        alternative = ""
    elif subtype == "role_swap_or_near_causal":
        prompt = (
            f"{prefix}, {source} is a stationary bystander beside {receiver['canonical_phrase']}. "
            "A separate neutral object approaches slowly but stops with a clear gap before completing the registered trigger."
        )
        alternative = ""
    else:
        require(subtype == "same_footprint_alternative_cause", "specificity subtype invalid")
        alternative = spec["alternative_cause"]
        prompt = (
            f"{prefix}, {source} stays clearly visible and motionless on a display stand beside "
            f"{receiver['canonical_phrase']}. Then {alternative}; {spec['prompt_footprint']}. The protected object remains noncausal."
        )
    return prompt, alternative


def build_ontologies(water_source_bank_path: Path) -> dict[str, Any]:
    bank_payload = json.loads(water_source_bank_path.read_text(encoding="utf-8"))
    require(bank_payload.get("counts") == {"new_ontology": 56, "original_training": 8, "total": 64}, "water bank counts invalid")
    new_bank_entries = [row for row in bank_payload["entries"] if row["membership"] == "new_bank_source"]
    require(len(new_bank_entries) == 56, "water public bank must contain 56 new entries")
    compact_augmentation = tuple(
        (row["source_id"], row["source_phrase"], row["head_lemma"]) for row in new_bank_entries
    )
    ontologies: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        originals = [
            _source(
                mechanism, f"{mechanism}_original_{source_id}", phrase, "original_training",
                f"{mechanism}_original_family_{source_id}", bank_index=index,
                provenance=f"{DATASET_VERSION}:registered_original_training_source_v1",
            )
            for index, (source_id, phrase) in enumerate(ORIGINAL_SOURCES[mechanism])
        ]
        if mechanism in COMPACT_MECHANISMS:
            augmentation = [
                _source(
                    mechanism, f"{mechanism}_shared_aug_{source_id}", phrase, "augmentation_bank",
                    f"shared_compact_aug_{slug(head)}", bank_index=8 + index,
                    provenance="water_impact_dynamic_v4/source_bank_public64_registry_v2.json:shared_compact_pool_import",
                    shared_pool_id=f"compact_impact_aug_{index:02d}",
                )
                for index, (source_id, phrase, head) in enumerate(compact_augmentation)
            ]
            holdout = [
                _source(
                    mechanism, f"{mechanism}_shared_holdout_{index:02d}_{head}",
                    f"one palm-sized dense solid {MATERIALS[index % len(MATERIALS)]} {head}",
                    "eval_holdout", f"shared_compact_holdout_{head}", bank_index=None,
                    provenance=f"{DATASET_VERSION}:shared_compact_holdout_pool_v1",
                    shared_pool_id=f"compact_impact_holdout_{index:02d}",
                )
                for index, head in enumerate(COMPACT_HOLDOUT_HEADS)
            ]
        else:
            augmentation = _formula_sources(mechanism, "augmentation_bank", 56)
            holdout = _formula_sources(mechanism, "eval_holdout", 48)
        train_receivers = _train_receiver_rows(mechanism)
        fresh_receivers = _fresh_receivers(mechanism)
        ontology = {
            "mechanism_index": mechanism_index,
            "historical_capability_index": HISTORICAL_CAPABILITY_INDEX[mechanism],
            "mechanism": mechanism,
            "mechanism_name": MECHANISM_SPECS[mechanism]["name"],
            "registered_causal_structure": {
                "trigger": MECHANISM_SPECS[mechanism]["trigger"],
                "footprint": MECHANISM_SPECS[mechanism]["footprint"],
                "counterfactual_state": MECHANISM_SPECS[mechanism]["counterfactual"],
                "source_end_state": MECHANISM_SPECS[mechanism]["source_end_state"],
            },
            "prompt_templates": {
                "training": ["train_direct_v1", "train_natural_v1"],
                "formal_heldout": ["eval_direct_v1", "eval_natural_v1"],
                "specificity": list(SPECIFICITY_SUBTYPES),
                "construction": "structured_field_rebuild_no_substring_replacement",
            },
            "compatibility_rule": "all registered sources in each membership pool are physically compatible with all registered receivers for this mechanism",
            "compatibility_contract": {
                "policy": "homogeneous_all_to_all_fail_closed",
                "source_required_tags": list(MECHANISM_SPECS[mechanism]["source_physical_tags"]),
                "receiver_required_tags": list(MECHANISM_SPECS[mechanism]["receiver_physical_tags"]),
                "all_registered_pairs_compatible": True,
                "registered_pair_counts": {
                    "original_x_train": 8 * 12,
                    "bank64_x_train": 64 * 12,
                    "holdout48_x_fresh56": 48 * 56,
                    "holdout48_x_anchor8": 48 * 8,
                    "bank64_x_fresh56": 64 * 56,
                },
            },
            "excluded_constructs": (
                ["toy-car tracks", "ink stains"] if mechanism == "surface_trace" else
                ["fall-flat prompt refresh"] if mechanism == "rigid_collision" else []
            ),
            "original_training_sources": originals,
            "augmentation_sources": augmentation,
            "source_bank64": originals + augmentation,
            "eval_holdout_sources": holdout,
            "train_receivers": train_receivers,
            "fresh_eval_receivers": fresh_receivers,
            "seen_receiver_anchor_ids": [row["receiver_id"] for row in train_receivers[:8]],
            "shared_compact_pool_provenance": (
                "The same frozen compact-rigid augmentation and holdout vocabulary is imported into five independent mechanism registries; prompts, compatibility tags, receivers, and adapters remain mechanism-specific."
                if mechanism in COMPACT_MECHANISMS else "not_applicable"
            ),
        }
        validate_ontology(ontology)
        ontologies.append(ontology)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_version": DATASET_VERSION,
        "status": "frozen_pre_generation",
        "mechanism_order": list(MECHANISM_ORDER),
        "counts_per_mechanism": {
            "original_training_sources": 8,
            "augmentation_sources": 56,
            "source_bank": 64,
            "eval_holdout_sources": 48,
            "train_receivers": 12,
            "fresh_eval_receivers": 56,
            "seen_receiver_anchors": 8,
        },
        "mechanisms": ontologies,
    }
    validate_ontology_payload(payload)
    return payload


def contains_unresolved_sentinel(value: Any) -> bool:
    forbidden = re.compile(r"(?:^|[_\s-])(todo|tbd|placeholder|unknown)(?:$|[_\s-])", re.I)
    if isinstance(value, str):
        return forbidden.search(value) is not None
    if isinstance(value, Mapping):
        return any(contains_unresolved_sentinel(k) or contains_unresolved_sentinel(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_unresolved_sentinel(item) for item in value)
    return False


def validate_ontology(ontology: Mapping[str, Any]) -> None:
    mechanism = ontology["mechanism"]
    originals = ontology["original_training_sources"]
    augmentation = ontology["augmentation_sources"]
    bank = ontology["source_bank64"]
    holdout = ontology["eval_holdout_sources"]
    train_receivers = ontology["train_receivers"]
    fresh_receivers = ontology["fresh_eval_receivers"]
    anchors = ontology["seen_receiver_anchor_ids"]
    require((len(originals), len(augmentation), len(bank), len(holdout)) == (8, 56, 64, 48), f"{mechanism}: source counts invalid")
    require((len(train_receivers), len(fresh_receivers), len(anchors)) == (12, 56, 8), f"{mechanism}: receiver counts invalid")
    for label, rows, id_key, phrase_key in (
        ("bank", bank, "source_id", "canonical_phrase"),
        ("holdout", holdout, "source_id", "canonical_phrase"),
        ("train receivers", train_receivers, "receiver_id", "canonical_phrase"),
        ("fresh receivers", fresh_receivers, "receiver_id", "canonical_phrase"),
    ):
        require(len({row[id_key] for row in rows}) == len(rows), f"{mechanism}: duplicate {label} IDs")
        require(len({normalize(row[phrase_key]) for row in rows}) == len(rows), f"{mechanism}: duplicate {label} phrases")
        require(all(row[phrase_key].strip() == row[phrase_key] and row[phrase_key] for row in rows), f"{mechanism}: blank {label} phrase")
    require(not ({row["source_id"] for row in bank} & {row["source_id"] for row in holdout}), f"{mechanism}: bank/holdout source ID overlap")
    require(not ({row["normalized_phrase"] for row in bank} & {row["normalized_phrase"] for row in holdout}), f"{mechanism}: bank/holdout source phrase overlap")
    require(not ({row["semantic_family"] for row in bank} & {row["semantic_family"] for row in holdout}), f"{mechanism}: bank/holdout semantic-family overlap")
    require(not ({row["receiver_id"] for row in train_receivers} & {row["receiver_id"] for row in fresh_receivers}), f"{mechanism}: train/fresh receiver ID overlap")
    require(not ({row["normalized_phrase"] for row in train_receivers} & {row["normalized_phrase"] for row in fresh_receivers}), f"{mechanism}: train/fresh receiver phrase overlap")
    require(set(anchors) <= {row["receiver_id"] for row in train_receivers}, f"{mechanism}: anchor outside train receivers")
    require(len(set(anchors)) == 8, f"{mechanism}: anchor IDs repeat")
    require([row["bank_index"] for row in bank] == list(range(64)), f"{mechanism}: bank indices invalid")
    require(all(row["membership"] == "eval_holdout" for row in holdout), f"{mechanism}: holdout membership invalid")
    contract = ontology["compatibility_contract"]
    require(
        contract
        == {
            "policy": "homogeneous_all_to_all_fail_closed",
            "source_required_tags": list(MECHANISM_SPECS[mechanism]["source_physical_tags"]),
            "receiver_required_tags": list(MECHANISM_SPECS[mechanism]["receiver_physical_tags"]),
            "all_registered_pairs_compatible": True,
            "registered_pair_counts": {
                "original_x_train": 96,
                "bank64_x_train": 768,
                "holdout48_x_fresh56": 2688,
                "holdout48_x_anchor8": 384,
                "bank64_x_fresh56": 3584,
            },
        },
        f"{mechanism}: compatibility contract changed",
    )
    expected_source_tags = set(contract["source_required_tags"])
    expected_receiver_tags = set(contract["receiver_required_tags"])
    require(
        all(set(row["physical_tags"]) == expected_source_tags for row in bank + holdout),
        f"{mechanism}: source compatibility tags changed",
    )
    require(
        all(set(row["physical_tags"]) == expected_receiver_tags for row in train_receivers + fresh_receivers),
        f"{mechanism}: receiver compatibility tags changed",
    )


def validate_ontology_payload(payload: Mapping[str, Any]) -> None:
    mechanisms = payload["mechanisms"]
    require(len(mechanisms) == 7, "ontology must contain exactly seven mechanisms")
    require([row["mechanism"] for row in mechanisms] == list(MECHANISM_ORDER), "ontology mechanism order changed")
    require(
        [row["historical_capability_index"] for row in mechanisms]
        == [HISTORICAL_CAPABILITY_INDEX[mechanism] for mechanism in MECHANISM_ORDER],
        "historical capability indices changed",
    )
    all_source_ids: list[str] = []
    all_receiver_ids: list[str] = []
    for ontology in mechanisms:
        validate_ontology(ontology)
        all_source_ids.extend(row["source_id"] for row in ontology["source_bank64"])
        all_source_ids.extend(row["source_id"] for row in ontology["eval_holdout_sources"])
        all_receiver_ids.extend(row["receiver_id"] for row in ontology["train_receivers"])
        all_receiver_ids.extend(row["receiver_id"] for row in ontology["fresh_eval_receivers"])
    require(len(all_source_ids) == len(set(all_source_ids)), "source IDs are not globally unique")
    require(len(all_receiver_ids) == len(set(all_receiver_ids)), "receiver IDs are not globally unique")
    compact = [row for row in mechanisms if row["mechanism"] in COMPACT_MECHANISMS]
    require(len(compact) == 5, "shared compact mechanism count changed")
    augmentation_projection = [
        [(row["shared_pool_id"], row["canonical_phrase"]) for row in ontology["augmentation_sources"]]
        for ontology in compact
    ]
    holdout_projection = [
        [(row["shared_pool_id"], row["canonical_phrase"]) for row in ontology["eval_holdout_sources"]]
        for ontology in compact
    ]
    require(all(item == augmentation_projection[0] for item in augmentation_projection[1:]), "shared compact augmentation pool diverged")
    require(all(item == holdout_projection[0] for item in holdout_projection[1:]), "shared compact holdout pool diverged")
    require(not contains_unresolved_sentinel(payload), "ontology contains unresolved sentinel")


def _ontology_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["mechanism"]: row for row in payload["mechanisms"]}


def _contains_registered_lexeme(text: str, lexemes: Iterable[str]) -> bool:
    tokens = normalize(text).split()
    return any(token.startswith(normalize(lexeme)) for token in tokens for lexeme in lexemes)


def _validate_target_prompt(mechanism: str, prompt: str, source_phrase: str) -> None:
    require(normalize(source_phrase) not in normalize(prompt), f"{mechanism}: target prompt contains source phrase")
    require(
        not _contains_registered_lexeme(prompt, MECHANISM_SPECS[mechanism]["target_forbidden"]),
        f"{mechanism}: target prompt contains trigger/footprint language",
    )


def _target_seed(mechanism_index: int, source_index: int, receiver_index: int, style_index: int) -> int:
    return TARGET_BASE_SEED + 10_000 * mechanism_index + 100 * source_index + 2 * receiver_index + style_index


def build_target_candidates(
    ontology_payload: Mapping[str, Any],
    *,
    water_train_pairs_path: Path,
    water_screen_path: Path,
    prompt_dir: Path,
) -> tuple[list[dict[str, Any]], dict[Path, bytes]]:
    ontologies = _ontology_map(ontology_payload)
    historical_train = read_csv(water_train_pairs_path)
    historical_screen = read_csv(water_screen_path)
    require(len(historical_train) == 192 and len(historical_screen) == 192, "Water historical target inputs must contain 192 rows")
    screen_by_pair = {row["pair_id"]: row for row in historical_screen}
    require(len(screen_by_pair) == 192, "Water historical screen pair IDs repeat")
    require({row["pair_id"] for row in historical_train} == set(screen_by_pair), "Water historical target bindings disagree")
    require(Counter(row["final_status"] for row in historical_screen) == {"accept": 178, "reject": 14}, "Water historical target screen must remain 178/14")

    rows: list[dict[str, Any]] = []
    prompt_payloads: dict[Path, bytes] = {}
    used_seeds: set[int] = set()
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        ontology = ontologies[mechanism]
        sources = ontology["original_training_sources"]
        receivers = ontology["train_receivers"]
        source_by_historical = {
            source_id: (index, sources[index])
            for index, (source_id, _phrase) in enumerate(ORIGINAL_SOURCES[mechanism])
        }
        receiver_by_phrase = {
            normalize(row["canonical_phrase"]): (index, row) for index, row in enumerate(receivers)
        }
        shard_path = prompt_dir / f"target_candidates_{mechanism}.prompts"
        canonical_shard_path = (
            Path("prompts") / DATASET_VERSION / f"target_candidates_{mechanism}.prompts"
        ).as_posix()
        shard_lines: list[str] = []
        if mechanism == "water_impact":
            ordered_historical = historical_train
            require(
                Counter(row["prompt_variant"] for row in ordered_historical) == {"direct": 96, "natural": 96},
                "Water historical prompt-style balance changed",
            )
            for local_index, historical in enumerate(ordered_historical):
                require(historical["source_id"] in source_by_historical, "Water historical source is outside registered eight")
                source_index, source = source_by_historical[historical["source_id"]]
                require(normalize(historical["source_object"]) == source["normalized_phrase"], "Water historical source phrase changed")
                require(normalize(historical["receiver"]) in receiver_by_phrase, "Water historical receiver is outside registered twelve")
                receiver_index, receiver = receiver_by_phrase[normalize(historical["receiver"])]
                style = historical["prompt_variant"]
                require(style in PROMPT_STYLES, "Water historical prompt style invalid")
                style_index = PROMPT_STYLES.index(style)
                expected_local_index = source_index * 24 + receiver_index * 2 + style_index
                require(local_index == expected_local_index, "Water historical 8x12x2 order changed")
                screen = screen_by_pair[historical["pair_id"]]
                require(screen["seed"] == historical["seed"], "Water historical seed binding changed")
                require(screen["source_id"] == historical["source_id"], "Water historical source binding changed")
                require(screen["receiver_id"] == historical["receiver_id"], "Water historical receiver binding changed")
                seed = int(historical["seed"])
                factual = historical["training_prompt"]
                target = historical["target_generation_prompt"]
                _validate_target_prompt(mechanism, target, source["canonical_phrase"])
                candidate_id = f"tgt7m_m{mechanism_index:02d}_s{source_index:02d}_r{receiver_index:02d}_v{style_index}"
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "candidate_id": candidate_id,
                    "global_index": mechanism_index * 192 + local_index,
                    "mechanism_index": mechanism_index,
                    "historical_capability_index": HISTORICAL_CAPABILITY_INDEX[mechanism],
                    "mechanism": mechanism,
                    "mechanism_name": ontology["mechanism_name"],
                    "source_index": source_index,
                    "source_id": source["source_id"],
                    "source_object": source["canonical_phrase"],
                    "receiver_index": receiver_index,
                    "receiver_id": receiver["receiver_id"],
                    "receiver": receiver["canonical_phrase"],
                    "prompt_style": style,
                    "factual_prompt": factual,
                    "target_prompt": target,
                    "expected_trigger": MECHANISM_SPECS[mechanism]["trigger"],
                    "expected_footprint": MECHANISM_SPECS[mechanism]["footprint"],
                    "expected_counterfactual_state": MECHANISM_SPECS[mechanism]["counterfactual"],
                    "seed": seed,
                    "num_frames": VIDEO["num_frames"],
                    "fps": VIDEO["fps"],
                    "target_origin": "water_v1_reuse",
                    "prompt_shard": "",
                    "prompt_shard_index": "",
                    "height": VIDEO["height"],
                    "width": VIDEO["width"],
                    "seed_strategy": "historical_water_v1_seed",
                    "intended_use": "training_target_candidate",
                    "historical_pair_id": historical["pair_id"],
                    "historical_screen_status": screen["final_status"],
                    "target_video_path": screen["video_path"],
                    "excluded_content_phrase": MECHANISM_SPECS[mechanism]["excluded"],
                }
                require(tuple(row) == TARGET_FIELDS, "target field order changed")
                rows.append(row)
                require(seed not in used_seeds, "target seeds repeat")
                used_seeds.add(seed)
        else:
            for source_index, source in enumerate(sources):
                for receiver_index, receiver in enumerate(receivers):
                    for style_index, style in enumerate(PROMPT_STYLES):
                        local_index = source_index * 24 + receiver_index * 2 + style_index
                        candidate_id = f"tgt7m_m{mechanism_index:02d}_s{source_index:02d}_r{receiver_index:02d}_v{style_index}"
                        factual = factual_prompt(mechanism, source["canonical_phrase"], receiver, style, eval_wording=False)
                        target = target_prompt(mechanism, receiver, style)
                        _validate_target_prompt(mechanism, target, source["canonical_phrase"])
                        seed = _target_seed(mechanism_index, source_index, receiver_index, style_index)
                        row = {
                            "protocol_version": PROTOCOL_VERSION,
                            "candidate_id": candidate_id,
                            "global_index": mechanism_index * 192 + local_index,
                            "mechanism_index": mechanism_index,
                            "historical_capability_index": HISTORICAL_CAPABILITY_INDEX[mechanism],
                            "mechanism": mechanism,
                            "mechanism_name": ontology["mechanism_name"],
                            "source_index": source_index,
                            "source_id": source["source_id"],
                            "source_object": source["canonical_phrase"],
                            "receiver_index": receiver_index,
                            "receiver_id": receiver["receiver_id"],
                            "receiver": receiver["canonical_phrase"],
                            "prompt_style": style,
                            "factual_prompt": factual,
                            "target_prompt": target,
                            "expected_trigger": MECHANISM_SPECS[mechanism]["trigger"],
                            "expected_footprint": MECHANISM_SPECS[mechanism]["footprint"],
                            "expected_counterfactual_state": MECHANISM_SPECS[mechanism]["counterfactual"],
                            "seed": seed,
                            "num_frames": VIDEO["num_frames"],
                            "fps": VIDEO["fps"],
                            "target_origin": "new_generation_v2",
                            "prompt_shard": canonical_shard_path,
                            "prompt_shard_index": local_index,
                            "height": VIDEO["height"],
                            "width": VIDEO["width"],
                            "seed_strategy": "1100000_plus_10000m_plus_100s_plus_2r_plus_v",
                            "intended_use": "training_target_candidate",
                            "historical_pair_id": "",
                            "historical_screen_status": "",
                            "target_video_path": f"outputs/{DATASET_VERSION}/target_generation/{mechanism}/videos/{candidate_id}.mp4",
                            "excluded_content_phrase": MECHANISM_SPECS[mechanism]["excluded"],
                        }
                        require(tuple(row) == TARGET_FIELDS, "target field order changed")
                        rows.append(row)
                        require(seed not in used_seeds, "target seeds repeat")
                        used_seeds.add(seed)
                        shard_lines.append(
                            f"{target} | {MECHANISM_SPECS[mechanism]['excluded']} | {MECHANISM_SPECS[mechanism]['counterfactual']}"
                        )
            require(len(shard_lines) == 192, f"{mechanism}: target prompt shard count invalid")
            prompt_payloads[shard_path] = ("\n".join(shard_lines) + "\n").encode("utf-8")
    validate_target_rows(rows)
    return rows, prompt_payloads


def validate_target_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 1_344, "target candidate count must be 1344")
    require([int(row["global_index"]) for row in rows] == list(range(1_344)), "target global indices invalid")
    require(len({row["candidate_id"] for row in rows}) == 1_344, "target candidate IDs repeat")
    require(len({int(row["seed"]) for row in rows}) == 1_344, "target seeds repeat")
    require(Counter(row["target_origin"] for row in rows) == {"water_v1_reuse": 192, "new_generation_v2": 1_152}, "target origins invalid")
    for mechanism in MECHANISM_ORDER:
        selected = [row for row in rows if row["mechanism"] == mechanism]
        require(len(selected) == 192, f"{mechanism}: target candidate count invalid")
        require(Counter(row["prompt_style"] for row in selected) == {"direct": 96, "natural": 96}, f"{mechanism}: target style balance invalid")
        require(len({row["source_id"] for row in selected}) == 8, f"{mechanism}: target source count invalid")
        require(len({row["receiver_id"] for row in selected}) == 12, f"{mechanism}: target receiver count invalid")
        require(len({(row["source_id"], row["receiver_id"], row["prompt_style"]) for row in selected}) == 192, f"{mechanism}: target crossing incomplete")
        if mechanism == "water_impact":
            require(all(row["prompt_shard"] == row["prompt_shard_index"] == "" for row in selected), "Water reuse rows must not enter a new prompt shard")
        else:
            expected_shard = (Path("prompts") / DATASET_VERSION / f"target_candidates_{mechanism}.prompts").as_posix()
            require({row["prompt_shard"] for row in selected} == {expected_shard}, f"{mechanism}: canonical prompt-shard path invalid")
            require([int(row["prompt_shard_index"]) for row in selected] == list(range(192)), f"{mechanism}: prompt-shard indices invalid")
        for row in selected:
            require(tuple(row) == TARGET_FIELDS, "target row schema changed")


def _load_eval_salt(path: Path) -> tuple[bytes, str]:
    require(path.is_file() and not path.is_symlink(), "evaluation seed salt must be a regular file")
    raw = path.read_bytes()
    stripped = raw.strip()
    if len(raw) == 32:
        salt = raw
    elif re.fullmatch(rb"[0-9a-f]{64}", stripped):
        salt = bytes.fromhex(stripped.decode("ascii"))
    else:
        raise ValueError("evaluation seed salt must contain exactly 32 raw bytes or 64 lowercase hex characters")
    require(len(set(salt)) >= 8, "evaluation seed salt has insufficient byte diversity")
    return salt, sha256_bytes(salt)


def _eval_seed(case_id: str, salt: bytes, forbidden: set[int], used: set[int]) -> tuple[int, int]:
    for nonce in range(1_000_000):
        digest = hashlib.sha256(
            b"causal-role-erasure-7m-eval-seed-v1\0"
            + salt
            + b"\0"
            + case_id.encode("utf-8")
            + b"\0"
            + str(nonce).encode("ascii")
        ).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed != 0 and seed not in forbidden and seed not in used:
            return seed, nonce
    raise ValueError(f"unable to derive a unique evaluation seed for {case_id}")


def _causal_source_receiver(
    ontology: Mapping[str, Any], group_index: int, cell: int, replicate: int
) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    if group_index == 0:
        return ontology["eval_holdout_sources"][cell], ontology["fresh_eval_receivers"][cell], "eval_holdout"
    if group_index == 1:
        anchors = {row["receiver_id"]: row for row in ontology["train_receivers"]}
        return ontology["eval_holdout_sources"][8 + cell], anchors[ontology["seen_receiver_anchor_ids"][cell]], "eval_holdout"
    require(group_index == 2, "causal group index invalid")
    source_cell = cell // 2
    source = ontology["original_training_sources"][source_cell] if replicate == 0 else ontology["augmentation_sources"][source_cell]
    membership = "original_training" if replicate == 0 else "augmentation_bank"
    return source, ontology["fresh_eval_receivers"][8 + cell], membership


def _m6_id(mechanism: str, membership: str, style: str) -> str:
    return f"m6_{mechanism}_{membership}_{style}"


def build_formal_cases(
    ontology_payload: Mapping[str, Any], salt: bytes, target_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ontologies = _ontology_map(ontology_payload)
    forbidden = {int(row["seed"]) for row in target_rows} | {TRAINING_SEED} | set(range(979_000, 979_010))
    for mechanism_index in range(8):
        for combination_index in range(8):
            for repetition_index in range(3):
                forbidden.add(940_000 + 1_000 * mechanism_index + 10 * combination_index + repetition_index)
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(MECHANISM_ORDER):
        ontology = ontologies[mechanism]
        local_case_index = 0
        for group_index, group in enumerate(GENERALIZATION_GROUPS):
            for style_index, style in enumerate(PROMPT_STYLES):
                for lexical_index, lexicalization in enumerate(FOOTPRINT_LEXICALIZATIONS):
                    for replicate in range(2):
                        cell = 4 * style_index + 2 * lexical_index + replicate
                        source, receiver, membership = _causal_source_receiver(ontology, group_index, cell, replicate)
                        prompt = factual_prompt(
                            mechanism, source["canonical_phrase"], receiver, style,
                            eval_wording=True, footprint_lexicalization=lexicalization,
                        )
                        if lexicalization == "implicit":
                            require(
                                not _contains_registered_lexeme(prompt, MECHANISM_SPECS[mechanism]["footprint_forbidden"]),
                                f"{mechanism}: implicit causal prompt leaks footprint language",
                            )
                        else:
                            require(MECHANISM_SPECS[mechanism]["prompt_footprint"] in prompt, f"{mechanism}: explicit causal prompt omits footprint")
                        case_id = f"eval7m_m{mechanism_index:02d}_c{local_case_index:02d}"
                        seed, nonce = _eval_seed(case_id, salt, forbidden, used)
                        used.add(seed)
                        m6_pair_id = ""
                        if lexicalization == "explicit":
                            if group_index == 2:
                                m6_pair_id = _m6_id(mechanism, membership, style)
                            elif group_index == 0 and replicate == 0:
                                m6_pair_id = _m6_id(mechanism, "eval_holdout", style)
                        row = {
                            "protocol_version": PROTOCOL_VERSION,
                            "case_id": case_id,
                            "global_case_index": len(rows),
                            "mechanism_index": mechanism_index,
                            "historical_capability_index": HISTORICAL_CAPABILITY_INDEX[mechanism],
                            "mechanism": mechanism,
                            "mechanism_name": ontology["mechanism_name"],
                            "case_kind": "causal",
                            "generalization_group": group,
                            "source_membership": membership,
                            "prompt_style": style,
                            "footprint_lexicalization": lexicalization,
                            "specificity_subtype": "",
                            "semantic_replicate": replicate,
                            "source_id": source["source_id"],
                            "source_object": source["canonical_phrase"],
                            "receiver_id": receiver["receiver_id"],
                            "receiver": receiver["canonical_phrase"],
                            "prompt": prompt,
                            "expected_trigger": MECHANISM_SPECS[mechanism]["trigger"],
                            "expected_footprint": MECHANISM_SPECS[mechanism]["footprint"],
                            "expected_counterfactual_state": MECHANISM_SPECS[mechanism]["counterfactual"],
                            "protected_object": "",
                            "acceptable_alternative_cause": "",
                            "m6_pair_id": m6_pair_id,
                            "seed": seed,
                            "seed_nonce": nonce,
                            "num_frames": VIDEO["num_frames"],
                            "fps": VIDEO["fps"],
                            "height": VIDEO["height"],
                            "width": VIDEO["width"],
                        }
                        require(tuple(row) == FORMAL_FIELDS, "formal case field order changed")
                        rows.append(row)
                        local_case_index += 1
        require(local_case_index == 24, f"{mechanism}: causal local-case count invalid")
        for membership_index, membership in enumerate(SOURCE_MEMBERSHIPS):
            for style_index, style in enumerate(PROMPT_STYLES):
                for subtype_index, subtype in enumerate(SPECIFICITY_SUBTYPES):
                    if membership == "original_training":
                        source = ontology["original_training_sources"][(2 * style_index + subtype_index) % 8]
                    elif membership == "augmentation_bank":
                        source = ontology["augmentation_sources"][(6 + 3 * style_index + subtype_index) % 56]
                    else:
                        source = ontology["eval_holdout_sources"][16 + 3 * style_index + subtype_index]
                    receiver = ontology["fresh_eval_receivers"][16 + membership_index * 6 + style_index * 3 + subtype_index]
                    m6_pair_id = ""
                    if subtype == "same_noun_noncausal":
                        m6_pair_id = _m6_id(mechanism, membership, style)
                        if membership == "original_training":
                            source = ontology["original_training_sources"][2 * style_index]
                            receiver = ontology["fresh_eval_receivers"][8 + 4 * style_index]
                        elif membership == "augmentation_bank":
                            source = ontology["augmentation_sources"][2 * style_index]
                            receiver = ontology["fresh_eval_receivers"][9 + 4 * style_index]
                        else:
                            source = ontology["eval_holdout_sources"][4 * style_index]
                            receiver = ontology["fresh_eval_receivers"][4 * style_index]
                    prompt, alternative = _specificity_prompt(mechanism, source["canonical_phrase"], receiver, style, subtype)
                    case_id = f"eval7m_m{mechanism_index:02d}_c{local_case_index:02d}"
                    seed, nonce = _eval_seed(case_id, salt, forbidden, used)
                    used.add(seed)
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "case_id": case_id,
                        "global_case_index": len(rows),
                        "mechanism_index": mechanism_index,
                        "historical_capability_index": HISTORICAL_CAPABILITY_INDEX[mechanism],
                        "mechanism": mechanism,
                        "mechanism_name": ontology["mechanism_name"],
                        "case_kind": "specificity",
                        "generalization_group": "specificity",
                        "source_membership": membership,
                        "prompt_style": style,
                        "footprint_lexicalization": "",
                        "specificity_subtype": subtype,
                        "semantic_replicate": "",
                        "source_id": source["source_id"],
                        "source_object": source["canonical_phrase"],
                        "receiver_id": receiver["receiver_id"],
                        "receiver": receiver["canonical_phrase"],
                        "prompt": prompt,
                        "expected_trigger": MECHANISM_SPECS[mechanism]["trigger"],
                        "expected_footprint": MECHANISM_SPECS[mechanism]["footprint"],
                        "expected_counterfactual_state": "the protected source remains visible and noncausal",
                        "protected_object": source["canonical_phrase"],
                        "acceptable_alternative_cause": alternative,
                        "m6_pair_id": m6_pair_id,
                        "seed": seed,
                        "seed_nonce": nonce,
                        "num_frames": VIDEO["num_frames"],
                        "fps": VIDEO["fps"],
                        "height": VIDEO["height"],
                        "width": VIDEO["width"],
                    }
                    require(tuple(row) == FORMAL_FIELDS, "formal case field order changed")
                    rows.append(row)
                    local_case_index += 1
        require(local_case_index == 42, f"{mechanism}: formal local-case count invalid")
    validate_formal_cases(rows, forbidden)
    return rows


def validate_formal_cases(rows: Sequence[Mapping[str, Any]], forbidden: set[int]) -> None:
    require(len(rows) == 294, "formal case count must be 294")
    require([int(row["global_case_index"]) for row in rows] == list(range(294)), "formal global indices invalid")
    require(len({row["case_id"] for row in rows}) == 294, "formal case IDs repeat")
    seeds = {int(row["seed"]) for row in rows}
    require(len(seeds) == 294 and not seeds & forbidden, "formal seeds repeat or overlap a forbidden seed")
    for mechanism in MECHANISM_ORDER:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        causal = [row for row in mechanism_rows if row["case_kind"] == "causal"]
        specificity = [row for row in mechanism_rows if row["case_kind"] == "specificity"]
        require(len(causal) == 24 and len(specificity) == 18, f"{mechanism}: formal 24/18 split invalid")
        require(Counter(row["generalization_group"] for row in causal) == {group: 8 for group in GENERALIZATION_GROUPS}, f"{mechanism}: causal groups unbalanced")
        for group in GENERALIZATION_GROUPS:
            group_rows = [row for row in causal if row["generalization_group"] == group]
            require(Counter((row["prompt_style"], row["footprint_lexicalization"]) for row in group_rows) == {(style, lexical): 2 for style in PROMPT_STYLES for lexical in FOOTPRINT_LEXICALIZATIONS}, f"{mechanism}: causal cell balance invalid")
        require(Counter(row["source_membership"] for row in specificity) == {membership: 6 for membership in SOURCE_MEMBERSHIPS}, f"{mechanism}: specificity memberships unbalanced")
        require(Counter(row["prompt_style"] for row in specificity) == {style: 9 for style in PROMPT_STYLES}, f"{mechanism}: specificity wording unbalanced")
        require(Counter(row["specificity_subtype"] for row in specificity) == {subtype: 6 for subtype in SPECIFICITY_SUBTYPES}, f"{mechanism}: specificity subtypes unbalanced")
        m6 = [row for row in mechanism_rows if row["m6_pair_id"]]
        require(Counter(row["m6_pair_id"] for row in m6) == {_m6_id(mechanism, membership, style): 2 for membership in SOURCE_MEMBERSHIPS for style in PROMPT_STYLES}, f"{mechanism}: M6 pairing invalid")
        for pair_id in {_m6_id(mechanism, membership, style) for membership in SOURCE_MEMBERSHIPS for style in PROMPT_STYLES}:
            pair = [row for row in m6 if row["m6_pair_id"] == pair_id]
            require({row["case_kind"] for row in pair} == {"causal", "specificity"}, f"{mechanism}: M6 lacks causal/specificity pair")
            require(len({row["source_id"] for row in pair}) == 1, f"{mechanism}: M6 source identity mismatch")
            require(len({row["receiver_id"] for row in pair}) == 1, f"{mechanism}: M6 receiver identity mismatch")


def build_identification_subset(formal_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for mechanism in ("water_impact", "brittle_fracture"):
        rows = [row for row in formal_rows if row["mechanism"] == mechanism]
        causal = []
        for group_index, group in enumerate(GENERALIZATION_GROUPS):
            for style_index, style in enumerate(PROMPT_STYLES):
                for lexical_index, lexical in enumerate(FOOTPRINT_LEXICALIZATIONS):
                    wanted = (group_index + style_index + lexical_index) % 2
                    matches = [
                        row for row in rows
                        if row["case_kind"] == "causal"
                        and row["generalization_group"] == group
                        and row["prompt_style"] == style
                        and row["footprint_lexicalization"] == lexical
                        and int(row["semantic_replicate"]) == wanted
                    ]
                    require(len(matches) == 1, f"{mechanism}: identification causal selection invalid")
                    causal.extend(matches)
        specificity = []
        for membership_index, membership in enumerate(SOURCE_MEMBERSHIPS):
            for style_index, style in enumerate(PROMPT_STYLES):
                omitted = (membership_index + style_index) % 3
                specificity.extend(
                    row for row in rows
                    if row["case_kind"] == "specificity"
                    and row["source_membership"] == membership
                    and row["prompt_style"] == style
                    and SPECIFICITY_SUBTYPES.index(row["specificity_subtype"]) != omitted
                )
        require(len(causal) == 12 and len(specificity) == 12, f"{mechanism}: identification 12/12 split invalid")
        for row in causal + specificity:
            item = {
                "protocol_version": PROTOCOL_VERSION,
                "identification_case_index": len(output),
                "mechanism": mechanism,
                "case_id": row["case_id"],
                "case_kind": row["case_kind"],
                "generalization_group": row["generalization_group"],
                "source_membership": row["source_membership"],
                "prompt_style": row["prompt_style"],
                "footprint_lexicalization": row["footprint_lexicalization"],
                "specificity_subtype": row["specificity_subtype"],
                "seed": row["seed"],
                "included_streams": "matched_control,V4",
                "additional_streams": "generic_paraphrase,bystander_token",
            }
            require(tuple(item) == IDENTIFICATION_FIELDS, "identification field order changed")
            output.append(item)
    validate_identification_subset(output)
    return output


def validate_identification_subset(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 48, "identification semantic subset must contain 48 rows")
    require(len({(row["mechanism"], row["case_id"]) for row in rows}) == 48, "identification cases repeat")
    for mechanism in ("water_impact", "brittle_fracture"):
        selected = [row for row in rows if row["mechanism"] == mechanism]
        causal = [row for row in selected if row["case_kind"] == "causal"]
        specificity = [row for row in selected if row["case_kind"] == "specificity"]
        require(len(causal) == 12 and len(specificity) == 12, f"{mechanism}: identification split invalid")
        require(Counter(row["generalization_group"] for row in causal) == {group: 4 for group in GENERALIZATION_GROUPS}, f"{mechanism}: identification causal groups invalid")
        require(Counter(row["prompt_style"] for row in specificity) == {style: 6 for style in PROMPT_STYLES}, f"{mechanism}: identification specificity wording invalid")
        require(Counter(row["source_membership"] for row in specificity) == {membership: 4 for membership in SOURCE_MEMBERSHIPS}, f"{mechanism}: identification specificity membership invalid")
        require(Counter(row["specificity_subtype"] for row in specificity) == {subtype: 4 for subtype in SPECIFICITY_SUBTYPES}, f"{mechanism}: identification specificity subtype invalid")


def build_run_matrix(preserve_count: int) -> list[dict[str, Any]]:
    require(preserve_count == 36, "shared preservation manifest must contain 36 rows")
    specs: list[tuple[str, str, str]] = []
    specs.extend((mechanism, "matched_control", "main") for mechanism in MECHANISM_ORDER)
    specs.extend((mechanism, "V4", "main") for mechanism in MECHANISM_ORDER)
    for mechanism in ("water_impact", "brittle_fracture"):
        specs.extend((mechanism, arm, "identification") for arm in ("generic_paraphrase", "bystander_token"))
    rows: list[dict[str, Any]] = []
    for index, (mechanism, arm, group) in enumerate(specs):
        mechanism_index = MECHANISM_ORDER.index(mechanism)
        run_id = f"train7m_{mechanism}_{arm.lower()}"
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "run_index": index,
            "run_id": run_id,
            "mechanism_index": mechanism_index,
            "mechanism": mechanism,
            "arm": arm,
            "run_group": group,
            "training_seed": TRAINING_SEED,
            "updates": 200,
            "erase_updates": 100,
            "preserve_updates": 100,
            "selected_erase_rows_required": 178,
            "preserve_rows": 36,
            "lora_rank": 16,
            "lora_alpha": 16,
            "learning_rate": "5e-5",
            "target_teacher_weight": 4,
            "preservation_weight": 4,
            "checkpoint": 200,
            "inference_scale": 1.25,
            "authorization_state": "requires_selected178_and_frozen_caches",
            "output_dir": f"outputs/{DATASET_VERSION}/training/{run_id}",
        }
        require(tuple(row) == RUN_FIELDS, "run-matrix field order changed")
        rows.append(row)
    require(len(rows) == 18 and len({row["run_id"] for row in rows}) == 18, "18-run matrix invalid")
    require(Counter(row["arm"] for row in rows) == {"matched_control": 7, "V4": 7, "generic_paraphrase": 2, "bystander_token": 2}, "run arms invalid")
    return rows


def _write_exclusive(paths_to_bytes: Mapping[Path, bytes]) -> None:
    require(len(paths_to_bytes) == len(set(paths_to_bytes)), "output paths repeat")
    collisions = [path for path in paths_to_bytes if path.exists() or path.is_symlink()]
    require(not collisions, "refusing to overwrite existing output: " + ", ".join(map(str, collisions)))
    for parent in {path.parent for path in paths_to_bytes}:
        parent.mkdir(parents=True, exist_ok=True)
        require(parent.is_dir() and not parent.is_symlink(), f"output parent is invalid: {parent}")
    created: list[Path] = []
    try:
        for path, raw in paths_to_bytes.items():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o644)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "water_source_bank": args.water_source_bank,
        "water_train_pairs": args.water_train_pairs,
        "water_screen": args.water_screen,
        "preserve_manifest": args.preserve_manifest,
    }
    for logical_name, path in input_paths.items():
        _assert_frozen_input(path, logical_name)
    preserve_rows = read_csv(args.preserve_manifest)
    require(len(preserve_rows) == 36, "preservation manifest count changed")
    salt, salt_sha256 = _load_eval_salt(args.eval_seed_salt)
    ontology = build_ontologies(args.water_source_bank)
    targets, prompt_payloads = build_target_candidates(
        ontology,
        water_train_pairs_path=args.water_train_pairs,
        water_screen_path=args.water_screen,
        prompt_dir=args.prompt_dir,
    )
    formal = build_formal_cases(ontology, salt, targets)
    identification = build_identification_subset(formal)
    runs = build_run_matrix(len(preserve_rows))

    data_paths = {
        "ontology_registry": args.output_dir / "ontology_registry.json",
        "target_candidates": args.output_dir / "target_candidates.csv",
        "formal_cases": args.output_dir / "formal_cases.csv",
        "identification_subset": args.output_dir / "identification_subset.csv",
        "run_matrix": args.output_dir / "run_matrix.csv",
    }
    payloads: dict[Path, bytes] = {
        data_paths["ontology_registry"]: canonical_json_bytes(ontology),
        data_paths["target_candidates"]: csv_bytes(targets, TARGET_FIELDS),
        data_paths["formal_cases"]: csv_bytes(formal, FORMAL_FIELDS),
        data_paths["identification_subset"]: csv_bytes(identification, IDENTIFICATION_FIELDS),
        data_paths["run_matrix"]: csv_bytes(runs, RUN_FIELDS),
        **prompt_payloads,
    }
    artifact_sha256 = {
        (Path("data") / DATASET_VERSION / path.name).as_posix(): sha256_bytes(payloads[path])
        for path in data_paths.values()
    }
    artifact_sha256.update(
        {
            (Path("prompts") / DATASET_VERSION / path.name).as_posix(): sha256_bytes(raw)
            for path, raw in prompt_payloads.items()
        }
    )
    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_version": DATASET_VERSION,
        "status": "static_build_complete_pre_media",
        "eval_seed_salt_sha256": salt_sha256,
        "raw_eval_seed_salt_emitted": False,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "water_reuse": {
            "target_origin": "water_v1_reuse",
            "candidate_count": 192,
            "historical_accept_count": 178,
            "historical_reject_count": 14,
            "generation_manifest_sha256": WATER_TARGET_GENERATION_MANIFEST_SHA256,
            "generation_configuration": {
                "baseline": "negative_prompt",
                "num_inference_steps": 25,
                "guidance_scale": 5.0,
                "num_frames": 49,
                "fps": 8,
                "height": 480,
                "width": 832,
                "dtype": "bf16",
            },
        },
        "new_target_generation_configuration": {
            "mechanisms": list(MECHANISM_ORDER[1:]),
            "candidate_count": 1_152,
            "baseline": "negative_prompt",
            "num_inference_steps": 25,
            "guidance_scale": 5.0,
            "num_frames": 49,
            "fps": 8,
            "height": 480,
            "width": 832,
            "dtype": "bf16",
        },
        "counts": {
            "mechanisms": 7,
            "target_candidates": 1_344,
            "target_candidates_per_mechanism": 192,
            "new_target_prompt_shards": 6,
            "formal_cases": 294,
            "causal_cases": 168,
            "specificity_cases": 126,
            "identification_semantic_cases": 48,
            "identification_additional_outputs": 96,
            "training_runs": 18,
            "preserve_rows": 36,
        },
        "selection_contract": {
            "eligible_atomic_fields": [
                "receiver_present_and_recognizable",
                "source_absent",
                "trigger_absent",
                "footprint_absent",
                "video_quality_at_least_partial",
                "natural_motion_at_least_partial",
            ],
            "selected_rows_per_mechanism": 178,
            "if_eligible_below_178": "invalidate_mechanism_data_version",
            "if_eligible_above_178": "select ascending SHA256 of target-select-v1 pipe candidate_id",
            "scientific_seed_retry_allowed": False,
        },
        "artifact_sha256": artifact_sha256,
    }
    registry_path = args.output_dir / "build_registry.json"
    payloads[registry_path] = canonical_json_bytes(registry)
    _write_exclusive(payloads)
    return registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-seed-salt", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "data" / DATASET_VERSION,
    )
    parser.add_argument(
        "--prompt-dir", type=Path,
        default=project_root / "prompts" / DATASET_VERSION,
    )
    parser.add_argument(
        "--water-source-bank", type=Path,
        default=project_root / "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json",
    )
    parser.add_argument(
        "--water-train-pairs", type=Path,
        default=project_root / "data/water_impact_dynamic_v1/train_pairs.csv",
    )
    parser.add_argument(
        "--water-screen", type=Path,
        default=project_root / "data/water_impact_dynamic_v1/train_targets_v1_screen_final.csv",
    )
    parser.add_argument(
        "--preserve-manifest", type=Path,
        default=project_root / "data/protocol_v1/preserve_manifest.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = build_all(args)
    print(
        f"Built {registry['counts']['target_candidates']} target candidates, "
        f"{registry['counts']['formal_cases']} formal cases, and "
        f"{registry['counts']['training_runs']} training runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
