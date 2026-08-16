#!/usr/bin/env python3
"""Fail-closed protocol helpers for the v4 causal and specificity evaluation.

This module intentionally contains no private source ontology or holdout data.
It defines the byte-level interfaces through which an isolated evaluator may
open Stage-0/Stage-1 commitments after the v4 checkpoint and review artifacts
are frozen.  It never reads or resolves sealed-final36 paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL = "water_impact_dynamic_v4_eval_protocol_v2"
COMMITMENT_PROTOCOL = "water_impact_dynamic_v4_eval_commitment_registry_v2"
OPENING_PROTOCOL = "water_impact_dynamic_v4_eval_commitment_opening_v2"
GATE_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_machine_gate_registry_v2"
TRAINING_CODE_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_training_code_registry_v2"
RUNTIME_REGISTRY_PROTOCOL = "water_impact_dynamic_v4_runtime_registry_v2"
TRAINING_AUTHORIZATION_PROTOCOL = "water_impact_dynamic_v4_training_authorization_v2"
CHECKPOINT_ELIGIBILITY_PROTOCOL = "water_impact_dynamic_v4_checkpoint_eligibility_v2"
GENERATION_MANIFEST_PROTOCOL = "water_impact_dynamic_v4_generation_manifest_v2"
BLIND_REVIEW_PROTOCOL = "water_impact_dynamic_v4_blind_review_v2"
FINAL_REVIEW_PACKAGE_COMMITMENT_PROTOCOL = (
    "water_impact_dynamic_v4_final_review_package_commitment_v2"
)
FINAL_REVIEW_FREEZE_PROTOCOL = "water_impact_dynamic_v4_final_review_freeze_v2"
SELECTION_BINDING_PROTOCOL = "water_impact_dynamic_v4_selection_binding_v2"
GENERATION_SPEC_PROTOCOL = "water_impact_dynamic_v4_generation_spec_v2"
FORBIDDEN_SEED_INVENTORY_PROTOCOL = (
    "water_impact_dynamic_v4_forbidden_seed_inventory_v2"
)

DATA_ROOT = "data/water_impact_dynamic_v4"
DATASET_VERSION = "v4_dev72_v2"
CAUSAL_STAGE0 = f"{DATA_ROOT}/causal_stage0_commitment_v2.json"
CAUSAL_STAGE1 = f"{DATA_ROOT}/causal_stage1_commitment_v2.json"
SPECIFICITY_STAGE0 = f"{DATA_ROOT}/specificity_stage0_commitment_v2.json"
SPECIFICITY_STAGE1 = f"{DATA_ROOT}/specificity_stage1_commitment_v2.json"
PENDING_STAGE0_COMMITMENTS = {
    "causal": f"{DATA_ROOT}/causal_stage0_public_commitment_v2.json",
    "specificity": f"{DATA_ROOT}/specificity_stage0_public_commitment_v2.json",
}
PUBLIC_SOURCE_BANK = f"{DATA_ROOT}/source_bank_public64_registry_v2.json"
PUBLIC_HOLDOUT_COMMITMENT = f"{DATA_ROOT}/holdout_public_commitment_v2.json"
FROZEN_PUBLIC_SOURCE_BANK_SHA256 = (
    "473af632f8100e9e7c46c35e5fd679c9729bc80d19af12aaa78a1a0c69c9f814"
)
FROZEN_PUBLIC_HOLDOUT_COMMITMENT_SHA256 = (
    "6751a4d3b66491328909853b99bc8e6d06468a30b71f5bb746c7a744692fe84d"
)
FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256 = (
    "0d7fab1befdc197a7ae7f864a84c1f1ac3d029d5d72f9a513303892e48ec2477"
)
CAUSAL_PRIVATE_COMPONENT_FILENAMES = (
    "causal_stage0_candidates_private_v2.json",
    "causal_stage0_templates_private_v2.json",
    "causal_stage0_field_rules_private_v2.json",
    "causal_stage0_render_config_private_v2.json",
    "causal_stage0_selection_rules_private_v2.json",
    "causal_stage0_secrets_private_v2.json",
)
CAUSAL_ONTOLOGY_FILENAMES = (
    "source_ontology_private80_v2.json",
    "source_split_private_v2.json",
    "holdout_registry_private24_v2.json",
    "receiver_ontology_private32_v2.json",
)
SPECIFICITY_PRIVATE_COMPONENT_FILENAMES = (
    "specificity_stage0_candidates_private_v2.json",
    "specificity_stage0_new_bank_assignment_private_v2.json",
    "specificity_stage0_templates_private_v2.json",
    "specificity_stage0_field_rules_private_v2.json",
    "specificity_stage0_render_config_private_v2.json",
    "specificity_stage0_selection_rules_private_v2.json",
    "specificity_stage0_secrets_private_v2.json",
)
GATE_REGISTRY = f"{DATA_ROOT}/v4_machine_gate_registry_v2.json"
TRAINING_CODE_REGISTRY = f"{DATA_ROOT}/v4_training_code_registry_v2.json"
RUNTIME_REGISTRY = f"{DATA_ROOT}/v4_runtime_registry_v2.json"
TRAINING_AUTHORIZATION = f"{DATA_ROOT}/v4_training_authorization_v2.json"
V4_OUTPUT_DIR = "outputs/water_impact_dynamic_v4/adapter_source_slot_randomized_v2"
CHECKPOINT_ELIGIBILITY = f"{V4_OUTPUT_DIR}/checkpoint_eligibility_v2.json"
V4_CHECKPOINT = f"{V4_OUTPUT_DIR}/checkpoint-000200"
RUN_REGISTRATION = f"{V4_OUTPUT_DIR}/run_registration_v2.json"
SCALE_SANITY = f"{V4_OUTPUT_DIR}/target_prompt_scale_sanity_v2.json"
TRAINING_STATE = f"{V4_CHECKPOINT}/training_state_v2.json"
NULL_SIDECAR_PREFLIGHT = (
    "outputs/water_impact_dynamic_v4/null_sidecar_preflight_v2.json"
)
PROMPT_SIDECAR_DIR = "outputs/water_impact_dynamic_v4/source_slot_prompt_cache_v2"
PROMPT_SIDECAR_MANIFEST = f"{PROMPT_SIDECAR_DIR}/cache_manifest_v2.json"
PROMPT_SIDECAR_PROTOCOL = "water_impact_dynamic_v4_source_prompt_sidecar_v2"
EXPECTED_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
EXPECTED_SIDECAR_RUNTIME_VERSIONS = {
    "torch": "2.6.0",
    "diffusers": "0.33.1",
    "transformers": "4.51.3",
    "peft": "0.15.2",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
}
FROZEN_MODEL_CONTENT_INVENTORY_SHA256 = (
    "0a8566eeab29dfbc04303167ce1904b65b964dd1579959645d1f93e19ba15ddf"
)
FROZEN_TRAIN_MANIFEST_SHA256 = (
    "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
)
FROZEN_BASE_CACHE_INVENTORY_SHA256 = (
    "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
)
FROZEN_TEACHER_CACHE_INVENTORY_SHA256 = (
    "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
)
FROZEN_TRANSFORMER_INVENTORY_SHA256 = (
    "fe0c20ba99318c4b6d28d153b839888bbcf8a7a856796979566c305788dc18ac"
)
FROZEN_SAMPLE_ORDER_SHA256 = (
    "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb"
)
FROZEN_NOISE_SIGMA_RNG_INITIAL_SHA256 = (
    "49b65850c0793680efb3a7cfc023601e240f13acb78ddb3aa483794c68136704"
)
FROZEN_NOISE_SIGMA_RNG_FINAL_SHA256 = (
    "79ff6c9a3db46b02896073cc95e8d05d185e813c844475e14b1ae460dd61b33f"
)
FROZEN_INITIAL_LORA_SHA256 = (
    "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8"
)

RUNTIME_REGISTRY_PAYLOAD: dict[str, Any] = {
    "protocol": RUNTIME_REGISTRY_PROTOCOL,
    "status": "frozen",
    "dataset_version": DATASET_VERSION,
    "runtime_root": "models/.wan-runtime",
    "python_executable": "models/.wan-runtime/bin/python",
    "sys_prefix_policy": "realpath(sys.prefix)==realpath(runtime_root)",
    "python": {"implementation": "CPython", "version": "3.11.15"},
    "torch": {
        "distribution_version": "2.6.0",
        "module_version": "2.6.0+cu124",
    },
    "cuda": {
        "available_required": True,
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
    },
    "packages": {
        "accelerate": "1.14.0",
        "diffusers": "0.33.1",
        "huggingface-hub": "0.36.2",
        "numpy": "2.4.6",
        "peft": "0.15.2",
        "protobuf": "7.35.1",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.2",
        "tokenizers": "0.21.4",
        "torch": "2.6.0",
        "transformers": "4.51.3",
    },
}

DATASETS = ("causal", "specificity")
METHODS = ("original", "v3b", "v4")
CANDIDATE_METHODS = ("v3b", "v4")
ARM_CODES = ("O", "A", "B")
CAUSAL_GROUPS = (
    "holdout_source_seen_receiver",
    "seen_source_new_receiver",
    "holdout_source_new_receiver",
)
HOLDOUT_GROUPS = (
    "holdout_source_seen_receiver",
    "holdout_source_new_receiver",
)
PROMPT_VARIANTS = ("direct", "natural")
SPECIFICITY_MEMBERSHIPS = ("original_source", "new_bank_source", "holdout_source")
REPLICATES = {"causal": 3, "specificity": 2}
CASE_COUNTS = {"causal": 24, "specificity": 18}
UNIT_COUNTS = {"causal": 72, "specificity": 36}
CANDIDATE_COUNTS = {"causal": 48, "specificity": 36}
SEED_DOMAINS = {
    "causal": "causal-eval-seed-v2",
    # Executable only after a future specificity Stage-0 independently binds
    # this exact domain and formula; its absence remains fail-closed.
    "specificity": "specificity-eval-seed-v2",
}
SOURCE_SLOT_REGISTRY_SCHEMA = "water_impact_dynamic_v4_source_slot_registry_v2"
V2_SUPERSEDES = {
    "dataset_version": "v4_dev72_v1",
    "status": "preflight_dataset_invalid",
    "reason_code": "physical_source_ontology_invalid",
    "aggregate_audit": {
        "public_new_bank_pass": 34,
        "public_new_bank_fail": 22,
        "private_holdout_pass": 13,
        "private_holdout_fail": 11,
        "stage0_affected_candidates": 16,
        "stage0_distinct_valid_holdout_heads": 13,
        "stage0_required_distinct_holdout_heads": 16,
        "stage0_global_constraint_feasible": False,
    },
    "prior_public_sha256": {
        "source_bank_public64_registry_v1.json": (
            "1bb725f66ec2303b32b7a7681c5afe6013c333d908477d1713bf898855ec7177"
        ),
        "holdout_public_commitment_v1.json": (
            "3f56fee55ab0b93cd26a3478d73d84f5560e306eb2e9510cd7b47de574a98ce1"
        ),
        "causal_stage0_public_commitment_v1.json": (
            "994de60571606d46a016d2f15a41b024c68860940f35e7c002845077affab9c0"
        ),
    },
}
CURATION_AUDIT: dict[str, Any] = {
    "legacy_original_source_policy": {
        "certificate_policy": (
            "do not fabricate v2 mass, density, dimensions, or negative-buoyancy "
            "certificates for historical sources"
        ),
        "count": 8,
        "eligibility_gate": (
            "full 49-frame Original screening requires source_visibility=2, "
            "footprint_visibility>=1, receiver>=1, quality>=1, and causal_link=2"
        ),
        "specificity_gate": (
            "the matched Original hard-negative must independently satisfy the frozen "
            "specificity eligibility rule"
        ),
        "stage0_scope": (
            "seen_source_new_receiver only; never count legacy original sources as "
            "heldout generalization"
        ),
        "status": "legacy_original_source_exempt",
    },
    "pre_freeze_rejected_drafts": [
        {
            "aggregate_audit": {
                "curated_new80_fail": 2,
                "curated_new80_pass": 78,
                "maximum_solid_fill_ratio": 1.0728,
                "new_bank56_fail": 2,
                "new_bank56_pass": 54,
                "private_holdout24_fail": 0,
                "private_holdout24_pass": 24,
            },
            "reason_code": "solid_mass_exceeded_density_times_bounding_volume",
            "rejected_public_sha256": {
                "causal_stage0_public_commitment_v2.json": (
                    "8792133f709a1736c30fdc8172687837f24bf6f1f2616109b56474a7136c1a66"
                ),
                "holdout_public_commitment_v2.json": (
                    "24b8a7a4fb587c1ecc66b239024a716c47ed77346b0a703a2aef5c1cad3eff5a"
                ),
                "source_bank_public64_registry_v2.json": (
                    "cea52e0b4948462825175655e4c893820205d10c51d12ef2943cee7593fa3952"
                ),
            },
            "status": "pre_freeze_draft_rejected",
        }
    ],
    "status": "strict_new80_pass_after_pre_freeze_revision",
    "strict_physical_result": {
        "curated_new80_fail": 0,
        "curated_new80_pass": 80,
        "mass_density_bounding_volume_inconsistency_count": 0,
        "new_bank56_fail": 0,
        "new_bank56_pass": 56,
        "private_holdout24_fail": 0,
        "private_holdout24_pass": 24,
    },
    "strict_physical_scope": {
        "curated_new_source_count": 80,
        "new_bank_source_count": 56,
        "private_holdout_source_count": 24,
    },
}
FIELD_NORMALIZATION_RULES = {
    "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
    "protocol": SOURCE_SLOT_REGISTRY_SCHEMA,
    "dataset_version": DATASET_VERSION,
    "unicode": "NFKC",
    "case": "Unicode casefold",
    "punctuation": "replace every non-ASCII alphanumeric run with one space",
    "whitespace": "strip and collapse to one ASCII space",
    "head_rule": "curator-assigned single singular final lexical token",
    "canonical_record": "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF",
}
RANK_FORMULAS = {
    "causal": (
        "sha256(utf8('causal-selector-v2') || 0x00 || "
        "utf8(private_selector_salt) || 0x00 || canonical_candidate_record_bytes)"
    ),
    "specificity": (
        "sha256(utf8('specificity-selector-v2') || 0x00 || "
        "utf8(private_selector_salt) || 0x00 || canonical_candidate_record_bytes)"
    ),
}
SEED_FORMULAS = {
    "causal": (
        "uint32(first_4_bytes(sha256(utf8('causal-eval-seed-v2') || 0x00 || "
        "utf8(private_evaluation_seed_salt) || 0x00 || utf8(pair_id) || 0x00 || "
        "utf8(decimal_replicate))), big_endian)"
    ),
    "specificity": (
        "uint32(first_4_bytes(sha256(utf8('specificity-eval-seed-v2') || 0x00 || "
        "utf8(private_evaluation_seed_salt) || 0x00 || utf8(pair_id) || 0x00 || "
        "utf8(decimal_replicate))), big_endian)"
    ),
}
CAUSAL_SELECTION_RULES = {
    "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
    "protocol": SOURCE_SLOT_REGISTRY_SCHEMA,
    "dataset_version": DATASET_VERSION,
    "qualification": {
        "source_visibility": 2,
        "footprint_visibility_min": 1,
        "receiver_min": 1,
        "quality_min": 1,
        "causal_link": 2,
    },
    "panel": (
        "two independent screening curators; third curator adjudicates every atomic disagreement"
    ),
    "cell_quota": (
        "exactly four qualified cases from each of the six group x prompt_variant cells"
    ),
    "global_constraints": [
        "16 distinct holdout head lemmas across the two holdout groups",
        "each of the eight original training sources exactly once in seen_source_new_receiver",
        "eight distinct historical receivers in holdout_source_seen_receiver",
        "16 distinct new receivers across both new-receiver groups",
        "all 24 selected receiver identities are unique",
    ],
    "ranking_domain": "causal-selector-v2",
    "ranking_formula": RANK_FORMULAS["causal"],
    "subset_algorithm": (
        "within each cell order qualified candidates by rank; enumerate quota-respecting "
        "combinations in lexicographic rank-tuple order with early constraint pruning; choose "
        "the first globally feasible 24-case subset; equal ranks or no feasible subset "
        "invalidate v4_dev72_v2; there is no reserve queue"
    ),
    "evaluation_seed_domain": "causal-eval-seed-v2",
    "evaluation_seed_formula": SEED_FORMULAS["causal"],
    "replicates": 3,
    "required_selected_cases": 24,
    "required_evaluation_units": 72,
}
CAUSAL_TEMPLATE_FILL_RULES = {
    "direct": {
        "source_phrase": "python_str_capitalize",
        "receiver_phrase": "identity",
    },
    "natural": {"source_phrase": "identity", "receiver_phrase": "identity"},
}
CAUSAL_TEMPLATE_NON_SUBSTITUTION_RULE = (
    "construct only from structured source_phrase, receiver_phrase, and prompt_variant "
    "fields; substring replacement is forbidden"
)
CAUSAL_CANONICAL_TEMPLATES = {
    "direct": (
        "A simple realistic close-up video in one continuous shot. {source_phrase} falls "
        "naturally from above, enters the center of the water in {receiver_phrase}, and makes "
        "contact. The contact produces a visible brief splash followed by circular ripples "
        "spreading across the water. A soft reflected highlight moves slowly from left to right "
        "across the water and receiver throughout the shot. The viewpoint, receiver, and "
        "background geometry remain stable."
    ),
    "natural": (
        "A simple realistic close-up video in one continuous shot. In a single natural motion, "
        "{source_phrase} drops into the center of the water in {receiver_phrase}. After it "
        "touches the water, a short splash rises and expanding ripples travel outward. A soft "
        "reflected highlight moves slowly from left to right across the water and receiver "
        "throughout the shot. The viewpoint, receiver, and background geometry remain stable."
    ),
}
CAUSAL_RENDER_CONFIGURATION = {
    "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
    "protocol": SOURCE_SLOT_REGISTRY_SCHEMA,
    "dataset_version": DATASET_VERSION,
    "arm": "Original_only",
    "model_family": "Wan 2.1 T2V 1.3B",
    "model_content_inventory_binding": (
        "pending exact already-frozen v3b path-plus-file-bytes inventory digest from an "
        "independently authorized binder"
    ),
    "steps": 25,
    "cfg": 5,
    "frames": 49,
    "width": 832,
    "height": 480,
    "fps": 8,
    "dtype": "bf16",
    "adapter": None,
    "screening_scope": "all 49 frames for every candidate",
}
SPECIFICITY_TEMPLATE_FILL_RULES = {
    "direct": {"source_phrase": "python_str_capitalize", "receiver_phrase": "identity"},
    "natural": {"source_phrase": "identity", "receiver_phrase": "identity"},
}
SPECIFICITY_TEMPLATE_NON_SUBSTITUTION_RULE = CAUSAL_TEMPLATE_NON_SUBSTITUTION_RULE
SPECIFICITY_CANONICAL_TEMPLATES = {
    "direct": (
        "A simple realistic close-up video in one continuous shot. {source_phrase} stays "
        "on a dry support beside the water in {receiver_phrase}, never contacts the water, "
        "and does not cause a splash or ripple. The viewpoint, receiver, support, and "
        "background geometry remain stable."
    ),
    "natural": (
        "A simple realistic close-up video in one continuous shot. Beside the water in "
        "{receiver_phrase}, {source_phrase} remains visible on a dry support without touching "
        "the water. No splash or ripple is caused. The viewpoint, receiver, support, and "
        "background geometry remain stable."
    ),
}
SPECIFICITY_RENDER_CONFIGURATION = dict(CAUSAL_RENDER_CONFIGURATION)
SPECIFICITY_SELECTION_RULES = {
    "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
    "qualification": {
        "protected_object_visibility": 2,
        "receiver_min": 1,
        "quality_min": 1,
        "noncausal_role_adherence": 2,
    },
    "panel": (
        "two independent screening curators; third curator adjudicates every atomic disagreement"
    ),
    "cell_quota": (
        "exactly three qualified cases from each of the six membership x prompt_variant cells"
    ),
    "global_constraints": [
        "all 18 selected source head lemmas are unique",
        "six original and six holdout cases exactly preserve selected causal source-receiver pairs",
        "six holdout cases cover both causal holdout groups and both prompt variants",
    ],
    "ranking_formula": RANK_FORMULAS["specificity"],
    "subset_algorithm": (
        "order all qualified candidates by rank; enumerate quota-respecting combinations in "
        "lexicographic rank-tuple order with early constraint pruning; choose the first globally "
        "feasible 18-case subset; equal ranks or no feasible subset invalidate the data version; "
        "there is no reserve queue"
    ),
    "evaluation_seed_formula": SEED_FORMULAS["specificity"],
    "replicates": 2,
    "required_selected_cases": 18,
    "required_evaluation_units": 36,
}

IMPACT_PLAUSIBILITY_FIELDS = frozenset(
    {
        "compact_and_rigid",
        "curator_note",
        "density_g_cm3",
        "dimensions_cm",
        "entity_state",
        "food_or_produce",
        "flexible_or_film_like",
        "fragile",
        "loose_aggregate",
        "mass_g",
        "material",
        "natural_drop_entry",
        "negative_buoyancy",
        "porous",
        "powder",
        "predominantly_buoyant_or_windborne",
        "size_class",
        "source_specific_feature",
        "verdict",
        "visually_recognizable",
        "visible_brief_splash_or_ripple_plausible",
    }
)
SOURCE_STRATA_FIELDS = frozenset(
    {"origin", "food_status", "shape_class", "color_family", "material_family", "texture_class"}
)
CURATION_STRATA = frozenset(
    {"machined_steel", "cast_iron", "brass_bronze", "dense_alloy"}
)
ORIGINAL_TRAINING_SOURCES = {
    "water_droplet": "one large clear water droplet",
    "ice_cube": "one small transparent ice cube",
    "red_apple": "one small red apple",
    "green_lime": "one small green lime",
    "blue_marble": "one blue glass marble",
    "wooden_cube": "one small light wooden cube",
    "steel_ball": "one polished steel ball bearing",
    "plastic_block": "one small red plastic toy block",
}
HISTORICAL_TRAINING_RECEIVERS = {
    "shallow_pond": "a calm shallow pond",
    "glass_bowl": "a transparent glass mixing bowl filled with water",
    "white_basin": "a wide white ceramic basin filled with water",
    "glass_tank": "a clear rectangular glass tank filled with water",
    "metal_bucket": "a clean metal bucket filled with water",
    "kitchen_sink": "a stainless-steel kitchen sink basin filled with water",
    "porcelain_bowl": "a plain porcelain soup bowl filled with water",
    "cooking_pot": "a black cooking pot filled with water",
    "garden_birdbath": "a round stone birdbath filled with water",
    "glass_dish": "a rectangular glass baking dish filled with water",
    "plastic_tub": "a blue plastic storage tub filled with water",
    "laboratory_beaker": "a laboratory beaker filled with clear water",
}
RECEIVER_TYPES = frozenset(
    {
        "agricultural_feature", "cave_feature", "civil_waterway", "coastal_feature",
        "coastal_wetland", "desert_feature", "desert_waterway", "drainage_feature",
        "drainage_structure", "excavation_feature", "floodplain_feature",
        "fortification_waterway", "geothermal_feature", "glacial_feature",
        "harbor_feature", "karst_feature", "landscape_feature", "landscape_waterway",
        "mill_waterway", "natural_stream", "river_crossing", "river_feature",
        "tidal_feature", "wetland_feature", "wetland_waterway",
    }
)

FRAME_COUNT = 49
WIDTH = 832
HEIGHT = 480
FPS = Fraction(8, 1)
GENERATION_SPEC: dict[str, Any] = {
    "pipeline": "WanPipeline",
    "model": "models/Wan2.1-T2V-1.3B-Diffusers",
    "baseline": "clean",
    "num_inference_steps": 25,
    "guidance_scale": 5.0,
    "num_frames": FRAME_COUNT,
    "width": WIDTH,
    "height": HEIGHT,
    "fps": 8,
    "dtype": "bfloat16",
    "device": "cuda",
    "enable_model_cpu_offload": False,
    "enable_sequential_cpu_offload": False,
    "vae_slicing": True,
    "vae_tiling": True,
    "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
    "activation_gate": None,
    "attention_gate": None,
    "persistent_activation_gate": False,
    "attention_suppression_strength": 20.0,
    "v3b_lora_scale": 1.25,
    "v4_lora_scale": 1.25,
    "checkpoint_step": 200,
}

V3B_CHECKPOINT = (
    "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1/"
    "checkpoint-000200"
)
V3B_CHECKPOINT_SHA256 = "f40f15f0a51c840db3e4fa8e2f931bdf89a4e5787f642513161e48d848fd723f"

CAUSAL_SCORE_FIELDS = (
    "target_visibility_0_absent_2_clear",
    "footprint_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
)
CAUSAL_REFERENCE_ONLY_FIELDS = ("causal_link_0_absent_2_clear",)
SPECIFICITY_SCORE_FIELDS = (
    "protected_object_visibility_0_absent_2_clear",
    "receiver_preservation_0_bad_2_good",
    "video_quality_0_bad_2_good",
    "noncausal_role_adherence_0_bad_2_good",
)

GATE_SPEC: dict[str, Any] = {
    "decision": "all_causal_specificity_and_role_selectivity_checks_must_pass",
    "registered_denominators": {"U": 72, "W": 36, "causal_cases": 24, "specificity_cases": 18},
    "unusable_treatment_contribution": 0,
    "usable_definition": "receiver>=1 and quality>=1",
    "strict_definition": "target=0, footprint=0, receiver=2, quality=2",
    "causal_conditions": [
        "all_provenance_valid",
        "abs(E)>=66 and abs(C)>=64",
        "abs(K)>=20 and each_group_variant_cell_has_at_least_3_K_cases",
        "Delta_T(C)>=18",
        "each_replicate_Delta_T>=3",
        "each_group_Delta_T>=3 and each_variant_Delta_T>=3",
        "each_group_variant_Delta_T>=0",
        "paired_usable_target_improvements_on_C>=18",
        "positive_G_T_cases_in_K>=10",
        "Delta_T(C_hold)>=12 and paired_holdout_improvements>=12",
        "positive_holdout_G_T_cases_in_K>=7",
        "clear_to_absent_on_C>=6",
        "holdout_clear_to_absent>=4_from_distinct_cases_covering_both_groups_and_variants",
        "holdout_cases_with_clear_to_absent_on_at_least_2_of_3_replicates>=2",
        "usable_absent_target_v4_on_C>=v3b_plus_6",
        "v4_usable_on_U>=68",
        "v4_receiver_points_on_U>=max(114,v3b_minus_6)",
        "v4_quality_points_on_U>=max(96,v3b_minus_6)",
        "footprint_Delta_C_and_C_hold_and_each_group_nonnegative_and_each_cell>=-1",
        "v4_strict_on_C>=6_with_holdout>=4_covering_all_groups_both_variants_and_4_cases",
        "paired_strict_gains>=4_with_holdout>=3_from_3_distinct_cases",
    ],
    "specificity_conditions": [
        "abs(H)>=33 and abs(D)>=32",
        "each_membership_H>=11_D>=10_and_each_cell_D>=5",
        "abs(K_D)>=15",
        "v3b_PV_D>=ceil(1.5D)_and_v4_PV_D>=max(v3b_minus_3,floor)",
        "PV_membership_variant_cell_baseline_floors_and_v4_margins",
        "NR_global_membership_variant_cell_baseline_floors_and_v4_margins",
        "v4_PV_and_NR_absolute_floors_on_H_and_all_partitions",
        "usable_absent_protected_object_global_and_membership_caps",
        "v4_usable_on_W>=33",
        "v4_receiver_points_on_W>=max(57,v3b_minus_3)",
        "v4_quality_points_on_W>=max(48,v3b_minus_3)",
    ],
    "matched_role_selectivity": {
        "mapping_pairs": 6,
        "complete_pairs_min": 5,
        "role_selective_pairs_min": 3,
        "coverage": "both_holdout_causal_groups_and_both_prompt_variants",
        "role_selective_with_clear_to_absent_min": 2,
    },
    "secondary_cluster_bootstrap": {
        "estimand": "mean_G_T_over_complete_cases_K",
        "iterations": 10000,
        "percentile_interval": [0.025, 0.975],
        "seed": 26016004,
        "gate_override": False,
    },
    "post_checkpoint_outcomes": [
        "invalid_run",
        "inconclusive_invalid_evaluation",
        "valid_negative_ablation",
        "eligible_for_separate_main_experiment_preregistration",
    ],
}

# Private artifact names are semantic interfaces, not paths.  Public registry
# files reveal only the digest, byte size, and row count for each name.
STAGE_ARTIFACTS: dict[tuple[str, int], tuple[str, ...]] = {
    ("causal", 0): (
        "candidate_manifest_48",
        "source_bank_registry_64",
        "source_ontology_80",
        "source_split_80",
        "holdout_registry_24",
        "receiver_ontology_32",
        "canonical_templates",
        "field_normalization",
        "raw_root_bundle",
        "raw_render_configuration",
        "stage0_secrets",
        "screening_seed",
        "screening_generation_spec",
        "selector_salt",
        "ranking_formula",
        "constrained_subset_algorithm",
        "evaluation_seed_salt",
        "seed_derivation_formula",
        "forbidden_seed_inventory",
    ),
    ("causal", 1): (
        "screening_generation_manifest",
        "screening_candidate_binding",
        "screening_review_a",
        "screening_review_b",
        "screening_dispute_template",
        "screening_adjudication",
        "screening_freeze_manifest",
        "eligibility_table_48",
        "selector_output_24",
        "selected_case_manifest_24",
        "unit_manifest_U_72",
    ),
    ("specificity", 0): (
        "candidate_manifest_36",
        "new_bank_selection_and_receiver_assignment",
        "canonical_templates",
        "field_normalization",
        "raw_root_bundle",
        "raw_render_configuration",
        "stage0_secrets",
        "screening_seed",
        "screening_generation_spec",
        "selector_salt",
        "ranking_formula",
        "constrained_subset_algorithm",
        "evaluation_seed_salt",
        "seed_derivation_formula",
        "forbidden_seed_inventory",
    ),
    ("specificity", 1): (
        "screening_generation_manifest",
        "screening_candidate_binding",
        "screening_review_a",
        "screening_review_b",
        "screening_dispute_template",
        "screening_adjudication",
        "screening_freeze_manifest",
        "eligibility_table_36",
        "selector_output_18",
        "selected_case_manifest_18",
        "unit_manifest_W_36",
        "holdout_mapping_M_6",
    ),
}

EXPECTED_COMMITMENT_ROW_COUNTS: dict[tuple[str, int, str], int] = {
    ("causal", 0, "candidate_manifest_48"): 48,
    ("causal", 0, "source_bank_registry_64"): 64,
    ("causal", 0, "source_ontology_80"): 80,
    ("causal", 0, "source_split_80"): 80,
    ("causal", 0, "holdout_registry_24"): 24,
    ("causal", 0, "receiver_ontology_32"): 32,
    ("causal", 1, "screening_review_a"): 48,
    ("causal", 1, "screening_review_b"): 48,
    ("causal", 1, "screening_candidate_binding"): 48,
    ("causal", 1, "eligibility_table_48"): 48,
    ("causal", 1, "selected_case_manifest_24"): 24,
    ("causal", 1, "unit_manifest_U_72"): 72,
    ("specificity", 0, "candidate_manifest_36"): 36,
    ("specificity", 0, "new_bank_selection_and_receiver_assignment"): 12,
    ("specificity", 1, "screening_review_a"): 36,
    ("specificity", 1, "screening_review_b"): 36,
    ("specificity", 1, "screening_candidate_binding"): 36,
    ("specificity", 1, "eligibility_table_36"): 36,
    ("specificity", 1, "selected_case_manifest_18"): 18,
    ("specificity", 1, "unit_manifest_W_36"): 36,
    ("specificity", 1, "holdout_mapping_M_6"): 6,
}

TRAINING_AUTHORIZATION_REFS = {
    "source_bank_registry": PUBLIC_SOURCE_BANK,
    "holdout_public_commitment": PUBLIC_HOLDOUT_COMMITMENT,
    "causal_stage0": CAUSAL_STAGE0,
    "causal_stage1": CAUSAL_STAGE1,
    "specificity_stage0": SPECIFICITY_STAGE0,
    "specificity_stage1": SPECIFICITY_STAGE1,
    "gate_registry": GATE_REGISTRY,
    "runtime_registry": RUNTIME_REGISTRY,
    "code_registry": TRAINING_CODE_REGISTRY,
}

TRAINING_CODE_ARTIFACTS = {
    "trainer": "scripts/train_wan_waterdrop_lora_v4.py",
    "launcher": "scripts/run_water_impact_dynamic_sft_v4_source_slot.sh",
    "source_mapping": "scripts/build_water_impact_dynamic_v4_source_mapping.py",
    "preparer": "scripts/prepare_water_impact_dynamic_v4_prompt_cache.py",
    "runtime_registry_builder": "scripts/build_water_impact_dynamic_v4_runtime_registry.py",
    "design_doc": "docs/water_impact_dynamic_v4_source_slot_randomization.md",
    "eval_protocol": "scripts/water_impact_dynamic_v4_eval_protocol.py",
    "eval_selector": "scripts/select_water_impact_dynamic_v4_eval.py",
    "eval_blind_builder": "scripts/build_water_impact_dynamic_v4_blind_review.py",
    "eval_scorer": "scripts/score_water_impact_dynamic_v4.py",
    "eval_runner": "scripts/run_water_impact_dynamic_v4_eval.py",
    "generator": "scripts/generate_wan_clean.py",
}

PUBLIC_REVIEW_BASE_FIELDS = frozenset(
    {
        "review_id",
        "anonymous_unit",
        "arm_code",
        "object_phrase",
        "receiver_description",
        "composite_path",
        "video_path",
        "notes",
    }
)

HEX64 = frozenset("0123456789abcdef")
FORBIDDEN_PUBLIC_FIELDS = {
    "method",
    "pair_id",
    "semantic_case_id",
    "specificity_case_id",
    "group",
    "membership",
    "prompt_variant",
    "replicate",
    "seed",
    "source_path",
    "source_video_path",
    "checkpoint",
    "sha256",
    "content_sha256",
    "provenance",
    "answer_key",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError("fieldnames are required when writing an empty CSV")
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(f"missing non-symlink artifact: {path}")
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_artifact_inventory(project_root: Path, model_path: str | Path) -> dict[str, Any]:
    """Hash every meaningful model byte using the v3c-frozen inventory algorithm."""

    root = resolve_path(project_root, model_path)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"missing frozen model directory: {root}")
    excluded_suffixes = (".tmp", ".lock", ".incomplete", "~")
    descendants = list(root.rglob("*"))
    symlinks = [path for path in descendants if path.is_symlink()]
    if symlinks:
        raise ValueError(f"model artifact inventory forbids symlinks: {symlinks[0]}")
    files = sorted(
        path
        for path in descendants
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and not path.name.endswith(excluded_suffixes)
    )
    if not files:
        raise ValueError("model artifact inventory is empty")
    required = {
        "model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
    }
    relative_names = {path.relative_to(root).as_posix() for path in files}
    if not required <= relative_names:
        raise FileNotFoundError(
            f"model artifact inventory is missing: {sorted(required - relative_names)}"
        )
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                file_digest.update(chunk)
        digest.update(b"\n")
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_digest.hexdigest()}
        )
    return {
        "algorithm": "sha256_ordered_relative_path_nul_bytes_newline_with_file_records_v1",
        "root": str(model_path),
        "excluded": ["any .cache directory", "*.tmp", "*.lock", "*.incomplete", "*~"],
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": records,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _require_sha256(value: Any, label: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return str(value)


def resolve_path(project_root: Path, registered: str | Path) -> Path:
    reject_sealed_final36_path(registered)
    path = Path(registered)
    return path if path.is_absolute() else project_root / path


def reject_sealed_final36_path(*values: str | Path) -> None:
    """Reject sealed-final36 lexically, before any filesystem access occurs."""

    for value in values:
        lowered = str(value).lower().replace("_", "-")
        if "final36" in lowered or "sealed-final" in lowered:
            raise ValueError("sealed-final36 access is forbidden during v4 development")


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(key) or _contains_placeholder(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    if value is None:
        # ``row_count: null`` is the registered representation for non-CSV
        # byte artifacts; required scientific fields are checked separately.
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        return not lowered or any(
            token in lowered
            for token in ("placeholder", "todo", "tbd", "fill_me", "to_be_frozen")
        )
    return False


def _load_exact_json(path: Path, label: str) -> dict[str, Any]:
    reject_sealed_final36_path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: missing non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON root must be an object")
    return payload


def validate_commitment_registry(
    path: Path,
    *,
    dataset: str,
    stage: int,
    expected_stage0_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a public Stage-0/Stage-1 digest registry without opening data."""

    if dataset not in DATASETS or stage not in (0, 1):
        raise ValueError("unsupported commitment dataset/stage")
    payload = _load_exact_json(path, f"{dataset} Stage-{stage} registry")
    expected_keys = {
        "protocol",
        "dataset",
        "dataset_version",
        "stage",
        "status",
        "sealed_final36_status",
        "artifacts",
    }
    if stage == 1:
        expected_keys.add("stage0_registry_sha256")
    if set(payload) != expected_keys:
        raise ValueError(f"{dataset} Stage-{stage}: registry fields are not exact")
    if (
        payload["protocol"] != COMMITMENT_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError(f"{dataset} Stage-{stage}: protocol/dataset/version mismatch")
    if payload["stage"] != stage or payload["status"] != "committed":
        raise ValueError(f"{dataset} Stage-{stage}: registry is not committed")
    if payload["sealed_final36_status"] != "unopened":
        raise ValueError("sealed-final36 must remain unopened")
    if stage == 1:
        _require_sha256(payload["stage0_registry_sha256"], "stage0_registry_sha256")
        if expected_stage0_sha256 is None or payload["stage0_registry_sha256"] != expected_stage0_sha256:
            raise ValueError(f"{dataset} Stage-1 does not bind the exact Stage-0 registry bytes")
    artifacts = payload["artifacts"]
    required = set(STAGE_ARTIFACTS[(dataset, stage)])
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError(f"{dataset} Stage-{stage}: artifact commitments are not exact")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes", "row_count"}:
            raise ValueError(f"{dataset} Stage-{stage}/{name}: commitment record is not exact")
        _require_sha256(record["sha256"], f"{dataset} Stage-{stage}/{name}")
        if not isinstance(record["size_bytes"], int) or record["size_bytes"] <= 0:
            raise ValueError(f"{dataset} Stage-{stage}/{name}: size_bytes must be positive")
        if record["row_count"] is not None and (
            not isinstance(record["row_count"], int) or record["row_count"] < 0
        ):
            raise ValueError(f"{dataset} Stage-{stage}/{name}: invalid row_count")
        expected_rows = EXPECTED_COMMITMENT_ROW_COUNTS.get((dataset, stage, name))
        if expected_rows is not None and record["row_count"] != expected_rows:
            raise ValueError(
                f"{dataset} Stage-{stage}/{name}: row_count must be exactly {expected_rows}"
            )
    if _contains_placeholder(payload):
        raise ValueError(f"{dataset} Stage-{stage}: placeholder content is forbidden")
    return payload


def validate_forbidden_seed_inventory(path: Path, *, dataset: str) -> set[int]:
    """Validate the independent auditor's exact, identity-free numeric inventory."""

    payload = _load_exact_json(path, "forbidden seed inventory")
    if set(payload) != {
        "protocol",
        "dataset",
        "status",
        "seed_encoding",
        "source_commitments",
        "seeds",
    }:
        raise ValueError("forbidden seed inventory fields are not exact")
    if (
        payload["protocol"] != FORBIDDEN_SEED_INVENTORY_PROTOCOL
        or payload["dataset"] != dataset
        or payload["status"] != "frozen_by_independent_seed_auditor"
        or payload["seed_encoding"] != "nonnegative JSON integer below 2^63"
    ):
        raise ValueError("forbidden seed inventory protocol is not frozen")
    sources = payload["source_commitments"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("forbidden seed inventory has no source commitments")
    names: list[str] = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"name", "sha256", "seed_count"}:
            raise ValueError("forbidden seed source commitment is not exact")
        if not isinstance(source["name"], str) or not source["name"].strip():
            raise ValueError("forbidden seed source name is blank")
        _require_sha256(source["sha256"], "forbidden seed source")
        if not isinstance(source["seed_count"], int) or source["seed_count"] < 0:
            raise ValueError("forbidden seed source count is invalid")
        names.append(source["name"])
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("forbidden seed sources must be unique and sorted by name")
    values = payload["seeds"]
    if (
        not isinstance(values, list)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= 1 << 63
            for value in values
        )
        or values != sorted(values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("forbidden numeric seeds must be unique sorted signed-63 values")
    if sum(source["seed_count"] for source in sources) < len(values):
        raise ValueError("forbidden seed inventory exceeds its audited source counts")
    return set(values)


def _private_opening_path(path: Path, private_root: Path, label: str) -> None:
    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("selection contract private root must be a real directory")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: missing evaluator-only artifact")
    try:
        path.resolve(strict=True).relative_to(private_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label}: artifact escapes evaluator-only private root") from exc


def _secret_commitment(name: str, secret: str) -> str:
    record = json.dumps(
        {"name": name, "secret": secret},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256((record + "\n").encode("utf-8")).hexdigest()


def expected_selection_binding(
    *,
    dataset: str,
    public_pending_sha256: str,
    root_bundle_sha256: str,
    component_sha256: Mapping[str, str],
    generation_spec_sha256: str,
    model_inventory_sha256: str,
    runtime_registry_sha256: str,
    screening_seed_sha256: str,
    selector_salt_sha256: str,
    evaluation_seed_salt_sha256: str,
    forbidden_seed_inventory_sha256: str,
    preselection_seed_audit_sha256: str,
    preselection_seed_count: int,
) -> dict[str, Any]:
    """Build the exact pre-screening bridge for the curator's `pair_id` symbol."""

    if dataset not in DATASETS:
        raise ValueError("selection binding dataset is unsupported")
    component_filenames = (
        CAUSAL_PRIVATE_COMPONENT_FILENAMES
        if dataset == "causal"
        else SPECIFICITY_PRIVATE_COMPONENT_FILENAMES
    )
    if set(component_sha256) != set(component_filenames):
        raise ValueError("selection binding component inventory is not exact")
    for label, value in (
        ("public pending commitment", public_pending_sha256),
        ("root bundle", root_bundle_sha256),
        ("generation spec", generation_spec_sha256),
        ("model inventory", model_inventory_sha256),
        ("runtime registry", runtime_registry_sha256),
        ("screening seed", screening_seed_sha256),
        ("selector salt", selector_salt_sha256),
        ("evaluation seed salt", evaluation_seed_salt_sha256),
        ("forbidden seed inventory", forbidden_seed_inventory_sha256),
        ("preselection seed audit", preselection_seed_audit_sha256),
        *((name, value) for name, value in component_sha256.items()),
    ):
        _require_sha256(value, label)
    if preselection_seed_count != CANDIDATE_COUNTS[dataset] * REPLICATES[dataset]:
        raise ValueError("preselection seed audit count is not exact")
    if dataset == "causal":
        candidate_projection = {
            "source": "candidate_manifest.candidates in frozen list order",
            "candidate_id": "case_id",
            "semantic_case_id": "case_id",
            "receiver": "receiver_phrase",
            "prompt": "canonical_prompt",
            "copy_rule": "copy all frozen candidate fields byte-for-byte before adding aliases",
        }
        cell_order = [
            {"group": group, "prompt_variant": variant, "quota": 4}
            for group in CAUSAL_GROUPS
            for variant in PROMPT_VARIANTS
        ]
        selected_id_field = "semantic_case_id"
        namespace_prefix = "causal"
        render_filename = "causal_stage0_render_config_private_v2.json"
        selection_rules = CAUSAL_SELECTION_RULES
    else:
        candidate_projection = {
            "source": "candidate_manifest.candidates in frozen list order",
            "candidate_id": "case_id",
            "specificity_case_id": "case_id",
            "receiver": "receiver_phrase",
            "prompt": "canonical_prompt",
            "copy_rule": "copy all frozen candidate fields byte-for-byte before adding aliases",
        }
        cell_order = [
            {"membership": membership, "prompt_variant": variant, "quota": 3}
            for membership in SPECIFICITY_MEMBERSHIPS
            for variant in PROMPT_VARIANTS
        ]
        selected_id_field = "specificity_case_id"
        namespace_prefix = "specificity"
        render_filename = "specificity_stage0_render_config_private_v2.json"
        selection_rules = SPECIFICITY_SELECTION_RULES
    return {
        "protocol": SELECTION_BINDING_PROTOCOL,
        "dataset": dataset,
        "dataset_version": DATASET_VERSION,
        "status": "frozen_before_original_screening",
        "public_pending_commitment": {
            "path": PENDING_STAGE0_COMMITMENTS[dataset],
            "sha256": public_pending_sha256,
        },
        "curator_root": {
            "bundle_sha256": root_bundle_sha256,
            "components": dict(component_sha256),
        },
        "downstream_artifacts": {
            "generation_spec_sha256": generation_spec_sha256,
            "model_inventory_sha256": model_inventory_sha256,
            "runtime_registry_sha256": runtime_registry_sha256,
            "screening_seed_sha256": screening_seed_sha256,
            "selector_salt_sha256": selector_salt_sha256,
            "evaluation_seed_salt_sha256": evaluation_seed_salt_sha256,
            "forbidden_seed_inventory_sha256": forbidden_seed_inventory_sha256,
        },
        "candidate_projection": candidate_projection,
        "ranking_contract": {
            "canonical_record_bytes": FIELD_NORMALIZATION_RULES["canonical_record"],
            "domain": f"{namespace_prefix}-selector-v2",
            "formula": RANK_FORMULAS[dataset],
            "tie_policy": "equal ranks invalidate the data version",
            "cell_order": cell_order,
            "rank_tuple": (
                f"sort the {CASE_COUNTS[dataset]} selected candidate ranks ascending; compare "
                f"the complete {CASE_COUNTS[dataset]}-element tuples lexicographically; choose "
                "the first globally feasible tuple"
            ),
            "global_selection_semantics": selection_rules["subset_algorithm"],
        },
        "seed_contract": {
            "domain": SEED_DOMAINS[dataset],
            "formula": SEED_FORMULAS[dataset],
            "evaluation_seed_namespace": f"v4-{namespace_prefix}-evaluation-v2",
            "screening_seed_namespace": f"v4-{namespace_prefix}-stage0-screening-v2",
            "screening_seed_commitment_name": f"{namespace_prefix}_screening_seed_v2",
            "selector_salt_commitment_name": f"{namespace_prefix}_stage0_selector_salt_v2",
            "evaluation_seed_salt_commitment_name": f"{namespace_prefix}_evaluation_seed_salt_v2",
            "salt_encoding": "UTF-8",
            "candidate_id_field": "case_id",
            "selected_id_field": selected_id_field,
            "formula_id_symbol": "pair_id",
            "id_binding": (
                f"candidate.case_id == selected.{selected_id_field} == formula.pair_id byte-for-byte"
            ),
            "replicate_encoding": "UTF-8 unsigned canonical decimal without leading zeros",
            "digest_projection": "first 4 bytes, big-endian unsigned uint32",
            "screening_separation": (
                "the screening seed must differ from every derived evaluation seed and must "
                "not occur in the external forbidden inventory"
            ),
            "collision_policy": (
                "any collision with the forbidden inventory, screening seed, or another "
                "evaluation seed invalidates the data version; no retries"
            ),
            "preselection_audit": {
                "candidate_count": CANDIDATE_COUNTS[dataset],
                "replicates_per_candidate": REPLICATES[dataset],
                "derived_seed_count": preselection_seed_count,
                "ordered_case_replicate_seed_sha256": preselection_seed_audit_sha256,
                "pairwise_unique": True,
                "disjoint_from_external_forbidden": True,
                "disjoint_from_screening_seed": True,
            },
        },
        "render_mapping": {
            "raw_render_component_sha256": component_sha256[render_filename],
            "generation_spec_sha256": generation_spec_sha256,
            "model_inventory_sha256": model_inventory_sha256,
            "runtime_registry_sha256": runtime_registry_sha256,
            "field_mapping": {
                "arm": "Original_only -> method original, clean base model, no adapter",
                "model_family": "Wan 2.1 T2V 1.3B -> WanPipeline and frozen model inventory",
                "steps": "num_inference_steps",
                "cfg": "guidance_scale",
                "frames": "num_frames",
                "width": "width",
                "height": "height",
                "fps": "fps",
                "dtype": "bf16 generator dtype / bfloat16 protocol dtype",
                "screening_scope": "full 49-frame decode and full-video reviewer media",
            },
        },
    }


def audit_preselection_seed_space(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    private_salt: str,
    screening_seed: int,
    forbidden_seeds: Iterable[int],
) -> dict[str, Any]:
    """Audit every candidate×replicate seed before any Original render."""

    if dataset not in DATASETS or len(candidate_rows) != CANDIDATE_COUNTS[dataset]:
        raise ValueError("preselection seed audit requires the exact candidate pool")
    forbidden = {int(value) for value in forbidden_seeds}
    occupied = set(forbidden)
    occupied.add(screening_seed)
    records: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for row in candidate_rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError("preselection seed audit candidate IDs are blank or duplicate")
        case_ids.add(case_id)
        for replicate in range(REPLICATES[dataset]):
            seed = derive_seed(
                private_salt, case_id, replicate, dataset=dataset
            )
            if seed in occupied:
                raise ValueError(
                    "preselection derived seed collides with forbidden, screening, or peer seed"
                )
            occupied.add(seed)
            records.append(
                {"case_id": case_id, "replicate": replicate, "seed": seed}
            )
    expected_count = CANDIDATE_COUNTS[dataset] * REPLICATES[dataset]
    if len(records) != expected_count:
        raise ValueError("preselection seed audit count differs from protocol")
    return {
        "derived_seed_count": len(records),
        "ordered_case_replicate_seed_sha256": canonical_json_sha256(records),
    }


def _normalize_registry_phrase(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("ontology phrase must be a nonempty NUL-free string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _validate_impact_plausibility(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != IMPACT_PLAUSIBILITY_FIELDS:
        raise ValueError(f"{label}: impact-plausibility fields are not exact")
    dimensions = value["dimensions_cm"]
    numeric_dimensions = (
        isinstance(dimensions, list)
        and len(dimensions) == 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            and 2.5 <= item <= 15.0
            for item in dimensions
        )
    )
    density = value["density_g_cm3"]
    mass = value["mass_g"]
    if (
        value["verdict"] != "pass"
        or value["compact_and_rigid"] is not True
        or value["natural_drop_entry"] is not True
        or value["visible_brief_splash_or_ripple_plausible"] is not True
        or value["negative_buoyancy"] is not True
        or value["visually_recognizable"] is not True
        or value["predominantly_buoyant_or_windborne"] is not False
        or value["flexible_or_film_like"] is not False
        or value["fragile"] is not False
        or value["powder"] is not False
        or value["loose_aggregate"] is not False
        or value["porous"] is not False
        or value["food_or_produce"] is not False
        or value["entity_state"] not in {"solid_one_piece", "rigid_locked_assembly"}
        or not isinstance(value["material"], str)
        or not value["material"].strip()
        or not isinstance(density, (int, float))
        or isinstance(density, bool)
        or not math.isfinite(density)
        or not 3.0 <= density <= 20.0
        or not isinstance(mass, int)
        or isinstance(mass, bool)
        or not 350 <= mass <= 1200
        or not numeric_dimensions
        or max(dimensions) < 8.0
        or mass > density * math.prod(dimensions)
        or value["size_class"] != "palm_sized_explicit"
        or not isinstance(value["source_specific_feature"], str)
        or not value["source_specific_feature"].strip()
        or not isinstance(value["curator_note"], str)
        or not value["curator_note"].strip()
    ):
        raise ValueError(f"{label}: impact-plausibility verdict is not an executable pass")


def validate_causal_candidate_ontology_bindings(
    project_root: Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_ontology_path: Path,
    source_split_path: Path,
    holdout_registry_path: Path,
    receiver_ontology_path: Path,
) -> None:
    """Rebind every causal candidate to the frozen v2 source/receiver ontology."""

    source_payload = _load_exact_json(source_ontology_path, "private source ontology")
    source_top = {"schema", "protocol", "dataset_version", "source_count", "sources"}
    if not isinstance(source_payload, dict) or set(source_payload) != source_top or (
        source_payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or source_payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or source_payload["dataset_version"] != DATASET_VERSION
        or source_payload["source_count"] != 80
        or not isinstance(source_payload["sources"], list)
        or len(source_payload["sources"]) != 80
    ):
        raise ValueError("private source ontology identity/count is not exact")
    source_fields = {
        "source_id", "source_phrase", "normalized_phrase", "head_lemma", "origin",
        "food_status", "shape_class", "color_family", "material_family", "texture_class",
        "impact_plausibility", "curator", "curation_stratum",
    }
    sources: dict[str, Mapping[str, Any]] = {}
    stratum_counts: Counter[str] = Counter()
    for index, row in enumerate(source_payload["sources"]):
        if not isinstance(row, dict) or set(row) != source_fields:
            raise ValueError(f"private source ontology row {index} fields are not exact")
        string_fields = source_fields - {"impact_plausibility"}
        if any(
            not isinstance(row[field], str) or not row[field] or "\x00" in row[field]
            for field in string_fields
        ):
            raise ValueError(f"private source ontology row {index} has invalid text")
        source_id = row["source_id"]
        if source_id in sources:
            raise ValueError("private source ontology IDs are not unique")
        if (
            row["normalized_phrase"] != _normalize_registry_phrase(row["source_phrase"])
            or row["head_lemma"] != _normalize_registry_phrase(row["head_lemma"]).replace(" ", "_")
            or row["curation_stratum"] not in CURATION_STRATA
        ):
            raise ValueError(f"private source ontology row {index} is noncanonical")
        _validate_impact_plausibility(
            row["impact_plausibility"], label=f"private source ontology row {index}"
        )
        sources[source_id] = row
        stratum_counts[row["curation_stratum"]] += 1
    if stratum_counts != Counter({name: 20 for name in CURATION_STRATA}):
        raise ValueError("private source ontology curation strata are not exactly balanced")

    split_payload = _load_exact_json(source_split_path, "private source split")
    if not isinstance(split_payload, dict) or set(split_payload) != {
        "schema", "protocol", "dataset_version", "domain", "rows"
    } or (
        split_payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or split_payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or split_payload["dataset_version"] != DATASET_VERSION
        or split_payload["domain"] != "bank-holdout-v2"
        or not isinstance(split_payload["rows"], list)
        or len(split_payload["rows"]) != 80
    ):
        raise ValueError("private source split identity/count is not exact")
    split_rows: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(split_payload["rows"]):
        if not isinstance(row, dict) or set(row) != {
            "source_id", "membership", "split_rank_sha256"
        } or row["membership"] not in {"new_bank_source", "holdout_source"} or not is_sha256(
            row["split_rank_sha256"]
        ):
            raise ValueError(f"private source split row {index} is not exact")
        source_id = row["source_id"]
        if source_id not in sources or source_id in split_rows:
            raise ValueError("private source split IDs differ from the ontology")
        split_rows[source_id] = row["membership"]
        split_counts[row["membership"]] += 1
    if set(split_rows) != set(sources) or split_counts != Counter(
        {"new_bank_source": 56, "holdout_source": 24}
    ):
        raise ValueError("private source split inventory/counts are not exact")

    holdout_payload = _load_exact_json(holdout_registry_path, "private holdout registry")
    if not isinstance(holdout_payload, dict) or set(holdout_payload) != {
        "schema", "protocol", "dataset_version", "registry", "ordering", "entries"
    } or (
        holdout_payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or holdout_payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or holdout_payload["dataset_version"] != DATASET_VERSION
        or holdout_payload["registry"] != "private_ordered_holdout24_v2"
        or not isinstance(holdout_payload["ordering"], str)
        or not holdout_payload["ordering"]
        or not isinstance(holdout_payload["entries"], list)
        or len(holdout_payload["entries"]) != 24
    ):
        raise ValueError("private holdout registry identity/count is not exact")
    holdout_fields = {
        "holdout_index", "source_id", "source_phrase", "normalized_phrase", "head_lemma",
        "impact_plausibility", "strata",
    }
    holdouts: dict[str, Mapping[str, Any]] = {}
    previous_id = ""
    for index, row in enumerate(holdout_payload["entries"]):
        if not isinstance(row, dict) or set(row) != holdout_fields or row["holdout_index"] != index:
            raise ValueError(f"private holdout registry row {index} is not exact")
        source_id = row["source_id"]
        source = sources.get(source_id)
        if (
            not isinstance(source_id, str)
            or source_id <= previous_id
            or source is None
            or split_rows.get(source_id) != "holdout_source"
            or row["source_phrase"] != source["source_phrase"]
            or row["normalized_phrase"] != source["normalized_phrase"]
            or row["head_lemma"] != source["head_lemma"]
            or row["impact_plausibility"] != source["impact_plausibility"]
            or not isinstance(row["strata"], dict)
            or set(row["strata"]) != SOURCE_STRATA_FIELDS
            or any(row["strata"][field] != source[field] for field in SOURCE_STRATA_FIELDS)
        ):
            raise ValueError(f"private holdout registry row {index} differs from source/split")
        _validate_impact_plausibility(row["impact_plausibility"], label=f"holdout row {index}")
        previous_id = source_id
        holdouts[source_id] = row
    if set(holdouts) != {source_id for source_id, membership in split_rows.items() if membership == "holdout_source"}:
        raise ValueError("private holdout registry does not equal the split holdout set")

    receiver_payload = _load_exact_json(receiver_ontology_path, "private receiver ontology")
    if not isinstance(receiver_payload, dict) or set(receiver_payload) != {
        "schema", "protocol", "dataset_version", "receiver_count", "receivers",
        "historical_receiver_blacklist_count",
    } or (
        receiver_payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or receiver_payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or receiver_payload["dataset_version"] != DATASET_VERSION
        or receiver_payload["receiver_count"] != 32
        or receiver_payload["historical_receiver_blacklist_count"] != 52
        or not isinstance(receiver_payload["receivers"], list)
        or len(receiver_payload["receivers"]) != 32
    ):
        raise ValueError("private receiver ontology identity/count is not exact")
    receiver_fields = {
        "receiver_id", "receiver_phrase", "normalized_phrase", "head_lemma", "receiver_type",
        "curator_note", "curator",
    }
    receivers: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(receiver_payload["receivers"]):
        if not isinstance(row, dict) or set(row) != receiver_fields or any(
            not isinstance(row[field], str) or not row[field] or "\x00" in row[field]
            for field in receiver_fields
        ):
            raise ValueError(f"private receiver ontology row {index} is not exact")
        receiver_id = row["receiver_id"]
        if receiver_id in receivers or row["receiver_type"] not in RECEIVER_TYPES or (
            row["normalized_phrase"] != _normalize_registry_phrase(row["receiver_phrase"])
            or row["head_lemma"] != _normalize_registry_phrase(row["head_lemma"]).replace(" ", "_")
        ):
            raise ValueError(f"private receiver ontology row {index} is noncanonical")
        receivers[receiver_id] = row

    bank_payload = _load_exact_json(resolve_path(project_root, PUBLIC_SOURCE_BANK), "public source bank")
    bank_entries = bank_payload.get("entries") if isinstance(bank_payload, dict) else None
    if (
        not isinstance(bank_payload, dict)
        or set(bank_payload)
        != {
            "schema", "protocol", "dataset_version", "registry", "canonical_json",
            "canonical_builder_sha256", "training_manifest_sha256", "source_assignment_salt",
            "source_assignment_algorithm", "counts", "entries", "bank_entries_sha256",
            "supersedes", "curation_audit",
        }
        or bank_payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bank_payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bank_payload["dataset_version"] != DATASET_VERSION
        or bank_payload["registry"] != "public_augmentation_bank64_v2"
        or bank_payload["canonical_json"] != FIELD_NORMALIZATION_RULES["canonical_record"]
        or bank_payload["counts"] != {"new_ontology": 56, "original_training": 8, "total": 64}
        or bank_payload["supersedes"] != V2_SUPERSEDES
        or bank_payload["curation_audit"] != CURATION_AUDIT
        or not isinstance(bank_entries, list)
        or len(bank_entries) != 64
        or hashlib.sha256(
            (
                json.dumps(
                    bank_entries,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        != bank_payload["bank_entries_sha256"]
    ):
        raise ValueError("public source bank must contain exactly 64 entries")
    original_bank: dict[str, Mapping[str, Any]] = {}
    new_bank: dict[str, Mapping[str, Any]] = {}
    for row in bank_entries:
        if not isinstance(row, dict):
            raise ValueError("public source bank row is not an object")
        membership = row.get("membership")
        source_id = row.get("source_id")
        if membership == "original_training_source":
            if set(row) != {
                "bank_index", "source_id", "source_phrase", "normalized_phrase",
                "head_lemma", "membership", "physical_audit_status",
            } or row["physical_audit_status"] != "legacy_original_source_exempt":
                raise ValueError("public original source physical status/fields are not exact")
            original_bank[str(source_id)] = row
        elif membership == "new_bank_source":
            if set(row) != {
                "bank_index", "source_id", "source_phrase", "normalized_phrase",
                "head_lemma", "membership", "physical_audit_status", "strata",
                "impact_plausibility",
            } or row["physical_audit_status"] != "strict_physical_pass_v2":
                raise ValueError("public new-bank physical status/fields are not exact")
            _validate_impact_plausibility(
                row["impact_plausibility"], label=f"public new-bank source {source_id}"
            )
            new_bank[str(source_id)] = row
        else:
            raise ValueError("public source bank membership is invalid")
    if set(original_bank) != set(ORIGINAL_TRAINING_SOURCES) or len(new_bank) != 56:
        raise ValueError("public source bank original/new inventory is not exact")
    for index, (source_id, phrase) in enumerate(ORIGINAL_TRAINING_SOURCES.items()):
        row = original_bank[source_id]
        if (
            row.get("bank_index") != index
            or row.get("source_phrase") != phrase
            or row.get("normalized_phrase") != _normalize_registry_phrase(phrase)
            or row.get("head_lemma") != _normalize_registry_phrase(row.get("head_lemma", "")).replace(" ", "_")
        ):
            raise ValueError("public original training source differs from the frozen builder")
    if set(new_bank) != {source_id for source_id, membership in split_rows.items() if membership == "new_bank_source"}:
        raise ValueError("public new-bank inventory differs from private source split")
    for source_id, row in new_bank.items():
        source = sources[source_id]
        if any(
            row.get(field) != source[field]
            for field in ("source_id", "source_phrase", "normalized_phrase", "head_lemma")
        ) or row["impact_plausibility"] != source["impact_plausibility"] or row["strata"] != {
            field: source[field] for field in SOURCE_STRATA_FIELDS
        }:
            raise ValueError("public new-bank identity differs from private source ontology")

    if len(candidates) != 48:
        raise ValueError("causal candidate ontology binding requires all 48 candidates")
    for row in candidates:
        group = str(row["group"])
        source_id = str(row["source_id"])
        receiver_id = str(row["receiver_id"])
        if group == "seen_source_new_receiver":
            source = original_bank.get(source_id)
            if source is None:
                raise ValueError("candidate original source is absent from the public source bank")
        else:
            source = holdouts.get(source_id)
            if source is None:
                raise ValueError("candidate holdout source is absent from the real holdout registry")
        if (
            row["source_phrase"] != source["source_phrase"]
            or row["source_head_lemma"] != source["head_lemma"]
            or row["source_physical_audit_status"]
            != source.get(
                "physical_audit_status",
                "strict_physical_pass_v2" if group != "seen_source_new_receiver" else "",
            )
        ):
            raise ValueError("candidate source identity differs from its frozen ontology row")
        if group == "holdout_source_seen_receiver":
            if HISTORICAL_TRAINING_RECEIVERS.get(receiver_id) != row["receiver_phrase"]:
                raise ValueError("candidate historical receiver differs from the training ontology")
        else:
            receiver = receivers.get(receiver_id)
            if receiver is None or receiver["receiver_phrase"] != row["receiver_phrase"]:
                raise ValueError("candidate new receiver differs from the private receiver ontology")


def prepare_selection_binding(
    project_root: Path,
    *,
    dataset: str,
    private_root: Path,
    candidate_manifest_path: Path,
    canonical_templates_path: Path,
    field_rules_path: Path,
    render_configuration_path: Path,
    selection_rules_path: Path,
    secrets_path: Path,
    root_bundle_path: Path,
    generation_spec_path: Path,
    screening_seed_path: Path,
    selector_salt_path: Path,
    evaluation_seed_salt_path: Path,
    forbidden_seed_inventory_path: Path,
    source_ontology_path: Path | None = None,
    source_split_path: Path | None = None,
    holdout_registry_path: Path | None = None,
    receiver_ontology_path: Path | None = None,
    new_bank_assignment_path: Path | None = None,
    causal_stage0_registry_path: Path | None = None,
    causal_stage1_registry_path: Path | None = None,
    causal_selected_path: Path | None = None,
    causal_unit_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate pending/raw/downstream inputs and construct, but do not write, a binding."""

    if dataset == "specificity":
        required = (
            new_bank_assignment_path,
            causal_stage0_registry_path,
            causal_stage1_registry_path,
            causal_selected_path,
            causal_unit_manifest_path,
        )
        if any(path is None for path in required):
            raise ValueError("specificity Stage-0 requires its assignment and causal Stage-1 chain")
        return prepare_specificity_selection_binding(
            project_root,
            private_root=private_root,
            candidate_manifest_path=candidate_manifest_path,
            new_bank_assignment_path=new_bank_assignment_path,
            canonical_templates_path=canonical_templates_path,
            field_rules_path=field_rules_path,
            render_configuration_path=render_configuration_path,
            selection_rules_path=selection_rules_path,
            secrets_path=secrets_path,
            root_bundle_path=root_bundle_path,
            generation_spec_path=generation_spec_path,
            screening_seed_path=screening_seed_path,
            selector_salt_path=selector_salt_path,
            evaluation_seed_salt_path=evaluation_seed_salt_path,
            forbidden_seed_inventory_path=forbidden_seed_inventory_path,
            causal_stage0_registry_path=causal_stage0_registry_path,
            causal_stage1_registry_path=causal_stage1_registry_path,
            causal_selected_path=causal_selected_path,
            causal_unit_manifest_path=causal_unit_manifest_path,
        )
    if dataset != "causal":
        raise ValueError("unsupported selection binding dataset")
    if any(
        path is None
        for path in (
            source_ontology_path,
            source_split_path,
            holdout_registry_path,
            receiver_ontology_path,
        )
    ):
        raise ValueError("causal Stage-0 requires source/split/holdout/receiver ontology openings")
    component_paths = {
        "causal_stage0_candidates_private_v2.json": candidate_manifest_path,
        "causal_stage0_templates_private_v2.json": canonical_templates_path,
        "causal_stage0_field_rules_private_v2.json": field_rules_path,
        "causal_stage0_render_config_private_v2.json": render_configuration_path,
        "causal_stage0_selection_rules_private_v2.json": selection_rules_path,
        "causal_stage0_secrets_private_v2.json": secrets_path,
    }
    downstream_paths = {
        "root bundle": root_bundle_path,
        "generation spec": generation_spec_path,
        "screening seed": screening_seed_path,
        "selector salt": selector_salt_path,
        "evaluation seed salt": evaluation_seed_salt_path,
        "forbidden seed inventory": forbidden_seed_inventory_path,
        "source ontology": source_ontology_path,
        "source split": source_split_path,
        "holdout registry": holdout_registry_path,
        "receiver ontology": receiver_ontology_path,
    }
    for name, path in {**component_paths, **downstream_paths}.items():
        _private_opening_path(path, private_root, name)

    pending_path = resolve_path(project_root, PENDING_STAGE0_COMMITMENTS[dataset])
    pending = _load_exact_json(pending_path, "public pending Stage-0 commitment")
    if file_sha256(pending_path) != FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256:
        raise ValueError("public causal Stage-0 pending bytes differ from canonical v2")
    expected_pending_keys = {
        "authorization_status",
        "candidate_count",
        "candidate_manifest_sha256",
        "canonical_json",
        "canonical_templates_sha256",
        "cell_counts",
        "curation_audit",
        "dataset_version",
        "evaluation_seed_salt_commitment_sha256",
        "field_normalization_sha256",
        "protocol",
        "public_metadata",
        "registry",
        "remaining_blockers",
        "render_configuration_sha256",
        "schema",
        "screening_seed_commitment_sha256",
        "selector_rules_sha256",
        "selector_salt_commitment_sha256",
        "stage",
        "stage0_bundle_file_sha256",
        "status",
        "supersedes",
    }
    if set(pending) != expected_pending_keys:
        raise ValueError("pending public Stage-0 fields are not exact")
    if (
        pending["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or pending["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or pending["registry"] != "causal_stage0_public_commitment_v2"
        or pending["dataset_version"] != DATASET_VERSION
        or pending["stage"] != 0
        or pending["candidate_count"] != 48
        or pending["status"] != "frozen_components_pending_external_bindings"
        or pending["authorization_status"] != "not_authorized"
        or pending["canonical_json"] != FIELD_NORMALIZATION_RULES["canonical_record"]
        or pending["supersedes"] != V2_SUPERSEDES
        or pending["curation_audit"] != CURATION_AUDIT
    ):
        raise ValueError("pending public Stage-0 overstates or changes its frozen scope")
    expected_cells = {
        f"{group}:{variant}": 8
        for group in CAUSAL_GROUPS
        for variant in PROMPT_VARIANTS
    }
    expected_public_metadata = {
        "candidates_per_cell": 8,
        "evaluation_seed_domain": "causal-eval-seed-v2",
        "evaluation_seed_namespace": "v4-causal-evaluation-v2",
        "evaluation_unit_target": 72,
        "full_frame_screening_required": True,
        "groups": list(CAUSAL_GROUPS),
        "no_reserve_queue": True,
        "prompt_variants": list(PROMPT_VARIANTS),
        "ranking_domain": "causal-selector-v2",
        "replicates_per_selected_case": 3,
        "screening_arm": "Original_only",
        "screening_seed_namespace": "v4-causal-stage0-screening-v2",
        "selected_case_target": 24,
        "selection_per_cell": 4,
        "source_physical_policy": CURATION_AUDIT["legacy_original_source_policy"],
    }
    blockers = pending["remaining_blockers"]
    if (
        pending["cell_counts"] != expected_cells
        or pending["public_metadata"] != expected_public_metadata
        or not isinstance(blockers, list)
        or blockers
        != [
            "an independent seed auditor must commit the complete forbidden numeric seed inventory and prove disjointness",
            "an independent binder must commit the exact already-frozen v3b path-plus-file-bytes model inventory digest",
        ]
    ):
        raise ValueError("pending public Stage-0 metadata/blocker contract differs")
    for field in (
        "candidate_manifest_sha256",
        "canonical_templates_sha256",
        "field_normalization_sha256",
        "render_configuration_sha256",
        "selector_rules_sha256",
        "screening_seed_commitment_sha256",
        "selector_salt_commitment_sha256",
        "evaluation_seed_salt_commitment_sha256",
        "stage0_bundle_file_sha256",
    ):
        _require_sha256(pending[field], f"pending Stage-0/{field}")

    component_hashes = {name: file_sha256(path) for name, path in component_paths.items()}
    pending_component_fields = {
        "candidate_manifest_sha256": "causal_stage0_candidates_private_v2.json",
        "canonical_templates_sha256": "causal_stage0_templates_private_v2.json",
        "field_normalization_sha256": "causal_stage0_field_rules_private_v2.json",
        "render_configuration_sha256": "causal_stage0_render_config_private_v2.json",
        "selector_rules_sha256": "causal_stage0_selection_rules_private_v2.json",
    }
    if any(
        pending[field] != component_hashes[filename]
        for field, filename in pending_component_fields.items()
    ):
        raise ValueError("pending Stage-0 component digest does not match opened bytes")
    bundle = _load_exact_json(root_bundle_path, "private Stage-0 root bundle")
    if set(bundle) != {
        "schema",
        "protocol",
        "dataset_version",
        "stage",
        "status",
        "components",
        "source_bank_entries_sha256",
        "holdout_registry_file_sha256",
    }:
        raise ValueError("private Stage-0 root bundle fields are not exact")
    if (
        bundle["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bundle["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bundle["dataset_version"] != DATASET_VERSION
        or bundle["stage"] != 0
        or bundle["status"] != "frozen_components_pending_external_bindings"
        or bundle["components"] != component_hashes
        or pending["stage0_bundle_file_sha256"] != file_sha256(root_bundle_path)
    ):
        raise ValueError("private Stage-0 root/component binding mismatch")
    bank_path = resolve_path(project_root, PUBLIC_SOURCE_BANK)
    holdout_path = resolve_path(project_root, PUBLIC_HOLDOUT_COMMITMENT)
    bank = _load_exact_json(bank_path, "public source bank")
    holdout = _load_exact_json(holdout_path, "public holdout commitment")
    if file_sha256(bank_path) != FROZEN_PUBLIC_SOURCE_BANK_SHA256:
        raise ValueError("public source bank bytes differ from canonical v2")
    if file_sha256(holdout_path) != FROZEN_PUBLIC_HOLDOUT_COMMITMENT_SHA256:
        raise ValueError("public holdout commitment bytes differ from canonical v2")
    if (
        bundle["source_bank_entries_sha256"] != bank.get("bank_entries_sha256")
        or bundle["holdout_registry_file_sha256"]
        != holdout.get("holdout_registry_file_sha256")
        or bank.get("curation_audit") != CURATION_AUDIT
        or holdout.get("curation_audit") != CURATION_AUDIT
        or pending["curation_audit"] != bank.get("curation_audit")
    ):
        raise ValueError("private Stage-0 root differs from public source commitments")

    field_rules = _load_exact_json(field_rules_path, "selection field rules")
    if field_rules != FIELD_NORMALIZATION_RULES:
        raise ValueError("selection field-normalization rules differ from executable contract")
    selection_rules = _load_exact_json(selection_rules_path, "selection rules")
    if selection_rules != CAUSAL_SELECTION_RULES:
        raise ValueError("causal selection rules differ from executable contract")
    templates = _load_exact_json(canonical_templates_path, "canonical templates")
    if templates != {
        "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
        "protocol": SOURCE_SLOT_REGISTRY_SCHEMA,
        "dataset_version": DATASET_VERSION,
        "canonical_builder_sha256": file_sha256(
            resolve_path(project_root, "scripts/build_water_impact_dynamic_pairs_v1.py")
        ),
        "prompt_templates": CAUSAL_CANONICAL_TEMPLATES,
        "template_fill_rules": CAUSAL_TEMPLATE_FILL_RULES,
        "non_substitution_rule": CAUSAL_TEMPLATE_NON_SUBSTITUTION_RULE,
    }:
        raise ValueError("canonical template artifact differs from exact non-substitution contract")
    normalized_candidates = load_normalized_candidate_manifest(
        candidate_manifest_path,
        dataset=dataset,
        canonical_templates_path=canonical_templates_path,
    )
    validate_causal_candidate_ontology_bindings(
        project_root,
        normalized_candidates,
        source_ontology_path=source_ontology_path,
        source_split_path=source_split_path,
        holdout_registry_path=holdout_registry_path,
        receiver_ontology_path=receiver_ontology_path,
    )
    render = _load_exact_json(render_configuration_path, "raw render configuration")
    if render != CAUSAL_RENDER_CONFIGURATION:
        raise ValueError("raw render configuration differs from executable contract")

    secrets = _load_exact_json(secrets_path, "Stage-0 secrets")
    if set(secrets) != {
        "schema",
        "protocol",
        "dataset_version",
        "evaluation_seed_namespace",
        "evaluation_seed_salt",
        "screening_seed",
        "screening_seed_namespace",
        "selector_salt",
    } or (
        secrets["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or secrets["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or secrets["dataset_version"] != DATASET_VERSION
        or secrets["evaluation_seed_namespace"] != "v4-causal-evaluation-v2"
        or secrets["screening_seed_namespace"] != "v4-causal-stage0-screening-v2"
        or not isinstance(secrets["screening_seed"], int)
        or isinstance(secrets["screening_seed"], bool)
        or not 3_000_000_000 <= secrets["screening_seed"] < 3_500_000_000
        or not re.fullmatch(r"[0-9a-f]{64}", str(secrets["selector_salt"]))
        or not re.fullmatch(r"[0-9a-f]{64}", str(secrets["evaluation_seed_salt"]))
        or secrets["selector_salt"] == secrets["evaluation_seed_salt"]
    ):
        raise ValueError("Stage-0 secret schema/value domains are not exact")

    screening_text = screening_seed_path.read_bytes()
    selector_text = selector_salt_path.read_bytes()
    evaluation_text = evaluation_seed_salt_path.read_bytes()
    expected_screening_text = f"{secrets['screening_seed']}\n".encode("ascii")
    expected_selector_text = f"{secrets['selector_salt']}\n".encode("ascii")
    expected_evaluation_text = f"{secrets['evaluation_seed_salt']}\n".encode("ascii")
    if (
        screening_text != expected_screening_text
        or selector_text != expected_selector_text
        or evaluation_text != expected_evaluation_text
    ):
        raise ValueError("standalone seed/salt openings differ from curator secrets")
    if (
        pending["screening_seed_commitment_sha256"]
        != _secret_commitment("causal_screening_seed_v2", str(secrets["screening_seed"]))
        or pending["selector_salt_commitment_sha256"]
        != _secret_commitment("causal_stage0_selector_salt_v2", secrets["selector_salt"])
        or pending["evaluation_seed_salt_commitment_sha256"]
        != _secret_commitment(
            "causal_evaluation_seed_salt_v2", secrets["evaluation_seed_salt"]
        )
    ):
        raise ValueError("private seed/salt opening does not match public commitment")

    generation_spec = _load_exact_json(generation_spec_path, "generation spec")
    runtime_ref = generation_spec.get("runtime_registry")
    if set(generation_spec) != {
        "protocol",
        "status",
        "model_inventory_sha256",
        "runtime_registry",
        "generation_spec",
        "source_mode",
    } or (
        generation_spec["protocol"] != GENERATION_SPEC_PROTOCOL
        or generation_spec["status"] != "frozen_before_original_render"
        or generation_spec["source_mode"] != "Original_screening_then_matched_O_v3b_v4"
        or generation_spec["generation_spec"] != GENERATION_SPEC
        or generation_spec["model_inventory_sha256"]
        != FROZEN_MODEL_CONTENT_INVENTORY_SHA256
        or not isinstance(runtime_ref, dict)
        or set(runtime_ref) != {"path", "sha256"}
        or runtime_ref["path"] != RUNTIME_REGISTRY
        or not is_sha256(runtime_ref["sha256"])
    ):
        raise ValueError("generation spec differs from exact render/model contract")
    validate_runtime_registry(
        resolve_path(project_root, RUNTIME_REGISTRY),
        runtime_ref["sha256"],
    )
    forbidden = validate_forbidden_seed_inventory(
        forbidden_seed_inventory_path, dataset=dataset
    )
    if secrets["screening_seed"] in forbidden:
        raise ValueError("screening seed occurs in the external forbidden inventory")
    seed_audit = audit_preselection_seed_space(
        normalized_candidates,
        dataset=dataset,
        private_salt=secrets["evaluation_seed_salt"],
        screening_seed=secrets["screening_seed"],
        forbidden_seeds=forbidden,
    )
    return expected_selection_binding(
        dataset=dataset,
        public_pending_sha256=file_sha256(pending_path),
        root_bundle_sha256=file_sha256(root_bundle_path),
        component_sha256=component_hashes,
        generation_spec_sha256=file_sha256(generation_spec_path),
        model_inventory_sha256=generation_spec["model_inventory_sha256"],
        runtime_registry_sha256=runtime_ref["sha256"],
        screening_seed_sha256=file_sha256(screening_seed_path),
        selector_salt_sha256=file_sha256(selector_salt_path),
        evaluation_seed_salt_sha256=file_sha256(evaluation_seed_salt_path),
        forbidden_seed_inventory_sha256=file_sha256(forbidden_seed_inventory_path),
        preselection_seed_audit_sha256=seed_audit[
            "ordered_case_replicate_seed_sha256"
        ],
        preselection_seed_count=seed_audit["derived_seed_count"],
    )


def prepare_specificity_selection_binding(
    project_root: Path,
    *,
    private_root: Path,
    candidate_manifest_path: Path,
    new_bank_assignment_path: Path,
    canonical_templates_path: Path,
    field_rules_path: Path,
    render_configuration_path: Path,
    selection_rules_path: Path,
    secrets_path: Path,
    root_bundle_path: Path,
    generation_spec_path: Path,
    screening_seed_path: Path,
    selector_salt_path: Path,
    evaluation_seed_salt_path: Path,
    forbidden_seed_inventory_path: Path,
    causal_stage0_registry_path: Path,
    causal_stage1_registry_path: Path,
    causal_selected_path: Path,
    causal_unit_manifest_path: Path,
) -> dict[str, Any]:
    """Construct the specificity Stage-0 bridge only after causal Stage-1 exists."""

    component_paths = {
        "specificity_stage0_candidates_private_v2.json": candidate_manifest_path,
        "specificity_stage0_new_bank_assignment_private_v2.json": new_bank_assignment_path,
        "specificity_stage0_templates_private_v2.json": canonical_templates_path,
        "specificity_stage0_field_rules_private_v2.json": field_rules_path,
        "specificity_stage0_render_config_private_v2.json": render_configuration_path,
        "specificity_stage0_selection_rules_private_v2.json": selection_rules_path,
        "specificity_stage0_secrets_private_v2.json": secrets_path,
    }
    for label, path in {
        **component_paths,
        "specificity root bundle": root_bundle_path,
        "specificity generation spec": generation_spec_path,
        "specificity screening seed": screening_seed_path,
        "specificity selector salt": selector_salt_path,
        "specificity evaluation seed salt": evaluation_seed_salt_path,
        "specificity forbidden seeds": forbidden_seed_inventory_path,
        "causal selected24": causal_selected_path,
        "causal U72": causal_unit_manifest_path,
    }.items():
        _private_opening_path(path, private_root, label)
    expected_causal0 = resolve_path(project_root, CAUSAL_STAGE0)
    expected_causal1 = resolve_path(project_root, CAUSAL_STAGE1)
    if (
        causal_stage0_registry_path.resolve() != expected_causal0.resolve()
        or causal_stage1_registry_path.resolve() != expected_causal1.resolve()
    ):
        raise ValueError("specificity Stage-0 causal registry paths differ from protocol")
    causal0 = validate_commitment_registry(
        causal_stage0_registry_path, dataset="causal", stage=0
    )
    causal1 = validate_commitment_registry(
        causal_stage1_registry_path,
        dataset="causal",
        stage=1,
        expected_stage0_sha256=file_sha256(causal_stage0_registry_path),
    )
    del causal0
    if (
        file_sha256(causal_selected_path)
        != causal1["artifacts"]["selected_case_manifest_24"]["sha256"]
        or file_sha256(causal_unit_manifest_path)
        != causal1["artifacts"]["unit_manifest_U_72"]["sha256"]
    ):
        raise ValueError("specificity Stage-0 differs from committed causal selected24/U72")
    causal_selected = read_csv(causal_selected_path)
    causal_units = read_csv(causal_unit_manifest_path)
    validate_causal_selected_cases(causal_selected)
    validate_causal_unit_manifest(causal_units)

    pending_path = resolve_path(project_root, PENDING_STAGE0_COMMITMENTS["specificity"])
    pending = _load_exact_json(pending_path, "specificity pending Stage-0 commitment")
    expected_pending_fields = {
        "schema",
        "dataset_version",
        "stage",
        "status",
        "authorization_status",
        "candidate_count",
        "candidate_manifest_sha256",
        "new_bank_assignment_sha256",
        "canonical_templates_sha256",
        "field_normalization_sha256",
        "render_configuration_sha256",
        "selector_rules_sha256",
        "screening_seed_commitment_sha256",
        "selector_salt_commitment_sha256",
        "evaluation_seed_salt_commitment_sha256",
        "stage0_bundle_file_sha256",
        "causal_stage0_registry_sha256",
        "causal_stage1_registry_sha256",
        "selected_case_manifest_24_sha256",
        "unit_manifest_U_72_sha256",
        "remaining_blockers",
    }
    if not isinstance(pending, dict) or set(pending) != expected_pending_fields:
        raise ValueError("specificity pending Stage-0 fields are not exact")
    if (
        pending["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or pending["dataset_version"] != DATASET_VERSION
        or pending["stage"] != 0
        or pending["candidate_count"] != 36
        or pending["status"] != "frozen_components_pending_external_bindings"
        or pending["authorization_status"] != "not_authorized"
        or pending["causal_stage0_registry_sha256"]
        != file_sha256(causal_stage0_registry_path)
        or pending["causal_stage1_registry_sha256"]
        != file_sha256(causal_stage1_registry_path)
        or pending["selected_case_manifest_24_sha256"]
        != file_sha256(causal_selected_path)
        or pending["unit_manifest_U_72_sha256"]
        != file_sha256(causal_unit_manifest_path)
        or pending["remaining_blockers"]
        != [
            "independent forbidden numeric seed inventory",
            "exact full-model path-plus-file-bytes inventory digest",
        ]
    ):
        raise ValueError("specificity pending Stage-0 identity/dependency mismatch")
    component_hashes = {name: file_sha256(path) for name, path in component_paths.items()}
    pending_map = {
        "candidate_manifest_sha256": "specificity_stage0_candidates_private_v2.json",
        "new_bank_assignment_sha256": "specificity_stage0_new_bank_assignment_private_v2.json",
        "canonical_templates_sha256": "specificity_stage0_templates_private_v2.json",
        "field_normalization_sha256": "specificity_stage0_field_rules_private_v2.json",
        "render_configuration_sha256": "specificity_stage0_render_config_private_v2.json",
        "selector_rules_sha256": "specificity_stage0_selection_rules_private_v2.json",
    }
    for field in (*pending_map, "screening_seed_commitment_sha256", "selector_salt_commitment_sha256", "evaluation_seed_salt_commitment_sha256", "stage0_bundle_file_sha256"):
        _require_sha256(pending[field], f"specificity pending/{field}")
    if any(
        pending[field] != component_hashes[filename]
        for field, filename in pending_map.items()
    ):
        raise ValueError("specificity pending component digest differs from opened bytes")

    bundle = _load_exact_json(root_bundle_path, "specificity private Stage-0 bundle")
    if set(bundle) != {
        "schema",
        "protocol",
        "dataset_version",
        "stage",
        "status",
        "components",
        "source_bank_entries_sha256",
        "causal_stage0_registry_sha256",
        "causal_stage1_registry_sha256",
        "selected_case_manifest_24_sha256",
        "unit_manifest_U_72_sha256",
    } or (
        bundle["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bundle["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or bundle["dataset_version"] != DATASET_VERSION
        or bundle["stage"] != 0
        or bundle["status"] != "frozen_components_pending_external_bindings"
        or bundle["components"] != component_hashes
        or pending["stage0_bundle_file_sha256"] != file_sha256(root_bundle_path)
        or bundle["causal_stage0_registry_sha256"]
        != file_sha256(causal_stage0_registry_path)
        or bundle["causal_stage1_registry_sha256"]
        != file_sha256(causal_stage1_registry_path)
        or bundle["selected_case_manifest_24_sha256"]
        != file_sha256(causal_selected_path)
        or bundle["unit_manifest_U_72_sha256"]
        != file_sha256(causal_unit_manifest_path)
    ):
        raise ValueError("specificity private root/dependency binding mismatch")
    bank = _load_exact_json(resolve_path(project_root, PUBLIC_SOURCE_BANK), "public source bank")
    if bundle["source_bank_entries_sha256"] != bank.get("bank_entries_sha256"):
        raise ValueError("specificity bundle differs from public source bank")

    if _load_exact_json(field_rules_path, "specificity field rules") != FIELD_NORMALIZATION_RULES:
        raise ValueError("specificity field rules differ from executable contract")
    if _load_exact_json(selection_rules_path, "specificity selection rules") != SPECIFICITY_SELECTION_RULES:
        raise ValueError("specificity selection rules differ from executable contract")
    normalized = load_normalized_candidate_manifest(
        candidate_manifest_path,
        dataset="specificity",
        canonical_templates_path=canonical_templates_path,
    )
    candidate_payload = _load_exact_json(candidate_manifest_path, "specificity candidates")
    if candidate_payload["causal_stage1_registry_sha256"] != file_sha256(
        causal_stage1_registry_path
    ):
        raise ValueError("specificity candidates do not bind current causal Stage-1")
    causal_by_id = {str(row["semantic_case_id"]): row for row in causal_selected}
    for row in normalized:
        if row["membership"] == "new_bank_source":
            continue
        causal = causal_by_id.get(str(row["causal_case_id"]))
        expected_membership = (
            "original_source"
            if causal is not None and causal.get("source_membership") == "original_source"
            else "holdout_source"
        )
        if causal is None or row["membership"] != expected_membership or any(
            str(row[field])
            != str(
                causal[
                    "receiver" if field == "receiver_phrase" else field
                ]
            )
            for field in (
                "source_id",
                "source_phrase",
                "source_head_lemma",
                "receiver_id",
                "receiver_phrase",
                "prompt_variant",
            )
        ):
            raise ValueError("matched specificity candidate differs from causal selected case")

    assignment = _load_exact_json(new_bank_assignment_path, "specificity new-bank assignment")
    if set(assignment) != {
        "schema",
        "dataset_version",
        "causal_stage1_registry_sha256",
        "assignment_count",
        "assignments",
    } or (
        assignment["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or assignment["dataset_version"] != DATASET_VERSION
        or assignment["causal_stage1_registry_sha256"]
        != file_sha256(causal_stage1_registry_path)
        or assignment["assignment_count"] != 12
        or not isinstance(assignment["assignments"], list)
        or len(assignment["assignments"]) != 12
    ):
        raise ValueError("specificity new-bank assignment identity/count mismatch")
    assignment_fields = {
        "case_id",
        "source_id",
        "source_phrase",
        "source_head_lemma",
        "receiver_id",
        "receiver_phrase",
        "prompt_variant",
        "rank_sha256",
    }
    new_rows = {row["case_id"]: row for row in normalized if row["membership"] == "new_bank_source"}
    public_bank = {
        str(row.get("source_id")): row
        for row in bank.get("entries", [])
        if isinstance(row, dict) and row.get("membership") == "new_bank_source"
    }
    seen_sources: set[str] = set()
    seen_receivers: set[str] = set()
    for row in assignment["assignments"]:
        if not isinstance(row, dict) or set(row) != assignment_fields or not is_sha256(row["rank_sha256"]):
            raise ValueError("specificity new-bank assignment row is not exact")
        candidate = new_rows.get(str(row["case_id"]))
        bank_row = public_bank.get(str(row["source_id"]))
        if candidate is None or bank_row is None or any(
            str(row[field]) != str(candidate[field])
            for field in assignment_fields - {"rank_sha256"}
        ) or (
            str(row["source_id"]) != str(bank_row["source_id"])
            or str(row["source_phrase"]) != str(bank_row["source_phrase"])
            or str(row["source_head_lemma"]) != str(bank_row["head_lemma"])
        ):
            raise ValueError("specificity new-bank assignment is not source/candidate bound")
        if row["source_id"] in seen_sources or row["receiver_id"] in seen_receivers:
            raise ValueError("specificity new-bank assignment identities are not unique")
        seen_sources.add(row["source_id"])
        seen_receivers.add(row["receiver_id"])
    if set(new_rows) != {str(row["case_id"]) for row in assignment["assignments"]}:
        raise ValueError("specificity new-bank assignment does not cover exact candidate rows")

    if _load_exact_json(render_configuration_path, "specificity render config") != SPECIFICITY_RENDER_CONFIGURATION:
        raise ValueError("specificity render config differs from executable contract")
    secrets = _load_exact_json(secrets_path, "specificity Stage-0 secrets")
    if set(secrets) != {
        "schema",
        "evaluation_seed_namespace",
        "evaluation_seed_salt",
        "screening_seed",
        "screening_seed_namespace",
        "selector_salt",
    } or (
        secrets["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or secrets["evaluation_seed_namespace"] != "v4-specificity-evaluation-v2"
        or secrets["screening_seed_namespace"] != "v4-specificity-stage0-screening-v2"
        or not isinstance(secrets["screening_seed"], int)
        or isinstance(secrets["screening_seed"], bool)
        or not 3_500_000_000 <= secrets["screening_seed"] < 4_000_000_000
        or not re.fullmatch(r"[0-9a-f]{64}", str(secrets["selector_salt"]))
        or not re.fullmatch(r"[0-9a-f]{64}", str(secrets["evaluation_seed_salt"]))
        or secrets["selector_salt"] == secrets["evaluation_seed_salt"]
    ):
        raise ValueError("specificity secret schema/value domains are not exact")
    if (
        screening_seed_path.read_bytes() != f"{secrets['screening_seed']}\n".encode("ascii")
        or selector_salt_path.read_bytes() != f"{secrets['selector_salt']}\n".encode("ascii")
        or evaluation_seed_salt_path.read_bytes()
        != f"{secrets['evaluation_seed_salt']}\n".encode("ascii")
        or pending["screening_seed_commitment_sha256"]
        != _secret_commitment("specificity_screening_seed_v2", str(secrets["screening_seed"]))
        or pending["selector_salt_commitment_sha256"]
        != _secret_commitment("specificity_stage0_selector_salt_v2", secrets["selector_salt"])
        or pending["evaluation_seed_salt_commitment_sha256"]
        != _secret_commitment("specificity_evaluation_seed_salt_v2", secrets["evaluation_seed_salt"])
    ):
        raise ValueError("specificity seed/salt opening differs from public commitment")
    generation_spec = _load_exact_json(generation_spec_path, "specificity generation spec")
    runtime_ref = generation_spec.get("runtime_registry") if isinstance(generation_spec, dict) else None
    if set(generation_spec) != {
        "protocol", "status", "model_inventory_sha256", "runtime_registry", "generation_spec", "source_mode"
    } or (
        generation_spec["protocol"] != GENERATION_SPEC_PROTOCOL
        or generation_spec["status"] != "frozen_before_original_render"
        or generation_spec["source_mode"] != "Original_screening_then_matched_O_v3b_v4"
        or generation_spec["generation_spec"] != GENERATION_SPEC
        or generation_spec["model_inventory_sha256"] != FROZEN_MODEL_CONTENT_INVENTORY_SHA256
        or not isinstance(runtime_ref, dict)
        or runtime_ref.get("path") != RUNTIME_REGISTRY
        or set(runtime_ref) != {"path", "sha256"}
        or not is_sha256(runtime_ref["sha256"])
    ):
        raise ValueError("specificity generation spec differs from exact contract")
    validate_runtime_registry(resolve_path(project_root, RUNTIME_REGISTRY), runtime_ref["sha256"])
    forbidden = validate_forbidden_seed_inventory(
        forbidden_seed_inventory_path, dataset="specificity"
    )
    causal_seeds = {int(row["seed"]) for row in causal_units}
    if not causal_seeds <= forbidden:
        raise ValueError("specificity forbidden inventory does not contain all causal U seeds")
    if secrets["screening_seed"] in forbidden:
        raise ValueError("specificity screening seed occurs in forbidden inventory")
    seed_audit = audit_preselection_seed_space(
        normalized,
        dataset="specificity",
        private_salt=secrets["evaluation_seed_salt"],
        screening_seed=secrets["screening_seed"],
        forbidden_seeds=forbidden,
    )
    return expected_selection_binding(
        dataset="specificity",
        public_pending_sha256=file_sha256(pending_path),
        root_bundle_sha256=file_sha256(root_bundle_path),
        component_sha256=component_hashes,
        generation_spec_sha256=file_sha256(generation_spec_path),
        model_inventory_sha256=generation_spec["model_inventory_sha256"],
        runtime_registry_sha256=runtime_ref["sha256"],
        screening_seed_sha256=file_sha256(screening_seed_path),
        selector_salt_sha256=file_sha256(selector_salt_path),
        evaluation_seed_salt_sha256=file_sha256(evaluation_seed_salt_path),
        forbidden_seed_inventory_sha256=file_sha256(forbidden_seed_inventory_path),
        preselection_seed_audit_sha256=seed_audit["ordered_case_replicate_seed_sha256"],
        preselection_seed_count=seed_audit["derived_seed_count"],
    )


def validate_selection_contract_opening(
    project_root: Path,
    *,
    dataset: str,
    stage0_registry: Mapping[str, Any],
    private_root: Path,
    candidate_manifest_path: Path,
    canonical_templates_path: Path,
    field_rules_path: Path,
    render_configuration_path: Path,
    selection_rules_path: Path,
    secrets_path: Path,
    root_bundle_path: Path,
    generation_spec_path: Path,
    screening_seed_path: Path,
    selector_salt_path: Path,
    evaluation_seed_salt_path: Path,
    forbidden_seed_inventory_path: Path,
    selection_binding_path: Path,
    source_ontology_path: Path | None = None,
    source_split_path: Path | None = None,
    holdout_registry_path: Path | None = None,
    receiver_ontology_path: Path | None = None,
    new_bank_assignment_path: Path | None = None,
    causal_stage0_registry_path: Path | None = None,
    causal_stage1_registry_path: Path | None = None,
    causal_selected_path: Path | None = None,
    causal_unit_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Open the complete Stage-0 contract inside the isolated evaluator."""

    expected_binding = prepare_selection_binding(
        project_root,
        dataset=dataset,
        private_root=private_root,
        candidate_manifest_path=candidate_manifest_path,
        canonical_templates_path=canonical_templates_path,
        field_rules_path=field_rules_path,
        render_configuration_path=render_configuration_path,
        selection_rules_path=selection_rules_path,
        secrets_path=secrets_path,
        root_bundle_path=root_bundle_path,
        generation_spec_path=generation_spec_path,
        screening_seed_path=screening_seed_path,
        selector_salt_path=selector_salt_path,
        evaluation_seed_salt_path=evaluation_seed_salt_path,
        forbidden_seed_inventory_path=forbidden_seed_inventory_path,
        source_ontology_path=source_ontology_path,
        source_split_path=source_split_path,
        holdout_registry_path=holdout_registry_path,
        receiver_ontology_path=receiver_ontology_path,
        new_bank_assignment_path=new_bank_assignment_path,
        causal_stage0_registry_path=causal_stage0_registry_path,
        causal_stage1_registry_path=causal_stage1_registry_path,
        causal_selected_path=causal_selected_path,
        causal_unit_manifest_path=causal_unit_manifest_path,
    )
    _private_opening_path(selection_binding_path, private_root, "selection binding")
    binding = _load_exact_json(selection_binding_path, "selection binding")
    if binding != expected_binding:
        raise ValueError("selection binding is not the exact pre-screening executable contract")
    candidate_name = f"candidate_manifest_{CANDIDATE_COUNTS[dataset]}"
    artifacts = stage0_registry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage-0 selection artifact registry is missing")
    committed_paths = {
        candidate_name: candidate_manifest_path,
        "canonical_templates": canonical_templates_path,
        "field_normalization": field_rules_path,
        "raw_root_bundle": root_bundle_path,
        "raw_render_configuration": render_configuration_path,
        "stage0_secrets": secrets_path,
        "screening_seed": screening_seed_path,
        "screening_generation_spec": generation_spec_path,
        "selector_salt": selector_salt_path,
        "ranking_formula": selection_rules_path,
        "constrained_subset_algorithm": selection_rules_path,
        "evaluation_seed_salt": evaluation_seed_salt_path,
        "seed_derivation_formula": selection_binding_path,
        "forbidden_seed_inventory": forbidden_seed_inventory_path,
    }
    if dataset == "causal":
        if any(
            path is None
            for path in (
                source_ontology_path,
                source_split_path,
                holdout_registry_path,
                receiver_ontology_path,
            )
        ):
            raise ValueError("causal opening lacks required ontology artifacts")
        committed_paths.update(
            {
                "source_bank_registry_64": resolve_path(
                    project_root, PUBLIC_SOURCE_BANK
                ),
                "source_ontology_80": source_ontology_path,
                "source_split_80": source_split_path,
                "holdout_registry_24": holdout_registry_path,
                "receiver_ontology_32": receiver_ontology_path,
            }
        )
    if dataset == "specificity":
        if new_bank_assignment_path is None:
            raise ValueError("specificity opening lacks new-bank assignment")
        committed_paths["new_bank_selection_and_receiver_assignment"] = (
            new_bank_assignment_path
        )
    for name, path in committed_paths.items():
        commitment = artifacts.get(name)
        if not isinstance(commitment, Mapping) or (
            path.stat().st_size != commitment.get("size_bytes")
            or file_sha256(path) != commitment.get("sha256")
        ):
            raise ValueError(f"{name}: bytes differ from Stage-0 commitment")
    if artifacts["ranking_formula"] != artifacts["constrained_subset_algorithm"]:
        raise ValueError("ranking and constrained-subset commitments must bind one rules file")
    return binding


def load_normalized_candidate_manifest(
    path: Path,
    *,
    dataset: str,
    canonical_templates_path: Path,
) -> list[dict[str, Any]]:
    """Pure Stage-0 JSON -> screening/selector row projection."""

    if dataset == "specificity":
        payload = _load_exact_json(path, "private specificity candidate manifest")
        if set(payload) != {
            "schema",
            "dataset_version",
            "stage",
            "candidate_count",
            "causal_stage1_registry_sha256",
            "candidates",
        }:
            raise ValueError("specificity candidate-manifest top-level fields are not exact")
        if (
            payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
            or payload["dataset_version"] != DATASET_VERSION
            or payload["stage"] != 0
            or payload["candidate_count"] != 36
            or not is_sha256(payload["causal_stage1_registry_sha256"])
            or not isinstance(payload["candidates"], list)
            or len(payload["candidates"]) != 36
        ):
            raise ValueError("specificity candidate-manifest identity/count differs from protocol")
        templates = _load_exact_json(canonical_templates_path, "specificity templates")
        if templates != {
            "schema": SOURCE_SLOT_REGISTRY_SCHEMA,
            "prompt_templates": SPECIFICITY_CANONICAL_TEMPLATES,
            "template_fill_rules": SPECIFICITY_TEMPLATE_FILL_RULES,
            "non_substitution_rule": SPECIFICITY_TEMPLATE_NON_SUBSTITUTION_RULE,
        }:
            raise ValueError("specificity templates differ from the exact noncausal contract")
        required = {
            "case_id",
            "membership",
            "prompt_variant",
            "source_id",
            "source_phrase",
            "source_head_lemma",
            "receiver_id",
            "receiver_phrase",
            "causal_case_id",
            "template_id",
            "canonical_prompt",
            "canonical_record_sha256",
        }
        output: list[dict[str, Any]] = []
        for row in payload["candidates"]:
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("specificity candidate record fields are not exact")
            canonical = dict(row)
            digest = canonical.pop("canonical_record_sha256")
            canonical_bytes = (
                json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            if hashlib.sha256(canonical_bytes).hexdigest() != digest:
                raise ValueError("specificity candidate canonical record hash mismatch")
            if any(
                not isinstance(row[field], str) or "\x00" in row[field]
                for field in required
            ):
                raise ValueError("specificity candidate fields must be NUL-free strings")
            membership = row["membership"]
            variant = row["prompt_variant"]
            if membership not in SPECIFICITY_MEMBERSHIPS or variant not in PROMPT_VARIANTS:
                raise ValueError("specificity candidate cell differs from protocol")
            if row["template_id"] != variant:
                raise ValueError("specificity template ID differs from prompt variant")
            if membership == "new_bank_source":
                if row["causal_case_id"]:
                    raise ValueError("new-bank specificity case must not claim a causal case")
            elif not row["causal_case_id"]:
                raise ValueError("matched specificity case lacks causal case identity")
            filled_source = (
                row["source_phrase"].capitalize()
                if variant == "direct"
                else row["source_phrase"]
            )
            expected_prompt = SPECIFICITY_CANONICAL_TEMPLATES[variant].format(
                source_phrase=filled_source,
                receiver_phrase=row["receiver_phrase"],
            )
            if row["canonical_prompt"] != expected_prompt:
                raise ValueError("specificity prompt differs from structured canonical template")
            output.append(
                {
                    **dict(row),
                    "candidate_id": row["case_id"],
                    "specificity_case_id": row["case_id"],
                    "receiver": row["receiver_phrase"],
                    "prompt": row["canonical_prompt"],
                }
            )
        if len({row["candidate_id"] for row in output}) != 36:
            raise ValueError("specificity candidate IDs are not unique")
        expected_cells = {
            ("original_source", "direct"): 4,
            ("original_source", "natural"): 4,
            ("new_bank_source", "direct"): 6,
            ("new_bank_source", "natural"): 6,
            ("holdout_source", "direct"): 8,
            ("holdout_source", "natural"): 8,
        }
        if Counter((row["membership"], row["prompt_variant"]) for row in output) != Counter(
            expected_cells
        ):
            raise ValueError("specificity candidate cell inventory differs from protocol")
        return output
    if dataset != "causal":
        raise ValueError("unsupported candidate projection dataset")
    payload = _load_exact_json(path, "private causal candidate manifest")
    if set(payload) != {
        "schema",
        "protocol",
        "dataset_version",
        "stage",
        "candidate_count",
        "candidates",
    }:
        raise ValueError("causal candidate-manifest top-level fields are not exact")
    if (
        payload["schema"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or payload["protocol"] != SOURCE_SLOT_REGISTRY_SCHEMA
        or payload["dataset_version"] != DATASET_VERSION
        or payload["stage"] != 0
        or payload["candidate_count"] != 48
        or not isinstance(payload["candidates"], list)
        or len(payload["candidates"]) != 48
    ):
        raise ValueError("causal candidate-manifest identity/count differs from protocol")
    templates = _load_exact_json(canonical_templates_path, "canonical templates")
    if (
        templates.get("prompt_templates") != CAUSAL_CANONICAL_TEMPLATES
        or templates.get("template_fill_rules") != CAUSAL_TEMPLATE_FILL_RULES
        or templates.get("non_substitution_rule") != CAUSAL_TEMPLATE_NON_SUBSTITUTION_RULE
    ):
        raise ValueError("causal candidate projection requires exact canonical templates")
    required = {
        "case_id",
        "group",
        "prompt_variant",
        "source_membership",
        "source_id",
        "source_phrase",
        "source_head_lemma",
        "source_physical_audit_status",
        "receiver_membership",
        "receiver_id",
        "receiver_phrase",
        "canonical_prompt",
        "canonical_record_sha256",
    }
    output: list[dict[str, Any]] = []
    for row in payload["candidates"]:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("causal candidate record fields are not exact")
        canonical_record = dict(row)
        record_sha256 = canonical_record.pop("canonical_record_sha256")
        canonical_bytes = (
            json.dumps(
                canonical_record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        if hashlib.sha256(canonical_bytes).hexdigest() != record_sha256:
            raise ValueError("candidate canonical record hash mismatch")
        variant = str(row["prompt_variant"])
        if variant not in PROMPT_VARIANTS:
            raise ValueError("candidate prompt variant differs from protocol")
        group = str(row["group"])
        if group not in CAUSAL_GROUPS:
            raise ValueError("candidate group differs from protocol")
        expected_source_membership = (
            "original_source"
            if group == "seen_source_new_receiver"
            else "holdout_source"
        )
        expected_receiver_membership = (
            "seen_receiver"
            if group == "holdout_source_seen_receiver"
            else "new_receiver"
        )
        if (
            row["source_membership"] != expected_source_membership
            or row["receiver_membership"] != expected_receiver_membership
            or row["source_physical_audit_status"]
            != (
                "legacy_original_source_exempt"
                if expected_source_membership == "original_source"
                else "strict_physical_pass_v2"
            )
        ):
            raise ValueError("candidate group/membership fields are inconsistent")
        for field in (
            "case_id",
            "source_id",
            "source_phrase",
            "source_head_lemma",
            "source_physical_audit_status",
            "receiver_id",
            "receiver_phrase",
            "canonical_prompt",
            "canonical_record_sha256",
        ):
            value = row[field]
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"candidate {field} must be a nonempty NUL-free string")
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", row["source_head_lemma"]):
            raise ValueError("candidate source head lemma is not normalized")
        source = str(row["source_phrase"])
        receiver = str(row["receiver_phrase"])
        filled_source = source.capitalize() if variant == "direct" else source
        canonical_prompt = CAUSAL_CANONICAL_TEMPLATES[variant].format(
            source_phrase=filled_source,
            receiver_phrase=receiver,
        )
        if str(row["canonical_prompt"]) != canonical_prompt:
            raise ValueError("candidate prompt differs from structured canonical construction")
        case_id = str(row["case_id"])
        if not case_id:
            raise ValueError("candidate case_id is blank")
        output.append(
            {
                **dict(row),
                "candidate_id": case_id,
                "semantic_case_id": case_id,
                "receiver": receiver,
                "prompt": canonical_prompt,
            }
        )
    if len({row["candidate_id"] for row in output}) != 48:
        raise ValueError("candidate case IDs are not unique")
    expected_cells = {
        (group, variant): 8 for group in CAUSAL_GROUPS for variant in PROMPT_VARIANTS
    }
    if Counter((row["group"], row["prompt_variant"]) for row in output) != Counter(
        expected_cells
    ):
        raise ValueError("causal candidate cell inventory differs from protocol")
    return output


def validate_commitment_opening(
    project_root: Path,
    opening_path: Path,
    *,
    dataset: str,
    stage0_path: Path,
    stage1_path: Path,
    private_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Open both stages inside an evaluator-only root and recompute every digest."""

    if not private_root.is_dir() or private_root.is_symlink():
        raise ValueError("private evaluator root must be a real directory")
    private_resolved = private_root.resolve(strict=True)
    if not opening_path.is_file() or opening_path.is_symlink():
        raise FileNotFoundError(f"{dataset}: commitment opening is missing")
    try:
        opening_path.resolve(strict=True).relative_to(private_resolved)
    except ValueError as exc:
        raise ValueError(f"{dataset}: commitment opening escapes evaluator-only root") from exc
    stage0 = validate_commitment_registry(stage0_path, dataset=dataset, stage=0)
    stage0_sha = file_sha256(stage0_path)
    stage1 = validate_commitment_registry(
        stage1_path, dataset=dataset, stage=1, expected_stage0_sha256=stage0_sha
    )
    payload = _load_exact_json(opening_path, f"{dataset} commitment opening")
    if set(payload) != {
        "protocol",
        "dataset",
        "dataset_version",
        "stage0_registry_sha256",
        "stage1_registry_sha256",
        "artifacts",
    }:
        raise ValueError(f"{dataset}: opening fields are not exact")
    if (
        payload["protocol"] != OPENING_PROTOCOL
        or payload["dataset"] != dataset
        or payload["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError(f"{dataset}: opening protocol mismatch")
    if payload["stage0_registry_sha256"] != stage0_sha:
        raise ValueError(f"{dataset}: opening Stage-0 hash mismatch")
    if payload["stage1_registry_sha256"] != file_sha256(stage1_path):
        raise ValueError(f"{dataset}: opening Stage-1 hash mismatch")
    combined = {
        **{name: record for name, record in stage0["artifacts"].items()},
        **{name: record for name, record in stage1["artifacts"].items()},
    }
    if not isinstance(payload["artifacts"], dict) or set(payload["artifacts"]) != set(combined):
        raise ValueError(f"{dataset}: opening artifact inventory is not exact")
    paths: dict[str, Path] = {}
    for name, opened in payload["artifacts"].items():
        if not isinstance(opened, dict) or set(opened) != {"path", "sha256", "size_bytes", "row_count"}:
            raise ValueError(f"{dataset}/{name}: opening record is not exact")
        path = resolve_path(project_root, str(opened["path"]))
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{dataset}/{name}: opened artifact is missing")
        resolved = path.resolve(strict=True)
        if dataset == "causal" and name == "source_bank_registry_64":
            expected_public = resolve_path(project_root, PUBLIC_SOURCE_BANK).resolve(
                strict=True
            )
            if resolved != expected_public:
                raise ValueError(
                    "causal/source_bank_registry_64 must open the exact canonical public path"
                )
        else:
            try:
                resolved.relative_to(private_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"{dataset}/{name}: artifact escapes evaluator-only root"
                ) from exc
        committed = combined[name]
        actual = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": _structured_row_count(path, int(opened["row_count"]))
            if opened["row_count"] is not None
            else None,
        }
        if {key: opened[key] for key in actual} != actual or actual != committed:
            raise ValueError(f"{dataset}/{name}: opened bytes do not match public commitment")
        paths[name] = path
    return payload, paths


def _structured_row_count(path: Path, expected: int) -> int:
    if path.suffix.lower() == ".csv":
        return len(read_csv(path))
    if path.suffix.lower() != ".json":
        raise ValueError(f"row_count commitment requires CSV/JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        preferred = (
            "candidates",
            "rows",
            "reviews",
            "eligibility",
            "selected_cases",
            "units",
            "mapping",
            "entries",
        )
        matches = [key for key in preferred if isinstance(payload.get(key), list)]
        if len(matches) == 1:
            return len(payload[matches[0]])
        exact = [key for key in matches if len(payload[key]) == expected]
        if len(exact) == 1:
            return len(payload[exact[0]])
    raise ValueError(f"cannot identify one committed row inventory in JSON artifact: {path}")


def validate_gate_registry(path: Path, expected_gate_spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_exact_json(path, "v4 machine gate registry")
    if set(payload) != {
        "protocol",
        "status",
        "dataset_version",
        "sealed_final36_status",
        "gate_spec",
        "gate_spec_sha256",
        "scorer_sha256",
    }:
        raise ValueError("machine gate registry fields are not exact")
    if (
        payload["protocol"] != GATE_REGISTRY_PROTOCOL
        or payload["status"] != "frozen"
        or payload["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError("machine gate registry is not frozen")
    if payload["sealed_final36_status"] != "unopened":
        raise ValueError("machine gate registry attempts to open sealed-final36")
    _require_sha256(payload["scorer_sha256"], "scorer_sha256")
    expected = json.loads(json.dumps(expected_gate_spec))
    if payload["gate_spec"] != expected:
        raise ValueError("machine gate registry differs from executable gate spec")
    if payload["gate_spec_sha256"] != canonical_json_sha256(expected):
        raise ValueError("machine gate spec digest mismatch")
    if _contains_placeholder(payload):
        raise ValueError("machine gate registry contains a placeholder")
    return payload


def validate_runtime_registry(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Validate the public runtime contract without importing the GPU runtime.

    Evaluation provenance needs the exact frozen registry bytes, but an
    isolated scorer is not required to load CUDA.  The training launcher and
    trainer separately validate the live interpreter against the same file.
    """

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("v4 runtime registry is missing or symlinked")
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "runtime registry")
        if file_sha256(path) != expected_sha256:
            raise ValueError("v4 runtime registry byte hash mismatch")
    payload = _load_exact_json(path, "v4 runtime registry")
    if payload != RUNTIME_REGISTRY_PAYLOAD:
        raise ValueError("v4 runtime registry differs from the exact runtime contract")
    return payload


def validate_training_code_registry(project_root: Path, path: Path) -> dict[str, Any]:
    payload = _load_exact_json(path, "v4 training code registry")
    if set(payload) != {"protocol", "status", "runtime_registry", "artifacts"}:
        raise ValueError("training code registry fields are not exact")
    if payload["protocol"] != TRAINING_CODE_REGISTRY_PROTOCOL or payload["status"] != "frozen":
        raise ValueError("training code registry is not frozen")
    runtime = payload["runtime_registry"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"path", "sha256"}
        or runtime["path"] != RUNTIME_REGISTRY
    ):
        raise ValueError("training code registry runtime reference is not exact")
    _require_sha256(runtime["sha256"], "training code registry/runtime registry")
    validate_runtime_registry(
        resolve_path(project_root, RUNTIME_REGISTRY), runtime["sha256"]
    )
    if not isinstance(payload["artifacts"], dict) or set(payload["artifacts"]) != set(TRAINING_CODE_ARTIFACTS):
        raise ValueError("training code registry artifact inventory is not exact")
    for name, registered_path in TRAINING_CODE_ARTIFACTS.items():
        record = payload["artifacts"][name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"training code registry/{name}: record is not exact")
        if record["path"] != registered_path:
            raise ValueError(f"training code registry/{name}: path differs from protocol")
        _require_sha256(record["sha256"], f"training code registry/{name}")
        artifact = resolve_path(project_root, registered_path)
        if not artifact.is_file() or artifact.is_symlink() or file_sha256(artifact) != record["sha256"]:
            raise ValueError(f"training code registry/{name}: byte hash mismatch")
    if _contains_placeholder(payload):
        raise ValueError("training code registry contains a placeholder")
    return payload


def validate_training_authorization(
    project_root: Path,
    *,
    expected_gate_spec: Mapping[str, Any],
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Require all four data stages and the exact frozen machine gate."""

    path = authorization_path or resolve_path(project_root, TRAINING_AUTHORIZATION)
    payload = _load_exact_json(path, "v4 training authorization")
    if set(payload) != {
        "protocol",
        "status",
        "dataset_version",
        "sealed_final36_status",
        *TRAINING_AUTHORIZATION_REFS,
    }:
        raise ValueError("training authorization fields are not exact")
    if payload["protocol"] != TRAINING_AUTHORIZATION_PROTOCOL or payload["status"] != "authorized":
        raise ValueError("v4 training is not authorized")
    if payload["dataset_version"] != DATASET_VERSION:
        raise ValueError("training authorization dataset version differs from protocol")
    if payload["sealed_final36_status"] != "unopened":
        raise ValueError("training authorization must keep sealed-final36 unopened")
    resolved_refs: dict[str, Path] = {}
    for name, registered_path in TRAINING_AUTHORIZATION_REFS.items():
        record = payload[name]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"training authorization/{name}: reference is not exact")
        if record["path"] != registered_path:
            raise ValueError(f"training authorization/{name}: path differs from protocol")
        _require_sha256(record["sha256"], f"training authorization/{name}")
        ref_path = resolve_path(project_root, registered_path)
        if not ref_path.is_file() or ref_path.is_symlink():
            raise FileNotFoundError(f"training authorization/{name}: frozen artifact is missing")
        if file_sha256(ref_path) != record["sha256"]:
            raise ValueError(f"training authorization/{name}: hash mismatch")
        resolved_refs[name] = ref_path
    frozen_public_hashes = {
        "source_bank_registry": FROZEN_PUBLIC_SOURCE_BANK_SHA256,
        "holdout_public_commitment": FROZEN_PUBLIC_HOLDOUT_COMMITMENT_SHA256,
    }
    for name, expected_sha256 in frozen_public_hashes.items():
        if payload[name]["sha256"] != expected_sha256:
            label = (
                "public source bank"
                if name == "source_bank_registry"
                else "public holdout commitment"
            )
            raise ValueError(
                f"training authorization {label} differs from the canonical v2 bytes"
            )
    causal0 = validate_commitment_registry(
        resolved_refs["causal_stage0"], dataset="causal", stage=0
    )
    validate_commitment_registry(
        resolved_refs["causal_stage1"],
        dataset="causal",
        stage=1,
        expected_stage0_sha256=file_sha256(resolved_refs["causal_stage0"]),
    )
    specificity0 = validate_commitment_registry(
        resolved_refs["specificity_stage0"], dataset="specificity", stage=0
    )
    validate_commitment_registry(
        resolved_refs["specificity_stage1"],
        dataset="specificity",
        stage=1,
        expected_stage0_sha256=file_sha256(resolved_refs["specificity_stage0"]),
    )
    bank_record = causal0["artifacts"]["source_bank_registry_64"]
    if (
        bank_record["sha256"] != payload["source_bank_registry"]["sha256"]
        or bank_record["size_bytes"]
        != resolved_refs["source_bank_registry"].stat().st_size
    ):
        raise ValueError(
            "training authorization public source bank differs from causal Stage-0"
        )
    holdout_public = _load_exact_json(
        resolved_refs["holdout_public_commitment"],
        "training-authorized public holdout commitment",
    )
    holdout_registry_sha = holdout_public.get("holdout_registry_file_sha256")
    _require_sha256(
        holdout_registry_sha,
        "training authorization/public holdout private-registry commitment",
    )
    if holdout_registry_sha != causal0["artifacts"]["holdout_registry_24"]["sha256"]:
        raise ValueError(
            "training authorization public holdout differs from causal Stage-0"
        )
    del causal0, specificity0
    validate_gate_registry(resolved_refs["gate_registry"], expected_gate_spec)
    validate_runtime_registry(
        resolved_refs["runtime_registry"], payload["runtime_registry"]["sha256"]
    )
    code = validate_training_code_registry(project_root, resolved_refs["code_registry"])
    if code["runtime_registry"] != payload["runtime_registry"]:
        raise ValueError("authorization/code registry runtime references differ")
    if _contains_placeholder(payload):
        raise ValueError("training authorization contains a placeholder")
    return payload


CHECKPOINT_HASH_FIELDS = (
    "model_content_inventory_sha256",
    "transformer_inventory_sha256",
    "source_bank_registry_sha256",
    "source_mapping_registry_sha256",
    "active100_mapping_sha256",
    "full178_mapping_sha256",
    "train_manifest_sha256",
    "base_cache_inventory_sha256",
    "teacher_cache_inventory_sha256",
    "prompt_sidecar_inventory_sha256",
    "sample_order_sha256",
    "noise_sigma_rng_initial_sha256",
    "noise_sigma_rng_final_sha256",
    "initial_lora_sha256",
    "trainer_sha256",
    "launcher_sha256",
    "training_authorization_sha256",
    "training_code_registry_sha256",
    "runtime_registry_sha256",
)

TRAINING_STATE_PROTOCOL = "water_impact_dynamic_v4_source_slot_randomized_teacher_v2"
SCALE_SANITY_PROTOCOL = "water_impact_dynamic_v4_source_slot_scale_sanity_v2"
FINAL_LORA_FINITE_PROTOCOL = "water_impact_dynamic_v4_final_lora_finite_check_v2"
NULL_PREFLIGHT_PROTOCOL = "water_impact_dynamic_v4_null_sidecar_preflight_v2"
TRAINING_MANIFEST = "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
BASE_CACHE_DIR = "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
TEACHER_CACHE_DIR = "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
SOURCE_MAPPING_REGISTRY = f"{DATA_ROOT}/source_mapping_v2.json"
CANONICAL_PROMPT_BUILDER = "scripts/build_water_impact_dynamic_pairs_v1.py"
EXPECTED_TRAINING_CONFIG: dict[str, Any] = {
    "model": GENERATION_SPEC["model"],
    "height": 480,
    "width": 832,
    "num_frames": 49,
    "max_steps": 200,
    "learning_rate": 5e-5,
    "rank": 16,
    "alpha": 16,
    "grad_accum": 1,
    "save_every": 200,
    "seed": 26000,
    "device": "cuda",
    "role": "all",
    "objective": "source_slot_target_prompt_teacher",
    "balanced_roles": True,
    "preserve_weight": 4.0,
    "target_prompt_teacher_weight": 4.0,
    "target_prompt_calibration_id": (
        "v4_retain_v3b_lambda4_first16_output_gradient_v1"
    ),
    "sanity_mean_min": 0.2,
    "sanity_mean_max": 0.5,
    "sanity_single_max": 1.0,
}
ONLY_TRAINING_INTERVENTION = (
    "erase factual prompt_embeds replaced by registered augmented source-slot sidecar"
)


def _finite_number(value: Any, *, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        raise ValueError("numeric provenance value is non-finite or outside its domain")
    return float(value)


def validate_final_lora_finite_check(value: Any) -> dict[str, Any]:
    expected_fields = {
        "protocol",
        "status",
        "digest_algorithm",
        "trainable_parameter_count",
        "trainable_element_count",
        "nonfinite_trainable_parameter_count",
        "nonfinite_trainable_element_count",
        "lora_state_tensor_count",
        "lora_state_element_count",
        "nonfinite_lora_state_tensor_count",
        "nonfinite_lora_state_element_count",
        "trainable_state_sha256",
        "lora_state_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("final LoRA finite-check fields are not exact")
    if (
        value["protocol"] != FINAL_LORA_FINITE_PROTOCOL
        or value["status"] != "passed"
        or value["digest_algorithm"]
        != "sha256_sorted_name_shape_dtype_raw_bytes_v1"
    ):
        raise ValueError("final LoRA finite check did not pass the frozen protocol")
    for field in (
        "trainable_parameter_count",
        "trainable_element_count",
        "lora_state_tensor_count",
        "lora_state_element_count",
    ):
        if not isinstance(value[field], int) or isinstance(value[field], bool) or value[field] <= 0:
            raise ValueError(f"final LoRA finite check/{field} must be positive")
    for field in (
        "nonfinite_trainable_parameter_count",
        "nonfinite_trainable_element_count",
        "nonfinite_lora_state_tensor_count",
        "nonfinite_lora_state_element_count",
    ):
        if value[field] != 0 or isinstance(value[field], bool):
            raise ValueError(f"final LoRA finite check/{field} must be integer zero")
    _require_sha256(value["trainable_state_sha256"], "final trainable state")
    _require_sha256(value["lora_state_sha256"], "final LoRA state")
    if _contains_placeholder(value):
        raise ValueError("final LoRA finite check contains a placeholder")
    return value


def validate_safetensors_finite_inventory(
    path: Path, *, expected_evidence: Mapping[str, Any]
) -> dict[str, int]:
    """Independently parse the saved safetensors file and reject NaN/Inf values."""

    evidence = validate_final_lora_finite_check(dict(expected_evidence))
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("checkpoint LoRA safetensors is missing or symlinked")
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("checkpoint LoRA safetensors header is truncated")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 1 or header_length > 16 * 1024 * 1024 or 8 + header_length > size:
            raise ValueError("checkpoint LoRA safetensors header length is invalid")
        header_bytes = handle.read(header_length)
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint LoRA safetensors header is invalid") from exc
        if not isinstance(header, dict) or not header:
            raise ValueError("checkpoint LoRA safetensors tensor inventory is empty")
        records: list[tuple[int, int, str, int]] = []
        tensor_count = element_count = 0
        dtype_sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
        for name, record in header.items():
            if name == "__metadata__":
                if not isinstance(record, dict) or any(
                    not isinstance(key, str) or not isinstance(item, str)
                    for key, item in record.items()
                ):
                    raise ValueError("checkpoint LoRA safetensors metadata is invalid")
                continue
            if not isinstance(name, str) or not name or not isinstance(record, dict) or set(record) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError("checkpoint LoRA safetensors tensor record is not exact")
            dtype = record["dtype"]
            shape = record["shape"]
            offsets = record["data_offsets"]
            if (
                dtype not in dtype_sizes
                or not isinstance(shape, list)
                or not shape
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    for dimension in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or offset < 0
                    for offset in offsets
                )
                or offsets[1] <= offsets[0]
            ):
                raise ValueError("checkpoint LoRA safetensors tensor domain is invalid")
            elements = math.prod(shape)
            if offsets[1] - offsets[0] != elements * dtype_sizes[dtype]:
                raise ValueError("checkpoint LoRA safetensors tensor byte length mismatch")
            tensor_count += 1
            element_count += elements
            records.append((offsets[0], offsets[1], dtype, elements))
        if tensor_count == 0:
            raise ValueError("checkpoint LoRA safetensors has no tensors")
        records.sort()
        expected_start = 0
        for start, end, _, _ in records:
            if start != expected_start:
                raise ValueError("checkpoint LoRA safetensors offsets have a gap or overlap")
            expected_start = end
        data_start = 8 + header_length
        if data_start + expected_start != size:
            raise ValueError("checkpoint LoRA safetensors has unregistered trailing bytes")
        masks = {
            "F16": ("<H", 0x7C00),
            "BF16": ("<H", 0x7F80),
            "F32": ("<I", 0x7F800000),
            "F64": ("<Q", 0x7FF0000000000000),
        }
        for start, end, dtype, _ in records:
            handle.seek(data_start + start)
            remaining = end - start
            item_size = dtype_sizes[dtype]
            unpack_format, exponent_mask = masks[dtype]
            while remaining:
                read_size = min(8 * 1024 * 1024, remaining)
                read_size -= read_size % item_size
                chunk = handle.read(read_size)
                if len(chunk) != read_size:
                    raise ValueError("checkpoint LoRA safetensors data is truncated")
                if any(
                    (bits & exponent_mask) == exponent_mask
                    for (bits,) in struct.iter_unpack(unpack_format, chunk)
                ):
                    raise ValueError("checkpoint LoRA safetensors contains NaN or Inf")
                remaining -= read_size
    if (
        tensor_count != evidence["lora_state_tensor_count"]
        or element_count != evidence["lora_state_element_count"]
    ):
        raise ValueError("saved LoRA tensor inventory differs from finite-check evidence")
    return {"tensor_count": tensor_count, "element_count": element_count}


def validate_scale_sanity_payload(
    payload: Any, *, expected_run_registration_sha256: str
) -> dict[str, Any]:
    fields = {
        "protocol",
        "status",
        "dataset_version",
        "passed",
        "run_registration_sha256",
        "calibration_id",
        "formula",
        "aggregation",
        "weight",
        "mean_min",
        "mean_max",
        "single_max",
        "observation_count",
        "mean_raw_loss_ratio",
        "mean_weighted_loss_ratio",
        "mean_weighted_output_grad_ratio",
        "median_weighted_output_grad_ratio",
        "max_weighted_output_grad_ratio",
        "observations",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("scale sanity fields are not exact")
    if (
        payload["protocol"] != SCALE_SANITY_PROTOCOL
        or payload["status"] != "passed"
        or payload["dataset_version"] != DATASET_VERSION
        or payload["passed"] is not True
        or payload["run_registration_sha256"] != expected_run_registration_sha256
        or payload["calibration_id"]
        != EXPECTED_TRAINING_CONFIG["target_prompt_calibration_id"]
        or payload["formula"]
        != "g_i = 4 * sqrt(target_prompt_teacher_loss / flow_loss)"
        or payload["aggregation"]
        != "arithmetic_mean_over_first_16_actual_erase_updates"
        or payload["weight"] != 4.0
        or payload["mean_min"] != 0.2
        or payload["mean_max"] != 0.5
        or payload["single_max"] != 1.0
        or payload["observation_count"] != 16
    ):
        raise ValueError("scale sanity did not pass the exact registered gate")
    observations = payload["observations"]
    observation_fields = {
        "global_step",
        "erase_ordinal",
        "manifest_index",
        "scene_id",
        "assigned_source_id",
        "flow_loss",
        "target_prompt_teacher_loss",
        "raw_loss_ratio",
        "weighted_output_gradient_norm_ratio",
    }
    if not isinstance(observations, list) or len(observations) != 16:
        raise ValueError("scale sanity requires exactly 16 observations")
    raw_values: list[float] = []
    gradient_values: list[float] = []
    steps: list[int] = []
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or set(row) != observation_fields:
            raise ValueError("scale sanity observation fields are not exact")
        if (
            not isinstance(row["global_step"], int)
            or isinstance(row["global_step"], bool)
            or not 1 <= row["global_step"] <= 200
            or row["erase_ordinal"] != index
            or isinstance(row["erase_ordinal"], bool)
            or not isinstance(row["manifest_index"], int)
            or isinstance(row["manifest_index"], bool)
            or row["manifest_index"] < 0
            or not isinstance(row["scene_id"], str)
            or not row["scene_id"]
            or not isinstance(row["assigned_source_id"], str)
            or not row["assigned_source_id"]
        ):
            raise ValueError("scale sanity observation identity is invalid")
        flow = _finite_number(row["flow_loss"], positive=True)
        teacher = _finite_number(row["target_prompt_teacher_loss"])
        if teacher < 0:
            raise ValueError("scale sanity target-teacher loss is negative")
        raw = _finite_number(row["raw_loss_ratio"])
        gradient = _finite_number(row["weighted_output_gradient_norm_ratio"])
        if raw < 0 or gradient < 0 or not math.isclose(raw, teacher / flow, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("scale sanity raw loss ratio does not recompute")
        if not math.isclose(gradient, 4.0 * math.sqrt(raw), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("scale sanity output-gradient ratio does not recompute")
        steps.append(row["global_step"])
        raw_values.append(raw)
        gradient_values.append(gradient)
    if steps != sorted(set(steps)):
        raise ValueError("scale sanity observation steps are not strictly increasing")
    sorted_gradients = sorted(gradient_values)
    recomputed = {
        "mean_raw_loss_ratio": sum(raw_values) / 16,
        "mean_weighted_loss_ratio": 4.0 * sum(raw_values) / 16,
        "mean_weighted_output_grad_ratio": sum(gradient_values) / 16,
        "median_weighted_output_grad_ratio": (
            sorted_gradients[7] + sorted_gradients[8]
        )
        / 2,
        "max_weighted_output_grad_ratio": max(gradient_values),
    }
    for field, expected in recomputed.items():
        actual = _finite_number(payload[field])
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"scale sanity {field} does not recompute")
    if not (
        0.2 <= recomputed["mean_weighted_output_grad_ratio"] <= 0.5
        and recomputed["max_weighted_output_grad_ratio"] <= 1.0
    ):
        raise ValueError("scale sanity registered bounds did not pass")
    if _contains_placeholder(payload):
        raise ValueError("scale sanity contains a placeholder")
    return payload


def validate_run_registration_payload(
    project_root: Path,
    registration: Any,
    *,
    eligibility: Mapping[str, Any],
    authorization: Mapping[str, Any],
    code_registry: Mapping[str, Any],
) -> dict[str, Any]:
    base_fields = {
        "protocol",
        "status",
        "dataset_version",
        "created_utc",
        "output_dir",
        "only_training_intervention",
        "train_manifest_path",
        "train_manifest_sha256",
        "base_cache_path",
        "base_cache_inventory_sha256",
        "teacher_cache_path",
        "teacher_cache_inventory_sha256",
        "source_bank_registry_path",
        "source_bank_registry_sha256",
        "holdout_public_commitment_path",
        "holdout_public_commitment_sha256",
        "holdout_count",
        "source_mapping_registry_path",
        "source_mapping_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "prompt_sidecar_path",
        "prompt_sidecar_inventory_sha256",
        "prompt_sidecar_manifest_sha256",
        "model_content_inventory_sha256",
        "transformer_inventory_sha256",
        "preflight_artifact_path",
        "preflight_artifact_sha256",
        "training_authorization_path",
        "training_authorization_sha256",
        "training_code_registry_path",
        "training_code_registry_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "authorization_source",
        "git_commit",
        "git_upstream",
        "expected_initial_lora_sha256",
        "expected_noise_sigma_rng_initial_sha256",
        "expected_noise_sigma_rng_final_sha256",
        "expected_sample_order_sha256",
        "training_config",
    }
    code_fields = {
        field
        for name in TRAINING_CODE_ARTIFACTS
        for field in (f"{name}_path", f"{name}_sha256")
    }
    if not isinstance(registration, dict) or set(registration) != base_fields | code_fields:
        raise ValueError("checkpoint run registration fields are not exact")
    expected_literals = {
        "protocol": TRAINING_STATE_PROTOCOL,
        "status": "registered",
        "dataset_version": DATASET_VERSION,
        "output_dir": V4_OUTPUT_DIR,
        "only_training_intervention": ONLY_TRAINING_INTERVENTION,
        "train_manifest_path": TRAINING_MANIFEST,
        "base_cache_path": BASE_CACHE_DIR,
        "teacher_cache_path": TEACHER_CACHE_DIR,
        "source_bank_registry_path": PUBLIC_SOURCE_BANK,
        "holdout_public_commitment_path": PUBLIC_HOLDOUT_COMMITMENT,
        "holdout_count": 24,
        "source_mapping_registry_path": SOURCE_MAPPING_REGISTRY,
        "canonical_prompt_builder_path": CANONICAL_PROMPT_BUILDER,
        "prompt_sidecar_path": PROMPT_SIDECAR_DIR,
        "preflight_artifact_path": NULL_SIDECAR_PREFLIGHT,
        "training_authorization_path": TRAINING_AUTHORIZATION,
        "training_code_registry_path": TRAINING_CODE_REGISTRY,
        "runtime_registry_path": RUNTIME_REGISTRY,
        "authorization_source": "independent_audited_committed_and_pushed",
        "training_config": EXPECTED_TRAINING_CONFIG,
    }
    if any(registration.get(name) != expected for name, expected in expected_literals.items()):
        raise ValueError("checkpoint run registration identity/configuration mismatch")
    if not isinstance(registration["created_utc"], str) or not registration["created_utc"]:
        raise ValueError("checkpoint run registration created_utc is blank")
    git_commit = registration["git_commit"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(character not in HEX64 for character in git_commit)
        or not isinstance(registration["git_upstream"], str)
        or not registration["git_upstream"]
    ):
        raise ValueError("checkpoint run registration git provenance is invalid")
    direct_hashes = {
        "train_manifest_sha256": "train_manifest_sha256",
        "base_cache_inventory_sha256": "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256": "teacher_cache_inventory_sha256",
        "source_bank_registry_sha256": "source_bank_registry_sha256",
        "source_mapping_registry_sha256": "source_mapping_registry_sha256",
        "active100_mapping_sha256": "active100_mapping_sha256",
        "full178_mapping_sha256": "full178_mapping_sha256",
        "prompt_sidecar_inventory_sha256": "prompt_sidecar_inventory_sha256",
        "model_content_inventory_sha256": "model_content_inventory_sha256",
        "transformer_inventory_sha256": "transformer_inventory_sha256",
        "preflight_artifact_sha256": None,
        "training_authorization_sha256": "training_authorization_sha256",
        "training_code_registry_sha256": "training_code_registry_sha256",
        "runtime_registry_sha256": "runtime_registry_sha256",
        "expected_initial_lora_sha256": "initial_lora_sha256",
        "expected_noise_sigma_rng_initial_sha256": "noise_sigma_rng_initial_sha256",
        "expected_noise_sigma_rng_final_sha256": "noise_sigma_rng_final_sha256",
        "expected_sample_order_sha256": "sample_order_sha256",
    }
    for registration_field, eligibility_field in direct_hashes.items():
        expected = (
            eligibility["preflight"]["sha256"]
            if registration_field == "preflight_artifact_sha256"
            else eligibility[eligibility_field]
        )
        if registration[registration_field] != expected:
            raise ValueError(
                f"checkpoint run registration {registration_field} differs from eligibility"
            )
    if (
        registration["holdout_public_commitment_sha256"]
        != authorization["holdout_public_commitment"]["sha256"]
        or registration["source_bank_registry_sha256"]
        != authorization["source_bank_registry"]["sha256"]
    ):
        raise ValueError("checkpoint run registration public dataset refs differ from authorization")
    for name, registered_path in TRAINING_CODE_ARTIFACTS.items():
        record = code_registry["artifacts"][name]
        if registration[f"{name}_path"] != registered_path or registration[
            f"{name}_sha256"
        ] != record["sha256"]:
            raise ValueError(f"checkpoint run registration code ref differs: {name}")
    builder_path = resolve_path(project_root, CANONICAL_PROMPT_BUILDER)
    if (
        not builder_path.is_file()
        or builder_path.is_symlink()
        or registration["canonical_prompt_builder_sha256"] != file_sha256(builder_path)
    ):
        raise ValueError("checkpoint run registration canonical prompt builder differs")
    mapping_path = resolve_path(project_root, SOURCE_MAPPING_REGISTRY)
    if (
        not mapping_path.is_file()
        or mapping_path.is_symlink()
        or file_sha256(mapping_path) != eligibility["source_mapping_registry_sha256"]
    ):
        raise ValueError("checkpoint run registration source mapping bytes differ")
    _require_sha256(
        registration["prompt_sidecar_manifest_sha256"],
        "checkpoint run registration/prompt sidecar manifest",
    )
    if _contains_placeholder(registration):
        raise ValueError("checkpoint run registration contains a placeholder")
    return registration


def validate_training_state_payload(
    project_root: Path,
    state: Any,
    *,
    eligibility: Mapping[str, Any],
    registration: Mapping[str, Any],
    preflight: Mapping[str, Any],
    sanity: Mapping[str, Any],
    authorization: Mapping[str, Any],
    code_registry: Mapping[str, Any],
    weights_path: Path,
) -> dict[str, Any]:
    fields = {
        "protocol",
        "status",
        "dataset_version",
        "step",
        "max_steps",
        "only_training_intervention",
        "training_config",
        "manifest",
        "train_manifest_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "source_bank_registry_sha256",
        "holdout_public_commitment_sha256",
        "holdout_public_commitment_path",
        "holdout_count",
        "source_mapping_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "prompt_sidecar_inventory_sha256",
        "prompt_sidecar_manifest_sha256",
        "model_content_inventory_sha256",
        "transformer_inventory_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "preflight_artifact_sha256",
        "training_authorization_path",
        "training_authorization_sha256",
        "training_code_registry_path",
        "training_code_registry_sha256",
        "run_registration_sha256",
        "scale_sanity_sha256",
        "initial_lora_sha256",
        "sample_order_sha256",
        "noise_sigma_rng_initial_sha256",
        "noise_sigma_rng_final_sha256",
        "role_step_counts",
        "active_source_counts",
        "mean_loss_last_20",
        "mean_target_prompt_teacher_loss_last_20",
        "mean_preserve_loss_last_20",
        "trainer_sha256",
        "launcher_sha256",
        "final_lora_finite_check",
    }
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("eligible checkpoint training-state fields are not exact")
    if (
        state["protocol"] != TRAINING_STATE_PROTOCOL
        or state["status"] != "eligible_training_complete"
        or state["dataset_version"] != DATASET_VERSION
        or state["step"] != 200
        or state["max_steps"] != 200
        or state["only_training_intervention"] != ONLY_TRAINING_INTERVENTION
        or state["training_config"] != EXPECTED_TRAINING_CONFIG
        or state["manifest"] != TRAINING_MANIFEST
        or state["holdout_public_commitment_path"] != PUBLIC_HOLDOUT_COMMITMENT
        or state["holdout_count"] != 24
        or state["canonical_prompt_builder_path"] != CANONICAL_PROMPT_BUILDER
        or state["runtime_registry_path"] != RUNTIME_REGISTRY
        or state["training_authorization_path"] != TRAINING_AUTHORIZATION
        or state["training_code_registry_path"] != TRAINING_CODE_REGISTRY
        or state["role_step_counts"] != {"erase": 100, "preserve": 100}
    ):
        raise ValueError("eligible checkpoint training-state identity/configuration mismatch")
    direct_fields = (
        "train_manifest_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "source_bank_registry_sha256",
        "source_mapping_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "prompt_sidecar_inventory_sha256",
        "model_content_inventory_sha256",
        "transformer_inventory_sha256",
        "runtime_registry_sha256",
        "initial_lora_sha256",
        "sample_order_sha256",
        "noise_sigma_rng_initial_sha256",
        "noise_sigma_rng_final_sha256",
        "trainer_sha256",
        "launcher_sha256",
        "training_authorization_sha256",
        "training_code_registry_sha256",
    )
    for field in direct_fields:
        if state[field] != eligibility[field]:
            raise ValueError(f"eligible checkpoint training-state {field} mismatch")
    if (
        state["holdout_public_commitment_sha256"]
        != authorization["holdout_public_commitment"]["sha256"]
        or state["prompt_sidecar_manifest_sha256"]
        != registration["prompt_sidecar_manifest_sha256"]
        or state["canonical_prompt_builder_sha256"]
        != registration["canonical_prompt_builder_sha256"]
        or state["preflight_artifact_sha256"] != eligibility["preflight"]["sha256"]
        or state["run_registration_sha256"]
        != eligibility["run_registration"]["sha256"]
        or state["scale_sanity_sha256"] != eligibility["scale_sanity"]["sha256"]
        or state["scale_sanity_sha256"] != file_sha256(
            resolve_path(project_root, SCALE_SANITY)
        )
    ):
        raise ValueError("eligible checkpoint training-state artifact cross-binding mismatch")
    if state["final_lora_finite_check"] != eligibility["final_lora_finite_check"]:
        raise ValueError("training-state/eligibility final LoRA finite checks differ")
    validate_final_lora_finite_check(state["final_lora_finite_check"])
    validate_safetensors_finite_inventory(
        weights_path, expected_evidence=state["final_lora_finite_check"]
    )
    active_counts = state["active_source_counts"]
    if (
        not isinstance(active_counts, dict)
        or not active_counts
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for name, count in active_counts.items()
        )
        or sum(active_counts.values()) != 100
    ):
        raise ValueError("eligible checkpoint active source counts are invalid")
    mapping = _load_exact_json(
        resolve_path(project_root, SOURCE_MAPPING_REGISTRY),
        "checkpoint-bound source mapping registry",
    )
    if mapping.get("active_source_counts") != active_counts:
        raise ValueError("eligible checkpoint active source counts differ from frozen mapping")
    for field in (
        "mean_loss_last_20",
        "mean_target_prompt_teacher_loss_last_20",
        "mean_preserve_loss_last_20",
    ):
        if _finite_number(state[field]) < 0:
            raise ValueError(f"eligible checkpoint training-state {field} is negative")
    if sanity.get("passed") is not True or preflight.get("status") != "passed":
        raise ValueError("eligible checkpoint wraps failed sanity or preflight evidence")
    if (
        state["trainer_sha256"] != code_registry["artifacts"]["trainer"]["sha256"]
        or state["launcher_sha256"]
        != code_registry["artifacts"]["launcher"]["sha256"]
    ):
        raise ValueError("eligible checkpoint training-state code hashes differ")
    if _contains_placeholder(state):
        raise ValueError("eligible checkpoint training-state contains a placeholder")
    return state


def validate_null_preflight_payload(
    project_root: Path,
    payload: Any,
    *,
    eligibility: Mapping[str, Any],
    registration: Mapping[str, Any],
    authorization: Mapping[str, Any],
    code_registry: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint null-sidecar preflight is not an object")
    required_fields = {
        "protocol",
        "status",
        "dataset_version",
        "train_manifest_sha256",
        "source_bank_registry_sha256",
        "source_bank_registry_path",
        "holdout_public_commitment_sha256",
        "holdout_public_commitment_path",
        "holdout_count",
        "causal_stage0_public_commitment_path",
        "causal_stage0_public_commitment_sha256",
        "source_mapping_registry_sha256",
        "source_mapping_registry_path",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "prompt_sidecar_inventory_sha256",
        "prompt_sidecar_manifest_sha256",
        "preparer_sha256",
        "model_content_inventory_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "transformer_inventory_sha256",
        "model_artifact_inventory",
        "model_revision",
        "seed",
        "sample_order_sha256",
        "noise_sigma_rng_initial_sha256",
        "noise_sigma_rng_final_sha256",
        "initial_lora_sha256",
        "original_reencode_count",
        "original_reencode_binding_sha256",
        "unique_augmented_reencode_count",
        "augmented_reencode_row_count",
        "augmented_reencode_binding_sha256",
        "augmented_reencode_all_rows_byte_equal",
        "tokenizer_binding",
        "integration_manifest_index",
        "integration_scene_id",
        "v3b_reference_path",
        "v4_null_sidecar_path",
        "null_sidecar_substitution",
        "forward_loss_gradient_equal",
        "rng_restored_between_signatures",
        "trainable_state_restored_between_signatures",
        "v3b_reference_signature",
        "v4_null_sidecar_signature",
        "optimizer_created",
    }
    if set(payload) != required_fields:
        raise ValueError("checkpoint null-sidecar preflight fields are not exact")
    expected_literals = {
        "protocol": NULL_PREFLIGHT_PROTOCOL,
        "status": "passed",
        "dataset_version": DATASET_VERSION,
        "source_bank_registry_path": PUBLIC_SOURCE_BANK,
        "holdout_public_commitment_path": PUBLIC_HOLDOUT_COMMITMENT,
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": PENDING_STAGE0_COMMITMENTS[
            "causal"
        ],
        "causal_stage0_public_commitment_sha256": (
            FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        "source_mapping_registry_path": SOURCE_MAPPING_REGISTRY,
        "canonical_prompt_builder_path": CANONICAL_PROMPT_BUILDER,
        "runtime_registry_path": RUNTIME_REGISTRY,
        "seed": 26000,
        "original_reencode_count": 178,
        "augmented_reencode_row_count": 178,
        "augmented_reencode_all_rows_byte_equal": True,
        "v3b_reference_path": "frozen_base_cache_prompt_embeds",
        "v4_null_sidecar_path": (
            "v4_sidecar_loader_with_fresh_original_augmented_prompt_embeds"
        ),
        "null_sidecar_substitution": (
            "fresh_original_embedding_for_augmented_embedding"
        ),
        "forward_loss_gradient_equal": True,
        "rng_restored_between_signatures": True,
        "trainable_state_restored_between_signatures": True,
        "optimizer_created": False,
    }
    if any(payload.get(field) != expected for field, expected in expected_literals.items()):
        raise ValueError("checkpoint null-sidecar preflight did not pass the exact protocol")
    direct_fields = (
        "train_manifest_sha256",
        "source_bank_registry_sha256",
        "source_mapping_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "prompt_sidecar_inventory_sha256",
        "model_content_inventory_sha256",
        "runtime_registry_sha256",
        "transformer_inventory_sha256",
        "sample_order_sha256",
        "noise_sigma_rng_initial_sha256",
        "noise_sigma_rng_final_sha256",
        "initial_lora_sha256",
    )
    if any(payload[field] != eligibility[field] for field in direct_fields):
        raise ValueError("checkpoint null-sidecar preflight differs from eligibility")
    if (
        payload["holdout_public_commitment_sha256"]
        != authorization["holdout_public_commitment"]["sha256"]
        or payload["prompt_sidecar_manifest_sha256"]
        != registration["prompt_sidecar_manifest_sha256"]
        or payload["canonical_prompt_builder_sha256"]
        != registration["canonical_prompt_builder_sha256"]
        or payload["v3b_reference_signature"]
        != payload["v4_null_sidecar_signature"]
    ):
        raise ValueError("checkpoint null-sidecar preflight cross-binding mismatch")
    if (
        payload["preparer_sha256"]
        != code_registry["artifacts"]["preparer"]["sha256"]
    ):
        raise ValueError("checkpoint null-sidecar preflight preparer differs from code registry")
    pending_stage0_path = resolve_path(
        project_root, PENDING_STAGE0_COMMITMENTS["causal"]
    )
    if (
        pending_stage0_path.is_symlink()
        or not pending_stage0_path.is_file()
        or file_sha256(pending_stage0_path)
        != FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
    ):
        raise ValueError("checkpoint null-sidecar public causal Stage-0 bytes differ")
    inventory = payload["model_artifact_inventory"]
    if (
        not isinstance(inventory, dict)
        or set(inventory)
        != {"algorithm", "root", "excluded", "file_count", "sha256", "files"}
        or inventory.get("algorithm")
        != "sha256_ordered_relative_path_nul_bytes_newline_with_file_records_v1"
        or inventory.get("root") != GENERATION_SPEC["model"]
        or inventory.get("sha256") != eligibility["model_content_inventory_sha256"]
        or not isinstance(inventory.get("file_count"), int)
        or inventory["file_count"] <= 0
        or not isinstance(inventory.get("files"), list)
        or len(inventory["files"]) != inventory["file_count"]
        or payload["model_revision"] != EXPECTED_MODEL_REVISION
    ):
        raise ValueError("checkpoint null-sidecar preflight model provenance is invalid")
    model_file_fields = {"path", "size", "sha256"}
    model_files = inventory["files"]
    if any(
        not isinstance(record, dict)
        or set(record) != model_file_fields
        or not isinstance(record["path"], str)
        or not record["path"]
        or not isinstance(record["size"], int)
        or isinstance(record["size"], bool)
        or record["size"] < 0
        or not is_sha256(record["sha256"])
        for record in model_files
    ) or [record["path"] for record in model_files] != sorted(
        record["path"] for record in model_files
    ):
        raise ValueError("checkpoint null-sidecar preflight model file inventory is invalid")
    tokenizer_records = [
        record for record in model_files if record["path"].startswith("tokenizer/")
    ]
    expected_tokenizer_binding = {
        "path": f"{GENERATION_SPEC['model']}/tokenizer",
        "file_count": len(tokenizer_records),
        "inventory_sha256": canonical_json_sha256(tokenizer_records),
    }
    if not tokenizer_records or payload["tokenizer_binding"] != expected_tokenizer_binding:
        raise ValueError("checkpoint null-sidecar tokenizer inventory is not exact")
    if (
        not isinstance(payload["unique_augmented_reencode_count"], int)
        or isinstance(payload["unique_augmented_reencode_count"], bool)
        or not 1 <= payload["unique_augmented_reencode_count"] <= 178
        or not isinstance(payload["integration_manifest_index"], int)
        or isinstance(payload["integration_manifest_index"], bool)
        or payload["integration_manifest_index"] < 0
        or not isinstance(payload["integration_scene_id"], str)
        or not payload["integration_scene_id"]
    ):
        raise ValueError("checkpoint null-sidecar preflight observation identity is invalid")
    _require_sha256(
        payload["original_reencode_binding_sha256"],
        "checkpoint null-sidecar original reencode binding",
    )
    _require_sha256(
        payload["augmented_reencode_binding_sha256"],
        "checkpoint null-sidecar augmented reencode binding",
    )
    signature_fields = {
        "prediction_sha256",
        "teacher_prediction_sha256",
        "flow_loss_sha256",
        "teacher_loss_sha256",
        "combined_loss_sha256",
        "gradient_state_sha256",
    }
    for label in ("v3b_reference_signature", "v4_null_sidecar_signature"):
        signature = payload[label]
        if not isinstance(signature, dict) or set(signature) != signature_fields:
            raise ValueError(f"checkpoint null-sidecar {label} is not exact")
        for field, value in signature.items():
            _require_sha256(value, f"checkpoint null-sidecar {label}/{field}")
    if _contains_placeholder(payload):
        raise ValueError("checkpoint null-sidecar preflight contains a placeholder")
    return payload


def _named_file_inventory_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"registered cache artifact is missing or symlinked: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def validate_prompt_sidecar_manifest_payload(
    project_root: Path,
    payload: Any,
    *,
    eligibility: Mapping[str, Any],
    registration: Mapping[str, Any],
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open the public sidecar manifest and its complete 178-file inventory."""

    fields = {
        "protocol",
        "status",
        "dataset_version",
        "source_manifest",
        "source_manifest_sha256",
        "source_mapping_registry",
        "source_mapping_registry_sha256",
        "source_bank_registry_sha256",
        "holdout_public_commitment_path",
        "holdout_public_commitment_sha256",
        "holdout_count",
        "causal_stage0_public_commitment_path",
        "causal_stage0_public_commitment_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256",
        "base_cache_dir",
        "base_cache_inventory_sha256",
        "teacher_cache_dir",
        "teacher_cache_inventory_sha256",
        "model",
        "model_content_inventory_sha256",
        "runtime_registry_path",
        "runtime_registry_sha256",
        "transformer_inventory_sha256",
        "model_artifact_inventory",
        "model_revision",
        "dtype",
        "shape",
        "max_sequence_length",
        "truncation_allowed",
        "erase_row_count",
        "unique_original_prompt_count",
        "unique_augmented_prompt_count",
        "max_registered_token_length",
        "original_reencode_binding_sha256",
        "sidecar_record_binding_sha256",
        "cache_inventory_sha256",
        "files",
        "runtime_versions",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("checkpoint prompt-sidecar manifest fields are not exact")
    expected_literals = {
        "protocol": PROMPT_SIDECAR_PROTOCOL,
        "status": "prepared",
        "dataset_version": DATASET_VERSION,
        "source_manifest": TRAINING_MANIFEST,
        "source_mapping_registry": SOURCE_MAPPING_REGISTRY,
        "holdout_public_commitment_path": PUBLIC_HOLDOUT_COMMITMENT,
        "holdout_count": 24,
        "causal_stage0_public_commitment_path": PENDING_STAGE0_COMMITMENTS[
            "causal"
        ],
        "causal_stage0_public_commitment_sha256": (
            FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        "canonical_prompt_builder_path": CANONICAL_PROMPT_BUILDER,
        "base_cache_dir": BASE_CACHE_DIR,
        "teacher_cache_dir": TEACHER_CACHE_DIR,
        "model": GENERATION_SPEC["model"],
        "runtime_registry_path": RUNTIME_REGISTRY,
        "dtype": "torch.bfloat16",
        "shape": [1, 226, 4096],
        "max_sequence_length": 226,
        "truncation_allowed": False,
        "erase_row_count": 178,
        "unique_original_prompt_count": 178,
        "runtime_versions": EXPECTED_SIDECAR_RUNTIME_VERSIONS,
        "model_revision": EXPECTED_MODEL_REVISION,
    }
    if any(payload.get(field) != expected for field, expected in expected_literals.items()):
        raise ValueError("checkpoint prompt-sidecar identity/configuration mismatch")
    direct_fields = (
        "source_manifest_sha256",
        "source_mapping_registry_sha256",
        "source_bank_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "model_content_inventory_sha256",
        "runtime_registry_sha256",
        "transformer_inventory_sha256",
        "cache_inventory_sha256",
    )
    eligibility_names = {
        "source_manifest_sha256": "train_manifest_sha256",
        "cache_inventory_sha256": "prompt_sidecar_inventory_sha256",
    }
    for field in direct_fields:
        if payload[field] != eligibility[eligibility_names.get(field, field)]:
            raise ValueError(f"checkpoint prompt-sidecar {field} differs from eligibility")
    if (
        payload["holdout_public_commitment_sha256"]
        != authorization["holdout_public_commitment"]["sha256"]
        or payload["canonical_prompt_builder_sha256"]
        != registration["canonical_prompt_builder_sha256"]
        or payload["original_reencode_binding_sha256"]
        != preflight["original_reencode_binding_sha256"]
        or payload["model_artifact_inventory"]
        != preflight["model_artifact_inventory"]
    ):
        raise ValueError("checkpoint prompt-sidecar provenance cross-binding mismatch")
    for field in ("original_reencode_binding_sha256", "sidecar_record_binding_sha256"):
        _require_sha256(payload[field], f"checkpoint prompt-sidecar/{field}")

    manifest_path = resolve_path(project_root, TRAINING_MANIFEST)
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or file_sha256(manifest_path) != FROZEN_TRAIN_MANIFEST_SHA256
    ):
        raise ValueError("checkpoint prompt-sidecar source manifest bytes differ")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    erase = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("training_role") == "erase"
    ]
    if len(rows) != 214 or len(erase) != 178 or len(
        {row.get("prompt") for _, row in erase}
    ) != 178:
        raise ValueError("checkpoint prompt-sidecar source manifest inventory differs")
    expected_files = [
        f"{index:03d}_{row['scene_id']}.pt" for index, row in erase
    ]
    if payload["files"] != expected_files:
        raise ValueError("checkpoint prompt-sidecar ordered file inventory differs")

    mapping_path = resolve_path(project_root, SOURCE_MAPPING_REGISTRY)
    if (
        mapping_path.is_symlink()
        or not mapping_path.is_file()
        or file_sha256(mapping_path) != eligibility["source_mapping_registry_sha256"]
    ):
        raise ValueError("checkpoint prompt-sidecar source mapping bytes differ")
    mapping_payload = _load_exact_json(mapping_path, "checkpoint-bound source mapping")
    mapping_fields = {
        "protocol", "status", "dataset_version", "source_bank_registry_sha256",
        "source_bank_schema", "source_bank_registry", "source_bank_size",
        "source_bank_entries_sha256", "holdout_public_commitment_path",
        "holdout_public_commitment_sha256", "holdout_count",
        "source_assignment_salt_sha256", "source_assignment_algorithm_sha256",
        "train_manifest_sha256", "canonical_prompt_builder_path",
        "canonical_prompt_builder_sha256", "seed", "balanced_schedule",
        "sample_order_sha256", "full178_mapping_sha256",
        "active100_mapping_sha256", "active_source_counts",
        "active_prompt_variant_counts", "active_receiver_counts",
        "full_prompt_variant_counts", "full_receiver_counts",
        "active_source_count_min", "active_source_count_max", "erase_row_count",
        "active_erase_count", "mapping",
    }
    records = mapping_payload.get("mapping") if isinstance(mapping_payload, dict) else None
    if (
        not isinstance(mapping_payload, dict)
        or set(mapping_payload) != mapping_fields
        or mapping_payload["protocol"] != "water_impact_dynamic_v4_source_mapping_v2"
        or mapping_payload["status"] != "frozen"
        or mapping_payload["dataset_version"] != DATASET_VERSION
        or mapping_payload["source_bank_registry_sha256"]
        != eligibility["source_bank_registry_sha256"]
        or mapping_payload["holdout_public_commitment_sha256"]
        != authorization["holdout_public_commitment"]["sha256"]
        or mapping_payload["train_manifest_sha256"] != FROZEN_TRAIN_MANIFEST_SHA256
        or mapping_payload["seed"] != 26000
        or mapping_payload["erase_row_count"] != 178
        or mapping_payload["active_erase_count"] != 100
        or not isinstance(records, list)
        or len(records) != 178
    ):
        raise ValueError("checkpoint prompt-sidecar source mapping contract differs")
    active = sorted(
        (record for record in records if record.get("active_erase_ordinal") is not None),
        key=lambda record: record.get("active_erase_ordinal"),
    )
    if (
        len(active) != 100
        or [record.get("active_erase_ordinal") for record in active] != list(range(100))
        or canonical_json_sha256(records) != payload["full178_mapping_sha256"]
        or canonical_json_sha256(active) != payload["active100_mapping_sha256"]
        or mapping_payload["full178_mapping_sha256"] != payload["full178_mapping_sha256"]
        or mapping_payload["active100_mapping_sha256"] != payload["active100_mapping_sha256"]
        or mapping_payload["sample_order_sha256"] != eligibility["sample_order_sha256"]
    ):
        raise ValueError("checkpoint prompt-sidecar mapping digests do not recompute")
    augmented_prompts = {
        record.get("augmented_factual_prompt") for record in records
    }
    if (
        None in augmented_prompts
        or len(augmented_prompts) != payload["unique_augmented_prompt_count"]
        or payload["unique_augmented_prompt_count"]
        != preflight["unique_augmented_reencode_count"]
    ):
        raise ValueError("checkpoint prompt-sidecar augmented prompt inventory differs")

    if (
        not isinstance(payload["max_registered_token_length"], int)
        or isinstance(payload["max_registered_token_length"], bool)
        or not 1 <= payload["max_registered_token_length"] <= 226
    ):
        raise ValueError("checkpoint prompt-sidecar registered token length is invalid")
    sidecar_dir = resolve_path(project_root, PROMPT_SIDECAR_DIR)
    if sidecar_dir.is_symlink() or not sidecar_dir.is_dir():
        raise FileNotFoundError("checkpoint prompt-sidecar directory is missing or symlinked")
    expected_paths = [sidecar_dir / name for name in expected_files]
    expected_children = set(expected_files) | {Path(PROMPT_SIDECAR_MANIFEST).name}
    children = list(sidecar_dir.iterdir())
    if (
        {child.name for child in children} != expected_children
        or any(child.is_symlink() or not child.is_file() for child in children)
        or _named_file_inventory_sha256(expected_paths)
        != eligibility["prompt_sidecar_inventory_sha256"]
    ):
        raise ValueError("checkpoint prompt-sidecar byte inventory differs")
    if _contains_placeholder(payload):
        raise ValueError("checkpoint prompt-sidecar contains a placeholder")
    return payload


def validate_checkpoint_eligibility(project_root: Path, path: Path | None = None) -> dict[str, Any]:
    """Validate the trainer's sole eligible step-200 checkpoint attestation."""

    eligibility_path = path or resolve_path(project_root, CHECKPOINT_ELIGIBILITY)
    if eligibility_path.resolve() != resolve_path(
        project_root, CHECKPOINT_ELIGIBILITY
    ).resolve():
        raise ValueError("checkpoint eligibility path is outside the v2 namespace")
    payload = _load_exact_json(eligibility_path, "v4 checkpoint eligibility")
    required = {
        "protocol",
        "status",
        "dataset_version",
        "step",
        "checkpoint",
        "run_registration",
        "preflight",
        "scale_sanity",
        "role_step_counts",
        "final_lora_finite_check",
        *CHECKPOINT_HASH_FIELDS,
    }
    if set(payload) != required:
        raise ValueError("checkpoint eligibility fields are not exact")
    if payload["protocol"] != CHECKPOINT_ELIGIBILITY_PROTOCOL:
        raise ValueError("checkpoint eligibility protocol mismatch")
    if payload["dataset_version"] != DATASET_VERSION:
        raise ValueError("checkpoint eligibility dataset version mismatch")
    if payload["status"] != "eligible" or payload["step"] != 200:
        raise ValueError("only an eligible step-200 checkpoint may be evaluated")
    for name in CHECKPOINT_HASH_FIELDS:
        _require_sha256(payload[name], f"checkpoint eligibility/{name}")
    fixed_hashes = {
        "model_content_inventory_sha256": FROZEN_MODEL_CONTENT_INVENTORY_SHA256,
        "transformer_inventory_sha256": FROZEN_TRANSFORMER_INVENTORY_SHA256,
        "source_bank_registry_sha256": FROZEN_PUBLIC_SOURCE_BANK_SHA256,
        "train_manifest_sha256": FROZEN_TRAIN_MANIFEST_SHA256,
        "base_cache_inventory_sha256": FROZEN_BASE_CACHE_INVENTORY_SHA256,
        "teacher_cache_inventory_sha256": FROZEN_TEACHER_CACHE_INVENTORY_SHA256,
        "sample_order_sha256": FROZEN_SAMPLE_ORDER_SHA256,
        "noise_sigma_rng_initial_sha256": (
            FROZEN_NOISE_SIGMA_RNG_INITIAL_SHA256
        ),
        "noise_sigma_rng_final_sha256": FROZEN_NOISE_SIGMA_RNG_FINAL_SHA256,
        "initial_lora_sha256": FROZEN_INITIAL_LORA_SHA256,
    }
    for field, expected_sha256 in fixed_hashes.items():
        if payload[field] != expected_sha256:
            raise ValueError(
                f"checkpoint eligibility/{field} differs from the frozen digest"
            )
    if payload["role_step_counts"] != {"erase": 100, "preserve": 100}:
        raise ValueError("checkpoint role counts must be exactly 100/100")
    validate_final_lora_finite_check(payload["final_lora_finite_check"])
    expected_artifact_paths = {
        "checkpoint": V4_CHECKPOINT,
        "run_registration": RUN_REGISTRATION,
        "preflight": NULL_SIDECAR_PREFLIGHT,
        "scale_sanity": SCALE_SANITY,
    }
    for label, expected_path in expected_artifact_paths.items():
        record = payload.get(label)
        if not isinstance(record, dict) or record.get("path") != expected_path:
            raise ValueError(
                f"checkpoint eligibility/{label}: path is outside the v2 namespace"
            )
    authorization = resolve_path(project_root, TRAINING_AUTHORIZATION)
    code_registry = resolve_path(project_root, TRAINING_CODE_REGISTRY)
    runtime_registry = resolve_path(project_root, RUNTIME_REGISTRY)
    if file_sha256(authorization) != payload["training_authorization_sha256"]:
        raise ValueError("checkpoint eligibility training-authorization hash mismatch")
    if file_sha256(code_registry) != payload["training_code_registry_sha256"]:
        raise ValueError("checkpoint eligibility training-code registry hash mismatch")
    if file_sha256(runtime_registry) != payload["runtime_registry_sha256"]:
        raise ValueError("checkpoint eligibility runtime-registry hash mismatch")
    authorization_payload = validate_training_authorization(
        project_root,
        expected_gate_spec=GATE_SPEC,
        authorization_path=authorization,
    )
    code_ref = authorization_payload.get("code_registry")
    if not isinstance(code_ref, dict) or code_ref != {
        "path": TRAINING_CODE_REGISTRY,
        "sha256": payload["training_code_registry_sha256"],
    }:
        raise ValueError("checkpoint eligibility is not cross-bound to authorization code registry")
    runtime_ref = authorization_payload.get("runtime_registry")
    expected_runtime_ref = {
        "path": RUNTIME_REGISTRY,
        "sha256": payload["runtime_registry_sha256"],
    }
    if runtime_ref != expected_runtime_ref:
        raise ValueError("checkpoint eligibility is not cross-bound to authorization runtime registry")
    validate_runtime_registry(runtime_registry, payload["runtime_registry_sha256"])
    code_payload = validate_training_code_registry(project_root, code_registry)
    if code_payload["runtime_registry"] != expected_runtime_ref:
        raise ValueError("checkpoint eligibility/code registry runtime references differ")
    if payload["trainer_sha256"] != code_payload["artifacts"]["trainer"]["sha256"]:
        raise ValueError("checkpoint eligibility trainer hash differs from frozen code registry")
    if payload["launcher_sha256"] != code_payload["artifacts"]["launcher"]["sha256"]:
        raise ValueError("checkpoint eligibility launcher hash differs from frozen code registry")
    for label, expected_fields in {
        "checkpoint": {"path", "weights_sha256", "training_state_sha256"},
        "run_registration": {"path", "sha256"},
        "preflight": {"path", "sha256"},
        "scale_sanity": {"path", "sha256"},
    }.items():
        record = payload[label]
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError(f"checkpoint eligibility/{label}: record is not exact")
        artifact = resolve_path(project_root, str(record["path"]))
        if not artifact.exists() or artifact.is_symlink():
            raise FileNotFoundError(f"checkpoint eligibility/{label}: artifact missing")
        hash_field = "weights_sha256" if label == "checkpoint" else "sha256"
        _require_sha256(record[hash_field], f"checkpoint eligibility/{label}/{hash_field}")
        if label == "checkpoint":
            _require_sha256(record["training_state_sha256"], "training_state_sha256")
            if str(record["path"]) != V4_CHECKPOINT or not artifact.is_dir():
                raise ValueError("checkpoint eligibility must bind the exact checkpoint-000200 directory")
            files = {item.name: item for item in artifact.iterdir()}
            if set(files) != {"pytorch_lora_weights.safetensors", "training_state_v2.json"} or any(
                item.is_symlink() or not item.is_file() for item in files.values()
            ):
                raise ValueError("eligible checkpoint inventory must contain exactly weights and training_state")
            if file_sha256(files["pytorch_lora_weights.safetensors"]) != record["weights_sha256"]:
                raise ValueError("eligible checkpoint weights byte hash mismatch")
            if file_sha256(files["training_state_v2.json"]) != record["training_state_sha256"]:
                raise ValueError("eligible checkpoint training-state byte hash mismatch")
            training_state = _load_exact_json(
                files["training_state_v2.json"], "eligible checkpoint training state"
            )
            if training_state.get("dataset_version") != DATASET_VERSION:
                raise ValueError("eligible checkpoint training-state dataset version mismatch")
            if training_state.get("model_content_inventory_sha256") != payload[
                "model_content_inventory_sha256"
            ]:
                raise ValueError(
                    "eligible checkpoint training-state full-model inventory mismatch"
                )
            weights_path = files["pytorch_lora_weights.safetensors"]
        else:
            if not artifact.is_file() or file_sha256(artifact) != record["sha256"]:
                raise ValueError(f"checkpoint eligibility/{label}: byte hash mismatch")
    registration_path = resolve_path(project_root, str(payload["run_registration"]["path"]))
    preflight_path = resolve_path(project_root, str(payload["preflight"]["path"]))
    sanity_path = resolve_path(project_root, str(payload["scale_sanity"]["path"]))
    registration = _load_exact_json(registration_path, "checkpoint-bound run registration")
    preflight = _load_exact_json(preflight_path, "checkpoint-bound null-sidecar preflight")
    sanity = _load_exact_json(sanity_path, "checkpoint-bound scale sanity")
    validate_run_registration_payload(
        project_root,
        registration,
        eligibility=payload,
        authorization=authorization_payload,
        code_registry=code_payload,
    )
    validate_null_preflight_payload(
        project_root,
        preflight,
        eligibility=payload,
        registration=registration,
        authorization=authorization_payload,
        code_registry=code_payload,
    )
    live_model_inventory = model_artifact_inventory(
        project_root, GENERATION_SPEC["model"]
    )
    if live_model_inventory != preflight["model_artifact_inventory"]:
        raise ValueError("checkpoint preflight model inventory differs from current bytes")
    validate_scale_sanity_payload(
        sanity,
        expected_run_registration_sha256=payload["run_registration"]["sha256"],
    )
    registration_expected = {
        "dataset_version": DATASET_VERSION,
        "runtime_registry_path": RUNTIME_REGISTRY,
        "runtime_registry_sha256": payload["runtime_registry_sha256"],
        "training_authorization_path": TRAINING_AUTHORIZATION,
        "training_authorization_sha256": payload["training_authorization_sha256"],
        "training_code_registry_path": TRAINING_CODE_REGISTRY,
        "training_code_registry_sha256": payload["training_code_registry_sha256"],
        "preflight_artifact_path": str(payload["preflight"]["path"]),
        "preflight_artifact_sha256": payload["preflight"]["sha256"],
        "prompt_sidecar_path": PROMPT_SIDECAR_DIR,
        "prompt_sidecar_inventory_sha256": payload[
            "prompt_sidecar_inventory_sha256"
        ],
    }
    for name, expected in registration_expected.items():
        if registration.get(name) != expected:
            raise ValueError(f"checkpoint run registration {name} mismatch")
    sidecar_manifest_sha256 = registration.get("prompt_sidecar_manifest_sha256")
    _require_sha256(
        sidecar_manifest_sha256,
        "checkpoint run registration/prompt sidecar manifest",
    )
    cross_fields = (
        "model_content_inventory_sha256",
        "transformer_inventory_sha256",
        "source_bank_registry_sha256",
        "source_mapping_registry_sha256",
        "active100_mapping_sha256",
        "full178_mapping_sha256",
        "train_manifest_sha256",
        "base_cache_inventory_sha256",
        "teacher_cache_inventory_sha256",
        "prompt_sidecar_inventory_sha256",
        "runtime_registry_sha256",
    )
    for name in cross_fields:
        if registration.get(name) != payload[name]:
            raise ValueError(f"checkpoint run registration {name} differs from eligibility")
    preflight_expected = {
        "dataset_version": DATASET_VERSION,
        "runtime_registry_path": RUNTIME_REGISTRY,
        "runtime_registry_sha256": payload["runtime_registry_sha256"],
        "prompt_sidecar_inventory_sha256": payload[
            "prompt_sidecar_inventory_sha256"
        ],
        "prompt_sidecar_manifest_sha256": sidecar_manifest_sha256,
        "transformer_inventory_sha256": payload["transformer_inventory_sha256"],
        "model_content_inventory_sha256": payload[
            "model_content_inventory_sha256"
        ],
        "source_bank_registry_sha256": payload["source_bank_registry_sha256"],
        "source_mapping_registry_sha256": payload["source_mapping_registry_sha256"],
        "active100_mapping_sha256": payload["active100_mapping_sha256"],
        "full178_mapping_sha256": payload["full178_mapping_sha256"],
        "train_manifest_sha256": payload["train_manifest_sha256"],
        "base_cache_inventory_sha256": payload["base_cache_inventory_sha256"],
        "teacher_cache_inventory_sha256": payload["teacher_cache_inventory_sha256"],
        "sample_order_sha256": payload["sample_order_sha256"],
        "noise_sigma_rng_initial_sha256": payload[
            "noise_sigma_rng_initial_sha256"
        ],
        "noise_sigma_rng_final_sha256": payload["noise_sigma_rng_final_sha256"],
        "initial_lora_sha256": payload["initial_lora_sha256"],
    }
    for name, expected in preflight_expected.items():
        if preflight.get(name) != expected:
            raise ValueError(f"checkpoint preflight {name} differs from eligibility")
    sidecar_path = resolve_path(project_root, PROMPT_SIDECAR_MANIFEST)
    if (
        not sidecar_path.is_file()
        or sidecar_path.is_symlink()
        or file_sha256(sidecar_path) != sidecar_manifest_sha256
    ):
        raise ValueError("checkpoint prompt-sidecar manifest byte hash mismatch")
    sidecar = _load_exact_json(sidecar_path, "checkpoint-bound prompt sidecar manifest")
    validate_prompt_sidecar_manifest_payload(
        project_root,
        sidecar,
        eligibility=payload,
        registration=registration,
        preflight=preflight,
        authorization=authorization_payload,
    )
    sidecar_expected = {
        "dataset_version": DATASET_VERSION,
        "runtime_registry_path": RUNTIME_REGISTRY,
        "runtime_registry_sha256": payload["runtime_registry_sha256"],
        "cache_inventory_sha256": payload["prompt_sidecar_inventory_sha256"],
        "transformer_inventory_sha256": payload["transformer_inventory_sha256"],
        "model_content_inventory_sha256": payload[
            "model_content_inventory_sha256"
        ],
        "source_bank_registry_sha256": payload["source_bank_registry_sha256"],
        "holdout_public_commitment_path": PUBLIC_HOLDOUT_COMMITMENT,
        "holdout_public_commitment_sha256": authorization_payload[
            "holdout_public_commitment"
        ]["sha256"],
        "causal_stage0_public_commitment_path": PENDING_STAGE0_COMMITMENTS[
            "causal"
        ],
        "causal_stage0_public_commitment_sha256": (
            FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256
        ),
        "source_mapping_registry_sha256": payload["source_mapping_registry_sha256"],
        "train_manifest_sha256": payload["train_manifest_sha256"],
        "base_cache_inventory_sha256": payload["base_cache_inventory_sha256"],
        "teacher_cache_inventory_sha256": payload["teacher_cache_inventory_sha256"],
    }
    for name, expected in sidecar_expected.items():
        if sidecar.get(name) != expected:
            raise ValueError(f"checkpoint prompt-sidecar {name} differs from eligibility")
    validate_training_state_payload(
        project_root,
        training_state,
        eligibility=payload,
        registration=registration,
        preflight=preflight,
        sanity=sanity,
        authorization=authorization_payload,
        code_registry=code_payload,
        weights_path=weights_path,
    )
    if _contains_placeholder(payload):
        raise ValueError("checkpoint eligibility contains a placeholder")
    return payload


def derive_seed(
    private_salt: str,
    stable_case_id: str,
    replicate: int,
    *,
    dataset: str,
) -> int:
    """Apply the Stage-0 domain-separated uint32 seed formula exactly."""

    if dataset not in DATASETS:
        raise ValueError("seed derivation requires a causal/specificity domain")
    if (
        not isinstance(private_salt, str)
        or not private_salt
        or not isinstance(stable_case_id, str)
        or not stable_case_id
        or not isinstance(replicate, int)
        or isinstance(replicate, bool)
        or replicate < 0
    ):
        raise ValueError(
            "seed derivation requires a nonempty salt/ID and nonnegative integer replicate"
        )
    payload = b"\0".join(
        (
            SEED_DOMAINS[dataset].encode("utf-8"),
            private_salt.encode("utf-8"),
            stable_case_id.encode("utf-8"),
            str(replicate).encode("utf-8"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def derive_unit_rows(
    selected_cases: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    private_salt: str,
    forbidden_seeds: Iterable[int] = (),
) -> list[dict[str, Any]]:
    if dataset not in DATASETS or len(selected_cases) != CASE_COUNTS[dataset]:
        raise ValueError(f"{dataset}: selected case count must be exactly {CASE_COUNTS.get(dataset)}")
    case_field = "semantic_case_id" if dataset == "causal" else "specificity_case_id"
    seen_cases: set[str] = set()
    output: list[dict[str, Any]] = []
    seeds: set[int] = set(int(value) for value in forbidden_seeds)
    for case in selected_cases:
        case_id = str(case.get(case_field, ""))
        if not case_id or case_id in seen_cases:
            raise ValueError(f"{dataset}: duplicate or blank case ID")
        seen_cases.add(case_id)
        for replicate in range(REPLICATES[dataset]):
            seed = derive_seed(
                private_salt,
                case_id,
                replicate,
                dataset=dataset,
            )
            if seed in seeds:
                raise ValueError(f"{dataset}: seed collision; registered data version is invalid")
            seeds.add(seed)
            output.append(
                {
                    "unit_id": f"{dataset[0]}u{len(output):03d}",
                    **dict(case),
                    "replicate": replicate,
                    "seed": seed,
                }
            )
    return output


def _score_int(row: Mapping[str, Any], field: str, label: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid score for {field}") from exc
    if value not in {0, 1, 2}:
        raise ValueError(f"{label}: {field} must be 0, 1, or 2")
    return value


def validate_causal_selected_cases(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 24:
        raise ValueError("causal selected manifest must contain exactly 24 cases")
    expected_cells = {(group, variant) for group in CAUSAL_GROUPS for variant in PROMPT_VARIANTS}
    cells = Counter((str(row["group"]), str(row["prompt_variant"])) for row in rows)
    if set(cells) != expected_cells or any(value != 4 for value in cells.values()):
        raise ValueError("causal selection must contain four cases per group/variant cell")
    if any(
        str(row.get("source_physical_audit_status"))
        != (
            "legacy_original_source_exempt"
            if str(row["group"]) == "seen_source_new_receiver"
            else "strict_physical_pass_v2"
        )
        for row in rows
    ):
        raise ValueError("causal selected source physical-audit status is invalid")
    case_ids = [str(row["semantic_case_id"]) for row in rows]
    if len(set(case_ids)) != 24 or any(not value for value in case_ids):
        raise ValueError("causal semantic case IDs must be nonempty and unique")
    receiver_ids = [str(row["receiver_id"]) for row in rows]
    if len(set(receiver_ids)) != 24:
        raise ValueError("all 24 selected causal receiver identities must be unique")
    holdout = [row for row in rows if str(row["group"]) in HOLDOUT_GROUPS]
    if len({str(row["source_head_lemma"]) for row in holdout}) != 16:
        raise ValueError("selected holdout cases must use 16 distinct head lemmas")
    seen_source = [row for row in rows if str(row["group"]) == "seen_source_new_receiver"]
    if len({str(row["source_id"]) for row in seen_source}) != 8:
        raise ValueError("the eight original sources must each occur exactly once")
    seen_receiver = [
        row for row in rows if str(row["group"]) == "holdout_source_seen_receiver"
    ]
    if len({str(row["receiver_id"]) for row in seen_receiver}) != 8:
        raise ValueError("holdout-source/seen-receiver cases require eight distinct receivers")
    new_receiver = [row for row in rows if str(row["group"]) != "holdout_source_seen_receiver"]
    if len({str(row["receiver_id"]) for row in new_receiver}) != 16:
        raise ValueError("the two new-receiver groups require 16 distinct receivers")


def validate_causal_unit_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 72:
        raise ValueError("U must contain exactly 72 units")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("group")) not in CAUSAL_GROUPS or str(row.get("prompt_variant")) not in PROMPT_VARIANTS:
            raise ValueError("U contains an unexpected group or prompt variant")
        by_case[str(row["semantic_case_id"])].append(row)
    if len(by_case) != 24:
        raise ValueError("U must contain exactly 24 semantic cases")
    for case_id, case_rows in by_case.items():
        if {int(row["replicate"]) for row in case_rows} != {0, 1, 2}:
            raise ValueError(f"{case_id}: causal replicates must be exactly 0,1,2")
        invariant = {
            (
                str(row["group"]),
                str(row["prompt_variant"]),
                str(row["source_id"]),
                str(row["source_phrase"]),
                str(row["source_physical_audit_status"]),
                str(row["receiver_id"]),
                str(row["receiver"]),
                str(row["prompt"]),
            )
            for row in case_rows
        }
        if len(invariant) != 1:
            raise ValueError(f"{case_id}: semantic fields differ across replicates")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != 72:
        raise ValueError("U seeds must be globally unique")
    selected = [
        next(dict(row) for row in case_rows if int(row["replicate"]) == 0)
        for case_rows in by_case.values()
    ]
    validate_causal_selected_cases(selected)


def validate_specificity_selected_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    causal_cases: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != 18:
        raise ValueError("specificity selected manifest must contain exactly 18 cases")
    cells = Counter((str(row["membership"]), str(row["prompt_variant"])) for row in rows)
    expected = {
        (membership, variant)
        for membership in SPECIFICITY_MEMBERSHIPS
        for variant in PROMPT_VARIANTS
    }
    if set(cells) != expected or any(value != 3 for value in cells.values()):
        raise ValueError("specificity selection must contain three cases per membership/variant cell")
    if len({str(row["specificity_case_id"]) for row in rows}) != 18:
        raise ValueError("specificity case IDs must be unique")
    if len({str(row["source_head_lemma"]) for row in rows}) != 18:
        raise ValueError("specificity selection must use 18 unique nouns")
    causal_by_id = {str(row["semantic_case_id"]): row for row in causal_cases}
    matched = [row for row in rows if str(row["membership"]) in {"original_source", "holdout_source"}]
    if len(matched) != 12:
        raise ValueError("specificity selection requires six original and six holdout matched cases")
    for row in matched:
        causal = causal_by_id.get(str(row.get("causal_case_id", "")))
        if causal is None:
            raise ValueError("matched specificity case references an unknown causal case")
        for field in ("source_id", "source_phrase", "source_head_lemma", "receiver_id", "receiver"):
            if str(row[field]) != str(causal[field]):
                raise ValueError(f"specificity/causal matched field differs: {field}")
    holdout = [row for row in rows if str(row["membership"]) == "holdout_source"]
    covered = {(str(causal_by_id[str(row["causal_case_id"])]["group"]), str(row["prompt_variant"])) for row in holdout}
    if {group for group, _ in covered} != set(HOLDOUT_GROUPS) or {
        variant for _, variant in covered
    } != set(PROMPT_VARIANTS):
        raise ValueError("holdout specificity cases must cover both causal groups and variants")


def validate_specificity_unit_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    causal_cases: Sequence[Mapping[str, Any]],
    causal_seeds: Iterable[int] = (),
) -> None:
    if len(rows) != 36:
        raise ValueError("W must contain exactly 36 units")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("membership")) not in SPECIFICITY_MEMBERSHIPS:
            raise ValueError("W contains an unexpected membership")
        by_case[str(row["specificity_case_id"])].append(row)
    if len(by_case) != 18:
        raise ValueError("W must contain exactly 18 cases")
    for case_id, case_rows in by_case.items():
        if {int(row["replicate"]) for row in case_rows} != {0, 1}:
            raise ValueError(f"{case_id}: specificity replicates must be exactly 0,1")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != 36 or set(seeds) & {int(value) for value in causal_seeds}:
        raise ValueError("W seeds must be unique and disjoint from U/historical seeds")
    selected = [next(dict(row) for row in case_rows if int(row["replicate"]) == 0) for case_rows in by_case.values()]
    validate_specificity_selected_cases(selected, causal_cases=causal_cases)


def validate_holdout_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    causal_cases: Sequence[Mapping[str, Any]],
    specificity_cases: Sequence[Mapping[str, Any]],
) -> None:
    if len(rows) != 6:
        raise ValueError("M must contain exactly six holdout mappings")
    causal = {str(row["semantic_case_id"]): row for row in causal_cases}
    specificity = {str(row["specificity_case_id"]): row for row in specificity_cases}
    seen_causal: set[str] = set()
    seen_specificity: set[str] = set()
    for row in rows:
        causal_id = str(row["causal_case_id"])
        specificity_id = str(row["specificity_case_id"])
        if causal_id in seen_causal or specificity_id in seen_specificity:
            raise ValueError("M must be one-to-one")
        seen_causal.add(causal_id)
        seen_specificity.add(specificity_id)
        left = causal.get(causal_id)
        right = specificity.get(specificity_id)
        if left is None or right is None or str(right.get("membership")) != "holdout_source":
            raise ValueError("M references a non-holdout or unknown case")
        for field in ("source_id", "source_phrase", "receiver_id", "receiver"):
            if str(left[field]) != str(right[field]) or str(row[field]) != str(left[field]):
                raise ValueError(f"M does not preserve exact {field}")


def validate_public_review_columns(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("public review table is empty")
    fields = set(rows[0])
    allowed_schemas = {
        frozenset(PUBLIC_REVIEW_BASE_FIELDS)
        | frozenset(CAUSAL_SCORE_FIELDS)
        | frozenset(CAUSAL_REFERENCE_ONLY_FIELDS),
        frozenset(PUBLIC_REVIEW_BASE_FIELDS) | frozenset(SPECIFICITY_SCORE_FIELDS),
    }
    if frozenset(fields) not in allowed_schemas:
        raise ValueError("public review table columns are not an exact registered schema")
    leaked = fields & FORBIDDEN_PUBLIC_FIELDS
    if leaked:
        raise ValueError(f"public review table leaks private columns: {sorted(leaked)}")
    if any(set(row) != fields for row in rows):
        raise ValueError("public review rows have inconsistent columns")


def _default_decode(path: Path) -> dict[str, int]:
    import av

    frame_count = 0
    width = height = 0
    rate: Fraction | None = None
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"expected exactly one video stream: {path}")
        stream = streams[0]
        if stream.average_rate is not None:
            rate = Fraction(stream.average_rate.numerator, stream.average_rate.denominator)
        for frame in container.decode(video=0):
            frame_count += 1
            width, height = frame.width, frame.height
    if rate is None:
        raise ValueError(f"video has no exact average frame rate: {path}")
    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps_numerator": rate.numerator,
        "fps_denominator": rate.denominator,
    }


def validate_generation_bundle(
    project_root: Path,
    *,
    dataset: str,
    unit_rows: Sequence[Mapping[str, Any]],
    manifest_paths: Mapping[str, Path],
    checkpoint_eligibility_path: Path,
    decode: Callable[[Path], Mapping[str, int]] = _default_decode,
) -> dict[str, dict[str, Path]]:
    """Validate exact O/v3b/v4 manifests and decoded video inventories."""

    if dataset not in DATASETS or set(manifest_paths) != set(METHODS):
        raise ValueError("generation bundle requires exact original/v3b/v4 manifests")
    expected_n = UNIT_COUNTS[dataset]
    if len(unit_rows) != expected_n:
        raise ValueError(f"{dataset}: unexpected unit count")
    by_unit = {str(row["unit_id"]): row for row in unit_rows}
    if len(by_unit) != expected_n:
        raise ValueError(f"{dataset}: duplicate unit ID")
    unit_manifest_sha = canonical_json_sha256([dict(row) for row in unit_rows])
    eligibility = validate_checkpoint_eligibility(project_root, checkpoint_eligibility_path)
    eligibility_sha = file_sha256(checkpoint_eligibility_path)
    all_paths: set[Path] = set()
    all_inodes: set[tuple[int, int]] = set()
    all_hashes: set[str] = set()
    output: dict[str, dict[str, Path]] = {}
    shared_model_inventory_sha: str | None = None
    shared_runtime_registry_sha: str | None = None
    for method in METHODS:
        manifest_path = manifest_paths[method]
        if manifest_path.name != "v4_generation_manifest_v2.json":
            raise ValueError(
                f"{dataset}/{method}: generation manifest path is outside the v2 namespace"
            )
        manifest = _load_exact_json(manifest_path, f"{dataset}/{method} generation manifest")
        required = {
            "protocol",
            "dataset",
            "dataset_version",
            "method",
            "unit_manifest_canonical_sha256",
            "generation_spec",
            "model_inventory_sha256",
            "runtime_registry_sha256",
            "method_artifact",
            "videos",
        }
        if set(manifest) != required:
            raise ValueError(f"{dataset}/{method}: generation manifest fields are not exact")
        if (
            manifest["protocol"] != GENERATION_MANIFEST_PROTOCOL
            or manifest["dataset"] != dataset
            or manifest["dataset_version"] != DATASET_VERSION
            or manifest["method"] != method
            or manifest["unit_manifest_canonical_sha256"] != unit_manifest_sha
            or manifest["generation_spec"] != GENERATION_SPEC
        ):
            raise ValueError(f"{dataset}/{method}: generation protocol binding mismatch")
        _require_sha256(manifest["model_inventory_sha256"], "model_inventory_sha256")
        if (
            manifest["model_inventory_sha256"]
            != eligibility["model_content_inventory_sha256"]
        ):
            raise ValueError("generation full-model inventory differs from eligible checkpoint")
        if shared_model_inventory_sha is None:
            shared_model_inventory_sha = manifest["model_inventory_sha256"]
        elif manifest["model_inventory_sha256"] != shared_model_inventory_sha:
            raise ValueError("generation arms use different model inventories")
        _require_sha256(
            manifest["runtime_registry_sha256"], "runtime_registry_sha256"
        )
        if shared_runtime_registry_sha is None:
            shared_runtime_registry_sha = manifest["runtime_registry_sha256"]
        elif manifest["runtime_registry_sha256"] != shared_runtime_registry_sha:
            raise ValueError("generation arms use different runtime registries")
        if manifest["runtime_registry_sha256"] != eligibility["runtime_registry_sha256"]:
            raise ValueError("generation runtime differs from eligible checkpoint runtime")
        method_artifact = manifest["method_artifact"]
        if method == "original":
            if method_artifact != {"kind": "base_model"}:
                raise ValueError("Original method artifact must be the unadapted base model")
        elif method == "v3b":
            if method_artifact != {
                "kind": "lora_checkpoint",
                "path": V3B_CHECKPOINT,
                "sha256": V3B_CHECKPOINT_SHA256,
                "scale": 1.25,
                "step": 200,
            }:
                raise ValueError("v3b generation does not bind the frozen checkpoint")
        else:
            expected_v4 = {
                "kind": "lora_checkpoint",
                "checkpoint_eligibility_path": str(checkpoint_eligibility_path),
                "checkpoint_eligibility_sha256": eligibility_sha,
                "path": eligibility["checkpoint"]["path"],
                "weights_sha256": eligibility["checkpoint"]["weights_sha256"],
                "scale": 1.25,
                "step": 200,
            }
            if method_artifact != expected_v4:
                raise ValueError("v4 generation does not bind the eligible checkpoint")
        videos = manifest["videos"]
        if not isinstance(videos, list) or len(videos) != expected_n:
            raise ValueError(f"{dataset}/{method}: video inventory count mismatch")
        method_videos: dict[str, Path] = {}
        for index, record in enumerate(videos):
            if not isinstance(record, dict) or set(record) != {
                "unit_id",
                "index",
                "path",
                "size_bytes",
                "sha256",
                "prompt_sha256",
                "seed",
                "frame_count",
                "width",
                "height",
                "fps_numerator",
                "fps_denominator",
            }:
                raise ValueError(f"{dataset}/{method}: video record fields are not exact")
            unit_id = str(record["unit_id"])
            unit = by_unit.get(unit_id)
            if unit is None or record["index"] != index:
                raise ValueError(f"{dataset}/{method}: unit order mismatch")
            if record["seed"] != int(unit["seed"]):
                raise ValueError(f"{dataset}/{method}/{unit_id}: seed mismatch")
            if record["prompt_sha256"] != hashlib.sha256(str(unit["prompt"]).encode("utf-8")).hexdigest():
                raise ValueError(f"{dataset}/{method}/{unit_id}: prompt hash mismatch")
            path = resolve_path(project_root, str(record["path"]))
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"{dataset}/{method}/{unit_id}: video missing")
            resolved = path.resolve(strict=True)
            inode = (path.stat().st_dev, path.stat().st_ino)
            digest = file_sha256(path)
            if resolved in all_paths or inode in all_inodes or digest in all_hashes:
                raise ValueError("cross-arm or within-arm video path/inode/content reuse detected")
            all_paths.add(resolved)
            all_inodes.add(inode)
            all_hashes.add(digest)
            decoded = dict(decode(path))
            expected_media = {
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "frame_count": FRAME_COUNT,
                "width": WIDTH,
                "height": HEIGHT,
                "fps_numerator": FPS.numerator,
                "fps_denominator": FPS.denominator,
            }
            actual_media = {field: record[field] for field in expected_media}
            if actual_media != expected_media:
                raise ValueError(f"{dataset}/{method}/{unit_id}: manifest media contract mismatch")
            if any(decoded.get(field) != value for field, value in expected_media.items() if field not in {"size_bytes", "sha256"}):
                raise ValueError(f"{dataset}/{method}/{unit_id}: decoded media contract mismatch")
            method_videos[unit_id] = path
        output[method] = method_videos
    if shared_runtime_registry_sha is None:
        raise ValueError("generation bundle has no runtime registry binding")
    validate_runtime_registry(
        resolve_path(project_root, RUNTIME_REGISTRY), shared_runtime_registry_sha
    )
    return output


def exact_stage_paths(project_root: Path) -> dict[str, Path]:
    """Return registered public paths without probing any private/final data."""

    return {name: resolve_path(project_root, path) for name, path in TRAINING_AUTHORIZATION_REFS.items()}


__all__ = [name for name in globals() if name.isupper()] + [
    "canonical_json_sha256",
    "artifact_sha256",
    "audit_preselection_seed_space",
    "derive_seed",
    "derive_unit_rows",
    "expected_selection_binding",
    "file_sha256",
    "load_normalized_candidate_manifest",
    "model_artifact_inventory",
    "prepare_selection_binding",
    "read_csv",
    "reject_sealed_final36_path",
    "resolve_path",
    "validate_causal_selected_cases",
    "validate_causal_unit_manifest",
    "validate_checkpoint_eligibility",
    "validate_commitment_opening",
    "validate_commitment_registry",
    "validate_gate_registry",
    "validate_generation_bundle",
    "validate_forbidden_seed_inventory",
    "validate_holdout_mapping",
    "validate_public_review_columns",
    "validate_runtime_registry",
    "validate_selection_contract_opening",
    "validate_specificity_selected_cases",
    "validate_specificity_unit_manifest",
    "validate_training_authorization",
    "validate_training_code_registry",
    "write_csv",
]
