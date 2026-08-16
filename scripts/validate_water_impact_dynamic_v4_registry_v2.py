#!/usr/bin/env python3
"""Fail-closed public/private validator for v4 source-slot registry v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "water_impact_dynamic_v4_source_slot_registry_v2"
PROTOCOL = SCHEMA
DATASET_VERSION = "v4_dev72_v2"
CANONICAL_JSON = "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF"
SPLIT_RULE = "rank all 80 curated v2 new-source canonical records by SHA-256 domain bank-holdout-v2 with the private v2 split salt; first 56 are bank and remaining 24 are holdout; ties are invalid"
STRICT_PHYSICAL_STATUS = "strict_physical_pass_v2"
LEGACY_PHYSICAL_STATUS = "legacy_original_source_exempt"
PUBLIC_BANK_NAME = "source_bank_public64_registry_v2.json"
PUBLIC_HOLDOUT_NAME = "holdout_public_commitment_v2.json"
PUBLIC_STAGE0_NAME = "causal_stage0_public_commitment_v2.json"
STANDARD_STAGE0_NAME = "causal_stage0_commitment_v2.json"
TRAIN_REL = Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv")
TEST_REL = Path("data/water_impact_dynamic_v1/test_pairs.csv")
BUILDER_REL = Path("scripts/build_water_impact_dynamic_pairs_v1.py")
EVENT_RISK_ROOTS = (
    "water", "drop", "splash", "ripple", "impact", "fall", "wave", "sink",
    "plunge", "pour", "spray", "collision", "contact", "enter", "entry", "cavity",
)
AMBIGUOUS_SIZE_WORDS = ("small", "miniature", "tiny", "little")
PROHIBITED_SOURCE_CATEGORY_WORDS = (
    "food", "fruit", "vegetable", "produce", "apple", "lime", "berry", "walnut",
    "strawberry", "bread", "candy", "powder", "foam", "sponge", "paper", "cloth",
    "fabric", "leaf", "feather", "cork", "shell", "flake", "pellet", "granule",
)
IMPACT_KEYS = {
    "verdict", "compact_and_rigid", "natural_drop_entry",
    "visible_brief_splash_or_ripple_plausible", "predominantly_buoyant_or_windborne",
    "flexible_or_film_like", "fragile", "powder", "loose_aggregate", "porous",
    "food_or_produce", "negative_buoyancy", "visually_recognizable", "entity_state",
    "material", "density_g_cm3", "mass_g", "dimensions_cm", "size_class",
    "source_specific_feature", "curator_note",
}
STRATA_KEYS = {"origin", "food_status", "shape_class", "color_family", "material_family", "texture_class"}

SUPERSEDES = {
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
        "source_bank_public64_registry_v1.json": "1bb725f66ec2303b32b7a7681c5afe6013c333d908477d1713bf898855ec7177",
        "holdout_public_commitment_v1.json": "3f56fee55ab0b93cd26a3478d73d84f5560e306eb2e9510cd7b47de574a98ce1",
        "causal_stage0_public_commitment_v1.json": "994de60571606d46a016d2f15a41b024c68860940f35e7c002845077affab9c0",
    },
}

CURATION_AUDIT = {
    "status": "strict_new80_pass_after_pre_freeze_revision",
    "strict_physical_scope": {
        "curated_new_source_count": 80,
        "new_bank_source_count": 56,
        "private_holdout_source_count": 24,
    },
    "strict_physical_result": {
        "curated_new80_pass": 80,
        "curated_new80_fail": 0,
        "new_bank56_pass": 56,
        "new_bank56_fail": 0,
        "private_holdout24_pass": 24,
        "private_holdout24_fail": 0,
        "mass_density_bounding_volume_inconsistency_count": 0,
    },
    "legacy_original_source_policy": {
        "status": LEGACY_PHYSICAL_STATUS,
        "count": 8,
        "certificate_policy": "do not fabricate v2 mass, density, dimensions, or negative-buoyancy certificates for historical sources",
        "stage0_scope": "seen_source_new_receiver only; never count legacy original sources as heldout generalization",
        "eligibility_gate": "full 49-frame Original screening requires source_visibility=2, footprint_visibility>=1, receiver>=1, quality>=1, and causal_link=2",
        "specificity_gate": "the matched Original hard-negative must independently satisfy the frozen specificity eligibility rule",
    },
    "pre_freeze_rejected_drafts": [
        {
            "status": "pre_freeze_draft_rejected",
            "reason_code": "solid_mass_exceeded_density_times_bounding_volume",
            "aggregate_audit": {
                "curated_new80_pass": 78,
                "curated_new80_fail": 2,
                "new_bank56_pass": 54,
                "new_bank56_fail": 2,
                "private_holdout24_pass": 24,
                "private_holdout24_fail": 0,
                "maximum_solid_fill_ratio": 1.0728,
            },
            "rejected_public_sha256": {
                "source_bank_public64_registry_v2.json": "cea52e0b4948462825175655e4c893820205d10c51d12ef2943cee7593fa3952",
                "holdout_public_commitment_v2.json": "24b8a7a4fb587c1ecc66b239024a716c47ed77346b0a703a2aef5c1cad3eff5a",
                "causal_stage0_public_commitment_v2.json": "8792133f709a1736c30fdc8172687837f24bf6f1f2616109b56474a7136c1a66",
            },
        }
    ],
}

LEGACY_RECEIVER_CONCEPTS = (
    "teacup", "goblet", "chalice", "flask", "vase", "urn", "amphora", "washpan",
    "pitcher", "carafe", "decanter", "canteen", "thermos", "mug", "kettle", "teapot",
    "cauldron", "wok", "ramekin", "tureen", "crock", "planter", "wheelbarrow", "canoe",
    "dinghy", "brazier", "helmet", "umbrella", "hubcap", "toolbox", "cradle", "hollow",
)

PRIVATE_FILES = {
    "curate_v4_registry_v2.py",
    "source_curation_pool_private_v2.json",
    "source_ontology_private80_v2.json",
    "source_split_private_v2.json",
    "holdout_registry_private24_v2.json",
    "source_history_matrix_private_v2.json",
    "source_receiver_matrix_private_v2.json",
    "source_impact_matrix_private_v2.json",
    "receiver_curation_pool_private_v2.json",
    "receiver_ontology_private32_v2.json",
    "receiver_history_matrix_private_v2.json",
    "salts_private_v2.json",
    "causal_stage0_candidates_private_v2.json",
    "causal_stage0_templates_private_v2.json",
    "causal_stage0_field_rules_private_v2.json",
    "causal_stage0_render_config_private_v2.json",
    "causal_stage0_selection_rules_private_v2.json",
    "causal_stage0_secrets_private_v2.json",
    "causal_stage0_bundle_private_v2.json",
    "registry_private_manifest_v2.json",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_phrase(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def salted_rank(salt: str, domain: str, value: Any) -> str:
    payload = domain.encode("utf-8") + b"\x00" + salt.encode("ascii") + b"\x00" + canonical_bytes(value)
    return sha256_bytes(payload)


def commitment(name: str, secret: str) -> str:
    return sha256_bytes(canonical_bytes({"name": name, "secret": secret}))


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValidationError("required JSON artifact is unreadable") from exc


def load_builder(repo: Path):
    spec = importlib.util.spec_from_file_location("v4_v2_validator_builder", repo / BUILDER_REL)
    require(spec is not None and spec.loader is not None, "canonical builder import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_aggregates(repo: Path, builder):
    train_sources: dict[str, str] = {}
    train_receivers: dict[str, str] = {}
    roles: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    with (repo / TRAIN_REL).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            role = row["training_role"]
            roles[role] += 1
            if role != "erase":
                continue
            source_id = row["source_id"].strip()
            source_phrase = row["source_object"].strip()
            receiver_id = row["receiver_id"].strip()
            receiver_phrase = row["receiver"].strip()
            variant = row["prompt_variant"].strip()
            train_sources[source_id] = source_phrase
            train_receivers[receiver_id] = receiver_phrase
            variants[variant] += 1
            rebuilt = builder.factual_prompt(source_phrase, receiver_phrase, variant)
            require(rebuilt == row["training_prompt"] == row["prompt"], "canonical prompt reconstruction failed")
    require(roles == Counter({"erase": 178, "preserve": 36}), "training role counts changed")
    require(variants == Counter({"natural": 90, "direct": 88}), "training variant counts changed")
    test_sources: dict[str, str] = {}
    test_receivers: dict[str, str] = {}
    # Consume only aggregate source/receiver identity fields from this allowlisted table.
    with (repo / TEST_REL).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            test_sources[row["source_id"].strip()] = row["source_object"].strip()
            test_receivers[row["receiver_id"].strip()] = row["receiver"].strip()
    require(train_sources == dict(builder.TRAIN_SOURCES), "training sources changed")
    require(train_receivers == dict(builder.TRAIN_RECEIVERS), "training receivers changed")
    return train_sources, train_receivers, test_sources, test_receivers


def assignment_contract() -> dict[str, Any]:
    return {
        "algorithm_id": "fixed64_permutation_cycle_partitioned_hash_swap_v2",
        "permutation": {
            "hash": "sha256",
            "payload": "utf8(source_assignment_salt) || 0x00 || utf8('permute-v2') || 0x00 || utf8(source_id)",
            "ordering": "ascending digest over all 64 entries; equal digests are invalid",
            "application": "construct this one fixed 64-item permutation once, then repeat it cyclically by erase ordinal; never reshuffle per cycle",
        },
        "collision_policy": {
            "processing_order": "ascending erase ordinal",
            "partitions": [[0, 100], [100, 178]],
            "candidate_scope": "only the current ordinal's partition: active100 or remaining78",
            "eligibility": "the candidate's assigned source differs from the current row's original source and the current assigned source differs from the candidate row's original source",
            "candidate_rank": "sha256(utf8(source_assignment_salt) || 0x00 || utf8('swap-v2') || 0x00 || utf8(decimal_position) || 0x00 || utf8(decimal_candidate))",
            "selection": "choose the eligible candidate with the smallest digest; equal digests or no eligible candidate are fatal",
            "operation": "swap the two assigned sources and continue; the completed 178-row mapping must have zero original-source collisions",
        },
        "rng_rule": "assignment consumes neither sample-order RNG nor noise/sigma RNG",
        "mapping_commitments": "the active first-100 and full-178 canonical mappings must each be SHA-256 bound after reconstruction of the exact v3b balanced sample schedule",
    }


def aggregate_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = ["origin", "food_status", "shape_class", "color_family", "material_family", "texture_class"]
    result: dict[str, Any] = {"count": len(items)}
    for key in dimensions:
        result[key] = dict(sorted(Counter(str(item[key]) for item in items).items()))
    result["origin_x_food_status"] = dict(sorted(Counter(f"{item['origin']}:{item['food_status']}" for item in items).items()))
    return result


def jaccard(left: str, right: str) -> float:
    a = set(normalize_phrase(left).split())
    b = set(normalize_phrase(right).split())
    return len(a & b) / len(a | b)


def validate_impact(item: dict[str, Any]) -> None:
    audit = item.get("impact_plausibility")
    require(isinstance(audit, dict) and set(audit) == IMPACT_KEYS, "impact audit key set mismatch")
    require(audit["verdict"] == "pass", "impact audit verdict failed")
    for key in ("compact_and_rigid", "natural_drop_entry", "visible_brief_splash_or_ripple_plausible", "negative_buoyancy", "visually_recognizable"):
        require(audit[key] is True, "required impact audit boolean failed")
    for key in ("predominantly_buoyant_or_windborne", "flexible_or_film_like", "fragile", "powder", "loose_aggregate", "porous", "food_or_produce"):
        require(audit[key] is False, "prohibited impact audit category present")
    require(audit["entity_state"] in {"solid_one_piece", "rigid_locked_assembly"}, "entity state invalid")
    require(isinstance(audit["material"], str) and bool(audit["material"].strip()), "material missing")
    require(isinstance(audit["density_g_cm3"], (int, float)) and not isinstance(audit["density_g_cm3"], bool) and 3.0 <= audit["density_g_cm3"] <= 20.0, "density invalid")
    require(isinstance(audit["mass_g"], int) and not isinstance(audit["mass_g"], bool) and 350 <= audit["mass_g"] <= 1200, "mass invalid")
    dims = audit["dimensions_cm"]
    require(isinstance(dims, list) and len(dims) == 3 and all(isinstance(value, (int, float)) and not isinstance(value, bool) and 2.5 <= value <= 15.0 for value in dims) and max(dims) >= 8.0, "dimensions invalid")
    require(float(audit["mass_g"]) <= float(audit["density_g_cm3"]) * math.prod(float(value) for value in dims), "mass exceeds density times bounding volume")
    require(audit["size_class"] == "palm_sized_explicit", "size class invalid")
    require(isinstance(audit["source_specific_feature"], str) and bool(audit["source_specific_feature"].strip()), "source-specific feature missing")
    require(isinstance(audit["curator_note"], str) and bool(audit["curator_note"].strip()), "curator note missing")


def validate_source_identity(item: dict[str, Any]) -> None:
    require(item.get("normalized_phrase") == normalize_phrase(item.get("source_phrase", "")), "source normalization mismatch")
    require(isinstance(item.get("head_lemma"), str) and normalize_phrase(item["head_lemma"]) == item["head_lemma"] and " " not in item["head_lemma"], "source head must be one normalized token")
    require(item["normalized_phrase"].split()[-1] == item["head_lemma"], "source head is not the final token")
    identity = "|".join((item.get("source_id", ""), item.get("source_phrase", ""), item.get("head_lemma", ""))).casefold()
    require(not any(root in identity or root in item["normalized_phrase"] for root in EVENT_RISK_ROOTS), "source identity has literal or subword event risk")
    tokens = set(item["normalized_phrase"].split())
    require(not (tokens & set(AMBIGUOUS_SIZE_WORDS)), "source phrase uses ambiguous size language")
    require(not (tokens & set(PROHIBITED_SOURCE_CATEGORY_WORDS)), "source phrase contains a prohibited risk category")
    require("palm sized" in item["normalized_phrase"] and "dense" in tokens, "source phrase lacks explicit palm size or density")
    validate_impact(item)
    require(item["head_lemma"] in normalize_phrase(item["impact_plausibility"]["curator_note"]).split(), "curator note is not source-specific")


def validate_stat_shape(value: dict[str, Any], expected_count: int) -> None:
    dimensions = ("origin", "food_status", "shape_class", "color_family", "material_family", "texture_class", "origin_x_food_status")
    require(isinstance(value, dict) and set(value) == {"count", *dimensions} and value.get("count") == expected_count, "aggregate strata shape or count mismatch")
    for key in dimensions:
        part = value.get(key)
        require(isinstance(part, dict) and all(isinstance(v, int) and v >= 0 for v in part.values()) and sum(part.values()) == expected_count, "aggregate strata dimension invalid")


def validate_public_objects(repo: Path, staging: Path, bank: dict[str, Any], holdout_public: dict[str, Any], stage0_public: dict[str, Any]) -> dict[str, Any]:
    builder = load_builder(repo)
    train_sources, train_receivers, test_sources, test_receivers = identity_aggregates(repo, builder)
    expected_bank_keys = {
        "schema", "protocol", "dataset_version", "registry", "canonical_json", "supersedes", "curation_audit",
        "counts", "canonical_builder_sha256", "training_manifest_sha256", "bank_entries_sha256",
        "source_assignment_salt", "source_assignment_algorithm", "entries",
    }
    require(set(bank) == expected_bank_keys, "public bank top-level key set mismatch")
    require(bank["schema"] == SCHEMA and bank["protocol"] == PROTOCOL and bank["dataset_version"] == DATASET_VERSION, "public bank version mismatch")
    require(bank["registry"] == "public_augmentation_bank64_v2" and bank["supersedes"] == SUPERSEDES, "public bank identity or supersession mismatch")
    require(bank["curation_audit"] == CURATION_AUDIT, "public bank curation audit mismatch")
    require(bank["canonical_json"] == CANONICAL_JSON, "public bank canonical JSON contract mismatch")
    require(bank["counts"] == {"total": 64, "original_training": 8, "new_ontology": 56}, "public bank counts mismatch")
    require(bank["canonical_builder_sha256"] == sha256_file(repo / BUILDER_REL), "builder hash mismatch")
    require(bank["training_manifest_sha256"] == sha256_file(repo / TRAIN_REL), "training manifest hash mismatch")
    require(is_hex64(bank["source_assignment_salt"]), "public assignment salt shape invalid")
    require(bank["source_assignment_algorithm"] == assignment_contract(), "v2 assignment contract mismatch")
    entries = bank["entries"]
    require(isinstance(entries, list) and len(entries) == 64 and [row.get("bank_index") for row in entries] == list(range(64)), "bank entry count or indices invalid")
    require(bank["bank_entries_sha256"] == sha256_bytes(canonical_bytes(entries)), "bank entry commitment mismatch")
    original = entries[:8]
    new = entries[8:]
    expected_original = [
        {
            "bank_index": index,
            "source_id": source_id,
            "source_phrase": phrase,
            "normalized_phrase": normalize_phrase(phrase),
            "head_lemma": normalize_phrase(phrase).split()[-1],
            "membership": "original_training_source",
            "physical_audit_status": LEGACY_PHYSICAL_STATUS,
        }
        for index, (source_id, phrase) in enumerate(builder.TRAIN_SOURCES)
    ]
    require(original == expected_original, "original eight exception entries changed")
    require(len({row["source_id"] for row in new}) == 56 and len({row["normalized_phrase"] for row in new}) == 56 and len({row["head_lemma"] for row in new}) == 56, "new bank identity uniqueness failed")
    historic_source_phrases = {normalize_phrase(value) for value in {**test_sources, **train_sources}.values()}
    historic_source_heads = {value.split()[-1] for value in historic_source_phrases}
    for row in new:
        require(set(row) == {"bank_index", "source_id", "source_phrase", "normalized_phrase", "head_lemma", "membership", "physical_audit_status", "impact_plausibility", "strata"}, "new bank entry key set mismatch")
        require(row["membership"] == "new_bank_source" and row["physical_audit_status"] == STRICT_PHYSICAL_STATUS and set(row["strata"]) == STRATA_KEYS, "new bank membership, physical status, or strata invalid")
        validate_source_identity(row)
        require(row["normalized_phrase"] not in historic_source_phrases and row["head_lemma"] not in historic_source_heads, "new bank overlaps historical source")
    public_notes = [row["impact_plausibility"]["curator_note"] for row in new]
    public_features = [row["impact_plausibility"]["source_specific_feature"] for row in new]
    require(len(set(public_notes)) == 56 and len(set(public_features)) == 56, "public source-specific audit text is not unique")

    expected_holdout_keys = {
        "schema", "protocol", "dataset_version", "registry", "canonical_json", "supersedes", "curation_audit",
        "holdout_count", "holdout_registry_file_sha256", "split_rule", "split_salt_commitment_sha256",
        "aggregate_strata", "cross_role_checks",
    }
    require(set(holdout_public) == expected_holdout_keys, "holdout top-level key set mismatch")
    require(holdout_public["schema"] == SCHEMA and holdout_public["protocol"] == PROTOCOL and holdout_public["dataset_version"] == DATASET_VERSION, "holdout version mismatch")
    require(holdout_public["registry"] == "public_holdout24_commitment_v2" and holdout_public["supersedes"] == SUPERSEDES, "holdout identity or supersession mismatch")
    require(holdout_public["curation_audit"] == CURATION_AUDIT, "holdout curation audit mismatch")
    require(holdout_public["canonical_json"] == CANONICAL_JSON, "holdout canonical JSON contract mismatch")
    require(holdout_public["holdout_count"] == 24 and is_hex64(holdout_public["holdout_registry_file_sha256"]) and is_hex64(holdout_public["split_salt_commitment_sha256"]), "holdout commitment invalid")
    require(holdout_public["split_rule"] == SPLIT_RULE, "holdout split rule mismatch")
    aggregate = holdout_public["aggregate_strata"]
    require(set(aggregate) == {"curated_new80", "new_bank56", "private_holdout24"}, "aggregate strata partitions invalid")
    validate_stat_shape(aggregate["curated_new80"], 80)
    validate_stat_shape(aggregate["new_bank56"], 56)
    validate_stat_shape(aggregate["private_holdout24"], 24)
    new_stats_rows = [{**row["strata"]} for row in new]
    require(aggregate_stats(new_stats_rows) == aggregate["new_bank56"], "new-bank public strata mismatch")
    cross = holdout_public["cross_role_checks"]
    zero_keys = (
        "normalized_phrase_overlap_count", "normalized_head_overlap_count", "event_literal_or_subword_risk_count",
        "ambiguous_size_language_count", "prohibited_physical_category_count",
        "historical_source_semantic_equivalence_count", "historical_source_near_duplicate_count",
        "source_receiver_semantic_equivalence_count", "source_receiver_near_duplicate_count",
        "new_receiver_historical_semantic_equivalence_count", "new_receiver_historical_near_duplicate_count",
        "impact_plausibility_failure_count",
    )
    require(set(cross) == {*zero_keys, "impact_plausibility_pass_count", "source_specific_note_unique_count", "matrix_scope"}, "public cross-role audit key set mismatch")
    require(all(cross.get(key) == 0 for key in zero_keys), "public cross-role or physical audit failed")
    require(cross.get("impact_plausibility_pass_count") == 80 and cross.get("source_specific_note_unique_count") == 80, "public physical pass or unique note count mismatch")
    require(cross.get("matrix_scope") == {"new_sources": 80, "historical_sources": 14, "new_receivers": 32, "historical_receivers": 52, "complete_receivers": 84}, "public matrix scope mismatch")

    expected_stage0_keys = {
        "schema", "protocol", "dataset_version", "registry", "stage", "status",
        "authorization_status", "canonical_json", "supersedes", "curation_audit", "remaining_blockers",
        "candidate_count", "cell_counts", "candidate_manifest_sha256",
        "canonical_templates_sha256", "field_normalization_sha256",
        "render_configuration_sha256", "selector_rules_sha256",
        "screening_seed_commitment_sha256", "selector_salt_commitment_sha256",
        "evaluation_seed_salt_commitment_sha256", "stage0_bundle_file_sha256",
        "public_metadata",
    }
    require(set(stage0_public) == expected_stage0_keys, "Stage-0 public top-level key set mismatch")
    require(stage0_public["schema"] == SCHEMA and stage0_public["protocol"] == PROTOCOL and stage0_public["dataset_version"] == DATASET_VERSION, "Stage-0 version mismatch")
    require(stage0_public["registry"] == "causal_stage0_public_commitment_v2" and stage0_public["supersedes"] == SUPERSEDES, "Stage-0 identity or supersession mismatch")
    require(stage0_public["curation_audit"] == CURATION_AUDIT, "Stage-0 curation audit mismatch")
    require(stage0_public["stage"] == 0 and stage0_public["canonical_json"] == CANONICAL_JSON, "Stage-0 public stage or canonical JSON contract mismatch")
    require(stage0_public["status"] == "frozen_components_pending_external_bindings" and stage0_public["authorization_status"] == "not_authorized", "Stage-0 authorization boundary invalid")
    require(stage0_public["remaining_blockers"] == [
        "an independent seed auditor must commit the complete forbidden numeric seed inventory and prove disjointness",
        "an independent binder must commit the exact already-frozen v3b path-plus-file-bytes model inventory digest",
    ], "Stage-0 blocker set mismatch")
    require(stage0_public["candidate_count"] == 48, "Stage-0 candidate count mismatch")
    expected_cells = {
        "holdout_source_seen_receiver:direct": 8,
        "holdout_source_seen_receiver:natural": 8,
        "seen_source_new_receiver:direct": 8,
        "seen_source_new_receiver:natural": 8,
        "holdout_source_new_receiver:direct": 8,
        "holdout_source_new_receiver:natural": 8,
    }
    require(stage0_public["cell_counts"] == expected_cells, "Stage-0 cell counts mismatch")
    for key in (
        "candidate_manifest_sha256", "canonical_templates_sha256", "field_normalization_sha256",
        "render_configuration_sha256", "selector_rules_sha256", "screening_seed_commitment_sha256",
        "selector_salt_commitment_sha256", "evaluation_seed_salt_commitment_sha256", "stage0_bundle_file_sha256",
    ):
        require(is_hex64(stage0_public[key]), "Stage-0 digest shape invalid")
    metadata = stage0_public["public_metadata"]
    require(metadata == {
        "groups": ["holdout_source_seen_receiver", "seen_source_new_receiver", "holdout_source_new_receiver"],
        "prompt_variants": ["direct", "natural"],
        "candidates_per_cell": 8,
        "selection_per_cell": 4,
        "selected_case_target": 24,
        "replicates_per_selected_case": 3,
        "evaluation_unit_target": 72,
        "ranking_domain": "causal-selector-v2",
        "evaluation_seed_domain": "causal-eval-seed-v2",
        "screening_seed_namespace": "v4-causal-stage0-screening-v2",
        "evaluation_seed_namespace": "v4-causal-evaluation-v2",
        "screening_arm": "Original_only",
        "full_frame_screening_required": True,
        "no_reserve_queue": True,
        "source_physical_policy": CURATION_AUDIT["legacy_original_source_policy"],
    }, "Stage-0 public metadata mismatch")
    require(not (staging / STANDARD_STAGE0_NAME).exists(), "authorizing Stage-0 wrapper exists before external bindings")
    return {"bank_count": 64, "new_bank_count": 56, "holdout_count": 24, "stage0_candidate_count": 48}


def validate_public(repo: Path, staging: Path) -> dict[str, Any]:
    return validate_public_objects(
        repo,
        staging,
        load_json(staging / PUBLIC_BANK_NAME),
        load_json(staging / PUBLIC_HOLDOUT_NAME),
        load_json(staging / PUBLIC_STAGE0_NAME),
    )


def selection_feasible(rows: list[dict[str, Any]]) -> bool:
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault(f"{row['group']}:{row['prompt_variant']}", []).append(row)
    order = [
        "holdout_source_seen_receiver:direct", "holdout_source_seen_receiver:natural",
        "seen_source_new_receiver:direct", "seen_source_new_receiver:natural",
        "holdout_source_new_receiver:direct", "holdout_source_new_receiver:natural",
    ]

    def search(index: int, hold_heads: set[str], original_ids: set[str], seen_receiver_ids: set[str], new_receiver_ids: set[str]) -> bool:
        if index == len(order):
            return len(hold_heads) == 16 and len(original_ids) == 8 and len(seen_receiver_ids) == 8 and len(new_receiver_ids) == 16
        cell_name = order[index]
        for combo in itertools.combinations(cells[cell_name], 4):
            next_hold = set(hold_heads)
            next_original = set(original_ids)
            next_seen_receivers = set(seen_receiver_ids)
            next_new_receivers = set(new_receiver_ids)
            valid = True
            for row in combo:
                if row["source_membership"] == "holdout_source":
                    if row["source_head_lemma"] in next_hold:
                        valid = False
                        break
                    next_hold.add(row["source_head_lemma"])
                else:
                    if row["source_id"] in next_original:
                        valid = False
                        break
                    next_original.add(row["source_id"])
                if row["receiver_membership"] == "seen_receiver":
                    if row["receiver_id"] in next_seen_receivers:
                        valid = False
                        break
                    next_seen_receivers.add(row["receiver_id"])
                else:
                    if row["receiver_id"] in next_new_receivers:
                        valid = False
                        break
                    next_new_receivers.add(row["receiver_id"])
            if valid and search(index + 1, next_hold, next_original, next_seen_receivers, next_new_receivers):
                return True
        return False

    return search(0, set(), set(), set(), set())


def validate_private(repo: Path, staging: Path, private: Path) -> dict[str, Any]:
    public_summary = validate_public(repo, staging)
    require(private.is_dir(), "private v2 directory missing")
    actual = {path.name for path in private.iterdir() if path.is_file()}
    require(actual == PRIVATE_FILES and not any(path.is_dir() for path in private.iterdir()), "private v2 file inventory mismatch")
    manifest = load_json(private / "registry_private_manifest_v2.json")
    require(manifest["schema"] == SCHEMA and manifest["protocol"] == PROTOCOL and manifest["dataset_version"] == DATASET_VERSION and manifest["manifest"] == "private_clean_registry_manifest_v2", "private manifest identity mismatch")
    component_names = PRIVATE_FILES - {"curate_v4_registry_v2.py", "registry_private_manifest_v2.json"}
    require(set(manifest["private_components"]) == component_names, "private component set mismatch")
    for name in component_names:
        require(manifest["private_components"][name] == sha256_file(private / name), "private component hash mismatch")
    require(manifest["generator_sha256"] == sha256_file(private / "curate_v4_registry_v2.py"), "private curator hash mismatch")
    expected_public_hashes = {
        PUBLIC_BANK_NAME: sha256_file(staging / PUBLIC_BANK_NAME),
        PUBLIC_HOLDOUT_NAME: sha256_file(staging / PUBLIC_HOLDOUT_NAME),
        PUBLIC_STAGE0_NAME: sha256_file(staging / PUBLIC_STAGE0_NAME),
    }
    require(manifest["public_staging_files"] == expected_public_hashes, "private/public staging binding mismatch")
    expected_inputs = {
        "train_dynamic_sft_preserve_v2.csv": sha256_file(repo / TRAIN_REL),
        "test_pairs.csv": sha256_file(repo / TEST_REL),
        "build_water_impact_dynamic_pairs_v1.py": sha256_file(repo / BUILDER_REL),
    }
    require(manifest["allowlisted_inputs"] == expected_inputs, "allowlisted input hash mismatch")

    bank = load_json(staging / PUBLIC_BANK_NAME)
    holdout_public = load_json(staging / PUBLIC_HOLDOUT_NAME)
    stage0_public = load_json(staging / PUBLIC_STAGE0_NAME)
    salts = load_json(private / "salts_private_v2.json")
    salt_keys = {
        "source_ontology_salt", "source_split_salt", "receiver_ontology_salt",
        "causal_stage0_selector_salt", "causal_evaluation_seed_salt", "causal_screening_seed_token",
    }
    require(set(salts) == {"schema", "protocol", "dataset_version", *salt_keys}, "v2 salt registry key set mismatch")
    require(salts["schema"] == SCHEMA and salts["protocol"] == PROTOCOL and salts["dataset_version"] == DATASET_VERSION, "v2 salt registry identity mismatch")
    require(all(is_hex64(salts[key]) for key in salt_keys), "v2 salt shape invalid")
    require(len({salts[key] for key in salt_keys}) == len(salt_keys), "private v2 salts are not mutually distinct")
    require(bank["source_assignment_salt"] not in {salts[key] for key in salt_keys}, "public assignment salt reuses private salt")
    require(holdout_public["split_salt_commitment_sha256"] == commitment("source_split_salt_v2", salts["source_split_salt"]), "v2 split salt commitment mismatch")

    pool_doc = load_json(private / "source_curation_pool_private_v2.json")
    ontology_doc = load_json(private / "source_ontology_private80_v2.json")
    pool = pool_doc["candidates"]
    ontology = ontology_doc["sources"]
    require(pool_doc["candidate_count"] == 100 and len(pool) == 100, "source pool count mismatch")
    require(ontology_doc["source_count"] == 80 and len(ontology) == 80, "ontology count mismatch")
    for collection, expected_count in ((pool, 100), (ontology, 80)):
        require(len({row["source_id"] for row in collection}) == expected_count and len({row["normalized_phrase"] for row in collection}) == expected_count and len({row["head_lemma"] for row in collection}) == expected_count, "private source identity uniqueness failed")
        for row in collection:
            validate_source_identity(row)
    notes = [row["impact_plausibility"]["curator_note"] for row in ontology]
    features = [row["impact_plausibility"]["source_specific_feature"] for row in ontology]
    require(len(set(notes)) == 80 and len(set(features)) == 80, "ontology source-specific notes or features are not unique")
    require(max(jaccard(notes[i], notes[j]) for i in range(80) for j in range(i + 1, 80)) < 0.78, "ontology curator notes are boilerplate-similar")
    expected_ontology: list[dict[str, Any]] = []
    for stratum in ("machined_steel", "cast_iron", "brass_bronze", "dense_alloy"):
        cell = [row for row in pool if row["curation_stratum"] == stratum]
        require(len(cell) == 25, "source curation stratum size mismatch")
        expected_ontology.extend(sorted(cell, key=lambda row: salted_rank(salts["source_ontology_salt"], "source-ontology-v2", row))[:20])
    expected_ontology.sort(key=lambda row: row["source_id"])
    require(ontology == expected_ontology, "v2 ontology selection is not deterministic")

    builder = load_builder(repo)
    train_sources, train_receivers, test_sources, test_receivers = identity_aggregates(repo, builder)
    historic_source_map = {**test_sources, **train_sources}
    require(len(historic_source_map) == 14, "historical source identity count mismatch")
    historic_source_phrases = {normalize_phrase(value) for value in historic_source_map.values()}
    historic_source_heads = {value.split()[-1] for value in historic_source_phrases}
    require(all(row["normalized_phrase"] not in historic_source_phrases and row["head_lemma"] not in historic_source_heads for row in ontology), "ontology overlaps historical sources")

    split_doc = load_json(private / "source_split_private_v2.json")
    require(split_doc["domain"] == "bank-holdout-v2" and len(split_doc["rows"]) == 80, "v2 split domain or count mismatch")
    split_by_id = {row["source_id"]: row for row in split_doc["rows"]}
    ranked = sorted(ontology, key=lambda row: salted_rank(salts["source_split_salt"], "bank-holdout-v2", row))
    bank_ids = {row["source_id"] for row in ranked[:56]}
    holdout_ids = {row["source_id"] for row in ranked[56:]}
    for row in ontology:
        expected_rank = salted_rank(salts["source_split_salt"], "bank-holdout-v2", row)
        expected_membership = "new_bank_source" if row["source_id"] in bank_ids else "holdout_source"
        require(split_by_id[row["source_id"]] == {"source_id": row["source_id"], "membership": expected_membership, "split_rank_sha256": expected_rank}, "v2 split row mismatch")
    public_new_ids = {row["source_id"] for row in bank["entries"][8:]}
    require(public_new_ids == bank_ids and public_new_ids.isdisjoint(holdout_ids), "public bank/private split mismatch")
    by_id = {row["source_id"]: row for row in ontology}
    new_bank_items = [by_id[key] for key in sorted(bank_ids)]
    holdout_items = [by_id[key] for key in sorted(holdout_ids)]
    aggregate = holdout_public["aggregate_strata"]
    require(aggregate_stats(ontology) == aggregate["curated_new80"] and aggregate_stats(new_bank_items) == aggregate["new_bank56"] and aggregate_stats(holdout_items) == aggregate["private_holdout24"], "private/public aggregate strata mismatch")
    holdout_path = private / "holdout_registry_private24_v2.json"
    holdout_registry = load_json(holdout_path)
    holdout_entries = holdout_registry["entries"]
    require(len(holdout_entries) == 24 and [row["source_id"] for row in holdout_entries] == sorted(holdout_ids), "private holdout registry invalid")
    require(sha256_file(holdout_path) == holdout_public["holdout_registry_file_sha256"], "holdout byte commitment mismatch")
    require(all(row["impact_plausibility"]["verdict"] == "pass" for row in holdout_entries), "private holdout contains failed source")

    receiver_pool_doc = load_json(private / "receiver_curation_pool_private_v2.json")
    receiver_doc = load_json(private / "receiver_ontology_private32_v2.json")
    receiver_pool_rows = receiver_pool_doc["candidates"]
    selected_receivers = receiver_doc["receivers"]
    require(receiver_pool_doc["candidate_count"] == 40 and len(receiver_pool_rows) == 40, "receiver pool count mismatch")
    require(receiver_doc["receiver_count"] == 32 and receiver_doc["historical_receiver_blacklist_count"] == 52 and len(selected_receivers) == 32, "receiver ontology count mismatch")
    expected_receivers = sorted(sorted(receiver_pool_rows, key=lambda row: salted_rank(salts["receiver_ontology_salt"], "receiver-ontology-v2", row))[:32], key=lambda row: row["receiver_id"])
    require(selected_receivers == expected_receivers, "receiver ontology selection is not deterministic")
    require(len({row["receiver_id"] for row in selected_receivers}) == 32 and len({row["head_lemma"] for row in selected_receivers}) == 32, "receiver identity uniqueness failed")
    expected_receiver_keys = {
        "receiver_id", "receiver_phrase", "normalized_phrase", "head_lemma",
        "receiver_type", "curator_note", "curator",
    }
    for row in selected_receivers:
        require(set(row) == expected_receiver_keys, "receiver ontology row key set mismatch")
        require(row["normalized_phrase"] == normalize_phrase(row["receiver_phrase"]), "receiver normalization mismatch")
        require(normalize_phrase(row["head_lemma"]) == row["head_lemma"] and " " not in row["head_lemma"], "receiver head must be one normalized token")
        require(isinstance(row["curator_note"], str) and row["head_lemma"] in normalize_phrase(row["curator_note"]).split(), "receiver curator note is not identity-specific")
    actual_history_receivers = {**test_receivers, **train_receivers}
    require(len(actual_history_receivers) == 20 and len(LEGACY_RECEIVER_CONCEPTS) == 32, "historical receiver scope mismatch")
    historical_receiver_ids = set(actual_history_receivers) | {f"pre_v2_receiver_{head}" for head in LEGACY_RECEIVER_CONCEPTS}
    selected_receiver_ids = {row["receiver_id"] for row in selected_receivers}
    require(selected_receiver_ids.isdisjoint(historical_receiver_ids), "new receiver ID overlaps history")
    historical_receiver_phrases = {normalize_phrase(value) for value in actual_history_receivers.values()} | {normalize_phrase(head) for head in LEGACY_RECEIVER_CONCEPTS}
    historical_receiver_heads = {receiver_id.split("_")[-1] for receiver_id in actual_history_receivers} | set(LEGACY_RECEIVER_CONCEPTS)
    require(all(row["normalized_phrase"] not in historical_receiver_phrases and row["head_lemma"] not in historical_receiver_heads for row in selected_receivers), "new receiver phrase or head overlaps history")
    require({row["head_lemma"] for row in ontology}.isdisjoint({row["head_lemma"] for row in selected_receivers}), "new source and receiver heads overlap")

    source_history_rows = load_json(private / "source_history_matrix_private_v2.json")["rows"]
    expected_source_history_pairs = {(source_id, historic_id) for source_id in by_id for historic_id in historic_source_map}
    require(
        len(source_history_rows) == 80 * 14
        and all(set(row) == {"new_source_id", "historical_source_id", "semantic_equivalent", "obvious_near_duplicate", "curator_note"} for row in source_history_rows)
        and {(row["new_source_id"], row["historical_source_id"]) for row in source_history_rows} == expected_source_history_pairs
        and all(row["semantic_equivalent"] is False and row["obvious_near_duplicate"] is False and isinstance(row["curator_note"], str) and bool(row["curator_note"].strip()) for row in source_history_rows),
        "source/history semantic matrix invalid",
    )
    receiver_history_rows = load_json(private / "receiver_history_matrix_private_v2.json")["rows"]
    expected_receiver_history_pairs = {(receiver_id, historic_id) for receiver_id in selected_receiver_ids for historic_id in historical_receiver_ids}
    require(
        len(receiver_history_rows) == 32 * 52
        and all(set(row) == {"new_receiver_id", "historical_receiver_id", "semantic_equivalent", "obvious_near_duplicate", "curator_note"} for row in receiver_history_rows)
        and {(row["new_receiver_id"], row["historical_receiver_id"]) for row in receiver_history_rows} == expected_receiver_history_pairs
        and all(row["semantic_equivalent"] is False and row["obvious_near_duplicate"] is False and isinstance(row["curator_note"], str) and bool(row["curator_note"].strip()) for row in receiver_history_rows),
        "receiver/history semantic matrix invalid",
    )
    source_receiver_rows = load_json(private / "source_receiver_matrix_private_v2.json")["rows"]
    complete_receiver_ids = historical_receiver_ids | selected_receiver_ids
    expected_source_receiver_pairs = {(source_id, receiver_id) for source_id in by_id for receiver_id in complete_receiver_ids}
    require(
        len(source_receiver_rows) == 80 * 84
        and all(set(row) == {"new_source_id", "receiver_id", "semantic_equivalent", "obvious_near_duplicate", "curator_note"} for row in source_receiver_rows)
        and {(row["new_source_id"], row["receiver_id"]) for row in source_receiver_rows} == expected_source_receiver_pairs
        and all(row["semantic_equivalent"] is False and row["obvious_near_duplicate"] is False and isinstance(row["curator_note"], str) and bool(row["curator_note"].strip()) for row in source_receiver_rows),
        "source/receiver semantic matrix invalid",
    )
    impact_rows = load_json(private / "source_impact_matrix_private_v2.json")["rows"]
    expected_impact_rows = [{"source_id": row["source_id"], **row["impact_plausibility"]} for row in ontology]
    require(impact_rows == expected_impact_rows, "source impact matrix invalid")

    candidates_doc = load_json(private / "causal_stage0_candidates_private_v2.json")
    candidates = candidates_doc["candidates"]
    require(set(candidates_doc) == {"schema", "protocol", "dataset_version", "stage", "candidate_count", "candidates"}, "Stage-0 candidate manifest key set invalid")
    require(candidates_doc["schema"] == SCHEMA and candidates_doc["protocol"] == PROTOCOL and candidates_doc["dataset_version"] == DATASET_VERSION and candidates_doc["stage"] == 0, "Stage-0 candidate manifest identity invalid")
    require(candidates_doc["candidate_count"] == 48 and len(candidates) == 48 and len({row["case_id"] for row in candidates}) == 48, "Stage-0 candidate identity count invalid")
    require([row["case_id"] for row in candidates] == sorted(row["case_id"] for row in candidates) and all(row["case_id"].startswith("v4v2c_") for row in candidates), "Stage-0 candidate ordering or v2 ID namespace invalid")
    expected_cell_counts = stage0_public["cell_counts"]
    require(dict(sorted(Counter(f"{row['group']}:{row['prompt_variant']}" for row in candidates).items())) == expected_cell_counts, "Stage-0 private cell counts mismatch")
    holdout_candidate_heads = set()
    selected_receivers_by_id = {row["receiver_id"]: row for row in selected_receivers}
    expected_candidate_keys = {
        "case_id", "group", "prompt_variant", "source_membership", "source_id",
        "source_physical_audit_status", "source_phrase", "source_head_lemma", "receiver_membership", "receiver_id",
        "receiver_phrase", "canonical_prompt", "canonical_record_sha256",
    }
    for row in candidates:
        require(set(row) == expected_candidate_keys, "Stage-0 candidate row key set invalid")
        require(row["prompt_variant"] in {"direct", "natural"}, "Stage-0 prompt variant invalid")
        require(row["group"] in {"holdout_source_seen_receiver", "seen_source_new_receiver", "holdout_source_new_receiver"}, "Stage-0 group invalid")
        expected_source_membership = "original_source" if row["group"] == "seen_source_new_receiver" else "holdout_source"
        expected_receiver_membership = "seen_receiver" if row["group"] == "holdout_source_seen_receiver" else "new_receiver"
        require(row["source_membership"] == expected_source_membership and row["receiver_membership"] == expected_receiver_membership, "Stage-0 group membership binding invalid")
        expected_physical_status = LEGACY_PHYSICAL_STATUS if row["source_membership"] == "original_source" else STRICT_PHYSICAL_STATUS
        require(row["source_physical_audit_status"] == expected_physical_status, "Stage-0 source physical status mismatch")
        base = dict(row)
        record_hash = base.pop("canonical_record_sha256")
        require(record_hash == sha256_bytes(canonical_bytes(base)), "Stage-0 record hash mismatch")
        require(row["canonical_prompt"] == builder.factual_prompt(row["source_phrase"], row["receiver_phrase"], row["prompt_variant"]), "Stage-0 canonical prompt mismatch")
        if row["source_membership"] == "holdout_source":
            require(row["source_id"] in holdout_ids and by_id[row["source_id"]]["impact_plausibility"]["verdict"] == "pass", "Stage-0 holdout source binding invalid")
            source_record = by_id[row["source_id"]]
            require(row["source_phrase"] == source_record["source_phrase"] and row["source_head_lemma"] == source_record["head_lemma"], "Stage-0 holdout source fields do not bind to registry")
            holdout_candidate_heads.add(row["source_head_lemma"])
        else:
            require(row["source_id"] in train_sources, "Stage-0 original source binding invalid")
            require(row["source_phrase"] == train_sources[row["source_id"]] and row["source_head_lemma"] == normalize_phrase(train_sources[row["source_id"]]).split()[-1], "Stage-0 original source fields do not bind to training ontology")
        if row["receiver_membership"] == "new_receiver":
            require(row["receiver_id"] in selected_receiver_ids, "Stage-0 new receiver binding invalid")
            receiver_record = selected_receivers_by_id[row["receiver_id"]]
            require(row["receiver_phrase"] == receiver_record["receiver_phrase"], "Stage-0 new receiver phrase does not bind to ontology")
        else:
            require(row["receiver_id"] in train_receivers, "Stage-0 seen receiver binding invalid")
            require(row["receiver_phrase"] == train_receivers[row["receiver_id"]], "Stage-0 seen receiver phrase does not bind to training ontology")
    require(len(holdout_candidate_heads) == 24, "Stage-0 holdout candidate heads do not cover all 24 valid identities")
    require(selection_feasible(candidates), "Stage-0 global 16-head constrained selection is infeasible")

    templates = load_json(private / "causal_stage0_templates_private_v2.json")
    expected_templates = {variant: builder.factual_prompt("{source_phrase}", "{receiver_phrase}", variant) for variant in ("direct", "natural")}
    require(templates["prompt_templates"] == expected_templates and templates["template_fill_rules"] == {
        "direct": {"source_phrase": "python_str_capitalize", "receiver_phrase": "identity"},
        "natural": {"source_phrase": "identity", "receiver_phrase": "identity"},
    }, "Stage-0 template interface mismatch")
    for row in candidates:
        source_value = row["source_phrase"].capitalize() if row["prompt_variant"] == "direct" else row["source_phrase"]
        rendered = templates["prompt_templates"][row["prompt_variant"]].format(source_phrase=source_value, receiver_phrase=row["receiver_phrase"])
        require(rendered == row["canonical_prompt"], "Stage-0 template fill mismatch")
    rules = load_json(private / "causal_stage0_selection_rules_private_v2.json")
    require(rules["ranking_domain"] == "causal-selector-v2" and rules["evaluation_seed_domain"] == "causal-eval-seed-v2", "private Stage-0 domain mismatch")
    require("causal-selector-v2" in rules["ranking_formula"] and "causal-eval-seed-v2" in rules["evaluation_seed_formula"], "private Stage-0 formula mismatch")
    stage0_secrets = load_json(private / "causal_stage0_secrets_private_v2.json")
    require(stage0_secrets["screening_seed_namespace"] == "v4-causal-stage0-screening-v2" and stage0_secrets["evaluation_seed_namespace"] == "v4-causal-evaluation-v2", "private Stage-0 namespace mismatch")
    require(stage0_secrets["selector_salt"] == salts["causal_stage0_selector_salt"] and stage0_secrets["evaluation_seed_salt"] == salts["causal_evaluation_seed_salt"], "private Stage-0 salt binding mismatch")
    require(stage0_public["screening_seed_commitment_sha256"] == commitment("causal_screening_seed_v2", str(stage0_secrets["screening_seed"])), "screening seed commitment mismatch")
    require(stage0_public["selector_salt_commitment_sha256"] == commitment("causal_stage0_selector_salt_v2", stage0_secrets["selector_salt"]), "selector salt commitment mismatch")
    require(stage0_public["evaluation_seed_salt_commitment_sha256"] == commitment("causal_evaluation_seed_salt_v2", stage0_secrets["evaluation_seed_salt"]), "evaluation seed salt commitment mismatch")
    components = manifest["private_components"]
    for public_key, private_name in (
        ("candidate_manifest_sha256", "causal_stage0_candidates_private_v2.json"),
        ("canonical_templates_sha256", "causal_stage0_templates_private_v2.json"),
        ("field_normalization_sha256", "causal_stage0_field_rules_private_v2.json"),
        ("render_configuration_sha256", "causal_stage0_render_config_private_v2.json"),
        ("selector_rules_sha256", "causal_stage0_selection_rules_private_v2.json"),
        ("stage0_bundle_file_sha256", "causal_stage0_bundle_private_v2.json"),
    ):
        require(stage0_public[public_key] == components[private_name], "Stage-0 component commitment mismatch")
    bundle = load_json(private / "causal_stage0_bundle_private_v2.json")
    require(bundle["dataset_version"] == DATASET_VERSION and bundle["status"] == "frozen_components_pending_external_bindings", "Stage-0 private bundle status invalid")
    require(bundle["source_bank_entries_sha256"] == bank["bank_entries_sha256"] and bundle["holdout_registry_file_sha256"] == holdout_public["holdout_registry_file_sha256"], "Stage-0 source registry binding mismatch")

    public_text = "\n".join((staging / name).read_text(encoding="utf-8") for name in (PUBLIC_BANK_NAME, PUBLIC_HOLDOUT_NAME, PUBLIC_STAGE0_NAME))
    require(all(salts[key] not in public_text for key in salt_keys), "private v2 salt leaked to public staging")
    require(all(row["source_id"] not in public_text and row["source_phrase"] not in public_text and row["normalized_phrase"] not in public_text for row in holdout_entries), "private v2 holdout identity leaked to public staging")

    return {
        **public_summary,
        "ontology_count": 80,
        "source_pool_count": 100,
        "impact_pass_count": 80,
        "holdout_impact_pass_count": 24,
        "new_receiver_count": 32,
        "historical_receiver_screen_count": 52,
        "stage0_valid_holdout_head_count": 24,
        "stage0_global_selection_feasible": True,
        "private_file_count": 20,
        "private_manifest_sha256": sha256_file(private / "registry_private_manifest_v2.json"),
        "stage0_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=default_repo / "data" / "water_impact_dynamic_v4",
    )
    parser.add_argument("--private-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.private_dir is None:
            summary = validate_public(args.repo_root.resolve(), args.staging_dir.resolve())
            scope = "public_v2_staging"
        else:
            summary = validate_private(args.repo_root.resolve(), args.staging_dir.resolve(), args.private_dir.resolve())
            scope = "public_v2_staging_and_private_v2"
    except ValidationError as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", "scope": scope, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
