#!/usr/bin/env python3
"""Build the public, deterministic v4 source-slot assignment registry.

This tool consumes only the 64-item public augmentation bank plus the public
holdout and Stage-0 commitments.  It must never be pointed at, derive, or
enumerate the evaluator's private holdout registry.  The output binds the
exact v3b balanced schedule without touching the noise/sigma RNG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from build_water_impact_dynamic_pairs_v1 import (
    TEST_RECEIVERS,
    TEST_SOURCES,
    TRAIN_RECEIVERS,
    TRAIN_SOURCES,
    factual_prompt,
)


PROTOCOL = "water_impact_dynamic_v4_source_mapping_v2"
DATASET_VERSION = "v4_dev72_v2"
BANK_SCHEMA = "water_impact_dynamic_v4_source_slot_registry_v2"
BANK_REGISTRY = "public_augmentation_bank64_v2"
HOLDOUT_REGISTRY = "public_holdout24_commitment_v2"
STAGE0_PUBLIC_REGISTRY = "causal_stage0_public_commitment_v2"
EXPECTED_MANIFEST_SHA256 = (
    "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
)
EXPECTED_ROWS = 214
EXPECTED_ERASE_ROWS = 178
EXPECTED_PRESERVE_ROWS = 36
EXPECTED_ACTIVE_ERASE_ROWS = 100
EXPECTED_BANK_SIZE = 64
EXPECTED_ORIGINAL_BANK_SIZE = 8
EXPECTED_NEW_BANK_SIZE = 56
EXPECTED_SEED = 26000
STRICT_PHYSICAL_STATUS = "strict_physical_pass_v2"
LEGACY_PHYSICAL_STATUS = "legacy_original_source_exempt"
EXPECTED_HOLDOUT_SPLIT_RULE = (
    "rank all 80 curated v2 new-source canonical records by SHA-256 domain "
    "bank-holdout-v2 with the private v2 split salt; first 56 are bank and remaining "
    "24 are holdout; ties are invalid"
)
STRATA_FIELDS = (
    "origin",
    "food_status",
    "shape_class",
    "color_family",
    "material_family",
    "texture_class",
)
EXPECTED_SAMPLE_ORDER_SHA256 = (
    "a0c819a58fbf3f5be184f7b7708d30134d2f9dc0ebc98d79415f19d1cacc87cb"
)
PROMPT_BUILDER_PATH = Path("scripts/build_water_impact_dynamic_pairs_v1.py")
PROMPT_BUILDER_FILE = Path(__file__).resolve().with_name(
    "build_water_impact_dynamic_pairs_v1.py"
)
EXPECTED_SOURCE_ASSIGNMENT_ALGORITHM = {
    "algorithm_id": "fixed64_permutation_cycle_partitioned_hash_swap_v2",
    "permutation": {
        "hash": "sha256",
        "payload": (
            "utf8(source_assignment_salt) || 0x00 || utf8('permute-v2') || 0x00 || "
            "utf8(source_id)"
        ),
        "ordering": "ascending digest over all 64 entries; equal digests are invalid",
        "application": (
            "construct this one fixed 64-item permutation once, then repeat it cyclically "
            "by erase ordinal; never reshuffle per cycle"
        ),
    },
    "collision_policy": {
        "partitions": [[0, 100], [100, 178]],
        "processing_order": "ascending erase ordinal",
        "candidate_scope": "only the current ordinal's partition: active100 or remaining78",
        "eligibility": (
            "the candidate's assigned source differs from the current row's original source "
            "and the current assigned source differs from the candidate row's original source"
        ),
        "candidate_rank": (
            "sha256(utf8(source_assignment_salt) || 0x00 || utf8('swap-v2') || 0x00 || "
            "utf8(decimal_position) || 0x00 || utf8(decimal_candidate))"
        ),
        "selection": (
            "choose the eligible candidate with the smallest digest; equal digests or no "
            "eligible candidate are fatal"
        ),
        "operation": (
            "swap the two assigned sources and continue; the completed 178-row mapping must "
            "have zero original-source collisions"
        ),
    },
    "mapping_commitments": (
        "the active first-100 and full-178 canonical mappings must each be SHA-256 bound "
        "after reconstruction of the exact v3b balanced sample schedule"
    ),
    "rng_rule": "assignment consumes neither sample-order RNG nor noise/sigma RNG",
}
EXPECTED_V2_SUPERSEDES = {
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
EXPECTED_CURATION_AUDIT = {
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
        "certificate_policy": (
            "do not fabricate v2 mass, density, dimensions, or negative-buoyancy "
            "certificates for historical sources"
        ),
        "stage0_scope": (
            "seen_source_new_receiver only; never count legacy original sources as "
            "heldout generalization"
        ),
        "eligibility_gate": (
            "full 49-frame Original screening requires source_visibility=2, "
            "footprint_visibility>=1, receiver>=1, quality>=1, and causal_link=2"
        ),
        "specificity_gate": (
            "the matched Original hard-negative must independently satisfy the frozen "
            "specificity eligibility rule"
        ),
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
                "source_bank_public64_registry_v2.json": (
                    "cea52e0b4948462825175655e4c893820205d10c51d12ef2943cee7593fa3952"
                ),
                "holdout_public_commitment_v2.json": (
                    "24b8a7a4fb587c1ecc66b239024a716c47ed77346b0a703a2aef5c1cad3eff5a"
                ),
                "causal_stage0_public_commitment_v2.json": (
                    "8792133f709a1736c30fdc8172687837f24bf6f1f2616109b56474a7136c1a66"
                ),
            },
        }
    ],
}
EXPECTED_STAGE0_BLOCKERS = [
    "an independent seed auditor must commit the complete forbidden numeric seed inventory and prove disjointness",
    "an independent binder must commit the exact already-frozen v3b path-plus-file-bytes model inventory digest",
]
EXPECTED_STAGE0_PUBLIC_METADATA = {
    "candidates_per_cell": 8,
    "evaluation_seed_domain": "causal-eval-seed-v2",
    "evaluation_seed_namespace": "v4-causal-evaluation-v2",
    "evaluation_unit_target": 72,
    "full_frame_screening_required": True,
    "groups": [
        "holdout_source_seen_receiver",
        "seen_source_new_receiver",
        "holdout_source_new_receiver",
    ],
    "no_reserve_queue": True,
    "prompt_variants": ["direct", "natural"],
    "ranking_domain": "causal-selector-v2",
    "replicates_per_selected_case": 3,
    "screening_arm": "Original_only",
    "screening_seed_namespace": "v4-causal-stage0-screening-v2",
    "selected_case_target": 24,
    "selection_per_cell": 4,
    "source_physical_policy": EXPECTED_CURATION_AUDIT[
        "legacy_original_source_policy"
    ],
}

PRIVATE_KEY_FRAGMENTS = (
    "all_sources",
    "candidate_sources",
    "holdout_sources",
    "holdout_phrases",
    "private_path",
    "private_registry",
    "split_salt",
)
FORBIDDEN_SOURCE_WORDS = {
    "collision",
    "impact",
    "ripple",
    "ripples",
    "splash",
    "splashes",
    "water",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_lf_sha256(value: Any) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for field in STRATA_FIELDS:
        result[field] = dict(
            sorted(Counter(str(row[field]) for row in rows).items())
        )
    result["origin_x_food_status"] = dict(
        sorted(
            Counter(
                f"{row['origin']}:{row['food_status']}" for row in rows
            ).items()
        )
    )
    return result


def validate_aggregate_strata(value: Any, expected_count: int) -> None:
    dimensions = {*STRATA_FIELDS, "origin_x_food_status"}
    if (
        not isinstance(value, dict)
        or set(value) != {"count", *dimensions}
        or value.get("count") != expected_count
    ):
        raise ValueError("public holdout aggregate strata shape/count mismatch")
    for field in dimensions:
        counts = value[field]
        if (
            not isinstance(counts, dict)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for key, count in counts.items()
            )
            or sum(counts.values()) != expected_count
        ):
            raise ValueError(
                f"public holdout aggregate strata dimension is invalid: {field}"
            )


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} must be a frozen lowercase SHA-256, got {value!r}")
    return value


def _reject_private_payload(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(fragment in normalized for fragment in PRIVATE_KEY_FRAGMENTS):
                raise ValueError(
                    f"public source-bank registry exposes forbidden private field {path}.{key}"
                )
            _reject_private_payload(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_private_payload(child, f"{path}[{index}]")


def validate_public_bank_registry(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate the implementer-visible bank without following private refs."""

    require_sha256(expected_sha256, "source-bank registry hash")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"public source-bank registry is missing or symlinked: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"source-bank registry hash mismatch: {actual_sha256} != {expected_sha256}"
        )
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid public source-bank registry: {exc}") from exc
    if not isinstance(registry, dict):
        raise ValueError("public source-bank registry must be a JSON object")
    _reject_private_payload(registry)
    exact_fields = {
        "schema",
        "protocol",
        "registry",
        "dataset_version",
        "canonical_json",
        "supersedes",
        "curation_audit",
        "canonical_builder_sha256",
        "training_manifest_sha256",
        "counts",
        "bank_entries_sha256",
        "source_assignment_salt",
        "source_assignment_algorithm",
        "entries",
    }
    if set(registry) != exact_fields:
        raise ValueError(f"public source-bank fields are not exact: {sorted(registry)}")
    if (
        registry["schema"] != BANK_SCHEMA
        or registry["protocol"] != BANK_SCHEMA
        or registry["registry"] != BANK_REGISTRY
        or registry["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError("public source-bank registry identity mismatch")
    if registry["canonical_json"] != (
        "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF"
    ):
        raise ValueError("public source-bank canonical JSON rule mismatch")
    if registry["supersedes"] != EXPECTED_V2_SUPERSEDES:
        raise ValueError("public source-bank supersedes record differs from v2 protocol")
    if registry["curation_audit"] != EXPECTED_CURATION_AUDIT:
        raise ValueError("public source-bank curation audit differs from v2 protocol")
    if registry["canonical_builder_sha256"] != file_sha256(PROMPT_BUILDER_FILE):
        raise ValueError("public source bank binds the wrong canonical builder")
    if registry["training_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise ValueError("public source bank binds the wrong training manifest")
    if registry["counts"] != {
        "new_ontology": EXPECTED_NEW_BANK_SIZE,
        "original_training": EXPECTED_ORIGINAL_BANK_SIZE,
        "total": EXPECTED_BANK_SIZE,
    }:
        raise ValueError("public source-bank counts mismatch")
    assignment_salt = registry["source_assignment_salt"]
    if not isinstance(assignment_salt, str) or not HEX64.fullmatch(assignment_salt):
        raise ValueError("public registry requires a lowercase hex-64 source_assignment_salt")
    if registry["source_assignment_algorithm"] != EXPECTED_SOURCE_ASSIGNMENT_ALGORITHM:
        raise ValueError("public registry source_assignment_algorithm differs from protocol")

    sources = registry.get("entries")
    if not isinstance(sources, list) or len(sources) != EXPECTED_BANK_SIZE:
        raise ValueError(f"public source bank must contain {EXPECTED_BANK_SIZE} source rows")
    if canonical_json_lf_sha256(sources) != registry["bank_entries_sha256"]:
        raise ValueError("public source-bank entry commitment mismatch")
    required = {
        "bank_index",
        "source_id",
        "source_phrase",
        "normalized_phrase",
        "head_lemma",
        "membership",
        "physical_audit_status",
    }
    normalized_phrases: set[str] = set()
    original_heads: set[str] = set()
    new_heads: set[str] = set()
    new_impact_features: set[str] = set()
    new_impact_notes: set[str] = set()
    source_ids: set[str] = set()
    membership_counts: Counter[str] = Counter()
    original_expected = {source_id: phrase for source_id, phrase in TRAIN_SOURCES}
    historical_source_ids = {
        source_id for source_id, _ in (*TRAIN_SOURCES, *TEST_SOURCES)
    }
    historical_source_phrases = {
        normalize_text(phrase) for _, phrase in (*TRAIN_SOURCES, *TEST_SOURCES)
    }
    historical_source_heads = {phrase.split()[-1] for phrase in historical_source_phrases}
    historical_receiver_phrases = {
        normalize_text(phrase) for _, phrase in (*TRAIN_RECEIVERS, *TEST_RECEIVERS)
    }
    historical_receiver_heads = {
        phrase.split()[-1] for phrase in historical_receiver_phrases
    } | {
        normalize_text(receiver_id).split()[-1]
        for receiver_id, _ in (*TRAIN_RECEIVERS, *TEST_RECEIVERS)
    }

    for index, source in enumerate(sources):
        membership = source.get("membership") if isinstance(source, dict) else None
        expected_fields = required | (
            {"strata", "impact_plausibility"}
            if membership == "new_bank_source"
            else set()
        )
        if not isinstance(source, dict) or set(source) != expected_fields:
            raise ValueError(f"source row {index} is missing required public fields")
        if source["bank_index"] != index:
            raise ValueError(f"source row {index} has a noncanonical bank_index")
        string_fields = required - {"bank_index"}
        if any(not isinstance(source[field], str) for field in string_fields):
            raise ValueError(f"source row {index} fields must all be strings")
        source_id = source["source_id"].strip()
        phrase = source["source_phrase"].strip()
        head = normalize_text(source["head_lemma"])
        membership = source["membership"]
        normalized_phrase = normalize_text(phrase)
        if not source_id or not phrase or not head or not normalized_phrase:
            raise ValueError(f"source row {index} contains an empty normalized value")
        if source_id in source_ids:
            raise ValueError(f"duplicate source_id in bank: {source_id}")
        if normalized_phrase in normalized_phrases:
            raise ValueError(f"duplicate normalized source phrase in bank: {phrase}")
        if normalized_phrase in historical_receiver_phrases:
            raise ValueError(f"source phrase overlaps a historical receiver: {phrase}")
        if source["normalized_phrase"] != normalized_phrase:
            raise ValueError(f"source row {index} normalized_phrase is noncanonical")
        if source["head_lemma"] != head:
            raise ValueError(f"source row {index} head_lemma is noncanonical")
        if membership not in {"original_training_source", "new_bank_source"}:
            raise ValueError(f"invalid source membership at row {index}: {membership!r}")
        if membership == "original_training_source":
            if original_expected.get(source_id) != phrase:
                raise ValueError(
                    f"original bank entry does not exactly match frozen training source {source_id}"
                )
            if source["physical_audit_status"] != LEGACY_PHYSICAL_STATUS:
                raise ValueError("original bank entry has invalid legacy physical status")
            original_heads.add(head)
        else:
            if source["physical_audit_status"] != STRICT_PHYSICAL_STATUS:
                raise ValueError("new bank entry is not strict-physical-pass v2")
            if source_id in historical_source_ids or phrase in original_expected.values():
                raise ValueError("a historical source identity is mislabeled as new_bank")
            if set(normalized_phrase.split()) & FORBIDDEN_SOURCE_WORDS:
                raise ValueError(f"new source phrase contains event vocabulary: {phrase}")
            if normalized_phrase in historical_source_phrases:
                raise ValueError(f"new source phrase overlaps a historical source: {phrase}")
            if (
                head in historical_source_heads
                or head in historical_receiver_heads
                or head in original_heads
                or head in new_heads
            ):
                raise ValueError(f"new source head lemma is not disjoint: {head}")
            strata = source["strata"]
            if not isinstance(strata, dict) or set(strata) != {
                "color_family",
                "food_status",
                "material_family",
                "origin",
                "shape_class",
                "texture_class",
            } or any(not isinstance(value, str) or not value for value in strata.values()):
                raise ValueError(f"new source row {index} has invalid strata")
            impact = source["impact_plausibility"]
            if not isinstance(impact, dict) or set(impact) != {
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
            }:
                raise ValueError(f"new source row {index} has invalid impact audit fields")
            dimensions = impact.get("dimensions_cm")
            numeric_dimensions = (
                isinstance(dimensions, list)
                and len(dimensions) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and 2.5 <= value <= 15.0
                    for value in dimensions
                )
            )
            density = impact.get("density_g_cm3")
            mass = impact.get("mass_g")
            if (
                impact["verdict"] != "pass"
                or impact["compact_and_rigid"] is not True
                or impact["natural_drop_entry"] is not True
                or impact["visible_brief_splash_or_ripple_plausible"] is not True
                or impact["negative_buoyancy"] is not True
                or impact["visually_recognizable"] is not True
                or impact["predominantly_buoyant_or_windborne"] is not False
                or impact["flexible_or_film_like"] is not False
                or impact["fragile"] is not False
                or impact["powder"] is not False
                or impact["loose_aggregate"] is not False
                or impact["porous"] is not False
                or impact["food_or_produce"] is not False
                or impact["entity_state"]
                not in {"solid_one_piece", "rigid_locked_assembly"}
                or not isinstance(impact["material"], str)
                or not impact["material"].strip()
                or not isinstance(density, (int, float))
                or isinstance(density, bool)
                or not math.isfinite(density)
                or not 3.0 <= density <= 20.0
                or not isinstance(mass, int)
                or isinstance(mass, bool)
                or not 350 <= mass <= 1200
                or not numeric_dimensions
                or max(dimensions) < 8.0
                or float(mass)
                > float(density)
                * math.prod(float(value) for value in dimensions)
                or impact["size_class"] != "palm_sized_explicit"
                or not isinstance(impact["source_specific_feature"], str)
                or not impact["source_specific_feature"].strip()
                or not isinstance(impact["curator_note"], str)
                or not impact["curator_note"].strip()
            ):
                raise ValueError(f"new source row {index} failed public impact audit")
            feature = impact["source_specific_feature"].strip()
            note = impact["curator_note"].strip()
            if feature in new_impact_features or note in new_impact_notes:
                raise ValueError(
                    "new source impact features and curator notes must be unique"
                )
            new_impact_features.add(feature)
            new_impact_notes.add(note)
            new_heads.add(head)
        source_ids.add(source_id)
        normalized_phrases.add(normalized_phrase)
        membership_counts[membership] += 1

    if original_heads & new_heads:
        raise ValueError("new source head lemmas overlap original training heads")

    if membership_counts != Counter(
        {
            "original_training_source": EXPECTED_ORIGINAL_BANK_SIZE,
            "new_bank_source": EXPECTED_NEW_BANK_SIZE,
        }
    ):
        raise ValueError(f"unexpected source membership counts: {dict(membership_counts)}")
    expected_original_rows = [
        {
            "bank_index": index,
            "source_id": source_id,
            "source_phrase": phrase,
            "normalized_phrase": normalize_text(phrase),
            "head_lemma": normalize_text(phrase).split()[-1],
            "membership": "original_training_source",
            "physical_audit_status": LEGACY_PHYSICAL_STATUS,
        }
        for index, (source_id, phrase) in enumerate(TRAIN_SOURCES)
    ]
    if sources[:EXPECTED_ORIGINAL_BANK_SIZE] != expected_original_rows:
        raise ValueError("public bank does not contain the exact ordered original eight")
    return registry, actual_sha256


def validate_public_holdout_commitment(
    path: Path,
    *,
    expected_sha256: str,
    bank_registry: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate commitment metadata only; never follow the private registry hash."""

    require_sha256(expected_sha256, "public holdout commitment hash")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"public holdout commitment is missing or symlinked: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"public holdout commitment hash mismatch: {actual} != {expected_sha256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    exact_fields = {
        "schema",
        "protocol",
        "registry",
        "dataset_version",
        "canonical_json",
        "supersedes",
        "curation_audit",
        "holdout_count",
        "holdout_registry_file_sha256",
        "split_rule",
        "split_salt_commitment_sha256",
        "aggregate_strata",
        "cross_role_checks",
    }
    if not isinstance(payload, dict) or set(payload) != exact_fields:
        raise ValueError("public holdout commitment fields are not exact")
    if (
        payload["schema"] != BANK_SCHEMA
        or payload["protocol"] != BANK_SCHEMA
        or payload["registry"] != HOLDOUT_REGISTRY
        or payload["dataset_version"] != DATASET_VERSION
    ):
        raise ValueError("public holdout commitment identity mismatch")
    if payload["canonical_json"] != (
        "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF"
    ):
        raise ValueError("public holdout commitment canonical JSON rule mismatch")
    if payload["supersedes"] != EXPECTED_V2_SUPERSEDES:
        raise ValueError("public holdout supersedes record differs from v2 protocol")
    if payload["curation_audit"] != EXPECTED_CURATION_AUDIT:
        raise ValueError("public holdout curation audit differs from v2 protocol")
    if payload["holdout_count"] != 24:
        raise ValueError("public holdout commitment must bind exactly 24 phrases")
    if payload["split_rule"] != EXPECTED_HOLDOUT_SPLIT_RULE:
        raise ValueError("public holdout commitment split rule differs from v2 protocol")
    require_sha256(payload["holdout_registry_file_sha256"], "private holdout file commitment")
    require_sha256(payload["split_salt_commitment_sha256"], "private split-salt commitment")
    checks = payload["cross_role_checks"]
    zero_fields = {
        "ambiguous_size_language_count",
        "event_literal_or_subword_risk_count",
        "historical_source_near_duplicate_count",
        "historical_source_semantic_equivalence_count",
        "impact_plausibility_failure_count",
        "new_receiver_historical_near_duplicate_count",
        "new_receiver_historical_semantic_equivalence_count",
        "normalized_head_overlap_count",
        "normalized_phrase_overlap_count",
        "prohibited_physical_category_count",
        "source_receiver_near_duplicate_count",
        "source_receiver_semantic_equivalence_count",
    }
    if not isinstance(checks, dict) or set(checks) != {
        *zero_fields,
        "impact_plausibility_pass_count",
        "matrix_scope",
        "source_specific_note_unique_count",
    }:
        raise ValueError("public holdout commitment cross-role checks are not exact")
    for field in zero_fields:
        if checks.get(field) != 0:
            raise ValueError(f"public holdout cross-role check failed: {field}")
    if (
        checks["impact_plausibility_pass_count"] != 80
        or checks["source_specific_note_unique_count"] != 80
        or checks["matrix_scope"]
        != {
            "complete_receivers": 84,
            "historical_receivers": 52,
            "historical_sources": 14,
            "new_receivers": 32,
            "new_sources": 80,
        }
    ):
        raise ValueError("public holdout impact-plausibility audit mismatch")
    aggregate = payload["aggregate_strata"]
    if not isinstance(aggregate, dict) or set(aggregate) != {
        "curated_new80",
        "new_bank56",
        "private_holdout24",
    }:
        raise ValueError("public holdout aggregate counts mismatch")
    validate_aggregate_strata(aggregate["curated_new80"], 80)
    validate_aggregate_strata(aggregate["new_bank56"], 56)
    validate_aggregate_strata(aggregate["private_holdout24"], 24)
    new_bank_strata = [
        dict(source["strata"])
        for source in bank_registry.get("entries", [])
        if source.get("membership") == "new_bank_source"
    ]
    if len(new_bank_strata) != EXPECTED_NEW_BANK_SIZE or aggregate_strata(
        new_bank_strata
    ) != aggregate["new_bank56"]:
        raise ValueError("public holdout new-bank strata do not match source bank")
    return payload, actual


def validate_public_stage0_commitment(
    path: Path,
    *,
    expected_sha256: str,
    bank_registry: dict[str, Any],
    holdout_commitment: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate the public Stage-0 commitment without following private refs."""

    require_sha256(expected_sha256, "public causal Stage-0 commitment hash")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"public causal Stage-0 commitment is missing or symlinked: {path}"
        )
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"public causal Stage-0 commitment hash mismatch: {actual} != {expected_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid public causal Stage-0 commitment: {exc}") from exc
    exact_fields = {
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
    if not isinstance(payload, dict) or set(payload) != exact_fields:
        raise ValueError("public causal Stage-0 commitment fields are not exact")
    if (
        payload["schema"] != BANK_SCHEMA
        or payload["protocol"] != BANK_SCHEMA
        or payload["dataset_version"] != DATASET_VERSION
        or payload["registry"] != STAGE0_PUBLIC_REGISTRY
        or payload["stage"] != 0
        or payload["status"] != "frozen_components_pending_external_bindings"
        or payload["authorization_status"] != "not_authorized"
        or payload["canonical_json"]
        != "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF"
        or payload["supersedes"] != EXPECTED_V2_SUPERSEDES
    ):
        raise ValueError("public causal Stage-0 commitment identity mismatch")
    if not (
        payload["curation_audit"]
        == bank_registry.get("curation_audit")
        == holdout_commitment.get("curation_audit")
        == EXPECTED_CURATION_AUDIT
    ):
        raise ValueError("public bank/holdout/Stage-0 curation audits differ")
    expected_cells = {
        f"{group}:{variant}": 8
        for group in EXPECTED_STAGE0_PUBLIC_METADATA["groups"]
        for variant in EXPECTED_STAGE0_PUBLIC_METADATA["prompt_variants"]
    }
    if (
        payload["remaining_blockers"] != EXPECTED_STAGE0_BLOCKERS
        or payload["candidate_count"] != 48
        or payload["cell_counts"] != expected_cells
        or payload["public_metadata"] != EXPECTED_STAGE0_PUBLIC_METADATA
    ):
        raise ValueError("public causal Stage-0 policy metadata mismatch")
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
        require_sha256(payload[field], f"public causal Stage-0/{field}")
    return payload, actual


def load_frozen_rows(manifest: Path, *, expected_sha256: str) -> list[dict[str, str]]:
    require_sha256(expected_sha256, "training manifest hash")
    if manifest.is_symlink() or not manifest.is_file():
        raise FileNotFoundError(f"frozen training manifest is missing or symlinked: {manifest}")
    actual = file_sha256(manifest)
    if actual != expected_sha256:
        raise ValueError(f"frozen training manifest hash mismatch: {actual} != {expected_sha256}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} manifest rows, found {len(rows)}")
    counts = Counter(row.get("training_role") for row in rows)
    if counts != Counter({"erase": EXPECTED_ERASE_ROWS, "preserve": EXPECTED_PRESERVE_ROWS}):
        raise ValueError(f"unexpected training-role counts: {dict(counts)}")
    scene_ids = [row.get("scene_id", "") for row in rows]
    if not all(scene_ids) or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("training manifest scene_id values must be non-empty and unique")
    for row in rows:
        if row["training_role"] != "erase":
            continue
        rebuilt = factual_prompt(row["source_object"], row["receiver"], row["prompt_variant"])
        if rebuilt != row["prompt"] or rebuilt != row["training_prompt"]:
            raise ValueError(
                f"canonical factual_prompt does not reproduce frozen row {row['scene_id']}"
            )
    return rows


def balanced_v3b_schedule(
    rows: list[dict[str, str]], *, seed: int = EXPECTED_SEED, steps: int = 200
) -> list[int]:
    """Return the exact v3b sample indices using a schedule-only RNG."""

    role_indices: dict[str, list[int]] = {"erase": [], "preserve": []}
    for index, row in enumerate(rows):
        role_indices[row["training_role"]].append(index)
    if not all(role_indices.values()):
        raise ValueError("balanced schedule requires erase and preserve rows")
    order_rng = random.Random(seed)
    for role in role_indices:  # insertion order is part of the v3b contract
        order_rng.shuffle(role_indices[role])
    cursors = {"erase": 0, "preserve": 0}
    schedule: list[int] = []
    for step in range(1, steps + 1):
        role = "erase" if step % 2 else "preserve"
        cursor = cursors[role]
        if cursor >= len(role_indices[role]):
            order_rng.shuffle(role_indices[role])
            cursor = 0
        schedule.append(role_indices[role][cursor])
        cursors[role] = cursor + 1
    return schedule


def sample_order_sha256(rows: list[dict[str, str]], schedule: list[int]) -> str:
    digest = hashlib.sha256()
    for step, manifest_index in enumerate(schedule, start=1):
        row = rows[manifest_index]
        digest.update(
            f"{step}:{row['training_role']}:{row['scene_id']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _bank_permutation(
    sources: list[dict[str, str]], salt: str
) -> list[dict[str, str]]:
    ranked = [
        (
            hashlib.sha256(
                f"{salt}\0permute-v2\0{source['source_id']}".encode("utf-8")
            ).hexdigest(),
            source,
        )
        for source in sources
    ]
    ranks = [rank for rank, _ in ranked]
    if len(set(ranks)) != len(ranks):
        raise ValueError("source-assignment permutation contains equal SHA-256 ranks")
    return [source for _, source in sorted(ranked, key=lambda item: item[0])]


def _repair_derangement(
    assignments: list[dict[str, str]],
    ordered_rows: list[tuple[int, dict[str, str]]],
    *,
    salt: str,
) -> None:
    """Repair collisions by deterministic swaps without changing counts."""

    for position in range(len(assignments)):
        if assignments[position]["source_phrase"] != ordered_rows[position][1]["source_object"]:
            continue
        candidates: list[int] = []
        # Never cross the active100 boundary: doing so could change the
        # registered 1-or-2 active frequency while leaving full counts intact.
        partition = (
            range(0, EXPECTED_ACTIVE_ERASE_ROWS)
            if position < EXPECTED_ACTIVE_ERASE_ROWS
            else range(EXPECTED_ACTIVE_ERASE_ROWS, len(assignments))
        )
        for other in partition:
            if other == position:
                continue
            if assignments[other]["source_phrase"] == ordered_rows[position][1]["source_object"]:
                continue
            if assignments[position]["source_phrase"] == ordered_rows[other][1]["source_object"]:
                continue
            candidates.append(other)
        if not candidates:
            raise ValueError(f"cannot repair source-slot collision at ordered position {position}")
        ranked_candidates = [
            (
                hashlib.sha256(
                    f"{salt}\0swap-v2\0{position}\0{candidate}".encode("utf-8")
                ).hexdigest(),
                candidate,
            )
            for candidate in candidates
        ]
        if len({rank for rank, _ in ranked_candidates}) != len(ranked_candidates):
            raise ValueError(f"equal source-swap candidate ranks at position {position}")
        other = min(ranked_candidates, key=lambda item: item[0])[1]
        assignments[position], assignments[other] = assignments[other], assignments[position]
    collisions = [
        position
        for position, assigned in enumerate(assignments)
        if assigned["source_phrase"] == ordered_rows[position][1]["source_object"]
    ]
    if collisions:
        raise ValueError(f"derangement repair left collisions: {collisions}")


def build_mapping(
    rows: list[dict[str, str]],
    bank: dict[str, Any],
    *,
    bank_registry_sha256: str,
    holdout_commitment_path: str,
    holdout_commitment_sha256: str,
    manifest_sha256: str,
    seed: int = EXPECTED_SEED,
) -> dict[str, Any]:
    if seed != EXPECTED_SEED:
        raise ValueError(f"v4 must retain v3b seed {EXPECTED_SEED}")
    schedule = balanced_v3b_schedule(rows, seed=seed, steps=200)
    order_hash = sample_order_sha256(rows, schedule)
    if order_hash != EXPECTED_SAMPLE_ORDER_SHA256:
        raise ValueError(
            f"v3b sample-order digest drift: {order_hash} != {EXPECTED_SAMPLE_ORDER_SHA256}"
        )
    active_indices = [index for index in schedule if rows[index]["training_role"] == "erase"]
    if len(active_indices) != EXPECTED_ACTIVE_ERASE_ROWS or len(set(active_indices)) != len(active_indices):
        raise ValueError("first 100 erase updates must be 100 distinct v3b erase rows")
    erase_indices = [index for index, row in enumerate(rows) if row["training_role"] == "erase"]
    remaining = [index for index in erase_indices if index not in set(active_indices)]
    ordered_indices = active_indices + remaining
    ordered_rows = [(index, rows[index]) for index in ordered_indices]

    require_sha256(holdout_commitment_sha256, "public holdout commitment hash")
    if PROMPT_BUILDER_FILE.is_symlink() or not PROMPT_BUILDER_FILE.is_file():
        raise FileNotFoundError("canonical factual-prompt builder is missing or symlinked")
    sources = bank["entries"]
    salt = bank["source_assignment_salt"]
    permutation = _bank_permutation(sources, salt)
    assignments = [
        permutation[ordinal % len(permutation)]
        for ordinal in range(EXPECTED_ERASE_ROWS)
    ]
    _repair_derangement(assignments, ordered_rows, salt=salt)

    active_counts = Counter(
        source["source_id"] for source in assignments[:EXPECTED_ACTIVE_ERASE_ROWS]
    )
    if set(active_counts.values()) - {1, 2} or len(active_counts) != EXPECTED_BANK_SIZE:
        raise ValueError(f"active source counts are not balanced 1-or-2: {dict(active_counts)}")
    if max(active_counts.values()) - min(active_counts.values()) != 1:
        raise ValueError("active source count difference must be exactly one for 100/64")

    erase_ordinal = {manifest_index: ordinal for ordinal, manifest_index in enumerate(erase_indices)}
    active_ordinal = {
        manifest_index: ordinal for ordinal, manifest_index in enumerate(active_indices)
    }
    records_by_index: dict[int, dict[str, Any]] = {}
    for (manifest_index, row), assigned in zip(ordered_rows, assignments):
        augmented = factual_prompt(
            assigned["source_phrase"], row["receiver"], row["prompt_variant"]
        )
        if augmented == row["prompt"]:
            raise ValueError(f"source-slot treatment did not change prompt {row['scene_id']}")
        records_by_index[manifest_index] = {
            "manifest_index": manifest_index,
            "scene_id": row["scene_id"],
            "erase_ordinal": erase_ordinal[manifest_index],
            "active_erase_ordinal": active_ordinal.get(manifest_index),
            "original_source_id": row["source_id"],
            "original_source_phrase": row["source_object"],
            "assigned_source_id": assigned["source_id"],
            "assigned_source_phrase": assigned["source_phrase"],
            "assigned_source_membership": assigned["membership"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
            "prompt_variant": row["prompt_variant"],
            "original_factual_prompt": row["prompt"],
            "augmented_factual_prompt": augmented,
        }
    records = [records_by_index[index] for index in erase_indices]
    active_records = sorted(
        (record for record in records if record["active_erase_ordinal"] is not None),
        key=lambda record: record["active_erase_ordinal"],
    )
    active_variant_counts = Counter(record["prompt_variant"] for record in active_records)
    active_receiver_counts = Counter(record["receiver_id"] for record in active_records)
    full_variant_counts = Counter(record["prompt_variant"] for record in records)
    full_receiver_counts = Counter(record["receiver_id"] for record in records)
    full_hash = canonical_json_sha256(records)
    active_hash = canonical_json_sha256(active_records)
    return {
        "protocol": PROTOCOL,
        "status": "frozen",
        "dataset_version": DATASET_VERSION,
        "source_bank_registry_sha256": bank_registry_sha256,
        "source_bank_schema": BANK_SCHEMA,
        "source_bank_registry": BANK_REGISTRY,
        "source_bank_size": EXPECTED_BANK_SIZE,
        "source_bank_entries_sha256": bank["bank_entries_sha256"],
        "holdout_public_commitment_path": holdout_commitment_path,
        "holdout_public_commitment_sha256": holdout_commitment_sha256,
        "holdout_count": 24,
        "source_assignment_salt_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest(),
        "source_assignment_algorithm_sha256": canonical_json_sha256(
            EXPECTED_SOURCE_ASSIGNMENT_ALGORITHM
        ),
        "train_manifest_sha256": manifest_sha256,
        "canonical_prompt_builder_path": str(PROMPT_BUILDER_PATH),
        "canonical_prompt_builder_sha256": file_sha256(PROMPT_BUILDER_FILE),
        "seed": seed,
        "balanced_schedule": "alternate_erase_on_odd_steps_preserve_on_even_steps_v3b_v1",
        "sample_order_sha256": order_hash,
        "full178_mapping_sha256": full_hash,
        "active100_mapping_sha256": active_hash,
        "active_source_counts": dict(sorted(active_counts.items())),
        "active_prompt_variant_counts": dict(sorted(active_variant_counts.items())),
        "active_receiver_counts": dict(sorted(active_receiver_counts.items())),
        "full_prompt_variant_counts": dict(sorted(full_variant_counts.items())),
        "full_receiver_counts": dict(sorted(full_receiver_counts.items())),
        "active_source_count_min": min(active_counts.values()),
        "active_source_count_max": max(active_counts.values()),
        "erase_row_count": len(records),
        "active_erase_count": len(active_records),
        "mapping": records,
    }


def atomic_write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite mapping registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"),
    )
    parser.add_argument("--manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument(
        "--source-bank-registry",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json"),
    )
    parser.add_argument("--source-bank-registry-sha256", required=True)
    parser.add_argument(
        "--holdout-public-commitment",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/holdout_public_commitment_v2.json"),
    )
    parser.add_argument("--holdout-public-commitment-sha256", required=True)
    parser.add_argument(
        "--causal-stage0-public-commitment",
        type=Path,
        default=Path(
            "data/water_impact_dynamic_v4/causal_stage0_public_commitment_v2.json"
        ),
    )
    parser.add_argument("--causal-stage0-public-commitment-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/water_impact_dynamic_v4/source_mapping_v2.json"),
    )
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    args = parser.parse_args()

    bank, bank_sha256 = validate_public_bank_registry(
        args.source_bank_registry,
        expected_sha256=args.source_bank_registry_sha256,
    )
    holdout, holdout_sha256 = validate_public_holdout_commitment(
        args.holdout_public_commitment,
        expected_sha256=args.holdout_public_commitment_sha256,
        bank_registry=bank,
    )
    validate_public_stage0_commitment(
        args.causal_stage0_public_commitment,
        expected_sha256=args.causal_stage0_public_commitment_sha256,
        bank_registry=bank,
        holdout_commitment=holdout,
    )
    rows = load_frozen_rows(args.manifest, expected_sha256=args.manifest_sha256)
    mapping = build_mapping(
        rows,
        bank,
        bank_registry_sha256=bank_sha256,
        holdout_commitment_path=str(args.holdout_public_commitment),
        holdout_commitment_sha256=holdout_sha256,
        manifest_sha256=args.manifest_sha256,
        seed=args.seed,
    )
    atomic_write_new_json(args.output, mapping)
    print(
        f"Wrote {len(mapping['mapping'])} deranged source assignments to {args.output}; "
        f"active100={mapping['active100_mapping_sha256']} "
        f"full178={mapping['full178_mapping_sha256']} "
        f"file_sha256={file_sha256(args.output)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
