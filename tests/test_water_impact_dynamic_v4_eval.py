from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import AbstractContextManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_water_impact_dynamic_v4_blind_review as builder  # noqa: E402
import score_water_impact_dynamic_v4 as scorer  # noqa: E402
import select_water_impact_dynamic_v4_eval as selector  # noqa: E402
import water_impact_dynamic_v4_eval_protocol as protocol  # noqa: E402
import run_water_impact_dynamic_v4_eval as runner  # noqa: E402


def synthetic_public_hash_patch(
    fixture: dict[str, object],
) -> AbstractContextManager[object]:
    """Pin a temporary fixture exactly without weakening production defaults."""

    frozen = fixture["synthetic_public_hashes"]
    assert isinstance(frozen, dict)
    return mock.patch.multiple(protocol, **frozen)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_causal_units() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_index = 0
    for group in protocol.CAUSAL_GROUPS:
        for variant in protocol.PROMPT_VARIANTS:
            for within in range(4):
                case_id = f"causal_{case_index:02d}"
                source_id = f"source_{case_index:02d}"
                receiver_id = f"receiver_{case_index:02d}"
                for replicate in range(3):
                    rows.append(
                        {
                            "unit_id": f"cu{len(rows):03d}",
                            "semantic_case_id": case_id,
                            "group": group,
                            "source_membership": "original_source" if group == "seen_source_new_receiver" else "holdout_source",
                            "prompt_variant": variant,
                            "source_id": source_id,
                            "source_phrase": f"object {case_index}",
                            "source_head_lemma": f"object_{case_index}",
                            "source_physical_audit_status": (
                                "legacy_original_source_exempt"
                                if group == "seen_source_new_receiver"
                                else "strict_physical_pass_v2"
                            ),
                            "receiver_id": receiver_id,
                            "receiver": f"receiver {case_index}",
                            "prompt": f"prompt {case_index}",
                            "replicate": replicate,
                            "seed": 1000 + len(rows),
                        }
                    )
                case_index += 1
    return rows


def make_causal_screening_candidates() -> list[dict[str, object]]:
    candidates = []
    index = 0
    for group in protocol.CAUSAL_GROUPS:
        for variant in protocol.PROMPT_VARIANTS:
            for within in range(8):
                case_id = f"candidate_{index}"
                record = {
                    "case_id": case_id,
                    "group": group,
                    "prompt_variant": variant,
                    "source_membership": "original_source"
                    if group == "seen_source_new_receiver"
                    else "holdout_source",
                    "source_id": f"source_{index}"
                    if group != "seen_source_new_receiver"
                    else f"original_{within}",
                    "source_phrase": f"object {index}",
                    "source_head_lemma": f"object_{index}",
                    "source_physical_audit_status": (
                        "legacy_original_source_exempt"
                        if group == "seen_source_new_receiver"
                        else "strict_physical_pass_v2"
                    ),
                    "receiver_membership": "new_receiver",
                    "receiver_id": f"receiver_{index}",
                    "receiver_phrase": f"receiver {index}",
                    "canonical_prompt": f"prompt {index}",
                }
                canonical = (
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
                candidates.append(
                    {
                        **record,
                        "canonical_record_sha256": hashlib.sha256(canonical).hexdigest(),
                        "candidate_id": case_id,
                        "semantic_case_id": case_id,
                        "receiver": record["receiver_phrase"],
                        "prompt": record["canonical_prompt"],
                    }
                )
                index += 1
    return candidates


def make_synthetic_stage0_fixture(root: Path) -> dict[str, object]:
    """Create identity-free synthetic bytes for executable Stage-0 tests."""

    private = root / "private"
    private.mkdir(parents=True)
    (root / "scripts").mkdir()
    shutil.copyfile(
        PROJECT_ROOT / "scripts/build_water_impact_dynamic_pairs_v1.py",
        root / "scripts/build_water_impact_dynamic_pairs_v1.py",
    )

    def write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    templates_payload = {
        "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
        "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
        "dataset_version": protocol.DATASET_VERSION,
        "canonical_builder_sha256": protocol.file_sha256(
            root / "scripts/build_water_impact_dynamic_pairs_v1.py"
        ),
        "prompt_templates": protocol.CAUSAL_CANONICAL_TEMPLATES,
        "template_fill_rules": protocol.CAUSAL_TEMPLATE_FILL_RULES,
        "non_substitution_rule": protocol.CAUSAL_TEMPLATE_NON_SUBSTITUTION_RULE,
    }
    def impact(index: int) -> dict[str, object]:
        return {
            "compact_and_rigid": True,
            "curator_note": f"synthetic physical audit {index}",
            "density_g_cm3": 7.8,
            "dimensions_cm": [8.0, 4.0, 3.0],
            "entity_state": "solid_one_piece",
            "food_or_produce": False,
            "flexible_or_film_like": False,
            "fragile": False,
            "loose_aggregate": False,
            "mass_g": 500,
            "material": "steel",
            "natural_drop_entry": True,
            "negative_buoyancy": True,
            "porous": False,
            "powder": False,
            "predominantly_buoyant_or_windborne": False,
            "size_class": "palm_sized_explicit",
            "source_specific_feature": f"feature_{index}",
            "verdict": "pass",
            "visually_recognizable": True,
            "visible_brief_splash_or_ripple_plausible": True,
        }

    new_sources: list[dict[str, object]] = []
    for index in range(80):
        source_id = (
            f"bank_source_{index:03d}" if index < 56 else f"holdout_source_{index - 56:03d}"
        )
        phrase = f"one dense synthetic token {index:03d}"
        new_sources.append(
            {
                "source_id": source_id,
                "source_phrase": phrase,
                "normalized_phrase": phrase,
                "head_lemma": f"token_{index:03d}",
                "origin": "manufactured",
                "food_status": "nonfood",
                "shape_class": "compact",
                "color_family": "gray",
                "material_family": "metal",
                "texture_class": "smooth",
                "impact_plausibility": impact(index),
                "curator": "synthetic_test_curator",
                "curation_stratum": tuple(sorted(protocol.CURATION_STRATA))[index // 20],
            }
        )
    source_by_id = {str(row["source_id"]): row for row in new_sources}
    original_items = list(protocol.ORIGINAL_TRAINING_SOURCES.items())
    historical_items = list(protocol.HISTORICAL_TRAINING_RECEIVERS.items())
    candidates: list[dict[str, object]] = []
    new_receiver_index = 0
    index = 0
    for group in protocol.CAUSAL_GROUPS:
        for variant_index, variant in enumerate(protocol.PROMPT_VARIANTS):
            for within in range(8):
                if group == "seen_source_new_receiver":
                    source_id, source = original_items[within]
                    source_head = {
                        "water_droplet": "droplet", "ice_cube": "cube", "red_apple": "apple",
                        "green_lime": "lime", "blue_marble": "marble", "wooden_cube": "cube",
                        "steel_ball": "bearing", "plastic_block": "block",
                    }[source_id]
                else:
                    holdout_index = within if group == "holdout_source_seen_receiver" else 8 + variant_index * 8 + within
                    source_id = f"holdout_source_{holdout_index:03d}"
                    source_row = source_by_id[source_id]
                    source = str(source_row["source_phrase"])
                    source_head = str(source_row["head_lemma"])
                if group == "holdout_source_seen_receiver":
                    receiver_id, receiver = historical_items[within]
                    receiver_membership = "seen_receiver"
                else:
                    receiver_id = f"receiver_new_{new_receiver_index:03d}"
                    receiver = f"a synthetic water receiver {new_receiver_index:03d}"
                    new_receiver_index += 1
                    receiver_membership = "new_receiver"
                filled = source.capitalize() if variant == "direct" else source
                record = {
                    "case_id": f"synthetic_case_{index:02d}",
                    "group": group,
                    "prompt_variant": variant,
                    "source_membership": "original_source" if group == "seen_source_new_receiver" else "holdout_source",
                    "source_id": source_id,
                    "source_phrase": source,
                    "source_head_lemma": source_head,
                    "source_physical_audit_status": (
                        "legacy_original_source_exempt"
                        if group == "seen_source_new_receiver"
                        else "strict_physical_pass_v2"
                    ),
                    "receiver_membership": receiver_membership,
                    "receiver_id": receiver_id,
                    "receiver_phrase": receiver,
                    "canonical_prompt": protocol.CAUSAL_CANONICAL_TEMPLATES[variant].format(
                        source_phrase=filled, receiver_phrase=receiver
                    ),
                }
                canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                candidates.append({**record, "canonical_record_sha256": hashlib.sha256(canonical).hexdigest()})
                index += 1
    component_paths = {
        "causal_stage0_candidates_private_v2.json": private
        / "causal_stage0_candidates_private_v2.json",
        "causal_stage0_templates_private_v2.json": private
        / "causal_stage0_templates_private_v2.json",
        "causal_stage0_field_rules_private_v2.json": private
        / "causal_stage0_field_rules_private_v2.json",
        "causal_stage0_render_config_private_v2.json": private
        / "causal_stage0_render_config_private_v2.json",
        "causal_stage0_selection_rules_private_v2.json": private
        / "causal_stage0_selection_rules_private_v2.json",
        "causal_stage0_secrets_private_v2.json": private
        / "causal_stage0_secrets_private_v2.json",
    }
    write_json(
        component_paths["causal_stage0_candidates_private_v2.json"],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "candidate_count": 48,
            "candidates": candidates,
        },
    )
    write_json(
        component_paths["causal_stage0_templates_private_v2.json"], templates_payload
    )
    write_json(
        component_paths["causal_stage0_field_rules_private_v2.json"],
        protocol.FIELD_NORMALIZATION_RULES,
    )
    write_json(
        component_paths["causal_stage0_render_config_private_v2.json"],
        protocol.CAUSAL_RENDER_CONFIGURATION,
    )
    write_json(
        component_paths["causal_stage0_selection_rules_private_v2.json"],
        protocol.CAUSAL_SELECTION_RULES,
    )
    selector_salt = "a" * 64
    evaluation_salt = "b" * 64
    screening_seed = 3_000_000_001
    write_json(
        component_paths["causal_stage0_secrets_private_v2.json"],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "evaluation_seed_namespace": "v4-causal-evaluation-v2",
            "evaluation_seed_salt": evaluation_salt,
            "screening_seed": screening_seed,
            "screening_seed_namespace": "v4-causal-stage0-screening-v2",
            "selector_salt": selector_salt,
        },
    )
    component_hashes = {
        name: protocol.file_sha256(path) for name, path in component_paths.items()
    }
    data = root / protocol.DATA_ROOT
    data.mkdir(parents=True)
    bank = root / protocol.PUBLIC_SOURCE_BANK
    holdout = root / protocol.PUBLIC_HOLDOUT_COMMITMENT
    bank_entries: list[dict[str, object]] = []
    original_heads = ["droplet", "cube", "apple", "lime", "marble", "cube", "bearing", "block"]
    for bank_index, ((source_id, phrase), head) in enumerate(zip(original_items, original_heads)):
        bank_entries.append(
            {"bank_index": bank_index, "source_id": source_id, "source_phrase": phrase,
             "normalized_phrase": phrase, "head_lemma": head, "membership": "original_training_source",
             "physical_audit_status": "legacy_original_source_exempt"}
        )
    for offset, source_row in enumerate(new_sources[:56], 8):
        bank_entries.append(
            {"bank_index": offset, "source_id": source_row["source_id"],
             "source_phrase": source_row["source_phrase"], "normalized_phrase": source_row["normalized_phrase"],
             "head_lemma": source_row["head_lemma"], "membership": "new_bank_source",
             "physical_audit_status": "strict_physical_pass_v2",
             "strata": {field: source_row[field] for field in protocol.SOURCE_STRATA_FIELDS},
             "impact_plausibility": source_row["impact_plausibility"]}
        )
    bank_entries_sha = hashlib.sha256(
        (json.dumps(bank_entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    write_json(
        bank,
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "registry": "public_augmentation_bank64_v2",
            "canonical_json": protocol.FIELD_NORMALIZATION_RULES["canonical_record"],
            "canonical_builder_sha256": protocol.file_sha256(root / "scripts/build_water_impact_dynamic_pairs_v1.py"),
            "training_manifest_sha256": "1" * 64,
            "source_assignment_salt": "2" * 64,
            "source_assignment_algorithm": {"synthetic": "test-only"},
            "counts": {"new_ontology": 56, "original_training": 8, "total": 64},
            "entries": bank_entries,
            "bank_entries_sha256": bank_entries_sha,
            "supersedes": protocol.V2_SUPERSEDES,
            "curation_audit": protocol.CURATION_AUDIT,
        },
    )
    write_json(
        holdout,
        {"holdout_registry_file_sha256": "d" * 64, "curation_audit": protocol.CURATION_AUDIT},
    )
    ontology_paths = {
        "source_ontology_80": private / "source_ontology_private80_v2.json",
        "source_split_80": private / "source_split_private_v2.json",
        "holdout_registry_24": private / "holdout_registry_private24_v2.json",
        "receiver_ontology_32": private / "receiver_ontology_private32_v2.json",
    }
    write_json(
        ontology_paths["source_ontology_80"],
        {"schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA, "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
         "dataset_version": protocol.DATASET_VERSION, "source_count": 80, "sources": new_sources},
    )
    split_rows = [
        {"source_id": row["source_id"], "membership": "new_bank_source" if index < 56 else "holdout_source",
         "split_rank_sha256": hashlib.sha256(f"split-{index}".encode()).hexdigest()}
        for index, row in enumerate(new_sources)
    ]
    write_json(
        ontology_paths["source_split_80"],
        {"schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA, "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
         "dataset_version": protocol.DATASET_VERSION, "domain": "bank-holdout-v2", "rows": split_rows},
    )
    holdout_entries = []
    for index, row in enumerate(new_sources[56:]):
        holdout_entries.append(
            {"holdout_index": index, "source_id": row["source_id"], "source_phrase": row["source_phrase"],
             "normalized_phrase": row["normalized_phrase"], "head_lemma": row["head_lemma"],
             "impact_plausibility": row["impact_plausibility"],
             "strata": {field: row[field] for field in protocol.SOURCE_STRATA_FIELDS}}
        )
    write_json(
        ontology_paths["holdout_registry_24"],
        {"schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA, "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
         "dataset_version": protocol.DATASET_VERSION, "registry": "private_ordered_holdout24_v2",
         "ordering": "source_id ascending", "entries": holdout_entries},
    )
    receiver_types = sorted(protocol.RECEIVER_TYPES)
    receivers = [
        {"receiver_id": f"receiver_new_{index:03d}", "receiver_phrase": f"a synthetic water receiver {index:03d}",
         "normalized_phrase": f"a synthetic water receiver {index:03d}", "head_lemma": f"receiver_{index:03d}",
         "receiver_type": receiver_types[index % len(receiver_types)], "curator_note": f"note {index}",
         "curator": "synthetic_test_curator"}
        for index in range(32)
    ]
    write_json(
        ontology_paths["receiver_ontology_32"],
        {"schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA, "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
         "dataset_version": protocol.DATASET_VERSION, "receiver_count": 32, "receivers": receivers,
         "historical_receiver_blacklist_count": 52},
    )
    bundle = private / "causal_stage0_bundle_private_v2.json"
    write_json(
        bundle,
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "status": "frozen_components_pending_external_bindings",
            "components": component_hashes,
            "source_bank_entries_sha256": bank_entries_sha,
            "holdout_registry_file_sha256": "d" * 64,
        },
    )

    def commitment(name: str, secret: str) -> str:
        payload = json.dumps(
            {"name": name, "secret": secret},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()

    pending = root / protocol.PENDING_STAGE0_COMMITMENTS["causal"]
    write_json(
        pending,
        {
            "authorization_status": "not_authorized",
            "candidate_count": 48,
            "candidate_manifest_sha256": component_hashes[
                "causal_stage0_candidates_private_v2.json"
            ],
            "canonical_json": protocol.FIELD_NORMALIZATION_RULES["canonical_record"],
            "canonical_templates_sha256": component_hashes[
                "causal_stage0_templates_private_v2.json"
            ],
            "cell_counts": {
                f"{group}:{variant}": 8
                for group in protocol.CAUSAL_GROUPS
                for variant in protocol.PROMPT_VARIANTS
            },
            "curation_audit": protocol.CURATION_AUDIT,
            "dataset_version": protocol.DATASET_VERSION,
            "evaluation_seed_salt_commitment_sha256": commitment(
                "causal_evaluation_seed_salt_v2", evaluation_salt
            ),
            "field_normalization_sha256": component_hashes[
                "causal_stage0_field_rules_private_v2.json"
            ],
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "public_metadata": {
                "candidates_per_cell": 8,
                "evaluation_seed_domain": "causal-eval-seed-v2",
                "evaluation_seed_namespace": "v4-causal-evaluation-v2",
                "evaluation_unit_target": 72,
                "full_frame_screening_required": True,
                "groups": list(protocol.CAUSAL_GROUPS),
                "no_reserve_queue": True,
                "prompt_variants": list(protocol.PROMPT_VARIANTS),
                "ranking_domain": "causal-selector-v2",
                "replicates_per_selected_case": 3,
                "screening_arm": "Original_only",
                "screening_seed_namespace": "v4-causal-stage0-screening-v2",
                "selected_case_target": 24,
                "selection_per_cell": 4,
                "source_physical_policy": protocol.CURATION_AUDIT["legacy_original_source_policy"],
            },
            "remaining_blockers": [
                "an independent seed auditor must commit the complete forbidden numeric seed inventory and prove disjointness",
                "an independent binder must commit the exact already-frozen v3b path-plus-file-bytes model inventory digest",
            ],
            "render_configuration_sha256": component_hashes[
                "causal_stage0_render_config_private_v2.json"
            ],
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "registry": "causal_stage0_public_commitment_v2",
            "screening_seed_commitment_sha256": commitment(
                "causal_screening_seed_v2", str(screening_seed)
            ),
            "selector_rules_sha256": component_hashes[
                "causal_stage0_selection_rules_private_v2.json"
            ],
            "selector_salt_commitment_sha256": commitment(
                "causal_stage0_selector_salt_v2", selector_salt
            ),
            "stage": 0,
            "stage0_bundle_file_sha256": protocol.file_sha256(bundle),
            "status": "frozen_components_pending_external_bindings",
            "supersedes": json.loads(json.dumps(protocol.V2_SUPERSEDES)),
        },
    )
    model_sha = protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256
    runtime_registry = root / protocol.RUNTIME_REGISTRY
    write_json(runtime_registry, protocol.RUNTIME_REGISTRY_PAYLOAD)
    runtime_sha = protocol.file_sha256(runtime_registry)
    generation_spec = private / "generation_spec.json"
    write_json(
        generation_spec,
        {
            "protocol": protocol.GENERATION_SPEC_PROTOCOL,
            "status": "frozen_before_original_render",
            "model_inventory_sha256": model_sha,
            "runtime_registry": {
                "path": protocol.RUNTIME_REGISTRY,
                "sha256": runtime_sha,
            },
            "generation_spec": protocol.GENERATION_SPEC,
            "source_mode": "Original_screening_then_matched_O_v3b_v4",
        },
    )
    screening_seed_path = private / "screening_seed.txt"
    selector_salt_path = private / "selector_salt.txt"
    evaluation_salt_path = private / "evaluation_seed_salt.txt"
    screening_seed_path.write_text(f"{screening_seed}\n", encoding="ascii")
    selector_salt_path.write_text(f"{selector_salt}\n", encoding="ascii")
    evaluation_salt_path.write_text(f"{evaluation_salt}\n", encoding="ascii")
    forbidden = private / "forbidden_seeds.json"
    write_json(
        forbidden,
        {
            "protocol": protocol.FORBIDDEN_SEED_INVENTORY_PROTOCOL,
            "dataset": "causal",
            "status": "frozen_by_independent_seed_auditor",
            "seed_encoding": "nonnegative JSON integer below 2^63",
            "source_commitments": [
                {"name": "synthetic_history", "sha256": "e" * 64, "seed_count": 2}
            ],
            "seeds": [1, 2],
        },
    )
    binding = private / "selection_binding.json"
    synthetic_public_hashes = {
        "FROZEN_CAUSAL_STAGE0_PUBLIC_COMMITMENT_SHA256": protocol.file_sha256(
            pending
        ),
        "FROZEN_PUBLIC_SOURCE_BANK_SHA256": protocol.file_sha256(bank),
        "FROZEN_PUBLIC_HOLDOUT_COMMITMENT_SHA256": protocol.file_sha256(holdout),
    }
    with mock.patch.multiple(protocol, **synthetic_public_hashes):
        expected_binding = protocol.prepare_selection_binding(
            root,
            dataset="causal",
            private_root=private,
            candidate_manifest_path=component_paths[
                "causal_stage0_candidates_private_v2.json"
            ],
            canonical_templates_path=component_paths[
                "causal_stage0_templates_private_v2.json"
            ],
            field_rules_path=component_paths[
                "causal_stage0_field_rules_private_v2.json"
            ],
            render_configuration_path=component_paths[
                "causal_stage0_render_config_private_v2.json"
            ],
            selection_rules_path=component_paths[
                "causal_stage0_selection_rules_private_v2.json"
            ],
            secrets_path=component_paths["causal_stage0_secrets_private_v2.json"],
            root_bundle_path=bundle,
            generation_spec_path=generation_spec,
            screening_seed_path=screening_seed_path,
            selector_salt_path=selector_salt_path,
            evaluation_seed_salt_path=evaluation_salt_path,
            forbidden_seed_inventory_path=forbidden,
            source_ontology_path=ontology_paths["source_ontology_80"],
            source_split_path=ontology_paths["source_split_80"],
            holdout_registry_path=ontology_paths["holdout_registry_24"],
            receiver_ontology_path=ontology_paths["receiver_ontology_32"],
        )
    write_json(binding, expected_binding)

    def record(path: Path, rows: int | None = None) -> dict[str, object]:
        return {
            "sha256": protocol.file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": rows,
        }

    rules_record = record(
        component_paths["causal_stage0_selection_rules_private_v2.json"]
    )
    stage0_payload = {
        "protocol": protocol.COMMITMENT_PROTOCOL,
        "dataset": "causal",
        "dataset_version": protocol.DATASET_VERSION,
        "stage": 0,
        "status": "committed",
        "sealed_final36_status": "unopened",
        "artifacts": {
            "candidate_manifest_48": record(
                component_paths["causal_stage0_candidates_private_v2.json"], 48
            ),
            "source_bank_registry_64": record(bank, 64),
            "source_ontology_80": record(ontology_paths["source_ontology_80"], 80),
            "source_split_80": record(ontology_paths["source_split_80"], 80),
            "holdout_registry_24": record(ontology_paths["holdout_registry_24"], 24),
            "receiver_ontology_32": record(ontology_paths["receiver_ontology_32"], 32),
            "canonical_templates": record(
                component_paths["causal_stage0_templates_private_v2.json"]
            ),
            "field_normalization": record(
                component_paths["causal_stage0_field_rules_private_v2.json"]
            ),
            "raw_root_bundle": record(bundle),
            "raw_render_configuration": record(
                component_paths["causal_stage0_render_config_private_v2.json"]
            ),
            "stage0_secrets": record(
                component_paths["causal_stage0_secrets_private_v2.json"]
            ),
            "screening_seed": record(screening_seed_path),
            "screening_generation_spec": record(generation_spec),
            "selector_salt": record(selector_salt_path),
            "ranking_formula": rules_record,
            "constrained_subset_algorithm": dict(rules_record),
            "evaluation_seed_salt": record(evaluation_salt_path),
            "seed_derivation_formula": record(binding),
            "forbidden_seed_inventory": record(forbidden),
        },
    }
    stage0 = root / protocol.CAUSAL_STAGE0
    write_json(stage0, stage0_payload)
    normalized = protocol.load_normalized_candidate_manifest(
        component_paths["causal_stage0_candidates_private_v2.json"],
        dataset="causal",
        canonical_templates_path=component_paths[
            "causal_stage0_templates_private_v2.json"
        ],
    )
    return {
        "root": root,
        "private": private,
        "components": component_paths,
        "ontologies": ontology_paths,
        "bundle": bundle,
        "generation_spec": generation_spec,
        "screening_seed": screening_seed_path,
        "selector_salt": selector_salt_path,
        "evaluation_salt": evaluation_salt_path,
        "forbidden": forbidden,
        "binding": binding,
        "stage0": stage0,
        "normalized": normalized,
        "model_sha256": model_sha,
        "runtime_sha256": runtime_sha,
        "synthetic_public_hashes": synthetic_public_hashes,
    }


def make_synthetic_screening_generation(
    fixture: dict[str, object],
) -> Path:
    root = fixture["root"]
    private = fixture["private"]
    normalized = fixture["normalized"]
    components = fixture["components"]
    assert isinstance(root, Path)
    assert isinstance(private, Path)
    assert isinstance(normalized, list)
    assert isinstance(components, dict)
    output = private / "screening_raw"
    videos_dir = output / "videos"
    videos_dir.mkdir(parents=True)
    seed = int(Path(fixture["screening_seed"]).read_text(encoding="ascii").strip())
    reservation = {
        "protocol": runner.SCREENING_GENERATION_PROTOCOL,
        "dataset": "causal",
        "dataset_version": protocol.DATASET_VERSION,
        "method": "original",
        "stage0_registry_sha256": protocol.file_sha256(Path(fixture["stage0"])),
        "candidate_manifest_sha256": protocol.file_sha256(
            components["causal_stage0_candidates_private_v2.json"]
        ),
        "screening_seed_sha256": protocol.file_sha256(Path(fixture["screening_seed"])),
        "model_inventory_sha256": fixture["model_sha256"],
        "runtime_registry_sha256": fixture["runtime_sha256"],
    }
    (output / ".run_reservation_v2.json").write_text(
        json.dumps(reservation, indent=2) + "\n", encoding="utf-8"
    )
    (output / "prompts.txt").write_text(
        "".join(
            f"{row['prompt']} | {row['source_phrase']} | registered v4 evaluation\n"
            for row in normalized
        ),
        encoding="utf-8",
    )
    generation = {
        "baseline": "clean",
        "seed": 42,
        "seeds": [seed] * 48,
        "num_inference_steps": 25,
        "guidance_scale": 5.0,
        "num_frames": 49,
        "fps": 8,
        "height": 480,
        "width": 832,
        "dtype": "bf16",
        "device": "cuda",
        "enable_model_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "vae_slicing": True,
        "vae_tiling": True,
        "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
        "lora_path": None,
        "lora_sha256": None,
        "lora_scale": 1.0,
        "activation_gate_dir": None,
        "persistent_activation_gate": False,
        "lora_target_phrases": [],
        "attention_gate_dir": None,
        "attention_suppression_phrases": [],
        "attention_suppression_strength": 20.0,
    }
    items: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index, row in enumerate(normalized):
        video = videos_dir / f"video_{index:03d}.mp4"
        video.write_bytes(f"synthetic-video-{index:03d}".encode("ascii"))
        items.append(
            {
                "index": index,
                "prompt": row["prompt"],
                "target_concept": row["source_phrase"],
                "expected_effect": "registered v4 evaluation",
                "seed": seed,
                "video_path": str(video),
            }
        )
        records.append(
            {
                "unit_id": f"screen_c_{index:03d}",
                "index": index,
                "path": str(video),
                "size_bytes": video.stat().st_size,
                "sha256": protocol.file_sha256(video),
                "prompt_sha256": hashlib.sha256(
                    str(row["prompt"]).encode("utf-8")
                ).hexdigest(),
                "seed": seed,
                "frame_count": 49,
                "width": 832,
                "height": 480,
                "fps_numerator": 8,
                "fps_denominator": 1,
            }
        )
    raw_path = output / "generation_manifest.json"
    raw_path.write_text(
        json.dumps(
            {
                "created_at_utc": "synthetic-before-review",
                "baseline": "clean",
                "pipeline": "WanPipeline",
                "model": protocol.GENERATION_SPEC["model"],
                "dry_run": False,
                "prompts": str(output / "prompts.txt"),
                "generation": generation,
                "items": items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = output / "v4_screening_generation_manifest_v2.json"
    manifest_path.write_text(
        json.dumps(
            {
                **reservation,
                "raw_generation_manifest": {
                    "path": str(raw_path),
                    "sha256": protocol.file_sha256(raw_path),
                },
                "generation_spec": protocol.GENERATION_SPEC,
                "videos": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def make_synthetic_specificity_stage0_fixture(root: Path) -> dict[str, object]:
    """Build a synthetic future specificity Stage-0 after a causal Stage-1."""

    causal = make_synthetic_stage0_fixture(root)
    private = causal["private"]
    assert isinstance(private, Path)

    def write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    causal_scored = [
        {
            **row,
            **{
                field: 2
                for field in selector.CAUSAL_SCREENING_FIELDS.values()
            },
        }
        for row in causal["normalized"]
    ]
    causal_selected_raw, _ = selector.select_causal_cases(
        causal_scored, "a" * 64
    )
    causal_selected = selector._private_case_rows(causal_selected_raw, "causal")
    causal_units = protocol.derive_unit_rows(
        causal_selected,
        dataset="causal",
        private_salt="b" * 64,
        forbidden_seeds={1, 2, 3_000_000_001},
    )
    causal_selected_path = private / "causal_selected_cases_v2.csv"
    causal_units_path = private / "causal_unit_manifest_U_v2.csv"
    protocol.write_csv(causal_selected_path, causal_selected)
    protocol.write_csv(causal_units_path, causal_units)

    def record(path: Path, rows: int | None = None) -> dict[str, object]:
        return {
            "sha256": protocol.file_sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": rows,
        }

    causal_stage1_artifacts: dict[str, dict[str, object]] = {}
    for name in protocol.STAGE_ARTIFACTS[("causal", 1)]:
        if name == "selected_case_manifest_24":
            causal_stage1_artifacts[name] = record(causal_selected_path, 24)
        elif name == "unit_manifest_U_72":
            causal_stage1_artifacts[name] = record(causal_units_path, 72)
        else:
            causal_stage1_artifacts[name] = {
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
                "size_bytes": 1,
                "row_count": protocol.EXPECTED_COMMITMENT_ROW_COUNTS.get(
                    ("causal", 1, name)
                ),
            }
    causal_stage1_path = root / protocol.CAUSAL_STAGE1
    write_json(
        causal_stage1_path,
        {
            "protocol": protocol.COMMITMENT_PROTOCOL,
            "dataset": "causal",
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 1,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "stage0_registry_sha256": protocol.file_sha256(Path(causal["stage0"])),
            "artifacts": causal_stage1_artifacts,
        },
    )
    protocol.validate_commitment_registry(
        causal_stage1_path,
        dataset="causal",
        stage=1,
        expected_stage0_sha256=protocol.file_sha256(Path(causal["stage0"])),
    )

    bank_path = root / protocol.PUBLIC_SOURCE_BANK
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_rows = [
        row for row in bank["entries"] if row["membership"] == "new_bank_source"
    ]
    matched_original = [
        row
        for row in causal_selected
        if row["group"] == "seen_source_new_receiver"
    ]
    matched_holdout = [
        row for row in causal_selected if row["group"] in protocol.HOLDOUT_GROUPS
    ]
    candidates: list[dict[str, str]] = []
    bank_index = 0
    for membership in protocol.SPECIFICITY_MEMBERSHIPS:
        for variant in protocol.PROMPT_VARIANTS:
            if membership == "original_source":
                sources = [
                    row for row in matched_original if row["prompt_variant"] == variant
                ]
            elif membership == "holdout_source":
                sources = [
                    row for row in matched_holdout if row["prompt_variant"] == variant
                ]
            else:
                sources = bank_rows[bank_index : bank_index + 6]
                bank_index += 6
            for source_position, source in enumerate(sources):
                if membership == "new_bank_source":
                    source_id = str(source["source_id"])
                    source_phrase = str(source["source_phrase"])
                    source_head = str(source["head_lemma"])
                    receiver_id = f"specificity_receiver_{variant}_{source_position}"
                    receiver_phrase = f"a synthetic dry receiver {variant} {source_position}"
                    causal_case_id = ""
                else:
                    source_id = str(source["source_id"])
                    source_phrase = str(source["source_phrase"])
                    source_head = str(source["source_head_lemma"])
                    receiver_id = str(source["receiver_id"])
                    receiver_phrase = str(source["receiver"])
                    causal_case_id = str(source["semantic_case_id"])
                filled_source = (
                    source_phrase.capitalize() if variant == "direct" else source_phrase
                )
                record_row = {
                    "case_id": f"specificity_candidate_{len(candidates):02d}",
                    "membership": membership,
                    "prompt_variant": variant,
                    "source_id": source_id,
                    "source_phrase": source_phrase,
                    "source_head_lemma": source_head,
                    "receiver_id": receiver_id,
                    "receiver_phrase": receiver_phrase,
                    "causal_case_id": causal_case_id,
                    "template_id": variant,
                    "canonical_prompt": protocol.SPECIFICITY_CANONICAL_TEMPLATES[
                        variant
                    ].format(
                        source_phrase=filled_source,
                        receiver_phrase=receiver_phrase,
                    ),
                }
                canonical = (
                    json.dumps(
                        record_row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
                candidates.append(
                    {
                        **record_row,
                        "canonical_record_sha256": hashlib.sha256(canonical).hexdigest(),
                    }
                )

    component_paths = {
        name: private / name for name in protocol.SPECIFICITY_PRIVATE_COMPONENT_FILENAMES
    }
    write_json(
        component_paths["specificity_stage0_candidates_private_v2.json"],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "candidate_count": 36,
            "causal_stage1_registry_sha256": protocol.file_sha256(causal_stage1_path),
            "candidates": candidates,
        },
    )
    assignments = [
        {
            "case_id": row["case_id"],
            "source_id": row["source_id"],
            "source_phrase": row["source_phrase"],
            "source_head_lemma": row["source_head_lemma"],
            "receiver_id": row["receiver_id"],
            "receiver_phrase": row["receiver_phrase"],
            "prompt_variant": row["prompt_variant"],
            "rank_sha256": hashlib.sha256(
                f"assignment-{row['case_id']}".encode("utf-8")
            ).hexdigest(),
        }
        for row in candidates
        if row["membership"] == "new_bank_source"
    ]
    write_json(
        component_paths[
            "specificity_stage0_new_bank_assignment_private_v2.json"
        ],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "causal_stage1_registry_sha256": protocol.file_sha256(causal_stage1_path),
            "assignment_count": 12,
            "assignments": assignments,
        },
    )
    write_json(
        component_paths["specificity_stage0_templates_private_v2.json"],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "prompt_templates": protocol.SPECIFICITY_CANONICAL_TEMPLATES,
            "template_fill_rules": protocol.SPECIFICITY_TEMPLATE_FILL_RULES,
            "non_substitution_rule": protocol.SPECIFICITY_TEMPLATE_NON_SUBSTITUTION_RULE,
        },
    )
    write_json(
        component_paths["specificity_stage0_field_rules_private_v2.json"],
        protocol.FIELD_NORMALIZATION_RULES,
    )
    write_json(
        component_paths["specificity_stage0_render_config_private_v2.json"],
        protocol.SPECIFICITY_RENDER_CONFIGURATION,
    )
    write_json(
        component_paths["specificity_stage0_selection_rules_private_v2.json"],
        protocol.SPECIFICITY_SELECTION_RULES,
    )
    screening_seed = 3_500_000_001
    selector_salt = "c" * 64
    evaluation_salt = "d" * 64
    write_json(
        component_paths["specificity_stage0_secrets_private_v2.json"],
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "evaluation_seed_namespace": "v4-specificity-evaluation-v2",
            "evaluation_seed_salt": evaluation_salt,
            "screening_seed": screening_seed,
            "screening_seed_namespace": "v4-specificity-stage0-screening-v2",
            "selector_salt": selector_salt,
        },
    )
    component_hashes = {
        name: protocol.file_sha256(path) for name, path in component_paths.items()
    }
    bundle_path = private / "specificity_stage0_bundle_private_v2.json"
    write_json(
        bundle_path,
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "protocol": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "status": "frozen_components_pending_external_bindings",
            "components": component_hashes,
            "source_bank_entries_sha256": bank["bank_entries_sha256"],
            "causal_stage0_registry_sha256": protocol.file_sha256(Path(causal["stage0"])),
            "causal_stage1_registry_sha256": protocol.file_sha256(causal_stage1_path),
            "selected_case_manifest_24_sha256": protocol.file_sha256(
                causal_selected_path
            ),
            "unit_manifest_U_72_sha256": protocol.file_sha256(causal_units_path),
        },
    )

    def commitment(name: str, secret: str) -> str:
        return protocol._secret_commitment(name, secret)

    pending_path = root / protocol.PENDING_STAGE0_COMMITMENTS["specificity"]
    write_json(
        pending_path,
        {
            "schema": protocol.SOURCE_SLOT_REGISTRY_SCHEMA,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": 0,
            "status": "frozen_components_pending_external_bindings",
            "authorization_status": "not_authorized",
            "candidate_count": 36,
            "candidate_manifest_sha256": component_hashes[
                "specificity_stage0_candidates_private_v2.json"
            ],
            "new_bank_assignment_sha256": component_hashes[
                "specificity_stage0_new_bank_assignment_private_v2.json"
            ],
            "canonical_templates_sha256": component_hashes[
                "specificity_stage0_templates_private_v2.json"
            ],
            "field_normalization_sha256": component_hashes[
                "specificity_stage0_field_rules_private_v2.json"
            ],
            "render_configuration_sha256": component_hashes[
                "specificity_stage0_render_config_private_v2.json"
            ],
            "selector_rules_sha256": component_hashes[
                "specificity_stage0_selection_rules_private_v2.json"
            ],
            "screening_seed_commitment_sha256": commitment(
                "specificity_screening_seed_v2", str(screening_seed)
            ),
            "selector_salt_commitment_sha256": commitment(
                "specificity_stage0_selector_salt_v2", selector_salt
            ),
            "evaluation_seed_salt_commitment_sha256": commitment(
                "specificity_evaluation_seed_salt_v2", evaluation_salt
            ),
            "stage0_bundle_file_sha256": protocol.file_sha256(bundle_path),
            "causal_stage0_registry_sha256": protocol.file_sha256(Path(causal["stage0"])),
            "causal_stage1_registry_sha256": protocol.file_sha256(causal_stage1_path),
            "selected_case_manifest_24_sha256": protocol.file_sha256(
                causal_selected_path
            ),
            "unit_manifest_U_72_sha256": protocol.file_sha256(causal_units_path),
            "remaining_blockers": [
                "independent forbidden numeric seed inventory",
                "exact full-model path-plus-file-bytes inventory digest",
            ],
        },
    )
    screening_seed_path = private / "specificity_screening_seed.txt"
    selector_salt_path = private / "specificity_selector_salt.txt"
    evaluation_salt_path = private / "specificity_evaluation_seed_salt.txt"
    screening_seed_path.write_text(f"{screening_seed}\n", encoding="ascii")
    selector_salt_path.write_text(f"{selector_salt}\n", encoding="ascii")
    evaluation_salt_path.write_text(f"{evaluation_salt}\n", encoding="ascii")
    forbidden_path = private / "specificity_forbidden_seeds.json"
    forbidden_values = sorted({1, 2, *(int(row["seed"]) for row in causal_units)})
    write_json(
        forbidden_path,
        {
            "protocol": protocol.FORBIDDEN_SEED_INVENTORY_PROTOCOL,
            "dataset": "specificity",
            "status": "frozen_by_independent_seed_auditor",
            "seed_encoding": "nonnegative JSON integer below 2^63",
            "source_commitments": [
                {
                    "name": "causal_U_72",
                    "sha256": protocol.file_sha256(causal_units_path),
                    "seed_count": 72,
                },
                {"name": "synthetic_history", "sha256": "e" * 64, "seed_count": 2},
            ],
            "seeds": forbidden_values,
        },
    )
    return {
        **causal,
        "causal_selected": causal_selected,
        "causal_units": causal_units,
        "causal_selected_path": causal_selected_path,
        "causal_units_path": causal_units_path,
        "causal_stage1": causal_stage1_path,
        "specificity_components": component_paths,
        "specificity_bundle": bundle_path,
        "specificity_pending": pending_path,
        "specificity_screening_seed": screening_seed_path,
        "specificity_selector_salt": selector_salt_path,
        "specificity_evaluation_salt": evaluation_salt_path,
        "specificity_forbidden": forbidden_path,
        "specificity_binding": private / "specificity_selection_binding_v2.json",
        "specificity_stage0": root / protocol.SPECIFICITY_STAGE0,
    }


def make_specificity_units(causal_units: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    causal_cases = {}
    for row in causal_units:
        causal_cases.setdefault(str(row["semantic_case_id"]), row)
    original_matches = [
        row for row in causal_cases.values() if row["group"] == "seen_source_new_receiver"
    ][:6]
    holdout_matches = [
        row
        for row in causal_cases.values()
        if row["group"] in protocol.HOLDOUT_GROUPS
    ]
    # Pick three per variant while covering both holdout groups.
    chosen_holdout = []
    for variant in protocol.PROMPT_VARIANTS:
        candidates = [row for row in holdout_matches if row["prompt_variant"] == variant]
        chosen_holdout.extend([candidates[0], candidates[1], candidates[4]])
    matched_by_cell = {
        ("original_source", variant): [row for row in original_matches if row["prompt_variant"] == variant][:3]
        for variant in protocol.PROMPT_VARIANTS
    }
    # There are four original matches per variant in causal selection.
    for variant in protocol.PROMPT_VARIANTS:
        if len(matched_by_cell[("original_source", variant)]) < 3:
            matched_by_cell[("original_source", variant)] = [
                row
                for row in causal_cases.values()
                if row["group"] == "seen_source_new_receiver" and row["prompt_variant"] == variant
            ][:3]
    matched_by_cell.update(
        {
            ("holdout_source", variant): [row for row in chosen_holdout if row["prompt_variant"] == variant]
            for variant in protocol.PROMPT_VARIANTS
        }
    )
    units: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    case_index = 0
    for membership in protocol.SPECIFICITY_MEMBERSHIPS:
        for variant in protocol.PROMPT_VARIANTS:
            for within in range(3):
                matched = matched_by_cell.get((membership, variant), [None] * 3)[within]
                if matched is None:
                    source_id = f"bank_source_{case_index}"
                    source_phrase = f"bank object {case_index}"
                    source_head = f"bank_{case_index}"
                    receiver_id = f"bank_receiver_{case_index}"
                    receiver = f"bank receiver {case_index}"
                    causal_case_id = ""
                else:
                    source_id = str(matched["source_id"])
                    source_phrase = str(matched["source_phrase"])
                    source_head = str(matched["source_head_lemma"])
                    receiver_id = str(matched["receiver_id"])
                    receiver = str(matched["receiver"])
                    causal_case_id = str(matched["semantic_case_id"])
                case = {
                    "specificity_case_id": f"spec_{case_index:02d}",
                    "membership": membership,
                    "prompt_variant": variant,
                    "source_id": source_id,
                    "source_phrase": source_phrase,
                    "source_head_lemma": source_head,
                    "receiver_id": receiver_id,
                    "receiver": receiver,
                    "causal_case_id": causal_case_id,
                    "prompt": f"noncausal prompt {case_index}",
                }
                selected.append(case)
                for replicate in range(2):
                    units.append(
                        {
                            "unit_id": f"su{len(units):03d}",
                            **case,
                            "replicate": replicate,
                            "seed": 5000 + len(units),
                        }
                    )
                case_index += 1
    mapping = [
        {
            "specificity_case_id": row["specificity_case_id"],
            "causal_case_id": row["causal_case_id"],
            "source_id": row["source_id"],
            "source_phrase": row["source_phrase"],
            "receiver_id": row["receiver_id"],
            "receiver": row["receiver"],
        }
        for row in selected
        if row["membership"] == "holdout_source"
    ]
    return units, mapping


def make_causal_scores(units: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for unit in units:
        for method in protocol.METHODS:
            target = 0 if method == "v4" else 2
            footprint = 0 if method == "v4" else 1
            rows.append(
                {
                    **unit,
                    "method": method,
                    protocol.CAUSAL_SCORE_FIELDS[0]: target,
                    protocol.CAUSAL_SCORE_FIELDS[1]: footprint,
                    protocol.CAUSAL_SCORE_FIELDS[2]: 2,
                    protocol.CAUSAL_SCORE_FIELDS[3]: 2,
                    protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]: 2 if method == "original" else "",
                }
            )
    return rows


def make_specificity_scores(units: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **unit,
            "group": "",
            "method": method,
            protocol.SPECIFICITY_SCORE_FIELDS[0]: 2,
            protocol.SPECIFICITY_SCORE_FIELDS[1]: 2,
            protocol.SPECIFICITY_SCORE_FIELDS[2]: 2,
            protocol.SPECIFICITY_SCORE_FIELDS[3]: 2,
        }
        for unit in units
        for method in protocol.METHODS
    ]


class CommitmentAndFailClosedTests(unittest.TestCase):
    def _write_registry(self, path: Path, dataset: str, stage: int, stage0_sha: str | None = None):
        artifacts = {}
        for name in protocol.STAGE_ARTIFACTS[(dataset, stage)]:
            artifacts[name] = {
                "sha256": hashlib.sha256(name.encode()).hexdigest(),
                "size_bytes": 1,
                "row_count": protocol.EXPECTED_COMMITMENT_ROW_COUNTS.get(
                    (dataset, stage, name)
                ),
            }
        payload = {
            "protocol": protocol.COMMITMENT_PROTOCOL,
            "dataset": dataset,
            "dataset_version": protocol.DATASET_VERSION,
            "stage": stage,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "artifacts": artifacts,
        }
        if stage == 1:
            payload["stage0_registry_sha256"] = stage0_sha
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return payload

    def test_stage1_is_hash_chained_and_missing_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage0 = root / "stage0.json"
            stage1 = root / "stage1.json"
            self._write_registry(stage0, "causal", 0)
            self._write_registry(stage1, "causal", 1, protocol.file_sha256(stage0))
            protocol.validate_commitment_registry(stage0, dataset="causal", stage=0)
            protocol.validate_commitment_registry(
                stage1,
                dataset="causal",
                stage=1,
                expected_stage0_sha256=protocol.file_sha256(stage0),
            )
            stage0.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact Stage-0"):
                protocol.validate_commitment_registry(
                    stage1,
                    dataset="causal",
                    stage=1,
                    expected_stage0_sha256=protocol.file_sha256(stage0),
                )
            with self.assertRaisesRegex(FileNotFoundError, "training authorization"):
                protocol.validate_training_authorization(root, expected_gate_spec=protocol.GATE_SPEC)

    def test_checkpoint_and_final36_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint eligibility"):
                protocol.validate_checkpoint_eligibility(root)
            with self.assertRaisesRegex(ValueError, "sealed-final36"):
                protocol.resolve_path(root, "data/v3c_sealed_final36.csv")

    def test_failed_sanity_and_pseudo_training_state_cannot_be_wrapped_eligible(self) -> None:
        observations = []
        for index in range(16):
            raw = 0.01
            observations.append(
                {
                    "global_step": index + 1,
                    "erase_ordinal": index,
                    "manifest_index": index,
                    "scene_id": f"scene_{index:02d}",
                    "assigned_source_id": f"source_{index:02d}",
                    "flow_loss": 1.0,
                    "target_prompt_teacher_loss": raw,
                    "raw_loss_ratio": raw,
                    "weighted_output_gradient_norm_ratio": 0.4,
                }
            )
        sanity = {
            "protocol": protocol.SCALE_SANITY_PROTOCOL,
            "status": "passed",
            "dataset_version": protocol.DATASET_VERSION,
            "passed": True,
            "run_registration_sha256": "a" * 64,
            "calibration_id": protocol.EXPECTED_TRAINING_CONFIG[
                "target_prompt_calibration_id"
            ],
            "formula": "g_i = 4 * sqrt(target_prompt_teacher_loss / flow_loss)",
            "aggregation": "arithmetic_mean_over_first_16_actual_erase_updates",
            "weight": 4.0,
            "mean_min": 0.2,
            "mean_max": 0.5,
            "single_max": 1.0,
            "observation_count": 16,
            "mean_raw_loss_ratio": 0.01,
            "mean_weighted_loss_ratio": 0.04,
            "mean_weighted_output_grad_ratio": 0.4,
            "median_weighted_output_grad_ratio": 0.4,
            "max_weighted_output_grad_ratio": 0.4,
            "observations": observations,
        }
        protocol.validate_scale_sanity_payload(
            sanity, expected_run_registration_sha256="a" * 64
        )
        failed = json.loads(json.dumps(sanity))
        failed["status"] = "registered_scale_sanity_termination"
        failed["passed"] = False
        with self.assertRaisesRegex(ValueError, "did not pass"):
            protocol.validate_scale_sanity_payload(
                failed, expected_run_registration_sha256="a" * 64
            )

        finite = {
            "protocol": protocol.FINAL_LORA_FINITE_PROTOCOL,
            "status": "passed",
            "digest_algorithm": "sha256_sorted_name_shape_dtype_raw_bytes_v1",
            "trainable_parameter_count": 1,
            "trainable_element_count": 1,
            "nonfinite_trainable_parameter_count": 0,
            "nonfinite_trainable_element_count": 0,
            "lora_state_tensor_count": 1,
            "lora_state_element_count": 1,
            "nonfinite_lora_state_tensor_count": 0,
            "nonfinite_lora_state_element_count": 0,
            "trainable_state_sha256": "b" * 64,
            "lora_state_sha256": "c" * 64,
        }
        state = {
            "protocol": protocol.TRAINING_STATE_PROTOCOL,
            "status": "eligible_training_complete",
            "dataset_version": protocol.DATASET_VERSION,
            "step": 199,
            "max_steps": 200,
            "only_training_intervention": protocol.ONLY_TRAINING_INTERVENTION,
            "training_config": protocol.EXPECTED_TRAINING_CONFIG,
            "manifest": protocol.TRAINING_MANIFEST,
            "train_manifest_sha256": "d" * 64,
            "base_cache_inventory_sha256": "d" * 64,
            "teacher_cache_inventory_sha256": "d" * 64,
            "source_bank_registry_sha256": "d" * 64,
            "holdout_public_commitment_sha256": "d" * 64,
            "holdout_public_commitment_path": protocol.PUBLIC_HOLDOUT_COMMITMENT,
            "holdout_count": 24,
            "source_mapping_registry_sha256": "d" * 64,
            "active100_mapping_sha256": "d" * 64,
            "full178_mapping_sha256": "d" * 64,
            "canonical_prompt_builder_path": protocol.CANONICAL_PROMPT_BUILDER,
            "canonical_prompt_builder_sha256": "d" * 64,
            "prompt_sidecar_inventory_sha256": "d" * 64,
            "prompt_sidecar_manifest_sha256": "d" * 64,
            "model_content_inventory_sha256": protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256,
            "transformer_inventory_sha256": "d" * 64,
            "runtime_registry_path": protocol.RUNTIME_REGISTRY,
            "runtime_registry_sha256": "d" * 64,
            "preflight_artifact_sha256": "d" * 64,
            "training_authorization_path": protocol.TRAINING_AUTHORIZATION,
            "training_authorization_sha256": "d" * 64,
            "training_code_registry_path": protocol.TRAINING_CODE_REGISTRY,
            "training_code_registry_sha256": "d" * 64,
            "run_registration_sha256": "d" * 64,
            "scale_sanity_sha256": "d" * 64,
            "initial_lora_sha256": "d" * 64,
            "sample_order_sha256": "d" * 64,
            "noise_sigma_rng_initial_sha256": "d" * 64,
            "noise_sigma_rng_final_sha256": "d" * 64,
            "role_step_counts": {"erase": 100, "preserve": 100},
            "active_source_counts": {"source": 100},
            "mean_loss_last_20": 1.0,
            "mean_target_prompt_teacher_loss_last_20": 1.0,
            "mean_preserve_loss_last_20": 1.0,
            "trainer_sha256": "d" * 64,
            "launcher_sha256": "d" * 64,
            "final_lora_finite_check": finite,
        }
        with self.assertRaisesRegex(ValueError, "identity/configuration"):
            protocol.validate_training_state_payload(
                Path("."),
                state,
                eligibility={},
                registration={},
                preflight={},
                sanity={},
                authorization={},
                code_registry={},
                weights_path=Path("missing.safetensors"),
            )

    def test_saved_lora_safetensors_is_independently_checked_for_finiteness(self) -> None:
        evidence = {
            "protocol": protocol.FINAL_LORA_FINITE_PROTOCOL,
            "status": "passed",
            "digest_algorithm": "sha256_sorted_name_shape_dtype_raw_bytes_v1",
            "trainable_parameter_count": 1,
            "trainable_element_count": 1,
            "nonfinite_trainable_parameter_count": 0,
            "nonfinite_trainable_element_count": 0,
            "lora_state_tensor_count": 1,
            "lora_state_element_count": 1,
            "nonfinite_lora_state_tensor_count": 0,
            "nonfinite_lora_state_element_count": 0,
            "trainable_state_sha256": "a" * 64,
            "lora_state_sha256": "b" * 64,
        }

        def safetensor(value: float) -> bytes:
            header = json.dumps(
                {
                    "adapter.weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            return struct.pack("<Q", len(header)) + header + struct.pack("<f", value)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            path.write_bytes(safetensor(1.0))
            self.assertEqual(
                protocol.validate_safetensors_finite_inventory(
                    path, expected_evidence=evidence
                ),
                {"tensor_count": 1, "element_count": 1},
            )
            path.write_bytes(safetensor(float("inf")))
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                protocol.validate_safetensors_finite_inventory(
                    path, expected_evidence=evidence
                )

    def test_v2_checkpoint_identity_and_paths_reject_v1_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eligibility_path = root / protocol.CHECKPOINT_ELIGIBILITY
            payload = {
                "protocol": protocol.CHECKPOINT_ELIGIBILITY_PROTOCOL,
                "status": "eligible",
                "dataset_version": "v4_dev72_v1",
                "step": 200,
                "checkpoint": {
                    "path": protocol.V4_CHECKPOINT,
                    "weights_sha256": "a" * 64,
                    "training_state_sha256": "b" * 64,
                },
                "run_registration": {"path": "run_registration_v2.json", "sha256": "c" * 64},
                "preflight": {"path": "null_sidecar_preflight_v2.json", "sha256": "d" * 64},
                "scale_sanity": {"path": "target_prompt_scale_sanity_v2.json", "sha256": "e" * 64},
                "role_step_counts": {"erase": 100, "preserve": 100},
                "final_lora_finite_check": {
                    "protocol": protocol.FINAL_LORA_FINITE_PROTOCOL,
                    "status": "passed",
                    "digest_algorithm": "sha256_sorted_name_shape_dtype_raw_bytes_v1",
                    "trainable_parameter_count": 1,
                    "trainable_element_count": 1,
                    "nonfinite_trainable_parameter_count": 0,
                    "nonfinite_trainable_element_count": 0,
                    "lora_state_tensor_count": 1,
                    "lora_state_element_count": 1,
                    "nonfinite_lora_state_tensor_count": 0,
                    "nonfinite_lora_state_element_count": 0,
                    "trainable_state_sha256": "a" * 64,
                    "lora_state_sha256": "b" * 64,
                },
                **{name: "f" * 64 for name in protocol.CHECKPOINT_HASH_FIELDS},
            }
            eligibility_path.parent.mkdir(parents=True, exist_ok=True)
            eligibility_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "dataset version mismatch"):
                protocol.validate_checkpoint_eligibility(root, eligibility_path)
            payload["dataset_version"] = protocol.DATASET_VERSION
            payload["model_content_inventory_sha256"] = (
                protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256
            )
            payload["transformer_inventory_sha256"] = (
                protocol.FROZEN_TRANSFORMER_INVENTORY_SHA256
            )
            payload["source_bank_registry_sha256"] = (
                protocol.FROZEN_PUBLIC_SOURCE_BANK_SHA256
            )
            payload["train_manifest_sha256"] = protocol.FROZEN_TRAIN_MANIFEST_SHA256
            payload["base_cache_inventory_sha256"] = (
                protocol.FROZEN_BASE_CACHE_INVENTORY_SHA256
            )
            payload["teacher_cache_inventory_sha256"] = (
                protocol.FROZEN_TEACHER_CACHE_INVENTORY_SHA256
            )
            payload["sample_order_sha256"] = protocol.FROZEN_SAMPLE_ORDER_SHA256
            payload["noise_sigma_rng_initial_sha256"] = (
                protocol.FROZEN_NOISE_SIGMA_RNG_INITIAL_SHA256
            )
            payload["noise_sigma_rng_final_sha256"] = (
                protocol.FROZEN_NOISE_SIGMA_RNG_FINAL_SHA256
            )
            payload["initial_lora_sha256"] = protocol.FROZEN_INITIAL_LORA_SHA256
            payload["run_registration"] = {
                "path": (
                    "outputs/water_impact_dynamic_v4/adapter_source_slot_randomized_v1/"
                    "run_registration_v1.json"
                ),
                "sha256": "c" * 64,
            }
            payload["preflight"]["path"] = protocol.NULL_SIDECAR_PREFLIGHT
            payload["scale_sanity"]["path"] = protocol.SCALE_SANITY
            eligibility_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "outside the v2 namespace"):
                protocol.validate_checkpoint_eligibility(root, eligibility_path)

        v2_paths = (
            protocol.CAUSAL_STAGE0,
            protocol.CAUSAL_STAGE1,
            protocol.SPECIFICITY_STAGE0,
            protocol.SPECIFICITY_STAGE1,
            protocol.PUBLIC_SOURCE_BANK,
            protocol.PUBLIC_HOLDOUT_COMMITMENT,
            protocol.GATE_REGISTRY,
            protocol.TRAINING_CODE_REGISTRY,
            protocol.RUNTIME_REGISTRY,
            protocol.TRAINING_AUTHORIZATION,
            protocol.CHECKPOINT_ELIGIBILITY,
            protocol.V4_CHECKPOINT,
            protocol.RUN_REGISTRATION,
            protocol.SCALE_SANITY,
            protocol.TRAINING_STATE,
            protocol.NULL_SIDECAR_PREFLIGHT,
            protocol.PROMPT_SIDECAR_DIR,
            protocol.PROMPT_SIDECAR_MANIFEST,
        )
        self.assertTrue(all("v2" in path for path in v2_paths))

    def test_runtime_and_code_registry_are_byte_cross_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / protocol.RUNTIME_REGISTRY
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text(
                json.dumps(protocol.RUNTIME_REGISTRY_PAYLOAD) + "\n",
                encoding="utf-8",
            )
            artifacts: dict[str, dict[str, str]] = {}
            for name, relative in protocol.TRAINING_CODE_ARTIFACTS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"synthetic {name}\n", encoding="utf-8")
                artifacts[name] = {
                    "path": relative,
                    "sha256": protocol.file_sha256(path),
                }
            code = root / protocol.TRAINING_CODE_REGISTRY
            code.write_text(
                json.dumps(
                    {
                        "protocol": protocol.TRAINING_CODE_REGISTRY_PROTOCOL,
                        "status": "frozen",
                        "runtime_registry": {
                            "path": protocol.RUNTIME_REGISTRY,
                            "sha256": protocol.file_sha256(runtime_path),
                        },
                        "artifacts": artifacts,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            protocol.validate_training_code_registry(root, code)
            runtime_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
                protocol.validate_training_code_registry(root, code)

    def test_gate_registry_rejects_self_consistent_to_be_frozen_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate_spec = json.loads(json.dumps(protocol.GATE_SPEC))
            gate_spec["decision"] = "TO_BE_FROZEN_AFTER_FORMAL_REVIEW"
            payload = {
                "protocol": protocol.GATE_REGISTRY_PROTOCOL,
                "status": "frozen",
                "dataset_version": protocol.DATASET_VERSION,
                "sealed_final36_status": "unopened",
                "gate_spec": gate_spec,
                "gate_spec_sha256": protocol.canonical_json_sha256(gate_spec),
                "scorer_sha256": "a" * 64,
            }
            path = root / "gate.json"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "placeholder"):
                protocol.validate_gate_registry(path, gate_spec)

    def test_training_authorization_rejects_public_stage0_mix_and_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / relative
                for name, relative in protocol.TRAINING_AUTHORIZATION_REFS.items()
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            paths["source_bank_registry"].write_text(
                "synthetic-public-bank-v1\n", encoding="utf-8"
            )
            private_holdout_sha = "a" * 64
            paths["holdout_public_commitment"].write_text(
                json.dumps(
                    {"holdout_registry_file_sha256": private_holdout_sha}
                )
                + "\n",
                encoding="utf-8",
            )
            for name, path in paths.items():
                if name not in {"source_bank_registry", "holdout_public_commitment"}:
                    path.write_text(f"synthetic-{name}\n", encoding="utf-8")
            authorization = root / protocol.TRAINING_AUTHORIZATION

            def payload() -> dict[str, object]:
                return {
                    "protocol": protocol.TRAINING_AUTHORIZATION_PROTOCOL,
                    "status": "authorized",
                    "dataset_version": protocol.DATASET_VERSION,
                    "sealed_final36_status": "unopened",
                    **{
                        name: {
                            "path": relative,
                            "sha256": protocol.file_sha256(paths[name]),
                        }
                        for name, relative in protocol.TRAINING_AUTHORIZATION_REFS.items()
                    },
                }

            authorization.write_text(json.dumps(payload()) + "\n", encoding="utf-8")
            causal0 = {
                "artifacts": {
                    "source_bank_registry_64": {
                        "sha256": protocol.file_sha256(
                            paths["source_bank_registry"]
                        ),
                        "size_bytes": paths["source_bank_registry"].stat().st_size,
                        "row_count": 64,
                    },
                    "holdout_registry_24": {
                        "sha256": private_holdout_sha,
                        "size_bytes": 1,
                        "row_count": 24,
                    },
                }
            }
            runtime_ref = payload()["runtime_registry"]
            with (
                mock.patch.object(
                    protocol,
                    "FROZEN_PUBLIC_SOURCE_BANK_SHA256",
                    protocol.file_sha256(paths["source_bank_registry"]),
                ),
                mock.patch.object(
                    protocol,
                    "FROZEN_PUBLIC_HOLDOUT_COMMITMENT_SHA256",
                    protocol.file_sha256(paths["holdout_public_commitment"]),
                ),
                mock.patch.object(
                    protocol, "validate_commitment_registry", return_value=causal0
                ),
                mock.patch.object(protocol, "validate_gate_registry"),
                mock.patch.object(protocol, "validate_runtime_registry"),
                mock.patch.object(
                    protocol,
                    "validate_training_code_registry",
                    return_value={"runtime_registry": runtime_ref},
                ),
            ):
                protocol.validate_training_authorization(
                    root,
                    expected_gate_spec=protocol.GATE_SPEC,
                    authorization_path=authorization,
                )
                paths["source_bank_registry"].write_text(
                    "synthetic-public-bank-v2\n", encoding="utf-8"
                )
                authorization.write_text(
                    json.dumps(payload()) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "public source bank"):
                    protocol.validate_training_authorization(
                        root,
                        expected_gate_spec=protocol.GATE_SPEC,
                        authorization_path=authorization,
                    )

                paths["source_bank_registry"].write_text(
                    "synthetic-public-bank-v1\n", encoding="utf-8"
                )
                paths["holdout_public_commitment"].write_text(
                    json.dumps({"holdout_registry_file_sha256": "b" * 64})
                    + "\n",
                    encoding="utf-8",
                )
                authorization.write_text(
                    json.dumps(payload()) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "public holdout"):
                    protocol.validate_training_authorization(
                        root,
                        expected_gate_spec=protocol.GATE_SPEC,
                        authorization_path=authorization,
                    )

    def test_selection_binding_opens_exact_rules_and_rejects_wrong_domain(self) -> None:
        components = {
            name: hashlib.sha256(name.encode("utf-8")).hexdigest()
            for name in protocol.CAUSAL_PRIVATE_COMPONENT_FILENAMES
        }
        binding = protocol.expected_selection_binding(
            dataset="causal",
            public_pending_sha256="1" * 64,
            root_bundle_sha256="2" * 64,
            component_sha256=components,
            generation_spec_sha256="3" * 64,
            model_inventory_sha256="4" * 64,
            runtime_registry_sha256="9" * 64,
            screening_seed_sha256="5" * 64,
            selector_salt_sha256="6" * 64,
            evaluation_seed_salt_sha256="7" * 64,
            forbidden_seed_inventory_sha256="8" * 64,
            preselection_seed_audit_sha256="f" * 64,
            preselection_seed_count=144,
        )
        self.assertEqual(binding["ranking_contract"]["domain"], "causal-selector-v2")
        self.assertEqual(binding["seed_contract"]["domain"], "causal-eval-seed-v2")
        self.assertEqual(binding["seed_contract"]["digest_projection"], "first 4 bytes, big-endian unsigned uint32")
        self.assertIn("semantic_case_id", binding["seed_contract"]["id_binding"])
        specificity_components = {
            name: hashlib.sha256(name.encode("utf-8")).hexdigest()
            for name in protocol.SPECIFICITY_PRIVATE_COMPONENT_FILENAMES
        }
        specificity = protocol.expected_selection_binding(
                dataset="specificity",
                public_pending_sha256="1" * 64,
                root_bundle_sha256="2" * 64,
                component_sha256=specificity_components,
                generation_spec_sha256="3" * 64,
                model_inventory_sha256="4" * 64,
                runtime_registry_sha256="9" * 64,
                screening_seed_sha256="5" * 64,
                selector_salt_sha256="6" * 64,
                evaluation_seed_salt_sha256="7" * 64,
                forbidden_seed_inventory_sha256="8" * 64,
                preselection_seed_audit_sha256="f" * 64,
                preselection_seed_count=72,
            )
        self.assertEqual(
            specificity["ranking_contract"]["domain"], "specificity-selector-v2"
        )

    def test_real_v2_curator_bytes_open_when_private_fixture_is_available(self) -> None:
        """Exercise the frozen curator bytes without publishing a Stage-0 wrapper."""

        registered_private = os.environ.get("V4_PRIVATE_REGISTRY_V2_DIR")
        if not registered_private:
            self.skipTest("V4_PRIVATE_REGISTRY_V2_DIR is not configured")
        source_private = Path(registered_private)
        if not source_private.is_dir() or source_private.is_symlink():
            self.fail("V4_PRIVATE_REGISTRY_V2_DIR must be a real directory")

        required_private = (
            *protocol.CAUSAL_PRIVATE_COMPONENT_FILENAMES,
            *protocol.CAUSAL_ONTOLOGY_FILENAMES,
            "causal_stage0_bundle_private_v2.json",
        )
        for name in required_private:
            path = source_private / name
            if not path.is_file() or path.is_symlink():
                self.fail(f"missing real v2 private fixture component: {name}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private"
            private.mkdir()
            for name in required_private:
                shutil.copyfile(source_private / name, private / name)
            (root / "scripts").mkdir()
            shutil.copyfile(
                PROJECT_ROOT / "scripts/build_water_impact_dynamic_pairs_v1.py",
                root / "scripts/build_water_impact_dynamic_pairs_v1.py",
            )
            for registered in (
                protocol.PUBLIC_SOURCE_BANK,
                protocol.PUBLIC_HOLDOUT_COMMITMENT,
                protocol.PENDING_STAGE0_COMMITMENTS["causal"],
            ):
                destination = root / registered
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT_ROOT / registered, destination)

            def write_json(path: Path, payload: object) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            runtime = root / protocol.RUNTIME_REGISTRY
            write_json(runtime, protocol.RUNTIME_REGISTRY_PAYLOAD)
            generation_spec = private / "causal_generation_spec_v2.json"
            write_json(
                generation_spec,
                {
                    "protocol": protocol.GENERATION_SPEC_PROTOCOL,
                    "status": "frozen_before_original_render",
                    "model_inventory_sha256": protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256,
                    "runtime_registry": {
                        "path": protocol.RUNTIME_REGISTRY,
                        "sha256": protocol.file_sha256(runtime),
                    },
                    "generation_spec": protocol.GENERATION_SPEC,
                    "source_mode": "Original_screening_then_matched_O_v3b_v4",
                },
            )
            secrets = json.loads(
                (private / "causal_stage0_secrets_private_v2.json").read_text(
                    encoding="utf-8"
                )
            )
            screening = private / "causal_screening_seed_v2.txt"
            selector_salt = private / "causal_selector_salt_v2.txt"
            evaluation_salt = private / "causal_evaluation_seed_salt_v2.txt"
            screening.write_text(f"{secrets['screening_seed']}\n", encoding="ascii")
            selector_salt.write_text(f"{secrets['selector_salt']}\n", encoding="ascii")
            evaluation_salt.write_text(
                f"{secrets['evaluation_seed_salt']}\n", encoding="ascii"
            )
            forbidden = private / "causal_forbidden_seed_inventory_v2.json"
            write_json(
                forbidden,
                {
                    "protocol": protocol.FORBIDDEN_SEED_INVENTORY_PROTOCOL,
                    "dataset": "causal",
                    "status": "frozen_by_independent_seed_auditor",
                    "seed_encoding": "nonnegative JSON integer below 2^63",
                    "source_commitments": [
                        {
                            "name": "synthetic_empty_history_for_schema_regression",
                            "sha256": "e" * 64,
                            "seed_count": 0,
                        }
                    ],
                    "seeds": [],
                },
            )
            binding = protocol.prepare_selection_binding(
                root,
                dataset="causal",
                private_root=private,
                candidate_manifest_path=private
                / "causal_stage0_candidates_private_v2.json",
                canonical_templates_path=private
                / "causal_stage0_templates_private_v2.json",
                field_rules_path=private
                / "causal_stage0_field_rules_private_v2.json",
                render_configuration_path=private
                / "causal_stage0_render_config_private_v2.json",
                selection_rules_path=private
                / "causal_stage0_selection_rules_private_v2.json",
                secrets_path=private / "causal_stage0_secrets_private_v2.json",
                root_bundle_path=private / "causal_stage0_bundle_private_v2.json",
                generation_spec_path=generation_spec,
                screening_seed_path=screening,
                selector_salt_path=selector_salt,
                evaluation_seed_salt_path=evaluation_salt,
                forbidden_seed_inventory_path=forbidden,
                source_ontology_path=private / "source_ontology_private80_v2.json",
                source_split_path=private / "source_split_private_v2.json",
                holdout_registry_path=private / "holdout_registry_private24_v2.json",
                receiver_ontology_path=private / "receiver_ontology_private32_v2.json",
            )
            self.assertEqual(binding["dataset_version"], protocol.DATASET_VERSION)
            self.assertEqual(
                binding["seed_contract"]["preselection_audit"]["derived_seed_count"],
                144,
            )
            binding_path = private / "causal_selection_binding_v2.json"
            write_json(binding_path, binding)

            def record(path: Path, rows: int | None = None) -> dict[str, object]:
                return {
                    "sha256": protocol.file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": rows,
                }

            rules = private / "causal_stage0_selection_rules_private_v2.json"
            rules_record = record(rules)
            stage0_payload = {
                "protocol": protocol.COMMITMENT_PROTOCOL,
                "dataset": "causal",
                "dataset_version": protocol.DATASET_VERSION,
                "stage": 0,
                "status": "committed",
                "sealed_final36_status": "unopened",
                "artifacts": {
                    "candidate_manifest_48": record(
                        private / "causal_stage0_candidates_private_v2.json", 48
                    ),
                    "source_bank_registry_64": record(
                        root / protocol.PUBLIC_SOURCE_BANK, 64
                    ),
                    "source_ontology_80": record(
                        private / "source_ontology_private80_v2.json", 80
                    ),
                    "source_split_80": record(
                        private / "source_split_private_v2.json", 80
                    ),
                    "holdout_registry_24": record(
                        private / "holdout_registry_private24_v2.json", 24
                    ),
                    "receiver_ontology_32": record(
                        private / "receiver_ontology_private32_v2.json", 32
                    ),
                    "canonical_templates": record(
                        private / "causal_stage0_templates_private_v2.json"
                    ),
                    "field_normalization": record(
                        private / "causal_stage0_field_rules_private_v2.json"
                    ),
                    "raw_root_bundle": record(
                        private / "causal_stage0_bundle_private_v2.json"
                    ),
                    "raw_render_configuration": record(
                        private / "causal_stage0_render_config_private_v2.json"
                    ),
                    "stage0_secrets": record(
                        private / "causal_stage0_secrets_private_v2.json"
                    ),
                    "screening_seed": record(screening),
                    "screening_generation_spec": record(generation_spec),
                    "selector_salt": record(selector_salt),
                    "ranking_formula": rules_record,
                    "constrained_subset_algorithm": dict(rules_record),
                    "evaluation_seed_salt": record(evaluation_salt),
                    "seed_derivation_formula": record(binding_path),
                    "forbidden_seed_inventory": record(forbidden),
                },
            }
            stage0_path = root / protocol.CAUSAL_STAGE0
            write_json(stage0_path, stage0_payload)
            stage0 = protocol.validate_commitment_registry(
                stage0_path, dataset="causal", stage=0
            )
            protocol.validate_selection_contract_opening(
                root,
                dataset="causal",
                stage0_registry=stage0,
                private_root=private,
                candidate_manifest_path=private
                / "causal_stage0_candidates_private_v2.json",
                canonical_templates_path=private
                / "causal_stage0_templates_private_v2.json",
                field_rules_path=private
                / "causal_stage0_field_rules_private_v2.json",
                render_configuration_path=private
                / "causal_stage0_render_config_private_v2.json",
                selection_rules_path=rules,
                secrets_path=private / "causal_stage0_secrets_private_v2.json",
                root_bundle_path=private / "causal_stage0_bundle_private_v2.json",
                generation_spec_path=generation_spec,
                screening_seed_path=screening,
                selector_salt_path=selector_salt,
                evaluation_seed_salt_path=evaluation_salt,
                forbidden_seed_inventory_path=forbidden,
                selection_binding_path=binding_path,
                source_ontology_path=private / "source_ontology_private80_v2.json",
                source_split_path=private / "source_split_private_v2.json",
                holdout_registry_path=private / "holdout_registry_private24_v2.json",
                receiver_ontology_path=private / "receiver_ontology_private32_v2.json",
            )

    def test_full_stage0_opening_rejects_component_and_binding_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_synthetic_stage0_fixture(Path(directory))
            components = fixture["components"]
            assert isinstance(components, dict)
            stage0 = protocol.validate_commitment_registry(
                fixture["stage0"], dataset="causal", stage=0
            )

            def validate() -> None:
                ontologies = fixture["ontologies"]
                assert isinstance(ontologies, dict)
                with synthetic_public_hash_patch(fixture):
                    protocol.validate_selection_contract_opening(
                        fixture["root"],
                        dataset="causal",
                        stage0_registry=stage0,
                        private_root=fixture["private"],
                        candidate_manifest_path=components[
                            "causal_stage0_candidates_private_v2.json"
                        ],
                        canonical_templates_path=components[
                            "causal_stage0_templates_private_v2.json"
                        ],
                        field_rules_path=components[
                            "causal_stage0_field_rules_private_v2.json"
                        ],
                        render_configuration_path=components[
                            "causal_stage0_render_config_private_v2.json"
                        ],
                        selection_rules_path=components[
                            "causal_stage0_selection_rules_private_v2.json"
                        ],
                        secrets_path=components[
                            "causal_stage0_secrets_private_v2.json"
                        ],
                        root_bundle_path=fixture["bundle"],
                        generation_spec_path=fixture["generation_spec"],
                        screening_seed_path=fixture["screening_seed"],
                        selector_salt_path=fixture["selector_salt"],
                        evaluation_seed_salt_path=fixture["evaluation_salt"],
                        forbidden_seed_inventory_path=fixture["forbidden"],
                        selection_binding_path=fixture["binding"],
                        source_ontology_path=ontologies["source_ontology_80"],
                        source_split_path=ontologies["source_split_80"],
                        holdout_registry_path=ontologies["holdout_registry_24"],
                        receiver_ontology_path=ontologies["receiver_ontology_32"],
                    )

            validate()
            pending = fixture["root"] / protocol.PENDING_STAGE0_COMMITMENTS["causal"]
            pending_bytes = pending.read_bytes()
            pending_payload = json.loads(pending.read_text(encoding="utf-8"))
            pending_payload["supersedes"]["aggregate_audit"][
                "stage0_global_constraint_feasible"
            ] = True
            pending.write_text(
                json.dumps(pending_payload, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "public causal Stage-0 pending bytes"
            ):
                validate()
            pending.write_bytes(pending_bytes)
            validate()
            field_rules = components["causal_stage0_field_rules_private_v2.json"]
            field_rules.write_text(field_rules.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "component digest"):
                validate()

    def test_causal_opening_pins_public_bytes_and_rejects_hash_rebinding(self) -> None:
        def prepare(fixture: dict[str, object]) -> None:
            components = fixture["components"]
            ontologies = fixture["ontologies"]
            assert isinstance(components, dict)
            assert isinstance(ontologies, dict)
            with synthetic_public_hash_patch(fixture):
                protocol.prepare_selection_binding(
                    fixture["root"],
                    dataset="causal",
                    private_root=fixture["private"],
                    candidate_manifest_path=components[
                        "causal_stage0_candidates_private_v2.json"
                    ],
                    canonical_templates_path=components[
                        "causal_stage0_templates_private_v2.json"
                    ],
                    field_rules_path=components[
                        "causal_stage0_field_rules_private_v2.json"
                    ],
                    render_configuration_path=components[
                        "causal_stage0_render_config_private_v2.json"
                    ],
                    selection_rules_path=components[
                        "causal_stage0_selection_rules_private_v2.json"
                    ],
                    secrets_path=components[
                        "causal_stage0_secrets_private_v2.json"
                    ],
                    root_bundle_path=fixture["bundle"],
                    generation_spec_path=fixture["generation_spec"],
                    screening_seed_path=fixture["screening_seed"],
                    selector_salt_path=fixture["selector_salt"],
                    evaluation_seed_salt_path=fixture["evaluation_salt"],
                    forbidden_seed_inventory_path=fixture["forbidden"],
                    source_ontology_path=ontologies["source_ontology_80"],
                    source_split_path=ontologies["source_split_80"],
                    holdout_registry_path=ontologies["holdout_registry_24"],
                    receiver_ontology_path=ontologies["receiver_ontology_32"],
                )

        public_cases = (
            (
                "pending",
                protocol.PENDING_STAGE0_COMMITMENTS["causal"],
                "public causal Stage-0 pending bytes",
            ),
            (
                "source_bank",
                protocol.PUBLIC_SOURCE_BANK,
                "public source bank bytes",
            ),
            (
                "holdout",
                protocol.PUBLIC_HOLDOUT_COMMITMENT,
                "public holdout commitment bytes",
            ),
        )
        for label, registered, error in public_cases:
            with self.subTest(public_artifact=label), tempfile.TemporaryDirectory() as directory:
                fixture = make_synthetic_stage0_fixture(Path(directory))
                path = Path(fixture["root"]) / registered
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, error):
                    prepare(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = make_synthetic_stage0_fixture(Path(directory))
            components = fixture["components"]
            assert isinstance(components, dict)
            candidate_path = components[
                "causal_stage0_candidates_private_v2.json"
            ]
            candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate = candidates["candidates"][0]
            candidate["case_id"] = "attacker_rebound_case_id"
            canonical = dict(candidate)
            canonical.pop("canonical_record_sha256")
            candidate["canonical_record_sha256"] = hashlib.sha256(
                (
                    json.dumps(
                        canonical,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            candidate_path.write_text(
                json.dumps(candidates, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            bundle_path = Path(fixture["bundle"])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["components"][
                "causal_stage0_candidates_private_v2.json"
            ] = protocol.file_sha256(candidate_path)
            bundle_path.write_text(
                json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            pending_path = (
                Path(fixture["root"])
                / protocol.PENDING_STAGE0_COMMITMENTS["causal"]
            )
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            pending["candidate_manifest_sha256"] = protocol.file_sha256(
                candidate_path
            )
            pending["stage0_bundle_file_sha256"] = protocol.file_sha256(
                bundle_path
            )
            pending_path.write_text(
                json.dumps(pending, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "public causal Stage-0 pending bytes"
            ):
                prepare(fixture)

    def test_specificity_stage0_authorizer_selector_W_M_and_stage1_are_reachable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_synthetic_specificity_stage0_fixture(Path(directory))
            components = fixture["specificity_components"]
            assert isinstance(components, dict)
            args = argparse.Namespace(
                private_root=fixture["private"],
                candidate_manifest=components[
                    "specificity_stage0_candidates_private_v2.json"
                ],
                new_bank_assignment=components[
                    "specificity_stage0_new_bank_assignment_private_v2.json"
                ],
                canonical_templates=components[
                    "specificity_stage0_templates_private_v2.json"
                ],
                field_normalization=components[
                    "specificity_stage0_field_rules_private_v2.json"
                ],
                render_configuration=components[
                    "specificity_stage0_render_config_private_v2.json"
                ],
                selection_rules=components[
                    "specificity_stage0_selection_rules_private_v2.json"
                ],
                stage0_secrets=components[
                    "specificity_stage0_secrets_private_v2.json"
                ],
                root_bundle=fixture["specificity_bundle"],
                generation_spec=fixture["generation_spec"],
                screening_seed_file=fixture["specificity_screening_seed"],
                selector_salt_file=fixture["specificity_selector_salt"],
                evaluation_seed_salt_file=fixture["specificity_evaluation_salt"],
                forbidden_seed_inventory=fixture["specificity_forbidden"],
                causal_stage0_registry=fixture["stage0"],
                causal_stage1_registry=fixture["causal_stage1"],
                causal_selected=fixture["causal_selected_path"],
                causal_unit_manifest=fixture["causal_units_path"],
                selection_binding_output=fixture["specificity_binding"],
                stage0_output=fixture["specificity_stage0"],
            )
            with (
                mock.patch.object(runner.Path, "cwd", return_value=fixture["root"]),
                mock.patch.object(
                    protocol,
                    "model_artifact_inventory",
                    return_value={
                        "sha256": protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256
                    },
                ),
            ):
                self.assertEqual(runner.authorize_specificity_stage0(args), 0)
            stage0 = protocol.validate_commitment_registry(
                fixture["specificity_stage0"], dataset="specificity", stage=0
            )
            protocol.validate_selection_contract_opening(
                fixture["root"],
                dataset="specificity",
                stage0_registry=stage0,
                private_root=fixture["private"],
                candidate_manifest_path=args.candidate_manifest,
                new_bank_assignment_path=args.new_bank_assignment,
                canonical_templates_path=args.canonical_templates,
                field_rules_path=args.field_normalization,
                render_configuration_path=args.render_configuration,
                selection_rules_path=args.selection_rules,
                secrets_path=args.stage0_secrets,
                root_bundle_path=args.root_bundle,
                generation_spec_path=args.generation_spec,
                screening_seed_path=args.screening_seed_file,
                selector_salt_path=args.selector_salt_file,
                evaluation_seed_salt_path=args.evaluation_seed_salt_file,
                forbidden_seed_inventory_path=args.forbidden_seed_inventory,
                selection_binding_path=args.selection_binding_output,
                causal_stage0_registry_path=args.causal_stage0_registry,
                causal_stage1_registry_path=args.causal_stage1_registry,
                causal_selected_path=args.causal_selected,
                causal_unit_manifest_path=args.causal_unit_manifest,
            )

            candidates = protocol.load_normalized_candidate_manifest(
                args.candidate_manifest,
                dataset="specificity",
                canonical_templates_path=args.canonical_templates,
            )
            screened = [
                {
                    **row,
                    **{
                        field: 2
                        for field in selector.SPECIFICITY_SCREENING_FIELDS.values()
                    },
                }
                for row in candidates
            ]
            selected_raw, eligibility = selector.select_specificity_cases(
                screened,
                private_salt="c" * 64,
                causal_cases=fixture["causal_selected"],
            )
            selected = selector._private_case_rows(selected_raw, "specificity")
            causal_seeds = {int(row["seed"]) for row in fixture["causal_units"]}
            units = protocol.derive_unit_rows(
                selected,
                dataset="specificity",
                private_salt="d" * 64,
                forbidden_seeds={1, 2, 3_500_000_001, *causal_seeds},
            )
            protocol.validate_specificity_unit_manifest(
                units,
                causal_cases=fixture["causal_selected"],
                causal_seeds=causal_seeds,
            )
            mapping = [
                {
                    "specificity_case_id": row["specificity_case_id"],
                    "causal_case_id": row["causal_case_id"],
                    "source_id": row["source_id"],
                    "source_phrase": row["source_phrase"],
                    "receiver_id": row["receiver_id"],
                    "receiver": row["receiver"],
                }
                for row in selected
                if row["membership"] == "holdout_source"
            ]
            protocol.validate_holdout_mapping(
                mapping,
                causal_cases=fixture["causal_selected"],
                specificity_cases=selected,
            )
            colliding = [dict(row) for row in units]
            colliding[0]["seed"] = next(iter(causal_seeds))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                protocol.validate_specificity_unit_manifest(
                    colliding,
                    causal_cases=fixture["causal_selected"],
                    causal_seeds=causal_seeds,
                )
            wrong_mapping = [dict(row) for row in mapping]
            wrong_mapping[0]["receiver_id"] = "changed"
            with self.assertRaisesRegex(ValueError, "receiver_id"):
                protocol.validate_holdout_mapping(
                    wrong_mapping,
                    causal_cases=fixture["causal_selected"],
                    specificity_cases=selected,
                )

            output_dir = Path(fixture["private"]) / "specificity_selector_output"
            output_dir.mkdir()
            protocol.write_csv(output_dir / "eligibility_v2.csv", eligibility)
            protocol.write_csv(output_dir / "selected_cases_v2.csv", selected)
            protocol.write_csv(output_dir / "unit_manifest_v2.csv", units)
            protocol.write_csv(output_dir / "holdout_mapping_M_v2.csv", mapping)
            summary = {
                "protocol": protocol.PROTOCOL,
                "dataset": "specificity",
                "dataset_version": protocol.DATASET_VERSION,
                "candidate_count": 36,
                "eligible_count": 36,
                "selected_count": 18,
                "unit_count": 36,
                "selection_rank_tuple": [
                    next(
                        row["selection_rank_sha256"]
                        for row in eligibility
                        if row["candidate_id"] == selected_row["candidate_id"]
                    )
                    for selected_row in selected
                ],
            }
            (output_dir / "selector_output_v2.json").write_text(
                json.dumps(summary) + "\n", encoding="utf-8"
            )
            frozen_root = Path(fixture["private"]) / "specificity_freeze"
            frozen_root.mkdir()
            generation = frozen_root / "screen_generation.json"
            generation.write_text("{}\n", encoding="utf-8")
            answer = frozen_root / "candidate_binding.csv"
            protocol.write_csv(
                answer,
                [{"review_id": f"s{index:03d}"} for index in range(36)],
            )
            package = frozen_root / "package.json"
            package.write_text(
                json.dumps(
                    {
                        "generation_manifest": {
                            "path": str(generation),
                            "sha256": protocol.file_sha256(generation),
                        },
                        "answer_key": {
                            "path": str(answer),
                            "sha256": protocol.file_sha256(answer),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frozen_paths = {"screening_package": package}
            for name in (
                "reviewer_a",
                "reviewer_b",
                "dispute_template",
                "adjudication",
                "canonical_eligibility",
                "adjudication_audit",
            ):
                path = frozen_root / f"{name}.csv"
                if name in {"reviewer_a", "reviewer_b"}:
                    protocol.write_csv(
                        path,
                        [{"review_id": f"s{index:03d}"} for index in range(36)],
                    )
                else:
                    path.write_text("synthetic\n", encoding="utf-8")
                frozen_paths[name] = path
            freeze_payload = {
                "artifacts": {
                    name: {"path": str(path), "sha256": protocol.file_sha256(path)}
                    for name, path in frozen_paths.items()
                }
            }
            freeze_manifest = frozen_root / "freeze.json"
            freeze_manifest.write_text(
                json.dumps(freeze_payload) + "\n", encoding="utf-8"
            )
            stage1_path = Path(fixture["root"]) / protocol.SPECIFICITY_STAGE1
            selector._publish_stage1_registry(
                Path(fixture["root"]),
                dataset="specificity",
                stage0_registry_path=fixture["specificity_stage0"],
                freeze_manifest_path=freeze_manifest,
                freeze_payload=freeze_payload,
                output_dir=output_dir,
                stage1_output=stage1_path,
            )
            protocol.validate_commitment_registry(
                stage1_path,
                dataset="specificity",
                stage=1,
                expected_stage0_sha256=protocol.file_sha256(
                    fixture["specificity_stage0"]
                ),
            )

    def test_preselection_seed_audit_rejects_screen_and_peer_collision(self) -> None:
        candidates = make_causal_screening_candidates()
        with mock.patch.object(protocol, "derive_seed", return_value=3_000_000_001):
            with self.assertRaisesRegex(ValueError, "derived seed collides"):
                protocol.audit_preselection_seed_space(
                    candidates,
                    dataset="causal",
                    private_salt="b" * 64,
                    screening_seed=3_000_000_001,
                    forbidden_seeds={1, 2},
                )

    def test_causal_ontology_and_physical_status_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_synthetic_stage0_fixture(Path(directory))
            ontologies = fixture["ontologies"]
            assert isinstance(ontologies, dict)

            def validate(rows=None) -> None:
                protocol.validate_causal_candidate_ontology_bindings(
                    fixture["root"],
                    fixture["normalized"] if rows is None else rows,
                    source_ontology_path=ontologies["source_ontology_80"],
                    source_split_path=ontologies["source_split_80"],
                    holdout_registry_path=ontologies["holdout_registry_24"],
                    receiver_ontology_path=ontologies["receiver_ontology_32"],
                )

            validate()
            candidates = [dict(row) for row in fixture["normalized"]]
            candidates[0]["source_physical_audit_status"] = (
                "legacy_original_source_exempt"
                if candidates[0]["source_physical_audit_status"] == "strict_physical_pass_v2"
                else "strict_physical_pass_v2"
            )
            with self.assertRaisesRegex(ValueError, "physical-audit|physical status|identity"):
                validate(candidates)

            bank_path = fixture["root"] / protocol.PUBLIC_SOURCE_BANK
            bank_bytes = bank_path.read_bytes()
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            bank["entries"][8]["physical_audit_status"] = "legacy_original_source_exempt"
            bank["bank_entries_sha256"] = hashlib.sha256(
                (json.dumps(bank["entries"], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            ).hexdigest()
            bank_path.write_text(json.dumps(bank) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "physical status"):
                validate()
            bank_path.write_bytes(bank_bytes)

            source_path = ontologies["source_ontology_80"]
            holdout_path = ontologies["holdout_registry_24"]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            source["sources"][56]["impact_plausibility"]["mass_g"] = 1200
            source["sources"][56]["impact_plausibility"]["density_g_cm3"] = 3.0
            source["sources"][56]["impact_plausibility"]["dimensions_cm"] = [8.0, 3.0, 2.5]
            holdout["entries"][0]["impact_plausibility"] = source["sources"][56][
                "impact_plausibility"
            ]
            source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            holdout_path.write_text(json.dumps(holdout) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "impact-plausibility"):
                validate()


class ScreeningFreezeTests(unittest.TestCase):
    @staticmethod
    def _decode(_path: Path) -> dict[str, int]:
        return {
            "frame_count": 49,
            "width": 832,
            "height": 480,
            "fps_numerator": 8,
            "fps_denominator": 1,
        }

    @staticmethod
    def _composite(path: Path, video: Path) -> None:
        path.write_bytes(b"composite:" + video.name.encode("ascii"))

    def _inputs(self):
        candidates = make_causal_screening_candidates()
        fields = tuple(selector.CAUSAL_SCREENING_FIELDS.values())
        reviewer_a = [
            {**row, **{field: "2" for field in fields}, "notes": ""}
            for row in candidates
        ]
        reviewer_b = [dict(row) for row in reviewer_a]
        reviewer_b[0][fields[0]] = "0"
        return candidates, reviewer_a, reviewer_b

    def _public_inputs(self, dataset: str = "causal"):
        fields = tuple(selector._screening_fields(dataset).values())
        row_count = protocol.CANDIDATE_COUNTS[dataset]
        template = [
            {
                "review_id": f"s{index:03d}",
                "object_phrase": f"anonymous object {index}",
                "receiver_description": f"anonymous receiver {index}",
                "video_path": f"/public/media/s{index:03d}.mp4",
                "composite_path": f"/public/composites/s{index:03d}.jpg",
                **{field: "" for field in fields},
                "notes": "",
            }
            for index in range(row_count)
        ]
        reviewer_a = [
            {**row, **{field: "2" for field in fields}} for row in template
        ]
        reviewer_b = [dict(row) for row in reviewer_a]
        reviewer_b[0][fields[0]] = "0"
        reviewer_b[1][fields[1]] = "1"
        return template, reviewer_a, reviewer_b

    def test_public_dispute_derivation_is_exact_atomic_and_exclusive(self) -> None:
        template, reviewer_a, reviewer_b = self._public_inputs()
        disputes = selector.derive_public_screening_disputes(
            "causal", template, reviewer_a, reviewer_b
        )
        self.assertEqual(
            disputes,
            [
                {"review_id": "s000", "field": "source"},
                {"review_id": "s001", "field": "footprint"},
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            template_path = root / "template.csv"
            reviewer_a_path = root / "review_a.csv"
            reviewer_b_path = root / "review_b.csv"
            output = root / "disputes.csv"
            protocol.write_csv(template_path, template)
            protocol.write_csv(reviewer_a_path, reviewer_a)
            protocol.write_csv(reviewer_b_path, reviewer_b)
            args = argparse.Namespace(
                dataset="causal",
                public_root=root,
                template=template_path,
                reviewer_a=reviewer_a_path,
                reviewer_b=reviewer_b_path,
                output=output,
            )
            with mock.patch("builtins.print") as emit:
                self.assertEqual(selector._cmd_derive_disputes(args), 0)
            status = json.loads(emit.call_args.args[0])
            self.assertEqual(status["status"], "disputes_derived")
            self.assertEqual(status["dispute_count"], 2)
            self.assertEqual(status["sha256"], protocol.file_sha256(output))
            self.assertEqual(protocol.read_csv(output), disputes)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                selector._cmd_derive_disputes(args)

    def test_public_dispute_derivation_rejects_mutation_and_handles_no_disputes(self) -> None:
        template, reviewer_a, reviewer_b = self._public_inputs()
        changed = [dict(row) for row in reviewer_a]
        changed[0]["video_path"] = "/changed.mp4"
        with self.assertRaisesRegex(ValueError, "changed blinded public metadata"):
            selector.derive_public_screening_disputes(
                "causal", template, changed, reviewer_b
            )
        changed = [dict(row) for row in reviewer_a]
        changed[0][selector.CAUSAL_SCREENING_FIELDS["source"]] = "3"
        with self.assertRaisesRegex(ValueError, "must be 0, 1, or 2"):
            selector.derive_public_screening_disputes(
                "causal", template, changed, reviewer_b
            )
        identical = [dict(row) for row in reviewer_a]
        self.assertEqual(
            selector.derive_public_screening_disputes(
                "causal", template, reviewer_a, identical
            ),
            [],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "disputes.csv"
            selector._atomic_write_new_csv(
                output, [], fieldnames=("review_id", "field")
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "review_id,field\n")

    def test_public_dispute_derivation_enforces_specificity_inventory_and_ids(self) -> None:
        template, reviewer_a, reviewer_b = self._public_inputs("specificity")
        fields = tuple(selector.SPECIFICITY_SCREENING_FIELDS.values())
        self.assertEqual(len(template), 36)
        self.assertEqual(
            selector.derive_public_screening_disputes(
                "specificity", template, reviewer_a, reviewer_b
            ),
            [
                {"review_id": "s000", "field": "protected"},
                {"review_id": "s001", "field": "receiver"},
            ],
        )
        with self.assertRaisesRegex(ValueError, "each contain 36 rows"):
            selector.derive_public_screening_disputes(
                "specificity", template[:-1], reviewer_a, reviewer_b
            )
        duplicate = [dict(row) for row in reviewer_a]
        duplicate[-1]["review_id"] = "s000"
        with self.assertRaisesRegex(ValueError, "duplicate or differ"):
            selector.derive_public_screening_disputes(
                "specificity", template, duplicate, reviewer_b
            )
        invalid = [dict(row) for row in reviewer_a]
        invalid[0][fields[0]] = ""
        with self.assertRaisesRegex(ValueError, "invalid"):
            selector.derive_public_screening_disputes(
                "specificity", template, invalid, reviewer_b
            )

    def test_public_dispute_cli_rejects_duplicate_or_reordered_raw_header(self) -> None:
        template, reviewer_a, reviewer_b = self._public_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            template_path = root / "template.csv"
            reviewer_b_path = root / "review_b.csv"
            malformed_path = root / "review_a_malformed.csv"
            output = root / "disputes.csv"
            protocol.write_csv(template_path, template)
            protocol.write_csv(reviewer_b_path, reviewer_b)
            header = list(selector._public_screening_columns("causal"))

            def write_raw(columns: list[str]) -> None:
                with malformed_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(columns)
                    for row in reviewer_a:
                        writer.writerow([row[column] for column in columns])

            duplicate_header = [*header, header[-2]]
            write_raw(duplicate_header)
            args = argparse.Namespace(
                dataset="causal",
                public_root=root,
                template=template_path,
                reviewer_a=malformed_path,
                reviewer_b=reviewer_b_path,
                output=output,
            )
            with self.assertRaisesRegex(ValueError, "header is not exact"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

            reordered_header = list(header)
            reordered_header[1], reordered_header[2] = (
                reordered_header[2],
                reordered_header[1],
            )
            write_raw(reordered_header)
            with self.assertRaisesRegex(ValueError, "header is not exact"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

    def test_public_dispute_cli_rejects_nonpublic_or_symlink_paths_without_output(self) -> None:
        template, reviewer_a, reviewer_b = self._public_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            public_root = root / "public"
            public_root.mkdir()
            reviews = root / "review_results"
            reviews.mkdir()
            template_path = public_root / "template.csv"
            reviewer_a_path = public_root / "review_a.csv"
            reviewer_b_path = public_root / "review_b.csv"
            for path, rows in (
                (template_path, template),
                (reviewer_a_path, reviewer_a),
                (reviewer_b_path, reviewer_b),
            ):
                protocol.write_csv(path, rows)
            output = reviews / "disputes.csv"
            args = argparse.Namespace(
                dataset="causal",
                public_root=public_root,
                template=template_path,
                reviewer_a=reviewer_a_path,
                reviewer_b=reviewer_b_path,
                output=output,
            )

            outside = root / "outside.csv"
            protocol.write_csv(outside, reviewer_a)
            args.reviewer_a = outside
            with self.assertRaisesRegex(ValueError, "contained by --public-root"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

            leaf_link = public_root / "review_a_link.csv"
            leaf_link.symlink_to(reviewer_a_path.name)
            args.reviewer_a = leaf_link
            with self.assertRaisesRegex(ValueError, "no symlink component"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

            nested = public_root / "nested"
            nested.mkdir()
            nested_review = nested / "review_a.csv"
            protocol.write_csv(nested_review, reviewer_a)
            directory_link = public_root / "nested_link"
            directory_link.symlink_to(nested.name, target_is_directory=True)
            args.reviewer_a = directory_link / nested_review.name
            with self.assertRaisesRegex(ValueError, "no symlink component"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

            args.reviewer_a = reviewer_a_path
            linked_reviews = root / "linked_review_results"
            linked_reviews.symlink_to(reviews.name, target_is_directory=True)
            args.output = linked_reviews / output.name
            with self.assertRaisesRegex(ValueError, "no symlink ancestor"):
                selector._cmd_derive_disputes(args)
            self.assertFalse(output.exists())

            output_link = reviews / "disputes_link.csv"
            output_target = reviews / "must_not_be_created.csv"
            output_link.symlink_to(output_target.name)
            args.output = output_link
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                selector._cmd_derive_disputes(args)
            self.assertTrue(output_link.is_symlink())
            self.assertFalse(output_target.exists())

            args.output = root / "missing_review_results" / output.name
            with self.assertRaises(FileNotFoundError):
                selector._cmd_derive_disputes(args)
            self.assertFalse(args.output.parent.exists())

            args.output = output
            with mock.patch("builtins.print"):
                self.assertEqual(selector._cmd_derive_disputes(args), 0)
            self.assertEqual(
                protocol.read_csv(output),
                selector.derive_public_screening_disputes(
                    "causal", template, reviewer_a, reviewer_b
                ),
            )

    def test_public_dispute_atomic_publish_does_not_touch_stale_temp_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "disputes.csv"
            stale = root / f".{output.name}.tmp.{os.getpid()}"
            target = root / "must_not_be_created.csv"
            stale.symlink_to(target.name)
            with self.assertRaises(FileExistsError):
                selector._atomic_write_new_csv(
                    output, [], fieldnames=("review_id", "field")
                )
            self.assertTrue(stale.is_symlink())
            self.assertFalse(output.exists())
            self.assertFalse(target.exists())

    def test_freeze_missing_public_inputs_fails_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private_root = Path(directory)
            frozen_dir = private_root / "frozen"
            arguments = {
                "project_root": private_root,
                "dataset": "causal",
                "package_manifest_path": private_root / "package.json",
                "private_root": private_root,
                "candidate_manifest_path": private_root / "candidates.json",
                "canonical_templates_path": private_root / "templates.json",
                "screening_seed_path": private_root / "seed.txt",
                "generation_spec_path": private_root / "generation.json",
                "reviewer_a_path": private_root / "review_a.csv",
                "reviewer_b_path": private_root / "review_b.csv",
                "dispute_path": private_root / "disputes.csv",
                "adjudication_path": private_root / "adjudication.csv",
                "canonical_path": frozen_dir / "eligibility.csv",
                "audit_path": frozen_dir / "audit.csv",
                "freeze_manifest_path": private_root / "freeze.json",
            }
            before = sorted(path.relative_to(private_root) for path in private_root.rglob("*"))
            with self.assertRaisesRegex(FileNotFoundError, "derive-disputes"):
                selector.freeze_screening_reviews(**arguments)
            after = sorted(path.relative_to(private_root) for path in private_root.rglob("*"))
            self.assertEqual(after, before)

            protocol.write_csv(
                arguments["dispute_path"], [], fieldnames=("review_id", "field")
            )
            before = sorted(path.relative_to(private_root) for path in private_root.rglob("*"))
            with self.assertRaisesRegex(FileNotFoundError, "adjudication"):
                selector.freeze_screening_reviews(**arguments)
            after = sorted(path.relative_to(private_root) for path in private_root.rglob("*"))
            self.assertEqual(after, before)

    def test_screening_merge_rejects_metadata_dispute_and_adjudication_tamper(self) -> None:
        candidates, reviewer_a, reviewer_b = self._inputs()
        disputes = selector.derive_screening_disputes(
            "causal", candidates, reviewer_a, reviewer_b
        )
        self.assertEqual(disputes, [{"candidate_id": "candidate_0", "field": "source"}])
        with self.assertRaisesRegex(ValueError, "exact atomic"):
            selector.merge_screening_reviews(
                "causal", candidates, reviewer_a, reviewer_b, [], []
            )
        with self.assertRaisesRegex(ValueError, "requires blinded adjudication"):
            selector.merge_screening_reviews(
                "causal", candidates, reviewer_a, reviewer_b, disputes, []
            )
        changed = [dict(row) for row in reviewer_a]
        changed[0]["receiver"] = "changed after freeze"
        with self.assertRaisesRegex(ValueError, "changed metadata"):
            selector.merge_screening_reviews(
                "causal", candidates, changed, reviewer_b, disputes, []
            )
        canonical, audit = selector.merge_screening_reviews(
            "causal",
            candidates,
            reviewer_a,
            reviewer_b,
            disputes,
            [
                {
                    "candidate_id": "candidate_0",
                    "field": "source",
                    "score": "1",
                    "brief_reason": "full-video blind review",
                }
            ],
        )
        self.assertEqual(canonical[0]["source_visibility_0_absent_2_clear"], 1)
        self.assertEqual(canonical[0]["eligible"], "no")
        self.assertEqual(len(audit), 1)

    def test_selector_validates_hash_bound_screening_freeze(self) -> None:
        candidates, reviewer_a, reviewer_b = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.csv"
            a_path = root / "a.csv"
            b_path = root / "b.csv"
            dispute_path = root / "disputes.csv"
            adjudication_path = root / "adjudication.csv"
            protocol.write_csv(candidate_path, candidates)
            protocol.write_csv(a_path, reviewer_a)
            protocol.write_csv(b_path, reviewer_b)
            protocol.write_csv(
                dispute_path,
                [{"candidate_id": "candidate_0", "field": "source"}],
            )
            protocol.write_csv(
                adjudication_path,
                [
                    {
                        "candidate_id": "candidate_0",
                        "field": "source",
                        "score": 1,
                        "brief_reason": "blind",
                    }
                ],
            )
            canonical = root / "frozen" / "eligibility.csv"
            audit = root / "frozen" / "audit.csv"
            freeze = root / "screening_freeze.json"
            selector._freeze_unbound_screening_rows(
                dataset="causal",
                candidate_path=candidate_path,
                reviewer_a_path=a_path,
                reviewer_b_path=b_path,
                dispute_path=dispute_path,
                adjudication_path=adjudication_path,
                canonical_path=canonical,
                audit_path=audit,
                freeze_manifest_path=freeze,
            )
            _, rows = selector._validate_unbound_screening_freeze(
                root, freeze, dataset="causal", private_root=root
            )
            self.assertEqual(len(rows), 48)
            canonical.write_text(canonical.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
                selector._validate_unbound_screening_freeze(
                    root, freeze, dataset="causal", private_root=root
                )

    def test_secure_original_package_freeze_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = make_synthetic_stage0_fixture(Path(directory))
            generation_manifest = make_synthetic_screening_generation(fixture)
            components = fixture["components"]
            normalized = fixture["normalized"]
            private_root = fixture["private"]
            assert isinstance(components, dict)
            assert isinstance(normalized, list)
            assert isinstance(private_root, Path)
            package_parent = private_root / "screening_package"
            public_dir = package_parent / "public"
            private_dir = package_parent / "private"
            selector.build_screening_review_package(
                project_root=fixture["root"],
                dataset="causal",
                normalized_candidates=normalized,
                candidate_manifest_path=components[
                    "causal_stage0_candidates_private_v2.json"
                ],
                screening_seed_path=fixture["screening_seed"],
                generation_spec_path=fixture["generation_spec"],
                stage0_registry_path=fixture["stage0"],
                generation_manifest_path=generation_manifest,
                public_dir=public_dir,
                private_dir=private_dir,
                composite_builder=self._composite,
                decode=self._decode,
                verify_model_bytes=False,
            )
            package_manifest = private_dir / "screening_package_manifest_v2.json"
            selector.validate_screening_review_package(
                fixture["root"],
                dataset="causal",
                manifest_path=package_manifest,
                private_root=private_root,
                candidate_manifest_path=components[
                    "causal_stage0_candidates_private_v2.json"
                ],
                canonical_templates_path=components[
                    "causal_stage0_templates_private_v2.json"
                ],
                screening_seed_path=fixture["screening_seed"],
                generation_spec_path=fixture["generation_spec"],
                decode=self._decode,
                verify_model_bytes=False,
            )

            projection_path = private_dir / "screening_candidates_v2.csv"
            projection_bytes = projection_path.read_bytes()
            manifest_bytes = package_manifest.read_bytes()
            projection = protocol.read_csv(projection_path)
            projection[0]["semantic_case_id"] = "tampered-derived-id"
            protocol.write_csv(projection_path, projection)
            package = json.loads(package_manifest.read_text(encoding="utf-8"))
            package["candidate_projection"]["sha256"] = protocol.file_sha256(
                projection_path
            )
            package_manifest.write_text(json.dumps(package) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "projection differs"):
                selector.validate_screening_review_package(
                    fixture["root"],
                    dataset="causal",
                    manifest_path=package_manifest,
                    private_root=private_root,
                    candidate_manifest_path=components[
                        "causal_stage0_candidates_private_v2.json"
                    ],
                    canonical_templates_path=components[
                        "causal_stage0_templates_private_v2.json"
                    ],
                    screening_seed_path=fixture["screening_seed"],
                    generation_spec_path=fixture["generation_spec"],
                    decode=self._decode,
                    verify_model_bytes=False,
                )
            projection_path.write_bytes(projection_bytes)
            package_manifest.write_bytes(manifest_bytes)

            template = protocol.read_csv(public_dir / "screening_review_v2.csv")
            fields = tuple(selector.CAUSAL_SCREENING_FIELDS.values())
            reviewer_a = [
                {**row, **{field: "2" for field in fields}} for row in template
            ]
            reviewer_b = [dict(row) for row in reviewer_a]
            reviewer_b[0][fields[0]] = "0"
            reviews = private_root / "screening_reviews"
            reviews.mkdir()
            a_path = reviews / "review_a.csv"
            b_path = reviews / "review_b.csv"
            dispute_path = reviews / "disputes.csv"
            adjudication_path = reviews / "adjudication.csv"
            protocol.write_csv(a_path, reviewer_a)
            protocol.write_csv(b_path, reviewer_b)
            protocol.write_csv(
                dispute_path,
                selector.derive_public_screening_disputes(
                    "causal", template, reviewer_a, reviewer_b
                ),
                fieldnames=("review_id", "field"),
            )
            protocol.write_csv(
                adjudication_path,
                [
                    {
                        "review_id": "s000",
                        "field": "source",
                        "score": 1,
                        "brief_reason": "full-video blinded adjudication",
                    }
                ],
            )
            frozen_dir = private_root / "screening_frozen"
            canonical_path = frozen_dir / "eligibility.csv"
            audit_path = frozen_dir / "audit.csv"
            freeze_path = reviews / "freeze.json"
            freeze_payload = selector.freeze_screening_reviews(
                project_root=fixture["root"],
                dataset="causal",
                package_manifest_path=package_manifest,
                private_root=private_root,
                candidate_manifest_path=components[
                    "causal_stage0_candidates_private_v2.json"
                ],
                canonical_templates_path=components[
                    "causal_stage0_templates_private_v2.json"
                ],
                screening_seed_path=fixture["screening_seed"],
                generation_spec_path=fixture["generation_spec"],
                reviewer_a_path=a_path,
                reviewer_b_path=b_path,
                dispute_path=dispute_path,
                adjudication_path=adjudication_path,
                canonical_path=canonical_path,
                audit_path=audit_path,
                freeze_manifest_path=freeze_path,
                decode=self._decode,
                verify_model_bytes=False,
            )
            selector.validate_screening_freeze(
                fixture["root"],
                freeze_path,
                dataset="causal",
                private_root=private_root,
                candidate_manifest_path=components[
                    "causal_stage0_candidates_private_v2.json"
                ],
                canonical_templates_path=components[
                    "causal_stage0_templates_private_v2.json"
                ],
                screening_seed_path=fixture["screening_seed"],
                generation_spec_path=fixture["generation_spec"],
                decode=self._decode,
                verify_model_bytes=False,
            )
            selector_output = private_root / "selector_output"
            selector_output.mkdir()
            protocol.write_csv(
                selector_output / "eligibility_v2.csv",
                [{"candidate_id": f"c{index:03d}"} for index in range(48)],
            )
            protocol.write_csv(
                selector_output / "selected_cases_v2.csv",
                [{"semantic_case_id": f"c{index:03d}"} for index in range(24)],
            )
            protocol.write_csv(
                selector_output / "unit_manifest_v2.csv",
                [{"unit_id": f"u{index:03d}"} for index in range(72)],
            )
            (selector_output / "selector_output_v2.json").write_text(
                json.dumps({"status": "synthetic-selected"}) + "\n",
                encoding="utf-8",
            )
            stage1_path = Path(fixture["root"]) / protocol.CAUSAL_STAGE1
            selector._publish_stage1_registry(
                Path(fixture["root"]),
                dataset="causal",
                stage0_registry_path=Path(fixture["stage0"]),
                freeze_manifest_path=freeze_path,
                freeze_payload=freeze_payload,
                output_dir=selector_output,
                stage1_output=stage1_path,
            )
            protocol.validate_commitment_registry(
                stage1_path,
                dataset="causal",
                stage=1,
                expected_stage0_sha256=protocol.file_sha256(Path(fixture["stage0"])),
            )
            anonymous = public_dir / "media/s000.mp4"
            anonymous.write_bytes(b"wrong-video")
            with self.assertRaisesRegex(ValueError, "byte|media"):
                selector.validate_screening_freeze(
                    fixture["root"],
                    freeze_path,
                    dataset="causal",
                    private_root=private_root,
                    candidate_manifest_path=components[
                        "causal_stage0_candidates_private_v2.json"
                    ],
                    canonical_templates_path=components[
                        "causal_stage0_templates_private_v2.json"
                    ],
                    screening_seed_path=fixture["screening_seed"],
                    generation_spec_path=fixture["generation_spec"],
                    decode=self._decode,
                    verify_model_bytes=False,
                )


class SelectorAndAssignmentTests(unittest.TestCase):
    def test_seed_formula_is_domain_separated_uint32_and_id_sensitive(self) -> None:
        expected = int.from_bytes(
            hashlib.sha256(
                b"causal-eval-seed-v2\0seed-salt\0semantic-case-7\0" + b"2"
            ).digest()[:4],
            "big",
        )
        actual = protocol.derive_seed(
            "seed-salt", "semantic-case-7", 2, dataset="causal"
        )
        self.assertEqual(actual, expected)
        self.assertLess(actual, 1 << 32)
        self.assertNotEqual(
            actual,
            protocol.derive_seed(
                "seed-salt", "semantic-case-8", 2, dataset="causal"
            ),
        )
        wrong_domain = int.from_bytes(
            hashlib.sha256(
                b"specificity-eval-seed-v2\0seed-salt\0semantic-case-7\0" + b"2"
            ).digest()[:4],
            "big",
        )
        self.assertNotEqual(actual, wrong_domain)

    def test_selection_rank_ignores_screening_scores_but_not_scientific_record(self) -> None:
        row = make_causal_screening_candidates()[0]
        scored = {
            **row,
            "source_visibility_0_absent_2_clear": 2,
            "footprint_visibility_0_absent_2_clear": 1,
            "receiver_preservation_0_bad_2_good": 2,
            "video_quality_0_bad_2_good": 2,
            "causal_link_0_absent_2_clear": 2,
        }
        changed_score = {**scored, "source_visibility_0_absent_2_clear": 0}
        self.assertEqual(
            selector.selection_rank(scored, "salt"),
            selector.selection_rank(changed_score, "salt"),
        )
        changed_alias = {**scored, "prompt": "untrusted normalized alias"}
        self.assertEqual(
            selector.selection_rank(scored, "salt"),
            selector.selection_rank(changed_alias, "salt"),
        )
        changed_prompt = {**scored, "canonical_prompt": "different registered prompt"}
        changed_record = {
            key: changed_prompt[key]
            for key in selector.CAUSAL_CANONICAL_RECORD_FIELDS
        }
        changed_prompt["canonical_record_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    changed_record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        self.assertNotEqual(
            selector.selection_rank(scored, "salt"),
            selector.selection_rank(changed_prompt, "salt"),
        )
        canonical_record = {
            key: row[key] for key in selector.CAUSAL_CANONICAL_RECORD_FIELDS
        }
        canonical = (
            json.dumps(
                canonical_record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        expected = hashlib.sha256(
            b"causal-selector-v2\0salt\0" + canonical
        ).hexdigest()
        wrong_domain = hashlib.sha256(
            b"specificity-selector-v1\0salt\0" + canonical
        ).hexdigest()
        self.assertEqual(selector.selection_rank(row, "salt"), expected)
        self.assertNotEqual(selector.selection_rank(row, "salt"), wrong_domain)

    def test_causal_selector_and_blocked_assignment(self) -> None:
        candidates = make_causal_screening_candidates()
        for row in candidates:
            row.update(
                {
                    "source_visibility_0_absent_2_clear": 2,
                    "footprint_visibility_0_absent_2_clear": 1,
                    "receiver_preservation_0_bad_2_good": 2,
                    "video_quality_0_bad_2_good": 2,
                    "causal_link_0_absent_2_clear": 2,
                }
            )
        selected, eligibility = selector.select_causal_cases(candidates, "selector-salt")
        self.assertEqual(len(selected), 24)
        self.assertEqual(sum(row["eligible"] == "yes" for row in eligibility), 48)
        protocol.validate_causal_selected_cases(selected)
        units = protocol.derive_unit_rows(selected, dataset="causal", private_salt="seed-salt")
        assignment = builder.derive_causal_ab_assignment(units, "blind-salt")
        builder.validate_blocked_assignment(units, "causal", assignment)

    def test_specificity_assignment_swaps_every_case(self) -> None:
        causal = make_causal_units()
        units, _ = make_specificity_units(causal)
        assignment = builder.derive_specificity_ab_assignment(units, "blind-salt")
        builder.validate_blocked_assignment(units, "specificity", assignment)


class EvalRunnerTests(unittest.TestCase):
    def test_generation_command_is_fixed_and_has_no_skip_resume(self) -> None:
        command = runner._generation_command(
            python="python",
            prompt_path=Path("prompts.txt"),
            output_dir=Path("output"),
            seeds=[11, 12],
            checkpoint=None,
        )
        rendered = " ".join(command)
        for token in (
            "--steps 25",
            "--guidance-scale 5",
            "--num-frames 49",
            "--fps 8",
            "--height 480",
            "--width 832",
            "--dtype bf16",
            "--device cuda",
            "--vae-slicing",
            "--vae-tiling",
        ):
            self.assertIn(token, rendered)
        self.assertNotIn("--skip-existing", command)
        source = (PROJECT_ROOT / "scripts/run_water_impact_dynamic_v4_eval.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('add_parser("final', source)

    def test_raw_generation_config_is_exact_and_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / ".run_reservation_v2.json").write_text("{}\n", encoding="utf-8")
            (output / "prompts.txt").write_text(
                "prompt | object | registered v4 evaluation\n", encoding="utf-8"
            )
            video_dir = output / "videos"
            video_dir.mkdir()
            video = video_dir / "000_prompt_seed7.mp4"
            video.write_bytes(b"video")
            rows = [{"prompt": "prompt", "source_phrase": "object", "seed": 7}]
            config = {
                "baseline": "clean",
                "seed": 42,
                "seeds": [7],
                "num_inference_steps": 25,
                "guidance_scale": 5.0,
                "num_frames": 49,
                "fps": 8,
                "height": 480,
                "width": 832,
                "dtype": "bf16",
                "device": "cuda",
                "enable_model_cpu_offload": False,
                "enable_sequential_cpu_offload": False,
                "vae_slicing": True,
                "vae_tiling": True,
                "prompt_encode_device_policy": "cpu_when_offloaded_else_selected_device",
                "lora_path": None,
                "lora_sha256": None,
                "lora_scale": 1.0,
                "activation_gate_dir": None,
                "persistent_activation_gate": False,
                "lora_target_phrases": [],
                "attention_gate_dir": None,
                "attention_suppression_phrases": [],
                "attention_suppression_strength": 20.0,
            }
            raw = {
                "created_at_utc": "ignored",
                "baseline": "clean",
                "pipeline": "WanPipeline",
                "model": runner.MODEL,
                "dry_run": False,
                "prompts": str(output / "prompts.txt"),
                "generation": config,
                "items": [
                    {
                        "index": 0,
                        "prompt": "prompt",
                        "target_concept": "object",
                        "expected_effect": "registered v4 evaluation",
                        "seed": 7,
                        "video_path": str(video),
                    }
                ],
            }
            manifest = output / "generation_manifest.json"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            _, videos = runner._validate_raw_generation(
                output, rows, [7], checkpoint=None
            )
            self.assertEqual(videos, [video])
            raw["generation"]["vae_tiling"] = False
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                runner._validate_raw_generation(output, rows, [7], checkpoint=None)

    def test_generation_spec_and_model_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "generation_spec.json"
            payload = {
                "protocol": runner.GENERATION_SPEC_PROTOCOL,
                "status": "frozen_before_original_render",
                "model_inventory_sha256": protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256,
                "runtime_registry": {
                    "path": protocol.RUNTIME_REGISTRY,
                    "sha256": "b" * 64,
                },
                "generation_spec": protocol.GENERATION_SPEC,
                "source_mode": "Original_screening_then_matched_O_v3b_v4",
            }
            spec.write_text(json.dumps(payload), encoding="utf-8")
            runner._validate_generation_spec(
                spec,
                committed_sha256=protocol.file_sha256(spec),
                private_root=root,
            )
            payload["generation_spec"] = {
                **protocol.GENERATION_SPEC,
                "guidance_scale": 7.0,
            }
            spec.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable contract"):
                runner._validate_generation_spec(
                    spec,
                    committed_sha256=protocol.file_sha256(spec),
                    private_root=root,
                )
            model = root / "model"
            model.mkdir()
            for relative in (
                "model_index.json",
                "transformer/config.json",
                "text_encoder/config.json",
                "tokenizer/tokenizer_config.json",
            ):
                path = model / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (model / "weights.bin").write_bytes(b"model-v1")
            with mock.patch.object(runner, "MODEL", str(model)):
                actual = protocol.model_artifact_inventory(root, model)["sha256"]
                runner._model_inventory(root, actual)
                (model / "weights.bin").write_bytes(b"model-v2")
                with self.assertRaisesRegex(ValueError, "model bytes differ"):
                    runner._model_inventory(root, actual)

    def test_generation_runtime_rejects_wrong_python_and_registry_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_path = root / protocol.RUNTIME_REGISTRY
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text(
                json.dumps(protocol.RUNTIME_REGISTRY_PAYLOAD) + "\n",
                encoding="utf-8",
            )
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copyfile(
                PROJECT_ROOT
                / protocol.TRAINING_CODE_ARTIFACTS["runtime_registry_builder"],
                root / protocol.TRAINING_CODE_ARTIFACTS["runtime_registry_builder"],
            )
            runtime_sha = protocol.file_sha256(runtime_path)
            generation_spec = {
                "runtime_registry": {
                    "path": protocol.RUNTIME_REGISTRY,
                    "sha256": runtime_sha,
                }
            }
            calls: list[tuple[object, object]] = []

            def fake_run(command: object, **kwargs: object) -> None:
                calls.append((command, kwargs))

            runner._validate_generation_runtime(
                root,
                generation_spec,
                str(protocol.RUNTIME_REGISTRY_PAYLOAD["python_executable"]),
                run=fake_run,
            )
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "interpreter differs"):
                runner._validate_generation_runtime(
                    root, generation_spec, sys.executable, run=fake_run
                )
            runtime_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
                runner._validate_generation_runtime(
                    root,
                    generation_spec,
                    str(protocol.RUNTIME_REGISTRY_PAYLOAD["python_executable"]),
                    run=fake_run,
                )

    def test_output_reservation_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            runner._reserve_output(output, {"status": "reserved"})
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                runner._reserve_output(output, {"status": "second"})


class BlindReviewTests(unittest.TestCase):
    @staticmethod
    def fake_composite(path: Path, videos):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("|".join(sorted(videos)).encode())

    def test_public_package_has_O_A_B_and_no_private_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            units = make_causal_units()
            unit_manifest = root / "unit_manifest_v2.csv"
            protocol.write_csv(unit_manifest, units)
            videos = {method: {} for method in protocol.METHODS}
            manifests = {}
            model_sha = protocol.FROZEN_MODEL_CONTENT_INVENTORY_SHA256
            runtime_sha = "b" * 64
            code_registry = root / protocol.TRAINING_CODE_REGISTRY
            code_registry.parent.mkdir(parents=True)
            code_registry.write_text("synthetic frozen code registry\n", encoding="utf-8")
            code_sha = protocol.file_sha256(code_registry)
            for method in protocol.METHODS:
                manifest = root / method / "v4_generation_manifest_v2.json"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    json.dumps(
                        {
                            "dataset": "causal",
                            "method": method,
                            "model_inventory_sha256": model_sha,
                            "runtime_registry_sha256": runtime_sha,
                        }
                    ),
                    encoding="utf-8",
                )
                manifests[method] = manifest
                for unit in units:
                    path = root / "source" / method / f"{unit['unit_id']}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{method}:{unit['unit_id']}".encode())
                    videos[method][str(unit["unit_id"])] = path
            eligibility = root / "eligibility.json"
            eligibility.write_text(
                json.dumps(
                    {
                        "model_content_inventory_sha256": model_sha,
                        "runtime_registry_sha256": runtime_sha,
                        "training_code_registry_sha256": code_sha,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            public = root / "review_public"
            private = root / "review_private"
            eligibility_payload = json.loads(eligibility.read_text(encoding="utf-8"))
            with (
                mock.patch.object(
                    protocol, "validate_checkpoint_eligibility", return_value=eligibility_payload
                ),
                mock.patch.object(protocol, "validate_training_code_registry"),
                mock.patch.object(protocol, "validate_runtime_registry"),
            ):
                builder.build_review_package(
                    project_root=root,
                    dataset="causal",
                    unit_rows=units,
                    unit_manifest_path=unit_manifest,
                    videos=videos,
                    generation_manifest_paths=manifests,
                    checkpoint_eligibility_path=eligibility,
                    public_dir=public,
                    private_dir=private,
                    package_commitment_path=root / "review_package_commitment_v2.json",
                    private_salt="blind-salt",
                    composite_builder=self.fake_composite,
                )
            review = read_csv(public / "blind_review_v2.csv")
            key = read_csv(private / "answer_key_v2.csv")
            self.assertEqual(len(review), 216)
            self.assertEqual(len(key), 216)
            self.assertEqual({row["arm_code"] for row in review}, {"O", "A", "B"})
            self.assertFalse(set(review[0]) & protocol.FORBIDDEN_PUBLIC_FIELDS)
            self.assertEqual({row["method"] for row in key}, set(protocol.METHODS))
            self.assertEqual({path.name for path in public.iterdir()}, {"blind_review_v2.csv", "composites", "media"})
            self.assertEqual({path.name for path in private.iterdir()}, {"answer_key_v2.csv", "review_manifest_v2.json"})

            commitment = root / "review_package_commitment_v2.json"
            with (
                mock.patch.object(
                    protocol, "validate_checkpoint_eligibility", return_value=eligibility_payload
                ),
                mock.patch.object(protocol, "validate_training_code_registry"),
                mock.patch.object(protocol, "validate_runtime_registry"),
            ):
                builder.validate_review_package_commitment(
                    root,
                    dataset="causal",
                    commitment_path=commitment,
                    template_path=public / "blind_review_v2.csv",
                    answer_key_path=private / "answer_key_v2.csv",
                    review_manifest_path=private / "review_manifest_v2.json",
                    unit_manifest_path=unit_manifest,
                    checkpoint_eligibility_path=eligibility,
                )
                with self.assertRaisesRegex(ValueError, "opened assignment salt"):
                    builder.validate_review_package(
                        root,
                        dataset="causal",
                        unit_rows=units,
                        template_path=public / "blind_review_v2.csv",
                        answer_key_path=private / "answer_key_v2.csv",
                        review_manifest_path=private / "review_manifest_v2.json",
                        package_commitment_path=commitment,
                        assignment_salt="wrong-salt",
                        unit_manifest_path=unit_manifest,
                        checkpoint_eligibility_path=eligibility,
                        decode=lambda _path: {
                            "frame_count": 49, "width": 832, "height": 480,
                            "fps_numerator": 8, "fps_denominator": 1,
                        },
                    )

                reviewer_a = root / "reviewer_a.csv"
                reviewer_b = root / "reviewer_b.csv"
                scored = [dict(row) for row in review]
                for row in scored:
                    for field in protocol.CAUSAL_SCORE_FIELDS:
                        row[field] = "2"
                    row[protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]] = (
                        "2" if row["arm_code"] == "O" else ""
                    )
                protocol.write_csv(reviewer_a, scored)
                protocol.write_csv(reviewer_b, scored)
                adjudication = root / "adjudication.csv"
                protocol.write_csv(
                    adjudication,
                    [],
                    fieldnames=("review_id", "field", "score", "brief_reason"),
                )
                freeze = root / "review_freeze_v2.json"
                scorer.freeze_review_artifacts(
                    project_root=root,
                    dataset="causal",
                    package_commitment_path=commitment,
                    template_path=public / "blind_review_v2.csv",
                    reviewer_a_path=reviewer_a,
                    reviewer_b_path=reviewer_b,
                    dispute_path=root / "disputes.csv",
                    adjudication_path=adjudication,
                    canonical_path=root / "frozen" / "canonical.csv",
                    audit_path=root / "frozen" / "audit.csv",
                    freeze_manifest_path=freeze,
                )
                frozen_payload = json.loads(freeze.read_text(encoding="utf-8"))
                self.assertEqual(
                    frozen_payload["review_package_commitment"]["sha256"],
                    protocol.file_sha256(commitment),
                )

            extra = dict(review[0])
            extra["semantic_case_id"] = "leak"
            with self.assertRaisesRegex(ValueError, "exact registered schema"):
                protocol.validate_public_review_columns([extra])

            frozen_hash = protocol.file_sha256(root / "review_freeze_v2.json")
            for method in ("v3b", "v4"):
                payload = json.loads(manifests[method].read_text(encoding="utf-8"))
                payload["attack_rebind"] = True
                manifests[method].write_text(json.dumps(payload) + "\n", encoding="utf-8")
            key_rows = protocol.read_csv(private / "answer_key_v2.csv")
            key_rows[0]["method"] = "v4" if key_rows[0]["method"] != "v4" else "v3b"
            protocol.write_csv(private / "answer_key_v2.csv", key_rows)
            review_manifest = json.loads(
                (private / "review_manifest_v2.json").read_text(encoding="utf-8")
            )
            review_manifest["assignment_sha256"] = "0" * 64
            (private / "review_manifest_v2.json").write_text(
                json.dumps(review_manifest) + "\n", encoding="utf-8"
            )
            next(iter((public / "media").iterdir())).write_bytes(b"rebound-video")
            next(iter((public / "composites").iterdir())).write_bytes(b"rebound-composite")
            with (
                mock.patch.object(
                    protocol, "validate_checkpoint_eligibility", return_value=eligibility_payload
                ),
                mock.patch.object(protocol, "validate_training_code_registry"),
                mock.patch.object(protocol, "validate_runtime_registry"),
            ):
                with self.assertRaisesRegex(ValueError, "byte hash mismatch|bytes changed"):
                    scorer.validate_review_freeze(
                        root, root / "review_freeze_v2.json", dataset="causal"
                    )
            self.assertEqual(protocol.file_sha256(root / "review_freeze_v2.json"), frozen_hash)

    def test_merge_requires_every_atomic_dispute_and_keeps_candidate_link_blank(self) -> None:
        units = make_causal_units()
        template = []
        for position, unit in enumerate(units):
            for code in protocol.ARM_CODES:
                template.append(
                    {
                        "review_id": f"r{position:03d}_{code}",
                        "anonymous_unit": f"r{position:03d}",
                        "arm_code": code,
                        "object_phrase": str(unit["source_phrase"]),
                        "receiver_description": str(unit["receiver"]),
                        "composite_path": f"r{position:03d}.jpg",
                        "video_path": f"r{position:03d}_{code}.mp4",
                        **{field: "" for field in (*protocol.CAUSAL_SCORE_FIELDS, *protocol.CAUSAL_REFERENCE_ONLY_FIELDS)},
                        "notes": "",
                    }
                )
        reviewer_a = [dict(row) for row in template]
        reviewer_b = [dict(row) for row in template]
        for rows in (reviewer_a, reviewer_b):
            for row in rows:
                for field in protocol.CAUSAL_SCORE_FIELDS:
                    row[field] = "2"
                if row["arm_code"] == "O":
                    row[protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]] = "2"
        reviewer_b[0][protocol.CAUSAL_SCORE_FIELDS[0]] = "0"
        disputes = scorer.derive_dispute_template("causal", template, reviewer_a, reviewer_b)
        with self.assertRaisesRegex(ValueError, "requires blinded adjudication"):
            scorer.merge_blind_reviews("causal", template, reviewer_a, reviewer_b, disputes, [])
        canonical, audit = scorer.merge_blind_reviews(
            "causal",
            template,
            reviewer_a,
            reviewer_b,
            disputes,
            [{"review_id": disputes[0]["review_id"], "field": "target", "score": "1", "brief_reason": "blind"}],
        )
        self.assertEqual(canonical[0][protocol.CAUSAL_SCORE_FIELDS[0]], 1)
        candidate = next(row for row in canonical if row["arm_code"] == "A")
        self.assertEqual(candidate[protocol.CAUSAL_REFERENCE_ONLY_FIELDS[0]], "")
        self.assertEqual(len(audit), 1)


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.causal_units = make_causal_units()
        self.spec_units, self.mapping = make_specificity_units(self.causal_units)
        self.causal_scores = make_causal_scores(self.causal_units)
        self.spec_scores = make_specificity_scores(self.spec_units)

    def test_all_21_causal_11_specificity_and_role_gates_pass(self) -> None:
        causal = scorer.compute_causal_gate(self.causal_scores)
        specificity = scorer.compute_specificity_gate(self.spec_scores)
        role = scorer.compute_role_selectivity_gate(
            self.causal_scores,
            self.spec_scores,
            self.mapping,
            causal_gate=causal,
            specificity_gate=specificity,
        )
        self.assertEqual(len(causal["checks"]), 21)
        self.assertEqual(len(specificity["checks"]), 11)
        self.assertTrue(causal["passed"])
        self.assertTrue(specificity["passed"])
        self.assertTrue(role["passed"])
        outcome = scorer.classify_post_checkpoint_outcome(causal, specificity, role)
        self.assertEqual(outcome["outcome"], "eligible_for_separate_main_experiment_preregistration")
        self.assertTrue(outcome["promote_v4"])

    def test_unusable_v4_gets_zero_suppression_and_no_absence_credit(self) -> None:
        first_unit = str(self.causal_units[0]["unit_id"])
        row = next(
            row for row in self.causal_scores if row["unit_id"] == first_unit and row["method"] == "v4"
        )
        row[protocol.CAUSAL_SCORE_FIELDS[2]] = 0
        gate = scorer.compute_causal_gate(self.causal_scores)
        self.assertNotIn(first_unit, gate["paired_target_improvement_units"])
        self.assertNotIn(first_unit, gate["clear_to_absent_units"])
        self.assertNotIn(first_unit, gate["strict_v4_units"])
        self.assertEqual(gate["target_delta"]["C"], 142)

    def test_unusable_specificity_gets_zero_PV_NR_and_no_absence_credit(self) -> None:
        first_unit = str(self.spec_units[0]["unit_id"])
        row = next(
            row for row in self.spec_scores if row["unit_id"] == first_unit and row["method"] == "v4"
        )
        row[protocol.SPECIFICITY_SCORE_FIELDS[0]] = 0
        row[protocol.SPECIFICITY_SCORE_FIELDS[1]] = 0
        row[protocol.SPECIFICITY_SCORE_FIELDS[3]] = 0
        gate = scorer.compute_specificity_gate(self.spec_scores)
        self.assertEqual(gate["usable_absent_protected"]["v4_D"], 0)
        self.assertEqual(gate["PV"]["global_D"]["v4"], 70)
        self.assertEqual(gate["NR"]["global_D"]["v4"], 70)

    def test_outcome_classes_are_fail_closed(self) -> None:
        causal = scorer.compute_causal_gate(self.causal_scores, provenance_valid=False)
        specificity = scorer.compute_specificity_gate(self.spec_scores)
        role = scorer.compute_role_selectivity_gate(
            self.causal_scores, self.spec_scores, self.mapping,
            causal_gate=causal, specificity_gate=specificity,
        )
        self.assertEqual(
            scorer.classify_post_checkpoint_outcome(causal, specificity, role)["outcome"],
            "invalid_run",
        )
        self.assertEqual(
            scorer.classify_precheckpoint_outcome("registered_scale_sanity_termination")["outcome"],
            "registered_scale_sanity_termination",
        )
        with self.assertRaisesRegex(ValueError, "unregistered"):
            scorer.classify_precheckpoint_outcome("retry_with_new_bank")

    def test_scorer_exception_records_nonpromotable_invalid_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "score-output"
            args = argparse.Namespace(
                output_dir=output,
                training_authorization=root / "wrong-authorization.json",
                checkpoint_eligibility=root / "wrong-checkpoint.json",
                private_root=root,
            )
            with mock.patch.object(scorer.Path, "cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "training-authorization path"):
                    scorer._cmd_score(args)
            outcome = json.loads((output / "outcome_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(outcome["outcome"], "invalid_run")
            self.assertFalse(outcome["promote_v4"])
            self.assertEqual(outcome["status"], "terminal")
            self.assertNotIn("error_message", outcome)

            sealed_output = root / "sealed_final36" / "score-output"
            with self.assertRaisesRegex(ValueError, "sealed-final36"):
                scorer._cmd_score(argparse.Namespace(output_dir=sealed_output))
            self.assertFalse(sealed_output.exists())


if __name__ == "__main__":
    unittest.main()
