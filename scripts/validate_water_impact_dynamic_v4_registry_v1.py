#!/usr/bin/env python3
"""Fail-closed public/private validator for the v4 source-slot registries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "water_impact_dynamic_v4_source_slot_registry_v1"
PUBLIC_BANK_REL = Path("data/water_impact_dynamic_v4/source_bank_public64_registry_v1.json")
PUBLIC_HOLDOUT_REL = Path("data/water_impact_dynamic_v4/holdout_public_commitment_v1.json")
PUBLIC_STAGE0_REL = Path("data/water_impact_dynamic_v4/causal_stage0_public_commitment_v1.json")
STANDARD_STAGE0_REL = Path("data/water_impact_dynamic_v4/causal_stage0_commitment.json")
TRAIN_REL = Path("data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv")
TEST_REL = Path("data/water_impact_dynamic_v1/test_pairs.csv")
BUILDER_REL = Path("scripts/build_water_impact_dynamic_pairs_v1.py")
FORBIDDEN_PATH_TOKENS = ("sealed", "final36", "review", "score", "generation", "output", "media")
BANNED_EVENT_WORDS = {
    "water", "splash", "splashes", "ripple", "ripples", "impact", "impacts",
    "collision", "collisions", "contact", "contacts", "droplet", "drop", "drops",
    "enter", "enters", "entry", "fall", "falls", "cavity", "wave", "waves",
}
PRIVATE_FILES = {
    "curate_v4_registry.py",
    "source_curation_pool_private_v1.json",
    "source_ontology_private80_v1.json",
    "source_split_private_v1.json",
    "holdout_registry_private24_v1.json",
    "source_history_matrix_private_v1.json",
    "source_receiver_matrix_private_v1.json",
    "source_impact_matrix_private_v1.json",
    "receiver_ontology_private_v1.json",
    "salts_private_v1.json",
    "causal_stage0_candidates_private_v1.json",
    "causal_stage0_templates_private_v1.json",
    "causal_stage0_field_rules_private_v1.json",
    "causal_stage0_render_config_private_v1.json",
    "causal_stage0_selection_rules_private_v1.json",
    "causal_stage0_secrets_private_v1.json",
    "causal_stage0_bundle_private_v1.json",
    "registry_private_manifest_v1.json",
}


class ValidationError(RuntimeError):
    """Raised without embedding private record values in the message."""


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
    path = repo / BUILDER_REL
    spec = importlib.util.spec_from_file_location("v4_registry_allowed_builder", path)
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
    require(variants == Counter({"natural": 90, "direct": 88}), "erase variant counts changed")

    test_sources: dict[str, str] = {}
    test_receivers: dict[str, str] = {}
    # Deliberately consume no record-level fields other than the aggregate identity blacklist.
    with (repo / TEST_REL).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            test_sources[row["source_id"].strip()] = row["source_object"].strip()
            test_receivers[row["receiver_id"].strip()] = row["receiver"].strip()
    require(train_sources == dict(builder.TRAIN_SOURCES), "training source identities changed")
    require(train_receivers == dict(builder.TRAIN_RECEIVERS), "training receiver identities changed")
    return train_sources, train_receivers, test_sources, test_receivers


def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = ["origin", "food_status", "shape_class", "color_family", "material_family", "texture_class"]
    result: dict[str, Any] = {"count": len(items)}
    for key in dimensions:
        result[key] = dict(sorted(Counter(str(item[key]) for item in items).items()))
    result["origin_x_food_status"] = dict(sorted(Counter(f"{item['origin']}:{item['food_status']}" for item in items).items()))
    return result


def validate_stat_shape(value: Any, expected_count: int) -> None:
    require(isinstance(value, dict) and value.get("count") == expected_count, "aggregate strata count mismatch")
    for key in ("origin", "food_status", "shape_class", "color_family", "material_family", "texture_class", "origin_x_food_status"):
        part = value.get(key)
        require(isinstance(part, dict) and all(isinstance(v, int) and v >= 0 for v in part.values()), "aggregate strata shape invalid")
        require(sum(part.values()) == expected_count, "aggregate strata dimension does not sum to count")


def validate_public_objects(repo: Path, bank: dict[str, Any], holdout_public: dict[str, Any], stage0_public: dict[str, Any]) -> dict[str, Any]:
    builder = load_builder(repo)
    train_sources, train_receivers, test_sources, test_receivers = identity_aggregates(repo, builder)

    require(bank.get("schema") == SCHEMA, "public bank schema mismatch")
    require(bank.get("registry") == "public_augmentation_bank64_v1", "public bank registry name mismatch")
    require(bank.get("counts") == {"total": 64, "original_training": 8, "new_ontology": 56}, "public bank counts mismatch")
    require(bank.get("canonical_builder_sha256") == sha256_file(repo / BUILDER_REL), "canonical builder hash mismatch")
    require(bank.get("training_manifest_sha256") == sha256_file(repo / TRAIN_REL), "training manifest hash mismatch")
    require(is_hex64(bank.get("source_assignment_salt")), "public assignment salt must be an independent hex-64 value")
    assignment = bank.get("source_assignment_algorithm")
    require(isinstance(assignment, dict), "assignment algorithm missing")
    require(assignment.get("algorithm_id") == "fixed64_permutation_cycle_partitioned_hash_swap_v1", "assignment algorithm ID mismatch")
    require(set(assignment) == {"algorithm_id", "permutation", "collision_policy", "rng_rule", "mapping_commitments"}, "assignment algorithm incomplete")
    permutation = assignment.get("permutation")
    require(isinstance(permutation, dict) and permutation.get("hash") == "sha256", "assignment permutation hash mismatch")
    require(permutation.get("payload") == "utf8(source_assignment_salt) || 0x00 || utf8('permute') || 0x00 || utf8(source_id)", "assignment permutation payload mismatch")
    require("once" in permutation.get("application", "") and "never reshuffle per cycle" in permutation.get("application", ""), "assignment must use one fixed cyclic permutation")
    collision = assignment.get("collision_policy")
    require(isinstance(collision, dict) and collision.get("processing_order") == "ascending erase ordinal", "assignment collision order mismatch")
    require(collision.get("partitions") == [[0, 100], [100, 178]], "assignment collision partitions mismatch")
    require(collision.get("candidate_rank") == "sha256(utf8(source_assignment_salt) || 0x00 || utf8('swap') || 0x00 || utf8(decimal_position) || 0x00 || utf8(decimal_candidate))", "assignment collision rank mismatch")
    require("smallest digest" in collision.get("selection", "") and "no eligible candidate are fatal" in collision.get("selection", ""), "assignment collision selection mismatch")

    entries = bank.get("entries")
    require(isinstance(entries, list) and len(entries) == 64, "public bank must contain 64 entries")
    require(bank.get("bank_entries_sha256") == sha256_bytes(canonical_bytes(entries)), "bank-entry commitment mismatch")
    require([entry.get("bank_index") for entry in entries] == list(range(64)), "bank indices are not canonical")
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
        }
        for index, (source_id, phrase) in enumerate(builder.TRAIN_SOURCES)
    ]
    require(original == expected_original, "the eight original-source exceptions changed")
    require(all(entry.get("membership") == "new_bank_source" for entry in new), "new-bank membership label mismatch")
    require(len({entry.get("source_id") for entry in new}) == 56, "new-bank source IDs are not unique")
    require(len({entry.get("normalized_phrase") for entry in new}) == 56, "new-bank phrases are not unique")
    require(len({entry.get("head_lemma") for entry in new}) == 56, "new-bank head lemmas are not unique")
    historic_source_phrases = {normalize_phrase(value) for value in {**test_sources, **train_sources}.values()}
    historic_source_heads = {value.split()[-1] for value in historic_source_phrases}
    historic_receiver_phrases = {normalize_phrase(value) for value in {**test_receivers, **train_receivers}.values()}
    historic_receiver_id_heads = {key.split("_")[-1] for key in {**test_receivers, **train_receivers}}
    for entry in new:
        require(entry.get("normalized_phrase") == normalize_phrase(entry.get("source_phrase", "")), "new-bank phrase normalization mismatch")
        require(entry.get("head_lemma") == normalize_phrase(entry.get("head_lemma", "")), "new-bank head normalization mismatch")
        require(entry["normalized_phrase"] not in historic_source_phrases, "new-bank phrase overlaps a historical source")
        require(entry["head_lemma"] not in historic_source_heads, "new-bank head overlaps a historical source")
        require(entry["normalized_phrase"] not in historic_receiver_phrases, "new-bank phrase overlaps a historical receiver")
        require(entry["head_lemma"] not in historic_receiver_id_heads, "new-bank head overlaps a historical receiver")
        require(not (set(entry["normalized_phrase"].split()) & BANNED_EVENT_WORDS), "new-bank phrase contains event vocabulary")
        strata = entry.get("strata")
        require(isinstance(strata, dict) and set(strata) == {"origin", "food_status", "shape_class", "color_family", "material_family", "texture_class"}, "new-bank strata incomplete")
        impact = entry.get("impact_plausibility")
        require(isinstance(impact, dict) and impact.get("verdict") == "pass", "public new-bank impact audit missing")
        require(impact.get("compact_and_rigid") is True and impact.get("natural_drop_entry") is True, "public new-bank impact audit failed")
        require(impact.get("visible_brief_splash_or_ripple_plausible") is True, "public new-bank response plausibility failed")
        require(impact.get("predominantly_buoyant_or_windborne") is False and impact.get("flexible_or_film_like") is False, "public new-bank buoyancy/flexibility audit failed")

    require(holdout_public.get("schema") == SCHEMA, "holdout commitment schema mismatch")
    require(holdout_public.get("registry") == "public_holdout24_commitment_v1", "holdout commitment name mismatch")
    require(holdout_public.get("holdout_count") == 24, "holdout count mismatch")
    require(is_hex64(holdout_public.get("holdout_registry_file_sha256")), "holdout registry commitment invalid")
    require(is_hex64(holdout_public.get("split_salt_commitment_sha256")), "split-salt commitment invalid")
    aggregate = holdout_public.get("aggregate_strata")
    require(isinstance(aggregate, dict) and set(aggregate) == {"curated_new80", "new_bank56", "private_holdout24"}, "aggregate strata partitions invalid")
    validate_stat_shape(aggregate["curated_new80"], 80)
    validate_stat_shape(aggregate["new_bank56"], 56)
    validate_stat_shape(aggregate["private_holdout24"], 24)
    new_as_sources = [{**entry["strata"]} for entry in new]
    require(stats(new_as_sources) == aggregate["new_bank56"], "public new-bank strata mismatch")
    cross = holdout_public.get("cross_role_checks")
    require(isinstance(cross, dict), "cross-role summary missing")
    for key in ("normalized_phrase_overlap_count", "normalized_head_overlap_count", "historical_source_semantic_equivalence_count", "source_receiver_semantic_equivalence_count", "event_vocabulary_overlap_count", "impact_plausibility_failure_count"):
        require(cross.get(key) == 0, "a public disjointness or plausibility check failed")
    require(cross.get("impact_plausibility_pass_count") == 80, "impact-plausibility pass count mismatch")

    require(stage0_public.get("schema") == SCHEMA, "Stage-0 schema mismatch")
    require(stage0_public.get("dataset_version") == "v4_dev72_v1" and stage0_public.get("stage") == 0, "Stage-0 identity mismatch")
    require(stage0_public.get("candidate_count") == 48, "Stage-0 candidate count mismatch")
    expected_cells = {
        "holdout_source_seen_receiver:direct": 8,
        "holdout_source_seen_receiver:natural": 8,
        "seen_source_new_receiver:direct": 8,
        "seen_source_new_receiver:natural": 8,
        "holdout_source_new_receiver:direct": 8,
        "holdout_source_new_receiver:natural": 8,
    }
    require(stage0_public.get("cell_counts") == expected_cells, "Stage-0 cell counts mismatch")
    require(stage0_public.get("status") == "frozen_components_pending_forbidden_seed_inventory", "Stage-0 status overstates readiness")
    require(stage0_public.get("authorization_status") == "not_authorized", "Stage-0 must remain unauthorized")
    blockers = stage0_public.get("remaining_blockers")
    require(isinstance(blockers, list) and len(blockers) == 2, "Stage-0 blocker set mismatch")
    require(any("forbidden numeric seed inventory" in item for item in blockers), "Stage-0 seed blocker missing")
    require(any("path-plus-file-bytes inventory digest" in item for item in blockers), "Stage-0 model-inventory blocker missing")
    for key in (
        "candidate_manifest_sha256", "canonical_templates_sha256", "field_normalization_sha256",
        "render_configuration_sha256", "selector_rules_sha256", "screening_seed_commitment_sha256",
        "selector_salt_commitment_sha256", "evaluation_seed_salt_commitment_sha256", "stage0_bundle_file_sha256",
    ):
        require(is_hex64(stage0_public.get(key)), "Stage-0 commitment field invalid")
    public_meta = stage0_public.get("public_metadata")
    require(isinstance(public_meta, dict), "Stage-0 public metadata missing")
    require(public_meta.get("selected_case_target") == 24 and public_meta.get("evaluation_unit_target") == 72, "Stage-0 target counts mismatch")
    require(not (repo / STANDARD_STAGE0_REL).exists(), "authorizing Stage-0 wrapper exists before seed inventory")
    return {"bank_count": 64, "new_bank_count": 56, "holdout_count": 24, "stage0_candidate_count": 48}


def validate_public(repo: Path) -> dict[str, Any]:
    bank = load_json(repo / PUBLIC_BANK_REL)
    holdout_public = load_json(repo / PUBLIC_HOLDOUT_REL)
    stage0_public = load_json(repo / PUBLIC_STAGE0_REL)
    return validate_public_objects(repo, bank, holdout_public, stage0_public)


def validate_private(repo: Path, private: Path) -> dict[str, Any]:
    public_summary = validate_public(repo)
    require(private.is_dir(), "private registry directory is missing")
    require(not any(token in part.casefold() for part in private.parts for token in FORBIDDEN_PATH_TOKENS), "private registry path violates clean-room path policy")
    actual = {path.name for path in private.iterdir() if path.is_file()}
    require(actual == PRIVATE_FILES, "private registry file inventory mismatch")
    require(not any(path.is_dir() for path in private.iterdir()), "nested private directories are forbidden")

    manifest = load_json(private / "registry_private_manifest_v1.json")
    require(manifest.get("schema") == SCHEMA and manifest.get("manifest") == "private_clean_registry_manifest_v1", "private manifest identity mismatch")
    component_names = PRIVATE_FILES - {"curate_v4_registry.py", "registry_private_manifest_v1.json"}
    require(set(manifest.get("private_components", {})) == component_names, "private manifest component set mismatch")
    for name in component_names:
        require(manifest["private_components"][name] == sha256_file(private / name), "private component hash mismatch")
    require(manifest.get("generator_sha256") == sha256_file(private / "curate_v4_registry.py"), "private curator hash mismatch")
    expected_public_hashes = {
        PUBLIC_BANK_REL.as_posix(): sha256_file(repo / PUBLIC_BANK_REL),
        PUBLIC_HOLDOUT_REL.as_posix(): sha256_file(repo / PUBLIC_HOLDOUT_REL),
        PUBLIC_STAGE0_REL.as_posix(): sha256_file(repo / PUBLIC_STAGE0_REL),
    }
    require(manifest.get("public_files") == expected_public_hashes, "private/public file binding mismatch")
    expected_inputs = {
        "train_dynamic_sft_preserve_v2.csv": sha256_file(repo / TRAIN_REL),
        "test_pairs.csv": sha256_file(repo / TEST_REL),
        "build_water_impact_dynamic_pairs_v1.py": sha256_file(repo / BUILDER_REL),
    }
    require(manifest.get("allowlisted_inputs") == expected_inputs, "allowlisted input binding mismatch")

    bank = load_json(repo / PUBLIC_BANK_REL)
    holdout_public = load_json(repo / PUBLIC_HOLDOUT_REL)
    stage0_public = load_json(repo / PUBLIC_STAGE0_REL)
    salts = load_json(private / "salts_private_v1.json")
    required_salts = {
        "ontology_sampling_salt", "split_salt", "causal_stage0_selector_salt",
        "causal_evaluation_seed_salt", "causal_screening_seed_token",
    }
    require(salts.get("schema") == SCHEMA and required_salts <= set(salts), "private salt registry incomplete")
    require(all(is_hex64(salts[key]) for key in required_salts), "private salt shape invalid")
    require(bank["source_assignment_salt"] not in {salts[key] for key in required_salts}, "public assignment salt reuses a private salt")
    require(holdout_public["split_salt_commitment_sha256"] == commitment("split_salt", salts["split_salt"]), "split-salt commitment mismatch")

    pool_doc = load_json(private / "source_curation_pool_private_v1.json")
    ontology_doc = load_json(private / "source_ontology_private80_v1.json")
    pool = pool_doc.get("candidates")
    ontology = ontology_doc.get("sources")
    require(pool_doc.get("candidate_count") == 128 and isinstance(pool, list) and len(pool) == 128, "private curation pool count mismatch")
    require(ontology_doc.get("source_count") == 80 and isinstance(ontology, list) and len(ontology) == 80, "private ontology count mismatch")
    for collection, expected in ((pool, 128), (ontology, 80)):
        require(len({item.get("source_id") for item in collection}) == expected, "private source IDs are not unique")
        require(len({item.get("normalized_phrase") for item in collection}) == expected, "private source phrases are not unique")
        require(len({item.get("head_lemma") for item in collection}) == expected, "private source heads are not unique")
    expected_ontology: list[dict[str, Any]] = []
    for origin, food_status in (("natural", "food"), ("natural", "nonfood"), ("manufactured", "food"), ("manufactured", "nonfood")):
        cell = [item for item in pool if item.get("origin") == origin and item.get("food_status") == food_status and item.get("impact_plausibility", {}).get("verdict") == "pass"]
        require(len(cell) >= 20, "impact-plausible curation stratum is undersized")
        expected_ontology.extend(sorted(cell, key=lambda item: salted_rank(salts["ontology_sampling_salt"], "ontology-v1", item))[:20])
    expected_ontology.sort(key=lambda item: item["source_id"])
    require(ontology == expected_ontology, "private ontology selection is not deterministic")
    for item in ontology:
        impact = item.get("impact_plausibility", {})
        require(impact.get("verdict") == "pass", "ontology contains an impact-plausibility rejection")
        require(impact.get("compact_and_rigid") is True and impact.get("natural_drop_entry") is True, "ontology contains a noncompact or implausible source")
        require(impact.get("visible_brief_splash_or_ripple_plausible") is True, "ontology source lacks visible-response plausibility")
        require(impact.get("predominantly_buoyant_or_windborne") is False and impact.get("flexible_or_film_like") is False, "ontology contains buoyant or flexible source")
        require(not (set(item["normalized_phrase"].split()) & BANNED_EVENT_WORDS), "ontology contains event vocabulary")

    builder = load_builder(repo)
    train_sources, train_receivers, test_sources, test_receivers = identity_aggregates(repo, builder)
    historic_source_phrases = {normalize_phrase(value) for value in {**test_sources, **train_sources}.values()}
    historic_source_heads = {value.split()[-1] for value in historic_source_phrases}
    receiver_doc = load_json(private / "receiver_ontology_private_v1.json")
    historic_receivers = receiver_doc.get("historical_receivers")
    new_receivers = receiver_doc.get("new_receivers")
    require(isinstance(historic_receivers, list) and len(historic_receivers) == len({**test_receivers, **train_receivers}), "historical receiver aggregate mismatch")
    require(isinstance(new_receivers, list) and len(new_receivers) == 32, "new receiver ontology count mismatch")
    require(len({item.get("receiver_id") for item in new_receivers}) == 32, "new receiver IDs are not unique")
    receiver_phrases = {item["normalized_phrase"] for item in historic_receivers + new_receivers}
    receiver_heads = {item["receiver_id"].split("_")[-1] for item in historic_receivers} | {item["head_lemma"] for item in new_receivers}
    for item in ontology:
        require(item["normalized_phrase"] not in historic_source_phrases and item["head_lemma"] not in historic_source_heads, "ontology overlaps a historical source")
        require(item["normalized_phrase"] not in receiver_phrases and item["head_lemma"] not in receiver_heads, "ontology overlaps receiver ontology")

    split_doc = load_json(private / "source_split_private_v1.json")
    split_rows = split_doc.get("rows")
    require(isinstance(split_rows, list) and len(split_rows) == 80, "private split row count mismatch")
    split_by_id = {row.get("source_id"): row for row in split_rows}
    require(len(split_by_id) == 80, "private split IDs are not unique")
    ranked = sorted(ontology, key=lambda item: salted_rank(salts["split_salt"], "bank-holdout-v1", item))
    bank_ids = {item["source_id"] for item in ranked[:56]}
    holdout_ids = {item["source_id"] for item in ranked[56:]}
    for item in ontology:
        row = split_by_id[item["source_id"]]
        require(row.get("split_rank_sha256") == salted_rank(salts["split_salt"], "bank-holdout-v1", item), "private split rank mismatch")
        expected_membership = "new_bank_source" if item["source_id"] in bank_ids else "holdout_source"
        require(row.get("membership") == expected_membership, "private split membership mismatch")
    public_new_ids = {entry["source_id"] for entry in bank["entries"] if entry["membership"] == "new_bank_source"}
    require(public_new_ids == bank_ids and public_new_ids.isdisjoint(holdout_ids), "public bank/private split mismatch")

    ontology_by_id = {item["source_id"]: item for item in ontology}
    new_bank_items = [ontology_by_id[key] for key in sorted(bank_ids)]
    holdout_items = [ontology_by_id[key] for key in sorted(holdout_ids)]
    aggregate = holdout_public["aggregate_strata"]
    require(stats(ontology) == aggregate["curated_new80"], "curated ontology strata commitment mismatch")
    require(stats(new_bank_items) == aggregate["new_bank56"], "new-bank strata commitment mismatch")
    require(stats(holdout_items) == aggregate["private_holdout24"], "holdout strata commitment mismatch")
    holdout_path = private / "holdout_registry_private24_v1.json"
    holdout_registry = load_json(holdout_path)
    holdout_entries = holdout_registry.get("entries")
    require(isinstance(holdout_entries, list) and len(holdout_entries) == 24, "private holdout registry count mismatch")
    require([entry.get("source_id") for entry in holdout_entries] == sorted(holdout_ids), "private holdout ordering mismatch")
    require(sha256_file(holdout_path) == holdout_public["holdout_registry_file_sha256"], "holdout file commitment mismatch")

    history_matrix = load_json(private / "source_history_matrix_private_v1.json").get("rows")
    expected_history_pairs = {(item["source_id"], source_id) for item in ontology for source_id in {**test_sources, **train_sources}}
    require(isinstance(history_matrix, list) and len(history_matrix) == len(expected_history_pairs), "historical semantic matrix size mismatch")
    require({(row.get("new_source_id"), row.get("historical_source_id")) for row in history_matrix} == expected_history_pairs, "historical semantic matrix coverage mismatch")
    require(all(row.get("semantic_equivalent") is False for row in history_matrix), "historical semantic-equivalence rejection failed")
    receiver_matrix = load_json(private / "source_receiver_matrix_private_v1.json").get("rows")
    receiver_ids = {item["receiver_id"] for item in historic_receivers + new_receivers}
    expected_receiver_pairs = {(item["source_id"], receiver_id) for item in ontology for receiver_id in receiver_ids}
    require(isinstance(receiver_matrix, list) and len(receiver_matrix) == len(expected_receiver_pairs), "source/receiver semantic matrix size mismatch")
    require({(row.get("new_source_id"), row.get("receiver_id")) for row in receiver_matrix} == expected_receiver_pairs, "source/receiver semantic matrix coverage mismatch")
    require(all(row.get("semantic_equivalent") is False for row in receiver_matrix), "source/receiver semantic-equivalence rejection failed")
    impact_rows = load_json(private / "source_impact_matrix_private_v1.json").get("rows")
    require(isinstance(impact_rows, list) and len(impact_rows) == 80, "impact-plausibility matrix count mismatch")
    require({row.get("source_id") for row in impact_rows} == set(ontology_by_id), "impact-plausibility matrix coverage mismatch")
    require(all(row.get("verdict") == "pass" for row in impact_rows), "impact-plausibility matrix contains a rejection")

    candidates_doc = load_json(private / "causal_stage0_candidates_private_v1.json")
    candidates = candidates_doc.get("candidates")
    require(candidates_doc.get("candidate_count") == 48 and isinstance(candidates, list) and len(candidates) == 48, "Stage-0 private candidate count mismatch")
    require(len({row.get("case_id") for row in candidates}) == 48, "Stage-0 case IDs are not unique")
    cell_counts = Counter(f"{row.get('group')}:{row.get('prompt_variant')}" for row in candidates)
    require(dict(sorted(cell_counts.items())) == stage0_public["cell_counts"], "Stage-0 private cell counts mismatch")
    train_receiver_ids = set(train_receivers)
    new_receiver_ids = {item["receiver_id"] for item in new_receivers}
    original_ids = set(train_sources)
    for row in candidates:
        base = dict(row)
        record_hash = base.pop("canonical_record_sha256", None)
        require(record_hash == sha256_bytes(canonical_bytes(base)), "Stage-0 candidate record hash mismatch")
        require(row.get("prompt_variant") in {"direct", "natural"}, "Stage-0 prompt variant invalid")
        rebuilt = builder.factual_prompt(row.get("source_phrase", ""), row.get("receiver_phrase", ""), row["prompt_variant"])
        require(row.get("canonical_prompt") == rebuilt, "Stage-0 canonical prompt mismatch")
        group = row.get("group")
        if group == "holdout_source_seen_receiver":
            require(row.get("source_id") in holdout_ids and row.get("receiver_id") in train_receiver_ids, "Stage-0 holdout/seen group identity mismatch")
        elif group == "seen_source_new_receiver":
            require(row.get("source_id") in original_ids and row.get("receiver_id") in new_receiver_ids, "Stage-0 seen/new group identity mismatch")
        elif group == "holdout_source_new_receiver":
            require(row.get("source_id") in holdout_ids and row.get("receiver_id") in new_receiver_ids, "Stage-0 holdout/new group identity mismatch")
        else:
            raise ValidationError("Stage-0 group invalid")

    templates = load_json(private / "causal_stage0_templates_private_v1.json")
    require(templates.get("canonical_builder_sha256") == sha256_file(repo / BUILDER_REL), "Stage-0 template builder binding mismatch")
    expected_templates = {variant: builder.factual_prompt("{source_phrase}", "{receiver_phrase}", variant) for variant in ("direct", "natural")}
    require(templates.get("prompt_templates") == expected_templates, "Stage-0 canonical templates mismatch")
    require(templates.get("template_fill_rules") == {
        "direct": {"source_phrase": "python_str_capitalize", "receiver_phrase": "identity"},
        "natural": {"source_phrase": "identity", "receiver_phrase": "identity"},
    }, "Stage-0 template fill rules mismatch")
    for row in candidates:
        source_value = row["source_phrase"].capitalize() if row["prompt_variant"] == "direct" else row["source_phrase"]
        formatted = templates["prompt_templates"][row["prompt_variant"]].format(
            source_phrase=source_value,
            receiver_phrase=row["receiver_phrase"],
        )
        require(formatted == row["canonical_prompt"], "Stage-0 template fill rule does not reproduce candidate prompt")
    render = load_json(private / "causal_stage0_render_config_private_v1.json")
    require({key: render.get(key) for key in ("arm", "model_family", "steps", "cfg", "frames", "width", "height", "fps", "dtype", "adapter")} == {
        "arm": "Original_only", "model_family": "Wan 2.1 T2V 1.3B", "steps": 25, "cfg": 5,
        "frames": 49, "width": 832, "height": 480, "fps": 8, "dtype": "bf16", "adapter": None,
    }, "Stage-0 render configuration mismatch")
    stage0_secrets = load_json(private / "causal_stage0_secrets_private_v1.json")
    require(stage0_public["screening_seed_commitment_sha256"] == commitment("causal_screening_seed", str(stage0_secrets.get("screening_seed"))), "screening-seed commitment mismatch")
    require(stage0_public["selector_salt_commitment_sha256"] == commitment("causal_stage0_selector_salt", stage0_secrets.get("selector_salt", "")), "selector-salt commitment mismatch")
    require(stage0_public["evaluation_seed_salt_commitment_sha256"] == commitment("causal_evaluation_seed_salt", stage0_secrets.get("evaluation_seed_salt", "")), "evaluation-seed salt commitment mismatch")
    require(stage0_secrets.get("selector_salt") == salts["causal_stage0_selector_salt"] and stage0_secrets.get("evaluation_seed_salt") == salts["causal_evaluation_seed_salt"], "Stage-0 secret registry mismatch")

    components = manifest["private_components"]
    require(stage0_public["candidate_manifest_sha256"] == components["causal_stage0_candidates_private_v1.json"], "Stage-0 candidate commitment mismatch")
    require(stage0_public["canonical_templates_sha256"] == components["causal_stage0_templates_private_v1.json"], "Stage-0 template commitment mismatch")
    require(stage0_public["field_normalization_sha256"] == components["causal_stage0_field_rules_private_v1.json"], "Stage-0 field-rule commitment mismatch")
    require(stage0_public["render_configuration_sha256"] == components["causal_stage0_render_config_private_v1.json"], "Stage-0 render commitment mismatch")
    require(stage0_public["selector_rules_sha256"] == components["causal_stage0_selection_rules_private_v1.json"], "Stage-0 selector commitment mismatch")
    require(stage0_public["stage0_bundle_file_sha256"] == components["causal_stage0_bundle_private_v1.json"], "Stage-0 bundle commitment mismatch")
    bundle = load_json(private / "causal_stage0_bundle_private_v1.json")
    require(bundle.get("status") == "frozen_components_pending_forbidden_seed_inventory", "private Stage-0 status overstates readiness")
    require(bundle.get("source_bank_entries_sha256") == bank["bank_entries_sha256"] and bundle.get("holdout_registry_file_sha256") == holdout_public["holdout_registry_file_sha256"], "Stage-0 source registry binding mismatch")

    public_text = "\n".join((repo / path).read_text(encoding="utf-8") for path in (PUBLIC_BANK_REL, PUBLIC_HOLDOUT_REL, PUBLIC_STAGE0_REL))
    for key in required_salts:
        require(salts[key] not in public_text, "a private salt leaked into a public artifact")
    require(stage0_secrets["selector_salt"] not in public_text and stage0_secrets["evaluation_seed_salt"] not in public_text, "a Stage-0 private salt leaked into a public artifact")
    for entry in holdout_entries:
        require(entry["source_id"] not in public_text and entry["source_phrase"] not in public_text and entry["normalized_phrase"] not in public_text, "a holdout identity leaked into a public artifact")

    return {
        **public_summary,
        "ontology_count": 80,
        "impact_plausibility_pass_count": 80,
        "private_file_count": len(PRIVATE_FILES),
        "private_manifest_sha256": sha256_file(private / "registry_private_manifest_v1.json"),
        "stage0_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.private_dir is None:
            summary = validate_public(args.repo_root.resolve())
            scope = "public"
        else:
            summary = validate_private(args.repo_root.resolve(), args.private_dir.resolve())
            scope = "public_and_private"
    except ValidationError as exc:
        print(json.dumps({"status": "invalid", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"status": "valid", "scope": scope, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
