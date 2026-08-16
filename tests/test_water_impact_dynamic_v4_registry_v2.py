#!/usr/bin/env python3
"""Mutation and private-opening tests for the v4_dev72_v2 registry."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "water_impact_dynamic_v4"
VALIDATOR_PATH = REPO / "scripts" / "validate_water_impact_dynamic_v4_registry_v2.py"
SPEC = importlib.util.spec_from_file_location("v4_registry_v2_validator", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("v2 validator import failed")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def load_public() -> tuple[dict, dict, dict]:
    return tuple(
        validator.load_json(DATA / name)
        for name in (
            validator.PUBLIC_BANK_NAME,
            validator.PUBLIC_HOLDOUT_NAME,
            validator.PUBLIC_STAGE0_NAME,
        )
    )


def rebind_bank_entries(bank: dict) -> None:
    bank["bank_entries_sha256"] = validator.sha256_bytes(validator.canonical_bytes(bank["entries"]))


class PublicRegistryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank, cls.holdout, cls.stage0 = load_public()

    def assert_public_invalid(self, bank: dict, holdout: dict, stage0: dict) -> None:
        with self.assertRaises(validator.ValidationError):
            validator.validate_public_objects(REPO, DATA, bank, holdout, stage0)

    def test_public_baseline_is_valid_and_non_authorizing(self) -> None:
        summary = validator.validate_public(REPO, DATA)
        self.assertEqual(summary, {
            "bank_count": 64,
            "new_bank_count": 56,
            "holdout_count": 24,
            "stage0_candidate_count": 48,
        })
        self.assertEqual(self.stage0["authorization_status"], "not_authorized")
        self.assertFalse((DATA / validator.STANDARD_STAGE0_NAME).exists())
        self.assertEqual(self.bank["curation_audit"], validator.CURATION_AUDIT)
        self.assertEqual(self.holdout["curation_audit"], validator.CURATION_AUDIT)
        self.assertEqual(self.stage0["curation_audit"], validator.CURATION_AUDIT)

    def test_exactly_original_eight_are_legacy_exempt(self) -> None:
        self.assertEqual(
            [row["physical_audit_status"] for row in self.bank["entries"][:8]],
            [validator.LEGACY_PHYSICAL_STATUS] * 8,
        )
        self.assertEqual(
            [row["physical_audit_status"] for row in self.bank["entries"][8:]],
            [validator.STRICT_PHYSICAL_STATUS] * 56,
        )
        bank = copy.deepcopy(self.bank)
        bank["entries"][8]["physical_audit_status"] = validator.LEGACY_PHYSICAL_STATUS
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

    def test_v1_identities_and_domains_are_rejected(self) -> None:
        mutations = []
        bank = copy.deepcopy(self.bank)
        bank["dataset_version"] = "v4_dev72_v1"
        mutations.append((bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0)))
        bank = copy.deepcopy(self.bank)
        bank["source_assignment_algorithm"]["algorithm_id"] = "fixed64_permutation_cycle_partitioned_hash_swap_v1"
        mutations.append((bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0)))
        bank = copy.deepcopy(self.bank)
        bank["source_assignment_algorithm"]["permutation"]["payload"] = bank["source_assignment_algorithm"]["permutation"]["payload"].replace("permute-v2", "permute")
        mutations.append((bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0)))
        bank = copy.deepcopy(self.bank)
        bank["source_assignment_algorithm"]["collision_policy"]["candidate_rank"] = bank["source_assignment_algorithm"]["collision_policy"]["candidate_rank"].replace("swap-v2", "swap")
        mutations.append((bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0)))
        for mutation in mutations:
            with self.subTest():
                self.assert_public_invalid(*mutation)

    def test_assignment_partition_and_single_permutation_contract_is_exact(self) -> None:
        for key, value in (
            (("collision_policy", "partitions"), [[0, 178]]),
            (("collision_policy", "processing_order"), "descending erase ordinal"),
            (("permutation", "application"), "reshuffle every cycle"),
        ):
            bank = copy.deepcopy(self.bank)
            bank["source_assignment_algorithm"][key[0]][key[1]] = value
            with self.subTest(field=key):
                self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

    def test_impact_boolean_and_numeric_failures_are_rejected(self) -> None:
        changes = (
            ("food_or_produce", True),
            ("fragile", True),
            ("negative_buoyancy", False),
            ("density_g_cm3", 2.99),
            ("mass_g", 349),
            ("dimensions_cm", [2.4, 8.0, 8.0]),
        )
        for key, value in changes:
            bank = copy.deepcopy(self.bank)
            bank["entries"][8]["impact_plausibility"][key] = value
            rebind_bank_entries(bank)
            with self.subTest(field=key):
                self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

        bank = copy.deepcopy(self.bank)
        audit = bank["entries"][8]["impact_plausibility"]
        audit["density_g_cm3"] = 3.0
        audit["mass_g"] = 1200
        audit["dimensions_cm"] = [8.0, 2.5, 2.5]
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

    def test_event_size_and_category_risks_are_rejected(self) -> None:
        bank = copy.deepcopy(self.bank)
        bank["entries"][8]["source_id"] += "_splash"
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

        bank = copy.deepcopy(self.bank)
        row = bank["entries"][8]
        row["source_phrase"] = row["source_phrase"].replace("palm-sized", "small")
        row["normalized_phrase"] = validator.normalize_phrase(row["source_phrase"])
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

        bank = copy.deepcopy(self.bank)
        row = bank["entries"][8]
        row["source_phrase"] = f"a dense palm-sized foam {row['head_lemma']}"
        row["normalized_phrase"] = validator.normalize_phrase(row["source_phrase"])
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

    def test_duplicate_source_specific_notes_are_rejected(self) -> None:
        bank = copy.deepcopy(self.bank)
        first, second = bank["entries"][8:10]
        shared = f"Distinct audit of {first['head_lemma']} and {second['head_lemma']}."
        first["impact_plausibility"]["curator_note"] = shared
        second["impact_plausibility"]["curator_note"] = shared
        rebind_bank_entries(bank)
        self.assert_public_invalid(bank, copy.deepcopy(self.holdout), copy.deepcopy(self.stage0))

    def test_supersession_and_stage0_boundary_are_exact(self) -> None:
        holdout = copy.deepcopy(self.holdout)
        holdout["supersedes"]["reason_code"] = "replacement_without_preflight_invalidation"
        self.assert_public_invalid(copy.deepcopy(self.bank), holdout, copy.deepcopy(self.stage0))

        for field, value in (
            ("authorization_status", "authorized"),
            ("status", "committed"),
        ):
            stage0 = copy.deepcopy(self.stage0)
            stage0[field] = value
            with self.subTest(field=field):
                self.assert_public_invalid(copy.deepcopy(self.bank), copy.deepcopy(self.holdout), stage0)

        stage0 = copy.deepcopy(self.stage0)
        stage0["public_metadata"]["screening_seed_namespace"] = "v4-causal-stage0-screening-v1"
        self.assert_public_invalid(copy.deepcopy(self.bank), copy.deepcopy(self.holdout), stage0)

        stage0 = copy.deepcopy(self.stage0)
        stage0["public_metadata"]["ranking_domain"] = "causal-selector-v1"
        self.assert_public_invalid(copy.deepcopy(self.bank), copy.deepcopy(self.holdout), stage0)


PRIVATE_ENV = os.environ.get("V4_PRIVATE_REGISTRY_V2_DIR")


@unittest.skipUnless(PRIVATE_ENV, "private v2 opening not requested")
class PrivateRegistryV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.private = Path(PRIVATE_ENV).resolve()

    def test_private_opening_is_valid(self) -> None:
        summary = validator.validate_private(REPO, DATA, self.private)
        self.assertEqual(summary["ontology_count"], 80)
        self.assertEqual(summary["holdout_impact_pass_count"], 24)
        self.assertEqual(summary["stage0_candidate_count"], 48)
        self.assertEqual(summary["stage0_valid_holdout_head_count"], 24)
        self.assertTrue(summary["stage0_global_selection_feasible"])
        self.assertFalse(summary["stage0_authorized"])

    def test_all_80_sources_have_strict_unique_audits(self) -> None:
        ontology = validator.load_json(self.private / "source_ontology_private80_v2.json")["sources"]
        self.assertEqual(len(ontology), 80)
        for row in ontology:
            validator.validate_source_identity(row)
        notes = [row["impact_plausibility"]["curator_note"] for row in ontology]
        features = [row["impact_plausibility"]["source_specific_feature"] for row in ontology]
        self.assertEqual(len(set(notes)), 80)
        self.assertEqual(len(set(features)), 80)
        self.assertLess(max(validator.jaccard(notes[i], notes[j]) for i in range(80) for j in range(i + 1, 80)), 0.78)

    def test_private_holdout_has_24_strict_passes(self) -> None:
        rows = validator.load_json(self.private / "holdout_registry_private24_v2.json")["entries"]
        self.assertEqual(len(rows), 24)
        for row in rows:
            validator.validate_source_identity(row)

    def test_semantic_matrices_are_complete_and_all_zero(self) -> None:
        cases = (
            ("source_history_matrix_private_v2.json", 80 * 14),
            ("receiver_history_matrix_private_v2.json", 32 * 52),
            ("source_receiver_matrix_private_v2.json", 80 * 84),
        )
        for name, count in cases:
            rows = validator.load_json(self.private / name)["rows"]
            with self.subTest(matrix=name):
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row["semantic_equivalent"] is False for row in rows))
                self.assertTrue(all(row["obvious_near_duplicate"] is False for row in rows))

    def test_stage0_is_feasible_and_registry_bound(self) -> None:
        rows = validator.load_json(self.private / "causal_stage0_candidates_private_v2.json")["candidates"]
        holdout = {
            row["source_id"]: row
            for row in validator.load_json(self.private / "holdout_registry_private24_v2.json")["entries"]
        }
        receivers = {
            row["receiver_id"]: row
            for row in validator.load_json(self.private / "receiver_ontology_private32_v2.json")["receivers"]
        }
        builder = validator.load_builder(REPO)
        train_sources = dict(builder.TRAIN_SOURCES)
        train_receivers = dict(builder.TRAIN_RECEIVERS)
        self.assertEqual(len(rows), 48)
        self.assertTrue(validator.selection_feasible(rows))
        for row in rows:
            if row["source_membership"] == "holdout_source":
                source = holdout[row["source_id"]]
                self.assertEqual((row["source_phrase"], row["source_head_lemma"]), (source["source_phrase"], source["head_lemma"]))
            else:
                self.assertEqual(row["source_phrase"], train_sources[row["source_id"]])
            if row["receiver_membership"] == "new_receiver":
                self.assertEqual(row["receiver_phrase"], receivers[row["receiver_id"]]["receiver_phrase"])
            else:
                self.assertEqual(row["receiver_phrase"], train_receivers[row["receiver_id"]])

        impossible = copy.deepcopy(rows)
        for row in impossible:
            if row["source_membership"] == "holdout_source":
                row["source_head_lemma"] = "one_duplicate_head"
        self.assertFalse(validator.selection_feasible(impossible))

    def test_private_identities_and_salts_are_absent_from_public_bytes(self) -> None:
        public_text = "\n".join(
            (DATA / name).read_text(encoding="utf-8")
            for name in (
                validator.PUBLIC_BANK_NAME,
                validator.PUBLIC_HOLDOUT_NAME,
                validator.PUBLIC_STAGE0_NAME,
            )
        )
        salts = validator.load_json(self.private / "salts_private_v2.json")
        for key in (
            "source_ontology_salt", "source_split_salt", "receiver_ontology_salt",
            "causal_stage0_selector_salt", "causal_evaluation_seed_salt",
            "causal_screening_seed_token",
        ):
            self.assertNotIn(salts[key], public_text)
        for row in validator.load_json(self.private / "holdout_registry_private24_v2.json")["entries"]:
            self.assertNotIn(row["source_id"], public_text)
            self.assertNotIn(row["source_phrase"], public_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
