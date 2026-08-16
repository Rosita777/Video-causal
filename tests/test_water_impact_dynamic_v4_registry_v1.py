from __future__ import annotations

import copy
import importlib.util
import json
import os
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO / "scripts/validate_water_impact_dynamic_v4_registry_v1.py"
SPEC = importlib.util.spec_from_file_location("v4_registry_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class PublicRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bank = validator.load_json(REPO / validator.PUBLIC_BANK_REL)
        cls.holdout = validator.load_json(REPO / validator.PUBLIC_HOLDOUT_REL)
        cls.stage0 = validator.load_json(REPO / validator.PUBLIC_STAGE0_REL)

    def test_public_registry_validates(self) -> None:
        summary = validator.validate_public(REPO)
        self.assertEqual(summary["bank_count"], 64)
        self.assertEqual(summary["new_bank_count"], 56)
        self.assertEqual(summary["holdout_count"], 24)
        self.assertEqual(summary["stage0_candidate_count"], 48)

    def test_bank_cardinality_mutation_fails_closed(self) -> None:
        bank = copy.deepcopy(self.bank)
        bank["entries"].pop()
        with self.assertRaises(validator.ValidationError):
            validator.validate_public_objects(REPO, bank, self.holdout, self.stage0)

    def test_assignment_salt_mutation_fails_closed(self) -> None:
        bank = copy.deepcopy(self.bank)
        bank["source_assignment_salt"] = "not-a-valid-public-assignment-salt"
        with self.assertRaises(validator.ValidationError):
            validator.validate_public_objects(REPO, bank, self.holdout, self.stage0)

    def test_assignment_contract_is_fixed_cycle_and_partitioned(self) -> None:
        algorithm = self.bank["source_assignment_algorithm"]
        self.assertEqual(algorithm["algorithm_id"], "fixed64_permutation_cycle_partitioned_hash_swap_v1")
        self.assertEqual(algorithm["collision_policy"]["partitions"], [[0, 100], [100, 178]])
        self.assertIn("never reshuffle per cycle", algorithm["permutation"]["application"])
        self.assertIn("smallest digest", algorithm["collision_policy"]["selection"])

    def test_holdout_count_mutation_fails_closed(self) -> None:
        holdout = copy.deepcopy(self.holdout)
        holdout["holdout_count"] = 23
        with self.assertRaises(validator.ValidationError):
            validator.validate_public_objects(REPO, self.bank, holdout, self.stage0)

    def test_stage0_count_mutation_fails_closed(self) -> None:
        stage0 = copy.deepcopy(self.stage0)
        stage0["candidate_count"] = 47
        with self.assertRaises(validator.ValidationError):
            validator.validate_public_objects(REPO, self.bank, self.holdout, stage0)

    def test_original_exceptions_are_scoped_to_first_eight(self) -> None:
        original = self.bank["entries"][:8]
        new = self.bank["entries"][8:]
        original_heads = [entry["head_lemma"] for entry in original]
        self.assertLess(len(set(original_heads)), len(original_heads))
        self.assertEqual(len({entry["head_lemma"] for entry in new}), 56)
        self.assertTrue(any("droplet" in entry["normalized_phrase"].split() for entry in original))
        self.assertFalse(any(set(entry["normalized_phrase"].split()) & validator.BANNED_EVENT_WORDS for entry in new))

    def test_public_new_bank_has_per_item_impact_audit(self) -> None:
        new = self.bank["entries"][8:]
        self.assertTrue(all(entry["impact_plausibility"]["verdict"] == "pass" for entry in new))
        self.assertTrue(all(entry["impact_plausibility"]["compact_and_rigid"] for entry in new))
        self.assertTrue(all(entry["impact_plausibility"]["visible_brief_splash_or_ripple_plausible"] for entry in new))
        self.assertFalse(any(entry["impact_plausibility"]["predominantly_buoyant_or_windborne"] for entry in new))
        self.assertFalse(any(entry["impact_plausibility"]["flexible_or_film_like"] for entry in new))

    def test_stage0_stays_unauthorized_until_seed_inventory(self) -> None:
        self.assertEqual(self.stage0["authorization_status"], "not_authorized")
        self.assertFalse((REPO / validator.STANDARD_STAGE0_REL).exists())


PRIVATE_ENV = os.environ.get("V4_PRIVATE_REGISTRY_DIR")


@unittest.skipUnless(PRIVATE_ENV, "set V4_PRIVATE_REGISTRY_DIR for private registry validation")
class PrivateRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert PRIVATE_ENV is not None
        cls.private = Path(PRIVATE_ENV).resolve()

    def test_private_registry_validates(self) -> None:
        summary = validator.validate_private(REPO, self.private)
        self.assertEqual(summary["ontology_count"], 80)
        self.assertEqual(summary["impact_plausibility_pass_count"], 80)
        self.assertFalse(summary["stage0_authorized"])

    def test_private_values_do_not_appear_publicly(self) -> None:
        public_text = "\n".join(
            (REPO / path).read_text(encoding="utf-8")
            for path in (validator.PUBLIC_BANK_REL, validator.PUBLIC_HOLDOUT_REL, validator.PUBLIC_STAGE0_REL)
        )
        salts = validator.load_json(self.private / "salts_private_v1.json")
        secret_keys = (
            "ontology_sampling_salt",
            "split_salt",
            "causal_stage0_selector_salt",
            "causal_evaluation_seed_salt",
            "causal_screening_seed_token",
        )
        self.assertFalse(any(salts[key] in public_text for key in secret_keys))
        holdout = validator.load_json(self.private / "holdout_registry_private24_v1.json")
        self.assertFalse(
            any(
                entry["source_id"] in public_text
                or entry["source_phrase"] in public_text
                or entry["normalized_phrase"] in public_text
                for entry in holdout["entries"]
            )
        )

    def test_all_private_sources_pass_impact_audit(self) -> None:
        matrix = validator.load_json(self.private / "source_impact_matrix_private_v1.json")
        self.assertEqual(len(matrix["rows"]), 80)
        self.assertTrue(all(row["verdict"] == "pass" for row in matrix["rows"]))
        self.assertTrue(all(row["compact_and_rigid"] for row in matrix["rows"]))
        self.assertTrue(all(row["visible_brief_splash_or_ripple_plausible"] for row in matrix["rows"]))
        self.assertFalse(any(row["predominantly_buoyant_or_windborne"] for row in matrix["rows"]))
        self.assertFalse(any(row["flexible_or_film_like"] for row in matrix["rows"]))

    def test_stage0_private_cardinality(self) -> None:
        stage0 = validator.load_json(self.private / "causal_stage0_candidates_private_v1.json")
        self.assertEqual(stage0["candidate_count"], 48)
        self.assertEqual(len(stage0["candidates"]), 48)


if __name__ == "__main__":
    unittest.main()
