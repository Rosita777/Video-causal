from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/build_causal_role_erasure_7mechanism_main_v2.py"
SPEC = importlib.util.spec_from_file_location("build_cre7m_main_v2", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SevenMechanismMainV2BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data_dir = cls.root / "data"
        cls.prompt_dir = cls.root / "prompts"
        cls.salt = bytes(range(32))
        cls.salt_path = cls.root / "eval_seed_salt.bin"
        cls.salt_path.write_bytes(cls.salt)
        exit_code = builder.main(
            [
                "--eval-seed-salt",
                str(cls.salt_path),
                "--output-dir",
                str(cls.data_dir),
                "--prompt-dir",
                str(cls.prompt_dir),
            ]
        )
        assert exit_code == 0
        cls.ontology = json.loads((cls.data_dir / "ontology_registry.json").read_text(encoding="utf-8"))
        cls.targets = read_csv(cls.data_dir / "target_candidates.csv")
        cls.formal = read_csv(cls.data_dir / "formal_cases.csv")
        cls.identification = read_csv(cls.data_dir / "identification_subset.csv")
        cls.runs = read_csv(cls.data_dir / "run_matrix.csv")
        cls.registry = json.loads((cls.data_dir / "build_registry.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_complete_ontology_counts_and_disjointness(self) -> None:
        self.assertEqual(self.ontology["protocol_version"], builder.PROTOCOL_VERSION)
        self.assertEqual([row["mechanism"] for row in self.ontology["mechanisms"]], list(builder.MECHANISM_ORDER))
        self.assertFalse(builder.contains_unresolved_sentinel(self.ontology))
        for ontology in self.ontology["mechanisms"]:
            self.assertEqual(len(ontology["original_training_sources"]), 8)
            self.assertEqual(len(ontology["augmentation_sources"]), 56)
            self.assertEqual(len(ontology["source_bank64"]), 64)
            self.assertEqual(len(ontology["eval_holdout_sources"]), 48)
            self.assertEqual(len(ontology["train_receivers"]), 12)
            self.assertEqual(len(ontology["fresh_eval_receivers"]), 56)
            self.assertEqual(len(ontology["seen_receiver_anchor_ids"]), 8)
            builder.validate_ontology(ontology)
            bank = ontology["source_bank64"]
            holdout = ontology["eval_holdout_sources"]
            self.assertFalse({row["source_id"] for row in bank} & {row["source_id"] for row in holdout})
            self.assertFalse({row["normalized_phrase"] for row in bank} & {row["normalized_phrase"] for row in holdout})
            self.assertFalse({row["semantic_family"] for row in bank} & {row["semantic_family"] for row in holdout})
            if ontology["mechanism"] in builder.COMPACT_MECHANISMS:
                self.assertIn("same frozen compact-rigid", ontology["shared_compact_pool_provenance"])

    def test_target_manifest_has_canonical_schema_and_complete_crossings(self) -> None:
        self.assertEqual(len(self.targets), 1_344)
        self.assertEqual(tuple(self.targets[0]), builder.TARGET_FIELDS)
        self.assertEqual([int(row["global_index"]) for row in self.targets], list(range(1_344)))
        self.assertEqual(len({row["candidate_id"] for row in self.targets}), 1_344)
        self.assertEqual(len({int(row["seed"]) for row in self.targets}), 1_344)
        self.assertEqual(
            Counter(row["target_origin"] for row in self.targets),
            {"water_v1_reuse": 192, "new_generation_v2": 1_152},
        )
        for mechanism in builder.MECHANISM_ORDER:
            rows = [row for row in self.targets if row["mechanism"] == mechanism]
            self.assertEqual(len(rows), 192)
            self.assertEqual(len({row["source_id"] for row in rows}), 8)
            self.assertEqual(len({row["receiver_id"] for row in rows}), 12)
            self.assertEqual(Counter(row["prompt_style"] for row in rows), {"direct": 96, "natural": 96})
            self.assertEqual(
                len({(row["source_id"], row["receiver_id"], row["prompt_style"]) for row in rows}),
                192,
            )

    def test_water_reuse_and_six_new_prompt_shards_are_exact(self) -> None:
        water = [row for row in self.targets if row["mechanism"] == "water_impact"]
        self.assertEqual(Counter(row["historical_screen_status"] for row in water), {"accept": 178, "reject": 14})
        self.assertTrue(all(row["prompt_shard"] == row["prompt_shard_index"] == "" for row in water))
        self.assertTrue(all(row["historical_pair_id"] and row["target_video_path"] for row in water))
        shards = sorted(self.prompt_dir.glob("*.prompts"))
        self.assertEqual(len(shards), 6)
        self.assertNotIn("water_impact", {path.stem.removeprefix("target_candidates_") for path in shards})
        for path in shards:
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 192)
            self.assertTrue(all(line.count(" | ") == 2 for line in lines))
        for row in self.targets:
            if row["mechanism"] != "water_impact":
                self.assertEqual(
                    row["prompt_shard"],
                    f"prompts/{builder.DATASET_VERSION}/target_candidates_{row['mechanism']}.prompts",
                )
        self.assertEqual(
            self.registry["water_reuse"]["generation_manifest_sha256"],
            builder.WATER_TARGET_GENERATION_MANIFEST_SHA256,
        )
        self.assertEqual(self.registry["water_reuse"]["generation_configuration"]["baseline"], "negative_prompt")
        self.assertEqual(self.registry["new_target_generation_configuration"]["baseline"], "negative_prompt")

    def test_target_prompts_exclude_source_trigger_and_footprint(self) -> None:
        for row in self.targets:
            self.assertNotIn(builder.normalize(row["source_object"]), builder.normalize(row["target_prompt"]))
            self.assertFalse(
                builder._contains_registered_lexeme(
                    row["target_prompt"], builder.MECHANISM_SPECS[row["mechanism"]]["target_forbidden"]
                )
            )

    def test_formal_294_balance_seed_uniqueness_and_implicit_rule(self) -> None:
        self.assertEqual(len(self.formal), 294)
        self.assertEqual(tuple(self.formal[0]), builder.FORMAL_FIELDS)
        self.assertEqual(len({row["case_id"] for row in self.formal}), 294)
        formal_seeds = {int(row["seed"]) for row in self.formal}
        target_seeds = {int(row["seed"]) for row in self.targets}
        self.assertEqual(len(formal_seeds), 294)
        self.assertFalse(formal_seeds & target_seeds)
        for mechanism in builder.MECHANISM_ORDER:
            rows = [row for row in self.formal if row["mechanism"] == mechanism]
            causal = [row for row in rows if row["case_kind"] == "causal"]
            specificity = [row for row in rows if row["case_kind"] == "specificity"]
            self.assertEqual((len(causal), len(specificity)), (24, 18))
            self.assertEqual(Counter(row["generalization_group"] for row in causal), {group: 8 for group in builder.GENERALIZATION_GROUPS})
            self.assertEqual(Counter(row["source_membership"] for row in specificity), {membership: 6 for membership in builder.SOURCE_MEMBERSHIPS})
            self.assertEqual(Counter(row["prompt_style"] for row in specificity), {style: 9 for style in builder.PROMPT_STYLES})
            self.assertEqual(Counter(row["specificity_subtype"] for row in specificity), {subtype: 6 for subtype in builder.SPECIFICITY_SUBTYPES})
            for row in causal:
                if row["footprint_lexicalization"] == "implicit":
                    self.assertFalse(
                        builder._contains_registered_lexeme(
                            row["prompt"], builder.MECHANISM_SPECS[mechanism]["footprint_forbidden"]
                        )
                    )

    def test_m6_pairs_share_source_and_receiver(self) -> None:
        for mechanism in builder.MECHANISM_ORDER:
            mechanism_rows = [row for row in self.formal if row["mechanism"] == mechanism]
            pair_ids = {row["m6_pair_id"] for row in mechanism_rows if row["m6_pair_id"]}
            self.assertEqual(len(pair_ids), 6)
            for pair_id in pair_ids:
                pair = [row for row in mechanism_rows if row["m6_pair_id"] == pair_id]
                self.assertEqual(len(pair), 2)
                self.assertEqual({row["case_kind"] for row in pair}, {"causal", "specificity"})
                self.assertEqual(len({row["source_id"] for row in pair}), 1)
                self.assertEqual(len({row["receiver_id"] for row in pair}), 1)

    def test_water_fracture_identification_subset_and_18_runs(self) -> None:
        self.assertEqual(len(self.identification), 48)
        for mechanism in ("water_impact", "brittle_fracture"):
            rows = [row for row in self.identification if row["mechanism"] == mechanism]
            causal = [row for row in rows if row["case_kind"] == "causal"]
            specificity = [row for row in rows if row["case_kind"] == "specificity"]
            self.assertEqual((len(causal), len(specificity)), (12, 12))
            self.assertEqual(Counter(row["generalization_group"] for row in causal), {group: 4 for group in builder.GENERALIZATION_GROUPS})
            self.assertEqual(Counter(row["source_membership"] for row in specificity), {membership: 4 for membership in builder.SOURCE_MEMBERSHIPS})
            self.assertEqual(Counter(row["specificity_subtype"] for row in specificity), {subtype: 4 for subtype in builder.SPECIFICITY_SUBTYPES})
        self.assertEqual(len(self.runs), 18)
        self.assertEqual(
            Counter(row["arm"] for row in self.runs),
            {"matched_control": 7, "V4": 7, "generic_paraphrase": 2, "bystander_token": 2},
        )
        self.assertTrue(all(row["training_seed"] == "26000" for row in self.runs))
        self.assertTrue(all(row["selected_erase_rows_required"] == "178" for row in self.runs))

    def test_eval_salt_is_required_committed_and_not_emitted(self) -> None:
        self.assertEqual(self.registry["eval_seed_salt_sha256"], hashlib.sha256(self.salt).hexdigest())
        self.assertFalse(self.registry["raw_eval_seed_salt_emitted"])
        registry_text = (self.data_dir / "build_registry.json").read_text(encoding="utf-8")
        self.assertNotIn(self.salt.hex(), registry_text)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_salt = root / "bad.bin"
            bad_salt.write_bytes(b"too-short")
            with self.assertRaisesRegex(ValueError, "evaluation seed salt"):
                builder.main(
                    [
                        "--eval-seed-salt",
                        str(bad_salt),
                        "--output-dir",
                        str(root / "data"),
                        "--prompt-dir",
                        str(root / "prompts"),
                    ]
                )

    def test_tampered_frozen_input_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tampered = root / "source_bank.json"
            shutil.copyfile(
                PROJECT_ROOT / "data/water_impact_dynamic_v4/source_bank_public64_registry_v2.json",
                tampered,
            )
            tampered.write_bytes(tampered.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                builder.main(
                    [
                        "--eval-seed-salt",
                        str(self.salt_path),
                        "--water-source-bank",
                        str(tampered),
                        "--output-dir",
                        str(root / "data"),
                        "--prompt-dir",
                        str(root / "prompts"),
                    ]
                )
            self.assertFalse((root / "data").exists())

    def test_duplicate_or_semantically_overlapping_ontology_fails_closed(self) -> None:
        ontology = json.loads(json.dumps(self.ontology["mechanisms"][0]))
        ontology["eval_holdout_sources"][0]["semantic_family"] = ontology["source_bank64"][0]["semantic_family"]
        with self.assertRaisesRegex(ValueError, "semantic-family overlap"):
            builder.validate_ontology(ontology)

    def test_refuses_to_overwrite_a_completed_build(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            builder.main(
                [
                    "--eval-seed-salt",
                    str(self.salt_path),
                    "--output-dir",
                    str(self.data_dir),
                    "--prompt-dir",
                    str(self.prompt_dir),
                ]
            )


if __name__ == "__main__":
    unittest.main()
