from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy  # Keep NumPy resident while temporary dependency stubs are restored.


ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
MAPPING = SCRIPTS / "build_water_impact_dynamic_v4_source_mapping.py"
RUNTIME_BUILDER = SCRIPTS / "build_water_impact_dynamic_v4_runtime_registry.py"
PREPARER = SCRIPTS / "prepare_water_impact_dynamic_v4_prompt_cache.py"
TRAINER = SCRIPTS / "train_wan_waterdrop_lora_v4.py"
LAUNCHER = SCRIPTS / "run_water_impact_dynamic_sft_v4_source_slot.sh"
MANIFEST = ROOT / "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mapping_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("v4_mapping_test", MAPPING)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runtime_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("v4_runtime_test", RUNTIME_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_dependency_modules() -> dict[str, ModuleType]:
    modules = {
        name: ModuleType(name)
        for name in (
            "torch",
            "diffusers",
            "diffusers.utils",
            "peft",
            "transformers",
        )
    }
    modules["torch"].no_grad = lambda: (lambda function: function)
    modules["torch"].bfloat16 = "torch.bfloat16"
    modules["torch"].Tensor = object
    modules["torch"].nn = SimpleNamespace(Module=object)
    modules["torch"].cuda = SimpleNamespace(is_available=lambda: False)
    modules["diffusers"].WanPipeline = object
    modules["diffusers"].WanTransformer3DModel = object
    modules["diffusers.utils"].convert_state_dict_to_diffusers = lambda value: value
    modules["peft"].LoraConfig = object
    modules["peft"].get_peft_model_state_dict = lambda value: value
    modules["transformers"].AutoTokenizer = object
    return modules


def load_preparer_and_trainer():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    fake = fake_dependency_modules()
    with mock.patch.dict(sys.modules, fake):
        prep_spec = importlib.util.spec_from_file_location(
            "prepare_water_impact_dynamic_v4_prompt_cache", PREPARER
        )
        preparer = importlib.util.module_from_spec(prep_spec)
        assert prep_spec.loader is not None
        sys.modules[prep_spec.name] = preparer
        prep_spec.loader.exec_module(preparer)
        train_spec = importlib.util.spec_from_file_location("v4_trainer_test", TRAINER)
        trainer = importlib.util.module_from_spec(train_spec)
        assert train_spec.loader is not None
        train_spec.loader.exec_module(trainer)
    return preparer, trainer


def public_bank_fixture(module) -> dict[str, object]:
    original = []
    for bank_index, (source_id, phrase) in enumerate(module.TRAIN_SOURCES):
        original.append(
            {
                "bank_index": bank_index,
                "source_id": source_id,
                "source_phrase": phrase,
                "normalized_phrase": module.normalize_text(phrase),
                "head_lemma": phrase.split()[-1],
                "membership": "original_training_source",
                "physical_audit_status": module.LEGACY_PHYSICAL_STATUS,
            }
        )
    new = [
        {
            "bank_index": index + 8,
            "source_id": f"fixture_source_{index:02d}",
            "source_phrase": f"one compact amber fixture item {index:02d}",
            "normalized_phrase": f"one compact amber fixture item {index:02d}",
            "head_lemma": f"fixturehead{index:02d}",
            "membership": "new_bank_source",
            "physical_audit_status": module.STRICT_PHYSICAL_STATUS,
            "strata": {
                "color_family": "amber",
                "food_status": "nonfood",
                "material_family": "resin",
                "origin": "manufactured",
                "shape_class": "oval",
                "texture_class": "smooth",
            },
            "impact_plausibility": {
                "compact_and_rigid": True,
                "curator_note": f"fixture object {index:02d} passes public impact review",
                "density_g_cm3": 4.0,
                "dimensions_cm": [10.0, 5.0, 3.0],
                "entity_state": "solid_one_piece",
                "food_or_produce": False,
                "flexible_or_film_like": False,
                "fragile": False,
                "loose_aggregate": False,
                "mass_g": 500,
                "material": "fixture resin composite",
                "natural_drop_entry": True,
                "negative_buoyancy": True,
                "porous": False,
                "powder": False,
                "predominantly_buoyant_or_windborne": False,
                "size_class": "palm_sized_explicit",
                "source_specific_feature": f"fixture-specific feature {index:02d}",
                "verdict": "pass",
                "visually_recognizable": True,
                "visible_brief_splash_or_ripple_plausible": True,
            },
        }
        for index in range(56)
    ]
    entries = original + new
    return {
        "schema": module.BANK_SCHEMA,
        "protocol": module.BANK_SCHEMA,
        "registry": module.BANK_REGISTRY,
        "dataset_version": module.DATASET_VERSION,
        "canonical_json": "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF",
        "supersedes": json.loads(json.dumps(module.EXPECTED_V2_SUPERSEDES)),
        "curation_audit": json.loads(json.dumps(module.EXPECTED_CURATION_AUDIT)),
        "canonical_builder_sha256": module.file_sha256(module.PROMPT_BUILDER_FILE),
        "training_manifest_sha256": module.EXPECTED_MANIFEST_SHA256,
        "counts": {"new_ontology": 56, "original_training": 8, "total": 64},
        "bank_entries_sha256": module.canonical_json_lf_sha256(entries),
        "source_assignment_salt": "1" * 64,
        "source_assignment_algorithm": module.EXPECTED_SOURCE_ASSIGNMENT_ALGORITHM,
        "entries": entries,
    }


def public_holdout_fixture(module, bank=None) -> dict[str, object]:
    if bank is None:
        bank = public_bank_fixture(module)
    new_bank_strata = [
        dict(row["strata"])
        for row in bank["entries"]
        if row["membership"] == "new_bank_source"
    ]

    def aggregate(count: int) -> dict[str, object]:
        return {
            "count": count,
            "origin": {"manufactured": count},
            "food_status": {"nonfood": count},
            "shape_class": {"fixture": count},
            "color_family": {"fixture": count},
            "material_family": {"fixture": count},
            "texture_class": {"fixture": count},
            "origin_x_food_status": {"manufactured:nonfood": count},
        }

    return {
        "schema": module.BANK_SCHEMA,
        "protocol": module.BANK_SCHEMA,
        "registry": module.HOLDOUT_REGISTRY,
        "dataset_version": module.DATASET_VERSION,
        "canonical_json": "UTF-8 JSON, sorted keys, separators comma/colon, trailing LF",
        "supersedes": json.loads(json.dumps(module.EXPECTED_V2_SUPERSEDES)),
        "curation_audit": json.loads(json.dumps(module.EXPECTED_CURATION_AUDIT)),
        "holdout_count": 24,
        "holdout_registry_file_sha256": "3" * 64,
        "split_rule": module.EXPECTED_HOLDOUT_SPLIT_RULE,
        "split_salt_commitment_sha256": "4" * 64,
        "aggregate_strata": {
            "curated_new80": aggregate(80),
            "new_bank56": module.aggregate_strata(new_bank_strata),
            "private_holdout24": aggregate(24),
        },
        "cross_role_checks": {
            "ambiguous_size_language_count": 0,
            "event_literal_or_subword_risk_count": 0,
            "historical_source_near_duplicate_count": 0,
            "historical_source_semantic_equivalence_count": 0,
            "impact_plausibility_failure_count": 0,
            "impact_plausibility_pass_count": 80,
            "matrix_scope": {
                "complete_receivers": 84,
                "historical_receivers": 52,
                "historical_sources": 14,
                "new_receivers": 32,
                "new_sources": 80,
            },
            "new_receiver_historical_near_duplicate_count": 0,
            "new_receiver_historical_semantic_equivalence_count": 0,
            "normalized_head_overlap_count": 0,
            "normalized_phrase_overlap_count": 0,
            "prohibited_physical_category_count": 0,
            "source_receiver_near_duplicate_count": 0,
            "source_receiver_semantic_equivalence_count": 0,
            "source_specific_note_unique_count": 80,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def authorization_fixture(trainer, root: Path) -> tuple[Path, dict[str, object]]:
    mapping = load_mapping_module()
    bank = public_bank_fixture(mapping)
    bank_path = root / trainer.EXPECTED_BANK_REGISTRY
    write_json(bank_path, bank)
    holdout = public_holdout_fixture(mapping, bank)
    holdout_path = root / trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT
    write_json(holdout_path, holdout)
    runtime = load_runtime_module()
    runtime_path = root / trainer.EXPECTED_RUNTIME_REGISTRY
    write_json(runtime_path, runtime.expected_registry_payload())
    runtime_record = {
        "path": str(trainer.EXPECTED_RUNTIME_REGISTRY),
        "sha256": sha256(runtime_path),
    }
    for index, relative in enumerate(trainer.CODE_ARTIFACT_PATHS.values()):
        artifact = root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"frozen-code-{index}\n".encode("utf-8"))
    code_registry = {
        "protocol": trainer.TRAINING_CODE_REGISTRY_PROTOCOL,
        "status": "frozen",
        "runtime_registry": runtime_record,
        "artifacts": {
            name: {"path": str(relative), "sha256": sha256(root / relative)}
            for name, relative in trainer.CODE_ARTIFACT_PATHS.items()
        },
    }
    code_path = root / trainer.EXPECTED_TRAINING_CODE_REGISTRY
    write_json(code_path, code_registry)

    refs: dict[str, dict[str, str]] = {
        "source_bank_registry": {
            "path": str(trainer.EXPECTED_BANK_REGISTRY),
            "sha256": sha256(bank_path),
        },
        "holdout_public_commitment": {
            "path": str(trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT),
            "sha256": sha256(holdout_path),
        },
    }
    for dataset in ("causal", "specificity"):
        stage0_name = f"{dataset}_stage0"
        stage1_name = f"{dataset}_stage1"
        stage0_path = root / trainer.AUTHORIZATION_REF_PATHS[stage0_name]
        stage0 = {
            "protocol": trainer.COMMITMENT_PROTOCOL,
            "dataset": dataset,
            "dataset_version": trainer.DATASET_VERSION,
            "stage": 0,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256(
                        f"{dataset}:0:{name}".encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": 1,
                    "row_count": trainer.EXPECTED_COMMITMENT_ROW_COUNTS.get(
                        (dataset, 0, name)
                    ),
                }
                for name in trainer.STAGE_ARTIFACTS[(dataset, 0)]
            },
        }
        if dataset == "causal":
            stage0["artifacts"]["source_bank_registry_64"]["sha256"] = sha256(
                bank_path
            )
            stage0["artifacts"]["holdout_registry_24"]["sha256"] = holdout[
                "holdout_registry_file_sha256"
            ]
        write_json(stage0_path, stage0)
        stage1_path = root / trainer.AUTHORIZATION_REF_PATHS[stage1_name]
        stage1 = {
            "protocol": trainer.COMMITMENT_PROTOCOL,
            "dataset": dataset,
            "dataset_version": trainer.DATASET_VERSION,
            "stage": 1,
            "status": "committed",
            "sealed_final36_status": "unopened",
            "stage0_registry_sha256": sha256(stage0_path),
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256(
                        f"{dataset}:1:{name}".encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": 1,
                    "row_count": trainer.EXPECTED_COMMITMENT_ROW_COUNTS.get(
                        (dataset, 1, name)
                    ),
                }
                for name in trainer.STAGE_ARTIFACTS[(dataset, 1)]
            },
        }
        write_json(stage1_path, stage1)
        refs[stage0_name] = {
            "path": trainer.AUTHORIZATION_REF_PATHS[stage0_name],
            "sha256": sha256(stage0_path),
        }
        refs[stage1_name] = {
            "path": trainer.AUTHORIZATION_REF_PATHS[stage1_name],
            "sha256": sha256(stage1_path),
        }

    gate_path = root / trainer.AUTHORIZATION_REF_PATHS["gate_registry"]
    gate_spec = json.loads(json.dumps(trainer.EXPECTED_GATE_SPEC))
    write_json(
        gate_path,
        {
            "protocol": trainer.GATE_REGISTRY_PROTOCOL,
            "status": "frozen",
            "dataset_version": trainer.DATASET_VERSION,
            "sealed_final36_status": "unopened",
            "gate_spec": gate_spec,
            "gate_spec_sha256": trainer.canonical_json_sha256(gate_spec),
            "scorer_sha256": code_registry["artifacts"]["eval_scorer"]["sha256"],
        },
    )
    refs["gate_registry"] = {
        "path": trainer.AUTHORIZATION_REF_PATHS["gate_registry"],
        "sha256": sha256(gate_path),
    }
    refs["runtime_registry"] = runtime_record
    refs["code_registry"] = {
        "path": trainer.AUTHORIZATION_REF_PATHS["code_registry"],
        "sha256": sha256(code_path),
    }
    payload: dict[str, object] = {
        "protocol": trainer.AUTHORIZATION_PROTOCOL,
        "status": "authorized",
        "dataset_version": trainer.DATASET_VERSION,
        "sealed_final36_status": "unopened",
        **refs,
    }
    authorization_path = root / trainer.EXPECTED_AUTHORIZATION
    write_json(authorization_path, payload)
    return authorization_path, payload


class SourceMappingTest(unittest.TestCase):
    def test_public_stage0_schema_and_three_way_curation_audit_are_exact(self) -> None:
        module = load_mapping_module()
        bank = public_bank_fixture(module)
        holdout = public_holdout_fixture(module, bank)
        stage0 = json.loads(
            (
                ROOT
                / "data/water_impact_dynamic_v4/causal_stage0_public_commitment_v2.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "causal_stage0_public_commitment_v2.json"
            write_json(path, stage0)
            validated, _ = module.validate_public_stage0_commitment(
                path,
                expected_sha256=sha256(path),
                bank_registry=bank,
                holdout_commitment=holdout,
            )
            self.assertEqual(validated, stage0)
            stage0["curation_audit"]["status"] = "self-consistent-but-wrong"
            write_json(path, stage0)
            with self.assertRaisesRegex(ValueError, "curation audits differ"):
                module.validate_public_stage0_commitment(
                    path,
                    expected_sha256=sha256(path),
                    bank_registry=bank,
                    holdout_commitment=holdout,
                )

    def test_v3b_schedule_and_balanced_deranged_mapping(self) -> None:
        module = load_mapping_module()
        rows = module.load_frozen_rows(
            MANIFEST, expected_sha256=module.EXPECTED_MANIFEST_SHA256
        )
        schedule = module.balanced_v3b_schedule(rows)
        self.assertEqual(
            module.sample_order_sha256(rows, schedule),
            module.EXPECTED_SAMPLE_ORDER_SHA256,
        )
        self.assertEqual(
            [rows[index]["training_role"] for index in schedule],
            ["erase", "preserve"] * 100,
        )

        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "public_bank.json"
            registry_path.write_text(
                json.dumps(public_bank_fixture(module)), encoding="utf-8"
            )
            registry, registry_hash = module.validate_public_bank_registry(
                registry_path, expected_sha256=sha256(registry_path)
            )
            holdout_path = Path(directory) / "holdout_commitment.json"
            holdout_path.write_text(
                json.dumps(public_holdout_fixture(module, registry)), encoding="utf-8"
            )
            _, holdout_hash = module.validate_public_holdout_commitment(
                holdout_path,
                expected_sha256=sha256(holdout_path),
                bank_registry=registry,
            )
            first = module.build_mapping(
                rows,
                registry,
                bank_registry_sha256=registry_hash,
                holdout_commitment_path=str(holdout_path),
                holdout_commitment_sha256=holdout_hash,
                manifest_sha256=module.EXPECTED_MANIFEST_SHA256,
            )
            second = module.build_mapping(
                rows,
                registry,
                bank_registry_sha256=registry_hash,
                holdout_commitment_path=str(holdout_path),
                holdout_commitment_sha256=holdout_hash,
                manifest_sha256=module.EXPECTED_MANIFEST_SHA256,
            )

        self.assertEqual(first, second)
        self.assertEqual(first["active_source_count_min"], 1)
        self.assertEqual(first["active_source_count_max"], 2)
        self.assertEqual(len(first["active_source_counts"]), 64)
        self.assertEqual(sum(first["active_source_counts"].values()), 100)
        self.assertEqual(len(first["mapping"]), 178)
        self.assertTrue(
            all(
                row["assigned_source_phrase"] != row["original_source_phrase"]
                for row in first["mapping"]
            )
        )
        active = sorted(
            (row for row in first["mapping"] if row["active_erase_ordinal"] is not None),
            key=lambda row: row["active_erase_ordinal"],
        )
        self.assertEqual(module.canonical_json_sha256(active), first["active100_mapping_sha256"])
        self.assertEqual(
            module.canonical_json_sha256(first["mapping"]),
            first["full178_mapping_sha256"],
        )

    def test_public_bank_rejects_private_holdout_payload(self) -> None:
        module = load_mapping_module()
        payload = public_bank_fixture(module)
        payload["holdout_sources"] = ["must-not-be-visible"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leak.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden private field"):
                module.validate_public_bank_registry(path, expected_sha256=sha256(path))

    def test_public_bank_rejects_assignment_algorithm_tamper(self) -> None:
        module = load_mapping_module()
        payload = public_bank_fixture(module)
        payload["source_assignment_algorithm"] = json.loads(
            json.dumps(module.EXPECTED_SOURCE_ASSIGNMENT_ALGORITHM)
        )
        payload["source_assignment_algorithm"]["permutation"]["application"] += " altered"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered_algorithm.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_assignment_algorithm"):
                module.validate_public_bank_registry(path, expected_sha256=sha256(path))

    def test_public_v2_supersedes_and_impact_schema_are_fail_closed(self) -> None:
        module = load_mapping_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "bank.json"
            bank = public_bank_fixture(module)
            bank["supersedes"]["reason_code"] = "altered"
            write_json(bank_path, bank)
            with self.assertRaisesRegex(ValueError, "supersedes"):
                module.validate_public_bank_registry(
                    bank_path, expected_sha256=sha256(bank_path)
                )

            bank = public_bank_fixture(module)
            bank["entries"][8]["impact_plausibility"]["negative_buoyancy"] = False
            bank["bank_entries_sha256"] = module.canonical_json_lf_sha256(
                bank["entries"]
            )
            write_json(bank_path, bank)
            with self.assertRaisesRegex(ValueError, "failed public impact audit"):
                module.validate_public_bank_registry(
                    bank_path, expected_sha256=sha256(bank_path)
                )

            bank = public_bank_fixture(module)
            bank["entries"][8]["impact_plausibility"]["mass_g"] = 1200
            bank["bank_entries_sha256"] = module.canonical_json_lf_sha256(
                bank["entries"]
            )
            write_json(bank_path, bank)
            with self.assertRaisesRegex(ValueError, "failed public impact audit"):
                module.validate_public_bank_registry(
                    bank_path, expected_sha256=sha256(bank_path)
                )

            holdout_path = root / "holdout.json"
            valid_bank = public_bank_fixture(module)
            holdout = public_holdout_fixture(module, valid_bank)
            holdout["supersedes"]["aggregate_audit"][
                "stage0_global_constraint_feasible"
            ] = True
            write_json(holdout_path, holdout)
            with self.assertRaisesRegex(ValueError, "supersedes"):
                module.validate_public_holdout_commitment(
                    holdout_path,
                    expected_sha256=sha256(holdout_path),
                    bank_registry=valid_bank,
                )

    def test_mapping_registry_write_is_atomic_and_exclusive(self) -> None:
        module = load_mapping_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            module.atomic_write_new_json(path, {"status": "frozen"})
            self.assertEqual(json.loads(path.read_text()), {"status": "frozen"})
            with self.assertRaises(FileExistsError):
                module.atomic_write_new_json(path, {"status": "changed"})

    def test_mapping_loader_recomputes_unique_bank_salt_manifest_assignment(self) -> None:
        module = load_mapping_module()
        preparer, _ = load_preparer_and_trainer()
        rows = module.load_frozen_rows(
            MANIFEST, expected_sha256=module.EXPECTED_MANIFEST_SHA256
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "bank.json"
            bank_path.write_text(json.dumps(public_bank_fixture(module)), encoding="utf-8")
            bank, bank_hash = module.validate_public_bank_registry(
                bank_path, expected_sha256=sha256(bank_path)
            )
            holdout_path = root / "holdout.json"
            holdout_path.write_text(
                json.dumps(public_holdout_fixture(module)), encoding="utf-8"
            )
            holdout_hash = sha256(holdout_path)
            mapping = module.build_mapping(
                rows,
                bank,
                bank_registry_sha256=bank_hash,
                holdout_commitment_path=str(holdout_path),
                holdout_commitment_sha256=holdout_hash,
                manifest_sha256=module.EXPECTED_MANIFEST_SHA256,
            )
            mapping_path = root / "mapping.json"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            preparer.load_mapping_registry(
                mapping_path,
                rows,
                expected_sha256=sha256(mapping_path),
                expected_bank_sha256=bank_hash,
                expected_holdout_commitment_path=str(holdout_path),
                expected_holdout_commitment_sha256=holdout_hash,
                bank_registry=bank,
            )

            active = [
                record
                for record in mapping["mapping"]
                if record["active_erase_ordinal"] is not None
            ]
            left = right = None
            for first_index, first in enumerate(active):
                for second in active[first_index + 1 :]:
                    if (
                        first["assigned_source_phrase"] != second["original_source_phrase"]
                        and second["assigned_source_phrase"] != first["original_source_phrase"]
                    ):
                        left, right = first, second
                        break
                if left is not None:
                    break
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            assert left is not None and right is not None
            assigned_fields = (
                "assigned_source_id",
                "assigned_source_phrase",
                "assigned_source_membership",
            )
            left_values = {field: left[field] for field in assigned_fields}
            for field in assigned_fields:
                left[field] = right[field]
                right[field] = left_values[field]
            for record in (left, right):
                record["augmented_factual_prompt"] = module.factual_prompt(
                    record["assigned_source_phrase"],
                    record["receiver"],
                    record["prompt_variant"],
                )
            active_canonical = sorted(
                active, key=lambda record: record["active_erase_ordinal"]
            )
            mapping["active100_mapping_sha256"] = module.canonical_json_sha256(
                active_canonical
            )
            mapping["full178_mapping_sha256"] = module.canonical_json_sha256(
                mapping["mapping"]
            )
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique reconstruction"):
                preparer.load_mapping_registry(
                    mapping_path,
                    rows,
                    expected_sha256=sha256(mapping_path),
                    expected_bank_sha256=bank_hash,
                    expected_holdout_commitment_path=str(holdout_path),
                    expected_holdout_commitment_sha256=holdout_hash,
                    bank_registry=bank,
                )


class PromptPreflightAndTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preparer, cls.trainer = load_preparer_and_trainer()

    def test_runtime_registry_schema_versions_and_exclusive_write(self) -> None:
        runtime = load_runtime_module()
        payload = runtime.expected_registry_payload()
        self.assertEqual(payload["python"]["version"], "3.11.15")
        self.assertEqual(payload["packages"]["torch"], "2.6.0")
        self.assertEqual(payload["torch"]["module_version"], "2.6.0+cu124")
        self.assertEqual(payload["cuda"]["torch_cuda_version"], "12.4")
        self.assertEqual(payload["cuda"]["cudnn_version"], 90100)
        self.assertEqual(
            payload["sys_prefix_policy"],
            "realpath(sys.prefix)==realpath(runtime_root)",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            runtime.atomic_write_new_json(path, payload)
            self.assertEqual(
                runtime.validate_runtime_registry(
                    path,
                    sha256(path),
                    project_root=Path(directory),
                    verify_current_runtime=False,
                ),
                payload,
            )
            with self.assertRaises(FileExistsError):
                runtime.atomic_write_new_json(path, payload)
            payload["packages"]["torch"] = "tampered"
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "frozen runtime contract"):
                runtime.validate_runtime_registry(
                    path,
                    sha256(path),
                    project_root=Path(directory),
                    verify_current_runtime=False,
                )

    def test_runtime_registry_is_bound_across_the_training_chain(self) -> None:
        preparer = PREPARER.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('"runtime_registry_sha256": args.runtime_registry_sha256', preparer)
        self.assertIn("validate_runtime_registry(", preparer)
        self.assertIn(
            '"runtime_registry": "data/water_impact_dynamic_v4/v4_runtime_registry_v2.json"',
            trainer,
        )
        self.assertIn(
            'set(payload) != {"protocol", "status", "runtime_registry", "artifacts"}',
            trainer,
        )
        self.assertGreaterEqual(
            trainer.count('"runtime_registry_sha256": args.runtime_registry_sha256'),
            4,
        )
        self.assertIn("EXPECTED_RUNTIME_REGISTRY_SHA256", launcher)
        self.assertIn('--runtime-registry-sha256 "$EXPECTED_RUNTIME_REGISTRY_SHA256"', launcher)

    def test_runtime_identity_accepts_registered_environment_symlinks(self) -> None:
        runtime = load_runtime_module()
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            project_root = temporary / "workspace"
            actual_runtime = temporary / "actual-runtime"
            (project_root / "models").mkdir(parents=True)
            (actual_runtime / "bin").mkdir(parents=True)
            (project_root / "models/.wan-runtime").symlink_to(
                actual_runtime, target_is_directory=True
            )
            base_python = temporary / "python3.11"
            base_python.write_bytes(b"fixture interpreter")
            (actual_runtime / "bin/python").symlink_to(base_python)
            alternate_python = actual_runtime / "bin/python3.11"
            alternate_python.symlink_to(base_python)

            fake_torch = ModuleType("torch")
            fake_torch.__version__ = runtime.EXPECTED_TORCH_MODULE_VERSION
            fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
            fake_torch.version = SimpleNamespace(
                cuda=runtime.EXPECTED_TORCH_CUDA_VERSION
            )
            fake_torch.backends = SimpleNamespace(
                cudnn=SimpleNamespace(version=lambda: runtime.EXPECTED_CUDNN_VERSION)
            )
            with mock.patch.object(runtime.sys, "prefix", str(actual_runtime)), mock.patch.object(
                runtime.sys, "executable", str(alternate_python)
            ), mock.patch.object(
                runtime.platform, "python_implementation", return_value="CPython"
            ), mock.patch.object(
                runtime.platform,
                "python_version",
                return_value=runtime.EXPECTED_PYTHON_VERSION,
            ), mock.patch.object(
                runtime.importlib.metadata,
                "version",
                side_effect=lambda name: runtime.EXPECTED_PACKAGE_VERSIONS[name],
            ), mock.patch.dict(sys.modules, {"torch": fake_torch}):
                runtime.validate_current_runtime(project_root)

    def test_model_inventory_hashes_all_non_cache_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            files = {
                "model_index.json": b"model",
                "transformer/config.json": b"transformer",
                "transformer/weights.safetensors": b"weights",
                "text_encoder/config.json": b"text",
                "tokenizer/tokenizer_config.json": b"tokenizer",
            }
            for relative, data in files.items():
                path = model / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            ignored = model / ".cache/huggingface/metadata"
            ignored.parent.mkdir(parents=True)
            ignored.write_bytes(b"ignored")
            for suffix in (".tmp", ".lock", ".incomplete", "~"):
                (model / f"ignored{suffix}").write_bytes(b"ignored temporary")
            first = self.preparer.compute_model_content_inventory(model)
            ignored.write_bytes(b"changed but ignored")
            second = self.preparer.compute_model_content_inventory(model)
            self.assertEqual(first, second)
            self.assertEqual([row["path"] for row in first["files"]], sorted(files))
            self.assertEqual(
                set(first),
                {"algorithm", "root", "excluded", "file_count", "sha256", "files"},
            )
            self.assertEqual(
                self.preparer.MODEL_INVENTORY_ALGORITHM,
                "sha256_ordered_relative_path_nul_bytes_newline_with_file_records_v1",
            )
            self.assertEqual(
                self.preparer.EXPECTED_MODEL_CONTENT_INVENTORY_SHA256,
                "0a8566eeab29dfbc04303167ce1904b65b964dd1579959645d1f93e19ba15ddf",
            )
            (model / "transformer/weights.safetensors").write_bytes(b"WEIGHTS")
            third = self.preparer.compute_model_content_inventory(model)
            self.assertNotEqual(first, third)

    def test_direct_trainer_pins_canonical_public_bank_and_holdout_hashes(self) -> None:
        valid = SimpleNamespace(
            source_bank_registry_sha256=(
                self.trainer.EXPECTED_BANK_REGISTRY_SHA256
            ),
            holdout_public_commitment_sha256=(
                self.trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256
            ),
        )
        self.trainer.validate_canonical_public_hashes(valid)

        wrong_bank = SimpleNamespace(
            source_bank_registry_sha256="a" * 64,
            holdout_public_commitment_sha256=(
                self.trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256
            ),
        )
        with self.assertRaisesRegex(ValueError, "canonical public v2"):
            self.trainer.validate_canonical_public_hashes(wrong_bank)

        wrong_holdout = SimpleNamespace(
            source_bank_registry_sha256=(
                self.trainer.EXPECTED_BANK_REGISTRY_SHA256
            ),
            holdout_public_commitment_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "canonical public v2"):
            self.trainer.validate_canonical_public_hashes(wrong_holdout)

        with mock.patch.object(self.trainer, "parse_args", return_value=wrong_bank), \
             mock.patch.object(self.trainer, "validate_output_reservation") as reserve:
            with self.assertRaisesRegex(ValueError, "canonical public v2"):
                self.trainer.main()
            reserve.assert_not_called()

    def test_training_authorization_rehashes_public_refs_and_frozen_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            result = self.trainer.validate_training_authorization(
                path, sha256(path), project_root=root, verify_current_runtime=False
            )
            self.assertEqual(result, payload)
            payload["extra"] = {"path": "x", "sha256": "b" * 64}
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "exact fields"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

    def test_v2_dataset_identity_and_paths_fail_closed(self) -> None:
        module = load_mapping_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "bank_v2.json"
            bank = public_bank_fixture(module)
            bank["dataset_version"] = "v4_dev72_v1"
            write_json(bank_path, bank)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                module.validate_public_bank_registry(
                    bank_path, expected_sha256=sha256(bank_path)
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            payload["dataset_version"] = "v4_dev72_v1"
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "dataset version"):
                self.trainer.validate_training_authorization(
                    path,
                    sha256(path),
                    project_root=root,
                    verify_current_runtime=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            stage0_path = root / self.trainer.AUTHORIZATION_REF_PATHS["causal_stage0"]
            stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
            stage0["dataset_version"] = "v4_dev72_v1"
            write_json(stage0_path, stage0)
            payload["causal_stage0"]["sha256"] = sha256(stage0_path)
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                self.trainer.validate_training_authorization(
                    path,
                    sha256(path),
                    project_root=root,
                    verify_current_runtime=False,
                )

        v2_paths = (
            self.trainer.EXPECTED_BANK_REGISTRY,
            self.trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT,
            self.trainer.EXPECTED_MAPPING_REGISTRY,
            self.trainer.EXPECTED_PROMPT_SIDECAR_DIR,
            self.trainer.EXPECTED_PREFLIGHT,
            self.trainer.EXPECTED_AUTHORIZATION,
            self.trainer.EXPECTED_TRAINING_CODE_REGISTRY,
            self.trainer.EXPECTED_RUNTIME_REGISTRY,
            self.trainer.EXPECTED_OUTPUT_DIR,
        )
        self.assertTrue(all("v2" in str(path) for path in v2_paths))
        self.assertTrue(
            all("v2" in path for path in self.trainer.AUTHORIZATION_REF_PATHS.values())
        )

    def test_training_authorization_rejects_missing_or_tampered_public_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = authorization_fixture(self.trainer, root)
            causal0 = root / self.trainer.AUTHORIZATION_REF_PATHS["causal_stage0"]
            causal0.write_bytes(causal0.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "causal_stage0 byte hash mismatch"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = authorization_fixture(self.trainer, root)
            specificity1 = root / self.trainer.AUTHORIZATION_REF_PATHS["specificity_stage1"]
            specificity1.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "specificity_stage1"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

    def test_training_code_registry_rejects_code_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            trainer_copy = root / self.trainer.CODE_ARTIFACT_PATHS["trainer"]
            trainer_copy.write_bytes(trainer_copy.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(ValueError, "trainer byte hash mismatch"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )
            self.assertEqual(payload["sealed_final36_status"], "unopened")

    def test_runtime_registry_tamper_cannot_self_rebind_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization_path, authorization = authorization_fixture(
                self.trainer, root
            )
            runtime_path = root / self.trainer.EXPECTED_RUNTIME_REGISTRY
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime["packages"]["numpy"] = "tampered"
            write_json(runtime_path, runtime)
            authorization["runtime_registry"]["sha256"] = sha256(runtime_path)
            code_path = root / self.trainer.EXPECTED_TRAINING_CODE_REGISTRY
            code = json.loads(code_path.read_text(encoding="utf-8"))
            code["runtime_registry"] = dict(authorization["runtime_registry"])
            write_json(code_path, code)
            authorization["code_registry"]["sha256"] = sha256(code_path)
            write_json(authorization_path, authorization)
            with self.assertRaisesRegex(ValueError, "frozen runtime contract"):
                self.trainer.validate_training_authorization(
                    authorization_path,
                    sha256(authorization_path),
                    project_root=root,
                    verify_current_runtime=False,
                )

    def test_training_authorization_requires_exact_stage_artifacts_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            causal0_path = root / self.trainer.AUTHORIZATION_REF_PATHS["causal_stage0"]
            causal0 = json.loads(causal0_path.read_text(encoding="utf-8"))
            causal0["artifacts"].pop("forbidden_seed_inventory")
            write_json(causal0_path, causal0)
            payload["causal_stage0"]["sha256"] = sha256(causal0_path)
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "artifact inventory is not exact"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = authorization_fixture(self.trainer, root)
            specificity1_path = root / self.trainer.AUTHORIZATION_REF_PATHS[
                "specificity_stage1"
            ]
            specificity1 = json.loads(specificity1_path.read_text(encoding="utf-8"))
            specificity1["artifacts"]["screening_candidate_binding"]["row_count"] = 35
            write_json(specificity1_path, specificity1)
            payload["specificity_stage1"]["sha256"] = sha256(specificity1_path)
            write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "row count must be 36"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

    def test_training_authorization_rejects_gate_placeholder_and_wrong_exact_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, authorization = authorization_fixture(self.trainer, root)
            gate_path = root / self.trainer.AUTHORIZATION_REF_PATHS["gate_registry"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_spec"] = {"decision": "TO_BE_FROZEN_GATE_SPEC"}
            gate["gate_spec_sha256"] = self.trainer.canonical_json_sha256(
                gate["gate_spec"]
            )
            write_json(gate_path, gate)
            authorization["gate_registry"]["sha256"] = sha256(gate_path)
            write_json(path, authorization)
            with self.assertRaisesRegex(ValueError, "contains a placeholder"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, authorization = authorization_fixture(self.trainer, root)
            gate_path = root / self.trainer.AUTHORIZATION_REF_PATHS["gate_registry"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_spec"] = {"decision": "self-consistent-but-wrong"}
            gate["gate_spec_sha256"] = self.trainer.canonical_json_sha256(
                gate["gate_spec"]
            )
            write_json(gate_path, gate)
            authorization["gate_registry"]["sha256"] = sha256(gate_path)
            write_json(path, authorization)
            with self.assertRaisesRegex(ValueError, "differs from canonical protocol"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

    def test_training_authorization_rejects_public_bank_stage_mix_and_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, authorization = authorization_fixture(self.trainer, root)
            bank_path = root / self.trainer.EXPECTED_BANK_REGISTRY
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            bank["source_assignment_salt"] = "2" * 64
            write_json(bank_path, bank)
            authorization["source_bank_registry"]["sha256"] = sha256(bank_path)
            write_json(path, authorization)
            with self.assertRaisesRegex(ValueError, "source bank differs"):
                self.trainer.validate_training_authorization(
                    path, sha256(path), project_root=root, verify_current_runtime=False
                )

    def test_sidecar_exact_metadata_and_augmented_reencode_reject_tamper(self) -> None:
        scalar_expected = {
            "protocol": self.preparer.CACHE_PROTOCOL,
            "dataset_version": self.preparer.DATASET_VERSION,
            "scene_id": "fixture_scene",
        }
        payload = {
            **scalar_expected,
            "augmented_prompt_embeds": object(),
            "augmented_prompt_embeds_sha256": "a" * 64,
            "registered_token_length": 10,
            "extra": "self-signed-extra",
        }
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            self.preparer.validate_prompt_sidecar_payload(
                payload, scalar_expected=scalar_expected, path=Path("fixture.pt")
            )
        payload.pop("extra")
        payload["scene_id"] = "wrong_scene"
        with self.assertRaisesRegex(ValueError, "drifted scene_id"):
            self.preparer.validate_prompt_sidecar_payload(
                payload, scalar_expected=scalar_expected, path=Path("fixture.pt")
            )
        payload["scene_id"] = "TO_BE_FROZEN_SCENE"
        placeholder_expected = dict(scalar_expected)
        placeholder_expected["scene_id"] = payload["scene_id"]
        with self.assertRaisesRegex(ValueError, "contains a placeholder"):
            self.preparer.validate_prompt_sidecar_payload(
                payload,
                scalar_expected=placeholder_expected,
                path=Path("fixture.pt"),
            )

        rows = [
            {"training_role": "erase", "scene_id": "scene_0"},
            {"training_role": "erase", "scene_id": "scene_1"},
        ]
        mapping = {
            0: {
                "augmented_factual_prompt": "prompt zero",
                "assigned_source_id": "source_0",
            },
            1: {
                "augmented_factual_prompt": "prompt one",
                "assigned_source_id": "source_1",
            },
        }
        sidecars = {0: Path("zero.pt"), 1: Path("one.pt")}
        # Model a malicious swap whose per-file embedding hashes were re-signed;
        # the independent fresh encoder comparison must ignore that self-report.
        stored = {
            Path("zero.pt"): {
                "augmented_prompt_embeds": "embedding-one",
                "augmented_prompt_embeds_sha256": "resigned-one",
            },
            Path("one.pt"): {
                "augmented_prompt_embeds": "embedding-zero",
                "augmented_prompt_embeds_sha256": "resigned-zero",
            },
        }
        with mock.patch.object(
            self.preparer.torch,
            "load",
            side_effect=lambda path, **_: stored[path],
            create=True,
        ), mock.patch.object(
            self.preparer.torch,
            "equal",
            side_effect=lambda left, right: left == right,
            create=True,
        ):
            with self.assertRaisesRegex(ValueError, "fresh augmented prompt encoding"):
                self.preparer._validate_augmented_reencodes(
                    rows,
                    mapping,
                    sidecars,
                    {
                        "prompt zero": "embedding-zero",
                        "prompt one": "embedding-one",
                    },
                )

    def test_step200_nonfinite_grad_and_final_lora_are_ineligible(self) -> None:
        class Scalar:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def __float__(self):
                return float(self.value)

        original_nn = self.trainer.torch.nn
        try:
            self.trainer.torch.nn = SimpleNamespace(
                utils=SimpleNamespace(
                    clip_grad_norm_=lambda _parameters, _max_norm: Scalar(float("inf"))
                )
            )
            with self.assertRaisesRegex(FloatingPointError, "step 200"):
                self.trainer.clip_and_validate_grad_norm(
                    [object()], max_norm=1.0, step=200
                )
            self.trainer.torch.nn.utils.clip_grad_norm_ = (
                lambda _parameters, _max_norm: Scalar(float("nan"))
            )
            with self.assertRaisesRegex(FloatingPointError, "step 200"):
                self.trainer.clip_and_validate_grad_norm(
                    [object()], max_norm=1.0, step=200
                )
        finally:
            self.trainer.torch.nn = original_nn

        class FakeTensor:
            def __init__(self, finite_count, *, requires_grad=True):
                self.finite_count = finite_count
                self.requires_grad = requires_grad

            def numel(self):
                return 2

            def detach(self):
                return self

        class FiniteMask:
            def __init__(self, count):
                self.count = count

            def sum(self):
                return self

            def item(self):
                return self.count

        transformer = SimpleNamespace(
            named_parameters=lambda: [("lora_trainable", FakeTensor(2))]
        )
        with mock.patch.object(
            self.trainer.torch, "Tensor", FakeTensor
        ), mock.patch.object(
            self.trainer.torch,
            "isfinite",
            side_effect=lambda tensor: FiniteMask(tensor.finite_count),
            create=True,
        ), mock.patch.object(
            self.trainer,
            "get_peft_model_state_dict",
            return_value={"lora_saved": FakeTensor(1)},
        ):
            with self.assertRaisesRegex(FloatingPointError, "non-finite"):
                self.trainer.validate_final_lora_finiteness(transformer)
    def test_scale_gate_uses_constant_v3b_weight(self) -> None:
        metrics = self.trainer.scale_metrics([0.01, 0.04], 4.0)
        self.assertAlmostEqual(metrics["mean_weighted_output_grad_ratio"], 0.6)
        self.assertAlmostEqual(metrics["max_weighted_output_grad_ratio"], 0.8)

    def test_output_reservation_rejects_reuse_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / ".run_reservation").write_text("reserved\n")
            (output / "run_registration_v2.json").write_text("{}\n")
            args = SimpleNamespace(output_dir=output)
            self.trainer.validate_output_reservation(args)
            (output / "old-checkpoint").mkdir()
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                self.trainer.validate_output_reservation(args)

    def test_run_registration_exactly_binds_code_registry_and_frozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code_registry_path = root / "code_registry.json"
            code_registry = {
                "protocol": self.trainer.TRAINING_CODE_REGISTRY_PROTOCOL,
                "status": "frozen",
                "runtime_registry": {
                    "path": str(self.trainer.EXPECTED_RUNTIME_REGISTRY),
                    "sha256": "8" * 64,
                },
                "artifacts": {
                    name: {
                        "path": str(path),
                        "sha256": sha256(ROOT / path),
                    }
                    for name, path in self.trainer.CODE_ARTIFACT_PATHS.items()
                },
            }
            write_json(code_registry_path, code_registry)
            args = SimpleNamespace(
                run_registration=root / "run_registration_v2.json",
                run_registration_sha256="",
                source_bank_registry_sha256="1" * 64,
                holdout_public_commitment_sha256="2" * 64,
                source_mapping_registry_sha256="3" * 64,
                prompt_sidecar_inventory_sha256="4" * 64,
                prompt_sidecar_manifest_sha256="5" * 64,
                model_content_inventory_sha256=(
                    self.preparer.EXPECTED_MODEL_CONTENT_INVENTORY_SHA256
                ),
                runtime_registry=self.trainer.EXPECTED_RUNTIME_REGISTRY,
                runtime_registry_sha256="8" * 64,
                preflight_artifact_sha256="7" * 64,
                training_authorization_sha256=hashlib.sha256(
                    b"committed-authorization\n"
                ).hexdigest(),
                training_code_registry=code_registry_path,
                training_code_registry_sha256=sha256(code_registry_path),
            )
            mapping = {
                "active100_mapping_sha256": "9" * 64,
                "full178_mapping_sha256": "a" * 64,
                "canonical_prompt_builder_path": str(
                    self.preparer.PROMPT_BUILDER_PATH
                ),
                "canonical_prompt_builder_sha256": sha256(
                    self.preparer.PROMPT_BUILDER_FILE
                ),
            }
            registration = {
                "protocol": self.trainer.PROTOCOL,
                "status": "registered",
                "dataset_version": self.trainer.DATASET_VERSION,
                "created_utc": "2026-08-16T00:00:00+00:00",
                "output_dir": str(self.trainer.EXPECTED_OUTPUT_DIR),
                "only_training_intervention": (
                    "erase factual prompt_embeds replaced by registered augmented "
                    "source-slot sidecar"
                ),
                "train_manifest_path": str(self.trainer.EXPECTED_MANIFEST),
                "train_manifest_sha256": self.trainer.EXPECTED_MANIFEST_SHA256,
                "base_cache_path": str(self.trainer.EXPECTED_BASE_CACHE_DIR),
                "base_cache_inventory_sha256": self.preparer.EXPECTED_BASE_CACHE_SHA256,
                "teacher_cache_path": str(self.trainer.EXPECTED_TEACHER_CACHE_DIR),
                "teacher_cache_inventory_sha256": self.preparer.EXPECTED_TEACHER_CACHE_SHA256,
                "source_bank_registry_path": str(self.trainer.EXPECTED_BANK_REGISTRY),
                "source_bank_registry_sha256": args.source_bank_registry_sha256,
                "holdout_public_commitment_path": str(
                    self.trainer.EXPECTED_HOLDOUT_PUBLIC_COMMITMENT
                ),
                "holdout_public_commitment_sha256": args.holdout_public_commitment_sha256,
                "holdout_count": 24,
                "source_mapping_registry_path": str(
                    self.trainer.EXPECTED_MAPPING_REGISTRY
                ),
                "source_mapping_registry_sha256": args.source_mapping_registry_sha256,
                "active100_mapping_sha256": mapping["active100_mapping_sha256"],
                "full178_mapping_sha256": mapping["full178_mapping_sha256"],
                "canonical_prompt_builder_path": mapping[
                    "canonical_prompt_builder_path"
                ],
                "canonical_prompt_builder_sha256": mapping[
                    "canonical_prompt_builder_sha256"
                ],
                "prompt_sidecar_path": str(self.trainer.EXPECTED_PROMPT_SIDECAR_DIR),
                "prompt_sidecar_inventory_sha256": args.prompt_sidecar_inventory_sha256,
                "prompt_sidecar_manifest_sha256": args.prompt_sidecar_manifest_sha256,
                "model_content_inventory_sha256": args.model_content_inventory_sha256,
                "transformer_inventory_sha256": self.trainer.FROZEN_TRANSFORMER_INVENTORY_SHA256,
                "runtime_registry_path": str(self.trainer.EXPECTED_RUNTIME_REGISTRY),
                "runtime_registry_sha256": args.runtime_registry_sha256,
                "preflight_artifact_path": str(self.trainer.EXPECTED_PREFLIGHT),
                "preflight_artifact_sha256": args.preflight_artifact_sha256,
                "training_authorization_path": str(self.trainer.EXPECTED_AUTHORIZATION),
                "training_authorization_sha256": args.training_authorization_sha256,
                "training_code_registry_path": str(
                    self.trainer.EXPECTED_TRAINING_CODE_REGISTRY
                ),
                "training_code_registry_sha256": args.training_code_registry_sha256,
                "authorization_source": "independent_audited_committed_and_pushed",
                "git_commit": "d" * 40,
                "git_upstream": "origin/fixture",
                "expected_initial_lora_sha256": self.preparer.EXPECTED_INITIAL_LORA_SHA256,
                "expected_sample_order_sha256": self.trainer.EXPECTED_SAMPLE_ORDER_SHA256,
                "expected_noise_sigma_rng_initial_sha256": self.preparer.EXPECTED_NOISE_RNG_INITIAL_SHA256,
                "expected_noise_sigma_rng_final_sha256": self.preparer.EXPECTED_NOISE_RNG_FINAL_SHA256,
                "training_config": self.trainer.EXPECTED_CONFIG,
            }
            for name, record in code_registry["artifacts"].items():
                registration[f"{name}_path"] = record["path"]
                registration[f"{name}_sha256"] = record["sha256"]
            write_json(args.run_registration, registration)
            args.run_registration_sha256 = sha256(args.run_registration)
            def fake_git_output(command, **kwargs):
                if command == ["git", "rev-parse", "HEAD"]:
                    return "d" * 40 + "\n"
                if command[-1] == "@{upstream}":
                    return "origin/fixture\n"
                if command[:2] == ["git", "show"]:
                    return b"committed-authorization\n"
                raise AssertionError(command)

            with mock.patch.object(
                self.trainer.subprocess,
                "check_output",
                side_effect=fake_git_output,
            ), mock.patch.object(self.trainer.subprocess, "run"), mock.patch.object(
                self.trainer, "validate_runtime_registry"
            ):
                validated, _ = self.trainer.validate_run_registration(args, mapping)
            self.assertEqual(validated, registration)
            registration["dataset_version"] = "v4_dev72_v1"
            write_json(args.run_registration, registration)
            args.run_registration_sha256 = sha256(args.run_registration)
            with self.assertRaisesRegex(ValueError, "dataset_version mismatch"):
                with mock.patch.object(
                    self.trainer.subprocess,
                    "check_output",
                    side_effect=fake_git_output,
                ), mock.patch.object(self.trainer.subprocess, "run"), mock.patch.object(
                    self.trainer, "validate_runtime_registry"
                ):
                    self.trainer.validate_run_registration(args, mapping)
            registration["dataset_version"] = self.trainer.DATASET_VERSION
            registration["unregistered_extra"] = True
            write_json(args.run_registration, registration)
            args.run_registration_sha256 = sha256(args.run_registration)
            with self.assertRaisesRegex(ValueError, "fields are not exact"):
                with mock.patch.object(
                    self.trainer.subprocess,
                    "check_output",
                    side_effect=fake_git_output,
                ), mock.patch.object(self.trainer.subprocess, "run"), mock.patch.object(
                    self.trainer, "validate_runtime_registry"
                ):
                    self.trainer.validate_run_registration(args, mapping)

    def test_source_sidecar_is_the_only_erase_conditioning_change(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("prompt_embeds = base_prompt_embeds", source)
        self.assertIn('prompt_embeds = sidecar["augmented_prompt_embeds"]', source)
        self.assertIn("flow_loss + args.target_prompt_teacher_weight * target_teacher_loss", source)
        self.assertIn("combined_loss = args.preserve_weight * preserve_loss", source)
        self.assertGreaterEqual(source.count('reduction="none"'), 3)
        self.assertNotIn("2 * sigma", source)
        self.assertNotIn("anti_guidance", source)
        self.assertNotIn("factual_latents", source)

    def test_sanity_precedes_backward_and_checkpoint(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        train = source[source.index("def train(") :]
        sanity = train.index("sanity = write_scale_sanity(")
        backward = train.index("loss.backward()")
        checkpoint = train.index("checkpoint_path =")
        save = train.index("atomic_save_lora(")
        self.assertLess(sanity, backward)
        self.assertLess(backward, checkpoint)
        self.assertLess(checkpoint, save)
        self.assertIn("if sanity is None or not sanity.get", train[backward:checkpoint])

    def test_eligibility_schema_binds_registered_step200_inputs(self) -> None:
        source = TRAINER.read_text(encoding="utf-8")
        for value in (
            "water_impact_dynamic_v4_checkpoint_eligibility_v2",
            '"dataset_version": DATASET_VERSION',
            '"status": "eligible"',
            '"step": 200',
            '"active100_mapping_sha256"',
            '"full178_mapping_sha256"',
            '"noise_sigma_rng_final_sha256"',
            '"prompt_sidecar_inventory_sha256"',
            '"training_authorization_sha256"',
            '"training_code_registry_sha256"',
            '"runtime_registry_sha256"',
        ):
            self.assertIn(value, source)

    def test_null_preflight_restores_rng_before_signature_comparison(self) -> None:
        source = PREPARER.read_text(encoding="utf-8")
        preflight = source[source.index("def run_null_preflight(") :]
        base = preflight.index(
            "v3b_reference_signature = _v3b_reference_forward_signature("
        )
        restore_cpu = preflight.index("torch.set_rng_state(cpu_rng_state)")
        restore_cuda = preflight.index("torch.cuda.set_rng_state_all(cuda_rng_state)")
        null = preflight.index(
            "v4_null_sidecar_signature = _v4_null_sidecar_forward_signature("
        )
        self.assertLess(base, restore_cpu)
        self.assertLess(restore_cpu, restore_cuda)
        self.assertLess(restore_cuda, null)
        reference_function = source[
            source.index("def _v3b_reference_forward_signature(") :
            source.index("def _v4_null_sidecar_forward_signature(")
        ]
        null_function = source[
            source.index("def _v4_null_sidecar_forward_signature(") :
            source.index("def run_null_preflight(")
        ]
        self.assertIn('reduction="none"', reference_function)
        self.assertIn('reduction="none"', null_function)
        self.assertIn('null_sidecar["augmented_prompt_embeds"]', null_function)

        trainer_source = TRAINER.read_text(encoding="utf-8")
        validator = trainer_source[
            trainer_source.index("def validate_preflight(") :
            trainer_source.index("def validate_run_registration(")
        ]
        self.assertIn("set(payload) != exact_fields", validator)
        self.assertIn(
            'payload["original_reencode_binding_sha256"] != sidecar_manifest.get(',
            validator,
        )
        self.assertIn('payload[field] != model_provenance[field]', validator)

    def test_launcher_remains_fail_closed_until_curator_and_preflight_freeze(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "6988d19ef759b6fd7c15543a5b1774656a20dace82ea82fe5812004f763bb4c2",
            launcher,
        )
        self.assertNotIn("TO_BE_FROZEN_AFTER_MAPPING_REVIEW", launcher)
        self.assertIn("TO_BE_FROZEN_AFTER_NULL_SIDECAR_PREFLIGHT", launcher)
        self.assertNotIn("prepare-cache", launcher)
        result = subprocess.run(
            ["bash", str(LAUNCHER), "train"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prompt sidecar inventory is not frozen", result.stderr)
        self.assertFalse((ROOT / self.trainer.EXPECTED_OUTPUT_DIR).exists())

    def test_launcher_rehashes_authorization_public_refs_and_code(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        start_marker = (
            'verify_authorization_chain() {\n'
            '  "$PYTHON" - \\\n'
            '    "$TRAINING_AUTHORIZATION" \\\n'
            '    "$EXPECTED_TRAINING_AUTHORIZATION_SHA256" \\\n'
            '    "$EXPECTED_SOURCE_BANK_REGISTRY_SHA256" \\\n'
            '    "$EXPECTED_HOLDOUT_PUBLIC_COMMITMENT_SHA256" <<\'PY\'\n'
        )
        start = launcher.index(start_marker) + len(start_marker)
        end = launcher.index("\nPY\n}\n\nread_authorization_git_provenance", start)
        verifier = launcher[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization, payload = authorization_fixture(self.trainer, root)
            command = [
                sys.executable,
                "-c",
                verifier,
                str(authorization),
                sha256(authorization),
                payload["source_bank_registry"]["sha256"],
                payload["holdout_public_commitment"]["sha256"],
            ]
            result = subprocess.run(
                command, cwd=root, text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            eval_runner = root / self.trainer.CODE_ARTIFACT_PATHS["eval_runner"]
            original_eval_runner = eval_runner.read_bytes()
            eval_runner.write_bytes(original_eval_runner + b"tampered\n")
            result = subprocess.run(
                command, cwd=root, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("eval_runner byte hash mismatch", result.stderr)
            eval_runner.write_bytes(original_eval_runner)

            gate_path = root / self.trainer.AUTHORIZATION_REF_PATHS["gate_registry"]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            gate["gate_spec"]["decision"] = "TO_BE_FROZEN_GATE"
            gate["gate_spec_sha256"] = self.trainer.canonical_json_sha256(
                gate["gate_spec"]
            )
            write_json(gate_path, gate)
            payload["gate_registry"]["sha256"] = sha256(gate_path)
            write_json(authorization, payload)
            command[4] = sha256(authorization)
            result = subprocess.run(
                command, cwd=root, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contains a placeholder", result.stderr)

    def test_no_private_holdout_or_final36_paths_in_training_implementation(self) -> None:
        joined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (MAPPING, RUNTIME_BUILDER, PREPARER, TRAINER, LAUNCHER)
        )
        self.assertNotIn("v3c_sealed_final36.csv", joined)
        self.assertNotIn("private_source_registry", joined)
        self.assertNotIn("private_holdout_registry", joined)


if __name__ == "__main__":
    unittest.main()
