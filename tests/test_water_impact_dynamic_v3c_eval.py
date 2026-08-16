from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_water_impact_dynamic_v3c_blind_review as builder  # noqa: E402
import score_water_impact_dynamic_v3c_fresh_dev24 as scorer  # noqa: E402
import water_impact_dynamic_v3c_eval_protocol as protocol  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V3CStage2Tests(unittest.TestCase):
    def make_training_payloads(self):
        registration_sha = "a" * 64
        sanity_sha = "b" * 64
        registration = {
            "protocol": protocol.V3C_TRAINING_PROTOCOL,
            "output_dir": protocol.V3C_TRAINING_ROOT,
            "calibration_id": protocol.V3C_CALIBRATION_ID,
            "target_prompt_teacher_base_weight": 4.0,
            "target_prompt_teacher_schedule": "2*sigma",
            "target_prompt_teacher_effective_weight_formula": "4 * (2 * sigma)",
            "target_prompt_teacher_expected_mean_weight": 4.0,
            "sanity_mean_min": 0.2,
            "sanity_mean_max": 0.5,
            "sanity_single_max": 1.0,
            "checkpoint_policy": "no_checkpoint_before_scale_sanity_passes",
            "model_revision": protocol.MODEL_REVISION,
            "transformer_inventory_algorithm": protocol.TRANSFORMER_INVENTORY_ALGORITHM,
            "transformer_inventory": protocol.FROZEN_TRANSFORMER_INVENTORY,
            "transformer_inventory_sha256": protocol.FROZEN_TRANSFORMER_INVENTORY_SHA256,
            "training_config": {
                "model": protocol.MODEL,
                "height": 480,
                "width": 832,
                "num_frames": 49,
                "max_steps": 200,
                "learning_rate": 5e-5,
                "rank": 16,
                "alpha": 16,
                "grad_accum": 1,
                "seed": 26000,
                "device": "cuda",
                "role": "all",
                "objective": "target_prompt_teacher_sigma_weighted",
                "balanced_roles": True,
                "preserve_weight": 4.0,
                "save_every": 25,
                "target_prompt_calibration_id": protocol.V3C_CALIBRATION_ID,
                "target_prompt_teacher_base_weight": 4.0,
                "target_prompt_teacher_schedule": "2*sigma",
                "sanity_mean_min": 0.2,
                "sanity_mean_max": 0.5,
                "sanity_single_max": 1.0,
            },
        }
        observations = []
        for index in range(16):
            sigma = 0.5
            ratio = 0.0049
            effective = 8 * sigma
            observations.append(
                {
                    "global_step": 2 * index + 1,
                    "flow_loss": 1.0,
                    "target_prompt_teacher_loss": ratio,
                    "sigma": sigma,
                    "effective_teacher_weight": effective,
                    "raw_loss_ratio": ratio,
                    "weighted_loss_ratio": effective * ratio,
                    "weighted_output_gradient_norm_ratio": effective * math.sqrt(ratio),
                }
            )
        outputs = [row["weighted_output_gradient_norm_ratio"] for row in observations]
        sanity = {
            "protocol": protocol.V3C_SANITY_PROTOCOL,
            "calibration_id": protocol.V3C_CALIBRATION_ID,
            "run_registration_sha256": registration_sha,
            "formula": "g_i = 8 * sigma_i * sqrt(target_prompt_teacher_loss / flow_loss)",
            "aggregation": "arithmetic_mean_over_first_16_erase_steps",
            "base_weight": 4.0,
            "weight_schedule": "2*sigma",
            "mean_output_gradient_norm_ratio_min": 0.2,
            "mean_output_gradient_norm_ratio_max": 0.5,
            "single_output_gradient_norm_ratio_max": 1.0,
            "observation_count": 16,
            "max_weighted_output_gradient_norm_ratio": max(outputs),
            "mean_raw_loss_ratio": 0.0049,
            "mean_sigma": 0.5,
            "mean_effective_teacher_weight": 4.0,
            "mean_weighted_loss_ratio": 0.0196,
            "mean_weighted_output_grad_ratio": 0.28,
            "median_weighted_output_grad_ratio": 0.28,
            "passed": True,
            "observations": observations,
        }
        state = {
            "step": 200,
            "max_steps": 200,
            "model": protocol.MODEL,
            "objective": "target_prompt_teacher_sigma_weighted",
            "target_prompt_teacher_enabled": True,
            "target_prompt_teacher_weight": 4.0,
            "target_prompt_teacher_weight_semantics": "base_weight_before_2*sigma_schedule",
            "target_prompt_teacher_schedule": "2*sigma",
            "target_prompt_teacher_effective_weight_formula": "4 * (2 * sigma)",
            "target_prompt_calibration_id": protocol.V3C_CALIBRATION_ID,
            "teacher_adapter_mode": "disabled",
            "teacher_stop_gradient": True,
            "teacher_uses_same_noisy_latent": True,
            "teacher_uses_same_timestep": True,
            "teacher_scale_sanity_protocol": protocol.V3C_SANITY_PROTOCOL,
            "teacher_scale_sanity_count": 16,
            "teacher_scale_sanity_passed": True,
            "run_registration_path": f"{protocol.V3C_TRAINING_ROOT}/run_registration.json",
            "run_registration_sha256": registration_sha,
            "teacher_scale_sanity_path": f"{protocol.V3C_TRAINING_ROOT}/target_prompt_scale_sanity.json",
            "teacher_scale_sanity_sha256": sanity_sha,
            "transformer_inventory_algorithm": protocol.TRANSFORMER_INVENTORY_ALGORITHM,
            "transformer_inventory": protocol.FROZEN_TRANSFORMER_INVENTORY,
            "transformer_inventory_sha256": protocol.FROZEN_TRANSFORMER_INVENTORY_SHA256,
        }
        return state, registration, sanity, registration_sha, sanity_sha

    def test_training_semantics_accept_only_passed_eligible_checkpoint(self) -> None:
        state, registration, sanity, registration_sha, sanity_sha = self.make_training_payloads()
        protocol.validate_v3c_training_semantics(
            state=state,
            registration=registration,
            sanity=sanity,
            registration_sha256=registration_sha,
            sanity_sha256=sanity_sha,
        )
        sanity["passed"] = False
        with self.assertRaisesRegex(ValueError, "passed"):
            protocol.validate_v3c_training_semantics(
                state=state,
                registration=registration,
                sanity=sanity,
                registration_sha256=registration_sha,
                sanity_sha256=sanity_sha,
            )

    def test_model_inventory_detects_weight_tampering_but_ignores_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            (model / "transformer").mkdir(parents=True)
            (model / ".cache").mkdir()
            (model / "transformer" / "config.json").write_text("{}", encoding="utf-8")
            weights = model / "transformer" / "weights.bin"
            weights.write_bytes(b"weights-v1")
            cache = model / ".cache" / "metadata"
            cache.write_bytes(b"revision-a")
            with patch.object(protocol, "MODEL", "model"):
                first = protocol.model_artifact_inventory(root)
                cache.write_bytes(b"revision-b")
                self.assertEqual(first, protocol.model_artifact_inventory(root))
                weights.write_bytes(b"weights-v2")
                self.assertNotEqual(first["sha256"], protocol.model_artifact_inventory(root)["sha256"])

    def test_generation_inventory_must_contain_training_frozen_transformer(self) -> None:
        inventory = {"files": [dict(record) for record in protocol.FROZEN_TRANSFORMER_INVENTORY]}
        protocol.validate_frozen_transformer_in_model_inventory(inventory)
        inventory["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "training-frozen inventory"):
            protocol.validate_frozen_transformer_in_model_inventory(inventory)

    def test_missing_actual_stage2_and_committed_template_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "generation is locked"):
                protocol.load_stage2_registration(Path(directory))
        template = json.loads((PROJECT_ROOT / protocol.STAGE2_TEMPLATE).read_text(encoding="utf-8"))
        self.assertTrue(protocol._contains_placeholder(template))
        self.assertEqual(template["status"], "placeholder_fail_closed")

    def test_launcher_requires_stage2_for_all_arms_and_has_no_final_command(self) -> None:
        launcher = (PROJECT_ROOT / "scripts/run_water_impact_dynamic_v3c_fresh_dev24.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("stage2_preflight", launcher)
        self.assertIn(".model_inventory_sha256", launcher)
        self.assertIn("sealed-final36 intentionally has no launcher command", launcher)
        self.assertNotIn("sealed_final36.prompts", launcher)


class V3CBuilderTests(unittest.TestCase):
    @staticmethod
    def fake_composite(path: Path, pair: str, group: str, methods: list[str], videos):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{pair}:{group}:{methods}".encode("utf-8"))

    def test_builds_isolated_24_pair_48_candidate_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows = [
                {
                    "pair_id": f"pair_{index}",
                    "generalization_group": protocol.GENERALIZATION_GROUPS[index % 3],
                    "source_object": f"object_{index}",
                    "receiver": f"receiver_{index}",
                }
                for index in range(24)
            ]
            videos = {label: {} for label in ("original", "v3b", "v3c")}
            manifests = {}
            for label in videos:
                for index in range(24):
                    path = root / label / f"{index}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{label}:{index}".encode("utf-8"))
                    videos[label][index] = path
                manifest = root / f"{label}.json"
                manifest.write_text(json.dumps({"label": label}), encoding="utf-8")
                manifests[label] = manifest
            stage2 = root / "stage2.json"
            stage2.write_text("{}\n", encoding="utf-8")
            public = root / "review_public"
            private = root / "review_private"
            payload = builder.build_review_package(
                eval_rows=eval_rows,
                videos=videos,
                generation_manifests=manifests,
                stage2_path=stage2,
                stage2_payload={},
                public_dir=public,
                private_dir=private,
                composite_builder=self.fake_composite,
            )
            review = read_csv(public / "blind_review.csv")
            key = read_csv(private / "answer_key.csv")
            self.assertEqual(len(review), 48)
            self.assertEqual(len(key), 48)
            self.assertNotIn("method", review[0])
            self.assertEqual({row["method"] for row in key}, {"v3b", "v3c"})
            self.assertEqual({path.name for path in public.iterdir()}, {"blind_review.csv", "composites", "media"})
            self.assertEqual({path.name for path in private.iterdir()}, {"answer_key.csv", "review_manifest.json"})
            self.assertEqual(len(payload["anonymous_media_sha256"]), 48)


class V3CScorerTests(unittest.TestCase):
    def make_gate_rows(self) -> list[dict[str, object]]:
        output = []
        for index in range(24):
            group = protocol.GENERALIZATION_GROUPS[index % 3]
            output.append(
                {
                    "sample_index": index,
                    "generalization_group": group,
                    "method": "v3b",
                    protocol.SCORE_FIELDS[0]: 2,
                    protocol.SCORE_FIELDS[1]: 1,
                    protocol.SCORE_FIELDS[2]: 2,
                    protocol.SCORE_FIELDS[3]: 2,
                }
            )
            target = 0 if index < 6 else 2
            footprint = 0 if index < 2 else 1
            output.append(
                {
                    "sample_index": index,
                    "generalization_group": group,
                    "method": "v3c",
                    protocol.SCORE_FIELDS[0]: target,
                    protocol.SCORE_FIELDS[1]: footprint,
                    protocol.SCORE_FIELDS[2]: 2,
                    protocol.SCORE_FIELDS[3]: 2,
                }
            )
        return output

    def test_all_or_nothing_gate_passes_registered_boundary(self) -> None:
        _, gate = scorer.compute_gate(self.make_gate_rows())
        self.assertTrue(gate["promote_v3c_and_unseal_final36"])
        self.assertEqual(len(gate["paired_target_improvements"]), 6)
        self.assertEqual(len(gate["clear_to_absent_improvements"]), 6)

    def test_unusable_treatment_gets_zero_valid_points_and_absence_credit(self) -> None:
        rows = self.make_gate_rows()
        treatment = next(
            row for row in rows if row["method"] == "v3c" and row["sample_index"] == 0
        )
        treatment[protocol.SCORE_FIELDS[2]] = 0
        summaries, gate = scorer.compute_gate(rows)
        summary = {row["method"]: row for row in summaries}
        self.assertEqual(gate["v3c_target_suppression_points_on_C"], 10)
        self.assertEqual(gate["v3c_footprint_suppression_points_on_C"], 24)
        self.assertEqual(summary["v3c"]["usable_absent_target_n"], 5)
        self.assertEqual(len(gate["paired_target_improvements"]), 5)
        self.assertFalse(gate["checks"]["paired_target_improvements_at_least_6"])
        self.assertFalse(gate["promote_v3c_and_unseal_final36"])

    def test_two_reviewers_and_blinded_adjudicator_merge_atomically(self) -> None:
        template = []
        reviewer_a = []
        reviewer_b = []
        for index in range(48):
            base = {
                "review_id": f"r{index // 2:03d}_{'AB'[index % 2]}",
                "sample_index": str(index // 2),
                "pair_id": f"pair_{index // 2}",
                "generalization_group": protocol.GENERALIZATION_GROUPS[index % 3],
                "candidate_code": "AB"[index % 2],
                "composite_path": f"r{index // 2:03d}.jpg",
                "candidate_video_path": f"{index}.mp4",
                "source_object": "object",
                "receiver": "receiver",
                **{field: "" for field in protocol.SCORE_FIELDS},
                "notes": "",
            }
            template.append(base)
            a = dict(base)
            b = dict(base)
            for field, value in zip(protocol.SCORE_FIELDS, (2, 1, 2, 2)):
                a[field] = str(value)
                b[field] = str(value)
            reviewer_a.append(a)
            reviewer_b.append(b)
        reviewer_b[0][protocol.SCORE_FIELDS[0]] = "0"
        canonical, audit = scorer.merge_blind_reviews(
            template,
            reviewer_a,
            reviewer_b,
            [{"review_id": "r000_A", "field": "target", "score": "1", "brief_reason": "blind"}],
        )
        self.assertEqual(canonical[0][protocol.SCORE_FIELDS[0]], 1)
        self.assertEqual(len(audit), 1)
        with self.assertRaisesRegex(ValueError, "requires blinded adjudication"):
            scorer.merge_blind_reviews(template, reviewer_a, reviewer_b, [])


if __name__ == "__main__":
    unittest.main()
