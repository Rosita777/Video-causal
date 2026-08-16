from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_water_impact_dynamic_v3b_blind_review as builder  # noqa: E402
import score_water_impact_dynamic_v3b_eval12 as scorer  # noqa: E402
import water_impact_dynamic_v3b_eval_protocol as protocol  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V3BProtocolTests(unittest.TestCase):
    def make_sanity(self) -> dict[str, object]:
        raw = [0.0025 + index * 0.0001 for index in range(16)]
        output_ratios = [4.0 * math.sqrt(value) for value in raw]
        observations = [
            {
                "global_step": 2 * index + 1,
                "scene_id": f"scene_{index}",
                "flow_loss": 1.0,
                "target_prompt_teacher_loss": value,
                "raw_loss_ratio": value,
                "weighted_loss_ratio": 4.0 * value,
                "weighted_output_gradient_norm_ratio": output_ratios[index],
            }
            for index, value in enumerate(raw)
        ]
        return {
            "protocol": protocol.SCALE_SANITY_PROTOCOL,
            "calibration_id": protocol.CALIBRATION_ID,
            "run_registration_sha256": protocol.V3B_REGISTRATION_SHA256,
            "formula": "s_i = weight * sqrt(target_prompt_teacher_loss / flow_loss)",
            "aggregation": "arithmetic_mean_over_first_16_erase_steps",
            "weight": 4.0,
            "mean_output_gradient_norm_ratio_min": 0.2,
            "mean_output_gradient_norm_ratio_max": 0.5,
            "single_output_gradient_norm_ratio_max": 1.0,
            "observation_count": 16,
            "max_weighted_output_gradient_norm_ratio": max(output_ratios),
            "mean_raw_loss_ratio": statistics.fmean(raw),
            "mean_weighted_loss_ratio": 4.0 * statistics.fmean(raw),
            "mean_weighted_output_grad_ratio": statistics.fmean(output_ratios),
            "median_weighted_output_grad_ratio": statistics.median(output_ratios),
            "passed": True,
            "observations": observations,
        }

    def test_scale_sanity_recomputes_all_registered_ratios(self) -> None:
        protocol._validate_sanity(self.make_sanity())

    def test_scale_sanity_rejects_tampered_observation(self) -> None:
        payload = self.make_sanity()
        payload["observations"][0]["weighted_output_gradient_norm_ratio"] = 0.9  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "output-gradient ratio"):
            protocol._validate_sanity(payload)

    def test_frozen_v3b_artifact_hashes_are_fully_registered(self) -> None:
        self.assertEqual(len(protocol.V3B_CHECKPOINT_SHA256), 64)
        self.assertEqual(len(protocol.V3B_WEIGHTS_SHA256), 64)
        self.assertEqual(len(protocol.V3B_TRAINING_STATE_SHA256), 64)
        self.assertEqual(len(protocol.V3B_REGISTRATION_SHA256), 64)
        self.assertEqual(len(protocol.V3B_SANITY_SHA256), 64)
        self.assertNotEqual(protocol.V3B_CHECKPOINT_SHA256, protocol.V3B_WEIGHTS_SHA256)

    def test_launcher_stdin_preflight_can_import_protocol_module(self) -> None:
        launcher = (PROJECT_ROOT / "scripts" / "run_water_impact_dynamic_v3b_eval12.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -', launcher)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "scripts")
        result = subprocess.run(
            [sys.executable, "-c", "import water_impact_dynamic_v3b_eval_protocol"],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_revision_reads_live_metadata_first_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = (
                root
                / protocol.MODEL
                / ".cache/huggingface/download/model_index.json.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(f"{protocol.MODEL_REVISION}\nextra\n", encoding="utf-8")
            result = protocol.validate_model_revision(root)
            self.assertEqual(result["model_revision"], protocol.MODEL_REVISION)
            metadata.write_text("wrong-revision\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model revision mismatch"):
                protocol.validate_model_revision(root)

    def test_generation_loader_rejects_unregistered_extra_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            video_dir = run_dir / "videos"
            video_dir.mkdir(parents=True)
            eval_rows = []
            items = []
            for index, seed in enumerate(protocol.SEEDS):
                row = {
                    "training_prompt": f"prompt {index}",
                    "source_object": f"object {index}",
                    "expected_factual_event": f"effect {index}",
                    "seed": str(seed),
                }
                eval_rows.append(row)
                video = video_dir / f"{index}.mp4"
                video.write_bytes(b"video")
                items.append(
                    {
                        "index": index,
                        "prompt": row["training_prompt"],
                        "target_concept": row["source_object"],
                        "expected_effect": row["expected_factual_event"],
                        "seed": seed,
                        "video_path": str(video),
                    }
                )
            generation = {
                "baseline": "clean",
                "seed": 42,
                "seeds": list(protocol.SEEDS),
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
                "activation_gate_dir": None,
                "persistent_activation_gate": False,
                "lora_target_phrases": [],
                "attention_gate_dir": None,
                "attention_suppression_phrases": [],
                "attention_suppression_strength": 20.0,
                "lora_path": None,
                "lora_sha256": None,
                "lora_scale": 1.0,
            }
            (run_dir / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "baseline": "clean",
                        "pipeline": "WanPipeline",
                        "model": protocol.MODEL,
                        "dry_run": False,
                        "prompts": protocol.PROMPTS,
                        "generation": generation,
                        "items": items,
                    }
                ),
                encoding="utf-8",
            )
            (video_dir / "extra.mp4").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "unregistered"):
                protocol.load_generation_run(root, "run", "original", eval_rows)


class V3BBuilderTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        eval_rows = []
        videos: dict[str, dict[int, Path]] = {
            "original": {},
            "balanced": {},
            "v3b": {},
        }
        for index in range(12):
            eval_rows.append(
                {
                    "pair_id": f"pair_{index}",
                    "generalization_group": f"group_{index % 3}",
                    "source_object": f"object_{index}",
                    "receiver": f"receiver_{index}",
                }
            )
            for label in videos:
                path = root / label / f"{index}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{label}:{index}".encode("utf-8"))
                videos[label][index] = path
        manifests = {}
        for label in videos:
            path = root / f"{label}_manifest.json"
            path.write_text(json.dumps({"label": label}), encoding="utf-8")
            manifests[label] = path
        return eval_rows, videos, manifests

    @staticmethod
    def fake_composite(
        output_path: Path,
        pair_id: str,
        group: str,
        methods: list[str],
        paths: dict[str, Path],
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"{pair_id}:{group}:{','.join(methods)}".encode("utf-8"))

    def test_builds_12_pair_24_row_package_with_all_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows, videos, manifests = self.make_inputs(root)
            public = root / "review_v3_public"
            private = root / "review_v3_private"
            payload = builder.build_review_package(
                eval_rows=eval_rows,
                videos=videos,
                manifest_paths=manifests,
                training_provenance={"balanced": {"sha256": "a"}, "v3b": {"sha256": "b"}},
                public_dir=public,
                private_dir=private,
                composite_builder=self.fake_composite,
            )
            review = read_csv(public / "blind_review.csv")
            key = read_csv(private / "answer_key.csv")
            self.assertEqual(len(review), 24)
            self.assertEqual(len(key), 24)
            self.assertEqual({row["method"] for row in key}, {"balanced", "v3b"})
            self.assertNotIn("method", review[0])
            self.assertNotIn("video_path", review[0])
            self.assertIn("candidate_video_path", review[0])
            self.assertEqual(len(payload["composite_sha256"]), 12)
            self.assertEqual(len(payload["anonymous_media_sha256"]), 24)
            self.assertEqual(set(payload["video_sha256"]), {"original", "balanced", "v3b"})
            self.assertTrue(all(len(arm) == 12 for arm in payload["video_sha256"].values()))
            self.assertEqual(set(payload["generation_manifests"]), {"original", "balanced", "v3b"})
            self.assertEqual(
                {path.name for path in public.iterdir()},
                {"blind_review.csv", "composites", "media"},
            )
            self.assertEqual(
                {path.name for path in private.iterdir()},
                {"answer_key.csv", "review_manifest.json"},
            )
            key_by_id = {row["review_id"]: row for row in key}
            for row in review:
                anonymous = Path(row["candidate_video_path"])
                source = Path(key_by_id[row["review_id"]]["video_path"])
                self.assertTrue(anonymous.is_file())
                self.assertFalse(anonymous.is_symlink())
                self.assertFalse(anonymous.samefile(source))
                self.assertEqual(protocol.file_sha256(anonymous), protocol.file_sha256(source))

    def test_rejects_video_reused_across_controlled_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows, videos, manifests = self.make_inputs(root)
            videos["v3b"][0] = videos["balanced"][0]
            with self.assertRaisesRegex(ValueError, "overlapping video"):
                builder.build_review_package(
                    eval_rows=eval_rows,
                    videos=videos,
                    manifest_paths=manifests,
                    training_provenance={},
                    public_dir=root / "review_v3_public",
                    private_dir=root / "review_v3_private",
                    composite_builder=self.fake_composite,
                )

    def test_rejects_composite_paths_swapped_between_review_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows, videos, manifests = self.make_inputs(root)
            public = root / "review_v3_public"
            private = root / "review_v3_private"
            builder.build_review_package(
                eval_rows=eval_rows,
                videos=videos,
                manifest_paths=manifests,
                training_provenance={},
                public_dir=public,
                private_dir=private,
                composite_builder=self.fake_composite,
            )
            review = read_csv(public / "blind_review.csv")
            key = read_csv(private / "answer_key.csv")
            manifest_path = private / "review_manifest.json"
            scorer.validate_frozen_blind_assignment(
                root, public, eval_rows, review, key
            )
            first = next(row["composite_path"] for row in review if row["review_id"] == "r000_A")
            second = next(row["composite_path"] for row in review if row["review_id"] == "r001_A")
            for row in review:
                if row["review_id"].startswith("r000_"):
                    row["composite_path"] = second
                elif row["review_id"].startswith("r001_"):
                    row["composite_path"] = first
            with self.assertRaisesRegex(ValueError, "composite-path assignment"):
                scorer.validate_frozen_blind_assignment(
                    root, public, eval_rows, review, key
                )

    def test_rejects_anonymous_video_path_swap_and_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows, videos, manifests = self.make_inputs(root)
            public = root / "review_v3_public"
            private = root / "review_v3_private"
            payload = builder.build_review_package(
                eval_rows=eval_rows,
                videos=videos,
                manifest_paths=manifests,
                training_provenance={},
                public_dir=public,
                private_dir=private,
                composite_builder=self.fake_composite,
            )
            review = read_csv(public / "blind_review.csv")
            key = read_csv(private / "answer_key.csv")
            manifest_path = private / "review_manifest.json"
            scorer.validate_anonymous_media(
                root, public, payload, review, key, videos
            )
            original_path = review[0]["candidate_video_path"]
            review[0]["candidate_video_path"] = review[1]["candidate_video_path"]
            with self.assertRaisesRegex(ValueError, "candidate-video path"):
                scorer.validate_frozen_blind_assignment(
                    root, public, eval_rows, review, key
                )
            review[0]["candidate_video_path"] = original_path
            Path(original_path).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "anonymous-media hash mismatch"):
                scorer.validate_anonymous_media(
                    root, public, payload, review, key, videos
                )

    def test_public_private_inventory_rejects_key_leak_into_public_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_rows, videos, manifests = self.make_inputs(root)
            public = root / "review_v3_public"
            private = root / "review_v3_private"
            builder.build_review_package(
                eval_rows=eval_rows,
                videos=videos,
                manifest_paths=manifests,
                training_provenance={},
                public_dir=public,
                private_dir=private,
                composite_builder=self.fake_composite,
            )
            review_path = public / "blind_review.csv"
            key_path = private / "answer_key.csv"
            manifest_path = private / "review_manifest.json"
            with (
                patch.object(scorer, "PUBLIC_OUTPUT_DIR", public),
                patch.object(scorer, "PRIVATE_OUTPUT_DIR", private),
            ):
                scorer.validate_public_private_inventory(
                    root, review_path, key_path, manifest_path
                )
                (public / "answer_key.csv").write_bytes(key_path.read_bytes())
                with self.assertRaisesRegex(ValueError, "unexpected entry"):
                    scorer.validate_public_private_inventory(
                        root, review_path, key_path, manifest_path
                    )


class V3BScorerTests(unittest.TestCase):
    @staticmethod
    def make_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        groups = ("unseen_source", "unseen_receiver", "unseen_source_and_receiver")
        for index in range(12):
            rows.append(
                {
                    "sample_index": index,
                    "generalization_group": groups[index % 3],
                    "method": "balanced",
                    scorer.SCORE_FIELDS[0]: 2,
                    scorer.SCORE_FIELDS[1]: 1,
                    scorer.SCORE_FIELDS[2]: 2,
                    scorer.SCORE_FIELDS[3]: 2,
                }
            )
            target = (0, 1, 1)[index] if index < 3 else 2
            footprint = 0 if index == 0 else 1
            rows.append(
                {
                    "sample_index": index,
                    "generalization_group": groups[index % 3],
                    "method": "v3b",
                    scorer.SCORE_FIELDS[0]: target,
                    scorer.SCORE_FIELDS[1]: footprint,
                    scorer.SCORE_FIELDS[2]: 2,
                    scorer.SCORE_FIELDS[3]: 2,
                }
            )
        return rows

    def test_frozen_gate_promotes_only_when_mechanism_and_preservation_pass(self) -> None:
        _, gate = scorer.compute_gate(self.make_rows())
        self.assertTrue(gate["mechanism_positive"])
        self.assertTrue(gate["preservation_positive"])
        self.assertTrue(gate["promote_v3b_operating_point"])
        self.assertEqual(gate["v3b_target_suppression_points_on_control_usable"], 4)

    def test_unusable_v3b_sample_receives_zero_paired_suppression_points(self) -> None:
        rows = self.make_rows()
        treatment = next(
            row for row in rows if row["method"] == "v3b" and row["sample_index"] == 0
        )
        treatment[scorer.SCORE_FIELDS[2]] = 0
        _, gate = scorer.compute_gate(rows)
        self.assertEqual(gate["v3b_target_suppression_points_on_control_usable"], 2)
        self.assertEqual(gate["v3b_footprint_suppression_points_on_control_usable"], 11)
        self.assertFalse(gate["mechanism_positive"])

    def test_quality_absolute_floor_is_16_not_v3a_floor_22(self) -> None:
        rows = self.make_rows()
        for row in rows:
            if row["method"] == "balanced":
                row[scorer.SCORE_FIELDS[3]] = 1
            elif int(row["sample_index"]) >= 4:
                row[scorer.SCORE_FIELDS[3]] = 1
        _, gate = scorer.compute_gate(rows)
        self.assertEqual(
            sum(
                int(row[scorer.SCORE_FIELDS[3]])
                for row in rows
                if row["method"] == "v3b"
            ),
            16,
        )
        self.assertTrue(gate["checks"]["quality_absolute_floor_16"])
        self.assertTrue(gate["promote_v3b_operating_point"])


if __name__ == "__main__":
    unittest.main()
