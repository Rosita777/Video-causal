from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "train_wan_waterdrop_lora.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("train_wan_waterdrop_lora", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PairedSeparationLossTest(unittest.TestCase):
    def test_zero_when_margin_is_satisfied(self) -> None:
        module = load_module()
        prediction = torch.zeros((1, 2, 1, 1, 1))
        counterfactual = torch.zeros_like(prediction)
        factual = torch.ones_like(prediction)
        mask = torch.ones((1, 1, 1, 1, 1))

        loss = module.paired_separation_loss(prediction, counterfactual, factual, mask, 0.05)

        self.assertAlmostEqual(loss.item(), 0.0)

    def test_penalizes_factual_preference_and_has_gradient(self) -> None:
        module = load_module()
        prediction = torch.ones((1, 2, 1, 1, 1), requires_grad=True)
        counterfactual = torch.zeros_like(prediction)
        factual = torch.ones_like(prediction)
        mask = torch.ones((1, 1, 1, 1, 1))

        loss = module.paired_separation_loss(prediction, counterfactual, factual, mask, 0.05)
        loss.backward()

        self.assertAlmostEqual(loss.item(), 1.05, places=5)
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)


class FactualRedirectLossTest(unittest.TestCase):
    def test_zero_for_prediction_that_reconstructs_counterfactual(self) -> None:
        module = load_module()
        sigma = torch.tensor([0.5]).view(1, 1, 1, 1, 1)
        noisy_factual = torch.ones((1, 2, 1, 1, 1))
        counterfactual = torch.zeros_like(noisy_factual)
        prediction = (noisy_factual - counterfactual) / sigma
        mask = torch.ones((1, 1, 1, 1, 1))

        loss = module.factual_redirect_loss(
            prediction, noisy_factual, counterfactual, sigma, mask
        )

        self.assertAlmostEqual(loss.item(), 0.0)

    def test_penalizes_wrong_endpoint_and_has_gradient(self) -> None:
        module = load_module()
        sigma = torch.tensor([0.5]).view(1, 1, 1, 1, 1)
        noisy_factual = torch.ones((1, 2, 1, 1, 1))
        counterfactual = torch.zeros_like(noisy_factual)
        prediction = torch.zeros_like(noisy_factual, requires_grad=True)
        mask = torch.ones((1, 1, 1, 1, 1))

        loss = module.factual_redirect_loss(
            prediction, noisy_factual, counterfactual, sigma, mask
        )
        loss.backward()

        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)


class TrainingSeedTest(unittest.TestCase):
    def test_reseeding_reproduces_all_training_rngs(self) -> None:
        module = load_module()

        module.seed_training(26000)
        first = (random.random(), np.random.rand(), torch.randn(4))
        module.seed_training(26000)
        second = (random.random(), np.random.rand(), torch.randn(4))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_trainable_fingerprint_changes_with_weights_not_frozen_state(self) -> None:
        module = load_module()
        layer = torch.nn.Linear(3, 2)
        layer.bias.requires_grad_(False)

        original = module.trainable_state_sha256(layer)
        with torch.no_grad():
            layer.bias.add_(1)
        frozen_change = module.trainable_state_sha256(layer)
        with torch.no_grad():
            layer.weight.add_(1)
        trainable_change = module.trainable_state_sha256(layer)

        self.assertEqual(original, frozen_change)
        self.assertNotEqual(original, trainable_change)

    def test_trainable_fingerprint_supports_bfloat16(self) -> None:
        module = load_module()
        layer = torch.nn.Linear(3, 2, dtype=torch.bfloat16)

        digest = module.trainable_state_sha256(layer)

        self.assertEqual(len(digest), 64)

    def test_cache_inventory_fingerprint_changes_with_same_size_content(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000_scene.pt"
            path.write_bytes(b"alpha")
            original = module.cache_inventory_sha256([path])
            path.write_bytes(b"omega")
            changed = module.cache_inventory_sha256([path])

        self.assertNotEqual(original, changed)

    def test_cached_row_validator_rejects_prompt_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000_scene.pt"
            torch.save(
                {
                    "scene_id": "scene",
                    "prompt": "cached prompt",
                    "training_role": "erase",
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "cached prompt"):
                module.validate_cached_rows(
                    [path],
                    [
                        {
                            "scene_id": "scene",
                            "prompt": "frozen prompt",
                            "training_role": "erase",
                        }
                    ],
                )


class TargetPromptTeacherTest(unittest.TestCase):
    def test_run_registration_binds_canonical_code_and_frozen_training_config(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir_relative = Path(
                "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_scale4_v1"
            )
            output_dir = root / output_dir_relative
            output_dir.mkdir(parents=True)
            launcher = root / "scripts/run_water_impact_dynamic_sft_v3b_teacher.sh"
            protocol_doc = root / "docs/water_impact_dynamic_v3b_target_prompt_teacher.md"
            launcher.parent.mkdir()
            protocol_doc.parent.mkdir()
            launcher.write_text("launcher\n", encoding="utf-8")
            protocol_doc.write_text("protocol\n", encoding="utf-8")
            frozen_artifacts = {
                "outputs/water_impact_dynamic_v3b/logs/train_target_prompt_teacher_v1.log": (
                    "c0f35542d9be763ea4a446af773e0e22fe44913b019b89aca51588780f5719ba"
                ),
                "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/"
                "checkpoint-000025/pytorch_lora_weights.safetensors": (
                    "2ee9f08c83d291630c09efcdf5bf0f8ae082f7b23b4c6be0ed89de791377ff3b"
                ),
                "outputs/water_impact_dynamic_v3b/adapter_target_prompt_teacher_v1/"
                "checkpoint-000025/training_state.json": (
                    "d51fe90cedc168125e773f4c44ad458cc2baf84f409df6ed29f20cc09bcae854"
                ),
            }
            for relative in frozen_artifacts:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            args = SimpleNamespace(
                output_dir=output_dir_relative,
                manifest=Path(
                    "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
                ),
                cache_dir=Path(
                    "outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2"
                ),
                target_prompt_cache_dir=Path(
                    "outputs/water_impact_dynamic_v3b/teacher_prompt_cache_v1"
                ),
                target_prompt_calibration_id="lambda4_from_lambda1_first16_output_gradient_v1",
                target_prompt_teacher_weight=4.0,
                target_prompt_sanity_min_output_grad_ratio=0.2,
                target_prompt_sanity_max_output_grad_ratio=0.5,
                target_prompt_sanity_max_single_output_grad_ratio=1.0,
                target_prompt_cache_sha256=(
                    "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
                ),
                model=Path("models/Wan2.1-T2V-1.3B-Diffusers"),
                height=480,
                width=832,
                num_frames=49,
                max_steps=200,
                learning_rate=5e-5,
                rank=16,
                alpha=16,
                grad_accum=1,
                seed=26000,
                device="cuda",
                role="all",
                objective="target_prompt_teacher",
                balanced_roles=True,
                preserve_weight=4.0,
            )
            training_config = {
                "model": "models/Wan2.1-T2V-1.3B-Diffusers",
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
                "objective": "target_prompt_teacher",
                "balanced_roles": True,
                "preserve_weight": 4.0,
                "target_prompt_calibration_id": (
                    "lambda4_from_lambda1_first16_output_gradient_v1"
                ),
                "target_prompt_teacher_weight": 4.0,
                "sanity_mean_min": 0.2,
                "sanity_mean_max": 0.5,
                "sanity_single_max": 1.0,
            }
            registration = {
                "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_scale4_v1",
                "calibration_id": args.target_prompt_calibration_id,
                "output_dir": str(output_dir_relative),
                "target_prompt_teacher_weight": 4.0,
                "sanity_mean_min": 0.2,
                "sanity_mean_max": 0.5,
                "sanity_single_max": 1.0,
                "sanity_formula": (
                    "s_i = weight * sqrt(target_prompt_teacher_loss / flow_loss)"
                ),
                "sanity_aggregation": "arithmetic_mean_over_first_16_erase_steps",
                "selection_rule": "nearest_power_of_two(0.30 / mean_i(sqrt(r_i)))",
                "train_manifest_sha256": (
                    "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
                ),
                "base_cache_sha256": (
                    "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
                ),
                "teacher_cache_sha256": (
                    "6cf7ba0112d8df0e0a5253a7a943411fcfd85c56fc6df580446776b49d1ac9a9"
                ),
                "expected_initial_lora_sha256": (
                    "af163fcb6706c8403ffb1eaa9001cb2b9ac8ef86110663e8b20000961bb270a8"
                ),
                "lambda1_scale_invalid": True,
                "lambda1_generation_count": 0,
                "lambda1_mean_raw_loss_ratio_first_16": 0.005843,
                "training_config": training_config,
                "trainer_path": str(Path(module.__file__).resolve()),
                "trainer_sha256": "trainer-hash",
                "launcher_path": "scripts/run_water_impact_dynamic_sft_v3b_teacher.sh",
                "launcher_sha256": "launcher-hash",
                "protocol_doc_path": "docs/water_impact_dynamic_v3b_target_prompt_teacher.md",
                "protocol_doc_sha256": "protocol-hash",
                "lambda1_artifacts": [
                    {"path": path, "sha256": digest}
                    for path, digest in frozen_artifacts.items()
                ],
            }
            (output_dir / "run_registration.json").write_text(
                json.dumps(registration), encoding="utf-8"
            )

            def fake_hash(path: Path) -> str:
                resolved = Path(path).resolve()
                if resolved == Path(module.__file__).resolve():
                    return "trainer-hash"
                if resolved == launcher.resolve():
                    return "launcher-hash"
                if resolved == protocol_doc.resolve():
                    return "protocol-hash"
                relative = str(resolved.relative_to(root))
                if relative in frozen_artifacts:
                    return frozen_artifacts[relative]
                return "registration-hash"

            with mock.patch.object(module, "file_sha256", side_effect=fake_hash):
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    path, digest, payload = module.validate_target_prompt_run_registration(
                        args,
                        manifest_sha256=(
                            "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
                        ),
                        base_cache_sha256=(
                            "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
                        ),
                    )
                finally:
                    os.chdir(previous)
            self.assertEqual(path, output_dir_relative / "run_registration.json")
            self.assertEqual(digest, "registration-hash")
            self.assertEqual(payload["training_config"], training_config)

            args.learning_rate = 1e-4
            previous = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(ValueError, "outside the frozen protocol"):
                    module.validate_target_prompt_run_registration(
                        args,
                        manifest_sha256=(
                            "3d4d8cbf9244b1575357f0ac74380cd7cb6265df4d1a85bf450de4cac120aee4"
                        ),
                        base_cache_sha256=(
                            "4654ee67b8937d248963839b744e59f4f535f8be8b8eee421aef264fb3cd4d65"
                        ),
                    )
            finally:
                os.chdir(previous)

    def test_scale_sanity_uses_weighted_per_step_output_gradient_ratio(self) -> None:
        module = load_module()
        raw_ratios = [0.01] * 8 + [0.09] * 8
        observations = [
            {
                "global_step": 2 * index + 1,
                "scene_id": f"scene_{index}",
                "flow_loss": 0.1,
                "target_prompt_teacher_loss": 0.1 * raw_ratio,
                "raw_loss_ratio": raw_ratio,
                "weighted_loss_ratio": raw_ratio,
                "weighted_output_gradient_norm_ratio": raw_ratio**0.5,
            }
            for index, raw_ratio in enumerate(raw_ratios)
        ]
        with tempfile.TemporaryDirectory() as directory:
            payload = module.write_target_prompt_scale_sanity(
                Path(directory),
                observations,
                weight=1.0,
                mean_min=0.19,
                mean_max=0.21,
                single_max=1.0,
                calibration_id="fixture-calibration",
                run_registration_sha256="a" * 64,
            )

            self.assertTrue(payload["passed"])
            self.assertAlmostEqual(
                payload["mean_weighted_output_grad_ratio"], 0.2
            )
            self.assertAlmostEqual(payload["mean_weighted_loss_ratio"], 0.05)
            self.assertEqual(payload["calibration_id"], "fixture-calibration")
            self.assertEqual(payload["run_registration_sha256"], "a" * 64)
            self.assertTrue(
                (Path(directory) / "target_prompt_scale_sanity.json").is_file()
            )
            with self.assertRaises(FileExistsError):
                module.write_target_prompt_scale_sanity(
                    Path(directory),
                    observations,
                    weight=1.0,
                    mean_min=0.19,
                    mean_max=0.21,
                    single_max=1.0,
                    calibration_id="fixture-calibration",
                    run_registration_sha256="a" * 64,
                )

        with tempfile.TemporaryDirectory() as directory:
            payload = module.write_target_prompt_scale_sanity(
                Path(directory),
                observations,
                weight=4.0,
                mean_min=0.1,
                mean_max=1.0,
                single_max=1.0,
                calibration_id="fixture-calibration",
                run_registration_sha256="a" * 64,
            )
            self.assertFalse(payload["passed"])

        calibrated = module.target_prompt_scale_metrics([0.005843] * 16, 4.0)
        self.assertAlmostEqual(
            calibrated["mean_weighted_output_grad_ratio"],
            4.0 * 0.005843**0.5,
        )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            module.target_prompt_scale_metrics([float("nan")], 4.0)

    def test_pair_forward_restores_adapters_when_teacher_raises(self) -> None:
        module = load_module()

        class FailingTransformer:
            adapters_enabled = True

            def disable_adapters(self) -> None:
                self.adapters_enabled = False

            def enable_adapters(self) -> None:
                self.adapters_enabled = True

            def __call__(self, **kwargs):
                raise RuntimeError("teacher failed")

        transformer = FailingTransformer()
        with self.assertRaisesRegex(RuntimeError, "teacher failed"):
            module.forward_target_prompt_teacher_pair(
                transformer,
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
            )
        self.assertTrue(transformer.adapters_enabled)

    def test_pair_forward_uses_frozen_target_then_adapter_factual_on_same_state(self) -> None:
        module = load_module()

        class MockTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.adapters_enabled = True
                self.calls: list[tuple[bool, object, object, object]] = []

            def disable_adapters(self) -> None:
                self.adapters_enabled = False

            def enable_adapters(self) -> None:
                self.adapters_enabled = True

            def forward(
                self,
                *,
                hidden_states,
                timestep,
                encoder_hidden_states,
                return_dict,
            ):
                self.calls.append(
                    (
                        self.adapters_enabled,
                        hidden_states,
                        timestep,
                        encoder_hidden_states,
                    )
                )
                return (hidden_states * self.scale,)

        transformer = MockTransformer()
        noisy = torch.ones(2)
        timestep = torch.tensor([500.0])
        factual = torch.tensor([1.0])
        target = torch.tensor([2.0])

        teacher, student = module.forward_target_prompt_teacher_pair(
            transformer, noisy, timestep, factual, target
        )

        self.assertEqual(len(transformer.calls), 2)
        teacher_call, student_call = transformer.calls
        self.assertFalse(teacher_call[0])
        self.assertTrue(student_call[0])
        self.assertIs(teacher_call[1], noisy)
        self.assertIs(student_call[1], noisy)
        self.assertIs(teacher_call[2], timestep)
        self.assertIs(student_call[2], timestep)
        self.assertIs(teacher_call[3], target)
        self.assertIs(student_call[3], factual)
        self.assertFalse(teacher.requires_grad)
        self.assertTrue(student.requires_grad)
        self.assertTrue(transformer.adapters_enabled)

    def test_frozen_manifest_prompt_binding_is_stable(self) -> None:
        module = load_module()
        manifest = (
            Path(__file__).parents[1]
            / "data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv"
        )
        args = SimpleNamespace(
            manifest=manifest,
            role="all",
            objective="target_prompt_teacher",
            target_prompt_teacher_weight=1.0,
        )

        rows = module.load_rows(args)

        self.assertEqual(
            module.target_prompt_binding_sha256(rows),
            "9b1cc6e5bbdbe60b8f9f8378dc0ea11fea2fe82d8c73d6f9afed3f72b2bd00cc",
        )
        self.assertEqual(
            len({row["target_generation_prompt"] for row in rows if row["training_role"] == "erase"}),
            24,
        )

    def test_teacher_loss_stops_teacher_gradient_and_weight_zero_is_plain(self) -> None:
        module = load_module()
        prediction = torch.tensor([1.0, 2.0], requires_grad=True)
        teacher = torch.tensor([0.0, 0.0], requires_grad=True)
        flow = (prediction - 3.0).square().mean()

        teacher_loss, combined = module.combine_target_prompt_teacher_loss(
            flow, prediction, teacher, 1.0
        )
        combined.backward()

        self.assertGreater(teacher_loss.item(), 0)
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(teacher.grad)

        prediction2 = torch.tensor([1.0, 2.0], requires_grad=True)
        flow2 = (prediction2 - 3.0).square().mean()
        _, no_op = module.combine_target_prompt_teacher_loss(
            flow2, prediction2, teacher, 0.0
        )
        self.assertEqual(no_op.item(), flow2.item())

    def test_target_prompt_sidecar_validator_binds_payload_bytes(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("fixture\n", encoding="utf-8")
            cache_dir = root / "teacher"
            cache_dir.mkdir()
            model = root / "model"
            revision_path = (
                model / ".cache/huggingface/download/model_index.json.metadata"
            )
            revision_path.parent.mkdir(parents=True)
            revision_path.write_text(
                "0fad780a534b6463e45facd96134c9f345acfa5b\n", encoding="utf-8"
            )
            rows = [
                {
                    "scene_id": "scene",
                    "training_role": "erase",
                    "target_generation_prompt": "calm water",
                },
                {
                    "scene_id": "preserve",
                    "training_role": "preserve",
                    "target_generation_prompt": "",
                },
            ]
            embedding = torch.zeros((1, 226, 4096), dtype=torch.bfloat16)
            cache_path = cache_dir / "000_scene.pt"
            torch.save(
                {
                    "manifest_index": 0,
                    "scene_id": "scene",
                    "training_role": "erase",
                    "target_generation_prompt": "calm water",
                    "teacher_prompt_embeds": embedding,
                    "teacher_prompt_embeds_sha256": module.tensor_sha256(embedding),
                    "model": str(model),
                    "manifest_sha256": module.file_sha256(manifest),
                },
                cache_path,
            )
            inventory_hash = module.cache_inventory_sha256([cache_path])
            unique_embedding_digest = hashlib.sha256()
            unique_embedding_digest.update(b"calm water\0")
            unique_embedding_digest.update(module.tensor_sha256(embedding).encode("ascii"))
            unique_embedding_digest.update(b"\n")
            (cache_dir / "cache_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": "water_impact_dynamic_v3b_target_prompt_teacher_v1",
                        "source_manifest": str(manifest),
                        "source_manifest_sha256": module.file_sha256(manifest),
                        "model": str(model),
                        "dtype": "torch.bfloat16",
                        "do_classifier_free_guidance": False,
                        "max_sequence_length": 226,
                        "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
                        "erase_row_count": 1,
                        "unique_prompt_count": 1,
                        "prompt_binding_sha256": module.target_prompt_binding_sha256(rows),
                        "cache_inventory_sha256": inventory_hash,
                        "unique_embedding_sha256": unique_embedding_digest.hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                target_prompt_cache_dir=cache_dir,
                target_prompt_cache_sha256=inventory_hash,
                model=model,
                manifest=manifest,
            )

            paths, provenance = module.validate_target_prompt_cache(args, rows)

            self.assertEqual(paths, {0: cache_path})
            self.assertEqual(provenance["target_prompt_cache_entry_count"], 1)
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            payload["teacher_prompt_embeds"][0, 0, 0] = 1
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "teacher embedding hash mismatch"):
                module.validate_target_prompt_cache(args, rows)
            payload["teacher_prompt_embeds_sha256"] = module.tensor_sha256(
                payload["teacher_prompt_embeds"]
            )
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                module.validate_target_prompt_cache(args, rows)


if __name__ == "__main__":
    unittest.main()
