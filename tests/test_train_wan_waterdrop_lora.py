from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "train_wan_waterdrop_lora.py"
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


if __name__ == "__main__":
    unittest.main()
