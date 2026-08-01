from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
