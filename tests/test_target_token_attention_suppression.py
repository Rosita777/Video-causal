from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "target_token_attention_suppression.py"
    spec = importlib.util.spec_from_file_location("target_token_attention_suppression", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeAttention:
    def __init__(self) -> None:
        self.processor = object()

    def set_processor(self, processor) -> None:
        self.processor = processor


class FakeBlock:
    def __init__(self) -> None:
        self.attn2 = FakeAttention()


class FakeTransformer:
    def __init__(self) -> None:
        self.blocks = [FakeBlock()]


class TargetTokenAttentionSuppressionTest(unittest.TestCase):
    def test_bias_only_intersects_gate_and_target_tokens(self) -> None:
        module = load_module()
        controller = module.TargetTokenAttentionController(FakeTransformer(), strength=7.0)
        controller.set_gate(torch.tensor([[[[0.0, 1.0]]]]))
        controller.set_token_mask(torch.tensor([False, True, False]))

        bias = controller.attention_bias(
            batch=1,
            query_tokens=2,
            key_tokens=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        expected = torch.zeros((1, 1, 2, 3))
        expected[0, 0, 1, 1] = -7.0
        torch.testing.assert_close(bias, expected)

    def test_single_gate_expands_across_batch(self) -> None:
        module = load_module()
        controller = module.TargetTokenAttentionController(FakeTransformer(), strength=5.0)
        controller.set_gate(torch.ones((1, 1, 1, 2)))
        controller.set_token_mask(torch.tensor([False, True, False]))

        bias = controller.attention_bias(
            batch=2,
            query_tokens=2,
            key_tokens=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(float(bias[0, 0, :, 1].sum()), -10.0)
        self.assertEqual(float(bias[1, 0, :, 1].sum()), -10.0)


if __name__ == "__main__":
    unittest.main()
