from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "causal_lora_activation_gate.py"
    spec = importlib.util.spec_from_file_location("causal_lora_activation_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLoraLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = torch.nn.Linear(2, 2, bias=False)
        self.adapter = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.eye_(self.base_layer.weight)
        torch.nn.init.constant_(self.adapter.weight, 1.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base_layer(values) + self.adapter(values)


class FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = FakeLoraLinear()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.to_q(values)


class FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_k = FakeLoraLinear()
        self.to_v = FakeLoraLinear()


class FakeCrossAttentionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = FakeAttention()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.attn2.to_k(values)


class FakeConditionalTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = FakeLoraLinear()

    def forward(
        self,
        values: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del encoder_hidden_states
        return self.to_q(values)


class CausalLoRAActivationGateTest(unittest.TestCase):
    def test_temporal_persistence_keeps_spatial_union_after_first_event(self) -> None:
        module = load_module()
        gate = torch.tensor(
            [[[[0.0, 0.0, 0.0]], [[0.0, 0.5, 0.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]]]
        )

        persistent = module.make_temporally_persistent_gate(gate)

        expected = torch.tensor(
            [[[[0.0, 0.0, 0.0]], [[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]]]
        )
        torch.testing.assert_close(persistent, expected)

    def test_gates_adapter_residual_per_token(self) -> None:
        module = load_module()
        transformer = FakeTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        controller.set_gate(torch.tensor([[[[0.0, 1.0]]]]))

        output = transformer(values)

        expected = torch.tensor([[[1.0, 2.0], [10.0, 11.0]]])
        torch.testing.assert_close(output, expected)

    def test_zero_gate_blocks_adapter_gradient(self) -> None:
        module = load_module()
        transformer = FakeTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        controller.set_gate(torch.zeros((1, 1, 1, 2)))
        values = torch.ones((1, 2, 2), requires_grad=True)

        transformer(values).sum().backward()

        torch.testing.assert_close(
            transformer.to_q.adapter.weight.grad,
            torch.zeros_like(transformer.to_q.adapter.weight.grad),
        )
        self.assertGreater(values.grad.abs().sum().item(), 0.0)

    def test_empty_video_gate_disables_mismatched_text_tokens(self) -> None:
        module = load_module()
        transformer = FakeTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        controller.set_gate(torch.zeros((1, 1, 1, 2)))
        text_values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        output = transformer(text_values)

        torch.testing.assert_close(output, text_values)

    def test_target_text_gate_only_changes_selected_cross_attention_token(self) -> None:
        module = load_module()
        transformer = FakeCrossAttentionTransformer()
        controller = module.CausalLoRAActivationGate(transformer)
        controller.set_gate(torch.ones((1, 1, 1, 2)))
        controller.set_text_gate(torch.tensor([0.0, 1.0, 0.0]))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        output = transformer(values)

        expected = values.clone()
        expected[:, 1] = torch.tensor([[10.0, 11.0]])
        torch.testing.assert_close(output, expected)

    def test_empty_text_gate_disables_video_lora_path(self) -> None:
        module = load_module()
        transformer = FakeTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        controller.set_gate(torch.ones((1, 1, 1, 2)))
        controller.set_text_gate(torch.zeros((1, 3)))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        output = transformer(values)

        torch.testing.assert_close(output, values)

    def test_missing_target_embedding_disables_conditional_video_path(self) -> None:
        module = load_module()
        transformer = FakeConditionalTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        controller.set_gate(torch.ones((1, 1, 1, 2)))
        controller.set_text_gate(torch.tensor([0.0, 1.0, 0.0]))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        empty_prompt = torch.zeros((1, 3, 4))

        output = transformer(values, encoder_hidden_states=empty_prompt)

        torch.testing.assert_close(output, values)

    def test_present_target_embedding_enables_conditional_video_path(self) -> None:
        module = load_module()
        transformer = FakeConditionalTransformer()
        controller = module.CausalLoRAActivationGate(transformer, target_suffixes=("to_q",))
        controller.set_gate(torch.ones((1, 1, 1, 2)))
        controller.set_text_gate(torch.tensor([0.0, 1.0, 0.0]))
        values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target_prompt = torch.zeros((1, 3, 4))
        target_prompt[:, 1] = 1.0

        output = transformer(values, encoder_hidden_states=target_prompt)

        expected = torch.tensor([[[4.0, 5.0], [10.0, 11.0]]])
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
