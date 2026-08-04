#!/usr/bin/env python3
"""Spatially gate PEFT LoRA residuals while preserving the frozen base output."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class CausalLoRAActivationGate:
    """Apply ``base + gate * (adapter - base)`` to video-token LoRA layers."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        target_suffixes: Iterable[str] = ("to_q", "to_k", "to_v", "to_out.0"),
    ) -> None:
        self._flat_gate: torch.Tensor | None = None
        self._enabled = True
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        suffixes = tuple(target_suffixes)
        for name, module in transformer.named_modules():
            if name.endswith(suffixes) and hasattr(module, "base_layer"):
                self._handles.append(module.register_forward_hook(self._gate_output))
        if not self._handles:
            raise ValueError("No PEFT LoRA target modules were found for activation gating")

    @property
    def module_count(self) -> int:
        return len(self._handles)

    def set_gate(self, gate: torch.Tensor) -> None:
        if gate.ndim == 5:
            if gate.shape[1] != 1:
                raise ValueError(f"Expected a singleton gate channel, got {tuple(gate.shape)}")
            gate = gate[:, 0]
        if gate.ndim == 3:
            gate = gate.unsqueeze(0)
        if gate.ndim != 4:
            raise ValueError(f"Expected gate shape [B,T,H,W], got {tuple(gate.shape)}")
        self._flat_gate = gate.flatten(1).unsqueeze(-1).clamp(0.0, 1.0)

    def clear_gate(self) -> None:
        self._flat_gate = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_gate()

    def _gate_output(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._enabled or self._flat_gate is None or not torch.is_tensor(output):
            return None
        if output.ndim != 3 or not inputs or not torch.is_tensor(inputs[0]):
            return None
        gate = self._flat_gate
        if gate.shape[0] == 1 and output.shape[0] > 1:
            gate = gate.expand(output.shape[0], -1, -1)
        if gate.shape[:2] != output.shape[:2]:
            # Cross-attention K/V project text tokens and must remain ungated.
            return None
        base_layer = getattr(module, "base_layer")
        base_output = base_layer(inputs[0])
        gate = gate.to(device=output.device, dtype=output.dtype)
        return base_output + gate * (output - base_output)
