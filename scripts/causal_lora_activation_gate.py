#!/usr/bin/env python3
"""Spatially gate PEFT LoRA residuals while preserving the frozen base output."""

from __future__ import annotations

from collections.abc import Iterable

import torch


class CausalLoRAActivationGate:
    """Gate LoRA residuals by video region and optional target-text tokens."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        target_suffixes: Iterable[str] = ("to_q", "to_k", "to_v", "to_out.0"),
    ) -> None:
        self._flat_gate: torch.Tensor | None = None
        self._text_gate: torch.Tensor | None = None
        self._context_switch: torch.Tensor | None = None
        self._enabled = True
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._context_handle = transformer.register_forward_pre_hook(
            self._capture_text_condition,
            with_kwargs=True,
        )
        suffixes = tuple(target_suffixes)
        for name, module in transformer.named_modules():
            if name.endswith(suffixes) and hasattr(module, "base_layer"):
                is_cross_attention_text = name.endswith(("attn2.to_k", "attn2.to_v"))
                self._handles.append(
                    module.register_forward_hook(
                        self._make_gate_hook(is_cross_attention_text)
                    )
                )
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

    def set_text_gate(self, gate: torch.Tensor) -> None:
        if gate.ndim == 1:
            gate = gate.unsqueeze(0)
        if gate.ndim == 2:
            gate = gate.unsqueeze(-1)
        if gate.ndim != 3 or gate.shape[-1] != 1:
            raise ValueError(f"Expected text gate shape [B,L] or [B,L,1], got {tuple(gate.shape)}")
        self._text_gate = gate.clamp(0.0, 1.0)

    def clear_gate(self) -> None:
        self._flat_gate = None
        self._text_gate = None
        self._context_switch = None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context_handle.remove()
        self.clear_gate()

    def _capture_text_condition(self, module, args, kwargs) -> None:
        del module, args
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if self._text_gate is None or not torch.is_tensor(encoder_hidden_states):
            self._context_switch = None
            return
        text_gate = self._text_gate.to(device=encoder_hidden_states.device)
        self._text_gate = text_gate
        if text_gate.shape[0] == 1 and encoder_hidden_states.shape[0] > 1:
            text_gate = text_gate.expand(encoder_hidden_states.shape[0], -1, -1)
        if text_gate.shape[:2] != encoder_hidden_states.shape[:2]:
            raise ValueError(
                f"Text gate shape {tuple(text_gate.shape[:2])} does not match "
                f"encoder shape {tuple(encoder_hidden_states.shape[:2])}"
            )
        selected = encoder_hidden_states.float() * text_gate.to(
            device=encoder_hidden_states.device,
            dtype=torch.float32,
        )
        self._context_switch = (selected.abs().sum(dim=(1, 2)) > 0).to(
            dtype=torch.float32
        )[:, None, None]

    def _make_gate_hook(self, is_cross_attention_text: bool):
        def hook(module, inputs, output):
            return self._gate_output(
                module,
                inputs,
                output,
                is_cross_attention_text=is_cross_attention_text,
            )

        return hook

    def _gate_output(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        *,
        is_cross_attention_text: bool,
    ) -> torch.Tensor | None:
        if not self._enabled or not torch.is_tensor(output):
            return None
        if output.ndim != 3 or not inputs or not torch.is_tensor(inputs[0]):
            return None
        if self._flat_gate is None and self._text_gate is None:
            return None

        text_gate = self._text_gate
        if text_gate is not None and text_gate.shape[0] == 1 and output.shape[0] > 1:
            text_gate = text_gate.expand(output.shape[0], -1, -1)
        if text_gate is not None and text_gate.shape[0] != output.shape[0]:
            return None
        phrase_switch = (
            text_gate.amax(dim=1, keepdim=True) if text_gate is not None else None
        )
        if phrase_switch is not None and self._context_switch is not None:
            phrase_switch = phrase_switch * self._context_switch.to(
                device=phrase_switch.device,
                dtype=phrase_switch.dtype,
            )

        if is_cross_attention_text and text_gate is not None:
            if text_gate.shape[1] != output.shape[1]:
                raise ValueError(
                    f"Text gate length {text_gate.shape[1]} does not match "
                    f"cross-attention length {output.shape[1]}"
                )
            gate = text_gate
            if self._context_switch is not None:
                gate = gate * self._context_switch.to(
                    device=gate.device,
                    dtype=gate.dtype,
                )
            if self._flat_gate is not None:
                video_switch = self._flat_gate.amax(dim=1, keepdim=True).to(
                    device=gate.device,
                    dtype=gate.dtype,
                )
                gate = gate * video_switch
        elif self._flat_gate is not None:
            gate = self._flat_gate
            if phrase_switch is not None:
                gate = gate.to(
                    device=phrase_switch.device,
                    dtype=phrase_switch.dtype,
                )
                gate = gate * phrase_switch
        else:
            gate = phrase_switch.expand(-1, output.shape[1], -1)

        if gate.shape[0] == 1 and output.shape[0] > 1:
            gate = gate.expand(output.shape[0], -1, -1)
        if gate.shape[0] != output.shape[0]:
            return None
        if gate.shape[1] != output.shape[1]:
            # Text K/V have no video-grid position. Use a per-sample switch so
            # an empty video gate still disables every LoRA path exactly.
            gate = gate.amax(dim=1, keepdim=True).expand(-1, output.shape[1], -1)
        base_layer = getattr(module, "base_layer")
        base_output = base_layer(inputs[0])
        gate = gate.to(device=output.device, dtype=output.dtype)
        return base_output + gate * (output - base_output)
