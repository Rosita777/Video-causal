#!/usr/bin/env python3
"""Suppress selected text tokens for gated Wan cross-attention queries."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def find_token_mask(
    tokenizer,
    prompt: str,
    phrases: list[str],
    max_length: int = 512,
) -> torch.Tensor:
    encoded = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]
    mask = torch.zeros_like(encoded, dtype=torch.bool)
    ids = encoded.tolist()
    for phrase in phrases:
        phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if not phrase_ids:
            continue
        for start in range(len(ids) - len(phrase_ids) + 1):
            if ids[start : start + len(phrase_ids)] == phrase_ids:
                mask[start : start + len(phrase_ids)] = True
    if not mask.any():
        raise ValueError(f"No suppression phrase tokens found in prompt: {phrases}")
    return mask


class TargetTokenWanAttnProcessor:
    def __init__(self, controller: "TargetTokenAttentionController") -> None:
        self.controller = controller

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.add_k_proj is not None:
            raise NotImplementedError("Target-token suppression currently supports Wan T2V only")
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        bias = self.controller.attention_bias(
            batch=query.shape[0],
            query_tokens=query.shape[2],
            key_tokens=key.shape[2],
            device=query.device,
            dtype=query.dtype,
        )
        if attention_mask is not None:
            bias = attention_mask + bias
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class TargetTokenAttentionController:
    def __init__(self, transformer: torch.nn.Module, strength: float = 20.0) -> None:
        if strength <= 0:
            raise ValueError("Suppression strength must be positive")
        self.strength = float(strength)
        self._flat_gate: torch.Tensor | None = None
        self._token_mask: torch.Tensor | None = None
        self._original_processors = []
        for block in transformer.blocks:
            self._original_processors.append(block.attn2.processor)
            block.attn2.set_processor(TargetTokenWanAttnProcessor(self))

    def set_gate(self, gate: torch.Tensor) -> None:
        if gate.ndim == 3:
            gate = gate.unsqueeze(0)
        if gate.ndim != 4:
            raise ValueError(f"Expected gate [B,T,H,W], got {tuple(gate.shape)}")
        self._flat_gate = gate.flatten(1).clamp(0, 1)

    def set_token_mask(self, token_mask: torch.Tensor) -> None:
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        if token_mask.ndim != 2:
            raise ValueError(f"Expected token mask [B,L], got {tuple(token_mask.shape)}")
        self._token_mask = token_mask.bool()

    def attention_bias(
        self,
        *,
        batch: int,
        query_tokens: int,
        key_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._flat_gate is None or self._token_mask is None:
            raise RuntimeError("Attention gate and token mask must be set before generation")
        gate = self._flat_gate
        token_mask = self._token_mask
        if gate.shape[1] != query_tokens or token_mask.shape[1] != key_tokens:
            raise ValueError(
                f"Attention shape mismatch: gate={tuple(gate.shape)} token_mask={tuple(token_mask.shape)} "
                f"query={query_tokens} key={key_tokens}"
            )
        if gate.shape[0] == 1 and batch > 1:
            gate = gate.expand(batch, -1)
            token_mask = token_mask.expand(batch, -1)
        if gate.shape[0] != batch or token_mask.shape[0] != batch:
            raise ValueError(f"Batch mismatch for attention suppression: {batch}")
        return (
            -self.strength
            * gate[:, None, :, None].to(device=device, dtype=dtype)
            * token_mask[:, None, None, :].to(device=device, dtype=dtype)
        )

    def remove(self, transformer: torch.nn.Module) -> None:
        for block, processor in zip(transformer.blocks, self._original_processors, strict=True):
            block.attn2.set_processor(processor)
        self._original_processors.clear()
