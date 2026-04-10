from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSpec:
    layer_index: int
    query_heads: int
    query_head_dim: int
    source_kv_cache: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class MtpDrafterConfig:
    input_activation_dim: int
    projected_activation_dim: int
    model_dim: int
    mlp_hidden_dim: int
    num_layers: int
    vocab_size: int
    attention_specs: tuple[AttentionSpec, ...]
    base_kv_inputs: tuple[str, ...]
    notes: tuple[str, ...] = ()
