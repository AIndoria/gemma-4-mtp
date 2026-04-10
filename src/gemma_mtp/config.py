from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSpec:
    layer_index: int
    query_heads: int
    query_head_dim: int
    rope_base: float = 10000.0
    kv_heads: int = 0
    queries_per_kv: int = 0
    key_cache_name: str | None = None
    value_cache_name: str | None = None
    local_window_size: int | None = None
    source_kv_cache: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class MtpDrafterConfig:
    input_activation_dim: int = 5120
    projected_activation_dim: int = 2560
    model_dim: int = 256
    mlp_hidden_dim: int = 2048
    num_layers: int = 4
    vocab_size: int = 262144
    attention_specs: tuple[AttentionSpec, ...] = ()
    base_kv_inputs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
