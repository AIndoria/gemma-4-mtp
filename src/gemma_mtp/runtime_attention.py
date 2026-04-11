from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .config import AttentionSpec


NEG_INF = -1.0e30


def resolve_decode_position(
    input_pos: Tensor | None,
    param_tensor: Tensor | None,
    context_size: int,
) -> int:
    if input_pos is not None:
        flat = input_pos.reshape(-1)
        if flat.numel():
            return max(0, min(int(flat[-1].item()), context_size - 1))

    if param_tensor is not None:
        flat = param_tensor.reshape(-1)
        if flat.numel() >= 3:
            end_index = int(flat[2].item())
            return max(0, min(end_index - 1, context_size - 1))

    return max(0, context_size - 1)


def resolve_local_window_bounds(
    position: int,
    *,
    context_size: int,
    window_size: int,
    param_tensor: Tensor | None = None,
) -> tuple[int, int]:
    start = max(0, position + 1 - window_size)
    end = min(context_size, start + window_size)
    return start, end


def resolve_mask_bounds(
    position: int,
    *,
    context_size: int,
    window_size: int,
) -> tuple[int, int]:
    # The mask slides with the position: [pos - 511, pos]
    start = max(0, position + 1 - window_size)
    return start, position + 1


def build_local_window_mask(
    position: int,
    *,
    context_size: int,
    window_size: int,
    device: torch.device,
    param_tensor: Tensor | None = None,
) -> Tensor:
    indices = torch.arange(context_size, device=device, dtype=torch.int64).view(1, 1, 1, -1)
    start, end = resolve_mask_bounds(position, context_size=context_size, window_size=window_size)
    # The mask should be true for indices in [start, end)
    # AND indices <= position (causal)
    in_window = (indices >= start) & (indices < end)
    causal = indices <= position
    return in_window & causal


def _maybe_dequantize_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        if value.is_floating_point():
            return value
        raise ValueError(
            "Quantized cache tensors must be passed as a mapping with tensor/scale metadata."
        )

    if isinstance(value, tuple) and len(value) in (2, 3):
        tensor = value[0]
        scale = value[1]
        zero_point = value[2] if len(value) == 3 else 0
    elif isinstance(value, dict):
        tensor = value["tensor"]
        scale = value["scale"]
        zero_point = value.get("zero_point", 0)
    else:
        raise TypeError(f"Unsupported cache entry type: {type(value)!r}")

    if not isinstance(tensor, Tensor):
        raise TypeError("Quantized cache entry must contain a torch.Tensor under `tensor`.")

    return (tensor.to(torch.float32) - float(zero_point)) * float(scale)


def resolve_cache_tensor(base_kv_cache: dict[str, Any], name: str) -> Tensor:
    if name not in base_kv_cache:
        raise KeyError(f"Missing KV cache entry: {name}")
    return _maybe_dequantize_tensor(base_kv_cache[name])


def prepare_key_cache(
    key_cache: Tensor,
    *,
    window_start: int,
    window_end: int,
) -> Tensor:
    if key_cache.dim() != 4:
        raise ValueError(f"Expected 4D key cache, got shape {tuple(key_cache.shape)}")
    return key_cache[..., window_start:window_end, :]


def prepare_value_cache(
    value_cache: Tensor,
    *,
    window_start: int,
    window_end: int,
    head_dim: int,
) -> Tensor:
    if value_cache.dim() != 4:
        raise ValueError(f"Expected 4D value cache, got shape {tuple(value_cache.shape)}")
    # TFLite layout is [B, H, D, C].
    if value_cache.shape[-2] == head_dim:
        return value_cache[..., window_start:window_end].transpose(-1, -2)
    return value_cache[..., window_start:window_end, :]


def reshape_grouped_query(
    q: Tensor,
    *,
    spec: AttentionSpec,
) -> Tensor:
    batch_size, steps, _ = q.shape
    q = q.view(batch_size, steps, spec.query_heads, spec.query_head_dim)
    q = q.view(batch_size, steps, spec.kv_heads, spec.queries_per_kv, spec.query_head_dim)
    return q.permute(0, 2, 1, 3, 4).reshape(
        batch_size,
        spec.kv_heads,
        steps * spec.queries_per_kv,
        spec.query_head_dim,
    )


def apply_query_rope(
    q: Tensor,
    *,
    spec: AttentionSpec,
    input_pos: Tensor | None = None,
    param_tensor: Tensor | None = None,
) -> Tensor:
    if q.dim() != 4:
        raise ValueError(f"Expected query tensor shaped [B, T, H, D], got {tuple(q.shape)}")
    if spec.query_head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dim, got {spec.query_head_dim}")

    position = resolve_decode_position(input_pos, param_tensor, context_size=32003)
    half_dim = spec.query_head_dim // 2
    rotary_dims = spec.rope_rotary_dims or spec.query_head_dim
    if rotary_dims % 2 != 0:
        raise ValueError(f"RoPE rotary dims must be even, got {rotary_dims}")
    if rotary_dims > spec.query_head_dim:
        raise ValueError(
            f"RoPE rotary dims {rotary_dims} exceed head dim {spec.query_head_dim}"
        )
    active_half_dim = rotary_dims // 2

    # Layer 1 is unusually sensitive to long-context RoPE angle error.
    # Preserved LiteRT tensors match better when its deeper decode positions
    # use float64 trig, while shallower positions still prefer float32.
    angle_dtype = (
        torch.float64
        if spec.layer_index == 1 and position >= 1200
        else torch.float32
    )
    exponent = torch.arange(active_half_dim, device=q.device, dtype=angle_dtype) / half_dim
    inv_freq = torch.pow(torch.tensor(spec.rope_base, device=q.device, dtype=angle_dtype), -exponent)
    angles = inv_freq * float(position)
    cos = torch.cos(angles).view(1, 1, 1, active_half_dim).to(dtype=q.dtype)
    sin = torch.sin(angles).view(1, 1, 1, active_half_dim).to(dtype=q.dtype)

    q1, q2 = q[..., :half_dim].clone(), q[..., half_dim:].clone()
    q1_active = q1[..., :active_half_dim].clone()
    q2_active = q2[..., :active_half_dim].clone()
    q1[..., :active_half_dim] = q1_active * cos - q2_active * sin
    q2[..., :active_half_dim] = q1_active * sin + q2_active * cos
    return torch.cat([q1, q2], dim=-1)


def exact_attention_context(
    q: Tensor,
    *,
    spec: AttentionSpec,
    key_cache: Tensor,
    value_cache: Tensor,
    input_pos: Tensor | None = None,
    mask: Tensor | None = None,
    param_tensor: Tensor | None = None,
    fill_value: float = NEG_INF,
) -> Tensor:
    context_size = key_cache.shape[-2]
    if spec.local_window_size is not None:
        position = resolve_decode_position(input_pos, param_tensor, context_size)
        window_start, window_end = resolve_local_window_bounds(
            position,
            context_size=context_size,
            window_size=spec.local_window_size,
            param_tensor=param_tensor,
        )
        local_mask = build_local_window_mask(
            position,
            context_size=context_size,
            window_size=spec.local_window_size,
            device=q.device,
            param_tensor=param_tensor,
        )
    else:
        local_mask = torch.ones((1, 1, 1, context_size), dtype=torch.bool, device=q.device)
        window_start, window_end = 0, context_size

    if mask is not None and spec.local_window_size is None:
        effective_mask = mask.to(dtype=torch.bool, device=q.device)
    else:
        effective_mask = local_mask

    k = prepare_key_cache(key_cache, window_start=window_start, window_end=window_end).to(q.dtype)
    v = prepare_value_cache(
        value_cache,
        window_start=window_start,
        window_end=window_end,
        head_dim=spec.query_head_dim,
    ).to(q.dtype)

    local_scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    full_scores = torch.full(
        (q.shape[0], q.shape[1], q.shape[2], context_size),
        fill_value,
        dtype=q.dtype,
        device=q.device,
    )
    full_scores[..., window_start:window_end] = local_scores

    if effective_mask is not None:
        full_scores = torch.where(
            effective_mask,
            full_scores,
            torch.full_like(full_scores, fill_value),
        )

    probs = torch.softmax(full_scores, dim=-1)
    probs_window = probs[..., window_start:window_end]
    context = torch.einsum("bhqk,bhkd->bhqd", probs_window, v)
    return context
