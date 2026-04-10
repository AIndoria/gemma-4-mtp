from __future__ import annotations

import argparse

import torch

from gemma_mtp import infer_config_from_graph_export
from gemma_mtp.runtime_attention import (
    exact_attention_context,
    prepare_key_cache,
    prepare_value_cache,
    resolve_decode_position,
    resolve_local_window_bounds,
)


def sliced_attention_context(
    q: torch.Tensor,
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_pos: torch.Tensor,
    window_size: int,
    head_dim: int,
) -> torch.Tensor:
    context_size = key_cache.shape[-2]
    position = resolve_decode_position(input_pos, None, context_size)
    window_start, window_end = resolve_local_window_bounds(
        position,
        context_size=context_size,
        window_size=window_size,
    )
    k = prepare_key_cache(key_cache, window_start=window_start, window_end=window_end).to(q.dtype)
    v = prepare_value_cache(
        value_cache,
        window_start=window_start,
        window_end=window_end,
        head_dim=head_dim,
    ).to(q.dtype)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-json",
        default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json",
    )
    args = parser.parse_args()

    config = infer_config_from_graph_export(args.graph_json)
    layer0 = config.attention_specs[0]
    layer3 = config.attention_specs[3]

    q0 = torch.randn(1, layer0.kv_heads, layer0.queries_per_kv, layer0.query_head_dim)
    k22 = torch.randn(1, layer0.kv_heads, 32003, layer0.query_head_dim)
    v22 = torch.randn(1, layer0.kv_heads, layer0.query_head_dim, 32003)
    input_pos = torch.tensor([33], dtype=torch.int32)

    exact_local = exact_attention_context(
        q0,
        spec=layer0,
        key_cache=k22,
        value_cache=v22,
        input_pos=input_pos,
        mask=None,
        param_tensor=None,
    )
    sliced_local = sliced_attention_context(
        q0,
        key_cache=k22,
        value_cache=v22,
        input_pos=input_pos,
        window_size=layer0.local_window_size or 512,
        head_dim=layer0.query_head_dim,
    )

    q3 = torch.randn(1, layer3.kv_heads, layer3.queries_per_kv, layer3.query_head_dim)
    k23 = torch.randn(1, layer3.kv_heads, 32003, layer3.query_head_dim)
    v23 = torch.randn(1, layer3.kv_heads, layer3.query_head_dim, 32003)
    mask = torch.zeros(1, 1, 1, 32003, dtype=torch.bool)
    mask[..., :129] = True

    exact_full = exact_attention_context(
        q3,
        spec=layer3,
        key_cache=k23,
        value_cache=v23,
        input_pos=None,
        mask=mask,
        param_tensor=None,
    )

    k23_prepared = prepare_key_cache(k23, window_start=0, window_end=k23.shape[-2])
    v23_prepared = prepare_value_cache(
        v23,
        window_start=0,
        window_end=v23.shape[-1],
        head_dim=layer3.query_head_dim,
    )
    full_scores = torch.einsum("bhqd,bhkd->bhqk", q3, k23_prepared)
    full_scores = torch.where(mask, full_scores, torch.full_like(full_scores, -1.0e30))
    full_probs = torch.softmax(full_scores, dim=-1)
    reference_full = torch.einsum("bhqk,bhkd->bhqd", full_probs, v23_prepared)

    print("local_max_abs_diff", float((exact_local - sliced_local).abs().max()))
    print("full_max_abs_diff", float((exact_full - reference_full).abs().max()))


if __name__ == "__main__":
    main()
