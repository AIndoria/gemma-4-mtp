from __future__ import annotations

import argparse

import numpy as np
import torch

from gemma_mtp import (
    GemmaMtpDrafter,
    TFLiteModelReader,
    build_partial_state_dict,
    hydrate_runtime_quantization,
    infer_config_from_graph_export,
)
from gemma_mtp.runtime_attention import (
    apply_query_rope,
    prepare_key_cache,
    prepare_value_cache,
    reshape_grouped_query,
    resolve_cache_tensor,
)


PROBS_TENSOR = (
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/"
    "dot_attn/dot_attn._qkv_fn/div;layer_0/attn.dot_product_attention/"
    "attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/exp;"
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/"
    "dot_attn/dot_attn._qkv_fn/broadcast_in_dim;layer_0/attn.dot_product_attention/"
    "attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/reduce_sum;"
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/"
    "dot_attn/dot_attn._qkv_fn/reduce_max;layer_0/attn.dot_product_attention/"
    "attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/max;"
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/"
    "dot_attn/dot_attn._qkv_fn/sub"
)
CTX_TENSOR = (
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/"
    "dot_attn/dot_attn._qkv_fn/composite1"
)


def _build_model(graph_json: str, tflite_model: str) -> GemmaMtpDrafter:
    config = infer_config_from_graph_export(graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(graph_json, tflite_model)
    model.load_state_dict(state_dict, strict=False)
    hydrate_runtime_quantization(model, graph_json, tflite_model)
    return model


def _build_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23"):
        cache[name] = {
            "tensor": torch.from_numpy(inputs[name].copy()),
            "scale": float(reader.tensor_info(f"mtp_drafter_{name}:0").quantization.scale[0]),
            "zero_point": 0,
        }
    return cache


def _compute_query(
    model: GemmaMtpDrafter,
    inputs: dict[str, np.ndarray],
) -> torch.Tensor:
    block = model.blocks[0]
    with torch.no_grad():
        activations = torch.from_numpy(inputs["activations"].copy())
        input_pos = torch.from_numpy(inputs["input_pos"].copy())
        param_tensor = torch.from_numpy(inputs["param_tensor"].copy())
        pre_project = model.pre_project(activations)
        pre_attn = block.pre_attn_norm(pre_project)
        q = block.attention.q_proj(pre_attn).view(1, 1, 4, 256)
        q = block.attention.query_norm(q)
        q = apply_query_rope(
            q,
            spec=block.attention.spec,
            input_pos=input_pos,
            param_tensor=param_tensor,
        )
    return reshape_grouped_query(q.reshape(1, 1, -1), spec=block.attention.spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-json",
        default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json",
    )
    parser.add_argument(
        "--tflite-model",
        default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite",
    )
    parser.add_argument("--position", type=int, default=700)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    inputs = np.load(f"data/derived/tflite_inputs_{args.position}.npz")
    outputs = np.load(f"data/derived/tflite_outputs_{args.position}.npz")
    model = _build_model(args.graph_json, args.tflite_model)
    reader = TFLiteModelReader(args.tflite_model)
    cache = _build_cache(reader, inputs)
    block = model.blocks[0]

    q = _compute_query(model, inputs)
    key_cache = resolve_cache_tensor(cache, block.attention.spec.key_cache_name or "")
    value_cache = resolve_cache_tensor(cache, block.attention.spec.value_cache_name or "")
    logical_start = max(0, args.position + 1 - 512)
    logical_end = args.position + 1
    window_len = logical_end - logical_start

    target_probs = torch.from_numpy(outputs[PROBS_TENSOR])
    target_probs_window = target_probs[..., logical_start:logical_end]
    target_ctx = torch.from_numpy(outputs[CTX_TENSOR])

    key_results: list[tuple[float, int]] = []
    max_offset = key_cache.shape[-2] - window_len
    for offset in range(0, max_offset + 1):
        k = prepare_key_cache(key_cache, window_start=offset, window_end=offset + window_len).to(q.dtype)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
        full_scores = torch.full(
            (q.shape[0], q.shape[1], q.shape[2], key_cache.shape[-2]),
            -1.0e30,
            dtype=q.dtype,
        )
        full_scores[..., logical_start:logical_end] = scores
        probs = torch.softmax(full_scores, dim=-1)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                probs.flatten().float(),
                target_probs.flatten().float(),
                dim=0,
            )
        )
        key_results.append((cosine, offset))
    key_results.sort(reverse=True)

    best_key_offset = key_results[0][1]
    k = prepare_key_cache(
        key_cache,
        window_start=best_key_offset,
        window_end=best_key_offset + window_len,
    ).to(q.dtype)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    probs = torch.softmax(scores, dim=-1)

    value_results: list[tuple[float, int]] = []
    for offset in range(0, value_cache.shape[-1] - window_len + 1):
        v = prepare_value_cache(
            value_cache,
            window_start=offset,
            window_end=offset + window_len,
            head_dim=block.attention.spec.query_head_dim,
        ).to(q.dtype)
        ctx = torch.einsum("bhqk,bhkd->bhqd", probs, v)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                ctx.flatten().float(),
                target_ctx.flatten().float(),
                dim=0,
            )
        )
        value_results.append((cosine, offset))
    value_results.sort(reverse=True)

    t_probs_support = (target_probs_window > 1e-8).nonzero().shape[0]
    print("position", args.position)
    print("logical_window", logical_start, logical_end)
    print("target_probs_support", t_probs_support)
    print("best_key_offsets", key_results[: args.top_k])
    print("best_value_offsets", value_results[: args.top_k])


if __name__ == "__main__":
    main()
