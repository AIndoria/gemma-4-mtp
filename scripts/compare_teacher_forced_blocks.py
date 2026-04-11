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
    exact_attention_context,
    reshape_grouped_query,
    resolve_cache_tensor,
)


def _compare(name: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (pred - ref).abs()
    cosine = torch.nn.functional.cosine_similarity(
        pred.flatten().float(),
        ref.flatten().float(),
        dim=0,
    )
    print(
        f"{name: <32} mean_abs_diff {float(diff.mean()):.6f} "
        f"max_abs_diff {float(diff.max()):.6f} cosine_similarity {float(cosine):.6f}"
    )


def _build_model(graph_json: str, tflite_model: str) -> GemmaMtpDrafter:
    config = infer_config_from_graph_export(graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(graph_json, tflite_model)
    model.load_state_dict(state_dict, strict=False)
    hydrate_runtime_quantization(model, graph_json, tflite_model)
    return model


def _build_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "tensor": torch.from_numpy(inputs[name].copy()),
            "scale": float(reader.tensor_info(f"mtp_drafter_{name}").quantization.scale[0]),
            "zero_point": 0,
        }
        for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23")
    }


def _teacher_forced_block_outputs(
    model: GemmaMtpDrafter,
    layer_index: int,
    hidden_states: torch.Tensor,
    *,
    mask: torch.Tensor,
    cache: dict[str, dict[str, object]],
    input_pos: torch.Tensor,
    param_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block = model.blocks[layer_index]
    head_dim = block.attention.spec.query_head_dim

    pre_attn = block.pre_attn_norm(hidden_states)
    q = block.attention.q_proj(pre_attn).view(1, 1, 4, head_dim)
    q = block.attention.query_norm(q)
    q = apply_query_rope(
        q,
        spec=block.attention.spec,
        input_pos=input_pos,
        param_tensor=param_tensor,
    )
    q_grouped = reshape_grouped_query(q.reshape(1, 1, -1), spec=block.attention.spec)
    context = exact_attention_context(
        q_grouped,
        spec=block.attention.spec,
        key_cache=resolve_cache_tensor(cache, block.attention.spec.key_cache_name or ""),
        value_cache=resolve_cache_tensor(cache, block.attention.spec.value_cache_name or ""),
        input_pos=input_pos,
        mask=mask,
        param_tensor=param_tensor,
    )
    attn_out = block.attention.o_proj_forward(context.reshape(1, 1, -1))
    block_out = block(
        hidden_states,
        mask=mask,
        base_kv_cache=cache,
        input_pos=input_pos,
        param_tensor=param_tensor,
    )
    return q, context, attn_out, block_out


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
    parser.add_argument(
        "--inputs",
        default="data/derived/tflite_compare_inputs_700.npz",
    )
    parser.add_argument(
        "--outputs",
        default="data/derived/tflite_compare_outputs_700_deep.npz",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    args = parser.parse_args()

    model = _build_model(args.graph_json, args.tflite_model)
    inputs = np.load(args.inputs)
    outputs = np.load(args.outputs)
    reader = TFLiteModelReader(args.tflite_model)
    cache = _build_cache(reader, inputs)

    mask = torch.from_numpy(inputs["mask"].copy())
    input_pos = torch.from_numpy(inputs["input_pos"].copy())
    param_tensor = torch.from_numpy(inputs["param_tensor"].copy())

    for layer_index in args.layers:
        prev_key = f"layer_{layer_index - 1}/layer_{layer_index - 1}.post_qkv/add1"
        if prev_key not in outputs:
            print(f"skip layer_{layer_index}: missing previous tensor {prev_key}")
            continue

        hidden_states = torch.from_numpy(outputs[prev_key])
        q, context, attn_out, block_out = _teacher_forced_block_outputs(
            model,
            layer_index,
            hidden_states,
            mask=mask,
            cache=cache,
            input_pos=input_pos,
            param_tensor=param_tensor,
        )

        print(f"\n--- layer_{layer_index} teacher-forced ---")
        q_key = (
            f"layer_{layer_index}/layer_{layer_index}.pre_q/attn.pre_q/"
            "attn._pre_attention_query_fn/maybe_rope/concatenate"
        )
        ctx_key = (
            f"layer_{layer_index}/attn.dot_product_attention/"
            "attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"
        )
        attn_out_key = (
            f"layer_{layer_index}/layer_{layer_index}.post_qkv/attn.post_qkv/"
            "attn_vec_einsum/composite1"
        )
        block_key = f"layer_{layer_index}/layer_{layer_index}.post_qkv/add1"

        if q_key in outputs:
            _compare(f"layer_{layer_index}.query_rope", q, torch.from_numpy(outputs[q_key]))
        if ctx_key in outputs:
            _compare(f"layer_{layer_index}.attn_context", context, torch.from_numpy(outputs[ctx_key]))
        if attn_out_key in outputs:
            _compare(f"layer_{layer_index}.attn_out", attn_out, torch.from_numpy(outputs[attn_out_key]))
        if block_key in outputs:
            _compare(f"layer_{layer_index}.block_out", block_out, torch.from_numpy(outputs[block_key]))


if __name__ == "__main__":
    main()
