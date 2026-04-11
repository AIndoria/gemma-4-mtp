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


def _compute_block0_context(
    model: GemmaMtpDrafter,
    inputs: dict[str, np.ndarray],
    cache: dict[str, dict[str, object]],
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
        q = reshape_grouped_query(q.reshape(1, 1, -1), spec=block.attention.spec)
        context = exact_attention_context(
            q,
            spec=block.attention.spec,
            key_cache=resolve_cache_tensor(cache, block.attention.spec.key_cache_name or ""),
            value_cache=resolve_cache_tensor(cache, block.attention.spec.value_cache_name or ""),
            input_pos=input_pos,
            param_tensor=param_tensor,
        )
    return context.reshape(1, 1, -1)


def _report_stats(pos: int, mine: torch.Tensor, ref: torch.Tensor) -> None:
    delta = mine - ref
    exact = (delta == 0)
    off_by_one = delta.abs() <= 1
    print(f"\n--- Position {pos} ---")
    print("exact_bin_match", float(exact.float().mean()))
    print("off_by_one_match", float(off_by_one.float().mean()))
    print("mean_abs_bin_delta", float(delta.abs().float().mean()))
    print("max_abs_bin_delta", int(delta.abs().max()))
    print("nonzero_ref", int((ref != 0).sum()))
    print("nonzero_mine", int((mine != 0).sum()))

    worst = torch.topk(delta.abs().reshape(-1).float(), k=8)
    for rank, flat_index in enumerate(worst.indices.tolist(), start=1):
        mine_val = int(mine.reshape(-1)[flat_index])
        ref_val = int(ref.reshape(-1)[flat_index])
        print(
            f"worst_{rank}_index {flat_index} mine {mine_val} ref {ref_val} delta {mine_val - ref_val}"
        )


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
        "--positions",
        nargs="+",
        type=int,
        default=[100, 700],
    )
    args = parser.parse_args()

    model = _build_model(args.graph_json, args.tflite_model)
    reader = TFLiteModelReader(args.tflite_model)
    block = model.blocks[0]

    for pos in args.positions:
        inputs = np.load(f"data/derived/tflite_inputs_{pos}.npz")
        outputs = np.load(f"data/derived/tflite_outputs_{pos}.npz")
        cache = _build_cache(reader, inputs)

        my_context = _compute_block0_context(model, inputs, cache)
        my_bins = block.attention.quantize_o_proj_input(my_context)
        ref_bins = torch.from_numpy(
            outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite"]
        ).to(torch.int32)

        _report_stats(pos, my_bins, ref_bins)


if __name__ == "__main__":
    main()
