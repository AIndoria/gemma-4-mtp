from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gemma_mtp import GemmaMtpDrafter, TFLiteModelReader, build_partial_state_dict, infer_config_from_graph_export
from gemma_mtp.runtime_attention import (
    apply_query_rope,
    exact_attention_context,
    prepare_key_cache,
    prepare_value_cache,
    reshape_grouped_query,
    resolve_cache_tensor,
)
from gemma_mtp.config import MtpDrafterConfig


TENSOR_NAMES = (
    "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/composite1",
    "layer_0/layer_0.pre_q/pre_attention_norm/composite",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/composite1",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate",
    "layer_0/layer_0.pre_q/attn.pre_q/reshape",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/transpose",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/reshape",
    "layer_0/layer_0.post_qkv/post_attention_norm/composite",
    "layer_0/layer_0.post_qkv/add1",

)


def _input_scale(reader: TFLiteModelReader, signature_name: str) -> float:
    info = reader.tensor_info(f"mtp_drafter_{signature_name}")
    if info.quantization is None or info.quantization.scale.size != 1:
        raise ValueError(f"Expected single-scale quantization for {signature_name}")
    return float(info.quantization.scale[0])


def _torch_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23"):
        if name in inputs:
            val = inputs[name].copy()
            if val.dtype == np.int8:
                cache[name] = {
                    "tensor": torch.from_numpy(val),
                    "scale": _input_scale(reader, name),
                    "zero_point": 0,
                }
            else:
                cache[name] = torch.from_numpy(val)
    return cache


def _run_internal_dump(
    *,
    litert_python: str,
    tflite_model: str,
    inputs_path: str,
    outputs_path: str,
) -> None:
    cmd = [
        litert_python,
        "scripts/run_tflite_inference.py",
        "--tflite-model",
        tflite_model,
        "--inputs",
        inputs_path,
        "--outputs",
        outputs_path,
        "--preserve-all-tensors",
    ]
    for name in TENSOR_NAMES:
        cmd.extend(["--dump-tensor", name])
    subprocess.run(
        cmd,
        check=True,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV"}},
    )


def _compare(name: str, pred: torch.Tensor, ref: np.ndarray) -> None:
    ref_tensor = torch.from_numpy(ref)
    diff = (pred.detach().cpu() - ref_tensor).abs()
    cosine = torch.nn.functional.cosine_similarity(
        pred.detach().cpu().flatten().float(),
        ref_tensor.flatten().float(),
        dim=0,
    )
    print(
        f"{name: <30} max_abs_diff {float(diff.max()):.6f} mean_abs_diff {float(diff.mean()):.6f} cosine_similarity {float(cosine):.6f}"
    )


def _generate_synthetic_inputs(config: MtpDrafterConfig, input_pos_val: int = 700) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    cache_size = 32003
    inputs = {
        "activations": rng.standard_normal((1, 1, config.input_activation_dim)).astype(np.float32),
        "mask": rng.integers(0, 2, (1, 1, 1, cache_size)).astype(np.bool_),
        "input_pos": np.array([input_pos_val], dtype=np.int32),
        "param_tensor": rng.integers(0, 100, (1, 1, 1, 7)).astype(np.int32),
    }
    for spec in config.attention_specs:
        if spec.key_cache_name:
            inputs[spec.key_cache_name] = rng.integers(-2, 3, (1, spec.kv_heads, cache_size, spec.query_head_dim)).astype(np.int8)
        if spec.value_cache_name:
            inputs[spec.value_cache_name] = rng.integers(-2, 3, (1, spec.kv_heads, spec.query_head_dim, cache_size)).astype(np.int8)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json")
    parser.add_argument("--tflite-model", default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    parser.add_argument("--litert-python", default=".venv312/bin/python")
    parser.add_argument("--outputs-path", default="data/derived/tflite_internal_compare_outputs.npz")
    args = parser.parse_args()

    config = infer_config_from_graph_export(args.graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(args.graph_json, args.tflite_model)
    model.load_state_dict(state_dict, strict=False)
    
    for pos_val in [100]:
        print(f"\n--- Testing position {pos_val} ---")
        inputs = _generate_synthetic_inputs(config, input_pos_val=pos_val)
        inputs_path = f"data/derived/tflite_inputs_{pos_val}.npz"
        outputs_path = f"data/derived/tflite_outputs_{pos_val}.npz"
        np.savez(inputs_path, **inputs)

        _run_internal_dump(
            litert_python=args.litert_python,
            tflite_model=args.tflite_model,
            inputs_path=inputs_path,
            outputs_path=outputs_path,
        )

        tflite_outputs = np.load(outputs_path)
        reader = TFLiteModelReader(args.tflite_model)
        cache = _torch_cache(reader, inputs)

        activations = torch.from_numpy(inputs["activations"].copy())
        mask = torch.from_numpy(inputs["mask"].copy())
        input_pos = torch.from_numpy(inputs["input_pos"].copy())
        print("input_pos:", input_pos)
        param_tensor = torch.from_numpy(inputs["param_tensor"].copy())

        with torch.no_grad():
            x = model.pre_project(activations)
            block0 = model.blocks[0]
            pre_attn = block0.pre_attn_norm(x)
            q = block0.attention.q_proj(pre_attn).view(1, 1, 4, 256)
            query_norm = block0.attention.query_norm(q)
            query_rope = apply_query_rope(query_norm, spec=block0.attention.spec, input_pos=input_pos, param_tensor=param_tensor)
            
            k_cache = resolve_cache_tensor(cache, block0.attention.spec.key_cache_name or "")
            v_cache = resolve_cache_tensor(cache, block0.attention.spec.value_cache_name or "")
            attn_scores = torch.einsum("bhqd,bhkd->bhqk", reshape_grouped_query(query_rope.reshape(1, 1, -1), spec=block0.attention.spec), k_cache)

            attention_context = exact_attention_context(
                reshape_grouped_query(query_rope.reshape(1, 1, -1), spec=block0.attention.spec),
                spec=block0.attention.spec,
                key_cache=k_cache,
                value_cache=v_cache,
                input_pos=input_pos,
                mask=mask,
                param_tensor=param_tensor,
            )
            
            context_transposed = attention_context.reshape(1, 1, 2, 2, 256).permute(0, 2, 3, 1, 4).reshape(1, 1, 1024)
            weight = state_dict['blocks.0.attention.o_proj.weight']
            # Try integer matmul
            # TFLite Op 44 is FULLY_CONNECTED [97, 98, -1] -> [99]
            # 97: context_quant (int8), 98: weight (int8), 99: result (int32?)
            ctx_q = tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite"]
            w_raw = reader.read_raw("layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general")
            
            # (1, 1, 1024) @ (256, 1024).T -> (1, 1, 256)
            res_int = (ctx_q.astype(np.int32) @ w_raw.T.astype(np.int32))
            
            # Now dequantize the result
            # We need the output scale. T 99 or Op 45?
            # T 99 is int8 in dump_sg, but SG 0 trace said Op 45: DEQUANTIZE [99] -> [100]
            # Let's check T 99 quantization.
            info99 = reader.tensor_info(99)
            scale99 = info99.quantization.scale[0] if info99.quantization else 1.0
            res_f = res_int.astype(np.float32) * scale99
            
            _compare("block0.attn_out_int_matmul", torch.from_numpy(res_f), tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1"])
            
            block0_out = block0(
                x,
                mask=mask,
                base_kv_cache=cache,
                input_pos=input_pos,
                param_tensor=param_tensor,
            )

            _compare("block0.q_proj", q, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/composite1"])
            _compare("block0.query_norm", query_norm, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite"])
            _compare("block0.grouped_query", query_rope.view(1, 2, 2, 256), tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/reshape"])
            _compare("block0.attn_scores_window", attn_scores[..., 0:512], tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite"][..., 0:512])
            _compare("block0.attn_context", attention_context, tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"])
            _compare("block0.o_proj_in", context_transposed, tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/reshape"])
            _compare("block0.attn_out", attention_out, tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1"])
            _compare("block0.post_attn_norm", post_attn_norm, tflite_outputs["layer_0/layer_0.post_qkv/post_attention_norm/composite"])
            _compare("block0.out", block0_out, tflite_outputs["layer_0/layer_0.post_qkv/add1"])


if __name__ == "__main__":
    main()
