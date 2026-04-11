from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/div;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/exp;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/broadcast_in_dim;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/reduce_sum;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/reduce_max;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/max;layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/sub",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite1",
    "layer_0/layer_0.post_qkv/add1",
    "layer_3/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite",
    "layer_3/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1",
    "layer_3/layer_3.post_qkv/attn.post_qkv/attn_vec_einsum/composite1",
)


def _input_scale(reader: TFLiteModelReader, signature_name: str) -> float:
    info = reader.tensor_info(f"mtp_drafter_{signature_name}:0")
    if info.quantization is None or info.quantization.scale.size != 1:
        raise ValueError(f"Expected single-scale quantization for {signature_name}")
    return float(info.quantization.scale[0])


def _torch_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23"):
        if name in inputs:
            val = inputs[name].copy()
            cache[name] = {
                "tensor": torch.from_numpy(val),
                "scale": _input_scale(reader, name),
                "zero_point": 0,
            }
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
    
    window_start = max(0, input_pos_val + 1 - 512)
    window_end = input_pos_val + 1
    param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
    param_tensor[0, 0, 0, 0] = window_start
    param_tensor[0, 0, 0, 1] = window_end
    param_tensor[0, 0, 0, 2] = window_end

    inputs = {
        "activations": rng.standard_normal((1, 1, config.input_activation_dim)).astype(np.float32),
        "mask": np.ones((1, 1, 1, cache_size), dtype=np.bool_),
        "input_pos": np.array([input_pos_val], dtype=np.int32),
        "param_tensor": param_tensor,
    }
    for spec in config.attention_specs:
        if spec.key_cache_name:
            inputs[spec.key_cache_name] = rng.integers(-10, 11, (1, spec.kv_heads, cache_size, spec.query_head_dim)).astype(np.int8)
        if spec.value_cache_name:
            inputs[spec.value_cache_name] = rng.integers(-10, 11, (1, spec.kv_heads, spec.query_head_dim, cache_size)).astype(np.int8)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json")
    parser.add_argument("--tflite-model", default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite")
    parser.add_argument("--litert-python", default=".venv312/bin/python")
    args = parser.parse_args()

    config = infer_config_from_graph_export(args.graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(args.graph_json, args.tflite_model)
    model.load_state_dict(state_dict, strict=False)
    
    for pos_val in [100, 700]:
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

        with torch.no_grad():
            block0 = model.blocks[0]
            pre_project = model.pre_project(torch.from_numpy(inputs["activations"]))
            pre_attn = block0.pre_attn_norm(pre_project)
            q = block0.attention.q_proj(pre_attn).view(1, 1, 4, 256)
            q_norm = block0.attention.query_norm(q)
            q_rope = apply_query_rope(q_norm, spec=block0.attention.spec, input_pos=torch.from_numpy(inputs["input_pos"]), param_tensor=torch.from_numpy(inputs["param_tensor"]))
            q_grouped = reshape_grouped_query(q_rope.reshape(1, 1, -1), spec=block0.attention.spec)
            
            _compare("block0.query_rope", q_rope, tflite_outputs[TENSOR_NAMES[0]])
            
            # Adapter verification
            t_ctx = tflite_outputs[TENSOR_NAMES[3]]
            my_ctx_adapter = exact_attention_context(
                q_grouped,
                spec=block0.attention.spec,
                key_cache=resolve_cache_tensor(cache, block0.attention.spec.key_cache_name or ""),
                value_cache=resolve_cache_tensor(cache, block0.attention.spec.value_cache_name or ""),
                input_pos=torch.from_numpy(inputs["input_pos"]),
                param_tensor=torch.from_numpy(inputs["param_tensor"]),
            )
            _compare("block0.attn_context_ADAPTER", my_ctx_adapter, t_ctx)
            
            # Attn out verification
            t_attn_out = tflite_outputs[TENSOR_NAMES[5]]
            
            # Manual attn out with HQ order
            ctx_reshaped = my_ctx_adapter.reshape(1, 1, 1024)
            attn_out_manual = block0.attention.o_proj(ctx_reshaped)
            _compare("block0.attn_out_CURRENT", attn_out_manual, t_attn_out)

            # Whole block verification
            block0_out = tflite_outputs[TENSOR_NAMES[6]]
            # Use model() instead of block() to check everything
            model_out = model(
                torch.from_numpy(inputs["activations"]),
                mask=torch.from_numpy(inputs["mask"]),
                base_kv_cache=cache,
                input_pos=torch.from_numpy(inputs["input_pos"]),
                param_tensor=torch.from_numpy(inputs["param_tensor"]),
            )
            # all_hidden_states[0] is pre_project
            # all_hidden_states[1] is block 0 output
            _compare("model.block0_output", model_out.all_hidden_states[1], block0_out)

            
            # Block 3 verification
            block3 = model.blocks[3]
            # Block 3 input is hidden_states after block 2.
            # But we can just use random input to see if context and out match.
            b3_in = torch.randn(1, 1, 256)
            
            q3 = block3.attention.q_proj(block3.pre_attn_norm(b3_in)).view(1, 1, 4, 512)
            q3_norm = block3.attention.query_norm(q3)
            q3_rope = apply_query_rope(q3_norm, spec=block3.attention.spec, input_pos=torch.from_numpy(inputs["input_pos"]), param_tensor=torch.from_numpy(inputs["param_tensor"]))
            q3_grouped = reshape_grouped_query(q3_rope.reshape(1, 1, -1), spec=block3.attention.spec)
            
            # For Block 3, it's global attention (local_window_size is None)
            my_ctx3 = exact_attention_context(
                q3_grouped,
                spec=block3.attention.spec,
                key_cache=resolve_cache_tensor(cache, block3.attention.spec.key_cache_name or ""),
                value_cache=resolve_cache_tensor(cache, block3.attention.spec.value_cache_name or ""),
                input_pos=torch.from_numpy(inputs["input_pos"]),
                mask=torch.from_numpy(inputs["mask"]),
                param_tensor=torch.from_numpy(inputs["param_tensor"]),
            )
            # Wait, Block 3 uses TENSOR_NAMES[8] for context and [9] for attn_out
            t_ctx3 = tflite_outputs[TENSOR_NAMES[8]]
            # We can't match context because our b3_in is random!
            # But we can check if TFLite's internal context matches our o_proj.
            
            t_ctx3_torch = torch.from_numpy(t_ctx3)
            # Block 3 head_dim=512. queries_per_kv=2. kv_heads=2.
            # 2 * 2 * 512 = 2048.
            ctx3_reshaped = t_ctx3_torch.reshape(1, 1, 2048)
            attn_out3_manual = block3.attention.o_proj(ctx3_reshaped)
            t_attn_out3 = tflite_outputs[TENSOR_NAMES[9]]
            _compare("block3.attn_out", attn_out3_manual, t_attn_out3)


if __name__ == "__main__":
    main()
