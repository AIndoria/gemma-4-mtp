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
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1",
    "layer_0/layer_0.post_qkv/add1",
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
            
            _compare("block0.query_rope", q_rope, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate"])
            
            # Scores verification (Fixed window at 0:512)
            k_cache = resolve_cache_tensor(cache, block0.attention.spec.key_cache_name or "")
            q_grouped = reshape_grouped_query(q_rope.reshape(1, 1, -1), spec=block0.attention.spec)
            my_scores = torch.einsum("bhqd,bhkd->bhqk", q_grouped, k_cache[..., :512, :])
            
            t_scores = tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite"]
            _compare("block0.attn_scores_0:512", my_scores, t_scores[..., :512])
            
            # Context verification (Fixed window at 0:512)
            v_cache = resolve_cache_tensor(cache, block0.attention.spec.value_cache_name or "")
            t_probs = torch.softmax(torch.from_numpy(t_scores[..., :512]), dim=-1)
            my_ctx = torch.einsum("bhqk,bhdk->bhqd", t_probs, v_cache[..., :512])
            
            t_ctx = tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"]
            _compare("block0.attn_context_0:512", my_ctx, t_ctx)


if __name__ == "__main__":
    main()
