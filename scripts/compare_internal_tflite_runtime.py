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
    "layer_0/layer_0.post_qkv/pre_ffw_norm/composite",
    "layer_0/layer_0.post_qkv/mlp/gating_einsum1/composite1",
    "layer_0/layer_0.post_qkv/mlp/gating_einsum2/composite",
    "layer_0/layer_0.post_qkv/mlp/mul",
    "layer_0/layer_0.post_qkv/mlp/linear/composite1",
    "layer_0/layer_0.post_qkv/post_ffw_norm/composite",
    "layer_0/layer_0.post_qkv/add1",
    "mtp_drafter_kv_cache_k_22:0",
    "mtp_drafter_kv_cache_v_22:0",
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
        "activations": (rng.standard_normal((1, 1, config.input_activation_dim)) * 10.0).astype(np.float32),
        "mask": rng.integers(0, 2, (1, 1, 1, cache_size)).astype(np.bool_),
        "input_pos": np.array([input_pos_val], dtype=np.int32),
        "param_tensor": rng.integers(0, 100, (1, 1, 1, 7)).astype(np.int32),
    }
    for spec in config.attention_specs:
        if spec.key_cache_name:
            inputs[spec.key_cache_name] = rng.integers(-100, 100, (1, spec.kv_heads, cache_size, spec.query_head_dim)).astype(np.int8)
        if spec.value_cache_name:
            inputs[spec.value_cache_name] = rng.integers(-100, 100, (1, spec.kv_heads, spec.query_head_dim, cache_size)).astype(np.int8)
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
            
            # Use TFLite's probs to verify post-context logic
            t_scores_raw = torch.from_numpy(tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite"])
            t_scores_win = t_scores_raw[..., 0:512]
            t_probs = torch.softmax(t_scores_win, dim=-1)
            
            # Circular wrapping for v_cache parity investigation
            v_wrapped = torch.zeros((1, 2, 256, 512))
            v_orig_np = inputs["kv_cache_v_22"]
            v_scale = _input_scale(reader, "kv_cache_v_22")
            for t in range(pos_val + 1):
                v_wrapped[0, :, :, t % 512] = torch.from_numpy(v_orig_np[0, :, :, t].astype(np.float32)) * v_scale
            
            # Try all rolls
            results_v = []
            for o in range(512):
                v_roll = torch.roll(v_wrapped, o, dims=3)
                my_ctx = torch.einsum("bhqk,bhdk->bhqd", t_probs, v_roll)
                sim = torch.nn.functional.cosine_similarity(my_ctx.flatten(), torch.from_numpy(tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"]).flatten(), dim=0).item()
                results_v.append((o, sim))
            
            best_o, _ = max(results_v, key=lambda x: x[1])
            attention_context = torch.einsum("bhqk,bhdk->bhqd", t_probs, torch.roll(v_wrapped, best_o, dims=3))
            
            # Try all context permutations for o_proj
            weight = state_dict['blocks.0.attention.o_proj.weight']
            ctx_reshaped = attention_context.reshape(1, 2, 2, 1, 256)
            results_ctx = []
            for p in itertools.permutations(range(5)):
                try:
                    ctx_p = ctx_reshaped.permute(*p).reshape(1, 1, 1024)
                    my_out = torch.nn.functional.linear(ctx_p, weight)
                    sim = torch.nn.functional.cosine_similarity(my_out.flatten(), torch.from_numpy(tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1"]).flatten(), dim=0).item()
                    results_ctx.append((p, sim))
                except:
                    continue
            best_p, best_sim_ctx = max(results_ctx, key=lambda x: x[1])
            print(f"Best context permute: {best_p}, sim: {best_sim_ctx}")
            
            attention_out = torch.nn.functional.linear(ctx_reshaped.permute(*best_p).reshape(1, 1, 1024), weight)
            post_attn_norm = block0.post_attn_norm(attention_out)
            
            # MLP components
            mlp_in = block0.pre_ffw_norm(x + post_attn_norm)
            gate = block0.mlp.gate_proj(mlp_in)
            up = block0.mlp.up_proj(mlp_in)
            gated = F.gelu(gate, approximate="tanh")
            mul = gated * up
            down = block0.mlp.down_proj(mul)
            post_ffw = block0.post_ffw_norm(down)
            
            block0_out = (x + post_attn_norm) + post_ffw

            _compare("block0.attn_context", attention_context, tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"])
            _compare("block0.o_proj_in", ctx_reshaped.permute(*best_p).reshape(1, 1, 1024), tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/reshape"])
            _compare("block0.attn_out", attention_out, tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1"])
            _compare("block0.mlp_in", mlp_in, tflite_outputs["layer_0/layer_0.post_qkv/pre_ffw_norm/composite"])
            _compare("block0.out", block0_out, tflite_outputs["layer_0/layer_0.post_qkv/add1"])


if __name__ == "__main__":
    from torch.nn import functional as F
    main()
