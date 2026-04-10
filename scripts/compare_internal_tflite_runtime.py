from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from gemma_mtp import GemmaMtpDrafter, TFLiteModelReader, build_partial_state_dict, infer_config_from_graph_export
from gemma_mtp.runtime_attention import (
    apply_query_rope,
    exact_attention_context,
    reshape_grouped_query,
    resolve_cache_tensor,
)


TENSOR_NAMES = (
    "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/composite1",
    "layer_0/layer_0.pre_q/pre_attention_norm/composite",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/composite1",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite",
    "layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate",
    "layer_0/layer_0.pre_q/attn.pre_q/reshape",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1",
    "layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/transpose",
    "layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite1",
    "layer_0/layer_0.post_qkv/post_attention_norm/composite",
    "layer_0/layer_0.post_qkv/add1",
    "mtp_final_norm/composite",
)


def _input_scale(reader: TFLiteModelReader, signature_name: str) -> float:
    info = reader.tensor_info(f"mtp_drafter_{signature_name}")
    if info.quantization is None or info.quantization.scale.size != 1:
        raise ValueError(f"Expected single-scale quantization for {signature_name}")
    return float(info.quantization.scale[0])


def _torch_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23"):
        cache[name] = {
            "tensor": torch.from_numpy(inputs[name].copy()),
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
        name,
        "max_abs_diff",
        float(diff.max()),
        "mean_abs_diff",
        float(diff.mean()),
        "cosine_similarity",
        float(cosine),
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
        "--litert-python",
        default=".venv312/bin/python",
    )
    parser.add_argument(
        "--inputs-path",
        default="data/derived/tflite_compare_inputs.npz",
    )
    parser.add_argument(
        "--outputs-path",
        default="data/derived/tflite_internal_compare_outputs.npz",
    )
    args = parser.parse_args()

    inputs_path = Path(args.inputs_path)
    if not inputs_path.exists():
        raise SystemExit(
            "Missing shared runtime inputs. Run `PYTHONPATH=.vendor:src python scripts/compare_tflite_runtime.py` first."
        )

    _run_internal_dump(
        litert_python=args.litert_python,
        tflite_model=args.tflite_model,
        inputs_path=str(inputs_path),
        outputs_path=args.outputs_path,
    )

    inputs = np.load(inputs_path)
    tflite_outputs = np.load(args.outputs_path)

    config = infer_config_from_graph_export(args.graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(args.graph_json, args.tflite_model)
    model.load_state_dict(state_dict, strict=False)
    reader = TFLiteModelReader(args.tflite_model)
    cache = _torch_cache(reader, inputs)

    activations = torch.from_numpy(inputs["activations"].copy())
    mask = torch.from_numpy(inputs["mask"].copy())
    input_pos = torch.from_numpy(inputs["input_pos"].copy())
    param_tensor = torch.from_numpy(inputs["param_tensor"].copy())

    with torch.no_grad():
        x = model.pre_project(activations)
        block0 = model.blocks[0]
        pre_attn = block0.pre_attn_norm(x)
        q = block0.attention.q_proj(pre_attn).view(
            1,
            1,
            block0.attention.spec.query_heads,
            block0.attention.spec.query_head_dim,
        )
        query_norm = block0.attention.query_norm(q)
        query_rope = apply_query_rope(
            query_norm,
            spec=block0.attention.spec,
            input_pos=input_pos,
            param_tensor=param_tensor,
        )
        grouped_query = query_rope.view(1, 2, 2, 256)
        attention_context = exact_attention_context(
            reshape_grouped_query(query_rope.reshape(1, 1, -1), spec=block0.attention.spec),
            spec=block0.attention.spec,
            key_cache=resolve_cache_tensor(cache, block0.attention.spec.key_cache_name or ""),
            value_cache=resolve_cache_tensor(cache, block0.attention.spec.value_cache_name or ""),
            input_pos=input_pos,
            mask=mask,
            param_tensor=param_tensor,
        )
        context_transposed = attention_context.reshape(1, 1, 2, 2, 256).permute(0, 2, 3, 1, 4).reshape(1, 1, 4, 256)
        attention_out = block0.attention.o_proj(context_transposed.reshape(1, 1, -1))
        post_attn_norm = block0.post_attn_norm(x + attention_out)
        block0_out = block0(
            x,
            mask=mask,
            base_kv_cache=cache,
            input_pos=input_pos,
            param_tensor=param_tensor,
        )

        hidden = block0_out
        for block in model.blocks[1:]:
            hidden = block(
                hidden,
                mask=mask,
                base_kv_cache=cache,
                input_pos=input_pos,
                param_tensor=param_tensor,
            )
        final_norm = model.final_norm(hidden)

    comparisons = [
        ("pre_project", x, tflite_outputs["MtpDrafterModel.mtp_pre_project/mtp_pre_proj/composite1"]),
        ("block0.pre_attn_norm", pre_attn, tflite_outputs["layer_0/layer_0.pre_q/pre_attention_norm/composite"]),
        ("block0.q_proj", q, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/composite1"]),
        ("block0.query_norm", query_norm, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite"]),
        ("block0.query_rope", query_rope, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/attn._pre_attention_query_fn/maybe_rope/concatenate"]),
        ("block0.grouped_query", grouped_query, tflite_outputs["layer_0/layer_0.pre_q/attn.pre_q/reshape"]),
        ("block0.attn_context_grouped", attention_context, tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/composite1"]),
        ("block0.attn_context_transposed", context_transposed, tflite_outputs["layer_0/attn.dot_product_attention/attn.dot_product_attention_extensible/dot_attn/dot_attn._qkv_fn/transpose"]),
        ("block0.attn_out", attention_out, tflite_outputs["layer_0/layer_0.post_qkv/attn.post_qkv/attn_vec_einsum/composite1"]),
        ("block0.post_attn_norm", post_attn_norm, tflite_outputs["layer_0/layer_0.post_qkv/post_attention_norm/composite"]),
        ("block0.out", block0_out, tflite_outputs["layer_0/layer_0.post_qkv/add1"]),
        ("final_norm", final_norm, tflite_outputs["mtp_final_norm/composite"]),
    ]

    for name, pred, ref in comparisons:
        _compare(name, pred, ref)


if __name__ == "__main__":
    main()
