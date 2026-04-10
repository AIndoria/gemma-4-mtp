from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from gemma_mtp import (
    GemmaMtpDrafter,
    TFLiteModelReader,
    ZeroAttentionAdapter,
    build_partial_state_dict,
    infer_config_from_graph_export,
)


def _input_scale(reader: TFLiteModelReader, signature_name: str) -> float:
    info = reader.tensor_info(f"mtp_drafter_{signature_name}")
    if info.quantization is None or info.quantization.scale.size != 1:
        raise ValueError(f"Expected single-scale quantization for {signature_name}")
    return float(info.quantization.scale[0])


def _build_synthetic_inputs(
    reader: TFLiteModelReader,
    *,
    seed: int,
    decode_position: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    context_size = 32003
    mask_limit = min(context_size, decode_position + 257)
    window_start = max(0, decode_position + 1 - 512)
    window_end = decode_position + 1

    param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
    param_tensor[0, 0, 0, 0] = window_start
    param_tensor[0, 0, 0, 1] = window_end
    param_tensor[0, 0, 0, 2] = window_end

    inputs: dict[str, np.ndarray] = {
        "input_pos": np.array([decode_position], dtype=np.int32),
        "activations": rng.standard_normal((1, 1, 5120), dtype=np.float32),
        "param_tensor": param_tensor,
        "mask": np.zeros((1, 1, 1, context_size), dtype=np.bool_),
        "kv_cache_k_22": rng.integers(
            -128, 128, size=reader.tensor_info("mtp_drafter_kv_cache_k_22").shape, dtype=np.int8
        ),
        "kv_cache_v_22": rng.integers(
            -128, 128, size=reader.tensor_info("mtp_drafter_kv_cache_v_22").shape, dtype=np.int8
        ),
        "kv_cache_k_23": rng.integers(
            -128, 128, size=reader.tensor_info("mtp_drafter_kv_cache_k_23").shape, dtype=np.int8
        ),
        "kv_cache_v_23": rng.integers(
            -128, 128, size=reader.tensor_info("mtp_drafter_kv_cache_v_23").shape, dtype=np.int8
        ),
    }
    inputs["mask"][..., :mask_limit] = True
    return inputs


def _to_torch_cache(reader: TFLiteModelReader, inputs: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    for name in ("kv_cache_k_22", "kv_cache_v_22", "kv_cache_k_23", "kv_cache_v_23"):
        cache[name] = {
            "tensor": torch.from_numpy(inputs[name].copy()),
            "scale": _input_scale(reader, name),
            "zero_point": 0,
        }
    return cache


def _run_pytorch(
    graph_json: str,
    tflite_model: str,
    inputs: dict[str, np.ndarray],
    *,
    zero_attention: bool = False,
) -> dict[str, torch.Tensor]:
    config = infer_config_from_graph_export(graph_json)
    model = GemmaMtpDrafter(config)
    state_dict = build_partial_state_dict(graph_json, tflite_model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Unexpected state-dict mismatch. missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if zero_attention:
        for block in model.blocks:
            block.attention = ZeroAttentionAdapter(block.attention.spec, config.model_dim)

    reader = TFLiteModelReader(tflite_model)
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(inputs["activations"].copy()),
            mask=torch.from_numpy(inputs["mask"].copy()),
            base_kv_cache=_to_torch_cache(reader, inputs),
            input_pos=torch.from_numpy(inputs["input_pos"].copy()),
            param_tensor=torch.from_numpy(inputs["param_tensor"].copy()),
        )
    return {
        "logits": outputs.logits,
        "projected_activations": outputs.projected_activations,
    }


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
        "--decode-position",
        type=int,
        default=700,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--inputs-path",
        default="data/derived/tflite_compare_inputs.npz",
    )
    parser.add_argument(
        "--outputs-path",
        default="data/derived/tflite_compare_outputs.npz",
    )
    parser.add_argument(
        "--zero-attn-baseline",
        action="store_true",
    )
    args = parser.parse_args()

    reader = TFLiteModelReader(args.tflite_model)
    inputs = _build_synthetic_inputs(
        reader,
        seed=args.seed,
        decode_position=args.decode_position,
    )

    inputs_path = Path(args.inputs_path)
    outputs_path = Path(args.outputs_path)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(inputs_path, **inputs)

    subprocess.run(
        [
            args.litert_python,
            "scripts/run_tflite_inference.py",
            "--tflite-model",
            args.tflite_model,
            "--inputs",
            str(inputs_path),
            "--outputs",
            str(outputs_path),
        ],
        check=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "VIRTUAL_ENV"}
        },
    )

    with np.load(outputs_path) as data:
        tflite_outputs = {name: data[name] for name in data.files}

    pytorch_outputs = _run_pytorch(args.graph_json, args.tflite_model, inputs)
    zero_attention_outputs = (
        _run_pytorch(
            args.graph_json,
            args.tflite_model,
            inputs,
            zero_attention=True,
        )
        if args.zero_attn_baseline
        else None
    )

    for name in ("logits", "projected_activations"):
        torch_output = pytorch_outputs[name].detach().cpu()
        tflite_output = torch.from_numpy(tflite_outputs[name])
        diff = (torch_output - tflite_output).abs()
        print(f"{name}_max_abs_diff", float(diff.max()))
        print(f"{name}_mean_abs_diff", float(diff.mean()))
        cosine = torch.nn.functional.cosine_similarity(
            torch_output.flatten().float(),
            tflite_output.flatten().float(),
            dim=0,
        )
        print(f"{name}_cosine_similarity", float(cosine))
        if name == "logits":
            print(f"{name}_top1_match", int(torch_output.argmax()) == int(tflite_output.argmax()))

        if zero_attention_outputs is not None:
            baseline_output = zero_attention_outputs[name].detach().cpu()
            baseline_diff = (baseline_output - tflite_output).abs()
            print(f"{name}_zero_attn_max_abs_diff", float(baseline_diff.max()))
            print(f"{name}_zero_attn_mean_abs_diff", float(baseline_diff.mean()))


if __name__ == "__main__":
    main()
