from __future__ import annotations

from pathlib import Path

import torch

from .module import ExternalAttentionAdapter, GemmaMtpDrafter
from .plan import extract_linear_plan, extract_norm_plan
from .tflite_loader import TFLiteModelReader


def build_partial_state_dict(
    graph_json_path: str | Path,
    tflite_model_path: str | Path,
) -> dict[str, torch.Tensor]:
    reader = TFLiteModelReader(tflite_model_path)
    state_dict: dict[str, torch.Tensor] = {}

    for plan in extract_linear_plan(graph_json_path):
        if plan.filter_ref is None:
            continue
        weight = reader.read_dequantized(plan.filter_ref.tensor_name)
        state_dict[f"{plan.module_name}.weight"] = torch.from_numpy(weight.copy())

    for plan in extract_norm_plan(graph_json_path):
        weight = reader.read_dequantized(plan.tensor_ref.tensor_name)
        state_dict[plan.module_name] = torch.from_numpy(weight.copy())

    return state_dict


def hydrate_runtime_quantization(
    model: GemmaMtpDrafter,
    graph_json_path: str | Path,
    tflite_model_path: str | Path,
) -> None:
    reader = TFLiteModelReader(tflite_model_path)
    named_modules = dict(model.named_modules())

    for plan in extract_linear_plan(graph_json_path):
        if not plan.module_name.endswith("attention.o_proj"):
            continue
        if plan.input_ref is None or plan.filter_ref is None:
            continue

        module = named_modules.get(plan.module_name.rsplit(".", 1)[0])
        if not isinstance(module, ExternalAttentionAdapter):
            continue

        input_info = reader.tensor_info(plan.input_ref.tensor_name)
        weight_info = reader.tensor_info(plan.filter_ref.tensor_name)
        output_info = reader.tensor_info(plan.namespace)
        input_quant = input_info.quantization
        weight_quant = weight_info.quantization
        output_quant = output_info.quantization
        if input_quant is None or weight_quant is None or output_quant is None:
            continue

        module.set_o_proj_quantization(
            input_scale=torch.tensor(float(input_quant.scale[0]), dtype=torch.float32),
            input_zero_point=torch.tensor(
                int(input_quant.zero_point[0]) if input_quant.zero_point.size else 0,
                dtype=torch.float32,
            ),
            weight_q=torch.from_numpy(reader.read_raw(plan.filter_ref.tensor_name).copy()).to(torch.int8),
            weight_scale=torch.from_numpy(weight_quant.scale.copy()).to(torch.float32),
            weight_zero_point=torch.from_numpy(weight_quant.zero_point.copy()).to(torch.int32),
            output_scale=torch.tensor(float(output_quant.scale[0]), dtype=torch.float32),
            output_zero_point=torch.tensor(
                int(output_quant.zero_point[0]) if output_quant.zero_point.size else 0,
                dtype=torch.float32,
            ),
        )
