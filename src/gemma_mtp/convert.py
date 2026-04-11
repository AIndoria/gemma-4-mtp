from __future__ import annotations

from pathlib import Path
import re

import torch

from .graph import _attrs_to_dict, _top_graph, load_graph_export
from .module import ExternalAttentionAdapter, GemmaMtpDrafter, OutputQuantLinear
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
    graph = _top_graph(load_graph_export(graph_json_path))
    node_by_namespace = {node["namespace"]: node for node in graph["nodes"]}
    output_quant_modules = {
        "pre_project",
        *{f"blocks.{layer_index}.attention.q_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.gate_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.up_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.down_proj" for layer_index in range(model.config.num_layers)},
        "post_project",
    }
    input_quant_modules = {
        "pre_project",
        *{f"blocks.{layer_index}.attention.q_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.gate_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.up_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.down_proj" for layer_index in range(model.config.num_layers)},
        "post_project",
    }
    exact_weight_quant_modules = {
        "pre_project",
        *{f"blocks.{layer_index}.attention.q_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.gate_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.up_proj" for layer_index in range(model.config.num_layers)},
        *{f"blocks.{layer_index}.mlp.down_proj" for layer_index in range(model.config.num_layers)},
        "post_project",
    }

    for plan in extract_linear_plan(graph_json_path):
        module = named_modules.get(plan.module_name)
        output_quant = None
        output_zero_point = torch.tensor(0.0, dtype=torch.float32)
        node = node_by_namespace.get(plan.namespace)
        try:
            output_info = reader.tensor_info(plan.namespace)
        except KeyError:
            output_info = None
        if output_info is not None and output_info.quantization is not None:
            output_quant = torch.tensor(float(output_info.quantization.scale[0]), dtype=torch.float32)
            output_zero_point = torch.tensor(
                int(output_info.quantization.zero_point[0]) if output_info.quantization.zero_point.size else 0,
                dtype=torch.float32,
            )
        if node is not None:
            output_attrs = _attrs_to_dict(node["outputsMetadata"][0]["attrs"])
            quantization = output_attrs.get("quantization")
            if quantization is not None and output_quant is None:
                match = re.match(r"([0-9.eE+-]+) \* q", quantization)
                if match is not None:
                    output_quant = torch.tensor(float(match.group(1)), dtype=torch.float32)

        if (
            isinstance(module, OutputQuantLinear)
            and plan.module_name in input_quant_modules
            and plan.input_ref is not None
        ):
            input_quant = None
            input_zero_point = torch.tensor(0.0, dtype=torch.float32)
            try:
                input_info = reader.tensor_info(plan.input_ref.tensor_name)
            except KeyError:
                input_info = None
            if input_info is not None and input_info.quantization is not None:
                input_quant = torch.tensor(float(input_info.quantization.scale[0]), dtype=torch.float32)
                input_zero_point = torch.tensor(
                    int(input_info.quantization.zero_point[0]) if input_info.quantization.zero_point.size else 0,
                    dtype=torch.float32,
                )
            elif plan.input_ref.quantization is not None:
                match = re.match(r"([0-9.eE+-]+) \* q", plan.input_ref.quantization)
                if match is not None:
                    input_quant = torch.tensor(float(match.group(1)), dtype=torch.float32)

            if input_quant is not None:
                module.set_input_quantization(
                    input_scale=input_quant,
                    input_zero_point=input_zero_point,
                )

        if (
            isinstance(module, OutputQuantLinear)
            and plan.module_name in exact_weight_quant_modules
            and plan.filter_ref is not None
        ):
            weight_info = reader.tensor_info(plan.filter_ref.tensor_name)
            weight_quant = weight_info.quantization
            if weight_quant is not None:
                module.set_weight_quantization(
                    weight_q=torch.from_numpy(reader.read_raw(plan.filter_ref.tensor_name).copy()).to(torch.int8),
                    weight_scale=torch.from_numpy(weight_quant.scale.copy()).to(torch.float32),
                    weight_zero_point=torch.from_numpy(weight_quant.zero_point.copy()).to(torch.int32),
                )

        if (
            isinstance(module, OutputQuantLinear)
            and plan.module_name in output_quant_modules
            and output_quant is not None
        ):
            module.set_output_quantization(
                output_scale=output_quant,
                output_zero_point=output_zero_point,
            )

        if not plan.module_name.endswith("attention.o_proj"):
            continue
        if plan.input_ref is None or plan.filter_ref is None:
            continue

        adapter = named_modules.get(plan.module_name.rsplit(".", 1)[0])
        if not isinstance(adapter, ExternalAttentionAdapter):
            continue

        input_info = reader.tensor_info(plan.input_ref.tensor_name)
        weight_info = reader.tensor_info(plan.filter_ref.tensor_name)
        input_quant = input_info.quantization
        weight_quant = weight_info.quantization
        output_info = reader.tensor_info(plan.namespace)
        output_quant_info = output_info.quantization
        if input_quant is None or weight_quant is None or output_quant_info is None:
            continue

        adapter.set_o_proj_quantization(
            input_scale=torch.tensor(float(input_quant.scale[0]), dtype=torch.float32),
            input_zero_point=torch.tensor(
                int(input_quant.zero_point[0]) if input_quant.zero_point.size else 0,
                dtype=torch.float32,
            ),
            weight_q=torch.from_numpy(reader.read_raw(plan.filter_ref.tensor_name).copy()).to(torch.int8),
            weight_scale=torch.from_numpy(weight_quant.scale.copy()).to(torch.float32),
            weight_zero_point=torch.from_numpy(weight_quant.zero_point.copy()).to(torch.int32),
            output_scale=torch.tensor(float(output_quant_info.scale[0]), dtype=torch.float32),
            output_zero_point=torch.tensor(
                int(output_quant_info.zero_point[0]) if output_quant_info.zero_point.size else 0,
                dtype=torch.float32,
            ),
        )
