from __future__ import annotations

from pathlib import Path

import torch

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
