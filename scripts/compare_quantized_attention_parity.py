from __future__ import annotations

import argparse

import torch

from gemma_mtp import AttentionSpec, TFLiteModelReader, infer_config_from_graph_export
from gemma_mtp.module import GroupedQueryAttentionAdapter


def quantized_cache_entry(shape: tuple[int, ...], scale: float) -> dict[str, object]:
    tensor = torch.randint(-128, 128, shape, dtype=torch.int8)
    return {
        "tensor": tensor,
        "scale": scale,
        "zero_point": 0,
    }


def dequantize_entry(entry: dict[str, object]) -> torch.Tensor:
    tensor = entry["tensor"]
    scale = float(entry["scale"])
    zero_point = float(entry["zero_point"])
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor in quantized cache entry.")
    return (tensor.to(torch.float32) - zero_point) * scale


def run_layer_case(
    spec: AttentionSpec,
    *,
    reader: TFLiteModelReader,
    mask: torch.Tensor | None,
    input_pos: torch.Tensor | None,
    param_tensor: torch.Tensor | None,
) -> float:
    torch.manual_seed(0)
    adapter = GroupedQueryAttentionAdapter(spec, model_dim=256).eval()
    hidden_states = torch.randn(1, 1, 256)

    if spec.key_cache_name is None or spec.value_cache_name is None:
        raise ValueError(f"Layer {spec.layer_index} is missing cache names.")

    key_info = reader.tensor_info(f"mtp_drafter_{spec.key_cache_name}")
    value_info = reader.tensor_info(f"mtp_drafter_{spec.value_cache_name}")

    if key_info.quantization is None or value_info.quantization is None:
        raise ValueError("Expected quantization metadata for external KV caches.")

    key_entry = quantized_cache_entry(key_info.shape, float(key_info.quantization.scale[0]))
    value_entry = quantized_cache_entry(value_info.shape, float(value_info.quantization.scale[0]))

    quantized_output = adapter(
        hidden_states,
        mask=mask,
        base_kv_cache={
            spec.key_cache_name: key_entry,
            spec.value_cache_name: value_entry,
        },
        input_pos=input_pos,
        param_tensor=param_tensor,
    )
    float_output = adapter(
        hidden_states,
        mask=mask,
        base_kv_cache={
            spec.key_cache_name: dequantize_entry(key_entry),
            spec.value_cache_name: dequantize_entry(value_entry),
        },
        input_pos=input_pos,
        param_tensor=param_tensor,
    )
    return float((quantized_output - float_output).abs().max().detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-json",
        default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json",
    )
    parser.add_argument(
        "--tflite",
        default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite",
    )
    args = parser.parse_args()

    config = infer_config_from_graph_export(args.graph_json)
    reader = TFLiteModelReader(args.tflite)

    local_spec = config.attention_specs[0]
    full_spec = config.attention_specs[3]

    local_diff = run_layer_case(
        local_spec,
        reader=reader,
        mask=None,
        input_pos=torch.tensor([33], dtype=torch.int32),
        param_tensor=torch.tensor([[[[0, 34, 34, 0, 0, 0, 0]]]], dtype=torch.int32),
    )
    full_mask = torch.zeros(1, 1, 1, 32003, dtype=torch.bool)
    full_mask[..., :129] = True
    full_diff = run_layer_case(
        full_spec,
        reader=reader,
        mask=full_mask,
        input_pos=None,
        param_tensor=None,
    )

    print("local_quantized_max_abs_diff", local_diff)
    print("full_quantized_max_abs_diff", full_diff)


if __name__ == "__main__":
    main()
