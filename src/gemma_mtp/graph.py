from __future__ import annotations

import json
import re
from pathlib import Path

from .config import AttentionSpec, MtpDrafterConfig


def load_graph_export(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_shape(shape: str) -> tuple[str, tuple[int, ...]]:
    dtype, dims_blob = shape.split("[", maxsplit=1)
    dims_blob = dims_blob.rstrip("]")
    dims = tuple(int(part.strip()) for part in dims_blob.split(",") if part.strip())
    return dtype, dims


def _attrs_to_dict(attrs: list[dict]) -> dict[str, str]:
    return {item["key"]: item["value"] for item in attrs}


def _top_graph(export: dict) -> dict:
    return export["graphs"][0]


def _signature_tensors(graph: dict) -> dict[str, dict[str, str]]:
    tensors: dict[str, dict[str, str]] = {}
    for node in graph["nodes"]:
        for metadata in node.get("outputsMetadata", []):
            attrs = _attrs_to_dict(metadata.get("attrs", []))
            signature_name = attrs.get("signature_name")
            if signature_name:
                tensors[signature_name] = attrs
    return tensors


def _find_node_by_namespace_suffix(graph: dict, suffix: str) -> dict:
    for node in graph["nodes"]:
        namespace = node.get("namespace", "")
        if namespace.endswith(suffix):
            return node
    raise KeyError(f"Could not find node ending with namespace suffix: {suffix}")


def _find_layer_indices(graph: dict) -> list[int]:
    indices: set[int] = set()
    for node in graph["nodes"]:
        match = re.search(r"layer_(\d+)", node.get("namespace", ""))
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _infer_attention_specs(graph: dict, layer_indices: list[int]) -> tuple[AttentionSpec, ...]:
    specs: list[AttentionSpec] = []
    for layer_index in layer_indices:
        node = _find_node_by_namespace_suffix(
            graph,
            f"layer_{layer_index}.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite",
        )
        attrs = _attrs_to_dict(node["outputsMetadata"][0]["attrs"])
        _, dims = parse_shape(attrs["tensor_shape"])
        if len(dims) != 4:
            raise ValueError(
                f"Expected 4D query_norm output for layer {layer_index}, got {dims}"
            )
        query_heads = dims[2]
        query_head_dim = dims[3]

        specs.append(
            AttentionSpec(
                layer_index=layer_index,
                query_heads=query_heads,
                query_head_dim=query_head_dim,
                notes="External KV mapping is still unresolved; see docs/findings.md.",
            )
        )
    return tuple(specs)


def infer_config_from_graph_export(path: str | Path) -> MtpDrafterConfig:
    export = load_graph_export(path)
    graph = _top_graph(export)
    signature_tensors = _signature_tensors(graph)

    input_activation_dim = parse_shape(signature_tensors["activations"]["tensor_shape"])[1][-1]
    projected_activation_dim = parse_shape(
        signature_tensors["projected_activations"]["tensor_shape"]
    )[1][-1]
    vocab_size = parse_shape(signature_tensors["logits"]["tensor_shape"])[1][-1]

    pre_project_node = _find_node_by_namespace_suffix(
        graph,
        "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/btm,md->btd/dot_general1",
    )
    pre_project_attrs = _attrs_to_dict(pre_project_node["outputsMetadata"][0]["attrs"])
    model_dim = parse_shape(pre_project_attrs["tensor_shape"])[1][-1]

    mlp_node = _find_node_by_namespace_suffix(
        graph,
        "layer_0.post_qkv/mlp/gating_einsum1/btd,df->btf/dot_general1",
    )
    mlp_attrs = _attrs_to_dict(mlp_node["outputsMetadata"][0]["attrs"])
    mlp_hidden_dim = parse_shape(mlp_attrs["tensor_shape"])[1][-1]

    layer_indices = _find_layer_indices(graph)
    base_kv_inputs = tuple(
        name for name in sorted(signature_tensors) if name.startswith("kv_cache_")
    )
    attention_specs = _infer_attention_specs(graph, layer_indices)

    notes = (
        "The graph exposes four drafter blocks, but only kv_cache_{22,23} signature inputs.",
        "This likely means the drafter consumes late base-model KV state instead of owning four independent external KV cache pairs.",
    )

    return MtpDrafterConfig(
        input_activation_dim=input_activation_dim,
        projected_activation_dim=projected_activation_dim,
        model_dim=model_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        num_layers=len(layer_indices),
        vocab_size=vocab_size,
        attention_specs=attention_specs,
        base_kv_inputs=base_kv_inputs,
        notes=notes,
    )
