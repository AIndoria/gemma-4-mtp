from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .graph import _attrs_to_dict, _top_graph, load_graph_export, parse_shape


@dataclass(frozen=True)
class TensorRef:
    node_id: str
    node_label: str
    tensor_name: str | None
    tensor_shape: tuple[int, ...] | None
    quantization: str | None


@dataclass(frozen=True)
class LinearPlan:
    module_name: str
    namespace: str
    output_shape: tuple[int, ...]
    input_ref: TensorRef | None
    filter_ref: TensorRef | None
    bias_ref: TensorRef | None


@dataclass(frozen=True)
class ParameterPlan:
    module_name: str
    namespace: str
    tensor_ref: TensorRef


def _node_index(graph: dict) -> dict[str, dict]:
    return {node["id"]: node for node in graph["nodes"]}


def _tensor_ref_from_source(
    graph: dict,
    node: dict,
    target_input_id: str,
) -> TensorRef | None:
    source_node_id = None
    source_output_id = None
    for edge in node.get("incomingEdges", []):
        if edge["targetNodeInputId"] == target_input_id:
            source_node_id = edge["sourceNodeId"]
            source_output_id = edge["sourceNodeOutputId"]
            break

    if source_node_id is None or source_output_id is None:
        return None

    source_node = _node_index(graph)[source_node_id]
    metadata = next(
        item for item in source_node.get("outputsMetadata", []) if item["id"] == source_output_id
    )
    attrs = _attrs_to_dict(metadata.get("attrs", []))
    shape = attrs.get("tensor_shape")
    return TensorRef(
        node_id=source_node_id,
        node_label=source_node["label"],
        tensor_name=attrs.get("tensor_name"),
        tensor_shape=parse_shape(shape)[1] if shape else None,
        quantization=attrs.get("quantization"),
    )


def _module_name_from_namespace(namespace: str) -> str | None:
    if namespace == "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/btm,md->btd/dot_general1":
        return "pre_project"

    if namespace == "MtpDrafterModel.mtp_post_project/mtp_post_proj/btd,dm->btm/dot_general1":
        return "post_project"

    if namespace == "MtpDrafterModel.decode_softmax/transformer.decode_softmax/embedder.decode/btd,md->btm/dot_general":
        return "logits_head"

    match = re.match(r"layer_(\d+)/layer_\1\.(.+)", namespace)
    if not match:
        return None

    layer_index = int(match.group(1))
    suffix = match.group(2)

    layer_map = {
        "pre_q/attn.pre_q/attn._pre_attention_query_fn/q_einsum/btd,dH->btH/dot_general1":
            f"blocks.{layer_index}.attention.q_proj",
        "post_qkv/attn.post_qkv/attn_vec_einsum/btH,Hd->btd/dot_general1":
            f"blocks.{layer_index}.attention.o_proj",
        "post_qkv/mlp/gating_einsum1/btd,df->btf/dot_general1":
            f"blocks.{layer_index}.mlp.gate_proj",
        "post_qkv/mlp/gating_einsum2/btd,df->btf/dot_general1":
            f"blocks.{layer_index}.mlp.up_proj",
        "post_qkv/mlp/linear/btf,fd->btd/dot_general1":
            f"blocks.{layer_index}.mlp.down_proj",
    }
    return layer_map.get(suffix)


def extract_linear_plan(path: str | Path) -> list[LinearPlan]:
    export = load_graph_export(path)
    graph = _top_graph(export)
    plans: list[LinearPlan] = []

    for node in graph["nodes"]:
        if node.get("label") != "fully_connected":
            continue
        module_name = _module_name_from_namespace(node.get("namespace", ""))
        if module_name is None:
            continue

        output_attrs = _attrs_to_dict(node["outputsMetadata"][0]["attrs"])
        output_shape = parse_shape(output_attrs["tensor_shape"])[1]

        plans.append(
            LinearPlan(
                module_name=module_name,
                namespace=node["namespace"],
                output_shape=output_shape,
                input_ref=_tensor_ref_from_source(graph, node, "0"),
                filter_ref=_tensor_ref_from_source(graph, node, "1"),
                bias_ref=_tensor_ref_from_source(graph, node, "2"),
            )
        )

    return plans


def extract_norm_plan(path: str | Path) -> list[ParameterPlan]:
    export = load_graph_export(path)
    graph = _top_graph(export)

    namespace_to_module: dict[str, str] = {
        "mtp_final_norm/composite": "final_norm.weight",
    }
    layer_indices = sorted(
        {
            int(match.group(1))
            for node in graph["nodes"]
            for match in [re.search(r"layer_(\d+)", node.get("namespace", ""))]
            if match is not None
        }
    )
    for layer_index in layer_indices:
        namespace_to_module[
            f"layer_{layer_index}/layer_{layer_index}.pre_q/pre_attention_norm/composite"
        ] = f"blocks.{layer_index}.pre_attn_norm.weight"
        namespace_to_module[
            f"layer_{layer_index}/layer_{layer_index}.pre_q/attn.pre_q/attn._pre_attention_query_fn/query_norm/composite"
        ] = f"blocks.{layer_index}.attention.query_norm.weight"
        namespace_to_module[
            f"layer_{layer_index}/layer_{layer_index}.post_qkv/post_attention_norm/composite"
        ] = f"blocks.{layer_index}.post_attn_norm.weight"
        namespace_to_module[
            f"layer_{layer_index}/layer_{layer_index}.post_qkv/pre_ffw_norm/composite"
        ] = f"blocks.{layer_index}.pre_ffw_norm.weight"
        namespace_to_module[
            f"layer_{layer_index}/layer_{layer_index}.post_qkv/post_ffw_norm/composite"
        ] = f"blocks.{layer_index}.post_ffw_norm.weight"

    plans: list[ParameterPlan] = []
    for node in graph["nodes"]:
        namespace = node.get("namespace", "")
        module_name = namespace_to_module.get(namespace)
        if module_name is None:
            continue
        tensor_ref = _tensor_ref_from_source(graph, node, "1")
        if tensor_ref is None:
            raise ValueError(f"Norm node {namespace} did not expose a weight input")
        plans.append(
            ParameterPlan(
                module_name=module_name,
                namespace=namespace,
                tensor_ref=tensor_ref,
            )
        )

    return plans
