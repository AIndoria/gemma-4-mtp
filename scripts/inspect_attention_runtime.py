from __future__ import annotations

import argparse
import json

from gemma_mtp import infer_config_from_graph_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_json",
        nargs="?",
        default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json",
    )
    args = parser.parse_args()

    config = infer_config_from_graph_export(args.graph_json)
    rows = []
    for spec in config.attention_specs:
        rows.append(
            {
                "layer_index": spec.layer_index,
                "query_heads": spec.query_heads,
                "kv_heads": spec.kv_heads,
                "queries_per_kv": spec.queries_per_kv,
                "query_head_dim": spec.query_head_dim,
                "key_cache_name": spec.key_cache_name,
                "value_cache_name": spec.value_cache_name,
                "local_window_size": spec.local_window_size,
                "notes": spec.notes,
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
