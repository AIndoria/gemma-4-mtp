from __future__ import annotations

import argparse
import json
from dataclasses import asdict

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
    print(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
