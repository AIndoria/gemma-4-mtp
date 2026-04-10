from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gemma_mtp.plan import extract_linear_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_json",
        nargs="?",
        default="data/hf/extracted/mtp_graph_json_aiedge_model_explorer_extracted.json",
    )
    args = parser.parse_args()

    plans = extract_linear_plan(args.graph_json)
    print(json.dumps([asdict(plan) for plan in plans], indent=2))


if __name__ == "__main__":
    main()
