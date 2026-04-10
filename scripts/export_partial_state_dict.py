from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gemma_mtp.convert import build_partial_state_dict
from gemma_mtp import GemmaMtpDrafter, infer_config_from_graph_export


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
        "--output",
        default="data/derived/mtp_partial_state_dict.pt",
    )
    args = parser.parse_args()

    state_dict = build_partial_state_dict(args.graph_json, args.tflite_model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)

    config = infer_config_from_graph_export(args.graph_json)
    model = GemmaMtpDrafter(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"saved {output_path}")
    print(f"loaded_keys {len(state_dict)}")
    print(f"missing_keys {len(missing)}")
    print(f"unexpected_keys {len(unexpected)}")
    print("sample_missing", missing[:12])


if __name__ == "__main__":
    main()
