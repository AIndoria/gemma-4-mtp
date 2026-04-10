from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tflite-model",
        default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite",
    )
    parser.add_argument(
        "--inputs",
        required=True,
    )
    parser.add_argument(
        "--outputs",
        required=True,
    )
    parser.add_argument(
        "--signature-key",
        default="mtp_drafter",
    )
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.tflite_model)
    runner = interpreter.get_signature_runner(args.signature_key)

    with np.load(args.inputs) as data:
        inputs = {name: data[name] for name in data.files}

    outputs = runner(**inputs)

    output_path = Path(args.outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **outputs)


if __name__ == "__main__":
    main()
