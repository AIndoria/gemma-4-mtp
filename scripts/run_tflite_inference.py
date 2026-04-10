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
    parser.add_argument(
        "--preserve-all-tensors",
        action="store_true",
    )
    parser.add_argument(
        "--dump-tensor",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    with np.load(args.inputs) as data:
        inputs = {name: data[name] for name in data.files}

    if args.dump_tensor:
        interpreter = Interpreter(
            model_path=args.tflite_model,
            experimental_preserve_all_tensors=args.preserve_all_tensors,
        )
        interpreter.allocate_tensors()
        input_details = {detail["name"]: detail["index"] for detail in interpreter.get_input_details()}
        for name, value in inputs.items():
            full_name = f"mtp_drafter_{name}:0"
            interpreter.set_tensor(input_details[full_name], value)
            if "activations" in name:
                print(f"Input {name} RMS after set: {np.sqrt(np.mean(interpreter.get_tensor(input_details[full_name]).astype(np.float32)**2))}")
            if "kv_cache_k_22" in name:
                print(f"Input {name} RMS after set: {np.sqrt(np.mean(interpreter.get_tensor(input_details[full_name]).astype(np.float32)**2))}")
        interpreter.invoke()

        tensor_details = {detail["name"]: detail["index"] for detail in interpreter.get_tensor_details()}
        outputs = {
            name: interpreter._interpreter.GetTensor(tensor_details[name], 0)
            for name in args.dump_tensor
        }
    else:
        interpreter = Interpreter(model_path=args.tflite_model)
        runner = interpreter.get_signature_runner(args.signature_key)
        outputs = runner(**inputs)

    output_path = Path(args.outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **outputs)


if __name__ == "__main__":
    main()
