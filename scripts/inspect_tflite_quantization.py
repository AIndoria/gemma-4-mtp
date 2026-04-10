from __future__ import annotations

import argparse

from gemma_mtp import TFLiteModelReader


def describe_quantization(name: str, quant: object) -> str:
    if quant is None:
        return f"{name}: quant=None"

    return (
        f"{name}: scale={quant.scale.tolist()} "
        f"zero_point={quant.zero_point.tolist()} "
        f"quantized_dimension={quant.quantized_dimension}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tflite",
        default="data/hf/Section11_TFLiteModel_tf_lite_mtp_drafter.tflite",
    )
    args = parser.parse_args()

    reader = TFLiteModelReader(args.tflite)

    print("Inputs")
    for info in reader.input_tensor_infos():
        print(
            f"- {info.name} shape={info.shape} tensor_type={info.tensor_type} "
            f"buffer_index={info.buffer_index}"
        )
        print(f"  {describe_quantization(info.name, info.quantization)}")

    print("Outputs")
    for info in reader.output_tensor_infos():
        print(
            f"- {info.name} shape={info.shape} tensor_type={info.tensor_type} "
            f"buffer_index={info.buffer_index}"
        )
        print(f"  {describe_quantization(info.name, info.quantization)}")


if __name__ == "__main__":
    main()
