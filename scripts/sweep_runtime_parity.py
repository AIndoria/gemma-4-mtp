from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from compare_tflite_runtime import _build_synthetic_inputs, _run_pytorch
from gemma_mtp import TFLiteModelReader


def _run_tflite(
    *,
    litert_python: str,
    tflite_model: str,
    inputs_path: Path,
    outputs_path: Path,
) -> dict[str, np.ndarray]:
    subprocess.run(
        [
            litert_python,
            "scripts/run_tflite_inference.py",
            "--tflite-model",
            tflite_model,
            "--inputs",
            str(inputs_path),
            "--outputs",
            str(outputs_path),
        ],
        check=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "VIRTUAL_ENV"}
        },
    )

    with np.load(outputs_path) as data:
        return {name: data[name] for name in data.files}


def _evaluate_case(
    *,
    graph_json: str,
    tflite_model: str,
    litert_python: str,
    decode_position: int,
    seed: int,
    workdir: Path,
) -> dict[str, float | int]:
    reader = TFLiteModelReader(tflite_model)
    inputs = _build_synthetic_inputs(
        reader,
        seed=seed,
        decode_position=decode_position,
    )

    inputs_path = workdir / f"inputs_pos{decode_position}_seed{seed}.npz"
    outputs_path = workdir / f"outputs_pos{decode_position}_seed{seed}.npz"
    np.savez(inputs_path, **inputs)
    tflite_outputs = _run_tflite(
        litert_python=litert_python,
        tflite_model=tflite_model,
        inputs_path=inputs_path,
        outputs_path=outputs_path,
    )
    pytorch_outputs = _run_pytorch(graph_json, tflite_model, inputs)

    logits = pytorch_outputs["logits"].detach().cpu()
    projected = pytorch_outputs["projected_activations"].detach().cpu()
    logits_ref = torch.from_numpy(tflite_outputs["logits"])
    projected_ref = torch.from_numpy(tflite_outputs["projected_activations"])

    logits_cosine = torch.nn.functional.cosine_similarity(
        logits.flatten().float(),
        logits_ref.flatten().float(),
        dim=0,
    )
    projected_cosine = torch.nn.functional.cosine_similarity(
        projected.flatten().float(),
        projected_ref.flatten().float(),
        dim=0,
    )

    return {
        "position": decode_position,
        "seed": seed,
        "logits_cosine": float(logits_cosine),
        "projected_cosine": float(projected_cosine),
        "logits_top1_match": int(int(logits.argmax()) == int(logits_ref.argmax())),
        "logits_mean_abs_diff": float((logits - logits_ref).abs().mean()),
        "projected_mean_abs_diff": float((projected - projected_ref).abs().mean()),
    }


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
        "--litert-python",
        default=".venv312/bin/python",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=[0, 1, 10, 50, 100, 256, 511, 512, 700, 1000, 1500],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--workdir",
        default="data/derived/runtime_sweep",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int]] = []
    for position in args.positions:
        for seed in args.seeds:
            result = _evaluate_case(
                graph_json=args.graph_json,
                tflite_model=args.tflite_model,
                litert_python=args.litert_python,
                decode_position=position,
                seed=seed,
                workdir=workdir,
            )
            results.append(result)
            print(
                f"pos={position: <4} seed={seed: <2} "
                f"logits_cosine={result['logits_cosine']:.6f} "
                f"projected_cosine={result['projected_cosine']:.6f} "
                f"top1={bool(result['logits_top1_match'])}"
            )

    top1_matches = sum(int(result["logits_top1_match"]) for result in results)
    min_logits_cosine = min(float(result["logits_cosine"]) for result in results)
    min_projected_cosine = min(float(result["projected_cosine"]) for result in results)

    print("")
    print(f"cases={len(results)}")
    print(f"top1_match_rate={top1_matches}/{len(results)}")
    print(f"min_logits_cosine={min_logits_cosine:.6f}")
    print(f"min_projected_cosine={min_projected_cosine:.6f}")

    failing = [
        result
        for result in results
        if not int(result["logits_top1_match"])
    ]
    if failing:
        print("failing_cases:")
        for result in failing:
            print(
                f"  pos={int(result['position'])} seed={int(result['seed'])} "
                f"logits_cosine={float(result['logits_cosine']):.6f} "
                f"projected_cosine={float(result['projected_cosine']):.6f}"
            )


if __name__ == "__main__":
    main()
