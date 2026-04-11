# Gemma 4 MTP Reverse-Engineering Workspace

This workspace is a starting point for reconstructing the `mtp_drafter` TFLite
graph from `shadowlilac/gemma-4-e4b-mtp-extraction-effort` into a usable
PyTorch module.

Current status:

- The runtime contract has been documented in [docs/findings.md](/home/aindoria/Projects/gemma-4-mtp/docs/findings.md).
- Artifact download and graph inspection scripts live in `scripts/`.
- A first PyTorch scaffold lives in `src/gemma_mtp/`.

Quick start:

```bash
python -m pip install --target .vendor flatbuffers tflite
python3.12 -m venv .venv312
.venv312/bin/python -m pip install ai-edge-litert
PYTHONPATH=src python scripts/fetch_artifacts.py
PYTHONPATH=src python scripts/fetch_artifacts.py --include-tflite
PYTHONPATH=src python scripts/inspect_drafter.py
PYTHONPATH=src python scripts/inspect_attention_runtime.py
PYTHONPATH=src python scripts/compare_attention_parity.py
PYTHONPATH=.vendor:src python scripts/inspect_tflite_quantization.py
PYTHONPATH=.vendor:src python scripts/compare_quantized_attention_parity.py
PYTHONPATH=.vendor:src python scripts/compare_tflite_runtime.py --zero-attn-baseline
PYTHONPATH=.vendor:src python scripts/compare_internal_tflite_runtime.py
PYTHONPATH=src python scripts/extract_linear_plan.py
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=.vendor:src python scripts/export_partial_state_dict.py
```

What exists today:

- A graph parser that infers the drafter's top-level shapes from the exported
  AI Edge Model Explorer JSON.
- A `GemmaMtpDrafter` scaffold that models the known pre-projection, four
  drafter blocks, final norm, logits head, and post-projection.
- A TFLite flatbuffer reader that can extract and dequantize known linears from
  the real `mtp_drafter.tflite` file.
- Recovered external KV-cache quantization metadata:
  - `kv_cache_v_22`: INT8, scale `0.047244105488061905`
  - `kv_cache_k_22`: INT8, scale `0.00596147496253252`
  - `kv_cache_v_23`: INT8, scale `0.01785714365541935`
  - `kv_cache_k_23`: INT8, scale `0.001090860809199512`
- A real LiteRT invoke harness that runs the `.tflite` in an isolated Python
  3.12 env and compares actual TFLite outputs against the current PyTorch port.
- A block-level internal parity harness that can compare teacher-forced and
  self-fed execution against preserved TFLite tensors.
- Runtime quantization recovery for `pre_project`, `q_proj`, the MLP linears,
  `o_proj`, and `post_project`.

## Current Status

The reverse-engineered PyTorch port is now very close to the real LiteRT
runtime on the tested decode positions.

Latest checked parity:

- position `50`
  - logits cosine `0.9978`
  - projected activations cosine `1.0000`
  - top-1 match `True`
- position `700`
  - logits cosine `0.9936`
  - projected activations cosine `0.9980`
  - top-1 match `True`
- position `1000`
  - logits cosine `0.9889`
  - projected activations cosine `0.9959`
  - top-1 match `True`

The detailed running log and current caveats live in `docs/findings.md`.

## License

- Original repository code and documentation are licensed under
  `AGPL-3.0-only`. See `LICENSE`.
- Gemma-derived artifacts and upstream materials are not relicensed by this
  repo. See `NOTICE`.
