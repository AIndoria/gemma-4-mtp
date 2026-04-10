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
PYTHONPATH=src python scripts/fetch_artifacts.py
PYTHONPATH=src python scripts/fetch_artifacts.py --include-tflite
PYTHONPATH=src python scripts/inspect_drafter.py
PYTHONPATH=src python scripts/inspect_attention_runtime.py
PYTHONPATH=src python scripts/compare_attention_parity.py
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
- An explicit attention adapter boundary for the still-unresolved part: how the
  drafter consumes the base model KV caches named `kv_cache_22` and
  `kv_cache_23`.

What is still unresolved:

- Exact weight extraction from the TFLite flatbuffer.
- Exact dequantization semantics for every layer.
- The detailed attention/cache mapping inside the drafter blocks.
