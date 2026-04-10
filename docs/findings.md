# Findings

Date: 2026-04-10

## Summary

The extracted Gemma 4 `mtp_drafter` looks recoverable. The biggest shift is
that the open LiteRT-LM runtime already reveals the speculative decoding
protocol, so the remaining work is mostly graph-to-module reconstruction and
weight handling rather than guessing how MTP works.

## Sources Consulted

- Hugging Face repo:
  `https://huggingface.co/shadowlilac/gemma-4-e4b-mtp-extraction-effort`
- LiteRT-LM runtime:
  `runtime/executor/llm_litert_mtp_drafter.h`
- LiteRT-LM runtime:
  `runtime/executor/llm_litert_mtp_drafter.cc`
- LiteRT-LM runtime:
  `runtime/executor/llm_litert_compiled_model_executor.cc`

## What The Runtime Code Confirms

`LlmLiteRtMtpDrafter` makes the draft/verify loop explicit:

1. The drafter takes:
   - `input_pos`
   - `mask`
   - `activations`
   - several `kv_cache_*` buffers
   - optional `param_tensor`
2. The drafter predicts multiple candidate tokens greedily.
3. The base model `verify` signature re-scores the original input token plus
   drafted tokens.
4. Verification accepts drafted tokens until the first mismatch.
5. The first mismatch becomes the "bonus token".

The runtime also shows a crucial hidden-state contract:

- Drafter input `activations` is a concatenation of:
  - token embedding lookup
  - previous projected activations
- Drafter output `projected_activations` is fed back into the next draft step.

## What The Exported Graph Confirms

The AI Edge Model Explorer JSON is structured enough to recover the high-level
module layout.

### Top-level signature tensors

Inputs:

- `input_pos`: `int32[1]`
- `activations`: `float32[1,1,5120]`
- `mask`: `bool[1,1,1,32003]`
- `param_tensor`: `int32[1,1,1,7]`
- `kv_cache_v_22`: `int8[1,2,256,32003]`
- `kv_cache_k_22`: `int8[1,2,32003,256]`
- `kv_cache_v_23`: `int8[1,2,512,32003]`
- `kv_cache_k_23`: `int8[1,2,32003,512]`

Outputs:

- `logits`: `float32[1,1,262144]`
- `projected_activations`: `float32[1,1,2560]`

### Inferred high-level architecture

Top-level named components in the graph:

- `MtpDrafterModel.mtp_pre_project`
- `layer_0`
- `layer_1`
- `layer_2`
- `layer_3`
- `mtp_final_norm`
- `MtpDrafterModel.decode_softmax`
- `MtpDrafterModel.mtp_post_project`

This strongly suggests:

- a pre-projection from the 5120-wide activation input into a small internal
  drafter width
- 4 transformer-like blocks
- a final norm
- a logits projection
- a post-projection back into the 2560-wide projected activation space

### Shapes recovered from named linears

Pre/post projection:

- pre-project output: `int8[1,1,256]`
- post-project output before dequant: `int8[1,1,2560]`

Per-block feed-forward shapes:

- gating branch 1: `int8[1,1,2048]`
- gating branch 2: `int8[1,1,2048]`
- mlp output: `int8[1,1,256]`

Per-block attention clues:

- layers 0-2 query path normalizes to `float32[1,1,4,256]`
- layer 3 query path normalizes to `float32[1,1,4,512]`
- layers 0-2 query projection output: `int8[1,1,1024]`
- layer 3 query projection output: `int8[1,1,2048]`

These are consistent with:

- internal drafter width: `256`
- MLP hidden width: `2048`
- 4 query heads in each block
- larger attention head size in the last block

## Most Important Interpretation

The external KV inputs are named `kv_cache_22` and `kv_cache_23`, while the
drafter graph itself contains `layer_0` through `layer_3`.

The most plausible reading is:

- the drafter is not a completely standalone 4-layer transformer with its own
  4 external KV-cache pairs
- instead, it is a compact drafter stacked on top of the base Gemma model's
  late hidden state / late KV state
- the external caches likely belong to base-model layers 22 and 23

That boundary matters because it means the initial PyTorch port should not hard
code a guessed internal KV implementation everywhere. It should isolate that
part behind an adapter interface.

## Attention / KV Findings

Further inspection of the real `mtp_drafter.tflite` clarifies the attention
story substantially.

### Cache partitioning

- `layer_0`, `layer_1`, and `layer_2` all consume:
  - `kv_cache_k_22`
  - `kv_cache_v_22`
- `layer_3` alone consumes:
  - `kv_cache_k_23`
  - `kv_cache_v_23`

This is not four separate external cache pairs. It is a 4-block drafter that
reuses late base-model cache state.

### Grouped-query layout

The reshaped query tensors are:

- layers 0-2: `float32[1,2,2,256]`
- layer 3: `float32[1,2,2,512]`

This implies:

- KV heads: `2`
- queries per KV head: `2`
- total query heads: `4`

So the attention is best interpreted as grouped-query attention rather than
plain 4-head attention with 4 independent KV heads.

### Sliding-window vs full-context

The `runtime_bmm` subgraphs reveal two behaviors:

- layers 0-2 use a sliced runtime path that crops the cache to a 512-token
  window before the batched matmul
- layer 3 uses a direct dequantize + batched matmul path over the full cache

So the current best interpretation is:

- layers 0-2: local attention window of `512`
- layer 3: full-context attention

The local-window mask is not vague anymore. The constants show:

- `arith.constant15 = [0, 1, 2, ..., 32002]`
- `arith.constant14 = [512, 513, 514, ..., 32514]`
- mask rule:
  - `input_pos >= arange(0, 32003)`
  - `input_pos < arange(0, 32003) + 512`

So the allowed local positions are exactly:

- `position - 511 <= key_index <= position`

or equivalently:

- `start = max(0, position + 1 - 512)`
- `end = position + 1`

### `param_tensor`

The LiteRT runtime helper fills `param_tensor` as:

- `[start_index, end_index, end_index]`

The runtime comments state:

- the first 2 parameters are for cache update
- the 3rd parameter is the end channel index for `runtime_batched_matmul`

In practice, this is what the sliced `runtime_bmm` subgraphs use to choose the
active local-attention window.

## Attention Parity Check

The repo now includes `scripts/compare_attention_parity.py`, which checks:

- local-window exact full-length masking vs sliced-window formulation
- full-context masked attention vs straightforward masked softmax attention

Current result:

- local max abs diff: `2.38e-07`
- full max abs diff: `0.0`

That gives good confidence that the first-pass PyTorch attention helper matches
the TFLite graph semantics closely for the score/value path.

## Confidence

High confidence:

- overall draft/verify protocol
- existence of pre-project, 4 blocks, final norm, logits head, post-project
- main tensor shapes: `5120 -> 256 -> 2560`, vocab `262144`
- gated MLP structure

Medium confidence:

- 4 query heads per block
- last block using larger per-head width than the first 3

Low confidence / open questions:

- exact attention implementation inside each block
- exact mapping from drafter blocks to the base model KV cache tensors
- exact dequantization and weight packing rules for every tensor
- exact activation function variant inside `mlp/activation_fn/composite`

## What I Started Implementing

This workspace now includes:

- `scripts/fetch_artifacts.py`
  - pulls the HF metadata and graph export locally
- `scripts/inspect_drafter.py`
  - infers the drafter config from the graph JSON
- `src/gemma_mtp/graph.py`
  - parser and config inference logic
- `src/gemma_mtp/module.py`
  - first PyTorch scaffold with an explicit external attention adapter boundary

## Recommended Next Steps

1. Download the full `mtp_drafter.tflite` file locally.
2. Build a TFLite flatbuffer walker that extracts:
   - tensor metadata
   - operator list
   - buffer payloads
   - quantization parameters
3. Map the named linear weights into the current PyTorch scaffold.
4. Replace the placeholder attention adapter with a real implementation once
   cache usage is decoded.
