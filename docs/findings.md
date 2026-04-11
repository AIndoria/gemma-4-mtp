# Findings

Date: 2026-04-11

## Current Status

The reconstruction is no longer in the "mystery architecture" phase. The main
module layout, weight extraction, grouped-query attention structure, quantized
`o_proj` runtime path, and the final block's unusual partial RoPE are all now
recovered well enough to reproduce individual blocks at very high parity.

What is **not** true yet is "fully verified end-to-end." The remaining gap is
best described as **accumulated hidden-state drift**, not a missing top-level
module or a broken output head.

## Current Best Understanding

1. **Block topology is verified.**
   - `hidden = hidden + post_attn_norm(attn(pre_attn_norm(hidden)))`
   - `hidden = hidden + post_ffw_norm(mlp(pre_ffw_norm(hidden)))`
2. **MLP structure is verified as GEGLU with tanh-approx GELU.**
3. **Attention partitioning is verified.**
   - layers `0-2` use `kv_cache_{k,v}_22`
   - layer `3` uses `kv_cache_{k,v}_23`
4. **Windowing is verified as sliding logical attention over the raw cache.**
   - local layers attend to `position - 511 <= key_index <= position`
   - the LiteRT `runtime_bmm` subgraphs derive the slice start from the decode
     position, not from a fixed `0:512` window
5. **Final-block RoPE is only partially active.**
   - the layer-3 RoPE constant has length `256`, but only the first `64`
     frequencies are non-zero
   - in practice this means the last block rotates only the first `64` dims of
     each 256-d half, i.e. `128` total rotary dims inside a `512`-wide head

## Latest Milestone

The newest checkpoint recovered the missing quantization contracts for the
remaining linear stack:

- `pre_project`: input-side int8 snap plus output-side int8 snap
- all `q_proj` linears: input-side int8 snap
- all MLP linears (`gate_proj`, `up_proj`, `down_proj`): input-side int8 snap
- `post_project`: input-side int8 snap

The important lesson from this pass is that the direction of quantization
matters:

- for `q_proj`, MLP, and `post_project`, **input-side** snapping was the real
  missing contract
- for `pre_project`, the correct behavior required both the input and output
  quantized path; naive one-sided approximations were misleading

With those contracts in place, the tested runtime path is now very close to
TFLite, but broader stress testing shows a small number of remaining top-1
flips under harder synthetic seeds.

### Teacher-forced parity at position 700

Using exact TFLite hidden states as inputs:

- `layer_1.block_out`: cosine `1.0000`
- `layer_2.block_out`: cosine `1.0000`
- `layer_3.query_rope`: cosine `1.0000`
- `layer_3.attn_context`: cosine `1.0000`
- `layer_3.attn_out`: cosine `1.0000`
- `layer_3.block_out`: cosine `1.0000`

This is strong evidence that the recovered block topology, grouped-query
attention path, quantized `o_proj`, and final-block partial RoPE are all
substantially correct.

### End-to-end parity snapshot

Using `scripts/compare_tflite_runtime.py` on current code:

- position `50`
  - logits cosine: `0.9978`
  - projected activations cosine: `1.0000`
  - top-1 match: `True`
- position `700`
  - logits cosine: `0.9936`
  - projected activations cosine: `0.9980`
  - top-1 match: `True`
- position `1000`
  - logits cosine: `0.9889`
  - projected activations cosine: `0.9959`
  - top-1 match: `True`

This is the first checkpoint where parity is consistently very strong across
shallow, middle, and deeper decode positions **and** all three tested positions
agree on top-1 token selection.

### Broader runtime sweep

The repo now includes `scripts/sweep_runtime_parity.py`, which automates the
end-to-end TFLite-vs-PyTorch comparison across multiple decode positions and
RNG seeds.

On an updated stress pass over positions `50`, `700`, `1000`, and `1500` with
seeds `0-4`:

- cases checked: `20`
- top-1 matches: `18/20`
- minimum logits cosine: `0.9885`
- minimum projected-activations cosine: `0.9817`

Failing cases from that sweep:

- position `1500`, seed `0`
- position `1500`, seed `3`

These failures are not wide divergences. In each case, cosine remains strong
and the wrong prediction comes from a winner flip among a small set of
high-scoring candidates.

### Current best explanation for the remaining gap

The broad architecture now looks settled. The remaining problem is small
self-fed hidden-state drift that compounds across layers for some long-context
inputs.

For one failing case (`position 1500`, `seed 3`), the self-fed layer outputs
compare to TFLite as:

- `layer_0_out`: cosine `1.0000`
- `layer_1_out`: cosine `0.9999`
- `layer_2_out`: cosine `0.9948`
- `layer_3_out`: cosine `0.9930`

That pattern suggests the current model is very close, but not yet bit-stable
enough to guarantee identical top-1 ranking across all tested synthetic cases.

### What the newest isolation checks show

Two additional points are now clear:

1. **`pre_project` was still a real source of one-bin mismatch.**
   - switching from rounded graph-export scales to exact internal TFLite tensor
     quantization improved the stress sweep from `17/20` to `18/20`
   - two former failing cases (`position 700, seed 1` and `position 1500,
     seed 1`) now match top-1
2. **The final remaining drift is no longer in `pre_project` or block 0.**
   - for the stubborn long-context failures, `pre_project` is now exact
   - block 0 is exact or effectively exact when compared directly against
     preserved TFLite tensors
   - the first visible self-fed drift now starts in layer 1 and then compounds
     through layers 2 and 3

### Final heads are not the main problem

If exact TFLite final hidden states are fed into our recovered output path:

- `final_norm`: cosine `1.0000`
- `logits_head`: cosine `0.9987`
- `post_project`: cosine `0.9850`

That means the remaining end-to-end gap was mostly **hidden-state drift from
missing quantization contracts inside the recurrent stack**, not a bad final
projection.

## Remaining Work

The highest-value unresolved area is now narrow:

1. reduce the small self-fed drift that still appears in some longer-context
   synthetic cases
2. determine whether that drift comes from one last runtime contract in block 0
   or from accumulated float-vs-quantized behavior across all blocks
3. broaden verification further, ideally with more decode positions and less
   synthetic activation traces
4. decide when the reconstruction is strong enough to export as the default
   recovered PyTorch implementation

## Implementation Status

- `44/44` known linear/norm parameters are still mapped into
  `data/derived/mtp_partial_state_dict.pt`
- the PyTorch scaffold includes:
  - grouped-query attention over external KV caches
  - quantized `o_proj` runtime emulation
  - input-quantized `q_proj`, MLP, and `post_project` runtime emulation
  - exact quantized `pre_project` runtime emulation
  - recovered layer-3 partial rotary behavior
- the repo also now includes `scripts/compare_teacher_forced_blocks.py` for
  localizing parity by block using preserved TFLite intermediates
- the repo also now includes `scripts/sweep_runtime_parity.py` for automated
  position/seed stress testing

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

## Quantized KV Cache Findings

The external cache contract is clearer now because the real TFLite input
metadata exposes the quantization parameters directly.

- `kv_cache_v_22`
  - tensor name: `mtp_drafter_kv_cache_v_22:0`
  - quantization: per-tensor INT8
  - scale: `0.047244105488061905`
  - zero point: `0`
- `kv_cache_k_22`
  - tensor name: `mtp_drafter_kv_cache_k_22:0`
  - quantization: per-tensor INT8
  - scale: `0.00596147496253252`
  - zero point: `0`
- `kv_cache_v_23`
  - tensor name: `mtp_drafter_kv_cache_v_23:0`
  - quantization: per-tensor INT8
  - scale: `0.01785714365541935`
  - zero point: `0`
- `kv_cache_k_23`
  - tensor name: `mtp_drafter_kv_cache_k_23:0`
  - quantization: per-tensor INT8
  - scale: `0.001090860809199512`
  - zero point: `0`

This matters because it means the adapter can consume raw cache tensors without
needing any per-channel handling for these external inputs.

## Quantized Cache Parity Check

The repo now also includes:

- `scripts/inspect_tflite_quantization.py`
  - prints the real TFLite input/output quantization contract
- `scripts/compare_quantized_attention_parity.py`
  - verifies that feeding raw INT8 KV caches plus recovered scales through the
    adapter matches feeding explicitly dequantized float caches

Current result:

- local quantized max abs diff: `0.0`
- full quantized max abs diff: `0.0`

## Real TFLite Runtime Comparison

The repo now includes a two-part runtime harness:

- `scripts/run_tflite_inference.py`
  - runs the real `.tflite` model through `ai-edge-litert`
- `scripts/compare_tflite_runtime.py`
  - generates synthetic full-shape inputs
  - invokes the real TFLite model in an isolated Python 3.12 env
  - runs the current PyTorch port on the exact same inputs
  - compares output deltas
  - can optionally report a zero-attention baseline

Current result with `decode_position=700`, full raw INT8 caches, and the
zero-attention baseline enabled:

- logits:
  - max abs diff: `41.72`
  - mean abs diff: `7.84`
  - cosine similarity: `0.319`
  - top-1 argmax match: `False`
  - zero-attention mean abs diff: `9.37`
- projected activations:
  - max abs diff: `42.67`
  - mean abs diff: `3.31`
  - cosine similarity: `0.396`
  - zero-attention mean abs diff: `3.34`

Interpretation:

- the recovered attention path is helping
  - logits improve relative to the zero-attention baseline
  - projected activations improve sharply in max error and cosine after adding
    recovered query-side RoPE
- but the current port is still not numerically close to the real model
- that makes the remaining gap more likely to live in one or more of:
  - exact runtime attention/KV-cache semantics inside `dot_attn._qkv_fn`
  - whether the external key/value caches need additional runtime-side handling
  - residual ordering or post-attention/post-ffw normalization behavior
  - any hidden per-op quantization / requantization semantics we are still
    treating as plain float math

## Internal Layer-0 Parity Localization

`scripts/compare_internal_tflite_runtime.py` now compares selected internal
TFLite tensors against the current PyTorch block-0 path using the exact same
saved synthetic inputs from `scripts/compare_tflite_runtime.py`.

Current result:

- very close through the query path
  - `pre_project`: cosine `0.9990`
  - `block0.pre_attn_norm`: cosine `0.9990`
  - `block0.q_proj`: cosine `0.9987`
  - `block0.query_norm`: cosine `0.9987`
  - `block0.query_rope`: cosine `0.9987`
  - `block0.grouped_query`: cosine `0.9987`
- divergence begins inside attention proper
  - `block0.attn_context_grouped`: cosine `0.5008`
  - `block0.attn_out`: cosine `0.5972`
  - `block0.post_attn_norm`: cosine `0.4551`
  - `block0.out`: cosine `0.4735`
  - `final_norm`: cosine `0.1960`

This is the strongest current localization signal. The remaining work is not
the query projection or RoPE path anymore. It is the exact behavior of the
custom attention runtime over the external KV caches.

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
- exact source of the remaining runtime parity gap

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
4. Trace the remaining runtime parity gap by instrumenting per-block outputs and
   matching one unresolved component at a time.
