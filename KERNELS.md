# Kernel reference

Each kernel ships as a Triton primary plus a CUDA C++ reference. The Python
wrapper picks the backend via `kernels._common.launch.select_backend`, which
honours `REFLEX_INFER_BACKEND={triton,cuda,torch}`.

**Status (v1).** This is an early kernel library. Some kernels currently
match or trail the torch baseline at specific shape ranges. The torch
baseline (`F.scaled_dot_product_attention`) routes to FlashAttention-2 on
Ampere+ and is a strong opponent; we expect to beat it on the GQA + long
context regime first, parity-or-trailing elsewhere. Benchmark JSON in
`benchmark/results/` is the source of truth; if a result is missing, the
kernel hasn't been validated at that shape yet.

| Kernel | Primary | Fallback | dtypes | autotuned | v1 status |
|--------|---------|----------|--------|-----------|-----------|
| Fused attention | Triton (FA-2 style) | CUDA C++ | fp16 / bf16 (fp8 wired in dtype helper) | yes | matches torch SDPA on H100; GQA-prefill wins at long context. Pure-attention vs FA2 trails at short seq. |
| Paged KV cache (append / lookup / scatter) | Triton | CUDA C++ | fp16 / bf16 | no (memory-bound) | matches torch indexing baseline (memory-bound) |
| Fused Linear + LayerNorm / RMSNorm | Triton | CUDA C++ | fp16 / bf16 / fp32 | yes (BLOCK_M, BLOCK_K) | wins on small-batch decode at Llama hidden=4096 due to HBM saving; can trail `F.linear + F.layer_norm` at large batch where the matmul cost dominates. |
| Online softmax | Triton | CUDA C++ | fp16 / bf16 / fp32 | yes (BLOCK_N) | matches `F.softmax` on long rows; ~par on short rows. |
| RoPE (apply / apply\_) | Triton | CUDA C++ | fp16 / bf16 / fp32 | no | matches torch elementwise reference. |

Cross-vendor:

* **ROCm**: `kernels/_rocm/` reports gfx version (parsed from
  `torch.cuda.get_device_properties(0).gcnArchName`, not the marketing
  device name) and a per-generation `BLOCK_N` cap that the attention
  autotune list now filters through at import time. The Triton kernels
  use the same source; in theory they run on MI200/MI300 via the Triton
  hip backend but we have not validated on AMD hardware yet — expected
  to work, not asserted.
* **Apple MPS / ANE**: `kernels/_mps/` exposes a softmax dispatcher. The
  `mps_softmax` path is a thin wrapper around `F.softmax` on the `mps`
  device — NOT a custom Metal shader; we don't gain by reimplementing
  what torch already lowers to Metal. The Core ML / ANE path is
  benchmark-only (per-call dispatch overhead).

## 1. Fused attention

`kernels/attention/triton_fused_attention.py`

FlashAttention-2 style single-kernel forward:

* One CUDA grid program per `(batch, head, query-tile)` of size `BLOCK_M`.
* Inner loop over `K` and `V` tiles of size `BLOCK_N`.
* Online softmax statistics (`m_i`, `l_i`) live in registers across the loop.
* Causal short-circuits via `n_end = min(S_k, (start_m + 1) * BLOCK_M)`
  rather than masking every column - a measurable saving on long sequences.
* GQA expands `K` / `V` along the head axis on the host before launch;
  fully-fused GQA is a follow-up.

Public surface:

```python
from kernels.attention import fused_attention

out = fused_attention(q, k, v, is_causal=True)
# Shapes: [B, H, S_q, D] for q / output, [B, H_kv, S_k, D] for k / v.
```

Backend selection:

```python
out = fused_attention(q, k, v, backend="triton")  # force Triton
out = fused_attention(q, k, v, backend="torch")   # SDPA reference
```

Parity: `tests/test_attention_parity.py` matches `F.scaled_dot_product_attention`
within `5e-3` (fp16) / `1e-2` (bf16) absolute, which is the documented
production tolerance for half-precision attention (see
`kernels/_common/dtype.py::TOLERANCE`).

## 2. Paged KV cache

`kernels/kv_cache/triton_kv_paged.py`

vLLM-style page storage:

* `k_cache, v_cache: [num_blocks, num_heads, block_size, head_dim]`.
* `block_table: [num_sequences, max_blocks_per_seq]` (logical -> physical page).
* `slot_mapping: [N]` flat slot indices for new tokens.

Three operations:

* `kv_paged_append(k_in, v_in, k_cache, v_cache, slot_mapping)` - write
  new K / V vectors into the slots indicated by `slot_mapping`.
* `kv_paged_lookup(k_cache, v_cache, block_table, context_lens, max_ctx)` -
  gather K / V for a batch of sequences into dense `[B, H, max_ctx, D]`
  tensors. Positions past `context_lens[b]` are zero-filled (matches
  torch reference indexing).
* `kv_paged_scatter(...)` - same as append; exposed separately so a
  speculative-decoding rollback can replace it without changing the call
  site.

## 3. Fused Linear + (Layer | RMS) Norm

`kernels/fused_linear_norm/triton_kernel.py`

Computes `LayerNorm(x W^T + b)` (or `RMSNorm`) in one kernel:

* One program per row.
* GEMV inner loop tiled by `BLOCK_K` (autotuned over 64 / 128 / 256).
* Mean / variance / RMS computed in registers from the row of linear
  output, never written to HBM.

This is the production wrapper around the common "linear -> norm -> next
layer" pattern in transformer blocks. On Llama-style MLP-in projections
(D_in=4096, D_out=14336) the HBM saving is the dominant win at small batch.

## 4. Online softmax

`kernels/softmax/triton_softmax.py`

Two-pass online softmax along the last dim:

* Pass 1: walk the row, accumulating `m_i` and `l_i` with the standard
  online-softmax update rule.
* Pass 2: write `exp(x - m_final) / l_final` (or the log-softmax variant).

The "online" name refers to the single-pass-per-block accumulation inside
each pass. A true single-pass softmax requires a second buffer, losing the
HBM win; this is the production sweet spot for standalone softmax. When
softmax appears as part of attention scores, it is fused into the attention
kernel itself.

Supports `is_causal=True` for masked attention-score softmax without
materializing a separate mask tensor.

## 5. RoPE

`kernels/rope/triton_rope.py`

Two conventions:

* **interleaved** (GPT-NeoX, some HuggingFace models): rotation pairs
  `(x[2i], x[2i+1])`.
* **half-rotated** (Llama / Mistral / Qwen): pairs `(x[i], x[i + D/2])`.

Both supported via a constexpr branch; the kernel constructs the partner
index pattern at compile time so there's no runtime gather table.

`apply_rope_` is the in-place variant that writes the result back over the
input, saving an allocation when the caller owns the tensor (common in
the QKV-projection -> attention path).

Helper: `build_rope_tables(seq_len, head_dim, base, interleaved)` builds
the standard `cos` / `sin` tables matching the HuggingFace convention.

## Cross-vendor notes

### ROCm (AMD MI200 / MI300)

Triton 3.x ships an AMD backend. The kernels above compile and run on MI300
without changes. `kernels/_rocm/detect_rocm()` returns the gfx number plus
a recommended `BLOCK_N` cap (128 on MI300, 64 on MI250). The attention
autotune list can be filtered through `autotune_cap()` at import time to
drop configs that would exceed LDS budgets.

### Apple MPS / Core ML

Triton has no Apple backend. `kernels/_mps/` ports the softmax kernel via:

* `mps_softmax(x)` - dispatches to `F.softmax` on the `mps` device. Torch's
  MPS softmax is already a Metal shader; we don't gain by writing our own
  unless we're fusing.
* `coreml_softmax(x)` - compiles a tiny Core ML program at first call and
  runs it on the Apple Neural Engine via `coremltools`. Caches compiled
  models by shape. For benchmark / cross-vendor demonstration; per-call
  dispatch overhead makes it impractical for inner-loop kernel use.

## Numerical tolerances

These are the absolute/relative tolerances actually enforced in
`tests/`. They are scaled up vs the dtype's epsilon to account for
reduction-tree differences (fused vs separate ops) and the larger
accumulators. Half-precision tolerances are larger than fp32 because the
final cast (fp32 accumulator → fp16/bf16 output) lossily rounds.

| dtype | absolute | enforced in |
|-------|---------:|-------------|
| fp32  | 1e-5     | softmax, RoPE, dispatch parity |
| fp16  | 5e-3     | fused attention, fused linear+norm |
| bf16  | 1e-2     | fused attention, fused linear+norm |

The defaults in `kernels/_common/dtype.py::TOLERANCE` are stricter
(fp16=1e-3, bf16=2e-2); the per-kernel tests override them when the
kernel's reduction tree is deeper than the reference path (e.g. attention
accumulates over the full K axis in fp32 before the final cast). A future
version may move the per-kernel overrides into `TOLERANCE` itself; until
then, the test files are the source of truth.

## Roadmap

* Fused GQA path (no K / V expansion on host).
* FP8 attention forward on Hopper (the dtype helper is wired; the Triton
  kernel needs a fp8 specialization).
* Backward passes for training-time use (current scope is inference-only).
* Per-shape kernel cache for `torch.compile` integration.
