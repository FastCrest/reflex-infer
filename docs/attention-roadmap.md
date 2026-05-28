# Attention Roadmap

## Current State

`reflex-infer` currently has a generic contiguous-KV attention path:

- `flash_attention_decode`
- `flash_attention_prefill_batched`

This is the equivalent of a normal decode/prefill attention implementation:
compute `Q * K^T`, softmax, then multiply by `V`. It is the right correctness
baseline and the right first integration target for `reflex-llm`.

It is not an XQA-style serving decode engine.

## Architecture Split

The attention stack should stay split into three tiers:

1. Contiguous KV decode/prefill.
2. Paged KV-cache decode.
3. Jetson XQA-lite decode.

The first tier is already extracted from `reflex-llm`. The second and third
tiers are future work.

## Paged Decode

Paged decode is the generic serving path for variable-length requests and
page-table-managed KV cache. It should prioritize compatibility:

- MHA and GQA.
- Variable sequence lengths.
- Page-table metadata from the runtime.
- Continuous batching.
- FP16 KV first, then INT8 or 4-bit KV.

For `reflex-infer`, this should be a separate API from contiguous decode. It
needs explicit page metadata instead of assuming `[seq_len, n_kv_heads,
head_dim]` contiguous KV layout.

## XQA-Lite Decode

XQA-lite should be treated as a memory-system-centric decode engine, not a
drop-in rename of flash attention.

Initial Jetson scope:

- Orin Nano 8GB first.
- Batch size 1 to small continuous batches.
- GQA-heavy Qwen shapes first.
- FP16 KV baseline.
- INT8 KV next.
- 4-bit KV only after the baseline proves correctness and power benefit.

Primary goal:

- Reduce KV-cache bytes moved per generated token.

Secondary goals:

- Reduce launch overhead.
- Avoid unnecessary page traversal overhead.
- Keep register/shared-memory pressure low enough for 8 SMs.
- Preserve deterministic fallback to the generic attention path.

Out of scope for the first Jetson XQA-lite pass:

- Hopper/Blackwell-specific assumptions.
- Speculative decoding.
- MLA.
- FP8 or NVFP4 formats.
- Multi-GPU scheduling.

## API Direction

Capability discovery must expose these separately:

- `decode_attention`: contiguous generic decode path.
- `paged_decode_attention`: page-table-backed generic decode path.
- `xqa_lite_decode`: Jetson-specialized decode path.

Do not route runtime calls to XQA-lite implicitly. `reflex-llm` should choose
the path after checking model shape, KV format, sequence length, page layout,
and device profile.

## Validation Rule

Every XQA-lite fast path must compare against:

- contiguous FP16 decode for small synthetic cases;
- paged decode for page-table cases;
- end-to-end `reflex-llm` text parity on the target Qwen model;
- power and thermal logs on the exact Jetson power mode.

No throughput or tokens-per-joule claim is valid until measured on Jetson
hardware.

## External Reference

FlashInfer documents batch decode with paged KV cache separately from its XQA
API. This roadmap uses that distinction as an architectural reference, but the
implementation target here is Jetson Orin rather than Hopper/Blackwell serving.

- https://docs.flashinfer.ai/api/attention.html
- https://docs.flashinfer.ai/generated/flashinfer.xqa.xqa.html
