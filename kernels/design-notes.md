# Kernel Design Notes

## Candidate Kernel Targets

Initial constraints:

- Device: Jetson Orin Nano 8GB.
- Runtime style: standalone CUDA first.
- Model: Qwen 3.5B-class model, exact config TBD.
- Precision: GGUF `Q4_K_M` first.
- Baselines: `llama.cpp` and `tensorrt-edge-llm`.

### Decode Attention

Most likely first target. Decode attention is repeatedly executed one token at a
time and can become memory-bandwidth dominated due to KV-cache reads. Jetson's
shared DRAM and power limits make this a good target for edge-specific tuning.

Questions:

- What KV-cache layout gives the best coalescing for batch 1-4?
- Does GQA/MQA change the optimal mapping of warps to heads?
- Can the kernel avoid extra intermediate writes?
- Is Tensor Core use practical for the target shape, or is memory movement the
  limiting factor?

### KV Append

Appending new K/V vectors can be small but frequent. It may be worth fusing with
RoPE or decode attention if launch overhead dominates.

Questions:

- Does fusion reduce launch overhead enough to offset a more complex kernel?
- Can the append path maintain compatibility with paged KV-cache layouts?

### RoPE

RoPE is simple, frequent, and often memory-bound. It may be useful as a fused
component rather than a standalone contribution.

Questions:

- Should RoPE be fused into Q/K projection output handling?
- Does precomputing sin/cos tables improve Jetson memory behavior or just move
  the bandwidth cost?

### Sampling

Sampling can matter in low-latency local inference, especially when the GPU is
underused during CPU-side sampling.

Questions:

- Is top-k/top-p sampling measurable in the end-to-end critical path?
- Is GPU sampling worth the launch overhead at batch 1?

### Quantized GEMV / GEMM

Quantized projection and MLP paths may dominate before attention for small
models. This is broader and harder than decode attention but may be necessary
for end-to-end gains.

Questions:

- Is int4/int8 dequantization memory-bound or compute-bound on Orin?
- Can per-group scales be loaded efficiently?
- Which shapes match deployed edge models?

## Design Principles

- Optimize for p95 token latency, not only average throughput.
- Keep the first kernel shape-specialized; generality can come later.
- Minimize allocations and per-token launches.
- Prefer static workspace and reusable buffers.
- Validate correctness before profiling.
- Avoid overly broad APIs until the winning kernel shape is known.

## Correctness Plan

- Compare output against PyTorch reference at fp32/fp16 tolerances.
- Test multiple sequence lengths, head counts, head dimensions, and batch sizes.
- Include edge cases: empty/short context, non-power-of-two sequence length,
  paged cache boundary, and max configured context.
- Track max absolute error, relative error, and downstream token differences.

## First Prototype Choice

Recommended first prototype:

`standalone_qwen_q4_decode_microbenchmark`

Why:

- It matches the chosen Qwen/Q4/Orin Nano target.
- It can be validated without committing to a full runtime integration.
- It can be compared directly against isolated timings from `llama.cpp` and
  `tensorrt-edge-llm`.
- It forces the project to define the real Q4 packing format before optimizing.
