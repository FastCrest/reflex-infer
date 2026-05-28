# Research Notes

## Problem Framing

Most modern LLM inference kernels are designed and benchmarked around datacenter
GPUs. Jetson devices are different:

- integrated GPU and CPU share system DRAM;
- power mode and clock limits are central to reproducibility;
- thermal throttling can invalidate long benchmark runs;
- small-batch latency matters more than high-throughput serving;
- memory capacity may be large relative to power budget, but bandwidth and SM
  count are constrained compared with datacenter GPUs;
- CPU preprocessing, token sampling, and kernel launch overhead can matter for
  short prompts and local agent workflows.

`reflex-infer` should be framed as Jetson-aware inference kernels, not as a
general FlashInfer replacement.

## Potential Contribution Shape

The paper needs one clear contribution. Good candidates:

1. Jetson-aware attention decode kernel for small batch and GQA/MQA.
2. KV-cache layout optimized for shared-memory edge devices.
3. Fused decode-side pipeline: RoPE, QK, softmax, value accumulation, and KV
   append where shape allows.
4. Tokens-per-joule optimization methodology for Jetson LLM inference.
5. A reproducible benchmark harness for Jetson kernel research.

The strongest paper is likely a combination of a kernel plus a measurement
methodology. A methodology-only paper may be easier to finish but weaker unless
the profiling dataset is very good.

## FlashInfer Relationship

FlashInfer is the closest comparison point. The research should not claim that
FlashInfer is unsuitable for Jetson without measurements. A fair framing:

- FlashInfer is broad and production-oriented.
- `reflex-infer` explores a narrower Jetson-specific point in the design space.
- Compare only supported operator shapes and hardware.
- If FlashInfer cannot run a given shape on a target Jetson, report that as a
  compatibility boundary, not a performance win.

## Jetson-Specific Constraints To Measure

- `nvpmodel` mode and whether `jetson_clocks` is active.
- GPU frequency, EMC/memory controller frequency, and thermal state.
- Power draw and tokens per joule.
- CUDA version, JetPack version, driver version.
- Whether buffers use device memory, pinned memory, registered host memory, or
  unified memory.
- CPU-side tokenization and sampling overhead.
- Kernel launch count per generated token.
- KV-cache memory footprint and bandwidth.

## Early Research Questions

- Does a persistent or CUDA Graph-based decode loop reduce per-token latency on
  Jetson enough to matter?
- Is decode attention memory-bandwidth bound on Orin for typical local LLM
  shapes?
- Can a Jetson-oriented KV-cache layout reduce cache misses or improve memory
  coalescing for GQA/MQA?
- Is int4/int8 dequantization plus attention compute faster than fp16/bf16
  paths on Orin-class Tensor Cores for small batch?
- Does fusing RoPE and KV append into attention improve latency, or does it hurt
  maintainability without enough benefit?
- Which bottleneck dominates end-to-end latency: CUDA kernels, CPU scheduling,
  memory allocation, sampling, tokenizer, or thermal/power limits?

## Non-Goals For The First Paper

- Training kernels.
- Multi-node serving.
- General datacenter GPU optimization.
- Claims across all Jetson devices.
- Full LLM runtime from scratch.
- Replacing TensorRT-LLM or FlashInfer broadly.

