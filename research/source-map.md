# Source Map

Checked: 2026-05-28

These are primary or near-primary sources to anchor the research. Prefer these
over blog summaries or forum posts when writing the paper.

## NVIDIA Jetson And CUDA

### CUDA for Tegra Application Note

URL: https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html

Use for Jetson/Tegra-specific memory behavior. Key points:

- Tegra iGPU, CPU, device memory, host memory, and unified memory share physical
  SoC DRAM.
- Memory types have different CPU/iGPU caching behavior.
- Duplicate allocations and explicit transfers can sometimes be avoided, but
  the best choice depends on access pattern.
- I/O coherency starts with Xavier-class SoCs; full system memory coherency is
  described for Thor and later.
- `cudaMemGetInfo()` has caveats on integrated GPUs because GPU memory is drawn
  from shared system memory.

### NVIDIA Jetson Modules

URL: https://developer.nvidia.com/embedded/jetson-modules

Use for product positioning and hardware families. Useful facts:

- Jetson is positioned as compact, power-efficient edge AI hardware.
- Jetson AGX Orin is advertised up to 275 TOPS.
- Jetson Orin NX is advertised up to 157 TOPS.
- Jetson Orin Nano series is advertised up to 67 TOPS.
- Jetson Thor introduces a newer high-end Jetson family, so the paper should
  distinguish Orin-focused results from Thor-forward design ideas.

### CUDA Programming Guide

URL: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

Use for CUDA programming model, execution model, memory model, CUDA Graphs,
streams, asynchronous execution, unified memory, and kernel-level tuning
terminology.

### Nsight Systems

URL: https://docs.nvidia.com/cuda/nsight-systems/index.html

Use for end-to-end timeline profiling. Nsight Systems is appropriate for seeing
launch overhead, CPU/GPU overlap, memory copies, runtime gaps, and system-level
bottlenecks.

### Nsight Compute

URL: https://docs.nvidia.com/cuda/nsight-compute/index.html

Use for per-kernel profiling. Nsight Compute is appropriate for SM utilization,
memory throughput, occupancy, warp stalls, shared memory behavior, and comparing
kernel variants.

## FlashInfer

### FlashInfer GitHub Repository

URL: https://github.com/flashinfer-ai/flashinfer

Use as an implementation reference and baseline candidate. Current repo
positioning:

- High-performance GPU kernels for inference.
- Supports attention, GEMM, MoE, sampling, and other operators.
- Attention includes paged/ragged KV-cache, decode, prefill, append, MLA,
  cascade attention, sparse attention, and mixed batching.
- GPU support starts at SM75 and includes newer architectures; support details
  vary by feature.

### FlashInfer Paper

URL: https://arxiv.org/abs/2501.01005

Use for academic positioning. The paper frames FlashInfer as an efficient and
customizable attention engine for LLM inference serving with kernel-level and
end-to-end evaluations.

### FlashInfer Introduction Blog

URL: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html

Use for design intuition around prefill, decode, append attention, GQA, and
paged KV-cache.

### NVIDIA Technical Blog On FlashInfer

URL: https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/

Use for NVIDIA-facing framing of FlashInfer operator families and integration
model. Treat as contextual source rather than a substitute for measured results.

