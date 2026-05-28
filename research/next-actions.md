# Next Actions

## Phase 1: Define Scope

- Use Jetson Orin Nano 8GB as the first target.
- Use standalone CUDA as the first prototype path.
- Use `llama.cpp` and `tensorrt-edge-llm` as the first two baselines.
- Target a Qwen 3.5B-class model first.
- Target GGUF `Q4_K_M` precision first.
- Pick the exact model ID before writing kernels.

## Phase 2: Baseline

- Build a reference PyTorch implementation for target shapes.
- Run a simple end-to-end decode loop.
- Capture Nsight Systems timeline.
- Identify kernel launch count per generated token.
- Record power, clocks, and thermals.

## Phase 3: First Kernel

- Implement a simple CUDA decode-attention baseline.
- Validate against PyTorch output.
- Add CUDA-event microbenchmarks.
- Profile with Nsight Compute.
- Iterate on memory layout, warp mapping, and vectorized loads.

## Phase 4: Paper Evidence

- Compare against at least two baselines.
- Add ablation experiments.
- Run p50/p95/p99 latency measurements.
- Run tokens-per-joule measurements.
- Write limitations before writing the abstract claim strongly.

## Immediate Open Decisions

- Exact Qwen model ID: Hugging Face repo or GGUF source.
- Exact target operator: Q4 GEMV/dequant path, decode attention, KV append, or
  fused decode path.
- Exact `tensorrt-edge-llm` install path and supported quantization mode on
  Orin Nano 8GB.
