# Paper Plan

Working title:

`Reflex-Infer: Jetson-Aware GPU Kernels for Low-Latency Edge Inference`

## One-Sentence Thesis

Edge inference on NVIDIA Jetson has a different memory, power, and latency cost
model from datacenter GPU serving, and kernels specialized for that cost model
can reduce per-token latency and improve energy efficiency for small-batch
inference.

## Locked Initial Scope

- Target device: Jetson Orin Nano 8GB.
- Prototype style: standalone CUDA, before runtime integration.
- Target model: Qwen 3.5B-class model, exact Hugging Face/GGUF/model config TBD.
- Target precision: GGUF `Q4_K_M`.
- Baselines: `llama.cpp` and `tensorrt-edge-llm`.
- First benchmark goal: isolate the decode-side Q4 bottleneck for the selected
  Qwen shape before claiming end-to-end speedup.

## Abstract Draft

Large language model inference is increasingly moving to edge devices where
latency, privacy, and offline operation matter. Existing high-performance GPU
inference kernels are usually optimized and evaluated on datacenter GPUs, while
NVIDIA Jetson-class devices use integrated GPUs, shared SoC memory, constrained
power budgets, and deployment-specific thermal limits. This paper introduces
`Reflex-Infer`, a Jetson-aware kernel research prototype for low-latency
small-batch inference. We identify Jetson-specific bottlenecks in decode-time
attention and KV-cache movement, design kernels and memory layouts around those
constraints, and evaluate latency, throughput, and tokens-per-joule against
standard inference baselines. The goal is not to replace broad inference
libraries, but to show that edge-specific kernel specialization can improve the
latency-energy frontier on Jetson hardware.

## Claims That Need Evidence

- Claim 1: Jetson inference bottlenecks differ from datacenter GPU bottlenecks
  for small-batch local inference.
- Claim 2: A selected `reflex-infer` kernel reduces median and p95 token latency
  on at least one Jetson target.
- Claim 3: The improvement is not only a microbenchmark artifact; it appears in
  an end-to-end decode loop.
- Claim 4: The optimized path improves or preserves tokens-per-joule.
- Claim 5: The design is reproducible across power modes and stable across
  thermal conditions.

Do not include any claim that cannot be backed by benchmark data.

## Proposed Paper Structure

1. Introduction
   - Edge inference motivation.
   - Why Jetson is attractive and different.
   - Summary of contribution.

2. Background
   - Transformer inference phases: prefill, decode, append.
   - KV-cache and GQA/MQA.
   - Jetson memory architecture and power constraints.
   - Related systems: FlashInfer, TensorRT-LLM, llama.cpp CUDA, vLLM/SGLang
     where relevant.

3. Bottleneck Study
   - Baseline model and workload.
   - Nsight Systems timeline.
   - Nsight Compute kernel metrics.
   - Memory and power observations.

4. Reflex-Infer Design
   - Target workload.
   - Kernel design.
   - KV-cache layout.
   - Launch/capture strategy, possibly CUDA Graphs.
   - Precision and quantization assumptions.

5. Implementation
   - CUDA/C++ extension or runtime integration.
   - Python benchmark harness.
   - Shape specialization strategy.
   - Build and deployment notes for Jetson.

6. Evaluation
   - Hardware setup.
   - Jetson Orin Nano 8GB power mode, clocks, thermal logging.
   - Software setup.
   - Baselines: `llama.cpp`, `tensorrt-edge-llm`, and standalone CUDA reference.
   - Microbenchmarks.
   - End-to-end token latency.
   - Energy and thermal stability.
   - Ablations.

7. Limitations
   - Hardware coverage.
   - Model shape coverage.
   - Maintenance cost.
   - Compatibility with existing inference frameworks.

8. Conclusion
   - What the evidence supports.
   - What remains future work.

## Evaluation Tables To Produce

- Hardware/software matrix.
- Kernel latency by shape.
- End-to-end token latency by prompt length and output length.
- Tokens per second and tokens per joule.
- p50/p95/p99 latency.
- Power mode sensitivity.
- Thermal throttling observations.
- Ablation table for each optimization.

## Minimum Viable Paper

The smallest credible version needs:

- one Jetson target: Orin Nano 8GB;
- one model family: Qwen 3.5B-class model;
- one optimized decode-side kernel;
- at least two baselines: `llama.cpp` and `tensorrt-edge-llm`;
- reproducible measurement scripts;
- kernel-level and end-to-end results;
- a clear limitation section.
