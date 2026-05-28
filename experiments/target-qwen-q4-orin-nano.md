# Target Spec: Qwen Q4 On Jetson Orin Nano 8GB

Status: initial target definition.

## Fixed Decisions

- Device: NVIDIA Jetson Orin Nano 8GB.
- First implementation path: standalone CUDA.
- Runtime consumer: `reflex-llm`.
- First model family: Qwen 3.5B-class model.
- First precision: GGUF `Q4_K_M`.
- Baseline 1: `llama.cpp`.
- Baseline 2: `tensorrt-edge-llm`.

## Decisions Still Needed Before Kernel Work

### Exact Model ID

The project currently says "Qwen 3.5B-class" because the exact model ID is not
locked. The kernel shape must be derived from the selected model config.

Record:

- Hugging Face repo or GGUF file source.
- Parameter count.
- Hidden size.
- Layer count.
- Attention head count.
- KV head count.
- Head dimension.
- Vocabulary size.
- RoPE settings.
- Context length.

### Q4 Format

The first target format is GGUF `Q4_K_M`, matching the current llama.cpp-style
baseline path. Other 4-bit formats stay out of scope until the first Qwen
`Q4_K_M` kernel path is measured.

Format details to confirm from the exact model file and loader:

- packing layout for GGUF `Q4_K_M`;
- block/group size;
- scale/min storage;
- whether activations remain fp16;
- whether KV-cache remains fp16, int8, or int4;
- dequantization location: separate kernel, fused GEMV/GEMM, or fused attention.

## First Benchmark Shape

Start with batch 1 because Orin Nano is most likely to be used for interactive
single-user local inference.

Initial benchmark matrix:

- Batch: 1.
- Prompt length: 128, 512, 2048.
- Decode length: 64, 128.
- Context after prefill: 128, 512, 2048, 4096 if memory allows.
- Threads/blocks: determined by first kernel prototype.

Expand to batch 2 and 4 only after batch 1 is stable.

## Baseline Procedure

### llama.cpp

Record:

- llama.cpp commit SHA.
- build flags.
- CUDA architecture flags.
- exact GGUF model file and quantization.
- command line.
- prompt file.
- `nvidia-smi` is not available on Jetson, so use `tegrastats`/Jetson tooling.
- prefill tokens/sec and decode tokens/sec.
- p50/p95 decode token latency if available or measured externally.

### tensorrt-edge-llm

Record:

- repo/package source.
- commit SHA or release version.
- TensorRT version.
- model conversion command.
- quantization mode.
- engine build settings.
- runtime command.
- whether the model and Q4 path are exactly comparable to llama.cpp.

### reflex-llm

Record:

- `reflex-llm` commit SHA.
- whether kernels are legacy in-repo or external `reflex-infer`;
- exact `reflex-infer` commit SHA when external kernels are enabled;
- model file and quantization format;
- command line and runtime flags.

## Standalone CUDA Prototype

The first prototype should not depend on a complete LLM runtime. It should:

- load or synthesize tensors with the exact Qwen shape;
- implement one target operator;
- compare output against a reference implementation;
- measure CUDA-event latency;
- optionally dump tensors for cross-checking with Python.

Candidate first operators:

1. Q4 dequant + GEMV for one projection.
2. Decode attention over existing KV-cache.
3. KV append.
4. Fused RoPE + Q/K handling.

Recommended first operator:

Q4 dequant + GEMV, unless profiling shows decode attention is clearly the
dominant latency component in the selected baseline.

Reason: for a Q4 model on a small Jetson GPU, dequantized matrix-vector paths
may dominate token latency before attention does. The paper should let profiling
choose the first optimized kernel.

## Success Criteria

Minimum useful result:

- correct output within defined tolerance;
- standalone kernel faster than simple CUDA reference;
- measurable impact on an end-to-end or near-end-to-end decode loop;
- no regression in tokens-per-joule.

Paper-worthy result:

- p50 and p95 decode latency improvement over at least one baseline;
- clear explanation from Nsight Compute metrics;
- reproducible power/thermal logs;
- ablation showing which optimization mattered.
