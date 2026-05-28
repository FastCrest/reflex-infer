# reflex-infer Library Architecture

## Goal

Build a Jetson-only CUDA kernel library for LLM inference. The closest external
analogy is FlashInfer, but the scope is narrower:

- NVIDIA Jetson first.
- Edge memory and power constraints first.
- Small-batch local inference first.
- Explicit support for Orin Nano 8GB before larger Jetson devices.

`reflex-infer` should be used by `reflex-llm` as an external dependency once the
kernel API stabilizes.

The current repository exports the first build-level API:

- CMake target: `reflex::infer`
- Public header: `include/reflex/infer.h`
- Package config: `reflex-inferConfig.cmake`
- Q4 dispatcher source: `src/dispatch.cpp`
- CUDA sources: `src/kernels/q4_gemm.cu` and `src/kernels/attention.cu`

This target builds CUDA kernels when a CUDA compiler is available. On hosts
without CUDA it builds stubs so API consumers and packaging checks can still
compile.

## Initial Target

- Device: Jetson Orin Nano 8GB.
- Model family: Qwen 3.5B or Qwen3 4B class.
- Quantization: GGUF `Q4_K_M` first.
- Runtime consumer: `reflex-llm`.
- Baselines: `llama.cpp` and `tensorrt-edge-llm`.

## Future Targets

Hardware:

- Jetson Orin NX 16GB.
- Jetson AGX Orin 64GB.
- Jetson Thor.

Models:

- Qwen family.
- Phi-4 family.
- Other compact edge models after the abstraction is stable.

Quantization roadmap:

- GGUF `Q4_K_M` first.
- GGUF `Q4_0` later if needed.
- AWQ4 / GPTQ4 if the runtime integration needs them.
- FP16 reference kernels for validation.

## API Principles

- Keep the kernel API independent from `reflex-llm`.
- Do not parse GGUF inside `reflex-infer`.
- Do not own tokenizer, sampler, HTTP, persistence, or prompt logic.
- Accept explicit tensor pointers, strides, shapes, quantization descriptors,
  workspace, CUDA stream, and hardware profile.
- Return clear unsupported-status codes instead of silently falling back.

## Core Abstractions

`HardwareProfile`

- device class;
- CUDA SM version;
- SM count;
- shared memory limits;
- memory bandwidth estimate;
- total usable memory;
- power profile label.

`ModelShape`

- family;
- hidden size;
- intermediate size;
- layer count;
- attention heads;
- KV heads;
- head dimension;
- max context;
- RoPE settings.

`QuantDescriptor`

- format;
- block size;
- group size;
- scale type;
- zero-point type;
- packing order;
- dequant policy.

`KernelWorkspace`

- temporary storage pointer;
- size in bytes;
- ownership stays with caller;
- stream-safety documented per operator.

## Operator Roadmap

Phase 1:

- Q4 dequant + GEMV.
- Q4_K MMQ prefill microbenchmarks.
- Decode attention microbenchmarks.

Phase 2:

- Runtime-facing C++ wrappers.
- Capability discovery.
- Fallback-compatible status codes.
- Integration option in `reflex-llm`.
- Q4 dispatcher API for GGUF K-quant GEMV/GEMM calls.
- Physical extraction of Q4 GEMV/GEMM and attention kernels from `reflex-llm`.

Phase 3:

- Orin NX and AGX Orin profile tuning.
- Phi-4 model-shape validation.
- More quantization layouts.

Phase 4:

- Thor-specific kernels and dispatch path.

## Integration Contract With reflex-llm

`reflex-llm` passes:

- hardware profile from Jetson probe;
- model shape from GGUF metadata;
- quantization descriptors from loaded tensor metadata;
- input/output tensor pointers;
- runtime-owned quantized weight pointers and CUDA-visible aliases for
  mmap-backed weights;
- KV-cache pointers and layout metadata;
- workspace pointer and size;
- CUDA stream.

`reflex-infer` returns:

- success or unsupported status;
- optional workspace requirement;
- optional profiling labels for benchmark attribution.

`reflex-infer` must not allocate large runtime buffers internally. Jetson memory
budgeting belongs to `reflex-llm`.
