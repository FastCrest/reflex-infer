# Benchmark Plan

## Measurement Principles

- Record exact hardware, JetPack, CUDA, driver, PyTorch, and compiler versions.
- Fix the Jetson power mode before every run.
- Record whether `jetson_clocks` is enabled.
- Log GPU frequency, EMC frequency, temperature, and power during each run.
- Warm up before measuring.
- Use CUDA events for kernel-level timing.
- Use Nsight Systems for end-to-end timeline analysis.
- Use Nsight Compute for selected kernel variants only; profiler replay can
  distort end-to-end timing.

## Hardware Matrix

Initial target:

- Jetson Orin Nano 8GB.

Secondary targets:

- Jetson AGX Orin 64GB or 32GB.
- Jetson Orin NX 16GB.
- Jetson Thor only if available later; do not mix Thor results into an
  Orin-focused claim without clearly separating architectures.

## Software Matrix

Record:

- JetPack version.
- Jetson Linux version.
- CUDA version.
- Python version.
- PyTorch version.
- TensorRT / TensorRT-LLM version if used.
- FlashInfer version or commit SHA if used.
- Compiler and flags for custom kernels.

## Baselines

Candidate baselines:

- `llama.cpp` CUDA path for end-to-end local inference comparison.
- `tensorrt-edge-llm` for Jetson-oriented TensorRT comparison.
- A simple standalone CUDA reference baseline for microbenchmarks.
- PyTorch attention or SDPA path, if available on target, as a correctness and
  sanity reference.
- FlashInfer, only if the target shape and Jetson software stack are supported.

## Workloads

Start small and representative:

- Model: Qwen 3.5B-class model, exact model ID/config TBD.
- Precision: GGUF `Q4_K_M` first.
- Batch sizes: 1, 2, 4.
- Prompt lengths: 128, 512, 2048, 4096.
- Generated tokens: 64, 128, 256.
- Head count, KV-head count, hidden size, intermediate size, and head dimension:
  derive from the exact Qwen config before writing kernels.
- Compare short interactive prompts and long-context decode separately.

## Metrics

Kernel-level:

- latency in microseconds;
- effective memory bandwidth;
- achieved occupancy;
- SM throughput;
- DRAM throughput;
- L2 hit rate where available;
- registers per thread;
- shared memory per block;
- launch configuration.

End-to-end:

- prefill latency;
- decode p50/p95/p99 per-token latency;
- tokens per second;
- wall power;
- tokens per joule;
- peak memory;
- thermal throttling events.

## Experiment Sequence

1. Establish a working baseline decode loop.
2. Run `llama.cpp` Q4 baseline on Orin Nano 8GB.
3. Run `tensorrt-edge-llm` Q4 or closest supported baseline.
4. Profile end-to-end with Nsight Systems.
5. Identify top kernel and non-kernel latency components.
6. Build a standalone CUDA microbenchmark for the target Qwen Q4 operator.
7. Implement first optimized kernel variant.
8. Validate numerical correctness against a high-precision reference.
9. Profile with Nsight Compute.
10. Add ablations one at a time.
11. Integrate into a standalone decode loop.
12. Re-run latency, power, and thermal experiments.

## Reproducibility Checklist

- Save raw logs.
- Save profiler reports.
- Save benchmark command lines.
- Save git commit SHA.
- Save model config.
- Save input prompt set or synthetic shape generator seed.
- Save power mode and clock settings.
- Save ambient temperature if possible.
