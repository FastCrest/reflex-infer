# Reflex-Infer

Research workspace and future library for Jetson-optimized GPU kernels for
low-latency inference.

Working name: `reflex-infer`

Status: early research and planning. No performance claims are established yet.

Long-term role: `reflex-infer` should become the reusable CUDA kernel library
used by `reflex-llm`, similar in spirit to FlashInfer but scoped to NVIDIA
Jetson devices and edge inference constraints.

Current build status: `reflex-infer` exports a `reflex::infer` CMake target
with capability-discovery types, Q4 GEMV/GEMM kernels, embedding-row dequant,
and decode/prefill attention kernels. CUDA kernels are built when a CUDA
compiler is available; otherwise the package builds host stubs for API
consumers.

## Core Idea

`reflex-infer` investigates whether inference kernels can be specialized for
NVIDIA Jetson-class edge devices rather than tuned primarily for datacenter
GPUs. The likely focus is small-batch, latency-sensitive LLM or transformer
inference where memory movement, KV-cache layout, launch overhead, and power
limits dominate.

## Initial Target

- Hardware: NVIDIA Jetson Orin Nano 8GB.
- Implementation mode: standalone CUDA first.
- First model target: Qwen 3.5B-class model, exact model ID/config TBD.
- First precision target: GGUF `Q4_K_M`.
- Baselines: `llama.cpp` and `tensorrt-edge-llm`.
- First operator focus: Q4 decode path for the selected Qwen model, starting
  with the smallest standalone CUDA microbenchmark that matches the model's
  actual projection/KV-cache shapes.

## Initial Research Question

Can Jetson-aware CUDA kernels reduce end-to-end token latency and improve
tokens-per-joule for edge inference compared with general-purpose inference
kernels?

## Working Hypothesis

Jetson devices have integrated GPUs sharing SoC DRAM with the CPU. That changes
the cost model compared with discrete GPUs: memory allocation choices, CPU/GPU
buffer sharing, power modes, and thermal stability matter more. A kernel library
that explicitly optimizes around these constraints may outperform generic
server-oriented kernels for constrained edge workloads.

## Project Layout

- `research/source-map.md`: primary sources and what each source is useful for.
- `research/research-notes.md`: early technical observations and open questions.
- `paper/paper-plan.md`: paper outline, claims to prove, and contribution shape.
- `experiments/benchmark-plan.md`: benchmark matrix and measurement protocol.
- `kernels/design-notes.md`: candidate kernel directions.
- `docs/library-architecture.md`: planned `reflex-infer` library API and
  hardware/model abstraction.
- `docs/attention-roadmap.md`: contiguous decode, paged decode, and Jetson
  XQA-lite attention roadmap.
- `include/reflex/infer.h`: first public C++ API surface consumed by
  `reflex-llm`.
- `src/dispatch.cpp`: public dispatcher implementation.
- `src/kernels/`: Jetson CUDA kernels and non-CUDA stubs.
- `benchmarks/`: future benchmark scripts and collected results.
- `assets/`: future diagrams, plots, and paper figures.

## Build Interface

Use as a sibling checkout from `reflex-llm`:

```bash
cmake -B build -DREFLEX_LLM_USE_REFLEX_INFER=ON
```

Or install the package directly:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/reflex-infer
```

Installed consumers can use:

```cmake
find_package(reflex-infer CONFIG REQUIRED)
target_link_libraries(app PRIVATE reflex::infer)
```

## Guardrails

- Do not claim speedup until measured on actual Jetson hardware.
- Separate kernel microbenchmarks from end-to-end inference results.
- Report power mode, clocks, thermals, JetPack/CUDA versions, model shape, batch
  size, context length, precision, and quantization settings with every result.
- Compare against strong baselines: FlashInfer where supported, TensorRT-LLM,
  `tensorrt-edge-llm`, `llama.cpp` CUDA, PyTorch/eager or compiled baselines,
  and any Jetson-specific NVIDIA sample where relevant.

## Python kernel library

In addition to the standalone CUDA targets above, `reflex-infer` ships an
early cross-vendor Python kernel library used by Reflex Cloud's
deterministic-mode runtime. Honest status: this is a v1 kernel set with
on-going optimization work. Some kernels currently match or trail torch
baselines on specific shape ranges (`torch.nn.functional` routes to
FlashAttention-2 on Ampere+, which is a strong baseline). See
[KERNELS.md](KERNELS.md) for per-kernel notes on where each path is
expected to win, match, or trail torch / SDPA. We add benchmark JSON for
each shape sweep we run; if a kernel does not beat the torch baseline we
document that rather than hide it.

Public surface:

```python
from kernels.attention import fused_attention
from kernels.kv_cache import kv_paged_append, kv_paged_lookup, kv_paged_scatter
from kernels.fused_linear_norm import fused_linear_layernorm, fused_linear_rmsnorm
from kernels.softmax import online_softmax
from kernels.rope import apply_rope, apply_rope_, build_rope_tables
```

Every kernel has a Triton primary path (NVIDIA + AMD via Triton's hip
backend) and a CUDA C++ reference loaded lazily via
`torch.utils.cpp_extension`. Apple is covered for the softmax kernel via
MPS and Core ML (see `kernels/_mps/`). Parity tests are in `tests/`, with
the GPU tests gated behind `@pytest.mark.gpu`.

Run the benchmarks:

```bash
python -m benchmark.run_all --warmup 5 --iters 30
# JSON + Markdown output in benchmark/results/
```

License for the Python kernel library files: Apache-2.0 (per-file SPDX
header). The existing CMake build (above) remains under BSL 1.1 as before.

## License

Business Source License 1.1. See [LICENSE](LICENSE).
