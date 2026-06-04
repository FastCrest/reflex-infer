# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Shared benchmark utilities.

We deliberately don't depend on torch.utils.benchmark because it pulls in
the timer's warmup logic which doesn't play well with Triton autotuning
(autotune itself runs on the first call and looks like an outlier). Instead
we provide a small, explicit benchmark loop that:

* Calls the kernel once to trigger autotune / JIT compilation.
* Calls the kernel ``warmup`` times to warm caches and reach steady-state
  clocks.
* Runs ``iters`` measured calls, each timed individually with CUDA events.
* Returns the median time (robust to occasional context switches) plus the
  p99 for tail-latency reporting.

Reported metrics per kernel:

* ``median_ms``     : steady-state median latency.
* ``p99_ms``        : tail latency.
* ``tflops``        : computed from a kernel-supplied FLOP count.
* ``gbps``          : computed from a kernel-supplied bytes-moved count.
* ``speedup_vs_X``  : ratio of median time of reference X to this kernel.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class BenchResult:
    """One measured run."""
    name: str
    config: Dict[str, object]
    median_ms: float
    p99_ms: float
    tflops: Optional[float] = None
    gbps: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)


def cuda_time(fn: Callable, *, warmup: int = 5, iters: int = 30) -> List[float]:
    """Time ``fn`` with CUDA events. Returns per-iter times in ms.

    Warms up first, then runs ``iters`` measured calls. We use one start/end
    event per iter rather than amortizing because the host-side dispatch
    overhead is part of what we're measuring (a real model calls these
    kernels one at a time).
    """
    if not torch.cuda.is_available():
        # CPU fallback: wall-clock time with perf_counter. Lower fidelity but
        # at least the benchmark script doesn't crash.
        times = []
        for _ in range(warmup):
            fn()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
        return times

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summarize(times: List[float]) -> Dict[str, float]:
    """Compact summary used in JSON output."""
    sorted_t = sorted(times)
    n = len(sorted_t)
    return {
        "median_ms": statistics.median(sorted_t),
        "mean_ms": statistics.mean(sorted_t),
        "p50_ms": sorted_t[n // 2],
        "p99_ms": sorted_t[min(n - 1, max(0, int(n * 0.99) - 1))],
        "min_ms": sorted_t[0],
        "max_ms": sorted_t[-1],
        "n": n,
    }


def compute_tflops(flops: float, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return flops / (ms * 1e-3) / 1e12


def compute_gbps(bytes_moved: float, ms: float) -> float:
    if ms <= 0:
        return 0.0
    return bytes_moved / (ms * 1e-3) / 1e9


def device_info() -> Dict[str, str]:
    """Snapshot of the active device for the JSON output."""
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        return {
            "device": torch.cuda.get_device_name(0),
            "cc": f"{cap[0]}.{cap[1]}",
            "torch_version": torch.__version__,
        }
    return {"device": "cpu", "cc": "n/a", "torch_version": torch.__version__}


def write_results(results: List[Dict], out_dir: Path, name: str) -> None:
    """Write a JSON file with the raw results plus a Markdown table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    md_path = out_dir / f"{name}.md"
    payload = {
        "device": device_info(),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(payload, name))


def _render_markdown(payload: Dict, name: str) -> str:
    lines = [
        f"# {name} benchmark",
        "",
        f"Device: {payload['device']['device']}  ",
        f"Compute capability: {payload['device']['cc']}  ",
        f"Torch: {payload['device']['torch_version']}",
        "",
        "| config | backend | median ms | p99 ms | TFLOPS | GB/s |",
        "|--------|---------|----------:|-------:|-------:|-----:|",
    ]
    for r in payload["results"]:
        cfg = ", ".join(f"{k}={v}" for k, v in r.get("config", {}).items())
        lines.append(
            f"| {cfg} | {r.get('backend', '?')} | "
            f"{r.get('median_ms', float('nan')):.3f} | "
            f"{r.get('p99_ms', float('nan')):.3f} | "
            f"{r.get('tflops', 0.0):.2f} | "
            f"{r.get('gbps', 0.0):.1f} |"
        )
    return "\n".join(lines) + "\n"
