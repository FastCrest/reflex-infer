# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Run every kernel benchmark and emit a combined summary.

Usage:
    python -m benchmark.run_all
    python -m benchmark.run_all --iters 100 --warmup 10

Each per-kernel benchmark writes its own JSON + MD in
``benchmark/results/``. ``run_all`` additionally writes
``benchmark/results/summary.md`` with one table per kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from . import bench_attention, bench_fused_linear_norm, bench_kv_cache, bench_rope, bench_softmax
from ._harness import device_info, write_results


BENCHMARKS = [
    ("attention", bench_attention.run_bench),
    ("kv_cache", bench_kv_cache.run_bench),
    ("fused_linear_norm", bench_fused_linear_norm.run_bench),
    ("softmax", bench_softmax.run_bench),
    ("rope", bench_rope.run_bench),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = ap.parse_args()

    summary: Dict[str, List[Dict]] = {}
    for name, fn in BENCHMARKS:
        print(f"\n=== {name} ===")
        try:
            results = fn(args)
        except Exception as e:
            print(f"benchmark {name} failed: {e}")
            results = []
        write_results(results, args.out, name)
        summary[name] = results

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"device": device_info(), "benchmarks": summary},
            indent=2,
        )
    )

    md_lines = [
        "# reflex_infer benchmark summary",
        "",
        f"Device: `{device_info()['device']}`",
        "",
    ]
    for name, results in summary.items():
        md_lines.append(f"## {name}")
        md_lines.append("")
        md_lines.append("| config | backend | median ms | TFLOPS | GB/s | speedup |")
        md_lines.append("|--------|---------|----------:|-------:|-----:|--------:|")
        for r in results:
            cfg = ", ".join(f"{k}={v}" for k, v in r.get("config", {}).items())
            speedup = r.get("speedup_vs_torch_ref", r.get("speedup_vs_torch_sdpa"))
            md_lines.append(
                f"| {cfg} | {r.get('backend', '?')} | "
                f"{r.get('median_ms', float('nan')):.3f} | "
                f"{r.get('tflops', 0.0):.2f} | "
                f"{r.get('gbps', 0.0):.1f} | "
                f"{(f'{speedup:.2f}x' if speedup else '-')} |"
            )
        md_lines.append("")
    (args.out / "summary.md").write_text("\n".join(md_lines))
    print(f"\nWrote summary to {args.out}/summary.md")


if __name__ == "__main__":
    main()
