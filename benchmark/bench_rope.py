# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Benchmark RoPE vs the torch reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from kernels.rope import apply_rope, build_rope_tables
from kernels.rope.triton_rope import torch_reference_rope

from ._harness import compute_gbps, cuda_time, summarize, write_results


# (name, B, S, H, D, dtype, interleaved)
CONFIGS = [
    ("llama8b_prefill", 1,  4096, 32, 128, torch.bfloat16, False),
    ("llama8b_decode",  16, 1,    32, 128, torch.bfloat16, False),
    ("interleaved",     4,  1024, 16, 64,  torch.float16,  True),
]


def run_bench(args) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for (name, B, S, H, D, dtype, interleaved) in CONFIGS:
        x = torch.randn(B, S, H, D, dtype=dtype, device=device)
        cos, sin = build_rope_tables(
            S, D, device=torch.device(device), dtype=dtype,
            interleaved=interleaved,
        )
        bytes_moved = 2 * x.numel() * x.element_size()  # read + write

        def _ref():
            return torch_reference_rope(
                x, cos, sin, interleaved=interleaved, layout="bshd",
            )

        def _triton():
            return apply_rope(
                x, cos, sin, interleaved=interleaved, layout="bshd",
                backend="triton",
            )

        try:
            ref_times = cuda_time(_ref, warmup=args.warmup, iters=args.iters)
            ref_stats = summarize(ref_times)
        except Exception as e:
            print(f"[{name}] ref failed: {e}")
            ref_stats = None
        try:
            t_times = cuda_time(_triton, warmup=args.warmup, iters=args.iters)
            t_stats = summarize(t_times)
        except Exception as e:
            print(f"[{name}] triton failed: {e}")
            t_stats = None

        cfg = dict(name=name, B=B, S=S, H=H, D=D, dtype=str(dtype),
                   interleaved=interleaved)
        if ref_stats is not None:
            results.append({
                "config": cfg, "backend": "torch_ref",
                **ref_stats,
                "gbps": compute_gbps(bytes_moved, ref_stats["median_ms"]),
            })
        if t_stats is not None:
            entry = {
                "config": cfg, "backend": "reflex_triton",
                **t_stats,
                "gbps": compute_gbps(bytes_moved, t_stats["median_ms"]),
            }
            if ref_stats is not None:
                entry["speedup_vs_torch_ref"] = ref_stats["median_ms"] / t_stats["median_ms"]
            results.append(entry)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    results = run_bench(args)
    write_results(results, args.out, "rope")
    print(f"Wrote {len(results)} results to {args.out}/rope.json")


if __name__ == "__main__":
    main()
