# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Benchmark online softmax vs torch.softmax."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from kernels.softmax import online_softmax
from kernels.softmax.triton_softmax import torch_reference_softmax

from ._harness import compute_gbps, cuda_time, summarize, write_results


# (name, B, M, N, dtype, causal, log)
CONFIGS = [
    ("attn_score_2k",  4,  32 * 2048, 2048,  torch.float16, True,  False),
    ("attn_score_8k",  1,  32 * 1024, 8192,  torch.float16, True,  False),
    ("lm_head_50k",    8,  1024,      50257, torch.float16, False, False),
    ("logsoftmax_1k",  8,  1024,      1024,  torch.float16, False, True),
]


def run_bench(args) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for (name, B, M, N, dtype, causal, log) in CONFIGS:
        x = torch.randn(B, M, N, dtype=dtype, device=device)
        bytes_moved = 2 * x.numel() * x.element_size()  # read + write

        def _ref():
            return torch_reference_softmax(x, is_causal=causal, log=log)

        def _triton():
            return online_softmax(x, is_causal=causal, log=log, backend="triton")

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

        cfg = dict(name=name, B=B, M=M, N=N, dtype=str(dtype), causal=causal, log=log)
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
    write_results(results, args.out, "softmax")
    print(f"Wrote {len(results)} results to {args.out}/softmax.json")


if __name__ == "__main__":
    main()
