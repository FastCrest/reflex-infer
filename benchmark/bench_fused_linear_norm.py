# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Benchmark fused Linear + LayerNorm / RMSNorm vs the torch reference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from kernels.fused_linear_norm import (
    fused_linear_layernorm,
    fused_linear_rmsnorm,
)
from kernels.fused_linear_norm.triton_kernel import (
    torch_reference_fused_linear_norm,
)

from ._harness import compute_gbps, compute_tflops, cuda_time, summarize, write_results


# (name, B, S, D_in, D_out, norm_type, dtype)
CONFIGS = [
    ("llama8b_mlp_in",  1, 1024, 4096,  14336, "rms",   torch.bfloat16),
    ("llama8b_attn_out", 1, 1024, 4096,  4096,  "rms",   torch.bfloat16),
    ("gpt2_mlp_in",     4, 1024, 768,   3072,  "layer", torch.float16),
    ("decode_small",    1, 1,    4096,  4096,  "rms",   torch.bfloat16),
]


def run_bench(args) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for (name, B, S, D_in, D_out, norm_type, dtype) in CONFIGS:
        x = torch.randn(B, S, D_in, dtype=dtype, device=device)
        W = torch.randn(D_out, D_in, dtype=dtype, device=device)
        bias = torch.randn(D_out, dtype=dtype, device=device)
        gamma = torch.randn(D_out, dtype=dtype, device=device)
        beta = torch.randn(D_out, dtype=dtype, device=device) if norm_type == "layer" else None

        flops = 2.0 * B * S * D_in * D_out  # matmul, ignore norm flops
        bytes_in = B * S * D_in * x.element_size()
        bytes_w = D_out * D_in * W.element_size()
        bytes_out = B * S * D_out * x.element_size()
        bytes_moved = bytes_in + bytes_w + bytes_out

        def _ref():
            return torch_reference_fused_linear_norm(
                x, W, bias=bias, gamma=gamma, beta=beta,
                norm_type=norm_type,
            )

        if norm_type == "layer":
            def _triton():
                return fused_linear_layernorm(
                    x, W, bias=bias, gamma=gamma, beta=beta, backend="triton",
                )
        else:
            def _triton():
                return fused_linear_rmsnorm(
                    x, W, bias=bias, gamma=gamma, backend="triton",
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

        cfg = dict(name=name, B=B, S=S, D_in=D_in, D_out=D_out,
                   norm_type=norm_type, dtype=str(dtype))
        if ref_stats is not None:
            results.append({
                "config": cfg, "backend": "torch_ref",
                **ref_stats,
                "tflops": compute_tflops(flops, ref_stats["median_ms"]),
                "gbps": compute_gbps(bytes_moved, ref_stats["median_ms"]),
            })
        if t_stats is not None:
            entry = {
                "config": cfg, "backend": "reflex_triton",
                **t_stats,
                "tflops": compute_tflops(flops, t_stats["median_ms"]),
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
    write_results(results, args.out, "fused_linear_norm")
    print(f"Wrote {len(results)} results to {args.out}/fused_linear_norm.json")


if __name__ == "__main__":
    main()
