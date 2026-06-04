# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Benchmark the fused attention kernel against torch SDPA / FlashAttention.

Sweeps a small workload matrix that covers the regimes we care about:

* short-context decode (B=8,  S_q=1,    S_k=2048)
* long-context prefill (B=2,  S_q=4096, S_k=4096)
* GQA (H=32, H_kv=4) at the long-context prefill shape

We report median latency, TFLOPS (computed from 4 * B * H * S_q * S_k * D
for the QK and PV matmuls, ignoring softmax), and GB/s of K + V read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from kernels.attention import fused_attention, torch_reference_attention

from ._harness import (
    compute_gbps,
    compute_tflops,
    cuda_time,
    summarize,
    write_results,
)


# --- workload definitions ---------------------------------------------------
CONFIGS = [
    # name, B, H, H_kv, S_q, S_k, D, dtype, causal
    ("decode_b8_s2k",  8, 32, 32, 1,    2048, 128, torch.float16, True),
    ("decode_b1_s8k",  1, 32, 32, 1,    8192, 128, torch.float16, True),
    ("prefill_4k",     2, 32, 32, 4096, 4096, 128, torch.float16, True),
    ("gqa_prefill_4k", 2, 32, 4,  4096, 4096, 128, torch.float16, True),
    ("bf16_prefill",   2, 32, 32, 4096, 4096, 128, torch.bfloat16, True),
]


def attention_flops(B: int, H: int, S_q: int, S_k: int, D: int) -> float:
    """FlashAttention's accounting: 4*B*H*S_q*S_k*D for QK + PV.

    Excludes softmax FLOPs because they're tiny vs the matmuls.
    """
    return 4.0 * B * H * S_q * S_k * D


def attention_bytes(B: int, H_kv: int, S_q: int, S_k: int, D: int, dtype) -> float:
    """K + V reads: 2 * B * H_kv * S_k * D * sizeof(dtype). Q + O are tiny."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 2.0 * B * H_kv * S_k * D * elem


def run_bench(args) -> List[Dict]:
    if not torch.cuda.is_available():
        print("CUDA unavailable; running attention benchmark on CPU (low fidelity).")
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for (name, B, H, H_kv, S_q, S_k, D, dtype, causal) in CONFIGS:
        q = torch.randn(B, H, S_q, D, dtype=dtype, device=device)
        k = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)
        v = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)

        # If GQA: SDPA reference needs same heads, so expand for the
        # reference. The triton path handles it internally.
        if H_kv != H:
            k_ref = k.repeat_interleave(H // H_kv, dim=1)
            v_ref = v.repeat_interleave(H // H_kv, dim=1)
        else:
            k_ref = k
            v_ref = v

        flops = attention_flops(B, H, S_q, S_k, D)
        bytes_moved = attention_bytes(B, H_kv, S_q, S_k, D, dtype)

        # --- torch SDPA reference ---
        def _torch_call():
            return torch_reference_attention(
                q, k_ref, v_ref, is_causal=causal,
            )

        # --- reflex_infer Triton path ---
        def _triton_call():
            return fused_attention(
                q, k, v, is_causal=causal, backend="triton",
            )

        try:
            torch_times = cuda_time(
                _torch_call, warmup=args.warmup, iters=args.iters
            )
            torch_stats = summarize(torch_times)
        except Exception as e:
            print(f"[{name}] torch path failed: {e}")
            torch_stats = None

        try:
            triton_times = cuda_time(
                _triton_call, warmup=args.warmup, iters=args.iters
            )
            triton_stats = summarize(triton_times)
        except Exception as e:
            print(f"[{name}] triton path failed: {e}")
            triton_stats = None

        cfg = dict(
            name=name, B=B, H=H, H_kv=H_kv, S_q=S_q, S_k=S_k, D=D,
            dtype=str(dtype), causal=causal,
        )
        if torch_stats is not None:
            results.append({
                "config": cfg,
                "backend": "torch_sdpa",
                **torch_stats,
                "tflops": compute_tflops(flops, torch_stats["median_ms"]),
                "gbps": compute_gbps(bytes_moved, torch_stats["median_ms"]),
            })
        if triton_stats is not None:
            speedup = (
                torch_stats["median_ms"] / triton_stats["median_ms"]
                if torch_stats is not None else None
            )
            entry = {
                "config": cfg,
                "backend": "reflex_triton",
                **triton_stats,
                "tflops": compute_tflops(flops, triton_stats["median_ms"]),
                "gbps": compute_gbps(bytes_moved, triton_stats["median_ms"]),
            }
            if speedup is not None:
                entry["speedup_vs_torch_sdpa"] = speedup
            results.append(entry)

    return results


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

    results = run_bench(args)
    write_results(results, args.out, "attention")
    print(f"Wrote {len(results)} results to {args.out}/attention.json")


if __name__ == "__main__":
    main()
