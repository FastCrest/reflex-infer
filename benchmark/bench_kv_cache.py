# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Benchmark paged KV cache append + lookup.

We test:

* append a batch of N new tokens (one per sequence in the batch).
* lookup a batch of sequences with varying context length.

Compared against the torch reference (scatter via indexing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from kernels.kv_cache import kv_paged_append, kv_paged_lookup
from kernels.kv_cache.triton_kv_paged import (
    torch_reference_kv_paged_append,
    torch_reference_kv_paged_lookup,
)

from ._harness import (
    compute_gbps,
    cuda_time,
    summarize,
    write_results,
)


# (name, B, num_heads, head_dim, block_size, ctx_len, dtype)
CONFIGS = [
    ("append_b32",  32, 32, 128, 16, 2048, torch.float16),
    ("append_b128", 128, 32, 128, 16, 2048, torch.float16),
    ("lookup_b32_ctx2k",  32, 32, 128, 16, 2048, torch.float16),
    ("lookup_b8_ctx8k",   8, 32, 128, 16, 8192, torch.float16),
]


def _make_paged_cache(num_heads, head_dim, block_size, num_blocks, dtype, device):
    k = torch.zeros(num_blocks, num_heads, block_size, head_dim, dtype=dtype, device=device)
    v = torch.zeros_like(k)
    return k, v


def run_bench(args) -> List[Dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for (name, B, num_heads, head_dim, block_size, ctx_len, dtype) in CONFIGS:
        max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
        total_blocks = B * max_blocks_per_seq + 16

        k_cache, v_cache = _make_paged_cache(
            num_heads, head_dim, block_size, total_blocks, dtype, device,
        )

        if name.startswith("append"):
            # One new token per sequence.
            k_in = torch.randn(B, num_heads, head_dim, dtype=dtype, device=device)
            v_in = torch.randn_like(k_in)
            # Pick slots in distinct pages so we don't artificially serialize.
            slot_mapping = (torch.arange(B, device=device, dtype=torch.int64)
                            * block_size)
            bytes_moved = (
                2 * B * num_heads * head_dim * k_in.element_size()
                + 8 * B  # slot indices
            )

            def _ref():
                # Reset to avoid stateful comparisons; not measured because
                # we wrap the actual call in cuda_time and the reset is part
                # of every iter for both paths.
                torch_reference_kv_paged_append(
                    k_in, v_in, k_cache, v_cache, slot_mapping,
                )

            def _triton():
                kv_paged_append(
                    k_in, v_in, k_cache, v_cache, slot_mapping,
                    backend="triton",
                )

            try:
                ref_times = cuda_time(_ref, warmup=args.warmup, iters=args.iters)
                ref_stats = summarize(ref_times)
            except Exception as e:
                print(f"[{name}] reference failed: {e}")
                ref_stats = None
            try:
                trit_times = cuda_time(_triton, warmup=args.warmup, iters=args.iters)
                trit_stats = summarize(trit_times)
            except Exception as e:
                print(f"[{name}] triton failed: {e}")
                trit_stats = None
        else:
            # Lookup
            context_lens = torch.full((B,), ctx_len, dtype=torch.int32, device=device)
            block_table = torch.randint(
                0, total_blocks, (B, max_blocks_per_seq),
                dtype=torch.int64, device=device,
            )
            # Fill the cache so the lookup has real reads.
            k_cache.normal_()
            v_cache.normal_()
            bytes_moved = 2 * B * num_heads * ctx_len * head_dim * k_cache.element_size()

            def _ref():
                return torch_reference_kv_paged_lookup(
                    k_cache, v_cache, block_table, context_lens, ctx_len,
                )

            def _triton():
                return kv_paged_lookup(
                    k_cache, v_cache, block_table, context_lens, ctx_len,
                    backend="triton",
                )

            try:
                ref_times = cuda_time(_ref, warmup=args.warmup, iters=args.iters)
                ref_stats = summarize(ref_times)
            except Exception as e:
                print(f"[{name}] reference failed: {e}")
                ref_stats = None
            try:
                trit_times = cuda_time(_triton, warmup=args.warmup, iters=args.iters)
                trit_stats = summarize(trit_times)
            except Exception as e:
                print(f"[{name}] triton failed: {e}")
                trit_stats = None

        cfg = dict(
            name=name, B=B, num_heads=num_heads, head_dim=head_dim,
            block_size=block_size, ctx_len=ctx_len, dtype=str(dtype),
        )
        if ref_stats is not None:
            results.append({
                "config": cfg, "backend": "torch_ref",
                **ref_stats,
                "gbps": compute_gbps(bytes_moved, ref_stats["median_ms"]),
            })
        if trit_stats is not None:
            entry = {
                "config": cfg, "backend": "reflex_triton",
                **trit_stats,
                "gbps": compute_gbps(bytes_moved, trit_stats["median_ms"]),
            }
            if ref_stats is not None:
                entry["speedup_vs_torch_ref"] = (
                    ref_stats["median_ms"] / trit_stats["median_ms"]
                )
            results.append(entry)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    results = run_bench(args)
    write_results(results, args.out, "kv_cache")
    print(f"Wrote {len(results)} results to {args.out}/kv_cache.json")


if __name__ == "__main__":
    main()
