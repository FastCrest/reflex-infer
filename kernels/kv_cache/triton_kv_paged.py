# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Triton paged KV cache kernels (vLLM-style).

The paged KV cache stores K and V in fixed-size pages so logical sequences
can be allocated without contiguous physical memory; this is what enables
batching variable-length sequences without padding.

Storage layout (matches vLLM's):

* ``k_cache``: [num_blocks, num_heads, block_size, head_dim]
* ``v_cache``: [num_blocks, num_heads, block_size, head_dim]
* ``block_table``: [num_sequences, max_blocks_per_seq] -> page indices
* ``slot_mapping``: [total_tokens] -> flat slot index
    = page_index * block_size + position_in_page

Three operations:

1. **append**: write new K, V vectors for the just-decoded token(s) into the
   pages indicated by slot_mapping. One slot per token.
2. **lookup** (gather): given a block_table and a context_length per
   sequence, materialize [batch, num_heads, max_ctx, head_dim] K/V for the
   attention kernel.
3. **scatter**: write back updates to specific (sequence, position) slots,
   used by speculative decoding rollback.

We provide Triton kernels for all three. They are pure memory-movement
kernels so the win vs torch is mostly avoiding the gather-via-indexing
overhead and avoiding the materialized copies.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import maybe_triton

triton = maybe_triton()

if triton is not None:
    import triton.language as tl

    # ---------- append ----------
    @triton.jit
    def _kv_paged_append_kernel(
        k_in,       # [N, num_heads, head_dim]
        v_in,
        k_cache,    # [num_blocks, num_heads, block_size, head_dim]
        v_cache,
        slot_mapping,  # [N] int64
        # Strides.
        ki_n, ki_h, ki_d,
        vi_n, vi_h, vi_d,
        kc_b, kc_h, kc_s, kc_d,
        vc_b, vc_h, vc_s, vc_d,
        # Sizes.
        N: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program handles BLOCK_N tokens for a single head.
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        offs_d = tl.arange(0, HEAD_DIM)

        # Load slot indices for this tile.
        slots = tl.load(slot_mapping + offs_n, mask=n_mask, other=0)
        # page_idx = slot // BLOCK_SIZE; pos_in_page = slot % BLOCK_SIZE.
        page_idx = slots // BLOCK_SIZE
        pos = slots % BLOCK_SIZE

        # Load K and V from the dense input tensor.
        k_ptrs = k_in + offs_n[:, None] * ki_n + pid_h * ki_h + offs_d[None, :] * ki_d
        v_ptrs = v_in + offs_n[:, None] * vi_n + pid_h * vi_h + offs_d[None, :] * vi_d
        k_vals = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v_vals = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Compute destination pointers into the paged cache.
        kc_ptrs = (
            k_cache
            + page_idx[:, None] * kc_b
            + pid_h * kc_h
            + pos[:, None] * kc_s
            + offs_d[None, :] * kc_d
        )
        vc_ptrs = (
            v_cache
            + page_idx[:, None] * vc_b
            + pid_h * vc_h
            + pos[:, None] * vc_s
            + offs_d[None, :] * vc_d
        )

        tl.store(kc_ptrs, k_vals, mask=n_mask[:, None])
        tl.store(vc_ptrs, v_vals, mask=n_mask[:, None])

    # ---------- lookup (gather) ----------
    @triton.jit
    def _kv_paged_lookup_kernel(
        k_cache,
        v_cache,
        k_out,         # [B, num_heads, max_ctx, head_dim]
        v_out,
        block_table,   # [B, max_blocks_per_seq] int64
        context_lens,  # [B] int32
        # Strides.
        kc_b, kc_h, kc_s, kc_d,
        vc_b, vc_h, vc_s, vc_d,
        ko_b, ko_h, ko_s, ko_d,
        vo_b, vo_h, vo_s, vo_d,
        bt_b, bt_blk,
        # Sizes.
        BLOCK_SIZE: tl.constexpr,
        MAX_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # One program per (sequence, head, token-tile).
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)

        ctx_len = tl.load(context_lens + pid_b)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < ctx_len
        offs_d = tl.arange(0, HEAD_DIM)

        # block_idx = offs_n // BLOCK_SIZE; pos = offs_n % BLOCK_SIZE.
        blk_idx = offs_n // BLOCK_SIZE
        pos = offs_n % BLOCK_SIZE

        # Look up physical page indices via the block table.
        bt_ptrs = block_table + pid_b * bt_b + blk_idx * bt_blk
        page_idx = tl.load(bt_ptrs, mask=n_mask, other=0)

        # Gather K and V from the cache.
        kc_ptrs = (
            k_cache
            + page_idx[:, None] * kc_b
            + pid_h * kc_h
            + pos[:, None] * kc_s
            + offs_d[None, :] * kc_d
        )
        vc_ptrs = (
            v_cache
            + page_idx[:, None] * vc_b
            + pid_h * vc_h
            + pos[:, None] * vc_s
            + offs_d[None, :] * vc_d
        )
        k_vals = tl.load(kc_ptrs, mask=n_mask[:, None], other=0.0)
        v_vals = tl.load(vc_ptrs, mask=n_mask[:, None], other=0.0)

        # Store to dense output.
        ko_ptrs = (
            k_out
            + pid_b * ko_b
            + pid_h * ko_h
            + offs_n[:, None] * ko_s
            + offs_d[None, :] * ko_d
        )
        vo_ptrs = (
            v_out
            + pid_b * vo_b
            + pid_h * vo_h
            + offs_n[:, None] * vo_s
            + offs_d[None, :] * vo_d
        )
        tl.store(ko_ptrs, k_vals, mask=n_mask[:, None])
        tl.store(vo_ptrs, v_vals, mask=n_mask[:, None])


def triton_kv_paged_append(
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Append new K, V vectors into the paged cache, in place.

    Args:
        k_in: [N, num_heads, head_dim] new K vectors.
        v_in: [N, num_heads, head_dim] new V vectors.
        k_cache: [num_blocks, num_heads, block_size, head_dim] paged K cache.
        v_cache: [num_blocks, num_heads, block_size, head_dim] paged V cache.
        slot_mapping: [N] int64 flat slot indices.

    No return; the cache tensors are mutated.
    """
    if triton is None:
        raise RuntimeError("triton not importable")
    N, num_heads, head_dim = k_in.shape
    num_blocks, _, block_size, _ = k_cache.shape
    BLOCK_N = 32

    grid = (triton.cdiv(N, BLOCK_N), num_heads)
    _kv_paged_append_kernel[grid](
        k_in, v_in, k_cache, v_cache, slot_mapping,
        *k_in.stride(),
        *v_in.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        N, num_heads, head_dim, block_size, BLOCK_N,
    )


def triton_kv_paged_lookup(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx: int,
) -> tuple:
    """Gather K and V from the paged cache into dense tensors.

    Args:
        k_cache, v_cache: paged storage; same layout as in append.
        block_table: [B, max_blocks_per_seq] int64.
        context_lens: [B] int32, valid length per sequence.
        max_ctx: padding length for the output dense tensors.

    Returns:
        (k_out, v_out): [B, num_heads, max_ctx, head_dim] each. Positions
        beyond context_lens[b] are zero-filled (matching torch indexing).
    """
    if triton is None:
        raise RuntimeError("triton not importable")
    B = context_lens.shape[0]
    num_blocks, num_heads, block_size, head_dim = k_cache.shape

    k_out = torch.zeros(
        (B, num_heads, max_ctx, head_dim),
        dtype=k_cache.dtype, device=k_cache.device,
    )
    v_out = torch.empty_like(k_out)

    BLOCK_N = 32
    grid = (B, num_heads, triton.cdiv(max_ctx, BLOCK_N))

    bt_strides = block_table.stride()
    _kv_paged_lookup_kernel[grid](
        k_cache, v_cache, k_out, v_out, block_table, context_lens,
        *k_cache.stride(),
        *v_cache.stride(),
        *k_out.stride(),
        *v_out.stride(),
        bt_strides[0], bt_strides[1],
        block_size, max_ctx, head_dim, BLOCK_N,
    )
    return k_out, v_out


def triton_kv_paged_scatter(
    k_updates: torch.Tensor,
    v_updates: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Overwrite specific slots, used by speculative decoding rollback.

    Conceptually identical to append from the kernel's POV; we expose it as
    a separate function so the caller's intent is documented and so a future
    rollback-with-undo can branch here.
    """
    triton_kv_paged_append(k_updates, v_updates, k_cache, v_cache, slot_mapping)


def torch_reference_kv_paged_append(
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Reference paged-append via scatter_, for parity tests."""
    num_blocks, num_heads, block_size, head_dim = k_cache.shape
    page_idx = slot_mapping // block_size
    pos = slot_mapping % block_size
    # k_cache[page_idx, :, pos, :] = k_in
    k_cache[page_idx, :, pos, :] = k_in
    v_cache[page_idx, :, pos, :] = v_in


def torch_reference_kv_paged_lookup(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx: int,
) -> tuple:
    """Reference paged-lookup; same semantics as triton_kv_paged_lookup."""
    B = context_lens.shape[0]
    num_blocks, num_heads, block_size, head_dim = k_cache.shape
    k_out = torch.zeros(
        (B, num_heads, max_ctx, head_dim),
        dtype=k_cache.dtype, device=k_cache.device,
    )
    v_out = torch.zeros_like(k_out)
    for b in range(B):
        ctx_len = int(context_lens[b].item())
        for t in range(ctx_len):
            blk = int(block_table[b, t // block_size].item())
            pos = t % block_size
            k_out[b, :, t, :] = k_cache[blk, :, pos, :]
            v_out[b, :, t, :] = v_cache[blk, :, pos, :]
    return k_out, v_out


__all__ = [
    "triton_kv_paged_append",
    "triton_kv_paged_lookup",
    "triton_kv_paged_scatter",
    "torch_reference_kv_paged_append",
    "torch_reference_kv_paged_lookup",
]
