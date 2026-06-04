# SPDX-License-Identifier: Apache-2.0
"""Tests for paged KV cache (append + lookup + scatter)."""

from __future__ import annotations

import pytest
import torch

from kernels.kv_cache import (
    kv_paged_append,
    kv_paged_lookup,
    kv_paged_scatter,
)
from kernels.kv_cache.triton_kv_paged import (
    torch_reference_kv_paged_append,
    torch_reference_kv_paged_lookup,
)
from kernels._common.launch import has_triton


pytestmark = [pytest.mark.gpu]


def _allocate_cache(num_blocks, num_heads, block_size, head_dim, dtype):
    k = torch.zeros(
        num_blocks, num_heads, block_size, head_dim,
        dtype=dtype, device="cuda",
    )
    v = torch.zeros_like(k)
    return k, v


def test_append_parity():
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(0)
    N, num_heads, head_dim = 32, 8, 64
    block_size, num_blocks = 16, 8
    dtype = torch.float16

    k_in = torch.randn(N, num_heads, head_dim, dtype=dtype, device="cuda")
    v_in = torch.randn_like(k_in)
    # Distinct slot per token; spread across multiple pages.
    slot_mapping = torch.arange(N, dtype=torch.int64, device="cuda") * 2

    k_tri, v_tri = _allocate_cache(num_blocks, num_heads, block_size, head_dim, dtype)
    k_ref, v_ref = _allocate_cache(num_blocks, num_heads, block_size, head_dim, dtype)

    kv_paged_append(k_in, v_in, k_tri, v_tri, slot_mapping, backend="triton")
    torch_reference_kv_paged_append(k_in, v_in, k_ref, v_ref, slot_mapping)

    torch.testing.assert_close(k_tri, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_tri, v_ref, atol=0, rtol=0)


def test_lookup_parity():
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(1)
    B = 4
    num_heads, head_dim = 8, 64
    block_size = 16
    ctx_len = 96  # 6 pages per sequence
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = B * max_blocks_per_seq + 4
    dtype = torch.float16

    k_cache = torch.randn(
        num_blocks, num_heads, block_size, head_dim,
        dtype=dtype, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    # Each sequence gets a contiguous slice of pages.
    block_table = torch.arange(
        B * max_blocks_per_seq, dtype=torch.int64, device="cuda",
    ).reshape(B, max_blocks_per_seq)
    context_lens = torch.full((B,), ctx_len, dtype=torch.int32, device="cuda")

    k_tri, v_tri = kv_paged_lookup(
        k_cache, v_cache, block_table, context_lens, ctx_len, backend="triton",
    )
    k_ref, v_ref = torch_reference_kv_paged_lookup(
        k_cache, v_cache, block_table, context_lens, ctx_len,
    )
    torch.testing.assert_close(k_tri, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(v_tri, v_ref, atol=0, rtol=0)


def test_lookup_short_context_zero_filled():
    """Positions past context_lens[b] must be zero."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(2)
    B = 2
    num_heads, head_dim = 4, 32
    block_size = 8
    max_ctx = 64
    max_blocks_per_seq = max_ctx // block_size
    num_blocks = B * max_blocks_per_seq + 2
    dtype = torch.float16

    k_cache = torch.randn(
        num_blocks, num_heads, block_size, head_dim,
        dtype=dtype, device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(
        B * max_blocks_per_seq, dtype=torch.int64, device="cuda",
    ).reshape(B, max_blocks_per_seq)
    context_lens = torch.tensor([16, 40], dtype=torch.int32, device="cuda")

    k, v = kv_paged_lookup(
        k_cache, v_cache, block_table, context_lens, max_ctx, backend="triton",
    )

    # Pad region for seq 0 starts at position 16.
    assert (k[0, :, 16:, :] == 0).all()
    assert (v[0, :, 16:, :] == 0).all()
    # Pad region for seq 1 starts at position 40.
    assert (k[1, :, 40:, :] == 0).all()
    assert (v[1, :, 40:, :] == 0).all()


def test_scatter_parity():
    """Scatter is functionally identical to append for the same slots."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(3)
    N, num_heads, head_dim = 8, 4, 32
    block_size, num_blocks = 8, 4
    dtype = torch.float16

    k_update = torch.randn(N, num_heads, head_dim, dtype=dtype, device="cuda")
    v_update = torch.randn_like(k_update)
    slot_mapping = torch.arange(N, dtype=torch.int64, device="cuda")

    k_a, v_a = _allocate_cache(num_blocks, num_heads, block_size, head_dim, dtype)
    k_b, v_b = _allocate_cache(num_blocks, num_heads, block_size, head_dim, dtype)

    kv_paged_append(k_update, v_update, k_a, v_a, slot_mapping, backend="triton")
    kv_paged_scatter(k_update, v_update, k_b, v_b, slot_mapping, backend="triton")

    torch.testing.assert_close(k_a, k_b, atol=0, rtol=0)
    torch.testing.assert_close(v_a, v_b, atol=0, rtol=0)
