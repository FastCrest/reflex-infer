# SPDX-License-Identifier: Apache-2.0
"""Parity tests for fused attention vs torch SDPA."""

from __future__ import annotations

import math

import pytest
import torch

from kernels.attention import fused_attention, torch_reference_attention
from kernels._common.launch import has_triton


pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_parity_small(dtype, causal):
    """Triton output matches torch SDPA within tier-appropriate tol."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(0)
    B, H, S, D = 2, 4, 64, 64
    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")

    ref = torch_reference_attention(q, k, v, is_causal=causal)
    out = fused_attention(q, k, v, is_causal=causal, backend="triton")

    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


@pytest.mark.parametrize("S_q,S_k", [(128, 128), (256, 1024), (1, 2048)])
def test_attention_parity_shapes(S_q, S_k):
    """Parity holds for prefill, decode-like, and asymmetric shapes."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(1)
    B, H, D = 1, 4, 64
    dtype = torch.float16
    q = torch.randn(B, H, S_q, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, S_k, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, S_k, D, dtype=dtype, device="cuda")

    # Only the "prefill / decode" causal regime is well-defined when S_q != S_k;
    # SDPA's is_causal masks the upper triangle of a square matrix, so we test
    # the non-causal case for asymmetric shapes.
    causal = (S_q == S_k)

    ref = torch_reference_attention(q, k, v, is_causal=causal)
    out = fused_attention(q, k, v, is_causal=causal, backend="triton")
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_attention_gqa_parity():
    """GQA: 32 Q heads sharing 4 K/V heads matches the expanded reference."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(2)
    B, H, H_kv, S, D = 1, 32, 4, 256, 64
    dtype = torch.float16
    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")

    out = fused_attention(q, k, v, is_causal=True, backend="triton")

    # Reference: expand K, V along the head axis.
    repeat = H // H_kv
    k_e = k.repeat_interleave(repeat, dim=1)
    v_e = v.repeat_interleave(repeat, dim=1)
    ref = torch_reference_attention(q, k_e, v_e, is_causal=True)

    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_attention_dispatch_torch_fallback():
    """Forcing the 'torch' backend always matches the SDPA reference."""
    torch.manual_seed(3)
    B, H, S, D = 1, 2, 16, 32
    q = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.float32, device="cuda")

    out = fused_attention(q, k, v, backend="torch")
    ref = torch_reference_attention(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_attention_sm_scale_override():
    """Custom sm_scale propagates correctly."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(4)
    B, H, S, D = 1, 2, 32, 32
    q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

    scale = 1.0 / (math.sqrt(D) * 0.5)  # half the default
    out = fused_attention(q, k, v, sm_scale=scale, backend="triton")
    ref = torch_reference_attention(q, k, v, sm_scale=scale)
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
