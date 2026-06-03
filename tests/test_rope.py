# SPDX-License-Identifier: Apache-2.0
"""Parity tests for RoPE."""

from __future__ import annotations

import pytest
import torch

from kernels.rope import apply_rope, apply_rope_, build_rope_tables
from kernels.rope.triton_rope import torch_reference_rope
from kernels._common.launch import has_triton


pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleaved", [True, False])
def test_rope_parity(dtype, interleaved):
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(0)
    B, S, H, D = 2, 64, 4, 64
    x = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    cos, sin = build_rope_tables(
        S, D, device=torch.device("cuda"), dtype=dtype, interleaved=interleaved,
    )

    ref = torch_reference_rope(x, cos, sin, interleaved=interleaved, layout="bshd")
    out = apply_rope(
        x, cos, sin, interleaved=interleaved, layout="bshd", backend="triton",
    )
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


def test_rope_inplace_matches_outplace():
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(1)
    B, S, H, D = 1, 32, 2, 64
    dtype = torch.float16
    x = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    cos, sin = build_rope_tables(S, D, device=torch.device("cuda"), dtype=dtype)

    expected = apply_rope(x, cos, sin, backend="triton")
    x_inplace = x.clone()
    apply_rope_(x_inplace, cos, sin, backend="triton")

    torch.testing.assert_close(x_inplace, expected, atol=0, rtol=0)


def test_rope_bhsd_layout():
    """Layout='bhsd' matches the transpose-call-transpose path."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(2)
    B, H, S, D = 1, 2, 16, 64
    dtype = torch.float16
    x = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    cos, sin = build_rope_tables(S, D, device=torch.device("cuda"), dtype=dtype)

    out = apply_rope(x, cos, sin, layout="bhsd", backend="triton")
    ref = torch_reference_rope(x, cos, sin, layout="bhsd")
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)


def test_rope_zero_position_is_identity():
    """At position 0, cos=1 and sin=0, so RoPE is the identity."""
    if not has_triton():
        pytest.skip("triton not importable")
    B, S, H, D = 1, 1, 2, 64
    dtype = torch.float32
    x = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    cos, sin = build_rope_tables(S, D, device=torch.device("cuda"), dtype=dtype)
    out = apply_rope(x, cos, sin, backend="triton")
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
