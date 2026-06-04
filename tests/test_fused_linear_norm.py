# SPDX-License-Identifier: Apache-2.0
"""Parity tests for fused Linear + (Layer | RMS) Norm."""

from __future__ import annotations

import pytest
import torch

from kernels.fused_linear_norm import (
    fused_linear_layernorm,
    fused_linear_rmsnorm,
)
from kernels.fused_linear_norm.triton_kernel import (
    torch_reference_fused_linear_norm,
)
from kernels._common.launch import has_triton


pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [True, False])
def test_layernorm_parity(dtype, has_bias):
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(0)
    B, S, D_in, D_out = 2, 32, 128, 256
    x = torch.randn(B, S, D_in, dtype=dtype, device="cuda")
    W = torch.randn(D_out, D_in, dtype=dtype, device="cuda") / D_in ** 0.5
    bias = torch.randn(D_out, dtype=dtype, device="cuda") if has_bias else None
    gamma = torch.randn(D_out, dtype=dtype, device="cuda")
    beta = torch.randn(D_out, dtype=dtype, device="cuda")

    ref = torch_reference_fused_linear_norm(
        x, W, bias=bias, gamma=gamma, beta=beta, norm_type="layer",
    )
    out = fused_linear_layernorm(
        x, W, bias=bias, gamma=gamma, beta=beta, backend="triton",
    )
    atol = 5e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_parity(dtype):
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(1)
    B, S, D_in, D_out = 1, 64, 512, 512
    x = torch.randn(B, S, D_in, dtype=dtype, device="cuda")
    W = torch.randn(D_out, D_in, dtype=dtype, device="cuda") / D_in ** 0.5
    gamma = torch.randn(D_out, dtype=dtype, device="cuda")

    ref = torch_reference_fused_linear_norm(
        x, W, gamma=gamma, norm_type="rms",
    )
    out = fused_linear_rmsnorm(
        x, W, gamma=gamma, backend="triton",
    )
    atol = 5e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


def test_no_affine_parity():
    """No gamma/beta: pure normalization after linear."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(2)
    B, S, D_in, D_out = 1, 16, 64, 64
    dtype = torch.float16
    x = torch.randn(B, S, D_in, dtype=dtype, device="cuda")
    W = torch.randn(D_out, D_in, dtype=dtype, device="cuda") / D_in ** 0.5

    ref = torch_reference_fused_linear_norm(
        x, W, norm_type="layer",
    )
    out = fused_linear_layernorm(
        x, W, backend="triton",
    )
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
