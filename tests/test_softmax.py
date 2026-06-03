# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the online softmax kernel."""

from __future__ import annotations

import pytest
import torch

from kernels.softmax import online_softmax
from kernels.softmax.triton_softmax import torch_reference_softmax
from kernels._common.launch import has_triton


pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("N", [64, 1024, 4096])
def test_softmax_parity(dtype, N):
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(0)
    M = 32
    x = torch.randn(M, N, dtype=dtype, device="cuda")

    ref = torch_reference_softmax(x)
    out = online_softmax(x, backend="triton")

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)
    # Each row sums to ~1.
    sums = out.sum(dim=-1).float()
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_log_softmax_parity(dtype):
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(1)
    x = torch.randn(8, 1024, dtype=dtype, device="cuda")
    ref = torch_reference_softmax(x, log=True)
    out = online_softmax(x, log=True, backend="triton")
    atol = 5e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


def test_softmax_extreme_values():
    """Numerical stability with values that would overflow exp without
    the running-max subtraction."""
    if not has_triton():
        pytest.skip("triton not importable")
    x = torch.tensor(
        [[1e3, 1e3 + 5, 1e3 - 1, 1e3], [0.0, 1.0, 2.0, 3.0]],
        dtype=torch.float32, device="cuda",
    )
    out = online_softmax(x, backend="triton")
    ref = torch_reference_softmax(x)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(out).all()


def test_softmax_causal_parity():
    """Causal mode matches the additive-mask reference."""
    if not has_triton():
        pytest.skip("triton not importable")
    torch.manual_seed(2)
    N = 128
    x = torch.randn(N, N, dtype=torch.float32, device="cuda")
    ref = torch_reference_softmax(x, is_causal=True)
    out = online_softmax(x, is_causal=True, backend="triton")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
