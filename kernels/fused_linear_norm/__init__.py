# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Public surface for fused Linear + (Layer | RMS) Norm.

Common pattern in transformer blocks:

    h = LayerNorm(Linear(x))         # pre-LN MLP entry
    h = RMSNorm(Linear(h))           # Llama / Qwen

Fusing the two avoids writing the linear output to HBM, which dominates for
small batch / decoder workloads.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import select_backend
from .triton_kernel import (
    torch_reference_fused_linear_norm,
    triton_fused_linear_norm,
)


def fused_linear_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Fused Linear + LayerNorm."""
    choice = select_backend(backend)
    if choice == "triton":
        return triton_fused_linear_norm(
            x, weight,
            bias=bias, gamma=gamma, beta=beta,
            eps=eps, norm_type="layer",
        )
    return torch_reference_fused_linear_norm(
        x, weight,
        bias=bias, gamma=gamma, beta=beta,
        eps=eps, norm_type="layer",
    )


def fused_linear_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Fused Linear + RMSNorm (Llama / Qwen style)."""
    choice = select_backend(backend)
    if choice == "triton":
        return triton_fused_linear_norm(
            x, weight,
            bias=bias, gamma=gamma, beta=None,
            eps=eps, norm_type="rms",
        )
    return torch_reference_fused_linear_norm(
        x, weight,
        bias=bias, gamma=gamma, beta=None,
        eps=eps, norm_type="rms",
    )


__all__ = [
    "fused_linear_layernorm",
    "fused_linear_rmsnorm",
]
