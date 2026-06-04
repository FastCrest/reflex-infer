# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Public surface for RoPE."""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import select_backend
from .triton_rope import (
    build_rope_tables,
    torch_reference_rope,
    triton_apply_rope,
)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    layout: str = "bshd",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Apply RoPE to ``x``; returns a new tensor."""
    choice = select_backend(backend)
    if choice == "triton":
        return triton_apply_rope(
            x, cos, sin,
            interleaved=interleaved,
            layout=layout,
            inplace=False,
        )
    return torch_reference_rope(
        x, cos, sin, interleaved=interleaved, layout=layout
    )


def apply_rope_(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    layout: str = "bshd",
    backend: Optional[str] = None,
) -> torch.Tensor:
    """In-place RoPE. Returns the (mutated) input tensor."""
    choice = select_backend(backend)
    if choice == "triton":
        return triton_apply_rope(
            x, cos, sin,
            interleaved=interleaved,
            layout=layout,
            inplace=True,
        )
    # Fallback path: compute reference then copy in.
    y = torch_reference_rope(
        x, cos, sin, interleaved=interleaved, layout=layout
    )
    x.copy_(y)
    return x


__all__ = [
    "apply_rope",
    "apply_rope_",
    "build_rope_tables",
]
