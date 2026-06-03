# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Public surface for the optimized softmax kernel."""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import select_backend
from .triton_softmax import torch_reference_softmax, triton_online_softmax


def online_softmax(
    x: torch.Tensor,
    *,
    is_causal: bool = False,
    log: bool = False,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Row-wise softmax along the last dim.

    Args:
        x: arbitrary shape; softmax is along the last axis.
        is_causal: if True, treats the last-axis index as the position and
            masks positions > row (matches scaled_dot_product_attention's
            causal mode applied to the score matrix).
        log: returns log_softmax instead of softmax.
        backend: one of triton/cuda/torch; default is auto.
    """
    choice = select_backend(backend)
    if choice == "triton":
        return triton_online_softmax(x, is_causal=is_causal, log=log)
    return torch_reference_softmax(x, is_causal=is_causal, log=log)


__all__ = ["online_softmax"]
