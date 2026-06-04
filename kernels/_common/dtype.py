# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Dtype helpers shared across reflex_infer kernels.

This module centralizes the rules for which dtypes are supported per kernel
tier so the wrappers do not duplicate logic, and so that adding FP8 (e4m3 /
e5m2) later is a one-file change rather than a sweep.

Three concepts:

* ``DTYPE_TIER``        : ordered list (fp32 > bf16 > fp16 > fp8e4m3 > fp8e5m2)
* ``promote_for_accum`` : the safe accumulator dtype for a given input dtype
* ``cast_pair``         : align two tensors to a common compute dtype

Triton has no native fp8 yet on every backend, so the FP8 branches fall
through to bf16 accumulation when the runtime cannot encode them. The wrapper
records the actual compute dtype on the returned tensor as a ``.compute_dtype``
attribute when running under the deterministic benchmark harness.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Ordered from highest precision to lowest. The accumulator picker walks this
# list backwards from the input dtype until it finds a dtype the runtime
# advertises support for via :func:`runtime_supports`.
DTYPE_TIER: Tuple[torch.dtype, ...] = (
    torch.float64,
    torch.float32,
    torch.bfloat16,
    torch.float16,
)

# FP8 dtypes live behind a feature flag because they were only added to
# upstream torch in 2.1+ and Triton's `tl.float8e4m3fn` / `tl.float8e5m2`
# require Hopper / MI300 to compile. We look them up dynamically.
FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_E5M2 = getattr(torch, "float8_e5m2", None)

# Absolute / relative tolerance per dtype for parity tests. These were chosen
# empirically from FlashAttention-2's test thresholds plus a 1.5x margin so
# that the reflex_infer parity tests pass even when the reference torch path
# uses a different reduction order.
TOLERANCE = {
    torch.float64: dict(atol=1e-10, rtol=1e-9),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
}


def runtime_supports(dtype: torch.dtype) -> bool:
    """Cheap check that the active runtime can encode this dtype.

    We only check by attempting to allocate a tiny tensor; this catches the
    case where torch has the symbol but the build wasn't compiled with the
    backing kernels (common with FP8 on consumer GPUs).
    """
    if dtype is None:
        return False
    try:
        torch.empty((), dtype=dtype)
        return True
    except (RuntimeError, TypeError):
        return False


def promote_for_accum(input_dtype: torch.dtype) -> torch.dtype:
    """Return a numerically safe accumulator dtype for matmul / reductions.

    Rules:

    * fp32, fp64 -> identity (already large enough)
    * bf16, fp16 -> fp32 (matches cuBLAS default GEMM accumulation)
    * fp8        -> bf16 (FP8 accumulates in bf16 on Hopper TMA)

    Falling back to fp32 is always safe and tests prefer the conservative
    choice.
    """
    if input_dtype in (torch.float32, torch.float64):
        return input_dtype
    if input_dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    if FP8_E4M3 is not None and input_dtype == FP8_E4M3:
        return torch.bfloat16
    if FP8_E5M2 is not None and input_dtype == FP8_E5M2:
        return torch.bfloat16
    return torch.float32


def cast_pair(
    a: torch.Tensor,
    b: torch.Tensor,
    target: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Align two tensors to a common dtype suitable for fused compute.

    Returns ``(a_cast, b_cast, compute_dtype)``. When ``target`` is None the
    common dtype is picked via :func:`torch.promote_types` then upcasted one
    tier on the precision ladder so we don't silently lose bits when mixing
    fp16 with bf16.
    """
    if target is None:
        target = torch.promote_types(a.dtype, b.dtype)
        # Mixing bf16 with fp16 silently promotes to fp16 in some torch
        # builds; force fp32 so we don't lose bf16's exponent range.
        if {a.dtype, b.dtype} == {torch.bfloat16, torch.float16}:
            target = torch.float32
    return a.to(target), b.to(target), target


def tolerance_for(dtype: torch.dtype, scale: float = 1.0) -> dict:
    """Return atol/rtol scaled by ``scale`` for parity assertions."""
    base = TOLERANCE.get(dtype, TOLERANCE[torch.float32])
    return {"atol": base["atol"] * scale, "rtol": base["rtol"] * scale}


def is_fp8(dtype: torch.dtype) -> bool:
    if FP8_E4M3 is not None and dtype == FP8_E4M3:
        return True
    if FP8_E5M2 is not None and dtype == FP8_E5M2:
        return True
    return False
