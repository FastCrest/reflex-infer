# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Public surface for the fused attention kernel.

Usage:
    from reflex_infer.kernels.attention import fused_attention

    out = fused_attention(q, k, v, is_causal=True)

``fused_attention`` dispatches Triton (Ampere+ / ROCm) or CUDA C++ (older
NVIDIA) or torch SDPA (CPU / unsupported), driven by
``_common.launch.select_backend``.

The torch.compile-friendly ``torch.library`` registration only kicks in when
torch>=2.4 is present; otherwise we still work in eager but dynamo will
graph-break on the boundary.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import select_backend, wrap_kernel
from .triton_fused_attention import (
    torch_reference_attention,
    triton_fused_attention,
)

# CUDA C++ backend is loaded lazily on first use. The .cu file ships as
# source; cpp_extension compiles it the first time it's needed and caches the
# .so in ~/.cache/torch_extensions. This intentionally keeps wheel builds
# fast and avoids requiring nvcc at install time.
_cuda_ext = None


def _load_cuda_extension():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext
    from pathlib import Path
    from torch.utils.cpp_extension import load
    src = Path(__file__).with_name("cuda_fused_attention.cu")
    _cuda_ext = load(
        name="reflex_infer_attention_cuda",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _cuda_ext


def _cuda_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """CUDA fallback. Loads + JITs the .cu on first call.

    Note: the CUDA path does not currently support attn_mask; it falls back
    to torch SDPA in that case (this is a deliberate scope cut for v1; the
    Triton path is the primary fast path and supports both).
    """
    if attn_mask is not None:
        return torch_reference_attention(
            q, k, v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
        )
    import math
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    ext = _load_cuda_extension()
    return ext.fused_attention_cuda(
        q.contiguous(), k.contiguous(), v.contiguous(),
        is_causal, float(sm_scale),
    )


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Backend-selecting fused attention.

    Args mirror ``torch.nn.functional.scaled_dot_product_attention``.

    ``backend`` overrides the auto-selection; useful for benchmarks and
    parity tests. Set to ``triton``, ``cuda``, or ``torch``.
    """
    choice = select_backend(backend)
    if choice == "triton":
        return triton_fused_attention(
            q, k, v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
        )
    if choice == "cuda":
        return _cuda_fused_attention(
            q, k, v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
        )
    return torch_reference_attention(
        q, k, v,
        is_causal=is_causal,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
    )


__all__ = [
    "fused_attention",
    "triton_fused_attention",
    "torch_reference_attention",
]
