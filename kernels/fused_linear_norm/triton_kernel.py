# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Triton fused linear + LayerNorm kernel.

Computes:

    y = LayerNorm(x @ W^T + b, gamma, beta, eps)

in a single kernel, so the linear-projection output is never written to HBM
between the matmul and the normalization. The win is purely memory: a
[B*S, D_out] activation can be many MB and writing+reading it round-trips
through L2 / HBM. We instead reduce along the row in-register and write the
normalized output once.

Two variants supported:

* **LayerNorm**: subtract mean, divide by sqrt(var + eps), then scale.
* **RMSNorm**:   divide by sqrt(mean_squared + eps), then scale. (Common in
                  Llama / Qwen / Mistral.)

Pick via the ``norm_type`` argument. The Triton kernel branches on a
compile-time constant so each variant is a separate specialization.

Limitations:
* The matmul tile is single-row by D_in by D_out. For very large D_in this
  is suboptimal; the autotuned path picks BLOCK_K to balance. A more
  aggressive split-K variant is left as a follow-up.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import maybe_triton

triton = maybe_triton()

if triton is not None:
    import triton.language as tl

    _CONFIGS = [
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_CONFIGS, key=["D_IN", "D_OUT"])
    @triton.jit
    def _fused_linear_norm_kernel(
        X, W, BIAS, GAMMA, BETA, Y,
        eps,
        # Strides.
        sx_n, sx_k,
        sw_o, sw_k,
        sy_n, sy_o,
        # Sizes.
        N,
        D_IN: tl.constexpr,
        D_OUT: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NORM_TYPE: tl.constexpr,  # 0 = LayerNorm, 1 = RMSNorm
        HAS_AFFINE: tl.constexpr,
    ):
        # One program per row.
        row = tl.program_id(0)
        if row >= N:
            return

        offs_o = tl.arange(0, D_OUT)

        # Compute the full linear output for this row.
        acc = tl.zeros([D_OUT], dtype=tl.float32)
        for k_start in range(0, D_IN, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < D_IN
            x_ptrs = X + row * sx_n + offs_k * sx_k
            x = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # W is [D_OUT, D_IN]; load a tile [D_OUT, BLOCK_K].
            w_ptrs = (
                W
                + offs_o[:, None] * sw_o
                + offs_k[None, :] * sw_k
            )
            w = tl.load(
                w_ptrs,
                mask=k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # acc[:] += w[:, k_block] @ x[k_block]
            acc += tl.sum(w * x[None, :], axis=1)

        if HAS_BIAS:
            bias = tl.load(BIAS + offs_o).to(tl.float32)
            acc += bias

        # ---- Normalization ----
        if NORM_TYPE == 0:
            # LayerNorm: subtract mean, divide by sqrt(var + eps).
            mean = tl.sum(acc, axis=0) / D_OUT
            centered = acc - mean
            var = tl.sum(centered * centered, axis=0) / D_OUT
            rstd = 1.0 / tl.sqrt(var + eps)
            normalized = centered * rstd
        else:
            # RMSNorm: divide by RMS.
            mean_sq = tl.sum(acc * acc, axis=0) / D_OUT
            rstd = 1.0 / tl.sqrt(mean_sq + eps)
            normalized = acc * rstd

        if HAS_AFFINE:
            gamma = tl.load(GAMMA + offs_o).to(tl.float32)
            normalized = normalized * gamma
            if NORM_TYPE == 0:
                beta = tl.load(BETA + offs_o).to(tl.float32)
                normalized = normalized + beta

        y_ptrs = Y + row * sy_n + offs_o * sy_o
        tl.store(y_ptrs, normalized.to(Y.dtype.element_ty))


def triton_fused_linear_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    norm_type: str = "layer",
) -> torch.Tensor:
    """Fused linear + (Layer | RMS) norm.

    Args:
        x: [N, D_in] or [..., D_in]; flattened to 2-D internally.
        weight: [D_out, D_in].
        bias: optional [D_out].
        gamma, beta: optional [D_out] affine params. Beta is unused for RMSNorm.
        eps: numerical stability.
        norm_type: 'layer' or 'rms'.
    """
    if triton is None:
        raise RuntimeError("triton not importable")
    orig_shape = x.shape
    D_in = x.shape[-1]
    x_2d = x.reshape(-1, D_in).contiguous()
    N = x_2d.shape[0]
    D_out = weight.shape[0]

    y = torch.empty((N, D_out), dtype=x.dtype, device=x.device)

    has_bias = bias is not None
    has_affine = gamma is not None
    norm_type_int = 0 if norm_type.lower() == "layer" else 1

    if not has_bias:
        bias = x_2d  # dummy pointer, never read
    if not has_affine:
        gamma = x_2d
        beta = x_2d
    elif beta is None:
        beta = gamma  # never read for RMS

    _fused_linear_norm_kernel[(N,)](
        x_2d, weight, bias, gamma, beta, y,
        float(eps),
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        N, D_in, D_out,
        HAS_BIAS=has_bias,
        NORM_TYPE=norm_type_int,
        HAS_AFFINE=has_affine,
    )

    return y.reshape(*orig_shape[:-1], D_out)


def torch_reference_fused_linear_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    norm_type: str = "layer",
) -> torch.Tensor:
    """Reference path: torch.nn.functional ops chained.

    Used for parity tests and the 'torch' backend.
    """
    import torch.nn.functional as F
    y = F.linear(x, weight, bias)
    D_out = weight.shape[0]
    if norm_type.lower() == "layer":
        return F.layer_norm(
            y, (D_out,),
            weight=gamma,
            bias=beta,
            eps=eps,
        )
    # RMSNorm
    rms = y.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    out = (y.float() * rms).to(y.dtype)
    if gamma is not None:
        out = out * gamma
    return out


__all__ = [
    "triton_fused_linear_norm",
    "torch_reference_fused_linear_norm",
]
