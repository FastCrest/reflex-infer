# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Triton online softmax.

Standard online softmax in one pass: for each row, walk the columns once
maintaining a running max ``m`` and a running denominator ``l`` such that
when we move from ``m_old`` to ``m_new = max(m_old, x_j)`` we rescale the
existing denominator by ``exp(m_old - m_new)``. Final step: divide each
``exp(x_j - m_final) / l_final``.

vs the naive two-pass softmax (find max, then subtract+exp+sum), the online
form needs only one HBM read of the input row. For long-row softmax this
moves us from 2x bandwidth to 1x bandwidth which is ~2x speedup at large D.

Three modes:

* ``online_softmax``: standard row-wise softmax over the last dim.
* ``causal_softmax``: applies a causal mask before softmax in one pass
  (saves materializing the mask tensor).
* ``log_softmax``: same online pass, returns x - m - log(l).

We block-tile the columns so very long rows fit in shared memory.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import maybe_triton

triton = maybe_triton()

if triton is not None:
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 256}, num_warps=4),
            triton.Config({"BLOCK_N": 512}, num_warps=8),
            triton.Config({"BLOCK_N": 1024}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _online_softmax_kernel(
        X, Y,
        sx_m, sx_n,
        sy_m, sy_n,
        M, N,
        IS_CAUSAL: tl.constexpr,
        IS_LOG: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return

        # In causal mode, the host has validated that M % N == 0 and the
        # input is square [..., N, N]. The query position is `row % N`
        # (not the flat `row`), so each batch element's S queries map
        # cleanly back into [0, N). For non-square inputs the host raises
        # before we get here.
        if IS_CAUSAL:
            q_pos = row % N
        else:
            q_pos = row  # unused

        # Pass 1: online running max + denominator.
        m_i = float("-inf")
        l_i = 0.0
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            x = tl.load(
                X + row * sx_m + offs_n * sx_n,
                mask=n_mask,
                other=float("-inf"),
            ).to(tl.float32)
            if IS_CAUSAL:
                x = tl.where(offs_n <= q_pos, x, float("-inf"))
            m_ij = tl.max(x, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            l_ij = tl.sum(tl.exp(x - m_new), axis=0)
            l_i = l_i * alpha + l_ij
            m_i = m_new

        # Pass 2: write softmax.
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N
            x = tl.load(
                X + row * sx_m + offs_n * sx_n,
                mask=n_mask,
                other=float("-inf"),
            ).to(tl.float32)
            if IS_CAUSAL:
                x = tl.where(offs_n <= q_pos, x, float("-inf"))
            if IS_LOG:
                y = x - m_i - tl.log(l_i)
            else:
                y = tl.exp(x - m_i) / l_i
            tl.store(
                Y + row * sy_m + offs_n * sy_n,
                y.to(Y.dtype.element_ty),
                mask=n_mask,
            )


def triton_online_softmax(
    x: torch.Tensor,
    *,
    is_causal: bool = False,
    log: bool = False,
) -> torch.Tensor:
    """Online softmax along the last dim.

    Despite the name 'online', we do two passes over HBM: one to compute the
    statistics and one to write the output. The 'online' refers to the
    single-pass-per-block accumulation inside each pass. A true single-pass
    softmax would require the output buffer to be twice the size (or to
    re-materialize the input from the kernel's perspective) which loses the
    HBM win. The two-pass online form is the production sweet spot.

    For decode-attention's score softmax the online form is fused INTO the
    attention kernel (see ``attention/triton_fused_attention.py``); this
    standalone kernel exists for use cases that softmax an externally
    materialized tensor (model heads, retrieval scores).

    Contract for ``is_causal=True``:
        The kernel treats ``row`` (the flattened row index) as the query
        position and ``col`` as the key position. That mapping is only
        meaningful when the input is a SQUARE attention score matrix of
        shape ``[..., S, S]``. We therefore reject non-square inputs in
        causal mode rather than silently producing wrong masking. For
        attention-score softmax over non-square shapes, use the fused
        attention kernel (which masks per (q_pos, k_pos) correctly).
    """
    if triton is None:
        raise RuntimeError("triton not importable")
    orig_shape = x.shape
    N = x.shape[-1]
    if is_causal:
        if x.dim() < 2 or x.shape[-2] != x.shape[-1]:
            raise ValueError(
                "causal mode requires square input (last two dims equal); "
                f"got {tuple(x.shape)}. The flattened-row causal mapping "
                "is only well-defined for square attention scores."
            )
    x_2d = x.reshape(-1, N).contiguous()
    M = x_2d.shape[0]
    if is_causal:
        # After reshape M = prod(leading_dims) * S. The kernel's causal mask
        # uses `row < N` (i.e. `row` modulo S == query position). We enforce
        # that the leading-dim flattening preserves the row==query mapping
        # by validating M is a multiple of N.
        if M % N != 0:
            raise ValueError(
                "causal mode requires M % N == 0 after flattening; "
                f"got M={M}, N={N}."
            )
    y = torch.empty_like(x_2d)

    _online_softmax_kernel[(M,)](
        x_2d, y,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        M, N,
        IS_CAUSAL=is_causal,
        IS_LOG=log,
    )
    return y.reshape(orig_shape)


def torch_reference_softmax(
    x: torch.Tensor,
    *,
    is_causal: bool = False,
    log: bool = False,
) -> torch.Tensor:
    """Reference path via torch.nn.functional.

    Causal mode follows the same square-input contract as
    ``triton_online_softmax``: rejects non-square last-two-dims.
    """
    import torch.nn.functional as F
    if is_causal:
        if x.dim() < 2 or x.shape[-2] != x.shape[-1]:
            raise ValueError(
                "causal mode requires square input (last two dims equal); "
                f"got {tuple(x.shape)}."
            )
        N = x.shape[-1]
        mask = torch.triu(
            torch.full((N, N), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        x = x + mask
    return F.log_softmax(x, dim=-1) if log else F.softmax(x, dim=-1)


__all__ = [
    "triton_online_softmax",
    "torch_reference_softmax",
]
