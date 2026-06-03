# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Triton RoPE (Rotary Position Embedding) kernel.

RoPE rotates pairs of dimensions of the Q/K vectors by an angle derived
from the position. Two common conventions:

* **interleaved** (HuggingFace Llama / GPT-NeoX): pairs are
  (x[2i], x[2i+1]) and the rotation is applied to consecutive elements.
* **half-rotated** (Meta Llama, Mistral): the head_dim is split into two
  halves; rotation pairs (x[i], x[i + D/2]).

We support both. The kernel takes precomputed cos/sin tables (so callers
that share these across layers don't recompute them).

Two variants:

* ``apply_rope``: returns a new tensor.
* ``apply_rope_``: in-place; saves an allocation when the Q/K tensors are
  already owned by the caller (common in the projection -> attention path).

Shapes:
* ``x``: [..., seq, num_heads, head_dim] OR [..., num_heads, seq, head_dim]
* ``cos``, ``sin``: [seq, head_dim] (or [head_dim] if you broadcast yourself)

We support both layouts via the ``layout`` argument because different
codebases pick differently and we don't want to force a transpose.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._common.launch import maybe_triton

triton = maybe_triton()

if triton is not None:
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        X, Y, COS, SIN,
        # Strides for X (assumed contiguous along head_dim).
        sx_b, sx_s, sx_h, sx_d,
        sy_b, sy_s, sy_h, sy_d,
        sc_s, sc_d,
        # Sizes.
        B, S, H,
        D: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # One program per (batch, seq, head_tile).
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        offs_d = tl.arange(0, D)

        # Load cos / sin for this position. Same for all heads in the head
        # tile, so we load once and broadcast.
        cos = tl.load(COS + pid_s * sc_s + offs_d * sc_d).to(tl.float32)
        sin = tl.load(SIN + pid_s * sc_s + offs_d * sc_d).to(tl.float32)

        # Load X tile.
        x_ptrs = (
            X
            + pid_b * sx_b
            + pid_s * sx_s
            + offs_h[:, None] * sx_h
            + offs_d[None, :] * sx_d
        )
        x = tl.load(x_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

        # Construct the "rotated" half. The semantics depend on the layout.
        if INTERLEAVED:
            # x_rot[2i]   = -x[2i+1]
            # x_rot[2i+1] =  x[2i]
            # We achieve this by permuting along D in pairs. Implement via
            # gather with a custom index pattern.
            # idx_even = 2 * (offs_d // 2)
            # idx_odd  = idx_even + 1
            # negated_partner[2i]   = -x[2i+1]
            # negated_partner[2i+1] =  x[2i]
            pair = offs_d // 2
            is_even = (offs_d % 2) == 0
            partner_idx = tl.where(is_even, 2 * pair + 1, 2 * pair)
            partner_ptrs = (
                X
                + pid_b * sx_b
                + pid_s * sx_s
                + offs_h[:, None] * sx_h
                + partner_idx[None, :] * sx_d
            )
            partner = tl.load(
                partner_ptrs, mask=h_mask[:, None], other=0.0
            ).to(tl.float32)
            sign = tl.where(is_even, -1.0, 1.0)
            x_rot = sign[None, :] * partner
        else:
            # Half-rotated: pair (i, i + D/2)
            half = D // 2
            in_first_half = offs_d < half
            partner_idx = tl.where(
                in_first_half, offs_d + half, offs_d - half
            )
            partner_ptrs = (
                X
                + pid_b * sx_b
                + pid_s * sx_s
                + offs_h[:, None] * sx_h
                + partner_idx[None, :] * sx_d
            )
            partner = tl.load(
                partner_ptrs, mask=h_mask[:, None], other=0.0
            ).to(tl.float32)
            # x_rot[i]         = -x[i + D/2]   for i < D/2
            # x_rot[i + D/2]   =  x[i]         for i >= D/2
            sign = tl.where(in_first_half, -1.0, 1.0)
            x_rot = sign[None, :] * partner

        y = x * cos[None, :] + x_rot * sin[None, :]

        y_ptrs = (
            Y
            + pid_b * sy_b
            + pid_s * sy_s
            + offs_h[:, None] * sy_h
            + offs_d[None, :] * sy_d
        )
        tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=h_mask[:, None])


def _resolve_layout(x: torch.Tensor, layout: str) -> Tuple[int, int, int, int]:
    """Return (B, S, H, D) extracting from the tensor based on layout."""
    if layout == "bshd":
        B, S, H, D = x.shape
    elif layout == "bhsd":
        B, H, S, D = x.shape
    else:
        raise ValueError(f"unknown layout {layout!r}; want 'bshd' or 'bhsd'")
    return B, S, H, D


def triton_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    layout: str = "bshd",
    inplace: bool = False,
) -> torch.Tensor:
    """Apply RoPE to ``x``.

    Args:
        x: [B, S, H, D] (bshd) or [B, H, S, D] (bhsd).
        cos, sin: [S, D] precomputed tables.
        interleaved: True for GPT-NeoX style pair (2i, 2i+1); False for
            Llama-half style (i, i + D/2).
        layout: 'bshd' or 'bhsd'.
        inplace: if True, writes Y over X (saves an allocation).
    """
    if triton is None:
        raise RuntimeError("triton not importable")
    B, S, H, D = _resolve_layout(x, layout)
    x = x.contiguous()
    y = x if inplace else torch.empty_like(x)

    sx = x.stride()
    sy = y.stride()
    if layout == "bshd":
        sx_b, sx_s, sx_h, sx_d = sx
        sy_b, sy_s, sy_h, sy_d = sy
    else:  # bhsd
        sx_b, sx_h, sx_s, sx_d = sx
        sy_b, sy_h, sy_s, sy_d = sy

    sc_s, sc_d = cos.stride()
    BLOCK_H = 4
    grid = (B, S, triton.cdiv(H, BLOCK_H))
    _rope_kernel[grid](
        x, y, cos, sin,
        sx_b, sx_s, sx_h, sx_d,
        sy_b, sy_s, sy_h, sy_d,
        sc_s, sc_d,
        B, S, H, D,
        INTERLEAVED=interleaved,
        BLOCK_H=BLOCK_H,
    )
    return y


def torch_reference_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
    layout: str = "bshd",
) -> torch.Tensor:
    """Reference RoPE in pure torch.

    Used for parity tests and the 'torch' backend.
    """
    B, S, H, D = _resolve_layout(x, layout)
    if layout == "bhsd":
        # Reshape to [B, S, H, D] for the math then transpose back.
        x = x.transpose(1, 2)

    cos_b = cos.view(1, S, 1, D)
    sin_b = sin.view(1, S, 1, D)

    if interleaved:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_e = cos_b[..., 0::2]
        sin_e = sin_b[..., 0::2]
        # For interleaved RoPE, paired rotation:
        #   y_even = x_even * cos - x_odd * sin
        #   y_odd  = x_even * sin + x_odd * cos
        y_even = x_even * cos_e - x_odd * sin_e
        y_odd = x_even * sin_e + x_odd * cos_e
        y = torch.stack((y_even, y_odd), dim=-1).flatten(-2)
    else:
        half = D // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos_1 = cos_b[..., :half]
        sin_1 = sin_b[..., :half]
        # y_first  = x1 * cos - x2 * sin
        # y_second = x1 * sin + x2 * cos
        y_first = x1 * cos_1 - x2 * sin_1
        y_second = x1 * sin_1 + x2 * cos_1
        y = torch.cat((y_first, y_second), dim=-1)

    if layout == "bhsd":
        y = y.transpose(1, 2).contiguous()
    return y


def build_rope_tables(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the standard cos/sin tables.

    For the half-rotated convention, the first half of head_dim holds the
    cosine factor for each (pair-index, position) and the second half holds
    the same value (so the kernel can broadcast cleanly). For the
    interleaved convention, cos[2i] = cos[2i+1] = cos(position * theta_i).
    This matches the HuggingFace convention.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)  # [seq, half]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    if interleaved:
        # Duplicate each element so cos[2i] = cos[2i+1].
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    else:
        # Concat: first half and second half are identical.
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    return cos.to(dtype), sin.to(dtype)


__all__ = [
    "triton_apply_rope",
    "torch_reference_rope",
    "build_rope_tables",
]
