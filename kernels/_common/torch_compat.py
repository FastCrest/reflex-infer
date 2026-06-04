# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Make reflex_infer kernels behave like ``torch.nn.functional``.

Three goals:

* Drop-in replacement for ``F.scaled_dot_product_attention`` and friends so
  the model definition does not change when a user opts into reflex_infer.
* Autograd support via thin :class:`torch.autograd.Function` subclasses (the
  backward pass falls through to the torch reference for now; the production
  forward path is what matters for inference).
* ``torch.compile`` friendliness: every wrapper registers itself as a custom
  op via ``torch.library.custom_op`` when available, so dynamo can trace
  through reflex_infer calls without graph breaks.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch


# torch.library.custom_op was stabilized in 2.4. We feature-detect rather
# than hard-pin so older torch (2.3) still imports this module.
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _identity_decorator(*_args, **_kwargs):
    def _wrap(fn):
        return fn
    return _wrap


def custom_op(
    qualname: str,
    *,
    mutates_args=(),
    device_types=("cuda",),
):
    """Best-effort wrapper around ``torch.library.custom_op``.

    Falls back to a no-op decorator on older torch so the kernel module still
    imports. Dynamo will graph-break on the call but eager works.
    """
    if _HAS_CUSTOM_OP:
        return torch.library.custom_op(
            qualname,
            mutates_args=mutates_args,
            device_types=device_types,
        )
    return _identity_decorator()


def autograd_passthrough(forward_fn: Callable, backward_ref: Callable):
    """Wrap a forward-only kernel with an autograd-aware Function.

    Backward delegates to ``backward_ref`` (typically the torch reference) so
    training stays correct while inference uses the fast path. We do NOT
    attempt to write a fused backward kernel in this version of the library
    because Reflex Cloud's deterministic mode is inference-only.
    """

    class _ReflexFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            ctx.save_for_backward(*[a for a in args if torch.is_tensor(a)])
            ctx.kwargs = kwargs
            return forward_fn(*args, **kwargs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            saved = ctx.saved_tensors
            with torch.enable_grad():
                detached = [t.detach().requires_grad_(True) for t in saved]
                out = backward_ref(*detached, **ctx.kwargs)
                grads = torch.autograd.grad(
                    out, detached, grad_outputs=grad_outputs[0]
                )
            # Map grads back into the original args slot order. Non-tensor
            # args take ``None`` in the output tuple.
            return (*grads, *([None] * len(ctx.kwargs)))

    @functools.wraps(forward_fn)
    def _apply(*args, **kwargs):
        if any(torch.is_tensor(a) and a.requires_grad for a in args):
            return _ReflexFn.apply(*args, **kwargs)
        return forward_fn(*args, **kwargs)

    return _apply


def assert_same_layout(*tensors: torch.Tensor) -> None:
    """Assert every tensor has the same shape and strides.

    Used by KV-cache append where the storage layout is part of the contract
    with the caller (vLLM-style pages).
    """
    if not tensors:
        return
    ref = tensors[0]
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != ref.shape:
            raise ValueError(
                f"tensor {i} shape mismatch: {t.shape} vs {ref.shape}"
            )
        if t.stride() != ref.stride():
            raise ValueError(
                f"tensor {i} stride mismatch: {t.stride()} vs {ref.stride()}"
            )


def maybe_view_as_4d(
    x: torch.Tensor, batch: int, heads: int, seq: int, dim: int
) -> torch.Tensor:
    """Reshape into [B, H, S, D] regardless of input layout.

    Accepts [B, H, S, D] (passthrough), [B, S, H, D] (transpose), and
    [B*S, H, D] (insert seq). Kept as a small helper because every attention
    kernel re-derives this and we want one place to fix it.
    """
    if x.dim() == 4 and x.shape == (batch, heads, seq, dim):
        return x
    if x.dim() == 4 and x.shape == (batch, seq, heads, dim):
        return x.transpose(1, 2).contiguous()
    if x.dim() == 3 and x.shape == (batch * seq, heads, dim):
        return x.view(batch, seq, heads, dim).transpose(1, 2).contiguous()
    raise ValueError(
        f"cannot view shape {tuple(x.shape)} as [B={batch}, H={heads}, "
        f"S={seq}, D={dim}]"
    )
