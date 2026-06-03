# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Public surface for paged KV cache operations.

Three entrypoints mirror vLLM's pattern:

* ``kv_paged_append``  : write new K/V vectors into the paged cache.
* ``kv_paged_lookup``  : gather K/V for a batch of sequences into dense
                          tensors (for use with the attention kernel).
* ``kv_paged_scatter`` : overwrite specific slots (speculative rollback).

The kernels mutate ``k_cache`` / ``v_cache`` in place to avoid copies.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._common.launch import select_backend
from .triton_kv_paged import (
    torch_reference_kv_paged_append,
    torch_reference_kv_paged_lookup,
    triton_kv_paged_append,
    triton_kv_paged_lookup,
    triton_kv_paged_scatter,
)


def kv_paged_append(
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    backend: Optional[str] = None,
) -> None:
    """Append new K, V into the paged KV cache (in place).

    See ``triton_kv_paged_append`` for shape details. ``slot_mapping`` must
    be int64 and on the same device as the cache.
    """
    choice = select_backend(backend)
    if choice == "triton":
        triton_kv_paged_append(k_in, v_in, k_cache, v_cache, slot_mapping)
    else:
        torch_reference_kv_paged_append(
            k_in, v_in, k_cache, v_cache, slot_mapping
        )


def kv_paged_lookup(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_ctx: int,
    *,
    backend: Optional[str] = None,
) -> tuple:
    """Gather K, V from the paged cache into dense tensors."""
    choice = select_backend(backend)
    if choice == "triton":
        return triton_kv_paged_lookup(
            k_cache, v_cache, block_table, context_lens, max_ctx
        )
    return torch_reference_kv_paged_lookup(
        k_cache, v_cache, block_table, context_lens, max_ctx
    )


def kv_paged_scatter(
    k_updates: torch.Tensor,
    v_updates: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    backend: Optional[str] = None,
) -> None:
    """Overwrite specific slots (speculative decoding rollback)."""
    choice = select_backend(backend)
    if choice == "triton":
        triton_kv_paged_scatter(k_updates, v_updates, k_cache, v_cache, slot_mapping)
    else:
        torch_reference_kv_paged_append(
            k_updates, v_updates, k_cache, v_cache, slot_mapping
        )


__all__ = [
    "kv_paged_append",
    "kv_paged_lookup",
    "kv_paged_scatter",
]
