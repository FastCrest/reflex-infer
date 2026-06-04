# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""ROCm port notes for reflex_infer.

Triton 3.x supports AMD MI200 / MI300 via its hip backend, so the existing
Triton kernels in ``kernels/attention``, ``kernels/kv_cache``, etc. run
unchanged on ROCm. There is no separate ROCm kernel tree.

What this subpackage holds:

* :func:`detect_rocm` - reports gfx version, LDS budget, and a recommended
  autotune cap.
* :func:`autotune_cap` - returns the BLOCK_N upper bound for the active
  ROCm device; the attention kernel reads it on import.

We picked BLOCK_N <= 128 on MI300 because LDS budget is tighter than H100;
running with BLOCK_N=256 spilled to LDS on early Triton 3.0 builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RocmCapability:
    is_rocm: bool
    gfx: int                  # e.g. 942 (MI300X) or 90a (MI250)
    device_name: str
    lds_kb: int               # Local Data Share budget in KB
    recommended_block_n: int  # Tuned cap for attention kernel


def _parse_gfx(arch_name: str, device_name: str) -> int:
    """Parse the AMD gfx number from gcnArchName or fall back to device name.

    `gcnArchName` is the canonical source: torch exposes it on
    ``torch.cuda.get_device_properties(0)`` for ROCm builds. Values look like
    ``"gfx942"``, ``"gfx90a"``, etc. We prefer it over the marketing name
    (``"AMD Instinct MI300X"``) because the marketing name doesn't include
    the gfx token at all.
    """
    candidates = (arch_name or "").lower().split() + (device_name or "").lower().split()
    for token in candidates:
        if token.startswith("gfx"):
            try:
                return int(token[3:].rstrip("acx"))
            except ValueError:
                continue
    return 0


def detect_rocm() -> Optional[RocmCapability]:
    """Return a :class:`RocmCapability` if we're running on ROCm, else None."""
    if not torch.cuda.is_available():
        return None
    is_rocm = bool(getattr(torch.version, "hip", None))
    if not is_rocm:
        return None
    name = torch.cuda.get_device_name(0)
    # Prefer gcnArchName from device properties; the marketing device name
    # (e.g. "AMD Instinct MI300X") does NOT contain a gfx token so the
    # previous parser silently fell through to gfx=0 on real hardware.
    arch_name = ""
    try:
        props = torch.cuda.get_device_properties(0)
        arch_name = getattr(props, "gcnArchName", "") or ""
    except Exception:
        arch_name = ""
    gfx = _parse_gfx(arch_name, name)

    # Hard-coded LDS budgets per gfx generation. Sourced from the AMD ISA
    # docs (MI300: 64KB per CU pair, MI250: 64KB per CU, MI100: 64KB per CU).
    lds_kb = 64
    # MI300 prefers smaller BLOCK_N because of WMMA tile shape constraints.
    if gfx >= 940:
        recommended_block_n = 128
    elif gfx >= 900:
        recommended_block_n = 64
    else:
        recommended_block_n = 32

    return RocmCapability(
        is_rocm=True,
        gfx=gfx,
        device_name=name,
        lds_kb=lds_kb,
        recommended_block_n=recommended_block_n,
    )


def autotune_cap() -> int:
    """Return the BLOCK_N upper bound for the active ROCm device.

    Used by the attention kernel's autotune list to clip configs at import
    time. On non-ROCm we return a large sentinel so all configs survive.
    """
    cap = detect_rocm()
    if cap is None:
        return 1 << 30
    return cap.recommended_block_n


__all__ = ["RocmCapability", "detect_rocm", "autotune_cap"]
