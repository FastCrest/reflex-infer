# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Backend selection and launch helpers.

reflex_infer's wrappers all delegate to :func:`select_backend` which picks
between Triton (primary), CUDA C++ (reference / fallback), and a pure-torch
reference path. The choice is driven by:

1. Environment override ``REFLEX_INFER_BACKEND=triton|cuda|torch``.
2. Per-call ``backend`` kwarg.
3. Device capability probe.

We intentionally make backend selection cheap (~microseconds) so wrappers can
call it on every invocation rather than caching a closure at import-time;
this lets benchmarks switch backends without restarting Python.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class Capability:
    """Runtime device capability snapshot.

    Used by Triton kernel selection logic to pick block sizes / num_warps that
    are valid for the active SM. ``cc`` is the CUDA compute capability as a
    flat integer (e.g. 80 for Ampere, 90 for Hopper). On ROCm we synthesize
    a pseudo-cc; see ``rocm_cc``.
    """
    has_cuda: bool
    has_rocm: bool
    cc: int            # NVIDIA compute capability, 0 if not CUDA
    rocm_cc: int       # AMD gfx number (e.g. 942 for MI300), 0 if not ROCm
    device_name: str
    has_tensor_cores: bool


@functools.lru_cache(maxsize=1)
def probe_capability() -> Capability:
    """Probe the active device once and cache the result.

    We do not invalidate the cache when the active device changes; users who
    switch devices mid-run should call :func:`probe_capability.cache_clear`.
    """
    if not torch.cuda.is_available():
        return Capability(False, False, 0, 0, "cpu", False)

    is_rocm = bool(getattr(torch.version, "hip", None))
    if is_rocm:
        # ROCm exposes gfx version via the device name; e.g. "AMD Instinct
        # MI300X" or props.gcnArchName. We don't have a direct API across all
        # torch builds, so we parse the device name conservatively.
        name = torch.cuda.get_device_name(0)
        rocm_cc = 0
        for token in name.lower().split():
            if token.startswith("gfx"):
                try:
                    rocm_cc = int(token[3:])
                except ValueError:
                    pass
        return Capability(False, True, 0, rocm_cc, name, True)

    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    return Capability(
        has_cuda=True,
        has_rocm=False,
        cc=cc,
        rocm_cc=0,
        device_name=torch.cuda.get_device_name(0),
        has_tensor_cores=cc >= 70,  # Volta and later
    )


def _env_backend() -> Optional[str]:
    val = os.environ.get("REFLEX_INFER_BACKEND")
    if val is None:
        return None
    val = val.strip().lower()
    if val not in {"triton", "cuda", "torch"}:
        raise ValueError(
            f"REFLEX_INFER_BACKEND={val!r}; expected triton|cuda|torch"
        )
    return val


def select_backend(
    requested: Optional[str] = None,
    *,
    triton_min_cc: int = 80,
    force_torch_on_cpu: bool = True,
) -> str:
    """Return the backend string ('triton', 'cuda', or 'torch') to dispatch to.

    Resolution order:

    1. Explicit ``requested`` kwarg (raises if unknown).
    2. ``REFLEX_INFER_BACKEND`` env var.
    3. Capability probe: Triton on cc>=triton_min_cc (Ampere+) or ROCm,
       CUDA fallback on older NVIDIA cards, torch reference on CPU.
    """
    if requested is not None:
        if requested not in {"triton", "cuda", "torch"}:
            raise ValueError(f"backend={requested!r}; expected triton|cuda|torch")
        return requested

    env = _env_backend()
    if env is not None:
        return env

    cap = probe_capability()
    if not cap.has_cuda and not cap.has_rocm:
        return "torch" if force_torch_on_cpu else "cuda"
    if cap.has_rocm:
        return "triton"  # Triton 3.x supports MI200/MI300
    if cap.cc >= triton_min_cc:
        return "triton"
    return "cuda"


def has_triton() -> bool:
    """Cheap test for whether triton is importable. Cached at first call."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def maybe_triton():
    """Return the triton module if available, else None. Tests use this to
    skip when triton isn't installed (CI without GPU)."""
    try:
        import triton
        return triton
    except ImportError:
        return None


def deterministic_seed(seed: int = 0) -> None:
    """Seed every RNG reflex_infer touches.

    Used by the benchmark harness so two runs of the same kernel return the
    same numerical result (modulo accumulator order). We do NOT enable
    ``torch.use_deterministic_algorithms`` here because some kernels we
    measure rely on non-deterministic atomic adds; that toggle lives in the
    benchmark harness itself.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_contiguous(*tensors: torch.Tensor) -> tuple:
    """Force every input contiguous; Triton kernels assume row-major strides."""
    return tuple(t.contiguous() if not t.is_contiguous() else t for t in tensors)


def wrap_kernel(
    triton_fn: Callable,
    cuda_fn: Optional[Callable],
    torch_fn: Callable,
) -> Callable:
    """Return a callable that selects the backend at call time.

    Kept as a small helper so each kernel module's ``__init__.py`` reads as a
    pure dispatch table.
    """

    def _dispatch(*args, backend: Optional[str] = None, **kwargs):
        choice = select_backend(backend)
        if choice == "triton":
            return triton_fn(*args, **kwargs)
        if choice == "cuda" and cuda_fn is not None:
            return cuda_fn(*args, **kwargs)
        return torch_fn(*args, **kwargs)

    _dispatch.triton = triton_fn
    _dispatch.cuda = cuda_fn
    _dispatch.torch = torch_fn
    return _dispatch
