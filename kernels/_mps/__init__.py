# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Apple Neural Engine / MPS port.

Triton has no Apple backend. The two practical Apple paths are:

1. **MPS** (Metal Performance Shaders) - reachable from PyTorch via the
   ``mps`` device. Reasonable performance on M-series GPUs; everything is
   shaders.
2. **Core ML / ANE** - the Apple Neural Engine. Reached via the ``coremltools``
   compiler producing an .mlpackage that runs on the Neural Engine block.
   No torch hook; you compile a model graph offline.

For the v1 cross-vendor proof we port the softmax kernel, since it's the
smallest standalone kernel that exercises the Reflex Cloud deterministic
mode contract (numerically stable, dtype-aware) and verifies the wrapper
selection logic works without a Triton path.

Two implementations:

* :func:`mps_softmax` - uses ``torch.nn.functional.softmax`` on an MPS
  tensor. We treat this as our "MPS-native" path because torch's softmax
  is already implemented as a Metal kernel; we don't gain by writing our
  own metal shader unless we're fusing.
* :func:`coreml_softmax` - compiles a tiny Core ML model on first call and
  runs it via ``coremltools``. This is the ANE-resident path. We expose it
  as a benchmark-only target because the per-call dispatch overhead makes
  it impractical for fine-grained kernel use.

Both are guarded behind ``has_mps()`` / ``has_coreml()`` so the module
imports cleanly on non-Apple hosts.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch


@functools.lru_cache(maxsize=1)
def has_mps() -> bool:
    """True if the running torch has working MPS support."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


@functools.lru_cache(maxsize=1)
def has_coreml() -> bool:
    """True if coremltools is importable (deferred until needed)."""
    try:
        import coremltools  # noqa: F401
        return True
    except ImportError:
        return False


def mps_softmax(
    x: torch.Tensor,
    *,
    is_causal: bool = False,
    log: bool = False,
) -> torch.Tensor:
    """Run softmax on an MPS tensor.

    The input is moved to ``mps`` if it's not there yet, then back to the
    original device on return. For benchmark fidelity callers should already
    have an MPS tensor.
    """
    if not has_mps():
        raise RuntimeError("MPS not available on this platform")
    orig_device = x.device
    x_mps = x.to("mps") if orig_device.type != "mps" else x
    if is_causal:
        N = x_mps.shape[-1]
        mask = torch.triu(
            torch.full((N, N), float("-inf"), device="mps", dtype=x_mps.dtype),
            diagonal=1,
        )
        x_mps = x_mps + mask
    if log:
        y = torch.nn.functional.log_softmax(x_mps, dim=-1)
    else:
        y = torch.nn.functional.softmax(x_mps, dim=-1)
    return y.to(orig_device)


_coreml_cache = {}


def coreml_softmax(
    x: torch.Tensor,
    *,
    log: bool = False,
) -> torch.Tensor:
    """Compile-and-run a tiny softmax Core ML model on the Neural Engine.

    The model is cached by (shape, dtype) tuple because Core ML's compile
    step is expensive (seconds). For benchmark use only.

    Note: the ANE prefers fp16 and reshapes the input to [1, C, 1, 1] for
    its softmax op; we follow that convention. Inputs are flattened along
    all but the softmax axis, run through CoreML, then reshaped back.
    """
    if not has_coreml():
        raise RuntimeError(
            "coremltools not installed; pip install coremltools to use the "
            "Apple Neural Engine path"
        )
    import numpy as np
    import coremltools as ct

    orig_shape = x.shape
    N = orig_shape[-1]
    flat = x.reshape(-1, N).detach().cpu().numpy().astype(np.float16)

    key = (N, "log" if log else "softmax")
    if key not in _coreml_cache:
        # Build a single-op Core ML program.
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil import Program, Function

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(ct.RangeDim(1, 65536), N), dtype=np.float16),
            ]
        )
        def _prog(x):
            if log:
                # log_softmax = log(softmax)
                sm = mb.softmax(x=x, axis=-1)
                return mb.log(x=sm)
            return mb.softmax(x=x, axis=-1)

        model = ct.convert(
            _prog,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        _coreml_cache[key] = model

    model = _coreml_cache[key]
    out = model.predict({"x": flat})
    # CoreML output dict keys vary; pick the only entry.
    arr = next(iter(out.values()))
    y = torch.from_numpy(arr).to(x.device).to(x.dtype)
    return y.reshape(orig_shape)


__all__ = [
    "has_mps",
    "has_coreml",
    "mps_softmax",
    "coreml_softmax",
]
