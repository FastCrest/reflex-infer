# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""Shared utilities for reflex_infer kernels.

Three concerns live here:

* ``launch``  - backend selection (Triton vs CUDA C++ vs torch reference),
                 capability probing, deterministic seeding for grids.
* ``dtype``   - dtype conversion / cast-rule helpers spanning fp32/bf16/fp16
                 and the experimental fp8 (e4m3 / e5m2) tiers.
* ``torch_compat`` - thin wrappers that make our kernels behave like
                      ``torch.nn.functional`` (autograd-friendly Function
                      subclasses, ``__torch_dispatch__`` hooks).
"""

from . import dtype, launch, torch_compat  # noqa: F401
