# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FastCrest
"""reflex_infer.kernels

Cross-vendor production kernel library used by Reflex Cloud deterministic
mode and (eventually) Reflex Compile. Triton is the primary path because it
covers NVIDIA + AMD with one source. CUDA C++ implementations live alongside
as reference and as a fallback for older GPUs without Triton support.

Public surface:
    from reflex_infer.kernels.attention import fused_attention
    from reflex_infer.kernels.kv_cache import kv_paged_append, kv_paged_lookup
    from reflex_infer.kernels.fused_linear_norm import fused_linear_layernorm
    from reflex_infer.kernels.softmax import online_softmax
    from reflex_infer.kernels.rope import apply_rope, apply_rope_

Every kernel routes through ``_common.launch.select_backend`` which honours
the ``REFLEX_INFER_BACKEND`` environment variable (``triton`` / ``cuda`` /
``torch``) for deterministic benchmarking and CI.
"""

from . import _common  # noqa: F401
