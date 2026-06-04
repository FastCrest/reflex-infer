# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for reflex_infer kernel tests.

We tag GPU-required tests with ``@pytest.mark.gpu`` and skip them
automatically when CUDA isn't available, so CI without a GPU can still run
the import-time and CPU-fallback paths.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA-capable GPU to run",
    )
    config.addinivalue_line(
        "markers",
        "triton: requires triton importable",
    )
