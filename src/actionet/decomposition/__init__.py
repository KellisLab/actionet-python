"""Decomposition of expression matrices.

Two public modules mirroring libactionet's ``decomposition/``:

- :mod:`svd` — :func:`run_svd` (truncated SVD dispatcher).
- :mod:`kernel` — :func:`reduce_kernel`, :func:`reduce_kernel_from_svd`,
  :func:`smooth_kernel` (ACTIONet kernel matrix construction).
"""

from .kernel import reduce_kernel, reduce_kernel_from_svd, smooth_kernel
from .svd import run_svd

__all__ = [
    "reduce_kernel",
    "reduce_kernel_from_svd",
    "run_svd",
    "smooth_kernel",
]
