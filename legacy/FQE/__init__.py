"""Archived legacy FQE implementation.

The supported production package is ``fqe`` from ``packages/fqe``. This module
is retained only behind compatibility shims for old notebooks and scripts.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The archived `legacy.FQE` namespace is deprecated; install and import the `fqe` package instead.",
    DeprecationWarning,
    stacklevel=2,
)
