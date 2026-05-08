"""Deprecated compatibility namespace for legacy FQE modules.

New code should import from the production package under ``packages/fqe``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "The `FQE` namespace is deprecated; use the `fqe` package from `packages/fqe` instead.",
    DeprecationWarning,
    stacklevel=2,
)
