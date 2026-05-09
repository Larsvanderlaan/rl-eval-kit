"""Backward-compatible boosted FORI API.

The boosted implementation is organized behind focused modules such as
``configs``, ``models``, ``validation``, ``nuisance_lgbm``, ``targets``, and
``stabilization``. This module remains as a compatibility facade for existing
imports, including tests and downstream research code that use private helpers.
"""

from __future__ import annotations

from . import _boosted_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

for _name in (
    "ActionRatioConfig",
    "SourceStateRatioConfig",
    "TransitionRatioConfig",
    "OccupancyRegressionConfig",
    "DiscountedOccupancyRatioModel",
):
    if _name in globals():
        globals()[_name].__module__ = __name__

__all__ = list(getattr(_impl, "__all__", []))

del _name, _impl
