"""Public tuning API facade.

Implementation details live in ``_tuning_impl`` and are grouped for
contributors through ``_tuning_candidates``, ``_tuning_cv``, ``_tuning_refit``,
and ``_tuning_scoring``. This module preserves the stable public import path.
"""

from __future__ import annotations

from functools import wraps

from . import _tuning_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

for _name in (
    "CandidateResult",
    "FoldResult",
    "OccupancySearchSpace",
    "OccupancyTargetValidationCandidateResult",
    "OccupancyTargetValidationResult",
    "OccupancyTuningConfig",
    "OccupancyTuningResult",
    "StagedCVCandidateRow",
    "StagedCVFoldRow",
    "StagedCVResult",
):
    if _name in globals():
        globals()[_name].__module__ = __name__

__all__ = list(getattr(_impl, "__all__", []))


def _sync_patchable_backend_globals() -> None:
    for _backend_name in (
        "fit_discounted_occupancy_ratio",
        "fit_discounted_occupancy_ratio_neural",
        "fit_google_dualdice_occupancy_ratio",
    ):
        if _backend_name in globals():
            setattr(_impl, _backend_name, globals()[_backend_name])


@wraps(_impl.tune_occupancy_ratio_auto)
def tune_occupancy_ratio_auto(*args, **kwargs):
    _sync_patchable_backend_globals()
    return _impl.tune_occupancy_ratio_auto(*args, **kwargs)


@wraps(_impl.tune_occupancy_ratio)
def tune_occupancy_ratio(*args, **kwargs):
    _sync_patchable_backend_globals()
    return _impl.tune_occupancy_ratio(*args, **kwargs)


@wraps(_impl.tune_occupancy_ratio_with_target_validation)
def tune_occupancy_ratio_with_target_validation(*args, **kwargs):
    _sync_patchable_backend_globals()
    return _impl.tune_occupancy_ratio_with_target_validation(*args, **kwargs)


del _name
