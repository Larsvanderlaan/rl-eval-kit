"""Backward-compatible neural FORI API.

The neural implementation is organized behind focused modules such as
``neural_configs``, ``neural_models``, ``neural_nuisance``, ``neural_targets``,
and ``neural_fit``. This module remains as a compatibility facade for existing
imports, including tests and downstream research code that use private helpers.
"""

from __future__ import annotations

from functools import wraps

from . import _neural_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

for _name in (
    "NeuralActionRatioConfig",
    "NeuralSourceStateRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralDiscountedOccupancyRatioModel",
):
    if _name in globals():
        globals()[_name].__module__ = __name__

__all__ = list(getattr(_impl, "__all__", []))


def _sync_optional_backend_globals() -> None:
    _impl.torch = globals().get("torch", _impl.torch)
    _impl.nn = globals().get("nn", _impl.nn)


@wraps(_impl.fit_action_ratio_neural)
def fit_action_ratio_neural(*args, **kwargs):
    _sync_optional_backend_globals()
    return _impl.fit_action_ratio_neural(*args, **kwargs)


@wraps(_impl.fit_source_state_ratio_neural)
def fit_source_state_ratio_neural(*args, **kwargs):
    _sync_optional_backend_globals()
    return _impl.fit_source_state_ratio_neural(*args, **kwargs)


@wraps(_impl.fit_transition_ratio_neural)
def fit_transition_ratio_neural(*args, **kwargs):
    _sync_optional_backend_globals()
    return _impl.fit_transition_ratio_neural(*args, **kwargs)


@wraps(_impl.fit_discounted_occupancy_ratio_neural)
def fit_discounted_occupancy_ratio_neural(*args, **kwargs):
    _sync_optional_backend_globals()
    return _impl.fit_discounted_occupancy_ratio_neural(*args, **kwargs)


@wraps(_impl.tune_discounted_occupancy_ratio_neural_cv)
def tune_discounted_occupancy_ratio_neural_cv(*args, **kwargs):
    _sync_optional_backend_globals()
    return _impl.tune_discounted_occupancy_ratio_neural_cv(*args, **kwargs)


del _name
