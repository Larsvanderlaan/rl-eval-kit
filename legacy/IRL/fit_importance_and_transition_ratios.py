from __future__ import annotations

from importlib import import_module

from IRL._occupancy_ratio_compat import ensure_occupancy_ratio_path

ensure_occupancy_ratio_path()
_module = import_module("occupancy_ratio.fit_importance_and_transition_ratios")

globals().update(
    {
        name: value
        for name, value in _module.__dict__.items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
