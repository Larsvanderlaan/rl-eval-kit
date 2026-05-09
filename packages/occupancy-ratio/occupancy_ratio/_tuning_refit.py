"""Final-refit and fallback guardrail helpers for tuning."""

from occupancy_ratio._tuning_impl import (
    _build_configs,
    _final_refit_penalty,
    _final_weight_metrics,
    _is_baseline_candidate,
    _refit_seed,
    _select_refit_candidate,
    _should_fallback_to_baseline,
    _weak_moment_instability_fallback,
)

__all__ = [
    "_build_configs",
    "_final_refit_penalty",
    "_final_weight_metrics",
    "_is_baseline_candidate",
    "_refit_seed",
    "_select_refit_candidate",
    "_should_fallback_to_baseline",
    "_weak_moment_instability_fallback",
]
