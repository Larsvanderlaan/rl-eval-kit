"""Boosted fixed-point target builders and transition-cache helpers."""

from occupancy_ratio._boosted_impl import (
    _BatchCache,
    _build_transition_caches,
    _checked_vector,
    _combine_builder_diags,
    _make_transition_reference_features,
    _prepare_source_weights,
    _source_state_ratio_summary,
    make_crossfit_forward_occupancy_dataset,
    make_direct_adjoint_occupancy_dataset,
    make_forward_occupancy_dataset,
)

__all__ = [
    "make_crossfit_forward_occupancy_dataset",
    "make_direct_adjoint_occupancy_dataset",
    "make_forward_occupancy_dataset",
    "_BatchCache",
    "_build_transition_caches",
    "_checked_vector",
    "_combine_builder_diags",
    "_make_transition_reference_features",
    "_prepare_source_weights",
    "_source_state_ratio_summary",
]
