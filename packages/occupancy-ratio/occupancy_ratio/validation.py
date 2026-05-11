"""Shared boosted estimator input validation and mode-resolution helpers."""

from occupancy_ratio._boosted_impl import (
    _as_2d,
    _optional_binary_vector,
    _prepare_target_action_samples,
    _resolve_continuation,
    _resolve_initial_ratio_mode,
    _resolve_known_action_ratio_inputs,
    _resolve_one_step_ratio_mode,
    _validate_aligned_inputs,
    _validate_base_transition_inputs,
    _validate_initial_action_inputs,
    _validate_initial_state_inputs,
    _validate_next_target_actions,
    _validate_occupancy_stabilization_config,
    _validate_ratio_prediction_config,
    _validate_target_action_rows,
)

__all__ = [
    "_as_2d",
    "_optional_binary_vector",
    "_prepare_target_action_samples",
    "_resolve_continuation",
    "_resolve_initial_ratio_mode",
    "_resolve_known_action_ratio_inputs",
    "_resolve_one_step_ratio_mode",
    "_validate_aligned_inputs",
    "_validate_base_transition_inputs",
    "_validate_initial_action_inputs",
    "_validate_initial_state_inputs",
    "_validate_next_target_actions",
    "_validate_occupancy_stabilization_config",
    "_validate_ratio_prediction_config",
    "_validate_target_action_rows",
]
