from __future__ import annotations

from causal_ope_benchmark.types import EstimatorInformationContract


COMMON_VISIBLE_ARRAYS = (
    "unit_id",
    "time",
    "states",
    "actions",
    "rewards",
    "next_states",
    "terminals",
    "action_available",
    "missingness_mask",
    "censoring",
    "behavior_propensity",
    "target_actions",
    "target_propensity",
    "target_propensity_observed_action",
    "next_target_actions",
    "target_action_probabilities",
    "next_target_action_probabilities",
    "initial_states",
    "initial_actions",
    "initial_action_probabilities",
    "action_dose",
    "target_action_dose",
    "dose_available",
)


def contract_for_family(family: str) -> EstimatorInformationContract:
    """Return the public estimator information contract for a family."""
    if family == "streamlift":
        return EstimatorInformationContract(
            family="streamlift",
            visible_arrays=COMMON_VISIBLE_ARRAYS,
            visible_metadata=(
                "family",
                "scenario",
                "sample_size",
                "gamma",
                "observed_horizon",
                "long_horizon",
                "forecast_horizons",
                "assignment_regime",
                "leaderboard_eligible",
                "state_features",
                "action_names",
                "outcome_names",
                "primary_observed_endpoint",
                "decision_endpoint",
                "surrogate_window",
                "campaign_mode",
                "campaign_length",
                "nonstationarity",
                "shift_time",
            ),
            notes="Short-panel experiment data; private surrogate-validity labels and long-horizon truth are hidden.",
        )
    if family == "streamretain":
        return EstimatorInformationContract(
            family="streamretain",
            visible_arrays=COMMON_VISIBLE_ARRAYS,
            visible_metadata=(
                "family",
                "scenario",
                "sample_size",
                "gamma",
                "trajectory_horizon",
                "target_policy",
                "leaderboard_eligible",
                "state_features",
                "action_names",
                "outcome_names",
                "dose_field",
                "action_constraints",
                "target_policy_distance",
                "nonstationarity",
                "shift_time",
            ),
            notes="Subscription lifecycle OPE with fixed named target policies and public action masks.",
        )
    if family == "clinic_dtr":
        return EstimatorInformationContract(
            family="clinic_dtr",
            visible_arrays=COMMON_VISIBLE_ARRAYS,
            visible_metadata=(
                "family",
                "scenario",
                "sample_size",
                "gamma",
                "trajectory_horizon",
                "target_policy",
                "leaderboard_eligible",
                "state_features",
                "action_names",
                "outcome_names",
                "dose_field",
                "action_constraints",
                "target_policy_distance",
                "nonstationarity",
                "shift_time",
            ),
            notes="Dynamic treatment-regime OPE with public censoring, missingness, and action eligibility.",
        )
    if family == "epicare":
        return EstimatorInformationContract(
            family="epicare",
            visible_arrays=COMMON_VISIBLE_ARRAYS,
            visible_metadata=(
                "family",
                "scenario",
                "sample_size",
                "gamma",
                "trajectory_horizon",
                "target_policy",
                "leaderboard_eligible",
                "state_features",
                "action_names",
                "outcome_names",
                "external_benchmark",
                "gym_env_id",
                "gym_api",
                "action_constraints",
                "target_policy_distance",
            ),
            notes="Optional external EpiCare Gym benchmark converted to the public longitudinal OPE schema.",
        )
    raise ValueError(f"Unsupported family '{family}'.")
