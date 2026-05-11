from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from causal_ope_benchmark.adapters import to_effect_panel, to_fqe_dataset, to_occupancy_ratio_dataset, to_survival_panel
from causal_ope_benchmark.policies import target_policy_probabilities
from causal_ope_benchmark.types import Array, BenchmarkProblem, LongitudinalDataset, TruthBundle, action_index
from causal_ope_benchmark.validation import validate_status


@dataclass
class EstimatorResult:
    """One estimator result before truth-based scoring."""

    estimator: str
    status: str
    estimates: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_sec: float = 0.0
    skip_reason: str = ""
    diagnostic_only: bool = False
    tuning_rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = validate_status(self.status, context=f"{self.estimator}.status")


def run_estimator(estimator: str, problem: BenchmarkProblem, config: Any | None = None) -> EstimatorResult:
    """Run a lightweight baseline estimator."""
    start = time.time()
    try:
        if estimator == "oracle_diagnostic":
            result = _oracle_diagnostic(problem.truth)
        elif estimator == "naive_short_term":
            result = _naive_short_term(problem.dataset)
        elif estimator == "streamlift_stratified_gcomp":
            result = _streamlift_stratified_gcomp(problem.dataset)
        elif estimator == "direct_method":
            result = _direct_method(problem.dataset)
        elif estimator == "ipw":
            result = _ipw(problem.dataset, normalize=False)
        elif estimator == "snipw":
            result = _ipw(problem.dataset, normalize=True)
        elif estimator == "doubly_robust":
            result = _doubly_robust(problem.dataset)
        elif estimator == "linear_fqe":
            result = _linear_fqe(problem.dataset)
        elif estimator == "boosted_fqe":
            result = _package_boosted_fqe(problem.dataset, config=config)
        elif estimator == "boosted_fqe_auto":
            result = _package_fqe_auto(problem.dataset, config=config, family="boosted")
        elif estimator == "neural_fqe":
            result = _package_neural_fqe(problem.dataset, config=config, diagnostic_streamlift=False)
        elif estimator == "neural_fqe_auto":
            result = _package_fqe_auto(problem.dataset, config=config, family="neural")
        elif estimator == "neural_fqe_streamlift_diagnostic":
            result = _package_neural_fqe(problem.dataset, config=config, diagnostic_streamlift=True)
        elif estimator == "discounted_occupancy_boosted":
            result = _package_discounted_occupancy_ope(problem.dataset, config=config, family="boosted", tuned=False)
        elif estimator == "discounted_occupancy_neural":
            result = _package_discounted_occupancy_ope(problem.dataset, config=config, family="neural", tuned=False)
        elif estimator == "discounted_occupancy_boosted_auto":
            result = _package_discounted_occupancy_ope(problem.dataset, config=config, family="boosted", tuned=True)
        elif estimator == "discounted_occupancy_neural_auto":
            result = _package_discounted_occupancy_ope(problem.dataset, config=config, family="neural", tuned=True)
        elif estimator == "ipcw_rmst":
            result = _ipcw_rmst(problem.dataset)
        else:
            result = EstimatorResult(estimator=estimator, status="skipped", skip_reason="unknown estimator")
    except Exception as exc:
        result = EstimatorResult(estimator=estimator, status="error", skip_reason=f"{type(exc).__name__}: {exc}")
    result.runtime_sec = float(time.time() - start)
    return result


def _oracle_diagnostic(truth: TruthBundle) -> EstimatorResult:
    estimates: dict[str, float] = {}
    estimates.update({key: float(value) for key, value in truth.values.items()})
    estimates.update({key: float(value) for key, value in truth.effects.items()})
    estimates.update({key: float(value) for key, value in truth.rmst.items()})
    estimates.update({key: float(value) for key, value in truth.subgroup_effects.items()})
    if "survival_target" in truth.survival_curves:
        estimates["survival_horizon"] = float(np.asarray(truth.survival_curves["survival_target"])[-1])
    return EstimatorResult(
        estimator="oracle_diagnostic",
        status="ok",
        estimates=estimates,
        diagnostics={"diagnostic_only": 1, "interval_available": 0},
        diagnostic_only=True,
    )


def _naive_short_term(dataset: LongitudinalDataset) -> EstimatorResult:
    if dataset.family != "streamlift":
        return EstimatorResult(estimator="naive_short_term", status="skipped", skip_reason="StreamLift-only baseline")
    panel = to_effect_panel(dataset)
    action = action_index(dataset.actions)
    first_time = np.asarray(dataset.time) == 0
    treat_units = set(np.asarray(dataset.unit_id)[first_time & (action == 1)].tolist())
    control_units = set(np.asarray(dataset.unit_id)[first_time & (action == 0)].tolist())
    assigned = None if dataset.assigned_arm is None else np.asarray(dataset.assigned_arm)
    unit_rewards = _unit_discounted_rewards(dataset, max_time=int(dataset.metadata_public["observed_horizon"]))
    treat = np.array([value for unit, value in unit_rewards.items() if unit in treat_units], dtype=np.float64)
    control = np.array([value for unit, value in unit_rewards.items() if unit in control_units], dtype=np.float64)
    if treat.size == 0 or control.size == 0:
        return EstimatorResult(estimator="naive_short_term", status="error", skip_reason="empty treatment arm")
    observed_effect = float(np.mean(treat) - np.mean(control))
    observed_horizon = max(1, int(dataset.metadata_public["observed_horizon"]))
    estimates = {}
    for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"):
        h = int(raw)
        estimates[f"effect_horizon_{h}"] = observed_effect * (1.0 - dataset.gamma**h) / (1.0 - dataset.gamma**observed_horizon)
        estimates[f"tot_effect_horizon_{h}"] = estimates[f"effect_horizon_{h}"]
    if _streamlift_include_infinite_horizon(dataset):
        scale = _infinite_discount_factor_sum(dataset.gamma) / _discount_factor_sum(dataset.gamma, observed_horizon)
        estimates["effect_horizon_infinite"] = observed_effect * scale
        estimates["tot_effect_horizon_infinite"] = estimates["effect_horizon_infinite"]
    if assigned is not None:
        assigned_treat_units = set(np.asarray(dataset.unit_id)[first_time & (assigned == 1)].tolist())
        assigned_control_units = set(np.asarray(dataset.unit_id)[first_time & (assigned == 0)].tolist())
        assigned_treat = np.array([value for unit, value in unit_rewards.items() if unit in assigned_treat_units], dtype=np.float64)
        assigned_control = np.array([value for unit, value in unit_rewards.items() if unit in assigned_control_units], dtype=np.float64)
        if assigned_treat.size and assigned_control.size:
            itt_observed = float(np.mean(assigned_treat) - np.mean(assigned_control))
            for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"):
                h = int(raw)
                estimates[f"itt_effect_horizon_{h}"] = itt_observed * (1.0 - dataset.gamma**h) / (1.0 - dataset.gamma**observed_horizon)
            if _streamlift_include_infinite_horizon(dataset):
                scale = _infinite_discount_factor_sum(dataset.gamma) / _discount_factor_sum(dataset.gamma, observed_horizon)
                estimates["itt_effect_horizon_infinite"] = itt_observed * scale
    se = float(np.sqrt(np.var(treat) / max(treat.size, 1) + np.var(control) / max(control.size, 1)))
    intervals = {
        key: (float(value - 1.96 * se), float(value + 1.96 * se))
        for key, value in estimates.items()
    }
    return EstimatorResult(
        estimator="naive_short_term",
        status="ok",
        estimates=estimates,
        intervals=intervals,
        diagnostics={"interval_available": 1, "effect_panel_units": int(panel.baseline_state.shape[0])},
    )


def _direct_method(dataset: LongitudinalDataset) -> EstimatorResult:
    x = _feature_matrix(dataset.states, dataset.actions)
    beta = _ridge_solve(x, dataset.rewards, ridge=1e-3)
    target_probs, expectation_mode, probability_storage = _target_action_probabilities(dataset)
    pred = _expected_linear_q(dataset.states, target_probs, beta)
    horizon = _horizon(dataset)
    value = float(np.mean(pred) * _discount_factor_sum(dataset.gamma, horizon))
    estimates = _family_value_estimates(dataset, value)
    if dataset.family == "streamlift":
        estimates = _streamlift_direct_effects(dataset, beta)
    return EstimatorResult(
        estimator="direct_method",
        status="ok",
        estimates=estimates,
        diagnostics={
            "interval_available": 0,
            "regression_dim": int(x.shape[1]),
            "target_policy_expectation_mode": expectation_mode,
            "target_probability_storage": probability_storage,
            "fqe_adapter_row_expansion_factor": 1.0,
        },
    )


def _ipw(dataset: LongitudinalDataset, *, normalize: bool) -> EstimatorResult:
    if dataset.family == "streamlift":
        return _streamlift_ipw(dataset, normalize=normalize)
    ratios = _target_over_behavior_ratio(dataset)
    unit_values = _weighted_unit_values(dataset, ratios, normalize=normalize)
    value = float(np.mean(list(unit_values.values()))) if unit_values else 0.0
    ess = _ess_fraction(np.asarray(list(_unit_final_weights(dataset, ratios).values()), dtype=np.float64))
    return EstimatorResult(
        estimator="snipw" if normalize else "ipw",
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics={
            "interval_available": 0,
            "ess_fraction": ess,
            "weight_p95": float(np.quantile(ratios, 0.95)),
            "diagnostic_only_ess": ess,
        },
    )


def _doubly_robust(dataset: LongitudinalDataset) -> EstimatorResult:
    if dataset.family == "streamlift":
        direct = _direct_method(dataset)
        ipw = _streamlift_ipw(dataset, normalize=False)
        estimates = {}
        for key in set(direct.estimates) | set(ipw.estimates):
            if key in direct.estimates and key in ipw.estimates:
                estimates[key] = float(0.5 * direct.estimates[key] + 0.5 * ipw.estimates[key])
            elif key in direct.estimates:
                estimates[key] = direct.estimates[key]
            else:
                estimates[key] = ipw.estimates[key]
        return EstimatorResult(estimator="doubly_robust", status="ok", estimates=estimates, diagnostics={"interval_available": 0})
    x = _feature_matrix(dataset.states, dataset.actions)
    beta = _ridge_solve(x, dataset.rewards, ridge=1e-3)
    pred_behavior = x @ beta
    target_probs, expectation_mode, probability_storage = _target_action_probabilities(dataset)
    pred_target = _expected_linear_q(dataset.states, target_probs, beta)
    ratios = _target_over_behavior_ratio(dataset)
    residual = np.asarray(dataset.rewards) - pred_behavior
    pseudo = pred_target + np.clip(ratios, 0.0, 50.0) * residual
    value = float(np.mean(pseudo) * _discount_factor_sum(dataset.gamma, _horizon(dataset)))
    return EstimatorResult(
        estimator="doubly_robust",
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics={
            "interval_available": 0,
            "ess_fraction": _ess_fraction(ratios),
            "target_policy_expectation_mode": expectation_mode,
            "target_probability_storage": probability_storage,
            "fqe_adapter_row_expansion_factor": 1.0,
        },
    )


def _linear_fqe(dataset: LongitudinalDataset) -> EstimatorResult:
    if dataset.family == "streamlift":
        return EstimatorResult(estimator="linear_fqe", status="skipped", skip_reason="FQE is not used for StreamLift forecasting")
    phi = _feature_matrix(dataset.states, dataset.actions)
    beta = np.zeros(phi.shape[1], dtype=np.float64)
    next_probs, next_expectation_mode, next_probability_storage = _next_target_action_probabilities(dataset)
    for _ in range(25):
        next_value = _expected_linear_q(dataset.next_states, next_probs, beta)
        target = dataset.rewards + dataset.gamma * (1.0 - _fqe_terminals(dataset)) * next_value
        beta = _ridge_solve(phi, target, ridge=5e-3)
    initial_probs, initial_expectation_mode, initial_probability_storage = _initial_target_action_probabilities(dataset)
    value = float(np.mean(_expected_linear_q(dataset.initial_states, initial_probs, beta)))
    return EstimatorResult(
        estimator="linear_fqe",
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics={
            "interval_available": 0,
            "fqe_iterations": 25,
            "target_policy_expectation_mode": _combine_expectation_modes(next_expectation_mode, initial_expectation_mode),
            "target_probability_storage": _combine_probability_storage(next_probability_storage, initial_probability_storage),
            "fqe_adapter_row_expansion_factor": 1.0,
        },
    )


def _ipcw_rmst(dataset: LongitudinalDataset) -> EstimatorResult:
    if dataset.family != "clinic_dtr":
        return EstimatorResult(estimator="ipcw_rmst", status="skipped", skip_reason="ClinicDTR-only survival baseline")
    panel = to_survival_panel(dataset)
    ratios = np.asarray(panel.target_propensity, dtype=np.float64) / np.clip(np.asarray(panel.behavior_propensity, dtype=np.float64), 1e-12, np.inf)
    horizon = _horizon(dataset)
    units = np.unique(panel.unit_id)
    censor_survival = _kaplan_meier_censor_survival(panel, horizon)
    survival_curve: list[float] = []
    ipc_weights: list[float] = []
    for t in range(horizon):
        total = 0.0
        for unit in units:
            idx = np.flatnonzero(panel.unit_id == unit)
            idx = idx[np.argsort(panel.time[idx])]
            observed = idx[panel.time[idx] <= t]
            if observed.size == 0:
                continue
            if np.any((panel.censored[observed] > 0.5) & (panel.time[observed] < t)):
                continue
            alive_after_t = 0.0 if np.any(panel.event[observed] > 0.5) else 1.0
            if alive_after_t == 0.0:
                continue
            last_time = int(np.max(panel.time[observed]))
            if last_time < t:
                continue
            cumulative_ratio = float(np.prod(np.clip(ratios[observed], 0.0, 50.0)))
            weight = float(np.clip(cumulative_ratio / max(float(censor_survival[t]), 0.05), 0.0, 50.0))
            ipc_weights.append(weight)
            total += weight * alive_after_t
        survival_curve.append(float(np.clip(total / max(units.size, 1), 0.0, 1.0)))
    rmst = float(np.sum(survival_curve))
    survival = float(survival_curve[-1]) if survival_curve else 0.0
    weights_arr = np.asarray(ipc_weights, dtype=np.float64)
    return EstimatorResult(
        estimator="ipcw_rmst",
        status="ok",
        estimates={"rmst": rmst, "survival_horizon": survival, "policy_value": rmst / max(1.0, _horizon(dataset))},
        diagnostics={
            "interval_available": 0,
            "censoring_rate": float(np.mean(panel.censored)),
            "censor_survival_min": float(np.min(censor_survival)) if censor_survival.size else 1.0,
            "ess_fraction": _ess_fraction(weights_arr) if weights_arr.size else 0.0,
        },
    )


def _package_neural_fqe(
    dataset: LongitudinalDataset,
    *,
    config: Any | None,
    diagnostic_streamlift: bool,
) -> EstimatorResult:
    estimator_name = "neural_fqe_streamlift_diagnostic" if diagnostic_streamlift else "neural_fqe"
    if dataset.family == "streamlift" and not diagnostic_streamlift:
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="StreamLift uses causal forecasting; use neural_fqe_streamlift_diagnostic.")
    if dataset.family != "streamlift" and diagnostic_streamlift:
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="StreamLift-only diagnostic estimator")
    try:
        from fqe import fit_fqe_neural
    except ModuleNotFoundError as exc:
        return EstimatorResult(estimator=estimator_name, status="missing_dependency", skip_reason=f"fqe/torch unavailable: {exc}")
    fqe_data = to_fqe_dataset(dataset, target_policy_expectation_mode="exact_discrete")
    cfg = _neural_fqe_config(dataset, config=config, seed=int(dataset.seed), tuned=False)
    model = fit_fqe_neural(
        states=fqe_data.states,
        actions=fqe_data.actions,
        next_states=fqe_data.next_states,
        next_actions=fqe_data.next_actions,
        rewards=fqe_data.rewards,
        gamma=fqe_data.gamma,
        terminals=fqe_data.terminals,
        sample_weight=fqe_data.sample_weight,
        config=cfg,
    )
    value = _estimate_model_target_policy_value(model, fqe_data.initial_states, fqe_data.initial_action_probabilities, fqe_data.initial_actions)
    diagnostics = dict(model.diagnostics)
    diagnostics.update(
        {
            "interval_available": 0,
            "fqe_backend": "neural",
            "fqe_censoring_terminal_rate": float(np.mean(fqe_data.terminals)),
            "fqe_raw_terminal_rate": float(np.mean(dataset.terminals)),
            "target_policy_expectation_mode": fqe_data.target_policy_expectation_mode,
            "target_probability_storage": fqe_data.target_probability_storage,
            "fqe_adapter_row_expansion_factor": fqe_data.row_expansion_factor,
        }
    )
    return EstimatorResult(
        estimator=estimator_name,
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics=diagnostics,
        diagnostic_only=bool(diagnostic_streamlift),
    )


def _package_boosted_fqe(dataset: LongitudinalDataset, *, config: Any | None) -> EstimatorResult:
    if dataset.family == "streamlift":
        return EstimatorResult(estimator="boosted_fqe", status="skipped", skip_reason="StreamLift uses causal forecasting, not primary FQE scoring.")
    try:
        from fqe import fit_fqe_lgbm
    except ModuleNotFoundError as exc:
        return EstimatorResult(estimator="boosted_fqe", status="missing_dependency", skip_reason=f"fqe/lightgbm unavailable: {exc}")
    fqe_data = to_fqe_dataset(dataset, target_policy_expectation_mode="exact_discrete")
    cfg = _boosted_fqe_config(dataset, config=config, seed=int(dataset.seed), tuned=False)
    model = fit_fqe_lgbm(
        states=fqe_data.states,
        actions=fqe_data.actions,
        next_states=fqe_data.next_states,
        next_actions=fqe_data.next_actions,
        rewards=fqe_data.rewards,
        gamma=fqe_data.gamma,
        terminals=fqe_data.terminals,
        sample_weight=fqe_data.sample_weight,
        config=cfg,
    )
    value = _estimate_model_target_policy_value(model, fqe_data.initial_states, fqe_data.initial_action_probabilities, fqe_data.initial_actions)
    diagnostics = dict(model.diagnostics)
    diagnostics.update(
        {
            "interval_available": 0,
            "fqe_backend": "boosted",
            "fqe_censoring_terminal_rate": float(np.mean(fqe_data.terminals)),
            "fqe_raw_terminal_rate": float(np.mean(dataset.terminals)),
            "target_policy_expectation_mode": fqe_data.target_policy_expectation_mode,
            "target_probability_storage": fqe_data.target_probability_storage,
            "fqe_adapter_row_expansion_factor": fqe_data.row_expansion_factor,
        }
    )
    return EstimatorResult(
        estimator="boosted_fqe",
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics=diagnostics,
    )


def _package_fqe_auto(
    dataset: LongitudinalDataset,
    *,
    config: Any | None,
    family: str,
) -> EstimatorResult:
    estimator_name = f"{family}_fqe_auto"
    if dataset.family == "streamlift":
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="StreamLift uses causal forecasting, not primary FQE scoring.")
    try:
        from fqe import FQESearchSpace, FQETuningConfig, tune_fqe_auto
    except ModuleNotFoundError as exc:
        return EstimatorResult(estimator=estimator_name, status="missing_dependency", skip_reason=f"fqe unavailable: {exc}")
    if family not in {"boosted", "neural"}:
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="unknown FQE family")
    fqe_data = to_fqe_dataset(dataset, target_policy_expectation_mode="exact_discrete")
    search_space = FQESearchSpace(
        boosted=_boosted_fqe_config(dataset, config=config, seed=int(dataset.seed), tuned=True),
        neural=_neural_fqe_config(dataset, config=config, seed=int(dataset.seed), tuned=True),
    )
    tuning_config = FQETuningConfig(
        families=(family,),
        cv_folds=3,
        seed=int(dataset.seed) + (17_003 if family == "boosted" else 29_011),
        budget=_automl_budget(config),
        max_candidates=4 if _is_smoke(config) else 8,
        promotion_candidates=2 if _is_smoke(config) else 3,
        refit=True,
        stable_fallback=True,
    )
    tuned = tune_fqe_auto(
        states=fqe_data.states,
        actions=fqe_data.actions,
        next_states=fqe_data.next_states,
        next_actions=fqe_data.next_actions,
        rewards=fqe_data.rewards,
        gamma=fqe_data.gamma,
        terminals=fqe_data.terminals,
        sample_weight=fqe_data.sample_weight,
        initial_states=fqe_data.initial_states,
        initial_actions=fqe_data.initial_actions,
        families=(family,),
        search_space=search_space,
        config=tuning_config,
    )
    if tuned.model is None:
        return EstimatorResult(estimator=estimator_name, status="error", skip_reason="FQE AutoML did not return a refit model.")
    value = _estimate_model_target_policy_value(tuned.model, fqe_data.initial_states, fqe_data.initial_action_probabilities, fqe_data.initial_actions)
    diagnostics = dict(tuned.model.diagnostics)
    diagnostics.update(
        {
            "interval_available": 0,
            "fqe_backend": family,
            "fqe_automl": 1,
            "selected_candidate_id": tuned.selected_candidate_id,
            "selected_family": tuned.selected_family,
            "target_policy_expectation_mode": fqe_data.target_policy_expectation_mode,
            "target_probability_storage": fqe_data.target_probability_storage,
            "fqe_adapter_row_expansion_factor": fqe_data.row_expansion_factor,
        }
    )
    return EstimatorResult(
        estimator=estimator_name,
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics=diagnostics,
        tuning_rows=_tuning_rows_from_result(tuned, estimator=estimator_name, dataset=dataset),
    )


def _package_discounted_occupancy_ope(
    dataset: LongitudinalDataset,
    *,
    config: Any | None,
    family: str,
    tuned: bool,
) -> EstimatorResult:
    suffix = "_auto" if tuned else ""
    estimator_name = f"discounted_occupancy_{family}{suffix}"
    if dataset.family == "streamlift":
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="StreamLift uses causal forecasting, not discounted occupancy OPE.")
    try:
        from occupancy_ratio import tune_occupancy_ratio_auto
    except ModuleNotFoundError as exc:
        return EstimatorResult(estimator=estimator_name, status="missing_dependency", skip_reason=f"occupancy-ratio unavailable: {exc}")
    if family == "boosted":
        try:
            from occupancy_ratio import fit_discounted_occupancy_ratio
        except ModuleNotFoundError as exc:
            return EstimatorResult(estimator=estimator_name, status="missing_dependency", skip_reason=f"occupancy-ratio/lightgbm unavailable: {exc}")
    elif family == "neural":
        try:
            from occupancy_ratio import fit_discounted_occupancy_ratio_neural
        except ModuleNotFoundError as exc:
            return EstimatorResult(estimator=estimator_name, status="missing_dependency", skip_reason=f"occupancy-ratio/torch unavailable: {exc}")
    else:
        return EstimatorResult(estimator=estimator_name, status="skipped", skip_reason="unknown occupancy family")

    occ_data = to_occupancy_ratio_dataset(dataset)
    terminals = 1.0 - np.asarray(occ_data.masks, dtype=np.float64)
    known_action_ratio = _known_action_ratio(dataset)
    common_kwargs = dict(
        states=occ_data.states,
        actions=occ_data.actions,
        next_states=occ_data.next_states,
        target_actions=occ_data.target_actions,
        gamma=occ_data.gamma,
        initial_states=occ_data.initial_states,
        initial_actions=occ_data.initial_actions,
        initial_weights=occ_data.initial_weights,
        target_next_actions=occ_data.next_target_actions,
        terminals=terminals,
        initial_ratio_mode="auto",
        one_step_ratio_mode="auto",
    )
    tuning_rows: list[dict[str, Any]] = []
    if tuned:
        tuned_result = tune_occupancy_ratio_auto(
            **{key: value for key, value in common_kwargs.items() if key != "terminals"},
            rewards=occ_data.rewards,
            families=(family,),
            search_space=_occupancy_search_space(dataset, config=config, family=family),
            config=_occupancy_tuning_config(dataset, config=config, family=family),
        )
        model = tuned_result.model
        tuning_rows = _occupancy_tuning_rows(tuned_result, estimator=estimator_name, dataset=dataset)
    else:
        if family == "boosted":
            model = fit_discounted_occupancy_ratio(
                **common_kwargs,
                action_ratio_values=known_action_ratio,
                known_action_ratio_clip_max=50.0,
                occupancy=_boosted_occupancy_config(dataset, config=config),
                action_ratio=_boosted_action_ratio_config(dataset, config=config),
                source_state_ratio=_boosted_source_ratio_config(dataset, config=config),
                transition_ratio=_boosted_transition_ratio_config(dataset, config=config),
            )
        else:
            model = fit_discounted_occupancy_ratio_neural(
                **common_kwargs,
                action_ratio_values=known_action_ratio,
                known_action_ratio_clip_max=50.0,
                occupancy=_neural_occupancy_config(dataset, config=config),
                action_ratio=_neural_action_ratio_config(dataset, config=config),
                source_state_ratio=_neural_source_ratio_config(dataset, config=config),
                transition_ratio=_neural_transition_ratio_config(dataset, config=config),
            )
    raw_weights = np.asarray(model.predict_state_action_ratio(occ_data.states, occ_data.actions, clip=False), dtype=np.float64).reshape(-1)
    weights = np.asarray(model.predict_state_action_ratio(occ_data.states, occ_data.actions, clip=True), dtype=np.float64).reshape(-1)
    value, value_diag = _discounted_occupancy_value(occ_data.rewards, weights, gamma=occ_data.gamma, horizon=_horizon(dataset))
    diagnostics = {
        "interval_available": 0,
        "occupancy_backend": family,
        "occupancy_automl": int(bool(tuned)),
        "known_action_ratio_used": int(not tuned),
        "target_policy_expectation_mode": "sampled_action",
        "target_probability_storage": "sampled_action_only",
        "ess_fraction": _ess_fraction(weights),
        "diagnostic_only_ess": _ess_fraction(weights),
        "weight_p95": float(np.quantile(weights, 0.95)) if weights.size else 0.0,
        "weight_cv": _coefficient_of_variation(weights),
        "raw_weight_cv": _coefficient_of_variation(raw_weights),
        "weight_mean": float(np.mean(weights)) if weights.size else 0.0,
        "weight_max": float(np.max(weights)) if weights.size else 0.0,
    }
    diagnostics.update(value_diag)
    diagnostics.update(_public_model_diagnostics(model))
    if tuned:
        diagnostics.update(
            {
                "selected_candidate_id": getattr(tuned_result, "selected_candidate_id", ""),
                "selected_family": getattr(tuned_result, "selected_family", family),
            }
        )
    return EstimatorResult(
        estimator=estimator_name,
        status="ok",
        estimates=_family_value_estimates(dataset, value),
        diagnostics=diagnostics,
        tuning_rows=tuning_rows,
    )


def _boosted_fqe_config(dataset: LongitudinalDataset, *, config: Any | None, seed: int, tuned: bool):
    from fqe import BoostedFQEConfig

    base_iterations = int(getattr(config, "fqe_num_iterations", 24))
    if _is_epicare_core(dataset, config):
        iterations = max(48 if tuned else 64, base_iterations if not tuned else base_iterations // 2)
        num_leaves = 31
        min_leaf = max(5, min(30, dataset.n // 50))
        patience = 12
    else:
        iterations = max(8, base_iterations // 2 if not tuned else base_iterations // 3)
        num_leaves = 15 if _is_smoke(config) else 31
        min_leaf = 5 if _is_smoke(config) else 10
        patience = 6 if _is_smoke(config) else 10
    return BoostedFQEConfig.stable_defaults(
        num_iterations=int(iterations),
        validation_fraction=0.20,
        patience=int(patience),
        seed=int(seed),
        show_progress=False,
        lgb_params={
            "learning_rate": 0.05,
            "num_leaves": int(num_leaves),
            "min_data_in_leaf": int(min_leaf),
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 1,
        },
    )


def _neural_fqe_config(dataset: LongitudinalDataset, *, config: Any | None, seed: int, tuned: bool):
    from fqe import NeuralFQEConfig

    if _is_epicare_core(dataset, config):
        hidden_dims = (128, 128)
        iterations = max(40 if tuned else 64, int(getattr(config, "fqe_num_iterations", 48)))
        steps = max(12 if tuned else 20, int(getattr(config, "fqe_gradient_steps_per_iteration", 16)))
        batch_size = max(256, int(getattr(config, "fqe_batch_size", 256)))
        learning_rate = 5e-4
        tau = 0.08
        patience = 12
    else:
        hidden_dims = tuple(int(width) for width in getattr(config, "fqe_hidden_dims", (32, 32)))
        iterations = int(getattr(config, "fqe_num_iterations", 24))
        steps = int(getattr(config, "fqe_gradient_steps_per_iteration", 10))
        if tuned:
            iterations = max(3, iterations // 2)
            steps = max(2, steps // 2)
        batch_size = int(getattr(config, "fqe_batch_size", 128))
        learning_rate = 1e-3 if _is_smoke(config) else 5e-4
        tau = 0.35 if _is_smoke(config) else 0.10
        patience = 6 if _is_smoke(config) else 10
    return NeuralFQEConfig.stable_defaults(
        hidden_dims=hidden_dims,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        num_iterations=int(iterations),
        gradient_steps_per_iteration=int(steps),
        target_update_tau=float(tau),
        validation_fraction=0.20,
        patience=int(patience),
        seed=int(seed),
        device="cpu",
        show_progress=False,
    )


def _boosted_occupancy_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import OccupancyRegressionConfig

    smoke = _is_smoke(config)
    if _is_epicare_core(dataset, config):
        return OccupancyRegressionConfig.stable_defaults(
            num_iterations=60,
            trees_per_iteration=1,
            mcmc_samples=24,
            batch_size=512,
            validation_fraction=0.20,
            patience=12,
            seed=int(dataset.seed) + 41_101,
            show_progress=False,
            lgb_params={
                "learning_rate": 0.06,
                "num_leaves": 31,
                "min_data_in_leaf": max(5, min(30, dataset.n // 50)),
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 1,
            },
        )
    return OccupancyRegressionConfig.stable_defaults(
        num_iterations=max(2, min(8, int(getattr(config, "fqe_num_iterations", 24)) // 3)) if smoke else 30,
        trees_per_iteration=1,
        mcmc_samples=4 if smoke else 16,
        batch_size=128 if smoke else 512,
        validation_fraction=0.20,
        patience=3 if smoke else 8,
        seed=int(dataset.seed) + 41_101,
        show_progress=False,
        lgb_params={
            "learning_rate": 0.08,
            "num_leaves": 15 if smoke else 31,
            "min_data_in_leaf": 2 if smoke else 15,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 1,
        },
    )


def _boosted_action_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import ActionRatioConfig

    rounds = _boosted_nuisance_rounds(dataset, config)
    return ActionRatioConfig.stable_defaults(
        num_boost_round=rounds,
        validation_fraction=0.20,
        early_stopping_rounds=3 if _is_smoke(config) else 8,
        refit_on_all_data=True,
        prediction_max=50.0,
        density_ratio_loss="lsif",
        show_progress=False,
        lgb_params=_boosted_nuisance_lgb_params(dataset, config),
    )


def _boosted_source_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import SourceStateRatioConfig

    rounds = _boosted_nuisance_rounds(dataset, config)
    return SourceStateRatioConfig.stable_defaults(
        num_boost_round=rounds,
        validation_fraction=0.20,
        early_stopping_rounds=3 if _is_smoke(config) else 8,
        refit_on_all_data=True,
        prediction_max=50.0,
        density_ratio_loss="lsif",
        show_progress=False,
        lgb_params=_boosted_nuisance_lgb_params(dataset, config),
    )


def _boosted_transition_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import TransitionRatioConfig

    smoke = _is_smoke(config)
    rounds = max(_boosted_nuisance_rounds(dataset, config), 8 if smoke else 80)
    return TransitionRatioConfig.stable_defaults(
        num_boost_round=rounds,
        permutation_samples=2 if smoke else (16 if _is_epicare_core(dataset, config) else 8),
        validation_fraction=0.20,
        early_stopping_rounds=3 if smoke else 8,
        refit_on_all_data=True,
        prediction_max=50.0,
        density_ratio_loss="lsif",
        show_progress=False,
        lgb_params=_boosted_nuisance_lgb_params(dataset, config),
    )


def _neural_occupancy_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import NeuralOccupancyRegressionConfig

    smoke = _is_smoke(config)
    smoke_outer = max(2, min(8, int(getattr(config, "fqe_num_iterations", 24))))
    smoke_steps = max(1, min(4, int(getattr(config, "fqe_gradient_steps_per_iteration", 10))))
    smoke_one_step = max(5, min(80, int(getattr(config, "fqe_num_iterations", 24)) * smoke_steps))
    hidden = (128, 128) if _is_epicare_core(dataset, config) else tuple(int(width) for width in getattr(config, "fqe_hidden_dims", (32, 32)))
    return NeuralOccupancyRegressionConfig.stable_defaults(
        hidden_dims=hidden,
        activation="silu",
        learning_rate=5e-4 if _is_epicare_core(dataset, config) else (1e-3 if smoke else 5e-4),
        weight_decay=1e-5,
        batch_size=256 if smoke else 512,
        num_iterations=smoke_outer if smoke else (60 if _is_epicare_core(dataset, config) else 30),
        gradient_steps_per_iteration=smoke_steps if smoke else (8 if _is_epicare_core(dataset, config) else 6),
        mcmc_samples=8 if smoke else 24,
        validation_fraction=0.20,
        patience=5 if smoke else 12,
        validation_warmup_iterations=1,
        fixed_point_damping=0.5,
        normalize_occupancy=True,
        occupancy_ratio_max=50.0,
        direct_adjoint_steps=max(5, smoke_one_step // 2) if smoke else 128,
        direct_one_step_max_steps=smoke_one_step if smoke else 256,
        direct_one_step_hidden_dims=hidden,
        direct_one_step_density_ratio_loss="lsif",
        direct_one_step_prediction_max=50.0,
        direct_one_step_moment_calibration="scalar",
        grad_clip_norm=5.0,
        device="cpu",
        seed=int(dataset.seed) + 51_101,
        show_progress=False,
    )


def _neural_action_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import NeuralActionRatioConfig

    return NeuralActionRatioConfig.balanced_defaults(**_neural_nuisance_kwargs(dataset, config, seed_offset=51_211, transition=False))


def _neural_source_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import NeuralSourceStateRatioConfig

    return NeuralSourceStateRatioConfig.balanced_defaults(**_neural_nuisance_kwargs(dataset, config, seed_offset=51_307, transition=False))


def _neural_transition_ratio_config(dataset: LongitudinalDataset, *, config: Any | None):
    from occupancy_ratio import NeuralTransitionRatioConfig

    kwargs = _neural_nuisance_kwargs(dataset, config, seed_offset=51_401, transition=True)
    kwargs["permutation_samples"] = 4 if _is_smoke(config) else 16
    return NeuralTransitionRatioConfig.balanced_defaults(**kwargs)


def _occupancy_search_space(dataset: LongitudinalDataset, *, config: Any | None, family: str):
    from occupancy_ratio import OccupancySearchSpace

    return OccupancySearchSpace(
        boosted_occupancy=_boosted_occupancy_config(dataset, config=config),
        boosted_action_ratio=_boosted_action_ratio_config(dataset, config=config),
        boosted_source_state_ratio=_boosted_source_ratio_config(dataset, config=config),
        boosted_transition_ratio=_boosted_transition_ratio_config(dataset, config=config),
        neural_occupancy=_neural_occupancy_config(dataset, config=config),
        neural_action_ratio=_neural_action_ratio_config(dataset, config=config),
        neural_source_state_ratio=_neural_source_ratio_config(dataset, config=config),
        neural_transition_ratio=_neural_transition_ratio_config(dataset, config=config),
    )


def _occupancy_tuning_config(dataset: LongitudinalDataset, *, config: Any | None, family: str):
    from occupancy_ratio import OccupancyTuningConfig

    smoke = _is_smoke(config)
    return OccupancyTuningConfig(
        families=(family,),
        cv_folds=3,
        seed=int(dataset.seed) + (61_003 if family == "boosted" else 71_003),
        budget=_automl_budget(config),
        max_candidates=4 if smoke else 8,
        promotion_candidates=2 if smoke else 3,
        refit=True,
        stable_fallback=True,
        include_google_dualdice=False,
        stagewise=True,
        initial_ratio_mode_candidates=("auto", "factored"),
        one_step_ratio_mode_candidates=("auto", "factored"),
    )


def _boosted_nuisance_rounds(dataset: LongitudinalDataset, config: Any | None) -> int:
    if _is_smoke(config):
        return max(4, min(12, int(getattr(config, "fqe_num_iterations", 24))))
    if _is_epicare_core(dataset, config):
        return 120
    return 60


def _boosted_nuisance_lgb_params(dataset: LongitudinalDataset, config: Any | None) -> dict[str, Any]:
    smoke = _is_smoke(config)
    return {
        "learning_rate": 0.08 if smoke else 0.05,
        "num_leaves": 15 if smoke else 31,
        "min_data_in_leaf": 2 if smoke else max(5, min(30, dataset.n // 50)),
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": 1,
    }


def _neural_nuisance_kwargs(
    dataset: LongitudinalDataset,
    config: Any | None,
    *,
    seed_offset: int,
    transition: bool,
) -> dict[str, Any]:
    smoke = _is_smoke(config)
    hidden = (128, 128) if _is_epicare_core(dataset, config) else tuple(int(width) for width in getattr(config, "fqe_hidden_dims", (32, 32)))
    smoke_steps = max(
        5,
        min(
            80,
            int(getattr(config, "fqe_num_iterations", 24))
            * max(1, min(4, int(getattr(config, "fqe_gradient_steps_per_iteration", 10)))),
        ),
    )
    return {
        "hidden_dims": hidden,
        "activation": "silu",
        "learning_rate": 5e-4 if not smoke else 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 256 if smoke else 512,
        "max_steps": smoke_steps if smoke else (1000 if transition and _is_epicare_core(dataset, config) else 800),
        "validation_fraction": 0.20,
        "patience": 6 if smoke else 12,
        "prediction_max": 50.0,
        "density_ratio_loss": "lsif",
        "moment_calibration": "scalar",
        "device": "cpu",
        "seed": int(dataset.seed) + int(seed_offset),
    }


def _known_action_ratio(dataset: LongitudinalDataset) -> Array:
    target = np.asarray(dataset.target_propensity_observed_action, dtype=np.float64).reshape(-1)
    behavior = np.asarray(dataset.behavior_propensity, dtype=np.float64).reshape(-1)
    return target / np.clip(behavior, 1e-12, np.inf)


def _discounted_occupancy_value(rewards: Array, weights: Array, *, gamma: float, horizon: int) -> tuple[float, dict[str, float]]:
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(w))
    if r.shape != w.shape or total <= 0.0:
        return 0.0, {"occupancy_weighted_reward": 0.0, "occupancy_weight_sum": total}
    weighted_reward = float(np.sum(w * r) / total)
    scale = _discount_factor_sum(float(gamma), int(horizon))
    return float(weighted_reward * scale), {
        "occupancy_weighted_reward": weighted_reward,
        "occupancy_value_scale": float(scale),
        "occupancy_weight_sum": total,
    }


def _public_model_diagnostics(model: Any) -> dict[str, Any]:
    raw = getattr(model, "diagnostics", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            out[f"occupancy_{key}"] = value
    return out


def _tuning_rows_from_result(tuned: Any, *, estimator: str, dataset: LongitudinalDataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tuned.candidate_rows():
        out = _tuning_row_metadata(estimator=estimator, dataset=dataset, tuning_stage="automl_candidate")
        out.update(dict(row))
        rows.append(out)
    for row in tuned.fold_rows():
        out = _tuning_row_metadata(estimator=estimator, dataset=dataset, tuning_stage="automl_fold")
        out.update(dict(row))
        rows.append(out)
    return rows


def _occupancy_tuning_rows(tuned: Any, *, estimator: str, dataset: LongitudinalDataset) -> list[dict[str, Any]]:
    rows = _tuning_rows_from_result(tuned, estimator=estimator, dataset=dataset)
    for row in tuned.first_stage_candidate_rows():
        out = _tuning_row_metadata(estimator=estimator, dataset=dataset, tuning_stage="automl_first_stage_candidate")
        out.update(dict(row))
        rows.append(out)
    for row in tuned.first_stage_fold_rows():
        out = _tuning_row_metadata(estimator=estimator, dataset=dataset, tuning_stage="automl_first_stage_fold")
        out.update(dict(row))
        rows.append(out)
    return rows


def _tuning_row_metadata(*, estimator: str, dataset: LongitudinalDataset, tuning_stage: str) -> dict[str, Any]:
    return {
        "estimator": estimator,
        "dataset": dataset.name,
        "family": dataset.family,
        "scenario": dataset.scenario,
        "seed": int(dataset.seed),
        "sample_size": int(dataset.metadata_public.get("sample_size", dataset.n)),
        "gamma": float(dataset.gamma),
        "target_policy": dataset.metadata_public.get("target_policy", ""),
        "tuning_stage": tuning_stage,
    }


def _coefficient_of_variation(values: Array) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _automl_budget(config: Any | None) -> str:
    budget = str(getattr(config, "automl_tuning", "off"))
    if budget in {"fast", "balanced"}:
        return budget
    return "fast" if _is_smoke(config) else "balanced"


def _is_smoke(config: Any | None) -> bool:
    return str(getattr(config, "profile", "smoke")) == "smoke"


def _is_epicare_core(dataset: LongitudinalDataset, config: Any | None) -> bool:
    return dataset.family == "epicare" and not _is_smoke(config)


def _streamlift_direct_effects(dataset: LongitudinalDataset, beta: Array) -> dict[str, float]:
    base = np.asarray(dataset.states)
    control = np.zeros_like(dataset.actions)
    treatment = np.zeros_like(dataset.actions)
    control[:, 0] = 1.0
    treatment[:, 1] = 1.0
    step_effect = float(np.mean(_feature_matrix(base, treatment) @ beta - _feature_matrix(base, control) @ beta))
    estimates = {}
    for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"):
        h = int(raw)
        estimates[f"effect_horizon_{h}"] = step_effect * _discount_factor_sum(dataset.gamma, h)
        estimates[f"tot_effect_horizon_{h}"] = estimates[f"effect_horizon_{h}"]
    if _streamlift_include_infinite_horizon(dataset):
        estimates["effect_horizon_infinite"] = step_effect * _infinite_discount_factor_sum(dataset.gamma)
        estimates["tot_effect_horizon_infinite"] = estimates["effect_horizon_infinite"]
    return estimates


def _streamlift_stratified_gcomp(dataset: LongitudinalDataset) -> EstimatorResult:
    if dataset.family != "streamlift":
        return EstimatorResult(estimator="streamlift_stratified_gcomp", status="skipped", skip_reason="StreamLift-only baseline")
    actions = action_index(dataset.actions)
    states = np.asarray(dataset.states, dtype=np.float64)
    next_states = np.asarray(dataset.next_states, dtype=np.float64)
    rewards = np.asarray(dataset.rewards, dtype=np.float64)
    terminals = np.asarray(dataset.terminals, dtype=np.float64)
    forecast_horizons = tuple(int(raw) for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"))
    max_horizon = max(forecast_horizons)
    rollout_horizon = max(max_horizon, _streamlift_infinite_rollout_horizon(dataset)) if _streamlift_include_infinite_horizon(dataset) else max_horizon
    models: dict[int, dict[str, Array]] = {}
    row_counts: dict[int, int] = {}
    for arm in (0, 1):
        idx = np.flatnonzero(actions == arm)
        row_counts[arm] = int(idx.size)
        if idx.size < 5:
            return EstimatorResult(
                estimator="streamlift_stratified_gcomp",
                status="error",
                skip_reason=f"too few rows in StreamLift action arm {arm}",
            )
        features = _streamlift_dynamics_features(states[idx])
        models[arm] = {
            "delta": _ridge_solve(features, next_states[idx] - states[idx], ridge=2e-2),
            "reward": _ridge_solve(features, rewards[idx], ridge=2e-2),
            "terminal_logit": _logistic_ridge_fit(features, terminals[idx], ridge=2e-2),
        }
    values = {
        0: _streamlift_rollout_stratified_model(dataset, models, assigned_arm=0, horizon=rollout_horizon, forecast_horizons=forecast_horizons),
        1: _streamlift_rollout_stratified_model(dataset, models, assigned_arm=1, horizon=rollout_horizon, forecast_horizons=forecast_horizons),
    }
    noncompliance = _streamlift_observed_noncompliance(dataset)
    estimates: dict[str, float] = {}
    for horizon in forecast_horizons:
        control = float(values[0][horizon])
        treatment = float(values[1][horizon])
        effect = treatment - control
        estimates[f"value_control_horizon_{horizon}"] = control
        estimates[f"value_treatment_horizon_{horizon}"] = treatment
        estimates[f"effect_horizon_{horizon}"] = effect
        estimates[f"tot_effect_horizon_{horizon}"] = effect
        estimates[f"itt_effect_horizon_{horizon}"] = float((1.0 - 2.0 * noncompliance) * effect)
    if _streamlift_include_infinite_horizon(dataset):
        control = float(values[0][rollout_horizon])
        treatment = float(values[1][rollout_horizon])
        effect = treatment - control
        estimates["value_control_horizon_infinite"] = control
        estimates["value_treatment_horizon_infinite"] = treatment
        estimates["effect_horizon_infinite"] = effect
        estimates["tot_effect_horizon_infinite"] = effect
        estimates["itt_effect_horizon_infinite"] = float((1.0 - 2.0 * noncompliance) * effect)
    return EstimatorResult(
        estimator="streamlift_stratified_gcomp",
        status="ok",
        estimates=estimates,
        diagnostics={
            "interval_available": 0,
            "gcomp_model": "arm_stratified_stationary_ridge",
            "gcomp_feature_dim": int(_streamlift_dynamics_features(states[:1]).shape[1]),
            "gcomp_control_rows": row_counts[0],
            "gcomp_treatment_rows": row_counts[1],
            "gcomp_campaign_mode": dataset.metadata_public.get("campaign_mode", ""),
            "gcomp_campaign_length": int(dataset.metadata_public.get("campaign_length", 1)),
            "gcomp_observed_noncompliance_rate": float(noncompliance),
        },
    )


def _streamlift_rollout_stratified_model(
    dataset: LongitudinalDataset,
    models: dict[int, dict[str, Array]],
    *,
    assigned_arm: int,
    horizon: int,
    forecast_horizons: tuple[int, ...],
) -> dict[int, float]:
    initial = np.asarray(dataset.initial_states, dtype=np.float64)
    state = initial.copy()
    alive = np.ones(state.shape[0], dtype=np.float64)
    running = np.zeros(state.shape[0], dtype=np.float64)
    out: dict[int, float] = {}
    reward_low, reward_high = _streamlift_reward_clip(dataset)
    for t in range(int(horizon)):
        arm = _streamlift_public_campaign_action(int(assigned_arm), t, dataset)
        features = _streamlift_dynamics_features(state)
        reward = np.clip(features @ models[arm]["reward"], reward_low, reward_high)
        hazard = np.clip(_sigmoid_array(features @ models[arm]["terminal_logit"]), 0.0, 0.95)
        running += (float(dataset.gamma) ** t) * alive * reward
        if (t + 1) in forecast_horizons:
            out[t + 1] = float(np.mean(running))
        delta = features @ models[arm]["delta"]
        state = _project_streamlift_state(state + delta, initial=initial, step=t + 1)
        alive *= 1.0 - hazard
    out[int(horizon)] = float(np.mean(running))
    return out


def _streamlift_dynamics_features(states: Array) -> Array:
    s = np.asarray(states, dtype=np.float64)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    n = s.shape[0]

    def col(index: int) -> Array:
        if index < s.shape[1]:
            return s[:, [index]]
        return np.zeros((n, 1), dtype=np.float64)

    selected = [0, 1, 2, 4, 5, 8, 9, 10, 11]
    squares = [col(index) ** 2 for index in selected]
    interactions = [
        col(0) * col(4),
        col(0) * col(8),
        col(1) * col(10),
        col(2) * col(9),
        col(4) * col(10),
        col(5) * col(6),
    ]
    return np.column_stack([np.ones(n, dtype=np.float64), s, *squares, *interactions])


def _project_streamlift_state(state: Array, *, initial: Array, step: int) -> Array:
    out = np.clip(np.asarray(state, dtype=np.float64), 0.0, 1.0)
    stable_cols = [3, 6, 8, 9, 10, 11]
    for col in stable_cols:
        if col < out.shape[1] and col < initial.shape[1]:
            out[:, col] = initial[:, col]
    if out.shape[1] > 7 and initial.shape[1] > 7:
        out[:, 7] = np.clip(initial[:, 7] + float(step) / 36.0, 0.0, 1.0)
    return out


def _streamlift_public_campaign_action(assigned_arm: int, time_index: int, dataset: LongitudinalDataset) -> int:
    mode = str(dataset.metadata_public.get("campaign_mode", dataset.metadata_public.get("streamlift_campaign_mode", "finite_campaign")))
    campaign_length = int(dataset.metadata_public.get("campaign_length", 1))
    if mode == "one_shot" and int(time_index) > 0:
        return 0
    if mode == "finite_campaign" and int(time_index) >= campaign_length:
        return 0
    return int(assigned_arm)


def _streamlift_observed_noncompliance(dataset: LongitudinalDataset) -> float:
    if dataset.assigned_arm is None or dataset.received_treatment is None:
        return 0.0
    first_idx = np.array([np.flatnonzero(dataset.unit_id == unit)[0] for unit in np.unique(dataset.unit_id)], dtype=np.int64)
    assigned = np.asarray(dataset.assigned_arm, dtype=np.float64)[first_idx]
    received = np.asarray(dataset.received_treatment, dtype=np.float64)[first_idx]
    return float(np.mean(assigned != received)) if assigned.size else 0.0


def _streamlift_reward_clip(dataset: LongitudinalDataset) -> tuple[float, float]:
    rewards = np.asarray(dataset.rewards, dtype=np.float64)
    if rewards.size == 0:
        return -1e6, 1e6
    spread = float(np.std(rewards))
    return float(np.min(rewards) - 2.0 * spread), float(np.max(rewards) + 2.0 * spread)


def _streamlift_ipw(dataset: LongitudinalDataset, *, normalize: bool) -> EstimatorResult:
    action = action_index(dataset.actions)
    rewards_by_unit = _unit_discounted_rewards(dataset, max_time=int(dataset.metadata_public["observed_horizon"]))
    first_rows = np.array([np.flatnonzero(dataset.unit_id == unit)[0] for unit in rewards_by_unit], dtype=np.int64)
    first_action = action[first_rows]
    first_behavior = np.asarray(dataset.behavior_propensity)[first_rows]
    values = np.asarray(list(rewards_by_unit.values()), dtype=np.float64)
    treat_w = (first_action == 1) / np.clip(first_behavior, 1e-12, np.inf)
    ctrl_w = (first_action == 0) / np.clip(first_behavior, 1e-12, np.inf)
    if normalize:
        treat_mean = float(np.sum(treat_w * values) / max(np.sum(treat_w), 1e-12))
        ctrl_mean = float(np.sum(ctrl_w * values) / max(np.sum(ctrl_w), 1e-12))
    else:
        treat_mean = float(np.mean(treat_w * values))
        ctrl_mean = float(np.mean(ctrl_w * values))
    observed_effect = treat_mean - ctrl_mean
    observed_horizon = max(1, int(dataset.metadata_public["observed_horizon"]))
    estimates = {}
    for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"):
        h = int(raw)
        estimates[f"effect_horizon_{h}"] = observed_effect * (1.0 - dataset.gamma**h) / (1.0 - dataset.gamma**observed_horizon)
        estimates[f"tot_effect_horizon_{h}"] = estimates[f"effect_horizon_{h}"]
    if _streamlift_include_infinite_horizon(dataset):
        scale = _infinite_discount_factor_sum(dataset.gamma) / _discount_factor_sum(dataset.gamma, observed_horizon)
        estimates["effect_horizon_infinite"] = observed_effect * scale
        estimates["tot_effect_horizon_infinite"] = estimates["effect_horizon_infinite"]
    if dataset.assigned_arm is not None and dataset.assignment_propensity is not None:
        first_assigned = np.asarray(dataset.assigned_arm)[first_rows].astype(np.int64)
        first_assignment_propensity = np.asarray(dataset.assignment_propensity)[first_rows]
        assign_treat_w = (first_assigned == 1) / np.clip(first_assignment_propensity, 1e-12, np.inf)
        assign_ctrl_w = (first_assigned == 0) / np.clip(1.0 - first_assignment_propensity, 1e-12, np.inf)
        if normalize:
            assign_treat_mean = float(np.sum(assign_treat_w * values) / max(np.sum(assign_treat_w), 1e-12))
            assign_ctrl_mean = float(np.sum(assign_ctrl_w * values) / max(np.sum(assign_ctrl_w), 1e-12))
        else:
            assign_treat_mean = float(np.mean(assign_treat_w * values))
            assign_ctrl_mean = float(np.mean(assign_ctrl_w * values))
        itt_observed = assign_treat_mean - assign_ctrl_mean
        for raw in str(dataset.metadata_public["forecast_horizons"]).split("|"):
            h = int(raw)
            estimates[f"itt_effect_horizon_{h}"] = itt_observed * (1.0 - dataset.gamma**h) / (1.0 - dataset.gamma**observed_horizon)
        if _streamlift_include_infinite_horizon(dataset):
            scale = _infinite_discount_factor_sum(dataset.gamma) / _discount_factor_sum(dataset.gamma, observed_horizon)
            estimates["itt_effect_horizon_infinite"] = itt_observed * scale
    weights = treat_w + ctrl_w
    return EstimatorResult(
        estimator="snipw" if normalize else "ipw",
        status="ok",
        estimates=estimates,
        diagnostics={"interval_available": 0, "ess_fraction": _ess_fraction(weights), "weight_p95": float(np.quantile(weights, 0.95))},
    )


def _target_action_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str, str]:
    if dataset.target_action_probabilities is not None:
        return np.asarray(dataset.target_action_probabilities, dtype=np.float64), "exact_discrete", "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        probs = target_policy_probabilities(
            dataset.family,
            str(policy_name),
            dataset.states,
            availability=dataset.action_available,
            time=dataset.time,
        )
        return probs, "exact_discrete", "named_policy"
    return np.asarray(dataset.target_actions, dtype=np.float64), "sampled_action", "sampled_action_only"


def _next_target_action_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str, str]:
    if dataset.next_target_action_probabilities is not None:
        return np.asarray(dataset.next_target_action_probabilities, dtype=np.float64), "exact_discrete", "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        probs = target_policy_probabilities(
            dataset.family,
            str(policy_name),
            dataset.next_states,
            time=np.asarray(dataset.time) + 1,
        )
        return probs, "exact_discrete", "named_policy"
    return np.asarray(dataset.next_target_actions, dtype=np.float64), "sampled_action", "sampled_action_only"


def _initial_target_action_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str, str]:
    if dataset.initial_action_probabilities is not None:
        return np.asarray(dataset.initial_action_probabilities, dtype=np.float64), "exact_discrete", "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        initial_idx = np.asarray(dataset.splits.get("initial", []), dtype=np.int64)
        availability = None
        if initial_idx.shape[0] == np.asarray(dataset.initial_states).shape[0] and initial_idx.size:
            availability = np.asarray(dataset.action_available)[initial_idx]
        probs = target_policy_probabilities(
            dataset.family,
            str(policy_name),
            dataset.initial_states,
            availability=availability,
            time=np.zeros(np.asarray(dataset.initial_states).shape[0], dtype=np.int64),
        )
        return probs, "exact_discrete", "named_policy"
    return np.asarray(dataset.initial_actions, dtype=np.float64), "sampled_action", "sampled_action_only"


def _expected_linear_q(states: Array, action_probabilities: Array, beta: Array) -> Array:
    probs = np.asarray(action_probabilities, dtype=np.float64)
    s = np.asarray(states, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != s.shape[0]:
        raise ValueError("action_probabilities must align with states.")
    action_eye = np.eye(probs.shape[1], dtype=np.float64)
    states_rep = np.repeat(s, probs.shape[1], axis=0)
    actions_rep = np.tile(action_eye, (s.shape[0], 1))
    q = (_feature_matrix(states_rep, actions_rep) @ beta).reshape(s.shape[0], probs.shape[1])
    return np.sum(probs * q, axis=1)


def _estimate_model_target_policy_value(
    model: Any,
    initial_states: Array,
    initial_action_probabilities: Array | None,
    fallback_initial_actions: Array,
) -> float:
    if initial_action_probabilities is None:
        return float(model.estimate_policy_value(initial_states, fallback_initial_actions))
    probs = np.asarray(initial_action_probabilities, dtype=np.float64)
    s = np.asarray(initial_states, dtype=np.float64)
    action_eye = np.eye(probs.shape[1], dtype=np.float64)
    states_rep = np.repeat(s, probs.shape[1], axis=0)
    actions_rep = np.tile(action_eye, (s.shape[0], 1))
    q = np.asarray(model.predict_q(states_rep, actions_rep), dtype=np.float64).reshape(s.shape[0], probs.shape[1])
    return float(np.mean(np.sum(probs * q, axis=1)))


def _combine_expectation_modes(*modes: str) -> str:
    unique = tuple(dict.fromkeys(modes))
    if len(unique) == 1:
        return unique[0]
    if "sampled_action" in unique:
        return "mixed_with_sampled_action_fallback"
    return "mixed"


def _combine_probability_storage(*storage: str) -> str:
    unique = tuple(dict.fromkeys(storage))
    if len(unique) == 1:
        return unique[0]
    if "sampled_action_only" in unique:
        return "mixed_with_sampled_action_fallback"
    return "mixed"


def _feature_matrix(states: Array, actions: Array) -> Array:
    s = np.asarray(states, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    return np.column_stack([np.ones(s.shape[0]), s, a, s[:, : min(4, s.shape[1])] * np.sum(a * np.arange(a.shape[1]), axis=1, keepdims=True)])


def _ridge_solve(x: Array, y: Array, *, ridge: float) -> Array:
    xtx = x.T @ x
    penalty = float(ridge) * np.eye(xtx.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.pinv(xtx + penalty) @ x.T @ np.asarray(y, dtype=np.float64)


def _logistic_ridge_fit(x: Array, y: Array, *, ridge: float, max_iter: int = 50) -> Array:
    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    beta = np.zeros(features.shape[1], dtype=np.float64)
    mean = float(np.clip(np.mean(target), 1e-4, 1.0 - 1e-4))
    beta[0] = np.log(mean / (1.0 - mean))
    penalty = float(ridge) * np.eye(features.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    for _ in range(int(max_iter)):
        logits = np.clip(features @ beta, -35.0, 35.0)
        probs = _sigmoid_array(logits)
        weights = np.clip(probs * (1.0 - probs), 1e-5, np.inf)
        gradient = features.T @ (probs - target) + penalty @ beta
        hessian = (features.T * weights) @ features + penalty
        step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return beta


def _sigmoid_array(values: Array) -> Array:
    z = np.clip(np.asarray(values, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _family_value_estimates(dataset: LongitudinalDataset, value: float) -> dict[str, float]:
    del dataset
    return {"policy_value": float(value)}


def _target_over_behavior_ratio(dataset: LongitudinalDataset) -> Array:
    return np.asarray(dataset.target_propensity_observed_action, dtype=np.float64) / np.asarray(dataset.behavior_propensity, dtype=np.float64)


def _fqe_terminals(dataset: LongitudinalDataset) -> Array:
    return np.maximum(np.asarray(dataset.terminals, dtype=np.float64), np.asarray(dataset.censoring, dtype=np.float64))


def _kaplan_meier_censor_survival(panel: Any, horizon: int) -> Array:
    """Marginal probability of remaining uncensored through the start of each interval."""
    survival = np.ones(int(horizon), dtype=np.float64)
    current = 1.0
    for t in range(int(horizon)):
        survival[t] = max(current, 0.05)
        at_risk = np.flatnonzero(panel.time == t)
        if at_risk.size == 0:
            continue
        censored = np.sum((panel.censored[at_risk] > 0.5) & (panel.event[at_risk] < 0.5))
        current *= max(0.0, 1.0 - float(censored) / float(at_risk.size))
    return survival


def _unit_discounted_rewards(dataset: LongitudinalDataset, *, max_time: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for unit in np.unique(dataset.unit_id):
        idx = np.flatnonzero(dataset.unit_id == unit)
        idx = idx[np.asarray(dataset.time)[idx] < max_time]
        rewards = np.asarray(dataset.rewards)[idx]
        times = np.asarray(dataset.time)[idx]
        out[int(unit)] = float(np.sum((dataset.gamma**times) * rewards))
    return out


def _unit_final_weights(dataset: LongitudinalDataset, ratios: Array) -> dict[int, float]:
    out: dict[int, float] = {}
    for unit in np.unique(dataset.unit_id):
        idx = np.flatnonzero(dataset.unit_id == unit)
        sorted_idx = idx[np.argsort(dataset.time[idx])]
        out[int(unit)] = float(np.prod(np.clip(ratios[sorted_idx], 0.0, 50.0)))
    return out


def _weighted_unit_values(dataset: LongitudinalDataset, ratios: Array, *, normalize: bool) -> dict[int, float]:
    unit_values = {}
    for unit in np.unique(dataset.unit_id):
        idx = np.flatnonzero(dataset.unit_id == unit)
        sorted_idx = idx[np.argsort(dataset.time[idx])]
        cumulative = 1.0
        total = 0.0
        total_weight = 0.0
        for row in sorted_idx:
            cumulative *= float(np.clip(ratios[row], 0.0, 50.0))
            weight = cumulative
            total += (dataset.gamma ** int(dataset.time[row])) * weight * float(dataset.rewards[row])
            total_weight += weight
        if normalize and total_weight > 0.0:
            total /= total_weight / max(1, len(sorted_idx))
        unit_values[int(unit)] = float(total)
    return unit_values


def _horizon(dataset: LongitudinalDataset) -> int:
    if "trajectory_horizon" in dataset.metadata_public:
        return int(dataset.metadata_public["trajectory_horizon"])
    if "long_horizon" in dataset.metadata_public:
        return int(dataset.metadata_public["long_horizon"])
    return int(np.max(dataset.time) + 1)


def _discount_factor_sum(gamma: float, horizon: int) -> float:
    return float(np.sum(float(gamma) ** np.arange(int(horizon))))


def _infinite_discount_factor_sum(gamma: float) -> float:
    return float(1.0 / max(1.0 - float(gamma), 1e-12))


def _streamlift_include_infinite_horizon(dataset: LongitudinalDataset) -> bool:
    value = dataset.metadata_public.get("infinite_horizon", False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _streamlift_infinite_rollout_horizon(dataset: LongitudinalDataset) -> int:
    raw = dataset.metadata_public.get("infinite_horizon_max_steps", "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(np.ceil(np.log(1e-5) / np.log(float(dataset.gamma)))) if float(dataset.gamma) > 0.0 else 1


def _ess_fraction(weights: Array) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or np.sum(w * w) <= 0.0:
        return 0.0
    return float((np.sum(w) ** 2) / (w.size * np.sum(w * w)))
