from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from causal_ope_benchmark.config import DomainScenario
from causal_ope_benchmark.contracts import contract_for_family
from causal_ope_benchmark.policies import CLINIC_ACTIONS, STREAM_ACTIONS, STREAMLIFT_ACTIONS, get_fixed_policy
from causal_ope_benchmark.types import (
    Array,
    BenchmarkProblem,
    EstimatorInformationContract,
    LongitudinalDataset,
    TruthBundle,
    chosen_probability,
    normalize_action_probs,
    one_hot,
)


@dataclass(frozen=True)
class _StreamLiftLatent:
    habit: Array
    taste_depth: Array
    price_sensitivity: Array
    competitor_pull: Array
    household_complexity: Array
    subgroup: Array


def make_streamlift_problem(
    *,
    sample_size: int,
    gamma: float,
    seed: int,
    scenario: DomainScenario,
    observed_horizon: int,
    target_policy: str = "moderate",
    forecast_horizons: Sequence[int] = (6, 12, 24, 36),
    long_horizon: int = 36,
    include_infinite_horizon: bool = False,
    infinite_horizon_max_steps: int = 240,
    mc_truth_rollouts: int = 96,
) -> BenchmarkProblem:
    """Create a short-panel streaming experiment forecasting problem."""
    effective_campaign_length = int(scenario.campaign_length)
    if scenario.streamlift_campaign_mode == "finite_campaign":
        effective_campaign_length = min(int(observed_horizon), int(scenario.campaign_length))
    elif scenario.streamlift_campaign_mode == "one_shot":
        effective_campaign_length = 1
    scenario = replace(scenario, campaign_length=effective_campaign_length)
    rng = np.random.default_rng(seed)
    latent, baseline = _streamlift_initial(sample_size, rng)
    target = get_fixed_policy("streamlift", target_policy)
    initial_availability = np.ones((sample_size, len(STREAMLIFT_ACTIONS)), dtype=np.float64)
    initial_target_probs = target.probabilities(
        baseline,
        initial_availability,
        np.zeros(sample_size, dtype=np.int64),
    )
    assignment_prob = _streamlift_assignment_prob(
        baseline,
        latent,
        scenario,
        target_treatment_prob=initial_target_probs[:, 1],
    )
    assigned = rng.binomial(1, assignment_prob).astype(np.int64)
    received = assigned.copy()
    noncompliance = rng.random(sample_size) < float(scenario.noncompliance_rate)
    received[noncompliance] = 1 - received[noncompliance]
    initial_target_actions = _sample_initial_target_actions(
        baseline,
        initial_availability,
        target,
        rng,
    )

    rows: list[dict[str, object]] = []
    for i in range(sample_size):
        state = baseline[i].copy()
        active = True
        for t in range(int(observed_horizon)):
            availability = np.ones(2, dtype=np.float64)
            action = _streamlift_campaign_action(int(received[i]), t, scenario)
            received_prob_1 = _received_treatment_probability(
                float(assignment_prob[i]),
                float(scenario.noncompliance_rate),
            )
            behavior_probs = _streamlift_campaign_adjusted_probs(
                np.array([1.0 - received_prob_1, received_prob_1], dtype=np.float64),
                t,
                scenario,
            )
            behavior_p = float(behavior_probs[action])
            raw_target_probs = target.probabilities(state.reshape(1, -1), availability.reshape(1, -1), np.array([t]))[0]
            target_probs = _streamlift_campaign_adjusted_probs(raw_target_probs, t, scenario)
            target_action = int(rng.choice(2, p=target_probs))
            next_state, reward, terminal, components = _streamlift_step(
                state,
                action,
                latent,
                unit=i,
                t=t,
                scenario=scenario,
                rng=rng,
                active=active,
            )
            next_raw_target_probs = target.probabilities(next_state.reshape(1, -1), availability.reshape(1, -1), np.array([t + 1]))[0]
            next_target_probs = _streamlift_campaign_adjusted_probs(next_raw_target_probs, t + 1, scenario)
            next_target_action = int(rng.choice(2, p=next_target_probs))
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminal": terminal,
                    "availability": availability,
                    "behavior_p": behavior_p,
                    "behavior_probs": behavior_probs,
                    "target_probs": target_probs,
                    "target_action": target_action,
                    "next_target_action": next_target_action,
                    "next_target_probs": next_target_probs,
                    "assigned_arm": int(assigned[i]),
                    "received_treatment": int(received[i]),
                    "assignment_propensity": float(assignment_prob[i]),
                    "components": components,
                }
            )
            state = next_state
            active = active and terminal < 0.5
            if not active:
                break

    dataset = _dataset_from_rows(
        rows=rows,
        name=f"streamlift_{_public_scenario_label(scenario.name)}_seed{seed}",
        family="streamlift",
        scenario=scenario,
        gamma=gamma,
        seed=seed,
        n_actions=2,
        initial_states=baseline,
        initial_action_indices=initial_target_actions,
        assigned_arm_by_unit=assigned,
        received_treatment_by_unit=received,
        assignment_propensity_by_unit=assignment_prob,
        metadata_public={
            "family": "streamlift",
            "scenario": _public_scenario_label(scenario.name),
            "sample_size": int(sample_size),
            "gamma": float(gamma),
            "observed_horizon": int(observed_horizon),
            "long_horizon": int(long_horizon),
            "forecast_horizons": "|".join(str(int(h)) for h in forecast_horizons),
            "horizon_modes": "finite|infinite" if include_infinite_horizon else "finite",
            "infinite_horizon": bool(include_infinite_horizon),
            "infinite_horizon_max_steps": (
                int(_streamlift_infinite_horizon_steps(gamma, infinite_horizon_max_steps)) if include_infinite_horizon else ""
            ),
            "assignment_regime": "randomized" if scenario.confounding == "randomized" else "covariate_based",
            "leaderboard_eligible": bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
            "state_features": "engagement|habit|payment_friction|plan_value|fatigue|risk_marker|subgroup|tenure|taste_depth|price_sensitivity|competitor_pull|household_complexity",
            "action_names": "|".join(STREAMLIFT_ACTIONS),
            "outcome_names": "revenue|retained|discount_cost",
            "primary_endpoint": "revenue",
            "secondary_endpoints": "retained|discount_cost",
            "endpoint_horizons": "|".join(str(int(h)) for h in forecast_horizons),
            "nonstationarity_regime": _nonstationarity_regime(scenario),
            "nonstationarity": "regime_shift" if scenario.nonstationarity else "none",
            "shift_time": 12 if scenario.nonstationarity else "",
            "action_constraints": "none",
            "target_policy_distance": _mean_behavior_target_tv(rows),
            "target_action_probability_fields": "|".join(f"target_action_prob_{i}" for i in range(len(STREAMLIFT_ACTIONS))),
            "next_target_action_probability_fields": "|".join(f"next_target_action_prob_{i}" for i in range(len(STREAMLIFT_ACTIONS))),
            "streamlift_campaign_mode": str(scenario.streamlift_campaign_mode),
            "campaign_mode": str(scenario.streamlift_campaign_mode),
            "campaign_length": int(scenario.campaign_length),
            "primary_observed_endpoint": "revenue",
            "decision_endpoint": "discounted_ltv",
            "surrogate_window": int(observed_horizon),
        },
    )
    truth = _streamlift_truth(
        baseline=baseline,
        latent=latent,
        scenario=scenario,
        gamma=gamma,
        forecast_horizons=tuple(int(h) for h in forecast_horizons),
        long_horizon=int(long_horizon),
        include_infinite_horizon=bool(include_infinite_horizon),
        infinite_horizon_max_steps=int(infinite_horizon_max_steps),
        mc_rollouts=int(mc_truth_rollouts),
        seed=seed + 17_000,
        dataset_name=dataset.name,
        assignment_prob=assignment_prob,
        actions=dataset.actions,
        target_propensity=dataset.target_propensity_observed_action,
        behavior_propensity=dataset.behavior_propensity,
    )
    return BenchmarkProblem(dataset=dataset, truth=truth)


def make_streamretain_problem(
    *,
    sample_size: int,
    gamma: float,
    seed: int,
    scenario: DomainScenario,
    target_policy: str = "moderate",
    horizon: int = 24,
    mc_truth_rollouts: int = 96,
) -> BenchmarkProblem:
    """Create a streaming subscription lifecycle OPE problem."""
    rng = np.random.default_rng(seed)
    baseline, latent = _streamretain_initial(sample_size, rng)
    target = get_fixed_policy("streamretain", target_policy)
    initial_target_actions = _sample_initial_target_actions(
        baseline,
        np.vstack([_stream_action_availability(state, scenario) for state in baseline]),
        target,
        rng,
    )
    rows = []
    for i in range(sample_size):
        state = baseline[i].copy()
        for t in range(int(horizon)):
            availability = _stream_action_availability(state, scenario)
            target_probs = target.probabilities(state.reshape(1, -1), availability.reshape(1, -1), np.array([t]))[0]
            behavior_probs = _stream_behavior_probs(state, availability, scenario, target_probs)
            action = int(rng.choice(len(STREAM_ACTIONS), p=behavior_probs))
            target_action = int(rng.choice(len(STREAM_ACTIONS), p=target_probs))
            action_dose = _stream_action_dose(state, action, scenario, rng)
            target_action_dose = _stream_action_dose_mean(state, target_action, scenario)
            next_state, reward, terminal, censored, components = _streamretain_step(
                state,
                action,
                action_dose,
                latent[i],
                t,
                scenario,
                rng,
            )
            next_availability = _stream_action_availability(next_state, scenario)
            next_target_probs = target.probabilities(next_state.reshape(1, -1), next_availability.reshape(1, -1), np.array([t + 1]))[0]
            next_target_action = int(rng.choice(len(STREAM_ACTIONS), p=next_target_probs))
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminal": terminal,
                    "availability": availability,
                    "behavior_p": behavior_probs[action],
                    "behavior_probs": behavior_probs,
                    "target_probs": target_probs,
                    "target_action": target_action,
                    "next_target_action": next_target_action,
                    "next_target_probs": next_target_probs,
                    "censoring": censored,
                    "action_dose": action_dose,
                    "target_action_dose": target_action_dose,
                    "dose_available": 1.0,
                    "components": components,
                }
            )
            state = next_state
            if terminal >= 0.5 or censored >= 0.5:
                break
    dataset = _dataset_from_rows(
        rows=rows,
        name=f"streamretain_{_public_scenario_label(scenario.name)}_{target_policy}_seed{seed}",
        family="streamretain",
        scenario=scenario,
        gamma=gamma,
        seed=seed,
        n_actions=len(STREAM_ACTIONS),
        initial_states=baseline,
        initial_action_indices=initial_target_actions,
        metadata_public={
            "family": "streamretain",
            "scenario": _public_scenario_label(scenario.name),
            "sample_size": int(sample_size),
            "gamma": float(gamma),
            "trajectory_horizon": int(horizon),
            "target_policy": str(target_policy),
            "leaderboard_eligible": bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
            "state_features": "engagement|tenure|price_sensitivity|plan_value|fatigue|churn_risk|subgroup|season",
            "action_names": "|".join(STREAM_ACTIONS),
            "outcome_names": "subscription_revenue|ad_revenue|intervention_cost|retained|fatigue|action_dose",
            "primary_endpoint": "subscription_revenue",
            "secondary_endpoints": "ad_revenue|intervention_cost|retained|fatigue",
            "endpoint_horizons": str(int(horizon)),
            "dose_field": "action_dose",
            "nonstationarity_regime": _nonstationarity_regime(scenario),
            "nonstationarity": "regime_shift" if scenario.nonstationarity else "none",
            "shift_time": 12 if scenario.nonstationarity else "",
            "action_constraints": "active" if scenario.action_constraints else "inactive",
            "target_policy_distance": _mean_behavior_target_tv(rows),
            "target_action_probability_fields": "|".join(f"target_action_prob_{i}" for i in range(len(STREAM_ACTIONS))),
            "next_target_action_probability_fields": "|".join(f"next_target_action_prob_{i}" for i in range(len(STREAM_ACTIONS))),
        },
    )
    truth = _policy_value_truth(
        family="streamretain",
        dataset_name=dataset.name,
        baseline=baseline,
        latent=latent,
        scenario=scenario,
        target_policy=target_policy,
        gamma=gamma,
        horizon=int(horizon),
        mc_rollouts=int(mc_truth_rollouts),
        seed=seed + 29_000,
    )
    truth.oracle_ratios["row_target_over_behavior"] = np.asarray(dataset.target_propensity_observed_action) / np.asarray(dataset.behavior_propensity)
    return BenchmarkProblem(dataset=dataset, truth=truth)


def make_clinic_dtr_problem(
    *,
    sample_size: int,
    gamma: float,
    seed: int,
    scenario: DomainScenario,
    target_policy: str = "safety_constrained",
    horizon: int = 24,
    mc_truth_rollouts: int = 96,
) -> BenchmarkProblem:
    """Create a cardiometabolic dynamic treatment-regime OPE problem."""
    rng = np.random.default_rng(seed)
    baseline, latent = _clinic_initial(sample_size, rng)
    target = get_fixed_policy("clinic_dtr", target_policy)
    initial_target_actions = _sample_initial_target_actions(
        baseline,
        np.vstack([_clinic_action_availability(state, scenario) for state in baseline]),
        target,
        rng,
    )
    rows = []
    for i in range(sample_size):
        state = baseline[i].copy()
        for t in range(int(horizon)):
            availability = _clinic_action_availability(state, scenario)
            target_probs = target.probabilities(state.reshape(1, -1), availability.reshape(1, -1), np.array([t]))[0]
            behavior_probs = _clinic_behavior_probs(state, availability, scenario, target_probs)
            action = int(rng.choice(len(CLINIC_ACTIONS), p=behavior_probs))
            target_action = int(rng.choice(len(CLINIC_ACTIONS), p=target_probs))
            action_dose = _clinic_action_dose(state, action, scenario, rng)
            target_action_dose = _clinic_action_dose_mean(state, target_action, scenario)
            next_state, reward, terminal, censored, components = _clinic_step(state, action, action_dose, latent[i], t, scenario, rng)
            next_availability = _clinic_action_availability(next_state, scenario)
            next_target_probs = target.probabilities(next_state.reshape(1, -1), next_availability.reshape(1, -1), np.array([t + 1]))[0]
            next_target_action = int(rng.choice(len(CLINIC_ACTIONS), p=next_target_probs))
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "terminal": terminal,
                    "availability": availability,
                    "behavior_p": behavior_probs[action],
                    "behavior_probs": behavior_probs,
                    "target_probs": target_probs,
                    "target_action": target_action,
                    "next_target_action": next_target_action,
                    "next_target_probs": next_target_probs,
                    "censoring": censored,
                    "action_dose": action_dose,
                    "target_action_dose": target_action_dose,
                    "dose_available": 1.0,
                    "components": components,
                }
            )
            state = next_state
            if terminal >= 0.5 or censored >= 0.5:
                break
    dataset = _dataset_from_rows(
        rows=rows,
        name=f"clinic_dtr_{_public_scenario_label(scenario.name)}_{target_policy}_seed{seed}",
        family="clinic_dtr",
        scenario=scenario,
        gamma=gamma,
        seed=seed,
        n_actions=len(CLINIC_ACTIONS),
        initial_states=baseline,
        initial_action_indices=initial_target_actions,
        metadata_public={
            "family": "clinic_dtr",
            "scenario": _public_scenario_label(scenario.name),
            "sample_size": int(sample_size),
            "gamma": float(gamma),
            "trajectory_horizon": int(horizon),
            "target_policy": str(target_policy),
            "leaderboard_eligible": bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
            "state_features": "age|diabetes|ascvd|blood_pressure|ldl|hba1c|kidney_risk|adherence|toxicity|subgroup",
            "action_names": "|".join(CLINIC_ACTIONS),
            "outcome_names": "qaly|event_free|toxicity|biomarker|dose|action_dose",
            "primary_endpoint": "qaly",
            "secondary_endpoints": "event_free|toxicity|biomarker|dose",
            "endpoint_horizons": str(int(horizon)),
            "dose_field": "action_dose",
            "nonstationarity_regime": _nonstationarity_regime(scenario),
            "nonstationarity": "regime_shift" if scenario.nonstationarity else "none",
            "shift_time": 12 if scenario.nonstationarity else "",
            "action_constraints": "active" if scenario.action_constraints else "inactive",
            "target_policy_distance": _mean_behavior_target_tv(rows),
            "target_action_probability_fields": "|".join(f"target_action_prob_{i}" for i in range(len(CLINIC_ACTIONS))),
            "next_target_action_probability_fields": "|".join(f"next_target_action_prob_{i}" for i in range(len(CLINIC_ACTIONS))),
        },
    )
    truth = _policy_value_truth(
        family="clinic_dtr",
        dataset_name=dataset.name,
        baseline=baseline,
        latent=latent,
        scenario=scenario,
        target_policy=target_policy,
        gamma=gamma,
        horizon=int(horizon),
        mc_rollouts=int(mc_truth_rollouts),
        seed=seed + 31_000,
    )
    truth.oracle_ratios["row_target_over_behavior"] = np.asarray(dataset.target_propensity_observed_action) / np.asarray(dataset.behavior_propensity)
    return BenchmarkProblem(dataset=dataset, truth=truth)


def _sample_initial_target_actions(
    states: Array,
    availability: Array,
    policy,
    rng: np.random.Generator,
) -> Array:
    probs = policy.probabilities(np.asarray(states), np.asarray(availability), np.zeros(np.asarray(states).shape[0], dtype=np.int64))
    return np.asarray([rng.choice(probs.shape[1], p=row) for row in probs], dtype=np.int64)


def _received_treatment_probability(assignment_prob: float, noncompliance_rate: float) -> float:
    p_assign = float(assignment_prob)
    p_flip = float(noncompliance_rate)
    return float(np.clip(p_assign * (1.0 - p_flip) + (1.0 - p_assign) * p_flip, 1e-12, 1.0 - 1e-12))


def _dataset_from_rows(
    *,
    rows: list[dict[str, object]],
    name: str,
    family: str,
    scenario: DomainScenario,
    gamma: float,
    seed: int,
    n_actions: int,
    initial_states: Array,
    initial_action_indices: Array,
    metadata_public: dict[str, object],
    assigned_arm_by_unit: Array | None = None,
    received_treatment_by_unit: Array | None = None,
    assignment_propensity_by_unit: Array | None = None,
) -> LongitudinalDataset:
    if not rows:
        raise ValueError("simulator produced no rows.")
    unit_id = np.asarray([row["unit"] for row in rows], dtype=np.int64)
    time = np.asarray([row["time"] for row in rows], dtype=np.int64)
    states = np.vstack([row["state"] for row in rows]).astype(np.float64)
    next_states = np.vstack([row["next_state"] for row in rows]).astype(np.float64)
    state_mask = _missingness_mask(states, scenario, np.random.default_rng(seed + 41_000))
    observed_states = states.copy()
    observed_next_states = next_states.copy()
    observed_states[state_mask > 0.0] = 0.0
    next_mask = _missingness_mask(next_states, scenario, np.random.default_rng(seed + 42_000))
    observed_next_states[next_mask > 0.0] = 0.0
    action_idx = np.asarray([row["action"] for row in rows], dtype=np.int64)
    target_idx = np.asarray([row["target_action"] for row in rows], dtype=np.int64)
    next_target_idx = np.asarray([row["next_target_action"] for row in rows], dtype=np.int64)
    target_probs = np.vstack([row["target_probs"] for row in rows]).astype(np.float64)
    next_target_probs = np.vstack([row.get("next_target_probs", row["target_probs"]) for row in rows]).astype(np.float64)
    has_dose = any("action_dose" in row or "target_action_dose" in row for row in rows)
    action_dose = None
    target_action_dose = None
    dose_available = None
    if has_dose:
        action_dose = np.asarray([row.get("action_dose", 0.0) for row in rows], dtype=np.float64)
        target_action_dose = np.asarray(
            [
                row.get(
                    "target_action_dose",
                    float(action_dose[i]) if int(target_idx[i]) == int(action_idx[i]) else 0.0,
                )
                for i, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        dose_available = np.asarray([row.get("dose_available", 1.0) for row in rows], dtype=np.float64)
    censoring = np.asarray([row.get("censoring", 0.0) for row in rows], dtype=np.float64)
    behavior_propensity = np.clip(np.asarray([row["behavior_p"] for row in rows], dtype=np.float64), 1e-12, np.inf)
    actions = one_hot(action_idx, n_actions)
    target_actions = one_hot(target_idx, n_actions)
    target_propensity_observed_action = chosen_probability(target_probs, actions)
    outcome_names = sorted({name for row in rows for name in dict(row["components"]).keys()})
    components = {}
    for key in outcome_names:
        components[key] = np.asarray([dict(row["components"]).get(key, 0.0) for row in rows], dtype=np.float64)
    if action_dose is not None:
        components["action_dose"] = action_dose
    if target_action_dose is not None:
        components["target_action_dose"] = target_action_dose
    for action in range(n_actions):
        components[f"target_action_prob_{action}"] = target_probs[:, action].astype(np.float64)
        components[f"next_target_action_prob_{action}"] = next_target_probs[:, action].astype(np.float64)
    n = len(rows)
    train = np.flatnonzero((unit_id % 5) != 0)
    eval_idx = np.flatnonzero((unit_id % 5) == 0)
    initial_rows = np.array([np.flatnonzero(unit_id == i)[0] for i in np.unique(unit_id)], dtype=np.int64)
    initial_target_probs = target_probs[initial_rows].astype(np.float64)
    assigned_arm = None
    received_treatment = None
    assignment_propensity = None
    if assigned_arm_by_unit is not None:
        unit_assigned = np.asarray(assigned_arm_by_unit, dtype=np.float64)
        assigned_arm = unit_assigned[unit_id]
    if received_treatment_by_unit is not None:
        unit_received = np.asarray(received_treatment_by_unit, dtype=np.float64)
        received_treatment = unit_received[unit_id]
    if assignment_propensity_by_unit is not None:
        unit_assignment_propensity = np.asarray(assignment_propensity_by_unit, dtype=np.float64)
        assignment_propensity = unit_assignment_propensity[unit_id]
    return LongitudinalDataset(
        name=name,
        family=family,  # type: ignore[arg-type]
        scenario=str(metadata_public["scenario"]),
        unit_id=unit_id,
        time=time,
        states=observed_states,
        actions=actions,
        rewards=np.asarray([row["reward"] for row in rows], dtype=np.float64),
        next_states=observed_next_states,
        terminals=np.asarray([row["terminal"] for row in rows], dtype=np.float64),
        action_available=np.vstack([row["availability"] for row in rows]).astype(np.float64),
        missingness_mask=state_mask,
        censoring=censoring,
        behavior_propensity=behavior_propensity,
        target_actions=target_actions,
        target_propensity=chosen_probability(target_probs, target_actions),
        target_propensity_observed_action=target_propensity_observed_action,
        next_target_actions=one_hot(next_target_idx, n_actions),
        gamma=float(gamma),
        seed=int(seed),
        splits={"train": train, "eval": eval_idx, "initial": initial_rows[:n]},
        initial_states=initial_states.astype(np.float64),
        initial_actions=one_hot(initial_action_indices, n_actions),
        outcome_components=components,
        metadata_public=dict(metadata_public),
        information_contract=_contract_with_public_metadata(family, metadata_public),
        assigned_arm=assigned_arm,
        received_treatment=received_treatment,
        assignment_propensity=assignment_propensity,
        target_action_probabilities=target_probs,
        next_target_action_probabilities=next_target_probs,
        initial_action_probabilities=initial_target_probs,
        action_dose=action_dose,
        target_action_dose=target_action_dose,
        dose_available=dose_available,
    )


def _contract_with_public_metadata(family: str, metadata_public: dict[str, object]) -> EstimatorInformationContract:
    base = contract_for_family(family)
    visible_metadata = tuple(dict.fromkeys((*base.visible_metadata, *metadata_public.keys())))
    return replace(base, visible_metadata=visible_metadata)


def _nonstationarity_regime(scenario: DomainScenario) -> str:
    return "regime_shift_after_t12" if scenario.nonstationarity else "stationary"


def _mean_behavior_target_tv(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    behavior = np.vstack([row["behavior_probs"] for row in rows]).astype(np.float64)
    target = np.vstack([row["target_probs"] for row in rows]).astype(np.float64)
    return float(np.mean(0.5 * np.sum(np.abs(behavior - target), axis=1)))


def _calibrated_binary_behavior_prob(target_prob: Array, nominal_prob: Array, *, desired_tv: float) -> Array:
    target = np.clip(np.asarray(target_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    nominal = np.clip(np.asarray(nominal_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    desired = float(np.clip(desired_tv, 0.0, 1.0))
    out = np.empty_like(target)
    for i, p_target in enumerate(target):
        if desired <= 0.0:
            out[i] = p_target
            continue
        reference = float(nominal[i])
        ref_tv = abs(reference - float(p_target))
        if ref_tv + 1e-12 < desired:
            far_low = abs(0.0 - float(p_target))
            far_high = abs(1.0 - float(p_target))
            reference = 0.0 if far_low >= far_high else 1.0
            ref_tv = max(far_low, far_high)
        if ref_tv <= 1e-12:
            out[i] = p_target
            continue
        alpha = min(1.0, desired / ref_tv)
        out[i] = (1.0 - alpha) * p_target + alpha * reference
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def _calibrated_behavior_probs(target_probs: Array, nominal_probs: Array, availability: Array, *, desired_tv: float) -> Array:
    target = normalize_action_probs(np.asarray(target_probs, dtype=np.float64).reshape(1, -1), np.asarray(availability, dtype=np.float64).reshape(1, -1))[0]
    nominal = normalize_action_probs(np.asarray(nominal_probs, dtype=np.float64).reshape(1, -1), np.asarray(availability, dtype=np.float64).reshape(1, -1))[0]
    desired = float(np.clip(desired_tv, 0.0, 1.0))
    if desired <= 0.0:
        return target
    reference = nominal
    ref_tv = _policy_tv(reference, target)
    if ref_tv + 1e-12 < desired:
        reference = _farthest_available_policy(target, availability)
        ref_tv = _policy_tv(reference, target)
    if ref_tv <= 1e-12:
        return target
    alpha = min(1.0, desired / ref_tv)
    mixed = (1.0 - alpha) * target + alpha * reference
    return normalize_action_probs(mixed.reshape(1, -1), np.asarray(availability, dtype=np.float64).reshape(1, -1))[0]


def _policy_tv(first: Array, second: Array) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))))


def _farthest_available_policy(target_probs: Array, availability: Array) -> Array:
    target = np.asarray(target_probs, dtype=np.float64).reshape(-1)
    available = np.asarray(availability, dtype=np.float64).reshape(-1) > 0.0
    out = np.zeros_like(target)
    if not np.any(available):
        out[int(np.argmin(target))] = 1.0
        return out
    available_idx = np.flatnonzero(available)
    out[int(available_idx[np.argmin(target[available_idx])])] = 1.0
    return out


def _streamlift_initial(n: int, rng: np.random.Generator) -> tuple[_StreamLiftLatent, Array]:
    subgroup = rng.binomial(1, 0.42, size=n).astype(np.float64)
    habit = np.clip(rng.beta(3.5, 2.5, size=n) - 0.08 * subgroup, 0.02, 0.98)
    taste_depth = np.clip(rng.beta(2.8, 2.0, size=n), 0.02, 0.98)
    price_sens = np.clip(rng.beta(2.0 + subgroup, 3.0, size=n), 0.02, 0.98)
    competitor = np.clip(rng.beta(1.8, 4.0, size=n) + 0.15 * subgroup, 0.02, 0.98)
    household = np.clip(rng.poisson(2.2, size=n) / 5.0, 0.0, 1.0)
    engagement = np.clip(0.45 * habit + 0.35 * taste_depth + rng.normal(0, 0.10, size=n), 0.02, 0.98)
    payment = np.clip(0.30 * price_sens + 0.20 * competitor + rng.normal(0, 0.08, size=n), 0.02, 0.98)
    plan_value = np.clip(0.35 + 0.25 * household - 0.12 * price_sens + rng.normal(0, 0.08, size=n), 0.02, 0.98)
    fatigue = np.clip(0.15 + 0.20 * competitor + rng.normal(0, 0.06, size=n), 0.0, 0.95)
    risk_marker = np.clip(0.35 * price_sens + 0.40 * competitor - 0.25 * habit + rng.normal(0, 0.08, size=n), 0.0, 1.0)
    tenure = np.clip(rng.gamma(2.0, 0.18, size=n), 0.0, 1.0)
    baseline = np.column_stack(
        [
            engagement,
            habit,
            payment,
            plan_value,
            fatigue,
            risk_marker,
            subgroup,
            tenure,
            taste_depth,
            price_sens,
            competitor,
            household,
        ]
    )
    latent = _StreamLiftLatent(
        habit=habit,
        taste_depth=taste_depth,
        price_sensitivity=price_sens,
        competitor_pull=competitor,
        household_complexity=household,
        subgroup=subgroup,
    )
    return latent, baseline


def _streamlift_assignment_prob(
    baseline: Array,
    latent: _StreamLiftLatent,
    scenario: DomainScenario,
    *,
    target_treatment_prob: Array,
) -> Array:
    if scenario.confounding == "randomized":
        nominal_received = np.full(baseline.shape[0], 0.5, dtype=np.float64)
    else:
        scale = {"good": 0.7, "moderate": 1.2, "weak": 2.0, "structural_gap": 3.0}[scenario.overlap]
        score = -0.20 + scale * (0.9 * baseline[:, 5] + 0.5 * baseline[:, 2] - 0.4 * baseline[:, 0])
        if scenario.confounding == "latent":
            score += 1.0 * latent.competitor_pull - 0.8 * latent.habit
        lower = {"good": 0.15, "moderate": 0.08, "weak": 0.02, "structural_gap": 0.005}[scenario.overlap]
        upper = 1.0 - lower
        nominal_received = np.clip(_sigmoid(score), lower, upper)
    received_prob = _calibrated_binary_behavior_prob(
        np.asarray(target_treatment_prob, dtype=np.float64),
        nominal_received,
        desired_tv=float(scenario.target_policy_distance),
    )
    flip = float(scenario.noncompliance_rate)
    if flip >= 0.5:
        return np.clip(received_prob, 1e-6, 1.0 - 1e-6)
    assignment_prob = (received_prob - flip) / max(1e-12, 1.0 - 2.0 * flip)
    lower = {"good": 0.02, "moderate": 0.01, "weak": 0.005, "structural_gap": 0.001}[scenario.overlap]
    return np.clip(assignment_prob, lower, 1.0 - lower)


def _streamlift_campaign_action(assigned_action: int, t: int, scenario: DomainScenario) -> int:
    mode = str(scenario.streamlift_campaign_mode)
    if mode == "one_shot" and int(t) > 0:
        return 0
    if mode == "finite_campaign" and int(t) >= int(scenario.campaign_length):
        return 0
    return int(assigned_action)


def _streamlift_campaign_adjusted_probs(probs: Array, t: int, scenario: DomainScenario) -> Array:
    arr = np.asarray(probs, dtype=np.float64).reshape(-1).copy()
    mode = str(scenario.streamlift_campaign_mode)
    if (mode == "one_shot" and int(t) > 0) or (mode == "finite_campaign" and int(t) >= int(scenario.campaign_length)):
        arr[:] = 0.0
        arr[0] = 1.0
        return arr
    return arr / max(float(np.sum(arr)), 1e-12)


def _streamlift_step(
    state: Array,
    action: int,
    latent: _StreamLiftLatent,
    unit: int,
    t: int,
    scenario: DomainScenario,
    rng: np.random.Generator,
    active: bool,
) -> tuple[Array, float, float, dict[str, float]]:
    if not active:
        return state.copy(), 0.0, 1.0, {"revenue": 0.0, "retained": 0.0, "discount_cost": 0.0}
    subgroup = float(state[6]) if state.shape[0] > 6 else float(latent.subgroup[unit])
    effect = _streamlift_effect(action, t, scenario, subgroup)
    drift = 0.06 if scenario.nonstationarity and t >= 12 else 0.0
    taste_depth = float(state[8]) if state.shape[0] > 8 else float(latent.taste_depth[unit])
    price_sensitivity = float(state[9]) if state.shape[0] > 9 else float(latent.price_sensitivity[unit])
    competitor_pull = float(state[10]) if state.shape[0] > 10 else float(latent.competitor_pull[unit])
    household_complexity = float(state[11]) if state.shape[0] > 11 else float(latent.household_complexity[unit])
    engagement = np.clip(
        0.70 * state[0] + 0.18 * taste_depth + effect["engagement"] - 0.04 * state[4] - 0.03 * drift + rng.normal(0, 0.035),
        0.0,
        1.0,
    )
    habit = np.clip(0.85 * state[1] + 0.10 * engagement + effect["habit"] + rng.normal(0, 0.025), 0.0, 1.0)
    fatigue = np.clip(0.78 * state[4] + 0.11 * action + 0.02 * competitor_pull + rng.normal(0, 0.025), 0.0, 1.0)
    payment = np.clip(0.88 * state[2] - 0.05 * action + 0.02 * price_sensitivity + rng.normal(0, 0.025), 0.0, 1.0)
    risk = np.clip(0.45 * competitor_pull + 0.30 * payment + 0.25 * fatigue - 0.35 * habit + drift + rng.normal(0, 0.035), 0.0, 1.0)
    hazard = _sigmoid(-2.7 + 2.1 * risk + 0.9 * payment + 0.7 * competitor_pull - 1.4 * habit - effect["retention"])
    churn = float(rng.random() < hazard)
    retained = 1.0 - churn
    discount_cost = 2.2 * action * (0.6 + price_sensitivity)
    revenue = retained * (10.5 + 5.5 * state[3] + 2.5 * engagement - discount_cost)
    next_state = np.array(
        [
            engagement,
            habit,
            payment,
            state[3],
            fatigue,
            risk,
            state[6],
            min(1.0, state[7] + 1.0 / 36.0),
            taste_depth,
            price_sensitivity,
            competitor_pull,
            household_complexity,
        ]
    )
    return next_state, float(revenue), churn, {"revenue": float(revenue), "retained": retained, "discount_cost": float(discount_cost)}


def _streamlift_effect(action: int, t: int, scenario: DomainScenario, subgroup: float) -> dict[str, float]:
    del t
    if action == 0:
        return {"engagement": 0.0, "habit": 0.0, "retention": 0.0}
    hetero = 1.0 + float(scenario.subgroup_heterogeneity) * (0.5 - subgroup) * 0.35
    if scenario.delay_pattern == "immediate":
        engagement, habit, retention = 0.14, 0.035, 0.42
    elif scenario.delay_pattern == "delayed_benefit":
        engagement, habit, retention = 0.030, 0.035, 0.065
    elif scenario.delay_pattern == "short_harm_long_benefit":
        engagement, habit, retention = -0.030, 0.055, 0.095
    else:
        engagement, habit, retention = 0.075, -0.045, -0.085
    if scenario.surrogate_validity == "weak":
        engagement *= 1.2
        retention *= 0.45
    elif scenario.surrogate_validity == "misleading":
        engagement = abs(engagement) + 0.05
        retention *= -0.55
    elif scenario.surrogate_validity == "sign_reversal":
        engagement = abs(engagement) + 0.06
        habit *= -0.50
        retention = -abs(retention) - 0.08
    return {"engagement": engagement * hetero, "habit": habit * hetero, "retention": retention * hetero}


def _streamlift_truth(
    *,
    baseline: Array,
    latent: _StreamLiftLatent,
    scenario: DomainScenario,
    gamma: float,
    forecast_horizons: tuple[int, ...],
    long_horizon: int,
    include_infinite_horizon: bool,
    infinite_horizon_max_steps: int,
    mc_rollouts: int,
    seed: int,
    dataset_name: str,
    assignment_prob: Array,
    actions: Array,
    target_propensity: Array,
    behavior_propensity: Array,
) -> TruthBundle:
    rng = np.random.default_rng(seed)
    infinite_steps = _streamlift_infinite_horizon_steps(gamma, infinite_horizon_max_steps) if include_infinite_horizon else int(long_horizon)
    simulation_horizon = max(int(long_horizon), int(infinite_steps))
    values_by_action = {0: {h: [] for h in forecast_horizons}, 1: {h: [] for h in forecast_horizons}}
    infinite_values_by_action = {0: [], 1: []}
    survival_by_action = {0: np.zeros(long_horizon, dtype=np.float64), 1: np.zeros(long_horizon, dtype=np.float64)}
    subgroup_values = {0: {h: {0: [], 1: []} for h in forecast_horizons}, 1: {h: {0: [], 1: []} for h in forecast_horizons}}
    short_effects = []
    for action in (0, 1):
        for i in range(baseline.shape[0]):
            discounted = {h: [] for h in forecast_horizons}
            retained_curve = []
            for _ in range(mc_rollouts):
                state = baseline[i].copy()
                active = True
                rewards = []
                retained = []
                for t in range(simulation_horizon):
                    campaign_action = _streamlift_campaign_action(action, t, scenario)
                    state, reward, terminal, _ = _streamlift_step(state, campaign_action, latent, i, t, scenario, rng, active)
                    active = active and terminal < 0.5
                    rewards.append(reward)
                    if t < long_horizon:
                        retained.append(float(active))
                rewards_arr = np.asarray(rewards, dtype=np.float64)
                for h in forecast_horizons:
                    discounted[h].append(_discounted_sum(rewards_arr[:h], gamma))
                if include_infinite_horizon:
                    infinite_values_by_action[action].append(_discounted_sum(rewards_arr, gamma))
                retained_curve.append(retained)
            for h in forecast_horizons:
                mean_val = float(np.mean(discounted[h]))
                values_by_action[action][h].append(mean_val)
                subgroup_values[action][h][int(baseline[i, 6])].append(mean_val)
            survival_by_action[action] += np.mean(np.asarray(retained_curve, dtype=np.float64), axis=0)
    for action in (0, 1):
        survival_by_action[action] /= baseline.shape[0]
    effects = {}
    standard_errors = {}
    for h in forecast_horizons:
        treatment_effect = float(np.mean(values_by_action[1][h]) - np.mean(values_by_action[0][h]))
        effects[f"effect_horizon_{h}"] = treatment_effect
        effects[f"tot_effect_horizon_{h}"] = treatment_effect
        effects[f"itt_effect_horizon_{h}"] = float((1.0 - 2.0 * scenario.noncompliance_rate) * treatment_effect)
        standard_errors[f"value_treatment_horizon_{h}"] = _se(values_by_action[1][h])
        standard_errors[f"value_control_horizon_{h}"] = _se(values_by_action[0][h])
        effect_se = float(np.sqrt(_se(values_by_action[1][h]) ** 2 + _se(values_by_action[0][h]) ** 2))
        standard_errors[f"effect_horizon_{h}"] = effect_se
        standard_errors[f"tot_effect_horizon_{h}"] = effect_se
        standard_errors[f"itt_effect_horizon_{h}"] = abs(1.0 - 2.0 * scenario.noncompliance_rate) * effect_se
    if include_infinite_horizon:
        infinite_treatment = float(np.mean(infinite_values_by_action[1]))
        infinite_control = float(np.mean(infinite_values_by_action[0]))
        infinite_effect = infinite_treatment - infinite_control
        effects["effect_horizon_infinite"] = infinite_effect
        effects["tot_effect_horizon_infinite"] = infinite_effect
        effects["itt_effect_horizon_infinite"] = float((1.0 - 2.0 * scenario.noncompliance_rate) * infinite_effect)
        standard_errors["value_treatment_horizon_infinite"] = _se(infinite_values_by_action[1])
        standard_errors["value_control_horizon_infinite"] = _se(infinite_values_by_action[0])
        infinite_effect_se = float(np.sqrt(_se(infinite_values_by_action[1]) ** 2 + _se(infinite_values_by_action[0]) ** 2))
        standard_errors["effect_horizon_infinite"] = infinite_effect_se
        standard_errors["tot_effect_horizon_infinite"] = infinite_effect_se
        standard_errors["itt_effect_horizon_infinite"] = abs(1.0 - 2.0 * scenario.noncompliance_rate) * infinite_effect_se
    short_effects.append(effects[f"effect_horizon_{min(forecast_horizons)}"])
    long_effect = effects[f"effect_horizon_{max(forecast_horizons)}"]
    del assignment_prob, actions
    finite_values = {
        f"value_treatment_horizon_{h}": float(np.mean(values_by_action[1][h]))
        for h in forecast_horizons
    } | {
        f"value_control_horizon_{h}": float(np.mean(values_by_action[0][h]))
        for h in forecast_horizons
    }
    if include_infinite_horizon:
        finite_values.update(
            {
                "value_treatment_horizon_infinite": float(np.mean(infinite_values_by_action[1])),
                "value_control_horizon_infinite": float(np.mean(infinite_values_by_action[0])),
            }
        )
    return TruthBundle(
        dataset_name=dataset_name,
        family="streamlift",
        values=finite_values,
        effects=effects | {"surrogate_bias_horizon_long": float(short_effects[0] - long_effect)},
        survival_curves={
            "survival_control": survival_by_action[0],
            "survival_treatment": survival_by_action[1],
        },
        subgroup_effects=_streamlift_subgroup_effects(subgroup_values, forecast_horizons),
        oracle_ratios={"row_target_over_behavior": np.asarray(target_propensity) / np.asarray(behavior_propensity)},
        latent_parameters={"latent_dim": 6},
        target_mc_values={key: value for key, value in effects.items() if key.startswith("effect_horizon_")},
        target_standard_errors=standard_errors,
        truth_noise_floor=_truth_noise_floor({**effects, **finite_values}),
        mc_rollouts=int(mc_rollouts),
        private_metadata={
            "scenario_private_name": scenario.name,
            "surrogate_validity": scenario.surrogate_validity,
            "confounding": scenario.confounding,
            "overlap": scenario.overlap,
            "infinite_horizon_truncation_steps": int(infinite_steps) if include_infinite_horizon else "",
            "surrogate_bias_horizon_long": float(short_effects[0] - long_effect),
        },
        leaderboard_eligible=bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
    )


def _streamlift_infinite_horizon_steps(gamma: float, requested_steps: int) -> int:
    discount = float(gamma)
    requested = int(requested_steps)
    if discount <= 0.0:
        return max(1, requested)
    truncation = int(np.ceil(np.log(1e-5) / np.log(discount)))
    return max(1, requested, truncation)


def _streamlift_subgroup_effects(
    subgroup_values: dict[int, dict[int, dict[int, list[float]]]],
    forecast_horizons: tuple[int, ...],
) -> dict[str, float]:
    effects: dict[str, float] = {}
    final_horizon = max(forecast_horizons)
    for horizon in forecast_horizons:
        for group in (0, 1):
            control = subgroup_values[0][horizon][group]
            treatment = subgroup_values[1][horizon][group]
            if not control or not treatment:
                continue
            value = float(np.mean(treatment) - np.mean(control))
            effects[f"subgroup_{group}_effect_horizon_{horizon}"] = value
            if horizon == final_horizon:
                effects[f"subgroup_{group}_effect"] = value
    return effects


def _streamretain_initial(n: int, rng: np.random.Generator) -> tuple[Array, Array]:
    subgroup = rng.binomial(1, 0.38, size=n).astype(np.float64)
    price = np.clip(rng.beta(2.2 + subgroup, 3.0, size=n), 0.01, 0.99)
    latent_loyalty = np.clip(rng.beta(3.0, 2.4, size=n) - 0.12 * subgroup, 0.01, 0.99)
    competitor = np.clip(rng.beta(1.8, 4.0, size=n) + 0.12 * subgroup, 0.01, 0.99)
    engagement = np.clip(0.55 * latent_loyalty + rng.normal(0, 0.12, size=n), 0.01, 0.99)
    tenure = np.clip(rng.gamma(2.0, 0.16, size=n), 0.0, 1.0)
    plan = np.clip(0.40 + 0.25 * rng.random(n) - 0.10 * price, 0.05, 0.95)
    fatigue = np.clip(0.15 + 0.15 * competitor + rng.normal(0, 0.06, size=n), 0.0, 0.95)
    risk = np.clip(0.45 * competitor + 0.30 * price - 0.35 * latent_loyalty + rng.normal(0, 0.08, size=n), 0.0, 1.0)
    season = rng.random(n)
    states = np.column_stack([engagement, tenure, price, plan, fatigue, risk, subgroup, season])
    latent = np.column_stack([latent_loyalty, competitor, price, subgroup])
    return states, latent


def _stream_action_availability(state: Array, scenario: DomainScenario | None = None) -> Array:
    if scenario is not None and not scenario.action_constraints:
        return np.ones(len(STREAM_ACTIONS), dtype=np.float64)
    avail = np.ones(len(STREAM_ACTIONS), dtype=np.float64)
    if state[4] > 0.82:
        avail[2] = 0.0
    if state[2] < 0.25:
        avail[3] = 0.0
        avail[5] = 0.0
    if state[5] < 0.40:
        avail[4] = 0.0
        avail[8] = 0.0
    if state[3] < 0.30:
        avail[6] = 0.0
    return avail


def _stream_behavior_probs(state: Array, availability: Array, scenario: DomainScenario, target_probs: Array) -> Array:
    base = np.array([0.58, 0.10, 0.08, 0.07, 0.04, 0.03, 0.03, 0.04, 0.03], dtype=np.float64)
    risk_shift = float(state[5])
    base[1] += 0.08 * risk_shift
    base[3] += 0.10 * state[2]
    base[8] += 0.06 * (risk_shift > 0.70)
    if scenario.overlap == "weak":
        base[0] += 0.25
        base[3:] *= 0.55
    elif scenario.overlap == "moderate":
        base[0] += 0.10
        base[1:] *= 0.90
    if scenario.confounding == "latent":
        base[3] += 0.08 * risk_shift
    nominal = normalize_action_probs(base.reshape(1, -1), availability.reshape(1, -1))[0]
    return _calibrated_behavior_probs(
        np.asarray(target_probs, dtype=np.float64),
        nominal,
        availability,
        desired_tv=float(scenario.target_policy_distance),
    )


def _stream_action_dose(state: Array, action: int, scenario: DomainScenario, rng: np.random.Generator) -> float:
    mean = _stream_action_dose_mean(state, action, scenario)
    if action == 0:
        return 0.0
    return float(np.clip(mean + rng.normal(0.0, 0.08), 0.05, 1.35))


def _stream_action_dose_mean(state: Array, action: int, scenario: DomainScenario) -> float:
    if action == 0:
        return 0.0
    base = np.array([0.0, 0.45, 0.55, 0.85, 0.50, 0.65, 0.40, 0.55, 0.75], dtype=np.float64)[action]
    risk_boost = 0.15 * float(state[5] > 0.65)
    price_boost = 0.10 * float(action in {3, 5, 8}) * float(state[2] > 0.55)
    if not scenario.action_constraints:
        base *= 0.95
    return float(np.clip(base + risk_boost + price_boost, 0.05, 1.35))


def _streamretain_step(
    state: Array,
    action: int,
    action_dose: float,
    latent: Array,
    t: int,
    scenario: DomainScenario,
    rng: np.random.Generator,
) -> tuple[Array, float, float, float, dict[str, float]]:
    loyalty, competitor, price, subgroup = latent
    dose = float(np.clip(action_dose, 0.0, 1.5))
    action_effect = np.array([0.00, 0.045, 0.025, 0.035, 0.020, 0.010, 0.030, 0.020, 0.015])[action] * dose
    cost = np.array([0.00, 0.20, 0.35, 2.80, 1.20, 0.80, 0.70, 0.90, 1.60])[action] * dose
    fatigue_delta = np.array([0.00, 0.015, 0.060, 0.025, 0.010, 0.015, 0.005, 0.020, 0.010])[action] * dose
    if scenario.delay_pattern == "delayed_benefit" and t < 3:
        action_effect *= 0.45
    if scenario.nonstationarity and t >= 12:
        action_effect *= 0.75
        competitor = min(1.0, competitor + 0.08)
    seasonality = 0.04 * np.sin(2.0 * np.pi * (state[7] + t / 12.0))
    engagement = np.clip(0.75 * state[0] + 0.12 * loyalty + action_effect - 0.04 * state[4] + seasonality + rng.normal(0, 0.035), 0.0, 1.0)
    fatigue = np.clip(0.82 * state[4] + fatigue_delta + 0.03 * competitor + rng.normal(0, 0.025), 0.0, 1.0)
    risk = np.clip(0.55 * state[5] + 0.30 * competitor + 0.25 * price + 0.20 * fatigue - 0.45 * engagement + rng.normal(0, 0.035), 0.0, 1.0)
    hazard = _sigmoid(-2.9 + 2.25 * risk + 0.8 * competitor - 1.0 * engagement - 0.25 * dose * (action in {3, 4, 8}))
    churn = float(rng.random() < hazard)
    censor = _censoring_draw(state, risk, scenario, rng)
    retained = 1.0 - churn
    subscription_revenue = retained * (9.5 + 5.5 * state[3] + 1.0 * state[0])
    ad_revenue = retained * (1.0 - state[3]) * max(0.0, 1.2 - 0.2 * (action == 6))
    reward = subscription_revenue + ad_revenue - cost - 0.6 * fatigue
    next_state = np.array([engagement, min(1.0, state[1] + 1.0 / 36.0), state[2], state[3], fatigue, risk, subgroup, (state[7] + 1.0 / 12.0) % 1.0])
    return next_state, float(reward), churn, censor, {
        "subscription_revenue": float(subscription_revenue),
        "ad_revenue": float(ad_revenue),
        "intervention_cost": float(cost),
        "retained": retained,
        "fatigue": float(fatigue),
        "action_dose": dose,
    }


def _clinic_initial(n: int, rng: np.random.Generator) -> tuple[Array, Array]:
    subgroup = rng.binomial(1, 0.46, size=n).astype(np.float64)
    age = np.clip(rng.normal(0.55 + 0.08 * subgroup, 0.15, size=n), 0.18, 0.95)
    diabetes = rng.binomial(1, 0.45 + 0.10 * subgroup, size=n).astype(np.float64)
    ascvd = rng.binomial(1, 0.25 + 0.20 * age, size=n).astype(np.float64)
    bp = np.clip(rng.normal(0.48 + 0.10 * diabetes + 0.10 * subgroup, 0.15, size=n), 0.0, 1.0)
    ldl = np.clip(rng.normal(0.46 + 0.12 * ascvd, 0.14, size=n), 0.0, 1.0)
    hba1c = np.clip(rng.normal(0.44 + 0.22 * diabetes, 0.15, size=n), 0.0, 1.0)
    kidney = np.clip(rng.normal(0.22 + 0.25 * diabetes + 0.18 * age, 0.12, size=n), 0.0, 1.0)
    adherence = np.clip(rng.beta(4.0 - 0.8 * subgroup, 2.4, size=n), 0.02, 0.98)
    toxicity = np.clip(rng.beta(1.4, 7.0, size=n) + 0.10 * kidney, 0.0, 1.0)
    states = np.column_stack([age, diabetes, ascvd, bp, ldl, hba1c, kidney, adherence, toxicity, subgroup])
    latent = np.column_stack([
        np.clip(rng.normal(0.0, 1.0, size=n), -2.5, 2.5),
        rng.beta(2.0, 3.0, size=n),
        subgroup,
    ])
    return states, latent


def _clinic_action_availability(state: Array, scenario: DomainScenario | None = None) -> Array:
    if scenario is not None and not scenario.action_constraints:
        return np.ones(len(CLINIC_ACTIONS), dtype=np.float64)
    avail = np.ones(len(CLINIC_ACTIONS), dtype=np.float64)
    if state[6] > 0.72 or state[8] > 0.68:
        avail[2] = 0.0
        avail[3] = 0.0
    if state[3] > 0.75 or state[5] > 0.78:
        avail[4] = 0.0
    return avail


def _clinic_behavior_probs(state: Array, availability: Array, scenario: DomainScenario, target_probs: Array) -> Array:
    risk = 0.35 * state[3] + 0.25 * state[4] + 0.30 * state[5] + 0.20 * state[6]
    probs = np.array([0.38, 0.16, 0.16, 0.16, 0.14], dtype=np.float64)
    probs[2] += 0.20 * ((state[3] > 0.55) or (state[4] > 0.55))
    probs[3] += 0.22 * (state[5] > 0.55)
    probs[4] += 0.18 * (state[8] > 0.55)
    if scenario.overlap == "weak":
        probs[0] += 0.20
        probs[2:4] *= 0.60
    if scenario.confounding in {"observed", "latent"}:
        probs[2:4] += 0.08 * risk
    nominal = normalize_action_probs(probs.reshape(1, -1), availability.reshape(1, -1))[0]
    return _calibrated_behavior_probs(
        np.asarray(target_probs, dtype=np.float64),
        nominal,
        availability,
        desired_tv=float(scenario.target_policy_distance),
    )


def _clinic_action_dose(state: Array, action: int, scenario: DomainScenario, rng: np.random.Generator) -> float:
    mean = _clinic_action_dose_mean(state, action, scenario)
    if action == 0:
        return 0.0
    return float(np.clip(mean + rng.normal(0.0, 0.07), -0.85, 1.35))


def _clinic_action_dose_mean(state: Array, action: int, scenario: DomainScenario) -> float:
    if action == 0:
        return 0.0
    base = np.array([0.0, 0.30, 0.85, 0.85, -0.45], dtype=np.float64)[action]
    risk = 0.30 * float(state[3]) + 0.25 * float(state[4]) + 0.30 * float(state[5]) + 0.15 * float(state[6])
    adherence = float(state[7])
    if action in {2, 3}:
        base += 0.18 * risk - 0.08 * (1.0 - adherence)
    elif action == 4:
        base -= 0.12 * float((state[6] > 0.65) or (state[8] > 0.55))
    elif action == 1:
        base += 0.08 * (1.0 - adherence)
    if not scenario.action_constraints:
        base *= 0.90
    return float(np.clip(base, -0.85, 1.35))


def _clinic_step(
    state: Array,
    action: int,
    action_dose: float,
    latent: Array,
    t: int,
    scenario: DomainScenario,
    rng: np.random.Generator,
) -> tuple[Array, float, float, float, dict[str, float]]:
    frailty, adherence_trait, subgroup = latent
    drift = 0.08 if scenario.nonstationarity and t >= 12 else 0.0
    dose = float(np.clip(action_dose, -1.0, 1.5))
    intensify_dose = max(dose, 0.0)
    deintensify_dose = max(-dose, 0.0)
    bp_change = -0.070 * intensify_dose * (action == 2) - 0.035 * intensify_dose * (action == 1) + 0.060 * deintensify_dose * (action == 4)
    ldl_change = -0.058 * intensify_dose * (action == 2) - 0.028 * intensify_dose * (action == 1) + 0.052 * deintensify_dose * (action == 4)
    hba1c_change = -0.075 * intensify_dose * (action == 3) - 0.035 * intensify_dose * (action == 1) + 0.060 * deintensify_dose * (action == 4)
    adherence = np.clip(0.75 * state[7] + 0.18 * adherence_trait - 0.05 * intensify_dose * (action in {2, 3}) + 0.04 * (action == 1) + rng.normal(0, 0.035), 0.0, 1.0)
    bp = np.clip(0.86 * state[3] + bp_change * adherence + 0.025 * state[0] + drift + rng.normal(0, 0.035), 0.0, 1.0)
    ldl = np.clip(0.88 * state[4] + ldl_change * adherence + rng.normal(0, 0.03), 0.0, 1.0)
    hba1c = np.clip(0.86 * state[5] + hba1c_change * adherence + 0.02 * state[1] + rng.normal(0, 0.035), 0.0, 1.0)
    kidney = np.clip(0.90 * state[6] + 0.030 * hba1c + 0.025 * bp + 0.015 * state[0] + rng.normal(0, 0.02), 0.0, 1.0)
    toxicity = np.clip(0.75 * state[8] + 0.09 * intensify_dose * (action in {2, 3}) + 0.04 * kidney - 0.07 * deintensify_dose * (action == 4) + rng.normal(0, 0.03), 0.0, 1.0)
    event_risk = _sigmoid(-3.5 + 1.4 * bp + 1.0 * ldl + 1.2 * hba1c + 1.4 * kidney + 0.55 * state[2] + 0.45 * frailty + 0.5 * drift)
    death = float(rng.random() < event_risk)
    censor = _censoring_draw(state, event_risk, scenario, rng)
    event_free = 1.0 - death
    biomarker = 1.0 - np.mean([bp, ldl, hba1c])
    qaly = event_free * (0.78 + 0.12 * biomarker - 0.18 * toxicity) - 0.05 * abs(dose)
    next_state = np.array([state[0], state[1], state[2], bp, ldl, hba1c, kidney, adherence, toxicity, subgroup])
    return next_state, float(qaly), death, censor, {
        "qaly": float(qaly),
        "event_free": event_free,
        "toxicity": float(toxicity),
        "biomarker": float(biomarker),
        "dose": float(dose),
        "action_dose": float(dose),
    }


def _policy_value_truth(
    *,
    family: str,
    dataset_name: str,
    baseline: Array,
    latent: Array,
    scenario: DomainScenario,
    target_policy: str,
    gamma: float,
    horizon: int,
    mc_rollouts: int,
    seed: int,
) -> TruthBundle:
    rng = np.random.default_rng(seed)
    policy = get_fixed_policy(family, target_policy)  # type: ignore[arg-type]
    values = []
    subgroup_values: dict[int, list[float]] = {0: [], 1: []}
    survival = np.zeros(horizon, dtype=np.float64)
    rmst_values = []
    biomarker_values = []
    contact_values = []
    spend_values = []
    fatigue_values = []
    contraindicated_values = []
    high_toxicity_intensification_values = []
    monitoring_values = []
    intensity_values = []
    for i in range(baseline.shape[0]):
        unit_values = []
        unit_survival = []
        unit_rmst = []
        unit_biomarker = []
        for _ in range(mc_rollouts):
            state = baseline[i].copy()
            rewards = []
            alive = True
            survival_path = []
            biomarkers = []
            contacts = []
            spends = []
            fatigues = []
            contraindicated = []
            high_toxicity_intensifications = []
            monitoring = []
            intensities = []
            for t in range(horizon):
                availability = _stream_action_availability(state, scenario) if family == "streamretain" else _clinic_action_availability(state, scenario)
                probs = policy.probabilities(state.reshape(1, -1), availability.reshape(1, -1), np.array([t]))[0]
                action = int(rng.choice(probs.shape[0], p=probs))
                if family == "streamretain":
                    action_dose = _stream_action_dose(state, action, scenario, rng)
                    state, reward, terminal, censor, components = _streamretain_step(state, action, action_dose, latent[i], t, scenario, rng)
                    del censor
                    alive = alive and terminal < 0.5
                    biomarkers.append(float(components["retained"]))
                    contacts.append(float(action != 0))
                    spends.append(float(components["intervention_cost"]))
                    fatigues.append(float(components["fatigue"]))
                else:
                    pre_state = state.copy()
                    action_dose = _clinic_action_dose(state, action, scenario, rng)
                    state, reward, terminal, censor, components = _clinic_step(state, action, action_dose, latent[i], t, scenario, rng)
                    del censor
                    alive = alive and terminal < 0.5
                    biomarkers.append(float(components["biomarker"]))
                    intensify = float(action in {2, 3})
                    contraindicated.append(float(intensify and ((pre_state[6] > 0.72) or (pre_state[8] > 0.68))))
                    high_toxicity_intensifications.append(float(intensify and float(components["toxicity"]) > 0.55))
                    monitoring.append(float(action == 0))
                    intensities.append(abs(float(action_dose)))
                rewards.append(reward)
                survival_path.append(float(alive))
                if not alive:
                    break
            rewards_arr = np.asarray(rewards, dtype=np.float64)
            unit_values.append(_discounted_sum(rewards_arr, gamma))
            padded_survival = np.zeros(horizon, dtype=np.float64)
            padded_survival[: len(survival_path)] = survival_path
            unit_survival.append(padded_survival)
            unit_rmst.append(float(np.sum(padded_survival)))
            unit_biomarker.append(float(np.mean(biomarkers)) if biomarkers else 0.0)
            if contacts:
                contact_values.append(float(np.mean(contacts)))
                spend_values.append(float(np.mean(spends)))
                fatigue_values.append(float(np.mean(fatigues)))
            if contraindicated:
                contraindicated_values.append(float(np.mean(contraindicated)))
                high_toxicity_intensification_values.append(float(np.mean(high_toxicity_intensifications)))
                monitoring_values.append(float(np.mean(monitoring)))
                intensity_values.append(float(np.mean(intensities)))
        mean_value = float(np.mean(unit_values))
        values.append(mean_value)
        subgroup_col = 6 if family == "streamretain" else baseline.shape[1] - 1
        subgroup_values[int(round(baseline[i, subgroup_col]))].append(mean_value)
        survival += np.mean(np.asarray(unit_survival), axis=0)
        rmst_values.append(float(np.mean(unit_rmst)))
        biomarker_values.append(float(np.mean(unit_biomarker)))
    survival /= baseline.shape[0]
    value = float(np.mean(values))
    truth_values = {"policy_value": value}
    rmst = {}
    survival_curves = {}
    if family == "clinic_dtr":
        rmst = {"rmst": float(np.mean(rmst_values))}
        survival_curves = {"survival_target": survival}
        truth_values["survival_horizon"] = float(survival[-1])
        truth_values["biomarker_mean"] = float(np.mean(biomarker_values))
    else:
        truth_values["retention_horizon"] = float(survival[-1])
    standard_errors = {"policy_value": _se(values)}
    if "biomarker_mean" in truth_values:
        standard_errors["biomarker_mean"] = _se(biomarker_values)
    if "rmst" in rmst:
        standard_errors["rmst"] = _se(rmst_values)
    if "survival_horizon" in truth_values:
        p = float(truth_values["survival_horizon"])
        standard_errors["survival_horizon"] = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(baseline.shape[0] * mc_rollouts, 1)))
    if "retention_horizon" in truth_values:
        p = float(truth_values["retention_horizon"])
        standard_errors["retention_horizon"] = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(baseline.shape[0] * mc_rollouts, 1)))
    private_metadata: dict[str, object] = {
        "scenario_private_name": scenario.name,
        "confounding": scenario.confounding,
        "overlap": scenario.overlap,
        "target_policy": target_policy,
    }
    if family == "streamretain":
        private_metadata.update(
            {
                "budget_limit": 2.0,
                "fatigue_limit": 1.25,
                "target_policy_contact_rate": float(np.mean(contact_values)) if contact_values else 0.0,
                "target_policy_budget_observed": float(np.mean(spend_values)) if spend_values else 0.0,
                "target_policy_fatigue_observed": float(np.mean(fatigue_values)) if fatigue_values else 0.0,
            }
        )
    else:
        private_metadata.update(
            {
                "contraindicated_action_rate_limit": 0.05,
                "high_toxicity_intensification_rate_limit": 0.05,
                "target_policy_contraindicated_action_rate": float(np.mean(contraindicated_values)) if contraindicated_values else 0.0,
                "target_policy_high_toxicity_intensification_rate": (
                    float(np.mean(high_toxicity_intensification_values)) if high_toxicity_intensification_values else 0.0
                ),
                "target_policy_monitoring_action_rate": float(np.mean(monitoring_values)) if monitoring_values else 0.0,
                "target_policy_action_intensity": float(np.mean(intensity_values)) if intensity_values else 0.0,
            }
        )
    return TruthBundle(
        dataset_name=dataset_name,
        family=family,  # type: ignore[arg-type]
        values=truth_values,
        survival_curves=survival_curves,
        rmst=rmst,
        subgroup_effects={
            f"subgroup_{g}_value": float(np.mean(vals))
            for g, vals in subgroup_values.items()
            if vals
        },
        latent_parameters={"latent_dim": int(latent.shape[1]) if latent.ndim == 2 else 0},
        target_mc_values=truth_values.copy(),
        target_standard_errors=standard_errors,
        truth_noise_floor=_truth_noise_floor(truth_values | rmst),
        mc_rollouts=int(mc_rollouts),
        private_metadata=private_metadata,
        leaderboard_eligible=bool(scenario.leaderboard_eligible and scenario.confounding != "latent"),
    )


def _missingness_mask(states: Array, scenario: DomainScenario, rng: np.random.Generator) -> Array:
    if scenario.missingness == "none":
        return np.zeros_like(states, dtype=np.float64)
    if scenario.missingness == "mcar":
        prob = np.full_like(states, 0.03, dtype=np.float64)
    elif scenario.missingness == "mar":
        row_prob = np.clip(0.02 + 0.08 * states[:, [0]] + 0.05 * states[:, [-1]], 0.0, 0.18)
        prob = np.repeat(row_prob, states.shape[1], axis=1)
    else:
        row_prob = np.clip(0.03 + 0.12 * states[:, [min(5, states.shape[1] - 1)]], 0.0, 0.25)
        prob = np.repeat(row_prob, states.shape[1], axis=1)
    return (rng.random(states.shape) < prob).astype(np.float64)


def _censoring_draw(state: Array, risk: float, scenario: DomainScenario, rng: np.random.Generator) -> float:
    if scenario.censoring == "none":
        return 0.0
    if scenario.censoring == "administrative":
        p = 0.004
    elif scenario.censoring == "informative":
        p = 0.006 + 0.055 * float(risk)
    else:
        p = 0.004 + 0.035 * float(risk) + 0.020 * float(state[-1])
    return float(rng.random() < min(0.35, p))


def _discounted_sum(values: Array, gamma: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sum((float(gamma) ** np.arange(arr.shape[0])) * arr))


def _se(values: Sequence[float] | Array) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _truth_noise_floor(values: dict[str, float]) -> dict[str, float]:
    return {key: float(max(1e-8, 0.0025 * max(1.0, abs(float(value))))) for key, value in values.items()}


def _sigmoid(x: Array | float) -> Array | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _public_scenario_label(name: str) -> str:
    mapping = {
        "clean_randomized_good_overlap": "cell_a",
        "randomized_good_overlap": "cell_a",
        "observed_moderate_overlap": "cell_b",
        "observed_confounding_moderate_overlap": "cell_b",
        "observed_weak_overlap": "cell_c",
        "weak_overlap_misleading_surrogate": "cell_c",
        "missing_censoring_nonstationary_stress": "stress_a",
        "latent_confounding_sensitivity": "stress_b",
    }
    return mapping.get(str(name), "cell_custom")
