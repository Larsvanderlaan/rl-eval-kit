from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_ope_benchmark.policies import target_policy_probabilities
from causal_ope_benchmark.types import ActionSpec, Array, LongitudinalDataset, assert_no_forbidden_public_keys
from causal_ope_benchmark.validation import validate_longitudinal_dataset


@dataclass
class OccupancyRatioAdapterDataset:
    """Flat arrays compatible with occupancy-ratio fitters."""

    setting: str
    states: Array
    actions: Array
    next_states: Array
    target_actions: Array
    next_target_actions: Array
    rewards: Array
    initial_states: Array
    initial_actions: Array
    initial_weights: Array
    masks: Array
    gamma: float
    seed: int
    sample_size: int
    behavior_propensity: Array
    target_propensity: Array
    target_propensity_observed_action: Array
    action_dose: Array | None = None
    target_action_dose: Array | None = None
    dose_available: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_no_forbidden_public_keys(self.metadata)


@dataclass
class FQEAdapterDataset:
    """Flat arrays compatible with FQE-style estimators."""

    name: str
    domain: str
    states: Array
    actions: Array
    next_states: Array
    next_actions: Array
    rewards: Array
    terminals: Array
    gamma: float
    seed: int
    initial_states: Array
    initial_actions: Array
    target_eval_states: Array
    target_eval_actions: Array
    behavior_eval_states: Array
    behavior_eval_actions: Array
    sample_weight: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    target_actions: Array | None = None
    action_spec: ActionSpec | None = None
    target_action_probabilities: Array | None = None
    next_target_action_probabilities: Array | None = None
    initial_action_probabilities: Array | None = None
    target_eval_action_probabilities: Array | None = None
    action_dose: Array | None = None
    target_action_dose: Array | None = None
    dose_available: Array | None = None
    target_policy_expectation_mode: str = "sampled_action"
    target_probability_storage: str = "sampled_action_only"
    row_expansion_factor: float = 1.0
    source_row_index: Array | None = None

    def __post_init__(self) -> None:
        assert_no_forbidden_public_keys(self.metadata)
        if self.target_actions is None:
            self.target_actions = np.asarray(self.actions)
        n = int(np.asarray(self.states).shape[0])
        for name in ("actions", "next_actions", "target_actions"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2:
                raise ValueError(f"FQEAdapterDataset.{name} must be 2D.")
            if value.shape[0] != n:
                raise ValueError(f"FQEAdapterDataset.{name} must have {n} rows.")
        if self.sample_weight is not None and np.asarray(self.sample_weight).reshape(-1).shape[0] != n:
            raise ValueError("FQEAdapterDataset.sample_weight must match states rows.")
        self.target_action_probabilities = _validate_optional_probabilities(
            self.target_action_probabilities,
            n_rows=n,
            n_actions=np.asarray(self.actions).shape[1],
            name="target_action_probabilities",
        )
        self.next_target_action_probabilities = _validate_optional_probabilities(
            self.next_target_action_probabilities,
            n_rows=n,
            n_actions=np.asarray(self.actions).shape[1],
            name="next_target_action_probabilities",
        )
        self.initial_action_probabilities = _validate_optional_probabilities(
            self.initial_action_probabilities,
            n_rows=np.asarray(self.initial_states).shape[0],
            n_actions=np.asarray(self.actions).shape[1],
            name="initial_action_probabilities",
        )
        self.target_eval_action_probabilities = _validate_optional_probabilities(
            self.target_eval_action_probabilities,
            n_rows=np.asarray(self.target_eval_states).shape[0],
            n_actions=np.asarray(self.actions).shape[1],
            name="target_eval_action_probabilities",
        )
        for name in ("action_dose", "target_action_dose", "dose_available"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).reshape(-1).shape[0] != n:
                raise ValueError(f"FQEAdapterDataset.{name} must match states rows.")
        if self.source_row_index is not None:
            idx = np.asarray(self.source_row_index, dtype=np.int64).reshape(-1)
            if idx.shape[0] != n:
                raise ValueError("FQEAdapterDataset.source_row_index must match states rows.")
            if idx.size and idx.min() < 0:
                raise ValueError("FQEAdapterDataset.source_row_index must be nonnegative.")
            self.source_row_index = idx


@dataclass
class EffectPanel:
    """Estimator-visible unit-level short-panel causal effect data for StreamLift."""

    name: str
    unit_id: Array
    arm: Array
    received_treatment: Array
    baseline_state: Array
    observed_time: Array
    observed_reward: Array
    observed_retained: Array
    behavior_propensity: Array
    assignment_propensity: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_no_forbidden_public_keys(self.metadata)
        n_units = np.asarray(self.unit_id).shape[0]
        for name, value in (
            ("arm", self.arm),
            ("received_treatment", self.received_treatment),
            ("baseline_state", self.baseline_state),
            ("observed_time", self.observed_time),
            ("observed_reward", self.observed_reward),
            ("observed_retained", self.observed_retained),
            ("behavior_propensity", self.behavior_propensity),
        ):
            if np.asarray(value).shape[0] != n_units:
                raise ValueError(f"EffectPanel.{name} must have one row per unit.")
        if self.assignment_propensity is not None and np.asarray(self.assignment_propensity).shape[0] != n_units:
            raise ValueError("EffectPanel.assignment_propensity must have one row per unit.")


@dataclass
class SurvivalPanel:
    """Estimator-visible survival panel for clinical and retention benchmarks."""

    name: str
    unit_id: Array
    time: Array
    event: Array
    censored: Array
    action: Array
    covariates: Array
    behavior_propensity: Array
    target_propensity: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_no_forbidden_public_keys(self.metadata)


def to_occupancy_ratio_dataset(dataset: LongitudinalDataset) -> OccupancyRatioAdapterDataset:
    """Convert public longitudinal data to an occupancy-ratio adapter dataset."""
    validate_longitudinal_dataset(dataset)
    metadata = _adapter_metadata(dataset)
    return OccupancyRatioAdapterDataset(
        setting=dataset.name,
        states=np.asarray(dataset.states),
        actions=np.asarray(dataset.actions),
        next_states=np.asarray(dataset.next_states),
        target_actions=np.asarray(dataset.target_actions),
        next_target_actions=np.asarray(dataset.next_target_actions),
        rewards=np.asarray(dataset.rewards),
        initial_states=np.asarray(dataset.initial_states),
        initial_actions=np.asarray(dataset.initial_actions),
        initial_weights=np.full(np.asarray(dataset.initial_states).shape[0], 1.0 / np.asarray(dataset.initial_states).shape[0]),
        masks=1.0 - _fqe_terminals(dataset),
        gamma=float(dataset.gamma),
        seed=int(dataset.seed),
        sample_size=int(dataset.metadata_public.get("sample_size", dataset.n)),
        behavior_propensity=np.asarray(dataset.behavior_propensity),
        target_propensity=np.asarray(dataset.target_propensity_observed_action),
        target_propensity_observed_action=np.asarray(dataset.target_propensity_observed_action),
        action_dose=None if dataset.action_dose is None else np.asarray(dataset.action_dose),
        target_action_dose=None if dataset.target_action_dose is None else np.asarray(dataset.target_action_dose),
        dose_available=None if dataset.dose_available is None else np.asarray(dataset.dose_available),
        metadata=metadata,
    )


def to_fqe_dataset(
    dataset: LongitudinalDataset,
    *,
    target_policy_expectation_mode: str = "sampled_action",
) -> FQEAdapterDataset:
    """Convert public longitudinal data to an FQE adapter dataset."""
    validate_longitudinal_dataset(dataset)
    eval_idx = np.asarray(dataset.splits.get("eval", np.arange(dataset.n)), dtype=np.int64)
    if eval_idx.size == 0:
        eval_idx = np.arange(dataset.n)
    mode = str(target_policy_expectation_mode)
    if mode not in {"sampled_action", "exact_discrete"}:
        raise ValueError("target_policy_expectation_mode must be 'sampled_action' or 'exact_discrete'.")
    target_probs, target_storage = _current_target_probabilities(dataset)
    next_probs, next_storage = _next_target_probabilities(dataset)
    initial_probs, initial_storage = _initial_target_probabilities(dataset)
    storage = _combine_probability_storage(target_storage, next_storage, initial_storage)
    states = np.asarray(dataset.states)
    actions = np.asarray(dataset.actions)
    next_states = np.asarray(dataset.next_states)
    next_actions = np.asarray(dataset.next_target_actions)
    rewards = np.asarray(dataset.rewards)
    terminals = _fqe_terminals(dataset)
    target_actions = np.asarray(dataset.target_actions)
    sample_weight = None
    row_expansion_factor = 1.0
    adapter_target_probs = target_probs
    adapter_next_probs = next_probs
    action_dose = None if dataset.action_dose is None else np.asarray(dataset.action_dose)
    target_action_dose = None if dataset.target_action_dose is None else np.asarray(dataset.target_action_dose)
    dose_available = None if dataset.dose_available is None else np.asarray(dataset.dose_available)
    source_row_index = np.arange(dataset.n, dtype=np.int64)
    if mode == "exact_discrete":
        expanded = _expand_fqe_rows_for_exact_discrete_expectation(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
            target_actions=target_actions,
            next_action_probabilities=next_probs,
        )
        states = expanded["states"]
        actions = expanded["actions"]
        next_states = expanded["next_states"]
        next_actions = expanded["next_actions"]
        rewards = expanded["rewards"]
        terminals = expanded["terminals"]
        target_actions = expanded["target_actions"]
        sample_weight = expanded["sample_weight"]
        adapter_target_probs = target_probs[expanded["row_indices"].astype(np.int64)]
        adapter_next_probs = next_probs[expanded["row_indices"].astype(np.int64)]
        row_indices = expanded["row_indices"].astype(np.int64)
        source_row_index = row_indices
        action_dose = None if action_dose is None else action_dose[row_indices]
        target_action_dose = None if target_action_dose is None else target_action_dose[row_indices]
        dose_available = None if dose_available is None else dose_available[row_indices]
        row_expansion_factor = float(states.shape[0] / max(dataset.n, 1))
    return FQEAdapterDataset(
        name=dataset.name,
        domain=dataset.family,
        states=states,
        actions=actions,
        target_actions=target_actions,
        next_states=next_states,
        next_actions=next_actions,
        rewards=rewards,
        terminals=terminals,
        gamma=float(dataset.gamma),
        seed=int(dataset.seed),
        initial_states=np.asarray(dataset.initial_states),
        initial_actions=np.asarray(dataset.initial_actions),
        target_eval_states=np.asarray(dataset.states)[eval_idx],
        target_eval_actions=np.asarray(dataset.target_actions)[eval_idx],
        behavior_eval_states=np.asarray(dataset.states)[eval_idx],
        behavior_eval_actions=np.asarray(dataset.actions)[eval_idx],
        sample_weight=sample_weight,
        metadata={
            **_adapter_metadata(dataset),
            "target_policy_expectation_mode": mode,
            "target_probability_storage": storage,
            "fqe_adapter_row_expansion_factor": row_expansion_factor,
        },
        action_spec=dataset.action_spec,
        target_action_probabilities=adapter_target_probs,
        next_target_action_probabilities=adapter_next_probs,
        initial_action_probabilities=initial_probs,
        target_eval_action_probabilities=target_probs[eval_idx],
        action_dose=action_dose,
        target_action_dose=target_action_dose,
        dose_available=dose_available,
        target_policy_expectation_mode=mode,
        target_probability_storage=storage,
        row_expansion_factor=row_expansion_factor,
        source_row_index=source_row_index,
    )


def to_effect_panel(dataset: LongitudinalDataset) -> EffectPanel:
    """Convert StreamLift data to a short-panel effect estimation view."""
    validate_longitudinal_dataset(dataset)
    if dataset.family != "streamlift":
        raise ValueError("to_effect_panel is only defined for StreamLift datasets.")
    units = np.unique(dataset.unit_id)
    first_idx = np.array([np.flatnonzero(dataset.unit_id == unit)[0] for unit in units], dtype=np.int64)
    if dataset.assigned_arm is None:
        arm = np.argmax(dataset.initial_actions[: first_idx.shape[0]], axis=1)
    else:
        arm = np.asarray(dataset.assigned_arm)[first_idx].astype(np.int64)
    received = np.argmax(dataset.actions[first_idx], axis=1)
    retained = np.asarray(dataset.outcome_components.get("retained", 1.0 - dataset.terminals), dtype=np.float64)
    unit_rows = [np.flatnonzero(dataset.unit_id == unit) for unit in units]
    max_len = max((idx.size for idx in unit_rows), default=0)
    observed_time = np.full((units.shape[0], max_len), -1, dtype=np.int64)
    observed_reward = np.full((units.shape[0], max_len), np.nan, dtype=np.float64)
    observed_retained = np.full((units.shape[0], max_len), np.nan, dtype=np.float64)
    for row_id, idx in enumerate(unit_rows):
        order = idx[np.argsort(np.asarray(dataset.time)[idx])]
        length = order.size
        observed_time[row_id, :length] = np.asarray(dataset.time)[order]
        observed_reward[row_id, :length] = np.asarray(dataset.rewards)[order]
        observed_retained[row_id, :length] = retained[order]
    return EffectPanel(
        name=dataset.name,
        unit_id=np.asarray(units),
        arm=arm,
        received_treatment=received,
        baseline_state=np.asarray(dataset.initial_states)[: first_idx.shape[0]],
        observed_time=observed_time,
        observed_reward=observed_reward,
        observed_retained=observed_retained,
        behavior_propensity=np.asarray(dataset.behavior_propensity)[first_idx],
        assignment_propensity=None if dataset.assignment_propensity is None else np.asarray(dataset.assignment_propensity)[first_idx],
        metadata=_adapter_metadata(dataset),
    )


def to_survival_panel(dataset: LongitudinalDataset) -> SurvivalPanel:
    """Convert public longitudinal data to a survival panel."""
    validate_longitudinal_dataset(dataset)
    return SurvivalPanel(
        name=dataset.name,
        unit_id=np.asarray(dataset.unit_id),
        time=np.asarray(dataset.time),
        event=np.asarray(dataset.terminals),
        censored=np.asarray(dataset.censoring),
        action=np.argmax(np.asarray(dataset.actions), axis=1),
        covariates=np.asarray(dataset.states),
        behavior_propensity=np.asarray(dataset.behavior_propensity),
        target_propensity=np.asarray(dataset.target_propensity_observed_action),
        metadata=_adapter_metadata(dataset),
    )


def _adapter_metadata(dataset: LongitudinalDataset) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in dataset.metadata_public.items()
        if key in set(dataset.information_contract.visible_metadata if dataset.information_contract else dataset.metadata_public)
    }
    assert_no_forbidden_public_keys(metadata)
    return metadata


def _fqe_terminals(dataset: LongitudinalDataset) -> Array:
    """Continuation mask convention for generic FQE adapters.

    Censoring is observation loss, not a domain event, but array-based FQE
    cannot continue through unobserved futures. The adapter therefore stops
    Bellman continuation at either event terminal or censoring and reports the
    raw censoring indicator separately.
    """
    return np.maximum(np.asarray(dataset.terminals, dtype=np.float64), np.asarray(dataset.censoring, dtype=np.float64))


def _current_target_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str]:
    if dataset.target_action_probabilities is not None:
        return np.asarray(dataset.target_action_probabilities, dtype=np.float64), "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        return (
            target_policy_probabilities(
                dataset.family,
                str(policy_name),
                dataset.states,
                availability=dataset.action_available,
                time=dataset.time,
            ),
            "named_policy",
        )
    return np.asarray(dataset.target_actions, dtype=np.float64), "sampled_action_only"


def _next_target_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str]:
    if dataset.next_target_action_probabilities is not None:
        return np.asarray(dataset.next_target_action_probabilities, dtype=np.float64), "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        return (
            target_policy_probabilities(
                dataset.family,
                str(policy_name),
                dataset.next_states,
                time=np.asarray(dataset.time) + 1,
            ),
            "named_policy",
        )
    return np.asarray(dataset.next_target_actions, dtype=np.float64), "sampled_action_only"


def _initial_target_probabilities(dataset: LongitudinalDataset) -> tuple[Array, str]:
    if dataset.initial_action_probabilities is not None:
        return np.asarray(dataset.initial_action_probabilities, dtype=np.float64), "full_matrix"
    policy_name = dataset.metadata_public.get("target_policy")
    if policy_name is not None:
        initial_idx = np.asarray(dataset.splits.get("initial", []), dtype=np.int64)
        availability = None
        if initial_idx.shape[0] == np.asarray(dataset.initial_states).shape[0] and initial_idx.size:
            availability = np.asarray(dataset.action_available)[initial_idx]
        return (
            target_policy_probabilities(
                dataset.family,
                str(policy_name),
                dataset.initial_states,
                availability=availability,
                time=np.zeros(np.asarray(dataset.initial_states).shape[0], dtype=np.int64),
            ),
            "named_policy",
        )
    return np.asarray(dataset.initial_actions, dtype=np.float64), "sampled_action_only"


def _combine_probability_storage(*storage: str) -> str:
    unique = tuple(dict.fromkeys(storage))
    if len(unique) == 1:
        return unique[0]
    if "sampled_action_only" in unique:
        return "mixed_with_sampled_action_fallback"
    return "mixed"


def _expand_fqe_rows_for_exact_discrete_expectation(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    rewards: Array,
    terminals: Array,
    target_actions: Array,
    next_action_probabilities: Array,
) -> dict[str, Array]:
    probs = np.asarray(next_action_probabilities, dtype=np.float64)
    n, n_actions = probs.shape
    action_eye = np.eye(n_actions, dtype=np.float64)
    row, col = np.nonzero(probs > 0.0)
    if row.size == 0:
        raise ValueError("next_action_probabilities must contain positive probability mass.")
    return {
        "states": np.asarray(states)[row],
        "actions": np.asarray(actions)[row],
        "next_states": np.asarray(next_states)[row],
        "next_actions": action_eye[col],
        "rewards": np.asarray(rewards)[row],
        "terminals": np.asarray(terminals)[row],
        "target_actions": np.asarray(target_actions)[row],
        "sample_weight": probs[row, col],
        "row_indices": row.astype(np.int64),
    }


def _validate_optional_probabilities(
    value: Array | None,
    *,
    n_rows: int,
    n_actions: int,
    name: str,
) -> Array | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (int(n_rows), int(n_actions)):
        raise ValueError(f"FQEAdapterDataset.{name} must have shape ({int(n_rows)}, {int(n_actions)}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"FQEAdapterDataset.{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"FQEAdapterDataset.{name} must be nonnegative.")
    if arr.shape[0] and not np.allclose(arr.sum(axis=1), 1.0):
        raise ValueError(f"FQEAdapterDataset.{name} rows must sum to 1.")
    return arr
