from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np


Array = np.ndarray
BenchmarkFamily = Literal["streamlift", "streamretain", "clinic_dtr", "epicare"]
SplitName = Literal["train", "eval", "initial"]
ActionEncoding = Literal["one_hot"]


TRUTH_FORBIDDEN_TOKENS = (
    "truth",
    "true_",
    "oracle",
    "latent",
    "private",
    "target_mc",
    "monte_carlo",
    "surrogate_validity",
    "scenario_private",
)


@dataclass(frozen=True)
class EstimatorInformationContract:
    """Estimator-visible information contract for one benchmark family.

    Parameters
    ----------
    family:
        Benchmark family that owns this contract.
    visible_arrays:
        Public array fields estimators may receive.
    visible_metadata:
        Public metadata keys estimators may inspect.
    allows_behavior_propensity:
        Whether behavior propensities are estimator-visible.
    allows_target_propensity:
        Whether target-policy propensities or samplers are estimator-visible.
    allows_censoring:
        Whether censoring indicators are estimator-visible.
    notes:
        Human-readable contract notes.
    """

    family: BenchmarkFamily
    visible_arrays: tuple[str, ...]
    visible_metadata: tuple[str, ...]
    allows_behavior_propensity: bool = True
    allows_target_propensity: bool = True
    allows_censoring: bool = True
    notes: str = ""

    def assert_public_metadata_allowed(self, metadata: dict[str, Any]) -> None:
        unknown = sorted(set(metadata) - set(self.visible_metadata))
        if unknown:
            raise ValueError(f"Public metadata contains keys outside the information contract: {unknown}.")
        assert_no_forbidden_public_keys(metadata)


@dataclass
class ActionSpec:
    """Discrete action-space description.

    Parameters
    ----------
    names:
        Ordered action names. Their order defines the one-hot column index.
    encoding:
        Action encoding used by public arrays.
    """

    names: tuple[str, ...]
    encoding: ActionEncoding = "one_hot"

    def __init__(self, names: Sequence[str], encoding: ActionEncoding = "one_hot") -> None:
        clean = tuple(str(name) for name in names)
        if not clean:
            raise ValueError("ActionSpec.names must be nonempty.")
        if len(set(clean)) != len(clean):
            raise ValueError("ActionSpec.names must be unique.")
        if encoding != "one_hot":
            raise ValueError("Only one_hot action encoding is currently supported.")
        self.names = clean
        self.encoding = encoding

    @property
    def n_actions(self) -> int:
        return len(self.names)

    def validate_actions(self, actions: Array, *, name: str = "actions") -> None:
        arr = np.asarray(actions)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D one-hot action array.")
        if arr.shape[1] != self.n_actions:
            raise ValueError(f"{name} must have {self.n_actions} action columns.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"{name} must be in [0, 1].")
        if arr.shape[0] and not np.allclose(arr.sum(axis=1), 1.0):
            raise ValueError(f"{name} must have exactly one active action per row.")


@dataclass
class LongitudinalDataset:
    """Estimator-visible longitudinal benchmark data.

    The dataset deliberately contains no scorer-only truth. Use
    :class:`TruthBundle` for oracle values, latent parameters, private scenario
    labels, target Monte Carlo values, or true density ratios.
    """

    name: str
    family: BenchmarkFamily
    scenario: str
    unit_id: Array
    time: Array
    states: Array
    actions: Array
    rewards: Array
    next_states: Array
    terminals: Array
    action_available: Array
    missingness_mask: Array
    censoring: Array
    behavior_propensity: Array
    target_actions: Array
    target_propensity: Array
    target_propensity_observed_action: Array
    next_target_actions: Array
    gamma: float
    seed: int
    splits: dict[SplitName, Array]
    initial_states: Array
    initial_actions: Array
    outcome_components: dict[str, Array] = field(default_factory=dict)
    metadata_public: dict[str, Any] = field(default_factory=dict)
    information_contract: EstimatorInformationContract | None = None
    assigned_arm: Array | None = None
    received_treatment: Array | None = None
    assignment_propensity: Array | None = None
    action_spec: ActionSpec | None = None
    target_action_probabilities: Array | None = None
    next_target_action_probabilities: Array | None = None
    initial_action_probabilities: Array | None = None
    action_dose: Array | None = None
    target_action_dose: Array | None = None
    dose_available: Array | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.states).shape[0])
        if n == 0:
            raise ValueError("states must be nonempty.")
        for name in (
            "unit_id",
            "time",
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
        ):
            value = np.asarray(getattr(self, name))
            if value.shape[0] != n:
                raise ValueError(f"{name} must have {n} rows.")
        if np.asarray(self.states).ndim != 2 or np.asarray(self.next_states).ndim != 2:
            raise ValueError("states and next_states must be 2D arrays.")
        if np.asarray(self.actions).ndim != 2 or np.asarray(self.target_actions).ndim != 2:
            raise ValueError("actions and target_actions must be 2D arrays.")
        if np.asarray(self.states).shape != np.asarray(self.next_states).shape:
            raise ValueError("states and next_states must have identical shape.")
        if np.asarray(self.actions).shape != np.asarray(self.target_actions).shape:
            raise ValueError("actions and target_actions must have identical shape.")
        if self.action_spec is None:
            self.action_spec = _infer_action_spec(self.metadata_public, np.asarray(self.actions).shape[1])
        self.action_spec.validate_actions(self.actions, name="actions")
        self.action_spec.validate_actions(self.target_actions, name="target_actions")
        self.action_spec.validate_actions(self.next_target_actions, name="next_target_actions")
        if np.asarray(self.initial_actions).ndim != 2 or np.asarray(self.initial_actions).shape[1] != self.action_dim:
            raise ValueError("initial_actions must be a 2D array with action_dim columns.")
        self.action_spec.validate_actions(self.initial_actions, name="initial_actions")
        if np.asarray(self.action_available).shape != np.asarray(self.actions).shape:
            raise ValueError("action_available must match actions shape.")
        if np.asarray(self.missingness_mask).shape != np.asarray(self.states).shape:
            raise ValueError("missingness_mask must match states shape.")
        if not (0.0 <= float(self.gamma) < 1.0):
            raise ValueError("gamma must be in [0, 1).")
        for key, values in self.splits.items():
            if key not in {"train", "eval", "initial"}:
                raise ValueError(f"Unknown split '{key}'.")
            idx = np.asarray(values, dtype=np.int64)
            if idx.ndim != 1:
                raise ValueError(f"split {key} must be a 1D index array.")
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"split {key} contains out-of-bounds row indices.")
        for name, value in self.outcome_components.items():
            arr = np.asarray(value)
            if arr.shape[0] != n:
                raise ValueError(f"outcome component {name} must have {n} rows.")
        for name in ("assigned_arm", "received_treatment", "assignment_propensity"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape[0] != n:
                raise ValueError(f"{name} must have {n} rows when supplied.")
        for name in ("action_dose", "target_action_dose", "dose_available"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value)
                if arr.reshape(-1).shape[0] != n:
                    raise ValueError(f"{name} must have {n} rows when supplied.")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} must contain only finite values.")
        self.target_action_probabilities = _coerce_probability_matrix(
            self.target_action_probabilities,
            fallback=self.target_propensity if np.asarray(self.target_propensity).ndim == 2 else None,
            n_rows=n,
            n_actions=self.action_dim,
            name="target_action_probabilities",
        )
        self.next_target_action_probabilities = _coerce_probability_matrix(
            self.next_target_action_probabilities,
            fallback=None,
            n_rows=n,
            n_actions=self.action_dim,
            name="next_target_action_probabilities",
        )
        self.initial_action_probabilities = _coerce_probability_matrix(
            self.initial_action_probabilities,
            fallback=None,
            n_rows=np.asarray(self.initial_actions).shape[0],
            n_actions=self.action_dim,
            name="initial_action_probabilities",
        )
        assert_no_forbidden_public_keys(self.metadata_public)
        if self.information_contract is not None:
            self.information_contract.assert_public_metadata_allowed(self.metadata_public)

    @property
    def n(self) -> int:
        return int(np.asarray(self.states).shape[0])

    @property
    def state_dim(self) -> int:
        return int(np.asarray(self.states).shape[1])

    @property
    def action_dim(self) -> int:
        return int(np.asarray(self.actions).shape[1])

    @property
    def target_policy_action_probabilities(self) -> Array | None:
        return self.target_action_probabilities

    @property
    def next_target_policy_action_probabilities(self) -> Array | None:
        return self.next_target_action_probabilities

    @property
    def initial_target_action_probabilities(self) -> Array | None:
        return self.initial_action_probabilities


@dataclass
class TruthBundle:
    """Scorer-only oracle quantities for a benchmark problem."""

    dataset_name: str
    family: BenchmarkFamily
    values: dict[str, float] = field(default_factory=dict)
    effects: dict[str, float] = field(default_factory=dict)
    survival_curves: dict[str, Array] = field(default_factory=dict)
    rmst: dict[str, float] = field(default_factory=dict)
    subgroup_effects: dict[str, float] = field(default_factory=dict)
    oracle_ratios: dict[str, Array] = field(default_factory=dict)
    latent_parameters: dict[str, Any] = field(default_factory=dict)
    target_mc_values: dict[str, Any] = field(default_factory=dict)
    target_standard_errors: dict[str, float] = field(default_factory=dict)
    truth_noise_floor: dict[str, float] = field(default_factory=dict)
    mc_rollouts: int | None = None
    private_metadata: dict[str, Any] = field(default_factory=dict)
    leaderboard_eligible: bool = True


@dataclass
class BenchmarkProblem:
    """One benchmark problem with public data and sealed truth."""

    dataset: LongitudinalDataset
    truth: TruthBundle

    def __post_init__(self) -> None:
        if self.dataset.name != self.truth.dataset_name:
            raise ValueError("dataset and truth must refer to the same dataset name.")
        if self.dataset.family != self.truth.family:
            raise ValueError("dataset and truth families must match.")


@dataclass(frozen=True)
class FixedPolicy:
    """Named fixed policy used for evaluation, not learned from data."""

    name: str
    family: BenchmarkFamily
    probability_fn: Callable[[Array, Array, Array], Array]

    def probabilities(self, states: Array, availability: Array, time: Array) -> Array:
        probs = np.asarray(self.probability_fn(states, availability, time), dtype=np.float64)
        return normalize_action_probs(probs, availability)

    def sample_actions(self, states: Array, availability: Array, time: Array, rng: np.random.Generator) -> Array:
        probs = self.probabilities(states, availability, time)
        out = np.zeros_like(probs)
        for i, row in enumerate(probs):
            out[i, int(rng.choice(row.shape[0], p=row))] = 1.0
        return out


def normalize_action_probs(probs: Array, availability: Array) -> Array:
    """Normalize action probabilities after enforcing availability masks."""
    arr = np.asarray(probs, dtype=np.float64)
    avail = np.asarray(availability, dtype=np.float64)
    if arr.shape != avail.shape:
        raise ValueError("probs and availability must have the same shape.")
    masked = np.clip(arr, 0.0, np.inf) * (avail > 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    empty = totals.reshape(-1) <= 0.0
    if np.any(empty):
        fallback = np.zeros_like(masked[empty])
        first_available = np.argmax(avail[empty] > 0.0, axis=1)
        fallback[np.arange(fallback.shape[0]), first_available] = 1.0
        masked[empty] = fallback
        totals = masked.sum(axis=1, keepdims=True)
    return masked / totals


def one_hot(indices: Array, size: int) -> Array:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= int(size)):
        raise ValueError("indices out of bounds for one_hot size.")
    out = np.zeros((idx.shape[0], int(size)), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def action_index(actions: Array) -> Array:
    arr = np.asarray(actions)
    if arr.ndim == 1 or arr.shape[1] == 1:
        return arr.reshape(-1).astype(np.int64)
    return np.argmax(arr, axis=1).astype(np.int64)


def chosen_probability(probs: Array, actions: Array) -> Array:
    idx = action_index(actions)
    arr = np.asarray(probs, dtype=np.float64)
    return np.clip(arr[np.arange(idx.shape[0]), idx], 1e-12, np.inf)


def _infer_action_spec(metadata: dict[str, Any], n_actions: int) -> ActionSpec:
    raw_names = metadata.get("action_names")
    if raw_names is None:
        return ActionSpec(tuple(f"action_{i}" for i in range(int(n_actions))))
    if isinstance(raw_names, str):
        names = tuple(part for part in raw_names.split("|") if part)
    else:
        names = tuple(str(part) for part in raw_names)
    if len(names) != int(n_actions):
        raise ValueError("metadata_public['action_names'] must match action_dim.")
    return ActionSpec(names)


def _coerce_probability_matrix(
    value: Array | None,
    *,
    fallback: Array | None,
    n_rows: int,
    n_actions: int,
    name: str,
) -> Array | None:
    source = fallback if value is None else value
    if source is None:
        return None
    arr = np.asarray(source, dtype=np.float64)
    if arr.shape != (int(n_rows), int(n_actions)):
        raise ValueError(f"{name} must have shape ({int(n_rows)}, {int(n_actions)}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    totals = arr.sum(axis=1)
    if arr.shape[0] and not np.allclose(totals, 1.0):
        raise ValueError(f"{name} rows must sum to 1.")
    return arr


def assert_no_forbidden_public_keys(metadata: dict[str, Any]) -> None:
    for key in metadata:
        lowered = str(key).lower()
        if any(token in lowered for token in TRUTH_FORBIDDEN_TOKENS):
            raise ValueError(f"Public metadata key '{key}' looks like truth/private information.")
