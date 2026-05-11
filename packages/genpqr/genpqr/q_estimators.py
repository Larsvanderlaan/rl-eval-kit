"""Q-estimation backends for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Literal

import numpy as np

from genpqr.action_head_fqe import ActionHeadNeuralFQEstimator
from genpqr.deeppqr import DeepPQRAnchorQEstimator
from genpqr.exceptions import GenPQRAdapterError, GenPQRConfigurationError, GenPQRMissingDependencyError
from genpqr.neural_deeppqr import NeuralDeepPQRAnchorQEstimator
from genpqr.registry import register_q_estimator, resolve_registered_q_estimator
from genpqr.types import ActionSpaceSpec, Array, EstimatedPolicy, NormalizationPolicy
from genpqr.validation import as_1d_float, as_2d_float, optional_terminals


QFamily = Literal["neural", "boosted"]


@dataclass
class WrappedFittedQFunction:
    """Adapter around a fitted Q model from another backend."""

    model: Any
    action_space: ActionSpaceSpec
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    action_encoding: Literal["matrix", "index"] = "matrix"

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Predict Q-values for state-action rows."""

        states_2d = as_2d_float(states, "states")
        if self.action_encoding == "index":
            encoded = self.action_space.action_indices(actions, n_rows=states_2d.shape[0])
        else:
            encoded = self.action_space.action_matrix(actions, n_rows=states_2d.shape[0])
        if not hasattr(self.model, "predict_q"):
            raise GenPQRAdapterError(f"{self.backend} model does not expose predict_q.")
        values = as_1d_float(
            self.model.predict_q(states_2d, encoded),
            f"{self.backend} model predict_q",
            n_rows=states_2d.shape[0],
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{self.backend} model returned invalid Q predictions.")
        return values

    def predict_q_matrix(self, states: Array) -> Array:
        """Predict all finite-action Q-values."""

        if self.action_space.kind != "discrete":
            raise GenPQRConfigurationError("predict_q_matrix is only available for discrete action spaces.")
        states_2d = as_2d_float(states, "states")
        q_cols = []
        for action in range(int(self.action_space.n_actions)):
            q_cols.append(self.predict_q(states_2d, np.full(states_2d.shape[0], action, dtype=np.int64)))
        return np.stack(q_cols, axis=1)

    def expected_q(
        self,
        states: Array,
        normalization_policy: NormalizationPolicy,
        *,
        n_action_samples: int,
        rng: np.random.Generator,
    ) -> Array:
        """Estimate ``E_mu[Q(s, A)]``."""

        states_2d = as_2d_float(states, "states")
        if self.action_space.kind == "discrete" and hasattr(normalization_policy, "predict_proba"):
            probs = normalization_policy.predict_proba(states_2d)  # type: ignore[attr-defined]
            return np.sum(probs * self.predict_q_matrix(states_2d), axis=1)
        samples = normalization_policy.sample(states_2d, rng, int(n_action_samples))
        return _average_sampled_q(self, states_2d, samples, self.action_space)


@dataclass
class ConstantFittedQFunction:
    """Small portable constant-Q function for smoke tests and baselines."""

    action_space: ActionSpaceSpec
    value: float = 0.0
    backend: str = "constant_q"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.diagnostics:
            self.diagnostics = {"backend": self.backend}

    def predict_q(self, states: Array, actions: Array) -> Array:
        """Return a constant Q-value for every state-action row."""

        states_2d = as_2d_float(states, "states")
        self.action_space.validate_actions(actions, n_rows=states_2d.shape[0])
        return np.full(states_2d.shape[0], float(self.value), dtype=np.float64)

    def expected_q(
        self,
        states: Array,
        normalization_policy: NormalizationPolicy,
        *,
        n_action_samples: int,
        rng: np.random.Generator,
    ) -> Array:
        """Return the exact constant expectation under any normalization policy."""

        del normalization_policy, n_action_samples, rng
        return np.full(as_2d_float(states, "states").shape[0], float(self.value), dtype=np.float64)


@dataclass
class FQEQEstimator:
    """Adapter for the repository's production `fqe` package."""

    family: QFamily = "neural"
    config: Any | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    n_next_action_samples: int = 8

    def preflight(self, **_: Any) -> None:
        """Validate FQE adapter context before optional imports."""

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
    ) -> WrappedFittedQFunction:
        """Fit FQE on GenPQR pseudo-rewards."""

        del policy
        action_space = normalization_policy.action_space
        encoded_actions = action_space.action_matrix(actions, n_rows=np.asarray(states).shape[0])
        if self.n_next_action_samples <= 0:
            raise ValueError("n_next_action_samples must be positive.")

        def sampler(next_states_arg: Array, rng: np.random.Generator, n_samples: int) -> Array:
            sampled = normalization_policy.sample(next_states_arg, rng, n_samples)
            return action_space.encode_samples(sampled, n_rows=next_states_arg.shape[0], name="next normalization actions")

        if self.family == "neural":
            try:
                from fqe import NeuralFQEConfig, fit_fqe_neural_from_policy
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise GenPQRMissingDependencyError("Install genpqr[fqe,torch] to use neural FQE.") from exc
            config = self.config
            if config is None:
                config = NeuralFQEConfig.stable_defaults(**self.config_overrides)
            elif self.config_overrides:
                raise GenPQRConfigurationError("Pass either config or config_overrides, not both.")
            model = fit_fqe_neural_from_policy(
                states=states,
                actions=encoded_actions,
                next_states=next_states,
                rewards=pseudo_rewards,
                gamma=gamma,
                next_action_sampler=sampler,
                n_next_action_samples=int(self.n_next_action_samples),
                terminals=terminals,
                sample_weight=sample_weight,
                config=config,
            )
            return WrappedFittedQFunction(model=model, action_space=action_space, backend="fqe_neural", diagnostics=dict(model.diagnostics))
        if self.family == "boosted":
            try:
                from fqe import BoostedFQEConfig, fit_fqe_from_policy
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise GenPQRMissingDependencyError("Install genpqr[fqe] to use boosted FQE.") from exc
            config = self.config
            if config is None:
                config = BoostedFQEConfig.stable_defaults(**self.config_overrides)
            elif self.config_overrides:
                raise GenPQRConfigurationError("Pass either config or config_overrides, not both.")
            model = fit_fqe_from_policy(
                states=states,
                actions=encoded_actions,
                next_states=next_states,
                rewards=pseudo_rewards,
                gamma=gamma,
                next_action_sampler=sampler,
                n_next_action_samples=int(self.n_next_action_samples),
                terminals=terminals,
                sample_weight=sample_weight,
                config=config,
            )
            return WrappedFittedQFunction(model=model, action_space=action_space, backend="fqe_boosted", diagnostics=dict(model.diagnostics))
        raise GenPQRConfigurationError(f"Unknown FQE family '{self.family}'.")


@dataclass
class AutoNeuralFQEstimator:
    """Action-aware neural FQE router.

    Discrete action spaces use the GenPQR action-head neural backend by default
    to avoid over-sharing across one-hot action features. Continuous action
    spaces continue to use the generic neural FQE adapter.
    """

    config: Any | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    n_next_action_samples: int = 8
    prefer_action_head_for_discrete: bool = True

    def preflight(self, **_: Any) -> None:
        """Validate shared configuration before optional imports."""

        if self.config is not None and self.config_overrides:
            raise GenPQRConfigurationError("Pass either config or config_overrides, not both.")

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
    ) -> Any:
        """Fit an action-head discrete FQE or generic continuous neural FQE."""

        self.preflight()
        action_space = normalization_policy.action_space
        if action_space.kind == "discrete" and bool(self.prefer_action_head_for_discrete):
            return ActionHeadNeuralFQEstimator(
                config=self.config,
                config_overrides=dict(self.config_overrides),
                n_next_action_samples=int(self.n_next_action_samples),
            ).fit(
                states=states,
                actions=actions,
                next_states=next_states,
                pseudo_rewards=pseudo_rewards,
                normalization_policy=normalization_policy,
                gamma=gamma,
                terminals=terminals,
                sample_weight=sample_weight,
                policy=policy,
            )
        return FQEQEstimator(
            family="neural",
            config=self.config,
            config_overrides=dict(self.config_overrides),
            n_next_action_samples=int(self.n_next_action_samples),
        ).fit(
            states=states,
            actions=actions,
            next_states=next_states,
            pseudo_rewards=pseudo_rewards,
            normalization_policy=normalization_policy,
            gamma=gamma,
            terminals=terminals,
            sample_weight=sample_weight,
            policy=policy,
        )


@dataclass
class D3RLPYFQEstimator:
    """Lazy adapter for d3rlpy FQE.

    d3rlpy FQE evaluates a d3rlpy algorithm object, so callers must provide the
    policy/algorithm to evaluate through ``algo``.
    """

    algo: Any
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    n_steps: int = 10_000
    device: str | bool | int = "cpu:0"
    allow_ordered_episodes: bool = False

    def preflight(self, *, episode_ids: Array | None = None, dataset_metadata: dict[str, Any] | None = None, **_: Any) -> None:
        """Validate ordered-episode context before importing d3rlpy."""

        if not self.allow_ordered_episodes:
            raise GenPQRConfigurationError(
                "d3rlpy FQE consumes ordered episode datasets and cannot safely use arbitrary row-wise "
                "GenPQR transitions with explicit next_states. Set allow_ordered_episodes=True only when "
                "the input rows are ordered episodes compatible with d3rlpy's MDPDataset semantics."
            )
        if episode_ids is None:
            raise GenPQRConfigurationError(
                "d3rlpy FQE with allow_ordered_episodes=True requires episode_ids or an EpisodeDataset."
            )
        if not (dataset_metadata or {}).get("ordered_episodes_validated", False):
            raise GenPQRConfigurationError(
                "d3rlpy FQE requires strict ordered episode validation; use EpisodeDataset or "
                "TransitionDataset.from_arrays(..., strict_episodes=True)."
            )
        if not (dataset_metadata or {}).get("d3rlpy_transition_compatible", False):
            raise GenPQRConfigurationError(
                "d3rlpy FQE requires explicit next_states to match ordered MDPDataset episode shifts."
            )

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
        episode_ids: Array | None = None,
        dataset_metadata: dict[str, Any] | None = None,
    ) -> WrappedFittedQFunction:
        """Fit d3rlpy FQE using pseudo-rewards."""

        del sample_weight, policy
        self.preflight(episode_ids=episode_ids, dataset_metadata=dataset_metadata)
        action_space = normalization_policy.action_space
        obs = as_2d_float(states, "states")
        next_obs = as_2d_float(next_states, "next_states", n_rows=obs.shape[0])
        if next_obs.shape != obs.shape:
            raise ValueError("next_states must match states shape for d3rlpy FQE.")
        _validate_d3rlpy_shifted_transitions(
            states=obs,
            next_states=next_obs,
            terminals=terminals,
            episode_ids=episode_ids,
        )
        pseudo_rewards_1d = as_1d_float(pseudo_rewards, "pseudo_rewards", n_rows=obs.shape[0])
        try:
            from d3rlpy.dataset import MDPDataset
            from d3rlpy.ope import DiscreteFQE, FQE, FQEConfig
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError("Install genpqr[d3rlpy] to use d3rlpy FQE.") from exc
        acts = action_space.action_indices(actions, n_rows=obs.shape[0]) if action_space.kind == "discrete" else action_space.action_matrix(actions, n_rows=obs.shape[0])
        dones = np.zeros(obs.shape[0], dtype=bool) if terminals is None else np.asarray(terminals).reshape(-1).astype(bool)
        dataset = MDPDataset(observations=obs, actions=acts, rewards=pseudo_rewards_1d, terminals=dones)
        config_kwargs = {"gamma": gamma, **self.config_kwargs}
        config = FQEConfig(**config_kwargs)
        fqe_cls = DiscreteFQE if action_space.kind == "discrete" else FQE
        fqe = fqe_cls(algo=self.algo, config=config, device=self.device)
        fqe.fit(dataset, n_steps=int(self.n_steps))
        action_encoding = "index" if action_space.kind == "discrete" else "matrix"
        return WrappedFittedQFunction(
            model=_D3RLPYQModel(fqe),
            action_space=action_space,
            backend="d3rlpy_fqe",
            action_encoding=action_encoding,
        )


@dataclass
class _D3RLPYQModel:
    fqe: Any

    def predict_q(self, states: Array, actions: Array) -> Array:
        if hasattr(self.fqe, "predict_value"):
            return as_1d_float(self.fqe.predict_value(states, actions), "d3rlpy predict_value", n_rows=np.asarray(states).shape[0])
        if hasattr(self.fqe, "predict"):
            return as_1d_float(self.fqe.predict(states, actions), "d3rlpy predict", n_rows=np.asarray(states).shape[0])
        raise GenPQRAdapterError("d3rlpy FQE object does not expose a known Q prediction method.")


@dataclass
class ScopeRLDatasetBoundQEstimator:
    """Lazy adapter for SCOPE-RL value-learning methods.

    The adapter maps GenPQR transitions and pseudo-rewards to a SCOPE-RL-style
    logged dataset, then asks SCOPE-RL to produce value predictions. Because
    SCOPE-RL APIs are environment/evaluation-policy centric, callers must
    provide ``env`` and ``evaluation_policies`` when using this backend.
    """

    method: Literal["fqe", "mql"] = "mql"
    env: Any | None = None
    evaluation_policies: Any | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    prediction_key: str | None = None
    allow_dataset_bound_predictions: bool = False

    def preflight(self, **_: Any) -> None:
        """Validate required context before importing SCOPE-RL."""

        if self.env is None or self.evaluation_policies is None:
            raise GenPQRConfigurationError(
                "SCOPE-RL Q adapters require env and evaluation_policies; "
                "use fqe_neural/fqe_boosted or pass a custom QEstimator for offline-only workflows."
            )
        if not self.allow_dataset_bound_predictions:
            raise GenPQRAdapterError(
                "SCOPE-RL's public OPE input path returns dataset-bound prediction arrays, not a reusable "
                "Q-function object. Set allow_dataset_bound_predictions=True only for fitted-row diagnostics, "
                "or pass a custom QEstimator that wraps a reusable SCOPE-RL value model."
            )

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
    ) -> WrappedFittedQFunction:
        """Fit SCOPE-RL value learning and wrap the resulting predictions."""

        del next_states, gamma, sample_weight, policy
        self.preflight()
        try:
            from scope_rl.ope import CreateOPEInput
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError("Install genpqr[scope-rl] to use SCOPE-RL value learning.") from exc
        action_space = normalization_policy.action_space
        logged_dataset = scope_logged_dataset_from_transitions(
            states=states,
            actions=actions,
            rewards=pseudo_rewards,
            terminals=terminals,
            action_space=action_space,
        )
        prep = CreateOPEInput(env=self.env, model_args=self.model_args)
        q_method = "mql" if self.method == "mql" else "fqe"
        input_dict = prep.obtain_whole_inputs(
            logged_dataset=logged_dataset,
            evaluation_policies=self.evaluation_policies,
            require_value_prediction=True,
            q_function_method=q_method,
        )
        key = self.prediction_key or _find_scope_prediction_key(input_dict)
        if key is None:
            raise GenPQRAdapterError(
                "SCOPE-RL did not return a recognized Q prediction key; pass prediction_key explicitly."
            )
        raise GenPQRAdapterError(
            "SCOPE-RL dataset-bound predictions are diagnostics-only and are not an action-aware reusable "
            "Q function. Use ReusableScopeRLQEstimator with a model exposing predict_q(...)."
        )


ScopeRLQEstimator = ScopeRLDatasetBoundQEstimator


@dataclass
class _ScopePredictionModel:
    input_dict: dict[str, Any]
    prediction_key: str
    n_rows: int

    def predict_q(self, states: Array, actions: Array) -> Array:
        del actions
        if np.asarray(states).shape[0] != int(self.n_rows):
            raise GenPQRAdapterError("Dataset-bound SCOPE-RL predictions can only be used on the fitted rows.")
        pred = np.asarray(self.input_dict[self.prediction_key], dtype=np.float64)
        if pred.ndim > 1:
            pred = pred.reshape(pred.shape[0], -1)[:, 0]
        return as_1d_float(pred, "scope_rl dataset-bound prediction", n_rows=self.n_rows)


@dataclass
class ReusableScopeRLQEstimator:
    """Adapter for user-provided reusable SCOPE-RL-style value models.

    The model or factory must return an object exposing ``predict_q`` after
    fitting. This avoids depending on SCOPE-RL's dataset-bound OPE input path.
    """

    model_or_factory: Any
    fit_kwargs: dict[str, Any] = field(default_factory=dict)
    action_encoding: Literal["matrix", "index"] = "matrix"

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        pseudo_rewards: Array,
        normalization_policy: NormalizationPolicy,
        gamma: float,
        terminals: Array | None = None,
        sample_weight: Array | None = None,
        policy: EstimatedPolicy | None = None,
    ) -> WrappedFittedQFunction:
        """Fit a reusable SCOPE-RL-style value model."""

        del policy
        action_space = normalization_policy.action_space
        model = (
            self.model_or_factory()
            if inspect.isclass(self.model_or_factory)
            or (callable(self.model_or_factory) and not hasattr(self.model_or_factory, "predict_q"))
            else self.model_or_factory
        )
        if not hasattr(model, "fit") or not hasattr(model, "predict_q"):
            raise GenPQRAdapterError("ReusableScopeRLQEstimator requires a model with fit(...) and predict_q(...).")
        encoded_actions = (
            action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
            if self.action_encoding == "index"
            else action_space.action_matrix(actions, n_rows=np.asarray(states).shape[0])
        )
        model.fit(
            states=states,
            actions=encoded_actions,
            next_states=next_states,
            rewards=pseudo_rewards,
            gamma=gamma,
            terminals=terminals,
            sample_weight=sample_weight,
            **self.fit_kwargs,
        )
        return WrappedFittedQFunction(
            model=model,
            action_space=action_space,
            backend="scope_rl_reusable",
            action_encoding=self.action_encoding,
        )


def scope_logged_dataset_from_transitions(
    *,
    states: Array,
    actions: Array,
    rewards: Array,
    terminals: Array | None,
    action_space: ActionSpaceSpec,
) -> dict[str, Any]:
    """Create a minimal SCOPE-RL-style logged-dataset payload."""

    if action_space.kind != "discrete":
        raise GenPQRConfigurationError("The built-in SCOPE-RL adapter currently supports discrete actions.")
    states_2d = as_2d_float(states, "states")
    idx = action_space.action_indices(actions, n_rows=states_2d.shape[0])
    rewards_1d = as_1d_float(rewards, "rewards", n_rows=states_2d.shape[0])
    dones = np.zeros(states_2d.shape[0], dtype=bool) if terminals is None else np.asarray(terminals).reshape(-1).astype(bool)
    return {
        "size": int(states_2d.shape[0]),
        "n_trajectories": int(states_2d.shape[0]),
        "step_per_trajectory": 1,
        "action_type": "discrete",
        "n_actions": int(action_space.n_actions),
        "state_dim": int(states_2d.shape[1]),
        "state": states_2d[:, None, :],
        "action": idx.reshape(-1, 1),
        "reward": rewards_1d.reshape(-1, 1),
        "done": dones.reshape(-1, 1),
        "terminal": dones.reshape(-1, 1),
        "pscore": np.ones((states_2d.shape[0], 1), dtype=np.float64),
        "behavior_policy": "genpqr_behavior",
        "dataset_id": 0,
    }


def _find_scope_prediction_key(input_dict: dict[str, Any]) -> str | None:
    candidates = (
        "state_action_value_prediction",
        "q_function_prediction",
        "state_action_value",
        "estimated_state_action_value",
    )
    for key in candidates:
        if key in input_dict:
            return key
    return None


def _average_sampled_q(
    q_function: WrappedFittedQFunction,
    states: Array,
    samples: Array,
    action_space: ActionSpaceSpec,
) -> Array:
    arr = np.asarray(samples)
    if action_space.kind == "discrete":
        encoded = action_space.encode_samples(arr, n_rows=states.shape[0], name="sampled actions")
        if encoded.ndim == 2:
            return q_function.predict_q(states, encoded)
        values = []
        for j in range(encoded.shape[1]):
            values.append(q_function.predict_q(states, encoded[:, j, :]))
        return np.mean(np.stack(values, axis=1), axis=1)
    if arr.ndim == 2:
        return q_function.predict_q(states, arr)
    if arr.ndim != 3:
        raise ValueError("sampled actions must have shape (n, d) or (n, n_samples, d).")
    values = []
    for j in range(arr.shape[1]):
        values.append(q_function.predict_q(states, arr[:, j, :]))
    return np.mean(np.stack(values, axis=1), axis=1)


def _validate_d3rlpy_shifted_transitions(
    *,
    states: Array,
    next_states: Array,
    terminals: Array | None,
    episode_ids: Array | None,
) -> None:
    states_2d = as_2d_float(states, "states")
    next_states_2d = as_2d_float(next_states, "next_states", n_rows=states_2d.shape[0])
    if episode_ids is None:
        raise GenPQRConfigurationError("d3rlpy FQE requires episode_ids.")
    ids = np.asarray(episode_ids).reshape(-1)
    if ids.shape[0] != states_2d.shape[0]:
        raise ValueError("episode_ids must contain one value per row.")
    done = optional_terminals(terminals, states_2d.shape[0]) > 0.0
    if states_2d.shape[0] <= 1:
        return
    same_episode_next = ids[1:] == ids[:-1]
    check = same_episode_next & ~done[:-1]
    if np.any(check) and not np.allclose(next_states_2d[:-1][check], states_2d[1:][check]):
        raise GenPQRConfigurationError(
            "d3rlpy FQE requires next_states[i] to equal states[i+1] within each nonterminal episode."
        )


def resolve_q_estimator(q: str | Any, **kwargs: Any) -> Any:
    """Resolve a named or user-supplied Q estimator."""

    if not isinstance(q, str):
        return q
    key = q.lower()
    aliases = {
        "neural_fqe": "fqe_neural",
        "boosted_fqe": "fqe_boosted",
        "neural_fqe_auto": "auto_neural_fqe",
        "fqe_neural_auto": "auto_neural_fqe",
        "auto_fqe": "auto_neural_fqe",
        "scope_fqe": "scope_rl_fqe",
        "scope_mql": "scope_rl_mql",
        "minimax_q": "scope_rl_mql",
        "neural_deeppqr": "deeppqr_neural",
        "deep_pqr_neural": "deeppqr_neural",
    }
    return resolve_registered_q_estimator(aliases.get(key, key), **kwargs)


def _register_builtin_q_estimators() -> None:
    register_q_estimator("fqe_neural", lambda **kwargs: FQEQEstimator(family="neural", **kwargs), overwrite=True)
    register_q_estimator("fqe_boosted", lambda **kwargs: FQEQEstimator(family="boosted", **kwargs), overwrite=True)
    register_q_estimator("auto_neural_fqe", AutoNeuralFQEstimator, overwrite=True)
    register_q_estimator("fqe_action_head_neural", ActionHeadNeuralFQEstimator, overwrite=True)
    register_q_estimator("action_head_neural_fqe", ActionHeadNeuralFQEstimator, overwrite=True)
    register_q_estimator("stratified_neural_fqe", ActionHeadNeuralFQEstimator, overwrite=True)
    register_q_estimator("d3rlpy_fqe", D3RLPYFQEstimator, overwrite=True)
    register_q_estimator("scope_rl_fqe", lambda **kwargs: ScopeRLDatasetBoundQEstimator(method="fqe", **kwargs), overwrite=True)
    register_q_estimator("scope_rl_mql", lambda **kwargs: ScopeRLDatasetBoundQEstimator(method="mql", **kwargs), overwrite=True)
    register_q_estimator(
        "scope_rl_dataset_bound_fqe",
        lambda **kwargs: ScopeRLDatasetBoundQEstimator(method="fqe", **kwargs),
        overwrite=True,
    )
    register_q_estimator(
        "scope_rl_dataset_bound_mql",
        lambda **kwargs: ScopeRLDatasetBoundQEstimator(method="mql", **kwargs),
        overwrite=True,
    )
    register_q_estimator("deeppqr_linear", DeepPQRAnchorQEstimator, overwrite=True)
    register_q_estimator("deeppqr_neural", NeuralDeepPQRAnchorQEstimator, overwrite=True)
    register_q_estimator("neural_deeppqr", NeuralDeepPQRAnchorQEstimator, overwrite=True)


_register_builtin_q_estimators()
