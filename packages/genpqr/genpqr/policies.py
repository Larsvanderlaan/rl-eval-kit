"""Policy-estimation backends for GenPQR."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Literal

import numpy as np

from genpqr.exceptions import GenPQRAdapterError, GenPQRConfigurationError, GenPQRMissingDependencyError
from genpqr.registry import register_policy_estimator, registered_policy_estimator_factory, resolve_registered_policy_estimator
from genpqr.types import ActionSpaceSpec, Array
from genpqr.validation import as_1d_float, as_2d_float, optional_weights


PolicyName = Literal[
    "airl",
    "deep_airl",
    "deep_gail",
    "deep_bc",
    "imitation_airl",
    "imitation_gail",
    "imitation_bc",
    "behavior_cloning_native",
    "d3rlpy_bc",
]


def _standardize_fit(states: Array) -> tuple[Array, Array]:
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _features(states: Array, mean: Array, std: Array, *, quadratic: bool = True) -> Array:
    x = (as_2d_float(states, "states") - mean) / std
    parts = [np.ones((x.shape[0], 1), dtype=np.float64), x]
    if quadratic:
        parts.append(x**2)
    return np.concatenate(parts, axis=1)


def _softmax(logits: Array) -> Array:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-300, None)


@dataclass
class NativeDiscretePolicy:
    """Dependency-light softmax behavior policy."""

    theta: Array
    input_mean: Array
    input_std: Array
    action_space: ActionSpaceSpec
    prob_clip_min: float = 1e-4
    prob_clip_max: float = 1.0

    def predict_logits(self, states: Array) -> Array:
        """Return action logits."""

        return _features(states, self.input_mean, self.input_std) @ self.theta

    def predict_proba(self, states: Array) -> Array:
        """Return action probabilities."""

        probs = _softmax(self.predict_logits(states))
        if self.prob_clip_min > 0.0 or self.prob_clip_max < 1.0:
            probs = np.clip(probs, self.prob_clip_min, self.prob_clip_max)
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)
        return probs

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return log probabilities for state-action rows."""

        probs = self.predict_proba(states)
        idx = self.action_space.action_indices(actions, n_rows=probs.shape[0])
        return np.log(np.clip(probs[np.arange(probs.shape[0]), idx], 1e-300, None))

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample finite actions from the policy."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        probs = self.predict_proba(states)
        draws = np.empty((probs.shape[0], int(n_samples)), dtype=np.int64)
        choices = np.arange(int(self.action_space.n_actions), dtype=np.int64)
        for i, row in enumerate(probs):
            draws[i] = rng.choice(choices, size=int(n_samples), p=row)
        return draws.reshape(-1) if int(n_samples) == 1 else draws


@dataclass
class NativeGaussianPolicy:
    """Linear-Gaussian behavior policy for continuous actions."""

    beta: Array
    input_mean: Array
    input_std: Array
    covariance_diag: Array
    action_space: ActionSpaceSpec

    def predict_mean(self, states: Array) -> Array:
        """Return conditional action means."""

        return _features(states, self.input_mean, self.input_std) @ self.beta

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return diagonal-Gaussian log densities."""

        states_2d = as_2d_float(states, "states")
        actions_2d = self.action_space.action_matrix(actions, n_rows=states_2d.shape[0])
        mean = self.predict_mean(states_2d)
        var = np.clip(self.covariance_diag, 1e-10, None)
        diff = actions_2d - mean
        const = np.log(2.0 * np.pi * var).sum()
        return -0.5 * (const + ((diff**2) / var).sum(axis=1))

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample continuous actions."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        mean = self.predict_mean(states)
        scale = np.sqrt(np.clip(self.covariance_diag, 1e-10, None))
        draws = rng.normal(loc=mean[:, None, :], scale=scale.reshape(1, 1, -1), size=(mean.shape[0], int(n_samples), mean.shape[1]))
        return draws[:, 0, :] if int(n_samples) == 1 else draws


@dataclass
class BehaviorCloningPolicyEstimator:
    """Dependency-light behavior cloning.

    Discrete actions use multinomial logistic regression on standardized linear
    and quadratic state features. Continuous actions use ridge-regularized
    linear Gaussian regression.
    """

    learning_rate: float = 0.05
    n_epochs: int = 600
    l2: float = 1e-3
    ridge: float = 1e-3
    prob_clip_min: float = 1e-4
    prob_clip_max: float = 1.0
    seed: int = 123

    def preflight(self, **_: Any) -> None:
        """Validate native behavior-cloning context before fitting."""

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array | None = None,
        terminals: Array | None = None,
        action_space: ActionSpaceSpec,
        sample_weight: Array | None = None,
        env: Any | None = None,
    ) -> NativeDiscretePolicy | NativeGaussianPolicy:
        """Fit the native behavior-cloning policy."""

        del next_states, terminals, env
        states_2d = as_2d_float(states, "states")
        weights = optional_weights(sample_weight, states_2d.shape[0])
        mean, std = _standardize_fit(states_2d)
        x = _features(states_2d, mean, std)
        weights_norm = np.ones(states_2d.shape[0], dtype=np.float64) if weights is None else weights / np.mean(weights)
        if action_space.kind == "continuous":
            y = action_space.action_matrix(actions, n_rows=states_2d.shape[0])
            xtw = x.T * weights_norm.reshape(1, -1)
            gram = xtw @ x + self.ridge * np.eye(x.shape[1])
            rhs = xtw @ y
            beta = np.linalg.solve(gram, rhs)
            resid = y - x @ beta
            cov = np.average(resid**2, axis=0, weights=weights_norm)
            cov = np.maximum(cov, 1e-4)
            return NativeGaussianPolicy(
                beta=beta,
                input_mean=mean,
                input_std=std,
                covariance_diag=cov,
                action_space=action_space,
            )

        y = action_space.action_indices(actions, n_rows=states_2d.shape[0])
        rng = np.random.default_rng(self.seed)
        theta = rng.normal(scale=0.01, size=(x.shape[1], int(action_space.n_actions)))
        y_one_hot = action_space.one_hot(y)
        step = float(self.learning_rate)
        for epoch in range(int(self.n_epochs)):
            probs = _softmax(x @ theta)
            grad = x.T @ ((probs - y_one_hot) * weights_norm[:, None]) / max(x.shape[0], 1)
            grad += float(self.l2) * theta
            theta -= step * grad
            if (epoch + 1) % 200 == 0:
                step *= 0.5
        return NativeDiscretePolicy(
            theta=theta,
            input_mean=mean,
            input_std=std,
            action_space=action_space,
            prob_clip_min=float(self.prob_clip_min),
            prob_clip_max=float(self.prob_clip_max),
        )


@dataclass
class ImitationPolicyEstimator:
    """Lazy adapter for HumanCompatibleAI ``imitation`` BC/GAIL/AIRL.

    AIRL and GAIL require ``env`` because the generator policy must roll out in
    an environment. The adapter deliberately fails loudly when the optional
    stack or environment is unavailable.
    """

    algorithm: Literal["airl", "gail", "bc"] = "airl"
    total_timesteps: int = 20_000
    demo_batch_size: int = 256
    seed: int = 123
    kwargs: dict[str, Any] = field(default_factory=dict)

    def preflight(self, *, env: Any | None = None, **_: Any) -> None:
        """Validate required context before importing optional dependencies."""

        if env is None:
            raise GenPQRConfigurationError(
                f"policy='{self.algorithm}' requires env for the imitation adapter; "
                "use policy='behavior_cloning_native' for offline-only behavior cloning."
            )

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array | None = None,
        terminals: Array | None = None,
        action_space: ActionSpaceSpec,
        sample_weight: Array | None = None,
        env: Any | None = None,
    ) -> Any:
        """Fit an imitation policy and return an adapter exposing log-probs."""

        del sample_weight
        self.preflight(env=env)
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError(
                "The imitation policy adapter requires torch and stable-baselines3. "
                "Install genpqr[imitation]."
            ) from exc
        try:
            from imitation.data.types import Transitions
            from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError(
                "The imitation policy adapter requires the HumanCompatibleAI imitation package. "
                "Install genpqr[imitation]."
            ) from exc

        if action_space.kind != "discrete":
            acts = action_space.action_matrix(actions, n_rows=states.shape[0])
        else:
            acts = action_space.action_indices(actions, n_rows=states.shape[0])
        obs = as_2d_float(states, "states")
        next_obs = obs if next_states is None else as_2d_float(next_states, "next_states", n_rows=obs.shape[0])
        dones = np.zeros(obs.shape[0], dtype=bool) if terminals is None else np.asarray(terminals).reshape(-1).astype(bool)
        infos = np.array([{} for _ in range(obs.shape[0])], dtype=object)
        demonstrations = Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)
        venv = env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])
        gen_algo = PPO("MlpPolicy", venv, seed=int(self.seed), verbose=0)

        if self.algorithm == "bc":
            try:
                from imitation.algorithms.bc import BC
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise GenPQRMissingDependencyError("Install genpqr[imitation] to use imitation BC.") from exc
            trainer = BC(observation_space=venv.observation_space, action_space=venv.action_space, demonstrations=demonstrations, rng=np.random.default_rng(self.seed))
            trainer.train(n_epochs=int(self.kwargs.get("n_epochs", 20)))
            return SB3PolicyAdapter(trainer.policy, action_space)

        reward_cls = BasicShapedRewardNet if self.algorithm == "airl" else BasicRewardNet
        reward_net = reward_cls(venv.observation_space, venv.action_space)
        if self.algorithm == "airl":
            try:
                from imitation.algorithms.adversarial.airl import AIRL
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise GenPQRMissingDependencyError("Install genpqr[imitation] to use AIRL.") from exc
            trainer = AIRL(
                demonstrations=demonstrations,
                demo_batch_size=int(self.demo_batch_size),
                venv=venv,
                gen_algo=gen_algo,
                reward_net=reward_net,
                **self.kwargs,
            )
        else:
            try:
                from imitation.algorithms.adversarial.gail import GAIL
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise GenPQRMissingDependencyError("Install genpqr[imitation] to use GAIL.") from exc
            trainer = GAIL(
                demonstrations=demonstrations,
                demo_batch_size=int(self.demo_batch_size),
                venv=venv,
                gen_algo=gen_algo,
                reward_net=reward_net,
                **self.kwargs,
            )
        trainer.train(total_timesteps=int(self.total_timesteps))
        return SB3PolicyAdapter(trainer.policy, action_space, torch_module=torch)


@dataclass
class SB3PolicyAdapter:
    """Adapter exposing the GenPQR policy contract for SB3 policies."""

    policy: Any
    action_space: ActionSpaceSpec
    torch_module: Any | None = None

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return SB3 policy log probabilities."""

        torch = self.torch_module
        if torch is None:  # pragma: no cover - depends on optional stack.
            import torch as torch_import

            torch = torch_import
        obs = torch.as_tensor(as_2d_float(states, "states"), dtype=torch.float32)
        if self.action_space.kind == "discrete":
            acts_np = self.action_space.action_indices(actions, n_rows=obs.shape[0])
            acts = torch.as_tensor(acts_np, dtype=torch.long)
        else:
            acts = torch.as_tensor(self.action_space.action_matrix(actions, n_rows=obs.shape[0]), dtype=torch.float32)
        with torch.no_grad():
            dist = self.policy.get_distribution(obs)
            logp = dist.log_prob(acts)
        return logp.detach().cpu().numpy().reshape(-1).astype(np.float64)

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample from an SB3 policy.

        SB3 does not use NumPy generators internally; this method is intended
        for convenience and returns repeated independent policy samples.
        """

        del rng
        obs = as_2d_float(states, "states")
        draws = []
        for _ in range(int(n_samples)):
            action, _ = self.policy.predict(obs, deterministic=False)
            draws.append(action)
        arr = np.stack(draws, axis=1)
        return arr[:, 0] if int(n_samples) == 1 else arr


@dataclass
class D3RLPYBCPolicyEstimator:
    """Lazy d3rlpy behavior-cloning adapter."""

    n_steps: int = 10_000
    device: str | bool | int = "cpu:0"
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    allow_approximate_log_prob: bool = False

    def preflight(self, **_: Any) -> None:
        """Validate required context before importing optional dependencies."""

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array | None = None,
        terminals: Array | None = None,
        action_space: ActionSpaceSpec,
        sample_weight: Array | None = None,
        env: Any | None = None,
    ) -> Any:
        """Fit d3rlpy BC and return a policy adapter."""

        del sample_weight, env
        try:
            from d3rlpy.algos import BCConfig, DiscreteBCConfig
            from d3rlpy.dataset import MDPDataset
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
            raise GenPQRMissingDependencyError("Install genpqr[d3rlpy] to use d3rlpy BC.") from exc
        obs = as_2d_float(states, "states")
        acts = action_space.action_indices(actions, n_rows=obs.shape[0]) if action_space.kind == "discrete" else action_space.action_matrix(actions, n_rows=obs.shape[0])
        rewards = np.zeros(obs.shape[0], dtype=np.float64)
        dones = np.zeros(obs.shape[0], dtype=bool) if terminals is None else np.asarray(terminals).reshape(-1).astype(bool)
        dataset = MDPDataset(observations=obs, actions=acts, rewards=rewards, terminals=dones)
        config_cls = DiscreteBCConfig if action_space.kind == "discrete" else BCConfig
        algo = config_cls(**self.config_kwargs).create(device=self.device)
        algo.fit(dataset, n_steps=int(self.n_steps))
        return D3RLPYPolicyAdapter(
            algo,
            action_space,
            allow_approximate_log_prob=bool(self.allow_approximate_log_prob),
        )


@dataclass
class D3RLPYPolicyAdapter:
    """Adapter exposing d3rlpy algorithms as GenPQR policies."""

    algo: Any
    action_space: ActionSpaceSpec
    log_prob_floor: float = 1e-4
    allow_approximate_log_prob: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def log_prob(self, states: Array, actions: Array) -> Array:
        """Return calibrated log probabilities when the d3rlpy object exposes them."""

        obs = as_2d_float(states, "states")
        if self.action_space.kind == "discrete":
            idx = self.action_space.action_indices(actions, n_rows=obs.shape[0])
            if hasattr(self.algo, "predict_proba"):
                probs = np.asarray(self.algo.predict_proba(obs), dtype=np.float64)
            else:
                if not self.allow_approximate_log_prob:
                    raise GenPQRAdapterError(
                        "d3rlpy policy log_prob requires predict_proba for discrete actions. "
                        "Set allow_approximate_log_prob=True to opt into deterministic proxy scores."
                    )
                pred = np.asarray(self.algo.predict(obs)).reshape(-1)
                probs = np.full((obs.shape[0], int(self.action_space.n_actions)), self.log_prob_floor, dtype=np.float64)
                probs[np.arange(obs.shape[0]), pred.astype(int)] = 1.0
                probs = probs / probs.sum(axis=1, keepdims=True)
                self.diagnostics["approximate_log_prob"] = True
            if probs.shape != (obs.shape[0], int(self.action_space.n_actions)):
                raise GenPQRAdapterError("d3rlpy predict_proba must return shape (n, n_actions).")
            if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
                raise GenPQRAdapterError("d3rlpy predict_proba returned invalid probabilities.")
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)
            return np.log(np.clip(probs[np.arange(obs.shape[0]), idx], self.log_prob_floor, None))
        acts = self.action_space.action_matrix(actions, n_rows=obs.shape[0])
        if hasattr(self.algo, "log_prob"):
            return as_1d_float(self.algo.log_prob(obs, acts), "d3rlpy policy log_prob", n_rows=obs.shape[0])
        if hasattr(self.algo, "predict_log_prob"):
            return as_1d_float(
                self.algo.predict_log_prob(obs, acts),
                "d3rlpy policy predict_log_prob",
                n_rows=obs.shape[0],
            )
        if not self.allow_approximate_log_prob:
            raise GenPQRAdapterError(
                "continuous d3rlpy policies must expose log_prob/predict_log_prob. "
                "Set allow_approximate_log_prob=True to opt into unit-Gaussian proxy scores."
            )
        mean = np.asarray(self.algo.predict(obs), dtype=np.float64)
        if mean.shape != acts.shape:
            raise GenPQRAdapterError("d3rlpy continuous policy predictions must have shape (n, action_dim).")
        diff = acts - mean
        self.diagnostics["approximate_log_prob"] = True
        return -0.5 * np.sum(diff**2, axis=1)

    def sample(self, states: Array, rng: np.random.Generator, n_samples: int = 1) -> Array:
        """Sample by repeatedly querying the d3rlpy policy."""

        del rng
        obs = as_2d_float(states, "states")
        draws = [np.asarray(self.algo.sample_action(obs)) for _ in range(int(n_samples))]
        arr = np.stack(draws, axis=1)
        return arr[:, 0] if int(n_samples) == 1 else arr


def resolve_policy_estimator(policy: str | Any, **kwargs: Any) -> Any:
    """Resolve a named or user-supplied policy estimator."""

    if not isinstance(policy, str):
        return policy
    key = policy.lower()
    aliases = {
        "airl": "imitation_airl",
        "deep_airl": "imitation_airl",
        "deep_gail": "imitation_gail",
        "deep_bc": "imitation_bc",
    }
    key = aliases.get(key, key)
    return resolve_registered_policy_estimator(key, **kwargs)


def policy_estimator_accepts_parameter(policy: str, parameter: str) -> bool:
    """Return whether a named policy factory can receive ``parameter``."""

    key = policy.lower()
    aliases = {
        "airl": "imitation_airl",
        "deep_airl": "imitation_airl",
        "deep_gail": "imitation_gail",
        "deep_bc": "imitation_bc",
    }
    factory = registered_policy_estimator_factory(aliases.get(key, key))
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD or name == parameter
        for name, param in signature.parameters.items()
    )


def _register_builtin_policies() -> None:
    register_policy_estimator("behavior_cloning_native", BehaviorCloningPolicyEstimator, overwrite=True)
    register_policy_estimator(
        "imitation_airl",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="airl", **kwargs),
        overwrite=True,
    )
    register_policy_estimator(
        "imitation_gail",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="gail", **kwargs),
        overwrite=True,
    )
    register_policy_estimator(
        "imitation_bc",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="bc", **kwargs),
        overwrite=True,
    )
    register_policy_estimator(
        "deep_airl",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="airl", **kwargs),
        overwrite=True,
    )
    register_policy_estimator(
        "deep_gail",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="gail", **kwargs),
        overwrite=True,
    )
    register_policy_estimator(
        "deep_bc",
        lambda **kwargs: ImitationPolicyEstimator(algorithm="bc", **kwargs),
        overwrite=True,
    )
    register_policy_estimator("d3rlpy_bc", D3RLPYBCPolicyEstimator, overwrite=True)


_register_builtin_policies()
