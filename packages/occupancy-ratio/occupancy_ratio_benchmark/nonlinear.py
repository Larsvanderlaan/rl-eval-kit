from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from occupancy_ratio.fit_importance_and_transition_ratios import (
    importance_ratio_eval,
    importance_ratio_objective,
    make_importance_ratio_long_arrays_from_sa,
    make_lgb_importance_ratio_dataset,
)
from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray


@dataclass(frozen=True)
class NonlinearPolicy:
    gain: Array
    bias: float
    action_sd: float

    def mean_action(self, states: Array) -> Array:
        s = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        linear = s @ np.asarray(self.gain, dtype=np.float64).reshape(1, 2).T
        nonlinear = 0.25 * np.sin(s[:, [0]]) - 0.15 * np.cos(s[:, [1]])
        return linear + nonlinear + float(self.bias)

    def sample(self, states: Array, rng: np.random.Generator) -> Array:
        return self.mean_action(states) + float(self.action_sd) * rng.normal(size=(states.shape[0], 1))

    def logpdf(self, states: Array, actions: Array) -> Array:
        mean = self.mean_action(states).reshape(-1)
        action = np.asarray(actions, dtype=np.float64).reshape(-1)
        var = float(self.action_sd) ** 2
        return -0.5 * (np.log(2.0 * np.pi * var) + ((action - mean) ** 2) / var)


@dataclass(frozen=True)
class NonlinearSystem:
    behavior_policy: NonlinearPolicy
    target_policy: NonlinearPolicy
    process_noise_sd: float = 0.10
    initial_mean: Array = field(default_factory=lambda: np.array([0.6, -0.4], dtype=np.float64))
    initial_cov: Array = field(default_factory=lambda: np.diag([0.65, 0.45]).astype(np.float64))

    def step_mean(self, states: Array, actions: Array) -> Array:
        s = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
        mean = np.empty_like(s)
        mean[:, [0]] = 0.70 * s[:, [0]] + 0.18 * np.sin(s[:, [1]]) + 0.45 * a
        mean[:, [1]] = 0.62 * s[:, [1]] + 0.12 * np.cos(s[:, [0]]) - 0.20 * a
        return mean

    def step(self, states: Array, actions: Array, rng: np.random.Generator) -> Array:
        return self.step_mean(states, actions) + float(self.process_noise_sd) * rng.normal(size=(states.shape[0], 2))


def make_nonlinear_system() -> NonlinearSystem:
    return NonlinearSystem(
        behavior_policy=NonlinearPolicy(gain=np.array([[0.25, -0.30]], dtype=np.float64), bias=-0.15, action_sd=0.28),
        target_policy=NonlinearPolicy(gain=np.array([[-0.35, -0.05]], dtype=np.float64), bias=0.10, action_sd=0.22),
    )


def _sample_reference_joint(system: NonlinearSystem, n: int, rng: np.random.Generator) -> tuple[Array, Array]:
    states = rng.multivariate_normal(system.initial_mean, system.initial_cov, size=int(n))
    actions = system.behavior_policy.sample(states, rng)
    return states, actions


def _sample_target_discounted_joint(
    system: NonlinearSystem,
    *,
    gamma: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    states = rng.multivariate_normal(system.initial_mean, system.initial_cov, size=int(n))
    times = rng.geometric(p=max(1.0 - float(gamma), 1e-6), size=int(n)) - 1
    max_time = int(np.max(times)) if times.size else 0
    for t in range(max_time):
        active = times > t
        if not np.any(active):
            break
        actions_t = system.target_policy.sample(states[active], rng)
        states[active] = system.step(states[active], actions_t, rng)
    actions = system.target_policy.sample(states, rng)
    return states, actions


def _fit_joint_ratio_lgbm(
    *,
    reference_features: Array,
    target_features: Array,
    seed: int,
    num_boost_round: int,
) -> lgb.Booster:
    x_long, y_long, w_long = make_importance_ratio_long_arrays_from_sa(reference_features, target_features)
    dataset = make_lgb_importance_ratio_dataset(x_long, y_long, w_long)
    dataset.set_init_score(np.ones(x_long.shape[0], dtype=np.float64))

    def objective(preds: Array, data: lgb.Dataset) -> tuple[Array, Array]:
        return importance_ratio_objective(preds, data, eps=1e-3)

    params = {
        "objective": objective,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": int(seed),
        "num_threads": 0,
        "min_sum_hessian_in_leaf": 0,
        "lambda_l2": 0.0,
        "boost_from_average": False,
    }
    return lgb.train(
        params=params,
        train_set=dataset,
        valid_sets=[dataset],
        valid_names=["train"],
        feval=importance_ratio_eval,
        num_boost_round=int(num_boost_round),
        callbacks=[],
    )


def make_nonlinear_dataset(
    *,
    gamma: float,
    sample_size: int,
    seed: int,
    mc_truth_samples: int,
) -> BenchmarkDataset:
    system = make_nonlinear_system()
    rng = np.random.default_rng(seed)
    states, actions = _sample_reference_joint(system, int(sample_size), rng)
    next_states = system.step(states, actions, rng)
    target_actions = system.target_policy.sample(states, rng)
    next_target_actions = system.target_policy.sample(next_states, rng)
    initial_states = rng.multivariate_normal(system.initial_mean, system.initial_cov, size=max(256, min(2_000, sample_size)))
    initial_actions = system.target_policy.sample(initial_states, rng)

    truth_rng = np.random.default_rng(seed + 872_177)
    n_truth = int(max(mc_truth_samples, 2_000))
    reference_states, reference_actions = _sample_reference_joint(system, n_truth, truth_rng)
    target_states, target_actions_mc = _sample_target_discounted_joint(
        system,
        gamma=gamma,
        n=n_truth,
        rng=truth_rng,
    )
    reference_features = np.concatenate([reference_states, reference_actions], axis=1)
    target_features = np.concatenate([target_states, target_actions_mc], axis=1)
    truth_booster = _fit_joint_ratio_lgbm(
        reference_features=reference_features,
        target_features=target_features,
        seed=seed + 99,
        num_boost_round=80 if n_truth >= 20_000 else 35,
    )
    row_features = np.concatenate([states, actions], axis=1)
    true_ratio = np.maximum(1.0 + truth_booster.predict(row_features), 1e-8)
    true_action_ratio = np.exp(system.target_policy.logpdf(states, actions) - system.behavior_policy.logpdf(states, actions))
    rewards = -0.5 * np.sum(states**2, axis=1) - 0.15 * np.sum(actions**2, axis=1) + 0.25 * np.sin(states[:, 0])

    return BenchmarkDataset(
        setting="nonlinear_monte_carlo",
        states=states,
        actions=actions,
        next_states=next_states,
        target_actions=target_actions,
        next_target_actions=next_target_actions,
        rewards=rewards,
        true_ratio=true_ratio,
        true_action_ratio=true_action_ratio,
        true_transition_ratio=None,
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=np.ones(initial_states.shape[0], dtype=np.float64),
        masks=np.ones(int(sample_size), dtype=np.float64),
        gamma=float(gamma),
        seed=int(seed),
        sample_size=int(sample_size),
        metadata={
            "truth_source": "large_mc_lgbm_ratio",
            "reference_distribution": "initial_state_gaussian",
            "mc_truth_samples": n_truth,
            "state_dim": 2,
            "action_dim": 1,
        },
    )
