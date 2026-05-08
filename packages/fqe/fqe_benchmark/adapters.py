from __future__ import annotations

from importlib import import_module
import time
from typing import Any

import numpy as np

from fqe_benchmark.types import BenchmarkConfig, BenchmarkDataset, EstimatorPreflight, FittedEstimator
from fqe import (
    BoostedFQEConfig,
    NeuralFQEConfig,
    fit_fqe_lgbm,
    fit_fqe_neural,
    tune_fqe_cv,
    tune_fqe_neural_cv,
)


Array = np.ndarray


def estimator_registry() -> dict[str, object]:
    adapters = [
        OursBoostedFQEAdapter(),
        OursBoostedFQETunedAdapter(),
        OursNeuralFQEAdapter(),
        OursNeuralFQETunedAdapter(),
        LegacyBoostedFQEAdapter(),
        LegacyNeuralFQEAdapter(),
        ControlledLinearFQEAdapter(),
        D3RLPYFQEAdapter(),
        GooglePolicyEvalFQEAdapter(),
        DeepOPEReferenceFQEAdapter(),
    ]
    return {adapter.name: adapter for adapter in adapters}


class OursBoostedFQEAdapter:
    name = "ours_boosted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "array-based package FQE adapters are not enabled for Hopper placeholder rows; use hopper_fqe_benchmark.")
        try:
            import lightgbm  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "lightgbm is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        model = fit_fqe_lgbm(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            config=_boosted_config(config, seed=seed, tuned=False),
        )
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics=dict(model.diagnostics))


class OursBoostedFQETunedAdapter(OursBoostedFQEAdapter):
    name = "ours_boosted_fqe_tuned"

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        base = _boosted_config(config, seed=seed, tuned=True)
        tuned = tune_fqe_cv(
            param_grid=(
                {"loss": "squared", "lgb_params": {"num_leaves": 15}},
                {"loss": "huber", "huber_delta_scale": 1.345, "lgb_params": {"num_leaves": 31}},
                {"loss": "huber", "huber_delta_scale": 2.0, "lgb_params": {"num_leaves": 31}},
            ),
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            base_config=base,
            fit_final=True,
        )
        runtime = time.perf_counter() - start
        model = tuned["model"]
        diagnostics = dict(model.diagnostics)
        diagnostics.update({"best_params": tuned["best_params"], "best_score": tuned["best_score"]})
        return FittedEstimator(self.name, model, runtime, diagnostics=diagnostics, tuning_runtime_sec=runtime)


class OursNeuralFQEAdapter:
    name = "ours_neural_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "array-based package neural FQE adapters are not enabled for Hopper placeholder rows; use hopper_fqe_benchmark.")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "torch is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        model = fit_fqe_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            config=_neural_config(config, seed=seed, tuned=False),
        )
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics=dict(model.diagnostics))


class OursNeuralFQETunedAdapter(OursNeuralFQEAdapter):
    name = "ours_neural_fqe_tuned"

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        base = _neural_config(config, seed=seed, tuned=True)
        tuned = tune_fqe_neural_cv(
            param_grid=(
                {"loss": "squared", "hidden_dims": (32,) if config.stage == "smoke" else (128, 128)},
                {"loss": "huber", "hidden_dims": (32, 32) if config.stage == "smoke" else (256, 256)},
            ),
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            base_config=base,
            fit_final=True,
        )
        runtime = time.perf_counter() - start
        model = tuned["model"]
        diagnostics = dict(model.diagnostics)
        diagnostics.update({"best_params": tuned["best_params"], "best_score": tuned["best_score"]})
        return FittedEstimator(self.name, model, runtime, diagnostics=diagnostics, tuning_runtime_sec=runtime)


class LegacyBoostedFQEAdapter:
    name = "legacy_boosted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "legacy boosted FQE adapter is not enabled for Hopper placeholder rows.")
        try:
            import_module("FQE.fqe_boosted")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"legacy boosted import failed: {type(exc).__name__}: {exc}")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        legacy = import_module("FQE.fqe_boosted")
        start = time.perf_counter()
        result = legacy.fit_fqe_boosted(
            S=dataset.states,
            A=dataset.actions,
            S_next=dataset.next_states,
            A_next=_flatten_next_actions(dataset.next_actions),
            rewards=dataset.rewards,
            discount_factor=dataset.gamma,
            is_terminal_outcome=dataset.terminals,
            weights=dataset.sample_weight,
            lgb_params={"num_iterations": 1, "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5, "verbosity": -1},
            fit_control={"num_boost_rounds": max(4, min(config.boosted_num_iterations, 30)), "early_stopping_rounds": 5},
            seed=seed,
            verbose=False,
        )
        model = _LegacyBoostedModel(result["model"], dataset.state_dim, dataset.action_dim)
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics={"legacy_keys": sorted(result.keys())})


class LegacyNeuralFQEAdapter:
    name = "legacy_neural_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "legacy neural FQE adapter is not enabled for Hopper placeholder rows.")
        try:
            import_module("FQE.fqe_neural")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"legacy neural import failed: {type(exc).__name__}: {exc}")
        return EstimatorPreflight("unsupported_setting", "legacy neural FQE implementation in FQE/fqe_neural.py is value-only and does not expose Q(s,a) prediction.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("legacy_neural_fqe is unsupported for Q-mode benchmark datasets.")


class ControlledLinearFQEAdapter:
    name = "controlled_linear_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "controlled linear FQE is not enabled for Hopper placeholder rows.")
        if dataset is not None and dataset.domain not in {"controlled_synthetic", "tabular"}:
            return EstimatorPreflight("unsupported_setting", "controlled linear FQE is only enabled for truth-known datasets.")
        try:
            import_module("FQE_neurips.controlled_discounted_benchmark.fqe")
        except Exception as exc:
            return EstimatorPreflight(
                "ok",
                f"controlled benchmark import unavailable; using generic linear FQE fallback ({type(exc).__name__}: {exc}).",
            )
        return EstimatorPreflight("ok", "controlled benchmark import available; using generic linear adapter with matching feature-style baseline.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        model = _fit_linear_fqe(dataset, ridge=1e-4, n_iters=80 if config.stage == "smoke" else 200)
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics={"ridge": 1e-4})


class D3RLPYFQEAdapter:
    name = "d3rlpy_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "d3rlpy Hopper adapter is preflighted but not wired in v1.")
        try:
            import d3rlpy  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "d3rlpy is not installed.")
        return EstimatorPreflight("unsupported_setting", "d3rlpy adapter is preflighted but not enabled for synthetic array datasets in v1.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("d3rlpy adapter is not enabled for this v1 benchmark.")


class GooglePolicyEvalFQEAdapter:
    name = "google_policy_eval_fqe_l2"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        policy_eval = config.google_research_path / "policy_eval"
        if not policy_eval.exists():
            return EstimatorPreflight("missing_dependency", f"missing Google policy_eval at {policy_eval}.")
        if dataset is not None and dataset.domain == "offline_hopper":
            benchmark_dir = config.hopper_artifact_dir / "benchmark" / "dope"
            if not benchmark_dir.exists():
                return EstimatorPreflight("missing_data", f"missing Deep OPE benchmark artifacts at {benchmark_dir}.")
            return EstimatorPreflight("unsupported_setting", "Google policy_eval Hopper execution should be delegated to hopper_fqe_benchmark in v1.")
        return EstimatorPreflight("unsupported_setting", "Google policy_eval FQE-L2 is only enabled for Hopper/Deep OPE data.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("Google policy_eval FQE-L2 is not enabled for this dataset.")


class DeepOPEReferenceFQEAdapter:
    name = "deep_ope_reference_fqe_l2"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        reference_path = config.hopper_artifact_dir / "benchmark" / "dope" / "d4rl_fqel2.pkl"
        if not reference_path.exists():
            return EstimatorPreflight("missing_data", f"missing Deep OPE reference file {reference_path}.")
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "Deep OPE reference file is present; reference-row materialization is delegated to hopper_fqe_benchmark in v1.")
        return EstimatorPreflight("unsupported_setting", "Deep OPE reference rows are only emitted for Hopper policy-ranking benchmarks.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("Deep OPE reference rows are not fitted estimators.")


class _LegacyBoostedModel:
    def __init__(self, booster: Any, state_dim: int, action_dim: int) -> None:
        self.booster = booster
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    def predict_q(self, states: Array, actions: Array) -> Array:
        features = np.concatenate(
            [
                np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim),
                np.asarray(actions, dtype=np.float64).reshape(-1, self.action_dim),
            ],
            axis=1,
        )
        return np.asarray(self.booster.predict(features), dtype=np.float64).reshape(-1)

    def estimate_policy_value(self, initial_states: Array, initial_actions: Array) -> float:
        return float(np.mean(self.predict_q(initial_states, initial_actions)))


class _LinearFQEModel:
    def __init__(self, theta: Array, state_dim: int, action_dim: int) -> None:
        self.theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    def predict_q(self, states: Array, actions: Array) -> Array:
        return _linear_features(states, actions) @ self.theta

    def estimate_policy_value(self, initial_states: Array, initial_actions: Array) -> float:
        return float(np.mean(self.predict_q(initial_states, initial_actions)))


def _boosted_config(config: BenchmarkConfig, *, seed: int, tuned: bool) -> BoostedFQEConfig:
    iterations = config.boosted_tune_num_iterations if tuned else config.boosted_num_iterations
    return BoostedFQEConfig.stable_defaults(
        num_iterations=int(iterations),
        validation_fraction=0.25,
        patience=5 if config.stage == "smoke" else 12,
        seed=int(seed),
        infer_value_bounds=True,
        show_progress=False,
        lgb_params={
            "learning_rate": 0.08,
            "num_leaves": 15 if config.stage == "smoke" else 31,
            "min_data_in_leaf": 5 if config.stage == "smoke" else 30,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 1,
        },
    )


def _neural_config(config: BenchmarkConfig, *, seed: int, tuned: bool) -> NeuralFQEConfig:
    iterations = config.neural_tune_num_iterations if tuned else config.neural_num_iterations
    steps = config.neural_tune_gradient_steps_per_iteration if tuned else config.neural_gradient_steps_per_iteration
    return NeuralFQEConfig.stable_defaults(
        hidden_dims=(32, 32) if config.stage == "smoke" else (256, 256),
        learning_rate=2e-3 if config.stage == "smoke" else 3e-4,
        weight_decay=1e-4,
        batch_size=64 if config.stage == "smoke" else 512,
        num_iterations=int(iterations),
        gradient_steps_per_iteration=int(steps),
        target_update_tau=0.5 if config.stage == "smoke" else 0.05,
        validation_fraction=0.25,
        patience=4 if config.stage == "smoke" else 8,
        seed=int(seed),
        device="cpu",
        infer_value_bounds=True,
        show_progress=False,
    )


def _flatten_next_actions(next_actions: Array) -> Array:
    arr = np.asarray(next_actions, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, 0, :]
    return arr


def _linear_features(states: Array, actions: Array) -> Array:
    s = np.asarray(states, dtype=np.float64).reshape(states.shape[0], -1)
    a = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
    x = np.concatenate([s, a], axis=1)
    quad = [x[:, i : i + 1] * x[:, j : j + 1] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    return np.concatenate([np.ones((x.shape[0], 1)), x, *quad], axis=1)


def _fit_linear_fqe(dataset: BenchmarkDataset, *, ridge: float, n_iters: int) -> _LinearFQEModel:
    phi = _linear_features(dataset.states, dataset.actions)
    phi_next = _linear_features(dataset.next_states, _flatten_next_actions(dataset.next_actions))
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    weights = np.ones_like(rewards) if dataset.sample_weight is None else np.asarray(dataset.sample_weight, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-12)
    gram = phi.T @ (weights[:, None] * phi) + float(ridge) * np.eye(phi.shape[1])
    reward_rhs = phi.T @ (weights * rewards)
    next_mat = phi.T @ (weights[:, None] * phi_next)
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    for _ in range(int(n_iters)):
        rhs = reward_rhs + dataset.gamma * (next_mat @ theta)
        theta_new = np.linalg.solve(gram, rhs)
        if np.linalg.norm(theta_new - theta) <= 1e-10 * max(1.0, np.linalg.norm(theta)):
            theta = theta_new
            break
        theta = theta_new
    return _LinearFQEModel(theta=theta, state_dim=dataset.state_dim, action_dim=dataset.action_dim)
