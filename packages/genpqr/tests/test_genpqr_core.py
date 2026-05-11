from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from genpqr import (
    ActionSpaceSpec,
    BehaviorCloningPolicyEstimator,
    ContinuousNormalizationPolicy,
    EpisodeDataset,
    DeepPQRAnchorQEstimator,
    DiscreteNormalizationPolicy,
    GenPQRDiagnostics,
    GenPQRAdapterError,
    FQEQEstimator,
    GenPQRConfig,
    GenPQRConfigurationError,
    NeuralDeepPQRAnchorQEstimator,
    ReusableScopeRLQEstimator,
    ScopeRLQEstimator,
    available_policy_estimators,
    available_q_estimators,
    fit_genpqr,
    fit_genpqr_crossfit,
    load_genpqr_result,
    register_q_estimator,
)
from genpqr import testing as genpqr_testing
from genpqr.benchmarks import make_tabular_chain
from genpqr.q_estimators import WrappedFittedQFunction


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_import_is_lazy_for_heavy_backends() -> None:
    code = (
        "import sys; import genpqr; "
        "mods=('torch','imitation','d3rlpy','scope_rl','stable_baselines3','gymnasium'); "
        "print({m: (m in sys.modules) for m in mods})"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{os.getcwd()}/packages/genpqr:{os.getcwd()}/packages/fqe"
    out = subprocess.check_output([sys.executable, "-c", code], cwd=os.getcwd(), env=env, text=True)
    assert "'torch': False" in out
    assert "'imitation': False" in out
    assert "'d3rlpy': False" in out
    assert "'scope_rl': False" in out


def test_action_space_validation_and_discrete_normalization() -> None:
    spec = ActionSpaceSpec.discrete(3)
    actions = np.array([0, 2, 1])
    encoded = spec.action_matrix(actions)
    assert encoded.shape == (3, 3)
    assert np.allclose(encoded.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="outside"):
        spec.validate_actions(np.array([0, 3]))

    mu = DiscreteNormalizationPolicy.anchor(3, 1)
    states = np.zeros((4, 2))
    assert np.all(mu.sample(states, np.random.default_rng(0), 1) == 1)
    assert np.allclose(mu.predict_proba(states)[:, 1], 1.0)
    with pytest.raises(ValueError, match="0/1"):
        spec.validate_actions(np.array([[0.5, 0.5, 0.0]]), n_rows=1)
    assert spec.encode_samples(np.eye(3, dtype=int), n_rows=3).shape == (3, 3)
    multi = spec.encode_samples(np.array([[0, 1], [1, 2], [2, 0]]), n_rows=3)
    assert multi.shape == (3, 2, 3)


def test_transition_and_episode_datasets_preserve_boundaries() -> None:
    first = {
        "states": np.array([[0.0], [1.0]]),
        "actions": np.array([0, 1]),
        "next_states": np.array([[1.0], [2.0]]),
        "terminals": np.array([0.0, 1.0]),
    }
    second = {
        "states": np.array([[2.0], [3.0]]),
        "actions": np.array([1, 0]),
        "next_states": np.array([[3.0], [2.0]]),
        "terminals": np.array([0.0, 1.0]),
    }
    episodes = EpisodeDataset.from_episodes([first, second], action_space=ActionSpaceSpec.discrete(2))
    flat = episodes.to_transition_dataset()
    assert flat.n_rows == 4
    assert np.array_equal(flat.episode_ids, np.array([0, 0, 1, 1]))
    folds = flat.make_folds(n_folds=2, seed=0, episode_respecting=True)
    for _, test_idx in folds:
        assert np.unique(flat.episode_ids[test_idx]).shape[0] == 1
    assert flat.to_d3rlpy_kwargs()["observations"].shape == (4, 1)
    assert flat.to_scope_rl_logged_dataset()["state"].shape == (2, 2, 1)


@dataclass
class MatrixQ:
    q_matrix: np.ndarray
    action_space: ActionSpaceSpec

    def predict_q(self, states, actions):
        idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
        rows = np.arange(np.asarray(states).shape[0]) % self.q_matrix.shape[0]
        return self.q_matrix[rows, idx]

    def predict_q_matrix(self, states):
        rows = np.arange(np.asarray(states).shape[0]) % self.q_matrix.shape[0]
        return self.q_matrix[rows]

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        del n_action_samples, rng
        return np.sum(normalization_policy.predict_proba(states) * self.predict_q_matrix(states), axis=1)


def test_reward_recovery_formula_discrete() -> None:
    from genpqr.recovery import GenPQRRewardFunction

    spec = ActionSpaceSpec.discrete(2)
    q = MatrixQ(np.array([[1.0, 3.0], [2.0, -2.0]]), spec)
    mu = DiscreteNormalizationPolicy.uniform(2)
    reward = GenPQRRewardFunction(q, mu, anchor_function=0.5)
    states = np.zeros((2, 1))
    matrix = reward.predict_reward_matrix(states)
    assert np.allclose(matrix[0], [-0.5, 1.5])
    assert np.allclose(matrix[1], [2.5, -1.5])
    assert np.allclose(reward.normalization_residual(states), 0.0)


def test_reward_recovery_rejects_bad_q_vector_shapes() -> None:
    from genpqr.recovery import GenPQRRewardFunction

    class BadShapeQ:
        action_space = ActionSpaceSpec.discrete(2)

        def predict_q(self, states, actions):
            del actions
            return np.ones((1, np.asarray(states).shape[0]))

        def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
            del normalization_policy, n_action_samples, rng
            return np.zeros(np.asarray(states).shape[0])

    reward = GenPQRRewardFunction(BadShapeQ(), DiscreteNormalizationPolicy.uniform(2))
    with pytest.raises(ValueError, match="shape"):
        reward.predict_reward(np.zeros((3, 1)), np.array([0, 1, 0]))


def test_dataset_fit_diagnostics_and_serialization(tmp_path: Path) -> None:
    dataset = make_tabular_chain(24, seed=13)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.0,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear", policy_config={"n_epochs": 20}),
    )
    assert isinstance(result.diagnostics_report, GenPQRDiagnostics)
    assert result.diagnostics["reward_finite_fraction"] == 1.0
    result.save(str(tmp_path / "saved"))
    loaded = load_genpqr_result(tmp_path / "saved")
    assert np.allclose(
        loaded.predict_reward(dataset.states[:3], dataset.actions[:3]),
        result.predict_reward(dataset.states[:3], dataset.actions[:3]),
    )


def test_default_airl_requires_environment() -> None:
    states = np.zeros((4, 1))
    actions = np.array([0, 1, 0, 1])
    with pytest.raises(GenPQRConfigurationError, match="requires env"):
        fit_genpqr(states=states, actions=actions, next_states=states, terminals=np.zeros(4), gamma=0.9)


def test_native_behavior_cloning_discrete_log_prob_and_sampling() -> None:
    states = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    actions = np.array([0, 0, 1, 1])
    spec = ActionSpaceSpec.discrete(2)
    policy = BehaviorCloningPolicyEstimator(n_epochs=80, learning_rate=0.1).fit(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.zeros(4),
        action_space=spec,
    )
    probs = policy.predict_proba(states)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(np.isfinite(policy.log_prob(states, actions)))
    draws = policy.sample(states, np.random.default_rng(3), n_samples=3)
    assert draws.shape == (4, 3)
    genpqr_testing.check_estimated_policy_contract(policy, states=states, actions=actions, action_space=spec)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_native_bc_plus_neural_fqe_smoke() -> None:
    from fqe import NeuralFQEConfig

    rng = np.random.default_rng(10)
    states = rng.normal(size=(30, 2))
    actions = (states[:, 0] > 0.0).astype(int)
    next_states = rng.normal(size=(30, 2))
    terminals = np.zeros(30)
    config = GenPQRConfig(
        policy="behavior_cloning_native",
        q=FQEQEstimator(
            family="neural",
            n_next_action_samples=2,
            config=NeuralFQEConfig.stable_defaults(
                hidden_dims=(16,),
                activation="relu",
                num_iterations=3,
                gradient_steps_per_iteration=3,
                batch_size=16,
                early_stopping=False,
                infer_value_bounds=False,
                seed=5,
            ),
        ),
        n_action_samples=2,
        policy_config={"n_epochs": 40, "learning_rate": 0.1},
    )
    result = fit_genpqr(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=config,
    )
    pred = result.predict_reward(states[:5], actions[:5])
    assert pred.shape == (5,)
    assert np.all(np.isfinite(pred))


def test_deeppqr_anchor_q_reconstructs_stratified_log_ratio() -> None:
    class FixedPolicy:
        action_space = ActionSpaceSpec.discrete(3)

        def predict_proba(self, states):
            base = np.array([0.2, 0.5, 0.3])
            return np.tile(base, (np.asarray(states).shape[0], 1))

        def log_prob(self, states, actions):
            idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
            return np.log(self.predict_proba(states)[np.arange(np.asarray(states).shape[0]), idx])

        def sample(self, states, rng, n_samples=1):
            return np.zeros((np.asarray(states).shape[0], n_samples), dtype=int)

    states = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    actions = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1])
    policy = FixedPolicy()
    result = fit_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(12),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(3),
        normalization_policy=DiscreteNormalizationPolicy.anchor(3, 0),
        config=GenPQRConfig(policy=policy, q=DeepPQRAnchorQEstimator(anchor_action=0, n_iterations=5)),
    )
    q_matrix = result.q_function.predict_q_matrix(states[:3])
    expected_shift = np.log(np.array([0.2, 0.5, 0.3])) - np.log(0.2)
    assert np.allclose(q_matrix - q_matrix[:, [0]], expected_shift.reshape(1, -1))
    assert result.diagnostics["q_anchor_count"] == 5
    genpqr_testing.check_fitted_q_contract(
        result.q_function,
        states=states[:4],
        actions=actions[:4],
        action_space=ActionSpaceSpec.discrete(3),
        normalization_policy=DiscreteNormalizationPolicy.anchor(3, 0),
    )


def test_deeppqr_uses_general_normalization_policy_in_anchor_bellman_target() -> None:
    class FixedPolicy:
        action_space = ActionSpaceSpec.discrete(2)

        def predict_proba(self, states):
            base = np.array([0.25, 0.75])
            return np.tile(base, (np.asarray(states).shape[0], 1))

        def log_prob(self, states, actions):
            idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
            return np.log(self.predict_proba(states)[np.arange(np.asarray(states).shape[0]), idx])

        def sample(self, states, rng, n_samples=1):
            return np.zeros((np.asarray(states).shape[0], n_samples), dtype=int)

    states = np.zeros((10, 1))
    actions = np.zeros(10, dtype=int)
    policy = FixedPolicy()
    gamma = 0.5
    result = fit_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.zeros(10),
        gamma=gamma,
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(
            policy=policy,
            q=DeepPQRAnchorQEstimator(anchor_action=0, ridge=1e-10, n_iterations=80),
        ),
    )
    log_probs = np.log(np.array([0.25, 0.75]))
    normalization_shift = np.mean(log_probs) - log_probs[0]
    expected_anchor_value = (log_probs[0] + gamma * normalization_shift) / (1.0 - gamma)
    assert np.allclose(result.q_function.predict_anchor_value(states[:3]), expected_anchor_value, atol=1e-5)
    q_matrix = result.q_function.predict_q_matrix(states[:3])
    assert np.allclose(q_matrix - q_matrix[:, [0]], (log_probs - log_probs[0]).reshape(1, -1))


def test_deeppqr_rejects_zero_positive_anchor_weight() -> None:
    class FixedPolicy:
        action_space = ActionSpaceSpec.discrete(2)

        def predict_proba(self, states):
            return np.full((np.asarray(states).shape[0], 2), 0.5)

        def log_prob(self, states, actions):
            del actions
            return np.full(np.asarray(states).shape[0], np.log(0.5))

        def sample(self, states, rng, n_samples=1):
            return np.zeros((np.asarray(states).shape[0], n_samples), dtype=int)

    estimator = DeepPQRAnchorQEstimator(anchor_action=0)
    with pytest.raises(GenPQRConfigurationError, match="positive-weight anchor"):
        estimator.fit(
            states=np.zeros((3, 1)),
            actions=np.array([0, 0, 1]),
            next_states=np.zeros((3, 1)),
            pseudo_rewards=np.zeros(3),
            normalization_policy=DiscreteNormalizationPolicy.uniform(2),
            gamma=0.9,
            sample_weight=np.array([0.0, 0.0, 1.0]),
            policy=FixedPolicy(),
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_neural_deeppqr_smoke_reconstructs_log_ratio() -> None:
    class FixedPolicy:
        action_space = ActionSpaceSpec.discrete(2)

        def predict_proba(self, states):
            base = np.array([0.3, 0.7])
            return np.tile(base, (np.asarray(states).shape[0], 1))

        def log_prob(self, states, actions):
            idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
            return np.log(self.predict_proba(states)[np.arange(np.asarray(states).shape[0]), idx])

        def sample(self, states, rng, n_samples=1):
            return np.zeros((np.asarray(states).shape[0], n_samples), dtype=int)

    states = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    actions = np.zeros(12, dtype=int)
    result = fit_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(12),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(
            policy=FixedPolicy(),
            q=NeuralDeepPQRAnchorQEstimator(
                anchor_action=0,
                hidden_dims=(8,),
                max_epochs=5,
                batch_size=6,
                patience=2,
                validation_fraction=0.2,
                seed=7,
            ),
        ),
    )
    q_matrix = result.q_function.predict_q_matrix(states[:3])
    expected_shift = np.log(np.array([0.3, 0.7])) - np.log(0.3)
    assert np.allclose(q_matrix - q_matrix[:, [0]], expected_shift.reshape(1, -1), atol=1e-6)
    assert result.diagnostics["q_backend"] == "neural_deep_pqr_anchor"


def test_custom_normalization_policy_requires_action_space() -> None:
    class BadNormalizationPolicy:
        def sample(self, states, rng, n_samples=1):
            return np.zeros((np.asarray(states).shape[0], n_samples), dtype=int)

    states = np.zeros((3, 1))
    actions = np.array([0, 1, 0])
    with pytest.raises(GenPQRConfigurationError, match="action_space"):
        fit_genpqr(
            states=states,
            actions=actions,
            next_states=states,
            terminals=np.zeros(3),
            gamma=0.0,
            action_space=ActionSpaceSpec.discrete(2),
            normalization_policy=BadNormalizationPolicy(),
            config=GenPQRConfig(policy="behavior_cloning_native", q=DeepPQRAnchorQEstimator(anchor_action=0)),
        )


def test_continuous_custom_policy_and_q_with_sampled_normalization() -> None:
    class GaussianPolicy:
        action_space = ActionSpaceSpec.continuous(1)

        def log_prob(self, states, actions):
            actions = self.action_space.action_matrix(actions, n_rows=np.asarray(states).shape[0])
            mean = np.asarray(states)[:, [0]]
            return -0.5 * ((actions - mean) ** 2).reshape(-1)

        def sample(self, states, rng, n_samples=1):
            mean = np.asarray(states)[:, [0]]
            draws = rng.normal(loc=mean[:, None, :], scale=1.0, size=(mean.shape[0], n_samples, 1))
            return draws[:, 0, :] if n_samples == 1 else draws

    class LinearQEstimator:
        def fit(self, **kwargs):
            del kwargs
            return self

        action_space = ActionSpaceSpec.continuous(1)

        def predict_q(self, states, actions):
            actions = self.action_space.action_matrix(actions, n_rows=np.asarray(states).shape[0])
            return np.asarray(states)[:, 0] + actions[:, 0]

        def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
            samples = normalization_policy.sample(states, rng, n_action_samples)
            return np.mean([self.predict_q(states, samples[:, j, :]) for j in range(samples.shape[1])], axis=0)

    mu = ContinuousNormalizationPolicy(
        action_dim=1,
        sampler=lambda states, rng, n: np.zeros((states.shape[0], n, 1)),
    )
    states = np.array([[0.0], [1.0], [2.0]])
    actions = np.array([[0.5], [1.5], [2.5]])
    result = fit_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.zeros(3),
        gamma=0.0,
        action_space=ActionSpaceSpec.continuous(1),
        normalization_policy=mu,
        config=GenPQRConfig(policy=GaussianPolicy(), q=LinearQEstimator(), n_action_samples=4),
    )
    rewards = result.predict_reward(states, actions)
    assert np.allclose(rewards, actions.reshape(-1))
    assert result.diagnostics["normalization_residual_abs_mean"] is None
    assert np.isfinite(result.diagnostics["continuous_mc_standard_error_mean"])


def test_crossfit_covers_each_row_once_and_refits_final() -> None:
    dataset = make_tabular_chain(30, seed=9)
    result = fit_genpqr_crossfit(
        dataset=dataset,
        gamma=0.0,
        n_folds=3,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear", policy_config={"n_epochs": 10}),
    )
    held_out = np.concatenate(result.fold_indices)
    assert np.array_equal(np.sort(held_out), np.arange(dataset.n_rows))
    assert result.final_result is not None
    assert result.out_of_fold_rewards.shape == (dataset.n_rows,)


def test_crossfit_rejects_row_bound_normalization_policy() -> None:
    dataset = make_tabular_chain(12, seed=11)
    probs = np.full((dataset.n_rows, 2), 0.5)
    with pytest.raises(GenPQRConfigurationError, match="row-bound"):
        fit_genpqr_crossfit(
            dataset=dataset,
            gamma=0.0,
            n_folds=3,
            normalization_policy=DiscreteNormalizationPolicy(2, probs),
            config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear", policy_config={"n_epochs": 5}),
        )


def test_registry_and_presets_resolve() -> None:
    class RegisteredQEstimator:
        def fit(self, **kwargs):
            del kwargs
            return MatrixQ(np.tile(np.array([[0.0, 1.0]]), (3, 1)), ActionSpaceSpec.discrete(2))

    register_q_estimator("unit_test_registered_q", RegisteredQEstimator, overwrite=True)
    assert "unit_test_registered_q" in available_q_estimators()
    assert "behavior_cloning_native" in available_policy_estimators()
    for name in (
        "airl_fast",
        "airl_balanced",
        "airl_paper",
        "gail_fast",
        "gail_balanced",
        "bc_boosted_fast",
        "bc_neural_balanced",
        "deeppqr_linear",
        "deeppqr_neural",
    ):
        assert isinstance(GenPQRConfig.from_preset(name), GenPQRConfig)


def test_contract_checkers_validate_policy_and_q_estimators() -> None:
    states = np.array([[-1.0], [0.0], [1.0], [2.0]])
    actions = np.array([0, 0, 1, 1])
    spec = ActionSpaceSpec.discrete(2)
    estimator = BehaviorCloningPolicyEstimator(n_epochs=10)
    genpqr_testing.check_policy_estimator_contract(
        estimator,
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.zeros(4),
        action_space=spec,
    )

    class MatrixQEstimator:
        def fit(self, **kwargs):
            del kwargs
            return MatrixQ(np.tile(np.array([[1.0, 2.0]]), (4, 1)), spec)

    policy = estimator.fit(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.zeros(4),
        action_space=spec,
    )
    genpqr_testing.check_q_estimator_contract(
        MatrixQEstimator(),
        states=states,
        actions=actions,
        next_states=states,
        pseudo_rewards=np.zeros(4),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        gamma=0.0,
        terminals=np.zeros(4),
        policy=policy,
    )


def test_sampled_discrete_expectation_handles_single_sample() -> None:
    class ConstantSampler:
        action_space = ActionSpaceSpec.discrete(2)

        def sample(self, states, rng, n_samples=1):
            assert n_samples == 1
            return np.ones(np.asarray(states).shape[0], dtype=int)

    class QModel:
        def predict_q(self, states, actions):
            del states
            return np.asarray(actions)[:, 1] if np.asarray(actions).ndim == 2 else np.asarray(actions, dtype=float)

    wrapped = WrappedFittedQFunction(QModel(), ActionSpaceSpec.discrete(2), backend="test")
    values = wrapped.expected_q(np.zeros((3, 1)), ConstantSampler(), n_action_samples=1, rng=np.random.default_rng(0))
    assert np.allclose(values, 1.0)


def test_d3rlpy_bc_seed_is_not_injected_before_optional_import() -> None:
    states = np.zeros((2, 1))
    actions = np.array([0, 1])
    with pytest.raises(Exception) as excinfo:
        fit_genpqr(
            states=states,
            actions=actions,
            next_states=states,
            terminals=np.zeros(2),
            gamma=0.0,
            action_space=ActionSpaceSpec.discrete(2),
            normalization_policy=DiscreteNormalizationPolicy.uniform(2),
            config=GenPQRConfig(policy="d3rlpy_bc", q=DeepPQRAnchorQEstimator(anchor_action=0)),
        )
    assert "seed" not in str(excinfo.value)


def test_scope_rl_adapter_preflight_or_mapping_error() -> None:
    estimator = ScopeRLQEstimator(method="mql")
    with pytest.raises(GenPQRConfigurationError, match="env and evaluation_policies"):
        estimator.fit(
            states=np.zeros((2, 1)),
            actions=np.array([0, 1]),
            next_states=np.zeros((2, 1)),
            pseudo_rewards=np.zeros(2),
            normalization_policy=DiscreteNormalizationPolicy.uniform(2),
            gamma=0.9,
        )


def test_scope_rl_dataset_bound_guard_runs_before_optional_import() -> None:
    estimator = ScopeRLQEstimator(method="mql", env=object(), evaluation_policies=object())
    with pytest.raises(GenPQRAdapterError, match="dataset-bound"):
        estimator.fit(
            states=np.zeros((2, 1)),
            actions=np.array([0, 1]),
            next_states=np.zeros((2, 1)),
            pseudo_rewards=np.zeros(2),
            normalization_policy=DiscreteNormalizationPolicy.uniform(2),
            gamma=0.9,
        )


def test_reusable_scope_rl_q_estimator_wraps_user_model() -> None:
    class ReusableModel:
        def fit(self, states, actions, next_states, rewards, gamma, terminals, sample_weight=None):
            del states, actions, next_states, gamma, terminals, sample_weight
            self.mean_reward = float(np.mean(rewards))

        def predict_q(self, states, actions):
            del actions
            return np.full(np.asarray(states).shape[0], self.mean_reward)

    estimator = ReusableScopeRLQEstimator(ReusableModel)
    q = estimator.fit(
        states=np.zeros((3, 1)),
        actions=np.array([0, 1, 0]),
        next_states=np.zeros((3, 1)),
        pseudo_rewards=np.array([1.0, 2.0, 3.0]),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        gamma=0.0,
    )
    assert np.allclose(q.predict_q(np.zeros((2, 1)), np.array([0, 1])), 2.0)
