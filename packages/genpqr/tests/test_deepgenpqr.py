from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from genpqr import (
    ActionSpaceSpec,
    ContinuousNormalizationPolicy,
    ConstantFittedQFunction,
    DeepGenPQRConfig,
    DiscreteNormalizationPolicy,
    GenPQRConfigurationError,
    TransitionDataset,
    fit_deep_genpqr,
    list_deepgenpqr_presets,
    load_deep_genpqr_result,
    save_deep_genpqr_result,
)
from genpqr.neural_deeppqr import _resolve_anchor_rows


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FixedDiscretePolicy:
    action_space = ActionSpaceSpec.discrete(2)

    def predict_proba(self, states):
        return np.tile(np.array([0.7, 0.3]), (np.asarray(states).shape[0], 1))

    def log_prob(self, states, actions):
        probs = np.tile(np.array([0.7, 0.3]), (np.asarray(states).shape[0], 1))
        idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
        return np.log(probs[np.arange(idx.shape[0]), idx])

    def sample(self, states, rng, n_samples=1):
        del rng
        draws = np.zeros((np.asarray(states).shape[0], int(n_samples)), dtype=np.int64)
        return draws.reshape(-1) if int(n_samples) == 1 else draws


class ConstantQEstimator:
    def fit(self, *, normalization_policy, **kwargs):
        del kwargs
        return ConstantFittedQFunction(
            action_space=normalization_policy.action_space,
            value=1.0,
            backend="unit_neural_fqe",
        )


class ConstantQFunction:
    diagnostics = {"backend": "unit_neural_fqe"}

    def __init__(self, action_space):
        self.action_space = action_space

    def predict_q(self, states, actions):
        self.action_space.validate_actions(actions, n_rows=np.asarray(states).shape[0])
        return np.ones(np.asarray(states).shape[0])

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        del normalization_policy, n_action_samples, rng
        return np.ones(np.asarray(states).shape[0])


class NonPortableQEstimator:
    def fit(self, *, normalization_policy, **kwargs):
        del kwargs
        return ConstantQFunction(normalization_policy.action_space)


class GaussianPolicy:
    action_space = ActionSpaceSpec.continuous(1)

    def log_prob(self, states, actions):
        del states
        action_matrix = self.action_space.action_matrix(actions)
        return -0.5 * np.sum(action_matrix**2, axis=1)

    def sample(self, states, rng, n_samples=1):
        states = np.asarray(states)
        return rng.normal(size=(states.shape[0], int(n_samples), 1))


class ContinuousQEstimator:
    def fit(self, *, normalization_policy, **kwargs):
        del kwargs
        return ContinuousQFunction(normalization_policy.action_space)


class ContinuousQFunction:
    diagnostics = {"backend": "unit_continuous_neural_fqe"}

    def __init__(self, action_space):
        self.action_space = action_space

    def predict_q(self, states, actions):
        states = np.asarray(states)
        actions = self.action_space.action_matrix(actions, n_rows=states.shape[0])
        return states[:, 0] + actions[:, 0]

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        samples = normalization_policy.sample(states, rng, n_action_samples)
        values = [self.predict_q(states, samples[:, j, :]) for j in range(samples.shape[1])]
        return np.mean(np.stack(values, axis=1), axis=1)


def test_deepgenpqr_presets_are_public() -> None:
    presets = list_deepgenpqr_presets()
    assert "deepgenpqr_airl_fqe_balanced" in presets
    assert "deepgenpqr_airl_anchor_fast" in presets
    assert isinstance(DeepGenPQRConfig.from_preset("deepgenpqr_bc_fqe_debug"), DeepGenPQRConfig)


def test_deepgenpqr_default_airl_requires_env_before_q_imports() -> None:
    states = np.zeros((4, 1))
    actions = np.array([0, 1, 0, 1])
    with pytest.raises(GenPQRConfigurationError, match="requires env"):
        fit_deep_genpqr(
            states=states,
            actions=actions,
            next_states=states,
            terminals=np.ones(4),
            gamma=0.0,
            action_space=ActionSpaceSpec.discrete(2),
        )


def test_deepgenpqr_pooled_mode_with_arrays_and_dataset() -> None:
    states = np.arange(4.0).reshape(-1, 1)
    actions = np.array([0, 1, 0, 1])
    config = DeepGenPQRConfig(policy=FixedDiscretePolicy(), q_backend=ConstantQEstimator())
    array_result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(4),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        config=config,
    )
    dataset = TransitionDataset.from_arrays(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(4),
        action_space=ActionSpaceSpec.discrete(2),
    )
    dataset_result = fit_deep_genpqr(dataset=dataset, gamma=0.0, config=config)
    assert np.all(np.isfinite(array_result.predict_reward(states, actions)))
    assert np.all(np.isfinite(dataset_result.predict_reward(states, actions)))
    assert array_result.diagnostics["deepgenpqr_mode"] == "pooled_fqe"
    assert array_result.diagnostics["pooled_actions"] is True
    assert array_result.summary()["q_backend"] == "unit_neural_fqe"


def test_deepgenpqr_normalization_config_builds_policy() -> None:
    states = np.arange(4.0).reshape(-1, 1)
    actions = np.array([0, 1, 0, 1])
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(4),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        config=DeepGenPQRConfig(
            policy=FixedDiscretePolicy(),
            q_backend=ConstantQEstimator(),
            normalization_config={"kind": "anchor", "anchor_action": 0},
        ),
    )
    assert result.normalization_policy.predict_proba(states)[:, 0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_deepgenpqr_continuous_mc_diagnostics_are_seeded() -> None:
    states = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    actions = np.zeros((8, 1))

    def sampler(states_arg, rng, n_samples):
        return rng.normal(loc=0.0, scale=0.1, size=(states_arg.shape[0], int(n_samples), 1))

    mu = ContinuousNormalizationPolicy(action_dim=1, sampler=sampler)
    config = DeepGenPQRConfig(
        policy=GaussianPolicy(),
        q_backend=ContinuousQEstimator(),
        seed=19,
        n_action_samples=5,
    )
    first = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(8),
        gamma=0.0,
        action_space=ActionSpaceSpec.continuous(1),
        normalization_policy=mu,
        config=config,
    )
    second = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(8),
        gamma=0.0,
        action_space=ActionSpaceSpec.continuous(1),
        normalization_policy=mu,
        config=config,
    )
    assert np.allclose(first.predict_reward(states, actions), second.predict_reward(states, actions))
    assert np.isfinite(first.diagnostics["normalization_mc_se"])


def test_deepgenpqr_portable_serialization_roundtrip_for_pooled_fake_q(tmp_path) -> None:
    states = np.arange(8.0).reshape(-1, 1)
    actions = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(8),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        config=DeepGenPQRConfig(
            policy="behavior_cloning_native",
            q_backend=ConstantQEstimator(),
            policy_config={"n_epochs": 10},
        ),
    )
    result.save(tmp_path / "deep-portable")
    loaded = load_deep_genpqr_result(tmp_path / "deep-portable")
    assert loaded.q_mode == "pooled_fqe"
    assert loaded.q_backend == "unit_neural_fqe"
    assert np.allclose(loaded.predict_reward(states, actions), result.predict_reward(states, actions))


def test_deepgenpqr_pickle_requires_explicit_opt_in(tmp_path) -> None:
    states = np.arange(4.0).reshape(-1, 1)
    actions = np.array([0, 1, 0, 1])
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(4),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        config=DeepGenPQRConfig(policy=FixedDiscretePolicy(), q_backend=NonPortableQEstimator()),
    )
    save_deep_genpqr_result(result, tmp_path / "deep-unsafe")
    with pytest.raises(ValueError, match="allow_pickle"):
        load_deep_genpqr_result(tmp_path / "deep-unsafe")
    loaded = load_deep_genpqr_result(tmp_path / "deep-unsafe", allow_pickle=True)
    assert loaded.q_mode == "pooled_fqe"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="action-head neural FQE requires torch")
def test_deepgenpqr_action_head_portable_serialization_roundtrip(tmp_path) -> None:
    states = np.linspace(-1.0, 1.0, 36).reshape(-1, 1)
    actions = (np.arange(36) % 3).astype(np.int64)
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=np.roll(states, shift=-1, axis=0),
        terminals=np.ones(36),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(3),
        normalization_policy=DiscreteNormalizationPolicy.anchor(3, 0),
        anchor_function=0.0,
        config=DeepGenPQRConfig(
            policy="behavior_cloning_native",
            q_mode="pooled_fqe",
            policy_config={"n_epochs": 20, "seed": 23},
            q_config={
                "config_overrides": {
                    "hidden_dims": (16,),
                    "head_hidden_dims": (8,),
                    "batch_size": 12,
                    "num_iterations": 4,
                    "gradient_steps_per_iteration": 2,
                    "patience": 2,
                    "seed": 23,
                },
                "n_next_action_samples": 3,
            },
            seed=23,
        ),
    )
    save_deep_genpqr_result(result, tmp_path / "deep-action-head")
    with (tmp_path / "deep-action-head" / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["serialization_mode"] == "portable"
    assert manifest["q_backend"] == "action_head_neural_fqe"
    assert manifest["genpqr_payload"]["q_function"]["class"] == "ActionHeadNeuralFQEFunction"
    loaded = load_deep_genpqr_result(tmp_path / "deep-action-head")
    assert loaded.q_backend == "action_head_neural_fqe"
    assert np.allclose(
        loaded.predict_reward(states, actions),
        result.predict_reward(states, actions),
        atol=1e-6,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="action-head neural FQE requires torch")
def test_deepgenpqr_action_head_pickle_fallback_for_nonportable_policy(tmp_path) -> None:
    states = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
    actions = (np.arange(30) % 2).astype(np.int64)
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(30),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.anchor(2, 0),
        anchor_function=0.0,
        config=DeepGenPQRConfig(
            policy=FixedDiscretePolicy(),
            q_mode="pooled_fqe",
            q_config={
                "config_overrides": {
                    "hidden_dims": (16,),
                    "head_hidden_dims": (8,),
                    "batch_size": 10,
                    "num_iterations": 3,
                    "gradient_steps_per_iteration": 2,
                    "patience": 2,
                    "seed": 41,
                },
                "n_next_action_samples": 2,
            },
            seed=41,
        ),
    )
    save_deep_genpqr_result(result, tmp_path / "deep-action-head-unsafe")
    with pytest.raises(ValueError, match="allow_pickle"):
        load_deep_genpqr_result(tmp_path / "deep-action-head-unsafe")
    loaded = load_deep_genpqr_result(tmp_path / "deep-action-head-unsafe", allow_pickle=True)
    assert loaded.q_backend == "action_head_neural_fqe"
    assert np.allclose(
        loaded.predict_reward(states, actions),
        result.predict_reward(states, actions),
        atol=1e-6,
    )


def test_auto_neural_fqe_routes_continuous_actions_to_generic_backend(monkeypatch) -> None:
    from genpqr import q_estimators

    calls: dict[str, object] = {}

    class FakeGenericFQE:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, **kwargs):
            calls["fit"] = kwargs
            return ContinuousQFunction(kwargs["normalization_policy"].action_space)

    monkeypatch.setattr(q_estimators, "FQEQEstimator", FakeGenericFQE)
    estimator = q_estimators.AutoNeuralFQEstimator()
    states = np.zeros((5, 1))
    mu = ContinuousNormalizationPolicy(
        action_dim=1,
        sampler=lambda states_arg, rng, n_samples: np.zeros((states_arg.shape[0], int(n_samples), 1)),
    )
    fitted = estimator.fit(
        states=states,
        actions=np.zeros((5, 1)),
        next_states=states,
        pseudo_rewards=np.zeros(5),
        normalization_policy=mu,
        gamma=0.0,
        terminals=np.ones(5),
        policy=GaussianPolicy(),
    )
    assert calls["init"]["family"] == "neural"
    assert fitted.diagnostics["backend"] == "unit_continuous_neural_fqe"


def test_deepgenpqr_anchor_fallback_error_is_strict() -> None:
    states = np.zeros((6, 1))
    actions = np.zeros(6, dtype=np.int64)
    with pytest.raises(GenPQRConfigurationError, match="anchor"):
        fit_deep_genpqr(
            states=states,
            actions=actions,
            next_states=states,
            terminals=np.ones(6),
            gamma=0.0,
            action_space=ActionSpaceSpec.discrete(2),
            config=DeepGenPQRConfig(
                policy=FixedDiscretePolicy(),
                q_mode="anchor_deeppqr",
                anchor_action=1,
                min_anchor_count=1,
            ),
        )


def test_deepgenpqr_anchor_fallback_refits_pooled() -> None:
    states = np.zeros((6, 1))
    actions = np.zeros(6, dtype=np.int64)
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(6),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        config=DeepGenPQRConfig(
            policy=FixedDiscretePolicy(),
            q_mode="anchor_deeppqr",
            q_backend=ConstantQEstimator(),
            anchor_action=1,
            min_anchor_count=1,
            anchor_fallback="pooled_fqe",
        ),
    )
    assert result.q_mode == "pooled_fqe"
    assert result.diagnostics["anchor_fallback_used"] is True
    assert "anchor_fallback_to_pooled_fqe" in result.diagnostics["warning_codes"]


def test_neural_deeppqr_continuous_anchor_masks() -> None:
    action_space = ActionSpaceSpec.continuous(1)
    states = np.zeros((4, 1))
    actions = np.array([[0.0], [0.2], [0.0], [0.3]])
    fixed_mask, _, _, fixed_diag = _resolve_anchor_rows(
        action_space=action_space,
        states=states,
        actions=actions,
        next_states=states,
        anchor_action=0.0,
        anchor_selector=None,
        anchor_tolerance=1e-12,
    )
    selector_mask, _, _, selector_diag = _resolve_anchor_rows(
        action_space=action_space,
        states=states,
        actions=actions,
        next_states=states,
        anchor_action=0.0,
        anchor_selector=lambda s, a: np.isclose(a[:, 0], 0.0),
        anchor_tolerance=1e-12,
    )
    assert fixed_mask.tolist() == [True, False, True, False]
    assert selector_mask.tolist() == [True, False, True, False]
    assert fixed_diag["anchor_kind"] == "continuous_fixed"
    assert selector_diag["anchor_kind"] == "continuous_selector"
    callable_anchor_mask, _, _, callable_diag = _resolve_anchor_rows(
        action_space=action_space,
        states=states,
        actions=actions,
        next_states=states,
        anchor_action=lambda s: np.full((s.shape[0], 1), 0.3),
        anchor_selector=lambda s, a: a[:, 0] > 0.25,
        anchor_tolerance=1e-12,
    )
    assert callable_anchor_mask.tolist() == [False, False, False, True]
    assert callable_diag["anchor_kind"] == "continuous_selector"
    with pytest.raises(GenPQRConfigurationError, match="anchor_selector selected rows"):
        _resolve_anchor_rows(
            action_space=action_space,
            states=states,
            actions=actions,
            next_states=states,
            anchor_action=0.0,
            anchor_selector=lambda s, a: a[:, 0] > 0.25,
            anchor_tolerance=1e-12,
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_deepgenpqr_anchor_mode_continuous_smoke() -> None:
    states = np.zeros((8, 1))
    actions = np.zeros((8, 1))
    mu = ContinuousNormalizationPolicy(
        action_dim=1,
        sampler=lambda states_arg, rng, n_samples: np.zeros((states_arg.shape[0], int(n_samples), 1)),
    )
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(8),
        gamma=0.0,
        action_space=ActionSpaceSpec.continuous(1),
        normalization_policy=mu,
        config=DeepGenPQRConfig(
            policy=GaussianPolicy(),
            q_mode="anchor_deeppqr",
            anchor_action=0.0,
            anchor_tolerance=1e-12,
            min_anchor_count=1,
            q_config={"hidden_dims": (4,), "max_epochs": 3, "patience": 2, "batch_size": 4},
        ),
    )
    assert result.diagnostics["anchor_enabled"] is True
    assert result.diagnostics["anchor_support"]["anchor_count"] == 8
    assert np.all(np.isfinite(result.predict_reward(states, actions)))
