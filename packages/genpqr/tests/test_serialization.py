from __future__ import annotations

import numpy as np
import pytest

from genpqr import (
    ActionSpaceSpec,
    DiscreteNormalizationPolicy,
    GenPQRConfig,
    fit_genpqr,
    load_genpqr_result,
    save_genpqr_result,
)
from genpqr.benchmarks import make_tabular_chain


class CustomQ:
    action_space = ActionSpaceSpec.discrete(2)

    def fit(self, **kwargs):
        del kwargs
        return self

    def predict_q(self, states, actions):
        del actions
        return np.zeros(np.asarray(states).shape[0])

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        del normalization_policy, n_action_samples, rng
        return np.zeros(np.asarray(states).shape[0])


def test_safe_portable_serialization_roundtrip(tmp_path) -> None:
    dataset = make_tabular_chain(24, seed=14)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.0,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q="deeppqr_linear", policy_config={"n_epochs": 20}),
    )
    save_genpqr_result(result, tmp_path / "portable")
    loaded = load_genpqr_result(tmp_path / "portable")
    assert np.allclose(
        loaded.predict_reward(dataset.states[:5], dataset.actions[:5]),
        result.predict_reward(dataset.states[:5], dataset.actions[:5]),
    )


def test_pickle_fallback_requires_explicit_opt_in(tmp_path) -> None:
    dataset = make_tabular_chain(10, seed=2)
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.0,
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=GenPQRConfig(policy="behavior_cloning_native", q=CustomQ(), policy_config={"n_epochs": 5}),
    )
    save_genpqr_result(result, tmp_path / "unsafe")
    with pytest.raises(ValueError, match="allow_pickle"):
        load_genpqr_result(tmp_path / "unsafe")
    assert load_genpqr_result(tmp_path / "unsafe", allow_pickle=True).action_space.kind == "discrete"
