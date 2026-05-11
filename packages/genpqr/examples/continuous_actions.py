"""Continuous-action GenPQR example."""

from __future__ import annotations

import numpy as np

from genpqr import ActionSpaceSpec, ContinuousNormalizationPolicy, GenPQRConfig, fit_genpqr
from genpqr.benchmarks import make_linear_gaussian


class LinearQEstimator:
    action_space = ActionSpaceSpec.continuous(1)

    def fit(self, **kwargs):
        return self

    def predict_q(self, states, actions):
        actions = self.action_space.action_matrix(actions, n_rows=np.asarray(states).shape[0])
        return np.asarray(states)[:, 0] + actions[:, 0]

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        samples = normalization_policy.sample(states, rng, n_action_samples)
        return np.mean([self.predict_q(states, samples[:, j, :]) for j in range(samples.shape[1])], axis=0)


def main() -> None:
    dataset = make_linear_gaussian(30)
    mu = ContinuousNormalizationPolicy(
        action_dim=1,
        sampler=lambda states, rng, n: rng.normal(size=(states.shape[0], n, 1)),
    )
    result = fit_genpqr(
        dataset=dataset,
        gamma=0.0,
        normalization_policy=mu,
        config=GenPQRConfig(policy="behavior_cloning_native", q=LinearQEstimator(), n_action_samples=8),
    )
    print(result.diagnostics["continuous_mc_standard_error_mean"])


if __name__ == "__main__":
    main()
