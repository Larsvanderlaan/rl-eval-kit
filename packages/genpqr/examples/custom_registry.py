"""Custom estimator registry example."""

from __future__ import annotations

import numpy as np

from genpqr import ActionSpaceSpec, DiscreteNormalizationPolicy, available_q_estimators, register_q_estimator
from genpqr import testing as genpqr_testing


class MyQEstimator:
    def fit(self, **kwargs):
        del kwargs
        return MyQFunction()


class MyQFunction:
    action_space = ActionSpaceSpec.discrete(2)

    def predict_q(self, states, actions):
        idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
        return idx.astype(float)

    def predict_q_matrix(self, states):
        return np.tile(np.array([[0.0, 1.0]]), (np.asarray(states).shape[0], 1))

    def expected_q(self, states, normalization_policy, *, n_action_samples, rng):
        del n_action_samples, rng
        return np.sum(normalization_policy.predict_proba(states) * self.predict_q_matrix(states), axis=1)


def main() -> None:
    register_q_estimator("my_q", MyQEstimator, overwrite=True)
    states = np.zeros((3, 1))
    genpqr_testing.check_fitted_q_contract(
        MyQFunction(),
        states=states,
        actions=np.array([0, 1, 0]),
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
    )
    print("my_q" in available_q_estimators())


if __name__ == "__main__":
    main()
