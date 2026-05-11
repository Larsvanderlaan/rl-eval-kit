"""Continuous-action DeepGenPQR anchor-selector example."""

from __future__ import annotations

import numpy as np

from genpqr import ActionSpaceSpec, ContinuousNormalizationPolicy, DeepGenPQRConfig, fit_deep_genpqr


class GaussianPolicy:
    action_space = ActionSpaceSpec.continuous(1)

    def log_prob(self, states, actions):
        del states
        actions = self.action_space.action_matrix(actions)
        return -0.5 * actions[:, 0] ** 2

    def sample(self, states, rng, n_samples=1):
        return rng.normal(size=(np.asarray(states).shape[0], int(n_samples), 1))


def main() -> None:
    states = np.zeros((10, 1))
    actions = np.zeros((10, 1))
    mu = ContinuousNormalizationPolicy(
        action_dim=1,
        sampler=lambda states_arg, rng, n: np.zeros((states_arg.shape[0], int(n), 1)),
    )
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(10),
        gamma=0.0,
        action_space=ActionSpaceSpec.continuous(1),
        normalization_policy=mu,
        config=DeepGenPQRConfig(
            policy=GaussianPolicy(),
            q_mode="anchor_deeppqr",
            anchor_action=0.0,
            anchor_selector=lambda states_arg, actions_arg: np.isclose(actions_arg[:, 0], 0.0),
            min_anchor_count=1,
            q_config={"hidden_dims": (8,), "max_epochs": 5, "patience": 2},
        ),
    )
    print(result.diagnostics["anchor_support"])


if __name__ == "__main__":
    main()
