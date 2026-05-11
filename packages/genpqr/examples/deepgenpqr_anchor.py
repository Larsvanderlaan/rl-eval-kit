"""DeepGenPQR with DeepPQR-style finite-action anchor Q."""

from __future__ import annotations

import numpy as np

from genpqr import ActionSpaceSpec, DeepGenPQRConfig, DiscreteNormalizationPolicy, fit_deep_genpqr


class FixedPolicy:
    action_space = ActionSpaceSpec.discrete(2)

    def predict_proba(self, states):
        return np.tile(np.array([0.6, 0.4]), (np.asarray(states).shape[0], 1))

    def log_prob(self, states, actions):
        idx = self.action_space.action_indices(actions, n_rows=np.asarray(states).shape[0])
        probs = self.predict_proba(states)
        return np.log(probs[np.arange(idx.shape[0]), idx])

    def sample(self, states, rng, n_samples=1):
        draws = np.zeros((np.asarray(states).shape[0], int(n_samples)), dtype=np.int64)
        return draws.reshape(-1) if int(n_samples) == 1 else draws


def main() -> None:
    states = np.linspace(-1.0, 1.0, 12).reshape(-1, 1)
    actions = np.zeros(12, dtype=np.int64)
    result = fit_deep_genpqr(
        states=states,
        actions=actions,
        next_states=states,
        terminals=np.ones(12),
        gamma=0.0,
        action_space=ActionSpaceSpec.discrete(2),
        normalization_policy=DiscreteNormalizationPolicy.uniform(2),
        config=DeepGenPQRConfig(
            policy=FixedPolicy(),
            q_mode="anchor_deeppqr",
            anchor_action=0,
            min_anchor_count=1,
            q_config={"hidden_dims": (8,), "max_epochs": 5, "patience": 2},
        ),
    )
    print(result.summary())


if __name__ == "__main__":
    main()
