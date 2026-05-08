from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianLinearPolicy:
    """Linear-Gaussian policy a ~ N(K s, sigma_a^2)."""

    gain: np.ndarray
    action_sd: float
    name: str = "policy"

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=np.float64).reshape(1, -1)
        object.__setattr__(self, "gain", gain)
        if self.action_sd < 0.0:
            raise ValueError("action_sd must be nonnegative.")

    @property
    def state_dim(self) -> int:
        return int(self.gain.shape[1])

    @property
    def action_dim(self) -> int:
        return 1

    def mean_action(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim)
        return states_arr @ self.gain.T

    def conditional_action_variance(self) -> float:
        return float(self.action_sd**2)

    def sample_actions(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.mean_action(states)
        noise = self.action_sd * rng.normal(size=mean.shape)
        return mean + noise

    def joint_moments_from_state_gaussian(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state_mean_arr = np.asarray(state_mean, dtype=np.float64).reshape(-1)
        state_cov_arr = np.asarray(state_cov, dtype=np.float64).reshape(self.state_dim, self.state_dim)
        mean_action = (self.gain @ state_mean_arr).reshape(1)
        cross_cov = state_cov_arr @ self.gain.T
        action_var = float(self.gain @ state_cov_arr @ self.gain.T + self.action_sd**2)
        joint_mean = np.concatenate([state_mean_arr, mean_action], axis=0)
        joint_cov = np.block(
            [
                [state_cov_arr, cross_cov],
                [cross_cov.T, np.asarray([[action_var]], dtype=np.float64)],
            ]
        )
        return joint_mean, joint_cov
