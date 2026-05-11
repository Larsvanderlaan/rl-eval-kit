from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import QuadraticStateActionFunction
from .policies import GaussianLinearPolicy


@dataclass
class LinearGaussianEnvConfig:
    """Controlled linear-Gaussian benchmark family."""

    B: np.ndarray = field(
        default_factory=lambda: np.array([[0.9, 0.05], [0.0, 0.8]], dtype=np.float64)
    )
    C: np.ndarray = field(default_factory=lambda: np.array([[0.8], [0.25]], dtype=np.float64))
    target_gain: np.ndarray = field(default_factory=lambda: np.array([[-0.9, -0.3]], dtype=np.float64))
    behavior_shift_direction: np.ndarray = field(default_factory=lambda: np.array([[0.8, -0.6]], dtype=np.float64))
    state_reward_matrix: np.ndarray = field(
        default_factory=lambda: np.diag(np.array([1.25, 0.55], dtype=np.float64))
    )
    state_goal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    action_penalty: float = 0.35
    initial_mean: np.ndarray = field(default_factory=lambda: np.array([2.0, -1.3], dtype=np.float64))
    initial_cov: np.ndarray = field(
        default_factory=lambda: np.diag(np.array([0.50, 0.35], dtype=np.float64))
    )
    target_action_sd: float = 0.10
    behavior_action_sd: float = 0.10
    process_noise_sd: float = 0.05
    reward_noise_sd: float = 0.0
    stability_tol: float = 0.995

    def __post_init__(self) -> None:
        self.B = np.asarray(self.B, dtype=np.float64).reshape(2, 2)
        self.C = np.asarray(self.C, dtype=np.float64).reshape(2, 1)
        self.target_gain = np.asarray(self.target_gain, dtype=np.float64).reshape(1, 2)
        self.behavior_shift_direction = np.asarray(self.behavior_shift_direction, dtype=np.float64).reshape(1, 2)
        self.state_reward_matrix = np.asarray(self.state_reward_matrix, dtype=np.float64).reshape(2, 2)
        self.state_goal = np.asarray(self.state_goal, dtype=np.float64).reshape(2)
        self.initial_mean = np.asarray(self.initial_mean, dtype=np.float64).reshape(2)
        self.initial_cov = np.asarray(self.initial_cov, dtype=np.float64).reshape(2, 2)


@dataclass
class LinearGaussianEnv:
    config: LinearGaussianEnvConfig

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def process_noise_cov(self) -> np.ndarray:
        return (self.config.process_noise_sd**2) * np.eye(self.state_dim, dtype=np.float64)

    def target_policy(self) -> GaussianLinearPolicy:
        return GaussianLinearPolicy(
            gain=self.config.target_gain,
            action_sd=self.config.target_action_sd,
            name="target_policy",
        )

    def behavior_policy(self, shift: float, action_sd: float | None = None) -> GaussianLinearPolicy:
        gain = self.config.target_gain + float(shift) * self.config.behavior_shift_direction
        policy = GaussianLinearPolicy(
            gain=gain,
            action_sd=self.config.behavior_action_sd if action_sd is None else float(action_sd),
            name=f"behavior_shift_{shift:.2f}",
        )
        if not self.is_stable(policy):
            radius = self.closed_loop_spectral_radius(policy)
            raise ValueError(
                f"Behavior policy at shift={shift:.3f} is unstable with spectral radius {radius:.4f}."
            )
        return policy

    def closed_loop_matrix(self, policy: GaussianLinearPolicy) -> np.ndarray:
        return self.config.B + self.config.C @ policy.gain

    def closed_loop_spectral_radius(self, policy: GaussianLinearPolicy) -> float:
        eigvals = np.linalg.eigvals(self.closed_loop_matrix(policy))
        return float(np.max(np.abs(eigvals)))

    def is_stable(self, policy: GaussianLinearPolicy) -> bool:
        return bool(self.closed_loop_spectral_radius(policy) < self.config.stability_tol)

    def closed_loop_noise_cov(self, policy: GaussianLinearPolicy) -> np.ndarray:
        return self.process_noise_cov + (policy.action_sd**2) * (self.config.C @ self.config.C.T)

    def state_transition_mean(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim)
        actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
        return states_arr @ self.config.B.T + actions_arr @ self.config.C.T

    def step(self, states: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = self.state_transition_mean(states, actions)
        noise = self.config.process_noise_sd * rng.normal(size=mean.shape)
        return mean + noise

    def reward_function(self) -> QuadraticStateActionFunction:
        state_goal = self.config.state_goal
        q_goal = self.config.state_reward_matrix @ state_goal
        constant = -float(state_goal @ q_goal)
        linear = np.array([2.0 * q_goal[0], 2.0 * q_goal[1], 0.0], dtype=np.float64)
        quadratic = np.zeros((3, 3), dtype=np.float64)
        quadratic[:2, :2] = -self.config.state_reward_matrix
        quadratic[2, 2] = -float(self.config.action_penalty)
        return QuadraticStateActionFunction(
            constant=constant,
            linear=linear,
            quadratic=quadratic,
        )

    def expected_reward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.reward_function().evaluate(states, actions)

    def sample_reward(self, states: np.ndarray, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean_reward = self.expected_reward(states, actions)
        if self.config.reward_noise_sd <= 0.0:
            return mean_reward
        return mean_reward + self.config.reward_noise_sd * rng.normal(size=mean_reward.shape[0])
