from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .envs import LinearGaussianEnv
from .features import QuadraticStateActionFunction, QuadraticStateValueFunction
from .policies import GaussianLinearPolicy


def _logsumexp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    max_val = np.max(values, axis=axis, keepdims=True)
    stabilized = values - max_val
    summed = np.sum(np.exp(stabilized), axis=axis, keepdims=True)
    out = max_val + np.log(np.maximum(summed, 1e-300))
    return np.squeeze(out, axis=axis)


@dataclass
class GaussianMixtureDensity:
    """Finite Gaussian mixture with exact sampling and log-density evaluation."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        weights = weights / np.maximum(weights.sum(), 1e-12)
        means = np.asarray(self.means, dtype=np.float64)
        covs = np.asarray(self.covariances, dtype=np.float64)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "covariances", covs)
        dim = means.shape[1]
        log_norms = []
        cholesky = []
        for cov in covs:
            chol = np.linalg.cholesky(cov + 1e-10 * np.eye(dim, dtype=np.float64))
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            log_norms.append(-0.5 * (dim * np.log(2.0 * np.pi) + log_det))
            cholesky.append(chol)
        object.__setattr__(self, "_log_norms", np.asarray(log_norms, dtype=np.float64))
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def marginal(self, dims: slice | list[int] | np.ndarray) -> "GaussianMixtureDensity":
        return GaussianMixtureDensity(
            weights=self.weights.copy(),
            means=self.means[:, dims],
            covariances=self.covariances[:, dims][:, :, dims],
        )

    def sample(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        component_ids = rng.choice(self.weights.shape[0], size=n_samples, p=self.weights)
        draws = np.zeros((n_samples, self.dim), dtype=np.float64)
        for component in range(self.weights.shape[0]):
            mask = component_ids == component
            if not np.any(mask):
                continue
            draws[mask] = rng.multivariate_normal(
                mean=self.means[component],
                cov=self.covariances[component],
                size=int(mask.sum()),
            )
        return draws

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self.dim)
        logs = []
        for idx, weight in enumerate(self.weights):
            diff = pts - self.means[idx]
            solved = np.linalg.solve(self._cholesky[idx], diff.T)
            maha = np.sum(solved * solved, axis=0)
            component = np.log(np.maximum(weight, 1e-300)) + self._log_norms[idx] - 0.5 * maha
            logs.append(component)
        stacked = np.vstack(logs)
        return _logsumexp(stacked, axis=0)


@dataclass
class PolicyMomentSequence:
    state_means: list[np.ndarray]
    state_covariances: list[np.ndarray]
    stationary_mean: np.ndarray
    stationary_cov: np.ndarray
    head_steps: int


@dataclass
class PolicyTruth:
    q_function: QuadraticStateActionFunction
    v_function: QuadraticStateValueFunction
    policy_value: float


def stationary_state_covariance(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
    tol: float = 1e-10,
    max_iters: int = 20_000,
) -> np.ndarray:
    transition = env.closed_loop_matrix(policy)
    noise_cov = env.closed_loop_noise_cov(policy)
    cov = np.zeros_like(noise_cov)
    for _ in range(max_iters):
        new_cov = transition @ cov @ transition.T + noise_cov
        if np.max(np.abs(new_cov - cov)) < tol:
            cov = new_cov
            break
        cov = new_cov
    return cov


def build_policy_moment_sequence(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
    gamma: float,
    min_head_steps: int | None = None,
    max_head_steps: int = 160,
    moment_tol: float = 1e-5,
) -> PolicyMomentSequence:
    transition = env.closed_loop_matrix(policy)
    noise_cov = env.closed_loop_noise_cov(policy)
    stationary_mean = np.zeros(env.state_dim, dtype=np.float64)
    stationary_cov = stationary_state_covariance(env, policy)
    state_means = [env.config.initial_mean.copy()]
    state_covariances = [env.config.initial_cov.copy()]
    if min_head_steps is None:
        min_head_steps = 60 if gamma >= 0.99 else 35
    head_steps = max_head_steps
    for step in range(max_head_steps):
        mean_next = transition @ state_means[-1]
        cov_next = transition @ state_covariances[-1] @ transition.T + noise_cov
        state_means.append(mean_next)
        state_covariances.append(cov_next)
        if step + 1 >= min_head_steps:
            mean_gap = np.max(np.abs(mean_next - stationary_mean))
            cov_gap = np.max(np.abs(cov_next - stationary_cov))
            if mean_gap < moment_tol and cov_gap < moment_tol:
                head_steps = step + 1
                break
    return PolicyMomentSequence(
        state_means=state_means,
        state_covariances=state_covariances,
        stationary_mean=stationary_mean,
        stationary_cov=stationary_cov,
        head_steps=head_steps,
    )


def build_discounted_occupancy_mixture(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
    gamma: float,
    min_head_steps: int | None = None,
    max_head_steps: int = 160,
    moment_tol: float = 1e-5,
) -> tuple[GaussianMixtureDensity, GaussianMixtureDensity, PolicyMomentSequence]:
    sequence = build_policy_moment_sequence(
        env=env,
        policy=policy,
        gamma=gamma,
        min_head_steps=min_head_steps,
        max_head_steps=max_head_steps,
        moment_tol=moment_tol,
    )
    means = []
    covs = []
    weights = []
    for t in range(sequence.head_steps + 1):
        weight_t = (1.0 - gamma) * (gamma**t)
        joint_mean, joint_cov = policy.joint_moments_from_state_gaussian(
            sequence.state_means[t],
            sequence.state_covariances[t],
        )
        means.append(joint_mean)
        covs.append(joint_cov)
        weights.append(weight_t)
    tail_weight = float(gamma ** (sequence.head_steps + 1))
    if tail_weight > 1e-12:
        tail_mean, tail_cov = policy.joint_moments_from_state_gaussian(
            sequence.stationary_mean,
            sequence.stationary_cov,
        )
        means.append(tail_mean)
        covs.append(tail_cov)
        weights.append(tail_weight)
    joint_mixture = GaussianMixtureDensity(
        weights=np.asarray(weights, dtype=np.float64),
        means=np.asarray(means, dtype=np.float64),
        covariances=np.asarray(covs, dtype=np.float64),
    )
    return joint_mixture, joint_mixture.marginal([0, 1]), sequence


def build_stationary_occupancy_mixture(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
) -> tuple[GaussianMixtureDensity, GaussianMixtureDensity, PolicyMomentSequence]:
    """One-component Gaussian state-action distribution under the stationary closed loop."""

    stationary_cov = stationary_state_covariance(env, policy)
    joint_mean, joint_cov = policy.joint_moments_from_state_gaussian(
        np.zeros(env.state_dim, dtype=np.float64),
        stationary_cov,
    )
    joint_mixture = GaussianMixtureDensity(
        weights=np.ones(1, dtype=np.float64),
        means=joint_mean.reshape(1, -1),
        covariances=joint_cov.reshape(1, joint_cov.shape[0], joint_cov.shape[1]),
    )
    sequence = PolicyMomentSequence(
        state_means=[np.zeros(env.state_dim, dtype=np.float64)],
        state_covariances=[stationary_cov],
        stationary_mean=np.zeros(env.state_dim, dtype=np.float64),
        stationary_cov=stationary_cov,
        head_steps=0,
    )
    return joint_mixture, joint_mixture.marginal([0, 1]), sequence


def build_reference_occupancy_mixture(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
    ratio_gamma: float,
) -> tuple[GaussianMixtureDensity, GaussianMixtureDensity, PolicyMomentSequence, str]:
    """Build the distribution targeted by the weighting gamma.

    `ratio_gamma < 1` gives discounted occupancy. `ratio_gamma == 1` gives the
    stationary closed-loop state-action distribution.
    """

    if ratio_gamma >= 1.0 - 1e-12:
        joint, state, sequence = build_stationary_occupancy_mixture(env, policy)
        return joint, state, sequence, "stationary"
    joint, state, sequence = build_discounted_occupancy_mixture(env, policy, gamma=ratio_gamma)
    return joint, state, sequence, "discounted"


def solve_policy_truth(
    env: LinearGaussianEnv,
    policy: GaussianLinearPolicy,
    gamma: float,
    tol: float = 1e-10,
    max_iters: int = 20_000,
) -> PolicyTruth:
    reward_q = env.reward_function()
    reward_v = reward_q.to_state_value(policy)
    transition = env.closed_loop_matrix(policy)
    noise_cov = env.closed_loop_noise_cov(policy)

    value_fn = QuadraticStateValueFunction(
        constant=0.0,
        linear=np.zeros(env.state_dim, dtype=np.float64),
        quadratic=np.zeros((env.state_dim, env.state_dim), dtype=np.float64),
    )
    for _ in range(max_iters):
        new_value = QuadraticStateValueFunction(
            constant=reward_v.constant + gamma * (
                value_fn.constant + float(np.trace(value_fn.quadratic @ noise_cov))
            ),
            linear=reward_v.linear + gamma * (transition.T @ value_fn.linear),
            quadratic=reward_v.quadratic + gamma * (transition.T @ value_fn.quadratic @ transition),
        )
        delta = max(
            abs(new_value.constant - value_fn.constant),
            float(np.max(np.abs(new_value.linear - value_fn.linear))),
            float(np.max(np.abs(new_value.quadratic - value_fn.quadratic))),
        )
        value_fn = new_value
        if delta < tol:
            break

    transition_sa = np.concatenate([env.config.B, env.config.C], axis=1)
    q_function = QuadraticStateActionFunction(
        constant=reward_q.constant
        + gamma * (
            value_fn.constant + float(np.trace(value_fn.quadratic @ env.process_noise_cov))
        ),
        linear=reward_q.linear + gamma * (transition_sa.T @ value_fn.linear),
        quadratic=reward_q.quadratic + gamma * (transition_sa.T @ value_fn.quadratic @ transition_sa),
    )
    policy_value = value_fn.expectation_under_gaussian(env.config.initial_mean, env.config.initial_cov)
    return PolicyTruth(q_function=q_function, v_function=value_fn, policy_value=policy_value)
