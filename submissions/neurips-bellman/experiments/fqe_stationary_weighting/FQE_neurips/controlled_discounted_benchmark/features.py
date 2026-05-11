from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import GaussianLinearPolicy


def _ensure_states_actions(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, 1)
    if states_arr.shape[0] != actions_arr.shape[0]:
        raise ValueError("states and actions must have the same number of rows.")
    return states_arr, actions_arr


@dataclass(frozen=True)
class QuadraticStateValueFunction:
    """Quadratic state value c + l^T s + s^T H s."""

    constant: float
    linear: np.ndarray
    quadratic: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", np.asarray(self.linear, dtype=np.float64).reshape(2))
        quad = np.asarray(self.quadratic, dtype=np.float64).reshape(2, 2)
        object.__setattr__(self, "quadratic", 0.5 * (quad + quad.T))

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        quad = np.einsum("ni,ij,nj->n", states_arr, self.quadratic, states_arr)
        return self.constant + states_arr @ self.linear + quad

    def expectation_under_gaussian(self, mean: np.ndarray, cov: np.ndarray) -> float:
        mean_arr = np.asarray(mean, dtype=np.float64).reshape(2)
        cov_arr = np.asarray(cov, dtype=np.float64).reshape(2, 2)
        quad = float(mean_arr @ self.quadratic @ mean_arr + np.trace(self.quadratic @ cov_arr))
        return float(self.constant + self.linear @ mean_arr + quad)

    def expectation_under_transition(self, means: np.ndarray, transition_cov: np.ndarray) -> np.ndarray:
        means_arr = np.asarray(means, dtype=np.float64).reshape(-1, 2)
        cov_arr = np.asarray(transition_cov, dtype=np.float64).reshape(2, 2)
        quad = np.einsum("ni,ij,nj->n", means_arr, self.quadratic, means_arr)
        trace_term = float(np.trace(self.quadratic @ cov_arr))
        return self.constant + means_arr @ self.linear + quad + trace_term


@dataclass(frozen=True)
class QuadraticStateActionFunction:
    """Quadratic state-action function c + l^T x + x^T H x for x=(s1,s2,a)."""

    constant: float
    linear: np.ndarray
    quadratic: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear", np.asarray(self.linear, dtype=np.float64).reshape(3))
        quad = np.asarray(self.quadratic, dtype=np.float64).reshape(3, 3)
        object.__setattr__(self, "quadratic", 0.5 * (quad + quad.T))

    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr, actions_arr = _ensure_states_actions(states, actions)
        stacked = np.concatenate([states_arr, actions_arr], axis=1)
        quad = np.einsum("ni,ij,nj->n", stacked, self.quadratic, stacked)
        return self.constant + stacked @ self.linear + quad

    def to_state_value(self, policy: GaussianLinearPolicy) -> QuadraticStateValueFunction:
        gain = policy.gain.reshape(-1)
        gain_map = np.array([[1.0, 0.0], [0.0, 1.0], [gain[0], gain[1]]], dtype=np.float64)
        linear_state = gain_map.T @ self.linear
        quadratic_state = gain_map.T @ self.quadratic @ gain_map
        constant_state = float(self.constant + self.quadratic[2, 2] * policy.action_sd**2)
        return QuadraticStateValueFunction(
            constant=constant_state,
            linear=linear_state,
            quadratic=quadratic_state,
        )


class StateActionFeatureMap:
    """Polynomial state-action features with exact policy expectations."""

    VALID_REGIMES = (
        "well_specified",
        "misspecified_affine",
        "misspecified_state_affine",
        "misspecified_diag_quad",
    )

    def __init__(self, regime: str) -> None:
        if regime not in self.VALID_REGIMES:
            raise ValueError(f"Unknown feature regime '{regime}'.")
        self.regime = regime

    @property
    def dimension(self) -> int:
        if self.regime == "well_specified":
            return 10
        if self.regime == "misspecified_state_affine":
            return 3
        if self.regime == "misspecified_affine":
            return 4
        return 7

    def transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr, actions_arr = _ensure_states_actions(states, actions)
        s1 = states_arr[:, [0]]
        s2 = states_arr[:, [1]]
        a = actions_arr
        if self.regime == "well_specified":
            return np.concatenate(
                [
                    np.ones_like(a),
                    s1,
                    s2,
                    a,
                    s1**2,
                    s1 * s2,
                    s1 * a,
                    s2**2,
                    s2 * a,
                    a**2,
                ],
                axis=1,
            )
        if self.regime == "misspecified_state_affine":
            return np.concatenate([np.ones_like(a), s1, s2], axis=1)
        if self.regime == "misspecified_affine":
            return np.concatenate([np.ones_like(a), s1, s2, a], axis=1)
        return np.concatenate([np.ones_like(a), s1, s2, a, s1**2, s2**2, a**2], axis=1)

    def expected_features_given_state(
        self,
        states: np.ndarray,
        policy: GaussianLinearPolicy,
    ) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        mean_action = policy.mean_action(states_arr)
        action_second = mean_action**2 + policy.action_sd**2
        s1 = states_arr[:, [0]]
        s2 = states_arr[:, [1]]
        a_mean = mean_action
        if self.regime == "well_specified":
            return np.concatenate(
                [
                    np.ones_like(a_mean),
                    s1,
                    s2,
                    a_mean,
                    s1**2,
                    s1 * s2,
                    s1 * a_mean,
                    s2**2,
                    s2 * a_mean,
                    action_second,
                ],
                axis=1,
            )
        if self.regime == "misspecified_state_affine":
            return np.concatenate([np.ones_like(a_mean), s1, s2], axis=1)
        if self.regime == "misspecified_affine":
            return np.concatenate([np.ones_like(a_mean), s1, s2, a_mean], axis=1)
        return np.concatenate([np.ones_like(a_mean), s1, s2, a_mean, s1**2, s2**2, action_second], axis=1)

    def quadratic_form_from_theta(self, theta: np.ndarray) -> QuadraticStateActionFunction:
        theta_arr = np.asarray(theta, dtype=np.float64).reshape(self.dimension)
        linear = np.zeros(3, dtype=np.float64)
        quad = np.zeros((3, 3), dtype=np.float64)
        constant = float(theta_arr[0])
        if self.regime == "well_specified":
            linear[:] = theta_arr[1:4]
            quad[0, 0] = theta_arr[4]
            quad[0, 1] = quad[1, 0] = 0.5 * theta_arr[5]
            quad[0, 2] = quad[2, 0] = 0.5 * theta_arr[6]
            quad[1, 1] = theta_arr[7]
            quad[1, 2] = quad[2, 1] = 0.5 * theta_arr[8]
            quad[2, 2] = theta_arr[9]
        elif self.regime == "misspecified_state_affine":
            linear[:2] = theta_arr[1:3]
        elif self.regime == "misspecified_affine":
            linear[:] = theta_arr[1:4]
        else:
            linear[:] = theta_arr[1:4]
            quad[0, 0] = theta_arr[4]
            quad[1, 1] = theta_arr[5]
            quad[2, 2] = theta_arr[6]
        return QuadraticStateActionFunction(
            constant=constant,
            linear=linear,
            quadratic=quad,
        )


@dataclass(frozen=True)
class RBFStateValueFunction:
    feature_map: "RatioFeatureMap"
    theta: np.ndarray
    policy: GaussianLinearPolicy

    def evaluate(self, states: np.ndarray) -> np.ndarray:
        return self.feature_map.expected_given_state(states, self.policy) @ self.theta

    def expectation_under_gaussian(self, mean: np.ndarray, cov: np.ndarray) -> float:
        features = self.feature_map.expectation_under_state_gaussian(mean, cov, self.policy)
        return float(features @ self.theta)

    def expectation_under_transition(self, means: np.ndarray, transition_cov: np.ndarray) -> np.ndarray:
        means_arr = np.asarray(means, dtype=np.float64).reshape(-1, 2)
        return np.asarray(
            [
                self.expectation_under_gaussian(mean, transition_cov)
                for mean in means_arr
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RBFStateActionFunction:
    feature_map: "RatioFeatureMap"
    theta: np.ndarray

    def evaluate(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.feature_map.transform(states, actions) @ self.theta

    def to_state_value(self, policy: GaussianLinearPolicy) -> RBFStateValueFunction:
        return RBFStateValueFunction(
            feature_map=self.feature_map,
            theta=np.asarray(self.theta, dtype=np.float64).reshape(-1),
            policy=policy,
        )


def _gaussian_rbf_expectation(mean: np.ndarray, cov: np.ndarray, center: np.ndarray, bandwidth: float) -> float:
    dim = mean.shape[0]
    bw2 = float(bandwidth**2)
    system = cov + bw2 * np.eye(dim, dtype=np.float64)
    diff = mean - center
    sign, logdet = np.linalg.slogdet(np.eye(dim, dtype=np.float64) + cov / bw2)
    if sign <= 0:
        raise ValueError("Encountered non-positive determinant in Gaussian RBF expectation.")
    exponent = -0.5 * float(diff @ np.linalg.solve(system, diff))
    return float(np.exp(-0.5 * logdet + exponent))


@dataclass
class RatioFeatureMap:
    """Shared ratio-estimation basis: polynomials plus deterministic RBF centers."""

    centers: np.ndarray
    bandwidth: float
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    @classmethod
    def from_behavior_samples(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        n_centers: int = 12,
        bandwidth: float | str = "median",
        bandwidth_scale: float = 1.0,
        standardize_features: bool = False,
    ) -> "RatioFeatureMap":
        states_arr, actions_arr = _ensure_states_actions(states, actions)
        stacked = np.concatenate([states_arr, actions_arr], axis=1)
        if stacked.shape[0] == 0 or n_centers <= 0:
            centers = np.zeros((0, 3), dtype=np.float64)
            feature_map = cls(centers=centers, bandwidth=1.0)
            if not standardize_features:
                return feature_map
            raw_features = feature_map._raw_transform(states_arr, actions_arr)
            mean, scale = cls._standardization_from_raw_features(raw_features)
            return cls(centers=centers, bandwidth=1.0, feature_mean=mean, feature_scale=scale)
        if stacked.shape[0] <= n_centers:
            centers = stacked.copy()
        else:
            indices = np.linspace(0, stacked.shape[0] - 1, num=n_centers, dtype=np.int64)
            centers = stacked[indices]
        if bandwidth == "median":
            if centers.shape[0] <= 1:
                bw = 1.0
            else:
                diffs = centers[:, None, :] - centers[None, :, :]
                distances = np.sqrt(np.sum(diffs**2, axis=2))
                upper = distances[np.triu_indices(centers.shape[0], k=1)]
                positive = upper[upper > 1e-10]
                bw = float(np.median(positive)) if positive.size > 0 else 1.0
        else:
            bw = float(bandwidth)
        bw = max(bw * float(bandwidth_scale), 1e-3)
        feature_map = cls(centers=centers, bandwidth=bw)
        if not standardize_features:
            return feature_map
        raw_features = feature_map._raw_transform(states_arr, actions_arr)
        mean, scale = cls._standardization_from_raw_features(raw_features)
        return cls(centers=centers, bandwidth=bw, feature_mean=mean, feature_scale=scale)

    @staticmethod
    def _standardization_from_raw_features(raw_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(raw_features, axis=0)
        scale = np.std(raw_features, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        # Keep the intercept as an actual intercept so the mean-one penalty
        # remains directly interpretable.
        mean[0] = 0.0
        scale[0] = 1.0
        return mean.astype(np.float64), scale.astype(np.float64)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return features
        return (features - self.feature_mean.reshape(1, -1)) / self.feature_scale.reshape(1, -1)

    @property
    def dimension(self) -> int:
        return int(7 + self.centers.shape[0])

    def _raw_transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr, actions_arr = _ensure_states_actions(states, actions)
        stacked = np.concatenate([states_arr, actions_arr], axis=1)
        base = np.concatenate(
            [
                np.ones((stacked.shape[0], 1), dtype=np.float64),
                stacked,
                stacked**2,
            ],
            axis=1,
        )
        if self.centers.shape[0] == 0:
            return base
        diffs = stacked[:, None, :] - self.centers[None, :, :]
        sq_norm = np.sum(diffs**2, axis=2)
        rbf = np.exp(-0.5 * sq_norm / (self.bandwidth**2))
        return np.concatenate([base, rbf], axis=1)

    def transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._standardize(self._raw_transform(states, actions))

    def _raw_expected_given_state(self, states: np.ndarray, policy: GaussianLinearPolicy) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        mean_action = policy.mean_action(states_arr)
        action_var = policy.conditional_action_variance()
        base = np.concatenate(
            [
                np.ones((states_arr.shape[0], 1), dtype=np.float64),
                states_arr,
                mean_action,
                states_arr**2,
                mean_action**2 + action_var,
            ],
            axis=1,
        )
        if self.centers.shape[0] == 0:
            return base
        bw2 = self.bandwidth**2
        state_diff = states_arr[:, None, :] - self.centers[None, :, :2]
        state_term = np.exp(-0.5 * np.sum(state_diff**2, axis=2) / bw2)
        action_diff = mean_action - self.centers[None, :, 2]
        action_term = np.exp(-0.5 * (action_diff**2) / (bw2 + action_var))
        prefactor = np.sqrt(bw2 / (bw2 + action_var))
        rbf = prefactor * state_term * action_term
        return np.concatenate([base, rbf], axis=1)

    def expected_given_state(self, states: np.ndarray, policy: GaussianLinearPolicy) -> np.ndarray:
        return self._standardize(self._raw_expected_given_state(states, policy))

    def expected_features_given_state(
        self,
        states: np.ndarray,
        policy: GaussianLinearPolicy,
    ) -> np.ndarray:
        return self.expected_given_state(states, policy)

    def _raw_expectation_under_initial_distribution(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        policy: GaussianLinearPolicy,
    ) -> np.ndarray:
        joint_mean, joint_cov = policy.joint_moments_from_state_gaussian(state_mean, state_cov)
        second_moment = joint_cov + np.outer(joint_mean, joint_mean)
        base = np.array(
            [
                1.0,
                joint_mean[0],
                joint_mean[1],
                joint_mean[2],
                second_moment[0, 0],
                second_moment[1, 1],
                second_moment[2, 2],
            ],
            dtype=np.float64,
        )
        if self.centers.shape[0] == 0:
            return base
        rbf = np.array(
            [
                _gaussian_rbf_expectation(joint_mean, joint_cov, center=center, bandwidth=self.bandwidth)
                for center in self.centers
            ],
            dtype=np.float64,
        )
        return np.concatenate([base, rbf], axis=0)

    def expectation_under_initial_distribution(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        policy: GaussianLinearPolicy,
    ) -> np.ndarray:
        return self.expectation_under_state_gaussian(state_mean, state_cov, policy)

    def expectation_under_state_gaussian(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        policy: GaussianLinearPolicy,
    ) -> np.ndarray:
        raw = self._raw_expectation_under_initial_distribution(state_mean, state_cov, policy)
        if self.feature_mean is None or self.feature_scale is None:
            return raw
        return (raw - self.feature_mean) / self.feature_scale

    def function_from_theta(self, theta: np.ndarray) -> RBFStateActionFunction:
        return RBFStateActionFunction(
            feature_map=self,
            theta=np.asarray(theta, dtype=np.float64).reshape(self.dimension),
        )
