from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class ContinuousFeatureEncoder:
    """Feature map for continuous state-action ratio estimation."""

    def __init__(
        self,
        stats: StandardizationStats,
        *,
        include_quadratic: bool = False,
        include_cross_terms: bool = False,
    ) -> None:
        self.stats = stats
        self.include_quadratic = bool(include_quadratic)
        self.include_cross_terms = bool(include_cross_terms)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        eps: float = 1e-6,
        include_quadratic: bool = False,
        include_cross_terms: bool = False,
    ) -> "ContinuousFeatureEncoder":
        observation_mean = np.asarray(observations, dtype=np.float64).mean(axis=0)
        action_mean = np.asarray(actions, dtype=np.float64).mean(axis=0)
        observation_std = np.asarray(observations, dtype=np.float64).std(axis=0) + eps
        action_std = np.asarray(actions, dtype=np.float64).std(axis=0) + eps
        stats = StandardizationStats(
            observation_mean=observation_mean,
            observation_std=observation_std,
            action_mean=action_mean,
            action_std=action_std,
        )
        return cls(
            stats,
            include_quadratic=include_quadratic,
            include_cross_terms=include_cross_terms,
        )

    def _normalize_observations(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        return (obs - self.stats.observation_mean) / self.stats.observation_std

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        act = np.asarray(actions, dtype=np.float64)
        return (act - self.stats.action_mean) / self.stats.action_std

    def transform(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64)
        act = np.asarray(actions, dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        if act.ndim == 1:
            act = act[None, :]
        if obs.shape[0] != act.shape[0]:
            raise ValueError("Observations and actions must have the same number of rows.")

        obs_norm = self._normalize_observations(obs)
        act_norm = self._normalize_actions(act)
        features = [
            np.ones((obs_norm.shape[0], 1), dtype=np.float64),
            obs_norm,
            act_norm,
        ]
        if self.include_quadratic:
            features.extend([obs_norm**2, act_norm**2])
        if self.include_cross_terms:
            features.append((obs_norm[:, :, None] * act_norm[:, None, :]).reshape(obs_norm.shape[0], -1))
        return np.concatenate(features, axis=1).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        obs_dim = int(self.stats.observation_mean.shape[0])
        act_dim = int(self.stats.action_mean.shape[0])
        dim = 1 + obs_dim + act_dim
        if self.include_quadratic:
            dim += obs_dim + act_dim
        if self.include_cross_terms:
            dim += obs_dim * act_dim
        return dim
