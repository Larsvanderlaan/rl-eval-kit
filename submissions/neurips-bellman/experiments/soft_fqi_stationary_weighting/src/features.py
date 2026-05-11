from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_indices(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions, dtype=np.int64).reshape(-1)


def linear_q_features(states: np.ndarray, actions: np.ndarray, action_vectors: np.ndarray) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    action_idx = _as_indices(actions)
    avec = np.asarray(action_vectors, dtype=np.float64)[action_idx]
    x = states_arr[:, [0]]
    y = states_arr[:, [1]]
    ax = avec[:, [0]]
    ay = avec[:, [1]]
    goal = np.exp(-0.5 * ((x - 0.62) ** 2 + (y - 0.62) ** 2) / (0.28**2))
    decoy = np.exp(-0.5 * ((x + 0.55) ** 2 + (y - 0.45) ** 2) / (0.30**2))
    return np.concatenate(
        [
            np.ones_like(x),
            x,
            y,
            ax,
            ay,
            x**2,
            y**2,
            ax**2 + ay**2,
            goal,
            decoy,
        ],
        axis=1,
    )


def neural_q_features(states: np.ndarray, actions: np.ndarray, action_vectors: np.ndarray) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    action_idx = _as_indices(actions)
    action_one_hot = np.eye(action_vectors.shape[0], dtype=np.float64)[action_idx]
    avec = np.asarray(action_vectors, dtype=np.float64)[action_idx]
    return np.concatenate([states_arr, avec, action_one_hot], axis=1)


def _grid_state_action_centers(states: np.ndarray, action_vectors: np.ndarray, n_state_centers: int) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
    action_vecs = np.asarray(action_vectors, dtype=np.float64)
    side = max(int(np.sqrt(n_state_centers)), 1)
    x_centers = np.linspace(np.min(states_arr[:, 0]), np.max(states_arr[:, 0]), side)
    y_centers = np.linspace(np.min(states_arr[:, 1]), np.max(states_arr[:, 1]), side)
    state_centers = np.asarray(np.meshgrid(x_centers, y_centers, indexing="ij")).reshape(2, -1).T
    centers = []
    for state_center in state_centers:
        for action_vec in action_vecs:
            centers.append(np.concatenate([state_center, action_vec]))
    return np.asarray(centers, dtype=np.float64)


def _median_center_bandwidth(centers: np.ndarray, bandwidth_scale: float) -> float:
    if centers.shape[0] > 1:
        diffs = centers[:, None, :] - centers[None, :, :]
        distances = np.sqrt(np.sum(diffs * diffs, axis=2))
        upper = distances[np.triu_indices(centers.shape[0], k=1)]
        positive = upper[upper > 1e-10]
        bandwidth = float(np.median(positive)) if positive.size else 0.5
    else:
        bandwidth = 0.5
    return max(float(bandwidth_scale) * bandwidth, 1e-3)


@dataclass
class RichQFeatureMap:
    centers: np.ndarray
    bandwidth: float
    action_vectors: np.ndarray
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    @classmethod
    def from_grid(
        cls,
        states: np.ndarray,
        action_vectors: np.ndarray,
        *,
        n_state_centers: int = 36,
        bandwidth_scale: float = 0.65,
        standardize_features: bool = False,
    ) -> "RichQFeatureMap":
        centers = _grid_state_action_centers(states, action_vectors, n_state_centers)
        bandwidth = _median_center_bandwidth(centers, bandwidth_scale)
        action_vecs = np.asarray(action_vectors, dtype=np.float64)
        feature_map = cls(centers=centers, bandwidth=bandwidth, action_vectors=action_vecs)
        if not standardize_features:
            return feature_map
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        state_idx, action_idx = np.meshgrid(
            np.arange(states_arr.shape[0]),
            np.arange(action_vecs.shape[0]),
            indexing="ij",
        )
        raw = feature_map._raw_transform(states_arr[state_idx.reshape(-1)], action_idx.reshape(-1))
        mean, scale = RatioFeatureMap._standardization_from_raw_features(raw)
        return cls(
            centers=centers,
            bandwidth=bandwidth,
            action_vectors=action_vecs,
            feature_mean=mean,
            feature_scale=scale,
        )

    @property
    def dimension(self) -> int:
        return int(12 + self.centers.shape[0])

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return features
        return (features - self.feature_mean.reshape(1, -1)) / self.feature_scale.reshape(1, -1)

    def _raw_transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        action_idx = _as_indices(actions)
        avec = self.action_vectors[action_idx]
        x = states_arr[:, [0]]
        y = states_arr[:, [1]]
        ax = avec[:, [0]]
        ay = avec[:, [1]]
        goal = np.exp(-0.5 * ((x - 0.62) ** 2 + (y - 0.62) ** 2) / (0.28**2))
        decoy = np.exp(-0.5 * ((x + 0.55) ** 2 + (y - 0.45) ** 2) / (0.30**2))
        base = np.concatenate(
            [
                np.ones_like(x),
                x,
                y,
                ax,
                ay,
                x**2,
                y**2,
                ax**2 + ay**2,
                goal,
                decoy,
                x * ax + y * ay,
                x * y,
            ],
            axis=1,
        )
        if self.centers.shape[0] == 0:
            return base
        stacked = np.concatenate([states_arr, avec], axis=1)
        diff = stacked[:, None, :] - self.centers[None, :, :]
        sq_norm = np.sum(diff * diff, axis=2)
        rbf = np.exp(-0.5 * sq_norm / (self.bandwidth**2))
        return np.concatenate([base, rbf], axis=1)

    def transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._standardize(self._raw_transform(states, actions))


@dataclass
class QFeatureMap:
    kind: str
    action_vectors: np.ndarray
    rich_map: RichQFeatureMap | None = None

    @classmethod
    def from_grid(
        cls,
        kind: str,
        states: np.ndarray,
        action_vectors: np.ndarray,
        *,
        n_state_centers: int = 36,
        bandwidth_scale: float = 0.65,
        standardize_features: bool = False,
    ) -> "QFeatureMap":
        normalized = str(kind).lower()
        if normalized == "linear":
            return cls(kind="linear", action_vectors=np.asarray(action_vectors, dtype=np.float64))
        if normalized == "rich_rbf":
            rich_map = RichQFeatureMap.from_grid(
                states,
                action_vectors,
                n_state_centers=n_state_centers,
                bandwidth_scale=bandwidth_scale,
                standardize_features=standardize_features,
            )
            return cls(kind="rich_rbf", action_vectors=np.asarray(action_vectors, dtype=np.float64), rich_map=rich_map)
        raise ValueError(f"Unknown q feature class '{kind}'.")

    @property
    def dimension(self) -> int:
        if self.kind == "linear":
            return 10
        assert self.rich_map is not None
        return self.rich_map.dimension

    def transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            return linear_q_features(states, actions, self.action_vectors)
        assert self.rich_map is not None
        return self.rich_map.transform(states, actions)


@dataclass
class RatioFeatureMap:
    centers: np.ndarray
    bandwidth: float
    action_vectors: np.ndarray
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None

    @classmethod
    def from_grid(
        cls,
        states: np.ndarray,
        action_vectors: np.ndarray,
        *,
        n_state_centers: int = 16,
        bandwidth_scale: float = 1.0,
        standardize_features: bool = False,
    ) -> "RatioFeatureMap":
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        action_vecs = np.asarray(action_vectors, dtype=np.float64)
        centers_arr = _grid_state_action_centers(states_arr, action_vecs, n_state_centers)
        bandwidth = _median_center_bandwidth(centers_arr, bandwidth_scale)
        feature_map = cls(centers=centers_arr, bandwidth=bandwidth, action_vectors=action_vecs)
        if not standardize_features:
            return feature_map
        state_idx, action_idx = np.meshgrid(
            np.arange(states_arr.shape[0]),
            np.arange(action_vecs.shape[0]),
            indexing="ij",
        )
        raw_features = feature_map._raw_transform(states_arr[state_idx.reshape(-1)], action_idx.reshape(-1))
        mean, scale = cls._standardization_from_raw_features(raw_features)
        return cls(
            centers=centers_arr,
            bandwidth=bandwidth,
            action_vectors=action_vecs,
            feature_mean=mean,
            feature_scale=scale,
        )

    @classmethod
    def from_behavior_samples(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        action_vectors: np.ndarray,
        *,
        n_centers: int = 64,
        bandwidth_scale: float = 1.0,
        standardize_features: bool = False,
    ) -> "RatioFeatureMap":
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        action_idx = _as_indices(actions)
        action_vecs = np.asarray(action_vectors, dtype=np.float64)
        stacked = np.concatenate([states_arr, action_vecs[action_idx]], axis=1)
        if stacked.shape[0] == 0 or n_centers <= 0:
            centers_arr = np.zeros((0, 4), dtype=np.float64)
        elif stacked.shape[0] <= n_centers:
            centers_arr = stacked.copy()
        else:
            indices = np.linspace(0, stacked.shape[0] - 1, num=int(n_centers), dtype=np.int64)
            centers_arr = stacked[indices]
        if centers_arr.shape[0] > 1:
            diffs = centers_arr[:, None, :] - centers_arr[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            upper = distances[np.triu_indices(centers_arr.shape[0], k=1)]
            positive = upper[upper > 1e-10]
            bandwidth = float(np.median(positive)) if positive.size else 0.5
        else:
            bandwidth = 0.5
        bandwidth = max(float(bandwidth_scale) * bandwidth, 1e-3)
        feature_map = cls(centers=centers_arr, bandwidth=bandwidth, action_vectors=action_vecs)
        if not standardize_features:
            return feature_map
        raw_features = feature_map._raw_transform(states_arr, action_idx)
        mean, scale = cls._standardization_from_raw_features(raw_features)
        return cls(
            centers=centers_arr,
            bandwidth=bandwidth,
            action_vectors=action_vecs,
            feature_mean=mean,
            feature_scale=scale,
        )

    @property
    def dimension(self) -> int:
        return int(10 + self.centers.shape[0])

    @staticmethod
    def _standardization_from_raw_features(raw_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(raw_features, axis=0)
        scale = np.std(raw_features, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        mean[0] = 0.0
        scale[0] = 1.0
        return mean.astype(np.float64), scale.astype(np.float64)

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            return features
        return (features - self.feature_mean.reshape(1, -1)) / self.feature_scale.reshape(1, -1)

    def _raw_transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        action_idx = _as_indices(actions)
        avec = self.action_vectors[action_idx]
        x = states_arr[:, [0]]
        y = states_arr[:, [1]]
        ax = avec[:, [0]]
        ay = avec[:, [1]]
        base = np.concatenate(
            [
                np.ones_like(x),
                x,
                y,
                ax,
                ay,
                x**2,
                y**2,
                ax**2 + ay**2,
                x * ax + y * ay,
                x * y,
            ],
            axis=1,
        )
        if self.centers.shape[0] == 0:
            return base
        stacked = np.concatenate([states_arr, avec], axis=1)
        diff = stacked[:, None, :] - self.centers[None, :, :]
        sq_norm = np.sum(diff * diff, axis=2)
        rbf = np.exp(-0.5 * sq_norm / (self.bandwidth**2))
        return np.concatenate([base, rbf], axis=1)

    def transform(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._standardize(self._raw_transform(states, actions))

    def expected_under_policy(self, states: np.ndarray, policy: np.ndarray) -> np.ndarray:
        states_arr = np.asarray(states, dtype=np.float64).reshape(-1, 2)
        out = np.zeros((states_arr.shape[0], self.dimension), dtype=np.float64)
        action_grid = np.arange(self.action_vectors.shape[0], dtype=np.int64)
        for action_idx in action_grid:
            feats = self._raw_transform(states_arr, np.full(states_arr.shape[0], action_idx, dtype=np.int64))
            out += policy[:, [action_idx]] * feats
        return self._standardize(out)
