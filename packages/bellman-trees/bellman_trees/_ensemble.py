from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ._adapters import SklearnLeafAdapter, XGBoostLeafAdapter
from ._base import SerializableEstimatorMixin
from ._data import BellmanTransitionData
from ._features import average_next_features, hstack_csr
from ._native_tree import BellmanAggregationTree
from ._solver import solve_projected_bellman
from ._weights import stabilize_weights


Array = np.ndarray


class BellmanAggregationForest(SerializableEstimatorMixin):
    """Randomized honest BAT features with one final joint Bellman solve."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        n_estimators: int = 100,
        max_depth: int = 4,
        max_leaves: int = 16,
        max_bins: int = 32,
        min_samples_leaf: int = 20,
        min_weighted_leaf_mass: float = 1e-8,
        min_leaf_ess: float = 5.0,
        max_samples: float | int = 0.8,
        max_features: str | float | int | None = "sqrt",
        bootstrap: bool = True,
        complexity_penalty: float = 0.0,
        ridge: float = 1e-8,
        solver_method: str = "direct",
        solver_max_iter: int = 500,
        solver_tol: float = 1e-8,
        weight_clip_quantile: float | None = 0.995,
        max_weight: float | None = None,
        weight_uniform_mix: float = 0.0,
        target_ess_fraction: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_weighted_leaf_mass = min_weighted_leaf_mass
        self.min_leaf_ess = min_leaf_ess
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.complexity_penalty = complexity_penalty
        self.ridge = ridge
        self.solver_method = solver_method
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.weight_clip_quantile = weight_clip_quantile
        self.max_weight = max_weight
        self.weight_uniform_mix = weight_uniform_mix
        self.target_ess_fraction = target_ess_fraction
        self.random_state = random_state

    def fit(
        self,
        X: Array,
        reward: Array,
        X_next: Array,
        sample_weight: Array | None = None,
    ) -> "BellmanAggregationForest":
        data = BellmanTransitionData(X=X, reward=reward, X_next=X_next, sample_weight=sample_weight)
        weights = stabilize_weights(
            data.sample_weight,
            data.n_samples,
            max_weight=self.max_weight,
            clip_quantile=self.weight_clip_quantile,
            uniform_mix=self.weight_uniform_mix,
            target_ess_fraction=self.target_ess_fraction,
        )
        rng = np.random.default_rng(self.random_state)
        self.trees_: list[BellmanAggregationTree] = []
        self.tree_feature_indices_: list[np.ndarray] = []
        self.split_history_: list[dict[str, Any]] = []
        for b in range(int(self.n_estimators)):
            row_idx = self._sample_rows(data.n_samples, rng)
            feat_idx = self._sample_features(data.n_features, rng)
            tree = BellmanAggregationTree(
                gamma=float(self.gamma),
                max_depth=int(self.max_depth),
                max_leaves=int(self.max_leaves),
                max_bins=int(self.max_bins),
                min_samples_leaf=int(self.min_samples_leaf),
                min_weighted_leaf_mass=float(self.min_weighted_leaf_mass),
                min_leaf_ess=float(self.min_leaf_ess),
                complexity_penalty=float(self.complexity_penalty),
                ridge=float(self.ridge),
                honest=True,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                weight_clip_quantile=self.weight_clip_quantile,
                max_weight=self.max_weight,
                weight_uniform_mix=self.weight_uniform_mix,
                target_ess_fraction=self.target_ess_fraction,
            )
            tree.fit(
                data.X[row_idx][:, feat_idx],
                data.reward[row_idx],
                _subset_next_features(data.X_next[row_idx], feat_idx),
                sample_weight=weights.values[row_idx],
            )
            self.trees_.append(tree)
            self.tree_feature_indices_.append(feat_idx)
            self.split_history_.append({"tree": b, "splits": tree.split_history_})

        phi = self.transform(data.X)
        phi_next = self.transform_next(data.X_next)
        solve = solve_projected_bellman(
            phi,
            phi_next,
            data.reward,
            weights.values,
            gamma=float(self.gamma),
            ridge=float(self.ridge),
            method=self.solver_method,
            max_iter=int(self.solver_max_iter),
            tol=float(self.solver_tol),
        )
        self.theta_ = solve.theta
        self.solver_info_ = solve.diagnostics
        self.feature_info_ = {
            "n_trees": int(len(self.trees_)),
            "n_features": int(phi.shape[1]),
            "leaf_counts": [int(len(tree.leaf_node_ids_)) for tree in self.trees_],
            "joint_solve": True,
        }
        self.diagnostics_ = {
            **weights.diagnostics,
            "n_trees": int(len(self.trees_)),
            "n_features": int(phi.shape[1]),
            "joint_solve": True,
        }
        return self

    def predict(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        return np.asarray(self.transform(X_eval) @ self.theta_, dtype=np.float64).reshape(-1)

    def transform(self, X: Array) -> sparse.csr_matrix:
        if not hasattr(self, "trees_"):
            raise RuntimeError("Estimator is not fitted.")
        blocks = [
            tree.transform(np.asarray(X, dtype=np.float64)[:, feat_idx]) * (1.0 / max(len(self.trees_), 1))
            for tree, feat_idx in zip(self.trees_, self.tree_feature_indices_)
        ]
        return hstack_csr(blocks)

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        return average_next_features(self.transform, X_next)

    def _sample_rows(self, n: int, rng: np.random.Generator) -> Array:
        if isinstance(self.max_samples, float):
            m = max(2, int(round(float(self.max_samples) * n)))
        else:
            m = max(2, min(int(self.max_samples), n))
        return rng.integers(0, n, size=m, dtype=np.int64) if self.bootstrap else rng.choice(n, size=m, replace=False)

    def _sample_features(self, p: int, rng: np.random.Generator) -> Array:
        spec = self.max_features
        if spec is None:
            k = p
        elif spec == "sqrt":
            k = max(1, int(np.sqrt(p)))
        elif spec == "log2":
            k = max(1, int(np.log2(max(p, 2))))
        elif isinstance(spec, float):
            k = max(1, int(round(float(spec) * p)))
        else:
            k = max(1, min(int(spec), p))
        return np.sort(rng.choice(p, size=min(k, p), replace=False)).astype(np.int64)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "theta_"):
            raise RuntimeError("Estimator is not fitted.")


@dataclass
class BellmanLeafEnsembleRegressor(SerializableEstimatorMixin):
    """Projected Bellman solve on leaf features from an external tree ensemble."""

    feature_model: Any
    prefit: bool = False
    feature_target: str = "reward"
    gamma: float = 0.99
    ridge: float = 1e-8
    solver_method: str = "direct"
    solver_max_iter: int = 500
    solver_tol: float = 1e-8
    adapter: str = "auto"
    weight_clip_quantile: float | None = 0.995
    max_weight: float | None = None
    weight_uniform_mix: float = 0.0
    target_ess_fraction: float | None = None

    def fit(
        self,
        X: Array,
        reward: Array,
        X_next: Array,
        sample_weight: Array | None = None,
    ) -> "BellmanLeafEnsembleRegressor":
        data = BellmanTransitionData(X=X, reward=reward, X_next=X_next, sample_weight=sample_weight)
        weights = stabilize_weights(
            data.sample_weight,
            data.n_samples,
            max_weight=self.max_weight,
            clip_quantile=self.weight_clip_quantile,
            uniform_mix=self.weight_uniform_mix,
            target_ess_fraction=self.target_ess_fraction,
        )
        target = self._feature_target(data)
        model = self.feature_model if self.prefit else deepcopy(self.feature_model)
        self.adapter_ = self._make_adapter(model)
        self.adapter_.fit(data.X, None if self.prefit else target, sample_weight=weights.values)
        phi = self.adapter_.transform(data.X)
        phi_next = self.adapter_.transform_next(data.X_next)
        solve = solve_projected_bellman(
            phi,
            phi_next,
            data.reward,
            weights.values,
            gamma=float(self.gamma),
            ridge=float(self.ridge),
            method=self.solver_method,
            max_iter=int(self.solver_max_iter),
            tol=float(self.solver_tol),
        )
        self.theta_ = solve.theta
        self.model_ = model
        self.solver_info_ = solve.diagnostics
        self.feature_info_ = {
            "n_features": int(phi.shape[1]),
            "adapter": type(self.adapter_).__name__,
            "feature_target": self.feature_target,
            "prefit": bool(self.prefit),
        }
        self.diagnostics_ = {**weights.diagnostics, **self.feature_info_}
        self.split_history_ = []
        return self

    def predict(self, X_eval: Array) -> Array:
        if not hasattr(self, "theta_"):
            raise RuntimeError("Estimator is not fitted.")
        return np.asarray(self.adapter_.transform(X_eval) @ self.theta_, dtype=np.float64).reshape(-1)

    def transform(self, X: Array) -> sparse.csr_matrix:
        if not hasattr(self, "adapter_"):
            raise RuntimeError("Estimator is not fitted.")
        return self.adapter_.transform(X)

    def _feature_target(self, data: BellmanTransitionData) -> Array:
        if self.feature_target == "reward":
            return data.reward
        if self.feature_target == "zero":
            return np.zeros(data.n_samples, dtype=np.float64)
        raise ValueError("feature_target must be 'reward' or 'zero'.")

    def _make_adapter(self, model: Any) -> SklearnLeafAdapter | XGBoostLeafAdapter:
        if self.adapter == "sklearn":
            return SklearnLeafAdapter(model)
        if self.adapter == "xgboost":
            return XGBoostLeafAdapter(model)
        module = type(model).__module__.lower()
        if "xgboost" in module:
            return XGBoostLeafAdapter(model)
        return SklearnLeafAdapter(model)


def _subset_next_features(X_next: Array, feat_idx: Array) -> Array:
    xp = np.asarray(X_next, dtype=np.float64)
    if xp.ndim == 2:
        return xp[:, feat_idx]
    return xp[:, :, feat_idx]
