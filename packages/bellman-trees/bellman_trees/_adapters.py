from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ._features import average_next_features, leaf_matrix_from_columns


Array = np.ndarray


@dataclass
class SklearnLeafAdapter:
    """Convert sklearn tree ensembles into sparse leaf-feature matrices."""

    model: Any
    n_prediction_bins: int = 32
    column_maps_: list[dict[int, int]] | None = None
    fallback_thresholds_: list[np.ndarray] | None = None

    def fit(self, X: Array, y: Array | None = None, sample_weight: Array | None = None) -> "SklearnLeafAdapter":
        if y is not None and not hasattr(self.model, "tree_") and not hasattr(self.model, "estimators_"):
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                self.model.fit(X, y)
        elif y is not None and not _is_fitted_tree_model(self.model):
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                self.model.fit(X, y)
        leaves = self.apply(X)
        _, self.column_maps_ = leaf_matrix_from_columns(leaves)
        return self

    def apply(self, X: Array) -> Array:
        x = np.asarray(X, dtype=np.float64)
        if hasattr(self.model, "apply"):
            leaves = self.model.apply(x)
            arr = np.asarray(leaves)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            return arr.astype(np.int64, copy=False)
        # HistGradientBoostingRegressor has no stable public leaf API. We use
        # staged prediction bins as deterministic public feature columns.
        if hasattr(self.model, "staged_predict"):
            staged = np.column_stack([pred for pred in self.model.staged_predict(x)])
            return self._prediction_bins(staged)
        if hasattr(self.model, "predict"):
            pred = np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1, 1)
            return self._prediction_bins(pred)
        raise TypeError("model must expose apply, staged_predict, or predict.")

    def transform(self, X: Array) -> sparse.csr_matrix:
        if self.column_maps_ is None:
            raise RuntimeError("Adapter is not fitted.")
        matrix, _ = leaf_matrix_from_columns(self.apply(X), column_maps=self.column_maps_)
        return matrix

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        return average_next_features(self.transform, X_next)

    def _prediction_bins(self, values: Array) -> Array:
        vals = np.asarray(values, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if self.fallback_thresholds_ is None:
            self.fallback_thresholds_ = []
            for j in range(vals.shape[1]):
                col = vals[:, j]
                probs = np.linspace(0.0, 1.0, int(self.n_prediction_bins) + 1)[1:-1]
                thresholds = np.unique(np.quantile(col, probs)) if col.size else np.empty(0)
                self.fallback_thresholds_.append(thresholds)
        bins = np.zeros(vals.shape, dtype=np.int64)
        for j, thresholds in enumerate(self.fallback_thresholds_):
            bins[:, j] = np.digitize(vals[:, j], thresholds, right=False)
        return bins


@dataclass
class XGBoostLeafAdapter:
    """Optional adapter for XGBoost models with leaf-index prediction."""

    model: Any
    column_maps_: list[dict[int, int]] | None = None

    def fit(self, X: Array, y: Array | None = None, sample_weight: Array | None = None) -> "XGBoostLeafAdapter":
        if y is not None and not _is_xgboost_fitted(self.model):
            try:
                self.model.fit(X, y, sample_weight=sample_weight)
            except TypeError:
                self.model.fit(X, y)
        leaves = self.apply(X)
        _, self.column_maps_ = leaf_matrix_from_columns(leaves)
        return self

    def apply(self, X: Array) -> Array:
        if hasattr(self.model, "apply"):
            leaves = self.model.apply(X)
        elif hasattr(self.model, "predict"):
            leaves = self.model.predict(X, pred_leaf=True)
        else:
            raise TypeError("XGBoost model must expose apply or predict(..., pred_leaf=True).")
        arr = np.asarray(leaves)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr.astype(np.int64, copy=False)

    def transform(self, X: Array) -> sparse.csr_matrix:
        if self.column_maps_ is None:
            raise RuntimeError("Adapter is not fitted.")
        matrix, _ = leaf_matrix_from_columns(self.apply(X), column_maps=self.column_maps_)
        return matrix

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        return average_next_features(self.transform, X_next)


def _is_fitted_tree_model(model: Any) -> bool:
    return any(hasattr(model, attr) for attr in ("tree_", "estimators_", "_predictors"))


def _is_xgboost_fitted(model: Any) -> bool:
    try:
        booster = model.get_booster()
    except Exception:
        return False
    return booster is not None
