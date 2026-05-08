from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ._base import SerializableEstimatorMixin
from ._features import average_next_features, leaf_matrix_from_columns
from ._hist_gbt_fast import HAS_NUMBA, apply_tree_numba
from ._solver import solve_projected_bellman
from ._streaming_solver import StreamingFeatureBatch, solve_streaming_bellman
from ._weights import stabilize_weights


Array = np.ndarray


@dataclass(frozen=True)
class QuantileBinner:
    """Column-wise quantile thresholds for compact histogram bins."""

    thresholds: list[Array]
    max_bins: int
    n_bins_per_feature: Array
    bin_dtype: Any


@dataclass(frozen=True)
class _Split:
    feature_index: int
    threshold_bin: int
    default_left: bool
    gain: float
    left_rows: Array
    right_rows: Array


@dataclass
class _HistogramTree:
    children_left: Array
    children_right: Array
    feature_index: Array
    threshold_bin: Array
    default_left: Array
    leaf_value: Array
    leaf_index: Array
    gain: Array
    depth: Array
    n_node_samples: Array
    sum_hessian: Array

    @property
    def n_leaves(self) -> int:
        return int(np.sum(self.children_left < 0))


def _as_2d_allow_nan(name: str, value: Array) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values.")
    return np.ascontiguousarray(arr)


def _as_next_allow_nan(value: Array, n: int, p: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape != (n, p):
            raise ValueError(f"X_next must have shape {(n, p)} or (n, m, {p}), got {arr.shape}.")
        out = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[0] != n or arr.shape[2] != p:
            raise ValueError(f"X_next must have shape {(n, p)} or (n, m, {p}), got {arr.shape}.")
        out = np.ascontiguousarray(arr)
    else:
        raise ValueError(f"X_next must be 2D or 3D, got shape {arr.shape}.")
    if np.any(np.isinf(out)):
        raise ValueError("X_next contains infinite values.")
    return out


def _as_1d_finite(name: str, value: Array, n: int) -> Array:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)


def _validate_fraction(name: str, value: float) -> float:
    out = float(value)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return out


def _fit_quantile_binner(X: Array, max_bins: int = 256) -> QuantileBinner:
    """Fit deterministic column-wise quantile bins, ignoring NaNs."""

    x = _as_2d_allow_nan("X", X)
    bins = int(max_bins)
    if bins < 2:
        raise ValueError("max_bins must be at least 2.")
    thresholds: list[Array] = []
    counts: list[int] = []
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        if finite.size <= 1:
            thr = np.empty(0, dtype=np.float64)
        else:
            unique = np.unique(finite)
            if unique.size <= 1:
                thr = np.empty(0, dtype=np.float64)
            elif unique.size <= bins:
                thr = 0.5 * (unique[:-1] + unique[1:])
            else:
                n_thresholds = min(bins - 1, unique.size - 1)
                probs = np.linspace(0.0, 1.0, n_thresholds + 2, dtype=np.float64)[1:-1]
                thr = np.unique(np.quantile(finite, probs, method="linear")).astype(np.float64)
                if thr.size:
                    lo = float(np.min(finite))
                    hi = float(np.max(finite))
                    thr = thr[(thr > lo) & (thr < hi)]
        thresholds.append(np.ascontiguousarray(thr, dtype=np.float64))
        counts.append(int(thr.size + 1))
    dtype = np.uint16 if bins <= np.iinfo(np.uint16).max else np.uint32
    return QuantileBinner(
        thresholds=thresholds,
        max_bins=bins,
        n_bins_per_feature=np.asarray(counts, dtype=np.int32),
        bin_dtype=dtype,
    )


def _transform_bins(X: Array, binner: QuantileBinner) -> tuple[Array, Array]:
    x = _as_2d_allow_nan("X", X)
    if x.shape[1] != len(binner.thresholds):
        raise ValueError(f"X has {x.shape[1]} columns, expected {len(binner.thresholds)}.")
    out = np.zeros(x.shape, dtype=binner.bin_dtype)
    missing = ~np.isfinite(x)
    for j, thr in enumerate(binner.thresholds):
        keep = ~missing[:, j]
        if np.any(keep):
            out[keep, j] = np.searchsorted(thr, x[keep, j], side="right").astype(binner.bin_dtype, copy=False)
    return np.ascontiguousarray(out), np.ascontiguousarray(missing)


def _xgb_split_gain(
    left_grad: Array | float,
    left_hess: Array | float,
    right_grad: Array | float,
    right_hess: Array | float,
    parent_grad: float,
    parent_hess: float,
    *,
    l2_leaf_reg: float,
    split_gamma: float,
) -> Array | float:
    reg = float(l2_leaf_reg)
    gain = 0.5 * (
        np.asarray(left_grad) ** 2 / (np.asarray(left_hess) + reg)
        + np.asarray(right_grad) ** 2 / (np.asarray(right_hess) + reg)
        - float(parent_grad) ** 2 / (float(parent_hess) + reg)
    ) - float(split_gamma)
    return gain


def _leaf_update(grad_sum: float, hess_sum: float, *, l2_leaf_reg: float, learning_rate: float) -> float:
    return float(-float(learning_rate) * grad_sum / (hess_sum + float(l2_leaf_reg)))


def _splitmix64(values: Array) -> Array:
    x = np.asarray(values, dtype=np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _sample_feature_indices(
    features: Array,
    colsample_bynode: float,
    rng: np.random.Generator,
) -> Array:
    frac = _validate_fraction("colsample_bynode", colsample_bynode)
    if frac >= 1.0 or features.size <= 1:
        return features
    k = max(1, int(round(frac * features.size)))
    return np.sort(rng.choice(features, size=k, replace=False)).astype(np.int64)


def _find_best_split(
    *,
    bins: Array,
    missing: Array,
    binner: QuantileBinner,
    grad: Array,
    hess: Array,
    rows: Array,
    feature_indices: Array,
    min_samples_leaf: int,
    min_child_weight: float,
    l2_leaf_reg: float,
    split_gamma: float,
) -> _Split | None:
    if rows.size < 2 * int(min_samples_leaf):
        return None
    g_rows = grad[rows]
    h_rows = hess[rows]
    parent_grad = float(np.sum(g_rows))
    parent_hess = float(np.sum(h_rows))
    parent_count = int(rows.size)
    if parent_hess < 2.0 * float(min_child_weight):
        return None

    best_feature = -1
    best_threshold = -1
    best_default_left = True
    best_gain = -np.inf

    for feature in feature_indices:
        j = int(feature)
        n_bins = int(binner.n_bins_per_feature[j])
        if n_bins <= 1:
            continue

        row_missing = missing[rows, j]
        non_missing = ~row_missing
        if not np.any(non_missing):
            continue

        row_bins = bins[rows[non_missing], j].astype(np.int64, copy=False)
        g_non = g_rows[non_missing]
        h_non = h_rows[non_missing]
        bin_grad = np.bincount(row_bins, weights=g_non, minlength=n_bins).astype(np.float64, copy=False)
        bin_hess = np.bincount(row_bins, weights=h_non, minlength=n_bins).astype(np.float64, copy=False)
        bin_count = np.bincount(row_bins, minlength=n_bins).astype(np.int64, copy=False)

        missing_grad = float(np.sum(g_rows[row_missing]))
        missing_hess = float(np.sum(h_rows[row_missing]))
        missing_count = int(np.sum(row_missing))

        left_grad_non = np.cumsum(bin_grad)[:-1]
        left_hess_non = np.cumsum(bin_hess)[:-1]
        left_count_non = np.cumsum(bin_count)[:-1]
        total_non_grad = float(np.sum(bin_grad))
        total_non_hess = float(np.sum(bin_hess))
        total_non_count = int(np.sum(bin_count))
        right_grad_non = total_non_grad - left_grad_non
        right_hess_non = total_non_hess - left_hess_non
        right_count_non = total_non_count - left_count_non

        for default_left in (False, True):
            if default_left:
                left_grad = left_grad_non + missing_grad
                left_hess = left_hess_non + missing_hess
                left_count = left_count_non + missing_count
                right_grad = right_grad_non
                right_hess = right_hess_non
                right_count = right_count_non
            else:
                left_grad = left_grad_non
                left_hess = left_hess_non
                left_count = left_count_non
                right_grad = right_grad_non + missing_grad
                right_hess = right_hess_non + missing_hess
                right_count = right_count_non + missing_count

            valid = (
                (left_count >= int(min_samples_leaf))
                & (right_count >= int(min_samples_leaf))
                & (left_hess >= float(min_child_weight))
                & (right_hess >= float(min_child_weight))
            )
            if not np.any(valid):
                continue

            gains = _xgb_split_gain(
                left_grad,
                left_hess,
                right_grad,
                right_hess,
                parent_grad,
                parent_hess,
                l2_leaf_reg=float(l2_leaf_reg),
                split_gamma=float(split_gamma),
            )
            gains = np.where(valid, gains, -np.inf)
            pos = int(np.argmax(gains))
            gain = float(gains[pos])
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = pos
                best_default_left = bool(default_left)

    if best_feature < 0 or not np.isfinite(best_gain) or best_gain <= 0.0:
        return None

    row_missing = missing[rows, best_feature]
    goes_left = bins[rows, best_feature].astype(np.int64, copy=False) <= int(best_threshold)
    if best_default_left:
        goes_left = goes_left | row_missing
    else:
        goes_left = goes_left & ~row_missing
    left_rows = rows[goes_left]
    right_rows = rows[~goes_left]
    if left_rows.size < int(min_samples_leaf) or right_rows.size < int(min_samples_leaf):
        return None
    return _Split(
        feature_index=int(best_feature),
        threshold_bin=int(best_threshold),
        default_left=bool(best_default_left),
        gain=float(best_gain),
        left_rows=np.ascontiguousarray(left_rows, dtype=np.int64),
        right_rows=np.ascontiguousarray(right_rows, dtype=np.int64),
    )


def _fit_histogram_tree(
    *,
    bins: Array,
    missing: Array,
    binner: QuantileBinner,
    grad: Array,
    hess: Array,
    rows: Array,
    feature_indices: Array,
    max_depth: int,
    max_leaves: int,
    min_samples_leaf: int,
    min_child_weight: float,
    l2_leaf_reg: float,
    split_gamma: float,
    learning_rate: float,
    colsample_bynode: float,
    rng: np.random.Generator,
) -> tuple[_HistogramTree, list[dict[str, Any]]]:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("cannot fit a histogram tree with zero rows.")
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    if feature_indices.size == 0:
        raise ValueError("at least one feature is required.")

    children_left: list[int] = [-1]
    children_right: list[int] = [-1]
    split_feature: list[int] = [-1]
    threshold_bin: list[int] = [-1]
    default_left: list[bool] = [True]
    leaf_value: list[float] = [
        _leaf_update(
            float(np.sum(grad[rows])),
            float(np.sum(hess[rows])),
            l2_leaf_reg=float(l2_leaf_reg),
            learning_rate=float(learning_rate),
        )
    ]
    gain: list[float] = [0.0]
    depth: list[int] = [0]
    n_node_samples: list[int] = [int(rows.size)]
    sum_hessian: list[float] = [float(np.sum(hess[rows]))]
    node_rows: dict[int, Array] = {0: rows}
    leaves: set[int] = {0}
    history: list[dict[str, Any]] = []

    while len(leaves) < int(max_leaves):
        best_node = -1
        best_split: _Split | None = None
        best_gain = -np.inf
        for node in sorted(leaves):
            if depth[node] >= int(max_depth):
                continue
            node_features = _sample_feature_indices(feature_indices, colsample_bynode, rng)
            split = _find_best_split(
                bins=bins,
                missing=missing,
                binner=binner,
                grad=grad,
                hess=hess,
                rows=node_rows[node],
                feature_indices=node_features,
                min_samples_leaf=int(min_samples_leaf),
                min_child_weight=float(min_child_weight),
                l2_leaf_reg=float(l2_leaf_reg),
                split_gamma=float(split_gamma),
            )
            if split is not None and split.gain > best_gain:
                best_node = node
                best_split = split
                best_gain = float(split.gain)
        if best_node < 0 or best_split is None:
            break

        left_id = len(children_left)
        right_id = left_id + 1
        children_left[best_node] = left_id
        children_right[best_node] = right_id
        split_feature[best_node] = int(best_split.feature_index)
        threshold_bin[best_node] = int(best_split.threshold_bin)
        default_left[best_node] = bool(best_split.default_left)
        gain[best_node] = float(best_split.gain)

        for child_rows in (best_split.left_rows, best_split.right_rows):
            children_left.append(-1)
            children_right.append(-1)
            split_feature.append(-1)
            threshold_bin.append(-1)
            default_left.append(True)
            child_grad = float(np.sum(grad[child_rows]))
            child_hess = float(np.sum(hess[child_rows]))
            leaf_value.append(
                _leaf_update(
                    child_grad,
                    child_hess,
                    l2_leaf_reg=float(l2_leaf_reg),
                    learning_rate=float(learning_rate),
                )
            )
            gain.append(0.0)
            depth.append(depth[best_node] + 1)
            n_node_samples.append(int(child_rows.size))
            sum_hessian.append(float(child_hess))

        leaves.remove(best_node)
        leaves.add(left_id)
        leaves.add(right_id)
        node_rows[left_id] = best_split.left_rows
        node_rows[right_id] = best_split.right_rows
        del node_rows[best_node]
        history.append(
            {
                "node": int(best_node),
                "depth": int(depth[best_node]),
                "feature": int(best_split.feature_index),
                "threshold_bin": int(best_split.threshold_bin),
                "default_left": bool(best_split.default_left),
                "gain": float(best_split.gain),
                "left_samples": int(best_split.left_rows.size),
                "right_samples": int(best_split.right_rows.size),
            }
        )

    leaf_index = np.full(len(children_left), -1, dtype=np.int32)
    for pos, node in enumerate(sorted(leaves)):
        leaf_index[node] = int(pos)

    tree = _HistogramTree(
        children_left=np.asarray(children_left, dtype=np.int32),
        children_right=np.asarray(children_right, dtype=np.int32),
        feature_index=np.asarray(split_feature, dtype=np.int32),
        threshold_bin=np.asarray(threshold_bin, dtype=np.int32),
        default_left=np.asarray(default_left, dtype=bool),
        leaf_value=np.asarray(leaf_value, dtype=np.float64),
        leaf_index=leaf_index,
        gain=np.asarray(gain, dtype=np.float64),
        depth=np.asarray(depth, dtype=np.int32),
        n_node_samples=np.asarray(n_node_samples, dtype=np.int64),
        sum_hessian=np.asarray(sum_hessian, dtype=np.float64),
    )
    return tree, history


def _apply_one_tree(tree: _HistogramTree, bins: Array, missing: Array, *, use_numba: bool = False) -> tuple[Array, Array]:
    if use_numba:
        out = apply_tree_numba(
            tree.children_left,
            tree.children_right,
            tree.feature_index,
            tree.threshold_bin,
            tree.default_left,
            tree.leaf_index,
            tree.leaf_value,
            bins,
            missing,
        )
        if out is not None:
            return out
    n = bins.shape[0]
    node = np.zeros(n, dtype=np.int32)
    while True:
        internal = tree.children_left[node] >= 0
        if not np.any(internal):
            break
        rows = np.nonzero(internal)[0]
        current = node[rows]
        feature = tree.feature_index[current]
        threshold = tree.threshold_bin[current]
        is_missing = missing[rows, feature]
        finite_left = bins[rows, feature].astype(np.int32, copy=False) <= threshold
        go_left = np.where(is_missing, tree.default_left[current], finite_left)
        node[rows] = np.where(go_left, tree.children_left[current], tree.children_right[current]).astype(np.int32, copy=False)
    return tree.leaf_index[node].astype(np.int32, copy=False), tree.leaf_value[node].astype(np.float64, copy=False)


class BellmanHistogramGradientBoostingRegressor(SerializableEstimatorMixin):
    """XGBoost-style histogram GBT leaf features followed by a Bellman solve.

    The boosted trees are trained on the immediate reward as a fast,
    non-iterative feature-construction target. Final predictions are not the
    boosted raw predictions: they are produced by solving the weighted
    projected Bellman equation on the sparse boosted-leaf feature map.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        max_leaves: int | None = None,
        max_bins: int = 256,
        min_samples_leaf: int = 20,
        min_child_weight: float = 1.0,
        l2_leaf_reg: float = 1.0,
        split_gamma: float = 0.0,
        subsample: float = 1.0,
        max_samples_per_tree: int | None = None,
        colsample_bytree: float = 1.0,
        colsample_bynode: float = 1.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        solver_method: str = "auto",
        solver_max_iter: int = 500,
        solver_tol: float = 1e-8,
        ridge: float = 1e-8,
        weight_clip_quantile: float | None = 0.995,
        max_weight: float | None = None,
        weight_uniform_mix: float = 0.0,
        target_ess_fraction: float | None = None,
        backend: str = "auto",
        feature_storage: str = "auto",
        hash_dim: int = 65_536,
        batch_size: int = 65_536,
        dtype: str = "float32",
        accumulator_dtype: str = "float64",
        max_exact_features: int = 8192,
        random_state: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.l2_leaf_reg = l2_leaf_reg
        self.split_gamma = split_gamma
        self.subsample = subsample
        self.max_samples_per_tree = max_samples_per_tree
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.solver_method = solver_method
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.ridge = ridge
        self.weight_clip_quantile = weight_clip_quantile
        self.max_weight = max_weight
        self.weight_uniform_mix = weight_uniform_mix
        self.target_ess_fraction = target_ess_fraction
        self.backend = backend
        self.feature_storage = feature_storage
        self.hash_dim = hash_dim
        self.batch_size = batch_size
        self.dtype = dtype
        self.accumulator_dtype = accumulator_dtype
        self.max_exact_features = max_exact_features
        self.random_state = random_state

    def fit(
        self,
        X: Array,
        reward: Array,
        X_next: Array,
        sample_weight: Array | None = None,
    ) -> "BellmanHistogramGradientBoostingRegressor":
        start = time.perf_counter()
        x = _as_2d_allow_nan("X", X)
        y = _as_1d_finite("reward", reward, x.shape[0])
        xp = _as_next_allow_nan(X_next, x.shape[0], x.shape[1])
        raw_weight = None if sample_weight is None else _as_1d_finite("sample_weight", sample_weight, x.shape[0])
        if raw_weight is not None and np.any(raw_weight < 0.0):
            raise ValueError("sample_weight must be nonnegative.")
        weights = stabilize_weights(
            raw_weight,
            x.shape[0],
            max_weight=self.max_weight,
            clip_quantile=self.weight_clip_quantile,
            uniform_mix=self.weight_uniform_mix,
            target_ess_fraction=self.target_ess_fraction,
        )
        self._validate_parameters(x.shape[0], x.shape[1])
        self.backend_ = self._resolve_backend()
        rng = np.random.default_rng(self.random_state)

        train_idx, val_idx = self._train_validation_split(x.shape[0], rng)
        bin_start = time.perf_counter()
        self.binner_ = _fit_quantile_binner(x[train_idx], int(self.max_bins))
        bins, missing = _transform_bins(x, self.binner_)
        binning_time = time.perf_counter() - bin_start

        base_score = float(np.sum(weights.values[train_idx] * y[train_idx]) / np.sum(weights.values[train_idx]))
        raw_pred = np.full(x.shape[0], base_score, dtype=np.float64)
        self.base_score_ = base_score
        self.trees_: list[_HistogramTree] = []
        self.split_history_: list[dict[str, Any]] = []
        train_loss: list[float] = []
        validation_loss: list[float] = []
        best_iteration = -1
        best_loss = np.inf
        rounds_since_improvement = 0

        train_start = time.perf_counter()
        for iteration in range(int(self.n_estimators)):
            row_idx = self._sample_rows(train_idx, rng)
            tree_features = self._sample_features(x.shape[1], rng, self.colsample_bytree)
            grad = weights.values * (raw_pred - y)
            hess = weights.values.copy()
            tree, history = _fit_histogram_tree(
                bins=bins,
                missing=missing,
                binner=self.binner_,
                grad=grad,
                hess=hess,
                rows=row_idx,
                feature_indices=tree_features,
                max_depth=int(self.max_depth),
                max_leaves=int(self.max_leaves or 2 ** int(self.max_depth)),
                min_samples_leaf=int(self.min_samples_leaf),
                min_child_weight=float(self.min_child_weight),
                l2_leaf_reg=float(self.l2_leaf_reg),
                split_gamma=float(self.split_gamma),
                learning_rate=float(self.learning_rate),
                colsample_bynode=float(self.colsample_bynode),
                rng=rng,
            )
            _, update = _apply_one_tree(tree, bins, missing, use_numba=self.backend_ == "numba")
            raw_pred += update
            self.trees_.append(tree)
            self.split_history_.append({"tree": int(iteration), "splits": history})

            train_loss.append(self._weighted_mse(y[train_idx], raw_pred[train_idx], weights.values[train_idx]))
            if val_idx.size:
                current_val_loss = self._weighted_mse(y[val_idx], raw_pred[val_idx], weights.values[val_idx])
                validation_loss.append(current_val_loss)
                if current_val_loss + 1e-12 < best_loss:
                    best_loss = current_val_loss
                    best_iteration = iteration
                    rounds_since_improvement = 0
                else:
                    rounds_since_improvement += 1
                if self.early_stopping_rounds is not None and rounds_since_improvement >= int(self.early_stopping_rounds):
                    break
            else:
                best_iteration = iteration
        training_time = time.perf_counter() - train_start

        if val_idx.size and best_iteration >= 0 and best_iteration + 1 < len(self.trees_):
            self.trees_ = self.trees_[: best_iteration + 1]
            self.split_history_ = self.split_history_[: best_iteration + 1]
            train_loss = train_loss[: best_iteration + 1]
            validation_loss = validation_loss[: best_iteration + 1]
        if not self.trees_:
            raise RuntimeError("boosting produced no trees.")

        self._configure_feature_layout(x.shape[0])
        feature_start = time.perf_counter()
        if self.feature_storage_ == "csr" and str(self.solver_method) not in {
            "streaming_direct",
            "streaming_fqe",
            "streaming_lstd_iterative",
        }:
            phi = self.transform(x)
            phi_next = self.transform_next(xp)
            feature_build_time = time.perf_counter() - feature_start
            solve_start = time.perf_counter()
            solve = solve_projected_bellman(
                phi,
                phi_next,
                y,
                weights.values,
                gamma=float(self.gamma),
                ridge=float(self.ridge),
                method=self.solver_method,
                max_iter=int(self.solver_max_iter),
                tol=float(self.solver_tol),
            )
            n_solver_features = int(phi.shape[1])
        else:
            next_bins, next_missing = self._transform_next_bins(xp)
            feature_build_time = time.perf_counter() - feature_start
            solve_start = time.perf_counter()
            method = self._resolve_streaming_solver_method()
            solve = solve_streaming_bellman(
                lambda: self._iter_streaming_batches(
                    bins=bins,
                    missing=missing,
                    next_bins=next_bins,
                    next_missing=next_missing,
                    reward=y,
                    weights=weights.values,
                ),
                n_features=int(self.n_solver_features_),
                gamma=float(self.gamma),
                ridge=float(self.ridge),
                method=method,
                max_iter=int(self.solver_max_iter),
                tol=float(self.solver_tol),
            )
            n_solver_features = int(self.n_solver_features_)
        solve_time = time.perf_counter() - solve_start

        self.theta_ = solve.theta
        self.solver_info_ = solve.diagnostics
        self.feature_info_ = {
            "n_trees": int(len(self.trees_)),
            "n_features": int(n_solver_features),
            "n_features_raw": int(self.raw_leaf_features_),
            "n_features_solver": int(n_solver_features),
            "leaf_counts": [int(tree.n_leaves) for tree in self.trees_],
            "feature_scale": float(1.0 / max(len(self.trees_), 1)),
            "feature_target": "reward",
            "feature_storage": self.feature_storage_,
            "hash_dim": None if self.feature_storage_ != "hashed" else int(self.hash_dim),
            "hash_load_factor": None
            if self.feature_storage_ != "hashed"
            else float(self.raw_leaf_features_ / max(int(self.hash_dim), 1)),
            "joint_bellman_solve": True,
        }
        self.raw_training_loss_ = train_loss
        self.raw_validation_loss_ = validation_loss
        self.diagnostics_ = {
            **weights.diagnostics,
            **self.feature_info_,
            "best_iteration": int(best_iteration if best_iteration >= 0 else len(self.trees_) - 1),
            "binning_time": float(binning_time),
            "boosting_time": float(training_time),
            "feature_build_time": float(feature_build_time),
            "leaf_apply_time": float(solve.diagnostics.get("leaf_apply_time", 0.0)),
            "moment_accumulation_time": float(solve.diagnostics.get("moment_accumulation_time", 0.0)),
            "bellman_solve_time": float(solve_time),
            "backend": self.backend_,
            "numba_available": bool(HAS_NUMBA),
            "n_rows": int(x.shape[0]),
            "estimated_memory_mb": float(self._estimate_memory_mb(x.shape[0], x.shape[1])),
            "fit_time": float(time.perf_counter() - start),
            "raw_train_loss": float(train_loss[-1]) if train_loss else float("nan"),
            "raw_validation_loss": float(validation_loss[-1]) if validation_loss else float("nan"),
        }
        return self

    def predict(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        if getattr(self, "feature_storage_", "csr") != "csr":
            return self._predict_streaming(X_eval)
        return np.asarray(self.transform(X_eval) @ self.theta_, dtype=np.float64).reshape(-1)

    def raw_predict(self, X_eval: Array) -> Array:
        self._check_booster_is_fitted()
        bins, missing = _transform_bins(X_eval, self.binner_)
        pred = np.full(bins.shape[0], float(self.base_score_), dtype=np.float64)
        for tree in self.trees_:
            _, update = _apply_one_tree(tree, bins, missing, use_numba=getattr(self, "backend_", "numpy") == "numba")
            pred += update
        return pred

    def apply(self, X: Array) -> Array:
        self._check_booster_is_fitted()
        bins, missing = _transform_bins(X, self.binner_)
        if bins.shape[0] > int(getattr(self, "batch_size", 65_536)):
            out = np.empty((bins.shape[0], len(self.trees_)), dtype=np.int32)
            for start in range(0, bins.shape[0], int(self.batch_size)):
                stop = min(start + int(self.batch_size), bins.shape[0])
                out[start:stop] = self._apply_binned(bins[start:stop], missing[start:stop])
            return out
        return self._apply_binned(bins, missing)

    def _apply_binned(self, bins: Array, missing: Array) -> Array:
        out = np.empty((bins.shape[0], len(self.trees_)), dtype=np.int32)
        for j, tree in enumerate(self.trees_):
            leaves, _ = _apply_one_tree(tree, bins, missing, use_numba=getattr(self, "backend_", "numpy") == "numba")
            out[:, j] = leaves
        return out

    def transform(self, X: Array) -> sparse.csr_matrix:
        if getattr(self, "feature_storage_", "csr") != "csr" and np.asarray(X).shape[0] >= 250_000:
            warnings.warn(
                "Materializing sparse leaf features for a large streaming/hashed model can use substantial memory.",
                RuntimeWarning,
                stacklevel=2,
            )
        leaves = self.apply(X)
        return self._leaf_features_to_csr(leaves)

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        self._check_booster_is_fitted()
        return average_next_features(self.transform, X_next)

    def _column_maps_from_training(self, X: Array) -> list[dict[int, int]]:
        leaves = self.apply(X)
        _, column_maps = leaf_matrix_from_columns(leaves)
        return column_maps

    def _resolve_backend(self) -> str:
        backend = str(self.backend)
        if backend == "auto":
            return "numba" if HAS_NUMBA else "numpy"
        if backend == "numba" and not HAS_NUMBA:
            return "numpy"
        if backend not in {"numpy", "numba"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'numba'.")
        return backend

    def _configure_feature_layout(self, n_samples: int) -> None:
        self.leaf_counts_ = np.asarray([int(tree.n_leaves) for tree in self.trees_], dtype=np.int64)
        self.leaf_offsets_ = np.cumsum(np.concatenate([[0], self.leaf_counts_[:-1]])).astype(np.int64)
        self.raw_leaf_features_ = int(np.sum(self.leaf_counts_))
        requested = str(self.feature_storage)
        if requested == "auto":
            storage = "streaming" if int(n_samples) >= 250_000 else "csr"
        elif requested in {"csr", "streaming", "hashed"}:
            storage = requested
        else:
            raise ValueError("feature_storage must be 'auto', 'csr', 'streaming', or 'hashed'.")
        if storage == "streaming" and self.raw_leaf_features_ > int(self.max_exact_features):
            storage = "hashed"
        self.feature_storage_ = storage
        self.n_solver_features_ = int(self.hash_dim) if storage == "hashed" else int(self.raw_leaf_features_)
        self.column_maps_ = [{leaf: leaf for leaf in range(int(count))} for count in self.leaf_counts_]

    def _resolve_streaming_solver_method(self) -> str:
        method = str(self.solver_method)
        if method == "auto":
            if self.feature_storage_ == "hashed" or int(self.n_solver_features_) > 4096:
                return "streaming_fqe"
            return "streaming_direct"
        if method in {"streaming_direct", "streaming_fqe", "streaming_lstd_iterative"}:
            return method
        if method in {"direct", "iterative"}:
            return "streaming_direct" if int(self.n_solver_features_) <= 4096 else "streaming_fqe"
        raise ValueError("Unsupported solver_method for streaming feature storage.")

    def _leaf_indices_values(self, leaves: Array, *, average_factor: float = 1.0) -> tuple[Array, Array]:
        arr = np.asarray(leaves, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError("leaves must be a 2D array.")
        scale = float(average_factor) / float(max(arr.shape[1], 1))
        if getattr(self, "feature_storage_", "csr") == "hashed":
            cols, signs = self._hash_leaf_columns(arr)
            return cols, signs.astype(np.float64) * scale
        cols = arr + self.leaf_offsets_[None, :]
        vals = np.full(cols.shape, scale, dtype=np.float64)
        return cols.astype(np.int64, copy=False), vals

    def _hash_leaf_columns(self, leaves: Array) -> tuple[Array, Array]:
        arr = np.asarray(leaves, dtype=np.uint64)
        tree_ids = np.arange(arr.shape[1], dtype=np.uint64)[None, :]
        key = (arr + np.uint64(1)) ^ ((tree_ids + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15))
        hashed = _splitmix64(key)
        cols = (hashed % np.uint64(int(self.hash_dim))).astype(np.int64)
        signs = np.where(((hashed >> np.uint64(63)) & np.uint64(1)) == 0, 1.0, -1.0)
        return cols, signs

    def _leaf_features_to_csr(self, leaves: Array) -> sparse.csr_matrix:
        if getattr(self, "feature_storage_", "csr") == "hashed":
            cols, vals = self._leaf_indices_values(leaves)
            n = cols.shape[0]
            rows = np.repeat(np.arange(n, dtype=np.int64), cols.shape[1])
            return sparse.csr_matrix((vals.reshape(-1), (rows, cols.reshape(-1))), shape=(n, int(self.n_solver_features_)))
        maps = getattr(self, "column_maps_", None)
        matrix, column_maps = leaf_matrix_from_columns(leaves, column_maps=maps)
        if maps is None:
            self.column_maps_ = column_maps
        return matrix

    def _transform_next_bins(self, X_next: Array) -> tuple[Array, Array]:
        xp = np.asarray(X_next)
        if xp.ndim == 2:
            return _transform_bins(xp, self.binner_)
        if xp.ndim != 3:
            raise ValueError(f"X_next must be 2D or 3D, got shape {xp.shape}.")
        n, m, p = xp.shape
        bins, missing = _transform_bins(xp.reshape(n * m, p), self.binner_)
        return bins.reshape(n, m, p), missing.reshape(n, m, p)

    def _next_indices_values_from_binned(self, bins: Array, missing: Array) -> tuple[Array, Array]:
        if bins.ndim == 2:
            return self._leaf_indices_values(self._apply_binned(bins, missing))
        n, m, p = bins.shape
        leaves = self._apply_binned(bins.reshape(n * m, p), missing.reshape(n * m, p))
        idx, vals = self._leaf_indices_values(leaves, average_factor=1.0 / float(m))
        return idx.reshape(n, m * leaves.shape[1]), vals.reshape(n, m * leaves.shape[1])

    def _iter_streaming_batches(
        self,
        *,
        bins: Array,
        missing: Array,
        next_bins: Array,
        next_missing: Array,
        reward: Array,
        weights: Array,
    ):
        n = bins.shape[0]
        batch_size = int(self.batch_size)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            cur_leaves = self._apply_binned(bins[start:stop], missing[start:stop])
            cur_idx, cur_vals = self._leaf_indices_values(cur_leaves)
            next_idx, next_vals = self._next_indices_values_from_binned(next_bins[start:stop], next_missing[start:stop])
            yield StreamingFeatureBatch(
                current_indices=cur_idx,
                current_values=cur_vals,
                next_indices=next_idx,
                next_values=next_vals,
                reward=reward[start:stop],
                weight=weights[start:stop],
            )

    def _predict_streaming(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        x = _as_2d_allow_nan("X", X_eval)
        out = np.empty(x.shape[0], dtype=np.float64)
        batch_size = int(self.batch_size)
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            leaves = self.apply(x[start:stop])
            idx, vals = self._leaf_indices_values(leaves)
            out[start:stop] = np.sum(vals * self.theta_[idx], axis=1)
        return out

    def _estimate_memory_mb(self, n: int, p: int) -> float:
        bin_bytes = int(n) * int(p) * (2 if int(self.max_bins) <= np.iinfo(np.uint16).max else 4)
        missing_bytes = int(n) * int(p)
        theta_bytes = int(getattr(self, "n_solver_features_", 0)) * 8
        dense_solver_bytes = 0
        if getattr(self, "feature_storage_", "csr") != "hashed" and int(getattr(self, "n_solver_features_", 0)) <= 4096:
            dense_solver_bytes = int(getattr(self, "n_solver_features_", 0)) ** 2 * 8
        return float((bin_bytes + missing_bytes + theta_bytes + dense_solver_bytes) / 1_000_000)

    def _validate_parameters(self, n: int, p: int) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive.")
        if not 0.0 < float(self.learning_rate) <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.max_leaves is not None and int(self.max_leaves) < 1:
            raise ValueError("max_leaves must be positive when provided.")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be positive.")
        if int(self.min_samples_leaf) * 2 > n:
            raise ValueError("min_samples_leaf is too large for the sample size.")
        if float(self.min_child_weight) < 0.0:
            raise ValueError("min_child_weight must be nonnegative.")
        if float(self.l2_leaf_reg) < 0.0 or float(self.ridge) < 0.0:
            raise ValueError("regularization parameters must be nonnegative.")
        if float(self.split_gamma) < 0.0:
            raise ValueError("split_gamma must be nonnegative.")
        _validate_fraction("subsample", self.subsample)
        if self.max_samples_per_tree is not None and int(self.max_samples_per_tree) < 2 * int(self.min_samples_leaf):
            raise ValueError("max_samples_per_tree must be at least 2 * min_samples_leaf when provided.")
        _validate_fraction("colsample_bytree", self.colsample_bytree)
        _validate_fraction("colsample_bynode", self.colsample_bynode)
        if self.early_stopping_rounds is not None and int(self.early_stopping_rounds) < 1:
            raise ValueError("early_stopping_rounds must be positive when provided.")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if str(self.backend) not in {"auto", "numpy", "numba"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'numba'.")
        if str(self.feature_storage) not in {"auto", "csr", "streaming", "hashed"}:
            raise ValueError("feature_storage must be 'auto', 'csr', 'streaming', or 'hashed'.")
        if int(self.hash_dim) <= 0:
            raise ValueError("hash_dim must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if str(self.dtype) not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        if str(self.accumulator_dtype) != "float64":
            raise ValueError("accumulator_dtype currently must be 'float64'.")
        if int(self.max_exact_features) <= 0:
            raise ValueError("max_exact_features must be positive.")
        if p < 1:
            raise ValueError("X must contain at least one feature.")

    def _train_validation_split(self, n: int, rng: np.random.Generator) -> tuple[Array, Array]:
        if self.early_stopping_rounds is None or float(self.validation_fraction) <= 0.0:
            return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
        n_val = max(1, int(round(float(self.validation_fraction) * n)))
        n_val = min(n_val, n - max(1, 2 * int(self.min_samples_leaf)))
        if n_val <= 0:
            return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
        perm = rng.permutation(n)
        return np.ascontiguousarray(perm[n_val:], dtype=np.int64), np.ascontiguousarray(perm[:n_val], dtype=np.int64)

    def _sample_rows(self, train_idx: Array, rng: np.random.Generator) -> Array:
        frac = _validate_fraction("subsample", self.subsample)
        if frac >= 1.0 or train_idx.size <= 1:
            if self.max_samples_per_tree is None or train_idx.size <= int(self.max_samples_per_tree):
                return np.ascontiguousarray(train_idx, dtype=np.int64)
            m = min(train_idx.size, int(self.max_samples_per_tree))
            return np.ascontiguousarray(rng.choice(train_idx, size=m, replace=False), dtype=np.int64)
        m = max(2 * int(self.min_samples_leaf), int(round(frac * train_idx.size)))
        if self.max_samples_per_tree is not None:
            m = min(m, int(self.max_samples_per_tree))
        m = min(m, train_idx.size)
        return np.ascontiguousarray(rng.choice(train_idx, size=m, replace=False), dtype=np.int64)

    def _sample_features(self, p: int, rng: np.random.Generator, frac: float) -> Array:
        keep = _validate_fraction("colsample_bytree", frac)
        if keep >= 1.0 or p <= 1:
            return np.arange(p, dtype=np.int64)
        k = max(1, int(round(keep * p)))
        return np.sort(rng.choice(p, size=k, replace=False)).astype(np.int64)

    @staticmethod
    def _weighted_mse(y: Array, pred: Array, weights: Array) -> float:
        total = float(np.sum(weights))
        if total <= 0.0:
            return float("nan")
        return float(np.sum(weights * (y - pred) ** 2) / total)

    def _check_booster_is_fitted(self) -> None:
        if not hasattr(self, "trees_") or not hasattr(self, "binner_"):
            raise RuntimeError("Estimator is not fitted.")

    def _check_is_fitted(self) -> None:
        self._check_booster_is_fitted()
        if not hasattr(self, "theta_"):
            raise RuntimeError("Estimator is not fitted.")
