from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ._base import SerializableEstimatorMixin
from ._features import average_next_features
from ._hist_gbt import (
    QuantileBinner,
    _HistogramTree,
    _apply_one_tree,
    _as_2d_allow_nan,
    _as_next_allow_nan,
    _fit_histogram_tree,
    _fit_quantile_binner,
    _splitmix64,
    _transform_bins,
    _validate_fraction,
)
from ._hist_gbt_fast import HAS_NUMBA
from ._occupancy import discounted_flow_moment, solve_discounted_occupancy_ratio
from ._streaming_solver import (
    StreamingInitialFeatureBatch,
    StreamingOccupancyFeatureBatch,
    solve_streaming_occupancy_ratio,
)
from ._weights import stabilize_weights


Array = np.ndarray


@dataclass(frozen=True)
class _SignedFlowEvents:
    X: Array
    signed_mass: Array
    current_events: int
    next_events: int
    initial_events: int
    sampled: bool = False


def _as_initial_allow_nan(value: Array, p: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[1] != p:
            raise ValueError(f"X_initial must have {p} columns, got {arr.shape[1]}.")
        out = np.ascontiguousarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] != p:
            raise ValueError(f"X_initial must have trailing dimension {p}, got {arr.shape[2]}.")
        out = np.ascontiguousarray(arr)
    else:
        raise ValueError(f"X_initial must be 2D or 3D, got shape {arr.shape}.")
    if np.any(np.isinf(out)):
        raise ValueError("X_initial contains infinite values.")
    return out


def _as_nonnegative_weight(name: str, value: Array | None, n: int) -> Array:
    if value is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if float(np.sum(arr)) <= 0.0:
        raise ValueError(f"{name} must have positive sum.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _flatten_reference_events(X_ref: Array, mass: Array) -> tuple[Array, Array]:
    arr = np.asarray(X_ref, dtype=np.float64)
    weights = np.asarray(mass, dtype=np.float64).reshape(-1)
    if arr.ndim == 2:
        if arr.shape[0] != weights.shape[0]:
            raise ValueError(f"reference weights must have length {arr.shape[0]}, got {weights.shape[0]}.")
        return np.ascontiguousarray(arr), np.ascontiguousarray(weights)
    if arr.ndim != 3:
        raise ValueError(f"reference features must be 2D or 3D, got shape {arr.shape}.")
    n, m, p = arr.shape
    if weights.shape[0] != n:
        raise ValueError(f"reference weights must have length {n}, got {weights.shape[0]}.")
    return np.ascontiguousarray(arr.reshape(n * m, p)), np.repeat(weights / float(m), m)


def _build_signed_flow_events(
    X: Array,
    X_next: Array,
    X_initial: Array,
    ratio: Array,
    sample_weight: Array,
    initial_weight: Array,
    *,
    gamma: float,
    max_event_rows: int | None = None,
    rng: np.random.Generator | None = None,
) -> _SignedFlowEvents:
    """Create signed empirical flow-balance events for one boosting round."""

    x = _as_2d_allow_nan("X", X)
    xp = _as_next_allow_nan(X_next, x.shape[0], x.shape[1])
    x0 = _as_initial_allow_nan(X_initial, x.shape[1])
    rho = np.asarray(ratio, dtype=np.float64).reshape(-1)
    if rho.shape[0] != x.shape[0]:
        raise ValueError(f"ratio must have length {x.shape[0]}, got {rho.shape[0]}.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("ratio contains non-finite values.")
    w = _as_nonnegative_weight("sample_weight", sample_weight, x.shape[0])
    wi = _as_nonnegative_weight("initial_weight", initial_weight, x0.shape[0])

    normalized_weighted_ratio = w * rho / float(np.sum(w))
    current_mass = normalized_weighted_ratio
    next_mass = -float(gamma) * normalized_weighted_ratio
    initial_mass = -(1.0 - float(gamma)) * wi / float(np.sum(wi))

    sampled = False
    next_multiplier = 1 if xp.ndim == 2 else int(xp.shape[1])
    initial_multiplier = 1 if x0.ndim == 2 else int(x0.shape[1])
    n_current = int(x.shape[0])
    n_next = int(x.shape[0] * next_multiplier)
    n_initial = int(x0.shape[0] * initial_multiplier)
    total_events = n_current + n_next + n_initial
    if max_event_rows is not None and total_events > int(max_event_rows):
        if rng is None:
            rng = np.random.default_rng()
        k = max(1, int(max_event_rows))
        probability = np.concatenate(
            [
                np.abs(current_mass),
                np.repeat(np.abs(next_mass) / float(next_multiplier), next_multiplier),
                np.repeat(np.abs(initial_mass) / float(initial_multiplier), initial_multiplier),
            ]
        )
        total = float(np.sum(probability))
        if total <= 0.0 or not np.isfinite(total):
            probability = None
        else:
            probability = probability / total
        idx = rng.choice(total_events, size=k, replace=False, p=probability).astype(np.int64)
        current_keep = idx < n_current
        next_keep = (idx >= n_current) & (idx < n_current + n_next)
        initial_keep = idx >= n_current + n_next
        blocks: list[Array] = []
        masses: list[Array] = []
        if np.any(current_keep):
            current_idx = idx[current_keep]
            blocks.append(x[current_idx])
            masses.append(current_mass[current_idx])
        if np.any(next_keep):
            next_idx = idx[next_keep] - n_current
            if xp.ndim == 2:
                blocks.append(xp[next_idx])
                masses.append(next_mass[next_idx])
            else:
                flat_xp = xp.reshape(n_next, xp.shape[2])
                blocks.append(flat_xp[next_idx])
                masses.append(np.repeat(next_mass / float(next_multiplier), next_multiplier)[next_idx])
        if np.any(initial_keep):
            initial_idx = idx[initial_keep] - n_current - n_next
            if x0.ndim == 2:
                blocks.append(x0[initial_idx])
                masses.append(initial_mass[initial_idx])
            else:
                flat_x0 = x0.reshape(n_initial, x0.shape[2])
                blocks.append(flat_x0[initial_idx])
                masses.append(np.repeat(initial_mass / float(initial_multiplier), initial_multiplier)[initial_idx])
        event_x = np.vstack(blocks)
        signed_mass = np.concatenate(masses).astype(np.float64, copy=False)
        sampled = True
    else:
        next_x, next_signed_mass = _flatten_reference_events(xp, next_mass)
        initial_x, initial_signed_mass = _flatten_reference_events(x0, initial_mass)
        event_x = np.vstack([x, next_x, initial_x])
        signed_mass = np.concatenate([current_mass, next_signed_mass, initial_signed_mass]).astype(np.float64, copy=False)

    return _SignedFlowEvents(
        X=np.ascontiguousarray(event_x, dtype=np.float64),
        signed_mass=np.ascontiguousarray(signed_mass, dtype=np.float64),
        current_events=int(x.shape[0]),
        next_events=int(n_next),
        initial_events=int(n_initial),
        sampled=sampled,
    )


def _leaf_matrix_from_histogram_leaves(leaves: Array, leaf_counts: list[int]) -> sparse.csr_matrix:
    arr = np.asarray(leaves, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"leaf array must be 1D or 2D after squeezing, got shape {arr.shape}.")
    n, t = arr.shape
    if len(leaf_counts) != t:
        raise ValueError("leaf_counts length does not match number of leaf columns.")
    if t == 0:
        return sparse.csr_matrix((n, 0), dtype=np.float64)
    counts = np.asarray(leaf_counts, dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError("leaf counts must be positive.")
    if np.any(arr < 0) or np.any(arr >= counts.reshape(1, -1)):
        raise ValueError("leaf assignments are outside the fitted leaf range.")
    offsets = np.cumsum(np.concatenate([np.array([0], dtype=np.int64), counts[:-1]]))
    rows = np.repeat(np.arange(n, dtype=np.int64), t)
    cols = (arr + offsets.reshape(1, -1)).reshape(-1)
    data = np.full(n * t, 1.0 / float(t), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, int(np.sum(counts))), dtype=np.float64)


class DiscountedOccupancyHistogramGradientBoostingRatioEstimator(SerializableEstimatorMixin):
    """Histogram boosted leaf features for discounted occupancy-ratio solving.

    Trees are grown on signed discounted flow-balance residual events. Their
    raw leaf values are only a partition-construction device; final ratios are
    produced by a nonnegative sparse flow-balance solve on the boosted leaves.
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
        min_child_weight: float = 1e-6,
        l2_leaf_reg: float = 1.0,
        split_gamma: float = 0.0,
        hessian_floor: float = 1e-8,
        subsample: float = 1.0,
        max_event_rows: int | None = 200_000,
        colsample_bytree: float = 1.0,
        colsample_bynode: float = 1.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.1,
        solver: str = "auto",
        ratio_refresh_interval: int = 1,
        inner_solver_tol: float = 1e-4,
        inner_solver_max_iter: int = 100,
        final_solver_tol: float = 1e-6,
        final_solver_max_iter: int = 1000,
        ridge: float = 1e-6,
        dense_threshold: int = 2048,
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
        self.hessian_floor = hessian_floor
        self.subsample = subsample
        self.max_event_rows = max_event_rows
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.solver = solver
        self.ratio_refresh_interval = ratio_refresh_interval
        self.inner_solver_tol = inner_solver_tol
        self.inner_solver_max_iter = inner_solver_max_iter
        self.final_solver_tol = final_solver_tol
        self.final_solver_max_iter = final_solver_max_iter
        self.ridge = ridge
        self.dense_threshold = dense_threshold
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
        X_next: Array,
        X_initial: Array,
        sample_weight: Array | None = None,
        initial_weight: Array | None = None,
    ) -> "DiscountedOccupancyHistogramGradientBoostingRatioEstimator":
        start = time.perf_counter()
        x = _as_2d_allow_nan("X", X)
        xp = _as_next_allow_nan(X_next, x.shape[0], x.shape[1])
        x0 = _as_initial_allow_nan(X_initial, x.shape[1])
        raw_weight = None if sample_weight is None else _as_nonnegative_weight("sample_weight", sample_weight, x.shape[0])
        initial_w = _as_nonnegative_weight("initial_weight", initial_weight, x0.shape[0])
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
        self.binner_ = self._fit_binner(x[train_idx], xp[train_idx], x0, rng)
        binning_time = time.perf_counter() - bin_start

        ratio = np.ones(x.shape[0], dtype=np.float64)
        self.trees_: list[_HistogramTree] = []
        self.split_history_: list[dict[str, Any]] = []
        self.flow_train_loss_: list[float] = []
        self.flow_validation_loss_: list[float] = []
        self.round_timing_: list[dict[str, float | int]] = []

        beta_warm: Array | None = None
        best_iteration = -1
        best_loss = np.inf
        rounds_since_improvement = 0
        train_start = time.perf_counter()

        for iteration in range(int(self.n_estimators)):
            round_start = time.perf_counter()
            event_start = time.perf_counter()
            events = _build_signed_flow_events(
                x[train_idx],
                xp[train_idx],
                x0,
                ratio[train_idx],
                weights.values[train_idx],
                initial_w,
                gamma=float(self.gamma),
                max_event_rows=None if self.max_event_rows is None else int(self.max_event_rows),
                rng=rng,
            )
            event_time = time.perf_counter() - event_start
            bin_start = time.perf_counter()
            bins, missing = _transform_bins(events.X, self.binner_)
            event_binning_time = time.perf_counter() - bin_start
            sample_start = time.perf_counter()
            rows = self._sample_event_rows(events.X.shape[0], rng)
            tree_features = self._sample_features(x.shape[1], rng, self.colsample_bytree)
            sampling_time = time.perf_counter() - sample_start
            grad = -events.signed_mass
            hess = np.abs(events.signed_mass) + float(self.hessian_floor)
            tree_start = time.perf_counter()
            tree, history = _fit_histogram_tree(
                bins=bins,
                missing=missing,
                binner=self.binner_,
                grad=grad,
                hess=hess,
                rows=rows,
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
            tree_time = time.perf_counter() - tree_start
            self.trees_.append(tree)
            self.split_history_.append(
                {
                    "tree": int(iteration),
                    "splits": history,
                    "event_rows": int(events.X.shape[0]),
                    "event_l1_mass": float(np.sum(np.abs(events.signed_mass))),
                    "sampled_events": bool(events.sampled),
                }
            )

            inner_solve_time = 0.0
            should_refresh_ratio = self._should_refresh_ratio(iteration, val_idx.size > 0)
            if should_refresh_ratio:
                inner_solve_start = time.perf_counter()
                solve, phi, solve_timing = self._solve_current_features(
                    x,
                    xp,
                    x0,
                    weights.values,
                    initial_w,
                    tol=float(self.inner_solver_tol),
                    max_iter=int(self.inner_solver_max_iter),
                    warm_start=beta_warm,
                )
                inner_solve_time = time.perf_counter() - inner_solve_start
                beta_warm = solve.beta
                ratio = self._predict_ratio_with_beta(x, beta_warm) if phi is None else np.asarray(phi @ beta_warm, dtype=np.float64).reshape(-1)
                train_loss = float(solve.diagnostics.get("moment_violation_mean_square", np.nan))
                current_width = int(self.n_solver_features_)
            else:
                train_loss = self.flow_train_loss_[-1] if self.flow_train_loss_ else float("nan")
                current_width = int(sum(tree.n_leaves for tree in self.trees_))
            self.flow_train_loss_.append(train_loss)
            current_score = train_loss
            validation_time = 0.0
            if val_idx.size:
                validation_start = time.perf_counter()
                val_loss = self._flow_loss_for_indices(x, xp, x0, weights.values, initial_w, val_idx, beta_warm)
                validation_time = time.perf_counter() - validation_start
                self.flow_validation_loss_.append(val_loss)
                current_score = val_loss
                if val_loss + 1e-12 < best_loss:
                    best_loss = val_loss
                    best_iteration = iteration
                    rounds_since_improvement = 0
                else:
                    rounds_since_improvement += 1
                if self.early_stopping_rounds is not None and rounds_since_improvement >= int(self.early_stopping_rounds):
                    break
            else:
                best_iteration = iteration
                best_loss = min(best_loss, current_score)
            self.round_timing_.append(
                {
                    "iteration": int(iteration),
                    "event_rows": int(events.X.shape[0]),
                    "sampled_rows": int(rows.size),
                    "n_features": int(current_width),
                    "ratio_refreshed": int(should_refresh_ratio),
                    "event_build_time": float(event_time),
                    "event_binning_time": float(event_binning_time),
                    "sampling_time": float(sampling_time),
                    "tree_fit_time": float(tree_time),
                    "inner_solve_time": float(inner_solve_time),
                    "feature_build_time": float(solve_timing["feature_build_time"]) if should_refresh_ratio else 0.0,
                    "inner_solver_time": float(solve_timing["solver_time"]) if should_refresh_ratio else 0.0,
                    "validation_time": float(validation_time),
                    "round_time": float(time.perf_counter() - round_start),
                }
            )

        boosting_time = time.perf_counter() - train_start
        if not self.trees_:
            raise RuntimeError("boosting produced no trees.")
        if val_idx.size and best_iteration >= 0 and best_iteration + 1 < len(self.trees_):
            keep = best_iteration + 1
            self.trees_ = self.trees_[:keep]
            self.split_history_ = self.split_history_[:keep]
            self.flow_train_loss_ = self.flow_train_loss_[:keep]
            self.flow_validation_loss_ = self.flow_validation_loss_[:keep]
            self.round_timing_ = self.round_timing_[:keep]
            beta_warm = None

        solve_start = time.perf_counter()
        final_solve, phi, final_solve_timing = self._solve_current_features(
            x,
            xp,
            x0,
            weights.values,
            initial_w,
            tol=float(self.final_solver_tol),
            max_iter=int(self.final_solver_max_iter),
            warm_start=beta_warm,
        )
        solve_time = time.perf_counter() - solve_start

        self.beta_ = final_solve.beta
        self.theta_ = final_solve.beta
        self.initial_weight_ = initial_w
        self.solver_info_ = final_solve.diagnostics
        self.feature_info_ = {
            "n_trees": int(len(self.trees_)),
            "n_features": int(self.n_solver_features_),
            "n_features_raw": int(self.raw_leaf_features_),
            "n_features_solver": int(self.n_solver_features_),
            "n_input_features": int(x.shape[1]),
            "n_initial": int(x0.shape[0]),
            "leaf_counts": [int(tree.n_leaves) for tree in self.trees_],
            "feature_scale": float(1.0 / max(len(self.trees_), 1)),
            "feature_target": "discounted_flow_balance",
            "feature_storage": self.feature_storage_,
            "hash_dim": None if self.feature_storage_ != "hashed" else int(self.hash_dim),
            "hash_load_factor": None
            if self.feature_storage_ != "hashed"
            else float(self.raw_leaf_features_ / max(int(self.hash_dim), 1)),
            "joint_occupancy_solve": True,
            "solver": str(final_solve.diagnostics.get("solver", self.solver)),
            "ratio_refresh_interval": int(self.ratio_refresh_interval),
        }
        final_ratio = self._predict_ratio_with_beta(x, self.beta_) if phi is None else np.asarray(phi @ self.beta_, dtype=np.float64).reshape(-1)
        self.diagnostics_ = {
            **weights.diagnostics,
            **self.feature_info_,
            "best_iteration": int(best_iteration if best_iteration >= 0 else len(self.trees_) - 1),
            "binning_time": float(binning_time),
            "boosting_time": float(boosting_time),
            "occupancy_solve_time": float(solve_time),
            "final_feature_build_time": float(final_solve_timing["feature_build_time"]),
            "final_solver_time": float(final_solve_timing["solver_time"]),
            "fit_time": float(time.perf_counter() - start),
            "flow_train_loss": float(self.flow_train_loss_[-1]) if self.flow_train_loss_ else float("nan"),
            "flow_validation_loss": float(self.flow_validation_loss_[-1]) if self.flow_validation_loss_ else float("nan"),
            "ratio_min": float(np.min(final_ratio)) if final_ratio.size else float("nan"),
            "ratio_max": float(np.max(final_ratio)) if final_ratio.size else float("nan"),
            "ratio_mean": float(np.sum(weights.values * final_ratio) / float(np.sum(weights.values))),
            "backend": self.backend_,
            "numba_available": bool(HAS_NUMBA),
            "n_rows": int(x.shape[0]),
            "estimated_memory_mb": float(self._estimate_memory_mb(x.shape[0], x.shape[1])),
        }
        self.diagnostics_.update(self._aggregate_round_timings())
        return self

    def predict_ratio(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        return self._predict_ratio_with_beta(X_eval, self.beta_)

    def predict(self, X_eval: Array) -> Array:
        return self.predict_ratio(X_eval)

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
                "Materializing sparse leaf features for a large streaming/hashed occupancy model can use substantial memory.",
                RuntimeWarning,
                stacklevel=2,
            )
        leaves = self.apply(X)
        return self._leaf_features_to_csr(leaves)

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        self._check_booster_is_fitted()
        return average_next_features(self.transform, X_next)

    def transform_initial(self, X_initial: Array) -> sparse.csr_matrix:
        self._check_booster_is_fitted()
        return average_next_features(self.transform, _as_initial_allow_nan(X_initial, len(self.binner_.thresholds)))

    def _solve_current_features(
        self,
        X: Array,
        X_next: Array,
        X_initial: Array,
        sample_weight: Array,
        initial_weight: Array,
        *,
        tol: float,
        max_iter: int,
        warm_start: Array | None,
    ) -> tuple[Any, sparse.csr_matrix | None, dict[str, float]]:
        previous_maps = getattr(self, "column_maps_", None)
        self._configure_feature_layout(np.asarray(X).shape[0])
        feature_start = time.perf_counter()
        use_streaming = self.feature_storage_ != "csr" or str(self.solver) in {
            "streaming_direct",
            "streaming_fista",
        }
        if use_streaming:
            bins, missing = _transform_bins(X, self.binner_)
            next_bins, next_missing = self._transform_next_bins(X_next)
            initial_bins, initial_missing = self._transform_initial_bins(X_initial)
            next_maps = self._histogram_column_maps()
            self.column_maps_ = next_maps
            phi = None
        else:
            phi, phi_next, phi_initial, next_maps = self._feature_matrices(X, X_next, X_initial)
        feature_time = time.perf_counter() - feature_start
        solver_start = time.perf_counter()
        if use_streaming:
            method = self._resolve_streaming_solver_method()
            solve = solve_streaming_occupancy_ratio(
                lambda: self._iter_streaming_batches(
                    bins=bins,
                    missing=missing,
                    next_bins=next_bins,
                    next_missing=next_missing,
                    weights=sample_weight,
                ),
                lambda: self._iter_initial_batches(
                    initial_bins=initial_bins,
                    initial_missing=initial_missing,
                    initial_weight=initial_weight,
                ),
                n_features=int(self.n_solver_features_),
                gamma=float(self.gamma),
                ridge=float(self.ridge),
                method=method,
                max_iter=int(max_iter),
                tol=float(tol),
                normalize=True,
            )
        else:
            warm = self._warm_start_for_feature_maps(warm_start, previous_maps, next_maps, phi.shape[1])
            solve = solve_discounted_occupancy_ratio(
                phi,
                phi_next,
                phi_initial,
                sample_weight,
                initial_weight,
                gamma=float(self.gamma),
                ridge=float(self.ridge),
                nonnegative=True,
                solver=str(self.solver),
                normalize=True,
                dense_threshold=int(self.dense_threshold),
                tol=float(tol),
                max_iter=int(max_iter),
                warm_start=warm,
            )
        solver_time = time.perf_counter() - solver_start
        return solve, phi, {"feature_build_time": float(feature_time), "solver_time": float(solver_time)}

    def _feature_matrices(
        self,
        X: Array,
        X_next: Array,
        X_initial: Array,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, list[dict[int, int]]]:
        leaves = self.apply(X)
        leaf_counts = self._leaf_counts()
        phi = _leaf_matrix_from_histogram_leaves(leaves, leaf_counts)
        next_maps = self._histogram_column_maps()
        self.column_maps_ = next_maps
        phi_next = self.transform_next(X_next)
        phi_initial = self.transform_initial(X_initial)
        return phi.tocsr(), phi_next.tocsr(), phi_initial.tocsr(), next_maps

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
        solver = str(self.solver)
        if solver == "auto":
            return "streaming_direct" if int(self.n_solver_features_) <= 4096 else "streaming_fista"
        if solver in {"streaming_direct", "streaming_fista"}:
            return solver
        if solver in {"lsq_linear", "fista"}:
            return "streaming_direct" if int(self.n_solver_features_) <= 4096 else "streaming_fista"
        raise ValueError("Unsupported solver for streaming occupancy feature storage.")

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
        signs = np.ones(cols.shape, dtype=np.float64)
        return cols, signs

    def _leaf_features_to_csr(self, leaves: Array) -> sparse.csr_matrix:
        if not hasattr(self, "leaf_offsets_"):
            self._configure_feature_layout(0)
        if getattr(self, "feature_storage_", "csr") == "hashed":
            cols, vals = self._leaf_indices_values(leaves)
            n = cols.shape[0]
            rows = np.repeat(np.arange(n, dtype=np.int64), cols.shape[1])
            return sparse.csr_matrix((vals.reshape(-1), (rows, cols.reshape(-1))), shape=(n, int(self.n_solver_features_)))
        return _leaf_matrix_from_histogram_leaves(leaves, [int(count) for count in self.leaf_counts_]).tocsr()

    def _transform_next_bins(self, X_next: Array) -> tuple[Array, Array]:
        xp = np.asarray(X_next, dtype=np.float64)
        if xp.ndim == 2:
            return _transform_bins(xp, self.binner_)
        if xp.ndim != 3:
            raise ValueError(f"X_next must be 2D or 3D, got shape {xp.shape}.")
        n, m, p = xp.shape
        bins, missing = _transform_bins(xp.reshape(n * m, p), self.binner_)
        return bins.reshape(n, m, p), missing.reshape(n, m, p)

    def _transform_initial_bins(self, X_initial: Array) -> tuple[Array, Array]:
        x0 = _as_initial_allow_nan(X_initial, len(self.binner_.thresholds))
        if x0.ndim == 2:
            return _transform_bins(x0, self.binner_)
        n, m, p = x0.shape
        bins, missing = _transform_bins(x0.reshape(n * m, p), self.binner_)
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
        weights: Array,
    ):
        n = bins.shape[0]
        batch_size = int(self.batch_size)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            cur_leaves = self._apply_binned(bins[start:stop], missing[start:stop])
            cur_idx, cur_vals = self._leaf_indices_values(cur_leaves)
            next_idx, next_vals = self._next_indices_values_from_binned(next_bins[start:stop], next_missing[start:stop])
            yield StreamingOccupancyFeatureBatch(
                current_indices=cur_idx,
                current_values=cur_vals,
                next_indices=next_idx,
                next_values=next_vals,
                weight=weights[start:stop],
            )

    def _iter_initial_batches(
        self,
        *,
        initial_bins: Array,
        initial_missing: Array,
        initial_weight: Array,
    ):
        if initial_bins.ndim == 2:
            n = initial_bins.shape[0]
            batch_size = int(self.batch_size)
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                leaves = self._apply_binned(initial_bins[start:stop], initial_missing[start:stop])
                idx, vals = self._leaf_indices_values(leaves)
                yield StreamingInitialFeatureBatch(indices=idx, values=vals, weight=initial_weight[start:stop])
            return
        n, m, p = initial_bins.shape
        batch_size = int(self.batch_size)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            flat_bins = initial_bins[start:stop].reshape((stop - start) * m, p)
            flat_missing = initial_missing[start:stop].reshape((stop - start) * m, p)
            leaves = self._apply_binned(flat_bins, flat_missing)
            idx, vals = self._leaf_indices_values(leaves, average_factor=1.0 / float(m))
            yield StreamingInitialFeatureBatch(
                indices=idx.reshape(stop - start, m * leaves.shape[1]),
                values=vals.reshape(stop - start, m * leaves.shape[1]),
                weight=initial_weight[start:stop],
            )

    def _predict_ratio_with_beta(self, X_eval: Array, beta: Array) -> Array:
        self._check_booster_is_fitted()
        x = _as_2d_allow_nan("X", X_eval)
        if getattr(self, "feature_storage_", "csr") == "csr":
            return np.asarray(self.transform(x) @ beta, dtype=np.float64).reshape(-1)
        out = np.empty(x.shape[0], dtype=np.float64)
        batch_size = int(self.batch_size)
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            leaves = self.apply(x[start:stop])
            idx, vals = self._leaf_indices_values(leaves)
            out[start:stop] = np.sum(vals * np.asarray(beta, dtype=np.float64).reshape(-1)[idx], axis=1)
        return out

    def _estimate_memory_mb(self, n: int, p: int) -> float:
        bin_bytes = int(n) * int(p) * (2 if int(self.max_bins) <= np.iinfo(np.uint16).max else 4)
        missing_bytes = int(n) * int(p)
        beta_bytes = int(getattr(self, "n_solver_features_", 0)) * 8
        dense_solver_bytes = 0
        if getattr(self, "feature_storage_", "csr") != "hashed" and int(getattr(self, "n_solver_features_", 0)) <= 4096:
            dense_solver_bytes = int(getattr(self, "n_solver_features_", 0)) ** 2 * 8
        return float((bin_bytes + missing_bytes + beta_bytes + dense_solver_bytes) / 1_000_000)

    def _aggregate_round_timings(self) -> dict[str, float]:
        keys = [
            "event_build_time",
            "event_binning_time",
            "sampling_time",
            "tree_fit_time",
            "inner_solve_time",
            "feature_build_time",
            "inner_solver_time",
            "validation_time",
            "round_time",
        ]
        if not getattr(self, "round_timing_", None):
            return {f"total_{key}": 0.0 for key in keys} | {f"mean_{key}": 0.0 for key in keys}
        totals = {key: float(sum(float(row[key]) for row in self.round_timing_)) for key in keys}
        means = {key: value / float(len(self.round_timing_)) for key, value in totals.items()}
        return {
            **{f"total_{key}": value for key, value in totals.items()},
            **{f"mean_{key}": value for key, value in means.items()},
        }

    @staticmethod
    def _warm_start_for_feature_maps(
        warm_start: Array | None,
        previous_maps: list[dict[int, int]] | None,
        next_maps: list[dict[int, int]],
        next_width: int,
    ) -> Array | None:
        if warm_start is None:
            return None
        warm = np.asarray(warm_start, dtype=np.float64).reshape(-1)
        if warm.shape[0] == int(next_width):
            return np.ascontiguousarray(warm, dtype=np.float64)
        if previous_maps is None:
            return None
        previous_width = int(sum(len(mapping) for mapping in previous_maps))
        if warm.shape[0] != previous_width:
            return None
        previous_trees = len(previous_maps)
        next_trees = len(next_maps)
        if previous_trees <= 0 or next_trees != previous_trees + 1:
            return None
        out = np.zeros(int(next_width), dtype=np.float64)
        out[:previous_width] = warm * (float(next_trees) / float(previous_trees))
        return out

    def _flow_loss_for_indices(
        self,
        X: Array,
        X_next: Array,
        X_initial: Array,
        sample_weight: Array,
        initial_weight: Array,
        idx: Array,
        beta: Array,
    ) -> float:
        phi = self.transform(X[idx])
        phi_next = self.transform_next(X_next[idx])
        phi_initial = self.transform_initial(X_initial)
        moment = discounted_flow_moment(
            phi,
            phi_next,
            phi_initial,
            beta,
            sample_weight[idx],
            initial_weight,
            gamma=float(self.gamma),
        )
        return float(np.mean(moment**2)) if moment.size else 0.0

    def _fit_binner(self, X: Array, X_next: Array, X_initial: Array, rng: np.random.Generator) -> QuantileBinner:
        cap = None if self.max_event_rows is None else int(self.max_event_rows)
        blocks = [self._sample_rows_for_binner(np.asarray(X, dtype=np.float64), cap, rng)]
        xp = np.asarray(X_next, dtype=np.float64)
        if xp.ndim == 2:
            blocks.append(self._sample_rows_for_binner(xp, cap, rng))
        else:
            blocks.append(self._sample_rows_for_binner(xp.reshape(xp.shape[0] * xp.shape[1], xp.shape[2]), cap, rng))
        x0 = np.asarray(X_initial, dtype=np.float64)
        if x0.ndim == 2:
            blocks.append(self._sample_rows_for_binner(x0, cap, rng))
        else:
            blocks.append(self._sample_rows_for_binner(x0.reshape(x0.shape[0] * x0.shape[1], x0.shape[2]), cap, rng))
        support = np.vstack(blocks)
        if cap is not None and support.shape[0] > cap:
            idx = np.sort(rng.choice(support.shape[0], size=cap, replace=False)).astype(np.int64)
            support = support[idx]
        return _fit_quantile_binner(support, int(self.max_bins))

    @staticmethod
    def _sample_rows_for_binner(X: Array, cap: int | None, rng: np.random.Generator) -> Array:
        arr = np.asarray(X, dtype=np.float64)
        if cap is None or arr.shape[0] <= cap:
            return arr
        idx = np.sort(rng.choice(arr.shape[0], size=int(cap), replace=False)).astype(np.int64)
        return arr[idx]

    def _column_maps_from_training(self, X: Array) -> list[dict[int, int]]:
        return self._histogram_column_maps()

    def _leaf_counts(self) -> list[int]:
        self._check_booster_is_fitted()
        return [int(tree.n_leaves) for tree in self.trees_]

    def _histogram_column_maps(self) -> list[dict[int, int]]:
        return [{leaf_id: leaf_id for leaf_id in range(count)} for count in self._leaf_counts()]

    def _validate_parameters(self, n: int, p: int) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive.")
        if not 0.0 < float(self.learning_rate) <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.max_leaves is not None and int(self.max_leaves) < 1:
            raise ValueError("max_leaves must be positive when provided.")
        if int(self.max_bins) < 2:
            raise ValueError("max_bins must be at least 2.")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be positive.")
        if int(self.min_samples_leaf) * 2 > max(2, 3 * n):
            raise ValueError("min_samples_leaf is too large for the generated flow-event sample.")
        if float(self.min_child_weight) < 0.0:
            raise ValueError("min_child_weight must be nonnegative.")
        if float(self.l2_leaf_reg) < 0.0 or float(self.ridge) < 0.0:
            raise ValueError("regularization parameters must be nonnegative.")
        if float(self.split_gamma) < 0.0:
            raise ValueError("split_gamma must be nonnegative.")
        if float(self.hessian_floor) < 0.0:
            raise ValueError("hessian_floor must be nonnegative.")
        if float(self.inner_solver_tol) <= 0.0 or float(self.final_solver_tol) <= 0.0:
            raise ValueError("solver tolerances must be positive.")
        if int(self.inner_solver_max_iter) <= 0 or int(self.final_solver_max_iter) <= 0:
            raise ValueError("solver max_iter values must be positive.")
        if int(self.ratio_refresh_interval) <= 0:
            raise ValueError("ratio_refresh_interval must be positive.")
        if int(self.dense_threshold) < 1:
            raise ValueError("dense_threshold must be positive.")
        if self.max_event_rows is not None and int(self.max_event_rows) < 1:
            raise ValueError("max_event_rows must be positive when provided.")
        _validate_fraction("subsample", self.subsample)
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
        n_val = min(n_val, n - max(1, int(self.min_samples_leaf)))
        if n_val <= 0:
            return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
        perm = rng.permutation(n)
        return np.ascontiguousarray(perm[n_val:], dtype=np.int64), np.ascontiguousarray(perm[:n_val], dtype=np.int64)

    def _sample_event_rows(self, n_events: int, rng: np.random.Generator) -> Array:
        rows = np.arange(n_events, dtype=np.int64)
        frac = _validate_fraction("subsample", self.subsample)
        if frac >= 1.0 or n_events <= 1:
            return rows
        m = max(2 * int(self.min_samples_leaf), int(round(frac * n_events)))
        m = min(m, n_events)
        return np.ascontiguousarray(rng.choice(rows, size=m, replace=False), dtype=np.int64)

    def _sample_features(self, p: int, rng: np.random.Generator, frac: float) -> Array:
        keep = _validate_fraction("colsample_bytree", frac)
        if keep >= 1.0 or p <= 1:
            return np.arange(p, dtype=np.int64)
        k = max(1, int(round(keep * p)))
        return np.sort(rng.choice(p, size=k, replace=False)).astype(np.int64)

    def _should_refresh_ratio(self, iteration: int, has_validation: bool) -> bool:
        if has_validation:
            return True
        return (int(iteration) + 1) % int(self.ratio_refresh_interval) == 0

    def _check_booster_is_fitted(self) -> None:
        if not hasattr(self, "trees_") or not hasattr(self, "binner_"):
            raise RuntimeError("Estimator is not fitted.")

    def _check_is_fitted(self) -> None:
        self._check_booster_is_fitted()
        if not hasattr(self, "beta_"):
            raise RuntimeError("Estimator is not fitted.")
