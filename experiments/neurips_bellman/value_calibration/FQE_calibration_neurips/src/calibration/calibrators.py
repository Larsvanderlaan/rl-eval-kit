from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


Array = np.ndarray


class BaseCalibrator:
    name = "base"

    def predict(self, prediction: Array) -> Array:
        raise NotImplementedError


class IdentityCalibrator(BaseCalibrator):
    name = "identity"

    def predict(self, prediction: Array) -> Array:
        return np.asarray(prediction, dtype=float)


@dataclass
class LinearCalibrator(BaseCalibrator):
    model: LinearRegression
    name: str = "linear"

    def predict(self, prediction: Array) -> Array:
        x = np.asarray(prediction, dtype=float).reshape(-1, 1)
        return self.model.predict(x).astype(float)


@dataclass
class HistogramCalibrator(BaseCalibrator):
    bin_edges: Array
    bin_values: Array
    name: str = "histogram"

    def predict(self, prediction: Array) -> Array:
        x = np.asarray(prediction, dtype=float).reshape(-1)
        bins = np.searchsorted(self.bin_edges[1:-1], x, side="right")
        return self.bin_values[np.clip(bins, 0, len(self.bin_values) - 1)]


@dataclass
class IsotonicCalibrator(BaseCalibrator):
    model: IsotonicRegression
    name: str = "isotonic"

    def predict(self, prediction: Array) -> Array:
        return self.model.predict(np.asarray(prediction, dtype=float).reshape(-1)).astype(float)


@dataclass
class HybridCalibrator(BaseCalibrator):
    bin_edges: Array
    isotonic: IsotonicRegression
    name: str = "isotonic_histogram"

    def predict(self, prediction: Array) -> Array:
        x = np.asarray(prediction, dtype=float).reshape(-1)
        return self.isotonic.predict(x).astype(float)


@dataclass
class IteratedBellmanCalibrator(BaseCalibrator):
    base: BaseCalibrator
    method: str
    n_iterations: int
    target_type: str
    name: str

    def predict(self, prediction: Array) -> Array:
        return self.base.predict(prediction)


@dataclass
class ValueBellmanCalibrator(BaseCalibrator):
    base: BaseCalibrator
    method: str
    n_iterations: int
    diagnostics: dict[str, float | str]
    name: str

    def predict(self, prediction: Array) -> Array:
        return self.base.predict(prediction)


def _valid_xy(prediction: Array, target: Array) -> tuple[Array, Array]:
    x = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two finite calibration points.")
    return x[mask], y[mask]


def _valid_weights(weights: Array | None, mask: Array) -> Array | None:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != mask.shape[0]:
        raise ValueError("sample_weight must have the same length as predictions.")
    w = w[mask]
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    if float(np.sum(w)) <= 0:
        return None
    return w


def _valid_xyw(prediction: Array, target: Array, sample_weight: Array | None) -> tuple[Array, Array, Array | None]:
    x = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(target, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two finite calibration points.")
    return x[mask], y[mask], _valid_weights(sample_weight, mask)


def _weighted_quantile(x: Array, weights: Array | None, quantiles: Array) -> Array:
    x = np.asarray(x, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if weights is None:
        return np.quantile(x, q)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    total = float(np.sum(ws))
    if total <= 0:
        return np.quantile(x, q)
    cdf = np.cumsum(ws) / total
    return np.interp(q, cdf, xs, left=xs[0], right=xs[-1])


def _weighted_mean(y: Array, weights: Array | None) -> float:
    if weights is None:
        return float(np.mean(y))
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(y))
    return float(np.sum(weights * y) / total)


def _histogram_fit(
    x: Array,
    y: Array,
    n_bins: int,
    strategy: str,
    min_bin_size: int,
    sample_weight: Array | None = None,
) -> HistogramCalibrator:
    n_bins = max(2, min(int(n_bins), max(2, len(x) // max(1, min_bin_size))))
    if strategy == "equal_width":
        edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
    else:
        edges = _weighted_quantile(x, sample_weight, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(float(np.min(x)) - 1e-8, float(np.max(x)) + 1e-8, 3)
    bin_ids = np.searchsorted(edges[1:-1], x, side="right")
    global_mean = _weighted_mean(y, sample_weight)
    values = np.zeros(edges.size - 1, dtype=float)
    for b in range(values.size):
        mask = bin_ids == b
        if np.sum(mask) >= max(1, min_bin_size):
            values[b] = _weighted_mean(y[mask], None if sample_weight is None else sample_weight[mask])
        else:
            values[b] = global_mean
    return HistogramCalibrator(edges, values)


def fit_calibrator(
    calibrator: str,
    prediction: Array,
    target: Array,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 20,
    sample_weight: Array | None = None,
) -> BaseCalibrator:
    x, y, w = _valid_xyw(prediction, target, sample_weight)
    if calibrator in {"none", "identity", None}:
        return IdentityCalibrator()
    if calibrator == "linear":
        return LinearCalibrator(LinearRegression().fit(x.reshape(-1, 1), y, sample_weight=w))
    if calibrator == "histogram":
        return _histogram_fit(x, y, n_bins=n_bins, strategy=bin_strategy, min_bin_size=min_bin_size, sample_weight=w)
    if calibrator == "isotonic":
        order = np.argsort(x)
        ordered_w = None if w is None else w[order]
        model = IsotonicRegression(out_of_bounds="clip").fit(x[order], y[order], sample_weight=ordered_w)
        return IsotonicCalibrator(model)
    if calibrator == "isotonic_histogram":
        hist = _histogram_fit(x, y, n_bins=n_bins, strategy=bin_strategy, min_bin_size=min_bin_size, sample_weight=w)
        centers = 0.5 * (hist.bin_edges[:-1] + hist.bin_edges[1:])
        bin_ids = np.searchsorted(hist.bin_edges[1:-1], x, side="right")
        if w is None:
            bin_weights = np.array([np.sum(bin_ids == b) for b in range(hist.bin_values.size)], dtype=float)
        else:
            bin_weights = np.array([np.sum(w[bin_ids == b]) for b in range(hist.bin_values.size)], dtype=float)
        bin_weights = np.maximum(bin_weights, 1e-12)
        order = np.argsort(centers)
        model = IsotonicRegression(out_of_bounds="clip").fit(
            centers[order],
            hist.bin_values[order],
            sample_weight=bin_weights[order],
        )
        return HybridCalibrator(hist.bin_edges, model)
    raise ValueError(f"Unknown calibrator '{calibrator}'.")


def is_iterated_bellman_calibrator(calibrator: str | None) -> bool:
    return calibrator in {"iterated_isotonic_bellman", "iterated_histogram_bellman"}


def fit_iterated_bellman_calibrator(
    calibrator: str,
    prediction: Array,
    next_prediction: Array,
    rewards: Array,
    gamma: float,
    target_type: str,
    n_iterations: int = 4,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 20,
) -> IteratedBellmanCalibrator:
    """Fit a one-dimensional Bellman calibration map by fixed-point iteration.

    The calibrator repeatedly forms Bellman targets

        reward + gamma * C_k(Q_next)

    and refits a monotone or histogram map from current Q predictions to those
    targets. Cross- and split-calibration still decide which observations enter
    this routine; this helper only consumes the calibration fold/split it is
    handed by the protocol code.
    """

    if calibrator == "iterated_isotonic_bellman":
        base_method = "isotonic"
    elif calibrator == "iterated_histogram_bellman":
        base_method = "histogram"
    else:
        raise ValueError(f"Unknown iterated Bellman calibrator '{calibrator}'.")

    pred = np.asarray(prediction, dtype=float).reshape(-1)
    nxt = np.asarray(next_prediction, dtype=float).reshape(-1)
    rew = np.asarray(rewards, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(nxt) & np.isfinite(rew)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two finite Bellman calibration points.")
    pred, nxt, rew = pred[mask], nxt[mask], rew[mask]
    n_iterations = max(1, int(n_iterations))
    base: BaseCalibrator | None = None
    for _ in range(n_iterations):
        if base is None:
            calibrated_next = nxt
        elif target_type == "td_residual":
            calibrated_next = nxt + base.predict(nxt)
        else:
            calibrated_next = base.predict(nxt)
        q_target = rew + float(gamma) * calibrated_next
        fit_target = q_target - pred if target_type == "td_residual" else q_target
        base = fit_calibrator(
            base_method,
            pred,
            fit_target,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            min_bin_size=min_bin_size,
        )
    assert base is not None
    return IteratedBellmanCalibrator(
        base=base,
        method=base_method,
        n_iterations=n_iterations,
        target_type=target_type,
        name=calibrator,
    )


def _value_calibrator_method(calibrator: str) -> str:
    if calibrator in {"linear", "histogram", "isotonic", "isotonic_histogram"}:
        return calibrator
    if calibrator == "iterated_isotonic_bellman":
        return "isotonic"
    if calibrator == "iterated_histogram_bellman":
        return "histogram"
    raise ValueError(f"Unknown value-space Bellman calibrator '{calibrator}'.")


def fit_value_bellman_calibrator(
    calibrator: str,
    values: Array,
    next_values: Array,
    rewards: Array,
    gamma: float,
    sample_weight: Array | None = None,
    n_iterations: int = 4,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 20,
) -> ValueBellmanCalibrator:
    """Fit an importance-weighted value-space Bellman/FVI calibrator.

    The input values are raw baseline value predictions V_hat(S) and
    V_hat(S'). The returned map g is applied to raw V_hat values, never to
    Q(s, a). At iteration k, targets are R + gamma * g_k(V_hat(S')).
    """

    method = _value_calibrator_method(calibrator)
    v = np.asarray(values, dtype=float).reshape(-1)
    vp = np.asarray(next_values, dtype=float).reshape(-1)
    r = np.asarray(rewards, dtype=float).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(vp) & np.isfinite(r)
    if sample_weight is not None:
        w_all = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w_all.shape[0] != v.shape[0]:
            raise ValueError("sample_weight must have the same length as values.")
        mask &= np.isfinite(w_all) & (w_all >= 0)
        w = w_all[mask]
    else:
        w = None
    if np.sum(mask) < 2:
        raise ValueError("Need at least two finite value calibration points.")
    v, vp, r = v[mask], vp[mask], r[mask]
    if w is not None and float(np.sum(w)) <= 0:
        w = None

    n_iterations = max(1, int(n_iterations))
    base: BaseCalibrator | None = None
    losses: list[float] = []
    for _ in range(n_iterations):
        next_calibrated = vp if base is None else base.predict(vp)
        target = r + float(gamma) * next_calibrated
        base = fit_calibrator(
            method,
            v,
            target,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            min_bin_size=min_bin_size,
            sample_weight=w,
        )
        pred = base.predict(v)
        if w is None:
            losses.append(float(np.mean((pred - target) ** 2)))
        else:
            losses.append(float(np.sum(w * (pred - target) ** 2) / max(float(np.sum(w)), 1e-12)))

    assert base is not None
    diagnostics = {
        "calibration_object": "value",
        "value_calibrator_method": method,
        "value_calibration_iterations": float(n_iterations),
        "value_calibration_loss_first": float(losses[0]) if losses else float("nan"),
        "value_calibration_loss_last": float(losses[-1]) if losses else float("nan"),
        "raw_value_min": float(np.nanmin(v)),
        "raw_value_max": float(np.nanmax(v)),
        "raw_value_prime_min": float(np.nanmin(vp)),
        "raw_value_prime_max": float(np.nanmax(vp)),
        "calibrated_value_min": float(np.nanmin(base.predict(v))),
        "calibrated_value_max": float(np.nanmax(base.predict(v))),
    }
    return ValueBellmanCalibrator(
        base=base,
        method=method,
        n_iterations=n_iterations,
        diagnostics=diagnostics,
        name=calibrator,
    )
