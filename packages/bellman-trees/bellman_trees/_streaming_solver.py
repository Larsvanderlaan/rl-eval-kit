from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg as splinalg


Array = np.ndarray


@dataclass(frozen=True)
class StreamingFeatureBatch:
    current_indices: Array
    current_values: Array
    next_indices: Array
    next_values: Array
    reward: Array
    weight: Array


@dataclass(frozen=True)
class StreamingBellmanSolveResult:
    theta: Array
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class StreamingOccupancyFeatureBatch:
    current_indices: Array
    current_values: Array
    next_indices: Array
    next_values: Array
    weight: Array


@dataclass(frozen=True)
class StreamingInitialFeatureBatch:
    indices: Array
    values: Array
    weight: Array


@dataclass(frozen=True)
class StreamingOccupancySolveResult:
    beta: Array
    diagnostics: dict[str, Any]


def solve_streaming_bellman(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    *,
    n_features: int,
    gamma: float,
    ridge: float,
    method: str = "auto",
    max_iter: int = 500,
    tol: float = 1e-4,
) -> StreamingBellmanSolveResult:
    """Solve a projected Bellman system from streamed sparse feature batches."""

    d = int(n_features)
    if d <= 0:
        raise ValueError("n_features must be positive.")
    if method == "auto":
        method = "streaming_direct" if d <= 4096 else "streaming_fqe"
    if method == "streaming_direct":
        return _solve_streaming_direct(make_batches, d, float(gamma), float(ridge), float(tol))
    if method == "streaming_fqe":
        return _solve_streaming_fqe(make_batches, d, float(gamma), float(ridge), int(max_iter), float(tol))
    if method == "streaming_lstd_iterative":
        return _solve_streaming_lstd_iterative(make_batches, d, float(gamma), float(ridge), int(max_iter), float(tol))
    raise ValueError("streaming method must be 'auto', 'streaming_direct', 'streaming_fqe', or 'streaming_lstd_iterative'.")


def solve_streaming_occupancy_ratio(
    make_batches: Callable[[], Iterable[StreamingOccupancyFeatureBatch]],
    make_initial_batches: Callable[[], Iterable[StreamingInitialFeatureBatch]],
    *,
    n_features: int,
    gamma: float,
    ridge: float = 1e-6,
    method: str = "auto",
    max_iter: int = 1000,
    tol: float = 1e-6,
    normalize: bool = True,
) -> StreamingOccupancySolveResult:
    """Solve nonnegative discounted occupancy-ratio moments from streamed leaf features."""

    d = int(n_features)
    if d <= 0:
        raise ValueError("n_features must be positive.")
    method_key = str(method)
    if method_key == "auto":
        method_key = "streaming_direct" if d <= 4096 else "streaming_fista"
    if method_key == "streaming_direct":
        return _solve_streaming_occupancy_direct(
            make_batches,
            make_initial_batches,
            d,
            float(gamma),
            float(ridge),
            float(tol),
            bool(normalize),
        )
    if method_key == "streaming_fista":
        return _solve_streaming_occupancy_fista(
            make_batches,
            make_initial_batches,
            d,
            float(gamma),
            float(ridge),
            int(max_iter),
            float(tol),
            bool(normalize),
        )
    raise ValueError("streaming occupancy method must be 'auto', 'streaming_direct', or 'streaming_fista'.")


def _batch_to_csr(indices: Array, values: Array, n_features: int) -> sparse.csr_matrix:
    idx = np.asarray(indices, dtype=np.int64)
    val = np.asarray(values, dtype=np.float64)
    if idx.shape != val.shape:
        raise ValueError("indices and values must have the same shape.")
    if idx.ndim != 2:
        raise ValueError("indices and values must be 2D.")
    n, k = idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = idx.reshape(-1)
    data = val.reshape(-1)
    keep = cols >= 0
    return sparse.csr_matrix((data[keep], (rows[keep], cols[keep])), shape=(n, int(n_features)))


def _solve_streaming_direct(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    n_features: int,
    gamma: float,
    ridge: float,
    tol: float,
) -> StreamingBellmanSolveResult:
    start_time = time.perf_counter()
    lhs = np.zeros((n_features, n_features), dtype=np.float64)
    rhs = np.zeros(n_features, dtype=np.float64)
    weight_sum = 0.0
    n_samples = 0
    nnz_current = 0
    nnz_next = 0
    for batch in make_batches():
        cur = _batch_to_csr(batch.current_indices, batch.current_values, n_features)
        nxt = _batch_to_csr(batch.next_indices, batch.next_values, n_features)
        w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        r = np.asarray(batch.reward, dtype=np.float64).reshape(-1)
        design = (cur - gamma * nxt).multiply(w[:, None])
        lhs += (cur.T @ design).toarray()
        rhs += np.asarray(cur.T @ (w * r), dtype=np.float64).reshape(-1)
        weight_sum += float(np.sum(w))
        n_samples += int(w.size)
        nnz_current += int(cur.nnz)
        nnz_next += int(nxt.nnz)
    accumulation_time = time.perf_counter() - start_time
    solve_start = time.perf_counter()
    if weight_sum <= 0.0:
        raise ValueError("streaming batches have non-positive total weight.")
    lhs = lhs / weight_sum
    rhs = rhs / weight_sum
    if ridge > 0.0:
        lhs.flat[:: n_features + 1] += ridge
    try:
        theta = np.linalg.solve(lhs, rhs)
        linear_solve = "solve"
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(lhs, rhs, rcond=tol)[0]
        linear_solve = "lstsq"
    residual = float(np.linalg.norm(lhs @ theta - rhs) / max(np.linalg.norm(rhs), 1e-12))
    linear_time = time.perf_counter() - solve_start
    return StreamingBellmanSolveResult(
        theta=np.asarray(theta, dtype=np.float64),
        diagnostics={
            "method": "streaming_direct",
            "linear_solve": linear_solve,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "weight_sum": float(weight_sum),
            "ridge": float(ridge),
            "gamma": float(gamma),
            "relative_residual": residual,
            "converged": bool(np.isfinite(residual)),
            "iterations": 1,
            "moment_accumulation_time": float(accumulation_time),
            "linear_solve_time": float(linear_time),
            "nnz_current": int(nnz_current),
            "nnz_next": int(nnz_next),
        },
    )


def _initial_source_from_batches(
    make_initial_batches: Callable[[], Iterable[StreamingInitialFeatureBatch]],
    n_features: int,
    gamma: float,
) -> tuple[Array, float, int, int]:
    source = np.zeros(n_features, dtype=np.float64)
    weight_sum = 0.0
    n_initial = 0
    nnz_initial = 0
    for batch in make_initial_batches():
        init = _batch_to_csr(batch.indices, batch.values, n_features)
        wi = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        source += np.asarray(init.T @ wi, dtype=np.float64).reshape(-1)
        weight_sum += float(np.sum(wi))
        n_initial += int(wi.size)
        nnz_initial += int(init.nnz)
    if weight_sum <= 0.0:
        raise ValueError("streaming initial batches have non-positive total weight.")
    return (1.0 - float(gamma)) * source / weight_sum, weight_sum, n_initial, nnz_initial


def _streaming_ratio_stats(
    make_batches: Callable[[], Iterable[StreamingOccupancyFeatureBatch]],
    beta: Array,
) -> dict[str, float | int]:
    weight_sum = 0.0
    weighted_mean = 0.0
    positive_weight_sum = 0.0
    positive_weight_sq_sum = 0.0
    min_ratio = np.inf
    max_ratio = -np.inf
    negative_count = 0
    n_samples = 0
    for batch in make_batches():
        w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        ratio = _row_dot(batch.current_indices, batch.current_values, beta)
        weight_sum += float(np.sum(w))
        weighted_mean += float(np.sum(w * ratio))
        positive = np.maximum(ratio, 0.0) * w
        positive_weight_sum += float(np.sum(positive))
        positive_weight_sq_sum += float(np.sum(positive**2))
        min_ratio = min(min_ratio, float(np.min(ratio)) if ratio.size else min_ratio)
        max_ratio = max(max_ratio, float(np.max(ratio)) if ratio.size else max_ratio)
        negative_count += int(np.sum(ratio < 0.0))
        n_samples += int(ratio.size)
    mean_ratio = weighted_mean / max(weight_sum, 1e-12)
    ess = positive_weight_sum**2 / max(positive_weight_sq_sum, 1e-12)
    return {
        "mean_ratio": float(mean_ratio),
        "min_ratio": float(min_ratio) if np.isfinite(min_ratio) else float("nan"),
        "max_ratio": float(max_ratio) if np.isfinite(max_ratio) else float("nan"),
        "negative_fraction": float(negative_count / max(n_samples, 1)),
        "ratio_ess_fraction": float(ess / max(n_samples, 1)),
    }


def _solve_streaming_occupancy_direct(
    make_batches: Callable[[], Iterable[StreamingOccupancyFeatureBatch]],
    make_initial_batches: Callable[[], Iterable[StreamingInitialFeatureBatch]],
    n_features: int,
    gamma: float,
    ridge: float,
    tol: float,
    normalize: bool,
) -> StreamingOccupancySolveResult:
    start_time = time.perf_counter()
    moment = np.zeros((n_features, n_features), dtype=np.float64)
    weight_sum = 0.0
    n_samples = 0
    nnz_current = 0
    nnz_next = 0
    for batch in make_batches():
        cur = _batch_to_csr(batch.current_indices, batch.current_values, n_features)
        nxt = _batch_to_csr(batch.next_indices, batch.next_values, n_features)
        w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        weighted_current = cur.multiply(w[:, None])
        moment += (cur.T @ weighted_current).toarray()
        moment -= float(gamma) * (nxt.T @ weighted_current).toarray()
        weight_sum += float(np.sum(w))
        n_samples += int(w.size)
        nnz_current += int(cur.nnz)
        nnz_next += int(nxt.nnz)
    if weight_sum <= 0.0:
        raise ValueError("streaming batches have non-positive total weight.")
    moment = moment / weight_sum
    source, initial_weight_sum, n_initial, nnz_initial = _initial_source_from_batches(
        make_initial_batches, n_features, gamma
    )
    accumulation_time = time.perf_counter() - start_time

    solve_start = time.perf_counter()
    if ridge > 0.0:
        system = np.vstack([moment, np.sqrt(ridge) * np.eye(n_features, dtype=np.float64)])
        target = np.concatenate([source, np.zeros(n_features, dtype=np.float64)])
    else:
        system = moment
        target = source
    out = optimize.lsq_linear(
        system,
        target,
        bounds=(0.0, np.inf),
        tol=float(tol),
        lsmr_tol=float(tol),
        max_iter=None,
    )
    beta = np.asarray(out.x, dtype=np.float64).reshape(-1)
    linear_time = time.perf_counter() - solve_start
    stats = _streaming_ratio_stats(make_batches, beta)
    raw_mean = float(stats["mean_ratio"])
    scale = raw_mean if normalize and np.isfinite(raw_mean) and abs(raw_mean) > 1e-12 else 1.0
    beta = beta / scale
    stats = _streaming_ratio_stats(make_batches, beta)
    residual = np.asarray(moment @ beta - source, dtype=np.float64).reshape(-1)
    objective = float(0.5 * np.dot(residual, residual) + 0.5 * ridge * np.dot(beta, beta))
    return StreamingOccupancySolveResult(
        beta=beta,
        diagnostics={
            "method": "streaming_direct",
            "solver": "streaming_direct",
            "linear_solve": "streaming_lsq_linear",
            "n_samples": int(n_samples),
            "n_initial": int(n_initial),
            "n_features": int(n_features),
            "sample_weight_sum": float(weight_sum),
            "initial_weight_sum": float(initial_weight_sum),
            "ridge": float(ridge),
            "gamma": float(gamma),
            "nonnegative": True,
            "iterations": int(out.nit),
            "converged": bool(out.success),
            "lsq_status": int(out.status),
            "lsq_success": bool(out.success),
            "lsq_cost": float(out.cost),
            "lsq_optimality": float(out.optimality),
            "objective": objective,
            "moment_violation_l2": float(np.linalg.norm(residual)),
            "moment_violation_mean_square": float(np.mean(residual**2)) if residual.size else 0.0,
            "raw_mean_ratio": float(raw_mean),
            "normalization_applied": bool(normalize),
            "normalization_scale": float(scale),
            "moment_accumulation_time": float(accumulation_time),
            "linear_solve_time": float(linear_time),
            "nnz_current": int(nnz_current),
            "nnz_next": int(nnz_next),
            "nnz_initial": int(nnz_initial),
            **stats,
        },
    )


@dataclass
class _StreamingOccupancyOperator:
    make_batches: Callable[[], Iterable[StreamingOccupancyFeatureBatch]]
    make_initial_batches: Callable[[], Iterable[StreamingInitialFeatureBatch]]
    n_features: int
    gamma: float
    ridge: float

    def __post_init__(self) -> None:
        self.source, self.initial_weight_sum, self.n_initial, self.nnz_initial = _initial_source_from_batches(
            self.make_initial_batches, self.n_features, self.gamma
        )
        self.weight_sum = 0.0
        self.n_samples = 0
        self.nnz_current = 0
        self.nnz_next = 0
        for batch in self.make_batches():
            w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
            self.weight_sum += float(np.sum(w))
            self.n_samples += int(w.size)
            self.nnz_current += int(np.asarray(batch.current_indices).size)
            self.nnz_next += int(np.asarray(batch.next_indices).size)
        if self.weight_sum <= 0.0:
            raise ValueError("streaming batches have non-positive total weight.")

    def matvec(self, beta: Array) -> Array:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        out = np.zeros(self.n_features, dtype=np.float64)
        for batch in self.make_batches():
            w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
            q = _row_dot(batch.current_indices, batch.current_values, b)
            cur_idx = np.asarray(batch.current_indices, dtype=np.int64)
            cur_val = np.asarray(batch.current_values, dtype=np.float64)
            nxt_idx = np.asarray(batch.next_indices, dtype=np.int64)
            nxt_val = np.asarray(batch.next_values, dtype=np.float64)
            np.add.at(out, cur_idx.reshape(-1), ((w * q)[:, None] * cur_val).reshape(-1))
            np.add.at(out, nxt_idx.reshape(-1), (-self.gamma * (w * q)[:, None] * nxt_val).reshape(-1))
        return out / self.weight_sum

    def rmatvec(self, u: Array) -> Array:
        v = np.asarray(u, dtype=np.float64).reshape(-1)
        out = np.zeros(self.n_features, dtype=np.float64)
        for batch in self.make_batches():
            w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
            delta_v = _row_dot(batch.current_indices, batch.current_values, v)
            delta_v -= self.gamma * _row_dot(batch.next_indices, batch.next_values, v)
            cur_idx = np.asarray(batch.current_indices, dtype=np.int64)
            cur_val = np.asarray(batch.current_values, dtype=np.float64)
            np.add.at(out, cur_idx.reshape(-1), ((w * delta_v)[:, None] * cur_val).reshape(-1))
        return out / self.weight_sum

    def residual(self, beta: Array) -> Array:
        return self.matvec(beta) - self.source

    def loss(self, beta: Array) -> float:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        moment = self.residual(b)
        return float(0.5 * np.dot(moment, moment) + 0.5 * self.ridge * np.dot(b, b))

    def loss_grad(self, beta: Array) -> tuple[float, Array]:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        moment = self.residual(b)
        grad = self.rmatvec(moment)
        if self.ridge > 0.0:
            grad = grad + self.ridge * b
        return float(0.5 * np.dot(moment, moment) + 0.5 * self.ridge * np.dot(b, b)), grad

    @staticmethod
    def projected_gradient_norm(beta: Array, grad: Array) -> float:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        g = np.asarray(grad, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(b - np.maximum(b - g, 0.0)))


def _streaming_initial_beta(op: _StreamingOccupancyOperator) -> Array:
    beta = np.ones(op.n_features, dtype=np.float64)
    stats = _streaming_ratio_stats(op.make_batches, beta)
    mean_ratio = float(stats["mean_ratio"])
    if np.isfinite(mean_ratio) and mean_ratio > 1e-12:
        beta = beta / mean_ratio
    return beta


def _estimate_streaming_lipschitz(op: _StreamingOccupancyOperator, *, n_iter: int = 8) -> float:
    rng = np.random.default_rng(19)
    vec = rng.normal(size=op.n_features)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return max(float(op.ridge), 1.0)
    vec /= norm
    eigenvalue = max(float(op.ridge), 0.0)
    for _ in range(max(1, int(n_iter))):
        out = op.rmatvec(op.matvec(vec))
        if op.ridge > 0.0:
            out = out + op.ridge * vec
        norm = float(np.linalg.norm(out))
        if norm <= 1e-30 or not np.isfinite(norm):
            break
        vec = out / norm
        eigenvalue = float(np.dot(vec, out))
    return max(eigenvalue, float(op.ridge), 1e-12)


def _solve_streaming_occupancy_fista(
    make_batches: Callable[[], Iterable[StreamingOccupancyFeatureBatch]],
    make_initial_batches: Callable[[], Iterable[StreamingInitialFeatureBatch]],
    n_features: int,
    gamma: float,
    ridge: float,
    max_iter: int,
    tol: float,
    normalize: bool,
) -> StreamingOccupancySolveResult:
    start_time = time.perf_counter()
    op = _StreamingOccupancyOperator(make_batches, make_initial_batches, n_features, gamma, ridge)
    setup_time = time.perf_counter() - start_time
    beta = _streaming_initial_beta(op)
    y = beta.copy()
    momentum = 1.0
    lipschitz = _estimate_streaming_lipschitz(op)
    step_lipschitz = lipschitz
    best_loss, best_grad = op.loss_grad(beta)
    initial_loss = float(best_loss)
    projected_gradient_norm = op.projected_gradient_norm(beta, best_grad)
    converged = False
    total_backtracks = 0
    fista_start = time.perf_counter()
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        y_loss, y_grad = op.loss_grad(y)
        local_lipschitz = max(step_lipschitz, 1e-12)
        for bt in range(20):
            candidate = np.maximum(y - y_grad / local_lipschitz, 0.0)
            diff = candidate - y
            candidate_loss = op.loss(candidate)
            bound = y_loss + float(np.dot(y_grad, diff)) + 0.5 * local_lipschitz * float(np.dot(diff, diff))
            if candidate_loss <= bound + 1e-12:
                break
            local_lipschitz *= 2.0
        total_backtracks += bt
        step_lipschitz = local_lipschitz
        candidate_loss, candidate_grad = op.loss_grad(candidate)
        projected_gradient_norm = op.projected_gradient_norm(candidate, candidate_grad)
        scale = max(1.0, float(np.linalg.norm(candidate)))
        if projected_gradient_norm <= tol * scale or float(np.linalg.norm(candidate - beta)) <= tol * scale:
            beta = candidate
            best_loss = candidate_loss
            best_grad = candidate_grad
            converged = True
            break
        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        if float(np.dot(candidate - beta, beta - y)) > 0.0:
            y = candidate.copy()
            momentum = 1.0
        else:
            y = candidate + ((momentum - 1.0) / new_momentum) * (candidate - beta)
            momentum = new_momentum
        beta = candidate
        best_loss = candidate_loss
        best_grad = candidate_grad
    fista_time = time.perf_counter() - fista_start
    stats = _streaming_ratio_stats(make_batches, beta)
    raw_mean = float(stats["mean_ratio"])
    scale = raw_mean if normalize and np.isfinite(raw_mean) and abs(raw_mean) > 1e-12 else 1.0
    beta = beta / scale
    final_loss, final_grad = op.loss_grad(beta)
    residual = op.residual(beta)
    stats = _streaming_ratio_stats(make_batches, beta)
    return StreamingOccupancySolveResult(
        beta=np.asarray(beta, dtype=np.float64),
        diagnostics={
            "method": "streaming_fista",
            "solver": "streaming_fista",
            "linear_solve": "streaming_projected_fista",
            "n_samples": int(op.n_samples),
            "n_initial": int(op.n_initial),
            "n_features": int(n_features),
            "sample_weight_sum": float(op.weight_sum),
            "initial_weight_sum": float(op.initial_weight_sum),
            "ridge": float(ridge),
            "gamma": float(gamma),
            "nonnegative": True,
            "iterations": int(iterations),
            "max_iter": int(max_iter),
            "converged": bool(converged),
            "objective": float(final_loss),
            "initial_objective": float(initial_loss),
            "objective_decreased": bool(final_loss <= initial_loss + 1e-12),
            "projected_gradient_norm": float(op.projected_gradient_norm(beta, final_grad)),
            "moment_violation_l2": float(np.linalg.norm(residual)),
            "moment_violation_mean_square": float(np.mean(residual**2)) if residual.size else 0.0,
            "raw_mean_ratio": float(raw_mean),
            "normalization_applied": bool(normalize),
            "normalization_scale": float(scale),
            "estimated_lipschitz": float(lipschitz),
            "final_lipschitz": float(step_lipschitz),
            "backtracking_steps": int(total_backtracks),
            "setup_time": float(setup_time),
            "linear_solve_time": float(fista_time),
            "nnz_current": int(op.nnz_current),
            "nnz_next": int(op.nnz_next),
            "nnz_initial": int(op.nnz_initial),
            **stats,
        },
    )


def _rhs_and_diag(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    n_features: int,
    ridge: float,
) -> tuple[Array, Array, float, int, int, int, float]:
    start_time = time.perf_counter()
    rhs = np.zeros(n_features, dtype=np.float64)
    diag = np.zeros(n_features, dtype=np.float64)
    weight_sum = 0.0
    n_samples = 0
    nnz_current = 0
    nnz_next = 0
    for batch in make_batches():
        w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        r = np.asarray(batch.reward, dtype=np.float64).reshape(-1)
        cur = _batch_to_csr(batch.current_indices, batch.current_values, n_features)
        rhs += np.asarray(cur.T @ (w * r), dtype=np.float64).reshape(-1)
        diag += np.asarray(cur.power(2).T @ w, dtype=np.float64).reshape(-1)
        weight_sum += float(np.sum(w))
        n_samples += int(w.size)
        nnz_current += int(cur.nnz)
        nnz_next += int(np.asarray(batch.next_indices).size)
    if weight_sum <= 0.0:
        raise ValueError("streaming batches have non-positive total weight.")
    rhs = rhs / weight_sum
    diag = diag / weight_sum
    if ridge > 0.0:
        diag = diag + ridge
    return rhs, np.maximum(diag, 1e-14), weight_sum, n_samples, nnz_current, nnz_next, time.perf_counter() - start_time


def _solve_streaming_fqe(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    n_features: int,
    gamma: float,
    ridge: float,
    max_iter: int,
    tol: float,
) -> StreamingBellmanSolveResult:
    rhs_reward, diag, weight_sum, n_samples, nnz_current, nnz_next, accumulation_time = _rhs_and_diag(
        make_batches, n_features, ridge
    )
    theta = np.zeros(n_features, dtype=np.float64)
    relaxation = min(0.05, max(0.005, 0.1 * (1.0 - gamma)))
    converged = False
    last_delta = float("inf")
    last_bound = float("inf")
    for iteration in range(1, max_iter + 1):
        iter_start = time.perf_counter()
        gradient = np.zeros(n_features, dtype=np.float64)
        for batch in make_batches():
            w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
            cur_idx = np.asarray(batch.current_indices, dtype=np.int64)
            cur_val = np.asarray(batch.current_values, dtype=np.float64)
            current_q = _row_dot(batch.current_indices, batch.current_values, theta)
            next_q = _row_dot(batch.next_indices, batch.next_values, theta)
            residual = np.asarray(batch.reward, dtype=np.float64).reshape(-1) + gamma * next_q - current_q
            contribution = (w * residual)[:, None] * cur_val / weight_sum
            np.add.at(gradient, cur_idx.reshape(-1), contribution.reshape(-1))
        theta_new = theta + relaxation * gradient / diag
        accumulation_time += time.perf_counter() - iter_start
        last_delta = _weighted_prediction_delta(make_batches, theta_new - theta, weight_sum)
        last_bound = float(gamma * last_delta / max(1.0 - gamma, 1e-12))
        theta = theta_new
        if last_bound <= tol:
            converged = True
            break
    return StreamingBellmanSolveResult(
        theta=np.asarray(theta, dtype=np.float64),
        diagnostics={
            "method": "streaming_fqe",
            "linear_solve": "preconditioned_streaming_fqe",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "weight_sum": float(weight_sum),
            "ridge": float(ridge),
            "gamma": float(gamma),
            "iterations": int(iteration),
            "max_iter": int(max_iter),
            "converged": bool(converged),
            "relaxation": float(relaxation),
            "last_prediction_delta": float(last_delta),
            "contraction_error_bound": float(last_bound),
            "moment_accumulation_time": float(accumulation_time),
            "nnz_current": int(nnz_current),
            "nnz_next": int(nnz_next),
        },
    )


def _solve_streaming_lstd_iterative(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    n_features: int,
    gamma: float,
    ridge: float,
    max_iter: int,
    tol: float,
) -> StreamingBellmanSolveResult:
    rhs_reward, _, weight_sum, n_samples, nnz_current, nnz_next, accumulation_time = _rhs_and_diag(
        make_batches, n_features, 0.0
    )
    matvec_time = 0.0

    def matvec(vec: Array) -> Array:
        nonlocal matvec_time
        start_time = time.perf_counter()
        out = np.zeros(n_features, dtype=np.float64)
        for batch in make_batches():
            cur = _batch_to_csr(batch.current_indices, batch.current_values, n_features)
            nxt = _batch_to_csr(batch.next_indices, batch.next_values, n_features)
            w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
            residual_feature = np.asarray(cur @ vec - gamma * (nxt @ vec), dtype=np.float64).reshape(-1)
            out += np.asarray(cur.T @ (w * residual_feature), dtype=np.float64).reshape(-1)
        out = out / weight_sum
        if ridge > 0.0:
            out = out + ridge * vec
        matvec_time += time.perf_counter() - start_time
        return out

    operator = splinalg.LinearOperator((n_features, n_features), matvec=matvec, dtype=np.float64)
    try:
        theta, info = splinalg.gmres(operator, rhs_reward, rtol=tol, atol=tol, maxiter=max_iter)
    except TypeError:
        theta, info = splinalg.gmres(operator, rhs_reward, tol=tol, maxiter=max_iter)
    residual = float(np.linalg.norm(matvec(theta) - rhs_reward) / max(np.linalg.norm(rhs_reward), 1e-12))
    return StreamingBellmanSolveResult(
        theta=np.asarray(theta, dtype=np.float64),
        diagnostics={
            "method": "streaming_lstd_iterative",
            "linear_solve": "gmres_streaming_matvec",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "weight_sum": float(weight_sum),
            "ridge": float(ridge),
            "gamma": float(gamma),
            "iterations": None if info < 0 else int(max_iter if info > 0 else 0),
            "gmres_info": int(info),
            "converged": bool(info == 0),
            "relative_residual": residual,
            "moment_accumulation_time": float(accumulation_time),
            "matvec_time": float(matvec_time),
            "nnz_current": int(nnz_current),
            "nnz_next": int(nnz_next),
        },
    )


def _row_dot(indices: Array, values: Array, theta: Array) -> Array:
    idx = np.asarray(indices, dtype=np.int64)
    val = np.asarray(values, dtype=np.float64)
    return np.sum(val * theta[idx], axis=1)


def _weighted_prediction_delta(
    make_batches: Callable[[], Iterable[StreamingFeatureBatch]],
    delta_theta: Array,
    weight_sum: float,
) -> float:
    total = 0.0
    for batch in make_batches():
        pred_delta = _row_dot(batch.current_indices, batch.current_values, delta_theta)
        w = np.asarray(batch.weight, dtype=np.float64).reshape(-1)
        total += float(np.sum(w * pred_delta**2))
    return float(np.sqrt(total / max(weight_sum, 1e-12)))
