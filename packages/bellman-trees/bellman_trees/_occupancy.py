from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg as splinalg

from ._base import SerializableEstimatorMixin
from ._data import BellmanTransitionData
from ._features import average_next_features
from ._native_tree import _CandidateSplit, _Node, BellmanAggregationTree
from ._solver import _as_csr
from ._weights import effective_sample_size, stabilize_weights


Array = np.ndarray


@dataclass(frozen=True)
class DiscountedOccupancySolveResult:
    beta: np.ndarray
    diagnostics: dict[str, float | int | str | bool]


@dataclass
class _FlowBalanceOperator:
    current: sparse.csr_matrix
    nxt: sparse.csr_matrix
    initial: sparse.csr_matrix
    sample_weight: np.ndarray
    initial_weight: np.ndarray
    gamma: float
    ridge: float

    def __post_init__(self) -> None:
        self.current = self.current.tocsr()
        self.nxt = self.nxt.tocsr()
        self.initial = self.initial.tocsr()
        self.sample_weight = np.asarray(self.sample_weight, dtype=np.float64).reshape(-1)
        self.initial_weight = np.asarray(self.initial_weight, dtype=np.float64).reshape(-1)
        self.weight_sum = float(np.sum(self.sample_weight))
        self.initial_weight_sum = float(np.sum(self.initial_weight))
        self.n_samples, self.n_features = self.current.shape
        self.source = (
            (1.0 - float(self.gamma))
            * np.asarray(self.initial.T @ self.initial_weight, dtype=np.float64).reshape(-1)
            / self.initial_weight_sum
        )

    def matvec(self, beta: Array) -> Array:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        q = np.asarray(self.current @ b, dtype=np.float64).reshape(-1)
        weighted = self.sample_weight * q
        out = np.asarray(self.current.T @ weighted, dtype=np.float64).reshape(-1)
        out -= float(self.gamma) * np.asarray(self.nxt.T @ weighted, dtype=np.float64).reshape(-1)
        return out / self.weight_sum

    def rmatvec(self, u: Array) -> Array:
        v = np.asarray(u, dtype=np.float64).reshape(-1)
        delta_v = np.asarray(self.current @ v, dtype=np.float64).reshape(-1)
        delta_v -= float(self.gamma) * np.asarray(self.nxt @ v, dtype=np.float64).reshape(-1)
        return np.asarray(self.current.T @ (self.sample_weight * delta_v), dtype=np.float64).reshape(-1) / self.weight_sum

    def residual(self, beta: Array) -> Array:
        return self.matvec(beta) - self.source

    def loss(self, beta: Array) -> float:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        moment = self.residual(b)
        return float(0.5 * np.dot(moment, moment) + 0.5 * float(self.ridge) * np.dot(b, b))

    def loss_grad(self, beta: Array) -> tuple[float, Array]:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        moment = self.residual(b)
        grad = self.rmatvec(moment)
        if self.ridge > 0.0:
            grad = grad + float(self.ridge) * b
        loss = float(0.5 * np.dot(moment, moment) + 0.5 * float(self.ridge) * np.dot(b, b))
        return loss, np.asarray(grad, dtype=np.float64)

    def projected_gradient_norm(self, beta: Array, grad: Array) -> float:
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        g = np.asarray(grad, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(b - np.maximum(b - g, 0.0)))

    def linear_operator(self) -> splinalg.LinearOperator:
        return splinalg.LinearOperator(
            shape=(self.n_features, self.n_features),
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )

    def augmented_operator(self) -> splinalg.LinearOperator:
        d = self.n_features
        if self.ridge <= 0.0:
            return self.linear_operator()
        root_ridge = float(np.sqrt(float(self.ridge)))

        def matvec(beta: Array) -> Array:
            b = np.asarray(beta, dtype=np.float64).reshape(-1)
            return np.concatenate([self.matvec(b), root_ridge * b])

        def rmatvec(u: Array) -> Array:
            v = np.asarray(u, dtype=np.float64).reshape(-1)
            return self.rmatvec(v[:d]) + root_ridge * v[d:]

        return splinalg.LinearOperator(shape=(2 * d, d), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    def augmented_target(self) -> Array:
        if self.ridge <= 0.0:
            return self.source
        return np.concatenate([self.source, np.zeros(self.n_features, dtype=np.float64)])

    def materialized_moment_matrix(self) -> sparse.csr_matrix:
        weighted_current = self.current.multiply(self.sample_weight[:, None])
        return ((self.current.T @ weighted_current) - float(self.gamma) * (self.nxt.T @ weighted_current)).tocsr() / self.weight_sum


def _as_weight(name: str, value: Array | None, n: int) -> Array:
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
    return np.ascontiguousarray(arr)


def _source_moment(
    phi_initial: sparse.spmatrix | Array,
    initial_weight: Array | None,
    *,
    gamma: float,
) -> Array:
    initial = _as_csr("phi_initial", phi_initial)
    wi = _as_weight("initial_weight", initial_weight, initial.shape[0])
    return (1.0 - float(gamma)) * np.asarray(initial.T @ wi, dtype=np.float64).reshape(-1) / float(np.sum(wi))


def _validated_warm_start(warm_start: Array | None, d: int, *, nonnegative: bool) -> Array | None:
    if warm_start is None:
        return None
    beta = np.asarray(warm_start, dtype=np.float64).reshape(-1)
    if beta.shape[0] != d:
        raise ValueError(f"warm_start must have length {d}, got {beta.shape[0]}.")
    if not np.all(np.isfinite(beta)):
        raise ValueError("warm_start contains non-finite values.")
    if nonnegative:
        beta = np.maximum(beta, 0.0)
    return np.ascontiguousarray(beta, dtype=np.float64)


def _initial_fista_beta(op: _FlowBalanceOperator, warm_start: Array | None) -> Array:
    if warm_start is not None:
        beta = np.asarray(warm_start, dtype=np.float64).reshape(-1).copy()
    else:
        beta = np.ones(op.n_features, dtype=np.float64)
    beta = np.maximum(beta, 0.0)
    ratio = np.asarray(op.current @ beta, dtype=np.float64).reshape(-1)
    mean_ratio = float(np.sum(op.sample_weight * ratio) / op.weight_sum)
    if np.isfinite(mean_ratio) and mean_ratio > 1e-12:
        beta = beta / mean_ratio
    return beta


def _estimate_lipschitz(op: _FlowBalanceOperator, *, n_iter: int = 20) -> float:
    d = op.n_features
    if d == 0:
        return 1.0
    rng = np.random.default_rng(17)
    vec = rng.normal(size=d)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return max(float(op.ridge), 1.0)
    vec /= norm
    eigenvalue = max(float(op.ridge), 0.0)
    for _ in range(max(1, int(n_iter))):
        out = op.rmatvec(op.matvec(vec))
        if op.ridge > 0.0:
            out = out + float(op.ridge) * vec
        norm = float(np.linalg.norm(out))
        if norm <= 1e-30 or not np.isfinite(norm):
            break
        vec = out / norm
        eigenvalue = float(np.dot(vec, out))
    return max(float(eigenvalue), float(op.ridge), 1e-12)


def _solve_projected_fista(
    op: _FlowBalanceOperator,
    *,
    tol: float,
    max_iter: int,
    warm_start: Array | None,
    fista_restart: bool,
    backtracking: bool,
) -> tuple[Array, dict[str, float | int | bool | str]]:
    beta = _initial_fista_beta(op, warm_start)
    y = beta.copy()
    momentum = 1.0
    lipschitz = _estimate_lipschitz(op)
    step_lipschitz = lipschitz
    best_loss, best_grad = op.loss_grad(beta)
    initial_loss = float(best_loss)
    total_backtracks = 0
    converged = False
    projected_grad_norm = op.projected_gradient_norm(beta, best_grad)
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        y_loss, y_grad = op.loss_grad(y)
        local_lipschitz = max(step_lipschitz, 1e-12)
        for bt in range(50):
            candidate = np.maximum(y - y_grad / local_lipschitz, 0.0)
            diff = candidate - y
            candidate_loss = op.loss(candidate)
            quadratic_bound = y_loss + float(np.dot(y_grad, diff)) + 0.5 * local_lipschitz * float(np.dot(diff, diff))
            if not backtracking or candidate_loss <= quadratic_bound + 1e-12:
                break
            local_lipschitz *= 2.0
        total_backtracks += bt
        step_lipschitz = local_lipschitz

        if candidate_loss > best_loss and backtracking:
            y = beta.copy()
            momentum = 1.0
            y_loss, y_grad = op.loss_grad(y)
            local_lipschitz = max(step_lipschitz, 1e-12)
            for bt in range(50):
                candidate = np.maximum(y - y_grad / local_lipschitz, 0.0)
                diff = candidate - y
                candidate_loss = op.loss(candidate)
                quadratic_bound = y_loss + float(np.dot(y_grad, diff)) + 0.5 * local_lipschitz * float(np.dot(diff, diff))
                if candidate_loss <= quadratic_bound + 1e-12:
                    break
                local_lipschitz *= 2.0
            total_backtracks += bt
            step_lipschitz = local_lipschitz

        candidate_loss, candidate_grad = op.loss_grad(candidate)
        projected_grad_norm = op.projected_gradient_norm(candidate, candidate_grad)
        scale = max(1.0, float(np.linalg.norm(candidate)))
        beta_step = float(np.linalg.norm(candidate - beta))
        if projected_grad_norm <= float(tol) * scale or beta_step <= float(tol) * scale:
            beta = candidate
            best_loss = candidate_loss
            best_grad = candidate_grad
            converged = True
            break

        new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        if fista_restart and float(np.dot(candidate - beta, beta - y)) > 0.0:
            y = candidate.copy()
            momentum = 1.0
        else:
            y = candidate + ((momentum - 1.0) / new_momentum) * (candidate - beta)
            momentum = new_momentum
        beta = candidate
        best_loss = candidate_loss
        best_grad = candidate_grad

    diagnostics: dict[str, float | int | bool | str] = {
        "linear_solve": "projected_fista",
        "iterations": int(iterations),
        "converged": bool(converged),
        "projected_gradient_norm": float(projected_grad_norm),
        "objective": float(best_loss),
        "initial_objective": float(initial_loss),
        "objective_decreased": bool(best_loss <= initial_loss + 1e-12),
        "estimated_lipschitz": float(lipschitz),
        "final_lipschitz": float(step_lipschitz),
        "backtracking_steps": int(total_backtracks),
    }
    return np.asarray(beta, dtype=np.float64), diagnostics


def discounted_flow_moment_from_ratio(
    phi: sparse.spmatrix | Array,
    phi_next: sparse.spmatrix | Array,
    phi_initial: sparse.spmatrix | Array,
    ratio: Array,
    sample_weight: Array | None = None,
    initial_weight: Array | None = None,
    *,
    gamma: float = 0.99,
) -> Array:
    """Moment imbalance for a supplied discounted occupancy-ratio prediction."""

    current = _as_csr("phi", phi)
    nxt = _as_csr("phi_next", phi_next)
    initial = _as_csr("phi_initial", phi_initial)
    if current.shape != nxt.shape:
        raise ValueError(f"phi and phi_next must have the same shape, got {current.shape} and {nxt.shape}.")
    if initial.shape[1] != current.shape[1]:
        raise ValueError(f"phi_initial must have {current.shape[1]} columns, got {initial.shape[1]}.")
    rho = np.asarray(ratio, dtype=np.float64).reshape(-1)
    if rho.shape[0] != current.shape[0]:
        raise ValueError(f"ratio must have length {current.shape[0]}, got {rho.shape[0]}.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("ratio contains non-finite values.")
    w = _as_weight("sample_weight", sample_weight, current.shape[0])
    source = _source_moment(initial, initial_weight, gamma=float(gamma))
    delta = current - float(gamma) * nxt
    empirical = np.asarray(delta.T @ (w * rho), dtype=np.float64).reshape(-1) / float(np.sum(w))
    return empirical - source


def discounted_flow_moment(
    phi: sparse.spmatrix | Array,
    phi_next: sparse.spmatrix | Array,
    phi_initial: sparse.spmatrix | Array,
    beta: Array,
    sample_weight: Array | None = None,
    initial_weight: Array | None = None,
    *,
    gamma: float = 0.99,
) -> Array:
    """Moment imbalance for a linear ratio model rho(x)=phi(x)^T beta."""

    current = _as_csr("phi", phi)
    rho = np.asarray(current @ np.asarray(beta, dtype=np.float64).reshape(-1), dtype=np.float64).reshape(-1)
    return discounted_flow_moment_from_ratio(
        current,
        phi_next,
        phi_initial,
        rho,
        sample_weight,
        initial_weight,
        gamma=float(gamma),
    )


def solve_discounted_occupancy_ratio(
    phi: sparse.spmatrix | Array,
    phi_next: sparse.spmatrix | Array,
    phi_initial: sparse.spmatrix | Array,
    sample_weight: Array | None = None,
    initial_weight: Array | None = None,
    *,
    gamma: float = 0.99,
    ridge: float = 1e-6,
    nonnegative: bool = True,
    solver: str = "auto",
    normalize: bool = True,
    dense_threshold: int = 2048,
    rank_tol: float = 1e-10,
    tol: float = 1e-6,
    max_iter: int | None = None,
    warm_start: Array | None = None,
    fista_restart: bool = True,
    backtracking: bool = True,
) -> DiscountedOccupancySolveResult:
    """Solve regularized projected discounted flow-balance equations."""

    started = time.perf_counter()
    current = _as_csr("phi", phi)
    nxt = _as_csr("phi_next", phi_next)
    initial = _as_csr("phi_initial", phi_initial)
    if current.shape != nxt.shape:
        raise ValueError(f"phi and phi_next must have the same shape, got {current.shape} and {nxt.shape}.")
    if initial.shape[1] != current.shape[1]:
        raise ValueError(f"phi_initial must have {current.shape[1]} columns, got {initial.shape[1]}.")
    n, d = current.shape
    if d == 0:
        raise ValueError("feature matrix has zero columns.")
    w = _as_weight("sample_weight", sample_weight, n)
    wi = _as_weight("initial_weight", initial_weight, initial.shape[0])
    weight_sum = float(np.sum(w))
    initial_weight_sum = float(np.sum(wi))
    warm = _validated_warm_start(warm_start, d, nonnegative=bool(nonnegative))
    op = _FlowBalanceOperator(
        current=current,
        nxt=nxt,
        initial=initial,
        sample_weight=w,
        initial_weight=wi,
        gamma=float(gamma),
        ridge=float(ridge),
    )

    solver_key = str(solver).lower()
    if solver_key == "auto":
        if bool(nonnegative):
            solver_key = "lsq_linear" if d <= int(dense_threshold) else "fista"
        else:
            solver_key = "dense_lstsq" if d <= int(dense_threshold) else "lsmr"
    valid_solvers = {"lsq_linear", "fista", "lsmr", "dense_lstsq", "lstsq"}
    if solver_key not in valid_solvers:
        raise ValueError(f"solver must be one of {sorted(valid_solvers | {'auto'})}, got {solver!r}.")
    if solver_key == "lsmr" and bool(nonnegative):
        raise ValueError("solver='lsmr' is unconstrained; use solver='fista' or solver='auto' when nonnegative=True.")
    if solver_key in {"dense_lstsq", "lstsq"} and bool(nonnegative):
        raise ValueError("dense_lstsq/lstsq are unconstrained; use lsq_linear or fista when nonnegative=True.")

    diagnostics: dict[str, float | int | str | bool] = {
        "n_samples": int(n),
        "n_initial": int(initial.shape[0]),
        "n_features": int(d),
        "gamma": float(gamma),
        "ridge": float(ridge),
        "nonnegative": bool(nonnegative),
        "sample_weight_sum": float(weight_sum),
        "initial_weight_sum": float(initial_weight_sum),
        "solver": str(solver_key),
        "requested_solver": str(solver),
        "tol": float(tol),
        "max_iter": -1 if max_iter is None else int(max_iter),
        "warm_start_used": bool(warm is not None),
    }

    if solver_key == "fista":
        beta, solver_diagnostics = _solve_projected_fista(
            op,
            tol=float(tol),
            max_iter=1000 if max_iter is None else int(max_iter),
            warm_start=warm,
            fista_restart=bool(fista_restart),
            backtracking=bool(backtracking),
        )
        diagnostics.update(solver_diagnostics)
    elif solver_key == "lsq_linear":
        moment_matrix = op.materialized_moment_matrix()
        if ridge > 0.0:
            system = sparse.vstack(
                [moment_matrix.tocsr(), np.sqrt(float(ridge)) * sparse.eye(d, format="csr", dtype=np.float64)],
                format="csr",
            )
            target = np.concatenate([op.source, np.zeros(d, dtype=np.float64)])
        else:
            system = moment_matrix.tocsr()
            target = op.source
        opt_system = system if d > int(dense_threshold) else _as_csr("system", system).toarray()
        out = optimize.lsq_linear(
            opt_system,
            target,
            bounds=(0.0, np.inf),
            tol=float(tol),
            lsmr_tol=float(tol),
            max_iter=max_iter,
        )
        beta = out.x
        diagnostics["linear_solve"] = "lsq_linear"
        diagnostics["iterations"] = int(out.nit)
        diagnostics["converged"] = bool(out.success)
        diagnostics["lsq_status"] = int(out.status)
        diagnostics["lsq_success"] = bool(out.success)
        diagnostics["lsq_cost"] = float(out.cost)
        diagnostics["lsq_optimality"] = float(out.optimality)
        diagnostics["lsq_iterations"] = int(out.nit)
        diagnostics["objective"] = float(op.loss(beta))
        _, grad = op.loss_grad(beta)
        diagnostics["projected_gradient_norm"] = float(op.projected_gradient_norm(beta, grad))
        if d <= int(dense_threshold):
            dense = np.asarray(opt_system, dtype=np.float64)
            rank = int(np.linalg.matrix_rank(dense, tol=rank_tol))
            diagnostics["rank"] = rank
            diagnostics["rank_deficient"] = bool(rank < d)
            diagnostics["condition_number"] = float(np.linalg.cond(dense)) if dense.size else float("nan")
    elif solver_key in {"dense_lstsq", "lstsq"}:
        moment_matrix = op.materialized_moment_matrix()
        if ridge > 0.0:
            system = sparse.vstack(
                [moment_matrix.tocsr(), np.sqrt(float(ridge)) * sparse.eye(d, format="csr", dtype=np.float64)],
                format="csr",
            )
            target = np.concatenate([op.source, np.zeros(d, dtype=np.float64)])
        else:
            system = moment_matrix.tocsr()
            target = op.source
        dense = _as_csr("system", system).toarray()
        rank = int(np.linalg.matrix_rank(dense, tol=rank_tol))
        diagnostics["rank"] = rank
        diagnostics["rank_deficient"] = bool(rank < d)
        diagnostics["condition_number"] = float(np.linalg.cond(dense)) if dense.size else float("nan")
        beta = np.linalg.lstsq(dense, target, rcond=rank_tol)[0]
        diagnostics["linear_solve"] = "lstsq"
        diagnostics["iterations"] = 1
        diagnostics["converged"] = True
        diagnostics["objective"] = float(op.loss(beta))
        diagnostics["projected_gradient_norm"] = float("nan")
    else:
        out = splinalg.lsmr(
            op.augmented_operator(),
            op.augmented_target(),
            atol=float(tol),
            btol=float(tol),
            maxiter=max_iter,
        )
        beta = out[0]
        diagnostics["linear_solve"] = "lsmr"
        diagnostics["iterations"] = int(out[2])
        diagnostics["converged"] = bool(out[1] in (1, 2))
        diagnostics["lsmr_istop"] = int(out[1])
        diagnostics["lsmr_iterations"] = int(out[2])
        diagnostics["lsmr_normr"] = float(out[3])
        diagnostics["lsmr_normar"] = float(out[4])
        diagnostics["lsmr_conda"] = float(out[6])
        diagnostics["objective"] = float(op.loss(beta))
        diagnostics["projected_gradient_norm"] = float("nan")

    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    raw_ratio = np.asarray(current @ beta, dtype=np.float64).reshape(-1)
    raw_mean = float(np.sum(w * raw_ratio) / weight_sum)
    diagnostics["raw_mean_ratio"] = raw_mean
    diagnostics["raw_min_ratio"] = float(np.min(raw_ratio)) if raw_ratio.size else float("nan")
    diagnostics["raw_max_ratio"] = float(np.max(raw_ratio)) if raw_ratio.size else float("nan")
    diagnostics["raw_negative_fraction"] = float(np.mean(raw_ratio < 0.0)) if raw_ratio.size else 0.0
    diagnostics["normalization_applied"] = bool(normalize)
    if normalize and np.isfinite(raw_mean) and abs(raw_mean) > rank_tol:
        beta = beta / raw_mean
        diagnostics["normalization_scale"] = raw_mean
    else:
        diagnostics["normalization_scale"] = 1.0
    ratio = np.asarray(current @ beta, dtype=np.float64).reshape(-1)
    diagnostics["mean_ratio"] = float(np.sum(w * ratio) / weight_sum)
    diagnostics["min_ratio"] = float(np.min(ratio)) if ratio.size else float("nan")
    diagnostics["max_ratio"] = float(np.max(ratio)) if ratio.size else float("nan")
    diagnostics["negative_fraction"] = float(np.mean(ratio < 0.0)) if ratio.size else 0.0
    moment = discounted_flow_moment(current, nxt, initial, beta, w, wi, gamma=float(gamma))
    final_objective, final_grad = op.loss_grad(beta)
    diagnostics["objective"] = float(final_objective)
    if bool(nonnegative):
        diagnostics["projected_gradient_norm"] = float(op.projected_gradient_norm(beta, final_grad))
    diagnostics["moment_violation_l2"] = float(np.linalg.norm(moment))
    diagnostics["moment_violation_mean_square"] = float(np.mean(moment**2)) if moment.size else 0.0
    positive_weights = np.maximum(ratio, 0.0) * w
    diagnostics["ratio_ess_fraction"] = float(effective_sample_size(positive_weights) / max(positive_weights.size, 1))
    diagnostics["elapsed_seconds"] = float(time.perf_counter() - started)
    return DiscountedOccupancySolveResult(beta=beta, diagnostics=diagnostics)


def _as_reference_features(name: str, value: Array, p: int) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim == 2:
        if arr.shape[1] != p:
            raise ValueError(f"{name} must have {p} columns, got {arr.shape[1]}.")
    elif arr.ndim == 3:
        if arr.shape[2] != p:
            raise ValueError(f"{name} must have trailing dimension {p}, got {arr.shape[2]}.")
    else:
        raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr)


class DiscountedOccupancyRatioTree(SerializableEstimatorMixin):
    """Flow-balancing tree for discounted state-action occupancy ratios."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        max_depth: int = 4,
        max_leaves: int = 16,
        max_bins: int = 32,
        min_samples_leaf: int = 20,
        min_weighted_leaf_mass: float = 1e-8,
        min_leaf_ess: float = 5.0,
        complexity_penalty: float = 0.0,
        min_improvement: float = 1e-10,
        ridge: float = 1e-6,
        nonnegative: bool = True,
        solver: str = "auto",
        solver_tol: float = 1e-6,
        solver_max_iter: int = 1000,
        split_score_mode: str = "aggregated_flow",
        honest: bool = True,
        estimation_fraction: float = 0.35,
        growth_score_fraction: float = 0.35,
        weight_clip_quantile: float | None = 0.995,
        max_weight: float | None = None,
        weight_uniform_mix: float = 0.0,
        target_ess_fraction: float | None = None,
        ratio_clip_min: float | None = 0.0,
        ratio_clip_max: float | None = None,
        min_ratio_ess_fraction: float | None = None,
        ratio_ess_penalty: float = 0.0,
        negative_ratio_penalty: float = 1.0,
        random_state: int | None = None,
        feature_indices: Array | None = None,
    ) -> None:
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_weighted_leaf_mass = min_weighted_leaf_mass
        self.min_leaf_ess = min_leaf_ess
        self.complexity_penalty = complexity_penalty
        self.min_improvement = min_improvement
        self.ridge = ridge
        self.nonnegative = nonnegative
        self.solver = solver
        self.solver_tol = solver_tol
        self.solver_max_iter = solver_max_iter
        self.split_score_mode = split_score_mode
        self.honest = honest
        self.estimation_fraction = estimation_fraction
        self.growth_score_fraction = growth_score_fraction
        self.weight_clip_quantile = weight_clip_quantile
        self.max_weight = max_weight
        self.weight_uniform_mix = weight_uniform_mix
        self.target_ess_fraction = target_ess_fraction
        self.ratio_clip_min = ratio_clip_min
        self.ratio_clip_max = ratio_clip_max
        self.min_ratio_ess_fraction = min_ratio_ess_fraction
        self.ratio_ess_penalty = ratio_ess_penalty
        self.negative_ratio_penalty = negative_ratio_penalty
        self.random_state = random_state
        self.feature_indices = feature_indices

    def fit(
        self,
        X: Array,
        X_next: Array,
        X_initial: Array,
        sample_weight: Array | None = None,
        initial_weight: Array | None = None,
    ) -> "DiscountedOccupancyRatioTree":
        data = BellmanTransitionData(
            X=X,
            reward=np.zeros(np.asarray(X).shape[0], dtype=np.float64),
            X_next=X_next,
            sample_weight=sample_weight,
        )
        self.X_initial_ = _as_reference_features("X_initial", X_initial, data.n_features)
        self.initial_weight_ = _as_weight("initial_weight", initial_weight, self.X_initial_.shape[0])
        self.n_input_features_ = data.n_features
        weights = stabilize_weights(
            data.sample_weight,
            data.n_samples,
            max_weight=self.max_weight,
            clip_quantile=self.weight_clip_quantile,
            uniform_mix=self.weight_uniform_mix,
            target_ess_fraction=self.target_ess_fraction,
        )
        self.weight_diagnostics_ = weights.diagnostics
        w = weights.values
        rng = np.random.default_rng(self.random_state)
        grow_fit_idx, grow_score_idx, estimation_idx = self._split_roles(data.n_samples, rng)
        self.root_ = _Node(node_id=0, depth=0)
        self._next_node_id = 1
        self.split_history_: list[dict[str, float | int | bool]] = []
        feature_indices = self._candidate_features(data.n_features, rng)

        while len(self._leaf_nodes()) < int(self.max_leaves):
            current_beta = self._fit_ratio_for_transform(data, w, grow_fit_idx, candidate=None).beta
            best: tuple[float, dict[str, Any], _CandidateSplit] | None = None
            raw_fit = self._apply_raw(data.X[grow_fit_idx])
            raw_score = self._apply_raw(data.X[grow_score_idx])
            for leaf in self._leaf_nodes():
                if leaf.depth >= int(self.max_depth):
                    continue
                local_idx = grow_fit_idx[raw_fit == leaf.node_id]
                if local_idx.size < 2 * int(self.min_samples_leaf):
                    continue
                for feature in feature_indices:
                    for threshold in self._thresholds(data.X[local_idx, feature]):
                        candidate = _CandidateSplit(
                            node_id=leaf.node_id,
                            feature_index=int(feature),
                            threshold=float(threshold),
                            left_id=self._next_node_id,
                            right_id=self._next_node_id + 1,
                        )
                        if not self._candidate_is_admissible(data, w, local_idx, candidate):
                            continue
                        stats = self._candidate_gain(
                            data,
                            w,
                            grow_fit_idx,
                            grow_score_idx,
                            current_beta,
                            candidate,
                            raw_score=raw_score,
                        )
                        gain = float(stats["penalized_gain"])
                        if best is None or gain > best[0]:
                            best = (gain, stats, candidate)
            if best is None:
                break
            best_gain, best_stats, best_candidate = best
            accepted = best_gain > float(self.min_improvement)
            self.split_history_.append(
                {
                    "leaf_node_id": int(best_candidate.node_id),
                    "feature_index": int(best_candidate.feature_index),
                    "threshold": float(best_candidate.threshold),
                    "baseline_loss": float(best_stats["baseline_loss"]),
                    "candidate_loss": float(best_stats["candidate_loss"]),
                    "gain": float(best_stats["gain"]),
                    "penalized_gain": float(best_gain),
                    "accepted": bool(accepted),
                }
            )
            if not accepted:
                break
            self._commit_split(best_candidate)

        self.leaf_node_ids_ = self._leaf_node_ids()
        if estimation_idx.size < max(2, int(self.min_samples_leaf)):
            estimation_idx = np.arange(data.n_samples, dtype=np.int64)
        solve = self._fit_ratio_for_transform(data, w, estimation_idx, candidate=None)
        phi = self.transform(data.X[estimation_idx])
        beta, post = self._postprocess_beta(solve.beta, phi, w[estimation_idx])
        self.beta_ = beta
        self.theta_ = beta
        self.solver_info_ = {**solve.diagnostics, **post}
        self.feature_info_ = {
            "n_leaves": int(phi.shape[1]),
            "n_input_features": int(data.n_features),
            "feature_indices": None if self.feature_indices is None else np.asarray(self.feature_indices).tolist(),
            "honest": bool(self.honest),
            "solver": str(self.solver),
            "split_score_mode": str(self.split_score_mode),
            "n_growth_fit": int(grow_fit_idx.size),
            "n_growth_score": int(grow_score_idx.size),
            "n_estimation": int(estimation_idx.size),
            "n_initial": int(self.X_initial_.shape[0]),
        }
        ratio_fit = np.asarray(phi @ self.beta_, dtype=np.float64).reshape(-1)
        ratio_weights = np.maximum(ratio_fit, 0.0) * w[estimation_idx]
        self.diagnostics_ = {
            **self.weight_diagnostics_,
            **post,
            "n_splits": int(len([row for row in self.split_history_ if row["accepted"]])),
            "n_leaves": int(phi.shape[1]),
            "ratio_ess_fraction": float(effective_sample_size(ratio_weights) / max(ratio_weights.size, 1)),
        }
        return self

    def predict_ratio(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        return np.asarray(self.transform(X_eval) @ self.beta_, dtype=np.float64).reshape(-1)

    def predict(self, X_eval: Array) -> Array:
        return self.predict_ratio(X_eval)

    def transform(self, X: Array) -> sparse.csr_matrix:
        return self._tree_impl().transform(self, X)

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        return average_next_features(self.transform, X_next)

    def transform_initial(self, X_initial: Array) -> sparse.csr_matrix:
        self._check_tree()
        return average_next_features(
            self.transform,
            _as_reference_features("X_initial", X_initial, self.n_input_features_),
        )

    def _fit_ratio_for_transform(
        self,
        data: BellmanTransitionData,
        weights: Array,
        fit_idx: Array,
        candidate: _CandidateSplit | None,
        warm_start: Array | None = None,
    ) -> DiscountedOccupancySolveResult:
        if self.split_score_mode == "aggregated_flow":
            return self._fit_ratio_aggregated(data, weights, fit_idx, candidate, warm_start=warm_start)
        if self.split_score_mode != "sparse_flow":
            raise ValueError("split_score_mode must be 'aggregated_flow' or 'sparse_flow'.")
        transform = lambda x: self._transform_with_candidate(x, candidate)
        phi = transform(data.X[fit_idx])
        phi_next = average_next_features(transform, data.X_next[fit_idx])
        phi_initial = average_next_features(transform, self.X_initial_)
        return solve_discounted_occupancy_ratio(
            phi,
            phi_next,
            phi_initial,
            weights[fit_idx],
            self.initial_weight_,
            gamma=float(self.gamma),
            ridge=float(self.ridge),
            nonnegative=bool(self.nonnegative),
            solver=str(self.solver),
            normalize=True,
            tol=float(self.solver_tol),
            max_iter=int(self.solver_max_iter),
            warm_start=warm_start,
        )

    def _fit_ratio_aggregated(
        self,
        data: BellmanTransitionData,
        weights: Array,
        fit_idx: Array,
        candidate: _CandidateSplit | None,
        warm_start: Array | None = None,
    ) -> DiscountedOccupancySolveResult:
        leaf_ids = self._candidate_leaf_ids(candidate)
        x_fit = data.X[fit_idx]
        w_fit = np.asarray(weights[fit_idx], dtype=np.float64).reshape(-1)
        current_assign = self._candidate_assignments(x_fit, candidate, leaf_ids)
        flow_matrix, source, behavior_mass = self._aggregate_flow_system(
            current_assign=current_assign,
            X_next=data.X_next[fit_idx],
            sample_weight=w_fit,
            candidate=candidate,
            leaf_ids=leaf_ids,
        )
        return self._solve_aggregated_system(
            flow_matrix,
            source,
            behavior_mass,
            warm_start=warm_start,
        )

    def _aggregate_flow_system(
        self,
        *,
        current_assign: Array,
        X_next: Array,
        sample_weight: Array,
        candidate: _CandidateSplit | None,
        leaf_ids: list[int],
    ) -> tuple[sparse.csr_matrix, Array, Array]:
        k = len(leaf_ids)
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        weight_sum = float(np.sum(w))
        current_mass = np.bincount(np.asarray(current_assign, dtype=np.int64), weights=w, minlength=k).astype(np.float64)
        transition = self._next_current_matrix(X_next, candidate, leaf_ids, current_assign, w)
        flow_matrix = sparse.diags(current_mass / weight_sum, format="csr") - float(self.gamma) * (transition / weight_sum)
        source = self._initial_source_by_leaf(candidate, leaf_ids)
        return flow_matrix.tocsr(), source, current_mass / weight_sum

    def _solve_aggregated_system(
        self,
        flow_matrix: sparse.csr_matrix,
        source: Array,
        behavior_mass: Array,
        *,
        warm_start: Array | None,
    ) -> DiscountedOccupancySolveResult:
        started = time.perf_counter()
        k = flow_matrix.shape[1]
        warm = _validated_warm_start(warm_start, k, nonnegative=bool(self.nonnegative))
        if bool(self.nonnegative):
            system = flow_matrix
            target = np.asarray(source, dtype=np.float64).reshape(-1)
            if self.ridge > 0.0:
                system = sparse.vstack(
                    [system, np.sqrt(float(self.ridge)) * sparse.eye(k, format="csr", dtype=np.float64)],
                    format="csr",
                )
                target = np.concatenate([target, np.zeros(k, dtype=np.float64)])
            out = optimize.lsq_linear(
                system if k > 2048 else system.toarray(),
                target,
                bounds=(0.0, np.inf),
                tol=float(self.solver_tol),
                lsmr_tol=float(self.solver_tol),
                max_iter=int(self.solver_max_iter),
            )
            beta = np.asarray(out.x, dtype=np.float64).reshape(-1)
            iterations = int(out.nit)
            converged = bool(out.success)
            linear_solve = "aggregated_lsq_linear"
            projected_gradient_norm = float(out.optimality)
        else:
            system = flow_matrix
            target = np.asarray(source, dtype=np.float64).reshape(-1)
            if self.ridge > 0.0:
                system = sparse.vstack(
                    [system, np.sqrt(float(self.ridge)) * sparse.eye(k, format="csr", dtype=np.float64)],
                    format="csr",
                )
                target = np.concatenate([target, np.zeros(k, dtype=np.float64)])
            out = splinalg.lsmr(system, target, atol=float(self.solver_tol), btol=float(self.solver_tol), maxiter=int(self.solver_max_iter))
            beta = np.asarray(out[0], dtype=np.float64).reshape(-1)
            iterations = int(out[2])
            converged = bool(out[1] in (1, 2))
            linear_solve = "aggregated_lsmr"
            projected_gradient_norm = float("nan")
        if warm is not None and beta.shape == warm.shape and not np.all(np.isfinite(beta)):
            beta = warm.copy()
        raw_ratio_mean = float(np.dot(behavior_mass, beta))
        if np.isfinite(raw_ratio_mean) and abs(raw_ratio_mean) > 1e-12:
            beta = beta / raw_ratio_mean
        ratio = beta
        moment = np.asarray(flow_matrix @ beta, dtype=np.float64).reshape(-1) - np.asarray(source, dtype=np.float64).reshape(-1)
        objective = float(0.5 * np.dot(moment, moment) + 0.5 * float(self.ridge) * np.dot(beta, beta))
        positive_weights = np.maximum(ratio, 0.0) * np.asarray(behavior_mass, dtype=np.float64).reshape(-1)
        diagnostics: dict[str, float | int | str | bool] = {
            "n_samples": -1,
            "n_initial": int(self.X_initial_.shape[0]),
            "n_features": int(k),
            "gamma": float(self.gamma),
            "ridge": float(self.ridge),
            "nonnegative": bool(self.nonnegative),
            "solver": str(self.solver),
            "linear_solve": linear_solve,
            "iterations": iterations,
            "converged": converged,
            "projected_gradient_norm": projected_gradient_norm,
            "objective": objective,
            "moment_violation_l2": float(np.linalg.norm(moment)),
            "moment_violation_mean_square": float(np.mean(moment**2)) if moment.size else 0.0,
            "mean_ratio": float(np.dot(behavior_mass, ratio)),
            "min_ratio": float(np.min(ratio)) if ratio.size else float("nan"),
            "max_ratio": float(np.max(ratio)) if ratio.size else float("nan"),
            "negative_fraction": float(np.mean(ratio < 0.0)) if ratio.size else 0.0,
            "ratio_ess_fraction": float(effective_sample_size(positive_weights) / max(positive_weights.size, 1)),
            "elapsed_seconds": float(time.perf_counter() - started),
            "warm_start_used": bool(warm is not None),
            "aggregated_flow": True,
        }
        return DiscountedOccupancySolveResult(beta=beta, diagnostics=diagnostics)

    def _candidate_gain(
        self,
        data: BellmanTransitionData,
        weights: Array,
        fit_idx: Array,
        score_idx: Array,
        current_beta: Array,
        candidate: _CandidateSplit,
        raw_score: Array | None = None,
    ) -> dict[str, float]:
        if self.split_score_mode == "aggregated_flow":
            return self._candidate_gain_aggregated(
                data,
                weights,
                fit_idx,
                score_idx,
                current_beta,
                candidate,
                raw_score=raw_score,
            )
        current_transform = lambda x: self._transform_with_candidate(x, None)
        candidate_transform = lambda x: self._transform_with_candidate(x, candidate)
        candidate_warm = self._candidate_warm_start(current_beta, candidate)
        candidate_solve = self._fit_ratio_for_transform(data, weights, fit_idx, candidate, warm_start=candidate_warm)
        phi_current_score = current_transform(data.X[score_idx])
        current_ratio = np.asarray(phi_current_score @ current_beta, dtype=np.float64).reshape(-1)
        phi_candidate_score = candidate_transform(data.X[score_idx])
        phi_candidate_next_score = average_next_features(candidate_transform, data.X_next[score_idx])
        phi_candidate_initial = average_next_features(candidate_transform, self.X_initial_)
        candidate_beta, _ = self._postprocess_beta(
            candidate_solve.beta,
            candidate_transform(data.X[fit_idx]),
            weights[fit_idx],
        )
        candidate_ratio = np.asarray(phi_candidate_score @ candidate_beta, dtype=np.float64).reshape(-1)
        baseline_loss = self._flow_loss(
            phi_candidate_score,
            phi_candidate_next_score,
            phi_candidate_initial,
            current_ratio,
            weights[score_idx],
        )
        candidate_loss = self._flow_loss(
            phi_candidate_score,
            phi_candidate_next_score,
            phi_candidate_initial,
            candidate_ratio,
            weights[score_idx],
        )
        dimension_penalty = float(self.complexity_penalty) * max(
            0,
            int(phi_candidate_score.shape[1] - phi_current_score.shape[1]),
        )
        gain = float(baseline_loss - candidate_loss)
        return {
            "baseline_loss": float(baseline_loss),
            "candidate_loss": float(candidate_loss),
            "gain": gain,
            "dimension_penalty": float(dimension_penalty),
            "penalized_gain": float(gain - dimension_penalty),
        }

    def _candidate_warm_start(self, current_beta: Array, candidate: _CandidateSplit) -> Array:
        parent_leaf_ids = self._leaf_node_ids()
        parent_map = {leaf_id: pos for pos, leaf_id in enumerate(parent_leaf_ids)}
        candidate_leaf_ids = self._candidate_leaf_ids(candidate)
        warm = np.zeros(len(candidate_leaf_ids), dtype=np.float64)
        split_value = float(np.asarray(current_beta, dtype=np.float64).reshape(-1)[parent_map[candidate.node_id]])
        for pos, leaf_id in enumerate(candidate_leaf_ids):
            warm[pos] = float(current_beta[parent_map[leaf_id]]) if leaf_id in parent_map else split_value
        return warm

    def _candidate_gain_aggregated(
        self,
        data: BellmanTransitionData,
        weights: Array,
        fit_idx: Array,
        score_idx: Array,
        current_beta: Array,
        candidate: _CandidateSplit,
        raw_score: Array | None = None,
    ) -> dict[str, float]:
        candidate_warm = self._candidate_warm_start(current_beta, candidate)
        candidate_solve = self._fit_ratio_aggregated(data, weights, fit_idx, candidate, warm_start=candidate_warm)
        parent_leaf_ids = self._candidate_leaf_ids(None)
        candidate_leaf_ids = self._candidate_leaf_ids(candidate)
        x_score = data.X[score_idx]
        score_weights = np.asarray(weights[score_idx], dtype=np.float64).reshape(-1)
        parent_assign = self._candidate_assignments(x_score, None, parent_leaf_ids, base_raw=raw_score)
        candidate_assign = self._candidate_assignments(x_score, candidate, candidate_leaf_ids, base_raw=raw_score)
        current_ratio = np.asarray(current_beta, dtype=np.float64).reshape(-1)[parent_assign]
        candidate_ratio = np.asarray(candidate_solve.beta, dtype=np.float64).reshape(-1)[candidate_assign]
        baseline_loss = self._flow_loss_aggregated(
            data.X_next[score_idx],
            score_weights,
            current_ratio,
            candidate,
            candidate_leaf_ids,
            candidate_assign,
        )
        candidate_loss = self._flow_loss_aggregated(
            data.X_next[score_idx],
            score_weights,
            candidate_ratio,
            candidate,
            candidate_leaf_ids,
            candidate_assign,
        )
        dimension_penalty = float(self.complexity_penalty) * max(0, len(candidate_leaf_ids) - len(parent_leaf_ids))
        gain = float(baseline_loss - candidate_loss)
        return {
            "baseline_loss": float(baseline_loss),
            "candidate_loss": float(candidate_loss),
            "gain": gain,
            "dimension_penalty": float(dimension_penalty),
            "penalized_gain": float(gain - dimension_penalty),
        }

    def _candidate_leaf_ids(self, candidate: _CandidateSplit | None) -> list[int]:
        leaf_ids = self._leaf_node_ids()
        if candidate is None:
            return leaf_ids
        return sorted([leaf_id for leaf_id in leaf_ids if leaf_id != candidate.node_id] + [candidate.left_id, candidate.right_id])

    def _raw_with_candidate(self, X: Array, candidate: _CandidateSplit | None, base_raw: Array | None = None) -> Array:
        x = np.asarray(X, dtype=np.float64)
        raw = np.asarray(base_raw, dtype=np.int64).reshape(-1).copy() if base_raw is not None else self._apply_raw(x)
        if candidate is None:
            return raw
        mask = raw == candidate.node_id
        if np.any(mask):
            go_left = x[mask, candidate.feature_index] <= candidate.threshold
            masked_pos = np.nonzero(mask)[0]
            raw[masked_pos[go_left]] = candidate.left_id
            raw[masked_pos[~go_left]] = candidate.right_id
        return raw

    def _candidate_assignments(
        self,
        X: Array,
        candidate: _CandidateSplit | None,
        leaf_ids: list[int],
        base_raw: Array | None = None,
    ) -> Array:
        raw = self._raw_with_candidate(X, candidate, base_raw=base_raw)
        mapping = {leaf_id: pos for pos, leaf_id in enumerate(leaf_ids)}
        out = np.array([mapping[int(leaf_id)] for leaf_id in raw], dtype=np.int64)
        return out

    def _weighted_leaf_counts(
        self,
        X: Array,
        candidate: _CandidateSplit | None,
        leaf_ids: list[int],
        row_weights: Array,
    ) -> Array:
        arr = np.asarray(X, dtype=np.float64)
        weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        if arr.ndim == 2:
            assignments = self._candidate_assignments(arr, candidate, leaf_ids)
            return np.bincount(assignments, weights=weights, minlength=len(leaf_ids)).astype(np.float64)
        if arr.ndim != 3:
            raise ValueError(f"expected 2D or 3D feature array, got shape {arr.shape}.")
        n, m, p = arr.shape
        flat = arr.reshape(n * m, p)
        assignments = self._candidate_assignments(flat, candidate, leaf_ids)
        repeated_weights = np.repeat(weights / float(m), m)
        return np.bincount(assignments, weights=repeated_weights, minlength=len(leaf_ids)).astype(np.float64)

    def _next_current_matrix(
        self,
        X_next: Array,
        candidate: _CandidateSplit | None,
        leaf_ids: list[int],
        current_assign: Array,
        row_weights: Array,
    ) -> sparse.csr_matrix:
        xp = np.asarray(X_next, dtype=np.float64)
        current = np.asarray(current_assign, dtype=np.int64).reshape(-1)
        weights = np.asarray(row_weights, dtype=np.float64).reshape(-1)
        k = len(leaf_ids)
        if xp.ndim == 2:
            next_assign = self._candidate_assignments(xp, candidate, leaf_ids)
            return sparse.csr_matrix((weights, (next_assign, current)), shape=(k, k), dtype=np.float64)
        if xp.ndim != 3:
            raise ValueError(f"X_next must be 2D or 3D, got shape {xp.shape}.")
        n, m, p = xp.shape
        flat = xp.reshape(n * m, p)
        next_assign = self._candidate_assignments(flat, candidate, leaf_ids)
        cols = np.repeat(current, m)
        values = np.repeat(weights / float(m), m)
        return sparse.csr_matrix((values, (next_assign, cols)), shape=(k, k), dtype=np.float64)

    def _initial_source_by_leaf(self, candidate: _CandidateSplit | None, leaf_ids: list[int]) -> Array:
        counts = self._weighted_leaf_counts(self.X_initial_, candidate, leaf_ids, self.initial_weight_)
        return (1.0 - float(self.gamma)) * counts / float(np.sum(self.initial_weight_))

    def _flow_loss_aggregated(
        self,
        X_next: Array,
        sample_weight: Array,
        ratio: Array,
        candidate: _CandidateSplit | None,
        leaf_ids: list[int],
        current_assign: Array,
    ) -> float:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        rho = np.asarray(ratio, dtype=np.float64).reshape(-1)
        weighted_ratio = w * rho
        weight_sum = float(np.sum(w))
        current_mass = np.bincount(np.asarray(current_assign, dtype=np.int64), weights=weighted_ratio, minlength=len(leaf_ids))
        next_mass = self._weighted_leaf_counts(X_next, candidate, leaf_ids, weighted_ratio)
        moment = current_mass / weight_sum
        moment -= float(self.gamma) * next_mass / weight_sum
        moment -= self._initial_source_by_leaf(candidate, leaf_ids)
        loss = float(np.mean(moment**2)) if moment.size else 0.0
        if self.negative_ratio_penalty > 0.0:
            loss += float(self.negative_ratio_penalty) * float(np.mean(np.minimum(rho, 0.0) ** 2))
        if self.min_ratio_ess_fraction is not None and self.ratio_ess_penalty > 0.0:
            positive = np.maximum(rho, 0.0) * w
            ess_fraction = effective_sample_size(positive) / max(positive.size, 1)
            shortfall = max(0.0, float(self.min_ratio_ess_fraction) - float(ess_fraction))
            loss += float(self.ratio_ess_penalty) * shortfall**2
        return loss

    def _flow_loss(
        self,
        phi: sparse.csr_matrix,
        phi_next: sparse.csr_matrix,
        phi_initial: sparse.csr_matrix,
        ratio: Array,
        sample_weight: Array,
    ) -> float:
        moment = discounted_flow_moment_from_ratio(
            phi,
            phi_next,
            phi_initial,
            ratio,
            sample_weight,
            self.initial_weight_,
            gamma=float(self.gamma),
        )
        loss = float(np.mean(moment**2)) if moment.size else 0.0
        rho = np.asarray(ratio, dtype=np.float64).reshape(-1)
        if self.negative_ratio_penalty > 0.0:
            loss += float(self.negative_ratio_penalty) * float(np.mean(np.minimum(rho, 0.0) ** 2))
        if self.min_ratio_ess_fraction is not None and self.ratio_ess_penalty > 0.0:
            positive = np.maximum(rho, 0.0) * np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            ess_fraction = effective_sample_size(positive) / max(positive.size, 1)
            shortfall = max(0.0, float(self.min_ratio_ess_fraction) - float(ess_fraction))
            loss += float(self.ratio_ess_penalty) * shortfall**2
        return loss

    def _postprocess_beta(self, beta: Array, phi_ref: sparse.spmatrix, weights: Array) -> tuple[Array, dict[str, float | bool]]:
        out = np.asarray(beta, dtype=np.float64).reshape(-1).copy()
        raw_ratio = np.asarray(phi_ref @ out, dtype=np.float64).reshape(-1)
        clipped = False
        if self.ratio_clip_min is not None:
            floor = float(self.ratio_clip_min)
            if np.any(out < floor):
                out = np.maximum(out, floor)
                clipped = True
        if self.ratio_clip_max is not None:
            ceiling = float(self.ratio_clip_max)
            if np.any(out > ceiling):
                out = np.minimum(out, ceiling)
                clipped = True
        ratio = np.asarray(phi_ref @ out, dtype=np.float64).reshape(-1)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        mean_ratio = float(np.sum(w * ratio) / max(np.sum(w), 1e-12))
        fallback = False
        if not np.isfinite(mean_ratio) or mean_ratio <= 1e-12:
            out = np.ones_like(out)
            ratio = np.asarray(phi_ref @ out, dtype=np.float64).reshape(-1)
            mean_ratio = float(np.sum(w * ratio) / max(np.sum(w), 1e-12))
            fallback = True
        out = out / mean_ratio
        final_ratio = np.asarray(phi_ref @ out, dtype=np.float64).reshape(-1)
        diagnostics = {
            "postprocess_clipped": bool(clipped),
            "postprocess_fallback_uniform": bool(fallback),
            "pre_clip_min_ratio": float(np.min(raw_ratio)) if raw_ratio.size else float("nan"),
            "pre_clip_max_ratio": float(np.max(raw_ratio)) if raw_ratio.size else float("nan"),
            "postprocess_mean_ratio": float(np.sum(w * final_ratio) / max(np.sum(w), 1e-12)),
            "postprocess_min_ratio": float(np.min(final_ratio)) if final_ratio.size else float("nan"),
            "postprocess_max_ratio": float(np.max(final_ratio)) if final_ratio.size else float("nan"),
        }
        return out, diagnostics

    def _split_roles(self, n: int, rng: np.random.Generator) -> tuple[Array, Array, Array]:
        return self._tree_impl()._split_roles(self, n, rng)

    def _candidate_features(self, p: int, rng: np.random.Generator) -> Array:
        return self._tree_impl()._candidate_features(self, p, rng)

    def _thresholds(self, values: Array) -> Array:
        return self._tree_impl()._thresholds(self, values)

    def _candidate_is_admissible(
        self,
        data: BellmanTransitionData,
        weights: Array,
        leaf_indices: Array,
        candidate: _CandidateSplit,
    ) -> bool:
        return self._tree_impl()._candidate_is_admissible(self, data, weights, leaf_indices, candidate)

    def _child_ok(self, weights: Array) -> bool:
        return self._tree_impl()._child_ok(self, weights)

    def _transform_with_candidate(self, X: Array, candidate: _CandidateSplit | None) -> sparse.csr_matrix:
        return self._tree_impl()._transform_with_candidate(self, X, candidate)

    def _commit_split(self, candidate: _CandidateSplit) -> None:
        self._tree_impl()._commit_split(self, candidate)

    def _apply_raw(self, X: Array) -> Array:
        return self._tree_impl()._apply_raw(self, X)

    def _leaf_nodes(self) -> list[_Node]:
        return self._tree_impl()._leaf_nodes(self)

    def _leaf_node_ids(self) -> list[int]:
        return self._tree_impl()._leaf_node_ids(self)

    def _find_node(self, node_id: int) -> _Node:
        return self._tree_impl()._find_node(self, node_id)

    def _check_tree(self) -> None:
        self._tree_impl()._check_tree(self)

    def _check_is_fitted(self) -> None:
        self._check_tree()
        if not hasattr(self, "beta_"):
            raise RuntimeError("Estimator is not fitted.")

    @staticmethod
    def _tree_impl() -> type[BellmanAggregationTree]:
        return BellmanAggregationTree
