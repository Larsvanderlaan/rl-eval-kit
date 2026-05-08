from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as splinalg


@dataclass(frozen=True)
class BellmanSolveResult:
    theta: np.ndarray
    diagnostics: dict[str, Any]


def _as_csr(name: str, value: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(value):
        mat = value.tocsr().astype(np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
        mat = sparse.csr_matrix(arr)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    return mat


def solve_projected_bellman(
    phi: sparse.spmatrix | np.ndarray,
    phi_next: sparse.spmatrix | np.ndarray,
    reward: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    gamma: float = 0.99,
    ridge: float = 1e-8,
    method: str = "direct",
    max_iter: int = 500,
    tol: float = 1e-8,
    dense_threshold: int = 2048,
    rank_diagnostics_threshold: int = 64,
    rank_tol: float = 1e-10,
) -> BellmanSolveResult:
    """Solve the weighted projected Bellman linear system."""

    current = _as_csr("phi", phi)
    nxt = _as_csr("phi_next", phi_next)
    if current.shape != nxt.shape:
        raise ValueError(f"phi and phi_next must have the same shape, got {current.shape} and {nxt.shape}.")
    n, d = current.shape
    r = np.asarray(reward, dtype=np.float64).reshape(-1)
    if r.shape[0] != n:
        raise ValueError(f"reward must have length {n}, got {r.shape[0]}.")
    if sample_weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"sample_weight must have length {n}, got {w.shape[0]}.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(w)):
        raise ValueError("reward and sample_weight must be finite.")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be nonnegative.")
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        raise ValueError("sample_weight must have positive sum.")
    design = (current - float(gamma) * nxt).multiply(w[:, None])
    lhs = (current.T @ design) / weight_sum
    rhs = np.asarray(current.T @ (w * r), dtype=np.float64).reshape(-1) / weight_sum
    if ridge > 0.0:
        lhs = lhs + float(ridge) * sparse.eye(d, format="csc", dtype=np.float64)
    else:
        lhs = lhs.tocsc()

    diagnostics: dict[str, Any] = {
        "n_samples": int(n),
        "n_features": int(d),
        "gamma": float(gamma),
        "ridge": float(ridge),
        "weight_sum": float(weight_sum),
        "solver": "dense" if d <= int(dense_threshold) else "sparse",
        "method": method,
    }

    if d == 0:
        raise ValueError("feature matrix has zero columns.")
    if method == "auto":
        method = "direct" if d <= int(dense_threshold) else "iterative"
        diagnostics["method"] = method
    if method == "iterative":
        return _solve_projected_bellman_iterative(
            current=current,
            nxt=nxt,
            reward=r,
            weights=w,
            weight_sum=weight_sum,
            gamma=float(gamma),
            ridge=float(ridge),
            max_iter=int(max_iter),
            tol=float(tol),
            dense_threshold=int(dense_threshold),
            diagnostics=diagnostics,
        )
    if method != "direct":
        raise ValueError("method must be 'direct', 'iterative', or 'auto'.")

    if d <= int(dense_threshold):
        dense = lhs.toarray()
        if d <= int(rank_diagnostics_threshold):
            rank = int(np.linalg.matrix_rank(dense, tol=rank_tol))
            diagnostics["rank"] = rank
            diagnostics["rank_deficient"] = bool(rank < d)
            diagnostics["condition_number"] = float(np.linalg.cond(dense)) if dense.size else float("nan")
            diagnostics["rank_diagnostics"] = "computed"
        else:
            diagnostics["rank_diagnostics"] = "skipped"
        try:
            theta = np.linalg.solve(dense, rhs)
            diagnostics["linear_solve"] = "solve"
        except np.linalg.LinAlgError:
            theta, _, rank, singular_values = np.linalg.lstsq(dense, rhs, rcond=rank_tol)
            diagnostics["rank"] = int(rank)
            diagnostics["rank_deficient"] = bool(int(rank) < d)
            if singular_values.size:
                diagnostics["condition_number"] = float(singular_values[0] / max(singular_values[-1], rank_tol))
            diagnostics["linear_solve"] = "lstsq"
        return BellmanSolveResult(theta=np.asarray(theta, dtype=np.float64), diagnostics=diagnostics)

    try:
        theta = splinalg.spsolve(lhs.tocsc(), rhs)
        diagnostics["linear_solve"] = "spsolve"
    except Exception:
        out = splinalg.lsqr(lhs.tocsr(), rhs, atol=rank_tol, btol=rank_tol)
        theta = out[0]
        diagnostics["linear_solve"] = "lsqr"
        diagnostics["lsqr_istop"] = int(out[1])
        diagnostics["lsqr_iterations"] = int(out[2])
    return BellmanSolveResult(theta=np.asarray(theta, dtype=np.float64), diagnostics=diagnostics)


def _solve_projected_bellman_iterative(
    *,
    current: sparse.csr_matrix,
    nxt: sparse.csr_matrix,
    reward: np.ndarray,
    weights: np.ndarray,
    weight_sum: float,
    gamma: float,
    ridge: float,
    max_iter: int,
    tol: float,
    dense_threshold: int,
    diagnostics: dict[str, Any],
) -> BellmanSolveResult:
    """Projected FQE iteration on a fixed feature map."""

    n, d = current.shape
    gram = (current.T @ current.multiply(weights[:, None])) / weight_sum
    cross = (current.T @ nxt.multiply(weights[:, None])) / weight_sum
    rhs_reward = np.asarray(current.T @ (weights * reward), dtype=np.float64).reshape(-1) / weight_sum
    if ridge > 0.0:
        gram = gram + ridge * sparse.eye(d, format="csc", dtype=np.float64)
    else:
        gram = gram.tocsc()

    if d <= dense_threshold:
        gram_dense = gram.toarray()
        if d <= 64:
            rank = int(np.linalg.matrix_rank(gram_dense))
            diagnostics["gram_rank"] = rank
            diagnostics["gram_rank_deficient"] = bool(rank < d)
            diagnostics["gram_condition_number"] = float(np.linalg.cond(gram_dense)) if gram_dense.size else float("nan")
            diagnostics["gram_rank_diagnostics"] = "computed"
        else:
            diagnostics["gram_rank_diagnostics"] = "skipped"
        try:
            cho = linalg.cho_factor(gram_dense, lower=True, check_finite=False)

            def solve_gram(rhs: np.ndarray) -> np.ndarray:
                return linalg.cho_solve(cho, rhs, check_finite=False)

            diagnostics["gram_solve"] = "cholesky"
        except linalg.LinAlgError:
            lu, piv = linalg.lu_factor(gram_dense, check_finite=False)

            def solve_gram(rhs: np.ndarray) -> np.ndarray:
                return linalg.lu_solve((lu, piv), rhs, check_finite=False)

            diagnostics["gram_solve"] = "lu"
    else:
        gram_csc = gram.tocsc()
        try:
            factorized = splinalg.factorized(gram_csc)

            def solve_gram(rhs: np.ndarray) -> np.ndarray:
                return np.asarray(factorized(rhs), dtype=np.float64)

            diagnostics["gram_solve"] = "sparse_factorized"
        except Exception:

            def solve_gram(rhs: np.ndarray) -> np.ndarray:
                return splinalg.lsqr(gram_csc, rhs, atol=tol, btol=tol, iter_lim=max(100, 2 * d))[0]

            diagnostics["gram_solve"] = "lsqr"

    theta = np.zeros(d, dtype=np.float64)
    converged = False
    last_delta = float("inf")
    last_bound = float("inf")
    for iteration in range(1, max_iter + 1):
        theta_new = solve_gram(rhs_reward + gamma * np.asarray(cross @ theta).reshape(-1))
        diff = theta_new - theta
        pred_delta = np.asarray(current @ diff).reshape(-1)
        last_delta = float(np.sqrt(np.sum(weights * pred_delta**2) / weight_sum))
        last_bound = float(gamma * last_delta / max(1.0 - gamma, 1e-12))
        theta = np.asarray(theta_new, dtype=np.float64)
        if last_bound <= tol:
            converged = True
            break

    diagnostics["linear_solve"] = "fixed_feature_fqe"
    diagnostics["iterations"] = int(iteration)
    diagnostics["max_iter"] = int(max_iter)
    diagnostics["converged"] = bool(converged)
    diagnostics["last_prediction_delta"] = float(last_delta)
    diagnostics["contraction_error_bound"] = float(last_bound)
    return BellmanSolveResult(theta=theta, diagnostics=diagnostics)
