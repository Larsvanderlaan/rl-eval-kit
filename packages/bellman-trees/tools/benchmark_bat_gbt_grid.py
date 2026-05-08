from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bellman_trees import BellmanHistogramGradientBoostingRegressor


@dataclass(frozen=True)
class Case:
    name: str
    n: int
    p: int
    n_estimators: int
    max_depth: int
    max_bins: int = 64
    subsample: float = 0.7
    max_samples_per_tree: int | None = None
    colsample_bytree: float = 0.7
    colsample_bynode: float = 0.8
    solver_method: str = "auto"
    feature_storage: str = "auto"
    hash_dim: int = 65_536


QUICK_CASES = [
    Case("small_lowdim", 5_000, 8, 50, 2, colsample_bytree=0.8),
    Case("medium_lowdim", 10_000, 8, 100, 2, colsample_bytree=0.8),
    Case("medium_widedim", 10_000, 64, 100, 2, colsample_bytree=0.4),
    Case("medium_deeper", 10_000, 32, 100, 3, colsample_bytree=0.5),
    Case("large_widedim", 25_000, 128, 100, 2, colsample_bytree=0.25),
]

RL_CASES = [
    *QUICK_CASES,
    Case("large_lowdim", 25_000, 16, 100, 2, colsample_bytree=0.8),
    Case("large_deeper", 25_000, 32, 100, 3, colsample_bytree=0.5),
    Case("many_leaf_features", 10_000, 32, 300, 3, colsample_bytree=0.5),
]

STRESS_CASES = [
    *RL_CASES,
    Case("hundredk_moderate", 100_000, 32, 100, 3, colsample_bytree=0.5),
    Case("hundredk_wide", 100_000, 128, 100, 2, colsample_bytree=0.25),
    Case("million_target", 1_000_000, 64, 100, 2, colsample_bytree=0.5, max_samples_per_tree=200_000),
    Case(
        "million_hashed",
        1_000_000,
        64,
        300,
        3,
        colsample_bytree=0.5,
        max_samples_per_tree=200_000,
        feature_storage="hashed",
    ),
]

MILLION_CASES = [
    Case("million_target", 1_000_000, 64, 100, 2, colsample_bytree=0.5, max_samples_per_tree=200_000),
]


def make_problem(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    noise = rng.normal(size=(n, p)).astype(np.float64)
    X_next = (0.70 * X + 0.20 * noise).astype(np.float64)
    signal = 1.0 + 0.4 * X[:, 0] - 0.2 * X[:, 1] ** 2
    if p > 2:
        signal += 0.25 * (X[:, 2] > 0.0)
    if p > 8:
        signal += 0.1 * np.sin(X[:, 7])
    if p > 32:
        signal += 0.05 * X[:, 31] * X[:, 3]
    reward = signal + 0.05 * rng.normal(size=n)
    return X, reward.astype(np.float64), X_next


def _estimated_sparse_pair_mb(n: int, n_trees: int) -> float:
    nnz = n * n_trees
    # Approximate two CSR matrices: current and next. SciPy usually uses
    # float64 data and int32 indices/indptr for these sizes.
    one = nnz * (8 + 4) + (n + 1) * 4
    return float(2 * one / 1_000_000)


def run_case(case: Case, seed: int) -> dict[str, float | int | str | bool]:
    X, reward, X_next = make_problem(case.n, case.p, seed)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=0.8,
        n_estimators=case.n_estimators,
        learning_rate=0.05,
        max_depth=case.max_depth,
        max_bins=case.max_bins,
        min_samples_leaf=max(20, case.n // 5000),
        min_child_weight=max(1.0, case.n / 20_000.0),
        subsample=case.subsample,
        max_samples_per_tree=case.max_samples_per_tree,
        colsample_bytree=case.colsample_bytree,
        colsample_bynode=case.colsample_bynode,
        solver_method=case.solver_method,
        feature_storage=case.feature_storage,
        hash_dim=case.hash_dim,
        solver_max_iter=300,
        solver_tol=1e-6,
        random_state=seed,
    )
    fit_start = time.perf_counter()
    model.fit(X, reward, X_next)
    fit_time = time.perf_counter() - fit_start

    eval_n = min(case.n, 20_000)
    apply_start = time.perf_counter()
    leaves = model.apply(X[:eval_n])
    apply_time = time.perf_counter() - apply_start
    transform_start = time.perf_counter()
    phi = model.transform(X[:eval_n])
    transform_time = time.perf_counter() - transform_start
    predict_start = time.perf_counter()
    pred = model.predict(X[:eval_n])
    predict_time = time.perf_counter() - predict_start

    n_trees = int(model.feature_info_["n_trees"])
    n_features = int(model.feature_info_["n_features"])
    return {
        "case": case.name,
        "n": case.n,
        "p": case.p,
        "trees": n_trees,
        "depth": case.max_depth,
        "bins": case.max_bins,
        "features": n_features,
        "feature_storage": str(model.feature_info_["feature_storage"]),
        "hash_dim": "" if model.feature_info_["hash_dim"] is None else int(model.feature_info_["hash_dim"]),
        "fit_s": fit_time,
        "binning_s": float(model.diagnostics_["binning_time"]),
        "boosting_s": float(model.diagnostics_["boosting_time"]),
        "feature_build_s": float(model.diagnostics_["feature_build_time"]),
        "solve_s": float(model.diagnostics_["bellman_solve_time"]),
        "apply_rows_per_s": eval_n / max(apply_time, 1e-12),
        "transform_rows_per_s": eval_n / max(transform_time, 1e-12),
        "predict_rows_per_s": eval_n / max(predict_time, 1e-12),
        "raw_train_loss": float(model.diagnostics_["raw_train_loss"]),
        "solver_method": str(model.solver_info_["method"]),
        "linear_solve": str(model.solver_info_["linear_solve"]),
        "solver_iterations": int(model.solver_info_.get("iterations", 0)),
        "converged": bool(model.solver_info_.get("converged", True)),
        "phi_nnz_eval": int(phi.nnz),
        "leaves_checksum": int(np.sum(leaves) % 1_000_000),
        "pred_mean": float(np.mean(pred)),
        "estimated_model_memory_mb": float(model.diagnostics_["estimated_memory_mb"]),
        "estimated_sparse_pair_mb": _estimated_sparse_pair_mb(case.n, n_trees),
    }


def _cases_for_preset(name: str) -> list[Case]:
    if name == "quick":
        return QUICK_CASES
    if name == "rl":
        return RL_CASES
    if name == "stress":
        return STRESS_CASES
    if name == "million":
        return MILLION_CASES
    raise ValueError(f"unknown preset {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scalability grid for Bellman histogram GBT.")
    parser.add_argument("--preset", choices=["quick", "rl", "stress", "million"], default="quick")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    cases = _cases_for_preset(args.preset)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    rows = []
    for i, case in enumerate(cases):
        print(f"running {case.name} n={case.n} p={case.p} trees={case.n_estimators} depth={case.max_depth}", flush=True)
        row = run_case(case, args.seed + i)
        rows.append(row)
        print(
            "  fit={fit_s:.3f}s bin={binning_s:.3f}s boost={boosting_s:.3f}s "
            "featbuild={feature_build_s:.3f}s solve={solve_s:.3f}s "
            "features={features} solver={solver_method}/{linear_solve}".format(**row),
            flush=True,
        )

    fieldnames = list(rows[0].keys()) if rows else []
    if args.csv is not None and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if rows:
        print("\nsummary_csv")
        writer = csv.DictWriter(_StdoutWriter(), fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _StdoutWriter:
    def write(self, text: str) -> None:
        print(text, end="")


if __name__ == "__main__":
    main()
