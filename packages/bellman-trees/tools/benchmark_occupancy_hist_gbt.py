from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bellman_trees import DiscountedOccupancyHistogramGradientBoostingRatioEstimator


@dataclass(frozen=True)
class Case:
    name: str
    n_estimators: int = 10
    max_depth: int = 2
    max_leaves: int = 4
    max_bins: int = 32
    subsample: float = 1.0
    max_event_rows: int | None = 5_000
    colsample_bytree: float = 1.0
    colsample_bynode: float = 1.0
    ratio_refresh_interval: int = 1
    inner_solver_max_iter: int = 10
    final_solver_max_iter: int = 50
    inner_solver_tol: float = 1e-3
    final_solver_tol: float = 1e-4
    dense_threshold: int = 8
    feature_storage: str = "auto"
    hash_dim: int = 65_536
    batch_size: int = 65_536
    max_exact_features: int = 8192


QUICK_CASES = [
    Case("baseline"),
    Case("uncapped_events", max_event_rows=None),
    Case("event_cap_2500", max_event_rows=2_500),
    Case("subsample_half", subsample=0.5),
    Case("depth1", max_depth=1, max_leaves=2),
    Case("bins16", max_bins=16),
    Case("colsample_half", colsample_bytree=0.5, colsample_bynode=0.75),
    Case("looser_inner_solve", inner_solver_max_iter=3, inner_solver_tol=3e-3),
    Case("refresh_every_4", ratio_refresh_interval=4),
    Case("small_exact_solver", dense_threshold=2_048),
    Case("streaming_direct", feature_storage="streaming", dense_threshold=2_048),
    Case("hashed_fista", feature_storage="hashed", hash_dim=512, final_solver_max_iter=30),
]

SCALING_CASES = [
    *QUICK_CASES,
    Case("more_trees", n_estimators=25, max_event_rows=3_000, inner_solver_max_iter=5, final_solver_max_iter=50),
    Case("deeper_trees", max_depth=3, max_leaves=8, max_event_rows=3_000),
]

MILLION_CASES = [
    Case(
        "million_streaming",
        n_estimators=50,
        max_depth=2,
        max_leaves=4,
        max_bins=32,
        subsample=0.5,
        max_event_rows=200_000,
        colsample_bytree=0.5,
        ratio_refresh_interval=10,
        inner_solver_max_iter=5,
        final_solver_max_iter=50,
        feature_storage="streaming",
        batch_size=65_536,
    ),
    Case(
        "million_hashed",
        n_estimators=100,
        max_depth=3,
        max_leaves=8,
        max_bins=32,
        subsample=0.5,
        max_event_rows=200_000,
        colsample_bytree=0.5,
        ratio_refresh_interval=10,
        inner_solver_max_iter=3,
        final_solver_max_iter=30,
        feature_storage="hashed",
        hash_dim=65_536,
        batch_size=65_536,
    ),
]


def make_problem(n: int, p: int, n_initial: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    noise = rng.normal(size=(n, p)).astype(np.float64)
    drift = 0.72 * X + 0.18 * noise
    if p > 1:
        drift[:, 0] += 0.4 * (X[:, 1] > 0.0)
    if p > 2:
        drift[:, 2] = 0.5 * X[:, 2] + 0.2 * np.sin(X[:, 0]) + 0.2 * noise[:, 2]
    X_next = drift.astype(np.float64)
    X_initial = rng.normal(loc=-0.35, scale=0.8, size=(n_initial, p)).astype(np.float64)
    return X, X_next, X_initial


def _case_kwargs(case: Case, seed: int) -> dict[str, Any]:
    return {
        "gamma": 0.8,
        "n_estimators": case.n_estimators,
        "learning_rate": 0.05,
        "max_depth": case.max_depth,
        "max_leaves": case.max_leaves,
        "max_bins": case.max_bins,
        "min_samples_leaf": 50,
        "min_child_weight": 1e-8,
        "l2_leaf_reg": 1.0,
        "hessian_floor": 1e-6,
        "subsample": case.subsample,
        "max_event_rows": case.max_event_rows,
        "colsample_bytree": case.colsample_bytree,
        "colsample_bynode": case.colsample_bynode,
        "ratio_refresh_interval": case.ratio_refresh_interval,
        "solver": "auto",
        "dense_threshold": case.dense_threshold,
        "inner_solver_tol": case.inner_solver_tol,
        "inner_solver_max_iter": case.inner_solver_max_iter,
        "final_solver_tol": case.final_solver_tol,
        "final_solver_max_iter": case.final_solver_max_iter,
        "feature_storage": case.feature_storage,
        "hash_dim": case.hash_dim,
        "batch_size": case.batch_size,
        "max_exact_features": case.max_exact_features,
        "weight_clip_quantile": None,
        "random_state": seed,
    }


def _rows_per_second(n: int, seconds: float) -> float:
    return float(n / max(seconds, 1e-12))


def run_case(case: Case, X: np.ndarray, X_next: np.ndarray, X_initial: np.ndarray, seed: int) -> dict[str, float | int | str | bool]:
    model = DiscountedOccupancyHistogramGradientBoostingRatioEstimator(**_case_kwargs(case, seed))
    fit_start = time.perf_counter()
    model.fit(X, X_next, X_initial)
    fit_time = time.perf_counter() - fit_start

    eval_n = min(X.shape[0], 5_000)
    apply_start = time.perf_counter()
    leaves = model.apply(X[:eval_n])
    apply_time = time.perf_counter() - apply_start
    transform_start = time.perf_counter()
    phi = model.transform(X[:eval_n])
    transform_time = time.perf_counter() - transform_start
    predict_start = time.perf_counter()
    pred = model.predict_ratio(X[:eval_n])
    predict_time = time.perf_counter() - predict_start

    timings = model.diagnostics_
    return {
        "case": case.name,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "n_initial": int(X_initial.shape[0]),
        "trees": int(model.feature_info_["n_trees"]),
        "features": int(model.feature_info_["n_features"]),
        "raw_features": int(model.feature_info_["n_features_raw"]),
        "feature_storage": str(model.feature_info_["feature_storage"]),
        "hash_dim": -1 if model.feature_info_["hash_dim"] is None else int(model.feature_info_["hash_dim"]),
        "max_event_rows": -1 if case.max_event_rows is None else int(case.max_event_rows),
        "batch_size": int(case.batch_size),
        "subsample": float(case.subsample),
        "max_depth": int(case.max_depth),
        "max_bins": int(case.max_bins),
        "ratio_refresh_interval": int(case.ratio_refresh_interval),
        "fit_s": float(fit_time),
        "binning_s": float(timings["binning_time"]),
        "boosting_s": float(timings["boosting_time"]),
        "final_solve_s": float(timings["occupancy_solve_time"]),
        "final_feature_build_s": float(timings["final_feature_build_time"]),
        "final_solver_s": float(timings["final_solver_time"]),
        "event_build_s": float(timings["total_event_build_time"]),
        "event_binning_s": float(timings["total_event_binning_time"]),
        "tree_fit_s": float(timings["total_tree_fit_time"]),
        "inner_solve_s": float(timings["total_inner_solve_time"]),
        "inner_feature_build_s": float(timings["total_feature_build_time"]),
        "inner_solver_s": float(timings["total_inner_solver_time"]),
        "validation_s": float(timings["total_validation_time"]),
        "mean_round_s": float(timings["mean_round_time"]),
        "solver": str(model.solver_info_["solver"]),
        "linear_solve": str(model.solver_info_["linear_solve"]),
        "solver_iterations": int(model.solver_info_.get("iterations", 0)),
        "converged": bool(model.solver_info_.get("converged", False)),
        "moment_l2": float(model.solver_info_["moment_violation_l2"]),
        "ratio_ess_fraction": float(model.solver_info_["ratio_ess_fraction"]),
        "apply_rows_per_s": _rows_per_second(eval_n, apply_time),
        "transform_rows_per_s": _rows_per_second(eval_n, transform_time),
        "predict_rows_per_s": _rows_per_second(eval_n, predict_time),
        "phi_nnz_eval": int(phi.nnz),
        "leaves_checksum": int(np.sum(leaves) % 1_000_000),
        "pred_mean": float(np.mean(pred)),
    }


def _cases_for_preset(name: str, trees: int | None) -> list[Case]:
    if name == "quick":
        cases = QUICK_CASES
    elif name == "scaling":
        cases = SCALING_CASES
    elif name == "million":
        cases = MILLION_CASES
    else:
        raise ValueError(f"unknown preset {name!r}")
    if trees is not None:
        cases = [replace(case, n_estimators=int(trees)) for case in cases]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed ablations for discounted occupancy histogram GBT.")
    parser.add_argument("--preset", choices=["quick", "scaling", "million"], default="quick")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--n-initial", type=int, default=None)
    parser.add_argument("--trees", type=int, default=None, help="Override n_estimators for every case.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    n = 1_000_000 if args.preset == "million" and args.n is None else (8_000 if args.n is None else args.n)
    p = 64 if args.preset == "million" and args.p is None else (8 if args.p is None else args.p)
    n_initial = 100_000 if args.preset == "million" and args.n_initial is None else (1_000 if args.n_initial is None else args.n_initial)
    X, X_next, X_initial = make_problem(n, p, n_initial, args.seed)
    cases = _cases_for_preset(args.preset, args.trees)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    rows = []
    for offset, case in enumerate(cases):
        print(f"running {case.name} n={n} p={p} trees={case.n_estimators}", flush=True)
        row = run_case(case, X, X_next, X_initial, args.seed + offset)
        rows.append(row)
        print(
            "  fit={fit_s:.3f}s event={event_build_s:.3f}s bin_events={event_binning_s:.3f}s "
            "tree={tree_fit_s:.3f}s feat={inner_feature_build_s:.3f}s inner_solver={inner_solver_s:.3f}s "
            "final={final_solve_s:.3f}s "
            "features={features} solver={solver}/{linear_solve}".format(**row),
            flush=True,
        )

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\nsummary_csv")
    writer = csv.DictWriter(_StdoutWriter(), fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


class _StdoutWriter:
    def write(self, text: str) -> None:
        print(text, end="")


if __name__ == "__main__":
    main()
