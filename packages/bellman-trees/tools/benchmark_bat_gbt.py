from __future__ import annotations

import argparse
import time

import numpy as np

from bellman_trees import BellmanHistogramGradientBoostingRegressor


def make_problem(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    drift = 0.65 * X + 0.15 * rng.normal(size=(n, p))
    X_next = drift.astype(np.float64)
    reward = (
        1.0
        + 0.5 * X[:, 0]
        - 0.25 * X[:, 1] ** 2
        + 0.2 * (X[:, 2] > 0.0 if p > 2 else X[:, 0] > 0.0)
    )
    return X, reward.astype(np.float64), X_next


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark custom Bellman histogram GBT.")
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-bins", type=int, default=128)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--max-samples-per-tree", type=int, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--feature-storage", choices=["auto", "csr", "streaming", "hashed"], default="auto")
    parser.add_argument("--solver-method", default="auto")
    parser.add_argument("--solver-max-iter", type=int, default=200)
    parser.add_argument("--solver-tol", type=float, default=1e-6)
    parser.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--hash-dim", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    X, reward, X_next = make_problem(args.n, args.p, args.seed)
    model = BellmanHistogramGradientBoostingRegressor(
        gamma=0.8,
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=args.max_depth,
        max_bins=args.max_bins,
        min_samples_leaf=max(20, args.n // 5000),
        min_child_weight=max(1.0, args.n / 20_000.0),
        subsample=args.subsample,
        max_samples_per_tree=args.max_samples_per_tree,
        colsample_bytree=args.colsample_bytree,
        colsample_bynode=0.8,
        feature_storage=args.feature_storage,
        solver_method=args.solver_method,
        backend=args.backend,
        batch_size=args.batch_size,
        hash_dim=args.hash_dim,
        solver_max_iter=args.solver_max_iter,
        solver_tol=args.solver_tol,
        random_state=args.seed,
    )
    start = time.perf_counter()
    model.fit(X, reward, X_next)
    fit_time = time.perf_counter() - start
    pred_start = time.perf_counter()
    pred = model.predict(X[: min(args.n, 20_000)])
    pred_time = time.perf_counter() - pred_start
    feature_density = len(model.trees_) / max(model.feature_info_["n_features"], 1)

    print("bellman_hist_gbt_benchmark")
    print(f"n={args.n}")
    print(f"p={args.p}")
    print(f"trees={model.feature_info_['n_trees']}")
    print(f"features={model.feature_info_['n_features']}")
    print(f"feature_storage={model.feature_info_['feature_storage']}")
    print(f"hash_dim={model.feature_info_['hash_dim']}")
    print(f"estimated_model_memory_mb={model.diagnostics_['estimated_memory_mb']:.1f}")
    print(f"feature_density_per_row={feature_density:.6f}")
    print(f"fit_time_sec={fit_time:.3f}")
    print(f"binning_time_sec={model.diagnostics_['binning_time']:.3f}")
    print(f"boosting_time_sec={model.diagnostics_['boosting_time']:.3f}")
    print(f"feature_build_time_sec={model.diagnostics_['feature_build_time']:.3f}")
    print(f"bellman_solve_time_sec={model.diagnostics_['bellman_solve_time']:.3f}")
    print(f"prediction_time_sec={pred_time:.3f}")
    print(f"prediction_throughput_rows_per_sec={pred.size / max(pred_time, 1e-12):.1f}")
    print(f"solver_method={model.solver_info_['method']}")
    print(f"solver_linear_solve={model.solver_info_['linear_solve']}")


if __name__ == "__main__":
    main()
