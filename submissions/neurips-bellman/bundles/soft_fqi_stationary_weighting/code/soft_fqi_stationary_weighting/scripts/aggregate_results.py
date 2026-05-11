#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _quantile(prob: float):
    def fn(values: pd.Series) -> float:
        arr = values.dropna().to_numpy(dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.quantile(arr, prob))

    fn.__name__ = f"q{int(prob * 100):02d}"
    return fn


def aggregate(results_dir: Path) -> tuple[Path, Path]:
    raw_path = results_dir / "raw_results.csv"
    raw = pd.read_csv(raw_path)
    group_cols = ["stage", "learner", "regime", "schedule", "method", "gamma_weight", "iteration"]
    if "q_class" in raw.columns:
        group_cols.append("q_class")
    metrics = [
        "stationary_q_rmse",
        "behavior_q_rmse",
        "stationary_advantage_q_rmse",
        "behavior_advantage_q_rmse",
        "stationary_bellman_rmse",
        "behavior_bellman_rmse",
        "stationary_projected_bellman_rmse",
        "behavior_projected_bellman_rmse",
        "cross_behavior_projected_bellman_rmse",
        "cross_stationary_projected_bellman_rmse",
        "policy_value_error",
        "norm_mismatch_ratio",
        "diverged",
        "failed",
        "effective_sample_size_fraction",
        "weight_max",
        "weight_q99",
        "oracle_log_ratio_rmse",
        "oracle_log_ratio_rmse_support",
        "oracle_estimated_weight_corr",
        "oracle_estimated_weight_rel_mse_clipped",
        "oracle_estimated_weight_calibration_slope",
        "moment_violation_l2",
        "ridge_primal",
        "ridge_dual",
        "cv_selected_ridge",
        "cv_selected_score",
        "weighted_gram_condition",
        "minimax_q_ridge",
        "minimax_critic_ridge",
        "minimax_residual_norm",
        "minimax_moment_l2",
        "minimax_critic_norm",
        "minimax_q_norm",
        "cv_minimax_ridge_selected",
        "cv_minimax_selected_ridge",
        "cv_minimax_selected_score",
    ]
    metrics = [metric for metric in metrics if metric in raw.columns]
    summary = (
        raw.groupby(group_cols, dropna=False)[metrics]
        .agg(["count", "mean", "median", _quantile(0.25), _quantile(0.75)])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in col if part != "") for col in summary.columns.to_flat_index()]
    summary_path = results_dir / "summary_by_iteration.csv"
    summary.to_csv(summary_path, index=False)

    final = raw[raw["is_final"] == 1].copy()
    final_group_cols = ["stage", "learner", "regime", "schedule", "method", "gamma_weight"]
    if "q_class" in final.columns:
        final_group_cols.append("q_class")
    final_summary = (
        final.groupby(final_group_cols, dropna=False)[metrics]
        .agg(["count", "mean", "median", _quantile(0.25), _quantile(0.75)])
        .reset_index()
    )
    final_summary.columns = ["_".join(str(part) for part in col if part != "") for col in final_summary.columns.to_flat_index()]
    final_path = results_dir / "summary_final.csv"
    final_summary.to_csv(final_path, index=False)
    return summary_path, final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate soft-FQI experiment results.")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    summary_path, final_path = aggregate(Path(args.results_dir))
    print(f"Wrote {summary_path}")
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
