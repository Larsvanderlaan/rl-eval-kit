from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("FQE_calibration_neurips/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("FQE_calibration_neurips/.cache").resolve()))

import pandas as pd


TABLE_COLUMNS = [
    "suite_name",
    "baseline_learner",
    "learner_variant",
    "learner_quality_regime",
    "calibration_difficulty",
    "main_figure_role",
    "calibration_protocol",
    "calibrator",
    "calibration_target",
    "sample_size",
    "coverage_setting",
    "policy_shift_setting",
    "misspecification_setting",
    "split_fraction",
    "mean_value_estimate",
    "value_bias",
    "bias_mc_se",
    "value_variance",
    "mse",
    "mse_mc_se",
    "relative_mse_vs_uncalibrated_all_data",
    "mse_win_rate_vs_uncalibrated_all_data",
    "calibration_evidence_status",
    "scalar_cancellation_flag",
    "relative_calibration_error_plugin_vs_uncalibrated_all_data",
    "calibration_error_plugin_win_rate_vs_uncalibrated_all_data",
    "relative_true_v_mse_vs_uncalibrated_all_data",
    "true_v_mse_win_rate_vs_uncalibrated_all_data",
    "relative_true_v_mse_vs_current_retrain_small",
    "relative_calibration_error_plugin_vs_current_retrain_small",
    "true_v_mse_win_rate_vs_current_retrain_small",
    "calibration_error_plugin_win_rate_vs_current_retrain_small",
    "true_v_mse",
    "true_v_mse_mc_se",
    "true_q_mse",
    "true_q_mse_mc_se",
    "bellman_outcome_mse",
    "bellman_outcome_mse_mc_se",
    "brier_score",
    "brier_score_mc_se",
    "bellman_calibration_error",
    "bellman_calibration_error_mc_se",
    "bellman_calibration_error_plugin",
    "bellman_calibration_error_debiased_raw",
    "bellman_calibration_error_debiased_raw_mc_se",
    "bellman_calibration_bins",
    "bellman_calibration_test_size",
    "value_oracle_pearson",
    "value_oracle_spearman",
    "value_calibration_slope",
    "value_calibration_intercept",
    "raw_value_oracle_pearson",
    "raw_value_oracle_spearman",
    "raw_value_calibration_slope",
    "raw_value_calibration_intercept",
    "importance_weight_ess",
    "importance_weight_max",
    "interval_coverage_95",
    "interval_length_95",
    "actual_bellman_iterations",
    "feature_dimension",
    "ridge_alpha",
    "q_train_min",
    "q_train_max",
    "q_train_std",
    "saddle_condition_proxy",
    "saddle_gradient_norm_last",
    "saddle_exploding_rate",
    "runtime",
    "runtime_se",
    "failure_rate",
]


TABLE_SUITES = {
    "main_nonlinear_results": ["main_nonlinear"],
    "well_specified_debug_results": ["well_specified_debug"],
    "coverage_sweep_summary": ["coverage_sweep"],
    "sample_size_sweep_summary": ["sample_size_sweep"],
    "misspecification_sweep_summary": ["misspecification_sweep"],
    "calibration_quality_sweep_summary": ["calibration_quality_sweep"],
    "mechanism_distortion_sweep_summary": ["mechanism_distortion_sweep"],
    "split_fraction_sweep_summary": ["split_fraction_sweep"],
    "baseline_family_summary": ["baseline_family_sweep", "baseline_learner_sweep"],
    "calibration_protocol_summary": ["calibration_protocol_sweep"],
    "undertraining_sweep_summary": ["undertraining_sweep"],
    "bellman_incomplete_sweep_summary": ["bellman_incomplete_sweep"],
    "model_misspecification_sweep_summary": ["model_misspecification_sweep"],
    "temporal_reward_shift_sweep_summary": ["temporal_reward_shift_sweep"],
    "focused_neurips_main_summary": [
        "model_misspecification_sweep",
        "undertraining_sweep",
        "temporal_reward_shift_sweep",
        "mechanism_distortion_sweep",
    ],
}


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in TABLE_COLUMNS if col in df.columns]
    out = df[cols].copy()
    sort_cols = [
        col
        for col in [
            "suite_name",
            "baseline_learner",
            "learner_quality_regime",
            "learner_variant",
            "calibration_protocol",
            "calibrator",
            "sample_size",
        ]
        if col in out
    ]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out


def _write_one(df: pd.DataFrame, tables_dir: Path, stem: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    table = _format_table(df)
    csv_path = tables_dir / f"{stem}.csv"
    tex_path = tables_dir / f"{stem}.tex"
    table.to_csv(csv_path, index=False)
    with tex_path.open("w") as handle:
        handle.write(table.to_latex(index=False, float_format=lambda x: f"{x:.4g}"))


def write_tables(results_dir: str | Path, eligible_only: bool = True) -> list[Path]:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    summary_path = results_dir / ("eligible_summary.csv" if eligible_only and (results_dir / "eligible_summary.csv").exists() else "summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file for tables: {summary_path}")
    summary = pd.read_csv(summary_path)
    written: list[Path] = []
    for stem, suite_names in TABLE_SUITES.items():
        subset = summary[summary["suite_name"].isin(suite_names)] if "suite_name" in summary else summary.iloc[0:0]
        _write_one(subset, tables_dir, stem)
        written.extend([tables_dir / f"{stem}.csv", tables_dir / f"{stem}.tex"])
    _write_one(summary, tables_dir, "all_eligible_summary" if eligible_only else "all_summary")
    written.extend([tables_dir / ("all_eligible_summary.csv" if eligible_only else "all_summary.csv")])
    return written
