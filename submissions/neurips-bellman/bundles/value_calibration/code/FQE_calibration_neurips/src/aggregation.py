from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("FQE_calibration_neurips/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("FQE_calibration_neurips/.cache").resolve()))

import numpy as np
import pandas as pd

MSE_EVIDENCE_RELATIVE_THRESHOLD = 0.98
CALIBRATION_EVIDENCE_RELATIVE_THRESHOLD = 0.95


SUMMARY_GROUP_COLS = [
    "run_mode",
    "suite_name",
    "environment_name",
    "environment_tier",
    "sample_size",
    "state_dimension",
    "coverage_setting",
    "policy_shift_setting",
    "reward_noise_setting",
    "misspecification_setting",
    "baseline_learner",
    "learner_variant",
    "learner_quality_regime",
    "calibration_difficulty",
    "main_figure_role",
    "calibrated",
    "calibration_protocol",
    "calibrator",
    "calibration_target",
    "split_fraction",
    "train_fraction",
]

BASELINE_MATCH_COLS = [
    "run_mode",
    "suite_name",
    "environment_name",
    "environment_tier",
    "sample_size",
    "state_dimension",
    "coverage_setting",
    "policy_shift_setting",
    "reward_noise_setting",
    "misspecification_setting",
    "baseline_learner",
    "learner_variant",
    "learner_quality_regime",
    "calibration_difficulty",
    "calibration_target",
]

SEED_BASELINE_MATCH_COLS = [
    "run_mode",
    "suite_name",
    "environment_name",
    "environment_tier",
    "sample_size",
    "state_dimension",
    "coverage_setting",
    "policy_shift_setting",
    "reward_noise_setting",
    "misspecification_setting",
    "baseline_learner",
    "learner_variant",
    "learner_quality_regime",
    "calibration_difficulty",
    "calibration_target",
    "replication_seed",
]


def _mc_se(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _read_raw_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(path for path in results_dir.glob("**/raw_results.csv") if path.name == "raw_results.csv")
    frames = [pd.read_csv(path) for path in files]
    if not frames:
        raise FileNotFoundError(f"No raw result CSVs found recursively in {results_dir}.")
    return pd.concat(frames, ignore_index=True)


def _ensure_columns(raw: pd.DataFrame) -> pd.DataFrame:
    if "failure_flag" in raw:
        default_eligible = ~raw["failure_flag"].astype(bool)
    else:
        default_eligible = True
    defaults = {
        "run_mode": "standalone",
        "suite_name": "standalone",
        "environment_tier": "unknown",
        "policy_shift_setting": np.nan,
        "misspecification_setting": "unknown",
        "learner_variant": raw["baseline_learner"] if "baseline_learner" in raw else "unknown",
        "learner_quality_regime": "well_tuned",
        "calibration_difficulty": "already_well_calibrated",
        "main_figure_role": "main",
        "split_fraction": raw["train_fraction"] if "train_fraction" in raw else 1.0,
        "main_evidence_eligible": default_eligible,
        "calibration_object": "value",
        "calibration_weight_scheme": "action_ratio",
        "importance_weight_ess": np.nan,
        "importance_weight_clip": np.nan,
        "importance_weight_max": np.nan,
        "true_v_mse": raw["true_value_function_mse"] if "true_value_function_mse" in raw else np.nan,
        "true_value_function_mse": raw["true_v_mse"] if "true_v_mse" in raw else np.nan,
        "true_q_mse": np.nan,
        "true_function_mse": raw["true_q_mse"] if "true_q_mse" in raw else np.nan,
        "bellman_outcome_mse": np.nan,
        "brier_score": raw["bellman_outcome_mse"] if "bellman_outcome_mse" in raw else np.nan,
        "bellman_brier_score": raw["bellman_outcome_mse"] if "bellman_outcome_mse" in raw else np.nan,
        "bellman_calibration_error": raw["calibration_error"] if "calibration_error" in raw else np.nan,
        "bellman_calibration_error_plugin": np.nan,
        "bellman_calibration_error_debiased_raw": np.nan,
        "bellman_calibration_bins": np.nan,
        "bellman_calibration_test_size": np.nan,
        "value_oracle_pearson": np.nan,
        "value_oracle_spearman": np.nan,
        "value_calibration_slope": np.nan,
        "value_calibration_intercept": np.nan,
        "value_reliability_curve_json": "",
        "raw_value_oracle_pearson": np.nan,
        "raw_value_oracle_spearman": np.nan,
        "raw_value_calibration_slope": np.nan,
        "raw_value_calibration_intercept": np.nan,
        "raw_value_reliability_curve_json": "",
        "interval_lower_95": np.nan,
        "interval_upper_95": np.nan,
        "interval_length_95": np.nan,
        "interval_coverage_95": np.nan,
        "model_diag_actual_bellman_iterations": np.nan,
        "model_diag_actual_epochs_per_iter": np.nan,
        "model_diag_feature_dimension": np.nan,
        "model_diag_ridge_alpha": np.nan,
        "model_diag_q_train_min": np.nan,
        "model_diag_q_train_max": np.nan,
        "model_diag_q_train_std": np.nan,
        "model_diag_saddle_condition_proxy": np.nan,
        "model_diag_saddle_gradient_norm_last": np.nan,
        "model_diag_saddle_exploding_flag": np.nan,
        "model_diag_saddle_nan_flag": np.nan,
    }
    for col, value in defaults.items():
        if col not in raw:
            raw[col] = value
    return raw


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _ensure_columns(raw.copy())
    summary = raw.groupby(SUMMARY_GROUP_COLS, dropna=False).agg(
        n_replications=("replication_seed", "nunique"),
        mean_value_estimate=("value_estimate", "mean"),
        value_bias=("value_error", "mean"),
        bias_mc_se=("value_error", _mc_se),
        value_variance=("value_estimate", "var"),
        mse=("squared_error", "mean"),
        mse_mc_se=("squared_error", _mc_se),
        bellman_residual=("bellman_residual", "mean"),
        calibration_error=("calibration_error", "mean"),
        true_v_mse=("true_v_mse", "mean"),
        true_v_mse_mc_se=("true_v_mse", _mc_se),
        true_value_function_mse=("true_value_function_mse", "mean"),
        true_q_mse=("true_q_mse", "mean"),
        true_q_mse_mc_se=("true_q_mse", _mc_se),
        true_function_mse=("true_function_mse", "mean"),
        bellman_outcome_mse=("bellman_outcome_mse", "mean"),
        bellman_outcome_mse_mc_se=("bellman_outcome_mse", _mc_se),
        brier_score=("brier_score", "mean"),
        brier_score_mc_se=("brier_score", _mc_se),
        bellman_brier_score=("bellman_brier_score", "mean"),
        bellman_calibration_error=("bellman_calibration_error", "mean"),
        bellman_calibration_error_mc_se=("bellman_calibration_error", _mc_se),
        bellman_calibration_error_plugin=("bellman_calibration_error_plugin", "mean"),
        bellman_calibration_error_debiased_raw=("bellman_calibration_error_debiased_raw", "mean"),
        bellman_calibration_error_debiased_raw_mc_se=("bellman_calibration_error_debiased_raw", _mc_se),
        bellman_calibration_bins=("bellman_calibration_bins", "mean"),
        bellman_calibration_test_size=("bellman_calibration_test_size", "mean"),
        value_oracle_pearson=("value_oracle_pearson", "mean"),
        value_oracle_spearman=("value_oracle_spearman", "mean"),
        value_calibration_slope=("value_calibration_slope", "mean"),
        value_calibration_intercept=("value_calibration_intercept", "mean"),
        raw_value_oracle_pearson=("raw_value_oracle_pearson", "mean"),
        raw_value_oracle_spearman=("raw_value_oracle_spearman", "mean"),
        raw_value_calibration_slope=("raw_value_calibration_slope", "mean"),
        raw_value_calibration_intercept=("raw_value_calibration_intercept", "mean"),
        importance_weight_ess=("importance_weight_ess", "mean"),
        importance_weight_max=("importance_weight_max", "mean"),
        interval_coverage_95=("interval_coverage_95", "mean"),
        interval_length_95=("interval_length_95", "mean"),
        interval_length_95_mc_se=("interval_length_95", _mc_se),
        actual_bellman_iterations=("model_diag_actual_bellman_iterations", "mean"),
        actual_epochs_per_iter=("model_diag_actual_epochs_per_iter", "mean"),
        feature_dimension=("model_diag_feature_dimension", "mean"),
        ridge_alpha=("model_diag_ridge_alpha", "mean"),
        q_train_min=("model_diag_q_train_min", "mean"),
        q_train_max=("model_diag_q_train_max", "mean"),
        q_train_std=("model_diag_q_train_std", "mean"),
        saddle_condition_proxy=("model_diag_saddle_condition_proxy", "mean"),
        saddle_gradient_norm_last=("model_diag_saddle_gradient_norm_last", "mean"),
        saddle_exploding_rate=("model_diag_saddle_exploding_flag", "mean"),
        saddle_nan_rate=("model_diag_saddle_nan_flag", "mean"),
        runtime=("runtime", "mean"),
        runtime_se=("runtime", _mc_se),
        failure_rate=("failure_flag", "mean"),
        eligible_fraction=("main_evidence_eligible", "mean"),
    ).reset_index()

    baseline_metric_cols = [
        "mse",
        "bellman_calibration_error_plugin",
        "bellman_calibration_error",
        "bellman_outcome_mse",
        "brier_score",
        "true_v_mse",
    ]
    baselines = summary[
        (summary["calibrated"] == False)  # noqa: E712
        & (summary["calibration_protocol"] == "uncalibrated_all_data")
    ][BASELINE_MATCH_COLS + baseline_metric_cols].rename(
        columns={col: f"uncalibrated_all_data_{col}" for col in baseline_metric_cols}
    )
    summary = summary.merge(baselines, on=BASELINE_MATCH_COLS, how="left")
    summary["relative_mse_vs_uncalibrated_all_data"] = (
        summary["mse"] / summary["uncalibrated_all_data_mse"].replace(0.0, pd.NA)
    )
    summary["relative_calibration_error_plugin_vs_uncalibrated_all_data"] = (
        summary["bellman_calibration_error_plugin"]
        / summary["uncalibrated_all_data_bellman_calibration_error_plugin"].replace(0.0, pd.NA)
    )
    summary["relative_brier_score_vs_uncalibrated_all_data"] = (
        summary["brier_score"] / summary["uncalibrated_all_data_brier_score"].replace(0.0, pd.NA)
    )
    summary["relative_true_v_mse_vs_uncalibrated_all_data"] = (
        summary["true_v_mse"] / summary["uncalibrated_all_data_true_v_mse"].replace(0.0, pd.NA)
    )
    current_retrain_metrics = ["true_v_mse", "bellman_calibration_error_plugin"]
    current_retrain = summary[
        (summary["calibrated"] == False)  # noqa: E712
        & (summary["calibration_protocol"] == "current_retrain_small")
    ][BASELINE_MATCH_COLS + current_retrain_metrics].rename(
        columns={col: f"current_retrain_small_{col}" for col in current_retrain_metrics}
    )
    summary = summary.merge(current_retrain, on=BASELINE_MATCH_COLS, how="left")
    summary["relative_true_v_mse_vs_current_retrain_small"] = (
        summary["true_v_mse"] / summary["current_retrain_small_true_v_mse"].replace(0.0, pd.NA)
    )
    summary["relative_calibration_error_plugin_vs_current_retrain_small"] = (
        summary["bellman_calibration_error_plugin"]
        / summary["current_retrain_small_bellman_calibration_error_plugin"].replace(0.0, pd.NA)
    )

    seed_baseline_metric_cols = ["squared_error", "bellman_calibration_error_plugin", "brier_score", "true_v_mse"]
    seed_baselines = raw[
        (raw["calibrated"] == False)  # noqa: E712
        & (raw["calibration_protocol"] == "uncalibrated_all_data")
        & (raw["calibrator"] == "none")
    ][SEED_BASELINE_MATCH_COLS + seed_baseline_metric_cols].rename(
        columns={col: f"seed_uncalibrated_all_data_{col}" for col in seed_baseline_metric_cols}
    )
    seed_matched = raw.merge(seed_baselines, on=SEED_BASELINE_MATCH_COLS, how="left")
    seed_matched["mse_win_rate_vs_uncalibrated_all_data"] = pd.to_numeric(
        seed_matched["squared_error"], errors="coerce"
    ).lt(pd.to_numeric(seed_matched["seed_uncalibrated_all_data_squared_error"], errors="coerce"))
    seed_matched["calibration_error_plugin_win_rate_vs_uncalibrated_all_data"] = pd.to_numeric(
        seed_matched["bellman_calibration_error_plugin"], errors="coerce"
    ).lt(pd.to_numeric(seed_matched["seed_uncalibrated_all_data_bellman_calibration_error_plugin"], errors="coerce"))
    seed_matched["brier_score_win_rate_vs_uncalibrated_all_data"] = pd.to_numeric(
        seed_matched["brier_score"], errors="coerce"
    ).lt(pd.to_numeric(seed_matched["seed_uncalibrated_all_data_brier_score"], errors="coerce"))
    seed_matched["true_v_mse_win_rate_vs_uncalibrated_all_data"] = pd.to_numeric(
        seed_matched["true_v_mse"], errors="coerce"
    ).lt(pd.to_numeric(seed_matched["seed_uncalibrated_all_data_true_v_mse"], errors="coerce"))
    seed_current_retrain = raw[
        (raw["calibrated"] == False)  # noqa: E712
        & (raw["calibration_protocol"] == "current_retrain_small")
        & (raw["calibrator"] == "none")
    ][SEED_BASELINE_MATCH_COLS + ["bellman_calibration_error_plugin", "true_v_mse"]].rename(
        columns={
            "bellman_calibration_error_plugin": "seed_current_retrain_small_bellman_calibration_error_plugin",
            "true_v_mse": "seed_current_retrain_small_true_v_mse",
        }
    )
    seed_matched = seed_matched.merge(seed_current_retrain, on=SEED_BASELINE_MATCH_COLS, how="left")
    seed_matched["true_v_mse_win_rate_vs_current_retrain_small"] = pd.to_numeric(
        seed_matched["true_v_mse"], errors="coerce"
    ).le(pd.to_numeric(seed_matched["seed_current_retrain_small_true_v_mse"], errors="coerce") * 1.05)
    seed_matched["calibration_error_plugin_win_rate_vs_current_retrain_small"] = pd.to_numeric(
        seed_matched["bellman_calibration_error_plugin"], errors="coerce"
    ).le(
        pd.to_numeric(seed_matched["seed_current_retrain_small_bellman_calibration_error_plugin"], errors="coerce")
        * 1.05
    )
    win_rates = seed_matched.groupby(SUMMARY_GROUP_COLS, dropna=False).agg(
        mse_win_rate_vs_uncalibrated_all_data=("mse_win_rate_vs_uncalibrated_all_data", "mean"),
        calibration_error_plugin_win_rate_vs_uncalibrated_all_data=(
            "calibration_error_plugin_win_rate_vs_uncalibrated_all_data",
            "mean",
        ),
        brier_score_win_rate_vs_uncalibrated_all_data=("brier_score_win_rate_vs_uncalibrated_all_data", "mean"),
        true_v_mse_win_rate_vs_uncalibrated_all_data=("true_v_mse_win_rate_vs_uncalibrated_all_data", "mean"),
        true_v_mse_win_rate_vs_current_retrain_small=("true_v_mse_win_rate_vs_current_retrain_small", "mean"),
        calibration_error_plugin_win_rate_vs_current_retrain_small=(
            "calibration_error_plugin_win_rate_vs_current_retrain_small",
            "mean",
        ),
    ).reset_index()
    summary = summary.merge(win_rates, on=SUMMARY_GROUP_COLS, how="left")

    mse_improved = pd.to_numeric(summary["relative_mse_vs_uncalibrated_all_data"], errors="coerce").lt(
        MSE_EVIDENCE_RELATIVE_THRESHOLD
    )
    cal_improved = pd.to_numeric(
        summary["relative_calibration_error_plugin_vs_uncalibrated_all_data"], errors="coerce"
    ).lt(CALIBRATION_EVIDENCE_RELATIVE_THRESHOLD)
    brier_improved = pd.to_numeric(summary["relative_brier_score_vs_uncalibrated_all_data"], errors="coerce").lt(
        CALIBRATION_EVIDENCE_RELATIVE_THRESHOLD
    )
    true_v_improved = pd.to_numeric(summary["relative_true_v_mse_vs_uncalibrated_all_data"], errors="coerce").lt(1.0)
    calibrated = summary["calibrated"].astype(bool)
    failed = pd.to_numeric(summary["failure_rate"], errors="coerce").fillna(0.0).gt(0.0) | pd.to_numeric(
        summary["eligible_fraction"], errors="coerce"
    ).fillna(0.0).le(0.0)
    status = np.full(summary.shape[0], "neutral", dtype=object)
    status[failed.to_numpy()] = "failed"
    strong = calibrated & ~failed & mse_improved & cal_improved
    mse_only = calibrated & ~failed & mse_improved & ~cal_improved
    calibration_only = calibrated & ~failed & ~mse_improved & cal_improved
    status[strong.to_numpy()] = "strong"
    status[mse_only.to_numpy()] = "mse_only"
    status[calibration_only.to_numpy()] = "calibration_only"
    summary["mse_improved_vs_uncalibrated_all_data"] = mse_improved
    summary["calibration_error_plugin_improved_vs_uncalibrated_all_data"] = cal_improved
    summary["brier_score_improved_vs_uncalibrated_all_data"] = brier_improved
    summary["true_v_mse_improved_vs_uncalibrated_all_data"] = true_v_improved
    summary["scalar_cancellation_flag"] = (
        calibrated
        & pd.to_numeric(summary["relative_mse_vs_uncalibrated_all_data"], errors="coerce").lt(0.5)
        & ~(cal_improved | true_v_improved)
    )
    summary["calibration_evidence_status"] = status
    return summary


def write_coverage_stratified(raw: pd.DataFrame, results_dir: str | Path) -> Path | None:
    """Write coverage/density-ratio stratum diagnostics when raw rows contain them."""
    results_dir = Path(results_dir)
    error_cols = sorted(
        col
        for col in raw.columns
        if re.fullmatch(r"coverage_stratum_\d+_error", str(col))
    )
    if not error_cols:
        return None

    meta_cols = [
        "run_mode",
        "suite_name",
        "environment_name",
        "environment_tier",
        "sample_size",
        "state_dimension",
        "coverage_setting",
        "policy_shift_setting",
        "reward_noise_setting",
        "misspecification_setting",
        "baseline_learner",
        "learner_variant",
        "learner_quality_regime",
        "calibration_difficulty",
        "main_figure_role",
        "calibrated",
        "calibration_protocol",
        "calibrator",
        "calibration_target",
        "split_fraction",
        "train_fraction",
        "failure_flag",
        "main_evidence_eligible",
        "replication_seed",
    ]
    present_meta = [col for col in meta_cols if col in raw.columns]
    records: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        base = {col: row[col] for col in present_meta}
        for error_col in error_cols:
            match = re.fullmatch(r"coverage_stratum_(\d+)_error", str(error_col))
            if not match:
                continue
            stratum = int(match.group(1))
            record = dict(base)
            record.update(
                {
                    "coverage_stratum": stratum,
                    "coverage_stratum_error": row.get(error_col, np.nan),
                    "coverage_stratum_calibration_error": row.get(
                        f"coverage_stratum_{stratum}_calibration_error", np.nan
                    ),
                    "coverage_stratum_count": row.get(f"coverage_stratum_{stratum}_count", np.nan),
                    "coverage_stratum_mean_ratio": row.get(f"coverage_stratum_{stratum}_mean_ratio", np.nan),
                }
            )
            records.append(record)

    long = pd.DataFrame.from_records(records)
    raw_path = results_dir / "coverage_stratified_error_raw.csv"
    long.to_csv(raw_path, index=False)

    group_cols = [
        col
        for col in present_meta
        if col not in {"replication_seed", "failure_flag", "main_evidence_eligible"}
    ] + ["coverage_stratum"]
    summary = long.groupby(group_cols, dropna=False).agg(
        n_replications=("replication_seed", "nunique"),
        mean_coverage_stratum_error=("coverage_stratum_error", "mean"),
        coverage_stratum_error_mc_se=("coverage_stratum_error", _mc_se),
        mean_coverage_stratum_calibration_error=("coverage_stratum_calibration_error", "mean"),
        coverage_stratum_calibration_error_mc_se=("coverage_stratum_calibration_error", _mc_se),
        mean_coverage_stratum_count=("coverage_stratum_count", "mean"),
        mean_coverage_stratum_ratio=("coverage_stratum_mean_ratio", "mean"),
        failure_rate=("failure_flag", "mean") if "failure_flag" in long else ("coverage_stratum_error", lambda _x: np.nan),
        eligible_fraction=("main_evidence_eligible", "mean")
        if "main_evidence_eligible" in long
        else ("coverage_stratum_error", lambda _x: np.nan),
    ).reset_index()
    out_path = results_dir / "coverage_stratified_error.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def write_calibration_evidence_audit(summary: pd.DataFrame, results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    audit_cols = [
        "run_mode",
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
        "misspecification_setting",
        "mse",
        "relative_mse_vs_uncalibrated_all_data",
        "mse_win_rate_vs_uncalibrated_all_data",
        "bellman_calibration_error_plugin",
        "relative_calibration_error_plugin_vs_uncalibrated_all_data",
        "calibration_error_plugin_win_rate_vs_uncalibrated_all_data",
        "bellman_calibration_error_debiased_raw",
        "bellman_calibration_error",
        "brier_score",
        "relative_brier_score_vs_uncalibrated_all_data",
        "brier_score_win_rate_vs_uncalibrated_all_data",
        "true_v_mse",
        "relative_true_v_mse_vs_uncalibrated_all_data",
        "true_v_mse_win_rate_vs_uncalibrated_all_data",
        "raw_value_oracle_spearman",
        "raw_value_oracle_pearson",
        "raw_value_calibration_slope",
        "raw_value_calibration_intercept",
        "calibration_evidence_status",
        "scalar_cancellation_flag",
        "failure_rate",
    ]
    present = [col for col in audit_cols if col in summary.columns]
    if "calibrated" in summary:
        calibrated_mask = summary["calibrated"].astype(str).str.lower().isin({"true", "1", "yes"})
        calibrated = summary[calibrated_mask].copy()
    else:
        calibrated = summary.copy()
    evidence_path = results_dir / "calibration_evidence_audit.csv"
    calibrated[present].to_csv(evidence_path, index=False)
    scalar = calibrated[calibrated.get("scalar_cancellation_flag", False).astype(bool)].copy()
    scalar_path = results_dir / "scalar_cancellation_audit.csv"
    scalar[present].to_csv(scalar_path, index=False)
    return [evidence_path, scalar_path]


def _q10(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(arr, 0.10)) if arr.size else float("nan")


def _q90(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.quantile(arr, 0.90)) if arr.size else float("nan")


def write_split_stability_diagnostics(raw: pd.DataFrame, results_dir: str | Path) -> Path | None:
    """Write seed-matched diagnostics for split-calibration volatility."""
    results_dir = Path(results_dir)
    raw = _ensure_columns(raw.copy())
    required = set(SEED_BASELINE_MATCH_COLS + ["calibration_protocol", "calibrated", "squared_error", "train_fraction"])
    if not required.issubset(raw.columns):
        return None

    split = raw[
        (raw["calibration_protocol"].astype(str) == "split")
        & (raw["calibrated"].astype(bool))
        & (raw["failure_flag"].astype(bool) == False)  # noqa: E712
    ].copy()
    if split.empty:
        return None

    all_data = raw[
        (raw["calibration_protocol"].astype(str) == "uncalibrated_all_data")
        & (raw["calibrator"].astype(str) == "none")
    ][SEED_BASELINE_MATCH_COLS + ["squared_error"]].rename(
        columns={"squared_error": "all_data_uncalibrated_squared_error"}
    )
    same_fraction = raw[
        (raw["calibration_protocol"].astype(str) == "uncalibrated_same_fraction")
        & (raw["calibrator"].astype(str) == "none")
    ][SEED_BASELINE_MATCH_COLS + ["train_fraction", "squared_error"]].rename(
        columns={"squared_error": "same_fraction_uncalibrated_squared_error"}
    )

    matched = split.merge(all_data, on=SEED_BASELINE_MATCH_COLS, how="left")
    matched = matched.merge(same_fraction, on=SEED_BASELINE_MATCH_COLS + ["train_fraction"], how="left")
    matched["relative_mse_vs_all_data_seed_matched"] = (
        pd.to_numeric(matched["squared_error"], errors="coerce")
        / pd.to_numeric(matched["all_data_uncalibrated_squared_error"], errors="coerce").replace(0.0, np.nan)
    )
    matched["relative_mse_vs_same_fraction_seed_matched"] = (
        pd.to_numeric(matched["squared_error"], errors="coerce")
        / pd.to_numeric(matched["same_fraction_uncalibrated_squared_error"], errors="coerce").replace(0.0, np.nan)
    )
    matched["wins_vs_all_data"] = matched["relative_mse_vs_all_data_seed_matched"].lt(1.0)
    matched["wins_vs_same_fraction"] = matched["relative_mse_vs_same_fraction_seed_matched"].lt(1.0)
    raw_path = results_dir / "split_stability_diagnostics_raw.csv"
    matched.to_csv(raw_path, index=False)

    group_cols = [
        col
        for col in [
            "run_mode",
            "suite_name",
            "environment_name",
            "environment_tier",
            "sample_size",
            "state_dimension",
            "coverage_setting",
            "policy_shift_setting",
            "reward_noise_setting",
            "misspecification_setting",
            "baseline_learner",
            "learner_variant",
            "learner_quality_regime",
            "calibration_difficulty",
            "main_figure_role",
            "calibration_target",
            "calibrator",
            "split_fraction",
            "train_fraction",
        ]
        if col in matched.columns
    ]
    summary = matched.groupby(group_cols, dropna=False).agg(
        n_replications=("replication_seed", "nunique"),
        mean_relative_mse_vs_all_data=("relative_mse_vs_all_data_seed_matched", "mean"),
        median_relative_mse_vs_all_data=("relative_mse_vs_all_data_seed_matched", "median"),
        q10_relative_mse_vs_all_data=("relative_mse_vs_all_data_seed_matched", _q10),
        q90_relative_mse_vs_all_data=("relative_mse_vs_all_data_seed_matched", _q90),
        mc_se_relative_mse_vs_all_data=("relative_mse_vs_all_data_seed_matched", _mc_se),
        win_rate_vs_all_data=("wins_vs_all_data", "mean"),
        mean_relative_mse_vs_same_fraction=("relative_mse_vs_same_fraction_seed_matched", "mean"),
        median_relative_mse_vs_same_fraction=("relative_mse_vs_same_fraction_seed_matched", "median"),
        q10_relative_mse_vs_same_fraction=("relative_mse_vs_same_fraction_seed_matched", _q10),
        q90_relative_mse_vs_same_fraction=("relative_mse_vs_same_fraction_seed_matched", _q90),
        mc_se_relative_mse_vs_same_fraction=("relative_mse_vs_same_fraction_seed_matched", _mc_se),
        win_rate_vs_same_fraction=("wins_vs_same_fraction", "mean"),
    ).reset_index()
    out_path = results_dir / "split_stability_diagnostics.csv"
    summary.to_csv(out_path, index=False)
    return out_path


def aggregate_results(results_dir: str | Path, write_tables: bool = True) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_raw_results(results_dir)
    raw = _ensure_columns(raw)
    raw_path = results_dir / "combined_raw_results.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize_raw(raw)
    summary_path = results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    write_calibration_evidence_audit(summary, results_dir)
    write_coverage_stratified(raw, results_dir)
    write_split_stability_diagnostics(raw, results_dir)

    eligible_raw = raw[(raw["main_evidence_eligible"] == True) & (raw["failure_flag"] == False)]  # noqa: E712
    if not eligible_raw.empty:
        eligible_summary = summarize_raw(eligible_raw)
    else:
        eligible_summary = summary.iloc[0:0].copy()
    eligible_summary.to_csv(results_dir / "eligible_summary.csv", index=False)

    if write_tables:
        from .tables import write_tables as write_summary_tables

        write_summary_tables(results_dir)
    return summary_path
