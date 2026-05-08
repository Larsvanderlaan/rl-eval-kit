from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CRITIC_FAMILIES = ["linear_fqe", "rf_fqe", "neural_fqe"]
DEFAULT_CALIBRATORS = ["none", "linear", "isotonic", "histogram", "isotonic_histogram"]

REQUIRED_RESULT_COLUMNS = [
    "critic_family",
    "critic_config_id",
    "calibrator_config_id",
    "policy_id",
    "seed",
    "method",
    "weighting",
    "bellman_calibration_error_plugin",
    "bellman_calibration_error",
    "q_bellman_mse",
    "absolute_ope_error",
]

REQUIRED_SUMMARY_COLUMNS = [
    "critic_family",
    "critic_config_id",
    "calibrator_config_id",
    "method",
    "weighting",
    "n_rows",
    "n_policies",
    "n_seeds",
    "mean_bellman_calibration_error_plugin",
    "mean_bellman_calibration_error",
    "mean_q_bellman_mse",
    "relative_bellman_calibration_error_plugin",
    "relative_bellman_calibration_error",
    "relative_q_bellman_mse",
    "bellman_calibration_win_rate",
    "q_bellman_mse_win_rate",
    "relative_bellman_calibration_error_plugin_ci_low",
    "relative_bellman_calibration_error_plugin_ci_high",
    "relative_bellman_calibration_error_ci_low",
    "relative_bellman_calibration_error_ci_high",
    "relative_q_bellman_mse_ci_low",
    "relative_q_bellman_mse_ci_high",
]

FINITE_RESULT_METRICS = [
    "bellman_calibration_error_plugin",
    "q_bellman_mse",
    "absolute_ope_error",
]

FINITE_SUMMARY_METRICS = [
    "mean_bellman_calibration_error_plugin",
    "mean_q_bellman_mse",
    "relative_bellman_calibration_error_plugin",
    "relative_q_bellman_mse",
]

DEBIASED_METRICS = [
    "bellman_calibration_error",
]

DEBIASED_SUMMARY_METRICS = [
    "mean_bellman_calibration_error",
    "relative_bellman_calibration_error",
]

BOOTSTRAP_CI_COLUMNS = [
    "relative_bellman_calibration_error_plugin_ci_low",
    "relative_bellman_calibration_error_plugin_ci_high",
    "relative_bellman_calibration_error_ci_low",
    "relative_bellman_calibration_error_ci_high",
    "relative_q_bellman_mse_ci_low",
    "relative_q_bellman_mse_ci_high",
]

WIN_RATE_COLUMNS = [
    "bellman_calibration_win_rate",
    "q_bellman_mse_win_rate",
]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str, errors: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        errors.append(f"{label} missing columns: {missing}")


def _require_finite(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
    errors: list[str],
    mask: pd.Series | None = None,
) -> None:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad = int((~np.isfinite(values)).sum())
            errors.append(f"{label}.{col} has {bad} nonfinite values")


def _check_manifest(
    manifest: dict[str, object],
    *,
    expected_units: int | None,
    expected_rows: int | None,
    errors: list[str],
) -> None:
    if expected_units is not None:
        for key in ["n_expected_units", "n_completed_unit_files"]:
            if int(manifest.get(key, -1)) != int(expected_units):
                errors.append(f"manifest {key}={manifest.get(key)}; expected {expected_units}")
    if expected_rows is not None and int(manifest.get("n_result_rows", -1)) != int(expected_rows):
        errors.append(f"manifest n_result_rows={manifest.get('n_result_rows')}; expected {expected_rows}")
    if int(manifest.get("n_failed_units", 0)) != 0:
        errors.append(f"manifest n_failed_units={manifest.get('n_failed_units')}")


def _check_selected_configs(
    tuned_path: Path,
    *,
    critic_families: list[str],
    expected_method: str | None,
    forbid_method: str | None,
    errors: list[str],
) -> dict[str, dict[str, object]]:
    try:
        tuned = _read_json(tuned_path)
    except FileNotFoundError:
        errors.append(f"missing tuned config file: {tuned_path}")
        return {}
    selected = tuned.get("selected", {})
    if not isinstance(selected, dict):
        errors.append("tuned_configs.json has no selected config dictionary")
        return {}
    missing = [family for family in critic_families if family not in selected]
    if missing:
        errors.append(f"tuned_configs.json missing selected critics: {missing}")
    for family in critic_families:
        config = selected.get(family, {})
        if not isinstance(config, dict):
            errors.append(f"selected config for {family} is not a dictionary")
            continue
        for key in ["selection_mode", "selection_metrics", "source_tuning_mode", "selected_by_method"]:
            if key not in config:
                errors.append(f"selected config for {family} missing {key}")
        method = str(config.get("selected_by_method", ""))
        if expected_method is not None and method != expected_method:
            errors.append(f"selected method for {family}={method}; expected {expected_method}")
        if forbid_method is not None and method == forbid_method:
            errors.append(f"selected method for {family} should not be {forbid_method}")
    return {str(k): dict(v) for k, v in selected.items() if isinstance(v, dict)}


def _matching_summary_row(summary: pd.DataFrame, family: str, config: dict[str, object]) -> pd.DataFrame:
    mask = (
        summary["critic_family"].astype(str).eq(str(family))
        & summary["critic_config_id"].astype(str).eq(str(config.get("critic_config_id")))
        & summary["calibrator_config_id"].astype(str).eq(str(config.get("calibrator_config_id")))
        & summary["method"].astype(str).eq(str(config.get("selected_by_method")))
    )
    return summary.loc[mask].copy()


def _check_credible_calibrators(
    summary: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    *,
    critic_families: list[str],
    max_q_mse_regression: float,
    plugin_threshold: float,
    min_plugin_improvers: int,
    errors: list[str],
) -> None:
    plugin_improvers = 0
    for family in critic_families:
        config = selected.get(family, {})
        row = _matching_summary_row(summary, family, config)
        if row.empty:
            errors.append(f"no summary row for selected calibrated config {family}")
            continue
        rel_q = float(pd.to_numeric(row["relative_q_bellman_mse"], errors="coerce").iloc[0])
        rel_plugin = float(pd.to_numeric(row["relative_bellman_calibration_error_plugin"], errors="coerce").iloc[0])
        if not np.isfinite(rel_q) or rel_q > max_q_mse_regression:
            errors.append(f"{family} selected relative_q_bellman_mse={rel_q}; max {max_q_mse_regression}")
        if np.isfinite(rel_plugin) and rel_plugin <= plugin_threshold:
            plugin_improvers += 1
    if plugin_improvers < min_plugin_improvers:
        errors.append(
            "selected calibrated configs with "
            f"relative_bellman_calibration_error_plugin <= {plugin_threshold}: "
            f"{plugin_improvers}; expected at least {min_plugin_improvers}"
        )


def validate(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    errors: list[str] = []
    manifest = _read_json(output_dir / "hopper_q_calibration_manifest.json")
    results = _read_csv(output_dir / "hopper_q_calibration_results.csv")
    summary = _read_csv(output_dir / "hopper_q_calibration_summary.csv")

    _check_manifest(
        manifest,
        expected_units=args.expected_units,
        expected_rows=args.expected_rows,
        errors=errors,
    )
    if args.expected_rows is not None and len(results) != int(args.expected_rows):
        errors.append(f"results has {len(results)} rows; expected {args.expected_rows}")
    if args.expected_summary_rows is not None and len(summary) != int(args.expected_summary_rows):
        errors.append(f"summary has {len(summary)} rows; expected {args.expected_summary_rows}")

    _require_columns(results, REQUIRED_RESULT_COLUMNS, "results", errors)
    _require_columns(summary, REQUIRED_SUMMARY_COLUMNS, "summary", errors)
    _require_finite(results, FINITE_RESULT_METRICS, "results", errors)
    _require_finite(summary, FINITE_SUMMARY_METRICS, "summary", errors)
    calibrated_mask = summary["method"].astype(str).ne("none") if "method" in summary else None
    _require_finite(summary, WIN_RATE_COLUMNS, "summary", errors, mask=calibrated_mask)
    if args.require_debiased_metrics:
        _require_finite(results, DEBIASED_METRICS, "results", errors)
        _require_finite(summary, DEBIASED_SUMMARY_METRICS, "summary", errors)
    if args.require_bootstrap_ci:
        _require_finite(summary, BOOTSTRAP_CI_COLUMNS, "summary", errors)

    if args.require_weighting_none:
        for label, frame in [("results", results), ("summary", summary)]:
            if "weighting" not in frame:
                continue
            values = sorted(frame["weighting"].astype(str).unique())
            if values != ["none"]:
                errors.append(f"{label} weighting values are {values}; expected ['none']")

    if args.require_all_methods:
        for family in args.critic_families:
            family_methods = set(summary.loc[summary["critic_family"].astype(str).eq(family), "method"].astype(str))
            missing = sorted(set(DEFAULT_CALIBRATORS) - family_methods)
            if missing:
                errors.append(f"{family} summary missing methods: {missing}")

    selected: dict[str, dict[str, object]] = {}
    if args.require_selected_configs or args.expect_base_selection or args.expect_calibrator_selection:
        selected = _check_selected_configs(
            output_dir / "tuned_configs.json",
            critic_families=args.critic_families,
            expected_method="none" if args.expect_base_selection else None,
            forbid_method="none" if args.expect_calibrator_selection else None,
            errors=errors,
        )
    if args.require_credible_calibrator:
        if not selected:
            selected = _check_selected_configs(
                output_dir / "tuned_configs.json",
                critic_families=args.critic_families,
                expected_method=None,
                forbid_method="none",
                errors=errors,
            )
        _check_credible_calibrators(
            summary,
            selected,
            critic_families=args.critic_families,
            max_q_mse_regression=float(args.max_q_mse_regression),
            plugin_threshold=float(args.plugin_threshold),
            min_plugin_improvers=int(args.min_plugin_improvers),
            errors=errors,
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hopper Q-calibration benchmark outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_units", type=int, default=None)
    parser.add_argument("--expected_rows", type=int, default=None)
    parser.add_argument("--expected_summary_rows", type=int, default=None)
    parser.add_argument("--critic_families", nargs="+", default=DEFAULT_CRITIC_FAMILIES)
    parser.add_argument("--require_selected_configs", action="store_true")
    parser.add_argument("--expect_base_selection", action="store_true")
    parser.add_argument("--expect_calibrator_selection", action="store_true")
    parser.add_argument("--require_credible_calibrator", action="store_true")
    parser.add_argument("--require_weighting_none", action="store_true")
    parser.add_argument("--require_all_methods", action="store_true")
    parser.add_argument("--require_debiased_metrics", action="store_true")
    parser.add_argument("--require_bootstrap_ci", action="store_true")
    parser.add_argument("--max_q_mse_regression", type=float, default=1.05)
    parser.add_argument("--plugin_threshold", type=float, default=0.90)
    parser.add_argument("--min_plugin_improvers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        errors = validate(args)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"PASS: {args.output_dir}")


if __name__ == "__main__":
    main()
