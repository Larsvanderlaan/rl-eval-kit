#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROMOTION_MSE_THRESHOLD = 0.98
PROMOTION_DIAGNOSTIC_THRESHOLD = 0.95
PROMOTION_WIN_RATE_THRESHOLD = 0.60
RAW_CORRELATION_THRESHOLD = 0.0
SLOPE_MISCALIBRATION_TOL = 0.10
INTERCEPT_MISCALIBRATION_TOL = 0.10
FINAL_CLAIM_MODES = {"final", "paper"}
TUNING_MODES = {"debug", "pilot", "confirm"}
PRIMARY_CALIBRATORS = {"linear", "isotonic", "histogram", "isotonic_histogram"}


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _str(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[col].astype(str)


def _bool(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    vals = frame[col]
    if vals.dtype == bool:
        return vals
    return vals.astype(str).str.lower().isin({"true", "1", "yes"})


def _contains_test_or_oracle(values: pd.Series) -> bool:
    cleaned = values.astype(str).str.replace("not_test_or_oracle", "", regex=False)
    return bool(cleaned.str.contains("test|oracle", case=False, na=False).any())


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        else:
            out[col] = out[col].astype(str)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *rows])


def _strict_flags(raw: pd.DataFrame) -> pd.DataFrame:
    strict_group_cols = [
        "run_mode",
        "suite_name",
        "environment_name",
        "sample_size",
        "state_dimension",
        "coverage_setting",
        "misspecification_setting",
        "baseline_learner",
        "learner_variant",
        "calibration_protocol",
        "calibrator",
        "calibration_target",
    ]
    present = [col for col in strict_group_cols if col in raw]
    if raw.empty or not present:
        return pd.DataFrame(columns=present)
    grouped = raw.groupby(present, dropna=False).agg(
        rescue_raw_rows=("calibrated", "size"),
        rescue_base_all_data_rate=("base_learner_used_all_data", "mean")
        if "base_learner_used_all_data" in raw
        else ("calibrated", lambda _x: np.nan),
        rescue_any_failure=("failure_flag", "max")
        if "failure_flag" in raw
        else ("calibrated", lambda _x: True),
        rescue_calibration_uses_test_or_oracle=(
            "calibration_data_provenance",
            _contains_test_or_oracle,
        )
        if "calibration_data_provenance" in raw
        else ("calibrated", lambda _x: True),
        rescue_train_not_test_or_oracle=(
            "train_data_provenance",
            lambda x: bool(x.astype(str).str.contains("not_test_or_oracle", case=False, na=False).all()),
        )
        if "train_data_provenance" in raw
        else ("calibrated", lambda _x: False),
        rescue_test_independent=(
            "test_data_provenance",
            lambda x: bool(x.astype(str).str.contains("independent", case=False, na=False).all()),
        )
        if "test_data_provenance" in raw
        else ("calibrated", lambda _x: False),
    ).reset_index()
    return grouped


def audit_rescue_submission(results_dir: str | Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    raw_path = results_dir / "combined_raw_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing combined raw results: {raw_path}")
    summary = pd.read_csv(summary_path, low_memory=False)
    raw = pd.read_csv(raw_path, low_memory=False)

    flags = _strict_flags(raw)
    join_cols = [col for col in flags.columns if col in summary.columns and col not in {
        "rescue_raw_rows",
        "rescue_base_all_data_rate",
        "rescue_any_failure",
        "rescue_calibration_uses_test_or_oracle",
        "rescue_train_not_test_or_oracle",
        "rescue_test_independent",
    }]
    audit = summary.merge(flags, on=join_cols, how="left") if join_cols else summary.copy()

    run_mode = _str(audit, "run_mode", "standalone")
    suite_name = _str(audit, "suite_name", "")
    calibrated = _bool(audit, "calibrated", False)
    protocol = _str(audit, "calibration_protocol", "")
    calibrator = _str(audit, "calibrator", "")
    role = _str(audit, "main_figure_role", "main")

    rel_v = _num(audit, "relative_true_v_mse_vs_uncalibrated_all_data")
    rel_mse = _num(audit, "relative_mse_vs_uncalibrated_all_data")
    rel_cal = _num(audit, "relative_calibration_error_plugin_vs_uncalibrated_all_data")
    win_rate = _num(audit, "true_v_mse_win_rate_vs_uncalibrated_all_data")
    cal_win_rate = _num(audit, "calibration_error_plugin_win_rate_vs_uncalibrated_all_data")
    rel_v_retrain = _num(audit, "relative_true_v_mse_vs_current_retrain_small")
    rel_cal_retrain = _num(audit, "relative_calibration_error_plugin_vs_current_retrain_small")
    win_v_retrain = _num(audit, "true_v_mse_win_rate_vs_current_retrain_small")
    win_cal_retrain = _num(audit, "calibration_error_plugin_win_rate_vs_current_retrain_small")
    failure_rate = _num(audit, "failure_rate", 1.0).fillna(1.0)
    eligible = _num(audit, "eligible_fraction", 0.0).fillna(0.0)
    raw_spearman = _num(audit, "raw_value_oracle_spearman")
    raw_pearson = _num(audit, "raw_value_oracle_pearson")
    raw_slope = _num(audit, "raw_value_calibration_slope")
    raw_intercept = _num(audit, "raw_value_calibration_intercept")
    base_all_rate = _num(audit, "rescue_base_all_data_rate", 1.0).fillna(1.0)

    temporal = suite_name.eq("temporal_reward_shift_sweep")
    strict_protocol = protocol.eq("cross") | (temporal & protocol.eq("recent_heldout"))
    strict_data_use = (
        calibrated
        & strict_protocol
        & base_all_rate.eq(0.0)
        & ~_bool(audit, "rescue_calibration_uses_test_or_oracle", True)
        & _bool(audit, "rescue_train_not_test_or_oracle", False)
        & _bool(audit, "rescue_test_independent", False)
    )
    raw_rank_informative = raw_spearman.gt(RAW_CORRELATION_THRESHOLD) & raw_pearson.gt(RAW_CORRELATION_THRESHOLD)
    raw_miscalibrated = (
        raw_slope.sub(1.0).abs().gt(SLOPE_MISCALIBRATION_TOL)
        | raw_intercept.abs().gt(INTERCEPT_MISCALIBRATION_TOL)
    )
    finite_metrics = pd.concat(
        [
            rel_v,
            rel_cal,
            win_rate,
            cal_win_rate,
            raw_spearman,
            raw_pearson,
            raw_slope,
            raw_intercept,
        ],
        axis=1,
    ).apply(lambda row: bool(np.all(np.isfinite(row.to_numpy(dtype=float)))), axis=1)
    value_improved = rel_v.lt(PROMOTION_MSE_THRESHOLD)
    scalar_mse_improved = rel_mse.lt(PROMOTION_MSE_THRESHOLD)
    bellman_supported = rel_cal.lt(PROMOTION_DIAGNOSTIC_THRESHOLD)
    stable = (
        failure_rate.eq(0.0)
        & eligible.ge(1.0)
        & finite_metrics
        & win_rate.ge(PROMOTION_WIN_RATE_THRESHOLD)
        & cal_win_rate.ge(PROMOTION_WIN_RATE_THRESHOLD)
    )
    temporal_retrain_supported = ~temporal | (
        protocol.eq("recent_heldout")
        & rel_v_retrain.le(1.05)
        & rel_cal_retrain.le(1.05)
        & win_v_retrain.ge(PROMOTION_WIN_RATE_THRESHOLD)
        & win_cal_retrain.ge(PROMOTION_WIN_RATE_THRESHOLD)
    )
    audit_pass = (
        strict_data_use
        & raw_rank_informative
        & raw_miscalibrated
        & value_improved
        & bellman_supported
        & stable
        & temporal_retrain_supported
    )

    audit["rescue_strict_data_use_pass"] = strict_data_use
    audit["rescue_raw_rank_informative"] = raw_rank_informative
    audit["rescue_raw_miscalibrated"] = raw_miscalibrated
    audit["rescue_value_improved"] = value_improved
    audit["rescue_bellman_supported"] = bellman_supported
    audit["rescue_temporal_retrain_supported"] = temporal_retrain_supported
    audit["rescue_stable_and_finite"] = stable
    audit["rescue_audit_pass"] = audit_pass
    audit["rescue_claim_mode"] = run_mode.isin(FINAL_CLAIM_MODES)
    audit["rescue_tuning_only"] = run_mode.isin(TUNING_MODES)

    label = np.full(audit.shape[0], "limitation", dtype=object)
    bad_data = calibrated & ~strict_data_use
    unstable = calibrated & strict_data_use & (~stable)
    mse_only = calibrated & strict_data_use & stable & (value_improved | scalar_mse_improved) & ~bellman_supported
    label[bad_data.to_numpy()] = "reject_leakage_or_data_use"
    label[unstable.to_numpy()] = "reject_unstable"
    label[mse_only.to_numpy()] = "reject_mse_only"
    label[(audit_pass & role.eq("mechanism")).to_numpy()] = "mechanism_only"
    label[(audit_pass & run_mode.isin(TUNING_MODES)).to_numpy()] = "tuning_only"
    label[(audit_pass & run_mode.isin(FINAL_CLAIM_MODES) & role.eq("appendix")).to_numpy()] = "appendix_support"
    label[(audit_pass & run_mode.isin(FINAL_CLAIM_MODES) & role.eq("main")).to_numpy()] = "promote_main"
    validation_control = suite_name.str.contains("well_specified|validation", case=False, na=False)
    label[validation_control.to_numpy()] = "validation_control"
    audit["rescue_audit_label"] = label

    audit_path = results_dir / "rescue_promotion_audit.csv"
    manifest_path = results_dir / "do_not_claim_manifest.csv"
    readout_path = results_dir / "rescue_readout.md"
    json_path = results_dir / "rescue_audit_summary.json"
    audit.to_csv(audit_path, index=False)

    do_not_claim = audit[
        audit["rescue_audit_label"].isin(
            {
                "reject_leakage_or_data_use",
                "reject_mse_only",
                "reject_unstable",
                "limitation",
                "mechanism_only",
                "tuning_only",
                "validation_control",
            }
        )
    ].copy()
    do_not_claim.to_csv(manifest_path, index=False)

    promoted = audit[audit["rescue_audit_label"].eq("promote_main")].copy()
    appendix = audit[audit["rescue_audit_label"].eq("appendix_support")].copy()
    counts = audit["rescue_audit_label"].value_counts(dropna=False).to_dict()
    organic_promoted = promoted[
        ~promoted.get("main_figure_role", pd.Series("", index=promoted.index)).astype(str).eq("mechanism")
        & ~promoted.get("suite_name", pd.Series("", index=promoted.index)).astype(str).str.contains(
            "well_specified|validation", case=False, na=False
        )
    ]
    payload = {
        "results_dir": str(results_dir),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "n_promote_main": int(promoted.shape[0]),
        "n_appendix_support": int(appendix.shape[0]),
        "n_organic_promote_main": int(organic_promoted.shape[0]),
        "thresholds": {
            "relative_true_v_mse": PROMOTION_MSE_THRESHOLD,
            "relative_bellman_calibration_error": PROMOTION_DIAGNOSTIC_THRESHOLD,
            "true_v_win_rate": PROMOTION_WIN_RATE_THRESHOLD,
            "raw_correlation": RAW_CORRELATION_THRESHOLD,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    display_cols = [
        "run_mode",
        "suite_name",
        "learner_variant",
        "calibrator",
        "misspecification_setting",
        "relative_true_v_mse_vs_uncalibrated_all_data",
        "relative_calibration_error_plugin_vs_uncalibrated_all_data",
        "relative_true_v_mse_vs_current_retrain_small",
        "relative_calibration_error_plugin_vs_current_retrain_small",
        "true_v_mse_win_rate_vs_uncalibrated_all_data",
        "calibration_error_plugin_win_rate_vs_uncalibrated_all_data",
        "raw_value_oracle_spearman",
        "raw_value_calibration_slope",
        "rescue_audit_label",
    ]
    present_cols = [col for col in display_cols if col in audit]
    lines = [
        "# Rescue Promotion Audit",
        "",
        f"Results directory: `{results_dir}`",
        "",
        "## Gate",
        "",
        "Rows are promotable only with strict cross-calibration or recent-heldout temporal calibration, positive raw value/oracle correlation, raw miscalibration, true-V MSE improvement, Bellman calibration-error improvement, value/calibration win rates >= 0.60, zero failures, finite diagnostics, and no test/oracle/no-split leakage.",
        "",
        "## Label Counts",
        "",
        *[f"- `{key}`: {value}" for key, value in sorted(counts.items())],
        "",
        "## Promoted Main Rows",
        "",
        _markdown_table(promoted[present_cols].head(20)),
        "",
        "## Appendix Support Rows",
        "",
        _markdown_table(appendix[present_cols].head(20)),
        "",
        "## Do-Not-Claim Rows",
        "",
        _markdown_table(do_not_claim[present_cols].head(30)),
        "",
    ]
    readout_path.write_text("\n".join(lines))
    return {"audit": audit_path, "manifest": manifest_path, "readout": readout_path, "summary": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit staged rescue calibration results.")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    outputs = audit_rescue_submission(args.results_dir)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
