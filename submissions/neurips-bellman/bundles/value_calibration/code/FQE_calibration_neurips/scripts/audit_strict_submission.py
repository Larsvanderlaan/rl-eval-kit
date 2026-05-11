#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PRIMARY_CALIBRATORS = {"histogram", "isotonic", "isotonic_histogram"}
ORGANIC_SUITES = {"model_misspecification_sweep", "bellman_incomplete_sweep"}


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
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


def _ratio_col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce")


def audit_strict_submission(results_dir: str | Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    raw_path = results_dir / "combined_raw_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing combined raw results: {raw_path}")

    summary = pd.read_csv(summary_path)
    raw = pd.read_csv(raw_path, low_memory=False)

    strict_raw = raw[
        raw.get("calibration_protocol", "").astype(str).eq("cross")
        & raw.get("calibrated", False).astype(bool)
    ].copy()
    strict_group_cols = [
        "suite_name",
        "baseline_learner",
        "learner_variant",
        "calibration_protocol",
        "calibrator",
        "calibration_target",
        "sample_size",
        "state_dimension",
        "coverage_setting",
        "misspecification_setting",
    ]
    strict_flags = strict_raw.groupby(strict_group_cols, dropna=False).agg(
        strict_cross_base_all_data_rate=("base_learner_used_all_data", "mean"),
        strict_cross_rows=("base_learner_used_all_data", "size"),
    ).reset_index()

    audit = summary.copy()
    audit = audit.merge(strict_flags, on=[c for c in strict_group_cols if c in audit.columns], how="left")
    is_cross_cal = audit.get("calibration_protocol", "").astype(str).eq("cross") & audit.get("calibrated", False).astype(bool)
    rel_v = _ratio_col(audit, "relative_true_v_mse_vs_uncalibrated_all_data")
    rel_cal = _ratio_col(audit, "relative_calibration_error_plugin_vs_uncalibrated_all_data")
    win_rate = _ratio_col(audit, "true_v_mse_win_rate_vs_uncalibrated_all_data")
    failure_rate = _ratio_col(audit, "failure_rate").fillna(1.0)
    eligible = _ratio_col(audit, "eligible_fraction").fillna(0.0)
    base_all_rate = _ratio_col(audit, "strict_cross_base_all_data_rate").fillna(1.0)
    audit["strict_cross_audit_pass"] = (
        is_cross_cal
        & rel_v.lt(0.98)
        & rel_cal.lt(0.95)
        & win_rate.ge(0.60)
        & failure_rate.le(0.0)
        & eligible.ge(1.0)
        & base_all_rate.eq(0.0)
    )
    audit["strict_cross_primary_calibrator"] = audit.get("calibrator", "").astype(str).isin(PRIMARY_CALIBRATORS)
    audit["strict_cross_organic_candidate"] = audit.get("suite_name", "").astype(str).isin(ORGANIC_SUITES)
    audit["strict_cross_promote_to_main"] = (
        audit["strict_cross_audit_pass"]
        & audit["strict_cross_primary_calibrator"]
        & audit.get("main_figure_role", "").astype(str).eq("main")
    )

    audit_path = results_dir / "strict_cross_promotion_audit.csv"
    audit.to_csv(audit_path, index=False)

    promoted = audit[audit["strict_cross_promote_to_main"]].copy()
    organic = promoted[promoted["strict_cross_organic_candidate"]].copy()
    if not organic.empty:
        organic = organic.sort_values(
            [
                "relative_true_v_mse_vs_uncalibrated_all_data",
                "relative_calibration_error_plugin_vs_uncalibrated_all_data",
            ],
            na_position="last",
        )
    finite = promoted[promoted.get("suite_name", "").astype(str).eq("undertraining_sweep")].copy()

    lines = [
        "# Strict Cross-Calibration Submission Audit",
        "",
        f"Results directory: `{results_dir}`",
        "",
        "## Promotion Rule",
        "",
        "A row is promotable only if it is strict cross-calibrated, uses no calibrated full-data refit, has no failures, improves held-out true-V MSE, improves Bellman calibration error, and wins on true-V MSE in at least 60% of replications.",
        "",
        "## Summary",
        "",
        f"- Promotable primary rows: {int(audit['strict_cross_promote_to_main'].sum())}",
        f"- Promotable organic rows: {int((audit['strict_cross_promote_to_main'] & audit['strict_cross_organic_candidate']).sum())}",
        f"- Promotable finite-iteration rows: {int((audit['strict_cross_promote_to_main'] & audit.get('suite_name', '').astype(str).eq('undertraining_sweep')).sum())}",
        "",
    ]
    if organic.empty:
        lines.extend(
            [
                "## Organic Regime Decision",
                "",
                "No model-misspecification row passed the strict promotion rule. Keep the main text focused on finite-iteration evidence.",
                "",
            ]
        )
    else:
        lines.extend(["## Best Organic Rows", ""])
        cols = [
            "suite_name",
            "learner_variant",
            "calibrator",
            "misspecification_setting",
            "relative_true_v_mse_vs_uncalibrated_all_data",
            "relative_calibration_error_plugin_vs_uncalibrated_all_data",
            "true_v_mse_win_rate_vs_uncalibrated_all_data",
        ]
        lines.append(_markdown_table(organic[cols].head(8)))
        lines.append("")
    if not finite.empty:
        lines.extend(["## Best Finite-Iteration Rows", ""])
        cols = [
            "learner_variant",
            "calibrator",
            "relative_true_v_mse_vs_uncalibrated_all_data",
            "relative_calibration_error_plugin_vs_uncalibrated_all_data",
            "true_v_mse_win_rate_vs_uncalibrated_all_data",
        ]
        lines.append(_markdown_table(finite[cols].sort_values("relative_true_v_mse_vs_uncalibrated_all_data").head(8)))
        lines.append("")

    readout_path = results_dir / "strict_cross_readout.md"
    readout_path.write_text("\n".join(lines) + "\n")
    return {"audit": audit_path, "readout": readout_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict pointwise-median cross-calibration results.")
    parser.add_argument("--results_dir", default="FQE_calibration_neurips/results/paper_strict_cross_submission_v2")
    args = parser.parse_args()
    outputs = audit_strict_submission(args.results_dir)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
