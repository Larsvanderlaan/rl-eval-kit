#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _median_iqr(row: pd.Series, metric: str) -> str:
    med = row.get(f"{metric}_median")
    q25 = row.get(f"{metric}_q25")
    q75 = row.get(f"{metric}_q75")
    if pd.isna(med):
        return ""
    if pd.isna(q25) or pd.isna(q75):
        return f"{med:.3g}"
    return f"{med:.3g} [{q25:.3g}, {q75:.3g}]"


def _median_iqr_values(values: pd.Series) -> str:
    arr = values.dropna()
    if arr.empty:
        return ""
    med = float(arr.median())
    q25 = float(arr.quantile(0.25))
    q75 = float(arr.quantile(0.75))
    return f"{med:.3g} [{q25:.3g}, {q75:.3g}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create compact LaTeX tables for the soft-FQI study.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tables-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else ROOT / "tables" / results_dir.name
    tables_dir.mkdir(parents=True, exist_ok=True)
    final_path = results_dir / "summary_final.csv"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing {final_path}; run aggregate_results.py first.")
    final = pd.read_csv(final_path)
    keep_cols = [
        "stage",
        "learner",
        "q_class",
        "regime",
        "schedule",
        "method",
        "gamma_weight",
        "stationary_q_rmse_median",
        "stationary_advantage_q_rmse_median",
        "behavior_q_rmse_median",
        "stationary_bellman_rmse_median",
        "stationary_projected_bellman_rmse_median",
        "behavior_projected_bellman_rmse_median",
        "policy_value_error_median",
        "diverged_mean",
        "failed_mean",
        "minimax_residual_norm_median",
        "minimax_critic_norm_median",
        "minimax_q_ridge_median",
        "minimax_critic_ridge_median",
    ]
    keep_cols = [col for col in keep_cols if col in final.columns]
    table = final[keep_cols].copy()
    csv_path = tables_dir / "main_final_summary.csv"
    tex_path = tables_dir / "main_final_summary.tex"
    table.to_csv(csv_path, index=False)
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, float_format="%.3g", escape=True))
    main = final[
        (final["stage"].astype(str) == "stage2_weights")
        & (final["learner"].astype(str) == "linear")
        & (final["method"].astype(str).isin(["unweighted", "oracle", "estimated_g0p95", "estimated_g1p0", "minimax"]))
    ].copy()
    if not main.empty:
        main["method_label"] = main["method"].replace(
            {
                "unweighted": "Unweighted",
                "oracle": "Oracle stationary",
                "estimated_g0p95": "Estimated $\\gamma_w=0.95$",
                "estimated_g1p0": "Estimated $\\gamma_w=1$",
                "minimax": "Minimax soft Q",
            }
        )
        compact_rows = []
        for _, row in main.sort_values(["schedule", "regime", "method"]).iterrows():
            compact_rows.append(
                {
                    "schedule": row["schedule"],
                    "regime": row["regime"],
                    "method": row["method_label"],
                    "stationary Q RMSE": _median_iqr(row, "stationary_q_rmse"),
                    "stationary advantage RMSE": _median_iqr(row, "stationary_advantage_q_rmse"),
                    "behavior Q RMSE": _median_iqr(row, "behavior_q_rmse"),
                    "stationary PBE": _median_iqr(row, "stationary_projected_bellman_rmse"),
                    "value error": _median_iqr(row, "policy_value_error"),
                    "failure rate": f"{row.get('failed_mean', 0.0):.2f}",
                }
            )
        compact = pd.DataFrame(compact_rows)
        compact_csv = tables_dir / "main_text_summary.csv"
        compact_tex = tables_dir / "main_text_summary.tex"
        compact.to_csv(compact_csv, index=False)
        with compact_tex.open("w", encoding="utf-8") as handle:
            handle.write(compact.to_latex(index=False, escape=False))
        print(f"Wrote {compact_csv}")
        print(f"Wrote {compact_tex}")
    raw_path = results_dir / "raw_results.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        if {"is_final", "failed", "method", "gamma_weight"}.issubset(raw.columns):
            final_raw = raw[
                (raw["is_final"] == 1)
                & (raw["failed"] == 0)
                & raw["method"].astype(str).str.startswith("estimated_g")
            ].copy()
        else:
            final_raw = pd.DataFrame()
        if not final_raw.empty and final_raw["gamma_weight"].nunique(dropna=True) > 1:
            rows = []
            group_cols = ["stage", "regime", "schedule", "method", "gamma_weight"]
            if "q_class" in final_raw.columns:
                group_cols.append("q_class")
            for keys, sub in final_raw.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                out = dict(zip(group_cols, keys, strict=True))
                cv_series = sub["cv_ridge_selected"] if "cv_ridge_selected" in sub.columns else pd.Series([0.0])
                cv_flag = float(cv_series.fillna(0.0).median())
                if "cv_selected_ridge" in sub.columns and cv_flag > 0.5:
                    selected_ridge = float(sub["cv_selected_ridge"].dropna().median())
                elif "ridge_primal" in sub.columns:
                    selected_ridge = float(sub["ridge_primal"].dropna().median())
                else:
                    selected_ridge = float("nan")
                out.update(
                    {
                        "ridge_mode": "CV" if cv_flag > 0.5 else "fixed",
                        "selected_ridge": selected_ridge,
                        "ESS fraction": _median_iqr_values(sub.get("effective_sample_size_fraction", pd.Series(dtype=float))),
                        "Projected Bellman": _median_iqr_values(
                            sub.get("stationary_projected_bellman_rmse", pd.Series(dtype=float))
                        ),
                        "Stationary advantage": _median_iqr_values(
                            sub.get("stationary_advantage_q_rmse", pd.Series(dtype=float))
                        ),
                        "Value error": _median_iqr_values(sub.get("policy_value_error", pd.Series(dtype=float))),
                    }
                )
                rows.append(out)
            ablation = pd.DataFrame(rows).sort_values(group_cols)
            ablation_csv = tables_dir / "weight_estimator_ablation_table.csv"
            ablation_tex = tables_dir / "weight_estimator_ablation_table.tex"
            ablation.to_csv(ablation_csv, index=False)
            with ablation_tex.open("w", encoding="utf-8") as handle:
                handle.write(ablation.to_latex(index=False, escape=False))
            print(f"Wrote {ablation_csv}")
            print(f"Wrote {ablation_tex}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
