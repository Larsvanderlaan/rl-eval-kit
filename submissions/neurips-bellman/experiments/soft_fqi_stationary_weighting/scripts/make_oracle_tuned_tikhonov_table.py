#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "tables" / "overnight_decision"

METRICS = [
    ("stationary_bellman_rmse", "Bellman RMSE", 1.0),
    ("stationary_advantage_q_rmse", "Advantage x1e3", 1000.0),
    ("stationary_q_rmse", "True Q RMSE", 1.0),
    ("policy_value_error", "Value error", 1.0),
]
ORACLE_CRITERIA = [
    ("stationary_bellman_rmse", "Bellman RMSE"),
]


def _read_final(name: str) -> pd.DataFrame:
    raw = pd.read_csv(RESULTS / name / "raw_results.csv", low_memory=False)
    return raw[(raw["failed"] == 0) & (raw["is_final"] == 1)].copy()


def _median_iqr(values: pd.Series, scale: float = 1.0) -> str:
    arr = values.dropna().to_numpy(dtype=float) * scale
    if arr.size == 0:
        return ""
    return f"{np.median(arr):.3g} [{np.quantile(arr, 0.25):.3g}, {np.quantile(arr, 0.75):.3g}]"


def _summarize_variants(
    df: pd.DataFrame,
    family: str,
    variant_cols: list[str],
    *,
    extra_medians: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    group_cols = ["regime"] + variant_cols
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"family": family}
        row.update(dict(zip(group_cols, keys, strict=True)))
        for metric, _, _ in METRICS:
            row[f"{metric}_median"] = float(sub[metric].median()) if metric in sub else np.nan
        for col in extra_medians or []:
            if col in sub:
                row[col] = float(sub[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_label(row: pd.Series) -> str:
    family = row["family"]
    if family == "estimated weights":
        gamma = row.get("gamma_weight", np.nan)
        gamma_text = "" if pd.isna(gamma) else f"$\\gamma_w={gamma:g}$"
        ridge_mode = str(row.get("ridge_mode", "")).replace("nan", "")
        selected = row.get("selected_ridge", np.nan)
        ridge_text = ""
        if not pd.isna(selected):
            ridge_text = f"$\\lambda={selected:g}$"
        return ", ".join(part for part in [gamma_text, ridge_mode, ridge_text] if part)
    stage = str(row.get("stage", ""))
    if stage.startswith("minimax_ridge_"):
        suffix = stage.removeprefix("minimax_ridge_")
        ridge = float(suffix.replace("p", "."))
        exponent = int(np.floor(np.log10(ridge)))
        mantissa = ridge / (10.0**exponent)
        if np.isclose(mantissa, 1.0):
            ridge_text = f"10^{{{exponent}}}"
        else:
            ridge_text = f"{mantissa:g}\\times 10^{{{exponent}}}"
        return f"$\\lambda_q=\\lambda_c={ridge_text}$"
    if stage == "minimax_cv_one_se":
        return "CV one-SE ridge"
    return stage.replace("_", " ")


def main() -> None:
    estimated = _read_final("weight_ablation_gamma_cv")
    estimated = estimated[estimated["method"].astype(str).str.startswith("estimated")].copy()
    estimated["ridge_mode"] = np.where(estimated["stage"].astype(str).str.contains("cv"), "CV", "fixed")
    # ridge_primal stores the actual Tikhonov value used after CV selection;
    # cv_ridge_selected is an internal selection code in some older outputs.
    estimated["selected_ridge"] = estimated["ridge_primal"]
    est_variants = _summarize_variants(
        estimated,
        "estimated weights",
        ["stage", "gamma_weight", "ridge_mode"],
        extra_medians=["selected_ridge"],
    )

    minimax = _read_final("minimax_tuning_fairness")
    minimax = minimax[minimax["method"].astype(str) == "minimax"].copy()
    mm_variants = _summarize_variants(minimax, "minimax soft Q", ["stage"])

    variants = pd.concat([est_variants, mm_variants], ignore_index=True)
    rows = []
    for (family, regime), sub in variants.groupby(["family", "regime"], dropna=False):
        for metric, metric_label in ORACLE_CRITERIA:
            best = sub.sort_values(f"{metric}_median", ascending=True).iloc[0]
            source = estimated if family == "estimated weights" else minimax
            mask = source["regime"].astype(str).eq(str(regime))
            if family == "estimated weights":
                mask &= source["stage"].astype(str).eq(str(best["stage"]))
                mask &= source["gamma_weight"].astype(float).eq(float(best["gamma_weight"]))
                mask &= source["ridge_mode"].astype(str).eq(str(best["ridge_mode"]))
            else:
                mask &= source["stage"].astype(str).eq(str(best["stage"]))
            chosen = source[mask]
            row = {
                "family": family,
                "regime": str(regime).replace("shift_", "shift "),
                "oracle criterion": metric_label,
                "selected Tikhonov/target": _variant_label(best),
            }
            for metric_name, label, metric_scale in METRICS:
                row[label] = _median_iqr(chosen[metric_name], metric_scale)
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["family", "regime", "oracle criterion"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "oracle_tuned_tikhonov_comparison.csv", index=False)
    (OUT / "oracle_tuned_tikhonov_comparison.tex").write_text(
        table.to_latex(index=False, escape=False), encoding="utf-8"
    )
    print(f"Wrote {OUT / 'oracle_tuned_tikhonov_comparison.csv'}")


if __name__ == "__main__":
    main()
