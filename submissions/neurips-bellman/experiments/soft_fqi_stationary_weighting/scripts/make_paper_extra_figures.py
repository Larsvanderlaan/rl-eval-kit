#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = ROOT / "results"
OUT = ROOT / "final_draft_bundle"
REGIMES = ["shift_001", "shift_09", "shift_18", "shift_27", "shift_36", "shift_45", "shift_54", "shift_63", "shift_72", "shift_80"]
COLORS = {
    "unweighted": "#4C4C4C",
    "oracle": "#0072B2",
    "estimated_g0p95": "#009E73",
    "minimax": "#CC79A7",
}
LABELS = {
    "unweighted": "Unweighted",
    "oracle": "Oracle stationary",
    "estimated_g0p95": "Estimated stationary",
    "minimax": "Minimax soft Q",
}
REGIME_COLORS = {
    "shift_27": "#0072B2",
    "shift_45": "#D55E00",
    "shift_72": "#009E73",
    "shift_80": "#CC79A7",
}


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name / "raw_results.csv", low_memory=False)


def _final(df: pd.DataFrame, *, schedule: str = "direct") -> pd.DataFrame:
    out = df[(df["failed"] == 0) & (df["is_final"] == 1)].copy()
    if "schedule" in out:
        out = out[out["schedule"].astype(str) == schedule]
    return out


def _summ(df: pd.DataFrame, metric: str, groups: list[str]) -> pd.DataFrame:
    return (
        df.groupby(groups, dropna=False)[metric]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )


def _final_main_comparison() -> pd.DataFrame:
    main = _final(_read("paper_main_500_stabilized_10regime"))
    main = main[
        (main["stage"].astype(str) == "stage2_weights")
        & (main["learner"].astype(str) == "linear")
        & (main["method"].astype(str).isin(["unweighted", "oracle", "estimated_g0p95"]))
    ]
    minimax = _final(_read("main_linear_minimax_10regime"))
    minimax = minimax[minimax["method"].astype(str) == "minimax"]
    data = pd.concat([main, minimax], ignore_index=True)
    return data[data["regime"].astype(str).isin(REGIMES)].copy()


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def minimax_vs_weighting() -> None:
    data = _final_main_comparison()
    fig, axes = plt.subplots(1, 3, figsize=(8.9, 2.75), sharex=True)
    metrics = [
        ("stationary_advantage_q_rmse", "A. Advantage error ($\\times 10^3$)", 1000.0),
        ("stationary_bellman_rmse", "B. Bellman residual", 1.0),
        ("stationary_q_rmse", "C. True $Q$ RMSE", 1.0),
    ]
    methods = ["unweighted", "oracle", "estimated_g0p95", "minimax"]
    x = np.arange(len(REGIMES))
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        tab = _summ(data, metric, ["regime", "method"])
        for method in methods:
            vals = []
            lo = []
            hi = []
            for regime in REGIMES:
                row = tab[(tab["regime"].astype(str) == regime) & (tab["method"].astype(str) == method)]
                if row.empty:
                    vals.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    continue
                med = float(row["median"].iloc[0]) * scale
                q25 = float(row["q25"].iloc[0]) * scale
                q75 = float(row["q75"].iloc[0]) * scale
                vals.append(med)
                lo.append(med - q25)
                hi.append(q75 - med)
            ax.errorbar(
                x,
                vals,
                yerr=np.vstack([lo, hi]),
                marker="o",
                linewidth=1.6,
                markersize=3.2,
                capsize=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(ylabel, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("shift_", "") for r in REGIMES], rotation=0)
        ax.set_xlabel("behavior shift")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("median [IQR]")
    axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.02))
    fig.tight_layout(w_pad=1.2)
    _save(fig, "minimax_vs_weighting_overlap_regimes")


def supporting_diagnostics() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.2))
    axes = axes.ravel()

    rich = _final(_read("negative_control_rich_q"))
    rich_regimes = ["shift_001", "shift_27", "shift_45", "shift_72"]
    tab = _summ(rich, "stationary_bellman_rmse", ["regime", "method"])
    for method in ["unweighted", "oracle", "estimated_g0p95", "minimax"]:
        y = []
        for regime in rich_regimes:
            row = tab[(tab["regime"].astype(str) == regime) & (tab["method"].astype(str) == method)]
            y.append(float(row["median"].iloc[0]) if not row.empty else np.nan)
        axes[0].plot(range(len(rich_regimes)), y, marker="o", color=COLORS[method], label=LABELS[method], linewidth=1.4)
    axes[0].set_title("A. Rich $Q$ features", loc="left")
    axes[0].set_xticks(range(len(rich_regimes)))
    axes[0].set_xticklabels([r.replace("shift_", "") for r in rich_regimes])
    axes[0].set_ylabel("Bellman residual")
    axes[0].grid(alpha=0.22)

    sample = _final(_read("sample_size_ess_sensitivity"))
    piv = _summ(sample, "stationary_bellman_rmse", ["stage", "regime", "method"])
    ess = _summ(sample[sample["method"].astype(str) == "estimated_g0p95"], "effective_sample_size_fraction", ["stage", "regime"])
    stages = sorted(sample["stage"].astype(str).unique(), key=lambda s: int(s.split("n")[-1]))
    for regime, color in zip(["shift_27", "shift_45", "shift_72"], ["#0072B2", "#D55E00", "#009E73"]):
        xs, ys = [], []
        for stage in stages:
            est = piv[(piv["stage"].astype(str) == stage) & (piv["regime"].astype(str) == regime) & (piv["method"].astype(str) == "estimated_g0p95")]
            ora = piv[(piv["stage"].astype(str) == stage) & (piv["regime"].astype(str) == regime) & (piv["method"].astype(str) == "oracle")]
            ess_row = ess[(ess["stage"].astype(str) == stage) & (ess["regime"].astype(str) == regime)]
            if est.empty or ora.empty:
                continue
            xs.append(float(ess_row["median"].iloc[0]) if not ess_row.empty else np.nan)
            ys.append(float(est["median"].iloc[0] - ora["median"].iloc[0]))
        axes[1].axhline(0.0, color="#888888", linewidth=0.8, alpha=0.6)
        axes[1].plot(xs, ys, marker="o", color=color, linewidth=1.4, label=regime.replace("shift_", "shift "))
    axes[1].set_title("B. Estimated-weight ESS", loc="left")
    axes[1].set_xlabel("estimated ESS fraction")
    axes[1].set_ylabel("Bellman residual gap to oracle")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7)

    gamma = _final(_read("weight_ablation_gamma_cv"))
    gamma = gamma[gamma["stage"].astype(str) == "gamma_ablation_fixed"]
    gtab = _summ(gamma, "stationary_bellman_rmse", ["regime", "gamma_weight"])
    for regime, color in zip(["shift_45", "shift_72", "shift_80"], ["#0072B2", "#D55E00", "#009E73"]):
        sub = gtab[gtab["regime"].astype(str) == regime].sort_values("gamma_weight")
        axes[2].plot(sub["gamma_weight"], sub["median"], marker="o", color=color, linewidth=1.4, label=regime.replace("shift_", "shift "))
    axes[2].set_title("C. Weight target $\\gamma_w$", loc="left")
    axes[2].set_xlabel("$\\gamma_w$")
    axes[2].set_ylabel("Bellman residual")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=False, fontsize=7)

    gamma_all = _final(_read("weight_ablation_gamma_cv"))
    gamma_all["ridge_mode"] = np.where(gamma_all["stage"].astype(str).str.contains("cv"), "CV", "fixed")
    etab = _summ(gamma_all, "effective_sample_size_fraction", ["regime", "gamma_weight", "ridge_mode"])
    for mode, marker in [("fixed", "o"), ("CV", "s")]:
        sub = etab[(etab["regime"].astype(str) == "shift_80") & (etab["ridge_mode"].astype(str) == mode)].sort_values("gamma_weight")
        axes[3].plot(sub["gamma_weight"], sub["median"], marker=marker, color="#CC79A7", linewidth=1.4, label=mode)
    axes[3].set_title("D. Severe-shift ESS", loc="left")
    axes[3].set_xlabel("$\\gamma_w$")
    axes[3].set_ylabel("ESS fraction")
    axes[3].set_ylim(bottom=0)
    axes[3].grid(alpha=0.22)
    axes[3].legend(frameon=False)

    axes[0].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout(h_pad=1.4, w_pad=1.3)
    _save(fig, "softfqi_supporting_diagnostics")


def minimax_oracle_sensitivity() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.2))
    axes = axes.ravel()

    tune = _final(_read("minimax_tuning_fairness"))
    stages = [
        "minimax_ridge_1e-6",
        "minimax_ridge_3e-6",
        "minimax_ridge_1e-5",
        "minimax_ridge_3e-5",
        "minimax_ridge_1e-4",
        "minimax_ridge_3e-4",
        "minimax_ridge_1e-3",
        "minimax_ridge_3e-3",
        "minimax_ridge_1e-2",
        "minimax_ridge_3e-2",
        "minimax_ridge_1e-1",
        "minimax_cv_one_se",
    ]
    labels = ["1e-6", "3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "CV"]
    fixed_stages = stages[:-1]
    fixed_labels = labels[:-1]
    for ax, metric, title, scale in [
        (axes[0], "stationary_bellman_rmse", "A. Bellman-tuned minimax", 1.0),
        (axes[1], "stationary_q_rmse", "B. True $Q$ error", 1.0),
        (axes[2], "stationary_advantage_q_rmse", "C. Advantage error", 1000.0),
    ]:
        tab = _summ(tune, metric, ["stage", "regime"])
        for regime, color in REGIME_COLORS.items():
            y = []
            for stage in fixed_stages:
                row = tab[(tab["stage"].astype(str) == stage) & (tab["regime"].astype(str) == regime)]
                y.append(scale * float(row["median"].iloc[0]) if not row.empty else np.nan)
            ax.plot(range(len(fixed_stages)), y, marker="o", linewidth=1.25, color=color, label=regime.replace("shift_", "shift "))
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(fixed_stages)))
        ax.set_xticklabels(fixed_labels, rotation=35, ha="right")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Bellman RMSE")
    axes[1].set_ylabel("true $Q$ RMSE")
    axes[2].set_ylabel("advantage RMSE ($\\times 10^3$)")
    axes[0].legend(frameon=False, fontsize=6, ncol=2, loc="best")

    oracle = _final(_read("oracle_stabilization_sensitivity"))
    otab = _summ(oracle, "stationary_advantage_q_rmse", ["stage", "regime"])
    oracle_stages = ["oracle_raw", "oracle_ess025", "oracle_ess050"]
    oracle_labels = ["raw", "ESS .25", "ESS .50"]
    for regime, color in zip(["shift_45", "shift_72", "shift_80"], ["#0072B2", "#D55E00", "#009E73"]):
        y = []
        for stage in oracle_stages:
            row = otab[(otab["stage"].astype(str) == stage) & (otab["regime"].astype(str) == regime)]
            y.append(1000.0 * float(row["median"].iloc[0]) if not row.empty else np.nan)
        axes[3].plot(range(len(oracle_stages)), y, marker="o", linewidth=1.4, color=color, label=regime.replace("shift_", "shift "))
    axes[3].set_title("D. Oracle stabilization", loc="left")
    axes[3].set_xticks(range(len(oracle_stages)))
    axes[3].set_xticklabels(oracle_labels)
    axes[3].set_ylabel("advantage RMSE ($\\times 10^3$)")
    axes[3].grid(alpha=0.22)
    axes[3].legend(frameon=False, fontsize=7)

    fig.tight_layout(h_pad=1.4, w_pad=1.3)
    _save(fig, "softfqi_minimax_oracle_sensitivity")


def weight_estimator_sensitivity() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.75))
    gamma = _final(_read("weight_ablation_gamma_cv"))
    gamma["ridge_mode"] = np.where(gamma["stage"].astype(str).str.contains("cv"), "CV", "fixed")
    for ax, metric, title, scale in [
        (axes[0], "stationary_bellman_rmse", "A. Bellman residual", 1.0),
        (axes[1], "stationary_advantage_q_rmse", "B. Advantage error", 1000.0),
        (axes[2], "effective_sample_size_fraction", "C. ESS fraction", 1.0),
    ]:
        tab = _summ(gamma, metric, ["regime", "gamma_weight", "ridge_mode"])
        for regime, color in zip(["shift_45", "shift_72", "shift_80"], ["#0072B2", "#D55E00", "#009E73"]):
            for mode, linestyle, marker in [("fixed", "-", "o"), ("CV", "--", "s")]:
                sub = tab[
                    (tab["regime"].astype(str) == regime)
                    & (tab["ridge_mode"].astype(str) == mode)
                ].sort_values("gamma_weight")
                label = f"{regime.replace('shift_', 'shift ')} {mode}" if metric == "stationary_bellman_rmse" else None
                ax.plot(
                    sub["gamma_weight"],
                    scale * sub["median"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.25,
                    markersize=3,
                    label=label,
                )
        ax.set_title(title, loc="left")
        ax.set_xlabel("$\\gamma_w$")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("RMSE ($\\times 10^3$)")
    axes[2].set_ylabel("fraction")
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="best")
    fig.tight_layout(w_pad=1.25)
    _save(fig, "softfqi_weight_estimator_sensitivity")


def population_geometry_control() -> None:
    linear = _final(_read("population_geometry_linear"))
    rich = _final(_read("population_geometry_rich_q"))
    data = pd.concat([linear.assign(feature_set="linear"), rich.assign(feature_set="rich RBF")], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.65), sharex=True)
    for ax, feature_set in zip(axes, ["linear", "rich RBF"]):
        sub = data[data["feature_set"].astype(str) == feature_set]
        tab = _summ(sub, "stationary_bellman_rmse", ["regime", "method"])
        regimes = [r for r in REGIMES if r in set(sub["regime"].astype(str))]
        x = np.arange(len(regimes))
        width = 0.34
        for offset, method in [(-width / 2, "unweighted"), (width / 2, "oracle")]:
            vals = []
            for regime in regimes:
                row = tab[(tab["regime"].astype(str) == regime) & (tab["method"].astype(str) == method)]
                vals.append(float(row["median"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + offset, vals, width=width, color=COLORS[method], label=LABELS[method])
        ax.set_title(f"{feature_set} population", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("shift_", "") for r in regimes], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Bellman residual")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.4)
    _save(fig, "softfqi_population_geometry_control")


def main() -> None:
    minimax_vs_weighting()
    supporting_diagnostics()
    minimax_oracle_sensitivity()
    weight_estimator_sensitivity()
    population_geometry_control()


if __name__ == "__main__":
    main()
