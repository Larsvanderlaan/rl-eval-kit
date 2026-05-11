#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REGIME_ORDER = [
    "shift_001",
    "shift_09",
    "shift_18",
    "shift_27",
    "shift_36",
    "shift_45",
    "shift_54",
    "shift_63",
    "shift_72",
    "shift_80",
]
FIVE_REGIME_ORDER = ["no_shift", "small_shift", "mild_shift", "moderate_shift", "severe_shift"]
LEGACY_REGIME_ORDER = ["on_policy", "mild_shift", "moderate_shift"]
REGIME_LABELS = {
    "shift_001": "0.00",
    "shift_09": "0.09",
    "shift_18": "0.18",
    "shift_27": "0.27",
    "shift_36": "0.36",
    "shift_45": "0.45",
    "shift_54": "0.54",
    "shift_63": "0.63",
    "shift_72": "0.72",
    "shift_80": "0.80",
    "no_shift": "No shift",
    "small_shift": "Small shift",
    "mild_shift": "Mild shift",
    "moderate_shift": "Moderate shift",
    "severe_shift": "Severe shift",
    "on_policy": "On-policy",
}
SEVERE_REGIME_CANDIDATES = ["shift_80", "severe_shift", "moderate_shift"]
METHOD_ORDER = ["unweighted", "oracle", "estimated_g0p95"]
METHOD_LABELS = {
    "unweighted": "Unweighted",
    "oracle": "Stab. oracle",
    "estimated_g0p95": "Estimated",
}
COLORS = {
    "unweighted": "#595959",
    "oracle": "#0072B2",
    "estimated_g0p95": "#009E73",
}
OVERLAP_COLORS = {
    "tv": "#A6611A",
    "ess": "#E6AB02",
}


def _regime_order(available: set[str]) -> list[str]:
    if "shift_001" in available:
        order = REGIME_ORDER
    elif "no_shift" in available:
        order = FIVE_REGIME_ORDER
    else:
        order = LEGACY_REGIME_ORDER
    return [regime for regime in order if regime in available]


def _main_data(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw[
        (raw["stage"].astype(str) == "stage2_weights")
        & (raw["learner"].astype(str) == "linear")
        & (raw["schedule"].astype(str) == "direct")
        & (raw["method"].astype(str).isin(METHOD_ORDER))
        & (raw["is_final"] == 1)
        & (raw["failed"] == 0)
    ].copy()
    if data.empty:
        raise ValueError("No direct stage2 linear final rows found.")
    return data


def _median_iqr(values: pd.Series, *, scale: float = 1.0) -> tuple[float, float, float]:
    arr = values.dropna().to_numpy(dtype=float) * float(scale)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def _shift_scores(results_dir: Path, regimes: list[str]) -> dict[str, float]:
    config_path = results_dir / "config.yaml"
    if not config_path.exists():
        return {regime: float(idx) for idx, regime in enumerate(regimes)}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mixtures = config.get("regimes", {})
    scores = {}
    for idx, regime in enumerate(regimes):
        mixture = mixtures.get(regime, {})
        if "target" in mixture:
            scores[regime] = 1.0 - float(mixture["target"])
        else:
            scores[regime] = float(idx)
    return scores


def _shift_diagnostics(results_dir: Path, regimes: list[str]) -> pd.DataFrame:
    ref = np.load(results_dir / "reference_context.npz")
    target_sa = ref["target_sa_dist"]
    shift_scores = _shift_scores(results_dir, regimes)
    rows = []
    for regime in regimes:
        behavior_sa = ref[f"{regime}_sa_dist"]
        ratio = target_sa / np.maximum(behavior_sa, 1e-12)
        ratio = ratio / np.maximum(np.sum(behavior_sa * ratio), 1e-300)
        rows.append(
            {
                "regime": regime,
                "shift_score": shift_scores[regime],
                "tv": float(0.5 * np.sum(np.abs(target_sa - behavior_sa))),
                "ess": float(1.0 / np.maximum(np.sum(behavior_sa * ratio * ratio), 1e-300)),
                "q99": float(np.quantile(ratio, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def _plot_grouped_iqr(
    ax: plt.Axes,
    data: pd.DataFrame,
    regimes: list[str],
    shift: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    scale: float = 1.0,
    ylim_pad: float = 0.12,
) -> None:
    x = np.array([float(shift.loc[shift["regime"] == r, "shift_score"].iloc[0]) for r in regimes])
    offsets = np.linspace(-0.012, 0.012, len(METHOD_ORDER)) if len(x) > 1 else np.zeros(len(METHOD_ORDER))
    max_y = 0.0
    for idx, method in enumerate(METHOD_ORDER):
        medians = []
        err_low = []
        err_high = []
        for regime in regimes:
            sub = data[(data["regime"] == regime) & (data["method"] == method)]
            median, q25, q75 = _median_iqr(sub[metric], scale=scale)
            medians.append(median)
            err_low.append(max(median - q25, 0.0))
            err_high.append(max(q75 - median, 0.0))
            max_y = max(max_y, q75)
        ax.errorbar(
            x + offsets[idx],
            medians,
            yerr=np.vstack([err_low, err_high]),
            fmt="o-",
            color=COLORS[method],
            label=METHOD_LABELS[method],
            ecolor="#222222",
            markersize=3.6,
            linewidth=1.2,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2f}" for value in x])
    ax.set_xlabel("Shift mass (1 - target mix.)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max_y * (1.0 + ylim_pad))
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)


def _plot_shift_panel(ax: plt.Axes, shift: pd.DataFrame, regimes: list[str]) -> None:
    x = np.array([float(shift.loc[shift["regime"] == r, "shift_score"].iloc[0]) for r in regimes])
    tv = [float(shift.loc[shift["regime"] == r, "tv"].iloc[0]) for r in regimes]
    ess = [float(shift.loc[shift["regime"] == r, "ess"].iloc[0]) for r in regimes]
    q99 = [float(shift.loc[shift["regime"] == r, "q99"].iloc[0]) for r in regimes]
    ax.plot(
        x,
        tv,
        marker="o",
        color=OVERLAP_COLORS["tv"],
        label="TV distance",
        linewidth=1.2,
        markersize=3.6,
    )
    ax.plot(
        x,
        ess,
        marker="s",
        color=OVERLAP_COLORS["ess"],
        label="ESS fraction",
        linewidth=1.2,
        markersize=3.6,
    )
    label_indices = range(len(q99)) if len(q99) <= 5 else [0, len(q99) // 2, len(q99) - 1]
    for idx in label_indices:
        x_value = x[idx]
        value = q99[idx]
        label = f"q99={value:.1f}" if value < 10 else f"q99={value:.0f}"
        ax.text(x_value, 1.04, label, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2f}" for value in x])
    ax.set_xlabel("Shift mass (1 - target mix.)")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Overlap diagnostic")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.05),
        ncol=2,
        fontsize=6.3,
        handlelength=1.1,
        columnspacing=0.9,
    )


def _plot_win_rates(ax: plt.Axes, data: pd.DataFrame) -> None:
    severe_regime = next(regime for regime in SEVERE_REGIME_CANDIDATES if regime in set(data["regime"]))
    severe = data[data["regime"] == severe_regime].copy()
    metrics = [
        ("policy_value_error", "Policy value"),
        ("stationary_advantage_q_rmse", "Advantage Q"),
        ("stationary_projected_bellman_rmse", "Projected Bellman"),
    ]
    methods = ["oracle", "estimated_g0p95"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.34
    for idx, method in enumerate(methods):
        wins = []
        for metric, _label in metrics:
            pivot = severe.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            paired = pivot[["unweighted", method]].dropna()
            wins.append(100.0 * float(np.mean(paired[method] < paired["unweighted"])))
        ax.bar(
            x + (idx - 0.5) * width,
            wins,
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            edgecolor="white",
            linewidth=0.7,
        )
    ax.axhline(50, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label in metrics], rotation=18, ha="right")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)


def make_figure(results_dir: Path, figures_dir: Path) -> tuple[Path, Path]:
    raw = pd.read_csv(results_dir / "raw_results.csv")
    data = _main_data(raw)
    regimes = _regime_order(set(data["regime"].astype(str)))
    shift = _shift_diagnostics(results_dir, regimes)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.45), constrained_layout=True)
    _plot_shift_panel(axes[0], shift, regimes)
    _plot_grouped_iqr(
        axes[1],
        data,
        regimes,
        shift,
        "stationary_projected_bellman_rmse",
        "Projected Bellman",
    )
    _plot_grouped_iqr(
        axes[2],
        data,
        regimes,
        shift,
        "stationary_advantage_q_rmse",
        "Advantage error (x1000)",
        scale=1000.0,
    )
    _plot_win_rates(axes[3], data)

    titles = [
        "A. Regime shift",
        "B. Stationary PBE",
        "C. Stationary advantage",
        "D. Severe-shift wins",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontweight="bold")

    for ax in axes[:3]:
        for label in ax.get_xticklabels():
            label.set_rotation(18)
            label.set_ha("right")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["unweighted"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["oracle"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["estimated_g0p95"]),
    ]
    legend_labels = ["Unweighted", "Stab. oracle", "Estimated"]
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=1.2,
        handlelength=1.1,
        handletextpad=0.4,
        fontsize=6.5,
    )

    png_path = figures_dir / "main_text_simulation_figure.png"
    pdf_path = figures_dir / "main_text_simulation_figure.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def make_appendix_policy_value_figure(results_dir: Path, figures_dir: Path) -> tuple[Path, Path]:
    raw = pd.read_csv(results_dir / "raw_results.csv")
    data = _main_data(raw)
    regimes = _regime_order(set(data["regime"].astype(str)))
    baseline = data[data["method"] == "unweighted"][
        ["regime", "seed", "policy_value_error"]
    ].rename(columns={"policy_value_error": "unweighted_policy_value_error"})
    data = data.merge(baseline, on=["regime", "seed"], how="left")
    data["policy_value_gain"] = (
        data["unweighted_policy_value_error"] - data["policy_value_error"]
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 2.45), constrained_layout=True)
    _plot_grouped_iqr(
        ax,
        data,
        regimes,
        _shift_diagnostics(results_dir, regimes),
        "policy_value_gain",
        "Policy value gain",
    )
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_title("Policy value gain over unweighted", loc="left", fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_rotation(18)
        label.set_ha("right")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=6.5)

    png_path = figures_dir / "appendix_policy_value_figure.png"
    pdf_path = figures_dir / "appendix_policy_value_figure.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _format_iqr(values: pd.Series, *, scale: float = 1.0, digits: int = 3) -> str:
    median, q25, q75 = _median_iqr(values, scale=scale)
    return f"{median:.{digits}g} [{q25:.{digits}g}, {q75:.{digits}g}]"


def make_table(results_dir: Path, snippet_dir: Path) -> tuple[Path, Path]:
    raw = pd.read_csv(results_dir / "raw_results.csv")
    data = _main_data(raw)
    regimes = _regime_order(set(data["regime"].astype(str)))
    n_datasets = int(data["seed"].nunique())
    rows = []
    for regime in regimes:
        for method in METHOD_ORDER:
            sub = data[(data["regime"] == regime) & (data["method"] == method)]
            rows.append(
                {
                    "Regime": REGIME_LABELS[regime],
                    "Method": METHOD_LABELS[method],
                    "Projected Bellman": _format_iqr(sub["stationary_projected_bellman_rmse"]),
                    "Stationary advantage error $\\times 10^3$": _format_iqr(
                        sub["stationary_advantage_q_rmse"], scale=1000.0
                    ),
                    "ESS fraction": _format_iqr(sub["effective_sample_size_fraction"])
                    if "effective_sample_size_fraction" in sub
                    else "",
                }
            )
    table = pd.DataFrame(rows)
    snippet_dir.mkdir(parents=True, exist_ok=True)
    csv_path = snippet_dir / "main_text_simulation_table.csv"
    tex_path = snippet_dir / "main_text_simulation_table.tex"
    table.to_csv(csv_path, index=False)
    latex = table.to_latex(index=False, escape=False)
    latex = latex.replace("\\begin{tabular}", "\\begin{tabular}")
    wrapped = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{Representative direct low-temperature soft-FQI diagnostics. Entries are median [IQR] across {n_datasets} offline datasets.}}\n"
        "\\label{tab:soft-fqi-main}\n"
        "\\small\n"
        f"{latex}"
        "\\end{table}\n"
    )
    tex_path.write_text(wrapped, encoding="utf-8")
    return csv_path, tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the NeurIPS main-text simulation figure and table.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "prelim_v3"))
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--snippet-dir", default=str(ROOT / "paper_snippets"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else ROOT / "figures" / results_dir.name
    snippet_dir = Path(args.snippet_dir)
    png_path, pdf_path = make_figure(results_dir, figures_dir)
    policy_png_path, policy_pdf_path = make_appendix_policy_value_figure(results_dir, figures_dir)
    csv_path, tex_path = make_table(results_dir, snippet_dir)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {policy_png_path}")
    print(f"Wrote {policy_pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
