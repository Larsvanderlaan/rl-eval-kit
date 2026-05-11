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


def _method_order(method: str) -> tuple[int, str]:
    if method == "unweighted":
        return (0, method)
    if method == "oracle":
        return (1, method)
    if method.startswith("estimated"):
        return (2, method)
    if method == "minimax":
        return (3, method)
    return (4, method)


def _method_label(method: str) -> str:
    return {
        "unweighted": "Unweighted",
        "oracle": "Oracle stationary",
        "estimated_g0p95": "Estimated $\\gamma_w=0.95$",
        "estimated_g1p0": "Estimated $\\gamma_w=1$",
        "minimax": "Minimax soft Q",
    }.get(method, method)


def _main_subset(raw: pd.DataFrame) -> pd.DataFrame:
    main = raw[
        (raw["stage"].astype(str) == "stage2_weights")
        & (raw["learner"].astype(str) == "linear")
        & (raw["method"].astype(str).isin(["unweighted", "oracle", "estimated_g0p95", "estimated_g1p0", "minimax"]))
        & (raw["failed"] == 0)
    ].copy()
    if main.empty:
        main = raw[
            (raw["learner"].astype(str).isin(["linear", "population"]))
            & (raw["method"].astype(str).isin(["unweighted", "oracle", "estimated_g0p95", "estimated_g1p0", "minimax"]))
            & (raw["failed"] == 0)
        ].copy()
    return main


def _plot_regimes(data: pd.DataFrame) -> list[str]:
    available = [str(regime) for regime in data["regime"].dropna().unique()]
    preferred = [
        "on_policy",
        "mild_shift",
        "moderate_shift",
        "shift_001",
        "shift_27",
        "shift_45",
        "shift_72",
        "shift_80",
    ]
    ordered = [regime for regime in preferred if regime in available]
    ordered.extend(sorted(regime for regime in available if regime not in set(ordered)))
    return ordered


def plot_main_learning_curves(raw: pd.DataFrame, figures_dir: Path) -> None:
    curve = _main_subset(raw)
    curve = curve[curve["iteration"] >= 0].copy()
    if curve.empty:
        return
    regimes = _plot_regimes(curve)
    if not regimes:
        return
    methods = ["unweighted", "oracle", "estimated_g0p95", "minimax"]
    methods = [method for method in methods if method in set(curve["method"].astype(str))]
    colors = {"unweighted": "#4C4C4C", "oracle": "#0072B2", "estimated_g0p95": "#009E73", "minimax": "#CC79A7"}
    styles = {"direct": "-", "annealed": "--"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.7 * len(regimes), 3.5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        reg = curve[curve["regime"].astype(str) == regime]
        for method in methods:
            for schedule, sched in reg[reg["method"].astype(str) == method].groupby("schedule"):
                grouped = sched.groupby("iteration")["stationary_q_rmse"]
                x = np.asarray(sorted(grouped.groups.keys()), dtype=float)
                median = grouped.median().reindex(x).to_numpy(dtype=float)
                q25 = grouped.quantile(0.25).reindex(x).to_numpy(dtype=float)
                q75 = grouped.quantile(0.75).reindex(x).to_numpy(dtype=float)
                label = f"{_method_label(method)}, {schedule}"
                ax.plot(x, median, styles.get(str(schedule), "-"), color=colors.get(method), label=label)
                ax.fill_between(x, q25, q75, color=colors.get(method), alpha=0.12)
        ax.set_title(regime.replace("_", " "))
        ax.set_xlabel("FQI iteration")
        ax.set_yscale("log")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("stationary-norm Q RMSE")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_learning_curves_stationary_q.png", dpi=260, bbox_inches="tight")
    fig.savefig(figures_dir / "main_learning_curves_stationary_q.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_advantage_learning_curves(raw: pd.DataFrame, figures_dir: Path) -> None:
    curve = _main_subset(raw)
    curve = curve[curve["iteration"] >= 0].copy()
    metric = "stationary_advantage_q_rmse"
    if curve.empty or metric not in curve:
        return
    regimes = _plot_regimes(curve)
    if not regimes:
        return
    methods = ["unweighted", "oracle", "estimated_g0p95", "estimated_g1p0", "minimax"]
    methods = [method for method in methods if method in set(curve["method"].astype(str))]
    colors = {
        "unweighted": "#4C4C4C",
        "oracle": "#0072B2",
        "estimated_g0p95": "#009E73",
        "estimated_g1p0": "#D55E00",
        "minimax": "#CC79A7",
    }
    styles = {"direct": "-", "annealed": "--"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.7 * len(regimes), 3.5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        reg = curve[curve["regime"].astype(str) == regime]
        for method in methods:
            for schedule, sched in reg[reg["method"].astype(str) == method].groupby("schedule"):
                grouped = sched.groupby("iteration")[metric]
                x = np.asarray(sorted(grouped.groups.keys()), dtype=float)
                median = grouped.median().reindex(x).to_numpy(dtype=float)
                q25 = grouped.quantile(0.25).reindex(x).to_numpy(dtype=float)
                q75 = grouped.quantile(0.75).reindex(x).to_numpy(dtype=float)
                ax.plot(x, median, styles.get(str(schedule), "-"), color=colors.get(method), label=f"{_method_label(method)}, {schedule}")
                ax.fill_between(x, q25, q75, color=colors.get(method), alpha=0.12)
        ax.set_title(regime.replace("_", " "))
        ax.set_xlabel("FQI iteration")
        ax.set_yscale("log")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("stationary advantage-Q RMSE")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_learning_curves_advantage_q.png", dpi=260, bbox_inches="tight")
    fig.savefig(figures_dir / "main_learning_curves_advantage_q.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_norm_mismatch_diagnostic(raw: pd.DataFrame, figures_dir: Path) -> None:
    final = _main_subset(raw)
    final = final[(final["is_final"] == 1) & (final["failed"] == 0)].copy()
    needed = {"behavior_q_rmse", "stationary_q_rmse"}
    if final.empty or not needed.issubset(final.columns):
        return
    regimes = _plot_regimes(final)
    if not regimes:
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.4 * len(regimes), 3.6), sharex=False, sharey=False)
    if len(regimes) == 1:
        axes = [axes]
    markers = {"unweighted": "o", "oracle": "s", "estimated_g0p95": "^", "estimated_g1p0": "D", "minimax": "P"}
    colors = {
        "unweighted": "#4C4C4C",
        "oracle": "#0072B2",
        "estimated_g0p95": "#009E73",
        "estimated_g1p0": "#D55E00",
        "minimax": "#CC79A7",
    }
    for ax, regime in zip(axes, regimes):
        reg = final[final["regime"].astype(str) == regime]
        summary = reg.groupby(["method", "schedule"], dropna=False)[["behavior_q_rmse", "stationary_q_rmse"]].median()
        mins = []
        maxs = []
        for (method, schedule), row in summary.iterrows():
            ax.scatter(
                row["behavior_q_rmse"],
                row["stationary_q_rmse"],
                marker=markers.get(str(method), "o"),
                color=colors.get(str(method), "#333333"),
                s=58,
                label=f"{_method_label(str(method))}, {schedule}",
            )
            mins.extend([row["behavior_q_rmse"], row["stationary_q_rmse"]])
            maxs.extend([row["behavior_q_rmse"], row["stationary_q_rmse"]])
        if mins and maxs:
            lo = max(min(mins) * 0.85, 1e-8)
            hi = max(maxs) * 1.15
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, alpha=0.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(regime.replace("_", " "))
        ax.set_xlabel("behavior-norm Q RMSE")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("stationary-norm Q RMSE")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_norm_mismatch_diagnostic.png", dpi=260, bbox_inches="tight")
    fig.savefig(figures_dir / "main_norm_mismatch_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_projected_bellman_diagnostic(raw: pd.DataFrame, figures_dir: Path) -> None:
    final = _main_subset(raw)
    final = final[(final["is_final"] == 1) & (final["failed"] == 0)].copy()
    metric = "stationary_projected_bellman_rmse"
    if final.empty or metric not in final:
        return
    methods = ["unweighted", "oracle", "estimated_g0p95", "minimax"]
    regimes = _plot_regimes(final)
    if not regimes:
        return
    data = (
        final[final["method"].astype(str).isin(methods)]
        .groupby(["regime", "method", "schedule"], dropna=False)[metric]
        .median()
        .reset_index()
    )
    if data.empty:
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.3 * len(regimes), 3.5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    colors = {"direct": "#999999", "annealed": "#56B4E9"}
    for ax, regime in zip(axes, regimes):
        reg = data[data["regime"].astype(str) == regime]
        labels = [method for method in methods if method in set(reg["method"].astype(str))]
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        for offset, schedule in [(-width / 2, "direct"), (width / 2, "annealed")]:
            vals = []
            for method in labels:
                hit = reg[(reg["method"].astype(str) == method) & (reg["schedule"].astype(str) == schedule)]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + offset, vals, width=width, color=colors.get(schedule), label=schedule)
        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(method) for method in labels], rotation=25, ha="right")
        ax.set_title(regime.replace("_", " "))
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("stationary projected Bellman error")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "main_projected_bellman_diagnostic.png", dpi=260, bbox_inches="tight")
    fig.savefig(figures_dir / "main_projected_bellman_diagnostic.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(raw: pd.DataFrame, figures_dir: Path) -> None:
    curve = raw[(raw["failed"] == 0) & (raw["iteration"] >= 0)].copy()
    if curve.empty:
        return
    metric = "stationary_q_rmse"
    for (stage, learner, schedule), sub in curve.groupby(["stage", "learner", "schedule"], dropna=False):
        regimes = list(sub["regime"].dropna().unique())
        fig, axes = plt.subplots(1, len(regimes), figsize=(5.0 * len(regimes), 3.6), sharey=True)
        if len(regimes) == 1:
            axes = [axes]
        for ax, regime in zip(axes, regimes):
            reg = sub[sub["regime"] == regime]
            for method in sorted(reg["method"].unique(), key=_method_order):
                meth = reg[reg["method"] == method]
                grouped = meth.groupby("iteration")[metric]
                x = np.asarray(sorted(grouped.groups.keys()), dtype=float)
                mean = grouped.mean().reindex(x).to_numpy(dtype=float)
                q25 = grouped.quantile(0.25).reindex(x).to_numpy(dtype=float)
                q75 = grouped.quantile(0.75).reindex(x).to_numpy(dtype=float)
                ax.plot(x, mean, label=method)
                ax.fill_between(x, q25, q75, alpha=0.18)
            ax.set_title(str(regime))
            ax.set_xlabel("FQI iteration")
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("stationary-norm RMSE")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{stage} / {learner} / {schedule}")
        fig.tight_layout()
        name = f"learning_curves_{stage}_{learner}_{schedule}".replace("/", "_")
        fig.savefig(figures_dir / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_final_distributions(raw: pd.DataFrame, figures_dir: Path) -> None:
    final = raw[(raw["is_final"] == 1) & (raw["failed"] == 0)].copy()
    if final.empty:
        return
    for (stage, learner, schedule), sub in final.groupby(["stage", "learner", "schedule"], dropna=False):
        labels = []
        data = []
        for (regime, method), grp in sub.groupby(["regime", "method"], dropna=False):
            labels.append(f"{regime}\n{method}")
            data.append(grp["stationary_q_rmse"].to_numpy(dtype=float))
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(data)), 4.0))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_yscale("log")
        ax.set_ylabel("final stationary-norm RMSE")
        ax.tick_params(axis="x", labelrotation=55)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{stage} / {learner} / {schedule}")
        fig.tight_layout()
        name = f"final_box_{stage}_{learner}_{schedule}".replace("/", "_")
        fig.savefig(figures_dir / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_weight_diagnostics(raw: pd.DataFrame, figures_dir: Path) -> None:
    final = raw[(raw["is_final"] == 1) & (raw["failed"] == 0) & raw["method"].astype(str).str.startswith("estimated")].copy()
    if final.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, metric, ylabel in [
        (axes[0], "oracle_log_ratio_rmse", "log-ratio RMSE"),
        (axes[1], "oracle_estimated_weight_corr", "correlation"),
        (axes[2], "effective_sample_size_fraction", "ESS fraction"),
    ]:
        if metric not in final:
            continue
        grouped = final.groupby(["regime", "gamma_weight"], dropna=False)[metric].median().reset_index()
        for regime, sub in grouped.groupby("regime"):
            ax.plot(sub["gamma_weight"], sub[metric], marker="o", label=regime)
        ax.set_xlabel("gamma_weight")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "weight_diagnostics.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "weight_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ess_sensitivity(raw: pd.DataFrame, figures_dir: Path) -> None:
    required = {"is_final", "failed", "method", "stage", "regime", "schedule", "seed", "n_samples"}
    if not required.issubset(raw.columns):
        return
    final = raw[(raw["is_final"] == 1) & (raw["failed"] == 0)].copy()
    estimated = final[final["method"].astype(str).str.startswith("estimated_g")].copy()
    oracle = final[final["method"].astype(str) == "oracle"].copy()
    if estimated.empty or oracle.empty:
        return
    metrics = [
        ("stationary_projected_bellman_rmse", "Projected Bellman gap"),
        ("stationary_advantage_q_rmse", "Advantage gap"),
        ("policy_value_error", "Value-error gap"),
    ]
    metrics = [(metric, label) for metric, label in metrics if metric in final.columns]
    if not metrics or "effective_sample_size_fraction" not in estimated.columns:
        return
    keys = ["stage", "regime", "schedule", "seed"]
    if "q_class" in final.columns:
        keys.append("q_class")
    oracle_cols = keys + [metric for metric, _label in metrics]
    oracle_ref = oracle[oracle_cols].rename(columns={metric: f"oracle_{metric}" for metric, _label in metrics})
    paired = estimated.merge(oracle_ref, on=keys, how="inner")
    if paired.empty:
        return
    n_values = sorted(int(value) for value in paired["n_samples"].dropna().unique())
    if len(n_values) <= 1 and paired["effective_sample_size_fraction"].nunique(dropna=True) <= 1:
        return
    color_values = {n: color for n, color in zip(n_values, plt.cm.viridis(np.linspace(0.15, 0.85, max(len(n_values), 1))), strict=False)}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, metrics):
        paired[f"{metric}_gap_to_oracle"] = paired[metric] - paired[f"oracle_{metric}"]
        for n_samples, sub in paired.groupby("n_samples"):
            ax.scatter(
                sub["effective_sample_size_fraction"],
                sub[f"{metric}_gap_to_oracle"],
                s=18,
                alpha=0.35,
                color=color_values.get(int(n_samples), "#555555"),
                label=f"n={int(n_samples)}",
            )
        med = (
            paired.groupby("n_samples", dropna=False)[["effective_sample_size_fraction", f"{metric}_gap_to_oracle"]]
            .median()
            .sort_index()
        )
        ax.plot(
            med["effective_sample_size_fraction"],
            med[f"{metric}_gap_to_oracle"],
            color="#222222",
            marker="o",
            linewidth=1.1,
            label="median",
        )
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Estimated-weight ESS fraction")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(len(labels), 5), fontsize=8)
    fig.savefig(figures_dir / "appendix_ess_sensitivity.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures_dir / "appendix_ess_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_heatmaps(results_dir: Path, figures_dir: Path) -> None:
    ref_path = results_dir / "reference_context.npz"
    if not ref_path.exists():
        return
    ref = np.load(ref_path)
    states = ref["states"]
    side = int(round(np.sqrt(states.shape[0])))
    names = ["target"] + sorted(name[:-11] for name in ref.files if name.endswith("_state_dist"))
    names = [name for name in names if name != "target"] if "target_state_dist" not in ref.files else names
    panels: list[tuple[str, np.ndarray]] = [("target", ref["target_state_dist"])]
    for key in sorted(ref.files):
        if key.endswith("_state_dist"):
            panels.append((key.replace("_state_dist", ""), ref[key]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, dist) in zip(axes, panels):
        image = dist.reshape(side, side).T
        im = ax.imshow(image, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figures_dir / "occupancy_heatmaps.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "occupancy_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_occupancy_ratio_heatmaps(results_dir: Path, figures_dir: Path) -> None:
    ref_path = results_dir / "reference_context.npz"
    if not ref_path.exists():
        return
    ref = np.load(ref_path)
    states = ref["states"]
    side = int(round(np.sqrt(states.shape[0])))
    target = ref["target_state_dist"]
    regimes = [key.replace("_state_dist", "") for key in sorted(ref.files) if key.endswith("_state_dist")]
    regimes = [regime for regime in regimes if regime != "target"]
    regimes = [regime for regime in ["on_policy", "mild_shift", "moderate_shift"] if regime in regimes]
    if not regimes:
        return
    fig, axes = plt.subplots(2, len(regimes), figsize=(3.3 * len(regimes), 6.0), constrained_layout=True)
    if len(regimes) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, regime in enumerate(regimes):
        behavior = ref[f"{regime}_state_dist"]
        ratio = target / np.maximum(behavior, 1e-12)
        ratio = ratio / np.maximum(np.sum(behavior * ratio), 1e-300)
        image = behavior.reshape(side, side).T
        im0 = axes[0, col].imshow(image, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
        axes[0, col].set_title(regime.replace("_", " "))
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
        log_ratio = np.log10(np.clip(ratio.reshape(side, side).T, 1e-3, 1e3))
        im1 = axes[1, col].imshow(log_ratio, origin="lower", extent=[-1, 1, -1, 1], cmap="coolwarm", vmin=-2, vmax=2)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
    axes[0, 0].set_ylabel("behavior occupancy")
    axes[1, 0].set_ylabel("log10 target / behavior")
    fig.savefig(figures_dir / "main_occupancy_ratio_heatmaps.png", dpi=260, bbox_inches="tight")
    fig.savefig(figures_dir / "main_occupancy_ratio_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-style plots for the soft-FQI study.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--profile", choices=["all", "main_text", "appendix"], default="all")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else ROOT / "figures" / results_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(results_dir / "raw_results.csv")
    if args.profile in {"all", "main_text"}:
        plot_main_learning_curves(raw, figures_dir)
        plot_main_advantage_learning_curves(raw, figures_dir)
        plot_norm_mismatch_diagnostic(raw, figures_dir)
        plot_projected_bellman_diagnostic(raw, figures_dir)
        plot_occupancy_ratio_heatmaps(results_dir, figures_dir)
    if args.profile in {"all", "appendix"}:
        plot_learning_curves(raw, figures_dir)
        plot_final_distributions(raw, figures_dir)
        plot_weight_diagnostics(raw, figures_dir)
        plot_ess_sensitivity(raw, figures_dir)
        plot_occupancy_heatmaps(results_dir, figures_dir)
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
