from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("FQE_calibration_neurips/.mplconfig").resolve())
os.environ["XDG_CACHE_HOME"] = str(Path("FQE_calibration_neurips/.cache").resolve())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .validation import load_gate


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=180)
    fig.savefig(figures_dir / f"{stem}.pdf")
    plt.close(fig)


def _mode_label(df: pd.DataFrame, results_dir: Path) -> str:
    if "run_mode" in df and not df["run_mode"].dropna().empty:
        return str(df["run_mode"].dropna().iloc[0])
    return "debug" if "debug" in str(results_dir) else "paper"


def _title(ax: plt.Axes, title: str, mode: str, n_replications: int) -> None:
    suffix = ""
    if mode == "debug" or n_replications < 2:
        suffix = " (debug only; no paper claim)"
    ax.set_title(title + suffix)


def _eligible_summary(results_dir: Path) -> pd.DataFrame:
    eligible = results_dir / "eligible_summary.csv"
    summary = results_dir / "summary.csv"
    if eligible.exists():
        return pd.read_csv(eligible)
    if summary.exists():
        return pd.read_csv(summary)
    from .aggregation import aggregate_results

    aggregate_results(results_dir)
    return pd.read_csv(eligible if eligible.exists() else summary)


def _errorbar_plot(
    df: pd.DataFrame,
    figures_dir: Path,
    *,
    x_col: str,
    y_col: str,
    stem: str,
    title: str,
    yerr_col: str | None = None,
    group_col: str = "baseline_learner",
    mode: str,
) -> list[Path]:
    if x_col not in df or y_col not in df or df.empty:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n_rep = int(pd.to_numeric(df.get("n_replications", pd.Series([1])), errors="coerce").max())
    for label, group in df.groupby(group_col, dropna=False):
        plot_group = group.sort_values(x_col)
        x = plot_group[x_col].astype(str)
        y = pd.to_numeric(plot_group[y_col], errors="coerce")
        yerr = None
        if yerr_col and yerr_col in plot_group and n_rep >= 2:
            yerr = pd.to_numeric(plot_group[yerr_col], errors="coerce")
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.4, capsize=3 if yerr is not None else 0, label=str(label))
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7)
    _title(ax, title, mode, n_rep)
    _save(fig, figures_dir, stem)
    return [figures_dir / f"{stem}.png", figures_dir / f"{stem}.pdf"]


def _scatter_plot(df: pd.DataFrame, figures_dir: Path, mode: str) -> list[Path]:
    if df.empty or not {"calibration_error", "value_bias"}.issubset(df.columns):
        return []
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    n_rep = int(pd.to_numeric(df.get("n_replications", pd.Series([1])), errors="coerce").max())
    for learner, group in df.groupby("baseline_learner", dropna=False):
        ax.scatter(group["calibration_error"], group["value_bias"].abs(), label=str(learner), alpha=0.75)
    ax.set_xlabel("calibration error")
    ax.set_ylabel("absolute value bias")
    ax.legend(fontsize=7)
    _title(ax, "Calibration error versus value error", mode, n_rep)
    _save(fig, figures_dir, "calibration_error_vs_value_error")
    return [figures_dir / "calibration_error_vs_value_error.png", figures_dir / "calibration_error_vs_value_error.pdf"]


def _evidence_scatter_plot(df: pd.DataFrame, figures_dir: Path, mode: str) -> list[Path]:
    required = {
        "relative_mse_vs_uncalibrated_all_data",
        "relative_calibration_error_plugin_vs_uncalibrated_all_data",
        "calibration_evidence_status",
    }
    if df.empty or not required.issubset(df.columns):
        return []
    subset = df[df["calibration_protocol"].astype(str) != "uncalibrated_all_data"].copy()
    if subset.empty:
        return []
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    n_rep = int(pd.to_numeric(subset.get("n_replications", pd.Series([1])), errors="coerce").max())
    for status, group in subset.groupby("calibration_evidence_status", dropna=False):
        ax.scatter(
            pd.to_numeric(group["relative_mse_vs_uncalibrated_all_data"], errors="coerce"),
            pd.to_numeric(group["relative_calibration_error_plugin_vs_uncalibrated_all_data"], errors="coerce"),
            label=str(status),
            alpha=0.75,
        )
    ax.axvline(1.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("relative value MSE")
    ax.set_ylabel("relative plug-in Bellman calibration error")
    ax.legend(fontsize=7)
    _title(ax, "MSE and calibration-error improvement", mode, n_rep)
    _save(fig, figures_dir, "mse_vs_calibration_error_improvement")
    return [
        figures_dir / "mse_vs_calibration_error_improvement.png",
        figures_dir / "mse_vs_calibration_error_improvement.pdf",
    ]


def _bias_variance_plot(df: pd.DataFrame, figures_dir: Path, mode: str) -> list[Path]:
    if df.empty or not {"value_bias", "value_variance"}.issubset(df.columns):
        return []
    subset = df.copy()
    subset["bias_squared"] = pd.to_numeric(subset["value_bias"], errors="coerce") ** 2
    subset = subset.groupby("baseline_learner", dropna=False)[["bias_squared", "value_variance"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(subset.shape[0])
    ax.bar(x, subset["bias_squared"], label="bias^2")
    ax.bar(x, subset["value_variance"].fillna(0.0), bottom=subset["bias_squared"], label="variance")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["baseline_learner"], rotation=30, ha="right")
    ax.set_ylabel("MSE components")
    ax.legend(fontsize=8)
    n_rep = int(pd.to_numeric(df.get("n_replications", pd.Series([1])), errors="coerce").max())
    _title(ax, "Bias-variance decomposition", mode, n_rep)
    _save(fig, figures_dir, "bias_variance_decomposition")
    return [figures_dir / "bias_variance_decomposition.png", figures_dir / "bias_variance_decomposition.pdf"]


def _coverage_stratified_plot(results_dir: Path, figures_dir: Path, mode: str) -> list[Path]:
    path = results_dir / "coverage_stratified_error.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"coverage_stratum", "mean_coverage_stratum_error", "baseline_learner", "calibration_protocol"}
    if df.empty or not required.issubset(df.columns):
        return []
    if "suite_name" in df and (df["suite_name"] == "coverage_sweep").any():
        df = df[df["suite_name"] == "coverage_sweep"].copy()
    if "eligible_fraction" in df:
        df = df[pd.to_numeric(df["eligible_fraction"], errors="coerce").fillna(0.0) > 0.0]
    if df.empty:
        return []

    df["method_label"] = df["baseline_learner"].astype(str) + " / " + df["calibration_protocol"].astype(str)
    grouped = (
        df.groupby(["method_label", "coverage_stratum"], dropna=False)
        .agg(
            n_replications=("n_replications", "max") if "n_replications" in df else ("mean_coverage_stratum_error", "size"),
            mean_coverage_stratum_error=("mean_coverage_stratum_error", "mean"),
            coverage_stratum_error_mc_se=("coverage_stratum_error_mc_se", "mean")
            if "coverage_stratum_error_mc_se" in df
            else ("mean_coverage_stratum_error", lambda _x: np.nan),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n_rep = int(pd.to_numeric(grouped.get("n_replications", pd.Series([1])), errors="coerce").max())
    for label, group in grouped.groupby("method_label", dropna=False):
        plot_group = group.sort_values("coverage_stratum")
        yerr = None
        if n_rep >= 2:
            yerr = pd.to_numeric(plot_group["coverage_stratum_error_mc_se"], errors="coerce")
        ax.errorbar(
            pd.to_numeric(plot_group["coverage_stratum"], errors="coerce"),
            pd.to_numeric(plot_group["mean_coverage_stratum_error"], errors="coerce"),
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=3 if yerr is not None else 0,
            label=str(label),
        )
    ax.set_xlabel("density-ratio quantile stratum")
    ax.set_ylabel("held-out Bellman error")
    ax.legend(fontsize=7)
    _title(ax, "Coverage-stratified error", mode, n_rep)
    _save(fig, figures_dir, "coverage_stratified_error")
    return [figures_dir / "coverage_stratified_error.png", figures_dir / "coverage_stratified_error.pdf"]


def _split_stability_plot(results_dir: Path, figures_dir: Path, mode: str) -> list[Path]:
    path = results_dir / "split_stability_diagnostics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    required = {"train_fraction", "mean_relative_mse_vs_all_data", "mean_relative_mse_vs_same_fraction", "calibrator"}
    if df.empty or not required.issubset(df.columns):
        return []
    if "main_figure_role" in df:
        df = df[df["main_figure_role"].astype(str) != "diagnostic_only"].copy()
    if df.empty:
        return []
    label_base = df["baseline_learner"].astype(str) if "baseline_learner" in df else pd.Series(["learner"] * len(df))
    df["method_label"] = label_base + " / " + df["calibrator"].astype(str)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n_rep = int(pd.to_numeric(df.get("n_replications", pd.Series([1])), errors="coerce").max())
    plotted = 0
    for label, group in df.groupby("method_label", dropna=False):
        if plotted >= 8:
            break
        plot_group = group.sort_values("train_fraction")
        x = pd.to_numeric(plot_group["train_fraction"], errors="coerce")
        ax.plot(
            x,
            pd.to_numeric(plot_group["mean_relative_mse_vs_all_data"], errors="coerce"),
            marker="o",
            linewidth=1.4,
            label=f"{label} vs all-data",
        )
        ax.plot(
            x,
            pd.to_numeric(plot_group["mean_relative_mse_vs_same_fraction"], errors="coerce"),
            marker="x",
            linewidth=1.0,
            linestyle="--",
            label=f"{label} vs same-fraction",
        )
        plotted += 1
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("training fraction")
    ax.set_ylabel("seed-matched relative MSE")
    ax.legend(fontsize=6)
    _title(ax, "Split-calibration stability", mode, n_rep)
    _save(fig, figures_dir, "split_stability_diagnostics")
    return [figures_dir / "split_stability_diagnostics.png", figures_dir / "split_stability_diagnostics.pdf"]


def _method_label(row: pd.Series) -> str:
    protocol = str(row.get("calibration_protocol", ""))
    calibrator = str(row.get("calibrator", ""))
    if protocol == "current_retrain_small":
        return "retrain"
    if protocol == "uncalibrated_all_data" or calibrator == "none":
        return "raw"
    return calibrator


def _focused_rows(summary: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if summary.empty or "suite_name" not in summary:
        return []
    candidates = [
        (
            "Affine misspec.",
            {
                "suite_name": "model_misspecification_sweep",
                "misspecification_setting": "affine",
                "learner_variant": "linear_fqe_misspecified",
            },
        ),
        (
            "Finite iteration",
            {
                "suite_name": "undertraining_sweep",
                "learner_variant": "random_feature_fqe_iter2",
            },
        ),
        (
            "Temporal shift",
            {
                "suite_name": "temporal_reward_shift_sweep",
                "learner_variant": "temporal_rf_fqe",
            },
        ),
        (
            "Affine mechanism",
            {
                "suite_name": "mechanism_distortion_sweep",
                "learner_variant": "random_feature_fqe_affine_distorted",
            },
        ),
        (
            "Monotone mechanism",
            {
                "suite_name": "mechanism_distortion_sweep",
                "learner_variant": "random_feature_fqe_monotone_saturation_distorted",
            },
        ),
    ]
    out: list[tuple[str, pd.DataFrame]] = []
    for label, filters in candidates:
        mask = pd.Series(True, index=summary.index)
        for col, value in filters.items():
            if col in summary:
                mask &= summary[col].astype(str).eq(str(value))
        group = summary[mask].copy()
        group = group[
            group["calibration_protocol"].astype(str).isin(
                {"uncalibrated_all_data", "cross", "recent_heldout", "current_retrain_small"}
            )
            & group["calibrator"].astype(str).isin({"none", "linear", "isotonic"})
        ].copy()
        if not group.empty:
            out.append((label, group))
    return out


def _best_reliability_row(raw: pd.DataFrame, group: pd.DataFrame, *, calibrator: str) -> pd.Series | None:
    if raw.empty or "value_reliability_curve_json" not in raw:
        return None
    filters = {
        "suite_name": group["suite_name"].iloc[0],
        "learner_variant": group["learner_variant"].iloc[0],
    }
    if "misspecification_setting" in group and "misspecification_setting" in raw:
        filters["misspecification_setting"] = group["misspecification_setting"].iloc[0]
    mask = pd.Series(True, index=raw.index)
    for col, value in filters.items():
        mask &= raw[col].astype(str).eq(str(value))
    mask &= raw["calibrator"].astype(str).eq(str(calibrator))
    subset = raw[mask].copy()
    if subset.empty:
        return None
    return subset.iloc[0]


def _draw_reliability_curve(ax: plt.Axes, row: pd.Series | None, *, label: str, color: str) -> None:
    if row is None:
        return
    text = str(row.get("value_reliability_curve_json", ""))
    if not text or text == "nan":
        return
    try:
        curve = json.loads(text)
    except json.JSONDecodeError:
        return
    if not curve:
        return
    x = [float(point["pred_mean"]) for point in curve]
    y = [float(point["true_mean"]) for point in curve]
    ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, label=label, color=color)


def _focused_neurips_plot(results_dir: Path, summary: pd.DataFrame, figures_dir: Path, mode: str) -> list[Path]:
    groups = _focused_rows(summary)
    if not groups:
        return []
    raw_path = results_dir / "combined_raw_results.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(len(groups), 2, figsize=(6.4, 2.35 * len(groups)), squeeze=False)
    n_rep = int(pd.to_numeric(summary.get("n_replications", pd.Series([1])), errors="coerce").max())
    colors = {"raw": "#4c78a8", "linear": "#f58518", "isotonic": "#54a24b", "retrain": "#b279a2"}
    for row_id, (label, group) in enumerate(groups):
        group = group.copy()
        group["method_label"] = group.apply(_method_label, axis=1)
        order = [method for method in ["raw", "linear", "isotonic"] if method in set(group["method_label"])]
        for col_id, metric in enumerate(
            [
                "relative_true_v_mse_vs_uncalibrated_all_data",
                "relative_calibration_error_plugin_vs_uncalibrated_all_data",
            ]
        ):
            ax = axes[row_id, col_id]
            values = []
            for method in order:
                sub = group[group["method_label"].eq(method)]
                values.append(float(pd.to_numeric(sub[metric], errors="coerce").iloc[0]) if not sub.empty else np.nan)
            ax.bar(np.arange(len(order)), values, color=[colors.get(method, "0.5") for method in order])
            ax.axhline(1.0, color="0.25", linewidth=0.8, linestyle="--")
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=20, ha="right")
            ax.set_ylim(0.0, max(1.25, np.nanmax(values) * 1.1 if np.isfinite(values).any() else 1.25))
            if col_id == 0:
                ax.set_ylabel(label)
            ax.set_title("Value MSE ratio" if col_id == 0 else "Bellman cal. ratio")
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    title = "Focused Bellman calibration story" + (" (debug only)" if mode == "debug" or n_rep < 2 else "")
    fig.suptitle(title, y=0.995)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(figures_dir / "focused_neurips_calibration_story.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / "focused_neurips_calibration_story.pdf", bbox_inches="tight")
    plt.close(fig)
    return [
        figures_dir / "focused_neurips_calibration_story.png",
        figures_dir / "focused_neurips_calibration_story.pdf",
    ]


def make_plots(results_dir: str | Path, figures_dir: str | Path, allow_invalid: bool = False) -> list[Path]:
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    gate = load_gate(results_dir)
    mode_hint = "paper" if "paper" in str(results_dir) or "paper" in str(figures_dir) else "debug"
    if gate and not gate.get("gate_passed", False) and mode_hint == "paper" and not allow_invalid:
        raise RuntimeError("Refusing to create paper plots because the well-specified validation gate failed.")

    df = _eligible_summary(results_dir)
    if "main_figure_role" in df:
        df = df[df["main_figure_role"].astype(str) != "diagnostic_only"].copy()
    if "calibration_evidence_status" in df:
        evidence_df = df[
            df["calibration_evidence_status"].astype(str).isin({"strong", "neutral"})
            | df["calibration_protocol"].astype(str).str.startswith("uncalibrated")
        ].copy()
    else:
        evidence_df = df
    mode = _mode_label(df, results_dir)
    made: list[Path] = []
    made += _focused_neurips_plot(results_dir, df, figures_dir, mode)
    made += _evidence_scatter_plot(df, figures_dir, mode)
    made += _errorbar_plot(
        evidence_df,
        figures_dir,
        x_col="sample_size",
        y_col="mse",
        yerr_col="mse_mc_se",
        stem="mse_vs_sample_size",
        title="MSE versus sample size",
        mode=mode,
    )
    made += _errorbar_plot(
        evidence_df,
        figures_dir,
        x_col="sample_size",
        y_col="relative_mse_vs_uncalibrated_all_data",
        stem="relative_mse_vs_sample_size",
        title="Relative MSE versus sample size",
        mode=mode,
    )
    x_shift = "policy_shift_setting" if "policy_shift_setting" in df else "coverage_setting"
    shift_df = evidence_df.copy()
    if "suite_name" in shift_df and (shift_df["suite_name"].astype(str) == "coverage_sweep").any():
        # Policy-shift figures are interpretable only inside the coverage sweep.
        # The full evidence table contains multiple suites with unrelated
        # policy-shift values; connecting those rows creates artificial spikes.
        shift_df = shift_df[shift_df["suite_name"].astype(str) == "coverage_sweep"].copy()
    if "calibration_protocol" in shift_df:
        shift_df = shift_df[
            shift_df["calibration_protocol"].astype(str).isin({"cross", "uncalibrated_all_data"})
        ].copy()
    made += _errorbar_plot(
        shift_df,
        figures_dir,
        x_col=x_shift,
        y_col="mse",
        yerr_col="mse_mc_se",
        stem="mse_vs_policy_shift",
        title="MSE versus policy shift",
        group_col="calibrator",
        mode=mode,
    )
    made += _errorbar_plot(
        shift_df,
        figures_dir,
        x_col=x_shift,
        y_col="relative_mse_vs_uncalibrated_all_data",
        stem="relative_mse_vs_policy_shift",
        title="Relative MSE versus policy shift",
        group_col="calibrator",
        mode=mode,
    )
    made += _bias_variance_plot(evidence_df, figures_dir, mode)
    made += _scatter_plot(evidence_df, figures_dir, mode)
    for x_col, y_col, stem, title in [
        ("calibrator", "mse", "calibrator_comparison", "Calibrator comparison"),
        ("calibration_protocol", "mse", "calibration_protocol_comparison", "Calibration protocol comparison"),
        ("train_fraction", "mse", "split_fraction_comparison", "Split-fraction comparison"),
        ("misspecification_setting", "relative_mse_vs_uncalibrated_all_data", "misspecification_sweep", "Misspecification/distortion sweep"),
        ("learner_quality_regime", "mse", "mse_by_learner_quality", "MSE by learner-quality regime"),
        ("calibration_difficulty", "relative_mse_vs_uncalibrated_all_data", "relative_mse_by_calibration_difficulty", "Relative MSE by calibration difficulty"),
        ("baseline_learner", "mse", "baseline_family_comparison", "Baseline-family comparison"),
        ("baseline_learner", "runtime", "runtime_comparison", "Runtime comparison"),
    ]:
        made += _errorbar_plot(evidence_df, figures_dir, x_col=x_col, y_col=y_col, stem=stem, title=title, mode=mode)
    if "actual_bellman_iterations" in df:
        under = df[df.get("suite_name", "").astype(str).eq("undertraining_sweep")].copy()
        made += _errorbar_plot(
            under,
            figures_dir,
            x_col="actual_bellman_iterations",
            y_col="relative_mse_vs_uncalibrated_all_data",
            stem="undertraining_iteration_sweep",
            title="Undertraining sweep by Bellman iterations",
            mode=mode,
        )
    if "feature_dimension" in df:
        incomplete = df[df.get("suite_name", "").astype(str).eq("bellman_incomplete_sweep")].copy()
        made += _errorbar_plot(
            incomplete,
            figures_dir,
            x_col="feature_dimension",
            y_col="relative_calibration_error_plugin_vs_uncalibrated_all_data",
            stem="bellman_incomplete_capacity_sweep",
            title="Bellman-incomplete capacity sweep",
            mode=mode,
        )
    made += _coverage_stratified_plot(results_dir, figures_dir, mode)
    made += _split_stability_plot(results_dir, figures_dir, mode)

    summary_path = results_dir / "summary.csv"
    if summary_path.exists():
        full = pd.read_csv(summary_path)
        made += _errorbar_plot(
            full,
            figures_dir,
            x_col="baseline_learner",
            y_col="failure_rate",
            stem="failure_rate_comparison",
            title="Failure-rate comparison",
            mode=mode,
        )
        made += _errorbar_plot(
            full,
            figures_dir,
            x_col="learner_quality_regime",
            y_col="failure_rate",
            stem="failure_rate_by_learner_quality",
            title="Failure rate by learner-quality regime",
            mode=mode,
        )
    return made
