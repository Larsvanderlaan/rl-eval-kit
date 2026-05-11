#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("FQE_calibration_neurips/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("FQE_calibration_neurips/.cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("relative_true_v_mse_vs_uncalibrated_all_data", "Value-fn. MSE", "#4c78a8"),
    ("relative_calibration_error_plugin_vs_uncalibrated_all_data", "Bellman cal.", "#f58518"),
]


def _policy_shift_label(x: float) -> str:
    if x <= 0.2:
        return "Good"
    if x <= 0.8:
        return "Moderate"
    return "Severe"


def _select_one(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if col not in df:
            raise KeyError(f"Missing required column {col!r}")
        mask &= df[col].astype(str).eq(str(value))
    out = df[mask].copy()
    if out.empty:
        raise ValueError(f"No row found for filters: {filters}")
    return out.iloc[0]


def _metric_values(row: pd.Series) -> list[float]:
    return [float(pd.to_numeric(row[col], errors="coerce")) for col, _, _ in METRICS]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, outdirs: list[Path], stem: str) -> None:
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
        if "paper_import_bundle" not in outdir.parts:
            fig.savefig(outdir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bar_panel(ax: plt.Axes, rows: list[pd.Series], labels: list[str], title: str, *, show_ylabel: bool) -> None:
    x = np.arange(len(rows))
    width = 0.30
    all_values: list[float] = []
    for j, (_, metric_label, color) in enumerate(METRICS):
        values = [_metric_values(row)[j] for row in rows]
        all_values.extend(values)
        ax.bar(x + (j - 0.5) * width, values, width=width, color=color, label=metric_label)
    ax.axhline(1.0, color="0.25", linewidth=0.8)
    finite = [v for v in all_values if np.isfinite(v)]
    ax.set_ylim(0.0, max(1.28, max(finite, default=1.0) + 0.12))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title, loc="left", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("ratio vs. uncalibrated")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)


def strict_cross_calibration_story(summary: pd.DataFrame, outdirs: list[Path]) -> None:
    """Readable multi-panel submission figure from the strict cross-calibration run."""
    finite_rows = [
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            learner_variant="random_feature_fqe_iter2",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            learner_variant="random_feature_fqe_iter3",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            learner_variant="random_feature_fqe_iter2",
            calibration_protocol="cross",
            calibrator="isotonic_histogram",
        ),
    ]
    monotone_rows = [
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            learner_variant="linear_fqe_misspecified",
            misspecification_setting="monotone_distortion",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            learner_variant="linear_fqe_misspecified",
            misspecification_setting="monotone_distortion",
            calibration_protocol="cross",
            calibrator="histogram",
        ),
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            learner_variant="linear_fqe_misspecified",
            misspecification_setting="monotone_distortion",
            calibration_protocol="cross",
            calibrator="isotonic_histogram",
        ),
    ]
    comparator_rows = [
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            learner_variant="random_feature_fqe_iter2",
            calibration_protocol="cross",
            calibrator="linear",
        ),
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            learner_variant="linear_fqe_misspecified",
            misspecification_setting="affine",
            calibration_protocol="cross",
            calibrator="linear",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), constrained_layout=False)
    _bar_panel(
        axes[0],
        finite_rows,
        ["RF K=2\nisotonic", "RF K=3\nisotonic", "RF K=2\niso.-hist."],
        "A. Finite-iteration bias",
        show_ylabel=True,
    )
    _bar_panel(
        axes[1],
        monotone_rows,
        ["Isotonic", "Histogram", "Iso.-hist."],
        "B. Monotone misspecification",
        show_ylabel=False,
    )
    _bar_panel(
        axes[2],
        comparator_rows,
        ["RF K=2\nlinear", "Affine FQE\nlinear"],
        "C. Linear comparator",
        show_ylabel=False,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.74, bottom=0.28, left=0.08, right=0.995, wspace=0.24)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    _save(fig, outdirs, "calibration_story_compact")


def _coverage_rows(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary[
        (summary["suite_name"].astype(str) == "coverage_sweep")
        & (summary["baseline_learner"].astype(str) == "random_feature_fqe")
        & (summary["calibration_protocol"].astype(str) == "cross")
        & (summary["calibrator"].astype(str) == "linear")
    ].copy()
    if df.empty:
        raise ValueError("No coverage-sweep rows found for random_feature_fqe cross+linear")
    df["policy_shift_numeric"] = pd.to_numeric(df["policy_shift_setting"], errors="coerce")
    df = df.sort_values("policy_shift_numeric")
    return df


def calibration_story_compact(summary: pd.DataFrame, coverage: pd.DataFrame, outdirs: list[Path]) -> None:
    undertraining_rows = [
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            baseline_learner="random_feature_fqe",
            learner_variant="random_feature_fqe_iter2",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
        _select_one(
            summary,
            suite_name="undertraining_sweep",
            baseline_learner="neural_fqe",
            learner_variant="neural_fqe_iter2",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
    ]
    misspec_rows = [
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            misspecification_setting="affine",
            baseline_learner="linear_fqe",
            learner_variant="linear_fqe_misspecified",
            calibration_protocol="cross",
            calibrator="linear",
        ),
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            misspecification_setting="affine",
            baseline_learner="random_feature_fqe",
            learner_variant="random_feature_fqe_restricted",
            calibration_protocol="cross",
            calibrator="linear",
        ),
        _select_one(
            summary,
            suite_name="model_misspecification_sweep",
            misspecification_setting="affine",
            baseline_learner="neural_fqe",
            learner_variant="neural_fqe_small_overregularized",
            calibration_protocol="cross",
            calibrator="isotonic",
        ),
    ]
    coverage_rows = _coverage_rows(summary)

    severe = coverage[
        (coverage["suite_name"].astype(str) == "coverage_sweep")
        & (coverage["coverage_setting"].astype(str) == "severe")
        & (coverage["baseline_learner"].astype(str) == "random_feature_fqe")
    ].copy()
    severe["method"] = np.where(
        severe["calibration_protocol"].astype(str).eq("uncalibrated_all_data"),
        "Uncalibrated",
        np.where(
            (severe["calibration_protocol"].astype(str).eq("cross"))
            & (severe["calibrator"].astype(str).eq("linear")),
            "Cross + linear",
            "drop",
        ),
    )
    severe = severe[severe["method"].isin(["Uncalibrated", "Cross + linear"])]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.35), constrained_layout=True)
    _bar_panel(
        axes[0, 0],
        undertraining_rows,
        ["RF FQE K=2\nisotonic", "Neural FQE K=2\nisotonic"],
        "A. Finite-iteration bias",
        show_ylabel=True,
    )
    _bar_panel(
        axes[0, 1],
        misspec_rows,
        ["Linear FQE", "Restricted RF", "Small neural"],
        "B. Affine misspecification",
        show_ylabel=False,
    )

    x = np.arange(len(coverage_rows))
    labels = [_policy_shift_label(float(v)) for v in coverage_rows["policy_shift_numeric"]]
    for col, metric_label, color in METRICS:
        axes[1, 0].plot(
            x,
            pd.to_numeric(coverage_rows[col], errors="coerce"),
            marker="o",
            linewidth=1.5,
            color=color,
            label=metric_label,
        )
    axes[1, 0].axhline(1.0, color="0.25", linewidth=0.8)
    axes[1, 0].set_ylim(0.3, 1.22)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel("ratio vs. uncalibrated")
    axes[1, 0].set_title("C. Coverage stress", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.2, linewidth=0.5)

    for method, group in severe.groupby("method", sort=False):
        group = group.sort_values("coverage_stratum")
        axes[1, 1].plot(
            pd.to_numeric(group["coverage_stratum"], errors="coerce"),
            pd.to_numeric(group["mean_coverage_stratum_calibration_error"], errors="coerce"),
            marker="o" if method == "Uncalibrated" else "s",
            linewidth=1.5,
            color="#777777" if method == "Uncalibrated" else "#f58518",
            label=method,
        )
    axes[1, 1].set_xlabel("density-ratio stratum")
    axes[1, 1].set_ylabel("Bellman cal. error")
    axes[1, 1].set_title("D. Severe coverage", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.2, linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    axes[1, 1].legend(frameon=False, loc="upper right")
    _save(fig, outdirs, "calibration_story_compact")


def relative_policy_shift(summary: pd.DataFrame, outdirs: list[Path]) -> None:
    df = summary[
        (summary["suite_name"].astype(str) == "coverage_sweep")
        & (summary["baseline_learner"].astype(str) == "random_feature_fqe")
        & (summary["calibration_protocol"].astype(str) == "cross")
        & (summary["calibrator"].astype(str).isin(["linear", "isotonic"]))
    ].copy()
    df["policy_shift_numeric"] = pd.to_numeric(df["policy_shift_setting"], errors="coerce")
    df = df.sort_values("policy_shift_numeric")

    fig, ax = plt.subplots(figsize=(3.25, 2.25), constrained_layout=True)
    for calibrator, group in df.groupby("calibrator", sort=False):
        ax.plot(
            group["policy_shift_numeric"],
            pd.to_numeric(group["relative_mse_vs_uncalibrated_all_data"], errors="coerce"),
            marker="o" if calibrator == "linear" else "s",
            linewidth=1.4,
            label=str(calibrator),
        )
    ax.axhline(1.0, color="0.25", linewidth=0.8)
    ax.set_ylim(0.95, 1.18)
    ax.set_xlabel("policy shift")
    ax.set_ylabel("relative value MSE")
    ax.set_title("Coverage sweep only")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    _save(fig, outdirs, "relative_mse_vs_policy_shift")


def coverage_limitation_compact(summary: pd.DataFrame, coverage: pd.DataFrame, outdirs: list[Path]) -> None:
    cov = _coverage_rows(summary)
    x = np.arange(len(cov))
    labels = [_policy_shift_label(float(v)) for v in cov["policy_shift_numeric"]]
    severe = coverage[
        (coverage["suite_name"].astype(str) == "coverage_sweep")
        & (coverage["coverage_setting"].astype(str) == "severe")
        & (coverage["baseline_learner"].astype(str) == "random_feature_fqe")
    ].copy()
    severe["method"] = np.where(
        severe["calibration_protocol"].astype(str).eq("uncalibrated_all_data"),
        "Uncalibrated",
        np.where(
            (severe["calibration_protocol"].astype(str).eq("cross"))
            & (severe["calibrator"].astype(str).eq("linear")),
            "Cross + linear",
            "drop",
        ),
    )
    severe = severe[severe["method"].isin(["Uncalibrated", "Cross + linear"])]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    for col, metric_label, color in METRICS:
        axes[0].plot(x, pd.to_numeric(cov[col], errors="coerce"), marker="o", linewidth=1.5, color=color, label=metric_label)
    axes[0].axhline(1.0, color="0.25", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.3, 1.22)
    axes[0].set_ylabel("ratio vs. uncalibrated")
    axes[0].set_title("A. Coverage stress", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="y", alpha=0.2, linewidth=0.5)

    for method, group in severe.groupby("method", sort=False):
        group = group.sort_values("coverage_stratum")
        axes[1].plot(
            pd.to_numeric(group["coverage_stratum"], errors="coerce"),
            pd.to_numeric(group["mean_coverage_stratum_calibration_error"], errors="coerce"),
            marker="o" if method == "Uncalibrated" else "s",
            linewidth=1.5,
            color="#777777" if method == "Uncalibrated" else "#f58518",
            label=method,
        )
    axes[1].set_xlabel("density-ratio stratum")
    axes[1].set_ylabel("Bellman cal. error")
    axes[1].set_title("B. Severe shift by stratum", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.5)
    _save(fig, outdirs, "coverage_limitation_compact")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="FQE_calibration_neurips/results/paper")
    parser.add_argument(
        "--figures_dir",
        action="append",
        help="Output directory. May be supplied multiple times.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdirs = (
        [Path(p) for p in args.figures_dir]
        if args.figures_dir
        else [
            Path("FQE_calibration_neurips/figures/paper"),
            Path("FQE_calibration_neurips/paper_import_bundle/figures"),
        ]
    )

    audit_path = results_dir / "strict_cross_promotion_audit.csv"
    if results_dir.name.startswith("paper_strict_cross_submission") and audit_path.exists():
        summary = pd.read_csv(audit_path)
    else:
        summary = pd.read_csv(results_dir / "summary.csv")
    coverage_path = results_dir / "coverage_stratified_error.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

    _style()
    suite_names = set(summary["suite_name"].astype(str)) if "suite_name" in summary else set()
    if results_dir.name.startswith("paper_strict_cross_submission") or "model_misspecification_sweep" not in suite_names:
        strict_cross_calibration_story(summary, outdirs)
    else:
        calibration_story_compact(summary, coverage, outdirs)
        paper_figure_outdirs = [p for p in outdirs if "paper_import_bundle" not in p.parts]
        if paper_figure_outdirs:
            relative_policy_shift(summary, paper_figure_outdirs)
            coverage_limitation_compact(summary, coverage, paper_figure_outdirs)
    for outdir in outdirs:
        print(f"Wrote compact import figures to {outdir}")


if __name__ == "__main__":
    main()
