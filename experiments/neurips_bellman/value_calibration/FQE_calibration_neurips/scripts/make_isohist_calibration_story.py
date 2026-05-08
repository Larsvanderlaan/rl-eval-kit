#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("relative_true_v_mse_vs_uncalibrated_all_data", "MSE ratio"),
    ("relative_calibration_error_plugin_vs_uncalibrated_all_data", "CAL ratio"),
]
COLORS = {"Linear": "#3572A1", "Histogram": "#B65C2A", "Iso-hist": "#7A5BA6"}


def _row(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key].astype(str).eq(str(value))
    out = df[mask]
    if out.empty:
        raise RuntimeError(f"No row for filters {filters}")
    return out.iloc[0]


def _values(row: pd.Series) -> tuple[float, float]:
    return tuple(float(row[col]) for col, _ in METRICS)


def main() -> None:
    strict = pd.read_csv("FQE_calibration_neurips/results/paper_strict_cross_submission_v3_50k/summary.csv")
    paper = pd.read_csv("FQE_calibration_neurips/results/paper/summary.csv")

    panels = [
        (
            "Affine\nmechanism",
            [
                ("Linear", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_affine_distorted", calibration_protocol="cross", calibrator="linear")),
                ("Histogram", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_affine_distorted", calibration_protocol="cross", calibrator="histogram")),
                ("Iso-hist", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_affine_distorted", calibration_protocol="cross", calibrator="isotonic_histogram")),
            ],
        ),
        (
            "Finite iter.\nK=4",
            [
                ("Linear", _row(strict, suite_name="undertraining_sweep", learner_variant="random_feature_fqe_iter4", calibration_protocol="cross", calibrator="linear")),
                ("Histogram", _row(strict, suite_name="undertraining_sweep", learner_variant="random_feature_fqe_iter4", calibration_protocol="cross", calibrator="histogram")),
                ("Iso-hist", _row(strict, suite_name="undertraining_sweep", learner_variant="random_feature_fqe_iter4", calibration_protocol="cross", calibrator="isotonic_histogram")),
            ],
        ),
        (
            "Monotone\nmisspec.",
            [
                ("Linear", _row(strict, suite_name="model_misspecification_sweep", learner_variant="linear_fqe_misspecified", misspecification_setting="monotone_distortion", calibration_protocol="cross", calibrator="linear")),
                ("Histogram", _row(strict, suite_name="model_misspecification_sweep", learner_variant="linear_fqe_misspecified", misspecification_setting="monotone_distortion", calibration_protocol="cross", calibrator="histogram")),
                ("Iso-hist", _row(strict, suite_name="model_misspecification_sweep", learner_variant="linear_fqe_misspecified", misspecification_setting="monotone_distortion", calibration_protocol="cross", calibrator="isotonic_histogram")),
            ],
        ),
        (
            "Monotone\nmechanism",
            [
                ("Linear", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_monotone_saturation_distorted", calibration_protocol="cross", calibrator="linear")),
                ("Histogram", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_monotone_saturation_distorted", calibration_protocol="cross", calibrator="histogram")),
                ("Iso-hist", _row(paper, suite_name="mechanism_distortion_sweep", learner_variant="random_feature_fqe_monotone_saturation_distorted", calibration_protocol="cross", calibrator="isotonic_histogram")),
            ],
        ),
    ]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, len(panels), figsize=(7.2, 2.25), sharey="row")
    for col, (title, methods) in enumerate(panels):
        labels = [name for name, _ in methods]
        x = np.arange(len(methods))
        vals = np.array([_values(row) for _, row in methods], dtype=float)
        for row_idx, (_, ylabel) in enumerate(METRICS):
            ax = axes[row_idx, col]
            ax.bar(x, vals[:, row_idx], color=[COLORS[name] for name in labels], width=0.58)
            ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle=(0, (2, 2)))
            ax.set_ylim(0.0, 1.08)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(title)
            for i, value in enumerate(vals[:, row_idx]):
                ax.text(i, min(value + 0.035, 1.04), f"{value:.2f}", ha="center", va="bottom", fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="0.88", linewidth=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name]) for name in ["Linear", "Histogram", "Iso-hist"]]
    fig.legend(handles, ["Linear", "Histogram", "Iso-hist"], loc="upper center", ncol=3, frameon=False)
    fig.text(0.98, 0.02, "Lower is better", ha="right", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.83), w_pad=0.8, h_pad=0.6)

    outdirs = [
        Path("FQE_calibration_neurips/paper_import_bundle/figures"),
        Path("FQE_calibration_neurips/figures/paper"),
        Path("/Users/larsvanderlaan/Downloads/paper_import_bundle/figures"),
        Path("/Users/larsvanderlaan/repos/paper_presentations/neurips_bellman/papers/calibration/figures"),
    ]
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "calibration_story_compact.pdf", bbox_inches="tight")
        fig.savefig(outdir / "calibration_story_compact.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
