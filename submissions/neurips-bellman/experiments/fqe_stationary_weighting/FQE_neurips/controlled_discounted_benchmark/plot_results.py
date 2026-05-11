from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "standard_fqe",
    "oracle_weighted_fqe",
    "oracle_weighted_fqe_clipped",
    "estimated_weighted_fqe",
    "estimated_weighted_fqe_clipped",
    "estimated_weighted_fqe_clip95",
    "estimated_weighted_fqe_clip99_ess40",
]

METHOD_LABELS = {
    "standard_fqe": "standard",
    "oracle_weighted_fqe": "oracle",
    "oracle_weighted_fqe_clipped": "oracle clipped",
    "estimated_weighted_fqe": "estimated",
    "estimated_weighted_fqe_clipped": "estimated clipped",
    "estimated_weighted_fqe_clip95": "estimated clip95",
    "estimated_weighted_fqe_clip99_ess40": "estimated clip99/ESS40",
    "linear_standard_fqe": "FQE",
    "linear_oracle_raw_fqe": "Oracle ratio",
    "linear_oracle_clipped_fqe": "oracle clipped",
    "linear_estimated_clipped_fqe": "estimated SW-FQE",
    "linear_estimated_fixed_cap_fqe": "fixed cap",
    "linear_estimated_ess_winsor_fqe": "ESS-adaptive",
    "linear_estimated_unregularized_fqe": "Local RBF (unreg.)",
    "linear_estimated_rbf_fixed_tikhonov_fqe": r"Local RBF (fixed $\eta=.1$)",
    "linear_estimated_rbf_oracle_tikhonov_fqe": "Local RBF (oracle-tuned)",
    "linear_estimated_quadratic_moment_fqe": "Correct quad. moment",
    "linear_estimated_quadratic_moment_cv_fqe": "Correct quad. (CV-reg.)",
    "linear_estimated_quadratic_moment_oracle_fqe": "Correct quad. (oracle-tuned)",
    "linear_estimated_tikhonov_fqe": "Local RBF (CV-Tikh.)",
    "linear_minimax_q_unregularized": "Minimax Q (unreg.)",
    "linear_minimax_q_cv_tikhonov": "Minimax Q (CV-Tikh.)",
    "linear_minimax_q_rbf": "Minimax Q (Tikh.)",
    "linear_minimax_q_oracle_tikhonov": "Minimax Q (oracle-tuned)",
    "linear_estimated_tikhonov_xfit_fqe": "CV-Tikh. xfit",
    "linear_estimated_tikhonov_1se_fqe": "CV-Tikh. 1SE",
    "linear_estimated_tikhonov_1se_xfit_fqe": "1SE xfit",
    "linear_estimated_cv_cap_fqe": "CV trunc.",
    "linear_neural_weighted_clipped_fqe": "linear neural weights",
    "neural_standard_fqe": "neural standard",
    "neural_oracle_clipped_fqe": "neural oracle clipped",
    "neural_estimated_clipped_fqe": "neural estimated clipped",
    "neural_neural_weighted_clipped_fqe": "neural neural weights",
}

METHOD_COLORS = {
    "standard_fqe": "#1f77b4",
    "oracle_weighted_fqe": "#2ca02c",
    "oracle_weighted_fqe_clipped": "#98df8a",
    "estimated_weighted_fqe": "#d62728",
    "estimated_weighted_fqe_clipped": "#ff7f0e",
    "estimated_weighted_fqe_clip95": "#9467bd",
    "estimated_weighted_fqe_clip99_ess40": "#8c564b",
    "linear_standard_fqe": "#1f77b4",
    "linear_oracle_raw_fqe": "#2ca02c",
    "linear_oracle_clipped_fqe": "#98df8a",
    "linear_estimated_clipped_fqe": "#ff7f0e",
    "linear_estimated_unregularized_fqe": "#d62728",
    "linear_estimated_rbf_fixed_tikhonov_fqe": "#e377c2",
    "linear_estimated_rbf_oracle_tikhonov_fqe": "#bcbd22",
    "linear_estimated_quadratic_moment_fqe": "#9467bd",
    "linear_estimated_quadratic_moment_cv_fqe": "#17becf",
    "linear_estimated_quadratic_moment_oracle_fqe": "#aec7e8",
    "linear_estimated_tikhonov_fqe": "#ff7f0e",
    "linear_minimax_q_unregularized": "#7f7f7f",
    "linear_minimax_q_cv_tikhonov": "#4d4d4d",
    "linear_minimax_q_rbf": "#000000",
    "linear_minimax_q_oracle_tikhonov": "#8c564b",
    "linear_estimated_fixed_cap_fqe": "#9467bd",
    "linear_estimated_ess_winsor_fqe": "#8c564b",
    "linear_estimated_cv_cap_fqe": "#7f7f7f",
    "linear_estimated_tikhonov_xfit_fqe": "#ffbb78",
    "linear_estimated_tikhonov_1se_fqe": "#bcbd22",
    "linear_estimated_tikhonov_1se_xfit_fqe": "#dbdb8d",
    "linear_neural_weighted_clipped_fqe": "#17becf",
}

for _eta_label, _eta_text in [
    ("0.001", "0.001"),
    ("0.01", "0.01"),
    ("1", "1"),
    ("10", "10"),
]:
    METHOD_LABELS[f"linear_estimated_rbf_fixed_tikhonov_eta_{_eta_label}_fqe"] = (
        rf"Local RBF (fixed $\eta={_eta_text}$)"
    )
    METHOD_LABELS[f"linear_minimax_q_eta_{_eta_label}"] = rf"Minimax Q (fixed $\eta={_eta_text}$)"
    METHOD_COLORS.setdefault(f"linear_estimated_rbf_fixed_tikhonov_eta_{_eta_label}_fqe", "#e377c2")
    METHOD_COLORS.setdefault(f"linear_minimax_q_eta_{_eta_label}", "#000000")


STATIONARY_PANEL_ESTIMATORS = [
    "linear_standard_fqe",
    "linear_oracle_raw_fqe",
    "linear_estimated_quadratic_moment_fqe",
    "linear_estimated_quadratic_moment_cv_fqe",
    "linear_estimated_quadratic_moment_oracle_fqe",
    "linear_minimax_q_unregularized",
    "linear_minimax_q_cv_tikhonov",
    "linear_minimax_q_rbf",
    "linear_minimax_q_oracle_tikhonov",
    "linear_estimated_fixed_cap_fqe",
    "linear_estimated_ess_winsor_fqe",
    "linear_estimated_unregularized_fqe",
    "linear_estimated_rbf_fixed_tikhonov_fqe",
    "linear_estimated_rbf_oracle_tikhonov_fqe",
    "linear_estimated_tikhonov_fqe",
    "linear_estimated_cv_cap_fqe",
]


def _raw_ratio_ess_lookup(rows: list[dict[str, str]]) -> dict[tuple[float, float], float]:
    """Coverage stress keyed by (shift, ratio_gamma), independent of estimator."""
    lookup: dict[tuple[float, float], float] = {}
    for row in rows:
        if row.get("estimator") != "linear_oracle_raw_fqe":
            continue
        key = (float(row["shift"]), float(row["ratio_gamma"]))
        lookup[key] = float(row["effective_sample_size_fraction"])
    return lookup


def _ess_range_label(values: list[float]) -> str:
    finite = [value for value in values if value > 0]
    if not finite:
        return "ESS n/a"
    lo, hi = min(finite), max(finite)
    if abs(lo - hi) < 0.005:
        return f"ESS {hi:.2f}"
    return f"ESS {lo:.2f}-{hi:.2f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_summary(results_root: Path) -> list[dict[str, str]]:
    for filename in ["final_summary.csv", "design_search_summary.csv", "smoke_summary.csv"]:
        path = results_root / filename
        if path.exists():
            return _read_csv(path)
    raise FileNotFoundError(f"No summary CSV found under {results_root}.")


def _select_main_slice(rows: list[dict[str, str]], results_root: Path) -> list[dict[str, str]]:
    selected_path = results_root / "selected_final_family.csv"
    if selected_path.exists():
        selected = _read_csv(selected_path)[0]
        preferred = [
            row
            for row in rows
            if row["feature_regime"] == "misspecified_affine"
            and int(float(row["sample_size"])) == 4000
            and float(row["gamma"]) == float(selected["gamma"])
            and float(row["process_noise_sd"]) == float(selected["process_noise_sd"])
            and float(row["behavior_action_sd"]) == float(selected["behavior_action_sd"])
            and float(row["shift"])
            in {
                float(selected["low_shift"]),
                float(selected["moderate_shift"]),
                float(selected["severe_shift"]),
            }
        ]
        if preferred:
            return preferred
    preferred = [
        row
        for row in rows
        if row["feature_regime"] == "misspecified_affine" and int(float(row["sample_size"])) == 4000
    ]
    if preferred:
        return preferred
    return rows


def _plot_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    path: Path,
    estimators: list[str] | None = None,
) -> None:
    if estimators is None:
        estimators = [estimator for estimator in METHOD_ORDER if any(row["estimator"] == estimator for row in rows)]
    plt.figure(figsize=(7.5, 4.8))
    for estimator in estimators:
        sub = [row for row in rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["shift"]))
        plt.plot(
            [float(row["shift"]) for row in sub],
            [float(row[metric]) for row in sub],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS.get(estimator),
            label=METHOD_LABELS.get(estimator, estimator),
        )
    plt.xlabel("Behavior-target shift")
    plt.ylabel(ylabel)
    if all(float(row[metric]) > 0.0 for row in rows if row.get(metric, "")):
        plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_weight_diagnostics(rows: list[dict[str, str]], path: Path) -> None:
    estimators = [
        estimator
        for estimator in sorted({row["estimator"] for row in rows})
        if estimator != "standard_fqe"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for estimator in estimators:
        sub = [row for row in rows if row["estimator"] == estimator]
        sub.sort(key=lambda row: float(row["shift"]))
        shifts = [float(row["shift"]) for row in sub]
        ess = [float(row["effective_sample_size_fraction"]) for row in sub]
        q99 = [float(row["weight_q99"]) for row in sub]
        max_weight = [float(row["weight_max"]) for row in sub]
        label = METHOD_LABELS.get(estimator, estimator)
        color = METHOD_COLORS.get(estimator)
        axes[0].plot(shifts, ess, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(shifts, q99, marker="o", linewidth=2.0, color=color, label=label)
        axes[2].plot(shifts, max_weight, marker="o", linewidth=2.0, color=color, label=label)
    axes[0].set_xlabel("Behavior-target shift")
    axes[0].set_ylabel("ESS fraction")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Behavior-target shift")
    axes[1].set_ylabel("Weight q99")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[2].set_xlabel("Behavior-target shift")
    axes[2].set_ylabel("Max weight")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_ratio_quality(rows: list[dict[str, str]], path: Path) -> None:
    estimators = [
        "estimated_weighted_fqe",
        "estimated_weighted_fqe_clipped",
        "estimated_weighted_fqe_clip95",
        "estimated_weighted_fqe_clip99_ess40",
        "oracle_weighted_fqe_clipped",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for estimator in estimators:
        sub = [row for row in rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["shift"]))
        shifts = [float(row["shift"]) for row in sub]
        ratio_rmse = [float(row["oracle_log_ratio_rmse"]) for row in sub]
        target_q = [float(row["target_q_mse"]) for row in sub]
        label = METHOD_LABELS.get(estimator, estimator)
        color = METHOD_COLORS.get(estimator)
        axes[0].plot(shifts, ratio_rmse, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(shifts, target_q, marker="o", linewidth=2.0, color=color, label=label)
    axes[0].set_xlabel("Behavior-target shift")
    axes[0].set_ylabel("Log-ratio RMSE vs oracle")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Behavior-target shift")
    axes[1].set_ylabel("Target occupancy Q MSE")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_summary_table(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "estimator",
        "shift",
        "target_q_mse",
        "behavior_q_mse",
        "policy_value_bias",
        "policy_value_variance",
        "policy_value_mse",
        "effective_sample_size_fraction",
        "weight_q95",
        "weight_q99",
        "weight_max",
        "fraction_clipped",
        "oracle_log_ratio_rmse",
        "oracle_estimated_weight_corr",
        "unstable_run_fraction",
        "unstable_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (float(item["shift"]), item["estimator"])):
            writer.writerow({column: row[column] for column in columns})


def _load_gamma_summary(results_root: Path) -> list[dict[str, str]]:
    for filename in ["gamma_sweep_summary.csv", "gamma_design_summary.csv", "gamma_smoke_summary.csv"]:
        path = results_root / filename
        if path.exists():
            return _read_csv(path)
    return []


def _gamma_main_slice(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = [
        row
        for row in rows
        if row["feature_regime"] == "misspecified_affine"
        and int(float(row["sample_size"])) == 4000
        and row["data_mode"] == "matched_iid"
    ]
    if preferred:
        return preferred
    return [row for row in rows if row.get("data_mode", "") == "matched_iid"]


def _gamma_feature_slice(rows: list[dict[str, str]], feature_regime: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("feature_regime") == feature_regime
        and int(float(row["sample_size"])) == 4000
        and row.get("data_mode") == "matched_iid"
    ]


def _plot_gamma_metric_by_shift(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    path: Path,
    estimators: list[str],
    fqe_family: str = "linear",
) -> None:
    shifts = sorted({float(row["shift"]) for row in rows})
    coverage_ess = _raw_ratio_ess_lookup(rows)
    fig, axes = plt.subplots(1, len(shifts), figsize=(7.2, 2.25), sharey=True)
    if len(shifts) == 1:
        axes = [axes]
    for ax, shift in zip(axes, shifts):
        for estimator in estimators:
            sub = [
                row
                for row in rows
                if row["estimator"] == estimator and row["fqe_family"] == fqe_family and float(row["shift"]) == shift
            ]
            if not sub:
                continue
            sub.sort(key=lambda row: float(row["ratio_gamma"]))
            ax.plot(
                [float(row["ratio_gamma"]) for row in sub],
                [float(row[metric]) for row in sub],
                marker="o",
                linewidth=1.4,
                color=METHOD_COLORS.get(estimator),
                label=METHOD_LABELS.get(estimator, estimator),
            )
        ess_values = [
            coverage_ess.get((shift, float(row["ratio_gamma"])), float(row["effective_sample_size_fraction"]))
            for row in rows
            if float(row["shift"]) == shift and row["fqe_family"] == fqe_family
        ]
        ax.set_title(f"shift={shift:g}\n({_ess_range_label(ess_values)})", fontsize=8)
        ax.set_xlabel(r"weight target $\gamma_\rho$", fontsize=8)
        ax.set_xlim(0.947, 1.003)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if all(float(row[metric]) > 0.0 for row in rows if row.get(metric, "")):
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel, fontsize=8)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_gamma_weight_quality(rows: list[dict[str, str]], path: Path) -> None:
    estimators = [
        "linear_oracle_raw_fqe",
        "linear_oracle_clipped_fqe",
        "linear_estimated_clipped_fqe",
        "linear_neural_weighted_clipped_fqe",
    ]
    moderate_rows = [row for row in rows if row["fqe_family"] == "linear" and float(row["shift"]) == 1.1]
    if not moderate_rows:
        moderate_rows = [row for row in rows if row["fqe_family"] == "linear"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for estimator in estimators:
        sub = [row for row in moderate_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["ratio_gamma"]))
        gamma_x = [float(row["ratio_gamma"]) for row in sub]
        ess_x = [float(row["effective_sample_size_fraction"]) for row in sub]
        label = METHOD_LABELS.get(estimator, estimator)
        color = METHOD_COLORS.get(estimator)
        axes[0].plot(gamma_x, ess_x, marker="o", linewidth=1.2, color=color, label=label)

        gamma_order = sorted(range(len(sub)), key=lambda idx: gamma_x[idx])
        axes[1].plot(
            [gamma_x[idx] for idx in gamma_order],
            [float(sub[idx]["weight_q99"]) for idx in gamma_order],
            marker="o",
            linewidth=1.2,
            color=color,
            label=label,
        )
        axes[2].plot(
            [gamma_x[idx] for idx in gamma_order],
            [float(sub[idx]["weight_max"]) for idx in gamma_order],
            marker="o",
            linewidth=1.2,
            color=color,
            label=label,
        )
    axes[0].set_xlabel(r"weight target $\gamma_\rho$", fontsize=8)
    axes[0].set_ylabel("ESS", fontsize=8)
    axes[1].set_xlabel(r"weight target $\gamma_\rho$", fontsize=8)
    axes[1].set_ylabel("weight q99", fontsize=8)
    axes[2].set_xlabel(r"weight target $\gamma_\rho$", fontsize=8)
    axes[2].set_ylabel("max weight", fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in axes:
        ax.set_xlim(0.947, 1.003)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_stationary_shift_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    path: Path,
    estimators: list[str],
    fqe_family: str = "linear",
) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == fqe_family and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.25))
    for estimator in estimators:
        sub = [row for row in sub_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])))
        ax.plot(
            [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub],
            [float(row[metric]) for row in sub],
            marker="o",
            linewidth=1.35,
            markersize=3.2,
            color=METHOD_COLORS.get(estimator),
            label=METHOD_LABELS.get(estimator, estimator),
        )
    ax.set_xlabel("ESS fraction", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=5.5, loc="best")
    if raw_ess_by_shift:
        ess_values = list(raw_ess_by_shift.values())
        ax.set_xticks(sorted(ess_values, reverse=True), minor=True)
        ax.tick_params(axis="x", which="minor", length=2.0)
        x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
        ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
        top_ticks = [
            tick
            for tick in [0.0, 1.0, 1.5, 1.9, 2.0, 2.06]
            if any(abs(float(row["shift"]) - tick) < 1e-12 for row in sub_rows)
        ]
        if top_ticks:
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            top.set_xticks([raw_ess_by_shift[tick] for tick in top_ticks])
            top.set_xticks(sorted(ess_values, reverse=True), minor=True)
            top.set_xticklabels([f"{tick:g}" for tick in top_ticks])
            top.set_xlabel(r"behavior-target shift $\Delta$", fontsize=7, labelpad=2)
            top.tick_params(labelsize=6, pad=1)
            top.tick_params(axis="x", which="minor", length=2.0)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_stationary_norm_diagnostics(rows: list[dict[str, str]], path: Path) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear" and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    metrics = [
        ("target_q_mse", "target stationary"),
        ("behavior_target_action_q_mse", "behavior states, target actions"),
        ("behavior_q_mse", "behavior"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        for estimator in STATIONARY_PANEL_ESTIMATORS:
            sub = [row for row in sub_rows if row["estimator"] == estimator and metric in row]
            if not sub:
                continue
            sub.sort(key=lambda row: raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])))
            ax.plot(
                [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub],
                [float(row[metric]) for row in sub],
                marker="o",
                linewidth=1.0,
                markersize=2.4,
                color=METHOD_COLORS.get(estimator),
                label=METHOD_LABELS.get(estimator, estimator),
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("ESS fraction", fontsize=7)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        if raw_ess_by_shift:
            ess_values = list(raw_ess_by_shift.values())
            x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
            ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
            ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    axes[0].set_ylabel("Q MSE", fontsize=8)
    axes[0].legend(frameon=False, fontsize=5.2, loc="best")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_stationary_shift_weight_quality(
    rows: list[dict[str, str]],
    path: Path,
    *,
    estimators: list[str] | None = None,
) -> None:
    if estimators is None:
        estimators = [
            "linear_oracle_raw_fqe",
            "linear_estimated_quadratic_moment_fqe",
            "linear_estimated_unregularized_fqe",
            "linear_estimated_fixed_cap_fqe",
            "linear_estimated_ess_winsor_fqe",
            "linear_estimated_tikhonov_fqe",
            "linear_estimated_cv_cap_fqe",
        ]
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear" and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25))
    for estimator in estimators:
        sub = [row for row in sub_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["shift"]))
        shifts = [float(row["shift"]) for row in sub]
        ess_x = [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub]
        label = METHOD_LABELS.get(estimator, estimator)
        color = METHOD_COLORS.get(estimator)
        axes[0].plot(
            shifts,
            [float(row["effective_sample_size_fraction"]) for row in sub],
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            color=color,
            label=label,
        )
        axes[1].plot(
            ess_x,
            [float(row["weight_q99"]) for row in sub],
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            color=color,
            label=label,
        )
        axes[2].plot(
            ess_x,
            [float(row["weight_max"]) for row in sub],
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            color=color,
            label=label,
        )
    axes[1].set_ylabel("weight q99", fontsize=8)
    axes[2].set_ylabel("max weight", fontsize=8)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].set_xlabel(r"shift $\Delta$", fontsize=8)
    axes[0].set_ylabel("empirical ESS fraction", fontsize=8)
    all_shifts = sorted({float(row["shift"]) for row in sub_rows})
    axes[0].set_xticks([tick for tick in [0.0, 0.5, 1.0, 1.5, 2.0] if tick in all_shifts])
    axes[0].set_xticks(all_shifts, minor=True)
    axes[0].tick_params(axis="x", which="minor", length=2.0)
    for axis_idx, ax in enumerate(axes[1:], start=1):
        ax.set_xlabel("ESS fraction (oracle)", fontsize=8)
        if raw_ess_by_shift:
            ess_values = list(raw_ess_by_shift.values())
            x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
            ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
            ax.set_xticks([0.8, 0.6, 0.4, 0.2, 0.05])
            ax.set_xticks(sorted(ess_values, reverse=True), minor=True)
            ax.tick_params(axis="x", which="minor", length=2.0)
            top_ticks = [
                tick
                for tick in [0.0, 1.0, 1.5, 1.9, 2.0, 2.06]
                if tick in raw_ess_by_shift
            ]
            if top_ticks:
                top = ax.twiny()
                top.set_xlim(ax.get_xlim())
                top.set_xticks([raw_ess_by_shift[tick] for tick in top_ticks])
                top.set_xticks(sorted(ess_values, reverse=True), minor=True)
                top.set_xticklabels([f"{tick:g}" for tick in top_ticks])
                top.set_xlabel(r"shift $\Delta$" if axis_idx == 1 else "", fontsize=6.5, labelpad=1.0)
                top.tick_params(labelsize=5.5, pad=0.5)
                top.tick_params(axis="x", which="minor", length=2.0)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=5.5, loc="best")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_normalized_tikhonov_sensitivity(rows: list[dict[str, str]], path: Path) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear" and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    selected_shifts = [
        shift
        for shift in [1.5, 1.8, 1.95, 2.0, 2.05]
        if any(abs(float(row["shift"]) - shift) < 1e-12 for row in sub_rows)
    ]
    if not selected_shifts:
        return
    fig, axes = plt.subplots(1, len(selected_shifts), figsize=(1.8 * len(selected_shifts), 2.05), sharey=True)
    if len(selected_shifts) == 1:
        axes = [axes]
    families = [
        ("local_rbf", "Local RBF", "#e377c2"),
        ("minimax", "Minimax Q", "#000000"),
    ]
    for ax, shift in zip(axes, selected_shifts):
        shift_rows = [row for row in sub_rows if abs(float(row["shift"]) - shift) < 1e-12]
        for family_key, label, color in families:
            family_rows = []
            for row in shift_rows:
                estimator = row["estimator"]
                if family_key == "local_rbf" and not estimator.startswith(
                    "linear_estimated_rbf_fixed_tikhonov"
                ):
                    continue
                if family_key == "minimax" and not (
                    estimator == "linear_minimax_q_rbf"
                    or estimator.startswith("linear_minimax_q_eta_")
                ):
                    continue
                try:
                    eta = float(row["normalized_tikhonov_eta"])
                    mse = float(row["target_q_mse"])
                except (KeyError, TypeError, ValueError):
                    continue
                if eta > 0 and mse > 0:
                    family_rows.append((eta, mse))
            family_rows = sorted(set(family_rows))
            if family_rows:
                ax.plot(
                    [item[0] for item in family_rows],
                    [item[1] for item in family_rows],
                    marker="o",
                    linewidth=1.25,
                    markersize=2.8,
                    color=color,
                    label=label,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(rf"$\Delta={shift:g}$", fontsize=7)
        ax.set_xlabel(r"normalized ridge $\eta$", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Target reference Q MSE", fontsize=7)
    axes[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_minimax_vs_weighting_overlap_regimes(rows: list[dict[str, str]], path: Path) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear"
        and row.get("feature_regime") == "misspecified_affine"
        and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    if not sub_rows:
        return
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    curve_estimators = [
        "linear_standard_fqe",
        "linear_oracle_raw_fqe",
        "linear_estimated_tikhonov_fqe",
        "linear_estimated_quadratic_moment_cv_fqe",
        "linear_minimax_q_cv_tikhonov",
        "linear_minimax_q_rbf",
    ]
    practical_estimators = [
        "linear_standard_fqe",
        "linear_estimated_tikhonov_fqe",
        "linear_estimated_quadratic_moment_cv_fqe",
        "linear_minimax_q_cv_tikhonov",
        "linear_minimax_q_rbf",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    for estimator in curve_estimators:
        sub = [row for row in sub_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])), reverse=True)
        ax.plot(
            [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub],
            [float(row["target_q_mse"]) for row in sub],
            marker="o",
            linewidth=1.35,
            markersize=3.0,
            color=METHOD_COLORS.get(estimator),
            label=METHOD_LABELS.get(estimator, estimator),
        )
    ax.axvspan(0.0, 0.23, color="#d9d9d9", alpha=0.35, lw=0)
    ax.text(0.205, 0.94, "severe tail", transform=ax.get_xaxis_transform(), fontsize=6.5, ha="center", va="top")
    ax.set_xlabel("ESS fraction (oracle)", fontsize=8)
    ax.set_ylabel("Target reference Q MSE", fontsize=8)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    if raw_ess_by_shift:
        ess_values = list(raw_ess_by_shift.values())
        x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
        ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
        ax.set_xticks([0.8, 0.6, 0.4, 0.2, 0.1])
        top_ticks = [
            tick
            for tick in [0.0, 1.0, 1.5, 1.8, 1.9, 1.95]
            if tick in raw_ess_by_shift
        ]
        if top_ticks:
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            top.set_xticks([raw_ess_by_shift[tick] for tick in top_ticks])
            top.set_xticklabels([f"{tick:g}" for tick in top_ticks])
            top.set_xlabel(r"behavior-target shift $\Delta$", fontsize=7, labelpad=2)
            top.tick_params(labelsize=6, pad=1)
    ax.legend(frameon=False, fontsize=5.5, loc="best")

    regimes = [
        ("usable\noverlap", lambda ess: ess >= 0.35),
        ("severe\ntail", lambda ess: ess <= 0.23),
    ]
    x_base = list(range(len(regimes)))
    width = 0.18
    offsets = [-2.0 * width, -1.0 * width, 0.0, 1.0 * width, 2.0 * width]
    for method_idx, estimator in enumerate(practical_estimators):
        heights = []
        for _name, keep in regimes:
            values = [
                float(row["target_q_mse"])
                for row in sub_rows
                if row["estimator"] == estimator
                and keep(raw_ess_by_shift.get(float(row["shift"]), float("nan")))
            ]
            regime_best = []
            for practical in practical_estimators:
                regime_best.extend(
                    float(row["target_q_mse"])
                    for row in sub_rows
                    if row["estimator"] == practical
                    and keep(raw_ess_by_shift.get(float(row["shift"]), float("nan")))
                )
            best = min(regime_best) if regime_best else float("nan")
            mean_value = sum(values) / len(values) if values else float("nan")
            heights.append(mean_value / best if best and best > 0 else float("nan"))
        axes[1].bar(
            [x + offsets[method_idx] for x in x_base],
            heights,
            width=width,
            color=METHOD_COLORS.get(estimator),
            label=METHOD_LABELS.get(estimator, estimator),
        )
    axes[1].set_xticks(x_base)
    axes[1].set_xticklabels([name for name, _keep in regimes], fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean Q MSE / best in regime", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(labelsize=7)
    axes[1].legend(frameon=False, fontsize=5.2, loc="upper left")
    fig.tight_layout(pad=0.35, w_pad=0.7)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_oracle_practical_tuned_comparison(rows: list[dict[str, str]], path: Path) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear"
        and row.get("feature_regime") == "misspecified_affine"
        and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    if not sub_rows:
        return
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    panels = [
        (
            "Practical tuning",
            [
                "linear_standard_fqe",
                "linear_estimated_tikhonov_fqe",
                "linear_estimated_quadratic_moment_cv_fqe",
                "linear_minimax_q_cv_tikhonov",
            ],
        ),
        (
            "Oracle tuning diagnostic",
            [
                "linear_standard_fqe",
                "linear_oracle_raw_fqe",
                "linear_estimated_rbf_oracle_tikhonov_fqe",
                "linear_estimated_quadratic_moment_oracle_fqe",
                "linear_minimax_q_oracle_tikhonov",
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)
    for ax, (title, estimators) in zip(axes, panels):
        for estimator in estimators:
            sub = [row for row in sub_rows if row["estimator"] == estimator]
            if not sub:
                continue
            sub.sort(
                key=lambda row: raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])),
                reverse=True,
            )
            linestyle = "--" if "oracle" in estimator and estimator != "linear_oracle_raw_fqe" else "-"
            linewidth = 1.45 if estimator in {"linear_standard_fqe", "linear_oracle_raw_fqe"} else 1.25
            ax.plot(
                [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub],
                [float(row["target_q_mse"]) for row in sub],
                marker="o",
                linewidth=linewidth,
                linestyle=linestyle,
                markersize=2.8,
                color=METHOD_COLORS.get(estimator),
                label=METHOD_LABELS.get(estimator, estimator),
            )
        ax.axvspan(0.0, 0.23, color="#d9d9d9", alpha=0.28, lw=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("ESS fraction (oracle)", fontsize=8)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if raw_ess_by_shift:
            ess_values = list(raw_ess_by_shift.values())
            x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
            ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
            ax.set_xticks([0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
            ax.set_xticks(sorted(ess_values, reverse=True), minor=True)
            ax.tick_params(axis="x", which="minor", length=2.0)
            top_ticks = [
                tick
                for tick in [0.0, 1.0, 1.5, 1.8, 1.9, 2.0, 2.06]
                if tick in raw_ess_by_shift
            ]
            if top_ticks:
                top = ax.twiny()
                top.set_xlim(ax.get_xlim())
                top.set_xticks([raw_ess_by_shift[tick] for tick in top_ticks])
                top.set_xticklabels([f"{tick:g}" for tick in top_ticks])
                top.set_xlabel(r"behavior-target shift $\Delta$", fontsize=7, labelpad=2)
                top.tick_params(labelsize=6, pad=1)
        ax.legend(frameon=False, fontsize=5.0, loc="best")
    axes[0].set_ylabel("Target reference Q MSE", fontsize=8)
    fig.tight_layout(pad=0.35, w_pad=0.65)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_minimax_tuning_sensitivity(rows: list[dict[str, str]], path: Path) -> None:
    sub_rows = [
        row
        for row in rows
        if row.get("fqe_family") == "linear"
        and row.get("feature_regime") == "misspecified_affine"
        and abs(float(row.get("ratio_gamma", 1.0)) - 1.0) < 1e-12
    ]
    if not sub_rows:
        return
    raw_ess_by_shift = {
        float(row["shift"]): float(row["effective_sample_size_fraction"])
        for row in sub_rows
        if row["estimator"] == "linear_oracle_raw_fqe"
    }
    estimators = [
        "linear_minimax_q_unregularized",
        "linear_minimax_q_eta_0.001",
        "linear_minimax_q_eta_0.01",
        "linear_minimax_q_rbf",
        "linear_minimax_q_eta_1",
        "linear_minimax_q_eta_10",
        "linear_minimax_q_cv_tikhonov",
    ]
    present = [estimator for estimator in estimators if any(row["estimator"] == estimator for row in sub_rows)]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    for estimator in present:
        sub = [row for row in sub_rows if row["estimator"] == estimator]
        sub.sort(key=lambda row: raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])), reverse=True)
        linestyle = "--" if estimator == "linear_minimax_q_cv_tikhonov" else "-"
        linewidth = 1.7 if estimator == "linear_minimax_q_cv_tikhonov" else 1.05
        ax.plot(
            [raw_ess_by_shift.get(float(row["shift"]), float(row["shift"])) for row in sub],
            [float(row["target_q_mse"]) for row in sub],
            marker="o",
            linewidth=linewidth,
            linestyle=linestyle,
            markersize=2.8,
            color=METHOD_COLORS.get(estimator),
            alpha=1.0 if estimator in {"linear_minimax_q_cv_tikhonov", "linear_minimax_q_unregularized"} else 0.72,
            label=METHOD_LABELS.get(estimator, estimator),
        )
    ax.set_xlabel("ESS fraction (oracle)", fontsize=8)
    ax.set_ylabel("Target reference Q MSE", fontsize=8)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    if raw_ess_by_shift:
        ess_values = list(raw_ess_by_shift.values())
        x_pad = 0.015 * max(max(ess_values) - min(ess_values), 1e-6)
        ax.set_xlim(max(ess_values) + x_pad, min(ess_values) - x_pad)
        ax.set_xticks([0.8, 0.6, 0.4, 0.2, 0.1])
        top_ticks = [
            tick
            for tick in [0.0, 1.0, 1.5, 1.8, 1.9, 1.95]
            if tick in raw_ess_by_shift
        ]
        if top_ticks:
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            top.set_xticks([raw_ess_by_shift[tick] for tick in top_ticks])
            top.set_xticklabels([f"{tick:g}" for tick in top_ticks])
            top.set_xlabel(r"behavior-target shift $\Delta$", fontsize=7, labelpad=2)
            top.tick_params(labelsize=6, pad=1)
    ax.legend(frameon=False, fontsize=5.3, ncol=2, loc="best")
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_gamma_ratio_quality(rows: list[dict[str, str]], path: Path) -> None:
    estimators = ["linear_estimated_clipped_fqe", "linear_neural_weighted_clipped_fqe"]
    moderate_rows = [row for row in rows if row["fqe_family"] == "linear" and float(row["shift"]) == 1.1]
    if not moderate_rows:
        moderate_rows = [row for row in rows if row["fqe_family"] == "linear"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for estimator in estimators:
        sub = [row for row in moderate_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["ratio_gamma"]))
        x = [float(row["ratio_gamma"]) for row in sub]
        label = METHOD_LABELS.get(estimator, estimator)
        color = METHOD_COLORS.get(estimator)
        axes[0].plot(x, [float(row["oracle_log_ratio_rmse"]) for row in sub], marker="o", color=color, label=label)
        axes[1].plot(x, [float(row["oracle_estimated_weight_corr"]) for row in sub], marker="o", color=color, label=label)
        axes[2].plot(x, [float(row["target_ratio_feature_calibration_l2"]) for row in sub], marker="o", color=color, label=label)
    axes[0].set_ylabel("Log-ratio RMSE")
    axes[1].set_ylabel("Correlation vs oracle")
    axes[2].set_ylabel("Target feature calibration")
    axes[0].set_yscale("log")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("ratio gamma")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_neural_vs_linear(rows: list[dict[str, str]], path: Path) -> None:
    estimators = [
        "linear_standard_fqe",
        "linear_neural_weighted_clipped_fqe",
        "neural_standard_fqe",
        "neural_neural_weighted_clipped_fqe",
    ]
    sub_rows = [row for row in rows if float(row["shift"]) == 1.1 and row["estimator"] in estimators]
    if not sub_rows:
        sub_rows = [row for row in rows if row["estimator"] in estimators]
    plt.figure(figsize=(7.5, 4.8))
    for estimator in estimators:
        sub = [row for row in sub_rows if row["estimator"] == estimator]
        if not sub:
            continue
        sub.sort(key=lambda row: float(row["ratio_gamma"]))
        plt.plot(
            [float(row["ratio_gamma"]) for row in sub],
            [float(row["policy_value_mse"]) for row in sub],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS.get(estimator),
            label=METHOD_LABELS.get(estimator, estimator),
        )
    plt.xlabel("ratio gamma")
    plt.ylabel("Policy value MSE")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_gamma_summary_table(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "value_gamma",
        "ratio_gamma",
        "reference_distribution",
        "shift",
        "sample_size",
        "feature_regime",
        "fqe_family",
        "estimator",
        "target_q_mse",
        "behavior_q_mse",
        "behavior_target_action_q_mse",
        "policy_value_mse",
        "effective_sample_size_fraction",
        "weight_q95",
        "weight_q99",
        "weight_max",
        "oracle_log_ratio_rmse",
        "oracle_estimated_weight_corr",
        "target_ratio_feature_calibration_l2",
        "target_fqe_feature_calibration_l2",
        "unstable_run_fraction",
        "unstable_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                float(item["ratio_gamma"]),
                float(item["shift"]),
                item["fqe_family"],
                item["estimator"],
            ),
        ):
            writer.writerow({column: row[column] for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot controlled discounted-occupancy FQE results.")
    parser.add_argument("--results-root", type=Path, default=Path("FQE_neurips/results"))
    args = parser.parse_args()

    figures_dir = args.results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary_rows = _load_summary(args.results_root)
    except FileNotFoundError:
        summary_rows = []
    if summary_rows:
        main_slice = _select_main_slice(summary_rows, args.results_root)
        _plot_metric(
            main_slice,
            metric="target_q_mse",
            ylabel="Target occupancy Q MSE",
            path=figures_dir / "target_occupancy_q_mse_vs_shift.png",
            estimators=[
                "standard_fqe",
                "oracle_weighted_fqe",
                "oracle_weighted_fqe_clipped",
                "estimated_weighted_fqe_clip99_ess40",
            ],
        )
        _plot_metric(
            main_slice,
            metric="behavior_q_mse",
            ylabel="Behavior occupancy Q MSE",
            path=figures_dir / "behavior_occupancy_q_mse_vs_shift.png",
            estimators=[
                "standard_fqe",
                "oracle_weighted_fqe",
                "oracle_weighted_fqe_clipped",
                "estimated_weighted_fqe_clip99_ess40",
            ],
        )
        _plot_metric(
            main_slice,
            metric="policy_value_mse",
            ylabel="Policy value MSE",
            path=figures_dir / "policy_value_mse_vs_shift.png",
            estimators=[
                "standard_fqe",
                "oracle_weighted_fqe",
                "oracle_weighted_fqe_clipped",
                "estimated_weighted_fqe_clip99_ess40",
            ],
        )
        _plot_weight_diagnostics(
            main_slice,
            figures_dir / "weight_diagnostics_vs_shift.png",
        )
        _plot_ratio_quality(
            main_slice,
            figures_dir / "ratio_quality_and_clipping_sensitivity_vs_shift.png",
        )
        _write_summary_table(main_slice, args.results_root / "estimator_shift_summary.csv")
    gamma_rows = _load_gamma_summary(args.results_root)
    if gamma_rows:
        gamma_slice = _gamma_main_slice(gamma_rows)
        main_linear_estimators = [
            "linear_standard_fqe",
            "linear_oracle_raw_fqe",
            "linear_estimated_quadratic_moment_fqe",
            "linear_estimated_quadratic_moment_cv_fqe",
            "linear_estimated_quadratic_moment_oracle_fqe",
            "linear_minimax_q_cv_tikhonov",
            "linear_minimax_q_rbf",
            "linear_estimated_rbf_fixed_tikhonov_fqe",
            "linear_estimated_tikhonov_fqe",
        ]
        unique_ratio_gammas = sorted({float(row["ratio_gamma"]) for row in gamma_slice})
        if len(unique_ratio_gammas) == 1 and abs(unique_ratio_gammas[0] - 1.0) < 1e-12:
            _plot_stationary_shift_metric(
                gamma_slice,
                metric="target_q_mse",
                ylabel="Target reference Q MSE",
                path=figures_dir / "gamma_sweep_target_reference_q_mse.png",
                estimators=main_linear_estimators,
                fqe_family="linear",
            )
            _plot_stationary_shift_metric(
                gamma_slice,
                metric="target_q_mse",
                ylabel="Target reference Q MSE",
                path=figures_dir / "gamma_sweep_stabilization_comparison_q_mse.png",
                estimators=STATIONARY_PANEL_ESTIMATORS,
                fqe_family="linear",
            )
            _plot_stationary_shift_metric(
                gamma_slice,
                metric="policy_value_mse",
                ylabel="Policy value MSE",
                path=figures_dir / "gamma_sweep_policy_value_mse.png",
                estimators=main_linear_estimators,
                fqe_family="linear",
            )
            _plot_stationary_shift_weight_quality(
                gamma_slice,
                figures_dir / "gamma_sweep_weight_quality.png",
                estimators=[
                    "linear_oracle_raw_fqe",
                    "linear_estimated_quadratic_moment_fqe",
                    "linear_estimated_quadratic_moment_cv_fqe",
                    "linear_estimated_quadratic_moment_oracle_fqe",
                    "linear_estimated_rbf_fixed_tikhonov_fqe",
                    "linear_estimated_tikhonov_fqe",
                ],
            )
            _plot_stationary_shift_weight_quality(
                gamma_slice,
                figures_dir / "gamma_sweep_weight_quality_all_stabilizers.png",
            )
            _plot_stationary_norm_diagnostics(
                gamma_slice,
                figures_dir / "gamma_sweep_norm_diagnostic_q_mse.png",
            )
            _plot_normalized_tikhonov_sensitivity(
                gamma_slice,
                figures_dir / "gamma_sweep_normalized_tikhonov_sensitivity.png",
            )
            _plot_minimax_vs_weighting_overlap_regimes(
                gamma_slice,
                figures_dir / "minimax_vs_weighting_overlap_regimes.png",
            )
            _plot_oracle_practical_tuned_comparison(
                gamma_slice,
                figures_dir / "oracle_practical_tuned_comparison.png",
            )
            _plot_minimax_tuning_sensitivity(
                gamma_slice,
                figures_dir / "minimax_tuning_sensitivity.png",
            )
        else:
            _plot_gamma_metric_by_shift(
                gamma_slice,
                metric="target_q_mse",
                ylabel="Target reference Q MSE",
                path=figures_dir / "gamma_sweep_target_reference_q_mse.png",
                estimators=main_linear_estimators,
                fqe_family="linear",
            )
            _plot_gamma_metric_by_shift(
                gamma_slice,
                metric="policy_value_mse",
                ylabel="Policy value MSE",
                path=figures_dir / "gamma_sweep_policy_value_mse.png",
                estimators=main_linear_estimators,
                fqe_family="linear",
            )
            _plot_gamma_weight_quality(
                gamma_slice,
                figures_dir / "gamma_sweep_weight_quality.png",
            )
            _plot_gamma_ratio_quality(
                gamma_slice,
                figures_dir / "gamma_sweep_ratio_quality.png",
            )
        if any(row["fqe_family"] == "neural" or "neural" in row["estimator"] for row in gamma_slice):
            _plot_neural_vs_linear(
                gamma_slice,
                figures_dir / "gamma_sweep_linear_vs_neural_fqe.png",
            )
        _write_gamma_summary_table(gamma_slice, args.results_root / "gamma_sweep_estimator_summary.csv")


if __name__ == "__main__":
    main()
