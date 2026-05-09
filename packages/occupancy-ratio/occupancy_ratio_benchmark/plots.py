from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> str:
    """Write compact benchmark plots when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"plotting skipped: {type(exc).__name__}: {exc}"

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "plotting skipped: no successful rows"

    settings = sorted({str(row["setting"]) for row in ok_rows})
    estimators = sorted({str(row["estimator"]) for row in ok_rows if row["estimator"] != "oracle"})
    x = np.arange(len(settings))
    width = 0.8 / max(len(estimators), 1)
    written = []

    metrics = (
        ("ratio_normalized_l1", "Normalized ratio L1", "ratio_normalized_l1.png"),
        ("ratio_rel_mse", "Relative MSE", "ratio_rel_mse.png"),
        ("log_ratio_rmse", "Log-ratio RMSE", "ratio_log_rmse.png"),
        ("ope_value_abs_error", "OPE absolute error", "ope_value_abs_error.png"),
        ("effective_sample_size_fraction", "ESS fraction", "ess_fraction.png"),
        ("ess_fraction_abs_error_to_truth", "ESS fraction error to truth", "ess_fraction_error_to_truth.png"),
        ("weight_q99_ratio_to_truth", "p99 weight ratio to truth", "weight_q99_ratio_to_truth.png"),
        ("clipping_fraction", "Clipping fraction", "clipping_fraction.png"),
        ("source_state_ratio_ess_fraction", "Source-state ESS fraction", "source_state_ess_fraction.png"),
        ("weight_q99", "p99 weight", "weight_q99.png"),
        ("weight_max", "max weight", "weight_max.png"),
        ("runtime_sec", "Runtime seconds", "runtime_sec.png"),
    )
    for metric_name, ylabel, filename in metrics:
        metric_rows = [row for row in ok_rows if metric_name in row]
        if not metric_rows:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, estimator in enumerate(estimators):
            vals = []
            for setting in settings:
                metric = [
                    float(row[metric_name])
                    for row in metric_rows
                    if row["setting"] == setting
                    and row["estimator"] == estimator
                    and np.isfinite(float(row[metric_name]))
                ]
                vals.append(float(np.mean(metric)) if metric else np.nan)
            ax.bar(x + idx * width, vals, width=width, label=estimator)
        ax.set_xticks(x + width * max(len(estimators) - 1, 0) / 2)
        ax.set_xticklabels(settings, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Occupancy Ratio Benchmark")
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path.name)
    failure_plot = _write_failure_rate_plot(output_dir, rows, plt)
    if failure_plot:
        written.append(failure_plot)
    dice_plot = _write_neural_vs_dice_plot(output_dir, rows, plt)
    if dice_plot:
        written.append(dice_plot)
    if not written:
        return "plotting skipped: no recognized metrics"
    return "wrote " + ", ".join(written)


def _write_failure_rate_plot(output_dir: Path, rows: list[dict[str, Any]], plt) -> str:
    estimators = sorted({str(row.get("estimator", "")) for row in rows if row.get("estimator")})
    if not estimators:
        return ""
    failure_rates = []
    for estimator in estimators:
        group = [row for row in rows if str(row.get("estimator", "")) == estimator]
        failures = [row for row in group if row.get("status") in {"error", "timeout"}]
        failure_rates.append(len(failures) / max(len(group), 1))
    if not any(rate > 0.0 for rate in failure_rates):
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(estimators)), failure_rates)
    ax.set_xticks(np.arange(len(estimators)))
    ax.set_xticklabels(estimators, rotation=25, ha="right")
    ax.set_ylabel("Failure/timeout rate")
    ax.set_title("Benchmark Stability")
    fig.tight_layout()
    path = output_dir / "failure_timeout_rate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _write_neural_vs_dice_plot(output_dir: Path, rows: list[dict[str, Any]], plt) -> str:
    by_cell_estimator: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        cell = (
            str(row.get("setting", "")),
            str(row.get("dataset_variant", "")),
            str(row.get("policy_shift", "")),
            str(row.get("gamma", "")),
            str(row.get("sample_size", "")),
            str(row.get("seed", "")),
        )
        by_cell_estimator.setdefault(cell, {})[str(row.get("estimator", ""))] = row
    ratios: dict[str, list[float]] = {}
    for group in by_cell_estimator.values():
        dice = group.get("google_dualdice_neural")
        if dice is None:
            continue
        dice_error = _finite_float(dice.get("ope_value_abs_error"))
        if not np.isfinite(dice_error):
            continue
        for estimator, row in group.items():
            if not estimator.startswith("neural_network"):
                continue
            error = _finite_float(row.get("ope_value_abs_error"))
            if np.isfinite(error):
                ratios.setdefault(estimator, []).append(error / max(dice_error, 1e-12))
    ratios = {name: values for name, values in ratios.items() if values}
    if not ratios:
        return ""
    estimators = sorted(ratios)
    values = [float(np.median(ratios[name])) for name in estimators]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(estimators)), values)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(estimators)))
    ax.set_xticklabels(estimators, rotation=25, ha="right")
    ax.set_ylabel("Median OPE error ratio vs Google DualDICE")
    ax.set_title("Neural FORI vs Google DualDICE")
    fig.tight_layout()
    path = output_dir / "neural_vs_dualdice_ope_ratio.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def _finite_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")
