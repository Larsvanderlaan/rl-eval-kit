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
        ("ratio_rel_mse", "Relative MSE", "ratio_rel_mse.png"),
        ("log_ratio_rmse", "Log-ratio RMSE", "ratio_log_rmse.png"),
        ("ope_value_abs_error", "OPE absolute error", "ope_value_abs_error.png"),
        ("effective_sample_size_fraction", "ESS fraction", "ess_fraction.png"),
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
    if not written:
        return "plotting skipped: no recognized metrics"
    return "wrote " + ", ".join(written)
