from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return f"plotting skipped: {type(exc).__name__}: {exc}"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "plotting skipped: no ok rows"
    _bar_plot(
        output_dir / "value_error.png",
        ok_rows,
        metric="policy_value_absolute_error",
        ylabel="Policy value absolute error",
        plt=plt,
    )
    _bar_plot(output_dir / "q_mse.png", ok_rows, metric="target_q_mse", ylabel="Target Q MSE", plt=plt, log=True)
    _bar_plot(output_dir / "runtime.png", ok_rows, metric="runtime_sec", ylabel="Runtime (sec)", plt=plt)
    return "plots written"


def _bar_plot(path: Path, rows: list[dict[str, Any]], *, metric: str, ylabel: str, plt, log: bool = False) -> None:
    values_by_estimator: dict[str, list[float]] = {}
    for row in rows:
        if metric in row and row[metric] not in {"", None}:
            values_by_estimator.setdefault(str(row["estimator"]), []).append(float(row[metric]))
    if not values_by_estimator:
        return
    estimators = sorted(values_by_estimator)
    values = [float(np.mean(values_by_estimator[name])) for name in estimators]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(estimators)), 4))
    ax.bar(np.arange(len(estimators)), values)
    ax.set_xticks(np.arange(len(estimators)))
    ax.set_xticklabels(estimators, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
