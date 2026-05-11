from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_plots(output_dir: Path, rows: list[dict[str, Any]], histories: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Write simple manuscript-ready diagnostic PDFs when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:
        return [f"plots skipped: matplotlib unavailable ({exc})"]

    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    metric_rows = [row for row in ok_rows if _finite(row.get("ratio_l1_nu"))]
    if metric_rows:
        labels = [str(row["estimator"]) for row in metric_rows]
        values = [float(row["ratio_l1_nu"]) for row in metric_rows]
        fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(labels)), 3.0))
        ax.bar(np.arange(len(labels)), values, color="#456990")
        ax.set_ylabel(r"$L^1(\nu)$ ratio error")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Exact occupancy-ratio error")
        fig.tight_layout()
        path = figure_dir / "estimator_ratio_l1.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    gamma_rows = [row for row in metric_rows if _finite(row.get("gamma"))]
    if gamma_rows:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for estimator in sorted({str(row["estimator"]) for row in gamma_rows}):
            group = [row for row in gamma_rows if str(row["estimator"]) == estimator]
            by_gamma: dict[float, list[float]] = {}
            for row in group:
                by_gamma.setdefault(float(row["gamma"]), []).append(float(row["ratio_l1_nu"]))
            xs = sorted(by_gamma)
            ys = [float(np.mean(by_gamma[x])) for x in xs]
            ax.plot(xs, ys, marker="o", label=estimator)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$L^1(\nu)$ ratio error")
        ax.set_title("Discount degradation")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figure_dir / "gamma_degradation.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    tail_rows = [row for row in ok_rows if _finite(row.get("effective_sample_size_fraction"))]
    if tail_rows:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        xs = [float(row.get("effective_sample_size_fraction")) for row in tail_rows]
        ys = [float(row.get("ratio_l1_nu", row.get("ratio_l1_empirical", np.nan))) for row in tail_rows]
        labels = [str(row["estimator"]) for row in tail_rows]
        ax.scatter(xs, ys, color="#ef767a")
        for x, y, label in zip(xs, ys, labels):
            if np.isfinite(y):
                ax.annotate(label, (x, y), fontsize=6, alpha=0.75)
        ax.set_xlabel("ESS fraction")
        ax.set_ylabel("Ratio error")
        ax.set_title("Tail stability versus error")
        fig.tight_layout()
        path = figure_dir / "ess_vs_error.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    curve_histories = {
        key: history
        for key, history in histories.items()
        if history and any("ratio_l1_nu" in row for row in history)
    }
    if curve_histories:
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for key, history in sorted(curve_histories.items()):
            ys = [float(row["ratio_l1_nu"]) for row in history if _finite(row.get("ratio_l1_nu"))]
            if ys:
                ax.plot(np.arange(1, len(ys) + 1), ys, label=key)
        ax.set_xlabel("FORI iteration")
        ax.set_ylabel(r"$L^1(\nu)$ ratio error")
        ax.set_title("Population iteration curves")
        ax.legend(fontsize=6)
        fig.tight_layout()
        path = figure_dir / "iteration_curves.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    return written


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
