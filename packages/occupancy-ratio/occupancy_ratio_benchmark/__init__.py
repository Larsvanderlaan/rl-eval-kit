"""Benchmarks for discounted occupancy density-ratio estimators."""

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "rltools-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.fori_cv import (
    FORICVCandidate,
    FORICVResult,
    compute_weight_diagnostics,
    fit_fori_cv_candidate,
    plot_cv_diagnostics,
    run_fori_cv_benchmark,
    score_fixed_point_residual,
    score_moment_balance,
    score_value_grouped_moment_balance,
    summarize_cv_results,
)
from occupancy_ratio_benchmark.runner import BenchmarkRunResult, run_benchmark

__all__ = [
    "BenchmarkRunResult",
    "FORICVCandidate",
    "FORICVResult",
    "OccupancyRatioBenchmarkConfig",
    "compute_weight_diagnostics",
    "fit_fori_cv_candidate",
    "plot_cv_diagnostics",
    "run_benchmark",
    "run_fori_cv_benchmark",
    "score_fixed_point_residual",
    "score_moment_balance",
    "score_value_grouped_moment_balance",
    "summarize_cv_results",
]
