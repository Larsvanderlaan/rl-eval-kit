"""Benchmarks for discounted occupancy density-ratio estimators."""

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "rltools-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.runner import BenchmarkRunResult, run_benchmark

__all__ = [
    "BenchmarkRunResult",
    "OccupancyRatioBenchmarkConfig",
    "run_benchmark",
]
