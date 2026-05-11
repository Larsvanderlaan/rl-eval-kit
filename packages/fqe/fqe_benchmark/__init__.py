"""Benchmark suite for FQE estimators."""

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "rltools-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

from fqe_benchmark.runner import run_benchmark
from fqe_benchmark.types import BenchmarkConfig, BenchmarkDataset, BenchmarkRunResult

__all__ = ["BenchmarkConfig", "BenchmarkDataset", "BenchmarkRunResult", "run_benchmark"]
