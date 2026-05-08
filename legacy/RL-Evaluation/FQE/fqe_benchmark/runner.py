from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import os
import platform
import traceback
from typing import Any

from fqe_benchmark.adapters import estimator_registry
from fqe_benchmark.data import make_datasets
from fqe_benchmark.io import write_csv, write_json
from fqe_benchmark.metrics import evaluate_fitted_estimator, summarize_rows
from fqe_benchmark.plots import write_plots
from fqe_benchmark.types import BenchmarkConfig, BenchmarkDataset, BenchmarkRunResult, EstimatorPreflight


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRunResult:
    """Run the configured FQE benchmark suite and write outputs."""
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_cache_dir(output_dir)
    registry = estimator_registry()
    datasets = make_datasets(config)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"preflight": {}, "failures": []}

    for dataset in datasets:
        for estimator_name in config.estimators:
            adapter = registry.get(estimator_name)
            if adapter is None:
                rows.append(_skip_row(config, dataset, estimator_name, EstimatorPreflight("unsupported_setting", "unknown estimator")))
                continue
            preflight = adapter.preflight(config, dataset)
            diagnostics["preflight"].setdefault(estimator_name, {})
            diagnostics["preflight"][estimator_name][dataset.name] = asdict(preflight)
            if not preflight.available:
                rows.append(_skip_row(config, dataset, estimator_name, preflight))
                continue
            try:
                fitted = adapter.fit(dataset, config, seed=dataset.seed)
                row = _base_row(config, dataset, estimator_name)
                row["status"] = "ok"
                row.update(evaluate_fitted_estimator(dataset, fitted))
                row.update(_compact_diagnostics(fitted.diagnostics))
                rows.append(row)
            except Exception as exc:
                if config.fail_fast:
                    raise
                rows.append(_error_row(config, dataset, estimator_name, exc))
                diagnostics["failures"].append(
                    {
                        "dataset": dataset.name,
                        "estimator": estimator_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )

    summary_rows = summarize_rows(rows)
    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    diagnostics_path = output_dir / "diagnostics.json"
    manifest_path = output_dir / "manifest.json"
    write_csv(results_path, rows)
    write_csv(summary_path, summary_rows)
    write_json(diagnostics_path, diagnostics)
    write_json(manifest_path, _manifest(config))
    plot_status = write_plots(output_dir, rows) if config.output_plots else "plotting disabled"
    diagnostics["plot_status"] = plot_status
    write_json(diagnostics_path, diagnostics)
    return BenchmarkRunResult(
        output_dir=output_dir,
        results_path=results_path,
        summary_path=summary_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        rows=rows,
        summary_rows=summary_rows,
    )


def _base_row(config: BenchmarkConfig, dataset: BenchmarkDataset, estimator: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": config.stage,
        "dataset": dataset.name,
        "domain": dataset.domain,
        "estimator": estimator,
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.n),
        "policy_shift": float(dataset.metadata.get("policy_shift", 0.0)),
    }
    row.update(dataset.metadata)
    return row


def _skip_row(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    estimator: str,
    preflight: EstimatorPreflight,
) -> dict[str, Any]:
    row = _base_row(config, dataset, estimator)
    row.update({"status": preflight.status, "skip_reason": preflight.reason})
    return row


def _error_row(config: BenchmarkConfig, dataset: BenchmarkDataset, estimator: str, exc: Exception) -> dict[str, Any]:
    row = _base_row(config, dataset, estimator)
    row.update({"status": "error", "skip_reason": f"{type(exc).__name__}: {exc}"})
    return row


def _compact_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[f"diag_{key}"] = value
        elif isinstance(value, dict):
            compact[f"diag_{key}"] = str(value)
    return compact


def _manifest(config: BenchmarkConfig) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "lightgbm", "torch", "matplotlib", "d3rlpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _ensure_runtime_cache_dir(output_dir) -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = output_dir / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
