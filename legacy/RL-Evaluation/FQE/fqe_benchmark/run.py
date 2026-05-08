from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fqe_benchmark.runner import run_benchmark
from fqe_benchmark.types import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FQE estimator benchmarks.")
    parser.add_argument("--stage", choices=("smoke", "core", "full"), default="smoke")
    parser.add_argument("--output-root", default="outputs/fqe_benchmark")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--policy-shifts", nargs="*", type=float, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig.for_stage(args.stage, output_root=Path(args.output_root))
    updates = {}
    for name, value in (
        ("datasets", args.datasets),
        ("estimators", args.estimators),
        ("seeds", args.seeds),
        ("sample_sizes", args.sample_sizes),
        ("gammas", args.gammas),
        ("policy_shifts", args.policy_shifts),
    ):
        if value is not None:
            updates[name] = tuple(value)
    if args.no_plots:
        updates["output_plots"] = False
    if args.fail_fast:
        updates["fail_fast"] = True
    if updates:
        config = replace(config, **updates)
    result = run_benchmark(config)
    print(f"Wrote results: {result.results_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote diagnostics: {result.diagnostics_path}")
    print(f"Wrote manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
