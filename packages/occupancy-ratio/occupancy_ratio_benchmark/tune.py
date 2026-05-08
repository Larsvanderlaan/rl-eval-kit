from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.runner import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run occupancy-ratio tuning sweeps.")
    parser.add_argument("--output-root", default="outputs/occupancy_ratio_benchmark_tuning")
    parser.add_argument("--settings", nargs="*", default=["linear_gaussian"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[500])
    parser.add_argument("--gammas", nargs="*", type=float, default=[0.5, 0.9])
    parser.add_argument("--huber-delta-scales", nargs="*", type=float, default=[1.345, 2.5, 5.0, 10.0])
    parser.add_argument("--include-squared", action="store_true")
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--boosted-num-iterations", type=int, default=40)
    parser.add_argument("--boosted-mcmc-samples", type=int, default=32)
    parser.add_argument("--neural-num-iterations", type=int, default=20)
    parser.add_argument("--neural-mcmc-samples", type=int, default=16)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    sweep_rows: list[dict[str, Any]] = []

    if args.include_squared:
        config = _base_config(args, output_root / "squared", losses=("squared",))
        result = run_benchmark(config)
        sweep_rows.extend(_tag_rows(result.rows, tuning_name="squared", huber_delta_scale=None))

    for scale in args.huber_delta_scales:
        name = f"huber_scale_{str(scale).replace('.', 'p')}"
        config = _base_config(args, output_root / name, losses=("huber",), huber_delta_scale=float(scale))
        result = run_benchmark(config)
        sweep_rows.extend(_tag_rows(result.rows, tuning_name=name, huber_delta_scale=float(scale)))

    summary_path = output_root / "tuning_results.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in sweep_rows for key in row})
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            writer.writerow(row)
    print(f"Wrote tuning results: {summary_path}")


def _base_config(
    args: argparse.Namespace,
    output_root: Path,
    *,
    losses: tuple[str, ...],
    huber_delta_scale: float = 1.345,
) -> OccupancyRatioBenchmarkConfig:
    return OccupancyRatioBenchmarkConfig(
        stage="smoke",
        output_root=output_root,
        settings=tuple(args.settings),
        estimators=("oracle", "boosted_tree", "neural_network") if args.include_neural else ("oracle", "boosted_tree"),
        include_google_dual_dice=False,
        seeds=tuple(args.seeds),
        sample_sizes=tuple(args.sample_sizes),
        gammas=tuple(args.gammas),
        boosted_losses=losses,
        boosted_num_iterations=int(args.boosted_num_iterations),
        boosted_mcmc_samples=int(args.boosted_mcmc_samples),
        boosted_batch_size=512,
        neural_losses=losses,
        neural_stabilization_presets=() if args.include_neural else ("huber_projection_damping_transition_norm",),
        neural_num_iterations=int(args.neural_num_iterations),
        neural_mcmc_samples=int(args.neural_mcmc_samples),
        neural_batch_size=512,
        huber_delta_scale=float(huber_delta_scale),
        write_plots=not args.no_plots,
    )


def _tag_rows(rows: list[dict[str, Any]], *, tuning_name: str, huber_delta_scale: float | None) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        new_row = dict(row)
        new_row["tuning_name"] = tuning_name
        new_row["tuning_huber_delta_scale"] = "" if huber_delta_scale is None else float(huber_delta_scale)
        tagged.append(new_row)
    return tagged


if __name__ == "__main__":
    main()
