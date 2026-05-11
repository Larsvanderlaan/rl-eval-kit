from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
)
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "rltools-matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

from occupancy_ratio_benchmark.fori_cv import FORICVCandidate, run_fori_cv_benchmark
from occupancy_ratio_benchmark.gym_control import GYM_CONTROL_SETTINGS, make_gym_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact leakage-safe FORI CV benchmark on Gym control suites.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fori_cv_gym_smoke"))
    parser.add_argument("--settings", nargs="+", default=sorted(GYM_CONTROL_SETTINGS))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--target-rollouts", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gradient-steps", type=int, default=6)
    parser.add_argument("--nuisance-steps", type=int, default=250)
    parser.add_argument("--mcmc-samples", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_summary: list[dict] = []
    for setting in args.settings:
        dataset = make_gym_control_dataset(
            setting=str(setting),
            gamma=float(args.gamma),
            sample_size=int(args.sample_size),
            seed=int(args.seed),
            target_value_rollouts=int(args.target_rollouts),
        )
        candidates = _neural_candidates(args)
        result = run_fori_cv_benchmark(
            dataset,
            candidates,
            k_folds=int(args.folds),
            seed=int(args.seed) + 31,
            rff_features=16,
            output_dir=args.output_dir / str(setting),
            keep_fold_weights=True,
        )
        for row in result.rows:
            row = dict(row)
            row["setting"] = str(setting)
            all_rows.append(_jsonable(row))
        for row in result.summary:
            row = dict(row)
            row["setting"] = str(setting)
            row["selected_candidate_for_setting"] = str(result.selected_candidate)
            all_summary.append(_jsonable(row))

    _write_csv(args.output_dir / "fold_rows.csv", all_rows)
    _write_csv(args.output_dir / "summary.csv", all_summary)
    with (args.output_dir / "run_config.json").open("w") as fh:
        json.dump(vars(args), fh, default=str, indent=2)


def _neural_candidates(args: argparse.Namespace) -> list[FORICVCandidate]:
    stable_occ = NeuralOccupancyRegressionConfig.stable_defaults(
        num_iterations=int(args.iterations),
        gradient_steps_per_iteration=int(args.gradient_steps),
        mcmc_samples=int(args.mcmc_samples),
        validation_fraction=0.2,
        patience=8,
        show_progress=False,
        batch_size=512,
    )
    stable_nuisance = dict(
        max_steps=int(args.nuisance_steps),
        validation_fraction=0.2,
        patience=10,
        batch_size=512,
    )
    parity_occ = NeuralOccupancyRegressionConfig.stable_defaults(
        num_iterations=int(args.iterations),
        gradient_steps_per_iteration=int(args.gradient_steps),
        mcmc_samples=int(args.mcmc_samples),
        validation_fraction=0.2,
        patience=8,
        show_progress=False,
        batch_size=512,
        hidden_dims=(256, 256),
        activation="relu",
    )
    parity_nuisance = dict(stable_nuisance, hidden_dims=(256, 256), activation="relu")
    return [
        FORICVCandidate(
            name="neural_stable",
            family="neural",
            occupancy=stable_occ,
            action_ratio=NeuralActionRatioConfig.stable_defaults(**stable_nuisance),
            source_state_ratio=NeuralSourceStateRatioConfig.stable_defaults(**stable_nuisance),
            transition_ratio=NeuralTransitionRatioConfig.stable_defaults(
                **stable_nuisance,
                permutation_samples=4,
            ),
        ),
        FORICVCandidate(
            name="neural_google_parity",
            family="neural",
            occupancy=parity_occ,
            action_ratio=NeuralActionRatioConfig.stable_defaults(**parity_nuisance),
            source_state_ratio=NeuralSourceStateRatioConfig.stable_defaults(**parity_nuisance),
            transition_ratio=NeuralTransitionRatioConfig.stable_defaults(
                **parity_nuisance,
                permutation_samples=4,
            ),
        ),
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key in {"weights", "true_ratio"}:
            continue
        try:
            json.dumps(value)
            out[key] = value
        except TypeError:
            out[key] = str(value)
    return out


if __name__ == "__main__":
    main()
