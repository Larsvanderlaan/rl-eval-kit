from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from occupancy_ratio.tuning import (
    OccupancySearchSpace,
    OccupancyTuningConfig,
    tune_occupancy_ratio_auto,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
)
from occupancy_ratio_benchmark.gym_control import make_gym_control_dataset
from occupancy_ratio_benchmark.estimators import _stabilize_torch_runtime


SETTINGS = ("gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper")


def main() -> None:
    _stabilize_torch_runtime()
    parser = argparse.ArgumentParser(description="Compact Gym probe for neural AutoML selector budgets.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/occupancy_ratio_automl_gym_selector_probe"))
    parser.add_argument("--settings", nargs="*", default=list(SETTINGS))
    parser.add_argument("--budgets", nargs="*", default=["fast", "balanced"], choices=("fast", "balanced"))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--gradient-steps", type=int, default=3)
    parser.add_argument("--mcmc-samples", type=int, default=8)
    parser.add_argument("--nuisance-steps", type=int, default=160)
    parser.add_argument("--target-rollouts", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "selected_rows.csv"
    candidate_path = args.output_dir / "candidate_rows.csv"
    completed = _completed(result_path) if not args.rerun else set()

    for setting in args.settings:
        for seed in args.seeds:
            dataset = make_gym_control_dataset(
                setting=str(setting),
                gamma=float(args.gamma),
                sample_size=int(args.sample_size),
                seed=int(seed),
                target_value_rollouts=int(args.target_rollouts),
            )
            for budget in args.budgets:
                key = (str(setting), int(seed), str(budget))
                if key in completed:
                    continue
                print(f"cell setting={setting} seed={seed} budget={budget}", flush=True)
                started = time.perf_counter()
                try:
                    tuned = tune_occupancy_ratio_auto(
                        states=dataset.states,
                        actions=dataset.actions,
                        next_states=dataset.next_states,
                        target_actions=dataset.target_actions,
                        target_next_actions=dataset.next_target_actions,
                        gamma=float(dataset.gamma),
                        initial_states=dataset.initial_states,
                        initial_actions=dataset.initial_actions,
                        initial_weights=dataset.initial_weights,
                        rewards=dataset.rewards,
                        search_space=_search_space(args),
                        config=OccupancyTuningConfig(
                            budget=str(budget),
                            cv_folds=int(args.cv_folds),
                            max_candidates=8 if budget == "fast" else 16,
                            promotion_candidates=2 if budget == "fast" else 4,
                            seed=int(seed + 70_001),
                        ),
                    )
                    weights = tuned.model.predict_state_action_ratio(dataset.states, dataset.actions)
                    value = float(np.mean(np.asarray(weights) * np.asarray(dataset.rewards)))
                    target = float(dataset.metadata["target_policy_value"])
                    selected = {
                        "setting": setting,
                        "seed": seed,
                        "sample_size": int(args.sample_size),
                        "gamma": float(args.gamma),
                        "budget": budget,
                        "status": "ok",
                        "selected_candidate_id": tuned.selected_candidate_id,
                        "selected_label": _label(tuned.selected_candidate_id, has_initial_states=True),
                        "ope_value_estimate": value,
                        "target_policy_value": target,
                        "ope_value_abs_error": abs(value - target),
                        "runtime_sec": time.perf_counter() - started,
                        "error": "",
                    }
                    _append_row(result_path, selected)
                    for row in tuned.candidate_rows():
                        out = {
                            "setting": setting,
                            "seed": seed,
                            "sample_size": int(args.sample_size),
                            "gamma": float(args.gamma),
                            "budget": budget,
                            "candidate_id": row.get("candidate_id", ""),
                            "candidate_label": _label(row.get("candidate_id", ""), has_initial_states=True),
                            "budget_stage": row.get("budget_stage", ""),
                            "selected": row.get("selected", ""),
                            "promoted": row.get("promoted", ""),
                            "score": row.get("score", ""),
                            "runtime_sec": row.get("runtime_sec", ""),
                        }
                        for key2, value2 in row.items():
                            if str(key2).startswith("metric_"):
                                out[str(key2)] = value2
                        _append_row(candidate_path, out)
                except Exception as exc:  # noqa: BLE001 - benchmark rows should preserve failures.
                    _append_row(
                        result_path,
                        {
                            "setting": setting,
                            "seed": seed,
                            "sample_size": int(args.sample_size),
                            "gamma": float(args.gamma),
                            "budget": budget,
                            "status": "failed",
                            "selected_candidate_id": "",
                            "selected_label": "",
                            "ope_value_estimate": "",
                            "target_policy_value": dataset.metadata.get("target_policy_value", ""),
                            "ope_value_abs_error": "",
                            "runtime_sec": time.perf_counter() - started,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )


def _search_space(args: argparse.Namespace) -> OccupancySearchSpace:
    dims = (int(args.hidden_dim), int(args.hidden_dim))
    occ = NeuralOccupancyRegressionConfig(
        num_iterations=int(args.iterations),
        gradient_steps_per_iteration=int(args.gradient_steps),
        mcmc_samples=int(args.mcmc_samples),
        batch_size=128,
        hidden_dims=dims,
        activation="silu",
        learning_rate=1e-3,
        fixed_point_damping=0.5,
        occupancy_ratio_max=50.0,
        early_stopping=True,
        show_progress=False,
    )
    nuisance = dict(
        max_steps=int(args.nuisance_steps),
        batch_size=128,
        hidden_dims=dims,
        activation="silu",
        learning_rate=1e-3,
        validation_fraction=0.2,
        patience=8,
        prediction_max=50.0,
    )
    trans = NeuralTransitionRatioConfig(
        **nuisance,
        permutation_samples=8,
    )
    return OccupancySearchSpace(
        neural_occupancy=occ,
        neural_action_ratio=NeuralActionRatioConfig(**nuisance),
        neural_source_state_ratio=NeuralSourceStateRatioConfig(**nuisance),
        neural_transition_ratio=trans,
    )


def _label(candidate_id: Any, *, has_initial_states: bool) -> str:
    try:
        idx = int(str(candidate_id).rsplit("_", 1)[-1])
    except Exception:
        return str(candidate_id)
    labels = [
        "stable",
        "google_parity",
        "relaxed_tail",
        "logistic_nuisance",
    ]
    if has_initial_states:
        labels.append("factored_source")
    labels.extend(
        [
            "small_width",
            "large_width",
            "tight_cap",
            "tight_nuisance_cap",
            "loose_nuisance_cap",
            "tight_cap_logistic_nuisance",
        ]
    )
    return labels[idx] if idx < len(labels) else f"candidate_{idx:03d}"


def _completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    out: set[tuple[str, int, str]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                out.add((str(row.get("setting")), int(row.get("seed", 0)), str(row.get("budget"))))
    return out


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.exists()
    fieldnames = list(row)
    if existing:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    rows = []
    if existing:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({key: _clean(row.get(key, "")) for key in fieldnames})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and (math.isnan(float(value)) or math.isinf(float(value))):
        return ""
    return value


if __name__ == "__main__":
    main()
