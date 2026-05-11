from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time
from argparse import Namespace
from typing import Any, Iterable

import numpy as np


for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "fqe"))
sys.path.insert(0, str(ROOT / "packages" / "occupancy-ratio"))

from fqe import FQESearchSpace, FQETuningConfig, NeuralFQEConfig, tune_fqe
from fqe_benchmark.data import make_gym_control_dataset as make_fqe_gym_dataset
from occupancy_ratio import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    OccupancySearchSpace,
    OccupancyTuningConfig,
    tune_occupancy_ratio,
)
from occupancy_ratio_benchmark.gym_control import make_gym_control_dataset as make_fori_gym_dataset


try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


SIZE_GRID: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("w4", (4,)),
    ("w8", (8,)),
    ("w12", (12,)),
    ("w16", (16,)),
    ("w24", (24,)),
    ("w32x32", (32, 32)),
    ("w48x48", (48, 48)),
    ("w64x64", (64, 64)),
    ("w96x96", (96, 96)),
    ("w128x128", (128, 128)),
)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selector_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for seed in _seeds(args):
        run_args = Namespace(**vars(args))
        run_args.seed = int(seed)
        for setting in args.settings:
            if not args.skip_fqe:
                rows, candidates, stages = run_fqe(setting=setting, args=run_args)
                selector_rows.extend(rows)
                candidate_rows.extend(candidates)
                stage_rows.extend(stages)
                _write_csv(out_dir / "selector_rows.csv", selector_rows)
                _write_csv(out_dir / "candidate_rows.csv", candidate_rows)
                _write_csv(out_dir / "stage_rows.csv", stage_rows)
            if not args.skip_fori:
                rows, candidates, stages = run_fori(setting=setting, args=run_args)
                selector_rows.extend(rows)
                candidate_rows.extend(candidates)
                stage_rows.extend(stages)
                _write_csv(out_dir / "selector_rows.csv", selector_rows)
                _write_csv(out_dir / "candidate_rows.csv", candidate_rows)
                _write_csv(out_dir / "stage_rows.csv", stage_rows)

    summary_rows = _summary_rows(selector_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    _print_summary(summary_rows)


def run_fqe(*, setting: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = make_fqe_gym_dataset(
        name=setting,
        sample_size=int(args.sample_size),
        gamma=float(args.gamma),
        seed=int(args.seed),
        n_eval=min(max(256, int(args.sample_size)), 1_024),
        n_initial_eval=min(max(128, int(args.sample_size) // 2), 512),
        target_value_rollouts=int(args.target_value_rollouts),
    )
    space = _fqe_space(args, seed=int(args.seed) + 1_000)
    selector_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    runs = []
    for count in _stage_counts(args):
        runs.append(
            (
            f"staged_prefix_cv_k{count}",
            FQETuningConfig(
                families=("neural",),
                cv_folds=int(args.cv_folds),
                seed=int(args.seed) + 11 + int(count),
                max_candidates=len(SIZE_GRID),
                promotion_candidates=len(SIZE_GRID),
                refit=True,
                screen_fraction=1.0,
                stable_fallback=False,
                staged_bootstrap_cv=True,
                staged_cv_iterations=int(count),
                staged_cv_n_bootstrap=int(args.bootstrap),
                staged_cv_min_survivors=1,
            ),
            )
        )
    runs.extend((
        (
            "naive_final_bellman_cv",
            FQETuningConfig(
                families=("neural",),
                cv_folds=int(args.cv_folds),
                seed=int(args.seed) + 12,
                max_candidates=len(SIZE_GRID),
                promotion_candidates=len(SIZE_GRID),
                refit=True,
                screen_fraction=1.0,
                score_bellman_weight=1.0,
                score_value_stability_weight=0.0,
                score_calibration_weight=0.0,
                score_runtime_weight=0.0,
                stable_fallback=False,
            ),
        ),
        (
            "product_composite_cv",
            FQETuningConfig(
                families=("neural",),
                cv_folds=int(args.cv_folds),
                seed=int(args.seed) + 13,
                max_candidates=len(SIZE_GRID),
                promotion_candidates=len(SIZE_GRID),
                refit=True,
                screen_fraction=1.0,
                stable_fallback=False,
            ),
        ),
    ))
    for selector, config in runs:
        start = time.perf_counter()
        result = tune_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=float(dataset.gamma),
            terminals=dataset.terminals,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            search_space=space,
            config=config,
        )
        runtime = time.perf_counter() - start
        selected_idx = _candidate_index(result.selected_candidate_id)
        value_estimate = _fqe_value_estimate(result.model, dataset)
        value_target = float(dataset.true_policy_value)
        selector_rows.append(
            {
                "family": "fqe",
                "setting": setting,
                "selector": selector,
                "sample_size": int(args.sample_size),
                "gamma": float(args.gamma),
                "seed": int(args.seed),
                "selected_candidate_id": result.selected_candidate_id,
                "selected_size_label": SIZE_GRID[selected_idx][0],
                "selected_hidden_dims": _dims_str(SIZE_GRID[selected_idx][1]),
                "selected_size_index": selected_idx,
                "policy_value_estimate": value_estimate,
                "policy_value_target": value_target,
                "policy_value_abs_error": abs(value_estimate - value_target),
                "runtime_sec": runtime,
            }
        )
        candidate_rows.extend(_fqe_candidate_rows(result, selector=selector, setting=setting, args=args))
        for row in result.staged_cv_rows():
            payload = _with_size_fields(row)
            payload["candidate_family"] = payload.pop("family", "")
            stage_rows.append(
                {
                    **payload,
                    "family": "fqe",
                    "setting": setting,
                    "selector": selector,
                    "seed": int(args.seed),
                }
            )
    return selector_rows, candidate_rows, stage_rows


def run_fori(*, setting: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = make_fori_gym_dataset(
        setting=setting,
        gamma=float(args.gamma),
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        target_value_rollouts=int(args.target_value_rollouts),
    )
    selector_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    run_specs: list[dict[str, Any]] = [
        {"selector": "naive_final_validation_cv", "score_method": "validation_loss", "seed_offset": 21},
        {"selector": "gmm_ratio_cv", "score_method": "bellman_gmm", "gmm_objective": "ratio", "seed_offset": 31},
        {"selector": "gmm_ope_cv", "score_method": "bellman_gmm", "gmm_objective": "ope", "seed_offset": 32},
        {"selector": "legacy_rank_cv", "score_method": "legacy_rank", "seed_offset": 33},
    ]
    run_specs.extend(
        {
            "selector": f"staged_prefix_cv_k{count}",
            "score_method": "validation_loss",
            "staged_bootstrap_cv": True,
            "staged_cv_iterations": int(count),
            "seed_offset": 41 + int(count),
        }
        for count in _stage_counts(args)
    )

    for spec in run_specs:
        selector = str(spec["selector"])
        start = time.perf_counter()
        result = _run_fori_tuning(
            dataset,
            args=args,
            candidate_items=list(enumerate(_fori_candidate_grid())),
            num_iterations=int(args.fori_final_iterations),
            seed=int(args.seed) + int(spec["seed_offset"]),
            refit=True,
            staged_bootstrap_cv=bool(spec.get("staged_bootstrap_cv", False)),
            staged_cv_iterations=int(spec.get("staged_cv_iterations", max(_stage_counts(args)))),
            score_method=str(spec.get("score_method", "validation_loss")),
            gmm_objective=str(spec.get("gmm_objective", "ratio")),
        )
        runtime = time.perf_counter() - start
        selected_orig_idx = _candidate_index(result.selected_candidate_id)
        value_estimate, value_target, value_abs_error, ess, weight_cv = _fori_value_diagnostics(result.model, dataset)
        selector_rows.append(
            {
                "family": "fori",
                "setting": setting,
                "selector": selector,
                "sample_size": int(args.sample_size),
                "gamma": float(args.gamma),
                "seed": int(args.seed),
                "selected_candidate_id": f"neural_{selected_orig_idx:03d}",
                "selected_size_label": SIZE_GRID[selected_orig_idx][0],
                "selected_hidden_dims": _dims_str(SIZE_GRID[selected_orig_idx][1]),
                "selected_size_index": selected_orig_idx,
                "policy_value_estimate": value_estimate,
                "policy_value_target": value_target,
                "policy_value_abs_error": value_abs_error,
                "effective_sample_size_fraction": ess,
                "weight_cv": weight_cv,
                "runtime_sec": runtime,
            }
        )
        candidate_rows.extend(
            _fori_candidate_rows(
                result,
                selector=selector,
                setting=setting,
                args=args,
                active_orig_indices=list(range(len(SIZE_GRID))),
            )
        )
        if result.staged_cv is not None:
            for row in result.staged_cv.candidate_dicts():
                payload = _with_size_fields(row)
                payload["candidate_family"] = payload.pop("family", "")
                stage_rows.append(
                    {
                        **payload,
                        "family": "fori",
                        "setting": setting,
                        "selector": selector,
                        "seed": int(args.seed),
                    }
                )
    return selector_rows, candidate_rows, stage_rows


def _fqe_space(args: argparse.Namespace, *, seed: int) -> FQESearchSpace:
    base = NeuralFQEConfig.stable_defaults(
        hidden_dims=(32, 32),
        learning_rate=float(args.fqe_learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_iterations=int(args.fqe_final_iterations),
        gradient_steps_per_iteration=int(args.fqe_gradient_steps),
        target_update_tau=0.35,
        validation_fraction=0.20,
        patience=4,
        min_improvement=1e-5,
        device="cpu",
        seed=int(seed),
    )
    return FQESearchSpace(
        neural=base,
        neural_candidates=[
            {
                "hidden_dims": dims,
                "weight_decay": _size_weight_decay(dims, args),
                "_meta": _size_complexity_meta(idx),
            }
            for idx, (_, dims) in enumerate(SIZE_GRID)
        ],
    )


def _fori_base_space(args: argparse.Namespace, *, num_iterations: int, seed: int, candidates: list[dict[str, dict[str, Any]]]) -> OccupancySearchSpace:
    nuisance_dims = tuple(int(width) for width in args.fori_nuisance_hidden_dims)
    occupancy = NeuralOccupancyRegressionConfig(
        hidden_dims=(32, 32),
        activation="silu",
        learning_rate=float(args.fori_learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_iterations=int(num_iterations),
        gradient_steps_per_iteration=int(args.fori_gradient_steps),
        mcmc_samples=int(args.fori_mcmc_samples),
        fixed_point_damping=0.5,
        min_outer_iterations=1,
        validation_warmup_iterations=0,
        patience=4,
        direct_adjoint_steps=int(args.fori_direct_adjoint_steps),
        direct_one_step_max_steps=int(args.fori_nuisance_steps),
        direct_one_step_hidden_dims=nuisance_dims,
        device="cpu",
        seed=int(seed),
    )
    action = NeuralActionRatioConfig(
        hidden_dims=nuisance_dims,
        batch_size=int(args.batch_size),
        max_steps=int(args.fori_nuisance_steps),
        patience=4,
        moment_calibration="scalar",
        density_ratio_loss="lsif",
        device="cpu",
        seed=int(seed) + 1,
    )
    source = NeuralSourceStateRatioConfig(
        hidden_dims=nuisance_dims,
        batch_size=int(args.batch_size),
        max_steps=int(args.fori_nuisance_steps),
        patience=4,
        moment_calibration="scalar",
        density_ratio_loss="lsif",
        device="cpu",
        seed=int(seed) + 2,
    )
    transition = NeuralTransitionRatioConfig(
        hidden_dims=nuisance_dims,
        batch_size=int(args.batch_size),
        max_steps=int(args.fori_nuisance_steps),
        permutation_samples=int(args.fori_transition_permutation_samples),
        patience=4,
        moment_calibration="scalar",
        density_ratio_loss="lsif",
        device="cpu",
        seed=int(seed) + 3,
    )
    return OccupancySearchSpace(
        neural_occupancy=occupancy,
        neural_action_ratio=action,
        neural_source_state_ratio=source,
        neural_transition_ratio=transition,
        neural_candidates=tuple(candidates),
    )


def _run_fori_tuning(
    dataset: Any,
    *,
    args: argparse.Namespace,
    candidate_items: list[tuple[int, dict[str, dict[str, Any]]]],
    num_iterations: int,
    seed: int,
    refit: bool,
    staged_bootstrap_cv: bool = False,
    staged_cv_iterations: int | None = None,
    score_method: str = "validation_loss",
    gmm_objective: str = "ratio",
) -> Any:
    candidates = [candidate for _, candidate in candidate_items]
    space = _fori_base_space(args, num_iterations=num_iterations, seed=seed, candidates=candidates)
    config = OccupancyTuningConfig(
        families=("neural",),
        cv_folds=int(args.cv_folds),
        seed=int(seed),
        budget="balanced",
        max_candidates=len(candidates),
        promotion_candidates=len(candidates),
        refit=bool(refit),
        screen_fraction=1.0,
        score_method=str(score_method),
        gmm_objective=str(gmm_objective),
        stable_fallback=False,
        stagewise=True,
        first_stage_cv_folds=int(args.cv_folds),
        initial_ratio_mode_candidates=("auto",),
        one_step_ratio_mode_candidates=("auto",),
        staged_bootstrap_cv=bool(staged_bootstrap_cv),
        staged_cv_iterations=int(staged_cv_iterations if staged_cv_iterations is not None else max(_stage_counts(args))),
        staged_cv_n_bootstrap=int(args.bootstrap),
        staged_cv_min_survivors=1,
    )
    return tune_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        target_next_actions=dataset.next_target_actions,
        rewards=dataset.rewards,
        gamma=float(dataset.gamma),
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        initial_weights=dataset.initial_weights,
        search_space=space,
        config=config,
        initial_ratio_mode="auto",
        one_step_ratio_mode="auto",
    )


def _fori_candidate_grid() -> list[dict[str, dict[str, Any]]]:
    return [
        {
            "_meta": _size_complexity_meta(idx),
            "occupancy": {
                "hidden_dims": dims,
                "weight_decay": 0.0 if max(dims) >= 96 else 1e-5,
            }
        }
        for idx, (_, dims) in enumerate(SIZE_GRID)
    ]


def _fqe_candidate_rows(result: Any, *, selector: str, setting: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for row in result.candidate_rows():
        payload = dict(row)
        payload["candidate_family"] = payload.get("family", "")
        idx = _candidate_index(str(row.get("candidate_id", "")))
        rows.append(
            {
                **payload,
                "family": "fqe",
                "setting": setting,
                "selector": selector,
                "seed": int(args.seed),
                "sample_size": int(args.sample_size),
                "gamma": float(args.gamma),
                "size_label": SIZE_GRID[idx][0],
                "hidden_dims": _dims_str(SIZE_GRID[idx][1]),
                "size_index": idx,
            }
        )
    return rows


def _fori_candidate_rows(
    result: Any,
    *,
    selector: str,
    setting: str,
    args: argparse.Namespace,
    active_orig_indices: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for row in result.candidate_rows():
        payload = dict(row)
        payload["candidate_family"] = payload.get("family", "")
        idx = _candidate_index(str(row.get("candidate_id", "")))
        if idx >= len(active_orig_indices):
            continue
        orig_idx = active_orig_indices[idx]
        rows.append(
            {
                **payload,
                "family": "fori",
                "setting": setting,
                "selector": selector,
                "seed": int(args.seed),
                "sample_size": int(args.sample_size),
                "gamma": float(args.gamma),
                "size_label": SIZE_GRID[orig_idx][0],
                "hidden_dims": _dims_str(SIZE_GRID[orig_idx][1]),
                "size_index": orig_idx,
            }
        )
    return rows


def _fqe_value_estimate(model: Any, dataset: Any) -> float:
    values = model.predict_q(dataset.initial_states, dataset.initial_actions)
    return float(np.mean(np.asarray(values, dtype=np.float64).reshape(-1)))


def _fori_value_diagnostics(model: Any, dataset: Any) -> tuple[float, float, float, float, float]:
    weights = np.asarray(model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True), dtype=np.float64).reshape(-1)
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    value_estimate = float(np.mean(weights * rewards))
    value_target = float(dataset.metadata["target_policy_value"])
    ess = float((np.sum(weights) ** 2) / max(float(np.sum(weights**2)), 1e-12) / weights.shape[0])
    weight_cv = float(np.std(weights) / max(abs(float(np.mean(weights))), 1e-12))
    return value_estimate, value_target, abs(value_estimate - value_target), ess, weight_cv


def _with_size_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    idx = _candidate_index(str(out.get("candidate_id", "")))
    out["size_label"] = SIZE_GRID[idx][0]
    out["hidden_dims"] = _dims_str(SIZE_GRID[idx][1])
    out["size_index"] = idx
    return out


def _summary_rows(selector_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(selector_rows, key=lambda row: (str(row["family"]), str(row["setting"]), str(row["selector"])))


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("family\tsetting\tseed\tselector\tselected\tabs_error\truntime_sec")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row.get("family", "")),
                    str(row.get("setting", "")),
                    str(row.get("seed", "")),
                    str(row.get("selector", "")),
                    str(row.get("selected_hidden_dims", "")),
                    f"{float(row.get('policy_value_abs_error', float('nan'))):.6g}",
                    f"{float(row.get('runtime_sec', float('nan'))):.2f}",
                ]
            )
        )


def _bootstrap_se(values: np.ndarray, *, iterations: int, rng: np.random.Generator) -> float:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    if int(iterations) <= 0:
        return float(np.std(finite, ddof=1) / np.sqrt(finite.size))
    draws = rng.integers(0, finite.size, size=(int(iterations), finite.size))
    return float(np.std(np.mean(finite[draws], axis=1), ddof=1))


def _candidate_index(candidate_id: str) -> int:
    try:
        return int(str(candidate_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _dims_str(dims: Iterable[int]) -> str:
    return "x".join(str(int(dim)) for dim in dims)


def _size_weight_decay(dims: tuple[int, ...], args: argparse.Namespace) -> float:
    return 0.0 if max(dims) >= int(args.large_width_cutoff) else float(args.weight_decay)


def _size_complexity_meta(idx: int) -> dict[str, Any]:
    return {"complexity_group": "neural_size_ladder", "complexity_rank": int(idx) + 1}


def _stage_counts(args: argparse.Namespace) -> tuple[int, ...]:
    raw = getattr(args, "stage_counts", None)
    if raw is None:
        raw = getattr(args, "stages", (3,))
    values = tuple(sorted({int(value) for value in raw}))
    return values or (3,)


def _seeds(args: argparse.Namespace) -> tuple[int, ...]:
    raw = getattr(args, "seeds", None)
    if raw is None:
        return (int(args.seed),)
    return tuple(int(seed) for seed in raw)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural staged-CV model-size screen for FQE and FORI.")
    parser.add_argument("--output-dir", default="outputs/neural_size_staged_cv_screen")
    parser.add_argument("--settings", nargs="+", default=("gym_pendulum",))
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--target-value-rollouts", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--stages", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--stage-counts", nargs="+", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--large-width-cutoff", type=int, default=96)
    parser.add_argument("--fqe-final-iterations", type=int, default=5)
    parser.add_argument("--fqe-gradient-steps", type=int, default=4)
    parser.add_argument("--fqe-learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--fori-final-iterations", type=int, default=3)
    parser.add_argument("--fori-gradient-steps", type=int, default=2)
    parser.add_argument("--fori-learning-rate", type=float, default=8e-4)
    parser.add_argument("--fori-mcmc-samples", type=int, default=2)
    parser.add_argument("--fori-nuisance-steps", type=int, default=16)
    parser.add_argument("--fori-nuisance-hidden-dims", nargs="+", type=int, default=(32, 32))
    parser.add_argument("--fori-transition-permutation-samples", type=int, default=2)
    parser.add_argument("--fori-direct-adjoint-steps", type=int, default=16)
    parser.add_argument("--skip-fqe", action="store_true")
    parser.add_argument("--skip-fori", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
