from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "rltools-matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

from occupancy_ratio.fit_occupancy_ratio_neural import (  # noqa: E402
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
)
from occupancy_ratio_benchmark.data import BenchmarkDataset  # noqa: E402
from occupancy_ratio_benchmark.external_baselines import (  # noqa: E402
    estimate_google_dualdice_neural,
    preflight_google_dualdice,
)
from occupancy_ratio_benchmark.fori_cv import (  # noqa: E402
    FORICVCandidate,
    compute_weight_diagnostics,
    fit_fori_cv_candidate,
    score_fixed_point_residual,
    score_moment_balance,
    score_value_grouped_moment_balance,
)
from occupancy_ratio_benchmark.gym_control import make_gym_control_dataset  # noqa: E402


GYM_SETTINGS = ("gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper")
FORI_CANDIDATES = (
    "neural_stable",
    "neural_google_parity",
    "neural_relaxed_tail",
    "neural_transition_norm",
    "neural_stable_factored",
    "neural_stable_logistic_nuisance",
)
SELECTORS = (
    "mb_value_grouped",
    "mb_reward",
    "mb_plain",
    "mb_reward_scalar",
    "mb_rff_only",
    "fp",
    "ess_composite",
    "best_validation_loss",
    "final_validation_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy resumable FORI selector benchmark on Gym suites.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output-dir", type=Path, default=Path(f"outputs/fori_cv_gym_heavy_overnight_{timestamp}"))
    parser.add_argument("--settings", nargs="+", default=list(GYM_SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[1000, 5000])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.95, 0.99])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--mcmc-samples", type=int, default=24)
    parser.add_argument("--action-steps", type=int, default=1000)
    parser.add_argument("--transition-steps", type=int, default=1400)
    parser.add_argument("--reward-steps", type=int, default=200)
    parser.add_argument("--reward-patience", type=int, default=15)
    parser.add_argument("--value-fqe-iterations", type=int, default=120)
    parser.add_argument("--value-fqe-patience", type=int, default=10)
    parser.add_argument("--target-rollouts", type=int, default=32)
    parser.add_argument("--dualdice-updates", nargs="+", type=int, default=[5000])
    parser.add_argument("--dualdice-batch-size", type=int, default=128)
    parser.add_argument("--external-repo-path", type=Path, default=Path("/tmp/google-research"))
    parser.add_argument("--no-dualdice", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "run_config.json", vars(args))
    failures_path = args.output_dir / "failures.csv"
    fold_path = args.output_dir / "fold_rows.csv"
    final_path = args.output_dir / "final_refits.csv"
    dice_path = args.output_dir / "dualdice_reference.csv"

    preflight = None if args.no_dualdice else preflight_google_dualdice(args.external_repo_path)
    for setting in args.settings:
        for seed in args.seeds:
            for sample_size in args.sample_sizes:
                for gamma in args.gammas:
                    cell = dict(setting=str(setting), seed=int(seed), sample_size=int(sample_size), gamma=float(gamma))
                    try:
                        dataset = make_gym_control_dataset(
                            setting=str(setting),
                            gamma=float(gamma),
                            sample_size=int(sample_size),
                            seed=int(seed),
                            target_value_rollouts=int(args.target_rollouts),
                        )
                    except Exception as exc:
                        _record_failure(failures_path, cell | {"stage": "dataset"}, exc)
                        continue

                    candidates = make_candidates(args)
                    _run_cv_cell(
                        dataset=dataset,
                        candidates=candidates,
                        args=args,
                        cell=cell,
                        fold_path=fold_path,
                        failures_path=failures_path,
                    )
                    _run_final_refits(
                        dataset=dataset,
                        candidates=candidates,
                        args=args,
                        cell=cell,
                        final_path=final_path,
                        failures_path=failures_path,
                    )
                    if not args.no_dualdice and preflight is not None:
                        _run_dualdice_reference(
                            dataset=dataset,
                            args=args,
                            cell=cell,
                            preflight=preflight,
                            dice_path=dice_path,
                            failures_path=failures_path,
                        )
                    _write_all_summaries(args.output_dir, write_plots=not bool(args.no_plots))

    _write_all_summaries(args.output_dir, write_plots=not bool(args.no_plots))


def make_candidates(args: argparse.Namespace) -> list[FORICVCandidate]:
    out = []
    for name in FORI_CANDIDATES:
        preset = name.removeprefix("neural_")
        occ_over, nuisance_max, density_loss, moment_calibration, initial_mode, one_step_mode, hidden_dims, activation = (
            _preset_options(args, preset)
        )
        base_occ = NeuralOccupancyRegressionConfig.stable_defaults(
            num_iterations=int(args.iterations),
            gradient_steps_per_iteration=int(args.gradient_steps),
            mcmc_samples=int(args.mcmc_samples),
            validation_fraction=0.2,
            patience=12,
            show_progress=False,
            batch_size=512,
            hidden_dims=hidden_dims,
            activation=activation,
            direct_one_step_density_ratio_loss=density_loss,
            direct_one_step_prediction_max=nuisance_max,
            direct_one_step_moment_calibration=moment_calibration,
            direct_one_step_max_steps=int(args.transition_steps),
            **occ_over,
        )
        nuisance_kwargs = dict(
            hidden_dims=hidden_dims,
            activation=activation,
            max_steps=int(args.action_steps),
            validation_fraction=0.2,
            patience=20,
            batch_size=512,
            prediction_max=nuisance_max,
            moment_calibration=moment_calibration,
            density_ratio_loss=density_loss,
        )
        transition_kwargs = dict(nuisance_kwargs)
        transition_kwargs["max_steps"] = int(args.transition_steps)
        transition_kwargs["permutation_samples"] = 4
        out.append(
            FORICVCandidate(
                name=name,
                family="neural",
                occupancy=base_occ,
                action_ratio=NeuralActionRatioConfig.stable_defaults(**nuisance_kwargs),
                source_state_ratio=NeuralSourceStateRatioConfig.stable_defaults(**nuisance_kwargs),
                transition_ratio=NeuralTransitionRatioConfig.stable_defaults(**transition_kwargs),
                initial_ratio_mode=initial_mode,
                one_step_ratio_mode=one_step_mode,
                metadata={"preset": preset},
            )
        )
    return out


def _preset_options(args: argparse.Namespace, preset: str) -> tuple[dict[str, Any], float | None, str, str, str, str, tuple[int, ...], str]:
    hidden_dims: tuple[int, ...] = (64, 64)
    activation = "silu"
    density_loss = "lsif"
    moment_calibration = "scalar"
    initial_mode = "auto"
    one_step_mode = "auto"
    nuisance_max: float | None = 50.0
    occ = dict(
        loss="huber",
        fixed_point_damping=0.5,
        normalize_occupancy=True,
        occupancy_ratio_max=50.0,
        clip_pseudo_outcomes=True,
        pseudo_outcome_upper_quantile=0.995,
        occupancy_sample_weight_mode="uniform",
        occupancy_sample_weight_max=20.0,
        normalize_transition_cache=False,
    )
    if preset == "google_parity":
        hidden_dims = (256, 256)
        activation = "relu"
    elif preset == "relaxed_tail":
        nuisance_max = 100.0
        occ.update(fixed_point_damping=0.75, occupancy_ratio_max=100.0, pseudo_outcome_upper_quantile=0.999)
    elif preset == "transition_norm":
        occ.update(normalize_transition_cache=True)
    elif preset == "stable_factored":
        initial_mode = "factored"
        one_step_mode = "factored"
    elif preset == "stable_logistic_nuisance":
        density_loss = "logistic"
    elif preset != "stable":
        raise ValueError(f"Unknown FORI preset '{preset}'.")
    return occ, nuisance_max, density_loss, moment_calibration, initial_mode, one_step_mode, hidden_dims, activation


def _run_cv_cell(
    *,
    dataset: BenchmarkDataset,
    candidates: list[FORICVCandidate],
    args: argparse.Namespace,
    cell: dict[str, Any],
    fold_path: Path,
    failures_path: Path,
) -> None:
    completed = _completed_keys(
        fold_path,
        ("setting", "seed", "sample_size", "gamma", "candidate", "fold"),
    )
    folds = _make_folds(dataset.n, int(args.folds), seed=int(cell["seed"]) + 31)
    for candidate in candidates:
        for fold, valid_idx in enumerate(folds):
            row_key = _key(cell | {"candidate": candidate.name, "fold": int(fold)}, ("setting", "seed", "sample_size", "gamma", "candidate", "fold"))
            if row_key in completed and not bool(args.rerun):
                continue
            train_idx = _complement_indices(dataset.n, valid_idx)
            start = time.perf_counter()
            try:
                fit = fit_fori_cv_candidate(dataset, candidate, train_idx, fold=fold, seed=int(cell["seed"]) + 31)
                weights = np.asarray(
                    fit.model.predict_state_action_ratio(dataset.states[valid_idx], dataset.actions[valid_idx], clip=True),
                    dtype=np.float64,
                )
                raw = np.asarray(
                    fit.model.predict_state_action_ratio(dataset.states[valid_idx], dataset.actions[valid_idx], clip=False),
                    dtype=np.float64,
                )
                row = cell | {
                    "candidate": candidate.name,
                    "fold": int(fold),
                    "status": "ok",
                    "runtime_sec": float(time.perf_counter() - start),
                }
                row.update(_selector_scores(fit, dataset, valid_idx, seed=int(cell["seed"]) + 31, args=args))
                row.update(score_fixed_point_residual(fit, dataset, valid_idx, seed=int(cell["seed"]) + 31))
                row.update(compute_weight_diagnostics(weights, raw_weights=raw))
                row.update(_fold_value_metrics(dataset, valid_idx, weights))
                row.update(_training_loss_metrics(fit.model))
                row.update(_candidate_settings(candidate, fit.model))
                _upsert_csv(fold_path, row, ("setting", "seed", "sample_size", "gamma", "candidate", "fold"))
                completed.add(row_key)
            except Exception as exc:
                _record_failure(failures_path, cell | {"stage": "cv", "candidate": candidate.name, "fold": int(fold)}, exc)


def _selector_scores(
    fit: Any,
    dataset: BenchmarkDataset,
    valid_idx: np.ndarray,
    *,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variants = {
        "reward": dict(reward_features=True, reward_phi_features=True, raw_features=True, rff_features=16),
        "plain": dict(reward_features=False, raw_features=True, rff_features=16),
        "reward_scalar": dict(reward_features=True, reward_phi_features=False, raw_features=True, rff_features=16),
        "rff_only": dict(reward_features=False, raw_features=False, rff_features=16),
    }
    out: dict[str, Any] = {}
    for suffix, kwargs in variants.items():
        scores = score_moment_balance(
            fit,
            dataset,
            valid_idx,
            seed=seed,
            reward_max_steps=int(args.reward_steps),
            reward_patience=int(args.reward_patience),
            **kwargs,
        )
        for key, value in scores.items():
            out[f"{key}_{suffix}"] = value
    out.update(
        score_value_grouped_moment_balance(
            fit,
            dataset,
            valid_idx,
            seed=seed,
            reward_max_steps=int(args.reward_steps),
            reward_patience=int(args.reward_patience),
            fqe_iterations=int(args.value_fqe_iterations),
            fqe_patience=int(args.value_fqe_patience),
        )
    )
    out["mb"] = out.get("mb_reward", float("nan"))
    out["mb_features"] = out.get("mb_features_reward", float("nan"))
    out["mb_reward_features"] = out.get("mb_reward_features_reward", float("nan"))
    return out


def _run_final_refits(
    *,
    dataset: BenchmarkDataset,
    candidates: list[FORICVCandidate],
    args: argparse.Namespace,
    cell: dict[str, Any],
    final_path: Path,
    failures_path: Path,
) -> None:
    completed = _completed_keys(final_path, ("setting", "seed", "sample_size", "gamma", "candidate"))
    for candidate in candidates:
        row_key = _key(cell | {"candidate": candidate.name}, ("setting", "seed", "sample_size", "gamma", "candidate"))
        if row_key in completed and not bool(args.rerun):
            continue
        start = time.perf_counter()
        try:
            train_idx = np.arange(dataset.n, dtype=np.int64)
            fit = fit_fori_cv_candidate(dataset, candidate, train_idx, fold=-1, seed=int(cell["seed"]) + 91)
            weights = np.asarray(fit.model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True), dtype=np.float64)
            raw = np.asarray(fit.model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False), dtype=np.float64)
            row = cell | {
                "candidate": candidate.name,
                "status": "ok",
                "runtime_sec": float(time.perf_counter() - start),
                **compute_weight_diagnostics(weights, raw_weights=raw),
                **_value_metrics(dataset, weights),
                **_weight_histogram(weights),
                **_training_loss_metrics(fit.model),
                **_candidate_settings(candidate, fit.model),
            }
            _upsert_csv(final_path, row, ("setting", "seed", "sample_size", "gamma", "candidate"))
        except Exception as exc:
            _record_failure(failures_path, cell | {"stage": "final_refit", "candidate": candidate.name}, exc)


def _run_dualdice_reference(
    *,
    dataset: BenchmarkDataset,
    args: argparse.Namespace,
    cell: dict[str, Any],
    preflight: Any,
    dice_path: Path,
    failures_path: Path,
) -> None:
    completed = _completed_keys(dice_path, ("setting", "seed", "sample_size", "gamma", "updates"))
    for updates in args.dualdice_updates:
        row_key = _key(cell | {"updates": int(updates)}, ("setting", "seed", "sample_size", "gamma", "updates"))
        if row_key in completed and not bool(args.rerun):
            continue
        try:
            result = estimate_google_dualdice_neural(
                dataset,
                preflight=preflight,
                num_updates=int(updates),
                batch_size=int(args.dualdice_batch_size),
                diagnostic_features=np.concatenate([dataset.states, dataset.actions], axis=1),
                value_diagnostics={},
            )
            weights = result.get("weights")
            row = cell | {
                "updates": int(updates),
                "status": str(result.get("status", "")),
                "runtime_sec": float(result.get("runtime_sec", 0.0)),
                "skip_reason": str(result.get("skip_reason", "")),
            }
            if weights is not None:
                weights_arr = np.asarray(weights, dtype=np.float64)
                row.update(compute_weight_diagnostics(weights_arr, raw_weights=result.get("raw_weights")))
                row.update(_value_metrics(dataset, weights_arr))
                row.update(_weight_histogram(weights_arr))
            diagnostics = result.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                for key in ("google_final_loss", "google_num_updates", "google_batch_size"):
                    if key in diagnostics:
                        row[key] = diagnostics[key]
            _upsert_csv(dice_path, row, ("setting", "seed", "sample_size", "gamma", "updates"))
        except Exception as exc:
            _record_failure(failures_path, cell | {"stage": "dualdice", "updates": int(updates)}, exc)


def _write_all_summaries(output_dir: Path, *, write_plots: bool) -> None:
    fold_rows = _read_csv(output_dir / "fold_rows.csv")
    final_rows = _read_csv(output_dir / "final_refits.csv")
    dice_rows = _read_csv(output_dir / "dualdice_reference.csv")
    candidate_summary = _candidate_summary(fold_rows)
    _write_csv(output_dir / "candidate_summary.csv", candidate_summary)
    selector_summary, selector_details = _selector_summary(candidate_summary, final_rows)
    _write_csv(output_dir / "selector_summary.csv", selector_summary)
    _write_csv(output_dir / "selector_details.csv", selector_details)
    _write_report(output_dir, selector_summary, candidate_summary, final_rows, dice_rows)
    if write_plots:
        _write_plots(output_dir, selector_summary, candidate_summary, final_rows, dice_rows)


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("setting", "seed", "sample_size", "gamma", "candidate")
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    out = []
    metric_names = [
        "mb_value_grouped",
        "mb_value_grouped_max_group",
        "mb_reward",
        "mb_plain",
        "mb_reward_scalar",
        "mb_rff_only",
        "fp",
        "ess_fraction",
        "mean_ratio",
        "q99_ratio",
        "max_ratio",
        "best_validation_loss",
        "final_validation_loss",
        "ope_value_abs_error",
    ]
    for group_key, group in sorted(groups.items()):
        row = dict(zip(keys, group_key))
        row["folds"] = len(group)
        row["invalid"] = any(_as_bool(item.get("invalid")) for item in group)
        for name in metric_names:
            vals = _finite_values(item.get(name) for item in group)
            row[f"{name}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{name}_se"] = _se(vals)
        row["ess_fraction_min"] = float(np.min(_finite_values(item.get("ess_fraction") for item in group)))
        row["stabilization_strength"] = float(np.mean(_finite_values(item.get("stabilization_strength") for item in group)))
        out.append(row)
    return out


def _selector_summary(candidate_summary: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    cell_keys = ("setting", "seed", "sample_size", "gamma")
    for row in final_rows:
        if row.get("status") != "ok":
            continue
        final_by_cell.setdefault(tuple(row.get(key) for key in cell_keys), {})[str(row.get("candidate"))] = row
    cv_by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in candidate_summary:
        cv_by_cell.setdefault(tuple(row.get(key) for key in cell_keys), []).append(row)

    rows = []
    detail_rows = []
    for selector in SELECTORS:
        regrets = []
        selected_errors = []
        wins = 0
        count = 0
        selected_counts: dict[str, int] = {}
        for cell, cv_rows in cv_by_cell.items():
            final = final_by_cell.get(cell, {})
            if not final:
                continue
            best_error = min(_float_or_inf(row.get("ope_value_abs_error")) for row in final.values())
            selected = _select_by_selector(cv_rows, selector)
            if selected is None:
                continue
            final_selected = final.get(str(selected))
            if final_selected is None:
                continue
            error = _float_or_inf(final_selected.get("ope_value_abs_error"))
            regret = error - best_error
            regrets.append(regret)
            selected_errors.append(error)
            wins += int(abs(regret) <= 1e-12)
            count += 1
            selected_counts[str(selected)] = selected_counts.get(str(selected), 0) + 1
            detail_rows.append(
                dict(
                    selector=selector,
                    setting=cell[0],
                    seed=cell[1],
                    sample_size=cell[2],
                    gamma=cell[3],
                    selected_candidate=str(selected),
                    selected_ope_abs_error=error,
                    best_fori_ope_abs_error=best_error,
                    regret=regret,
                )
            )
        arr = np.asarray(regrets, dtype=np.float64)
        err = np.asarray(selected_errors, dtype=np.float64)
        rows.append(
            dict(
                selector=selector,
                cells=count,
                win_rate=float(wins / max(count, 1)),
                mean_regret=float(np.mean(arr)) if arr.size else float("nan"),
                median_regret=float(np.median(arr)) if arr.size else float("nan"),
                worst_regret=float(np.max(arr)) if arr.size else float("nan"),
                mean_ope_abs_error=float(np.mean(err)) if err.size else float("nan"),
                selected_counts=json.dumps(selected_counts, sort_keys=True),
            )
        )
    return rows, detail_rows


def _select_by_selector(rows: list[dict[str, Any]], selector: str) -> str | None:
    valid = [row for row in rows if not _as_bool(row.get("invalid"))]
    if not valid:
        return None
    if selector == "mb_reward":
        return _argmin_with_stability_tie(valid, "mb_reward_mean")
    if selector == "mb_value_grouped":
        return _argmin_with_stability_tie(valid, "mb_value_grouped_mean")
    if selector == "mb_plain":
        return _argmin_with_stability_tie(valid, "mb_plain_mean")
    if selector == "mb_reward_scalar":
        return _argmin_with_stability_tie(valid, "mb_reward_scalar_mean")
    if selector == "mb_rff_only":
        return _argmin_with_stability_tie(valid, "mb_rff_only_mean")
    if selector == "fp":
        return _argmin_with_stability_tie(valid, "fp_mean")
    if selector == "best_validation_loss":
        return _argmin_with_stability_tie(valid, "best_validation_loss_mean")
    if selector == "final_validation_loss":
        return _argmin_with_stability_tie(valid, "final_validation_loss_mean")
    if selector == "ess_composite":
        scored = []
        for row in valid:
            ess = _float_or_nan(row.get("ess_fraction_mean"))
            mean_ratio = _float_or_nan(row.get("mean_ratio_mean"))
            p99 = _float_or_nan(row.get("q99_ratio_mean"))
            score = abs(mean_ratio - 1.0) + max(0.0, 0.05 - ess) + 0.001 * np.log1p(max(p99, 0.0))
            scored.append((score, row))
        return str(min(scored, key=lambda item: item[0])[1].get("candidate"))
    raise ValueError(f"Unknown selector '{selector}'.")


def _argmin_with_stability_tie(rows: list[dict[str, Any]], metric: str) -> str | None:
    scored = [(float(row.get(metric, float("inf"))), row) for row in rows]
    scored = [(score, row) for score, row in scored if np.isfinite(score)]
    if not scored:
        return None
    best = min(score for score, _ in scored)
    tied = [row for score, row in scored if score <= best + max(0.0, _float_or_nan(row.get(metric.replace("_mean", "_se"))))]
    return str(min(tied, key=lambda row: (_float_or_nan(row.get("stabilization_strength")), _float_or_nan(row.get(metric))))["candidate"])


def _write_report(
    output_dir: Path,
    selector_summary: list[dict[str, Any]],
    candidate_summary: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    dice_rows: list[dict[str, Any]],
) -> None:
    summary_rows = [row for row in selector_summary if row.get("selector") != "__detail__"]
    ordered = sorted(summary_rows, key=lambda row: _float_or_inf(row.get("mean_regret")))
    primary_names = {"mb_value_grouped", "mb_reward", "mb_plain", "mb_reward_scalar", "mb_rff_only"}
    primary_ordered = [row for row in ordered if row.get("selector") in primary_names]
    best_primary = primary_ordered[0] if primary_ordered else {}
    best_control = next((row for row in ordered if row.get("selector") not in primary_names), {})
    value_grouped = next((row for row in summary_rows if row.get("selector") == "mb_value_grouped"), {})
    reward = next((row for row in summary_rows if row.get("selector") == "mb_reward"), {})
    plain = next((row for row in summary_rows if row.get("selector") == "mb_plain"), {})
    default = "mb_plain"
    if (
        _float_or_inf(reward.get("mean_regret")) <= _float_or_inf(plain.get("mean_regret"))
        and _float_or_inf(reward.get("worst_regret")) <= _float_or_inf(plain.get("worst_regret"))
    ):
        default = "mb_reward"
    if (
        _float_or_inf(value_grouped.get("mean_regret")) <= _float_or_inf(next((row for row in summary_rows if row.get("selector") == default), {}).get("mean_regret"))
        and _float_or_inf(value_grouped.get("worst_regret")) <= _float_or_inf(plain.get("worst_regret"))
    ):
        default = "mb_value_grouped"
    lines = [
        "# Heavy FORI Gym Selector Report",
        "",
        f"Best primary selector by mean regret: `{best_primary.get('selector', '')}`.",
        f"Best diagnostic/control row by mean regret: `{best_control.get('selector', '')}`.",
        f"Default recommendation by reward-aware rule: `{default}`.",
        "",
        "## Selector Summary",
        "",
        "| selector | cells | win_rate | mean_regret | worst_regret | mean_ope_abs_error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row.get('selector')} | {row.get('cells')} | {_fmt(row.get('win_rate'))} | "
            f"{_fmt(row.get('mean_regret'))} | {_fmt(row.get('worst_regret'))} | {_fmt(row.get('mean_ope_abs_error'))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- FP is reported as an internal-consistency diagnostic only; selector regret excludes FP unless reading the explicit FP negative-control row.",
            "- DualDICE rows are report-only and are not included in FORI selector regret.",
            f"- Candidate-summary rows: {len(candidate_summary)}; final FORI rows: {len(final_rows)}; DualDICE rows: {len(dice_rows)}.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def _write_plots(
    output_dir: Path,
    selector_summary: list[dict[str, Any]],
    candidate_summary: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    dice_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [row for row in selector_summary if row.get("selector") != "__detail__"]
    if summary_rows:
        plt.figure(figsize=(8, 4))
        labels = [str(row["selector"]) for row in summary_rows]
        values = [_float_or_nan(row.get("mean_regret")) for row in summary_rows]
        plt.bar(labels, values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("mean regret")
        plt.tight_layout()
        plt.savefig(plot_dir / "selector_regret.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        values = [_float_or_nan(row.get("win_rate")) for row in summary_rows]
        plt.bar(labels, values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("win rate")
        plt.tight_layout()
        plt.savefig(plot_dir / "selector_win_rate.png", dpi=160)
        plt.close()

    joined = _join_candidate_final(candidate_summary, final_rows)
    if joined:
        plt.figure(figsize=(5, 4))
        plt.scatter([_float_or_nan(row.get("mb_reward_mean")) for row in joined], [_float_or_nan(row.get("ope_value_abs_error")) for row in joined], s=14, alpha=0.6)
        plt.xlabel("reward-aware MB")
        plt.ylabel("final OPE abs error")
        plt.tight_layout()
        plt.savefig(plot_dir / "mb_vs_oracle_ope_error.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.scatter([_float_or_nan(row.get("fp_mean")) for row in joined], [_float_or_nan(row.get("mb_reward_mean")) for row in joined], s=14, alpha=0.6)
        plt.xlabel("FP")
        plt.ylabel("reward-aware MB")
        plt.tight_layout()
        plt.savefig(plot_dir / "fp_vs_mb.png", dpi=160)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.scatter([_float_or_nan(row.get("ess_fraction_mean")) for row in joined], [_float_or_nan(row.get("mb_reward_mean")) for row in joined], s=14, alpha=0.6)
        plt.xlabel("ESS/n")
        plt.ylabel("reward-aware MB")
        plt.tight_layout()
        plt.savefig(plot_dir / "ess_vs_mb.png", dpi=160)
        plt.close()

    if dice_rows and final_rows:
        plt.figure(figsize=(5, 4))
        for label, rows in (("FORI", final_rows), ("DualDICE", dice_rows)):
            vals = [_float_or_nan(row.get("ope_value_abs_error")) for row in rows]
            vals = [val for val in vals if np.isfinite(val)]
            if vals:
                plt.hist(vals, bins=30, alpha=0.4, label=label)
        plt.xlabel("OPE abs error")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "dualdice_reference_comparison.png", dpi=160)
        plt.close()

    selected = _selected_final_rows(output_dir / "selector_details.csv", final_rows, selector="mb_reward")
    if selected:
        plt.figure(figsize=(6, 4))
        for row in selected[:12]:
            edges = _json_array(row.get("weight_hist_edges"))
            counts = _json_array(row.get("weight_hist_counts"))
            if len(edges) != len(counts) + 1 or not counts:
                continue
            mids = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
            total = max(float(np.sum(counts)), 1.0)
            label = f"{row.get('setting')} n={row.get('sample_size')} g={row.get('gamma')}"
            plt.plot(mids, np.asarray(counts, dtype=np.float64) / total, alpha=0.45, label=label)
        plt.xlabel("estimated ratio")
        plt.ylabel("fraction")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles[:6], labels[:6], fontsize=6)
        plt.tight_layout()
        plt.savefig(plot_dir / "selected_candidate_ratio_histograms.png", dpi=160)
        plt.close()


def _join_candidate_final(candidate_summary: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = {
        _key(row, ("setting", "seed", "sample_size", "gamma", "candidate")): row
        for row in final_rows
        if row.get("status") == "ok"
    }
    out = []
    for row in candidate_summary:
        match = final.get(_key(row, ("setting", "seed", "sample_size", "gamma", "candidate")))
        if match:
            out.append(dict(row) | {"ope_value_abs_error": match.get("ope_value_abs_error")})
    return out


def _selected_final_rows(selector_details_path: Path, final_rows: list[dict[str, Any]], *, selector: str) -> list[dict[str, Any]]:
    details = [row for row in _read_csv(selector_details_path) if row.get("selector") == selector]
    final = {
        _key(row, ("setting", "seed", "sample_size", "gamma", "candidate")): row
        for row in final_rows
        if row.get("status") == "ok"
    }
    out = []
    for row in details:
        key = (
            str(row.get("setting", "")),
            str(row.get("seed", "")),
            str(row.get("sample_size", "")),
            str(row.get("gamma", "")),
            str(row.get("selected_candidate", "")),
        )
        match = final.get(key)
        if match:
            out.append(match)
    return out


def _training_loss_metrics(model: Any) -> dict[str, Any]:
    diagnostics = getattr(model, "diagnostics", {}) or {}
    return {
        "best_validation_loss": _float_or_nan(diagnostics.get("occupancy_best_valid_loss")),
        "final_validation_loss": _float_or_nan(diagnostics.get("occupancy_final_valid_loss")),
        "gradient_steps_used": _float_or_nan(diagnostics.get("gradient_steps_used")),
        "accepted_count": _float_or_nan(diagnostics.get("accepted_count")),
    }


def _candidate_settings(candidate: FORICVCandidate, model: Any) -> dict[str, Any]:
    diagnostics = getattr(model, "diagnostics", {}) or {}
    return {
        "initial_ratio_mode": str(diagnostics.get("initial_ratio_mode", candidate.initial_ratio_mode)),
        "one_step_ratio_mode": str(diagnostics.get("one_step_ratio_mode", candidate.one_step_ratio_mode)),
        "fixed_point_damping": _float_or_nan(diagnostics.get("fixed_point_damping")),
        "occupancy_ratio_max": _float_or_nan(diagnostics.get("occupancy_ratio_max")),
        "stabilization_strength": _stabilization_strength(diagnostics),
    }


def _stabilization_strength(diagnostics: dict[str, Any]) -> float:
    damping = _float_or_nan(diagnostics.get("fixed_point_damping"))
    cap = _float_or_nan(diagnostics.get("occupancy_ratio_max"))
    strength = 0.0
    if np.isfinite(damping):
        strength += max(0.0, 1.0 - damping)
    if np.isfinite(cap) and cap > 0.0:
        strength += 1.0 / cap
    if _as_bool(diagnostics.get("normalize_occupancy")):
        strength += 0.05
    return float(strength)


def _fold_value_metrics(dataset: BenchmarkDataset, valid_idx: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)[valid_idx]
    estimate = float(np.mean(weights * rewards))
    target = _float_or_nan(dataset.metadata.get("target_policy_value"))
    return {
        "ope_value_estimate": estimate,
        "ope_value_target": target,
        "ope_value_abs_error": abs(estimate - target) if np.isfinite(target) else float("nan"),
    }


def _value_metrics(dataset: BenchmarkDataset, weights: np.ndarray) -> dict[str, Any]:
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    estimate = float(np.mean(np.asarray(weights, dtype=np.float64).reshape(-1) * rewards))
    target = _float_or_nan(dataset.metadata.get("target_policy_value"))
    return {
        "ope_value_estimate": estimate,
        "ope_value_target": target,
        "ope_value_abs_error": abs(estimate - target) if np.isfinite(target) else float("nan"),
    }


def _weight_histogram(weights: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"weight_hist_edges": [], "weight_hist_counts": []}
    upper = float(np.quantile(arr, 0.99))
    upper = max(upper, 1.0)
    edges = np.linspace(0.0, upper, 31)
    counts, edges = np.histogram(np.clip(arr, edges[0], edges[-1]), bins=edges)
    return {
        "weight_hist_edges": [float(x) for x in edges],
        "weight_hist_counts": [int(x) for x in counts],
    }


def _make_folds(n: int, k: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n), dtype=np.int64)
    rng.shuffle(idx)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(idx, int(k))]


def _complement_indices(n: int, valid_idx: np.ndarray) -> np.ndarray:
    mask = np.ones(int(n), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _completed_keys(path: Path, keys: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {_key(row, keys) for row in _read_csv(path) if row.get("status", "ok") == "ok"}


def _upsert_csv(path: Path, row: dict[str, Any], key_fields: tuple[str, ...]) -> None:
    rows = _read_csv(path)
    row = _jsonable(row)
    key = _key(row, key_fields)
    rows = [old for old in rows if _key(old, key_fields) != key]
    rows.append(row)
    _write_csv(path, rows)


def _record_failure(path: Path, context: dict[str, Any], exc: Exception) -> None:
    row = context | {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=8),
    }
    keys = tuple(key for key in ("setting", "seed", "sample_size", "gamma", "stage", "candidate", "fold", "updates") if key in row)
    _upsert_csv(path, row, keys or ("stage",))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([_jsonable(row) for row in rows])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(value, default=str)
        else:
            out[key] = value
    return out


def _json_array(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            pass
    return out


def _key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    out = []
    for value in values:
        val = _float_or_nan(value)
        if np.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def _se(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size else float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _float_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def _float_or_inf(value: Any) -> float:
    val = _float_or_nan(value)
    return val if np.isfinite(val) else float("inf")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _fmt(value: Any) -> str:
    val = _float_or_nan(value)
    return "" if not np.isfinite(val) else f"{val:.4g}"


if __name__ == "__main__":
    main()
