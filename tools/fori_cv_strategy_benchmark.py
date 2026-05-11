from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "occupancy-ratio"))

from occupancy_ratio import OccupancySearchSpace, OccupancyTuningConfig, tune_occupancy_ratio
from occupancy_ratio.neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    fit_discounted_occupancy_ratio_neural,
)
from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.data import BenchmarkDataset
from occupancy_ratio_benchmark.diagnostics import estimator_diagnostics_optional
from occupancy_ratio_benchmark.runner import make_dataset


try:  # pragma: no cover - depends on optional torch runtime.
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


SIZE_GRID: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("w4", (4,)),
    ("w8", (8,)),
    ("w16", (16,)),
    ("w32", (32,)),
    ("w64", (64,)),
    ("w128", (128,)),
    ("w32x32", (32, 32)),
    ("w64x64", (64, 64)),
    ("w128x128", (128, 128)),
    ("w256x256", (256, 256)),
)


CONTROLLED_SETTINGS = {"discrete_chain", "discrete_grid", "random_tabular_mdp", "linear_gaussian", "nonlinear_monte_carlo"}
GYM_SETTINGS = {"gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper"}


@dataclass(frozen=True)
class DatasetCell:
    setting: str
    sample_size: int
    gamma: float
    seed: int
    policy_shift: float | None = None
    target_value_rollouts: int | None = None

    @property
    def family(self) -> str:
        if self.setting.startswith("discrete") or self.setting == "random_tabular_mdp":
            return "discrete"
        if self.setting == "linear_gaussian":
            return "linear_gaussian"
        if self.setting == "nonlinear_monte_carlo":
            return "nonlinear"
        if self.setting.startswith("gym_"):
            return "gym"
        return "realistic"

    @property
    def cell_id(self) -> str:
        shift = "none" if self.policy_shift is None else f"{float(self.policy_shift):g}"
        return f"{self.setting}|n={self.sample_size}|g={self.gamma:g}|seed={self.seed}|shift={shift}"


def main() -> None:
    args = _parse_args()
    run_benchmark(args)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selector_rows: list[dict[str, Any]] = []
    candidate_truth_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    cells = _dataset_cells(args)
    for cell in cells:
        elapsed_min = (time.perf_counter() - started) / 60.0
        if _is_runtime_drop_cell(cell) and elapsed_min > float(args.time_budget_minutes):
            skipped_rows.append({**_cell_fields(cell), "skip_reason": "time_budget_drop_large_cell"})
            continue
        try:
            dataset = _make_cell_dataset(cell, args)
        except Exception as exc:
            skipped_rows.append({**_cell_fields(cell), "skip_reason": f"{type(exc).__name__}: {exc}"})
            continue
        candidates = _candidate_grid(input_dim=dataset.state_dim + dataset.action_dim, max_candidates=int(args.max_candidates))
        space = _search_space(args, candidates=candidates, dataset=dataset, seed=int(cell.seed) + 101)
        truth_rows = _fit_candidate_truth_rows(dataset=dataset, cell=cell, candidates=candidates, args=args)
        candidate_truth_rows.extend(truth_rows)
        truth_by_id = {str(row["candidate_id"]): row for row in truth_rows if not row.get("fit_error")}

        for selector in _selectors(args):
            if selector == "oracle_best":
                selector_rows.append(_oracle_selector_row(cell=cell, truth_by_id=truth_by_id))
                continue
            start = time.perf_counter()
            try:
                selected_id, tuning_result = _select_candidate(
                    selector=selector,
                    dataset=dataset,
                    space=space,
                    args=args,
                    seed=int(cell.seed) + _selector_seed_offset(selector),
                )
                error = ""
            except Exception as exc:
                selected_id = ""
                tuning_result = None
                error = f"{type(exc).__name__}: {exc}"
            runtime = float(time.perf_counter() - start)
            selector_rows.append(
                _selector_row(
                    cell=cell,
                    selector=selector,
                    selected_id=selected_id,
                    truth_by_id=truth_by_id,
                    runtime_sec=runtime,
                    error=error,
                    selection_source="fixed_baseline" if selector == "stable_baseline" else "proxy_cv",
                )
            )
            if tuning_result is not None:
                stage_rows.extend(_stage_rows_from_tuning(tuning_result, cell=cell, selector=selector))

        _write_csv(out_dir / "selector_rows.csv", selector_rows)
        _write_csv(out_dir / "candidate_truth_rows.csv", candidate_truth_rows)
        _write_csv(out_dir / "stage_rows.csv", stage_rows)
        _write_csv(out_dir / "skipped_cells.csv", skipped_rows)

    aggregate_rows = _aggregate_by_strategy(selector_rows)
    paired_rows = _paired_deltas(selector_rows, baseline_selector="staged_k3")
    ci_rows = _bootstrap_ci(selector_rows, iterations=int(args.analysis_bootstrap), seed=int(args.analysis_seed))
    guardrail_rows = _family_guardrails(selector_rows, aggregate_rows, target_selector="staged_k3")
    promotion = _promotion_decision(selector_rows, aggregate_rows, guardrail_rows, target_selector="staged_k3")
    summary = _run_summary(cells=cells, selector_rows=selector_rows, candidate_truth_rows=candidate_truth_rows, skipped_rows=skipped_rows, args=args)

    _write_csv(out_dir / "aggregate_by_strategy.csv", aggregate_rows)
    _write_csv(out_dir / "paired_deltas.csv", paired_rows)
    _write_csv(out_dir / "bootstrap_ci.csv", ci_rows)
    _write_csv(out_dir / "family_guardrails.csv", guardrail_rows)
    _write_csv(out_dir / "skipped_cells.csv", skipped_rows)
    _write_promotion_decision(out_dir, promotion)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(aggregate_rows, promotion)
    return {
        "selector_rows": selector_rows,
        "candidate_truth_rows": candidate_truth_rows,
        "stage_rows": stage_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "bootstrap_ci_rows": ci_rows,
        "family_guardrail_rows": guardrail_rows,
        "promotion_decision": promotion,
        "skipped_rows": skipped_rows,
        "run_summary": summary,
    }


def _dataset_cells(args: argparse.Namespace) -> list[DatasetCell]:
    cells: list[DatasetCell] = []
    settings = tuple(str(value) for value in args.settings if str(value).lower() != "none")
    controlled_sizes = tuple(int(value) for value in args.controlled_sample_sizes)
    controlled_gammas = tuple(float(value) for value in args.controlled_gammas)
    seeds = tuple(int(value) for value in args.seeds)
    linear_shifts = tuple(float(value) for value in args.linear_policy_shifts)
    gym_sizes = tuple(int(value) for value in args.gym_sample_sizes)
    gym_gammas = tuple(float(value) for value in args.gym_gammas)
    gym_seeds = tuple(int(value) for value in args.gym_seeds)
    for setting in settings:
        if setting in GYM_SETTINGS:
            for n in gym_sizes:
                for gamma in gym_gammas:
                    for seed in gym_seeds:
                        cells.append(
                            DatasetCell(
                                setting=setting,
                                sample_size=n,
                                gamma=gamma,
                                seed=seed,
                                target_value_rollouts=int(args.gym_target_value_rollouts),
                            )
                        )
        elif setting == "linear_gaussian":
            for n in controlled_sizes:
                for gamma in controlled_gammas:
                    for shift in linear_shifts:
                        for seed in seeds:
                            cells.append(DatasetCell(setting=setting, sample_size=n, gamma=gamma, seed=seed, policy_shift=shift))
        else:
            for n in controlled_sizes:
                for gamma in controlled_gammas:
                    for seed in seeds:
                        cells.append(DatasetCell(setting=setting, sample_size=n, gamma=gamma, seed=seed))
    return sorted(cells, key=lambda cell: (_is_runtime_drop_cell(cell), cell.family, cell.setting, cell.sample_size, cell.gamma, cell.seed, cell.policy_shift or 0.0))


def _make_cell_dataset(cell: DatasetCell, args: argparse.Namespace) -> BenchmarkDataset:
    cfg = OccupancyRatioBenchmarkConfig(
        profile="smoke" if args.profile == "smoke" else "medium",
        settings=(cell.setting,),
        sample_sizes=(int(cell.sample_size),),
        gammas=(float(cell.gamma),),
        seeds=(int(cell.seed),),
        linear_gaussian_policy_shifts=(1.0 if cell.policy_shift is None else float(cell.policy_shift),),
        mc_truth_samples=int(args.mc_truth_samples),
        gym_target_value_rollouts=int(cell.target_value_rollouts or args.gym_target_value_rollouts),
        source_state_correction_mode=str(args.source_state_correction_mode),
        write_plots=False,
    )
    return make_dataset(
        setting=cell.setting,
        gamma=float(cell.gamma),
        sample_size=int(cell.sample_size),
        seed=int(cell.seed),
        config=cfg,
        policy_shift=cell.policy_shift,
    )


def _candidate_grid(*, input_dim: int, max_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (label, dims) in enumerate(SIZE_GRID[: int(max_candidates)]):
        params = _mlp_parameter_count(int(input_dim), dims, output_dim=1)
        rows.append(
            {
                "occupancy": {"hidden_dims": dims},
                "_meta": {
                    "complexity_group": "fori_neural_occupancy_param_ladder",
                    "complexity_rank": params,
                    "size_label": label,
                },
            }
        )
    return rows


def _search_space(
    args: argparse.Namespace,
    *,
    candidates: Sequence[dict[str, Any]],
    dataset: BenchmarkDataset,
    seed: int,
) -> OccupancySearchSpace:
    occupancy = _occupancy_config(args, hidden_dims=tuple(args.stable_hidden_dims), seed=seed)
    action, source, transition = _nuisance_configs(args, seed=seed, hidden_dims=tuple(args.nuisance_hidden_dims))
    return OccupancySearchSpace(
        neural_occupancy=occupancy,
        neural_action_ratio=action,
        neural_source_state_ratio=source,
        neural_transition_ratio=transition,
        neural_candidates=tuple(dict(row) for row in candidates),
    )


def _fit_candidate_truth_rows(
    *,
    dataset: BenchmarkDataset,
    cell: DatasetCell,
    candidates: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    initial_states, initial_actions, initial_weights = _initial_ratio_inputs(dataset, args)
    for idx, overrides in enumerate(candidates):
        candidate_id = f"neural_{idx:03d}"
        start = time.perf_counter()
        fit_error = ""
        diagnostics: dict[str, Any] = {}
        try:
            occupancy = _candidate_occupancy_config(args, overrides=overrides, seed=int(cell.seed) + 50_000 + 1_003 * idx)
            action, source, transition = _nuisance_configs(
                args,
                seed=int(cell.seed) + 60_000 + 1_003 * idx,
                hidden_dims=tuple(args.nuisance_hidden_dims),
            )
            model = fit_discounted_occupancy_ratio_neural(
                states=dataset.states,
                actions=dataset.actions,
                next_states=dataset.next_states,
                target_actions=dataset.target_actions,
                gamma=float(dataset.gamma),
                initial_states=initial_states,
                initial_actions=initial_actions,
                initial_weights=initial_weights,
                target_next_actions=dataset.next_target_actions,
                occupancy=occupancy,
                action_ratio=action,
                source_state_ratio=source,
                transition_ratio=transition,
                initial_ratio_mode="auto",
                one_step_ratio_mode="auto",
            )
            raw = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=False)
            weights = model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True)
            diagnostics = _candidate_diagnostics(dataset, weights=weights, raw=raw, model_diagnostics=model.diagnostics)
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"
            diagnostics = _empty_candidate_diagnostics()
        rows.append(
            {
                **_cell_fields(cell),
                **_size_fields(candidate_id),
                "candidate_id": candidate_id,
                **diagnostics,
                "runtime_sec": float(time.perf_counter() - start),
                "fit_error": fit_error,
            }
        )
    return rows


def _occupancy_config(args: argparse.Namespace, *, hidden_dims: Sequence[int], seed: int) -> NeuralOccupancyRegressionConfig:
    return NeuralOccupancyRegressionConfig.stable_defaults(
        hidden_dims=tuple(int(width) for width in hidden_dims),
        activation=str(args.activation),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_iterations=int(args.final_iterations),
        gradient_steps_per_iteration=int(args.gradient_steps_per_iteration),
        mcmc_samples=int(args.mcmc_samples),
        validation_fraction=0.20,
        patience=max(1, int(args.patience)),
        validation_warmup_iterations=0,
        direct_adjoint_steps=int(args.direct_adjoint_steps) if args.direct_adjoint_steps is not None else None,
        direct_one_step_max_steps=int(args.direct_one_step_steps) if args.direct_one_step_steps is not None else None,
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm is not None else None,
        device="cpu",
        seed=int(seed),
        show_progress=False,
    )


def _candidate_occupancy_config(args: argparse.Namespace, *, overrides: dict[str, Any], seed: int) -> NeuralOccupancyRegressionConfig:
    occ = dict(dict(overrides).get("occupancy", {}) or {})
    return _occupancy_config(args, hidden_dims=tuple(occ.get("hidden_dims", tuple(args.stable_hidden_dims))), seed=seed)


def _nuisance_configs(
    args: argparse.Namespace,
    *,
    seed: int,
    hidden_dims: Sequence[int],
) -> tuple[NeuralActionRatioConfig, NeuralSourceStateRatioConfig, NeuralTransitionRatioConfig]:
    common = dict(
        hidden_dims=tuple(int(width) for width in hidden_dims),
        activation=str(args.activation),
        learning_rate=float(args.nuisance_learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        validation_fraction=0.20,
        patience=max(1, int(args.nuisance_patience)),
        prediction_max=float(args.nuisance_prediction_max) if args.nuisance_prediction_max is not None else None,
        density_ratio_loss="lsif",
        logistic_logit_clip=20.0,
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm is not None else None,
        device="cpu",
    )
    action = NeuralActionRatioConfig.stable_defaults(max_steps=int(args.action_steps), seed=int(seed + 7_001), **common)
    source = NeuralSourceStateRatioConfig.stable_defaults(max_steps=int(args.source_steps), seed=int(seed + 9_001), **common)
    transition = NeuralTransitionRatioConfig.stable_defaults(
        max_steps=int(args.transition_steps),
        permutation_samples=int(args.transition_permutation_samples),
        seed=int(seed + 8_001),
        **common,
    )
    return action, source, transition


def _select_candidate(
    *,
    selector: str,
    dataset: BenchmarkDataset,
    space: OccupancySearchSpace,
    args: argparse.Namespace,
    seed: int,
) -> tuple[str, Any]:
    if selector == "stable_baseline":
        return _stable_baseline_candidate_id(int(args.max_candidates), tuple(args.stable_hidden_dims)), None
    staged = selector.startswith("staged_k")
    cfg = OccupancyTuningConfig(
        families=("neural",),
        cv_folds=int(args.cv_folds),
        seed=int(seed),
        budget="balanced",
        max_candidates=int(args.max_candidates),
        promotion_candidates=int(args.max_candidates),
        refit=False,
        screen_fraction=1.0,
        stable_fallback=False,
        score_method=_score_method(selector),
        gmm_objective=_gmm_objective(selector),
        staged_bootstrap_cv=staged,
        staged_cv_iterations=_selector_k(selector) if staged else 3,
        staged_cv_n_bootstrap=int(args.bootstrap),
        staged_cv_min_survivors=1,
        staged_cv_loss_metric="validation_loss",
    )
    initial_states, initial_actions, initial_weights = _initial_ratio_inputs(dataset, args)
    result = tune_occupancy_ratio(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        gamma=float(dataset.gamma),
        initial_states=initial_states,
        initial_actions=initial_actions,
        initial_weights=initial_weights,
        target_next_actions=dataset.next_target_actions,
        rewards=dataset.rewards,
        search_space=space,
        config=cfg,
        initial_ratio_mode="auto",
        one_step_ratio_mode="auto",
    )
    return str(result.selected_candidate_id), result


def _candidate_diagnostics(
    dataset: BenchmarkDataset,
    *,
    weights: np.ndarray,
    raw: np.ndarray,
    model_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    reward_arr = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    diagnostics = estimator_diagnostics_optional(
        true_ratio=dataset.true_ratio,
        estimated_ratio=weights_arr,
        raw_ratio=raw_arr,
        reference_weights=dataset.reference_weights,
        feature_matrix=_diagnostic_features(dataset),
    )
    diagnostics.update(_value_diagnostics(dataset, weights_arr, reward_arr))
    diagnostics["near_uniform_collapse"] = _near_uniform_collapse(dataset, diagnostics)
    for key in (
        "initial_ratio_mode",
        "one_step_ratio_mode",
        "source_state_ratio_enabled",
        "source_state_ratio_ess_fraction",
        "source_state_ratio_max",
        "source_state_ratio_loss",
        "initial_joint_ratio_enabled",
        "initial_joint_ratio_ess_fraction",
        "initial_joint_ratio_max",
        "initial_joint_ratio_loss",
        "one_step_direct_ratio_enabled",
    ):
        diagnostics[key] = model_diagnostics.get(key, "")
    return diagnostics


def _empty_candidate_diagnostics() -> dict[str, Any]:
    keys = (
        "ope_value_estimate",
        "ope_value_target",
        "ope_value_abs_error",
        "ope_value_abs_error_se_units",
        "ratio_normalized_l1",
        "log_ratio_rmse",
        "effective_sample_size_fraction",
        "weight_cv",
        "weight_q99_to_median",
        "clipping_fraction",
        "near_uniform_collapse",
    )
    return {key: float("nan") for key in keys}


def _value_diagnostics(dataset: BenchmarkDataset, weights: np.ndarray, rewards: np.ndarray) -> dict[str, Any]:
    estimated = float(np.mean(weights * rewards))
    if dataset.true_ratio is not None:
        target = float(np.mean(np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1) * rewards))
        return {
            "ope_value_estimate": estimated,
            "ope_value_target": target,
            "ope_value_abs_error": float(abs(estimated - target)),
            "ope_value_abs_error_se_units": float("nan"),
        }
    if "target_policy_value" not in dataset.metadata:
        return {
            "ope_value_estimate": estimated,
            "ope_value_target": float("nan"),
            "ope_value_abs_error": float("inf"),
            "ope_value_abs_error_se_units": float("nan"),
        }
    target = float(dataset.metadata["target_policy_value"])
    se = _to_float(dataset.metadata.get("target_policy_value_se"))
    abs_error = float(abs(estimated - target))
    return {
        "ope_value_estimate": estimated,
        "ope_value_target": target,
        "ope_value_abs_error": abs_error,
        "ope_value_abs_error_se_units": abs_error / max(se, 1e-12) if np.isfinite(se) and se > 0.0 else float("nan"),
    }


def _selector_row(
    *,
    cell: DatasetCell,
    selector: str,
    selected_id: str,
    truth_by_id: dict[str, dict[str, Any]],
    runtime_sec: float,
    error: str,
    selection_source: str,
) -> dict[str, Any]:
    selected_truth = truth_by_id.get(str(selected_id), {})
    oracle_error = _oracle_error(truth_by_id)
    selected_error = _to_float(selected_truth.get("ope_value_abs_error"), float("inf"))
    payload = {
        **_cell_fields(cell),
        **_size_fields(str(selected_id)),
        "selector": selector,
        "selected_candidate_id": str(selected_id),
        "selection_source": selection_source,
        "ope_value_estimate": selected_truth.get("ope_value_estimate", float("nan")),
        "ope_value_target": selected_truth.get("ope_value_target", float("nan")),
        "ope_value_abs_error": selected_error,
        "oracle_regret": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else float("inf"),
        "ratio_normalized_l1": selected_truth.get("ratio_normalized_l1", float("nan")),
        "log_ratio_rmse": selected_truth.get("log_ratio_rmse", float("nan")),
        "effective_sample_size_fraction": selected_truth.get("effective_sample_size_fraction", float("nan")),
        "weight_cv": selected_truth.get("weight_cv", float("nan")),
        "weight_q99_to_median": selected_truth.get("weight_q99_to_median", float("nan")),
        "clipping_fraction": selected_truth.get("clipping_fraction", float("nan")),
        "near_uniform_collapse": selected_truth.get("near_uniform_collapse", float("nan")),
        "ope_value_abs_error_se_units": selected_truth.get("ope_value_abs_error_se_units", float("nan")),
        "runtime_sec": float(runtime_sec),
        "selection_error": error,
    }
    for key in (
        "initial_ratio_mode",
        "one_step_ratio_mode",
        "source_state_ratio_enabled",
        "source_state_ratio_ess_fraction",
        "source_state_ratio_max",
        "source_state_ratio_loss",
        "initial_joint_ratio_enabled",
        "initial_joint_ratio_ess_fraction",
        "initial_joint_ratio_max",
        "initial_joint_ratio_loss",
    ):
        payload[key] = selected_truth.get(key, "")
    return payload


def _oracle_selector_row(*, cell: DatasetCell, truth_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not truth_by_id:
        return _selector_row(
            cell=cell,
            selector="oracle_best",
            selected_id="",
            truth_by_id=truth_by_id,
            runtime_sec=0.0,
            error="no_finite_candidate_truth",
            selection_source="reporting_only",
        )
    selected_id = min(truth_by_id, key=lambda cid: _to_float(truth_by_id[cid].get("ope_value_abs_error"), float("inf")))
    return _selector_row(
        cell=cell,
        selector="oracle_best",
        selected_id=selected_id,
        truth_by_id=truth_by_id,
        runtime_sec=0.0,
        error="",
        selection_source="reporting_only",
    )


def _stage_rows_from_tuning(tuning_result: Any, *, cell: DatasetCell, selector: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hasattr(tuning_result, "staged_cv_candidate_rows"):
        for row in tuning_result.staged_cv_candidate_rows():
            payload = dict(row)
            payload.update(_cell_fields(cell))
            payload["selector"] = selector
            payload["row_type"] = "candidate_stage"
            payload.update(_size_fields(str(payload.get("candidate_id", ""))))
            rows.append(payload)
    if hasattr(tuning_result, "staged_cv_fold_rows"):
        for row in tuning_result.staged_cv_fold_rows():
            payload = dict(row)
            payload.update(_cell_fields(cell))
            payload["selector"] = selector
            payload["row_type"] = "fold_stage"
            payload.update(_size_fields(str(payload.get("candidate_id", ""))))
            rows.append(payload)
    return rows


def _aggregate_by_strategy(selector_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selector_rows if str(row.get("selector", "")) and str(row.get("selector")) != "oracle_best"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset_family"]), str(row["selector"])), []).append(row)
        groups.setdefault(("all", str(row["selector"])), []).append(row)
    win_keys = _winning_keys(rows)
    out: list[dict[str, Any]] = []
    for (family, selector), group in sorted(groups.items()):
        errors = _finite_values(row.get("ope_value_abs_error") for row in group)
        regrets = _finite_values(row.get("oracle_regret") for row in group)
        gym_norm = _finite_values(row.get("ope_value_abs_error_se_units") for row in group)
        runtimes = _finite_values(row.get("runtime_sec") for row in group)
        out.append(
            {
                "dataset_family": family,
                "selector": selector,
                "n_cells": len(group),
                "mean_ope_abs_error": _mean(errors),
                "median_ope_abs_error": _median(errors),
                "mean_oracle_regret": _mean(regrets),
                "median_oracle_regret": _median(regrets),
                "mean_gym_normalized_error": _mean(gym_norm),
                "median_gym_normalized_error": _median(gym_norm),
                "mean_runtime_sec": _mean(runtimes),
                "mean_ratio_normalized_l1": _mean(_finite_values(row.get("ratio_normalized_l1") for row in group)),
                "mean_log_ratio_rmse": _mean(_finite_values(row.get("log_ratio_rmse") for row in group)),
                "mean_ess_fraction": _mean(_finite_values(row.get("effective_sample_size_fraction") for row in group)),
                "mean_clipping_fraction": _mean(_finite_values(row.get("clipping_fraction") for row in group)),
                "near_uniform_collapse_rate": _mean(_finite_values(row.get("near_uniform_collapse") for row in group)),
                "win_count": sum((str(row["cell_id"]), str(selector)) in win_keys for row in group),
                "oracle_match_count": sum(_to_float(row.get("oracle_regret"), float("inf")) <= 1e-12 for row in group),
                "selected_complexities": ",".join(str(row.get("selected_hidden_dims", "")) for row in group),
            }
        )
    return out


def _paired_deltas(selector_rows: Sequence[dict[str, Any]], *, baseline_selector: str) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for row in selector_rows:
        selector = str(row.get("selector", ""))
        if not selector or selector == "oracle_best":
            continue
        by_cell.setdefault(str(row["cell_id"]), {})[selector] = row
    out: list[dict[str, Any]] = []
    for cell_id, rows in sorted(by_cell.items()):
        baseline = rows.get(baseline_selector)
        if baseline is None:
            continue
        for selector, row in sorted(rows.items()):
            if selector == baseline_selector:
                continue
            out.append(
                {
                    "cell_id": cell_id,
                    "setting": row.get("setting", ""),
                    "dataset_family": row.get("dataset_family", ""),
                    "selector": selector,
                    "baseline_selector": baseline_selector,
                    "oracle_regret_delta_vs_baseline": _to_float(row.get("oracle_regret")) - _to_float(baseline.get("oracle_regret")),
                    "ope_abs_error_delta_vs_baseline": _to_float(row.get("ope_value_abs_error")) - _to_float(baseline.get("ope_value_abs_error")),
                    "runtime_delta_vs_baseline": _to_float(row.get("runtime_sec")) - _to_float(baseline.get("runtime_sec")),
                    "selector_worse_on_regret": float(_to_float(row.get("oracle_regret")) > _to_float(baseline.get("oracle_regret")) + 1e-12),
                }
            )
    return out


def _bootstrap_ci(selector_rows: Sequence[dict[str, Any]], *, iterations: int, seed: int) -> list[dict[str, Any]]:
    rows = [row for row in selector_rows if str(row.get("selector", "")) not in {"", "oracle_best"}]
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for family in sorted({str(row.get("dataset_family", "")) for row in rows} | {"all"}):
        family_rows = rows if family == "all" else [row for row in rows if str(row.get("dataset_family", "")) == family]
        for selector in sorted({str(row.get("selector", "")) for row in family_rows}):
            group = [row for row in family_rows if str(row.get("selector", "")) == selector]
            for metric in ("oracle_regret", "ope_value_abs_error", "ope_value_abs_error_se_units"):
                values = _finite_values(row.get(metric) for row in group)
                if not values:
                    continue
                boot = _bootstrap_stat(values, iterations=max(0, int(iterations)), rng=rng, stat=np.median)
                out.append(
                    {
                        "dataset_family": family,
                        "selector": selector,
                        "metric": metric,
                        "n": len(values),
                        "estimate": float(np.median(values)),
                        "ci_low": float(np.quantile(boot, 0.025)) if boot else float("nan"),
                        "ci_high": float(np.quantile(boot, 0.975)) if boot else float("nan"),
                        "bootstrap_iterations": max(0, int(iterations)),
                    }
                )
    return out


def _family_guardrails(
    selector_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    *,
    target_selector: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_by_family = {
        str(row["dataset_family"]): row
        for row in aggregate_rows
        if str(row.get("selector", "")) == target_selector
    }
    for family in sorted({str(row["dataset_family"]) for row in aggregate_rows if str(row["dataset_family"]) != "all"}):
        family_aggs = [row for row in aggregate_rows if str(row.get("dataset_family", "")) == family]
        target = target_by_family.get(family)
        if target is None or not family_aggs:
            continue
        metric = "median_gym_normalized_error" if family == "gym" else "mean_oracle_regret"
        multiplier = 1.25 if family == "gym" else 1.10 if family in {"discrete", "linear_gaussian", "nonlinear"} else 1.25
        values = [_to_float(row.get(metric)) for row in family_aggs]
        values = [value for value in values if np.isfinite(value)]
        if not values:
            continue
        best = min(values)
        observed = _to_float(target.get(metric))
        threshold = best * multiplier if best > 0 else best + 1e-12
        rows.append(
            {
                "guardrail": f"{family}_{metric}",
                "dataset_family": family,
                "metric": metric,
                "target_selector": target_selector,
                "target_value": observed,
                "best_value": best,
                "threshold": threshold,
                "passed": float(np.isfinite(observed) and observed <= threshold + 1e-12),
            }
        )
    diagnostics = {
        "failure_rate": (_failure_rate_by_selector(selector_rows), 0.02, min),
        "near_uniform_collapse_rate": (_rate_by_selector(selector_rows, "near_uniform_collapse"), 0.02, min),
        "clipping_fraction": (_mean_by_selector(selector_rows, "clipping_fraction"), 0.02, min),
    }
    for metric, (values, tolerance, reducer) in diagnostics.items():
        if target_selector not in values:
            continue
        comparators = [name for name in ("product_composite_cv", "bellman_gmm_ratio", "bellman_gmm_ope") if name in values]
        if not comparators:
            continue
        best = reducer(values[name] for name in comparators)
        target = values[target_selector]
        rows.append(
            {
                "guardrail": metric,
                "dataset_family": "all",
                "metric": metric,
                "target_selector": target_selector,
                "target_value": target,
                "best_value": best,
                "threshold": best + tolerance,
                "passed": float(target <= best + tolerance + 1e-12),
            }
        )
    return rows


def _promotion_decision(
    selector_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    guardrail_rows: Sequence[dict[str, Any]],
    *,
    target_selector: str,
) -> dict[str, Any]:
    overall = {str(row["selector"]): row for row in aggregate_rows if str(row.get("dataset_family")) == "all"}
    target = overall.get(target_selector)
    required_names = ("product_composite_cv", "bellman_gmm_ratio", "bellman_gmm_ope")
    missing = [name for name in required_names if name not in overall]
    if target is None or missing:
        return {
            "promote": False,
            "target_selector": target_selector,
            "reason": "missing required selector rows",
            "missing_required_selectors": missing,
        }
    required = [overall[name] for name in required_names]
    target_median = _to_float(target.get("median_oracle_regret"))
    required_best = min(_to_float(row.get("median_oracle_regret")) for row in required)
    conditions = {
        "beats_product_and_gmm_median_regret": bool(target_median <= required_best + 1e-12),
        "guardrails_pass": bool(guardrail_rows and all(_to_float(row.get("passed"), 0.0) > 0.5 for row in guardrail_rows)),
    }
    return {
        "promote": bool(all(conditions.values())),
        "target_selector": target_selector,
        "conditions": conditions,
        "median_oracle_regret": target_median,
        "best_product_or_gmm_median_oracle_regret": required_best,
        "decision_rule": "FORI staged CV promotes only if it beats product composite and Bellman-GMM without worse collapse/clipping/source diagnostics.",
    }


def _write_promotion_decision(out_dir: Path, promotion: dict[str, Any]) -> None:
    (out_dir / "promotion_decision.json").write_text(json.dumps(promotion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FORI Staged-CV Promotion Decision",
        "",
        f"Promote default: `{bool(promotion.get('promote', False))}`",
        f"Target selector: `{promotion.get('target_selector', '')}`",
        "",
        f"Median oracle regret: {promotion.get('median_oracle_regret', float('nan'))}",
        f"Best product/GMM median oracle regret: {promotion.get('best_product_or_gmm_median_oracle_regret', float('nan'))}",
        "",
        "Conditions:",
    ]
    for key, value in sorted(dict(promotion.get("conditions", {})).items()):
        lines.append(f"- `{key}`: {bool(value)}")
    lines.extend(["", f"Decision rule: {promotion.get('decision_rule', '')}"])
    (out_dir / "promotion_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_summary(
    *,
    cells: Sequence[DatasetCell],
    selector_rows: Sequence[dict[str, Any]],
    candidate_truth_rows: Sequence[dict[str, Any]],
    skipped_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    completed_cell_ids = sorted({str(row["cell_id"]) for row in candidate_truth_rows})
    selector_names = tuple(str(selector) for selector in _selectors(args))
    expected_selector_rows = len(completed_cell_ids) * len(selector_names)
    expected_candidate_rows = len(completed_cell_ids) * int(args.max_candidates)
    return {
        "configured_cells": len(cells),
        "completed_cells": len(completed_cell_ids),
        "skipped_cells": len(skipped_rows),
        "selectors": list(selector_names),
        "expected_selector_rows": expected_selector_rows,
        "actual_selector_rows": len(selector_rows),
        "selector_rows_complete": len(selector_rows) == expected_selector_rows,
        "expected_candidate_truth_rows": expected_candidate_rows,
        "actual_candidate_truth_rows": len(candidate_truth_rows),
        "candidate_truth_rows_complete": len(candidate_truth_rows) == expected_candidate_rows,
    }


def _initial_ratio_inputs(dataset: BenchmarkDataset, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    mode = str(args.source_state_correction_mode)
    if mode == "never":
        return None, None, None
    if mode == "auto" and dataset.setting in CONTROLLED_SETTINGS:
        return None, None, None
    return dataset.initial_states, dataset.initial_actions, dataset.initial_weights


def _diagnostic_features(dataset: BenchmarkDataset) -> np.ndarray:
    states = np.asarray(dataset.states, dtype=np.float64).reshape(dataset.n, -1)
    actions = np.asarray(dataset.actions, dtype=np.float64).reshape(dataset.n, -1)
    return np.column_stack([np.ones(dataset.n, dtype=np.float64), states, actions])


def _near_uniform_collapse(dataset: BenchmarkDataset, diagnostics: dict[str, Any]) -> float:
    if dataset.true_ratio is None:
        return float("nan")
    truth = np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)
    truth_cv = float(np.std(truth) / max(abs(float(np.mean(truth))), 1e-12))
    pred_cv = _to_float(diagnostics.get("weight_cv"))
    return float(truth_cv >= 0.10 and np.isfinite(pred_cv) and pred_cv <= 0.05)


def _cell_fields(cell: DatasetCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "setting": cell.setting,
        "dataset_family": cell.family,
        "sample_size": int(cell.sample_size),
        "gamma": float(cell.gamma),
        "seed": int(cell.seed),
        "policy_shift": "" if cell.policy_shift is None else float(cell.policy_shift),
    }


def _size_fields(candidate_id: str) -> dict[str, Any]:
    idx = _candidate_index(candidate_id)
    if idx < 0 or idx >= len(SIZE_GRID):
        return {"selected_size_label": "", "selected_hidden_dims": "", "selected_complexity_rank": float("nan")}
    label, dims = SIZE_GRID[idx]
    return {
        "selected_size_label": label,
        "selected_hidden_dims": _dims_str(dims),
        "selected_complexity_rank": idx + 1,
    }


def _candidate_index(candidate_id: str) -> int:
    try:
        return int(str(candidate_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _stable_baseline_candidate_id(max_candidates: int, stable_hidden_dims: Sequence[int]) -> str:
    stable_dims = tuple(int(width) for width in stable_hidden_dims)
    for idx, (_, dims) in enumerate(SIZE_GRID[: int(max_candidates)]):
        if tuple(dims) == stable_dims:
            return f"neural_{idx:03d}"
    return "neural_000"


def _selectors(args: argparse.Namespace) -> tuple[str, ...]:
    if args.selectors:
        return tuple(str(selector) for selector in args.selectors)
    staged = tuple(f"staged_k{int(k)}" for k in args.stage_counts)
    return (*staged, "naive_final_bellman_cv", "product_composite_cv", "bellman_gmm_ratio", "bellman_gmm_ope", "stable_baseline", "oracle_best")


def _selector_k(selector: str) -> int:
    return int(str(selector).replace("staged_k", ""))


def _score_method(selector: str) -> str:
    if selector.startswith("staged_k") or selector == "naive_final_bellman_cv":
        return "validation_loss"
    if selector in {"bellman_gmm_ratio", "bellman_gmm_ope"}:
        return "bellman_gmm"
    return "legacy_rank"


def _gmm_objective(selector: str) -> str:
    return "ope" if selector == "bellman_gmm_ope" else "ratio"


def _selector_seed_offset(selector: str) -> int:
    return 1_000 + sum((idx + 1) * ord(char) for idx, char in enumerate(str(selector)))


def _oracle_error(truth_by_id: dict[str, dict[str, Any]]) -> float:
    if not truth_by_id:
        return float("inf")
    return min(_to_float(row.get("ope_value_abs_error"), float("inf")) for row in truth_by_id.values())


def _winning_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(row)
    keys: set[tuple[str, str]] = set()
    for cell_id, group in by_cell.items():
        finite = [row for row in group if np.isfinite(_to_float(row.get("ope_value_abs_error"), float("inf")))]
        if not finite:
            continue
        best = min(_to_float(row["ope_value_abs_error"]) for row in finite)
        for row in finite:
            if _to_float(row["ope_value_abs_error"]) <= best + 1e-12:
                keys.add((cell_id, str(row["selector"])))
    return keys


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        numeric = _to_float(value)
        if np.isfinite(numeric):
            out.append(numeric)
    return out


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _bootstrap_stat(values: Sequence[float], *, iterations: int, rng: np.random.Generator, stat: Any) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    if iterations <= 0:
        return [float(stat(arr))]
    out: list[float] = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, arr.size, size=arr.size)
        out.append(float(stat(arr[idx])))
    return out


def _failure_rate_by_selector(selector_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    selectors = sorted({str(row.get("selector", "")) for row in selector_rows if str(row.get("selector", "")) != "oracle_best"})
    for selector in selectors:
        rows = [row for row in selector_rows if str(row.get("selector", "")) == selector]
        if rows:
            out[selector] = float(np.mean([bool(str(row.get("selection_error", ""))) or not np.isfinite(_to_float(row.get("oracle_regret"), float("inf"))) for row in rows]))
    return out


def _rate_by_selector(selector_rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    selectors = sorted({str(row.get("selector", "")) for row in selector_rows if str(row.get("selector", "")) != "oracle_best"})
    for selector in selectors:
        values = _finite_values(row.get(key) for row in selector_rows if str(row.get("selector", "")) == selector)
        if values:
            out[selector] = float(np.mean(values))
    return out


def _mean_by_selector(selector_rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return _rate_by_selector(selector_rows, key)


def _mlp_parameter_count(input_dim: int, hidden_dims: Sequence[int], *, output_dim: int) -> float:
    dims = [int(input_dim), *(int(width) for width in hidden_dims), int(output_dim)]
    total = 0
    for left, right in zip(dims[:-1], dims[1:]):
        total += left * right + right
    return float(total)


def _dims_str(dims: Iterable[int]) -> str:
    return "x".join(str(int(dim)) for dim in dims)


def _is_runtime_drop_cell(cell: DatasetCell) -> bool:
    return bool((cell.family != "gym" and int(cell.sample_size) >= 2048 and abs(float(cell.gamma) - 0.7) < 1e-12) or (cell.family == "gym" and int(cell.sample_size) >= 4096))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(aggregate_rows: Sequence[dict[str, Any]], promotion: dict[str, Any]) -> None:
    print("dataset_family\tselector\tn\tmedian_regret\tmedian_ope_abs_error\tmean_runtime")
    for row in aggregate_rows:
        if str(row["dataset_family"]) != "all":
            continue
        print(
            "\t".join(
                [
                    str(row["dataset_family"]),
                    str(row["selector"]),
                    str(row["n_cells"]),
                    f"{float(row['median_oracle_regret']):.6g}",
                    f"{float(row['median_ope_abs_error']):.6g}",
                    f"{float(row['mean_runtime_sec']):.2f}",
                ]
            )
        )
    print(f"promote\t{bool(promotion.get('promote', False))}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FORI neural CV selection strategies.")
    parser.add_argument("--profile", choices=("smoke", "core_realistic", "overnight"), default=None)
    parser.add_argument("--output-dir", default="outputs/fori_cv_strategy_benchmark")
    parser.add_argument("--settings", nargs="+", default=("discrete_chain", "linear_gaussian", "nonlinear_monte_carlo", "gym_mountain_car_continuous"))
    parser.add_argument("--controlled-sample-sizes", nargs="+", type=int, default=(512,))
    parser.add_argument("--controlled-gammas", nargs="+", type=float, default=(0.7, 0.9))
    parser.add_argument("--linear-policy-shifts", nargs="+", type=float, default=(0.7, 1.2))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--gym-sample-sizes", nargs="+", type=int, default=(1024,))
    parser.add_argument("--gym-gammas", nargs="+", type=float, default=(0.9,))
    parser.add_argument("--gym-seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--gym-target-value-rollouts", type=int, default=24)
    parser.add_argument("--mc-truth-samples", type=int, default=8_000)
    parser.add_argument("--source-state-correction-mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--stage-counts", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--selectors", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=len(SIZE_GRID))
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--final-iterations", type=int, default=10)
    parser.add_argument("--gradient-steps-per-iteration", type=int, default=4)
    parser.add_argument("--mcmc-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stable-hidden-dims", nargs="+", type=int, default=(64, 64))
    parser.add_argument("--nuisance-hidden-dims", nargs="+", type=int, default=(32, 32))
    parser.add_argument("--activation", choices=("relu", "tanh", "silu", "gelu"), default="silu")
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--nuisance-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--nuisance-patience", type=int, default=3)
    parser.add_argument("--action-steps", type=int, default=120)
    parser.add_argument("--source-steps", type=int, default=120)
    parser.add_argument("--transition-steps", type=int, default=160)
    parser.add_argument("--transition-permutation-samples", type=int, default=3)
    parser.add_argument("--direct-adjoint-steps", type=int, default=32)
    parser.add_argument("--direct-one-step-steps", type=int, default=32)
    parser.add_argument("--nuisance-prediction-max", type=float, default=50.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--time-budget-minutes", type=float, default=60.0)
    parser.add_argument("--analysis-bootstrap", type=int, default=500)
    parser.add_argument("--analysis-seed", type=int, default=91_337)
    args = parser.parse_args(argv)
    return _apply_profile_defaults(args, argv)


def _apply_profile_defaults(args: argparse.Namespace, argv: Sequence[str] | None) -> argparse.Namespace:
    if args.profile is None:
        return args
    explicit = _explicit_options(argv)
    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "output_dir": "outputs/fori_cv_strategy_smoke",
            "settings": ("discrete_chain",),
            "controlled_sample_sizes": (48,),
            "controlled_gammas": (0.0,),
            "seeds": (0,),
            "selectors": ("staged_k1", "staged_k3", "naive_final_bellman_cv", "bellman_gmm_ratio", "oracle_best"),
            "max_candidates": 2,
            "cv_folds": 2,
            "bootstrap": 0,
            "final_iterations": 1,
            "gradient_steps_per_iteration": 1,
            "mcmc_samples": 2,
            "batch_size": 32,
            "nuisance_hidden_dims": (8,),
            "action_steps": 5,
            "source_steps": 5,
            "transition_steps": 5,
            "transition_permutation_samples": 1,
            "direct_adjoint_steps": 4,
            "direct_one_step_steps": 4,
            "analysis_bootstrap": 20,
            "mc_truth_samples": 200,
            "time_budget_minutes": 20.0,
        },
        "core_realistic": {
            "output_dir": "outputs/fori_cv_strategy_core_realistic",
            "settings": ("discrete_chain", "discrete_grid", "linear_gaussian", "nonlinear_monte_carlo", "gym_pendulum", "gym_mountain_car_continuous"),
            "controlled_sample_sizes": (512, 2048),
            "controlled_gammas": (0.7, 0.9, 0.95),
            "linear_policy_shifts": (0.7, 1.2, 2.0),
            "seeds": tuple(range(4)),
            "gym_sample_sizes": (1024,),
            "gym_gammas": (0.9,),
            "gym_seeds": tuple(range(4)),
            "gym_target_value_rollouts": 64,
            "mc_truth_samples": 20_000,
            "time_budget_minutes": 8.0 * 60.0,
        },
        "overnight": {
            "output_dir": "outputs/fori_cv_strategy_overnight",
            "settings": (
                "discrete_chain",
                "discrete_grid",
                "linear_gaussian",
                "nonlinear_monte_carlo",
                "gym_pendulum",
                "gym_mountain_car_continuous",
                "gym_halfcheetah",
                "gym_hopper",
            ),
            "controlled_sample_sizes": (512, 2048),
            "controlled_gammas": (0.7, 0.9, 0.95),
            "linear_policy_shifts": (0.7, 1.2, 2.0),
            "seeds": tuple(range(8)),
            "gym_sample_sizes": (1024, 4096),
            "gym_gammas": (0.9,),
            "gym_seeds": tuple(range(8)),
            "gym_target_value_rollouts": 96,
            "mc_truth_samples": 50_000,
            "time_budget_minutes": 24.0 * 60.0,
        },
    }
    for name, value in profiles[str(args.profile)].items():
        if f"--{name.replace('_', '-')}" not in explicit:
            setattr(args, name, value)
    return args


def _explicit_options(argv: Sequence[str] | None) -> set[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    out: set[str] = set()
    for token in raw:
        if not str(token).startswith("--"):
            continue
        out.add(str(token).split("=", 1)[0])
    return out


if __name__ == "__main__":
    main()
