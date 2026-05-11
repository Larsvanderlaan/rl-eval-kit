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
sys.path.insert(0, str(ROOT / "packages" / "fqe"))

from fqe import FQESearchSpace, FQETuningConfig, NeuralFQEConfig, fit_fqe_neural, tune_fqe
from fqe_benchmark.data import make_dataset, make_gym_control_dataset
from fqe_benchmark.types import BenchmarkDataset


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


@dataclass(frozen=True)
class DatasetCell:
    dataset: str
    sample_size: int
    gamma: float
    seed: int
    policy_shift: float | None = None
    target_value_rollouts: int | None = None

    @property
    def family(self) -> str:
        if self.dataset.startswith("tabular"):
            return "discrete"
        if self.dataset == "linear_gaussian":
            return "linear_gaussian"
        return "gym"

    @property
    def cell_id(self) -> str:
        shift = "none" if self.policy_shift is None else f"{float(self.policy_shift):g}"
        return f"{self.dataset}|n={self.sample_size}|g={self.gamma:g}|seed={self.seed}|shift={shift}"


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
            skipped_rows.append({**_cell_fields(cell), "skip_reason": "time_budget_drop_2048_gamma_0.7"})
            continue
        dataset = _make_cell_dataset(cell, args)
        candidates = _candidate_grid(input_dim=dataset.state_dim + dataset.action_dim, max_candidates=int(args.max_candidates))
        space = _search_space(args, candidates=candidates, seed=int(cell.seed) + 101)
        truth_rows = _fit_candidate_truth_rows(dataset=dataset, cell=cell, candidates=candidates, args=args)
        candidate_truth_rows.extend(truth_rows)
        truth_by_id = {str(row["candidate_id"]): row for row in truth_rows if not row.get("fit_error")}

        for selector in _selectors(args):
            if selector == "oracle_best":
                row = _oracle_selector_row(cell=cell, truth_by_id=truth_by_id)
                selector_rows.append(row)
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
                    selection_source="proxy_cv",
                )
            )
            if tuning_result is not None and hasattr(tuning_result, "staged_cv_rows"):
                for row in tuning_result.staged_cv_rows():
                    if not row:
                        continue
                    payload = dict(row)
                    payload.update(_cell_fields(cell))
                    payload["selector"] = selector
                    payload.update(_size_fields(str(payload.get("candidate_id", ""))))
                    stage_rows.append(payload)

        _write_csv(out_dir / "selector_rows.csv", selector_rows)
        _write_csv(out_dir / "candidate_truth_rows.csv", candidate_truth_rows)
        _write_csv(out_dir / "stage_rows.csv", stage_rows)
        _write_csv(out_dir / "skipped_cells.csv", skipped_rows)

    aggregate_rows = _aggregate_by_strategy(selector_rows)
    recommendation = _recommend_default(selector_rows, aggregate_rows)
    paired_rows = _paired_deltas(selector_rows, baseline_selector="staged_k3")
    ci_rows = _bootstrap_ci(
        selector_rows,
        iterations=int(args.analysis_bootstrap),
        seed=int(args.analysis_seed),
    )
    guardrail_rows = _family_guardrails(selector_rows, aggregate_rows, target_selector="staged_k3")
    promotion = _promotion_decision(
        selector_rows,
        aggregate_rows,
        guardrail_rows,
        target_selector="staged_k3",
        non_stage_selectors=("naive_final_bellman_cv", "product_composite_cv"),
    )
    summary = _run_summary(cells=cells, selector_rows=selector_rows, candidate_truth_rows=candidate_truth_rows, skipped_rows=skipped_rows, args=args)
    _write_csv(out_dir / "aggregate_by_strategy.csv", aggregate_rows)
    _write_csv(out_dir / "paired_deltas.csv", paired_rows)
    _write_csv(out_dir / "bootstrap_ci.csv", ci_rows)
    _write_csv(out_dir / "family_guardrails.csv", guardrail_rows)
    _write_csv(out_dir / "skipped_cells.csv", skipped_rows)
    _write_recommendation(out_dir, recommendation)
    _write_promotion_decision(out_dir, promotion)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(aggregate_rows, recommendation)
    return {
        "selector_rows": selector_rows,
        "candidate_truth_rows": candidate_truth_rows,
        "stage_rows": stage_rows,
        "aggregate_rows": aggregate_rows,
        "paired_rows": paired_rows,
        "bootstrap_ci_rows": ci_rows,
        "family_guardrail_rows": guardrail_rows,
        "recommendation": recommendation,
        "promotion_decision": promotion,
        "skipped_rows": skipped_rows,
        "run_summary": summary,
    }


def _dataset_cells(args: argparse.Namespace) -> list[DatasetCell]:
    cells: list[DatasetCell] = []
    synthetic_sizes = tuple(int(value) for value in args.synthetic_sample_sizes)
    synthetic_gammas = tuple(float(value) for value in args.synthetic_gammas)
    synthetic_seeds = tuple(int(value) for value in args.synthetic_seeds)
    if not bool(args.skip_discrete):
        discrete_datasets = tuple(str(value) for value in args.discrete_datasets if str(value).lower() != "none")
    else:
        discrete_datasets = ()
    if not bool(args.skip_linear):
        linear_shifts = tuple(float(value) for value in args.linear_policy_shifts)
    else:
        linear_shifts = ()
    if not bool(args.skip_gym):
        gym_datasets = tuple(str(value) for value in args.gym_datasets if str(value).lower() != "none")
    else:
        gym_datasets = ()
    for dataset in discrete_datasets:
        for n in synthetic_sizes:
            for gamma in synthetic_gammas:
                for seed in synthetic_seeds:
                    cells.append(DatasetCell(dataset=dataset, sample_size=n, gamma=gamma, seed=seed))
    for n in synthetic_sizes:
        for gamma in synthetic_gammas:
            for shift in linear_shifts:
                for seed in synthetic_seeds:
                    cells.append(
                        DatasetCell(
                            dataset="linear_gaussian",
                            sample_size=n,
                            gamma=gamma,
                            seed=seed,
                            policy_shift=shift,
                        )
                    )
    gym_sizes = tuple(int(value) for value in (args.gym_sample_sizes or (args.gym_sample_size,)))
    gym_gammas = tuple(float(value) for value in (args.gym_gammas or (args.gym_gamma,)))
    for dataset in gym_datasets:
        for n in gym_sizes:
            for gamma in gym_gammas:
                for seed in tuple(int(value) for value in args.gym_seeds):
                    cells.append(
                        DatasetCell(
                            dataset=dataset,
                            sample_size=int(n),
                            gamma=float(gamma),
                            seed=seed,
                            target_value_rollouts=int(args.gym_target_value_rollouts),
                        )
                    )
    return sorted(cells, key=lambda cell: (_is_runtime_drop_cell(cell), cell.family, cell.dataset, cell.sample_size, cell.gamma, cell.seed, cell.policy_shift or 0.0))


def _make_cell_dataset(cell: DatasetCell, args: argparse.Namespace) -> BenchmarkDataset:
    if cell.family == "gym":
        return make_gym_control_dataset(
            name=cell.dataset,
            sample_size=int(cell.sample_size),
            gamma=float(cell.gamma),
            seed=int(cell.seed),
            n_eval=int(args.n_eval),
            n_initial_eval=int(args.n_initial_eval),
            target_value_rollouts=int(cell.target_value_rollouts or args.gym_target_value_rollouts),
        )
    return make_dataset(
        name=cell.dataset,
        sample_size=int(cell.sample_size),
        gamma=float(cell.gamma),
        seed=int(cell.seed),
        policy_shift=0.0 if cell.policy_shift is None else float(cell.policy_shift),
        n_eval=int(args.n_eval),
        n_initial_eval=int(args.n_initial_eval),
    )


def _candidate_grid(*, input_dim: int, max_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (label, dims) in enumerate(SIZE_GRID[: int(max_candidates)]):
        params = _mlp_parameter_count(int(input_dim), dims, output_dim=1)
        rows.append(
            {
                "hidden_dims": dims,
                "_meta": {
                    "complexity_group": "fqe_neural_param_ladder",
                    "complexity_rank": params,
                    "size_label": label,
                },
            }
        )
    return rows


def _search_space(args: argparse.Namespace, *, candidates: Sequence[dict[str, Any]], seed: int) -> FQESearchSpace:
    base = NeuralFQEConfig.stable_defaults(
        hidden_dims=(64, 64),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_iterations=int(args.final_iterations),
        gradient_steps_per_iteration=int(args.gradient_steps_per_iteration),
        target_update_tau=float(args.target_update_tau),
        validation_fraction=0.20,
        patience=max(1, int(args.patience)),
        min_improvement=1e-5,
        device="cpu",
        seed=int(seed),
        show_progress=False,
    )
    return FQESearchSpace(neural=base, neural_candidates=tuple(dict(row) for row in candidates))


def _fit_candidate_truth_rows(
    *,
    dataset: BenchmarkDataset,
    cell: DatasetCell,
    candidates: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, overrides in enumerate(candidates):
        candidate_id = f"neural_{idx:03d}"
        start = time.perf_counter()
        fit_error = ""
        try:
            cfg = _candidate_config(args, overrides=overrides, seed=int(cell.seed) + 50_000 + 1_003 * idx)
            model = fit_fqe_neural(
                dataset.states,
                dataset.actions,
                dataset.next_states,
                dataset.next_actions,
                dataset.rewards,
                float(dataset.gamma),
                terminals=dataset.terminals,
                sample_weight=dataset.sample_weight,
                config=cfg,
            )
            policy_value = _policy_value(model, dataset)
            q_mse = _q_mse(model, dataset)
        except Exception as exc:
            policy_value = float("nan")
            q_mse = float("nan")
            fit_error = f"{type(exc).__name__}: {exc}"
        true_value = float(dataset.true_policy_value) if dataset.true_policy_value is not None else float("nan")
        abs_error = abs(policy_value - true_value) if np.isfinite(policy_value) and np.isfinite(true_value) else float("inf")
        rows.append(
            {
                **_cell_fields(cell),
                **_size_fields(candidate_id),
                "candidate_id": candidate_id,
                "policy_value_estimate": policy_value,
                "policy_value_target": true_value,
                "policy_value_abs_error": abs_error,
                "q_mse": q_mse,
                "gym_normalized_error": _gym_normalized_error(abs_error, dataset),
                "runtime_sec": float(time.perf_counter() - start),
                "fit_error": fit_error,
            }
        )
    return rows


def _candidate_config(args: argparse.Namespace, *, overrides: dict[str, Any], seed: int) -> NeuralFQEConfig:
    clean = {key: value for key, value in dict(overrides).items() if str(key) != "_meta"}
    return NeuralFQEConfig.stable_defaults(
        hidden_dims=tuple(int(width) for width in clean.get("hidden_dims", (64, 64))),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        num_iterations=int(args.final_iterations),
        gradient_steps_per_iteration=int(args.gradient_steps_per_iteration),
        target_update_tau=float(args.target_update_tau),
        validation_fraction=0.20,
        patience=max(1, int(args.patience)),
        min_improvement=1e-5,
        device="cpu",
        seed=int(seed),
        show_progress=False,
    )


def _select_candidate(
    *,
    selector: str,
    dataset: BenchmarkDataset,
    space: FQESearchSpace,
    args: argparse.Namespace,
    seed: int,
) -> tuple[str, Any]:
    staged = selector.startswith("staged_k")
    cfg = FQETuningConfig(
        families=("neural",),
        cv_folds=int(args.cv_folds),
        seed=int(seed),
        budget="balanced",
        max_candidates=int(args.max_candidates),
        promotion_candidates=int(args.max_candidates),
        refit=False,
        screen_fraction=1.0,
        stable_fallback=False,
        staged_bootstrap_cv=staged,
        staged_cv_iterations=_selector_k(selector) if staged else None,
        staged_cv_n_bootstrap=int(args.bootstrap),
        staged_cv_min_survivors=1,
        score_bellman_weight=1.0 if selector == "naive_final_bellman_cv" else FQETuningConfig.score_bellman_weight,
        score_value_stability_weight=0.0 if selector == "naive_final_bellman_cv" else FQETuningConfig.score_value_stability_weight,
        score_calibration_weight=0.0 if selector == "naive_final_bellman_cv" else FQETuningConfig.score_calibration_weight,
        score_runtime_weight=0.0 if selector == "naive_final_bellman_cv" else FQETuningConfig.score_runtime_weight,
    )
    result = tune_fqe(
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        next_actions=dataset.next_actions,
        rewards=dataset.rewards,
        gamma=float(dataset.gamma),
        terminals=dataset.terminals,
        sample_weight=dataset.sample_weight,
        initial_states=dataset.initial_states,
        initial_actions=dataset.initial_actions,
        search_space=space,
        config=cfg,
    )
    return str(result.selected_candidate_id), result


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
    selected_error = float(selected_truth.get("policy_value_abs_error", float("inf")))
    return {
        **_cell_fields(cell),
        **_size_fields(str(selected_id)),
        "selector": selector,
        "selected_candidate_id": str(selected_id),
        "selection_source": selection_source,
        "policy_value_estimate": selected_truth.get("policy_value_estimate", float("nan")),
        "policy_value_target": selected_truth.get("policy_value_target", float("nan")),
        "policy_value_abs_error": selected_error,
        "q_mse": selected_truth.get("q_mse", float("nan")),
        "oracle_regret": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else float("inf"),
        "gym_normalized_error": selected_truth.get("gym_normalized_error", float("nan")),
        "runtime_sec": float(runtime_sec),
        "selection_error": error,
    }


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
    selected_id = min(truth_by_id, key=lambda cid: float(truth_by_id[cid].get("policy_value_abs_error", float("inf"))))
    return _selector_row(
        cell=cell,
        selector="oracle_best",
        selected_id=selected_id,
        truth_by_id=truth_by_id,
        runtime_sec=0.0,
        error="",
        selection_source="reporting_only",
    )


def _aggregate_by_strategy(selector_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selector_rows if str(row.get("selector", "")) and str(row.get("selector")) != "oracle_best"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset_family"]), str(row["selector"])), []).append(row)
        groups.setdefault(("all", str(row["selector"])), []).append(row)
    win_keys = _winning_keys(rows)
    out: list[dict[str, Any]] = []
    for (family, selector), group in sorted(groups.items()):
        errors = _finite_values(row.get("policy_value_abs_error") for row in group)
        regrets = _finite_values(row.get("oracle_regret") for row in group)
        q_mses = _finite_values(row.get("q_mse") for row in group)
        gym_norm = _finite_values(row.get("gym_normalized_error") for row in group)
        runtimes = _finite_values(row.get("runtime_sec") for row in group)
        out.append(
            {
                "dataset_family": family,
                "selector": selector,
                "n_cells": len(group),
                "mean_abs_error": _mean(errors),
                "median_abs_error": _median(errors),
                "mean_oracle_regret": _mean(regrets),
                "median_oracle_regret": _median(regrets),
                "mean_q_mse": _mean(q_mses),
                "median_q_mse": _median(q_mses),
                "mean_gym_normalized_error": _mean(gym_norm),
                "median_gym_normalized_error": _median(gym_norm),
                "mean_runtime_sec": _mean(runtimes),
                "win_count": sum((str(row["cell_id"]), str(selector)) in win_keys for row in group),
                "oracle_match_count": sum(float(row.get("oracle_regret", float("inf"))) <= 1e-12 for row in group),
                "selected_complexities": ",".join(str(row.get("selected_hidden_dims", "")) for row in group),
            }
        )
    return out


def _recommend_default(selector_rows: Sequence[dict[str, Any]], aggregate_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall = [row for row in aggregate_rows if str(row["dataset_family"]) == "all"]
    family_rows = [row for row in aggregate_rows if str(row["dataset_family"]) != "all"]
    selectors = [str(row["selector"]) for row in overall]
    family_best: dict[str, float] = {}
    for row in family_rows:
        family = str(row["dataset_family"])
        family_best[family] = min(family_best.get(family, float("inf")), float(row["mean_oracle_regret"]))
    eligible: set[str] = set(selectors)
    for selector in selectors:
        for family, best in family_best.items():
            row = next((item for item in family_rows if str(item["selector"]) == selector and str(item["dataset_family"]) == family), None)
            if row is None:
                eligible.discard(selector)
                break
            threshold = best + max(1e-12, 0.25 * max(abs(best), 1e-12))
            if float(row["mean_oracle_regret"]) > threshold:
                eligible.discard(selector)
                break
    exact_by_selector = _exact_truth_median(selector_rows)
    gym_by_selector = _gym_median(selector_rows)
    sortable = []
    for row in overall:
        selector = str(row["selector"])
        sortable.append(
            (
                selector not in eligible,
                float(row["median_oracle_regret"]),
                float(row["median_abs_error"]),
                exact_by_selector.get(selector, float("inf")),
                gym_by_selector.get(selector, float("inf")),
                float(row["mean_runtime_sec"]),
                selector,
                row,
            )
        )
    if not sortable:
        return {"recommended_selector": "", "reason": "no completed non-oracle selectors", "eligible_selectors": []}
    chosen = min(sortable)
    selector = str(chosen[6])
    return {
        "recommended_selector": selector,
        "eligible_selectors": sorted(eligible),
        "median_oracle_regret": float(chosen[1]),
        "median_abs_error": float(chosen[2]),
        "exact_truth_median_abs_error": float(chosen[3]),
        "gym_median_normalized_error": float(chosen[4]),
        "mean_runtime_sec": float(chosen[5]),
        "decision_rule": "lowest median oracle regret, with family-level 25% mean-regret guardrail",
    }


def _write_recommendation(out_dir: Path, recommendation: dict[str, Any]) -> None:
    (out_dir / "recommendation.json").write_text(json.dumps(recommendation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# FQE CV Strategy Recommendation",
        "",
        f"Recommended selector: `{recommendation.get('recommended_selector', '')}`",
        "",
        f"Median oracle regret: {recommendation.get('median_oracle_regret', float('nan'))}",
        f"Median absolute value error: {recommendation.get('median_abs_error', float('nan'))}",
        f"Exact-truth median absolute error: {recommendation.get('exact_truth_median_abs_error', float('nan'))}",
        f"Gym median normalized error: {recommendation.get('gym_median_normalized_error', float('nan'))}",
        f"Mean selector runtime seconds: {recommendation.get('mean_runtime_sec', float('nan'))}",
        "",
        f"Decision rule: {recommendation.get('decision_rule', '')}",
    ]
    (out_dir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                    "dataset": row.get("dataset", ""),
                    "dataset_family": row.get("dataset_family", ""),
                    "selector": selector,
                    "baseline_selector": baseline_selector,
                    "oracle_regret_delta_vs_baseline": _to_float(row.get("oracle_regret")) - _to_float(baseline.get("oracle_regret")),
                    "policy_value_abs_error_delta_vs_baseline": _to_float(row.get("policy_value_abs_error"))
                    - _to_float(baseline.get("policy_value_abs_error")),
                    "runtime_delta_vs_baseline": _to_float(row.get("runtime_sec")) - _to_float(baseline.get("runtime_sec")),
                    "selector_worse_on_regret": float(
                        _to_float(row.get("oracle_regret")) > _to_float(baseline.get("oracle_regret")) + 1e-12
                    ),
                }
            )
    return out


def _bootstrap_ci(
    selector_rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [row for row in selector_rows if str(row.get("selector", "")) not in {"", "oracle_best"}]
    rng = np.random.default_rng(int(seed))
    out: list[dict[str, Any]] = []
    for family in sorted({str(row.get("dataset_family", "")) for row in rows} | {"all"}):
        family_rows = rows if family == "all" else [row for row in rows if str(row.get("dataset_family", "")) == family]
        for selector in sorted({str(row.get("selector", "")) for row in family_rows}):
            group = [row for row in family_rows if str(row.get("selector", "")) == selector]
            for metric in ("oracle_regret", "policy_value_abs_error", "gym_normalized_error"):
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
    by_family = {
        str(row["dataset_family"]): row
        for row in aggregate_rows
        if str(row.get("selector", "")) == target_selector
    }
    for family in sorted({str(row["dataset_family"]) for row in aggregate_rows if str(row["dataset_family"]) != "all"}):
        family_aggs = [
            row
            for row in aggregate_rows
            if str(row.get("dataset_family", "")) == family and str(row.get("selector", "")) != "oracle_best"
        ]
        target = by_family.get(family)
        if target is None or not family_aggs:
            continue
        if family == "gym":
            metric = "median_gym_normalized_error"
            limit_multiplier = 1.25
        else:
            metric = "mean_oracle_regret"
            limit_multiplier = 1.10 if family in {"discrete", "linear_gaussian"} else 1.25
        values = [_to_float(row.get(metric)) for row in family_aggs]
        values = [value for value in values if np.isfinite(value)]
        if not values:
            continue
        best = min(values)
        observed = _to_float(target.get(metric))
        threshold = best * limit_multiplier if best > 0 else best + 1e-12
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
    failure_rates = _failure_rate_by_selector(selector_rows)
    largest_rates = _largest_model_rate_by_selector(selector_rows)
    non_stage = [name for name in ("naive_final_bellman_cv", "product_composite_cv") if name in failure_rates]
    if non_stage and target_selector in failure_rates:
        best_failure = min(failure_rates[name] for name in non_stage)
        rows.append(
            {
                "guardrail": "failure_rate",
                "dataset_family": "all",
                "metric": "failure_rate",
                "target_selector": target_selector,
                "target_value": failure_rates[target_selector],
                "best_value": best_failure,
                "threshold": best_failure + 0.02,
                "passed": float(failure_rates[target_selector] <= best_failure + 0.02 + 1e-12),
            }
        )
    if non_stage and target_selector in largest_rates:
        best_largest = min(largest_rates[name] for name in non_stage)
        rows.append(
            {
                "guardrail": "largest_model_rate",
                "dataset_family": "all",
                "metric": "largest_model_rate",
                "target_selector": target_selector,
                "target_value": largest_rates[target_selector],
                "best_value": best_largest,
                "threshold": best_largest + 0.25,
                "passed": float(largest_rates[target_selector] <= best_largest + 0.25 + 1e-12),
            }
        )
    return rows


def _promotion_decision(
    selector_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    guardrail_rows: Sequence[dict[str, Any]],
    *,
    target_selector: str,
    non_stage_selectors: Sequence[str],
) -> dict[str, Any]:
    overall = {str(row["selector"]): row for row in aggregate_rows if str(row.get("dataset_family")) == "all"}
    target = overall.get(target_selector)
    non_stage = [overall[name] for name in non_stage_selectors if name in overall]
    if target is None or not non_stage:
        return {
            "promote": False,
            "target_selector": target_selector,
            "reason": "missing target or non-stage selector rows",
        }
    target_median = _to_float(target.get("median_oracle_regret"))
    all_medians = [_to_float(row.get("median_oracle_regret")) for row in overall.values()]
    best_overall = min(value for value in all_medians if np.isfinite(value))
    best_non_stage = min(_to_float(row.get("median_oracle_regret")) for row in non_stage)
    if best_non_stage > 0:
        improvement = (best_non_stage - target_median) / best_non_stage
    else:
        improvement = 1.0 if target_median <= best_non_stage + 1e-12 else float("-inf")
    conditions = {
        "best_or_tied_median_regret": bool(target_median <= best_overall + max(1e-12, 0.01 * abs(best_overall))),
        "non_stage_improvement_at_least_15pct": bool(improvement >= 0.15),
        "family_guardrails_pass": bool(guardrail_rows and all(float(row.get("passed", 0.0)) > 0.5 for row in guardrail_rows)),
    }
    promote = bool(all(conditions.values()))
    return {
        "promote": promote,
        "target_selector": target_selector,
        "conditions": conditions,
        "median_oracle_regret": target_median,
        "best_overall_median_oracle_regret": best_overall,
        "best_non_stage_median_oracle_regret": best_non_stage,
        "relative_improvement_vs_best_non_stage": improvement,
        "decision_rule": (
            "best/tied median regret, >=15% median-regret improvement over best non-stage, "
            "and all family/failure/largest-model guardrails pass"
        ),
    }


def _write_promotion_decision(out_dir: Path, promotion: dict[str, Any]) -> None:
    (out_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    conditions = promotion.get("conditions", {})
    lines = [
        "# FQE Staged-CV Promotion Decision",
        "",
        f"Promote default: `{bool(promotion.get('promote', False))}`",
        f"Target selector: `{promotion.get('target_selector', '')}`",
        "",
        f"Median oracle regret: {promotion.get('median_oracle_regret', float('nan'))}",
        f"Best non-stage median oracle regret: {promotion.get('best_non_stage_median_oracle_regret', float('nan'))}",
        f"Relative improvement vs best non-stage: {promotion.get('relative_improvement_vs_best_non_stage', float('nan'))}",
        "",
        "Conditions:",
    ]
    for key, value in sorted(dict(conditions).items()):
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


def _policy_value(model: Any, dataset: BenchmarkDataset) -> float:
    return float(model.estimate_policy_value(dataset.initial_states, dataset.initial_actions))


def _q_mse(model: Any, dataset: BenchmarkDataset) -> float:
    if dataset.true_q_fn is None:
        return float("nan")
    pred = np.asarray(model.predict_q(dataset.target_eval_states, dataset.target_eval_actions), dtype=np.float64).reshape(-1)
    truth = np.asarray(dataset.true_q_fn(dataset.target_eval_states, dataset.target_eval_actions), dtype=np.float64).reshape(-1)
    return float(np.mean((pred - truth) ** 2))


def _gym_normalized_error(abs_error: float, dataset: BenchmarkDataset) -> float:
    se = float(dataset.metadata.get("target_policy_value_se", float("nan")))
    if dataset.domain != "gym_control" or not np.isfinite(se) or se <= 0.0:
        return float("nan")
    return float(abs_error) / max(se, 1e-12)


def _cell_fields(cell: DatasetCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "dataset": cell.dataset,
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


def _selectors(args: argparse.Namespace) -> tuple[str, ...]:
    if args.selectors:
        return tuple(str(selector) for selector in args.selectors)
    staged = tuple(f"staged_k{int(k)}" for k in args.stage_counts)
    return (*staged, "naive_final_bellman_cv", "product_composite_cv", "oracle_best")


def _selector_k(selector: str) -> int:
    return int(str(selector).replace("staged_k", ""))


def _selector_seed_offset(selector: str) -> int:
    return 1_000 + sum((idx + 1) * ord(char) for idx, char in enumerate(str(selector)))


def _oracle_error(truth_by_id: dict[str, dict[str, Any]]) -> float:
    if not truth_by_id:
        return float("inf")
    return min(float(row.get("policy_value_abs_error", float("inf"))) for row in truth_by_id.values())


def _winning_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(row)
    keys: set[tuple[str, str]] = set()
    for cell_id, group in by_cell.items():
        finite = [row for row in group if np.isfinite(float(row.get("policy_value_abs_error", float("inf"))))]
        if not finite:
            continue
        best = min(float(row["policy_value_abs_error"]) for row in finite)
        for row in finite:
            if float(row["policy_value_abs_error"]) <= best + 1e-12:
                keys.add((cell_id, str(row["selector"])))
    return keys


def _exact_truth_median(selector_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for selector in {str(row.get("selector", "")) for row in selector_rows}:
        values = _finite_values(
            row.get("policy_value_abs_error")
            for row in selector_rows
            if str(row.get("selector", "")) == selector and str(row.get("dataset_family", "")) in {"discrete", "linear_gaussian"}
        )
        out[selector] = _median(values)
    return out


def _gym_median(selector_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for selector in {str(row.get("selector", "")) for row in selector_rows}:
        values = _finite_values(
            row.get("gym_normalized_error")
            for row in selector_rows
            if str(row.get("selector", "")) == selector and str(row.get("dataset_family", "")) == "gym"
        )
        out[selector] = _median(values)
    return out


def _finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out.append(numeric)
    return out


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out


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
        if not rows:
            continue
        out[selector] = float(
            np.mean(
                [
                    bool(str(row.get("selection_error", "")))
                    or not np.isfinite(_to_float(row.get("oracle_regret"), float("inf")))
                    for row in rows
                ]
            )
        )
    return out


def _largest_model_rate_by_selector(selector_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    selectors = sorted({str(row.get("selector", "")) for row in selector_rows if str(row.get("selector", "")) != "oracle_best"})
    for selector in selectors:
        rows = [row for row in selector_rows if str(row.get("selector", "")) == selector]
        if not rows:
            continue
        out[selector] = float(
            np.mean([_candidate_index(str(row.get("selected_candidate_id", ""))) == len(SIZE_GRID) - 1 for row in rows])
        )
    return out


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _mlp_parameter_count(input_dim: int, hidden_dims: Sequence[int], *, output_dim: int) -> float:
    dims = [int(input_dim), *(int(width) for width in hidden_dims), int(output_dim)]
    total = 0
    for left, right in zip(dims[:-1], dims[1:]):
        total += left * right + right
    return float(total)


def _dims_str(dims: Iterable[int]) -> str:
    return "x".join(str(int(dim)) for dim in dims)


def _is_runtime_drop_cell(cell: DatasetCell) -> bool:
    return bool(cell.family in {"discrete", "linear_gaussian"} and int(cell.sample_size) == 2048 and abs(float(cell.gamma) - 0.7) < 1e-12)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(aggregate_rows: Sequence[dict[str, Any]], recommendation: dict[str, Any]) -> None:
    print("dataset_family\tselector\tn\tmedian_regret\tmedian_abs_error\tmean_runtime")
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
                    f"{float(row['median_abs_error']):.6g}",
                    f"{float(row['mean_runtime_sec']):.2f}",
                ]
            )
        )
    print(f"recommended\t{recommendation.get('recommended_selector', '')}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FQE neural CV selection strategies.")
    parser.add_argument("--profile", choices=("smoke", "core_realistic", "overnight"), default=None)
    parser.add_argument("--output-dir", default="outputs/fqe_cv_strategy_benchmark")
    parser.add_argument("--discrete-datasets", nargs="+", default=("tabular_chain", "tabular_grid"))
    parser.add_argument("--linear-policy-shifts", nargs="+", type=float, default=(0.7, 1.2))
    parser.add_argument("--gym-datasets", nargs="+", default=("gym_pendulum", "gym_mountain_car_continuous"))
    parser.add_argument("--skip-discrete", action="store_true")
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--skip-gym", action="store_true")
    parser.add_argument("--synthetic-sample-sizes", nargs="+", type=int, default=(512, 2048))
    parser.add_argument("--synthetic-gammas", nargs="+", type=float, default=(0.7, 0.9))
    parser.add_argument("--synthetic-seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--gym-sample-size", type=int, default=1024)
    parser.add_argument("--gym-sample-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--gym-gamma", type=float, default=0.9)
    parser.add_argument("--gym-gammas", nargs="+", type=float, default=None)
    parser.add_argument("--gym-seeds", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--gym-target-value-rollouts", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--n-initial-eval", type=int, default=256)
    parser.add_argument("--stage-counts", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--selectors", nargs="+", default=None)
    parser.add_argument("--max-candidates", type=int, default=len(SIZE_GRID))
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--final-iterations", type=int, default=10)
    parser.add_argument("--gradient-steps-per-iteration", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--target-update-tau", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=3)
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
            "output_dir": "outputs/fqe_cv_strategy_smoke",
            "discrete_datasets": ("tabular_chain",),
            "skip_linear": True,
            "skip_gym": False,
            "synthetic_sample_sizes": (48,),
            "synthetic_gammas": (0.0,),
            "synthetic_seeds": (0,),
            "gym_datasets": ("gym_mountain_car_continuous",),
            "gym_sample_sizes": (64,),
            "gym_gammas": (0.9,),
            "gym_seeds": (0,),
            "gym_target_value_rollouts": 2,
            "n_eval": 32,
            "n_initial_eval": 16,
            "selectors": ("staged_k1", "staged_k3", "naive_final_bellman_cv", "oracle_best"),
            "max_candidates": 2,
            "cv_folds": 2,
            "bootstrap": 0,
            "final_iterations": 1,
            "gradient_steps_per_iteration": 1,
            "batch_size": 32,
            "analysis_bootstrap": 20,
            "time_budget_minutes": 20.0,
        },
        "core_realistic": {
            "output_dir": "outputs/fqe_cv_strategy_core_realistic",
            "discrete_datasets": ("tabular_chain", "tabular_grid"),
            "linear_policy_shifts": (0.25, 0.7, 1.2, 2.0),
            "synthetic_sample_sizes": (512, 2048, 8192),
            "synthetic_gammas": (0.7, 0.9, 0.95),
            "synthetic_seeds": tuple(range(8)),
            "gym_datasets": ("gym_pendulum", "gym_mountain_car_continuous"),
            "gym_sample_sizes": (1024, 4096),
            "gym_gammas": (0.9, 0.95),
            "gym_seeds": tuple(range(5)),
            "gym_target_value_rollouts": 64,
            "time_budget_minutes": 8.0 * 60.0,
        },
        "overnight": {
            "output_dir": "outputs/fqe_cv_strategy_overnight",
            "discrete_datasets": ("tabular_chain", "tabular_grid"),
            "linear_policy_shifts": (0.25, 0.7, 1.2, 2.0),
            "synthetic_sample_sizes": (512, 2048, 8192),
            "synthetic_gammas": (0.7, 0.9, 0.95),
            "synthetic_seeds": tuple(range(12)),
            "gym_datasets": ("gym_pendulum", "gym_mountain_car_continuous"),
            "gym_sample_sizes": (1024, 4096),
            "gym_gammas": (0.9, 0.95),
            "gym_seeds": tuple(range(8)),
            "gym_target_value_rollouts": 96,
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
