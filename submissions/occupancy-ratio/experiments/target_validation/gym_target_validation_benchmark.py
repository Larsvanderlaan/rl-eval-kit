from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fqe.fit_fqe import BoostedFQEConfig
import fqe.tuning as fqe_tuning
from fqe.tuning import FQESearchSpace, FQETuningConfig, tune_fqe
import occupancy_ratio.tuning as occ_tuning
from occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
)
from occupancy_ratio.tuning import OccupancySearchSpace, OccupancyTuningConfig, tune_occupancy_ratio
from occupancy_ratio_benchmark.gym_control import GYM_CONTROL_SETTINGS, _make_policies, make_gym_control_dataset


Array = np.ndarray


@dataclass(frozen=True)
class PolicyBundle:
    setting: str
    env_id: str
    target_policy: Any
    max_steps: int


@dataclass(frozen=True)
class TargetRollouts:
    states: Array
    actions: Array
    rewards: Array
    next_states: Array
    episode_ids: Array
    timesteps: Array
    continuation: Array
    tail_actions: Array
    returns: Array


@dataclass
class FQECandidateFit:
    candidate_id: str
    family: str
    overrides: dict[str, Any]
    model: Any | None
    policy_value: float
    true_error: float
    runtime_sec: float
    error: str = ""


@dataclass
class OccupancyCandidateFit:
    candidate_id: str
    candidate_label: str
    family: str
    overrides: dict[str, dict[str, Any]]
    model: Any | None
    ope_moment: float
    true_error: float
    guardrail_passed: bool
    weight_metrics: dict[str, float]
    runtime_sec: float
    error: str = ""


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Gym benchmark for target-validation-assisted FQE and occupancy-ratio tuning.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/target_validation_gym"))
    parser.add_argument("--settings", nargs="+", default=["gym_pendulum", "gym_mountain_car_continuous"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--validation-rollouts", nargs="+", type=int, default=[4, 16, 64])
    parser.add_argument("--horizons", nargs="+", type=int, default=[25, 100, 200])
    parser.add_argument("--truth-rollouts", type=int, default=96)
    parser.add_argument("--fqe-candidates", type=int, default=4)
    parser.add_argument("--occupancy-candidates", type=int, default=4)
    parser.add_argument("--skip-fqe", action="store_true")
    parser.add_argument("--skip-occupancy", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--enforce-compact-gate", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_rows.csv"
    candidate_path = args.output_dir / "candidate_rows.csv"
    completed = set() if args.rerun else _completed_cells(selected_path)
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for setting in args.settings:
        if setting not in GYM_CONTROL_SETTINGS:
            raise ValueError(f"Unknown Gym control setting '{setting}'.")
        for seed in args.seeds:
            base_key = (str(setting), int(seed))
            if base_key in completed:
                continue
            print(f"cell setting={setting} seed={seed}", flush=True)
            bundle = _policy_bundle(str(setting))
            dataset = make_gym_control_dataset(
                setting=str(setting),
                gamma=float(args.gamma),
                sample_size=int(args.sample_size),
                seed=int(seed),
                target_value_rollouts=max(4, min(int(args.truth_rollouts), 32)),
            )
            truth_mean, truth_se = _estimate_policy_value_raw(
                bundle,
                gamma=float(args.gamma),
                rollouts=int(args.truth_rollouts),
                seed=int(seed) + 901_001,
            )
            max_horizon = min(max(int(h) for h in args.horizons), int(bundle.max_steps))
            max_rollouts = max(int(n) for n in args.validation_rollouts)
            full_validation = _collect_target_rollouts(
                bundle,
                gamma=float(args.gamma),
                n_rollouts=max_rollouts,
                horizon=max_horizon,
                seed=int(seed) + 707_001,
            )
            fqe_fits: list[FQECandidateFit] = []
            fqe_proxy_id = ""
            if not args.skip_fqe:
                fqe_fits = _fit_fqe_candidates(
                    dataset,
                    gamma=float(args.gamma),
                    seed=int(seed),
                    truth_value=float(truth_mean),
                    max_candidates=int(args.fqe_candidates),
                )
                if not args.skip_proxy:
                    fqe_proxy_id = _select_fqe_proxy(dataset, gamma=float(args.gamma), seed=int(seed), max_candidates=int(args.fqe_candidates))
            occ_fits: list[OccupancyCandidateFit] = []
            occ_proxy_id = ""
            target_norm = (1.0 - float(args.gamma)) * float(truth_mean)
            if not args.skip_occupancy:
                occ_fits = _fit_occupancy_candidates(
                    dataset,
                    gamma=float(args.gamma),
                    seed=int(seed),
                    target_norm=float(target_norm),
                    max_candidates=int(args.occupancy_candidates),
                )
                if not args.skip_proxy:
                    occ_proxy_id = _select_occupancy_proxy(
                        dataset,
                        gamma=float(args.gamma),
                        seed=int(seed),
                        max_candidates=int(args.occupancy_candidates),
                    )

            for horizon in args.horizons:
                h = min(int(horizon), int(bundle.max_steps), max_horizon)
                for n_rollouts in args.validation_rollouts:
                    validation = _prefix_rollouts(full_validation, n_rollouts=int(n_rollouts), horizon=h, gamma=float(args.gamma))
                    cell_meta = {
                        "setting": str(setting),
                        "seed": int(seed),
                        "sample_size": int(args.sample_size),
                        "gamma": float(args.gamma),
                        "validation_rollouts": int(n_rollouts),
                        "validation_horizon": int(h),
                        "truth_raw_value": float(truth_mean),
                        "truth_raw_value_se": float(truth_se),
                        "truth_discounted_reward_moment": float(target_norm),
                        "finite_prefix_raw_value": float(np.mean(validation.returns)),
                        "finite_prefix_raw_value_se": _se(validation.returns),
                    }
                    if fqe_fits:
                        selected, candidates = _score_fqe_cell(
                            fqe_fits,
                            validation,
                            gamma=float(args.gamma),
                            seed=int(seed),
                            truth_value=float(truth_mean),
                            truth_value_se=float(truth_se),
                            proxy_candidate_id=fqe_proxy_id,
                        )
                        selected_rows.extend(_with_common(row, cell_meta) for row in selected)
                        candidate_rows.extend(_with_common(row, cell_meta) for row in candidates)
                    if occ_fits:
                        selected, candidates = _score_occupancy_cell(
                            occ_fits,
                            dataset,
                            validation,
                            gamma=float(args.gamma),
                            seed=int(seed),
                            target_norm=float(target_norm),
                            truth_value=float(truth_mean),
                            truth_value_se=float(truth_se),
                            proxy_candidate_id=occ_proxy_id,
                        )
                        selected_rows.extend(_with_common(row, cell_meta) for row in selected)
                        candidate_rows.extend(_with_common(row, cell_meta) for row in candidates)
            _append_rows(selected_path, selected_rows)
            _append_rows(candidate_path, candidate_rows)
            selected_rows.clear()
            candidate_rows.clear()

    selected_all = _read_csv(selected_path)
    summary_rows = _summarize_selected(selected_all)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_report(args.output_dir / "report.md", summary_rows, selected_all)
    gate_report = _compact_gate_report(summary_rows, selected_all)
    with (args.output_dir / "gate_report.json").open("w") as fh:
        json.dump(_jsonable(gate_report), fh, indent=2)
    with (args.output_dir / "run_config.json").open("w") as fh:
        json.dump(_jsonable(vars(args)), fh, indent=2)
    if bool(args.enforce_compact_gate) and not bool(gate_report["passed"]):
        failures = ", ".join(check["name"] for check in gate_report["checks"] if not check["passed"])
        raise SystemExit(f"compact target-validation gate failed: {failures}")
    print(f"wrote {args.output_dir}", flush=True)


def _policy_bundle(setting: str) -> PolicyBundle:
    import gymnasium as gym

    env_id = GYM_CONTROL_SETTINGS[setting]
    env = gym.make(env_id)
    try:
        state_dim = int(np.asarray(env.observation_space.shape).prod())
        action_dim = int(np.asarray(env.action_space.shape).prod())
        action_low = np.asarray(env.action_space.low, dtype=np.float64).reshape(action_dim)
        action_high = np.asarray(env.action_space.high, dtype=np.float64).reshape(action_dim)
        action_low, action_high = _finite_action_bounds(action_low, action_high)
        _, target_policy = _make_policies(
            setting=setting,
            state_dim=state_dim,
            action_low=action_low,
            action_high=action_high,
        )
        max_steps = int(getattr(env.spec, "max_episode_steps", None) or 1_000)
    finally:
        env.close()
    return PolicyBundle(setting=setting, env_id=env_id, target_policy=target_policy, max_steps=max_steps)


def _fit_fqe_candidates(
    dataset: Any,
    *,
    gamma: float,
    seed: int,
    truth_value: float,
    max_candidates: int,
) -> list[FQECandidateFit]:
    space = _fqe_search_space(max_candidates)
    cfg = FQETuningConfig(
        families=("boosted",),
        budget="balanced",
        max_candidates=int(max_candidates),
        promotion_candidates=min(2, int(max_candidates)),
        cv_folds=2,
        seed=int(seed) + 11_003,
    )
    candidates = fqe_tuning._make_candidates(space, cfg)
    fits: list[FQECandidateFit] = []
    terminals = 1.0 - np.asarray(dataset.masks, dtype=np.float64).reshape(-1)
    for rank, candidate in enumerate(candidates):
        start = time.perf_counter()
        model = None
        value = float("nan")
        error = ""
        try:
            candidate_cfg = fqe_tuning._build_config(
                family=str(candidate["family"]),
                overrides=dict(candidate["overrides"]),
                space=space,
                screen_fraction=1.0,
                seed=int(seed) + 31_001 + 10_001 * rank,
                force_final=True,
            )
            model = fqe_tuning._fit_family(
                family=str(candidate["family"]),
                mode="q",
                config=candidate_cfg,
                S=np.asarray(dataset.states, dtype=np.float64),
                A=np.asarray(dataset.actions, dtype=np.float64),
                S_next=np.asarray(dataset.next_states, dtype=np.float64),
                A_next=np.asarray(dataset.next_target_actions, dtype=np.float64),
                rewards=np.asarray(dataset.rewards, dtype=np.float64),
                gamma=float(gamma),
                terminals=terminals,
                sample_weight=np.ones_like(terminals, dtype=np.float64),
                categorical_feature=None,
            )
            value = float(model.estimate_policy_value(dataset.initial_states, dataset.initial_actions))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        fits.append(
            FQECandidateFit(
                candidate_id=str(candidate["candidate_id"]),
                family=str(candidate["family"]),
                overrides=dict(candidate["overrides"]),
                model=model,
                policy_value=float(value),
                true_error=abs(float(value) - float(truth_value)) if np.isfinite(value) else float("inf"),
                runtime_sec=float(time.perf_counter() - start),
                error=error,
            )
        )
    return fits


def _select_fqe_proxy(dataset: Any, *, gamma: float, seed: int, max_candidates: int) -> str:
    try:
        result = tune_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_target_actions,
            rewards=dataset.rewards,
            gamma=float(gamma),
            terminals=1.0 - np.asarray(dataset.masks, dtype=np.float64).reshape(-1),
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            search_space=_fqe_search_space(max_candidates),
            config=FQETuningConfig(
                families=("boosted",),
                budget="balanced",
                max_candidates=int(max_candidates),
                promotion_candidates=min(2, int(max_candidates)),
                cv_folds=2,
                seed=int(seed) + 41_001,
            ),
        )
        return str(result.selected_candidate_id)
    except Exception as exc:
        print(f"FQE proxy selector failed: {type(exc).__name__}: {exc}", flush=True)
        return ""


def _fit_occupancy_candidates(
    dataset: Any,
    *,
    gamma: float,
    seed: int,
    target_norm: float,
    max_candidates: int,
) -> list[OccupancyCandidateFit]:
    space = _occupancy_search_space(max_candidates)
    cfg = OccupancyTuningConfig(
        families=("boosted",),
        budget="balanced",
        max_candidates=int(max_candidates),
        promotion_candidates=min(2, int(max_candidates)),
        cv_folds=2,
        seed=int(seed) + 51_003,
        stagewise=False,
    )
    candidates = occ_tuning._make_candidates(
        space,
        cfg,
        has_initial_states=dataset.initial_states is not None,
        has_initial_actions=dataset.initial_actions is not None,
    )
    fits: list[OccupancyCandidateFit] = []
    action_shift = occ_tuning._action_shift(dataset.actions, dataset.target_actions)
    for rank, candidate in enumerate(candidates):
        start = time.perf_counter()
        model = None
        ope = float("nan")
        weight_metrics: dict[str, float] = {}
        guardrail_passed = False
        error = ""
        try:
            configs = occ_tuning._build_configs(
                family=str(candidate["family"]),
                overrides=dict(candidate["overrides"]),
                space=space,
                screen_fraction=1.0,
                seed=int(seed) + 61_001 + 10_001 * rank,
            )
            model = occ_tuning._fit_family(
                family=str(candidate["family"]),
                configs=configs,
                states=dataset.states,
                actions=dataset.actions,
                next_states=dataset.next_states,
                target_actions=dataset.target_actions,
                gamma=float(gamma),
                initial_states=dataset.initial_states,
                initial_actions=dataset.initial_actions,
                initial_weights=dataset.initial_weights,
                target_next_actions=dataset.next_target_actions,
                initial_ratio_mode="auto",
                one_step_ratio_mode="auto",
            )
            weights = np.asarray(model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True), dtype=np.float64).reshape(-1)
            ope = float(np.mean(weights * np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)))
            weight_metrics = occ_tuning._final_weight_metrics(weights, occ_tuning._scoring_weight_config(configs), action_shift=action_shift)
            guardrail_passed = bool(occ_tuning._target_validation_guardrails_pass(weight_metrics))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        fits.append(
            OccupancyCandidateFit(
                candidate_id=str(candidate["candidate_id"]),
                candidate_label=str(candidate.get("candidate_label", candidate["candidate_id"])),
                family=str(candidate["family"]),
                overrides=dict(candidate["overrides"]),
                model=model,
                ope_moment=float(ope),
                true_error=abs(float(ope) - float(target_norm)) if np.isfinite(ope) else float("inf"),
                guardrail_passed=bool(guardrail_passed),
                weight_metrics=weight_metrics,
                runtime_sec=float(time.perf_counter() - start),
                error=error,
            )
        )
    return fits


def _select_occupancy_proxy(dataset: Any, *, gamma: float, seed: int, max_candidates: int) -> str:
    try:
        result = tune_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            target_next_actions=dataset.next_target_actions,
            gamma=float(gamma),
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            initial_weights=dataset.initial_weights,
            rewards=dataset.rewards,
            search_space=_occupancy_search_space(max_candidates),
            config=OccupancyTuningConfig(
                families=("boosted",),
                budget="balanced",
                max_candidates=int(max_candidates),
                promotion_candidates=min(2, int(max_candidates)),
                cv_folds=2,
                seed=int(seed) + 71_001,
                stagewise=False,
            ),
        )
        return str(result.selected_candidate_id)
    except Exception as exc:
        print(f"occupancy proxy selector failed: {type(exc).__name__}: {exc}", flush=True)
        return ""


def _score_fqe_cell(
    fits: list[FQECandidateFit],
    validation: TargetRollouts,
    *,
    gamma: float,
    seed: int,
    truth_value: float,
    truth_value_se: float,
    proxy_candidate_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory = fqe_tuning._validate_fqe_target_trajectory(
        mode="q",
        validation_states=validation.states,
        validation_actions=validation.actions,
        validation_rewards=validation.rewards,
        validation_next_states=validation.next_states,
        validation_episode_ids=validation.episode_ids,
        validation_timestep=validation.timesteps,
        validation_terminals=None,
        validation_continuation=validation.continuation,
        validation_tail_actions=validation.tail_actions,
    )
    diagnostics = fqe_tuning._fqe_target_trajectory_diagnostics(trajectory, gamma=float(gamma), seed=int(seed) + 81_001)
    finite_value = float(np.mean(validation.returns))
    finite_value_se = _se(validation.returns)
    rows: list[dict[str, Any]] = []
    for idx, fit in enumerate(fits):
        nstep_score = float("inf")
        nstep_se = 0.0
        if fit.model is not None and not fit.error:
            metrics = fqe_tuning._score_fqe_target_trajectory(
                model=fit.model,
                mode="q",
                trajectory=trajectory,
                gamma=float(gamma),
                seed=int(seed) + 83_001 + idx,
            )
            nstep_score = float(metrics["validation_score"])
            nstep_se = float(metrics["validation_score_se"])
        rows.append(
            {
                "package": "fqe",
                "candidate_id": fit.candidate_id,
                "candidate_label": _candidate_label(fit.overrides),
                "selector_n_step_td_score": nstep_score,
                "selector_n_step_td_se": nstep_se,
                "selector_scalar_finite_score": abs(float(fit.policy_value) - finite_value) if np.isfinite(fit.policy_value) else float("inf"),
                "selector_scalar_finite_se": finite_value_se,
                "selector_scalar_truth_score": abs(float(fit.policy_value) - float(truth_value)) if np.isfinite(fit.policy_value) else float("inf"),
                "selector_scalar_truth_se": float(truth_value_se),
                "policy_value": fit.policy_value,
                "true_error": fit.true_error,
                "guardrail_passed": 1.0,
                "fit_runtime_sec": fit.runtime_sec,
                "error": fit.error,
            }
        )
    oracle = _best(rows, score_key="true_error")
    selected_rows = [
        _selector_row("fqe", "oracle_value", oracle, oracle, diagnostics),
        _selector_row("fqe", "target_n_step_td_min", _best(rows, score_key="selector_n_step_td_score"), oracle, diagnostics),
        _selector_row("fqe", "target_n_step_td", _one_se(rows, "selector_n_step_td_score", "selector_n_step_td_se"), oracle, diagnostics),
        _selector_row("fqe", "scalar_finite_prefix_min", _best(rows, score_key="selector_scalar_finite_score"), oracle, diagnostics),
        _selector_row("fqe", "scalar_finite_prefix", _one_se(rows, "selector_scalar_finite_score", "selector_scalar_finite_se"), oracle, diagnostics),
        _selector_row("fqe", "scalar_truth_value_min", _best(rows, score_key="selector_scalar_truth_score"), oracle, diagnostics),
        _selector_row("fqe", "scalar_truth_value", _one_se(rows, "selector_scalar_truth_score", "selector_scalar_truth_se"), oracle, diagnostics),
    ]
    if proxy_candidate_id:
        selected_rows.append(_selector_row("fqe", "proxy_cv", _by_candidate(rows, proxy_candidate_id), oracle, diagnostics))
    return selected_rows, rows


def _score_occupancy_cell(
    fits: list[OccupancyCandidateFit],
    dataset: Any,
    validation: TargetRollouts,
    *,
    gamma: float,
    seed: int,
    target_norm: float,
    truth_value: float,
    truth_value_se: float,
    proxy_candidate_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = occ_tuning._validate_occupancy_target_trajectory(
        validation_states=validation.states,
        validation_actions=validation.actions,
        validation_rewards=validation.rewards,
        validation_episode_ids=validation.episode_ids,
        validation_timestep=validation.timesteps,
        validation_terminals=None,
        validation_continuation=validation.continuation,
    )
    diagnostics = occ_tuning._occupancy_target_trajectory_diagnostics(target, gamma=float(gamma), seed=int(seed) + 91_001)
    finite_norm_target = (1.0 - float(gamma)) * float(np.mean(validation.returns))
    finite_norm_se = (1.0 - float(gamma)) * _se(validation.returns)
    rows: list[dict[str, Any]] = []
    for idx, fit in enumerate(fits):
        moment_score = float("inf")
        moment_se = 0.0
        if fit.model is not None and not fit.error:
            weights = np.asarray(fit.model.predict_state_action_ratio(dataset.states, dataset.actions, clip=True), dtype=np.float64).reshape(-1)
            metrics = occ_tuning._score_occupancy_discounted_moments(
                weights=weights,
                reference_states=dataset.states,
                reference_actions=dataset.actions,
                reference_rewards=dataset.rewards,
                target=target,
                gamma=float(gamma),
                seed=int(seed) + 93_001 + idx,
            )
            moment_score = float(metrics["validation_score"])
            moment_se = float(metrics["validation_score_se"])
        rows.append(
            {
                "package": "occupancy_ratio",
                "candidate_id": fit.candidate_id,
                "candidate_label": fit.candidate_label,
                "selector_discounted_moments_score": moment_score if fit.guardrail_passed else float("inf"),
                "selector_discounted_moments_se": moment_se,
                "selector_scalar_finite_score": abs(float(fit.ope_moment) - finite_norm_target) if fit.guardrail_passed else float("inf"),
                "selector_scalar_finite_se": finite_norm_se,
                "selector_scalar_truth_score": abs(float(fit.ope_moment) - float(target_norm)) if fit.guardrail_passed else float("inf"),
                "selector_scalar_truth_se": (1.0 - float(gamma)) * float(truth_value_se),
                "ope_discounted_reward_moment": fit.ope_moment,
                "true_error": fit.true_error,
                "guardrail_passed": float(fit.guardrail_passed),
                "fit_runtime_sec": fit.runtime_sec,
                "error": fit.error,
                **{f"weight_{key}": value for key, value in fit.weight_metrics.items()},
            }
        )
    oracle_all = _best(rows, score_key="true_error")
    oracle_guarded = _best([row for row in rows if float(row.get("guardrail_passed", 0.0)) > 0.0], score_key="true_error") or oracle_all
    selected_rows = [
        _selector_row("occupancy_ratio", "oracle_ope_all", oracle_all, oracle_all, diagnostics),
        _selector_row("occupancy_ratio", "oracle_ope_guarded", oracle_guarded, oracle_guarded, diagnostics),
        _selector_row(
            "occupancy_ratio",
            "target_discounted_moments_min",
            _best(rows, score_key="selector_discounted_moments_score"),
            oracle_guarded,
            diagnostics,
        ),
        _selector_row(
            "occupancy_ratio",
            "target_discounted_moments",
            _one_se(rows, "selector_discounted_moments_score", "selector_discounted_moments_se"),
            oracle_guarded,
            diagnostics,
        ),
        _selector_row(
            "occupancy_ratio",
            "scalar_finite_prefix_min",
            _best(rows, score_key="selector_scalar_finite_score"),
            oracle_guarded,
            diagnostics,
        ),
        _selector_row(
            "occupancy_ratio",
            "scalar_finite_prefix",
            _one_se(rows, "selector_scalar_finite_score", "selector_scalar_finite_se"),
            oracle_guarded,
            diagnostics,
        ),
        _selector_row(
            "occupancy_ratio",
            "scalar_truth_value_min",
            _best(rows, score_key="selector_scalar_truth_score"),
            oracle_guarded,
            diagnostics,
        ),
        _selector_row(
            "occupancy_ratio",
            "scalar_truth_value",
            _one_se(rows, "selector_scalar_truth_score", "selector_scalar_truth_se"),
            oracle_guarded,
            diagnostics,
        ),
    ]
    if proxy_candidate_id:
        selected_rows.append(_selector_row("occupancy_ratio", "proxy_cv", _by_candidate(rows, proxy_candidate_id), oracle_guarded, diagnostics))
    return selected_rows, rows


def _selector_row(
    package: str,
    selector: str,
    selected: dict[str, Any] | None,
    oracle: dict[str, Any] | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    selected = selected or {}
    oracle = oracle or {}
    selected_error = _float(selected.get("true_error", float("inf")))
    oracle_error = _float(oracle.get("true_error", float("inf")))
    return {
        "package": package,
        "selector": selector,
        "selected_candidate_id": selected.get("candidate_id", ""),
        "selected_candidate_label": selected.get("candidate_label", ""),
        "oracle_candidate_id": oracle.get("candidate_id", ""),
        "oracle_candidate_label": oracle.get("candidate_label", ""),
        "selected_true_error": selected_error,
        "oracle_true_error": oracle_error,
        "regret": selected_error - oracle_error if np.isfinite(selected_error) and np.isfinite(oracle_error) else float("inf"),
        "selected_oracle": float(str(selected.get("candidate_id", "")) == str(oracle.get("candidate_id", ""))),
        "selection_failed": float(not bool(selected)),
        "truncation_tail_mass_mean": _float(diagnostics.get("truncation_tail_mass_mean", float("nan"))),
        "truncation_tail_mass_max": _float(diagnostics.get("truncation_tail_mass_max", float("nan"))),
        "direct_target_return_mean": _float(diagnostics.get("direct_target_return_mean", float("nan"))),
        "direct_target_return_se": _float(diagnostics.get("direct_target_return_se", float("nan"))),
    }


def _fqe_search_space(max_candidates: int) -> FQESearchSpace:
    base = BoostedFQEConfig.stable_defaults(
        num_iterations=60,
        patience=6,
        validation_fraction=0.2,
        lgb_params={
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "lambda_l2": 1.0,
            "verbosity": -1,
        },
    )
    candidates = [
        {},
        {"num_iterations": 25, "lgb_params": {"num_leaves": 15, "min_data_in_leaf": 40, "lambda_l2": 4.0}},
        {"num_iterations": 80, "lgb_params": {"num_leaves": 63, "min_data_in_leaf": 10, "lambda_l2": 0.2}},
        {"loss": "squared", "infer_value_bounds": False, "learning_rate_backoff": 0.8},
    ]
    return FQESearchSpace(boosted=base, boosted_candidates=candidates[: int(max_candidates)])


def _occupancy_search_space(max_candidates: int) -> OccupancySearchSpace:
    ratio_lgb = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    occ_lgb = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    space = OccupancySearchSpace(
        boosted_occupancy=OccupancyRegressionConfig.stable_defaults(
            num_iterations=24,
            mcmc_samples=8,
            batch_size=512,
            patience=4,
            min_outer_iterations=3,
            direct_adjoint_num_boost_round=16,
            lgb_params=occ_lgb,
        ),
        boosted_action_ratio=ActionRatioConfig.stable_defaults(
            num_boost_round=50,
            early_stopping_rounds=5,
            lgb_params=ratio_lgb,
        ),
        boosted_source_state_ratio=SourceStateRatioConfig.stable_defaults(
            num_boost_round=50,
            early_stopping_rounds=5,
            lgb_params=ratio_lgb,
        ),
        boosted_transition_ratio=TransitionRatioConfig.stable_defaults(
            num_boost_round=50,
            early_stopping_rounds=5,
            permutation_samples=4,
            lgb_params=ratio_lgb,
        ),
        boosted_candidates=[
            {},
            {"occupancy": {"fixed_point_damping": 0.35, "occupancy_ratio_max": 25.0}},
            {"occupancy": {"fixed_point_damping": 0.75, "occupancy_ratio_max": 100.0, "pseudo_outcome_upper_quantile": 0.999}},
            {"occupancy": {"loss": "squared", "clip_pseudo_outcomes": False, "occupancy_sample_weight_mode": "sqrt_target"}},
        ][: int(max_candidates)],
    )
    return space


def _collect_target_rollouts(
    bundle: PolicyBundle,
    *,
    gamma: float,
    n_rollouts: int,
    horizon: int,
    seed: int,
) -> TargetRollouts:
    import gymnasium as gym

    rng = np.random.default_rng(int(seed))
    env = gym.make(bundle.env_id)
    states: list[Array] = []
    actions: list[Array] = []
    rewards: list[float] = []
    next_states: list[Array] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []
    continuation: list[float] = []
    returns: list[float] = []
    try:
        for episode in range(int(n_rollouts)):
            obs = _reset_env(env, rng)
            total = 0.0
            discount = 1.0
            for t in range(min(int(horizon), int(bundle.max_steps))):
                action = bundle.target_policy.sample(obs.reshape(1, -1), rng).reshape(-1)
                next_obs_raw, reward, terminated, truncated, _ = env.step(action.astype(env.action_space.dtype, copy=False))
                next_obs = np.asarray(next_obs_raw, dtype=np.float64).reshape(-1)
                done = bool(terminated or truncated)
                states.append(obs.copy())
                actions.append(action.astype(np.float64, copy=True))
                rewards.append(float(reward))
                next_states.append(next_obs.copy())
                episode_ids.append(int(episode))
                timesteps.append(int(t))
                continuation.append(0.0 if done else 1.0)
                total += discount * float(reward)
                discount *= float(gamma)
                obs = next_obs
                if done:
                    break
            returns.append(float(total))
    finally:
        env.close()
    state_arr = np.asarray(states, dtype=np.float64)
    next_state_arr = np.asarray(next_states, dtype=np.float64)
    tail_actions = bundle.target_policy.sample(next_state_arr, rng)
    return TargetRollouts(
        states=state_arr,
        actions=np.asarray(actions, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        next_states=next_state_arr,
        episode_ids=np.asarray(episode_ids),
        timesteps=np.asarray(timesteps),
        continuation=np.asarray(continuation, dtype=np.float64),
        tail_actions=np.asarray(tail_actions, dtype=np.float64),
        returns=np.asarray(returns, dtype=np.float64),
    )


def _prefix_rollouts(full: TargetRollouts, *, n_rollouts: int, horizon: int, gamma: float) -> TargetRollouts:
    mask = (np.asarray(full.episode_ids).reshape(-1) < int(n_rollouts)) & (np.asarray(full.timesteps).reshape(-1) < int(horizon))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("Requested validation prefix has no rows.")
    returns = []
    for episode in range(int(n_rollouts)):
        ep_idx = idx[full.episode_ids[idx] == episode]
        if ep_idx.size == 0:
            continue
        steps = full.timesteps[ep_idx]
        order = np.argsort(steps, kind="mergesort")
        ep_idx = ep_idx[order]
        discounts = float(gamma) ** np.arange(ep_idx.shape[0], dtype=np.float64)
        returns.append(float(np.sum(discounts * full.rewards[ep_idx])))
    return TargetRollouts(
        states=full.states[idx],
        actions=full.actions[idx],
        rewards=full.rewards[idx],
        next_states=full.next_states[idx],
        episode_ids=full.episode_ids[idx],
        timesteps=full.timesteps[idx],
        continuation=full.continuation[idx],
        tail_actions=full.tail_actions[idx],
        returns=np.asarray(returns, dtype=np.float64),
    )


def _estimate_policy_value_raw(bundle: PolicyBundle, *, gamma: float, rollouts: int, seed: int) -> tuple[float, float]:
    values = _collect_target_rollouts(
        bundle,
        gamma=float(gamma),
        n_rollouts=int(rollouts),
        horizon=int(bundle.max_steps),
        seed=int(seed),
    ).returns
    return float(np.mean(values)), _se(values)


def _one_se(rows: list[dict[str, Any]], score_key: str, se_key: str) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(_float(row.get(score_key, float("inf"))))]
    if not finite:
        return None
    best = min(finite, key=lambda row: _float(row[score_key]))
    threshold = _float(best[score_key]) + max(_float(best.get(se_key, 0.0)), 0.0)
    for row in rows:
        if row in finite and _float(row.get(score_key, float("inf"))) <= threshold:
            return row
    return best


def _best(rows: Iterable[dict[str, Any]], *, score_key: str) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(_float(row.get(score_key, float("inf"))))]
    return min(finite, key=lambda row: _float(row[score_key])) if finite else None


def _by_candidate(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("candidate_id", "")) == str(candidate_id):
            return row
    return None


def _finite_action_bounds(low: Array, high: Array) -> tuple[Array, Array]:
    lo = np.asarray(low, dtype=np.float64).copy()
    hi = np.asarray(high, dtype=np.float64).copy()
    lo[~np.isfinite(lo)] = -1.0
    hi[~np.isfinite(hi)] = 1.0
    same = hi <= lo
    lo[same] = -1.0
    hi[same] = 1.0
    return lo, hi


def _reset_env(env: Any, rng: np.random.Generator) -> Array:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    return np.asarray(obs, dtype=np.float64).reshape(-1)


def _se(values: Array) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _candidate_label(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "stable"
    return json.dumps(_jsonable(overrides), sort_keys=True, separators=(",", ":"))


def _with_common(row: dict[str, Any], common: dict[str, Any]) -> dict[str, Any]:
    out = dict(common)
    out.update(row)
    return _jsonable(out)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existing_fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_fields = list(reader.fieldnames or [])
    fields = list(existing_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    rewrite = bool(existing_fields) and fields != existing_fields
    prior: list[dict[str, Any]] = []
    if rewrite:
        with path.open("r", newline="") as fh:
            prior = list(csv.DictReader(fh))
    mode = "w" if rewrite or not path.exists() or path.stat().st_size == 0 else "a"
    with path.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            for row in prior:
                writer.writerow(row)
        for row in rows:
            writer.writerow(row)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _completed_cells(path: Path) -> set[tuple[str, int]]:
    rows = _read_csv(path)
    out: set[tuple[str, int]] = set()
    for row in rows:
        try:
            out.add((str(row["setting"]), int(row["seed"])))
        except Exception:
            continue
    return out


def _summarize_selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("package", "")),
            str(row.get("selector", "")),
            int(float(row.get("validation_rollouts", 0) or 0)),
            int(float(row.get("validation_horizon", 0) or 0)),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (package, selector, rollouts, horizon), group in sorted(groups.items()):
        regrets = np.asarray([_float(row.get("regret", float("nan"))) for row in group], dtype=np.float64)
        errors = np.asarray([_float(row.get("selected_true_error", float("nan"))) for row in group], dtype=np.float64)
        selected_oracle = np.asarray([_float(row.get("selected_oracle", 0.0)) for row in group], dtype=np.float64)
        tail = np.asarray([_float(row.get("truncation_tail_mass_mean", float("nan"))) for row in group], dtype=np.float64)
        summary.append(
            {
                "package": package,
                "selector": selector,
                "validation_rollouts": rollouts,
                "validation_horizon": horizon,
                "cells": len(group),
                "oracle_selection_rate": _nanmean(selected_oracle),
                "mean_regret": _nanmean(regrets),
                "median_regret": _nanmedian(regrets),
                "mean_selected_true_error": _nanmean(errors),
                "median_selected_true_error": _nanmedian(errors),
                "mean_tail_mass": _nanmean(tail),
            }
        )
    return summary


def _compact_gate_report(summary_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    del summary_rows
    checks = []
    target_rows = [
        row
        for row in selected_rows
        if not str(row.get("selector", "")).startswith("oracle") and str(row.get("selector", "")) != "proxy_cv"
    ]
    selection_failures = sum(1 for row in target_rows if _float(row.get("selection_failed", 0.0)) > 0.0)
    checks.append(
        {
            "name": "no_target_validation_selection_failures",
            "passed": bool(target_rows) and selection_failures == 0,
            "value": float(selection_failures),
            "threshold": 0.0,
        }
    )

    horizons: dict[int, list[float]] = {}
    for row in selected_rows:
        try:
            horizon = int(float(row.get("validation_horizon", 0)))
        except (TypeError, ValueError):
            continue
        horizons.setdefault(horizon, []).append(_float(row.get("truncation_tail_mass_mean", float("nan"))))
    tail_by_horizon = [(horizon, _nanmean(np.asarray(values, dtype=np.float64))) for horizon, values in sorted(horizons.items())]
    tail_decreases = len(tail_by_horizon) >= 2 and all(
        tail_by_horizon[idx + 1][1] <= tail_by_horizon[idx][1] + 1e-12 for idx in range(len(tail_by_horizon) - 1)
    )
    checks.append(
        {
            "name": "truncation_tail_mass_decreases_with_horizon",
            "passed": bool(tail_decreases),
            "value": tail_by_horizon,
            "threshold": "nonincreasing",
        }
    )

    rate_checks = [
        ("fqe_target_n_step_td_min_oracle_rate", "fqe", "target_n_step_td_min", 0.80),
        ("occupancy_scalar_finite_prefix_min_oracle_rate", "occupancy_ratio", "scalar_finite_prefix_min", 0.80),
        ("occupancy_target_discounted_moments_min_oracle_rate", "occupancy_ratio", "target_discounted_moments_min", 0.50),
    ]
    for name, package, selector, threshold in rate_checks:
        rate, count = _selector_oracle_rate(selected_rows, package=package, selector=selector)
        checks.append(
            {
                "name": name,
                "passed": bool(count > 0 and np.isfinite(rate) and rate >= threshold),
                "value": rate,
                "threshold": threshold,
                "cells": int(count),
            }
        )
    return {
        "passed": bool(checks and all(bool(check["passed"]) for check in checks)),
        "checks": checks,
    }


def _selector_oracle_rate(rows: list[dict[str, Any]], *, package: str, selector: str) -> tuple[float, int]:
    values = [
        _float(row.get("selected_oracle", float("nan")))
        for row in rows
        if str(row.get("package", "")) == str(package) and str(row.get("selector", "")) == str(selector)
    ]
    if not values:
        return float("nan"), 0
    return _nanmean(np.asarray(values, dtype=np.float64)), len(values)


def _nanmean(x: Array) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmedian(x: Array) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _write_report(path: Path, summary_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Gym Target-Validation Tuning Benchmark",
        "",
        "This benchmark fits each candidate once per setting/seed and then varies target-policy validation rollout count and finite horizon.",
        "",
        "## Aggregate Selector Summary",
        "",
        "| package | selector | rollouts | horizon | cells | oracle rate | median regret | median error | mean tail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {package} | {selector} | {validation_rollouts} | {validation_horizon} | {cells} | {oracle_selection_rate:.3g} | {median_regret:.3g} | {median_selected_true_error:.3g} | {mean_tail_mass:.3g} |".format(
                **row
            )
        )
    lines.extend(["", "## Selected Rows", ""])
    for row in selected_rows[:200]:
        lines.append(
            "- {package} {selector} setting={setting} seed={seed} H={validation_horizon} n={validation_rollouts}: selected={selected_candidate_id}, oracle={oracle_candidate_id}, regret={regret}".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
