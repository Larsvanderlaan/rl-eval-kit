#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.calibration.protocols import (  # noqa: E402
    ProtocolContext,
    _evaluate_row,
    run_cross_calibration,
    run_cross_calibrations,
    run_no_split,
    run_same_fraction_uncalibrated,
    run_split,
    run_uncalibrated_all_data,
)
from FQE_calibration_neurips.src.calibration.calibrators import fit_calibrator, fit_value_bellman_calibrator  # noqa: E402
from FQE_calibration_neurips.src.calibration.targets import (  # noqa: E402
    action_importance_weights,
    importance_weight_diagnostics,
    policy_value_predictions,
    value_calibration_arrays,
)
from FQE_calibration_neurips.src.comparison.baselines import run_split_comparator  # noqa: E402
from FQE_calibration_neurips.src.data import sample_initial_eval_states, sample_transition_batch  # noqa: E402
from FQE_calibration_neurips.src.estimators.baselines import fit_estimator  # noqa: E402
from FQE_calibration_neurips.src.environments import (  # noqa: E402
    NonlinearMDP,
    NonlinearMDPConfig,
    monte_carlo_oracle_value,
    monte_carlo_q_values,
    monte_carlo_v_values,
    monte_carlo_v_values_direct,
)
from FQE_calibration_neurips.src.policies import make_policy_pair  # noqa: E402
from FQE_calibration_neurips.src.utils import ensure_dir, load_config, timed, write_json  # noqa: E402


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _debug_override(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg["replications"] = min(int(cfg.get("replications", 1)), 1)
    cfg["sample_sizes"] = [min(int(cfg.get("sample_sizes", [120])[0]), 120)]
    cfg["state_dimensions"] = [min(int(cfg.get("state_dimensions", [4])[0]), 4)]
    cfg["coverage_settings"] = cfg.get("coverage_settings", ["good"])[:1]
    cfg["reward_noise_settings"] = [float(cfg.get("reward_noise_settings", [0.1])[0])]
    cfg["baseline_learners"] = ["random_feature_fqe", "neural_fqe"]
    cfg["calibration_protocols"] = ["cross", "split", "no_split"]
    cfg["calibrators"] = ["linear", "histogram", "isotonic", "isotonic_histogram"]
    cfg["calibration_targets"] = ["value_bellman"]
    cfg["importance_weight_scheme"] = "action_ratio"
    cfg["importance_weight_clip"] = float(cfg.get("importance_weight_clip", 20.0))
    cfg["normalize_importance_weights"] = bool(cfg.get("normalize_importance_weights", True))
    cfg["value_calibration_iterations"] = min(int(cfg.get("value_calibration_iterations", 4)), 2)
    cfg["split_fractions"] = [0.8]
    cfg["oracle_rollouts"] = min(int(cfg.get("oracle_rollouts", 300)), 120)
    cfg["initial_eval_states"] = min(int(cfg.get("initial_eval_states", 300)), 120)
    cfg["diagnostic_test_transitions"] = min(int(cfg.get("diagnostic_test_transitions", cfg.get("test_transitions", 300))), 160)
    cfg["true_q_rollouts_per_state"] = min(int(cfg.get("true_q_rollouts_per_state", 1)), 1)
    cfg["calibration_error_bins"] = min(int(cfg.get("calibration_error_bins", 50)), 8)
    cfg["calibration_error_min_bin_size"] = min(int(cfg.get("calibration_error_min_bin_size", 20)), 5)
    cfg["interval_bootstrap_reps"] = min(int(cfg.get("interval_bootstrap_reps", 200)), 80)
    learner_params = dict(cfg.get("learner_params", {}))
    learner_params["neural_fqe"] = {
        "hidden_dims": [16],
        "n_iters": 3,
        "epochs_per_iter": 2,
        "batch_size": 64,
        "lr": 0.002,
    }
    learner_params["random_feature_fqe"] = {"n_components": 32, "n_iters": 4, "ridge": 0.01}
    cfg["learner_params"] = learner_params
    return cfg


def _learner_params(config: dict[str, Any], learner: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(config.get("learner_params", {}).get(learner, {}))
    params = _deep_merge(params, override or {})
    if "hidden_dims" in params:
        params["hidden_dims"] = tuple(params["hidden_dims"])
    return params


def _current_retrain_params(config: dict[str, Any], learner: str, learner_variant: str, base_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(base_params)
    overrides = config.get("current_retrain_learner_params", {})
    if isinstance(overrides, dict):
        common = overrides.get("all", {})
        if isinstance(common, dict):
            params = _deep_merge(params, common)
        for key in (learner, learner_variant):
            specific = overrides.get(str(key), {})
            if isinstance(specific, dict):
                params = _deep_merge(params, specific)
    if "hidden_dims" in params:
        params["hidden_dims"] = tuple(params["hidden_dims"])
    return params


def _learner_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("learner_variants", {})
    names = config.get("baseline_learners")
    if isinstance(variants, dict) and variants:
        names = names or list(variants)
        specs = []
        for name in names:
            raw = dict(variants.get(str(name), {}))
            base = str(raw.get("base_learner", raw.get("learner", name)))
            specs.append(
                {
                    "learner_variant": str(name),
                    "base_learner": base,
                    "learner_quality_regime": str(raw.get("learner_quality_regime", "well_tuned")),
                    "calibration_difficulty": str(raw.get("calibration_difficulty", config.get("calibration_difficulty", "already_well_calibrated"))),
                    "main_figure_role": str(raw.get("main_figure_role", config.get("main_figure_role", "main"))),
                    "params": dict(raw.get("params", raw.get("learner_params", {}))),
                }
            )
        return specs
    return [
        {
            "learner_variant": str(learner),
            "base_learner": str(learner),
            "learner_quality_regime": str(config.get("learner_quality_regime", "well_tuned")),
            "calibration_difficulty": str(config.get("calibration_difficulty", "already_well_calibrated")),
            "main_figure_role": str(config.get("main_figure_role", "main")),
            "params": {},
        }
        for learner in (names or ["random_feature_fqe"])
    ]


def _calibration_difficulty(
    config: dict[str, Any],
    coverage: str,
    misspecification: str,
    learner_spec: dict[str, Any],
) -> str:
    if misspecification == "affine":
        return "affine_miscalibrated"
    if misspecification == "monotone_distortion":
        return "monotone_miscalibrated"
    if misspecification == "nonmonotone":
        return "nonmonotone_error"
    if misspecification == "bellman_incomplete":
        return "bellman_incomplete"
    if coverage in {"severe", "extrapolation"}:
        return "coverage_limited"
    return str(learner_spec.get("calibration_difficulty", config.get("calibration_difficulty", "already_well_calibrated")))


def _calibration_data_provenance(row: dict[str, Any]) -> str:
    protocol = str(row.get("calibration_protocol", ""))
    if protocol == "cross":
        return "pooled_out_of_fold_training_predictions"
    if protocol == "split":
        return "heldout_calibration_split"
    if protocol == "no_split":
        return "training_data_in_sample"
    if protocol == "recent_heldout":
        return "recent_current_regime_heldout"
    if protocol == "current_retrain_small":
        return "none"
    if protocol in {
        "fine_tuning_all_layers",
        "fine_tuning_final_layer",
        "offset_correction",
        "residual_correction",
        "regularized_toward_first_stage",
    }:
        return "heldout_calibration_split_comparator"
    return "none"


def _common_env_config(config: dict[str, Any], *, state_dim: int, reward_noise: float, seed: int) -> dict[str, Any]:
    return {
        "state_dim": int(state_dim),
        "n_actions": int(config.get("n_actions", 3)),
        "gamma": float(config.get("gamma", 0.95)),
        "reward_noise": float(reward_noise),
        "transition_noise": float(config.get("transition_noise", 0.25)),
        "horizon": int(config.get("horizon", 80)),
        "extrapolation_scale": float(config.get("extrapolation_scale", 0.0)),
        "reference_shift_scale": float(config.get("reference_shift_scale", 0.0)),
        "misspecification": str(config.get("misspecification", "none")),
        "seed": int(seed),
    }


def _augment_result_row(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    run_mode: str,
    suite_name: str,
    environment_tier: str,
    policy_shift: float,
    misspecification: str,
    train_seed: int,
    test_seed: int,
    oracle_seed: int,
    diagnostic_seed: int,
    learner_spec: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    split_fraction = float(out.get("train_fraction", 1.0))
    out.update(
        {
            "run_mode": run_mode,
            "suite_name": suite_name,
            "environment_tier": environment_tier,
            "policy_shift_setting": float(policy_shift),
            "misspecification_setting": misspecification,
            "oracle_value_method": str(config.get("oracle_value_method", "independent_monte_carlo_rollout")),
            "train_data_provenance": f"offline_behavior_batch_seed={train_seed};not_test_or_oracle",
            "calibration_data_provenance": _calibration_data_provenance(out),
            "test_data_provenance": (
                f"independent_test_seed={test_seed};independent_oracle_seed={oracle_seed};"
                f"independent_diagnostic_seed={diagnostic_seed}"
            ),
            "split_fraction": split_fraction,
            "learner_variant": str(learner_spec.get("learner_variant", out.get("baseline_learner", "unknown"))),
            "learner_quality_regime": str(learner_spec.get("learner_quality_regime", "well_tuned")),
            "calibration_difficulty": str(learner_spec.get("calibration_difficulty", "already_well_calibrated")),
            "main_figure_role": str(learner_spec.get("main_figure_role", "main")),
            "main_evidence_eligible": not bool(out.get("failure_flag", False)),
        }
    )
    return out


def _temporal_augment_row(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    run_mode: str,
    suite_name: str,
    policy_shift: float,
    old_seed: int,
    recent_seed: int,
    test_seed: int,
    oracle_seed: int,
    diagnostic_seed: int,
    learner_spec: dict[str, Any],
    n_old: int,
    n_recent: int,
) -> dict[str, Any]:
    out = _augment_result_row(
        row,
        config=config,
        run_mode=run_mode,
        suite_name=suite_name,
        environment_tier="temporal_reward_shift",
        policy_shift=policy_shift,
        misspecification="temporal_reward_shift",
        train_seed=old_seed,
        test_seed=test_seed,
        oracle_seed=oracle_seed,
        diagnostic_seed=diagnostic_seed,
        learner_spec=learner_spec,
    )
    protocol = str(out.get("calibration_protocol", ""))
    if protocol == "current_retrain_small":
        out["train_data_provenance"] = f"recent_current_regime_retrain_seed={recent_seed};not_test_or_oracle"
        out["calibration_data_provenance"] = "none"
    else:
        out["train_data_provenance"] = f"old_regime_behavior_batch_seed={old_seed};not_test_or_oracle"
        if protocol == "recent_heldout":
            if str(config.get("temporal_calibration_kind", "bellman")) == "rollout_value":
                out["calibration_data_provenance"] = (
                    f"recent_current_regime_rollout_value_heldout_seed={recent_seed};not_test_or_oracle"
                )
            else:
                out["calibration_data_provenance"] = (
                    f"recent_current_regime_heldout_seed={recent_seed};not_test_or_oracle"
                )
    out["test_data_provenance"] = (
        f"current_regime_independent_test_seed={test_seed};"
        f"independent_oracle_seed={oracle_seed};independent_diagnostic_seed={diagnostic_seed}"
    )
    out["temporal_old_train_size"] = int(n_old)
    out["temporal_recent_current_size"] = int(n_recent)
    out["temporal_recent_policy"] = str(config.get("recent_calibration_policy", "behavior"))
    out["temporal_calibration_kind"] = str(config.get("temporal_calibration_kind", "bellman"))
    out["temporal_reward_shift_intercept"] = float(config.get("current_reward_shift_intercept", 0.6))
    out["temporal_reward_shift_scale"] = float(config.get("current_reward_shift_scale", 1.25))
    return out


def _fit_recent_value_calibrator(
    calibrator_name: str,
    model: object,
    recent_batch: object,
    target_policy: object,
    gamma: float,
    config: dict[str, Any],
) -> object:
    values, next_values, rewards = value_calibration_arrays(model, recent_batch, target_policy)
    weights = action_importance_weights(
        recent_batch,
        float(config.get("importance_weight_clip", 20.0)),
        bool(config.get("normalize_importance_weights", True)),
    )
    params = dict(config.get("calibrator_params", {}))
    cal = fit_value_bellman_calibrator(
        calibrator_name,
        values,
        next_values,
        rewards,
        gamma,
        weights,
        n_iterations=int(
            params.get(
                "value_calibration_iterations",
                params.get("bellman_iterations", config.get("value_calibration_iterations", 4)),
            )
        ),
        n_bins=int(params.get("n_bins", 10)),
        bin_strategy=str(params.get("bin_strategy", "quantile")),
        min_bin_size=int(params.get("min_bin_size", 20)),
    )
    cal.diagnostics.update(
        importance_weight_diagnostics(
            weights,
            float(config.get("importance_weight_clip", 20.0)),
            bool(config.get("normalize_importance_weights", True)),
        )
    )
    return cal


def _fit_recent_rollout_value_calibrator(
    calibrator_name: str,
    model: object,
    recent_states,
    recent_value_targets,
    target_policy: object,
    config: dict[str, Any],
    recent_batch: object | None = None,
    gamma: float | None = None,
) -> object:
    method = calibrator_name
    values = policy_value_predictions(model, recent_states, target_policy)
    params = dict(config.get("calibrator_params", {}))
    cal = fit_calibrator(
        method,
        values,
        recent_value_targets,
        n_bins=int(params.get("n_bins", 10)),
        bin_strategy=str(params.get("bin_strategy", "quantile")),
        min_bin_size=int(params.get("min_bin_size", 20)),
    )
    refine_iterations = int(config.get("recent_rollout_bellman_refine_iterations", 0))
    if refine_iterations > 0:
        if recent_batch is None or gamma is None:
            raise ValueError("recent_batch and gamma are required for rollout Bellman refinement.")
        batch_values, batch_next_values, batch_rewards = value_calibration_arrays(model, recent_batch, target_policy)
        weights = action_importance_weights(
            recent_batch,
            float(config.get("importance_weight_clip", 20.0)),
            bool(config.get("normalize_importance_weights", True)),
        )
        for _ in range(refine_iterations):
            target = batch_rewards + float(gamma) * cal.predict(batch_next_values)
            cal = fit_calibrator(
                method,
                batch_values,
                target,
                n_bins=int(params.get("n_bins", 10)),
                bin_strategy=str(params.get("bin_strategy", "quantile")),
                min_bin_size=int(params.get("min_bin_size", 20)),
                sample_weight=weights,
            )
    cal.diagnostics = {
        "calibration_object": "value",
        "value_calibrator_method": str(calibrator_name),
        "value_calibration_source": "recent_current_rollout_value"
        if refine_iterations <= 0
        else "recent_current_rollout_value_bellman_refined",
        "value_calibration_iterations": float(refine_iterations),
        "raw_value_min": float(pd.Series(values).min()),
        "raw_value_max": float(pd.Series(values).max()),
        "calibrated_value_min": float(pd.Series(cal.predict(values)).min()),
        "calibrated_value_max": float(pd.Series(cal.predict(values)).max()),
    }
    return cal


def run_temporal_reward_shift_config(
    config: dict[str, Any],
    output_dir: str | Path,
    debug: bool = False,
    run_mode: str | None = None,
    suite_name: str | None = None,
) -> list[dict[str, Any]]:
    if debug:
        config = _debug_override(config)
    rows: list[dict[str, Any]] = []
    base_seed = int(config.get("seed", 123))
    run_mode = run_mode or ("debug" if debug else str(config.get("run_mode", "standalone")))
    suite_name = suite_name or str(config.get("suite_name", Path(str(output_dir)).name or "temporal_reward_shift_sweep"))
    seed_stride = int(config.get("replication_seed_stride", 10_000))
    old_sizes = config.get("old_train_sizes", config.get("sample_sizes", [2000]))
    recent_sizes = config.get("recent_current_sizes", [config.get("recent_current_size", 250)])
    coverage_settings = config.get("coverage_settings", ["moderate"])
    reward_noise_settings = config.get("reward_noise_settings", [0.1])
    target_type = "value_bellman"
    for rep in range(int(config.get("replications", 1))):
        seed = base_seed + rep * seed_stride
        for n_old in old_sizes:
            for n_recent in recent_sizes:
                for state_dim in config.get("state_dimensions", [6]):
                    for coverage in coverage_settings:
                        for reward_noise in reward_noise_settings:
                            env_seed = seed + 1
                            old_cfg = NonlinearMDPConfig(
                                **_common_env_config(config, state_dim=int(state_dim), reward_noise=float(reward_noise), seed=env_seed)
                            )
                            current_cfg = NonlinearMDPConfig(
                                **{
                                    **_common_env_config(
                                        config, state_dim=int(state_dim), reward_noise=float(reward_noise), seed=env_seed
                                    ),
                                    "reward_shift_intercept": float(config.get("current_reward_shift_intercept", 0.6)),
                                    "reward_shift_scale": float(config.get("current_reward_shift_scale", 1.25)),
                                }
                            )
                            old_env = NonlinearMDP(old_cfg)
                            current_env = NonlinearMDP(current_cfg)
                            shift = float(config.get("policy_shift", {}).get(str(coverage), 0.5))
                            target_policy, behavior_policy = make_policy_pair(
                                current_env.state_dim,
                                current_env.n_actions,
                                shift=shift,
                                coverage=str(coverage),
                                seed=seed + 2,
                            )
                            recent_policy_name = str(config.get("recent_calibration_policy", "behavior"))
                            if recent_policy_name not in {"behavior", "target"}:
                                raise ValueError("recent_calibration_policy must be 'behavior' or 'target'.")
                            recent_behavior_policy = target_policy if recent_policy_name == "target" else behavior_policy
                            old_seed, recent_seed = seed + 3, seed + 4
                            test_seed, oracle_seed, diagnostic_seed = seed + 5, seed + 6, seed + 7
                            old_train = sample_transition_batch(old_env, behavior_policy, target_policy, int(n_old), old_seed)
                            recent_current = sample_transition_batch(
                                current_env, recent_behavior_policy, target_policy, int(n_recent), recent_seed
                            )
                            test = sample_transition_batch(
                                current_env,
                                behavior_policy,
                                target_policy,
                                int(config.get("test_transitions", n_recent)),
                                test_seed,
                            )
                            diagnostic_test = sample_transition_batch(
                                current_env,
                                behavior_policy,
                                target_policy,
                                int(config.get("diagnostic_test_transitions", config.get("test_transitions", n_recent))),
                                diagnostic_seed,
                            )
                            true_v_rollouts = int(config.get("true_v_rollouts_per_state", 1))
                            if bool(config.get("direct_true_v_rollout", True)):
                                diagnostic_true_v_values = monte_carlo_v_values_direct(
                                    current_env, target_policy, diagnostic_test.states, true_v_rollouts, seed + 9
                                )
                            else:
                                diagnostic_true_v_values = monte_carlo_v_values(
                                    current_env, target_policy, diagnostic_test.states, true_v_rollouts, seed + 9
                                )
                            diagnostic_true_q_values = None
                            if bool(config.get("compute_true_q_mse", False)):
                                diagnostic_true_q_values = monte_carlo_q_values(
                                    current_env,
                                    target_policy,
                                    diagnostic_test.states,
                                    diagnostic_test.actions,
                                    int(config.get("true_q_rollouts_per_state", 1)),
                                    seed + 8,
                                )
                            recent_value_targets = None
                            temporal_calibration_kind = str(config.get("temporal_calibration_kind", "bellman"))
                            if temporal_calibration_kind == "rollout_value":
                                recent_value_targets = monte_carlo_v_values_direct(
                                    current_env,
                                    target_policy,
                                    recent_current.states,
                                    int(config.get("recent_value_rollouts_per_state", 5)),
                                    seed + 11,
                                )
                            elif temporal_calibration_kind != "bellman":
                                raise ValueError("temporal_calibration_kind must be 'bellman' or 'rollout_value'.")
                            initial_states = sample_initial_eval_states(
                                current_env,
                                int(config.get("initial_eval_states", 1000)),
                                seed + 10,
                                shifted=bool(config.get("shifted_initial_eval_states", False)),
                            )
                            oracle = monte_carlo_oracle_value(
                                current_env,
                                target_policy,
                                int(config.get("oracle_rollouts", 1000)),
                                oracle_seed,
                                initial_states=initial_states[: int(config.get("oracle_rollouts", 1000))],
                            )
                            for learner_spec_raw in _learner_specs(config):
                                learner_spec = dict(learner_spec_raw)
                                learner_spec["calibration_difficulty"] = "temporal_reward_shift"
                                learner_spec["learner_quality_regime"] = str(
                                    learner_spec.get("learner_quality_regime", "temporally_shifted")
                                )
                                learner = str(learner_spec["base_learner"])
                                learner_params = _learner_params(config, learner, dict(learner_spec.get("params", {})))
                                ctx = ProtocolContext(
                                    batch=old_train,
                                    test_batch=test,
                                    initial_states=initial_states,
                                    oracle_value=oracle,
                                    env=current_env,
                                    target_policy=target_policy,
                                    learner=learner,
                                    learner_params=learner_params,
                                    gamma=current_env.gamma,
                                    seed=seed,
                                    coverage=str(coverage),
                                    reward_noise=float(reward_noise),
                                    diagnostic_batch=diagnostic_test,
                                    diagnostic_true_q_values=diagnostic_true_q_values,
                                    diagnostic_true_v_values=diagnostic_true_v_values,
                                    calibration_error_bins=int(config.get("calibration_error_bins", 50)),
                                    calibration_error_min_bin_size=int(config.get("calibration_error_min_bin_size", 20)),
                                    calibration_error_folds=int(config.get("calibration_error_folds", 5)),
                                    importance_weight_scheme=str(config.get("importance_weight_scheme", "action_ratio")),
                                    importance_weight_clip=float(config.get("importance_weight_clip", 20.0)),
                                    normalize_importance_weights=bool(config.get("normalize_importance_weights", True)),
                                    value_calibration_iterations=int(config.get("value_calibration_iterations", 4)),
                                    interval_bootstrap_reps=int(config.get("interval_bootstrap_reps", 200)),
                                )
                                with timed() as tb:
                                    old_fit = fit_estimator(
                                        learner,
                                        old_train,
                                        current_env.n_actions,
                                        target_policy,
                                        current_env.gamma,
                                        learner_params,
                                        seed,
                                    )
                                old_model = old_fit.model
                                raw_row = _evaluate_row(
                                    ctx,
                                    old_model,
                                    None,
                                    {
                                        "calibrated": False,
                                        "protocol": "uncalibrated_all_data",
                                        "calibrator": "none",
                                        "calibration_target": target_type,
                                        "all_data": True,
                                        "sample_splitting": False,
                                        "train_fraction": 1.0,
                                        "calibration_fraction": 0.0,
                                    },
                                    tb.seconds,
                                )
                                rows.append(
                                    _temporal_augment_row(
                                        raw_row,
                                        config=config,
                                        run_mode=run_mode,
                                        suite_name=suite_name,
                                        policy_shift=shift,
                                        old_seed=old_seed,
                                        recent_seed=recent_seed,
                                        test_seed=test_seed,
                                        oracle_seed=oracle_seed,
                                        diagnostic_seed=diagnostic_seed,
                                        learner_spec=learner_spec,
                                        n_old=int(n_old),
                                        n_recent=int(n_recent),
                                    )
                                )
                                for calibrator_name in [str(x) for x in config.get("calibrators", ["linear", "isotonic"])]:
                                    with timed() as tb_cal:
                                        if temporal_calibration_kind == "rollout_value":
                                            cal = _fit_recent_rollout_value_calibrator(
                                                calibrator_name,
                                                old_model,
                                                recent_current.states,
                                                recent_value_targets,
                                                target_policy,
                                                config,
                                                recent_batch=recent_current,
                                                gamma=current_env.gamma,
                                            )
                                        else:
                                            cal = _fit_recent_value_calibrator(
                                                calibrator_name,
                                                old_model,
                                                recent_current,
                                                target_policy,
                                                current_env.gamma,
                                                config,
                                            )
                                    cal_row = _evaluate_row(
                                        ctx,
                                        old_model,
                                        cal,
                                        {
                                            "calibrated": True,
                                            "protocol": "recent_heldout",
                                            "calibrator": calibrator_name,
                                            "calibration_target": target_type,
                                            "all_data": False,
                                            "sample_splitting": True,
                                            "train_fraction": 1.0,
                                            "calibration_fraction": float(int(n_recent) / max(int(n_old), 1)),
                                        },
                                        tb.seconds + tb_cal.seconds,
                                    )
                                    rows.append(
                                        _temporal_augment_row(
                                            cal_row,
                                            config=config,
                                            run_mode=run_mode,
                                            suite_name=suite_name,
                                            policy_shift=shift,
                                            old_seed=old_seed,
                                            recent_seed=recent_seed,
                                            test_seed=test_seed,
                                            oracle_seed=oracle_seed,
                                            diagnostic_seed=diagnostic_seed,
                                            learner_spec=learner_spec,
                                            n_old=int(n_old),
                                            n_recent=int(n_recent),
                                        )
                                    )
                                with timed() as tb_retrain:
                                    retrain_params = _current_retrain_params(
                                        config,
                                        learner,
                                        str(learner_spec["learner_variant"]),
                                        learner_params,
                                    )
                                    recent_fit = fit_estimator(
                                        learner,
                                        recent_current,
                                        current_env.n_actions,
                                        target_policy,
                                        current_env.gamma,
                                        retrain_params,
                                        seed + 101,
                                    )
                                retrain_row = _evaluate_row(
                                    ctx,
                                    recent_fit.model,
                                    None,
                                    {
                                        "calibrated": False,
                                        "protocol": "current_retrain_small",
                                        "calibrator": "none",
                                        "calibration_target": target_type,
                                        "all_data": False,
                                        "sample_splitting": True,
                                        "train_fraction": float(int(n_recent) / max(int(n_old), 1)),
                                        "calibration_fraction": 0.0,
                                    },
                                    tb_retrain.seconds,
                                )
                                rows.append(
                                    _temporal_augment_row(
                                        retrain_row,
                                        config=config,
                                        run_mode=run_mode,
                                        suite_name=suite_name,
                                        policy_shift=shift,
                                        old_seed=old_seed,
                                        recent_seed=recent_seed,
                                        test_seed=test_seed,
                                        oracle_seed=oracle_seed,
                                        diagnostic_seed=diagnostic_seed,
                                        learner_spec=learner_spec,
                                        n_old=int(n_old),
                                        n_recent=int(n_recent),
                                    )
                                )
    if rows:
        all_keys = sorted({key for row in rows for key in row})
        rows = [{key: row.get(key, "") for key in all_keys} for row in rows]
    out_dir = ensure_dir(output_dir)
    raw_path = out_dir / ("debug_raw_results.csv" if debug else "raw_results.csv")
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    write_json(out_dir / ("debug_config.json" if debug else "config.json"), config)
    return rows


def run_config(
    config: dict[str, Any],
    output_dir: str | Path,
    debug: bool = False,
    run_mode: str | None = None,
    suite_name: str | None = None,
) -> list[dict[str, Any]]:
    if debug:
        config = _debug_override(config)
    if not bool(config.get("enable_legacy_q_calibration", False)):
        legacy_targets = set(config.get("calibration_targets", ["value_bellman"])) - {"value_bellman"}
        if legacy_targets:
            raise ValueError(
                "Q-space calibration targets are disabled by default. "
                f"Use calibration_targets: [value_bellman] or set enable_legacy_q_calibration=true for {sorted(legacy_targets)}."
            )
    if str(config.get("importance_weight_scheme", "action_ratio")) != "action_ratio":
        raise ValueError("Only importance_weight_scheme: action_ratio is implemented for value-space calibration.")
    if str(config.get("runner", "")) == "temporal_reward_shift":
        return run_temporal_reward_shift_config(config, output_dir, debug=debug, run_mode=run_mode, suite_name=suite_name)
    rows: list[dict[str, Any]] = []
    base_seed = int(config.get("seed", 123))
    run_mode = run_mode or ("debug" if debug else str(config.get("run_mode", "standalone")))
    suite_name = suite_name or str(config.get("suite_name", Path(str(output_dir)).name or "standalone"))
    misspecifications = config.get("misspecification_settings")
    if misspecifications is None:
        misspecifications = [config.get("misspecification", "none")]
    seed_stride = int(config.get("replication_seed_stride", 10_000))
    for rep in range(int(config.get("replications", 1))):
        seed = base_seed + rep * seed_stride
        for n in config.get("sample_sizes", [500]):
            for state_dim in config.get("state_dimensions", [6]):
                for coverage in config.get("coverage_settings", ["good"]):
                    for reward_noise in config.get("reward_noise_settings", [0.2]):
                        for misspecification in misspecifications:
                            misspecification = str(misspecification)
                            env_cfg = NonlinearMDPConfig(
                                state_dim=int(state_dim),
                                n_actions=int(config.get("n_actions", 3)),
                                gamma=float(config.get("gamma", 0.95)),
                                reward_noise=float(reward_noise),
                                transition_noise=float(config.get("transition_noise", 0.25)),
                                horizon=int(config.get("horizon", 80)),
                                extrapolation_scale=float(config.get("extrapolation_scale", 0.0)),
                                reference_shift_scale=float(config.get("reference_shift_scale", 0.0)),
                                misspecification=misspecification,
                                seed=seed + 1,
                            )
                            env = NonlinearMDP(env_cfg)
                            environment_tier = str(
                                config.get(
                                    "environment_tier",
                                    "well_specified" if misspecification == "well_specified_linear" else "nonlinear_synthetic",
                                )
                            )
                            shift = float(config.get("policy_shift", {}).get(coverage, 0.5))
                            target_policy, behavior_policy = make_policy_pair(
                                env.state_dim, env.n_actions, shift=shift, coverage=coverage, seed=seed + 2
                            )
                            train_seed, test_seed, oracle_seed, diagnostic_seed = seed + 3, seed + 4, seed + 6, seed + 7
                            train = sample_transition_batch(env, behavior_policy, target_policy, int(n), train_seed)
                            test = sample_transition_batch(
                                env, behavior_policy, target_policy, int(config.get("test_transitions", n)), test_seed
                            )
                            diagnostic_test = sample_transition_batch(
                                env,
                                behavior_policy,
                                target_policy,
                                int(config.get("diagnostic_test_transitions", config.get("test_transitions", n))),
                                diagnostic_seed,
                            )
                            true_q_rollouts = int(config.get("true_q_rollouts_per_state", 1))
                            true_v_rollouts = int(config.get("true_v_rollouts_per_state", true_q_rollouts))
                            if bool(config.get("compute_true_q_mse", True)):
                                diagnostic_true_q_values = monte_carlo_q_values(
                                    env,
                                    target_policy,
                                    diagnostic_test.states,
                                    diagnostic_test.actions,
                                    true_q_rollouts,
                                    seed + 8,
                                )
                            else:
                                diagnostic_true_q_values = None
                            if bool(config.get("direct_true_v_rollout", False)):
                                diagnostic_true_v_values = monte_carlo_v_values_direct(
                                    env,
                                    target_policy,
                                    diagnostic_test.states,
                                    true_v_rollouts,
                                    seed + 9,
                                )
                            else:
                                diagnostic_true_v_values = monte_carlo_v_values(
                                    env,
                                    target_policy,
                                    diagnostic_test.states,
                                    true_v_rollouts,
                                    seed + 9,
                                )
                            initial_states = sample_initial_eval_states(
                                env,
                                int(config.get("initial_eval_states", 1000)),
                                seed + 5,
                                shifted=bool(config.get("shifted_initial_eval_states", False)),
                            )
                            oracle = monte_carlo_oracle_value(
                                env,
                                target_policy,
                                int(config.get("oracle_rollouts", 1000)),
                                oracle_seed,
                                initial_states=initial_states[: int(config.get("oracle_rollouts", 1000))],
                            )

                            def append(row: dict[str, Any], learner_spec: dict[str, Any]) -> None:
                                rows.append(
                                    _augment_result_row(
                                        row,
                                        config=config,
                                        run_mode=run_mode,
                                        suite_name=suite_name,
                                        environment_tier=environment_tier,
                                        policy_shift=shift,
                                        misspecification=misspecification,
                                        train_seed=train_seed,
                                        test_seed=test_seed,
                                        oracle_seed=oracle_seed,
                                        diagnostic_seed=diagnostic_seed,
                                        learner_spec=learner_spec,
                                    )
                                )

                            for learner_spec_raw in _learner_specs(config):
                                learner_spec = dict(learner_spec_raw)
                                learner_spec["calibration_difficulty"] = _calibration_difficulty(
                                    config, str(coverage), misspecification, learner_spec
                                )
                                learner = str(learner_spec["base_learner"])
                                ctx = ProtocolContext(
                                    batch=train,
                                    test_batch=test,
                                    initial_states=initial_states,
                                    oracle_value=oracle,
                                    env=env,
                                    target_policy=target_policy,
                                    learner=learner,
                                    learner_params=_learner_params(config, learner, dict(learner_spec.get("params", {}))),
                                    gamma=env.gamma,
                                    seed=seed,
                                    coverage=str(coverage),
                                    reward_noise=float(reward_noise),
                                    diagnostic_batch=diagnostic_test,
                                    diagnostic_true_q_values=diagnostic_true_q_values,
                                    diagnostic_true_v_values=diagnostic_true_v_values,
                                    calibration_error_bins=int(config.get("calibration_error_bins", 50)),
                                    calibration_error_min_bin_size=int(config.get("calibration_error_min_bin_size", 20)),
                                    calibration_error_folds=int(config.get("calibration_error_folds", 5)),
                                    importance_weight_scheme=str(config.get("importance_weight_scheme", "action_ratio")),
                                    importance_weight_clip=float(config.get("importance_weight_clip", 20.0)),
                                    normalize_importance_weights=bool(config.get("normalize_importance_weights", True)),
                                    value_calibration_iterations=int(
                                        config.get(
                                            "value_calibration_iterations",
                                            config.get("calibrator_params", {}).get("value_calibration_iterations", 4),
                                        )
                                    ),
                                    interval_bootstrap_reps=int(config.get("interval_bootstrap_reps", 200)),
                                )
                                for target_type in config.get("calibration_targets", ["value_bellman"]):
                                    append(run_uncalibrated_all_data(ctx, str(target_type)), learner_spec)
                                    for frac in config.get("split_fractions", [0.8]):
                                        append(run_same_fraction_uncalibrated(ctx, float(frac), str(target_type)), learner_spec)
                                    cal_params = dict(config.get("calibrator_params", {}))
                                    calibrators = [str(calibrator) for calibrator in config.get("calibrators", ["linear"])]
                                    if "cross" in config.get("calibration_protocols", []):
                                        for row in run_cross_calibrations(
                                            ctx,
                                            calibrators,
                                            str(target_type),
                                            int(config.get("cross_folds", 5)),
                                            cal_params,
                                        ):
                                            append(row, learner_spec)
                                    for calibrator in config.get("calibrators", ["linear"]):
                                        if "no_split" in config.get("calibration_protocols", []):
                                            append(run_no_split(ctx, str(calibrator), str(target_type), cal_params), learner_spec)
                                        if "split" in config.get("calibration_protocols", []):
                                            for frac in config.get("split_fractions", [0.8]):
                                                append(run_split(ctx, str(calibrator), str(target_type), float(frac), cal_params), learner_spec)
                                    for comparator in config.get("split_comparators", []):
                                        for frac in config.get("split_fractions", [0.8]):
                                            append(
                                                run_split_comparator(
                                                    ctx,
                                                    str(comparator),
                                                    float(frac),
                                                    str(target_type),
                                                    dict(config.get("comparator_params", {})),
                                                ),
                                                learner_spec,
                                            )
    if rows:
        all_keys = sorted({key for row in rows for key in row})
        rows = [{key: row.get(key, "") for key in all_keys} for row in rows]
    out_dir = ensure_dir(output_dir)
    raw_path = out_dir / ("debug_raw_results.csv" if debug else "raw_results.csv")
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    write_json(out_dir / ("debug_config.json" if debug else "config.json"), config)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NeurIPS calibration experiment suite.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.replications is not None:
        config["replications"] = int(args.replications)
    rows = run_config(config, args.output_dir, debug=args.debug)
    print(f"Wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
