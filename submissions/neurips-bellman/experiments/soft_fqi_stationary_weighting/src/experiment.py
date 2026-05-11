from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .algorithms import FQIConfig, run_minimax_soft_q, run_neural_fqi, run_population_linear_fqi, run_sample_linear_fqi
from .data import sample_transition_batch
from .env import EnvConfig, GridConfig, build_grid_mdp
from .features import QFeatureMap, RatioFeatureMap
from .metrics import weighted_design_condition_number
from .soft_dp import (
    evaluate_soft_policy_value,
    soft_value_iteration,
    softmax_policy,
    state_action_distribution,
    stationary_state_distribution,
)
from .weights import (
    estimate_moment_weights,
    oracle_sample_weights,
    ratio_quality,
    stabilize_weights,
    summarize_weights,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dataclass_from_dict(cls: type, values: dict[str, Any]) -> Any:
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _merged_section(config: dict[str, Any], stage: dict[str, Any], section: str) -> dict[str, Any]:
    merged = dict(config.get(section, {}) or {})
    merged.update(stage.get(section, {}) or {})
    return merged


def _behavior_policy(pi_star: np.ndarray, pi_decoy: np.ndarray, mixture: dict[str, float]) -> np.ndarray:
    uniform = np.ones_like(pi_star) / pi_star.shape[1]
    policy = (
        float(mixture.get("target", 0.0)) * pi_star
        + float(mixture.get("decoy", 0.0)) * pi_decoy
        + float(mixture.get("uniform", 0.0)) * uniform
    )
    policy = np.maximum(policy, 1e-12)
    return policy / np.maximum(np.sum(policy, axis=1, keepdims=True), 1e-300)


def _safe_name(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return str(value).replace(".", "p").replace("-", "m")


def _parse_safe_float(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def _q_feature_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    algo = config.get("algorithm", {}) or {}
    feature_config = dict(config.get("q_features", {}) or {})
    q_class = str(config.get("q_class", algo.get("q_class", feature_config.get("class", "linear"))))
    return q_class, feature_config


def build_context(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    grid_config = _dataclass_from_dict(GridConfig, config.get("grid", {}))
    env_config = _dataclass_from_dict(EnvConfig, config.get("environment", {}))
    algo = config.get("algorithm", {})
    gamma = float(algo.get("gamma", 0.97))
    tau_final = float(algo.get("tau_final", 0.001))
    target_tol = float(algo.get("value_iteration_tol", 1e-10))
    max_iter = int(algo.get("value_iteration_max_iter", 20_000))
    decoy_tau = float(config.get("behavior", {}).get("decoy_tau", 0.02))

    mdp = build_grid_mdp(grid_config, env_config)
    target_vi = soft_value_iteration(
        mdp.transition,
        mdp.reward,
        gamma,
        tau_final,
        tol=target_tol,
        max_iter=max_iter,
    )
    decoy_vi = soft_value_iteration(
        mdp.transition,
        mdp.decoy_reward,
        gamma,
        decoy_tau,
        tol=max(target_tol, 1e-9),
        max_iter=max_iter,
    )
    pi_star = softmax_policy(target_vi.q, tau_final)
    pi_decoy = softmax_policy(decoy_vi.q, decoy_tau)
    target_state, target_resid, target_conv = stationary_state_distribution(mdp.transition, pi_star)
    target_sa = state_action_distribution(target_state, pi_star)
    rho0 = np.ones(mdp.n_states, dtype=np.float64) / mdp.n_states
    reference_value = evaluate_soft_policy_value(mdp.transition, mdp.reward, pi_star, gamma, tau_final, rho0)

    regimes: dict[str, dict[str, Any]] = {}
    for name, mixture in config.get("regimes", {}).items():
        policy = _behavior_policy(pi_star, pi_decoy, mixture)
        state_dist, resid, converged = stationary_state_distribution(mdp.transition, policy)
        sa_dist = state_action_distribution(state_dist, policy)
        regimes[name] = {
            "mixture": mixture,
            "policy": policy,
            "state_dist": state_dist,
            "sa_dist": sa_dist,
            "stationary_residual": resid,
            "stationary_converged": converged,
        }

    ratio_features = RatioFeatureMap.from_grid(
        mdp.states,
        mdp.actions,
        n_state_centers=int(config.get("weights", {}).get("n_state_centers", 16)),
        bandwidth_scale=float(config.get("weights", {}).get("bandwidth_scale", 1.0)),
        standardize_features=bool(config.get("weights", {}).get("standardize_features", False)),
    )
    q_class, q_features_config = _q_feature_config(config)
    q_feature_map = QFeatureMap.from_grid(
        q_class,
        mdp.states,
        mdp.actions,
        n_state_centers=int(q_features_config.get("n_state_centers", config.get("weights", {}).get("n_state_centers", 36))),
        bandwidth_scale=float(q_features_config.get("bandwidth_scale", config.get("weights", {}).get("bandwidth_scale", 0.65))),
        standardize_features=bool(q_features_config.get("standardize_features", False)),
    )

    ref_path = output_dir / "reference_context.npz"
    save_payload: dict[str, Any] = {
        "states": mdp.states,
        "actions": mdp.actions,
        "q_star": target_vi.q,
        "pi_star": pi_star,
        "target_state_dist": target_state,
        "target_sa_dist": target_sa,
        "pi_decoy": pi_decoy,
    }
    for regime_name, info in regimes.items():
        save_payload[f"{regime_name}_policy"] = info["policy"]
        save_payload[f"{regime_name}_state_dist"] = info["state_dist"]
        save_payload[f"{regime_name}_sa_dist"] = info["sa_dist"]
        save_payload[f"{regime_name}_oracle_ratio"] = target_sa / np.maximum(info["sa_dist"], 1e-12)
    np.savez_compressed(ref_path, **save_payload)

    metadata = {
        "target_value_iteration_iters": target_vi.n_iters,
        "target_value_iteration_delta": target_vi.sup_delta,
        "target_value_iteration_converged": target_vi.converged,
        "target_stationary_residual": target_resid,
        "target_stationary_converged": target_conv,
        "reference_value": reference_value,
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "ratio_feature_dim": ratio_features.dimension,
        "q_class": q_feature_map.kind,
        "q_feature_dim": q_feature_map.dimension,
    }
    (output_dir / "reference_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "mdp": mdp,
        "q_star": target_vi.q,
        "pi_star": pi_star,
        "pi_decoy": pi_decoy,
        "target_state": target_state,
        "target_sa": target_sa,
        "rho0": rho0,
        "reference_value": reference_value,
        "regimes": regimes,
        "ratio_features": ratio_features,
        "q_feature_map": q_feature_map,
        "metadata": metadata,
    }


def _make_fqi_config(config: dict[str, Any]) -> FQIConfig:
    algo = config.get("algorithm", {})
    return FQIConfig(
        gamma=float(algo.get("gamma", 0.97)),
        tau_final=float(algo.get("tau_final", 0.001)),
        n_iters=int(algo.get("n_iters", 60)),
        ridge=float(algo.get("ridge", 1e-4)),
        metrics_stride=int(algo.get("metrics_stride", 5)),
        anneal_taus=tuple(float(x) for x in algo.get("anneal_taus", [0.2, 0.05, 0.01, 0.003, 0.001])),
    )


def _method_specs(methods: list[str], gamma_weights: list[float]) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for method in methods:
        if method == "estimated":
            specs.extend((f"estimated_g{_safe_name(float(gamma_weight))}", float(gamma_weight)) for gamma_weight in gamma_weights)
        elif method.startswith("estimated_g"):
            specs.append((method, _parse_safe_float(method.removeprefix("estimated_g"))))
        else:
            specs.append((method, float("nan")))
    return specs


def _expand_seed_spec(spec: Any) -> list[int]:
    if isinstance(spec, dict):
        start = int(spec.get("start", 0))
        stop = int(spec["stop"])
        step = int(spec.get("step", 1))
        return list(range(start, stop, step))
    return [int(seed) for seed in list(spec)]


def _run_key(
    *,
    stage: str,
    learner: str,
    regime: str,
    seed: int,
    schedule: str,
    method: str,
    gamma_weight: float,
) -> str:
    gamma_part = "nan" if not np.isfinite(gamma_weight) else _safe_name(float(gamma_weight))
    return "|".join([stage, learner, regime, str(int(seed)), schedule, method, gamma_part])


def _atomic_write_raw(rows: list[dict[str, Any]], raw_path: Path) -> None:
    raw = pd.DataFrame(rows)
    tmp_path = raw_path.with_suffix(".csv.tmp")
    raw.to_csv(tmp_path, index=False)
    tmp_path.replace(raw_path)


def _failure_row(base_meta: dict[str, Any], error: BaseException) -> dict[str, Any]:
    row = dict(base_meta)
    row.update(
        {
            "iteration": -1,
            "is_final": 1,
            "failed": 1,
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "traceback": traceback.format_exc(limit=8),
        }
    )
    for key in [
        "stationary_q_rmse",
        "behavior_q_rmse",
        "stationary_advantage_q_rmse",
        "behavior_advantage_q_rmse",
        "stationary_bellman_rmse",
        "behavior_bellman_rmse",
        "stationary_projected_bellman_rmse",
        "behavior_projected_bellman_rmse",
        "cross_behavior_projected_bellman_rmse",
        "cross_stationary_projected_bellman_rmse",
        "stationary_optimal_action_agreement",
        "behavior_optimal_action_agreement",
        "policy_value_error",
        "norm_mismatch_ratio",
        "max_abs_q",
    ]:
        row[key] = float("nan")
    return row


def _prepare_weights(
    *,
    method_label: str,
    gamma_weight: float,
    learner: str,
    batch: Any,
    context: dict[str, Any],
    regime_info: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    mdp = context["mdp"]
    target_sa = context["target_sa"]
    behavior_sa = regime_info["sa_dist"]
    diagnostics: dict[str, Any] = {}
    if learner == "population":
        if method_label == "oracle":
            return None, target_sa, diagnostics
        return None, behavior_sa, diagnostics

    assert batch is not None
    if method_label in {"unweighted", "minimax"}:
        weights = np.ones(batch.states.shape[0], dtype=np.float64)
        diagnostics.update(summarize_weights(weights))
        diagnostics.update(
            {
                "weight_solver": "unit" if method_label == "unweighted" else "minimax_unweighted_moments",
                "gamma_weight": float("nan"),
                "normalization_error": 0.0,
                "moment_violation_l2": float("nan"),
            }
        )
        return weights, None, diagnostics

    oracle_weights = oracle_sample_weights(batch, target_sa, behavior_sa)
    if method_label == "oracle":
        weight_config = config.get("weights", {})
        weights = oracle_weights
        solver = "oracle_stationary_grid"
        if bool(weight_config.get("stabilize_oracle", False)):
            weights, stabilize_meta = stabilize_weights(
                oracle_weights,
                min_weight=float(weight_config.get("min_weight", 1e-8)),
                max_weight=float(weight_config.get("max_weight", 25.0)),
                clip_quantile=float(weight_config.get("clip_quantile", 0.99)),
                target_ess_fraction=float(weight_config.get("target_ess_fraction", 0.25)),
            )
            diagnostics.update({f"oracle_stabilized_{key}": value for key, value in stabilize_meta.items()})
            solver = "oracle_stationary_grid_stabilized"
        diagnostics.update(summarize_weights(weights))
        diagnostics.update(ratio_quality(oracle_weights, weights))
        diagnostics.update(
            {
                "weight_solver": solver,
                "gamma_weight": float("nan"),
                "normalization_error": 0.0,
                "moment_violation_l2": 0.0,
            }
        )
        return weights, None, diagnostics

    weight_config = config.get("weights", {})
    ratio_features = context["ratio_features"]
    if str(weight_config.get("center_source", "grid")) == "behavior_samples":
        ratio_features = RatioFeatureMap.from_behavior_samples(
            mdp.states[batch.states],
            batch.actions,
            mdp.actions,
            n_centers=int(weight_config.get("n_rbf_centers", 64)),
            bandwidth_scale=float(weight_config.get("bandwidth_scale", 1.0)),
            standardize_features=bool(weight_config.get("standardize_features", False)),
        )
    estimate = estimate_moment_weights(
        batch,
        states_grid=mdp.states,
        transition=mdp.transition,
        target_policy=context["pi_star"],
        behavior_state_dist=regime_info["state_dist"],
        ratio_features=ratio_features,
        gamma_weight=float(gamma_weight),
        ridge_primal=float(weight_config.get("ridge_primal", 1e-4)),
        ridge_dual=float(weight_config.get("ridge_dual", 1e-4)),
        normalization_penalty=float(weight_config.get("normalization_penalty", 10.0)),
        cv_ridge=bool(weight_config.get("cv_ridge", False)),
        cv_ridge_grid=tuple(float(x) for x in weight_config.get("cv_ridge_grid", [])) or None,
        cv_folds=int(weight_config.get("cv_folds", 3)),
        cv_seed=int(weight_config.get("cv_seed", 20260501)),
        cv_score_ridge_dual=(
            float(weight_config["cv_score_ridge_dual"]) if "cv_score_ridge_dual" in weight_config else None
        ),
        cv_selection_rule=str(weight_config.get("cv_selection_rule", "min")),
        min_weight=float(weight_config.get("min_weight", 1e-8)),
        max_weight=float(weight_config.get("max_weight", 25.0)),
        clip_quantile=float(weight_config.get("clip_quantile", 0.99)),
        target_ess_fraction=float(weight_config.get("target_ess_fraction", 0.25)),
    )
    diagnostics.update(estimate.diagnostics)
    diagnostics.update(ratio_quality(oracle_weights, estimate.weights))
    return estimate.weights, None, diagnostics


def run_experiment(config_path: str | Path, *, resume: bool = False, overwrite: bool = False) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    root = Path(__file__).resolve().parents[1]
    output_dir = root / str(config.get("output_dir", "results/debug"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_results.csv"
    if raw_path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{raw_path} already exists; pass --resume to continue or --overwrite to replace it.")
    shutil.copy2(config_path, output_dir / "config.yaml")
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    context = build_context(config, output_dir)
    mdp: GridMDP = context["mdp"]
    fqi_config = _make_fqi_config(config)
    rows: list[dict[str, Any]] = []
    completed_run_keys: set[str] = set()
    if resume and raw_path.exists():
        existing = pd.read_csv(raw_path)
        if "run_key" in existing.columns:
            final = existing[existing.get("is_final", 0) == 1]
            completed_run_keys = set(final["run_key"].dropna().astype(str))
        rows = existing.to_dict("records")
    base_seed = int(config.get("base_seed", 20260501))
    neural_config = config.get("neural", {})
    gamma_weights = [float(x) for x in config.get("weights", {}).get("gamma_weights", [0.0, 0.5, 0.95, 1.0])]
    checkpoint_every = max(int(config.get("checkpoint_every", 25)), 1)
    completed_since_checkpoint = 0

    for stage in config.get("stages", []):
        stage_data_config = _merged_section(config, stage, "data")
        n_samples = int(stage_data_config.get("n_samples", 4_000))
        stage_weight_config = _merged_section(config, stage, "weights")
        stage_config = dict(config)
        stage_config["data"] = stage_data_config
        stage_config["weights"] = stage_weight_config
        stage_minimax_config = _merged_section(config, stage, "minimax")
        stage_name = str(stage["name"])
        learners = list(stage.get("learners", ["linear"]))
        regimes = list(stage.get("regimes", config.get("regimes", {}).keys()))
        schedules = list(stage.get("schedules", ["direct"]))
        methods = list(stage.get("methods", ["unweighted", "oracle"]))
        seeds = _expand_seed_spec(stage.get("seeds", [0]))
        stage_gamma_weights = [float(x) for x in stage.get("gamma_weights", stage_weight_config.get("gamma_weights", gamma_weights))]
        method_specs = _method_specs(methods, stage_gamma_weights)
        for learner in learners:
            for regime_name in regimes:
                regime_info = context["regimes"][regime_name]
                for seed in seeds:
                    batch = None
                    if learner != "population":
                        batch = sample_transition_batch(
                            mdp,
                            regime_info["state_dist"],
                            regime_info["policy"],
                            n_samples=n_samples,
                            seed=base_seed + int(seed),
                        )
                    for schedule in schedules:
                        for method_label, gamma_weight in method_specs:
                            if learner == "population" and (method_label.startswith("estimated") or method_label == "minimax"):
                                continue
                            run_key = _run_key(
                                stage=stage_name,
                                learner=learner,
                                regime=regime_name,
                                seed=int(seed),
                                schedule=schedule,
                                method=method_label,
                                gamma_weight=gamma_weight,
                            )
                            if run_key in completed_run_keys:
                                continue
                            base_meta: dict[str, Any] = {
                                "run_key": run_key,
                                "experiment": str(config.get("name", output_dir.name)),
                                "stage": stage_name,
                                "learner": learner,
                                "regime": regime_name,
                                "schedule": schedule,
                                "method": method_label,
                                "seed": int(seed),
                                "n_samples": 0 if learner == "population" else n_samples,
                                "gamma": fqi_config.gamma,
                                "tau_final": fqi_config.tau_final,
                                "gamma_weight": gamma_weight,
                                "q_class": context["q_feature_map"].kind,
                                "q_feature_dim": context["q_feature_map"].dimension,
                                "failed": 0,
                                "error_type": "",
                                "error_message": "",
                                "stationary_residual_behavior": regime_info["stationary_residual"],
                            }
                            try:
                                sample_weights, projection_sa, weight_diag = _prepare_weights(
                                    method_label=method_label,
                                    gamma_weight=gamma_weight,
                                    learner=learner,
                                    batch=batch,
                                    context=context,
                                    regime_info=regime_info,
                                    config=stage_config,
                                )
                                base_meta.update(weight_diag)
                                if learner == "linear" and batch is not None:
                                    phi = context["q_feature_map"].transform(mdp.states[batch.states], batch.actions)
                                    gram_weights = sample_weights if sample_weights is not None else np.ones(batch.states.shape[0])
                                    base_meta["weighted_gram_condition"] = weighted_design_condition_number(
                                        phi, gram_weights, fqi_config.ridge
                                    )
                                elif learner == "population":
                                    state_ids, action_ids = np.meshgrid(
                                        np.arange(mdp.n_states), np.arange(mdp.n_actions), indexing="ij"
                                    )
                                    phi = context["q_feature_map"].transform(
                                        mdp.states[state_ids.reshape(-1)],
                                        action_ids.reshape(-1),
                                    )
                                    base_meta["weighted_gram_condition"] = weighted_design_condition_number(
                                        phi, projection_sa.reshape(-1), fqi_config.ridge
                                    )
                                if learner == "population":
                                    assert projection_sa is not None
                                    rows.extend(
                                        run_population_linear_fqi(
                                            mdp=mdp,
                                            q_star=context["q_star"],
                                            projection_sa_dist=projection_sa,
                                            target_sa_dist=context["target_sa"],
                                            behavior_sa_dist=regime_info["sa_dist"],
                                            schedule=schedule,
                                            fqi_config=fqi_config,
                                            reference_value=context["reference_value"],
                                            rho0=context["rho0"],
                                            base_meta=base_meta,
                                            q_feature_map=context["q_feature_map"],
                                        )
                                    )
                                elif learner == "linear":
                                    assert batch is not None and sample_weights is not None
                                    if method_label == "minimax":
                                        rows.extend(
                                            run_minimax_soft_q(
                                                mdp=mdp,
                                                batch=batch,
                                                q_star=context["q_star"],
                                                target_sa_dist=context["target_sa"],
                                                behavior_sa_dist=regime_info["sa_dist"],
                                                schedule=schedule,
                                                fqi_config=fqi_config,
                                                minimax_config=stage_minimax_config,
                                                ratio_features=context["ratio_features"],
                                                q_feature_map=context["q_feature_map"],
                                                reference_value=context["reference_value"],
                                                rho0=context["rho0"],
                                                seed=base_seed + int(seed),
                                                base_meta=base_meta,
                                            )
                                        )
                                    else:
                                        rows.extend(
                                            run_sample_linear_fqi(
                                                mdp=mdp,
                                                batch=batch,
                                                sample_weights=sample_weights,
                                                q_star=context["q_star"],
                                                target_sa_dist=context["target_sa"],
                                                behavior_sa_dist=regime_info["sa_dist"],
                                                schedule=schedule,
                                                fqi_config=fqi_config,
                                                reference_value=context["reference_value"],
                                                rho0=context["rho0"],
                                                base_meta=base_meta,
                                                q_feature_map=context["q_feature_map"],
                                            )
                                        )
                                elif learner == "neural":
                                    assert batch is not None and sample_weights is not None
                                    rows.extend(
                                        run_neural_fqi(
                                            mdp=mdp,
                                            batch=batch,
                                            sample_weights=sample_weights,
                                            q_star=context["q_star"],
                                            target_sa_dist=context["target_sa"],
                                            behavior_sa_dist=regime_info["sa_dist"],
                                            schedule=schedule,
                                            fqi_config=fqi_config,
                                            neural_config=neural_config,
                                            reference_value=context["reference_value"],
                                            rho0=context["rho0"],
                                            seed=base_seed + int(seed),
                                            base_meta=base_meta,
                                        )
                                    )
                                else:
                                    raise ValueError(f"Unknown learner '{learner}'.")
                            except Exception as exc:
                                rows.append(_failure_row(base_meta, exc))
                            completed_run_keys.add(run_key)
                            completed_since_checkpoint += 1
                            if completed_since_checkpoint >= checkpoint_every:
                                _atomic_write_raw(rows, raw_path)
                                completed_since_checkpoint = 0

    _atomic_write_raw(rows, raw_path)
    return raw_path
