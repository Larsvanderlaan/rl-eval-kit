from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.occupancy_ratio.fori_eval.finite_mdp import FiniteDataset
from experiments.occupancy_ratio.fori_eval.metrics import evaluate_grid_weights


Array = np.ndarray


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages" / "occupancy-ratio").exists():
            return parent
    return Path(__file__).resolve().parents[5]


RLEVALKIT_ROOT = _repo_root()
RLEVALKIT_OCCUPANCY_PATH = RLEVALKIT_ROOT / "packages" / "occupancy-ratio"


@dataclass
class EstimatorOutput:
    name: str
    status: str
    weights: Array | None
    raw_weights: Array | None = None
    scope: str = "grid"
    runtime_sec: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    skip_reason: str = ""


def run_oracle(dataset: FiniteDataset) -> EstimatorOutput:
    return EstimatorOutput(
        name="oracle",
        status="ok",
        weights=dataset.truth.omega_star.copy(),
        raw_weights=dataset.truth.omega_star.copy(),
        diagnostics={"estimator_family": "oracle"},
    )


def run_population_fori(
    dataset: FiniteDataset,
    *,
    name: str = "population_fori",
    num_iterations: int = 50,
    initial_ratio: float = 1.0,
    omega0: Array | None = None,
    c_pi: Array | None = None,
    m_override: str | None = None,
    clip_max: float | None = None,
    normalize: bool = False,
    damping: float = 1.0,
) -> EstimatorOutput:
    """Run exact population FORI variants on the finite grid."""

    start = time.perf_counter()
    truth = dataset.truth
    omega = np.full_like(truth.omega_star, float(initial_ratio), dtype=np.float64)
    init = truth.omega0 if omega0 is None else np.asarray(omega0, dtype=np.float64).reshape(-1)
    coverage = truth.c_pi if c_pi is None else np.asarray(c_pi, dtype=np.float64).reshape(-1)
    history: list[dict[str, Any]] = []

    for _ in range(int(num_iterations)):
        if m_override == "constant_one":
            m_omega = np.ones_like(omega)
        else:
            m_omega = truth.backward_conditional_mean(omega)
        raw_update = (1.0 - truth.gamma) * init + truth.gamma * coverage * m_omega
        projected = project_weights(raw_update, truth.nu, clip_max=clip_max, normalize=normalize)
        omega = (1.0 - float(damping)) * omega + float(damping) * projected
        history.append(evaluate_grid_weights(dataset=dataset, weights=omega))

    diagnostics = {"estimator_family": "population", "population_iterations": int(num_iterations)}
    diagnostics.update(history[-1] if history else evaluate_grid_weights(dataset=dataset, weights=omega))
    return EstimatorOutput(
        name=name,
        status="ok",
        weights=omega,
        raw_weights=omega,
        runtime_sec=time.perf_counter() - start,
        diagnostics=diagnostics,
        history=history,
    )


def run_boosted_tree(dataset: FiniteDataset, *, preset: str, profile: str, seed: int) -> EstimatorOutput:
    """Fit RLEvalKit's boosted occupancy-ratio estimator and predict on the full grid."""

    start = time.perf_counter()
    try:
        ensure_rlevalkit_path()
        from occupancy_ratio import (  # noqa: PLC0415
            ActionRatioConfig,
            OccupancyRegressionConfig,
            TransitionRatioConfig,
            fit_discounted_occupancy_ratio,
        )
    except Exception as exc:
        return EstimatorOutput(
            name=f"boosted_tree_{preset}",
            status="skipped",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=f"Could not import RLEvalKit occupancy_ratio: {exc}",
        )

    try:
        if preset == "auto":
            return skipped_auto_output("boosted_tree_auto", start)
        occ_kwargs = boosted_preset_kwargs(preset)
        nuisance_kwargs = boosted_nuisance_kwargs(preset)
        fast = str(profile) == "smoke"
        occupancy = OccupancyRegressionConfig(
            num_iterations=8 if fast else 30,
            trees_per_iteration=1,
            mcmc_samples=8 if fast else 24,
            batch_size=256 if fast else 512,
            seed=int(seed),
            show_progress=False,
            validation_fraction=0.2,
            patience=4 if fast else 8,
            lgb_params={
                "learning_rate": 0.08,
                "num_leaves": 15 if fast else 31,
                "min_data_in_leaf": 2 if fast else 20,
                "verbose": -1,
                "num_threads": 1,
                "seed": int(seed),
            },
            **occ_kwargs,
        )
        action_ratio = ActionRatioConfig(
            num_boost_round=10 if fast else 80,
            early_stopping_rounds=3 if fast else 8,
            validation_fraction=0.2,
            refit_on_all_data=True,
            show_progress=False,
            prediction_max=nuisance_kwargs["prediction_max"],
            moment_calibration=nuisance_kwargs["moment_calibration"],
            crossfit_folds=nuisance_kwargs["crossfit_folds"],
            density_ratio_loss=nuisance_kwargs["density_ratio_loss"],
            logistic_logit_clip=20.0,
            lgb_params={"num_leaves": 15, "min_data_in_leaf": 2, "verbose": -1, "num_threads": 1, "seed": int(seed)},
        )
        transition_ratio = TransitionRatioConfig(
            num_boost_round=12 if fast else 120,
            permutation_samples=3 if fast else 12,
            early_stopping_rounds=3 if fast else 8,
            validation_fraction=0.2,
            refit_on_all_data=True,
            show_progress=False,
            prediction_max=nuisance_kwargs["prediction_max"],
            moment_calibration=nuisance_kwargs["moment_calibration"],
            crossfit_folds=nuisance_kwargs["crossfit_folds"],
            density_ratio_loss=nuisance_kwargs["density_ratio_loss"],
            logistic_logit_clip=20.0,
            lgb_params={"num_leaves": 15, "min_data_in_leaf": 2, "verbose": -1, "num_threads": 1, "seed": int(seed)},
        )
        model = fit_discounted_occupancy_ratio(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=dataset.truth.gamma,
            occupancy=occupancy,
            action_ratio=action_ratio,
            transition_ratio=transition_ratio,
        )
        raw = model.predict_state_action_ratio(dataset.all_states, dataset.all_actions, clip=False)
        weights = model.predict_state_action_ratio(dataset.all_states, dataset.all_actions, clip=True)
        diagnostics = {
            "estimator_family": "boosted",
            "preset": preset,
            "history_length": len(model.history),
            **model.diagnostics,
            **evaluate_grid_weights(dataset=dataset, weights=weights, raw_weights=raw, history=model.history),
        }
        return EstimatorOutput(
            name=f"boosted_tree_{preset}",
            status="ok",
            weights=weights,
            raw_weights=raw,
            runtime_sec=time.perf_counter() - start,
            diagnostics=diagnostics,
            history=model.history,
        )
    except Exception as exc:
        return EstimatorOutput(
            name=f"boosted_tree_{preset}",
            status="error",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=str(exc),
        )


def run_neural_network(dataset: FiniteDataset, *, preset: str, profile: str, seed: int) -> EstimatorOutput:
    """Fit RLEvalKit's neural occupancy-ratio estimator and predict on the full grid."""

    start = time.perf_counter()
    try:
        ensure_rlevalkit_path()
        from occupancy_ratio import (  # noqa: PLC0415
            NeuralActionRatioConfig,
            NeuralOccupancyRegressionConfig,
            NeuralTransitionRatioConfig,
            fit_discounted_occupancy_ratio_neural,
        )
    except Exception as exc:
        return EstimatorOutput(
            name=f"neural_network_{preset}",
            status="skipped",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=f"Could not import RLEvalKit neural occupancy_ratio: {exc}",
        )

    try:
        if preset == "auto":
            return skipped_auto_output("neural_network_auto", start)
        fast = str(profile) == "smoke"
        occ_kwargs = neural_preset_kwargs(preset)
        nuisance_kwargs = neural_nuisance_kwargs(preset)
        hidden_dims = (8,) if fast else (64, 64)
        occupancy = NeuralOccupancyRegressionConfig(
            hidden_dims=hidden_dims,
            num_iterations=2 if fast else 20,
            gradient_steps_per_iteration=1 if fast else 4,
            mcmc_samples=2 if fast else 24,
            batch_size=32 if fast else 512,
            learning_rate=8e-4,
            weight_decay=1e-4,
            seed=int(seed),
            show_progress=False,
            validation_fraction=0.2,
            patience=2 if fast else 8,
            validation_warmup_iterations=1,
            device="cpu",
            **occ_kwargs,
        )
        action_ratio = NeuralActionRatioConfig(
            hidden_dims=hidden_dims,
            max_steps=5 if fast else 300,
            batch_size=32 if fast else 512,
            learning_rate=1e-3,
            patience=2 if fast else 12,
            prediction_max=nuisance_kwargs["prediction_max"],
            moment_calibration=nuisance_kwargs["moment_calibration"],
            crossfit_folds=nuisance_kwargs["crossfit_folds"],
            density_ratio_loss=nuisance_kwargs["density_ratio_loss"],
            logistic_logit_clip=20.0,
            seed=int(seed + 7_001),
            device="cpu",
        )
        transition_ratio = NeuralTransitionRatioConfig(
            hidden_dims=hidden_dims,
            max_steps=5 if fast else 400,
            permutation_samples=2 if fast else 4,
            batch_size=32 if fast else 512,
            learning_rate=1e-3,
            patience=2 if fast else 12,
            prediction_max=nuisance_kwargs["prediction_max"],
            moment_calibration=nuisance_kwargs["moment_calibration"],
            crossfit_folds=nuisance_kwargs["crossfit_folds"],
            density_ratio_loss=nuisance_kwargs["density_ratio_loss"],
            logistic_logit_clip=20.0,
            seed=int(seed + 8_001),
            device="cpu",
        )
        model = fit_discounted_occupancy_ratio_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            gamma=dataset.truth.gamma,
            occupancy=occupancy,
            action_ratio=action_ratio,
            transition_ratio=transition_ratio,
        )
        raw = model.predict_state_action_ratio(dataset.all_states, dataset.all_actions, clip=False)
        weights = model.predict_state_action_ratio(dataset.all_states, dataset.all_actions, clip=True)
        diagnostics = {
            "estimator_family": "neural",
            "preset": preset,
            "history_length": len(model.history),
            **model.diagnostics,
            **evaluate_grid_weights(dataset=dataset, weights=weights, raw_weights=raw, history=model.history),
        }
        return EstimatorOutput(
            name=f"neural_network_{preset}",
            status="ok",
            weights=weights,
            raw_weights=raw,
            runtime_sec=time.perf_counter() - start,
            diagnostics=diagnostics,
            history=model.history,
        )
    except Exception as exc:
        return EstimatorOutput(
            name=f"neural_network_{preset}",
            status="error",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=str(exc),
        )


def run_google_dualdice_sample(dataset: FiniteDataset, *, external_repo_path: str, seed: int) -> EstimatorOutput:
    """Run RLEvalKit's Google DualDICE adapter when available.

    The official adapter exposes weights only on logged rows, so these rows are
    marked as sample-scope diagnostics rather than exact full-grid diagnostics.
    """

    start = time.perf_counter()
    try:
        ensure_rlevalkit_path()
        from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig  # noqa: PLC0415
        from occupancy_ratio_benchmark.data import BenchmarkDataset  # noqa: PLC0415
        from occupancy_ratio_benchmark.estimators import estimate_google_dualdice  # noqa: PLC0415
        from occupancy_ratio_benchmark.external_baselines import preflight_google_dualdice  # noqa: PLC0415
    except Exception as exc:
        return EstimatorOutput(
            name="google_dualdice_neural",
            status="skipped",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=f"Could not import RLEvalKit DualDICE bridge: {exc}",
        )

    preflight = preflight_google_dualdice(external_repo_path)
    if not preflight.available:
        return EstimatorOutput(
            name="google_dualdice_neural",
            status="skipped",
            weights=None,
            runtime_sec=time.perf_counter() - start,
            skip_reason=preflight.reason,
        )

    bench = BenchmarkDataset(
        setting=dataset.setting,
        states=dataset.states,
        actions=dataset.actions,
        next_states=dataset.next_states,
        target_actions=dataset.target_actions,
        next_target_actions=dataset.next_target_actions,
        rewards=dataset.rewards,
        true_ratio=dataset.true_ratio_sample,
        initial_states=dataset.states[: min(512, dataset.sample_size)],
        initial_actions=dataset.target_actions[: min(512, dataset.sample_size)],
        initial_weights=np.ones(min(512, dataset.sample_size), dtype=np.float64),
        masks=np.ones(dataset.sample_size, dtype=np.float64),
        gamma=dataset.truth.gamma,
        seed=int(seed),
        sample_size=dataset.sample_size,
    )
    config = OccupancyRatioBenchmarkConfig(
        stage="smoke",
        include_google_dual_dice=True,
        external_repo_path=Path(external_repo_path),
        google_num_updates=50,
        google_batch_size=128,
    )
    result = estimate_google_dualdice(bench, config, preflight)
    return EstimatorOutput(
        name=result.estimator,
        status=result.status,
        weights=result.weights,
        raw_weights=result.raw_weights,
        scope="sample",
        runtime_sec=time.perf_counter() - start,
        diagnostics=dict(result.diagnostics),
        skip_reason=result.skip_reason,
    )


def project_weights(weights: Array, nu: Array, *, clip_max: float | None, normalize: bool) -> Array:
    out = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if clip_max is not None:
        out = np.minimum(out, float(clip_max))
    if normalize:
        mass = float(np.sum(np.asarray(nu, dtype=np.float64).reshape(-1) * out))
        if mass > 1e-12:
            out = out / mass
    return out


def skipped_auto_output(name: str, start: float) -> EstimatorOutput:
    return EstimatorOutput(
        name=name,
        status="skipped",
        weights=None,
        runtime_sec=time.perf_counter() - start,
        skip_reason=(
            "Auto selection is available in occupancy_ratio_benchmark; "
            "the exact full-grid harness runs explicit variants only."
        ),
    )


def boosted_preset_kwargs(preset: str) -> dict[str, Any]:
    if preset == "squared":
        return dict(
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
        )
    if preset in {"huber", "logistic_nuisance"}:
        return dict(
            loss="huber",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
        )
    if preset in {"stable", "transition_norm", "calibrated", "stable_logistic_nuisance"}:
        return dict(
            loss="huber",
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            normalize_transition_cache=bool(preset == "transition_norm"),
        )
    raise ValueError(f"Unknown boosted preset '{preset}'.")


def boosted_nuisance_kwargs(preset: str) -> dict[str, Any]:
    stable = preset not in {"squared", "huber", "logistic_nuisance"}
    return {
        "prediction_max": None if preset in {"squared", "huber", "logistic_nuisance"} else 50.0,
        "moment_calibration": "scalar" if preset == "calibrated" else "none",
        "crossfit_folds": 1,
        "density_ratio_loss": "logistic" if preset in {"logistic_nuisance", "stable_logistic_nuisance"} else "lsif",
        "stable": stable,
    }


def neural_preset_kwargs(preset: str) -> dict[str, Any]:
    if preset == "squared":
        return dict(
            loss="squared",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
        )
    if preset in {"huber", "logistic_nuisance"}:
        return dict(
            loss="huber",
            fixed_point_damping=1.0,
            normalize_occupancy=False,
            occupancy_ratio_max=None,
            clip_pseudo_outcomes=False,
        )
    if preset in {"stable", "transition_norm", "calibrated", "stable_logistic_nuisance"}:
        return dict(
            loss="huber",
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            normalize_transition_cache=bool(preset == "transition_norm"),
        )
    raise ValueError(f"Unknown neural preset '{preset}'.")


def neural_nuisance_kwargs(preset: str) -> dict[str, Any]:
    return {
        "prediction_max": None if preset in {"squared", "huber", "logistic_nuisance"} else 50.0,
        "moment_calibration": "scalar" if preset == "calibrated" else "none",
        "crossfit_folds": 1,
        "density_ratio_loss": "logistic" if preset in {"logistic_nuisance", "stable_logistic_nuisance"} else "lsif",
    }


def ensure_rlevalkit_path() -> None:
    path = str(RLEVALKIT_OCCUPANCY_PATH)
    if path not in sys.path:
        sys.path.insert(0, path)
