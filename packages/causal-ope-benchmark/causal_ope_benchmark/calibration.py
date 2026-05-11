from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from importlib import metadata
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal_ope_benchmark.adapters import to_fqe_dataset, to_occupancy_ratio_dataset
from causal_ope_benchmark.config import CausalOPEBenchmarkConfig, DomainScenario, FamilyName, scenarios_for_profile, sensitivity_scenarios_for_profile
from causal_ope_benchmark.constants import CALIBRATION_OUTPUT_FILES, CALIBRATION_SCHEMA_VERSION, DEFAULT_FAMILIES, PACKAGE_VERSION
from causal_ope_benchmark.io import write_csv, write_json
from causal_ope_benchmark.policies import POLICY_NAMES_BY_FAMILY
from causal_ope_benchmark.runner import make_problem
from causal_ope_benchmark.types import BenchmarkProblem


@dataclass(frozen=True)
class CalibrationStudyConfig:
    """Configuration for neural FQE and occupancy-ratio difficulty calibration."""

    preset: str = "core-lite"
    output_root: Path = Path("outputs/causal_ope_benchmark")
    families: Sequence[FamilyName] = DEFAULT_FAMILIES
    include_epicare: bool = False
    include_stress: bool = False
    seeds: Sequence[int] = (0, 1, 2)
    sample_sizes: Sequence[int] = (1000,)
    gammas: Sequence[float] = (0.95,)
    target_policies: Sequence[str] | None = ("conservative", "moderate", "aggressive", "budget_constrained", "safety_constrained")
    estimators: Sequence[str] = ("neural_fqe", "neural_occupancy")
    tuning_tracks: Sequence[str] = ("proxy", "oracle")
    trajectory_horizon: int = 24
    streamlift_observed_horizon: int = 3
    streamlift_long_horizon: int = 36
    mc_truth_rollouts: int = 96
    cv_folds: int = 3
    fqe_budget: str = "balanced"
    fqe_hidden_dims: Sequence[int] = (64, 64)
    fqe_num_iterations: int = 48
    fqe_gradient_steps_per_iteration: int = 15
    fqe_batch_size: int = 256
    fqe_max_candidates: int = 10
    fqe_promotion_candidates: int = 4
    occupancy_budget: str = "balanced"
    occupancy_hidden_dims: Sequence[int] = (64, 64)
    occupancy_num_iterations: int = 30
    occupancy_gradient_steps_per_iteration: int = 4
    occupancy_nuisance_max_steps: int = 400
    occupancy_mcmc_samples: int = 24
    occupancy_batch_size: int = 256
    occupancy_max_candidates: int = 8
    occupancy_promotion_candidates: int = 3
    occupancy_score_method: str = "bellman_gmm"
    finite_rate_threshold: float = 0.95
    normalized_error_threshold: float = 0.25
    fail_fast: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.preset not in {"smoke", "core-lite", "full"}:
            raise ValueError("preset must be 'smoke', 'core-lite', or 'full'.")
        families = tuple(str(family) for family in self.families)
        if self.include_epicare and "epicare" not in families:
            families = (*families, "epicare")
            object.__setattr__(self, "families", families)
        for family in self.families:
            if family not in {"streamlift", "streamretain", "clinic_dtr", "epicare"}:
                raise ValueError(f"Unsupported family '{family}'.")
        for track in self.tuning_tracks:
            if track not in {"proxy", "oracle"}:
                raise ValueError("tuning_tracks entries must be 'proxy' or 'oracle'.")
        for estimator in self.estimators:
            if estimator not in {"neural_fqe", "neural_occupancy"}:
                raise ValueError("estimators entries must be 'neural_fqe' or 'neural_occupancy'.")
        for budget in (self.fqe_budget, self.occupancy_budget):
            if budget not in {"fast", "balanced"}:
                raise ValueError("FQE and occupancy budgets must be 'fast' or 'balanced'.")
        if self.occupancy_score_method not in {"legacy_rank", "bellman_gmm"}:
            raise ValueError("occupancy_score_method must be 'legacy_rank' or 'bellman_gmm'.")
        if int(self.cv_folds) < 2:
            raise ValueError("cv_folds must be at least 2.")
        for value_name in ("trajectory_horizon", "streamlift_observed_horizon", "mc_truth_rollouts"):
            if int(getattr(self, value_name)) <= 0:
                raise ValueError(f"{value_name} must be positive.")

    @classmethod
    def for_preset(
        cls,
        preset: str,
        *,
        output_root: str | Path = Path("outputs/causal_ope_benchmark"),
    ) -> "CalibrationStudyConfig":
        """Return a named calibration preset."""

        if preset == "smoke":
            return cls(
                preset="smoke",
                output_root=Path(output_root),
                seeds=(0,),
                sample_sizes=(60,),
                gammas=(0.90,),
                target_policies=("moderate",),
                trajectory_horizon=8,
                streamlift_observed_horizon=2,
                mc_truth_rollouts=8,
                cv_folds=2,
                fqe_budget="fast",
                fqe_hidden_dims=(16,),
                fqe_num_iterations=2,
                fqe_gradient_steps_per_iteration=1,
                fqe_batch_size=32,
                fqe_max_candidates=2,
                fqe_promotion_candidates=1,
                occupancy_budget="fast",
                occupancy_hidden_dims=(16,),
                occupancy_num_iterations=2,
                occupancy_gradient_steps_per_iteration=1,
                occupancy_nuisance_max_steps=2,
                occupancy_mcmc_samples=2,
                occupancy_batch_size=32,
                occupancy_max_candidates=1,
                occupancy_promotion_candidates=1,
                occupancy_score_method="legacy_rank",
            )
        if preset == "core-lite":
            return cls(preset="core-lite", output_root=Path(output_root))
        if preset == "full":
            return cls(
                preset="full",
                output_root=Path(output_root),
                include_stress=True,
                seeds=tuple(range(5)),
                sample_sizes=(1000, 3000),
                target_policies=("conservative", "moderate", "aggressive", "budget_constrained", "safety_constrained"),
                mc_truth_rollouts=192,
                fqe_hidden_dims=(128, 128),
                fqe_num_iterations=80,
                fqe_gradient_steps_per_iteration=20,
                fqe_batch_size=512,
                fqe_max_candidates=10,
                occupancy_hidden_dims=(128, 128),
                occupancy_num_iterations=60,
                occupancy_gradient_steps_per_iteration=6,
                occupancy_nuisance_max_steps=600,
                occupancy_mcmc_samples=48,
                occupancy_batch_size=512,
                occupancy_max_candidates=10,
            )
        raise ValueError("preset must be 'smoke', 'core-lite', or 'full'.")

    def output_dir(self) -> Path:
        return self.output_root / "calibration" / self.preset


@dataclass
class CalibrationRunResult:
    """Output bundle for a completed calibration study."""

    output_dir: Path
    results_path: Path
    summary_path: Path
    candidates_path: Path
    manifest_path: Path
    readout_path: Path
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]


@dataclass
class _CalibrationEstimate:
    estimator: str
    tuning_track: str
    status: str
    estimate: float | None = None
    runtime_sec: float = 0.0
    skip_reason: str = ""
    diagnostics: dict[str, Any] | None = None
    diagnostic_only: bool = False


def run_calibration_study(config: CalibrationStudyConfig | None = None, **overrides: Any) -> CalibrationRunResult:
    """Run neural FQE and occupancy-ratio calibration screens."""

    cfg = config or CalibrationStudyConfig.for_preset(str(overrides.pop("preset", "core-lite")))
    if overrides:
        cfg = CalibrationStudyConfig(**{**cfg.__dict__, **overrides})
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for family in cfg.families:
        for scenario in _scenarios_for_calibration(cfg, str(family)):
            for sample_size in cfg.sample_sizes:
                for gamma in cfg.gammas:
                    for seed in cfg.seeds:
                        for target_policy in _target_policies_for_family(cfg, str(family)):
                            try:
                                problem = make_problem(
                                    family=str(family),
                                    scenario=scenario,
                                    sample_size=int(sample_size),
                                    gamma=float(gamma),
                                    seed=int(seed),
                                    observed_horizon=int(cfg.streamlift_observed_horizon),
                                    target_policy=str(target_policy),
                                    config=_problem_config(cfg),
                                )
                            except Exception as exc:
                                if cfg.fail_fast:
                                    raise
                                failures.append(traceback.format_exc())
                                rows.append(_problem_error_row(cfg, str(family), scenario, sample_size, gamma, seed, str(target_policy), exc))
                                continue
                            for estimator in cfg.estimators:
                                try:
                                    estimates, candidates = _run_estimator_tracks(str(estimator), problem, cfg, scenario)
                                except Exception as exc:
                                    if cfg.fail_fast:
                                        raise
                                    failures.append(traceback.format_exc())
                                    estimates = [
                                        _CalibrationEstimate(
                                            estimator=str(estimator),
                                            tuning_track=track,
                                            status="error",
                                            skip_reason=f"{type(exc).__name__}: {exc}",
                                            diagnostic_only=track == "oracle" or problem.dataset.family == "streamlift",
                                        )
                                        for track in cfg.tuning_tracks
                                    ]
                                    candidates = []
                                candidate_rows.extend(_with_problem_context(candidates, cfg, problem, scenario, str(target_policy)))
                                for estimate in estimates:
                                    rows.append(_row_from_estimate(cfg, problem, scenario, str(target_policy), estimate))
    summary_rows = _summarize_calibration_rows(rows, cfg)
    manifest = _manifest(cfg)
    readout = _render_readout(cfg, rows, summary_rows, candidate_rows, failures, manifest)
    results_path = output_dir / CALIBRATION_OUTPUT_FILES["results"]
    summary_path = output_dir / CALIBRATION_OUTPUT_FILES["summary"]
    candidates_path = output_dir / CALIBRATION_OUTPUT_FILES["candidates"]
    manifest_path = output_dir / CALIBRATION_OUTPUT_FILES["manifest"]
    readout_path = output_dir / CALIBRATION_OUTPUT_FILES["readout"]
    write_csv(results_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(candidates_path, candidate_rows)
    write_json(manifest_path, manifest)
    readout_path.write_text(readout, encoding="utf-8")
    return CalibrationRunResult(
        output_dir=output_dir,
        results_path=results_path,
        summary_path=summary_path,
        candidates_path=candidates_path,
        manifest_path=manifest_path,
        readout_path=readout_path,
        rows=rows,
        summary_rows=summary_rows,
        candidate_rows=candidate_rows,
    )


def _run_estimator_tracks(
    estimator: str,
    problem: BenchmarkProblem,
    config: CalibrationStudyConfig,
    scenario: DomainScenario,
) -> tuple[list[_CalibrationEstimate], list[dict[str, Any]]]:
    if estimator == "neural_fqe":
        return _run_neural_fqe_tracks(problem, config, scenario)
    if estimator == "neural_occupancy":
        return _run_neural_occupancy_tracks(problem, config, scenario)
    return (
        [
            _CalibrationEstimate(
                estimator=estimator,
                tuning_track=track,
                status="skipped",
                skip_reason="unknown calibration estimator",
                diagnostic_only=track == "oracle" or problem.dataset.family == "streamlift",
            )
            for track in config.tuning_tracks
        ],
        [],
    )


def _run_neural_fqe_tracks(
    problem: BenchmarkProblem,
    config: CalibrationStudyConfig,
    scenario: DomainScenario,
) -> tuple[list[_CalibrationEstimate], list[dict[str, Any]]]:
    start = time.perf_counter()
    dataset = problem.dataset
    try:
        from fqe import FQESearchSpace, FQETuningConfig, NeuralFQEConfig, fit_fqe_neural, tune_fqe_auto
    except ModuleNotFoundError as exc:
        return (_missing_rows("neural_fqe", config, dataset.family, exc), [])
    fqe_data = to_fqe_dataset(dataset, target_policy_expectation_mode="exact_discrete")
    groups = _groups_for_fqe(dataset.unit_id, fqe_data.source_row_index)
    base = NeuralFQEConfig.stable_defaults(
        hidden_dims=tuple(int(width) for width in config.fqe_hidden_dims),
        learning_rate=5e-4,
        weight_decay=1e-5,
        batch_size=int(config.fqe_batch_size),
        num_iterations=int(config.fqe_num_iterations),
        gradient_steps_per_iteration=int(config.fqe_gradient_steps_per_iteration),
        target_update_tau=0.10,
        validation_fraction=0.20,
        early_stopping=False,
        patience=999,
        seed=int(dataset.seed),
        device="cpu",
        show_progress=False,
    )
    space = FQESearchSpace(neural=base, neural_candidates=_fqe_candidate_grid(config))
    tuning_config = FQETuningConfig(
        families=("neural",),
        cv_folds=min(int(config.cv_folds), _max_group_folds(groups)),
        seed=int(dataset.seed) + 101,
        budget=str(config.fqe_budget),
        max_candidates=int(config.fqe_max_candidates),
        promotion_candidates=int(config.fqe_promotion_candidates),
        refit=True,
        screen_fraction=0.5 if config.fqe_budget == "balanced" else 0.3,
    )
    try:
        tuning = tune_fqe_auto(
            states=fqe_data.states,
            actions=fqe_data.actions,
            next_states=fqe_data.next_states,
            next_actions=fqe_data.next_actions,
            rewards=fqe_data.rewards,
            gamma=fqe_data.gamma,
            terminals=fqe_data.terminals,
            sample_weight=fqe_data.sample_weight,
            initial_states=fqe_data.initial_states,
            initial_actions=fqe_data.initial_actions,
            initial_weights=None,
            groups=groups,
            families=("neural",),
            search_space=space,
            config=tuning_config,
        )
    except ModuleNotFoundError as exc:
        return (_missing_rows("neural_fqe", config, dataset.family, exc), [])
    estimates: list[_CalibrationEstimate] = []
    candidates = _fqe_candidate_rows(tuning, tuning_track="proxy")
    elapsed = float(time.perf_counter() - start)
    if "proxy" in config.tuning_tracks:
        estimate = None if tuning.model is None else _estimate_fqe_model_value(tuning.model, fqe_data)
        estimates.append(
            _CalibrationEstimate(
                estimator="neural_fqe",
                tuning_track="proxy",
                status="ok" if _is_finite(estimate) else "error",
                estimate=estimate,
                runtime_sec=elapsed,
                skip_reason="" if _is_finite(estimate) else "proxy tuning did not produce a finite model estimate",
                diagnostics={
                    "selected_candidate_id": tuning.selected_candidate_id,
                    "selected_family": tuning.selected_family,
                    "selected_by": "proxy",
                    "candidate_count": len(tuning.candidates),
                    "fold_count": len(tuning.folds),
                    "fqe_adapter_row_expansion_factor": fqe_data.row_expansion_factor,
                    "fqe_row_count": int(fqe_data.states.shape[0]),
                },
                diagnostic_only=dataset.family == "streamlift",
            )
        )
    if "oracle" in config.tuning_tracks:
        target = _target_for_problem(problem).value
        oracle_candidate, oracle_cv_value, oracle_error = _select_oracle_fqe_candidate(tuning, target)
        oracle_estimate = None
        oracle_diag: dict[str, Any] = {
            "selected_by": "oracle_truth",
            "candidate_count": len(tuning.candidates),
            "fold_count": len(tuning.folds),
            "oracle_cv_value": oracle_cv_value,
            "oracle_cv_abs_error": oracle_error,
            "fqe_adapter_row_expansion_factor": fqe_data.row_expansion_factor,
            "fqe_row_count": int(fqe_data.states.shape[0]),
        }
        if oracle_candidate is not None:
            oracle_diag["selected_candidate_id"] = oracle_candidate.candidate_id
            try:
                oracle_cfg = replace(base, **dict(oracle_candidate.overrides), seed=int(dataset.seed) + 707_707, show_progress=False)
                model = fit_fqe_neural(
                    states=fqe_data.states,
                    actions=fqe_data.actions,
                    next_states=fqe_data.next_states,
                    next_actions=fqe_data.next_actions,
                    rewards=fqe_data.rewards,
                    gamma=fqe_data.gamma,
                    terminals=fqe_data.terminals,
                    sample_weight=fqe_data.sample_weight,
                    config=oracle_cfg,
                )
                oracle_estimate = _estimate_fqe_model_value(model, fqe_data)
                oracle_diag.update({f"final_{key}": value for key, value in getattr(model, "diagnostics", {}).items()})
            except Exception as exc:
                oracle_estimate = oracle_cv_value
                oracle_diag["oracle_refit_failed"] = 1
                oracle_diag["oracle_refit_error"] = f"{type(exc).__name__}: {exc}"
        estimates.append(
            _CalibrationEstimate(
                estimator="neural_fqe",
                tuning_track="oracle",
                status="ok" if _is_finite(oracle_estimate) else "error",
                estimate=oracle_estimate,
                runtime_sec=float(time.perf_counter() - start),
                skip_reason="" if _is_finite(oracle_estimate) else "oracle tuning did not produce a finite estimate",
                diagnostics=oracle_diag,
                diagnostic_only=True,
            )
        )
        candidates.extend(_fqe_candidate_rows(tuning, tuning_track="oracle", truth_target=target, oracle_candidate_id=None if oracle_candidate is None else oracle_candidate.candidate_id))
    return estimates, candidates


def _run_neural_occupancy_tracks(
    problem: BenchmarkProblem,
    config: CalibrationStudyConfig,
    scenario: DomainScenario,
) -> tuple[list[_CalibrationEstimate], list[dict[str, Any]]]:
    del scenario
    start = time.perf_counter()
    dataset = problem.dataset
    try:
        from occupancy_ratio import OccupancyTuningConfig, tune_occupancy_ratio_auto
    except ModuleNotFoundError as exc:
        return (_missing_rows("neural_occupancy", config, dataset.family, exc), [])
    try:
        space = _occupancy_search_space(config, int(dataset.seed))
    except ModuleNotFoundError as exc:
        return (_missing_rows("neural_occupancy", config, dataset.family, exc), [])
    occ_data = to_occupancy_ratio_dataset(dataset)
    tuning_config = OccupancyTuningConfig(
        families=("neural",),
        cv_folds=min(int(config.cv_folds), _max_group_folds(dataset.unit_id)),
        seed=int(dataset.seed) + 503,
        budget=str(config.occupancy_budget),
        max_candidates=int(config.occupancy_max_candidates),
        promotion_candidates=int(config.occupancy_promotion_candidates),
        refit=True,
        screen_fraction=0.5 if config.occupancy_budget == "balanced" else 0.3,
        score_method=str(config.occupancy_score_method),
        stagewise=True,
        first_stage_cv_folds=min(2, _max_group_folds(dataset.unit_id)),
        initial_ratio_mode_candidates=("auto", "factored"),
        one_step_ratio_mode_candidates=("auto", "factored"),
    )
    try:
        tuning = tune_occupancy_ratio_auto(
            states=occ_data.states,
            actions=occ_data.actions,
            next_states=occ_data.next_states,
            target_actions=occ_data.target_actions,
            gamma=occ_data.gamma,
            initial_states=occ_data.initial_states,
            initial_actions=occ_data.initial_actions,
            initial_weights=occ_data.initial_weights,
            target_next_actions=occ_data.next_target_actions,
            rewards=occ_data.rewards,
            groups=np.asarray(dataset.unit_id),
            families=("neural",),
            search_space=space,
            config=tuning_config,
            initial_ratio_mode="auto",
            one_step_ratio_mode="auto",
        )
    except ModuleNotFoundError as exc:
        return (_missing_rows("neural_occupancy", config, dataset.family, exc), [])
    estimates: list[_CalibrationEstimate] = []
    candidates = _occupancy_candidate_rows(tuning, tuning_track="proxy")
    weights = None if tuning.model is None else _predict_weights(tuning.model, occ_data.states, occ_data.actions)
    proxy_estimate = None if weights is None else _occupancy_value(weights, occ_data.rewards, occ_data.gamma, unit_id=dataset.unit_id, time=dataset.time)
    elapsed = float(time.perf_counter() - start)
    if "proxy" in config.tuning_tracks:
        estimates.append(
            _CalibrationEstimate(
                estimator="neural_occupancy",
                tuning_track="proxy",
                status="ok" if _is_finite(proxy_estimate) else "error",
                estimate=proxy_estimate,
                runtime_sec=elapsed,
                skip_reason="" if _is_finite(proxy_estimate) else "proxy tuning did not produce finite occupancy weights",
                diagnostics={
                    "selected_candidate_id": tuning.selected_candidate_id,
                    "selected_family": tuning.selected_family,
                    "selected_by": "proxy",
                    "candidate_count": len(tuning.candidates),
                    "fold_count": len(tuning.folds),
                    "stationary_normalized_value": None if weights is None else _occupancy_stationary_value(weights, occ_data.rewards, occ_data.gamma),
                    **_weight_diagnostics(weights),
                },
                diagnostic_only=dataset.family == "streamlift",
            )
        )
    if "oracle" in config.tuning_tracks:
        target = _target_for_problem(problem).value
        oracle_candidate, oracle_cv_value, oracle_error = _select_oracle_occupancy_candidate(tuning, target, occ_data.gamma)
        oracle_estimate = oracle_cv_value
        oracle_diag = {
            "selected_by": "oracle_truth",
            "candidate_count": len(tuning.candidates),
            "fold_count": len(tuning.folds),
            "oracle_cv_value": oracle_cv_value,
            "oracle_cv_abs_error": oracle_error,
        }
        if oracle_candidate is not None:
            oracle_diag["selected_candidate_id"] = oracle_candidate.candidate_id
            if oracle_candidate.candidate_id == tuning.selected_candidate_id and weights is not None:
                oracle_estimate = proxy_estimate
                oracle_diag.update(_weight_diagnostics(weights))
            else:
                oracle_diag["oracle_cv_value_used"] = 1
        estimates.append(
            _CalibrationEstimate(
                estimator="neural_occupancy",
                tuning_track="oracle",
                status="ok" if _is_finite(oracle_estimate) else "error",
                estimate=oracle_estimate,
                runtime_sec=float(time.perf_counter() - start),
                skip_reason="" if _is_finite(oracle_estimate) else "oracle tuning did not produce a finite estimate",
                diagnostics=oracle_diag,
                diagnostic_only=True,
            )
        )
        candidates.extend(
            _occupancy_candidate_rows(
                tuning,
                tuning_track="oracle",
                truth_target=target,
                gamma=occ_data.gamma,
                oracle_candidate_id=None if oracle_candidate is None else oracle_candidate.candidate_id,
            )
        )
    return estimates, candidates


@dataclass(frozen=True)
class _Target:
    name: str
    value: float
    mc_se: float
    noise_floor: float


def _target_for_problem(problem: BenchmarkProblem) -> _Target:
    truth = problem.truth
    dataset = problem.dataset
    if dataset.family == "streamlift":
        if "value_treatment_horizon_infinite" in truth.values and "value_control_horizon_infinite" in truth.values:
            horizon: int | str = "infinite"
        else:
            horizons = []
            for key in truth.values:
                if not key.startswith("value_treatment_horizon_"):
                    continue
                suffix = key.rsplit("_", maxsplit=1)[-1]
                if suffix.isdigit():
                    horizons.append(int(suffix))
            horizon = max(horizons) if horizons else int(dataset.metadata_public.get("long_horizon", 36))
        treatment_key = f"value_treatment_horizon_{horizon}"
        control_key = f"value_control_horizon_{horizon}"
        p_treat = 0.5
        if dataset.initial_action_probabilities is not None and dataset.initial_action_probabilities.shape[1] > 1:
            p_treat = float(np.mean(dataset.initial_action_probabilities[:, 1]))
        value = float(p_treat * truth.values[treatment_key] + (1.0 - p_treat) * truth.values[control_key])
        se = float(np.sqrt((p_treat * truth.target_standard_errors.get(treatment_key, 0.0)) ** 2 + ((1.0 - p_treat) * truth.target_standard_errors.get(control_key, 0.0)) ** 2))
        noise = max(float(truth.truth_noise_floor.get(treatment_key, 0.0)), float(truth.truth_noise_floor.get(control_key, 0.0)))
        return _Target(f"target_policy_value_horizon_{horizon}", value, se, noise)
    if "policy_value" in truth.values:
        return _Target(
            "policy_value",
            float(truth.values["policy_value"]),
            float(truth.target_standard_errors.get("policy_value", 0.0)),
            float(truth.truth_noise_floor.get("policy_value", 0.0)),
        )
    if truth.values:
        key = sorted(truth.values)[0]
        return _Target(key, float(truth.values[key]), float(truth.target_standard_errors.get(key, 0.0)), float(truth.truth_noise_floor.get(key, 0.0)))
    raise ValueError("TruthBundle contains no value target for calibration.")


def _row_from_estimate(
    config: CalibrationStudyConfig,
    problem: BenchmarkProblem,
    scenario: DomainScenario,
    target_policy: str,
    estimate: _CalibrationEstimate,
) -> dict[str, Any]:
    dataset = problem.dataset
    target = _target_for_problem(problem)
    diagnostics = estimate.diagnostics or {}
    row: dict[str, Any] = {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "preset": config.preset,
        "family": dataset.family,
        "dataset": dataset.name,
        "scenario": dataset.scenario,
        "scenario_public": dataset.scenario,
        "primary_calibration_cell": int(bool(scenario.leaderboard_eligible and scenario.name in {"clean_randomized_good_overlap", "observed_moderate_overlap"})),
        "estimator": estimate.estimator,
        "tuning_track": estimate.tuning_track,
        "selected_by": diagnostics.get("selected_by", estimate.tuning_track),
        "status": estimate.status,
        "skip_reason": estimate.skip_reason,
        "diagnostic_only": int(bool(estimate.diagnostic_only)),
        "leaderboard_eligible": 0,
        "leaderboard_result_eligible": 0,
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.metadata_public.get("sample_size", dataset.n)),
        "row_count": int(dataset.n),
        "target_policy": target_policy,
        "target_estimand": target.name,
        "truth_target": float(target.value),
        "truth_mc_se": float(target.mc_se),
        "truth_noise_floor": float(target.noise_floor),
        "estimate": "" if estimate.estimate is None else float(estimate.estimate),
        "runtime_sec": float(estimate.runtime_sec),
    }
    row.update(_scenario_diagnostics(problem))
    if _is_finite(estimate.estimate):
        error = float(estimate.estimate) - float(target.value)
        row["error"] = error
        row["abs_error"] = abs(error)
        row["normalizer"] = _normalizer(problem, target)
        row["normalized_abs_error"] = abs(error) / max(float(row["normalizer"]), 1e-12)
        row["finite_estimate"] = 1
    else:
        row["finite_estimate"] = 0
    row.update({f"diag_{key}": value for key, value in diagnostics.items() if _public_scalar(value)})
    return row


def _problem_error_row(
    config: CalibrationStudyConfig,
    family: str,
    scenario: DomainScenario,
    sample_size: int,
    gamma: float,
    seed: int,
    target_policy: str,
    exc: Exception,
) -> dict[str, Any]:
    status = "missing_dependency" if isinstance(exc, ModuleNotFoundError) else "error"
    return {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "preset": config.preset,
        "family": family,
        "dataset": "",
        "scenario": scenario.name,
        "scenario_public": scenario.name,
        "primary_calibration_cell": int(bool(scenario.leaderboard_eligible)),
        "estimator": "dataset",
        "tuning_track": "dataset",
        "selected_by": "dataset",
        "status": status,
        "skip_reason": f"{type(exc).__name__}: {exc}",
        "diagnostic_only": 1,
        "leaderboard_eligible": 0,
        "leaderboard_result_eligible": 0,
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "row_count": 0,
        "target_policy": target_policy,
        "runtime_sec": 0.0,
        "finite_estimate": 0,
    }


def _summarize_calibration_rows(rows: list[dict[str, Any]], config: CalibrationStudyConfig) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("family", "")), str(row.get("scenario", "")), str(row.get("estimator", "")), str(row.get("tuning_track", "")))
        groups.setdefault(key, []).append(row)
    verdicts = _difficulty_verdicts(rows, config)
    out: list[dict[str, Any]] = []
    for (family, scenario, estimator, track), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        finite = [row for row in ok if _truthy(row.get("finite_estimate"))]
        nae = [_as_float(row.get("normalized_abs_error")) for row in finite]
        abs_err = [_as_float(row.get("abs_error")) for row in finite]
        out.append(
            {
                "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
                "preset": config.preset,
                "family": family,
                "scenario": scenario,
                "estimator": estimator,
                "tuning_track": track,
                "n_rows": len(group),
                "ok_rows": len(ok),
                "finite_rows": len(finite),
                "finite_rate": len(finite) / max(len(group), 1),
                "median_abs_error": _nanmedian(abs_err),
                "median_normalized_abs_error": _nanmedian(nae),
                "p90_normalized_abs_error": _nanquantile(nae, 0.90),
                "runtime_sec_mean": _nanmean([_as_float(row.get("runtime_sec")) for row in group]),
                "difficulty_verdict": verdicts.get((family, scenario), "unclassified"),
            }
        )
    return out


def _difficulty_verdicts(rows: list[dict[str, Any]], config: CalibrationStudyConfig) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for family in sorted({str(row.get("family", "")) for row in rows}):
        if family == "streamlift":
            for scenario in sorted({str(row.get("scenario", "")) for row in rows if row.get("family") == family}):
                out[(family, scenario)] = "diagnostic_only"
            continue
        for scenario in sorted({str(row.get("scenario", "")) for row in rows if row.get("family") == family}):
            group = [row for row in rows if row.get("family") == family and row.get("scenario") == scenario and _truthy(row.get("primary_calibration_cell"))]
            if not group:
                out[(family, scenario)] = "sensitivity_or_nonprimary"
                continue
            proxy = [row for row in group if row.get("tuning_track") == "proxy" and row.get("status") == "ok"]
            oracle = [row for row in group if row.get("tuning_track") == "oracle" and row.get("status") == "ok"]
            proxy_nae = [_as_float(row.get("normalized_abs_error")) for row in proxy if _truthy(row.get("finite_estimate"))]
            oracle_nae = [_as_float(row.get("normalized_abs_error")) for row in oracle if _truthy(row.get("finite_estimate"))]
            finite_rate = len(proxy_nae) / max(len([row for row in group if row.get("tuning_track") == "proxy"]), 1)
            proxy_med = _nanmedian(proxy_nae)
            oracle_med = _nanmedian(oracle_nae)
            if finite_rate >= float(config.finite_rate_threshold) and np.isfinite(proxy_med) and proxy_med <= float(config.normalized_error_threshold):
                verdict = "estimable"
            elif np.isfinite(oracle_med) and oracle_med <= float(config.normalized_error_threshold):
                verdict = "tuning_gap"
            elif np.isfinite(oracle_med):
                verdict = "model_gap"
            else:
                verdict = "too_hard"
            out[(family, scenario)] = verdict
    return out


def _fqe_candidate_grid(config: CalibrationStudyConfig) -> list[dict[str, Any]]:
    base_dims = tuple(int(width) for width in config.fqe_hidden_dims)
    small_dims = tuple(max(8, min(width, 32)) for width in base_dims)
    large_dims = tuple(max(32, width * 2) for width in base_dims)
    candidates = [
        {},
        {"hidden_dims": small_dims, "learning_rate": 1e-3, "target_update_tau": 0.35},
        {"hidden_dims": large_dims, "learning_rate": 2e-4, "target_update_tau": 0.05},
        {"learning_rate": 1e-3},
        {"learning_rate": 2e-4},
        {"weight_decay": 0.0},
        {"weight_decay": 1e-4},
        {"target_update_tau": 0.35},
        {"target_update_tau": 0.05},
        {"loss": "squared"},
    ]
    return candidates[: int(config.fqe_max_candidates)]


def _occupancy_search_space(config: CalibrationStudyConfig, seed: int) -> Any:
    from occupancy_ratio import (
        NeuralActionRatioConfig,
        NeuralOccupancyRegressionConfig,
        NeuralSourceStateRatioConfig,
        NeuralTransitionRatioConfig,
        OccupancySearchSpace,
    )

    dims = tuple(int(width) for width in config.occupancy_hidden_dims)
    occ = NeuralOccupancyRegressionConfig.stable_defaults(
        hidden_dims=dims,
        batch_size=int(config.occupancy_batch_size),
        num_iterations=int(config.occupancy_num_iterations),
        gradient_steps_per_iteration=int(config.occupancy_gradient_steps_per_iteration),
        mcmc_samples=int(config.occupancy_mcmc_samples),
        learning_rate=5e-4,
        weight_decay=1e-5,
        occupancy_ratio_max=50.0,
        direct_one_step_prediction_max=50.0,
        seed=int(seed),
        device="cpu",
        show_progress=False,
    )
    action = NeuralActionRatioConfig.stable_defaults(
        hidden_dims=dims,
        batch_size=int(config.occupancy_batch_size),
        max_steps=int(config.occupancy_nuisance_max_steps),
        prediction_max=50.0,
        seed=int(seed) + 11,
        device="cpu",
    )
    source = NeuralSourceStateRatioConfig.stable_defaults(
        hidden_dims=dims,
        batch_size=int(config.occupancy_batch_size),
        max_steps=int(config.occupancy_nuisance_max_steps),
        prediction_max=50.0,
        seed=int(seed) + 13,
        device="cpu",
    )
    transition = NeuralTransitionRatioConfig.stable_defaults(
        hidden_dims=dims,
        batch_size=int(config.occupancy_batch_size),
        max_steps=int(config.occupancy_nuisance_max_steps),
        prediction_max=50.0,
        permutation_samples=2 if config.preset == "smoke" else 4,
        seed=int(seed) + 17,
        device="cpu",
    )
    small_dims = tuple(max(8, min(width, 32)) for width in dims)
    large_dims = tuple(max(32, width * 2) for width in dims)
    candidates = [
        {},
        {"occupancy": {"hidden_dims": small_dims, "occupancy_ratio_max": 25.0}, "action_ratio": {"hidden_dims": small_dims, "prediction_max": 25.0}, "source_state_ratio": {"hidden_dims": small_dims, "prediction_max": 25.0}, "transition_ratio": {"hidden_dims": small_dims, "prediction_max": 25.0}},
        {"occupancy": {"hidden_dims": large_dims, "occupancy_ratio_max": 100.0, "pseudo_outcome_upper_quantile": 0.999}, "action_ratio": {"hidden_dims": large_dims, "prediction_max": 100.0}, "source_state_ratio": {"hidden_dims": large_dims, "prediction_max": 100.0}, "transition_ratio": {"hidden_dims": large_dims, "prediction_max": 100.0}},
        {"occupancy": {"fixed_point_damping": 0.75, "occupancy_ratio_max": 100.0}, "action_ratio": {"prediction_max": 100.0}, "source_state_ratio": {"prediction_max": 100.0}, "transition_ratio": {"prediction_max": 100.0}},
        {"occupancy": {"fixed_point_damping": 0.35, "occupancy_ratio_max": 25.0}, "action_ratio": {"prediction_max": 25.0}, "source_state_ratio": {"prediction_max": 25.0}, "transition_ratio": {"prediction_max": 25.0}},
        {"modes": {"initial_ratio_mode": "factored", "one_step_ratio_mode": "factored"}},
    ]
    return OccupancySearchSpace(
        neural_occupancy=occ,
        neural_action_ratio=action,
        neural_source_state_ratio=source,
        neural_transition_ratio=transition,
        neural_candidates=candidates[: int(config.occupancy_max_candidates)],
    )


def _estimate_fqe_model_value(model: Any, fqe_data: Any) -> float:
    probs = np.asarray(fqe_data.initial_action_probabilities, dtype=np.float64)
    states = np.asarray(fqe_data.initial_states, dtype=np.float64)
    action_eye = np.eye(probs.shape[1], dtype=np.float64)
    states_rep = np.repeat(states, probs.shape[1], axis=0)
    actions_rep = np.tile(action_eye, (states.shape[0], 1))
    q = np.asarray(model.predict_q(states_rep, actions_rep), dtype=np.float64).reshape(states.shape[0], probs.shape[1])
    return float(np.mean(np.sum(probs * q, axis=1)))


def _predict_weights(model: Any, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_state_action_ratio(states, actions, clip=True), dtype=np.float64).reshape(-1)


def _occupancy_value(
    weights: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    *,
    unit_id: np.ndarray | None = None,
    time: np.ndarray | None = None,
) -> float:
    if unit_id is None or time is None:
        return _occupancy_stationary_value(weights, rewards, gamma)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    units = np.asarray(unit_id).reshape(-1)
    t = np.asarray(time, dtype=np.int64).reshape(-1)
    if not (w.shape[0] == r.shape[0] == units.shape[0] == t.shape[0]):
        raise ValueError("weights, rewards, unit_id, and time must align.")
    values = []
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        values.append(float(np.sum((float(gamma) ** t[idx]) * w[idx] * r[idx])))
    return float(np.mean(values)) if values else 0.0


def _occupancy_stationary_value(weights: np.ndarray, rewards: np.ndarray, gamma: float) -> float:
    normalized = float(np.mean(np.asarray(weights, dtype=np.float64) * np.asarray(rewards, dtype=np.float64).reshape(-1)))
    return normalized / max(1.0 - float(gamma), 1e-12)


def _fqe_candidate_rows(tuning: Any, *, tuning_track: str, truth_target: float | None = None, oracle_candidate_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tuning.candidate_rows():
        out = dict(row)
        out.update(
            {
                "row_type": "candidate",
                "estimator": "neural_fqe",
                "tuning_track": tuning_track,
                "selected_by": "oracle_truth" if tuning_track == "oracle" else "proxy",
                "diagnostic_only": int(tuning_track == "oracle"),
            }
        )
        candidate_value = _candidate_policy_value(tuning.candidates, str(out.get("candidate_id", "")))
        if truth_target is not None and _is_finite(candidate_value):
            out["oracle_cv_value"] = candidate_value
            out["oracle_cv_abs_error"] = abs(float(candidate_value) - float(truth_target))
        if oracle_candidate_id is not None:
            out["oracle_selected"] = int(str(out.get("candidate_id", "")) == str(oracle_candidate_id))
        rows.append(out)
    for row in tuning.fold_rows():
        out = dict(row)
        out.update(
            {
                "row_type": "fold",
                "estimator": "neural_fqe",
                "tuning_track": tuning_track,
                "selected_by": "oracle_truth" if tuning_track == "oracle" else "proxy",
                "diagnostic_only": int(tuning_track == "oracle"),
            }
        )
        rows.append(out)
    return rows


def _occupancy_candidate_rows(
    tuning: Any,
    *,
    tuning_track: str,
    truth_target: float | None = None,
    gamma: float | None = None,
    oracle_candidate_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tuning.candidate_rows():
        out = dict(row)
        out.update(
            {
                "row_type": "candidate",
                "estimator": "neural_occupancy",
                "tuning_track": tuning_track,
                "selected_by": "oracle_truth" if tuning_track == "oracle" else "proxy",
                "diagnostic_only": int(tuning_track == "oracle"),
            }
        )
        candidate_value = _candidate_reward_value(tuning.candidates, str(out.get("candidate_id", "")), gamma)
        if truth_target is not None and _is_finite(candidate_value):
            out["oracle_cv_value"] = candidate_value
            out["oracle_cv_abs_error"] = abs(float(candidate_value) - float(truth_target))
        if oracle_candidate_id is not None:
            out["oracle_selected"] = int(str(out.get("candidate_id", "")) == str(oracle_candidate_id))
        rows.append(out)
    for row in tuning.fold_rows():
        out = dict(row)
        out.update(
            {
                "row_type": "fold",
                "estimator": "neural_occupancy",
                "tuning_track": tuning_track,
                "selected_by": "oracle_truth" if tuning_track == "oracle" else "proxy",
                "diagnostic_only": int(tuning_track == "oracle"),
            }
        )
        rows.append(out)
    for row in tuning.first_stage_candidate_rows():
        out = dict(row)
        out.update(
            {
                "row_type": "first_stage_candidate",
                "estimator": "neural_occupancy",
                "tuning_track": tuning_track,
                "selected_by": "oracle_truth" if tuning_track == "oracle" else "proxy",
                "diagnostic_only": int(tuning_track == "oracle"),
            }
        )
        rows.append(out)
    return rows


def _select_oracle_fqe_candidate(tuning: Any, truth_target: float) -> tuple[Any | None, float | None, float | None]:
    best = None
    best_value = None
    best_error = float("inf")
    candidates = _preferred_candidate_stage(tuning.candidates)
    for candidate in candidates:
        value = _candidate_value_from_folds(candidate.fold_results, value_attr="policy_value")
        if not _is_finite(value):
            continue
        error = abs(float(value) - float(truth_target))
        if error < best_error:
            best = candidate
            best_value = value
            best_error = error
    return best, best_value, None if not np.isfinite(best_error) else float(best_error)


def _select_oracle_occupancy_candidate(tuning: Any, truth_target: float, gamma: float) -> tuple[Any | None, float | None, float | None]:
    best = None
    best_value = None
    best_error = float("inf")
    candidates = _preferred_candidate_stage(tuning.candidates)
    for candidate in candidates:
        value = _candidate_value_from_folds(candidate.fold_results, value_attr="reward_value")
        if _is_finite(value):
            value = float(value) / max(1.0 - float(gamma), 1e-12)
        if not _is_finite(value):
            continue
        error = abs(float(value) - float(truth_target))
        if error < best_error:
            best = candidate
            best_value = value
            best_error = error
    return best, best_value, None if not np.isfinite(best_error) else float(best_error)


def _preferred_candidate_stage(candidates: Sequence[Any]) -> list[Any]:
    full = [candidate for candidate in candidates if getattr(candidate, "budget_stage", "") == "full" and not getattr(candidate, "error", "")]
    if full:
        return full
    return [candidate for candidate in candidates if not getattr(candidate, "error", "")]


def _candidate_policy_value(candidates: Sequence[Any], candidate_id: str) -> float | None:
    matches = [candidate for candidate in _preferred_candidate_stage(candidates) if str(candidate.candidate_id) == str(candidate_id)]
    if not matches:
        return None
    return _candidate_value_from_folds(matches[-1].fold_results, value_attr="policy_value")


def _candidate_reward_value(candidates: Sequence[Any], candidate_id: str, gamma: float | None) -> float | None:
    if gamma is None:
        return None
    matches = [candidate for candidate in _preferred_candidate_stage(candidates) if str(candidate.candidate_id) == str(candidate_id)]
    if not matches:
        return None
    value = _candidate_value_from_folds(matches[-1].fold_results, value_attr="reward_value")
    if not _is_finite(value):
        return None
    return float(value) / max(1.0 - float(gamma), 1e-12)


def _candidate_value_from_folds(folds: Sequence[Any], *, value_attr: str) -> float | None:
    values = [float(getattr(fold, value_attr)) for fold in folds if _is_finite(getattr(fold, value_attr, None))]
    if not values:
        return None
    return float(np.mean(values))


def _with_problem_context(
    rows: list[dict[str, Any]],
    config: CalibrationStudyConfig,
    problem: BenchmarkProblem,
    scenario: DomainScenario,
    target_policy: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        enriched = {
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "preset": config.preset,
            "family": problem.dataset.family,
            "dataset": problem.dataset.name,
            "scenario": problem.dataset.scenario,
            "scenario_public": problem.dataset.scenario,
            "primary_calibration_cell": int(bool(scenario.leaderboard_eligible and scenario.name in {"clean_randomized_good_overlap", "observed_moderate_overlap"})),
            "seed": int(problem.dataset.seed),
            "sample_size": int(problem.dataset.metadata_public.get("sample_size", problem.dataset.n)),
            "gamma": float(problem.dataset.gamma),
            "target_policy": target_policy,
        }
        enriched.update(row)
        out.append(enriched)
    return out


def _missing_rows(estimator: str, config: CalibrationStudyConfig, family: str, exc: Exception) -> list[_CalibrationEstimate]:
    return [
        _CalibrationEstimate(
            estimator=estimator,
            tuning_track=track,
            status="missing_dependency",
            skip_reason=f"{type(exc).__name__}: {exc}",
            diagnostic_only=track == "oracle" or family == "streamlift",
        )
        for track in config.tuning_tracks
    ]


def _scenarios_for_calibration(config: CalibrationStudyConfig, family: str) -> tuple[DomainScenario, ...]:
    profile = "smoke" if config.preset == "smoke" else "core"
    scenarios = scenarios_for_profile(profile, family)[:2]
    if config.include_stress:
        scenarios = scenarios + sensitivity_scenarios_for_profile("full", family)
    return scenarios


def _target_policies_for_family(config: CalibrationStudyConfig, family: str) -> tuple[str, ...]:
    requested = tuple(config.target_policies or POLICY_NAMES_BY_FAMILY.get(family, ("moderate",)))
    allowed = set(POLICY_NAMES_BY_FAMILY.get(family, ()))
    return tuple(policy for policy in requested if policy in allowed)


def _problem_config(config: CalibrationStudyConfig) -> CausalOPEBenchmarkConfig:
    profile = "smoke" if config.preset == "smoke" else "core"
    return CausalOPEBenchmarkConfig(
        profile=profile,
        output_root=config.output_root,
        seeds=tuple(config.seeds),
        families=tuple(config.families),
        sample_sizes=tuple(config.sample_sizes),
        gammas=tuple(config.gammas),
        observed_horizons=(int(config.streamlift_observed_horizon),),
        target_policies=tuple(config.target_policies or ("moderate",)),
        estimators=(),
        trajectory_horizon=int(config.trajectory_horizon),
        streamlift_long_horizon=int(config.streamlift_long_horizon),
        mc_truth_rollouts=int(config.mc_truth_rollouts),
        fqe_hidden_dims=tuple(config.fqe_hidden_dims),
        fqe_num_iterations=int(config.fqe_num_iterations),
        fqe_gradient_steps_per_iteration=int(config.fqe_gradient_steps_per_iteration),
        fqe_batch_size=int(config.fqe_batch_size),
        fail_fast=bool(config.fail_fast),
    )


def _groups_for_fqe(unit_id: np.ndarray, source_row_index: np.ndarray | None) -> np.ndarray:
    units = np.asarray(unit_id)
    if source_row_index is None:
        return units
    return units[np.asarray(source_row_index, dtype=np.int64)]


def _max_group_folds(groups: np.ndarray) -> int:
    unique = np.unique(np.asarray(groups))
    return max(2, min(3, int(unique.shape[0])))


def _scenario_diagnostics(problem: BenchmarkProblem) -> dict[str, Any]:
    dataset = problem.dataset
    ratios = np.asarray(dataset.target_propensity_observed_action, dtype=np.float64) / np.clip(np.asarray(dataset.behavior_propensity, dtype=np.float64), 1e-12, np.inf)
    return {
        "diag_overlap_ratio_min": float(np.min(ratios)),
        "diag_overlap_ratio_p5": float(np.quantile(ratios, 0.05)),
        "diag_overlap_ratio_p50": float(np.quantile(ratios, 0.50)),
        "diag_target_behavior_policy_distance": float(np.mean(np.abs(ratios - 1.0))),
        "diag_terminal_rate": float(np.mean(dataset.terminals)),
        "diag_censoring_rate": float(np.mean(dataset.censoring)),
        "diag_missingness_rate": float(np.mean(dataset.missingness_mask)),
        "diag_reward_mean": float(np.mean(dataset.rewards)),
        "diag_reward_sd": float(np.std(dataset.rewards)),
    }


def _weight_diagnostics(weights: np.ndarray | None) -> dict[str, Any]:
    if weights is None:
        return {}
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return {"ess_fraction": 0.0, "weight_cv": float("nan")}
    return {
        "ess_fraction": _ess_fraction(w),
        "weight_mean": float(np.mean(w)),
        "weight_cv": float(np.std(w) / max(abs(float(np.mean(w))), 1e-12)),
        "weight_p95": float(np.quantile(w, 0.95)),
        "weight_p99": float(np.quantile(w, 0.99)),
        "weight_max": float(np.max(w)),
        "near_uniform_collapse": int(float(np.std(w)) < 1e-3),
    }


def _normalizer(problem: BenchmarkProblem, target: _Target) -> float:
    horizon = int(problem.dataset.metadata_public.get("trajectory_horizon", problem.dataset.metadata_public.get("long_horizon", np.max(problem.dataset.time) + 1)))
    discount_sum = float(np.sum(float(problem.dataset.gamma) ** np.arange(max(horizon, 1))))
    reward_scale = max(float(np.std(problem.dataset.rewards)), 1.0) * discount_sum
    return max(abs(float(target.value)), abs(float(target.noise_floor)), 1.96 * abs(float(target.mc_se)), 0.25 * reward_scale, 1e-8)


def _manifest(config: CalibrationStudyConfig) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "fqe", "torch", "occupancy-ratio", "occupancy_ratio", "gym", "gymnasium", "epicare"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "config": asdict(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "optional_dependencies": {
            "fqe": packages.get("fqe") is not None,
            "torch": packages.get("torch") is not None,
            "occupancy_ratio": packages.get("occupancy-ratio") is not None or packages.get("occupancy_ratio") is not None,
            "gym": packages.get("gym") is not None,
            "gymnasium": packages.get("gymnasium") is not None,
            "epicare": packages.get("epicare") is not None,
        },
    }


def _render_readout(
    config: CalibrationStudyConfig,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    failures: list[str],
    manifest: dict[str, Any],
) -> str:
    status_counts: dict[str, int] = {}
    verdicts: dict[tuple[str, str], str] = {}
    for row in rows:
        status_counts[str(row.get("status", ""))] = status_counts.get(str(row.get("status", "")), 0) + 1
    for row in summary:
        key = (str(row.get("family", "")), str(row.get("scenario", "")))
        verdicts[key] = str(row.get("difficulty_verdict", ""))
    optional = manifest.get("optional_dependencies", {})
    optional_text = ", ".join(f"{key}={'yes' if value else 'no'}" for key, value in sorted(optional.items())) if isinstance(optional, dict) else "unavailable"
    lines = [
        "# Neural Calibration Readout",
        "",
        f"- preset: `{config.preset}`",
        f"- result rows: `{len(rows)}`",
        f"- candidate rows: `{len(candidate_rows)}`",
        f"- failures: `{len(failures)}`",
        f"- optional dependencies: {optional_text}",
        "",
        "## Difficulty Verdicts",
        "",
        "| family | scenario | verdict |",
        "| --- | --- | --- |",
    ]
    for (family, scenario), verdict in sorted(verdicts.items()):
        lines.append(f"| {family} | {scenario} | {verdict} |")
    lines.extend(["", "## Estimator Summary", "", "| family | scenario | estimator | track | finite rate | median NAE | runtime |", "| --- | --- | --- | --- | ---: | ---: | ---: |"])
    for row in summary:
        lines.append(
            "| {family} | {scenario} | {estimator} | {track} | {finite} | {nae} | {runtime} |".format(
                family=row.get("family", ""),
                scenario=row.get("scenario", ""),
                estimator=row.get("estimator", ""),
                track=row.get("tuning_track", ""),
                finite=_fmt(row.get("finite_rate")),
                nae=_fmt(row.get("median_normalized_abs_error")),
                runtime=_fmt(row.get("runtime_sec_mean")),
            )
        )
    lines.extend(["", "## Status Summary", "", "| status | rows |", "| --- | ---: |"])
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "Oracle-track rows are diagnostic upper bounds only. They use sealed truth for candidate selection after fitting and are never leaderboard eligible.",
            "StreamLift FQE and occupancy rows are diagnostic-only because StreamLift remains a short-panel long-term causal forecasting benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _nanquantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _ess_fraction(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or float(np.sum(w * w)) <= 0.0:
        return 0.0
    return float((np.sum(w) ** 2) / (w.size * np.sum(w * w)))


def _public_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None


def _fmt(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.4g}"
