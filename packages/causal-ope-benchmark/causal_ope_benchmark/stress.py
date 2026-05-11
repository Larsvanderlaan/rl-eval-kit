from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from causal_ope_benchmark.adapters import to_fqe_dataset, to_occupancy_ratio_dataset
from causal_ope_benchmark.baselines import EstimatorResult, run_estimator
from causal_ope_benchmark.calibration import (
    CalibrationStudyConfig,
    _estimate_fqe_model_value,
    _normalizer,
    _occupancy_value,
    _run_estimator_tracks,
    _target_for_problem,
)
from causal_ope_benchmark.config import CausalOPEBenchmarkConfig, FamilyName
from causal_ope_benchmark.constants import DIFFICULTY_OUTPUT_FILES, DIFFICULTY_SCHEMA_VERSION, PACKAGE_VERSION
from causal_ope_benchmark.difficulty import (
    DIFFICULTY_NAMES,
    DifficultyCell,
    StressScale,
    describe_difficulty,
    difficulty_cells,
    sample_sizes_for_difficulty,
    seeds_for_scale,
    target_policies_for_difficulty,
)
from causal_ope_benchmark.io import write_csv, write_json
from causal_ope_benchmark.policies import POLICY_NAMES_BY_FAMILY
from causal_ope_benchmark.runner import make_problem
from causal_ope_benchmark.types import BenchmarkProblem


DEFAULT_STRESS_METHODS: tuple[str, ...] = (
    "naive_short_term",
    "streamlift_stratified_gcomp",
    "direct_method",
    "ipw",
    "snipw",
    "doubly_robust",
    "linear_fqe",
    "boosted_fqe_auto",
    "neural_fqe_auto",
    "discounted_occupancy_boosted_auto",
    "discounted_occupancy_neural_auto",
    "google_dualdice_neural",
    "google_dualdice_weighted_fqe",
    "dice_rl_neural",
)


@dataclass(frozen=True)
class DifficultyStressStudyConfig:
    """Configuration for systematic benchmark difficulty calibration."""

    scale: StressScale = "ci"
    output_root: Path = Path("outputs/causal_ope_benchmark")
    difficulties: Sequence[str] = DIFFICULTY_NAMES
    families: Sequence[FamilyName] = ("streamlift", "streamretain", "clinic_dtr")
    include_epicare: bool = False
    include_sensitivity: bool = False
    methods: Sequence[str] = DEFAULT_STRESS_METHODS
    seeds: Sequence[int] | None = None
    target_policies: Sequence[str] | None = None
    sample_sizes: Sequence[int] | None = None
    gammas: Sequence[float] | None = None
    mc_truth_rollouts: int | None = None
    google_research_path: Path = Path("/tmp/google-research")
    dice_rl_repo_path: Path = Path("/tmp/dice_rl")
    include_oracle_tracks: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "google_research_path", Path(self.google_research_path))
        object.__setattr__(self, "dice_rl_repo_path", Path(self.dice_rl_repo_path))
        if self.scale not in {"ci", "audit", "exhaustive"}:
            raise ValueError("scale must be 'ci', 'audit', or 'exhaustive'.")
        for difficulty in self.difficulties:
            describe_difficulty(str(difficulty))
        families = tuple(str(family) for family in self.families)
        if self.include_epicare and "epicare" not in families:
            families = (*families, "epicare")
            object.__setattr__(self, "families", families)
        for family in self.families:
            if family not in {"streamlift", "streamretain", "clinic_dtr", "epicare"}:
                raise ValueError(f"Unsupported family '{family}'.")
        if not self.methods:
            raise ValueError("methods must be nonempty.")
        if self.seeds is not None and not self.seeds:
            raise ValueError("seeds must be nonempty when supplied.")
        if self.sample_sizes is not None and any(int(size) <= 0 for size in self.sample_sizes):
            raise ValueError("sample_sizes must be positive.")
        if self.gammas is not None:
            for gamma in self.gammas:
                if not (0.0 <= float(gamma) < 1.0):
                    raise ValueError("gammas must be in [0, 1).")
        if self.mc_truth_rollouts is not None and int(self.mc_truth_rollouts) <= 0:
            raise ValueError("mc_truth_rollouts must be positive.")

    @classmethod
    def for_scale(
        cls,
        scale: StressScale,
        *,
        output_root: str | Path = Path("outputs/causal_ope_benchmark"),
    ) -> "DifficultyStressStudyConfig":
        """Return a named runtime scale."""

        if scale == "ci":
            return cls(
                scale="ci",
                output_root=Path(output_root),
                methods=("direct_method", "ipw", "linear_fqe", "neural_fqe_auto", "discounted_occupancy_neural_auto", "google_dualdice_neural"),
                mc_truth_rollouts=8,
            )
        if scale == "audit":
            return cls(scale="audit", output_root=Path(output_root), mc_truth_rollouts=96)
        if scale == "exhaustive":
            return cls(scale="exhaustive", output_root=Path(output_root), include_sensitivity=True, mc_truth_rollouts=192)
        raise ValueError("scale must be 'ci', 'audit', or 'exhaustive'.")

    def output_dir(self) -> Path:
        return self.output_root / "difficulty" / self.scale


@dataclass
class DifficultyStressRunResult:
    """Output bundle for a completed difficulty stress study."""

    output_dir: Path
    results_path: Path
    summary_path: Path
    candidates_path: Path
    manifest_path: Path
    readout_path: Path
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]


def run_difficulty_stress_study(
    config: DifficultyStressStudyConfig | None = None,
    **overrides: Any,
) -> DifficultyStressRunResult:
    """Run systematic stress tests across difficulty profiles."""

    cfg = config or DifficultyStressStudyConfig.for_scale(str(overrides.pop("scale", "ci")))  # type: ignore[arg-type]
    if overrides:
        cfg = DifficultyStressStudyConfig(**{**cfg.__dict__, **overrides})
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for difficulty in cfg.difficulties:
        spec = describe_difficulty(str(difficulty))
        for family in cfg.families:
            family_key = str(family)
            for cell in difficulty_cells(str(difficulty), family_key, scale=cfg.scale, include_sensitivity=cfg.include_sensitivity):  # type: ignore[arg-type]
                sample_sizes = tuple(int(size) for size in (cfg.sample_sizes or sample_sizes_for_difficulty(str(difficulty), family_key, scale=cfg.scale)))  # type: ignore[arg-type]
                gammas = tuple(float(gamma) for gamma in (cfg.gammas or spec.gammas))
                seeds = tuple(int(seed) for seed in (cfg.seeds or seeds_for_scale(cfg.scale)))
                policies = _target_policies(cfg, str(difficulty), family_key)
                for sample_size in sample_sizes:
                    for gamma in gammas:
                        for seed in seeds:
                            for target_policy in policies:
                                try:
                                    problem = make_problem(
                                        family=family_key,
                                        scenario=cell.scenario,
                                        sample_size=sample_size,
                                        gamma=gamma,
                                        seed=seed,
                                        observed_horizon=_observed_horizon(spec, seed),
                                        target_policy=target_policy,
                                        config=_problem_config(cfg, spec, family_key, sample_size, gamma),
                                    )
                                except Exception as exc:
                                    if cfg.fail_fast:
                                        raise
                                    failures.append(traceback.format_exc())
                                    rows.append(_problem_error_row(cfg, cell, family_key, sample_size, gamma, seed, target_policy, exc))
                                    continue
                                for method in _methods_for_family(cfg, family_key):
                                    try:
                                        estimates, candidates = _run_method_tracks(method, problem, cfg, cell)
                                    except Exception as exc:
                                        if cfg.fail_fast:
                                            raise
                                        failures.append(traceback.format_exc())
                                        estimates = [
                                            EstimatorResult(
                                                estimator=method,
                                                status="error",
                                                skip_reason=f"{type(exc).__name__}: {exc}",
                                                diagnostic_only=_diagnostic_method(method, family_key),
                                            )
                                        ]
                                        candidates = []
                                    candidate_rows.extend(_candidate_rows_with_context(candidates, cfg, problem, cell, target_policy))
                                    for result in estimates:
                                        rows.append(_row_from_result(cfg, problem, cell, target_policy, method, result))
    summary_rows = _summarize(rows)
    manifest = _manifest(cfg)
    readout = _render_readout(cfg, rows, summary_rows, candidate_rows, failures, manifest)
    results_path = output_dir / DIFFICULTY_OUTPUT_FILES["results"]
    summary_path = output_dir / DIFFICULTY_OUTPUT_FILES["summary"]
    candidates_path = output_dir / DIFFICULTY_OUTPUT_FILES["candidates"]
    manifest_path = output_dir / DIFFICULTY_OUTPUT_FILES["manifest"]
    readout_path = output_dir / DIFFICULTY_OUTPUT_FILES["readout"]
    write_csv(results_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(candidates_path, candidate_rows)
    write_json(manifest_path, manifest)
    readout_path.write_text(readout, encoding="utf-8")
    return DifficultyStressRunResult(
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


def _run_method_tracks(
    method: str,
    problem: BenchmarkProblem,
    config: DifficultyStressStudyConfig,
    cell: DifficultyCell,
) -> tuple[list[EstimatorResult], list[dict[str, Any]]]:
    if method == "neural_fqe_auto" and config.include_oracle_tracks:
        return _calibration_tracks_as_results("neural_fqe_auto", "neural_fqe", problem, config, cell)
    if method == "discounted_occupancy_neural_auto" and config.include_oracle_tracks:
        return _calibration_tracks_as_results("discounted_occupancy_neural_auto", "neural_occupancy", problem, config, cell)
    if method == "google_dualdice_neural":
        return ([_google_dualdice_neural(problem, config)], [])
    if method == "google_dualdice_weighted_fqe":
        return ([_google_dualdice_weighted_fqe(problem, config)], [])
    if method == "dice_rl_neural":
        return ([_dice_rl_neural(problem, config)], [])
    result = run_estimator(method, problem, config=_problem_config_from_problem(config, problem))
    return ([result], list(result.tuning_rows))


def _calibration_tracks_as_results(
    public_method: str,
    calibration_estimator: str,
    problem: BenchmarkProblem,
    config: DifficultyStressStudyConfig,
    cell: DifficultyCell,
) -> tuple[list[EstimatorResult], list[dict[str, Any]]]:
    cal_cfg = _calibration_config(config, problem)
    estimates, candidates = _run_estimator_tracks(calibration_estimator, problem, cal_cfg, cell.scenario)
    target = _target(problem, value_for_streamlift=_diagnostic_method(public_method, problem.dataset.family))
    out: list[EstimatorResult] = []
    for estimate in estimates:
        tuning_track = str(estimate.tuning_track)
        result = EstimatorResult(
            estimator=public_method,
            status=str(estimate.status),
            estimates={} if estimate.estimate is None else {target.name: float(estimate.estimate)},
            diagnostics={
                "tuning_track": tuning_track,
                "selected_by": (estimate.diagnostics or {}).get("selected_by", tuning_track),
                **{f"calibration_{key}": value for key, value in (estimate.diagnostics or {}).items()},
            },
            runtime_sec=float(estimate.runtime_sec),
            skip_reason=str(estimate.skip_reason),
            diagnostic_only=bool(estimate.diagnostic_only or tuning_track == "oracle" or _diagnostic_method(public_method, problem.dataset.family)),
        )
        out.append(result)
    return out, candidates


def _google_dualdice_neural(problem: BenchmarkProblem, config: DifficultyStressStudyConfig) -> EstimatorResult:
    if problem.dataset.family == "streamlift":
        diagnostic_only = True
    else:
        diagnostic_only = False
    try:
        from occupancy_ratio import fit_google_dualdice_occupancy_ratio, preflight_google_dualdice
    except ModuleNotFoundError as exc:
        return EstimatorResult("google_dualdice_neural", "missing_dependency", skip_reason=f"{type(exc).__name__}: {exc}", diagnostic_only=diagnostic_only)
    preflight = preflight_google_dualdice(config.google_research_path)
    if not preflight.available:
        return EstimatorResult("google_dualdice_neural", "missing_dependency", skip_reason=preflight.reason, diagnostic_only=diagnostic_only)
    start = time.perf_counter()
    occ_data = to_occupancy_ratio_dataset(problem.dataset)
    model = fit_google_dualdice_occupancy_ratio(
        states=occ_data.states,
        actions=occ_data.actions,
        next_states=occ_data.next_states,
        target_actions=occ_data.target_actions,
        target_next_actions=occ_data.next_target_actions,
        gamma=occ_data.gamma,
        terminals=occ_data.masks == 0.0,
        initial_states=occ_data.initial_states,
        initial_actions=occ_data.initial_actions,
        initial_weights=occ_data.initial_weights,
        google_research_path=config.google_research_path,
        num_updates=20 if config.scale == "ci" else 100,
        batch_size=min(128, max(16, occ_data.states.shape[0])),
        seed=int(problem.dataset.seed),
    )
    weights = np.asarray(model.predict_state_action_ratio(occ_data.states, occ_data.actions, clip=True), dtype=np.float64)
    value = _occupancy_value(weights, occ_data.rewards, occ_data.gamma, unit_id=problem.dataset.unit_id, time=problem.dataset.time)
    target = _target(problem, value_for_streamlift=True)
    return EstimatorResult(
        "google_dualdice_neural",
        "ok",
        estimates={target.name: value},
        diagnostics={"tuning_track": "proxy", "selected_by": "google_dualdice", **_weight_diagnostics(weights), **getattr(model, "diagnostics", {})},
        runtime_sec=float(time.perf_counter() - start),
        diagnostic_only=diagnostic_only,
    )


def _google_dualdice_weighted_fqe(problem: BenchmarkProblem, config: DifficultyStressStudyConfig) -> EstimatorResult:
    diagnostic_only = problem.dataset.family == "streamlift"
    try:
        from fqe import GoogleDualDICEConfig, NeuralFQEConfig, StationaryWeightedFQEConfig, fit_stationary_weighted_fqe, preflight_google_dualdice
    except ModuleNotFoundError as exc:
        return EstimatorResult("google_dualdice_weighted_fqe", "missing_dependency", skip_reason=f"{type(exc).__name__}: {exc}", diagnostic_only=diagnostic_only)
    preflight = preflight_google_dualdice(config.google_research_path)
    if not preflight.available:
        return EstimatorResult("google_dualdice_weighted_fqe", "missing_dependency", skip_reason=preflight.reason, diagnostic_only=diagnostic_only)
    start = time.perf_counter()
    fqe_data = to_fqe_dataset(problem.dataset, target_policy_expectation_mode="sampled_action")
    occ_data = to_occupancy_ratio_dataset(problem.dataset)
    neural_cfg = NeuralFQEConfig.stable_defaults(
        hidden_dims=(16,) if config.scale == "ci" else (64, 64),
        num_iterations=2 if config.scale == "ci" else 32,
        gradient_steps_per_iteration=1 if config.scale == "ci" else 8,
        batch_size=min(128, max(16, fqe_data.states.shape[0])),
        seed=int(problem.dataset.seed) + 71,
        device="cpu",
        show_progress=False,
    )
    sw_cfg = StationaryWeightedFQEConfig(
        family="neural",
        ratio_backend="google_dualdice",
        neural_config=neural_cfg,
        google_dualdice_config=GoogleDualDICEConfig(
            google_research_path=config.google_research_path,
            num_updates=20 if config.scale == "ci" else 100,
            batch_size=min(128, max(16, fqe_data.states.shape[0])),
            seed=int(problem.dataset.seed) + 73,
        ),
    )
    model = fit_stationary_weighted_fqe(
        states=fqe_data.states,
        actions=fqe_data.actions,
        next_states=fqe_data.next_states,
        target_actions=occ_data.target_actions,
        next_actions=fqe_data.next_actions,
        rewards=fqe_data.rewards,
        gamma=fqe_data.gamma,
        terminals=fqe_data.terminals,
        sample_weight=fqe_data.sample_weight,
        initial_states=fqe_data.initial_states,
        initial_actions=fqe_data.initial_actions,
        initial_weights=occ_data.initial_weights,
        target_next_actions=occ_data.next_target_actions,
        config=sw_cfg,
    )
    target = _target(problem, value_for_streamlift=True)
    value = _estimate_fqe_model_value(model, fqe_data)
    return EstimatorResult(
        "google_dualdice_weighted_fqe",
        "ok",
        estimates={target.name: value},
        diagnostics={"tuning_track": "proxy", "selected_by": "google_dualdice_weighted_fqe", **getattr(model, "diagnostics", {})},
        runtime_sec=float(time.perf_counter() - start),
        diagnostic_only=diagnostic_only,
    )


def _dice_rl_neural(problem: BenchmarkProblem, config: DifficultyStressStudyConfig) -> EstimatorResult:
    diagnostic_only = problem.dataset.family == "streamlift"
    try:
        from occupancy_ratio_benchmark.data import BenchmarkDataset
        from occupancy_ratio_benchmark.external_baselines import DICE_RL_DUALDICE_RECOVERY_FLAGS, estimate_google_dice_rl_neural, preflight_google_dice_rl
    except ModuleNotFoundError as exc:
        return EstimatorResult("dice_rl_neural", "missing_dependency", skip_reason=f"{type(exc).__name__}: {exc}", diagnostic_only=diagnostic_only)
    preflight = preflight_google_dice_rl(config.dice_rl_repo_path)
    if not preflight.available:
        return EstimatorResult("dice_rl_neural", "missing_dependency", skip_reason=preflight.reason, diagnostic_only=diagnostic_only)
    occ_data = to_occupancy_ratio_dataset(problem.dataset)
    dataset = BenchmarkDataset(
        setting=occ_data.setting,
        states=occ_data.states,
        actions=occ_data.actions,
        next_states=occ_data.next_states,
        target_actions=occ_data.target_actions,
        next_target_actions=occ_data.next_target_actions,
        rewards=occ_data.rewards,
        true_ratio=None,
        initial_states=occ_data.initial_states,
        initial_actions=occ_data.initial_actions,
        initial_weights=occ_data.initial_weights,
        masks=occ_data.masks,
        gamma=occ_data.gamma,
        seed=occ_data.seed,
        sample_size=occ_data.sample_size,
        metadata=occ_data.metadata,
    )
    result = estimate_google_dice_rl_neural(
        dataset,
        preflight=preflight,
        num_steps=20 if config.scale == "ci" else 100,
        batch_size=min(128, max(16, occ_data.states.shape[0])),
        learning_rate=1e-4,
        hidden_dims=(16,) if config.scale == "ci" else (64, 64),
        flags=DICE_RL_DUALDICE_RECOVERY_FLAGS,
        estimator_name="dice_rl_neural",
        diagnostic_features=np.concatenate([occ_data.states, occ_data.actions], axis=1),
        value_diagnostics={},
    )
    if result.get("status") != "ok":
        status = "missing_dependency" if str(result.get("status")) == "skipped" else str(result.get("status", "error"))
        return EstimatorResult(
            "dice_rl_neural",
            status,
            skip_reason=str(result.get("skip_reason", "")),
            diagnostics=dict(result.get("diagnostics", {})),
            runtime_sec=float(result.get("runtime_sec", 0.0)),
            diagnostic_only=diagnostic_only,
        )
    weights = np.asarray(result.get("weights"), dtype=np.float64).reshape(-1)
    value = _occupancy_value(weights, occ_data.rewards, occ_data.gamma, unit_id=problem.dataset.unit_id, time=problem.dataset.time)
    target = _target(problem, value_for_streamlift=True)
    return EstimatorResult(
        "dice_rl_neural",
        "ok",
        estimates={target.name: value},
        diagnostics={"tuning_track": "proxy", "selected_by": "dice_rl_neural", **dict(result.get("diagnostics", {})), **_weight_diagnostics(weights)},
        runtime_sec=float(result.get("runtime_sec", 0.0)),
        diagnostic_only=diagnostic_only,
    )


@dataclass(frozen=True)
class _StressTarget:
    name: str
    value: float
    mc_se: float
    noise_floor: float


def _target(problem: BenchmarkProblem, *, value_for_streamlift: bool = False) -> _StressTarget:
    if problem.dataset.family == "streamlift" and value_for_streamlift:
        raw = _target_for_problem(problem)
        return _StressTarget(raw.name, raw.value, raw.mc_se, raw.noise_floor)
    if problem.dataset.family == "streamlift":
        effect_keys = sorted(key for key in problem.truth.effects if key.startswith("effect_horizon_"))
        if "effect_horizon_infinite" in problem.truth.effects:
            key = "effect_horizon_infinite"
        elif effect_keys:
            key = max(effect_keys, key=lambda item: int(item.rsplit("_", maxsplit=1)[-1]))
        else:
            raw = _target_for_problem(problem)
            return _StressTarget(raw.name, raw.value, raw.mc_se, raw.noise_floor)
        return _StressTarget(
            key,
            float(problem.truth.effects[key]),
            float(problem.truth.target_standard_errors.get(key, 0.0)),
            float(problem.truth.truth_noise_floor.get(key, 0.0)),
        )
    raw = _target_for_problem(problem)
    return _StressTarget(raw.name, raw.value, raw.mc_se, raw.noise_floor)


def _row_from_result(
    config: DifficultyStressStudyConfig,
    problem: BenchmarkProblem,
    cell: DifficultyCell,
    target_policy: str,
    requested_method: str,
    result: EstimatorResult,
) -> dict[str, Any]:
    target = _target(problem, value_for_streamlift=_diagnostic_method(requested_method, problem.dataset.family))
    estimate = result.estimates.get(target.name)
    status = result.status
    skip_reason = result.skip_reason
    if status == "ok" and estimate is None:
        status = "incomplete"
        skip_reason = f"missing target estimand {target.name}"
    row: dict[str, Any] = {
        "difficulty_schema_version": DIFFICULTY_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "scale": config.scale,
        "difficulty": cell.difficulty,
        "family": problem.dataset.family,
        "dataset": problem.dataset.name,
        "scenario": problem.dataset.scenario,
        "scenario_public": problem.dataset.scenario,
        "stress_dimension": cell.stress_dimension,
        "primary_difficulty_cell": int(cell.primary),
        "sensitivity_cell": int(cell.sensitivity),
        "time_invariant_mdp": int(not cell.scenario.nonstationarity),
        "identifiable_primary_cell": int(cell.primary and cell.scenario.confounding != "latent" and cell.scenario.overlap != "structural_gap" and not cell.scenario.nonstationarity),
        "method": requested_method,
        "estimator": result.estimator,
        "tuning_track": str(result.diagnostics.get("tuning_track", "proxy")),
        "selected_by": str(result.diagnostics.get("selected_by", result.diagnostics.get("tuning_track", "proxy"))),
        "status": status,
        "skip_reason": skip_reason,
        "diagnostic_only": int(bool(result.diagnostic_only or _diagnostic_method(requested_method, problem.dataset.family))),
        "leaderboard_eligible": 0,
        "leaderboard_result_eligible": 0,
        "gamma": float(problem.dataset.gamma),
        "seed": int(problem.dataset.seed),
        "sample_size": int(problem.dataset.metadata_public.get("sample_size", problem.dataset.n)),
        "row_count": int(problem.dataset.n),
        "target_policy": target_policy,
        "target_estimand": target.name,
        "truth_target": float(target.value),
        "truth_mc_se": float(target.mc_se),
        "truth_noise_floor": float(target.noise_floor),
        "estimate": "" if estimate is None else float(estimate),
        "runtime_sec": float(result.runtime_sec),
    }
    row.update(_scenario_diagnostics(problem))
    row.update({f"diag_{key}": value for key, value in result.diagnostics.items() if _public_scalar(value)})
    if _is_finite(estimate):
        error = float(estimate) - float(target.value)
        row["error"] = error
        row["abs_error"] = abs(error)
        normalizer = _normalizer(problem, target)  # type: ignore[arg-type]
        row["normalizer"] = float(normalizer)
        row["normalized_abs_error"] = abs(error) / max(float(normalizer), 1e-12)
        row["finite_estimate"] = 1
    else:
        row["finite_estimate"] = 0
    return row


def _problem_error_row(
    config: DifficultyStressStudyConfig,
    cell: DifficultyCell,
    family: str,
    sample_size: int,
    gamma: float,
    seed: int,
    target_policy: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "difficulty_schema_version": DIFFICULTY_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "scale": config.scale,
        "difficulty": cell.difficulty,
        "family": family,
        "dataset": "",
        "scenario": cell.scenario.name,
        "scenario_public": cell.scenario.name,
        "stress_dimension": cell.stress_dimension,
        "primary_difficulty_cell": int(cell.primary),
        "sensitivity_cell": int(cell.sensitivity),
        "time_invariant_mdp": int(not cell.scenario.nonstationarity),
        "identifiable_primary_cell": int(cell.primary and cell.scenario.confounding != "latent" and cell.scenario.overlap != "structural_gap" and not cell.scenario.nonstationarity),
        "method": "dataset",
        "estimator": "dataset",
        "tuning_track": "dataset",
        "selected_by": "dataset",
        "status": "missing_dependency" if isinstance(exc, ModuleNotFoundError) else "error",
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


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    verdicts = _hardness_verdicts(rows)
    for row in rows:
        key = (
            str(row.get("difficulty", "")),
            str(row.get("family", "")),
            str(row.get("stress_dimension", "")),
            str(row.get("method", "")),
            str(row.get("tuning_track", "")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (difficulty, family, stress_dimension, method, track), group in sorted(groups.items()):
        finite = [row for row in group if row.get("status") == "ok" and _truthy(row.get("finite_estimate"))]
        nae = [_as_float(row.get("normalized_abs_error")) for row in finite]
        out.append(
            {
                "difficulty_schema_version": DIFFICULTY_SCHEMA_VERSION,
                "difficulty": difficulty,
                "family": family,
                "stress_dimension": stress_dimension,
                "method": method,
                "tuning_track": track,
                "n_rows": len(group),
                "ok_rows": sum(1 for row in group if row.get("status") == "ok"),
                "finite_rows": len(finite),
                "finite_rate": len(finite) / max(len(group), 1),
                "median_abs_error": _nanmedian([_as_float(row.get("abs_error")) for row in finite]),
                "median_normalized_abs_error": _nanmedian(nae),
                "p90_normalized_abs_error": _nanquantile(nae, 0.90),
                "runtime_sec_mean": _nanmean([_as_float(row.get("runtime_sec")) for row in group]),
                "overlap_p5_median": _nanmedian([_as_float(row.get("diag_overlap_ratio_p5")) for row in group]),
                "target_behavior_distance_median": _nanmedian([_as_float(row.get("diag_target_behavior_policy_distance")) for row in group]),
                "hardness_verdict": verdicts.get((difficulty, family), "unclassified"),
            }
        )
    return out


def _hardness_verdicts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for difficulty in sorted({str(row.get("difficulty", "")) for row in rows}):
        threshold = {"easy": 0.10, "medium": 0.25, "hard": 0.50, "realistic": 0.25}.get(difficulty, 0.25)
        finite_target = {"easy": 0.98, "medium": 0.95, "hard": 0.90, "realistic": 0.95}.get(difficulty, 0.95)
        for family in sorted({str(row.get("family", "")) for row in rows if row.get("difficulty") == difficulty}):
            group = [
                row
                for row in rows
                if row.get("difficulty") == difficulty
                and row.get("family") == family
                and _truthy(row.get("primary_difficulty_cell"))
                and not _truthy(row.get("sensitivity_cell"))
                and not _truthy(row.get("diagnostic_only"))
                and str(row.get("tuning_track", "proxy")) == "proxy"
            ]
            if not group:
                out[(difficulty, family)] = "diagnostic_only"
                continue
            finite = [row for row in group if row.get("status") == "ok" and _truthy(row.get("finite_estimate"))]
            finite_rate = len(finite) / max(len(group), 1)
            method_best = _best_method_nae(finite)
            simple_best = _best_method_nae([row for row in finite if _simple_method(str(row.get("method", "")))])
            oracle = [
                row
                for row in rows
                if row.get("difficulty") == difficulty
                and row.get("family") == family
                and _truthy(row.get("primary_difficulty_cell"))
                and str(row.get("tuning_track")) == "oracle"
                and not _truthy(row.get("diagnostic_only"))
                and row.get("status") == "ok"
                and _truthy(row.get("finite_estimate"))
            ]
            oracle_best = _best_method_nae(oracle)
            if finite_rate < finite_target:
                verdict = "too_hard"
            elif np.isfinite(method_best) and method_best <= threshold:
                if difficulty != "easy" and np.isfinite(simple_best) and simple_best <= min(0.10, threshold * 0.50):
                    verdict = "too_easy"
                else:
                    verdict = "well_calibrated"
            elif np.isfinite(oracle_best) and oracle_best <= threshold:
                verdict = "tuning_gap"
            elif np.isfinite(oracle_best):
                verdict = "model_gap"
            else:
                verdict = "too_hard"
            out[(difficulty, family)] = verdict
    return out


def _best_method_nae(rows: list[dict[str, Any]]) -> float:
    values_by_method: dict[str, list[float]] = {}
    for row in rows:
        val = _as_float(row.get("normalized_abs_error"))
        if np.isfinite(val):
            values_by_method.setdefault(str(row.get("method", "")), []).append(val)
    medians = [_nanmedian(values) for values in values_by_method.values()]
    medians = [value for value in medians if np.isfinite(value)]
    return float(min(medians)) if medians else float("nan")


def _candidate_rows_with_context(
    rows: list[dict[str, Any]],
    config: DifficultyStressStudyConfig,
    problem: BenchmarkProblem,
    cell: DifficultyCell,
    target_policy: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        enriched = {
            "difficulty_schema_version": DIFFICULTY_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "scale": config.scale,
            "difficulty": cell.difficulty,
            "family": problem.dataset.family,
            "dataset": problem.dataset.name,
            "scenario": problem.dataset.scenario,
            "scenario_public": problem.dataset.scenario,
            "stress_dimension": cell.stress_dimension,
            "primary_difficulty_cell": int(cell.primary),
            "sensitivity_cell": int(cell.sensitivity),
            "seed": int(problem.dataset.seed),
            "sample_size": int(problem.dataset.metadata_public.get("sample_size", problem.dataset.n)),
            "gamma": float(problem.dataset.gamma),
            "target_policy": target_policy,
        }
        enriched.update(row)
        out.append(enriched)
    return out


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
        "diag_reward_scale": float(max(np.std(dataset.rewards), 1e-12)),
        "diag_average_trajectory_length": float(dataset.n / max(len(np.unique(dataset.unit_id)), 1)),
        "diag_action_entropy": _action_entropy(dataset.actions),
    }


def _action_entropy(actions: np.ndarray) -> float:
    idx = np.argmax(np.asarray(actions), axis=1)
    counts = np.bincount(idx, minlength=np.asarray(actions).shape[1]).astype(np.float64)
    probs = counts / max(float(np.sum(counts)), 1.0)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def _weight_diagnostics(weights: np.ndarray) -> dict[str, Any]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        return {}
    return {
        "ess_fraction": float((np.sum(w) ** 2) / max(w.size * np.sum(w * w), 1e-12)),
        "weight_cv": float(np.std(w) / max(abs(float(np.mean(w))), 1e-12)),
        "weight_p95": float(np.quantile(w, 0.95)),
        "weight_p99": float(np.quantile(w, 0.99)),
        "weight_max": float(np.max(w)),
    }


def _manifest(config: DifficultyStressStudyConfig) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "fqe", "torch", "occupancy-ratio", "occupancy_ratio", "tensorflow", "gym", "gymnasium", "epicare"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "difficulty_schema_version": DIFFICULTY_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "config": asdict(config),
        "difficulty_profiles": [asdict(describe_difficulty(name)) for name in config.difficulties],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "optional_dependencies": {
            "fqe": packages.get("fqe") is not None,
            "torch": packages.get("torch") is not None,
            "occupancy_ratio": packages.get("occupancy-ratio") is not None or packages.get("occupancy_ratio") is not None,
            "tensorflow": packages.get("tensorflow") is not None,
            "gym": packages.get("gym") is not None,
            "gymnasium": packages.get("gymnasium") is not None,
            "epicare": packages.get("epicare") is not None,
        },
    }


def _render_readout(
    config: DifficultyStressStudyConfig,
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
        key = (str(row.get("difficulty", "")), str(row.get("family", "")))
        verdicts[key] = str(row.get("hardness_verdict", ""))
    optional = manifest.get("optional_dependencies", {})
    optional_text = ", ".join(f"{key}={'yes' if value else 'no'}" for key, value in sorted(optional.items())) if isinstance(optional, dict) else "unavailable"
    lines = [
        "# Difficulty Stress Readout",
        "",
        f"- scale: `{config.scale}`",
        f"- result rows: `{len(rows)}`",
        f"- candidate rows: `{len(candidate_rows)}`",
        f"- failures: `{len(failures)}`",
        f"- optional dependencies: {optional_text}",
        "",
        "Primary difficulty cells are stationary, identifiable MDPs. Latent-confounding, structural no-support, and nonstationary cells are sensitivity rows and never drive hardness verdicts.",
        "",
        "## Hardness Verdicts",
        "",
        "| difficulty | family | verdict |",
        "| --- | --- | --- |",
    ]
    for (difficulty, family), verdict in sorted(verdicts.items()):
        lines.append(f"| {difficulty} | {family} | {verdict} |")
    lines.extend(["", "## Method Summary", "", "| difficulty | family | stress | method | track | finite rate | median NAE | overlap p5 | runtime |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in summary:
        lines.append(
            "| {difficulty} | {family} | {stress} | {method} | {track} | {finite} | {nae} | {overlap} | {runtime} |".format(
                difficulty=row.get("difficulty", ""),
                family=row.get("family", ""),
                stress=row.get("stress_dimension", ""),
                method=row.get("method", ""),
                track=row.get("tuning_track", ""),
                finite=_fmt(row.get("finite_rate")),
                nae=_fmt(row.get("median_normalized_abs_error")),
                overlap=_fmt(row.get("overlap_p5_median")),
                runtime=_fmt(row.get("runtime_sec_mean")),
            )
        )
    lines.extend(["", "## Status Summary", "", "| status | rows |", "| --- | ---: |"])
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "Oracle-track rows use sealed truth for post-fit candidate selection and are diagnostic-only.", ""])
    return "\n".join(lines)


def _problem_config(
    config: DifficultyStressStudyConfig,
    spec: Any,
    family: str,
    sample_size: int,
    gamma: float,
) -> CausalOPEBenchmarkConfig:
    cal = _calibration_budget(config.scale)
    return CausalOPEBenchmarkConfig(
        profile="smoke" if config.scale == "ci" else "core",
        output_root=config.output_root,
        seeds=tuple(config.seeds or seeds_for_scale(config.scale)),
        families=(family,),  # type: ignore[list-item]
        sample_sizes=(int(sample_size),),
        gammas=(float(gamma),),
        observed_horizons=tuple(int(h) for h in spec.streamlift_observed_horizons),
        target_policies=tuple(config.target_policies or spec.target_policies),
        estimators=(),
        trajectory_horizon=int(spec.trajectory_horizon),
        streamlift_long_horizon=36,
        streamlift_include_infinite_horizon=bool(spec.streamlift_include_infinite_horizon),
        streamlift_infinite_horizon_max_steps=240,
        mc_truth_rollouts=int(config.mc_truth_rollouts or cal["mc_truth_rollouts"]),
        fqe_hidden_dims=tuple(cal["fqe_hidden_dims"]),
        fqe_num_iterations=int(cal["fqe_num_iterations"]),
        fqe_gradient_steps_per_iteration=int(cal["fqe_gradient_steps_per_iteration"]),
        fqe_batch_size=int(cal["fqe_batch_size"]),
        automl_tuning="fast" if config.scale == "ci" else "balanced",
        fail_fast=bool(config.fail_fast),
    )


def _problem_config_from_problem(config: DifficultyStressStudyConfig, problem: BenchmarkProblem) -> CausalOPEBenchmarkConfig:
    spec = describe_difficulty("medium")
    return _problem_config(config, spec, problem.dataset.family, int(problem.dataset.metadata_public.get("sample_size", problem.dataset.n)), float(problem.dataset.gamma))


def _calibration_config(config: DifficultyStressStudyConfig, problem: BenchmarkProblem) -> CalibrationStudyConfig:
    budget = _calibration_budget(config.scale)
    return CalibrationStudyConfig(
        preset="smoke" if config.scale == "ci" else "core-lite",
        output_root=config.output_root,
        families=(problem.dataset.family,),  # type: ignore[list-item]
        seeds=(int(problem.dataset.seed),),
        sample_sizes=(int(problem.dataset.metadata_public.get("sample_size", problem.dataset.n)),),
        gammas=(float(problem.dataset.gamma),),
        target_policies=(str(problem.dataset.metadata_public.get("target_policy", "moderate")),),
        estimators=("neural_fqe", "neural_occupancy"),
        tuning_tracks=("proxy", "oracle"),
        trajectory_horizon=int(problem.dataset.metadata_public.get("trajectory_horizon", 24)),
        streamlift_observed_horizon=int(problem.dataset.metadata_public.get("observed_horizon", 3)),
        streamlift_long_horizon=int(problem.dataset.metadata_public.get("long_horizon", 36)),
        mc_truth_rollouts=int(config.mc_truth_rollouts or budget["mc_truth_rollouts"]),
        cv_folds=2 if config.scale == "ci" else 3,
        fqe_budget="fast" if config.scale == "ci" else "balanced",
        fqe_hidden_dims=tuple(budget["fqe_hidden_dims"]),
        fqe_num_iterations=int(budget["fqe_num_iterations"]),
        fqe_gradient_steps_per_iteration=int(budget["fqe_gradient_steps_per_iteration"]),
        fqe_batch_size=int(budget["fqe_batch_size"]),
        fqe_max_candidates=2 if config.scale == "ci" else 10,
        fqe_promotion_candidates=1 if config.scale == "ci" else 4,
        occupancy_budget="fast" if config.scale == "ci" else "balanced",
        occupancy_hidden_dims=tuple(budget["occupancy_hidden_dims"]),
        occupancy_num_iterations=int(budget["occupancy_num_iterations"]),
        occupancy_gradient_steps_per_iteration=int(budget["occupancy_gradient_steps_per_iteration"]),
        occupancy_nuisance_max_steps=int(budget["occupancy_nuisance_max_steps"]),
        occupancy_mcmc_samples=int(budget["occupancy_mcmc_samples"]),
        occupancy_batch_size=int(budget["occupancy_batch_size"]),
        occupancy_max_candidates=1 if config.scale == "ci" else 8,
        occupancy_promotion_candidates=1 if config.scale == "ci" else 3,
        occupancy_score_method="legacy_rank" if config.scale == "ci" else "bellman_gmm",
    )


def _calibration_budget(scale: str) -> dict[str, Any]:
    if scale == "ci":
        return {
            "mc_truth_rollouts": 8,
            "fqe_hidden_dims": (16,),
            "fqe_num_iterations": 2,
            "fqe_gradient_steps_per_iteration": 1,
            "fqe_batch_size": 32,
            "occupancy_hidden_dims": (16,),
            "occupancy_num_iterations": 2,
            "occupancy_gradient_steps_per_iteration": 1,
            "occupancy_nuisance_max_steps": 2,
            "occupancy_mcmc_samples": 2,
            "occupancy_batch_size": 32,
        }
    if scale == "audit":
        return {
            "mc_truth_rollouts": 96,
            "fqe_hidden_dims": (64, 64),
            "fqe_num_iterations": 48,
            "fqe_gradient_steps_per_iteration": 15,
            "fqe_batch_size": 256,
            "occupancy_hidden_dims": (64, 64),
            "occupancy_num_iterations": 30,
            "occupancy_gradient_steps_per_iteration": 4,
            "occupancy_nuisance_max_steps": 400,
            "occupancy_mcmc_samples": 24,
            "occupancy_batch_size": 256,
        }
    return {
        "mc_truth_rollouts": 192,
        "fqe_hidden_dims": (128, 128),
        "fqe_num_iterations": 80,
        "fqe_gradient_steps_per_iteration": 20,
        "fqe_batch_size": 512,
        "occupancy_hidden_dims": (128, 128),
        "occupancy_num_iterations": 60,
        "occupancy_gradient_steps_per_iteration": 6,
        "occupancy_nuisance_max_steps": 600,
        "occupancy_mcmc_samples": 48,
        "occupancy_batch_size": 512,
    }


def _target_policies(config: DifficultyStressStudyConfig, difficulty: str, family: str) -> tuple[str, ...]:
    requested = tuple(config.target_policies or target_policies_for_difficulty(difficulty, family))  # type: ignore[arg-type]
    allowed = set(POLICY_NAMES_BY_FAMILY.get(family, ()))
    return tuple(policy for policy in requested if policy in allowed)


def _methods_for_family(config: DifficultyStressStudyConfig, family: str) -> tuple[str, ...]:
    requested = tuple(str(method) for method in config.methods)
    if family == "streamlift":
        streamlift = (
            "naive_short_term",
            "streamlift_stratified_gcomp",
            "direct_method",
            "ipw",
            "snipw",
            "doubly_robust",
            "neural_fqe_auto",
            "discounted_occupancy_neural_auto",
            "google_dualdice_neural",
        )
        return tuple(method for method in streamlift if method in requested)
    if family == "clinic_dtr":
        return tuple(method for method in requested if method not in {"naive_short_term", "streamlift_stratified_gcomp"}) + (("ipcw_rmst",) if "ipcw_rmst" not in requested else ())
    return tuple(method for method in requested if method not in {"naive_short_term", "streamlift_stratified_gcomp", "ipcw_rmst"})


def _observed_horizon(spec: Any, seed: int) -> int:
    horizons = tuple(int(h) for h in spec.streamlift_observed_horizons)
    return horizons[int(seed) % len(horizons)]


def _diagnostic_method(method: str, family: str) -> bool:
    if family == "streamlift" and ("fqe" in method or "occupancy" in method or "dualdice" in method or "dice" in method):
        return True
    return method in {"google_dualdice_neural", "google_dualdice_weighted_fqe", "dice_rl_neural"} and family == "streamlift"


def _simple_method(method: str) -> bool:
    return method in {"naive_short_term", "direct_method", "ipw", "snipw", "doubly_robust", "ipcw_rmst"}


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
