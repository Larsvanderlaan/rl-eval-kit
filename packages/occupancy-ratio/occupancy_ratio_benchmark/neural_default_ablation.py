from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.conservatism_audit import build_conservatism_audit_rows
from occupancy_ratio_benchmark.io import write_csv, write_json
from occupancy_ratio_benchmark.runner import run_benchmark


CONTROLLED_SETTINGS = {"discrete_chain", "discrete_grid", "linear_gaussian", "nonlinear_monte_carlo"}
GYM_SETTINGS = {"gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper"}
DEFAULT_ESTIMATORS = (
    "oracle",
    "boosted_tree_stable",
    "neural_network_stable",
    "neural_network_google_parity",
    "google_dualdice_neural",
)


@dataclass(frozen=True)
class NeuralDefaultCandidate:
    candidate_id: str
    neural_hidden_dims: tuple[int, ...]
    neural_activation: str
    neural_num_iterations: int
    neural_gradient_steps_per_iteration: int
    neural_action_steps: int
    neural_source_steps: int
    neural_transition_steps: int
    neural_direct_one_step_steps: int
    neural_direct_adjoint_steps: int


@dataclass(frozen=True)
class MatrixSpec:
    matrix_id: str
    settings: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    gammas: tuple[float, ...]
    seeds: tuple[int, ...]
    discrete_policy_shifts: tuple[float, ...] = ()
    linear_gaussian_policy_shifts: tuple[float, ...] = (1.0,)
    gym_target_value_rollouts: int = 64
    google_num_updates: int = 1_000
    google_batch_size: int = 128


@dataclass(frozen=True)
class NeuralDefaultAblationResult:
    output_root: Path
    config_paths: tuple[Path, ...]
    results_path: Path
    audit_path: Path
    summary_path: Path
    report_path: Path


CANDIDATES: tuple[NeuralDefaultCandidate, ...] = (
    NeuralDefaultCandidate(
        candidate_id="legacy_low_budget",
        neural_hidden_dims=(64, 64),
        neural_activation="silu",
        neural_num_iterations=30,
        neural_gradient_steps_per_iteration=4,
        neural_action_steps=400,
        neural_source_steps=400,
        neural_transition_steps=600,
        neural_direct_one_step_steps=400,
        neural_direct_adjoint_steps=32,
    ),
    NeuralDefaultCandidate(
        candidate_id="adjoint_only",
        neural_hidden_dims=(64, 64),
        neural_activation="silu",
        neural_num_iterations=30,
        neural_gradient_steps_per_iteration=4,
        neural_action_steps=400,
        neural_source_steps=400,
        neural_transition_steps=600,
        neural_direct_one_step_steps=400,
        neural_direct_adjoint_steps=128,
    ),
    NeuralDefaultCandidate(
        candidate_id="stage_budget",
        neural_hidden_dims=(64, 64),
        neural_activation="silu",
        neural_num_iterations=60,
        neural_gradient_steps_per_iteration=6,
        neural_action_steps=800,
        neural_source_steps=800,
        neural_transition_steps=1_000,
        neural_direct_one_step_steps=1_000,
        neural_direct_adjoint_steps=128,
    ),
    NeuralDefaultCandidate(
        candidate_id="stage_budget_long",
        neural_hidden_dims=(64, 64),
        neural_activation="silu",
        neural_num_iterations=80,
        neural_gradient_steps_per_iteration=8,
        neural_action_steps=1_000,
        neural_source_steps=1_000,
        neural_transition_steps=1_400,
        neural_direct_one_step_steps=1_400,
        neural_direct_adjoint_steps=128,
    ),
    NeuralDefaultCandidate(
        candidate_id="wide_relu_low_budget",
        neural_hidden_dims=(256, 256),
        neural_activation="relu",
        neural_num_iterations=30,
        neural_gradient_steps_per_iteration=4,
        neural_action_steps=400,
        neural_source_steps=400,
        neural_transition_steps=600,
        neural_direct_one_step_steps=400,
        neural_direct_adjoint_steps=32,
    ),
    NeuralDefaultCandidate(
        candidate_id="wide_relu_stage_budget",
        neural_hidden_dims=(256, 256),
        neural_activation="relu",
        neural_num_iterations=60,
        neural_gradient_steps_per_iteration=6,
        neural_action_steps=800,
        neural_source_steps=800,
        neural_transition_steps=1_000,
        neural_direct_one_step_steps=1_000,
        neural_direct_adjoint_steps=128,
    ),
)


def full_matrix_specs() -> tuple[MatrixSpec, ...]:
    seeds = tuple(range(5))
    return (
        MatrixSpec(
            matrix_id="tabular",
            settings=("discrete_chain", "discrete_grid"),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.99),
            seeds=seeds,
            discrete_policy_shifts=(0.0, 0.35, 0.65, 1.0, 1.5),
        ),
        MatrixSpec(
            matrix_id="gaussian",
            settings=("linear_gaussian",),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.95, 0.99),
            seeds=seeds,
            linear_gaussian_policy_shifts=(0.0, 0.25, 0.5, 1.0, 2.0, 3.0),
        ),
        MatrixSpec(
            matrix_id="gym",
            settings=("gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper"),
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.95, 0.99),
            seeds=seeds,
            gym_target_value_rollouts=64,
            google_num_updates=1_000,
            google_batch_size=128,
        ),
    )


def smoke_matrix_specs() -> tuple[MatrixSpec, ...]:
    return (
        MatrixSpec(
            matrix_id="smoke",
            settings=("discrete_chain", "linear_gaussian", "gym_pendulum"),
            sample_sizes=(300,),
            gammas=(0.9,),
            seeds=(0,),
            discrete_policy_shifts=(0.65,),
            linear_gaussian_policy_shifts=(1.0,),
            gym_target_value_rollouts=2,
            google_num_updates=10,
            google_batch_size=64,
        ),
    )


def run_neural_default_ablation(
    *,
    output_root: str | Path = "outputs/neural_default_ablation",
    matrix: str = "full",
    candidate_ids: Sequence[str] | None = None,
    include_google_dualdice: bool = True,
    external_repo_path: str | Path = "/tmp/google-research",
    resume: bool = True,
    write_plots: bool = False,
    estimator_timeout_sec: float | None = 900.0,
) -> NeuralDefaultAblationResult:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidates = _select_candidates(candidate_ids)
    specs = smoke_matrix_specs() if matrix == "smoke" else full_matrix_specs()

    config_paths: list[Path] = []
    merged_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for spec in specs:
            config = _make_config(
                candidate=candidate,
                spec=spec,
                output_root=root,
                include_google_dualdice=include_google_dualdice,
                external_repo_path=external_repo_path,
                resume=resume,
                write_plots=write_plots,
                estimator_timeout_sec=estimator_timeout_sec,
            )
            config_path = _write_config(root, candidate, spec, config)
            config_paths.append(config_path)
            result = run_benchmark(config)
            merged_rows.extend(
                _tag_rows(
                    result.rows,
                    candidate=candidate,
                    matrix_id=spec.matrix_id,
                )
            )

    results_path = root / "neural_default_ablation_results.csv"
    audit_path = root / "neural_default_ablation_conservatism_audit.csv"
    summary_path = root / "neural_default_ablation_summary.csv"
    report_path = root / "neural_default_ablation_report.md"
    audit_rows = build_conservatism_audit_rows(merged_rows)
    summary_rows = summarize_ablation_rows(merged_rows, audit_rows)

    write_csv(results_path, merged_rows)
    write_csv(audit_path, audit_rows)
    write_csv(summary_path, summary_rows)
    report_path.write_text(render_ablation_report(summary_rows), encoding="utf-8")
    return NeuralDefaultAblationResult(
        output_root=root,
        config_paths=tuple(config_paths),
        results_path=results_path,
        audit_path=audit_path,
        summary_path=summary_path,
        report_path=report_path,
    )


def summarize_ablation_rows(
    rows: Iterable[dict[str, Any]],
    audit_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_list = [dict(row) for row in rows]
    audit_counts = _audit_counts(audit_rows or build_conservatism_audit_rows(rows_list))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_list:
        groups.setdefault((str(row.get("candidate_id", "")), str(row.get("estimator", ""))), []).append(row)

    summary: list[dict[str, Any]] = []
    for (candidate_id, estimator), group in sorted(groups.items()):
        ok_rows = [row for row in group if row.get("status") == "ok"]
        controlled = [row for row in ok_rows if str(row.get("setting", "")) in CONTROLLED_SETTINGS]
        gym = [row for row in ok_rows if str(row.get("setting", "")) in GYM_SETTINGS]
        collapse_count = sum(int(_is_collapse(row)) for row in controlled)
        audit_key = (candidate_id, estimator)
        summary.append(
            {
                "candidate_id": candidate_id,
                "estimator": estimator,
                "row_count": len(group),
                "ok_count": len(ok_rows),
                "error_count": sum(int(row.get("status") == "error") for row in group),
                "skipped_count": sum(int(row.get("status") == "skipped") for row in group),
                "audit_fail_count": audit_counts.get(audit_key, {}).get("fail", 0),
                "audit_warn_count": audit_counts.get(audit_key, {}).get("warn", 0),
                "controlled_audit_fail_count": audit_counts.get(audit_key, {}).get("controlled_fail", 0),
                "controlled_rows": len(controlled),
                "controlled_score_median": _median(_controlled_score(row) for row in controlled),
                "controlled_ratio_normalized_l1_median": _median(
                    _to_float(row.get("ratio_normalized_l1")) for row in controlled
                ),
                "controlled_log_ratio_rmse_median": _median(_to_float(row.get("log_ratio_rmse")) for row in controlled),
                "collapse_count": collapse_count,
                "gym_rows": len(gym),
                "gym_ope_se_units_mean": _mean(_to_float(row.get("ope_value_abs_error_se_units")) for row in gym),
                "gym_ope_se_units_median": _median(_to_float(row.get("ope_value_abs_error_se_units")) for row in gym),
                "gym_ope_abs_error_mean": _mean(_to_float(row.get("ope_value_abs_error")) for row in gym),
                "gym_ope_abs_error_median": _median(_to_float(row.get("ope_value_abs_error")) for row in gym),
                "runtime_sec_mean": _mean(_to_float(row.get("runtime_sec")) for row in ok_rows),
                "runtime_sec_median": _median(_to_float(row.get("runtime_sec")) for row in ok_rows),
                "runtime_sec_max": _max(_to_float(row.get("runtime_sec")) for row in ok_rows),
                "ess_fraction_median": _median(_to_float(row.get("effective_sample_size_fraction")) for row in ok_rows),
                "weight_cv_median": _median(_to_float(row.get("weight_cv")) for row in ok_rows),
            }
        )
    return summary


def render_ablation_report(summary_rows: Sequence[dict[str, Any]]) -> str:
    decision = evaluate_stage_budget_promotion(summary_rows)
    lines = [
        "# Neural Stable Default Ablation",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "## Promotion Gates",
        "",
        "| gate | status | detail |",
        "|---|---:|---|",
    ]
    for gate in decision["gates"]:
        lines.append(
            "| {name} | {status} | {detail} |".format(
                name=gate["name"],
                status="pass" if gate["passed"] else "fail",
                detail=str(gate["detail"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Neural Stable Candidates",
            "",
            "| candidate | ok | controlled score median | gym SE mean | runtime median | audit fail | collapse |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _stable_summary_rows(summary_rows):
        lines.append(
            "| {candidate} | {ok} | {controlled} | {gym} | {runtime} | {fail} | {collapse} |".format(
                candidate=row.get("candidate_id", ""),
                ok=row.get("ok_count", 0),
                controlled=_fmt(row.get("controlled_score_median")),
                gym=_fmt(row.get("gym_ope_se_units_mean")),
                runtime=_fmt(row.get("runtime_sec_median")),
                fail=row.get("audit_fail_count", 0),
                collapse=row.get("collapse_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Wide ReLU Gate",
            "",
            decision["wide_relu_detail"],
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_stage_budget_promotion(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stable = {str(row.get("candidate_id")): row for row in _stable_summary_rows(summary_rows)}
    stage = stable.get("stage_budget")
    legacy = stable.get("legacy_low_budget")
    adjoint = stable.get("adjoint_only")
    google = _summary_row(summary_rows, "stage_budget", "google_dualdice_neural")
    wide = stable.get("wide_relu_stage_budget")

    gates = []
    stage_missing = stage is None
    gates.append(
        _gate(
            "controlled audit",
            not stage_missing and int(stage.get("controlled_audit_fail_count", 0)) == 0,
            (
                "stage_budget summary missing"
                if stage_missing
                else f"controlled audit failures={int(stage.get('controlled_audit_fail_count', 0))}"
            ),
        )
    )
    stage_score = _summary_float(stage, "controlled_score_median")
    legacy_score = _summary_float(legacy, "controlled_score_median")
    adjoint_score = _summary_float(adjoint, "controlled_score_median")
    gates.append(
        _gate(
            "controlled score",
            np.isfinite(stage_score)
            and np.isfinite(legacy_score)
            and np.isfinite(adjoint_score)
            and stage_score < legacy_score
            and stage_score < adjoint_score,
            f"stage={_fmt(stage_score)}, legacy={_fmt(legacy_score)}, adjoint={_fmt(adjoint_score)}",
        )
    )
    stage_gym_mean = _summary_float(stage, "gym_ope_se_units_mean")
    stage_gym_median = _summary_float(stage, "gym_ope_se_units_median")
    google_gym_mean = _summary_float(google, "gym_ope_se_units_mean")
    google_gym_median = _summary_float(google, "gym_ope_se_units_median")
    gates.append(
        _gate(
            "gym vs google",
            np.isfinite(stage_gym_mean)
            and np.isfinite(stage_gym_median)
            and np.isfinite(google_gym_mean)
            and np.isfinite(google_gym_median)
            and stage_gym_mean <= 1.10 * google_gym_mean
            and stage_gym_median <= google_gym_median,
            (
                f"stage mean/median={_fmt(stage_gym_mean)}/{_fmt(stage_gym_median)}, "
                f"google mean/median={_fmt(google_gym_mean)}/{_fmt(google_gym_median)}"
            ),
        )
    )
    gates.append(
        _gate(
            "collapse",
            not stage_missing and int(stage.get("collapse_count", 0)) == 0,
            (
                "stage_budget summary missing"
                if stage_missing
                else f"near-uniform-collapse rows={int(stage.get('collapse_count', 0))}"
            ),
        )
    )
    stage_runtime = _summary_float(stage, "runtime_sec_median")
    stage_runtime_max = _summary_float(stage, "runtime_sec_max")
    legacy_runtime = _summary_float(legacy, "runtime_sec_median")
    gates.append(
        _gate(
            "runtime",
            np.isfinite(stage_runtime)
            and np.isfinite(stage_runtime_max)
            and np.isfinite(legacy_runtime)
            and stage_runtime <= 2.0 * legacy_runtime
            and stage_runtime_max < 900.0,
            f"stage median/max={_fmt(stage_runtime)}/{_fmt(stage_runtime_max)}, legacy median={_fmt(legacy_runtime)}",
        )
    )

    wide_detail = _wide_relu_detail(stage, wide)
    promote = all(bool(gate["passed"]) for gate in gates)
    return {
        "decision": "promote_stage_budget" if promote else "keep_current_defaults",
        "gates": gates,
        "wide_relu_detail": wide_detail,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the neural stable-default ablation benchmark.")
    parser.add_argument("--output-root", default="outputs/neural_default_ablation")
    parser.add_argument("--matrix", choices=("full", "smoke"), default="full")
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    parser.add_argument("--external-repo-path", default="/tmp/google-research")
    parser.add_argument("--no-google-dualdice", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--estimator-timeout-sec", type=float, default=900.0)
    args = parser.parse_args()
    result = run_neural_default_ablation(
        output_root=args.output_root,
        matrix=args.matrix,
        candidate_ids=args.candidate_ids,
        include_google_dualdice=not args.no_google_dualdice,
        external_repo_path=args.external_repo_path,
        resume=not args.no_resume,
        write_plots=bool(args.write_plots),
        estimator_timeout_sec=args.estimator_timeout_sec,
    )
    print(f"Wrote merged results: {result.results_path}")
    print(f"Wrote audit: {result.audit_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote report: {result.report_path}")


def _make_config(
    *,
    candidate: NeuralDefaultCandidate,
    spec: MatrixSpec,
    output_root: Path,
    include_google_dualdice: bool,
    external_repo_path: str | Path,
    resume: bool,
    write_plots: bool,
    estimator_timeout_sec: float | None,
) -> OccupancyRatioBenchmarkConfig:
    profile = "smoke" if spec.matrix_id == "smoke" else "high_stakes"
    timeout = 120.0 if spec.matrix_id == "smoke" else estimator_timeout_sec
    return OccupancyRatioBenchmarkConfig(
        stage=profile,
        profile=profile,
        output_root=output_root / "runs" / candidate.candidate_id / spec.matrix_id,
        external_repo_path=Path(external_repo_path),
        settings=spec.settings,
        estimators=DEFAULT_ESTIMATORS,
        seeds=spec.seeds,
        sample_sizes=spec.sample_sizes,
        gammas=spec.gammas,
        discrete_policy_shifts=spec.discrete_policy_shifts,
        linear_gaussian_policy_shifts=spec.linear_gaussian_policy_shifts,
        boosted_estimator_presets=("stable",),
        neural_estimator_presets=("stable", "google_parity"),
        include_google_dual_dice=include_google_dualdice,
        boosted_num_iterations=80,
        boosted_mcmc_samples=48,
        boosted_batch_size=512,
        boosted_density_ratio_loss="lsif",
        boosted_fixed_point_damping=0.5,
        boosted_moment_calibration="scalar",
        neural_num_iterations=candidate.neural_num_iterations,
        neural_gradient_steps_per_iteration=candidate.neural_gradient_steps_per_iteration,
        neural_mcmc_samples=24,
        neural_batch_size=512,
        neural_hidden_dims=candidate.neural_hidden_dims,
        neural_activation=candidate.neural_activation,
        neural_action_steps=candidate.neural_action_steps,
        neural_source_steps=candidate.neural_source_steps,
        neural_transition_steps=candidate.neural_transition_steps,
        neural_direct_one_step_steps=candidate.neural_direct_one_step_steps,
        neural_direct_adjoint_steps=candidate.neural_direct_adjoint_steps,
        neural_density_ratio_loss="lsif",
        neural_fixed_point_damping=0.5,
        neural_moment_calibration="scalar",
        neural_nuisance_prediction_max=50.0,
        neural_occupancy_ratio_max=50.0,
        google_num_updates=spec.google_num_updates,
        google_batch_size=spec.google_batch_size,
        mc_truth_samples=50_000 if spec.matrix_id != "smoke" else 2_000,
        gym_target_value_rollouts=spec.gym_target_value_rollouts,
        source_state_correction_mode="auto",
        estimator_timeout_sec=timeout,
        resume=resume,
        write_plots=write_plots,
    )


def _write_config(
    output_root: Path,
    candidate: NeuralDefaultCandidate,
    spec: MatrixSpec,
    config: OccupancyRatioBenchmarkConfig,
) -> Path:
    path = output_root / "configs" / f"{candidate.candidate_id}_{spec.matrix_id}.json"
    payload = {
        "candidate_id": candidate.candidate_id,
        "matrix_id": spec.matrix_id,
        "overrides": _jsonable_config(config),
    }
    write_json(path, payload)
    return path


def _jsonable_config(config: OccupancyRatioBenchmarkConfig) -> dict[str, Any]:
    raw = asdict(config)
    raw.pop("config_path", None)
    raw.pop("config_sha256", None)
    return {key: _jsonable(value) for key, value in raw.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _select_candidates(candidate_ids: Sequence[str] | None) -> tuple[NeuralDefaultCandidate, ...]:
    if candidate_ids is None:
        return CANDIDATES
    wanted = {str(candidate_id) for candidate_id in candidate_ids}
    known = {candidate.candidate_id for candidate in CANDIDATES}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown candidate id(s): {', '.join(unknown)}")
    return tuple(candidate for candidate in CANDIDATES if candidate.candidate_id in wanted)


def _tag_rows(
    rows: Iterable[dict[str, Any]],
    *,
    candidate: NeuralDefaultCandidate,
    matrix_id: str,
) -> list[dict[str, Any]]:
    tags = {
        "candidate_id": candidate.candidate_id,
        "matrix_id": matrix_id,
        "candidate_neural_hidden_dims": "x".join(str(width) for width in candidate.neural_hidden_dims),
        "candidate_neural_activation": candidate.neural_activation,
        "candidate_neural_num_iterations": candidate.neural_num_iterations,
        "candidate_neural_gradient_steps_per_iteration": candidate.neural_gradient_steps_per_iteration,
        "candidate_neural_action_steps": candidate.neural_action_steps,
        "candidate_neural_source_steps": candidate.neural_source_steps,
        "candidate_neural_transition_steps": candidate.neural_transition_steps,
        "candidate_neural_direct_one_step_steps": candidate.neural_direct_one_step_steps,
        "candidate_neural_direct_adjoint_steps": candidate.neural_direct_adjoint_steps,
    }
    return [{**dict(row), **tags} for row in rows]


def _audit_counts(audit_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in audit_rows:
        key = (str(row.get("candidate_id", "")), str(row.get("estimator", "")))
        bucket = counts.setdefault(key, {"fail": 0, "warn": 0, "controlled_fail": 0})
        status = str(row.get("audit_status", "")).lower()
        setting = str(row.get("setting", ""))
        if status == "fail":
            bucket["fail"] += 1
            if setting in CONTROLLED_SETTINGS:
                bucket["controlled_fail"] += 1
        elif status == "warn":
            bucket["warn"] += 1
    return counts


def _controlled_score(row: dict[str, Any]) -> float:
    ratio_l1 = _to_float(row.get("ratio_normalized_l1"))
    if not np.isfinite(ratio_l1):
        ratio_l1 = _to_float(row.get("ratio_rel_mse"))
    if not np.isfinite(ratio_l1):
        ratio_l1 = _to_float(row.get("ratio_l1"))
    if not np.isfinite(ratio_l1):
        return np.nan
    log_rmse = _to_float(row.get("log_ratio_rmse"), 0.0)
    ope_abs_error = _to_float(row.get("ope_value_abs_error"), 0.0)
    return float(ratio_l1 + 0.25 * log_rmse + 0.10 * ope_abs_error)


def _is_collapse(row: dict[str, Any]) -> bool:
    ess = _to_float(row.get("effective_sample_size_fraction"))
    true_ess = _to_float(row.get("true_effective_sample_size_fraction"))
    cv = _to_float(row.get("weight_cv"))
    true_cv = _to_float(row.get("true_weight_cv"))
    return bool(
        np.isfinite(ess)
        and np.isfinite(true_ess)
        and np.isfinite(cv)
        and np.isfinite(true_cv)
        and ess > 0.95
        and true_ess < 0.80
        and cv < 0.05
        and true_cv > 0.05
    )


def _stable_summary_rows(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in summary_rows if row.get("estimator") == "neural_network_stable"]


def _summary_row(
    summary_rows: Sequence[dict[str, Any]],
    candidate_id: str,
    estimator: str,
) -> dict[str, Any] | None:
    for row in summary_rows:
        if row.get("candidate_id") == candidate_id and row.get("estimator") == estimator:
            return dict(row)
    return None


def _summary_float(row: dict[str, Any] | None, key: str) -> float:
    return np.nan if row is None else _to_float(row.get(key))


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _wide_relu_detail(stage: dict[str, Any] | None, wide: dict[str, Any] | None) -> str:
    if stage is None or wide is None:
        return "Wide ReLU gate cannot be evaluated because one of the candidate summaries is missing."
    stage_controlled = _summary_float(stage, "controlled_score_median")
    stage_gym = _summary_float(stage, "gym_ope_se_units_mean")
    wide_controlled = _summary_float(wide, "controlled_score_median")
    wide_gym = _summary_float(wide, "gym_ope_se_units_mean")
    if (
        np.isfinite(stage_controlled)
        and np.isfinite(stage_gym)
        and np.isfinite(wide_controlled)
        and np.isfinite(wide_gym)
        and wide_controlled < stage_controlled
        and wide_gym < stage_gym
    ):
        return "Wide ReLU beats 64x64 SiLU on controlled score and Gym OPE; consider promoting architecture."
    return (
        "Keep 64x64 SiLU unless a later full run shows wide ReLU wins both gates "
        f"(stage controlled/Gym={_fmt(stage_controlled)}/{_fmt(stage_gym)}, "
        f"wide controlled/Gym={_fmt(wide_controlled)}/{_fmt(wide_gym)})."
    )


def _to_float(value: Any, default: float = np.nan) -> float:
    if value in ("", None):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    return arr


def _mean(values: Iterable[float]) -> float:
    arr = _finite_values(values)
    return float(np.mean(arr)) if arr.size else np.nan


def _median(values: Iterable[float]) -> float:
    arr = _finite_values(values)
    return float(np.median(arr)) if arr.size else np.nan


def _max(values: Iterable[float]) -> float:
    arr = _finite_values(values)
    return float(np.max(arr)) if arr.size else np.nan


def _fmt(value: Any) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4g}"


if __name__ == "__main__":
    main()
