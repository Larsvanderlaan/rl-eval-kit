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


CONTROLLED_SETTINGS = {"discrete_chain", "discrete_grid", "linear_gaussian"}
GYM_SETTINGS = {"gym_pendulum", "gym_mountain_car_continuous", "gym_halfcheetah", "gym_hopper"}
TIE_REL_TOL = 0.02


@dataclass(frozen=True)
class MomentEvaluatorCandidate:
    evaluator_id: str
    extra_blocks: tuple[str, ...]
    description: str


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


@dataclass(frozen=True)
class MomentEvaluatorAblationResult:
    output_root: Path
    config_paths: tuple[Path, ...]
    results_path: Path
    tuning_path: Path
    audit_path: Path
    summary_path: Path
    delta_path: Path
    report_path: Path


EVALUATORS: tuple[MomentEvaluatorCandidate, ...] = (
    MomentEvaluatorCandidate("current", (), "Current held-out moment evaluator."),
    MomentEvaluatorCandidate("second_order", ("second_order",), "Adds squared PCA-geometry moments."),
    MomentEvaluatorCandidate("multiscale_rff", ("multiscale_rff",), "Adds shared multi-scale RFF moments."),
    MomentEvaluatorCandidate("support", ("support",), "Adds train-fold support/radius moments and strata."),
    MomentEvaluatorCandidate("policy_shift", ("policy_shift",), "Adds shared policy-shift proxy moments and strata."),
    MomentEvaluatorCandidate(
        "robust_core",
        ("multiscale_rff", "support", "policy_shift"),
        "Combines the main nonlinear, support, and policy-shift blocks.",
    ),
    MomentEvaluatorCandidate(
        "robust_all",
        ("second_order", "multiscale_rff", "support", "policy_shift"),
        "Combines all experimental blocks.",
    ),
)


def smoke_matrix_specs() -> tuple[MatrixSpec, ...]:
    return (
        MatrixSpec(
            matrix_id="smoke",
            settings=("discrete_chain", "discrete_grid", "linear_gaussian", "gym_pendulum"),
            sample_sizes=(300,),
            gammas=(0.9,),
            seeds=(0,),
            discrete_policy_shifts=(0.65,),
            linear_gaussian_policy_shifts=(1.0,),
            gym_target_value_rollouts=2,
        ),
    )


def reduced_matrix_specs() -> tuple[MatrixSpec, ...]:
    return (
        MatrixSpec(
            matrix_id="tabular_reduced",
            settings=("discrete_chain", "discrete_grid"),
            sample_sizes=(300,),
            gammas=(0.9,),
            seeds=(0,),
            discrete_policy_shifts=(0.65, 1.5),
        ),
        MatrixSpec(
            matrix_id="gaussian_reduced",
            settings=("linear_gaussian",),
            sample_sizes=(300,),
            gammas=(0.9,),
            seeds=(0,),
            linear_gaussian_policy_shifts=(1.0, 3.0),
        ),
        MatrixSpec(
            matrix_id="gym_reduced",
            settings=("gym_pendulum", "gym_mountain_car_continuous"),
            sample_sizes=(300,),
            gammas=(0.9,),
            seeds=(0,),
            gym_target_value_rollouts=4,
        ),
    )


def full_matrix_specs(*, include_large_gym: bool = True) -> tuple[MatrixSpec, ...]:
    seeds = tuple(range(5))
    gym_settings = ("gym_pendulum", "gym_mountain_car_continuous")
    if include_large_gym:
        gym_settings = (*gym_settings, "gym_halfcheetah", "gym_hopper")
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
            settings=gym_settings,
            sample_sizes=(1_000, 5_000),
            gammas=(0.9, 0.95, 0.99),
            seeds=seeds,
            gym_target_value_rollouts=64,
        ),
    )


def run_moment_evaluator_ablation(
    *,
    output_root: str | Path = "outputs/moment_evaluator_ablation",
    matrix: str = "smoke",
    evaluator_ids: Sequence[str] | None = None,
    automl_tuning: str = "balanced",
    include_large_gym: bool = True,
    external_repo_path: str | Path = "/tmp/google-research",
    resume: bool = True,
    write_plots: bool = False,
    estimator_timeout_sec: float | None = None,
) -> MomentEvaluatorAblationResult:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    evaluators = _select_evaluators(evaluator_ids)
    if matrix == "smoke":
        specs = smoke_matrix_specs()
    elif matrix == "reduced":
        specs = reduced_matrix_specs()
    else:
        specs = full_matrix_specs(include_large_gym=include_large_gym)

    config_paths: list[Path] = []
    merged_rows: list[dict[str, Any]] = []
    merged_tuning_rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        for spec in specs:
            config = _make_config(
                evaluator=evaluator,
                spec=spec,
                output_root=root,
                automl_tuning=automl_tuning,
                external_repo_path=external_repo_path,
                resume=resume,
                write_plots=write_plots,
                estimator_timeout_sec=estimator_timeout_sec,
            )
            config_path = _write_config(root, evaluator, spec, config)
            config_paths.append(config_path)
            result = run_benchmark(config)
            merged_rows.extend(_tag_rows(result.rows, evaluator=evaluator, matrix_id=spec.matrix_id))
            merged_tuning_rows.extend(_tag_rows(result.tuning_rows, evaluator=evaluator, matrix_id=spec.matrix_id))

    audit_rows = build_conservatism_audit_rows(merged_rows)
    delta_rows = paired_selection_delta_rows(merged_rows, tuning_rows=merged_tuning_rows)
    summary_rows = summarize_evaluator_rows(merged_rows, delta_rows, audit_rows)

    results_path = root / "moment_evaluator_ablation_results.csv"
    tuning_path = root / "moment_evaluator_ablation_tuning.csv"
    audit_path = root / "moment_evaluator_ablation_conservatism_audit.csv"
    summary_path = root / "moment_evaluator_ablation_summary.csv"
    delta_path = root / "moment_evaluator_ablation_selection_delta.csv"
    report_path = root / "moment_evaluator_ablation_report.md"
    write_csv(results_path, merged_rows)
    write_csv(tuning_path, merged_tuning_rows)
    write_csv(audit_path, audit_rows)
    write_csv(summary_path, summary_rows)
    write_csv(delta_path, delta_rows)
    report_path.write_text(render_evaluator_report(summary_rows, delta_rows), encoding="utf-8")
    return MomentEvaluatorAblationResult(
        output_root=root,
        config_paths=tuple(config_paths),
        results_path=results_path,
        tuning_path=tuning_path,
        audit_path=audit_path,
        summary_path=summary_path,
        delta_path=delta_path,
        report_path=report_path,
    )


def paired_selection_delta_rows(
    rows: Iterable[dict[str, Any]],
    tuning_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_list = [dict(row) for row in rows if row.get("estimator") == "neural_network_stable"]
    selected_by_eval_cell = _selected_tuning_by_eval_cell(tuning_rows or ())
    by_eval_cell: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in rows_list:
        evaluator_id = str(row.get("evaluator_id", ""))
        by_eval_cell[(evaluator_id, _cell_key(row))] = row

    control_rows = {
        cell: row
        for (evaluator_id, cell), row in by_eval_cell.items()
        if evaluator_id == "current"
    }
    out: list[dict[str, Any]] = []
    for (evaluator_id, cell), row in sorted(by_eval_cell.items()):
        if evaluator_id == "current":
            continue
        control = control_rows.get(cell)
        if control is None:
            continue
        control_score = _primary_score(control)
        treatment_score = _primary_score(row)
        if not (np.isfinite(control_score) and np.isfinite(treatment_score)):
            outcome = "missing"
            delta = np.nan
            rel_delta = np.nan
        else:
            delta = float(treatment_score - control_score)
            rel_delta = float(delta / max(abs(control_score), 1e-12))
            if abs(delta) <= TIE_REL_TOL * max(abs(control_score), 1e-12):
                outcome = "tie"
            elif delta < 0.0:
                outcome = "win"
            else:
                outcome = "loss"
        out.append(
            {
                "evaluator_id": evaluator_id,
                "matrix_id": row.get("matrix_id", ""),
                "setting": cell[0],
                "dataset_variant": cell[1],
                "policy_shift": cell[2],
                "gamma": cell[3],
                "sample_size": cell[4],
                "seed": cell[5],
                "score_type": _score_type(row),
                "control_score": control_score,
                "treatment_score": treatment_score,
                "score_delta": delta,
                "relative_score_delta": rel_delta,
                "outcome": outcome,
                "control_status": control.get("status", ""),
                "treatment_status": row.get("status", ""),
                "control_selected_candidate_id": selected_by_eval_cell.get(("current", cell), {}).get("candidate_id", ""),
                "treatment_selected_candidate_id": selected_by_eval_cell.get((evaluator_id, cell), {}).get("candidate_id", ""),
                "control_selected_score": selected_by_eval_cell.get(("current", cell), {}).get("score", ""),
                "treatment_selected_score": selected_by_eval_cell.get((evaluator_id, cell), {}).get("score", ""),
                "control_ope_value_abs_error": control.get("ope_value_abs_error", ""),
                "treatment_ope_value_abs_error": row.get("ope_value_abs_error", ""),
                "control_ratio_normalized_l1": control.get("ratio_normalized_l1", ""),
                "treatment_ratio_normalized_l1": row.get("ratio_normalized_l1", ""),
                "control_log_ratio_rmse": control.get("log_ratio_rmse", ""),
                "treatment_log_ratio_rmse": row.get("log_ratio_rmse", ""),
                "control_ess_fraction": control.get("effective_sample_size_fraction", ""),
                "treatment_ess_fraction": row.get("effective_sample_size_fraction", ""),
                "control_weight_cv": control.get("weight_cv", ""),
                "treatment_weight_cv": row.get("weight_cv", ""),
                "control_clipping_fraction": _final_clipping(control),
                "treatment_clipping_fraction": _final_clipping(row),
                "control_runtime_sec": control.get("runtime_sec", ""),
                "treatment_runtime_sec": row.get("runtime_sec", ""),
            }
        )
    return out


def summarize_evaluator_rows(
    rows: Iterable[dict[str, Any]],
    delta_rows: Iterable[dict[str, Any]] | None = None,
    audit_rows: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows_list = [dict(row) for row in rows if row.get("estimator") == "neural_network_stable"]
    deltas = [dict(row) for row in (delta_rows or paired_selection_delta_rows(rows_list))]
    audits = [dict(row) for row in (audit_rows or build_conservatism_audit_rows(rows_list))]
    audit_counts = _audit_counts(audits)
    delta_by_eval: dict[str, list[dict[str, Any]]] = {}
    for row in deltas:
        delta_by_eval.setdefault(str(row.get("evaluator_id", "")), []).append(row)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows_list:
        grouped.setdefault(str(row.get("evaluator_id", "")), []).append(row)

    summary: list[dict[str, Any]] = []
    for evaluator_id, group in sorted(grouped.items()):
        ok_rows = [row for row in group if row.get("status") == "ok"]
        controlled = [row for row in ok_rows if str(row.get("setting", "")) in CONTROLLED_SETTINGS]
        gym = [row for row in ok_rows if str(row.get("setting", "")) in GYM_SETTINGS]
        eval_deltas = delta_by_eval.get(evaluator_id, [])
        score_deltas = [_to_float(row.get("relative_score_delta")) for row in eval_deltas]
        summary.append(
            {
                "evaluator_id": evaluator_id,
                "extra_blocks": _evaluator_blocks(evaluator_id),
                "row_count": len(group),
                "ok_count": len(ok_rows),
                "error_count": sum(int(row.get("status") == "error") for row in group),
                "timeout_count": sum(int(row.get("status") == "timeout") for row in group),
                "skipped_count": sum(int(row.get("status") == "skipped") for row in group),
                "audit_fail_count": audit_counts.get(evaluator_id, {}).get("fail", 0),
                "audit_warn_count": audit_counts.get(evaluator_id, {}).get("warn", 0),
                "controlled_rows": len(controlled),
                "controlled_score_median": _median(_primary_score(row) for row in controlled),
                "controlled_ratio_normalized_l1_median": _median(_to_float(row.get("ratio_normalized_l1")) for row in controlled),
                "controlled_log_ratio_rmse_median": _median(_to_float(row.get("log_ratio_rmse")) for row in controlled),
                "collapse_count": sum(int(_is_collapse(row)) for row in controlled),
                "gym_rows": len(gym),
                "gym_score_median": _median(_primary_score(row) for row in gym),
                "gym_ope_se_units_median": _median(_to_float(row.get("ope_value_abs_error_se_units")) for row in gym),
                "gym_ope_abs_error_median": _median(_to_float(row.get("ope_value_abs_error")) for row in gym),
                "runtime_sec_median": _median(_to_float(row.get("runtime_sec")) for row in ok_rows),
                "runtime_sec_mean": _mean(_to_float(row.get("runtime_sec")) for row in ok_rows),
                "wins_vs_current": sum(int(row.get("outcome") == "win") for row in eval_deltas),
                "ties_vs_current": sum(int(row.get("outcome") == "tie") for row in eval_deltas),
                "losses_vs_current": sum(int(row.get("outcome") == "loss") for row in eval_deltas),
                "median_relative_score_delta_vs_current": _median(score_deltas),
                "mean_relative_score_delta_vs_current": _mean(score_deltas),
            }
        )
    return summary


def render_evaluator_report(summary_rows: Sequence[dict[str, Any]], delta_rows: Sequence[dict[str, Any]]) -> str:
    best = _recommended_evaluator(summary_rows)
    lines = [
        "# Moment Evaluator Ablation",
        "",
        f"Recommended evaluator: **{best}**",
        "",
        "Lower scores are better. Pairwise deltas compare each evaluator to `current` on matched cells.",
        "",
        "## Summary",
        "",
        "| evaluator | blocks | ok | win/tie/loss | median rel delta | controlled median | gym median | audit fail | collapse | runtime median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {evaluator} | {blocks} | {ok} | {wins}/{ties}/{losses} | {delta} | {controlled} | {gym} | {fail} | {collapse} | {runtime} |".format(
                evaluator=row.get("evaluator_id", ""),
                blocks=str(row.get("extra_blocks", "")).replace("|", "/"),
                ok=row.get("ok_count", 0),
                wins=row.get("wins_vs_current", 0),
                ties=row.get("ties_vs_current", 0),
                losses=row.get("losses_vs_current", 0),
                delta=_fmt(row.get("median_relative_score_delta_vs_current")),
                controlled=_fmt(row.get("controlled_score_median")),
                gym=_fmt(row.get("gym_score_median")),
                fail=row.get("audit_fail_count", 0),
                collapse=row.get("collapse_count", 0),
                runtime=_fmt(row.get("runtime_sec_median")),
            )
        )
    changed = [
        row
        for row in delta_rows
        if row.get("outcome") in {"win", "loss"} and np.isfinite(_to_float(row.get("relative_score_delta")))
    ]
    changed = sorted(changed, key=lambda row: abs(_to_float(row.get("relative_score_delta"))), reverse=True)
    lines.extend(
        [
            "",
            "## Largest Selection Deltas",
            "",
            "| evaluator | setting | shift | gamma | n | seed | selected candidates | outcome | control | treatment | rel delta |",
            "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in changed[:30]:
        lines.append(
            "| {evaluator} | {setting} | {shift} | {gamma} | {n} | {seed} | {selected} | {outcome} | {control} | {treatment} | {delta} |".format(
                evaluator=row.get("evaluator_id", ""),
                setting=row.get("setting", ""),
                shift=row.get("policy_shift", ""),
                gamma=row.get("gamma", ""),
                n=row.get("sample_size", ""),
                seed=row.get("seed", ""),
                selected=(
                    f"{row.get('control_selected_candidate_id', '')}->"
                    f"{row.get('treatment_selected_candidate_id', '')}"
                ).replace("|", "/"),
                outcome=row.get("outcome", ""),
                control=_fmt(row.get("control_score")),
                treatment=_fmt(row.get("treatment_score")),
                delta=_fmt(row.get("relative_score_delta")),
            )
        )
    return "\n".join(lines) + "\n"


def _selected_tuning_by_eval_cell(tuning_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, tuple[str, ...]], dict[str, Any]]:
    out: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for row in tuning_rows:
        if row.get("tuning_stage") != "automl_candidate":
            continue
        if str(row.get("estimator", "")) != "neural_network_stable":
            continue
        if str(row.get("budget_stage", "")) != "full":
            continue
        if _to_float(row.get("selected"), 0.0) < 0.5:
            continue
        evaluator_id = str(row.get("evaluator_id", ""))
        out[(evaluator_id, _cell_key(row))] = dict(row)
    return out


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run moment-evaluator ablations for neural AutoML CV.")
    parser.add_argument("--output-root", default="outputs/moment_evaluator_ablation")
    parser.add_argument("--matrix", choices=("smoke", "reduced", "full"), default="smoke")
    parser.add_argument("--evaluator-ids", nargs="*", default=None)
    parser.add_argument("--automl-tuning", choices=("fast", "balanced"), default="balanced")
    parser.add_argument("--external-repo-path", default="/tmp/google-research")
    parser.add_argument("--no-large-gym", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--estimator-timeout-sec", type=float, default=None)
    args = parser.parse_args()
    result = run_moment_evaluator_ablation(
        output_root=args.output_root,
        matrix=args.matrix,
        evaluator_ids=args.evaluator_ids,
        automl_tuning=args.automl_tuning,
        include_large_gym=not args.no_large_gym,
        external_repo_path=args.external_repo_path,
        resume=not args.no_resume,
        write_plots=bool(args.write_plots),
        estimator_timeout_sec=args.estimator_timeout_sec,
    )
    print(f"Wrote merged results: {result.results_path}")
    print(f"Wrote merged tuning: {result.tuning_path}")
    print(f"Wrote audit: {result.audit_path}")
    print(f"Wrote summary: {result.summary_path}")
    print(f"Wrote selection deltas: {result.delta_path}")
    print(f"Wrote report: {result.report_path}")


def _make_config(
    *,
    evaluator: MomentEvaluatorCandidate,
    spec: MatrixSpec,
    output_root: Path,
    automl_tuning: str,
    external_repo_path: str | Path,
    resume: bool,
    write_plots: bool,
    estimator_timeout_sec: float | None,
) -> OccupancyRatioBenchmarkConfig:
    light = spec.matrix_id == "smoke" or spec.matrix_id.endswith("_reduced")
    profile = "smoke" if light else "high_stakes"
    timeout = estimator_timeout_sec if estimator_timeout_sec is not None else (120.0 if light else None)
    return OccupancyRatioBenchmarkConfig(
        stage=profile,
        profile=profile,
        output_root=output_root / "runs" / evaluator.evaluator_id / spec.matrix_id,
        external_repo_path=Path(external_repo_path),
        settings=spec.settings,
        estimators=("oracle", "neural_network"),
        seeds=spec.seeds,
        sample_sizes=spec.sample_sizes,
        gammas=spec.gammas,
        discrete_policy_shifts=spec.discrete_policy_shifts,
        linear_gaussian_policy_shifts=spec.linear_gaussian_policy_shifts,
        neural_estimator_presets=("stable",),
        include_google_dual_dice=False,
        neural_num_iterations=20 if light else 80,
        neural_gradient_steps_per_iteration=3 if light else 8,
        neural_mcmc_samples=12 if light else 24,
        neural_action_steps=120 if light else 1_000,
        neural_transition_steps=160 if light else 1_400,
        neural_direct_one_step_steps=160 if light else 1_400,
        neural_direct_adjoint_steps=32 if light else 128,
        tune_cv=True,
        automl_tuning=str(automl_tuning),
        cv_folds=3,
        cv_moment_extra_blocks=evaluator.extra_blocks,
        source_state_correction_mode="auto",
        gym_target_value_rollouts=spec.gym_target_value_rollouts,
        mc_truth_samples=2_000 if light else 50_000,
        estimator_timeout_sec=timeout,
        resume=resume,
        write_plots=write_plots,
    )


def _write_config(
    output_root: Path,
    evaluator: MomentEvaluatorCandidate,
    spec: MatrixSpec,
    config: OccupancyRatioBenchmarkConfig,
) -> Path:
    path = output_root / "configs" / f"{evaluator.evaluator_id}_{spec.matrix_id}.json"
    payload = {
        "evaluator_id": evaluator.evaluator_id,
        "matrix_id": spec.matrix_id,
        "description": evaluator.description,
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


def _select_evaluators(evaluator_ids: Sequence[str] | None) -> tuple[MomentEvaluatorCandidate, ...]:
    if evaluator_ids is None:
        return EVALUATORS
    wanted = {str(evaluator_id) for evaluator_id in evaluator_ids}
    known = {candidate.evaluator_id for candidate in EVALUATORS}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown evaluator id(s): {', '.join(unknown)}")
    return tuple(candidate for candidate in EVALUATORS if candidate.evaluator_id in wanted)


def _tag_rows(
    rows: Iterable[dict[str, Any]],
    *,
    evaluator: MomentEvaluatorCandidate,
    matrix_id: str,
) -> list[dict[str, Any]]:
    tags = {
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_extra_blocks": ",".join(evaluator.extra_blocks),
        "matrix_id": matrix_id,
    }
    return [{**dict(row), **tags} for row in rows]


def _cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("setting", "")),
        str(row.get("dataset_variant", "")),
        str(row.get("policy_shift", "")),
        str(row.get("gamma", "")),
        str(row.get("sample_size", "")),
        str(row.get("seed", "")),
    )


def _score_type(row: dict[str, Any]) -> str:
    setting = str(row.get("setting", ""))
    if setting in CONTROLLED_SETTINGS and _to_float(row.get("ratio_truth_available"), 0.0) > 0.5:
        return "controlled_ratio"
    if setting in GYM_SETTINGS:
        return "gym_ope"
    return "ope"


def _primary_score(row: dict[str, Any]) -> float:
    if row.get("status") != "ok":
        return np.nan
    if _score_type(row) == "controlled_ratio":
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
    se_units = _to_float(row.get("ope_value_abs_error_se_units"))
    if np.isfinite(se_units):
        return float(se_units)
    return _to_float(row.get("ope_value_abs_error"))


def _recommended_evaluator(summary_rows: Sequence[dict[str, Any]]) -> str:
    candidates = [dict(row) for row in summary_rows if row.get("evaluator_id") != "current"]
    if not candidates:
        return "current"

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        fail = _to_float(row.get("audit_fail_count"), 0.0)
        collapse = _to_float(row.get("collapse_count"), 0.0)
        losses = _to_float(row.get("losses_vs_current"), 0.0)
        wins = _to_float(row.get("wins_vs_current"), 0.0)
        delta = _to_float(row.get("median_relative_score_delta_vs_current"), 0.0)
        return (fail, collapse, losses - wins, delta)

    best = min(candidates, key=key)
    current = next((dict(row) for row in summary_rows if row.get("evaluator_id") == "current"), None)
    if current is not None and key(best) >= (0.0, 0.0, 0.0, 0.0):
        return "current"
    return str(best.get("evaluator_id", "current"))


def _audit_counts(audit_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in audit_rows:
        evaluator_id = str(row.get("evaluator_id", ""))
        bucket = counts.setdefault(evaluator_id, {"fail": 0, "warn": 0})
        status = str(row.get("audit_status", "")).lower()
        if status == "fail":
            bucket["fail"] += 1
        elif status == "warn":
            bucket["warn"] += 1
    return counts


def _evaluator_blocks(evaluator_id: str) -> str:
    for candidate in EVALUATORS:
        if candidate.evaluator_id == evaluator_id:
            return ",".join(candidate.extra_blocks)
    return ""


def _final_clipping(row: dict[str, Any]) -> float:
    return max(
        _to_float(row.get("clipping_fraction"), 0.0),
        _to_float(row.get("projection_clipped_fraction_final"), 0.0),
    )


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


def _fmt(value: Any) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4g}"


if __name__ == "__main__":
    main()
