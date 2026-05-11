from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from causal_ope_benchmark.baselines import EstimatorResult, run_estimator
from causal_ope_benchmark.config import CausalOPEBenchmarkConfig, DomainScenario, scenarios_for_profile
from causal_ope_benchmark.constants import (
    BENCHMARK_SCHEMA_VERSION,
    DEFAULT_OUTPUT_FILES,
    FAMILY_REGISTRY_VERSION,
    PACKAGE_VERSION,
    RESULT_SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_MISSING_DEPENDENCY,
)
from causal_ope_benchmark.epicare import make_epicare_problem
from causal_ope_benchmark.exceptions import MissingOptionalDependency
from causal_ope_benchmark.io import write_csv, write_json
from causal_ope_benchmark.metrics import score_result, summarize_rows
from causal_ope_benchmark.schema import output_schema
from causal_ope_benchmark.simulators import make_clinic_dtr_problem, make_streamlift_problem, make_streamretain_problem
from causal_ope_benchmark.types import BenchmarkProblem


@dataclass
class BenchmarkRunResult:
    output_dir: Path
    results_path: Path
    summary_path: Path
    tuning_path: Path
    diagnostics_path: Path
    manifest_path: Path
    readout_path: Path
    output_schema_path: Path
    rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    tuning_rows: list[dict[str, Any]]


def run_benchmark(config: CausalOPEBenchmarkConfig) -> BenchmarkRunResult:
    """Run the configured realistic causal OPE benchmark suite."""
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"failures": [], "truth_leakage_checks": []}
    for family in config.families:
        for scenario in scenarios_for_profile(config.profile, family):
            for sample_size in config.sample_sizes:
                for gamma in config.gammas:
                    for seed in config.seeds:
                        target_policies = ("moderate",) if family == "streamlift" else tuple(config.target_policies)
                        observed_horizons = tuple(config.observed_horizons) if family == "streamlift" else (None,)
                        for observed_horizon in observed_horizons:
                            for target_policy in target_policies:
                                try:
                                    problem = make_problem(
                                        family=family,
                                        scenario=scenario,
                                        sample_size=int(sample_size),
                                        gamma=float(gamma),
                                        seed=int(seed),
                                        observed_horizon=int(observed_horizon or 1),
                                        target_policy=str(target_policy),
                                        config=config,
                                    )
                                    diagnostics["truth_leakage_checks"].append(_leakage_check(problem))
                                except Exception as exc:
                                    if config.fail_fast:
                                        raise
                                    diagnostics["failures"].append(traceback.format_exc())
                                    rows.append(_dataset_error_row(config, family, scenario, sample_size, gamma, seed, target_policy, observed_horizon, exc))
                                    continue
                                cell_rows: list[dict[str, Any]] = []
                                for estimator in config.estimators:
                                    result = run_estimator(str(estimator), problem, config=config)
                                    row = _row_from_result(config, problem, result, scenario, target_policy, observed_horizon)
                                    row.update(score_result(problem.dataset, problem.truth, result))
                                    rows.append(row)
                                    cell_rows.append(row)
                                    tuning_rows.extend(result.tuning_rows)
                                oracle_rows = _oracle_selected_rows(cell_rows)
                                rows.extend(oracle_rows)
    summary = summarize_rows(rows)
    diagnostics["leaderboard"] = _leaderboard_diagnostics(rows, summary)
    results_path = output_dir / DEFAULT_OUTPUT_FILES["results"]
    summary_path = output_dir / DEFAULT_OUTPUT_FILES["summary"]
    tuning_path = output_dir / DEFAULT_OUTPUT_FILES["tuning_results"]
    diagnostics_path = output_dir / DEFAULT_OUTPUT_FILES["diagnostics"]
    manifest_path = output_dir / DEFAULT_OUTPUT_FILES["manifest"]
    readout_path = output_dir / DEFAULT_OUTPUT_FILES["readout"]
    output_schema_path = output_dir / DEFAULT_OUTPUT_FILES["output_schema"]
    schema_payload = output_schema()
    manifest_payload = _manifest(config)
    write_csv(results_path, rows)
    write_csv(summary_path, summary)
    write_csv(tuning_path, tuning_rows)
    write_json(diagnostics_path, diagnostics)
    write_json(manifest_path, manifest_payload)
    write_json(output_schema_path, schema_payload)
    readout_path.write_text(_render_readout(config, rows, summary, diagnostics, manifest_payload), encoding="utf-8")
    return BenchmarkRunResult(
        output_dir=output_dir,
        results_path=results_path,
        summary_path=summary_path,
        tuning_path=tuning_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
        readout_path=readout_path,
        output_schema_path=output_schema_path,
        rows=rows,
        summary_rows=summary,
        tuning_rows=tuning_rows,
    )


def make_problem(
    *,
    family: str,
    scenario: DomainScenario,
    sample_size: int,
    gamma: float,
    seed: int,
    observed_horizon: int,
    target_policy: str,
    config: CausalOPEBenchmarkConfig,
) -> BenchmarkProblem:
    """Create one benchmark problem for a family."""
    if family == "streamlift":
        return make_streamlift_problem(
            sample_size=sample_size,
            gamma=gamma,
            seed=seed,
            scenario=scenario,
            observed_horizon=observed_horizon,
            target_policy=target_policy,
            forecast_horizons=tuple(int(h) for h in config.forecast_horizons),
            long_horizon=int(config.streamlift_long_horizon),
            include_infinite_horizon=bool(config.streamlift_include_infinite_horizon),
            infinite_horizon_max_steps=int(config.streamlift_infinite_horizon_max_steps),
            mc_truth_rollouts=int(config.mc_truth_rollouts),
        )
    if family == "streamretain":
        return make_streamretain_problem(
            sample_size=sample_size,
            gamma=gamma,
            seed=seed,
            scenario=scenario,
            target_policy=target_policy,
            horizon=int(config.trajectory_horizon),
            mc_truth_rollouts=int(config.mc_truth_rollouts),
        )
    if family == "clinic_dtr":
        return make_clinic_dtr_problem(
            sample_size=sample_size,
            gamma=gamma,
            seed=seed,
            scenario=scenario,
            target_policy=target_policy,
            horizon=int(config.trajectory_horizon),
            mc_truth_rollouts=int(config.mc_truth_rollouts),
        )
    if family == "epicare":
        return make_epicare_problem(
            sample_size=sample_size,
            gamma=gamma,
            seed=seed,
            scenario=scenario,
            target_policy=target_policy,
            horizon=int(config.trajectory_horizon),
            mc_truth_rollouts=int(config.mc_truth_rollouts),
        )
    raise ValueError(f"Unknown family '{family}'.")


def _row_from_result(
    config: CausalOPEBenchmarkConfig,
    problem: BenchmarkProblem,
    result: EstimatorResult,
    scenario: DomainScenario,
    target_policy: str,
    observed_horizon: int | None,
) -> dict[str, Any]:
    dataset = problem.dataset
    row: dict[str, Any] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "family_registry_version": FAMILY_REGISTRY_VERSION,
        "package_version": PACKAGE_VERSION,
        "profile": config.profile,
        "family": dataset.family,
        "dataset": dataset.name,
        "scenario": dataset.scenario,
        "estimator": result.estimator,
        "status": result.status,
        "skip_reason": result.skip_reason,
        "gamma": float(dataset.gamma),
        "seed": int(dataset.seed),
        "sample_size": int(dataset.metadata_public.get("sample_size", dataset.n)),
        "row_count": int(dataset.n),
        "target_policy": target_policy,
        "observed_horizon": "" if observed_horizon is None else int(observed_horizon),
        "runtime_sec": float(result.runtime_sec),
        "diagnostic_only": int(bool(result.diagnostic_only)),
        "leaderboard_eligible": int(bool(problem.truth.leaderboard_eligible)),
        "leaderboard_scenario_eligible": int(bool(problem.truth.leaderboard_eligible)),
        "scenario_cell": dataset.scenario,
        "scenario_public": dataset.scenario,
    }
    row.update(_scenario_diagnostics(problem))
    row.update({f"diag_{key}": value for key, value in result.diagnostics.items()})
    if result.status == "error":
        row["error_type"] = result.skip_reason.split(":", maxsplit=1)[0]
        row["error_message"] = result.skip_reason
    return row


def _oracle_selected_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append diagnostic-only post-hoc winners within one benchmark cell."""
    specs = (
        (
            "oracle_selected_fqe_diagnostic",
            {
                "boosted_fqe",
                "neural_fqe",
                "boosted_fqe_auto",
                "neural_fqe_auto",
            },
        ),
        (
            "oracle_selected_discounted_occupancy_diagnostic",
            {
                "discounted_occupancy_boosted",
                "discounted_occupancy_neural",
                "discounted_occupancy_boosted_auto",
                "discounted_occupancy_neural_auto",
            },
        ),
    )
    out: list[dict[str, Any]] = []
    for oracle_estimator, candidates in specs:
        ranked = [
            row
            for row in cell_rows
            if row.get("status") == "ok"
            and str(row.get("estimator", "")) in candidates
            and _sort_float(row.get("policy_value_abs_error")) < float("inf")
        ]
        if not ranked:
            continue
        best = min(
            ranked,
            key=lambda row: (
                _sort_float(row.get("policy_value_abs_error")),
                _sort_float(row.get("runtime_sec")),
                str(row.get("estimator", "")),
            ),
        )
        selected = dict(best)
        selected["estimator"] = oracle_estimator
        selected["diagnostic_only"] = 1
        selected["leaderboard_result_eligible"] = 0
        selected["leaderboard_score_available"] = 1
        reason = str(selected.get("leaderboard_ineligible_reason", "") or "")
        reasons = [item for item in reason.split("|") if item]
        if "diagnostic_only" not in reasons:
            reasons.append("diagnostic_only")
        selected["leaderboard_ineligible_reason"] = "|".join(reasons)
        selected["diag_oracle_selected_from"] = best.get("estimator", "")
        selected["diag_oracle_selected_policy_value_abs_error"] = best.get("policy_value_abs_error", "")
        selected["diag_oracle_selected_runtime_sec"] = best.get("runtime_sec", "")
        out.append(selected)
    return out


def _dataset_error_row(
    config: CausalOPEBenchmarkConfig,
    family: str,
    scenario: DomainScenario,
    sample_size: int,
    gamma: float,
    seed: int,
    target_policy: str,
    observed_horizon: int | None,
    exc: Exception,
) -> dict[str, Any]:
    status = STATUS_MISSING_DEPENDENCY if _is_missing_dependency(exc) else STATUS_ERROR
    reason = STATUS_MISSING_DEPENDENCY if status == STATUS_MISSING_DEPENDENCY else "dataset_error"
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "family_registry_version": FAMILY_REGISTRY_VERSION,
        "package_version": PACKAGE_VERSION,
        "profile": config.profile,
        "family": family,
        "dataset": "",
        "scenario": "dataset_error",
        "estimator": "dataset",
        "status": status,
        "skip_reason": f"{type(exc).__name__}: {exc}",
        "gamma": float(gamma),
        "seed": int(seed),
        "sample_size": int(sample_size),
        "row_count": 0,
        "target_policy": target_policy,
        "observed_horizon": "" if observed_horizon is None else int(observed_horizon),
        "runtime_sec": 0.0,
        "diagnostic_only": 1,
        "leaderboard_eligible": int(bool(scenario.leaderboard_eligible)),
        "leaderboard_scenario_eligible": int(bool(scenario.leaderboard_eligible)),
        "leaderboard_result_eligible": 0,
        "leaderboard_score_available": 0,
        "leaderboard_ineligible_reason": reason,
    }


def _leakage_check(problem: BenchmarkProblem) -> dict[str, Any]:
    public_keys = "|".join(sorted(problem.dataset.metadata_public))
    public_values = "|".join(str(value).lower() for value in problem.dataset.metadata_public.values())
    forbidden_value_tokens = ("surrogate_validity", "target_mc", "oracle", "latent")
    return {
        "dataset": problem.dataset.name,
        "public_key_count": len(problem.dataset.metadata_public),
        "public_keys_ok": int(not any(token in public_keys.lower() for token in forbidden_value_tokens)),
        "public_values_ok": int(not any(token in public_values for token in forbidden_value_tokens)),
        "truth_separate": int(problem.truth is not problem.dataset),
    }


def _scenario_diagnostics(problem: BenchmarkProblem) -> dict[str, Any]:
    dataset = problem.dataset
    ratios = np.asarray(dataset.target_propensity_observed_action) / np.asarray(dataset.behavior_propensity)
    actions = np.argmax(np.asarray(dataset.actions), axis=1)
    counts = np.bincount(actions, minlength=dataset.action_dim).astype(np.float64)
    probs = counts / max(float(np.sum(counts)), 1.0)
    positive = probs[probs > 0.0]
    action_entropy = float(-np.sum(positive * np.log(positive)))
    per_unit_lengths = [np.flatnonzero(dataset.unit_id == unit).shape[0] for unit in np.unique(dataset.unit_id)]
    out: dict[str, Any] = {
        "diag_action_entropy": action_entropy,
        "diag_overlap_ratio_min": float(np.min(ratios)),
        "diag_overlap_ratio_p5": float(np.quantile(ratios, 0.05)),
        "diag_overlap_ratio_p50": float(np.quantile(ratios, 0.50)),
        "diag_target_behavior_policy_distance": float(np.mean(np.abs(ratios - 1.0))),
        "diag_terminal_rate": float(np.mean(dataset.terminals)),
        "diag_censoring_rate": float(np.mean(dataset.censoring)),
        "diag_missingness_rate": float(np.mean(dataset.missingness_mask)),
        "diag_average_trajectory_length": float(np.mean(per_unit_lengths)) if per_unit_lengths else 0.0,
        "diag_reward_mean": float(np.mean(dataset.rewards)),
        "diag_reward_sd": float(np.std(dataset.rewards)),
        "diag_fqe_row_count": int(dataset.n),
    }
    if dataset.family == "streamretain":
        discount = np.asarray(dataset.outcome_components.get("intervention_cost", np.zeros(dataset.n)))
        fatigue = np.asarray(dataset.outcome_components.get("fatigue", np.zeros(dataset.n)))
        contact = np.isin(actions, [1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float64)
        out.update(
            {
                "diag_contact_rate": float(np.mean(contact)),
                "diag_discount_support_spend_mean": float(np.mean(discount)),
                "diag_fatigue_exposure_mean": float(np.mean(fatigue)),
                "diag_budget_violation": int(float(np.mean(discount)) > 2.0),
            }
        )
    if dataset.family == "clinic_dtr":
        toxicity = np.asarray(dataset.outcome_components.get("toxicity", np.zeros(dataset.n)))
        intensify = np.isin(actions, [2, 3]).astype(np.float64)
        contraindicated = intensify * ((dataset.states[:, 6] > 0.72) | (dataset.states[:, 8] > 0.68))
        action_intensity = (
            float(np.mean(np.abs(np.asarray(dataset.action_dose)))) if dataset.action_dose is not None else float(np.mean(actions))
        )
        out.update(
            {
                "diag_contraindicated_action_rate": float(np.mean(contraindicated)),
                "diag_high_toxicity_intensification_rate": float(np.mean(intensify * (toxicity > 0.55))),
                "diag_monitoring_action_rate": float(np.mean(actions == 0)),
                "diag_action_intensity_mean": action_intensity,
            }
        )
    return out


def _manifest(config: CausalOPEBenchmarkConfig) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pytest", "fqe", "torch", "lightgbm", "gym", "gymnasium", "scope_rl", "epicare"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "family_registry_version": FAMILY_REGISTRY_VERSION,
        "package_version": PACKAGE_VERSION,
        "config": _manifest_config(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "optional_dependencies": {
            "fqe": packages.get("fqe") is not None,
            "torch": packages.get("torch") is not None,
            "lightgbm": packages.get("lightgbm") is not None,
            "gym": packages.get("gym") is not None,
            "gymnasium": packages.get("gymnasium") is not None,
            "scope_rl": packages.get("scope_rl") is not None,
            "epicare": packages.get("epicare") is not None,
        },
        "git": _git_metadata(),
    }


def _manifest_config(config: CausalOPEBenchmarkConfig) -> dict[str, Any]:
    payload = dict(asdict(config))
    if "mc_truth_rollouts" in payload:
        payload["scorer_mc_rollouts"] = payload.pop("mc_truth_rollouts")
    return payload


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    out: dict[str, Any] = {}
    for key, args in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }.items():
        try:
            value = subprocess.run(args, cwd=root, check=True, capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            value = ""
        out[key] = value
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True, timeout=2).stdout
        out["dirty"] = bool(status.strip())
    except Exception:
        out["dirty"] = ""
    return out


def _leaderboard_diagnostics(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> dict[str, Any]:
    deployable = [row for row in rows if row.get("status") == "ok" and not _truthy(row.get("diagnostic_only"))]
    eligible = [row for row in deployable if _truthy(row.get("leaderboard_result_eligible"))]
    reason_counts: dict[str, int] = {}
    for row in deployable:
        reason_text = str(row.get("leaderboard_ineligible_reason", "") or "")
        for reason in reason_text.split("|"):
            if not reason:
                continue
            base = reason.split(":", maxsplit=1)[0]
            reason_counts[base] = reason_counts.get(base, 0) + 1
    best_by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({str(row.get("family", "")) for row in eligible}):
        family_rows = [row for row in eligible if row.get("family") == family]
        ranked = sorted(
            family_rows,
            key=lambda row: (
                _sort_float(row.get("calibrated_score")),
                _sort_float(row.get("primary_weighted_mae")),
                str(row.get("estimator", "")),
            ),
        )
        if ranked:
            best = ranked[0]
            best_by_family[family] = {
                "estimator": best.get("estimator", ""),
                "calibrated_score": best.get("calibrated_score", ""),
                "primary_weighted_mae": best.get("primary_weighted_mae", ""),
                "dataset": best.get("dataset", ""),
            }
    return {
        "deployable_rows": len(deployable),
        "eligible_rows": len(eligible),
        "ineligible_rows": len(deployable) - len(eligible),
        "ineligible_reason_counts": reason_counts,
        "best_by_family": best_by_family,
        "summary_rows": len(summary),
    }


def _render_readout(
    config: CausalOPEBenchmarkConfig,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    status_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    diagnostic_rows = 0
    for row in rows:
        status_counts[str(row.get("status", ""))] = status_counts.get(str(row.get("status", "")), 0) + 1
        family_counts[str(row.get("family", ""))] = family_counts.get(str(row.get("family", "")), 0) + 1
        diagnostic_rows += int(_truthy(row.get("diagnostic_only")))
    optional = manifest.get("optional_dependencies", {})
    optional_text = ", ".join(
        f"{name}={'yes' if bool(available) else 'no'}"
        for name, available in sorted(optional.items())
    ) if isinstance(optional, dict) else "unavailable"
    lines = [
        "# Causal OPE Benchmark Readout",
        "",
        f"- profile: `{config.profile}`",
        f"- rows: `{len(rows)}`",
        f"- failures: `{len(diagnostics.get('failures', []))}`",
        f"- diagnostic-only rows: `{diagnostic_rows}`",
        f"- package version: `{manifest.get('package_version', PACKAGE_VERSION)}`",
        f"- result schema: `{manifest.get('result_schema_version', RESULT_SCHEMA_VERSION)}`",
        f"- optional dependencies: {optional_text}",
        "",
        "## Family Summary",
        "",
        "| family | rows |",
        "| --- | ---: |",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "## Estimator Summary",
            "",
            "| family | estimator | ok/deployable/leaderboard | calibrated | weighted primary MAE | primary MAE | runtime |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary:
        lines.append(
            "| {family} | {estimator} | {ok}/{deployable}/{leaderboard} | {calibrated} | {weighted} | {mae} | {runtime} |".format(
                family=row.get("family", ""),
                estimator=row.get("estimator", ""),
                ok=row.get("ok_rows", ""),
                deployable=row.get("deployable_rows", ""),
                leaderboard=row.get("leaderboard_eligible_rows", ""),
                calibrated=_fmt(row.get("leaderboard_calibrated_score_mean")),
                weighted=_fmt(row.get("leaderboard_primary_weighted_mae_mean")),
                mae=_fmt(row.get("primary_mae_mean")),
                runtime=_fmt(row.get("runtime_sec_mean")),
            )
        )
    comparison_lines = _tree_vs_neural_comparison_lines(rows)
    if comparison_lines:
        lines.extend(["", "## Tree vs Neural", "", *comparison_lines])
    lines.extend(["", "## Status Summary", "", "| status | rows |", "| --- | ---: |"])
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    leaderboard = diagnostics.get("leaderboard", {})
    reason_counts = leaderboard.get("ineligible_reason_counts", {}) if isinstance(leaderboard, dict) else {}
    if isinstance(reason_counts, dict) and reason_counts:
        reason_text = ", ".join(f"{key}={value}" for key, value in sorted(reason_counts.items()))
    else:
        reason_text = "none"
    best_by_family = leaderboard.get("best_by_family", {}) if isinstance(leaderboard, dict) else {}
    lines.extend(
        [
            "",
            "## Leaderboard Diagnostics",
            "",
            f"- eligible/deployable rows: `{leaderboard.get('eligible_rows', 0)}/{leaderboard.get('deployable_rows', 0)}`",
            f"- ineligible reasons: {reason_text}",
            "- diagnostic-only rows are excluded from leaderboard eligibility.",
        ]
    )
    if isinstance(best_by_family, dict) and best_by_family:
        lines.extend(["", "| family | best estimator | calibrated | weighted primary MAE |", "| --- | --- | ---: | ---: |"])
        for family, best in sorted(best_by_family.items()):
            if not isinstance(best, dict):
                continue
            lines.append(
                "| {family} | {estimator} | {calibrated} | {weighted} |".format(
                    family=family,
                    estimator=best.get("estimator", ""),
                    calibrated=_fmt(best.get("calibrated_score")),
                    weighted=_fmt(best.get("primary_weighted_mae")),
                )
            )
    lines.extend(
        [
            "",
            "Diagnostics such as ESS, ratio error, clipping, missingness, and censoring are reported as diagnostics only and are not used for tuning or product selection.",
            "",
        ]
    )
    return "\n".join(lines)


def _tree_vs_neural_comparison_lines(rows: list[dict[str, Any]]) -> list[str]:
    specs = (
        ("FQE", "default", "boosted_fqe", "neural_fqe"),
        ("FQE", "auto", "boosted_fqe_auto", "neural_fqe_auto"),
        ("discounted_occupancy", "default", "discounted_occupancy_boosted", "discounted_occupancy_neural"),
        ("discounted_occupancy", "auto", "discounted_occupancy_boosted_auto", "discounted_occupancy_neural_auto"),
    )
    out = [
        "| family | estimator family | mode | pairs | boosted MAE | neural MAE | boosted-neural | boosted win rate | boosted runtime | neural runtime | boosted ESS | neural ESS | boosted weight CV | neural weight CV |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    emitted = False
    for family in sorted({str(row.get("family", "")) for row in rows if row.get("status") == "ok"}):
        family_rows = [row for row in rows if row.get("family") == family and row.get("status") == "ok" and not _truthy(row.get("diagnostic_only"))]
        for estimator_family, mode, boosted_name, neural_name in specs:
            pairs = _paired_rows(family_rows, boosted_name=boosted_name, neural_name=neural_name)
            if not pairs:
                continue
            boosted_errors = [_sort_float(boosted.get("policy_value_abs_error")) for boosted, _ in pairs]
            neural_errors = [_sort_float(neural.get("policy_value_abs_error")) for _, neural in pairs]
            diffs = [b - n for b, n in zip(boosted_errors, neural_errors)]
            boosted_rows = [boosted for boosted, _ in pairs]
            neural_rows = [neural for _, neural in pairs]
            out.append(
                "| {family} | {estimator_family} | {mode} | {pairs} | {b_mae} | {n_mae} | {diff} | {win} | {b_runtime} | {n_runtime} | {b_ess} | {n_ess} | {b_cv} | {n_cv} |".format(
                    family=family,
                    estimator_family=estimator_family,
                    mode=mode,
                    pairs=len(pairs),
                    b_mae=_fmt(_mean_number(boosted_errors)),
                    n_mae=_fmt(_mean_number(neural_errors)),
                    diff=_fmt(_mean_number(diffs)),
                    win=_fmt(_mean_number([float(diff < 0.0) for diff in diffs])),
                    b_runtime=_fmt(_mean_number(_finite_values(row.get("runtime_sec") for row in boosted_rows))),
                    n_runtime=_fmt(_mean_number(_finite_values(row.get("runtime_sec") for row in neural_rows))),
                    b_ess=_fmt(_mean_number(_finite_values(row.get("ess_fraction") for row in boosted_rows))),
                    n_ess=_fmt(_mean_number(_finite_values(row.get("ess_fraction") for row in neural_rows))),
                    b_cv=_fmt(_mean_number(_finite_values(row.get("diag_weight_cv") for row in boosted_rows))),
                    n_cv=_fmt(_mean_number(_finite_values(row.get("diag_weight_cv") for row in neural_rows))),
                )
            )
            emitted = True
    return out if emitted else []


def _paired_rows(rows: list[dict[str, Any]], *, boosted_name: str, neural_name: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    keys = ("dataset", "scenario", "gamma", "seed", "sample_size", "target_policy", "observed_horizon")
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        estimator = str(row.get("estimator", ""))
        if estimator not in {boosted_name, neural_name}:
            continue
        if _sort_float(row.get("policy_value_abs_error")) == float("inf"):
            continue
        key = tuple(row.get(part, "") for part in keys)
        by_key.setdefault(key, {})[estimator] = row
    return [
        (entry[boosted_name], entry[neural_name])
        for entry in by_key.values()
        if boosted_name in entry and neural_name in entry
    ]


def _finite_values(values) -> list[float]:
    out: list[float] = []
    for value in values:
        val = _sort_float(value)
        if np.isfinite(val):
            out.append(val)
    return out


def _mean_number(values) -> float | str:
    vals = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(vals)) if vals else ""


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _sort_float(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return val if np.isfinite(val) else float("inf")


def _fmt(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.4g}"


def _is_missing_dependency(exc: Exception) -> bool:
    return isinstance(exc, (MissingOptionalDependency, ModuleNotFoundError))
