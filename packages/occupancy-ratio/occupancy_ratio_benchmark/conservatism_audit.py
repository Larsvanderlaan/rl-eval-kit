from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from occupancy_ratio_benchmark.config import OccupancyRatioBenchmarkConfig
from occupancy_ratio_benchmark.io import write_csv
from occupancy_ratio_benchmark.run import load_config_file
from occupancy_ratio_benchmark.runner import run_benchmark


CONTROLLED_SETTINGS = {"discrete_chain", "discrete_grid", "linear_gaussian", "nonlinear_monte_carlo"}
GYM_SETTINGS = {"gym_pendulum", "gym_mountain_car_continuous", "gym_hopper", "gym_halfcheetah"}


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_conservatism_audit_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_list = [dict(row) for row in rows]
    google_by_cell = _google_ope_by_cell(rows_list)
    out = []
    for row in rows_list:
        audit = _audit_row(row, google_by_cell=google_by_cell)
        if audit is not None:
            out.append(audit)
    return out


def write_conservatism_audit(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    audit_rows = build_conservatism_audit_rows(rows)
    audit_path = output / "conservatism_audit.csv"
    report_path = output / "conservatism_audit.md"
    write_csv(audit_path, audit_rows)
    report_path.write_text(render_conservatism_report(audit_rows), encoding="utf-8")
    return {"audit": audit_path, "report": report_path}


def render_conservatism_report(audit_rows: Iterable[dict[str, Any]]) -> str:
    rows = list(audit_rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("audit_status") == "fail"]
    warnings = [row for row in rows if row.get("audit_status") == "warn"]
    lines = [
        "# Occupancy-Ratio Conservatism Audit",
        "",
        f"Rows audited: {len(rows)} ({len(ok_rows)} ok, {len(failed)} fail, {len(warnings)} warn).",
        "",
        "## Default Decisions",
        "",
        "| family | decision | best candidate | reason |",
        "|---|---|---|---|",
    ]
    for family in ("boosted", "neural"):
        decision = _family_default_decision(rows, family)
        lines.append(
            "| {family} | {decision} | {candidate} | {reason} |".format(
                family=family,
                decision=decision["decision"],
                candidate=decision["candidate"],
                reason=decision["reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Conservatism Failures",
            "",
            "| setting | estimator | seed | gamma | ESS | true ESS | std ratio | corr | reason |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in failed[:40]:
        lines.append(
            "| {setting} | {estimator} | {seed} | {gamma} | {ess} | {true_ess} | {std_ratio} | {corr} | {reason} |".format(
                setting=row.get("setting", ""),
                estimator=row.get("estimator", ""),
                seed=row.get("seed", ""),
                gamma=_fmt(row.get("gamma")),
                ess=_fmt(row.get("effective_sample_size_fraction")),
                true_ess=_fmt(row.get("true_effective_sample_size_fraction")),
                std_ratio=_fmt(row.get("std_ratio_to_truth")),
                corr=_fmt(row.get("ratio_corr")),
                reason=str(row.get("audit_reason", "")).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Snapshot",
            "",
            "| estimator | n | fail | warn | mean OPE abs error | mean log-ratio RMSE | mean std ratio | mean ESS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _candidate_summary(rows):
        lines.append(
            "| {estimator} | {n} | {fail} | {warn} | {ope} | {logrmse} | {std_ratio} | {ess} |".format(
                estimator=row["estimator"],
                n=row["n"],
                fail=row["fail"],
                warn=row["warn"],
                ope=_fmt(row["ope_value_abs_error_mean"]),
                logrmse=_fmt(row["log_ratio_rmse_mean"]),
                std_ratio=_fmt(row["std_ratio_to_truth_mean"]),
                ess=_fmt(row["effective_sample_size_fraction_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_conservatism_audit(config: OccupancyRatioBenchmarkConfig) -> dict[str, Path]:
    result = run_benchmark(config)
    return write_conservatism_audit(result.rows, result.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or score the occupancy-ratio conservatism audit.")
    parser.add_argument("--config", default=None, help="Benchmark config JSON to run before scoring.")
    parser.add_argument("--results", default=None, help="Existing benchmark results.csv to score.")
    parser.add_argument("--output-dir", default=None, help="Directory for conservatism_audit.{csv,md}.")
    parser.add_argument("--external-repo-path", default=None)
    args = parser.parse_args()
    if args.results is None and args.config is None:
        raise SystemExit("Provide --config to run or --results to score an existing results.csv.")
    if args.results is not None:
        results_path = Path(args.results)
        output_dir = Path(args.output_dir) if args.output_dir is not None else results_path.parent
        paths = write_conservatism_audit(read_csv_rows(results_path), output_dir)
    else:
        cfg = load_config_file(args.config)
        if args.output_dir is not None:
            cfg = replace(cfg, output_root=Path(args.output_dir))
        if args.external_repo_path is not None:
            cfg = replace(cfg, external_repo_path=Path(args.external_repo_path))
        paths = run_conservatism_audit(cfg)
    print(f"Wrote conservatism audit: {paths['audit']}")
    print(f"Wrote conservatism report: {paths['report']}")


def _audit_row(row: dict[str, Any], *, google_by_cell: dict[tuple[Any, ...], float]) -> dict[str, Any] | None:
    estimator = str(row.get("estimator", ""))
    if estimator == "oracle":
        return None
    out = dict(row)
    setting = str(row.get("setting", ""))
    status = str(row.get("status", ""))
    ess = _to_float(row.get("effective_sample_size_fraction"))
    true_ess = _to_float(row.get("true_effective_sample_size_fraction"))
    pred_std = _to_float(row.get("weight_std"))
    true_cv = _to_float(row.get("true_weight_cv"))
    pred_cv = _to_float(row.get("weight_cv"))
    std_ratio = _to_float(row.get("weight_cv_ratio_to_truth"))
    if not np.isfinite(std_ratio) and np.isfinite(pred_std) and np.isfinite(true_cv) and true_cv > 0.0:
        std_ratio = pred_cv / true_cv if np.isfinite(pred_cv) else np.nan
    clipping = max(
        _finite_or_zero(row.get("clipping_fraction")),
        _finite_or_zero(row.get("projection_clipped_fraction_final")),
        _finite_or_zero(row.get("target_clip_fraction_final")),
    )
    google_ope = google_by_cell.get(_cell_key(row), np.nan)
    ope_error = _to_float(row.get("ope_value_abs_error"))
    out.update(
        {
            "audit_family": _estimator_family(estimator),
            "std_ratio_to_truth": std_ratio,
            "audit_clipping_fraction": clipping,
            "google_ope_value_abs_error": google_ope,
            "ope_abs_error_delta_vs_google": (
                ope_error - google_ope if np.isfinite(ope_error) and np.isfinite(google_ope) else np.nan
            ),
        }
    )
    audit_status, reason = _audit_status(
        setting=setting,
        status=status,
        estimator=estimator,
        ess=ess,
        true_ess=true_ess,
        std_ratio=std_ratio,
        ratio_corr=_to_float(row.get("ratio_corr")),
        clipping=clipping,
        ope_error=ope_error,
        google_ope=google_ope,
    )
    out["audit_status"] = audit_status
    out["audit_reason"] = reason
    return out


def _audit_status(
    *,
    setting: str,
    status: str,
    estimator: str,
    ess: float,
    true_ess: float,
    std_ratio: float,
    ratio_corr: float,
    clipping: float,
    ope_error: float,
    google_ope: float,
) -> tuple[str, str]:
    if status != "ok":
        return "warn" if status == "skipped" else "fail", f"estimator status is {status}"
    reasons: list[str] = []
    warns: list[str] = []
    if setting in CONTROLLED_SETTINGS and "google_dualdice" not in estimator:
        if np.isfinite(ess) and np.isfinite(true_ess) and ess > 0.95 and true_ess < 0.80:
            reasons.append("near-uniform ESS under nonconstant oracle")
        if np.isfinite(std_ratio) and std_ratio < 0.50 and np.isfinite(true_ess) and true_ess < 0.90:
            reasons.append("predicted weight spread below half oracle spread")
        min_corr = 0.70 if setting == "discrete_grid" else 0.85
        if np.isfinite(ratio_corr) and ratio_corr < min_corr:
            reasons.append(f"ratio correlation below {min_corr:.2f}")
    if setting in GYM_SETTINGS:
        if np.isfinite(clipping) and clipping > 0.10:
            reasons.append("Gym estimate is cap/clipping driven")
        elif np.isfinite(clipping) and clipping > 0.02:
            warns.append("Gym clipping above 0.02")
        if np.isfinite(ope_error) and np.isfinite(google_ope) and ope_error > google_ope * 1.10 + 1e-8:
            warns.append("Gym OPE worse than Google DualDICE by more than 10%")
    if reasons:
        return "fail", "; ".join(reasons)
    if warns:
        return "warn", "; ".join(warns)
    return "pass", "passed conservatism audit"


def _family_default_decision(rows: list[dict[str, Any]], family: str) -> dict[str, str]:
    family_rows = [row for row in rows if row.get("audit_family") == family and row.get("status") == "ok"]
    if not family_rows:
        return {"decision": "no-data", "candidate": "", "reason": "no successful rows"}
    current_name = f"{family_to_estimator_prefix(family)}_stable"
    candidates = _candidate_summary(family_rows)
    passing = [row for row in candidates if row["fail"] == 0]
    if not passing:
        return {"decision": "keep", "candidate": current_name, "reason": "no candidate passed all audit checks"}
    best = min(passing, key=lambda row: _summary_score(row))
    current = next((row for row in candidates if row["estimator"] == current_name), None)
    if current is None:
        return {"decision": "promote", "candidate": best["estimator"], "reason": "current stable candidate absent"}
    current_score = _summary_score(current)
    best_score = _summary_score(best)
    if best["estimator"] != current_name and best_score + 0.05 < current_score:
        return {
            "decision": "promote",
            "candidate": best["estimator"],
            "reason": f"score {best_score:.3g} beats current {current_score:.3g}",
        }
    return {"decision": "keep", "candidate": current_name, "reason": "no robust candidate beats current stable default"}


def _candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("estimator", "")), []).append(row)
    out = []
    for estimator, group in groups.items():
        out.append(
            {
                "estimator": estimator,
                "n": len(group),
                "fail": sum(1 for row in group if row.get("audit_status") == "fail"),
                "warn": sum(1 for row in group if row.get("audit_status") == "warn"),
                "ope_value_abs_error_mean": _mean(row.get("ope_value_abs_error") for row in group),
                "log_ratio_rmse_mean": _mean(row.get("log_ratio_rmse") for row in group),
                "std_ratio_to_truth_mean": _mean(row.get("std_ratio_to_truth") for row in group),
                "effective_sample_size_fraction_mean": _mean(row.get("effective_sample_size_fraction") for row in group),
                "clipping_fraction_mean": _mean(row.get("audit_clipping_fraction") for row in group),
                "runtime_sec_mean": _mean(row.get("runtime_sec") for row in group),
            }
        )
    return sorted(out, key=lambda row: (_summary_score(row), row["estimator"]))


def _summary_score(row: dict[str, Any]) -> float:
    score = 10.0 * float(row.get("fail", 0)) + 2.0 * float(row.get("warn", 0))
    for key, weight in (
        ("log_ratio_rmse_mean", 1.0),
        ("ope_value_abs_error_mean", 0.25),
        ("clipping_fraction_mean", 3.0),
    ):
        value = _to_float(row.get(key))
        if np.isfinite(value):
            score += weight * value
    std_ratio = _to_float(row.get("std_ratio_to_truth_mean"))
    if np.isfinite(std_ratio):
        score += abs(1.0 - min(std_ratio, 2.0))
    return float(score)


def _google_ope_by_cell(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    out = {}
    for row in rows:
        if str(row.get("estimator", "")) != "google_dualdice_neural" or row.get("status") != "ok":
            continue
        value = _to_float(row.get("ope_value_abs_error"))
        if np.isfinite(value):
            out[_cell_key(row)] = value
    return out


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("setting", ""),
        row.get("dataset_variant", ""),
        row.get("policy_shift", ""),
        row.get("gamma", ""),
        row.get("sample_size", ""),
        row.get("seed", ""),
    )


def _estimator_family(estimator: str) -> str:
    if estimator.startswith("boosted_tree"):
        return "boosted"
    if estimator.startswith("neural_network"):
        return "neural"
    if estimator.startswith("google_dualdice"):
        return "google"
    return "other"


def family_to_estimator_prefix(family: str) -> str:
    return "boosted_tree" if family == "boosted" else "neural_network"


def _mean(values: Iterable[Any]) -> float:
    vals = [_to_float(value) for value in values]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else np.nan


def _to_float(value: Any) -> float:
    try:
        if value in ("", None):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finite_or_zero(value: Any) -> float:
    out = _to_float(value)
    return out if np.isfinite(out) else 0.0


def _fmt(value: Any) -> str:
    out = _to_float(value)
    if not np.isfinite(out):
        return ""
    return f"{out:.4g}"


if __name__ == "__main__":
    main()
