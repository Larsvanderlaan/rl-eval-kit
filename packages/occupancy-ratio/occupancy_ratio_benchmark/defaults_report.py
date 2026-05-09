from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from occupancy_ratio_benchmark.io import write_csv
from occupancy_ratio_benchmark.conservatism_audit import build_conservatism_audit_rows


GOOGLE_ESTIMATOR = "google_dualdice_neural"
DEFAULT_INELIGIBLE_ESTIMATORS = {
    "boosted_tree_squared",
    "boosted_tree_huber",
    "neural_network_squared",
    "neural_network_huber",
    "neural_network_google_parity",
}


def generate_defaults_report(results_csv: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Summarize benchmark CSVs into a default-selection report."""
    results_path = Path(results_csv)
    rows = _read_rows(results_path)
    rows = _attach_conservatism_status(rows)
    out_dir = Path(output_dir) if output_dir is not None else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _estimator_summary(rows)
    winner_rows = _balanced_winners(rows)
    dice_rows = _neural_vs_dice(rows)
    recommendation = _recommend_default(summary_rows)

    summary_path = out_dir / "defaults_summary.csv"
    winners_path = out_dir / "defaults_winners.csv"
    dice_path = out_dir / "defaults_neural_vs_dice.csv"
    report_path = out_dir / "defaults_report.md"
    write_csv(summary_path, summary_rows)
    write_csv(winners_path, winner_rows)
    write_csv(dice_path, dice_rows)
    report_path.write_text(
        _render_markdown(
            results_path=results_path,
            summary_rows=summary_rows,
            winner_rows=winner_rows,
            dice_rows=dice_rows,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "winners": winners_path,
        "neural_vs_dice": dice_path,
        "report": report_path,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _attach_conservatism_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add conservatism audit fields so default selection cannot ignore failures."""
    audit_rows = build_conservatism_audit_rows(rows)
    audit_by_key = {
        (_cell_key(row), str(row.get("estimator", ""))): row
        for row in audit_rows
    }
    out = []
    for row in rows:
        merged = dict(row)
        audit = audit_by_key.get((_cell_key(row), str(row.get("estimator", ""))))
        if audit is not None:
            merged["audit_status"] = audit.get("audit_status", "")
            merged["audit_reason"] = audit.get("audit_reason", "")
        out.append(merged)
    return out


def _to_float(value: Any, default: float = np.nan) -> float:
    if value in ("", None):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _balanced_score(row: dict[str, Any]) -> float:
    if row.get("status") != "ok":
        return float("inf")
    if str(row.get("audit_status", "")).lower() == "fail":
        return float("inf")
    if _has_safety_failure(row):
        return float("inf")
    if _row_has_ratio_truth(row):
        score = _ratio_truth_score(row)
    else:
        score = _ope_score(row)
    if not np.isfinite(score):
        return float("inf")
    return float(score + _safety_tiebreaker(row))


def _row_has_ratio_truth(row: dict[str, Any]) -> bool:
    available = _to_float(row.get("ratio_truth_available"))
    if np.isfinite(available):
        return bool(available > 0.5)
    has_ratio_metric = np.isfinite(_to_float(row.get("ratio_normalized_l1"))) or np.isfinite(
        _to_float(row.get("log_ratio_rmse"))
    )
    return bool(has_ratio_metric and row.get("truth_source") not in {"", None})


def _ratio_truth_score(row: dict[str, Any]) -> float:
    ratio_l1 = _to_float(row.get("ratio_normalized_l1"))
    log_rmse = _to_float(row.get("log_ratio_rmse"))
    ope = _to_float(row.get("ope_value_abs_error"), 0.0)
    if not np.isfinite(ratio_l1):
        ratio_l1 = _to_float(row.get("ratio_rel_mse"))
    if not np.isfinite(ratio_l1):
        ratio_l1 = _to_float(row.get("ratio_l1"))
    if not np.isfinite(ratio_l1):
        return _ope_score(row)
    score = ratio_l1
    if np.isfinite(log_rmse):
        score += 0.25 * log_rmse
    if np.isfinite(ope):
        score += 0.10 * ope
    return float(score)


def _ope_score(row: dict[str, Any]) -> float:
    se_units = _to_float(row.get("ope_value_abs_error_se_units"))
    if np.isfinite(se_units):
        return float(se_units)
    primary = _to_float(row.get("ope_value_abs_error"))
    if not np.isfinite(primary):
        primary = _to_float(row.get("absolute_error"))
    if not np.isfinite(primary):
        primary = _to_float(row.get("log_ratio_rmse"))
    return float(primary)


def _has_safety_failure(row: dict[str, Any]) -> bool:
    nonfinite_raw = _to_float(row.get("nonfinite_raw_fraction"), 0.0)
    negative_raw = _to_float(row.get("negative_raw_fraction"), 0.0)
    clipping = max(
        _to_float(row.get("clipping_fraction"), 0.0),
        _to_float(row.get("projection_clipped_fraction_final"), 0.0),
    )
    if np.isfinite(nonfinite_raw) and nonfinite_raw > 0.0:
        return True
    if np.isfinite(negative_raw) and negative_raw > 0.01:
        return True
    return bool(np.isfinite(clipping) and clipping > 0.50)


def _safety_tiebreaker(row: dict[str, Any]) -> float:
    negative_raw = max(0.0, _to_float(row.get("negative_raw_fraction"), 0.0))
    clipping = max(
        0.0,
        _to_float(row.get("clipping_fraction"), 0.0),
        _to_float(row.get("projection_clipped_fraction_final"), 0.0),
    )
    nonfinite_raw = max(0.0, _to_float(row.get("nonfinite_raw_fraction"), 0.0))
    return float(0.02 * clipping + 1.0 * negative_raw + 5.0 * nonfinite_raw)


def _cell_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("profile", row.get("stage", ""))),
        str(row.get("setting", "")),
        str(row.get("dataset_variant", "")),
        str(row.get("policy_shift", "")),
        str(row.get("gamma", "")),
        str(row.get("sample_size", "")),
        str(row.get("seed", "")),
    )


def _balanced_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("estimator") == "oracle":
            continue
        groups.setdefault(_cell_key(row), []).append(row)
    winners = []
    for key, group in groups.items():
        scored = [(_balanced_score(row), row) for row in group]
        scored = [(score, row) for score, row in scored if np.isfinite(score)]
        if not scored:
            continue
        score, row = min(scored, key=lambda item: item[0])
        winners.append(
            {
                "profile": key[0],
                "setting": key[1],
                "dataset_variant": key[2],
                "policy_shift": key[3],
                "gamma": key[4],
                "sample_size": key[5],
                "seed": key[6],
                "winning_estimator": row.get("estimator", ""),
                "balanced_score": float(score),
                "ope_value_abs_error": row.get("ope_value_abs_error", ""),
                "ratio_normalized_l1": row.get("ratio_normalized_l1", ""),
                "effective_sample_size_fraction": row.get("effective_sample_size_fraction", ""),
                "true_effective_sample_size_fraction": row.get("true_effective_sample_size_fraction", ""),
                "ess_fraction_abs_error_to_truth": row.get("ess_fraction_abs_error_to_truth", ""),
            }
        )
    return winners


def _estimator_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("estimator", "")), []).append(row)
    out = []
    for estimator, group in sorted(groups.items()):
        scores = np.asarray([_balanced_score(row) for row in group], dtype=np.float64)
        finite_scores = scores[np.isfinite(scores)]
        ok_rows = [row for row in group if row.get("status") == "ok"]
        out.append(
            {
                "estimator": estimator,
                "n_rows": len(group),
                "ok_rows": len(ok_rows),
                "error_rows": sum(row.get("status") == "error" for row in group),
                "timeout_rows": sum(row.get("status") == "timeout" for row in group),
                "skipped_rows": sum(row.get("status") == "skipped" for row in group),
                "audit_fail_rows": sum(str(row.get("audit_status", "")).lower() == "fail" for row in group),
                "audit_warn_rows": sum(str(row.get("audit_status", "")).lower() == "warn" for row in group),
                "balanced_score_median": float(np.median(finite_scores)) if finite_scores.size else "",
                "balanced_score_mean": float(np.mean(finite_scores)) if finite_scores.size else "",
                "ope_value_abs_error_median": _median_metric(ok_rows, "ope_value_abs_error"),
                "ratio_normalized_l1_median": _median_metric(ok_rows, "ratio_normalized_l1"),
                "log_ratio_rmse_median": _median_metric(ok_rows, "log_ratio_rmse"),
                "ess_fraction_median": _median_metric(ok_rows, "effective_sample_size_fraction"),
                "true_ess_fraction_median": _median_metric(ok_rows, "true_effective_sample_size_fraction"),
                "ess_abs_error_to_truth_median": _median_metric(ok_rows, "ess_fraction_abs_error_to_truth"),
                "weight_q99_ratio_to_truth_median": _median_metric(ok_rows, "weight_q99_ratio_to_truth"),
                "clipping_fraction_median": _median_metric(ok_rows, "clipping_fraction"),
            }
        )
    return out


def _median_metric(rows: list[dict[str, Any]], name: str) -> float | str:
    values = np.asarray([_to_float(row.get(name)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else ""


def _neural_vs_dice(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_estimator: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
    for row in rows:
        by_key_estimator[(_cell_key(row), str(row.get("estimator", "")))] = row
    neural_estimators = sorted(
        {
            str(row.get("estimator", ""))
            for row in rows
            if str(row.get("estimator", "")).startswith("neural_network")
        }
    )
    out = []
    for estimator in neural_estimators:
        comparisons = []
        for key, dice_row in [
            (key, row)
            for (key, name), row in by_key_estimator.items()
            if name == GOOGLE_ESTIMATOR and row.get("status") == "ok"
        ]:
            neural_row = by_key_estimator.get((key, estimator))
            if neural_row is None or neural_row.get("status") != "ok":
                continue
            neural_score = _balanced_score(neural_row)
            dice_score = _balanced_score(dice_row)
            if np.isfinite(neural_score) and np.isfinite(dice_score):
                comparisons.append((neural_score, dice_score))
        if not comparisons:
            continue
        neural_scores = np.asarray([pair[0] for pair in comparisons], dtype=np.float64)
        dice_scores = np.asarray([pair[1] for pair in comparisons], dtype=np.float64)
        out.append(
            {
                "estimator": estimator,
                "comparison_cells": int(len(comparisons)),
                "win_rate_vs_google_dualdice": float(np.mean(neural_scores <= dice_scores)),
                "median_score_ratio_vs_google": float(np.median(neural_scores / np.maximum(dice_scores, 1e-12))),
                "neural_score_median": float(np.median(neural_scores)),
                "google_score_median": float(np.median(dice_scores)),
            }
        )
    return out


def _recommend_default(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    neural_rows = [
        row
        for row in summary_rows
        if str(row.get("estimator", "")).startswith("neural_network")
        and row.get("balanced_score_median") not in {"", None}
        and int(row.get("audit_fail_rows", 0) or 0) == 0
        and str(row.get("estimator", "")) not in DEFAULT_INELIGIBLE_ESTIMATORS
    ]
    if neural_rows:
        neural_rows = sorted(neural_rows, key=lambda row: float(row["balanced_score_median"]))
        best = neural_rows[0]
        google = next((row for row in summary_rows if row.get("estimator") == GOOGLE_ESTIMATOR), None)
        if google is None or google.get("balanced_score_median") in {"", None}:
            return {
                "recommended_default": best["estimator"],
                "reason": "best neural median score; Google DualDICE unavailable",
            }
        google_score = float(google["balanced_score_median"])
        best_score = float(best["balanced_score_median"])
        if best_score <= 1.10 * google_score:
            return {
                "recommended_default": best["estimator"],
                "reason": "best neural preset is within the ratio/OPE guardrail against Google DualDICE",
            }
        return {
            "recommended_default": "neural_network_stable",
            "reason": "no neural finalist matched Google DualDICE under the balanced guardrails; keep stable LSIF",
        }

    boosted_candidates = {
        "boosted_tree_stable",
        "boosted_tree_relaxed_tail",
        "boosted_tree_stable_logistic_nuisance",
        "boosted_tree_auto",
    }
    boosted_rows = [
        row
        for row in summary_rows
        if str(row.get("estimator", "")) in boosted_candidates and row.get("balanced_score_median") not in {"", None}
        and int(row.get("audit_fail_rows", 0) or 0) == 0
    ]
    stable = next((row for row in boosted_rows if row.get("estimator") == "boosted_tree_stable"), None)
    if boosted_rows:
        boosted_rows = sorted(boosted_rows, key=lambda row: float(row["balanced_score_median"]))
        best = boosted_rows[0]
        if stable is None:
            return {
                "recommended_default": best["estimator"],
                "reason": "best boosted median score; stable boosted rows unavailable",
            }
        stable_score = float(stable["balanced_score_median"])
        best_score = float(best["balanced_score_median"])
        if best["estimator"] != "boosted_tree_stable" and best_score < 0.95 * stable_score:
            return {
                "recommended_default": best["estimator"],
                "reason": "boosted challenger materially improves median ratio/OPE score over stable",
            }
        return {
            "recommended_default": "boosted_tree_stable",
            "reason": "stable boosted default remains within the material ratio/OPE guardrail",
        }

    return {
        "recommended_default": "boosted_tree_stable",
        "reason": "no candidate passed all default-selection safety and conservatism checks; keep the stable baseline",
    }


def _render_markdown(
    *,
    results_path: Path,
    summary_rows: list[dict[str, Any]],
    winner_rows: list[dict[str, Any]],
    dice_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> str:
    lines = [
        "# Occupancy Ratio Defaults Report",
        "",
        f"Source results: `{results_path}`",
        "",
        f"Recommended default: `{recommendation['recommended_default']}`",
        "",
        str(recommendation["reason"]),
        "",
        "## Estimator Summary",
        "",
        "| estimator | ok/rows | median score | median ratio L1 | median OPE error | median ESS | median ESS error to truth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda item: _to_float(item.get("balanced_score_median"), float("inf"))):
        lines.append(
            "| {estimator} | {ok_rows}/{n_rows} | {score} | {ratio_l1} | {ope} | {ess} | {ess_gap} |".format(
                estimator=row["estimator"],
                ok_rows=row["ok_rows"],
                n_rows=row["n_rows"],
                score=_fmt(row.get("balanced_score_median")),
                ratio_l1=_fmt(row.get("ratio_normalized_l1_median")),
                ope=_fmt(row.get("ope_value_abs_error_median")),
                ess=_fmt(row.get("ess_fraction_median")),
                ess_gap=_fmt(row.get("ess_abs_error_to_truth_median")),
            )
        )
    lines.extend(["", "## Neural Vs Google DualDICE", "", "| estimator | cells | win rate | median score ratio |", "|---|---:|---:|---:|"])
    for row in dice_rows:
        lines.append(
            f"| {row['estimator']} | {row['comparison_cells']} | "
            f"{_fmt(row['win_rate_vs_google_dualdice'])} | {_fmt(row['median_score_ratio_vs_google'])} |"
        )
    lines.extend(["", "## Winner Count", "", "| estimator | wins |", "|---|---:|"])
    counts: dict[str, int] = {}
    for row in winner_rows:
        counts[str(row["winning_estimator"])] = counts.get(str(row["winning_estimator"]), 0) + 1
    for estimator, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {estimator} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = _to_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize occupancy-ratio benchmark defaults.")
    parser.add_argument("results_csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    paths = generate_defaults_report(args.results_csv, args.output_dir)
    print(f"Wrote report: {paths['report']}")


if __name__ == "__main__":
    main()
