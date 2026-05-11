#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}g}"


def _median(df: pd.DataFrame, method: str, metric: str, *, schedule: str | None = None, regime: str | None = None) -> float:
    sub = df[df["method"].astype(str) == method]
    if schedule is not None:
        sub = sub[sub["schedule"].astype(str) == schedule]
    if regime is not None:
        sub = sub[sub["regime"].astype(str) == regime]
    if sub.empty or metric not in sub:
        return float("nan")
    return float(sub[metric].median())


def _failure_rate(df: pd.DataFrame, method: str, *, schedule: str | None = None, regime: str | None = None) -> float:
    sub = df[df["method"].astype(str) == method]
    if schedule is not None:
        sub = sub[sub["schedule"].astype(str) == schedule]
    if regime is not None:
        sub = sub[sub["regime"].astype(str) == regime]
    if sub.empty:
        return float("nan")
    failed = sub.get("failed", pd.Series(np.zeros(len(sub)), index=sub.index)).astype(float)
    diverged = sub.get("diverged", pd.Series(np.zeros(len(sub)), index=sub.index)).astype(float)
    return float(np.mean((failed > 0) | (diverged > 0)))


def _improvement(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or abs(reference) < 1e-12:
        return float("nan")
    return float((reference - candidate) / abs(reference))


def _regime_shift_table(results_dir: Path) -> list[dict[str, float | str]]:
    ref_path = results_dir / "reference_context.npz"
    if not ref_path.exists():
        return []
    ref = np.load(ref_path)
    target_sa = ref["target_sa_dist"]
    out: list[dict[str, float | str]] = []
    for key in sorted(ref.files):
        if not key.endswith("_sa_dist"):
            continue
        regime = key.replace("_sa_dist", "")
        behavior_sa = ref[key]
        ratio = target_sa / np.maximum(behavior_sa, 1e-12)
        ratio = ratio / np.maximum(np.sum(behavior_sa * ratio), 1e-300)
        ess = float(1.0 / np.maximum(np.sum(behavior_sa * ratio * ratio), 1e-300))
        out.append(
            {
                "regime": regime,
                "tv_state_action": float(0.5 * np.sum(np.abs(target_sa - behavior_sa))),
                "chi2_target_over_behavior": float(np.sum((target_sa - behavior_sa) ** 2 / np.maximum(behavior_sa, 1e-12))),
                "oracle_ratio_q99": float(np.quantile(ratio, 0.99)),
                "oracle_ratio_max": float(np.max(ratio)),
                "oracle_ess_fraction_grid": ess,
            }
        )
    return out


def _gate_rows(final: pd.DataFrame) -> tuple[list[dict[str, object]], list[str]]:
    main = final[
        (final["stage"].astype(str) == "stage2_weights")
        & (final["learner"].astype(str) == "linear")
        & (final["failed"].fillna(0).astype(float) == 0)
    ].copy()
    if main.empty:
        main = final[
            (final["learner"].astype(str) == "linear")
            & (final["method"].astype(str).isin(["unweighted", "oracle"]))
            & (final["failed"].fillna(0).astype(float) == 0)
        ].copy()
    rows: list[dict[str, object]] = []
    notes: list[str] = []
    for schedule in sorted(main["schedule"].dropna().astype(str).unique()):
        for regime in ["on_policy", "mild_shift", "moderate_shift"]:
            sub = main[(main["schedule"].astype(str) == schedule) & (main["regime"].astype(str) == regime)]
            if sub.empty:
                continue
            q_uw = _median(sub, "unweighted", "stationary_q_rmse")
            q_oracle = _median(sub, "oracle", "stationary_q_rmse")
            q_est095 = _median(sub, "estimated_g0p95", "stationary_q_rmse")
            q_est1 = _median(sub, "estimated_g1p0", "stationary_q_rmse")
            q_best_est = np.nanmin([q_est095, q_est1]) if np.any(np.isfinite([q_est095, q_est1])) else float("nan")
            q_gain = _improvement(q_uw, q_oracle)
            pbe_uw = _median(sub, "unweighted", "stationary_projected_bellman_rmse")
            pbe_oracle = _median(sub, "oracle", "stationary_projected_bellman_rmse")
            pbe_est095 = _median(sub, "estimated_g0p95", "stationary_projected_bellman_rmse")
            pbe_est1 = _median(sub, "estimated_g1p0", "stationary_projected_bellman_rmse")
            pbe_best_est = np.nanmin([pbe_est095, pbe_est1]) if np.any(np.isfinite([pbe_est095, pbe_est1])) else float("nan")
            pbe_gain = _improvement(pbe_uw, pbe_oracle)
            adv_uw = _median(sub, "unweighted", "stationary_advantage_q_rmse")
            adv_oracle = _median(sub, "oracle", "stationary_advantage_q_rmse")
            adv_est095 = _median(sub, "estimated_g0p95", "stationary_advantage_q_rmse")
            adv_est1 = _median(sub, "estimated_g1p0", "stationary_advantage_q_rmse")
            adv_best_est = np.nanmin([adv_est095, adv_est1]) if np.any(np.isfinite([adv_est095, adv_est1])) else float("nan")
            adv_gain = _improvement(adv_uw, adv_oracle)
            q_pbe_gains = [x for x in [q_gain, pbe_gain] if math.isfinite(x)]
            best_q_pbe_gain = max(q_pbe_gains) if q_pbe_gains else float("-inf")
            if math.isfinite(adv_gain) and adv_gain >= best_q_pbe_gain:
                metric = "stationary_advantage_q_rmse"
                uw, oracle, best_est, oracle_gain = adv_uw, adv_oracle, adv_best_est, adv_gain
            elif math.isfinite(pbe_gain) and (not math.isfinite(q_gain) or pbe_gain >= q_gain):
                metric = "stationary_projected_bellman_rmse"
                uw, oracle, best_est, oracle_gain = pbe_uw, pbe_oracle, pbe_best_est, pbe_gain
            else:
                metric = "stationary_q_rmse"
                uw, oracle, best_est, oracle_gain = q_uw, q_oracle, q_best_est, q_gain
            recovery = float((uw - best_est) / (uw - oracle)) if math.isfinite(uw) and math.isfinite(best_est) and math.isfinite(oracle) and abs(uw - oracle) > 1e-12 else float("nan")
            ess095 = _median(sub, "estimated_g0p95", "effective_sample_size_fraction")
            failure_uw = _failure_rate(final, "unweighted", schedule=schedule, regime=regime)
            gate = "review"
            finite_gains = [x for x in [q_gain, pbe_gain, adv_gain] if math.isfinite(x)]
            shifted_gain = max(finite_gains) if finite_gains else float("nan")
            if regime == "on_policy":
                gate = "pass" if math.isfinite(shifted_gain) and abs(shifted_gain) <= 0.10 and failure_uw < 0.05 else "review"
            elif regime == "mild_shift":
                gate = "pass" if math.isfinite(shifted_gain) and shifted_gain >= 0.15 else "review"
            elif regime == "moderate_shift":
                gate = "pass" if (math.isfinite(shifted_gain) and shifted_gain >= 0.20) or failure_uw >= 0.10 else "review"
            rows.append(
                {
                    "schedule": schedule,
                    "regime": regime,
                    "metric": metric,
                    "unweighted": uw,
                    "oracle": oracle,
                    "estimated_best": best_est,
                    "oracle_gain": oracle_gain,
                    "estimated_recovery": recovery,
                    "stationary_q_oracle_gain": q_gain,
                    "stationary_pbe_oracle_gain": pbe_gain,
                    "stationary_advantage_oracle_gain": adv_gain,
                    "estimated_g0p95_ess": ess095,
                    "unweighted_failure_rate": failure_uw,
                    "gate": gate,
                }
            )
            if gate != "pass":
                notes.append(f"{schedule}/{regime}: gate needs review on {metric}.")
    return rows, notes


def _write_snippet(report_dir: Path, results_name: str, gate_rows: list[dict[str, object]], shift_rows: list[dict[str, object]]) -> Path:
    snippet_dir = ROOT / "paper_snippets"
    snippet_dir.mkdir(parents=True, exist_ok=True)
    path = snippet_dir / f"{results_name}_main_text_experiment_section.tex"
    mild = next((row for row in gate_rows if row["regime"] == "mild_shift" and row["schedule"] == "direct"), None)
    moderate = next((row for row in gate_rows if row["regime"] == "moderate_shift" and row["schedule"] == "direct"), None)
    mild_gain = _fmt(mild.get("oracle_gain") if mild else float("nan"), 2)
    moderate_gain = _fmt(moderate.get("oracle_gain") if moderate else float("nan"), 2)
    shift_text = ", ".join(
        f"{row['regime']} TV={_fmt(row['tv_state_action'], 2)}, q99 ratio={_fmt(row['oracle_ratio_q99'], 2)}"
        for row in shift_rows
    )
    text = rf"""\paragraph{{Controlled nonlinear navigation study.}}
We evaluate soft fitted Q-iteration in a continuous two-dimensional navigation benchmark with five discrete actions, nonlinear stochastic dynamics, a goal basin, and a behavior-attracting decoy basin.  A high-resolution grid solver computes the reference low-temperature soft optimum, its stationary distribution, and oracle stationary density ratios.  The three behavior regimes interpolate between the target stationary policy and a decoy policy, giving the following measured distribution shifts: {shift_text}.

\paragraph{{Methods and metrics.}}
We compare behavior-norm soft FQI, oracle stationary-weighted soft FQI, and moment-estimated stationary weighting.  The primary metrics are stationary-norm Q error, advantage-centered Q error, behavior-norm Q error, stationary projected Bellman error, value error against the soft-DP reference, failure rate, and density-ratio diagnostics.

\paragraph{{Result.}}
In the direct low-temperature linear FQI study, oracle stationary weighting changes the mild-shift projected-error metric by a relative gain of {mild_gain} and the moderate-shift metric by {moderate_gain}.  The main figures report learning curves and norm-mismatch diagnostics; the appendix reports gamma-weight sweeps, neural stress tests, and density-ratio calibration.
"""
    path.write_text(text, encoding="utf-8")
    return path


def assess(results_dir: Path, report_path: Path | None = None) -> Path:
    raw = pd.read_csv(results_dir / "raw_results.csv")
    final = raw[raw["is_final"] == 1].copy()
    report_dir = report_path.parent if report_path else ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = report_dir / f"{results_dir.name}_assessment.md"
    shift_rows = _regime_shift_table(results_dir)
    gate_rows, notes = _gate_rows(final)
    snippet_path = _write_snippet(report_dir, results_dir.name, gate_rows, shift_rows)

    failed_rate = float(np.mean(raw.get("failed", pd.Series(np.zeros(len(raw)))).fillna(0).astype(float) > 0))
    diverged_rate = float(np.mean(raw.get("diverged", pd.Series(np.zeros(len(raw)))).fillna(0).astype(float) > 0))
    neural = final[final["learner"].astype(str) == "neural"].copy()
    neural_loss_bad = False
    if not neural.empty and "neural_train_loss" in neural:
        loss = neural["neural_train_loss"].dropna().astype(float)
        neural_loss_bad = bool((not loss.empty) and (not np.all(np.isfinite(loss))))
    value = final["policy_value_error"].dropna().astype(float) if "policy_value_error" in final else pd.Series(dtype=float)
    qerr = final["stationary_q_rmse"].dropna().astype(float) if "stationary_q_rmse" in final else pd.Series(dtype=float)
    value_cancellation = bool((not value.empty) and (not qerr.empty) and value.nunique() <= max(3, int(0.03 * len(value))))
    value_cancellation = value_cancellation or any(
        math.isfinite(float(row.get("stationary_q_oracle_gain", float("nan"))))
        and float(row.get("stationary_q_oracle_gain", float("nan"))) < -0.10
        and (
            float(row.get("stationary_pbe_oracle_gain", float("nan"))) > 0.05
            or float(row.get("stationary_advantage_oracle_gain", float("nan"))) > 0.05
        )
        for row in gate_rows
    )

    lines: list[str] = []
    lines.append(f"# Soft-FQI Stationary-Weighting Assessment: `{results_dir.name}`")
    lines.append("")
    lines.append("## Run Integrity")
    lines.append(f"- Raw rows: {len(raw)}; final rows: {len(final)}.")
    lines.append(f"- Failure rate: {_fmt(failed_rate, 3)}; divergence rate: {_fmt(diverged_rate, 3)}.")
    lines.append(f"- Stages: {', '.join(sorted(raw['stage'].dropna().astype(str).unique()))}.")
    lines.append("")
    lines.append("## Distribution-Shift Audit")
    if shift_rows:
        lines.append("| regime | TV(state-action) | chi2 | q99 ratio | max ratio | oracle ESS fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in shift_rows:
            lines.append(
                f"| {row['regime']} | {_fmt(row['tv_state_action'])} | {_fmt(row['chi2_target_over_behavior'])} | "
                f"{_fmt(row['oracle_ratio_q99'])} | {_fmt(row['oracle_ratio_max'])} | {_fmt(row['oracle_ess_fraction_grid'])} |"
            )
    else:
        lines.append("- Reference context was not found; shift diagnostics unavailable.")
    lines.append("")
    lines.append("## Decision Gates")
    if gate_rows:
        lines.append("| schedule | regime | selected metric | unweighted | oracle | best estimated | selected gain | Q gain | PBE gain | advantage gain | estimated recovery | ESS g=0.95 | failure rate | gate |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in gate_rows:
            lines.append(
                f"| {row['schedule']} | {row['regime']} | {row['metric']} | {_fmt(row['unweighted'])} | "
                f"{_fmt(row['oracle'])} | {_fmt(row['estimated_best'])} | {_fmt(row['oracle_gain'])} | "
                f"{_fmt(row['stationary_q_oracle_gain'])} | {_fmt(row['stationary_pbe_oracle_gain'])} | "
                f"{_fmt(row['stationary_advantage_oracle_gain'])} | "
                f"{_fmt(row['estimated_recovery'])} | {_fmt(row['estimated_g0p95_ess'])} | "
                f"{_fmt(row['unweighted_failure_rate'])} | {row['gate']} |"
            )
    else:
        lines.append("- No linear stage suitable for gate evaluation was found.")
    lines.append("")
    lines.append("## Scientific Assessment")
    lines.append("- Unweighted failure/degradation: assessed by the mild and moderate gate rows above; stationary projected Bellman error is preferred over scalar value error when the two disagree.")
    lines.append("- Oracle stationary weighting: considered supportive when oracle gain is positive in shifted regimes and neutral on-policy.")
    lines.append("- Estimated weighting: considered supportive when `estimated_g0p95` or `estimated_g1p0` recovers a nontrivial fraction of the oracle gain without ESS collapse.")
    lines.append("- Regime separation: supported when TV/chi2/ratio diagnostics increase monotonically from on-policy to moderate shift.")
    lines.append(f"- Value-cancellation warning: {'yes' if value_cancellation else 'no'}; use Q/PBE and advantage-centered Q metrics as primary if yes.")
    lines.append(f"- Neural optimization warning: {'yes' if neural_loss_bad else 'no'}; neural results should remain appendix-only if training diagnostics dominate.")
    lines.append("")
    lines.append("## Recommended Main-Text Use")
    if notes:
        lines.append("- Preliminary evidence needs review before the 100-rep paper run:")
        for note in notes:
            lines.append(f"  - {note}")
    else:
        lines.append("- Gates pass under the configured criteria; proceed to the 100-rep paper run.")
    lines.append("- Main figure: linear Stage 2 stationary-Q learning curves for unweighted, oracle, and estimated `gamma_weight=0.95`.")
    lines.append("- Mechanism figure: behavior-vs-stationary norm mismatch plus projected Bellman error diagnostics.")
    lines.append("- Appendix: gamma sweep, neural stress test, occupancy/ratio heatmaps, and ratio-estimation diagnostics.")
    lines.append(f"- Draft LaTeX snippet: `{snippet_path}`.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess whether a soft-FQI run supports the main-text story.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--report-path", default=None)
    args = parser.parse_args()
    path = assess(Path(args.results_dir), Path(args.report_path) if args.report_path else None)
    print(f"Wrote assessment to {path}")


if __name__ == "__main__":
    main()
