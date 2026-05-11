#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.validation import load_gate  # noqa: E402


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _eligible_summary(results_dir: Path) -> pd.DataFrame:
    return _read_csv(results_dir / "eligible_summary.csv") if (results_dir / "eligible_summary.csv").exists() else _read_csv(results_dir / "summary.csv")


def _mechanism_checks(summary: pd.DataFrame) -> tuple[list[str], dict[str, bool]]:
    lines: list[str] = ["## Mechanism Checks", ""]
    status = {"affine": False, "monotone": False, "has_10_reps": False}
    mech = summary[
        (summary.get("suite_name", pd.Series(dtype=str)).astype(str) == "mechanism_distortion_sweep")
        & (summary.get("calibration_protocol", pd.Series(dtype=str)).astype(str) == "cross")
    ].copy()
    if mech.empty:
        lines.append("- `mechanism_distortion_sweep`: missing from summary.")
        return lines, status

    n_rep = int(pd.to_numeric(mech.get("n_replications", pd.Series([0])), errors="coerce").max())
    status["has_10_reps"] = n_rep >= 10
    lines.append(f"- Replications: `{n_rep}` (`{'pass' if status['has_10_reps'] else 'needs >=10'}`).")

    def _metric(frame: pd.DataFrame, col: str, fallback: str = "relative_mse_vs_uncalibrated_all_data") -> pd.Series:
        use_col = col if col in frame else fallback
        return pd.to_numeric(frame[use_col], errors="coerce")

    def _row_metric(row: pd.Series, col: str, fallback: str = "relative_mse_vs_uncalibrated_all_data") -> float:
        use_col = col if col in row.index else fallback
        return float(row[use_col])

    def _best_matched_mechanism(frame: pd.DataFrame, mode: str) -> tuple[pd.Series | None, pd.Series | None, bool]:
        """Return the best within-learner linear/isotonic comparison.

        Mechanism sweeps may include multiple distortion strengths under the
        same difficulty label. Comparing the first linear row to the globally
        best isotonic row can mix settings, so draft checks must be matched by
        learner variant.
        """
        best_pair: tuple[pd.Series | None, pd.Series | None, bool] = (None, None, False)
        best_score = float("inf")
        for _, group in frame.groupby("learner_variant", dropna=False):
            linear = group[group["calibrator"].astype(str) == "linear"].copy()
            nonlinear = group[group["calibrator"].astype(str).isin(["isotonic", "isotonic_histogram"])].copy()
            if linear.empty or nonlinear.empty:
                continue
            linear_row = linear.loc[_metric(linear, "relative_true_v_mse_vs_uncalibrated_all_data").idxmin()]
            nonlinear_row = nonlinear.loc[_metric(nonlinear, "relative_true_v_mse_vs_uncalibrated_all_data").idxmin()]
            linear_rel = _row_metric(linear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            nonlinear_rel = _row_metric(nonlinear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            linear_cal = float(linear_row.get("relative_calibration_error_plugin_vs_uncalibrated_all_data", float("inf")))
            nonlinear_cal = float(nonlinear_row.get("relative_calibration_error_plugin_vs_uncalibrated_all_data", float("inf")))
            linear_brier = float(linear_row.get("relative_brier_score_vs_uncalibrated_all_data", float("inf")))
            nonlinear_brier = float(nonlinear_row.get("relative_brier_score_vs_uncalibrated_all_data", float("inf")))
            if mode == "affine":
                passed = linear_rel < 1.0 and linear_rel <= nonlinear_rel + 0.05
                score = linear_rel
            else:
                better_bellman = (nonlinear_cal <= linear_cal - 0.01) or (nonlinear_brier <= linear_brier - 0.01)
                passed = nonlinear_rel < 1.0 and nonlinear_rel <= linear_rel + 0.02 and better_bellman
                score = nonlinear_rel if passed else nonlinear_rel + max(0.0, nonlinear_rel - linear_rel - 0.02)
            if passed and (not best_pair[2] or score < best_score):
                best_pair = (linear_row, nonlinear_row, True)
                best_score = score
            elif not best_pair[2] and score < best_score:
                best_pair = (linear_row, nonlinear_row, False)
                best_score = score
        return best_pair

    difficulty = mech.get("calibration_difficulty", pd.Series([""] * len(mech), index=mech.index)).astype(str)
    affine = mech[difficulty == "affine_miscalibrated"].copy()
    if affine.empty:
        lines.append("- Affine mechanism: missing.")
    else:
        linear_row, nonlinear_row, passed = _best_matched_mechanism(affine, "affine")
        if linear_row is None or nonlinear_row is None:
            lines.append("- Affine mechanism: missing linear or best-calibrator row.")
        else:
            linear_rel = _row_metric(linear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            nonlinear_rel = _row_metric(nonlinear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            nonlinear_name = str(nonlinear_row["calibrator"])
            variant = str(linear_row.get("learner_variant", ""))
            status["affine"] = passed
            verdict = "pass" if status["affine"] else "fail"
            lines.append(
                f"- Affine mechanism: matched variant `{variant}`, linear relative true-V MSE `{linear_rel:.3g}`, "
                f"best nonlinear `{nonlinear_name}` `{nonlinear_rel:.3g}`: `{verdict}`."
            )

    monotone = mech[difficulty == "monotone_miscalibrated"].copy()
    if monotone.empty:
        lines.append("- Monotone mechanism: missing.")
    else:
        linear_row, nonlinear_row, passed = _best_matched_mechanism(monotone, "monotone")
        if linear_row is None or nonlinear_row is None:
            lines.append("- Monotone mechanism: missing linear or isotonic/hybrid row.")
        else:
            linear_rel = _row_metric(linear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            nonlinear_rel = _row_metric(nonlinear_row, "relative_true_v_mse_vs_uncalibrated_all_data")
            nonlinear_name = str(nonlinear_row["calibrator"])
            variant = str(linear_row.get("learner_variant", ""))
            status["monotone"] = passed
            verdict = "pass" if status["monotone"] else "fail"
            lines.append(
                f"- Monotone mechanism: matched variant `{variant}`, best isotonic/hybrid `{nonlinear_name}` "
                f"relative true-V MSE `{nonlinear_rel:.3g}`, linear `{linear_rel:.3g}`: `{verdict}`."
            )
    return lines, status


def _split_checks(results_dir: Path) -> list[str]:
    lines = ["## Split-Stability Checks", ""]
    split = _read_csv(results_dir / "split_stability_diagnostics.csv")
    if split.empty:
        lines.append("- Split-stability diagnostics: missing.")
        return lines
    if "suite_name" in split:
        split = split[split["suite_name"].astype(str) != "well_specified_debug"].copy()
    stable = split[
        (pd.to_numeric(split.get("n_replications", 0), errors="coerce") >= 10)
        & (pd.to_numeric(split.get("mean_relative_mse_vs_all_data", 999), errors="coerce") < 1.0)
        & (pd.to_numeric(split.get("win_rate_vs_all_data", 0), errors="coerce") >= 0.60)
        & (pd.to_numeric(split.get("q90_relative_mse_vs_all_data", 999), errors="coerce") <= 1.25)
    ].copy()
    lines.append(f"- Stable split-calibration groups: `{len(stable)}`.")
    if stable.empty:
        lines.append("- Recommendation: do not make main-text claims from split-calibration wins yet.")
    else:
        stable = stable.sort_values("mean_relative_mse_vs_all_data").head(8)
        for _, row in stable.iterrows():
            lines.append(
                "- Stable candidate: "
                f"`{row.get('suite_name', '')}` / `{row.get('baseline_learner', '')}` / `{row.get('calibrator', '')}` "
                f"train fraction `{float(row.get('train_fraction', float('nan'))):.2g}`, "
                f"mean rel-MSE `{float(row.get('mean_relative_mse_vs_all_data', float('nan'))):.3g}`, "
                f"win rate `{float(row.get('win_rate_vs_all_data', float('nan'))):.2g}`."
            )
    return lines


def _failure_checks(summary: pd.DataFrame) -> list[str]:
    lines = ["## Failure Diagnostics", ""]
    if summary.empty or "failure_rate" not in summary:
        lines.append("- Failure diagnostics: unavailable.")
        return lines
    failures = summary[pd.to_numeric(summary["failure_rate"], errors="coerce").fillna(0.0) > 0.0].copy()
    lines.append(f"- Method groups with nonzero failure rate: `{len(failures)}`.")
    if not failures.empty:
        for _, row in failures.sort_values("failure_rate", ascending=False).head(8).iterrows():
            lines.append(
                f"- `{row.get('learner_variant', row.get('baseline_learner', ''))}` / "
                f"`{row.get('calibration_protocol', '')}` / `{row.get('calibrator', '')}`: "
                f"failure rate `{float(row.get('failure_rate', float('nan'))):.2g}`, "
                f"role `{row.get('main_figure_role', '')}`."
            )
    return lines


def inspect_paper_draft(results_dir: str | Path, figures_dir: str | Path | None = None) -> Path:
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else results_dir.parent.parent / "figures" / results_dir.name
    summary = _eligible_summary(results_dir)
    full_summary = _read_csv(results_dir / "summary.csv")
    gate = load_gate(results_dir)

    lines: list[str] = ["# Paper Draft Readout", ""]
    gate_passed = bool(gate and gate.get("gate_passed", False))
    lines.append(f"- Validation gate passed: `{gate_passed}`.")
    lines.append(f"- Results directory: `{results_dir}`.")
    lines.append(f"- Figures directory: `{figures_dir}`.")
    lines.append("")

    mechanism_lines, mechanism_status = _mechanism_checks(summary)
    lines.extend(mechanism_lines)
    lines.append("")
    lines.extend(_split_checks(results_dir))
    lines.append("")
    lines.extend(_failure_checks(full_summary if not full_summary.empty else summary))
    lines.append("")

    lines.extend(["## Candidate Main Figures", ""])
    if not gate_passed:
        lines.append("- None. The validation gate failed or is missing.")
    else:
        lines.append("- `mse_vs_sample_size` and `relative_mse_vs_sample_size` if quick-paper replications are present.")
        lines.append("- `coverage_stratified_error` as the failure/limited-overlap diagnostic.")
        if mechanism_status["has_10_reps"] and mechanism_status["affine"] and mechanism_status["monotone"]:
            lines.append("- `calibrator_comparison` / mechanism table: mechanism checks passed.")
        else:
            lines.append("- Mechanism calibrator figure: appendix/diagnostic until affine and monotone checks pass with >=10 reps.")
        lines.append("- `split_stability_diagnostics` only if stable split groups are listed above.")

    out = results_dir / "paper_draft_readout.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect quick paper-draft FQE calibration evidence.")
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results/paper"))
    parser.add_argument("--figures_dir", type=str, default=str(ROOT / "figures/paper"))
    args = parser.parse_args()
    out = inspect_paper_draft(args.results_dir, args.figures_dir)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
