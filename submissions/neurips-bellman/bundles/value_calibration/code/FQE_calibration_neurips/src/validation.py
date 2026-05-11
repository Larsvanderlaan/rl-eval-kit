from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_KEY = ["baseline_learner", "learner_variant", "calibration_protocol", "calibrator", "calibration_target"]


@dataclass(frozen=True)
class GateConfig:
    bias_threshold: float = 2.0
    mse_threshold: float = 8.0
    degradation_factor: float = 4.0
    degradation_abs_tolerance: float = 0.25
    sample_trend_factor: float = 1.25
    sample_trend_abs_tolerance: float = 1e-3


def _finite_series(values: pd.Series) -> bool:
    return bool(np.all(np.isfinite(pd.to_numeric(values, errors="coerce"))))


def evaluate_well_specified_gate(
    rows: list[dict[str, Any]] | pd.DataFrame,
    output_dir: str | Path,
    config: GateConfig | None = None,
) -> tuple[bool, pd.DataFrame]:
    cfg = config or GateConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows).copy()
    if raw.empty:
        raise ValueError("Validation gate received no rows.")
    if "learner_variant" not in raw:
        raw["learner_variant"] = raw["baseline_learner"] if "baseline_learner" in raw else "unknown"

    raw["oracle_finite"] = pd.to_numeric(raw["oracle_value"], errors="coerce").apply(np.isfinite)
    raw["estimate_finite"] = pd.to_numeric(raw["value_estimate"], errors="coerce").apply(np.isfinite)
    raw["oracle_independent"] = raw.get("oracle_value_method", "independent_monte_carlo_rollout").astype(str).str.contains(
        "independent", case=False, na=False
    )
    raw["no_test_or_oracle_training"] = (
        raw.get("train_data_provenance", "").astype(str).str.contains("not_test_or_oracle", na=False)
    )

    grouped = raw.groupby(METHOD_KEY + ["sample_size"], dropna=False).agg(
        mean_value_estimate=("value_estimate", "mean"),
        mean_bias=("value_error", "mean"),
        mse=("squared_error", "mean"),
        failure_rate=("failure_flag", "mean"),
        oracle_all_finite=("oracle_finite", "all"),
        estimates_all_finite=("estimate_finite", "all"),
        oracle_independent=("oracle_independent", "all"),
        no_test_or_oracle_training=("no_test_or_oracle_training", "all"),
        max_abs_value=("value_estimate", lambda x: float(pd.Series(x).abs().max())),
    ).reset_index()

    max_n = grouped["sample_size"].max()
    min_n = grouped["sample_size"].min()
    final = grouped[grouped["sample_size"] == max_n].copy()
    first = grouped[grouped["sample_size"] == min_n][METHOD_KEY + ["mse"]].rename(columns={"mse": "small_sample_mse"})
    final = final.merge(first, on=METHOD_KEY, how="left")
    final["sample_size_trend_ok"] = final["mse"].le(
        cfg.sample_trend_factor * final["small_sample_mse"].fillna(final["mse"]) + cfg.sample_trend_abs_tolerance
    )

    baseline = final[
        (final["calibration_protocol"] == "uncalibrated_all_data")
        & (final["calibrator"] == "none")
    ][["baseline_learner", "calibration_target", "mse"]].rename(columns={"mse": "all_data_baseline_mse"})
    final = final.merge(baseline, on=["baseline_learner", "calibration_target"], how="left")
    final["calibration_degradation_ok"] = True
    calibrated = final["calibration_protocol"].isin(["cross", "split", "no_split"])
    accurate_baseline = final["all_data_baseline_mse"].le(cfg.mse_threshold)
    final.loc[calibrated & accurate_baseline, "calibration_degradation_ok"] = final.loc[
        calibrated & accurate_baseline, "mse"
    ].le(
        cfg.degradation_factor * final.loc[calibrated & accurate_baseline, "all_data_baseline_mse"]
        + cfg.degradation_abs_tolerance
    )

    final["pass_well_specified"] = (
        final["oracle_all_finite"]
        & final["estimates_all_finite"]
        & final["oracle_independent"]
        & final["no_test_or_oracle_training"]
        & final["mean_bias"].abs().le(cfg.bias_threshold)
        & final["mse"].le(cfg.mse_threshold)
        & final["failure_rate"].eq(0.0)
        & final["sample_size_trend_ok"]
        & final["calibration_degradation_ok"]
    )

    merged = grouped.merge(
        final[
            METHOD_KEY
            + [
                "small_sample_mse",
                "all_data_baseline_mse",
                "sample_size_trend_ok",
                "calibration_degradation_ok",
                "pass_well_specified",
            ]
        ],
        on=METHOD_KEY,
        how="left",
    )
    merged.to_csv(output_dir / "well_specified_gate.csv", index=False)

    failures = final[~final["pass_well_specified"]]
    gate_passed = bool(failures.empty)
    payload = {
        "gate_passed": gate_passed,
        "n_method_groups": int(final.shape[0]),
        "n_failed_method_groups": int(failures.shape[0]),
        "failed_methods": failures[METHOD_KEY].to_dict("records"),
        "config": cfg.__dict__,
    }
    with (output_dir / "well_specified_gate.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    with (output_dir / "audit_notes.md").open("w") as handle:
        handle.write("# Validation Gate Audit Notes\n\n")
        handle.write(f"- Gate passed: `{gate_passed}`\n")
        handle.write(f"- Method groups checked: `{final.shape[0]}`\n")
        handle.write(f"- Failed method groups: `{failures.shape[0]}`\n")
        handle.write("- Oracle method required to contain `independent` in `oracle_value_method`.\n")
        handle.write("- Training provenance required to contain `not_test_or_oracle`.\n")
        if failures.empty:
            handle.write("\nAll well-specified gate checks passed.\n")
        else:
            handle.write("\nFailed groups:\n\n")
            handle.write(failures.to_string(index=False))
            handle.write("\n")
    return gate_passed, merged


def load_gate(results_dir: str | Path) -> dict[str, Any] | None:
    path = Path(results_dir) / "validation" / "well_specified_gate.json"
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def failed_method_keys(gate_frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    if "pass_well_specified" not in gate_frame:
        return set()
    failed = gate_frame[~gate_frame["pass_well_specified"].fillna(False)]
    return {
        tuple(str(row[col]) for col in METHOD_KEY)
        for _, row in failed[METHOD_KEY].drop_duplicates().iterrows()
    }


def apply_gate_to_rows(rows: list[dict[str, Any]], gate_frame: pd.DataFrame) -> list[dict[str, Any]]:
    failed = failed_method_keys(gate_frame)
    out = []
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in METHOD_KEY)
        updated = dict(row)
        if key in failed:
            updated["main_evidence_eligible"] = False
            if not bool(updated.get("failure_flag", False)):
                updated["diagnostic_warning_message"] = (
                    str(updated.get("diagnostic_warning_message", "")).strip(";")
                    + ";failed_well_specified_gate"
                ).strip(";")
        else:
            updated["main_evidence_eligible"] = bool(updated.get("main_evidence_eligible", True)) and not bool(
                updated.get("failure_flag", False)
            )
        out.append(updated)
    return out
