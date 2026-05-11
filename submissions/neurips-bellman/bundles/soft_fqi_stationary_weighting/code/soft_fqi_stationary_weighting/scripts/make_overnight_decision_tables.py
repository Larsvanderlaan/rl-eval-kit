#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


METRICS = [
    ("policy_value_error", "policy value error"),
    ("stationary_advantage_q_rmse", "stationary advantage"),
    ("stationary_projected_bellman_rmse", "stationary PBE"),
    ("stationary_q_rmse", "stationary Q"),
    ("effective_sample_size_fraction", "ESS fraction"),
    ("minimax_residual_norm", "minimax residual"),
]


def _read_final(results_dir: Path) -> pd.DataFrame:
    raw_path = results_dir / "raw_results.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(raw_path, low_memory=False)
    if "is_final" not in raw:
        return pd.DataFrame()
    return raw[(raw["is_final"] == 1) & (raw.get("failed", 0) == 0)].copy()


def _median_iqr(values: pd.Series, *, scale: float = 1.0) -> str:
    arr = values.dropna().to_numpy(dtype=float) * float(scale)
    if arr.size == 0:
        return ""
    return f"{np.median(arr):.3g} [{np.quantile(arr, 0.25):.3g}, {np.quantile(arr, 0.75):.3g}]"


def _write_table(table: pd.DataFrame, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    table.to_csv(csv_path, index=False)
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, escape=False))
    return csv_path, tex_path


def _main_comparison(main_dir: Path, minimax_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = _read_final(main_dir)
    minimax = _read_final(minimax_dir)
    if main.empty and minimax.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not main.empty:
        main = main[main["method"].astype(str).isin(["unweighted", "oracle", "estimated_g0p95"])].copy()
    combined = pd.concat([main, minimax], ignore_index=True)
    rows = []
    group_cols = ["regime", "schedule", "method"]
    for keys, sub in combined.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys, strict=True))
        for metric, label in METRICS:
            if metric in sub.columns:
                scale = 1000.0 if metric == "stationary_advantage_q_rmse" else 1.0
                row[label] = _median_iqr(sub[metric], scale=scale)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(group_cols) if rows else pd.DataFrame()

    win_rows = []
    if not combined.empty and "seed" in combined:
        key_cols = ["regime", "schedule", "seed"]
        for metric, label in METRICS[:3]:
            if metric not in combined.columns:
                continue
            pivot = combined.pivot_table(index=key_cols, columns="method", values=metric, aggfunc="first")
            if "minimax" not in pivot.columns:
                continue
            for baseline in ["unweighted", "oracle", "estimated_g0p95"]:
                if baseline not in pivot.columns:
                    continue
                paired = pivot[["minimax", baseline]].dropna()
                if paired.empty:
                    continue
                win_rows.append(
                    {
                        "metric": label,
                        "baseline": baseline,
                        "n_pairs": int(len(paired)),
                        "minimax_win_rate": float(np.mean(paired["minimax"] < paired[baseline])),
                        "median_minimax_minus_baseline": float(np.median(paired["minimax"] - paired[baseline])),
                    }
                )
    wins = pd.DataFrame(win_rows)
    return summary, wins


def _simple_group_table(results_dir: Path, group_cols: list[str], name: str) -> pd.DataFrame:
    final = _read_final(results_dir)
    if final.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in final.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        for metric, label in METRICS:
            if metric in sub.columns:
                scale = 1000.0 if metric == "stationary_advantage_q_rmse" else 1.0
                row[label] = _median_iqr(sub[metric], scale=scale)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create morning decision tables for the overnight soft-FQI sweeps.")
    parser.add_argument("--main-results-dir", default=str(ROOT / "results" / "paper_main_500_stabilized_10regime"))
    parser.add_argument("--minimax-results-dir", default=str(ROOT / "results" / "main_linear_minimax_10regime"))
    parser.add_argument("--oracle-results-dir", default=str(ROOT / "results" / "oracle_stabilization_sensitivity"))
    parser.add_argument("--minimax-tuning-results-dir", default=str(ROOT / "results" / "minimax_tuning_fairness"))
    parser.add_argument("--population-linear-results-dir", default=str(ROOT / "results" / "population_geometry_linear"))
    parser.add_argument("--population-rich-results-dir", default=str(ROOT / "results" / "population_geometry_rich_q"))
    parser.add_argument("--output-dir", default=str(ROOT / "tables" / "overnight_decision"))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    main_summary, main_wins = _main_comparison(Path(args.main_results_dir), Path(args.minimax_results_dir))
    if not main_summary.empty:
        _write_table(main_summary, out_dir, "main_minimax_comparison")
    if not main_wins.empty:
        _write_table(main_wins, out_dir, "main_minimax_win_rates")

    oracle = _simple_group_table(Path(args.oracle_results_dir), ["stage", "regime", "method"], "oracle")
    if not oracle.empty:
        _write_table(oracle, out_dir, "oracle_stabilization_sensitivity")

    minimax_tuning = _simple_group_table(Path(args.minimax_tuning_results_dir), ["stage", "regime", "method"], "minimax")
    if not minimax_tuning.empty:
        _write_table(minimax_tuning, out_dir, "minimax_tuning_fairness")

    pop_linear = _simple_group_table(Path(args.population_linear_results_dir), ["regime", "method"], "population_linear")
    pop_rich = _simple_group_table(Path(args.population_rich_results_dir), ["regime", "method"], "population_rich")
    population = pd.concat(
        [
            pop_linear.assign(q_class="linear") if not pop_linear.empty else pop_linear,
            pop_rich.assign(q_class="rich_rbf") if not pop_rich.empty else pop_rich,
        ],
        ignore_index=True,
    )
    if not population.empty:
        cols = ["q_class"] + [col for col in population.columns if col != "q_class"]
        _write_table(population[cols], out_dir, "population_geometry")

    print(f"Wrote decision tables to {out_dir}")


if __name__ == "__main__":
    main()
