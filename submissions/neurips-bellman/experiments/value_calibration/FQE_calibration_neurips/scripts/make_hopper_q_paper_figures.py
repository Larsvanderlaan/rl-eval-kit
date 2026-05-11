#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


METHOD_ORDER = ["linear", "isotonic", "histogram", "isotonic_histogram"]
METHOD_LABELS = {
    "linear": "Linear",
    "isotonic": "Isotonic",
    "histogram": "Histogram",
    "isotonic_histogram": "Iso-hist",
    "none": "Raw",
}
CRITIC_LABELS = {"neural_fqe": "Neural FQE"}
METRICS = {
    "bellman_calibration_error_plugin": "Plug-in cal. error",
    "bellman_calibration_error": "Debiased cal. error",
    "q_bellman_mse": "Q Bellman MSE",
    "absolute_ope_error": "OPE abs. error",
}
SUMMARY_REL_COLS = {
    "bellman_calibration_error_plugin": "relative_bellman_calibration_error_plugin",
    "bellman_calibration_error": "relative_bellman_calibration_error",
    "q_bellman_mse": "relative_q_bellman_mse",
    "absolute_ope_error": "relative_absolute_ope_error",
}
SUMMARY_CI_COLS = {
    "bellman_calibration_error_plugin": (
        "relative_bellman_calibration_error_plugin_ci_low",
        "relative_bellman_calibration_error_plugin_ci_high",
    ),
    "bellman_calibration_error": (
        "relative_bellman_calibration_error_ci_low",
        "relative_bellman_calibration_error_ci_high",
    ),
    "q_bellman_mse": ("relative_q_bellman_mse_ci_low", "relative_q_bellman_mse_ci_high"),
    "absolute_ope_error": ("relative_absolute_ope_error_ci_low", "relative_absolute_ope_error_ci_high"),
}
PRIMARY_METRICS = ["bellman_calibration_error_plugin", "bellman_calibration_error", "q_bellman_mse"]


@dataclass(frozen=True)
class RunInfo:
    label: str
    path: Path
    trajectories: int
    expected_units: int | None = None
    expected_rows: int | None = None
    expected_summary_rows: int = 10


def _candidate_runs(results_root: Path) -> list[RunInfo]:
    return [
        RunInfo(
            "Hopper 192 traj., 50 seeds",
            results_root / "hopper_q_calibration_deadline_final_nonlin_hopper192_s50",
            trajectories=192,
            expected_units=1100,
            expected_rows=5500,
        ),
        RunInfo(
            "Hopper 192 traj., 20 seeds",
            results_root / "hopper_q_calibration_deadline_final_nonlin_hopper192_s20",
            trajectories=192,
            expected_units=440,
            expected_rows=2200,
        ),
        RunInfo(
            "Hopper 512 traj., 10 seeds",
            results_root / "hopper_q_calibration_deadline_sample512_nonlin_s10",
            trajectories=512,
            expected_units=220,
            expected_rows=1100,
        ),
        RunInfo(
            "Hopper 128 traj., 5 seeds",
            results_root / "hopper_q_calibration_hour_nonlin_hopper128_s5",
            trajectories=128,
            expected_units=110,
            expected_rows=550,
        ),
    ]


def _read_json(path: Path) -> dict[str, object]:
    with path.open() as fh:
        return json.load(fh)


def _paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return (
        run_dir / "hopper_q_calibration_results.csv",
        run_dir / "hopper_q_calibration_summary.csv",
        run_dir / "hopper_q_calibration_manifest.json",
    )


def _is_valid_run(run: RunInfo, *, strict_expected: bool = True) -> bool:
    results_path, summary_path, manifest_path = _paths(run.path)
    if not (results_path.exists() and summary_path.exists() and manifest_path.exists()):
        return False
    try:
        manifest = _read_json(manifest_path)
        results = pd.read_csv(results_path)
        summary = pd.read_csv(summary_path)
    except Exception:
        return False
    if strict_expected:
        if run.expected_units is not None and int(manifest.get("n_completed_unit_files", -1)) != run.expected_units:
            return False
        if run.expected_units is not None and int(manifest.get("n_expected_units", -1)) != run.expected_units:
            return False
        if run.expected_rows is not None and int(manifest.get("n_result_rows", -1)) != run.expected_rows:
            return False
        if run.expected_summary_rows is not None and len(summary) != run.expected_summary_rows:
            return False
    if int(manifest.get("n_failed_units", 0)) != 0:
        return False
    if set(results.get("weighting", pd.Series(dtype=str)).astype(str)) != {"none"}:
        return False
    required_methods = set(METHOD_ORDER + ["none"])
    families = sorted(set(results["critic_family"].astype(str)))
    if "neural_fqe" not in families:
        return False
    for family in families:
        methods = set(summary.loc[summary["critic_family"].astype(str).eq(family), "method"].astype(str))
        if not required_methods.issubset(methods):
            return False
    required_result_cols = {"critic_family", "method", "seed", "policy_id", "weighting", *PRIMARY_METRICS}
    required_summary_cols = {
        "critic_family",
        "method",
        "weighting",
        "n_policies",
        "n_seeds",
        *SUMMARY_REL_COLS.values(),
    }
    if not required_result_cols.issubset(results.columns) or not required_summary_cols.issubset(summary.columns):
        return False
    for col in PRIMARY_METRICS:
        values = pd.to_numeric(results[col], errors="coerce")
        if not np.isfinite(values).all():
            return False
    for col in SUMMARY_REL_COLS.values():
        values = pd.to_numeric(summary[col], errors="coerce")
        if not np.isfinite(values).all():
            return False
    return True


def _load_run(run: RunInfo) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    results_path, summary_path, manifest_path = _paths(run.path)
    return pd.read_csv(results_path), pd.read_csv(summary_path), _read_json(manifest_path)


def _select_primary(results_root: Path, override: Path | None) -> RunInfo:
    if override is not None:
        run = RunInfo(f"User-selected {override.name}", override, trajectories=-1, expected_units=None, expected_rows=None)
        if not _is_valid_run(run, strict_expected=False):
            raise SystemExit(f"Selected run is not valid for paper figures: {override}")
        return run
    for run in _candidate_runs(results_root)[:2]:
        if _is_valid_run(run):
            return run
    raise SystemExit("No validated primary Hopper Q-calibration run found. Expected 50-seed or 20-seed run.")


def _validated_support_runs(results_root: Path, primary: RunInfo) -> list[RunInfo]:
    out: list[RunInfo] = []
    seen = {primary.path.resolve()}
    for run in _candidate_runs(results_root):
        if run.path.resolve() in seen:
            out.append(primary)
            continue
        if _is_valid_run(run):
            out.append(run)
    unique: list[RunInfo] = []
    used: set[Path] = set()
    for run in out:
        resolved = run.path.resolve()
        if resolved not in used:
            unique.append(run)
            used.add(resolved)
    return unique


def _unit_ratios(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["critic_family", "seed", "policy_id"]
    raw = results[results["method"].astype(str).eq("none")][keys + list(METRICS)].copy()
    raw = raw.rename(columns={col: f"raw_{col}" for col in METRICS})
    cal = results[~results["method"].astype(str).eq("none")].merge(raw, on=keys, how="left", validate="many_to_one")
    for col in METRICS:
        cal[f"relative_{col}"] = pd.to_numeric(cal[col], errors="coerce") / pd.to_numeric(
            cal[f"raw_{col}"], errors="coerce"
        )
    return cal


def _check_summary_matches_raw(results: pd.DataFrame, summary: pd.DataFrame, *, atol: float = 1e-8) -> None:
    cal = _unit_ratios(results)
    for (family, method), group in cal.groupby(["critic_family", "method"], dropna=False):
        row = summary[
            summary["critic_family"].astype(str).eq(str(family)) & summary["method"].astype(str).eq(str(method))
        ]
        if row.empty:
            raise SystemExit(f"Summary missing {family}/{method}")
        for metric, summary_col in SUMMARY_REL_COLS.items():
            raw_col = f"raw_{metric}"
            rel = pd.to_numeric(group[metric], errors="coerce").mean() / pd.to_numeric(group[raw_col], errors="coerce").mean()
            reported = float(pd.to_numeric(row[summary_col], errors="coerce").iloc[0])
            if math.isfinite(rel) and abs(rel - reported) > atol:
                raise SystemExit(
                    f"Summary mismatch for {family}/{method}/{metric}: raw={rel:.12g}, summary={reported:.12g}"
                )


def _ci(summary: pd.DataFrame, family: str, method: str, metric: str) -> tuple[float, float, float]:
    row = summary[
        summary["critic_family"].astype(str).eq(family) & summary["method"].astype(str).eq(method)
    ].iloc[0]
    value = float(row[SUMMARY_REL_COLS[metric]])
    lo_col, hi_col = SUMMARY_CI_COLS[metric]
    lo = float(row[lo_col]) if lo_col in row.index and pd.notna(row[lo_col]) else value
    hi = float(row[hi_col]) if hi_col in row.index and pd.notna(row[hi_col]) else value
    return value, lo, hi


def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_main(summary: pd.DataFrame, unit: pd.DataFrame, manifest: dict[str, object], figures_dir: Path) -> None:
    neural_summary = summary[summary["critic_family"].astype(str).eq("neural_fqe")].copy()
    neural_unit = unit[unit["critic_family"].astype(str).eq("neural_fqe")].copy()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 2.85), gridspec_kw={"width_ratios": [1.16, 1.0]})

    y = np.arange(len(METHOD_ORDER))
    offsets = [-0.13, 0.13]
    colors = {"bellman_calibration_error_plugin": "#2F6B9A", "q_bellman_mse": "#BF5B17"}
    labels = {"bellman_calibration_error_plugin": "Bellman calibration", "q_bellman_mse": "Q Bellman MSE"}
    for offset, metric in zip(offsets, ["bellman_calibration_error_plugin", "q_bellman_mse"]):
        values, lows, highs = [], [], []
        for method in METHOD_ORDER:
            value, lo, hi = _ci(neural_summary, "neural_fqe", method, metric)
            values.append(value)
            lows.append(value - lo)
            highs.append(hi - value)
        ax0.errorbar(
            values,
            y + offset,
            xerr=np.vstack([lows, highs]),
            fmt="o",
            color=colors[metric],
            ecolor=colors[metric],
            elinewidth=1.2,
            capsize=2.5,
            markersize=4.2,
            label=labels[metric],
        )
    ax0.axvline(1.0, color="0.25", linewidth=0.9, linestyle=(0, (3, 2)))
    ax0.set_yticks(y)
    ax0.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax0.invert_yaxis()
    max_panel_a = 1.0
    for method in METHOD_ORDER:
        for metric in ["bellman_calibration_error_plugin", "q_bellman_mse"]:
            value, _, _ = _ci(neural_summary, "neural_fqe", method, metric)
            max_panel_a = max(max_panel_a, value)
    ax0.set_xlim(0.55, max(1.15, min(1.8, 1.10 * max_panel_a)))
    ax0.set_xlabel("Ratio to raw neural FQE")
    ax0.tick_params(axis="both", labelsize=7)
    ax0.set_title("A. Conditional Q diagnostics", loc="left", fontweight="bold")
    ax0.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax0.legend(frameon=False, fontsize=7, loc="lower right")

    plot_rows = []
    for method in ["linear", "isotonic_histogram"]:
        for metric in ["bellman_calibration_error_plugin", "q_bellman_mse"]:
            vals = pd.to_numeric(
                neural_unit.loc[neural_unit["method"].astype(str).eq(method), f"relative_{metric}"],
                errors="coerce",
            ).dropna()
            for value in vals:
                plot_rows.append(
                    {
                        "label": f"{METHOD_LABELS[method]}\n{labels[metric].replace('Bellman ', '')}",
                        "method": method,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    dist = pd.DataFrame(plot_rows)
    order = [
        "Linear\ncalibration",
        "Linear\nQ MSE",
        "Iso-hist\ncalibration",
        "Iso-hist\nQ MSE",
    ]
    data = [dist.loc[dist["label"].eq(label), "value"].to_numpy() for label in order]
    bp = ax1.boxplot(data, positions=np.arange(len(order)), widths=0.48, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#D8E6F3" if i in (0, 2) else "#F1D5BF")
        patch.set_edgecolor("0.25")
    for i, values in enumerate(data):
        if values.size:
            jitter = np.linspace(-0.17, 0.17, min(values.size, 50))
            shown = np.sort(values)[:50] if values.size > 50 else values
            if values.size > 50:
                jitter = np.linspace(-0.17, 0.17, shown.size)
            ax1.scatter(i + jitter, shown, s=7, color="0.25", alpha=0.28, linewidths=0)
    ax1.axhline(1.0, color="0.25", linewidth=0.9, linestyle=(0, (3, 2)))
    ax1.set_xticks(np.arange(len(order)))
    ax1.set_xticklabels(order)
    max_panel_b = max((float(np.nanmax(values)) for values in data if values.size), default=1.0)
    ax1.set_ylim(0.0, max(1.20, min(1.9, 1.10 * max_panel_b)))
    ax1.set_ylabel("Unit-level ratio to raw")
    ax1.tick_params(axis="both", labelsize=7)
    ax1.set_title("B. Paired seed-policy units", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    _save(fig, figures_dir, "hopper_q_main_neural_fqe")


def _plot_audit(summary: pd.DataFrame, figures_dir: Path) -> None:
    rows = []
    family = "neural_fqe"
    for method in METHOD_ORDER:
        row = summary[
            summary["critic_family"].astype(str).eq(family) & summary["method"].astype(str).eq(method)
        ].iloc[0]
        rows.append((family, method, row))
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.25), sharey=True)
    row_labels = [f"{CRITIC_LABELS[f]}\n{METHOD_LABELS[m]}" for f, m, _ in rows]
    for ax, metric in zip(axes, METRICS):
        rel_col = SUMMARY_REL_COLS[metric]
        vals = np.array([float(row[rel_col]) for _, _, row in rows], dtype=float)
        display = np.log2(np.clip(vals, 0.25, 4.0))[:, None]
        im = ax.imshow(display, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i, value in enumerate(vals):
            shown = ">4" if value > 4 else f"{value:.2f}"
            ax.text(0, i, shown, ha="center", va="center", fontsize=6.8, color="black")
        ax.set_xticks([])
        ax.set_title(METRICS[metric], fontsize=8, fontweight="bold")
        ax.axhline(3.5, color="white", linewidth=1.5)
    axes[0].set_yticks(np.arange(len(row_labels)))
    axes[0].set_yticklabels(row_labels, fontsize=7)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
    cbar.set_label("log2 ratio to raw", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(top=0.86, left=0.19, right=0.93, wspace=0.22)
    _save(fig, figures_dir, "hopper_q_full_audit")


def _plot_robustness(runs: list[RunInfo], figures_dir: Path) -> None:
    rows = []
    for run in runs:
        _, summary, manifest = _load_run(run)
        if not _is_valid_run(run):
            continue
        n_seeds = int(summary["n_seeds"].max())
        for method in ["linear", "isotonic_histogram"]:
            row = summary[
                summary["critic_family"].astype(str).eq("neural_fqe") & summary["method"].astype(str).eq(method)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            for metric in ["bellman_calibration_error_plugin", "q_bellman_mse"]:
                lo_col, hi_col = SUMMARY_CI_COLS[metric]
                rows.append(
                    {
                        "run": run.label.replace("Hopper ", ""),
                        "trajectories": run.trajectories,
                        "n_seeds": n_seeds,
                        "method": method,
                        "metric": metric,
                        "value": float(row[SUMMARY_REL_COLS[metric]]),
                        "lo": float(row[lo_col]) if lo_col in row.index else float(row[SUMMARY_REL_COLS[metric]]),
                        "hi": float(row[hi_col]) if hi_col in row.index else float(row[SUMMARY_REL_COLS[metric]]),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty or df["run"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    metric_titles = {
        "bellman_calibration_error_plugin": "Plug-in calibration error",
        "q_bellman_mse": "Q Bellman MSE",
    }
    markers = {"linear": "o", "isotonic_histogram": "s"}
    colors = {"linear": "#2F6B9A", "isotonic_histogram": "#BF5B17"}
    run_order = (
        df[["run", "trajectories", "n_seeds"]]
        .drop_duplicates()
        .sort_values(["trajectories", "n_seeds"])
        .assign(label=lambda x: x["run"])
    )
    x_labels = run_order["label"].tolist()
    x_pos = np.arange(len(x_labels))
    for ax, metric in zip(axes, ["bellman_calibration_error_plugin", "q_bellman_mse"]):
        sub = df[df["metric"].eq(metric)]
        for method in ["linear", "isotonic_histogram"]:
            ms = sub[sub["method"].eq(method)].merge(run_order[["run"]], on="run", how="right")
            values = pd.to_numeric(ms["value"], errors="coerce")
            lows = values - pd.to_numeric(ms["lo"], errors="coerce")
            highs = pd.to_numeric(ms["hi"], errors="coerce") - values
            ax.errorbar(
                x_pos,
                values,
                yerr=np.vstack([lows.fillna(0), highs.fillna(0)]),
                marker=markers[method],
                color=colors[method],
                linewidth=1.2,
                capsize=2,
                label=METHOD_LABELS[method],
            )
        ax.axhline(1.0, color="0.25", linewidth=0.9, linestyle=(0, (3, 2)))
        ax.set_title(metric_titles[metric], fontsize=8.5, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=6.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Ratio to raw neural FQE")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Hopper neural-FQE robustness across completed runs", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, figures_dir, "hopper_q_seed_sample_robustness")


def _write_numbers(
    primary: RunInfo,
    summary: pd.DataFrame,
    manifest: dict[str, object],
    figures_dir: Path,
) -> None:
    rows = []
    for family, method in [
        ("neural_fqe", "linear"),
        ("neural_fqe", "isotonic_histogram"),
    ]:
        row = summary[
            summary["critic_family"].astype(str).eq(family) & summary["method"].astype(str).eq(method)
        ].iloc[0]
        rows.append(
            {
                "run_label": primary.label,
                "critic_family": family,
                "method": method,
                "n_seeds": int(row["n_seeds"]),
                "n_policies": int(row["n_policies"]),
                "relative_bellman_calibration_error_plugin": float(row["relative_bellman_calibration_error_plugin"]),
                "relative_bellman_calibration_error": float(row["relative_bellman_calibration_error"]),
                "relative_q_bellman_mse": float(row["relative_q_bellman_mse"]),
                "relative_absolute_ope_error": float(row["relative_absolute_ope_error"]),
                "bellman_calibration_win_rate": float(row["bellman_calibration_win_rate"])
                if pd.notna(row["bellman_calibration_win_rate"])
                else np.nan,
                "q_bellman_mse_win_rate": float(row["q_bellman_mse_win_rate"])
                if pd.notna(row["q_bellman_mse_win_rate"])
                else np.nan,
            }
        )
    numbers = pd.DataFrame(rows)
    figures_dir.mkdir(parents=True, exist_ok=True)
    numbers.to_csv(figures_dir / "hopper_q_main_numbers.csv", index=False)
    n_units = int(
        summary.loc[summary["critic_family"].astype(str).eq("neural_fqe"), ["n_seeds", "n_policies"]]
        .drop_duplicates()
        .prod(axis=1)
        .iloc[0]
    )
    neural_linear = numbers[(numbers["critic_family"].eq("neural_fqe")) & (numbers["method"].eq("linear"))].iloc[0]
    neural_iso = numbers[(numbers["critic_family"].eq("neural_fqe")) & (numbers["method"].eq("isotonic_histogram"))].iloc[0]
    tex = rf"""\newcommand{{\HopperQRunLabel}}{{{primary.label}}}
\newcommand{{\HopperQUnits}}{{{n_units}}}
\newcommand{{\HopperQSeeds}}{{{int(neural_iso['n_seeds'])}}}
\newcommand{{\HopperQPolicies}}{{{int(neural_iso['n_policies'])}}}
\newcommand{{\HopperQNeuralLinearCal}}{{{neural_linear['relative_bellman_calibration_error_plugin']:.3f}}}
\newcommand{{\HopperQNeuralLinearMSE}}{{{neural_linear['relative_q_bellman_mse']:.3f}}}
\newcommand{{\HopperQNeuralIsoHistCal}}{{{neural_iso['relative_bellman_calibration_error_plugin']:.3f}}}
\newcommand{{\HopperQNeuralIsoHistMSE}}{{{neural_iso['relative_q_bellman_mse']:.3f}}}
\newcommand{{\HopperQNeuralIsoHistOPE}}{{{neural_iso['relative_absolute_ope_error']:.3f}}}
"""
    (figures_dir / "hopper_q_main_numbers.tex").write_text(tex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make reviewer-facing Hopper conditional Q-calibration figures.")
    parser.add_argument("--results_root", default=str(ROOT / "results"))
    parser.add_argument("--figures_dir", default=str(ROOT / "paper_import_bundle" / "figures"))
    parser.add_argument("--primary_results_dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    figures_dir = Path(args.figures_dir)
    override = Path(args.primary_results_dir) if args.primary_results_dir else None
    primary = _select_primary(results_root, override)
    results, summary, manifest = _load_run(primary)
    _check_summary_matches_raw(results, summary)
    unit = _unit_ratios(results)
    _plot_main(summary, unit, manifest, figures_dir)
    _plot_audit(summary, figures_dir)
    _plot_robustness(_validated_support_runs(results_root, primary), figures_dir)
    _write_numbers(primary, summary, manifest, figures_dir)
    print(f"Primary Hopper Q run: {primary.label} ({primary.path})")
    print(f"Wrote Hopper Q paper figures and numbers to {figures_dir}")


if __name__ == "__main__":
    main()
