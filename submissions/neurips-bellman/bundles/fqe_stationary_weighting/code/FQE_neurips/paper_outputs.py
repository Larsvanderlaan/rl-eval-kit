from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hard_benchmark_experiment import evaluate_hard_benchmark_setting
from .hard_benchmark_longrun_plot import HardLongRunPlotConfig, generate_hard_longrun_plots
from .latent_garnet_benchmark import (
    LatentGarnetConfig,
    build_latent_garnet_benchmark,
    evaluate_fqe_methods_on_benchmark,
)


@dataclass
class PaperOutputsConfig:
    output_dir: str = "FQE_neurips/outputs/paper_figures"
    realistic_cache_dir: str | None = None
    hard_behavior_solid_prob: float = 0.40
    hard_dataset_size: int = 2500
    hard_gamma_eval: float = 0.95
    hard_linear_ridge: float = 1e-2
    hard_linear_outer_iters: int = 100
    hard_longrun_seeds: int = 50
    hard_secondary_seeds: int = 5
    realistic_coverages: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3)
    realistic_n_states: int = 60
    realistic_n_actions: int = 4
    realistic_latent_dim: int = 3
    realistic_branching_factor: int = 5
    realistic_dataset_size: int = 2500
    realistic_data_mode: str = "mixed"
    realistic_n_trajectories: int = 200
    realistic_iid_fraction: float = 0.5
    realistic_observation_mode: str = "rich"
    realistic_gamma_eval: float = 0.95
    realistic_gamma_ratio: float = 1.0
    realistic_pilot_seeds: int = 3
    realistic_final_seeds: int = 5
    realistic_quick: bool = False


METRIC_FIELDS = [
    "target_policy_relative_rmse",
    "stationary_q_rmse",
    "behavior_q_rmse",
    "stationary_v_rmse",
    "behavior_v_rmse",
    "initial_policy_value_abs_error",
    "initial_policy_value_relative_abs_error",
]


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text)


def _realistic_cache_root(config: PaperOutputsConfig) -> Path:
    if config.realistic_cache_dir is not None:
        path = Path(config.realistic_cache_dir)
    else:
        path = Path(config.output_dir) / "realistic_cache"
    _ensure_dir(path)
    return path


def _coverage_tag(coverage: float) -> str:
    return f"{coverage:.3f}".replace(".", "p")


def _realistic_cache_path(config: PaperOutputsConfig, coverage: float, seed: int) -> Path:
    root = _realistic_cache_root(config)
    name = (
        f"cov_{_coverage_tag(coverage)}"
        f"_seed_{seed}"
        f"_n_{config.realistic_dataset_size}"
        f"_mode_{config.realistic_data_mode}"
        f"_traj_{config.realistic_n_trajectories}"
        f"_iid_{str(config.realistic_iid_fraction).replace('.', 'p')}"
        f"_obs_{config.realistic_observation_mode}"
        f"_ge_{str(config.realistic_gamma_eval).replace('.', 'p')}"
        f"_gr_{str(config.realistic_gamma_ratio).replace('.', 'p')}"
        f"_quick_{int(config.realistic_quick)}.json"
    )
    return root / name


def _latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    cols = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _format_pm(values: list[float], decimals: int = 3) -> str:
    return f"{_median(values):.{decimals}f} $\\pm$ {_std(values):.{decimals}f}"


def _method_label(method: str) -> str:
    mapping = {
        "unweighted": "Unweighted",
        "weighted_policy_ratio": "Policy ratio",
        "oracle": "Oracle stationary",
        "weighted_linear_basic": "Linear stationary",
        "weighted_linear_flexible": "Flexible linear stationary",
        "weighted_neural": "Neural stationary",
        "weighted_neural_rkhs": "RKHS stationary",
    }
    return mapping.get(method, method)


def _family_label(family_key: str) -> str:
    mapping = {
        "linear_fqe_metrics": "Linear FQE",
        "neural_fqe_metrics": "Small neural FQE",
        "neural_fqe_flexible_metrics": "Flexible neural FQE",
        "linear_metrics": "Linear FQE",
        "neural_metrics": "Small neural FQE",
        "neural_flexible_metrics": "Flexible neural FQE",
    }
    return mapping[family_key]


def _realistic_seed_results(config: PaperOutputsConfig, coverage: float, seeds: range) -> list[dict[str, object]]:
    results = []
    for seed in seeds:
        cache_path = _realistic_cache_path(config, coverage=coverage, seed=seed)
        if cache_path.exists():
            results.append(json.loads(cache_path.read_text()))
            continue

        benchmark = build_latent_garnet_benchmark(
            LatentGarnetConfig(
                n_states=config.realistic_n_states,
                n_actions=config.realistic_n_actions,
                latent_dim=config.realistic_latent_dim,
                branching_factor=config.realistic_branching_factor,
                dataset_size=config.realistic_dataset_size,
                data_mode=config.realistic_data_mode,
                n_trajectories=config.realistic_n_trajectories,
                iid_fraction=config.realistic_iid_fraction,
                behavior_coverage=coverage,
                observation_mode=config.realistic_observation_mode,
                seed=seed,
            )
        )
        result = evaluate_fqe_methods_on_benchmark(
            benchmark,
            gamma_eval=config.realistic_gamma_eval,
            gamma_ratio=config.realistic_gamma_ratio,
            seed=seed,
            quick=config.realistic_quick,
        )
        _save_json(cache_path, result)
        results.append(result)
    return results


def _collect_realistic_grouped(
    config: PaperOutputsConfig,
    *,
    seed_offset: int,
    seed_count: int,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    seeds = range(seed_offset, seed_offset + seed_count)
    for coverage in config.realistic_coverages:
        grouped[str(coverage)] = _realistic_seed_results(config, coverage=coverage, seeds=seeds)
    return grouped


def _select_stationary_methods(pilot_grouped: dict[str, list[dict[str, object]]]) -> dict[str, str]:
    linear_candidates = ["weighted_linear_basic", "weighted_linear_flexible"]
    neural_candidates = ["weighted_neural", "weighted_neural_rkhs"]

    linear_scores = {}
    for method in linear_candidates:
        vals = [
            result["linear_fqe_metrics"][method]["target_policy_relative_rmse"]
            for per_cov in pilot_grouped.values()
            for result in per_cov
        ]
        linear_scores[method] = float(np.mean(vals))

    neural_scores = {}
    for method in neural_candidates:
        vals = [
            result[family_key][method]["target_policy_relative_rmse"]
            for per_cov in pilot_grouped.values()
            for result in per_cov
            for family_key in ("neural_fqe_metrics", "neural_fqe_flexible_metrics")
        ]
        neural_scores[method] = float(np.mean(vals))

    best_linear = min(linear_scores, key=linear_scores.get)
    best_neural = min(neural_scores, key=neural_scores.get)
    return {
        "best_linear_stationary": best_linear,
        "best_rkhs_or_neural_stationary": best_neural,
    }


def _compact_methods(selected_methods: dict[str, str]) -> list[tuple[str, str]]:
    return [
        ("unweighted", "Unweighted"),
        ("weighted_policy_ratio", "Policy ratio"),
        ("oracle", "Oracle stationary"),
        (selected_methods["best_linear_stationary"], f"{_method_label(selected_methods['best_linear_stationary'])} (selected)"),
        (
            selected_methods["best_rkhs_or_neural_stationary"],
            f"{_method_label(selected_methods['best_rkhs_or_neural_stationary'])} (selected)",
        ),
    ]


def _aggregate_realistic_rows(
    grouped: dict[str, list[dict[str, object]]],
    methods: list[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    family_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for coverage, results in grouped.items():
        coverage_rows.append(
            {
                "coverage": float(coverage),
                "overlap_mass": _median([r["overlap_metrics"]["realized_overlap_mass"] for r in results]),
                "min_density_ratio": _median(
                    [r["overlap_metrics"]["realized_min_density_ratio_nub_over_target"] for r in results]
                ),
                "oracle_ess_fraction": _median(
                    [r["weight_stability"]["oracle"]["effective_sample_size_fraction"] for r in results]
                ),
                "rkhs_ess_fraction": _median(
                    [r["weight_stability"]["neural_rkhs"]["effective_sample_size_fraction"] for r in results]
                ),
            }
        )
        for family_key in ("linear_fqe_metrics", "neural_fqe_metrics", "neural_fqe_flexible_metrics"):
            for method_key, method_label in methods:
                row = {
                    "family": _family_label(family_key),
                    "coverage": float(coverage),
                    "method": method_label,
                }
                for metric in METRIC_FIELDS:
                    values = [r[family_key][method_key][metric] for r in results]
                    row[metric] = _median(values)
                    row[f"{metric}_std"] = _std(values)
                family_rows.append(row)
    return family_rows, coverage_rows


def _aggregate_hard_secondary_rows(
    config: PaperOutputsConfig,
    methods: list[tuple[str, str]],
) -> list[dict[str, object]]:
    per_seed = [
        evaluate_hard_benchmark_setting(
            behavior_solid_prob=config.hard_behavior_solid_prob,
            dataset_size=config.hard_dataset_size,
            gamma_eval=config.hard_gamma_eval,
            n_outer_iters=config.hard_linear_outer_iters,
            seed=seed,
            include_neural=True,
            include_rkhs=True,
        )
        for seed in range(config.hard_secondary_seeds)
    ]

    rows: list[dict[str, object]] = []
    for family_key in ("linear_metrics", "neural_metrics", "neural_flexible_metrics"):
        for method_key, method_label in methods:
            row = {
                "family": _family_label(family_key),
                "method": method_label,
            }
            for metric in METRIC_FIELDS:
                values = [r[family_key][method_key][metric] for r in per_seed]
                row[metric] = _median(values)
                row[f"{metric}_std"] = _std(values)
            rows.append(row)
    return rows


def _plot_realistic_family_metric(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    families = ["Linear FQE", "Small neural FQE", "Flexible neural FQE"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        fam_rows = [row for row in rows if row["family"] == family]
        methods = sorted({row["method"] for row in fam_rows})
        for method in methods:
            method_rows = sorted(
                [row for row in fam_rows if row["method"] == method],
                key=lambda row: row["coverage"],
                reverse=True,
            )
            xs = [float(row["coverage"]) for row in method_rows]
            ys = [float(row[metric]) for row in method_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=method)
        ax.set_title(family)
        ax.set_xlabel("Behavior coverage")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].invert_xaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _realistic_table_tex(rows: list[dict[str, object]]) -> str:
    headers = ["Family", "Coverage", "Method", "Stat. Q", "Behav. Q", "Stat. V", "Behav. V", "Init. |V|"]
    body = [
        [
            row["family"],
            f"{row['coverage']:.2f}",
            row["method"],
            f"{row['stationary_q_rmse']:.3f}",
            f"{row['behavior_q_rmse']:.3f}",
            f"{row['stationary_v_rmse']:.3f}",
            f"{row['behavior_v_rmse']:.3f}",
            f"{row['initial_policy_value_abs_error']:.3f}",
        ]
        for row in rows
    ]
    return _latex_table(
        headers,
        body,
        caption="Realistic benchmark results with the frozen compact method set.",
        label="tab:realistic-main",
    )


def _hard_secondary_table_tex(rows: list[dict[str, object]]) -> str:
    headers = ["Family", "Method", "Stat. Q", "Behav. Q", "Stat. V", "Behav. V", "Init. |V|"]
    body = [
        [
            row["family"],
            row["method"],
            f"{row['stationary_q_rmse']:.3f}",
            f"{row['behavior_q_rmse']:.3f}",
            f"{row['stationary_v_rmse']:.3f}",
            f"{row['behavior_v_rmse']:.3f}",
            f"{row['initial_policy_value_abs_error']:.3f}",
        ]
        for row in rows
    ]
    return _latex_table(
        headers,
        body,
        caption="Hard benchmark secondary comparison across linear and neural FQE families.",
        label="tab:hard-secondary",
    )


def _hard_linear_summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for method, stats in summary["iqr_summary"].items():
        rows.append(
            {
                "method": _method_label("weighted_linear_basic" if method == "weighted_linear_basic" else method),
                "iter10": stats["iter10"],
                "iter25": stats["iter25"],
                "iter50": stats["iter50"],
                "final": stats["final"],
                "best": stats["best"],
                "best_iter": stats["best_iter"],
            }
        )
    return rows


def generate_paper_outputs(config: PaperOutputsConfig | None = None) -> dict[str, object]:
    if config is None:
        config = PaperOutputsConfig()

    output_dir = Path(config.output_dir)
    _ensure_dir(output_dir)

    hard_longrun = generate_hard_longrun_plots(
        HardLongRunPlotConfig(
            behavior_solid_prob=config.hard_behavior_solid_prob,
            dataset_size=config.hard_dataset_size,
            gamma_eval=config.hard_gamma_eval,
            ridge=config.hard_linear_ridge,
            n_outer_iters=config.hard_linear_outer_iters,
            seeds=config.hard_longrun_seeds,
            output_dir=config.output_dir,
        )
    )

    realistic_pilot = _collect_realistic_grouped(
        config,
        seed_offset=0,
        seed_count=config.realistic_pilot_seeds,
    )
    selected_methods = _select_stationary_methods(realistic_pilot)
    compact_methods = _compact_methods(selected_methods)

    realistic_final = _collect_realistic_grouped(
        config,
        seed_offset=config.realistic_pilot_seeds,
        seed_count=config.realistic_final_seeds,
    )
    realistic_rows, realistic_coverage_rows = _aggregate_realistic_rows(realistic_final, compact_methods)
    hard_secondary_rows = _aggregate_hard_secondary_rows(config, compact_methods)
    hard_linear_rows = _hard_linear_summary_rows(hard_longrun)

    _write_csv(output_dir / "realistic_main_table.csv", realistic_rows, fieldnames=list(realistic_rows[0].keys()))
    _write_csv(
        output_dir / "realistic_coverage_summary.csv",
        realistic_coverage_rows,
        fieldnames=list(realistic_coverage_rows[0].keys()),
    )
    _write_csv(output_dir / "hard_secondary_table.csv", hard_secondary_rows, fieldnames=list(hard_secondary_rows[0].keys()))
    _write_csv(output_dir / "hard_linear_summary.csv", hard_linear_rows, fieldnames=list(hard_linear_rows[0].keys()))

    _write_text(output_dir / "realistic_main_table.tex", _realistic_table_tex(realistic_rows))
    _write_text(output_dir / "hard_secondary_table.tex", _hard_secondary_table_tex(hard_secondary_rows))

    _plot_realistic_family_metric(
        realistic_rows,
        metric="stationary_q_rmse",
        ylabel="Stationary Q RMSE",
        path=output_dir / "realistic_stationary_q.png",
    )
    _plot_realistic_family_metric(
        realistic_rows,
        metric="initial_policy_value_abs_error",
        ylabel="Initial |V error|",
        path=output_dir / "realistic_initial_value.png",
    )

    summary = {
        "config": asdict(config),
        "selected_methods": selected_methods,
        "compact_methods": [{"key": key, "label": label} for key, label in compact_methods],
        "hard_longrun": hard_longrun,
        "hard_linear_rows": hard_linear_rows,
        "hard_secondary_rows": hard_secondary_rows,
        "realistic_coverage_rows": realistic_coverage_rows,
        "realistic_rows": realistic_rows,
    }
    _save_json(output_dir / "paper_summary.json", summary)
    return {
        "output_dir": str(output_dir.resolve()),
        "summary_path": str((output_dir / "paper_summary.json").resolve()),
        "selected_methods": selected_methods,
        "files": sorted(p.name for p in output_dir.iterdir()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final paper tables and plots for FQE_neurips.")
    parser.add_argument("--output-dir", type=str, default="FQE_neurips/outputs/paper_figures")
    parser.add_argument("--hard-longrun-seeds", type=int, default=50)
    parser.add_argument("--hard-secondary-seeds", type=int, default=5)
    parser.add_argument("--realistic-pilot-seeds", type=int, default=3)
    parser.add_argument("--realistic-final-seeds", type=int, default=5)
    parser.add_argument("--realistic-quick", action="store_true")
    parser.add_argument("--realistic-coverages", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3])
    parser.add_argument("--realistic-dataset-size", type=int, default=2500)
    args = parser.parse_args()

    output = generate_paper_outputs(
        PaperOutputsConfig(
            output_dir=args.output_dir,
            hard_longrun_seeds=args.hard_longrun_seeds,
            hard_secondary_seeds=args.hard_secondary_seeds,
            realistic_pilot_seeds=args.realistic_pilot_seeds,
            realistic_final_seeds=args.realistic_final_seeds,
            realistic_quick=args.realistic_quick,
            realistic_coverages=tuple(args.realistic_coverages),
            realistic_dataset_size=args.realistic_dataset_size,
        )
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
