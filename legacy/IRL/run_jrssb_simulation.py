from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IRL.jrssb_simulation import (
    JRSSBConfig,
    JRSSBOracle,
    run_bias_acceptance,
    run_example1a_policy_audit,
    run_example1b_policy_audit,
    run_example2_method_selection,
    run_example2_nuisance_audit,
    run_example2_paper_comparison,
    run_example2_semi_oracle_audit,
    run_example2_shakedown,
    run_example2_smoke_check,
    run_monte_carlo,
    run_single_replication,
    run_validation_suite,
    save_json,
    save_results_csv,
    save_summary_csv,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JRSSB IRL debiasing simulation study.")
    parser.add_argument("--output-dir", type=Path, default=Path("IRL/outputs/jrssb_simulation"))
    parser.add_argument(
        "--mode",
        choices=[
            "checks",
            "bias-acceptance",
            "pilot",
            "monte-carlo",
            "example1a-policy-audit",
            "example1b-policy-audit",
            "example2-method-pilot",
            "example2-nuisance-audit",
            "example2-paper-comparison",
            "example2-semi-oracle",
            "example2-smoke",
            "example2-shakedown",
        ],
        default="checks",
    )
    parser.add_argument("--examples", nargs="*", default=["1a", "1b", "2"])
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--ratio-mode", choices=["oracle", "coarse-estimated"], default="oracle")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--pilot-n", type=int, default=10_000)
    parser.add_argument("--checks-large-n", type=int, default=5_000)
    parser.add_argument("--state-jitter", action="store_true", default=False)
    parser.add_argument("--mc-repetitions", type=int, default=None)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--nuisance-sample-mode", choices=["crossfit", "independent"], default="independent")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--fqe-iters", type=int, default=12)
    parser.add_argument("--fqe-epochs", type=int, default=8)
    parser.add_argument(
        "--example1a-nuisance-method",
        choices=["neural-fqe", "neural-main-oracle-bellman"],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example1a-policy-estimator",
        choices=["bc", "maxent", "coarse", "blend", "structural-linear"],
        default="maxent",
    )
    parser.add_argument(
        "--example1b-nuisance-method",
        choices=["neural-fqe", "neural-main-oracle-bellman"],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example1b-policy-estimator",
        choices=["bc", "maxent", "structural", "structural-linear"],
        default="structural-linear",
    )
    parser.add_argument("--example1b-structural-iters", type=int, default=120)
    parser.add_argument("--example1b-structural-bellman-iters", type=int, default=80)
    parser.add_argument("--example1b-structural-learning-rate", type=float, default=0.05)
    parser.add_argument("--example1b-structural-smoothness-penalty", type=float, default=0.5)
    parser.add_argument("--example1b-structural-reward-scale", type=float, default=4.0)
    parser.add_argument(
        "--example2-nuisance-method",
        choices=[
            "neural",
            "coarse-oracle-bellman",
            "neural-coarse-bellman",
            "neural-main-oracle-bellman",
            "oracle-q",
            "oracle-all",
        ],
        default="neural-main-oracle-bellman",
    )
    parser.add_argument(
        "--example2-policy-estimator",
        choices=["bc", "maxent", "structural-linear"],
        default="maxent",
    )
    parser.add_argument(
        "--example2-methods",
        nargs="*",
        default=["neural", "neural-main-oracle-bellman", "neural-coarse-bellman"],
    )
    parser.add_argument("--example2-semi-oracle-n", type=int, default=2500)
    parser.add_argument("--example2-repeated-splits", type=int, default=1)
    parser.add_argument("--example2-ci-critical-value", type=float, default=1.96)
    parser.add_argument("--shakedown-reps", type=int, default=20)
    parser.add_argument("--acceptance-audit-repetitions", type=int, default=10)
    parser.add_argument("--acceptance-large-n", type=int, default=50_000)
    parser.add_argument("--acceptance-large-repetitions", type=int, default=5)
    parser.add_argument("--acceptance-pilot-repetitions", type=int, default=20)
    parser.add_argument("--acceptance-full-repetitions", type=int, default=50)
    parser.add_argument("--acceptance-confirmation-repetitions", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-oracle-cache", action="store_true", default=False)
    return parser.parse_args()


def maybe_plot(summary_rows: Sequence[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    example_ids = ["1a", "1b", "2"]
    labels = {"1a": "Example 1a", "1b": "Example 1b", "2": "Example 2"}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    for axis, example_id in zip(axes, example_ids):
        rows = [row for row in summary_rows if row["example_id"] == example_id]
        if not rows:
            continue
        methods = sorted({row.get("nuisance_method", "default") for row in rows})
        multi_method = len(methods) > 1
        for method in methods:
            method_rows = sorted((row for row in rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            suffix = f" ({method})" if multi_method else ""
            axis.plot(
                n_vals,
                [abs(row["plugin_bias"]) for row in method_rows],
                marker="o",
                label=f"|Bias| plugin{suffix}",
            )
            axis.plot(
                n_vals,
                [abs(row["if_bias"]) for row in method_rows],
                marker="o",
                linestyle="--",
                label=f"|Bias| IF{suffix}",
            )
            axis.plot(
                n_vals,
                [row["plugin_rmse"] for row in method_rows],
                marker="s",
                linestyle=":",
                label=f"RMSE plugin{suffix}",
            )
            axis.plot(
                n_vals,
                [row["if_rmse"] for row in method_rows],
                marker="s",
                linestyle="-.",
                label=f"RMSE IF{suffix}",
            )
        axis.set_title(labels[example_id])
        axis.set_xlabel("n")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Error scale")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "bias_rmse_panels.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True, sharex=True)
    for axis, example_id in zip(axes, example_ids):
        rows = [row for row in summary_rows if row["example_id"] == example_id]
        if not rows:
            continue
        methods = sorted({row.get("nuisance_method", "default") for row in rows})
        for method in methods:
            method_rows = sorted((row for row in rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            axis.plot(n_vals, [row["coverage_95"] for row in method_rows], marker="o", label=method)
        axis.axhline(0.95, color="#C44E52", linestyle="--", linewidth=1.0)
        axis.set_title(labels[example_id])
        axis.set_xlabel("n")
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("95% coverage")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_panels.png", dpi=200)
    plt.close(fig)

    ex2_rows = [row for row in summary_rows if row["example_id"] == "2"]
    if ex2_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
        methods = sorted({row.get("nuisance_method", "default") for row in ex2_rows})
        for method in methods:
            method_rows = sorted((row for row in ex2_rows if row.get("nuisance_method", "default") == method), key=lambda row: row["n"])
            n_vals = [row["n"] for row in method_rows]
            axes[0].plot(n_vals, [row["avg_pi0_action0_q01"] for row in method_rows], marker="o", label=method)
            axes[1].plot(n_vals, [row["avg_pi_ratio_q99"] for row in method_rows], marker="o", label=method)
            axes[2].plot(n_vals, [row["avg_nu_ratio_q99"] for row in method_rows], marker="o", label=method)
        axes[0].set_title("q01 of $\\hat\\pi(0|S)$")
        axes[1].set_title("q99 of $\\pi/\\hat\\pi$")
        axes[2].set_title("q99 of $\\nu/\\hat\\pi$")
        for axis in axes:
            axis.set_xlabel("n")
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "example2_overlap_panel.png", dpi=200)
        plt.close(fig)


def build_config(args: argparse.Namespace) -> JRSSBConfig:
    config = JRSSBConfig(
        state_jitter=args.state_jitter,
        bc_epochs=args.bc_epochs,
        fqe_iters=args.fqe_iters,
        fqe_epochs_per_iter=args.fqe_epochs,
        crossfit_folds=args.crossfit_folds,
        nuisance_sample_mode=args.nuisance_sample_mode,
        example1a_nuisance_method=args.example1a_nuisance_method,
        example1a_policy_estimator=args.example1a_policy_estimator,
        example1b_nuisance_method=args.example1b_nuisance_method,
        example1b_policy_estimator=args.example1b_policy_estimator,
        example1b_structural_iters=args.example1b_structural_iters,
        example1b_structural_bellman_iters=args.example1b_structural_bellman_iters,
        example1b_structural_learning_rate=args.example1b_structural_learning_rate,
        example1b_structural_smoothness_penalty=args.example1b_structural_smoothness_penalty,
        example1b_structural_reward_scale=args.example1b_structural_reward_scale,
        example2_nuisance_method=args.example2_nuisance_method,
        example2_policy_estimator=args.example2_policy_estimator,
        example2_repeated_splits=args.example2_repeated_splits,
        example2_ci_critical_value=args.example2_ci_critical_value,
        use_oracle_cache=not args.no_oracle_cache,
    )
    if args.mc_repetitions is not None:
        config.mc_repetitions = args.mc_repetitions
    return config


def run_checks(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    diagnostics = run_validation_suite(
        oracle=oracle,
        pilot_seed=args.seed,
        pilot_n=args.pilot_n,
        oracle_sample_n=args.checks_large_n,
    )
    save_json({"mode": "checks", "diagnostics": diagnostics}, output_dir / "checks.json")


def run_bias_acceptance_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_bias_acceptance(
        base_config=config,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
        audit_repetitions=args.acceptance_audit_repetitions,
        large_n=args.acceptance_large_n,
        large_repetitions=args.acceptance_large_repetitions,
        pilot_repetitions=args.acceptance_pilot_repetitions,
        full_repetitions=args.acceptance_full_repetitions,
        confirmation_repetitions=args.acceptance_confirmation_repetitions,
    )
    save_summary_csv(study["candidate_diagnostics"]["example1a_audit"]["rows"], output_dir / "example1a_audit_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_audit"]["summary"], output_dir / "example1a_audit_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_audit"]["rows"], output_dir / "example1b_audit_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_audit"]["summary"], output_dir / "example1b_audit_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_large_sample"]["rows"], output_dir / "example1a_large_sample_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1a_large_sample"]["summary"], output_dir / "example1a_large_sample_summary.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_large_sample"]["rows"], output_dir / "example1b_large_sample_rows.csv")
    save_summary_csv(study["candidate_diagnostics"]["example1b_large_sample"]["summary"], output_dir / "example1b_large_sample_summary.csv")
    save_summary_csv(study["pilot_coverage"]["example1a"], output_dir / "pilot_coverage_example1a.csv")
    save_summary_csv(study["pilot_coverage"]["example1b"], output_dir / "pilot_coverage_example1b.csv")
    save_summary_csv(study["pilot_coverage"]["example2"], output_dir / "pilot_coverage_example2.csv")
    save_summary_csv(study["stage4_full_coverage"], output_dir / "full_coverage_stage4.csv")
    save_summary_csv(study["stage5_confirmation_coverage"], output_dir / "full_coverage_confirmation.csv")
    save_summary_csv(study["final_full_coverage"], output_dir / "full_coverage_final.csv")
    save_summary_csv(study["baseline_secondary_coverage"], output_dir / "baseline_secondary_coverage.csv")
    save_summary_csv(study["secondary_finite_sample_checks"], output_dir / "secondary_checks.csv")
    save_json(
        {
            "acceptance": study["acceptance"],
            "selections": study["selections"],
            "invariants": study["invariants"],
        },
        output_dir / "acceptance.json",
    )


def run_pilot(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    results = [
        run_single_replication(
            oracle=oracle,
            n=args.pilot_n,
            seed=args.seed + offset,
            example_id=example_id,
            ratio_mode=args.ratio_mode,
        )
        for offset, example_id in enumerate(args.examples)
    ]
    save_results_csv(results, output_dir / "pilot_results.csv")
    save_summary_csv(summarize_results(results), output_dir / "pilot_summary.csv")
    diagnostics = run_validation_suite(
        oracle=oracle,
        pilot_seed=args.seed,
        pilot_n=args.pilot_n,
        oracle_sample_n=args.checks_large_n,
    )
    save_json({"mode": "pilot", "diagnostics": diagnostics}, output_dir / "pilot_checks.json")


def run_mc(oracle: JRSSBOracle, output_dir: Path, args: argparse.Namespace) -> None:
    results = run_monte_carlo(
        oracle=oracle,
        sample_sizes=args.sample_sizes,
        repetitions=args.repetitions,
        example_ids=args.examples,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    summary_rows = summarize_results(results)
    save_results_csv(results, output_dir / "results.csv")
    save_summary_csv(summary_rows, output_dir / "summary.csv")
    maybe_plot(summary_rows, output_dir)


def run_example2_method_pilot(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    sample_sizes = args.sample_sizes if args.sample_sizes is not None else [2500, 5000, 10000]
    repetitions = 100 if args.repetitions is None else args.repetitions
    study = run_example2_method_selection(
        base_config=config,
        methods=args.example2_methods,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["selection"], output_dir / "selection.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_paper_comparison_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    sample_sizes = args.sample_sizes if args.sample_sizes is not None else [5000, 10000]
    repetitions = args.shakedown_reps if args.repetitions is None else args.repetitions
    study = run_example2_paper_comparison(
        base_config=config,
        sample_sizes=sample_sizes,
        repetitions=repetitions,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_summary_csv(study["comparison"], output_dir / "comparison_table.csv")
    save_json(study["selection"], output_dir / "selection.json")
    save_json({"paper_methods": study["paper_methods"]}, output_dir / "paper_methods.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_semi_oracle(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    rows = run_example2_semi_oracle_audit(
        oracle=oracle,
        n=args.example2_semi_oracle_n,
        seed=args.seed,
        ratio_mode=args.ratio_mode,
    )
    save_json({"rows": rows}, output_dir / "semi_oracle.json")


def run_example1b_policy_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example1b_policy_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["soft_policy_self_check"], output_dir / "soft_policy_self_check.json")


def run_example1a_policy_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example1a_policy_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")


def run_example2_nuisance_audit_mode(output_dir: Path, args: argparse.Namespace) -> None:
    config = build_config(args)
    study = run_example2_nuisance_audit(
        base_config=config,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps if args.repetitions is None else args.repetitions,
    )
    save_summary_csv(study["rows"], output_dir / "rows.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")


def run_example2_smoke(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    study = run_example2_smoke_check(
        oracle=oracle,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        ratio_mode=args.ratio_mode,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["acceptance"], output_dir / "acceptance.json")
    maybe_plot(study["summary"], output_dir)


def run_example2_shakedown_mode(output_dir: Path, args: argparse.Namespace) -> None:
    oracle = JRSSBOracle(build_config(args))
    study = run_example2_shakedown(
        oracle=oracle,
        sample_sizes=args.sample_sizes if args.sample_sizes is not None else (5000, 10000),
        repetitions=args.shakedown_reps,
        ratio_mode=args.ratio_mode,
        jobs=args.jobs,
    )
    save_results_csv(study["results"], output_dir / "results.csv")
    save_summary_csv(study["summary"], output_dir / "summary.csv")
    save_json(study["checks"], output_dir / "checks.json")
    maybe_plot(study["summary"], output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": vars(args)}, output_dir / "run_config.json")

    if args.mode == "example2-method-pilot":
        run_example2_method_pilot(output_dir, args)
        return
    if args.mode == "example1b-policy-audit":
        run_example1b_policy_audit_mode(output_dir, args)
        return
    if args.mode == "example1a-policy-audit":
        run_example1a_policy_audit_mode(output_dir, args)
        return
    if args.mode == "bias-acceptance":
        run_bias_acceptance_mode(output_dir, args)
        return
    if args.mode == "example2-nuisance-audit":
        run_example2_nuisance_audit_mode(output_dir, args)
        return
    if args.mode == "example2-paper-comparison":
        run_example2_paper_comparison_mode(output_dir, args)
        return
    if args.mode == "example2-semi-oracle":
        run_example2_semi_oracle(output_dir, args)
        return
    if args.mode == "example2-smoke":
        run_example2_smoke(output_dir, args)
        return
    if args.mode == "example2-shakedown":
        run_example2_shakedown_mode(output_dir, args)
        return

    oracle = JRSSBOracle(build_config(args))
    if args.mode == "checks":
        run_checks(oracle, output_dir, args)
        return
    if args.mode == "pilot":
        run_pilot(oracle, output_dir, args)
        return
    run_mc(oracle, output_dir, args)


if __name__ == "__main__":
    main()
