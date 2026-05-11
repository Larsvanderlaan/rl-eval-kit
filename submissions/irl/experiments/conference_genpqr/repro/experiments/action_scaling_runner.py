"""Many-action comparison for DeepPQR vs GenPQR.

This script holds the overall sample size fixed and increases the number of
actions, so anchor support shrinks endogenously as |A| grows. It starts from
the strong 5-action regime used in the main matched experiment and asks how the
methods separate as coverage per action falls.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "irl_neurips_paper_repro_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_paper_experiments import (
    StudyConfig,
    _attach_counts,
    _build_data,
    _evaluate_deeppqr,
    _evaluate_genpqr,
    _fit_airl,
    _parallel_map,
    _summarise,
    _write_markdown_table,
)
from policy_estimation import fit_behavior_cloning_policy


@dataclass(frozen=True)
class ActionScalingSpec:
    """Configuration for the many-action study."""

    seeds: Sequence[int]
    action_counts: Sequence[int]
    n_train_trajectories: int = 2500
    n_test_trajectories: int = 300
    horizon: int = 10
    state_dim: int = 5
    anchor_logit_shift: float = 0.0


def _flatten(rows: Iterable[Iterable[Dict[str, float]]]) -> pd.DataFrame:
    return pd.DataFrame([row for chunk in rows for row in chunk])


def _evaluate_one(seed: int, n_actions: int, spec: ActionScalingSpec) -> List[Dict[str, float]]:
    study = StudyConfig(
        seeds=spec.seeds,
        n_test_trajectories=spec.n_test_trajectories,
        horizon=spec.horizon,
        n_actions=n_actions,
        state_dim=spec.state_dim,
    )
    train, test, mu, g, config = _build_data(
        seed=seed,
        n_train_trajectories=spec.n_train_trajectories,
        anchor_logit_shift=spec.anchor_logit_shift,
        config=study,
    )

    airl_policy = _fit_airl(train, config)
    bc_policy = fit_behavior_cloning_policy(
        train["states"],
        train["actions"],
        n_actions=config.n_actions,
        n_epochs=40,
    )

    rows = [
        _evaluate_genpqr(train, test, mu, g, config, airl_policy, "neural_fqe", "GenPQR (AIRL, neural FQE)"),
        _evaluate_deeppqr(train, test, config, airl_policy),
        _evaluate_genpqr(train, test, mu, g, config, bc_policy, "neural_fqe", "GenPQR (BC, neural FQE)"),
        _evaluate_deeppqr(train, test, config, bc_policy),
    ]
    rows[1]["method"] = "DeepPQR (AIRL policy)"
    rows[3]["method"] = "DeepPQR (BC policy)"

    out = []
    for row in rows:
        row["seed"] = float(seed)
        row["n_actions"] = float(n_actions)
        row["setting"] = f"{n_actions}_actions"
        row["n_train_trajectories"] = float(spec.n_train_trajectories)
        row["anchor_logit_shift"] = float(spec.anchor_logit_shift)
        out.append(_attach_counts(row, train))
    return out


def run_action_scaling_experiment(spec: ActionScalingSpec, n_jobs: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the many-action comparison."""
    items = [(seed, n_actions) for n_actions in spec.action_counts for seed in spec.seeds]
    worker = partial(_evaluate_one_wrapped, spec=spec)
    results = _flatten(_parallel_map(worker, items, n_jobs=n_jobs))
    summary = _summarise(results, ["n_actions", "method"])
    return results, summary


def _evaluate_one_wrapped(item, spec: ActionScalingSpec):
    seed, n_actions = item
    return _evaluate_one(seed=seed, n_actions=n_actions, spec=spec)


def _plot_many_action(summary: pd.DataFrame, output_dir: Path) -> None:
    method_order = [
        "GenPQR (BC, neural FQE)",
        "GenPQR (AIRL, neural FQE)",
        "DeepPQR (BC policy)",
        "DeepPQR (AIRL policy)",
    ]
    colors = {
        "GenPQR (BC, neural FQE)": "#E69F00",
        "GenPQR (AIRL, neural FQE)": "#D55E00",
        "DeepPQR (BC policy)": "#56B4E9",
        "DeepPQR (AIRL policy)": "#0072B2",
    }

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for method in method_order:
        subset = summary[summary["method"] == method].sort_values("n_actions")
        if subset.empty:
            continue
        x = subset["n_actions"].to_numpy()
        mse = subset["reward_mse_mean"].to_numpy()
        mse_ci = subset["reward_mse_ci95"].to_numpy()
        ax.errorbar(
            x,
            mse,
            yerr=mse_ci,
            marker="o",
            capsize=3,
            linewidth=2,
            markersize=5,
            color=colors[method],
            label=method,
        )
    ax.set_xlabel("Number of actions", fontsize=12)
    ax.set_ylabel("Reward MSE", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "many_action_reward_mse.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for method in method_order:
        subset = summary[summary["method"] == method].sort_values("n_actions")
        if subset.empty:
            continue
        x = subset["n_actions"].to_numpy()
        anchor = subset["anchor_count_mean"].to_numpy()
        ax.plot(
            x,
            anchor,
            marker="o",
            linewidth=2,
            markersize=5,
            color=colors[method],
            label=method,
        )
    ax.set_xlabel("Number of actions", fontsize=12)
    ax.set_ylabel("Anchor-action count", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "many_action_anchor_count.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--train-trajectories", type=int, default=2500)
    parser.add_argument("--test-trajectories", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=5)
    parser.add_argument("--action-counts", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--anchor-logit-shift", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="IRL_neurips/experiments_neurips/outputs/many_action_regime",
    )
    args = parser.parse_args()

    spec = ActionScalingSpec(
        seeds=list(range(args.replicates)),
        action_counts=args.action_counts,
        n_train_trajectories=args.train_trajectories,
        n_test_trajectories=args.test_trajectories,
        horizon=args.horizon,
        state_dim=args.state_dim,
        anchor_logit_shift=args.anchor_logit_shift,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, summary = run_action_scaling_experiment(spec, n_jobs=args.jobs)
    results.to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    table = summary[
        [
            "n_actions",
            "method",
            "anchor_count_mean",
            "anchor_fraction_mean",
            "reward_mse_mean",
            "reward_mse_ci95",
            "reward_corr_mean",
            "reward_corr_ci95",
            "policy_nll_mean",
            "policy_nll_ci95",
            "elapsed_mean",
            "elapsed_ci95",
        ]
    ].copy()
    _write_markdown_table(table, output_dir / "table.md")
    table.to_csv(output_dir / "table.csv", index=False)
    _plot_many_action(summary, output_dir)
    print(f"Many-action comparison saved to {output_dir}")


if __name__ == "__main__":
    main()
